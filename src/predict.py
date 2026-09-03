import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 256
LAYER1_SIZE = 1024
LAYER2_SIZE = 2048
TARGET_COUNT = 206


class MoaModel(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        layer1_size,
        layer2_size,
        batch_norm=False,
    ):
        super().__init__()

        self.layer1 = nn.Linear(num_features, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, num_targets)

        self.batch_norm1 = (
            nn.BatchNorm1d(layer1_size)
            if batch_norm
            else nn.Identity()
        )

        self.batch_norm2 = (
            nn.BatchNorm1d(layer2_size)
            if batch_norm
            else nn.Identity()
        )

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.layer3(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def predict_model(model, features, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            end = start + BATCH_SIZE

            batch = torch.tensor(
                features[start:end],
                dtype=torch.float32,
                device=device,
            )

            logits = model(batch)
            probabilities = torch.sigmoid(logits)

            predictions.append(probabilities.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python src/predict.py "
            "<models_path> <processed_data_path> <output_path>"
        )
        sys.exit(1)

    models_path = Path(sys.argv[1])
    processed_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    output_path.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Load processed test data
    # ---------------------------------------------------------
    print("\nLoading processed test data...")

    processed_test = pd.read_csv(processed_path / "test.csv")

    print(f"Processed test shape: {processed_test.shape}")

    # cp_type is column 1 and is removed during model training.
    processed_test = processed_test.drop(
        columns=processed_test.columns[1]
    )

    # After removing cp_type:
    # column 0 = sig_id
    # remaining columns = model features
    features = processed_test.iloc[:, 1:].values.astype(
        np.float32
    )

    print(f"Model input shape: {features.shape}")

    # ---------------------------------------------------------
    # Load all 5 fold models
    # ---------------------------------------------------------
    all_predictions = []

    for fold in range(5):

        model_file = models_path / f"MoaModel_fold{fold}.pth"

        print(f"\nLoading fold {fold}: {model_file}")

        checkpoint = torch.load(
            model_file,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint["model_state_dict"]

        checkpoint_num_features = state_dict["layer1.weight"].shape[1]
        num_targets = state_dict["layer3.weight"].shape[0]

        if features.shape[1] != checkpoint_num_features:
            raise ValueError(
                f"Feature mismatch for fold {fold}: "
                f"processed data has {features.shape[1]} features, "
                f"but model expects {checkpoint_num_features}."
            )

        if num_targets != TARGET_COUNT:
            raise ValueError(
                f"Target mismatch for fold {fold}: "
                f"model predicts {num_targets}, "
                f"expected {TARGET_COUNT}."
            )

        has_batch_norm = any(
            key.startswith("batch_norm1.")
            for key in state_dict.keys()
        )

        print(f"BatchNorm: {has_batch_norm}")

        model = MoaModel(
            num_features=checkpoint_num_features,
            num_targets=num_targets,
            layer1_size=LAYER1_SIZE,
            layer2_size=LAYER2_SIZE,
            batch_norm=has_batch_norm,
        )

        model.load_state_dict(state_dict)
        model.to(device)

        print(
            f"Fold {fold} model: "
            f"{checkpoint_num_features} → "
            f"{LAYER1_SIZE} → "
            f"{LAYER2_SIZE} → "
            f"{num_targets}"
        )

        fold_predictions = predict_model(
            model,
            features,
            device,
        )

        print(
            f"Fold {fold} prediction shape: "
            f"{fold_predictions.shape}"
        )

        all_predictions.append(fold_predictions)

        del model

        if device.type == "mps":
            torch.mps.empty_cache()

    # ---------------------------------------------------------
    # Average 5-fold predictions
    # ---------------------------------------------------------
    print("\nAveraging 5-fold predictions...")

    treatment_predictions = np.mean(
        np.stack(all_predictions, axis=0),
        axis=0,
    )

    print(
        f"Treatment prediction shape: "
        f"{treatment_predictions.shape}"
    )

    # ---------------------------------------------------------
    # Load ORIGINAL raw test data
    # ---------------------------------------------------------
    raw_test_path = Path("data/raw/test_features.csv")

    print(f"\nLoading original test data: {raw_test_path}")

    raw_test = pd.read_csv(raw_test_path)

    print(f"Original test shape: {raw_test.shape}")

    # ---------------------------------------------------------
    # Load sample submission
    # ---------------------------------------------------------
    sample_path = Path("data/raw/sample_submission.csv")

    print(f"\nLoading sample submission: {sample_path}")

    sample_submission = pd.read_csv(sample_path)

    target_columns = sample_submission.columns[1:]

    print(f"Sample submission rows: {len(sample_submission)}")
    print(f"Sample submission targets: {len(target_columns)}")

    if len(target_columns) != TARGET_COUNT:
        raise ValueError(
            f"Expected {TARGET_COUNT} target columns, "
            f"but sample submission has {len(target_columns)}."
        )

    if len(raw_test) != len(sample_submission):
        raise ValueError(
            f"Raw test rows ({len(raw_test)}) do not match "
            f"sample submission rows ({len(sample_submission)})."
        )

    # ---------------------------------------------------------
    # Identify treatment/control rows
    # ---------------------------------------------------------
    treatment_mask = raw_test["cp_type"] != "ctl_vehicle"
    control_mask = ~treatment_mask

    treatment_count = int(treatment_mask.sum())
    control_count = int(control_mask.sum())

    print(f"Treatment rows: {treatment_count}")
    print(f"Control rows:   {control_count}")

    if treatment_count != len(treatment_predictions):
        raise ValueError(
            f"Treatment row mismatch: raw test has "
            f"{treatment_count} treatment rows, but processed "
            f"prediction has {len(treatment_predictions)} rows."
        )

    # ---------------------------------------------------------
    # Reconstruct predictions in ORIGINAL test order
    # ---------------------------------------------------------
    predictions = np.zeros(
        (len(raw_test), TARGET_COUNT),
        dtype=np.float32,
    )

    predictions[treatment_mask.values] = treatment_predictions

    print(
        f"\nFinal prediction matrix: "
        f"{predictions.shape}"
    )

    # ---------------------------------------------------------
    # Build submission using sample_submission order
    # ---------------------------------------------------------
    submission = sample_submission.copy()

    # Ensure sig_id ordering matches the original test data.
    if not np.array_equal(
        submission["sig_id"].values,
        raw_test["sig_id"].values,
    ):
        print(
            "WARNING: sample_submission sig_id order differs "
            "from raw test. Aligning predictions by sig_id."
        )

        prediction_df = pd.DataFrame(
            predictions,
            columns=target_columns,
        )

        prediction_df.insert(
            0,
            "sig_id",
            raw_test["sig_id"].values,
        )

        submission = submission[["sig_id"]].merge(
            prediction_df,
            on="sig_id",
            how="left",
            validate="one_to_one",
        )
    else:
        submission.loc[:, target_columns] = predictions

    submission_path = output_path / "submission.csv"

    submission.to_csv(
        submission_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------
    print(f"\nSubmission saved: {submission_path}")
    print(f"Submission shape: {submission.shape}")

    print("\nVerification:")

    assert len(submission) == len(sample_submission)
    assert list(submission.columns) == list(sample_submission.columns)
    assert np.array_equal(
        submission["sig_id"].values,
        sample_submission["sig_id"].values,
    )

    # Control rows must have zero predictions.
    control_ids = set(raw_test.loc[control_mask, "sig_id"])

    if control_ids:
        control_submission = submission[
            submission["sig_id"].isin(control_ids)
        ]

        control_values = control_submission[
            target_columns
        ].values

        print(
            f"Maximum control prediction: "
            f"{control_values.max():.10f}"
        )

        if not np.allclose(control_values, 0.0):
            raise ValueError(
                "Control rows contain non-zero predictions."
            )

    print(
        f"Prediction minimum: "
        f"{predictions.min():.6f}"
    )
    print(
        f"Prediction maximum: "
        f"{predictions.max():.6f}"
    )
    print(
        f"Prediction mean: "
        f"{predictions.mean():.6f}"
    )

    print("\nFirst 5 rows:")
    print(submission.head())

    print("\nSUCCESS: Final 5-fold FE submission created.")


if __name__ == "__main__":
    main()
