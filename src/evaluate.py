import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


LAYER1_SIZE = 1024
LAYER2_SIZE = 2048
BATCH_SIZE = 256


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

        return self.layer3(x)


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super().__init__()

        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(out_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        out = F.relu(self.lin1(x))
        out = self.dropout(out)
        out = self.lin2(out)

        return out + residual


class ResNet(nn.Module):
    def __init__(self, num_features, num_targets, layer_sizes):
        super().__init__()

        layers = [
            nn.Linear(num_features, layer_sizes[0])
        ]

        for i in range(len(layer_sizes) - 1):
            layers.append(
                ResBlock(
                    layer_sizes[i],
                    layer_sizes[i + 1]
                )
            )

        layers.append(
            nn.Linear(layer_sizes[-1], num_targets)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_predictions(model, features, device):
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
            predictions.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    return np.concatenate(predictions, axis=0)


def mean_column_log_loss(y_true, y_pred):
    """
    Mean column-wise binary log loss.

    Uses a numerically safe probability clipping step so that
    log(0) never occurs.
    """

    eps = np.finfo(np.float64).eps

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    losses = -(
        y_true * np.log(y_pred)
        + (1.0 - y_true) * np.log(1.0 - y_pred)
    )

    return float(np.mean(losses))


def main():

    if len(sys.argv) != 4:
        print(
            "Usage: python src/evaluate.py "
            "<model_name> <models_path> <processed_data_path>"
        )
        sys.exit(1)

    model_name = sys.argv[1]
    models_path = Path(sys.argv[2])
    processed_path = Path(sys.argv[3])

    if model_name not in ["MoaModel", "ResNet"]:
        raise ValueError(
            "model_name must be MoaModel or ResNet"
        )

    device = get_device()

    print(f"Device: {device}")
    print(f"\nEvaluating {model_name} 5-fold models...")

    fold_scores = []

    for fold in range(5):

        print(f"\n========== FOLD {fold} ==========")

        valid_path = processed_path / f"valid_fold{fold}.csv"

        valid_df = pd.read_csv(valid_path)

        # Same preprocessing/feature selection used during training.
        # column 0 = sig_id
        # column 1 = cp_type, which is dropped
        valid_df = valid_df.drop(columns=valid_df.columns[1])

        # After dropping cp_type:
        # The last 206 columns are always the scored MoA targets.
        # Everything after sig_id and before the targets is a model feature.
        TARGET_COUNT = 206

        features = valid_df.iloc[:, 1:-TARGET_COUNT].values.astype(
            np.float32
        )

        targets = valid_df.iloc[:, -TARGET_COUNT:].values.astype(
            np.float32
        )

        print(f"Validation rows: {len(valid_df)}")
        print(f"Features: {features.shape[1]}")
        print(f"Targets: {targets.shape[1]}")

        model_file = models_path / f"{model_name}_fold{fold}.pth"

        checkpoint = torch.load(
            model_file,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint["model_state_dict"]

        num_features = features.shape[1]
        num_targets = targets.shape[1]

        if model_name == "MoaModel":
            has_batch_norm = any(
                key.startswith("batch_norm1.")
                for key in state_dict.keys()
            )

            model = MoaModel(
                num_features=num_features,
                num_targets=num_targets,
                layer1_size=LAYER1_SIZE,
                layer2_size=LAYER2_SIZE,
                batch_norm=has_batch_norm,
            )

            print(f"BatchNorm: {has_batch_norm}")
        else:
            model = ResNet(
                num_features=num_features,
                num_targets=num_targets,
                layer_sizes=[1024, 1024, 1024],
            )

        model.load_state_dict(state_dict)
        model.to(device)

        predictions = get_predictions(
            model,
            features,
            device,
        )

        score = mean_column_log_loss(
            targets,
            predictions,
        )

        fold_scores.append(score)

        print(f"Fold {fold} log loss: {score:.8f}")

        del model

        if device.type == "mps":
            torch.mps.empty_cache()

    average_score = np.mean(fold_scores)

    print("\n===================================")
    print("FINAL RESULT")
    print("===================================")

    for fold, score in enumerate(fold_scores):
        print(f"Fold {fold}: {score:.8f}")

    print(f"\nMean CV Log Loss: {average_score:.8f}")

    print("\nFinal model selected based on 5-fold cross-validation.")


if __name__ == "__main__":
    main()
