from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download


# ============================================================
# Configuration
# ============================================================

HF_REPO_ID = "Vicky8021/Mechanisms-of-Action-MoA-Prediction"
HF_REPO_TYPE = "model"

MODEL_FILES = [
    "MoaModel_fold0.pth",
    "MoaModel_fold1.pth",
    "MoaModel_fold2.pth",
    "MoaModel_fold3.pth",
    "MoaModel_fold4.pth",
]

ENCODER_FILE = "encoders.pkl"

TARGET_FILE = Path("target_names.txt")

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="MoA Prediction",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# Model
# ============================================================

class MoaModel(nn.Module):

    def __init__(
        self,
        num_features,
        num_targets,
        layer1_size=1024,
        layer2_size=2048,
        batch_norm=True,
    ):
        super().__init__()

        self.layer1 = nn.Linear(
            num_features,
            layer1_size,
        )

        self.layer2 = nn.Linear(
            layer1_size,
            layer2_size,
        )

        self.layer3 = nn.Linear(
            layer2_size,
            num_targets,
        )

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


# ============================================================
# Download Model from Hugging Face
# ============================================================

@st.cache_resource
def download_model(filename):

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
    )


# ============================================================
# Load Target Names
# ============================================================

@st.cache_data
def load_target_names():

    if not TARGET_FILE.exists():

        raise FileNotFoundError(
            "target_names.txt was not found."
        )

    with open(
        TARGET_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        target_names = [
            line.strip()
            .strip(",")
            .strip('"')
            for line in file
            if line.strip()
        ]

    if len(target_names) != 206:

        raise ValueError(
            f"Expected 206 target names, "
            f"but found {len(target_names)}."
        )

    return target_names


# ============================================================
# Load Encoders
# ============================================================

@st.cache_resource
def load_encoders():

    encoder_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=ENCODER_FILE,
        repo_type=HF_REPO_TYPE,
    )

    return joblib.load(encoder_path)


# ============================================================
# Feature Engineering
# ============================================================

def add_statistical_features(df):

    df = df.copy()

    gene_cols = [
        col
        for col in df.columns
        if col.startswith("g-")
    ]

    cell_cols = [
        col
        for col in df.columns
        if col.startswith("c-")
    ]

    if gene_cols:

        df["g_mean"] = df[gene_cols].mean(axis=1)
        df["g_std"] = df[gene_cols].std(axis=1)
        df["g_min"] = df[gene_cols].min(axis=1)
        df["g_max"] = df[gene_cols].max(axis=1)

    if cell_cols:

        df["c_mean"] = df[cell_cols].mean(axis=1)
        df["c_std"] = df[cell_cols].std(axis=1)
        df["c_min"] = df[cell_cols].min(axis=1)
        df["c_max"] = df[cell_cols].max(axis=1)

    return df


# ============================================================
# Input Validation
# ============================================================

def validate_input(df):

    required_columns = [
        "sig_id",
        "cp_type",
        "cp_time",
        "cp_dose",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:

        raise ValueError(
            "The uploaded CSV is empty."
        )

    if df["sig_id"].duplicated().any():

        raise ValueError(
            "Duplicate sig_id values found."
        )

    allowed_cp_types = {
        "trt_cp",
        "ctl_vehicle",
    }

    invalid_cp_types = (
        set(df["cp_type"].unique())
        - allowed_cp_types
    )

    if invalid_cp_types:

        raise ValueError(
            "Invalid cp_type values: "
            f"{invalid_cp_types}"
        )


# ============================================================
# Preprocess Uploaded Data
# ============================================================

def preprocess_input(
    df,
    encoders,
):

    df = df.copy()

    validate_input(df)

    # --------------------------------------------------------
    # Preserve original IDs
    # --------------------------------------------------------

    ids = df["sig_id"].copy()

    # --------------------------------------------------------
    # Identify treatment/control rows
    # --------------------------------------------------------

    treatment_mask = (
        df["cp_type"] == "trt_cp"
    )

    control_mask = (
        df["cp_type"] == "ctl_vehicle"
    )

    treatment_df = df.loc[
        treatment_mask
    ].copy()

    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    for col in [
        "cp_time",
        "cp_dose",
    ]:

        encoder = encoders[col]

        unknown_values = (
            set(treatment_df[col].unique())
            - set(encoder.classes_)
        )

        if unknown_values:

            raise ValueError(
                f"Unknown values in {col}: "
                f"{unknown_values}"
            )

        treatment_df[col] = encoder.transform(
            treatment_df[col]
        )

    # --------------------------------------------------------
    # Same statistical features as training
    # --------------------------------------------------------

    treatment_df = add_statistical_features(
        treatment_df
    )

    # --------------------------------------------------------
    # Remove non-model columns
    # --------------------------------------------------------

    feature_df = treatment_df.drop(
        columns=[
            "sig_id",
            "cp_type",
        ],
        errors="ignore",
    )

    return (
        feature_df,
        ids,
        treatment_mask,
        control_mask,
    )


# ============================================================
# Load 5-Fold Ensemble
# ============================================================

@st.cache_resource
def load_models():

    models = []

    for filename in MODEL_FILES:

        model_path = download_model(
            filename
        )

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )

        if "model_state_dict" not in checkpoint:

            raise ValueError(
                f"'model_state_dict' missing "
                f"from {filename}"
            )

        state_dict = checkpoint[
            "model_state_dict"
        ]

        # Detect architecture from weights
        input_features = state_dict[
            "layer1.weight"
        ].shape[1]

        layer1_size = state_dict[
            "layer1.weight"
        ].shape[0]

        layer2_size = state_dict[
            "layer2.weight"
        ].shape[0]

        num_targets = state_dict[
            "layer3.weight"
        ].shape[0]

        has_batch_norm = any(
            key.startswith(
                "batch_norm1."
            )
            for key in state_dict.keys()
        )

        model = MoaModel(
            num_features=input_features,
            num_targets=num_targets,
            layer1_size=layer1_size,
            layer2_size=layer2_size,
            batch_norm=has_batch_norm,
        )

        model.load_state_dict(
            state_dict
        )

        model.to(DEVICE)
        model.eval()

        models.append(model)

    return models


# ============================================================
# Prediction
# ============================================================

def predict(
    models,
    features,
):

    if features.empty:

        return np.empty(
            (0, 206),
            dtype=np.float32,
        )

    x = torch.tensor(
        features.values,
        dtype=torch.float32,
        device=DEVICE,
    )

    predictions = []

    with torch.no_grad():

        for model in models:

            logits = model(x)

            probabilities = torch.sigmoid(
                logits
            )

            predictions.append(
                probabilities
                .cpu()
                .numpy()
            )

    return np.mean(
        predictions,
        axis=0,
    )


# ============================================================
# Full Prediction Matrix
# ============================================================

def create_full_predictions(
    input_df,
    treatment_mask,
    probabilities,
    target_names,
):

    full_probabilities = np.zeros(
        (
            len(input_df),
            len(target_names),
        ),
        dtype=np.float32,
    )

    # Only treatment rows receive model predictions
    full_probabilities[
        treatment_mask.values
    ] = probabilities

    result_df = pd.DataFrame(
        full_probabilities,
        columns=target_names,
    )

    result_df.insert(
        0,
        "sig_id",
        input_df["sig_id"].values,
    )

    return result_df


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "Model Information"
)

st.sidebar.write(
    f"**Device:** `{DEVICE}`"
)

st.sidebar.write(
    "**Architecture:** MLP + Batch Normalization"
)

st.sidebar.write(
    "**Input Features:** 882"
)

st.sidebar.write(
    "**MoA Targets:** 206"
)

st.sidebar.write(
    "**Cross-Validation:** 5-Fold"
)

st.sidebar.write(
    "**Ensemble:** 5 Models"
)


# ============================================================
# Main Application
# ============================================================

st.title(
    "🧬 Mechanisms of Action Prediction"
)

st.markdown(
    """
### AI-powered multi-label drug MoA prediction

Upload a CSV containing drug-treatment features and the
model will predict **206 possible mechanisms of action**
using a 5-fold neural-network ensemble.
"""
)

st.info(
    "Final local 5-fold CV Log Loss: 0.01714092"
)


# ============================================================
# Upload
# ============================================================

st.header(
    "Upload Treatment Data"
)

uploaded_file = st.file_uploader(
    "Upload a CSV containing MoA test features",
    type=["csv"],
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    try:

        input_df = pd.read_csv(
            uploaded_file
        )

        validate_input(
            input_df
        )

        # ----------------------------------------------------
        # Dataset summary
        # ----------------------------------------------------

        st.subheader(
            "Uploaded Data"
        )

        treatment_count = int(
            (
                input_df["cp_type"]
                == "trt_cp"
            ).sum()
        )

        control_count = int(
            (
                input_df["cp_type"]
                == "ctl_vehicle"
            ).sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Samples",
            f"{len(input_df):,}",
        )

        col2.metric(
            "Treatment Samples",
            f"{treatment_count:,}",
        )

        col3.metric(
            "Control Samples",
            f"{control_count:,}",
        )

        st.write(
            f"Columns: **{input_df.shape[1]}**"
        )

        st.dataframe(
            input_df.head(),
            width="stretch",
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        if st.button(
            "🚀 Predict Mechanisms of Action",
            type="primary",
            width="stretch",
        ):

            with st.spinner(
                "Loading models and generating predictions..."
            ):

                target_names = (
                    load_target_names()
                )

                encoders = (
                    load_encoders()
                )

                models = (
                    load_models()
                )

                (
                    features,
                    ids,
                    treatment_mask,
                    control_mask,
                ) = preprocess_input(
                    input_df,
                    encoders,
                )

                # Verify model input
                if features.shape[1] != 882:

                    raise ValueError(
                        f"Expected 882 features, "
                        f"received "
                        f"{features.shape[1]}."
                    )

                probabilities = predict(
                    models,
                    features,
                )

                # Restore complete input alignment
                result_df = (
                    create_full_predictions(
                        input_df,
                        treatment_mask,
                        probabilities,
                        target_names,
                    )
                )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if len(result_df) != len(input_df):

                raise ValueError(
                    "Prediction row count does not "
                    "match uploaded data."
                )

            max_control_prediction = 0.0

            if control_count > 0:

                max_control_prediction = (
                    result_df.loc[
                        control_mask.values,
                        target_names,
                    ]
                    .to_numpy()
                    .max()
                )

            if max_control_prediction != 0:

                raise ValueError(
                    "Control-row predictions "
                    "are not zero."
                )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            st.success(
                "Prediction completed successfully!"
            )

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            st.header(
                "Prediction Summary"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Rows",
                f"{len(result_df):,}",
            )

            col2.metric(
                "Predicted Rows",
                f"{treatment_count:,}",
            )

            col3.metric(
                "Control Rows",
                f"{control_count:,}",
            )

            # ------------------------------------------------
            # Top MoA predictions
            # ------------------------------------------------

            st.header(
                "Top Predicted Mechanisms of Action"
            )

            treatment_results = (
                result_df.loc[
                    treatment_mask.values
                ]
                .reset_index(drop=True)
            )

            samples_to_show = min(
                10,
                len(treatment_results),
            )

            for row_idx in range(
                samples_to_show
            ):

                row = treatment_results.iloc[
                    row_idx
                ]

                values = (
                    row[target_names]
                    .astype(float)
                    .values
                )

                top_indices = np.argsort(
                    values
                )[::-1][:10]

                top_results = pd.DataFrame(
                    {
                        "Mechanism of Action": [
                            target_names[i]
                            for i in top_indices
                        ],
                        "Probability": [
                            values[i]
                            for i in top_indices
                        ],
                    }
                )

                top_results[
                    "Probability"
                ] = top_results[
                    "Probability"
                ].map(
                    lambda x: f"{x:.2%}"
                )

                st.subheader(
                    f"Sample {row_idx + 1} — "
                    f"{row['sig_id']}"
                )

                st.dataframe(
                    top_results,
                    width="stretch",
                    hide_index=True,
                )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            st.header(
                "Download Predictions"
            )

            csv_data = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="moa_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

            # ------------------------------------------------
            # Final Validation
            # ------------------------------------------------

            st.success(
                f"Validation passed: "
                f"{len(result_df):,} rows × "
                f"{len(result_df.columns)} columns"
            )

            st.caption(
                f"Control rows: {control_count:,} | "
                f"Maximum control prediction: "
                f"{max_control_prediction:.6f}"
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

else:

    st.warning(
        "Upload a CSV file to start prediction."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "MoA Prediction • PyTorch • "
    "5-Fold Neural Network Ensemble"
)