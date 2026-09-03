import logging
import click
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from utils import seed_everything
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict
import joblib
from collections import defaultdict

SEED = 42


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    try:
        logging.info(f"Loading data from {input_path}")
        return pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        raise


def encode_categorical(
    df: pd.DataFrame, cols: List[str], encoders: Dict = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features in the DataFrame using LabelEncoder.

    Args:
        df: Input DataFrame.
        cols: List of column names representing the categorical features.
        encoders: Dictionary of encoders for each categorical feature.

    Returns:
        A tuple containing the encoded DataFrame and the updated encoders.
    """
    logging.info(f"Encoding categorical features: {cols}")
    if encoders is None:
        encoders = defaultdict(LabelEncoder)

    for col in cols:
        try:
            df[col] = encoders[col].fit_transform(df[col])
        except Exception as e:
            logging.error(f"Error encoding column {col}: {e}")
            raise
    return df, encoders



def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-level statistical features for gene-expression
    and cell-viability measurements.
    """

    df = df.copy()

    gene_cols = [
        col for col in df.columns
        if col.startswith("g-")
    ]

    cell_cols = [
        col for col in df.columns
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


def preprocess_data(
    df: pd.DataFrame, encoders: Dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess the DataFrame by removing control samples and encoding categorical features.

    Args:
        df: Input DataFrame.
        encoders: Dictionary of encoders for categorical features.

    Returns:
        A tuple containing the preprocessed DataFrame and the updated encoders.
    """
    logging.info("Preprocessing data...")
    df = df[df["cp_type"] != "ctl_vehicle"].reset_index(drop=True)
    df, encoders = encode_categorical(df, ["cp_time", "cp_dose"], encoders)

    return df, encoders


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(input_path: str, output_path: str) -> None:
    """
    Main function to execute the data processing pipeline.

    Args:
        input_path: Path to the input data directory.
        output_path: Path to the output directory for processed data.

    Returns:
        None
    """
    log_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    seed_everything(SEED)
    input_path: Path = Path(input_path)
    output_path: Path = Path(output_path)

    # Loading data
    logging.info(f"Loading raw data from {input_path}")
    train_features: pd.DataFrame = load_data(input_path / "train_features.csv")
    train_targets: pd.DataFrame = load_data(input_path / "train_targets_scored.csv")
    test_features: pd.DataFrame = load_data(input_path / "test_features.csv")

    # Merge train features and targets
    train: pd.DataFrame = train_features.merge(train_targets, on="sig_id")

    # Preprocess data
    train, encoders = preprocess_data(train)

    # Add statistical features BEFORE the target columns.
    feature_columns = train_features.columns.tolist()
    feature_columns.remove("sig_id")

    gene_cols = [col for col in feature_columns if col.startswith("g-")]
    cell_cols = [col for col in feature_columns if col.startswith("c-")]

    engineered = pd.DataFrame(index=train.index)

    engineered["g_mean"] = train[gene_cols].mean(axis=1)
    engineered["g_std"] = train[gene_cols].std(axis=1)
    engineered["g_min"] = train[gene_cols].min(axis=1)
    engineered["g_max"] = train[gene_cols].max(axis=1)

    engineered["c_mean"] = train[cell_cols].mean(axis=1)
    engineered["c_std"] = train[cell_cols].std(axis=1)
    engineered["c_min"] = train[cell_cols].min(axis=1)
    engineered["c_max"] = train[cell_cols].max(axis=1)

    target_columns = train_targets.columns.tolist()[1:]

    train = pd.concat(
        [
            train.drop(columns=target_columns),
            engineered,
            train[target_columns],
        ],
        axis=1,
    )

    test_features, _ = preprocess_data(test_features, encoders)

    test_engineered = pd.DataFrame(index=test_features.index)

    test_engineered["g_mean"] = test_features[gene_cols].mean(axis=1)
    test_engineered["g_std"] = test_features[gene_cols].std(axis=1)
    test_engineered["g_min"] = test_features[gene_cols].min(axis=1)
    test_engineered["g_max"] = test_features[gene_cols].max(axis=1)

    test_engineered["c_mean"] = test_features[cell_cols].mean(axis=1)
    test_engineered["c_std"] = test_features[cell_cols].std(axis=1)
    test_engineered["c_min"] = test_features[cell_cols].min(axis=1)
    test_engineered["c_max"] = test_features[cell_cols].max(axis=1)

    test_features = pd.concat(
        [
            test_features,
            test_engineered,
        ],
        axis=1,
    )

    # Save encoders for future use
    joblib.dump(encoders, output_path / "encoders.pkl")

    # Split train data into train and valid
    kfold: StratifiedKFold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, train["cp_time"])):
        train_fold: pd.DataFrame = train.loc[train_idx, :]
        valid_fold: pd.DataFrame = train.loc[valid_idx, :]

        # Save data
        logging.info(f"Saving processed data to {output_path}")
        train_fold.to_csv(output_path / f"train_fold{fold}.csv", index=False)
        valid_fold.to_csv(output_path / f"valid_fold{fold}.csv", index=False)

    test_features.to_csv(output_path / "test.csv", index=False)


if __name__ == "__main__":
    main()
