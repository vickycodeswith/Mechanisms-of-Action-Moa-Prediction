# Mechanisms of Action (MoA) Prediction

An end-to-end deep learning project for predicting the **Mechanism of Action (MoA)** of compounds from high-dimensional gene-expression and cell-viability data.

This project uses a **5-fold cross-validated Feedforward Neural Network (MLP)** with feature engineering and Batch Normalization. The final model achieved a **mean 5-fold CV log loss of 0.01714092**.

---

## Project Overview

The goal of this project is to predict the biological mechanisms of action associated with compounds using cellular response data.

The model learns from:

- **Gene expression features (`g-*`)**
- **Cell viability features (`c-*`)**
- Treatment metadata:
  - Treatment time (`cp_time`)
  - Dose (`cp_dose`)
- Engineered statistical features

The task is a **multi-label binary classification problem** with **206 scored MoA targets**.

---

## Dataset

The project uses the **Kaggle Mechanisms of Action (MoA) Prediction** dataset.

The dataset contains:

- `train_features.csv` — training features
- `train_targets_scored.csv` — 206 scored MoA targets
- `train_targets_nonscored.csv` — additional non-scored targets
- `train_drug.csv` — drug identifiers
- `test_features.csv` — test features
- `sample_submission.csv` — submission format

### Feature groups

| Feature Group | Description |
|---|---|
| `g-*` | Gene expression measurements |
| `c-*` | Cell viability measurements |
| `cp_time` | Treatment duration |
| `cp_dose` | Treatment dose |
| `cp_type` | Compound/control indicator |

Control samples do not have MoA activity and are assigned zero predictions during inference.

---

## Machine Learning Approach

### Data preprocessing

The preprocessing pipeline:

1. Loads the raw Kaggle dataset.
2. Removes control perturbation samples from the training set.
3. Encodes categorical treatment information.
4. Separates features and 206 target labels.
5. Performs feature engineering.
6. Creates 5 cross-validation folds.
7. Saves processed training and validation data.

---

## Feature Engineering

In addition to the original gene-expression and cell-viability features, statistical features were created.

### Gene-expression statistics

- `g_mean`
- `g_std`
- `g_min`
- `g_max`

### Cell-viability statistics

- `c_mean`
- `c_std`
- `c_min`
- `c_max`

This increased the model input from:

```text
874 original model features
882 final model features
Final Model

The final selected model is a Feedforward Neural Network with Batch Normalization.

882 Input Features
        │
        ▼
Linear Layer
882 → 1024
        │
        ▼
Batch Normalization
        │
        ▼
ReLU
        │
        ▼
Linear Layer
1024 → 2048
        │
        ▼
Batch Normalization
        │
        ▼
ReLU
        │
        ▼
Linear Layer
2048 → 206
        │
        ▼
206 MoA Probabilities
Final configuration
Parameter	Value
Input features	882
Hidden layer 1	1024
Hidden layer 2	2048
Output targets	206
Batch normalization	Yes
Dropout	0.0
Batch size	64
Learning rate	0.001
Cross-validation	5-fold
Device	Apple MPS / CUDA / CPU
Experimentation

Several experiments were performed instead of selecting the first working model.

Lower log loss is better.

Experiment	Mean CV Log Loss
Original MLP	0.01794079
Feature Engineering MLP — 2 epochs	0.01775458
Feature Engineering MLP — 5 epochs	0.01762784
Feature Engineering + Dropout	0.01769693
Feature Engineering + Scheduler	0.01766826
Wider MLP	~0.01780423
Feature Engineering + BatchNorm	0.01714092 🏆
Model selection

Batch Normalization produced the best and most consistent validation performance across all five folds.

Final fold scores:

Fold	Log Loss
Fold 0	0.01746557
Fold 1	0.01709761
Fold 2	0.01700389
Fold 3	0.01716766
Fold 4	0.01696986
Mean	0.01714092
Evaluation Metric

The competition uses mean column-wise logarithmic loss.

Lower values indicate better performance.

For each of the 206 MoA targets, binary log loss is calculated and then averaged across all targets and samples.

The project's evaluation script reproduces this metric locally using the validation folds.

5-Fold Cross Validation

The final model uses five validation folds.

For each fold:

Training data is loaded.
Validation data is loaded.
A new model is initialized.
The model is trained.
The checkpoint with the best validation loss is saved.
The validation predictions are evaluated.
The five fold scores are averaged.

This reduces dependence on a single train/validation split and provides a more reliable estimate of model performance.

Prediction Pipeline

The final inference pipeline performs 5-fold ensemble prediction.

Test Data
    │
    ▼
Feature Preprocessing
    │
    ▼
882 Features
    │
    ├── Fold 0 Model
    ├── Fold 1 Model
    ├── Fold 2 Model
    ├── Fold 3 Model
    └── Fold 4 Model
    │
    ▼
Average 5 Predictions
    │
    ▼
206 MoA Probabilities
    │
    ▼
Restore Original Test Rows
    │
    ▼
Set Control Predictions to 0
    │
    ▼
submission.csv

The final prediction pipeline was successfully verified with:

3982 test rows
206 target columns
3624 treatment rows
358 control rows

Control predictions were verified to be exactly zero.

Project Structure
mechanisms-of-action-moa-prediction/
│
├── configs/
│   ├── MoaModel_baseline_config.json
│   ├── MoaModel_config.json
│   ├── MoaModel_FE_config.json
│   ├── MoaModel_FE_5EP_config.json
│   ├── MoaModel_FE_BATCHNORM_config.json
│   ├── MoaModel_FE_DROPOUT_config.json
│   ├── MoaModel_FE_SCHEDULER_config.json
│   └── MoaModel_FE_WIDE_config.json
│
├── data/
│   ├── raw/                  # Downloaded dataset, ignored by Git
│   ├── processed/            # Processed dataset
│   └── predictions_batchnorm/
│
├── models/                   # Model artifacts
│
├── notebooks/                # Analysis notebooks
│
├── reports/
│   └── figures/
│
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── metrics.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   └── ...
│
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
└── tox.ini
Technologies Used
Python
NumPy
Pandas
Scikit-learn
PyTorch
PyTorch MPS
Joblib
TensorBoard
Kaggle API
Git / GitHub
Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mechanisms-of-action-moa-prediction

Install the required Python packages:

pip install -r requirements.txt

Configure the Kaggle API:

~/.kaggle/kaggle.json

Set the correct permissions:

chmod 600 ~/.kaggle/kaggle.json
Download Dataset

The project uses the Kaggle MoA Prediction dataset.

After accepting the competition rules, download the dataset using the Kaggle API.

The raw data should be placed inside:

data/raw/
Data Preprocessing

Run:

python src/dataset.py data/raw/ data/processed_fe/

This creates the processed training and validation data used by the final model.

Train the Final Model

The final BatchNorm configuration is:

configs/MoaModel_FE_BATCHNORM_config.json

Run:

python src/train.py \
    --model_name MoaModel \
    --config configs/MoaModel_FE_BATCHNORM_config.json

The trained fold models are written to:

models_fe_batchnorm/
Evaluate the Model

Run:

python src/evaluate.py \
    MoaModel \
    models_fe_batchnorm \
    data/processed_fe

Expected result:

Mean CV Log Loss: 0.01714092
Generate Predictions

Run:

python src/predict.py \
    models_fe_batchnorm \
    data/processed_fe \
    data/predictions_batchnorm

The final submission is generated at:

data/predictions_batchnorm/submission.csv
Reproducibility

The project stores separate configuration files for the major experiments.

This makes it possible to reproduce the model comparison without manually changing architecture parameters inside the source code.

The final configuration is:

{
    "model_name": "MoaModel",
    "layer1_size": 1024,
    "layer2_size": 2048,
    "dropout_rate": 0.0,
    "batch_norm": true,
    "lr": 0.001,
    "batch_size": 64,
    "num_epochs": 5,
    "input_path": "data/processed_fe",
    "output_path": "models_fe_batchnorm",
    "folds": [0, 1, 2, 3, 4]
}
Key Results
Final performance
5-Fold CV Mean Log Loss
        ↓
     0.01714092
        ↓
     FINAL MODEL

The final model improved over the original MLP baseline:

Original MLP:       0.01794079
Final BatchNorm:    0.01714092

This represents approximately a 4.46% relative reduction in log loss compared with the original baseline.

Future Improvements

Potential future work includes:

FT-Transformer
TabNet
Hyperparameter optimization
Advanced feature selection
Model ensembling
Explainable AI
Target-specific modeling
External biological knowledge features
Automated experiment tracking
License

This project is released under the license included in the repository.

Author

Nitesh Kumar Yadav

AI/ML & Data Science Project Portfolio
