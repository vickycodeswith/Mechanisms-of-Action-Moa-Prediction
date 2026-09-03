# Mechanisms of Action (MoA) Prediction

A complete end-to-end machine learning project for predicting the **Mechanisms of Action (MoA)** of drugs from gene-expression and cell-viability data.

The project focuses on building, comparing, evaluating, and generating predictions from neural-network models using **5-fold cross-validation**.

---

## Project Overview

The goal of this project is to predict which biological mechanisms of action are activated by a given drug treatment.

The dataset contains:

- Gene-expression features
- Cell-viability features
- Treatment metadata
- Multiple binary MoA targets

The final pipeline includes:

```text
Raw Kaggle Dataset
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
5-Fold Cross-Validation
        ↓
Neural Network Training
        ↓
Batch Normalization
        ↓
Model Evaluation
        ↓
5-Fold Ensemble Prediction
        ↓
Submission File
Dataset

This project uses the Mechanisms of Action (MoA) Prediction dataset from Kaggle.

The competition task is a multi-label binary classification problem where the model predicts multiple possible MoA targets for each drug treatment.

Dataset Structure

The main files include:

train_features.csv
train_targets_scored.csv
train_targets_nonscored.csv
train_drug.csv
test_features.csv
sample_submission.csv

The training feature dataset contains:

Treatment metadata
Gene-expression features (g-*)
Cell-viability features (c-*)

The scored target dataset contains 206 MoA target variables.

The raw dataset is intentionally excluded from Git because of its size.

Machine Learning Problem

This is a multi-label binary classification problem.

For every treatment sample, the model predicts probabilities for 206 different MoA targets.

The output therefore has the form:

Input sample
    ↓
882 features
    ↓
Neural Network
    ↓
206 output probabilities
Evaluation Metric

The primary evaluation metric is mean column-wise binary log loss.

For each target column, binary log loss is calculated and then averaged across all 206 targets.

Lower values are better.

The final model achieved:

Mean 5-Fold CV Log Loss: 0.01714092

Important: 0.01714092 is the project's local 5-fold cross-validation result. It is not an official Kaggle leaderboard score.

Approach
1. Data Preprocessing

The preprocessing pipeline:

Loads the raw Kaggle datasets
Removes control-vehicle samples from the training data
Encodes categorical treatment variables
Processes gene-expression features
Processes cell-viability features
Creates train/validation folds
Saves the processed datasets locally

Run:

python src/dataset.py data/raw/ data/processed_fe/
2. Feature Engineering

Additional statistical features are created from the original gene and cell features.

Gene Features

The following statistics are calculated:

g_mean
g_std
g_min
g_max
Cell Features

The following statistics are calculated:

c_mean
c_std
c_min
c_max

These additional features provide the model with compact statistical information about the biological feature groups.

After preprocessing and feature engineering:

Input Features: 882
Targets: 206
Model Architecture

The final model is a feedforward neural network with Batch Normalization.

Architecture
Input
  │
  ▼
882 Features
  │
  ▼
Linear Layer
1024 Units
  │
  ▼
Batch Normalization
  │
  ▼
ReLU
  │
  ▼
Linear Layer
2048 Units
  │
  ▼
Batch Normalization
  │
  ▼
ReLU
  │
  ▼
Linear Layer
206 Units
  │
  ▼
MoA Predictions
Final Configuration
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
Model Training

The project uses 5-fold cross-validation.

Each fold trains a separate model:

Fold 0 → Model
Fold 1 → Model
Fold 2 → Model
Fold 3 → Model
Fold 4 → Model

The validation performance is measured independently for every fold.

Training Configuration
Parameter	Value
Input Features	882
Hidden Layer 1	1024
Hidden Layer 2	2048
Output Targets	206
Batch Size	64
Learning Rate	0.001
Maximum Epochs	5
Batch Normalization	Yes
Optimizer	Adam
Cross-Validation	5-Fold

The training pipeline also uses ReduceLROnPlateau learning-rate scheduling:

patience = 1
factor = 0.5
Experiments

Several experiments were performed to improve the baseline model.

Original MLP Baseline

Architecture:

874 Features
    ↓
1024
    ↓
2048
    ↓
206 Targets

Result:

Mean CV Log Loss: 0.01794079
Feature Engineering

Statistical features were added:

g_mean
g_std
g_min
g_max

c_mean
c_std
c_min
c_max

The resulting model used:

882 Features

The feature-engineered model improved the validation performance.

5-Epoch Training

Training for additional epochs was tested to determine whether the model could continue improving.

The validation loss generally improved during the first few epochs and then started showing signs of overfitting.

This experiment helped identify an effective training range for the final configuration.

Dropout Experiment

Dropout with:

dropout_rate = 0.2

was tested.

Result:

Mean CV Log Loss: 0.01769693

This was worse than the final BatchNorm configuration.

Learning-Rate Scheduler Experiment

A learning-rate scheduling experiment was also evaluated.

Result:

Mean CV Log Loss: 0.01766826

This was also worse than the final BatchNorm configuration.

Wider MLP Experiment

A wider first hidden layer was tested:

2048 → 2048

This did not outperform the final architecture.

Batch Normalization Experiment

Batch Normalization was added after the linear layers and before the ReLU activation.

Result:

Mean CV Log Loss: 0.01714092

This became the best verified configuration.

Final Model Comparison
Model	Mean 5-Fold CV Log Loss
Original MLP Baseline	0.01794079
MLP + Feature Engineering + BatchNorm	0.01714092
Improvement

The final model achieved approximately:

4.46% relative reduction in log loss

compared with the original MLP baseline.

Why Batch Normalization Was Selected

Among the tested configurations, the BatchNorm model produced the best verified 5-fold cross-validation result.

The improvement was:

Baseline
0.01794079
      ↓
BatchNorm
0.01714092

Therefore, the BatchNorm configuration was selected as the final model.

Prediction Pipeline

The prediction pipeline uses all five trained fold models.

Fold 0 Model ─┐
Fold 1 Model ─┤
Fold 2 Model ─┤
Fold 3 Model ─┼──→ Predictions
Fold 4 Model ─┘

The pipeline also handles control-vehicle samples.

For control-vehicle rows:

Prediction = 0

The final predictions are aligned with the original sig_id order.

Project Structure
mechanisms-of-action-moa-prediction/
│
├── configs/
│   ├── MoaModel_config.json
│   ├── MoaModel_baseline_config.json
│   ├── MoaModel_FE_config.json
│   ├── MoaModel_FE_5EP_config.json
│   ├── MoaModel_FE_BATCHNORM_config.json
│   ├── MoaModel_FE_DROPOUT_config.json
│   ├── MoaModel_FE_SCHEDULER_config.json
│   └── MoaModel_FE_WIDE_config.json
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   ├── exploratory/
│   └── reports/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── dataset.py
│   ├── dataset_baseline.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── metrics.py
│   ├── predict.py
│   ├── tabnet.py
│   ├── train.py
│   ├── train_baseline.py
│   ├── transformer.py
│   ├── utils.py
│   └── ...
│
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini

Generated datasets and model files are excluded from Git using .gitignore.

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
Git
GitHub
Installation
Clone the Repository
git clone https://github.com/vickycodeswith/Mechanisms-of-Action-Moa-Prediction.git
cd Mechanisms-of-Action-Moa-Prediction
Install Dependencies
pip install -r requirements.txt
Kaggle API Setup

Configure the Kaggle API credentials:

~/.kaggle/kaggle.json

Set the correct permissions:

chmod 600 ~/.kaggle/kaggle.json

After accepting the Kaggle competition rules, download the competition dataset.

Place the raw files inside:

data/raw/
Reproduce the Project
Step 1 — Download Dataset

Place the Kaggle dataset inside:

data/raw/
Step 2 — Preprocess Data
python src/dataset.py data/raw/ data/processed_fe/
Step 3 — Train Final Model
python src/train.py \
    --model_name MoaModel \
    --config configs/MoaModel_FE_BATCHNORM_config.json

Trained models will be generated in:

models_fe_batchnorm/
Step 4 — Evaluate
python src/evaluate.py \
    MoaModel \
    models_fe_batchnorm \
    data/processed_fe

Expected result:

Mean CV Log Loss: 0.01714092
Step 5 — Generate Predictions
python src/predict.py \
    models_fe_batchnorm \
    data/processed_fe \
    data/predictions_batchnorm

The final submission file will be:

data/predictions_batchnorm/submission.csv
Reproducibility

Major experiments are controlled through separate JSON configuration files.

This allows different model configurations to be tested without manually changing architecture parameters inside the source code.

The final configuration is:

configs/MoaModel_FE_BATCHNORM_config.json

This configuration reproduces the final BatchNorm experiment.

Hardware

The project was developed and tested on Apple Silicon using PyTorch's MPS (Metal Performance Shaders) backend when available.

The training code automatically selects:

CUDA
  ↓
MPS
  ↓
CPU

depending on available hardware.

Submission Validation

The generated submission is validated before completion.

The validation checks include:

Submission shape
Column names
sig_id order
Duplicate IDs
Missing values
Control-vehicle predictions

Final validation result:

Submission shape: (3982, 207)
Columns match: True
sig_id order matches: True
sig_id unique: True
NaN values: 0
Control rows: 358
Max control prediction: 0.0

SUBMISSION VALIDATION: PASS
Key Results
Original MLP
     │
     │ 0.01794079
     ▼
Feature Engineering
     │
     ▼
Batch Normalization
     │
     │ 0.01714092
     ▼
Final Model
Final Score
Mean 5-Fold CV Log Loss

0.01714092
Relative Improvement
≈ 4.46% lower log loss

compared with the original MLP baseline.

Future Improvements

Potential future improvements include:

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

vickycodeswith
