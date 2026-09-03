# 🧬 Mechanisms of Action (MoA) Prediction

> **End-to-end Machine Learning system for predicting the Mechanisms of Action (MoA) of drug treatments using gene-expression and cell-viability data.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Live Demo

### 👉 [**Launch the MoA Prediction Web App**](https://mechanisms-of-action-moa-prediction-app.streamlit.app/)

Try the deployed machine learning application directly in your browser.

The application provides an interactive interface for generating **Mechanisms of Action predictions** using the trained model.

---

## 📌 Project Overview

**Mechanisms of Action (MoA) Prediction** is a multi-label machine learning problem in computational biology and drug discovery.

The objective of this project is to predict which biological mechanisms of action are activated by a given drug treatment.

The model learns from:

* 🧬 Gene-expression features
* 🧪 Cell-viability features
* 💊 Treatment metadata
* 🎯 Multiple binary MoA targets

The project implements a complete ML workflow:

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
MoA Predictions
```

---

## 🎯 Problem Statement

This project treats MoA prediction as a **multi-label binary classification problem**.

For each drug treatment, the model predicts the probability of **206 different MoA targets**.

### Input

The final pipeline uses:

```text
882 Features
```

including processed biological and treatment-related information.

### Output

```text
206 MoA Prediction Probabilities
```

Each output represents the predicted probability of a specific mechanism of action.

---

## 📊 Dataset

This project uses the **Mechanisms of Action (MoA) Prediction dataset from Kaggle**.

The raw dataset is intentionally excluded from the repository because of its size.

### Main Dataset Files

```text
train_features.csv
train_targets_scored.csv
train_targets_nonscored.csv
train_drug.csv
test_features.csv
sample_submission.csv
```

### Feature Groups

| Feature Group      | Description                   |
| ------------------ | ----------------------------- |
| `g-*`              | Gene-expression features      |
| `c-*`              | Cell-viability features       |
| Treatment Metadata | Treatment-related information |
| MoA Targets        | 206 binary target variables   |

---

# 🧠 Machine Learning Approach

## 1. Data Preprocessing

The preprocessing pipeline:

* Loads the raw Kaggle datasets
* Removes control-vehicle samples from training data
* Encodes categorical treatment variables
* Processes gene-expression features
* Processes cell-viability features
* Creates cross-validation folds
* Saves processed datasets

Run:

```bash
python src/dataset.py data/raw/ data/processed_fe/
```

---

## 2. Feature Engineering

Additional statistical features are generated from the biological feature groups.

### Gene-Expression Statistics

```text
g_mean
g_std
g_min
g_max
```

### Cell-Viability Statistics

```text
c_mean
c_std
c_min
c_max
```

These features provide the neural network with compact statistical information about the underlying biological feature distributions.

### Final Feature Space

```text
Input Features: 882
Target Variables: 206
```

---

# 🏗️ Neural Network Architecture

The final model is a **Feedforward Neural Network with Batch Normalization**.

```text
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
```

---

## ⚙️ Final Model Configuration

| Parameter           |  Value |
| ------------------- | -----: |
| Input Features      |    882 |
| Hidden Layer 1      |   1024 |
| Hidden Layer 2      |   2048 |
| Output Targets      |    206 |
| Batch Size          |     64 |
| Learning Rate       |  0.001 |
| Maximum Epochs      |      5 |
| Batch Normalization |    Yes |
| Optimizer           |   Adam |
| Cross-Validation    | 5-Fold |
| Dropout             |    0.0 |

The training pipeline also uses a `ReduceLROnPlateau` learning-rate scheduler.

---

# 🔬 Model Experiments

Multiple experiments were performed to identify an effective architecture.

| Experiment                            | Mean 5-Fold CV Log Loss |
| ------------------------------------- | ----------------------: |
| Original MLP Baseline                 |            `0.01794079` |
| MLP + Feature Engineering + BatchNorm |        **`0.01714092`** |
| MLP + Dropout                         |            `0.01769693` |
| MLP + Scheduler                       |            `0.01766826` |

### 🏆 Best Configuration

The final **Feature Engineering + Batch Normalization MLP** achieved:

```text
Mean 5-Fold CV Log Loss
0.01714092
```

This represents approximately:

```text
4.46% relative reduction in log loss
```

compared with the original MLP baseline.

> **Note:** `0.01714092` is the project's local 5-fold cross-validation result and is **not an official Kaggle leaderboard score**.

---

# 🔁 5-Fold Cross-Validation

Five independent models are trained:

```text
Fold 0 ──→ Model 0
Fold 1 ──→ Model 1
Fold 2 ──→ Model 2
Fold 3 ──→ Model 3
Fold 4 ──→ Model 4
```

During prediction, the outputs from all five models are combined:

```text
Model 0 ─┐
Model 1 ─┤
Model 2 ─┤
Model 3 ─┼──→ Final MoA Predictions
Model 4 ─┘
```

This approach helps improve prediction robustness compared with relying on a single model.

---

# 📈 Evaluation Metric

The primary evaluation metric is:

## Mean Column-Wise Binary Log Loss

Binary log loss is calculated independently for every MoA target and then averaged across all **206 targets**.

```text
Lower Log Loss = Better Performance
```

### Final Result

```text
Baseline:
0.01794079

Final Model:
0.01714092

Relative Improvement:
≈ 4.46%
```

---

# 🔮 Prediction Pipeline

The prediction pipeline uses all five trained fold models.

For control-vehicle samples:

```text
Prediction = 0
```

The final predictions are aligned with the original `sig_id` order.

---

# ✅ Submission Validation

The generated submission is automatically validated for:

* Submission shape
* Column names
* `sig_id` ordering
* Duplicate IDs
* Missing values
* Control-vehicle predictions

### Validation Result

```text
Submission shape: (3982, 207)
Columns match: True
sig_id order matches: True
sig_id unique: True
NaN values: 0
Control rows: 358
Max control prediction: 0.0

SUBMISSION VALIDATION: PASS
```

---

# 📁 Project Structure

```text
Mechanisms-of-Action-Moa-Prediction/
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
├── target_names.txt
├── test_environment.py
└── tox.ini
```

---

# 🛠️ Technologies & Tools

### Programming

* Python

### Data Science

* NumPy
* Pandas
* Scikit-learn

### Deep Learning

* PyTorch
* Batch Normalization
* Adam Optimizer
* ReduceLROnPlateau

### Experimentation & Evaluation

* 5-Fold Cross-Validation
* Binary Log Loss
* TensorBoard

### Infrastructure & Development

* Kaggle API
* Git
* GitHub
* PyTorch MPS

### Deployment

* Streamlit

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/vickycodeswith/Mechanisms-of-Action-Moa-Prediction.git

cd Mechanisms-of-Action-Moa-Prediction
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Kaggle API Setup

Configure your Kaggle API credentials:

```text
~/.kaggle/kaggle.json
```

Set the appropriate permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

After accepting the relevant Kaggle competition rules, download the dataset and place the raw files inside:

```text
data/raw/
```

---

# ▶️ Reproduce the Project

## Step 1 — Prepare Dataset

Place the Kaggle dataset inside:

```text
data/raw/
```

---

## Step 2 — Preprocess Data

```bash
python src/dataset.py data/raw/ data/processed_fe/
```

---

## Step 3 — Train Final Model

```bash
python src/train.py \
    --model_name MoaModel \
    --config configs/MoaModel_FE_BATCHNORM_config.json
```

Trained models will be generated in:

```text
models_fe_batchnorm/
```

---

## Step 4 — Evaluate Model

```bash
python src/evaluate.py \
    MoaModel \
    models_fe_batchnorm \
    data/processed_fe
```

Expected result:

```text
Mean CV Log Loss: 0.01714092
```

---

## Step 5 — Generate Predictions

```bash
python src/predict.py \
    models_fe_batchnorm \
    data/processed_fe \
    data/predictions_batchnorm
```

The final prediction file will be:

```text
data/predictions_batchnorm/submission.csv
```

---

# 🖥️ Live Web Application

The trained model has been deployed as an interactive Streamlit application.

### 🚀 Try the Application

**[Open MoA Prediction App →](https://mechanisms-of-action-moa-prediction-app.streamlit.app/)**

The web application provides a user-friendly interface for interacting with the trained MoA prediction system without requiring users to run the complete ML pipeline locally.

---

# 🧪 Reproducibility

Model experiments are controlled through separate JSON configuration files.

This makes it possible to modify:

* Architecture
* Learning rate
* Batch size
* Number of epochs
* Batch normalization
* Dropout
* Other training parameters

without manually changing the core training code.

### Final Configuration

```text
configs/MoaModel_FE_BATCHNORM_config.json
```

This configuration reproduces the final Batch Normalization experiment.

---

# 🖥️ Hardware

The project was developed and tested on **Apple Silicon** using PyTorch's MPS backend when available.

The training code automatically selects the available device in the following order:

```text
CUDA
  ↓
MPS
  ↓
CPU
```

---

# 🔮 Future Improvements

Potential directions for improving the system include:

* [ ] FT-Transformer
* [ ] TabNet optimization
* [ ] Hyperparameter optimization
* [ ] Advanced feature selection
* [ ] Model ensembling
* [ ] Explainable AI
* [ ] Target-specific modeling
* [ ] External biological knowledge features
* [ ] Automated experiment tracking
* [ ] Advanced deep learning architectures

---

# 📚 Key Takeaways

This project demonstrates an end-to-end applied machine learning workflow covering:

```text
Data Engineering
      ↓
Feature Engineering
      ↓
Multi-Label Classification
      ↓
Deep Learning
      ↓
Cross-Validation
      ↓
Experimentation
      ↓
Model Evaluation
      ↓
Ensemble Prediction
      ↓
Model Deployment
```

The project combines **computational biology, machine learning, deep learning, feature engineering, model experimentation, and deployment** into a complete practical workflow.

---

# 📜 License

This project is released under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

# 👨‍💻 Author

**vickycodeswith**

GitHub:
https://github.com/vickycodeswith

---

## ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ **Star** on GitHub and exploring the live application.

### 🔗 Project Links

* 📂 **[GitHub Repository](https://github.com/vickycodeswith/Mechanisms-of-Action-Moa-Prediction)**
* 🚀 **[Live Streamlit App](https://mechanisms-of-action-moa-prediction-app.streamlit.app/)**
