<div align="center">

# Sleep Cognition ML

The open-source repository for an independent research project investigating feature reliability in machine learning prediction of cognitive impairment from sleep-derived physiological signals.

[![Python](https://img.shields.io/badge/Python-3.13-ffd43b?style=flat-square&logo=python&logoColor=white&labelColor=3776ab)](https://www.python.org/) [![Project Manager](https://img.shields.io/badge/Project_Manager-uv-de5fe9?style=flat-square&labelColor=24292e)](https://github.com/astral-sh/uv) [![License](https://img.shields.io/github/license/s-ray9/sleep-cognition-ml?style=flat-square&color=blue&label=License&labelColor=24292e)](LICENSE)

![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?style=flat-square&logo=scikit-learn&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-15045c?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-4d77cf?style=flat-square&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-3776ab?style=flat-square&logo=scipy&logoColor=white)

</div>

---

This project tests whether feature-importance rankings from machine learning models are reliable on small, imbalanced clinical datasets. It uses the 2026 PhysioNet Challenge sleep dataset as a test case. The analysis code is built on the Challenge's official example and scoring scripts.

## 📊 Results

### Model Performance

Cross-validated AUROC (area under the ROC curve); 0.5 = random guessing, 1.0 = perfect separation. ± indicates variability across cross-validation folds for a single model; ranges indicate the spread across multiple model types.

| Configuration | n | Model(s) | AUROC |
|---|---|---|---|
| Baseline | 40 | Random forest | 0.504 |
| All features (corrected) | 158 | Random forest | 0.737 ± 0.082 |
| Top-10 features (corrected) | 1,092<sup>a</sup> | Logistic regression, random forest, gradient boosting | 0.788–0.800 |

<sup>a</sup>*11 of 1,103 available patients excluded due to missing annotation files.*

### Feature Importance

Mean decrease in impurity; ± indicates variability across 20 random forest seeds trained on the same data (n = 1,092).

| Feature | Importance (mean ± std) | Status |
|---|---|---|
| BMI | 0.0563 ± 0.0116 | Reliable |
| Age | 0.0482 ± 0.0156 | Reliable |
| Respiratory zero-crossing rate | 0.0431 ± 0.0122 | Marginal |
| All other 72 features | ≤ 0.0431, std ≈ mean | Unreliable |

## 🛠️ Development

### Local Setup

```bash
# Clone the repository
git clone https://github.com/s-ray9/sleep-cognition-ml
cd sleep-cognition-ml

# Install dependencies
uv sync
```

### Usage

This project uses overnight polysomnography recordings from the [George B. Moody PhysioNet Challenge 2026](https://physionetchallenges.org/2026/) dataset.

#### Data Preparation

```bash
# Download the imbalanced training cohort (79 True, 934 False)
uv run python download_patients.py --split training --true 79 --false 934

# Sort training downloads into the Challenge-standard directory structure
uv run python sort_downloads.py data/training

# Download a balanced holdout cohort (5 True, 5 False)
uv run python download_patients.py --split holdout --true 5 --false 5

# Sort holdout downloads into the same directory structure
uv run python sort_downloads.py data/holdout
```

#### Model Training

```bash
# Train the model (200-tree random forest, class-weighted for imbalance)
uv run python train_model.py -d data/training -m model -v
```

*Note: this trains a single model via the official Challenge pipeline. This repository implements separate scripts for cross-validated evaluation, multi-seed feature-importance analysis, and cross-architecture comparison.*

## 🗂️ Project Structure

### Challenge Code

#### Unmodified

| File | Purpose | Arguments |
|---|---|---|
| `evaluate_model.py` | Scores predictions against labels | `-d <labels_file> -o <predictions_file> -p <prevalence_file>` |
| `helper_code.py` | Shared utilities used by the other files in this section | — |
| `run_model.py` | Runs a trained model on new data | `-d <data_folder> -m <model_folder> -o <output_folder> -v` |
| `train_model.py` | Trains a model | `-d <data_folder> -m <model_folder> -v` |

#### Modified

| File | Purpose | Arguments |
|---|---|---|
| `team_code.py` | Feature extraction and model definition | — |

### Original Code

Scripts use constants defined at the top of each file in place of command-line arguments where none are listed.

| File | Purpose | Arguments |
|---|---|---|
| `check_patient_overlap.py` | Verifies no patient contributes multiple sessions across splits | — |
| `compare_feature_subsets.py` | Cross-validated performance by feature subset size | — |
| `compare_models.py` | Cross-validated performance and importance across three model architectures | — |
| `download_patients.py` | Downloads patient data from Kaggle | `--split <training\|holdout> --true <n> --false <n>` |
| `evaluate_methodology.py` | 5-fold cross-validated AUROC on the full feature set | — |
| `experiment_utils.py` | Shared feature definitions and dataset loading | — |
| `rank_features.py` | Single-run feature importance from a trained model | — |
| `rank_features_multiseed.py` | Feature importance averaged across 20 random seeds | — |
| `sort_downloads.py` | Sorts downloaded files into the Challenge-standard directory structure | `<target_directory>` |

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

Code retained from the Challenge repository (see [Project Structure](#project-structure)) remains under their original **BSD 3-Clause License**. See [PHYSIONET-CHALLENGE-LICENSE](THIRD-PARTY-LICENSES/PHYSIONET-CHALLENGE-LICENSE) for more information.
