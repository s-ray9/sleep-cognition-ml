![s_ray9: systems developer, game developer.][banner]

*Banner background photo by [Conny Schneider](https://unsplash.com/@choys_) on [Unsplash](https://unsplash.com), modified for this project.*

# Sleep Cognition ML

<!-- [![Demo][demo-badge]][demo] -->
[![Latest Release][latest-release-badge]][latest-release]
[![License][license-badge]][license]

Independent research project testing how reliable feature-importance rankings are in machine learning models that predict cognitive impairment from sleep recordings.

## Tech stack

![Python][python-badge]
![NumPy][numpy-badge]
![pandas][pandas-badge]
![SciPy][scipy-badge]
![scikit-learn][scikit-learn-badge]
![uv][uv-badge]

## Overview

This project tests whether feature-importance rankings from machine learning models remain reliable on small, imbalanced clinical datasets. It uses the 2026 PhysioNet Challenge sleep dataset, which contains 1,103 patients and 84 positive cases, as a test case. A set of ten features predicted cognitive impairment consistently across three model types, with cross-validated AUROC between 0.79 and 0.80. The rankings of individual features changed with sample size and model type, and only age was consistently important.

## Try it

Find the latest demo release [here](https://github.com/s-ray9/sleep-cognition-ml/releases) and follow the instructions.

<img width="1017" height="905" alt="image" src="https://github.com/user-attachments/assets/e91b6d09-d51f-408d-b090-4b78ff30519a" />

https://github.com/user-attachments/assets/f6f938b7-0109-4abf-b44b-35f2fb838197

## Getting started

### Prerequisites

- Git
- Python (v3.13+)
- uv
- Kaggle API credentials

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/s-ray9/sleep-cognition-ml.git
   cd sleep-cognition-ml
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

### Usage

#### Data preparation

Download the training cohort of 79 positive and 1,013 negative patients:

```bash
uv run python download_patients.py --split training --true 79 --false 1013
```

Sort the training downloads into the directory structure the Challenge expects:

```bash
uv run python sort_downloads.py data/training
```

Download the holdout cohort of 5 positive and 5 negative patients:

```bash
uv run python download_patients.py --split holdout --true 5 --false 5
```

Sort the holdout downloads into the same structure:

```bash
uv run python sort_downloads.py data/holdout
```

#### Model training

Train the model, a 200-tree random forest with balanced class weights:

```bash
uv run python train_model.py -d data/training -m model -v
```

This command trains a single model through the official Challenge pipeline. The analysis scripts below produce the cross-validated and multi-seed results. Each of them reads its settings from constants at the top of the file, so none of them take command-line arguments.

#### Analysis

Verify that no patient contributes sessions to more than one split:

```bash
uv run python check_patient_overlap.py
```

Compute five-fold cross-validated AUROC on all 75 features:

```bash
uv run python evaluate_methodology.py
```

Compare cross-validated AUROC across feature subset sizes:

```bash
uv run python compare_feature_subsets.py
```

Compare performance and feature importance across the three model architectures:

```bash
uv run python compare_models.py
```

Compute feature importance from a single trained model:

```bash
uv run python rank_features.py
```

Average feature importance across 20 random seeds:

```bash
uv run python rank_features_multiseed.py
```

## Project structure

### Challenge code

#### Unmodified

| File                | Purpose                                                           | Arguments                                                     |
| ------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------- |
| `evaluate_model.py` | Scores predictions against labels                                 | `-d <labels_file> -o <predictions_file> -p <prevalence_file>` |
| `helper_code.py`    | Provides shared utilities used by the other files in this section | n/a                                                           |
| `run_model.py`      | Runs a trained model on new data                                  | `-d <data_folder> -m <model_folder> -o <output_folder> -v`    |
| `train_model.py`    | Trains a model                                                    | `-d <data_folder> -m <model_folder> -v`                       |

#### Modified

| File           | Purpose                                                                     | Arguments |
| -------------- | --------------------------------------------------------------------------- | --------- |
| `team_code.py` | Defines feature extraction, including heart rate variability, and the model | n/a       |

### Original code

| File                         | Purpose                                                                                      | Arguments                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `check_patient_overlap.py`   | Verifies that no patient contributes sessions to more than one split                         | n/a                                                                     |
| `compare_feature_subsets.py` | Computes cross-validated AUROC for each feature subset size                                  | n/a                                                                     |
| `compare_models.py`          | Compares cross-validated performance and feature importance across three model architectures | n/a                                                                     |
| `download_patients.py`       | Downloads patient data from Kaggle                                                           | `--split <training\|holdout> --true <n> --false <n> [--data-dir <dir>]` |
| `evaluate_methodology.py`    | Computes five-fold cross-validated AUROC on all 75 features                                  | n/a                                                                     |
| `experiment_utils.py`        | Provides shared feature definitions and dataset loading                                      | n/a                                                                     |
| `rank_features.py`           | Computes feature importance from a single trained model                                      | n/a                                                                     |
| `rank_features_multiseed.py` | Averages feature importance across 20 random seeds                                           | n/a                                                                     |
| `sort_downloads.py`          | Sorts downloaded files into the Challenge-standard directory structure                       | `<target_directory>`                                                    |

## Acknowledgements

This is independent research and not a Challenge submission. It builds on the example code and data of the [George B. Moody PhysioNet Challenge 2026](https://moody-challenge.physionet.org/2026).

### Data attribution

The underlying physiology data used in this project comes from the Human Sleep Project (version 2.0). If you use or adapt this code, please attribute the primary dataset:

> Sun, H., et al. (2023). The Human Sleep Project (version 2.0). _Brain Data Science Platform_. https://doi.org/10.60508/qjbv-hg78.

<!--
For the complete formal reference list (including the official 2026 Challenge overview and baseline CAISR algorithm papers), see [REPORT.md](REPORT.md).
-->

## License

Distributed under the MIT License. See [LICENSE] for more information.

Code retained from the Challenge repository (see [Challenge code](#challenge-code)) remains under its original BSD 3-Clause License. See [PHYSIONET-CHALLENGE-LICENSE](THIRD-PARTY-LICENSES/PHYSIONET-CHALLENGE-LICENSE) for more information.

[banner]: assets/banner.png
[license]: LICENSE
<!--
[demo]: https://huggingface.co/spaces/s-ray9/sleep-cognition-ml
[demo-badge]: https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue
-->
[latest-release]: https://github.com/s-ray9/sleep-cognition-ml/releases
[latest-release-badge]: https://img.shields.io/github/v/release/s-ray9/sleep-cognition-ml
[license-badge]: https://img.shields.io/github/license/s-ray9/sleep-cognition-ml?color=blue&label=License&logo=opensourceinitiative&logoColor=white
[python-badge]: https://img.shields.io/badge/Python-3.13-ffd43b?logo=python&logoColor=white&labelColor=3776ab
[numpy-badge]: https://img.shields.io/badge/NumPy-4d77cf?logo=numpy&logoColor=white
[pandas-badge]: https://img.shields.io/badge/pandas-15045c?logo=pandas&logoColor=white
[scipy-badge]: https://img.shields.io/badge/SciPy-3776ab?logo=scipy&logoColor=whit
[scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-f7931e?logo=scikit-learn&logoColor=white
[uv-badge]: https://img.shields.io/badge/uv-de5fe9?logo=uv&logoColor=white
