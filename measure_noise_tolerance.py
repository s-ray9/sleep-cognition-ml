import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from experiment_utils import TOP_10_FEATURES, build_feature_names, extract_dataset

TARGET_FEATURE = "limb_movement_index_auto"
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
DATA_FOLDER = "data/training"
N_FOLDS = 5


def main():
    print("Extracting features...")
    features, labels = extract_dataset(DATA_FOLDER)
    feature_names = build_feature_names()
    print(f"Extracted {len(features)} records ({labels.sum()} True, {(~labels).sum()} False).\n")

    indices = [feature_names.index(name) for name in TOP_10_FEATURES]
    target_idx_in_subset = TOP_10_FEATURES.index(TARGET_FEATURE)
    subset_features = features[:, indices]

    target_column = subset_features[:, target_idx_in_subset]
    valid_values = target_column[~np.isnan(target_column)]
    target_std = np.std(valid_values)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    print(f"{'noise (x std)':15s} {'AUROC':10s}")
    for noise_level in NOISE_LEVELS:
        noisy_features = subset_features.copy()
        rng = np.random.default_rng(42)
        noise = rng.normal(0, target_std * noise_level, size=noisy_features.shape[0])
        noisy_features[:, target_idx_in_subset] = noisy_features[:, target_idx_in_subset] + noise

        scores = cross_val_score(model, noisy_features, labels, cv=cv, scoring="roc_auc")
        print(f"{noise_level:<15} {scores.mean():.3f} +/- {scores.std():.3f}")


if __name__ == "__main__":
    main()
