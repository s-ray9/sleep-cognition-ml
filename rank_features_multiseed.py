import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from experiment_utils import build_feature_names, extract_dataset

N_SEEDS = 20
DATA_FOLDER = "data/training"


def main():
    print("Extracting features...")
    features, labels = extract_dataset(DATA_FOLDER)
    print(f"Extracted {len(features)} records ({labels.sum()} True, {(~labels).sum()} False).\n")

    feature_names = build_feature_names()
    all_importances = np.zeros((N_SEEDS, len(feature_names)))

    for seed in range(N_SEEDS):
        rf = RandomForestClassifier(n_estimators=12, max_leaf_nodes=34, random_state=seed)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", rf),
            ]
        )
        model.fit(features, labels)
        all_importances[seed] = model.named_steps["classifier"].feature_importances_
        print(f"Seed {seed + 1}/{N_SEEDS} done.")

    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    ranked = sorted(
        zip(feature_names, mean_importance, std_importance),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nfeature                        mean      std")
    for name, mean_val, std_val in ranked:
        print(f"{name:30s} {mean_val:.4f}   {std_val:.4f}")


if __name__ == "__main__":
    main()
