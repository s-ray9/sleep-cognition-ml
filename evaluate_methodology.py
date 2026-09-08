from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from experiment_utils import extract_dataset

DATA_FOLDER = "data/training"
N_FOLDS = 5


def main():
    print("Extracting features...")
    features, labels = extract_dataset(DATA_FOLDER)
    print(f"Extracted {len(features)} records ({labels.sum()} True, {(~labels).sum()} False).\n")

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
    scores = cross_val_score(model, features, labels, cv=cv, scoring="roc_auc")

    print(f"AUROC per fold: {[round(s, 3) for s in scores]}")
    print(f"Mean AUROC: {scores.mean():.3f} +/- {scores.std():.3f}")


if __name__ == "__main__":
    main()
