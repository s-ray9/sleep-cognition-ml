import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_utils import TOP_10_FEATURES, build_feature_names, extract_dataset

DATA_FOLDER = "data/training"
N_FOLDS = 5

MODELS = {
    "logistic_regression": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            ),
        ]
    ),
    "random_forest": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
            ),
        ]
    ),
    "gradient_boosting": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
            ),
        ]
    ),
}


def get_logistic_regression_importances(model, feature_names):
    coefs = model.named_steps["classifier"].coef_[0]
    return sorted(zip(feature_names, np.abs(coefs)), key=lambda x: x[1], reverse=True)


def main():
    print("Extracting features...")
    features, labels = extract_dataset(DATA_FOLDER)
    feature_names = build_feature_names()
    print(f"Extracted {len(features)} records ({labels.sum()} True, {(~labels).sum()} False).\n")

    indices = [feature_names.index(name) for name in TOP_10_FEATURES]
    subset_features = features[:, indices]

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    print("Cross-validated AUROC on top-10 features, by model type:\n")
    for name, model in MODELS.items():
        scores = cross_val_score(model, subset_features, labels, cv=cv, scoring="roc_auc")
        print(f"{name:20s} AUROC = {scores.mean():.3f} +/- {scores.std():.3f}")

    print("\nFitting logistic regression on full data to inspect coefficients...")
    lr_model = MODELS["logistic_regression"]
    lr_model.fit(subset_features, labels)
    ranked = get_logistic_regression_importances(lr_model, TOP_10_FEATURES)

    print("\nLogistic regression feature weights (|coefficient|, standardized):")
    for name, weight in ranked:
        print(f"{name:30s} {weight:.4f}")


if __name__ == "__main__":
    main()
