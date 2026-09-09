from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from experiment_utils import TOP_10_FEATURES, TOP_20_FEATURES, build_feature_names, extract_dataset

DATA_FOLDER = "data/training"
N_FOLDS = 5

FEATURE_SUBSETS = {
    "top_2": ["age", "bmi"],
    "top_10": TOP_10_FEATURES,
    "top_20": TOP_20_FEATURES,
    "all": None,
}


def evaluate_subset(features, labels, feature_names, subset_names):
    if subset_names is None:
        subset_features = features
    else:
        indices = [feature_names.index(name) for name in subset_names]
        subset_features = features[:, indices]

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
    scores = cross_val_score(model, subset_features, labels, cv=cv, scoring="roc_auc")
    return scores


def main():
    print("Extracting features...")
    features, labels = extract_dataset(DATA_FOLDER)
    feature_names = build_feature_names()
    print(f"Extracted {len(features)} records ({labels.sum()} True, {(~labels).sum()} False).\n")

    for subset_name, subset_features_list in FEATURE_SUBSETS.items():
        scores = evaluate_subset(features, labels, feature_names, subset_features_list)
        n_features = len(subset_features_list) if subset_features_list else features.shape[1]
        print(
            f"{subset_name:10s} ({n_features:2d} features): "
            f"AUROC = {scores.mean():.3f} +/- {scores.std():.3f}"
        )


if __name__ == "__main__":
    main()
