import joblib

from experiment_utils import build_feature_names


def main():
    model_data = joblib.load("model/model.sav")
    classifier = model_data["model"].named_steps["classifier"]
    importances = classifier.feature_importances_
    feature_names = build_feature_names()

    if len(feature_names) != len(importances):
        raise ValueError(f"Expected {len(importances)} feature names, built {len(feature_names)}")

    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    for name, score in ranked:
        print(f"{name:30s} {score:.4f}")


if __name__ == "__main__":
    main()
