import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEMOGRAPHICS = Path("data/training/demographics.csv")
OUTPUT = Path("age_bmi_model.json")


def main():
    frame = pd.read_csv(DEMOGRAPHICS)
    frame["label"] = frame["Cognitive_Impairment"].astype(str).map({"True": 1, "False": 0})
    frame["age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame["bmi"] = pd.to_numeric(frame["BMI"], errors="coerce")
    frame = frame.dropna(subset=["label", "age"])

    bmi_median = float(frame["bmi"].median())
    frame["bmi"] = frame["bmi"].fillna(bmi_median)

    features = frame[["age", "bmi"]].to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=int)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, features, labels, cv=cv, scoring="roc_auc")
    model.fit(features, labels)

    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]

    payload = {
        "age_mean": float(scaler.mean_[0]),
        "age_scale": float(scaler.scale_[0]),
        "bmi_mean": float(scaler.mean_[1]),
        "bmi_scale": float(scaler.scale_[1]),
        "bmi_median": bmi_median,
        "age_coefficient": float(classifier.coef_[0][0]),
        "bmi_coefficient": float(classifier.coef_[0][1]),
        "intercept": float(classifier.intercept_[0]),
        "age_range": [float(frame["age"].quantile(0.01)), float(frame["age"].quantile(0.99))],
        "bmi_range": [float(frame["bmi"].quantile(0.01)), float(frame["bmi"].quantile(0.99))],
        "patients": int(len(frame)),
        "positives": int(labels.sum()),
        "cv_auroc_mean": float(scores.mean()),
        "cv_auroc_std": float(scores.std()),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
