import argparse
import csv
import shutil
import subprocess
from pathlib import Path

DATASET = "physionet/physionetchallenge2026data"

DATA_DIR = Path("data")
MASTER_DEMOGRAPHICS = DATA_DIR / "training" / "demographics.csv"
MANIFEST_PATH = DATA_DIR / "used_patients.txt"


def ensure_master_demographics():
    if not MASTER_DEMOGRAPHICS.exists():
        print("demographics.csv not found — downloading it first...")
        (DATA_DIR / "training").mkdir(parents=True, exist_ok=True)
        download_file("demographics.csv", DATA_DIR / "training")


def download_file(remote_path, dest_dir):
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-f",
            remote_path,
            "-p",
            str(dest_dir),
            "--unzip",
        ]
    )


def load_used_patients():
    if not MANIFEST_PATH.exists():
        return set()
    return set(MANIFEST_PATH.read_text().splitlines())


def record_used_patients(patient_keys):
    with open(MANIFEST_PATH, "a") as f:
        for key in patient_keys:
            f.write(key + "\n")


def select_patients(rows, used, count_true, count_false):
    def patient_key(r):
        return f"{r['BidsFolder']}_ses-{r['SessionID']}"

    available_true = [
        r for r in rows if r["Cognitive_Impairment"] == "True" and patient_key(r) not in used
    ]
    available_false = [
        r for r in rows if r["Cognitive_Impairment"] == "False" and patient_key(r) not in used
    ]

    if len(available_true) < count_true:
        raise ValueError(
            f"Requested {count_true} True patients but only {len(available_true)} remain unused."
        )
    if len(available_false) < count_false:
        raise ValueError(
            f"Requested {count_false} False patients but only {len(available_false)} remain unused."
        )

    return available_true[:count_true] + available_false[:count_false]


def download_patients(selected, dest_dir):
    for i, row in enumerate(selected):
        site, sub, ses = row["SiteID"], row["BidsFolder"], row["SessionID"]
        print(f"\n[{i + 1}/{len(selected)}] {sub} session {ses} ({row['Cognitive_Impairment']})...")
        download_file(f"physiological_data/{site}/{sub}_ses-{ses}.edf", dest_dir)
        download_file(f"human_annotations/{site}/{sub}_ses-{ses}_expert_annotations.edf", dest_dir)
        download_file(
            f"algorithmic_annotations/{site}/{sub}_ses-{ses}_caisr_annotations.edf", dest_dir
        )
        record_used_patients([f"{sub}_ses-{ses}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["training", "holdout"], required=True)
    parser.add_argument("--true", type=int, default=20)
    parser.add_argument("--false", type=int, default=20)
    args = parser.parse_args()

    dest_dir = DATA_DIR / args.split
    dest_dir.mkdir(parents=True, exist_ok=True)

    ensure_master_demographics()

    with open(MASTER_DEMOGRAPHICS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    used = load_used_patients()
    selected = select_patients(rows, used, args.true, args.false)
    print(f"Selected {len(selected)} patients for {args.split}")

    download_patients(selected, dest_dir)

    [f"{r['BidsFolder']}_ses-{r['SessionID']}" for r in selected]

    if args.split != "training":
        shutil.copy(MASTER_DEMOGRAPHICS, dest_dir / "demographics.csv")
        print(f"Copied demographics.csv into {dest_dir}")

    print(f"\nDone downloading {args.split} set.")


if __name__ == "__main__":
    main()
