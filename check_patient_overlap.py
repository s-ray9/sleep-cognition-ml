import csv
from collections import Counter

with open("data/training/demographics.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

patient_ids = [r["BDSPPatientID"] for r in rows]
counts = Counter(patient_ids)
repeated = {pid: c for pid, c in counts.items() if c > 1}

print(f"Total rows: {len(rows)}")
print(f"Unique patients: {len(counts)}")
print(f"Patients with multiple sessions in training set: {len(repeated)}")
