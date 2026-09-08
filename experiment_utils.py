import os

import numpy as np

from helper_code import (
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    DEMOGRAPHICS_FILE,
    HEADERS,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    find_patients,
    load_demographics,
    load_diagnoses,
    load_signal_data,
)
from team_code import (
    extract_algorithmic_annotations_features,
    extract_demographic_features,
    extract_physiological_features,
)

LEAD_TYPES = ["eeg", "eog", "chin", "leg", "ecg", "resp", "spo2"]
SIGNAL_STATS = ["std", "mav", "zcr", "rms", "activity", "mobility", "complexity"]

DEMOGRAPHIC_FEATURE_NAMES = [
    "age",
    "sex_female",
    "sex_male",
    "sex_other",
    "race_asian",
    "race_black",
    "race_other",
    "race_unavailable",
    "race_white",
    "bmi",
]

HRV_FEATURE_NAMES = ["hrv_mean_hr", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"]

ALGORITHMIC_FEATURE_NAMES = [
    "ahi_auto",
    "arousal_index_auto",
    "limb_movement_index_auto",
    "wake_pct_auto",
    "n1_pct_auto",
    "n2_pct_auto",
    "n3_pct_auto",
    "rem_pct_auto",
    "sleep_efficiency_auto",
    "prob_wake_auto",
    "prob_n3_auto",
    "prob_arousal_auto",
]

TOP_10_FEATURES = [
    "age",
    "bmi",
    "limb_movement_index_auto",
    "prob_arousal_auto",
    "hrv_rmssd",
    "chin_mav",
    "arousal_index_auto",
    "eeg_zcr",
    "chin_activity",
    "prob_wake_auto",
]

TOP_20_FEATURES = TOP_10_FEATURES + [
    "chin_rms",
    "resp_zcr",
    "chin_zcr",
    "eog_zcr",
    "ecg_complexity",
    "rem_pct_auto",
    "leg_activity",
    "resp_mobility",
    "chin_std",
    "ecg_zcr",
]


def build_feature_names():
    names = list(DEMOGRAPHIC_FEATURE_NAMES)
    for lead in LEAD_TYPES:
        for stat in SIGNAL_STATS:
            names.append(f"{lead}_{stat}")
    names.extend(HRV_FEATURE_NAMES)
    names.extend(ALGORITHMIC_FEATURE_NAMES)
    return names


def extract_dataset(data_folder):
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)

    features = []
    labels = []

    for record in patient_metadata_list:
        patient_id = record[HEADERS["bids_folder"]]
        site_id = record[HEADERS["site_id"]]
        session_id = record[HEADERS["session_id"]]

        physiological_data_file = os.path.join(
            data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf"
        )
        if not os.path.exists(physiological_data_file):
            continue

        try:
            patient_data = load_demographics(patient_data_file, patient_id, session_id)
            demographic_features = extract_demographic_features(patient_data)

            physiological_data, physiological_fs = load_signal_data(physiological_data_file)
            physiological_features = extract_physiological_features(
                physiological_data, physiological_fs
            )

            algorithmic_annotations_file = os.path.join(
                data_folder,
                ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                site_id,
                f"{patient_id}_ses-{session_id}_caisr_annotations.edf",
            )
            algorithmic_annotations, _ = load_signal_data(algorithmic_annotations_file)
            algorithmic_features = extract_algorithmic_annotations_features(algorithmic_annotations)

            label = load_diagnoses(patient_data_file, patient_id)

            if label in (0, 1):
                features.append(
                    np.hstack([demographic_features, physiological_features, algorithmic_features])
                )
                labels.append(label)
        except Exception as e:
            print(f"Skipping {patient_id}: {e}")
            continue

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=bool)
