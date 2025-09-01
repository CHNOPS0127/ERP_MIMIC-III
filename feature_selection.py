#!/usr/bin/env python3


import os
import argparse
from typing import Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr, pointbiserialr


# ---------- Utilities ----------

def is_subject_folder(name: str) -> bool:
    """True if folder name is numeric (SUBJECT_ID)."""
    return name.isdigit()


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ---------- Step 1: Filter events by ITEMIDs ----------

def load_variable_map(path: str) -> pd.DataFrame:
    """Load updated_variable_selection_I.csv."""
    vm = pd.read_csv(path)
    # Expect columns: ITEMID, VARIABLE, LABEL
    return vm


def filter_events_for_subject(
    subject_dir_in: str,
    subject_dir_out: str,
    selected_itemids: Set,
    variable_map: pd.DataFrame,
) -> None:
    """Filter one subject's events by ITEMID and save to /preprocessed/<SUBJECT_ID>/filtered_events.csv."""
    events_path = os.path.join(subject_dir_in, "events.csv")
    if not os.path.isfile(events_path):
        return

    df = pd.read_csv(events_path)
    filtered = df[df["ITEMID"].isin(selected_itemids)].copy()

    # Join variable names/labels (left join as in original)
    filtered = filtered.merge(
        variable_map[["ITEMID", "VARIABLE", "LABEL"]],
        on="ITEMID",
        how="left"
    )

    ensure_dir(subject_dir_out)
    out_path = os.path.join(subject_dir_out, "filtered_events.csv")
    filtered.to_csv(out_path, index=False)


def run_event_filtering(raw_root: str, preproc_root: str, variable_map_path: str) -> None:
    """Filter all subjects' events using the variable map."""
    variable_map = load_variable_map(variable_map_path)
    selected_itemids = set(variable_map["ITEMID"])

    subjects = [d for d in os.listdir(raw_root) if is_subject_folder(d)]
    for sid in tqdm(subjects, desc="Filtering events by ITEMIDs"):
        in_dir = os.path.join(raw_root, sid)
        out_dir = os.path.join(preproc_root, sid)
        filter_events_for_subject(in_dir, out_dir, selected_itemids, variable_map)


# ---------- Step 2: TF-IDF features + merge with stays ----------

def build_tfidf_static_combined(raw_root: str, mimic_root: str, preproc_root: str) -> pd.DataFrame:
    """Compute TF-IDF features from NOTEEVENTS, select top terms, aggregate per subject, merge with all_stays."""
    # Load data (same columns and merges as your original)
    noteevents_path = os.path.join(mimic_root, "NOTEEVENTS.csv")
    all_stays_path = os.path.join(raw_root, "all_stays.csv")

    noteevents = pd.read_csv(noteevents_path, engine="python")
    stays = pd.read_csv(all_stays_path)

    # Keep needed columns from stays
    stays_subset = stays[["SUBJECT_ID", "LOS", "MORTALITY"]]

    # Consistent dtypes
    noteevents["SUBJECT_ID"] = noteevents["SUBJECT_ID"].astype(int)
    stays_subset["SUBJECT_ID"] = stays_subset["SUBJECT_ID"].astype(int)

    # Drop duplicate subject rows to avoid many-to-many merge (same as your code)
    stays_dedup = stays_subset.drop_duplicates(subset="SUBJECT_ID")

    # Merge by SUBJECT_ID
    noteevents_merged = pd.merge(
        noteevents,
        stays_dedup,
        on="SUBJECT_ID",
        how="inner"
    )

    # Drop missing used fields
    noteevents_merged = noteevents_merged.dropna(subset=["TEXT", "LOS", "MORTALITY", "SUBJECT_ID"])

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=300, stop_words="english", lowercase=True)
    X_tfidf = tfidf.fit_transform(noteevents_merged["TEXT"])
    feature_names = tfidf.get_feature_names_out()
    tfidf_matrix = X_tfidf.toarray()

    # Targets
    y_los = noteevents_merged["LOS"].values
    y_mortality = noteevents_merged["MORTALITY"].astype(int).values

    # Spearman correlation with LOS (top 20 by |corr|)
    spearman_corrs = []
    for i in range(tfidf_matrix.shape[1]):
        corr, _ = spearmanr(tfidf_matrix[:, i], y_los)
        spearman_corrs.append((feature_names[i], corr))
    top20_spearman = sorted(spearman_corrs, key=lambda x: abs(x[1]), reverse=True)[:20]

    # Point-biserial correlation with MORTALITY (top 20 by |corr|)
    pb_corrs = []
    for i in range(tfidf_matrix.shape[1]):
        try:
            pb_corr, _ = pointbiserialr(tfidf_matrix[:, i], y_mortality)
        except Exception:
            pb_corr = 0.0
        pb_corrs.append((feature_names[i], pb_corr))
    top20_pb = sorted(pb_corrs, key=lambda x: abs(x[1]), reverse=True)[:20]

    # Selected features (union)
    selected_features = list(set([f for f, _ in top20_spearman] + [f for f, _ in top20_pb]))
    print(f"\nTotal selected TF-IDF features: {len(selected_features)}")

    # Build reduced TF-IDF DataFrame (same steps as original)
    tfidf_df_full = pd.DataFrame(tfidf_matrix, columns=feature_names)
    tfidf_selected = tfidf_df_full[selected_features]
    # In original code this referenced df['SUBJECT_ID']; we use the merged notes' SUBJECT_IDs
    tfidf_selected.insert(0, "SUBJECT_ID", noteevents_merged["SUBJECT_ID"].values)

    # Aggregate per SUBJECT_ID (mean)
    tfidf_subject_level = tfidf_selected.groupby("SUBJECT_ID").mean().reset_index()

    # Merge with all_stays (inner join as in original)
    all_stays = pd.read_csv(all_stays_path)
    merged = pd.merge(all_stays, tfidf_subject_level, on="SUBJECT_ID", how="inner")

    # Save
    out_path = os.path.join(preproc_root, "tfidf_static_combined.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return merged


# ---------- Step 3: Build final static dataset ----------

def build_static_dataset(raw_root: str, mimic_root: str, preproc_root: str, merged: pd.DataFrame) -> None:
    """Merge selected diagnoses and ADMISSIONS fields into the combined static dataset."""
    diagnoses_path = os.path.join(raw_root, "all_diagnoses.csv")
    admissions_path = os.path.join(mimic_root, "ADMISSIONS.csv")

    diagnoses = pd.read_csv(diagnoses_path)
    admission_df = pd.read_csv(admissions_path)

    # Diagnosis codes to keep (unchanged list)
    diagnosis_labels = [
        '4019', '4280', '41401', '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720',
        '2859', '2449', '486', '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070',
        '40390', '3051', '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867',
        '4241', '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749',
        '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111',
        'V1251', '30000', '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652',
        '99811', '431', '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832',
        'E8788', '00845', '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
        '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254',
        '42823', '28529', 'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859',
        'V667', 'E8497', '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770',
        'V5865', '99662', '28860', '36201', '56210'
    ]

    # Standardize id columns (same as your code)
    for df in (diagnoses, admission_df, merged):
        df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(str)
    diagnoses["ICD9_CODE"] = diagnoses["ICD9_CODE"].astype(str).str.strip()
    diagnoses["HADM_ID"] = diagnoses["HADM_ID"].astype(str)
    admission_df["HADM_ID"] = admission_df["HADM_ID"].astype(str)
    merged["HADM_ID"] = merged["HADM_ID"].astype(str)

    # Filter selected diagnosis codes
    filtered_diag = diagnoses[diagnoses["ICD9_CODE"].isin(diagnosis_labels)]

    # Wide matrix of diagnosis indicators per SUBJECT_ID/HADM_ID
    diagnosis_wide = pd.crosstab(
        index=[filtered_diag["SUBJECT_ID"], filtered_diag["HADM_ID"]],
        columns=filtered_diag["SHORT_TITLE"]
    ).astype(int).reset_index()

    # Prefix diagnosis columns
    diagnosis_cols = diagnosis_wide.columns.difference(["SUBJECT_ID", "HADM_ID"])
    diagnosis_wide.rename(columns={col: f"Dia_{col}" for col in diagnosis_cols}, inplace=True)

    # Merge with TF-IDF + stays block
    static_data = pd.merge(
        merged,
        diagnosis_wide,
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner"
    )

    # Merge ADMISSIONS fields
    static_data_final = pd.merge(
        static_data,
        admission_df[["SUBJECT_ID", "HADM_ID", "MARITAL_STATUS", "INSURANCE", "ADMISSION_TYPE", "ADMISSION_LOCATION"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner"
    )

    # Save
    out_path = os.path.join(preproc_root, "static_data.csv")
    static_data_final.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess data (coherent with events validation).")
    parser.add_argument("raw_root", type=str, help="Root folder from events validation (has subject folders + all_*.csv).")
    parser.add_argument("--mimic_root", type=str, default=None, help="Folder with NOTEEVENTS.csv and ADMISSIONS.csv. Defaults to raw_root.")
    parser.add_argument("--variable_map", type=str, required=True, help="Path to updated_variable_selection_I.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root
    mimic_root = args.mimic_root or raw_root
    preproc_root = os.path.join(raw_root, "preprocessed")
    
    ensure_dir(preproc_root)

    # 1) Subject-level events filtering
    run_event_filtering(raw_root, preproc_root, args.variable_map)

    # 2) TF-IDF + stays merge
    merged = build_tfidf_static_combined(raw_root, mimic_root, preproc_root)

    # 3) Final static dataset
    build_static_dataset(raw_root, mimic_root, preproc_root, merged)

    print("\nDone. Outputs written to:", preproc_root)


if __name__ == "__main__":
    main()

