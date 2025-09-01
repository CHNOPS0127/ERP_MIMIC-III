#!/usr/bin/env python3


import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# -----------------------
# Small helpers
# -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def is_encounter_timeseries(filename: str) -> bool:
    return filename.endswith(".csv") and "_encounter_timeseries" in filename

def safe_to_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan

# -----------------------
# (1) LISTFILE
# -----------------------
def build_listfile(raw_root: str, preproc_root: str) -> str:
    """Create listfile.csv from encounter files + all_stays.csv."""
    dynamic_dir = os.path.join(preproc_root, "dynamic")
    stays_path = os.path.join(raw_root, "all_stays.csv")
    out_path = os.path.join(preproc_root, "listfile.csv")

    stays = pd.read_csv(stays_path)
    stays = stays[["ICUSTAY_ID", "LOS"]].copy()
    stays["ICUSTAY_ID"] = stays["ICUSTAY_ID"].apply(safe_to_int)

    records = []
    for fn in os.listdir(dynamic_dir):
        if not is_encounter_timeseries(fn):
            continue
        fpath = os.path.join(dynamic_dir, fn)
        df = pd.read_csv(fpath, nrows=1)  # only need first row
        if "ICUSTAY_ID" not in df.columns:
            print(f" Skipped {fn}: Missing ICUSTAY_ID")
            continue

        icustay_id = safe_to_int(df["ICUSTAY_ID"].iloc[0])
        if np.isnan(icustay_id):
            print(f" Skipped {fn}: invalid ICUSTAY_ID")
            continue

        match = stays[stays["ICUSTAY_ID"] == icustay_id]
        if match.empty:
            print(f" ICUSTAY_ID {icustay_id} not found in all_stays.csv")
            continue

        los_hours = int(round(float(match["LOS"].values[0]) * 24))
        for hour in range(1, los_hours + 1):
            records.append({
                "stay": fn,
                "ICUSTAY_ID": icustay_id,
                "period_length": hour,
                "y_true": los_hours - hour
            })

    listfile = pd.DataFrame(records)
    ensure_dir(preproc_root)
    listfile.to_csv(out_path, index=False)
    print(f"Saved listfile: {out_path}")
    return out_path

# -----------------------
# (2) Dynamic imputation
# -----------------------
GCS_COLS = [
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response"
]

VITAL_COLS = [
    "Bilirubin", "Blood Urea Nitrogen", "Capillary refill rate", "Creatinine",
    "Diastolic blood pressure", "Fraction inspired oxygen", "Glucose", "Heart Rate",
    "Hematocrit", "Hemoglobin", "Lactate", "Mean blood pressure",
    "Oxygen saturation", "Platelet Count", "Potassium",
    "Respiratory rate", "Sodium", "Systolic blood pressure", "Temperature",
    "White Blood Cell Count", "pH", "Urine Output",
    "Glascow coma scale total", "Height", "Weight"
]

ICU_MEDIANS = {
    "Bilirubin": 0.7, "Blood Urea Nitrogen": 23, "Capillary refill rate": 0, "Creatinine": 1,
    "Diastolic blood pressure": 59, "Fraction inspired oxygen": 0.4, "Glucose": 126,
    "Heart Rate": 85, "Hematocrit": 29.7, "Hemoglobin": 10, "Lactate": 2.9,
    "Mean blood pressure": 77, "Oxygen saturation": 98,
    "Platelet Count": 205, "Potassium": 4, "Respiratory rate": 19, "Sodium": 139,
    "Systolic blood pressure": 119, "Temperature": 37, "White Blood Cell Count": 10.3,
    "pH": 7.38, "Urine Output": 366, "Glascow coma scale total": 13,
    "Height": 170.09, "Weight": 81
}

GCS_DEFAULTS = {
    "Glascow coma scale eye opening": "4 Spontaneously",
    "Glascow coma scale motor response": "6 Obeys Commands",
    "Glascow coma scale verbal response": "1.0 ET/Trach"
}

UNIFIED_FEATURES = [
    "ICUSTAY_ID", "HOUR",
    "Bilirubin", "Blood Urea Nitrogen", "Capillary refill rate", "Creatinine",
    "Diastolic blood pressure", "Fraction inspired oxygen", "Glucose", "Heart Rate",
    "Hematocrit", "Hemoglobin", "Lactate", "Mean blood pressure",
    "Oxygen saturation", "Platelet Count", "Potassium",
    "Respiratory rate", "Sodium", "Systolic blood pressure", "Temperature",
    "White Blood Cell Count", "pH", "Urine Output",
    "Glascow coma scale total", "Height", "Weight",
    "Glascow coma scale eye opening", "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
]

def combine_dynamic(preproc_root: str) -> str:
    """Concatenate *_encounter_timeseries*.csv into dynamic_data.csv (aligned to UNIFIED_FEATURES)."""
    dynamic_dir = os.path.join(preproc_root, "dynamic")
    out_path = os.path.join(preproc_root, "dynamic_data.csv")

    frames = []
    files = [f for f in os.listdir(dynamic_dir) if is_encounter_timeseries(f)]
    for f in tqdm(files, desc="Combining dynamic files (aligned)"):
        fp = os.path.join(dynamic_dir, f)
        df = pd.read_csv(fp)

        missing = [c for c in UNIFIED_FEATURES if c not in df.columns]
        for c in missing:
            df[c] = np.nan

        df = df[UNIFIED_FEATURES].copy()
        df["HOUR"] = pd.to_numeric(df["HOUR"], errors="coerce")
        df["ICUSTAY_ID"] = pd.to_numeric(df["ICUSTAY_ID"], errors="coerce").astype("Int64")
        frames.append(df)

    if not frames:
        pd.DataFrame(columns=UNIFIED_FEATURES).to_csv(out_path, index=False)
        return out_path

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved dynamic_data (aligned): {out_path}")
    return out_path

def impute_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill within ICUSTAY_ID, then fill ICU medians; GCS label-encode with defaults."""
    df = df.sort_values(["ICUSTAY_ID", "HOUR"]).copy()

    present_gcs = [c for c in GCS_COLS if c in df.columns]
    for col in present_gcs:
        df[f"{col}_missing_flag"] = df[col].isna().astype(int)
    if present_gcs:
        df[present_gcs] = df[present_gcs].fillna({k: v for k, v in GCS_DEFAULTS.items() if k in present_gcs})
        enc = OrdinalEncoder()
        df[present_gcs] = enc.fit_transform(df[present_gcs]).astype("float32")

    present_vitals = [c for c in VITAL_COLS if c in df.columns]
    for col in present_vitals:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        flag_col = f"{col}_missing_flag"
        if flag_col not in df.columns:
            df[flag_col] = df[col].isna().astype(int)
        median_val = ICU_MEDIANS.get(col, np.nan)
        df[col] = df.groupby("ICUSTAY_ID")[col].transform(lambda x: x.ffill().fillna(median_val))

    return df

# -----------------------
# (3) Static cleaning
# -----------------------
def clean_static(static_csv: str, preproc_root: str) -> str:
    """Clean + encode static data; keep LOS and MORTALITY."""
    df = pd.read_csv(static_csv)
    df = df.dropna(subset=["ICUSTAY_ID", "LOS"])
    df = df[df["LOS"] >= 4 / 24]

    df["MARITAL_STATUS"] = df["MARITAL_STATUS"].fillna("UNKNOWN")
    med_age = df["AGE"].median()
    df.loc[df["AGE"] > 100, "AGE"] = med_age
    for col in ["ADMISSION_TYPE", "ADMISSION_LOCATION"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    drop_cols = [
        "LAST_CAREUNIT", "DBSOURCE", "INTIME", "OUTTIME", "ADMITTIME", "DISCHTIME",
        "DEATHTIME", "DOB", "DOD", "MORTALITY_INUNIT", "MORTALITY_INHOSPITAL",
        "DIAGNOSIS", "SUBJECT_ID", "HADM_ID"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    desired_diagnoses = [
        "Encountr palliative care", "Acute respiratry failure", "Septicemia NOS", "Severe sepsis",
        "Cardiac arrest", "Septic shock", "Do not resusctate status", "Acute necrosis of liver",
        "Cardiogenic shock", "Acute kidney failure NOS", "Ac kidny fail, tubr necr", "Acidosis",
        "Intracerebral hemorrhage", "Pneumonia, organism NOS", "Food/vomit pneumonitis", "Cerebral edema",
        "Coagulat defect NEC/NOS", "Hyperosmolality", "Crnry athrscl natve vssl", "CHF NOS",
        "Obstructive sleep apnea", "Pure hypercholesterolem", "Intermed coronary synd", "Angina pectoris NEC/NOS"
    ]
    keep_dia = ["Dia_" + d for d in desired_diagnoses]
    existing_dia = [c for c in keep_dia if c in df.columns]

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dummies = pd.get_dummies(df[cat_cols], drop_first=True).astype(int)
    num_cols = df.drop(columns=cat_cols).select_dtypes(include=[np.number])

    encoded = pd.concat([num_cols, dummies], axis=1)
    non_dia = [c for c in encoded.columns if not c.startswith("Dia_")]
    encoded = encoded[non_dia + existing_dia]

    if "AGE" in encoded.columns:
        scaler = StandardScaler()
        encoded["AGE"] = scaler.fit_transform(encoded[["AGE"]])

    out = os.path.join(preproc_root, "static_data_encoded.csv")
    encoded.to_csv(out, index=False)
    print(f"Saved static_data_encoded: {out}")
    return out

def run_checks(df_static: pd.DataFrame, df_dyn: pd.DataFrame, preproc_root: str) -> None:
    """Write quick missing/dtype report."""
    def report(df, name):
        missing = df.isna().sum()
        miss = missing[missing > 0]
        nonnum = df.columns[~df.apply(pd.api.types.is_numeric_dtype)]
        lines = [f"--- {name} ---"]
        lines += ["Missing:", str(miss)] if not miss.empty else ["Missing: none"]
        lines += ["Non-numeric cols:", str(nonnum.tolist())] if len(nonnum) > 0 else ["All numeric."]
        return "\n".join(lines)

    txt = []
    txt.append(report(df_static, "static_data_encoded"))
    txt.append(report(df_dyn, "dynamic_data_imputed"))
    ensure_dir(os.path.join(preproc_root, "checks"))
    with open(os.path.join(preproc_root, "checks", "missing_dtype_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(txt))
    print("Saved checks/missing_dtype_report.txt")

# -----------------------
# Main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Final preprocessing pipeline.")
    p.add_argument("raw_root", type=str, help="Root from events validation (has numeric subject folders + all_stays.csv).")
    return p.parse_args()

def main():
    args = parse_args()
    raw_root = args.raw_root
    preproc_root = os.path.join(raw_root, "preprocessed")
    ensure_dir(preproc_root)

    # (1) listfile
    listfile_csv = build_listfile(raw_root, preproc_root)

    # (2) combine dynamic + impute/encode
    dynamic_csv = combine_dynamic(preproc_root)
    dyn = pd.read_csv(dynamic_csv)
    dyn_imputed = impute_dynamic_features(dyn)
    dyn_imputed_path = os.path.join(preproc_root, "dynamic_data_imputed.csv")
    dyn_imputed.to_csv(dyn_imputed_path, index=False)
    print(f"Saved dynamic_data_imputed: {dyn_imputed_path}")

    # (3) clean static (expects static_data.csv already created earlier)
    static_in = os.path.join(preproc_root, "static_data.csv")
    static_encoded_csv = clean_static(static_in, preproc_root)

    # checks
    run_checks(pd.read_csv(static_encoded_csv), dyn_imputed, preproc_root)

    print("\nFinal preprocessing complete.")
    print("Outputs in:", preproc_root)
    print(" - listfile.csv")
    print(" - dynamic_data.csv")
    print(" - dynamic_data_imputed.csv")
    print(" - static_data_encoded.csv")
    print(" - checks/missing_dtype_report.txt")

if __name__ == "__main__":
    main()

