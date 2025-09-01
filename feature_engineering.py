#!/usr/bin/env python3
"""
Feature engineering for classical models (LogReg, RandomForest).

"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

# ----------------------------
# Config (feature lists)
# ----------------------------
GCS_COLS = [
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
]

VITAL_COLS = [
    "Bilirubin", "Blood Urea Nitrogen", "Capillary refill rate", "Creatinine",
    "Diastolic blood pressure", "Fraction inspired oxygen", "Glucose", "Heart Rate",
    "Hematocrit", "Hemoglobin", "Lactate", "Mean blood pressure",
    "Oxygen saturation", "Platelet Count", "Potassium",
    "Respiratory rate", "Sodium", "Systolic blood pressure", "Temperature",
    "White Blood Cell Count", "pH", "Urine Output",
    "Glascow coma scale total", "Height", "Weight",
]

ICU_MEDIANS = {
    "Bilirubin": 0.7, "Blood Urea Nitrogen": 23, "Capillary refill rate": 0, "Creatinine": 1,
    "Diastolic blood pressure": 59, "Fraction inspired oxygen": 0.4, "Glucose": 126,
    "Heart Rate": 85, "Hematocrit": 29.7, "Hemoglobin": 10, "Lactate": 2.9,
    "Mean blood pressure": 77, "Oxygen saturation": 98,
    "Platelet Count": 205, "Potassium": 4, "Respiratory rate": 19, "Sodium": 139,
    "Systolic blood pressure": 119, "Temperature": 37, "White Blood Cell Count": 10.3,
    "pH": 7.38, "Urine Output": 366, "Glascow coma scale total": 13, "Height": 170.09, "Weight": 81,
}

# diagnoses to keep (will select columns named Dia_<label> if present)
DESIRED_DIAGNOSES = [
    "Encountr palliative care", "Acute respiratry failure", "Septicemia NOS", "Severe sepsis",
    "Cardiac arrest", "Septic shock", "Do not resusctate status", "Acute necrosis of liver",
    "Cardiogenic shock", "Acute kidney failure NOS", "Ac kidny fail, tubr necr", "Acidosis",
    "Intracerebral hemorrhage", "Pneumonia, organism NOS", "Food/vomit pneumonitis", "Cerebral edema",
    "Coagulat defect NEC/NOS", "Hyperosmolality", "Crnry athrscl natve vssl", "CHF NOS",
    "Obstructive sleep apnea", "Pure hypercholesterolem", "Intermed coronary synd", "Angina pectoris NEC/NOS",
]

# time subwindows (percent of observed time)
SUBPERIODS = {
    "first10": ("percent_time", 0.0, 0.10),
    "first25": ("percent_time", 0.0, 0.25),
}

# univariate functions to compute on each subwindow
FUNC_MAP = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "len": len,
    "auc": lambda x: np.trapz(x),
    "delta": lambda x: (x[-1] - x[0]) if len(x) > 1 else 0.0,
    "slope": lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0.0,
    "reversals": lambda x: np.sum(np.diff(np.sign(np.diff(x))) != 0) if len(x) > 2 else 0,
    "entropy": lambda x: scipy_entropy(np.histogram(x, bins='auto', density=True)[0] + 1e-10),
}


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def present_columns(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


# ----------------------------
# Imputation (dynamic vitals)
# ----------------------------
def impute_vitals(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill within ICUSTAY_ID then fill ICU medians.
    Assumes dynamic_df has columns: ICUSTAY_ID, HOUR, and some VITAL_COLS.
    """
    df = dynamic_df.sort_values(["ICUSTAY_ID", "HOUR"]).copy()
    vitals = present_columns(df, VITAL_COLS)

    for col in vitals:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_missing_flag"] = df[col].isna().astype(int)
        med = ICU_MEDIANS.get(col, np.nan)
        df[col] = df.groupby("ICUSTAY_ID")[col].transform(lambda x: x.ffill().fillna(med))

    return df


# ----------------------------
# Time-window helpers
# ----------------------------
def get_subwindow(df_group: pd.DataFrame, col: str, mode: str, start: float, end: float) -> np.ndarray:
    if mode != "percent_time":
        raise ValueError("Only 'percent_time' mode is supported.")
    max_hour = df_group["HOUR"].max()
    t0, t1 = max_hour * start, max_hour * end
    vals = df_group.loc[(df_group["HOUR"] >= t0) & (df_group["HOUR"] < t1), col]
    return pd.to_numeric(vals, errors="coerce").dropna().astype(float).values


def extract_features_per_icustay(df_group: pd.DataFrame) -> pd.Series:
    feats = {}
    icu_id = df_group["ICUSTAY_ID"].iloc[0]
    vitals = present_columns(df_group, VITAL_COLS)

    for col in vitals:
        col_vals = df_group[col]
        if col_vals.dropna().empty:
            # fill all subperiod features with NaN and mark missing_rate=1.0
            for pname in SUBPERIODS:
                for fname in FUNC_MAP:
                    feats[f"{col}_{pname}_{fname}"] = np.nan
            feats[f"{col}_missing_rate"] = 1.0
            continue

        # subperiod features
        for pname, (mode, s, e) in SUBPERIODS.items():
            seg = get_subwindow(df_group, col, mode, s, e)
            for fname, func in FUNC_MAP.items():
                try:
                    feats[f"{col}_{pname}_{fname}"] = func(seg) if len(seg) else np.nan
                except Exception:
                    feats[f"{col}_{pname}_{fname}"] = np.nan

        # overall missingness on the full trace we were given
        feats[f"{col}_missing_rate"] = col_vals.isna().mean()

    feats["ICUSTAY_ID"] = icu_id
    return pd.Series(feats)


def summarize_gcs_mode(df_group: pd.DataFrame) -> pd.Series:
    """
    If dynamic already has ordinal-encoded GCS values (as in your preprocessing),
    this will report the modal numeric code per GCS column.
    """
    out = {}
    gcs_cols = present_columns(df_group, GCS_COLS)
    for col in gcs_cols:
        vals = pd.to_numeric(df_group[col], errors="coerce").dropna()
        out[col] = vals.mode().iloc[0] if not vals.empty else np.nan
    return pd.Series(out)


def build_dynamic_features(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute vitals, then aggregate per ICUSTAY_ID into summary features
    + GCS modal values.
    """
    dyn_imp = impute_vitals(dynamic_df)
    perstay = (
        dyn_imp.groupby("ICUSTAY_ID")
        .apply(lambda g: pd.concat([extract_features_per_icustay(g), summarize_gcs_mode(g)]))
        .reset_index(drop=True)
    )
    return perstay


# ----------------------------
# Static processing
# ----------------------------
def clean_static(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep adult stays (>=4h), basic fixes, keep selected Dia_* columns if present.
    Assumes static_df already encoded (from your preprocessing).
    """
    df = static_df.copy()
    df = df.dropna(subset=["ICUSTAY_ID", "LOS"])
    df = df[df["LOS"] >= 4 / 24]

    # basic fills
    if "MARITAL_STATUS" in df.columns:
        df["MARITAL_STATUS"] = df["MARITAL_STATUS"].fillna("UNKNOWN")
    if "ADMISSION_TYPE" in df.columns:
        df["ADMISSION_TYPE"] = df["ADMISSION_TYPE"].fillna("UNKNOWN")

    if "AGE" in df.columns:
        med_age = df["AGE"].median()
        df.loc[df["AGE"] > 100, "AGE"] = med_age

    # drop pure metadata if present (safe)
    drop_cols = [
        "LAST_CAREUNIT", "DBSOURCE", "INTIME", "OUTTIME", "ADMITTIME", "DISCHTIME",
        "DEATHTIME", "DOB", "DOD", "MORTALITY_INUNIT", "MORTALITY_INHOSPITAL", "DIAGNOSIS"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # keep only Dia_ columns of interest (if they exist)
    keep_dia = ["Dia_" + d for d in DESIRED_DIAGNOSES]
    existing = [c for c in keep_dia if c in df.columns]
    non_dia = [c for c in df.columns if not c.startswith("Dia_")]
    df = df[non_dia + existing]

    return df


# ----------------------------
# LOS bucketizing
# ----------------------------
def add_los_bucket(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 1, 2, 3, 4, 5, 7, 14, np.inf]
    labels = list(range(len(bins) - 1))
    df = df.copy()
    df["LOS_BUCKET"] = pd.cut(df["LOS"], bins=bins, labels=labels, right=False).astype(int)
    return df


# ----------------------------
# Standardize (for LR)
# ----------------------------
def standardize_for_lr(df: pd.DataFrame, id_col="ICUSTAY_ID") -> pd.DataFrame:
    """
    Standardize numeric columns (except targets) for LR.
    Leaves categorical/dummy intact.
    """
    df = df.copy()
    # columns to skip from scaling
    skip = {id_col, "LOS", "MORTALITY", "LOS_BUCKET"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# ----------------------------
# Save helpers
# ----------------------------
def grouped_split_and_save(df: pd.DataFrame, out_dir: str, prefix: str, id_col="ICUSTAY_ID", test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[id_col]
    train_idx, test_idx = next(gss.split(df, None, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_fp = os.path.join(out_dir, f"{prefix}_train.csv")
    test_fp = os.path.join(out_dir, f"{prefix}_test.csv")

    train_df.to_csv(train_fp, index=False)
    test_df.to_csv(test_fp, index=False)
    print(f"Saved: {train_fp}")
    print(f"Saved: {test_fp}")


# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(root: str, dynamic_path: str = None, static_path: str = None,
                 out_dir: str = None, do_split: bool = True, test_size: float = 0.2, seed: int = 42):
    preproc_root = os.path.join(root, "preprocessed")
    dynamic_path = dynamic_path or os.path.join(preproc_root, "dynamic_data_imputed.csv")
    static_path = static_path or os.path.join(preproc_root, "static_data_encoded.csv")
    out_dir = out_dir or os.path.join(preproc_root, "features")
    ensure_dir(out_dir)

    # Load
    dyn = pd.read_csv(dynamic_path)
    sta = pd.read_csv(static_path)

    # Sanity columns
    if "ICUSTAY_ID" not in dyn.columns or "HOUR" not in dyn.columns:
        raise ValueError("dynamic_data_imputed.csv must contain ICUSTAY_ID and HOUR.")
    if "ICUSTAY_ID" not in sta.columns or "LOS" not in sta.columns:
        raise ValueError("static_data_encoded.csv must contain ICUSTAY_ID and LOS.")

    # Dynamic -> per-stay features
    dyn_feats = build_dynamic_features(dyn)

    # Static clean
    sta_clean = clean_static(sta)

    # Merge
    combined = pd.merge(dyn_feats, sta_clean, on="ICUSTAY_ID", how="inner")

    # LOS bucket
    combined = add_los_bucket(combined)

    # Keep Dia_* subset if exists already handled in clean_static

    # Save raw RF-ready
    rf_all_fp = os.path.join(out_dir, "features_all_rf.csv")
    combined.to_csv(rf_all_fp, index=False)
    print(f"Saved: {rf_all_fp}")

    # Standardize numeric for LR
    lr_df = standardize_for_lr(combined, id_col="ICUSTAY_ID")
    lr_all_fp = os.path.join(out_dir, "features_all_lr.csv")
    lr_df.to_csv(lr_all_fp, index=False)
    print(f"Saved: {lr_all_fp}")

    # Optional grouped split
    if do_split:
        grouped_split_and_save(combined, out_dir, prefix="features_rf", id_col="ICUSTAY_ID", test_size=test_size, seed=seed)
        grouped_split_and_save(lr_df, out_dir, prefix="features_lr", id_col="ICUSTAY_ID", test_size=test_size, seed=seed)


def parse_args():
    p = argparse.ArgumentParser(description="Build LR/RF feature sets from preprocessed data.")
    p.add_argument("root", type=str, help="Events-validation root (contains preprocessed/).")
    p.add_argument("--dynamic_path", type=str, default=None, help="Override path to dynamic_data_imputed.csv.")
    p.add_argument("--static_path", type=str, default=None, help="Override path to static_data_encoded.csv.")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <root>/preprocessed/features).")
    p.add_argument("--no_split", action="store_true", help="Do not write train/test splits.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test size for GroupShuffleSplit (default 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        root=args.root,
        dynamic_path=args.dynamic_path,
        static_path=args.static_path,
        out_dir=args.out_dir,
        do_split=not args.no_split,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

