#!/usr/bin/env python3
# ml_utils.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, cohen_kappa_score

# --------- constants (shared feature lists) ---------
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

DESIRED_DIAGNOSES = [
    "Encountr palliative care", "Acute respiratry failure", "Septicemia NOS", "Severe sepsis",
    "Cardiac arrest", "Septic shock", "Do not resusctate status", "Acute necrosis of liver",
    "Cardiogenic shock", "Acute kidney failure NOS", "Ac kidny fail, tubr necr", "Acidosis",
    "Intracerebral hemorrhage", "Pneumonia, organism NOS", "Food/vomit pneumonitis", "Cerebral edema",
    "Coagulat defect NEC/NOS", "Hyperosmolality", "Crnry athrscl natve vssl", "CHF NOS",
    "Obstructive sleep apnea", "Pure hypercholesterolem", "Intermed coronary synd", "Angina pectoris NEC/NOS",
]

SUBPERIODS = {
    "first10": ("percent_time", 0.0, 0.10),
    "first25": ("percent_time", 0.0, 0.25),
}

# --------- small helpers ---------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def present_columns(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]

def drop_bad_cols(df: pd.DataFrame) -> pd.DataFrame:
    # drop unnamed/empty columns from accidental CSV artifacts
    keep = ~(df.columns.str.startswith("Unnamed") | (df.columns == ""))
    return df.loc[:, keep]

# --------- bootstrap + CI ---------
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

def bootstrap_metrics(y_true, y_pred, n_bootstrap: int = 1000, seed: int | None = None):
    """
    Bootstrap metrics (accuracy, linear kappa, quadratic kappa).
    Returns dict of lists for each metric.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    accs, lin_ks, qwk_ks = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t = y_true[idx]
        p = y_pred[idx]
        accs.append(accuracy_score(t, p))
        lin_ks.append(cohen_kappa_score(t, p, weights="linear"))
        qwk_ks.append(cohen_kappa_score(t, p, weights="quadratic"))

    return {
        "accuracy": accs,
        "linear_kappa": lin_ks,
        "qwk_kappa": qwk_ks,
    }

def ci(values, confidence: float = 0.95):
    """
    Return (mean, lower_ci, upper_ci, std) for a list/array of values.
    """
    values = np.asarray(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    alpha = 1 - confidence
    lower = float(np.percentile(values, 100 * (alpha / 2)))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return mean, lower, upper, std

# --------- grouping/splits ---------
def grouped_split_and_save(df: pd.DataFrame, out_dir: str, prefix: str,
                           id_col="ICUSTAY_ID", test_size=0.2, seed=42):
    """GroupShuffleSplit by ICUSTAY_ID and save train/test CSVs."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[id_col]
    train_idx, test_idx = next(gss.split(df, None, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    train_df.to_csv(os.path.join(out_dir, f"{prefix}_train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, f"{prefix}_test.csv"), index=False)

# --------- scaling for LR ---------
def standardize_for_lr(df: pd.DataFrame, id_col="ICUSTAY_ID") -> pd.DataFrame:
    """Z-score numeric cols except targets (LOS, MORTALITY, LOS_BUCKET)."""
    df = df.copy()
    skip = {id_col, "LOS", "MORTALITY", "LOS_BUCKET"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df



def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def drop_bad_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed/empty and constant columns."""
    if df is None or df.empty:
        return df
    out = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    out = out.loc[:, out.columns != ""]
    nunique = out.nunique(dropna=False)
    out = out.loc[:, nunique > 0]
    return out

def one_hot_align(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode any non-numeric columns jointly, then split back to aligned matrices.
    Also replaces inf with NaN and fills NaN with 0 (RF can't handle NaN).
    """
    train_df = drop_bad_cols(train_df)
    test_df  = drop_bad_cols(test_df)

    # cast bool -> int (optional, keeps it numeric)
    for df in (train_df, test_df):
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)

    # combine, get_dummies once to keep the same columns
    combo = pd.concat([train_df, test_df], axis=0, keys=["__train__", "__test__"])
    obj_cols = combo.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        combo = pd.get_dummies(combo, columns=obj_cols, dummy_na=True)

    combo = combo.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train_enc = combo.xs("__train__")
    X_test_enc  = combo.xs("__test__")
    # just in case, align columns
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    return X_train_enc, X_test_enc
