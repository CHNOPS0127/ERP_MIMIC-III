#!/usr/bin/env python3
"""
Dynamic events pipeline (coherent with events validation).

Inputs (under raw_root):
- Subject folders from events validation (each has events.csv and/or filtered_events.csv)
- all_stays.csv

Outputs (under raw_root/preprocessed):
- <SUBJECT_ID>/standardized_events.csv   # units standardized + HOUR computed
- <SUBJECT_ID>/wide_events.csv           # pivoted wide per timepoint
- non_numeric_summary.csv                # variable-level non-numeric summary
- dynamic/<SUBJECT_ID>_episode_timeseries*.csv  # episode files (per ICUSTAY)

Notes:
- Assumes <SUBJECT_ID>/filtered_events.csv already exists (created using updated_variable_selection_I.csv).
- Logic mirrors your original snippets; only organized into functions and unified paths.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------
# Helpers
# ---------------------------
def is_subject_folder(name: str) -> bool:
    """Numeric folder names only."""
    return name.isdigit()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def subjects_list(root: str):
    return [d for d in os.listdir(root) if is_subject_folder(d)]


def safe_id_str(series: pd.Series) -> pd.Series:
    """Cast IDs to string safely (keeps length; NaNs become <NA> string)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.astype("Int64").astype(str)


# ---------------------------
# 0) Unit standardization
# ---------------------------
def standardize_units_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Your unit rules, unchanged, with simple comments."""
    df = df.copy()

    # Treat GCS as categorical (skip numeric transforms)
    gcs_labels = [
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
    ]
    non_gcs_mask = ~df["VARIABLE"].isin(gcs_labels)

    # Helper numeric column
    df["VALUE_NUMERIC"] = pd.to_numeric(df["VALUE"], errors="coerce")

    # Temperature: F -> C
    temp_mask = non_gcs_mask & df["LABEL"].str.contains("temperature", case=False, na=False)
    fahr_unit_mask = df["VALUEUOM"].str.contains("f", case=False, na=False)
    fahr_numeric_mask = temp_mask & df["VALUE_NUMERIC"].notna() & (df["VALUE_NUMERIC"] > 79)
    fahr_mask = temp_mask & (fahr_unit_mask | fahr_numeric_mask)
    df.loc[fahr_mask, "VALUE"] = (df.loc[fahr_mask, "VALUE_NUMERIC"] - 32) * 5 / 9
    df.loc[temp_mask, "VALUEUOM"] = "C"

    # Weight: lb/oz -> kg
    weight_mask = non_gcs_mask & df["LABEL"].str.contains("weight", case=False, na=False)
    lb_mask = df["VALUEUOM"].str.contains("lb", case=False, na=False)
    oz_mask = df["VALUEUOM"].str.contains("oz", case=False, na=False)
    df.loc[weight_mask & oz_mask, "VALUE"] = df.loc[weight_mask & oz_mask, "VALUE_NUMERIC"] / 16
    df.loc[weight_mask & (lb_mask | oz_mask), "VALUE"] = (
        df.loc[weight_mask & (lb_mask | oz_mask), "VALUE_NUMERIC"] * 0.453592
    )
    df.loc[weight_mask, "VALUEUOM"] = "kg"

    # Height: in -> cm
    height_mask = non_gcs_mask & df["LABEL"].str.contains("height", case=False, na=False)
    inch_mask = df["VALUEUOM"].str.contains("in", case=False, na=False)
    df.loc[height_mask & inch_mask, "VALUE"] = df.loc[height_mask & inch_mask, "VALUE_NUMERIC"] * 2.54
    df.loc[height_mask, "VALUEUOM"] = "cm"

    # SpO2: 0-1 -> %
    oxy_mask = non_gcs_mask & df["LABEL"].str.contains("oxygen saturation", case=False, na=False)
    low_spo2 = oxy_mask & df["VALUE_NUMERIC"].notna() & (df["VALUE_NUMERIC"] <= 1.0)
    df.loc[low_spo2, "VALUE"] = df.loc[low_spo2, "VALUE_NUMERIC"] * 100
    df.loc[oxy_mask, "VALUEUOM"] = "%"

    # FiO2: percent -> fraction
    fio2_mask = non_gcs_mask & df["LABEL"].str.contains("fio2|Inspired O2 Fraction", case=False, na=False)
    fio2_values = pd.to_numeric(df.loc[fio2_mask, "VALUE"], errors="coerce")
    fio2_values = fio2_values.where(fio2_values <= 1.0, fio2_values / 100)
    df.loc[fio2_mask, "VALUE"] = fio2_values
    df.loc[fio2_mask, "VALUEUOM"] = "fraction"

    # Capillary refill: to binary
    cap_refill_mask = non_gcs_mask & df["LABEL"].str.contains("capillary refill", case=False, na=False)
    cap_refill_map = {"Normal <3 secs": 0, "Abnormal >3 secs": 1}
    df.loc[cap_refill_mask, "VALUE"] = df.loc[cap_refill_mask, "VALUE"].map(cap_refill_map)
    df = df[~(cap_refill_mask & df["VALUE"].isna())]
    df.loc[cap_refill_mask, "VALUEUOM"] = "binary"

    # Cleanup
    df.drop(columns=["VALUE_NUMERIC"], inplace=True)
    return df


def step_units_standardization(raw_root: str, preproc_root: str) -> None:
    """Read filtered_events.csv and write standardized_events.csv (per subject)."""
    subs = subjects_list(preproc_root) or subjects_list(raw_root)  # prefer preprocessed path
    for sid in tqdm(subs, desc="Standardizing units"):
        in_dir = os.path.join(preproc_root, sid) if os.path.isdir(os.path.join(preproc_root, sid)) else os.path.join(raw_root, sid)
        out_dir = os.path.join(preproc_root, sid)
        ensure_dir(out_dir)

        # prefer filtered_events in preprocessed; fall back to raw if needed
        candidates = [
            os.path.join(preproc_root, sid, "filtered_events.csv"),
            os.path.join(raw_root, sid, "filtered_events.csv"),
        ]
        events_path = next((p for p in candidates if os.path.isfile(p)), None)
        if not events_path:
            continue

        df = pd.read_csv(events_path)
        standardized_df = standardize_units_auto(df)
        standardized_df.to_csv(os.path.join(out_dir, "standardized_events.csv"), index=False)


# ---------------------------
# 1) Add HOUR (CHARTTIME - INTIME)
# ---------------------------
def step_add_hour(raw_root: str, preproc_root: str) -> None:
    """Merge INTIME from all_stays.csv and compute HOUR (>= 0)."""
    stays = pd.read_csv(os.path.join(raw_root, "all_stays.csv"))
    # match your casting intent
    stays["SUBJECT_ID"] = safe_id_str(stays["SUBJECT_ID"])
    stays["HADM_ID"] = safe_id_str(stays["HADM_ID"])
    stays["ICUSTAY_ID"] = safe_id_str(stays["ICUSTAY_ID"])

    for sid in tqdm(subjects_list(preproc_root), desc="Adding HOUR"):
        events_path = os.path.join(preproc_root, sid, "standardized_events.csv")
        if not os.path.isfile(events_path):
            continue

        df = pd.read_csv(events_path)

        # same casting intent for events
        for col in ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]:
            if col in df.columns:
                df[col] = safe_id_str(df[col])

        df = df.merge(
            stays[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]],
            on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
            how="left",
        )

        df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], errors="coerce")
        df["INTIME"] = pd.to_datetime(df["INTIME"], errors="coerce")
        df["HOUR"] = ((df["CHARTTIME"] - df["INTIME"]).dt.total_seconds() / 3600).astype(float)
        df = df[df["HOUR"] >= 0]

        df.to_csv(events_path, index=False)


# ---------------------------
# 2) Duplicate + non-numeric summary
# ---------------------------
def step_non_numeric_summary(preproc_root: str) -> str:
    """Scan standardized_events.csv across subjects and write non_numeric_summary.csv."""
    key_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VARIABLE", "LABEL"]
    total_rows = 0
    total_duplicates = 0
    non_numeric_frames = []
    all_events_for_totals = []

    for sid in tqdm(subjects_list(preproc_root), desc="Scanning non-numeric / duplicates"):
        path = os.path.join(preproc_root, sid, "standardized_events.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        total_rows += len(df)

        # duplicates count (keep your subset definition)
        dup = df[df.duplicated(subset=key_cols, keep=False)]
        total_duplicates += len(dup)

        # collect non-numeric rows
        df["VALUE_STR"] = df["VALUE"].astype(str)
        non_num = pd.to_numeric(df["VALUE_STR"], errors="coerce").isna()
        non_numeric_frames.append(df[non_num][["VARIABLE", "VALUE"]])
        all_events_for_totals.append(df[["VARIABLE", "VALUE"]])

    # overall stats
    pct = (total_duplicates / total_rows * 100) if total_rows else 0.0
    print(f"\nTotal Rows: {total_rows}")
    print(f"Total Duplicates: {total_duplicates}")
    print(f"Overall Duplication %: {pct:.2f}%")

    # build summary table (same idea as your code)
    if non_numeric_frames:
        combined_non_numeric = pd.concat(non_numeric_frames, ignore_index=True)
        summary = (
            combined_non_numeric.groupby("VARIABLE")["VALUE"]
            .agg([("Non-Numeric Count", "count"),
                  ("Examples", lambda x: x.value_counts().head(3).to_dict())])
            .reset_index()
        )
        totals = (
            pd.concat(all_events_for_totals, ignore_index=True)
            .groupby("VARIABLE")["VALUE"].count()
            .reset_index(name="Total Count")
        )
        summary = summary.merge(totals, on="VARIABLE", how="left")
        summary["Fraction Non-Numeric"] = summary["Non-Numeric Count"] / summary["Total Count"]
        summary = summary.sort_values("Fraction Non-Numeric", ascending=False)
    else:
        summary = pd.DataFrame(columns=["VARIABLE", "Non-Numeric Count", "Examples", "Total Count", "Fraction Non-Numeric"])

    out_path = os.path.join(preproc_root, "non_numeric_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------
# 3) Clean values using summary (drop low-contam non-numeric; convert special vars)
# ---------------------------
BILIRUBIN_MAP = {
    "NEG": 0.0, "NEGATIVE": 0.0,
    "TRACE": 0.5, "TR": 0.5,
    "SM": 1.0, "SMALL": 1.0,
    "MOD": 2.0, "MODERATE": 2.0,
    "LARGE": 3.0,
    "POS": 1.0, "POSITIVE": 1.0,
}


def convert_bilirubin(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s in BILIRUBIN_MAP:
        return BILIRUBIN_MAP[s]
    try:
        return float(s)
    except Exception:
        return np.nan


def convert_wbc(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    # ranges like "4.0 - 10.0" -> median
    if re.match(r"^\d+(\.\d+)?\s*-\s*\d+(\.\d+)?$", s):
        a, b = map(float, re.split(r"\s*-\s*", s))
        return np.median([a, b])
    try:
        return float(s)
    except Exception:
        return np.nan


def step_clean_values(preproc_root: str, summary_csv: str, frac_threshold: float = 0.03) -> None:
    """Drop non-numeric rows for low-contamination variables; convert Bilirubin and WBC."""
    summary_df = pd.read_csv(summary_csv) if os.path.isfile(summary_csv) else pd.DataFrame()
    low_contam = set(summary_df.loc[summary_df["Fraction Non-Numeric"] < frac_threshold, "VARIABLE"]) if not summary_df.empty else set()

    for sid in tqdm(subjects_list(preproc_root), desc="Cleaning values"):
        path = os.path.join(preproc_root, sid, "standardized_events.csv")
        if not os.path.isfile(path):
            continue

        df = pd.read_csv(path)
        df["VALUE_STR"] = df["VALUE"].astype(str)
        non_num = pd.to_numeric(df["VALUE_STR"], errors="coerce").isna()

        # drop only for low-contamination variables
        drop_mask = df["VARIABLE"].isin(low_contam) & non_num
        cleaned = df[~drop_mask].copy()

        # special conversions
        bmask = cleaned["VARIABLE"] == "Bilirubin"
        cleaned.loc[bmask, "VALUE"] = cleaned.loc[bmask, "VALUE"].apply(convert_bilirubin)

        wmask = cleaned["VARIABLE"] == "White Blood Cell Count"
        cleaned.loc[wmask, "VALUE"] = cleaned.loc[wmask, "VALUE"].apply(convert_wbc)

        cleaned.drop(columns=["VALUE_STR"], inplace=True)
        cleaned.to_csv(path, index=False)


# ---------------------------
# 4) Wide pivot per subject
# ---------------------------
def step_pivot_wide(preproc_root: str) -> None:
    """Make wide per-timepoint table; keep GCS text."""
    gcs_labels = {
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale verbal response",
    }
    required = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "HOUR", "VARIABLE", "VALUE"]

    for sid in tqdm(subjects_list(preproc_root), desc="Pivoting to wide"):
        in_path = os.path.join(preproc_root, sid, "standardized_events.csv")
        out_path = os.path.join(preproc_root, sid, "wide_events.csv")
        if not os.path.isfile(in_path):
            continue

        df = pd.read_csv(in_path)
        if not all(c in df.columns for c in required):
            print(f"Skipped {sid}: missing required columns")
            continue

        df["VALUE_NUMERIC"] = pd.to_numeric(df["VALUE"], errors="coerce")
        is_gcs = df["VARIABLE"].isin(gcs_labels)
        is_num = df["VALUE_NUMERIC"].notna()
        df = df[is_gcs | is_num].copy()

        df_gcs = df[is_gcs].copy()
        df_num = df[~is_gcs].copy()

        idx = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "HOUR"]

        pv_num = df_num.pivot_table(index=idx, columns="VARIABLE", values="VALUE_NUMERIC", aggfunc="mean")
        pv_gcs = df_gcs.pivot_table(index=idx, columns="VARIABLE", values="VALUE", aggfunc="first")

        wide = pd.concat([pv_num, pv_gcs], axis=1).reset_index()
        wide.to_csv(out_path, index=False)


# ---------------------------
# 5) Collect + split encounters
# ---------------------------
def step_collect_and_split(preproc_root: str) -> None:
    """Copy each subject's wide file into dynamic/ and split into per-ICUSTAY encounter files."""
    dynamic_dir = os.path.join(preproc_root, "dynamic")
    ensure_dir(dynamic_dir)

    # copy as <SUBJECT_ID>_encounters.csv
    for sid in tqdm(subjects_list(preproc_root), desc="Copying wide to dynamic"):
        src = os.path.join(preproc_root, sid, "wide_events.csv")
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dynamic_dir, f"{sid}_encounters.csv")
        pd.read_csv(src).to_csv(dst, index=False)

    # split each *_encounters.csv by ICUSTAY_ID
    for fname in os.listdir(dynamic_dir):
        if not fname.endswith("_encounters.csv"):
            continue
        sid = fname.split("_")[0]
        path = os.path.join(dynamic_dir, fname)
        try:
            df = pd.read_csv(path)
            if "ICUSTAY_ID" not in df.columns:
                print(f"Skipping {fname}: missing ICUSTAY_ID")
                continue
            count = 0
            for count, (icu, grp) in enumerate(df.groupby("ICUSTAY_ID"), start=1):
                out = os.path.join(dynamic_dir, f"{sid}_encounter_timeseries{count}.csv")
                grp.to_csv(out, index=False)
            print(f"Split {fname} into {count} encounter file(s)")
        finally:
            # remove the temporary *_encounters.csv file
            try:
                os.remove(path)
            except Exception:
                pass



# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic events pipeline (units -> HOUR -> clean -> wide -> episodes).")
    p.add_argument("raw_root", type=str, help="Root folder from events validation (has subject folders + all_stays.csv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root
    preproc_root = os.path.join(raw_root, "preprocessed")
    ensure_dir(preproc_root)

    # 0) Units
    step_units_standardization(raw_root, preproc_root)

    # 1) HOUR
    step_add_hour(raw_root, preproc_root)

    # 2) Summary
    summary_csv = step_non_numeric_summary(preproc_root)

    # 3) Clean values
    step_clean_values(preproc_root, summary_csv, frac_threshold=0.03)

    # 4) Wide pivot
    step_pivot_wide(preproc_root)

    # 5) Collect + split
    step_collect_and_split(preproc_root)

    print("\nDone. Outputs in:", preproc_root)


if __name__ == "__main__":
    main()
