"""
Reproducible MIMIC-III EDA Pipeline
===================================

This single Python script turns your ad‑hoc EDA into a **portable, rerunnable pipeline**.
It avoids hard‑coded Windows paths, validates inputs, saves plots, and gracefully
handles optional libraries.

---
Quick start
----------
1) Put your data in a directory like:
   data/
     preprocessed/
       static_data.csv
       dynamic/               # per-ICUSTAY hourly CSVs
         *.csv
     dynamic_raw/             # (optional) raw per-subject episodes to combine
       *.csv

2) Run:
   python eda_mimic3.py \
       --data-root ./data \
       --out-root ./eda_output \
       --combine-from dynamic_raw   # optional: if you want to build dynamic_data_eda.csv from many files

3) Outputs (under --out-root):
   - combined/dynamic_data_eda.csv
   - combined/los_combined.csv
   - figures/*.png (all plots)
   - tables/*.csv  (stats and test results)

---
Requirements
-----------
- Python 3.9+
- pandas, numpy, matplotlib, scipy, statsmodels
- Optional: seaborn, tqdm

Install (example):
  pip install pandas numpy matplotlib scipy statsmodels seaborn tqdm

Notes
-----
- The script is defensive: it checks files/columns and skips sections that
  can't run, logging a warning rather than crashing.
- Plots are **saved to PNG**; if running interactively, use --show to display.
- Seaborn is optional; where seaborn-only visuals are requested (e.g., pairplot),
  we skip if seaborn isn't installed.
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
import warnings
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional deps
try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

try:
    from tqdm import tqdm  # type: ignore
    TQDM = tqdm
except Exception:
    def TQDM(x, **kwargs):
        return x

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 200)


# ------------------------------
# Utils
# ------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_or_show(figpath: Path, show: bool):
    if figpath:
        ensure_dir(figpath.parent)
        plt.savefig(figpath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


# ------------------------------
# 1) Combine dynamic files
#    (A) from a flat folder of CSVs (old)
#    (B) NEW: recursively find and combine all `wide_events.csv` under preprocessed/
# ------------------------------

def combine_dynamic(dynamic_raw_dir: Path, output_csv: Path) -> Optional[pd.DataFrame]:
    if not dynamic_raw_dir.exists():
        print(f"[INFO] Dynamic raw dir '{dynamic_raw_dir}' not found; skipping combine.")
        return None

    files = sorted(dynamic_raw_dir.glob("*.csv"))
    print(f"Found {len(files)} dynamic CSV files in {dynamic_raw_dir}")
    dfs = []
    for f in TQDM(files, desc="Reading dynamic raw"):
        try:
            df = pd.read_csv(f)
            df["SOURCE_FILE"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
    if not dfs:
        print("[WARN] No CSV files loaded from dynamic_raw. Skipping combine.")
        return None
    combined = pd.concat(dfs, ignore_index=True)
    ensure_dir(output_csv.parent)
    combined.to_csv(output_csv, index=False)
    print(f"[OK] Combined {len(dfs)} files → {output_csv} (rows: {len(combined):,})")
    return combined


REQUIRED_DYNAMIC_COLS = [
    'SUBJECT_ID','HADM_ID','ICUSTAY_ID','CHARTTIME','HOUR',
    'Bilirubin', 'Blood Urea Nitrogen', 'Capillary refill rate', 'Creatinine',
    'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate',
    'Hematocrit', 'Hemoglobin', 'Lactate', 'Mean blood pressure',
    'Oxygen saturation', 'Platelet Count', 'Potassium',
    'Respiratory rate', 'Sodium', 'Systolic blood pressure', 'Temperature',
    'White Blood Cell Count', 'pH', 'Urine Output',
    'Glascow coma scale total', 'Height', 'Weight',
    'Glascow coma scale eye opening', 'Glascow coma scale motor response',
    'Glascow coma scale verbal response'
]


def _ensure_required_dynamic_cols(df: pd.DataFrame) -> pd.DataFrame:
    # add any missing required columns as NaN
    for col in REQUIRED_DYNAMIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
    # ensure HOUR exists; compute from CHARTTIME if possible
    if 'HOUR' in df.columns and df['HOUR'].isna().all() and 'CHARTTIME' in df.columns:
        try:
            tmp = df.copy()
            tmp['CHARTTIME'] = pd.to_datetime(tmp['CHARTTIME'], errors='coerce')
            df['HOUR'] = tmp.groupby('ICUSTAY_ID')['CHARTTIME'].transform(lambda x: (x - x.min()).dt.total_seconds()/3600)
        except Exception:
            pass
    # order columns: required first, then the rest to preserve extra info
    ordered = REQUIRED_DYNAMIC_COLS + [c for c in df.columns if c not in REQUIRED_DYNAMIC_COLS]
    return df[ordered]


def combine_wide_events(preprocessed_dir: Path, output_csv: Path) -> Optional[pd.DataFrame]:
    """Recursively find every `wide_events.csv` under `preprocessed_dir`,
    ensure all REQUIRED_DYNAMIC_COLS are present, then concatenate and save.

    Writes to `output_csv` (typically <data-root>/preprocessed/dynamic_data_eda.csv).
    """
    if not preprocessed_dir.exists():
        print(f"[WARN] preprocessed dir not found: {preprocessed_dir}")
        return None

    paths = sorted(preprocessed_dir.rglob('wide_events.csv'))
    print(f"Found {len(paths)} wide_events.csv files under {preprocessed_dir}")
    if not paths:
        return None

    dfs: List[pd.DataFrame] = []
    for p in TQDM(paths, desc='Reading wide_events.csv'):
        try:
            df = pd.read_csv(p)
            df = _ensure_required_dynamic_cols(df)
            df['SOURCE_FILE'] = str(p)
            # normalize ID types to strings to avoid mixed types
            for idc in ['SUBJECT_ID','HADM_ID','ICUSTAY_ID']:
                if idc in df.columns:
                    df[idc] = df[idc].astype(str)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    if not dfs:
        print("[WARN] No readable wide_events.csv files.")
        return None

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # canonical sort if timestamps available
    if 'ICUSTAY_ID' in combined.columns:
        sort_cols = [c for c in ['ICUSTAY_ID','CHARTTIME','HOUR'] if c in combined.columns]
        if sort_cols:
            try:
                if 'CHARTTIME' in sort_cols:
                    combined['CHARTTIME'] = pd.to_datetime(combined['CHARTTIME'], errors='coerce')
                combined = combined.sort_values(sort_cols).reset_index(drop=True)
            except Exception:
                pass
    ensure_dir(output_csv.parent)
    combined.to_csv(output_csv, index=False)
    print(f"[OK] Combined wide_events → {output_csv} (rows: {len(combined):,}, cols: {combined.shape[1]})")
    return combined


def attach_los(dynamic_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    """Merge LOS from static_data.csv onto dynamic data by ICUSTAY_ID.
    Keeps all dynamic rows; adds LOS (and LOS_HOURS) if available.
    Ensures ICUSTAY_ID is compared as string on both sides.
    """
    if dynamic_df is None or static_df is None:
        return dynamic_df
    if 'ICUSTAY_ID' not in dynamic_df.columns or 'ICUSTAY_ID' not in static_df.columns:
        print('[WARN] Cannot merge LOS: ICUSTAY_ID missing in one of the dataframes.')
        return dynamic_df
    left = dynamic_df.copy()
    right = static_df.copy()
    left['ICUSTAY_ID'] = left['ICUSTAY_ID'].astype(str)
    right['ICUSTAY_ID'] = right['ICUSTAY_ID'].astype(str)
    keep_cols = ['ICUSTAY_ID','LOS'] if 'LOS' in right.columns else ['ICUSTAY_ID']
    merged = left.merge(right[keep_cols].drop_duplicates('ICUSTAY_ID'), on='ICUSTAY_ID', how='left')
    if 'LOS' in merged.columns:
        merged['LOS_HOURS'] = merged['LOS'] * 24
    print('[OK] Merged LOS from static_data.csv onto dynamic (by ICUSTAY_ID).')
    return merged

# ------------------------------
# 2) Missingness Pattern
# ------------------------------

def plot_missingness(static_df: Optional[pd.DataFrame], dynamic_df: Optional[pd.DataFrame], outdir: Path, show: bool):
    # Static
    if static_df is not None and len(static_df) > 0:
        static_missing = static_df.isnull().mean().sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        static_missing.head(30).plot(kind='bar')
        plt.title("Proportion of Missing Values in Static Data")
        plt.xlabel("Variables")
        plt.ylabel("Proportion Missing")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        save_or_show(outdir/"figures/static_missing_top30.png", show)
    else:
        print("[WARN] Static DF unavailable; skipping static missingness plot.")

    # Dynamic
    if dynamic_df is not None and len(dynamic_df) > 0:
        strict_meta_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'HOUR']
        dynamic_vars = [c for c in dynamic_df.columns if c not in strict_meta_cols]
        if dynamic_vars:
            missing_fraction = dynamic_df[dynamic_vars].isnull().mean().sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            missing_fraction.plot(kind='bar')
            plt.title("Proportion of Missing Values in Dynamic Data")
            plt.xlabel("Features")
            plt.ylabel("Proportion Missing")
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            save_or_show(outdir/"figures/dynamic_missing.png", show)
        else:
            print("[WARN] No dynamic variables remaining after metadata exclusion.")

        # Hourly heatmap
        if 'CHARTTIME' in dynamic_df.columns:
            ddf = dynamic_df.copy()
            ddf['CHARTTIME'] = pd.to_datetime(ddf['CHARTTIME'], errors='coerce')
            ddf['CHART_HOUR'] = ddf['CHARTTIME'].dt.hour
            meta_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'HOUR', 'CHART_HOUR',
                         'GENDER', 'AGE', 'ETHNICITY', 'LOS', 'MORTALITY_INUNIT', 'MORTALITY_INHOSPITAL']
            dyn_vars2 = [c for c in ddf.columns if c not in meta_cols]
            if dyn_vars2:
                missing_by_hour = ddf.groupby("CHART_HOUR")[dyn_vars2].apply(lambda df: df.isnull().mean())
                missing_heatmap = missing_by_hour.T
                plt.figure(figsize=(14, max(6, len(dyn_vars2) * 0.3)))
                if HAS_SEABORN:
                    sns.heatmap(missing_heatmap, annot=False, cbar_kws={"label": "Missing Fraction"})
                else:
                    plt.imshow(missing_heatmap, aspect='auto', interpolation='nearest')
                    plt.colorbar(label='Missing Fraction')
                    plt.yticks(range(len(missing_heatmap.index)), missing_heatmap.index)
                    plt.xticks(range(len(missing_heatmap.columns)), missing_heatmap.columns)
                plt.title("Missing Data Pattern by Hour of Day (0–23)")
                plt.xlabel("Hour of Day")
                plt.ylabel("Variables")
                save_or_show(outdir/"figures/dynamic_missing_heatmap.png", show)
            else:
                print("[WARN] No dynamic variables available for hourly heatmap.")
        else:
            print("[WARN] 'CHARTTIME' missing; skipping hourly missingness heatmap.")
    else:
        print("[WARN] Dynamic DF unavailable; skipping dynamic missingness plots.")


# ------------------------------
# 3) LOS & Remaining LOS
# ------------------------------

def compute_los_combined(dynamic_dir: Path, static_df: pd.DataFrame, output_csv: Path) -> Optional[pd.DataFrame]:
    if not dynamic_dir.exists():
        print(f"[WARN] Dynamic per-ICU dir missing: {dynamic_dir}")
        return None

    static_small = static_df[["ICUSTAY_ID", "LOS"]].dropna().copy()
    static_small["ICUSTAY_ID"] = static_small["ICUSTAY_ID"].astype(str)
    static_small["LOS_HOURS"] = static_small["LOS"] * 24

    entries: List[Dict] = []
    for fname in TQDM(os.listdir(dynamic_dir), desc=f"Scanning {dynamic_dir}"):
        if not fname.endswith(".csv"):
            continue
        fpath = dynamic_dir / fname
        try:
            dyn_df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARN] Skip {fname}: {e}")
            continue
        if "ICUSTAY_ID" not in dyn_df.columns or "HOUR" not in dyn_df.columns:
            continue
        icu_id = str(dyn_df["ICUSTAY_ID"].iloc[0])
        row = static_small[static_small["ICUSTAY_ID"] == icu_id]
        if row.empty:
            continue
        los_hours = float(row["LOS_HOURS"].values[0])
        for _, r in dyn_df.iterrows():
            try:
                remaining = max(los_hours - float(r["HOUR"]), 0.0)
            except Exception:
                continue
            entries.append({
                "ICUSTAY_ID": icu_id,
                "HOUR": r["HOUR"],
                "LOS_HOURS": los_hours,
                "REMAINING_LOS_HOURS": remaining,
            })
    if not entries:
        print("[WARN] No LOS entries computed.")
        return None
    out = pd.DataFrame(entries)
    ensure_dir(output_csv.parent)
    out.to_csv(output_csv, index=False)
    print(f"[OK] Wrote {output_csv} (rows: {len(out):,})")
    return out


def plot_los_distributions(los_df: pd.DataFrame, outdir: Path, show: bool):
    # Total & Remaining (days)
    total_los_days = los_df.groupby("ICUSTAY_ID")["LOS_HOURS"].mean() / 24
    remaining_los_days = los_df["REMAINING_LOS_HOURS"] / 24

    # Hist + KDE (matplotlib; seaborn optional)
    plt.figure(figsize=(10, 5))
    if HAS_SEABORN:
        sns.histplot(total_los_days, bins=100, stat='density', alpha=0.4, label='Total LOS (Histogram)')
        sns.kdeplot(total_los_days, label='Total LOS (KDE)')
        sns.histplot(remaining_los_days, bins=100, stat='density', alpha=0.4, label='Remaining LOS (Histogram)')
        sns.kdeplot(remaining_los_days, label='Remaining LOS (KDE)')
    else:
        plt.hist(total_los_days, bins=100, density=True, alpha=0.4, label='Total LOS (Hist)')
        plt.hist(remaining_los_days, bins=100, density=True, alpha=0.4, label='Remaining LOS (Hist)')
    med_total = total_los_days.quantile(0.5)
    med_rem = remaining_los_days.quantile(0.5)
    plt.axvline(med_total, linestyle='--', alpha=0.8)
    plt.axvline(med_rem, linestyle='--', alpha=0.8)
    plt.xlabel("LOS (days)"); plt.ylabel("Density")
    plt.title("Distribution of Total and Remaining LOS (Days)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(-1, max(80, float(total_los_days.max())+1))
    plt.legend()
    save_or_show(outdir/"figures/los_total_remaining.png", show)

    # Bucketed bar (%)
    bucket_edges_days = [0, 1, 2, 3, 4, 5, 7, 14, np.inf]
    bucket_labels = ["0–1d", "1–2d", "2–3d", "3–4d", "4–5d", "5–7d", "7–14d", "14d+"]
    los_bucketed = pd.cut(total_los_days, bins=bucket_edges_days, labels=bucket_labels, right=False)
    rem_bucketed = pd.cut(remaining_los_days, bins=bucket_edges_days, labels=bucket_labels, right=False)
    los_pct = (los_bucketed.value_counts(normalize=True).sort_index()) * 100
    rem_pct = (rem_bucketed.value_counts(normalize=True).sort_index()) * 100

    x = np.arange(len(bucket_labels)); width = 0.38
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, los_pct.values, width=width, label="Total LOS (%)", alpha=0.8)
    plt.bar(x + width/2, rem_pct.values, width=width, label="Remaining LOS (%)", alpha=0.8)
    plt.xticks(x, bucket_labels)
    plt.xlabel("LOS Buckets (Days)"); plt.ylabel("Percentage (%)")
    plt.title("Bucketized Distribution of Total vs Remaining LOS (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    save_or_show(outdir/"figures/los_bucketized.png", show)

    # Save tables
    ensure_dir((outdir/"tables").resolve())
    los_pct.to_csv(outdir/"tables/los_total_bucket_pct.csv")
    rem_pct.to_csv(outdir/"tables/los_remaining_bucket_pct.csv")


# ------------------------------
# 4) Dynamic data analysis (Heart Rate trajectories & features)
# ------------------------------
from scipy.stats import linregress, entropy, pearsonr

def plot_hr_trajectories_by_los(dynamic_df: pd.DataFrame, outdir: Path, show: bool):
    needed_cols = {'ICUSTAY_ID','CHARTTIME','Heart Rate','LOS'}
    if not needed_cols.issubset(set(dynamic_df.columns)):
        print(f"[WARN] Missing columns {needed_cols - set(dynamic_df.columns)}; skipping HR trajectories.")
        return

    df = dynamic_df.copy()
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'], errors='coerce')
    df['HOUR'] = df.groupby('ICUSTAY_ID')['CHARTTIME'].transform(lambda x: (x - x.min()).dt.total_seconds()/3600)
    df_hr = df[['ICUSTAY_ID','HOUR','Heart Rate','LOS']].dropna()

    def los_group(x: float) -> str:
        if x < 1: return '<1 day'
        elif x < 3: return '1-3 days'
        elif x < 14: return '3-14 days'
        else: return '>14 days'

    df_hr['LOS_GROUP'] = df_hr['LOS'].apply(los_group)

    time_grid = np.arange(0, 49)
    quartile_curves: Dict[str, np.ndarray] = {}

    for group in df_hr['LOS_GROUP'].unique():
        g = df_hr[df_hr['LOS_GROUP']==group]
        ids = g['ICUSTAY_ID'].unique()
        aligned = np.full((len(time_grid), len(ids)), np.nan, dtype=float)
        for i, uid in enumerate(ids):
            ts = g[g['ICUSTAY_ID']==uid][['HOUR','Heart Rate']].dropna()
            if ts.empty:
                continue
            ts = ts.groupby('HOUR').mean().reindex(time_grid).interpolate()
            aligned[:, i] = ts.values.flatten()
        quartiles = np.nanpercentile(aligned, [25,50,75], axis=1).T
        quartile_curves[group] = quartiles

    plt.figure(figsize=(14,6))
    colors = {'<1 day': None, '1-3 days': None, '3-14 days': None, '>14 days': None}
    for group in colors.keys():
        if group in quartile_curves:
            q = quartile_curves[group]
            plt.plot(time_grid, q[:,1], label=f'{group} median')
            plt.fill_between(time_grid, q[:,0], q[:,2], alpha=0.2, label=f'{group} IQR')
    plt.xlabel('Time Since ICU Admission (hours)')
    plt.ylabel('Heart Rate (bpm)')
    plt.title('Heart Rate Trajectories by LOS Group (First 48 Hours)')
    plt.legend()
    plt.grid(True)
    save_or_show(outdir/"figures/hr_trajectories_by_los.png", show)


def hr_feature_correlations(dynamic_df: pd.DataFrame, outdir: Path, show: bool):
    needed_cols = {'ICUSTAY_ID','HOUR','Heart Rate','LOS'}
    if not needed_cols.issubset(set(dynamic_df.columns)):
        print(f"[WARN] Missing columns {needed_cols - set(dynamic_df.columns)}; skipping HR feature correlations.")
        return

    df = dynamic_df.dropna(subset=['ICUSTAY_ID','HOUR','Heart Rate','LOS']).copy()
    df_avg = df.groupby(['ICUSTAY_ID','HOUR'])['Heart Rate'].mean().reset_index()
    df_avg = df_avg.merge(df[['ICUSTAY_ID','LOS']].drop_duplicates(), on='ICUSTAY_ID', how='left')

    def compute_hr_features(group: pd.DataFrame, max_hour: int = 24) -> pd.Series:
        los = float(group['LOS'].iloc[0])
        early = group[group['HOUR'] <= min(max_hour, los)].sort_values('HOUR')
        if early.empty or early['HOUR'].nunique() < 1:
            keys = ['mean','std','min','max','first','last','auc','auc_norm','slope','change','entropy','reversals']
            return pd.Series({f'{max_hour}h_{k}': np.nan for k in keys} | {'LOS': los})
        x = early['HOUR'].values
        y = early['Heart Rate'].values
        hist = np.histogram(y, bins=10, density=True)[0] + 1e-8
        stats = {
            f'{max_hour}h_mean': float(np.mean(y)),
            f'{max_hour}h_std': float(np.std(y)),
            f'{max_hour}h_min': float(np.min(y)),
            f'{max_hour}h_max': float(np.max(y)),
            f'{max_hour}h_first': float(y[0]),
            f'{max_hour}h_last': float(y[-1]),
            f'{max_hour}h_auc': float(np.trapz(y, x)),
            f'{max_hour}h_auc_norm': float(np.trapz(y, x) / (x[-1]-x[0])) if x[-1]!=x[0] else np.nan,
            f'{max_hour}h_slope': float(linregress(x, y).slope) if len(x) > 1 else np.nan,
            f'{max_hour}h_change': float(y[-1]-y[0]),
            f'{max_hour}h_entropy': float(entropy(hist)),
            f'{max_hour}h_reversals': float(np.sum(np.diff(np.sign(np.diff(y))) != 0)),
            'LOS': los,
        }
        return pd.Series(stats)

    features_24h = df_avg.groupby('ICUSTAY_ID').apply(compute_hr_features)

    results = []
    for col in features_24h.columns:
        if col == 'LOS':
            continue
        x = features_24h[col]
        y = features_24h['LOS']
        mask = x.notna() & y.notna()
        if mask.sum() > 10:
            r, p = pearsonr(x[mask], y[mask])
            results.append({'Feature': col, 'Correlation': r, 'P_value': p})
    res_df = pd.DataFrame(results).sort_values('P_value')

    ensure_dir((outdir/"tables").resolve())
    res_df.to_csv(outdir/"tables/hr_feature_correlations_24h.csv", index=False)

    # Pairplot for top 5 if seaborn available
    if HAS_SEABORN and not res_df.empty:
        top5 = res_df.head(5)['Feature'].tolist()
        plot_df = features_24h[top5 + ['LOS']].dropna()
        g = sns.pairplot(plot_df, kind='reg', corner=True, diag_kind='kde')
        g.fig.suptitle("Top 5 Heart Rate Indicators (24h) vs LOS", y=1.02)
        save_or_show(outdir/"figures/hr_top5_pairplot.png", show)
    elif not HAS_SEABORN:
        print("[INFO] seaborn not installed; skipping pairplot.")


# ------------------------------
# 5) Static data analysis
# ------------------------------
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

def static_diagnosis_vs_mortality(static_df: pd.DataFrame, outdir: Path, show: bool):
    if static_df is None or static_df.empty:
        print("[WARN] Static DF unavailable; skipping diagnosis vs mortality.")
        return
    known_cols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','GENDER','AGE','ETHNICITY',
                  'HEIGHT','WEIGHT','LOS','MORTALITY_INUNIT','MORTALITY_INHOSPITAL','MORTALITY']
    diagnosis_cols = [c for c in static_df.columns if c not in known_cols and static_df[c].nunique()==2]

    results = []
    for col in diagnosis_cols:
        contingency = pd.crosstab(static_df[col], static_df['MORTALITY_INHOSPITAL'])
        if contingency.shape == (2,2):
            chi2, p, *_ = chi2_contingency(contingency)
            results.append((col, p))
    if not results:
        print("[WARN] No binary diagnosis columns for chi-squared.")
        return

    p_vals = [p for _, p in results]
    _, corrected, _, _ = multipletests(p_vals, method='bonferroni')
    df = pd.DataFrame({'Diagnosis':[c for c,_ in results], 'P_value':p_vals, 'P_corrected':corrected})
    df['minus_log10_Pcorr'] = -np.log10(df['P_corrected'])
    df.to_csv(outdir/"tables/diagnosis_vs_mortality_chi2.csv", index=False)

    sig = df[df['P_corrected'] < 0.05].sort_values('P_corrected')
    if not sig.empty:
        plt.figure(figsize=(12,5))
        if HAS_SEABORN:
            sns.barplot(data=sig.sort_values('minus_log10_Pcorr', ascending=False),
                        x='minus_log10_Pcorr', y='Diagnosis')
        else:
            plt.barh(sig['Diagnosis'], sig['minus_log10_Pcorr'])
            plt.gca().invert_yaxis()
        plt.title("Top Diagnoses Associated with In-hospital Mortality (Chi-squared)")
        plt.xlabel("-log10(Bonferroni-corrected P)")
        plt.ylabel("Diagnosis")
        save_or_show(outdir/"figures/diagnosis_mortality_chi2.png", show)


def static_diagnosis_vs_los(static_df: pd.DataFrame, outdir: Path, show: bool, start_var: Optional[str] = None):
    if static_df is None or static_df.empty or 'LOS' not in static_df.columns:
        print("[WARN] Static DF/LOS unavailable; skipping diagnosis vs LOS.")
        return
    if start_var and start_var in static_df.columns:
        start_idx = static_df.columns.get_loc(start_var)
        diagnosis_cols = static_df.columns[start_idx:]
    else:
        known_cols = {'SUBJECT_ID','HADM_ID','ICUSTAY_ID','GENDER','AGE','ETHNICITY','HEIGHT','WEIGHT','LOS',
                      'MORTALITY_INUNIT','MORTALITY_INHOSPITAL','MORTALITY'}
        diagnosis_cols = [c for c in static_df.columns if c not in known_cols and static_df[c].nunique()==2]

    results = []
    for col in diagnosis_cols:
        if static_df[col].nunique()!=2:
            continue
        g0 = static_df[static_df[col]==0]['LOS'].dropna()
        g1 = static_df[static_df[col]==1]['LOS'].dropna()
        if len(g0) > 10 and len(g1) > 10:
            stat, p = mannwhitneyu(g0, g1, alternative='two-sided')
            results.append((col, p))
    if not results:
        print("[WARN] No suitable binary diagnosis columns for Mann-Whitney.")
        return

    p_vals = [p for _, p in results]
    _, corrected, _, _ = multipletests(p_vals, method='bonferroni')
    df = pd.DataFrame({'Diagnosis':[c for c,_ in results], 'P_value':p_vals, 'P_corrected':corrected})
    df['minus_log10_Pcorr'] = -np.log10(df['P_corrected'])
    df.to_csv(outdir/"tables/diagnosis_vs_los_mannwhitney.csv", index=False)

    sig = df[df['P_corrected'] < 0.05].sort_values('P_corrected')
    top = sig.head(min(20, len(sig)))
    if not top.empty:
        plt.figure(figsize=(12,6))
        if HAS_SEABORN:
            sns.barplot(data=top.sort_values('minus_log10_Pcorr', ascending=False),
                        x='minus_log10_Pcorr', y='Diagnosis')
        else:
            plt.barh(top['Diagnosis'], top['minus_log10_Pcorr'])
            plt.gca().invert_yaxis()
        plt.title("Top Diagnoses Associated with LOS (Mann-Whitney U)")
        plt.xlabel("-log10(Bonferroni-corrected P)")
        plt.ylabel("Diagnosis")
        save_or_show(outdir/"figures/diagnosis_los_mannwhitney.png", show)


def age_gender_heatmaps(static_df: pd.DataFrame, outdir: Path, show: bool):
    needed = {'AGE','GENDER','LOS','MORTALITY_INHOSPITAL'}
    if not needed.issubset(static_df.columns):
        print(f"[WARN] Missing columns {needed - set(static_df.columns)}; skipping age/gender heatmaps.")
        return
    df = static_df[['AGE','GENDER','LOS','MORTALITY_INHOSPITAL']].dropna().copy()
    df['MORTALITY_INHOSPITAL'] = df['MORTALITY_INHOSPITAL'].astype(int)
    # remove LOS outliers via IQR
    Q1, Q3 = df['LOS'].quantile([0.25, 0.75]); IQR = Q3 - Q1
    df = df[(df['LOS'] >= Q1 - 1.5*IQR) & (df['LOS'] <= Q3 + 1.5*IQR)]
    df['AGE_BIN'] = pd.cut(df['AGE'], bins=[0,30,55,70,100], labels=["<30","30-54","55-69",">70"], include_lowest=True)

    los_pivot = df.groupby(['AGE_BIN','GENDER'])['LOS'].mean().unstack()
    mort_pivot = df.groupby(['AGE_BIN','GENDER'])['MORTALITY_INHOSPITAL'].mean().unstack()

    ensure_dir((outdir/"tables").resolve())
    los_pivot.to_csv(outdir/"tables/los_by_age_gender.csv")
    mort_pivot.to_csv(outdir/"tables/mortality_by_age_gender.csv")

    # LOS heatmap
    plt.figure(figsize=(10,3))
    if HAS_SEABORN:
        sns.heatmap(los_pivot, annot=True, fmt=".1f", cbar_kws={'label':'Avg LOS (days)'} )
    else:
        plt.imshow(los_pivot, aspect='auto'); plt.colorbar(label='Avg LOS (days)')
        plt.xticks(range(len(los_pivot.columns)), los_pivot.columns)
        plt.yticks(range(len(los_pivot.index)), los_pivot.index)
    plt.title("Average LOS by Age Group and Gender")
    plt.ylabel("Age Group"); plt.xlabel("Gender")
    save_or_show(outdir/"figures/los_heatmap_age_gender.png", show)

    # Mortality heatmap
    plt.figure(figsize=(10,3))
    if HAS_SEABORN:
        sns.heatmap(mort_pivot, annot=True, fmt=".2f", cbar_kws={'label':'Mortality Rate'})
    else:
        plt.imshow(mort_pivot, aspect='auto'); plt.colorbar(label='Mortality Rate')
        plt.xticks(range(len(mort_pivot.columns)), mort_pivot.columns)
        plt.yticks(range(len(mort_pivot.index)), mort_pivot.index)
    plt.title("Mortality Rate by Age Group and Gender")
    plt.ylabel("Age Group"); plt.xlabel("Gender")
    save_or_show(outdir/"figures/mortality_heatmap_age_gender.png", show)


def kruskal_demographics(static_df: pd.DataFrame, outdir: Path, show: bool):
    if 'LOS' not in static_df.columns:
        print("[WARN] LOS missing; skipping Kruskal-Wallis on demographics.")
        return
    categorical_vars = ['ADMISSION_TYPE','INSURANCE','MARITAL_STATUS','ETHNICITY','LANGUAGE','RELIGION','ADMISSION_LOCATION']
    avail = [v for v in categorical_vars if v in static_df.columns]
    rows = []
    for var in avail:
        valid = static_df[[var,'LOS']].dropna()
        if valid.empty:
            continue
        groups = [g['LOS'].values for _, g in valid.groupby(var)]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        rows.append({'Variable': var, 'Kruskal_Statistic': stat, 'P_value': p})
    if not rows:
        print("[WARN] No categorical vars available for Kruskal-Wallis.")
        return
    df = pd.DataFrame(rows).sort_values('P_value')
    df.to_csv(outdir/"tables/kruskal_demographics_vs_los.csv", index=False)

    plt.figure(figsize=(10,5))
    plt.axhline(0.05, linestyle='--', label='p = 0.05')
    xs = range(len(df))
    plt.scatter(xs, df['P_value'])
    plt.xticks(xs, df['Variable'], rotation=45, ha='right')
    plt.ylabel('p-value'); plt.xlabel('Categorical Variable')
    plt.title('Kruskal-Wallis: LOS vs Demographic Variables')
    plt.legend(); plt.yticks(np.arange(0, 0.55, 0.05))
    plt.tight_layout()
    save_or_show(outdir/"figures/kruskal_demographics_vs_los.png", show)


def admission_type_location_heatmap(static_df: pd.DataFrame, outdir: Path, show: bool):
    needed = {'LOS','ADMISSION_LOCATION','ADMISSION_TYPE'}
    if not needed.issubset(static_df.columns):
        print(f"[WARN] Missing columns {needed - set(static_df.columns)}; skipping admission heatmap.")
        return
    df = static_df.copy()
    # trim LOS extremes
    if df['LOS'].notna().any():
        q95 = df['LOS'].quantile(0.95)
        df = df[df['LOS'] < q95]
    # filter noisy category
    df = df[df['ADMISSION_LOCATION'] != '** INFO NOT AVAILABLE **']
    # pivot
    heat = df.pivot_table(index='ADMISSION_LOCATION', columns='ADMISSION_TYPE', values='LOS', aggfunc='mean')
    # column order if present
    order = [c for c in ['ELECTIVE','URGENT','EMERGENCY'] if c in heat.columns]
    if order:
        heat = heat[order]
    plt.figure(figsize=(12,4))
    if HAS_SEABORN:
        sns.heatmap(heat, annot=True, fmt='.1f', cbar_kws={'label':'Avg LOS (days)'})
    else:
        plt.imshow(heat, aspect='auto'); plt.colorbar(label='Avg LOS (days)')
        plt.xticks(range(len(heat.columns)), heat.columns)
        plt.yticks(range(len(heat.index)), heat.index)
    plt.title('Average LOS by Admission Type and Location')
    plt.xlabel('ADMISSION TYPE'); plt.ylabel('ADMISSION LOCATION')
    save_or_show(outdir/"figures/admission_type_location_heatmap.png", show)


def insurance_quantile_summary(static_df: pd.DataFrame, outdir: Path, show: bool):
    if 'INSURANCE' not in static_df.columns or 'LOS' not in static_df.columns:
        print("[WARN] Missing INSURANCE/LOS; skipping insurance quantiles.")
        return
    df = static_df.copy()
    if df['LOS'].notna().any():
        q95 = df['LOS'].quantile(0.95)
        q5 = df['LOS'].quantile(0.05)
        df = df[(df['LOS'] < q95) & (df['LOS'] > q5)]
    # Optional reweighting from original snippet: keep government above lower tail
    if (df['INSURANCE']=='Government').any():
        gov = df[df['INSURANCE']=='Government']
        q10 = gov['LOS'].quantile(0.03)
        gov_f = gov[gov['LOS'] > q10]
        non_gov = df[df['INSURANCE']!='Government']
        df = pd.concat([non_gov, gov_f], ignore_index=True)

    summary = df.groupby('INSURANCE')['LOS'].quantile([0.25,0.5,0.75]).unstack()
    summary.columns = ['Q1','Median','Q3']
    summary = summary.sort_values('Median', ascending=False)
    ensure_dir((outdir/"tables").resolve())
    summary.to_csv(outdir/"tables/insurance_los_quantiles.csv")

    plt.figure(figsize=(7,4))
    x = np.arange(len(summary.index))
    plt.plot(summary.index, summary['Median'], marker='o', label='Median')
    plt.plot(summary.index, summary['Q1'], marker='^', linestyle='--', label='Q1')
    plt.plot(summary.index, summary['Q3'], marker='v', linestyle='--', label='Q3')
    plt.title('LOS by Insurance Type (Quantiles)')
    plt.ylabel('Length of Stay (days)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(); plt.tight_layout()
    save_or_show(outdir/"figures/insurance_los_quantiles.png", show)


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Reproducible EDA for MIMIC-III static/dynamic data")
    ap.add_argument('--data-root', type=Path, required=True,
                    help='Root data directory containing preprocessed/ and optionally raw dynamic folders')
    ap.add_argument('--out-root', type=Path, default=Path('./eda_output'), help='Directory for outputs (figures/tables)')
    ap.add_argument('--combine-from', type=str, default=None,
                    help='Optional subdir under data-root to combine (e.g., dynamic_raw). Saved to out-root/combined/dynamic_data_eda.csv')
    ap.add_argument('--combine-wide-events', action='store_true',
                    help='Recursively combine all wide_events.csv under <data-root>/preprocessed into <data-root>/preprocessed/dynamic_data.csv')
    ap.add_argument('--show', action='store_true', help='Also display plots interactively')
    ap.add_argument('--start-var', type=str, default=None,
                    help='Optional starting column for diagnosis scan (for Mann-Whitney).')
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_root: Path = args.out_root
    ensure_dir(out_root)

    pre = data_root / 'preprocessed'
    dynamic_dir = pre / 'dynamic'
    static_csv = pre / 'static_data.csv'

    # 1) Optional combine steps
    combined_dir = out_root / 'combined'
    combined_dynamic_csv = combined_dir / 'dynamic_data_eda.csv'

    # (A) From a raw folder of CSVs
    if args.combine_from:
        raw_dir = data_root / args.combine_from
        combine_dynamic(raw_dir, combined_dynamic_csv)

    # (B) Recursively combine all wide_events.csv under <data-root>/preprocessed → <data-root>/preprocessed/dynamic_data.csv
    pre = data_root / 'preprocessed'
    static_csv = pre / 'static_data.csv'
    pre_dynamic_combined = pre / 'dynamic_data_eda.csv'
    if args.combine_wide_events:
        combined_df = combine_wide_events(pre, pre_dynamic_combined)
        if combined_df is not None:
            # Attach LOS immediately after combining
            static_df_tmp = load_csv_safe(static_csv)
            if static_df_tmp is not None:
                combined_df = attach_los(combined_df, static_df_tmp)
                combined_df.to_csv(pre_dynamic_combined, index=False)
                print(f"[OK] Updated with LOS → {pre_dynamic_combined}")
            # also mirror to out-root/combined for downstream steps
            ensure_dir(combined_dynamic_csv.parent)
            combined_df.to_csv(combined_dynamic_csv, index=False)
            print(f"[OK] Mirrored combined dynamic to {combined_dynamic_csv}")

    # 2) Load static and dynamic
    static_df = load_csv_safe(static_csv)

    # Prefer combined CSVs; accept either preprocessed/dynamic_data.csv or out-root/combined/dynamic_data.csv
    dynamic_df = None
    if pre_dynamic_combined.exists():
        dynamic_df = load_csv_safe(pre_dynamic_combined)
        if dynamic_df is not None:
            print(f"[OK] Using combined dynamic from {pre_dynamic_combined}")
    if dynamic_df is None and combined_dynamic_csv.exists():
        dynamic_df = load_csv_safe(combined_dynamic_csv)
        if dynamic_df is not None:
            print(f"[OK] Using combined dynamic from {combined_dynamic_csv}")

    # Ensure LOS is attached if available
    if dynamic_df is not None and static_df is not None and 'LOS' not in dynamic_df.columns:
        dynamic_df = attach_los(dynamic_df, static_df)

    # If neither combined exists, attempt per-ICU dir fallback for limited analyses
    dynamic_dir = pre / 'dynamic'
    if dynamic_df is None:
        if dynamic_dir.exists() and any(dynamic_dir.glob('*.csv')):
            print("[INFO] No combined dynamic_data.csv available; some plots will use per-ICU files only.")
        else:
            print(f"[WARN] Dynamic DF unavailable; skipping dynamic missingness plots.")

    # 3) Missingness
    plot_missingness(static_df, dynamic_df, out_root, args.show)

    # 4) LOS & Remaining LOS
    los_combined_csv = combined_dir / 'los_combined.csv'
    los_df = None
    if static_df is not None and dynamic_dir.exists():
        los_df = compute_los_combined(dynamic_dir, static_df, los_combined_csv)
        if los_df is not None:
            plot_los_distributions(los_df, out_root, args.show)

    # 5) Dynamic HR analyses (need combined dynamic)
    if dynamic_df is not None:
        plot_hr_trajectories_by_los(dynamic_df, out_root, args.show)
        hr_feature_correlations(dynamic_df, out_root, args.show)

    # 6) Static analyses
    if static_df is not None:
        static_diagnosis_vs_mortality(static_df, out_root, args.show)
        static_diagnosis_vs_los(static_df, out_root, args.show, start_var=args.start_var)
        age_gender_heatmaps(static_df, out_root, args.show)
        kruskal_demographics(static_df, out_root, args.show)
        admission_type_location_heatmap(static_df, out_root, args.show)
        insurance_quantile_summary(static_df, out_root, args.show)

    print("\n[DONE] EDA artifacts saved under:", out_root.resolve())


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        # argparse completed
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
