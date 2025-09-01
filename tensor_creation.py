#!/usr/bin/env python3


import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

# --- features we expect in dynamic_data_imputed.csv ---
GCS_COLS = [
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
]
NUMERICAL_COLUMNS = [
    "Bilirubin", "Blood Urea Nitrogen", "Capillary refill rate", "Creatinine",
    "Diastolic blood pressure", "Fraction inspired oxygen", "Glucose", "Heart Rate",
    "Hematocrit", "Hemoglobin", "Lactate", "Mean blood pressure",
    "Oxygen saturation", "Platelet Count", "Potassium",
    "Respiratory rate", "Sodium", "Systolic blood pressure", "Temperature",
    "White Blood Cell Count", "pH", "Urine Output",
    "Glascow coma scale total", "Height", "Weight",
]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def standardize_numerical_columns(df: pd.DataFrame, numerical_columns: list, standardize=True):
    """Z-score standardize numeric feature columns."""
    df = df.copy()
    scaler = None
    if standardize and numerical_columns:
        scaler = StandardScaler()
        present = [c for c in numerical_columns if c in df.columns]
        for col in present:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if present:
            df[present] = scaler.fit_transform(df[present])
    return df, scaler

def discretize_to_hourly_bins(df: pd.DataFrame, id_column="ICUSTAY_ID", time_column="HOUR", value_columns=None):
    """Floor to integer hour bins, keep last per hour, then ffill within stay."""
    df = df.copy()
    df[time_column] = df[time_column].astype(float)
    df[id_column] = df[id_column].astype(str)

    if value_columns is None:
        value_columns = [c for c in df.columns
                         if c not in [id_column, time_column]
                         and not c.endswith("_missing_flag")
                         and np.issubdtype(df[c].dtype, np.number)]

    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    flag_columns = [c for c in df.columns if c.endswith("_missing_flag")]
    relevant_flags = [c for c in flag_columns if c.replace("_missing_flag", "") in value_columns]

    df[time_column + "_BIN"] = np.floor(df[time_column]).astype(int)

    agg = {c: "last" for c in value_columns}
    for f in relevant_flags:
        agg[f] = "max"

    aggregated = df.groupby([id_column, time_column + "_BIN"])[value_columns + relevant_flags].agg(agg).reset_index()

    parts = []
    for icu_id, g in aggregated.groupby(id_column):
        g = g.set_index(time_column + "_BIN")
        full_idx = pd.RangeIndex(start=g.index.min(), stop=g.index.max() + 1, name=time_column + "_BIN")
        g = g.reindex(full_idx).reset_index()
        g[id_column] = icu_id
        parts.append(g)

    result = pd.concat(parts, ignore_index=True)
    result[value_columns] = result.groupby(id_column)[value_columns].ffill()
    result[relevant_flags] = result.groupby(id_column)[relevant_flags].ffill().fillna(0)
    return result.set_index([id_column, time_column + "_BIN"])

def build_sequences(discretized_df: pd.DataFrame, time_window: int):
    """Build padded sequences for first time_window hours; return X, lengths, icu_id_list."""
    discretized_df.index = discretized_df.index.set_levels(
        discretized_df.index.levels[0].astype(int), level="ICUSTAY_ID"
    )
    X_seq_list, icu_id_list = [], []
    for icu_id, g in discretized_df.groupby(level=0):
        g = g[g.index.get_level_values("HOUR_BIN") < time_window]
        if g.empty:
            continue
        g_sorted = g.sort_index(level=1)
        arr = g_sorted.values  # [T, D]
        X_seq_list.append(torch.tensor(arr, dtype=torch.float32))
        icu_id_list.append(int(icu_id))

    if not X_seq_list:
        return None, None, []

    X_padded = pad_sequence(X_seq_list, batch_first=True)
    seq_lengths = torch.tensor([t.shape[0] for t in X_seq_list])
    return X_padded, seq_lengths, icu_id_list

def build_hourly_labels(listfile_csv: str, icu_id_list: list, time_window: int, min_label_hour: int):
    """Create per-hour class labels and mask using listfile."""
    lf = pd.read_csv(listfile_csv)
    lf = lf[(lf["period_length"] >= min_label_hour) & (lf["period_length"] <= time_window)].copy()

    icu_to_idx = {icu: idx for idx, icu in enumerate(icu_id_list)}
    N, T = len(icu_id_list), time_window
    y_hourly = torch.full((N, T), -1, dtype=torch.long)
    hour_mask = torch.zeros((N, T), dtype=torch.bool)

    # bins in hours (same day cut points as LOS)
    hour_bins = [int(x * 24) for x in [0, 1, 2, 3, 4, 5, 7, 14]] + [np.inf]
    lf["y_true_class"] = pd.cut(
        lf["y_true"], bins=hour_bins, labels=list(range(len(hour_bins) - 1))),  # right=False default is True now
    lf["y_true_class"] = lf["y_true_class"].astype(int)

    for _, row in lf.iterrows():
        icu = int(row["ICUSTAY_ID"])
        t = int(row["period_length"])
        cls = int(row["y_true_class"])
        if icu not in icu_to_idx or not (min_label_hour <= t < T):
            continue
        idx = icu_to_idx[icu]
        y_hourly[idx, t] = cls
        hour_mask[idx, t] = True

    return y_hourly, hour_mask

def create_tensors(raw_root: str, time_window: int, task: str, min_label_hour: int, standardize: bool):
    """Main tensor creation entry."""
    preproc_root = os.path.join(raw_root, "preprocessed")
    tens_dir = os.path.join(preproc_root, "tensor")
    ensure_dir(tens_dir)

    dyn_path = os.path.join(preproc_root, "dynamic_data_imputed.csv")
    stat_path = os.path.join(preproc_root, "static_data_encoded.csv")
    listfile_path = os.path.join(preproc_root, "listfile.csv")

    dyn = pd.read_csv(dyn_path)
    stat = pd.read_csv(stat_path)

    # choose value columns present
    present_nums = [c for c in NUMERICAL_COLUMNS if c in dyn.columns]
    present_gcs = [c for c in GCS_COLS if c in dyn.columns]
    value_cols = present_nums + present_gcs

    # optional standardization (numerical only)
    dyn_std, _ = standardize_numerical_columns(dyn, present_nums, standardize=standardize)

    # discretize to hourly bins
    disc = discretize_to_hourly_bins(dyn_std, id_column="ICUSTAY_ID", value_columns=value_cols)

    # build sequences
    X_padded, seq_lengths, icu_id_list = build_sequences(disc, time_window=time_window)
    if X_padded is None:
        print("No sequences found in the chosen time window.")
        return

    # align static by ICUSTAY_ID
    if "ICUSTAY_ID" not in stat.columns:
        raise ValueError("static_data_encoded.csv must contain ICUSTAY_ID.")
    stat["ICUSTAY_ID"] = pd.to_numeric(stat["ICUSTAY_ID"], errors="coerce").astype("Int64")
    stat = stat.dropna(subset=["ICUSTAY_ID"]).set_index("ICUSTAY_ID")
    stat.index = stat.index.astype(int)

    keep_idx = [k for k, icu in enumerate(icu_id_list) if icu in stat.index]
    dropped = len(icu_id_list) - len(keep_idx)
    if dropped:
        print(f"Dropping {dropped} sequences not present in static_data_encoded.csv.")

    X_padded = X_padded[keep_idx]
    seq_lengths = seq_lengths[keep_idx]
    icu_id_list = [icu_id_list[k] for k in keep_idx]

    # build static + targets
    if "LOS" not in stat.columns or "MORTALITY" not in stat.columns:
        raise ValueError("static_data_encoded.csv must include LOS and MORTALITY.")
    static_rows = [stat.loc[i].drop(labels=["LOS", "MORTALITY"]).values for i in icu_id_list]
    static_tensor = torch.tensor(static_rows, dtype=torch.float32)

    y_total = torch.tensor([float(stat.loc[i, "LOS"]) for i in icu_id_list], dtype=torch.float32)
    y_mortality = torch.tensor([float(stat.loc[i, "MORTALITY"]) for i in icu_id_list], dtype=torch.float32)

    # LOS class bins (days)
    los_bins = [0, 1, 2, 3, 4, 5, 7, 14, np.inf]
    y_total_class = np.digitize(y_total.numpy(), bins=los_bins) - 1
    y_total_class = torch.tensor(y_total_class, dtype=torch.long)

    # ----- filename helper with TW suffix -----
    def fp(name: str) -> str:
        return os.path.join(tens_dir, f"{name}_{time_window}.pt")

    # save common outputs (with TW suffix)
    torch.save(X_padded,      fp("X_padded_tensor"))
    torch.save(y_total,       fp("y_total_tensor"))
    torch.save(static_tensor, fp("static_tensor"))
    torch.save(y_total_class, fp("y_total_class_tensor"))
    torch.save(seq_lengths,   fp("seq_lengths"))
    torch.save(icu_id_list,   fp("icu_id_list"))

    if task == "los":
        print("Saved LOS tensors (regression + classification).")
    else:  # multitask
        # hourly labels & mask
        y_hourly, hour_mask = build_hourly_labels(
            listfile_csv=listfile_path,
            icu_id_list=icu_id_list,
            time_window=time_window,
            min_label_hour=min_label_hour,
        )
        # save extra multitask targets (with TW suffix)
        torch.save(y_mortality, fp("y_mortality_tensor"))
        torch.save(y_hourly,    fp("y_hourly_tensor"))
        torch.save(hour_mask,   fp("hour_mask"))
        print("Saved multitask tensors (LOS reg/class + mortality + hourly labels).")

    print(f"Tensors saved in: {tens_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="Create tensors from preprocessed data.")
    p.add_argument("raw_root", type=str, help="Root from events validation (…/preprocessed must exist).")
    p.add_argument("--time_window", type=int, default=48, help="Hours to include (default: 48).")
    p.add_argument("--task", choices=["los", "multitask"], default="multitask",
                   help="Choose 'los' or 'multitask'.")
    p.add_argument("--min_label_hour", type=int, default=5,
                   help="Earliest hour to include in hourly labels (default: 5).")
    p.add_argument("--no_standardize", action="store_true",
                   help="Disable numerical standardization before discretizing.")
    return p.parse_args()

def main():
    args = parse_args()
    create_tensors(
        raw_root=args.raw_root,
        time_window=args.time_window,
        task=args.task,
        min_label_hour=args.min_label_hour,
        standardize=not args.no_standardize,
    )

if __name__ == "__main__":
    main()

