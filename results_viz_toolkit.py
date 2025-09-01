#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Only used for ROC/PR
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


# -------------------------------
# Styling helpers
# -------------------------------
def boxed_axes(ax, box_lw: float = 1.0, tick_len: float = 4.0, tick_w: float = 1.0):
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(box_lw)
    ax.tick_params(direction="in", length=tick_len, width=tick_w)


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir



# 1) Forest Plots

DEMO_ROWS = [
    # full_label, model, horizon, approach, kappa_mean, kappa_low, kappa_high, mad_mean, mad_low, mad_high
    ("Logistic Regression", "Logistic Regression", None,   None,        0.6451, 0.6343, 0.6563, 42.7925, 40.3719, 45.2132),
    ("Random Forest",       "Random Forest",       None,   None,        0.6578, 0.6456, 0.6726, 37.0217, 35.9831, 38.0646),
    ("BiLSTM 48h-EndFuse",      "BiLSTM",              "48h",  "EndFuse", 0.7092, 0.6991, 0.7188, 36.6326, 34.7036, 38.7088),
    ("BiLSTM 48h-AttenFuse",     "BiLSTM",              "48h",  "AttenFuse",0.7126, 0.7032, 0.7219, 37.5885, 35.6437, 39.4021),
    ("BiLSTM 72h-EndFuse",      "BiLSTM",              "72h",  "EndFuse", 0.8142, 0.8070, 0.8210, 30.0313, 28.2085, 31.9896),
    ("BiLSTM 72h-AttenFuse",     "BiLSTM",              "72h",  "AttenFuse",0.8235, 0.8165, 0.8303, 29.1001, 28.6201, 29.5801),
    ("BiGRUs 48h-EndFuse",      "BiGRUs",              "48h",  "EndFuse", 0.7102, 0.7006, 0.7201, 37.9020, 36.9820, 38.8220),
    ("BiGRUs 48h-AttenFuse",     "BiGRUs",              "48h",  "AttenFuse",0.7209, 0.7103, 0.7311, 31.7227, 30.0889, 33.3817),
    ("BiGRUs 72h-EndFuse",      "BiGRUs",              "72h",  "EndFuse", 0.8156, 0.8085, 0.8227, 28.8796, 27.1169, 30.7284),
    ("BiGRUs 72h-AttenFuse",     "BiGRUs",              "72h",  "AttenFuse",0.8168, 0.8097, 0.8239, 34.1264, 32.2444, 36.0334),
    ("Multitask 48h", "Multitask", "48h", "Total", 0.7171, 0.7077, 0.7272, 33.3635, 31.6017, 35.2021),
]


def _compact_label(row: pd.Series) -> str:
    model_map = {"Logistic Regression": "LR", "Random Forest": "RF",
                 "BiLSTM": "BiLSTM", "BiGRUs": "BiGRU", "Multitask": "Multitask"}
    if row["model"] in ["Logistic Regression", "Random Forest"]:
        return model_map[row["model"]]
    if row["model"] == "Multitask":
        return f"{model_map['Multitask']}-Total-{str(row['horizon']).replace('h','h')}"
    roman = {"EndFuse": "EndFuse", "AttenFuse": "AttenFuse"}.get(row["approach"], row["approach"])
    return f"{model_map[row['model']]}-{roman}-{str(row['horizon']).replace('h','h')}"


def load_forest_df(csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(DEMO_ROWS, columns=[
            "full_label","model","horizon","approach",
            "kappa_mean","kappa_low","kappa_high",
            "mad_mean","mad_low","mad_high"
        ])
    if "label" not in df.columns:
        df["label"] = df.apply(_compact_label, axis=1)
    return df


def boxed_forest(plot_df: pd.DataFrame, x: str, lo: str, hi: str,
                 title: str, xlabel: str, sort_ascending: bool,
                 baseline_value: Optional[float] = None, outfile: Optional[Path] = None,
                 note: Optional[str] = None) -> None:
    plot_df = plot_df.sort_values(x, ascending=sort_ascending).reset_index(drop=True)
    y = np.arange(len(plot_df))
    xerr = np.vstack([plot_df[x] - plot_df[lo], plot_df[hi] - plot_df[x]])

    fig, ax = plt.subplots(figsize=(9, max(5, 0.5*len(plot_df)+1)))
    ax.errorbar(plot_df[x], y, xerr=xerr, fmt='o', capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    boxed_axes(ax)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    if baseline_value is not None:
        ax.axvline(baseline_value, linestyle="--", linewidth=1)

    if not sort_ascending:
        ax.invert_yaxis()

    if note:
        fig.text(0.5, -0.02, note, ha='center', fontsize=9, fontstyle='italic')

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def cmd_forest(args: argparse.Namespace) -> None:
    outdir = ensure_outdir(args.outdir)
    df = load_forest_df(args.csv)

    # Baseline from a row matching baseline_label (default: Logistic Regression)
    baseline_val = None
    if args.baseline_label and args.which == "kappa":
        m = df[df["full_label"].astype(str).str.lower() == args.baseline_label.lower()]
        if len(m):
            baseline_val = float(m["kappa_mean"].iloc[0])
    if args.baseline_label and args.which == "mad":
        m = df[df["full_label"].astype(str).str.lower() == args.baseline_label.lower()]
        if len(m):
            baseline_val = float(m["mad_mean"].iloc[0])

    note = "N.B. ‘Multitask-Total-48h’ represents Total LOS from a 48-hour observation window in a multitask framework."
    if args.which == "kappa":
        outfile = outdir / "forest_kappa_boxed_compact.png"
        boxed_forest(df, "kappa_mean", "kappa_low", "kappa_high",
                     title="Kappa (95% CI)",
                     xlabel="Linearly Weighted Kappa",
                     sort_ascending=False,
                     baseline_value=baseline_val,
                     outfile=outfile,
                     note=note)
        print(f"[saved] {outfile}")
    elif args.which == "mad":
        outfile = outdir / "forest_mad_boxed_compact.png"
        boxed_forest(df, "mad_mean", "mad_low", "mad_high",
                     title="MAD (95% CI)",
                     xlabel="Mean Absolute Deviation (hours)",
                     sort_ascending=True,
                     baseline_value=baseline_val,
                     outfile=outfile,
                     note=note)
        print(f"[saved] {outfile}")
    else:
        raise ValueError("--which must be 'kappa' or 'mad'")



# 2) Pairwise win-rate heatmap

def load_prediction_maps(map_csv: str) -> List[dict]:
    """
    Expects a CSV with columns: path,id_col,true_col,pred_col,label
    """
    df = pd.read_csv(map_csv)
    required = {"path","id_col","true_col","pred_col","label"}
    missing = required - set(c.lower() for c in df.columns)
    # accept case-insensitive by normalizing
    cols = {c.lower():c for c in df.columns}
    if missing:
        raise ValueError(f"Mapping CSV must include columns: {sorted(required)}")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "path": r[cols["path"]],
            "id": r[cols["id_col"]],
            "true": r[cols["true_col"]],
            "pred": r[cols["pred_col"]],
            "label": r[cols["label"]],
        })
    return rows


def pairwise_winrate(map_csv: str, outdir: Path) -> None:
    specs = load_prediction_maps(map_csv)
    # Load and harmonize
    dfs = []
    for spec in specs:
        f = Path(spec["path"]).expanduser()
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")
        use_cols = [spec["id"], spec["true"], spec["pred"]]
        df = pd.read_csv(f, usecols=use_cols).copy()
        df.columns = ["ID", "true", spec["label"]]
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["ID", "true"], how="inner")

    if merged.empty:
        raise ValueError("No common IDs across all files (after matching true labels).")

    y_true = merged["true"].to_numpy().astype(int)
    model_labels = [spec["label"] for spec in specs]
    pred_mat = np.vstack([merged[lbl].to_numpy().astype(int) for lbl in model_labels])
    K, N = pred_mat.shape

    errors = np.abs(pred_mat - y_true.reshape(1, -1))
    M = np.full((K, K), np.nan, dtype=float)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            M[i, j] = 100.0 * np.mean(errors[i] <= errors[j])

    # Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, vmin=0, vmax=100, cmap="Greens", interpolation="nearest")

    for i in range(K):
        for j in range(K):
            text = "-" if (i == j or np.isnan(M[i, j])) else f"{M[i, j]:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="white")

    ax.set_xticks(np.arange(K)); ax.set_yticks(np.arange(K))
    ax.set_xticklabels(model_labels, rotation=30,  ha="right")
    ax.set_yticklabels(model_labels)

    boxed_axes(ax)
    # table-like grid
    ax.set_xticks(np.arange(-.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-.5, K, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win rate (%): err(row) ≤ err(col)")

    ax.set_title(f"Pairwise win-rate (off-by-k), N={N} common IDs", pad=14)
    fig.tight_layout()

    out_png = outdir / "pairwise_winrate_off_by_k.png"
    out_csv = outdir / "pairwise_winrate_off_by_k.csv"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    pd.DataFrame(M, index=model_labels, columns=model_labels).to_csv(out_csv, float_format="%.2f")
    plt.close(fig)
    print(f"[saved] {out_png}\n[saved] {out_csv}")


def cmd_pairwise(args: argparse.Namespace) -> None:
    outdir = ensure_outdir(args.outdir)
    pairwise_winrate(args.map, outdir)



# 3) Macro ROC & PR (multi-class)

def load_probs(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a predictions CSV and return:
        y_true: (N,) integer labels
        probs:  (N, C) predicted probabilities ordered by class 0..C-1

    Auto-detects:
      - true label column: one of {y_true, true, true_class, label, labels, target}
      - prob columns: p{c}, prob_class_{c}, prob_{c}, softmax_{c}
    """
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    # true labels
    true_candidates = [c for c in cols if c.lower() in
                       ("y_true", "true", "true_class", "label", "labels", "target")]
    if not true_candidates:
        raise ValueError(f"No true label column found in {csv_path}. "
                         f"Columns: {cols[:10]}...")
    y_true = df[true_candidates[0]].to_numpy()

    # probability columns
    prob_map = {}
    for c in cols:
        m = re.match(r"^p(\d+)$", c)                # p0, p1, ...
        n = re.match(r"^prob_class_(\d+)$", c)      # prob_class_0, ...
        k = re.match(r"^prob_(\d+)$", c)            # prob_0, ...
        s = re.match(r"^softmax_(\d+)$", c)         # softmax_0, ...
        if m: prob_map[int(m.group(1))] = c
        elif n: prob_map[int(n.group(1))] = c
        elif k: prob_map[int(k.group(1))] = c
        elif s: prob_map[int(s.group(1))] = c

    if not prob_map:
        raise ValueError(f"No probability columns found in {csv_path}.")

    C = max(prob_map.keys()) + 1
    probs = df[[prob_map[i] for i in range(C)]].to_numpy(dtype=float)
    return y_true, probs


def macro_roc(y_true: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a macro-average ROC curve for a multi-class problem.
    Returns (fpr_grid, mean_tpr, macro_auc).
    """
    C = probs.shape[1]
    present = [c for c in range(C) if (y_true == c).any()]
    y_bin = label_binarize(y_true, classes=np.arange(C))

    fprs, tprs = [], []
    for c in present:
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        fprs.append(fpr)
        tprs.append(tpr)

    all_fpr = np.unique(np.concatenate(fprs))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fprs, tprs):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(fprs)
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc


def macro_pr(y_true: np.ndarray, probs: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a macro-average PR curve for a multi-class problem by
    averaging interpolated precision over a fixed recall grid.
    Returns (recall_grid, mean_precision, macro_AP).
    """
    C = probs.shape[1]
    present = [c for c in range(C) if (y_true == c).any()]
    y_bin = label_binarize(y_true, classes=np.arange(C))

    recall_grid = np.linspace(0, 1, n_points)
    prec_curves, ap_list = [], []
    for c in present:
        prec, rec, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
        # interpolate onto a monotonically increasing recall grid
        prec_interp = np.interp(recall_grid, rec[::-1], prec[::-1])
        prec_curves.append(prec_interp)
        ap_list.append(average_precision_score(y_bin[:, c], probs[:, c]))

    mean_prec = np.mean(prec_curves, axis=0)
    macro_ap = float(np.mean(ap_list))
    return recall_grid, mean_prec, macro_ap


def cmd_macro(args: argparse.Namespace) -> None:
    outdir = ensure_outdir(args.outdir)
    # mapping CSV: columns name,path
    map_df = pd.read_csv(args.map)
    name_col = [c for c in map_df.columns if c.lower() == "name"][0]
    path_col = [c for c in map_df.columns if c.lower() == "path"][0]

    # Plot ROC
    plt.figure(figsize=(8, 8))
    legend_items = []
    for _, r in map_df.iterrows():
        name, path = r[name_col], r[path_col]
        try:
            y, p = load_probs(path)
            fpr, tpr, auc_macro = macro_roc(y, p)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_macro:.2f})", linewidth=1.3)
        except Exception as e:
            print(f"[WARN] Skipping {name}: {e}")
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-average ROC Curve")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    out_png = outdir / "macro_roc_all_models.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

    # Plot PR
    plt.figure(figsize=(8, 8))
    for _, r in map_df.iterrows():
        name, path = r[name_col], r[path_col]
        try:
            y, p = load_probs(path)
            rec, prec, ap_macro = macro_pr(y, p)
            plt.plot(rec, prec, label=f"{name} (AP={ap_macro:.2f})", linewidth=1.3)
        except Exception as e:
            print(f"[WARN] Skipping {name}: {e}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro-average Precision-Recall Curve")
    ax = plt.gca()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    out_png = outdir / "macro_pr_all_models.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")



# 4) Per-class ROC & PR

PRETTY_LABELS: Dict[object, str] = {
    # e.g., 0: "0–1d", 1: "1–2d", ...
}


def style_axes(ax):
    boxed_axes(ax, box_lw=1.0, tick_len=5.0, tick_w=1.0)


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "y_true","true","label","labels","y","target","class",
        "true_label","true_class","ground_truth",
        "LOS_BUCKET","los_bucket","gold","truth"
    ]
    for c in candidates:
        if c in df.columns: return c
    one_hot = [c for c in df.columns if c.lower().startswith(("true_","y_true_"))]
    if one_hot:
        df["__derived_true_label__"] = df[one_hot].values.argmax(axis=1)
        return "__derived_true_label__"
    return None


def detect_probability_columns(df: pd.DataFrame) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    prob_cols = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s)==0: continue
        if ((s>=0)&(s<=1)).mean()>0.95 and s.var()>0:
            prob_cols.append(c)
    preferred = [c for c in prob_cols if any(k in c.lower() for k in ("prob","pred","p_","p("))]
    return preferred if preferred else prob_cols


def map_classes_to_prob_columns(classes: List[object], prob_cols: List[str]) -> Dict[object, str]:
    def norm(x): return str(x).strip().lower()
    lc_map = {c.lower(): c for c in prob_cols}
    class_to_prob = {}
    for cls in classes:
        n = norm(cls); matched = None
        for k_lower, orig in lc_map.items():
            if k_lower.endswith(f"_{n}") or k_lower.endswith(f"-{n}") or k_lower==n:
                matched = orig; break
            m = re.search(r"(\d+)$", k_lower)
            if m and m.group(1)==str(cls):
                matched = orig; break
        if matched: class_to_prob[cls]=matched
    if len(class_to_prob)!=len(classes) and len(prob_cols)>=len(classes):
        chosen = sorted(prob_cols, key=lambda x: str(x))[:len(classes)]
        class_to_prob = {cls: pc for cls, pc in zip(classes, chosen)}
    return class_to_prob


def pretty_name(x): return PRETTY_LABELS.get(x, x)


def cmd_perclass(args: argparse.Namespace) -> None:
    outdir = ensure_outdir(args.outdir)
    df = pd.read_csv(args.file)

    label_col = detect_label_column(df)
    if label_col is None:
        raise ValueError("No true-label column found.")
    prob_cols = detect_probability_columns(df)
    if not prob_cols:
        raise ValueError("No probability columns detected (numeric in [0,1]).")

    classes = sorted(pd.Series(df[label_col].unique()).dropna().tolist(), key=lambda x: str(x))
    class_to_prob = map_classes_to_prob_columns(classes, prob_cols)
    if not class_to_prob:
        raise ValueError("Could not map classes to probability columns.")

    # ROC overlay
    roc_path = outdir / "roc_per_class_overlay.png"
    plt.figure(figsize=(10, 7)); ax = plt.gca()
    roc_rows = []
    for cls in classes:
        pcol = class_to_prob[cls]
        y_true = (df[label_col]==cls).astype(int).values
        y_score = df[pcol].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{pretty_name(cls)} (AUC={roc_auc:.3f})")
        roc_rows.append({"class": cls, "prob_column": pcol, "roc_auc": roc_auc})
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC (one-vs-rest)")
    ax.legend(loc="lower right", ncol=2, fontsize="medium", frameon=True)
    style_axes(ax); plt.tight_layout(); plt.savefig(roc_path, dpi=200); plt.close()
    print(f"[saved] {roc_path}")

    # PR overlay
    pr_path = outdir / "pr_per_class_overlay.png"
    plt.figure(figsize=(10, 7)); ax = plt.gca()
    pr_rows = []
    for cls in classes:
        pcol = class_to_prob[cls]
        y_true = (df[label_col]==cls).astype(int).values
        y_score = df[pcol].values
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{pretty_name(cls)} (AP={ap:.3f})")
        pr_rows.append({"class": cls, "prob_column": pcol, "average_precision": ap})
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class Precision–Recall (one-vs-rest)")
    ax.legend(loc="lower left", ncol=2, fontsize="medium", frameon=True)
    style_axes(ax); plt.tight_layout(); plt.savefig(pr_path, dpi=200); plt.close()
    print(f"[saved] {pr_path}")

    # Save metrics
    metrics = pd.merge(pd.DataFrame(roc_rows), pd.DataFrame(pr_rows),
                       on=["class","prob_column"], how="outer").sort_values(
                           "class", key=lambda s: s.astype(str)).reset_index(drop=True)
    metrics_path = outdir / "per_class_metrics_overlay.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[saved] {metrics_path}")



# 5) Attention/Importance visualizations

def cmd_attention(args: argparse.Namespace) -> None:
    outdir = ensure_outdir(args.outdir)

    # Temporal attention
    if args.temporal_attn:
        attn = np.load(args.temporal_attn)
        if attn.ndim == 2:  # (N, T)
            attn_mean = attn.mean(axis=0)
        else:  # (T,)
            attn_mean = attn
        plt.figure(figsize=(12, 4))
        plt.plot(attn_mean)
        plt.title("Average Temporal Attention Weights Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Average Attention Weight")
        plt.grid(True)
        out = outdir / "temporal_attention_mean.png"
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
        print(f"[saved] {out}")

    # Channel-wise attention / feature weights
    if args.channel_weights:
        cw = np.load(args.channel_weights).astype(float)
        names = None
        if args.channel_names and Path(args.channel_names).exists():
            with open(args.channel_names, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            if len(names) != len(cw):
                print(f"[WARN] channel_names length {len(names)} != weights length {len(cw)}; ignoring names.")
                names = None
        plt.figure(figsize=(max(12, len(cw)*0.25), 8))
        idx = np.arange(len(cw))
        plt.bar(idx, cw)
        if names:
            plt.xticks(idx, names, rotation=90)
        else:
            plt.xticks(idx, [f"f{i}" for i in idx], rotation=90)
        plt.ylabel("Attention weight")
        plt.title("Average Channel-wise Attention Weights")
        plt.tight_layout()
        out = outdir / "channel_attention_weights.png"
        plt.savefig(out, dpi=200); plt.close()
        print(f"[saved] {out}")

    # Static feature gates
    if args.static_gates:
        sg = np.load(args.static_gates).astype(float)
        names = None
        if args.static_names and Path(args.static_names).exists():
            with open(args.static_names, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            if len(names) != len(sg):
                print(f"[WARN] static_names length {len(names)} != gates length {len(sg)}; ignoring names.")
                names = None
        plt.figure(figsize=(max(12, len(sg)*0.25), 8))
        idx = np.arange(len(sg))
        plt.bar(idx, sg)
        if names:
            plt.xticks(idx, names, rotation=90)
        else:
            plt.xticks(idx, [f"s{i}" for i in idx], rotation=90)
        plt.ylabel("Average gate (0–1)")
        plt.title("Static Feature Importance (Gates)")
        plt.tight_layout()
        out = outdir / "static_feature_gates.png"
        plt.savefig(out, dpi=200); plt.close()
        print(f"[saved] {out}")



# Main Execution

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproducible visualization toolkit for LOS model results.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Forest
    pf = sub.add_parser("forest", help="Forest plot for Kappa or MAD.")
    pf.add_argument("--csv", type=str, default=None, help="CSV with forest data. If omitted, uses built-in demo rows.")
    pf.add_argument("--which", choices=["kappa","mad"], required=True, help="Which metric to plot.")
    pf.add_argument("--baseline_label", type=str, default="Logistic Regression", help="Row to use as vertical baseline")
    pf.add_argument("--outdir", type=str, required=True, help="Output directory")
    pf.set_defaults(func=cmd_forest)

    # Pairwise
    pp = sub.add_parser("pairwise", help="Pairwise win-rate heatmap from predictions map CSV.")
    pp.add_argument("--map", type=str, required=True, help="CSV mapping with columns: path,id_col,true_col,pred_col,label")
    pp.add_argument("--outdir", type=str, required=True, help="Output directory")
    pp.set_defaults(func=cmd_pairwise)

    # Macro
    pm = sub.add_parser("macro", help="Macro-average ROC/PR from probability files (mapping CSV with columns: name,path).")
    pm.add_argument("--map", type=str, required=True, help="CSV mapping of model name to probability CSV path")
    pm.add_argument("--outdir", type=str, required=True, help="Output directory")
    pm.set_defaults(func=cmd_macro)

    # Per-class
    pc = sub.add_parser("perclass", help="Per-class ROC/PR from a single probability CSV.")
    pc.add_argument("--file", type=str, required=True, help="Probability CSV path")
    pc.add_argument("--outdir", type=str, required=True, help="Output directory")
    pc.set_defaults(func=cmd_perclass)

    # Attention
    pa = sub.add_parser("attention", help="Optional attention/importance visualizations from saved arrays.")
    pa.add_argument("--temporal_attn", type=str, help=".npy file with (T,) or (N,T) temporal attention")
    pa.add_argument("--channel_weights", type=str, help=".npy file with (C,) channel-wise weights")
    pa.add_argument("--channel_names", type=str, help="txt file with C lines for time-series feature names")
    pa.add_argument("--static_gates", type=str, help=".npy file with (S,) static feature gate values")
    pa.add_argument("--static_names", type=str, help="txt file with S lines for static feature names")
    pa.add_argument("--outdir", type=str, required=True, help="Output directory")
    pa.set_defaults(func=cmd_attention)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# ---------- Helper to generate macro curves from a map CSV ----------
def macro_from_map(map_csv: str, outdir: Path) -> None:
    map_df = pd.read_csv(map_csv)
    name_col = [c for c in map_df.columns if c.lower() == "name"][0]
    path_col = [c for c in map_df.columns if c.lower() == "path"][0]

    # ROC
    plt.figure(figsize=(8, 8))
    for _, r in map_df.iterrows():
        name, path = r[name_col], r[path_col]
        try:
            y, p = load_probs(path)
            fpr, tpr, auc_macro = macro_roc(y, p)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_macro:.2f})", linewidth=1.3)
        except Exception as e:
            print(f"[WARN] Skipping {name}: {e}")
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-average ROC Curve")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    out_png = outdir / "macro_roc_all_models.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

    # PR
    plt.figure(figsize=(8, 8))
    for _, r in map_df.iterrows():
        name, path = r[name_col], r[path_col]
        try:
            y, p = load_probs(path)
            rec, prec, ap_macro = macro_pr(y, p)
            plt.plot(rec, prec, label=f"{name} (AP={ap_macro:.2f})", linewidth=1.3)
        except Exception as e:
            print(f"[WARN] Skipping {name}: {e}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro-average Precision-Recall Curve")
    ax = plt.gca()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    out_png = outdir / "macro_pr_all_models.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")


# ---------- Run in one command ----------
def cmd_all(args: argparse.Namespace) -> None:
    outdir = ensure_outdir(args.outdir)
    print(f"[info] Output directory: {outdir}")

    # 1) Forest plots (Kappa & MAD)
    try:
        df = load_forest_df(args.forest_csv)
        # baselines
        baseline_val_kappa = baseline_val_mad = None
        if args.baseline_label:
            m = df[df["full_label"].astype(str).str.lower() == args.baseline_label.lower()]
            if len(m):
                baseline_val_kappa = float(m["kappa_mean"].iloc[0])
                baseline_val_mad   = float(m["mad_mean"].iloc[0])
        note = "N.B. ‘Multitask-Total-48h’ represents Total LOS from a 48-hour observation window in a multitask framework."
        boxed_forest(df, "kappa_mean", "kappa_low", "kappa_high",
                     title="Kappa (95% CI)", xlabel="Linearly Weighted Kappa",
                     sort_ascending=False, baseline_value=baseline_val_kappa,
                     outfile=outdir / "forest_kappa_boxed_compact.png", note=note)
        print(f"[saved] {outdir / 'forest_kappa_boxed_compact.png'}")

        boxed_forest(df, "mad_mean", "mad_low", "mad_high",
                     title="MAD (95% CI)", xlabel="Mean Absolute Deviation (hours)",
                     sort_ascending=True, baseline_value=baseline_val_mad,
                     outfile=outdir / 'forest_mad_boxed_compact.png', note=note)
        print(f"[saved] {outdir / 'forest_mad_boxed_compact.png'}")
    except Exception as e:
        print(f"[WARN] Forest plots skipped: {e}")

    # 2) Pairwise win-rate heatmap
    if args.pairwise_map:
        try:
            pairwise_winrate(args.pairwise_map, outdir)
        except Exception as e:
            print(f"[WARN] Pairwise heatmap skipped: {e}")
    else:
        print("[info] Pairwise heatmap skipped (no --pairwise_map provided).")

    # 3) Macro ROC & PR (multi-model)
    if args.macro_map:
        try:
            macro_from_map(args.macro_map, outdir)
        except Exception as e:
            print(f"[WARN] Macro ROC/PR skipped: {e}")
    else:
        print("[info] Macro ROC/PR skipped (no --macro_map provided).")

    # 4) Per-class ROC & PR (single file)
    if args.perclass_file:
        try:
            # Reuse per-class internals
            pc_ns = argparse.Namespace(file=args.perclass_file, outdir=str(outdir))
            cmd_perclass(pc_ns)
        except Exception as e:
            print(f"[WARN] Per-class ROC/PR skipped: {e}")
    else:
        print("[info] Per-class ROC/PR skipped (no --perclass_file provided).")


# ---- Extend CLI with 'all' command ----
_old_build_parser = build_parser
def build_parser() -> argparse.ArgumentParser:
    p = _old_build_parser()
    sub = p._subparsers._group_actions[0]

    pall = p.add_subparsers(dest="cmd", required=True)

    # The previous code built subparsers already; we can't easily access them here.
    # Instead, re-create a new parser and merge is complex. Easiest: add a fresh subparser via the existing 'sub' action.
    all_parser = sub.add_parser("all", help="Run all plots (forest, pairwise, macro, perclass) in one go.")
    all_parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    all_parser.add_argument("--forest_csv", type=str, default=None, help="CSV for forest plots (if omitted, uses demo rows)")
    all_parser.add_argument("--baseline_label", type=str, default="Logistic Regression", help="Baseline row label for vertical line")
    all_parser.add_argument("--pairwise_map", type=str, default=None, help="Mapping CSV for pairwise heatmap")
    all_parser.add_argument("--macro_map", type=str, default=None, help="Mapping CSV (name,path) for macro ROC/PR")
    all_parser.add_argument("--perclass_file", type=str, default=None, help="Single probability CSV for per-class curves")
    all_parser.set_defaults(func=cmd_all)
    return p
