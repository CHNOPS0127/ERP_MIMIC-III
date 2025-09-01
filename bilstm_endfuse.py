#!/usr/bin/env python3
"""
bilstm_endfuse.py
-----------------
BiLSTM + End-Fusion (static MLP) for ICU LOS bucket classification.

- Reuses rnn_utils.py for loading tensors, grouped splits, CV training, test eval, and bootstrapping
- Supports single-config or simple hyperparam sweeps via CLI lists

Required tensors (created by your tensor creation pipeline):
  X_padded_tensor_*.pt
  y_total_class_tensor_*.pt
  static_tensor_*.pt
  seq_lengths_*.pt
  icu_id_list_*.pt

Example (PowerShell, one line):
  python ".\bilstm_endfuse.py" --X_path "D:\..\X_padded_tensor_48.pt" --y_total_class_path "D:\..\y_total_class_tensor_48.pt" --static_path "D:\..\static_tensor_48.pt" --seq_lengths_path "D:\..\seq_lengths_48.pt" --icu_ids_path "D:\..\icu_id_list_48.pt" --results_dir "D:\..\results\BiLSTM-EndFuse" --hidden_dim 128 --dropout 0.3 --num_layers 2 --batch_size 64 --learning_rate 1e-3 --n_splits 5 --epochs 100

You can sweep by passing multiple values (space-separated) for any of:
  --hidden_dim --dropout --num_layers --batch_size --learning_rate --seed
"""

from __future__ import annotations
import os
import json
import copy
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid

from rnn_utils import (
    set_seed, get_device, load_single_task_tensors, make_group_splits,
    train_eval_single_task, train_final_single_task,
    evaluate_single_task_test, bootstrap_single_task,
    save_metrics_table, save_predictions_table, SingleTaskDataset
)

# -------------------- Model --------------------

class BiLSTM_EndFuse(nn.Module):
    """
    Sequence encoder: BiLSTM -> take last forward/backward states, concat (2*hidden_dim)
    Static encoder: small MLP (64-d) with BatchNorm and Dropout
    Head: [seq(2H) || static(64)] -> 16 -> num_classes
    """
    def __init__(self, input_dim: int, static_dim: int, num_classes: int,
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x_padded: torch.Tensor, seq_lengths: torch.Tensor, static_features: torch.Tensor):
        # Pack -> BiLSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x_padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)  # h_n shape = (num_layers*2, B, H)

        # Last fwd & bwd states (because bidirectional=True)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (B, 2H)

        # Static path
        static_z = self.static_mlp(static_features)  # (B, 64)

        # End-fusion
        fused = torch.cat([h_concat, static_z], dim=1)
        logits = self.fc(fused)
        return logits


# -------------------- Utilities --------------------

def model_builder_factory(input_dim: int, static_dim: int, num_classes: int, hp: Dict[str, Any]):
    def build():
        return BiLSTM_EndFuse(
            input_dim=input_dim,
            static_dim=static_dim,
            num_classes=num_classes,
            hidden_dim=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        )
    return build


def derive_results_subdir(results_root: str, hp: Dict[str, Any]) -> str:
    tag = f"hd{hp['hidden_dim']}_nl{hp['num_layers']}_do{hp['dropout']}_bs{hp['batch_size']}_lr{hp['learning_rate']}_seed{hp['seed']}"
    out = os.path.join(results_root, tag)
    os.makedirs(out, exist_ok=True)
    return out


def evaluate_with_probs(model: nn.Module,
                        X: torch.Tensor, y: torch.Tensor, static: torch.Tensor, seq: torch.Tensor,
                        icu_ids: np.ndarray, batch_size: int, device: torch.device,
                        out_csv: str):
    """
    Run model on a dataset and save predictions + per-class probabilities.
    Assumes y are class indices (long).
    """
    ds = SingleTaskDataset(X, seq, static, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    preds, truths, ids, probs = [], [], [], []
    with torch.no_grad():
        for i, (xb, sl, sb, yb) in enumerate(dl):
            xb, sl, sb, yb = xb.to(device), sl.to(device), sb.to(device), yb.to(device)
            logits = model(xb, sl, sb)
            p = torch.softmax(logits, dim=1)
            pred = p.argmax(dim=1)

            preds.extend(pred.detach().cpu().numpy())
            truths.extend(yb.detach().cpu().numpy())
            probs.extend(p.detach().cpu().numpy())

            # reconstruct ICUSTAY_IDs from batch index
            start = i * batch_size
            ids.extend(icu_ids[start : start + len(yb)])

    num_classes = probs[0].shape[0] if probs else 0
    prob_cols = {f"prob_class_{c}": [row[c] for row in probs] for c in range(num_classes)}

    df = pd.DataFrame({
        "ICUSTAY_ID": ids,
        "y_true": truths,
        "y_pred": preds,
        **prob_cols,
    })
    df.to_csv(out_csv, index=False)
    return df


# -------------------- Main training/sweep --------------------

def run_single_config(hp: Dict[str, Any],
                      X: torch.Tensor, y: torch.Tensor, static: torch.Tensor, seq: torch.Tensor, icu_ids: np.ndarray,
                      results_root: str, n_splits: int, epochs: int, device: torch.device) -> Dict[str, Any]:
    set_seed(hp["seed"])

    # Train/test split (grouped by ICU stay)
    tr_idx, te_idx = make_group_splits(icu_ids, test_size=0.2, seed=hp["seed"])
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    sb_tr, sb_te = static[tr_idx], static[te_idx]
    sl_tr, sl_te = seq[tr_idx], seq[te_idx]
    icu_tr, icu_te = icu_ids[tr_idx], icu_ids[te_idx]

    input_dim = X.shape[2]
    static_dim = static.shape[1]
    num_classes = int(y.max().item() + 1)

    model_builder = model_builder_factory(input_dim, static_dim, num_classes, hp)
    out_dir = derive_results_subdir(results_root, hp)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hp, f, indent=2)

    # ---- Cross-validated training (uses early stopping on linear kappa) ----
    folds, fold_preds, best_artifacts, _ = train_eval_single_task(
        model_builder=model_builder,
        X_train=X_tr, y_train=y_tr, static_train=sb_tr, seq_train=sl_tr, icu_train=icu_tr,
        num_classes=num_classes,
        params={"batch_size": hp["batch_size"], "learning_rate": hp["learning_rate"]},
        device=device,
        results_dir=out_dir,
        n_splits=n_splits,
        early_stopping_patience=10,
        max_epochs=epochs,
    )

    # Save CV summary (already written by utils); also save the "best score" snapshot
    torch.save(best_artifacts["best_state_dict"], os.path.join(out_dir, "best_cv_state_dict.pt"))

    # ---- Train final on all training data ----
    final_model = train_final_single_task(
        model_builder, X_tr, y_tr, sb_tr, sl_tr,
        params={"batch_size": hp["batch_size"], "learning_rate": hp["learning_rate"]},
        device=device
    )
    torch.save(final_model.state_dict(), os.path.join(out_dir, "best_final_model.pt"))

    # ---- Evaluate on held-out test ----
    # 1) simple test metrics/preds via helper
    metrics, df_simple = evaluate_single_task_test(
        final_model, X_te, y_te, sb_te, sl_te, icu_te,
        params={"batch_size": hp["batch_size"]},
        device=device,
        results_dir=out_dir
    )
    # 2) predictions WITH probabilities
    df_probs = evaluate_with_probs(
        final_model, X_te, y_te, sb_te, sl_te, icu_te,
        batch_size=hp["batch_size"], device=device,
        out_csv=os.path.join(out_dir, "test_predictions_with_probs.csv"),
    )

    # ---- Bootstrap CIs on test ----
    boot = bootstrap_single_task(
        truths=df_simple["true_class"].tolist(),
        preds=df_simple["predicted_class"].tolist(),
        n_bootstrap=1000, seed=hp["seed"]
    )
    # Save both point metrics and bootstrap summary
    save_metrics_table(metrics | {"boot_" + k: v for k, v in boot.items()},
                       os.path.join(out_dir, "test_metrics_with_bootstrap.csv"))

    print(f"[Done] {out_dir}")
    return {
        "out_dir": out_dir,
        "cv_best_linear_kappa": best_artifacts["best_score"],
        "test_metrics": metrics,
        "bootstrap": boot
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="BiLSTM End-Fusion LOS classifier (uses rnn_utils).")

    # tensor paths
    p.add_argument("--X_path", type=str, required=True)
    p.add_argument("--y_total_class_path", type=str, required=True)
    p.add_argument("--static_path", type=str, required=True)
    p.add_argument("--seq_lengths_path", type=str, required=True)
    p.add_argument("--icu_ids_path", type=str, required=True)

    p.add_argument("--results_dir", type=str, default=None, help="Where to save outputs. Default: <root>/results/BiLSTM-EndFuse")

    # sweep-able hparams (pass one or more values)
    p.add_argument("--hidden_dim", type=int, nargs="+", default=[128])
    p.add_argument("--num_layers", type=int, nargs="+", default=[2])
    p.add_argument("--dropout", type=float, nargs="+", default=[0.3])
    p.add_argument("--batch_size", type=int, nargs="+", default=[64])
    p.add_argument("--learning_rate", type=float, nargs="+", default=[1e-3])
    p.add_argument("--seed", type=int, nargs="+", default=[42])

    # training controls
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)

    args = p.parse_args()

    # derive default results_dir if not provided
    if args.results_dir is None:
        # <.../preprocessed/tensor>/../../results/BiLSTM-EndFuse
        root = os.path.abspath(os.path.join(os.path.dirname(args.X_path), "..", ".."))
        args.results_dir = os.path.join(root, "results", "BiLSTM-EndFuse")
    os.makedirs(args.results_dir, exist_ok=True)

    # load tensors
    X, y, static, seq, icu_ids = load_single_task_tensors(
        X_path=args.X_path,
        y_total_class_path=args.y_total_class_path,
        static_path=args.static_path,
        seq_lengths_path=args.seq_lengths_path,
        icu_ids_path=args.icu_ids_path,
    )

    device = get_device()
    print(f"Device: {device}")

    # build param grid from CLI lists
    grid = ParameterGrid({
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    })

    results_summary: List[Dict[str, Any]] = []
    for i, hp in enumerate(grid, start=1):
        print(f"\n=== Config {i}/{len(grid)}: {hp} ===")
        out = run_single_config(
            hp=hp,
            X=X, y=y, static=static, seq=seq, icu_ids=icu_ids,
            results_root=args.results_dir,
            n_splits=args.n_splits,
            epochs=args.epochs,
            device=device
        )
        results_summary.append({
            **hp,
            "out_dir": out["out_dir"],
            "cv_best_linear_kappa": out["cv_best_linear_kappa"],
            "test_accuracy": out["test_metrics"]["accuracy"],
            "test_linear_kappa": out["test_metrics"]["linear_kappa"],
            "test_quadratic_kappa": out["test_metrics"]["quadratic_kappa"],
            "test_kappa": out["test_metrics"]["kappa"],
        })

    # save sweep summary
    pd.DataFrame(results_summary).to_csv(
        os.path.join(args.results_dir, "sweep_summary.csv"), index=False
    )
    print(f"\nAll done. Summary saved to: {os.path.join(args.results_dir, 'sweep_summary.csv')}")


if __name__ == "__main__":
    main()
