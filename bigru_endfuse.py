#!/usr/bin/env python3
r"""
bigru_endfuse.py — BiGRU + EndFuse (static fusion) for LOS bucket classification.

Requires: rnn_utils.py in the same folder (or importable on PYTHONPATH).

Inputs (tensor files you already created):
  --X_path              e.g. ...\preprocessed\tensor\X_padded_tensor_48.pt
  --y_total_class_path  e.g. ...\preprocessed\tensor\y_total_class_tensor_48.pt
  --static_path         e.g. ...\preprocessed\tensor\static_tensor_48.pt
  --seq_lengths_path    e.g. ...\preprocessed\tensor\seq_lengths_48.pt
  --icu_ids_path        e.g. ...\preprocessed\tensor\icu_id_list_48.pt

Outputs:
  <results_dir>\ 
    - los_cv_metrics_summary.csv
    - los_cv_predictions.csv
    - best_cv_model.pt
    - best_test_predictions.csv
    - test_metrics_summary.csv
    - test_bootstrap_summary.csv
"""

import os
import argparse
import copy
from typing import Dict, Any

import torch
import torch.nn as nn

from rnn_utils import (
    set_seed, get_device, load_single_task_tensors, make_group_splits,
    train_eval_single_task, train_final_single_task, evaluate_single_task_test,
    bootstrap_single_task, save_metrics_table
)


# -------------------- Model --------------------

class BiGRU_EndFuse(nn.Module):
    """BiGRU over time-series + small MLP on static features, fused at the end."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        static_dim: int,
        num_classes: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Encode static features
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Classifier on [GRU_last_fw_bw | static_64]
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x_padded, seq_lengths, static_features):
        # Pack (handles variable-length)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)            # h_n: (num_layers*2, B, H)
        h_forward = h_n[-2]                  # (B, H)
        h_backward = h_n[-1]                 # (B, H)
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (B, 2H)

        static_enc = self.static_mlp(static_features)         # (B, 64)
        fused = torch.cat([h_concat, static_enc], dim=1)      # (B, 2H+64)
        logits = self.fc(fused)                               # (B, C)
        return logits


# -------------------- Runner --------------------

def build_model_fn(input_dim: int, static_dim: int, num_classes: int, hp: Dict[str, Any]):
    def _builder():
        return BiGRU_EndFuse(
            input_dim=input_dim,
            hidden_dim=hp["hidden_dim"],
            static_dim=static_dim,
            num_classes=num_classes,
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        )
    return _builder


def main():
    p = argparse.ArgumentParser(description="BiGRU-EndFuse LOS classifier (uses rnn_utils).")
    # Required tensor paths
    p.add_argument("--X_path", required=True, type=str)
    p.add_argument("--y_total_class_path", required=True, type=str)
    p.add_argument("--static_path", required=True, type=str)
    p.add_argument("--seq_lengths_path", required=True, type=str)
    p.add_argument("--icu_ids_path", required=True, type=str)

    # Output + runtime
    p.add_argument("--results_dir", required=True, type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.20)

    # Model / training hyperparams (single config)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--num_classes", type=int, default=8)

    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ---- Load tensors
    X, y_class, static, seq_lengths, icu_ids = load_single_task_tensors(
        X_path=args.X_path,
        y_total_class_path=args.y_total_class_path,
        static_path=args.static_path,
        seq_lengths_path=args.seq_lengths_path,
        icu_ids_path=args.icu_ids_path,
    )

    # ---- Train/test split by ICUSTAY_ID groups
    tr_idx, te_idx = make_group_splits(icu_ids, test_size=args.test_size, seed=args.seed)

    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y_class[tr_idx], y_class[te_idx]
    static_train, static_test = static[tr_idx], static[te_idx]
    seq_train, seq_test = seq_lengths[tr_idx], seq_lengths[te_idx]
    icu_train, icu_test = icu_ids[tr_idx], icu_ids[te_idx]

    input_dim = X.shape[2]
    static_dim = static.shape[1]
    num_classes = args.num_classes

    # ---- One config (you can grid outside if you want)
    hp = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    print(f"\n=== Config === {hp}")

    model_fn = build_model_fn(input_dim=input_dim, static_dim=static_dim,
                              num_classes=num_classes, hp=hp)

    # ---- CV training (on train set)
    folds, fold_preds, best_artifacts, _ = train_eval_single_task(
        model_builder=model_fn,
        X_train=X_train, y_train=y_train, static_train=static_train, seq_train=seq_train, icu_train=icu_train,
        num_classes=num_classes,
        params=hp,
        device=device,
        results_dir=args.results_dir,
        n_splits=args.n_splits,
        early_stopping_patience=args.early_stopping_patience,
        max_epochs=args.max_epochs,
    )

    # Save best CV model weights
    best_state = best_artifacts["best_state_dict"]
    torch.save(best_state, os.path.join(args.results_dir, "best_cv_model.pt"))

    # ---- Train final model on full train set
    final_model = train_final_single_task(
        model_builder=model_fn,
        X_train=X_train, y_train=y_train, static_train=static_train, seq_train=seq_train,
        params=hp, device=device
    )

    # ---- Test evaluation (saves best_test_predictions.csv & test_metrics_summary.csv)
    test_metrics, df_test = evaluate_single_task_test(
        model=final_model,
        X_test=X_test, y_test=y_test, static_test=static_test, seq_test=seq_test,
        icu_test=icu_test, params=hp, device=device, results_dir=args.results_dir
    )
    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # ---- Bootstrap CIs on test
    boots = bootstrap_single_task(
        truths=df_test["true_class"].tolist(),
        preds=df_test["predicted_class"].tolist(),
        n_bootstrap=1000,
        seed=args.seed,
    )
    save_metrics_table(boots, os.path.join(args.results_dir, "test_bootstrap_summary.csv"))
    print("\n=== Bootstrap (95% CI) ===")
    for k in ["kappa", "linear_kappa", "quadratic_kappa", "accuracy"]:
        mean_key = f"{k}_mean"
        ci_key = f"{k}_ci"
        lo, hi = boots[ci_key]
        print(f"{k}: {boots[mean_key]:.4f}  CI[{lo:.4f}, {hi:.4f}]")

    print("\nDone. Artifacts in:", args.results_dir)


if __name__ == "__main__":
    main()
