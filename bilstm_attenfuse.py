#!/usr/bin/env python3
# bilstm_endfuse.py
r"""
BiLSTM-EndFuse: LOS bucket classification with sequence + static feature fusion.

This script:
  1) Loads tensors (X, y_class, static, seq_lengths, icu_ids)
  2) Makes a group-aware train/test split (ICUSTAY_ID groups)
  3) Runs 5-fold CV on the train set using rnn_utils.train_eval_single_task
  4) Trains a final model on the full train set
  5) Evaluates on the held-out test set and saves metrics/predictions
  6) Bootstraps test metrics for 95% CIs

Requirements:
  - rnn_utils.py in the same folder (or on your PYTHONPATH)
  - PyTorch, scikit-learn, pandas, numpy

Example (PowerShell):
  python .\bilstm_endfuse.py `
    --X_path             "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects\preprocessed\tensor\X_padded_tensor_48.pt" `
    --y_total_class_path "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects\preprocessed\tensor\y_total_class_tensor_48.pt" `
    --static_path        "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects\preprocessed\tensor\static_tensor_48.pt" `
    --seq_lengths_path   "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects\preprocessed\tensor\seq_lengths_48.pt" `
    --icu_ids_path       "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects\preprocessed\tensor\icu_id_list_48.pt" `
    --results_dir        "D:\DATA72000ERP\mimic3-data\data\trial\random_1000_subjects\results\BiLSTM-EndFuse" `
    --hidden_dim 128 --dropout 0.3 --num_layers 2 --batch_size 64 --learning_rate 1e-3 --seed 42
"""

import os
import argparse
import json
import copy
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rnn_utils import (  # make sure rnn_utils.py is importable
    set_seed,
    get_device,
    load_single_task_tensors,
    make_group_splits,
    train_eval_single_task,
    train_final_single_task,
    evaluate_single_task_test,
    bootstrap_single_task,
)

# --------------------------- Model ---------------------------

class BiLSTM_LOS_Classifier(nn.Module):
    """
    End-fusion: concatenate last BiLSTM hidden state (fwd+bwd) with a
    transformed static feature vector, then classify.
    """
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
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x_padded, seq_lengths, static_features):
        # pack -> BiLSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x_padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers*2, B, H)

        # last fwd/bwd layers
        h_forward = h_n[-2]  # (B, H)
        h_backward = h_n[-1]  # (B, H)
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (B, 2H)

        static_z = self.static_mlp(static_features)  # (B, 64)

        fused = torch.cat([h_concat, static_z], dim=1)  # (B, 2H+64)
        logits = self.fc(fused)
        return logits


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train/Evaluate BiLSTM-EndFuse for LOS buckets.")
    # data
    p.add_argument("--X_path", type=str, required=True)
    p.add_argument("--y_total_class_path", type=str, required=True)
    p.add_argument("--static_path", type=str, required=True)
    p.add_argument("--seq_lengths_path", type=str, required=True)
    p.add_argument("--icu_ids_path", type=str, required=True)

    # results
    p.add_argument("--results_dir", type=str, required=True)

    # model/training
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)

    # CV & reproducibility
    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# --------------------------- Runner ---------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Device: {device.type}")

    # Load tensors
    X, y_class, static, seq_lengths, icu_ids = load_single_task_tensors(
        X_path=args.X_path,
        y_total_class_path=args.y_total_class_path,
        static_path=args.static_path,
        seq_lengths_path=args.seq_lengths_path,
        icu_ids_path=args.icu_ids_path,
    )

    # Train/Test split by group (ICUSTAY_ID)
    tr_idx, te_idx = make_group_splits(icu_ids, test_size=0.2, seed=args.seed)

    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y_class[tr_idx], y_class[te_idx]
    sb_train, sb_test = static[tr_idx], static[te_idx]
    sl_train, sl_test = seq_lengths[tr_idx], seq_lengths[te_idx]
    icu_train, icu_test = icu_ids[tr_idx], icu_ids[te_idx]

    input_dim = X_train.shape[2]
    static_dim = sb_train.shape[1]
    print(
        f"Shapes | X_train: {tuple(X_train.shape)}  static: {tuple(sb_train.shape)}  "
        f"y_train: {tuple(y_train.shape)}  seq_len: {tuple(sl_train.shape)}"
    )

    # Build model factory for rnn_utils
    def model_builder():
        return BiLSTM_LOS_Classifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            static_dim=static_dim,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    # Training params for rnn_utils
    params: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    print("\n=== Cross-Validation Training ===")
    folds, fold_preds, best_artifacts, _ = train_eval_single_task(
        model_builder=model_builder,
        X_train=X_train,
        y_train=y_train,
        static_train=sb_train,
        seq_train=sl_train,
        icu_train=icu_train,
        num_classes=args.num_classes,
        params=params,
        device=device,
        results_dir=args.results_dir,
        n_splits=args.cv_splits,
        early_stopping_patience=args.patience,
        max_epochs=args.max_epochs,
    )
    print(f"Best CV linear κ: {best_artifacts['best_score']:.4f}")

    # Train final model on all training data
    print("\n=== Final Training on Full Train Set ===")
    final_model = train_final_single_task(
        model_builder=model_builder,
        X_train=X_train,
        y_train=y_train,
        static_train=sb_train,
        seq_train=sl_train,
        params=params,
        device=device,
    )

    # Save final model
    final_model_fp = os.path.join(args.results_dir, "best_final_model.pt")
    torch.save(final_model.state_dict(), final_model_fp)
    print(f"Saved final model -> {final_model_fp}")

    # Evaluate on held-out test set
    print("\n=== Test Evaluation ===")
    test_metrics, df_test = evaluate_single_task_test(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        static_test=sb_test,
        seq_test=sl_test,
        icu_test=icu_test,
        params=params,
        device=device,
        results_dir=args.results_dir,
    )
    print("Test metrics:", test_metrics)

    # Bootstrap CIs
    print("\n=== Bootstrap (n=1000) ===")
    boots = bootstrap_single_task(
        truths=list(df_test["true_class"].values),
        preds=list(df_test["predicted_class"].values),
        n_bootstrap=1000,
        seed=args.seed,
    )
    # Save bootstrap summary
    boot_path = os.path.join(args.results_dir, "bootstrap_summary.json")
    with open(boot_path, "w") as f:
        json.dump(boots, f, indent=2)
    print("Bootstrap summary saved ->", boot_path)

    print("\nDone. Artifacts in:", args.results_dir)


if __name__ == "__main__":
    main()
