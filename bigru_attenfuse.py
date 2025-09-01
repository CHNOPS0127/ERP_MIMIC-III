#!/usr/bin/env python3
"""
BiGRU + Attention Fusion (single-task LOS classification)

- Uses rnn_utils.py for:
  * set_seed, get_device
  * load_single_task_tensors
  * make_group_splits (grouped train/test by ICUSTAY_ID)
  * train_eval_single_task (CV + early stopping)
  * train_final_single_task (final fit on all training data)
  * evaluate_single_task_test (point metrics + test predictions file)
  * bootstrap_single_task (95% CIs via bootstrap)
  * save_metrics_table, save_predictions_table

Run example (one config):
  python bigru_attenfuse.py ^
    --X_path D:\...\tensor\X_padded_tensor.pt ^
    --y_total_class_path D:\...\tensor\y_total_class_tensor.pt ^
    --static_path D:\...\tensor\static_tensor.pt ^
    --seq_lengths_path D:\...\tensor\seq_lengths.pt ^
    --icu_ids_path D:\...\tensor\icu_id_list.pt ^
    --results_dir D:\...\results\BiGRU-AttenFuse ^
    --hidden_dim 128 --num_layers 2 --dropout 0.3 --batch_size 64 --learning_rate 1e-3 --seed 42

Grid search style (comma-separated lists are allowed):
  --hidden_dim 128,256 --dropout 0.2,0.3 --batch_size 32,64 --learning_rate 0.001,0.0005 --num_layers 1,2 --seed 42
"""

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rnn_utils import (
    set_seed, get_device, load_single_task_tensors, make_group_splits,
    train_eval_single_task, train_final_single_task, evaluate_single_task_test,
    bootstrap_single_task, save_metrics_table, save_predictions_table
)

# -------------------- Model blocks --------------------

class StaticGatedBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        feat = F.relu(self.fc(x))
        gate = torch.sigmoid(self.gate(x))
        return self.dropout(feat * gate)


class TemporalPatternAttention(nn.Module):
    """Multi-kernel temporal conv + learned attention pooling with positional encoding."""
    def __init__(self, input_dim: int, num_filters: int = 64, filter_sizes: List[int] = [3, 5, 7, 9], max_len: int = 256):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, num_filters, k, padding=k // 2) for k in filter_sizes])
        self.total_filters = num_filters * len(filter_sizes)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, self.total_filters))
        self.norm = nn.LayerNorm(self.total_filters)
        self.attn = nn.Linear(self.total_filters, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask: torch.Tensor, return_weights: bool = False):
        # x: (B, T, C) -> conv wants (B, C, T)
        x = x.transpose(1, 2)
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]        # list of (B, F, T)
        conv_out = torch.cat(conv_outputs, dim=1).transpose(1, 2)      # (B, T, F_total)
        T = conv_out.size(1)
        conv_out = self.norm(conv_out + self.pos_encoding[:, :T, :])

        attn_scores = self.attn(conv_out).squeeze(-1)                  # (B, T)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)              # (B, T)

        pooled = torch.sum(conv_out * attn_weights.unsqueeze(-1), dim=1)  # (B, F_total)
        pooled = self.dropout(pooled)
        return (pooled, attn_weights) if return_weights else pooled


class BiGRU_TPA_LOS_Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, static_dim: int,
                 num_classes: int = 8, dropout: float = 0.3, num_layers: int = 2,
                 static_hidden_dim: int = 64):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.attn_pool = TemporalPatternAttention(input_dim=hidden_dim * 2, num_filters=64)
        self.mean_proj = nn.Linear(hidden_dim * 2, 256)
        self.max_proj  = nn.Linear(hidden_dim * 2, 256)
        self.fusion_dropout = nn.Dropout(0.1)
        self.pool_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        self.static_encoder = StaticGatedBlock(static_dim, hidden_dim=static_hidden_dim, dropout=dropout)

        self.fc1 = nn.Linear(256 + static_hidden_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.res_proj = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_padded: torch.Tensor, seq_lengths: torch.Tensor, static_features: torch.Tensor,
                return_attn: bool = False):
        # Pack variable length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x_padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.bigru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)  # (B, T, 2H)

        B, T, _ = unpacked.size()
        device = unpacked.device
        time_idx = torch.arange(T, device=device).unsqueeze(0)          # (1, T)
        mask = time_idx < seq_lengths.unsqueeze(1)                       # (B, T), True where valid

        # Attention pool (masked)
        attn_pool = self.attn_pool(unpacked, mask)                       # (B, F*)

        # Masked mean
        mask_f = mask.unsqueeze(-1)                                      # (B, T, 1)
        lengths = mask_f.sum(dim=1).clamp_min(1)                         # (B, 1)
        mean_pool = (unpacked * mask_f).sum(dim=1) / lengths             # (B, 2H)

        # Masked max
        neg_inf = torch.finfo(unpacked.dtype).min
        masked = unpacked.masked_fill(~mask_f, neg_inf)
        max_pool = masked.amax(dim=1)                                    # (B, 2H)

        # Projections + learned fusion
        mean_proj = self.mean_proj(mean_pool)
        max_proj  = self.max_proj(max_pool)
        weights = torch.softmax(self.pool_weights, dim=0)
        fused_pool = weights[0] * attn_pool + weights[1] * mean_proj + weights[2] * max_proj
        fused_pool = self.fusion_dropout(fused_pool)

        # Static fusion
        static_out = self.static_encoder(static_features)
        combined = torch.cat([fused_pool, static_out], dim=1)

        # Head with residual
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x_res = self.res_proj(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x + x_res
        logits = self.out(x)
        return logits

# -------------------- Runner helpers --------------------

def parse_list_int(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]

def parse_list_float(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]

def run_single_config(
    cfg: Dict[str, Any],
    X, y, static, seq_lengths, icu_ids,
    num_classes: int,
    input_dim: int,
    static_dim: int,
    device: torch.device,
    base_results_dir: str,
    cv_splits: int,
    max_epochs: int,
    early_stop: int,
    test_size: float,
    bootstrap_iters: int,
):
    set_seed(cfg["seed"])
    # group split for held-out test
    tr_idx, te_idx = make_group_splits(icu_ids, test_size=test_size, seed=cfg["seed"])
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    S_tr, S_te = static[tr_idx], static[te_idx]
    L_tr, L_te = seq_lengths[tr_idx], seq_lengths[te_idx]
    icu_tr, icu_te = icu_ids[tr_idx], icu_ids[te_idx]

    # results subdir per config
    tag = f"hd{cfg['hidden_dim']}_nl{cfg['num_layers']}_do{cfg['dropout']}_bs{cfg['batch_size']}_lr{cfg['learning_rate']}_seed{cfg['seed']}"
    out_dir = os.path.join(base_results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # model builder closure for rnn_utils
    def model_builder():
        return BiGRU_TPA_LOS_Classifier(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            static_dim=static_dim,
            num_classes=num_classes,
            dropout=cfg["dropout"],
            num_layers=cfg["num_layers"],
        )

    # CV train/eval
    folds, fold_preds, best_artifacts, _ = train_eval_single_task(
        model_builder=model_builder,
        X_train=X_tr, y_train=y_tr, static_train=S_tr, seq_train=L_tr, icu_train=icu_tr,
        num_classes=num_classes,
        params={"batch_size": cfg["batch_size"], "learning_rate": cfg["learning_rate"]},
        device=device,
        results_dir=out_dir,
        n_splits=cv_splits,
        early_stopping_patience=early_stop,
        max_epochs=max_epochs,
    )

    # Final train on all training data
    final_model = train_final_single_task(
        model_builder, X_tr, y_tr, S_tr, L_tr,
        params={"batch_size": cfg["batch_size"], "learning_rate": cfg["learning_rate"]},
        device=device
    )
    torch.save(final_model.state_dict(), os.path.join(out_dir, "best_final_model.pt"))

    # Test evaluation (point estimates + predictions CSV)
    test_metrics, df_test = evaluate_single_task_test(
        final_model, X_te, y_te, S_te, L_te, icu_te,
        params={"batch_size": cfg["batch_size"]}, device=device, results_dir=out_dir
    )
    save_metrics_table(test_metrics, os.path.join(out_dir, "test_metrics_summary.csv"))

    # Bootstrap (95% CI)
    boots = bootstrap_single_task(
        truths=df_test["true_class"].tolist(),
        preds=df_test["predicted_class"].tolist(),
        n_bootstrap=bootstrap_iters,
        seed=cfg["seed"]
    )
    with open(os.path.join(out_dir, "test_bootstrap_summary.json"), "w") as f:
        json.dump(boots, f, indent=2)

    # Save config & a small readme
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write("BiGRU-AttenFuse single-task LOS classification\n")
        f.write(f"Best CV linear kappa: {best_artifacts.get('best_score', 'N/A')}\n")

    return {
        "out_dir": out_dir,
        "cv_best_linear_kappa": best_artifacts.get("best_score", -1.0),
        "test_metrics": test_metrics,
        "bootstrap": boots
    }

# -------------------- CLI --------------------

def build_argparser():
    p = argparse.ArgumentParser(description="BiGRU-AttenFuse (single-task LOS classification) using rnn_utils.py")
    # tensor paths
    p.add_argument("--X_path", required=True, type=str)
    p.add_argument("--y_total_class_path", required=True, type=str)
    p.add_argument("--static_path", required=True, type=str)
    p.add_argument("--seq_lengths_path", required=True, type=str)
    p.add_argument("--icu_ids_path", required=True, type=str)

    p.add_argument("--results_dir", required=True, type=str)

    # grid-able hyperparams (comma-separated lists allowed)
    p.add_argument("--hidden_dim", default="128", type=str, help="e.g. '128' or '128,256'")
    p.add_argument("--num_layers", default="2", type=str, help="e.g. '2' or '1,2'")
    p.add_argument("--dropout", default="0.3", type=str, help="e.g. '0.3' or '0.2,0.3'")
    p.add_argument("--batch_size", default="64", type=str, help="e.g. '64' or '32,64'")
    p.add_argument("--learning_rate", default="0.001", type=str, help="e.g. '0.001' or '0.001,0.0005'")
    p.add_argument("--seed", default="42", type=str, help="e.g. '42' or '1,2,3'")

    # training controls
    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--early_stopping", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--bootstrap", type=int, default=1000)

    return p

def main():
    args = build_argparser().parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = get_device()
    print("Device:", device.type)

    # Load tensors
    X, y, static, seq_lengths, icu_ids = load_single_task_tensors(
        args.X_path, args.y_total_class_path, args.static_path, args.seq_lengths_path, args.icu_ids_path
    )
    input_dim  = X.shape[-1]
    static_dim = static.shape[-1]
    num_classes = int(torch.max(y).item() + 1)  # assumes classes are 0..K-1

    # Build parameter grid (comma-separated lists allowed)
    hds  = parse_list_int(args.hidden_dim)
    nls  = parse_list_int(args.num_layers)
    dos  = parse_list_float(args.dropout)
    bss  = parse_list_int(args.batch_size)
    lrs  = parse_list_float(args.learning_rate)
    sds  = parse_list_int(args.seed)

    grid: List[Dict[str, Any]] = []
    for hd in hds:
        for nl in nls:
            for do in dos:
                for bs in bss:
                    for lr in lrs:
                        for sd in sds:
                            grid.append({
                                "hidden_dim": hd, "num_layers": nl, "dropout": do,
                                "batch_size": bs, "learning_rate": lr, "seed": sd
                            })

    print(f"\nTotal configs: {len(grid)}")
    best_cv = -1.0
    best_cfg = None
    best_result_dir = None

    for i, cfg in enumerate(grid, start=1):
        print(f"\n=== Config {i}/{len(grid)}: {cfg} ===")
        out = run_single_config(
            cfg, X, y, static, seq_lengths, icu_ids,
            num_classes=num_classes, input_dim=input_dim, static_dim=static_dim,
            device=device, base_results_dir=args.results_dir,
            cv_splits=args.cv_splits, max_epochs=args.max_epochs,
            early_stop=args.early_stopping, test_size=args.test_size,
            bootstrap_iters=args.bootstrap,
        )
        if out["cv_best_linear_kappa"] > best_cv:
            best_cv = out["cv_best_linear_kappa"]
            best_cfg = cfg
            best_result_dir = out["out_dir"]

    # Write top-level best summary
    summary = {
        "best_cv_linear_kappa": best_cv,
        "best_config": best_cfg,
        "best_results_dir": best_result_dir
    }
    with open(os.path.join(args.results_dir, "best_overall_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Finished ===")
    print("Best CV linear kappa:", best_cv)
    print("Best config:", best_cfg)
    print("Best results dir:", best_result_dir)

if __name__ == "__main__":
    main()
