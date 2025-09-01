#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import json
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, roc_auc_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

# ---- local helpers ----
from rnn_utils import (
    set_seed, get_device, load_multitask_tensors, MultitaskDataset,
    make_group_splits, save_metrics_table, save_predictions_table
)


# -------------------- Model --------------------

class BiLSTM_Multitask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 128,
        num_total_los_classes: int = 8,
        num_hourly_los_classes: int = 8,
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

        # Static encoder
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )

        # Shared head
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Task heads
        self.fc_total_los = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, num_total_los_classes),
        )
        self.fc_mortality = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, 1),
        )
        self.hourly_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, num_hourly_los_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x_padded, seq_lengths, static_features):
        packed = nn.utils.rnn.pack_padded_sequence(
            x_padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (h_n, _) = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=x_padded.shape[1]
        )

        # Hourly head on every timestep
        logits_hourly = self.hourly_fc(lstm_out)  # (B, T, C_hourly)

        # Global rep from last fwd+bwd
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_concat = torch.cat([h_fwd, h_bwd], dim=1)

        static_enc = self.static_mlp(static_features)
        shared = self.fc_shared(torch.cat([h_concat, static_enc], dim=1))

        logits_total = self.fc_total_los(shared)            # (B, 8)
        logits_mort = self.fc_mortality(shared).squeeze(1)  # (B,)

        return logits_total, logits_mort, logits_hourly


# -------------------- Loss --------------------

def multitask_loss_weighted(
    logits_total, logits_mort, logits_hourly,
    y_total_class, y_mortality, y_hourly, hour_mask,
    class_weights_total=None, class_weights_hourly=None,
    task_weights=(1.0, 1.0, 1.0), label_smoothing=0.1,
):
    loss_total_fn = nn.CrossEntropyLoss(
        weight=class_weights_total, label_smoothing=label_smoothing
    )
    loss_mort_fn = nn.BCEWithLogitsLoss()
    loss_hourly_fn = nn.CrossEntropyLoss(
        weight=class_weights_hourly, ignore_index=-1, label_smoothing=label_smoothing
    )

    # total los
    loss_total = loss_total_fn(logits_total, y_total_class)
    # mortality
    loss_mort = loss_mort_fn(logits_mort, y_mortality.float())

    # hourly (mask invalid steps)
    B, T, C = logits_hourly.shape
    logits_flat = logits_hourly.reshape(-1, C)
    y_hourly_flat = y_hourly.reshape(-1).clone()
    if hour_mask is not None:
        mask_flat = hour_mask.reshape(-1)
        y_hourly_flat[~mask_flat] = -1
    loss_hourly = loss_hourly_fn(logits_flat, y_hourly_flat)

    total_loss = (task_weights[0] * loss_total +
                  task_weights[1] * loss_mort +
                  task_weights[2] * loss_hourly)
    return total_loss, loss_total.item(), loss_mort.item(), loss_hourly.item()


# -------------------- Evaluation --------------------

@torch.no_grad()
def evaluate_multitask(model, loader, device, with_probs: bool = False):
    model.eval()
    total_true, total_pred, total_prob = [], [], []
    mort_true, mort_prob = [], []
    hourly_records = []  # list of dict rows

    for batch in loader:
        x, seq_l, static, y_total, y_mort, y_hourly, hour_mask, icu_ids = batch
        x, seq_l, static = x.to(device), seq_l.to(device), static.to(device)
        y_total = y_total.to(device)
        y_mort = y_mort.to(device)
        y_hourly = y_hourly.to(device)
        hour_mask = hour_mask.to(device)

        out_total, out_mort, out_hourly = model(x, seq_l, static)

        # total LOS
        tp = torch.argmax(out_total, dim=1)
        total_true.extend(y_total.cpu().numpy().tolist())
        total_pred.extend(tp.cpu().numpy().tolist())
        if with_probs:
            total_prob.extend(torch.softmax(out_total, dim=1).cpu().numpy().tolist())

        # mortality
        mp = torch.sigmoid(out_mort)
        mort_true.extend(y_mort.cpu().numpy().tolist())
        mort_prob.extend(mp.cpu().numpy().tolist())

        # hourly
        hourly_pred = torch.argmax(out_hourly, dim=2)  # (B, T)
        B, T = hourly_pred.shape
        for i in range(B):
            icu = int(icu_ids[i])
            for t in range(T):
                if hour_mask[i, t]:
                    row = {
                        "ICUSTAY_ID": icu,
                        "hour_bin": int(t),
                        "y_true_hourly": int(y_hourly[i, t].cpu()),
                        "y_pred_hourly": int(hourly_pred[i, t].cpu()),
                    }
                    if with_probs:
                        probs = torch.softmax(out_hourly[i, t], dim=0).cpu().numpy()
                        for c in range(probs.shape[0]):
                            row[f"prob_class_{c}"] = float(probs[c])
                    hourly_records.append(row)

    # metrics
    metrics = {
        "total_los_accuracy": accuracy_score(total_true, total_pred),
        "total_los_kappa_linear": cohen_kappa_score(total_true, total_pred, weights="linear"),
        "total_los_kappa_qwk": cohen_kappa_score(total_true, total_pred, weights="quadratic"),
    }
    # mortality (probabilities in mort_prob)
    try:
        metrics["mortality_auc_roc"] = roc_auc_score(mort_true, mort_prob)
        metrics["mortality_auc_pr"] = average_precision_score(mort_true, mort_prob)
        metrics["mortality_accuracy"] = accuracy_score(
            mort_true, (np.array(mort_prob) >= 0.5).astype(int)
        )
    except Exception:
        metrics["mortality_auc_roc"] = np.nan
        metrics["mortality_auc_pr"] = np.nan
        metrics["mortality_accuracy"] = np.nan

    # hourly
    if len(hourly_records) > 0:
        h_true = [r["y_true_hourly"] for r in hourly_records]
        h_pred = [r["y_pred_hourly"] for r in hourly_records]
        metrics["hourly_accuracy"] = accuracy_score(h_true, h_pred)
        metrics["hourly_kappa_linear"] = cohen_kappa_score(h_true, h_pred, weights="linear")
        metrics["hourly_kappa_qwk"] = cohen_kappa_score(h_true, h_pred, weights="quadratic")

    outputs = {
        "total": (total_true, total_pred, np.array(total_prob) if with_probs else None),
        "mortality": (mort_true, mort_prob),
        "hourly": hourly_records,
    }
    return metrics, outputs


# -------------------- Grid search (quick) --------------------

def grid_search(
    X_tr, static_tr, seq_tr, y_total_tr, y_mort_tr, y_hourly_tr, hour_mask_tr, icu_tr,
    param_grid: Dict, device: torch.device, cv_folds: int = 3, quick_epochs: int = 20
) -> Dict:
    print("Starting Grid Search...")
    best_score = -1.0
    best_params = None

    # hourly class weights from valid positions
    hourly_flat = y_hourly_tr[hour_mask_tr].cpu().numpy()
    hourly_classes = np.unique(hourly_flat)
    hourly_w = compute_class_weight("balanced", classes=hourly_classes, y=hourly_flat)
    # total los weights
    total_classes = np.unique(y_total_tr.cpu().numpy())
    total_w = compute_class_weight("balanced", classes=total_classes, y=y_total_tr.cpu().numpy())

    for params in ParameterGrid(param_grid):
        print(f"\nParams: {params}")
        gkf = GroupKFold(n_splits=cv_folds)
        fold_scores = []

        for k, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_total_tr, icu_tr), start=1):
            tr_ds = MultitaskDataset(
                X_tr[tr_idx], seq_tr[tr_idx], static_tr[tr_idx],
                y_total_tr[tr_idx], y_mort_tr[tr_idx],
                y_hourly_tr[tr_idx], hour_mask_tr[tr_idx], icu_tr[tr_idx]
            )
            va_ds = MultitaskDataset(
                X_tr[va_idx], seq_tr[va_idx], static_tr[va_idx],
                y_total_tr[va_idx], y_mort_tr[va_idx],
                y_hourly_tr[va_idx], hour_mask_tr[va_idx], icu_tr[va_idx]
            )
            tr_loader = DataLoader(tr_ds, batch_size=params["batch_size"], shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=params["batch_size"], shuffle=False)

            model = BiLSTM_Multitask(
                input_dim=X_tr.shape[2],
                static_dim=static_tr.shape[1],
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
            ).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

            total_w_t = torch.tensor(total_w, dtype=torch.float32, device=device)
            hourly_w_t = torch.tensor(hourly_w, dtype=torch.float32, device=device)

            best_lin_k = -1.0
            patience = 3
            no_improve = 0

            for ep in range(quick_epochs):
                model.train()
                for batch in tr_loader:
                    x, seq_l, static, y_tot, y_mo, y_hr, h_mask, _ = batch
                    x, seq_l, static = x.to(device), seq_l.to(device), static.to(device)
                    y_tot = y_tot.to(device)
                    y_mo = y_mo.to(device)
                    y_hr = y_hr.to(device)
                    h_mask = h_mask.to(device)

                    opt.zero_grad()
                    lo_tot, lo_mo, lo_hr = model(x, seq_l, static)
                    loss, _, _, _ = multitask_loss_weighted(
                        lo_tot, lo_mo, lo_hr, y_tot, y_mo, y_hr, h_mask,
                        class_weights_total=total_w_t, class_weights_hourly=hourly_w_t
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                # validate
                metrics, _ = evaluate_multitask(model, va_loader, device)
                lin_k = metrics["total_los_kappa_linear"]
                sch.step(lin_k)
                if lin_k > best_lin_k:
                    best_lin_k = lin_k
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            fold_scores.append(best_lin_k)

        mean_k = float(np.mean(fold_scores))
        std_k = float(np.std(fold_scores))
        print(f"  CV linear kappa: {mean_k:.4f} ± {std_k:.4f}")

        if mean_k > best_score:
            best_score = mean_k
            best_params = dict(params)

    print(f"\nBest params: {best_params}")
    print(f"Best CV (linear kappa): {best_score:.4f}")
    return best_params


# -------------------- CV train (full) --------------------

def cv_train_full(
    X_tr, static_tr, seq_tr, y_total_tr, y_mort_tr, y_hourly_tr, hour_mask_tr, icu_tr,
    best_params: Dict, device: torch.device, results_dir: str, n_splits: int = 5,
) -> Tuple[Dict, Dict]:
    os.makedirs(results_dir, exist_ok=True)

    # compute class weights once (on all train)
    hourly_flat = y_hourly_tr[hour_mask_tr].cpu().numpy()
    hourly_w = compute_class_weight("balanced", classes=np.unique(hourly_flat), y=hourly_flat)
    total_w = compute_class_weight("balanced", classes=np.unique(y_total_tr.cpu().numpy()), y=y_total_tr.cpu().numpy())
    hourly_w_t = torch.tensor(hourly_w, dtype=torch.float32, device=device)
    total_w_t = torch.tensor(total_w, dtype=torch.float32, device=device)

    gkf = GroupKFold(n_splits=n_splits)
    cv_metrics = []
    best_overall = -1.0
    best_state = None

    all_total_rows = []
    all_mort_rows = []
    all_hourly_rows = []

    fold = 1
    for tr_idx, va_idx in gkf.split(X_tr, y_total_tr, icu_tr):
        print(f"\n===== Fold {fold} / {n_splits} =====")
        tr_ds = MultitaskDataset(
            X_tr[tr_idx], seq_tr[tr_idx], static_tr[tr_idx],
            y_total_tr[tr_idx], y_mort_tr[tr_idx],
            y_hourly_tr[tr_idx], hour_mask_tr[tr_idx], icu_tr[tr_idx]
        )
        va_ds = MultitaskDataset(
            X_tr[va_idx], seq_tr[va_idx], static_tr[va_idx],
            y_total_tr[va_idx], y_mort_tr[va_idx],
            y_hourly_tr[va_idx], hour_mask_tr[va_idx], icu_tr[va_idx]
        )

        tr_loader = DataLoader(tr_ds, batch_size=best_params["batch_size"], shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=best_params["batch_size"], shuffle=False)

        model = BiLSTM_Multitask(
            input_dim=X_tr.shape[2],
            static_dim=static_tr.shape[1],
            hidden_dim=best_params["hidden_dim"],
            dropout=best_params["dropout"],
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

        best_fold_lin_k = -1.0
        patience = 15
        no_improve = 0

        for ep in range(100):
            model.train()
            for batch in tr_loader:
                x, seq_l, s, y_tot, y_mo, y_hr, h_mask, _ = batch
                x, seq_l, s = x.to(device), seq_l.to(device), s.to(device)
                y_tot = y_tot.to(device)
                y_mo = y_mo.to(device)
                y_hr = y_hr.to(device)
                h_mask = h_mask.to(device)

                opt.zero_grad()
                lo_tot, lo_mo, lo_hr = model(x, seq_l, s)
                loss, _, _, _ = multitask_loss_weighted(
                    lo_tot, lo_mo, lo_hr, y_tot, y_mo, y_hr, h_mask,
                    class_weights_total=total_w_t, class_weights_hourly=hourly_w_t
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            metrics, outputs = evaluate_multitask(model, va_loader, device)
            lin_k = metrics["total_los_kappa_linear"]
            sch.step(lin_k)

            if lin_k > best_fold_lin_k:
                best_fold_lin_k = lin_k
                no_improve = 0
                best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                # cache outputs for saving after fold completes
                fold_total_true, fold_total_pred, _ = outputs["total"]
                fold_mort_true, fold_mort_prob = outputs["mortality"]
                fold_hourly = outputs["hourly"]
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break

        # save fold preds
        total_df = pd.DataFrame({
            "fold": fold,
            "ICUSTAY_ID": icu_tr[va_idx],
            "y_true": fold_total_true,
            "y_pred": fold_total_pred,
        })
        all_total_rows.append(total_df)

        mort_df = pd.DataFrame({
            "fold": fold,
            "ICUSTAY_ID": icu_tr[va_idx],
            "y_true": fold_mort_true,
            "y_pred_prob": fold_mort_prob,
            "y_pred_binary": (np.array(fold_mort_prob) >= 0.5).astype(int),
        })
        all_mort_rows.append(mort_df)

        if len(fold_hourly) > 0:
            hf = pd.DataFrame(fold_hourly)
            hf["fold"] = fold
            all_hourly_rows.append(hf)

        # record metrics
        metrics["fold"] = fold
        cv_metrics.append(metrics)

        # best overall
        if best_fold_lin_k > best_overall:
            best_overall = best_fold_lin_k
            best_state = best_fold_state

        fold += 1

    # write CV artifacts
    cv_df = pd.DataFrame(cv_metrics)
    cv_df.to_csv(os.path.join(results_dir, "cv_metrics_summary.csv"), index=False)
    pd.concat(all_total_rows, ignore_index=True).to_csv(
        os.path.join(results_dir, "cv_total_los_predictions.csv"), index=False
    )
    pd.concat(all_mort_rows, ignore_index=True).to_csv(
        os.path.join(results_dir, "cv_mortality_predictions.csv"), index=False
    )
    if len(all_hourly_rows) > 0:
        pd.concat(all_hourly_rows, ignore_index=True).to_csv(
            os.path.join(results_dir, "cv_hourly_los_predictions.csv"), index=False
        )

    return {"best_state": best_state, "best_cv_linear_kappa": best_overall}, {"cv_metrics": cv_metrics}


# -------------------- Bootstrap (test) --------------------

def bootstrap_metrics(truths, preds, n=1000, seed=42, weights=None):
    rng = np.random.default_rng(seed)
    truths = np.asarray(truths)
    preds = np.asarray(preds)
    N = len(truths)

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    ks, ls, qs, accs = [], [], [], []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        t = truths[idx]
        p = preds[idx]
        ks.append(cohen_kappa_score(t, p))
        ls.append(cohen_kappa_score(t, p, weights="linear"))
        qs.append(cohen_kappa_score(t, p, weights="quadratic"))
        accs.append(accuracy_score(t, p))

    out = {
        "kappa_mean": float(np.mean(ks)), "kappa_ci": ci(ks),
        "linear_kappa_mean": float(np.mean(ls)), "linear_kappa_ci": ci(ls),
        "quadratic_kappa_mean": float(np.mean(qs)), "quadratic_kappa_ci": ci(qs),
        "accuracy_mean": float(np.mean(accs)), "accuracy_ci": ci(accs),
    }
    return out


# -------------------- Main --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    # data paths
    ap.add_argument("--X", required=True, help="Path to X tensor (.pt)")
    ap.add_argument("--y_total", required=True, help="Path to y_total_class tensor (.pt)")
    ap.add_argument("--static", required=True, help="Path to static tensor (.pt)")
    ap.add_argument("--seq_lengths", required=True, help="Path to seq_lengths tensor (.pt)")
    ap.add_argument("--icu_ids", required=True, help="Path to icu_id_list tensor (.pt)")
    ap.add_argument("--y_mort", required=True, help="Path to mortality labels (.pt)")
    ap.add_argument("--y_hourly", required=True, help="Path to hourly remaining LOS labels (.pt)")
    ap.add_argument("--hour_mask", required=True, help="Path to boolean hour mask (.pt)")

    # output
    ap.add_argument("--results_dir", default="results/multitask", help="Where to save outputs")

    # config / grid
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--gs_folds", type=int, default=3)
    ap.add_argument("--gs_epochs", type=int, default=20)

    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--dropout", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3])
    ap.add_argument("--bs", type=int, nargs="+", default=[32, 64])

    ap.add_argument("--no_grid", action="store_true", help="Skip grid search; use --use_params")
    ap.add_argument("--use_params", type=str, default="", help='JSON: {"hidden_dim":128,"dropout":0.3,"learning_rate":0.001,"batch_size":64}')

    ap.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for test")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # load data
    (X, y_total, static, seq_lengths, icu_ids,
     y_mort, y_hourly, hour_mask) = load_multitask_tensors(
        args.X, args.y_total, args.static, args.seq_lengths, args.icu_ids,
        args.y_mort, args.y_hourly, args.hour_mask
    )

    # split train/test by groups
    tr_idx, te_idx = make_group_splits(icu_ids, test_size=0.2, seed=args.seed)

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_total_tr, y_total_te = y_total[tr_idx], y_total[te_idx]
    static_tr, static_te = static[tr_idx], static[te_idx]
    seq_tr, seq_te = seq_lengths[tr_idx], seq_lengths[te_idx]
    y_mort_tr, y_mort_te = y_mort[tr_idx], y_mort[te_idx]
    y_hour_tr, y_hour_te = y_hourly[tr_idx], y_hourly[te_idx]
    mask_tr, mask_te = hour_mask[tr_idx], hour_mask[te_idx]
    icu_tr, icu_te = icu_ids[tr_idx], icu_ids[te_idx]

    print(f"Train size: {len(X_tr)} | Test size: {len(X_te)}")

    # grid params
    if args.no_grid and args.use_params:
        best_params = json.loads(args.use_params)
        print(f"Using provided params: {best_params}")
    else:
        param_grid = {
            "hidden_dim": args.hidden,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "batch_size": args.bs,
        }
        best_params = grid_search(
            X_tr, static_tr, seq_tr, y_total_tr, y_mort_tr, y_hour_tr, mask_tr, icu_tr,
            param_grid, device=device, cv_folds=args.gs_folds, quick_epochs=args.gs_epochs
        )

    # full CV with best params
    best_art, cv_art = cv_train_full(
        X_tr, static_tr, seq_tr, y_total_tr, y_mort_tr, y_hour_tr, mask_tr, icu_tr,
        best_params, device=device, results_dir=args.results_dir, n_splits=args.cv_folds
    )
    torch.save(best_art, os.path.join(args.results_dir, "best_cv_artifacts.pt"))

    # train final on all training data with best params
    final_model = BiLSTM_Multitask(
        input_dim=X_tr.shape[2],
        static_dim=static_tr.shape[1],
        hidden_dim=best_params["hidden_dim"],
        dropout=best_params["dropout"],
    ).to(device)
    final_model.load_state_dict({k: v.to(device) for k, v in best_art["best_state"].items()})

    # test loader
    te_ds = MultitaskDataset(
        X_te, seq_te, static_te, y_total_te, y_mort_te, y_hour_te, mask_te, icu_te
    )
    te_loader = DataLoader(te_ds, batch_size=best_params["batch_size"], shuffle=False)

    # test eval (with probs)
    print("\nEvaluating on test set...")
    test_metrics, test_outputs = evaluate_multitask(final_model, te_loader, device, with_probs=True)
    print(json.dumps(test_metrics, indent=2))
    save_metrics_table(test_metrics, os.path.join(args.results_dir, "test_metrics_summary.csv"))

    # save test predictions
    total_true, total_pred, total_prob = test_outputs["total"]
    mort_true, mort_prob = test_outputs["mortality"]
    hourly_rows = test_outputs["hourly"]

    pd.DataFrame({
        "ICUSTAY_ID": icu_te,
        "y_true": total_true,
        "y_pred": total_pred,
        **{f"prob_class_{c}": total_prob[:, c] for c in range(total_prob.shape[1])}
    }).to_csv(os.path.join(args.results_dir, "test_total_los_predictions_with_probs.csv"), index=False)

    pd.DataFrame({
        "ICUSTAY_ID": icu_te,
        "y_true": mort_true,
        "y_pred_prob": mort_prob,
        "y_pred_binary": (np.array(mort_prob) >= 0.5).astype(int)
    }).to_csv(os.path.join(args.results_dir, "test_mortality_predictions_with_probs.csv"), index=False)

    if len(hourly_rows) > 0:
        pd.DataFrame(hourly_rows).to_csv(
            os.path.join(args.results_dir, "test_hourly_los_predictions_with_probs.csv"), index=False
        )

    # bootstrap on total LOS
    print("\nBootstrap on total LOS (n={}):".format(args.bootstrap))
    boot = bootstrap_metrics(total_true, total_pred, n=args.bootstrap, seed=args.seed)
    with open(os.path.join(args.results_dir, "test_bootstrap_summary.json"), "w") as f:
        json.dump(boot, f, indent=2)

    # save model
    torch.save(
        {
            "state_dict": best_art["best_state"],
            "best_params": best_params,
            "input_dim": X_tr.shape[2],
            "static_dim": static_tr.shape[1],
        },
        os.path.join(args.results_dir, "best_model_multitask.pth"),
    )

    print("\nAll done. Artifacts saved to:", args.results_dir)


if __name__ == "__main__":
    main()

