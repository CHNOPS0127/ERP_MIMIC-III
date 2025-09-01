


from __future__ import annotations

import os
import json
import copy
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight


# -------------------- Repro --------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Data loading --------------------

def _load_tensor_maybe(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    return torch.load(path)


def load_single_task_tensors(
    X_path: str,
    y_total_class_path: str,
    static_path: str,
    seq_lengths_path: str,
    icu_ids_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    X = torch.load(X_path)
    y = torch.load(y_total_class_path)
    static = torch.load(static_path)
    seq_lengths = torch.load(seq_lengths_path)
    icu_ids = np.array(torch.load(icu_ids_path))
    return X, y, static, seq_lengths, icu_ids


def load_multitask_tensors(
    X_path: str,
    y_total_class_path: str,
    static_path: str,
    seq_lengths_path: str,
    icu_ids_path: str,
    y_mortality_path: str,
    y_hourly_path: str,
    hour_mask_path: str,
) -> Tuple[torch.Tensor, ...]:
    X = torch.load(X_path)
    y_total = torch.load(y_total_class_path)
    static = torch.load(static_path)
    seq_lengths = torch.load(seq_lengths_path)
    icu_ids = np.array(torch.load(icu_ids_path))
    y_mortality = torch.load(y_mortality_path)
    y_hourly = torch.load(y_hourly_path)
    hour_mask = torch.load(hour_mask_path)
    return X, y_total, static, seq_lengths, icu_ids, y_mortality, y_hourly, hour_mask


# -------------------- Datasets --------------------

class SingleTaskDataset(Dataset):
    def __init__(self, X, seq_lengths, static, y_class):
        self.X = X
        self.seq_lengths = seq_lengths
        self.static = static
        self.y = y_class

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.seq_lengths[idx], self.static[idx], self.y[idx]


class MultitaskDataset(Dataset):
    def __init__(self, X, seq_lengths, static, y_total_class, y_mortality, y_hourly, hour_mask, icu_ids):
        self.X = X
        self.seq_lengths = seq_lengths
        self.static = static
        self.y_total_class = y_total_class
        self.y_mortality = y_mortality
        self.y_hourly = y_hourly
        self.hour_mask = hour_mask
        self.icu_ids = icu_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.seq_lengths[idx],
            self.static[idx],
            self.y_total_class[idx],
            self.y_mortality[idx],
            self.y_hourly[idx],
            self.hour_mask[idx],
            self.icu_ids[idx],
        )


# -------------------- Splitting --------------------

def make_group_splits(icu_ids: np.ndarray, test_size: float = 0.2, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dummy_X = np.zeros_like(icu_ids)
    dummy_y = np.zeros_like(icu_ids)
    train_idx, test_idx = next(splitter.split(dummy_X, dummy_y, groups=icu_ids))
    return train_idx, test_idx


# -------------------- Losses / class weights --------------------

def build_class_weights(y_train: torch.Tensor, device: torch.device) -> torch.Tensor:
    classes = np.unique(y_train.cpu().numpy())
    weights = compute_class_weight("balanced", classes=classes, y=y_train.cpu().numpy())
    return torch.tensor(weights, dtype=torch.float32, device=device)


# -------------------- Metrics helpers --------------------

@dataclass
class FoldMetrics:
    fold: int
    kappa: float
    linear_kappa: float
    quadratic_kappa: float
    accuracy: float
    confusion_matrix: List[List[int]]


def summarize_fold_metrics(folds: List[FoldMetrics]) -> pd.DataFrame:
    df = pd.DataFrame([{
        "Fold": f.fold,
        "Kappa": f.kappa,
        "Linear_Kappa": f.linear_kappa,
        "Quadratic_Kappa": f.quadratic_kappa,
        "Accuracy": f.accuracy
    } for f in folds])
    return df


def mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # returns boolean mask: True where valid
    device = lengths.device
    rng = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
    return rng < lengths.unsqueeze(1)  # (B, T)


# -------------------- Training (single-task) --------------------

def train_eval_single_task(
    model_builder,  # callable -> nn.Module
    X_train, y_train, static_train, seq_train, icu_train,
    num_classes: int,
    params: Dict[str, Any],
    device: torch.device,
    results_dir: str,
    n_splits: int = 5,
    early_stopping_patience: int = 10,
    max_epochs: int = 100,
) -> Tuple[List[FoldMetrics], List[pd.DataFrame], Dict[str, Any], Dict[str, Any]]:

    os.makedirs(results_dir, exist_ok=True)

    # Prepare dataset for CV
    train_dataset = SingleTaskDataset(X_train, seq_train, static_train, y_train)
    kf = GroupKFold(n_splits=n_splits)
    folds: List[FoldMetrics] = []
    fold_predictions: List[pd.DataFrame] = []

    best_overall_linear_kappa = -1.0
    best_state_dict: Optional[Dict[str, Any]] = None

    for fold, (tr_idx, val_idx) in enumerate(
        kf.split(np.arange(len(train_dataset)), groups=icu_train), start=1
    ):
        tr_set, val_set = Subset(train_dataset, tr_idx), Subset(train_dataset, val_idx)
        tr_loader = DataLoader(tr_set, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

        model = model_builder().to(device)
        criterion = nn.CrossEntropyLoss(weight=build_class_weights(y_train[tr_idx], device))
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
        )


        best_linear_kappa = -1.0
        patience_counter = 0
        best_fold_state = None
        best_preds, best_true, best_ids = [], [], []

        for epoch in range(max_epochs):
            # ---- train ----
            model.train()
            for xb, sl, sb, yb in tr_loader:
                xb, sl, sb, yb = xb.to(device), sl.to(device), sb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb, sl, sb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # ---- valid ----
            model.eval()
            preds, truths, ids = [], [], []
            with torch.no_grad():
                for i, (xb, sl, sb, yb) in enumerate(val_loader):
                    xb, sl, sb, yb = xb.to(device), sl.to(device), sb.to(device), yb.to(device)
                    logits = model(xb, sl, sb)
                    pred = torch.argmax(logits, dim=1)
                    preds.extend(pred.cpu().numpy())
                    truths.extend(yb.cpu().numpy())
                    # reconstruct ICUSTAY_IDs for this mini-batch
                    start = i * params["batch_size"]
                    ids.extend([icu_train[val_idx[j]] for j in range(start, start + len(yb))])

            kappa = cohen_kappa_score(truths, preds)
            linear_kappa = cohen_kappa_score(truths, preds, weights="linear")
            quadratic_kappa = cohen_kappa_score(truths, preds, weights="quadratic")
            acc = accuracy_score(truths, preds)
            scheduler.step(linear_kappa)

            if linear_kappa > best_linear_kappa:
                best_linear_kappa = linear_kappa
                patience_counter = 0
                best_fold_state = copy.deepcopy(model.state_dict())
                best_preds, best_true, best_ids = preds, truths, ids
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        # Save fold model
        torch.save(best_fold_state, os.path.join(results_dir, f"best_model_fold_{fold}.pt"))

        fm = FoldMetrics(
            fold=fold,
            kappa=cohen_kappa_score(best_true, best_preds),
            linear_kappa=best_linear_kappa,
            quadratic_kappa=cohen_kappa_score(best_true, best_preds, weights="quadratic"),
            accuracy=accuracy_score(best_true, best_preds),
            confusion_matrix=confusion_matrix(best_true, best_preds).tolist(),
        )
        folds.append(fm)

        # Save fold preds
        df_fold = pd.DataFrame({"ICUSTAY_ID": best_ids, "true_class": best_true, "predicted_class": best_preds, "fold": fold})
        fold_predictions.append(df_fold)

        # Track overall best model across folds
        if best_linear_kappa > best_overall_linear_kappa and best_fold_state is not None:
            best_overall_linear_kappa = best_linear_kappa
            best_state_dict = best_fold_state

    # Final aggregated artifacts
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())  # type: ignore

    # Save CV summaries
    cv_df = summarize_fold_metrics(folds)
    cv_df.to_csv(os.path.join(results_dir, "los_cv_metrics_summary.csv"), index=False)
    pd.concat(fold_predictions, ignore_index=True).to_csv(
        os.path.join(results_dir, "los_cv_predictions.csv"), index=False
    )

    return folds, fold_predictions, {"best_state_dict": best_state_dict, "best_score": best_overall_linear_kappa}, params


def train_final_single_task(model_builder, X_train, y_train, static_train, seq_train, params, device):
    train_ds = SingleTaskDataset(X_train, seq_train, static_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss(weight=build_class_weights(y_train, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)

    model.train()
    for epoch in range(50):
        for xb, sl, sb, yb in train_loader:
            xb, sl, sb, yb = xb.to(device), sl.to(device), sb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, sl, sb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


def evaluate_single_task_test(model, X_test, y_test, static_test, seq_test, icu_test, params, device, results_dir):
    test_ds = SingleTaskDataset(X_test, seq_test, static_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)
    model.eval()
    preds, truths, ids = [], [], []
    with torch.no_grad():
        for i, (xb, sl, sb, yb) in enumerate(test_loader):
            xb, sl, sb, yb = xb.to(device), sl.to(device), sb.to(device), yb.to(device)
            logits = model(xb, sl, sb)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            truths.extend(yb.cpu().numpy())
            start = i * params["batch_size"]
            ids.extend(icu_test[start : start + len(yb)])

    # Save test preds
    df_test = pd.DataFrame({"ICUSTAY_ID": ids, "true_class": truths, "predicted_class": preds})
    df_test.to_csv(os.path.join(results_dir, "best_test_predictions.csv"), index=False)

    metrics = {
        "kappa": cohen_kappa_score(truths, preds),
        "linear_kappa": cohen_kappa_score(truths, preds, weights="linear"),
        "quadratic_kappa": cohen_kappa_score(truths, preds, weights="quadratic"),
        "accuracy": accuracy_score(truths, preds),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, "test_metrics_summary.csv"), index=False)
    return metrics, df_test


def bootstrap_single_task(truths: List[int], preds: List[int], n_bootstrap: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(truths)
    def ci(x):
        lower = np.percentile(x, 2.5)
        upper = np.percentile(x, 97.5)
        return lower, upper

    ks, ls, qs, accs = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t = np.array(truths)[idx]
        p = np.array(preds)[idx]
        ks.append(cohen_kappa_score(t, p))
        ls.append(cohen_kappa_score(t, p, weights="linear"))
        qs.append(cohen_kappa_score(t, p, weights="quadratic"))
        accs.append(accuracy_score(t, p))

    out = {
        "kappa_mean": float(np.mean(ks)),
        "kappa_ci": ci(ks),
        "linear_kappa_mean": float(np.mean(ls)),
        "linear_kappa_ci": ci(ls),
        "quadratic_kappa_mean": float(np.mean(qs)),
        "quadratic_kappa_ci": ci(qs),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_ci": ci(accs),
    }
    return out


# -------------------- I/O helpers --------------------

def save_metrics_table(metrics: Dict[str, Any], path: str):
    pd.DataFrame([metrics]).to_csv(path, index=False)


def save_predictions_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
