#!/usr/bin/env python3



import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, make_scorer
)

from ml_utils import ensure_dir, drop_bad_cols, bootstrap_metrics, ci, one_hot_align

def train_rf(
    root: str,
    out_dir: str | None = None,
    use_lr_features: bool = False,
    cv_splits: int = 3,
    n_jobs: int = -1,
    bootstrap_iters: int = 1000,
):
    preproc = os.path.join(root, "preprocessed")
    feats_dir = os.path.join(preproc, "features")
    out_dir = out_dir or os.path.join(root, "results", "RF")
    ensure_dir(out_dir)

    # Pick train/test feature files
    if use_lr_features:
        train_fp = os.path.join(feats_dir, "features_lr_train.csv")
        test_fp  = os.path.join(feats_dir, "features_lr_test.csv")
    else:
        train_fp = os.path.join(feats_dir, "features_rf_train.csv")
        test_fp  = os.path.join(feats_dir, "features_rf_test.csv")

    # Load
    train_df = pd.read_csv(train_fp)
    test_df  = pd.read_csv(test_fp)

    if "LOS_BUCKET" not in train_df.columns:
        raise ValueError("Train file must include LOS_BUCKET. Run feature_engineering.py first.")
    train_df = train_df.dropna(subset=["LOS_BUCKET"]).reset_index(drop=True)
    test_has_target = "LOS_BUCKET" in test_df.columns

    id_col = "ICUSTAY_ID" if "ICUSTAY_ID" in train_df.columns else None

    # Build X/y
    drop_cols = {"LOS", "LOS_BUCKET"}
    if id_col: drop_cols.add(id_col)

    X_train = drop_bad_cols(train_df.drop(columns=list(drop_cols & set(train_df.columns)), errors="ignore")).copy()
    y_train = train_df["LOS_BUCKET"].astype(int)

    X_test = drop_bad_cols(test_df.drop(columns=list(drop_cols & set(test_df.columns)), errors="ignore")).copy()
    y_test = test_df["LOS_BUCKET"].astype(int) if test_has_target else None

    # Align and make numeric
    X_train, X_test = one_hot_align(X_train, X_test)

    # Targets should be ints
    y_train = y_train.astype(int)
    if test_has_target:
        y_test = y_test.astype(int)


    # ---- Grid search RF (optimize linear kappa, also log QWK/accuracy) ----
    kappa_lin = make_scorer(cohen_kappa_score, weights="linear")
    kappa_qwk = make_scorer(cohen_kappa_score, weights="quadratic")
    scoring = {
        "kappa_linear": kappa_lin,
        "kappa_quadratic": kappa_qwk,
        "accuracy": "accuracy",
    }
    param_grid = {
        "n_estimators": [100],
        "max_depth": [20, None],
        "class_weight": ["balanced"],
        # You can add more: "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=n_jobs),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42),
        scoring=scoring,
        refit="kappa_linear",
        n_jobs=n_jobs,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    # Save grid results
    cvres = pd.DataFrame(grid.cv_results_)
    cvres.to_csv(os.path.join(out_dir, "grid_search_results.csv"), index=False)

    print("\n=== Grid Search (val scores) ===")
    for params, lin, qwk, acc in zip(
        cvres["params"], cvres["mean_test_kappa_linear"],
        cvres["mean_test_kappa_quadratic"], cvres["mean_test_accuracy"]
    ):
        print(f"{params} -> linκ {lin:.4f} | qwk {qwk:.4f} | acc {acc:.4f}")
    print(f"Best: {grid.best_params_} | linκ {grid.best_score_:.4f}")

    # Save best model
    best_model = grid.best_estimator_
    joblib.dump(best_model, os.path.join(out_dir, "best_random_forest_model.pkl"))

    # ---- Test predictions ----
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    classes = best_model.classes_

    # Save predictions + probabilities
    results = pd.DataFrame({"y_pred": y_pred})
    for i, cls in enumerate(classes):
        results[f"prob_class_{cls}"] = y_proba[:, i]
    if id_col and id_col in test_df.columns:
        results[id_col] = test_df[id_col].values
    if test_has_target:
        results["y_true"] = y_test.values

    # Pretty order
    order = []
    if id_col and id_col in results.columns: order.append(id_col)
    if "y_true" in results.columns: order.append("y_true")
    order.append("y_pred")
    order.extend([c for c in results.columns if c.startswith("prob_class_")])
    results = results[order]
    results.to_csv(os.path.join(out_dir, "random_forest_test_probabilities.csv"), index=False)
    print("\nSaved predictions with probabilities.")

    # ---- Metrics + bootstrap CIs ----
    if test_has_target:
        acc = accuracy_score(y_test, y_pred)
        lin = cohen_kappa_score(y_test, y_pred, weights="linear")
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        cm  = confusion_matrix(y_test, y_pred)

        print("\n=== Test (point) ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Linear Kappa: {lin:.4f}")
        print(f"Quadratic Kappa: {qwk:.4f}")
        print(f"Confusion matrix:\n{cm}")

        print(f"\nBootstrap (n={bootstrap_iters})...")
        boots = bootstrap_metrics(list(y_test.values), list(y_pred), n_bootstrap=bootstrap_iters, seed=42)

        summary = {}
        for k, vals in boots.items():
            mean, lo, hi, sd = ci(vals, confidence=0.95)
            summary[k] = {"mean": mean, "lower_ci": lo, "upper_ci": hi, "std": sd}
            print(f"{k}: {mean:.4f} (95% CI [{lo:.4f}, {hi:.4f}], sd={sd:.4f})")

        # Save detailed files
        with open(os.path.join(out_dir, "metrics_test_bootstrap.txt"), "w") as f:
            f.write("=== Test point estimates ===\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Linear Kappa: {lin:.4f}\n")
            f.write(f"Quadratic Kappa: {qwk:.4f}\n")
            f.write("Confusion matrix:\n")
            f.write(np.array2string(cm))
            f.write("\n\n=== Bootstrap (95% CI) ===\n")
            for k, s in summary.items():
                f.write(f"{k}: {s['mean']:.4f} (95% CI [{s['lower_ci']:.4f}, {s['upper_ci']:.4f}], sd={s['std']:.4f})\n")

        pd.DataFrame(boots).to_csv(os.path.join(out_dir, "bootstrap_results.csv"), index=False)
        pd.DataFrame(summary).T.to_csv(os.path.join(out_dir, "bootstrap_summary_stats.csv"))

    print("\nDone. Artifacts in:", out_dir)

def parse_args():
    p = argparse.ArgumentParser(description="Train Random Forest on LOS buckets.")
    p.add_argument("root", type=str, help="Root that contains preprocessed/features/")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <root>/results/RF)")
    p.add_argument("--use_lr_features", action="store_true",
                   help="Use features_lr_train/test.csv instead of features_rf_*.csv.")
    p.add_argument("--cv_splits", type=int, default=3, help="CV splits (default 3)")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for RF/GridSearch (default -1)")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for CIs (default 1000)")
    return p.parse_args()

def main():
    args = parse_args()
    train_rf(
        root=args.root,
        out_dir=args.out_dir,
        use_lr_features=args.use_lr_features,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
        bootstrap_iters=args.bootstrap,
    )

if __name__ == "__main__":
    main()

