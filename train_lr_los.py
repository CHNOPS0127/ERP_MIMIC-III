#!/usr/bin/env python3
# train_logreg_los.py
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, make_scorer
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

from ml_utils import (
    ensure_dir, drop_bad_cols, bootstrap_metrics, ci, one_hot_align
)

warnings.filterwarnings("ignore")

def train_logreg(
    root: str,
    use_lr_features: bool = False,
    out_dir: str | None = None,
    n_splits_cv: int = 3,
    n_bootstrap: int = 100,
):
    """Train multinomial Logistic Regression on LOS buckets."""
    preproc = os.path.join(root, "preprocessed")
    feats_dir = os.path.join(preproc, "features")
    out_dir = out_dir or os.path.join(root, "results", "LR")
    ensure_dir(out_dir)

    # choose inputs
    if use_lr_features:
        train_fp = os.path.join(feats_dir, "features_lr_train.csv")
        test_fp  = os.path.join(feats_dir, "features_lr_test.csv")
        do_scale_here = False
    else:
        train_fp = os.path.join(feats_dir, "features_rf_train.csv")
        test_fp  = os.path.join(feats_dir, "features_rf_test.csv")
        do_scale_here = True

    # load
    train_df = pd.read_csv(train_fp)
    test_df  = pd.read_csv(test_fp)

    if "LOS_BUCKET" not in train_df.columns:
        raise ValueError("Train file must include LOS_BUCKET. Run feature_engineering.py first.")
    train_df = train_df.dropna(subset=["LOS_BUCKET"]).reset_index(drop=True)
    test_has_target = "LOS_BUCKET" in test_df.columns

    id_col = "ICUSTAY_ID" if "ICUSTAY_ID" in train_df.columns else None

    # build X,y
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


    # optional scaling for RF features
    scaler = None
    if do_scale_here:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # feature selection
    base_model = LogisticRegression(max_iter=1000, class_weight="balanced",
                                    multi_class="multinomial", solver="lbfgs")
    rfecv = RFECV(
        estimator=base_model,
        step=10,
        cv=StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    rfecv.fit(X_train, y_train)

    sel_mask = rfecv.support_
    selected_features = X_train.columns[sel_mask].tolist()
    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]

    # save selected features
    pd.Series(selected_features, name="feature").to_csv(
        os.path.join(out_dir, "selected_features.csv"), index=False
    )

    # grid search (optimize QWK)
    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
    param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial"),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42),
        scoring=kappa_scorer,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train_sel, y_train)

    # save cv results + model + scaler
    pd.DataFrame(grid.cv_results_).to_csv(os.path.join(out_dir, "grid_search_results.csv"), index=False)
    best_model = grid.best_estimator_
    joblib.dump(best_model, os.path.join(out_dir, "best_logistic_model.pkl"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    # test predictions
    y_pred  = best_model.predict(X_test_sel)
    y_proba = best_model.predict_proba(X_test_sel)
    classes = best_model.classes_

    results = pd.DataFrame({"y_pred": y_pred})
    for i, cls in enumerate(classes):
        results[f"prob_class_{cls}"] = y_proba[:, i]
    if id_col and id_col in test_df.columns:
        results[id_col] = test_df[id_col].values
    if test_has_target:
        results["y_true"] = y_test.values

    # order columns
    order = []
    if id_col and id_col in results.columns: order.append(id_col)
    if "y_true" in results.columns: order.append("y_true")
    order.append("y_pred")
    order.extend([c for c in results.columns if c.startswith("prob_class_")])
    results = results[order]
    results.to_csv(os.path.join(out_dir, "logistic_test_predictions.csv"), index=False)

    # point metrics + bootstrap
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

        boots = bootstrap_metrics(list(y_test.values), list(y_pred), n_bootstrap=n_bootstrap)
        summary = {}
        for k, vals in boots.items():
            mean, lo, hi, sd = ci(vals, 0.95)
            summary[k] = {"mean": mean, "lower_ci": lo, "upper_ci": hi, "std": sd}
            print(f"{k}: {mean:.4f} (95% CI [{lo:.4f}, {hi:.4f}], sd={sd:.4f})")

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

# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser(description="Train multinomial Logistic Regression on LOS buckets.")
    p.add_argument("root", type=str, help="Root that contains preprocessed/features/")
    p.add_argument("--use_lr_features", action="store_true",
                   help="Use features_lr_train/test.csv (already standardized). Default uses RF features and scales here.")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <root>/results/LR)")
    p.add_argument("--cv_splits", type=int, default=3, help="CV splits for RFECV and GridSearch (default 3)")
    p.add_argument("--bootstrap", type=int, default=100, help="Bootstrap iterations (default 100)")
    return p.parse_args()

def main():
    args = parse_args()
    train_logreg(
        root=args.root,
        use_lr_features=args.use_lr_features,
        out_dir=args.out_dir,
        n_splits_cv=args.cv_splits,
        n_bootstrap=args.bootstrap,
    )

if __name__ == "__main__":
    main()
