"""
src/train.py
============
STAGE 3 — Model training & temporal validation.

Trains an XGBoost regressor to predict pitcher K/9 and validates it with a
strict TEMPORAL split: features/targets drawn from March–May train the model;
June is held out for validation. This mirrors real deployment (fit on the past,
forecast the future) and — unlike random K-fold — does not leak future form.

An overfitting diagnostic compares train vs. validation error and prints
concrete remediation guidance (regularization / feature pruning) when the gap
is large.

Run standalone:  python -m src.train
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import config
from src import features, ingest

warnings.filterwarnings("ignore")


def _windows(df: pd.DataFrame):
    """Return (train_windows, valid_windows) as (feat_start, feat_end, out_start,
    out_end) tuples.

    Real mode: train features = Mar 20–Apr 30 -> target = May; valid features =
    Mar 20–May 31 -> target = June. Both honor "train on Mar–May, validate June".

    Smoke mode: only a couple of weeks of data exist, so we split the available
    span into early/late halves to preserve the same past->future structure on a
    miniature scale (enough to prove the pipeline runs end to end).
    """
    lo, hi = df["game_date"].min(), df["game_date"].max()
    span_days = (hi - lo).days

    if not config.SMOKE_TEST and span_days > 60:
        season_start = pd.Timestamp(config.SEASON_START)
        train = (season_start, pd.Timestamp("2026-04-30"),
                 pd.Timestamp("2026-05-01"), pd.Timestamp(config.TRAIN_END))
        valid = (season_start, pd.Timestamp(config.TRAIN_END),
                 pd.Timestamp(config.VALID_START), pd.Timestamp(config.AS_OF_DATE))
        return train, valid

    # --- smoke / short-span fallback ---
    q1 = lo + (hi - lo) * 0.40
    q2 = lo + (hi - lo) * 0.55
    q3 = lo + (hi - lo) * 0.70
    train = (lo, q1, q1, q2)
    valid = (lo, q3, q3, hi)
    print("[train] SMOKE_TEST split (degenerate, for pipeline validation only)")
    return train, valid


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def _overfit_report(train_m: dict, valid_m: dict) -> dict:
    """Flag overfitting and emit remediation guidance."""
    gap = valid_m["rmse"] - train_m["rmse"]
    ratio = valid_m["rmse"] / train_m["rmse"] if train_m["rmse"] else np.inf
    overfit = ratio > 1.5 and gap > 0.5
    guidance = []
    if overfit:
        guidance = [
            "Validation RMSE >> train RMSE: model is overfitting.",
            "Regularization: increase reg_alpha (L1) / reg_lambda (L2); lower max_depth to 2.",
            "Capacity: reduce n_estimators or learning_rate; raise min_child_weight.",
            "Sampling: lower subsample / colsample_bytree (e.g. 0.6).",
            "Feature pruning: drop low-SHAP / highly-collinear features (see explain.py).",
        ]
    else:
        guidance = ["Train/validation gap within tolerance — no overfitting flagged."]
    return {"train_rmse": train_m["rmse"], "valid_rmse": valid_m["rmse"],
            "gap": float(gap), "ratio": float(ratio),
            "overfitting": bool(overfit), "guidance": guidance}


def train(df: pd.DataFrame | None = None) -> dict:
    """Full training routine. Returns a dict with the model, feature list,
    datasets, and metrics (also persisted to disk)."""
    if df is None:
        df = ingest.run()

    (tf0, tf1, to0, to1), (vf0, vf1, vo0, vo1) = _windows(df)

    # In smoke mode relax the sample thresholds (tiny windows -> few pitches).
    min_p = 20 if config.SMOKE_TEST else config.MIN_PITCHES
    min_t = 5 if config.SMOKE_TEST else config.MIN_BATTERS_FACED

    train_tbl = features.build_dataset(df, tf0, tf1, to0, to1, min_p, min_t)
    valid_tbl = features.build_dataset(df, vf0, vf1, vo0, vo1, min_p, min_t)

    feat_cols = features.feature_columns(train_tbl)
    # Align validation columns to the training feature set.
    feat_cols = [c for c in feat_cols if c in valid_tbl.columns]

    X_tr, y_tr = train_tbl[feat_cols], train_tbl[config.TARGET]
    X_va, y_va = valid_tbl[feat_cols], valid_tbl[config.TARGET]

    if len(X_tr) < 5 or len(X_va) < 3:
        raise RuntimeError(
            f"Too few samples to train (train={len(X_tr)}, valid={len(X_va)}). "
            "Run with SMOKE_TEST=False and a full-season pull for real results."
        )

    model = XGBRegressor(
        eval_metric="rmse",
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        **config.XGB_PARAMS,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    train_m = _metrics(y_tr, model.predict(X_tr))
    valid_m = _metrics(y_va, model.predict(X_va))
    overfit = _overfit_report(train_m, valid_m)

    # Persist artifacts.
    model.save_model(config.MODEL_FILE)
    report = {
        "target": config.TARGET,
        "n_features": len(feat_cols),
        "features": feat_cols,
        "train_metrics": train_m,
        "valid_metrics": valid_m,
        "overfit_report": overfit,
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
    }
    config.METRICS_FILE.write_text(json.dumps(report, indent=2))

    print("\n=== TEMPORAL VALIDATION (train: Mar–May | validate: June) ===")
    print(f" train  : RMSE={train_m['rmse']:.3f}  MAE={train_m['mae']:.3f}  "
          f"R2={train_m['r2']:.3f}  n={train_m['n']}")
    print(f" valid  : RMSE={valid_m['rmse']:.3f}  MAE={valid_m['mae']:.3f}  "
          f"R2={valid_m['r2']:.3f}  n={valid_m['n']}")
    print(f" overfit: {overfit['overfitting']}  (valid/train RMSE ratio="
          f"{overfit['ratio']:.2f})")
    for line in overfit["guidance"]:
        print("   -", line)
    print(f"[train] model -> {config.MODEL_FILE}")
    print(f"[train] metrics -> {config.METRICS_FILE}")

    return {"model": model, "features": feat_cols,
            "train_tbl": train_tbl, "valid_tbl": valid_tbl, "report": report}


if __name__ == "__main__":
    train()