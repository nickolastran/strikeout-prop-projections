"""
src/explain.py
==============
STAGE 4 — Explainability (SHAP).

Answers the question the spec poses: *which features most drive a pitcher's
predicted strikeout rate?* — globally across the cohort, and locally for
spotlight aces (Tarik Skubal, Paul Skenes).

SHAP (SHapley Additive exPlanations) attributes each pitcher's predicted K/9 to
the individual features in an additive, game-theoretically fair way. For a tree
model we use the exact, fast TreeExplainer.

Run standalone (after train):  python -m src.explain
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")               # headless backend (no display in this env)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config

warnings.filterwarnings("ignore")


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """Mean(|SHAP|) per feature — the global driver ranking."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    imp = (pd.DataFrame({"feature": X.columns,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))

    # Beeswarm summary plot.
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.tight_layout()
    fig_path = config.OUT_FIGURES / "shap_global_summary.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[explain] global SHAP summary -> {fig_path}")
    return imp


def explain_pitcher(model, X: pd.DataFrame, names: pd.Series, who: str) -> dict | None:
    """Per-pitcher SHAP waterfall: what pushes THIS ace's predicted K/9 up/down.

    `names` aligns row-for-row with X. Match is case-insensitive substring so
    "Skubal" finds "Skubal, Tarik".
    """
    mask = names.str.contains(who.split()[-1], case=False, na=False)
    if not mask.any():
        print(f"[explain] '{who}' not found in cohort (insufficient sample or "
              f"outside date window) — skipping.")
        return None

    idx = np.where(mask.values)[0][0]
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)                       # Explanation object
    row = sv[idx]

    plt.figure()
    shap.plots.waterfall(row, max_display=12, show=False)
    plt.tight_layout()
    safe = who.replace(" ", "_").lower()
    fig_path = config.OUT_FIGURES / f"shap_{safe}.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    contribs = (pd.Series(row.values, index=X.columns)
                .sort_values(key=np.abs, ascending=False))
    pred = float(row.values.sum() + row.base_values)
    print(f"\n[explain] {names.iloc[idx]}  predicted K/9 ≈ {pred:.2f}")
    print("  top drivers (SHAP contribution to K/9):")
    for feat, val in contribs.head(6).items():
        arrow = "↑" if val > 0 else "↓"
        print(f"    {arrow} {feat:<18} {val:+.3f}")
    print(f"  waterfall -> {fig_path}")
    return {"pitcher": names.iloc[idx], "pred_k9": pred,
            "top_drivers": contribs.head(8).to_dict()}


def run(artifacts: dict) -> dict:
    """Generate global + spotlight explanations from train() artifacts."""
    model = artifacts["model"]
    # Explain on the validation cohort (the "current" pitchers we forecast).
    tbl = artifacts["valid_tbl"]
    feat_cols = artifacts["features"]
    X = tbl[feat_cols]
    names = tbl["player_name"]

    imp = global_importance(model, X)
    print("\n=== GLOBAL FEATURE IMPORTANCE (mean |SHAP|) ===")
    print(imp.head(12).to_string(index=False))

    spotlight = {}
    for who in config.SPOTLIGHT_PITCHERS:
        res = explain_pitcher(model, X, names, who)
        if res:
            spotlight[who] = res

    return {"global_importance": imp, "spotlight": spotlight}


if __name__ == "__main__":
    from src import train
    arts = train.train()
    run(arts)
