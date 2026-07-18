"""
config.py
=========
Central configuration for the MLB strikeout-forecasting pipeline.

Keeping dates, paths, the temporal-split boundary, and model hyperparameters in
one place makes the ingestion / processing / training scripts purely mechanical
and easy to re-run with different settings.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Season window
# ----------------------------------------------------------------------------
# We are forecasting the REMAINDER of the 2026 season, so we train on everything
# from Opening Day through "today" (2026-06-25). pybaseball's Statcast endpoint
# expects ISO date strings.
SEASON = 2026
SEASON_START = "2026-03-20"      # 2026 Opening Day window (spring carryover trimmed in cleaning)
AS_OF_DATE = "2026-06-25"        # "current" date — everything after this is the future we predict

# ----------------------------------------------------------------------------
# Temporal validation split
# ----------------------------------------------------------------------------
# Sabermetric rationale: a *random* K-fold split would leak future information
# into the training set (a pitcher's June form is partly explained by his May
# form). A temporal split mimics how the model is actually used in production:
# fit on the past, predict the future. Train on Mar–May, validate on June.
TRAIN_END = "2026-05-31"
VALID_START = "2026-06-01"

# ----------------------------------------------------------------------------
# Smoke-test mode
# ----------------------------------------------------------------------------
# When SMOKE_TEST is True the ingest step pulls only a short date range so the
# full pipeline can be validated end-to-end in seconds instead of minutes.
SMOKE_TEST = True
SMOKE_START = "2026-04-01"
SMOKE_END = "2026-04-14"

# ----------------------------------------------------------------------------
# Population filters
# ----------------------------------------------------------------------------
# Only model pitchers with a meaningful sample. A pitcher who has thrown 12
# pitches all year produces noise, not signal.
MIN_PITCHES = 200          # minimum total pitches to be included in the cohort
MIN_BATTERS_FACED = 50     # minimum plate appearances faced (TBF) for stable rate stats

# Rolling-window lengths (in days) for recent-form features.
ROLL_WINDOWS = [14, 30]

# ----------------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------------
# Decision (confirmed): model the RATE stat K/9 directly. Total rest-of-season
# strikeouts is then a clean downstream multiply (K9 / 9 * projected IP).
TARGET = "k_per_9"

# Pitchers we want individual SHAP explanations for.
SPOTLIGHT_PITCHERS = ["Tarik Skubal", "Paul Skenes"]

# ----------------------------------------------------------------------------
# XGBoost hyperparameters
# ----------------------------------------------------------------------------
# Conservative defaults chosen to resist overfitting on a small (a few hundred
# pitchers) tabular dataset: shallow trees, strong subsampling, L1/L2 penalties.
XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,            # shallow trees -> lower variance
    subsample=0.8,          # row subsampling -> regularization
    colsample_bytree=0.8,   # feature subsampling -> regularization
    min_child_weight=5,     # require >=5 effective samples per leaf
    reg_alpha=0.5,          # L1 penalty (feature pruning pressure)
    reg_lambda=2.0,         # L2 penalty
    random_state=42,
    n_jobs=-1,
)
# Early stopping rounds used during fit against the validation fold.
EARLY_STOPPING_ROUNDS = 30

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_MODELS = ROOT / "outputs" / "models"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_FIGURES = ROOT / "outputs" / "figures"

RAW_STATCAST = DATA_RAW / "statcast_2026.parquet"
FEATURE_MATRIX = DATA_PROCESSED / "pitcher_features.parquet"
MODEL_FILE = OUT_MODELS / "xgb_k9.json"
METRICS_FILE = OUT_METRICS / "validation_metrics.json"

for _p in (DATA_RAW, DATA_PROCESSED, OUT_MODELS, OUT_METRICS, OUT_FIGURES):
    _p.mkdir(parents=True, exist_ok=True)