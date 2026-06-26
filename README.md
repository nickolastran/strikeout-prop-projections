# MLB 2026 Strikeout Forecasting Model

Forecasts pitcher strikeout rate (**K/9**) for the remainder of the 2026 MLB
season from Statcast pitch-level data, using a regularized XGBoost regressor with
a strict temporal validation split and SHAP explainability.

## Architecture (modular by pipeline stage)

```
config.py            # dates, split boundaries, paths, hyperparameters
main.py              # orchestrates ingest -> features -> train -> explain
src/
  ingest.py          # STAGE 1: pybaseball Statcast pull, cache, clean
  features.py        # STAGE 2: pitch -> pitcher feature engineering + target
  train.py           # STAGE 3: temporal split, XGBoost, overfitting diagnostics
  explain.py         # STAGE 4: SHAP global + per-pitcher (Skubal / Skenes)
data/raw|processed/  # cached parquet
outputs/models|metrics|figures/
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py            # smoke test (fast, short date range)
python main.py --force    # force a fresh Statcast pull
```

## Smoke test vs. real forecast

`config.SMOKE_TEST = True` (default) pulls only a **two-week** slice so the whole
pipeline can be validated in seconds. The split is *degenerate* in this mode and
the metrics are not meaningful (K/9 measured over 2–4 days is pure noise → R²≈0).

For a real rest-of-2026 forecast:

```python
# config.py
SMOKE_TEST = False
```

Then `python main.py --force`. This pulls the full Mar–Jun season (~600K+ pitch
rows, several minutes) and uses the production temporal split:

| Split | Feature window | Target window (K/9) |
|-------|----------------|---------------------|
| Train | Mar 20 – Apr 30 | May |
| Valid | Mar 20 – May 31 | June |

This mirrors deployment — fit on the past, forecast the future — and avoids the
look-ahead leakage a random K-fold split would introduce.

## Features & sabermetric rationale

| Group | Features | Why it predicts strikeouts |
|-------|----------|----------------------------|
| **Stuff** | `velo_avg/max`, `spin_avg`, `hbreak_avg`, `vbreak_avg`, `extension_avg` | Velocity, spin and sharp late break are the physical inputs that miss bats; extension adds *perceived* velocity. |
| **Release** | `release_consistency` (1 / (1+σx+σz)) | A repeatable release point tunnels pitches and suppresses walks → sustained K ability. |
| **Outcome** | `whiff_rate`, `chase_rate` (O-Swing), `z_contact_rate`, `csw_rate`, `k_pct` | Whiff% is the best single K proxy; chase generates cheap strikes; **low** Z-Contact% is elite swing-and-miss. |
| **Situational** | `ip_per_game`, rolling `k9/whiff/kpct_last14d/30d` | Workload context + **recent form**, which dominates near-term K-rate forecasting. |

Target: **K/9** = strikeouts / (outs ÷ 3) × 9. Total rest-of-season SO is the
downstream multiply `K9 / 9 × projected_remaining_IP`.

## Overfitting controls

XGBoost is configured defensively (`max_depth=3`, `subsample`/`colsample=0.8`,
`reg_alpha=0.5`, `reg_lambda=2.0`, `min_child_weight=5`) with early stopping on
the validation fold. `train.py` prints an overfitting report comparing train vs.
validation RMSE and emits concrete remediation steps (stronger L1/L2, shallower
trees, feature pruning by SHAP) when the gap is large.

## Explainability

`explain.py` writes a global SHAP beeswarm (`outputs/figures/shap_global_summary.png`)
and per-pitcher waterfall plots for Tarik Skubal and Paul Skenes, attributing each
ace's predicted K/9 to individual features.
```
