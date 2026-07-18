"""
src/daily_projections.py
========================
Daily strikeout-prop projection table.

For every probable starting pitcher on a given date, project the full
distribution of strikeouts and report the probability of clearing each
threshold (1+ K through 12+ K) — the standard "K prop" board.

Modeling
--------
A pitcher's strikeouts in a start are modeled as

    SO ~ Binomial(n = expected batters faced, p = expected K rate)

where `p` (K%) is his season strikeout rate per plate appearance and `n` is his
typical batters faced per start. Then

    Prob(j+ K) = P(SO >= j) = binom.sf(j - 1, n, p)

Binomial (vs. plain Poisson) is the natural choice because each batter faced is
one Bernoulli trial that either ends in a strikeout or not.

Data sources
------------
  * Rate profiles  -> season Statcast (via src.ingest) — K% and BF/start.
  * Daily slate    -> MLB StatsAPI probable pitchers + opponents.

Run:  python -m src.daily_projections 2026-06-25
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import binom

import config
from src import ingest
from src.features import _pitch_flags

warnings.filterwarnings("ignore")

# Strikeout thresholds to report (columns "Prob 1+ K" ... "Prob 12+ K").
THRESHOLDS = list(range(1, 13))
OUT_DIR = config.ROOT / "outputs" / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Season rate profiles (per pitcher) from Statcast
# ----------------------------------------------------------------------------
def build_rate_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher K% and batters-faced-per-start from season pitch data.

    Returns a frame indexed by MLBAM `pitcher` id with k_rate, bf_per_start,
    and supporting counts. Only the games a pitcher actually appeared in are
    used to estimate his per-start batters-faced workload.
    """
    fl = _pitch_flags(df)
    g = fl.groupby("pitcher")
    prof = pd.DataFrame(index=g.size().index)
    prof["player_name"] = g["player_name"].first()
    prof["tbf"] = g["is_terminal"].sum()                 # batters faced (season)
    prof["k"] = g["is_strikeout"].sum()
    prof["games"] = g["game_date"].nunique()
    # K% — strikeouts per plate appearance. The "p" of our binomial.
    prof["k_rate"] = prof["k"] / prof["tbf"]
    # Batters faced per appearance — the "n" of our binomial.
    prof["bf_per_start"] = prof["tbf"] / prof["games"]
    return prof


# ----------------------------------------------------------------------------
# 2. Daily slate of probable starters
# ----------------------------------------------------------------------------
def get_slate(date_iso: str) -> pd.DataFrame:
    """Probable starters for `date_iso` (YYYY-MM-DD) with team, opponent and
    MLBAM id (resolved via player lookup so we can join to Statcast)."""
    import statsapi

    d = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%m/%d/%Y")
    games = statsapi.schedule(date=d)

    rows = []
    for gm in games:
        for side, opp_side in (("home", "away"), ("away", "home")):
            name = gm.get(f"{side}_probable_pitcher")
            if not name or name.strip() in ("", "TBD"):
                continue
            rows.append({
                "pitcher_name": name.strip(),
                "team": gm.get(f"{side}_name"),
                "opponent": gm.get(f"{opp_side}_name"),
            })

    # Resolve MLBAM ids (cache lookups to avoid duplicate API calls).
    id_cache: dict[str, int | None] = {}
    for r in rows:
        nm = r["pitcher_name"]
        if nm not in id_cache:
            hits = statsapi.lookup_player(nm)
            id_cache[nm] = int(hits[0]["id"]) if hits else None
        r["pitcher"] = id_cache[nm]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Strikeout distribution
# ----------------------------------------------------------------------------
def threshold_probs(k_rate: float, bf: float) -> dict[int, float]:
    """Prob(j+ K) for each threshold under SO ~ Binomial(round(bf), k_rate)."""
    n = int(round(bf))
    return {j: float(binom.sf(j - 1, n, k_rate)) for j in THRESHOLDS}


# ----------------------------------------------------------------------------
# 4. Assemble the board
# ----------------------------------------------------------------------------
def project(date_iso: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the full projection table for a date."""
    if df is None:
        df = ingest.run()
    profiles = build_rate_profiles(df)

    # League-average fallbacks for pitchers without enough season data (rookies,
    # call-ups). Weight by batters faced so the average reflects real innings.
    lg_k_rate = float(profiles["k"].sum() / profiles["tbf"].sum())
    # Restrict the BF fallback to starter-like workloads (>=15 BF/appearance) so
    # relievers don't drag the "expected batters faced" default down.
    starters = profiles[profiles["bf_per_start"] >= 15]
    lg_bf = float(starters["bf_per_start"].median()) if len(starters) else 22.0

    slate = get_slate(date_iso)
    if slate.empty:
        print(f"[projections] no probable starters found for {date_iso}")
        return pd.DataFrame()

    records = []
    for _, r in slate.iterrows():
        pid = r["pitcher"]
        if pid in profiles.index and profiles.loc[pid, "tbf"] >= 20:
            k_rate = float(profiles.loc[pid, "k_rate"])
            bf = float(profiles.loc[pid, "bf_per_start"])
            src = "season"
        else:
            k_rate, bf, src = lg_k_rate, lg_bf, "lg_avg"  # insufficient sample

        probs = threshold_probs(k_rate, bf)
        rec = {
            "Pitcher": r["pitcher_name"],
            "Team": r["team"],
            "Opponent": r["opponent"],
            "Expected K Rate": round(k_rate, 3),
            "Expected Batters Faced": round(bf, 1),
            "_exp_k": k_rate * bf,        # for sorting
            "_src": src,
        }
        rec.update({f"Prob {j}+ K": probs[j] for j in THRESHOLDS})
        records.append(rec)

    board = (pd.DataFrame(records)
             .sort_values("_exp_k", ascending=False)
             .reset_index(drop=True))
    board.index = board.index + 1            # 1-based rank like the reference
    return board


# ----------------------------------------------------------------------------
# 5. Rendering
# ----------------------------------------------------------------------------
def _styled(board: pd.DataFrame):
    """Green-gradient styled HTML table matching the reference look."""
    prob_cols = [f"Prob {j}+ K" for j in THRESHOLDS]
    show = board.drop(columns=["_exp_k", "_src"])
    sty = (show.style
           .format({c: "{:.3f}" for c in prob_cols})
           .format({"Expected K Rate": "{:.3f}", "Expected Batters Faced": "{:.1f}"})
           .background_gradient(cmap="Greens", subset=prob_cols, vmin=0, vmax=1)
           .set_caption("Projected Starting Pitcher Strikeouts"))
    return sty


def save(board: pd.DataFrame, date_iso: str) -> tuple:
    csv_path = OUT_DIR / f"strikeouts_{date_iso}.csv"
    html_path = OUT_DIR / f"strikeouts_{date_iso}.html"
    board.drop(columns=["_exp_k", "_src"]).to_csv(csv_path)
    _styled(board).to_html(html_path)
    return csv_path, html_path


def run(date_iso: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    board = project(date_iso, df=df)
    if board.empty:
        return board
    csv_path, html_path = save(board, date_iso)

    prob_cols = [f"Prob {j}+ K" for j in THRESHOLDS]
    disp = board.drop(columns=["_exp_k", "_src"]).copy()
    for c in prob_cols + ["Expected K Rate"]:
        disp[c] = disp[c].map(lambda v: f"{v:.3f}")
    pd.set_option("display.max_columns", None, "display.width", 250)
    print(disp.to_string())
    print(f"\n[projections] {len(board)} starters | "
          f"{(board['_src'] == 'lg_avg').sum()} on league-avg fallback")
    print(f"[projections] CSV  -> {csv_path}")
    print(f"[projections] HTML -> {html_path}")
    return board


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else config.AS_OF_DATE
    run(date_arg)