"""
src/ingest.py
=============
STAGE 1 — Data ingestion & cleaning.

Pulls 2026 pitch-level Statcast data via pybaseball, caches it to parquet so we
never hammer Baseball Savant twice, and performs first-pass cleaning:
  * drop rows that are not real pitches,
  * drop deprecated / fully-null columns,
  * coerce the handful of numeric "stuff" columns we rely on downstream.

Run standalone:  python -m src.ingest
"""
from __future__ import annotations

import sys
import warnings

import pandas as pd

import config

# pybaseball is noisy and occasionally emits FutureWarnings from its pandas usage.
warnings.filterwarnings("ignore")


# Columns we actually need downstream. Pulling the full ~90-column Statcast frame
# is fine, but we validate that these critical ones exist.
REQUIRED_COLS = [
    "game_date", "pitcher", "player_name", "p_throws",
    "release_speed", "release_spin_rate", "release_extension",
    "release_pos_x", "release_pos_z",
    "pfx_x", "pfx_z",                 # horizontal / vertical movement (feet)
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "description", "events", "zone", "type", "balls", "strikes",
    "inning", "at_bat_number", "pitch_number",
]


def _fetch_statcast(start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around pybaseball.statcast (imported lazily so config-only
    imports of this module don't pay the import cost)."""
    from pybaseball import statcast

    print(f"[ingest] pulling Statcast {start} -> {end} (this can take a few minutes)...")
    df = statcast(start_dt=start, end_dt=end)
    print(f"[ingest] pulled {len(df):,} raw pitch rows")
    return df


def load_raw(force: bool = False) -> pd.DataFrame:
    """Return the raw Statcast frame, using the on-disk cache when available.

    Honors config.SMOKE_TEST to pull only a short window for fast end-to-end
    validation of the pipeline.
    """
    if config.RAW_STATCAST.exists() and not force:
        print(f"[ingest] loading cached raw data from {config.RAW_STATCAST}")
        return pd.read_parquet(config.RAW_STATCAST)

    if config.SMOKE_TEST:
        start, end = config.SMOKE_START, config.SMOKE_END
        print("[ingest] SMOKE_TEST mode: pulling a short date range only")
    else:
        start, end = config.SEASON_START, config.AS_OF_DATE

    df = _fetch_statcast(start, end)
    df.to_parquet(config.RAW_STATCAST, index=False)
    print(f"[ingest] cached raw data to {config.RAW_STATCAST}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """First-pass cleaning of the raw pitch frame.

    Sabermetric note: we keep the data at PITCH granularity here — every row is
    one pitch. Aggregation to the pitcher level happens in features.py. The job
    of this function is only to guarantee each surviving row is a valid pitch
    with the fields we need.
    """
    n0 = len(df)

    # 1. Validate critical columns are present (Statcast schema drift guard).
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"[ingest] Statcast frame missing expected columns: {missing}")

    # 2. Keep only rows that represent an actual pitch. `type` is S/B/X (strike/
    #    ball/in-play); deprecated rows or game-state rows have nulls here.
    df = df[df["type"].notna()].copy()

    # 3. A real pitch must identify the pitcher and have a measured velocity.
    #    Rows with null pitcher id or null release_speed are tracking dropouts
    #    or non-pitch events (e.g. pickoffs) — not useful for our model.
    df = df[df["pitcher"].notna() & df["release_speed"].notna()].copy()
    df["pitcher"] = df["pitcher"].astype(int)

    # 4. Constrain to the season-of-interest window. (Cache may contain spring
    #    or future rows; AS_OF_DATE is our hard "present" boundary.)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[
        (df["game_date"] >= pd.Timestamp(config.SEASON_START))
        & (df["game_date"] <= pd.Timestamp(config.AS_OF_DATE))
    ].copy()

    # 5. Coerce the numeric "stuff" columns; median-impute the small fraction of
    #    tracking dropouts so a single missing spin reading doesn't drop a pitch.
    numeric_stuff = [
        "release_speed", "release_spin_rate", "release_extension",
        "release_pos_x", "release_pos_z", "pfx_x", "pfx_z",
        "plate_x", "plate_z", "sz_top", "sz_bot",
    ]
    for col in numeric_stuff:
        # Cast to float64 — some Statcast columns (e.g. release_spin_rate) arrive
        # as pandas nullable Int64, which rejects float median imputation.
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    # Impute per-pitcher where possible (a knuckleballer's spin != an ace's),
    # falling back to the global median.
    df[numeric_stuff] = (
        df.groupby("pitcher")[numeric_stuff]
        .transform(lambda s: s.fillna(s.median()))
    )
    df[numeric_stuff] = df[numeric_stuff].fillna(df[numeric_stuff].median())

    print(f"[ingest] cleaned: {n0:,} -> {len(df):,} rows "
          f"({df['pitcher'].nunique():,} distinct pitchers)")
    return df


def run(force: bool = False) -> pd.DataFrame:
    """Ingest entry point: fetch (or load cache) -> clean -> return."""
    raw = load_raw(force=force)
    return clean(raw)


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    cleaned = run(force=force_flag)
    print(cleaned[["game_date", "player_name", "release_speed",
                   "description"]].head(10).to_string(index=False))
