"""
src/features.py
===============
STAGE 2 — Feature engineering.

Transforms the cleaned PITCH-level Statcast frame into a PITCHER-level feature
matrix. Every feature here is chosen for a concrete sabermetric reason, noted
inline. The output is one row per qualifying pitcher with:

  * "Stuff" features      — what the pitch physically does (velo, spin, break,
                            release consistency).
  * "Outcome" features    — whiff%, chase% (O-Swing), Z-Contact% — the swing
                            decisions / contact-quality the stuff induces.
  * "Situational" features — workload (IP/start) and recent-form rolling rates.

The target (K/9) is computed over a *separate, later* time window than the
features so the model learns "current profile -> FUTURE strikeout rate" rather
than memorizing the present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# ----------------------------------------------------------------------------
# Statcast `description` taxonomy
# ----------------------------------------------------------------------------
# A "swing" is any pitch the batter offered at. Whiffs are swings that missed.
SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "foul", "foul_bunt", "hit_into_play", "missed_bunt", "foul_pitchout",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt",
}

# Map terminal-PA `events` to outs recorded, so we can estimate innings pitched
# from pitch data alone (IP = total outs / 3).
OUTS_BY_EVENT = {
    "strikeout": 1, "strikeout_double_play": 2,
    "field_out": 1, "force_out": 1, "fielders_choice_out": 1,
    "fielders_choice": 1, "other_out": 1, "sac_fly": 1, "sac_bunt": 1,
    "grounded_into_double_play": 2, "double_play": 2, "sac_fly_double_play": 2,
    "sac_bunt_double_play": 2, "triple_play": 3,
}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}


def _rate(numer: float, denom: float) -> float:
    """Safe rate that returns NaN on a zero denominator (no fake 0%)."""
    return numer / denom if denom else np.nan


def _pitch_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Attach boolean/int helper columns used by the aggregations."""
    df = df.copy()
    desc = df["description"]
    df["is_swing"] = desc.isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = desc.isin(WHIFF_DESCRIPTIONS)
    # Statcast `zone`: 1-9 are inside the strike zone, 11-14 are outside it.
    df["in_zone"] = df["zone"].le(9)
    df["out_zone"] = df["zone"].ge(11)
    df["is_terminal"] = df["events"].notna()           # last pitch of a PA
    df["is_strikeout"] = df["events"].isin(STRIKEOUT_EVENTS)
    df["outs_recorded"] = df["events"].map(OUTS_BY_EVENT).fillna(0).astype(int)
    return df


def aggregate_window(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one time-window of pitches to per-pitcher STUFF + OUTCOME rows.

    Returns a DataFrame indexed by `pitcher`.
    """
    df = _pitch_flags(df)
    g = df.groupby("pitcher")

    out = pd.DataFrame(index=g.size().index)
    out["player_name"] = g["player_name"].first()
    out["pitches"] = g.size()
    out["tbf"] = g["is_terminal"].sum()                # batters faced
    out["games"] = g["game_date"].nunique()

    # ---- STUFF: physical pitch characteristics (averaged over all pitches) ----
    # Velocity and spin are the canonical "stuff" inputs; higher generally maps
    # to more swing-and-miss.
    out["velo_avg"] = g["release_speed"].mean()
    out["velo_max"] = g["release_speed"].max()
    out["spin_avg"] = g["release_spin_rate"].mean()
    # Movement (pfx in feet). We take magnitude of horizontal break and the
    # signed vertical break; sharper movement misses more bats.
    out["hbreak_avg"] = g["pfx_x"].apply(lambda s: s.abs().mean())
    out["vbreak_avg"] = g["pfx_z"].mean()
    out["extension_avg"] = g["release_extension"].mean()  # perceived velocity

    # Release-point CONSISTENCY: low stdev of release x/z = repeatable delivery,
    # which suppresses walks and tunnels pitches -> sustained strikeout ability.
    out["release_x_std"] = g["release_pos_x"].std()
    out["release_z_std"] = g["release_pos_z"].std()
    out["release_consistency"] = 1.0 / (
        1.0 + out["release_x_std"].fillna(0) + out["release_z_std"].fillna(0)
    )  # higher = more consistent (0-1ish)

    # ---- OUTCOME: swing decisions & contact the stuff induces ----
    swings = g["is_swing"].sum()
    whiffs = g["is_whiff"].sum()
    # Whiff% = swings-and-misses / swings. The single best pitch-level proxy for
    # strikeout skill.
    out["whiff_rate"] = [
        _rate(w, s) for w, s in zip(whiffs, swings)
    ]

    # Chase% (O-Swing%) = swings at pitches OUT of the zone / out-of-zone pitches.
    # Pitchers who get chases generate cheap strikes and expand counts.
    out_zone_swings = g.apply(lambda x: (x["is_swing"] & x["out_zone"]).sum())
    out_zone_pitches = g["out_zone"].sum()
    out["chase_rate"] = [
        _rate(s, p) for s, p in zip(out_zone_swings, out_zone_pitches)
    ]

    # Z-Contact% = contact on IN-zone swings / in-zone swings. LOW z-contact is
    # elite — even when hitters swing at strikes, they miss.
    in_zone_swings = g.apply(lambda x: (x["is_swing"] & x["in_zone"]).sum())
    in_zone_contact = g.apply(
        lambda x: (x["is_swing"] & x["in_zone"] & ~x["is_whiff"]).sum()
    )
    out["z_contact_rate"] = [
        _rate(c, s) for c, s in zip(in_zone_contact, in_zone_swings)
    ]
    # Called-strike + whiff rate (CSW%) — a robust, count-stable stuff metric.
    called_strikes = g.apply(lambda x: (x["description"] == "called_strike").sum())
    out["csw_rate"] = [
        _rate(cs + w, p) for cs, w, p in zip(called_strikes, whiffs, out["pitches"])
    ]

    # ---- SITUATIONAL: workload ----
    outs = g["outs_recorded"].sum()
    ip = outs / 3.0                                    # estimated innings pitched
    out["ip"] = ip
    out["ip_per_game"] = out["ip"] / out["games"].replace(0, np.nan)

    return out


def compute_target(df: pd.DataFrame) -> pd.Series:
    """K/9 over a window: strikeouts / innings_pitched * 9 (per pitcher)."""
    df = _pitch_flags(df)
    g = df.groupby("pitcher")
    k = g["is_strikeout"].sum()
    ip = g["outs_recorded"].sum() / 3.0
    k9 = (k / ip.replace(0, np.nan)) * 9.0
    return k9.rename("k_per_9")


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Inclusive date slice helper."""
    return df[(df["game_date"] >= pd.Timestamp(start))
              & (df["game_date"] <= pd.Timestamp(end))]


def build_dataset(
    df: pd.DataFrame,
    feat_start, feat_end,
    out_start, out_end,
    min_pitches: int = config.MIN_PITCHES,
    min_tbf: int = config.MIN_BATTERS_FACED,
) -> pd.DataFrame:
    """Assemble a modeling table: features from the feature window joined to the
    K/9 target measured in a strictly LATER outcome window.

    Only pitchers meeting the minimum-sample thresholds in the feature window are
    kept (rate stats on tiny samples are noise, not signal).
    """
    feat_df = _slice(df, feat_start, feat_end)
    out_df = _slice(df, out_start, out_end)

    X = aggregate_window(feat_df)

    # --- recent-form rolling features (last 14 / 30 days ending at feat_end) ---
    # Rationale: a pitcher's MOST RECENT form is more predictive of his near-term
    # strikeout rate than his season-long average (form, health, mechanical tweaks).
    for w in config.ROLL_WINDOWS:
        sub = _slice(df, pd.Timestamp(feat_end) - pd.Timedelta(days=w), feat_end)
        if len(sub):
            recent = aggregate_window(sub)
            sub_flags = _pitch_flags(sub).groupby("pitcher")
            X[f"k9_last{w}d"] = compute_target(sub)
            X[f"whiff_last{w}d"] = recent["whiff_rate"]
            # K% over the recent window (strikeouts / batters faced).
            X[f"kpct_last{w}d"] = (
                sub_flags["is_strikeout"].sum() / sub_flags["is_terminal"].sum()
            )
        else:
            X[f"k9_last{w}d"] = np.nan
            X[f"whiff_last{w}d"] = np.nan
            X[f"kpct_last{w}d"] = np.nan

    # Season-to-date (feature window) K% as well.
    feat_flags = _pitch_flags(feat_df).groupby("pitcher")
    X["k_pct"] = feat_flags["is_strikeout"].sum() / feat_flags["is_terminal"].sum()

    # --- attach target ---
    X["k_per_9"] = compute_target(out_df)

    # --- sample filters ---
    keep = (X["pitches"] >= min_pitches) & (X["tbf"] >= min_tbf)
    X = X[keep]
    # Must have a measurable target in the outcome window.
    X = X[X["k_per_9"].notna() & np.isfinite(X["k_per_9"])]

    print(f"[features] window feat=({feat_start}->{feat_end}) "
          f"out=({out_start}->{out_end}) -> {len(X)} pitchers")
    return X.reset_index()


# Columns fed to the model (everything numeric except identifiers/target/leaky).
def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"pitcher", "player_name", "k_per_9",
            "release_x_std", "release_z_std"}  # folded into release_consistency
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


if __name__ == "__main__":
    from src import ingest
    data = ingest.run()
    lo, hi = data["game_date"].min(), data["game_date"].max()
    mid = lo + (hi - lo) / 2
    demo = build_dataset(data, lo, mid, mid, hi, min_pitches=20, min_tbf=5)
    print(demo.head().to_string(index=False))
