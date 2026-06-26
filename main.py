"""
main.py
=======
End-to-end orchestrator: ingest -> features -> train -> explain.

    python main.py            # uses config.SMOKE_TEST setting
    python main.py --force    # force a fresh Statcast pull (ignore cache)

For a REAL rest-of-2026 forecast, set SMOKE_TEST = False in config.py first
(pulls the full Mar–Jun season; takes several minutes).
"""
from __future__ import annotations

import sys

import config
from src import ingest, train, explain


def main() -> None:
    force = "--force" in sys.argv
    print("=" * 70)
    print(f" MLB 2026 Strikeout Forecast  |  SMOKE_TEST={config.SMOKE_TEST}")
    print("=" * 70)

    # STAGE 1 — ingest & clean
    df = ingest.run(force=force)

    # STAGE 2+3 — features + temporal-split training (train() handles both)
    artifacts = train.train(df)

    # STAGE 4 — SHAP explainability
    explain.run(artifacts)

    print("\nDone. Artifacts in outputs/ (models, metrics, figures).")


if __name__ == "__main__":
    main()
