"""One-shot refresh: re-run ingestion across all configured leagues, then
retrain the model. Convenient for cron jobs / weekly re-pulls.

Usage:
    python -m backend.refresh
"""
from __future__ import annotations
import sys

from backend.ingestion.run import run_ingestion
from backend.models.train import train


def main() -> int:
    print("[refresh] starting ingestion…", flush=True)
    summary = run_ingestion()
    print(f"[refresh] ingestion: matches_added={summary['matches_added']} "
          f"seen={summary['matches_seen']} error={summary['error']}", flush=True)
    if summary["error"]:
        return 1

    print("[refresh] training model…", flush=True)
    try:
        m = train()
        print(f"[refresh] model: n_train={m.n_train} n_val={m.n_val} "
              f"acc={m.accuracy_val:.3f} log_loss={m.log_loss_val:.4f}", flush=True)
    except Exception as e:
        print(f"[refresh] training failed (non-fatal): {e}", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
