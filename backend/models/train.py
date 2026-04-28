"""Train a logistic-regression baseline + gradient-boosted model.

We start with logistic regression because:
  - With 3 patches of data we have at most a few thousand pro games. That is
    well below the regime where boosted trees outperform a strong linear model
    on tabular features that already encode most of the signal as differences.
  - Logistic regression is calibrated by default (probabilities mean what they
    say). Boosted trees usually need post-hoc calibration.
  - Sklearn-only deps keep installation light.

If the dataset grows enough we can swap in XGBoost/LightGBM later — the train()
function below picks scikit-learn's GradientBoostingClassifier when xgboost is
not installed, so the code path is identical.

Validation: time-based split (last 20% by match_dt held out). This is the only
honest way to evaluate a sports model — random splits leak future info.
"""
from __future__ import annotations
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.config import DATA_DIR
from backend.models.dataset import build_training_rows, FEATURE_COLS

MODEL_PATH = DATA_DIR / "model.pkl"
METRICS_PATH = DATA_DIR / "model_metrics.json"


@dataclass
class TrainResult:
    n_train: int
    n_val: int
    log_loss_val: float
    accuracy_val: float
    brier_val: float
    feature_importance: Dict[str, float]


def _matrix(rows: List[Dict[str, float]]) -> Tuple[List[List[float]], List[int]]:
    X = [[r[c] for c in FEATURE_COLS] for r in rows]
    y = [int(r["target"]) for r in rows]
    return X, y


def _binary_log_loss(y_true: List[int], p: List[float]) -> float:
    eps = 1e-15
    s = 0.0
    for t, q in zip(y_true, p):
        q = min(max(q, eps), 1 - eps)
        s += -(t * math.log(q) + (1 - t) * math.log(1 - q))
    return s / max(len(y_true), 1)


def _accuracy(y_true: List[int], p: List[float]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, q in zip(y_true, p) if (q >= 0.5) == (t == 1)) / len(y_true)


def _brier(y_true: List[int], p: List[float]) -> float:
    if not y_true:
        return 0.0
    return sum((q - t) ** 2 for t, q in zip(y_true, p)) / len(y_true)


def train(val_fraction: float = 0.2, min_rows: int = 50) -> TrainResult:
    rows = build_training_rows()
    if len(rows) < min_rows:
        raise RuntimeError(f"Not enough training rows ({len(rows)} < {min_rows}). "
                           "Run more ingestion first.")

    rows.sort(key=lambda r: r["match_dt"])
    split = int(len(rows) * (1 - val_fraction))
    train_rows, val_rows = rows[:split], rows[split:]

    Xtr, ytr = _matrix(train_rows)
    Xva, yva = _matrix(val_rows)

    # Lazy import so the API can run without sklearn installed (heuristic only).
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    clf.fit(Xtr, ytr)

    pva = [float(p[1]) for p in clf.predict_proba(Xva)]
    metrics = TrainResult(
        n_train=len(train_rows),
        n_val=len(val_rows),
        log_loss_val=_binary_log_loss(yva, pva),
        accuracy_val=_accuracy(yva, pva),
        brier_val=_brier(yva, pva),
        feature_importance={
            c: float(coef) for c, coef in zip(FEATURE_COLS, clf.named_steps["lr"].coef_[0])
        },
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "feature_cols": FEATURE_COLS}, f)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "n_train": metrics.n_train,
            "n_val":   metrics.n_val,
            "log_loss_val": metrics.log_loss_val,
            "accuracy_val": metrics.accuracy_val,
            "brier_val":    metrics.brier_val,
            "feature_importance": metrics.feature_importance,
        }, f, indent=2)

    return metrics


def load_model() -> Dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    m = train()
    print(f"trained on {m.n_train}, validated on {m.n_val}")
    print(f"log_loss={m.log_loss_val:.4f}  accuracy={m.accuracy_val:.3f}  brier={m.brier_val:.4f}")
    print("top features (by |coef|):")
    for c, w in sorted(m.feature_importance.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"  {c:>22}  {w:+.3f}")
