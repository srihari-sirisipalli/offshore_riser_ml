import numpy as np
import pandas as pd
from typing import Dict, Any


def cv_fold_consistency(cv_scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Summarize CV fold consistency for metrics.
    Expects cv_scores like {"cmae": np.array([...]), "crmse": np.array([...])}
    """
    rows = []
    for metric, scores in cv_scores.items():
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            continue
        rows.append({
            "metric": metric,
            "folds": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "range": float(np.max(scores) - np.min(scores)),
        })
    return pd.DataFrame(rows)


def overfitting_gaps(train_scores: Dict[str, float], val_scores: Dict[str, float]) -> pd.DataFrame:
    """
    Calculate overfitting gaps between train and validation for key metrics.
    """
    rows = []
    for metric in set(train_scores.keys()) | set(val_scores.keys()):
        train_val = train_scores.get(metric)
        val_val = val_scores.get(metric)
        if train_val is None or val_val is None:
            continue
        gap = val_val - train_val
        rows.append({
            "metric": metric,
            "train": train_val,
            "val": val_val,
            "gap": gap,
        })
    return pd.DataFrame(rows)
