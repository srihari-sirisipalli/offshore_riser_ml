import numpy as np
import pandas as pd
from typing import Dict

SAFETY_TIERS = {
    "CRITICAL": (15, np.inf),
    "WARNING": (10, 15),
    "ACCEPTABLE": (0, 10),
}


def safety_threshold_summary(errors: pd.Series) -> pd.DataFrame:
    """
    Categorize absolute errors into safety tiers and summarize counts/percentages.
    """
    total = len(errors)
    rows = []
    for tier, (low, high) in SAFETY_TIERS.items():
        mask = (errors >= low) & (errors < high)
        count = int(mask.sum())
        pct = (count / total) * 100 if total else 0.0
        rows.append({"tier": tier, "count": count, "percentage": pct})
    return pd.DataFrame(rows)
