import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any


def compare_rounds(
    baseline_errors: np.ndarray,
    candidate_errors: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform paired statistical significance tests between two error arrays.
    Returns p-values and effect sizes.
    """
    baseline_errors = np.asarray(baseline_errors)
    candidate_errors = np.asarray(candidate_errors)

    # Drop NaNs
    mask = ~np.isnan(baseline_errors) & ~np.isnan(candidate_errors)
    if not mask.any():
        return {"p_value_ttest": np.nan, "p_value_wilcoxon": np.nan, "cohens_d": np.nan, "significant": False}

    base = baseline_errors[mask]
    cand = candidate_errors[mask]

    diff = cand - base
    # If identical (or all zero diff), skip tests to avoid warnings
    if np.allclose(diff, 0, equal_nan=True):
        return {
            "p_value_ttest": np.nan,
            "p_value_wilcoxon": np.nan,
            "cohens_d": 0.0,
            "significant": False,
        }

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(base, cand, nan_policy="omit")

    # Wilcoxon signed-rank (requires non-zero differences)
    try:
        p_wilcoxon = stats.wilcoxon(base, cand, zero_method="wilcox", correction=False).pvalue
    except ValueError:
        p_wilcoxon = np.nan

    # Effect size (Cohen's d for paired)
    d = diff.mean() / (diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan)

    significant = (not np.isnan(p_wilcoxon) and p_wilcoxon < alpha) or (not np.isnan(p_ttest) and p_ttest < alpha)

    return {
        "p_value_ttest": float(p_ttest) if not np.isnan(p_ttest) else np.nan,
        "p_value_wilcoxon": float(p_wilcoxon) if not np.isnan(p_wilcoxon) else np.nan,
        "cohens_d": float(d) if not np.isnan(d) else np.nan,
        "significant": bool(significant),
    }
