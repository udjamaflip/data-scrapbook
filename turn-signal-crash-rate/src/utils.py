"""
utils.py – Shared helpers used across every pipeline stage.

Covers:
  - CSV I/O with schema validation
  - Data quality flagging
  - Rate calculations
  - Bootstrap confidence intervals
  - Console summary formatting
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    RATE_DENOMINATOR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def read_csv_validated(
    path: Path,
    required_cols: Sequence[str] = (),
    optional_cols: Sequence[str] = (),
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load a CSV and validate that all *required_cols* are present.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any required column is missing from the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            f"  → Place the file at that location, or run with --demo to use "
            f"the built-in demo dataset."
        )

    df = pd.read_csv(path, dtype=dtype, low_memory=False, comment="#")
    logger.info("Loaded %d rows from %s", len(df), path.name)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"File '{path.name}' is missing required columns: {missing}\n"
            f"  → See data/raw/templates/ for the expected schema."
        )

    present_optional = [c for c in optional_cols if c in df.columns]
    absent_optional = [c for c in optional_cols if c not in df.columns]
    if absent_optional:
        logger.warning(
            "Optional columns absent in %s (will be filled with NaN): %s",
            path.name,
            absent_optional,
        )
        for col in absent_optional:
            df[col] = np.nan

    return df


def write_csv(df: pd.DataFrame, path: Path, description: str = "") -> None:
    """Write *df* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "Wrote %d rows × %d cols to %s%s",
        len(df),
        len(df.columns),
        path.name,
        f" ({description})" if description else "",
    )


# ---------------------------------------------------------------------------
# String normalisation helpers
# ---------------------------------------------------------------------------


def strip_and_lower(s: Any) -> str:
    """Strip whitespace and lowercase a value; return '' for null/non-string."""
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def coerce_year(val: Any, min_year: int = 1980, max_year: int = 2030) -> int | None:
    """
    Convert *val* to an integer year, returning None if conversion fails or
    the value is outside the plausible range.
    """
    try:
        year = int(float(val))
    except (TypeError, ValueError):
        return None
    if year < min_year or year > max_year:
        return None
    return year


# ---------------------------------------------------------------------------
# Data quality helpers
# ---------------------------------------------------------------------------


def flag_low_count(
    df: pd.DataFrame,
    group_col: str,
    count_col: str,
    threshold: int,
    flag_col: str = "low_count_flag",
) -> pd.DataFrame:
    """
    Add a boolean column *flag_col* that is True when the count for a group
    is below *threshold*.
    """
    df = df.copy()
    group_counts = df.groupby(group_col)[count_col].transform("sum")
    df[flag_col] = group_counts < threshold
    return df


def add_demo_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Mark every row with DEMO_DATA=True so demo rows are never silently merged."""
    df = df.copy()
    df["DEMO_DATA"] = True
    return df


# ---------------------------------------------------------------------------
# Rate calculations
# ---------------------------------------------------------------------------


def compute_rate(
    numerator: pd.Series,
    denominator: pd.Series,
    per: int = RATE_DENOMINATOR,
) -> pd.Series:
    """
    Return *numerator* / *denominator* × *per*, with NaN where denominator ≤ 0.

    This is the core normalisation: crashes per 100 000 registered vehicle-years.
    Raw crash counts are meaningless without this step because popular models
    appear in more crashes simply by being more common on the road.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(denominator > 0, numerator / denominator * per, np.nan)
    return pd.Series(rate, index=numerator.index)


def compute_all_rates(
    df: pd.DataFrame,
    rvy_col: str = "registered_vehicle_years",
) -> pd.DataFrame:
    """
    Add per-100k-RVY rate columns for every count column that exists in *df*.

    Only adds a rate column when the corresponding count column is present.
    """
    df = df.copy()
    count_to_rate = {
        "fatal_crash_count": "fatal_crashes_per_100k_rvy",
        "occupant_death_count": "deaths_per_100k_rvy",
        "total_incident_count": "incidents_per_100k_rvy",
        "claim_count": "claims_per_100k_rvy",
    }
    if rvy_col not in df.columns:
        logger.warning(
            "Column '%s' not found – rate columns will not be computed.", rvy_col
        )
        return df

    for count_col, rate_col in count_to_rate.items():
        if count_col in df.columns:
            df[rate_col] = compute_rate(df[count_col], df[rvy_col])
        else:
            logger.debug("Count column '%s' absent; skipping rate '%s'.", count_col, rate_col)

    return df


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    statistic: Any = np.mean,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Return a bootstrap percentile confidence interval (low, high) for *statistic*
    applied to *data*.

    Returns (NaN, NaN) when *data* has fewer than 2 non-NaN observations.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    boot_stats = [
        statistic(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(boot_stats, alpha))
    hi = float(np.quantile(boot_stats, 1.0 - alpha))
    return lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent samples.

    Uses the pooled standard deviation.  Returns NaN if either group has
    fewer than 2 observations.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Welch's t-test for two independent samples with unequal variances.

    Returns (t_statistic, p_value).  Returns (NaN, NaN) for degenerate inputs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    result = stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Mann-Whitney U test (non-parametric) for two independent samples.

    Returns (U_statistic, p_value).  Returns (NaN, NaN) for degenerate inputs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 1 or len(b) < 1:
        return float("nan"), float("nan")
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary_box(title: str, lines: Iterable[str]) -> None:
    """Print a simple ASCII box with a title and content lines."""
    body = list(lines)
    width = max(len(title), max((len(l) for l in body), default=0)) + 4
    bar = "-" * width
    print(f"\n+{bar}+")
    print(f"|  {title:<{width - 2}}|")
    print(f"+{bar}+")
    for line in body:
        wrapped = textwrap.wrap(line, width=width - 4) or [""]
        for wline in wrapped:
            print(f"|  {wline:<{width - 2}}|")
    print(f"+{bar}+\n")


def warn_small_sample(group_name: str, n: int, min_n: int) -> None:
    """Emit a warning when a statistical group is too small."""
    if n < min_n:
        logger.warning(
            "Small sample warning: group '%s' has only %d observations "
            "(minimum recommended: %d). Statistical results may be unreliable.",
            group_name,
            n,
            min_n,
        )
