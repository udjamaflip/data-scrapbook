"""
analysis.py – Compute normalized crash rates and statistical comparisons.

This module answers the research question:
  "Do US-market vehicles with non-amber rear turn signals have higher
   fatal crash rates than vehicles with amber rear turn signals,
   after normalizing for fleet exposure?"

Key design choices
  - Rates are always per 100 000 registered vehicle-years (RVY).
  - Raw crash counts are NEVER treated as meaningful without normalisation.
  - The core headline comparison excludes 'mixed' and 'unknown' rows.
  - Sensitivity analyses include those rows to bound the uncertainty.
  - Statistical tests used: Welch's t-test (parametric) and Mann-Whitney U
    (non-parametric). Both are reported; neither is "the truth".
  - Effect size is Cohen's d.
  - Bootstrap 95% confidence intervals are computed on the mean rate difference.
  - Sample size warnings are emitted when n < MIN_SAMPLE_SIZE per group.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ANALYSIS_YEAR_MIN,
    ANALYSIS_YEAR_MAX,
    COLOR_AMBER,
    COLOR_NON_AMBER,
    CORE_ANALYSIS_COLORS,
    MIN_CONFIDENCE,
    MIN_SAMPLE_SIZE,
    OUTPUTS_TABLES,
    RATE_DENOMINATOR,
    ROLLING_WINDOW,
)
from src.utils import (
    bootstrap_ci,
    cohens_d,
    compute_all_rates,
    mann_whitney_u,
    warn_small_sample,
    welch_t_test,
    write_csv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_analysis(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str]:
    """
    Full analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Output of matching.match_datasets() – the joined analysis dataset.

    Returns
    -------
    (results, enriched_df, rate_col)
      results     – dict mapping table name → DataFrame
      enriched_df – analysis_df with rate columns added
      rate_col    – name of the primary rate column used
    """
    logger.info("Starting analysis on %d rows.", len(df))

    # Compute normalised rates
    df = _prepare_rates(df)

    # Determine primary rate column (prefer fatal crashes; fall back to incidents)
    rate_col = _primary_rate_col(df)
    logger.info("Primary rate column for headline analysis: '%s'", rate_col)

    # --- Core dataset (amber vs non_amber, high-confidence only) ---
    core = _core_subset(df)

    results: dict[str, pd.DataFrame] = {}

    # Per-colour summary
    results["summary_by_signal_color"] = _summary_by_color(core, rate_col)

    # Per-year summary
    results["summary_by_year"] = _summary_by_year(df, core, rate_col)

    # Per-manufacturer summary
    results["summary_by_manufacturer"] = _summary_by_manufacturer(core, rate_col)

    # Statistical comparison
    results["statistical_comparison"] = _statistical_comparison(core, rate_col)

    # Sensitivity analyses
    results["sensitivity_analysis"] = _sensitivity_analysis(df, rate_col)

    # Data quality report
    results["data_quality_report"] = _data_quality_report(df)

    # Save all tables
    for name, tdf in results.items():
        write_csv(tdf, OUTPUTS_TABLES / f"{name}.csv", name)

    _log_headline(results.get("statistical_comparison"), results.get("summary_by_signal_color"), rate_col)

    return results, df, rate_col


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-100k-RVY rate columns if not already present."""
    df = df.copy()
    df = compute_all_rates(df, rvy_col="registered_vehicle_years")
    return df


def _primary_rate_col(df: pd.DataFrame) -> str:
    """Choose the best available rate column."""
    for col in [
        "fatal_crashes_per_100k_rvy",
        "deaths_per_100k_rvy",
        "incidents_per_100k_rvy",
        "claims_per_100k_rvy",
    ]:
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError(
        "No usable rate column found in analysis dataset. "
        "Ensure at least one count column and registered_vehicle_years are present."
    )


def _core_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows eligible for the headline comparison:
      - colour is amber or non_amber
      - eligibility_for_core_analysis is True
      - confidence_score >= MIN_CONFIDENCE
    """
    mask = (
        df["rear_signal_color_standardized"].isin(CORE_ANALYSIS_COLORS)
        & df.get("eligibility_for_core_analysis", pd.Series(True, index=df.index))
        & (df.get("confidence_score", pd.Series(1.0, index=df.index)) >= MIN_CONFIDENCE)
    )
    core = df[mask].copy()
    logger.info(
        "Core analysis subset: %d rows (%d amber, %d non_amber).",
        len(core),
        (core["rear_signal_color_standardized"] == COLOR_AMBER).sum(),
        (core["rear_signal_color_standardized"] == COLOR_NON_AMBER).sum(),
    )
    return core


def _group_stats(
    values: pd.Series,
    group_label: str,
) -> dict[str, Any]:
    """Compute descriptive statistics for a 1-D series of rates."""
    arr = values.dropna().values
    n = len(arr)
    warn_small_sample(group_label, n, MIN_SAMPLE_SIZE)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "min": np.nan, "max": np.nan}
    ci_lo, ci_hi = bootstrap_ci(arr)
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else np.nan,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _summary_by_color(core: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    """Grouped mean/median/CI for amber vs non_amber over the full period."""
    rows = []
    for color in [COLOR_AMBER, COLOR_NON_AMBER]:
        vals = core.loc[core["rear_signal_color_standardized"] == color, rate_col]
        stats = _group_stats(vals, color)
        rows.append({"signal_color": color, "rate_col": rate_col, **stats})
    return pd.DataFrame(rows)


def _summary_by_year(
    full: pd.DataFrame,
    core: pd.DataFrame,
    rate_col: str,
) -> pd.DataFrame:
    """Per calendar year comparison."""
    year_col = "incident_year" if "incident_year" in core.columns else "model_year"
    rows = []
    all_years = sorted(
        y for y in core[year_col].dropna().unique()
        if ANALYSIS_YEAR_MIN <= y <= ANALYSIS_YEAR_MAX
    )
    for yr in all_years:
        yr_df = core[core[year_col] == yr]
        for color in [COLOR_AMBER, COLOR_NON_AMBER]:
            vals = yr_df.loc[yr_df["rear_signal_color_standardized"] == color, rate_col]
            stats = _group_stats(vals, f"{color}/{yr}")
            rows.append({"year": yr, "signal_color": color, "rate_col": rate_col, **stats})

    result = pd.DataFrame(rows)

    # Rolling average
    if not result.empty:
        for color in [COLOR_AMBER, COLOR_NON_AMBER]:
            mask = result["signal_color"] == color
            result.loc[mask, f"rolling_{ROLLING_WINDOW}yr_mean"] = (
                result.loc[mask, "mean"]
                .rolling(ROLLING_WINDOW, min_periods=1)
                .mean()
                .values
            )

    return result


def _summary_by_manufacturer(core: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    """Per-manufacturer comparison for major manufacturers."""
    rows = []
    makes = core["make_norm"].value_counts()
    # Keep manufacturers that appear in both colour groups
    eligible_makes = []
    for make in makes.index:
        mdf = core[core["make_norm"] == make]
        colors_present = set(mdf["rear_signal_color_standardized"].unique())
        if colors_present >= {COLOR_AMBER, COLOR_NON_AMBER}:
            eligible_makes.append(make)
        elif len(mdf) >= 5:
            eligible_makes.append(make)

    for make in eligible_makes:
        for color in [COLOR_AMBER, COLOR_NON_AMBER]:
            vals = core.loc[
                (core["make_norm"] == make)
                & (core["rear_signal_color_standardized"] == color),
                rate_col,
            ]
            stats = _group_stats(vals, f"{make}/{color}")
            rows.append({"manufacturer": make, "signal_color": color, "rate_col": rate_col, **stats})

    return pd.DataFrame(rows)


def _statistical_comparison(core: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    """
    Two-group statistical tests: amber vs non_amber.

    Returns a single-row DataFrame with all test statistics.
    """
    amber_vals = core.loc[
        core["rear_signal_color_standardized"] == COLOR_AMBER, rate_col
    ].dropna().values
    non_amber_vals = core.loc[
        core["rear_signal_color_standardized"] == COLOR_NON_AMBER, rate_col
    ].dropna().values

    t_stat, t_p = welch_t_test(amber_vals, non_amber_vals)
    u_stat, u_p = mann_whitney_u(amber_vals, non_amber_vals)
    d = cohens_d(amber_vals, non_amber_vals)

    # Bootstrap CI on the mean difference (non_amber - amber)
    diff_samples = []
    rng = np.random.default_rng(42)
    for _ in range(2000):
        a_boot = rng.choice(amber_vals, size=len(amber_vals), replace=True) if len(amber_vals) > 0 else np.array([np.nan])
        b_boot = rng.choice(non_amber_vals, size=len(non_amber_vals), replace=True) if len(non_amber_vals) > 0 else np.array([np.nan])
        diff_samples.append(float(np.nanmean(b_boot)) - float(np.nanmean(a_boot)))
    diff_ci_lo = float(np.quantile(diff_samples, 0.025))
    diff_ci_hi = float(np.quantile(diff_samples, 0.975))

    row = {
        "rate_col": rate_col,
        "n_amber": len(amber_vals),
        "n_non_amber": len(non_amber_vals),
        "mean_amber": float(np.mean(amber_vals)) if len(amber_vals) > 0 else np.nan,
        "mean_non_amber": float(np.mean(non_amber_vals)) if len(non_amber_vals) > 0 else np.nan,
        "mean_diff_non_amber_minus_amber": float(np.mean(non_amber_vals) - np.mean(amber_vals))
        if (len(amber_vals) > 0 and len(non_amber_vals) > 0) else np.nan,
        "diff_ci_lower_95": diff_ci_lo,
        "diff_ci_upper_95": diff_ci_hi,
        "welch_t_statistic": t_stat,
        "welch_t_pvalue": t_p,
        "mann_whitney_u_statistic": u_stat,
        "mann_whitney_u_pvalue": u_p,
        "cohens_d": d,
        "interpretation": _interpret_effect(d, t_p),
        "caution": (
            f"Small sample amber (n={len(amber_vals)}). " if len(amber_vals) < MIN_SAMPLE_SIZE else ""
        ) + (
            f"Small sample non_amber (n={len(non_amber_vals)}). " if len(non_amber_vals) < MIN_SAMPLE_SIZE else ""
        ),
    }
    return pd.DataFrame([row])


def _interpret_effect(d: float, p: float) -> str:
    """Produce a plain-language interpretation of effect size and significance."""
    if np.isnan(d) or np.isnan(p):
        return "insufficient data"
    sig = "statistically significant (p<0.05)" if p < 0.05 else "not statistically significant (p>=0.05)"
    if abs(d) < 0.2:
        size = "negligible effect size"
    elif abs(d) < 0.5:
        size = "small effect size"
    elif abs(d) < 0.8:
        size = "medium effect size"
    else:
        size = "large effect size"
    direction = "amber > non_amber" if d > 0 else "non_amber > amber"
    return f"{sig}; {size} ({direction}); Cohen's d={d:.3f}"


def _sensitivity_analysis(df: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    """
    Compare results under different inclusion criteria to test robustness.
    """
    scenarios: list[dict[str, Any]] = []

    def _run(label: str, subset: pd.DataFrame) -> None:
        for color in [COLOR_AMBER, COLOR_NON_AMBER]:
            vals = subset.loc[
                subset["rear_signal_color_standardized"] == color, rate_col
            ].dropna().values
            n = len(vals)
            mean = float(np.mean(vals)) if n > 0 else np.nan
            ci_lo, ci_hi = bootstrap_ci(vals) if n >= 2 else (np.nan, np.nan)
            scenarios.append({
                "scenario": label,
                "signal_color": color,
                "n": n,
                "mean_rate": mean,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            })

    # Scenario 1: core only (high-confidence amber/non_amber)
    _run("core_only", _core_subset(df))

    # Scenario 2: include mixed (mixed is counted as non_amber for bounding)
    df2 = df.copy()
    df2.loc[df2["rear_signal_color_standardized"] == "mixed", "rear_signal_color_standardized"] = COLOR_NON_AMBER
    _run("include_mixed_as_non_amber", df2[df2["rear_signal_color_standardized"].isin(CORE_ANALYSIS_COLORS)])

    # Scenario 3: high-confidence rows only (confidence >= 0.85)
    high_conf = df[
        df["rear_signal_color_standardized"].isin(CORE_ANALYSIS_COLORS)
        & (df.get("confidence_score", pd.Series(1.0, index=df.index)) >= 0.85)
    ]
    _run("high_confidence_only", high_conf)

    # Scenario 4: exact matches only
    if "match_quality" in df.columns:
        exact_only = df[
            df["rear_signal_color_standardized"].isin(CORE_ANALYSIS_COLORS)
            & (df["match_quality"] == "exact")
        ]
        _run("exact_match_only", exact_only)

    return pd.DataFrame(scenarios)


def _data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a summary of data quality issues."""
    rows = [
        {"metric": "total_rows", "value": len(df), "notes": "All rows in analysis dataset"},
        {"metric": "rows_with_color_unknown", "value": int((df["rear_signal_color_standardized"] == "unknown").sum()),
         "notes": "Cannot assign signal colour"},
        {"metric": "rows_with_color_mixed", "value": int((df["rear_signal_color_standardized"] == "mixed").sum()),
         "notes": "Model-year had both amber and non-amber trims"},
        {"metric": "rows_eligible_core_analysis",
         "value": int(df.get("eligibility_for_core_analysis", pd.Series(False)).sum()),
         "notes": "Amber or non_amber with confidence >= threshold"},
        {"metric": "rows_with_no_exposure", "value": int(df["registered_vehicle_years"].isna().sum()),
         "notes": "Rates cannot be computed; excluded from rate analysis"},
        {"metric": "rows_low_exposure_quality",
         "value": int((df.get("exposure_quality_flag", pd.Series("")) == "low").sum()),
         "notes": "Exposure from low-quality source"},
        {"metric": "rows_low_incident_quality",
         "value": int((df.get("incident_quality_flag", pd.Series("")) == "low").sum()),
         "notes": "Incident data from low-quality source"},
        {"metric": "rows_demo_data",
         "value": int(df.get("DEMO_DATA", pd.Series(False)).sum()),
         "notes": "SYNTHETIC demo rows – DO NOT use for real conclusions"},
        {"metric": "unique_makes", "value": df["make_norm"].nunique(), "notes": ""},
        {"metric": "unique_models", "value": df[["make_norm", "model_norm"]].drop_duplicates().shape[0], "notes": ""},
        {"metric": "year_range",
         "value": f"{df.get('incident_year', df['model_year']).min():.0f}–{df.get('incident_year', df['model_year']).max():.0f}",
         "notes": ""},
    ]
    return pd.DataFrame(rows)


def _log_headline(
    stats_df: pd.DataFrame | None,
    color_df: pd.DataFrame | None,
    rate_col: str,
) -> None:
    """Print a concise headline summary to the console."""
    logger.info("=" * 60)
    logger.info("HEADLINE ANALYSIS RESULT")
    logger.info("Rate column: %s (per %d registered vehicle-years)", rate_col, RATE_DENOMINATOR)

    if color_df is not None and not color_df.empty:
        for _, row in color_df.iterrows():
            logger.info(
                "  %-12s: mean=%.3f  median=%.3f  n=%d  95%%CI [%.3f, %.3f]",
                row["signal_color"],
                row.get("mean", float("nan")),
                row.get("median", float("nan")),
                int(row.get("n", 0)),
                row.get("ci_lower", float("nan")),
                row.get("ci_upper", float("nan")),
            )

    if stats_df is not None and not stats_df.empty:
        row = stats_df.iloc[0]
        logger.info("  Interpretation: %s", row.get("interpretation", ""))
        if row.get("caution"):
            logger.warning("  ⚠ CAUTION: %s", row.get("caution", ""))

    logger.info("=" * 60)
