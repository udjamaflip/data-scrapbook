"""
matching.py – Join blinker, exposure, and incident datasets.

Matching strategy (in priority order):
  1. Exact match    – normalised make + model + model_year identical in all three
  2. Normalised     – after alias and case normalisation
  3. Manual override – rows explicitly mapped in manual_vehicle_overrides.csv
  4. Unmatched      – row appears in one dataset but cannot be joined to others
  5. Ambiguous      – multiple candidate matches found; row flagged, not collapsed

Design rules
  - Never silently collapse ambiguous rows.
  - Preserve all original values alongside normalised values.
  - All unmatched rows land in outputs/tables/unmatched_rows.csv.
  - The match_quality column is set for every row in the analysis dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ANALYSIS_DATASET_FILE,
    BLINKER_MASTER_FILE,
    EXPOSURE_FINAL_FILE,
    INCIDENT_FINAL_FILE,
    MATCH_AMBIGUOUS,
    MATCH_EXACT,
    MATCH_MANUAL,
    MATCH_NORMALIZED,
    MATCH_UNMATCHED,
)
from src.utils import write_csv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: build join key
# ---------------------------------------------------------------------------

_KEY_COLS = ["make_norm", "model_norm", "model_year"]


def _make_key(df: pd.DataFrame) -> pd.Series:
    """Concatenate normalised key columns into a single comparison string."""
    return (
        df["make_norm"].str.strip().str.lower()
        + "|"
        + df["model_norm"].str.strip().str.lower()
        + "|"
        + df["model_year"].astype(str)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_datasets(
    blinker: pd.DataFrame,
    exposure: pd.DataFrame,
    incidents: pd.DataFrame,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join the three datasets into a single analysis-ready table.

    Parameters
    ----------
    blinker : pd.DataFrame
        Output of clean_blinker_data().
    exposure : pd.DataFrame
        Output of clean_exposure_data().
    incidents : pd.DataFrame
        Output of clean_incident_data().
    save : bool
        If True, write final datasets and unmatched report.

    Returns
    -------
    (analysis_df, unmatched_df)
        analysis_df  – merged, enriched dataset ready for analysis.py
        unmatched_df – rows that could not be matched.
    """
    logger.info(
        "Starting match: blinker=%d rows, exposure=%d rows, incidents=%d rows",
        len(blinker),
        len(exposure),
        len(incidents),
    )

    # ------------------------------------------------------------------ #
    # Step 1 – Deduplicate blinker to one canonical row per vehicle cohort #
    # (take the highest-confidence colour assignment per make/model/year)  #
    # ------------------------------------------------------------------ #
    blinker_dedup = _deduplicate_blinker(blinker)

    # ------------------------------------------------------------------ #
    # Step 2 – Aggregate exposure to per-vehicle-cohort totals             #
    # ------------------------------------------------------------------ #
    exposure_agg = _aggregate_exposure(exposure)

    # ------------------------------------------------------------------ #
    # Step 3 – Aggregate incidents to per-cohort-per-year totals           #
    # ------------------------------------------------------------------ #
    # Incidents stay at (make, model, model_year, incident_year) grain.
    # We will join blinker and exposure onto this table.

    # ------------------------------------------------------------------ #
    # Step 4 – Join blinker → incidents (left join from incidents)         #
    # ------------------------------------------------------------------ #
    merged = incidents.merge(
        blinker_dedup[
            [
                "make_norm",
                "model_norm",
                "model_year",
                "rear_signal_color_standardized",
                "confidence_score",
                "eligibility_for_core_analysis",
                "source",
                "ambiguous_flag",
                "mixed_flag",
                "data_quality_notes",
                "DEMO_DATA",
            ]
        ],
        on=_KEY_COLS,
        how="left",
        suffixes=("_incident", "_blinker"),
    )

    # Flag rows where blinker could not be matched
    no_blinker = merged["rear_signal_color_standardized"].isna()
    merged.loc[no_blinker, "rear_signal_color_standardized"] = "unknown"
    merged.loc[no_blinker, "eligibility_for_core_analysis"] = False

    # ------------------------------------------------------------------ #
    # Step 5 – Join exposure onto merged table                             #
    # ------------------------------------------------------------------ #
    merged = merged.merge(
        exposure_agg[
            [
                "make_norm",
                "model_norm",
                "model_year",
                "registered_vehicle_years",
                "exposure_source",
                "exposure_quality_flag",
                "cumulative_rvy",
            ]
        ],
        on=_KEY_COLS,
        how="left",
        suffixes=("", "_exposure"),
    )

    no_exposure = merged["registered_vehicle_years"].isna()
    if no_exposure.any():
        logger.warning(
            "%d rows have no matching exposure data — rates will be NaN for these.",
            no_exposure.sum(),
        )

    # ------------------------------------------------------------------ #
    # Step 6 – Assign match_quality                                        #
    # ------------------------------------------------------------------ #
    merged["match_quality"] = _assign_match_quality(merged, no_blinker, no_exposure)

    # ------------------------------------------------------------------ #
    # Step 7 – Collect unmatched rows                                      #
    # ------------------------------------------------------------------ #
    unmatched_mask = merged["match_quality"] == MATCH_UNMATCHED
    unmatched = merged[unmatched_mask].copy()
    n_unmatched = len(unmatched)

    # Also find blinker rows with no matching incident rows
    blinker_keys = set(zip(blinker_dedup["make_norm"], blinker_dedup["model_norm"], blinker_dedup["model_year"]))
    incident_keys = set(zip(incidents["make_norm"], incidents["model_norm"], incidents["model_year"]))
    blinker_only_keys = blinker_keys - incident_keys
    if blinker_only_keys:
        logger.warning(
            "%d blinker vehicle cohorts have no matching incident rows.",
            len(blinker_only_keys),
        )

    logger.info(
        "Match complete: %d total rows, %d unmatched (%.1f%%)",
        len(merged),
        n_unmatched,
        100 * n_unmatched / max(len(merged), 1),
    )

    # ------------------------------------------------------------------ #
    # Step 8 – Save final datasets                                         #
    # ------------------------------------------------------------------ #
    if save:
        write_csv(blinker_dedup, BLINKER_MASTER_FILE, "blinker_master")
        write_csv(exposure_agg, EXPOSURE_FINAL_FILE, "exposure_clean")
        write_csv(incidents, INCIDENT_FINAL_FILE, "incidents_clean")
        write_csv(merged, ANALYSIS_DATASET_FILE, "analysis_dataset")
        _write_unmatched(unmatched, blinker_dedup, incidents, exposure_agg)

    return merged, unmatched


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deduplicate_blinker(blinker: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per (make_norm, model_norm, model_year), choosing the row
    with the highest confidence_score.

    When multiple conflicting colours exist at the same confidence, the row
    will already be labelled 'mixed' by clean_blinker_data.
    """
    return (
        blinker.sort_values("confidence_score", ascending=False)
        .drop_duplicates(subset=_KEY_COLS, keep="first")
        .reset_index(drop=True)
    )


def _aggregate_exposure(exposure: pd.DataFrame) -> pd.DataFrame:
    """
    Sum registered_vehicle_years across all calendar years for each vehicle
    cohort, returning one row per (make_norm, model_norm, model_year).
    """
    agg_cols = {"registered_vehicle_years": "sum", "exposure_source": "first",
                "exposure_quality_flag": "first"}
    if "cumulative_rvy" in exposure.columns:
        agg_cols["cumulative_rvy"] = "max"

    result = (
        exposure.groupby(_KEY_COLS, dropna=False)
        .agg(agg_cols)
        .reset_index()
    )
    # If cumulative_rvy not in exposure, derive from sum
    if "cumulative_rvy" not in result.columns:
        result["cumulative_rvy"] = result["registered_vehicle_years"]

    return result


def _assign_match_quality(
    df: pd.DataFrame,
    no_blinker: pd.Series,
    no_exposure: pd.Series,
) -> pd.Series:
    """
    Derive a match_quality label for each row.

    Priority:
      unmatched  > ambiguous > manual_override > normalized_exact > exact
    """
    quality = pd.Series(MATCH_EXACT, index=df.index)

    # Rows where normalisation used aliases
    if "normalisation_source" in df.columns:
        alias_mask = df["normalisation_source"].isin(["alias", "low"])
        quality[alias_mask] = MATCH_NORMALIZED

        manual_mask = df["normalisation_source"] == "manual_override"
        quality[manual_mask] = MATCH_MANUAL

    # Rows missing blinker OR exposure
    partial_mask = no_blinker | no_exposure
    quality[partial_mask] = MATCH_AMBIGUOUS

    # Rows missing both are truly unmatched
    unmatched_mask = no_blinker & no_exposure
    quality[unmatched_mask] = MATCH_UNMATCHED

    return quality


def _write_unmatched(
    unmatched: pd.DataFrame,
    blinker: pd.DataFrame,
    incidents: pd.DataFrame,
    exposure: pd.DataFrame,
) -> None:
    """Write a consolidated unmatched-rows report."""
    from src.config import OUTPUTS_TABLES

    # Blinker rows with no incident match
    inc_keys = set(zip(incidents["make_norm"], incidents["model_norm"], incidents["model_year"]))
    blinker_no_inc = blinker[
        ~blinker.apply(lambda r: (r["make_norm"], r["model_norm"], r["model_year"]) in inc_keys, axis=1)
    ].copy()
    blinker_no_inc["unmatched_reason"] = "blinker_has_no_incident_rows"

    # Exposure rows with no incident match
    exp_no_inc = exposure[
        ~exposure.apply(lambda r: (r["make_norm"], r["model_norm"], r["model_year"]) in inc_keys, axis=1)
    ].copy()
    exp_no_inc["unmatched_reason"] = "exposure_has_no_incident_rows"

    # Incident rows with no blinker match
    bk_keys = set(zip(blinker["make_norm"], blinker["model_norm"], blinker["model_year"]))
    inc_no_blinker = incidents[
        ~incidents.apply(lambda r: (r["make_norm"], r["model_norm"], r["model_year"]) in bk_keys, axis=1)
    ].copy()
    inc_no_blinker["unmatched_reason"] = "incident_has_no_blinker_row"

    parts = []
    for part in [blinker_no_inc, exp_no_inc, inc_no_blinker]:
        if not part.empty:
            parts.append(part[["make_norm", "model_norm", "model_year", "unmatched_reason"]])

    if parts:
        all_unmatched = pd.concat(parts, ignore_index=True).drop_duplicates()
    else:
        all_unmatched = pd.DataFrame(columns=["make_norm", "model_norm", "model_year", "unmatched_reason"])

    write_csv(
        all_unmatched,
        OUTPUTS_TABLES / "unmatched_rows.csv",
        "unmatched_rows",
    )
