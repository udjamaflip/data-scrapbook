"""
clean_incident_data.py – Validate and standardise the incident dataset.

Key concerns:
  - FARS counts are reliable (census) but cover only fatal crashes.
  - CRSS counts are sampled; raw counts should NOT be used without weights.
  - Make/model coding varies across years and sources; flag low-confidence rows.
  - Never treat a missing count column as zero – use NaN to preserve honesty.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    ANALYSIS_YEAR_MAX,
    ANALYSIS_YEAR_MIN,
    INCIDENT_CLEAN_FILE,
    EXPOSURE_QUALITY_HIGH,
    EXPOSURE_QUALITY_LOW,
    EXPOSURE_QUALITY_MEDIUM,
)
from src.utils import write_csv

logger = logging.getLogger(__name__)

_SOURCE_QUALITY: dict[str, str] = {
    "nhtsa_fars": EXPOSURE_QUALITY_HIGH,
    "fars": EXPOSURE_QUALITY_HIGH,
    "nhtsa_crss": EXPOSURE_QUALITY_MEDIUM,
    "crss": EXPOSURE_QUALITY_MEDIUM,
    "nhtsa_complaints": EXPOSURE_QUALITY_LOW,
    "insurance_claims": EXPOSURE_QUALITY_MEDIUM,
    "hldi": EXPOSURE_QUALITY_HIGH,
    "demo": EXPOSURE_QUALITY_LOW,
    "synthetic": EXPOSURE_QUALITY_LOW,
}


def _quality_flag(source: object) -> str:
    if pd.isna(source):
        return EXPOSURE_QUALITY_LOW
    key = str(source).strip().lower().replace(" ", "_").replace("-", "_")
    return _SOURCE_QUALITY.get(key, EXPOSURE_QUALITY_MEDIUM)


def clean_incident_data(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Validate, filter, and enrich the incident dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_incident_data().
    save : bool
        Write cleaned output to data/intermediate/.

    Returns
    -------
    pd.DataFrame
        Cleaned incident data ready for matching.
    """
    df = df.copy()

    # --- Year filter ---
    before = len(df)
    df = df[
        df["incident_year"].between(ANALYSIS_YEAR_MIN, ANALYSIS_YEAR_MAX, inclusive="both")
        | df["incident_year"].isna()
    ]
    if len(df) < before:
        logger.info(
            "Dropped %d incident rows outside year range [%d, %d].",
            before - len(df),
            ANALYSIS_YEAR_MIN,
            ANALYSIS_YEAR_MAX,
        )

    # --- Count column validation ---
    count_cols = [
        "fatal_crash_count",
        "occupant_death_count",
        "total_incident_count",
        "claim_count",
    ]
    present_counts = [c for c in count_cols if c in df.columns]
    if not present_counts:
        raise ValueError(
            "Incident dataset has no count columns "
            f"({count_cols}). At least one is required."
        )

    # Ensure counts are non-negative integers; keep NaN (do NOT fill with 0)
    for col in present_counts:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negative = (df[col] < 0).sum()
        if negative:
            logger.warning(
                "%d negative values in '%s' – setting to NaN.", negative, col
            )
            df.loc[df[col] < 0, col] = float("nan")

    # Warn if CRSS data present without weights
    if "crss_weight" not in df.columns and "crss" in str(df["incident_source"].unique()).lower():
        logger.warning(
            "CRSS data detected but no 'crss_weight' column found. "
            "Raw CRSS counts are a sample and must be weighted for US totals. "
            "Use these numbers with caution."
        )

    # --- Quality flags ---
    df["incident_quality_flag"] = df["incident_source"].apply(_quality_flag)

    n_low = (df["incident_quality_flag"] == EXPOSURE_QUALITY_LOW).sum()
    if n_low:
        logger.warning(
            "%d incident rows flagged as low quality.", n_low
        )

    # --- Aggregate: sum counts per (make, model, model_year, incident_year) ---
    key_cols = ["make_norm", "model_norm", "model_year", "incident_year"]
    agg_dict = {col: "sum" for col in present_counts if col in df.columns}
    agg_dict["incident_source"] = "first"
    agg_dict["incident_quality_flag"] = "first"
    agg_dict["DEMO_DATA"] = "first"
    if "normalisation_confidence" in df.columns:
        agg_dict["normalisation_confidence"] = "min"

    df_agg = (
        df.groupby(key_cols, dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    dupes_before = len(df) - len(df_agg)
    if dupes_before > 0:
        logger.info(
            "Aggregated %d duplicate incident rows into %d unique keys.",
            dupes_before,
            len(df_agg),
        )

    logger.info(
        "Incident dataset cleaned: %d rows, %d unique vehicle cohorts.",
        len(df_agg),
        df_agg[["make_norm", "model_norm", "model_year"]].drop_duplicates().shape[0],
    )

    if save:
        write_csv(df_agg, INCIDENT_CLEAN_FILE, description="incident_clean")

    return df_agg
