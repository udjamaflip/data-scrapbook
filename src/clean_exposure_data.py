"""
clean_exposure_data.py – Clean and validate the exposure dataset.

Computes:
  - registered_vehicle_years  – cumulative exposure for a make/model/year
  - exposure_quality_flag     – high / medium / low based on source
  - exposure data quality report notes

Critical note
-------------
Crash rates are computed as:

    crashes / registered_vehicle_years × 100 000

A registered vehicle-year (RVY) represents one vehicle registered for one
calendar year.  Summing across calendar years gives total fleet exposure.
For example, if a 2005 Toyota Camry had 500 000 units registered in 2005,
400 000 in 2006, and 320 000 in 2007, its total RVY through 2007 is 1 220 000.

Without this normalisation, any model with a large fleet will appear riskier
simply because more vehicles exist to be involved in crashes.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    ANALYSIS_YEAR_MAX,
    ANALYSIS_YEAR_MIN,
    EXPOSURE_CLEAN_FILE,
    EXPOSURE_QUALITY_HIGH,
    EXPOSURE_QUALITY_LOW,
    EXPOSURE_QUALITY_MEDIUM,
)
from src.utils import write_csv

logger = logging.getLogger(__name__)

_SOURCE_QUALITY: dict[str, str] = {
    "ihs_markit": EXPOSURE_QUALITY_HIGH,
    "experian_autocount": EXPOSURE_QUALITY_HIGH,
    "r_l_polk": EXPOSURE_QUALITY_HIGH,
    "nhtsa_registration": EXPOSURE_QUALITY_MEDIUM,
    "fhwa_mv1": EXPOSURE_QUALITY_LOW,
    "fhwa": EXPOSURE_QUALITY_LOW,
    "estimated": EXPOSURE_QUALITY_LOW,
    "demo": EXPOSURE_QUALITY_LOW,
    "synthetic": EXPOSURE_QUALITY_LOW,
}


def _quality_flag(source: object) -> str:
    if pd.isna(source):
        return EXPOSURE_QUALITY_LOW
    key = str(source).strip().lower().replace(" ", "_").replace("-", "_")
    return _SOURCE_QUALITY.get(key, EXPOSURE_QUALITY_MEDIUM)


def clean_exposure_data(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Clean exposure data and compute registered_vehicle_years.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_exposure_data().
    save : bool
        Write cleaned output to data/intermediate/.

    Returns
    -------
    pd.DataFrame
        Cleaned exposure data with RVY and quality flags.
    """
    df = df.copy()

    # --- Year range filter ---
    before = len(df)
    df = df[
        df["year_of_exposure"].between(ANALYSIS_YEAR_MIN, ANALYSIS_YEAR_MAX, inclusive="both")
        | df["year_of_exposure"].isna()
    ]
    if len(df) < before:
        logger.info(
            "Dropped %d exposure rows outside year range [%d, %d].",
            before - len(df),
            ANALYSIS_YEAR_MIN,
            ANALYSIS_YEAR_MAX,
        )

    # --- Drop rows with no usable exposure figure ---
    has_vehicles_on_road = "vehicles_on_road" in df.columns
    has_rvy = "registered_vehicle_years" in df.columns

    if has_vehicles_on_road:
        df["vehicles_on_road"] = pd.to_numeric(df["vehicles_on_road"], errors="coerce")

    if has_rvy:
        df["registered_vehicle_years"] = pd.to_numeric(
            df["registered_vehicle_years"], errors="coerce"
        )

    # If RVY column is absent or all-NaN, derive it from vehicles_on_road.
    # One calendar year of registrations ≈ one registered vehicle-year per vehicle.
    if not has_rvy or df["registered_vehicle_years"].isna().all():
        if has_vehicles_on_road:
            logger.info(
                "registered_vehicle_years absent; deriving from vehicles_on_road "
                "(1 RVY = 1 vehicle × 1 year)."
            )
            df["registered_vehicle_years"] = df["vehicles_on_road"]
        else:
            logger.error(
                "Neither 'vehicles_on_road' nor 'registered_vehicle_years' found in "
                "exposure data.  Cannot compute rates."
            )
            df["registered_vehicle_years"] = float("nan")

    # Drop rows where RVY is still null or zero (unusable for rate computation)
    bad_rvy = df["registered_vehicle_years"].isna() | (df["registered_vehicle_years"] <= 0)
    if bad_rvy.any():
        logger.warning(
            "%d exposure rows have null/zero registered_vehicle_years and will be "
            "excluded from rate calculations.",
            bad_rvy.sum(),
        )

    # --- Aggregate: sum RVY across calendar years for each make/model/year ---
    key_cols = ["make_norm", "model_norm", "model_year"]

    # We keep the per-year rows for time-series analysis, but also compute a
    # cumulative RVY column per vehicle cohort.
    df = df.sort_values(key_cols + ["year_of_exposure"])
    df["cumulative_rvy"] = df.groupby(key_cols)["registered_vehicle_years"].cumsum()

    # --- Exposure quality flag ---
    df["exposure_quality_flag"] = df["exposure_source"].apply(_quality_flag)

    n_low = (df["exposure_quality_flag"] == EXPOSURE_QUALITY_LOW).sum()
    if n_low:
        logger.warning(
            "%d exposure rows flagged as low quality. "
            "Rates derived from these rows should be treated with caution.",
            n_low,
        )

    # --- Validate ---
    key_dupes = df.duplicated(subset=key_cols + ["year_of_exposure"], keep=False)
    if key_dupes.any():
        logger.warning(
            "%d duplicate (make, model, model_year, year_of_exposure) rows detected. "
            "Check your exposure source for double-counting.",
            key_dupes.sum(),
        )

    logger.info(
        "Exposure dataset cleaned: %d rows, %d unique vehicle cohorts.",
        len(df),
        df[key_cols].drop_duplicates().shape[0],
    )

    if save:
        write_csv(df, EXPOSURE_CLEAN_FILE, description="exposure_clean")

    return df
