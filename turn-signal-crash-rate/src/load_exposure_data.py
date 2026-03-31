"""
load_exposure_data.py – Ingest vehicle registration / exposure data.

Expected input: data/raw/exposure_data.csv  (or exposure_data_demo.csv in demo mode)

Exposure data answers: "how many vehicle-years were each make/model/year
on the road?"  Without this, raw crash counts are meaningless — a popular
model will appear in more crashes simply because more of them exist.

Ideal sources (in preference order):
  1. IHS Markit / Experian AutoCount  – make/model/year breakdowns (proprietary)
  2. NHTSA Complaints Registration Data – partial make/model coverage
  3. FHWA Highway Statistics Table MV-1  – state totals by vehicle type only
     (not by make/model – must be apportioned or treated as unknown)

If no granular source is available, the user can supply their own CSV following
the template in data/raw/templates/exposure_data_template.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    EXPOSURE_DEMO_FILE,
    EXPOSURE_FILE,
)
from src.utils import read_csv_validated
from src.vehicle_normalization import (
    load_make_aliases,
    load_model_aliases,
    load_manual_overrides,
    normalise_dataframe,
)

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "make",
    "model",
    "model_year",
    "year_of_exposure",
    "exposure_source",
]

OPTIONAL_COLS = [
    "vehicles_on_road",
    "registered_vehicle_years",
    "exposure_quality_flag",
]


def load_exposure_data(demo: bool = False) -> pd.DataFrame:
    """
    Load and vehicle-normalise the exposure dataset.

    Parameters
    ----------
    demo : bool
        If True, load the demo dataset.

    Returns
    -------
    pd.DataFrame
        Exposure data with normalised vehicle keys.
    """
    path: Path = EXPOSURE_DEMO_FILE if demo else EXPOSURE_FILE

    df = read_csv_validated(
        path,
        required_cols=REQUIRED_COLS,
        optional_cols=OPTIONAL_COLS,
    )

    df["DEMO_DATA"] = demo
    if demo:
        logger.warning(
            "DEMO MODE: exposure data loaded from '%s'. "
            "Results are NOT suitable for real-world conclusions.",
            path.name,
        )

    # Coerce numeric columns
    for col in ["vehicles_on_road", "registered_vehicle_years"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year_of_exposure"] = pd.to_numeric(df["year_of_exposure"], errors="coerce")

    # Normalise vehicle keys
    make_aliases = load_make_aliases()
    model_aliases = load_model_aliases()
    overrides = load_manual_overrides()

    df = normalise_dataframe(
        df,
        make_col="make",
        model_col="model",
        year_col="model_year",
        make_aliases=make_aliases,
        model_aliases=model_aliases,
        overrides=overrides,
    )

    logger.info(
        "Loaded exposure dataset: %d rows, years %s–%s.",
        len(df),
        int(df["year_of_exposure"].min()) if not df["year_of_exposure"].isna().all() else "?",
        int(df["year_of_exposure"].max()) if not df["year_of_exposure"].isna().all() else "?",
    )
    return df
