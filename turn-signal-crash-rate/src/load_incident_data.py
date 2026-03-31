"""
load_incident_data.py – Ingest crash / incident data.

Expected input: data/raw/incident_data.csv  (or incident_data_demo.csv in demo mode)

Primary public source:
  NHTSA FARS (Fatality Analysis Reporting System)
  https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
  Annual data tables; vehicle file contains make, model, model_year.

Secondary public source:
  NHTSA CRSS (Crash Report Sampling System)
  https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system
  Sampled (not census) – requires weighting for population estimates.

Limitations documented here (and surfaced to the user):
  - FARS covers ONLY fatal crashes (at least one fatality).
  - CRSS is a probability sample; counts must be weighted to estimate US totals.
  - Make/model coding in FARS is inconsistent across years (pre-2010 especially).
  - Not every file will have all columns – the schema is nullable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import INCIDENT_DEMO_FILE, INCIDENT_FILE
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
    "incident_year",
    "incident_source",
]

OPTIONAL_COLS = [
    "fatal_crash_count",
    "occupant_death_count",
    "total_incident_count",
    "claim_count",
    "incident_quality_flag",
    "crss_weight",  # CRSS probability weight – required for non-fatal estimates
]


def load_incident_data(demo: bool = False) -> pd.DataFrame:
    """
    Load and vehicle-normalise the incident dataset.

    Parameters
    ----------
    demo : bool
        If True, load the demo dataset.

    Returns
    -------
    pd.DataFrame
        Incident data with normalised vehicle keys.
    """
    path: Path = INCIDENT_DEMO_FILE if demo else INCIDENT_FILE

    df = read_csv_validated(
        path,
        required_cols=REQUIRED_COLS,
        optional_cols=OPTIONAL_COLS,
    )

    df["DEMO_DATA"] = demo
    if demo:
        logger.warning(
            "DEMO MODE: incident data loaded from '%s'. "
            "Results are NOT suitable for real-world conclusions.",
            path.name,
        )

    # Coerce count columns to numeric; errors → NaN (not 0)
    for col in ["fatal_crash_count", "occupant_death_count", "total_incident_count", "claim_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["incident_year"] = pd.to_numeric(df["incident_year"], errors="coerce")

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
        "Loaded incident dataset: %d rows, years %s–%s.",
        len(df),
        int(df["incident_year"].min()) if not df["incident_year"].isna().all() else "?",
        int(df["incident_year"].max()) if not df["incident_year"].isna().all() else "?",
    )
    return df
