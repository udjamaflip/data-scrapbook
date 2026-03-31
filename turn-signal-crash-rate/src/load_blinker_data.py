"""
load_blinker_data.py – Ingest the rear turn-signal colour dataset.

Expected input: data/raw/blinker_colors.csv  (or blinker_colors_demo.csv in demo mode)

The blinker dataset is the hardest to obtain: no comprehensive public source
maps every US make/model/year to rear turn-signal colour.  The user must curate
this file manually (or use the demo data to validate the pipeline).

See data/raw/templates/blinker_colors_template.csv for the required schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    BLINKER_DEMO_FILE,
    BLINKER_FILE,
)
from src.utils import read_csv_validated
from src.vehicle_normalization import (
    load_make_aliases,
    load_model_aliases,
    load_manual_overrides,
    normalise_dataframe,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required / optional columns
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "make",
    "model",
    "model_year",
    "rear_signal_color_raw",
    "source",
]

OPTIONAL_COLS = [
    "manufacturer",
    "source_url",
    "confidence_score",
    "trim_scope",
    "notes",
    "market",
    "ambiguous_flag",
    "mixed_flag",
]


def load_blinker_data(demo: bool = False) -> pd.DataFrame:
    """
    Load the blinker colour dataset from disk.

    Parameters
    ----------
    demo : bool
        If True, load the clearly-labelled demo dataset instead of the real one.

    Returns
    -------
    pd.DataFrame
        Raw blinker data with vehicle normalisation applied.
        Includes a DEMO_DATA column (True/False) for traceability.
    """
    path: Path = BLINKER_DEMO_FILE if demo else BLINKER_FILE

    df = read_csv_validated(
        path,
        required_cols=REQUIRED_COLS,
        optional_cols=OPTIONAL_COLS,
    )

    # Tag demo rows so they are never silently mixed with real data
    df["DEMO_DATA"] = demo
    if demo:
        logger.warning(
            "DEMO MODE: blinker data loaded from '%s'. "
            "Results are NOT suitable for real-world conclusions.",
            path.name,
        )

    # Restrict to US-market rows if market column present
    if "market" in df.columns:
        before = len(df)
        df = df[df["market"].str.upper().isin(["US", "USA", "UNITED STATES", ""]) | df["market"].isna()]
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d non-US market rows from blinker dataset.", dropped)

    # Normalise vehicle identifiers
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
        "Loaded blinker dataset: %d rows, %d unique (make, model, year) combos.",
        len(df),
        df[["make_norm", "model_norm", "model_year"]].drop_duplicates().shape[0],
    )
    return df
