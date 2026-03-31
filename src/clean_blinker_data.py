"""
clean_blinker_data.py – Standardise and validate the blinker colour dataset.

Transforms raw rear_signal_color_raw into a canonical rear_signal_color_standardized
value: amber | non_amber | mixed | unknown.

Also computes:
  - confidence_score       – 0–1 quality of the colour assignment
  - ambiguous_flag         – True if the source is ambiguous
  - mixed_flag             – True if the model-year had both amber and non-amber
  - eligibility_for_core_analysis – True only for clean amber/non_amber rows
  - data_quality_notes     – any warnings about this row
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    COLOR_AMBER,
    COLOR_MIXED,
    COLOR_NON_AMBER,
    COLOR_UNKNOWN,
    MIN_CONFIDENCE,
    VALID_COLORS,
)
from src.utils import write_csv
from src.config import BLINKER_CLEAN_FILE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw → standardised colour mapping
# ---------------------------------------------------------------------------

# Keys are cleaned tokens (lower-case, stripped).
_COLOR_MAP: dict[str, str] = {
    # Amber
    "amber": COLOR_AMBER,
    "yellow": COLOR_AMBER,
    "orange": COLOR_AMBER,
    "amber/yellow": COLOR_AMBER,
    "yellow/amber": COLOR_AMBER,
    # Non-amber (red, clear, white, etc.)
    "red": COLOR_NON_AMBER,
    "clear": COLOR_NON_AMBER,
    "white": COLOR_NON_AMBER,
    "red/clear": COLOR_NON_AMBER,
    "clear/red": COLOR_NON_AMBER,
    "non-amber": COLOR_NON_AMBER,
    "non_amber": COLOR_NON_AMBER,
    "non amber": COLOR_NON_AMBER,   # produced when underscore is replaced with space
    "nonamber": COLOR_NON_AMBER,
    # Mixed
    "mixed": COLOR_MIXED,
    "varies": COLOR_MIXED,
    "trim dependent": COLOR_MIXED,
    "trim-dependent": COLOR_MIXED,
    "both": COLOR_MIXED,
    "amber or red": COLOR_MIXED,
    "red or amber": COLOR_MIXED,
    # Unknown
    "unknown": COLOR_UNKNOWN,
    "n/a": COLOR_UNKNOWN,
    "na": COLOR_UNKNOWN,
    "": COLOR_UNKNOWN,
    "not specified": COLOR_UNKNOWN,
    "unconfirmed": COLOR_UNKNOWN,
}


def _standardise_color(raw: object) -> str:
    """Map a raw colour string to one of the four canonical values."""
    if pd.isna(raw):
        return COLOR_UNKNOWN
    token = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    return _COLOR_MAP.get(token, COLOR_UNKNOWN)


# ---------------------------------------------------------------------------
# Confidence scoring from source quality
# ---------------------------------------------------------------------------

# Higher score = more trustworthy source.
_SOURCE_CONFIDENCE: dict[str, float] = {
    "manufacturer_spec": 1.0,
    "nhtsa_recall": 0.95,
    "nhtsa_tsb": 0.90,
    "owner_manual": 0.90,
    "dealer_documentation": 0.85,
    "independent_review": 0.80,
    "enthusiast_forum": 0.65,
    "crowdsourced": 0.60,
    "inferred": 0.50,
    "unknown": 0.30,
    "": 0.30,
}


def _source_confidence(source: object) -> float:
    """Return a confidence score based on the source quality label."""
    if pd.isna(source):
        return _SOURCE_CONFIDENCE["unknown"]
    key = str(source).strip().lower().replace(" ", "_")
    return _SOURCE_CONFIDENCE.get(key, 0.50)


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------


def clean_blinker_data(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Standardise blinker colour data and compute quality flags.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_blinker_data().
    save : bool
        If True, write the cleaned dataset to data/intermediate/.

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched blinker dataset.
    """
    df = df.copy()

    # --- Standardise colour ---
    df["rear_signal_color_standardized"] = df["rear_signal_color_raw"].apply(
        _standardise_color
    )

    n_unknown = (df["rear_signal_color_standardized"] == COLOR_UNKNOWN).sum()
    if n_unknown:
        logger.warning(
            "%d rows have unknown/unmapped rear_signal_color_raw values. "
            "These will be excluded from the core analysis.",
            n_unknown,
        )

    # --- Confidence score ---
    # Use existing confidence_score if provided; otherwise derive from source quality.
    if "confidence_score" not in df.columns or df["confidence_score"].isna().all():
        df["confidence_score"] = df["source"].apply(_source_confidence)
    else:
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
        # Where it's NaN, fill from source quality
        mask = df["confidence_score"].isna()
        df.loc[mask, "confidence_score"] = df.loc[mask, "source"].apply(_source_confidence).astype(float)

    # Cap between 0 and 1
    df["confidence_score"] = df["confidence_score"].clip(0.0, 1.0)

    # --- Flags ---
    df["ambiguous_flag"] = df["rear_signal_color_standardized"] == COLOR_UNKNOWN
    df["mixed_flag"] = df["rear_signal_color_standardized"] == COLOR_MIXED

    # Detect within-model-year conflicts (same make/model/year with different colours)
    key_cols = ["make_norm", "model_norm", "model_year"]
    color_counts = (
        df.groupby(key_cols)["rear_signal_color_standardized"]
        .nunique()
        .rename("_n_colors")
        .reset_index()
    )
    df = df.merge(color_counts, on=key_cols, how="left")
    conflict_mask = df["_n_colors"] > 1
    if conflict_mask.any():
        logger.warning(
            "%d rows have conflicting colour assignments for the same "
            "make/model/year – marking as mixed.",
            conflict_mask.sum(),
        )
        df.loc[conflict_mask, "rear_signal_color_standardized"] = COLOR_MIXED
        df.loc[conflict_mask, "mixed_flag"] = True
        df.loc[conflict_mask, "confidence_score"] = df.loc[
            conflict_mask, "confidence_score"
        ].clip(upper=0.6)
    df.drop(columns=["_n_colors"], inplace=True)

    # --- Eligibility for core analysis ---
    # True only when colour is amber or non_amber AND confidence meets threshold.
    df["eligibility_for_core_analysis"] = (
        df["rear_signal_color_standardized"].isin(
            {COLOR_AMBER, COLOR_NON_AMBER}
        )
        & (df["confidence_score"] >= MIN_CONFIDENCE)
    )

    # --- Data quality notes ---
    notes: list[str] = []
    for _, row in df.iterrows():
        row_notes = []
        if row["rear_signal_color_standardized"] == COLOR_UNKNOWN:
            row_notes.append("colour_unmapped")
        if row["confidence_score"] < MIN_CONFIDENCE:
            row_notes.append(f"low_confidence({row['confidence_score']:.2f})")
        if row["mixed_flag"]:
            row_notes.append("mixed_within_model_year")
        notes.append("; ".join(row_notes) if row_notes else "ok")
    df["data_quality_notes"] = notes

    # --- Market default ---
    if "market" not in df.columns:
        df["market"] = "US"

    # --- Validate standardised colour values ---
    invalid = ~df["rear_signal_color_standardized"].isin(VALID_COLORS)
    if invalid.any():
        raise ValueError(
            f"BUG: {invalid.sum()} rows have invalid standardised colour values. "
            f"Check _COLOR_MAP in clean_blinker_data.py."
        )

    # --- Log summary ---
    counts = df["rear_signal_color_standardized"].value_counts()
    logger.info("Blinker colour distribution after cleaning: %s", counts.to_dict())
    logger.info(
        "Rows eligible for core analysis: %d / %d",
        df["eligibility_for_core_analysis"].sum(),
        len(df),
    )

    if save:
        write_csv(df, BLINKER_CLEAN_FILE, description="blinker_clean")

    return df
