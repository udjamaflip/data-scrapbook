"""
vehicle_normalization.py – Robust make/model/year normalisation.

Provides:
  - normalize_make()         – canonical make name
  - normalize_model()        – canonical model name
  - normalize_year()         – validated integer year
  - normalize_vehicle_key()  – all three combined, with confidence scoring
  - load_make_aliases()      – from YAML
  - load_model_aliases()     – from YAML
  - load_manual_overrides()  – from CSV
  - apply_manual_overrides() – patch a DataFrame using the override table

Design rules
  • Always preserve raw values alongside normalised ones.
  • Never silently collapse rows when confidence is low.
  • Manual overrides take highest priority; normalised aliases are second.
  • Confidence scoring: exact=1.0, alias=0.9, manual=0.8, low=0.5, unknown=0.0
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import (
    ANALYSIS_YEAR_MAX,
    ANALYSIS_YEAR_MIN,
    MAKE_ALIASES_FILE,
    MANUAL_OVERRIDES_FILE,
    MODEL_ALIASES_FILE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence score constants
# ---------------------------------------------------------------------------

CONF_EXACT: float = 1.0
CONF_ALIAS: float = 0.9
CONF_MANUAL: float = 0.8
CONF_LOW: float = 0.5
CONF_UNKNOWN: float = 0.0


# ---------------------------------------------------------------------------
# Internal text helpers
# ---------------------------------------------------------------------------


def _ascii_fold(text: str) -> str:
    """Decompose Unicode characters to ASCII equivalents (e.g. accented chars)."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )


def _clean_token(text: Any) -> str:
    """
    Return a canonical comparison token:
      - strip whitespace
      - ASCII fold
      - lower-case
      - collapse internal whitespace
      - remove punctuation except hyphens between alphanumeric chars
    """
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = _ascii_fold(s)
    s = s.lower()
    # Remove punctuation that is NOT a hyphen between alphanumeric chars
    s = re.sub(r"[^\w\s-]", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Alias loading
# ---------------------------------------------------------------------------


def _load_yaml_alias_map(path: Path) -> dict[str, str]:
    """
    Load a YAML file that maps raw/variant names → canonical names.

    Expected format::

        Volkswagen:
          - VW
          - Volkwagen    # common misspelling
        General Motors:
          - GM
          - GMC

    Returns a flat dict: token → canonical_name.
    """
    if not path.exists():
        logger.warning("Alias file not found: %s – no aliases will be applied.", path)
        return {}

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, list[str]] = yaml.safe_load(fh) or {}

    alias_map: dict[str, str] = {}
    for canonical, variants in raw.items():
        canonical_token = _clean_token(canonical)
        # The canonical itself maps to itself (so we can always resolve)
        alias_map[canonical_token] = canonical
        for variant in variants or []:
            alias_map[_clean_token(variant)] = canonical

    logger.debug("Loaded %d alias entries from %s", len(alias_map), path.name)
    return alias_map


def load_make_aliases() -> dict[str, str]:
    return _load_yaml_alias_map(MAKE_ALIASES_FILE)


def load_model_aliases() -> dict[str, str]:
    return _load_yaml_alias_map(MODEL_ALIASES_FILE)


# ---------------------------------------------------------------------------
# Manual overrides
# ---------------------------------------------------------------------------


def load_manual_overrides() -> pd.DataFrame:
    """
    Load the manual vehicle override table.

    Expected columns:
      raw_make, raw_model, raw_model_year,
      override_make, override_model, override_model_year,
      override_notes

    Returns an empty DataFrame if the file is missing.
    """
    if not MANUAL_OVERRIDES_FILE.exists():
        logger.warning(
            "Manual override file not found: %s – no overrides applied.",
            MANUAL_OVERRIDES_FILE,
        )
        return pd.DataFrame(
            columns=[
                "raw_make",
                "raw_model",
                "raw_model_year",
                "override_make",
                "override_model",
                "override_model_year",
                "override_notes",
            ]
        )

    df = pd.read_csv(MANUAL_OVERRIDES_FILE, dtype=str)
    logger.info("Loaded %d manual vehicle overrides.", len(df))
    return df


def apply_manual_overrides(
    df: pd.DataFrame,
    overrides: pd.DataFrame,
    make_col: str = "make_norm",
    model_col: str = "model_norm",
    year_col: str = "model_year",
) -> pd.DataFrame:
    """
    Apply manual overrides to *df* in-place (returns a copy).

    Rows whose (make_norm, model_norm, model_year) triple matches a row in
    *overrides* will have their normalised fields patched, and their
    normalisation_source set to 'manual_override'.
    """
    if overrides.empty:
        return df

    df = df.copy()
    override_applied = 0

    for _, row in overrides.iterrows():
        raw_make_t = _clean_token(row.get("raw_make", ""))
        raw_model_t = _clean_token(row.get("raw_model", ""))
        raw_year = row.get("raw_model_year", "")
        try:
            raw_year_int: int | None = int(float(raw_year))
        except (TypeError, ValueError):
            raw_year_int = None

        mask = (
            df[make_col].apply(_clean_token) == raw_make_t
        ) & (
            df[model_col].apply(_clean_token) == raw_model_t
        )
        if raw_year_int is not None:
            mask &= df[year_col] == raw_year_int

        if not mask.any():
            continue

        if pd.notna(row.get("override_make")):
            df.loc[mask, make_col] = str(row["override_make"]).strip()
        if pd.notna(row.get("override_model")):
            df.loc[mask, model_col] = str(row["override_model"]).strip()
        if pd.notna(row.get("override_model_year")):
            try:
                df.loc[mask, year_col] = int(float(row["override_model_year"]))
            except (TypeError, ValueError):
                pass

        df.loc[mask, "normalisation_source"] = "manual_override"
        df.loc[mask, "normalisation_confidence"] = CONF_MANUAL
        override_applied += mask.sum()

    logger.info("Applied %d manual override patches.", override_applied)
    return df


# ---------------------------------------------------------------------------
# Core normalisation functions
# ---------------------------------------------------------------------------


def normalize_make(
    raw: Any,
    alias_map: dict[str, str] | None = None,
) -> tuple[str, float, str]:
    """
    Normalise a raw make string.

    Returns
    -------
    (normalised_make, confidence, source)
      source is one of: 'exact', 'alias', 'low'
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return "", CONF_UNKNOWN, "unknown"

    token = _clean_token(raw)

    if alias_map and token in alias_map:
        canonical = alias_map[token]
        source = "exact" if _clean_token(canonical) == token else "alias"
        conf = CONF_EXACT if source == "exact" else CONF_ALIAS
        return canonical, conf, source

    # Title-case the cleaned token as a best-effort fallback
    fallback = token.title()
    return fallback, CONF_LOW, "low"


def normalize_model(
    raw: Any,
    alias_map: dict[str, str] | None = None,
) -> tuple[str, float, str]:
    """
    Normalise a raw model string.

    Returns (normalised_model, confidence, source).
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return "", CONF_UNKNOWN, "unknown"

    token = _clean_token(raw)

    if alias_map and token in alias_map:
        canonical = alias_map[token]
        source = "exact" if _clean_token(canonical) == token else "alias"
        conf = CONF_EXACT if source == "exact" else CONF_ALIAS
        return canonical, conf, source

    # Title-case as best-effort
    fallback = re.sub(r"\s+", " ", str(raw).strip()).title()
    return fallback, CONF_LOW, "low"


def normalize_year(raw: Any) -> tuple[int | None, float, str]:
    """
    Coerce *raw* to an integer model year within the configured range.

    Returns (year_int_or_None, confidence, source).
    """
    try:
        year = int(float(raw))
    except (TypeError, ValueError):
        return None, CONF_UNKNOWN, "unknown"

    if ANALYSIS_YEAR_MIN <= year <= ANALYSIS_YEAR_MAX:
        return year, CONF_EXACT, "exact"

    # Out of configured range – keep but flag low confidence
    logger.debug("Year %d is outside configured range [%d, %d].", year, ANALYSIS_YEAR_MIN, ANALYSIS_YEAR_MAX)
    return year, CONF_LOW, "low"


# ---------------------------------------------------------------------------
# Combined row-level normalisation
# ---------------------------------------------------------------------------


def normalize_vehicle_row(
    make_raw: Any,
    model_raw: Any,
    year_raw: Any,
    make_aliases: dict[str, str] | None = None,
    model_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Normalise a single vehicle row and return a dict of enriched fields.

    Returned keys::

        make_norm, model_norm, model_year,
        normalisation_confidence, normalisation_source
    """
    make_norm, make_conf, make_src = normalize_make(make_raw, make_aliases)
    model_norm, model_conf, model_src = normalize_model(model_raw, model_aliases)
    year_norm, year_conf, year_src = normalize_year(year_raw)

    # Overall confidence is the minimum of the three individual scores.
    overall_conf = min(make_conf, model_conf, year_conf)

    sources = {make_src, model_src, year_src} - {"exact"}
    if not sources:
        overall_src = "exact"
    elif "alias" in sources:
        overall_src = "alias"
    elif "low" in sources:
        overall_src = "low"
    else:
        overall_src = "unknown"

    return {
        "make_norm": make_norm,
        "model_norm": model_norm,
        "model_year": year_norm,
        "normalisation_confidence": overall_conf,
        "normalisation_source": overall_src,
    }


def normalise_dataframe(
    df: pd.DataFrame,
    make_col: str = "make",
    model_col: str = "model",
    year_col: str = "model_year",
    make_aliases: dict[str, str] | None = None,
    model_aliases: dict[str, str] | None = None,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Vectorised normalisation of an entire DataFrame.

    Adds columns:
      make_norm, model_norm, model_year (overwritten),
      normalisation_confidence, normalisation_source

    Original raw columns are preserved as <col>_raw if not already present.
    """
    df = df.copy()

    # Preserve raw values
    for col, raw_col in [(make_col, "make_raw"), (model_col, "model_raw"), (year_col, "model_year_raw")]:
        if col in df.columns and raw_col not in df.columns:
            df[raw_col] = df[col]

    results = df.apply(
        lambda row: normalize_vehicle_row(
            row.get(make_col),
            row.get(model_col),
            row.get(year_col),
            make_aliases=make_aliases,
            model_aliases=model_aliases,
        ),
        axis=1,
        result_type="expand",
    )

    df["make_norm"] = results["make_norm"]
    df["model_norm"] = results["model_norm"]
    df["model_year"] = results["model_year"]
    df["normalisation_confidence"] = results["normalisation_confidence"]
    df["normalisation_source"] = results["normalisation_source"]

    # Apply manual overrides (highest priority)
    if overrides is not None and not overrides.empty:
        df = apply_manual_overrides(df, overrides)

    n_low = (df["normalisation_confidence"] < 0.7).sum()
    if n_low > 0:
        logger.warning(
            "%d rows have low normalisation confidence (<0.7). "
            "Review 'normalisation_source' column.",
            n_low,
        )

    return df
