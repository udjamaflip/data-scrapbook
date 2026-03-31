"""
config.py – Central configuration for the turn-signal crash-rate pipeline.

All paths, constants, and environment-variable overrides live here.
Import this module first in any pipeline stage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW: Path = Path(os.getenv("INPUT_DIR", str(PROJECT_ROOT / "data" / "raw")))
DATA_TEMPLATES: Path = DATA_RAW / "templates"
DATA_INTERMEDIATE: Path = PROJECT_ROOT / "data" / "intermediate"
DATA_FINAL: Path = PROJECT_ROOT / "data" / "final"
OUTPUTS_CHARTS: Path = PROJECT_ROOT / "outputs" / "charts"
OUTPUTS_TABLES: Path = PROJECT_ROOT / "outputs" / "tables"

# ---------------------------------------------------------------------------
# Raw input file names
# ---------------------------------------------------------------------------

BLINKER_FILE: Path = DATA_RAW / "blinker_colors.csv"
BLINKER_DEMO_FILE: Path = DATA_RAW / "blinker_colors_demo.csv"

EXPOSURE_FILE: Path = DATA_RAW / "exposure_data.csv"
EXPOSURE_DEMO_FILE: Path = DATA_RAW / "exposure_data_demo.csv"

INCIDENT_FILE: Path = DATA_RAW / "incident_data.csv"
INCIDENT_DEMO_FILE: Path = DATA_RAW / "incident_data_demo.csv"

MAKE_ALIASES_FILE: Path = DATA_TEMPLATES / "make_aliases.yaml"
MODEL_ALIASES_FILE: Path = DATA_TEMPLATES / "model_aliases.yaml"
MANUAL_OVERRIDES_FILE: Path = DATA_TEMPLATES / "manual_vehicle_overrides.csv"

# ---------------------------------------------------------------------------
# Intermediate outputs
# ---------------------------------------------------------------------------

BLINKER_CLEAN_FILE: Path = DATA_INTERMEDIATE / "blinker_clean.csv"
EXPOSURE_CLEAN_FILE: Path = DATA_INTERMEDIATE / "exposure_clean.csv"
INCIDENT_CLEAN_FILE: Path = DATA_INTERMEDIATE / "incident_clean.csv"

# ---------------------------------------------------------------------------
# Final outputs
# ---------------------------------------------------------------------------

BLINKER_MASTER_FILE: Path = DATA_FINAL / "blinker_master.csv"
EXPOSURE_FINAL_FILE: Path = DATA_FINAL / "exposure_clean.csv"
INCIDENT_FINAL_FILE: Path = DATA_FINAL / "incidents_clean.csv"
ANALYSIS_DATASET_FILE: Path = DATA_FINAL / "analysis_dataset.csv"

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------

# Minimum confidence score (0–1) required for a blinker row to be included
# in the core (headline) analysis.
MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.7"))

# Minimum number of make/model/year observations per signal-colour group
# before statistical tests are considered meaningful.
MIN_SAMPLE_SIZE: int = int(os.getenv("MIN_SAMPLE_SIZE", "30"))

# Standardised colour labels used throughout the pipeline.
COLOR_AMBER: str = "amber"
COLOR_NON_AMBER: str = "non_amber"
COLOR_MIXED: str = "mixed"
COLOR_UNKNOWN: str = "unknown"
VALID_COLORS: frozenset[str] = frozenset(
    {COLOR_AMBER, COLOR_NON_AMBER, COLOR_MIXED, COLOR_UNKNOWN}
)

# Colours eligible for the core headline comparison.
CORE_ANALYSIS_COLORS: frozenset[str] = frozenset({COLOR_AMBER, COLOR_NON_AMBER})

# Match-quality labels used in matching.py.
MATCH_EXACT: str = "exact"
MATCH_NORMALIZED: str = "normalized_exact"
MATCH_MANUAL: str = "manual_override"
MATCH_UNMATCHED: str = "unmatched"
MATCH_AMBIGUOUS: str = "ambiguous"

# Exposure quality labels.
EXPOSURE_QUALITY_HIGH: str = "high"
EXPOSURE_QUALITY_MEDIUM: str = "medium"
EXPOSURE_QUALITY_LOW: str = "low"

# Year range for the analysis.  Adjust as data allows.
ANALYSIS_YEAR_MIN: int = 1990
ANALYSIS_YEAR_MAX: int = 2024

# Rate denominator – crashes per N registered vehicle-years.
RATE_DENOMINATOR: int = 100_000

# Bootstrap CI parameters.
BOOTSTRAP_N_RESAMPLES: int = 2_000
BOOTSTRAP_CI_LEVEL: float = 0.95

# Rolling-average window in years.
ROLLING_WINDOW: int = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("pipeline")


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for path in (
        DATA_INTERMEDIATE,
        DATA_FINAL,
        OUTPUTS_CHARTS,
        OUTPUTS_TABLES,
    ):
        path.mkdir(parents=True, exist_ok=True)
