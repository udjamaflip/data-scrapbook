"""
main.py - Full pipeline orchestrator and CLI entry point.

Usage
-----
Run the full pipeline with demo data:
    python -m src.main --demo

Run with real data:
    python -m src.main --input-dir data/raw --output-dir outputs

Options:
    --input-dir   Path to directory containing raw input CSVs (default: data/raw)
    --output-dir  Path to output root directory (default: outputs)
    --demo        Use built-in synthetic demo data instead of real input files
    --no-charts   Skip chart generation (useful for CI or data-only runs)
    --log-level   Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)

Pipeline stages
---------------
1.  load + clean blinker colour data
2.  load + clean exposure data
3.  load + clean incident data
4.  match all three datasets
5.  run analysis
6.  generate charts
7.  print console summary

Exit codes
----------
0 - success
1 - input file missing
2 - data validation error
3 - unexpected runtime error
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click

logger = logging.getLogger("pipeline.main")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--input-dir",
    "input_dir",
    type=click.Path(exists=False),
    default=None,
    help="Directory containing raw input CSVs (overrides INPUT_DIR env var).",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Root output directory (overrides OUTPUT_DIR env var).",
)
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Use synthetic demo data (no real input files required).",
)
@click.option(
    "--no-charts",
    "no_charts",
    is_flag=True,
    default=False,
    help="Skip chart generation.",
)
@click.option(
    "--log-level",
    "log_level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
def cli(
    input_dir: str | None,
    output_dir: str | None,
    demo: bool,
    no_charts: bool,
    log_level: str,
) -> None:
    """
    US rear turn-signal colour vs crash-rate analysis pipeline.

    Run 'python -m src.main --demo' to test with synthetic data.
    """
    _configure_logging(log_level)
    _apply_path_overrides(input_dir, output_dir)

    exit_code = run_pipeline(demo=demo, generate_charts=not no_charts)
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(demo: bool = False, generate_charts: bool = True) -> int:
    """
    Execute the full pipeline end-to-end.

    Returns 0 on success, non-zero on failure.
    """
    # Import here so path overrides from CLI take effect before config is read
    import src.config as cfg
    from src.clean_blinker_data import clean_blinker_data
    from src.clean_exposure_data import clean_exposure_data
    from src.clean_incident_data import clean_incident_data
    from src.load_blinker_data import load_blinker_data
    from src.load_exposure_data import load_exposure_data
    from src.load_incident_data import load_incident_data
    from src.matching import match_datasets
    from src.analysis import run_analysis
    from src.utils import print_summary_box

    cfg.ensure_output_dirs()

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1 - Blinker colour data
    # ------------------------------------------------------------------
    logger.info("Stage 1/6 - Loading blinker colour data …")
    try:
        blinker_raw = load_blinker_data(demo=demo)
        blinker = clean_blinker_data(blinker_raw, save=True)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except ValueError as exc:
        logger.error("Blinker data validation error: %s", exc)
        return 2

    # ------------------------------------------------------------------
    # Stage 2 - Exposure data
    # ------------------------------------------------------------------
    logger.info("Stage 2/6 - Loading exposure data …")
    try:
        exposure_raw = load_exposure_data(demo=demo)
        exposure = clean_exposure_data(exposure_raw, save=True)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except ValueError as exc:
        logger.error("Exposure data validation error: %s", exc)
        return 2

    # ------------------------------------------------------------------
    # Stage 3 - Incident data
    # ------------------------------------------------------------------
    logger.info("Stage 3/6 - Loading incident data …")
    try:
        incidents_raw = load_incident_data(demo=demo)
        incidents = clean_incident_data(incidents_raw, save=True)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except ValueError as exc:
        logger.error("Incident data validation error: %s", exc)
        return 2

    # ------------------------------------------------------------------
    # Stage 4 - Matching
    # ------------------------------------------------------------------
    logger.info("Stage 4/6 - Matching datasets …")
    try:
        analysis_df, unmatched = match_datasets(blinker, exposure, incidents, save=True)
    except Exception as exc:
        logger.error("Matching failed: %s", exc, exc_info=True)
        return 3

    # ------------------------------------------------------------------
    # Stage 5 - Analysis
    # ------------------------------------------------------------------
    logger.info("Stage 5/6 - Running analysis …")
    try:
        results, enriched_df, rate_col = run_analysis(analysis_df)
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        return 3

    # ------------------------------------------------------------------
    # Stage 6 - Charts
    # ------------------------------------------------------------------
    chart_paths: list[Path] = []
    if generate_charts:
        logger.info("Stage 6/6 - Generating charts …")
        try:
            from src.charts import generate_all_charts

            chart_paths = generate_all_charts(
                analysis_df=enriched_df,
                results=results,
                rate_col=rate_col,
                is_demo=demo,
            )
        except Exception as exc:
            logger.error("Chart generation failed: %s", exc, exc_info=True)
            # Non-fatal - continue to summary
    else:
        logger.info("Stage 6/6 - Chart generation skipped (--no-charts).")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    _print_final_summary(
        demo=demo,
        blinker=blinker,
        exposure=exposure,
        incidents=incidents,
        analysis_df=enriched_df,
        unmatched=unmatched,
        results=results,
        chart_paths=chart_paths,
        elapsed=elapsed,
    )

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    import src.config  # noqa: F401 - triggers basicConfig
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def _apply_path_overrides(input_dir: str | None, output_dir: str | None) -> None:
    """Patch config path constants if CLI overrides were provided."""
    import os
    if input_dir:
        os.environ["INPUT_DIR"] = input_dir
    if output_dir:
        os.environ["OUTPUT_DIR"] = output_dir


def _pick_rate_col(df: "pd.DataFrame") -> str:  # type: ignore[name-defined]
    import pandas as pd
    for col in [
        "fatal_crashes_per_100k_rvy",
        "deaths_per_100k_rvy",
        "incidents_per_100k_rvy",
        "claims_per_100k_rvy",
    ]:
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError("No usable rate column found in analysis dataset.")


def _print_final_summary(
    demo: bool,
    blinker: "pd.DataFrame",  # type: ignore[name-defined]
    exposure: "pd.DataFrame",  # type: ignore[name-defined]
    incidents: "pd.DataFrame",  # type: ignore[name-defined]
    analysis_df: "pd.DataFrame",  # type: ignore[name-defined]
    unmatched: "pd.DataFrame",  # type: ignore[name-defined]
    results: dict,
    chart_paths: list[Path],
    elapsed: float,
) -> None:
    import src.config as cfg
    from src.utils import print_summary_box

    # ── Data quality ──
    stat_df = results.get("statistical_comparison")
    interp = ""
    if stat_df is not None and not stat_df.empty:
        interp = str(stat_df.iloc[0].get("interpretation", ""))

    print_summary_box(
        "Pipeline complete",
        [
            f"Run mode       : {'DEMO (synthetic data)' if demo else 'REAL DATA'}",
            f"Elapsed        : {elapsed:.1f}s",
            f"Blinker rows   : {len(blinker):,}",
            f"Exposure rows  : {len(exposure):,}",
            f"Incident rows  : {len(incidents):,}",
            f"Analysis rows  : {len(analysis_df):,}",
            f"Unmatched rows : {len(unmatched):,}",
            f"Charts written : {len(chart_paths)}",
            f"Headline result: {interp or '(see outputs/tables/statistical_comparison.csv)'}",
        ],
    )

    print_summary_box(
        "Output files written",
        [
            str(cfg.BLINKER_MASTER_FILE),
            str(cfg.EXPOSURE_FINAL_FILE),
            str(cfg.INCIDENT_FINAL_FILE),
            str(cfg.ANALYSIS_DATASET_FILE),
            str(cfg.OUTPUTS_TABLES / "summary_by_signal_color.csv"),
            str(cfg.OUTPUTS_TABLES / "summary_by_year.csv"),
            str(cfg.OUTPUTS_TABLES / "summary_by_manufacturer.csv"),
            str(cfg.OUTPUTS_TABLES / "statistical_comparison.csv"),
            str(cfg.OUTPUTS_TABLES / "sensitivity_analysis.csv"),
            str(cfg.OUTPUTS_TABLES / "unmatched_rows.csv"),
            str(cfg.OUTPUTS_TABLES / "data_quality_report.csv"),
        ]
        + [str(p) for p in chart_paths],
    )

    if demo:
        print_summary_box(
            "!! DEMO MODE - what to do next",
            [
                "1. Blinker colour data (REQUIRED, must be manually curated):",
                f"   -> Place at: {cfg.BLINKER_FILE}",
                f"   -> Template: {cfg.DATA_TEMPLATES / 'blinker_colors_template.csv'}",
                "",
                "2. Exposure data (REQUIRED for accurate rates):",
                f"   -> Place at: {cfg.EXPOSURE_FILE}",
                f"   -> Template: {cfg.DATA_TEMPLATES / 'exposure_data_template.csv'}",
                "   -> Sources: IHS Markit, Experian AutoCount, NHTSA, FHWA",
                "",
                "3. Incident data (REQUIRED):",
                f"   -> Place at: {cfg.INCIDENT_FILE}",
                f"   -> Template: {cfg.DATA_TEMPLATES / 'incident_data_template.csv'}",
                "   -> Sources: NHTSA FARS (nhtsa.gov/research-data/fatality-analysis-reporting-system-fars)",
                "",
                "4. Re-run with real data:",
                "   python -m src.main",
                "",
                "5. To update vehicle name aliases:",
                f"   -> Edit: {cfg.MAKE_ALIASES_FILE}",
                f"   -> Edit: {cfg.MODEL_ALIASES_FILE}",
                "",
                "6. To add manual vehicle overrides:",
                f"   -> Edit: {cfg.MANUAL_OVERRIDES_FILE}",
            ],
        )
    else:
        print_summary_box(
            "Next steps",
            [
                "• Review outputs/tables/data_quality_report.csv for warnings",
                "• Review outputs/tables/unmatched_rows.csv - add overrides if needed",
                "• Review outputs/tables/statistical_comparison.csv for headline stats",
                "• Open outputs/charts/ for visual summaries",
                "• Remember: this is observational data; interpret as correlation only",
            ],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
