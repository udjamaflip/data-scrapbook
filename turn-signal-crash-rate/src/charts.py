"""
charts.py – Generate all charts for the pipeline.

Eight chart types + three alternative crash-rate visualisations,
all using matplotlib (no seaborn). One figure per chart.

Visual design:
  - Warm cream background  (#F8F5EE)
  - Amber signal colour    (#C8891C) – warm golden, not aggressive
  - Non-amber colour       (#A04848) – dusty brick, not aggressive
  - Off-white grid with    (#E4D8C8) warm tint
  - Charcoal typography    (#2C2C2C)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from src.config import (
    COLOR_AMBER,
    COLOR_MIXED,
    COLOR_NON_AMBER,
    COLOR_UNKNOWN,
    OUTPUTS_CHARTS,
    RATE_DENOMINATOR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

# Core signal-colour palette – warm & muted, not aggressive
_AMBER        = "#C8891C"   # warm golden amber
_NONAMBER     = "#A04848"   # dusty brick / terracotta
_AMBER_FILL   = "#E8C070"   # lighter fill for area shading
_NONAMBER_FILL = "#D08080"  # lighter fill for area shading
_MIXED        = "#8E9BA8"   # blue-grey neutral
_UNKNOWN      = "#BABABA"   # light silver

_COLOR_PALETTE = {
    COLOR_AMBER:     _AMBER,
    COLOR_NON_AMBER: _NONAMBER,
    COLOR_MIXED:     _MIXED,
    COLOR_UNKNOWN:   _UNKNOWN,
}
_FILL_PALETTE = {
    COLOR_AMBER:     _AMBER_FILL,
    COLOR_NON_AMBER: _NONAMBER_FILL,
}

# Canvas / typography
_BG         = "#F8F5EE"   # warm cream
_GRID       = "#E4D8C8"   # warm grid
_SPINE      = "#C8BCA8"   # spine / border colour
_TEXT       = "#2C2C2C"   # charcoal
_SUBTEXT    = "#6A6055"   # secondary text
_LABEL_SIZE = 10
_TITLE_SIZE = 12

_DPI          = 180
_FIGSIZE_WIDE = (11, 6)
_FIGSIZE_SQ   = (8, 7)


def _apply_style(ax: plt.Axes) -> None:
    """Apply the shared visual style to an Axes object."""
    fig = ax.get_figure()
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.grid(color=_GRID, linewidth=0.8, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)


def _source_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.004, text, ha="left", va="bottom",
             fontsize=6.5, color=_SUBTEXT)


def _save(fig: plt.Figure, name: str) -> Path:
    OUTPUTS_CHARTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_CHARTS / name
    fig.savefig(path, dpi=_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved chart: %s", path.name)
    return path


def _auto_rate_scale(median_raw: float) -> tuple[float, str]:
    if median_raw <= 0:
        return 1_000_000, "per million"
    for scale, label in [
        (1_000, "per 1,000"),
        (10_000, "per 10,000"),
        (100_000, "per 100,000"),
        (1_000_000, "per million"),
    ]:
        if 1.0 <= median_raw * scale < 1_000:
            return scale, label
    return 1_000_000, "per million"


def _auto_vol_scale(max_vol: float) -> tuple[float, str]:
    if max_vol >= 1_000_000:
        return 1_000_000.0, "millions"
    if max_vol >= 1_000:
        return 1_000.0, "thousands"
    return 1.0, "vehicles"


# ---------------------------------------------------------------------------
# Chart 1 – Rate by year (rolling average + CI band)
# ---------------------------------------------------------------------------

def chart_line_rate_by_year(
    summary_by_year: pd.DataFrame,
    rate_col: str,
    is_demo: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    _apply_style(ax)

    roll_col = [c for c in summary_by_year.columns if c.startswith("rolling_")]

    for color in [COLOR_AMBER, COLOR_NON_AMBER]:
        sub = summary_by_year[summary_by_year["signal_color"] == color].sort_values("year")
        if sub.empty:
            continue
        c      = _COLOR_PALETTE[color]
        cfill  = _FILL_PALETTE.get(color, c)
        label  = "Amber signal" if color == COLOR_AMBER else "Non-amber (red) signal"

        ax.fill_between(sub["year"], sub["ci_lower"].clip(lower=0), sub["ci_upper"],
                        color=cfill, alpha=0.25, label=f"{label} 95% CI")
        ax.plot(sub["year"], sub["mean"], color=c, lw=1.4, alpha=0.55,
                linestyle="--")
        if roll_col and roll_col[0] in sub.columns:
            ax.plot(sub["year"], sub[roll_col[0]], color=c, lw=2.6,
                    label=f"{label}")

    ax.set_xlabel("Year", fontsize=_LABEL_SIZE)
    ax.set_ylabel(f"Fatal crash rate  (per {RATE_DENOMINATOR:,} RVY)", fontsize=_LABEL_SIZE)
    ax.set_title("Normalised Fatal Crash Rate by Year\n"
                 "Amber vs Non-Amber Rear Turn Signals · US Market",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    ax.legend(fontsize=8.5, framealpha=0.85, facecolor=_BG, edgecolor=_SPINE)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    _source_note(fig, f"Source: NHTSA FARS 2012-2022. RVY = registered vehicle-years. Rate column: {rate_col}.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "line_chart_rate_by_year.png")


# ---------------------------------------------------------------------------
# Chart 2 – Pooled average rate (bar)
# ---------------------------------------------------------------------------

def chart_bar_avg_rate_by_color(
    summary_by_color: pd.DataFrame,
    rate_col: str,
    is_demo: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_style(ax)

    colors_order = [COLOR_AMBER, COLOR_NON_AMBER]
    sub = (summary_by_color[summary_by_color["signal_color"].isin(colors_order)]
           .set_index("signal_color").reindex(colors_order))

    labels     = ["Amber signal", "Non-amber (red) signal"]
    means      = sub["mean"].values
    ci_lo      = sub["ci_lower"].values
    ci_hi      = sub["ci_upper"].values
    yerr_lo    = np.clip(means - ci_lo, 0, None)
    yerr_hi    = np.clip(ci_hi - means, 0, None)
    bar_colors = [_COLOR_PALETTE[c] for c in colors_order]

    bars = ax.bar(labels, means, color=bar_colors, width=0.45, alpha=0.88,
                  yerr=[yerr_lo, yerr_hi], capsize=7,
                  error_kw={"elinewidth": 1.8, "ecolor": _TEXT, "capthick": 1.8},
                  zorder=3)

    for bar, mean, n in zip(bars, means, sub["n"].values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(yerr_hi) * 0.08,
                f"{mean:.3f}\n(n={int(n)})",
                ha="center", va="bottom", fontsize=9.5, color=_TEXT)

    ax.set_ylabel(f"Mean fatal crash rate  (per {RATE_DENOMINATOR:,} RVY)",
                  fontsize=_LABEL_SIZE)
    ax.set_title("Average Normalised Crash Rate · Full Period Pooled\n"
                 "Amber vs Non-Amber Rear Turn Signals · US Market",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    _source_note(fig, f"Error bars = 95% bootstrap CI. Source: {rate_col}.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "bar_chart_avg_rate_by_color.png")


# ---------------------------------------------------------------------------
# Chart 3 – Boxplot: rate distribution
# ---------------------------------------------------------------------------

def chart_boxplot_rate_distribution(
    df: pd.DataFrame,
    rate_col: str,
    is_demo: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQ)
    _apply_style(ax)

    colors_order = [COLOR_AMBER, COLOR_NON_AMBER, COLOR_UNKNOWN]
    data, labels, bp_colors = [], [], []

    for c in colors_order:
        sub = df[df["rear_signal_color_standardized"] == c][rate_col].dropna()
        if len(sub) < 2:
            continue
        data.append(sub.values)
        name = "Amber" if c == COLOR_AMBER else ("Non-amber (red)" if c == COLOR_NON_AMBER else c.title())
        labels.append(f"{name}\n(n={len(sub)})")
        bp_colors.append(_COLOR_PALETTE[c])

    if not data:
        logger.warning("No data for boxplot.")
        plt.close(fig)
        return OUTPUTS_CHARTS / "boxplot_rate_distribution.png"

    bp = ax.boxplot(data, patch_artist=True, notch=True, vert=True,
                    medianprops={"color": _TEXT, "linewidth": 2.2},
                    flierprops={"marker": "o", "markersize": 3.5,
                                "alpha": 0.45, "markeredgewidth": 0},
                    whiskerprops={"color": _SPINE, "linewidth": 1.2},
                    capprops={"color": _SPINE, "linewidth": 1.2},
                    boxprops={"linewidth": 1.2})

    for patch, color in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f"Fatal crash rate  (per {RATE_DENOMINATOR:,} RVY)", fontsize=_LABEL_SIZE)
    ax.set_title("Distribution of Normalised Crash Rates\nby Rear Turn-Signal Colour",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    _source_note(fig, "Notches = 95% CI around median. Outlier dots shown.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "boxplot_rate_distribution.png")


# ---------------------------------------------------------------------------
# Chart 4 – Manufacturer comparison
# ---------------------------------------------------------------------------

def chart_manufacturer_comparison(
    summary_by_manufacturer: pd.DataFrame,
    rate_col: str,
    top_n: int = 14,
    is_demo: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    _apply_style(ax)

    amber_df     = summary_by_manufacturer[summary_by_manufacturer["signal_color"] == COLOR_AMBER]
    non_amber_df = summary_by_manufacturer[summary_by_manufacturer["signal_color"] == COLOR_NON_AMBER]

    common    = set(amber_df["manufacturer"]) & set(non_amber_df["manufacturer"])
    one_sided = (set(amber_df[amber_df["n"] >= 5]["manufacturer"])
                 | set(non_amber_df[non_amber_df["n"] >= 5]["manufacturer"]))
    all_makes = sorted(common | one_sided)[:top_n]

    if not all_makes:
        logger.warning("No manufacturers for comparison chart.")
        plt.close(fig)
        return OUTPUTS_CHARTS / "manufacturer_comparison.png"

    x     = np.arange(len(all_makes))
    width = 0.38

    amber_means = [
        amber_df.loc[amber_df["manufacturer"] == m, "mean"].values[0]
        if m in amber_df["manufacturer"].values else np.nan for m in all_makes
    ]
    non_amber_means = [
        non_amber_df.loc[non_amber_df["manufacturer"] == m, "mean"].values[0]
        if m in non_amber_df["manufacturer"].values else np.nan for m in all_makes
    ]

    ax.bar(x - width / 2, amber_means,     width, label="Amber signal",
           color=_AMBER,    alpha=0.85, zorder=3)
    ax.bar(x + width / 2, non_amber_means, width, label="Non-amber (red) signal",
           color=_NONAMBER, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(all_makes, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel(f"Mean fatal crash rate  (per {RATE_DENOMINATOR:,} RVY)",
                  fontsize=_LABEL_SIZE)
    ax.set_title("Normalised Crash Rate by Manufacturer\nAmber vs Non-Amber Rear Turn Signals",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    ax.legend(fontsize=10, framealpha=0.85, facecolor=_BG, edgecolor=_SPINE)
    _source_note(fig, f"Only manufacturers with n≥5 vehicle-model-years shown. Source: {rate_col}.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "manufacturer_comparison.png")


# ---------------------------------------------------------------------------
# Chart 5 – Model count by year (stacked area)
# ---------------------------------------------------------------------------

def chart_model_count_by_year(
    df: pd.DataFrame,
    is_demo: bool = False,
) -> Path:
    year_col = "incident_year" if "incident_year" in df.columns else "model_year"
    fig, ax  = plt.subplots(figsize=_FIGSIZE_WIDE)
    _apply_style(ax)

    colors_order = [COLOR_AMBER, COLOR_NON_AMBER, COLOR_UNKNOWN]
    labels_map   = {COLOR_AMBER: "Amber", COLOR_NON_AMBER: "Non-amber (red)",
                    COLOR_UNKNOWN: "Unknown"}
    years = sorted(df[year_col].dropna().unique())

    data: dict[str, list[int]] = {c: [] for c in colors_order}
    for yr in years:
        yr_df = df[df[year_col] == yr]
        for c in colors_order:
            data[c].append(int((yr_df["rear_signal_color_standardized"] == c).sum()))

    bottom = np.zeros(len(years))
    for c in colors_order:
        vals = np.array(data[c], dtype=float)
        ax.bar(years, vals, bottom=bottom, color=_COLOR_PALETTE[c],
               alpha=0.85, label=labels_map[c], zorder=3)
        bottom += vals

    ax.set_xlabel("Year", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Vehicle-model-year observations", fontsize=_LABEL_SIZE)
    ax.set_title("Crash Observations by Signal Colour and Year\n"
                 "Stacked — shows relative fleet representation in FARS data",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    ax.legend(fontsize=9, framealpha=0.85, facecolor=_BG, edgecolor=_SPINE)
    _source_note(fig, "Source: NHTSA FARS 2012-2022 matched to researcher-curated blinker dataset.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "model_count_by_year.png")


# ---------------------------------------------------------------------------
# Chart 6 – Sensitivity analysis (dot-and-whisker)
# ---------------------------------------------------------------------------

def chart_sensitivity_analysis(
    sensitivity: pd.DataFrame,
    rate_col: str,
    is_demo: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_style(ax)

    scenarios  = sensitivity["scenario"].unique()
    y_positions = np.arange(len(scenarios))
    offset = 0.18

    for color, sign in [(COLOR_AMBER, -1), (COLOR_NON_AMBER, 1)]:
        label = "Amber signal" if color == COLOR_AMBER else "Non-amber (red) signal"
        c     = _COLOR_PALETTE[color]
        for i, scenario in enumerate(scenarios):
            row = sensitivity[
                (sensitivity["scenario"] == scenario) & (sensitivity["signal_color"] == color)
            ]
            if row.empty:
                continue
            mean  = row["mean_rate"].values[0]
            ci_lo = row["ci_lower"].values[0]
            ci_hi = row["ci_upper"].values[0]
            y     = y_positions[i] + sign * offset
            ax.errorbar(
                mean, y,
                xerr=[[max(mean - ci_lo, 0)], [max(ci_hi - mean, 0)]],
                fmt="o", color=c, markersize=8, capsize=5,
                linewidth=1.5, capthick=1.5,
                label=label if i == 0 else "_nolegend_",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=9)
    ax.set_xlabel(f"Mean fatal crash rate  (per {RATE_DENOMINATOR:,} RVY)", fontsize=_LABEL_SIZE)
    ax.set_title("Sensitivity Analysis: Rate Estimates Across Inclusion Scenarios\n"
                 "Error bars = 95% bootstrap CI",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    ax.legend(fontsize=9, framealpha=0.85, facecolor=_BG, edgecolor=_SPINE)
    ax.axvline(0, color=_SPINE, linewidth=1.0, linestyle="--")
    _source_note(fig, f"Scenarios differ in which rows are included. Source: {rate_col}.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "sensitivity_analysis.png")


# ---------------------------------------------------------------------------
# Chart 7 – Raw crash count by year
# ---------------------------------------------------------------------------

def chart_raw_crash_count_by_year(
    df: pd.DataFrame,
    is_demo: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    _apply_style(ax)

    year_col  = "incident_year" if "incident_year" in df.columns else "model_year"
    count_col = next(
        (c for c in ["fatal_crash_count", "occupant_death_count",
                     "total_incident_count", "claim_count"]
         if c in df.columns and df[c].notna().any()), None
    )
    if count_col is None:
        plt.close(fig)
        return OUTPUTS_CHARTS / "raw_crash_count_by_year.png"

    plotted = False
    for color in [COLOR_AMBER, COLOR_NON_AMBER]:
        sub = (df[df["rear_signal_color_standardized"] == color]
               .groupby(year_col, dropna=True)[count_col].sum()
               .reset_index().sort_values(year_col))
        if sub.empty:
            continue
        c     = _COLOR_PALETTE[color]
        cfill = _FILL_PALETTE.get(color, c)
        label = "Amber signal" if color == COLOR_AMBER else "Non-amber (red) signal"
        ax.fill_between(sub[year_col], 0, sub[count_col], color=cfill, alpha=0.25)
        ax.plot(sub[year_col], sub[count_col], color=c, lw=2.5,
                marker="o", markersize=6, label=label, zorder=3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return OUTPUTS_CHARTS / "raw_crash_count_by_year.png"

    ax.set_xlabel("Year", fontsize=_LABEL_SIZE)
    ax.set_ylabel(count_col.replace("_", " ").title(), fontsize=_LABEL_SIZE)
    ax.set_title("Raw Fatal Crash Count by Year\nAmber vs Non-Amber Rear Turn Signals · US Market",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    ax.legend(fontsize=10, framealpha=0.85, facecolor=_BG, edgecolor=_SPINE)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.text(0.5, 0.97,
            "NOTE: Raw counts reflect fleet size, not risk per vehicle.\n"
            "A larger fleet produces more crashes even at a LOWER per-vehicle rate.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#6B1A1A",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF0D8",
                      edgecolor="#C8891C", alpha=0.92))

    _source_note(fig, "Source: NHTSA FARS 2012-2022. Raw counts only — see normalised rate charts for comparison.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "raw_crash_count_by_year.png")


# ---------------------------------------------------------------------------
# Helpers shared by crash-rate chart variants
# ---------------------------------------------------------------------------

def _build_rate_series(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], float, float, str, str]:
    """Compute per-year crash rate for each signal colour. Returns (series, scale, unit_label, count_col, denom_col)."""
    year_col  = "incident_year" if "incident_year" in df.columns else "model_year"
    count_col = next(
        (c for c in ["fatal_crash_count", "occupant_death_count",
                     "total_incident_count", "claim_count"]
         if c in df.columns and df[c].notna().any()), None
    )
    denom_col = next(
        (c for c in ["vehicles_on_road", "registered_vehicle_years"]
         if c in df.columns and df[c].notna().any()), None
    )

    color_series: dict[str, pd.DataFrame] = {}
    all_raw: list[float] = []

    if count_col and denom_col:
        for color in [COLOR_AMBER, COLOR_NON_AMBER]:
            sub = (df[df["rear_signal_color_standardized"] == color]
                   .groupby(year_col, dropna=True)[[count_col, denom_col]].sum()
                   .reset_index().sort_values(year_col))
            sub = sub[sub[denom_col] > 0].copy()
            if sub.empty:
                continue
            sub["_raw"] = sub[count_col] / sub[denom_col]
            all_raw.extend(sub["_raw"].tolist())
            color_series[color] = sub

    median_raw = float(pd.Series(all_raw).median()) if all_raw else 0.0
    scale, unit_label = _auto_rate_scale(median_raw)
    return color_series, scale, unit_label, (count_col or ""), (denom_col or "")


# ---------------------------------------------------------------------------
# Chart 8-A – Split-panel: rate (top) + fleet volume (bottom)
# ---------------------------------------------------------------------------

def chart_crash_rate_split_panel(
    df: pd.DataFrame,
    is_demo: bool = False,
) -> Path:
    """
    Main crash-rate chart.

    Top panel  : crash rate per signal colour as bold lines with shaded area.
    Bottom panel: fleet volume (vehicles on road) as bars — context for the rate.
    Sharing the x-axis makes the relationship immediately clear.
    """
    color_series, scale, unit_label, count_col, denom_col = _build_rate_series(df)
    if not color_series:
        return OUTPUTS_CHARTS / "crash_rate_by_year.png"

    vol_scale, vol_label = _auto_vol_scale(
        max(sub[denom_col].max() for sub in color_series.values())
    )
    all_years = sorted({y for sub in color_series.values()
                        for y in sub.iloc[:, 0].tolist()})
    year_col = "incident_year" if "incident_year" in df.columns else "model_year"

    fig, (ax_rate, ax_vol) = plt.subplots(
        2, 1, figsize=(11, 7.5),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
        sharex=True,
    )
    for ax in (ax_rate, ax_vol):
        _apply_style(ax)

    fig.patch.set_facecolor(_BG)

    # -- Top panel: rate lines --
    for color, sub in color_series.items():
        c     = _COLOR_PALETTE[color]
        cfill = _FILL_PALETTE.get(color, c)
        label = "Amber signal" if color == COLOR_AMBER else "Non-amber (red) signal"
        sub["rate"] = sub["_raw"] * scale

        ax_rate.fill_between(sub[year_col], 0, sub["rate"],
                             color=cfill, alpha=0.20)
        ax_rate.plot(sub[year_col], sub["rate"],
                     color=c, lw=2.8, marker="o", markersize=7,
                     label=label, zorder=4)

        # Annotate last data point
        last = sub.iloc[-1]
        ax_rate.annotate(
            f"{last['rate']:.1f}",
            xy=(last[year_col], last["rate"]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=8.5, color=c, fontweight="bold",
        )

    ax_rate.set_ylabel(f"Fatal crashes  ({unit_label} vehicles)", fontsize=_LABEL_SIZE)
    ax_rate.set_title(
        "Fatal Crash Rate by Year  ·  Amber vs Non-Amber Rear Turn Signals  ·  US Market\n"
        "Lower panel shows fleet size — larger fleets produce more raw crashes",
        fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT, pad=10,
    )
    ax_rate.legend(fontsize=9.5, framealpha=0.9, facecolor=_BG, edgecolor=_SPINE,
                   loc="upper left")
    ax_rate.set_ylim(bottom=0)
    ax_rate.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # -- Bottom panel: fleet volume bars --
    bar_w = 0.38
    colors_order = [c for c in [COLOR_AMBER, COLOR_NON_AMBER] if c in color_series]
    for i, color in enumerate(colors_order):
        sub   = color_series[color]
        c     = _COLOR_PALETTE[color]
        cfill = _FILL_PALETTE.get(color, c)
        label = "Amber fleet" if color == COLOR_AMBER else "Non-amber fleet"
        x_pos = sub[year_col] + (i - len(colors_order) / 2 + 0.5) * bar_w
        ax_vol.bar(x_pos, sub[denom_col] / vol_scale,
                   width=bar_w, color=cfill, alpha=0.75,
                   edgecolor=c, linewidth=0.8, label=label, zorder=3)

    ax_vol.set_ylabel(f"Fleet\n({vol_label})", fontsize=8.5, color=_SUBTEXT)
    ax_vol.tick_params(axis="y", labelcolor=_SUBTEXT, labelsize=8)
    ax_vol.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_vol.set_ylim(bottom=0)
    ax_vol.legend(fontsize=8, framealpha=0.85, facecolor=_BG, edgecolor=_SPINE,
                  loc="upper left")
    ax_vol.set_xlabel("Year", fontsize=_LABEL_SIZE)

    _source_note(
        fig,
        f"Rate = {count_col} / {denom_col} x {scale:,}.  "
        f"Fleet = {denom_col} ({vol_label}).  Source: NHTSA FARS 2012-2022 + sales survival model.",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    return _save(fig, "crash_rate_by_year.png")


# ---------------------------------------------------------------------------
# Chart 8-B – Connected dot plot with fleet-size markers
# ---------------------------------------------------------------------------

def chart_crash_rate_dotplot(
    df: pd.DataFrame,
    is_demo: bool = False,
) -> Path:
    """
    Alternative style B: connected dot plot.

    Each dot = one year.  Dot size scales with fleet size, so you can see
    that years with a large dot are statistically more reliable.
    End-of-line labels replace the legend for a cleaner look.
    """
    color_series, scale, unit_label, count_col, denom_col = _build_rate_series(df)
    if not color_series:
        return OUTPUTS_CHARTS / "crash_rate_style_dotplot.png"

    year_col = "incident_year" if "incident_year" in df.columns else "model_year"

    # Dot size: map fleet size to marker area (20–400 pt²)
    all_vols = np.concatenate([sub[denom_col].values for sub in color_series.values()])
    vol_min, vol_max = all_vols.min(), all_vols.max()

    def _marker_size(v: float) -> float:
        if vol_max == vol_min:
            return 120.0
        return 30 + 320 * (v - vol_min) / (vol_max - vol_min)

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_style(ax)
    fig.patch.set_facecolor(_BG)

    for color, sub in color_series.items():
        c     = _COLOR_PALETTE[color]
        cfill = _FILL_PALETTE.get(color, c)
        label = "Amber signal" if color == COLOR_AMBER else "Non-amber (red) signal"
        sub = sub.copy()
        sub["rate"]       = sub["_raw"] * scale
        sub["dot_size"]   = sub[denom_col].apply(_marker_size)

        # Thin connecting line
        ax.plot(sub[year_col], sub["rate"],
                color=c, lw=1.4, alpha=0.55, zorder=2)

        # Dots sized by fleet
        ax.scatter(sub[year_col], sub["rate"],
                   s=sub["dot_size"], color=cfill,
                   edgecolors=c, linewidths=1.5, zorder=4, alpha=0.85)

        # End label
        last = sub.iloc[-1]
        ax.text(last[year_col] + 0.15, last["rate"],
                label, fontsize=9, color=c, fontweight="bold",
                va="center")

    ax.set_xlabel("Year", fontsize=_LABEL_SIZE)
    ax.set_ylabel(f"Fatal crashes  ({unit_label} vehicles)", fontsize=_LABEL_SIZE)
    ax.set_title(
        "Fatal Crash Rate by Year  ·  Amber vs Non-Amber Rear Turn Signals\n"
        "Dot size = fleet size (larger dot = more vehicles, more statistical weight)",
        fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT,
    )
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Size legend — use a dummy invisible point to avoid empty-array crash
    import matplotlib.lines as mlines
    legend_handles = []
    for fleet_k, label_txt in [(100_000, "100k vehicles"), (500_000, "500k"),
                               (1_000_000, "1M")]:
        if fleet_k > vol_max:
            continue
        ms = (_marker_size(fleet_k) ** 0.5) / 2  # convert area→radius for markersize
        h = mlines.Line2D([], [], linestyle="none", marker="o",
                          markersize=ms, color="#BBBBBB",
                          markeredgecolor=_SPINE, markeredgewidth=1.2,
                          alpha=0.7, label=label_txt)
        legend_handles.append(h)
    if legend_handles:
        ax.legend(handles=legend_handles, title="Fleet size", fontsize=8.5,
                  title_fontsize=8.5, framealpha=0.85, facecolor=_BG,
                  edgecolor=_SPINE, loc="upper right")

    _source_note(fig,
                 f"Rate = {count_col} / {denom_col} x {scale:,}.  Source: NHTSA FARS 2012-2022.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "crash_rate_style_dotplot.png")


# ---------------------------------------------------------------------------
# Chart 8-C – Grouped annual bars (side-by-side year comparison)
# ---------------------------------------------------------------------------

def chart_crash_rate_grouped_bars(
    df: pd.DataFrame,
    is_demo: bool = False,
) -> Path:
    """
    Alternative style C: grouped bar chart — one pair of bars per year.

    Ideal for directly comparing the two groups year-by-year without needing
    to trace lines. The gap between bars is instantly visible.
    """
    color_series, scale, unit_label, count_col, denom_col = _build_rate_series(df)
    if not color_series:
        return OUTPUTS_CHARTS / "crash_rate_style_bars.png"

    year_col = "incident_year" if "incident_year" in df.columns else "model_year"
    all_years = sorted({y for sub in color_series.values()
                        for y in sub[year_col].tolist()})

    fig, ax = plt.subplots(figsize=(13, 6))
    _apply_style(ax)
    fig.patch.set_facecolor(_BG)

    bar_w = 0.38
    x     = np.arange(len(all_years))
    colors_order = [c for c in [COLOR_AMBER, COLOR_NON_AMBER] if c in color_series]

    for i, color in enumerate(colors_order):
        sub   = color_series[color].copy()
        sub["rate"] = sub["_raw"] * scale
        yr_rate = dict(zip(sub[year_col], sub["rate"]))
        rates   = [yr_rate.get(yr, np.nan) for yr in all_years]

        c     = _COLOR_PALETTE[color]
        cfill = _FILL_PALETTE.get(color, c)
        label = "Amber signal" if color == COLOR_AMBER else "Non-amber (red) signal"

        offset = (i - len(colors_order) / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, rates, bar_w,
                        color=cfill, edgecolor=c, linewidth=1.2,
                        label=label, alpha=0.85, zorder=3)

        # Small rate label on top of each bar
        for bar, rate in zip(bars, rates):
            if not np.isnan(rate):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{rate:.1f}",
                        ha="center", va="bottom", fontsize=7, color=c, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(all_years, fontsize=9.5)
    ax.set_xlabel("Year", fontsize=_LABEL_SIZE)
    ax.set_ylabel(f"Fatal crashes  ({unit_label} vehicles)", fontsize=_LABEL_SIZE)
    ax.set_title(
        "Fatal Crash Rate by Year  ·  Amber vs Non-Amber Rear Turn Signals\n"
        "Grouped bars — each pair shows one year's comparison directly",
        fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT,
    )
    ax.legend(fontsize=10, framealpha=0.85, facecolor=_BG, edgecolor=_SPINE)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    _source_note(fig,
                 f"Rate = {count_col} / {denom_col} x {scale:,}.  Source: NHTSA FARS 2012-2022.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "crash_rate_style_bars.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_all_charts(
    analysis_df: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    rate_col: str,
    is_demo: bool = False,
) -> list[Path]:
    """Generate all charts and return a list of output paths."""
    paths: list[Path] = []

    def _try(fn: Any, *args: Any, **kwargs: Any) -> None:
        try:
            paths.append(fn(*args, **kwargs))
        except Exception as exc:
            logger.error("Chart '%s' failed: %s", fn.__name__, exc, exc_info=True)

    _try(chart_line_rate_by_year,
         results.get("summary_by_year", pd.DataFrame()), rate_col, is_demo)

    _try(chart_bar_avg_rate_by_color,
         results.get("summary_by_signal_color", pd.DataFrame()), rate_col, is_demo)

    _try(chart_boxplot_rate_distribution, analysis_df, rate_col, is_demo)

    _try(chart_manufacturer_comparison,
         results.get("summary_by_manufacturer", pd.DataFrame()), rate_col, is_demo=is_demo)

    _try(chart_model_count_by_year, analysis_df, is_demo)

    _try(chart_sensitivity_analysis,
         results.get("sensitivity_analysis", pd.DataFrame()), rate_col, is_demo)

    _try(chart_raw_crash_count_by_year, analysis_df, is_demo)

    # Three visual styles for the crash-rate-by-year chart
    _try(chart_crash_rate_split_panel,  analysis_df, is_demo)  # main (saved as crash_rate_by_year.png)
    _try(chart_crash_rate_dotplot,      analysis_df, is_demo)  # style B
    _try(chart_crash_rate_grouped_bars, analysis_df, is_demo)  # style C

    return paths
