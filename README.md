# US Rear Turn-Signal Colour vs Fatal Crash Rate

> **Research question:** Do US-market vehicles with **non-amber** rear turn signals
> have higher fatal crash rates than vehicles with **amber** rear turn signals,
> after normalising for fleet exposure?

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚠ Important caveats

1. **This analysis is observational, not experimental.** Correlation is not causation.
   Dozens of confounders (vehicle class, age, geography, driver demographics, safety
   technology era) are not fully controlled for in this pipeline.

2. **Raw crash counts are meaningless without exposure normalisation.**
   A popular model will appear in more crashes simply because more of them are on the
   road. All rates in this project are per **100 000 registered vehicle-years (RVY)**.

3. **The blinker colour dataset must be manually curated.**
   No comprehensive public source exists that maps every US make/model/year to rear
   turn-signal colour. The pipeline ships with clearly-labelled **synthetic demo data**
   for validation only. Real conclusions require a real blinker dataset.

4. **Exposure data granularity is a known weak point.**
   FHWA registration data is aggregated by vehicle class, not make/model.
   Ideal sources (IHS Markit, Experian AutoCount) are proprietary.
   The pipeline accepts user-supplied data in the template format.

5. **FARS covers only fatal crashes.**
   Non-fatal crash data (NHTSA CRSS) is a sample and requires probability weighting.
   Claims data from HLDI would be ideal but requires licensing.

---

## Project goal

Build a reproducible, production-quality pipeline that:

1. Classifies US-market vehicles by rear turn-signal colour (amber / non_amber / unknown)
2. Joins that against fleet exposure data (registered vehicle-years by make/model/year)
3. Joins that against crash/incident data (NHTSA FARS or equivalent)
4. Computes **normalised crash rates** per 100 000 registered vehicle-years
5. Performs **statistical comparisons** (Welch's t-test, Mann-Whitney U, Cohen's d, bootstrap CIs)
6. Generates **charts and summary tables** across many model years
7. Documents **assumptions, data quality issues, and limitations** clearly

---

## Directory structure

```
data-scrapbook/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example                  ← copy to .env and configure
├── scripts/
│   └── build_real_data.py        ← downloads NHTSA FARS + builds blinker/exposure CSVs
├── src/
│   ├── config.py                 ← paths, constants, logging
│   ├── utils.py                  ← shared helpers, rate calculations, stats
│   ├── vehicle_normalization.py  ← make/model/year normalisation + alias mapping
│   ├── load_blinker_data.py
│   ├── clean_blinker_data.py
│   ├── load_exposure_data.py
│   ├── clean_exposure_data.py
│   ├── load_incident_data.py
│   ├── clean_incident_data.py
│   ├── matching.py               ← joins all three datasets with match-quality flags
│   ├── analysis.py               ← normalised rates, statistics, stratification
│   ├── charts.py                 ← matplotlib chart generation
│   └── main.py                   ← pipeline orchestrator + CLI
├── data/
│   ├── raw/
│   │   ├── blinker_colors.csv        ← YOU SUPPLY (or generate via build_real_data.py)
│   │   ├── exposure_data.csv         ← YOU SUPPLY (or generate via build_real_data.py)
│   │   ├── incident_data.csv         ← YOU SUPPLY (or generate via build_real_data.py)
│   │   ├── blinker_colors_demo.csv   ← synthetic demo (pipeline smoke-test only)
│   │   ├── exposure_data_demo.csv    ← synthetic demo
│   │   ├── incident_data_demo.csv    ← synthetic demo
│   │   └── templates/
│   │       ├── blinker_colors_template.csv
│   │       ├── exposure_data_template.csv
│   │       ├── incident_data_template.csv
│   │       ├── manual_vehicle_overrides.csv  ← edit to fix name mismatches
│   │       ├── make_aliases.yaml             ← edit to add make aliases
│   │       └── model_aliases.yaml            ← edit to add model aliases
│   ├── intermediate/              ← cleaned single-source files (auto-generated)
│   └── final/                     ← analysis-ready joined datasets (auto-generated)
└── outputs/
    ├── charts/                    ← PNG charts (auto-generated)
    └── tables/                    ← CSV summary tables (auto-generated)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download real data (recommended)

`scripts/build_real_data.py` handles three things automatically:

- **Blinker colours** — researcher-curated dataset (~5 000 make/model/year rows) built
  from manufacturer specs and owner documentation
- **Exposure estimates** — US annual sales data with a survival-curve fleet model
- **NHTSA FARS crashes** — downloads 2012–2022 fatality data directly from nhtsa.gov
  (~300 MB, takes ~1 minute)

```bash
python scripts/build_real_data.py
```

Then run the full pipeline:

```bash
python -m src.main
```

### 3. Smoke-test with synthetic demo data (no downloads required)

```bash
python -m src.main --demo
```

> **Do not interpret demo output as real findings.**

---

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--demo` | off | Use bundled synthetic demo data |
| `--input-dir PATH` | `data/raw` | Directory containing input CSVs |
| `--output-dir PATH` | `outputs` | Root output directory |
| `--no-charts` | off | Skip chart generation |
| `--log-level LEVEL` | `INFO` | Logging verbosity (`DEBUG` / `INFO` / `WARNING`) |

---

## Required datasets

### A. Blinker colour dataset (`data/raw/blinker_colors.csv`)

`build_real_data.py` generates a curated starting point. If you want to extend or
correct it, the relevant data structure is `_BLINKER_SPEC` in `scripts/build_real_data.py`.

| Column | Required | Notes |
|---|---|---|
| `make` | ✓ | |
| `model` | ✓ | |
| `model_year` | ✓ | |
| `rear_signal_color_raw` | ✓ | Free text; pipeline standardises to amber / non_amber / unknown |
| `source` | ✓ | Provenance string |
| `confidence_score` | — | 0–1; rows < 0.7 excluded from headline analysis |
| `notes` | — | Free text |

Use `data/raw/templates/blinker_colors_template.csv` as schema reference.

### B. Exposure dataset (`data/raw/exposure_data.csv`)

`build_real_data.py` generates fleet estimates from annual US sales + a scrappage
survival curve. For higher-quality exposure data:

| Source | Granularity | Availability |
|---|---|---|
| IHS Markit / S&P Global Mobility | make/model/year/state | Proprietary |
| Experian AutoCount | make/model/year | Proprietary |
| R.L. Polk | make/model/year | Proprietary |
| FHWA Highway Statistics (MV-1) | Vehicle class only | Free |

### C. Incident dataset (`data/raw/incident_data.csv`)

`build_real_data.py` downloads NHTSA FARS directly. Primary free public source:

| Source | Coverage | URL |
|---|---|---|
| **NHTSA FARS** | Fatal crashes (census) | https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars |
| NHTSA CRSS | All severities (sample) | https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system |
| HLDI Loss Data | Insurance claims | Proprietary (IIHS) |

---

## How to update manual mappings

### Make / model aliases

Edit `data/raw/templates/make_aliases.yaml` or `model_aliases.yaml`:

```yaml
Volkswagen:
  - VW
  - Volkwagen        # covers common misspelling in source data
```

### Manual vehicle overrides

Edit `data/raw/templates/manual_vehicle_overrides.csv` for row-level corrections:

```csv
raw_make,raw_model,raw_model_year,override_make,override_model,override_model_year,override_notes
Chev,Silverado,2003,Chevrolet,Silverado,2003,"Source uses abbreviated make"
```

**Always fill in `override_notes`** — silent overrides are prohibited by design.

---

## How to interpret the charts

| Chart | What it shows |
|---|---|
| `crash_rate_by_year.png` | Rate over time (split panel: lines + fleet volume bars) |
| `crash_rate_style_dotplot.png` | Rate over time — dot size ∝ fleet size |
| `crash_rate_style_bars.png` | Grouped bars per year for direct comparison |
| `raw_crash_count_by_year.png` | Raw counts (no normalisation — illustrates the problem) |
| `bar_chart_avg_rate_by_color.png` | Pooled average ± 95% bootstrap CI |
| `boxplot_rate_distribution.png` | Distribution of per-model rates by colour |
| `manufacturer_comparison.png` | Rate by manufacturer, amber vs non-amber side-by-side |
| `model_count_by_year.png` | Data coverage — how many vehicle cohorts per colour per year |
| `sensitivity_analysis.png` | Results under different inclusion criteria |
| `line_chart_rate_by_year.png` | Simple rate lines (reference) |

**95% confidence intervals are bootstrap-based** (2 000 resamples). They reflect
uncertainty in the mean rate estimate, not prediction intervals.

---

## Major limitations

### 1. No causal identification
Signal colour is a design choice correlated with many other vehicle attributes
(country of origin, vehicle class, regulatory era, brand positioning). This
pipeline cannot isolate the effect of signal colour from those confounders.

### 2. Blinker colour data quality
The blinker colour dataset is manually curated and will be incomplete.
Rows with `eligibility_for_core_analysis = False` are excluded from the headline
result but included in sensitivity analyses.

### 3. Exposure data granularity
Without make/model-level registration counts, all rates are approximate.
The pipeline clearly flags `exposure_quality_flag = low` rows.
The bundled survival-curve model uses annual sales data and a 5.5% annual
scrappage rate — reasonable but not exact.

### 4. FARS covers only fatal crashes
Fatal crashes are the most reliably counted but represent a tiny fraction of
all crashes. The findings may not generalise to non-fatal or property-damage crashes.

### 5. Confounders not controlled
- Vehicle class (trucks have different crash dynamics than sedans)
- Driver demographics
- Geographic distribution (urban vs rural crash rates differ)
- Safety technology era (ABS, ESC, automatic braking)
- Luxury vs fleet vs commercial use patterns

Stratification support is built in — filter the `data/final/analysis_dataset.csv`
directly to probe subgroups.

---

## Why raw counts are misleading without exposure normalisation

Suppose:
- Model A (amber): 1 000 fatal crashes, 5 000 000 vehicles on road
- Model B (non-amber): 800 fatal crashes, 2 000 000 vehicles on road

Raw count → Model A looks more dangerous (1 000 > 800).

Normalised rate:
- Model A: 1 000 / 5 000 000 × 100 000 = **20 per 100k RVY**
- Model B: 800 / 2 000 000 × 100 000 = **40 per 100k RVY**

Model B is actually **twice as risky per vehicle** — the opposite conclusion.
This is why all analysis uses rates, not counts.

---

## Data dictionary

See `data/final/data_dictionary.csv` for machine-readable field definitions.

| Column | Datasets | Meaning |
|---|---|---|
| `make_norm` | all | Normalised make name (use for joins) |
| `model_norm` | all | Normalised model name (use for joins) |
| `model_year` | all | Model year (integer) |
| `rear_signal_color_standardized` | blinker, analysis | `amber` / `non_amber` / `unknown` |
| `confidence_score` | blinker, analysis | 0–1; rows < MIN_CONFIDENCE excluded from core |
| `eligibility_for_core_analysis` | blinker, analysis | True = included in headline comparison |
| `registered_vehicle_years` | exposure, analysis | Fleet exposure denominator |
| `fatal_crash_count` | incidents, analysis | Fatal crashes (null ≠ zero) |
| `fatal_crashes_per_100k_rvy` | analysis | Primary normalised rate |
| `match_quality` | analysis | `exact` / `normalized_exact` / `manual_override` / `unmatched` |
| `DEMO_DATA` | all | True if row is from synthetic demo — never mix with real data |

---

## Extending the pipeline

- **Add new data sources:** implement a new `load_*.py` + `clean_*.py` pair following the
  existing pattern, then call them from `main.py`.
- **Add stratification variables:** add columns (e.g. `body_class`, `safety_era`) to the
  blinker dataset and filter the analysis dataset accordingly.
- **Add regression models:** extend `analysis.py` with `statsmodels` OLS or Poisson
  regression controlling for year fixed effects and manufacturer fixed effects.
- **Bootstrap more statistics:** `utils.bootstrap_ci()` accepts any `statistic` callable.

---

## License

MIT License. See [LICENSE](LICENSE).

Real input data (NHTSA FARS etc.) is subject to the terms of its respective source agency.
The synthetic demo data is not real and must not be cited or published.
