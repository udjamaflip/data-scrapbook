# GitHub Copilot Instructions — data-scrapbook

This repository is a collection of self-contained data analysis projects ("scrapbook").
Each project lives in its own subdirectory.

---

## Repository conventions

- **One project per subdirectory.** Each project is fully self-contained with its own
  `README.md`, `requirements.txt`, `pyproject.toml`, `src/`, `scripts/`, and `data/` tree.
- **No cross-project imports.** Projects must not import from each other.
- **Modular pipeline pattern.** Each project follows: ingest → clean → normalise → match → analyse → output.
- **Data files are never committed.** Real input CSVs go in `data/raw/` (gitignored).
  Only template CSVs (`data/raw/templates/`) and synthetic demo files (`*_demo.csv`) are committed.
- **Generated outputs are never committed.** `data/intermediate/`, `data/final/`,
  `outputs/charts/`, `outputs/tables/` are all gitignored.
- **Python 3.10+.** Use `from __future__ import annotations` for forward references.
- **pandas for data, matplotlib for charts.** No seaborn. No plotly.
- **Type hints on all public functions.**
- **Logging via the standard library `logging` module.** Never use `print()` for pipeline
  output — use `logger.info()` / `logger.warning()` etc.
- **Fail fast with actionable errors.** If a required input file is missing, raise a clear
  `FileNotFoundError` with the expected path and a hint about how to generate it.
- **Never silently drop rows.** Log a warning and write dropped rows to an `unmatched_rows.csv`.
- **Rates over raw counts.** Never present raw crash counts as a meaningful comparison.
  Always normalise by exposure (registered vehicle-years or equivalent).

---

## Project: turn-signal-crash-rate

**Location:** `turn-signal-crash-rate/`

**Research question:** Do US-market vehicles with non-amber rear turn signals have higher
fatal crash rates than vehicles with amber rear turn signals, after normalising for fleet exposure?

**Data sources:**
- `scripts/build_real_data.py` — downloads NHTSA FARS (2012–2022) and builds blinker colour
  + fleet exposure datasets. Run this first before the main pipeline.
- `data/raw/blinker_colors.csv` — researcher-curated: make/model/year → rear signal colour
- `data/raw/exposure_data.csv` — fleet size estimates (annual sales × survival curve)
- `data/raw/incident_data.csv` — NHTSA FARS fatal crash counts by make/model/year

**Key design decisions:**
- Blinker colour categories: `amber`, `non_amber`, `unknown` (no `mixed`)
- Core analysis excludes rows with `confidence_score < 0.7`
- Primary rate: `fatal_crashes_per_100k_rvy` (per 100 000 registered vehicle-years)
- Statistical tests: Welch's t-test + Mann-Whitney U + Cohen's d + bootstrap 95% CIs
- US market only (FMVSS 108 permits red or amber — European variants differ)

**Running the pipeline:**
```bash
cd turn-signal-crash-rate
pip install -r requirements.txt
python scripts/build_real_data.py   # ~1 min, downloads ~300 MB NHTSA data
python -m src.main                  # full pipeline
python -m src.main --demo           # smoke-test with synthetic data (no download)
```

**Key blinker colour facts (US market):**
- Toyota, Honda, Subaru, Nissan, Mazda, Hyundai, Kia → **amber** (global ECE standard maintained)
- Porsche, Volvo, Jaguar, Land Rover → **amber** (ECE amber maintained in US-spec)
- Ford, GM (Chevrolet/GMC/Buick/Cadillac), Ram, Jeep, Dodge, Chrysler, Lincoln → **non_amber** (red)
- BMW, Audi, Mercedes-Benz, Volkswagen → **non_amber** in US (FMVSS 108 allows shared red housing)
- Tesla Model S/X/3 → **non_amber**; Model Y 2020 → **non_amber**, 2021+ → **amber**
- Source: FMVSS 108 (US) permits either colour; ECE Reg 6 (Europe) requires amber

**Extending the pipeline:**
- Add a new make/model/year blinker entry: edit `_BLINKER_SPEC` in `scripts/build_real_data.py`
- Fix a name mismatch: edit `data/raw/templates/manual_vehicle_overrides.csv`
- Add a make alias: edit `data/raw/templates/make_aliases.yaml`
- Add a new chart: add a function to `src/charts.py` following the `_apply_style()` pattern,
  then call it from `generate_all_charts()` at the bottom of that file

**Chart palette:**
- Background: `#F8F5EE` (warm cream)
- Amber signal: `#C8891C` (line/edge), `#E8C070` (fill)
- Non-amber signal: `#A04848` (line/edge), `#D08080` (fill)
- Grid: `#E4D8C8`
- Always call `_apply_style(ax)` on every Axes object; always pass
  `facecolor=fig.get_facecolor()` to `fig.savefig()`

---

## Adding a new project

1. Create a new subdirectory: `mkdir <project-name>`
2. Copy the structure from `turn-signal-crash-rate/` as a template
3. Add a row to the table in the root `README.md`
4. Add project-specific data ignore rules to `<project-name>/.gitignore`

---

## What Copilot should NOT do in this repo

- Do not commit real data files (any `*.csv` not in `templates/` or ending in `_demo.csv`)
- Do not use seaborn, plotly, or bokeh — matplotlib only
- Do not present raw counts as findings — always normalise
- Do not silently drop rows — always log and save unmatched records
- Do not add `print()` statements — use the `logging` module
- Do not hardcode absolute paths — use `Path(__file__).resolve().parent` patterns
