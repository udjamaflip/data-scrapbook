# GitHub Copilot Instructions — data-scrapbook

This repository is a collection of self-contained data investigation projects.
Each project asks a specific question and answers it with real data, honest analysis,
and clearly documented limitations.

---

## The one rule that overrides everything else

**Never fabricate, synthesise, or simulate data.**

If real data is unavailable:
1. Find another source — no matter how manual or tedious that process is
2. If no source exists, say so explicitly and stop — do not approximate, interpolate,
   or invent plausible-looking numbers
3. A project with an honest "this data does not exist publicly" conclusion is
   more valuable than a project with fake data dressed up as real

This applies to everything: input datasets, sample rows, placeholder values,
"illustrative" examples, demo files, test fixtures. If it is not sourced, it
does not go in the analysis.

The only exception: clearly-labelled smoke-test fixtures used solely to verify
pipeline mechanics (not to draw conclusions), explicitly flagged with
DEMO_DATA=TRUE and never mixed with real data.

---

## Data sourcing principles

- **Cite every source.** Every dataset must have a source and source_url column
  or equivalent provenance field. If you cannot cite it, you cannot use it.
- **Record retrieval date.** For any manually collected or scraped data, record
  when it was retrieved — sources change.
- **Prefer primary sources.** Manufacturer specs > enthusiast wikis > forums.
  Government databases > news articles > blog posts.
- **Be honest about confidence.** Use a confidence_score (0-1) or equivalent flag.
  Low-confidence rows must be excluded from headline results, not quietly included.
- **Document dead ends.** If you tried a source and it did not work out, note that in
  the project README so future contributors do not repeat the effort.
- **Manual collection is fine.** Spending hours reading spec sheets and entering data
  by hand is legitimate data engineering. It just needs to be documented.

---

## Analysis principles

- **Rates over raw counts.** Never present raw counts as a meaningful comparison
  without normalising for exposure, population, or denominator.
- **Show the denominator.** Always make the exposure or base population visible in
  charts and tables — not just the rate.
- **Exclude, do not impute.** If data is missing or low-quality, exclude that row and
  note the gap. Do not fill, forward-fill, or guess.
- **Separate correlation from causation.** Always state explicitly when a finding
  is correlational. Never imply causation from observational data.
- **Surface confounders.** Identify and document the most plausible confounders even
  if you cannot control for them. Stratify where the data allows.
- **Small samples require explicit warnings.** Any statistical result with n < 30 per
  group must be flagged. Any result with n < 10 must be suppressed from headline output.
- **Effect size, not just p-value.** Always report Cohen's d or equivalent alongside
  significance tests.

---

## Code conventions

- **Python 3.10+.** Use `from __future__ import annotations` for forward references.
- **pandas for data, matplotlib for charts.** No seaborn. No plotly. No bokeh.
- **Type hints on all public functions.**
- **Logging via the standard logging module.** Never use print() for pipeline output.
- **Fail fast with actionable errors.** If a required input file is missing, raise a
  clear FileNotFoundError with the expected path and instructions for obtaining the data.
- **Never silently drop rows.** Log a warning, write dropped rows to unmatched_rows.csv
  or equivalent, and count them in the data quality report.
- **No hardcoded absolute paths.** Use Path(__file__).resolve().parent patterns.
- **No hardcoded credentials.** Use .env files (gitignored) and python-dotenv.

---

## Repository conventions

- **One project per subdirectory.** Each project is fully self-contained with its own
  README.md, requirements.txt, pyproject.toml, src/, scripts/, and data/ tree.
- **No cross-project imports.** Projects must not import from each other.
- **Data files are never committed** (except templates and clearly-labelled demo fixtures).
  Real input data goes in data/raw/ (gitignored). Generated outputs go in
  data/intermediate/, data/final/, outputs/ (all gitignored).
- **Every project README must include:**
  - The specific research question
  - Where each dataset came from (source, URL, retrieval method)
  - Known data quality issues and limitations
  - What conclusions can and cannot be drawn

---

## Adding a new project

1. Create a new subdirectory: mkdir <project-name>
2. Use turn-signal-crash-rate/ as a structural template
3. Add a row to the table in the root README.md
4. Add a project-level .gitignore for data/output paths
5. Start with the research question and data sourcing — write the README before
   writing any pipeline code

---

## Available tools and skills

### Firecrawl — web scraping and crawling

A self-hosted Firecrawl instance is available for this repository.

- **Base URL:** `https://FIRECRAWL_HOST`
- **Auth:** `FIRECRAWL_API_KEY` from `.env` (pass as `Authorization: Bearer <key>` header)
- **SDK:** `pip install firecrawl-py` then `from firecrawl import FirecrawlApp`

Capabilities:

| Endpoint | What it does |
|---|---|
| `POST /v1/scrape` | Scrape a single URL → clean markdown, HTML, or structured JSON |
| `POST /v1/crawl` | Crawl an entire site and return all pages as markdown |
| `POST /v1/search` | Web search returning full page content from results |
| `POST /v1/interact` | Scrape then click/fill forms/navigate dynamic content |

When to use Firecrawl:

- A data source exists on a website but has no downloadable CSV or API
- You need to scrape manufacturer spec pages, government databases, or product listings
- JavaScript rendering is required (SPAs, dynamically-loaded tables)
- You need to crawl an entire documentation site or spec archive

Rules for Firecrawl use:

- **Always store the raw scraped content** in `data/raw/` alongside the extracted data so the scrape is reproducible
- **Record the scrape date** — scraped data must have a `retrieved_at` field
- **Rate-limit politely** — add delays between requests, respect `robots.txt` unless instructed otherwise
- **Scraping is not fabrication** — extracted content is real data; document the source URL for every row
- Use a `.env` file (gitignored) for `FIRECRAWL_BASE_URL` and `FIRECRAWL_API_KEY`; add both to `.env.example`

Example usage:

```python
import os
from firecrawl import FirecrawlApp

app = FirecrawlApp(
    api_key=os.environ["FIRECRAWL_API_KEY"],
    api_url=os.environ.get("FIRECRAWL_BASE_URL", "https://FIRECRAWL_HOST"),
)
result = app.scrape_url("https://example.com/specs", formats=["markdown"])
```

---

## What Copilot must never do

- Fabricate data, create sample datasets, or fill gaps with plausible-looking values
- Commit real data files
- Present raw counts as findings without normalisation
- Silently drop rows
- Use print() instead of logging
- Hardcode absolute paths
- Use seaborn, plotly, or bokeh
- Imply causation from observational data
- Suppress small-sample warnings to make results look cleaner
