# data-scrapbook

A collection of self-contained data analysis projects.
Each project lives in its own subdirectory with its own README, dependencies, and pipeline.

---

## Projects

| Directory | Description |
|---|---|
| [`turn-signal-crash-rate/`](turn-signal-crash-rate/) | Do US vehicles with non-amber rear turn signals have higher fatal crash rates? End-to-end pipeline: NHTSA FARS data → exposure normalisation → statistical analysis → charts |

---

## Structure

Each project is self-contained:

```
<project-name>/
├── README.md          ← project-specific documentation
├── requirements.txt   ← project-specific dependencies
├── pyproject.toml
├── scripts/           ← data acquisition scripts
├── src/               ← pipeline source code
├── data/
│   ├── raw/           ← input data (gitignored except templates & demo)
│   ├── intermediate/  ← auto-generated (gitignored)
│   └── final/         ← auto-generated (gitignored)
└── outputs/
    ├── charts/        ← auto-generated PNG charts (gitignored)
    └── tables/        ← auto-generated CSV summaries (gitignored)
```

## License

MIT — see individual project directories for details.
