# kv-chickadee

Reanalysis of the Chettih et al. 2024 chickadee barcode data, asking one question they and the
Fang et al. 2026 model left untested: **when caches share a site, does retrieval reactivate its
own cache's barcode, or just the shared site barcode?**

**Finding:** site-level, not episode-level. The barcode acts as a content-addressable *index*,
not a queryable key-value *key*. A reanalysis (reproduces the published control, then tests the
co-located regime) — not a discovery.

## Layout
- `PAPER_OUTLINE.md` — paper skeleton, repo status, checklist.
- `STUDENT_TASKS.md` — onboarding tasks.
- `analysis/` — extraction + analyses; `RESULTS.md` has the findings.
- `data/DATA_LOCATION.md` — where the data is.

## Run
Extraction (MATLAB + cluster) is done. Analysis runs on a laptop from the included `events/` (~112 MB):
```
python analysis/analyze_forward.py   # numpy, scipy, scikit-learn
```

## Data
Chettih et al. 2024, Dryad `doi:10.5061/dryad.7h44j101z` (raw 38 GB not in repo).
