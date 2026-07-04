# kv-chickadee

Reanalysis of the Chettih et al. 2024 chickadee barcode data: when caches share a site, does a
retrieval reactivate *its own* cache's barcode, or just the shared site barcode?

**Finding — site-level, not episode-level.** The barcode is a content-addressable *index*, not a
key-value *key*. (Reanalysis: reproduces the published control, then tests the co-located regime
that the original paper and the Fang et al. 2026 model both skip. Not a discovery.)

## Results (54 sessions, 5 birds)
- Reactivation is real: retrieval ↔ own cache ≫ random cache (+0.082, p=1.7e-10).
- But not episode-specific: among co-located caches, own vs other-same-site Δ=−0.007 (p=0.87).
- Not SNR: the split-half ceiling shows structure is resolvable at encoding (Δ=+0.28) — it just doesn't transfer to retrieval.
- Cue-triggered: checks reactivate ≈ retrievals, ≫ visits (check−visit +0.25, p=3.5e-18) — reactivation follows site engagement, not the retrieval act.
- Scope: 56% of sites co-located; effect bounded small (CI upper ≈24% of reactivation); bird IND106 trends positive.

→ content-addressed, cue-triggered, site granularity — consistent with Fang 2026's caveat and Smulders & Cheng 2025 (cued recall). Contrast: the fly mushroom body is key-addressed (KC key → learned MBON readout); the two bracket the KV addressing spectrum.

## Run (laptop, from `events/`)
`python analysis/analyze_forward.py` — needs numpy, scipy, scikit-learn.
- `analyze_forward.py` — main result (event-specificity + ceiling + per-bird + controls).
- `analyze_events.py` — positive control (corr vs inter-site distance).
- `analyze_specificity_snr.py` — SNR robustness · `analyze_checks.py` — cue-triggered test.
- `extract_events.m` — MATLAB extraction → `events/` (one-time, cluster, done).
- `analysis/archive/` — abandoned KV-thesis scripts (wrong window; ignore).

## Data
Chettih et al. 2024, Dryad `doi:10.5061/dryad.7h44j101z` (CC0). Raw 38 GB not in repo; the
extracted per-event vectors in `events/` (112 MB) are all the analysis needs.
