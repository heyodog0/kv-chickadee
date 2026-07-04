# analysis

Extraction + reanalysis of the chickadee barcode data. Analysis runs on a laptop from `events/`;
extraction needs MATLAB + cluster (one-time, done). Every run is logged in `RESULTS.md`.

## Use these
- `extract_events.m` — MATLAB; authors' `parseCacheActions` windows → `events/*.mat`.
- `analyze_events.py` — positive control: cache–retrieval correlation vs inter-site distance.
- `analyze_forward.py` — **main result**: event-specificity + split-half ceiling + per-bird + controls.
- `analyze_specificity_snr.py` — SNR robustness: raw vs whitened vs PC-residual readouts.
- `analyze_checks.py` — cue-triggered read: do checks reactivate the barcode (vs a visit place baseline)?

Run: `python analyze_forward.py` (numpy, scipy, scikit-learn).

## Legacy (earlier KV-thesis exploration; kept for history, don't use — wrong event window)
- `extract_barcodes.py` — first single-session extraction.
- `multi_session.py` — v1 multi-session (place-contaminated).
- `multi_session_v2.py` — v2, events from `seedChanges` (still wrong window).
- `barcode_isolated.py` — tried subtracting a place field (made overlap worse).
- `event_specificity.py` — earlier co-located test, pre-faithful-extraction.
- `analyze_crosstalk.py` — tested the ρ√N crosstalk law (not supported; abandoned).

## Cluster job wrappers
- `run_extract.sh`, `run_es.sh`, `run_iso.sh` — Slurm scripts that ran the above on FASRC.
