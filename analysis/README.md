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

## Cluster job wrapper
- `run_extract.sh` — Slurm script that runs `extract_events.m` on FASRC.

## archive/
Abandoned KV-thesis exploration (wrong event window; superseded). Kept for history, don't use —
see `archive/README.md`.
