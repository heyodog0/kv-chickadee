# archive — abandoned KV-thesis exploration

Kept for history only. **Do not use.** All of these predate the faithful extraction and use the
wrong event window (the seed-record interval, not the authors' `parseCacheActions` windows), so
their results are place-contaminated or superseded. The current pipeline is in `../`.

- `extract_barcodes.py` — first single-session extraction.
- `multi_session.py` — v1 multi-session (place-contaminated).
- `multi_session_v2.py` — v2, events from `seedChanges` (still wrong window).
- `barcode_isolated.py` — tried subtracting a place field (made overlap worse).
- `event_specificity.py` — earlier co-located test, pre-faithful-extraction.
- `analyze_crosstalk.py` — tested the ρ√N crosstalk law (not supported; abandoned).
- `run_es.sh`, `run_iso.sh` — Slurm wrappers for two of the above.
