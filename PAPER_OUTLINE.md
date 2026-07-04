# Paper outline — chickadee barcode reanalysis

**Title:** Content-addressed, not key-addressed: barcode reactivation for co-located caches in the chickadee hippocampus
**Thesis:** For co-located caches, retrieval reactivates the shared *site* barcode, not the individual cache's — a content-addressable index, not a queryable key.
**Type:** reanalysis of open data. A test of a KV prediction, not a discovery.

## Sections
1. **Intro** — KV memory (GFI 2025): key- vs content-addressed reads. Barcodes are index-like; Fang 2026 model them as content-addressable, not KV keys — untested for co-located caches (shared cue). We test it.
2. **Methods** — authors' `parseCacheActions` windows; barcode = mean pop. vector; event-specific reactivation; leave-one-out site control; split-half ceiling.
3. **Results** — (1) event-specific reactivation is real: retrieval ↔ own cache ≫ random cache (+0.082, p=1.7e-10); (2) but NOT among co-located caches (Δ=−0.007, p=0.87); (3) not SNR (ceiling +0.28); (4) cue-triggered: checks reactivate ≈ retrievals, ≫ visits (check−visit +0.25, p=3.5e-18); (5) common regime (56%), effect bounded small, IND106 outlier.
4. **Discussion** — content-addressed and cue-triggered (checks reactivate, not just retrievals), not a KV key (grounds Fang/GFI + cued recall, Smulders & Cheng 2025). Contrast: fly mushroom body = key-addressed (KC key → learned MBON readout). The two bracket the KV addressing spectrum.
5. **Limitations** — reanalysis; soft, heterogeneous null; ceiling temporally adjacent.

## Repo status
Done: extraction (`extract_events.m`) → `events/`; event-specific reactivation + co-located null + ceiling + controls (`analyze_forward.py`); SNR check (`analyze_specificity_snr.py`); distance-curve figure (`analyze_events.py`).
Archived: earlier KV-thesis exploration moved to `analysis/archive/`.

## Checklist
- [ ] Figures: ceiling-vs-retrieval bracket; per-bird + CI + prevalence.
- [ ] Draft Methods → Intro/Results/Discussion.
- [ ] One owned extension (decoding read / per-bird / load-order).
- [x] Check-reactivation test — cue-triggered (checks reactivate), `analyze_checks.py`.
- [ ] Verify refs; format for arXiv.
