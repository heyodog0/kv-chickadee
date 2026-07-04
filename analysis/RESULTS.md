# Results

Reanalysis of Chettih 2024 (54 sessions, 5 birds), using the authors' `parseCacheActions` event
windows. Scripts: `analyze_events.py`, `analyze_forward.py`.

**Positive control (event-specific reactivation).** A retrieval matches its *own* cache far more
than a random different-site cache (0.240 vs 0.158; +0.082, p=1.7e-10) — real reactivation.
Correlating to the same-site cache *mean* instead measures place (event-unique barcodes average
out; visits score as high as retrievals), so we do not use it as the control.

**Main result — site-specific, not episode-specific.** Among co-located caches (leave-one-out
site template removed): retrieval vs own cache 0.281 vs other-same-site 0.288 (Δ=−0.007, 27/54,
p=0.87); read accuracy 0.277 vs chance 0.268. No episode-specific reactivation.

**Not an SNR floor.** Split-half within a cache: own half matches its own other half far above
other same-site caches (Δ=+0.28, 54/54, p=1.6e-10); cross-site power control +0.082 (p=1.7e-10).
Event structure is resolvable at encoding; it just doesn't transfer to retrieval.

**Honest scope.** Co-location common (56% of sites ≥2 caches). Effect bounded small but not
negligible (95% CI [−0.035, +0.020]; upper ≈ 24% of the +0.082 reactivation effect). One bird (IND106) trends
positive (+0.039). Temporal-proximity control negligible. Barcode overlap ρ=0.36 (within-site 0.45).

**Cue-triggered read (checks reactivate, no retrieval needed).** At single-cache sites, while a
cache is stored, checks re-express the cache-time population pattern about as much as retrievals
(check 0.275 vs retrieval 0.287) and far above the visit place baseline (0.025): check−visit
+0.250, 108/119 caches, p=3.5e-18 (`analyze_checks.py`). Reactivation is triggered by site
engagement, not the retrieval act. (Place-controlled by visits; the remaining alternative to a
barcode is a shared site-manipulation state, and episode-specificity is the null above.)

**Interpretation.** Content-addressable index / cued recall, not a queryable key-value key —
consistent with Smulders & Cheng 2025 and Fang 2026's caveat; fills the co-located regime both left.

**Note on the pipeline.** Earlier scripts used the seed-record window (not the authors' event
windows) and found a spurious place-driven effect; naive place-subtraction made it worse. Fixed
by using `parseCacheActions`.
