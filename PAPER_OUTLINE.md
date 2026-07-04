# Paper outline — chickadee barcode reanalysis

**Title:** Content-addressed, not key-addressed: barcode reactivation for co-located caches in the chickadee hippocampus
**Thesis:** For co-located caches, retrieval reactivates the shared *site* barcode, not the individual cache's. In key-value terms, the barcode acts as a content-addressable **index**, not a queryable **key** — the read is addressed by place/context, not a per-episode key.
**Type:** reanalysis of open data — reproduce a published result, then test a KV prediction in a new regime. A *test*, not a demonstration of KV. Not a discovery.

## Sections
1. **Intro** — key-value memory (Gershman, Fiete & Irie 2025) distinguishes *key-addressed* retrieval (arbitrary key, matched by similarity) from *content-addressed* retrieval (cued by context, pattern completion). Barcodes are index-like ("hash codes", Chettih 2024), and Fang et al. 2026 (same group) explicitly model them as content-addressable — *not* KV keys. But that is a modeling claim, untested where it bites: caches at one site share the content cue, so a content-addressed read should fail to individuate them — the regime the model excludes. We test it directly in the neural data.
2. **Methods** — authors' `parseCacheActions` windows; barcode = mean pop. vector; positive control; leave-one-out event-specificity; split-half ceiling; controls.
3. **Results** — (1) positive control reproduces (+0.089); (2) no episode-specificity (Δ=−0.007, p=0.87); (3) not SNR (ceiling +0.28); (4) honest scope (56% of sites co-located; CI upper = 51% of effect; IND106 outlier).
4. **Discussion** — the barcode is content-addressed, not a strict KV key (grounds Fang's own caveat + GFI's content-addressable case + cued, not free, recall, Smulders & Cheng 2025).
   - **KV contrast (fly mushroom body):** the mushroom body is the *key-addressed* end — a content-independent, near-orthogonal Kenyon-cell tag (the key) read out by learned MBON weights (Litwin-Kumar 2017; Dasgupta 2017). The chickadee barcode looks similarly sparse/key-like but works the opposite way (content-addressed). The two **bracket the KV addressing spectrum in the brain** — a clean way to say what the barcode is *not*.
   - Fills Fang's excluded regime; bears on capacity / directed forgetting.
5. **Limitations** — reanalysis; soft, between-bird-heterogeneous null; ceiling is temporally adjacent.

## Repo status (what's on its plate)
Done:
- [x] Extraction pipeline (`extract_events.m`) — 54 sessions → `events/`.
- [x] Positive control (`analyze_events.py`) — site-level reactivation reproduced.
- [x] Event-specificity + ceiling + controls (`analyze_forward.py`, `analyze_specificity_snr.py`).
- [x] Figures 1–2 (`out/analyze_events.png`, `out/analyze_forward.png`).

Legacy (earlier, abandoned KV-thesis exploration — prune):
`barcode_isolated.py`, `multi_session*.py`, `event_specificity.py`, `extract_barcodes.py`, `analyze_crosstalk.py`.

## Checklist (to do)
- [ ] Figure 3 — ceiling-vs-retrieval bracket.
- [ ] Figure 4 — per-bird Δ + equivalence CI + co-location prevalence.
- [ ] Draft Methods (most mechanical / reusable first).
- [ ] Draft Intro, Results, Discussion.
- [ ] One owned extension analysis (decoding read / per-bird / load-order).
- [ ] Prune legacy scripts; tidy repo.
- [ ] Verify references; format for arXiv (LaTeX/PDF).
