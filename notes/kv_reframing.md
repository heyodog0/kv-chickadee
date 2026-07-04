# KV reframing (current, honest)

Key-value memory (Gershman, Fiete & Irie 2025): memory = (key, value) pairs; a read matches a
query to a key to fetch a value. Two addressing modes:
- **Key-addressed:** an arbitrary key you query to select a *specific* item, even among items
  that share content.
- **Content-addressed (index):** you query with content/context; pattern completion fills the
  rest; items sharing that context can't be individuated.

The authors already place the barcode on the content-addressed side: Fang et al. 2026 (same
group as Chettih) explicitly state barcodes are *not* KV keys — the memory is content-addressable
— and Chettih 2024 calls them "hash codes" (index-like). But that is a modeling claim. It is
untested where it bites: caches at one site share the content cue, so a content-addressed read
should fail to tell them apart.

**Result:** it does fail — retrieval individuates the *site*, not the co-located *episode*. This
is the direct neural test of the content-addressable characterization, in the co-located regime
the model excludes. Consistent with Fang 2026's caveat and Smulders & Cheng 2025 (cued, not free,
recall).

**Abandoned:** the "barcode = cleanest KV key + ρ√N crosstalk law on neural keys" thesis did not
survive — the crosstalk law is textbook (Hopfield/Willshaw), and the fly mushroom body is the
cleaner key-addressed instance. See `../PAPER_OUTLINE.md`.
