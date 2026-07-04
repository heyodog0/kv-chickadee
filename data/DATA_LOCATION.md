# Data location

**Source:** Chettih, Mackevicius, Hale & Aronov 2024, *Cell*. Dryad `doi:10.5061/dryad.7h44j101z`
(38 GB; not in this repo). 54 sessions, 5 birds.

**Cluster:** `/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/` (FASRC scratch,
~90-day purge; access via the `fasrc` SSH function). Extracted per-event vectors are in
`events/*.mat` (113 MB) — this is all the analysis needs.

**Structure (`Grid Caching Data/`):** `action parsing code/` (authors' MATLAB) + one folder per
session with `alignedSpikesAndPosture.mat` (spikes + posture) and `annotatedSeeds.mat`
(cache/retrieval annotations).

**Key fields:** `alignedData.spks` (100 units × ~660k bins; v7.3/HDF5 → h5py, transposed),
`alignedData.smPts` (posture); `annotatedSeeds` (v7 → scipy.io): `seedChanges`, `cacheNum`,
`newCacheTimes`/`endCacheTimes`, `countData.siteNum`.

**Extraction:** `analysis/extract_events.m` (MATLAB R2024b) runs `parseCacheActions` → `events/`.
One-time, done. Persist anything long-lived to `/n/holylabs/LABS/gershman_lab/`.
