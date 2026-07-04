"""
First barcode analysis on one Grid Caching session.

Reimplements the authors' selectAndNormSpikes (README.m) in Python, extracts a
per-cache-event population vector ("barcode") by averaging normalized spikes over
each event window, then runs two KV-relevant diagnostics:

  (1) DISCRIMINABILITY  — can we decode which cache SITE an event is at, from the
      population vector? (key-like: high identity information)
  (2) PLACE vs BARCODE  — within-site vs across-site population-vector correlation.
      Pure place code => within-site >> across-site (same location, similar code).
      Barcode (key-like) => within-site correlation collapses toward across-site.

Run on the cluster:
  uv run --python 3.12 --with numpy --with scipy --with h5py --with scikit-learn \
      python extract_barcodes.py
"""
import os, glob, itertools, random
import numpy as np
import h5py
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

ROOT = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/Grid Caching Data"
SESS = sorted(d for d in glob.glob(ROOT + "/*_*") if os.path.isdir(d))
S = SESS[0]
print("session:", os.path.basename(S), "| total sessions:", len(SESS))

# ---- load aligned neural data (MATLAB v7.3 = HDF5; h5py returns transposed) ----
with h5py.File(os.path.join(S, "alignedSpikesAndPosture.mat"), "r") as h:
    ad = h["alignedData"]
    spks = np.asarray(ad["spks"], dtype=np.float32)     # (nUnits, nBins)
    wv = ad["wvStruct"]
    contam   = np.asarray(wv["contam"]).ravel()
    meanRate = np.asarray(wv["meanRate"]).ravel()
    idx      = np.asarray(wv["idx"]).ravel()
nUnits, nBins = spks.shape
print("spks:", spks.shape, "| min/mean/max: %.2f/%.3f/%.1f" % (spks.min(), spks.mean(), spks.max()))

# ---- selectAndNormSpikes (useMUA=True) ----
maxContam, minRate, stdReg, bsWin = 1.0, 0.02, 1e-2, 30 * 60**2 + 1  # 108001 bins ~ 30 min
good = np.where((contam < maxContam) & (meanRate > minRate) & np.isin(idx, [1, 2, 3]))[0]
print("good units:", len(good), "of", nUnits)
Sg = spks[good, :]                                   # (nGood, nBins)
Sn = Sg / (stdReg + Sg.std(axis=1, keepdims=True))
Sn = Sn - uniform_filter1d(Sn, size=bsWin, axis=1, mode="nearest")   # subtract slow baseline
Sn = Sn.astype(np.float32)

# ---- events from annotatedSeeds (MATLAB v7) ----
a = sio.loadmat(os.path.join(S, "annotatedSeeds.mat"), struct_as_record=False, squeeze_me=True)["annotatedSeeds"]
newT = np.atleast_1d(np.asarray(a.newCacheTimes, dtype=float))
endT = np.atleast_1d(np.asarray(a.endCacheTimes, dtype=float))
toS  = np.atleast_2d(np.asarray(a.toCache_site, dtype=float))     # (nEvents, 128)
frS  = np.atleast_2d(np.asarray(a.fromCache_site, dtype=float))
nE = len(newT)
print("events:", nE, "| time range: %.0f..%.0f (nBins=%d)" % (newT.min(), newT.max(), nBins))
print("toCache_site rowsum unique(first 8):", np.unique(np.abs(toS).sum(1))[:8])

# site label per event = the cache site involved; keep only unambiguous single-site events
involve = np.abs(toS) + np.abs(frS)
site = np.argmax(involve, axis=1)
clean = involve.sum(1) == 1                 # exactly one site touched -> unambiguous label
print("clean single-site events:", int(clean.sum()), "of", nE)
f0 = np.clip((newT - 1).astype(int), 0, nBins - 1)
f1 = np.clip(endT.astype(int),        1, nBins)

# ---- barcode = mean normalized population vector over the event window ----
B = np.zeros((nE, len(good)), dtype=np.float32)
durs = np.zeros(nE, dtype=int)
for e in range(nE):
    lo, hi = f0[e], max(f0[e] + 1, f1[e])
    B[e] = Sn[:, lo:hi].mean(axis=1)
    durs[e] = hi - lo
print("barcodes:", B.shape, "| median event dur (bins): %d (~%.2fs)" % (np.median(durs), np.median(durs)/60))

# ---- (1) discriminability: decode cache SITE from the population vector ----
from collections import Counter
cnt = Counter(site[clean].tolist())
keep = np.array([s for s, c in cnt.items() if c >= 5])
mask = clean & np.isin(site, keep)
Bk, yk = B[mask], site[mask]
Bn = normalize(Bk)                                   # L2 -> NearestCentroid ~ cosine
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
for tr, te in skf.split(Bn, yk):
    clf = NearestCentroid().fit(Bn[tr], yk[tr])
    accs.append((clf.predict(Bn[te]) == yk[te]).mean())
chance = 1.0 / len(keep)
print("\n(1) SITE DECODING  sites=%d events=%d" % (len(keep), mask.sum()))
print("    accuracy = %.3f ± %.3f   (chance = %.3f, x%.1f)" %
      (np.mean(accs), np.std(accs), chance, np.mean(accs) / chance))

# ---- (2) place vs barcode: within-site vs across-site correlation ----
def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
random.seed(0)
ev = np.where(mask)[0]
pairs = list(itertools.combinations(range(len(ev)), 2))
random.shuffle(pairs); pairs = pairs[:8000]
win, acr = [], []
for i, j in pairs:
    c = corr(B[ev[i]], B[ev[j]])
    (win if site[ev[i]] == site[ev[j]] else acr).append(c)
print("\n(2) PLACE vs BARCODE  (population-vector correlation)")
print("    within-site: %.3f (n=%d)" % (np.mean(win), len(win)))
print("    across-site: %.3f (n=%d)" % (np.mean(acr), len(acr)))
print("    within − across = %.3f" % (np.mean(win) - np.mean(acr)))
print("    [pure place => within >> across ; barcode/key-like => within ≈ across, both low]")

# ---- (3) variance split: site-explained (place/value) vs event-unique (barcode/key) ----
grand = Bk.mean(0)
cent = {s: Bk[yk == s].mean(0) for s in keep}
C = np.vstack([cent[s] for s in yk])                 # per-event site centroid
total   = float(((Bk - grand) ** 2).sum(1).mean())
between = float(((C  - grand) ** 2).sum(1).mean())   # place/content (survives averaging)
within  = float(((Bk - C)     ** 2).sum(1).mean())   # event-unique (barcode)
print("\n(3) VARIANCE SPLIT of the per-event population vector")
print("    site-explained (place / value): %.1f%%" % (100 * between / total))
print("    event-unique   (barcode / key): %.1f%%" % (100 * within  / total))
print("    [key-like memory => most variance is event-unique, on top of a smaller shared place code]")
