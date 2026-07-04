"""
Multi-session barcode analysis across all 54 Grid Caching sessions.

Neurons differ per session, so every metric is computed WITHIN a session and the
metrics are pooled across sessions (each session is one replication).

Events (from annotatedSeeds, aligned to newCacheTimes):
  cache     = seed placed  -> site = argmax(toCache_site),   rows with toCache_site.sum>0
  retrieval = seed removed -> site = argmax(fromCache_site),  rows with fromCache_site.sum>0
barcode(event) = mean normalized population vector over [newCacheTimes, endCacheTimes].

Per-session metrics (on cache events):
  (1) DISCRIMINABILITY  site-decoding accuracy (5-fold NearestCentroid, cosine) vs chance
  (2) VARIANCE SPLIT    between-site (place/value) vs within-site (barcode/key) fraction
  (3) DECORRELATION     within-site vs across-site barcode correlation
  (4) REACTIVATION      does a retrieval barcode match the SAME-site cache centroid more
                        than other-site cache centroids? (the KV read: key reactivates)
"""
import os, glob, json, itertools, random
import numpy as np
import h5py
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/Grid Caching Data"
OUT  = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/out"
os.makedirs(OUT, exist_ok=True)
SESS = sorted(d for d in glob.glob(ROOT + "/*_*") if os.path.isdir(d))
MINEV = 4   # min events per site to include that site

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

def load_norm(S):
    with h5py.File(os.path.join(S, "alignedSpikesAndPosture.mat"), "r") as h:
        ad = h["alignedData"]
        spks = np.asarray(ad["spks"], dtype=np.float32)         # (nUnits, nBins)
        wv = ad["wvStruct"]
        contam = np.asarray(wv["contam"]).ravel()
        meanRate = np.asarray(wv["meanRate"]).ravel()
        idx = np.asarray(wv["idx"]).ravel()
    good = np.where((contam < 1.0) & (meanRate > 0.02) & np.isin(idx, [1, 2, 3]))[0]
    Sg = spks[good, :]
    Sn = Sg / (1e-2 + Sg.std(axis=1, keepdims=True))
    Sn = (Sn - uniform_filter1d(Sn, size=30*60**2+1, axis=1, mode="nearest")).astype(np.float32)
    a = sio.loadmat(os.path.join(S, "annotatedSeeds.mat"), struct_as_record=False, squeeze_me=True)["annotatedSeeds"]
    newT = np.atleast_1d(np.asarray(a.newCacheTimes, dtype=float))
    endT = np.atleast_1d(np.asarray(a.endCacheTimes, dtype=float))
    toS  = np.atleast_2d(np.asarray(a.toCache_site, dtype=float))
    frS  = np.atleast_2d(np.asarray(a.fromCache_site, dtype=float))
    cacheLoc = np.atleast_2d(np.asarray(a.cacheLoc, dtype=float))   # (128, 2) site coords
    return Sn, newT, endT, toS, frS, cacheLoc, len(good)

def barcodes(Sn, newT, endT, rows):
    nb = Sn.shape[1]
    f0 = np.clip((newT - 1).astype(int), 0, nb - 1)
    f1 = np.clip(endT.astype(int), 1, nb)
    B = np.zeros((len(rows), Sn.shape[0]), dtype=np.float32)
    for i, e in enumerate(rows):
        lo, hi = f0[e], max(f0[e] + 1, f1[e])
        B[i] = Sn[:, lo:hi].mean(axis=1)
    return B

def analyze(S):
    Sn, newT, endT, toS, frS, cacheLoc, nGood = load_norm(S)
    cache_rows = np.where(toS.sum(1) > 0)[0]; cache_site = np.argmax(toS, 1)[cache_rows]
    retr_rows  = np.where(frS.sum(1) > 0)[0]; retr_site  = np.argmax(frS, 1)[retr_rows]
    if len(cache_rows) < 8:
        return None
    Bc = barcodes(Sn, newT, endT, cache_rows)
    # sites with enough cache events
    from collections import Counter
    cnt = Counter(cache_site.tolist())
    keep = np.array([s for s, c in cnt.items() if c >= MINEV])
    if len(keep) < 2:
        return None
    m = np.isin(cache_site, keep)
    X, y = Bc[m], cache_site[m]
    res = {"session": os.path.basename(S), "nGood": int(nGood),
           "n_cache": int(len(cache_rows)), "n_retr": int(len(retr_rows)),
           "n_sites": int(len(keep)), "n_used": int(m.sum())}
    # (1) decoding
    Xn = normalize(X)
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, y):
        accs.append((NearestCentroid().fit(Xn[tr], y[tr]).predict(Xn[te]) == y[te]).mean())
    res["acc"] = float(np.mean(accs)); res["chance"] = 1.0 / len(keep)
    # (2) variance split
    grand = X.mean(0); cent = {s: X[y == s].mean(0) for s in keep}
    C = np.vstack([cent[s] for s in y])
    tot = float(((X - grand) ** 2).sum(1).mean())
    res["frac_barcode"] = float(((X - C) ** 2).sum(1).mean() / tot)   # within-site
    res["frac_place"]   = float(((C - grand) ** 2).sum(1).mean() / tot)  # between-site
    # (3) decorrelation
    random.seed(0); ev = np.arange(len(X))
    pairs = list(itertools.combinations(ev, 2)); random.shuffle(pairs); pairs = pairs[:6000]
    win, acr, dc = [], [], []
    for i, j in pairs:
        c = corr(X[i], X[j])
        if y[i] == y[j]:
            win.append(c)
        else:
            acr.append(c)
            dc.append((float(np.linalg.norm(cacheLoc[y[i]] - cacheLoc[y[j]])), c))
    res["corr_within"] = float(np.mean(win)) if win else np.nan
    res["corr_across"] = float(np.mean(acr)) if acr else np.nan
    res["dc"] = dc[:2000]   # (inter-site distance, barcode corr) for the place control
    # (4) reactivation: retrieval barcode vs same-site vs other-site cache centroid
    if len(retr_rows) >= 4 and len(keep) >= 2:
        Br = barcodes(Sn, newT, endT, retr_rows)
        same, other = [], []
        for i, s in enumerate(retr_site):
            if s not in cent: continue
            same.append(corr(Br[i], cent[s]))
            other.append(np.mean([corr(Br[i], cent[s2]) for s2 in keep if s2 != s]))
        if same:
            res["react_same"] = float(np.mean(same)); res["react_other"] = float(np.mean(other))
            res["react_n"] = len(same)
    return res

results = []
for i, S in enumerate(SESS):
    try:
        r = analyze(S)
        if r: results.append(r); print("[%2d/%d] %-22s sites=%d acc=%.2f(ch%.2f) barcode%%=%.0f react=%s"
              % (i+1, len(SESS), r["session"], r["n_sites"], r["acc"], r["chance"],
                 100*r["frac_barcode"], ("%.2f/%.2f" % (r.get("react_same",np.nan), r.get("react_other",np.nan)))))
        else:
            print("[%2d/%d] %-22s skipped (too few events/sites)" % (i+1, len(SESS), os.path.basename(S)))
    except Exception as e:
        print("[%2d/%d] %-22s ERROR %r" % (i+1, len(SESS), os.path.basename(S), e))

json.dump([{k: v for k, v in r.items() if k != "dc"} for r in results],
          open(os.path.join(OUT, "multi_session.json"), "w"), indent=1)

# ---- pooled summary ----
def arr(k): return np.array([r[k] for r in results if k in r and r[k] == r[k]])
acc, ch = arr("acc"), arr("chance")
fb, fp = arr("frac_barcode"), arr("frac_place")
cw, ca = arr("corr_within"), arr("corr_across")
rs, ro = arr("react_same"), arr("react_other")
print("\n==================  POOLED OVER %d SESSIONS  ==================" % len(results))
print("(1) site decoding : acc %.3f ± %.3f   chance %.3f   (%d/%d sessions above chance)"
      % (acc.mean(), acc.std(), ch.mean(), int((acc > ch).sum()), len(acc)))
print("(2) variance split: barcode/event-unique %.0f%% ± %.0f   place/site %.0f%%"
      % (100*fb.mean(), 100*fb.std(), 100*fp.mean()))
print("(3) decorrelation : within-site %.3f   across-site %.3f   (within−across %.3f)"
      % (cw.mean(), ca.mean(), (cw - ca).mean()))
if len(rs):
    from scipy.stats import wilcoxon
    d = rs - ro
    try: p = wilcoxon(d).pvalue
    except Exception: p = np.nan
    print("(4) reactivation  : retrieval↔cache  same-site %.3f  other-site %.3f  (Δ=%.3f, %d/%d sessions same>other, p=%.1e)"
          % (rs.mean(), ro.mean(), d.mean(), int((d > 0).sum()), len(d), p))

# (5) PLACE CONTROL: does across-site barcode correlation decay with physical distance?
#     place code => corr decays with inter-site distance ; arbitrary barcode => flat with distance
DC = np.array([pt for r in results if "dc" in r for pt in r["dc"]])   # (Npairs, 2): distance, corr
xb = yb = None
if len(DC) > 100:
    from scipy.stats import pearsonr
    dd, cc = DC[:, 0], DC[:, 1]
    pr, pp = pearsonr(dd, cc)
    q = np.quantile(dd, [0, 1/3, 2/3, 1.0])
    print("(5) place control : across-site barcode corr vs inter-site distance (n=%d pairs)" % len(DC))
    for lo_, hi_ in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3])]:
        mk = (dd >= lo_) & (dd <= hi_)
        print("      dist %.2f–%.2f : corr %.3f (n=%d)" % (lo_, hi_, cc[mk].mean(), int(mk.sum())))
    print("      Pearson r(distance, corr) = %.3f (p=%.1e)   [~0 => arbitrary/barcode ; strongly<0 => place]" % (pr, pp))
    order = np.argsort(dd); k = max(1, len(dd) // 20)
    xb = [dd[order][i:i+k].mean() for i in range(0, len(dd), k)]
    yb = [cc[order][i:i+k].mean() for i in range(0, len(dd), k)]

# ---- figure ----
fig, ax = plt.subplots(1, 5, figsize=(18.5, 3.4))
ax[0].scatter(ch, acc, s=18); lim=[0, max(acc.max(), ch.max())*1.1]
ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlabel("chance"); ax[0].set_ylabel("decode acc"); ax[0].set_title("(1) discriminability")
ax[1].hist(100*fb, bins=15); ax[1].set_xlabel("event-unique (barcode) % of variance"); ax[1].set_title("(2) barcode dominance")
ax[2].scatter(ca, cw, s=18); L=[min(ca.min(),cw.min()), max(ca.max(),cw.max())]
ax[2].plot(L, L, "k--", lw=1); ax[2].set_xlabel("across-site corr"); ax[2].set_ylabel("within-site corr"); ax[2].set_title("(3) decorrelation")
if len(rs):
    ax[3].scatter(ro, rs, s=18); L2=[min(ro.min(),rs.min()), max(ro.max(),rs.max())]
    ax[3].plot(L2, L2, "k--", lw=1); ax[3].set_xlabel("other-site"); ax[3].set_ylabel("same-site"); ax[3].set_title("(4) reactivation (read)")
if xb is not None:
    ax[4].plot(xb, yb, ".-"); ax[4].axhline(0, color="k", lw=.6)
    ax[4].set_xlabel("inter-site distance"); ax[4].set_ylabel("across-site barcode corr"); ax[4].set_title("(5) place control")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "multi_session.png"), dpi=110)
print("\nsaved:", os.path.join(OUT, "multi_session.json"), "and multi_session.png")
