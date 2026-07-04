"""
Analyze faithfully-extracted events (from extract_events.m, authors' parseCacheActions).

(1) POSITIVE CONTROL — reproduce Chettih 2024 Fig 1E/F (and Fang 2026 Fig 3D):
    population-vector correlation vs inter-site distance for
      visit-visit  -> smooth place decay
      cache-retrieval -> smooth decay PLUS a sharp same-site boost = barcode reactivation.
    If we recover the same-site boost, the extraction is validated.

(2) EVENT-SPECIFICITY IN THE CO-LOCATED-CACHE GAP (the un-scooped test):
    among caches at the SAME site, does a retrieval reactivate ITS OWN cache
    (matched cacheNum) above other same-site caches?  Leave-one-out site-template
    removal. own>other => KV key-addressing / pattern separation among co-located
    memories; own==other => content-addressable index (place-cued blend).

Run on cluster: ./.venv/bin/python analyze_events.py
"""
import os, glob, json, itertools, random
import numpy as np
import scipy.io as sio
from scipy.stats import wilcoxon, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EV  = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/events"
OUT = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/out"
os.makedirs(OUT, exist_ok=True)
FILES = sorted(glob.glob(EV + "/*.mat"))

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else 0.0

def clean(V, site, loc):
    V = np.atleast_2d(np.asarray(V, dtype=float))
    site = np.asarray(site, dtype=float).ravel()
    loc = np.atleast_2d(np.asarray(loc, dtype=float))
    if V.shape[0] != site.shape[0] and V.shape[1] == site.shape[0]:
        V = V.T
    ok = np.isfinite(V).all(1) & np.isfinite(site) & np.isfinite(loc).all(1)
    return V[ok], site[ok], loc[ok], ok

# distance bins in normalized arena units (15 in = 1 unit); adjacent sites ~small.
DBINS = np.array([0.0, 1e-6, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0, 1.5, 10.0])

def dist_curve(Va, la, Vb, lb, same_pair, maxpairs=6000):
    """mean corr in distance bins for pairs (a,b); same_pair excludes identical index when a is b."""
    ia = np.arange(len(Va)); ib = np.arange(len(Vb))
    pairs = [(i, j) for i in ia for j in ib if (not same_pair) or i < j]
    random.shuffle(pairs); pairs = pairs[:maxpairs]
    d = np.array([np.linalg.norm(la[i] - lb[j]) for i, j in pairs])
    c = np.array([corr(Va[i], Vb[j]) for i, j in pairs])
    binidx = np.digitize(d, DBINS) - 1
    curve = np.full(len(DBINS) - 1, np.nan)
    for b in range(len(DBINS) - 1):
        m = binidx == b
        if m.sum() >= 5: curve[b] = c[m].mean()
    return curve

random.seed(0)
vv_curves, cr_curves = [], []
rho_all, rho_within = [], []
es_rows = []          # per-session: own, other-same-site
es_load = []          # per-retrieval: (N competing same-site caches, own-is-argmax hit, chance)
summ = []

for f in FILES:
    S = os.path.basename(f)[:-4]
    m = sio.loadmat(f)
    visV, visSite, visLoc, _ = clean(m["visV"], m["visSite"], m["visLoc"])
    cacheV, cacheSite, cacheLoc, cok = clean(m["cacheV"], m["cacheSite"], m["cacheLoc"])
    retV, retSite, retLoc, rok = clean(m["retV"], m["retSite"], m["retLoc"])
    cnum_cache = np.asarray(m["cnum_cache"], dtype=float).ravel()[cok] if m["cnum_cache"].size else np.array([])
    cnum_ret   = np.asarray(m["cnum_ret"], dtype=float).ravel()[rok] if m["cnum_ret"].size else np.array([])

    # (1) positive control curves
    if len(visV) >= 5:
        vv_curves.append(dist_curve(visV, visLoc, visV, visLoc, same_pair=True))
    if len(cacheV) >= 3 and len(retV) >= 3:
        cr_curves.append(dist_curve(cacheV, cacheLoc, retV, retLoc, same_pair=False))

    # rho among caches (overall and within-site)
    if len(cacheV) >= 3:
        G = np.corrcoef(cacheV); iu = np.triu_indices(len(cacheV), 1)
        rho_all.append(float(np.mean(np.abs(G[iu]))))
        wsame = [abs(G[i, j]) for i, j in zip(*iu) if cacheSite[i] == cacheSite[j]]
        if wsame: rho_within.append(float(np.mean(wsame)))

    # (2) event-specificity among co-located caches (leave-one-out site template)
    own_s, oth_s = [], []
    for i in range(len(retV)):
        if cnum_ret.size == 0 or cnum_ret[i] <= 0: continue
        cand = np.where(cacheSite == retSite[i])[0]
        if len(cand) < 2: continue
        js = cand[cnum_cache[cand] == cnum_ret[i]]
        if len(js) == 0: continue
        j = js[0]
        scores = {}
        for k in cand:
            others = cand[cand != k]
            T = cacheV[others].mean(0)
            scores[k] = corr(retV[i] - T, cacheV[k] - T)
        own = scores[j]; oth = np.mean([scores[k] for k in cand if k != j])
        own_s.append(own); oth_s.append(oth)
        best = max(scores, key=scores.get)
        es_load.append((len(cand), int(best == j), 1.0 / len(cand)))
    if own_s:
        es_rows.append((np.mean(own_s), np.mean(oth_s), len(own_s)))

    summ.append({"session": S, "n_vis": int(len(visV)), "n_cache": int(len(cacheV)),
                 "n_retr": int(len(retV))})
    print("%-22s vis=%d cache=%d retr=%d  es_n=%d" % (S, len(visV), len(cacheV), len(retV),
          len(own_s) if own_s else 0))

json.dump(summ, open(os.path.join(OUT, "events_summary.json"), "w"), indent=1)

# ---- (1) positive control summary ----
vv = np.nanmean(np.vstack(vv_curves), 0) if vv_curves else None
cr = np.nanmean(np.vstack(cr_curves), 0) if cr_curves else None
centers = [ "same", "~0.1","~0.2","~0.4","~0.5","~0.7","~0.9","~1.2",">1.5"]
print("\n================  POSITIVE CONTROL: corr vs inter-site distance  ================")
print("bin:        " + "  ".join("%6s" % c for c in centers))
if vv is not None: print("visit-visit " + "  ".join(("%6.3f" % v) if v==v else "   nan" for v in vv))
if cr is not None: print("cache-retr  " + "  ".join(("%6.3f" % v) if v==v else "   nan" for v in cr))
if vv is not None and cr is not None:
    print("BARCODE REACTIVATION (same-site): cache-retr %.3f vs visit-visit %.3f (excess %+.3f)"
          % (cr[0], vv[0], cr[0]-vv[0]))
    print("  cache-retr same-site %.3f vs cache-retr different-site %.3f (boost %+.3f)"
          % (cr[0], np.nanmean(cr[2:]), cr[0]-np.nanmean(cr[2:])))

# ---- rho ----
if rho_all:
    print("\nbarcode overlap rho: all-pairs %.3f ± %.3f | within-site %.3f ± %.3f"
          % (np.mean(rho_all), np.std(rho_all),
             np.mean(rho_within) if rho_within else np.nan,
             np.std(rho_within) if rho_within else np.nan))

# ---- (2) event-specificity ----
print("\n================  EVENT-SPECIFICITY (co-located caches, site-template removed)  ================")
if es_rows:
    ER = np.array([(o, t) for (o, t, n) in es_rows])
    d = ER[:,0] - ER[:,1]; p = wilcoxon(d).pvalue if len(d) > 5 else np.nan
    print("own-cache %.4f  other-same-site %.4f  (Δ=%+.4f, %d/%d own>other, p=%.1e)"
          % (ER[:,0].mean(), ER[:,1].mean(), d.mean(), int((d>0).sum()), len(d), p))
    print("  [own>other => KV key-addressing among co-located caches; own==other => content-addressable index]")
if es_load:
    L = np.array(es_load)  # N, hit, chance
    N, H, C = L[:,0], L[:,1], L[:,2]
    print("read accuracy (own is argmax): %.4f vs chance %.4f (n=%d retrievals over %d same-site-competing sets)"
          % (H.mean(), C.mean(), len(L), int((N>=2).sum())))
    for lo_,hi_ in [(2,2),(3,3),(4,6),(7,999)]:
        m=(N>=lo_)&(N<=hi_)
        if m.sum(): print("   N %2d-%-3d: acc %.3f chance %.3f lift %+.3f (n=%d)"
                          %(lo_,min(hi_,int(N.max())),H[m].mean(),C[m].mean(),H[m].mean()-C[m].mean(),int(m.sum())))

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(centers))
if vv is not None: ax[0].plot(x, vv, "o-", label="visit-visit (place)")
if cr is not None: ax[0].plot(x, cr, "s-", color="purple", label="cache-retrieval")
ax[0].set_xticks(x); ax[0].set_xticklabels(centers, rotation=45); ax[0].axhline(0, color="k", lw=.5)
ax[0].set_xlabel("inter-site distance"); ax[0].set_ylabel("population corr"); ax[0].set_title("(1) positive control: barcode reactivation"); ax[0].legend()
if es_rows:
    ax[1].scatter(ER[:,1], ER[:,0], s=18); L2=[min(ER.min(),0), ER.max()]
    ax[1].plot(L2,L2,"k--",lw=1); ax[1].set_xlabel("other same-site"); ax[1].set_ylabel("own cache")
    ax[1].set_title("(2) co-located-cache event specificity")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "analyze_events.png"), dpi=110)
print("\nsaved out/analyze_events.png and events_summary.json")
