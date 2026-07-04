"""
Corrected multi-session analysis on TRUE cache/retrieval events.

Fix vs v1: events come from seedChanges (the authoritative +/-1 seed record), NOT
toCache_site (which flagged all ~2192 site interactions, mostly empty-site checks).
Per session there are only ~58 caches / ~60 retrievals; cacheNum links each retrieval
to its own cache (same cacheNum, same site). Pool across 54 sessions for power.

Tests:
 (A) EVENT-SPECIFIC REACTIVATION (the KV read): retrieval barcode vs its OWN cache
     (matched cacheNum) vs other caches, and vs other caches AT THE SAME SITE (place control).
 (B) BARCODE OVERLAP rho: mean |corr| among cache barcodes (near-orthogonality).
 (C) BEHAVIORAL CROSSTALK: does search cost (empty wrong-site checks right before a
     retrieval) grow with cache LOAD (seeds currently stored)?  KV crosstalk => yes.
"""
import os, glob, json
import numpy as np
import h5py, scipy.io as sio
from scipy.ndimage import uniform_filter1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/Grid Caching Data"
OUT  = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/out"
os.makedirs(OUT, exist_ok=True)
SESS = sorted(d for d in glob.glob(ROOT + "/*_*") if os.path.isdir(d))

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

def load_session(S):
    with h5py.File(os.path.join(S, "alignedSpikesAndPosture.mat"), "r") as h:
        ad = h["alignedData"]; spks = np.asarray(ad["spks"], dtype=np.float32); wv = ad["wvStruct"]
        contam = np.asarray(wv["contam"]).ravel(); meanRate = np.asarray(wv["meanRate"]).ravel(); idx = np.asarray(wv["idx"]).ravel()
    good = np.where((contam < 1.0) & (meanRate > 0.02) & np.isin(idx, [1, 2, 3]))[0]
    Sg = spks[good, :]; Sn = Sg / (1e-2 + Sg.std(1, keepdims=True))
    Sn = (Sn - uniform_filter1d(Sn, size=30*60**2+1, axis=1, mode="nearest")).astype(np.float32)
    a = sio.loadmat(os.path.join(S, "annotatedSeeds.mat"), struct_as_record=False, squeeze_me=True)["annotatedSeeds"]
    sc  = np.atleast_2d(np.asarray(a.seedChanges, dtype=float))
    cn  = np.asarray(a.cacheNum, dtype=float).ravel()
    newT = np.asarray(a.newCacheTimes, dtype=float).ravel()
    endT = np.asarray(a.endCacheTimes, dtype=float).ravel()
    isc = np.asarray(a.initSeedCounts, dtype=float).ravel()
    siteNum = np.asarray(a.countData.siteNum, dtype=int).ravel()
    return Sn, sc, cn, newT, endT, isc, siteNum

def barcodes(Sn, newT, endT, rows):
    nb = Sn.shape[1]; f0 = np.clip((newT-1).astype(int), 0, nb-1); f1 = np.clip(endT.astype(int), 1, nb)
    B = np.zeros((len(rows), Sn.shape[0]), dtype=np.float32)
    for i, e in enumerate(rows):
        lo, hi = f0[e], max(f0[e]+1, f1[e]); B[i] = Sn[:, lo:hi].mean(1)
    return B

def analyze(S):
    Sn, sc, cn, newT, endT, isc, siteNum = load_session(S)
    rs = sc.sum(1); cache = np.where(rs > 0)[0]; retr = np.where(rs < 0)[0]
    if len(cache) < 5 or len(retr) < 5: return None
    Bc = barcodes(Sn, newT, endT, cache); Br = barcodes(Sn, newT, endT, retr)
    csite = siteNum[cache]; rsite = siteNum[retr]; cnum = cn[cache]; rnum = cn[retr]
    r = {"session": os.path.basename(S), "n_cache": int(len(cache)), "n_retr": int(len(retr))}
    # (A) event-specific reactivation
    own, oth, oth_ss = [], [], []
    for i in range(len(retr)):
        if rnum[i] <= 0: continue
        j = np.where(cnum == rnum[i])[0]
        if len(j) == 0: continue
        j = j[0]; others = [k for k in range(len(cache)) if k != j]
        if not others: continue
        own.append(corr(Br[i], Bc[j]))
        oth.append(np.mean([corr(Br[i], Bc[k]) for k in others]))
        ss = [k for k in others if csite[k] == rsite[i]]
        if ss: oth_ss.append(np.mean([corr(Br[i], Bc[k]) for k in ss]))
    if own:
        r["react_own"] = float(np.mean(own)); r["react_oth"] = float(np.mean(oth)); r["n_match"] = len(own)
        if oth_ss: r["react_oth_ss"] = float(np.mean(oth_ss)); r["n_ss"] = len(oth_ss)
    # (B) barcode overlap rho
    if len(Bc) >= 3:
        G = np.corrcoef(Bc); r["rho"] = float(np.mean(np.abs(G[np.triu_indices(len(Bc), 1)])))
    # (C) behavioral crosstalk: search cost vs load
    typ = np.where(rs > 0, 1, np.where(rs < 0, -1, 0))
    order = np.argsort(newT); pos = {e: p for p, e in enumerate(order)}
    seeds0 = isc.sum()
    net = seeds0 + np.cumsum(sc.sum(1)[order])            # seeds stored over time
    costs, loads = [], []
    for i in range(len(retr)):
        e = retr[i]; p = pos[e]; cnt = 0; k = p - 1
        while k >= 0 and typ[order[k]] == 0 and siteNum[order[k]] != rsite[i]:
            cnt += 1; k -= 1
        costs.append(cnt); loads.append(max(net[p-1] if p > 0 else seeds0, 0))
    r["cost_load"] = list(map(list, zip(map(float, loads), map(float, costs))))
    return r

results = []
for i, S in enumerate(SESS):
    try:
        rr = analyze(S)
        if rr:
            results.append(rr)
            print("[%2d/%d] %-22s cache=%d retr=%d match=%s react own/oth=%s/%s rho=%s"
                  % (i+1, len(SESS), rr["session"], rr["n_cache"], rr["n_retr"], rr.get("n_match","-"),
                     ("%.2f"%rr["react_own"]) if "react_own" in rr else "-",
                     ("%.2f"%rr["react_oth"]) if "react_oth" in rr else "-",
                     ("%.2f"%rr["rho"]) if "rho" in rr else "-"))
        else:
            print("[%2d/%d] %-22s skipped" % (i+1, len(SESS), os.path.basename(S)))
    except Exception as e:
        print("[%2d/%d] %-22s ERROR %r" % (i+1, len(SESS), os.path.basename(S), e))

json.dump([{k: v for k, v in r.items() if k != "cost_load"} for r in results],
          open(os.path.join(OUT, "multi_v2.json"), "w"), indent=1)

def arr(k): return np.array([r[k] for r in results if k in r and r[k] == r[k]])
ro, rt = arr("react_own"), arr("react_oth"); rss = arr("react_oth_ss"); rho = arr("rho")
print("\n==================  TRUE EVENTS, POOLED OVER %d SESSIONS  ==================" % len(results))
print("mean caches/session %.0f  retrievals/session %.0f" % (arr("n_cache").mean(), arr("n_retr").mean()))
from scipy.stats import wilcoxon, pearsonr
if len(ro):
    d = ro - rt; p = wilcoxon(d).pvalue if len(d) > 5 else np.nan
    print("(A) event-specific reactivation: own-cache %.3f  other-cache %.3f  (Δ=%.3f, %d/%d own>other, p=%.1e)"
          % (ro.mean(), rt.mean(), d.mean(), int((d > 0).sum()), len(d), p))
    if len(rss):
        d2 = arr("react_own")[:len(rss)]  # align not guaranteed; report means only
        print("    place control — other caches SAME site: %.3f  (own %.3f)  [own>same-site => beyond place]"
              % (rss.mean(), ro.mean()))
if len(rho):
    print("(B) barcode overlap rho (mean |corr| among caches): %.3f ± %.3f  [near 0 => near-orthogonal keys]" % (rho.mean(), rho.std()))
# (C) behavioral crosstalk
CL = np.array([p for r in results if "cost_load" in r for p in r["cost_load"]])  # (N,2): load, cost
if len(CL) > 30:
    ld, co = CL[:, 0], CL[:, 1]
    pr, pp = pearsonr(ld, co)
    print("(C) behavioral: search cost vs cache load  (n=%d retrievals)" % len(CL))
    for lo_, hi_ in [(0, 1), (2, 3), (4, 99)]:
        m = (ld >= lo_) & (ld <= hi_)
        if m.sum(): print("      load %d-%d : mean search cost %.2f (n=%d)" % (lo_, min(hi_,int(ld.max())), co[m].mean(), int(m.sum())))
    print("      Pearson r(load, cost) = %.3f (p=%.1e)   [>0 => more stored caches -> more search errors = crosstalk]" % (pr, pp))

# figure
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
if len(ro):
    ax[0].scatter(rt, ro, s=20); L=[min(rt.min(),ro.min()), max(rt.max(),ro.max())]
    ax[0].plot(L, L, "k--", lw=1); ax[0].set_xlabel("other-cache"); ax[0].set_ylabel("own-cache"); ax[0].set_title("(A) event-specific reactivation")
if len(rho): ax[1].hist(rho, bins=15); ax[1].set_xlabel("mean |corr| among cache barcodes"); ax[1].set_title("(B) barcode overlap ρ")
if len(CL) > 30:
    bins = np.arange(0, int(ld.max())+2); mc = [co[(ld>=b)&(ld<b+1)].mean() if ((ld>=b)&(ld<b+1)).sum() else np.nan for b in bins]
    ax[2].plot(bins, mc, "o-"); ax[2].set_xlabel("cache load (seeds stored)"); ax[2].set_ylabel("search cost"); ax[2].set_title("(C) behavioral crosstalk")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "multi_v2.png"), dpi=110)
print("\nsaved out/multi_v2.json and multi_v2.png")
