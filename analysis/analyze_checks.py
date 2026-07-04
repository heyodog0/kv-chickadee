"""
Check-reactivation (event-specific, place-controlled): is the barcode read cue-triggered?

At single-cache ("baited") sites, while the cache is stored (after caching, before retrieval),
compare how much CHECKS vs VISITS at that site resemble the cache's barcode:
  check-corr  = mean corr(check_at_s, cache_k)     [site engagement]
  visit-corr  = mean corr(visit_at_s, cache_k)     [place-only baseline, same site]
Both share place (same site), so check-corr − visit-corr isolates barcode reactivation during a
check. >0 => engaging a site re-expresses its barcode without a retrieval (a cue-triggered,
content-addressed read). Retrieval-corr = corr(retrieval_k, cache_k) is the known reactivation.

Uses only event times already in events/ (no re-extraction needed). Run: python analyze_checks.py
"""
import os, glob, json
import numpy as np, scipy.io as sio
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
EV  = os.environ.get("EVENTS_DIR", os.path.join(HERE, "..", "events"))
FILES = sorted(glob.glob(os.path.join(EV, "*.mat")))
if not FILES:
    raise SystemExit(f"No events/*.mat in {EV}. Put the events/ folder at the repo root, or set EVENTS_DIR.")

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else 0.0

def load(m, kV, kS, kT):
    V = np.atleast_2d(np.asarray(m[kV], float)); site = np.asarray(m[kS], float).ravel()
    T = np.asarray(m[kT], float); T = T[:,0] if T.ndim == 2 and T.shape[1] >= 1 else T.ravel()
    if V.shape[0] != site.shape[0] and V.shape[1] == site.shape[0]: V = V.T
    ok = np.isfinite(V).all(1) & np.isfinite(site) & np.isfinite(T)
    return V[ok], site[ok], T[ok]

pairs = []   # per cache: (check_corr, visit_corr); plus retrieval_corr where available
retr_own = []
for f in FILES:
    m = sio.loadmat(f)
    cacheV, cacheSite, cacheT = load(m, "cacheV", "cacheSite", "cacheT")
    chkV, chkSite, chkT       = load(m, "chkV", "chkSite", "chkT")
    visV, visSite, visT       = load(m, "visV", "visSite", "visT")
    retV, retSite, retT       = load(m, "retV", "retSite", "retT")
    cnum_c = np.asarray(m["cnum_cache"], float).ravel()
    cnum_r = np.asarray(m["cnum_ret"], float).ravel()
    if len(cacheV) < 4: continue
    # single-cache sites only (no co-location ambiguity)
    sites, counts = np.unique(cacheSite, return_counts=True)
    single = set(sites[counts == 1].tolist())
    for k in range(len(cacheV)):
        s = cacheSite[k]
        if s not in single: continue
        t0 = cacheT[k]
        rk = retT[(cnum_r == cnum_c[k]) & np.isfinite(retT)]
        t_end = rk.min() if len(rk) else np.inf              # stored window [cache, retrieval)
        ci = np.where((chkSite == s) & (chkT >= t0) & (chkT < t_end))[0]
        vi = np.where((visSite == s) & (visT >= t0) & (visT < t_end))[0]
        if len(ci) < 1 or len(vi) < 1: continue
        cc = np.mean([corr(chkV[i], cacheV[k]) for i in ci])
        vc = np.mean([corr(visV[i], cacheV[k]) for i in vi])
        pairs.append((cc, vc))
        if len(rk):
            j = np.where((retSite == s) & (cnum_r == cnum_c[k]))[0]
            if len(j): retr_own.append(corr(retV[j[0]], cacheV[k]))

P = np.array(pairs)   # (n, 2): check_corr, visit_corr
print("======  CHECK-REACTIVATION (single-cache sites, place-controlled by visits)  ======")
print("n = %d caches with both checks and visits while stored\n" % len(P))
cc, vc = P[:,0], P[:,1]; d = cc - vc
p = wilcoxon(d).pvalue if len(d) > 5 else float("nan")
print("  check   corr to cache barcode: %+.4f" % cc.mean())
print("  visit   corr to cache barcode: %+.4f   (place baseline, same site)" % vc.mean())
print("  check − visit:                 %+.4f (%d/%d > 0, p=%.2g)" % (d.mean(), int((d>0).sum()), len(d), p))
if retr_own:
    print("  (ref) retrieval corr to own cache: %+.4f" % np.mean(retr_own))
verdict = ("checks re-express the barcode beyond place → the read is cue/site-triggered, not retrieval-only"
           if d.mean() > 0 and (wilcoxon(d).pvalue < 0.05 if len(d) > 5 else False)
           else "checks do NOT exceed the place baseline → reactivation is not evident during checks")
print("\nVERDICT:", verdict)
OUT = os.path.join(HERE, "..", "out"); os.makedirs(OUT, exist_ok=True)
json.dump({"check": float(cc.mean()), "visit": float(vc.mean()), "check_minus_visit": float(d.mean()),
           "n": int(len(P)), "retr_own": float(np.mean(retr_own)) if retr_own else None},
          open(os.path.join(OUT, "checks.json"), "w"), indent=1)
