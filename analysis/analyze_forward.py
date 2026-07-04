"""
Hardening the co-located-cache result for publication.

Claim: hippocampal barcode reactivation at retrieval is SITE-specific but NOT
EPISODE-specific. When multiple caches share a site, a retrieval reactivates the
shared site-level barcode, not the individual cache's unique barcode.

The decisive control is a MEASUREMENT CEILING: split each cache's own event window in
half and ask whether the first half matches its OWN second half more than other
same-site caches' halves (all with leave-one-out site-template removal). If event-unique
structure IS resolvable within a single cache event among co-located caches (ceiling > 0),
yet a RETRIEVAL still cannot pick out its own cache (realistic test == 0), the null is a
biological fact, not an SNR floor.

Reports:
  T0  site-level reactivation (positive control): same-site vs diff-site (must be >0)
  T1  CEILING: split-half within-cache, own vs other-same-site (LOO template)   [is event structure measurable?]
  T2  REALISTIC: retrieval vs own cache, own vs other-same-site (LOO template)    [does it transfer to retrieval?]
  + per-bird T2, bootstrap 95% CI on T2 delta expressed as a fraction of the site-level effect,
    temporal-proximity control on T2, and co-location prevalence.

Run: ./.venv/bin/python analyze_forward.py
"""
import os, glob, json, re
import numpy as np
import scipy.io as sio
from scipy.stats import wilcoxon, pearsonr

EV  = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/events"
OUT = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/out"
FILES = sorted(glob.glob(EV + "/*.mat"))

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else 0.0

def arr2(x):
    x = np.atleast_2d(np.asarray(x, float)); return x

def specificity(probe, target, probe_site, target_site, probe_id, target_id):
    """own vs other-same-site with leave-one-out site-template removed (from target set).
       probe_id/target_id link a probe to its own target (matched id); for split-half, id==row index."""
    own, oth, hit, chance = [], [], [], []
    for i in range(len(probe)):
        cand = np.where(target_site == probe_site[i])[0]
        if len(cand) < 2: continue
        js = cand[target_id[cand] == probe_id[i]]
        if len(js) == 0: continue
        j = js[0]; sc = {}
        for k in cand:
            T = target[cand[cand != k]].mean(0)
            sc[k] = corr(probe[i] - T, target[k] - T)
        own.append(sc[j]); oth.append(np.mean([sc[k] for k in cand if k != j]))
        hit.append(int(max(sc, key=sc.get) == j)); chance.append(1.0/len(cand))
    return own, oth, hit, chance

def site_reactivation(retV, retSite, cacheV, cacheSite):
    """T0 positive control: retrieval vs same-site cache mean vs diff-site cache mean."""
    same, diff = [], []
    for i in range(len(retV)):
        s = np.where(cacheSite == retSite[i])[0]; d = np.where(cacheSite != retSite[i])[0]
        if len(s) < 1 or len(d) < 3: continue
        same.append(corr(retV[i], cacheV[s].mean(0))); diff.append(corr(retV[i], cacheV[d].mean(0)))
    return same, diff

rows = {"T0": [], "T1": [], "T2": []}          # per-session (own_mean, oth_mean, n) or (same,diff,n)
by_bird = {}                                    # bird -> list of per-session T2 (own,oth)
t2_pairs_persess = []                            # (own,oth) per session for CI
temporal = []                                    # (dt, own_minus_oth) per retrieval
per_site_counts = []

for f in FILES:
    S = os.path.basename(f)[:-4]; bird = re.match(r'([A-Za-z]+\d+)', S).group(1)
    m = sio.loadmat(f)
    def g(k): return arr2(m[k])
    cacheV, cV1, cV2 = g("cacheV"), g("cacheV1"), g("cacheV2")
    retV = g("retV")
    cacheSite = np.asarray(m["cacheSite"], float).ravel(); retSite = np.asarray(m["retSite"], float).ravel()
    cnum_c = np.asarray(m["cnum_cache"], float).ravel(); cnum_r = np.asarray(m["cnum_ret"], float).ravel()
    cacheT = np.asarray(m["cacheT"], float); retT = np.asarray(m["retT"], float)
    cacheT = cacheT[:,0] if cacheT.ndim==2 and cacheT.shape[1]>=1 else cacheT.ravel()
    retT = retT[:,0] if retT.ndim==2 and retT.shape[1]>=1 else retT.ravel()
    # align rows
    def ok(V, site):
        return np.isfinite(V).all(1) & np.isfinite(site)
    kc = ok(cacheV, cacheSite) & np.isfinite(cV1).all(1) & np.isfinite(cV2).all(1)
    kr = ok(retV, retSite)
    cacheV,cV1,cV2,cacheSite,cnum_c,cacheT = cacheV[kc],cV1[kc],cV2[kc],cacheSite[kc],cnum_c[kc],cacheT[kc]
    retV,retSite,cnum_r,retT = retV[kr],retSite[kr],cnum_r[kr],retT[kr]
    if len(cacheV) < 6 or len(retV) < 6: continue

    # co-location prevalence
    for s in np.unique(cacheSite): per_site_counts.append(int((cacheSite==s).sum()))

    # T0 site-level reactivation (positive control)
    same, diff = site_reactivation(retV, retSite, cacheV, cacheSite)
    if same: rows["T0"].append((np.mean(same), np.mean(diff), len(same)))

    # T1 CEILING: split-half within-cache, own(=self) vs other-same-site
    cid = np.arange(len(cacheV))
    o,t,h,c = specificity(cV1, cV2, cacheSite, cacheSite, cid, cid)
    if o: rows["T1"].append((np.mean(o), np.mean(t), len(o)))

    # T2 REALISTIC: retrieval vs own cache (matched cnum) vs other-same-site
    valid_r = cnum_r > 0
    o,t,h,c = specificity(retV[valid_r], cacheV, retSite[valid_r], cacheSite, cnum_r[valid_r], cnum_c)
    if o:
        rows["T2"].append((np.mean(o), np.mean(t), len(o)))
        by_bird.setdefault(bird, []).append((np.mean(o), np.mean(t)))
        t2_pairs_persess.append((np.mean(o), np.mean(t)))
    # temporal control: per-retrieval own-minus-oth vs dt to own cache
    for i in np.where(valid_r)[0]:
        cand = np.where(cacheSite == retSite[i])[0]
        if len(cand) < 2: continue
        js = cand[cnum_c[cand] == cnum_r[i]]
        if len(js)==0: continue
        j = js[0]; oth_idx=[k for k in cand if k!=j]
        Town = cacheV[[k for k in cand if k!=j]].mean(0)
        own = corr(retV[i]-Town, cacheV[j]-Town)
        othv = np.mean([corr(retV[i]-cacheV[[x for x in cand if x!=k]].mean(0), cacheV[k]-cacheV[[x for x in cand if x!=k]].mean(0)) for k in oth_idx])
        dt = retT[i] - cacheT[j]
        if np.isfinite(dt): temporal.append((float(dt), float(own-othv)))

def summ(key, labels=("own","other")):
    A = np.array([(a,b) for (a,b,n) in rows[key]]); d = A[:,0]-A[:,1]
    p = wilcoxon(d).pvalue if len(d)>5 else float('nan')
    print("%s: %s %.4f  %s %.4f  (Δ=%+.4f, %d/%d, p=%.2g, nsess=%d)"
          % (key, labels[0], A[:,0].mean(), labels[1], A[:,1].mean(), d.mean(),
             int((d>0).sum()), len(d), p, len(d)))
    return A[:,0].mean(), A[:,1].mean(), d

print("======  HARDENED CO-LOCATED-CACHE ANALYSIS  ======")
t0o,t0d,_ = summ("T0", ("same-site","diff-site"))
site_effect = t0o - t0d
print("   -> site-level barcode reactivation (positive control) = %+.4f\n" % site_effect)
_,_,d1 = summ("T1", ("own-split","other-same-site"))
_,_,d2 = summ("T2", ("retr-own","other-same-site"))

# bracket interpretation
print("\nBRACKET: within-event ceiling Δ=%+.4f vs retrieval Δ=%+.4f" % (d1.mean(), d2.mean()))
if d1.mean() > 0.01 and abs(d2.mean()) < 0.01:
    print("  -> event-unique structure IS resolvable within a cache event, but does NOT transfer to retrieval.")
    print("     => reactivation is SITE-specific, not EPISODE-specific. Null is biological, not SNR.")

# equivalence bound on T2 delta (bootstrap 95% CI), as fraction of site effect
P = np.array(t2_pairs_persess); d2v = P[:,0]-P[:,1]
rng = np.random.default_rng(0)
boot = np.array([rng.choice(d2v, len(d2v), replace=True).mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print("\nEQUIVALENCE BOUND on retrieval event-specificity Δ: %.4f [95%% CI %.4f, %.4f]" % (d2v.mean(), lo, hi))
print("   as fraction of site-level effect (%.4f): CI upper = %.1f%% of the site effect"
      % (site_effect, 100*hi/site_effect))

# per-bird T2
print("\nPER-BIRD retrieval event-specificity (Δ = own - other-same-site):")
for bird, vals in sorted(by_bird.items()):
    A = np.array(vals); d = A[:,0]-A[:,1]
    print("   %-8s Δ=%+.4f (n=%d sessions, %d>0)" % (bird, d.mean(), len(d), int((d>0).sum())))

# temporal control
if len(temporal) > 50:
    TT = np.array(temporal); r,p = pearsonr(TT[:,0], TT[:,1])
    print("\nTEMPORAL CONTROL: corr(retrieval-cache time gap, own-minus-other) = %.3f (p=%.2g, n=%d)"
          % (r, p, len(TT)))
    print("   [~0 => the null is not created/masked by drift or temporal proximity]")

# co-location prevalence
PS = np.array(per_site_counts)
print("\nCO-LOCATION PREVALENCE: %d site-sessions; %.0f%% have >=2 caches (median %d, max %d)"
      % (len(PS), 100*(PS>=2).mean(), int(np.median(PS)), int(PS.max())))

json.dump({"site_effect": float(site_effect), "T1_delta": float(d1.mean()), "T2_delta": float(d2v.mean()),
           "T2_ci": [float(lo), float(hi)], "per_bird": {b: float(np.mean([x[0]-x[1] for x in v])) for b,v in by_bird.items()}},
          open(os.path.join(OUT,"forward.json"), "w"), indent=1)
print("\nsaved out/forward.json")
