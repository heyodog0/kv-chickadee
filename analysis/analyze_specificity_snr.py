"""
SNR discriminator for co-located-cache event specificity.

The raw-cosine leave-one-out test found own==other-same-site (p=0.87). Is that a genuine
null (content-addressable index) or just low SNR (sparse barcode transient buried under the
high-variance shared place/action component)? Test with three readouts of increasing power:

  raw        : cosine on the population vector (baseline; = analyze_events.py)
  whitened   : Ledoit-Wolf-shrunk Mahalanobis whitening (up-weights low-variance directions)
  pc-residual: remove the top-K shared PCs (place/action are low-D), keep the high-D barcode
               subspace, sweep K.

All use leave-one-out site-template removal, then compare a retrieval to its OWN co-located
cache (matched cacheNum) vs other same-site caches. POWER CONTROL: the same machinery on the
cross-site matched pair (place differs) must succeed, or the method lacks power.

If NO readout reveals own>other -> robust null -> content-addressable index in the co-located
regime. If one does -> event-specific reactivation was real but sub-cosine -> KV key-addressing.

Run: ./.venv/bin/python analyze_specificity_snr.py
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from scipy.stats import wilcoxon
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
EV  = os.environ.get("EVENTS_DIR", os.path.join(HERE, "..", "events"))   # repo-local events/ (laptop); override with EVENTS_DIR on the cluster
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)
FILES = sorted(glob.glob(os.path.join(EV, "*.mat")))
if not FILES:
    raise SystemExit(f"No events/*.mat in {EV}. Put the events/ folder at the repo root, or set EVENTS_DIR.")
KSWEEP = [1, 2, 3, 5, 8, 12]

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else 0.0

def clean(V, site, cnum):
    V = np.atleast_2d(np.asarray(V, float)); site = np.asarray(site, float).ravel()
    cnum = np.asarray(cnum, float).ravel() if np.size(cnum) else np.zeros(V.shape[0])
    if V.shape[0] != site.shape[0] and V.shape[1] == site.shape[0]: V = V.T
    ok = np.isfinite(V).all(1) & np.isfinite(site)
    return V[ok], site[ok], cnum[ok] if cnum.shape[0]==ok.shape[0] else cnum

def loo_specificity(cacheV, cacheSite, cnum_cache, retV, retSite, cnum_ret):
    """own vs other-same-site (leave-one-out site template) + argmax read accuracy."""
    own, oth, hit, ch = [], [], [], []
    for i in range(len(retV)):
        if cnum_ret[i] <= 0: continue
        cand = np.where(cacheSite == retSite[i])[0]
        if len(cand) < 2: continue
        js = cand[cnum_cache[cand] == cnum_ret[i]]
        if len(js) == 0: continue
        j = js[0]; sc = {}
        for k in cand:
            T = cacheV[cand[cand != k]].mean(0)
            sc[k] = corr(retV[i] - T, cacheV[k] - T)
        own.append(sc[j]); oth.append(np.mean([sc[k] for k in cand if k != j]))
        hit.append(int(max(sc, key=sc.get) == j)); ch.append(1.0 / len(cand))
    return own, oth, hit, ch

def cross_site_power(cacheV, cacheSite, cnum_cache, retV, retSite, cnum_ret, rng):
    """POWER CONTROL: own cache (same site) vs a random different-site cache."""
    own, oth = [], []
    for i in range(len(retV)):
        if cnum_ret[i] <= 0: continue
        js = np.where((cacheSite == retSite[i]) & (cnum_cache == cnum_ret[i]))[0]
        diff = np.where(cacheSite != retSite[i])[0]
        if len(js) == 0 or len(diff) < 3: continue
        own.append(corr(retV[i], cacheV[js[0]]))
        oth.append(np.mean([corr(retV[i], cacheV[k]) for k in rng.choice(diff, 3, replace=False)]))
    return own, oth

def whiten_fit(X):
    lw = LedoitWolf().fit(X); Sig = lw.covariance_; mu = X.mean(0)
    w, V = np.linalg.eigh(Sig)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-8))) @ V.T
    return mu, W

def pca_fit(X):
    mu = X.mean(0); Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt  # rows of Vt = PCs

results = {"raw": [], "whitened": [], "power": []}
pc_results = {k: [] for k in KSWEEP}
rng = np.random.default_rng(0)

for f in FILES:
    m = sio.loadmat(f)
    cacheV, cacheSite, cnum_cache = clean(m["cacheV"], m["cacheSite"], m["cnum_cache"])
    retV, retSite, cnum_ret = clean(m["retV"], m["retSite"], m["cnum_ret"])
    if len(cacheV) < 4 or len(retV) < 4: continue
    allX = np.vstack([cacheV, retV])

    # raw
    o, t, h, c = loo_specificity(cacheV, cacheSite, cnum_cache, retV, retSite, cnum_ret)
    if o: results["raw"].append((np.mean(o), np.mean(t), np.mean(h), np.mean(c), len(o)))
    # power control (cross-site, raw)
    po, pt = cross_site_power(cacheV, cacheSite, cnum_cache, retV, retSite, cnum_ret, rng)
    if po: results["power"].append((np.mean(po), np.mean(pt), len(po)))
    # whitened
    mu, W = whiten_fit(allX)
    cW = (cacheV - mu) @ W; rW = (retV - mu) @ W
    o, t, h, c = loo_specificity(cW, cacheSite, cnum_cache, rW, retSite, cnum_ret)
    if o: results["whitened"].append((np.mean(o), np.mean(t), np.mean(h), np.mean(c), len(o)))
    # pc-residual sweep
    mu2, Vt = pca_fit(allX)
    for K in KSWEEP:
        if K >= Vt.shape[0]: continue
        P = Vt[:K]
        cR = (cacheV - mu2) - ((cacheV - mu2) @ P.T) @ P
        rR = (retV - mu2) - ((retV - mu2) @ P.T) @ P
        o, t, h, c = loo_specificity(cR, cacheSite, cnum_cache, rR, retSite, cnum_ret)
        if o: pc_results[K].append((np.mean(o), np.mean(t), np.mean(h), np.mean(c), len(o)))

def summarize(rows, label):
    A = np.array([(o, t, h, c) for (o, t, h, c, n) in rows])
    d = A[:,0] - A[:,1]; p = wilcoxon(d).pvalue if len(d) > 5 else float("nan")
    print("%-16s own %.4f  other %.4f  (Δ=%+.4f, %d/%d own>oth, p=%.2g) | readacc %.3f vs chance %.3f"
          % (label, A[:,0].mean(), A[:,1].mean(), d.mean(), int((d>0).sum()), len(d), p,
             A[:,2].mean(), A[:,3].mean()))
    return {"label": label, "own": float(A[:,0].mean()), "other": float(A[:,1].mean()),
            "delta": float(d.mean()), "p": float(p), "n_sess": len(d),
            "readacc": float(A[:,2].mean()), "chance": float(A[:,3].mean())}

print("======  CO-LOCATED EVENT SPECIFICITY: SNR discriminator  ======")
out = {}
out["raw"] = summarize(results["raw"], "raw cosine")
out["whitened"] = summarize(results["whitened"], "whitened (LDA)")
for K in KSWEEP:
    if pc_results[K]: out["pc%d"%K] = summarize(pc_results[K], "pc-resid K=%d"%K)
# power control
P = np.array([(o, t) for (o, t, n) in results["power"]])
dp = P[:,0] - P[:,1]; pp = wilcoxon(dp).pvalue if len(dp) > 5 else float("nan")
print("\nPOWER CONTROL (cross-site matched pair): own-site %.4f vs diff-site %.4f (Δ=%+.4f, %d/%d, p=%.2g)"
      % (P[:,0].mean(), P[:,1].mean(), dp.mean(), int((dp>0).sum()), len(dp), pp))
print("  [power control MUST be strongly positive, else the readout lacks power]")
out["power"] = {"own": float(P[:,0].mean()), "diff": float(P[:,1].mean()), "delta": float(dp.mean()), "p": float(pp)}

verdict = "ROBUST NULL (content-addressable index): no readout reveals own>other-same-site" \
    if all(out[k]["delta"] < 0.01 or out[k]["p"] > 0.05 for k in out if k not in ("power",)) \
    else "SNR: some readout reveals event-specific reactivation (own>other)"
print("\nVERDICT:", verdict)
json.dump(out, open(os.path.join(OUT, "specificity_snr.json"), "w"), indent=1)
print("saved out/specificity_snr.json")
