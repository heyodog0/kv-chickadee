"""
The KV crosstalk law on measured barcodes: does retrieval fidelity degrade as error ~ rho*sqrt(N)?

Neither Chettih 2024 nor Fang 2026 tested this scaling on the neural data. We operationalize it
place-controlled, using the site-level barcode reactivation as the "read fidelity":

  reactivation boost(retrieval i) = corr(r_i, own-site cache centroid)
                                  - mean corr(r_i, different-site cache centroids)
  = the same-site excess over place decay (the +0.089 in the positive control), per retrieval.

Predictions (associative-memory / KV crosstalk):
  (B1) WRITE RULE: across-different-site barcode overlap rho vs memory load N (caches stored).
       independent random keys -> rho flat with N (=> sqrt(N) interference growth);
       active decorrelation -> rho falls with N (interference-minimizing / delta-like).
  (B2) READ FIDELITY vs LOAD: reactivation boost shrinks as N grows; fit boost ~ a - b*sqrt(N)
       vs a - b*N. sqrt(N) is the random-key KV signature.
  (B3) rho x N: sessions/epochs with higher rho have lower fidelity at matched N.
  Temporal-proximity control: partial out |t_retrieval - t_cache| (drift confound).

Load N at a retrieval = # caches with onset before it and not yet retrieved (uses event times).

Run: ./.venv/bin/python analyze_crosstalk.py
"""
import os, glob, json
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr

EV  = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/events"
OUT = "/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/out"
FILES = sorted(glob.glob(EV + "/*.mat"))

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else 0.0

def onset(T, n):
    T = np.atleast_2d(np.asarray(T, float))
    if T.shape[0] != n and T.shape[1] == n: T = T.T
    return T[:, 0] if T.size else np.full(n, np.nan)

def clean(V, site, T, cnum=None):
    V = np.atleast_2d(np.asarray(V, float)); site = np.asarray(site, float).ravel()
    if V.shape[0] != site.shape[0] and V.shape[1] == site.shape[0]: V = V.T
    t = onset(T, V.shape[0])
    ok = np.isfinite(V).all(1) & np.isfinite(site) & np.isfinite(t)
    cn = np.asarray(cnum, float).ravel() if cnum is not None and np.size(cnum) else np.zeros(V.shape[0])
    return V[ok], site[ok], t[ok], (cn[ok] if cn.shape[0]==ok.shape[0] else cn)

boost_load = []     # (N_load, boost, dt) per retrieval, pooled
rho_load = []       # (N_load, rho_diffsite) per session-epoch
sess_rho_fid = []   # (session rho, session mean boost)

for f in FILES:
    m = sio.loadmat(f)
    cacheV, cacheSite, cacheT, cnum_cache = clean(m["cacheV"], m["cacheSite"], m["cacheT"], m["cnum_cache"])
    retV, retSite, retT, cnum_ret = clean(m["retV"], m["retSite"], m["retT"], m["cnum_ret"])
    if len(cacheV) < 8 or len(retV) < 8: continue

    # site centroids will be recomputed per-retrieval excluding own cache
    sites = np.unique(cacheSite)

    # (B2/B3) read fidelity per retrieval, with load and temporal proximity
    boosts = []
    for i in range(len(retV)):
        s = retSite[i]
        same = np.where(cacheSite == s)[0]
        diff = np.where(cacheSite != s)[0]
        if len(same) < 1 or len(diff) < 3: continue
        # own-site centroid (exclude the matched own cache to avoid trivial identity)
        own_cn = cnum_ret[i]
        same_use = same[cnum_cache[same] != own_cn] if own_cn > 0 else same
        if len(same_use) < 1: same_use = same
        c_same = corr(retV[i], cacheV[same_use].mean(0))
        c_diff = corr(retV[i], cacheV[diff].mean(0))
        boost = c_same - c_diff
        # load = # caches stored (onset before retrieval, not yet retrieved before this retrieval)
        made = cacheT < retT[i]
        # a cache is "retrieved" if some retrieval with same cnum happened before retT[i]
        stored = made.copy()
        for k in np.where(made)[0]:
            rk = np.where((cnum_ret == cnum_cache[k]) & (retT < retT[i]))[0]
            if len(rk): stored[k] = False
        N = int(stored.sum())
        # temporal proximity: dt to own cache
        own_idx = np.where(cnum_cache == own_cn)[0] if own_cn > 0 else []
        dt = float(retT[i] - cacheT[own_idx[0]]) if len(own_idx) else np.nan
        boosts.append(boost)
        boost_load.append((N, boost, dt))

    # (B1) write rule: across-different-site rho vs cache order (load); rho among barcode-dominated diff-site pairs
    order = np.argsort(cacheT)
    for frac_lo, frac_hi in [(0.0,0.33),(0.33,0.66),(0.66,1.0)]:
        idx = order[int(frac_lo*len(order)):int(frac_hi*len(order))]
        if len(idx) < 5: continue
        rr = []
        for a in range(len(idx)):
            for b in range(a+1, len(idx)):
                if cacheSite[idx[a]] != cacheSite[idx[b]]:
                    rr.append(abs(corr(cacheV[idx[a]], cacheV[idx[b]])))
        if rr: rho_load.append((0.5*(frac_lo+frac_hi)*len(order), float(np.mean(rr))))

    # session-level rho (diff-site) and mean fidelity
    rr = []
    for a in range(len(cacheV)):
        for b in range(a+1, len(cacheV)):
            if cacheSite[a] != cacheSite[b]: rr.append(abs(corr(cacheV[a], cacheV[b])))
    if rr and boosts: sess_rho_fid.append((float(np.mean(rr)), float(np.mean(boosts))))

print("======  KV CROSSTALK LAW ON MEASURED BARCODES  ======")

# (B2) fidelity vs load
BL = np.array(boost_load)  # N, boost, dt
N, B, DT = BL[:,0], BL[:,1], BL[:,2]
print("\n(B2) READ FIDELITY (reactivation boost) vs memory load N   n=%d retrievals" % len(BL))
for lo,hi in [(0,4),(5,9),(10,19),(20,49),(50,9999)]:
    m = (N>=lo)&(N<=hi)
    if m.sum()>10: print("   N %3d-%-4d: boost %+.4f (n=%d)" % (lo, min(hi,int(N.max())), B[m].mean(), int(m.sum())))
mask = N > 0
r_lin, p_lin = pearsonr(N[mask], B[mask])
r_sqrt, p_sqrt = pearsonr(np.sqrt(N[mask]), B[mask])
print("   corr(N, boost)=%.3f (p=%.1e) ; corr(sqrt(N), boost)=%.3f (p=%.1e)" % (r_lin, p_lin, r_sqrt, p_sqrt))
# fit boost = a - b*sqrt(N); report b and R^2 vs linear
def fit_r2(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ coef; ss = 1 - ((y-yh)**2).sum()/(((y-y.mean())**2).sum()+1e-12)
    return coef, ss
c_s, r2_s = fit_r2(np.sqrt(N[mask]), B[mask])
c_n, r2_n = fit_r2(N[mask], B[mask])
print("   fit boost=a-b*sqrt(N): b=%.4f R2=%.3f | boost=a-b*N: b=%.5f R2=%.3f  [sqrt better => random-key KV law]"
      % (-c_s[1], r2_s, -c_n[1], r2_n))
# temporal control: partial correlation of boost~sqrt(N) controlling dt
ok = mask & np.isfinite(DT)
if ok.sum() > 50:
    def resid(y, x):
        A = np.vstack([np.ones_like(x), x]).T; c,*_=np.linalg.lstsq(A,y,rcond=None); return y - A@c
    bN = resid(B[ok], DT[ok]); bS = resid(np.sqrt(N[ok]), DT[ok])
    rp, pp = pearsonr(bS, bN)
    print("   temporal-proximity control: partial corr(sqrt(N), boost | dt) = %.3f (p=%.1e)" % (rp, pp))

# (B1) write rule
if rho_load:
    RL = np.array(rho_load)
    r, p = spearmanr(RL[:,0], RL[:,1])
    print("\n(B1) WRITE RULE: across-diff-site barcode overlap rho vs cache load")
    print("   Spearman(load, rho) = %.3f (p=%.1e)   [~0 => independent draws (rho flat, sqrt(N) growth); <0 => active decorrelation]" % (r, p))

# (B3) rho x fidelity across sessions
if sess_rho_fid:
    RF = np.array(sess_rho_fid)
    r, p = pearsonr(RF[:,0], RF[:,1])
    print("\n(B3) ACROSS SESSIONS: barcode overlap rho vs mean reactivation fidelity")
    print("   corr(rho, boost) = %.3f (p=%.1e), n=%d sessions  [<0 => higher key overlap -> worse read = KV crosstalk]" % (r, p, len(RF)))

json.dump({"boost_load": BL.tolist()[:2000], "rho_load": rho_load, "sess_rho_fid": sess_rho_fid,
           "fit": {"b_sqrt": float(-c_s[1]), "r2_sqrt": float(r2_s), "b_lin": float(-c_n[1]), "r2_lin": float(r2_n)}},
          open(os.path.join(OUT, "crosstalk.json"), "w"), indent=1)
print("\nsaved out/crosstalk.json")
