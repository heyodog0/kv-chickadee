"""
Place-isolated barcodes: the fair KV test.

v1/v2 negative root cause (see RESULTS.md 2026-07-03): the "barcode" was the raw
mean normalized population vector over an event window, which contains BOTH the
place code and the event-unique barcode. So barcode overlap rho was large (0.26)
and retrieval "reactivation" failed the same-site control (it was just place).

This script isolates the barcode the way Aronov et al. did in principle: subtract
the position-predicted (place) activity from each event's population vector and keep
the RESIDUAL. Place fields are estimated from NON-event frames (all site-interaction
windows excluded) so transient barcodes don't leak into the place map.

    barcode_e = observed_e  -  placefield(position during event e)

Then re-run, reporting BEFORE (raw, = v2) vs AFTER (place-isolated) side by side:

  (rho)  mean |corr| among cache barcodes           -> should drop toward 0 (orthogonal keys)
  (A)    event-specific reactivation, KV read WITH the same-site place control:
             retrieval vs OWN cache (matched cacheNum)
             retrieval vs OTHER caches SAME site   <- the control v2 failed
             retrieval vs OTHER caches DIFF site
         KV read is supported iff  own > other-same-site  after isolation.
  (X)    crosstalk vs load: per-retrieval read margin (own - mean other-same-site)
         and softmax read accuracy as a function of #competing caches (N).
         ML KV law: interference grows ~ rho*sqrt(N); margin should shrink with N.

Run on cluster (in the data dir):
  ./.venv/bin/python barcode_isolated.py
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
GRID = 18          # place-field grid resolution per axis
MIN_OCC = 8        # min occupancy frames to trust a grid cell
EVENT_BUF = 30     # frames to pad each event window when masking out for place fields (~0.5 s)

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else 0.0

def load_session(S):
    with h5py.File(os.path.join(S, "alignedSpikesAndPosture.mat"), "r") as h:
        ad = h["alignedData"]
        spks = np.asarray(ad["spks"], dtype=np.float32)
        wv = ad["wvStruct"]
        contam = np.asarray(wv["contam"]).ravel()
        meanRate = np.asarray(wv["meanRate"]).ravel()
        idx = np.asarray(wv["idx"]).ravel()
        smPts = np.asarray(ad["smPts"], dtype=np.float32)
    good = np.where((contam < 1.0) & (meanRate > 0.02) & np.isin(idx, [1, 2, 3]))[0]
    Sg = spks[good, :]
    Sn = Sg / (1e-2 + Sg.std(1, keepdims=True))
    Sn = (Sn - uniform_filter1d(Sn, size=30*60**2+1, axis=1, mode="nearest")).astype(np.float32)

    # position -> (nF,3,18); center-of-chest keypoint 5 (index 4) x,y
    if smPts.shape[0] == 18:
        smPts = np.transpose(smPts, (2, 1, 0))
    chest = smPts[:, :2, 4]                       # (nF, 2)

    a = sio.loadmat(os.path.join(S, "annotatedSeeds.mat"), struct_as_record=False, squeeze_me=True)["annotatedSeeds"]
    sc  = np.atleast_2d(np.asarray(a.seedChanges, dtype=float))
    cn  = np.asarray(a.cacheNum, dtype=float).ravel()
    newT = np.asarray(a.newCacheTimes, dtype=float).ravel()
    endT = np.asarray(a.endCacheTimes, dtype=float).ravel()
    isc = np.asarray(a.initSeedCounts, dtype=float).ravel()
    siteNum = np.asarray(a.countData.siteNum, dtype=int).ravel()
    return Sn, chest, sc, cn, newT, endT, isc, siteNum

def build_place_fields(Sn, chest, newT, endT):
    """Mean normalized activity per neuron per spatial grid cell, from NON-event frames."""
    nUnits, nBins = Sn.shape
    nF = min(nBins, chest.shape[0])
    Sn = Sn[:, :nF]; chest = chest[:nF]
    valid = np.isfinite(chest).all(1)
    # mask out event windows (all site interactions) so barcodes don't leak into place map
    ev_mask = np.zeros(nF, dtype=bool)
    f0 = np.clip((newT - 1 - EVENT_BUF).astype(int), 0, nF - 1)
    f1 = np.clip((endT + EVENT_BUF).astype(int), 1, nF)
    for lo, hi in zip(f0, f1):
        ev_mask[lo:hi] = True
    base = valid & ~ev_mask
    # grid edges from valid positions (robust range)
    lo_xy = np.nanpercentile(chest[valid], 1, axis=0)
    hi_xy = np.nanpercentile(chest[valid], 99, axis=0)
    ex = np.linspace(lo_xy[0], hi_xy[0], GRID + 1)
    ey = np.linspace(lo_xy[1], hi_xy[1], GRID + 1)
    gx = np.clip(np.digitize(chest[:, 0], ex) - 1, 0, GRID - 1)
    gy = np.clip(np.digitize(chest[:, 1], ey) - 1, 0, GRID - 1)
    cell = gx * GRID + gy                          # (nF,)
    ncell = GRID * GRID
    # accumulate per-cell sums over base frames
    occ = np.bincount(cell[base], minlength=ncell).astype(float)
    field = np.zeros((nUnits, ncell), dtype=np.float32)
    idx_base = np.where(base)[0]
    cb = cell[idx_base]
    for u in range(nUnits):
        s = np.bincount(cb, weights=Sn[u, idx_base], minlength=ncell)
        field[u] = s
    with np.errstate(invalid="ignore", divide="ignore"):
        field = field / np.maximum(occ, 1)[None, :]
    trusted = occ >= MIN_OCC
    field[:, ~trusted] = 0.0                        # untrusted cells -> predict grand mean (~0)
    return field, cell, nF

def event_vectors(Sn, chest, field, cell, nF, newT, endT, rows):
    """Return raw observed and place-residual population vectors for given event rows."""
    Sn = Sn[:, :nF]
    f0 = np.clip((newT - 1).astype(int), 0, nF - 1)
    f1 = np.clip(endT.astype(int), 1, nF)
    raw = np.zeros((len(rows), Sn.shape[0]), dtype=np.float32)
    res = np.zeros_like(raw)
    for i, e in enumerate(rows):
        lo, hi = f0[e], max(f0[e] + 1, f1[e])
        obs = Sn[:, lo:hi].mean(1)
        pred = field[:, cell[lo:hi]].mean(1)       # place prediction averaged over event frames
        raw[i] = obs
        res[i] = obs - pred
    return raw, res

def reactivation(Br, Bc, rnum, cnum, rsite, csite):
    """own / other-same-site / other-diff-site retrieval->cache correlation."""
    own, oss, ods = [], [], []
    per_retr = []   # (own, mean_other_ss, N_ss)
    for i in range(len(Br)):
        if rnum[i] <= 0: continue
        j = np.where(cnum == rnum[i])[0]
        if len(j) == 0: continue
        j = j[0]
        others = [k for k in range(len(Bc)) if k != j]
        ss = [k for k in others if csite[k] == rsite[i]]
        ds = [k for k in others if csite[k] != rsite[i]]
        o = corr(Br[i], Bc[j])
        own.append(o)
        if ss:
            m = np.mean([corr(Br[i], Bc[k]) for k in ss]); oss.append(m)
            per_retr.append((o, float(m), len(ss)))
        if ds:
            ods.append(np.mean([corr(Br[i], Bc[k]) for k in ds]))
    return own, oss, ods, per_retr

def analyze(S):
    Sn, chest, sc, cn, newT, endT, isc, siteNum = load_session(S)
    rs = sc.sum(1); cache = np.where(rs > 0)[0]; retr = np.where(rs < 0)[0]
    if len(cache) < 5 or len(retr) < 5: return None
    field, cell, nF = build_place_fields(Sn, chest, newT, endT)
    rawc, resc = event_vectors(Sn, chest, field, cell, nF, newT, endT, cache)
    rawr, resr = event_vectors(Sn, chest, field, cell, nF, newT, endT, retr)
    csite, rsite = siteNum[cache], siteNum[retr]
    cnum, rnum = cn[cache], cn[retr]
    out = {"session": os.path.basename(S), "n_cache": int(len(cache)), "n_retr": int(len(retr))}

    for tag, Bc, Br in [("raw", rawc, rawr), ("iso", resc, resr)]:
        # rho among cache barcodes
        if len(Bc) >= 3:
            G = np.corrcoef(Bc)
            out["rho_" + tag] = float(np.mean(np.abs(G[np.triu_indices(len(Bc), 1)])))
        own, oss, ods, per = reactivation(Br, Bc, rnum, cnum, rsite, csite)
        if own:  out["own_" + tag] = float(np.mean(own))
        if oss:  out["oss_" + tag] = float(np.mean(oss)); out["n_ss"] = len(oss)
        if ods:  out["ods_" + tag] = float(np.mean(ods))
        # crosstalk: per-retrieval read margin (own - other-same-site) vs N competing same-site caches
        if tag == "iso" and per:
            out["margin_load"] = [[int(n), float(o - m)] for (o, m, n) in per]
    return out

results = []
for i, S in enumerate(SESS):
    try:
        r = analyze(S)
        if r:
            results.append(r)
            print("[%2d/%d] %-22s c=%d r=%d | rho raw/iso %s/%s | own-oss raw %s iso %s"
                  % (i+1, len(SESS), r["session"], r["n_cache"], r["n_retr"],
                     ("%.2f"%r.get("rho_raw",np.nan)), ("%.2f"%r.get("rho_iso",np.nan)),
                     ("%+.3f"%(r.get("own_raw",np.nan)-r.get("oss_raw",np.nan))) if "oss_raw" in r else "-",
                     ("%+.3f"%(r.get("own_iso",np.nan)-r.get("oss_iso",np.nan))) if "oss_iso" in r else "-"))
        else:
            print("[%2d/%d] %-22s skipped" % (i+1, len(SESS), os.path.basename(S)))
    except Exception as e:
        import traceback; print("[%2d/%d] %-22s ERROR %r" % (i+1, len(SESS), os.path.basename(S), e)); traceback.print_exc()

json.dump([{k: v for k, v in r.items() if k != "margin_load"} for r in results],
          open(os.path.join(OUT, "barcode_isolated.json"), "w"), indent=1)

def arr(k): return np.array([r[k] for r in results if k in r and r[k] == r[k]])
from scipy.stats import wilcoxon, pearsonr
print("\n==================  PLACE-ISOLATED, POOLED OVER %d SESSIONS  ==================" % len(results))
for tag, label in [("raw", "RAW (=v2, place-contaminated)"), ("iso", "PLACE-ISOLATED (residual)")]:
    rho = arr("rho_" + tag); own = arr("own_" + tag); oss = arr("oss_" + tag); ods = arr("ods_" + tag)
    print("\n--- %s ---" % label)
    if len(rho): print("  rho (mean|corr| among caches): %.3f ± %.3f   [->0 = orthogonal keys]" % (rho.mean(), rho.std()))
    # align own/oss per session for a paired test
    pairs = np.array([(r["own_"+tag], r["oss_"+tag]) for r in results if ("own_"+tag in r and "oss_"+tag in r)])
    if len(pairs):
        d = pairs[:,0] - pairs[:,1]
        p = wilcoxon(d).pvalue if len(d) > 5 else np.nan
        print("  reactivation: own %.3f  other-SAME-site %.3f  (Δ=%+.3f, %d/%d own>same, p=%.1e)  <-- KV read w/ place control"
              % (pairs[:,0].mean(), pairs[:,1].mean(), d.mean(), int((d>0).sum()), len(d), p))
    if len(ods): print("  (ref) other-DIFF-site: %.3f" % ods.mean())

# crosstalk law: read margin vs N competing same-site caches (pooled per-retrieval, iso only)
ML = np.array([pt for r in results if "margin_load" in r for pt in r["margin_load"]])  # (n, 2): N, margin
if len(ML) > 30:
    N, mg = ML[:,0], ML[:,1]
    pr, pp = pearsonr(N, mg)
    print("\n--- CROSSTALK (place-isolated, per-retrieval, n=%d) ---" % len(ML))
    for lo_, hi_ in [(1,1),(2,3),(4,7),(8,999)]:
        m = (N>=lo_)&(N<=hi_)
        if m.sum(): print("  #competing same-site caches %2d-%-3d : read margin %+.3f (n=%d)"
                          % (lo_, min(hi_,int(N.max())), mg[m].mean(), int(m.sum())))
    print("  Pearson r(N, margin) = %+.3f (p=%.1e)   [<0 = interference grows with load, KV crosstalk]" % (pr, pp))

# figure: before/after rho, before/after reactivation delta, crosstalk
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
rr, ri = arr("rho_raw"), arr("rho_iso")
ax[0].hist([rr, ri], bins=12, label=["raw","isolated"]); ax[0].legend(); ax[0].set_title("(1) barcode overlap ρ"); ax[0].set_xlabel("mean |corr| among caches")
praw = np.array([(r["own_raw"], r["oss_raw"]) for r in results if "oss_raw" in r])
piso = np.array([(r["own_iso"], r["oss_iso"]) for r in results if "oss_iso" in r])
if len(praw) and len(piso):
    ax[1].scatter(praw[:,1], praw[:,0], s=16, alpha=.6, label="raw")
    ax[1].scatter(piso[:,1], piso[:,0], s=16, alpha=.6, label="isolated")
    L=[min(praw.min(),piso.min()), max(praw.max(),piso.max())]; ax[1].plot(L,L,"k--",lw=1)
    ax[1].set_xlabel("other-same-site"); ax[1].set_ylabel("own cache"); ax[1].set_title("(2) event-specific reactivation"); ax[1].legend()
if len(ML) > 30:
    bins = np.arange(1, min(int(N.max()),12)+1)
    mm = [mg[N==b].mean() if (N==b).sum() else np.nan for b in bins]
    ax[2].plot(bins, mm, "o-"); ax[2].axhline(0, color="k", lw=.6)
    ax[2].set_xlabel("# competing same-site caches (N)"); ax[2].set_ylabel("read margin (own−other)"); ax[2].set_title("(3) crosstalk vs load")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "barcode_isolated.png"), dpi=110)
print("\nsaved out/barcode_isolated.json and barcode_isolated.png")
