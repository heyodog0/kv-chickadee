"""
The decisive KV-read test: event-specific reactivation with a leave-one-out
SITE-TEMPLATE control (position-free).

Motivation: raw event vectors and a coarse body-position place-field subtraction
(barcode_isolated.py) are both dominated by a shared same-site component (fine place
+ a stereotyped caching-action signal). To ask whether a retrieval reactivates ITS
OWN cache beyond that shared component, remove the site template and test the residual.

For each retrieval i at site s with its own cache j* (matched cacheNum):
  candidates = caches at site s
  for each candidate k:
      T_k = mean of caches at s EXCLUDING k        # leave-one-out site template (unbiased)
      score_k = corr(Br[i] - T_k, Bc[k] - T_k)
  own_score  = score_{j*}
  other_score = mean_{k != j*} score_k
  own_is_argmax = (j* == argmax_k score_k)          # read accuracy; chance = 1/len(candidates)

KV read is supported iff, after removing the site template:
  own_score > other_score  AND  own-argmax accuracy > chance.

Crosstalk (ρ√N): does own-argmax read accuracy fall as the number of competing
same-site caches (N) grows? Report accuracy vs N and (accuracy - chance) vs N.

Run on cluster: ./.venv/bin/python event_specificity.py
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
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else 0.0

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
    siteNum = np.asarray(a.countData.siteNum, dtype=int).ravel()
    return Sn, sc, cn, newT, endT, siteNum

def barcodes(Sn, newT, endT, rows):
    nb = Sn.shape[1]; f0 = np.clip((newT-1).astype(int), 0, nb-1); f1 = np.clip(endT.astype(int), 1, nb)
    B = np.zeros((len(rows), Sn.shape[0]), dtype=np.float32)
    for i, e in enumerate(rows):
        lo, hi = f0[e], max(f0[e]+1, f1[e]); B[i] = Sn[:, lo:hi].mean(1)
    return B

def analyze(S):
    Sn, sc, cn, newT, endT, siteNum = load_session(S)
    rs = sc.sum(1); cache = np.where(rs > 0)[0]; retr = np.where(rs < 0)[0]
    if len(cache) < 5 or len(retr) < 5: return None
    Bc = barcodes(Sn, newT, endT, cache); Br = barcodes(Sn, newT, endT, retr)
    csite, rsite = siteNum[cache], siteNum[retr]; cnum, rnum = cn[cache], cn[retr]
    own_s, oth_s, hits, chances, loads = [], [], [], [], []
    hit_by_load = []   # (N_candidates, hit 0/1)
    for i in range(len(retr)):
        if rnum[i] <= 0: continue
        cand = np.where(csite == rsite[i])[0]                 # caches at the retrieval's site
        if len(cand) < 2: continue
        jstar = cand[cnum[cand] == rnum[i]]
        if len(jstar) == 0: continue
        jstar = jstar[0]
        scores = {}
        for k in cand:
            others = cand[cand != k]
            T = Bc[others].mean(0)                            # leave-one-out site template
            scores[k] = corr(Br[i] - T, Bc[k] - T)
        own = scores[jstar]
        oth = np.mean([scores[k] for k in cand if k != jstar])
        own_s.append(own); oth_s.append(oth)
        best = max(scores, key=scores.get)
        hit = int(best == jstar); hits.append(hit); chances.append(1.0/len(cand))
        loads.append(len(cand)); hit_by_load.append((int(len(cand)), hit))
    if not own_s: return None
    return {"session": os.path.basename(S), "n_cache": int(len(cache)), "n_retr": int(len(retr)),
            "n_test": len(own_s), "own": float(np.mean(own_s)), "oth": float(np.mean(oth_s)),
            "hit": float(np.mean(hits)), "chance": float(np.mean(chances)),
            "hit_by_load": hit_by_load}

results = []
for i, S in enumerate(SESS):
    try:
        r = analyze(S)
        if r:
            results.append(r)
            print("[%2d/%d] %-22s n=%3d own %.3f oth %.3f (Δ%+.3f) | read acc %.3f vs chance %.3f"
                  % (i+1, len(SESS), r["session"], r["n_test"], r["own"], r["oth"],
                     r["own"]-r["oth"], r["hit"], r["chance"]))
        else:
            print("[%2d/%d] %-22s skipped" % (i+1, len(SESS), os.path.basename(S)))
    except Exception as e:
        import traceback; print("[%2d/%d] ERROR %r" % (i+1, os.path.basename(S), e)); traceback.print_exc()

json.dump([{k: v for k, v in r.items() if k != "hit_by_load"} for r in results],
          open(os.path.join(OUT, "event_specificity.json"), "w"), indent=1)

def arr(k): return np.array([r[k] for r in results if k in r])
from scipy.stats import wilcoxon
own, oth = arr("own"), arr("oth"); hit, ch = arr("hit"), arr("chance")
print("\n==============  SITE-TEMPLATE-REMOVED EVENT SPECIFICITY, %d SESSIONS  ==============" % len(results))
d = own - oth; p = wilcoxon(d).pvalue
print("residual reactivation: own %.4f  other-same-site %.4f  (Δ=%+.4f, %d/%d own>oth, p=%.1e)"
      % (own.mean(), oth.mean(), d.mean(), int((d>0).sum()), len(d), p))
print("  [KV read => own > other-same-site AFTER removing the site template]")
dh = hit - ch; ph = wilcoxon(dh).pvalue
print("read accuracy (own is argmax): %.4f  vs chance %.4f  (Δ=%+.4f, %d/%d above, p=%.1e)"
      % (hit.mean(), ch.mean(), dh.mean(), int((dh>0).sum()), len(dh), ph))

# crosstalk: read accuracy vs N competing same-site caches
HL = np.array([pt for r in results for pt in r["hit_by_load"]])   # (n, 2): N, hit
if len(HL) > 30:
    N, H = HL[:,0], HL[:,1]
    print("\ncrosstalk — read accuracy vs # competing same-site caches (n=%d retrievals):" % len(HL))
    for lo_, hi_ in [(2,2),(3,3),(4,5),(6,9),(10,999)]:
        m = (N>=lo_)&(N<=hi_)
        if m.sum(): print("  N %2d-%-3d : read acc %.3f  chance %.3f  lift %+.3f (n=%d)"
                          % (lo_, min(hi_,int(N.max())), H[m].mean(), (1.0/N[m]).mean(),
                             H[m].mean()-(1.0/N[m]).mean(), int(m.sum())))

# figure
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
ax[0].scatter(oth, own, s=18); L=[min(oth.min(),own.min()), max(oth.max(),own.max())]
ax[0].plot(L,L,"k--",lw=1); ax[0].set_xlabel("other same-site (residual corr)")
ax[0].set_ylabel("own cache (residual corr)"); ax[0].set_title("event specificity (site template removed)")
if len(HL) > 30:
    bins = np.arange(2, min(int(N.max()),10)+1)
    acc = [H[N==b].mean() if (N==b).sum() else np.nan for b in bins]
    chc = [1.0/b for b in bins]
    ax[1].plot(bins, acc, "o-", label="read accuracy"); ax[1].plot(bins, chc, "k--", label="chance 1/N")
    ax[1].set_xlabel("# competing same-site caches (N)"); ax[1].set_ylabel("own-is-argmax accuracy")
    ax[1].set_title("crosstalk vs load"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "event_specificity.png"), dpi=110)
print("\nsaved out/event_specificity.json and event_specificity.png")
