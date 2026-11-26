"""Field-grade noise robustness: inject the empirical noise model at scale s_add into real Garaj curves and re-run NPE
with the matching noise-level input; report severity Spearman and 90% CI widths vs the clean (lab) result."""
import numpy as np, pandas as pd, torch, json, gc, ctypes
from scipy.stats import spearmanr
from npe import load
from real_obs import build
from simulate_bank import GRID, GS, PHI_NAMES
lib = ctypes.CDLL('libc.so.6'); model = load(); St, X = build('front'); N = len(St); J = {p: i for i, p in enumerate(PHI_NAMES)}
def inject(X, s, rng):
    Y = X.copy()
    for n in range(len(X)):
        for j in range(5):
            I = X[n, j] * GS[j] / 1000; d = np.gradient(I, GRID)
            I2 = I + d * rng.normal(0, 0.33e-3 * s) + rng.normal(0, 3.7e-3 * s, len(GRID))
            Y[n, j] = np.clip(I2 * 1000 / GS[j], -2, 12)
    return Y
def post(X, s, n=800):
    out = []
    for i in range(0, len(X), 8):
        b = len(X[i:i + 8])
        out.append(model.sample(torch.tensor(X[i:i + 8]), torch.ones(b, 5, dtype=torch.bool), torch.full((b, 5), float(s)), torch.full((b,), 25.0), n))
        gc.collect(); lib.malloc_trim(0)
    return np.concatenate(out)
def summarise(P):
    S = St.copy()
    for p in PHI_NAMES: S[p] = np.median(P[:, :, J[p]], 1)
    r = {}
    for fam, par in [('interconnect', 'log_Rs'), ('PID', 'log_Rsh_ref'), ('crack', 'log_Rsh_ref'), ('crack', 'logit_fc')]:
        rr = []
        for smp, d in S[S.sample_mode == fam].groupby('sample'):
            if fam == 'interconnect' and smp in ('E_2', 'E_3', 'F_2', 'F_3'): d = pd.concat([S[(S['sample'] == smp[0] + '_1') & (S.meas == 'ref')], d])
            if d.severity.nunique() > 2: rr.append(round(float(spearmanr(d.severity, d[par]).statistic), 2))
        r[f'{fam}:{par}'] = rr
    w = {p: float(np.median(np.quantile(P[:, :, J[p]], .95, 1) - np.quantile(P[:, :, J[p]], .05, 1))) for p in ['log_Rs', 'log_Rsh_ref', 'log_I01', 'logit_fc']}
    return r, w
res = {}
for s in [5.0, 15.0]:
    rng = np.random.default_rng(int(s)); Xn = inject(X, s, rng); r, w = summarise(post(Xn, np.sqrt(1 + s * s)))
    res[f's={s}'] = dict(trends=r, ci90=w); print(s, json.dumps(res[f's={s}']), flush=True)
json.dump(res, open('../results/field_noise.json', 'w'), indent=1)
