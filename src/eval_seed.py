"""Seed-variance evaluation with an identical protocol for each trained NPE (seed 0 = main model)."""
import os, gc, ctypes, sys, json, itertools
os.environ.setdefault('MALLOC_ARENA_MAX', '2')
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from npe import load, random_masks
from simulate_bank import PHI_NAMES
torch.set_num_threads(2); libc = ctypes.CDLL('libc.so.6'); J = {p: i for i, p in enumerate(PHI_NAMES)}
seed = int(sys.argv[1]); path = '../results/npe_v3.pt' if seed == 0 else f'../results/npe_v3_seed{seed}.pt'
m = load(path); rng = np.random.default_rng(100 + seed)
def post(X, M, s, tm, n):
    out = []; bs = max(1, 8000 // n)
    for i in range(0, len(X), bs):
        k = slice(i, i + bs); b = len(X[k])
        S = torch.tensor(np.broadcast_to(np.asarray(s, np.float32), (len(X), 5))[k].copy())
        T = torch.tensor(np.broadcast_to(np.asarray(tm, np.float32), (len(X),))[k].copy())
        out.append(m.sample(torch.tensor(X[k]), torch.tensor(M[k]), S, T, n).astype(np.float32)); gc.collect(); libc.malloc_trim(0)
    return np.concatenate(out)
res = dict(seed=seed)
d = np.load('../data/bank_test_v3.npz'); n = 400
Xs, Ps, Ss, Ts = d['X'][:n], d['phi'][:n], d['S'][:n], d['Tm'][:n]; Ms = random_masks(n, rng); Sm = post(Xs, Ms, Ss, Ts, 800)
cov = {a: [] for a in [0.5, 0.8, 0.9, 0.95]}
for p, j in J.items():
    for a in cov:
        lo, hi = np.quantile(Sm[:, :, j], [(1 - a) / 2, (1 + a) / 2], axis=1); cov[a].append(float(np.mean((Ps[:, j] >= lo) & (Ps[:, j] <= hi))))
res['coverage_range'] = {str(a): [min(v), max(v)] for a, v in cov.items()}
St = pd.read_csv('../results/v3_real_states.csv'); from real_obs import build; _, Xr = build('front'); N = len(Xr)
full = post(Xr, np.ones((N, 5), bool), 1.0, 25.0, 1000)
np.savez_compressed(f'../results/v3s{seed}_post.npz', full=full)
single = np.stack([post(Xr, np.eye(5, dtype=bool)[[g] * N], 1.0, 25.0, 400) for g in range(5)], 1)
z = []
for i in range(N):
    for a, b in itertools.combinations(range(5), 2):
        for p in ['log_Rs', 'log_Rsh_ref', 'log_I01', 'logit_fc']:
            x, y = single[i, a, :, J[p]], single[i, b, :, J[p]]; z.append((np.median(x) - np.median(y)) / np.sqrt(x.var() + y.var() + 1e-12))
z = np.abs(np.array(z)); res['z_share_lt2'] = float((z < 2).mean()); res['z_median'] = float(np.median(z))
res['consistency'] = {p: float(np.median(np.std(np.median(single[:, :, :, J[p]], 2), 1))) for p in ['log_Rs', 'log_Rsh_ref', 'log_I01', 'log_I02']}
for p in PHI_NAMES: St[f'med_{p}'] = np.median(full[:, :, J[p]], 1)
tr = {}
for fam, par in [('interconnect', 'log_Rs'), ('PID', 'log_Rsh_ref'), ('crack', 'log_Rsh_ref'), ('crack', 'logit_fc')]:
    rr = []
    for smp, dd in St[St.sample_mode == fam].groupby('sample'):
        if fam == 'interconnect' and smp in ('E_2', 'E_3', 'F_2', 'F_3'):
            dd = pd.concat([St[(St['sample'] == smp[0] + '_1') & (St.meas == 'ref')], dd])
        if dd.severity.nunique() > 2: rr.append(round(float(spearmanr(dd.severity, dd[f'med_{par}']).statistic), 2))
    tr[f'{fam}:{par}'] = rr
res['trends'] = tr
res['ci90_width'] = {p: float(np.median(np.quantile(full[:, :, J[p]], .95, 1) - np.quantile(full[:, :, J[p]], .05, 1))) for p in ['log_Rs', 'log_I01', 'log_Rsh_ref', 'T_cell']}
json.dump(res, open(f'../results/seed_eval_{seed}.json', 'w'), indent=1); print(json.dumps(res))
