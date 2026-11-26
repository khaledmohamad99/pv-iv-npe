"""v3 evaluation (measured-temperature + noise-level inputs).
Real-data noise level s is selected by the cross-irradiance calibration test (z-scores of single-curve posteriors of
the same cell state should be ~N(0,1)); then severity tracking, interval widths, information by subset, temperature
sensitivity (Tm = 25 vs 35 C) and simulation calibration."""
import numpy as np, pandas as pd, torch, itertools, json, gc, ctypes
from scipy.stats import spearmanr
from npe import load, random_masks
from simulate_bank import PHI_NAMES, LO, HI, GS
from real_obs import build

_libc = ctypes.CDLL('libc.so.6'); model = load(); J = {p: i for i, p in enumerate(PHI_NAMES)}


def post(X, M, s, tm, n=1000):
    out = []; bs = max(1, 8000 // n)
    for i in range(0, len(X), bs):
        k = slice(i, i + bs); b = len(X[k])
        S = torch.tensor(np.broadcast_to(np.asarray(s, np.float32), (len(X), 5))[k].copy())
        T = torch.tensor(np.broadcast_to(np.asarray(tm, np.float32), (len(X),))[k].copy())
        out.append(model.sample(torch.tensor(X[k]), torch.tensor(M[k]), S, T, n).astype(np.float32))
        gc.collect(); _libc.malloc_trim(0)
    return np.concatenate(out)


res = {}
# ---- simulation calibration
d = np.load('../data/bank_test_v3.npz'); rng = np.random.default_rng(7); n = 800
Xs, Ps, Ss, Ts = d['X'][:n], d['phi'][:n], d['S'][:n], d['Tm'][:n]; Ms = random_masks(n, rng)
Sm = post(Xs, Ms, Ss, Ts, 800)
cov = {}
for p, j in J.items():
    for a in [0.5, 0.8, 0.9, 0.95]:
        lo, hi = np.quantile(Sm[:, :, j], [(1 - a) / 2, (1 + a) / 2], axis=1); cov[(p, a)] = float(np.mean((Ps[:, j] >= lo) & (Ps[:, j] <= hi)))
pd.Series(cov).unstack().to_csv('../results/v3_sim_coverage.csv')

# ---- real data
St, Xr = build('front'); N = len(St); INTR = ['log_Rs', 'log_Rsh_ref', 'log_I01', 'logit_fc']


def consistency(s, tm=25.0):
    single = np.stack([post(Xr, np.eye(5, dtype=bool)[[g] * N], s, tm, 600) for g in range(5)], 1)
    z = []
    for i in range(N):
        for a, b in itertools.combinations(range(5), 2):
            for p in INTR:
                x, y = single[i, a, :, J[p]], single[i, b, :, J[p]]
                z.append(dict(s=s, param=p, z=(np.median(x) - np.median(y)) / np.sqrt(x.var() + y.var() + 1e-12)))
    return pd.DataFrame(z), single


calib = []; singles = {}
for s in [1.0, 2.0, 4.0, 8.0]:
    Z, single = consistency(s); singles[s] = single
    calib.append(dict(s=s, share_lt2=float((Z.z.abs() < 2).mean()), med_absz=float(Z.z.abs().median())))
    print(calib[-1], flush=True)
C = pd.DataFrame(calib); C.to_csv('../results/v3_real_noise_selection.csv', index=False)
# choose the smallest s whose |z| behaviour is consistent with calibration (share |z|<2 >= 0.95)
ok = C[C.share_lt2 >= 0.95]; s_star = float(ok.s.min()) if len(ok) else float(C.s.max()); res['s_star'] = s_star
print('selected s =', s_star, flush=True)

full = post(Xr, np.ones((N, 5), bool), s_star, 25.0, 2000)
full35 = post(Xr, np.ones((N, 5), bool), s_star, 35.0, 1000)
np.savez_compressed('../results/v3_real_posteriors.npz', full=full, full35=full35, single=singles[s_star])
for p in PHI_NAMES:
    St[f'med_{p}'] = np.median(full[:, :, J[p]], 1)
    St[f'lo_{p}'], St[f'hi_{p}'] = np.quantile(full[:, :, J[p]], [.05, .95], axis=1)
    St[f'med35_{p}'] = np.median(full35[:, :, J[p]], 1)
St.to_csv('../results/v3_real_states.csv', index=False)


def trends(col):
    out = {}
    for fam, par, sign in [('interconnect', 'log_Rs', 1), ('PID', 'log_Rsh_ref', -1), ('crack', 'logit_fc', 1), ('crack', 'log_Rsh_ref', -1)]:
        rr = []
        for smp, dd in St[St.sample_mode == fam].groupby('sample'):
            if fam == 'interconnect' and smp in ('E_2', 'E_3', 'F_2', 'F_3'):
                dd = pd.concat([St[(St['sample'] == smp[0] + '_1') & (St.meas == 'ref')], dd])
            if dd.severity.nunique() > 2: rr.append(round(float(spearmanr(dd.severity, dd[f'{col}_{par}']).statistic), 2))
        out[f'{fam}:{par}'] = dict(median_rho=float(np.median(rr)), per_sample=rr)
    return out


res['trends_T25'] = trends('med'); res['trends_T35'] = trends('med35')
res['ci90_width_real'] = {p: float(np.median(St[f'hi_{p}'] - St[f'lo_{p}'])) for p in PHI_NAMES}
res['ci90_width_real_rel_prior'] = {p: res['ci90_width_real'][p] / float(HI[J[p]] - LO[J[p]]) for p in PHI_NAMES}
# information by irradiance subset (normalised SD per parameter)
info = []
for k in range(1, 6):
    for sub in itertools.combinations(range(5), k):
        M = np.zeros((N, 5), bool); M[:, list(sub)] = True; Sk = post(Xr, M, s_star, 25.0, 300)
        info.append(dict(k=k, subset='+'.join(str(int(GS[x])) for x in sub),
                         **{f'nsd_{p}': float(np.mean(Sk[:, :, J[p]].std(1)) / (HI[J[p]] - LO[J[p]])) for p in INTR}))
pd.DataFrame(info).to_csv('../results/v3_real_info_subsets.csv', index=False)
json.dump(res, open('../results/v3_summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1)); print('done')
