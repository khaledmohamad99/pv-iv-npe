"""ITM module case study (36-cell modules, outdoor, one curve per condition, measured G and module temperature).
Module curves are mapped to an equivalent cell (V/36; current rescaled per panel by k = 8.8 A / clean-panel Isc at
1000 W m-2) and fed to the NPE at the 925 W m-2 slot with the measured module temperature (irradiance mismatch is
covered by the gain nuisance). Misspecification score: posterior-predictive RMSE (% Isc) after profiling the
per-curve nuisances; partial shading (bypass-diode steps) lies outside the cell model and should be flagged."""
import numpy as np, pandas as pd, torch, json
from scipy.optimize import least_squares
from load_data import load_itm
from simulate_bank import GRID, GS, phi_to_theta, temp_scale, GAIN_LO, GAIN_HI, PHI_NAMES
from physics import iv_curve
from npe import load

NC = 36; S_IN = 4.0
meta, curves = load_itm(); model = load()


def to_cell_grid(V, I):
    v = V / NC; o = np.argsort(v); v, i = v[o], I[o]
    edges = np.r_[GRID[0] - 0.0036, (GRID[1:] + GRID[:-1]) / 2, GRID[-1] + 0.0036]
    idx = np.digitize(v, edges) - 1; out = np.full(len(GRID), np.nan)
    for k in range(len(GRID)):
        m = idx == k
        if m.sum() > 3: out[k] = np.median(i[m])
    good = np.isfinite(out)
    return np.interp(GRID, GRID[good], out[good], right=-2.0)


rows, X, Tm = [], [], []
for (mod, pan), d in meta.groupby(['module', 'panel']):
    kc = d.index[d.cond == 'Clean'][0]; Vc, Ic = curves[kc]
    isc_c = np.interp(0, Vc[np.argsort(Vc)], Ic[np.argsort(Vc)]) * 1000 / d.loc[kc, 'G']
    k = 8.8 / isc_c
    for i, r in d.iterrows():
        V, I = curves[i]; c = to_cell_grid(V, I) * k
        x = np.zeros((5, len(GRID)), np.float32); x[0] = np.clip(c * 1000 / GS[0], -2, 12)
        X.append(x); Tm.append(r['T']); rows.append(dict(module=mod, panel=pan, cond=r.cond, G=r.G, T=r['T'], k=k))
X = np.stack(X); Tm = np.array(Tm, np.float32); M = np.zeros((len(X), 5), bool); M[:, 0] = True
S = np.full((len(X), 5), S_IN, np.float32)
post = model.sample(torch.tensor(X), torch.tensor(M), torch.tensor(S), torch.tensor(Tm), 600)


def one(phi, g, dT):
    T = phi[7] + dT; s1, s2, vt = temp_scale(np.array([T])); th = phi_to_theta(phi[None])
    th['I01'] = th['I01'] * s1; th['I02'] = th['I02'] * s2; th['Iph_ref'] = th['Iph_ref'] * (1 + 0.0005 * (T - 25))
    return iv_curve(GRID, np.array([GS[0] * g]), th, vt=vt[:, None])[0] * 1000 / GS[0]


rng = np.random.default_rng(0); out = []
for n, r in enumerate(rows):
    o = X[n, 0]; kk = (o > -1.0)
    e = []
    for phi in post[n][rng.choice(len(post[n]), 25, replace=False)]:
        f = least_squares(lambda q: (one(phi, q[0], q[1])[kk] - o[kk]) / o[0], [1.0, 0.0], bounds=([GAIN_LO, -5], [GAIN_HI, 5]), max_nfev=60)
        e.append(100 * np.sqrt(np.mean(f.fun ** 2)))
    J = {p: i for i, p in enumerate(PHI_NAMES)}
    out.append(dict(**r, ppc_rmse_pct=float(np.median(e)), Rs_med=float(10 ** np.median(post[n][:, J['log_Rs']])),
                    Rsh_med=float(10 ** np.median(post[n][:, J['log_Rsh_ref']])), Iph_med=float(np.median(post[n][:, J['Iph_ref']]))))
D = pd.DataFrame(out); D.to_csv('../results/itm_case.csv', index=False)
print(D[['module', 'panel', 'cond', 'G', 'T', 'ppc_rmse_pct', 'Rs_med', 'Rsh_med', 'Iph_med']].round(3).to_string(index=False))
# misspecification flag: threshold = max PPC RMSE among clean curves x 1.5
thr = 1.5 * D[D.cond == 'Clean'].ppc_rmse_pct.max()
D['flag'] = D.ppc_rmse_pct > thr
print('threshold %.3f%%' % thr); print(pd.crosstab(D.cond, D.flag))
json.dump(dict(threshold=thr, flagged=D.groupby('cond').flag.mean().to_dict()), open('../results/itm_flag.json', 'w'))
