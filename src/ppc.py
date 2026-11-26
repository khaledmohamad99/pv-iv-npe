"""Posterior predictive check on real cells.
For NPE samples (physical parameters only; nuisances were marginalised during training) we profile the per-curve
nuisances (irradiance gain, temperature offset) by bounded least squares, then report the RMSE (% of Isc) of each
sample. For MCMC samples the nuisances are part of the chain, so the RMSE is computed directly."""
import numpy as np, pandas as pd, sys
from scipy.optimize import least_squares
from real_obs import build
from simulate_bank import GRID, GS, phi_to_theta, temp_scale, GAIN_LO, GAIN_HI
from physics import iv_curve
from mcmc_v3 import curves

St, Xr = build('front')


def one_curve(phi, G, g, dT):
    T = phi[7] + dT; s1, s2, vt = temp_scale(np.array([T])); th = phi_to_theta(phi[None])
    th['I01'] = th['I01'] * s1; th['I02'] = th['I02'] * s2; th['Iph_ref'] = th['Iph_ref'] * (1 + 0.0005 * (T - 25))
    return iv_curve(GRID, np.array([G * g]), th, vt=vt[:, None])[0] * 1000 / G


def rmse_npe(phi, i):
    e = []
    for j, G in enumerate(GS):
        o = Xr[i, j]; k = o > -1
        r = least_squares(lambda q: (one_curve(phi, G, q[0], q[1])[k] - o[k]) / o[0], [1.0, 0.0],
                          bounds=([GAIN_LO, -4], [GAIN_HI, 4]), max_nfev=60)
        e.append(np.sqrt(np.mean(r.fun ** 2)))
    return 100 * np.mean(e)


def rmse_mcmc(q, i):
    e = []
    for j, I in enumerate(curves(q[None])):
        o = Xr[i, j]; k = o > -1; Ii = I[0] * 1000 / GS[j]
        e.append(np.sqrt(np.mean(((Ii - o)[k] / o[0]) ** 2)))
    return 100 * np.mean(e)


if __name__ == '__main__':
    ids = [int(x) for x in sys.argv[1].split(",")]; P = np.load('../results/npe_real_posteriors_v2.npz')['full']
    rng = np.random.default_rng(0); rows = []
    for i in ids:
        ch = np.load(f'../results/mcmc_v3_chain_{i}.npy')
        for q in ch[rng.choice(len(ch), 30, replace=False)]:
            rows.append(dict(state=i, method='MCMC', rmse=rmse_mcmc(q, i)))
        for phi in P[i][rng.choice(len(P[i]), 30, replace=False)]:
            rows.append(dict(state=i, method='NPE', rmse=rmse_npe(phi, i)))
        print(i, flush=True)
    D = pd.DataFrame(rows); D.to_csv('../results/ppc_v3.csv', index=False)
    print(D.groupby(['state', 'method']).rmse.median().unstack().round(3))
