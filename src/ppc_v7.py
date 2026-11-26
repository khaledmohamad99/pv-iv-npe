import numpy as np, pandas as pd, json
from ppc import rmse_npe, Xr
from mcmc_v5 import curves
from simulate_bank import GS, PHI_NAMES, LO, HI
def rmse_mcmc(q, i):
    e = []
    for j, I in enumerate(curves(q[None])):
        o = Xr[i, j]; k = o > -1; Ii = I[0] * 1000 / GS[j]; e.append(np.sqrt(np.mean(((Ii - o)[k] / o[0]) ** 2)))
    return 100 * np.mean(e)
P = np.load('../results/v3_real_posteriors.npz')['full']; rng = np.random.default_rng(0); rows = []; summ = []
for i in [5, 33, 47]:
    for tag in ['v7n', 'v7d']:
        ch = np.load(f'../results/mcmc_{tag}_chain_{i}.npy')
        for q in ch[rng.choice(len(ch), 30, replace=False)]: rows.append(dict(state=i, method='MCMC_' + tag, rmse=rmse_mcmc(q, i)))
        for k, p in enumerate(PHI_NAMES):
            a = ch[:, k]; b = P[i, :, k]; lo, hi = np.quantile(b, [.05, .95])
            summ.append(dict(state=i, tag=tag, param=p, mcmc_med=float(np.median(a)), npe_med=float(np.median(b)),
                             w_mcmc=float(np.quantile(a, .95) - np.quantile(a, .05)), w_npe=float(hi - lo), in90=float(lo <= np.median(a) <= hi),
                             d_rel=float(abs(np.median(a) - np.median(b)) / (HI[k] - LO[k]))))
    for phi in P[i][rng.choice(P.shape[1], 30, replace=False)]: rows.append(dict(state=i, method='NPE', rmse=rmse_npe(phi, i)))
    print(i, flush=True)
D = pd.DataFrame(rows); D.to_csv('../results/ppc_v7.csv', index=False); S = pd.DataFrame(summ); S.to_csv('../results/mcmc_v7_vs_npe.csv', index=False)
print(D.groupby(['state', 'method']).rmse.median().unstack().round(2))
for tag in ['v7n', 'v7d']:
    s = S[S.tag == tag]; print(tag, 'in90', round(s.in90.mean(), 2), 'med d_rel', round(s.d_rel.median(), 3))
    print(s[s.param.isin(['T_cell', 'log_Rs', 'log_Rsh_ref', 'log_I02'])][['state', 'param', 'mcmc_med', 'npe_med', 'w_mcmc', 'w_npe']].round(3).to_string(index=False))
