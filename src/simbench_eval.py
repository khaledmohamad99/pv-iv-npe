import numpy as np, pandas as pd, torch, glob, json, time
from npe import load
from simulate_bank import PHI_NAMES, LO, HI
m = load(); rows = []; times = []
for f in sorted(glob.glob('../results/simbench_case*.npz')):
    c = int(f.split('case')[1].split('.')[0]); d = np.load(f)
    ch = d['chain'][:, :8]; X = d['X'][None]; s = d['s'][None].astype(np.float32); tm = np.array([d['Tm']], np.float32); tr = d['truth'][:8]
    t = time.time(); P = m.sample(torch.tensor(X), torch.ones(1, 5, dtype=torch.bool), torch.tensor(s), torch.tensor(tm), 4000)[0]; times.append(time.time() - t)
    for k, p in enumerate(PHI_NAMES):
        a, b = ch[:, k], P[:, k]
        qa, qb = np.quantile(a, [.05, .5, .95]), np.quantile(b, [.05, .5, .95])
        rows.append(dict(case=c, noise=float(d['s'][0]), param=p, smd=abs(b.mean() - a.mean()) / (a.std() + 1e-12),
                         width_ratio=(qb[2] - qb[0]) / (qa[2] - qa[0] + 1e-12), truth_in_mcmc90=float(qa[0] <= tr[k] <= qa[2]),
                         truth_in_npe90=float(qb[0] <= tr[k] <= qb[2]), err_mcmc=abs(qa[1] - tr[k]) / (HI[k] - LO[k]), err_npe=abs(qb[1] - tr[k]) / (HI[k] - LO[k])))
D = pd.DataFrame(rows); D.to_csv('../results/simbench_compare.csv', index=False)
print(D.groupby('param')[['smd', 'width_ratio', 'truth_in_mcmc90', 'truth_in_npe90', 'err_mcmc', 'err_npe']].mean().round(3))
print('overall: median SMD %.2f | median width ratio NPE/MCMC %.2f | truth in 90%%: MCMC %.2f NPE %.2f | NPE sec %.2f' % (
    D.smd.median(), D.width_ratio.median(), D.truth_in_mcmc90.mean(), D.truth_in_npe90.mean(), np.mean(times)))
print(D.groupby('noise')[['smd', 'width_ratio']].median().round(2))
