"""Severity-tracking statistics: exact permutation p-values, posterior-propagated 90% intervals of Spearman rho,
and family-level stratified permutation tests (mean rho across samples)."""
import numpy as np, pandas as pd, itertools, json, math
from scipy.stats import spearmanr, rankdata
from simulate_bank import PHI_NAMES
St = pd.read_csv('../results/v3_real_states.csv'); P = np.load('../results/v3_real_posteriors.npz')['full']
J = {p: i for i, p in enumerate(PHI_NAMES)}; rng = np.random.default_rng(0)
def rho(x, y): return spearmanr(x, y).statistic
def groups(fam):
    for smp, dd in St[St.sample_mode == fam].groupby('sample'):
        if fam == 'interconnect' and smp in ('E_2', 'E_3', 'F_2', 'F_3'):
            dd = pd.concat([St[(St['sample'] == smp[0] + '_1') & (St.meas == 'ref')], dd])
        if dd.severity.nunique() > 2: yield smp, dd
def perm_p(sev, val, sign):
    obs = sign * rho(sev, val); n = len(sev)
    perms = itertools.permutations(range(n)) if math.factorial(n) <= 40320 else (rng.permutation(n) for _ in range(40000))
    cnt = tot = 0
    for p in perms:
        tot += 1; cnt += sign * rho(sev, np.asarray(val)[list(p)]) >= obs - 1e-12
    return cnt / tot
rows, fam_res = [], {}
for fam, par, sign in [('interconnect', 'log_Rs', 1), ('PID', 'log_Rsh_ref', -1), ('crack', 'log_Rsh_ref', -1), ('crack', 'logit_fc', 1)]:
    G = list(groups(fam)); stats = []
    for smp, dd in G:
        idx = dd.index.values; sev = dd.severity.values; med = dd[f'med_{par}'].values
        draws = P[idx][:, rng.integers(0, P.shape[1], 2000), J[par]]      # (n_states, 2000)
        rd = np.array([rho(sev, draws[:, k]) for k in range(draws.shape[1])])
        r = rho(sev, med)
        rows.append(dict(family=fam, param=par, sample=smp, n=len(idx), rho=round(r, 2), p_perm=perm_p(sev, med, sign),
                         rho_post_lo=round(float(np.nanquantile(rd, .05)), 2), rho_post_hi=round(float(np.nanquantile(rd, .95)), 2)))
        stats.append((sev, med))
    # family-level stratified permutation: statistic = mean signed rho; permute within each sample
    obs = np.mean([sign * rho(s, m) for s, m in stats]); B = 20000; cnt = 0
    for _ in range(B):
        cnt += np.mean([sign * rho(s, rng.permutation(m)) for s, m in stats]) >= obs - 1e-12
    fam_res[f'{fam}:{par}'] = dict(mean_signed_rho=round(float(obs), 3), p_stratified=(cnt + 1) / (B + 1), n_samples=len(stats))
D = pd.DataFrame(rows); D.to_csv('../results/severity_stats.csv', index=False)
print(D.to_string(index=False)); print(json.dumps(fam_res, indent=1))
json.dump(fam_res, open('../results/severity_family_tests.json', 'w'), indent=1)
