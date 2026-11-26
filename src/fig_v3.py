import numpy as np, pandas as pd, json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from simulate_bank import PHI_NAMES
from real_obs import build
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9, 'savefig.facecolor': 'white'})
St, Xr = build('front')
P = np.load('../results/v3_real_posteriors.npz'); full, single = P['full'], P['single']
j = {p: PHI_NAMES.index(p) for p in PHI_NAMES}

# ---------- Fig: calibration + accuracy vs number of curves (simulation)
C = pd.read_csv('../results/v3_sim_coverage.csv', index_col=0); A = pd.read_csv('../results/npe_sim_accuracy_vs_k_v2.csv')
lab = {'Iph_ref': '$I_{ph}$', 'log_I01': '$I_{01}$', 'log_I02': '$I_{02}$', 'log_Rs': '$R_s$', 'log_Rsh_ref': '$R_{sh}$', 'logit_fc': '$f_c$', 'log_Rc': '$R_c$', 'T_cell': '$T_{cell}$'}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
lv = [float(c) for c in C.columns]
for p in C.index: ax[0].plot(lv, C.loc[p].values, 'o-', ms=4, label=lab[p])
ax[0].plot([0.45, 1], [0.45, 1], 'k--', lw=1); ax[0].set_xlabel('Nominal credible level'); ax[0].set_ylabel('Empirical coverage')
ax[0].set_title('(a) Calibration on held-out simulations'); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
for p in PHI_NAMES:
    d = A[A.param == p]; ax[1].plot(d.k, d.nrmse, 'o-', ms=4, label=lab[p])
ax[1].set_xlabel('Number of irradiance levels in the observation'); ax[1].set_ylabel('NRMSE of posterior median'); ax[1].set_xticks(range(1, 6))
ax[1].set_title('(b) Accuracy vs measurement protocol'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('../figs/fig_v3_calibration.png', dpi=300)

# ---------- Fig: severity tracking on real cells with 90% credible intervals
def ci(arr): return np.median(arr), np.quantile(arr, .05), np.quantile(arr, .95)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for s in ['F_1', 'F_2', 'F_3']:
    rows = St.index[(St['sample'] == s) | ((St['sample'] == 'F_1') & (St.meas == 'ref'))]
    rows = [i for i in rows if St.loc[i, 'sample'] == s or St.loc[i, 'meas'] == 'ref']
    xs = St.loc[rows, 'severity'].values; o = np.argsort(xs)
    m = np.array([ci(1e3 * 10 ** full[i, :, j['log_Rs']]) for i in rows])[o]
    ax[0].errorbar(xs[o], m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='o-', ms=4, capsize=3, label=s)
ax[0].set_yscale('log'); ax[0].set_xlabel('Fraction of ribbon connections cut (ground truth)'); ax[0].set_ylabel('$R_s$ posterior (mΩ), 90% CI')
ax[0].set_title('(a) Interconnect failures'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which='both')
for s in ['A_1', 'A_2', 'B_1', 'B_2', 'C_1']:
    rows = St.index[St['sample'] == s]; xs = St.loc[rows, 'severity'].values; o = np.argsort(xs)
    m = np.array([ci(10 ** full[i, :, j['log_Rsh_ref']]) for i in rows])[o]
    ax[1].errorbar(xs[o], m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='o-', ms=4, capsize=3, label=s, alpha=1 if s[0] == 'A' else .45)
ax[1].set_yscale('log'); ax[1].set_xlabel('PID stress stage (ground truth)'); ax[1].set_ylabel('$R_{sh}$ posterior (Ω, at 1000 W m$^{-2}$), 90% CI')
ax[1].set_title('(b) PID (B, C: PID-resistant cells)'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which='both')
for s in ['G_1', 'G_2', 'G_3', 'H_1', 'H_2', 'H_3']:
    rows = St.index[St['sample'] == s]; xs = St.loc[rows, 'severity'].values; o = np.argsort(xs)
    m = np.array([ci(100 / (1 + np.exp(-full[i, :, j['logit_fc']]))) for i in rows])[o]
    ax[2].errorbar(xs[o] + 0.03 * (int(s[-1]) - 2), m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt='o-', ms=4, capsize=3, label=s)
ax[2].set_xlabel('Crack stage (ground truth)'); ax[2].set_ylabel('Cracked-area fraction $f_c$ (%), 90% CI'); ax[2].set_xticks([0, 1, 2, 3])
ax[2].set_title('(c) Cell cracks'); ax[2].legend(fontsize=7, ncol=2); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig('../figs/fig_v3_real_severity.png', dpi=300)

