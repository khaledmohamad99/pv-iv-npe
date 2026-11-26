"""v3 figures: irradiance consistency, value of irradiance levels, well-specified NPE vs exact-MCMC benchmark."""
import os, gc, ctypes
os.environ.setdefault('MALLOC_ARENA_MAX', '2')
import numpy as np, pandas as pd, glob, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from simulate_bank import PHI_NAMES, LO, HI
from real_obs import build
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'font.size': 9,
                     'font.family': 'DejaVu Sans'})
lab = {'Iph_ref': '$I_{ph}$', 'log_I01': '$\\log_{10} I_{01}$', 'log_I02': '$\\log_{10} I_{02}$', 'log_Rs': '$\\log_{10} R_s$',
       'log_Rsh_ref': '$\\log_{10} R_{sh}$', 'logit_fc': 'logit $f_c$', 'log_Rc': '$\\log_{10} R_c$', 'T_cell': '$T_{cell}$ (°C)'}
short = {'Iph_ref': '$I_{ph}$', 'log_I01': '$I_{01}$', 'log_I02': '$I_{02}$', 'log_Rs': '$R_s$', 'log_Rsh_ref': '$R_{sh}$',
         'logit_fc': '$f_c$', 'log_Rc': '$R_c$', 'T_cell': '$T$'}
j = {p: k for k, p in enumerate(PHI_NAMES)}
St, _ = build('front')
P = np.load('../results/v3_real_posteriors.npz'); single = P['single']

# ---------------- (1) irradiance consistency: deterministic fit vs NPE (v3, single-curve posteriors)
F = pd.read_csv('../results/fit_real_front.csv'); F['Rsh_ref'] = F.M1_Rsh * F.G / 1000.
rows = []
for i, r in St.reset_index(drop=True).iterrows():
    d = F[(F['sample'] == r['sample']) & (F.meas == r.meas)]
    if len(d) < 5: continue
    for p, col, jj in [('Rs', 'M1_Rs', 'log_Rs'), ('Rsh', 'Rsh_ref', 'log_Rsh_ref'), ('I01', 'M1_I01', 'log_I01'), ('I02', 'M1_I02', 'log_I02')]:
        rows.append(dict(state=i, param=p, deterministic=np.std(np.log10(d[col].values)),
                         npe=np.std(np.median(single[i, :, :, j[jj]], 1))))
D = pd.DataFrame(rows); D.to_csv('../results/consistency_det_vs_npe_v3.csv', index=False)
g = D.groupby('param')[['deterministic', 'npe']].median().loc[['Rs', 'Rsh', 'I01', 'I02']]
print('consistency (median SD of log10 estimate across 5 irradiances, n=%d states)\n' % D.state.nunique(), g.round(3))

# ---------------- (2) value of irradiance levels (v3)
I = pd.read_csv('../results/v3_real_info_subsets.csv')
ps = ['log_Rs', 'log_Rsh_ref', 'log_I01', 'logit_fc']
I['mean_nsd'] = I[['nsd_' + p for p in ps]].mean(axis=1)
best = I.sort_values('mean_nsd').groupby('k').head(1).sort_values('k')
print(best[['k', 'subset', 'mean_nsd']].round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.7))
x = np.arange(len(g))
ax[0].bar(x - .19, g.deterministic, .38, label='Deterministic least-squares fit', color='#B8B8B8', edgecolor='#555')
ax[0].bar(x + .19, g.npe, .38, label='NPE posterior median (this work)', color='#3E74B5', edgecolor='#1F3F66')
for k in range(len(g)):
    ax[0].text(x[k] - .19, g.deterministic.iloc[k] * 1.12, f'{g.deterministic.iloc[k]:.2f}', ha='center', fontsize=7)
    ax[0].text(x[k] + .19, g.npe.iloc[k] * 1.12, f'{g.npe.iloc[k]:.2f}', ha='center', fontsize=7, color='#1F3F66')
ax[0].set_xticks(x); ax[0].set_xticklabels(['$R_s$', '$R_{sh}$', '$I_{01}$', '$I_{02}$']); ax[0].set_yscale('log')
ax[0].set_ylim(0.01, 10)
ax[0].set_ylabel('Spread of single-curve estimates\nacross 5 irradiances (SD, decades)')
ax[0].set_title('(a) Irradiance consistency, %d real cell states' % D.state.nunique()); ax[0].legend(fontsize=7.5, loc='upper left')
ax[0].grid(alpha=.3, axis='y', which='both')
cols = ['#3E74B5', '#DD8452', '#55A868', '#C44E52']
full5 = I[I.k == 5].iloc[0]
for p, c in zip(ps, cols):
    b = I.sort_values('nsd_' + p).groupby('k').head(1).sort_values('k')
    ax[1].plot(b.k, b['nsd_' + p] / full5['nsd_' + p], 'o-', ms=4.5, color=c, label=short[p])
ax[1].axhline(1, color='k', lw=.8, ls='--'); ax[1].set_xticks(range(1, 6))
ax[1].set_xlabel('Number of irradiance levels measured (best subset)'); ax[1].set_ylabel('Posterior SD relative to all five levels')
ax[1].set_title('(b) Value of additional irradiance levels'); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.3)
ax[1].text(3.0, 1.55, 'Best single level: 925 W m$^{-2}$\nBest pair: 925 + 185 W m$^{-2}$', ha='center', fontsize=7.8,
           bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999'))
plt.tight_layout(); plt.savefig('../figs/fig_consistency_info_v3.png', dpi=300); plt.close()

import sys
if 'nobench' in sys.argv: sys.exit()
# ---------------- (3) well-specified benchmark vs exact MCMC
from npe import load
m = load(); libc = ctypes.CDLL('libc.so.6')
cases = sorted(glob.glob('../results/simbench_case*.npz'), key=lambda f: int(f.split('case')[1].split('.')[0]))
show = [0, 3]                     # one low-noise and one high-noise case for marginals
showp = ['log_Rs', 'log_Rsh_ref', 'log_I01', 'logit_fc']
Cmp = pd.read_csv('../results/simbench_compare.csv')
fig = plt.figure(figsize=(11.0, 5.6))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.5], hspace=.55, wspace=.38)
for r, c in enumerate(show):
    d = np.load(cases[c]); ch = d['chain'][:, :8]; tr = d['truth'][:8]
    with torch.no_grad():
        S = m.sample(torch.tensor(d['X'][None]), torch.ones(1, 5, dtype=torch.bool), torch.tensor(d['s'][None].astype(np.float32)),
                     torch.tensor(np.array([d['Tm']], np.float32)), 4000)[0]
    S = np.asarray(S); gc.collect(); libc.malloc_trim(0)
    for k, p in enumerate(showp):
        a = fig.add_subplot(gs[r, k]); q = j[p]
        lo, hi = np.quantile(np.r_[S[:, q], ch[:, q]], [.002, .998]); bins = np.linspace(lo, hi, 45)
        a.hist(S[:, q], bins, density=True, color='#3E74B5', alpha=.45, label='NPE (amortised)')
        a.hist(ch[:, q], bins, density=True, histtype='step', color='#C44E52', lw=1.4, label='Exact MCMC')
        a.axvline(tr[q], color='k', lw=1.1, ls='--', label='Truth')
        a.set_yticks([]); a.set_xlabel(lab[p], fontsize=8); a.tick_params(labelsize=7)
        if k == 0: a.set_ylabel(f'case {c + 1}\nnoise s = {float(d["s"][0]):g}', fontsize=8)
        if r == 0 and k == 0: a.legend(fontsize=6.5, loc='upper left', frameon=False)
a = fig.add_subplot(gs[:, 4])
order = PHI_NAMES
data = [Cmp[Cmp.param == p].width_ratio.values for p in order]
bp = a.boxplot(data, vert=False, patch_artist=True, widths=.55, medianprops=dict(color='k'))
for b in bp['boxes']: b.set(facecolor='#DCE7F4', edgecolor='#3E74B5')
a.set_yticks(range(1, 9)); a.set_yticklabels([short[p] for p in order]); a.set_xscale('log'); a.axvline(1, color='#C44E52', ls='--', lw=1)
a.set_xlabel('90% interval width ratio NPE / MCMC'); a.grid(alpha=.3, axis='x', which='both')
a.set_title('(b) Conservativeness, 6 cases × 8 params', fontsize=9)
fig.text(0.02, 0.97, '(a) Posterior marginals on simulated cells (23-D exact MCMC with all nuisances vs. NPE)', fontsize=9, fontweight='bold')
plt.savefig('../figs/fig_simbench_v3.png', dpi=300, bbox_inches='tight'); plt.close()
print('truth in 90%%: MCMC %.2f NPE %.2f | median width ratio %.2f' % (Cmp.truth_in_mcmc90.mean(), Cmp.truth_in_npe90.mean(), Cmp.width_ratio.median()))
