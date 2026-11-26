import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'font.size': 9})
B = json.load(open('../results/baselines_lofo.json')); B2 = json.load(open('../results/baselines2.json'))
keys = ['RF-raw', 'RF-feat', 'CNN', 'SimReg+LR', 'NPE+LR', 'LSQ+LR']
lab = ['RF\n(raw curves)', 'RF\n(I–V features)', '1D-CNN', 'Sim-trained point\nestimator + LR', 'NPE posterior\n+ LR (ours)', 'Least-squares\nparameters + LR']
get = lambda k: (B if k in B else B2)[k]
acc = [get(k)['acc'] for k in keys]; f1 = [get(k)['macroF1'] for k in keys]
sa = [B2['NPE+LR_v3s%d' % i]['acc'] for i in range(3)] + [B['NPE+LR']['acc']]
I = pd.read_csv('../results/itm_case.csv'); thr = json.load(open('../results/itm_flag.json'))['threshold']
P = json.load(open('../results/itm_processed_check.json')); print(P)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
x = np.arange(len(keys))
ax[0].bar(x - .18, acc, .36, color='#B8B8B8', edgecolor='#555', label='Accuracy'); ax[0].bar(x + .18, f1, .36, color='#3E74B5', edgecolor='#1F3F66', label='Macro-F1')
for k in range(len(keys)):
    ax[0].text(x[k] - .18, acc[k] + .015, f'{acc[k]:.2f}', ha='center', fontsize=7.5); ax[0].text(x[k] + .18, f1[k] + .015, f'{f1[k]:.2f}', ha='center', fontsize=7.5, color='#1F3F66')
ax[0].axhline(0.25, color='#C44E52', ls=':', lw=1.2, label='Chance level (4 classes)')
k = keys.index('NPE+LR'); ax[0].errorbar(x[k] - .18, acc[k], yerr=[[acc[k] - min(sa)], [max(sa) - acc[k]]], fmt='none', ecolor='k', capsize=3, lw=1, label='Range over 3 networks')
for j in range(3): ax[0].axvspan(-.5, 2.5, color='#F4F4F4', zorder=0) if j == 0 else None
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=7.3); ax[0].text(1, .93, 'appearance-based', ha='center', fontsize=7.5, color='#555'); ax[0].text(4, .93, 'physics-based parameters', ha='center', fontsize=7.5, color='#555'); ax[0].set_ylim(0, 1); ax[0].set_ylabel('Leave-one-family-out score')
ax[0].set_title('(a) Fault-mode identification on unseen cell families'); ax[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.9)); ax[0].grid(alpha=.3, axis='y')
col = {'Clean': '#55A868', 'Dirt': '#DD8452', 'Shadow': '#C44E52'}; off = {'Clean': -.22, 'Dirt': 0, 'Shadow': .22}
panels = list(dict.fromkeys(zip(I.module, I.panel)))
for i, (m, p) in enumerate(panels):
    for _, r in I[(I.module == m) & (I.panel == p)].iterrows():
        ax[1].scatter(i + off[r.cond], r.ppc_rmse_pct, s=34, color=col[r.cond], edgecolor='k', lw=.4, zorder=3, label=r.cond if i == 0 else None)
v = [val for k, val in P.items() if isinstance(val, (int, float)) and val > 10] if isinstance(P, dict) else []
L = pd.read_csv('../results/itm_threshold_lomo.csv')
for i, (m, p) in enumerate(panels): t = L[L.pid == f'{m}_{p}'].thr.iloc[0]; ax[1].hlines(t, i - .4, i + .4, color='k', ls='--', lw=1)
ax[1].text(-.4, thr * 1.12, 'leave-one-module-out threshold (1.63–1.69 % $I_{sc}$)', fontsize=7.5)
ax[1].set_yscale('log'); ax[1].set_xticks(range(len(panels))); ax[1].set_xticklabels([f'{"M5" if m.startswith("M5") else "SP090P"}-{p[-1]}' for m, p in panels], fontsize=8)
ax[1].set_ylabel('Posterior-predictive RMSE (% $I_{sc}$)'); ax[1].set_title('(b) Real modules (ITM): out-of-model detection')
ax[1].legend(fontsize=7.5, loc='upper left'); ax[1].grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig('../figs/fig_baselines_itm.png', dpi=300); print('ok')
