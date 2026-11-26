"""Graphical abstract (Elsevier: >= 1328 x 531 px, aspect ~2.5:1), real data from cell F_1 (interconnect failure)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from real_obs import build
from simulate_bank import GRID, PHI_NAMES
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})
St, X = build('front'); P = np.load('../results/v3_real_posteriors.npz')['full']; J = PHI_NAMES.index('log_Rs')
ids = [36, 34, 32]; labs = ['intact', '20% ribbons cut', '40% ribbons cut']; cols = ['#2F6DA8', '#E08A2B', '#B23A2E']

fig = plt.figure(figsize=(8.86, 3.54))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 10); bg.set_ylim(0, 4); bg.axis('off')
def box(x, y, w, h, fc, ec):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12', fc=fc, ec=ec, lw=1.4, zorder=0))
def arr(x0, x1, y):
    bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=16, color='#3A3A3A', lw=2.0))
box(0.12, 0.55, 3.0, 3.3, '#FFF8EC', '#C07A10'); box(3.62, 0.55, 2.55, 3.3, '#EEF7EE', '#3A8A3A'); box(6.67, 0.55, 3.2, 3.3, '#F6F0F9', '#7B4F9D')
bg.text(1.62, 3.6, 'Measured I–V curves', ha='center', fontsize=10, fontweight='bold')
bg.text(4.9, 3.6, 'Amortised neural\nposterior estimator', ha='center', va='top', fontsize=10, fontweight='bold', linespacing=1.15)
bg.text(8.27, 3.6, 'Calibrated diagnosis', ha='center', fontsize=10, fontweight='bold')
arr(3.15, 3.58, 2.2); arr(6.2, 6.63, 2.2)
# left: real curves (925 W m-2)
a = fig.add_axes([0.05, 0.25, 0.235, 0.6])
for i, l, c in zip(ids, labs, cols): a.plot(GRID, X[i, 0] * 0.925, color=c, lw=1.6, label=l)
a.set_xlim(0, 0.68); a.set_ylim(0, 8.6); a.set_xlabel('Voltage (V)', fontsize=7.5); a.set_ylabel('Current (A)', fontsize=7.5)
a.tick_params(labelsize=6.5); a.legend(fontsize=6.3, frameon=False, loc='lower left'); a.grid(alpha=.25)
# middle: text
bg.text(4.9, 2.35, 'trained once on\n3 × 10$^5$ simulated cells\n\nany subset of irradiance levels\n\nno fault labels\n\n< 1 s per cell', ha='center', va='center', fontsize=8.2, color='#2C2C2C', linespacing=1.15)
# right: posterior of Rs
b = fig.add_axes([0.70, 0.27, 0.27, 0.58])
for i, l, c in zip(ids, labs, cols):
    r = 1e3 * 10 ** P[i, :, J]; b.hist(r, bins=np.geomspace(3, 80, 50), color=c, alpha=.45, density=False)
    lo, hi = np.quantile(r, [.05, .95]); b.plot([lo, hi], [-12, -12], color=c, lw=2.5, solid_capstyle='butt')
b.set_xscale('log'); b.set_xticks([5, 10, 20, 50]); b.set_xticklabels(['5', '10', '20', '50']); b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); b.set_yticks([]); b.set_xlabel('Series resistance $R_s$ (mΩ), posterior', fontsize=7.5); b.tick_params(labelsize=6.5)
b.set_ylim(-25, None); b.text(0.03, 0.93, 'bars: 90% credible intervals', transform=b.transAxes, fontsize=6.3, color='#444')
bg.text(5.0, 0.22, 'Physics-based simulation-based inference  ·  identifiability-guided measurement design (2 sweeps ≈ 5)  ·  built-in misfit check',
        ha='center', va='center', fontsize=7.8, color='#444', style='italic')
plt.savefig('../figs/Graphical_abstract.png', dpi=300); print('ok')
