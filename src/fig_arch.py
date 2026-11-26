"""Fig. 3: architecture of the set-conditioned NPE as a clean left-to-right flowchart (white background, 300 dpi)."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})
fig = plt.figure(figsize=(12.6, 4.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 5.45); ax.axis('off')
ORANGE, GREEN, BLUE, PURPLE, GREY = ('#FFF4E0', '#C07A10'), ('#EAF5EA', '#3A8A3A'), ('#E8F1FA', '#2F6DA8'), ('#F4ECF7', '#7B4F9D'), ('#F7F7F7', '#9A9A9A')

def box(x, y, w, h, c, lw=1.6, ls='-'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.13', fc=c[0], ec=c[1], lw=lw, ls=ls, zorder=3))
def txt(x, y, s, fs=10, bold=False, color='#1E1E1E', **k):
    ax.text(x, y, s, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=color, zorder=6, linespacing=1.25, **k)
def arr(p, q, color='#3A3A3A', lw=1.6, rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, color=color, lw=lw, ls=ls, connectionstyle=f'arc3,rad={rad}', zorder=2, shrinkA=0, shrinkB=0))
def step(x, n, title):
    w = 0.118 * len(title) + 0.45; x0 = x - w / 2
    ax.add_patch(Circle((x0 + 0.17, 5.08), 0.17, fc='#3A3A3A', ec='none', zorder=6))
    ax.text(x0 + 0.17, 5.08, str(n), ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=7)
    ax.text(x0 + 0.43, 5.08, title, ha='left', va='center', fontsize=10, fontweight='bold', color='#333')
def iv_icon(x, y, w, h, color):
    v = np.linspace(0, 1, 60); i = 1 - np.exp((v - 0.93) / 0.06); i = np.clip(i, 0, None) / i.max()
    ax.plot(x - w / 2 + v * w, y - h / 2 + i * h, color=color, lw=1.4, zorder=5)
    ax.plot([x - w / 2, x - w / 2, x + w / 2], [y + h / 2, y - h / 2, y - h / 2], color='#777', lw=0.7, zorder=5)

# ---- 1 input: five irradiance slots, two measured
step(1.5, 1, 'Input curves')
ys = [4.3, 3.55, 2.8, 2.05, 1.3]; G = [925, 740, 555, 370, 185]; meas = [True, False, False, False, True]
for y, g, m in zip(ys, G, meas):
    c = ORANGE if m else GREY
    box(1.5, y, 2.3, 0.6, c, ls='-' if m else (0, (3, 2)), lw=1.6 if m else 1.1)
    iv_icon(0.82, y, 0.62, 0.36, c[1] if m else '#BBBBBB')
    txt(1.85, y + 0.08, f'{g} W m$^{{-2}}$', fs=8.8, bold=m, color='#1E1E1E' if m else '#9A9A9A')
    txt(1.85, y - 0.15, 'measured' if m else 'not measured', fs=7.6, color=c[1] if m else '#9A9A9A', style='italic')
txt(1.5, 0.55, 'side inputs per curve:\nnoise level s, measured $T$', fs=8.2, color='#555')

# ---- 2 shared encoder
EX, PX, CX, FX, OX = 4.4, 6.75, 8.75, 10.85, 13.0
step(EX, 2, 'Shared encoder')
box(EX, 2.8, 1.9, 1.7, GREEN)
txt(EX, 3.28, '$f_\\psi$', fs=15, bold=True)
txt(EX, 2.72, 'MLP applied to\neach curve', fs=8.6, color='#333')
txt(EX, 2.2, 'same weights', fs=8.0, color=GREEN[1], style='italic')
for y, m in zip(ys, meas):
    if m: arr((2.66, y), (EX - 0.96, 2.8 + (0.45 if y > 3 else -0.45)), color='#3A3A3A', rad=-0.12 if y > 3 else 0.12)

# ---- 3 pooling
step(PX, 3, 'Pool')
ax.add_patch(Circle((PX, 2.8), 0.62, fc=GREEN[0], ec=GREEN[1], lw=1.6, zorder=3))
txt(PX, 2.86, 'mean', fs=10, bold=True); txt(PX, 2.55, 'over curves', fs=7.6, color='#333')
txt(PX, 1.75, 'invariant to number\nand order of curves', fs=7.8, color='#555', style='italic')
arr((EX + 0.96, 2.8), (PX - 0.63, 2.8))

# ---- 4 context
step(CX, 4, 'Context')
box(CX, 2.8, 1.5, 1.2, GREEN)
txt(CX, 3.05, '$g_\\psi$', fs=14, bold=True); txt(CX, 2.58, 'vector $c$', fs=8.6, color='#333')
arr((PX + 0.63, 2.8), (CX - 0.76, 2.8))
txt(CX, 1.75, '+ mask of which\nlevels were used', fs=7.8, color='#555', style='italic')

# ---- 5 flow
step(FX, 5, 'Spline flow')
box(FX, 2.8, 1.9, 2.4, BLUE)
txt(FX, 3.72, '$z \\sim \\mathcal{N}(0, I)$', fs=9)
for k in range(5):
    ax.add_patch(FancyBboxPatch((FX - 0.45, 3.32 - k * 0.2), 0.9, 0.14, boxstyle='round,pad=0,rounding_size=0.03', fc='white', ec=BLUE[1], lw=0.8, zorder=4))
arr((FX, 3.58), (FX, 3.47), lw=1.0, color=BLUE[1])
txt(FX, 2.12, '5 invertible transforms\nconditioned on $c$', fs=7.8, color='#333')
arr((CX + 0.76, 2.8), (FX - 0.96, 2.8))

# ---- 6 output
step(OX, 6, 'Posterior')
box(OX, 2.8, 1.75, 2.4, PURPLE)
txt(OX, 3.7, '$p(\\theta\\mid x)$', fs=10)
xs = np.linspace(-2.2, 2.2, 40); yy = np.exp(-xs ** 2 / 1.1)
ax.fill_between(OX + xs * 0.26, 2.75, 2.75 + yy * 0.65, color=PURPLE[1], alpha=0.35, zorder=5, lw=0)
for sx in (-0.43, 0.43): ax.plot([OX + sx, OX + sx], [2.75, 3.15], color=PURPLE[1], lw=0.9, ls='--', zorder=6)
txt(OX, 2.55, '90% credible interval', fs=7.4, color=PURPLE[1])
txt(OX, 2.1, '8 parameters\n$R_s$, $R_{sh}$, $f_c$, …', fs=7.8, color='#333')
arr((FX + 0.96, 2.8), (OX - 0.88, 2.8))

# ---- training note
ax.add_patch(FancyBboxPatch((3.45, 0.22), 10.4, 0.55, boxstyle='round,pad=0,rounding_size=0.1', fc='#FBFBFB', ec='#C8C8C8', lw=0.9, ls='--', zorder=1))
txt(8.65, 0.495, 'Trained once on 3 × 10$^5$ simulated cells, each shown with a random subset of levels  →  one network serves every measurement protocol',
    fs=8.4, color='#444')
plt.savefig('../figs/Fig3_architecture.png', dpi=300); plt.close(); print('ok')
