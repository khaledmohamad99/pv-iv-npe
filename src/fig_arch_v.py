"""Fig. 3 (vertical): architecture of the set-conditioned NPE as a top-to-bottom flowchart (white background, 300 dpi).
Designed for ~5.2 in print width, so all text is >= 7 pt in print."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})
fig = plt.figure(figsize=(6.5, 8.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 13.23); ax.axis('off')
ORANGE, GREEN, BLUE, PURPLE, GREY = ('#FFF4E0', '#C07A10'), ('#EAF5EA', '#3A8A3A'), ('#E8F1FA', '#2F6DA8'), ('#F4ECF7', '#7B4F9D'), ('#F7F7F7', '#9A9A9A')
FT, FB, FN = 12, 10.5, 9.6   # title / body / note font sizes

def box(x, y, w, h, c, lw=1.6, ls='-', r=0.16):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=c[0], ec=c[1], lw=lw, ls=ls, zorder=3))
def txt(x, y, s, fs=FB, bold=False, color='#1E1E1E', ha='center', **k):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=color, zorder=6, linespacing=1.25, **k)
def arr(p, q, color='#3A3A3A', lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, connectionstyle=f'arc3,rad={rad}', zorder=2, shrinkA=0, shrinkB=0))
def num(y, n, x=0.42):
    ax.add_patch(Circle((x, y), 0.3, fc='#3A3A3A', ec='none', zorder=6)); txt(x, y, str(n), fs=11, bold=True, color='white')
def note(x, y, s): txt(x, y, s, fs=FN, color='#555', ha='left', style='italic')
def iv_icon(x, y, w, h, color):
    v = np.linspace(0, 1, 60); i = 1 - np.exp((v - 0.93) / 0.06); i = np.clip(i, 0, None) / i.max()
    ax.plot(x - w / 2 + v * w, y - h / 2 + i * h, color=color, lw=1.5, zorder=5)
    ax.plot([x - w / 2, x - w / 2, x + w / 2], [y + h / 2, y - h / 2, y - h / 2], color='#777', lw=0.8, zorder=5)

# 1 ---- input curves
num(12.7, 1); txt(0.95, 12.7, 'Input: I–V curves at any subset of the five irradiance levels', fs=FT - 0.5, bold=True, ha='left')
xs = [1.35, 3.17, 5.0, 6.83, 8.65]; G = [925, 740, 555, 370, 185]; meas = [True, False, False, False, True]; yc = 10.95
for x, g, m in zip(xs, G, meas):
    c = ORANGE if m else GREY
    box(x, yc, 1.62, 1.55, c, ls='-' if m else (0, (3, 2)), lw=1.7 if m else 1.1)
    iv_icon(x, yc + 0.3, 0.95, 0.5, c[1] if m else '#BBBBBB')
    txt(x, yc - 0.27, f'{g} W m$^{{-2}}$', fs=FN, bold=m, color='#1E1E1E' if m else '#9A9A9A')
    txt(x, yc - 0.58, 'measured' if m else 'masked', fs=FN - 0.6, color=c[1] if m else '#9A9A9A', style='italic')
txt(0.95, 12.15, 'each curve also carries its noise level $s$ and the measured temperature $T$', fs=FN, color='#555', style='italic', ha='left')

# 2 ---- shared encoder
ye = 8.75; num(ye, 2)
box(5.0, ye, 8.2, 1.15, GREEN)
txt(5.0, ye + 0.22, 'Shared encoder  $f_\\psi$', fs=FT, bold=True)
txt(5.0, ye - 0.25, 'the same MLP turns every measured curve into a 128-d embedding', fs=FN, color='#333')
for x, m in zip(xs, meas):
    if m: arr((x, yc - 0.78), (x if x < 5 else x, ye + 0.58))

# 3 ---- pooling
yp = 7.2; num(yp, 3)
box(5.0, yp, 4.6, 0.95, GREEN, r=0.45)
txt(5.0, yp, 'Average over curves', fs=FT - 0.5, bold=True)
arr((5.0, ye - 0.58), (5.0, yp + 0.48))
note(7.45, yp, 'any number,\nany order')

# 4 ---- context
yc4 = 5.75; num(yc4, 4)
box(5.0, yc4, 4.6, 0.95, GREEN)
txt(5.0, yc4, 'Context network  $g_\\psi$  →  $c$', fs=FT - 0.5, bold=True)
arr((5.0, yp - 0.48), (5.0, yc4 + 0.48))
note(7.45, yc4, '+ mask of the\nmeasured levels')

# 5 ---- spline flow
yf = 3.85; num(yf, 5)
box(5.0, yf, 6.6, 2.0, BLUE)
txt(5.0, yf + 0.68, 'Conditional neural spline flow', fs=FT, bold=True)
txt(2.55, yf - 0.2, '$z \\sim \\mathcal{N}(0, I)$', fs=FB)
for k in range(5):
    ax.add_patch(FancyBboxPatch((3.85 + k * 0.5, yf - 0.55), 0.34, 0.75, boxstyle='round,pad=0,rounding_size=0.05', fc='white', ec=BLUE[1], lw=1.0, zorder=4))
arr((3.45, yf - 0.18), (3.8, yf - 0.18), lw=1.2, color=BLUE[1]); arr((6.25, yf - 0.18), (6.6, yf - 0.18), lw=1.2, color=BLUE[1])
txt(7.4, yf - 0.2, '$\\theta$ samples', fs=FB)
txt(5.0, yf - 0.8, '5 invertible transforms, each conditioned on $c$', fs=FN, color='#333')
arr((5.0, yc4 - 0.48), (5.0, yf + 1.0))

# 6 ---- posterior
yo = 1.75; num(yo, 6)
box(5.0, yo, 6.6, 1.35, PURPLE)
gx = np.linspace(-2.2, 2.2, 40); gy = np.exp(-gx ** 2 / 1.1); cx = 2.55
ax.fill_between(cx + gx * 0.33, yo - 0.3, yo - 0.3 + gy * 0.8, color=PURPLE[1], alpha=0.35, zorder=5, lw=0)
for sx in (-0.55, 0.55): ax.plot([cx + sx, cx + sx], [yo - 0.3, yo + 0.18], color=PURPLE[1], lw=1.0, ls='--', zorder=6)
txt(cx, yo - 0.47, '90% CI', fs=FN - 1, color=PURPLE[1])
txt(5.9, yo + 0.28, 'Posterior  $p(\\theta \\mid x)$', fs=FT, bold=True)
txt(5.9, yo - 0.25, '8 parameters ($R_s$, $R_{sh}$, $f_c$, …)\nwith credible intervals', fs=FN, color='#333')
arr((5.0, yf - 1.0), (5.0, yo + 0.68))

# ---- training note
ax.add_patch(FancyBboxPatch((0.3, 0.12), 9.4, 0.62, boxstyle='round,pad=0,rounding_size=0.12', fc='#FBFBFB', ec='#C8C8C8', lw=0.9, ls='--', zorder=1))
txt(5.0, 0.43, 'Trained once on 3 × 10$^5$ simulated cells with random subsets of levels', fs=FN, color='#444')
plt.savefig('../figs/Fig3_architecture.png', dpi=300); plt.close(); print('ok')
