"""Fig. 1 as a compact workflow flowchart (white background, 300 dpi)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Rectangle, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
fig = plt.figure(figsize=(12.6, 5.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 6.0); ax.axis('off')
BLUE, GREEN, ORANGE, PURPLE, GREY = ('#E8F1FA', '#2F6DA8'), ('#EAF5EA', '#3A8A3A'), ('#FFF4E0', '#C07A10'), ('#F4ECF7', '#7B4F9D'), ('#F5F5F5', '#555555')
W, H = 2.2, 1.0

def label(x, y, title, sub=None, fs=10.5):
    if sub:
        ax.text(x, y + 0.12, title, ha='center', va='bottom', fontsize=fs, fontweight='bold', color='#1E1E1E', zorder=5)
        ax.text(x, y - 0.06, sub, ha='center', va='top', fontsize=fs - 2, color='#333', zorder=5)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=fs, fontweight='bold', color='#1E1E1E', zorder=5, linespacing=1.15)

def proc(x, y, title, sub, c, w=W, h=H, lw=1.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.14', fc=c[0], ec=c[1], lw=lw, zorder=3))
    label(x, y, title, sub)

def data(x, y, title, sub, c, w=W, h=H, s=0.22):
    ax.add_patch(Polygon([[x - w / 2 + s, y + h / 2], [x + w / 2 + s, y + h / 2], [x + w / 2 - s, y - h / 2], [x - w / 2 - s, y - h / 2]],
                         closed=True, fc=c[0], ec=c[1], lw=1.6, zorder=3))
    label(x, y, title, sub)

def cyl(x, y, title, sub, c, w=W, h=H):
    e = 0.24
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=c[0], ec='none', zorder=3))
    ax.plot([x - w / 2, x - w / 2], [y - h / 2, y + h / 2], color=c[1], lw=1.6, zorder=4)
    ax.plot([x + w / 2, x + w / 2], [y - h / 2, y + h / 2], color=c[1], lw=1.6, zorder=4)
    ax.add_patch(Ellipse((x, y - h / 2), w, e, fc=c[0], ec=c[1], lw=1.6, zorder=3))
    ax.add_patch(Rectangle((x - w / 2 + 0.02, y - h / 2), w - 0.04, 0.02, fc=c[0], ec='none', zorder=3))
    ax.add_patch(Ellipse((x, y + h / 2), w, e, fc='white', ec=c[1], lw=1.6, zorder=4))
    label(x, y - 0.05, title, sub)

def diamond(x, y, text, c, w=2.0, h=1.45):
    ax.add_patch(Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2], [x - w / 2, y]], closed=True, fc=c[0], ec=c[1], lw=1.6, zorder=3))
    label(x, y, text, fs=9.5)

def arr(p, q, ls='-', color='#3A3A3A', lw=1.6, rad=0.0, text=None, tpos=None, tfs=8.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}', zorder=2, shrinkA=0, shrinkB=0))
    if text:
        ax.text(*tpos, text, ha='center', va='center', fontsize=tfs, color=color, style='italic', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

# lanes
for y0, h, t in [(3.35, 2.55, 'OFFLINE  ·  done once'), (0.1, 3.05, 'ONLINE  ·  per measured cell')]:
    ax.add_patch(FancyBboxPatch((0.12, y0), 13.76, h, boxstyle='round,pad=0,rounding_size=0.12', fc='#FBFBFB', ec='#C8C8C8', lw=1.0, ls='--', zorder=0))
    ax.text(13.7, y0 + h - 0.22, t, ha='right', va='center', fontsize=9.5, fontweight='bold', color='#6A6A6A')

yt, yb = 4.45, 1.55
# offline row
proc(1.75, yt, 'Identifiability', '& measurement design\n(Fisher / EIG)', PURPLE)
data(4.55, yt, 'Prior p(θ)', '8 physical parameters', BLUE)
proc(7.25, yt, 'Physics simulator', 'two-diode + crack + noise', BLUE)
cyl(9.95, yt, 'Simulation bank', '3 × 10⁵ cells', BLUE)
proc(12.6, yt, 'Train NPE', 'set encoder + spline flow', GREEN, w=2.15)
arr((5.73, yt), (6.15, yt)); arr((8.35, yt), (8.85, yt)); arr((11.05, yt), (11.52, yt))
arr((7.25, yt + 0.5), (1.75, yt + 0.5), rad=0.32, text='sensitivities', tpos=(4.5, 5.48))

# online row
data(1.75, yb, 'I–V curves', 'any 1–5 irradiance levels', ORANGE)
proc(4.6, yb, 'Amortised NPE', '0.4–0.8 s per cell', GREEN)
data(7.35, yb, 'Posterior p(θ | x)', 'samples of 8 parameters', GREEN)
diamond(10.05, yb, 'Misfit ≤\nthreshold?', GREY)
proc(12.75, yb + 0.72, 'Diagnosis', '$R_s$, $R_{sh}$, $f_c$ ± 90% CI', ('#E6F4E6', '#2E7D32'), w=2.05, h=0.95, lw=2.0)
proc(12.75, yb - 0.72, 'Out-of-model flag', 'e.g. partial shading', ('#FCE8E6', '#B23A2E'), w=2.05, h=0.95, lw=2.0)
arr((2.85, yb), (3.5, yb)); arr((5.7, yb), (6.28, yb)); arr((8.45, yb), (9.05, yb))
arr((10.6, yb + 0.4), (11.72, yb + 0.72), color='#2E7D32', text='yes', tpos=(11.05, yb + 0.78))
arr((10.6, yb - 0.4), (11.72, yb - 0.72), color='#B23A2E', text='no', tpos=(11.05, yb - 0.78))
# cross-lane links
arr((1.75, yt - 0.5), (1.75, yb + 0.5))
ax.text(1.9, 3.0, 'measure at\n925 + 185 W m⁻²', ha='left', va='center', fontsize=8.4, color='#3A3A3A', style='italic', zorder=6, bbox=dict(boxstyle='round,pad=0.15', fc='#FBFBFB', ec='none'))
ax.plot([12.6, 12.6, 4.6], [yt - 0.5, 3.05, 3.05], color='#3A8A3A', lw=1.6, ls='--', zorder=2)
arr((4.6, 3.05), (4.6, yb + 0.5), ls='--', color='#3A8A3A')
ax.text(8.6, 3.05, 'trained weights (no retraining)', ha='center', va='center', fontsize=8.6, color='#3A8A3A', style='italic',
        bbox=dict(boxstyle='round,pad=0.15', fc='#FBFBFB', ec='none'), zorder=6)
plt.savefig('../figs/Fig1_framework.png', dpi=300); plt.close(); print('ok')
