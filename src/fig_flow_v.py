"""Fig. 1 (vertical): workflow flowchart, top-to-bottom, sized for ~5 in print width (white background, 300 dpi)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Rectangle, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})
fig = plt.figure(figsize=(6.5, 8.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 13.23); ax.axis('off')
BLUE, GREEN, ORANGE, PURPLE, GREY = ('#E8F1FA', '#2F6DA8'), ('#EAF5EA', '#3A8A3A'), ('#FFF4E0', '#C07A10'), ('#F4ECF7', '#7B4F9D'), ('#F5F5F5', '#555555')
FT, FN = 11.5, 9.4
MX, SX, W, H = 3.45, 7.85, 3.6, 0.95

def label(x, y, title, sub=None):
    if sub:
        ax.text(x, y + 0.13, title, ha='center', va='bottom', fontsize=FT, fontweight='bold', color='#1E1E1E', zorder=6)
        ax.text(x, y - 0.03, sub, ha='center', va='top', fontsize=FN, color='#333', zorder=6, linespacing=1.2)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=FT - 0.5, fontweight='bold', color='#1E1E1E', zorder=6, linespacing=1.15)
def proc(x, y, t, s, c, w=W, h=H, lw=1.7):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.15', fc=c[0], ec=c[1], lw=lw, zorder=3)); label(x, y, t, s)
def data(x, y, t, s, c, w=W, h=H, k=0.25):
    ax.add_patch(Polygon([[x - w / 2 + k, y + h / 2], [x + w / 2 + k, y + h / 2], [x + w / 2 - k, y - h / 2], [x - w / 2 - k, y - h / 2]], fc=c[0], ec=c[1], lw=1.7, zorder=3)); label(x, y, t, s)
def cyl(x, y, t, s, c, w=W, h=H):
    e = 0.28
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=c[0], ec='none', zorder=3))
    for xx in (x - w / 2, x + w / 2): ax.plot([xx, xx], [y - h / 2, y + h / 2], color=c[1], lw=1.7, zorder=4)
    ax.add_patch(Ellipse((x, y - h / 2), w, e, fc=c[0], ec=c[1], lw=1.7, zorder=3))
    ax.add_patch(Rectangle((x - w / 2 + 0.02, y - h / 2), w - 0.04, 0.02, fc=c[0], ec='none', zorder=3))
    ax.add_patch(Ellipse((x, y + h / 2), w, e, fc='white', ec=c[1], lw=1.7, zorder=4)); label(x, y - 0.06, t, s)
def diamond(x, y, t, c, w=2.9, h=1.45):
    ax.add_patch(Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2], [x - w / 2, y]], fc=c[0], ec=c[1], lw=1.7, zorder=3)); label(x, y, t)
def arr(p, q, color='#3A3A3A', ls='-', lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, linestyle=ls, connectionstyle=f'arc3,rad={rad}', zorder=2, shrinkA=0, shrinkB=0))
def tag(x, y, s, color='#3A3A3A', ha='center', fs=None):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs or FN, color=color, style='italic', zorder=7, linespacing=1.2, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

# lanes
for y0, h, t, ly in [(7.55, 5.55, 'OFFLINE  ·  done once', 12.8), (0.12, 7.25, 'ONLINE  ·  per measured cell', 0.45)]:
    ax.add_patch(FancyBboxPatch((0.12, y0), 9.76, h, boxstyle='round,pad=0,rounding_size=0.15', fc='#FBFBFB', ec='#C8C8C8', lw=1.0, ls='--', zorder=0))
    ax.text(9.65, ly, t, ha='right', va='center', fontsize=10, fontweight='bold', color='#6A6A6A')

# offline column
data(MX, 12.2, 'Prior p(θ)', '8 physical parameters', BLUE, h=0.9)
proc(MX, 10.95, 'Physics simulator', 'two-diode + crack + noise', BLUE, h=0.9)
cyl(MX, 9.6, 'Simulation bank', '3 × 10$^5$ simulated cells', BLUE, h=0.9)
proc(MX, 8.2, 'Train NPE', 'set encoder + spline flow', GREEN, h=0.8)
arr((MX, 11.75), (MX, 11.4)); arr((MX, 10.5), (MX, 10.19)); arr((MX, 9.01), (MX, 8.6))
proc(SX, 10.95, 'Identifiability', '& measurement design\n(Fisher / EIG)', PURPLE, w=3.3, h=1.25)
arr((MX + W / 2, 10.95), (SX - 1.65, 10.95)); tag((MX + W / 2 + SX - 1.65) / 2, 11.22, 'Jacobian', fs=8.6)

# online column
data(SX, 6.45, 'I–V curves', 'any 1–5 irradiance levels', ORANGE, w=3.3)
arr((SX, 10.95 - 0.63), (SX, 6.93)); tag(SX + 0.12, 8.7, 'measure at\n925 + 185 W m$^{-2}$', ha='left')
proc(MX, 6.45, 'Amortised NPE', '0.4–0.8 s per cell', GREEN)
arr((SX - 1.9, 6.45), (MX + W / 2, 6.45))
arr((MX, 7.8), (MX, 6.93), color='#3A8A3A', ls='--'); tag(MX + 0.12, 7.35, 'trained weights', color='#3A8A3A', ha='left')
data(MX, 4.95, 'Posterior p(θ | x)', 'samples of 8 parameters', GREEN)
arr((MX, 5.98), (MX, 5.43))
diamond(MX, 3.3, 'Misfit ≤\nthreshold?', GREY)
arr((MX, 4.47), (MX, 4.03))
proc(MX, 1.25, 'Diagnosis', '$R_s$, $R_{sh}$, $f_c$ ± 90% CI', ('#E6F4E6', '#2E7D32'), lw=2.2, h=1.0)
arr((MX, 2.57), (MX, 1.75), color='#2E7D32'); tag(MX + 0.15, 2.17, 'yes', color='#2E7D32', ha='left')
proc(SX, 3.3, 'Out-of-model flag', 'e.g. partial shading', ('#FCE8E6', '#B23A2E'), w=3.3, lw=2.2, h=1.0)
arr((MX + 1.45, 3.3), (SX - 1.65, 3.3), color='#B23A2E'); tag((MX + 1.45 + SX - 1.65) / 2, 3.58, 'no', color='#B23A2E')
plt.savefig('../figs/Fig1_framework.png', dpi=300); plt.close(); print('ok')
