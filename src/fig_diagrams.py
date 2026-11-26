"""Workflow and architecture diagrams (white background, 300 dpi)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
C = dict(phys='#E8F1FA', phys_e='#2F6DA8', ml='#EAF5EA', ml_e='#3A8A3A', data='#FFF4E0', data_e='#C07A10',
         out='#F4ECF7', out_e='#7B4F9D', grey='#F2F2F2', grey_e='#666666')


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, bold_first=True, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.012', fc=fc, ec=ec, lw=lw))
    lines = text.split('\n')
    if bold_first and len(lines) > 1:
        cy = y + h / 2
        ax.text(x + w / 2, cy + 0.006, lines[0], ha='center', va='bottom', fontsize=fs + 0.3, fontweight='bold', color='#222')
        ax.text(x + w / 2, cy - 0.004, '\n'.join(lines[1:]), ha='center', va='top', fontsize=fs - 0.7, color='#333', linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color='#222', linespacing=1.3,
                fontweight='bold' if (bold_first and len(lines) == 1) else 'normal')


def arrow(ax, p, q, color='#444', style='-|>', lw=1.3, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}'))


def panel(ax, x, y, w, h, title, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.015', fc='none', ec=ec, lw=1.0, ls='--'))
    ax.text(x + 0.012, y + h - 0.012, title, ha='left', va='top', fontsize=8.5, color=ec, fontweight='bold')


# ---------------- Fig. 1 workflow
fig = plt.figure(figsize=(12.5, 7.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1.25); ax.set_ylim(0, 0.70); ax.axis('off')
panel(ax, 0.015, 0.375, 0.40, 0.31, '(A) Physics-based generative model (offline)', C['phys_e'])
box(ax, 0.03, 0.525, 0.175, 0.12, 'Prior p(θ)\n8 physical parameters\nIph, I01, I02, Rs, Rsh, fc, Rc, T', C['phys'], C['phys_e'])
box(ax, 0.225, 0.525, 0.175, 0.12, 'Cell simulator\ntwo-diode + crack branch\nshunt scaling Rsh ∝ 1/G', C['phys'], C['phys_e'])
box(ax, 0.03, 0.39, 0.175, 0.115, 'Nuisances\nper-curve irradiance gain\nper-curve temperature offset', C['phys'], C['phys_e'])
box(ax, 0.225, 0.39, 0.175, 0.115, 'Empirical noise model\nvoltage jitter 0.33 mV\n+ white current 3.7 mA (×s)', C['phys'], C['phys_e'])
arrow(ax, (0.205, 0.585), (0.225, 0.585)); arrow(ax, (0.205, 0.45), (0.27, 0.525), rad=-0.2); arrow(ax, (0.3125, 0.505), (0.3125, 0.525))

panel(ax, 0.435, 0.375, 0.40, 0.31, '(B) Amortised neural posterior estimation (offline training)', C['ml_e'])
box(ax, 0.45, 0.525, 0.175, 0.12, 'Simulation bank\n3×10⁵ cells × 5 irradiances\n925–185 W m⁻², s ∈ [1, 20]', C['ml'], C['ml_e'])
box(ax, 0.645, 0.525, 0.175, 0.12, 'Random subsets\nany 1–5 curves per cell\n(protocol-agnostic training)', C['ml'], C['ml_e'])
box(ax, 0.45, 0.39, 0.37, 0.115, 'Set encoder + conditional neural spline flow\nmaximise  E[ log q_φ(θ | x_S, s, T_meas) ]  →  q_φ ≈ p(θ | x_S)', C['ml'], C['ml_e'])
arrow(ax, (0.40, 0.585), (0.45, 0.585)); arrow(ax, (0.625, 0.585), (0.645, 0.585)); arrow(ax, (0.7325, 0.525), (0.7325, 0.505))

panel(ax, 0.855, 0.375, 0.38, 0.31, '(C) Real measurements (public data)', C['data_e'])
box(ax, 0.87, 0.525, 0.35, 0.12, 'Garaj et al. single-cell laminates\n17 cells · PID / interconnect / cracks · staged severity\n5 irradiances × 3 replicate sweeps (69 unique states)', C['data'], C['data_e'])
box(ax, 0.87, 0.39, 0.35, 0.115, 'ITM modules (36 cells, outdoor)\nclean / soiled / partially shaded\nmeasured G and module temperature', C['data'], C['data_e'])

# inference arrow
ax.add_patch(FancyBboxPatch((0.30, 0.265), 0.67, 0.07, boxstyle='round,pad=0.0,rounding_size=0.012', fc='#FFFFFF', ec='#222', lw=1.6))
ax.text(0.635, 0.30, 'Amortised inference  q_φ(θ | x_obs)  ·  0.4–0.8 s per cell  ·  no labels, no retraining',
        ha='center', va='center', fontsize=8.6, fontweight='bold')
arrow(ax, (0.635, 0.39), (0.635, 0.335), lw=1.6); arrow(ax, (1.045, 0.39), (0.97, 0.30), lw=1.6, rad=0.15)

panel(ax, 0.015, 0.015, 1.22, 0.2, '(D) Engineering outputs and validation', C['out_e'])
outs = [('Identifiability map\nwhat an I–V curve can and\ncannot reveal (Fisher + NPE)'),
        ('Measurement design\nexpected information gain\nper irradiance subset'),
        ('Degradation diagnosis\nposteriors of Rs, Rsh, fc\nseverity tracking + calibration'),
        ('Trust checks\ncross-irradiance z-test,\nposterior predictive misfit'),
        ('Benchmarks\nexact MCMC, supervised RF/CNN,\nfield-grade noise')]
for k, t in enumerate(outs):
    box(ax, 0.03 + k * 0.2405, 0.03, 0.225, 0.14, t, C['out'], C['out_e'], fs=8.3)
arrow(ax, (0.635, 0.265), (0.635, 0.218), lw=1.6)
plt.savefig('../figs/Fig1_framework.png', dpi=300); plt.close()

# ---------------- Fig. 3 architecture
fig = plt.figure(figsize=(12.5, 4.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1.25); ax.set_ylim(0, 0.46); ax.axis('off')
ys = [0.36, 0.285, 0.21, 0.135, 0.06]; Gs = [925, 740, 555, 370, 185]
for k, (y, g) in enumerate(zip(ys, Gs)):
    m = k in (0, 4)
    box(ax, 0.01, y, 0.17, 0.062, f'I–V curve @ {g} W m⁻²\n96 pts, I·1000/G' + ('' if m else '  (masked)'),
        C['data'] if m else C['grey'], C['data_e'] if m else C['grey_e'], fs=7.6, bold_first=False)
    box(ax, 0.215, y, 0.15, 0.062, 'shared MLP  f_ψ\n[x, G, log s, T_meas] → 128', C['ml'] if m else C['grey'], C['ml_e'] if m else C['grey_e'], fs=7.6, bold_first=False)
    arrow(ax, (0.18, y + 0.031), (0.215, y + 0.031), color='#444' if m else '#AAA')
    arrow(ax, (0.365, y + 0.031), (0.425, 0.21 + 0.031), color='#444' if m else '#BBB', ls='-' if m else '--')
box(ax, 0.425, 0.175, 0.13, 0.13, 'Masked mean\npooling\n(permutation-\ninvariant)', C['ml'], C['ml_e'], fs=8.0, bold_first=False)
box(ax, 0.59, 0.175, 0.13, 0.13, 'Context MLP g_ψ\n[pooled, mask m]\n→ c ∈ ℝ¹²⁸', C['ml'], C['ml_e'], fs=8.0, bold_first=False)
box(ax, 0.755, 0.12, 0.2, 0.24, 'Conditional neural\nspline flow q_φ(θ | c)\n5 autoregressive RQ-spline\ntransforms, MADE 256-256\nbase N(0, I₈)', C['phys'], C['phys_e'], fs=8.2, bold_first=False)
box(ax, 0.99, 0.12, 0.25, 0.24, 'Posterior samples θ⁽ⁱ⁾\nIph, log I01, log I02, log Rs,\nlog Rsh, logit fc, log Rc, T\n→ credible intervals,\nseverity, misfit score', C['out'], C['out_e'], fs=8.2, bold_first=False)
arrow(ax, (0.555, 0.24), (0.59, 0.24)); arrow(ax, (0.72, 0.24), (0.755, 0.24)); arrow(ax, (0.955, 0.24), (0.99, 0.24))
ax.text(0.10, 0.44, 'Input: any subset S of 5 levels', ha='center', fontsize=8.5, fontweight='bold')
ax.text(0.29, 0.44, 'Per-curve embedding', ha='center', fontsize=8.5, fontweight='bold')
ax.text(0.655, 0.44, 'Set summary', ha='center', fontsize=8.5, fontweight='bold')
ax.text(0.855, 0.44, 'Posterior density', ha='center', fontsize=8.5, fontweight='bold')
ax.text(1.115, 0.44, 'Output', ha='center', fontsize=8.5, fontweight='bold')
ax.text(0.655, 0.075, 'Trained once by maximising log q_φ(θ | x_S) over 3×10⁵ simulated cells;\nmask m and noise level s make one network serve every measurement protocol.',
        ha='center', fontsize=7.8, color='#333')
plt.savefig('../figs/Fig3_architecture.png', dpi=300, bbox_inches='tight', pad_inches=0.08); plt.close()

# ---------------- Fig. 2 equivalent circuit (physics model)
fig = plt.figure(figsize=(9.4, 3.7)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 9.4); ax.set_ylim(0, 3.7); ax.axis('off')
def wire(xs, ys, c='#222', lw=1.4): ax.plot(xs, ys, color=c, lw=lw, solid_capstyle='round')
def src(x, y):
    ax.add_patch(plt.Circle((x, y), 0.28, fc='white', ec='#222', lw=1.4, zorder=3))
    ax.annotate('', (x, y + 0.18), (x, y - 0.18), arrowprops=dict(arrowstyle='-|>', lw=1.2), zorder=4)
def diode(x, y):
    ax.add_patch(plt.Polygon([[x - 0.18, y + 0.16], [x + 0.18, y + 0.16], [x, y - 0.14]], fc='white', ec='#222', lw=1.3, zorder=3))
    wire([x - 0.18, x + 0.18], [y - 0.16, y - 0.16])
def vres(x, y0, y1): ax.add_patch(plt.Rectangle((x - 0.1, y0), 0.2, y1 - y0, fc='white', ec='#222', lw=1.3, zorder=3))
def lab(x, y, t, **k): ax.text(x, y, t, fontsize=8.3, va='center', linespacing=1.25, **k)
top, bot, mid = 2.75, 0.85, 1.8
# intact region
xs, xd1, xd2, xr = 0.7, 1.85, 3.15, 4.4
wire([xs, xs], [bot, top]); wire([xs, 4.9], [top, top]); wire([xs, 8.6], [bot, bot])
src(xs, mid); lab(xs + 0.36, mid, '(1−f_c)·\nI_ph(G,T)')
for x, t in [(xd1, 'D1 (n = 1)\n(1−f_c)·I01'), (xd2, 'D2 (n = 2)\n(1−f_c)·I02')]:
    wire([x, x], [top, mid + 0.16]); diode(x, mid); wire([x, x], [mid - 0.16, bot]); lab(x + 0.26, mid, t)
wire([xr, xr], [top, bot]); vres(xr, 1.4, 2.2); lab(xr + 0.18, mid, 'R_sh·1000/G')
ax.add_patch(plt.Rectangle((4.9, top - 0.1), 0.8, 0.2, fc='white', ec='#222', lw=1.3, zorder=3)); ax.text(5.3, top + 0.2, 'R_s', ha='center', fontsize=8.5)
wire([5.7, 8.6], [top, top])
ax.text(2.55, 3.25, 'intact region (area 1−f_c)', ha='center', fontsize=9, fontweight='bold', color=C['phys_e'])
# cracked region: own junction, connected to the terminal through R_s + R_c
cx = 6.95
wire([cx, cx], [top, 2.4]); vres(cx, 2.0, 2.4); lab(cx + 0.18, 2.2, 'R_s + R_c')
wire([cx, cx], [2.0, 1.9]); wire([cx - 0.5, cx + 0.5], [1.9, 1.9])
wire([cx - 0.5, cx - 0.5], [1.9, bot]); src(cx - 0.5, 1.45); lab(cx - 0.86, 1.45, 'f_c·I_ph', ha='right')
wire([cx + 0.5, cx + 0.5], [1.9, 1.61]); diode(cx + 0.5, 1.45); wire([cx + 0.5, cx + 0.5], [1.29, bot]); lab(cx + 0.76, 1.45, 'f_c·I01,\nf_c·I02')
ax.text(cx, 3.25, 'cracked region (area f_c)', ha='center', fontsize=9, fontweight='bold', color='#A0392F')
ax.add_patch(plt.Circle((8.6, top), 0.06, color='#222')); ax.add_patch(plt.Circle((8.6, bot), 0.06, color='#222'))
ax.text(8.72, top, '+', fontsize=12, va='center'); ax.text(8.72, bot, '−', fontsize=12, va='center')
ax.annotate('', (9.05, bot + 0.1), (9.05, top - 0.1), arrowprops=dict(arrowstyle='<->', lw=1.0)); ax.text(9.15, mid, 'V', fontsize=10, va='center')
ax.text(4.7, 0.35, 'Per-curve nuisances j: G_true = g_j·G_nominal,  T_j = T + δT_j;   observed I = I(V) + (dI/dV)·δV_j + ε,   ε ~ N(0, (s·3.7 mA)²)',
        ha='center', fontsize=8.0, color='#333')
plt.savefig('../figs/Fig2_circuit.png', dpi=300); plt.close()
print('ok')
