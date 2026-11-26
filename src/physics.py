"""Cell-level two-diode PV model with a crack sub-region, vectorised with numpy.

Terminal current I(V) = I_main(V) + I_crk(V)
  main region (area 1-fc): Iph(1-fc), diodes I01(1-fc), I02(1-fc), series Rs, shunt Rsh
  cracked region (area fc): same per-area diode/photocurrent, series Rs + Rc (Rc -> inf: electrically isolated)
Each branch solves  Iph_k - D_k(x) - (x - V)/Rs_k = 0  for the junction voltage x by bisection
(g is strictly decreasing in x, so bisection is unconditionally robust).

theta (natural units), per cell at STC-like reference, T = 25 C:
  Iph_ref [A at 1000 W/m2], I01 [A], I02 [A], Rs [ohm], Rsh [ohm], fc [-], Rc [ohm], gsens [-] (irradiance sensor gain)
"""
import numpy as np

KB, Q = 1.380649e-23, 1.602176634e-19
VT25 = KB * 298.15 / Q  # thermal voltage at 25 C

NAMES = ['Iph_ref', 'I01', 'I02', 'Rs', 'Rsh', 'fc', 'Rc', 'gsens']


def _branch(V, Iph, I01, I02, Rs, Rsh, vt=VT25, n_iter=64, Imin=-60.0):
    """Solve one branch. All params broadcastable to V's shape. Returns current."""
    Rs = np.maximum(Rs, 1e-6)
    lo = V + Imin * Rs
    hi = V + np.maximum(Iph, 0) * Rs + 1e-9
    def g(x):
        return Iph - I01 * np.expm1(x / vt) - I02 * np.expm1(x / (2 * vt)) - x / Rsh - (x - V) / Rs
    lo = np.broadcast_to(lo, np.broadcast(lo, hi, Iph).shape).copy()
    hi = np.broadcast_to(hi, lo.shape).copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        gm = g(np.clip(mid, -5, 1.5))
        pos = gm > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    x = 0.5 * (lo + hi)
    return (x - V) / Rs


def iv_curve(V, G, theta, vt=VT25, shunt_scaling=True):
    """V: (n_v,) grid; G: scalar or (batch,) true irradiance; theta: dict of arrays (batch,) or scalars.
    Returns I with shape (batch, n_v). gsens multiplies the *reported* irradiance, i.e. the true
    photocurrent uses G while the diagnostician believes G_meas = gsens * G."""
    t = {k: np.atleast_1d(np.asarray(theta[k], float))[:, None] for k in NAMES if k in theta}
    G = np.atleast_1d(np.asarray(G, float))[:, None]
    V = np.asarray(V, float)[None, :]
    fc = t.get('fc', np.zeros_like(t['Iph_ref']))
    Rc = t.get('Rc', np.full_like(t['Iph_ref'], 1e6))
    Iph = t['Iph_ref'] * G / 1000.0
    # De Soto-type shunt scaling, confirmed on the Garaj cells (fitted Rsh ~ G^-0.83): Rsh(G) = Rsh_ref * 1000/G
    Rsh = t['Rsh'] * 1000.0 / G if shunt_scaling else t['Rsh']
    I_main = _branch(V, Iph * (1 - fc), t['I01'] * (1 - fc), t['I02'] * (1 - fc), t['Rs'], Rsh, vt)
    I_crk = np.where(fc > 1e-6,
                     _branch(V, Iph * fc, t['I01'] * fc, t['I02'] * fc, t['Rs'] + Rc, 1e9, vt),
                     0.0)
    return I_main + I_crk


def add_noise(I, V, rng, sd_V=0.33e-3, sd_I=3.7e-3, scale=1.0):
    """Empirical noise model from replicate sweeps: per-curve voltage jitter + white current noise."""
    b, n = I.shape
    dV = rng.normal(0, sd_V * scale, size=(b, 1))
    dIdV = np.gradient(I, V, axis=1)
    return I + dIdV * dV + rng.normal(0, sd_I * scale, size=I.shape)
