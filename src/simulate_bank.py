"""Simulation bank for amortised posterior estimation.
Each draw theta ~ prior is simulated at all five irradiance levels (one curve each) with the empirical noise model
(voltage jitter + white current noise) scaled by a per-curve factor s ~ LogUniform(1, 20) (lab -> field tracers).
Stored observation: current normalised by G/1000 on a fixed voltage grid V in [0, 0.70] V (96 points).
"""
import numpy as np, sys, time
from joblib import Parallel, delayed
from physics import iv_curve, add_noise

GRID = np.linspace(0.0, 0.70, 96)
GS = np.array([925., 740., 555., 370., 185.])
# phi coordinates (what the flow models)
PHI_NAMES = ['Iph_ref', 'log_I01', 'log_I02', 'log_Rs', 'log_Rsh_ref', 'logit_fc', 'log_Rc', 'T_cell']
LO = np.array([7.0, -12.0, -9.0, -2.7, -1.0, -6.0, -3.0, 15.0])
HI = np.array([10.2, -9.0, -4.0, -0.9, 3.5, 1.0, 1.0, 50.0])


def sample_prior(n, rng):
    return LO + (HI - LO) * rng.random((n, len(LO)))


def phi_to_theta(phi):
    return dict(Iph_ref=phi[:, 0], I01=10 ** phi[:, 1], I02=10 ** phi[:, 2], Rs=10 ** phi[:, 3],
                Rsh=10 ** phi[:, 4], fc=1 / (1 + np.exp(-phi[:, 5])), Rc=10 ** phi[:, 6])


EG = 1.12
TMIN, TMAX = 15.0, 50.0       # cell temperature prior (C), inferred jointly (8th dimension of phi)
GAIN_LO, GAIN_HI = 0.88, 1.15
T_MEAS_SD = 3.0               # temperature sensor uncertainty (K) for the measured-temperature input  # per-curve irradiance calibration nuisance (marginalised, not inferred)


def temp_scale(T):
    from physics import KB, Q
    Tk = T + 273.15
    s1 = (Tk / 298.15) ** 3 * np.exp(EG / (KB / Q) * (1 / 298.15 - 1 / Tk))
    s2 = (Tk / 298.15) ** 1.5 * np.exp(EG / (2 * KB / Q) * (1 / 298.15 - 1 / Tk))
    return s1, s2, KB * Tk / Q


def simulate(phi, rng, noise=True, smin=1.0, smax=20.0, nuisance=True):
    """phi[:, :7] as before (I01, I02 referred to 25 C), phi[:, 7] = cell temperature (C).
    Nuisances: per-curve irradiance gain g ~ U(GAIN_LO, GAIN_HI) and per-curve temperature jitter N(0, 1 K)."""
    n = len(phi)
    X = np.empty((n, len(GS), len(GRID)), np.float32)
    S = np.exp(rng.uniform(np.log(smin), np.log(smax), size=(n, len(GS)))).astype(np.float32)
    for j, G in enumerate(GS):
        T = phi[:, 7] + (rng.normal(0, 1.0, n) if nuisance else 0.0)
        s1, s2, vt = temp_scale(T)
        th = phi_to_theta(phi)
        th['I01'] = th['I01'] * s1; th['I02'] = th['I02'] * s2
        th['Iph_ref'] = th['Iph_ref'] * (1 + 0.0005 * (T - 25.0))
        g = rng.uniform(GAIN_LO, GAIN_HI, n) if nuisance else np.ones(n)
        I = _iv_vt(GRID, G * g, th, vt)
        if noise:
            # empirical noise at physical scale; apply per-curve scale
            I = np.stack([add_noise(I[i:i + 1], GRID, rng, scale=S[i, j])[0] for i in range(n)]) if n < 50 else \
                _noise_vec(I, rng, S[:, j])
        X[:, j] = np.clip(I * 1000.0 / G, -2.0, 12.0)
    return X, S


def _iv_vt(V, Gtrue, th, vt):
    """iv_curve with a per-sample thermal voltage: group by rounded vt for vectorisation."""
    return iv_curve(V, Gtrue, th, vt=np.asarray(vt, float)[:, None])


def _noise_vec(I, rng, s, sd_V=0.33e-3, sd_I=3.7e-3):
    dIdV = np.gradient(I, GRID, axis=1)
    dV = rng.normal(0, sd_V, size=(len(I), 1)) * s[:, None]
    return I + dIdV * dV + rng.normal(0, sd_I, size=I.shape) * s[:, None]


def chunk(seed, n):
    rng = np.random.default_rng(seed)
    phi = sample_prior(n, rng)
    X, S = simulate(phi, rng)
    Tm = (phi[:, 7] + rng.normal(0, T_MEAS_SD, len(phi))).astype(np.float32)  # measured module temperature
    return phi.astype(np.float32), X, S, Tm


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    out = sys.argv[2] if len(sys.argv) > 2 else '../data/bank_train.npz'
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    t = time.time()
    parts = Parallel(n_jobs=2)(delayed(chunk)(seed0 + k, 5000) for k in range(N // 5000))
    phi = np.concatenate([p[0] for p in parts]); X = np.concatenate([p[1] for p in parts]); S = np.concatenate([p[2] for p in parts])
    Tm = np.concatenate([p[3] for p in parts])
    ok = np.isfinite(X).all((1, 2))
    np.savez_compressed(out, phi=phi[ok], X=X[ok], S=S[ok], Tm=Tm[ok], grid=GRID, Gs=GS)
    print('saved', out, ok.sum(), 'of', N, 'in %.0f s' % (time.time() - t))
