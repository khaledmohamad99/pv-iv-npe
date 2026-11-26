"""Well-specified benchmark: observations simulated from exactly the NPE (v3) generative model; reference posterior by
MCMC with the exact likelihood. Unknowns: 8 physical parameters + 5 irradiance gains + 5 per-curve temperature offsets
+ 5 per-curve voltage shifts (noise model: I + dI/dV * dV + white noise). Clipped points are excluded, as in the
simulator's [-2, 12] normalisation."""
import numpy as np, emcee, time, sys, json
from physics import iv_curve
from simulate_bank import (GRID, GS, LO, HI, phi_to_theta, temp_scale, GAIN_LO, GAIN_HI, T_MEAS_SD, sample_prior)
SD_I0, SD_V0 = 3.7e-3, 0.33e-3
LOA = np.r_[LO, [GAIN_LO] * 5, [-5.0] * 5, [-5.0] * 5]; HIA = np.r_[HI, [GAIN_HI] * 5, [5.0] * 5, [5.0] * 5]


def model_curves(Q):
    out = []
    for j, G in enumerate(GS):
        T = Q[:, 7] + Q[:, 13 + j]; s1, s2, vt = temp_scale(T); th = phi_to_theta(Q)
        th['I01'] = th['I01'] * s1; th['I02'] = th['I02'] * s2; th['Iph_ref'] = th['Iph_ref'] * (1 + 0.0005 * (T - 25))
        out.append(iv_curve(GRID, G * Q[:, 8 + j], th, vt=vt[:, None]))
    return out


def observe(Q, s, rng):
    """Q (1,23) true state with dV in units of SD_V0*s; returns normalised obs (5,96) exactly like the bank."""
    X = np.empty((5, len(GRID)), np.float32)
    for j, I in enumerate(model_curves(Q)):
        dIdV = np.gradient(I[0], GRID); dV = Q[0, 18 + j] * SD_V0 * s[j]
        Io = I[0] + dIdV * dV + rng.normal(0, SD_I0 * s[j], len(GRID))
        X[j] = np.clip(Io * 1000 / G_(j), -2, 12)
    return X


def G_(j): return GS[j]


def make_logpost(X, s, Tm):
    keep = (X > -2 + 1e-6) & (X < 12 - 1e-6)
    def lp(P):
        inb = np.all((P >= LOA) & (P <= HIA), 1); out = np.full(len(P), -np.inf)
        if not inb.any(): return out
        Q = P[inb]
        ll = -0.5 * np.sum(Q[:, 13:18] ** 2, 1) - 0.5 * np.sum(Q[:, 18:23] ** 2, 1) - 0.5 * ((Tm - Q[:, 7]) / T_MEAS_SD) ** 2
        for j, I in enumerate(model_curves(Q)):
            dIdV = np.gradient(I, GRID, axis=1); Im = I + dIdV * (Q[:, 18 + j] * SD_V0 * s[j])[:, None]
            r = (Im * 1000 / GS[j] - X[j]) / (SD_I0 * s[j] * 1000 / GS[j])
            ll += -0.5 * np.sum(np.where(keep[j], r ** 2, 0.0), 1)
        out[inb] = ll; return out
    return lp


def make_case(seed, s_level):
    rng = np.random.default_rng(seed)
    phi = sample_prior(1, rng)[0]
    Q = np.r_[phi, rng.uniform(GAIN_LO, GAIN_HI, 5), rng.normal(0, 1, 5), rng.normal(0, 1, 5)][None]
    s = np.full(5, s_level, float); X = observe(Q, s, rng); Tm = phi[7] + rng.normal(0, T_MEAS_SD)
    return Q[0], X, s, Tm


if __name__ == '__main__':
    cases = [(101, 2.0), (102, 2.0), (103, 2.0), (201, 8.0), (202, 8.0), (203, 8.0)]
    pick = [int(x) for x in sys.argv[1].split(',')]
    for c in pick:
        seed, sl = cases[c]; Q, X, s, Tm = make_case(seed, sl)
        lp = make_logpost(X, s, Tm); rng = np.random.default_rng(seed)
        nw = 72; p0 = np.clip(Q + 1e-4 * (HIA - LOA) * rng.standard_normal((nw, 23)), LOA + 1e-7, HIA - 1e-7)
        smp = emcee.EnsembleSampler(nw, 23, lp, vectorize=True, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        t = time.time(); smp.run_mcmc(p0, 7000, progress=False); el = time.time() - t
        ch = smp.get_chain(discard=3500, thin=10, flat=True)
        tau = float(np.max(smp.get_autocorr_time(discard=3500, quiet=True)))
        np.savez(f'../results/simbench_case{c}.npz', chain=ch, truth=Q, X=X, s=s, Tm=Tm)
        print(c, json.dumps(dict(sec=el, tau=tau, acc=float(np.mean(smp.acceptance_fraction)))), flush=True)
