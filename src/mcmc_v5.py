"""Reference MCMC matching the NPE generative model exactly: 8 physical parameters (T_cell global) + 5 per-curve
irradiance gains U(0.88,1.15) + 5 per-curve temperature offsets N(0,1 K). Initialised at the joint least-squares optimum
(gain + temperature model, ~0.16-0.28% Isc RMSE) to avoid the non-convergence of random initialisation in 18-D."""
import numpy as np, emcee, time, sys, json
from scipy.optimize import least_squares
from physics import iv_curve
from simulate_bank import GRID, GS, LO, HI, phi_to_theta, temp_scale, GAIN_LO, GAIN_HI
from real_obs import build
import os
S_INFL = float(os.environ.get('S_INFL', '2.0')); SD_I, SD_V = 3.7e-3 * S_INFL, 0.33e-3 * S_INFL
T_MEAS, T_SD = 25.0, float(os.environ.get('T_SD', '3.0'))  # measured-temperature prior, identical to the NPE v3 input model
LOA = np.r_[LO, [GAIN_LO] * 5, [-4.0] * 5]; HIA = np.r_[HI, [GAIN_HI] * 5, [4.0] * 5]

def curves(Q):  # Q (n,18) -> list of 5 arrays (n,96) physical current
    out = []
    for j, G in enumerate(GS):
        T = Q[:, 7] + Q[:, 13 + j]; s1, s2, vt = temp_scale(T); th = phi_to_theta(Q)
        th['I01'] = th['I01'] * s1; th['I02'] = th['I02'] * s2; th['Iph_ref'] = th['Iph_ref'] * (1 + 0.0005 * (T - 25))
        out.append(iv_curve(GRID, G * Q[:, 8 + j], th, vt=vt[:, None]))
    return out

def make_logpost(Xobs):
    Iobs = Xobs * (GS[:, None] / 1000.0); keep = (Xobs > -2 + 1e-6) & (Xobs < 12 - 1e-6)  # v7: exclude clipped points
    def lp(P):
        inb = np.all((P >= LOA) & (P <= HIA), 1); out = np.full(len(P), -np.inf)
        if not inb.any(): return out
        Q = P[inb]; ll = -0.5 * np.sum(Q[:, 13:] ** 2, 1) - 0.5 * ((Q[:, 7] - T_MEAS) / T_SD) ** 2  # N(0,1K) prior on per-curve temperature offsets
        for j, I in enumerate(curves(Q)):
            dIdV = np.gradient(I, GRID, axis=1); var = SD_I ** 2 + (SD_V * dIdV) ** 2
            ll += -0.5 * np.sum(np.where(keep[j], (I - Iobs[j]) ** 2 / var + np.log(var), 0.0), 1)
        out[inb] = ll; return out
    return lp

def map_init_fast(i):
    """Initial point from the joint gain+temperature least-squares fit (joint_fit_gainT)."""
    from joint_fit_gainT import fit
    rm, x = fit(i)
    p, g, dT = x[:7], np.r_[1.0, x[7:11]], x[11:]
    T = 25.0 + dT.mean()
    return np.r_[p, T, g, dT - dT.mean()], rm


def map_init(Xobs):
    Iobs = Xobs * (GS[:, None] / 1000.0); keep = (Xobs > -2 + 1e-6) & (Xobs < 12 - 1e-6)  # v7: exclude clipped points
    def res(q):
        r = [(I[0] - Iobs[j])[keep[j]] / Iobs[0, 0] for j, I in enumerate(curves(q[None]))]
        return np.r_[np.concatenate(r), 0.002 * q[13:]]
    best = None
    for s in [[9, -10.5, -6, -2.2, 1.5, -2, -0.5, 35], [9, -10, -7, -2.0, 2.0, -3, 0, 30], [9, -11, -5.5, -2.4, 1.0, -1, -1, 40]]:
        q0 = np.r_[s, [1.0, 1.03, 1.05, 1.06, 1.01], [0.0] * 5]
        r = least_squares(res, q0, bounds=(LOA + 1e-6, HIA - 1e-6), x_scale='jac', max_nfev=1500)
        if best is None or r.cost < best.cost: best = r
    return best.x, 100 * np.sqrt(np.mean(best.fun[:-5] ** 2))

def run(Xobs, i, nw=64, nsteps=5000, seed=0):
    rng = np.random.default_rng(seed); lp = make_logpost(Xobs)
    q, rm = map_init_fast(i)
    q = np.clip(q, LOA + 1e-6, HIA - 1e-6)  # v6: keep the least-squares temperature (no override)
    p0 = q + 1e-4 * (HIA - LOA) * rng.standard_normal((nw, 18)); p0 = np.clip(p0, LOA + 1e-7, HIA - 1e-7)
    s = emcee.EnsembleSampler(nw, 18, lp, vectorize=True, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    t = time.time(); s.run_mcmc(p0, nsteps, progress=False); el = time.time() - t
    ch = s.get_chain(discard=nsteps // 2, thin=10, flat=True)
    tau = float(np.max(s.get_autocorr_time(discard=nsteps // 2, quiet=True)))
    return ch, el, tau, float(np.mean(s.acceptance_fraction)), rm, q

if __name__ == '__main__':
    St, Xr = build('front')
    for i in [int(x) for x in sys.argv[1].split(',')]:
        ch, el, tau, acc, rm, q = run(Xr[i], i); np.save(f'../results/mcmc_{os.environ.get("TAG", "v6")}_chain_{i}.npy', ch)
        print(i, St.loc[i, ['sample', 'meas']].tolist(), json.dumps(dict(sec=el, tau=tau, acc=acc, map_rmse_pct=rm, n=len(ch))), flush=True)
