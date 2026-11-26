"""Deterministic least-squares fit of the physics model to every real Garaj front-side curve.
Purpose: (i) simulator adequacy (residual RMSE), (ii) do fitted parameters follow the ground-truth stages?
Two models: M0 = two-diode (no crack branch); M1 = two-diode + crack sub-region (fc, Rc).
"""
import numpy as np, pandas as pd, sys
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from load_data import load_garaj
from physics import iv_curve


def prep(V, I, npts=120):
    m = I.mean(0) if I.ndim == 2 else I
    voc = np.interp(0, -m, V)
    f = (V >= 0) & (V <= voc + 0.01)
    v, i = V[f], m[f]
    grid = np.linspace(0, v.max(), npts)
    return grid, np.interp(grid, v, i), voc


def unpack(p, crack):
    th = dict(Iph_ref=p[0], I01=10 ** p[1], I02=10 ** p[2], Rs=10 ** p[3], Rsh=10 ** p[4])
    if crack:
        th['fc'] = 1 / (1 + np.exp(-p[5])); th['Rc'] = 10 ** p[6]
    return th


def fit_one(v, i, G, crack, seed=0):
    rng = np.random.default_rng(seed)
    isc = i[0]
    best = None
    starts = [[isc * 1000 / G, -11, -7.5, -2.3, 1.5], [isc * 1000 / G, -10, -6.5, -2.0, 2.5],
              [isc * 1000 / G, -12, -8.5, -2.6, 1.0], [isc * 1000 / G, -9.5, -7.0, -1.7, 2.0]]
    lb = [0.5 * isc * 1000 / G, -16, -12, -4, -1]
    ub = [1.5 * isc * 1000 / G, -6, -3, 0, 5]
    if crack:
        starts = [s + [c, r] for s in starts for c, r in [(-2.0, -0.5), (-1.0, 0.0), (-3.0, -1.5)]]
        lb += [-8, -3]; ub += [3, 3]
    for s in starts:
        r = least_squares(lambda p: (iv_curve(v, G, unpack(p, crack))[0] - i) / isc,
                          s, bounds=(lb, ub), x_scale='jac', max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    th = unpack(best.x, crack)
    rmse = np.sqrt(2 * best.cost / len(v))  # relative to Isc
    return th, rmse


def run(idx, meta, curves):
    r = meta.loc[idx]
    V, R = curves[idx]
    v, i, voc = prep(V, R)
    out = dict(idx=idx)
    for crack in (False, True):
        th, rmse = fit_one(v, i, r.G, crack)
        tag = 'M1' if crack else 'M0'
        out[f'{tag}_rmse'] = rmse
        for k, val in th.items():
            out[f'{tag}_{k}'] = val
    return out


if __name__ == '__main__':
    meta, curves = load_garaj('front')
    res = Parallel(n_jobs=2, verbose=0)(delayed(run)(k, meta, curves) for k in meta.index)
    R = pd.DataFrame(res).set_index('idx')
    out = meta.join(R)
    out.to_csv('../results/fit_real_front.csv', index=False)
    print(out[['M0_rmse', 'M1_rmse']].describe().round(5))
