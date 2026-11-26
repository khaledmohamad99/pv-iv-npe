"""Turn Garaj real curves into NPE observations: per (sample, state) a (5, 96) array on the simulator grid,
current normalised by G/1000 (replicate mean). Duplicate reference files (E_1..E_3 and F_1..F_3 share one ref
measurement) are collapsed to a single state to avoid double counting."""
import numpy as np, pandas as pd
from load_data import load_garaj
from simulate_bank import GRID, GS

IRR_ORDER = ['a', 'b', 'c', 'd', 'e']  # 925 ... 185, matches GS


def build(side='front'):
    meta, curves = load_garaj(side)
    states, X = [], []
    for (s, m), d in meta.groupby(['sample', 'meas'], sort=False):
        if m == 'ref' and s in ('E_2', 'E_3', 'F_2', 'F_3'):
            continue  # duplicated reference measurement
        if set(d.irr) != set(IRR_ORDER):
            continue
        arr = np.zeros((5, len(GRID)), np.float32)
        for j, g in enumerate(IRR_ORDER):
            k = d.index[d.irr == g][0]
            V, R = curves[k]
            i = R.mean(0)
            arr[j] = np.clip(np.interp(GRID, V, i) * 1000.0 / GS[j], -2.0, 12.0)
        r = d.iloc[0]
        states.append(dict(sample=s, family=s[0], meas=m, mode=r['mode'], severity=r.severity,
                           sample_mode=r.sample_mode, side=side))
        X.append(arr)
    return pd.DataFrame(states), np.stack(X)


if __name__ == '__main__':
    S, X = build()
    print(len(S), X.shape); print(S.groupby(['sample_mode', 'mode']).size())
