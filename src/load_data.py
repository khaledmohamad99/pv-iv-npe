"""Parse Garaj (DTU/CityU) single-cell I-V curves and ITM module curves into tidy arrays.

Garaj: each *_I.csv holds a voltage grid `_V` and replicate sweeps I0..Ik (same cell, same condition).
Irradiance codes a..e correspond to 925, 740, 555, 370, 185 W/m2 (dataset description: five levels 185-925 W/m2).
"""
import glob, re, os
import numpy as np, pandas as pd, scipy.io as sio

import os
ROOT = os.environ.get('GARAJ_DIR', 'data/PV_IV_Data')  # Garaj et al. (2021), doi:10.17632/vxj8bkkfkg.1
ITM = os.environ.get('ITM_DIR', 'data/ITM')  # Restrepo-Cuestas et al. (2025), Mendeley xjs42j8dtf v2
G_LEVEL = dict(a=925., b=740., c=555., d=370., e=185.)


def parse_meas(sample, meas):
    """Map folder name to (mode, severity) where severity is an ordinal ground truth."""
    if meas == 'ref':
        return 'ref', 0.0
    if meas.startswith('PID_'):
        return 'PID', float(meas.split('_')[1])            # hours/percent stage 20..100
    if meas.startswith('crack_'):
        return 'crack', float(meas.split('_')[1])          # stage 1..3
    # interconnect pattern p<bits>_n<bits>: 1 = ribbon intact, 0 = cut (positive / negative side)
    m = re.match(r'p(\d+)_n(\d+)', meas)
    p, n = m.group(1), m.group(2)
    cut = p.count('0') + n.count('0')
    frac = cut / (len(p) + len(n))
    return 'interconnect', frac                            # fraction of ribbon connections cut


def load_garaj(side='front'):
    rows, curves = [], []
    for f in sorted(glob.glob(f'{ROOT}/*/*/{side}/*_I.csv')):
        sample, meas, _, fn = f.replace(ROOT + '/', '').split('/')
        irr = fn[0]
        d = pd.read_csv(f)
        V = d['_V'].values.astype(float)
        reps = np.stack([d[c].values for c in d.columns if c.startswith('I')], 0).astype(float)
        o = np.argsort(V)
        V, reps = V[o], reps[:, o]
        mode, sev = parse_meas(sample, meas)
        rows.append(dict(sample=sample, family=sample[0], meas=meas, mode=mode, severity=sev,
                         irr=irr, G=G_LEVEL[irr], side=side, nrep=reps.shape[0]))
        curves.append((V, reps))
    meta = pd.DataFrame(rows)
    # tag degradation family of each sample (its non-ref measurements)
    fam = meta[meta['mode'] != 'ref'].groupby('sample')['mode'].first()
    meta['sample_mode'] = meta['sample'].map(fam)
    return meta, curves


def load_itm():
    rows, curves = [], []
    for f in sorted(glob.glob(f'{ITM}/RawData/*/ElectricalData/*.mat')):
        mod = f.split('/RawData/')[1].split('/')[0]
        name = os.path.basename(f)[:-4]
        panel, cond = name.split('_')
        d = sio.loadmat(f, squeeze_me=True)
        rows.append(dict(module=mod, panel=panel, cond=cond, G=float(d['IRR']), T=float(np.ravel(d['T'])[0])))
        curves.append((np.asarray(d['Vpanel'], float), np.asarray(d['Ipanel'], float)))
    return pd.DataFrame(rows), curves


if __name__ == '__main__':
    meta, curves = load_garaj('front')
    print(meta.groupby(['sample_mode', 'mode']).size())
    print(meta.groupby('sample_mode')['sample'].unique())
    V, R = curves[0]
    print('grid', V.min(), V.max(), len(V), 'reps', R.shape)
    print(meta[meta['mode'] == 'interconnect'][['sample', 'meas', 'severity']].drop_duplicates().to_string())
    mi, ci = load_itm()
    print(mi.to_string())
