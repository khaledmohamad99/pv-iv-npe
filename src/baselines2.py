"""Additional baselines (supervisor request): (i) deterministic least-squares parameters + LR, (ii) simulation-trained
amortised point estimator (same set encoder, MSE head) + LR. Same leave-one-family-out protocol as baselines.py."""
import numpy as np, pandas as pd, json, sys, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from real_obs import build
from simulate_bank import PHI_NAMES
St, X = build('front'); N = len(St)
CLS = ['ref', 'PID', 'interconnect', 'crack']; y = St['mode'].map({c: i for i, c in enumerate(CLS)}).values; fam = St.family.values
F = pd.read_csv('../results/fit_real_front.csv'); F['Rsh_ref'] = F.M1_Rsh * F.G / 1000.
def lsq_feats():
    out = []
    for _, r in St.iterrows():
        d = F[(F['sample'] == r['sample']) & (F.meas == r.meas)]
        v = np.c_[np.log10(d.M1_Rs), np.log10(d.Rsh_ref), np.log10(d.M1_I01), np.log10(d.M1_I02),
                  np.log(np.clip(d.M1_fc, 1e-4, 1 - 1e-4) / (1 - np.clip(d.M1_fc, 1e-4, 1 - 1e-4)))]
        out.append(np.r_[np.median(v, 0), v.std(0)])
    return np.array(out)
def lofo(Fx, name):
    proba = np.zeros((N, 4))
    for f in np.unique(fam):
        te = fam == f; tr = ~te
        sc = StandardScaler().fit(Fx[tr]); m = LogisticRegression(C=1.0, max_iter=4000, class_weight='balanced').fit(sc.transform(Fx[tr]), y[tr])
        full = np.zeros((te.sum(), 4)); full[:, m.classes_] = m.predict_proba(sc.transform(Fx[te])); proba[te] = full
    pred = proba.argmax(1)
    res = dict(acc=accuracy_score(y, pred), macroF1=f1_score(y, pred, average='macro'),
               per_family_acc={f: float(accuracy_score(y[fam == f], pred[fam == f])) for f in np.unique(fam)})
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 2) for a, b in v.items()}) for k, v in res.items()}), flush=True)
    return res
out = {}
path = '../results/baselines2.json'
if os.path.exists(path): out = json.load(open(path))
out['LSQ+LR'] = lofo(lsq_feats(), 'LSQ+LR')
P = np.load('../results/v3_real_posteriors.npz')['full']
cols = ['log_Rs', 'log_Rsh_ref', 'log_I01', 'log_I02', 'logit_fc']; J = [PHI_NAMES.index(c) for c in cols]
out['NPE-median-only+LR'] = lofo(np.median(P[:, :, J], 1), 'NPE-median-only+LR')
if os.path.exists('../results/simreg_real.npy'):
    R = np.load('../results/simreg_real.npy'); out['SimReg+LR'] = lofo(R[:, J], 'SimReg+LR')
for k in sorted(f for f in os.listdir('../results') if f.startswith('v3s') and f.endswith('_post.npz')):
    Q = np.load('../results/' + k)['full']; out['NPE+LR_' + k[:-9]] = lofo(np.c_[np.median(Q[:, :, J], 1), Q[:, :, J].std(1)], 'NPE+LR_' + k[:-9])
json.dump(out, open(path, 'w'), indent=1)
