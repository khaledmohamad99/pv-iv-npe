"""Supervised baselines vs label-free NPE features on real Garaj cells, leave-one-family-out (A,B,C | E,F | G,H).
Task: degradation-mode classification {ref, PID, interconnect, crack} from the five-irradiance observation.
 - RF-raw: random forest on the 5x96 normalised curves
 - RF-feat: random forest on engineered I-V features (Isc, Voc, FF, Pmp, low-V slope, high-V slope per irradiance)
 - CNN: small 1D CNN on the 5-channel curves
 - NPE+LR: logistic regression on NPE posterior summaries (medians + SDs of log Rs, log Rsh, log I01, log I02, logit fc)
NPE itself never saw a label; only the small LR head does. Also reported: within-sample severity Spearman of the
predicted probability of the true mode (supervised) vs the physical posterior parameter (NPE)."""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import spearmanr
from real_obs import build
from simulate_bank import GRID, PHI_NAMES

torch.set_num_threads(1)
St, X = build('front'); N = len(St)
CLS = ['ref', 'PID', 'interconnect', 'crack']; y = St['mode'].map({c: i for i, c in enumerate(CLS)}).values
fam = St.family.values


def feats(X):
    F = []
    for x in X:
        f = []
        for c in x:
            isc = c[0]; i0 = np.where(c <= 0)[0]; voc = GRID[i0[0]] if len(i0) else GRID[-1]
            p = GRID * c; pm = p.max(); ff = pm / (isc * voc + 1e-9)
            s_lo = np.polyfit(GRID[:15], c[:15], 1)[0]; k = max(2, (i0[0] if len(i0) else len(c)) - 1)
            s_hi = (c[k] - c[k - 2]) / (GRID[k] - GRID[k - 2])
            f += [isc, voc, ff, pm, s_lo, s_hi]
        F.append(f)
    return np.array(F)


class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.n = nn.Sequential(nn.Conv1d(5, 16, 5, padding=2), nn.GELU(), nn.MaxPool1d(2), nn.Conv1d(16, 32, 5, padding=2), nn.GELU(),
                            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 4))
    def forward(s, x): return s.n(x)


def fit_cnn(Xtr, ytr, Xte, seed=0):
    torch.manual_seed(seed); m = CNN(); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    xt = torch.tensor(Xtr / 10, dtype=torch.float32); yt = torch.tensor(ytr)
    w = torch.tensor(len(ytr) / (4 * np.maximum(np.bincount(ytr, minlength=4), 1)), dtype=torch.float32)
    for _ in range(400):
        opt.zero_grad(); l = nn.functional.cross_entropy(m(xt), yt, weight=w); l.backward(); opt.step()
    with torch.no_grad(): return torch.softmax(m(torch.tensor(Xte / 10, dtype=torch.float32)), 1).numpy()


P = np.load('../results/v3_real_posteriors.npz')['full']
cols = ['log_Rs', 'log_Rsh_ref', 'log_I01', 'log_I02', 'logit_fc']; J = [PHI_NAMES.index(c) for c in cols]
NPEF = np.c_[np.median(P[:, :, J], 1), P[:, :, J].std(1)]
FE = feats(X); RAW = X.reshape(N, -1)

methods = {}
for name in ['RF-raw', 'RF-feat', 'CNN', 'NPE+LR']:
    proba = np.zeros((N, 4))
    for f in np.unique(fam):
        te = fam == f; tr = ~te
        if name == 'RF-raw':
            m = RandomForestClassifier(500, random_state=0, class_weight='balanced').fit(RAW[tr], y[tr]); pr = m.predict_proba(RAW[te])
        elif name == 'RF-feat':
            m = RandomForestClassifier(500, random_state=0, class_weight='balanced').fit(FE[tr], y[tr]); pr = m.predict_proba(FE[te])
        elif name == 'CNN':
            pr = np.mean([fit_cnn(X[tr], y[tr], X[te], s) for s in range(3)], 0)
        else:
            sc = StandardScaler().fit(NPEF[tr]); m = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(sc.transform(NPEF[tr]), y[tr])
            pr = m.predict_proba(sc.transform(NPEF[te]))
        full = np.zeros((te.sum(), 4)); full[:, m.classes_ if name != 'CNN' else np.arange(4)] = pr; proba[te] = full
    pred = proba.argmax(1)
    # severity: Spearman, within each degraded sample, between severity and P(true mode)
    rho = []
    for s, d in St.groupby('sample'):
        if d.sample_mode.iloc[0] in ('PID', 'interconnect', 'crack') and d.severity.nunique() > 2:
            k = CLS.index(d.sample_mode.iloc[0]); rho.append(spearmanr(d.severity, proba[d.index, k]).statistic)
    methods[name] = dict(acc=accuracy_score(y, pred), macroF1=f1_score(y, pred, average='macro'),
                         per_family_acc={f: float(accuracy_score(y[fam == f], pred[fam == f])) for f in np.unique(fam)},
                         severity_rho_median=float(np.nanmedian(rho)))
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in methods[name].items()}), flush=True)
json.dump(methods, open('../results/baselines_lofo.json', 'w'), indent=1)
