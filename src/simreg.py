"""Simulation-trained amortised point estimator: same set encoder as the NPE, MSE regression head (no uncertainty)."""
import numpy as np, torch, torch.nn as nn, time, json
from npe import Encoder, random_masks, MID, HALF
from real_obs import build
torch.set_num_threads(2); rng = np.random.default_rng(0); torch.manual_seed(0)
d = np.load('../data/bank_train_v3.npz'); phi = ((d['phi'] - MID) / HALF).astype(np.float32); X = d['X']; SS = d['S']; TM = d['Tm']
n = len(phi); idx = rng.permutation(n); va, tr = idx[:10000], idx[10000:]
class Reg(nn.Module):
    def __init__(s): super().__init__(); s.enc = Encoder(); s.head = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 8))
    def forward(s, X, M, S, T): return s.head(s.enc(X, M, S, T))
m = Reg(); E = 12; bs = 512; opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=E * (len(tr) // bs))
for ep in range(E):
    t = time.time(); perm = rng.permutation(tr); m.train()
    for k in range(len(tr) // bs):
        b = perm[k * bs:(k + 1) * bs]; M = torch.tensor(random_masks(bs, rng))
        loss = ((m(torch.tensor(X[b]), M, torch.tensor(SS[b]), torch.tensor(TM[b])) - torch.tensor(phi[b])) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        vl = float(((m(torch.tensor(X[va]), torch.ones(10000, 5, dtype=torch.bool), torch.tensor(SS[va]), torch.tensor(TM[va])) - torch.tensor(phi[va])) ** 2).mean())
    print(json.dumps(dict(epoch=ep, val_mse=vl, sec=time.time() - t)), flush=True)
St, Xr = build('front'); N = len(Xr)
with torch.no_grad():
    R = m(torch.tensor(Xr), torch.ones(N, 5, dtype=torch.bool), torch.ones(N, 5), torch.full((N,), 25.0)).numpy() * HALF + MID
np.save('../results/simreg_real.npy', R); torch.save(m.state_dict(), '../results/simreg.pt'); print('saved')
