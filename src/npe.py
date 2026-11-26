"""Set-conditioned neural posterior estimation (NPE) for PV I-V diagnosis.
Encoder: shared MLP over each available curve (current/ (G/1000) on fixed grid + irradiance), masked mean pooling
(DeepSets) + availability mask -> context. Posterior: conditional neural spline flow (zuko NSF) over standardised phi.
Training draws a random non-empty subset of the five irradiance levels per example, so one network serves single-curve
and any multi-irradiance protocol (needed for the measurement-design study).
"""
import numpy as np, torch, torch.nn as nn, zuko, time, sys, json
from simulate_bank import LO, HI, GS, PHI_NAMES

torch.set_num_threads(2)
MID, HALF = (LO + HI) / 2, (HI - LO) / 2


class Encoder(nn.Module):
    def __init__(self, n_v=96, h=256, out=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_v + 3, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, out))
        self.g = nn.Sequential(nn.Linear(out + 5, h), nn.GELU(), nn.Linear(h, out))

    def forward(self, X, M, S, Tm):  # X (b,5,n_v); M (b,5) mask; S (b,5) noise scale; Tm (b,) measured temperature
        g = torch.tensor(GS / 1000., dtype=X.dtype, device=X.device).expand(X.shape[0], 5)[..., None]
        ls = (torch.log(S) / 3.0)[..., None]; tm = ((Tm - 25.0) / 10.0)[:, None, None].expand(-1, 5, 1)
        e = self.f(torch.cat([X / 10.0, g, ls, tm], -1))                     # (b,5,out)
        m = M.float()[..., None]
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1)
        return self.g(torch.cat([pooled, M.float()], -1))


class NPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.flow = zuko.flows.NSF(features=len(LO), context=128, transforms=5, hidden_features=[256, 256])

    def log_prob(self, phi_std, X, M, S, Tm):
        return self.flow(self.enc(X, M, S, Tm)).log_prob(phi_std)

    @torch.no_grad()
    def sample(self, X, M, S, Tm, n=2000):
        """Returns posterior samples in phi units, shape (b, n, d); out-of-prior draws are clipped."""
        d = self.flow(self.enc(X, M, S, Tm))
        z = d.sample((n,)).transpose(0, 1).numpy()
        phi = z * HALF + MID
        return np.clip(phi, LO, HI)


def random_masks(b, rng):
    k = rng.integers(1, 6, size=b)
    M = np.zeros((b, 5), bool)
    for i in range(b):
        M[i, rng.choice(5, k[i], replace=False)] = True
    return M


def train(bank='../data/bank_train_v3.npz', out='../results/npe_v3.pt', epochs=40, bs=512, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    d = np.load(bank); phi = (d['phi'] - MID) / HALF; X = d['X']; SS = d['S']; TM = d['Tm']
    n = len(phi); idx = rng.permutation(n); nv = 10000
    va, tr = idx[:nv], idx[nv:]
    model = NPE(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * (len(tr) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    Mva = torch.tensor(random_masks(nv, rng)); Xva = torch.tensor(X[va]); Pva = torch.tensor(phi[va], dtype=torch.float32)
    Sva = torch.tensor(SS[va]); Tva = torch.tensor(TM[va])
    best, log = 1e9, []
    for ep in range(epochs):
        t = time.time(); model.train(); perm = rng.permutation(tr); tl = []
        for s in range(len(tr) // bs):
            b = perm[s * bs:(s + 1) * bs]
            M = torch.tensor(random_masks(bs, rng))
            loss = -model.log_prob(torch.tensor(phi[b], dtype=torch.float32), torch.tensor(X[b]), M, torch.tensor(SS[b]), torch.tensor(TM[b])).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = np.mean([-model.log_prob(Pva[i:i + 2000], Xva[i:i + 2000], Mva[i:i + 2000], Sva[i:i + 2000], Tva[i:i + 2000]).mean().item() for i in range(0, nv, 2000)])
        log.append(dict(epoch=ep, train=float(np.mean(tl)), val=float(vl), sec=time.time() - t))
        print(json.dumps(log[-1]), flush=True)
        if vl < best:
            best = vl; torch.save(model.state_dict(), out)
    json.dump(log, open(out.replace('.pt', '_log.json'), 'w'))


def load(path='../results/npe_v3.pt'):
    m = NPE(); m.load_state_dict(torch.load(path)); m.eval(); return m


if __name__ == '__main__':
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
