"""LSQ + NPE features combined (supervisor suggestion), same LOFO protocol."""
import numpy as np, json
import baselines2 as B
from simulate_bank import PHI_NAMES
cols = ['log_Rs', 'log_Rsh_ref', 'log_I01', 'log_I02', 'logit_fc']; J = [PHI_NAMES.index(c) for c in cols]
L = B.lsq_feats(); out = {}
for tag, path in [('main', '../results/v3_real_posteriors.npz'), ('s0', '../results/v3s0_post.npz'), ('s1', '../results/v3s1_post.npz'), ('s2', '../results/v3s2_post.npz')]:
    P = np.load(path)['full']; N = np.c_[np.median(P[:, :, J], 1), P[:, :, J].std(1)]
    out[tag] = B.lofo(np.c_[L, N], 'LSQ+NPE+LR_' + tag)
json.dump(out, open('../results/baselines3.json', 'w'), indent=1)
