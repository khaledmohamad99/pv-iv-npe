import sys, torch
import npe
torch.set_num_threads(1)
s = int(sys.argv[1])
npe.train(out=f'../results/npe_v3_seed{s}.pt', epochs=32, seed=s)
