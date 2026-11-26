# Amortised neural posterior estimation for label-free PV degradation diagnosis from I–V curves

Code accompanying: *From current–voltage curves to calibrated degradation posteriors: physics-based amortised neural inference for label-free photovoltaic diagnosis* (Abdel Aziz, Medhat, Shams, Elseddik).

## Requirements
Python ≥ 3.10, numpy, scipy, pandas, scikit-learn, torch, zuko, emcee, matplotlib.

## Data (public)
- Garaj et al. (2021), single-cell I–V dataset: https://doi.org/10.17632/vxj8bkkfkg.1 → set `GARAJ_DIR`
- Restrepo-Cuestas et al. (2025), outdoor module dataset: Mendeley Data xjs42j8dtf (v2) → set `ITM_DIR`

## Pipeline (run from `src/`, outputs go to `../results`, `../figs`, `../data`)
1. `simulate_bank.py` – simulation bank (3×10^5 training, 10^4 test cells)
2. `npe.py 32` – train the set-conditioned neural spline flow (`train_seed.py <seed>` for extra seeds)
3. `run_ident.py` – Fisher identifiability and expected information gain
4. `fit_real.py`, `joint_fit_gainT.py` – deterministic least-squares fits
5. `eval_v3.py` – calibration, cross-irradiance test, real-data posteriors, severity trends, measurement design
6. `eval_seed.py <seed>` – identical evaluation for each trained network
7. `stats_severity.py` – exact/stratified permutation tests and posterior-propagated intervals
8. `mcmc_sim.py`, `simbench_eval.py` – well-specified exact-MCMC benchmark
9. `mcmc_v5.py` (env `TAG`, `S_INFL`, `T_SD`) and `ppc_v7.py` – real-data MCMC with/without discrepancy term
10. `baselines.py`, `simreg.py`, `baselines2.py`, `baselines3.py` – leave-one-family-out baselines (incl. least-squares + NPE combined)
11. `itm_case.py`, `field_noise.py` – module case study and field-grade noise
12. `fig_*.py` – figures

Trained networks (three seeds) are in `models/`.

## License
Released under the MIT License (see `LICENSE`).
