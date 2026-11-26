"""Local (Fisher-information) identifiability analysis of degradation parameters from a single forward I-V curve.

phi (estimation coordinates, all dimensionless/log):
  0 ln tau      optical transmittance (soiling)          -> multiplies photocurrent
  1 ln gsens    irradiance-sensor gain error             -> believed G = gsens*G_true
  2 log10 I01   diffusion recombination
  3 log10 I02   junction recombination (PID raises it)
  4 log10 Rs    series resistance (interconnect failure)
  5 log10 Rsh   shunt resistance (PID lowers it)
  6 logit fc    cracked-area fraction
  7 log10 Rc    resistance connecting cracked area
Observation model: I_obs(V) = I(V; G_true) + noise, but the diagnostician only knows G_meas = gsens*G_true,
so the forward model in terms of the *known* G_meas uses Iph = Iph_ref * tau * G_meas / gsens.
=> tau and gsens enter only through tau/gsens: exactly non-identifiable from one curve (rank-deficient Fisher).
Noise: empirical replicate model (voltage jitter sd_V, white current sd_I), heteroscedastic diagonal.
"""
import numpy as np, itertools, json
from physics import iv_curve

PNAMES = ['ln_tau', 'ln_gsens', 'log_I01', 'log_I02', 'log_Rs', 'log_Rsh', 'logit_fc', 'log_Rc']
LABELS = ['Soiling τ', 'Sensor gain', 'I01', 'I02 (PID)', 'Rs (interconnect)', 'Rsh (PID shunt)', 'Crack area fc', 'Crack Rc']


def to_theta(phi, Iph_ref):
    return dict(Iph_ref=Iph_ref * np.exp(phi[0]) / np.exp(phi[1]), I01=10 ** phi[2], I02=10 ** phi[3],
                Rs=10 ** phi[4], Rsh=10 ** phi[5], fc=1 / (1 + np.exp(-phi[6])), Rc=10 ** phi[7])


def model(phi, G, V, Iph_ref):
    return iv_curve(V, G, to_theta(phi, Iph_ref))[0]


def fisher(phi, G, V, Iph_ref, sd_V=0.33e-3, sd_I=3.7e-3, scale=1.0, h=1e-4):
    I0 = model(phi, G, V, Iph_ref)
    J = np.zeros((len(V), len(phi)))
    for k in range(len(phi)):
        dp = np.zeros_like(phi); dp[k] = h
        J[:, k] = (model(phi + dp, G, V, Iph_ref) - model(phi - dp, G, V, Iph_ref)) / (2 * h)
    dIdV = np.gradient(I0, V)
    var = (scale * sd_I) ** 2 + (scale * sd_V * dIdV) ** 2
    return J.T @ (J / var[:, None]), I0


def vgrid(phi, G, Iph_ref, n=120):
    Vd = np.linspace(0, 0.8, 801)
    I = model(phi, G, Vd, Iph_ref)
    voc = np.interp(0, -I, Vd)
    return np.linspace(0, voc, n)


def posterior_cov(F, prior_sd):
    P = np.diag(1 / np.asarray(prior_sd) ** 2)
    return np.linalg.inv(F + P)


if __name__ == '__main__':
    pass
