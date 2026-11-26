import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from load_data import load_garaj
from fit_real import prep
from physics import iv_curve
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','font.size':9,'savefig.facecolor':'white'})
meta,cur=load_garaj(); F=pd.read_csv('../results/fit_real_front.csv')
fig,ax=plt.subplots(1,3,figsize=(14,3.8))
# (a) crack stages G_1 at 925: data vs M1 fit (fits were made with constant-Rsh model -> evaluate with shunt_scaling=False)
cols=plt.cm.viridis(np.linspace(0,.85,4))
for c,m in zip(cols,['ref','crack_1','crack_2','crack_3']):
    k=meta.index[(meta['sample']=='G_1')&(meta.meas==m)&(meta.irr=='a')][0]; r=F.loc[k]
    v,i,_=prep(*cur[k]); th={p:r[f'M1_{p}'] for p in ['Iph_ref','I01','I02','Rs','Rsh','fc','Rc']}
    ax[0].plot(v,i,'o',ms=2,color=c,alpha=.5); ax[0].plot(v,iv_curve(v,925.,th,shunt_scaling=False)[0],'-',color=c,lw=1.2,label=f'{m} (RMSE {100*r.M1_rmse:.2f}% $I_{{sc}}$)')
ax[0].set_xlabel('Voltage (V)'); ax[0].set_ylabel('Current (A)'); ax[0].set_title('(a) Cell G_1, crack stages, 925 W m$^{-2}$'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
# (b) interconnect: fitted Rs vs fraction of ribbons cut
I=F[F.sample_mode=='interconnect']
for s,d in I[I.irr=='a'].groupby('sample'): ax[1].plot(d.severity,1e3*d.M1_Rs,'o-',ms=4,label=s)
ax[1].set_xlabel('Fraction of ribbon connections cut (ground truth)'); ax[1].set_ylabel('Fitted $R_s$ (mΩ)'); ax[1].set_yscale('log'); ax[1].set_title('(b) Interconnect failures, 925 W m$^{-2}$'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3,which='both')
# (c) PID: fitted Rsh_ref vs PID stage at 185 W/m2
P=F[F.sample_mode=='PID'].copy(); P['Rsh_ref']=P.M1_Rsh*P.G/1000
for s,d in P[P.irr=='e'].groupby('sample'): d=d.sort_values('severity'); ax[2].plot(d.severity,d.Rsh_ref,'o-',ms=4,label=s)
ax[2].set_xlabel('PID stress stage (ground truth)'); ax[2].set_ylabel('Fitted $R_{sh}$ at 1000 W m$^{-2}$ (Ω)'); ax[2].set_yscale('log'); ax[2].set_title('(c) PID, 185 W m$^{-2}$'); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3,which='both')
plt.tight_layout(); plt.savefig('../figs/fig_sim_adequacy.png',dpi=300)
