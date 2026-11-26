import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from identifiability import fisher, vgrid, posterior_cov, LABELS
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','font.size':9,'savefig.facecolor':'white'})
R=pd.read_csv('../results/ident_single_curve.csv'); E=pd.read_csv('../results/ident_eig_subsets.csv')
states=['Healthy','PID','Interconnect','Crack']; Gs=[185,370,555,740,925]
order=['Soiling τ','I01','I02 (PID)','Rs (interconnect)','Rsh (PID shunt)','Crack area fc','Crack Rc']
# Fig A: contraction heatmaps (field noise)
fig,axs=plt.subplots(1,4,figsize=(13,3.4),sharey=True)
for ax,s in zip(axs,states):
    M=R[(R.state==s)&(R.noise=='field')].pivot_table(index='label',columns='G',values='contraction').loc[order,Gs]
    im=ax.imshow(M.values,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v=M.values[i,j]; ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,color='white' if v>0.6 else 'black')
    ax.set_xticks(range(5)); ax.set_xticklabels(Gs); ax.set_xlabel('Irradiance (W m$^{-2}$)'); ax.set_title(s)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order)
cb=fig.colorbar(im,ax=axs,fraction=0.015,pad=0.01); cb.set_label('Posterior contraction (1 = fully identified)')
plt.savefig('../figs/fig_ident_contraction_field.png',dpi=300,bbox_inches='tight')
# Fig B: EIG fraction vs number of irradiance levels (best subset)
fig,ax=plt.subplots(figsize=(5,3.4))
for s,c in zip(states,['#4C72B0','#DD8452','#55A868','#C44E52']):
    for n,ls in [('lab','-'),('field','--')]:
        b=E[(E.state==s)&(E.noise==n)].groupby('k').frac_of_all.max()
        ax.plot(b.index,b.values,ls,marker='o',color=c,label=f'{s} ({n})',ms=4)
ax.set_xlabel('Number of irradiance levels measured'); ax.set_ylabel('Information vs all five levels'); ax.set_xticks(range(1,6)); ax.grid(alpha=.3); ax.legend(fontsize=6.5,ncol=2)
plt.savefig('../figs/fig_eig_levels.png',dpi=300,bbox_inches='tight')
# Fig C: I02-Rs confounding single vs dual irradiance (PID state, field noise)
IPH=8.98; base=np.array([0.0,0.0,np.log10(3e-11),np.log10(1.5e-5),np.log10(0.006),np.log10(10.),-6.0,0.0]); keep=[0,2,3,4,5,6,7]
prior=np.array([0.3,0.1,2,2,2,2,3,2])[keep]
def F(G): f,_=fisher(base,G,vgrid(base,G,IPH),IPH,scale=15.); return f[np.ix_(keep,keep)]
out={}
for name,Fs in [('925 only',F(925)),('185 only',F(185)),('925 + 185',F(925)+F(185))]:
    C=posterior_cov(Fs,prior); d=np.sqrt(np.diag(C)); out[name]=(C[2,3]/(d[2]*d[3]),d[2],d[3])
pd.DataFrame(out,index=['corr(I02,Rs)','sd log I02','sd log Rs']).T.round(3).to_csv('../results/ident_I02_Rs_dual.csv')
print(pd.read_csv('../results/ident_I02_Rs_dual.csv').to_string())
