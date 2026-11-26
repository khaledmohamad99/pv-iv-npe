import numpy as np, pandas as pd, itertools, json
from identifiability import fisher, vgrid, posterior_cov, PNAMES, LABELS
IPH=8.98
base=np.array([0.0, 0.0, np.log10(3e-11), np.log10(2.7e-6), np.log10(0.006), np.log10(300.*0.925), -6.0, 0.0])
states={'Healthy':{}, 'PID':{3:np.log10(1.5e-5),5:np.log10(10.)}, 'Interconnect':{4:np.log10(0.03)},
        'Crack':{6:np.log(0.15/0.85),7:np.log10(0.3)}}
prior_sd=np.array([0.3,0.1,2,2,2,2,3,2])
Gs=[925,740,555,370,185]; noise={'lab':1.0,'field':15.0}
# 1) rank deficiency with both tau and gsens
phi=base.copy(); F,_=fisher(phi,925,vgrid(phi,925,IPH),IPH)
ev=np.linalg.eigvalsh(F); print('eigs (both tau & gsens) min/max ratio: %.2e'%(ev[0]/ev[-1]))
# from here: gsens known (drop idx1) -> 7 params
keep=[0,2,3,4,5,6,7]
rows=[]; cors={}; Fstore={}
for sname,mods in states.items():
    phi=base.copy()
    for k,v in mods.items(): phi[k]=v
    for nname,sc in noise.items():
        for G in Gs:
            V=vgrid(phi,G,IPH); F,_=fisher(phi,G,V,IPH,scale=sc); F=F[np.ix_(keep,keep)]
            Fstore[(sname,nname,G)]=F
            C=posterior_cov(F,prior_sd[keep]); sd=np.sqrt(np.diag(C))
            for j,k in enumerate(keep):
                rows.append(dict(state=sname,noise=nname,G=G,param=PNAMES[k],label=LABELS[k],post_sd=sd[j],contraction=1-sd[j]/prior_sd[k]))
            if G in (925,185):
                d=np.sqrt(np.diag(C)); cors[(sname,nname,G)]=(C/np.outer(d,d)).tolist()
R=pd.DataFrame(rows); R.to_csv('../results/ident_single_curve.csv',index=False)
# 2) multi-irradiance subsets: EIG (nats) relative to all-five
eig=[]
for sname in states:
  for nname in noise:
    allF=sum(Fstore[(sname,nname,G)] for G in Gs); P=np.diag(prior_sd[keep]**2)
    full=0.5*np.linalg.slogdet(np.eye(len(keep))+P@allF)[1]
    for r in range(1,6):
      for sub in itertools.combinations(Gs,r):
        Fs=sum(Fstore[(sname,nname,G)] for G in sub)
        e=0.5*np.linalg.slogdet(np.eye(len(keep))+P@Fs)[1]
        eig.append(dict(state=sname,noise=nname,k=r,subset='+'.join(map(str,sub)),EIG=e,frac_of_all=e/full))
E=pd.DataFrame(eig); E.to_csv('../results/ident_eig_subsets.csv',index=False)
json.dump({'|'.join(map(str,k)):v for k,v in cors.items()},open('../results/ident_corr.json','w'))
pd.set_option('display.width',200)
print(R[R.noise=='lab'].pivot_table(index=['state','label'],columns='G',values='contraction').round(2).to_string())
print(R[R.noise=='field'].pivot_table(index=['state','label'],columns='G',values='contraction').round(2).to_string())
best=E.sort_values('EIG',ascending=False).groupby(['state','noise','k']).head(1)
print(best[best.k<=2].sort_values(['state','noise','k']).round(3).to_string())
