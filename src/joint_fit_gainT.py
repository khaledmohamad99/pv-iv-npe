import numpy as np
from scipy.optimize import least_squares
from real_obs import build
from physics import iv_curve, KB, Q
from simulate_bank import GRID, GS
S,X=build(); EG=1.12
def curve(p,Gnom,g,dT):
    T=298.15+dT; vt=KB*T/Q
    s=(T/298.15)**3*np.exp(EG/(KB/Q)*(1/298.15-1/T)); s2=(T/298.15)**1.5*np.exp(EG/(2*KB/Q)*(1/298.15-1/T))
    th=dict(Iph_ref=p[0],I01=10**p[1]*s,I02=10**p[2]*s2,Rs=10**p[3],Rsh=10**p[4],fc=1/(1+np.exp(-p[5])),Rc=10**p[6])
    return iv_curve(GRID,np.array([Gnom*g]),th,vt=vt)[0]*1000/Gnom
def fit(i):
    def res(q):
        p=q[:7]; g=np.r_[1.0,q[7:11]]; dT=q[11:]; r=[]
        for j,G in enumerate(GS):
            o=X[i,j]; k=o>-1; r.append((curve(p,G,g[j],dT[j])[k]-o[k])/o[0])
        return np.concatenate(r)
    best=None
    for s in [[9,-10.5,-6,-2.2,1.5,-2,-0.5],[9,-10,-7,-2.0,2.0,-3,0]]:
        q0=s+[1.03,1.05,1.06,1.01]+[0]*5; lb=[7,-13,-10,-3,-1.5,-8,-3]+[0.8]*4+[-10]*5; ub=[10.5,-8,-3,-0.5,4,3,2]+[1.2]*4+[10]*5
        r=least_squares(res,q0,bounds=(lb,ub),x_scale='jac',max_nfev=800)
        if best is None or r.cost<best.cost: best=r
    return 100*np.sqrt(np.mean(best.fun**2)), best.x
if __name__ == "__main__":
  for i in [5,0,33,47]:
      b,x=fit(i); print(S.loc[i,'sample'],S.loc[i,'meas'],'gain+dT RMSE pct: %.3f dT='%b,np.round(x[11:],1),flush=True)
