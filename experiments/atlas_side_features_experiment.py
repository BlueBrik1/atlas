"""Controlled ATLAS signed-crossing ablation: means, radii, brightness gradients.
Uses 120 shared Gaia observations, 1000 reference fields, fixed 40-candidate geometric verifier.
Offline: catalog features. Online time includes fingerprint ranking and geometric verification.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1'
import sys,time
import numpy as np,pandas as pd
from acceleration import njit
from paths import OUTPUT
import atlas_signed_crossing_fair_test as base
import lightweight_anchor_tracker as atlas
import lightweight_anchor_brightness as bright
import algebraic_65_reconstruction as recon

# Each anchor has 1 scalar plus 2 counts, 4 average vector components,
# 2 mean radii, 2 slope estimates. All outputs scaled to comparable units.
GRID=25

def features_one(p,m,a,b):
    n=len(p)
    ey=p[a]-p[b];norm=np.linalg.norm(ey)
    if norm<1e-10:return np.zeros(11,dtype=np.float32)
    ey/=norm;ex=np.array([ey[1],-ey[0]])
    v=p-p[a];x=v@ex;y=v@ey
    x[a]=0.;x[b]=0.
    cuts=np.arange(1,GRID)/GRID-.5
    counts=np.searchsorted(cuts,x,side='right')-np.searchsorted(cuts,0.,side='right')
    F=counts.sum()/((n-1)*(GRID-1))
    keep=np.ones(n,bool);keep[a]=False;keep[b]=False
    left=keep&(x<(-1e-8));right=keep&(x>1e-8)
    out=[F]
    for mask in [left,right]:
        nn=mask.sum();out.append(nn/max(1,n-2))
    for mask in [left,right]:
        nn=mask.sum()
        if nn:
            xx=x[mask];yy=y[mask];rr=np.hypot(xx,yy)
            out.extend([xx.mean(),yy.mean(),rr.mean()])
        else:out.extend([0.,0.,0.])
    # brightness gradient: magnitude change / normalized radial distance, ridge-regularized;
    # relative magnitude removes zero-point sensitivity. clip for noisy small side samples.
    for mask in [left,right]:
        if mask.sum()>=3:
            rr=np.linalg.norm(v[mask],axis=1)
            mag=m[mask]
            dr=rr-rr.mean();dm=mag-mag.mean()
            slope=(dr@dm)/(dr@dr+.04)
            out.append(np.clip(slope/5.,-1.,1.))
        else:out.append(0.)
    return np.asarray(out,np.float32)

def features(p,m):
    return np.stack([features_one(p,m,a,b) for a,b in atlas.anchor_pairs(p,m,6)])

@njit(cache=True)
def feature_scores(bank,query,weights):
    n,an,dim=bank.shape;out=np.empty(n,np.float32)
    for k in range(n):
        best=1e10
        for a in range(an):
            for b in range(query.shape[0]):
                score=0.;norm=0.
                for j in range(dim):
                    w=weights[j]
                    if w:
                        score+=w*abs(float(bank[k,a,j]-query[b,j]));norm+=w
                score/=norm
                if score<best:best=score
        out[k]=best
    return out

METHODS={
  'ATLAS_brightness':None,
  'scalar_pair': np.array([1]+[0]*10,np.float32),
  'scalar_vectors':np.array([1,0,0, .5,.5,.5,.5, 0,0, 0,0],np.float32),
  'scalar_vectors_radius':np.array([1,0,0, .5,.5,.5,.5, .5,.5, 0,0],np.float32),
  'scalar_vectors_radius_gradient':np.array([1,0,0, .5,.5,.5,.5, .5,.5, .2,.2],np.float32),
  'scalar_vectors_radius_counts':np.array([1,.3,.3, .5,.5,.5,.5, .5,.5, 0,0],np.float32),
  'scalar_all':np.array([1,.3,.3, .5,.5,.5,.5, .5,.5,.2,.2],np.float32),
}

def main():
    t=time.time();bank=np.full((base.N,6,11),1000,np.float32)
    for k,(p,m) in enumerate(zip(recon.refs,base.refm)):
        f=features(p,m);bank[k,:len(f)]=f
    print('offline bank ready',bank.shape,'seconds',round(time.time()-t,2),flush=True)
    fp=features(recon.refs[0],base.refm[0]);feature_scores(bank,fp,METHODS['scalar_all'])
    rng=np.random.default_rng(20261102)
    rows=[];obslist=[]
    for i in range(120):
        ob=base.make_obs(i,rng)
        if ob is not None:obslist.append((i,*ob))
    # diagnostic: compare alternate feature sets in same pipeline and 40-candidate verifier
    for t,cond,true,p,m in obslist:
        q=features(p,m)
        pairscore=np.abs(base.distdb-atlas.distance_histogram(p)).mean(axis=1)+(.25/32)*np.abs(base.brightdb-bright.relative_brightness_histogram(p,m)).sum(axis=1)
        pairids=np.argsort(pairscore)
        for method,weights in METHODS.items():
            start=time.perf_counter()
            if method=='ATLAS_brightness':
                ids=bright.hybrid_shortlist_brightness(p,m,base.geo_bank,base.geo_lengths,base.distdb,base.brightdb,top=40,anchor_quota=10,brightness_weight=.25)
            else:
                ascores=feature_scores(bank,q,weights)
                ids=base.merge(np.argsort(ascores),pairids,top=40,anchor_quota=10)
            rankms=(time.perf_counter()-start)*1000
            out=base.verify(p,ids,true)
            totms=(time.perf_counter()-start)*1000
            rows.append(dict(trial=t,condition=cond,method=method,rank_ms=rankms,total_ms=totms,near40=int(any(recon.angle(true,recon.directions[int(k)])<12 for k in ids)),**out))
        if (t+1)%20==0:
            df=pd.DataFrame(rows);df.to_csv(OUTPUT / 'atlas_side_features_results.csv',index=False)
            print('DONE',t+1,df.groupby('method').correct.sum().to_dict(),'seconds',round(time.time()-t0,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUTPUT / 'atlas_side_features_results.csv',index=False)
    print('FULL',df.groupby('method')[['correct','refined_correct','near40','rank_ms','total_ms','fallback']].agg({'correct':'sum','refined_correct':'sum','near40':'sum','rank_ms':'mean','total_ms':'mean','fallback':'sum'}).round(3).to_string())
    print('SEVERITY',df.groupby(['condition','method']).correct.sum().unstack().to_string())
if __name__=='__main__':
    t0=time.time();main()
