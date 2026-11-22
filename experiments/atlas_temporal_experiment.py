"""Controlled temporal ATLAS experiment. Gaia references and verifier reused unchanged.
No ground-truth centroid identities or attitude are used by registration/tracking.
Input: 5 simulated consecutive centroid images per test sequence.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1'
import sys,time,math
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from paths import OUTPUT
import atlas_six_anchor_fingerprint_test as six
import atlas_signed_crossing_fair_test as base
import atlas_side_features_experiment as side
import lightweight_anchor_tracker as atlas
import lightweight_anchor_brightness as bright
import algebraic_65_reconstruction as recon

N=base.N
W=np.array([1.,.3,.3,.5,.5,.5,.5],np.float32)
# Reference features, same as earlier experiments
GEO=six.bank

def register_to_ref(source,ref):
    """Small-motion robust rigid registration using iterative mutual nearest neighbors.
    No truth correspondence and no camera orientation are supplied to tracker.
    """
    tree=cKDTree(ref);pts=source.copy(); Rtot=np.eye(2);ttot=np.zeros(2)
    for _ in range(5):
        dst,ind=tree.query(pts,k=1)
        other=cKDTree(pts).query(ref,k=1)[1]
        src_ids=np.arange(len(pts))
        good=(dst<.034)&(other[ind]==src_ids)
        if good.sum()<3:good=dst<.027
        if good.sum()<3:break
        a=pts[good];b=ref[ind[good]]
        am=a.mean(axis=0);bm=b.mean(axis=0)
        u,_,vt=np.linalg.svd((a-am).T@(b-bm))
        d=np.linalg.det(u@vt)
        rot=u@np.diag([1.,d])@vt
        trans=bm-am@rot
        pts=pts@rot+trans
        Rtot=Rtot@rot;ttot=ttot@rot+trans
    return pts

def persist(frames,need=3):
    """Register frames to final frame; greedy multi-frame spatial association; median centroids/magnitudes."""
    last=frames[-1][0]
    aligned=[register_to_ref(p,last) if j!=len(frames)-1 else p.copy() for j,(p,m) in enumerate(frames)]
    tracks=[]
    for t,((_,magnitudes),coords) in enumerate(zip(frames,aligned)):
        active=set()
        for xyz,mag in zip(coords,magnitudes):
            best=-1;bd=.018
            for k,tr in enumerate(tracks):
                if k in active:continue
                d=float(np.linalg.norm(xyz-np.median(tr['xyz'],axis=0)))
                if d<bd:bd=d;best=k
            if best<0:
                tracks.append(dict(xyz=[xyz],mag=[mag],times=[t]));active.add(len(tracks)-1)
            else:
                tr=tracks[best];tr['xyz'].append(xyz);tr['mag'].append(mag);tr['times'].append(t);active.add(best)
    stable=[tr for tr in tracks if len(tr['times'])>=need]
    if len(stable)<6:stable=sorted(tracks,key=lambda x:len(x['times']),reverse=True)[:max(6,len(stable))]
    xy=np.array([np.median(t['xyz'],axis=0) for t in stable]);mm=np.array([np.median(t['mag']) for t in stable]);
    return xy,mm,dict(ntracks=len(tracks),nstable=len(stable),nlast=len(last))

def sequences(t,rng,frames=5,persistent_false=False):
    cond,miss,nfake,noise,photnoise=[('moderate',.12,1,.001,.08),('severe',.27,2,.0025,.18),('extreme',.40,3,.0045,.32)][t%3]
    c=rng.normal(size=3);c/=np.linalg.norm(c)
    out=base.proj_mag(c)
    if out is None:return None
    clean,mag=out
    # Fixed stars smoothly rotate and translate across frames; false stars are random,
    # with optional persistent moving interferer (a hard case for simple tracking).
    angle=rng.uniform(0,2*np.pi)
    omega=rng.uniform(-.014,.014)
    drift=np.zeros(2) # pure roll motion, consistent with fixed true boresight
    fakepoint=rng.uniform(-.35,.35,size=2)
    framesout=[];truthlabels=[]
    for j in range(frames):
        theta=angle+omega*j;cs,sn=np.cos(theta),np.sin(theta)
        rot=np.array([[cs,-sn],[sn,cs]])
        k=rng.random(len(clean))>=miss
        truep=clean[k]@rot.T+drift*j+rng.normal(0,noise,size=(int(k.sum()),2))
        truem=mag[k]+rng.normal(0,photnoise,int(k.sum()))
        falsep=rng.uniform(-.43,.43,size=(nfake,2))
        falsem=rng.uniform(3.5,7.2,size=nfake)
        if persistent_false and nfake>0:
            falsep[0]=fakepoint+np.array([.009*j,-.004*j])+rng.normal(0,.002,size=2)
        framesout.append((np.vstack((truep,falsep)),np.r_[truem,falsem]))
        truthlabels.append(np.r_[np.ones(len(truep),bool),np.zeros(nfake,bool)])
    return cond,c,framesout,truthlabels

# Exact same matcher and 10 anchor : 30 brightness-distance shortlist budget.
def shortlist(p,m,method):
    if method=='ATLAS':
        return bright.hybrid_shortlist_brightness(p,m,base.geo_bank,base.geo_lengths,base.distdb,base.brightdb,top=40,anchor_quota=10,brightness_weight=.25)
    q=side.features(p,m)[:,:7]
    pairscore=np.abs(base.distdb-atlas.distance_histogram(p)).mean(axis=1)+(.25/32)*np.abs(base.brightdb-bright.relative_brightness_histogram(p,m)).sum(axis=1)
    if method=='six_assignment':mode=2
    elif method=='single_best':mode=0
    else:raise ValueError(method)
    anchor_scores=six.scores(GEO,q,W,mode)
    return base.merge(np.argsort(anchor_scores),np.argsort(pairscore),top=40,anchor_quota=10)

def match(p,m,true,method):
    t0=time.perf_counter();ids=shortlist(p,m,method)
    rankms=(time.perf_counter()-t0)*1000
    result=base.verify(p,ids,true)
    totalms=(time.perf_counter()-t0)*1000
    return result,rankms,totalms,ids

if __name__=='__main__':
    # precompile
    q=side.features(recon.refs[0],base.refm[0])[:,:7]
    for mode in [0,2]:six.scores(GEO,q,W,mode)
    rng=np.random.default_rng(20261103)
    methods=['ATLAS','single_best','six_assignment']
    rows=[];tstart=time.time()
    for t in range(120):
        hard=(t>=90)
        res=sequences(t,rng,frames=5,persistent_false=hard)
        if res is None:continue
        cond,true,frames,labels=res
        lastp,lastm=frames[-1]
        t0=time.perf_counter();p5,m5,diag=persist(frames,need=3);tracker_ms=(time.perf_counter()-t0)*1000
        # diagnostic: fraction of last frame false stars included in filtered tracks (not used for matching)
        valid,truthlast=labels[-1].sum(),(~labels[-1]).sum()
        for source,p,m in [('single',lastp,lastm),('temporal5',p5,m5)]:
            for method in methods:
                result,rankms,totalms,ids=match(p,m,true,method)
                rows.append(dict(trial=t,condition=cond,hard_persistent_false=int(hard),source=source,method=method,
                    nstars=len(p),nlast=len(lastp),ntracked=diag['nstable'],tracker_ms=tracker_ms if source=='temporal5' else 0.,
                    rank_ms=rankms,solver_ms=totalms,full_ms=totalms+(tracker_ms if source=='temporal5' else 0.),
                    near40=int(any(recon.angle(true,recon.directions[int(k)])<12 for k in ids)),**result))
        if (t+1)%15==0:
            df=pd.DataFrame(rows);df.to_csv(OUTPUT / 'atlas_temporal_results.csv',index=False)
            print('DONE',t+1,'acc',df.groupby(['source','method']).correct.sum().to_dict(),'elapsed',round(time.time()-tstart,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUTPUT / 'atlas_temporal_results.csv',index=False)
    print('SUMMARY\n',df.groupby(['source','method']).agg(n=('trial','size'),correct=('correct','sum'),refined=('refined_correct','sum'),top40=('near40','sum'),ms=('full_ms','mean'),rankms=('rank_ms','mean'),tracker=('tracker_ms','mean'),nstars=('nstars','mean')).round(3).to_string(),flush=True)
    print('CORRUPTION\n',df.groupby(['condition','source','method']).correct.agg(['sum','count']).to_string(),flush=True)
    print('HARD\n',df.groupby(['hard_persistent_false','source','method']).correct.agg(['sum','count']).to_string(),flush=True)
