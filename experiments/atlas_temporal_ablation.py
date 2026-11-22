"""Temporal window-length and previously rejected feature ablation.
Reuses exact random 120 sequences from atlas_temporal_experiment.py.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1'
import sys,time
import numpy as np,pandas as pd
from paths import OUTPUT
import atlas_temporal_experiment as tmp
import atlas_side_features_experiment as side
import atlas_signed_crossing_fair_test as base
import lightweight_anchor_tracker as atlas
import lightweight_anchor_brightness as bright
import algebraic_65_reconstruction as recon

BANK=np.full((base.N,6,11),1000,np.float32)
for k,(p,m) in enumerate(zip(recon.refs,base.refm)):
    v=side.features(p,m);BANK[k,:len(v)]=v
WEIGHTS={
  'pairbright_only':None,
  'scalar_vectors':side.METHODS['scalar_vectors'],
  'vectors_radius':side.METHODS['scalar_vectors_radius'],
  'vectors_radius_gradient':side.METHODS['scalar_vectors_radius_gradient'],
}
q=side.features(recon.refs[0],base.refm[0]);side.feature_scores(BANK,q,WEIGHTS['scalar_vectors'])

def obtain(p,m,method):
    ds=np.abs(base.distdb-atlas.distance_histogram(p)).mean(axis=1)
    bs=np.abs(base.brightdb-bright.relative_brightness_histogram(p,m)).sum(axis=1)
    pair=np.argsort(ds+(.25/32)*bs)
    if method=='pairbright_only':return pair[:40]
    v=side.feature_scores(BANK,side.features(p,m),WEIGHTS[method]);return base.merge(np.argsort(v),pair,top=40,anchor_quota=10)

rng=np.random.default_rng(20261103);rows=[];start=time.time()
for t in range(120):
    seq=tmp.sequences(t,rng,frames=5,persistent_false=(t>=90))
    if seq is None:continue
    cond,true,frames,_=seq
    for window in [1,3,5]:
        beg=time.perf_counter()
        if window==1:p,m=frames[-1];track_ms=0
        else:p,m,_=tmp.persist(frames[-window:],need=(2 if window==3 else 3));track_ms=(time.perf_counter()-beg)*1000
        for method in WEIGHTS:
            t0=time.perf_counter();ids=obtain(p,m,method);rankms=(time.perf_counter()-t0)*1000
            out=base.verify(p,ids,true)
            fullms=(time.perf_counter()-t0)*1000+track_ms
            rows.append(dict(trial=t,condition=cond,persistent_false=int(t>=90),window=window,method=method,nstars=len(p),tracker_ms=track_ms,rank_ms=rankms,full_ms=fullms,**out))
    if (t+1)%20==0:
        pd.DataFrame(rows).to_csv(OUTPUT / 'atlas_temporal_ablation_results.csv',index=False)
        print('DONE',t+1,'sec',round(time.time()-start,1),flush=True)
df=pd.DataFrame(rows);df.to_csv(OUTPUT / 'atlas_temporal_ablation_results.csv',index=False)
print('SUMMARY\n',df.groupby(['window','method']).agg(correct=('correct','sum'),refined=('refined_correct','sum'),mean_ms=('full_ms','mean'),rankms=('rank_ms','mean'),tracker=('tracker_ms','mean')).round(3).to_string(),flush=True)
