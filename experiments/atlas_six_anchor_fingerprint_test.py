"""Fair ATLAS six-anchor vs independent-best-anchor ablation (Gaia)."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1'
import time,sys
import numpy as np,pandas as pd
from acceleration import njit
from paths import OUTPUT
import atlas_side_features_experiment as side
import atlas_signed_crossing_fair_test as base
import lightweight_anchor_tracker as atlas
import lightweight_anchor_brightness as bright
import algebraic_65_reconstruction as recon
OUT=OUTPUT / 'atlas_six_anchor_fingerprint_results.csv'
N=base.N
# 7 coordinates: signed crossings, two normalized side counts, two (x,y) average vectors.
weights=np.array([1.,.3,.3,.5,.5,.5,.5],np.float32)
bank=np.empty((N,6,7),np.float32)
for k,(p,m) in enumerate(zip(recon.refs,base.refm)):
 f=side.features(p,m)[:,:7]
 if len(f)<6:raise RuntimeError('Not enough anchors at reference '+str(k))
 bank[k]=f

@njit(cache=True)
def scores(bank,query,w,mode):
 n,A,D=bank.shape
 out=np.empty(n,np.float32)
 for k in range(n):
  costs=np.empty((6,6),np.float32)
  for i in range(6):
   for j in range(6):
    v=0.
    for d in range(D):v+=w[d]*abs(float(bank[k,i,d])-float(query[j,d]))
    costs[i,j]=v
  if mode==0: # existing best single candidate pair, index-agnostic
   best=1e9
   for i in range(6):
    for j in range(6):
     if costs[i,j]<best:best=costs[i,j]
   out[k]=best
  elif mode==1: # naive F1 matches F1 ...
   v=0.
   for i in range(6):v+=costs[i,i]
   out[k]=v/6
  elif mode==2: # optimal one-to-one anchor assignment
   dp=np.empty(64,np.float32)
   for q in range(64):dp[q]=1e9
   dp[0]=0.
   for mask in range(64):
    i=0
    for z in range(6):i+=(mask>>z)&1
    if i>=6:continue
    for j in range(6):
     if not (mask>>j)&1:
      newmask=mask|(1<<j)
      candidate=dp[mask]+costs[i,j]
      if candidate<dp[newmask]:dp[newmask]=candidate
   out[k]=dp[63]/6
  elif mode==3: # mean of best matches for each anchor in both directions (many-to-one)
   v=0.
   for i in range(6):
    low=1e9
    for j in range(6):
     if costs[i,j]<low:low=costs[i,j]
    v+=low
   for j in range(6):
    low=1e9
    for i in range(6):
     if costs[i,j]<low:low=costs[i,j]
    v+=low
   out[k]=v/12
 return out

def shortlist(ids_anchor,ids_pair):
 return base.merge(ids_anchor,ids_pair,top=40,anchor_quota=10)

def run():
 # warm compilation
 q=side.features(recon.refs[0],base.refm[0])[:,:7]
 for mode in range(4):scores(bank,q,weights,mode)
 rng=np.random.default_rng(20261102)
 rows=[];startall=time.time()
 methods=('single_best','six_index','six_assignment','six_symmetric','ATLAS_brightness')
 for t in range(120):
  obs=base.make_obs(t,rng)
  if obs is None:continue
  label,true,p,m=obs
  for method in methods:
   t0=time.perf_counter()
   if method=='ATLAS_brightness':
    ids=bright.hybrid_shortlist_brightness(p,m,base.geo_bank,base.geo_lengths,base.distdb,base.brightdb,top=40,anchor_quota=10,brightness_weight=.25)
   else:
    q=side.features(p,m)[:,:7]
    dist=atlas.distance_histogram(p)
    bh=bright.relative_brightness_histogram(p,m)
    pairscore=np.abs(base.distdb-dist).mean(axis=1)+(.25/32)*np.abs(base.brightdb-bh).sum(axis=1)
    ids_pair=np.argsort(pairscore)
    mode={'single_best':0,'six_index':1,'six_assignment':2,'six_symmetric':3}[method]
    v=scores(bank,q,weights,mode)
    ids=shortlist(np.argsort(v),ids_pair)
   rank_ms=(time.perf_counter()-t0)*1000
   out=base.verify(p,ids,true)
   total_ms=(time.perf_counter()-t0)*1000
   rows.append(dict(trial=t,condition=label,method=method,rank_ms=rank_ms,total_ms=total_ms,
                    near_top5=int(any(recon.angle(true,recon.directions[int(k)])<12 for k in ids[:5])),
                    near_top40=int(any(recon.angle(true,recon.directions[int(k)])<12 for k in ids)),**out))
  if (t+1)%10==0:
   df=pd.DataFrame(rows);df.to_csv(OUT,index=False)
   print('DONE',t+1,'correct',df.groupby('method').correct.sum().to_dict(),'elapsed',round(time.time()-startall,1),flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT,index=False)
 print('SUMMARY\n',df.groupby('method').agg(n=('trial','size'), correct=('correct','sum'),refined=('refined_correct','sum'),top5=('near_top5','sum'),top40=('near_top40','sum'),rank_ms=('rank_ms','mean'),total_ms=('total_ms','mean'),fallback=('fallback','sum')).round(3).to_string())
 print('SEVERITY\n',df.groupby(['condition','method']).correct.sum().unstack().to_string())
if __name__=='__main__':run()
