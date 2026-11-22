"""Controlled heldout comparison: brightest-anchor signed crossing scalar vs ATLAS.
All algorithms use same Gaia references, observations, geometric verifier, 40 candidate cap.
None uses truth labels in matching. The 3D fit is reported separately.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1'
import time,sys
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from acceleration import njit
from paths import OUTPUT
import lightweight_anchor_tracker as atlas
import lightweight_anchor_brightness as bright
import types
import algebraic_65_reconstruction as recon

N=len(recon.refs)

def proj_mag(c):
 e,up=recon.basis(c);zz=recon.xyz@c
 ids=np.flatnonzero(zz>.8); x=(recon.xyz[ids]@e)/zz[ids];y=(recon.xyz[ids]@up)/zz[ids]
 ids=ids[(x*x+y*y)<(recon.rad*.94)**2]
 if len(ids)<9:return None
 ids=ids[np.argsort(recon.mag[ids])[:22]]
 p=np.c_[(recon.xyz[ids]@e)/zz[ids],(recon.xyz[ids]@up)/zz[ids]]/(2*recon.rad)
 return p,recon.mag[ids]
refm=[proj_mag(c)[1] for c in recon.directions]
print('references',N,flush=True)
geo_bank,geo_lengths=atlas.build_database(recon.refs,refm,anchors=6,extra_coordinates=True)
distdb=atlas.build_distance_database(recon.refs)
brightdb=bright.build_brightness_database(recon.refs,refm)

# User's F = total signed number of crossed vertical gridlines, with B the brighter anchor.
# x is measured AFTER anchor rotation, with B at x=0 (the anchor line).
# For mathematical exactness, count crossing of fixed grid lines x in (-.5,.5).
# Need g(x)-g(0); count gridlines in (0,x] for positive x or [x,0) for negative x.
GRID=(5,8,10,25)

def signed_one(p,a,b,k):
 v=p[a]-p[b];norm=np.linalg.norm(v)
 if norm<1e-12:return 0.
 ey=v/norm;ex=np.array([ey[1],-ey[0]])
 x=(p-p[a])@ex; x[a]=0.;x[b]=0.
 cuts=np.arange(1,k)/k-.5
 # No zero line in these grids except for even k. For even grid k, zero is a line.
 # Subtract g(0) so anchor crosses nothing.
 signed=np.searchsorted(cuts,x,side='right')-np.searchsorted(cuts,0.,side='right')
 return float(signed.sum()) / (max(1,len(p)-1)*(k-1))

def scalar_fps(p,m,ks=(25,),maxanchors=6):
 pairs=atlas.anchor_pairs(p,m,maxanchors)
 return np.asarray([[signed_one(p,a,b,k) for k in ks] for a,b in pairs],np.float32)

# Main test is exactly the user's scalar formula (25x25 grid), also test 4-grid scalar variant.
sc25=np.full((N,6,1),100,np.float32)
sc4=np.full((N,6,4),100,np.float32)
for k,(p,m) in enumerate(zip(recon.refs,refm)):
 v=scalar_fps(p,m);w=scalar_fps(p,m,GRID)
 sc25[k,:len(v)]=v;sc4[k,:len(w)]=w

@njit(cache=True)
def scalar_scores(bank,query):
 N,A,D=bank.shape;out=np.empty(N,np.float32)
 for k in range(N):
  best=1e9
  for a in range(A):
   for b in range(len(query)):
    z=0.
    for j in range(D):z+=abs(float(bank[k,a,j])-float(query[b,j]))
    if z/D<best:best=z/D
  out[k]=best
 return out

# Same shortlist ordering as ATLAS hybrid: 10 anchor candidates + 30 pairwise candidates.
def merge(aids,pids,top=40,anchor_quota=10):
 ids=[];seen=set();bquota=top-anchor_quota
 for j in range(max(anchor_quota,bquota)):
  if j<anchor_quota:
   x=int(aids[j]);
   if x not in seen: ids.append(x);seen.add(x)
  if j<bquota:
   x=int(pids[j]);
   if x not in seen: ids.append(x);seen.add(x)
 for idx in np.r_[aids,pids]:
  if len(ids)>=top:break
  x=int(idx)
  if x not in seen:ids.append(x);seen.add(x)
 return np.array(ids[:top],int)

def get_scores(p,m):
 fp,ll=atlas.fingerprints(p,m,extra_coordinates=True)
 ascore=atlas._score(geo_bank,geo_lengths,fp,ll,.08)
 ds=np.abs(distdb-atlas.distance_histogram(p)).mean(axis=1)
 bs=np.abs(brightdb-bright.relative_brightness_histogram(p,m)).sum(axis=1)
 pair_score=ds+(.25/32)*bs
 s1=scalar_scores(sc25,scalar_fps(p,m));s4=scalar_scores(sc4,scalar_fps(p,m,GRID))
 return ascore,pair_score,s1,s4

def make_obs(t,rng):
 label,miss,nfake,noise,photnoise=[('moderate',.12,1,.001,.08),('severe',.27,2,.0025,.18),('extreme',.40,3,.0045,.32)][t%3]
 c=rng.normal(size=3);c/=np.linalg.norm(c)
 q=proj_mag(c)
 if q is None:return None
 p,m=q;aa=rng.uniform(0,2*np.pi);cs,sn=np.cos(aa),np.sin(aa)
 R=np.array([[cs,-sn],[sn,cs]])
 keep=rng.random(len(p))>=miss
 ob=p[keep]@R.T+rng.normal(0,noise,(keep.sum(),2))
 ob=np.vstack([ob,rng.uniform(-.43,.43,(nfake,2))])
 photo=np.random.default_rng(87921+10007*t+991812)
 om=np.r_[m[keep]+photo.normal(0,photnoise,keep.sum()),photo.uniform(3.5,7.2,nfake)]
 return label,c,ob,om

def verify(p,ids,true):
 # Exactly same geometry match threshold, top-five confidence and fallback logic for every method.
 geom={int(k):recon.geometric(p,int(k)) for k in ids[:5]}
 ordered=sorted(geom,key=lambda k:(-geom[k],k));best,second=ordered[:2]
 fallback=geom[best]<max(4,round(.70*min(len(p),len(recon.refs[best])))) or geom[best]-geom[second]<=1
 if fallback:
  for k in ids[5:]:geom[int(k)]=recon.geometric(p,int(k))
  best=max(geom,key=lambda k:(geom[k],-k))
 _,fit=recon.geometric(p,int(best),return_fit=True)
 old,refined,nmatched=recon.estimate_attitude(p,int(best),fit)
 return dict(correct=int(recon.angle(old,true)<.1),refined_correct=int(recon.angle(refined,true)<.1),error_deg=recon.angle(old,true),refined_error_deg=recon.angle(refined,true),fallback=int(fallback),inliers=nmatched)

if __name__=='__main__':
 q=scalar_fps(recon.refs[0],refm[0]);_ = scalar_scores(sc25,q)
 q=scalar_fps(recon.refs[0],refm[0],GRID);_ =scalar_scores(sc4,q)
 fp,ll=atlas.fingerprints(recon.refs[0],refm[0],extra_coordinates=True)
 _=atlas._score(geo_bank,geo_lengths,fp,ll,.08)
 # The scalar fingerprint should not change under arbitrary rotations of identical stars.
 testp=recon.refs[20].copy();testm=refm[20].copy();base=scalar_fps(testp,testm,GRID)
 R=np.array([[np.cos(1.234),-np.sin(1.234)],[np.sin(1.234),np.cos(1.234)]])
 rotfp=scalar_fps(testp@R.T,testm,GRID)
 print('rotation scalar absolute max error',float(np.max(np.abs(base-rotfp))),flush=True)
 rng=np.random.default_rng(20261102)
 rows=[]
 for t in range(120):
  obs=make_obs(t,rng)
  if obs is None:continue
  cond,true,p,m=obs
  # algorithms run independently, with full online time measured for each
  for method in ('ATLAS_brightness','scalar25_only','scalar4_only','scalar25_plus_pairbright','scalar4_plus_pairbright'):
   start=time.perf_counter()
   if method=='ATLAS_brightness':
    ids=bright.hybrid_shortlist_brightness(p,m,geo_bank,geo_lengths,distdb,brightdb,top=40,anchor_quota=10,brightness_weight=.25)
   else:
    scores=scalar_scores(sc25,scalar_fps(p,m)) if 'scalar25' in method else scalar_scores(sc4,scalar_fps(p,m,GRID))
    aidx=np.argsort(scores)
    if method.endswith('only'): ids=aidx[:40]
    else:
     pairscore=np.abs(distdb-atlas.distance_histogram(p)).mean(axis=1)+(.25/32)*np.abs(brightdb-bright.relative_brightness_histogram(p,m)).sum(axis=1)
     ids=merge(aidx,np.argsort(pairscore))
   ranking_ms=(time.perf_counter()-start)*1e3
   out=verify(p,ids,true)
   total_ms=(time.perf_counter()-start)*1e3
   near5=int(any(recon.angle(true,recon.directions[int(k)])<12 for k in ids[:5]))
   near40=int(any(recon.angle(true,recon.directions[int(k)])<12 for k in ids))
   rows.append(dict(trial=t,condition=cond,method=method,nstars=len(p),near_top5=near5,near_top40=near40,ranking_ms=ranking_ms,total_ms=total_ms,**out))
  if (t+1)%20==0:
   df=pd.DataFrame(rows);print('DONE',t+1,df.groupby('method').correct.sum().to_dict(),flush=True)
   df.to_csv(OUTPUT / 'atlas_signed_crossing_fair_results.csv',index=False)
 df=pd.DataFrame(rows)
 df.to_csv(OUTPUT / 'atlas_signed_crossing_fair_results.csv',index=False)
 print('RESULTS',flush=True)
 print(df.groupby('method')[['correct','refined_correct','near_top5','near_top40','ranking_ms','total_ms','fallback']].agg({'correct':'sum','refined_correct':'sum','near_top5':'sum','near_top40':'sum','ranking_ms':'mean','total_ms':'mean','fallback':'sum'}).round(3).to_string(),flush=True)
 print('SEVERITY',flush=True)
 print(df.groupby(['condition','method']).correct.sum().unstack().to_string(),flush=True)
