"""Evidence-led reconstruction of lost 65/100 Gaia star tracker experiment.

Not the lost original source. Builds 1000 Gaia reference patches, compares exactly
pair-enumerated and algebraic crossing fingerprints, runs a 5-candidate shortlist
with full 1000-field geometric fallback, and compares ESA Tetra3. All benchmark
observations have random sky directions unseen when the reference fields were
built. Also reports both original-style approximate pointing offset and a new,
physically grounded 3D Wahba/Kabsch attitude refinement.
"""
import os,sys,math,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from paths import CATALOG


S=pd.read_csv(CATALOG).dropna(subset=['ra','dec','phot_g_mean_mag'])
S=S[S.phot_g_mean_mag<=7].sort_values('phot_g_mean_mag')
ra=np.deg2rad(S.ra.to_numpy());dec=np.deg2rad(S.dec.to_numpy());mag=S.phot_g_mean_mag.to_numpy()
xyz=np.column_stack([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])
rad=np.tan(np.deg2rad(11));rng=np.random.default_rng(3192026)
N0=1000;ii=np.arange(N0);z=1-2*(ii+.5)/N0;phi=np.pi*(3-np.sqrt(5))*ii
initial_dirs=np.column_stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])

def basis(c):
 e=np.cross([0.,0.,1.],c)
 if np.linalg.norm(e)<1e-4:e=np.cross([0.,1.,0.],c)
 e/=np.linalg.norm(e);return e,np.cross(c,e)

def proj(c):
 e,up=basis(c);zz=xyz@c
 ids=np.where(zz>.8)[0]
 x=(xyz[ids]@e)/zz[ids];y=(xyz[ids]@up)/zz[ids]
 ids=ids[(x*x+y*y)<(rad*.94)**2]
 if len(ids)<9:return None
 ids=ids[np.argsort(mag[ids])[:22]]
 return np.column_stack([(xyz[ids]@e)/zz[ids],(xyz[ids]@up)/zz[ids]])/(2*rad)

refs=[];directions=[]
for c in initial_dirs:
 q=proj(c)
 if q is not None:refs.append(q);directions.append(c)
directions=np.asarray(directions);N=len(refs)
K=(5,8,10,25);cuts=[np.arange(1,k)/k-.5 for k in K]
Rots=np.array([[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]] for a in np.deg2rad(np.arange(0,360,15))])

def triangle_features(p, algebraic):
 n=len(p);out=[]
 for k,lines in zip(K,cuts):
  if algebraic:
   x=np.searchsorted(np.sort(p[:,0]),lines);y=np.searchsorted(np.sort(p[:,1]),lines)
   crossing=(x*(n-x)+y*(n-y)).sum()
  else:
   cell=np.floor(np.clip(p+.5,0,1-1e-9)*k).astype(int)
   a,b=np.triu_indices(n,1)
   crossing=np.abs(cell[a]-cell[b]).sum()
  # Multiply by (n-2) for actual triangle crossings, normalize by
  # (n-2)*comb(n,2): cancels algebraically, preserving exact ranking.
  out.append(crossing/max(1,n*(n-1)/2)/k)
 # Original pipeline's 64 8x8 star density features; NOT triangle information.
 bins=np.floor(np.clip(p+.5,0,1-1e-9)*8).astype(int)
 occupancy=np.bincount(bins[:,0]+8*bins[:,1],minlength=64)/max(1,n)
 return np.r_[out,occupancy]

Fref=np.stack([triangle_features(p,True) for p in refs])
PAIR=[]
for q in refs:
 i,j=np.triu_indices(len(q),1);PAIR.append((i,j,np.linalg.norm(q[i]-q[j],axis=1),cKDTree(q)))

def geometric(p,k, return_fit=False):
 """Preexisting 2-point RANSAC inlier matcher; compare all catalog fields on fallback."""
 q=refs[k];i,j=np.triu_indices(len(p),1)
 if len(i)==0:return (0,None) if return_fit else 0
 pl=np.linalg.norm(p[i]-p[j],axis=1)
 a,b,ql,tree=PAIR[k];dif=np.abs(pl[:,None]-ql[None,:]);ix,iy=np.where(dif<.018)
 if not len(ix):return (0,None) if return_fit else 0
 order=np.argsort(dif[ix,iy])[:12];best=0;bestfit=None
 for tt in order:
  u,v=i[ix[tt]],j[ix[tt]];w,z=a[iy[tt]],b[iy[tt]]
  for x,y in ((w,z),(z,w)):
   A=p[v]-p[u];B=q[y]-q[x]
   theta=np.arctan2(B[1],B[0])-np.arctan2(A[1],A[0]);c,s=np.cos(theta),np.sin(theta)
   R=np.array([[c,-s],[s,c]])
   mapped=(p-p[u])@R.T+q[x]
   d,ind=tree.query(mapped,workers=1)
   score=int((d<.018).sum())
   if score>best:best=score;bestfit=(ind,d,R,u,x)
 if return_fit:return best,bestfit
 return best

def estimate_attitude(p,k,fit):
 """Wahba fit from matched camera rays and catalog rays; score genuine 3D boresight."""
 if fit is None:return directions[k].copy(),directions[k].copy(),0
 ind,d,R,u,x=fit;good=np.flatnonzero(d<.018)
 center=directions[k];e,up=basis(center)
 trans=refs[k][x]-p[u]@R.T
 offset=center+(trans[0]*e+trans[1]*up)*2*rad;offset/=np.linalg.norm(offset)
 # Refine 3D orientation, using only geometrically aligned stars.
 if len(good)<3:return offset,offset,len(good)
 q=refs[k][ind[good]]
 world=center+2*rad*q[:,0,None]*e+2*rad*q[:,1,None]*up
 world/=np.linalg.norm(world,axis=1,keepdims=True)
 camera=np.column_stack([2*rad*p[good],np.ones(len(good))]);camera/=np.linalg.norm(camera,axis=1,keepdims=True)
 U,_,VT=np.linalg.svd(camera.T@world)
 diag=np.diag([1,1,np.linalg.det(U@VT)]);rot=U@diag@VT
 attitude=rot[2,:]
 return offset,attitude,len(good)

def shortlisted(p, algebraic):
 scores=np.full(N,np.inf)
 for rot in Rots:
  f=triangle_features(p@rot.T,algebraic)
  delta=np.abs(Fref-f)
  val=3*delta[:,:4].mean(axis=1)+1.5*delta[:,4:].sum(axis=1)
  scores=np.minimum(scores,val)
 return np.argsort(scores)[:5]

def get_obs(t):
 label,miss,nfake,noise=[('moderate',.12,1,.001),('severe',.27,2,.0025),('extreme',.4,3,.0045)][t%3]
 c=rng.normal(size=3);c/=np.linalg.norm(c);q=proj(c)
 if q is None:return None
 a=rng.uniform(0,2*np.pi);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 keep=rng.random(len(q))>=miss
 p=q[keep]@R.T+rng.normal(0,noise,(keep.sum(),2));p=np.vstack([p,rng.uniform(-.43,.43,(nfake,2))])
 return label,c,p

def angle(a,b):return float(np.degrees(np.arccos(np.clip(a@b,-1,1))))

