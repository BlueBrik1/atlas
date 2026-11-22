"""Multi-anchor, rotation-invariant triangle-gridline fingerprint for star fields.

Inputs: (N,2) calibrated/image-plane points and (N,) apparent magnitudes.
Offline: index references with build_database().  Online: shortlist() returns reference ids.
The geometric verification and 3D attitude fit are *separate*, unchanged components.

Research prototype; not flight software. Unlike single-isolated-star anchoring, it
uses six distinct farthest-nearest-neighbor anchor pairs to tolerate changing stars.
"""
import numpy as np
from scipy.spatial import cKDTree
from acceleration import njit

GRIDS=(5,8,10,25)
CUTS=tuple(np.arange(1,k)/k-.5 for k in GRIDS)
QUANTILES=20
ANCHORS=6
ANCHOR_WEIGHT=0.08

def anchor_pairs(p,m,limit=ANCHORS):
    """Rank unique pairs by the maximum nearest-neighbor radius of their source star."""
    if len(p)<3:return []
    d,nearest=cKDTree(p).query(p,k=2,workers=1)
    seen=set();pairs=[]
    for i in np.argsort(-d[:,1]):
        a,b=int(i),int(nearest[i,1])
        key=tuple(sorted((a,b)))
        if key in seen:continue
        seen.add(key)
        if m[a]>m[b] or (m[a]==m[b] and a>b):a,b=b,a
        pairs.append((a,b))
        if len(pairs)>=limit:break
    return pairs

def crossing_matrix(p,a,b,extra_coordinates=False):
    """Rows follow rank of third star's absolute perpendicular distance to anchor line.
    Columns are the four crossing totals for 5x5, 8x8, 10x10, 25x25 grids.
    Bright anchor a is y-positive. Normalize before comparing grid sizes.
    """
    v=p[a]-p[b];r=np.linalg.norm(v)
    if r<1e-12:raise ValueError('Coincident anchors')
    ey=v/r;ex=np.array([ey[1],-ey[0]])
    centered=p-(p[a]+p[b])/2
    x=centered@ex;y=centered@ey
    x[a]=0.;x[b]=0. # exactly on the line; avoids floating point grid boundary flips
    thirds=np.array([j for j in range(len(p)) if j!=a and j!=b])
    thirds=thirds[np.lexsort((y[thirds],x[thirds],np.abs(x[thirds])))]
    values=[]
    for k,cut in zip(GRIDS,CUTS):
        xi=np.searchsorted(cut,x);yi=np.searchsorted(cut,y)
        cx=np.abs(xi[a]-xi[b])+np.abs(xi[a]-xi[thirds])+np.abs(xi[b]-xi[thirds])
        cy=np.abs(yi[a]-yi[b])+np.abs(yi[a]-yi[thirds])+np.abs(yi[b]-yi[thirds])
        values.append((cx+cy)/(4*(k-1)))
    out=np.stack(values,axis=1)
    if extra_coordinates:
        # Optional compact relative-position channels, tested separately.
        out=np.column_stack((out,np.clip(x[thirds]*1.5,-1,1),np.clip(y[thirds]*1.5,-1,1)))
    return out

def quantize_rows(values,Q=QUANTILES):
    if len(values)==1:return np.repeat(values.astype(np.float32),Q,axis=0)
    return np.column_stack([np.interp(np.linspace(0,1,Q),np.linspace(0,1,len(values)),values[:,j]) for j in range(values.shape[1])]).astype(np.float32)

def fingerprints(p,m,limit=ANCHORS,extra_coordinates=False):
    pairs=anchor_pairs(p,m,limit)
    fps=np.array([quantize_rows(crossing_matrix(p,a,b,extra_coordinates)) for a,b in pairs],np.float32)
    lengths=np.array([np.linalg.norm(p[a]-p[b]) for a,b in pairs],np.float32)
    return fps,lengths

def build_database(reference_points,reference_magnitudes,anchors=ANCHORS,extra_coordinates=False):
    """Only performed offline; precompute onboard database before deployment."""
    d=6 if extra_coordinates else 4
    bank=np.empty((len(reference_points),anchors,QUANTILES,d),np.float32)
    lengths=np.empty((len(reference_points),anchors),np.float32)
    for idx,(p,m) in enumerate(zip(reference_points,reference_magnitudes)):
        fp,l=fingerprints(p,m,anchors,extra_coordinates)
        n=len(fp)
        bank[idx,:n]=fp;lengths[idx,:n]=l
        if n<anchors:
            bank[idx,n:]=1e3;lengths[idx,n:]=1e3
    return bank,lengths

@njit(cache=True)
def _score(bank,lengths,queries,qlengths,length_weight):
    N,A,Q,D=bank.shape
    out=np.empty(N,np.float32)
    for k in range(N):
        best=1e30
        for a in range(A):
            for b in range(len(queries)):
                s=0.
                for i in range(Q):
                    for j in range(D):s+=abs(float(bank[k,a,i,j])-float(queries[b,i,j]))
                s=s/(Q*D)+length_weight*abs(float(lengths[k,a])-float(qlengths[b]))/.8
                if s<best:best=s
        out[k]=best
    return out

def shortlist(points,magnitudes,bank,lengths,top=40,anchors=ANCHORS,
              length_weight=ANCHOR_WEIGHT,extra_coordinates=False):
    """No rotation search; returns closest reference field ids and their scores."""
    fp,l=fingerprints(points,magnitudes,anchors,extra_coordinates)
    scores=_score(bank,lengths,fp,l,length_weight)
    idx=np.argsort(scores)[:top]
    return idx,scores[idx]

def distance_histogram(p,nbins=32):
    """Rotation- and order-invariant 32-bin pairwise-distance histogram."""
    i,j=np.triu_indices(len(p),1)
    if len(i)==0:return np.zeros(nbins,np.float32)
    d=np.linalg.norm(p[i]-p[j],axis=1)
    h=np.histogram(d,bins=np.linspace(0,.98,nbins+1))[0]
    return (h/len(i)).astype(np.float32)

def build_distance_database(reference_points):
    return np.array([distance_histogram(p) for p in reference_points],np.float32)

def hybrid_shortlist(p,m,bank,lengths,distance_database,top=40,anchor_quota=10,
                     extra_coordinates=True):
    """Shortlist combining 10 anchor-field ids and 30 distance-histogram ids.
    Interleaves anchor and histogram candidates, putting both in first 5.
    Geometry verification follows afterward; this function only ranks candidates.
    """
    anchor_ids,_=shortlist(p,m,bank,lengths,top=top,extra_coordinates=extra_coordinates)
    pair_scores=np.abs(distance_database-distance_histogram(p)).mean(axis=1)
    pair_ids=np.argsort(pair_scores)
    a_quota=min(anchor_quota,top); b_quota=top-a_quota
    result=[];used=set()
    for j in range(max(a_quota,b_quota)):
        if j<a_quota:
            idx=int(anchor_ids[j])
            if idx not in used:result.append(idx);used.add(idx)
        if j<b_quota:
            idx=int(pair_ids[j])
            if idx not in used:result.append(idx);used.add(idx)
    for idx in np.r_[anchor_ids,pair_ids]:
        if len(result)>=top:break
        if int(idx) not in used:
            result.append(int(idx));used.add(int(idx))
    return np.asarray(result[:top],dtype=int)
