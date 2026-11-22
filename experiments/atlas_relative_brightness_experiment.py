"""Controlled ATLAS relative-brightness ablation: research simulation, not flight software.
Identical Gaia star fields for all algorithms. Tune on original 100, evaluate on unseen 120.
Algorithms: old ATLAS (10 six-anchor +30 distance), relative-anchor brightness,
pairwise distance+brightness joint histogram, and both together.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1'
import sys,time,csv,types
from pathlib import Path
import numpy as np
import pandas as pd
from acceleration import njit
from paths import OUTPUT
import lightweight_anchor_tracker as atlas
import algebraic_65_reconstruction as recon

BASE=OUTPUT
# Precomputed Gaia observation geometry, identical reference database across variants.
# The original catalog sorting by magnitude is preserved.
def proj_mag(c):
    e,up=recon.basis(c);zz=recon.xyz@c
    ids=np.flatnonzero(zz>.8)
    x=(recon.xyz[ids]@e)/zz[ids];y=(recon.xyz[ids]@up)/zz[ids]
    ids=ids[(x*x+y*y)<(recon.rad*.94)**2]
    if len(ids)<9:return None
    ids=ids[np.argsort(recon.mag[ids])[:22]]
    pp=np.column_stack([(recon.xyz[ids]@e)/zz[ids],(recon.xyz[ids]@up)/zz[ids]])/(2*recon.rad)
    return pp,recon.mag[ids]

refs=recon.refs
refm=[]
for i,c in enumerate(recon.directions):
    x,m=proj_mag(c)
    assert np.max(abs(x-refs[i]))<1e-9
    refm.append(m)
print('REFERENCE',len(refs),'stars',sum(map(len,refs)),'total',flush=True)

# Numerical brightness in magnitude units; normalize 0..1 by 5mag, avoid domination by extreme outliers.
def brighter_features(p,m,anchors=6,reference_anchor='mean'):
    pairs=atlas.anchor_pairs(p,m,anchors)
    ret=[]
    for a,b in pairs:
        v=p[a]-p[b]; dist=np.linalg.norm(v)
        ey=v/dist;ex=np.array([ey[1],-ey[0]])
        centered=p-(p[a]+p[b])/2
        x=centered@ex;y=centered@ey
        x[a]=x[b]=0.
        others=np.asarray([j for j in range(len(p)) if j!=a and j!=b],int)
        others=others[np.lexsort((y[others],x[others],abs(x[others])))]
        center=(m[a]+m[b])/2 if reference_anchor=='mean' else m[a]
        relative=np.clip((m[others]-center)/5,-1,1)
        ret.append(atlas.quantize_rows(relative[:,None])[:,0])
    return np.array(ret,np.float32)

N=len(refs);A=6;Q=20
geo_bank,geo_lengths=atlas.build_database(refs,refm,anchors=6,extra_coordinates=True)
print('GEO BANK',geo_bank.shape,flush=True)
brightness_bank=np.full((N,A,Q),5.,dtype=np.float32)
for k,(p,m) in enumerate(zip(refs,refm)):
    b=brighter_features(p,m)
    brightness_bank[k,:len(b)]=b
DB=atlas.build_distance_database(refs)

@njit(cache=True)
def score_brightness(gb,lb,bb,qgeo,qlen,qbright,weight):
    N,A,Q,D=gb.shape
    scores=np.empty(N,np.float32)
    for k in range(N):
        best=1e30
        for a in range(A):
            for b in range(len(qgeo)):
                geom=0.;bright=0.
                for i in range(Q):
                    for j in range(D):geom+=abs(float(gb[k,a,i,j])-float(qgeo[b,i,j]))
                    bright+=min(1.,abs(float(bb[k,a,i])-float(qbright[b,i])))
                geom=geom/(Q*D)+.08*abs(float(lb[k,a])-float(qlen[b]))/.8
                s=geom+weight*(bright/Q)
                if s<best:best=s
        scores[k]=best
    return scores

# Normalized joint hist retains association between distance and relative magnitude difference.
# Brightness differences remain invariant to global photometric offset.
def joint_hist(p,m):
    i,j=np.triu_indices(len(p),1)
    if not len(i):return np.zeros(32*8,np.float32)
    dd=np.linalg.norm(p[i]-p[j],axis=1)
    dm=np.abs(m[i]-m[j]);dm=np.clip(dm,0,5.999)
    xd=np.floor(dd*(32/.98)).astype(int);xm=np.floor(dm*(8./6.)).astype(int)
    valid=(xd>=0)&(xd<32)
    h=np.bincount((xd[valid]*8+xm[valid]),minlength=256).astype(np.float32)
    return h/len(i)

JB=np.asarray([joint_hist(p,m) for p,m in zip(refs,refm)],np.float32)

# Order candidates, exactly preserving original hybrid's 10:30 allocation and interleaving.
def hybrid_ids(a_scores,d_scores,top=40,anchor_quota=10):
    aidx=np.argsort(a_scores)[:top]; didx=np.argsort(d_scores)[:top]
    aq=min(anchor_quota,top);dq=top-aq
    result=[];seen=set()
    for t in range(max(aq,dq)):
        if t<aq and int(aidx[t]) not in seen:
            result.append(int(aidx[t]));seen.add(int(aidx[t]))
        if t<dq and int(didx[t]) not in seen:
            result.append(int(didx[t]));seen.add(int(didx[t]))
    for j in np.r_[aidx,didx]:
        if len(result)>=top:break
        if int(j) not in seen:result.append(int(j));seen.add(int(j))
    return result[:top]

# Generate separately from magnitude noise to avoid modifying original geometrical RNG sequences.
def make_obs(t,rng,seed_name):
    condition,missing,nfake,centroid_noise,photo_noise=[
        ('moderate',.12,1,.001,.08),('severe',.27,2,.0025,.18),('extreme',.40,3,.0045,.32)][t%3]
    c=rng.normal(size=3);c/=np.linalg.norm(c)
    result=proj_mag(c)
    if result is None:return None
    q,catmag=result
    aa=rng.uniform(0,2*np.pi)
    rot=np.array([[np.cos(aa),-np.sin(aa)],[np.sin(aa),np.cos(aa)]])
    keep=rng.random(len(q))>=missing
    pp=q[keep]@rot.T+rng.normal(0,centroid_noise,(keep.sum(),2))
    p=np.vstack([pp,rng.uniform(-.43,.43,(nfake,2))])
    phot=np.random.default_rng(87921+10007*t+(0 if seed_name=='historical' else 991812))
    observed_mags=np.r_[catmag[keep]+phot.normal(0,photo_noise,keep.sum()),phot.uniform(3.5,7.2,nfake)]
    return condition,c,p,observed_mags

@njit(cache=True)
def median_runtime_pad(vals):
    return 0.

# Warm up JIT before measuring
qgp,ql=atlas.fingerprints(refs[0],refm[0],extra_coordinates=True)
qbp=brighter_features(refs[0],refm[0])
_ = atlas._score(geo_bank,geo_lengths,qgp,ql,.08)
_ = score_brightness(geo_bank,geo_lengths,brightness_bank,qgp,ql,qbp,.1)

weights_a=[0.0,.05,.10,.20,.40]
weights_d=[0.0,.10,.25,.50,1.0]
all_methods=['baseline']+[f'anchor_{w:.2f}' for w in weights_a[1:]]+[f'pair_{w:.2f}' for w in weights_d[1:]]+['both_0.10_0.25']

rows=[]
rankings=[]
for set_name,seed,count in [('historical',3192026,100),('holdout',20261102,120)]:
    rng=np.random.default_rng(seed)
    all_data=[]
    for t in range(count):
        o=make_obs(t,rng,set_name)
        if o is not None:all_data.append((t,*o))
    print('OBS',set_name,len(all_data),flush=True)
    for t,condition,true,p,m in all_data:
        # independent timing per shortlist (excludes all offline catalogue work)
        start=time.perf_counter()
        qgeo,qlen=atlas.fingerprints(p,m,extra_coordinates=True)
        qbr=brighter_features(p,m)
        anchor_scores={w:score_brightness(geo_bank,geo_lengths,brightness_bank,qgeo,qlen,qbr,w) for w in weights_a}
        dq=atlas.distance_histogram(p)
        pair_geom=np.abs(DB-dq).mean(axis=1)
        joint=joint_hist(p,m)
        # compare both fractional count distributions by L1 norm, with same range [0,2]
        joint_diff=np.abs(JB-joint).sum(axis=1)
        pair_scores={w:pair_geom+w*joint_diff/32 for w in weights_d}
        prepared_ms=(time.perf_counter()-start)*1000
        cfg={'baseline':(0.,0.)}
        for w in weights_a[1:]:cfg[f'anchor_{w:.2f}']=(w,0.)
        for w in weights_d[1:]:cfg[f'pair_{w:.2f}']=(0.,w)
        cfg['both_0.10_0.25']=(.10,.25)
        rankings_by_method={name:hybrid_ids(anchor_scores[aw],pair_scores[dw]) for name,(aw,dw) in cfg.items()}
        # Geometric matches shared cache so no method gets a different calculation.
        match_cache={}
        # Use old hybrid verifier: if first 5 weak/ambiguous, verify all remaining candidates.
        # Same 40 candidate cap as before. The code is identical for each method.
        def run_match(ids):
            def match(k):
                if k not in match_cache:match_cache[k]=recon.geometric(p,k)
                return match_cache[k]
            geom={k:match(k) for k in ids[:5]}
            ranked=sorted(geom,key=lambda k:(-geom[k],k))
            best,second=ranked[:2]
            fallback=geom[best]<max(4,round(.70*min(len(p),len(refs[best])))) or geom[best]-geom[second]<=1
            if fallback:
                for k in ids[5:]:geom[k]=match(k)
                best=max(geom,key=lambda k:(geom[k],-k))
            # point estimate using source's fixed 3D attitude, separately old estimate
            _,fit=recon.geometric(p,best,return_fit=True)
            old,refined,nmatched=recon.estimate_attitude(p,best,fit)
            err_old=recon.angle(old,true);err3d=recon.angle(refined,true)
            return best,int(err_old<.1),int(err3d<.1),err_old,err3d,int(fallback),nmatched
        for method,ids in rankings_by_method.items():
            start=time.perf_counter();best,correct,c3d,old_err,new_err,fb,nmatched=run_match(ids)
            # runtime of verifier is measured only without shared cache below; score time separately.
            # For this sweep, cache prevents duplicated geometry work but biases timer, so don't report this as online runtime.
            rows.append(dict(dataset=set_name,trial=t,condition=condition,method=method,
                correct=correct,refined_correct=c3d,err_deg=old_err,refined_err_deg=new_err,
                fallback=fb,matched=nmatched,best=best,
                near_top5=int(np.any([recon.angle(true,recon.directions[k])<12 for k in ids[:5]])),
                near_top40=int(np.any([recon.angle(true,recon.directions[k])<12 for k in ids]))))
        if (t+1)%20==0:
            df=pd.DataFrame(rows);g=df[df.dataset==set_name]
            print('DONE',set_name,t+1,' '.join(f'{nm}:{int(g[g.method==nm].correct.sum())}' for nm in ['baseline','anchor_0.10','pair_0.25','both_0.10_0.25']),flush=True)
    pd.DataFrame(rows).to_csv(BASE/'atlas_relative_brightness_sweep.csv',index=False)

# Evaluation of timing with no geometry cache and no score cache; on all 120 holdout cases.
# Rebuild observations deterministically, run baseline & selected variants individually.
selected=['baseline','anchor_0.10','pair_0.25','both_0.10_0.25']
timings=[]
rng=np.random.default_rng(20261102)
for t in range(120):
    obs=make_obs(t,rng,'holdout')
    if obs is None:continue
    condition,true,p,m=obs
    for method in selected:
        wa,wd={'baseline':(0,0),'anchor_0.10':(.10,0),'pair_0.25':(0,.25),'both_0.10_0.25':(.10,.25)}[method]
        start=time.perf_counter()
        qgeo,qlen=atlas.fingerprints(p,m,extra_coordinates=True)
        if wa==0:ascore=atlas._score(geo_bank,geo_lengths,qgeo,qlen,.08)
        else:ascore=score_brightness(geo_bank,geo_lengths,brightness_bank,qgeo,qlen,brighter_features(p,m),wa)
        dh=atlas.distance_histogram(p);dscore=np.abs(DB-dh).mean(axis=1)
        if wd!=0:dscore=dscore+(wd/32)*np.abs(JB-joint_hist(p,m)).sum(axis=1)
        ids=hybrid_ids(ascore,dscore)
        score_ms=(time.perf_counter()-start)*1000
        start2=time.perf_counter()
        geom={k:recon.geometric(p,k) for k in ids[:5]}
        order=sorted(geom,key=lambda k:(-geom[k],k));b,s=order[:2]
        fallback=geom[b]<max(4,round(.70*min(len(p),len(refs[b])))) or geom[b]-geom[s]<=1
        if fallback:
            for k in ids[5:]:geom[k]=recon.geometric(p,k)
            b=max(geom,key=lambda k:(geom[k],-k))
        _,fit=recon.geometric(p,b,return_fit=True)
        old,ref,_=recon.estimate_attitude(p,b,fit)
        total=(time.perf_counter()-start)*1000
        timings.append(dict(dataset='holdout',trial=t,condition=condition,method=method,
             ranking_ms=score_ms,verify_ms=(time.perf_counter()-start2)*1000,total_ms=total,
             correct=int(recon.angle(old,true)<.1),refined_correct=int(recon.angle(ref,true)<.1),fallback=int(fallback)))
    if (t+1)%30==0:print('TIMED',t+1,flush=True)
pd.DataFrame(timings).to_csv(BASE/'atlas_relative_brightness_timing.csv',index=False)

x=pd.DataFrame(rows);dt=pd.DataFrame(timings)
print('===ACCURACY===',flush=True)
print(x.groupby(['dataset','method'])[['correct','refined_correct','near_top5','near_top40','fallback']].agg(['sum','mean']).round(3).to_string(),flush=True)
print('===TIMING===',flush=True)
print(dt.groupby('method')[['correct','refined_correct','ranking_ms','verify_ms','total_ms']].mean().round(3).to_string(),flush=True)
