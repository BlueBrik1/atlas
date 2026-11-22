"""ATLAS optional relative-brightness augmentation (research prototype).

Adds a 32x8 joint histogram of pairwise angular/projection distance and
absolute apparent magnitude difference to ATLAS's distance-histogram shortlist.
Preserves six-anchor geometry, fixed 10:30 shortlist allocation, and 40
candidate budget. Precompute all reference histograms offline.

Note: p is the same normalized 2D image-plane coordinate convention as
lightweight_anchor_tracker.py; magnitude m has lower = brighter.
"""
import numpy as np
import lightweight_anchor_tracker as atlas

DISTANCE_BINS=32
BRIGHTNESS_BINS=8

def relative_brightness_histogram(p,m):
    """One normalized 256-vector. Completely order- and rotation-invariant
    under pure 2D rotations; invariant to global magnitude zero-point shifts.
    """
    p=np.asarray(p);m=np.asarray(m)
    if len(p)!=len(m):raise ValueError('One magnitude per observed star required')
    i,j=np.triu_indices(len(p),1)
    if not len(i):return np.zeros(DISTANCE_BINS*BRIGHTNESS_BINS,np.float32)
    d=np.linalg.norm(p[i]-p[j],axis=1)
    dm=np.clip(np.abs(m[i]-m[j]),0,5.999)
    idist=np.floor(d*(DISTANCE_BINS/.98)).astype(int)
    imag=np.floor(dm*(BRIGHTNESS_BINS/6.)).astype(int)
    ok=(idist>=0)&(idist<DISTANCE_BINS)
    hist=np.bincount(idist[ok]*BRIGHTNESS_BINS+imag[ok],minlength=DISTANCE_BINS*BRIGHTNESS_BINS)
    return (hist/len(i)).astype(np.float32)

def build_brightness_database(reference_points,reference_magnitudes):
    """Offline preprocessing, reference list order must match ATLAS geometry bank."""
    return np.asarray([relative_brightness_histogram(p,m) for p,m in zip(reference_points,reference_magnitudes)],np.float32)

def hybrid_shortlist_brightness(p,m,anchor_bank,anchor_lengths,distance_database,
                                brightness_database,top=40,anchor_quota=10,
                                brightness_weight=.25):
    """Identical 10 anchor + 30 distance ranking with optional brightness scoring.

    Score=mean absolute 32-bin distance difference +
          brightness_weight/32 * L1(256-bin joint histogram difference).
    0 weight reproduces original ATLAS shortlist.
    """
    anchor_ids,_=atlas.shortlist(p,m,anchor_bank,anchor_lengths,top=top,
                                 extra_coordinates=True)
    diff=np.abs(distance_database-atlas.distance_histogram(p)).mean(axis=1)
    if brightness_weight:
        extra=np.abs(brightness_database-relative_brightness_histogram(p,m)).sum(axis=1)
        diff=diff+(brightness_weight/32)*extra
    pair_ids=np.argsort(diff)
    aq=min(anchor_quota,top);bq=top-aq
    ids=[];seen=set()
    for j in range(max(aq,bq)):
        if j<aq and int(anchor_ids[j]) not in seen:
            ids.append(int(anchor_ids[j]));seen.add(int(anchor_ids[j]))
        if j<bq and int(pair_ids[j]) not in seen:
            ids.append(int(pair_ids[j]));seen.add(int(pair_ids[j]))
    for j in np.r_[anchor_ids,pair_ids]:
        if len(ids)>=top:break
        if int(j) not in seen:
            ids.append(int(j));seen.add(int(j))
    return np.array(ids[:top],dtype=int)
