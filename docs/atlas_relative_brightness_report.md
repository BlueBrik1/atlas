# ATLAS relative brightness: controlled ablation (Nov 2026)

## Setup

- Gaia `phot_g_mean_mag<=7`; 22-degree FOV, 1,000 fixed reference sky fields, 22 brightest stars per patch.
- Two test sets: 100 original-seed (`3192026`) images, and independent 120-image (`20261102`) holdout. All variants receive exactly the same observations.
- Random planar rotation; moderate 12% missing/1 false/0.001 position noise/0.08 mag noise; severe 27%/2/0.0025/0.18 mag; extreme 40%/3/0.0045/0.32 mag. False magnitudes sampled uniformly 3.5 to 7.2.
- The original 40-candidate budget and identical 2-point RANSAC geometric verification are preserved; approximate original-style and separate 3D-fit pointing at 0.1 degrees.
- Baseline: 6 anchor hypotheses, with crossing+two signed coordinate channels and length prior; 10 anchor IDs plus 30 from 32-bin pair-distance histogram.
- Per-triangle brightness: normalized `(m_third - average(m_anchors))/5` sampled over third stars ordered by distance to anchor line. Scored at weight 0.1.
- Pairwise brightness: joint histogram of distance (32 bins) and absolute magnitude difference (8 bins), normalized, combined with baseline distance histogram, at weight 0.25. Additional memory ~1 MiB per 1,000 fields.
- Both features evaluated; multiple alternative weights explored in `atlas_relative_brightness_sweep.csv`, making results exploratory. All timing excludes offline precalculation and JIT compilation.

## Results on 120 independent images

| Method | Approx. pointing <0.1° | 3D-fit pointing <0.1° | Mean online time | Mean ranking |
|---|---:|---:|---:|---:|
| Baseline ATLAS | 67/120 (55.8%) | 80/120 (66.7%) | 39.35 ms | 5.27 ms |
| Add per-triangle brightness only (0.1) | 68/120 (56.7%) | 79/120 (65.8%) | 40.32 ms | 6.11 ms |
| Add pairwise-brightness histogram (0.25) | 72/120 (60.0%) | 89/120 (74.2%) | 38.64 ms | 5.64 ms |
| Both brightness methods | 73/120 (60.8%) | 89/120 (74.2%) | 39.26 ms | 6.31 ms |

## Historical 100 images (same observation seed as old work)

| Method | Approx. correct | 3D-fit correct |
|---|---:|---:|
| Baseline | 49/100 | 65/100 |
| Per-triangle brightness | 49/100 | 66/100 |
| Pairwise brightness | 51/100 | 67/100 |
| Both | 52/100 | 68/100 |

## Paired gains / caveats

- Holdout baseline -> pairwise brightness: approximate 6 newly correct, 1 newly wrong (two-sided exact McNemar p=0.125). 3D-fit: 10 newly correct, 1 newly wrong (p=0.0117, uncorrected for explored model weights).
- Baseline -> combined: approximate 7 gained, 1 lost (p=0.0703). 3D-fit: 10 gained, 1 lost.
- The little extra accuracy from per-triangle brightness is not worth its computation. Pairwise brightness is the recommended experimental addition.
- This is not a full-sky equivalent catalog comparison with Tetra3. Data simulated from catalog coordinates (not real camera frames); the 0.1-degree measure is a *boresight* error, not validated full three-axis attitude. Statistical certainty is limited, especially after a sweep of feature weights. The measured ~0.7 ms runtime speed difference is within performance variability and must not be interpreted as a speed improvement.
