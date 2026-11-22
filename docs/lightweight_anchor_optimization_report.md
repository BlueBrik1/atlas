# Lightweight anchored-triangle fingerprint: robustness optimization

## Main finding

A single farthest-nearest-neighbor anchor has low robustness under missing and false detections. Retaining up to six distinct candidate nearest-neighbor pairs and selecting the closest matching precomputed catalog fingerprint improves accuracy substantially without an appreciable change in the full geometric verifier's online runtime.

Additional improvements were obtained by using a compact, independent 32-bin pairwise-distance histogram to allocate 30 of the 40 verification candidates. The other 10 are provided by the multi-anchor triangle crossing matrix. This does **not** demonstrate that the anchored method outperforms the histogram alone: on the independent sample, the difference was only 2/90 correct.

## Exact changes

1. Offline, for each catalog patch, compute six unique candidate anchor pairs from stars ranked by their distance to their nearest neighbor, orienting the brighter star above the other.
2. For each anchor, sort remaining stars by absolute perpendicular distance to the anchor line and construct a 20-by-4 interpolated array of triangle-gridline crossings at 5, 8, 10, and 25 cells per side.
3. Optional: add each third star's signed perpendicular offset and vertical projection as two extra channels (20-by-6). This adds simple local orientation data but is no longer strictly crossings-only.
4. Compare the observation's six anchor hypotheses to the catalog's six stored candidates, using absolute differences and a small anchor-length penalty (0.08). NumPy/Numba implementation has no camera rotation search.
5. Offline compute an additional normalized 32-bin histogram of all pairwise star distances for each reference.
6. Online, select 10 distinct reference ids from the anchor matcher and 30 from the pairwise histogram; interleave these candidates so both are represented among the top five.
7. Use the **unchanged** geometric verifier, fallback rule, and at-most-40 candidate verification budget from the earlier anchored-triangle experiments.

## Dataset

- Real Gaia star coordinates and brightness values (magnitude <= 7), up to 22 stars per image, 22-degree camera field of view.
- 100 historically recovered image observations and 90 separately generated unseen-direction observations (seed 20261102).
- Each group cycles through moderate (12% loss, 1 false, 0.001 image-coordinate noise), severe (27%, 2 false, 0.0025), and extreme (40%, 3 false, 0.0045).
- Same 1000 reference sky fields for all evaluated custom methods; Tetra3 has its own database and solver, so catalogs are not equivalent.
- Online times exclude offline database creation and Numba's first-time JIT compilation.

## Accuracy and speed: 100 historical observations

| Method | Legacy-style pointing <=0.1 degrees | New 3D-fitted boresight <=0.1 degrees | Runtime, ms/image |
|---|---:|---:|---:|
| Single-anchor matrix, rank-interpolated | 16/100 | 23/100 | 39.8 |
| Six-anchor matrix, signed channels and length prior | 40/100 | 48/100 | 38.7 |
| Pairwise distance histogram alone | 49/100 | 63/100 | 29.5 |
| Hybrid: 10 anchor + 30 histogram shortlist | **50/100** | **65/100** | **34.5** |
| ESA Tetra3 (historical benchmark result) | 57/100 | N/A | 149.2 historical ms |

## Accuracy and speed: separate 90-observation validation

| Method | Legacy-style boresight <=0.1 degrees | New 3D-fitted boresight <=0.1 degrees | Runtime, ms/image |
|---|---:|---:|---:|
| Single-anchor matrix | 12/90 (13.3%) | 16/90 (17.8%) | 39.7 |
| Six-anchor crossing matrix, length penalty | 27/90 (30.0%) | 31/90 (34.4%) | 40.9 |
| Six-anchor + two signed coordinate channels | 32/90 (35.6%) | 41/90 (45.6%) | 40.0 |
| Pairwise distance histogram alone | 47/90 (52.2%) | 60/90 (66.7%) | 31.2 |
| **Hybrid 10 anchor + 30 histogram** | **49/90 (54.4%)** | **62/90 (68.9%)** | **34.5** |
| ESA Tetra3, separately run on identical observations | 51/90 (56.7%) | N/A | 160.6 |

The new fitted-boresight column uses the existing Kabsch/Wahba refinement and must not be compared as though it were exactly the same estimation procedure as the legacy-style column. These are boresight errors, not fully evaluated three-axis attitude errors.

The hybrid is 4.65x faster on this machine than Tetra3 in the independent benchmark, but it did not beat Tetra3's legacy-style accuracy (49 vs 51, statistically indistinguishable in 90 samples). Its 3D boresight refinement reached 62/90 compared to Tetra3 51/90, but the paired McNemar exact p-value was approximately 0.061; not sufficient for a strong accuracy superiority claim at 5% significance. The hybrid's 3D refinement beat the pure pairwise histogram by 2 observations, also inconclusive.

## Invariance and resource test

- Six-anchor signed fingerprint: 120/120 arbitrary planar rotations and star-list permutations returned identical arrays to tolerance; maximum measured component discrepancy 0.
- Offline database: 3,032,000 bytes (~2.89 MiB) total for 1,000 reference fields, six 20x6 anchor fingerprints per field and the 32-bin pairwise histogram.
- All 90 independent hybrid predictions were rechecked using the standalone `lightweight_anchor_tracker.py` rather than the initial experiment implementation, with identical correctness outcomes.

## Limitations

This is a simulation. It is not a flight-ready star tracker. The camera model uses idealized simulated Gaia positions, synthetic false stars and positional noise. Missing-star perturbation can change which anchor pairs exist. The 1,000-field catalog is sparse and not equivalent to the Tetra3 catalog. No hardware power or memory benchmarking and no independent 3-axis attitude assessment were performed. Timing was measured on shared host hardware, not flight processors. In addition, the original source for the historical 65% algebraic baseline remains unavailable. Preserve this limitation when comparing to that experiment.

## Files and commands

- Core method: `lightweight_anchor_tracker.py` (imports numpy, scipy, numba)
- Standalone validation: `lightweight_anchor_hybrid_check.py`
- The 90-observation results: `lightweight_anchor_hybrid_verified.csv`
- 190-observation algorithm ablation: `anchor_fallback_fusion_results.csv`
- Tetra3 independent 90-observation results: `lightweight_anchor_tetra_holdout_results.csv`

Minimal use:

```python
import lightweight_anchor_tracker as star
bank, lens = star.build_database(reference_points, reference_magnitudes,
                                 anchors=6, extra_coordinates=True)
pair_bank = star.build_distance_database(reference_points)
ids = star.hybrid_shortlist(observed_points, observed_magnitudes,
                            bank, lens, pair_bank, top=40, anchor_quota=10,
                            extra_coordinates=True)
# Then pass ids into a separately calibrated geometric verification / attitude solver.
```
