# Temporal ATLAS benchmark (2026-11-03)

## Experimental setup
- 120 independent synthetic five-frame sequences derived from Gaia stars, each with an unseen sky direction.
- The same 1,000 precomputed reference fields, scoring routines and 40-field geometry-verification budget across methods.
- A separate single-frame control uses the fifth frame of the *same sequence*.
- Simulated 22-degree field of view; random initial roll, roll rate uniform in [-0.014, 0.014] radians/frame; camera boresight held constant. Thus the evaluation truth is geometrically consistent.
- Moderate/severe/extreme respectively remove 12/27/40% of stars per frame and add 1/2/3 false stars; centroid noise std .001/.0025/.0045 in normalized image coordinates; photometric noise std .08/.18/.32 magnitudes.
- Last 30 sequences also include a recurring moving false detection; the other false detections are independently resampled each frame.
- Temporal correspondence uses robust mutual-nearest-neighbor rigid registration, followed by geometric clustering. A detection track is kept if seen in >=3/5 frames or >=2/3 frames; median centroids and median apparent magnitudes are used. Neither sky truth nor detection identities are passed to matching.
- Matching is unchanged: 10 candidate slots from the anchor channel, 30 from the pairwise brightness-distance channel (or 40 from the pairwise channel for pairbright_only); at most 40 candidates are geometrically verified.
- 0.1-degree correctness refers to boresight pointing estimates, **not** full three-axis attitude error. Additional 3D fit reported separately.
- Online runtime starts after star centroid and brightness detections are available, excludes star extraction from raw images and offline catalog generation, but includes temporal registration, median fusion, ranking, and geometric verification.

## Five-frame versus single-frame (120 sequences)
| Method | Single-frame correct | 5-frame correct | Single-frame ms | 5-frame total ms | Correct with separate 3D fit (single / 5-frame) |
|---|---:|---:|---:|---:|---:|
| ATLAS triangle + pairwise brightness | 64 | 91 | 38.46 | 54.47 | 79 / 104 |
| Lightweight best-anchor + pairwise | 61 | 88 | 33.82 | 50.06 | 76 / 101 |
| Six-anchor assignment + pairwise | 61 | 90 | 35.14 | 51.39 | 78 / 103 |

Across 120 matched cases, five-frame ATLAS gained 33 identifications and lost six relative to single-frame ATLAS (two-sided exact McNemar p=1.43e-5). The all-six-anchor assignment gained 34 and lost five (p=2.43e-6). These comparisons are within a single simulation distribution and cannot establish a flight-realistic improvement.

## Three-frame / five-frame ablation (120 sequences)
| Method | 1 frame correct | 3 frames correct | 5 frames correct | 1 frame total ms | 3 frames total ms | 5 frames total ms |
|---|---:|---:|---:|---:|---:|---:|
| Pairwise distance + brightness alone | 62 | 88 | 90 | 32.26 | 34.57 | 48.59 |
| Signed crossings + average vectors + pairwise | 61 | 87 | 88 | 34.57 | 38.82 | 50.53 |
| Above + mean radius | 61 | 85 | 89 | 34.46 | 39.40 | 50.44 |
| Above + brightness gradient | 61 | 86 | 90 | 34.43 | 39.18 | 51.11 |

Three-frame pairwise-only scored 35/40 moderate, 36/40 severe, and 17/40 extreme; single-frame pairwise-only scored 33/40 moderate, 25/40 severe, and 4/40 extreme. Correctness gains with three-frame pairwise-only: 30 changed failures to successes, four successes to failures (paired exact p=6.16e-6).

## Throughput and latency
- 1,000 / measured total ms describes throughput upper bound for continuously available tracked point detections on the benchmark CPU, **not** physical camera frame rate.
- Three-frame pairwise-only: 34.57 ms / solution (~28.9 solutions/s capacity); five-frame pairwise-only: 48.59 ms (~20.6 solutions/s capacity); five-frame triangle-ATLAS: 54.47 ms (~18.4 solutions/s capacity).
- For a 30-fps camera, collecting a new three-frame window from cold start takes 66.7 ms from first to third exposure (or about 100 ms if counting the full three frame periods), and collecting five takes 133.3 ms from first to fifth exposure. These are **additional sensor acquisition constraints**; measured compute times are not the whole end-to-end latency.
- Streaming overlapping windows may reuse past frames rather than acquiring five new images per solve; the prototype does not implement a real-time persistent tracker or measure embedded CPU scheduling.

## Limitations and interpretation
- Full-sky detection not tested: only 1,000 catalog patches and 120 unseen directions.
- Simulated false stars are predominantly independent between frames. Real false signals can persist; a moving false star was added in 30/120 sequences, but this does not model all persistent defects, satellites, hot pixels or spacecraft maneuvers.
- Camera motion includes only roll, with fixed pointing. Real yaw/pitch and translational shifts are not tested; rapid roll and cross-frame occlusions can break simple nearest-neighbor matching.
- Target star detections and magnitudes are generated from the same Gaia data family as the catalog, not real camera exposures.
- Results are simulated pointing accuracy, not a full spacecraft flight test.
- **Main result:** temporal filtering is the source of the gain. Requiring six anchors to match as a combined object still does not beat best single-anchor matching or ATLAS in five-frame performance; radius and brightness gradient add no compelling advantage.
- **Practical priority:** test a three-frame temporal tracker with pairwise brightness/distance features against the five-frame ATLAS baseline on real image sequences, with boresight motion, correlated false detections, camera extraction time and on-board processor profiling.

## Artifacts
- atlas_temporal_experiment.py; atlas_temporal_results.csv
- atlas_temporal_ablation.py; atlas_temporal_ablation_results.csv
