# ATLAS

ATLAS estimates where a camera is pointing from a noisy star field. It registers successive frames, keeps recurring star detections, and uses triangle geometry and relative brightness to select up to 40 candidate fields from a Gaia reference catalog. A geometric fit estimates the camera’s pointing direction.

## Results

On 120 simulated sequences with fixed pointing and changing roll, combining five frames improved the refined boresight estimate in aggregate. A trial counts as correct when its pointing error is below 0.1°.

| Input | Correct after refinement | Mean processing time |
| --- | ---: | ---: |
| Last frame alone | 79/120 (65.8%) | 38.46 ms |
| Five frames | 104/120 (86.7%) | 54.47 ms |

Both rows use the same 120 sky directions. Times include matching and verification, plus registration for five frames. Camera exposure, centroid extraction and offline catalog construction are excluded. Performance on real images and during yaw or pitch maneuvers remains unmeasured.

The ablation also matters. Under the approximate pointing metric, five-frame ATLAS scored 91/120 and pairwise distance plus brightness alone scored 90/120. The strongest evidence favors combining frames. The incremental contribution of triangle features is uncertain.

[Technical report (PDF)](docs/ATLAS_Report.pdf) · [Experiment conditions and ablations](docs/atlas_temporal_report.md) · [Per-trial data](results/atlas_temporal_results.csv)

## Run one identification

From the repository root, use Python 3.11 or 3.12.

```sh
python -m pip install numpy scipy pandas
python experiments/demo.py
```

The demo builds 1,000 reference fields from the included catalog and processes one seeded five-frame sequence. The first sequence should yield 22 retained tracks and a refined pointing error around 0.00715°. Inputs to the matcher are normalized 2D star centroids and apparent magnitudes.

## Repeat the temporal experiments

```sh
python -m pip install numba
python experiments/atlas_temporal_experiment.py
python experiments/atlas_temporal_ablation.py
```

Numba accelerates scoring for full runs. New outputs go to `runs/`. The archived measurements in `results/` are the source for the report. Runtime varies by machine.

## Data

The included Gaia bright-star extract contains source IDs, right ascension, declination and G-band magnitudes. The simulations use the same catalog for generating detections and building reference fields. Credit for the underlying observations belongs to ESA’s Gaia mission and the Gaia Data Processing and Analysis Consortium.
