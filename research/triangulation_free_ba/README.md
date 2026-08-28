# Triangulation-free BA synthetic mechanism probe

This directory contains a bounded host-side experiment inspired by
*Triangulation-Free Bundle Adjustment with Graduated Non-Convexity for Camera
Pose Refinement from Coarse Priors* (arXiv:2608.21008v1).

It is not a reproduction of the paper's MobileBrick or ScanNet++ results and it
is not a production pose refiner. The reference implementation and its exact
match databases are not available here, while this checkout has no Ceres,
Eigen, SuiteSparse, TBB, or OpenCV dependency. The probe therefore uses NumPy
and a tiny dense finite-difference Levenberg-Marquardt solver.

## Experiment controls

- A deterministic six-camera, non-planar pinhole scene supplies geometry,
  image observations, and a fixed pairwise match graph.
- Every camera receives the requested rotation magnitude and 5 mm translation
  per degree. A seed changes only the per-camera perturbation directions; image
  observations and matches remain identical across seeds and levels.
- Correct correspondences are the clear majority. The outlier count is fixed at
  12% of the inlier count (about 10.7% of all matches) and deliberately selected
  to support a coherent 16-degree/80 mm false-pose lobe. Its generator seed
  (`10000`) is separate from the evaluated pose seeds (`0..2`). This is a
  repeated-texture stressor, not a calibrated model of SIFT errors.
- Every observed feature has one positive depth represented in log space,
  initialized to 1 m. Each match contributes the paper's two symmetric
  cross-projection residuals.
- The robust loss is `rho(s) = a * atan(s / a)`. The plain arm uses `a=1e2`;
  the graduated arm warm-starts `1e4 -> 1e3 -> 1e2`.
- Both arms receive the same maximum total LM iteration budget. Early stopping
  can make the number of attempted iterations differ.
- All camera poses are optimized. The coarse-basin configuration uses the
  paper's position prior (`lambda_p=1e-2`), no orientation prior, and no trust
  fallback.
- Rotation and camera-center accuracy are primarily reported after Sim(3)
  alignment, as in the paper's basin study. Absolute metric camera-center error
  is printed separately so gauge drift stays visible.
- A separate clean structure-bootstrap panel excludes decoy matches and uses
  known synthetic track identities. Deterministic two-ray hypotheses are ranked
  by retained observations, reprojection error, and parallax, then gated and
  refined before a shared-point BA solve with matched loss and position
  regularization. This deliberately favors the triangulation control and
  isolates prior-dependent triangulation/gating and its resulting coverage loss
  in this fixture. A result is valid for whole-rig scoring only when the
  retained camera graph covers and connects all six cameras, all metrics are
  finite, and every observed point remains in front of its camera.

## What the explicit regressions test

1. Synthetic truth yields zero symmetric cross-projection residual.
2. The image observations and match graph do not change with perturbation level
   or seed, and no camera is secretly left at ground truth.
3. In the fixed clean 16-degree diagnostic, prior-pose triangulation retains 16
   of 72 observations across only four of six cameras, and its frozen
   reprojection cost is lowest at the prior along the sampled 21-value
   prior-to-truth interpolation. Accurate-pose structure and independently
   grid-minimized ray depths are lowest at truth on that path. After each
   ray-depth curve is min-max normalized, the first `a=1e4` descent step is at
   least five times larger than `a=1e2`.
4. Across three fixed perturbation seeds at 32 degrees/160 mm, the equal-budget
   GNC schedule produces more solutions below the paper's 1-degree threshold
   and a lower median aligned rotation error than the nominal solve.
5. At 16 degrees, the 4 px prior-triangulated graph covers only cameras
   `{0,1,3,4}`, so cameras 2 and 5 have no reprojection residual or orientation
   prior and whole-rig recovery is unobservable. Full structure triangulated
   from accurate poses recovers. Two cross-controls isolate the dominant cause:
   ungated prior-triangulated structure with all 72 observations recovers,
   while accurate points under the sparse 16-observation mask remain
   insufficient. Coverage loss, rather than initial XYZ values alone, drives
   this toy failure.
6. Retention is reported across explicit 2/4/8/16 px gates rather than tuning a
   gate after observing pose outcomes.
7. With clean matches and an accurate prior, neither lifted schedule materially
   moves the aligned solution.

The third and fifth results support the paper's structure-commitment mechanism
qualitatively: prior-dependent gating creates a sparse subgraph whose frozen
cost is lowest at the prior on the sampled interpolation, while full
accurate-pose or ungated structure recovers. They sharpen the diagnosis for
this fixture: retained coverage, more than the initial XYZ values, causes the
failure. Because the primary sparse graph omits two cameras, this is evidence
of coverage collapse, not proof that a fully observable six-camera BA is
trapped in a local minimum. The result is also narrower than Figure 2. For
non-comparable context, the paper's aston example retains roughly 9% of its
COLMAP observations, while this known-track toy retains 22% under its declared
4 px gate. Its point BA uses the same arctan loss and position prior for a
controlled comparison, not COLMAP's ordinary shared-point objective.

The fixed seeds are regression fixtures established with this implementation,
not a blinded confirmatory sample. In the current three-seed run both schedules
succeed at 16 degrees (3/3 each); the separation appears at 32 degrees (nominal
0/3, GNC 2/3), and neither schedule succeeds at 64 degrees. These toy counts
are useful mechanism evidence, not an estimate of the paper's success rates.

A denser sweep on that same fixed outlier graph shows a gradual transition:

| Perturbation | Nominal success | GNC success |
| --- | ---: | ---: |
| 20 degrees | 2/3 | 3/3 |
| 24 degrees | 1/3 | 3/3 |
| 28 degrees | 1/3 | 3/3 |
| 32 degrees | 0/3 | 2/3 |
| 36 degrees | 1/3 | 2/3 |
| 40 degrees | 0/3 | 1/3 |
| 48 degrees | 0/3 | 0/3 |

The result is not universal across outlier graphs. At 32 degrees over five
independent false-lobe seeds and three pose seeds each, nominal succeeds 7/15
and GNC 11/15; medians are 1.215 and 0.109 degrees. GNC wins nine paired cells,
loses five, and still reaches a 27.54-degree wrong consensus. Treat the schedule
as a favorable robustness trade in this toy, not a guarantee.

Depth initialization changes individual outcomes but not that aggregate
direction on the primary graph. Starting all lifted depths at 0.5/1/2 m gives
nominal success counts of 1/3, 0/3, and 1/3 at 32 degrees, versus GNC counts of
2/3, 2/3, and 3/3. The paper's 1 m initialization remains the primary result.

Run the research regressions explicitly (they are intentionally outside the
production wheel/release test matrix):

```sh
PYTHONPATH=. python3 research/triangulation_free_ba/test_validation.py -v
```

Run the multi-level report (three seeds by default):

```sh
PYTHONPATH=. python3 -m research.triangulation_free_ba.validation
```

Use `--iterations 8` for the smaller equal-budget sweep used during this
evaluation. The command prints every seed plus per-level success counts using
the paper's mean aligned rotation error below 1 degree rule.
Use `--initial-depth` and `--decoy-pose-seed` for the two declared sensitivity
axes without changing image observations or pose perturbations.

Reproduce the five-graph aggregate in one command:

```sh
PYTHONPATH=. python3 -m research.triangulation_free_ba.validation \
  --levels 32 --seeds 3 --iterations 8 \
  --decoy-pose-seeds 10000,10001,10002,10003,10004
```

Run the clean known-track structure and gate controls separately:

```sh
PYTHONPATH=. python3 -m research.triangulation_free_ba.validation \
  --levels 1,2,4,8,16,32 --seeds 3 --structure-only \
  --structure-gates 2,4,8,16 --structure-iterations 24
```

At the primary 4 px gate, every prior-triangulated graph is insufficient for
whole-rig scoring: it covers only two to five of the six cameras. Accurate-pose
structure covers all cameras and recovers all 18 cells to about 0.112 degrees
aligned. The command reports camera coverage, connected-component count,
positive cheirality, and valid/success counts so partial graphs cannot be
mistaken for recovered poses. Wider gates retain more observations; only
all-camera connected cells are counted.

## Important deviations and open validation work

- Dense finite differences, additive axis-angle updates, log-depth clipping,
  and explicit step clipping replace the paper's Ceres quaternion manifolds,
  positive parameter bounds, automatic derivatives, and sparse Schur solve.
- Intrinsic/distortion refinement, the nominal orientation prior, and the
  production trust fallback are omitted. This probe targets only the coarse
  fixed-intrinsics basin configuration.
- The ray-depth landscape uses a 72-value depth grid over 0.2-4 m, rather than
  solving each independent depth exactly.
- Its normalized forward-drop diagnostic follows one hand-selected path from
  the coarse prior to truth. It illustrates loss shape, not an optimizer
  gradient or a measured convergence-basin boundary.
- The triangulation control uses known synthetic track identities and ignores
  the constructed decoy edges. Its best-inlier-pair triangulator and one BA pass
  are neither COLMAP's track builder nor the paper's two triangulate/adjust
  rounds.
- The clean shared-point control and the constructed-outlier lifted/GNC sweep
  are separate experiment panels. Together they probe individual mechanism
  pieces; they are not an end-to-end reproduction of the paper's pipeline.
- The primary 4 px gate is a declared synthetic choice, not a default published
  by the paper. Its frozen shared-point curve also uses the local arctan loss;
  the paper does not publish an equivalent classical loss/normalization.
- Shared-point BA uses the same arctan loss and metric position prior as the
  lifted arm, but its residual graph and normalization necessarily differ. The
  paper's COLMAP control instead uses ordinary reprojection BA without these
  pose-prior terms.
- The accurate-structure arm triangulates under exact synthetic poses, stronger
  than the paper's structure triangulated from a merely accurate ARKit prior.
- The fixed false-pose outliers create one useful mechanism test; they do not
  establish robustness to real repeated texture, retrieval/GPS priors, lens
  errors, larger graphs, or different match distributions.
- Runtime and success counts from this dense toy solver are not comparable to
  the paper's CPU timing or benchmark tables.
- The weak metric position prior does not reliably preserve the absolute frame
  in this dense toy. On the clean exact-prior fixture, plain and GNC remain near
  truth after Sim(3) alignment but drift by about 23 mm and 80 mm respectively
  before alignment. This blocks production adoption even though the paper's
  primary aligned metric looks good.

A credible real-data gate needs the authors' released code or a host-only Ceres
port, raw pairwise feature matches, the exact perturbation protocol, and several
captured datasets. In this repository the method belongs before
`inputDataFromDescriptor()` and before normalization, emitting a non-destructive
refined-pose descriptor. It should not replace the existing Raw/CamP in-training
photometric modes. Any real integration also needs an explicit input-frame pose
delta guardrail before refined metric poses are accepted.
