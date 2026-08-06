# Geo context-view sampling — leakage ablations (design notes)

Problem: the geo encoder (frozen lagernvs = VGGT, a pose estimator) is fed the target
segment's own frames, so it can recover the target camera trajectory → **data-level
leakage** into the camera-diffusion conditioning. We want scene grounding without
leaking the trajectory.

## Options implemented in `dataset_dl3dv.py` (via `cfg.geo_view_sampling`)

### `even` (baseline, DEFAULT — currently trained)
- `geo_num_views=4` frames evenly spaced WITHIN the target segment `[s:e]`
  (`_even_indices`), e.g. relative `[0,16,32,48]` for a 49-frame segment.
- Leaks the target trajectory to VGGT. This is the reference/baseline run.

### `random_inseg` (shuffling ablation — run first)
Enable with `LATENTCAM_CONFIG=config_geo_shuf`. Anchor (frame s, kept at index 0 = VGGT
reference) + `geo_num_views-1`(=3) frames randomly sampled from `(s, e)`, **re-drawn every
access**. Same #views as `even` (fair comparison). Still in-segment → does not remove
leakage, but randomizes the frame SET each epoch (augmentation; avoids memorizing the
fixed even frames). Chosen over reorder-only shuffle, which is a VGGT no-op (see below).

### `frustum_cover` (pose-only max spatial coverage — IMPLEMENTED)
Enable with `cfg.geo_view_sampling='frustum_cover'`. Picks `geo_cover_k`(=6) views from the
WHOLE trajectory (in/out of segment ignored), within `geo_cover_radius`×seg_scale of the
anchor, whose view frustums together cover the most nearby space. Pose-only (no depth):
- "nearby space" = uniform 3D grid ball at the near cameras' mean look-at point.
- coverage element = (grid point × viewing-direction bin) so the same point seen from a NEW
  angle still counts → 6 views stay viewpoint-diverse even under DL3DV's wide FoV.
- greedy maximum set-coverage; view0 (max single coverage) = VGGT reference; when saturated
  (linear/forward trajectories) it pads via farthest-point sampling in (position+optical-axis)
  space so padded views stay spread (not near-duplicate). e.g. scene 487c: [0,17,1,2,3,4]
  (old nearest-pad) -> [0,17,44,25,39,31] (FPS-pad).
- Render sanity (2026-07-19): general_512 @ 256×448 fed the 6 frustum_cover views renders a
  coherent novel-view video (`lagernvs/render_frustum_cover.py`, DL3DV 1K a4c2) — the selected
  views carry enough scene info at geo resolution.
- depth cap `zmax` derived from candidate→ball distances (robust to forward trajectories).
- **radius swept 1.0–4.0 over 10 scenes** → default **2.5** (first radius with 0 degenerate
  selections; 82% coverage, 62° mean viewpoint diversity). Smaller = higher coverage but
  clustered/degenerate; larger = more diversity but views drift from target.
`scripts/vis_frustum_cover.py` (single), `vis_frustum_cover_multi.py` (4 scenes),
`frustum_cover_sweep.py` (radius sweep).

### `hybrid` (covis retrieval — IMPLEMENTED but NOT VALIDATED; parked)
Enable with `LATENTCAM_CONFIG=config_geo_covis`. Views = `geo_num_inseg`(3) +
`geo_num_covis`(3) = 6:
- **in-segment (3)**: evenly spaced in `[s, s+span]`, `span=num_frames//8` — near-anchor
  scene grounding, minimal trajectory leak.
- **out-of-segment covis (3)**: retrieved from frames OUTSIDE `[s:e]` that co-observe the
  anchor's look-at point → anchored scene context, decorrelated from the target path.
  - gate: candidate center within `geo_covis_radius`(2.0)×segment-scale of anchor,
    optical axis within `geo_covis_max_axis_deg`(80°), look-at point X projects inside
    the candidate frustum (depth>0, in image).
  - score: MVSNet-style piecewise-Gaussian on triangulation angle at X (θ0=`geo_covis_theta0`=10°).
  - diversity: farthest-point sampling on the **viewing direction toward X** (azimuth
    around the scene point) → avoids clustered picks. Falls back to in-segment if too
    few covis candidates (degenerate short/linear captures ~4% of segments).

### ⚠️ Provenance / status of `hybrid`
NOT a published method — a non-learned COMPOSITE:
- co-visibility score = **MVSNet (Yao et al., ECCV 2018)** view-selection score.
- diversity = **Farthest-Point Sampling (PointNet++, Qi et al. 2017)**, applied to view
  directions (our adaptation).
- frustum/FoV gate = standard MVS/SfM co-visibility practice.
- retrieval-for-disentanglement framing = our design; conceptually related to I3DM
  (learned retrieval, replaced here by geometry) and RecamMaster / CineScene
  disentanglement.
Because it is unvalidated, it is **parked**. Run the simpler shuffling ablation first.

## Coverage analysis (why hybrid is feasible on DL3DV)
`scripts/analyze_geo_retrieval_coverage.py` (500 scenes, 3274 segments): at radius =
1.0×segment-scale, **96%** of segments have ≥4 out-of-segment frames near the anchor;
only ~4% degenerate. `scripts/covis_compare.py` shows direction-FPS + wide gate reduces
selection clustering (minPairAngle 10.9°→52.6° on the sampled scene) and rescues
degenerate scenes.

## VGGT note (why "fix-first + reorder-rest" shuffle is a no-op)
VGGT (`aggregator.py`) uses a special camera/register token for frame index 0 and a
single shared token for ALL other frames, with no cross-frame positional encoding
(RoPE is 2D spatial within a frame). Global attention is permutation-equivariant across
frames. So reordering the non-first frames yields the same output token set; combined
with our order-invariant (all-ones-mask) cross-attention downstream, it does not change
the conditioning. A meaningful shuffle must change the SET of frames or WHICH frame is
the reference (index 0), not merely their order.
