# Changelog (latentcam)

All notable changes to the latentcam sub-project. Follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- **image dir is now resolved, not hardcoded** (`main/dataset_dl3dv.py` + 6 scripts). Added
  `IMAGE_DIR_NAMES = ('images_4', 'images_8', 'images')` and `scene_image_dir(scene_dir, names)`,
  which returns the first existing candidate (one `isdir` per candidate — no per-frame stat, which
  matters on lustre). `_parse_transforms` uses it via `getattr(cfg, 'image_dir_names',
  IMAGE_DIR_NAMES)`, so the search order is overridable from config. The same local `_img_dir()`
  helper replaced hardcoded `images_4` joins in `scripts/render/{render_geo_preds,
  render_scene_stitched,render_target_from_context}.py` and
  `scripts/context_select/{visualize_covis_retrieval,vis_frustum_cover,covis_compare}.py`.
  Motivation: the on-disk images were swapped to 480p (below) — nothing else in the pose/intrinsic
  math changes, because `transforms.json` is byte-identical between the 960p and 480p trees and
  always reports the ORIGINAL full resolution (w=3840, h=2160, fl_x=1720.22, cx=1920, cy=1080).
- **DL3DV images swapped 960p → 480p to reclaim disk** (`scripts/data/migrate_480_images.py`, new).
  `DL3DV-480/<chunk>/<scene>/images_8` (480×270) was `os.rename`d into the name-matched
  `DL3DV-960/DL3DV-10K/<chunk>/<scene>/` (same lustre FS → metadata rename, ~4s/1000 scenes),
  keeping the dir name `images_8`; then every `images_4` (960×540) under DL3DV-960 was removed,
  including chunks 8K–11K which have no 480p counterpart (explicit user decision — those scenes
  are not in `meta_worldtraj.csv`). Pre-flight verified 6098/6098 `meta_worldtraj.csv` scenes and
  7000/7000 scene dirs in 1K–7K match by name with identical frame counts and filenames
  (2,092,998 frames each side). Move runs before any delete, so no scene is ever image-less.
  Consequences: (a) the **training** geo path is essentially unaffected — `geo_image_hw =
  [256, 448]` vs a 480×270 source is still a downscale (270→256, 480→448), verified by loading
  a 1K sample post-migration: `scene_image_dir` → `images_8`, PIL size (480, 270),
  `hw_list` still (2160, 3840) from transforms.json, `images` (6, 3, 256, 448), `cam_param`
  (49, 11). (b) the standalone LagerNVS render scripts use SIZE=512, so those DO upscale now.
  (c) the geo latent cache (`DATA/DL3DV/latent_cache`) was computed from 960p → stale.
  (d) `meta.csv` (8048 rows) includes 1916 scenes in 8K–11K that now have NO image dir
  (10K:876, 11K:485, 8K:288, 9K:267); `meta_worldtraj.csv` (6098, 1K–7K) is clean. Image-
  dependent experiments inheriting the default `meta_csv: meta.csv` need switching.
- **lagernvs moved** `/data1/cympyc1785/lagernvs` → `camera_generation/tools/lagernvs`
  (same-FS rename; data/ are absolute symlinks so unaffected). Updated
  `lagernvs_repo_path`/`lagernvs_ckpt_path` in `main/conf/config.yaml` + `main/config.py` to
  the new path. A compat symlink at the old location is kept so in-flight runs / historical
  wandb configs / helper scripts keep resolving; remove it once all runs referencing the old
  path have finished.

### Fixed
- **top-down plots were a front/back view, not top-down** (`main/infer_validation_sample.py`,
  `scripts/render/topdown_swap.py`). They hardcoded the x-z plane, but transforms.json stores c2w
  in the nerfstudio frame where DL3DV's `applied_transform` (x↔y swap + z flip) is baked in — there
  world-up is X and motion lives in Y-Z, so x-z looked down the *forward* axis (a front/back view).
  Fix: anchor both GT and pred to the GT first camera (`inv(c2w[0]) @ c2w`, same first-frame
  anchoring as the GenDoP pyramid viz), which puts cam0 at the origin with the OpenGL camera axes
  (up=+Y, right=+X, forward=-Z). Drop up(+Y) → ground = X-Z; plot X horizontal, -Z vertical so the
  camera forward points up in the image (map-like). User-confirmed orientation.

### Changed
- **lazy dataset loading** (`main/dataset_dl3dv.py`, default `lazy_dataset: true`): `__init__` now
  builds only the lightweight sample/scene index (reading `prompts.json` + an n,h,w probe from
  `transforms.json`, no per-frame stat) and persists it to `<root>/.latentcam_index/<key>.pt`, so
  reruns load the index instantly instead of re-scanning all ~6k scenes. Scene poses/paths are
  parsed on demand in `__getitem__` (`_load_scene` + `_LazyScenes`, cached). Also dropped the
  per-frame `osp.isfile()` (~330 stats/scene on lustre) for a single `images_4` dir check — this
  was the main ~28-min init bottleneck. Verified byte-identical to the eager path (samples/cam_param/
  geo_c2w diff 0.0); `lazy_dataset: false` restores the old eager load. Note: with num_workers the
  per-worker scene cache grows toward the working set (no COW sharing like eager) — cap workers if
  RAM-bound.

### Changed
- **`scale_mode` naming unified** — "avg_scale" was overloaded (stored point-cloud value vs the
  camera-distance mean). Now `avg_scale` means **only** the stored point-cloud avg_scale
  (`<scene>/avg_scale/<seg>.json`), and the camera-distance one is `cam_dist_mean`:
  - `saved_avg_scale` → **`avg_scale`**, `_saved_avg_scale()` → `_avg_scale()`
  - `target_cam` → **`cam_dist_mean`**, `_camera_based_avg_scale()` → `_cam_dist_mean_scale()`,
    `_cam_avg_scale_context()` → `_cam_dist_mean_context()` (mode `context_longer` unchanged)
  Old spellings still work via `_SCALE_MODE_ALIASES` / `resolve_scale_mode(cfg)`, so existing
  yaml, wandb configs and in-flight resumes are unaffected (verified: `avg_scale` vs
  `saved_avg_scale` and `cam_dist_mean` vs `target_cam` both give cam_param diff 0.0).
  The batch now also carries **`norm_scale`** = the divisor the active mode produced; the old
  key `avg_scale` is kept as an alias of the same tensor (SCVideo's name), so every consumer
  (`train_latent_cam_dm.py`, `infer_*.py`, `gen_*.py`, `cache_geo_embeddings.py`) is unchanged.
  Updated: `main/conf/config.yaml` (`scale_mode: avg_scale`), `geo_worldtraj{,_before,_seglist}`,
  `geo_hybrid_shuf`, `textonly_savedscale`, `main/config.py` default (`cam_dist_mean`), and the
  analysis scripts' labels/columns (`scripts/render/compare_*`, `scripts/data/
  norm_camera_length_stats.py`, `scripts/coverage/compare_scene_span_scale.py`).
  `main/configs_backup/` left as-is (historical, covered by the aliases).

### Added
- **`CamDataset.from_segments(cfg, segments)`** (`main/dataset_dl3dv.py`): segment-scoped dataset
  for inference/rendering. Instead of indexing the whole corpus, it reads `meta.csv` once to map
  the flattened scene name back to its chunk, then opens `prompts.json`/`transforms.json` for only
  the requested scenes — cost is O(#requested segments). `__getitem__`, geo context sampling and
  normalization are untouched, so output is identical to the full dataset (verified: 10 segments,
  all tensors diff 0.0 vs `CamDataset(cfg,'train')`, and the re-dumped `render_inputs.pt` byte-match
  the previous run). Used by `main/dump_render_inputs.py` (single `RI_SEG`) and
  `main/dump_avgscale_render.py` (new optional `SEGS` env). Dumping the 10 avg_scale-test segments:
  **~20 min → 6.2 s**. For reference SCVideo has no such path — its `main/infer_cam_dm.py` calls
  `Trainer._make_batch_generator(include_train=False)`, which builds the full `CamDataset` and
  `random_split`s 90/10, so inference there loads exactly as much as training.
- **`scale_mode: saved_avg_scale`** (`main/dataset_dl3dv.py`): replicate SCVideo's original
  normalization (`data/dataset_large.py`) — normalize camera translations by the STORED
  point-cloud `avg_scale` (`<scene_dir>/avg_scale/<seg_key>.json` = mean ‖scene point − first
  camera‖, ~10–44) instead of the camera-based mean. Also mirrors SCVideo's intrinsics under this
  mode: width/height from principal point (cx·2, cy·2) + normalized relative to frame 0
  (frame0 intr → [1,1]). Falls back to camera-based if the json is missing. Other modes unchanged.
  Purpose: match the scale the default VAE (`vae_20260302_300`) was trained on (SCVideo used stored
  point-cloud avg_scale; the DL3DV port had silently switched to camera-based `target_cam`).
  Added `self.scene_dir_list` + `_saved_avg_scale()`.

### Fixed
- **porting divergence from SCVideo `dataset_large.py`** surfaced: the DL3DV `dataset_dl3dv.py`
  port had diverged in 3 places — (1) avg_scale source (camera-based vs stored point-cloud), (2) no
  frame-0-relative intrinsics normalization, (3) width/height from stored w,h vs cx·2/cy·2. All three
  are now reproducible via `scale_mode: saved_avg_scale` (existing camera-based modes left intact).

### Added (more)
- **`scripts/data/norm_camera_length_stats.py`**: camera-length distribution under the two training
  normalizations (point=target_cam vs dist=1.35·max) over ~5.4k real training segments (rebuilt
  standalone from transforms.json + prompts.json, no dataset load). Per-frame ‖center‖/avg_scale,
  per-segment span + path length (mean/std/var/percentiles) + histograms →
  `results/compare/normalization_camera_length/`. point var 0.345 (span 1–24, heavy tail) vs dist
  var 0.054 (span const 0.741, bounded ≤0.741).
- **`main/cache_geo_embeddings.py`**: precompute + cache frozen geo embeddings (fp16) per DL3DV
  segment (`data/DL3DV/latent_cache/<data_name>.pt`, (M,768)) so training can skip the per-step
  LagerNVS forward (frozen + deterministic context). fp16 chosen: geo_emb |max|≈14.6 → no overflow,
  rel-err 1.8e-4 (training already bf16). ~285GB for full worldtraj scope; pilot = 1K batch.
- **`main/dump_render_inputs.py`** (+ vendored `tools/lagernvs/render_pred_from_dump.py`, not
  committed): render a results/validation predicted trajectory with LagerNVS. Two-step to avoid the
  latentcam↔lagernvs `models` package clash — step 1 (latentcam) dumps the exact inference-time geo
  context (image paths + OpenCV-world c2w + intrinsics, same frames as at inference) + the pred
  cameras to `render_inputs.pt`; step 2 (lagernvs) loads context at 512, adjusts intrinsics,
  normalizes exactly like training (`normalize_extrinsics` = view0-relative + 1.35·max, camera_scale
  0.7407) and `build_cam_cond` → `render_chunked` with the same general_512 model used for geo
  conditioning → `render_pred_lagernvs.mp4` + `render_pred_grid.png` per model. worldtraj/align share
  the frustum_cover context; hybrid uses its own.
- **`main/conf/experiment/textonly_align.yaml`**: text-only counterpart of geo_worldtraj_align —
  no geo, but first_farthest_135 (1.35·max_dist) normalization + vae_worldtraj (latent_scale
  0.569379) + meta_worldtraj.csv. Purpose: text-only vs worldtraj_align isolates the geo effect
  under dist normalization. (Training launched on GPU5 per explicit user instruction — overrides
  the CLAUDE.md no-4~7 rule; run `dl3dv_textonly_align`, screen train5.)
- **`main/infer_textonly_batch.py`**: runs the text-only model on the worldtraj validation targets
  (inputs reconstructed from saved `_transforms_ref.json` + `_caption.json`, no CamDataset load) so
  text-only vs worldtraj can be compared PER-PAIR on identical targets. Writes
  `results/compare/text-only_vs_worldtraj/textonly_preds/`.
- **`scripts/render/compare_textonly_vs_worldtraj.py`**: per-pair top-down (GT + worldtraj+geo pred
  + text-only pred + geo-context stars, first-cam anchored X/-Z) on the same 160 targets + world
  pos_rmse; both use target_cam so the diff is geo on/off. `_summary.png` (paired) + `_scores.csv`.
- **`scripts/render/compare_textonly.py`**: visualizes the text-only model
  (`20260719_210144_dl3dv_textonly`, target_cam + no geo) inference → `results/compare/textonly/`.
  Per-target top-down (GT vs pred, first-cam anchored X/-Z) + world pos_rmse/rot + CLaTr; `_summary.png`;
  and `_vs_geo.png` a DISTRIBUTIONAL box comparison vs point(worldtraj)/dist(align) (targets are
  disjoint across models — distributions, not per-pair). Kept `normalization_point_vs_dist/` intact.
- **`preds_scores.csv`** (`main/evaluate/eval/src/eval_only.py`): per-sample dump of every
  wandb-logged eval metric. Per-sample columns `captions/{precision,recall,fscore}`,
  `clatr/clatr_score` (100·cos(pred-traj, text)), `clatr/pred_ref_cosine` (100·cos(pred-traj,
  GT-traj)); plus `clatr/{precision,recall,density,coverage,fcd}` repeated as run-level constants
  (distributional/set-level → no per-sample value). `preds.csv`/`preds_pcf.csv` unchanged.
- **`scripts/data/extract_geo_context.py`**: dumps the geo-context camera world centers for the
  160 validation targets (geo_worldtraj config; context selection is deterministic + identical for
  align) → `_geo_context.json`, so the comparison viz can overlay conditioning views as stars.
- **`scripts/render/compare_norm_topdown.py`**: compares two normalization schemes
  (point = target_cam vs dist = first_farthest_135) on the same 160 validation targets. Per-target
  top-down (GT + both preds + geo-context cameras as magenta stars, first-cam anchored X/-Z) +
  world-space scores (pos_rmse/rot) and per-target CLaTr score (from `_clatr.json`), plus a 2×2
  `_summary.png` (pos_rmse + CLaTr, sorted + paired) and `_scores.csv`. Metrics in denormalized
  world so they are comparable regardless of each model's normalization/VAE. Output →
  `results/compare/normalization_point_vs_dist/`.
- **`models/GenDoP/extrinsic2pyramid/vis_validation_anchor.py`**: trajectory pyramid viz that
  replicates GenDoP's **original** `dataset/extrinsic2pyramid/visualize.py::draw_json`
  preprocessing — first-frame anchoring (`c2ws = inv(c2w[0]) @ c2ws`) + optional 2-frame
  subsample — before calling `vis.py::draw_json`. The plain `vis.py` path omits anchoring and
  therefore inherits each dataset's arbitrary world up-axis (X for latentcam, Y for DataDoP),
  which is why front/top/side came out mislabelled. Anchoring re-expresses the trajectory in the
  first camera's frame so the views are canonical regardless of world up-axis. Verified: anchored
  DataDoP output matches the reference `shot_0003_traj_cleaning.png` exactly. Supersedes the
  earlier `vis_validation_rot.py` per-trajectory rotation hack (wrong approach). Single-file mode:
  `vis_validation_anchor.py IN.json OUT.png [--sub]`; no-arg mode sweeps `results/validation`.
- **`main/infer_swap_ablation.py`**: swap-ablation inference for the geo camera-DM models.
  For N fixed val samples (deterministic split), runs 3 modes and saves each per model/mode
  (`results/swap_ablation/<model>/<mode>/`): (a) normal, (b) ctxswap — keep the anchor context
  view (view0=frame s), take the rest from another sample, (c) textswap — keep context, swap
  text. Per-sample noise seeded so modes are directly comparable. Reads SWAP_CKPT/SWAP_OUT/
  SWAP_N/SWAP_TAG env + `experiment=` Hydra override.
- **`geo_cover_before_only`** flag (`dataset_dl3dv.py`): restrict out-of-segment geo context
  to frames BEFORE the target segment (index < s) instead of the longer out-of-seg side; the
  target segment must have frames before it, so first-segment targets are filtered out
  (`s < geo_cover_k`) — the target segment can be the 2nd segment onward. New Hydra experiment
  `conf/experiment/geo_worldtraj_before.yaml` (identical to geo_worldtraj except before-only
  context: first camera s + 5 out-of-target views drawn from earlier segments). Default off —
  existing geo configs unchanged.
- **`geo_lagernvs_skip_ctx_norm`** flag + `build_cam_token(override_scale=...)`
  (`models/geo_encoder.py`, `geo_encode`): skip LagerNVS's own 1.35·max(context) normalization
  and reuse the target's initial 1.35·max_dist (avg_scale) so target & geo latent share one
  frame+scale (full coordinate alignment). New Hydra experiment
  `conf/experiment/geo_worldtraj_align.yaml` (scale_mode=first_farthest_135 + vae_worldtraj
  VAE, latent_scale 0.569379; context selection identical to geo_worldtraj).
- **`geo_shuffle_keep_first`** flag (`dataset_dl3dv.py`): when shuffling geo context order,
  keep view0 (frame s / VGGT reference) fixed and shuffle only the rest. New Hydra experiment
  `conf/experiment/geo_hybrid_shuf.yaml` (hybrid: 2 in-target [view0=s] + 3 out-of-seg covis,
  first fixed + rest shuffled, target_cam scale, geo_posed, meta_worldtraj).
- **`geo_cover_subtract_first`** flag + `frustum_cover_select(prepicked=...)` in
  `dataset_dl3dv.py`: when the first camera (frame s) is a fixed context view, subtract its
  coverage from the greedy union first so the remaining k−1 views maximize RESIDUAL coverage.
  New Hydra experiment `conf/experiment/geo_worldtraj.yaml` (lagernvs, first-cam-fixed +
  residual-coverage out-of-seg selection, `scale_mode=target_cam`, geo_posed 1.35·max, default
  VAE/CLaTr, meta_worldtraj). Default off — existing geo configs unchanged.
- **`meta_worldtraj.csv`** (DL3DV root) = `meta.csv`[1K–7K] − `blacklist.csv` = 6098 scenes; the
  dataset now honors `cfg.meta_csv` (default `meta.csv`) so a run can select its scene list.
  `blacklist.csv` gained 34 scenes (21 teleport + 13 image/pose length-mismatch) detected by
  `filter_dl3dv.py`; entries reformatted to `<split>/<hash>` (matching normalized to basename).
- **`scale_mode='first_farthest_135'`** (`dataset_dl3dv._first_farthest_scale`): LagerNVS-style
  per-segment normalization = 1.35 × max(‖cam center − first camera‖). New Hydra experiment
  `conf/experiment/vae_worldtraj.yaml` (geo off, this scale, meta_worldtraj, batch 64) for a
  camera-VAE re-fit on the WorldTraj scope.

### Changed
- **Config package retired → `configs_backup/`.** The active pipeline is fully on Hydra
  (`conf/` + `hydra_cfg.load_cfg`); the dead `import configs` shim was stripped from all loaders.
  For the legacy scripts still on the Python configs, `config.py` / `config_large.py` /
  `config_vae.py` were pulled back into `main/` (root_dir depth reverted to `..`) so their flat
  `from config* import` resolves. (`config_large`/`config_vae` still crash at import on their
  hardcoded absent data paths — pre-existing; to fix when those scripts are needed.) Verified:
  base `config` and active `train_latent_cam_dm` both load correctly side by side.
- **`scripts/` reorganized by purpose** into subfolders: `render/` (15, LagerNVS NVS + shared
  render libs), `coverage/` (8, coverage dump/analysis/blacklist/scale study), `context_select/`
  (7, view-selection methods + viz), `vae/` (2), `data/` (2, dl3dv filter + run-dir migration),
  `viewer/` (1, viser). Cross-folder sibling imports preserved via a small `sys.path.append`
  snippet (scripts root + all subfolders) injected per file; the 3 scripts that reach `main/`
  via a relative `..` had their depth fixed (`../.. `). Verified: all 35 files parse; cross-bucket
  imports resolve (render/coverage/context_select/data/vae).
- **Hydra/OmegaConf config system** (`main/conf/`): all 27 Python configs auto-ported to
  `conf/config.yaml` (base, 82 fields) + `conf/experiment/*.yaml` (deltas), composition via
  Hydra defaults (base + experiment override) — mirrors the old inheritance. `main/hydra_cfg.py`
  `load_cfg()` composes with standard Hydra CLI (`experiment=rolling lr=1e-4`) AND back-compat
  `LATENTCAM_CONFIG` env; returns a `cfg` behaving like the old Config (attribute access, real
  `t5_dtype` torch dtype, mutable) + `cfg_dict`. Every active loader (train/gen/profile) swapped
  to `load_cfg` (verified: composed == Python `cfg_dict` for all 23 experiments, 0 mismatches).
  Training now saves the full resolved config as `config.yaml` into the run dir + wandb run dir
  (`hydra_cfg.save_cfg_yaml`). CLaTr eval keeps its **own** Hydra in a subprocess — no conflict
  (verified: GlobalHydra clean after `load_cfg`; CLaTr eval subprocess runs end-to-end).
  Python `configs/*.py` retained (legacy config_large-based scripts still use them).
- `scripts/migrate_run_dirs.py` — wrote estimated `config.yaml` into 17 existing `results/`
  folders (mapped by `exp_name` suffix) and renamed 4 wandb dirs to `<ts>_<exp_name>` by exact
  timestamp match (live run + 39 orphan smoke/offline dirs left untouched).
- **Config layout**: moved all `main/config*.py` (27 files) into a `main/configs/` package.
  `configs/__init__.py` self-registers its dir on `sys.path`, so flat module names
  (`import config as _base`, `LATENTCAM_CONFIG=config_rolling`) and inter-config imports keep
  working unchanged. Every config importer (train/infer/gen/profile scripts + scripts/verify_vae,
  vae_interp) gained a one-line `import configs` before loading a config. `root_dir` in
  `config.py`/`config_large.py`/`config_vae.py` fixed to `osp.join(cur_dir, '..', '..')` (now one
  level deeper), so it still resolves to the latentcam root.

### Added (data)
- `scripts/filter_dl3dv.py` — DL3DV meta filter mirroring scenetok's `build_dl3dv_meta_row`
  (transforms/image presence, num_images>=34, images==poses, **teleport camera rejection**,
  image size + consecutive-frame checks; optional `--require-prompts`). Runs over `--subs`
  (default 1K–7K), writes `meta_tmp.csv`, and diffs against `meta.csv[subs] - blacklist.csv`.
  Finding: current `meta.csv[1-7K]` still contains 34 scenes that fail the (fixed) filter —
  21 teleport + 13 image/pose length mismatch. `meta_tmp.csv` is the cleaned list.

### Added
- **Text-only ROLLING model (per-token Diffusion Forcing + rectified flow)** on the
  causal-VAE latent (W=13 tokens, D=64). One model → full_sequence / chunk_ar / rolling
  inference by tau schedule only.
  - `models/camera_diffusion_model_latent.py`: per-token FiLM (no gate). `timestep_embedding`
    accepts `(B,)` or `(B,T)`; `forward` detects `per_token` (t_embed.dim()==3) and applies
    element-wise `(B,T,D)` scale/shift, else legacy `(B,1,D)` broadcast (bit-identical when off).
  - `main/config.py`: `per_token_noise` (False), `cfg_dropout_p` (0.1), `loss_tau_min` (0.02).
  - `main/tau_sampler.py`: W=13 tau mixture (30% iid / 40% ramp / 15% boot-up / 15% full-seq)
    → `(tau [B,W], labels [B])`.
  - `main/train_latent_cam_dm.py`: `per_token_flow_loss` (rectified-flow, tau-masked, CFG
    dropout) + training branch (`elif per_token_noise`) between AR and DDPM; validation
    per-token branch with tau-bin / pattern val-loss logging (CLaTr guarded off for per-token).
  - `main/config_rolling.py` + `main/config_rolling_smoke.py`: text-only rolling configs
    (causal VAE, per_token_noise).
  - `main/rolling_sampler.py`: 3-mode inference (gen_full / gen_chunk_ar / gen_rolling) in
    latent space + `latent_to_traj` (causal-VAE decode → (49,9) pose9 + (49,4,4) w2c) +
    `jerk_spectrum` (period-4 token-boundary / period-12 chunk-boundary power).
  - `main/test_per_token.py`: unit tests (per-token modulation, tau sampler, loss zero-denom).
  - `main/gen_scene_rolling.py`: generate a FULL scene with the rolling ckpt. Modes: `full`/
    `chunk_ar` (per-segment gen + hard-snap chaining, has seams) and `rolling` (CONTINUOUS —
    one sliding window across the whole scene, text switches per emitted token's segment,
    decode ONCE → seam-free; single scene scale = mean of per-seg avg_scale). `--scene-skip`
    picks a different scene. Saves transforms_pred/ref.json (viser) + top-down plot + drift.
  - `main/rolling_sampler.py`: `gen_rolling_scene()` — continuous multi-segment rolling used
    by the above.

### Added (tooling)
- `main/gen_scene_align.py` — per-segment inference placed two ways + GT, top-down. (A) per-seg
  aligned (anchor reset to each segment's GT start → local shape, no accumulation) vs (B) chained
  (running anchor → accumulated drift) vs GT. `--auto-best N` scans loaded scenes and picks the N
  with lowest mean per-segment err; renders a GT | per-seg | chained 3-column figure per scene.
- `main/profile_geo.py`, `main/profile_rolling.py` — latency breakdowns. geo: per-sample cost
  is ~49% VGGT re-encode + ~38% 6-image disk load + ~8% frustum_cover select, DiT only ~1.7%
  (context changes every segment → geo latent re-extracted, no cache). rolling: stage timing
  of T5 encode / rolling denoise loop / VAE decode.
- `scripts/compare_scene_span_scale.py` — statistical comparison of camera-normalization
  scales: `scene_span` (NEW scene-UNIFIED: max ‖center[i]−center[0]‖ over the whole video,
  one value/scene) vs per-segment `target_cam` and `context_longer`. Reports within-scene CV
  (how much the scale "keeps changing") + magnitude ratios; dumps per-scene/per-segment CSVs.
- `scripts/render_scene_stitched.py` — new options (defaults preserve old behavior):
  `--first-view-fixed` (frame 0 always context; its coverage subtracted first, residual
  coverage drives the greedy picks) and `--scale-mode scene_span` (size the coverage ball by
  the scene-unified span instead of per-segment seg_scale). `frustum_cover_select` gains a
  `prepicked=` arg; coverage ball now centered on the target segment. `--viz-all` now honored.
- `scripts/render_geo_preds.py` — render a geo run's GENERATED cameras with LagerNVS. Reads
  `<run>/test/<name>_transforms_pred.json` (generated OpenGL c2w, anchored at GT frame s in scene
  world), recovers the segment [s:e] by matching ref endpoints to scene cameras, picks out-of-seg
  coverage context, and renders the generated path vs the GT path side by side. N samples.
- `scripts/render_scene_global16.py` — whole-scene LagerNVS render from GLOBAL context: pick
  k=16 views by coverage over Monte-Carlo points sampled inside a sphere centered at the views'
  average look-at point (least-squares ray intersection); render ALL scene views (strided) as
  targets from those 16. Outputs GT|render video + context-selection top-down (context, look-at
  center, sphere). 3 scenes → target-coverage 1.00.
- `scripts/render_scene_compare.py` — 3-panel stitched comparison [GT | per-seg-scale |
  firstfix+scene_span] per scene: renders each segment with BOTH context-selection strategies
  and concatenates for direct visual A/B (GT kept on the left).
- `scripts/viser_val_cameras.py` — viser frustum viewer for validation-saved cameras
  (results/<exp>/test): pred (red) / target-ref (blue) / rest-of-scene (grey), with a
  sequence slider + Load + Next. Handles coords (saved OpenGL c2w -> OpenCV for viser;
  ref/pred recovered to the scene world so all three overlay; verified ref≡scene dist=0).

### Fixed
- **Context-selection leakage**: `frustum_cover` context selection anchored on the target
  MIDPOINT and scaled by the target segment's own extent (`seg_scale` over [s:e]) — i.e. it
  used the yet-to-be-generated target trajectory, unreproducible at inference. New
  `geo_anchor_first_frame` option (config, `_sample_geo_frustum_cover`, `dump_coverage_selk`)
  anchors on the target's FIRST frame only (known at inference) and scales the radius by the
  CONTEXT movement; `frustum_cover_select` gains `look_centroid=` to keep the look direction
  target-free. `config_geo_camscale` now sets `geo_anchor_first_frame=True`. Legacy behavior
  preserved when the flag is off. Blacklist regenerated with the honest metric (backup:
  data/blacklist_selk_tau0.7.TARGETANCHORED.bak.csv).

### Added
- `geo_first_view_target_s` config option: geo context view0 = target segment's first camera s
  (+ (k-1) out-of-seg retrieved), so LagerNVS anchors the geo latent to frame s (origin-aligned
  with the generation target frame). `config_geo_viewS.py` + `data/blacklist_selk_viewS_tau0.7.csv`
  (coverage recomputed with view0=s over [s+1:e]; `dump_coverage_selk.py --first-view-s`).
- `scale_mode` config option (`config.py`, default `'target_cam'`) selecting the camera
  translation normalization scale. New `'context_longer'` mode computes `cam_avg_scale`
  from the LONGER out-of-segment side, chunked into `num_frames` windows, averaging each
  window's camera movement (`dataset_dl3dv._cam_avg_scale_context`). Leakage-free and
  reproducible at inference from context only. Existing `'target_cam'` path unchanged.
- `config_textonly_camscale.py` — text-only run using `scale_mode='context_longer'`.
- `scripts/dump_coverage.py`, `scripts/viz_coverage.py`, `scripts/viz_coverage_dump.py`,
  `scripts/viz_coverage_compare.py` — out-of-segment coverage dump + statistics/plots
  (`both`앞+뒤 vs `longer`긴 쪽만 context) for coverage-based segment blacklisting.
- `scripts/render_target_from_context.py` — LagerNVS NVS probe: render target segment
  from out-of-segment (longer-side) context only, compare to GT, across coverage bins.
- `scripts/render_target_pose_ablation.py` — posed vs unposed context cam_token A/B
  (LagerNVS eval default is posed; renders sharper/more consistent with context poses).
- `scripts/render_scene_stitched.py` — per-scene stitched NVS over all segments using
  coverage-aware out-of-segment context (frustum_cover on the longer side) + posed
  cam_token; also per-segment context-selection visualization.
- `geo_cover_out_of_seg` / `geo_posed` config options. `frustum_cover_select` gains an
  `allowed=` param (restrict candidates to a frame subset, e.g. out-of-segment). Dataset
  restricts frustum_cover to the longer out-of-segment side and emits geo-view geometry
  (`geo_c2w`/`geo_fxfycxcy`/`geo_hw`); `geo_encoder.build_cam_token` builds the lagernvs
  posed 11-dim cam_token (1.35*max norm, extri_intri_to_pose_encoding). All default OFF
  (existing unposed/in-segment behavior unchanged).
- `config_geo_camscale.py` — geo(lagernvs) posed + out-of-segment frustum_cover +
  `scale_mode=context_longer` (full-sequence, not AR).
- `geo_shuffle_order` config option (permute selected geo context view order each access;
  posed -> VGGT reference view changes). `config_geo_camscale_shuf.py` = shuffle ON.
- Per-SEGMENT coverage blacklist: `coverage_blacklist_path` config + dataset
  `_read_coverage_blacklist` skipping (scene, segment) pairs (NOT whole scenes).
  `scripts/dump_coverage_selk.py` (K=6 frustum_cover selected coverage — matches actual
  geo input) + `data/blacklist_selk_tau0.7.csv` (13,166/40,059 segments removed at selK<0.7).
- `scripts/render_borderline.py` — render specific (scene,segment) pairs (e.g. tau
  borderline survivors) to judge a filtering threshold.
- `frustum_cover_select` gains `ball_center=` (explicit coverage-ball center, e.g. frame s
  omnidirectional) and `look_centroid=` (target-free look direction). `config_geo_ballS.py`
  (ball@frame-s selection, no shuffle) + `data/blacklist_selk_ballS_tau0.7.csv`.
- `scripts/select_compare.py`, `scripts/render_select_headtohead.py` — ours vs I3DM-style
  (MC FOV-overlap) context selection (geometry + rendered A/B).
- `scripts/render_ar_continuity.py` — LagerNVS chunk-wise AR continuity render: rolling
  causal memory (source ∪ generated-so-far) vs static source-only, side by side.
- `scripts/render_ar_viz.py` — per-chunk anchor & context (source vs gen-past) top-down viz.
- `scripts/render_ar_anchor.py` — coordinate-anchor ablation: force LagerNVS view0 to the
  current chunk's first camera (A: re-anchor) vs the segment's first camera (B: fixed frame).
  Also emits a per-chunk context viz.
- `scripts/render_ar_scene.py` — whole-scene AR: chunk = a full 49-frame segment, render ALL
  segments of a scene in order with rolling causal memory (previous segments), concat to one
  long video; A(re-anchor)/B(fixed) anchor ablation + per-segment context viz.
- `scripts/render_scene_longer.py` — whole-scene NON-AR: per segment, context = K views from
  the full LONGER out-of-seg side only (ball@s), render all segments and concat; + per-segment
  context viz (shows the longer side auto-flipping after↔before across the scene).
- `scripts/render_scene_longer_anchor.py` — same longer out-of-seg context + the view0 anchor
  ablation A(view0=seg start) vs B(view0=scene start fixed); [GT|A|B] whole-scene concat.
- `exp_results/` — organized experiment artifacts (coverage figures/dumps/blacklists,
  render montages+videos, README index).
