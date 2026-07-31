# Changelog (latentcam)

All notable changes to the latentcam sub-project. Follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **`geo_worldtraj_decoupled.yaml` (신규 experiment) — `geo_worldtraj_camembed`에서 camera
  embedding만 뺀 ablation.** resolved config 기준 차이는 `geo_cam_embed: relfirst → null`
  단 하나(그 외 전부 동일, `diff`로 확인). 3-way:
  `geo_worldtraj`(first_view_target_s **true** / cam_embed off / geo_proj 768) vs
  **THIS**(false / off / 768) vs `geo_worldtraj_camembed`(false / relfirst / 896).
  즉 `THIS` vs `geo_worldtraj` = context view0를 target 첫 프레임에 고정하던 link를 끊은 비용,
  `camembed` vs `THIS` = 11-d pose가 그 비용을 얼마나 되사는지.
  **캐시 재생성 없음** — `geo_cam_embed`는 `_sample_geo_frustum_cover`를 건드리지 않고
  `geo_cam_param` 텐서만 추가하므로(`dataset_dl3dv.py:917-919`) context view 선택이 camembed run과
  bit-identical. 캐시 강제 off 상태의 실제 `__getitem__` 300 세그먼트 검증: cached `geo_idxs`
  300/300 일치, mismatch 0, unusable 0, `view0 == frame s` 0/300.
  `first_cam_not_included/`를 그대로 재사용한다.
- **`ctx_side_135max` divisor** (`main/dump_avgscale_render.py`) — windowing 없이 segment 밖
  **긴 쪽 전체**에 대해 `1.35·max‖c − c_side0‖`. `ctx_longer_135max`(D2, 윈도 평균)의 대조군.
- **`context_longer` divisor** (`main/dump_avgscale_render.py`) — 이미 구현되어 있는 동명의
  `scale_mode`(`dataset_dl3dv._cam_dist_mean_context`)를 렌더 테스트에도 추가.
  `ctx_longer_135max`와 **같은 윈도**를 쓰되 집계만 `1.35·max` 대신 `mean‖c − c_win0‖`.
  이걸로 렌더 스윕이 latentcam이 지금 당장 학습 가능한 leak-free divisor를 전부 덮는다.
- **`scripts/data/norm_divisor_compare.py` (신규)** — divisor 후보들을 **canonicalization 관점**에서
  비교. 1200 segment(전부 서로 다른 scene)에 대해 `m = max‖c_t − c_s‖ / D`(diffusion이 회귀해야
  하는 정규화된 도달거리)와 `r = D_lagernvs / D`(LagerNVS native 단위 대비 편차)를 계산하고,
  `sd(log10 m)` / `p95/p05` / `m>1` / `m<0.1` / `corr(log D, log maxd)`를 표로 낸다.
  출력: `results/compare/norm_divisor_compare/{stats.json, summary.md, per_segment.csv,
  _divisors.png}`. env: `N` / `CACHE_EXP` / `SPLIT` / `SEED`.
- **`scripts/data/norm_divisor_compare.py`: `ONE_PER_SCENE` / `WORKERS` / `OUT_NAME`** —
  **기본값(`ONE_PER_SCENE=1, WORKERS=1, N=1200`)은 기존 동작 그대로** (1200 segment 결과 재현 확인).
  `ONE_PER_SCENE=0`이면 scene당 1개로 줄이지 않고 split의 **전 segment**(39,830 / 6,097 scene)를
  집계하므로 scene이 실제 학습에서 보이는 빈도대로 가중된다. `WORKERS>1`은 그 경로를 fork pool로
  병렬화(`_sample_geo_frustum_cover`가 결정론적 greedy라 샤딩해도 결과 불변; 32 workers 기준
  전수 ~2분). `N=0` = 무제한. `OUT_NAME`으로 출력 폴더 분리.
  `stats.json`에 `n_scenes` / `one_per_scene` / `maxd_log10_sd` 추가.
  결과: `results/compare/norm_divisor_compare_all/`.
- **`scripts/vae/vae_divisor_recon.py` (신규)** — divisor를 바꾸면 **frozen camera VAE**가 아직
  멀쩡한지 확인. 400 segment에 대해 `dataset_dl3dv.__getitem__`과 동일하게 `cam_param`을 divisor별로
  재구성해 encode→decode 하고 `in_trans_std` / `lat_std_{raw,scaled}` / `vae_scale_for_1` /
  `rec_trans_{norm,world}` / `rec_rot6d`를 보고한다. `rec_trans_world`가 divisor 간 비교 가능한
  값이고, `vae_scale_for_1`은 그 divisor로 갈아탈 때 써야 할 `vae_latent_scale`이다.
  출력: `results/compare/norm_divisor_compare/vae_recon.json`.
- **`tools/lagernvs/render_static_probe.py` (신규; gitignore된 vendored 트리라 커밋에는 없음,
  작업 트리에만 존재)** — "context 카메라가 정지하면 렌더가 되는가"를 divisor 문제와 분리해서 측정.
  healthy segment의 `render_inputs.pt`를 재사용해 target 궤적은 그대로 두고 context만 4가지로
  다시 만들어 렌더한다: `ctrl6`(실제 6-view) / `single`(ctx0 하나, `num_cond_views=1` +
  `split='test'` → `build_cam_cond`가 `camera_scale=0, world_points_scale=1` 토큰 발행) /
  `dup6_consistent`(ctx0 이미지·포즈를 6배 = 물리적으로 정지한 카메라) /
  `dup6_contradict`(서로 다른 6장 이미지에 ctx0 포즈만 강제 = 포즈가 깨진 scene 재현).
  `single` vs `dup6_consistent`가 OOD 토큰 쌍 `(0,0)`의 비용을, `dup6_consistent` vs
  `dup6_contradict`가 pose-image 모순의 비용을 각각 분리한다.
  env: `RD_ROOT` / `RD_CKPT` / `RD_SIZE` / `RD_NSEG` / `RD_OUT` / `RD_CONDS`(렌더할 조건 부분집합) /
  `RD_VIDEO=1`(조건별 mp4 + `GT|cond1|cond2|...` 가로 concat mp4 — 프레임 그리드로는 안 보이는
  시간축 drift 확인용).
  출력: `results/compare/static_ctx_probe/{metrics.json, <seg>.png}`,
  영상은 `results/compare/static_ctx_video/`.
- **`scripts/data/scan_duplicate_poses.py` (신규)** — COLMAP 등록 실패로 **다수 프레임이 한 좌표에
  박혀 있는** scene을 전수 검출. scene마다 camera center를 `TOL`(기본 1e-4, scene extent 상대) 격자에
  버킷팅해서 최대 클러스터의 비율 `dup_frac`과 그 안의 최장 **연속** 구간 `dup_run`을 낸다.
  `dup_frac >= FRAC`(기본 0.10)이면 flag. env: `TOL` / `FRAC` / `WORKERS`(기본 32) / `CACHE_EXP` /
  `OUT_NAME`. 출력: `results/compare/scan_duplicate_poses/{per_scene.csv, flagged.csv,
  blacklist_rows.csv, stats.json}` — `blacklist_rows.csv`는 `<dl3dv_root>/blacklist.csv`에 그대로
  append 가능한 형식.
  train 6,097 scene 결과: **flagged 2개**, `dup_frac` p50 0.0030 / p99 0.0061 / p99.9 0.0120 /
  max 0.5815. 1·2위(0.582, 0.257)와 3위(0.034) 사이가 7.5배로 벌어져 경계가 깨끗하다.
- **`scripts/render/compare_norm_video.py`: `--layout row`, `--modes`** — 모든 mode를 가로 한 줄로
  붙인 비교 영상, 그리고 렌더할 mode 부분집합 선택. `ORDER`에 `ctx_side_135max` 추가.
- **정규화 ablation 렌더 파이프라인 (LagerNVS 자체 normalization 비활성화 + PSNR + 비교 영상)** —
  latentcam의 후보 divisor들을 LagerNVS 렌더로 검증하는 3단계. 기본값은 전부 기존 동작 유지.
  - `main/dump_avgscale_render.py`: `OUT_NAME`(출력 폴더), `ONE_PER_SCENE=1`(기본 0 = 기존처럼
    첫 N개 sample 그대로; 1이면 서로 다른 N개 scene), `tgt_image_paths`(PSNR용 GT 프레임),
    `scales` dict를 dump에 추가. `scales`는 네 divisor —
    `lagernvs`(=LagerNVS 자체 `1.35·max‖ctx center − ctx0‖`, control),
    `avg_scale`(현재 latentcam 학습값), `maxd_seg`(target 유래 → leak, 상한),
    `ctx_longer_135max`(segment 밖 긴 쪽을 `num_frames` 윈도로 나눠 `1.35·max`의 평균, leak 없음).
  - `tools/lagernvs/render_avgscale.py` (**gitignore된 vendored 트리라 커밋에는 없음, 작업 트리에만
    존재**): `RD_ROOT` / `RD_MODES`(기본 `avg_scale` → 출력 파일명까지
    이전과 동일)로 같은 segment를 여러 divisor로 렌더하고, `tgt_image_paths`가 있으면
    PSNR/SSIM/LPIPS를 `metrics.json`(mode별 평균 + `psnr_per_frame`)에 기록. 2개 이상 mode면
    `render_norm_compare.png`(행=mode+GT)도 생성.
  - `scripts/render/compare_norm_video.py` (신규): 위 mp4들을 하나로 합성. `--layout grid`(기본,
    2×3: GT / lagernvs / avg_scale / maxd_seg / ctx_longer_135max / PSNR 곡선+프레임 커서),
    `--layout pair`(mode별 `GT | render` width concat). 타일마다 divisor·해당 프레임 PSNR·평균
    PSNR 라벨.
- **`geo_cam_embed` — per-context-view camera embedding concatenated onto the geo tokens**
  (default **`null` = off**, geo conditioning byte-identical to before). With `'relfirst'` the
  dataset builds, for each context view `v`, an 11-d pose relative to the TARGET segment's FIRST
  camera `s` — `rel = w2c_v @ inv(w2c_s)`, `trans /= norm_scale`, parametrized exactly like
  `cam_param` (`rot6d = rel[:3,0] ++ rel[:3,1]`, `trans`, `fx/2cx`, `fy/2cy`) — and the training
  loop broadcasts it over that view's patch tokens, so `geo_emb` goes `(B, V·P, 768)` →
  `(B, V·P, 779)`. The model (`CameraDiffusionModel._lift_geo_cam`) splits the trailing 11 dims
  off, lifts them through a **trainable** MLP (`11 → geo_cam_embed_dim → geo_cam_embed_dim`,
  default 128) and re-concatenates, so `geo_proj` sees `768 + 128 = 896`. Carrying the raw dims
  inside `geo_emb` means no call-site signature changed and the frozen LagerNVS half stays
  cacheable. Translations use the target's `norm_scale`, i.e. the same units as the trajectory
  being generated; context intrinsics stay **raw** (under `intr_norm: rel` the target's own intr
  channels are ~[1,1], so raw is the only way context FoV reaches the model).
  Verified: `geo_proj.in_features` 896 vs baseline 768, params 64.905M vs 64.821M (Δ = 83,584 =
  exactly the MLP + wider `geo_proj`), broadcast checked per view, forward → `(B, 13, 64)`.
- **`conf/experiment/geo_worldtraj_camembed.yaml`** — the ablation that decouples the geo context
  from the camera being generated: `geo_first_view_target_s: false` (no target-segment frame is a
  context view at all, so all `geo_cover_k = 6` views are retrieved) +
  `geo_cover_subtract_first: false`, keeping `geo_cover_out_of_seg` / `geo_anchor_first_frame` /
  `geo_posed` and LagerNVS's own `1.35·max‖center‖` context normalization. `geo_cam_embed:
  relfirst` is what replaces the lost frame alignment. Its cache resolves to
  `first_cam_not_included/`, disjoint from `geo_worldtraj`'s `first_cam_included/`.
- **geo latent cache v2 format** — `cache_geo_embeddings.py` now writes
  `{'emb': (M,768) fp16, 'geo_idxs': (V,) int16}` instead of a bare tensor, and the dataset emits
  `geo_idxs` on the geo path. Storing the selected context frames lets a cache hit rebuild
  `geo_cam_param` without redoing the greedy coverage search. **Both formats are read**: v1 bare
  tensors still work for runs with `geo_cam_embed` off; a v1 file under a `geo_cam_embed` run
  falls back to the on-the-fly path rather than guessing. Round-trip verified bit-exact
  (`emb maxdiff 0.000000`, `geo_cam_param maxdiff 0.00000000`, 3/3) and cache-ON decodes no images.
- **`geo_latent_cache_dir` + a cache-read path in `dataset_dl3dv` / `train_latent_cam_dm`** — the
  precomputed frozen geo latents built by `cache_geo_embeddings.py` were being written but never
  read (nothing in the training code loaded them; `load_saved_pc_embeds` is the point-cloud flag
  and is unrelated). Now `cfg.geo_latent_cache_dir` (default **`null` = off**, i.e. the original
  per-step LagerNVS forward is completely unchanged) makes `CamDataset.__getitem__` read
  `<dir>/<first_cam_included|first_cam_not_included>/<iK>/<data_name>.pt` and return it as
  `geo_emb`, short-circuiting context-view selection and image decoding entirely; the trainer's
  new `geo_emb_from_cache()` rebuilds the all-ones mask (lagernvs marks every token valid,
  `geo_encoder.py:139`) and skips the encoder at both the train and validation call sites.
  A segment missing from the cache falls through to the on-the-fly path, so a partial cache is
  safe. Verified **bit-exact**: cache-ON vs cache-OFF `geo_emb` for the same segments gives
  `maxdiff 0.000000`, 3/3, and cache-ON carries no `images` key.
  Guard: this is only valid because `GeoEncoder.proj` is `nn.Identity` when the lagernvs native
  dim 768 equals `geo_latent_dim` (confirmed at runtime, `proj: Identity`). With
  `geo_latent_dim != 768` proj is a **trainable** `Linear` whose output must not be frozen into a
  file, so the dataset refuses the cache and prints why.
  Speed, measured on the full corpus (39830 samples, `bs8`, 1 GPU): cache **ON 4.16–4.22 it/s**
  (~17.8 min/epoch) vs **OFF 2.85 s/it = 0.351 it/s** (~3h33m/epoch) — **~12×**. A smoke-scale
  `max_scenes=40` comparison had shown *no* difference (ON 1.82/1.81 vs OFF 1.87/1.85 it/s); that
  measurement is invalid because 40 scenes' context images all fit in the OS page cache.
- **`conf/experiment/geo_worldtraj.yaml`: `scale_mode` `cam_dist_mean` → `avg_scale`** — matches
  the `textonly_savedscale_bs8/bs32` runs so geo-vs-text-only differ only in the `geo_*` keys.
  The geo latent cache stays valid: the geo path reads raw `self.extrinsics_list` for both view
  selection and the posed `cam_token` (`dataset_dl3dv.py:860-870`) and `geo_lagernvs_skip_ctx_norm`
  is false, so `build_cam_token` gets `override_scale=None` (`train_latent_cam_dm.py:74-76`) —
  `geo_emb` never sees `scale_mode`, no re-caching needed. Cost: with `avg_scale` + `intr_norm rel`
  this VAE measures latent std 0.44696, so `vae_latent_scale 0.96032625` gives a diffusion input
  std of 0.4654 (~2.15× small) instead of `cam_dist_mean`'s near-unit 1.034 — deliberately the
  same off-unit input as the text-only arms.
- **`conf/experiment/geo_worldtraj.yaml`: `geo_latent_cache_dir` enabled** — points at
  `/data1/cympyc1785/data/DL3DV/latent_cache`, whose `first_cam_included/` tree was built from
  exactly this experiment's context selection (`frustum_cover` + `out_of_seg` + `subtract_first`
  + `anchor_first` + `posed`). All 39830 segments present (missing 0 / extra 0, 266 GB). Set to
  `null` to force the original per-step LagerNVS forward.
- **`cache_geo_embeddings.py`: per-`{i}K` output layout, multi-batch runs, resumability, ETA** —
  the DL3DV scene root is now split into per-1000 batch dirs, so the cache mirrors it. New
  `CACHE_LAYOUT` selects `batch` (default, `<OUT>/<first_cam_*>/<iK>/<name>.pt`, with the
  `first_cam_*` level derived from `geo_first_view_target_s` because that flag changes which
  context views are encoded) or `flat` (**the original pilot layout, byte-for-byte unchanged**).
  `CACHE_BATCH` now takes one batch, a comma-separated list, or `all`; batches run in order,
  each resumable (an already-present file is skipped without a forward pass). Progress lines now
  report seg/s and ETA, and `CACHE_BS` exposes the loader batch size.
- **`conf/experiment/vae_dl3dv_1_7k.yaml`** — camera-VAE re-fit on the full DL3DV corpus the
  diffusion runs actually use (`meta.csv`, no coverage blacklist → **39830 segments / 6097
  scenes**), replacing the 1K-only fit behind the active default ckpt
  `vae_20260202_065659_400.pth` (SCVideo fit that on DL3DV 1K with one 49-frame window per scene,
  ~1000 samples). Deliberately a **drop-in**: `cam_dim: 32` + `scale_mode: avg_scale` +
  `intr_norm: raw`, so the resulting `state_dict` is key- and shape-identical to the old ckpt
  (verified) and swapping it changes only the VAE's training corpus. `batch_size 64`, `lr 1e-4`,
  `60 epochs`, `vae_beta 0.001`. Its `train/latent_std` (also written to
  `my_checkpoints/vae_dl3dv_1_7k/latent_std.txt`) becomes the matching `vae_latent_scale` —
  `0.4467666` belongs to the 1K-only fit and must not be reused with it.
- **`scripts/vae/vae_scale_matrix.py`: `vae_dl3dv_1_7k` added to the `CKPTS` matrix** (as
  `../my_checkpoints/vae_dl3dv_1_7k/last.pth`, 32-d) so the new fit is measured on exactly the
  same basis as the two reference ckpts. First measurement (1264 segments, `avg_scale`): latent
  std **0.60396** with `raw` (→ `vae_latent_scale` ≈ 0.60–0.62, and `0.4467666` would give input
  std 1.3518), recon L1 rot 0.00564 / trans 0.00900 / intr 0.00485, vs the 1K-only ckpt's 0.46312
  and 0.00466 / 0.00495 / 0.00293. The `raw` ≪ `rel` gap on `intr_L1` (0.00485 vs 0.25884) is
  preserved, i.e. the new ckpt is still a `raw` ckpt and still a strict drop-in.

- **`conf/experiment/textonly_savedscale_bs8.yaml` + `textonly_savedscale_bs32.yaml`** — the two
  arms of the batch-size comparison as self-contained configs instead of `experiment=
  textonly_savedscale batch_size=N` CLI overrides. Byte-identical apart from `exp_name` +
  `batch_size` (verified by resolving both and diffing all 19 relevant keys), and `exp_name`
  carries the batch size because the wandb run name is `<timestamp>_<exp_name>`
  (`train_latent_cam_dm.py:183` + `:194`), so each arm lands as its own wandb run without relying
  on the launcher remembering an override.
- **First `vae_dl3dv_1_7k` fit completed** (60 epochs, 39830 segments, GPU 2):
  `epoch 59 loss=0.01186 latent_std=0.635126`; ckpt at `my_checkpoints/vae_dl3dv_1_7k/last.pth`.
  Measured against the 1K-only ckpt on one basis (1264 segments, `avg_scale`): latent std
  **0.60943** `raw` (vs 0.46312), recon L1 rot **0.00562** / trans **0.00953** / intr **0.00433**
  (vs 0.00466 / 0.00495 / 0.00293), and `raw` ≪ `rel` on `intr_L1` (0.00433 vs 0.25666) so it is
  still a `raw` ckpt and a strict drop-in. Its matching `vae_latent_scale` is **0.635126** (the
  training-corpus std; `0.4467666` with this ckpt would give input std 1.3641). **Not adopted** —
  `config.yaml` still points at `vae_20260202_065659_400` + `0.4467666`; no diffusion run has used
  the new ckpt yet, so the recon-vs-corpus-coverage trade is unevaluated.

### Changed
- **`scripts/render/compare_textonly_vs_worldtraj.py` 를 2-way 하드코딩에서 N-way 로 일반화.**
  `--run LABEL=DIR`(반복 가능) / `--ref` / `--out` / `--ctx` / `--contact` / `--per-target` 추가.
  **인자 없이 실행하면 기존 2-way 동작(text-only vs worldtraj, 같은 출력 경로)이 그대로 재현된다** —
  기존 호출부는 수정 불필요. 출력물에 `_contact.png`(앞 N개 target 컨택트 시트)가 추가되었고,
  `_summary.png`는 run 수에 맞춰 `첫 run 대비 산점도 (N-1)개`로 늘어나며 `_scores.csv`도 run 수만큼
  컬럼이 붙는다. 공통 target은 모든 run이 예측을 남긴 것만 교집합으로 취한다.
  이걸로 만든 결과: `results/topdown_3way_textonly_worldtraj_camembed/`
  (text-only vs geo_worldtraj vs geo_worldtraj_camembed, 공통 target 160개).
  **주의**: 세 run 모두 학습 중이라 `test/`는 epoch이 서로 다른 스냅샷이다 —
  숫자 해석 시 그 폴더의 `README.md` 경고를 먼저 읽을 것.
- **`geo_anchor_first_frame` → `geo_cover_centered_at_s` 로 rename** (동작 변화 없음).
  옛 이름은 "첫 프레임이 anchor **view**로 들어간다"로 읽혔는데, 그건 `geo_first_view_target_s`가
  하는 일이다. 이 플래그는 `frustum_cover` greedy 탐색의 `anchor`/`ball_center`/`seg_scale`만
  정할 뿐 encoder context에 view를 추가하지 않는다 — 실측: `geo_first_view_target_s: false`이면
  `geo_cover_centered_at_s: true`여도 `view0 == frame s`가 **0/500**, `[s,e)` 안의 view가 **0/500**.
  (`true`이면 500/500.) rename 후 재검증에서도 세 config 전부 동작 동일.
  **옛 키는 deprecated로 계속 인식**된다(`dataset_dl3dv.py`에서 새 키가 없을 때만 fallback +
  경고 출력)므로 rename 이전 config/CLI override도 그대로 돌아간다.
  적용 범위: `main/conf/config.yaml`, `main/config.py`, `main/dataset_dl3dv.py`,
  `main/conf/experiment/*.yaml` 14개.
- **`meta_csv: meta_worldtraj.csv` pinned in `textonly_savedscale.yaml` and
  `vae_dl3dv_1_7k.yaml`** (was the `meta.csv` default), matching every other worldtraj-scoped
  experiment. `coverage_blacklist_path` stays `null`.
  **This changes no data** — measured, not assumed: both CSVs index to exactly 39830 samples /
  6097 scenes, and the scene sets and all 39830 sample IDs are identical (verified by diffing the
  two cached indexes). `meta.csv` lists 8048 scenes (1K–11K) but 8K–11K have no scene dirs on
  disk, and its 1K–7K portion (6132) exceeds `meta_worldtraj.csv` (6098) by 34 scenes that have
  `prompts.json` + `transforms.json` but no valid 49-frame segment, so they were dropped either
  way. The value of the switch is that the corpus is now stated explicitly in the config.
  Also measured: the scene-level `blacklist.csv` (295 scenes) is a no-op for both CSVs
  (`∩ = 0` for each) and `dataset_dl3dv._load_index` re-applies it unconditionally regardless, so
  no flag is needed to "enable" it.
- **`vae_latent_scale` set to SCVideo's verbatim `0.4467666`** (was our measured `0.46312`), keeping
  the DL3DV-only `config.py` triple otherwise unchanged (`vae_20260202_065659_400.pth`,
  `cam_dim: 32`, `intr_norm: raw`, `scale_mode: avg_scale`, `clatr_epoch109_dl3dv_seg_2.ckpt`).
  Explicit user decision to use SCVideo's constant rather than our re-measurement. Consequence,
  measured not estimated: the latent std of this triple on our 1264 DL3DV segments is 0.46312, so
  the diffusion input std is **1.0366** instead of 1.0000 (train/infer do
  `encode(traj) / vae_latent_scale`) — a 3.7% overshoot, from our segment definitions differing
  from SCVideo's one-49-frame-window-per-scene sampling.
  `config_large.py`'s line (`0.96032625` + `vae_20260302_300.pth` + `cam_dim 64` + `intr_norm: rel`
  + `clatr_epoch139_large.ckpt`) was evaluated and rejected in the same session. Recorded for
  future reference, since it is a valid alternative: that ckpt loads strict at 64-dim, gives
  `latent (16,13,64)` with the best recon of any cell (L1 rot 0.00504 / trans 0.00301 /
  intr 0.00222), `CameraDiffusionModel(cam_dim=64)` = 64.82M params; `config_large`'s dataset
  really does use saved `avg_scale` (`data/dataset_large.py:295-304`) + frame-0-relative
  intrinsics (line 313), so `avg_scale` + `rel` is its exact pipeline; 31 of its keys already match
  ours, the 7 that differ being `batch_size` 32, `save_epoch` 25, `num_thread` 4,
  `sample_data` 33980, and the three that make it a **point-cloud-conditioned,
  attention-supervised** run rather than text-only (`load_points`/`load_saved_pc_embeds` True,
  `model_type: baseline_attn_sup`). Its blocker is the same class of problem as above but larger:
  `0.96032625` was fit over SCVideo's MIXED corpus (DL3DV + DynamicVerse + dynpose-100k), while
  DL3DV-only measures 0.44696 → input std **0.4654**. `clatr_epoch139_large.ckpt` is a verified
  drop-in (both CLaTr ckpts: 260-key state_dict, no shape mismatch, 191,546,118 B; the input
  standardization lives in `evaluate/CLaTr/configs/dataset/standardization/0120.yaml`, not in the
  ckpt) but is trained on a different corpus, so its FD/PRDC/CLaTr-score would not be comparable
  with the `epoch109_dl3dv_seg_2` history we already have.
- **`cam_dim` + `intr_norm` pinned in every experiment that overrides `vae_ckpt_path`, plus the
  VAE-training experiments**, so each ckpt keeps its own latent dim and intrinsics convention
  regardless of the global default. Both values were set to what the current/pre-`intr_norm` code
  resolved to, i.e. **no behavior change** — the point is that these configs no longer silently
  depend on `config.yaml`'s defaults:
  * `cam_dim: 64` in `ar.yaml`, `ar_smoke.yaml`, `rolling.yaml`, `rolling_smoke.yaml`
    (CamVLA `causal_vae_v1_240.pth`), `textonly_align.yaml`, `geo_worldtraj_align.yaml`
    (`my_checkpoints/vae_worldtraj/last.pth`), and the VAE-training configs `vae_worldtraj.yaml`,
    `vae_dl3dv.yaml`, `vae_dl3dv_smoke.yaml`, `vae_dl3dv_avgscale.yaml`. All three non-SCVideo
    ckpts were inspected and are 64-dim (`encoder.to_mu.weight (64,64,1)`), so with the default now
    32 they would have hit the exact `size mismatch for encoder.to_mu.weight` crash recorded in
    FIX.log 2026-07-18.
  * `intr_norm: rel` in the four CamVLA configs (no `scale_mode` override → `avg_scale` → legacy
    `auto` == `rel`) and `vae_dl3dv_avgscale.yaml`; `intr_norm: raw` in `textonly_align.yaml`,
    `geo_worldtraj_align.yaml`, `vae_worldtraj.yaml` (fit under `first_farthest_135` → legacy
    `auto` == `raw`), `vae_dl3dv.yaml`, `vae_dl3dv_smoke.yaml` (`geo_lagernvs` → `raw`), and
    `textonly_savedscale.yaml`.
  Experiments that use the DEFAULT VAE and only override `scale_mode` (`geo_*`, `textonly`,
  `textonly_camscale*`, `smoke_*`) are intentionally left unpinned, so they inherit whatever
  `config.yaml` sets.
  *(Superseded later in this same `[Unreleased]` block — see "reverted to `config_large.py`'s VAE
  + CLaTr line" below. `textonly_savedscale.yaml` is now `rel`/64, not `raw`, and the unpinned
  experiments now inherit the 64-dim `vae_20260302_300` + `rel` default.)*
- **DL3DV root moved** `/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K` → `/data1/cympyc1785/data/DL3DV/scenes`
  (done by the user on disk). Updated every live reference: `main/conf/config.yaml` (`dl3dv_root`),
  `main/conf/experiment/geo_worldtraj_seglist.yaml` (`train_seg_list`/`test_seg_list`),
  `main/config.py`, `scripts/render/render_target_from_context.py`,
  `scripts/data/{make_latentcam_splits,filter_dl3dv,norm_camera_length_stats}.py`,
  `scripts/viewer/viser_val_cameras.py`, `scripts/context_select/{frustum_cover_sweep,
  vis_frustum_cover,visualize_covis_retrieval,viz_start_coverage_retrieval,vis_frustum_cover_multi,
  select_compare,covis_compare}.py`, `scripts/coverage/{dump_coverage_selk,viz_coverage,
  dump_coverage,analyze_geo_retrieval_coverage,blacklist_by_coverage}.py`. Also dropped the now-wrong
  "DL3DV-960" wording from `main/dataset_dl3dv.py` prints/docstring, `main/train_latent_cam_dm.py`,
  `main/train_vae_dl3dv.py`. `main/config.py`'s legacy `DL3DV_DATA_PATH` (used only by the old
  `dataset_seg.py`) was left as-is.
- **default VAE triple is now SCVideo's DL3DV-only setting** (`main/conf/config.yaml`,
  `main/conf/experiment/textonly_savedscale.yaml`): `vae_ckpt_path` → `vae_20260202_065659_400.pth`,
  `cam_dim: 32`, `intr_norm: raw`, `vae_latent_scale: 0.46312`, with `scale_mode: avg_scale`. This is
  what SCVideo's `main/config.py` uses (it pairs with `core_pkg/models/vae_intr.py`, i.e.
  `vae_intr_large` with `latent_dim=32` — the two modules are identical apart from that default and
  an extra `encode_sample`); `config_large.py`/`config_vae.py` (`0.96032625` + `20260302/300.pth`,
  64-dim) are the multi-dataset (DL3DV+DynamicVerse+dynpose) line and label the 32-dim pair `# old`.
  `vae_latent_scale` uses OUR measured std 0.46312 rather than SCVideo's 0.4467666, since
  `train_latent_cam_dm.py` divides the latent by it and our segment definitions differ (0.4467666
  verbatim would give diffusion-input std 1.037). Verified end to end: strict VAE load OK,
  `cam_param (16,49,11)` → `latent (16,13,32)`, diffusion-input std **1.0000** over 1264 samples,
  roundtrip L1 0.004424, `CameraDiffusionModel(cam_dim=32)` 64.79M params forward OK.
  Full matrix over 1264 DL3DV samples (`scripts/vae/vae_scale_matrix.py`; latent std | that std
  divided by each config constant | recon L1 rot/trans/intr):
  ```
  ckpt                       dim scale_mode          intr  lat.std /0.96033 /0.44677      rot    trans  intr_L1
  vae_20260302_300            64 avg_scale           rel   0.44696   0.4654   1.0004  0.00504  0.00301  0.00222
  vae_20260302_300            64 avg_scale           raw   0.43609   0.4541   0.9761  0.00695  0.00851  0.36288
  vae_20260202_065659_400     32 avg_scale           rel   0.63741   0.6637   1.4267  0.00648  0.01081  0.26291
  vae_20260202_065659_400     32 avg_scale           raw   0.46312   0.4822   1.0366  0.00466  0.00495  0.00293  <- active
  vae_20260302_300            64 cam_dist_mean       rel   0.99270   1.0337   2.2220  0.00792  0.01072  0.00400
  vae_20260302_300            64 cam_dist_mean       raw   0.98131   1.0219   2.1965  0.00895  0.01588  0.36148
  vae_20260202_065659_400     32 cam_dist_mean       rel   1.11188   1.1578   2.4887  0.01951  0.03242  0.25974
  vae_20260202_065659_400     32 cam_dist_mean       raw   1.00625   1.0478   2.2523  0.02013  0.03035  0.01340
  vae_20260302_300            64 first_farthest_135  rel   0.54116   0.5635   1.2113  0.00571  0.00457  0.00267
  vae_20260302_300            64 first_farthest_135  raw   0.52944   0.5513   1.1850  0.00718  0.01049  0.36245
  vae_20260202_065659_400     32 first_farthest_135  rel   0.70409   0.7332   1.5760  0.00840  0.01395  0.26203
  vae_20260202_065659_400     32 first_farthest_135  raw   0.54560   0.5681   1.2212  0.00691  0.00920  0.00462
  vae_20260302_300            64 context_longer      rel   1.06565   1.1097   2.3853  0.00804  0.01069  0.00399
  vae_20260302_300            64 context_longer      raw   1.05523   1.0988   2.3619  0.00910  0.01555  0.36159
  vae_20260202_065659_400     32 context_longer      rel   1.18505   1.2340   2.6525  0.01973  0.03262  0.26003
  vae_20260202_065659_400     32 context_longer      raw   1.08236   1.1271   2.4226  0.02123  0.03082  0.01273
  ```
  Reading it: the intrinsics convention is fixed by the CKPT (`20260302` → `rel` everywhere,
  `20260202` → `raw` everywhere; mismatching it costs ~100× on the intr channels), while
  `scale_mode` only rescales the translations and therefore the latent std. `0.96032625` belongs
  to `20260302` + `cam_dist_mean` + `rel` (0.99270); `0.4467666` to `20260202` + `avg_scale` +
  `raw` (0.46312). This SUPERSEDES the earlier version of this entry, which concluded that
  `0.4467666` belonged to the 64-dim ckpt — that 0.44696 agreement is a coincidence, and the
  conclusion was an artifact of the `intr_norm` bug below (the correct cell was not expressible).
  Note `cam_dim` must match the ckpt's `latent_dim` or the state_dict load hard-crashes on
  `encoder.to_mu.weight`.
  **Provenance of `vae_20260202_065659_400.pth`, from SCVideo's git history** (confirms the triple
  independently of our measurement). The run dir `20260202_065659` falls between SCVideo commits
  `cfd2cc0` (2026-02-02T04:22:35Z "Fix dataset") and `2f40e21` (06:59:23Z), so the launch tree is
  `cfd2cc0` + the `vae_intr` import edit that was committed the next day as `eb99764`
  (2026-02-03T02:25Z) — that edit is required, because at `cfd2cc0` `train_vae.py` still imported
  `vae_intr_large.CameraVAE()` whose default `latent_dim=64`, while the ckpt is 32-dim
  (`encoder.to_mu.weight (32,64,1)`). This also rules out reading the dir name as KST: 06:56:59 KST
  = 2026-02-01T21:56Z would precede `d4441b9` (02-02T02:12Z), the commit that first ADDED
  `vae_intr.py`. At that tree:
    * **train set = DL3DV `1K` only** — `config.py` had `dataset_dir = '.../DL3DV/scenes/1K'` (its
      only dataset key, unchanged from `d4441b9` through `a3713ae`), and `data/dataset.py:36`
      branches on `basename(dataset_path)[-1] == 'K'` → a flat `os.listdir('.../1K')` (1000 scenes
      locally). The `train_dataset_dir` 1K / `val_dataset_dir` 7K split only appears at `b27060c`
      (2026-02-23), three weeks after the ckpt; multi-chunk `build_dataset_dir_list()` is later
      still (`config_large.py`/`config_vae.py`).
    * **one sample per scene, frames 0-48** — `extrinsics[:num_frames]`, scenes with < 49 frames
      `continue`. No segment enumeration (that is our addition), which is why our 1264-segment
      measurement lands 3.7% off SCVideo's 0.4467666.
    * `avg_scale` + `raw` confirmed at the source: `normalize_camera_extrinsics_and_points`
      (`data_utils.py:20`) divides translations by `mean ||point - first cam||`, and
      `dataset.py:118-121` builds the intrinsics as `fx/(2cx), fy/(2cy)` with **no frame-0
      division** — the `rel` division (`dataset_large.py:313`) does not exist yet in this tree.
      The only later change to those lines (`eb99764`) is numpy → torch, semantics identical.
    * hyperparams at that tree: `batch_size 64`, `lr 1e-4`, `epochs 50000`, `save_epoch 10`,
      `vae_beta 1e-3`. `config.py` still pointed at the PREVIOUS ckpt (`20260123_074547/900.pth`,
      `vae_latent_scale 0.48848`); `0.4467666` + `20260202_065659/400.pth` were adopted at
      `a3713ae` (2026-02-05T06:42Z), with `dataset_dir` still `1K`.
- **`save_epoch: 1` → `10`** (`main/conf/config.yaml`), matching SCVideo's `config.py`. Only affects
  checkpoint-write frequency (per-epoch validation is unchanged).
- **`scripts/vae/vae_scale_matrix.py`** (new): measures latent std + per-component recon L1 over the
  whole (ckpt × scale_mode × intr_norm) grid, so a config constant like `0.4467666` can be traced
  back to the triple it was measured on. env `MAX_SCENES` / `MODES` / `CKPTS` / `META`.
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

- **Reverted to `config_large.py`'s VAE + CLaTr line as the global default** (explicit user
  decision, reversing the `0.4467666` / `vae_20260202_065659_400` entry above — that entry is
  superseded, not deleted, so the flip-flop stays legible). `config.yaml` **and** `config.py` now
  both carry: `vae_latent_scale: 0.96032625`, `vae_ckpt_path: checkpoints/vae_20260302_300.pth`,
  `clatr_ckpt_path: checkpoints/clatr_epoch139_large.ckpt`, `cam_dim: 64` (matching that ckpt's
  `latent_dim`), `num_cam: 13`, `intr_norm: rel`, `scale_mode: avg_scale`, `vae_beta: 0.001`.
  `avg_scale` + `rel` is genuinely `config_large`'s pipeline: `data/dataset_large.py:295-304`
  divides translations by the stored point-cloud `avg_scale` (our `saved_avg_scale`, an alias of
  `avg_scale`) and `:313` divides the intrinsics by frame 0.
  `textonly_savedscale.yaml` / `_bs8.yaml` / `_bs32.yaml` were switched from `intr_norm: raw` /
  unpinned `cam_dim` to pinned `rel` / `64`, and their comment blocks rewritten (they still
  claimed the 32-dim `0.4467666` triple and contained a "non-relative intrinsics / divided by
  frame 0" self-contradiction).
  Verified end-to-end on the resolved `textonly_savedscale_bs8` config: VAE strict-loads at
  `latent_dim 64`, `cam_param (1,49,11) → latent (1,13,64)`, `CameraDiffusionModel(cam_dim=64)`
  = **64.82M** params, recon L1 rot **0.00489** / trans **0.00282** / intr **0.00218** over 375
  segments (consistent with the 1264-segment 0.00504 / 0.00301 / 0.00222). Both arms smoke-tested
  to `EXIT=0` including the CLaTr eval path.
  Two caveats, measured not estimated:
  * `0.96032625` was fit over SCVideo's MIXED corpus (DL3DV + DynamicVerse + dynpose-100k). On
    DL3DV-only this triple's latent std is **0.44696** (1264 segments) / 0.43350 (375 segments),
    so the diffusion input std is **0.4654** / 0.4514, not 1.0 — latents reach the model ~2.15x
    too small. Kept verbatim per user instruction; `0.44696` is the value for exactly unit-std
    input on DL3DV-only. (`0.99270` for `cam_dist_mean` + `rel` is coincidentally near 0.96 and is
    **not** `config_large`'s normalization — it must not be used to justify the constant.)
  * `clatr_epoch139_large.ckpt` is an architecture-identical drop-in (260-key `state_dict`, no
    shape mismatch, both ckpts 191,546,118 B) but is trained on a different corpus, so **FD /
    PRDC / clatr_score from runs using it are not comparable with any number in
    `EXPERIMENTS.log` before 2026-07-30** — only epoch139-vs-epoch139.
  Audited all 34 experiment configs + the default by Hydra-composing each and comparing resolved
  `cam_dim` against the actual `encoder.to_mu.weight` dim of its resolved `vae_ckpt_path`:
  **0 mismatches**. Configs pinning their own ckpt are unaffected (`ar*`/`rolling*` → CamVLA
  `causal_vae_v1_240`, `textonly_align`/`geo_worldtraj_align` → `vae_worldtraj/last.pth`); the
  five VAE-training configs never load `vae_ckpt_path`. The unpinned `geo_*` / `textonly` /
  `textonly_camscale*` / `smoke_*` configs now inherit 64/`rel`/`vae_20260302_300`/`0.96032625`
  where they previously inherited 32/`raw`/`vae_20260202_065659_400`/`0.4467666` — intended, but
  it means results from those configs straddle two different VAEs.
### Fixed
- **`blacklist.csv`를 고쳐도 index 캐시가 낡은 채로 계속 쓰이던 문제** (`main/dataset_dl3dv.py`).
  `_load_index`의 캐시 키는 `meta_csv` / `num_frames` / `before_only` / `geo_cover_k` /
  `coverage_blacklist_path` / `max_scenes`만 담았는데, scene-level blacklist는 캐시를 **만들 때**
  적용된다(`:374`). 그래서 `blacklist.csv`에 scene을 추가해도 기존 캐시가 히트하면 그 scene이 계속
  학습에 들어갔다 — 조용히 틀리는 종류의 버그. `_blacklist_fingerprint()`(sha1 앞 8자리)를 추가해
  키에 `__bl<hash>`를 붙였다. 파일이 없으면 `none`. 내용이 안 바뀌면 키도 그대로라
  **기존 캐시 재사용 동작은 유지**되고, 편집하면 자동으로 재빌드된다(6,098 scene 스캔 ~37초).
- **DL3DV blacklist에 `duplicate_camera_centers` 2개 scene 추가** (데이터 파일
  `<dl3dv_root>/blacklist.csv`, 레포 밖. 백업: `blacklist.csv.bak_20260801`).
  `4K/50eb3c0d…8d5f`(dup_frac 0.582, 연속 142프레임 정지, N=368),
  `3K/b7da67fc…b1da`(0.257, 62프레임, N=331). 둘 다 COLMAP 등록이 끊겨 프레임 과반이 한 좌표에
  박혀 있는데 영상은 멀쩡히 움직인다 → context baseline이 0이라 LagerNVS 렌더가 어떤 divisor로도
  PSNR ~14.4에 갇히고(`results/norm_degenerate_check/`), `geo_lagernvs` divisor는 1.7e-5로 붕괴해
  정규화된 도달거리 `m`이 567,215까지 튄다. index 재빌드 결과 **39,830 → 39,817 sample /
  6,097 → 6,095 scene**.
- **동시 학습 간 CLaTr `lightning_logs` 버전 충돌로 `clatr_score` 1회 실패**
  (`main/evaluate/CLaTr/src/extraction.py`). `L.Trainer(...)`에 `logger` 인자가 없어 기본
  `TensorBoardLogger`가 붙는데, 이 로거는 `lightning_logs/`를 스캔해 다음 `version_<N>`을
  정한다 → 여러 학습이 같은 cwd에서 clatr eval을 동시에 띄우면 두 프로세스가 같은 N을 골라
  하나가 `FileExistsError`로 죽는다 (train1 @ `version_1113`). 해당 epoch의 clatr 지표만
  누락되고 학습 루프는 중단 없이 계속됐다. `logger=False`로 수정 — `trainer.predict`는 로깅을
  하지 않으므로 지표 값은 불변이고, 빈 version 디렉토리(1119개, 14M) 누적도 멈춘다.
- **cam_param's intrinsics convention was coupled to `scale_mode`, feeding the VAE the wrong
  encoding** (`main/dataset_dl3dv.py`, `main/conf/config.yaml`, `main/config.py`). New option
  `intr_norm: 'auto' | 'rel' | 'raw'` — `raw` = `fx/2cx, fy/2cy` (0.448, 0.796 for DL3DV),
  `rel` = the same divided by frame 0 (exactly 1.0 for DL3DV, i.e. SCVideo `dataset_large.py:313`),
  `auto` = the old coupling (`rel` iff `scale_mode == 'avg_scale'`) and remains the DEFAULT so
  existing configs reproduce bit-for-bit (verified `auto == rel` for `avg_scale`, `auto == raw` for
  `cam_dist_mean`, and the 9 extrinsic channels are byte-identical across the intr axis). The
  convention is a property of the VAE CKPT, not of the translation normalization, so the old
  coupling meant every `cam_dist_mean` / `context_longer` / `first_farthest_135` run fed `raw` to a
  ckpt that wants `rel` → intr recon L1 0.361 instead of 0.004, and the cell SCVideo's DL3DV-only
  config actually uses (`avg_scale` + `raw`) could not be expressed at all. Silent — no crash, no
  loss spike, only the 2 intrinsics channels are affected. The `raw` branch now derives width/height
  from the principal point (`2cx, 2cy`) like SCVideo rather than `transforms.json`'s `w, h`;
  identical for DL3DV (0 of 400 scenes differ). See `FIX.log` 2026-07-30.
- **index cache survived the dataset move with dead absolute paths** (`main/dataset_dl3dv.py`).
  `<root>/.latentcam_index/<key>.pt` stored `scene_dir_list` as ABSOLUTE paths, so after the
  root move the cache loaded fine but every scene dir pointed at the old location (`isdir` →
  False) and `__getitem__` would fail. Now the cache persists `scene_chunks` (paths relative to
  `self.root`) and rejoins them against the current root on load. Legacy absolute caches are
  still accepted, but only if `scene_dir_list[0]` still resolves; otherwise the cache is declared
  STALE and the index is rebuilt.
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
