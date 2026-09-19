# Changelog (CinemaTraj)

All notable changes to the CinemaTraj sub-project (카메라 뱅크 굽기 · Blender GT 렌더 · 캡션).
Follows [Keep a Changelog](https://keepachangelog.com/).

이 파일은 2026-09-20 에 시작했다. 그 이전 이력은 `git log -- camera_generation/models/Planner/CinemaTraj` 가 원본이다.

## [Unreleased]

### Added
- **TRUMANS 전량 뱅크 세대 `d207T` (`configs/bank/d207_trumans_pilot.json`) (2026-09-20).**
  TRUMANS-Lite 191 chunk 을 **d185 라우팅 축**(`route_presets` grid2x2 슬롯)으로 다시 굽는다.
  기존 `hole_bank_k6_d132` 는 route 단계 없이 tau/fit 에 `--nodes dyn_0` 을 직접 박은 옛 합집합
  preset 이라 d200 코퍼스에 pooled 할 수 없다. 폴더 이름을 `d207T` 로 가른 것은 D105 원칙
  (게이트 규약이 바뀌면 이름을 바꾼다).
  - anchor 는 **사람(`dyn_0`) 하나만** — `route_presets.py` 에 `--nodes` 가 없으므로
    `--max_dynamic_anchors 1 --max_static_anchors 0 --max_anchors 1` 로 같은 결과를 낸다.
  - chunk 당 최대 3 변이(anchor 1 × grid 슬롯 2 + free-moving 1)라 `--target_variants` 와
    fit 의 `--fallback_target` 을 둘 다 3 으로 맞췄다. 어긋나면 fallback 이 못 채우는 자리를 계속 재시도한다.
  - `--cloud_source npz` — TRUMANS `cloud.npz` 는 `--scene_scale_mode points_first_cam
    --scene_scale_stride 1` 로 구워져 있고, memory 경로로 다시 구우면 그 스케일 인자를 안 받아 `S` 게이지가 달라진다.
  - `--collision_source both` (d132 상속) — `.blend` 가 있어 `mesh_grid.npz` 가 소스 카메라가
    한 번도 안 본 벽·옆방까지 안다. d116 실측(chunk a00, 836 변이) depth 130 / mesh 357 / 겹침 0.
  - `stages` 에서 graph/cloud 를 뺐다. 191/191 편에 `.graph_s115` + `.cloud_s115` + `mesh_grid.npz` 가
    이미 있고(mesh GT 기반), 다시 지으면 같은 마커 아래 다른 게이지가 섞인다.
- **`trumans_gt_render.py --anim` — 프레임별 render job 을 animation job 한 번으로 (2026-09-20).**
  프레임마다 `bpy.ops.render.render()` 를 부르면 그 한 번이 Cycles 세션 하나라 씬을 통째로 다시
  device 로 올린다 (`use_persistent_data` 는 한 job 안에서만 먹는다). 카메라 pose 를 CONSTANT 보간
  keyframe 으로 굽고 `scene.frame_step` 을 맞춰 animation render 한 번으로 묶는다.
  - 실측(00add26c_a01 `_source`, 49 프레임 640×360 48 spp OPTIX, 같은 blend/pose):
    프레임별 job **272.6 s** (5.547 s/frame, first 제외) vs `--anim` **90.3 s** (1.84 s/frame) → **3.02×**.
  - 정확도: 픽셀 차이는 adaptive sampling 잡음 수준(meanAbs **0.0607/255**, maxAbs 24),
    카메라는 `c2w_opencv`/`c2w_blender_gl` maxAbs **4.768e-07** 로 일치.
  - **기본 off = 예전 동작 비트 동일.** `--passes rgb` + 균일 간격 프레임 + cycles 전용이라
    depth/index(프레임마다 엔진을 바꾼다)나 `--frame_list` 와는 assert 로 배타.
  - `render_meta` 에 `"anim"` 을 남긴다. `--anim` 일 때 프레임별 `timings_s` 는 job 총시간/프레임수라 합만 정확하다.
- **`bank_to_blender_poses.py --anim` / `--rgb_cdevice` (2026-09-20).**
  생성되는 `render.sh` 가 `trumans_gt_render.py --anim` 을 부르도록 배선. 기본 off 라 예전 `render.sh` 와 글자 동일.
  (플래그 이름이 `--cycles*` 로 시작하면 Cycles 애드온이 argv 를 prefix 매칭해 런을 죽인다 — 그래서 `--cdevice`/`--rgb_cdevice`.)

### Changed
- **GPU 정책: TRUMANS 작업에 한해 GPU 5 사용 가능** (2026-09-20 사용자 지시, `CLAUDE.md` 반영).
  뱅크 굽기·Blender 렌더·TRUMANS 캡션용 vLLM 이 해당한다. `run_bank.py` 는 `BANK_GPU_ALLOW=5` 를
  줘야 `--gpu 5` 를 받는다 (없으면 `--gpu` ∈ `"01234"` assert). 6,7 은 여전히 금지,
  TRUMANS 가 아닌 학습/추론은 0~4 그대로.

### 측정했으나 넣지 않은 것
- **정적 모디파이어 베이킹 (`--bake_static`) — 되돌렸다 (2026-09-20).**
  geometry 의 91%(898,861 vert 중 821,806; `vase_03` 하나가 geometry nodes 로 792,481)가 local space
  에서 프레임 불변인데 부모 empty 가 애니메이션이라 매 프레임 재평가된다. 베이킹하면 `--anim` 에서
  90.3 → 88.6 s (1.9%), 프레임별 job 에서 272.6 → 258.8 s (5.1%). 둘 다 잡음 수준이라 코드를 전부 제거했다.
- **여러 preset 을 한 프로세스에서 렌더 — 기각.** steady state 프로세스 오버헤드가 preset 당 ~6.4 s
  (blend 로드 + Blender 기동 4.29 s; 첫 프로세스만 OptiX 워밍으로 21.7 s) 라 묶어도 ~5%.
