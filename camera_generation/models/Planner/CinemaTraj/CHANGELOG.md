# Changelog (CinemaTraj)

All notable changes to the CinemaTraj sub-project (카메라 뱅크 굽기 · Blender GT 렌더 · 캡션).
Follows [Keep a Changelog](https://keepachangelog.com/).

이 파일은 2026-09-20 에 시작했다. 그 이전 이력은 `git log -- camera_generation/models/Planner/CinemaTraj` 가 원본이다.

## [Unreleased]

### Added
- **`eval_collision_rate.py` 에 `dynpose_d200` 코퍼스 + `--scenes` / `--extra_eval_dir` /
  `--per_scene_max` (2026-09-20).** D200 5 arm 의 testset 충돌률을 재려고 붙였다.
  기존 `vista_d121` 경로는 **비트 동일** — 새 키(`eval_data`/`graph_root`)는 코퍼스 dict 에서
  없으면 예전 상수(`Vista4D-Eval-Data`, `out/`)로 떨어지고, 새 인자 기본값은 전부 무동작이다.
  - 코퍼스마다 recon 루트와 scene_graph 루트가 다르다 — d200 은
    `DynPose-100K` + `out_dynpose`. `eval_data` 는 **`eval_data/` 의 부모**를 적는다
    (`scene_graph/io.py:88` 이 다시 붙인다. `run_gendop_eval.py:60-63` 에 같은 함정이 기록돼 있다).
  - `cloud.npz` 는 안 읽는다. 점군은 recon depth 에서 직접 세우고 `scene_graph.json` 에서는
    `S` 만 읽으므로 D178 의 cloud 삭제와 무관하다 — `--cloud_source` 같은 게 필요 없다.
  - `--scenes FILE` 로 D205 의 200 씬 표본만 잰다. 5,144 엔트리 × 1,017 씬 전량은 비싸고,
    같은 표본을 써야 `subject_in_frame` 표와 열을 나란히 놓을 수 있다.
  - `--per_scene_max` 는 **출력만** 자른다 (JSON `by_scene` 은 전량). 200 씬이면 씬 블록이
    200 개 찍힌다.
  - 실행 전 200 씬 전부에 `scene_graph.json` / `recon_and_seg` / `depths` / `seg_instances`
    가 있는지 확인했다 (결손 0). 중간에 죽지 않게 하려고 스킵 분기를 넣는 대신 사전 검증을 택했다 —
    조용한 스킵은 "다 쟀다"로 읽힌다.

### Changed
- **`run_gendop_eval.py` `dynpose_d200.ours` 에 arm ⑤ `d200_molmo2_dec_l21` 추가 (2026-09-20).**
  ⑤ 가 14:34:49 에 `epoch_cap 50` 완주(rc=0)해 testset eval 을 돌릴 수 있게 됐다. 이 항목이
  빠져 있으면 D208 합산 score 가 **⑤ 없이** 9시간을 돌고 끝난다 — ⑤ 는 epoch 50 val 에서
  loss_traj / loss_latent / clatr fcd / caption fscore 4개 1등이라 비교표의 핵심이다.
  15:33 에 시작한 score 를 2분 만에 끊고 ⑤ 를 채운 뒤 재기동했다.
  - ⑤ 만 run 디렉토리 날짜가 다르다 (`20260919_172530_...`, 09-19 재기동본). 동명 디렉토리가
    5개 있고 `ckpts/last.pth` 는 이 하나뿐이라, 앞 4개와 같은 09-18 이름으로 찾으면 없다.

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

- **`sample_camera_bank.py` / `fit_hole_ladder.py` `--mesh_margin_autoclamp` (2026-09-20).**
  D207 1차 전량에서 fit 이 191 chunk 중 **11 편**에서 D47 legality assert 로 죽었다 —
  `--mesh_margin_frac 0.08` 이 만든 임계(0.16~0.30 m)가 **소스 카메라 자신의 mesh 여유**
  (0.10~0.28 m)보다 커서다. TRUMANS 는 실내라 카메라가 벽에 붙는 chunk 가 있고, 소스 카메라를
  불법으로 판정하는 임계는 충돌이 아니라 버그다.
  `--mesh_margin_autoclamp f` 를 주면 **그런 chunk 에서만** 임계를 `f × 소스여유` 로 내린다.
  이미 합법인 chunk 는 코드 경로가 그대로라 **기본값 0.0 은 물론 켜도 비트 동일**이다.
  내렸을 때는 `behind_context` 가 `mesh_margin_clamped=(before, after)` 를 반환하고 로그를 찍는다.
  - `configs/bank/d207_trumans_pilot.json` 의 `fit.args` 에 `0.9` 로 켰다. 10/11 편이 rc=0 으로
    복구됐다 (180 편은 손대지 않았다). 남은 `tru_0ab03928_a13_s3f0k6` 은 다른 assert
    (`소스 카메라가 mesh 격자에서 unreachable`)라 이 플래그로는 안 산다.
- **`run_gendop_eval.py` score 단계에 외부 베이스라인·표본·cloud_source 배선 (2026-09-20).**
  전부 기본값이 예전 동작이라 기존 런과 비트 동일이다.
  - `--extra_eval_dir LABEL=DIR` (반복 가능) — 우리 파이프라인 밖에서 만든 eval 폴더
    (E.T./DIRECTOR)를 같은 표에 올린다. `--arm none` 은 GenDoP arm 을 하나도 안 올린다
    (`arms=[]` 와 `arms=None` 을 `or ARMS` 로 못 가르므로 `stage_score` 쪽 분기를 고쳤고,
    infer/evaldir 에서 빈 리스트가 조용히 "전부"로 되살아나는 것은 assert 로 막았다).
  - `--scenes FILE` / `--out_tag` — D205 의 200 씬 표본(`tmp/d205/sample200_scenes.txt`)으로
    재서 그 표와 열을 나란히 놓는다. `--subject_occlusion` 도 같이 넘긴다.
  - `--cloud_source {npz,memory}`.
- **`gendop_preds_to_eval_dir.py --caption_dir` (2026-09-20).**
  `--caption_from ref` 일 때 캡션을 가져올 폴더를 따로 준다. 텍스트 조건을 갈아끼운 런
  (`--text_dir`)이 지표 단계에서 **모델이 받지 않은** 문장을 읽는 것을 막는다.
  `run_gendop_eval.py` 는 `--text_dir` 를 줬을 때 자동으로 같이 넘긴다. 기본값 None = ref 폴더.
- **`dynpose_gendop_inputs.py --text_only` (2026-09-20).** rgbd 루프와 symlink 를 건너뛴다. 기본 off.
- **`run_director_batch.py` (신규, 2026-09-20).** DIRECTOR(E.T.) 를 split 전량에 돌려
  `gendop_preds_to_eval_dir.py` 가 먹는 npz 를 낸다. 모델·CLIP 을 한 번만 올리고 16 entry 씩
  묶어 `Diffuser.sample()` 을 부른다 (`StackedRandomGenerator` 가 배치 원소마다 별도
  `torch.Generator` 를 쓰므로 `seeds=[seed]*B` 는 개별 실행과 등가 — 이어달리기도 안전).
  E.T. world(m) → scene-graph G(u) → 코퍼스 world 변환을 담고, **카메라 축 규약을 런마다 실측**해
  찍는다 (5,144 entry: 카메라 +z 와 char−cam 사이 각 mean 36.63° / median 27.97° → **OpenCV**.
  `run_director_vista.py` 에 "미검증"으로 남아 있던 항목이다).

### Changed
- **`run_gendop_eval.py` `dynpose_d200` 의 `eval_data` 를 `DynPose-100K/eval_data` 로 정정
  (2026-09-20).** d200 은 d185+d199 pooled = dynpose-100k 이라 recon 이 거기 있다. d137 까지
  쓰던 `DynPose-LBM/eval_data` 를 물려받는 바람에 1,017 씬 중 **107 씬**(두 코퍼스가 겹치는 몫)만
  잡혔고, 그 107 을 보고 "rgbd 입력이 10% 뿐"이라고 잘못 읽어 `text_only=True` 를 걸었다.
  실제로는 `video.mp4` / `depths/00000.exr` 둘 다 **1,017/1,017** 이라 `gendop_rgbd` 도 돌릴 수
  있다 — `text_only` 플래그는 남기되 이 코퍼스에서는 뗐다. score 단계도 이 루트로 recon 을 읽는다.
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
