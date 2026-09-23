# CinemaTraj `eval/` 사용법

`eval/` 은 **파이프라인이 아니라 그 옆에 붙는 자(尺)** 다. 여기 있는 38개 스크립트는 뱅크를 굽지
않고 코퍼스를 바꾸지 않는다 — 이미 디스크에 있는 `bank.json` / `poses.npz` / eval 폴더 /
`_tag.json` 을 읽어서 **세고, 재고, 대조하고, 스윕한다.** 게이트가 아니라 감사(audit)라서
대부분 행을 지우지 않고 열만 붙인다. 유일하게 원본을 고치는 것은
`sanitize_director_focus.py`(그것도 기본이 dry run)다.

## 실행 규칙 — CinemaTraj 루트에서 돈다

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
<PY> eval/<script>.py ...
```

import 는 어디서 불러도 된다 — 각 스크립트가 `CINEMATRAJ_ROOT =
path.dirname(path.dirname(path.abspath(__file__)))` 를 계산해 `sys.path` 에 직접 넣는다
(`audit_bank_geometry.py`, `eval_collision_rate.py`, `smoke_micro_ops.py` … 전부 같은 관용구).
**문제는 기본값이다.** `--output_root` 가 `out` / `out_trumans`, `--out` 이
`results/20260907_d159_collision` 처럼 **상대경로**인 스크립트가 많아서, cwd 가 루트가 아니면
없는 디렉토리를 조용히 읽거나 엉뚱한 곳에 쓴다. rc=0 으로 끝나므로 안 들킨다.

### 인터프리터

```bash
PY_LC=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
PY_V4=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
PY_GD=/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python
```

`lbm.render.CloudRenderer` 를 import 하거나 OpenEXR depth 를 읽는 것은 `vista4d`,
GenDoP `core.*` 를 import 하는 것은 `GenDoP`, numpy/json 만 쓰는 것은 아무거나다. 각 절의
`env` 를 볼 것. 코드에서 못 가르는 것은 **미확정**으로 적었다.

### 프로젝트 규칙과 docstring 예시의 어긋남

- **GPU 는 0~3 만.** docstring 예시에 `CUDA_VISIBLE_DEVICES=5`(`eval_subject_in_frame.py`),
  `--cuda 6`(`time_vista_stages.py`) 가 남아 있다. 그대로 복붙하지 말 것.
  (`time_vista_stages.py` 의 argparse 기본값은 `--cuda "0"` 이라 기본값 자체는 안전하다.)
- **임시파일은 `/data1/cympyc1785/LatentCamVid/tmp/` 아래.** docstring 예시 7곳이 `/tmp/...`
  를 쓴다(§검증 결과). argparse 기본값 쪽은 이미 `LatentCamVid/tmp/` 로 옮겨져 있다
  (`audit_scene_scale --out`, `audit_tau_axes --out`, `time_vista_stages --out_root/--log_dir`,
  `probe_trumans_blend --out`, `vista_blend_check --out`).
- **testset 평가는 `last.pth`.** `eval_collision_rate.py` 의 `CORPORA` 에 박힌 arm 경로도
  전부 `__last` / `__epoch100__seed42` 형태의 산출 폴더지 `best` 가 아니다.

---

## 파일 38종 한눈에

| 파일 | 역할 |
|---|---|
| `audit_bank_geometry.py` | 뱅크 전 변이 × 49프레임의 G1(표면 뒤) / G3(가림) 위반을 세어 `geometry.csv` |
| `audit_bank_status.py` | `status`/`binding` 분포 집계 + `clamped_low` 완화 후보 목록 (렌더 0회) |
| `audit_lbm_distance.py` | LBM 카메라의 실제 미터 거리(`d_focus`/`d_human`)를 Lite 반경과 같은 자에 올린다 |
| `audit_lite_framing.py` | TRUMANS-Lite 프레이밍 = 보이는 OBB 픽셀 / 화면에 투영된 OBB 면적 |
| `audit_scene_scale.py` | 씬 단위 `S` 의 정의 후보 6종 + 프레임별 변동(`S_t_ratio`)을 나란히 실측 |
| `audit_tau_axes.py` | τ 대체 이동량 축 11종을 같은 뱅크에서 재고 Spearman/cv/frac_zero 로 비교 |
| `audit_trumans_pkl_camera.py` | TRUMANS 실촬 카메라 pkl 2편의 보행 구간 net 이동량 (인자 없음) |
| `eval_collision_rate.py` | 예측 궤적 충돌률 — G1 재투영 + kNN 점군, 단위 호(arc) 정규화까지 |
| `eval_subject_in_frame.py` | eval 폴더 여러 개를 같은 점군에 렌더해 subject_in_frame / hole / 가림 비교 |
| `compare_aim_timing.py` | 매 프레임 조준 vs keyframe 6 보간 3-arm 대조 (회전 매끄러움 ↔ 조준 오차) |
| `compare_camera_distributions.py` | 두 코퍼스 `_tag.json` 분포 대조 (GT vs GT), seg_kwargs 불일치는 죽인다 |
| `compare_lbm_vs_lite.py` | 같은 action 의 LBM arm / Lite arm 프레이밍 CSV 를 붙여 차이 표 |
| `diff_bank_variants.py` | 두 뱅크를 `variant_id` 로 조인해 `status`/`binding`/`knob` 뒤집힘만 센다 |
| `diff_f7_banks.py` | F7 시간축 절단 on/off 뱅크 — 회귀(안 바뀐 행 문자 동일) + 변경 목록 |
| `dynpose_gendop_inputs.py` | GenDoP baseline stage `inputs` → **GENDOP_USAGE.md** |
| `eval_dir_to_gendop_npz.py` | GenDoP baseline 역변환 (eval 폴더 → npz) → **GENDOP_USAGE.md** |
| `gendop_preds_to_eval_dir.py` | GenDoP baseline stage `evaldir` → **GENDOP_USAGE.md** |
| `gendop_release_eval.py` | GenDoP baseline 캡션 왕복 지표 → **GENDOP_USAGE.md** |
| `gendop_release_infer.py` | GenDoP baseline stage `infer` (릴리즈 ckpt 추론) → **GENDOP_USAGE.md** |
| `ablate_vlm_hole_perception.py` | VLM 이 magenta 구멍을 그림으로 보나 숫자만 읽나 — 9조건 ablation |
| `attribute_g1_hits.py` | G1 히트 픽셀을 seg instance × 기하(ground/below_cam/level)로 귀속 |
| `probe_g1_reference.py` | G1 임계의 기준 거리 후보 — 소스 카메라 자기 여유 `g1_src` 와 CV 순위 |
| `probe_near_depth_repeat.py` | `near_depth` 열의 재실행 흔들림을 뱅크와 같은 렌더 경로로 재현 |
| `probe_static_sdf.py` | 정적 점군만으로 SDF 가 되는지 (frame0 / 13프레임 / 동적 제외 3조건) |
| `probe_tau_divisor.py` | τ 분모 후보 6종 — 같은 사다리 단에서 두 씬 τ 비가 1 에 가까운가 |
| `probe_trumans_blend.py` | TRUMANS `.blend` 내부(오브젝트명·씬명·armature) headless 덤프 |
| `probe_wall_planes.py` | 벽 평면을 반공간 게이트로 쓸 수 있나 — 소스 카메라 49대 합법성으로 판정 |
| `rank_subject_motion.py` | split 을 subject 각이동량으로 정렬해 high-motion 부분 split 생성 |
| `rank_traj_text_match.py` | pred 궤적 vs GT 의 방향·길이·회전 오차, arm 간 격차로 데모 후보 정렬 |
| `slice_subject_in_frame.py` | 이미 낸 `subject_in_frame.json` 을 부분집합/그룹으로 재집계 (GPU 0) |
| `sweep_caption_thresholds.py` | DataDoP 분절기 임계 sweep (49프레임@fps10 눈금 이식, LLM 없음) |
| `sweep_collision_margin.py` | 충돌 임계 margin 스윕 곡선 + Blender raycast audit 합산 |
| `pick_warp_sample_scenes.py` | 재굽기 검증용 표본 씬 결정론적 선택 (moving N + static N) |
| `rescale_eval_preds_to_ref.py` | latentcam pred 를 ref 의 `rmax` 로 되맞춘 대조 arm 생성 |
| `sanitize_director_focus.py` | LBM Director `focus_ids` 에서 blend 에 실재하는 비-asset id 제거 |
| `smoke_micro_ops.py` | micro-adjust 연산 14종을 VLM 없이 전부 적용해 게이트 결과 표 |
| `time_vista_stages.py` | graph→pcd→bank→fit→bankemit 단계별 벽시계 시간 (격리 out_root) |
| `vista_blend_check.py` | `vista_to_lbm_demo.py` 산출 `.blend` 가 소스와 같은 씬인지 headless 확인 |

---

## 1. 뱅크 감사 (`audit_*`)

### `audit_bank_geometry.py`

뱅크 변이의 **기하 위반**을 궤적 전 프레임에서 센다 — G1(카메라가 표면 뒤로 들어감, 재투영만)
과 G3(subject 가림, 렌더 2-pass). `sample_camera_bank.py`/`fit_hole_ladder.py` 가 재는
`hole_fraction` 은 충돌을 못 잡는다(벽을 통과하면 벽 너머 관측이 그려져 `valid_mask` 가 멀쩡하다).
출력은 `<output_root>/<video>/<bank_dir>/geometry.csv` + 요약표.

```bash
CUDA_VISIBLE_DEVICES=1 $PY_V4 eval/audit_bank_geometry.py --video camel \
    --bank_dir hole_bank_k6_d121 --occlusion --occlusion_frames 7
```

| 인자 | 기본값 | 메모 |
|---|---|---|
| `--video` | (필수) | |
| `--bank_dir` | `bank` | fit 산출을 보려면 `hole_bank*` 를 명시 |
| `--behind_frames` | `7` | G1 이 되쏘는 소스 프레임 수 |
| `--margin_frac` / `--clear_frac` | `0.02` / `0.0` | `·S` 단위 |
| `--radius_px` | `0` | 히트 패치 반경 |
| `--occlusion` / `--no_occlusion` | off | 켜면 렌더 2배. 끄면 G1 만(렌더 0회) |
| `--occlusion_frames` | `5` | |
| `--obb_occlusion`, `--obb_occlusion_diag` | off | |
| `--min_occlusion_pass` | `0.4` | |
| `--fixed_focal` / `--no_fixed_focal` | **on** | 뱅크의 fixed_focal 과 맞춰야 화각이 안 떨린다 |
| `--tile_width` / `--tile_height` | `640` / `360` | |
| `--device` | `cuda` | |

env: `vista4d`.

### `audit_bank_status.py`

`fit_hole_ladder.py` 가 남긴 `status`/`binding` 두 열을 **코퍼스 전체에서 모아 본다** (새로 재지
않는다 — 렌더 0회, 전 코퍼스 수 초). `status` 는 `solved` / `clamped_low` / `unreached` /
`<gate>_limited` / `static` 이고 접미사 `+tau_floor` 는 knob 하한이 `tau_start + 0.02` 로 올라간
경우다. `binding` 은 `over()` 의 if/elif 체인에서 **처음 발화한** 게이트라 "이 게이트만 풀면
살아난다"가 아니라 "이 게이트가 첫 벽이었다"로 읽어야 한다.

```bash
$PY_LC eval/audit_bank_status.py --output_root out_dynpose --bank_dir hole_bank_d110 \
    --relax_list /data1/cympyc1785/LatentCamVid/tmp/d110/clamped_low.json \
    --csv /data1/cympyc1785/LatentCamVid/tmp/d110/status.csv
```

| 인자 | 기본값 | 메모 |
|---|---|---|
| `--output_root` | `out` | 상대경로 — 루트에서 실행 |
| `--bank_dir` | `hole_bank` | |
| `--videos` | `None` (전량) | |
| `--relax_list` | `None` | `clamped_low*` 행만 binding 별로 묶어 JSON |
| `--csv` | `None` | |
| `--top` | `40` | |

env: 아무거나 (numpy 도 안 쓴다).

### `audit_lbm_distance.py`

원본 LBM 이 고른 카메라의 거리를 **미터로** 만들어 Lite 뱅크 반경(1.5/2.2/3.0/4.0 m)과 같은 자에
올린다. `d_focus`(LBM 이 실제로 겨눈 물체까지)와 `d_human`(사람까지)을 **따로** 낸다 — LBM
`primary_focus_id` 는 40/42 가 소품이라 `d_focus` 만 보면 거리가 적당해 보인다.
출력 `<out>/lbm_distance.csv` + `<out>/lite_distance.csv`.

```bash
$PY_LC eval/audit_lbm_distance.py --verbose
```

| 인자 | 기본값 |
|---|---|
| `--lbm_root` | `.../Planner/Look-Before-Move/...` (`LBM_OUT_DEFAULT`) |
| `--recon_root` | `RECON_DEFAULT` (Lite probe/manifest/poses) |
| `--out` | `<CT>/out/trumans_lite_bank` |
| `--lite` / `--no_lite` | on |
| `--verbose` | off |

비고: 좌표는 전부 blend world(미터). `d_human` 분모는 앉은 자세에서 z 가 내려가므로 절대값이
아니라 **분포와 Lite 대비 위치**로 읽을 것. env: 아무거나 (numpy/csv).

### `audit_lite_framing.py`

프레이밍 = `occl_keep` = (안 가려진 픽셀) / (**화면에 투영된** OBB 면적). 분모가 화면 안
bbox 인 게 요점 — 전체 hull 을 분모로 쓰면 close-up 의 정상적인 화면 밖 이탈이 "가려짐"으로
읽힌다. 잘림은 `crop_keep`(Sutherland-Hodgman 해석적 면적)으로 **따로** 찍고 곱하지 않는다.
출력 `<out>/lite_framing.csv`(clip 당 1행) + `<out>/lite_framing_frames.csv`(프레임당 1행).

```bash
$PY_V4 eval/audit_lite_framing.py --videos tru_00add26c_a00 tru_0aa05d5a_a03 --verbose
```

| 인자 | 기본값 | 메모 |
|---|---|---|
| `--bank_manifest` | `<CT>/out/trumans_lite_bank/bank_manifest.json` | |
| `--videos` | `[]` (전량) | |
| `--min_framing` | `0.7` | 요약표 임계 |
| `--occlusion_eps` | `0.02` | `t_enter - eps` 보다 앞이면 가려짐 |
| `--z_near` | `0.05` | 12 edge 를 여기서 3D 클리핑 |
| `--include_crop` | off | 켜면 `framing = crop_keep × occl_keep` 옛 정의 |
| `--preview` / `--preview_fps` | off / `12` | |
| `--work` / `--out` | `<CT>/out/trumans_recon` / `<CT>/out/trumans_lite_bank` | |

비고: OBB 는 실루엣보다 크므로 `crop_keep` 은 잘림 과대, `occl_keep` 은 가림 과대 보고 쪽이다 —
순위용이지 절대 기준이 아니다. `nearclip` 열이 0 이 아니면 카메라가 몸통에 박혔다는 신호.
env: `vista4d` (OpenEXR).

### `audit_scene_scale.py`

씬 단위 `S` 의 정의 후보를 **같은 영상에서 나란히** 잰다: `S_f0`(현행, frame 0 non-sky 평균 ray
length) / `S_f0_nodyn` / `S_frames_mean` / `S_pts_first` / `S_pts_centroid` / `S_pts_own`.
핵심 열은 `S_t_p05/p50/p95`, `S_t_ratio(=max/min)`, `S_t_drift(=S_48/S_0)` — ratio 가 1 근처면
frame 0 하나로 충분하다는 직접 증거다. **고치지 않고 재기만 한다** (바꾸면 뱅크 재굽기가 따라온다).

```bash
$PY_V4 eval/audit_scene_scale.py --videos camel avocado-slice parkour snowboard
$PY_V4 eval/audit_scene_scale.py --all --stride 4 \
    --out /data1/cympyc1785/LatentCamVid/tmp/d-scale/scene_scale.csv
```

| 인자 | 기본값 |
|---|---|
| `--videos` / `--all` | `None` / off |
| `--stride` | `4` |
| `--max_points` | `2000000` |
| `--seed` | `0` |
| `--out` | `/data1/cympyc1785/LatentCamVid/tmp/scene_scale.csv` |
| `--eval_data` / `--vista4d_root` | `lbm.cloud` 기본값 |

env: `vista4d` (렌더 없음, CPU).

### `audit_tau_axes.py`

τ 를 대신할 이동량 축 후보 11종(`tau_source` `tau_shape` `path_S` `net_S` `rel_sub` `dr_ref`
`r_ratio` `az_deg` `elev_deg` `rot_deg` `subtend_ratio`)을 같은 뱅크에서 재고 세 조건으로
비교한다 — ① `--metric`(기본 `hole_fraction`)과의 Spearman ρ, ② 씬 간 median 변동계수
`cv_between`, ③ `frac_zero`. **셋은 서로 상충하므로 한 열로 줄 세우지 말 것** (`cv_between` 0 인
축은 씬 차이를 아예 안 보는 것일 수 있다).

```bash
$PY_V4 eval/audit_tau_axes.py --bank_dir hole_bank_k6_d99 --videos parkour snowboard camel \
    --out /data1/cympyc1785/LatentCamVid/tmp/d-tau/tau_axes.csv
```

| 인자 | 기본값 |
|---|---|
| `--bank_dir` | `hole_bank_k6_d99` |
| `--videos` / `--output_root` | `None` / `None` |
| `--metric` | `hole_fraction` |
| `--status` | `None` (전량) |
| `--out` | `/data1/cympyc1785/LatentCamVid/tmp/tau_axes.csv` |

env: `vista4d` (CPU).

### `audit_trumans_pkl_camera.py`

TRUMANS 실촬 카메라 pkl 2편에서 **사람이 걷는 창**과 **정지한 창**을 갈라 카메라 net |dt| /
프레임당 |dt| / 사람 net 을 median 으로 찍는다. 합성 소스 카메라의 `track_gain` 이 과한지
아닌지를 실측으로 가르는 용도.

```bash
$PY_LC eval/audit_trumans_pkl_camera.py
```

**인자가 없다.** 전부 모듈 상수다 — `ROOT=/data1/cympyc1785/data/trumans/Data_release`,
`PAIRS` 2편(`00add26c…`/`0aa05d5a…`), `W=49`, `FPS=25.0`, `SPEED=0.4`(보행 판정 m/s),
`SMPL_HORIZONTAL=[0,2]`(y-up 이라 수평은 x,z). 창 분류는 walk ≥0.8 / still ≤0.1.
pkl 구조를 모른 채 `(F,3)`/`(F,3,4)`/`(F,4,4)` 를 탐색하므로 모양이 다르면 `SystemExit`.
env: 아무거나 (numpy/pickle).

---

## 2. 지표 (`eval_*`)

### `eval_collision_rate.py`

예측 궤적의 충돌률을 **두 게이지**로 잰다. G1 = 카메라 중심을 소스 프레임에 재투영해
`z_cam + clear·S > depth + margin·S` (임계가 하나뿐이지만 **화면 밖은 못 본다**). kNN = 전 프레임
depth 를 unproject 한 점군에서 k번째 최근접거리 / S (화면 밖도 재지만 임계 r 을 밖에서 정해야 하고
점군 밀도=`--stride` 에 딸려 있다 → GT 위에서 `--knn_target_rate` 분위수로 캘리브레이션).
궤적 판정은 49프레임 중 하나라도 걸리면 충돌. **단위 호 정규화** `knn_per_arc = knn_frames /
arc_u` 를 같이 내고, 집계는 `mean`(엔트리별 비 평균, `arc_u < --arc_floor_u` 제외 → `n_static`)
과 `pooled`(Σ/Σ) 두 가지를 나란히 찍는다. 출력 `<out>/collision_rate.json` +
`<out>/collision_per_entry.csv`.

```bash
OPENCV_IO_ENABLE_OPENEXR=1 $PY_V4 eval/eval_collision_rate.py \
    --corpus dynpose_d200 \
    --scenes /data1/cympyc1785/LatentCamVid/tmp/d205/sample200_scenes.txt \
    --out results/20260920_d205_collision_d200
```

| 인자 | 기본값 | 메모 |
|---|---|---|
| `--corpus` | `vista_d121` | choices = `dynpose_d200`, `vista_d121`, `vista_d215` |
| `--out` | `<CT>/results/20260907_d159_collision` | |
| `--metric` | `both` | `g1` / `knn` / `both` |
| `--knn_k` | `10` | k=1 이면 depth 경계 부유점 하나가 판정을 뒤집는다 |
| `--knn_r` | `None` | 주면 캘리브레이션 생략 |
| `--knn_calib_arm` / `--knn_target_rate` | `gt` / `0.01` | |
| `--arc_floor_u` | `0.01` | 이하는 `mean` 에서 제외 |
| `--stride` / `--edge_thr` | `4` / `0.05` | 점군 밀도 · 깊이 불연속 컷 |
| `--behind_src_frames` | `49` | |
| `--behind_margin_frac` / `--behind_clear_frac` | `0.02` / `0.0` | |
| `--behind_radius_px` / `--behind_min_zcam` | `0` / `0.02` | |
| `--time_match` | off | |
| `--per_scene` / `--per_scene_max` | on / `12` | |
| `--limit` / `--scenes` / `--extra_eval_dir` | `0` / `None` / `[]` | `--extra_eval_dir` 는 `LABEL=DIR` 반복 |

비고 두 개. ① eval 폴더의 `*_transforms_{ref,pred}.json` 은 **DA3 world c2w(OpenGL)** 이고 recon
스케일 그대로라 되살릴 배율이 없다 — `S` 만 `scene_graph.json` 에서 읽는다. ② GenDoP arm 은
`gendop_preds_to_eval_dir.py` 가 `rmax` 를 GT 에서 빌려오므로 **크기가 아니라 모양만** 자기 것이다
(raw arm 을 쓰려면 GENDOP_USAGE.md §3). env: 미확정 — docstring 예시가
`OPENCV_IO_ENABLE_OPENEXR=1 python` 으로만 적혀 있다. EXR depth 를 읽으므로 그게 되는 env 라야
한다(`dynpose_gendop_inputs.py` docstring 은 vista4d cv2 가 `.exr` 에 None 을 돌려준다고 적어 둔
반례가 있으니 새 환경에서 처음 돌릴 때는 1 씬으로 먼저 확인할 것).

### `eval_subject_in_frame.py`

eval 폴더 여러 개를 **같은 점군에 렌더해서** 한 표에 올린다. `verify.py:measure()` 의 정의를
import 해서 쓴다 — `subject_in_frame`(실루엣 중심이 중앙 `center_box` 안), `hole_fraction`,
`subject_pixel_coverage`. `--subject_occlusion` 을 켜면 렌더 2-pass 로 `subject_visible_frac` /
`subject_visible_min` 이 붙는다 (위 세 지표로는 가림이 안 잡힌다 — 위치와 크기만 재기 때문).

```bash
CUDA_VISIBLE_DEVICES=0 $PY_V4 eval/eval_subject_in_frame.py \
    --ref_eval_dir /data1/.../eval_my/20260918_140904_dynpose_d200_da3__last \
    --name_prefix dynpose --cloud_root out_dynpose \
    --eval_data /data1/cympyc1785/data/DynPose-LBM \
    --corpus_root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose \
    --eval_dir ours=/data1/.../20260918_140904_dynpose_d200_da3__last \
    --eval_dir gendop_a=results/20260901_gendop_dynpose_val/eval_dir_case_a \
    --out results/20260901_gendop_dynpose_val/subject_in_frame.json
```

| 인자 | 기본값 | 메모 |
|---|---|---|
| `--ref_eval_dir` | (필수) | |
| `--eval_dir` | `[]` | `LABEL=DIR` 반복 |
| `--name_prefix` | `vista4d` | dynpose 코퍼스면 `dynpose` |
| `--cloud_root` | `<CT>/out` | |
| `--corpus_root` | `.../latentcam_da3_k6_d121` | `prompts.json` 의 `variant_id` 로 entry 별 subject 를 고른다 |
| `--center_box` | `0.8` | |
| `--sweep_lo/_hi/_step` | `0.1` / `1.0` / `0.1` | center_box 스윕 |
| `--fixed_focal` / `--no_fixed_focal` | **on** | |
| `--include_gt` | on | |
| `--temporal_persistence` | on | |
| `--subject_occlusion` | **off** | 기본 off = 기존 표와 비트 동일 |
| `--num_shards` / `--shard_id` / `--merge` | `1` / `0` / `None` | 샤딩 후 `--merge` 로 합침 |
| `--limit` / `--scenes` / `--height` / `--width` / `--device` | `0` / `None` / `None` / `None` / `cuda` | |

비고: subject 는 **씬이 아니라 entry 마다** 다르다 (`variant_id` 앞머리 `dyn_1__…` → `dyn_1`).
씬 단위로 고정하면 anchor 2개인 코퍼스에서 절반이 엉뚱한 물체를 잰다. arm 마다 world 스케일이
다를 수 있어서, raw GenDoP arm 의 `subject_in_frame` 은 구도와 크기를 **같이** 재는 값이다.
env: `vista4d` (CloudRenderer). 학습 중이면 eval 은 한 번에 하나만.

---

## 3. 비교·차분 (`compare_*`, `diff_*`)

### `compare_aim_timing.py`

`aim="look_at"` 변이를 세 arm 으로 **되만들어**(`emit_bank.decision_from_variant` →
`build_poses`) 조준 타이밍만 바꿔 비교한다: `everyframe`(k=0) / `kf6_smoothstep`(배포 뱅크) /
`kf6_smooth_kf`(D96 기본). 위치 채널은 세 arm 이 비트 동일해야 하므로 `dpos_max` 를 매 행에 찍는다.
열: `rot_med`/`rot_max`/`rot_ratio`, `accel_p95`, `aim_err_med`/`aim_err_max`/`aim_err_kf`, `sif`.

```bash
$PY_V4 eval/compare_aim_timing.py --out_root out_trumans --num_videos 4 \
    --presets orbit_left,orbit_right,push_in_arc_left,crane_up \
    --out results/20260901_aim_timing
```

| 인자 | 기본값 |
|---|---|
| `--out_root` | `out_trumans` |
| `--videos` / `--num_videos` | `''` / `12` |
| `--bank_dir` | `hole_bank_k6_d77` |
| `--presets` / `--status` | `''` / `solved` |
| `--max_variants` | `40` |
| `--center_box` | `0.8` |
| `--keep_free` / `--no_keep_free` | **off** (`set_defaults(keep_free=False)`) |
| `--out` | `''` (stdout 만; 주면 `<out>/aim_timing.json`) |

비고: `aim` 세 값 중 대조가 성립하는 것은 `look_at` 뿐이다. `free` 는 k=0 이 강제라 세 arm 이
동일하고, `traj` 는 k=0 이 "조준 안 함"이라 `aim_err` 이 preset 회전량을 재게 된다 — 둘은 개수만
찍고 제외한다. env: `da3`/`vista4d` 아무거나 (렌더 0회).

### `compare_camera_distributions.py`

두 코퍼스의 `_tag.json` 을 같은 눈금으로 대조한다 (GT vs GT, 모델 없음). **비교 성립 조건 3개를
읽어서 다르면 표를 내기 전에 죽는다** — `seg_kwargs` 완전 일치, `num_poses` 일치, `gendop_root`
일치(vendored/pipeline 이 `min_chunk_size` 10 vs 12). `--allow_mismatch` 로만 뚫린다.

```bash
$PY_V4 eval/compare_camera_distributions.py \
  --a_name "ours(d77 bank)" --a_glob "/data1/.../latentcam_da3_k6_d77/vista4d/*/da3/captions_gendop/*_tag.json" \
  --b_name "DataDoP GT"     --b_glob "out_captions/datadop_gt/*/*_tag.json" \
  --out results/20260830_gt_dist_compare
```

| 인자 | 기본값 |
|---|---|
| `--a_glob` / `--b_glob` | (필수) |
| `--a_name` / `--b_name` | `A` / `B` |
| `--out` | `None` (주면 `summary.json` + `level_hist.png`) |
| `--allow_mismatch` | off |
| `--floor` | `1e-06` |
| `--a_divisor_subdir` / `--b_divisor_subdir` | `None` |

비고: 양쪽 D=1(리스케일 없음)이고 게이지가 같다는 주장이 아니라 **레벨 차이 자체가 측정 대상**이다.
env: 미확정 (matplotlib 필요).

### `compare_lbm_vs_lite.py`

`audit_lite_framing.py` 를 **같은 코드·같은 눈금**으로 통과한 두 CSV 를 action 키로 붙인다.
값은 가공하지 않고 그대로 옮긴다. 출력 `<out>/lbm_vs_lite.csv` + 요약표.

```bash
$PY_LC eval/compare_lbm_vs_lite.py \
    --lbm  out/trumans_lbm_bank/lite_framing.csv \
    --lite out/trumans_lite_bank/lite_framing.csv \
    --out  out/trumans_lbm_bank
```

| 인자 | 기본값 |
|---|---|
| `--lbm` / `--lite` / `--out` | 전부 필수 |
| `--recording` | `''` — 비우면 LBM CSV 의 `recording` 열에서 자동 |

비고: **recording 을 반드시 걸러야 한다.** Lite CSV 는 recording 7편이 한 파일에 있어서 안 거르면
같은 action 번호가 여러 행이고, 그 경우 assert 로 죽는다. env: 아무거나 (csv).

### `diff_bank_variants.py`

두 뱅크를 `variant_id` 로 조인해 `status` / `binding` / `knob` 세 열의 뒤집힘만 센다. 임계 변경의
효과는 이 세 열에만 나타나고 나머지는 측정값이라 knob 이 같으면 따라서 같다. 한쪽에만 있는 id 는
따로 센다 — 그건 임계 효과가 아니라 뱅크가 다른 설정으로 구워진 것이다.

```bash
$PY_LC eval/diff_bank_variants.py \
    --base out/parkour/hole_bank_g1probe_off/bank.json \
    --new  out/parkour/hole_bank_g1probe_on/bank.json \
    --base_name "clear 0.10 absolute" --new_name "clear = margin + 0.3 x src_p10"
```

| 인자 | 기본값 |
|---|---|
| `--base` / `--new` | (필수) |
| `--base_name` / `--new_name` | `base` / `new` |
| `--top` | `25` |
| `--out_json` | `None` |

env: 아무거나 (json). 렌더 0회.

### `diff_f7_banks.py`

F7 시간축 절단 on/off 뱅크를 행 단위로 본다. ① **회귀 검사** — `hold_from` 이 비었거나
`num_frames` 인 행은 off 뱅크와 공유 열 전부가 문자 동일해야 한다. ② **변경 목록** —
`hold_from < num_frames` 인 행의 `status`/`binding`/`path_len_u`/`tau_max` 변화표.

```bash
$PY_LC eval/diff_f7_banks.py --video parkour \
    --off_bank hole_bank_f7_off --on_bank hole_bank_f7_on --top 8
```

| 인자 | 기본값 |
|---|---|
| `--video` | `parkour` |
| `--output_root` | `<CT>/out` |
| `--off_bank` / `--on_bank` | `hole_bank_f7_off` / `hole_bank_f7_on` |
| `--num_frames` | `49` |
| `--top` | `10` |

env: 아무거나 (csv).

---

## 4. 조사 (`ablate_*`, `attribute_*`, `probe_*`, `rank_*`, `slice_*`)

### `ablate_vlm_hole_perception.py`

VLM 이 렌더 구멍(magenta)을 **그림으로** 보는지, 프롬프트 텍스트의 `coverage` 숫자만 읽는지
가른다. 조건 9종 — `A_full` `B_no_num` `C_no_img` `D_conflict`(숫자만 1-x 로 뒤집음)
`E_paint`(A 가 고른 타일에 가짜 magenta) `F_paint_nonum` `G_shuffle` `H_neutral`(traj 전용)
`I_temp10`. 읽는 법: D 에서 안 움직이고 F 에서 움직이면 그림을 본다 / F 에서도 칠해진 타일을
계속 고르면 board 는 장식이다 / G·H·I 에서도 안 움직이면 지각이 아니라 prior 다.

```bash
$PY_V4 eval/ablate_vlm_hole_perception.py --turn select --videos camel --draws 5
```

| 인자 | 기본값 |
|---|---|
| `--root` / `--output` | `<CT>` / `<CT>/out/ablation` |
| `--videos` | `['camel', 'avocado-slice']` |
| `--turn` | `both` (`traj` / `select` / `both`) |
| `--draws` | `3` |
| `--temperature` | `0.1` |
| `--board_columns` / `--preset_columns` | `9` / `3` |
| `--api_base` | `http://127.0.0.1:22002/v1` |
| `--model` | `Qwen/Qwen3-VL-30B-A3B-Instruct` |

비고: **vLLM 을 먼저 올려야 한다** (22002). 상시 기동은 금지 — 단계 직전에 올리고 끝나면 내린다.
`--draws 1` 로 정합/불일치를 판정하지 말 것 (같은 조건에서 답이 흔들리는 것 자체가 결과다).
env: 미확정 (`lbm.vlm.VLMClient` + imageio).

### `attribute_g1_hits.py`

G1 히트가 **무슨 표면**에 대해 걸렸는지 귀속한다. 두 축 — ① seg instance(`SegInstances.
frame_masks`, 없으면 `unsegmented`), ② 기하(world 로 되올린 뒤 `ground.ground_z` 와 비교해
`ground` / `below_cam` / `level`). `judge_frame` 은 `viz.viz_g1_collision` 에서 그대로 import 한다
(G1 식이 세 군데로 갈라지면 안 된다).

```bash
$PY_V4 eval/attribute_g1_hits.py --video parkour --bank_dir hole_bank_k6_d121 \
    --worst_n 20 --behind_src_frames 49
```

| 인자 | 기본값 |
|---|---|
| `--video` | (필수) |
| `--bank_dir` | `hole_bank` |
| `--variant` / `--worst_n` | `None` / `10` |
| `--behind_src_frames` | `49` |
| `--behind_margin_frac` / `--behind_clear_frac` | `0.02` / `0.1` |
| `--behind_radius_px` | `2` |
| `--ground_band_u` | `0.1` |
| `--output_root` / `--eval_data` / `--vista4d_root` / `--seg_root` / `--seg_static_root` | `None` / `lbm.cloud` 기본값 |

비고: `--behind_clear_frac` 기본이 **0.1** 로 `audit_bank_geometry.py`(0.0) 와 다르다 — 두 표를
나란히 놓을 때 임계를 맞출 것. 출력은 stdout 집계뿐. env: 미확정 (렌더 0회, `scene_graph.io` 가
EXR 을 읽는다).

### `probe_g1_reference.py`

G1 임계의 **기준 거리**를 고른다. ① `g1_src` — 소스 카메라 자신을 **G1 그 metric 으로**(3D
최단거리가 아니라 z-depth 차) 재서 얻는 여유. 제안식은 `clear_frac = margin_frac + β·g1_src`
(β<1 이면 소스가 정의상 통과한다). ② 후보 기준거리 R 로 나눈 `g1_src / R` 이 씬 사이에서
안 흔들리는지를 변동계수로 순위 매긴다.

```bash
CUDA_VISIBLE_DEVICES=0 $PY_V4 eval/probe_g1_reference.py --num_videos 14 --beta 0.3
```

| 인자 | 기본값 |
|---|---|
| `--videos` / `--num_videos` | `[]` / `12` |
| `--behind_src_frames` | `49` |
| `--margin_frac` / `--clear_frac` / `--radius_px` | `0.02` / `0.1` / `2` |
| `--beta` | `0.3` |
| `--device` | `cuda` |

비고: `g1_src` 는 배포 `time_match=True` 와 맞추려고 **static 채널**로 잰다 (동적 채널은 소스
pose 에서 항상 공집합). env: `vista4d` (`CloudRenderer`).

### `probe_near_depth_repeat.py`

`near_depth` 열이 뱅크 재생성 사이에서 얼마나 흔들리는지를 **뱅크와 똑같은 경로로** 재현한다
(`picks = unique(linspace(0, num_frames-1, verify_frames).round())`, `near(f) =
percentile(depth[valid], near_pct)/S`, 열 = picks 최솟값). `near_depth` 는 depth 의 1 백분위라
**렌더 해상도가 안 맞으면 통째로 움직인다** — 타일 크기를 뱅크와 맞출 것.

```bash
CUDA_VISIBLE_DEVICES=0 $PY_V4 eval/probe_near_depth_repeat.py --video camel \
    --variant_ids dyn_0__orbit_left_arc__hole0.2 --repeats 5
```

| 인자 | 기본값 |
|---|---|
| `--video` / `--output_root` / `--bank_dir` | `camel` / `out` / `hole_bank` |
| `--variant_ids` | (필수, nargs+) |
| `--repeats` | `5` |
| `--verify_frames` | `13` |
| `--near_pct` | `1.0` |
| `--tile_height` / `--tile_width` | `360` / `640` |

env: `vista4d` (`CloudRenderer`, `cloud.npz` 필요 — D178 이후 씬은 cloud 가 없을 수 있다).

### `probe_static_sdf.py`

정적 점군만으로 SDF 가 되는지 실측한다 (조사용, 파이프라인에 안 걸려 있다). 세 조건을 같은
격자에서: `A` frame0 한 장 / `B` 소스 13프레임 동적 포함 / `C` 13프레임 동적 제외. **판정 기준은
unknown 비율이 아니다** (격자를 넓히면 얼마든지 는다) — ① 뱅크 카메라가 실제로 지나가는 자리에서
라벨이 결정되나, ② SDF 판정이 G1/G5 판정과 일치하나.

```bash
CUDA_VISIBLE_DEVICES=1 $PY_V4 eval/probe_static_sdf.py --video avocado-slice --grid 256
```

| 인자 | 기본값 |
|---|---|
| `--video` | `camel` |
| `--grid` / `--pad_frac` | `256` / `0.1` |
| `--src_frames` | `13` |
| `--margin_frac` | `0.02` |
| `--obb_floor` | `0.1532` |
| `--check_poses` | `2000` |
| `--device` | `cuda` |
| `--save` / `--no_save` | off |
| `--eval_data` | `/data1/cympyc1785/data/Vista4D-Eval-Data` |
| `--vista4d_root` | `.../video_generation/models/Vista4D` |

env: 미확정 (torch + `lbm.gates` + scipy — vista4d 계열).

### `probe_tau_divisor.py`

τ 분모 후보 6종을 같은 뱅크에 얹는다. 자의적인 지점 두 개 — ① z-depth 냐 점까지 거리냐
(`scene_graph/scale.py` 의 `S` 는 이미 ray length 라 리포 안에 게이지가 두 개 있다), ② frame0
한 장이냐 소스 전 프레임이냐. **판정 기준은 하나다: hole 사다리의 같은 단에서 두 씬의 τ 가 같은
값으로 나오나** (비가 1 에 가까울수록 좋은 분모).

```bash
$PY_V4 eval/probe_tau_divisor.py --videos camel avocado-slice --stride 2
```

| 인자 | 기본값 |
|---|---|
| `--videos` | `['camel', 'avocado-slice']` |
| `--output_root` | `None` |
| `--stride` | `4` |
| `--eval_data` / `--vista4d_root` / `--seg_root` / `--seg_static_root` | `lbm.cloud` 기본값 / `None` |

비고: 읽는 뱅크가 **`hole_bank` 로 하드코딩**돼 있다 (`<output_root>/<video>/hole_bank/bank.csv`,
`:130`). 다른 세대 뱅크를 보려면 코드를 고쳐야 한다. 출력은 stdout 만. env: 미확정 (numpy +
`scene_graph.io`).

### `probe_trumans_blend.py`

TRUMANS `.blend` 안에 실제로 뭐가 들어 있는지 headless Blender 로 덤프한다 — 오브젝트 이름,
씬 이름, armature/action. LBM 이 `asset_id` **이름으로** 오브젝트를 찾고 `Scene_<scene_id>` 규약으로
씬을 찾기 때문에, 이걸 먼저 확인하지 않으면 어댑터 JSON 이 전부 빗나간다. `strings` 로는 그
문자열이 오브젝트 이름인지 머티리얼 이름인지 알 수 없다.

```bash
BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BL <path>.blend --background --python eval/probe_trumans_blend.py -- \
    --out /data1/cympyc1785/LatentCamVid/tmp/trumans/probe.json
```

| 인자 | 기본값 |
|---|---|
| `--out` | `/data1/cympyc1785/LatentCamVid/tmp/trumans_blend_probe.json` |
| `--show` | `60` |

비고: Blender 안에서 도므로 이 리포의 다른 모듈을 import 하지 않는다(`bpy` 만). `--` 뒤 인자만
이 스크립트 몫이다. env: Blender 4.5.9 번들 python.

### `probe_wall_planes.py`

`relations.find_wall_planes` 가 찾은 벽 평면을 되살려 **반공간 게이트로 쓸 수 있는지** 판정한다
(지금은 `(normal, d)` 를 버리고 개수만 남긴다). 평면마다 inlier 비율·법선·원점거리, 정적 점군이
양쪽에 어떻게 갈리는지, **소스 카메라 49대가 어느 쪽인지**(safe side 정의), 뱅크 카메라가 safe
side 밖으로 나가는 비율을 낸다. 판정 규칙은 D47/D49/D51/D55/D56 과 같다 — 원본 footage 를
기각하는 임계는 버그다.

```bash
$PY_V4 eval/probe_wall_planes.py --videos camel avocado-slice
```

| 인자 | 기본값 |
|---|---|
| `--videos` | `['camel', 'avocado-slice']` |
| `--output_root` / `--bank_dir` | `out` / `hole_bank` |
| `--max_walls` | `3` |
| `--max_points` | `400000` |

env: 미확정 (numpy + `scene_graph.relations`). 출력은 stdout 만.

### `rank_subject_motion.py`

split 의 각 entry 를 **subject 각이동량** `ang = path_len_u / d_ref` 로 정렬하고 상위만 새 split
으로 뽑는다. 절대 이동량이 아니라 각인 이유 — `subject_in_frame` 은 화면 좌표 지표라 분모가 시청
거리여야 한다. `center_drift_u / d_ref`(순 변위)도 같이 찍는다: `path` 만 크고 `drift` 가 작으면
제자리 왕복이라 추종 난이도가 다르다.

```bash
$PY_V4 eval/rank_subject_motion.py \
    --split /data1/.../latentcam_da3_k6_d77/seg_list_vista4d_train.txt \
    --corpus /data1/.../latentcam_da3_k6_d77 --cloud_root out \
    --min_ang 1.0 --max_per_scene 12 \
    --out_split results/20260901_highmotion/seg_list_vista_highmotion.txt
```

| 인자 | 기본값 |
|---|---|
| `--split` / `--corpus` / `--cloud_root` | 전부 필수 |
| `--metric` | `ang_drift` (`ang_path` / `ang_drift`) |
| `--moving_only` / `--no_moving_only` | on |
| `--min_ang` | `1.0` |
| `--max_per_scene` | `0` (무제한) |
| `--out_split` | `None` |

비고: subject 는 entry 마다 다르다 (`eval_subject_in_frame.py:entry_subject` 와 같은 규칙).
env: 아무거나 (json/numpy).

### `rank_traj_text_match.py`

pred 궤적이 **텍스트대로 움직였는지**를 GT 대비로 잰다 (캡션이 GT 궤적에서 나온 문장이므로
"텍스트 준수" = "GT 와 같은 방향·크기·회전"). 전부 frame0 앵커를 푼 rel pose 위에서:
`dir_err_deg`(순변위 방향각) / `len_ratio`(경로길이 비) / `rot_err_deg`(마지막 R 의 측지각) /
`rot_mag_deg`. 충돌률·subject_in_frame 은 **지시 불이행**을 못 잡는다 — 구도가 우연히 맞으면
통과하기 때문.

```bash
$PY_V4 eval/rank_traj_text_match.py \
    --eval_dir "d124=/.../20260904_185554_vista4d_d121_molmo2__epoch100__seed42" \
    --eval_dir "gd_style=results/20260906_d156_gendop_d121/eval_dir_gendop_rgbd_gdstyle_raw_slerp_noscale" \
    --rank_pair d124=gd_style --require_moving --top 20
```

| 인자 | 기본값 |
|---|---|
| `--eval_dir` | (필수, `LABEL=DIR` 반복) |
| `--ref_eval_dir` | `None` |
| `--name_prefix` | `vista4d` |
| `--corpus_root` | `.../latentcam_da3_k6_d121` |
| `--static_u` | `0.02` — 이하면 `dir_err` 를 NaN 으로 두고 정렬에서 뺀다 |
| `--rank_pair` | `None` (`OURS=THEIRS`) |
| `--ours_max_dir_deg` | `30.0` |
| `--len_lo` / `--len_hi` | `0.5` / `2.0` |
| `--require_moving` | off |
| `--top` / `--out` | `20` / `None` |

env: 아무거나 (json/numpy).

### `slice_subject_in_frame.py`

이미 낸 `subject_in_frame.json` 을 **부분집합으로 다시 잘라** 표만 뽑는다 — 행마다 프레임별
subject 중심(`centers`)이 실려 있어서 `in_frame_at()` 을 다시 걸면 `center_box` 스윕까지 그대로
나온다. **새 GPU 시간 0.**

```bash
$PY_V4 eval/slice_subject_in_frame.py \
    --json results/20260901_gendop_dynpose_val/subject_in_frame.json \
    --group_split highmotion=results/20260901_highmotion/seg_dynpose_val_hm.txt
```

| 인자 | 기본값 |
|---|---|
| `--json` | (필수, 반복 가능 — `(scene, entry, arm)` 중복은 **먼저 준 파일**을 남긴다) |
| `--group_split` | `None` (`LABEL=FILE` 여러 개) |
| `--group_scene` | off |
| `--group_meta` | `None` (예: `aim` → `prompts.json` 필드값별) |
| `--corpus_root` | `.../latentcam_da3_k6_d121` |
| `--dataset` | `vista4d` |
| `--sweep_lo/_hi/_step` | `0.1` / `1.0` / `0.1` |
| `--out` | `None` |

비고: split 포맷은 `<dataset>/<scene>/<idx>` 한 줄씩 — `rank_subject_motion.py` 산출물이 그대로
들어간다. env: 아무거나 (json/numpy).

---

## 5. 스윕 (`sweep_*`)

### `sweep_caption_thresholds.py`

DataDoP 분절기 임계를 Vista 카메라(49프레임 @ fps 10)에 맞게 고르기 위한 sweep. LLM 호출 없음 —
preset 이름이 정답인 엔트리에서 config 별 outline 을 직접 찍어 본다. config 6종:
`datadop`(120, 30.0, 0.02, 0.4, 0.005, 18) / `resample120_f10` / `native_f10` / `native_f10_w7` /
`native_f10_t008` / `native_f10_t004`.

```bash
$PY_GD eval/sweep_caption_thresholds.py --only camel/zoom-out --configs native_f10_t008
```

| 인자 | 기본값 |
|---|---|
| `--only` | `None` (`scene/name` 또는 `scene`) |
| `--configs` | `list(CONFIGS)` = 위 6종 전부 |

비고 세 개(전부 `processing/segmentation.py` 실측): ① `segment_rigidbody_trajectories` 는 넘긴
`smoothing_window_size`/`min_chunk_size` 를 **15/10 으로 덮어쓴다** — 인자는 translation 분절에만
먹는다. ② 그 뒤 while 루프가 세그먼트 수 ≤4 가 될 때까지 window/min_chunk 를 5씩 키운다.
③ fps 는 translation 임계에만 작용한다 (`to_euler_angles` 는 fps 를 안 쓴다).
`ROOT` 는 `/data1/cympyc1785/LatentCamVid/DATA/Vista4D-Eval-Data/eval_data` 로 하드코딩인데
`DATA` 는 `/data1/cympyc1785/data` 심링크라 다른 스크립트와 같은 곳이다. env: `GenDoP`.

### `sweep_collision_margin.py`

뱅크 G1 의 margin 을 **스윕해서** 어느 값에서 몇 변이가 죽는지 곡선으로 낸다. mesh 격자는 EDT 라
임계를 바꾸는 데 재굽기가 필요 없다 (궤적당 최소 clearance 를 한 번 재면 나머지는 비교 연산).
`--audit` 로 `raycast_probe_*.json` 을 주면 Blender raycast 게이트(`--min_clearance` /
`--probe_distance`)도 같은 형식으로 같이 낸다.

```bash
$PY_V4 eval/sweep_collision_margin.py --video tru_1d076f8c_a00_s3f0k6 \
    --margins 0.00 0.02 0.04 0.08 0.15 0.25 0.35 0.50 \
    --out /data1/cympyc1785/LatentCamVid/tmp/d118/sweep_a00.json
```

| 인자 | 기본값 |
|---|---|
| `--video` | `[]` (nargs*) |
| `--output_root` / `--bank_dir` | `out_trumans` / `hole_bank_k6_d116` |
| `--mesh` / `--no_mesh` | on |
| `--audit` | `[]` (raycast JSON 경로들) |
| `--margins` | `0.00 0.02 0.04 0.06 0.08 0.12 0.16 0.20 0.25 0.30 0.35 0.50` |
| `--out` / `--plot` | `''` / `''` |

비고 — **함정 두 개가 docstring 에 적혀 있다.** ① `poses.npz` 는 이미 게이트를 통과한 pose 다.
그러니 `통과` 열은 "임계를 올리면 이만큼 살아남는다"가 아니라 "지금 궤적이 표면에서 얼마나
떨어져 있나"이고, `solved유지` 는 재굽기 생존율의 **하한**이다 (실제 손실은 탈락이 아니라 이동량
감소로 나타난다). ② EDT 격자가 voxel 0.05 m 양자화라 0 이 아닌 최솟값이 0.05 다 — 현재 임계
`0.02·S ≈ 0.04 m` 는 **한 번도 구속하지 않는다.** 0.05 미만 구간 곡선은 읽지 말 것.
읽을 열은 `binding`(정적/동적 채널이 다른 거리에서 죽는다)과 `solved유지`. env: `vista4d`
(numpy/scipy, GPU 불필요).

---

## 6. GenDoP baseline — **GENDOP_USAGE.md 를 볼 것**

아래 5개는 같은 폴더의 [`GENDOP_USAGE.md`](./GENDOP_USAGE.md) 가 배선·크기 규약(`--rescale` /
`--scale_token`)·텍스트 2종·손잡이 표까지 근거(파일:줄)와 함께 다룬다. 여기서는 역할 한 줄만
적는다. 드라이버는 `exec/run_gendop_eval.py` (`--stage {inputs,infer,evaldir,score,all}`).

| 파일 | 역할 | env |
|---|---|---|
| `dynpose_gendop_inputs.py` | stage `inputs` — dynpose 코퍼스를 GenDoP `text_rgbd` 레이아웃(rgbd 심링크 + 캡션 미러 루트)으로 | `GenDoP` (vista4d 는 cv2 가 `.exr` 에 None) |
| `gendop_release_infer.py` | stage `infer` — 릴리즈 ckpt 로 `<set>__<scene>__<name>.npz` 생성 | `GenDoP` |
| `gendop_preds_to_eval_dir.py` | stage `evaldir` — npz → latentcam eval 폴더 규약 (좌표·앵커·리샘플·rmax 4단계) | `index_pick` 은 아무거나 / `gendop_slerp` 은 `GenDoP` |
| `gendop_release_eval.py` | 캡션 왕복 일치도 (생성 궤적 재태깅 ↔ 원 태그) | `GenDoP` |
| `eval_dir_to_gendop_npz.py` | 역방향 — 우리 eval 폴더를 GenDoP npz 로 (지표를 한 harness 로 통일) | 아무거나 (numpy) |

**최소한 이것만은 기억할 것** (근거는 GENDOP_USAGE.md §3): 크기 손잡이가 둘이고 원본 레포 방식은
**둘 다 끈 것**이다 — `--no_scale_token` + `--raw`(드라이버) / `--no_rescale`(변환기). 기본값
on/on 은 예전 런과 비트동일하게 두려고 남긴 것이지 권장값이 아니다. 충돌률·subject_in_frame 처럼
절대 크기에 반응하는 지표는 켜 두면 GenDoP 이 아니라 GT 를 재게 된다.

---

## 7. 기타

### `pick_warp_sample_scenes.py`

뱅크를 다 구운 뒤 **눈으로 확인할 표본 씬**을 고른다 — 움직이는 subject N편 + 안 움직이는 N편.
두 버킷이 **다른 게이트를 시험**하기 때문이다: 움직이는 쪽은 `track_*` 가 켜져 추종 실패·조준
흔들림이 터지고, 안 움직이는 쪽은 `track_*` 가 `--track_min_drift_u` 에서 잘려 object-centric
preset 만 남아 hole 과 벽 뚫기가 터진다. 선택은 **결정론적**(RNG 없음) — 재굽기 전후를 짝지어
보기 위함. 1순위 preset 다양성, 2순위 drift, 3순위 이름.

```bash
$PY_LC eval/pick_warp_sample_scenes.py --output_root out_dynpose --bank_dir hole_bank_d129 \
    --num_moving 3 --num_static 3 --table \
    --out /data1/cympyc1785/LatentCamVid/tmp/d129/warp_sample.json
```

| 인자 | 기본값 |
|---|---|
| `--bank_dir` | (필수) |
| `--output_root` | `out` |
| `--videos` | `['all']` |
| `--num_moving` / `--num_static` | `2` / `2` |
| `--drift_thresh` | `0.05` (`sample_camera_bank.py --track_min_drift_u` 와 같은 값) |
| `--picked_only` | off |
| `--bucket_by` | `drift` (`drift` / `preset`) |
| `--table` / `--out` | off / `''` |

비고: anchor 선택 규칙이 `run_preset_warp_max_shard.sh` 와 같아야 한다(`dyn_0` 우선, 없으면
solved 행이 가장 많은 anchor). 여기서 분류한 subject 와 릴에 찍히는 subject 가 다르면 버킷이
거짓말이 된다. env: 아무거나 (json/csv).

### `rescale_eval_preds_to_ref.py`

latentcam pred 궤적을 **ref 의 `rmax` 에 맞춰** 다시 써서 "모양만 남기는" 대조 arm 을 만든다.
CLaTr 은 스케일에 민감한데(`trajectory_dataset.py:43 self.standardize = False`) GenDoP arm 은
rmax 를 GT 에서 받으므로, 그대로 나란히 놓으면 크기 축에서 한쪽만 정답을 미리 본 비교가 된다.
세 열(`k6 raw` / `k6 rmaxGT` / `gendop rmaxGT`)을 나란히 읽으면 축이 분리된다.

```bash
$PY_LC eval/rescale_eval_preds_to_ref.py \
    --eval_dir /data1/.../eval_my/vista4d_pgt_k6__last__full500 \
    --out_dir  /data1/.../eval_my/vista4d_pgt_k6__last__full500_rmaxGT
```

| 인자 | 기본값 |
|---|---|
| `--eval_dir` / `--out_dir` | (필수) |
| `--skip_static_ref` / `--no_skip_static_ref` | **on** |

비고: GT 가 정지(rmax 0)인 entry 는 rescale 이 pred 를 항등으로 뭉갠다 — "정지를 맞혔다"가 아니라
정보가 지워진 것이라 기본으로 빼고 개수를 센다. ref/caption 은 손대지 않고 복사한다
(`render_pred_depth_warp.py` 가 arm 끼리 ref 를 `max|diff| < 1e-5` 로 대조한다).
env: 아무거나 (numpy).

### `sanitize_director_focus.py`

LBM Director 산출물의 `focus_ids` 에서 **유령 asset** 을 뺀다. 문제는 해석 안 되는 id(`sofa` 등)가
아니라 — 그건 Cinematographer `_find_object` 가 `continue` 로 건너뛴다 — **asset 은 아닌데 blend
에는 실재하는 이름**이다. 그건 focus AABB 에 합쳐져 카메라를 물러나게 한다 (실측: `floor` 하나로
maxdim 2.22 m → 6.61 m, ×3.0). 스칼라 `primary_focus_id` 는 건드리지 않고 `primary_hits` 로만
보고한다.

```bash
# 기본이 dry run — 뺄 id 만 본다
$PY_LC eval/sanitize_director_focus.py \
    --output_root ../Look-Before-Move/Director/output --run_glob 'trumans_c49_w*' \
    --blend_objects /data1/cympyc1785/LatentCamVid/tmp/lbm/blend_objects.json
# 실제로 고친다 (원본은 <파일>.bak)
... --no_dry_run
```

| 인자 | 기본값 |
|---|---|
| `--output_root` / `--blend_objects` | (필수) |
| `--run_glob` | `trumans_c49_w*` |
| `--dry_run` / `--no_dry_run` | **`dry_run=True`** (`set_defaults`) |
| `--backup_suffix` | `.bak` |

비고: `eval/` 에서 **원본 파일을 고치는 유일한 스크립트**다. env: 아무거나 (json/shutil).

### `smoke_micro_ops.py`

micro-adjust 연산 14종을 VLM 없이 전부 적용해 게이트 결과를 표로 낸다. 필요한 이유 — camel
end-to-end 에서 VLM 이 round 0 에 `done:true` 를 내버려 `lbm/ops.py` 가 **한 번도 실행되지
않았다**. 재는 것은 ① `apply_op` 의 clamp/delta 계산(`CLAMPED` 열), ② 적용된 pose 의 게이트 통과
여부와 걸린 게이트. 출력 `<out>/<video>/trace/smoke_micro_ops.json`.

```bash
CUDA_VISIBLE_DEVICES=1 $PY_V4 eval/smoke_micro_ops.py --video camel --start B6 --allow_zoom
```

| 인자 | 기본값 |
|---|---|
| `--start` | `None` (board label. 없으면 1등 후보) |
| 나머지 전부 | **`lbm.loop.build_parser()` 에서 상속** — 게이트 임계·렌더 타일 크기를 루프와 공유 |

비고: 게이트 임계를 손으로 다시 적지 않는 것이 설계다 (테스트만 통과하고 루프는 떨어지는 상황을
막는다). 그래서 `--video` / `--output_root` / `--allow_zoom` 같은 인자는 `loop.py` 쪽 정의를 볼 것.
env: `vista4d` (`lbm.render.CloudRenderer`).

### `time_vista_stages.py`

뱅크 경로 전 단계(`graph → pcd → bank → fit → bankemit`)를 **격리된 `--out_root`** 로 처음부터
다시 돌려 벽시계 시간을 잰다. 격리가 핵심 — 기존 `out/<video>/` 를 건드리면 학습에 쓰고 있는
`scene_graph.json`/`cloud.npz`/`hole_bank_k6` 가 덮어써진다. 표에 `s/camera` 열을 같이 내고
(단계 시간은 영상 길이가 아니라 **뱅크 variant 수**에 붙는다) `graph`/`pcd` 는 고정비로 따로 표시한다.

```bash
$PY_V4 eval/time_vista_stages.py --videos camel --cuda 0 --stages graph,pcd,bank
$PY_V4 eval/time_vista_stages.py --videos camel --dry_run
```

| 인자 | 기본값 |
|---|---|
| `--videos` | (필수, nargs+) |
| `--stages` | `graph,pcd,bank,fit,bankemit` |
| `--out_root` | `/data1/cympyc1785/LatentCamVid/tmp/vista_timing` |
| `--bank_dir` | `hole_bank_k6` |
| `--cuda` | `"0"` (→ `CUDA_VISIBLE_DEVICES`) |
| `--log_dir` | `/data1/cympyc1785/LatentCamVid/tmp/vista_timing/logs` |
| `--dry_run` / `--no_dry_run` | off |

비고 두 개. ① 하위 프로세스는 `PY = sys.executable` 로 띄운다 — **이 스크립트를 부른 인터프리터가
그대로 각 stage 에 쓰인다.** 그러니 vista4d 로 부를 것. ② `bank`(sample_camera_bank)는 원본 뱅크
`out/<video>/bank/` 를 쓰고 `fit` 이 그걸 읽어 `hole_bank_k6/` 에 쓴다 — `--bank_dir` 가 어느
stage 에 걸리는지(`SOURCE_BANK_STAGES` vs `FIT_BANK_STAGES`) 헷갈리면 fit 이 자기 출력을 입력으로
읽는다. **docstring 예시의 `--cuda 6` 은 쓰지 말 것.**

### `vista_blend_check.py`

`vista_to_lbm_demo.py` 가 구운 `.blend` 가 **소스 영상과 같은 씬인지** headless 로 확인한다.
① 소스 카메라 재렌더 (scene graph 의 `cameras.cam_centers_g` + `K` 로 세워 원본 프레임과 대조 —
여기가 틀리면 `T_gw`/`u_meters`/K 중 하나가 어긋난 것이고 아래 전부가 무의미하다), ② 레이캐스트
적중률 (점군에는 면이 없어 항상 `hit=False` 가 나오고 그러면 LBM 의 가림 게이트가 **조용히
꺼진다** — 여기서 `hit=True` 가 나와야 어댑터가 그걸 고친 것이다), ③ 오브젝트 이름·면수·애니메이션 표.

```bash
BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BL out/lbm_demos_vista/camel/camel.blend --background \
    --python eval/vista_blend_check.py -- \
    --graph out/camel/scene_graph.json --vista out/lbm_demos_vista/camel/_vista.json \
    --frames 0,24,48 --out /data1/cympyc1785/LatentCamVid/tmp/vista/camel_check
```

| 인자 | 기본값 |
|---|---|
| `--graph` / `--vista` | (필수) |
| `--frames` | `0,24,48` |
| `--width` / `--height` | `0` / `0` (0 = 원본) |
| `--out` | `/data1/cympyc1785/LatentCamVid/tmp/vista_blend_check` |

비고: `--` 뒤 인자만 이 스크립트 몫이다 (Blender 규약). env: Blender 4.5.9 번들 python (`bpy`,
`mathutils`).

---

## 부록 — 읽을 때 반복해서 걸리는 것들

- **`bank_dir` 기본값이 스크립트마다 다르다.** `bank`(audit_bank_geometry) / `hole_bank`
  (audit_bank_status, attribute_g1_hits, probe_near_depth_repeat, probe_wall_planes,
  probe_tau_divisor 는 하드코딩) / `hole_bank_k6`(time_vista_stages) / `hole_bank_k6_d77`
  (compare_aim_timing) / `hole_bank_k6_d99`(audit_tau_axes) / `hole_bank_k6_d116`
  (sweep_collision_margin). 세대를 바꿀 때 전부 명시할 것.
- **`clear_frac` 기본값도 갈린다.** `audit_bank_geometry` 0.0 vs `attribute_g1_hits` /
  `probe_g1_reference` 0.1. 두 표를 나란히 놓으려면 맞춰야 한다.
- **`variant_id` 는 코퍼스 키가 아니다.** 재굽기해도 이름은 같고 knob/pose 만 바뀐다 —
  `diff_bank_variants.py` 의 조인도 그 전제 위에 있다. 세대를 섞어 조인하면 전부 "변경 없음" 으로
  통과하면서 틀린다.
- **감사 결과로 행을 지우지 않는다.** `audit_*` 는 열만 붙인다 (D39/D45 와 같은 이유 — 자를지는
  소비처가 정한다).
