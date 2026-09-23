# `fit/` — 파이프라인 스크립트 사용법

`fit/` 은 **원천 데이터를 카메라 뱅크와 학습 코퍼스로 바꾸는** 단계별 스크립트 모음이다.
다섯 하위 폴더가 그대로 단계 순서다:

```
ingest → graph → bank → caption → convert
```

- `ingest/` — 원천 데이터(Vista4D / DynPose-100K / TRUMANS `.blend`) → `recon_and_seg` + `seg_instances` 규약
- `graph/` — `scene_graph.json`, VLM 명사·인스턴스 지칭구
- `bank/` — preset routing → τ 사다리 → hole 사다리 fit → emit (카메라 뱅크 굽기)
- `caption/` — 뱅크 → 캡션 → 학습 코퍼스 / holdout split / `avg_scale`
- `convert/` — 뱅크를 Blender / Vista4D / DL3DV 포맷으로

## 실행 위치 (하드 룰)

**전부 CinemaTraj 루트에서 돌린다.**

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/dataset
```

이유는 스크립트가 자기 파일 경로에서 `CINEMATRAJ_ROOT` 를 역산해 `sys.path` 에 넣기 때문이다.
`fit/` 의 파일은 루트에서 **두 단계 아래**(`fit/<stage>/<file>.py`)라 `dirname()` 이 **세 겹**이다:

```python
CINEMATRAJ_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0, CINEMATRAJ_ROOT)
```

(`trumans_to_recon.py` · `trumans_lite_to_dl3dv.py` · `repair_recon_poses.py` ·
`vista4d_bank_to_dl3dv.py` 는 `HERE = dirname(abspath(__file__))` 를 먼저 잡고
`dirname(dirname(HERE))` 를 쓴다 — 같은 값이다.)

경로 자체는 절대경로로 풀리므로 다른 cwd 에서도 import 는 되지만, 스크립트의 **기본값**이
`out/` · `configs/` · `results/` 같은 상대 경로를 그대로 쓰는 곳이 여럿이라 루트가 아니면
산출물이 엉뚱한 데 떨어진다.

## 인터프리터

| 이름 | 경로 |
| --- | --- |
| latentcam | `/data1/cympyc1785/miniconda3/envs/latentcam/bin/python` |
| vista4d | `/data1/cympyc1785/miniconda3/envs/vista4d/bin/python` |
| vllm | `/data1/cympyc1785/miniconda3/envs/vllm/bin/python` |
| geocalib | `/data1/cympyc1785/miniconda3/envs/geocalib/bin/python` (`geocalib_gravity.py` 전용, kornia 필요) |
| Blender 내장 | `/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender` 가 `--python` 으로 실행 |

각 파일의 "주요 인자" 표 위 한 줄에 해당 스크립트의 env 를 적었다. 코드에서 판별이 안 되면 **미확정**이다.

## 공통 규칙

- **GPU 는 0~3 만 쓴다.** `CUDA_VISIBLE_DEVICES` 를 그 범위 밖으로 주지 않는다.
  (문서에 남아 있는 예시 중 `CUDA_VISIBLE_DEVICES=4` 이상은 과거 것이므로 따르지 말 것.)
- **임시 산출물은 `/data1/cympyc1785/LatentCamVid/tmp/` 아래.** `/tmp` 는 쓰지 않는다.
  몇몇 docstring 예시에 `/tmp/...` 가 남아 있는데(§비고에 표기) 그대로 복사하지 말고 `tmp/` 로 바꿔 쓴다.
- **vLLM 서버가 필요한 스크립트는 그 단계 직전에 서버를 올리고 끝나면 내린다.**
  대상: `graph/describe_instances_vlm.py`, `graph/extract_nouns_vlm.py`,
  `graph/trumans_vlm_action_tag.py`, `graph/extract_static_nouns.py`(`--source vlm` 경로).
  기동은 `bash exec/serve_qwen3vl.sh`, 기본 엔드포인트 `http://127.0.0.1:22002/v1`,
  모델 `Qwen/Qwen3-VL-30B-A3B-Instruct`. 상시 기동 금지.
- **Blender 를 부르는 스크립트**는 두 종류다.
  ① Blender 안에서 도는 워커 — `blender -b <blend> --python <script> -- <args>` 형태로 직접 띄운다
  (`trumans_export_mesh.py`, `trumans_scene_probe.py`, `trumans_blend_layout_worker.py`,
  `vista_blend_worker.py`, `trumans_first_pose_board.py`).
  ② Blender 를 `subprocess` 로 부르는 드라이버 — 일반 python 으로 띄우면 내부에서 ①을 실행한다
  (`build_trumans_mesh_grid.py`, `trumans_to_recon.py`, `bank_to_blender_poses.py`).
  Blender 는 `--python` 스크립트가 예외로 죽어도 **rc=0** 을 돌려주므로 종료코드로 성공 판정을 하면 안 된다.

## 뱅크 단계 내부 의존성

`fit/bank/` 안에서 서로 import 하므로 실행 순서가 정해져 있다 (화살표 = "왼쪽이 오른쪽을 import").

```
emit_bank        →  fit_hole_ladder   (FIT_TAU_MAX_SCALE, KNOB_RANGE, SHAPE_DEFAULTS, retime_info, truncate_hold)
emit_bank        →  sample_camera_bank (make_decision, register_external)
fit_hole_ladder  →  sample_camera_bank (FIT_TAU_MAX_SCALE, ROTATION_ONLY_PRESETS, behind_context,
                                        geometry_stats, make_decision, measure_trajectory,
                                        source_approach, source_elevation, source_g1_clear,
                                        source_obb_clear, source_standoff, register_external)
fit_hole_ladder  →  build_candidate_board (subject_track_volume)
sample_camera_bank → build_candidate_board (subject_track_volume)
```

즉 **실행 순서는 import 의 역방향**이다:

```
route_presets → sample_camera_bank (τ 뱅크 `bank/`) → fit_hole_ladder (hole 뱅크 `hole_bank*/`) → emit_bank (`canonical/`)
```

폴더 밖 의존도 몇 개 있다:

- `convert/bank_to_blender_poses.py` → `fit.bank.emit_bank` (`FIXED_FALLBACK`, `decision_from_variant`, `span_frac_for`)
- `caption/make_avg_scale_vista4d_ctxall.py` → `fit.ingest.trumans_to_recon` (`avg_scale_first_cam`)
- `ingest/build_trumans_mesh_grid.py` → `fit.convert.bank_to_blender_poses` (`BLENDER`, `chunk_paths`) — ingest 가 convert 를 참조한다
- `ingest/trumans_clip.py` → `fit.ingest.trumans_probe`
- `graph/trumans_vlm_action_tag.py` → `scripts.trumans_to_lbm_demo` (`read_actions`)

---

## `ingest/` — 원천 데이터 → recon/metadata 규약

| 파일 | 역할 |
| --- | --- |
| `dynpose_ingest.py` | DynPose-100K 대량 ingest 드라이버 (link→recon→nouns→metadata→sam3→dynmask, 샤딩) |
| `extend_dynpose_metadata.py` | VLM 명사(`vlm_nouns.json`)를 DynPose-LBM `metadata.csv` 에 행으로 추가 |
| `dynpose_dynamic_mask_from_seg.py` | dynpose recon 의 `dynamic_mask/` 를 SAM3 인스턴스 합집합으로 교체 (+정지 소품 강등) |
| `sam3_static_instances.py` | 정적 명사 → SAM3 text PCS → `seg_instances_static/` |
| `geocalib_gravity.py` | GeoCalib 으로 씬 중력축 사이드카 `geocalib_gravity.json` |
| `prep_clip_49.py` | 임의 mp4 → 49프레임 Lite 규약 클립 (stride 솎기) |
| `trumans_probe.py` | TRUMANS 배포본 훑기 — 무엇이 있고 없는지 확정 |
| `trumans_clip.py` | TRUMANS 녹화에서 49프레임 소스 클립 창을 점수로 고르고 잘라낸다 |
| `trumans_scene_probe.py` | `.blend` 에서 사람 궤적 + 카메라 가능 자리를 광선으로 잰다 (Blender 워커) |
| `trumans_to_recon.py` | TRUMANS action 구간 1개 → `recon_and_seg` + `seg_instances` (Blender 렌더 드라이버) |
| `trumans_export_mesh.py` | `.blend` evaluated mesh 삼각형 → `mesh_gt.npz` (Blender 워커) |
| `build_trumans_mesh_grid.py` | `mesh_gt.npz` → G1 mesh 점유/EDT 격자 `mesh_grid.npz` (Blender 1회 + numpy 1회) |
| `trumans_blend_layout_worker.py` | `.blend` → LBM 이 요구하는 레이아웃 수치 JSON (Blender 워커) |
| `trumans_frame_shift_startup.py` | 모든 Blender 프로세스에서 TRUMANS 애니메이션 프레임을 당기는 startup 훅 (CLI 없음) |
| `build_trumans_metadata.py` | TRUMANS-Lite manifest → Vista 와 같은 스키마의 `metadata.csv` |
| `trumans_lite_to_dl3dv.py` | TRUMANS-Lite 뱅크 → latentcam DL3DV `da3` 온디스크 포맷 |
| `repair_recon_poses.py` | 덮어써진 `poses_aNN.npz` 를 배포본 `cameras.npz` 에서 되돌린다 |
| `vista_blend_worker.py` | Vista4D 점군 삼각형 메시 → 원본 LBM 이 읽는 `.blend` (Blender 워커) |

### `dynpose_ingest.py`

DynPose-100K 1만여 편을 `recon_and_seg` 규약으로 굽는 **단계별 드라이버**다. 기존 bash 래퍼가
영상 하나당 프로세스 하나를 띄워 DA3 로딩이 전체 시간의 절반을 먹던 것을, **프로세스당 DA3 를 한 번만
올리고 배정분을 전부 도는** 구조로 바꿨다. 단계 사이에 배리어가 있고 `--stage launch` 가 샤드를 부채꼴로 띄운다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY fit/ingest/dynpose_ingest.py --stage link
$PY fit/ingest/dynpose_ingest.py --stage recon --dry_run
$PY fit/ingest/dynpose_ingest.py --stage launch --launch_stage recon --gpus 0,1,2,3
# nouns 단계 직전에 vLLM 을 올리고, 끝나면 내린다
$PY fit/ingest/dynpose_ingest.py --stage launch --launch_stage nouns --workers 8
$PY fit/ingest/dynpose_ingest.py --stage metadata
$PY fit/ingest/dynpose_ingest.py --stage launch --launch_stage sam3 --gpus 0,1,2,3
$PY fit/ingest/dynpose_ingest.py --stage launch --launch_stage dynmask --workers 8
```

env: vista4d (recon 단계가 DA3 를 같은 프로세스에 올린다; 하위 샤드는 `sys.executable` 로 자기 자신을 다시 부른다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--stage` | (필수) | `link/recon/nouns/metadata/sam3/dynmask/launch` |
| `--launch_stage` | `recon` | `recon/nouns/sam3/dynmask` — `--stage launch` 가 띄울 단계 |
| `--src_root` / `--out_root` / `--done_root` | 상수 | 원본 / 새 eval_data / 재사용할 기존 코퍼스 |
| `--shards` | `0000-0011` | `dynpose-NNNN` 범위 |
| `--work_dir` | `<repo>/tmp/d169` | 작업 파일 |
| `--num_shards` / `--shard_id` | `1` / `0` | 워커 수 / 이 워커 인덱스 |
| `--gpus` / `--workers` | `None` / `1` | launch 전용 |
| `--num_frames` / `--height` / `--width` | 상수 | 클립 규약 |
| `--da3_model_id` / `--da3_process_res` | 상수 / `-1` | `<=0` 이면 width 로 맞춘다 |
| `--save_conf` | `True` | `--no_save_conf` |
| `--nouns_dir` / `--noun_frames` | `None`(=`<out_root>/vlm_nouns`) / 상수 | |
| `--api_base` / `--vlm_model` | 상수 | vLLM 엔드포인트 |
| `--noun_source` | `vlm` | `vlm/category` |
| `--skip_done` | `True` | `--no_skip_done` |
| `--dry_run` | `False` | |

> 비고: `--gpus` 예시가 docstring 에 `0,1,2,3,4` 로 남아 있다. 지금 규칙은 **0~3** 이다.
> `nouns` 단계는 vLLM 서버를 **밖에서 먼저** 띄워야 한다 (드라이버가 서버를 관리하지 않는다).

### `extend_dynpose_metadata.py`

`extract_nouns_vlm.py` 가 뽑은 명사를 DynPose-LBM `metadata.csv` 에 **같은 스키마로** 덧붙인다.
하류 `sam3_seg_instances.py` 가 csv 에서 영상별 명사를 읽고 없으면 assert 로 죽기 때문이다.

```bash
python fit/ingest/extend_dynpose_metadata.py --dry_run     # 무엇이 붙는지만
python fit/ingest/extend_dynpose_metadata.py               # 실제 기록 (백업 자동)
```

env: 미확정 (csv/json 만 쓴다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--csv` | 상수 | 대상 `metadata.csv` |
| `--nouns` | 상수 | `extract_nouns_vlm.py` 출력 |
| `--frame_mode` | `multi` | multi(4프레임)만 쓴다 |
| `--max_nouns` | `0` | 0 = 전부, >0 이면 앞에서 자른다 |
| `--dry_run` | off | 쓰지 않고 표만 |
| `--overwrite_existing` | off | 이미 있는 video 행도 덮어쓴다 |

### `dynpose_dynamic_mask_from_seg.py`

dynpose recon 의 `dynamic_mask/` 를 `seg_instances/<video>/masks.npz` 의 프레임별 합집합 png 로 덮어쓴다.
vista 배포본과 달리 dynpose 는 recon 을 직접 돌려 dynamic 트랙 단계가 없기 때문이다.
`--demote_static_objects` 를 켜면 실제로 안 움직이는 소품을 dynamic 에서 뺀다 (`scene_graph.json` 선행 필요).

```bash
python fit/ingest/dynpose_dynamic_mask_from_seg.py \
  --eval_data /data1/cympyc1785/data/DynPose-LBM --videos 00e9f728-... 015b197d-...
python fit/ingest/dynpose_dynamic_mask_from_seg.py \
  --eval_data /data1/cympyc1785/data/DynPose-LBM \
  --demote_static_objects --output_root out_dynpose --dry_run
```

env: 아무거나 (numpy + PIL)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--eval_data` | (필수) | |
| `--videos` | `None` | None = 전량 |
| `--demote_static_objects` | `False` | `--no_demote_static_objects` |
| `--noun_category` | 상수 | |
| `--output_root` | `out_dynpose` | `scene_graph.json` 위치 |
| `--demote_max_drift_ratio` | `0.05` | 강등 판정 drift 상한 |
| `--demote_max_path_ratio` | `0.15` | 강등 판정 path 상한 |
| `--demote_worn` | `False` | 착용물도 강등 |
| `--dry_run` | `False` | |

> 비고: docstring 예시에 `/tmp` 산출 경로가 없지만, 이 스크립트는 recon 폴더를 **제자리에서 덮어쓴다**.
> 먼저 `--dry_run` 으로 대상 편수를 확인할 것.

### `sam3_static_instances.py`

`video_generation/scripts/sam3_seg_instances.py` 의 **정적 버전**. keyword 출처가 저자 정답
`metadata.csv:dynamic` 이 아니라 `fit/graph/extract_static_nouns.py` 산출물이고, 출력이
`seg_instances_static/<video>/{meta.json,masks.npz}` 로 갈린다. 포맷·샤딩은 원본과 같다.

```bash
CUDA_VISIBLE_DEVICES=1 python fit/ingest/sam3_static_instances.py --videos camel avocado-slice
CUDA_VISIBLE_DEVICES=1 python fit/ingest/sam3_static_instances.py --num_shards 4 --shard_id 0
```

env: vista4d (GPU 1장)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--eval_data` | 상수 | |
| `--output_root` | `None` | None = `<eval_data>/eval_data/seg_instances_static` |
| `--vista4d_root` | 상수 | |
| `--static_nouns` | 상수 | `extract_static_nouns.py` 출력 |
| `--nouns_field` | `static` | `static/surface` |
| `--videos` | `None` | None = `static_nouns.json` 전체 |
| `--num_shards` / `--shard_id` | `1` / `0` | |
| `--skip_done` | `True` | `--no_skip_done` |
| `--max_dynamic_frac` | `0.5` | 동적 마스크와 겹치면 버린다 |
| `--save_vis` | `False` | |

### `geocalib_gravity.py`

GeoCalib 으로 영상별 중력축을 재서 `<out>/<video>/geocalib_gravity.json` 사이드카로 남긴다.
`scene_graph/gravity.py` 의 ground RANSAC 이 바닥이 크게 보이는 실외에서만 맞기 때문이다
(vista 53편 중 15편이 `camera_up_fallback`). `build_scene_graph.py --gravity_source geocalib` 가 이 파일을 읽는다.

```bash
CUDA_VISIBLE_DEVICES=3 /data1/cympyc1785/miniconda3/envs/geocalib/bin/python \
  fit/ingest/geocalib_gravity.py --videos camel snowboard
```

env: **geocalib** (kornia 가 vista4d/da3 에 없다). cv2 로 `video.mp4` 만 읽으므로 Vista4D 로더는 안 쓴다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--eval_data` / `--output_root` / `--geocalib_root` | 상수 | |
| `--videos` | `None` | None = `--video_list` 또는 전체 |
| `--video_list` | `None` | 한 줄 1편 |
| `--num_shards` / `--shard_id` | `1` / `0` | |
| `--skip_done` | `True` | `--no_skip_done` |
| `--num_frames` | `5` | 균등 5장 |
| `--trim_keep_frac` | `0.75` | 프레임 간 합의 상위 비율만 |
| `--weights` | `pinhole` | `pinhole/distorted` |
| `--device` | `auto` | |
| `--prior_focal` | `True` | `--no_prior_focal` |

### `prep_clip_49.py`

임의 mp4 를 stride 로 솎아 49프레임 클립으로 만든다. `recon_and_seg_single.py` 가 항상 center-slice 하므로
30fps 원본을 그대로 먹이면 1.6초밖에 안 되고 TRUMANS-Lite 뱅크의 `frame_step 3`(10fps, 4.9초)과 구간이 어긋난다.

```bash
python fit/ingest/prep_clip_49.py \
  --input "/data1/.../jogging woman.mp4" \
  --output /data1/cympyc1785/LatentCamVid/tmp/jogging_woman_49.mp4 --stride 3 --out_fps 10
```

env: vista4d (cv2 · imageio)

| 인자 | 기본값 |
| --- | --- |
| `--input` / `--output` | (필수) |
| `--num_frames` | `49` |
| `--stride` | `3` (0 이면 전 구간 균등) |
| `--start` | `0` |
| `--out_fps` | `10.0` |

> 비고: docstring 예시 출력 경로가 `/data1/...` 라 규칙에 맞지만, 다른 TRUMANS 스크립트 예시들은 `/tmp` 를 쓴다. `tmp/` 로 바꿔 쓸 것.

### `trumans_probe.py`

TRUMANS 배포본을 훑어 **무엇이 있고 없는지**를 확정한다. 카메라 파라미터가 매니페스트에 없지만
`smplx_result`(world) / `smplx_result_in_cam`(camera) 쌍이 있으므로 두 root 변환의 차이로 extrinsic 을 복원할 수 있다는 것을 확인하는 자리다.

```bash
python fit/ingest/trumans_probe.py
python fit/ingest/trumans_probe.py --recordings 2023-01-14@22-06-10 \
  --dump_frames /data1/cympyc1785/LatentCamVid/tmp/tru
```

env: vista4d

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--trumans_root` | 상수 | |
| `--recordings` | `None` | None = 무작위 표본 |
| `--num_recordings` | `12` | |
| `--seed` | `0` | |
| `--dump_frames` | `None` | 프레임 PNG 폴더 |
| `--dump_count` / `--dump_stride` / `--dump_max` | `4` / `100` / `3` | |

> 비고: docstring 예시가 `--dump_frames /tmp/tru`. `tmp/` 로 바꿀 것.

### `trumans_clip.py`

TRUMANS 30fps 연속 녹화(1,500~3,400 프레임)에서 49프레임 창 하나를 **점수로** 골라 mp4 로 자른다.
손으로 고르면 재현이 안 되므로 창 점수표(`--rank`)와 action label·카메라 회전량을 같이 찍어
`metadata.csv` 행을 바로 쓸 수 있게 한다. `--identify` 는 클립 mp4 에서 녹화/시작 프레임을 역산한다.

```bash
python fit/ingest/trumans_clip.py --list_meta
python fit/ingest/trumans_clip.py --recording 2023-01-14@22-33-09 --rank
python fit/ingest/trumans_clip.py --recording 2023-01-14@22-33-09 --start 431 \
  --out /data1/cympyc1785/LatentCamVid/tmp/trumans_bedroom_w431.mp4
```

env: vista4d

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--trumans_root` | 상수 | |
| `--recording` | `None` | |
| `--start` | `0` | 창 시작 프레임 |
| `--num_frames` | `49` | |
| `--stride` | `1` | |
| `--out` | `/data1/cympyc1785/LatentCamVid/tmp/trumans_clip.mp4` | |
| `--rank` | `False` | 창 점수표만 찍고 끝 |
| `--step` / `--top` | `30` / `15` | `--rank` 창 간격 / 표시 개수 |
| `--list_meta` | `False` | |
| `--identify` | `None` | 클립 mp4 → 녹화/시작 프레임 |

> 비고: `--out` 기본값은 규칙에 맞는 `tmp/` 인데 docstring 예시는 `/tmp/...` 다. 기본값 쪽을 따를 것.
> `fit.ingest.trumans_probe` 를 import 한다.

### `trumans_scene_probe.py`

`.blend` 씬 기하를 광선으로 재서 **사람 궤적**과 **카메라가 설 수 있는 자리**(방위각 × 고도 × 반경 격자)를 JSON 으로 낸다.
`--verify_poses` 를 주면 이미 만든 `cam_c2w (N,4,4)` npz 를 검증 모드로 재기만 한다.
(이름이 비슷한 `trumans_probe.py` 는 배포본 *파일*을 훑는 다른 스크립트다.)

```bash
B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
R=/data1/cympyc1785/data/trumans/Data_release/Recordings_blend/00add26c-7a26-4a61-b192-b97aa493b3f3
$B -b $R/00add26c-7a26-4a61-b192-b97aa493b3f3.blend \
  --python fit/ingest/trumans_scene_probe.py -- \
  --frames 100 148 --anchor_frame 124 \
  --out /data1/cympyc1785/LatentCamVid/tmp/probe_00add26c.json
```

env: **Blender 내장 python** (`-b <blend> --python <this> --` 뒤에 인자)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--out` / `--frames` | (필수) | 결과 JSON / 프레임 범위 |
| `--frame_list` | `[]` | 개별 프레임 지정 |
| `--anchor_frame` | `-1` | -1 = 구간 중앙 |
| `--anchor_origin` | `obb_center` | |
| `--subject_kind` | `human` | `human/event/object` |
| `--prop_names` | `[]` | `obj_list.txt` 이름 |
| `--az_step` | `15.0` | 방위각 격자 간격(도) |
| `--elevations` | `[0.0, 12.0, 25.0, 40.0]` | |
| `--radii` | `[1.0, 1.5, 2.2, 3.0, 4.0, 5.5]` | |
| `--verify_poses` | `""` | npz `cam_c2w` 검증 모드 |
| `--probe_distance` | `1.5` | clearance 광선 최대 거리 |
| `--min_clearance` | `0.20` | 요약 출력용 임계 |
| `--los_samples` / `--los_bands` | `0` / `5` | 높이 띠 수 (LBM 도 5) |
| `--los_mode` | `vertex` | `vertex/obb_lbm` |

> 비고: docstring 예시가 `--out /tmp/probe_*.json`. `tmp/` 로 바꿀 것.
> Blender 는 이 스크립트가 예외로 죽어도 rc=0 이므로 출력 JSON 존재로 성공을 판정해야 한다.

### `trumans_to_recon.py`

TRUMANS action 구간 하나를 **LBM-Lite 입력 규약**(`recon_and_seg` + `seg_instances`)으로 굽는다.
Vista4D 배포본 포맷(RGBD + 카메라 + SAM3 track)을 TRUMANS 는 아무것도 주지 않으므로 `.blend` 를 직접 렌더한다 —
그래서 depth 와 사람 마스크가 추정치가 아니라 정답이다. 시작 pose 샘플링 → 게이트 검증 → Blender 렌더 순으로 돈다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY fit/ingest/trumans_to_recon.py --recording 2b4c9b84-... --list_actions
$PY fit/ingest/trumans_to_recon.py \
  --recording 00add26c-7a26-4a61-b192-b97aa493b3f3 --action 3 --seed 0 --out_video tru_00add26c_a03
```

env: vista4d (OpenEXR / imageio / numpy). Blender 는 `subprocess` 로 부른다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--recording` | (필수) | `Recordings_blend` 디렉토리명 |
| `--action` | `0` | `Actions/<seq>.txt` 의 0-based 줄 |
| `--list_actions` | off | 목록만 찍고 종료 |
| `--out_video` | `""` | 비우면 `tru_<uuid8>_a<NN>` |
| `--sequence` | `""` | 비우면 프레임 수로 자동 판별 |
| `--walk_actions` | `True` | `--no_walk_actions` |
| `--walk_speed` / `--walk_frac` / `--walk_smooth` / `--walk_max` | `0.4` / `0.8` / `9` / `8` | 보행 채굴 |
| `--motion_fps` / `--frame_step` / `--out_fps` / `--smooth_window` | `30.0` / `1` / `0.0` / `0` | |
| `--preset` / `--preset_scale` | `""` / `1.0` | 소스 카메라 preset |
| `--min_end_radius` | 상수 | |
| `--probe_only` / `--poses_override` / `--anchor_cell` | off / `""` / `[]` | |
| `--seed` | `0` | 시작 pose/preset 샘플 seed |
| `--subject_kind` | `human` | `human/auto/event/object` |
| `--prop_names` / `--anchor_origin` / `--aim_bias` | `[]` / `obb_center` / `None` | |
| `--track_gain` | `0.6` | 1 = 완전 추종 |
| `--aim_keyframes` | `0` | |
| `--min_clearance` / `--min_floor_drop` / `--min_clear_frac` / `--min_subject_dist` | `0.20` / `0.30` / `0.90` / `0.80` | 게이트 |
| `--max_tries` | `6` | 시작 pose 후보 수 |
| `--hold_fallback` | `False` | |
| `--blender_retries` | `1` | |
| `--radius_strata` | `True` | `--no_radius_strata` |
| `--force` | off | 검증 실패해도 렌더 강행 |
| `--res` / `--lens` | `[960, 540]` / `25.0` | |
| `--samples` | `16` | EEVEE RGB pass 샘플 수 |
| `--rgb_engine` | `eevee` | `eevee/cycles` |
| `--rgb_samples` / `--rgb_cdevice` | `128` / `GPU` | cycles 일 때 |
| `--sky_depth` | `1000.0` | 배경 clamp |
| `--trumans` / `--blender` / `--vista4d_root` / `--eval_data` | 상수 | |
| `--work` | `""` | 비우면 `out/trumans_recon` |

> 비고 (**깨짐**): 렌더 단계가 `path.join(HERE, "trumans_gt_render.py")` (`HERE = fit/ingest`) 를 부르는데
> 그 파일은 `viz/trumans_gt_render.py` 에만 있다. 같은 파일의 `trumans_scene_probe.py` 호출(902/1026행)은 `HERE` 기준으로 맞다.

### `trumans_export_mesh.py`

`.blend` 의 **evaluated mesh 삼각형**을 프레임별로 뽑아 `mesh_gt.npz` 로 굽는다.
depth 재투영 G1 은 소스 카메라에서 본 표면 껍데기 하나뿐이라 시야 밖은 채점 자체가 안 되는데,
TRUMANS 는 씬 전체가 `.blend` 라 부피를 안다.

```bash
BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BL <path>.blend --background --python fit/ingest/trumans_export_mesh.py -- \
  --frames 1777 1921 3 --out out_trumans/tru_1d076f8c_a00_s3f0k6/mesh_gt.npz
```

env: **Blender 내장 python**

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--frames` | (필수, 3개) | start end step |
| `--out` | (필수) | `mesh_gt.npz` 경로 |
| `--move_eps` | `5e-4` | 정적/동적 경계 (m) |

### `build_trumans_mesh_grid.py`

`tru_<rec8>_a<NN>_<tag>` 이름에서 recon 폴더를 되찾아 `.blend` 경로·프레임 범위·소스 카메라 npz 를 풀고,
**Blender 1회**(`trumans_export_mesh.py`) + **numpy 1회**(`lbm/mesh_collision.py`)로 `mesh_grid.npz` 를 만든다.
두 단계가 서로 다른 인터프리터에서 돌기 때문에 이 래퍼가 따로 있다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY fit/ingest/build_trumans_mesh_grid.py --video tru_1d076f8c_a00_s3f0k6 --output_root out_trumans
```

env: vista4d (scipy). Blender 는 `subprocess`.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | `tru_<rec8>_a<NN>_<tag>` |
| `--output_root` | `out_trumans` | |
| `--recon_root` | 상수 | |
| `--out` | `""` | 비면 `<root>/<video>/mesh_grid.npz` |
| `--blender` | `BLENDER` 상수 | |
| `--voxel` / `--pad` / `--clip` | `0.05` / `1.0` / `3.0` | 격자 한 칸(m) / AABB 여유(m) / 거리 상한(m) |
| `--move_eps` | `5e-4` | 정적/동적 경계 (m) |
| `--keep_tris` | `False` | |
| `--skip_done` | `True` | `--no_skip_done` |

> 비고 (**깨짐**): 44행이 `path.join(CINEMATRAJ_ROOT, "scripts", "trumans_export_mesh.py")` 를 가리키는데
> 그 파일은 지금 `fit/ingest/trumans_export_mesh.py` 다. `scripts/` 에는 없다.
> `fit.convert.bank_to_blender_poses` 에서 `BLENDER`, `chunk_paths` 를 import 한다 (ingest → convert 역참조).

### `trumans_blend_layout_worker.py`

`.blend` 에서 원본 Look-Before-Move 가 요구하는 **레이아웃 수치**(asset world 위치·크기)를 뽑는다.
TRUMANS 는 mesh 를 `<obj>_root_<obj>` EMPTY 에 매달아 애니메이션하므로 `obj.location` 을 그대로 쓰면 틀린다.

```bash
BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BL <scene>.blend --background --python fit/ingest/trumans_blend_layout_worker.py -- \
  --frame 0 --assets cup_01,oven_base_01 \
  --out /data1/cympyc1785/LatentCamVid/tmp/trumans_layout.json
```

env: **Blender 내장 python**

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--frame` | `0` | 레이아웃을 잴 프레임 |
| `--assets` | `""` | 쉼표 구분 오브젝트 이름 |
| `--include_armatures` | `True` | `--no_include_armatures` |
| `--out` | `/data1/cympyc1785/LatentCamVid/tmp/trumans_layout.json` | |

> 비고: argparse 기본값은 규칙에 맞는 `tmp/` 인데 docstring 예시만 `--out /tmp/layout.json` 이다. 기본값 쪽이 맞다.

### `trumans_frame_shift_startup.py`

**CLI 가 없다.** LBM 렌더 워커가 씬 프레임 범위를 무조건 `1..frame_count` 로 덮어써서
2077 프레임짜리 TRUMANS take 를 물리면 어느 shot 이든 첫 1.5초만 렌더되는 문제를,
Blender **startup 훅**(`load_post`)으로 애니메이션을 앞으로 당겨 해결한다.

```bash
export BLENDER_USER_SCRIPTS=<out>/_frame_shift     # 이 파일이 그 아래 startup/ 에 있다
export TRUMANS_FRAME_OFFSET=-50                    # 원본 프레임 51 -> 씬 프레임 1
python Engine/run_full_pipeline.py --demo-root <out>/<uuid>__w01_f0051_0069 ...
```

env: **Blender 내장 python** (직접 실행하지 않고 환경변수로 등록)

| 환경변수 | 설명 |
| --- | --- |
| `BLENDER_USER_SCRIPTS` | 이 파일을 담은 `startup/` 의 부모 |
| `TRUMANS_FRAME_OFFSET` | 프레임 오프셋 (음수면 당긴다) |

### `build_trumans_metadata.py`

TRUMANS-Lite manifest 737개의 `caption.{target,event}` 를 Vista 와 **같은 스키마**의 `metadata.csv` 로 만든다.
`build_bank_captions.py` 가 `event` 필드를 이 csv 의 `prompt` 열 첫 문장에서 읽고, `--videos all` 의 목록도 `video` 열에서 만들기 때문이다.

```bash
python fit/ingest/build_trumans_metadata.py                                        # 737행 전부
python fit/ingest/build_trumans_metadata.py \
  --out /data1/cympyc1785/LatentCamVid/tmp/meta.csv --dry_run
```

env: 미확정 (csv/json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--work` | 상수 | manifest 출처 |
| `--out` | 상수 | 쓸 csv 경로 |
| `--dry_run` | off | 안 쓰고 표만 |

> 비고: docstring 예시가 `--out /tmp/meta.csv`. `tmp/` 로 바꿀 것.

### `trumans_lite_to_dl3dv.py`

TRUMANS-Lite 뱅크(`recon_and_seg/<video>/`)를 latentcam 이 읽는 **DL3DV `da3` 온디스크 포맷**으로 바꾼다.
`dataset_mixed._VALID_NAMES` 에 TRUMANS 가 없지만 `OVERRIDE_WHITELIST` 의 `dl3dv_root`/`meta_csv` 를
갈아끼우면 그대로 돌기 때문에 새 dataset 클래스를 만들지 않는다.
`--layout per_recording` 이 기본이며 `per_clip` 은 옛 평면 배치(누수 있음)다.

```bash
python fit/ingest/trumans_lite_to_dl3dv.py --dry_run
python fit/ingest/trumans_lite_to_dl3dv.py --workers 8
```

env: 미확정 (numpy/이미지 I/O)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--recon_root` / `--work` / `--out_root` | 상수 | recon / manifest+world pose / 출력 |
| `--layout` | `per_recording` | `per_recording/per_clip` |
| `--meta_csv` | `meta_trumans.csv` | |
| `--chunk_prefix` | `trumans` | `""` 면 한 단계 (seg-list 불가) |
| `--test_recordings` | `["2b4c9b84"]` | |
| `--seg_list_prefix` | `seg_list_trumans` | |
| `--image_dir` | `images_4` | dl3dv `IMAGE_DIR_NAMES` 첫 후보 |
| `--avg_scale_refs` | `["context_first_cam"]` | |
| `--clip_avg_scale_ref` | `centroid` | |
| `--avg_scale_key` | `avg_scale` | |
| `--workers` | `7` | |
| `--skip_done` | `True` | `--no_skip_done` |
| `--dry_run` | off | |

### `repair_recon_poses.py`

`--work` 가 조용히 무시되어 덮어써진 `poses_aNN.npz` 를 배포본 `cameras.npz` 에서 되돌린다.
`trumans_lite_bank.py` 가 `--work` 를 `--video_suffix` 가 있을 때만 하위로 전달하기 때문에 생긴 사고를 고치는 도구다.
**기본은 진단만** 하고, `--apply` 가 있어야 쓴다.

```bash
python fit/ingest/repair_recon_poses.py --video tru_00add26c_a00          # 진단만
python fit/ingest/repair_recon_poses.py --video tru_00add26c_a00 --apply  # 되돌린다
```

env: 미확정 (numpy)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | `None` | 한 편만 |
| `--bank_manifest` | `None` | 또는 뱅크 매니페스트 전량 |
| `--work` | `<CinemaTraj>/out/trumans_recon` | |
| `--eval_data` | `/data1/cympyc1785/data/TRUMANS-Lite` | |
| `--tolerance` | `1e-6` | |
| `--backup_suffix` | `.overwritten` | |
| `--aim_base` | `chest` | `chest/obb_center` |
| `--aim_smooth_window` / `--aim_bias` | `11` / `0.0` | |
| `--apply` / `--force` | off / off | |

### `vista_blend_worker.py`

Vista4D 점군에서 뽑은 삼각형 메시를 **원본 LBM 이 읽을 수 있는 `.blend`** 로 굽는다.
LBM 의 `occlusion_check` 가 45개 표본점 dense 레이캐스트라 면이 없는 점군에는 레이가 안 맞아
가림이 항상 0 으로 나오기 때문이다.

```bash
BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
T=/data1/cympyc1785/LatentCamVid/tmp
$BL --background --python fit/ingest/vista_blend_worker.py -- \
  --geom $T/camel_geom.npz --meta $T/camel_geom.json --out $T/camel.blend
```

env: **Blender 내장 python**

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--geom` / `--meta` / `--out` | (필수) | `vista_to_lbm_demo` npz / 같은 이름 json / 저장할 `.blend` |
| `--engine` | `BLENDER_EEVEE_NEXT` | |

> 비고: docstring 예시가 `/tmp/camel*.{npz,json,blend}`. `tmp/` 로 바꿀 것.

---

## `graph/` — scene graph, VLM 명사·인스턴스 기술

| 파일 | 역할 |
| --- | --- |
| `extract_nouns_vlm.py` | 영상 프레임 → VLM → detector 에 먹일 명사 목록 (Stage 0) |
| `extract_static_nouns.py` | 정적 명사만 따로 — `sam3_static_instances.py` 의 keyword |
| `build_scene_graph.py` | 영상 1편 → `scene_graph.json` + 승인용 그림 (Part A 전체) |
| `describe_instances_vlm.py` | SAM3 인스턴스 → VLM → 캡션이 쓸 **지칭 표현** `instance_desc.json` |
| `retrieve_datadop_shapes.py` | plan 에 맞는 실제 영화 궤적을 DataDoP 에서 검색해 preset 모양으로 |
| `trumans_vlm_action_tag.py` | TRUMANS action 창을 VLM 으로 다시 서술 (LBM narrative 입력) |

### `extract_nouns_vlm.py`

영상 프레임을 VLM 에 보여 detector 에 먹일 명사를 뽑는다. 그동안 "VLM keyword" 라 부른 것은 사실
`metadata.csv:dynamic` 열(저자 정답)이었고, 이 스크립트가 그 자리를 실제 VLM 으로 대체한다.
GroundingDINO/SAM3 가 caption 의 부분문자열을 라벨로 돌려주므로 출력은 lowercase·단수·1~2 단어로 강제된다.

```bash
# vLLM 을 먼저 올린다
bash exec/serve_qwen3vl.sh &
python fit/graph/extract_nouns_vlm.py --videos camel avocado-slice \
  --frame_mode single multi --output out/vlm_nouns
# 끝나면 서버를 내린다
```

env: 미확정 (HTTP 클라이언트만 쓴다). **vLLM 서버 필요.**

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--videos` | `["camel", "avocado-slice"]` | |
| `--frame_mode` | `["single", "multi"]` | |
| `--num_frames` | `6` | |
| `--output` | `<HERE>/out/vlm_nouns` | |
| `--api_base` | `http://127.0.0.1:22002/v1` | |
| `--model` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | |
| `--temperature` | `0.1` | |
| `--max_repairs` | `3` | JSON 파싱 실패 시 재질의 횟수 |
| `--merge` | `False` | `--no_merge` |
| `--eval_data` | 상수 | |

### `extract_static_nouns.py`

정적 명사는 동적과 **다른 규칙**으로 걸러야 한다 (동적은 `metadata.csv:dynamic` 정답이 있지만 정적은 없다).
VLM 목록을 그대로 SAM3 에 넣으면 깨지는 것들 — 특히 바닥/벽 같은 surface — 를 빼고
`sam3_static_instances.py` 가 먹을 목록을 만든다.

```bash
python fit/graph/extract_static_nouns.py --videos camel avocado-slice
python fit/graph/extract_static_nouns.py --videos camel --source prompt --no_drop_surfaces
```

env: 미확정. `--source vlm`/`auto` 경로는 **vLLM 서버 필요** (기존 `vlm_nouns` 산출물을 읽는 경로면 불필요).

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--eval_data` | 상수 | |
| `--output` | `<HERE>/out/static_nouns` | |
| `--vlm_nouns` | `None` | None = `VLM_NOUNS_CANDIDATES` 순서대로 |
| `--videos` | `None` | None = `metadata.csv` 전체 |
| `--source` | `auto` | `auto/vlm/prompt` |
| `--frame_mode` | `multi` | `single/multi` |
| `--drop_surfaces` | `True` | `--no_drop_surfaces` |
| `--max_nouns` | `8` | |
| `--max_surface_nouns` | `0` | |
| `--merge` | `False` | `--no_merge` |

### `build_scene_graph.py`

영상 1편 → `scene_graph.json` + 승인용 overlay/topdown 그림. 렌더 게이트가 "여기서 보면 이렇게 보인다"만
말할 수 있는 데 비해 graph 는 **어디를 봐야 하는지**(subject 지목, 노드 OBB, 지면·중력축, 씬 스케일 `S`)를 준다.
뱅크 전 단계 전체가 이 파일 하나에 의존한다.

```bash
python fit/graph/build_scene_graph.py --video camel
python fit/graph/build_scene_graph.py --video avocado-slice --no_skip_done --topdown
```

env: vista4d (cv2 · scipy · imageio · numpy. torch/open3d/sklearn 불필요)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--eval_data` / `--vista4d_root` | 상수 | |
| `--output_root` | `None` | None = `<CinemaTraj>/out` |
| `--seg_root` / `--seg_static_root` | `None` | None = `<eval_data>/eval_data/seg_instances`(정적은 있으면 자동) |
| `--scene_scale_mode` | `points_first_cam` | `SCALE_MODES` |
| `--scene_scale_stride` | `1` | 픽셀 서브샘플 |
| `--seed` | `0` | |
| `--skip_done` | `True` | `--no_skip_done` |
| `--gravity_frames` / `--gravity_max_points` | `5` / `200000` | |
| `--gravity_source` | `geocalib` | `ransac/gt/geocalib/auto` |
| `--ground_source` | `pointcloud` | `pointcloud/gt` |
| `--floor_seg_root` / `--floor_quantile` | `None` / `0.5` | |
| `--subject_source` | `pointcloud` | `pointcloud/gt` |
| `--gt_root` | 상수 | |
| `--weak_inlier_ratio` | `0.0` | |
| `--min_area_frac` / `--min_frames` / `--min_score` | `0.002` / `5` / `0.3` | 노드 채택 하한 |
| `--max_points_per_frame` | `20000` | |
| `--depth_trim_quantile` / `--depth_mode_bins` / `--depth_mode_gap_frac` | `0.01` / `40` / `0.005` | |
| `--deinflate` | `False` | `--no_deinflate` |
| `--merge_voxel_u` / `--merge_iou` | `0.02` / `0.5` | 중복 노드 병합 |
| `--per_keyword_top_k` | `5` | goat 의 rock 31개용 |
| `--max_dyn_nodes` / `--max_stat_nodes` | `6` / `10` | |
| `--prescreen_factor` | `3` | lift 전 상한 = 상한 × 이것 |
| `--contact_u` / `--wall_u` / `--wall_contact_frac` / `--near_factor` | `0.05` / `0.08` / `0.15` / `1.5` | 엣지 판정 |
| `--temporal_edges` | `False` | `--no_temporal_edges` |
| `--overlay` / `--topdown` | `True` / `True` | 그림 산출 |

> 비고: `--gravity_source geocalib` 이 기본이므로 **`fit/ingest/geocalib_gravity.py` 가 먼저** 돌아 있어야 한다.
> 없으면 조용히 fallback 으로 떨어져 세로 슬롯(crane/pedestal)이 하류에서 통째로 빠진다.

### `describe_instances_vlm.py`

SAM3 인스턴스 crop 을 VLM 에 보여 캡션이 쓸 **12~14 단어 지칭구**(`instance_desc.json`)를 만든다.
앞단 `extract_nouns_vlm.py` 는 detector 입력이라 1~2 단어로 잘리고, 같은 라벨의 형제가 둘 이상이면
라벨만으로 어느 쪽인지 못 가리킨다. 그래서 SAM3 **뒤**에 붙는다. 기본은 뱅크가 실제로 쓴 anchor 만 기술한다.

```bash
bash exec/serve_qwen3vl.sh &
python fit/graph/describe_instances_vlm.py --videos avocado-slice camel --bank_dir hole_bank_k6_d99
python fit/graph/describe_instances_vlm.py --videos avocado-slice --all_nodes --dry_run
# 끝나면 서버를 내린다
```

env: 미확정 (HTTP 클라이언트). **vLLM 서버 필요.**

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--videos` | (필수) | |
| `--output_root` | `<HERE>/out` | |
| `--eval_data` | 상수 | |
| `--bank_dir` | `hole_bank_k6_d99` | anchor 목록 출처 |
| `--all_nodes` | `False` | 뱅크 anchor 말고 전 노드 |
| `--out_name` | `instance_desc.json` | |
| `--num_frames` | `6` | |
| `--tile` | `448` | 패널 한 칸 픽셀 |
| `--dim` | `0.28` | 마스크 바깥 밝기 배율 |
| `--crop_pad` | `0.35` | 타이트 crop 여백 비율 |
| `--api_base` | `http://127.0.0.1:22002/v1` | |
| `--model` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | |
| `--temperature` / `--max_repairs` | `0.1` / `3` | |
| `--dry_run` | `False` | `--no_dry_run` |

> 비고: `--bank_dir` 기본값이 옛 세대(`hole_bank_k6_d99`)라 새 세대를 굽고 나면 **반드시 명시**해야 한다.

### `retrieve_datadop_shapes.py`

우리 preset 36종은 전부 손으로 정의한 기하 primitive 라 교과서적으로 매끄럽다. 이 스크립트는
DataDoP 3,000 shot 의 태그에서 plan(원하는 움직임)에 맞는 **실제 영화 궤적**을 검색해 preset 모양(`external_shapes`)으로 뽑는다.
`route_presets --free_moving datadop` / `sample_camera_bank --external_shapes` 가 그 JSON 을 먹는다.

```bash
python fit/graph/retrieve_datadop_shapes.py \
  --plan_file configs/datadop_plan.json --out configs/datadop_shapes.json
python fit/graph/retrieve_datadop_shapes.py --move "move backward" --angular any --top_k 3
```

env: 미확정 (numpy + json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--tag_glob` | 상수 | DataDoP 태그 JSON glob |
| `--plan_file` | `None` | 없으면 라벨 분포에서 자동 생성 |
| `--move` / `--angular` / `--name` | `None` / `any` / `dd_adhoc` | 단발 조회용 |
| `--window` | `49` | 우리 프레임 수 |
| `--min_span` | `30` | 라벨 구간 최소 길이 |
| `--min_translation` | `0.02` | DataDoP world 단위 (D=1) |
| `--static_tau` | `0.02` | |
| `--rot_cap_deg` | `180.0` | |
| `--per_scene` | `1` | 한 영화에서 최대 몇 개 |
| `--top_k` | `4` | query 당 |
| `--min_pool` / `--min_purity` | `5` / `0.6` | |
| `--seed` | `0` | |
| `--out` | `None` | |

### `trumans_vlm_action_tag.py`

TRUMANS action 창을 contact sheet 로 만들어 VLM 에 보이고, LBM 의 narrative 입력으로 쓸 문장을 받는다.
`Actions/<seq>.txt` 라벨은 **동작 라벨**이지 shot 서술이 아니라 공간 정보가 통째로 비어 있다.
`--include_gaps` 를 켜면 라벨이 안 붙은 프레임(실측 64%)도 창으로 만들어 같이 태깅하고, 산출물 이름이 `_full` 로 갈린다.

```bash
bash exec/serve_qwen3vl.sh &
python fit/graph/trumans_vlm_action_tag.py --sequence 2023-01-17@00-55-00
python fit/graph/trumans_vlm_action_tag.py --sequence 2023-01-17@00-55-00 --include_gaps --verbose
# 끝나면 서버를 내린다
```

env: vista4d (cv2). **vLLM 서버 필요** — 기동은 `bash exec/serve_qwen3vl.sh`.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--sequence` | (필수) | 예 `2023-01-17@00-55-00` |
| `--trumans` | 상수 | |
| `--out` | `<CinemaTraj>/out/trumans_vlm_actions` | |
| `--windows` | `[]` | 특정 창만 |
| `--num_tiles` / `--tile_width` / `--grid_rows` | `6` / `648` / `2` | contact sheet |
| `--gamma` | `1.0` | >1 이면 밝힌다 (원본 median 58) |
| `--api_base` | `http://127.0.0.1:22002/v1` | |
| `--model` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | |
| `--temperature` | `0.1` | |
| `--verbose` | off | |
| `--include_gaps` | (플래그 쌍) | 미태그 구간도 창으로 |
| `--gap_min_frames` / `--gap_max_frames` | `15` / `120` | 짧으면 버리고 길면 균등 분할 |
| `--scene_objects` / `--object_motion` / `--rule_hint` / `--person_probe` | (플래그 쌍) | 프롬프트 구성 |
| `--person` | `""` | `""/man/woman/person` |
| `--person_tiles` / `--person_min_conf` | `4` / `0.6` | 밑돌면 they/their |
| `--recording` | `""` | 비우면 `scene_flag` 로 역산 |
| `--move_meters` / `--move_degrees` | `0.05` / `12.0` | 실측 노이즈 바닥 0.015 / 6.9 |
| `--suffix` | `""` | 산출물 이름 접미사 override |

> 비고: `scripts/trumans_to_lbm_demo.py` 의 `read_actions` 를 import 한다 (그 파일은 존재한다).
> `--tile_width` 를 줄이면 소품을 놓친다.

---

## `bank/` — preset routing → tau → hole ladder fit → emit

주 경로는 네 개다. 나머지 9개는 보조·후처리·TRUMANS 전용이다.

```
route_presets → sample_camera_bank → fit_hole_ladder → emit_bank
   (슬롯 배정)      (τ 뱅크 `bank/`)    (hole 뱅크 `hole_bank*/`)  (`canonical/`)
```

| 파일 | 역할 |
| --- | --- |
| `route_presets.py` | 렌더 없이 `scene_graph.json` 만 보고 anchor + preset 슬롯을 고른다 (변이 예산) |
| `sample_camera_bank.py` | 열거 + 실측으로 카메라를 만든다 — τ 사다리 뱅크 |
| `fit_hole_ladder.py` | preset 별 크기 상한을 이분법으로 푼다 — τ 사다리를 hole 사다리로 교체 |
| `emit_bank.py` | hole 뱅크 전량 → 태그 N개짜리 `canonical.json` 하나 |
| `build_candidate_board.py` | 후보 풀 → 게이트 → `board_candidates.png` (VLM 없이 사람이 먼저 본다) |
| `build_decision_fallback.py` | VLM 없이 `decision.json` 을 만드는 결정론적 fallback |
| `repick_bank.py` | 이미 구운 뱅크의 `picked` 열만 다시 달고 emit 재실행 (재굽기 없음) |
| `patch_bank_shape_mult.py` | 기존 hole 뱅크의 `shape_mult` 만 고친다 (fit 재실행 없이) |
| `merge_static_rung.py` | 정지 preset rung(별도 뱅크)을 기존 hole 뱅크에 붙인다 |
| `export_target_track.py` | 변이별 subject(anchor) OBB world 궤적 → `da3/target_track.npz` |
| `expand_preset_variants.py` | Cinematographer 카메라 1개를 여러 preset 변형으로 불린다 |
| `retime_camera_handoff.py` | Cinematographer 산출물 카메라 길이를 chunk 프레임 수에 맞춘다 |
| `trumans_first_pose_board.py` | `.blend` 에서 chunk 별 시작 pose 후보판 (Blender 워커) |

### `route_presets.py`

anchor 와 preset 슬롯을 **렌더 없이** 고른다. 뱅크가 anchor 전량 × preset 36 × τ 5 × hole 4 를 다 돌면
영상당 `9.7분 + 8.80초 × 변이수` 라 변이가 곧 시간이다. `--emit args` 가 하류에 줄 인자를 찍고,
`--emit_route` 가 예산·fallback 사다리를 담은 route JSON 을 남긴다 (`sample_camera_bank --preset_route` 가 읽는다).

```bash
python fit/bank/route_presets.py --video <id> --output_root out_dynpose             # 표
python fit/bank/route_presets.py --video <id> --output_root out_dynpose --emit args # 하류 인자
```

env: 미확정 (`scene_graph.json` 만 읽는다; GPU 불필요)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--output_root` | `None` | |
| `--min_area_frac` | `0.01` | anchor 화면 점유 하한 |
| `--max_dynamic_anchors` / `--max_static_anchors` | `3` / `3` | |
| `--max_anchors` | `0` | 둘을 적용한 뒤 거는 총합 상한 |
| `--min_anchor_sep` | `2.0` | |
| `--slot_plan` | `full` | `full/grid2x2` |
| `--free_moving` | `off` | `off/rotate/datadop` — target 절 없는 변이 |
| `--target_variants` | `5` | scene 당 변이 **상한** (요구 아님) |
| `--orbit_fallback` | `drop` | `drop/s_curve` — 좁은 span orbit 슬롯 처리 |
| `--orbit_min_span` | `0.0` | 기본 off. `120` 이면 d166~d171 재현 |
| `--slot_pair_fill` | `substitute` | `rotate/substitute` |
| `--vertical_fallback` | `False` | gravity 가 fallback 이어도 세로 슬롯 |
| `--vertical_gravity` | `ground_ransac,geocalib` | 세로 슬롯을 신뢰할 gravity 방법 |
| `--track_mode` | `add` | `add/replace/off` |
| `--track_min_drift_u` | `0.05` | `sample_camera_bank` 와 **같은 값**을 줘야 한다 |
| `--track_pair` / `--slot_rotate` | `False` / `False` | |
| `--anchor_min_drift_u` | `0.0` | |
| `--anchor_require_frame0` | `False` | frame 0 에 보이는 anchor 만 |
| `--anchor_ids_file` / `--slot_whitelist` / `--slot_whitelist_track` / `--force_presets` | `""` | |
| `--external_shapes` / `--num_external` | `None` / `1` | DataDoP 모양 |
| `--emit` | `table` | `table/args/presets/nodes` |
| `--out` | `None` | |
| `--emit_route` | `False` | `--emit args` 가 항상 `--preset_route` 를 붙이게 |
| `--skip_if_empty` | `False` | 슬롯 0 이면 rc=3 |

> 비고: `--orbit_min_span` 은 d157 807행 실측에서 hole·subject_visible 이 span 1.5°~315° 에서 평평해
> **기본 off** 가 되었다. 인자는 재현용으로 남겨 두었다.
> `--vertical_gravity` 에서 `geocalib` 을 빼면 세로 슬롯이 조용히 전량 사라진다 (dd10 10,857행에 crane/pedestal 0건).

### `sample_camera_bank.py`

VLM 선택자 대신 **열거 + 실측**으로 카메라를 만든다 (목표가 "최선의 1개"가 아니라 "다양한 N개"라서).
`bank/` 에 τ 사다리 뱅크를 남긴다. `--fit_tau` 가 기본 on 이라 요청 τ 를 이분법으로 맞추고,
`--drop_saturated` 가 상한에 붙은 변이를 버린다.

```bash
CUDA_VISIBLE_DEVICES=0 python fit/bank/sample_camera_bank.py --video camel
CUDA_VISIBLE_DEVICES=0 python fit/bank/sample_camera_bank.py --video camel \
  --tau_ladder 0.2 0.6 --trackings world lock --num_samples 40 --seed 0
```

env: vista4d (렌더러가 GPU 를 쓴다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--bank_dir` | `bank` | 출력 폴더 |
| `--eval_data` / `--vista4d_root` / `--seg_root` / `--seg_static_root` | 상수 / `None` | |
| `--output_root` | `None` | |
| `--device` | `cuda` | |
| `--skip_on_empty` / `--timing_json` | off / `None` | |
| `--nodes` | `None` | None = 자동 |
| `--min_area_frac` | `0.01` | anchor 화면 점유 하한 |
| `--max_dynamic_anchors` / `--max_static_anchors` / `--max_anchors` | `3` / `3` / `0` | |
| `--anchor_drop_surfaces` | `True` | `--no_anchor_drop_surfaces` |
| `--presets` | `None` | None = 전량 |
| `--preset_route` | `None` | `route_presets --emit_route` 산출 JSON |
| `--route_preset_override` | `None` | |
| `--target_variants` | `0` | |
| `--variant_pool` | `budget` | `budget/full` |
| `--external_shapes` / `--external_aim` | `None` / `traj` | `traj/look_at` |
| `--tau_ladder` | `list(TAU_LADDER)` | |
| `--speeds` | `["steady"]` | `steady/accel/decel/ease` |
| `--trackings` | `["lock"]` | |
| `--look_at_biases` | `[0.0]` | |
| `--follow_gains` / `--follow_smooths` | `["0"]` / `[9]` | |
| `--preset_tracking` | `True` | `--no_preset_tracking` |
| `--tau_ref` | `follow` | `TAU_REF_CHOICES` |
| `--tau_denom` | `z_med_frame0` | `TAU_DENOM_MODES` |
| `--allow_zoom` | `False` | |
| `--track_dynamic_only` | `True` | `--no_track_dynamic_only` |
| `--track_min_drift_u` | `0.05` | route 와 같은 값 |
| `--fixed_focal` | `False` | `--no_fixed_focal` |
| `--start_mode` | `source_frame0` | `source_frame0/board` |
| `--start_grid` | `off` | `off/front` |
| `--start_coverages` | (3개) | |
| `--start_min_ratio` / `--start_max_ratio` | `0.3` / `3.0` | |
| `--aim_anchor` / `--aim_ramp_frames` | `subject` / `12` | |
| `--traj_basis` | `source` | |
| `--aim_keyframes` | `[0]` | |
| `--keyframe_aim` | `auto` | `auto/target/preset_rel` |
| `--keyframe_ease` | `smooth_kf` | |
| `--deroll` | `True` | `--no_deroll` |
| `--orbit_span_frac` / `--min_sweep_deg` | `0.8` / `20.0` | |
| `--orbit_fixed_sweep` | `True` | `--no_orbit_fixed_sweep` |
| `--allow_legacy_scale` | `False` | |
| `--pan_deg_at_max` | `60.0` | |
| `--sweep_deg` / `--pan_deg` | `0.0` / `0.0` | 직접 지정 |
| `--fit_tau` | `True` | `--no_fit_tau` |
| `--drop_saturated` | `True` | `--no_drop_saturated` |
| `--min_subject_points` / `--min_subject_area` | `100` / `0.005` | |
| `--measure_frames` | `5` | |
| `--subject_visible` | `True` | `--no_subject_visible` |
| `--measure_behind` | `False` | G1 |
| `--behind_src_frames` / `--behind_margin_frac` / `--behind_clear_frac` / `--behind_radius_px` / `--behind_min_zcam` | `49` / `0.02` / `0.0` / `0` / `0.02` | |
| `--mesh_margin_frac` / `--mesh_margin_autoclamp` | `0.08` / `0.0` | |
| `--collision_time_match` | `True` | `--no_collision_time_match` |
| `--collision_source` | `depth` | `depth/mesh/both` |
| `--mesh_grid` | `""` | 비면 `<output_root>/<video>/mesh_grid.npz` |
| `--near_pct` | `1.0` | `near_depth` 백분위 |
| `--measure_standoff` / `--measure_obb` | `False` / `False` | |
| `--tile_width` / `--tile_height` / `--center_box` | `640` / `360` / `0.80` | |
| `--num_samples` | `0` | 0 = 열거한 전량 |
| `--seed` | `0` | |
| `--preview` | `True` | `--no_preview` |
| `--preview_max` / `--preview_columns` | `24` / `6` | |

> 비고: `--collision_source mesh` 는 `mesh_grid.npz` 가 있어야 한다 (`build_trumans_mesh_grid.py`).
> `--tau_denom`/`--tau_ref` 는 하류 `fit_hole_ladder` 와 **같아야** 뱅크가 이어진다.

### `fit_hole_ladder.py`

τ 사다리를 **hole 사다리**로 갈아끼운다. 같은 τ 라도 preset 모양에 따라 구멍이 크게 달라지므로
(camel `dyn_0` τ=0.35 에서 `straight_ease` 0.015 vs 다른 preset 수배), 요청한 hole 을 맞추는
knob 값을 preset 마다 이분법으로 푼다. `--hole_mode excess` 가 기본이라 목표는 `hole_static + Δ` 다.
G1(관통) · G5(OBB) · G6(고도/지면) · G7(접근) 게이트가 여기 붙어 있다.

```bash
CUDA_VISIBLE_DEVICES=1 python fit/bank/fit_hole_ladder.py --video camel
CUDA_VISIBLE_DEVICES=1 python fit/bank/fit_hole_ladder.py --video camel \
  --hole_ladder 0.2 0.4 --anchors dyn_0 --verify_frames 0
```

env: vista4d (렌더러가 GPU 를 쓴다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--bank_dir` / `--tau_bank_dir` | `hole_bank` / `bank` | 출력 / 입력(τ 뱅크) |
| `--hole_ladder` | `list(HOLE_LADDER)` | |
| `--cap_hole` | `0.0` | 0 = 사다리 맨 윗단 |
| `--hole_mode` | `excess` | `absolute/excess` |
| `--ladder_metric` | `hole` | `hole/shot_scale` |
| `--min_subject_in_frame` | `0.0` | |
| `--tau_floor_src` | `True` | `--no_tau_floor_src` |
| `--anchors` / `--presets` | `None` | None = τ 뱅크 전량 |
| `--static_rung` | `True` | `--no_static_rung` |
| `--iterations` | `4` | 이분법 횟수 |
| `--shape_headroom` / `--shape_doublings` | `2.0` / `4` | 모양 확대 여유 |
| `--collision_free` | `True` | `--no_collision_free` (G1) |
| `--max_behind_frac` | `0.0` | |
| `--behind_src_frames` / `--behind_margin_frac` / `--behind_clear_frac` / `--behind_min_zcam` / `--behind_radius_px` | `49` / `0.02` / `0.10` / `0.02` / `2` | |
| `--behind_clear_src_ratio` / `--behind_clear_src_pct` | `0.0` / `10.0` | |
| `--mesh_margin_frac` / `--mesh_margin_autoclamp` | `0.08` / `0.0` | |
| `--collision_time_match` | `True` | `--no_collision_time_match` |
| `--collision_source` | `depth` | `depth/mesh/both` |
| `--mesh_grid` | `""` | |
| `--max_behind_frac_dyn` | `None` | |
| `--measure_obb` / `--obb_gate` | `True` / `True` | G5 |
| `--obb_clear_src_ratio` | `0.3` | |
| `--elev_gate` | `True` | G6 |
| `--max_elev_deg` / `--elev_src_margin_deg` | `45.0` / `10.0` | |
| `--ground_gate` / `--min_ground_clear_ratio` | `True` / `0.2` | |
| `--approach_gate` / `--approach_src_ratio` | `True` / `0.3` | G7 |
| `--gate_before_render` | `True` | `--no_gate_before_render` |
| `--time_truncate` / `--min_move_frac` | `False` / `0.5` | |
| `--speed` / `--tracking` | `steady` / `lock` | |
| `--preset_tracking` | `True` | |
| `--look_at_bias` | `0.0` | |
| `--follow_keyframes` / `--follow_kf_interp` | `SHAPE_DEFAULTS` | |
| `--start_mode` / `--aim_anchor` / `--aim_ramp_frames` | `source_frame0` / `subject` / `SHAPE_DEFAULTS` | |
| `--traj_basis` / `--orbit_span_frac` | `SHAPE_DEFAULTS` | |
| `--min_sweep_deg` | `20.0` | |
| `--orbit_fixed_sweep` | `True` | |
| `--allow_legacy_scale` | `False` | |
| `--aim_keyframes` / `--keyframe_aim` | `SHAPE_DEFAULTS` | |
| `--keyframe_ease` | `smooth_kf` | |
| `--smooth_passes` / `--smooth_lambda` | `SHAPE_DEFAULTS` | |
| `--tau_ref` / `--tau_denom` | `follow` / `z_med_frame0` | |
| `--tau_knob_min` / `--tau_knob_max` | `TAU_KNOB_MIN_DEFAULT` / `0.0` | |
| `--deroll` | `True` | |
| `--fixed_focal` | `False` | |
| `--suspect` | `True` | 의심 태그 부여 |
| `--suspect_hole` / `--suspect_in_frame` / `--suspect_behind` / `--suspect_path_len` | `0.0` / `0.85` / `0.0` / `0.02` | |
| `--bisect_frames` / `--verify_frames` | `5` / `13` | 0 = 전 프레임 |
| `--subject_visible` / `--min_subject_visible` | `True` / `0.6` | |
| `--gate_static` | `False` | |
| `--fallback_ladder` / `--fallback_target` | `False` / `5` | |
| `--retry_status` | `["clamped_low"]` | |
| `--retry_suspect` | `[]` | |
| `--pick_budget` | `0` | |
| `--area_timeline` / `--composition` | `True` / `True` | 캡션이 읽을 열 |
| `--composition_min_area` / `--composition_max_nodes` | `0.004` / `3` | |
| `--tile_width` / `--tile_height` / `--center_box` | `640` / `360` / `0.80` | |

> 비고: `poses.npz` 에는 고르지 않은 행의 pose 까지 전부 들어간다 (그래서 `repick_bank.py` 가 가능하다).
> `poses.npz` 의 `target_hole` 은 요청치가 아니라 **달성치**이므로 조인 키로 쓰면 안 된다 — 키는 `variant_id`.

### `emit_bank.py`

hole 뱅크 전량을 **하나의** `canonical.json`(변이당 카메라 태그 1개)으로 만든다.
`decode/emit.py` 가 결정 하나를 canonical 로 바꾸는 데 비해, 우리가 만든 건 뱅크이기 때문이다.
`--strict` 가 기본 on 이라 재생성한 pose 가 뱅크와 `--pose_tol` 안에서 일치하는지 검사한다.

```bash
python fit/bank/emit_bank.py --video camel
python fit/bank/emit_bank.py --video camel --status solved --presets orbit_left_arc,truck_left
python fit/bank/emit_bank.py --video avocado-slice --bank_dir hole_bank_r80_g5abs --drop_folded
```

env: `da3` / vista4d 아무거나 (GPU 안 쓴다 — 렌더 0회)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--bank_dir` | `hole_bank` | |
| `--output_root` / `--out_dir` | `None` | 기본 `<repo>/out`, `<bank>/canonical` |
| `--external_shapes` / `--external_aim` | `None` | |
| `--tag_prefix` | `""` | |
| `--n_poses` | `21` | `emit_model_cams` 는 21 고정 |
| `--strict` | `True` | `--no_strict` |
| `--pose_tol` | `1e-9` | 재현 허용 오차 (u) |
| `--keyframe_ease` | `""` | 비면 뱅크 값, 없으면 `FIXED_FALLBACK`(=`smoothstep`) |
| `--smooth_passes` / `--smooth_lambda` | `None` | 비면 뱅크 값 |
| `--dump_poses` | `""` (`const="1"`) | |
| `--status` | `all` | 쉼표 구분 (`solved` 등) |
| `--anchors` / `--presets` / `--holes` | `None` | 쉼표 구분 필터 |
| `--min_path_len` | `0.0` | 이 아래는 뺀다 (u) |
| `--require` | `[]` | |
| `--picked_only` | `False` | `--no_picked_only` |
| `--drop_folded` | `False` | `--no_drop_folded` |

> 비고: `--keyframe_ease` 를 비워 두면 옛 뱅크는 `FIXED_FALLBACK = smoothstep` 으로 재현된다 (동결값).
> 새 뱅크는 `smooth_kf` 가 `bank.json` 에 실려 있으므로 그대로 따라간다.

### `build_candidate_board.py`

후보 카메라 풀을 게이트에 통과시켜 `board_candidates.png` 로 만든다. look-before-move 는
"모델에게 렌더를 보여준다"가 전부라, 보여줄 그림이 사람 눈에 말이 안 되면 프롬프트를 고쳐도 소용이 없다.
`sample_camera_bank` / `fit_hole_ladder` 가 이 파일의 `subject_track_volume` 을 import 한다.

```bash
CUDA_VISIBLE_DEVICES=0 python fit/bank/build_candidate_board.py --video camel
CUDA_VISIBLE_DEVICES=0 python fit/bank/build_candidate_board.py --video camel --subject_id dyn_0
```

env: vista4d (렌더러가 GPU 를 쓴다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--eval_data` / `--vista4d_root` | 상수 | |
| `--output_root` / `--seg_root` / `--seg_static_root` | `None` | |
| `--device` | `cuda` | |
| `--fixed_focal` | `True` | `--no_fixed_focal` |
| `--subject_id` | `None` | None = 자동 선택 |
| `--subject_min_area_frac` | `0.01` | |
| `--frame` | `0` | 후보를 정의하는 기준 프레임 |
| `--skip_done` | `False` | `--no_skip_done` |
| `--pool_mode` | `budget` | `budget/absolute` |
| `--num_azimuth` / `--num_elevation` / `--num_radius` | `5` / `3` / `3` | |
| `--azimuths` / `--elevations` / `--distances` | `AZIMUTHS_DEG` / `ELEVATIONS_DEG` / `DISTANCE_RATIOS` | |
| `--look_at_bias` | `0.0` | |
| `--az_margin_deg` | `180.0` | |
| `--max_tau` | `0.30` | |
| `--max_view_angle_deg` | `40.0` | |
| `--min_coverage` | `0.55` | |
| `--center_box` | `0.80` | |
| `--min_subject_area` / `--max_subject_area` | `0.03` / `0.50` | |
| `--min_occlusion_pass` | `0.40` | |
| `--behind_frames` | `7` | |
| `--board_size` / `--board_columns` | `9` / `3` | |
| `--tile_width` / `--tile_height` | `640` / `360` | |

> 비고: 이 파일의 `--fixed_focal` 은 기본 `True` 인데 `sample_camera_bank`/`fit_hole_ladder` 는 기본 `False` 다.
> 릴 렌더에서 화각이 떨리면 뱅크의 `fixed_focal` 값을 따라가고 있는지 먼저 확인할 것.

### `build_decision_fallback.py`

VLM 없이 `decision.json` 을 만든다 (§B6 의 결정론적 fallback 그대로).
"파이프라인은 절대 hard-fail 하지 않는다"가 계약이라, VLM 미부착 상태와 3회 재질의 실패가 같은 코드를 타야
하류(decode/emit/verify)가 그대로 돈다. `lbm/loop.py` 가 이 함수를 import 해서 쓴다.

```bash
python fit/bank/build_decision_fallback.py --video camel
```

env: 미확정 (json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--output_root` / `--output` | `None` | |
| `--preset` | `orbit_left` | |
| `--tracking` | `lock` | D127 에서 `drift` 삭제 |
| `--target_tau` | `0.20` | |
| `--orbit_span_frac` | `0.7` | |
| `--look_at_bias` | `0.0` | |
| `--start_tau_frac` | `0.5` | |

### `repick_bank.py`

이미 구운 뱅크의 `picked` 열만 다시 달고 emit 을 다시 돌린다. `fit_hole_ladder` 의 pick 은
사다리가 끝난 뒤의 **순수 후처리**이고 `poses.npz` 에 고르지 않은 행의 pose 까지 들어 있어서 재굽기가 필요 없다.

```bash
python fit/bank/repick_bank.py \
  --config configs/bank/d188_dynpose100k_track_objcentric.json \
  --per_anchor --num_shards 6 --shard_id 0
```

env: 미확정 (하위로 `emit_bank` 를 부른다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--config` | (필수) | 뱅크 config JSON |
| `--videos` | `["all"]` | all = 뱅크 있는 전량 |
| `--videos_file` | `""` | 한 줄 1편. `--videos` 보다 우선 |
| `--output_root` / `--bank_dir` | `""` | 빈 값 = config |
| `--pick_budget` | `None` | |
| `--per_anchor` | `False` | `--no_per_anchor` |
| `--picked_only` | `True` | `--no_picked_only` |
| `--no_emit` | `False` | 열만 바꾸고 끝 |
| `--dry_run` | `False` | 수율만 센다 |
| `--mark` | `.repicked` | |
| `--skip_done` | `True` | `--no_skip_done` |
| `--num_shards` / `--shard_id` | `1` / `0` | |

### `patch_bank_shape_mult.py`

이미 만들어진 hole 뱅크의 `shape_mult` 를 고친다 — `fit_hole_ladder` 재실행(영상당 ~1350 s) 없이.
모양 확대 루프가 `mult` 를 판정 **뒤에** 곱하던 버그를 사후 보정하는 도구다.

```bash
python fit/bank/patch_bank_shape_mult.py --videos basketball-four --dry_run
python fit/bank/patch_bank_shape_mult.py                      # out/ 전체, .bak 남기고 수정
python fit/bank/patch_bank_shape_mult.py --no_backup --bank_dir hole_bank
```

env: 미확정 (json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--out` | `OUT_DEFAULT` | |
| `--bank_dir` | `hole_bank` | |
| `--videos` | `""` | 쉼표 구분, 비우면 전체 |
| `--headroom` | `2.0` | `fit_hole_ladder --shape_headroom` 과 같은 값 |
| `--dry_run` | `False` | `--no_dry_run` |
| `--backup` | `True` | `--no_backup` |

### `merge_static_rung.py`

정지 preset rung(별도 뱅크)을 기존 hole 뱅크에 **붙인다**. D78 이전 `fit_hole_ladder` 가
`knob_kind()` 가 None 인 `STATIC_PRESETS`(`static_hold`, `static_look_at`, `track_hold`, `track_look_at`)를
preset 목록에서 통째로 빼 버려 τ 뱅크에는 있는데 emit 뱅크에 없던 것을 되살린다.

```bash
python fit/bank/merge_static_rung.py --video camel --dry_run
python fit/bank/merge_static_rung.py --video camel \
  --bank_dir hole_bank_k6_d77 --static_dir hole_bank_k6_d77_static
```

env: 미확정 (json + numpy)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--output_root` | `None` | |
| `--bank_dir` | `hole_bank_k6_d77` | 붙일 **대상** |
| `--static_dir` | `hole_bank_k6_d77_static` | 붙일 **것** |
| `--dry_run` | `False` | |

> 비고: 지금 `fit_hole_ladder` 는 `--static_rung` 이 기본 on 이라 새 세대에는 보통 필요 없다.

### `export_target_track.py`

변이별 subject(anchor) OBB world 궤적을 `<scene>/da3/target_track.npz` 로 낸다.
latentcam 의 `target_track_dim>0` arm 이 subject 3D 궤적을 카메라 diffusion 의 `x_t` 채널에 concat 하는데
학습 레이아웃에는 카메라·캡션만 있기 때문이다. 키 순서는 `target_poses.npz` 와 같다.

```bash
python fit/bank/export_target_track.py                       # 전 scene (vista4d)
python fit/bank/export_target_track.py --videos camel --dry_run
python fit/bank/export_target_track.py \
  --dl3dv_root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d194 \
  --corpus dynpose --out_root <CinemaTraj>/out_dynpose
```

env: 아무거나 (numpy 만 쓴다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--dl3dv_root` | 상수 | 코퍼스 루트 |
| `--corpus` | `vista4d` | `dl3dv_root` 아래 하위폴더 |
| `--out_root` | `None` | CinemaTraj `out/` (`scene_graph` 위치) |
| `--videos` | `None` | |
| `--max_rows` | `60` | 표에 찍을 최대 행 수 |
| `--dry_run` | off | |

> 비고: dynpose 는 `--corpus` 와 `--out_root` 를 **둘 다** 바꿔야 한다 (코퍼스 하위폴더와 scene_graph 위치가 다르다).

### `expand_preset_variants.py`

Cinematographer 카메라 하나를 **여러 trajectory preset 변형**으로 불린다.
Director 가 chunk 당 `camera_count: 1` 을 내고 VLM 이 preset 하나만 고르기 때문에,
확정된 시작/끝 pose 는 그대로 두고 preset 만 바꿔 궤적 데이터셋을 만든다.

```bash
python fit/bank/expand_preset_variants.py \
  --output_root ../Look-Before-Move/Cinematographer/output \
  --run_glob 'trumans_c49_w*' --num_presets 4 --dry_run
```

env: 미확정 (json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--output_root` | (필수) | `Cinematographer/output` |
| `--run_glob` | `trumans_c49_w*` | |
| `--num_presets` | `4` | |
| `--seed` | `0` | |
| `--include_static` | (플래그 쌍) | |
| `--vary_end` | (플래그 쌍) | 끝 pose 도 바꾼다 |
| `--min_travel_m` | `0.08` | |
| `--dry_run` | (플래그 쌍) | |
| `--backup_suffix` | `.variants.bak` | |

> 비고: **원본 산출물을 제자리에서 고친다.** 먼저 `--dry_run`, 그리고 `--backup_suffix` 백업을 남길 것.

### `retime_camera_handoff.py`

Cinematographer 산출물의 카메라 길이를 chunk 프레임 수에 맞춘다. Stage 2 가 낸
`target_frame_count` 가 110/80/95 로 제각각이라 49프레임 chunk 와 어긋나는 것을 고친다.

```bash
python fit/bank/retime_camera_handoff.py \
  --output_root ../Look-Before-Move/Cinematographer/output \
  --run_glob 'trumans_c49_w*' --frame_count 49 --fps 25 --dry_run
```

env: 미확정 (json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--output_root` | (필수) | `Cinematographer/output` |
| `--run_glob` | `trumans_c49_w*` | |
| `--frame_count` | `49` | chunk 프레임 수 |
| `--fps` | `25` | TRUMANS 씬 fps |
| `--dry_run` | (플래그 쌍) | |
| `--backup_suffix` | `.bak` | |

### `trumans_first_pose_board.py`

`.blend` 에서 chunk 별 **시작 pose 후보판**을 만든다 — 게이트만, VLM 없이.
DA3 좌표계는 첫 프레임이 항등이지만 그 프레임이 blender 씬의 어디인지를 몰라 "카메라가 벽 속인가 /
사람이 보이는가"를 씬 기하로 물을 수 없었다. 이 워커가 방위각 × 고도 × 반경 격자를 씬에서 직접 친다.

```bash
B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
R=/data1/cympyc1785/data/trumans/Data_release/Recordings_blend/00add26c-7a26-4a61-b192-b97aa493b3f3
$B -b $R/00add26c-7a26-4a61-b192-b97aa493b3f3.blend \
  --python fit/bank/trumans_first_pose_board.py -- \
  --chunk_starts 51 100 149 --num_frames 49 --frame_step 1 --out out/board_00add26c
```

env: **Blender 내장 python**

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--out` | (필수) | 결과 폴더 |
| `--chunk_starts` | `[]` | |
| `--chunk_stride` | `0` | |
| `--num_frames` | `49` | Lite 고정값 |
| `--frame_step` | `1` | |
| `--az_step` | `30.0` | 12방위 |
| `--elevations` | `[-10.0, 10.0, 25.0, 45.0]` | |
| `--radii` | `[1.1, 1.5, 2.0, 2.6]` | |
| `--radii_rel` | `[]` | |
| `--fit_margin` | `1.15` | `d_ref` 의 프레임 여유 |
| `--subject_kind` | `human` | `human/event/object` |
| `--prop_names` | `[]` | `obj_list.txt` 이름 |
| `--anchor_origin` | `obb_center` | |
| `--anchor_frame` | `mid` | `mid/start` |
| `--gate_frames` | `window` | `window/anchor` |
| `--aim_bias` | `0.20` | |
| `--facing_sample_step` / `--facing_min_speed` | `10` / `0.02` | |
| `--facing_bone` | `CC_Base_Hip` | TRUMANS 리그는 CC_Base (SMPL-X 아님) |
| `--facing_min_agreement` / `--facing_min_side_frac` | `0.35` / `0.70` | |
| `--eye_mesh` | `["CC_Base_Eye"]` | |
| `--probe_distance` | `1.5` | |
| `--min_clearance` / `--min_subject_dist` / `--min_height` | `0.20` / `0.80` / `0.30` | |
| `--scene_margin` | `0.50` | 씬 AABB 밖 허용 |
| `--min_crop_keep` | `0.85` | OBB 가 화면 안에 남는 비율 |
| `--min_area_frac` / `--max_area_frac` | `0.03` / `0.60` | |
| `--max_center_offset` | `0.60` | |
| `--lens` / `--sensor` / `--res` | `25.0` / `36.0` / `[960, 540]` | |
| `--samples` | `16` | EEVEE TAA |
| `--max_render` | `0` | 0 = 통과분 전부 |
| `--render_rejects` | `0` | |
| `--render_select` | `crop` | `crop/diverse` |
| `--no_render` | off | |

> 비고: `--min_crop_keep` 으로 "최선"을 고르면 벽 너머 칸이 1등으로 올라오는 사례가 있다.
> 보행 chunk 는 창 이동량이 1.9 m 를 넘어 정지 first-pose 가 0장 나올 수 있다.

---

## `caption/` — 뱅크 → 캡션 → 학습 코퍼스 / holdout split / avg_scale

| 파일 | 역할 |
| --- | --- |
| `build_bank_captions.py` | 뱅크 변이 → `{target, event, framing, motion}` 캡션 + 학습 프롬프트 (주 경로) |
| `make_holdout_split.py` | 씬 단위 층화 holdout — train:test = 9:1, val 은 test 안의 1% |
| `make_avg_scale_vista4d_ctxall.py` | Vista4D 코퍼스의 분모(`avg_scale`)를 context 점군 전체 기준으로 재계산 |
| `make_avg_scale_trumans.py` | TRUMANS Lite 클립의 `avg_scale.json` 을 기존 recon 폴더에서 채운다 |
| `gendop_style_captions.py` | 우리 코퍼스 문장을 GenDoP 학습 분포 어휘로 다시 쓴다 (motion+target) |
| `corpus_captions_to_eval_dir.py` | 코퍼스 `prompts*.json` → eval 폴더 모양의 캡션 디렉토리로 미러 |
| `caption_cameras_datadop.py` | 우리 카메라에 DataDoP 파이프라인 그대로 캡션을 단다 (태깅+LLM) |
| `caption_datadop_chunks.py` | DataDoP 원본 shot 을 49프레임 chunk 로 잘라 태깅만, 우리 plan-text 형식으로 |

### `build_bank_captions.py`

뱅크 행에는 카메라 기하(`preset`, `tau_max`, `subject_area_med`, `anchor_label`)만 있고 문장이 없다.
이 스크립트가 네 필드(`target`/`event`/`framing`/`motion`)를 **따로** 저장하고 학습 프롬프트는 그 중 일부만 조립한다
(필드마다 신뢰도가 다르기 때문). **D121 부터 기본은 자연어 한 문장**(`--prompt_style nl`)이고,
**D176 부터 nl 기본 문장은 `target` + `camera` 두 축만** 남긴다 — 크기 부사와 framing 은 실측과 문장이 어긋나서 뺐다.
`--magnitude` / `--nl_framing` 으로 예전 문장을 되살리고, `--prompt_style fields` 는 옛 형식을 문자 단위로 재현한다.

```bash
python fit/caption/build_bank_captions.py --videos all --bank_dir hole_bank_k6_d99
python fit/caption/build_bank_captions.py --videos camel --bank_dir hole_bank_k6_d99 --dry_run
# 예전 형식 그대로
python fit/caption/build_bank_captions.py --videos all --bank_dir hole_bank_k6 \
  --prompt_style fields --out_name captions_fields.json
```

env: 미확정 (json/csv 만 쓴다. VLM 호출 없음 — 지칭구는 `instance_desc.json` 에서 읽는다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--videos` | `["all"]` | all = 뱅크 있는 전량 |
| `--videos_file` | `""` | 한 줄 1편. `--videos` 보다 우선 |
| `--bank_dir` | `hole_bank_k6` | |
| `--output_root` | `None` | |
| `--metadata_csv` | `/data1/cympyc1785/data/Vista4D-Eval-Data/metadata.csv` | `event` 필드 출처 |
| `--presets` | `<CINEMATRAJ_ROOT>/configs/caption_presets.json` | preset 문구 표 |
| `--prompt_style` | `nl` | `nl/fields` |
| `--prompt_fields` | `target,motion` | `fields` 경로에서 쓸 필드 |
| `--out_name` | `captions.json` | |
| `--targetless_promote` | `False` | `--no_targetless_promote` |
| `--legacy_dolly_phrase` | `False` | 옛 "without re-aiming" 문구 복원 |
| `--label_map` / `--desc_override` / `--external_shapes` | `""` | |
| `--anchor_desc` | `True` | `--no_anchor_desc` — 지칭구 사용 |
| `--magnitude` | `None` | 3-state: 안 주면 형식이 정하고, 주면 못 박는다 |
| `--framing_timeline` | `True` | `--no_framing_timeline` |
| `--nl_framing` | `False` | `--no_nl_framing` — nl 문장에 framing 절 |
| `--framing_on_free` | `True` | `--no_framing_on_free` |
| `--framing_min_in_frame` | `0.85` | |
| `--framing_exit_min_in_frame` | `0.05` | |
| `--dry_run` | `False` | `--no_dry_run` |

> 비고: 크기 부사(`adverb_of`)는 요청이 아니라 **실현치**(`tau_max`/`pan_deg`)를 읽는다 — 설계상 그렇다.
> 눈금이 `knob_kind` 별로 다르므로(tau 계열은 |t|/z_med, pan 계열은 각도) 같은 버킷 표를 두 축에 쓰면 pan 이 전부 "dramatically" 가 된다.
> `metadata.csv` 가 없으면 `event` 가 비고 `--videos all` 목록도 안 만들어진다 (TRUMANS 는 `build_trumans_metadata.py` 선행).

### `make_holdout_split.py`

씬 단위 층화 holdout 목록 + seg_list 재작성. train:test = 9:1, val 은 test 안의 1% 다.
**재굽기가 필요 없다** — `vista4d_bank_to_dl3dv.py` 의 `--test_videos` 는 각 줄이 어느 seg_list 로 가는지만 정하고
chunk/npz 는 건드리지 않으므로, 이 스크립트는 기존 seg_list 두 개를 읽어 다시 나누기만 한다.

```bash
python fit/caption/make_holdout_split.py \
  --out_root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
  --src_prefix seg_list_dynpose --dst_prefix seg_list_dynpose_s91 \
  --test_frac 0.10 --val_frac 0.01
```

env: 미확정 (텍스트 처리)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--out_root` | 상수 (`latentcam_dynpose_...`) | 코퍼스 루트 |
| `--src_prefix` | `seg_list_dynpose` | 읽을 seg_list 접두사 |
| `--dst_prefix` | `seg_list_dynpose_s91` | 쓸 접두사 |
| `--chunk_prefix` | `dynpose` | seg_list 줄의 앞머리 |
| `--test_frac` / `--val_frac` | `0.10` / `0.01` | 코퍼스 대비 카메라 비율 |
| `--batch_size` | `8` | `val_max_batches` 환산용 |
| `--forced_test` / `--forced_val` | `<CINE>/configs/...` | 반드시 넣을 씬 목록 |
| `--seed` | `200` | |
| `--dry_run` | `False` | `--no_dry_run` |

> 비고: val 은 별도 파일이 아니라 **test 목록의 앞부분**이다. 비율을 바꾸면 `seg_list` 와
> `val_max_batches` 를 **같이** 고쳐야 한다. CLaTr 게이지는 train split 을 따라가므로 재분할하면 옛 ckpt 가 새 test 를 이미 본 상태가 된다.

### `make_avg_scale_vista4d_ctxall.py`

Vista4D latentcam 뱅크의 **분모(`avg_scale`)** 를 context 점군 전체 기준으로 다시 잰다.
기존 분모는 `scene_graph.json:scale.S`(frame0 한 장의 non-sky 평균 ray 길이)인데,
context range 는 [0,49) 전체라 카메라가 움직여 새 영역이 들어오는 씬에서 어긋난다.

```bash
python fit/caption/make_avg_scale_vista4d_ctxall.py --dry_run
python fit/caption/make_avg_scale_vista4d_ctxall.py --workers 8
python fit/caption/make_avg_scale_vista4d_ctxall.py --videos camel avocado-slice --no_skip_done
```

env: 미확정 (numpy). `fit.ingest.trumans_to_recon.avg_scale_first_cam` 을 import 한다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--eval_data` | 상수 | `recon_and_seg` 의 부모 |
| `--recon_root` | 상수 | 영상 목록만 여기서 |
| `--vista4d_root` | 상수 | |
| `--out_root` | 상수 | latentcam 레이아웃 루트 |
| `--ref_dir` | 상수 | `<scene>/da3/<여기>/` |
| `--videos` | `["all"]` | |
| `--stride` | `2` | 픽셀 서브샘플 |
| `--workers` | `8` | |
| `--skip_done` | `True` | `--no_skip_done` |
| `--dry_run` | off | |

> 비고: `intr_norm`/`avg_scale` 은 ckpt 속성이라 학습 때와 어긋나면 recon 이 터진다.
> `vae_latent_scale` 계열과 이름이 겹치니 어느 `avg_scale` 인지 확인할 것.

### `make_avg_scale_trumans.py`

TRUMANS Lite 클립의 `avg_scale.json` 을 **이미 만들어 둔 recon 폴더에서** 채운다.
avg_scale 은 depth 와 카메라만 있으면 되므로 96편을 다시 렌더(편당 ~3분)할 필요가 없다.
`--from_render` 는 work 디렉토리가 살아 있을 때 float32 원본 depth 를 쓴다.

```bash
python fit/caption/make_avg_scale_trumans.py --root /data1/cympyc1785/data/TRUMANS-Lite/eval_data
python fit/caption/make_avg_scale_trumans.py \
  --manifests '<CinemaTraj>/out/trumans_recon/*/manifest_a*.json' --from_render
```

env: 미확정 (numpy)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--root` | `""` | `<...>/eval_data` |
| `--manifests` | `[]` | glob 패턴 |
| `--from_render` | off | float32 원본 depth 사용 |
| `--stride` | `AVG_SCALE_STRIDE` | |
| `--skip_done` | off | |
| `--no_write` | off | 계산만 |
| `--check` | off | 기존 값과 대조 출력 |
| `--patch_manifest` | off | `manifest["render"]` 에도 심는다 |
| `--vista4d_root` | 상수 | |

> 비고: 이 스크립트만 `--skip_done` 기본값이 **off** 다 (다른 곳은 대개 `True`).

### `gendop_style_captions.py`

우리 코퍼스 entry 를 **GenDoP 학습 분포의 문장**(motion + target 만)으로 다시 쓴다.
우리 D121 자연어 캡션은 `framing`·`composition` 절이 붙어 있고 어휘도 우리 태거 것이라
GenDoP 에 그대로 넣으면 분포가 어긋난다.

```bash
python fit/caption/gendop_style_captions.py \
  --corpus /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121 \
  --split .../seg_list_vista4d_test.txt --prefix vista4d \
  --out results/20260906_d156_gendop_d121/text_gendop_style
```

env: 미확정 (json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--corpus` | (필수) | latentcam 레이아웃 루트 |
| `--split` | (필수) | `seg_list_*.txt` (`<prefix>/<scene>/<idx>`) |
| `--prefix` | `vista4d` | 코퍼스 하위 폴더 == 파일명 접두사 |
| `--out` | (필수) | `<out>/test/*_caption.json` |

### `corpus_captions_to_eval_dir.py`

코퍼스 `prompts*.json` 문장을 **eval 폴더 모양의 캡션 디렉토리**로 미러한다.
`gendop_release_infer.py --text_from_eval_dir` 와 `run_director_batch.py --text_dir` 가 둘 다
`<dir>/test/<prefix>_<scene>_<idx>_caption.json` (키 `Concise Interaction`) 을 읽기 때문이다.

```bash
python fit/caption/corpus_captions_to_eval_dir.py \
  --corpus /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
  --split .../seg_list_dynpose_s91_test.txt --prefix dynpose \
  --prompts_name prompts_mag.json \
  --out results/20260920_d208_gendop_d200/text_mag
```

env: 미확정 (json)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--corpus` / `--split` / `--out` | (필수) | |
| `--prefix` | `dynpose` | eval 폴더 파일명 접두사 |
| `--prompts_name` | `prompts.json` | |
| `--field` | `prompt_camera_with_scene_video` | |
| `--key` | `concise` | |

### `caption_cameras_datadop.py`

Vista4D eval 카메라(합성 target / recon 소스)에 **DataDoP 방식 그대로** 카메라 캡션을 단다.
GenDoP 이 DataDoP 캡션 분포로 학습됐으므로 "비슷한 말"이 아니라 **같은 파이프라인 산출물**이어야 한다.
`--no_llm` 이면 태깅(분절+outline)까지만 하고 LLM 을 안 부른다.

```bash
# 태깅만 — 방향 라벨이 preset 이름과 맞는지 먼저 본다
conda run -n GenDoP python fit/caption/caption_cameras_datadop.py --sets cameras --no_llm
# 전량 캡션 (로컬 Qwen3-VL)
conda run -n GenDoP python fit/caption/caption_cameras_datadop.py --sets cameras recon
```

env: **GenDoP** (docstring 예시가 `conda run -n GenDoP python`). LLM 경로는 vLLM 서버 필요.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--root` | `/data1/cympyc1785/LatentCamVid/DATA/...` | |
| `--sets` | `["cameras", "recon"]` | |
| `--datadop_root` / `--datadop_valid` / `--datadop_out` | 상수 | |
| `--datadop_limit` / `--datadop_seed` | `0` / `0` | 0 = valid 전량 |
| `--latentcam_root` | 상수 | |
| `--latentcam_split` | `None` | None = `<root>/seg_list_vista4d_test.txt` |
| `--num_poses` | `120` | DataDoP 고정값, 0 = 리샘플 안 함 |
| `--fps` / `--static_threshold` / `--diff_threshold` / `--angular_static_threshold` / `--smoothing_window_size` / `--min_chunk_size` | `SEG_DEFAULTS` | 분절기 파라미터 |
| `--shuffle_taxonomy` | (플래그 쌍) | |
| `--only` | `None` | `"scene/name"` 몇 개만 |
| `--gendop_root` | `VENDORED_GENDOP` | |
| `--out_subdir` | `captions` | |
| `--no_llm` | off | 태깅만 |
| `--api_base` / `--model` | 상수 | |
| `--overwrite` | off | |
| `--num_shards` / `--shard_id` | `1` / `0` | |

### `caption_datadop_chunks.py`

DataDoP 원본 shot 22,314개를 **49프레임 chunk 로 잘라 태깅만** 하고 우리 plan-text 형식으로 낸다.
DataDoP 은 물체 anchor 가 없으므로 `target` 은 `none` 고정이고 `motion` 만 GenDoP 분절기 어휘로 채운다.
`--num_poses 0` 이 기본이라 리샘플하지 않는다(규약 ②).

```bash
python fit/caption/caption_datadop_chunks.py --limit 50
for i in 0 1 2 3 4 5 6 7; do
  python fit/caption/caption_datadop_chunks.py --num_shards 8 --shard_id $i &
done; wait
python fit/caption/caption_datadop_chunks.py --merge
```

env: **GenDoP** (`SEG_DEFAULTS` 등 GenDoP 분절기를 import 한다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--datadop_root` / `--valid_txt` | 상수 | `""` 면 전량 |
| `--out` | `None` | |
| `--num_frames` | `49` | chunk 길이 |
| `--chunks_per_shot` / `--stride` | `3` / `36` | |
| `--num_poses` | `0` | 0 = 리샘플 안 함 |
| `--fps` / `--static_threshold` / `--diff_threshold` / `--angular_static_threshold` / `--smoothing_window_size` / `--min_chunk_size` | `SEG_DEFAULTS` | |
| `--join` | `", then "` | chunk 사이 접속어 |
| `--max_scenes` / `--limit` | `0` / `0` | |
| `--num_shards` / `--shard_id` | `1` / `0` | |
| `--progress` | `200` | 0 이면 조용히 |
| `--merge` | `False` | 샤드 병합 |
| `--dry_run` | `False` | |

> 비고: DataDoP world 단위는 MonST3R 게이지(pairwise 스케일 기하평균 0.5 고정)라 divisor 없이(D=1) 쓰는 게 맞다.

---

## `convert/` — 뱅크를 Blender / Vista4D / DL3DV 포맷으로

| 파일 | 역할 |
| --- | --- |
| `vista4d_bank_to_dl3dv.py` | Vista4D pseudo-GT 뱅크 → latentcam DL3DV `da3` 포맷 (학습 코퍼스) |
| `bank_to_vista4d_cams.py` | 뱅크 변이 또는 **모델 예측 궤적** → Vista4D eval 카메라 npz |
| `bank_to_blender_poses.py` | 뱅크 변이 → TRUMANS `.blend` 월드 카메라 npz + `render.sh` |

### `vista4d_bank_to_dl3dv.py`

`out/<video>/hole_bank*/` 뱅크를 latentcam 이 읽는 **DL3DV `da3` 온디스크 포맷**으로 굽는다.
새 dataset 클래스를 안 만드는 이유는 `trumans_lite_to_dl3dv.py` 와 같다 (`dl3dv_root` 를 갈아끼우면 돈다).
다른 것은 **target 궤적이 소스 궤적이 아니라는 것** 하나다.
`--bank_dirs` 로 여러 세대를 pool 하고 `--per_scene_cap` 으로 씬당 상한을 건다.

```bash
python fit/convert/vista4d_bank_to_dl3dv.py --dry_run
python fit/convert/vista4d_bank_to_dl3dv.py --workers 8
python fit/convert/vista4d_bank_to_dl3dv.py --bank_dir hole_bank_k6 --test_videos camel bmx-bumps
python fit/convert/vista4d_bank_to_dl3dv.py --drop_status clamped_low
python fit/convert/vista4d_bank_to_dl3dv.py \
  --bank_dirs hole_bank_d185 hole_bank_d198 hole_bank_d199 --per_scene_cap 6
```

env: 미확정 (numpy + 이미지 I/O, 멀티프로세스)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--cine_out` | `<CINEMATRAJ_ROOT>/out` | |
| `--bank_dir` | `hole_bank_k6` | 단일 세대 |
| `--bank_dirs` | `[]` | 여러 세대 pool (비면 `--bank_dir`) |
| `--per_scene_cap` | `0` | 0 = 상한 없음 |
| `--captions_name` | `captions.json` | |
| `--recon_root` / `--out_root` | 상수 | |
| `--videos` | `["all"]` | |
| `--meta_csv` | `meta_vista4d.csv` | |
| `--chunk_prefix` | `vista4d` | |
| `--test_videos` | 상수 | |
| `--test_hash_mod` | `0` | |
| `--test_roundrobin` | (플래그 쌍) | |
| `--seg_list_prefix` | `seg_list_vista4d` | |
| `--image_dir` | `images_4` | dl3dv `IMAGE_DIR_NAMES` 첫 후보 |
| `--image_scale` | `0.5` | |
| `--avg_scale_refs` | 상수 | |
| `--dedup` | `True` | `--no_dedup` |
| `--drop_status` | `[]` | 예: `clamped_low` |
| `--drop_suspect` | `[]` | `SUSPECT_TAGS` |
| `--picked_only` | `False` | `--no_picked_only` |
| `--workers` | `8` | |
| `--skip_done` | `True` | `--no_skip_done` |
| `--dry_run` | off | |

> 비고: `--test_videos` 는 각 줄이 어느 seg_list 로 가는지만 정한다 — chunk/npz 는 안 건드린다
> (그래서 `make_holdout_split.py` 가 export 없이 재분할할 수 있다).
> `variant_id` 는 코퍼스 키가 아니다 — 같은 이름으로 재굽기하면 캡션↔변이가 어긋난 채 통과한다.
> 세대를 새로 구우면 코퍼스 폴더 이름도 새로 줄 것.

### `bank_to_vista4d_cams.py`

뱅크 변이 **또는 모델 예측 궤적**을 Vista4D eval 카메라 npz(`eval_data/cameras/<video>/<tag>.npz`)로 쓴다.
`tools/recammaster/vista4d_prepare.py` 는 canonical(rmax=1 정규화 상대 궤적)만 받으므로 이쪽을 따로 둔다.
`--csv` 를 주면 `render_eval`/`inference_eval` 용 metadata 행도 같이 쓴다.

```bash
python fit/convert/bank_to_vista4d_cams.py --video snowboard \
  --variants fixk_track_bank:dyn_0__pedestal_up__tau0.6__steady__lock__b0__fauto__s9__k3 \
  --tag_prefix ct_ --csv <repo>/tmp/d221/meta.csv --seed 52106

python fit/convert/bank_to_vista4d_cams.py --video bmx-bumps --tag_prefix "" \
  --preds <...>/eval_my/d215_s42__last:vista4d_bmx-bumps_3=ct_d215_track_orbit_left_s42
```

env: vista4d (GPU 안 쓴다)

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | |
| `--variants` | `None` | `<bank_dir>:<variant_id>` |
| `--preds` | `None` | `<eval_dir>:<entry>=<tag>` |
| `--cams` | `None` | 직접 npz |
| `--pred_kind` | `pred` | `pred/ref` |
| `--frame0_tol` | `None` | 비면 bank 1e-6 / pred 0.05 |
| `--tag_prefix` | `ct_` | |
| `--eval_data` | 상수 | |
| `--cam_dir` | `None` | |
| `--output_root` | `None` | 뱅크 루트 (기본 `dataset/out`) |
| `--csv` | `None` | metadata 경로 |
| `--seed` | `"52106"` | metadata seed 열 |
| `--prompt` / `--dynamic` | `""` | metadata 열 |
| `--overwrite` | `False` | `--no_overwrite` |

> 비고: metadata 의 `dynamic` 열은 SAM3 키워드이고, **static anchor 도 이 열로 먹인다**.
> 버킷을 가르는 건 `kind` 가 아니라 `moving` 이므로 여기에 wall/floor 를 넣으면 벽이 static target 이 된다.
> `eval_dir` 이름에 태그를 안 물리면 eval 은 skip 하고 render 는 이전 preset 을 그리는데 둘 다 rc=0 이라 안 들킨다.

### `bank_to_blender_poses.py`

뱅크 변이를 TRUMANS `.blend` **월드 카메라 npz** 로 바꾸고, 그것을 돌릴 `render.sh` 를 생성한다.
depth warp 은 카메라가 크게 움직이면 화면 절반 이상이 hole 이 되어 "궤적이 나쁜 건지 warp 탓인지"를 못 가르므로,
Blender 로 실제 렌더하기 위한 경로다. `--raycast` 로 mesh 충돌을 실제 광선으로 다시 재고
`--raycast_solve` 면 이분법으로 배율을 낮춰 통과시킨다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY fit/convert/bank_to_blender_poses.py --video tru_0ac97866_a08_s3f0k6 \
  --output_root out_trumans_gt --bank_dir hole_bank_k6_d77 \
  --poses_npz out_trumans_gt/tru_0ac97866_a08_s3f0k6/hole_bank_k6_d77/poses_smooth_p4.npz \
  --anchor dyn_0 --out /data1/cympyc1785/LatentCamVid/tmp/blendcam_a08
bash /data1/cympyc1785/LatentCamVid/tmp/blendcam_a08/render.sh
```

env: 아무거나 (numpy 만 쓴다; 예시는 vista4d). Blender 는 `subprocess`.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` / `--out` | (필수) | `tru_<rec8>_a<NN>_<tag>` / 출력 폴더 |
| `--output_root` | `out_trumans_gt` | |
| `--bank_dir` | `hole_bank_k6_d77` | |
| `--poses_npz` | `""` | 비면 뱅크 기본 |
| `--anchor` | `dyn_0` | `""` 면 앵커 구분 없음 |
| `--presets` / `--variant_ids` | `[]` | 비면 전량 |
| `--recon_root` | `<CINEMATRAJ_ROOT>/out/trumans_recon` | |
| `--res` | `[640, 360]` | |
| `--rgb_samples` | `48` | |
| `--rgb_engine` | `cycles` | `cycles/eevee` |
| `--rgb_cdevice` | `GPU` | `CPU/GPU` |
| `--anim` | off | |
| `--gpu` / `--gpus` | `0` / `[]` | **0~3 만** |
| `--verify` | (플래그 쌍) | |
| `--raycast` | (플래그 쌍) | mesh 충돌 재측정 |
| `--deroll` | (플래그 쌍) | |
| `--raycast_max_rungs` | `0` | |
| `--raycast_solve` | (플래그 쌍) | 이분법으로 배율 낮춤 |
| `--raycast_iters` | `6` | 이분법 횟수(= Blender 기동 횟수) |
| `--min_raycast_scale` | `0.15` | |
| `--blender` | `BLENDER` 상수 | |
| `--min_clearance` / `--min_floor_drop` / `--min_clear_frac` / `--min_subject_dist` | `0.20` / `0.30` / `0.90` / `0.80` | 게이트 (m / 비율) |
| `--probe_distance` | `1.5` | |
| `--los_samples` / `--los_bands` / `--min_los_frac` | `0` / `5` / `0.0` | |
| `--aim_arms` | `""` | |

> 비고 (**깨짐 2건**):
> ① 203행이 `path.join(CINEMATRAJ_ROOT, "scripts", "trumans_scene_probe.py")` 를 부르는데
> 그 파일은 지금 `fit/ingest/trumans_scene_probe.py` 다 — `--raycast` 게이트(기본 on)가 동작하지 않는다.
> ② 570행이 생성하는 `render.sh` 가 `path.join(CINEMATRAJ_ROOT, "scripts", "trumans_gt_render.py")` 를 가리키는데
> 그 파일의 유일한 사본은 `viz/trumans_gt_render.py` 다.
>
> docstring 예시가 `--out /tmp/blendcam_a08`. `tmp/` 로 바꿀 것.
> mesh 충돌 판정은 여기 `--raycast` 에만 있다 — 뱅크끼리 비교하면 충돌 0건으로 나온다.
> `fit.bank.emit_bank` 에서 `FIXED_FALLBACK`, `decision_from_variant`, `span_frac_for` 를 import 한다.
