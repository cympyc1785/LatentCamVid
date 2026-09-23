# `viz/` 사용법

`viz/` 는 CinemaTraj 파이프라인의 **판정용 산출물**을 굽는 곳이다. 뱅크·eval·TRUMANS/LBM 이
내놓은 카메라(`poses.npz`, `transform_matrix`, `camera_handoff_v1.json`)를 받아
depth warp 릴 / 비교 그리드 / 진단 플롯 / viser 3D 뷰어로 바꾼다. **학습도 추론도 여기서 하지 않는다**
— 이미 있는 결과를 사람이 볼 수 있는 형태로 바꾸는 단계다.

## 실행 위치 — 반드시 CinemaTraj 루트에서

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/dataset
```

`viz/` 의 20여 개 파일이 자기 경로로부터 루트를 역산한다:

```python
CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, CINEMATRAJ_ROOT)
```

임포트 자체는 cwd 와 무관하게 풀린다. 문제는 **상대 경로 기본값**이다 —
`viser_scene --out_root 'out'`, `viz_tau_shape_topdown --out_root 'out_trumans'`,
`trumans_render_reel --work out/trumans_recon` 등은 전부 CT 루트 기준이다.
다른 디렉토리에서 돌리면 예외 없이 **빈 결과**가 나오고 rc 는 0 이다. 그래서 `cd <CT>` 가 규칙이다.

## 공통 규약

| 항목 | 규약 |
| --- | --- |
| mp4 인코딩 | **imageio + libx264** 만 (`codec="libx264", quality=6, macro_block_size=1`). cv2 의 `mp4v` 는 금지 — 파일은 멀쩡한데 VS Code 뷰어에서 안 열린다. `viz/` 전량 검사 결과 **위반 0건** (`board_contact_sheet.py` 의 `cv2.imwrite` 는 PNG contact sheet 라 해당 없음). |
| 비교 릴 배치 | **위 = 출력, 아래 = depth warp** 2줄. 출력만 보면 warp 탓인지 생성 탓인지 못 가른다. `render_director_depth --rows both`, `stack_videos --direction vertical` 이 이 배치를 만든다. |
| 영상+preset 요청 단위 | `source \| GT \| seed42 \| seed1234 \| seed2026` 를 이어붙인 것까지가 한 단위. `concat_videos.py` / `stack_videos.py` 로 합친다. |
| GPU | **0~3 만** 쓴다. `CUDA_VISIBLE_DEVICES=0` 처럼 명시. |
| 임시 파일 | `/data1/cympyc1785/LatentCamVid/tmp/` 아래. `/tmp` 는 쓰지 않는다 (시스템이 비운다). |
| 인터프리터 | latentcam `/data1/cympyc1785/miniconda3/envs/latentcam/bin/python`, vista4d `/data1/cympyc1785/miniconda3/envs/vista4d/bin/python`. 파일마다 docstring 의 `env:` 를 따랐고, 코드로 정할 수 없으면 **미확정**으로 적었다. |
| Blender | `trumans_*` 4종은 Blender 번들 python 으로만 돈다: `/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender -b <blend> -P viz/<script>.py -- <args>`. 번들 python 엔 imageio 가 없어서 **PNG 시퀀스만** 나온다. |
| `--cloud_source` | `lbm/render.py:add_cloud_source_args()` 가 붙이는 공용 인자. `npz`(기본) = `<out>/<video>/cloud.npz` 를 읽는다, `memory` = recon 에서 그 자리에서 굽는다(이때만 `--eval_data` 가 쓰인다). |

## 파일 목록

| 파일 | 역할 |
| --- | --- |
| `render_pred_depth_warp.py` | eval 예측 카메라로 점군을 warp 해 씬별 비교 릴을 굽는다 (핵심 판정 도구) |
| `render_eval_val_warp.py` | 위 스크립트를 한 프로세스 안에서 validation 전 씬에 돌려 씬당 mp4 1편 |
| `render_preset_grid_warp.py` | 한 anchor 에 대해 preset 여러 개를 격자로 warp 비교 |
| `render_target_cams_warp.py` | 배포 `target_poses.npz` 카메라를 warp 해 씬들을 격자로 |
| `render_target_swap_warp.py` | preset 고정, target 카메라만 바꿔가며 warp 비교 |
| `render_bank_videos.py` | 뱅크 variant 를 depth warp 프리뷰/클립으로 굽는다 |
| `render_director_depth.py` | director 결과 디렉토리의 샘플들을 rgb+depth 2줄 릴로 |
| `render_target_poses_depth.py` | 뱅크 카메라의 depth 맵을 npz 로 굽는다 (렌더 아님, 데이터 생성) |
| `preset_render_demo.py` | LBM camera handoff 를 읽어 preset 렌더 데모 명령을 구성 |
| `preset_render_report.py` | preset 데모 렌더 결과를 타일 영상/표로 정리 |
| `trumans_gt_render.py` | TRUMANS `.blend` 에서 rgb/depth/index pass 를 헤드리스 렌더 |
| `trumans_render_reel.py` | TRUMANS recon 렌더 결과를 격자 릴로 묶는다 |
| `trumans_approach_viz.py` | 접근 경로 march + clearance 를 Blender 안에서 시각화 |
| `trumans_frustum_viz.py` | board 후보 카메라 frustum 을 Blender 궤도 샷으로 |
| `trumans_raycast_viz.py` | poses 의 raycast/clearance 판정을 Blender 로 시각화 |
| `lbm_preview_reel.py` | LBM run 디렉토리의 프리뷰들을 한 릴로 |
| `lbm_render_reel.py` | LBM 실제 렌더(`renders/*/frames/*.png`)를 shot 별 mp4 + 릴로 |
| `walk_window_reel.py` | TRUMANS 보행 구간 채굴 결과를 창 단위 격자 릴로 |
| `aim_arm_reel.py` | 보간 arm(everyframe/kf6/...) 별 조준 궤적을 차트와 함께 비교 |
| `make_lite_previews.py` | recon_and_seg 의 seg overlay / depth 프리뷰 생성 |
| `seg_overlay_video.py` | 동적/정적 세그 마스크를 소스 영상 위에 겹친 영상 |
| `concat_videos.py` | 영상들을 시간축으로 이어붙인다 (라벨·간격 지원) |
| `stack_videos.py` | 영상들을 공간축(세로/가로/격자)으로 붙인다 |
| `monst3r_gen_videos.py` | 생성 영상들을 MonST3R 로 재구성해 pose/depth 를 뽑는다 |
| `board_contact_sheet.py` | board 후보 렌더를 정렬해 PNG contact sheet 한 장으로 |
| `plot_approach_clearance.py` | approach audit 의 clearance 곡선 플롯 + 프레임 PNG |
| `plot_eval_frustums.py` | eval 카메라 frustum 을 3D 궤도 애니메이션/정지컷으로 |
| `viz_g1_collision.py` | G1(뒤쪽 충돌) 게이트가 무엇을 걸렀는지 렌더로 확인 |
| `viz_gravity.py` | 중력축 추정 결과를 수직선/격자 오버레이로 검증 |
| `viz_tau_shape_topdown.py` | τ 사다리에 따라 궤적 모양이 어떻게 변하는지 탑다운 플롯 |
| `viz_f1_inplace.py` | DynPose 클립의 in-place(정지) 판정 f1 을 영상으로 |
| `viz_director_pilot.py` | director pilot run 들의 궤적을 평면 투영 애니메이션으로 |
| `viser_cloud.py` | 4D 점군 + 뱅크/번들 카메라를 브라우저에서 인터랙티브로 |
| `viser_scene.py` | 씬 그래프(정적/동적 노드 + 뱅크)를 viser 로 |
| `viser_frame.py` | 영상에서 특정 프레임을 골라 PNG 로 떨구는 viser 헬퍼 |

---

## 1. depth warp 릴

점군을 새 카메라에서 splatting 으로 다시 그린다. 구멍(hole)은 마젠타로 칠해진다.
**생성 결과를 판정하는 1차 도구**이고, 출력 영상만으로는 warp 문제와 생성 문제를 가를 수 없으므로
항상 warp 을 같이 낸다.

### `render_pred_depth_warp.py`

eval 이 떨군 예측 카메라(`transform_matrix`, OpenGL c2w)를 OpenCV 로 뒤집어 점군을 warp 한다.
`--eval_dir LABEL=DIR` 를 여러 번 주면 라벨별로 행이 늘어난 비교 릴이 나온다.
씬 하나당 `<name>__warp.mp4` + `all_entries.mp4` + `reel_<group>.mp4` 를 낸다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_pred_depth_warp.py \
    --video 2f8c1a... --entries 0 1 2 \
    --eval_dir d200=results/20260918_140904_dynpose_d200_da3 \
    --name_prefix dynpose --corpus_root /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3 \
    --cloud_source npz \
    --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/warp_demo
```

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | 필수 | 씬 이름 |
| `--eval_dir` | 필수, append | `LABEL=DIR` 형식, 여러 번 |
| `--entries` | `None` | 코퍼스 entry 인덱스들 |
| `--out_dir` | 필수 | 출력 디렉토리 |
| `--caption_dir` | `None` | append, 캡션 디렉토리 |
| `--corpus_root` | `/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3` | |
| `--eval_data` | `/data1/cympyc1785/data/Vista4D-Eval-Data` | |
| `--cloud_root` | `<CLOUD_ROOT>/out` | |
| `--name_prefix` | `vista4d` | 코퍼스 청크 접두사 |
| `--device` | `cuda` | |
| `--stride` / `--fps` | `1` / `12.0` | |
| `--with_source` / `--no_with_source` | `True` | 소스 영상 행 포함 |
| `--reel` / `--no_reel` | `True` | 그룹 릴 생성 |
| `--labels` / `--no_labels` | `True` | 타일 라벨 |
| `--label_mode` | `condition` (`condition`\|`target_motion`) | |
| `--order` | `index` (`index`\|`target_motion`\|`fscore`) | |
| `--allow_no_seg` | `False` | seg 없어도 진행 |
| `--allow_empty_dynamic_mask` | `False` | 동적 마스크 0개 허용 |
| `--temporal_persistence` | `auto` (`auto`\|`on`\|`off`) | |
| `--scores_csv` / `--f1_high` / `--f1_low` | `None` / `0.8` / `0.0` | `--order fscore` 용 |
| `--vista4d_root` | `VISTA4D_ROOT_DEFAULT` | |
| `--cloud_source` | `npz` | |

> 비고 — 동적 점이 0개인 `cloud.npz` 면 움직이는 물체가 49프레임 겹쳐 그려진다(NTP 렌더).
> `index.json` 의 `num_dynamic_points` 로 먼저 확인하고, 정말 없으면 `--allow_empty_dynamic_mask` 를
> 의식적으로 주는 것이지 기본으로 켜는 인자가 아니다.

### `render_eval_val_warp.py`

`render_pred_depth_warp.main()` 을 **한 프로세스 안에서** 씬 루프로 돌린다.
씬마다 프로세스를 띄우면 ~15 s 임포트가 씬 수만큼 곱해지기 때문이다.
씬당 `all_entries.mp4` 하나만 `<scene>.mp4` 로 끌어올리고 나머지는 지운다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_eval_val_warp.py \
    --entries_json /data1/cympyc1785/LatentCamVid/tmp/d201/val_moving_entries.json \
    --out_dir results/20260919_d200_val_warp --shard 0/3 \
    --eval_dir d200=results/20260918_140904_dynpose_d200_da3 \
    --name_prefix dynpose --cloud_source memory
```

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--entries_json` | 필수 | `[{scene, idx}, ...]` |
| `--out_dir` | 필수 | |
| `--shard` | `0/1` | `i/n` 로 씬을 나눈다 |
| `--overwrite` | `False` | |
| `--keep_entries` | `False` | 씬 폴더를 통째로 남긴다 |

> 비고 1 — 인식하지 못한 인자는 **그대로 `render_pred_depth_warp` 파서로 넘어간다**(`parse_known_args`).
> 위 표에 없는 플래그는 앞 절 표를 보면 된다.
> 비고 2 — **현재 임포트가 깨져 있다.** 34행 `from scripts import render_pred_depth_warp as rpdw` 가
> `ImportError` 를 낸다. 모듈이 `viz/` 로 옮겨졌으므로 `from viz import ...` 여야 한다.

### `render_preset_grid_warp.py`

anchor 하나를 고정하고 preset 여러 개를 격자로 나란히 warp 한다.
"이 앵커에서 어떤 preset 이 살아남는가" 를 한 장으로 본다.
출력은 `<out_dir>/<video>__<anchor>__preset_grid.mp4`.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_preset_grid_warp.py \
    --video snowboard --eval_dir results/20260918_140904_dynpose_d200_da3 \
    --anchor dyn_0 --presets orbit_left dolly_in truck_left \
    --columns 4 --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/grid
```

| 인자 | 기본값 |
| --- | --- |
| `--video` / `--eval_dir` / `--anchor` / `--out_dir` | 전부 필수 |
| `--presets` | `None` (전부) |
| `--columns` | `4` |
| `--corpus_root` | `/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3` |
| `--vista4d_root` | `VISTA4D_ROOT_DEFAULT` |
| `--device` / `--stride` / `--fps` | `cuda` / `1` / `12.0` |
| `--cloud_source` (+`--eval_data`) | `npz` |

### `render_target_cams_warp.py`

배포된 `target_poses.npz`(extrinsics 는 **w2c**) 카메라로 여러 씬을 warp 해 격자로 붙인다.
씬별 `<out_dir>/<scene>__target_warp.mp4`.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_target_cams_warp.py \
    --scenes snowboard camel avocado --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/tgt \
    --from_target_poses --per_preset 1 --columns 4
```

| 인자 | 기본값 |
| --- | --- |
| `--scenes` | 필수 (nargs `+`) |
| `--out_dir` | 필수 |
| `--root` | `EVAL_DATA_DEFAULT` |
| `--from_target_poses` | `False` |
| `--per_preset` | `1` |
| `--with_source` / `--no_with_source` | `True` |
| `--tile_width` / `--tile_height` / `--columns` / `--fps` | `480` / `270` / `4` / `12.0` |
| `--dl3dv_root` | `/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3` |
| `--device` | `cuda` |
| `--cloud_source` (+`--eval_data`) | `npz` |

### `render_target_swap_warp.py`

preset 을 고정하고 **target 카메라만** 갈아끼워 warp 한다. preset 이 원인인지 target 이 원인인지 가른다.
출력 `<out_dir>/<video>__<preset>__target_swap.mp4`.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_target_swap_warp.py \
    --video snowboard --eval_dir results/20260918_140904_dynpose_d200_da3 \
    --preset orbit_left --per_target 1 \
    --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/swap
```

| 인자 | 기본값 |
| --- | --- |
| `--video` / `--eval_dir` / `--preset` / `--out_dir` | 전부 필수 |
| `--per_target` | `1` |
| `--corpus_root` | `/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3` |
| `--device` / `--stride` / `--fps` | `cuda` / `1` / `12.0` |
| `--cloud_source` (+`--eval_data`) | `npz` |

---

## 2. 뱅크·preset 렌더

뱅크(`bank/`, `hole_bank_*`)가 가진 variant 카메라를 그대로 그려 재고를 확인하는 쪽.
1절이 "모델이 낸 카메라" 를 본다면 여기는 "우리가 구운 카메라" 를 본다.

### `render_bank_videos.py`

뱅크 디렉토리의 variant 를 depth warp 로 굽는다. 기본은 격자 `preview.mp4` 한 편,
`--per_variant` 면 `clips/<variant>.mp4` 도 따로 낸다.
`--anchors` / `--presets` / `--tau` / `--variant_ids` 로 재고를 좁힌다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_bank_videos.py \
    --video snowboard --bank_dir hole_bank_k6_d77 \
    --fixed_focal --presets orbit_left dolly_in --max_tiles 10 \
    --columns 5 --with_source --per_variant
```

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | 필수 | |
| `--bank_dir` | `bank` | |
| `--poses_npz` | `''` | 뱅크 대신 npz 직접 지정 |
| `--fixed_focal` / `--no_fixed_focal` | **`False`** | 아래 비고 |
| `--anchors` / `--presets` / `--variant_ids` | `None` | 필터 |
| `--tau` | `None` (float 리스트) | τ 필터 |
| `--deroll` / `--no_deroll` | `False` | |
| `--depth` / `--no_depth` | `False` | depth 행 추가 |
| `--depth_range` | `None` (2개) | |
| `--max_tiles` | `10` (`0` = 상한 없음) | |
| `--seed` | `0` | 타일 샘플링 |
| `--name` | `preview.mp4` | |
| `--label_prefix` | `''` | |
| `--pose_kind` | `auto` | |
| `--raw_preset_names` / `--no_raw_preset_names` | `False` | |
| `--columns` / `--tile_width` / `--tile_height` | `5` / `480` / `270` | |
| `--stride` / `--fps` | `1` / `12.0` | |
| `--with_source` / `--no_with_source` | `False` | |
| `--per_variant` / `--no_per_variant` | `False` | |
| `--sheet` / `--no_sheet` | `True` | contact sheet 도 |
| `--eval_data` / `--output_root` / `--vista4d_root` | `EVAL_DATA_DEFAULT` / `None` / `VISTA4D_ROOT_DEFAULT` | |
| `--seg_root` / `--seg_static_root` | `None` / `None` | |
| `--device` | `cuda` | |
| `--cloud_source` | `npz` | |

> 비고 (중요) — 릴 렌더는 **뱅크의 `fixed_focal` 설정을 따라야** 하는데, 이 스크립트는
> `bank.json` 의 `fixed_focal` 을 **읽지 않는다**. `--fixed_focal` 기본값이 `False` 라서
> 아무것도 안 주면 프레임별 DA3 focal 을 그대로 써서 화각이 떨린다(snowboard 기준 fx 진폭 6.99%,
> 가장자리 15 px/frame). 뱅크가 고정 focal 로 구워졌다면 **매번 `--fixed_focal` 을 명시**할 것.
> 같은 계열인 `render_target_poses_depth.py` 와 `viz_g1_collision.py` 는 이 플래그 기본값이
> **`True`** 다 — 세 스크립트의 기본값이 서로 어긋나 있다.

### `preset_render_demo.py`

LBM `Cinematographer/output/<run>/outputs/camera_handoff_v1.json` 을 읽어
preset 렌더 데모용 카메라를 만들고 (`--ladder` 면 travel 사다리까지) 실행 명령을 구성한다.
`--print_cmd` 로 명령만 찍어 보고 실행 여부를 고를 수 있다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/preset_render_demo.py \
    --run 20260901_trumans_run --demo_run trumans_presetdemo \
    --camera_index 0 --frames 49 --ladder --ladder_preset straight_ease --print_cmd
```

| 인자 | 기본값 |
| --- | --- |
| `--run` | 필수 |
| `--demo_run` | `trumans_presetdemo` |
| `--camera_index` | `0` |
| `--orbit_deg` / `--frames` | `45.0` / `49` |
| `--ladder` / `--no_ladder` | `False` |
| `--travels` | `0.23,0.4,0.6,0.8,1.0,1.2,1.4,2.0` |
| `--preset_end_scale` | `8.0` |
| `--ladder_preset` | `straight_ease` |
| `--forward` / `--no_forward` | `False` |
| `--print_cmd` / `--no_print_cmd` | `False` |

> 비고 — 인터프리터 **미확정** (docstring 에 env 표기 없음. vista4d python 으로 `--help` 는 통과).
> 그리고 LBM preset **이름은 실제 이동량과 다르다**: `pedestal_up` 은 상수 0.04 m 왕복(net 0),
> `straight_ease` 는 키프레임이 동일한 경우가 있다. 로그는 전부 "성공" 으로 찍히므로
> 이름만 보고 이동을 단정하지 말 것. 실효 이동 상한은 약 1.05 m (cap 1.4 × 가시성 가드 0.75).

### `preset_render_report.py`

`preset_render_demo` 로 구운 렌더 폴더를 훑어 preset 별 타일 영상(`<out>/preset_tiles.mp4`)과
표를 만든다. `--camdump` 로 카메라 덤프를 같이 읽어 이동량·회전량을 붙인다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/preset_render_report.py \
    --demo_run trumans_presetdemo \
    --camdump /data1/cympyc1785/LatentCamVid/tmp/viz_usage/camdump.json \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/preset_report
```

| 인자 | 기본값 |
| --- | --- |
| `--demo_run` | `trumans_presetdemo` |
| `--camdump` / `--out` | 둘 다 필수 |
| `--tile_w` / `--fps` | `320` / `12.5` |
| `--video` / `--no_video` | `True` |

> 비고 — 인터프리터 **미확정**. `--fps 12.5` 기본값은 LBM 렌더가 stride 2 로 저장된 것을 반영한다.

### `render_director_depth.py`

director 결과 디렉토리(`--result_dir`)의 샘플들을 **위 rgb / 아래 depth** 2줄 릴로 굽는다.
`<result_dir>/render/<tag>_rgb.mp4`, `_depth.mp4`, `concat.mp4`, `index.json` 이 나온다.
`--rows` 로 한 줄만 낼 수도 있다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_director_depth.py \
    --result_dir results/20260918_director_pilot --video snowboard \
    --rot et --rows both --columns 0 --fixed_focal
```

| 인자 | 기본값 |
| --- | --- |
| `--result_dir` / `--video` | 둘 다 필수 |
| `--subdirs` | `[]` |
| `--samples` / `--limit` | `None` / `0` |
| `--name` | `render` |
| `--rot` | `et` (`et`\|`lookat`) |
| `--look_at_bias` / `--subject_height_u` | `0.0` / `0.0` |
| `--fixed_focal` / `--no_fixed_focal` | `False` |
| `--depth_range` | `None` |
| `--rows` | `both` (`both`\|`warp`\|`depth`) |
| `--columns` / `--concat_name` | `0` / `concat.mp4` |
| `--tile_width` / `--tile_height` / `--fps` | `480` / `270` / `12.0` |
| `--eval_data` / `--output_root` / `--vista4d_root` | `EVAL_DATA_DEFAULT` / `None` / `VISTA4D_ROOT_DEFAULT` |
| `--device` / `--cloud_source` | `cuda` / `npz` |

> 비고 — `--rows both` 이 곧 "출력 위 / warp 아래" 규약이다. 문제 보고에는 이 2줄 형태로 낸다.
> `--rot et` 는 E.T. 규약(y-up·z-forward) 카메라를 뜻한다.

### `render_target_poses_depth.py`

이건 영상이 아니라 **데이터**를 만든다. 뱅크 카메라마다 depth 맵을 굽고
`<out>/<video>/target_depth/<variant_id>.npz` (depth `(T,H,W)` float16, hole=0) + `index.json` 을 남긴다.
`--preview` 를 주면 확인용 mp4 도 몇 편 낸다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/render_target_poses_depth.py \
    --videos snowboard camel --bank_dir hole_bank_k6 \
    --compress --check --preview 2
```

| 인자 | 기본값 |
| --- | --- |
| `--videos` | 필수 (nargs `+`) |
| `--bank_dir` | `hole_bank_k6` |
| `--dl3dv_root` | `/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3` |
| `--chunk_prefix` | `vista4d` |
| `--output_root` / `--out_dir` | `None` / `None` |
| `--fixed_focal` / `--no_fixed_focal` | **`True`** |
| `--variants` / `--limit` / `--scale` | `None` / `0` / `1.0` |
| `--compress` / `--no_compress` | `False` |
| `--check` / `--no_check` | `True` |
| `--preview` / `--fps` | `0` / `12.0` |
| `--dry_run` | `False` |
| `--device` / `--cloud_source` (+`--eval_data`) | `cuda` / `npz` |

> 비고 — `--fixed_focal` 기본값이 `True` 로 `render_bank_videos.py` 와 반대다. 두 산출물을
> 나란히 비교할 때 화각이 달라질 수 있다.

---

## 3. TRUMANS / LBM 릴

TRUMANS `.blend` GT 와 LBM(Look-Before-Move) 실행 결과를 보는 쪽.
`trumans_*` 4종은 **Blender 번들 python 안에서만** 돈다 (`bpy` 필요).

### `trumans_gt_render.py` (Blender)

TRUMANS `.blend` 를 헤드리스로 열어 지정 프레임의 `rgb,depth,index` pass 를 굽는다.
**depth/index 는 항상 Cycles 1spp** 로 뽑는다 — EEVEE_NEXT 에는 object index pass 가 없고
EEVEE 의 Z pass 는 실루엣에서 필터링돼 틀린 값이 나온다. rgb 만 EEVEE 로 빠르게 뽑을 수 있다.

```bash
BLENDER=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BLENDER -b /data1/cympyc1785/data/trumans/Data_release/scene.blend \
    -P viz/trumans_gt_render.py -- \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/gt --frames 0 24 48 \
    --passes rgb,depth,index --res 960 540 --samples 16 \
    --rgb_engine eevee --cdevice CPU
```

| 인자 | 기본값 |
| --- | --- |
| `--out` / `--frames` | 둘 다 필수 |
| `--frame_list` | `[]` |
| `--passes` | `rgb,depth,index` |
| `--poses` / `--scene_camera` / `--camera_pose_pkl` | `''` / `''` / `''` |
| `--res` / `--samples` | `[960, 540]` / `16` |
| `--rgb_engine` | `eevee` (`eevee`\|`cycles`) |
| `--rgb_samples` / `--rgb_bounces` | `128` / `8` |
| `--lens` / `--sensor` | `25.0` / `36.0` |
| `--cdevice` | `CPU` (`CPU`\|`GPU`) |
| `--rgb_cdevice` | `None` (`CPU`\|`GPU`) |
| `--keep_exr` / `--no_keep_exr` | `False` |
| `--anim` | `False` |

> 비고 1 — 장치 플래그가 `--cdevice` 인 이유: Cycles 애드온이 argv 를 접두사 매칭해서
> **`--cycles` 로 시작하는 CLI 플래그를 쓸 수 없다**.
> 비고 2 — Blender 번들 python 에 imageio 가 없어 **PNG 시퀀스만** 나온다. mp4 가 필요하면
> 이후 `concat_videos.py` / `stack_videos.py` 로 vista4d env 에서 묶는다.
> 비고 3 — `.blend` 는 3.3.6 저작인데 4.5.9 로 연다. EEVEE_NEXT 로 렌더하면 흰 번짐이 생기고,
> 같은 프레임 Cycles 는 번짐 0 이다. 투과 여부 판정은 index pass 로 한다.

### `trumans_frustum_viz.py` (Blender)

board 후보 카메라들의 frustum 을 씬 안에 그려 궤도 샷으로 돌린다.
`--only usable|rejected` 로 게이트를 통과한/걸린 후보만 볼 수 있다.

```bash
BLENDER=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BLENDER -b <scene.blend> -P viz/trumans_frustum_viz.py -- \
    --board out/trumans_board/<rec>/board.json \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/frustum \
    --only usable --orbit_frames 72 --cdevice GPU
```

| 인자 | 기본값 |
| --- | --- |
| `--board` / `--out` | 둘 다 필수 |
| `--chunk` | `None` |
| `--only` | `all` (`all`\|`usable`\|`rejected`) |
| `--frustum_len` / `--thickness` / `--emission` | `0.28` / `0.01` / `4.0` |
| `--rings` / `--no_rings` | `True` |
| `--hide_scene` / `--no_hide_scene` | `False` |
| `--orbit_frames` / `--orbit_radius` / `--orbit_scale` | `72` / `0.0` / `2.6` |
| `--orbit_elev` / `--orbit_az0` / `--orbit_lens` | `28.0` / `0.0` / `32.0` |
| `--clip_cut` | `1.35` |
| `--world` / `--world_strength` | `keep` (`keep`\|`dark`) / `1.0` |
| `--res` / `--samples` / `--bounces` | `[960, 540]` / `32` / `4` |
| `--cdevice` | `GPU` (`GPU`\|`CPU`) |

> 비고 — `--cdevice GPU` 라도 GPU 는 **0~3** 만. `CUDA_VISIBLE_DEVICES` 로 제한하고 띄운다.

### `trumans_raycast_viz.py` (Blender)

`poses.npz` 의 각 카메라에서 나가는 ray 와 clearance 판정을 씬 안에 그린다.
"왜 이 pose 가 걸렸는가" 를 눈으로 확인하는 용도.

```bash
BLENDER=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BLENDER -b <scene.blend> -P viz/trumans_raycast_viz.py -- \
    --poses out/trumans_bank/<rec>/poses.npz \
    --audit out/trumans_bank/<rec>/audit.json \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/raycast \
    --ray_stride 6 --min_clearance 0.2
```

| 인자 | 기본값 |
| --- | --- |
| `--poses` / `--audit` / `--out` | 전부 필수 |
| `--ray_stride` / `--probe_distance` / `--min_clearance` | `6` / `1.5` / `0.2` |
| `--lens` / `--sensor` / `--frustum_len` / `--thickness` | `32.0` / `36.0` / `0.3` / `0.012` |
| `--near_emphasis` | `none` (`none`\|`ball`\|`stripe`\|`both`) |
| `--near_ball_scale` / `--emission` | `4.0` / `4.0` |
| `--hide_scene` / `--no_hide_scene` | `False` |
| `--orbit_frames` / `--orbit_radius` / `--orbit_scale` | `72` / `0.0` / `2.4` |
| `--orbit_elev` / `--orbit_az0` / `--orbit_lens` | `24.0` / `0.0` / `32.0` |
| `--clip_cut` | `1.35` |
| `--world` / `--world_strength` | `keep` / `1.0` |
| `--res` / `--samples` / `--bounces` | `[960, 540]` / `24` / `4` |
| `--cdevice` | `GPU` |

> 비고 — **현재 임포트가 깨져 있다.** 50~56행 `from trumans_scene_probe import ...` 가
> `viz/` 를 `sys.path` 에 넣고 찾는데, `trumans_scene_probe.py` 는 `fit/ingest/` 에 있다.
> Blender 안에서 `ModuleNotFoundError` 가 난다.

### `trumans_approach_viz.py` (Blender)

시작점에서 방향을 잡고 한 발씩 전진하며(`--steps`) clearance 가 임계값 아래로 떨어지는 지점을
표시한다. 접근 경로 게이트를 눈으로 검증하는 도구.

```bash
BLENDER=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
$BLENDER -b <scene.blend> -P viz/trumans_approach_viz.py -- \
    --audit out/trumans_bank/<rec>/approach_audit.json \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/approach \
    --dir_mode farthest_wall --steps 40 --thresholds 0.5 0.35 0.2 0.1
```

| 인자 | 기본값 |
| --- | --- |
| `--audit` / `--out` | 둘 다 필수 |
| `--start` | `None` (3개) |
| `--start_from_path` / `--start_frame` / `--start_height` | `0` / `0` / `0.0` |
| `--frame` | `-1` |
| `--direction` | `None` (3개) |
| `--dir_mode` | `farthest_wall` (\|`nearest_wall`) |
| `--scout_distance` / `--steps` / `--stop_at` / `--probe_distance` | `8.0` / `40` / `0.05` / `1.5` |
| `--thresholds` | `[0.5, 0.35, 0.2, 0.1]` |
| `--post_rise` / `--text_size` | `0.55` / `0.11` |
| `--lens` / `--sensor` / `--frustum_len` / `--thickness` / `--emission` | `32.0` / `36.0` / `0.3` / `0.012` / `4.0` |
| `--obs_radius` / `--obs_scale` / `--obs_elev` | `0.0` / `2.2` / `22.0` |
| `--obs_side` / `--obs_lens` | `auto` (\|`flip`) / `32.0` |
| `--clip_cut` / `--cut_above` | `1.15` / `0.7` |
| `--focus` / `--focus_pad` | `march` (\|`crossings`) / `0.35` |
| `--world` / `--world_strength` | `keep` / `1.0` |
| `--res` / `--samples` / `--bounces` | `[960, 540]` / `24` / `4` |
| `--cdevice` | `GPU` |

> 비고 — `trumans_raycast_viz.py` 와 **같은 임포트 버그**다 (48~55행 `from trumans_scene_probe import ...`).

### `trumans_render_reel.py`

`out/trumans_recon` 아래의 렌더 폴더들을 mp4 로 묶는다. `--grid` 면 한 장의 격자 영상
(`<out>/<name>`), 아니면 렌더별 `<out>/<label>.mp4`.

```bash
$PY viz/trumans_render_reel.py \
    --tag s3f0k6 --limit 6 --grid --cols 3 --scale 0.5 --fps 12.5 \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/trumans_reel
```

| 인자 | 기본값 |
| --- | --- |
| `--work` | `<CT>/out/trumans_recon` |
| `--tag` | `s3f0k6` |
| `--renders` / `--dirs` | `None` / `None` |
| `--name` | `grid.mp4` |
| `--limit` / `--stride` / `--fps` | `6` / `1` / `12.5` |
| `--overlay` / `--no_overlay` | `False` |
| `--grid` / `--cols` / `--scale` | `False` / `3` / `0.5` |
| `--out` | `<CT>/out/trumans_reel` |

> 비고 — env 는 아무거나 (imageio 만 있으면 된다). `--work` 기본값이 상대 경로 조합이라
> CT 루트에서 돌려야 한다.

### `lbm_render_reel.py`

LBM run 디렉토리의 **실제 렌더**를 shot 별 mp4 로 굽고 이어붙인다.
진짜 프레임은 `renders/*/frames/*.png` 에 있다. 출력은
`<out_dir>/<run_id>__<shot_id>.mp4` + `<out_dir>/<run_id>__reel.mp4`.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/lbm_render_reel.py \
    --run_dir /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/Look-Before-Move/VideoEngineer/output/<run_id> \
    --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/lbm --fps 12.5
```

| 인자 | 기본값 |
| --- | --- |
| `--run_dir` | 필수 (`<LBM>/VideoEngineer/output/<run_id>`) |
| `--out_dir` | 필수 |
| `--fps` | `25.0` |
| `--width` | `0` (0 = 원본 폭) |

> 비고 — LBM 의 `clip.mp4` 는 **1프레임짜리**다. 그걸 보고 "렌더가 비었다" 고 판단하면 안 된다.
> 실제 렌더는 stride 2 로 저장돼서 **`--fps 12.5` 가 의도 속도**이고, 기본값 `25.0` 은 2배속이다.

### `lbm_preview_reel.py`

LBM run 디렉토리에 있는 프리뷰 이미지/영상들을 한 편으로 묶어 빠르게 훑는다.
각 항목을 `--hold_sec` 만큼 정지시켜 보여준다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/lbm_preview_reel.py \
    --run_dir <LBM>/VideoEngineer/output/<run_id> \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/lbm_previews.mp4
```

| 인자 | 기본값 |
| --- | --- |
| `--run_dir` | 필수 |
| `--out` | `/data1/cympyc1785/LatentCamVid/tmp/lbm_previews.mp4` |
| `--hold_sec` / `--fps` | `1.6` / `25` |
| `--width` / `--height` | `1280` / `800` |

### `walk_window_reel.py`

TRUMANS 녹화에서 보행 구간을 채굴해 창(window) 단위 격자 릴을 만든다.
TRUMANS 는 y-up 이라 수평면은 `(x, z)` 이고, 보행은 Actions 라벨에 **0건**이라 여기서 직접 캔다.

```bash
$PY viz/walk_window_reel.py \
    --recordings 00add26c 0a761819 --per_rec 1 --cols 3 --fps 15 \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/walk_windows.mp4
```

| 인자 | 기본값 |
| --- | --- |
| `--recordings` | `SEQUENCE` 전체 (`00add26c, 0a761819, 0aa05d5a, 1d43e076, 2b4c9b84, 3a19c7bb, 4ac2c1b3`) |
| `--per_rec` / `--cols` / `--scale` | `1` / `3` / `0.5` |
| `--fps` / `--true_fps` | `15` / `30.0` |
| `--walk_speed` / `--walk_frac` | `0.4` / `0.8` |
| `--walk_smooth` / `--walk_max` | `9` / `8` |
| `--trumans` | `/data1/cympyc1785/data/trumans/Data_release` |
| `--out` | `<results>/2026-08-23_trumans_walk/walk_windows.mp4` |

> 비고 — **현재 임포트가 깨져 있다.** 30~31행이 `viz/` 를 `sys.path` 에 넣고
> `from trumans_to_recon import NUM_FRAMES, mine_walk_actions` 를 하는데, 그 모듈은
> `fit/ingest/trumans_to_recon.py` 에 있다 → `ModuleNotFoundError`.
> 또 채굴량 자체가 상한이라 `walk_ratio 0.30` 은 도달 불가하고 실제 뱅크는 8%, 편당 0~4개다.

### `aim_arm_reel.py`

보간 arm(`everyframe`, `kf6_smoothstep`, `kf6_smooth_kf`)별로 같은 preset 을 나란히 놓고
조준 각속도 차트를 옆에 붙인다. **arm 간 위치는 동일**해야 하므로 `max|Δt| < 1e-6` 를 assert 한다.

```bash
$PY viz/aim_arm_reel.py \
    --cams out/aim_cams --preset orbit_left \
    --arms everyframe kf6_smoothstep kf6_smooth_kf \
    --kf 0 10 19 29 38 48 \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/aim_arms.mp4
```

| 인자 | 기본값 |
| --- | --- |
| `--cams` / `--preset` / `--out` | 전부 필수 |
| `--arms` | `['everyframe', 'kf6_smoothstep', 'kf6_smooth_kf']` |
| `--kf` | `[0, 10, 19, 29, 38, 48]` |
| `--chart_width` / `--fps` / `--gap` | `460` / `12.0` / `6` |

> 비고 — arm 을 고를 때 **jerk 로 판정하지 말 것**. cubic 은 jerk 가 구간 상수라 구조적으로 이긴다.
> 판정은 `|dv|` 로. 그리고 전 구간 slerp 은 keyframe 6개와 양립하지 않는다 —
> 바뀌는 건 회전량이 아니라 타이밍이다.

---

## 4. 영상 합성 유틸

다른 단계의 산출물을 붙이고 겹치는 범용 도구들. 납품 단위
(`source | GT | s42 | s1234 | s2026`) 를 만드는 것도 여기다.

> 그룹 배치 변경 1건 — `make_lite_previews.py` 는 요청서에서 TRUMANS/LBM 릴로 분류돼 있었으나,
> 실제로는 recon_and_seg 의 seg overlay / depth 프리뷰를 만드는 코드라 `seg_overlay_video.py` 와
> 같은 계열이다. 그래서 여기로 옮겼다.

### `concat_videos.py`

영상들을 **시간축**으로 이어붙인다. 납품 단위를 만드는 기본 도구.

```bash
$PY viz/concat_videos.py \
    --inputs source.mp4 gt.mp4 s42.mp4 s1234.mp4 s2026.mp4 \
    --labels source GT seed42 seed1234 seed2026 \
    --output /data1/cympyc1785/LatentCamVid/tmp/viz_usage/delivery.mp4 --fps 25.0
```

| 인자 | 기본값 |
| --- | --- |
| `--inputs` / `--output` | 둘 다 필수 |
| `--labels` | `None` |
| `--gap_frames` | `0` |
| `--background` | `20` |
| `--fps` | `25.0` |

> 비고 — env 는 아무거나 (imageio 만 필요).

### `stack_videos.py`

영상들을 **공간축**으로 붙인다. 세로/가로/격자.
"출력 위 / depth warp 아래" 2줄 규약이 `--direction vertical` 이다.

```bash
$PY viz/stack_videos.py \
    --inputs out.mp4 warp.mp4 --labels output warp \
    --direction vertical --gap 6 --fps 12.0 \
    --output /data1/cympyc1785/LatentCamVid/tmp/viz_usage/out_over_warp.mp4
```

| 인자 | 기본값 |
| --- | --- |
| `--inputs` / `--output` | 둘 다 필수 |
| `--direction` | `vertical` (\|`horizontal`) |
| `--gap` / `--background` / `--fps` | `6` / `20` / `12.0` |
| `--columns` / `--tile_width` | `None` / `None` |
| `--labels` | `None` |
| `--hold_short` / `--no_hold_short` | `False` |

> 비고 — 길이가 다른 입력은 기본적으로 짧은 쪽에서 끊긴다. `--hold_short` 를 주면
> 짧은 영상의 마지막 프레임을 유지한다.

### `seg_overlay_video.py`

동적/정적 세그 마스크를 소스 영상 위에 겹친 영상(`<out_dir>/<video>_seg.mp4`)을 만든다.
"이 씬에서 무엇이 동적으로 잡혔는가" 를 확인하는 1차 도구.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/seg_overlay_video.py \
    --video snowboard --banks hole_bank_k6_d77 \
    --alpha 0.55 --fps 10.0 --static \
    --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/seg
```

| 인자 | 기본값 |
| --- | --- |
| `--video` | 필수 |
| `--eval_data` | `/data1/cympyc1785/data/Vista4D-Eval-Data` |
| `--vista4d_root` | `VISTA4D_ROOT_DEFAULT` |
| `--out` | `<CT>/out` |
| `--out_dir` | `''` |
| `--banks` | `['hole_bank_k6_d77']` |
| `--fps` / `--alpha` | `10.0` / `0.55` |
| `--drop` / `--no_drop` | `False` |
| `--static` / `--no_static` | `True` |

### `make_lite_previews.py`

recon_and_seg 결과에서 `seg_overlay.mp4` 와 `depth.mp4` 를 만들고 둘을 겹친 비교본까지 낸다.
LBM-Lite 입력이 제대로 들어왔는지 확인하는 용도.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/make_lite_previews.py \
    --eval_data /data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM \
    --video <scene> --alpha 0.45 \
    --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/lite
```

| 인자 | 기본값 |
| --- | --- |
| `--eval_data` / `--video` / `--out_dir` | 전부 필수 |
| `--vista4d_root` | `VISTA4D_ROOT_DEFAULT` |
| `--seg_root` / `--seg_static_root` | `None` / `None` |
| `--alpha` | `0.45` |
| `--fps` | `0.0` (0 = recon fps 를 따른다) |

### `monst3r_gen_videos.py`

생성 영상(`video_seed=*.mp4`)을 MonST3R 로 재구성해 `pred_traj.txt`, `pred_intrinsics.txt`,
`frame_depth_%04d.npy`, `dynamic_mask_%d.png`, `scene.glb`, `run_meta.json` 을
각 preset 폴더의 `monst3r/` 아래 남긴다. **GPU 를 오래 쓴다** — 샤딩해서 돌린다.

```bash
# env: GenDoP conda env
CUDA_VISIBLE_DEVICES=0 python viz/monst3r_gen_videos.py \
    --root /data1/cympyc1785/LatentCamVid/DATA/Vista4D-Eval-Data/eval_data \
    --pattern 'video_seed=*.mp4' --num_shards 4 --shard_id 0 \
    --image_size 512 --niter 300 --skip_done
```

| 인자 | 기본값 |
| --- | --- |
| `--root` | `.../DATA/Vista4D-Eval-Data/eval_data` |
| `--pattern` | `video_seed=*.mp4` |
| `--scenes` | `None` |
| `--device` | `cuda` |
| `--image_size` | `512` (\|`224`) |
| `--niter` / `--schedule` | `300` / `linear` |
| `--min_conf_thr` | `1.1` |
| `--scenegraph_type` / `--winsize` | `swinstride` / `5` |
| `--temporal_smoothing_weight` | `0.01` |
| `--translation_weight` | `'1.0'` (문자열) |
| `--flow_loss_weight` / `--flow_loss_start_iter` / `--flow_loss_threshold` | `0.01` / `0.1` / `25` |
| `--fps` / `--num_frames` | `0` / `200` |
| `--shared_focal` / `--no_shared_focal` | `True` |
| `--skip_done` / `--no_skip_done` | `True` |
| `--num_shards` / `--shard_id` / `--limit` | `1` / `0` / `0` |

> 비고 — MonST3R 는 **shot 단위로 스케일을 정규화**한다. 그래서 여기서 나온 world 단위를
> 다른 코퍼스와 그대로 비교하면 안 된다. gradio stub 을 주입해 vendored repo 를 임포트한다.

### `board_contact_sheet.py`

board 후보 렌더들을 정렬해 **PNG 한 장**(`board_<chunk_tag>.png`)으로 붙인다. 영상이 아니다.
`--sort_by` 로 crop / shot size / facing / angle 순 정렬.

```bash
$PY viz/board_contact_sheet.py \
    --board out/trumans_board/<rec>/board.json \
    --sort_by crop --columns 6 --tile_width 480
```

| 인자 | 기본값 |
| --- | --- |
| `--board` | 필수 |
| `--out` | `''` (비면 board 디렉토리) |
| `--columns` | `6` |
| `--sort_by` | `crop` (\|`shot`\|`facing`\|`angle`) |
| `--tile_width` | `480` |

> 비고 1 — 출력이 `cv2.imwrite` 인데 **PNG 라 mp4v 금지 규약과 무관**하다.
> 비고 2 — 20행 주석대로 `import sys` 를 써야 한다. `from sys import path` 로 미리 묶으면
> cv2 가 `sys.path` 를 새 리스트로 갈아끼워서 insert 가 죽은 리스트에 들어가고
> `ModuleNotFoundError` 가 난다.
> 비고 3 — contact sheet PNG 한 장으로는 궤적이 어디서 무너지는지 안 보인다.
> 궤적 판정에는 타일 애니메이션(`render_bank_videos --sheet` 가 아니라 `preview.mp4`)을 쓴다.
> 비고 4 — 인터프리터 **미확정** (docstring 에 env 표기 없음).

---

## 5. 플롯·진단

수치가 왜 그렇게 나왔는지 보는 쪽. 대부분 matplotlib 플롯이거나, 게이트 하나를 겨냥한 렌더다.

### `plot_eval_frustums.py`

eval 카메라의 frustum 을 3D 로 그려 궤도 애니메이션(`<name>_frustum.mp4`)과 정지컷 PNG 를 낸다.
`--ref_eval_dir` 와 `--eval_dir`(append) 를 겹쳐 GT 대비 예측을 본다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
$PY viz/plot_eval_frustums.py \
    --ref_eval_dir results/<gt_dir> --eval_dir results/<pred_dir> \
    --video snowboard --entries 0 1 2 --prefix vista4d \
    --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/frustums
```

| 인자 | 기본값 |
| --- | --- |
| `--ref_eval_dir` / `--video` / `--entries` / `--out_dir` | 전부 필수 |
| `--eval_dir` | `[]` (append) |
| `--prefix` | `vista4d` |
| `--stride` / `--frustum_len` | `3` / `0.1` |
| `--mark_frame` / `--pad` / `--cmap` | `None` / `1.05` / `viridis` |
| `--orbit_frames` / `--elev` / `--az0` | `90` / `18.0` / `-60.0` |
| `--fps` | `15.0` |

> 비고 — env 는 **latentcam** (docstring 명시: matplotlib + imageio). GPU 를 쓰지 않는다.
> eval JSON 의 `transform_matrix` 는 OpenGL c2w 다 — OpenCV 로 보려면 `diag(1,-1,-1,1)` 을 곱한다.

### `plot_approach_clearance.py`

approach audit JSON 의 clearance 곡선을 그린다(`<out>/clearance.png`),
`--frames` 면 프레임별 PNG(`frame_%05d.png`)도 같이 낸다.

```bash
$PY viz/plot_approach_clearance.py \
    --approach out/trumans_bank/<rec>/approach_audit.json \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/clearance --frames
```

| 인자 | 기본값 |
| --- | --- |
| `--approach` / `--out` | 둘 다 필수 |
| `--width` / `--height` | `960` / `540` |
| `--frames` / `--no_frames` | `True` |

> 비고 — 인터프리터 **미확정**.

### `viz_g1_collision.py`

G1(카메라 뒤쪽 충돌) 게이트가 무엇을 걸렀는지 렌더로 확인한다.
`--worst` 로 가장 심한 variant 를 자동으로 고른다. 출력은 `--out` 또는 `<bank>/g1_<variant>.mp4`.
`--no_render` 를 주면 GPU 없이 수치만 본다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
CUDA_VISIBLE_DEVICES=0 $PY viz/viz_g1_collision.py \
    --video snowboard --bank_dir hole_bank --worst \
    --behind_src_frames 13 --behind_margin_frac 0.02 --tile_cols 5
```

| 인자 | 기본값 |
| --- | --- |
| `--video` | 필수 |
| `--output_root` / `--bank_dir` | `None` / `hole_bank` |
| `--variant` / `--worst` / `--out` | `None` / `False` / `None` |
| `--behind_src_frames` | `13` |
| `--behind_margin_frac` / `--behind_clear_frac` / `--behind_radius_px` | `0.02` / `0.1` / `2` |
| `--render` / `--no_render` | `True` |
| `--tile_width` / `--tile_cols` / `--fps` | `214` / `5` / `8` |
| `--fixed_focal` / `--no_fixed_focal` | **`True`** |
| `--device` | `cuda` |
| `--eval_data` / `--vista4d_root` | `EVAL_DATA_DEFAULT` / `VISTA4D_ROOT_DEFAULT` |
| `--seg_root` / `--seg_static_root` | `None` / `None` |

> 비고 — 여기도 `--fixed_focal` 기본값이 `True` 로 `render_bank_videos.py`(False) 와 어긋난다.

### `viz_gravity.py`

중력축 추정을 소스 영상 위에 수직선(`plumb`) 또는 격자(`grid`)로 그려 검증한다.
출력 `<out_root>/<video>/vis/gravity_<style>[_<compare>].mp4`, fps 는 recon fps 를 따른다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/viz_gravity.py --video snowboard --style plumb --compare cam_up --horizon
```

| 인자 | 기본값 |
| --- | --- |
| `--video` | 필수 |
| `--eval_data` | `/data1/cympyc1785/data/Vista4D-Eval-Data` |
| `--output_root` / `--vista4d_root` | `None` / `VISTA4D_ROOT_DEFAULT` |
| `--style` | `plumb` (\|`grid`) |
| `--compare` | `''` (\|`cam_up`) |
| `--plumb_len` | `0.3` |
| `--grid_cols` / `--grid_rows` | `5` / `3` |
| `--horizon` / `--no_horizon` | `True` |

> 비고 1 — 인터프리터 **미확정** (docstring 에 env 표기 없음).
> 비고 2 — 루트 변수 이름이 `HERE` 지만 값은 다른 파일의 `CINEMATRAJ_ROOT` 와 같다.
> 여기도 `import sys` 를 써야 한다 (cv2 의 `sys.path` 재바인딩).

### `viz_tau_shape_topdown.py`

τ 사다리를 따라 궤적 모양이 어떻게 변하는지 탑다운으로 겹쳐 그린다.
`--out` 이 비어 있지 않을 때만 `tau_shape_<video>__<anchor>.png` 와 `tau_shape_<video>.json` 을 쓴다
(상대 경로면 CT 루트에 붙는다).

```bash
$PY viz/viz_tau_shape_topdown.py \
    --video snowboard --anchor dyn_0 --bank_dir hole_bank_k6_d77 \
    --presets orbit_left,orbit_right --tau_lo 0.1 --tau_hi 3.0 --num_taus 10 \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/tau
```

| 인자 | 기본값 |
| --- | --- |
| `--video` | 필수 |
| `--out_root` | `out_trumans` (CT 루트 상대) |
| `--anchor` | `''` |
| `--bank_dir` | `hole_bank_k6_d77` |
| `--presets` | `orbit_left,orbit_right,orbit_left_pedestal_up` (콤마 구분) |
| `--taus` | `''` |
| `--tau_lo` / `--tau_hi` / `--num_taus` | `0.1` / `3.0` / `10` |
| `--auto_lo` / `--no_auto_lo` | `True` |
| `--out` | `''` (비면 저장 안 함) |

> 비고 1 — env 는 da3 / vista4d 아무거나 (docstring 명시).
> 비고 2 — 파일 마지막 줄에서 **`if __name__ == "__main__":` 가드 없이 `main()` 을 그냥 호출**한다.
> 즉 이 모듈을 import 하면 그 자리에서 실행된다. 다른 스크립트에서 함수만 가져다 쓰면 안 된다.
> 비고 3 — orbit 계열에서 τ 가 키우는 건 **sweep 각**이지 반경이 아니다.
> `obs_az_span` 이 좁으면 호가 원에서 떨어져 직선처럼 보인다.

### `viz_f1_inplace.py`

DynPose 클립이 실제로 정지(in-place)인지, drift 판정 f1 이 뭘 본 건지 영상으로 낸다.
`<out_dir>/<scene[:8]>__<node_id>__f1.mp4`.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/viz_f1_inplace.py \
    --clips <scene_a> <scene_b> --drift_thresh 0.05 --span_common 3.0 \
    --out_dir /data1/cympyc1785/LatentCamVid/tmp/viz_usage/f1
```

| 인자 | 기본값 |
| --- | --- |
| `--clips` / `--out_dir` | 둘 다 필수 |
| `--eval_data` | `/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM` |
| `--vista4d_root` | `VISTA4D_ROOT_DEFAULT` |
| `--out` | `<CT>/out_dynpose` |
| `--fps` | `10.0` |
| `--drift_thresh` / `--span_common` | `0.05` / `3.0` |

> 비고 — track drift 게이트는 **각도가 아니라 scene scale 단위**다. 안 움직이는 track 이 보이면
> 배선이 아니라 문턱 단위를 의심하고 `drift/d_ref` 로 판정한다.

### `viz_director_pilot.py`

director pilot run 들의 궤적을 평면(top/side)에 투영해 애니메이션으로 겹쳐 본다.
`--space` 로 좌표계를 고른다 (`et` = E.T. y-up·z-forward, `rel` = 상대, `grav` = 중력 정렬).

```bash
# env: GenDoP conda env
python viz/viz_director_pilot.py \
    --runs results/<run_a> results/<run_b> --space et --planes top side \
    --max_seeds 6 --fps 12 \
    --out /data1/cympyc1785/LatentCamVid/tmp/viz_usage/director_pilot.mp4
```

| 인자 | 기본값 |
| --- | --- |
| `--runs` / `--out` | 둘 다 필수 |
| `--max_seeds` / `--fps` | `6` / `12` |
| `--stick` | `0.35` |
| `--space` | `et` (\|`rel`\|`grav`) |
| `--planes` | `['top', 'side']` |
| `--source_cam` / `--no_source_cam` | `True` |
| `--aim_line` / `--flip_y` | `False` / `False` |

> 비고 — 좌우 부호를 **렌더 눈대중으로 판정하지 말 것**(2/2 틀렸던 전적). 본/열 부호로 검증한다.
> 각도 부호 요약에 median 은 쓰지 않는다 (±178 이 0 으로 떨어진다).

---

## 6. viser 인터랙티브

브라우저에서 3D 로 돌려 보는 뷰어들. 포트를 열고 붙어 있으므로 **장시간 점유하지 말 것**.
전부 `viser >= 1.1.0` 이 필요하다 — latentcam 의 viser 1.0.30 은 `handle.thickness` 에서 죽는다.
그래서 이 셋은 **vista4d** env 로 돌린다.

### `viser_cloud.py`

4D 점군 + 소스/플랜 카메라 궤적 + OBB + 뱅크/번들 variant 를 브라우저에서 돌려 본다.
`viz/` 에서 가장 큰 파일(1795행)이고 인자도 가장 많다. 여기엔 자주 쓰는 것만 적는다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/viser_cloud.py \
    --video snowboard --bank hole_bank_k6_d77 \
    --variant 'dyn_0__orbit_left_arc__tau0.35' \
    --up -y --port 8084
```

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | `snowboard` | |
| `--out` | `<CT>/out` | |
| `--port` | `8084` | |
| `--max_static` / `--max_dyn_per_frame` | `1500000` / `60000` | 점 상한 |
| `--point_size` | `0.0` (0 = 자동) | |
| `--cam_stride` | `1` | |
| `--cam_fov_deg` / `--cam_fov_src` | `60.0` (`FRUSTUM_FOV_DEG`) / `False` | |
| `--cam_scale` | `0.0` | |
| `--up` | `-y` (`+x,-x,+y,-y,+z,-z`) | 표시 상방향 |
| `--bank` / `--banks` / `--variant` | `''` / `[]` / `[]` | |
| `--bundle` / `--bundle_presets` / `--bundle_arms` | `''` / `[]` / `[]` | |
| `--pin` / `--max_pins` / `--pin_palette` | `[]` / `8` / `False` | 궤적 고정 |
| `--no_cloud` | `False` | 점군 끄기 |
| `--obb` / `--no_obb` | `True` | |
| `--ground_grid` / `--no_ground_grid` / `--ground_on` | `True` / `False` | |
| `--src_color` / `--src_now_color` | `''` / `''` | |
| `--plan_color` / `--plan_now_color` | `''` / `''` | |
| `--plan_gradient` / `--no_plan_gradient` | `True` | |
| `--plan_color_start` / `--plan_color_end` | `''` / `''` | |
| `--obb_color_dyn` / `--obb_color_static` | `''` / `''` | |
| `--paths_on` / `--probe_on` / `--now_cams` / `--source_on` | 전부 `False` | 초기 토글 |
| `--obb_anim` / `--obb_node` | `dyn` (\|`moving`) / `dyn_0` | |
| `--viewer_root` | `VIEWER_ROOT_DEFAULT` | |

> 비고 1 — 번들 경로는 `d221_bundles/<scene>/<preset>/` 처럼 **세대 접미사 없이** 놓는다
> (`warp_d241.mp4` 같은 이름 금지).
> 비고 2 — cloud world 는 OpenCV 규약이고, scene graph 의 obb/track 은 graph frame G 에 있다
> (`frames.T_wg` 가 있어야 world 로 온다).

### `viser_scene.py`

씬 그래프(정적/동적 노드)와 뱅크 카메라를 함께 띄운다. `viser_cloud` 보다 가볍고 노드 중심이다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/viser_scene.py --video snowboard --bank hole_bank_k6_d151 --port 8080
```

| 인자 | 기본값 |
| --- | --- |
| `--video` | 필수 |
| `--out_root` | `out` (CT 루트 상대) |
| `--bank` | `hole_bank_k6_d151` |
| `--pin` | `[]` (append) |
| `--max_static` / `--max_dynamic` | `300000` / `400000` |
| `--point_size` | `0.008` |
| `--cam_scale` / `--cam_thickness` / `--cam_stride` | `0.08` / `3.0` / `4` |
| `--static_nodes` / `--no_static_nodes` | `True` |
| `--host` / `--port` / `--seed` | `127.0.0.1` / `8080` / `0` |

> 비고 — `--out_root 'out'` 이 상대 경로다. CT 루트가 아니면 조용히 빈 씬이 뜬다.

### `viser_frame.py`

영상 여러 편을 띄워 놓고 특정 프레임을 골라 PNG 로 떨군다
(`<out_dir>/<prefix>_f<번호>.png`). 보고용 정지컷을 고를 때 쓴다.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
$PY viz/viser_frame.py --video a.mp4 b.mp4 --frame 24 --name_from parent --port 8095
```

| 인자 | 기본값 |
| --- | --- |
| `--video` | 필수 (nargs `+`) |
| `--port` | `8095` |
| `--out_dir` | `/data1/cympyc1785/LatentCamVid/tmp/frames` |
| `--frame` | `0` |
| `--name_from` | `parent` (\|`stem`\|`both`) |

---

## 부록 — 알려진 문제

| 파일 | 증상 |
| --- | --- |
| `render_eval_val_warp.py:34` | `from scripts import render_pred_depth_warp` → `ImportError`. 모듈이 `viz/` 로 옮겨졌다 (`from viz import ...` 여야 함). |
| `walk_window_reel.py:30-31` | `from trumans_to_recon import ...` → `ModuleNotFoundError`. 모듈은 `fit/ingest/trumans_to_recon.py`. |
| `trumans_approach_viz.py:48-55` | `from trumans_scene_probe import ...` → Blender 안에서 `ModuleNotFoundError`. 모듈은 `fit/ingest/trumans_scene_probe.py`. |
| `trumans_raycast_viz.py:50-56` | 위와 동일. |
| `viz_tau_shape_topdown.py` | `__main__` 가드 없이 모듈 최상위에서 `main()` 호출 — import 만 해도 실행된다. |
| `--fixed_focal` 기본값 불일치 | `render_bank_videos.py` = `False`, `render_target_poses_depth.py` = `True`, `viz_g1_collision.py` = `True`. |
