# pipeline.md — 영상 1편 → (source video, target camera, caption) 코퍼스

이 문서는 **데이터가 만들어지는 순서와 각 단계가 그 모양인 이유**만 다룬다.
2026-09-23 (R7) 에 트리가 `camera_generation/models/Planner/CinemaTraj/` 에서 여기로 옮겨왔고,
2026-09-22 (R6) 에 평평하던 `scripts/` 가 갈래로 나뉘었다 — **아래 경로는 전부 재분류 후 기준**이다.

| 무엇을 찾나 | 어디 |
|---|---|
| 각 결정의 근거·기각된 대안 | `DECISIONS.md` (D1~) |
| preset 어휘 46종 전량 + 캡션 문구 | `presets.md` |
| 설계 원칙 (왜 전수 열거인가, 왜 게이트인가) | `README.md` |
| **실행 방법** (인자·env·샤딩) | `USAGE.md` → `exec/` `fit/` `eval/` `viz/` USAGE |
| 변경 이력 | `CHANGELOG.md` |

---

## 0. 한 장 요약

```
                      ┌──────────────── ingest (코퍼스마다 다름) ────────────────┐
  원천 영상/씬  ─────► │  recon_and_seg/ · seg_instances/ · metadata.csv          │
                      └──────────────────────────┬───────────────────────────────┘
                                                 │  ← 여기부터 아래는 **코퍼스 공통**
        exec/run_bank.py  (단일 드라이버, 세대 = configs/bank/<gen>.json 한 장)
                                                 │
   graph ──► cloud ──► route ──► tau ──► fit ──► emit
     │         │         │        │       │        └ canonical.json (21 pose, rel c2w)
     │         │         │        │       └ 게이트를 안 깨는 최대 크기를 이분법으로
     │         │         │        └ anchor × preset × 사다리단 전수 열거 + 게이트
     │         │         └ anchor 몇 개 · preset 어느 슬롯 (씬당 카메라 수를 여기서 정한다)
     │         └ RGBD 49프레임 → world 점군 (기본 메모리 상주, 디스크에 안 남긴다)
     └ scene_graph.json — OBB · track · 중력 · 단위 S · 관계
                                                 │
        exec/run_corpus_export.py
   desc ──► merge_desc ──► captions ──► export ──► verify
                                                 └ DL3DV 레이아웃 (latentcam 학습 입력)
```

`run_bank.py` 의 여섯 단계는 `exec/run_bank.py:68 STAGE_ORDER` 와 `:79 STAGE_ENTRY` 가 원본이다.
**세대(dNNN)마다 달라지는 것은 스크립트가 아니라 config JSON 한 장이다** — 새 세대에 새 스크립트를
만들지 않는다.

### 단위·기호

- **`S`** — scene unit. frame0 non-sky 픽셀의 ray-depth 평균 (`scene_graph/scale.py`).
  `1 u ≜ S DA3 units`. 리포의 `avg_scale` / `S_da3` / `norm_scale` 과 같은 게이지다.
- **`τ(f)`** — `|p_plan(f) − p_src(f)| / z_med(f)`. plan 카메라가 소스 카메라에서 얼마나
  벗어났나를 그 프레임 median depth 로 나눈 값. **시차 예산**이다.
- **`hole`** — `1 − valid_mask.mean()`. 점군 렌더에서 비어 있는 픽셀 비율 =
  하류 video model 이 채워야 하는 양의 직접 측정치.
- **`knob`** — 궤적 크기 손잡이. preset 마다 의미가 다르다 (sweep 각 / dolly 거리 / …).
- **anchor** — 카메라가 겨누는 대상 노드 (`dyn_*` / `stat_*`). 시작 pose 가 아니다.

---

# Part A — 실사 영상 (Vista4D / DynPose-100K)

## A0. 입력 규약

모든 하류가 읽는 단 하나의 포맷은 Vista4D 배포본 레이아웃이다 (`scene_graph/io.py:load_scene`):

```
<eval_data>/eval_data/
  recon_and_seg/<video>/     video.mp4 · depths/ · masks/ · cameras.npz · predictions.npz
  seg_instances/<video>/     meta.json · masks.npz      # 동적 인스턴스 (SAM3)
  seg_instances_static/<video>/                          # 정적 인스턴스 (SAM3, 선택)
metadata.csv                 name,video,camera,seed,prompt,dynamic,do_sky_seg,source,video_id
```

- `cameras.npz` 의 `cam_c2w` 는 **frame0 앵커** (`cam_c2w[0] == I`), OpenCV 규약.
- 49프레임 고정.
- Vista4D 는 배포본이 이미 이 모양이다. DynPose-100K 는 `fit/ingest/dynpose_ingest.py` 가
  원본 영상에서 만든다 (`--stage link|recon|nouns|metadata|sam3|dynmask`, 샤딩 `--stage launch`).

> **`metadata.csv` 의 `dynamic` 열은 static anchor 도 먹인다.** 버킷을 가르는 것은 `kind` 가
> 아니라 `moving` 이다 — 그 열에 `wall`/`floor` 를 넣으면 벽이 static target 이 된다.

## A1. VLM 명사 — `fit/graph/extract_nouns_vlm.py`

영상에서 **detector 에 먹일 명사**를 뽑는다. `--frame_mode multi` 는 frame 0/16/32/48 네 장을
Qwen3-VL-30B (로컬 vLLM `http://127.0.0.1:22002/v1`) 에 보내고 `{dynamic:[], static:[]}` 를 받는다.
`single`(frame0 한 장)은 detector 공정비교용 조건이지 코퍼스용이 아니다 — 움직임이 보여야
dynamic/static 을 가를 수 있다.

명사는 `lowercase, singular, 1-2 words` 로 강제한다. SAM3/GDINO 는 caption 의 부분문자열을 라벨로
돌려주므로 형용사구가 길면 phrase 가 엉킨다. **긴 지칭 표현은 여기가 아니라 A10 에서 만든다.**

산출: `<output>/vlm_nouns.json` (`records[] = {video, frame_mode, dynamic, static}`).

> 이 스크립트는 **끝날 때 한 번만** 파일을 쓴다 (`main` 안의 `:233 json.dump` 하나뿐).
> 600편을 한 프로세스로 돌리면
> 590편째 예외 하나에 1~2시간이 날아가므로 드라이버가 `--merge` 로 25편씩 끊어 부른다.
> vLLM 서버는 **이 단계 직전에 올리고 끝나면 내린다** (`exec/serve_qwen3vl.sh`).

## A2. 인스턴스 분할 (SAM3 text PCS)

**동적** — keyword 출처는 `metadata.csv:dynamic` (저자 정답). DynPose 는
`fit/ingest/dynpose_ingest.py --stage sam3`, Vista4D 는 배포본이 이미 갖고 있다.

**정적** — 2단계다. 정적 명사는 아무도 정답을 안 적어놨기 때문이다.

1. `fit/graph/extract_static_nouns.py` — `vlm_nouns.json:static` → keyword. 두 가지를 거른다:
   - **동적과 겹치는 명사**를 버린다. 같은 물체가 `dyn_*` 와 `stat_*` 두 노드로 생기면
     3D IoU 0.5 를 못 넘겨 안 합쳐지고, VLM 프롬프트에 같은 물체가 두 번 나온다.
   - **광역 표면**(`ground`/`wall`/`floor`)을 버린다 (`--drop_surfaces` 기본 True). OBB 를 씌우면
     씬 전체를 덮는 상자가 되고 `near` 엣지가 모든 노드에 걸린다. 이 둘은 이미 기하로 처리된다
     (`relations.ground_height` / `relations.find_wall_planes`).
   - `--max_nouns 8`. 산출 `static_nouns.json` (`static_nouns_v1`) 에 버린 명사와 사유도 남는다.
2. `fit/ingest/sam3_static_instances.py` — 위 keyword 로 SAM3, 출력은 `seg_instances_static/`
   (배포본 `seg_instances/` 를 **안 덮는다**). 저장 전에 배포본 `dynamic_mask` 와의 겹침이
   `--max_dynamic_frac 0.5` 를 넘는 track 을 뺀다 — VLM 이 static 이라 부른 게 실제로 움직이는
   경우(avocado-slice `wheelchair`)가 있고, 그게 `stat_*` 가 되면 `supported_by` 가 "움직이는
   것에 얹혀 있다"는 틀린 관계를 만들어 그대로 캡션에 나간다. 뺀 track 은 `static_report.json`.

`static_mask` 가 이미 있는데 왜 SAM3 를 또 도나: `static_mask` 는 인스턴스 구분이 없는 한 장짜리
이진 마스크라 "the fence" 를 지목할 수 없다. 정적 노드의 용도는 자유공간이 아니라 **VLM/캡션이
물체를 이름으로 부르는 것**이다.

> **정지 소품 강등** (D177) — `fit/ingest/dynpose_dynamic_mask_from_seg.py` 가 `dynamic_mask/`
> 를 SAM3 인스턴스 합집합으로 바꾸면서, 실제로 안 움직이는 dyn 노드를 static 으로 내린다.
> 손에 쥔 물체는 여기서 안 건드린다 — 쥐고 흔들면 어차피 path 게이트가 잡는다.

## A3. 중력 사이드카 — `fit/ingest/geocalib_gravity.py`

`--num_frames 5` 프레임에서 GeoCalib 로 카메라 up 을 재고 `up_world = R_c2w[f] @ vec3d[f]` 로
world 로 올린 뒤 `--trim_keep_frac 0.75` 절사평균. 산출 `out*/<video>/geocalib_gravity.json`
(`up_world`, `confidence`, `spread_deg`). env 는 **`geocalib`** 전용 (kornia 가 vista4d/da3 에 없다).

왜 별도 단계인가: ground RANSAC 은 "지면 평면이 보여야" 되는데 vista 53편 중 15편(28%)이
`camera_up_fallback` 으로 떨어졌고, 통과한 38편 중에도 벽/책상 상판을 지면으로 문 게 여럿이다.
중력축이 틀어지면 OBB yaw/extent/center 와 `roll=0` 기준이 **전부 같이** 틀어진다.

---

## A4. `graph` — `fit/graph/build_scene_graph.py` (env `da3`)

영상 1편 → `out*/<video>/scene_graph.json` (스키마 `planner_scene_graph_v1`). 내부 순서:

| | 무엇 | 핵심 인자 |
|---|---|---|
| scale | `S`, `S_allpix`, `parallax_ratio`, `z_med` | `--scene_scale_mode points_first_cam --scene_scale_stride 1` |
| gravity | GeoCalib 사이드카 → `up_world` → `T_gw` | `--gravity_source geocalib` (기본, strict — 사이드카 없으면 assert) |
| lift | depth+mask → world 점군 | `valid = ~sky & isfinite & z>0 & conf>0.5 & ~dilate(&#124;∇log z&#124;>0.05, 2px)` |
| instances | track → 프레임별 점군, sliver 기각, 키워드 중복 병합 | `--min_area_frac 0.002 --min_frames 5 --min_score 0.3 --merge_iou 0.5`; 상한 `--max_dyn_nodes 6 --max_stat_nodes 10` |
| OBB | 중력정렬 `cv2.minAreaRect`, 저시차 깊이축 보정 | 동적은 프레임별 fit → yaw unwrap → 5프레임 median |
| relations | `supported_by` / `against_wall` / `near` | `--contact_u 0.05 --wall_u 0.08 --near_factor 1.5` |

`--gravity_source` 는 `geocalib`(strict) / `auto`(사이드카 없으면 조용히 ground RANSAC) / `gt`.
**코퍼스 전체가 같은 게이지를 써야 한다** — 절반만 사이드카가 있는 상태에서 `auto` 로 구우면
중력축이 씬마다 다른 출처에서 나온다 (D148).

`--seg_static_root` 는 기본 `None` = `<eval_data>/eval_data/seg_instances_static` 을 **있으면 자동
사용**한다 (`scene_graph/io.py:106`). 두 루트를 각각 `dyn` / `stat` 로 읽어 합친다.

산출 노드가 하류에서 실제로 소비되는 항목: `obb{center,extent,yaw,R}` · `track{center_smooth,
yaw,conf}` · `obs_az_span_deg` · `viewing_distance.d_ref` · `moving` · `label`/`aliases`/`near` ·
`gravity.up_world`.

검증용 `vis/obb_overlay.mp4` 를 같이 낸다 — **박스가 물체를 안 감싸면 하류가 전부 무효**라
사람 승인은 여기서 한 번 받는다 (`--no_overlay --no_topdown` 으로 끌 수 있다).

> **BLAS 스레드 캡을 반드시 건다** (`run_bank.py:69-77`). 안 걸면 graph 가 프로세스당 391
> threads 를 124코어 위에 띄워 서로를 기다린다 — 같은 인자 · 산출물 md5 동일인데
> user 12,326.7 s (3136% CPU) 대 254.0 s. 차이는 연산이 아니라 spin-wait 다.

## A5. `cloud` — `python -m lbm.cloud`

RGBD 49프레임을 world 점군으로 올린다. `visible (N,F)` 를 같이 받아 `visible.sum(1)==1` 인 것이
동적 점이다. 이 점군이 뒤의 모든 게이트(hole/충돌/구도)를 재는 **렌더러의 입력**이다.

렌더러는 우리 코드가 아니라 **Vista4D 트리를 `sys.path` 에 얹어 빌려 쓴다**
(`lbm/cloud.py:35 import_vista4d`). 기본 루트는 이 트리 기준 `../../video_generation/models/Vista4D`
(`lbm/cloud.py:28 VISTA4D_ROOT_DEFAULT`) 이고, 거기서 `utils/point_cloud/point_cloud.py` 의
`unproject` · `render` 와 `utils/point_cloud/preprocess.py` 의 `preprocess_scene` 을 가져온다.
Vista4D 는 패키지가 아니라 `utils.*` 로 절대 import 하는 스크립트 트리라서 import 가 아니라
경로 주입이다 — **Vista4D 트리가 옮겨가면 여기가 먼저 죽는다.**

**D178 부터 기본이 `--cloud_source memory` 다** — 점군을 `cloud.npz` 로 디스크에 안 남긴다.
in-process 실행(`"exec": "inproc"`)에서 tau/fit 이 같은 프로세스 안에서 점군을 공유하므로
재구축 비용이 없고, 10k 규모에서 디스크가 먼저 찬다. `npz` 를 쓰는 곳은 TRUMANS 처럼
점군을 여러 드라이버가 나눠 읽는 경우뿐이다.

> **cloud 는 dynmask 뒤에 구워야 한다.** 앞서 구우면 `dynamic_mask` 가 비어 동적 점이 0개가
> 되고, 그 씬은 hole 이 조용히 낮게 나온다 (D175-a/D176-c). 굽기 전 빈 마스크 가드가 있다.

`graph` 와 `cloud` 는 뱅크와 독립이라 **마커 파일**(`.graph_<gen>` / `.cloud_<gen>`)로 재실행을
막는다. 뱅크 축만 바뀐 재굽기는 이 두 단계를 안 돈다 — 같은 결과에 편당 ~90 s 를 쓰기 때문이다.

## A6. `route` — `fit/bank/route_presets.py`

**씬당 카메라가 몇 대 나오는지를 정하는 단계다.** anchor 를 고르고 각 anchor 에 preset 슬롯을
배정한다. 라우팅을 안 쓰면 anchor 전량 × preset 전량이 열거된다 (편당 수백 변이) — 씬이 1만 편
규모가 되면 변이가 곧 시간이다. 실측 비용은 `9.7분 + 8.80초 × 변이수` (R=0.887, 84편 회귀).

슬롯 어휘 8개: `recede / advance / lateral / rotate / arc / orbit / vertical / static`.
슬롯당 preset 을 고르는 신호는 전부 **fit 보다 앞서 나오는 것**들이다:

| 신호 | 쓰임 |
|---|---|
| `node.moving` | `track_*` 로 갈지. 안 움직이는 anchor 에 `track_` 을 붙이면 궤적이 비-track 짝과 비트 단위로 같아진다 |
| `obs_az_span_deg` | orbit 가능 여부. 좁으면 s_curve 로 대체 |
| `gravity.method` | `camera_up_fallback` 이면 "위"가 world up 이 아니다 → 세로 슬롯(pedestal/crane) 제외 |
| `cam_c2w_world` | 소스가 트럭한 **반대쪽**을 고른다 — 소스가 안 본 면이 hole 이 크고 그게 이 데이터의 값어치다 |

**씬당 카메라 수는 `--slot_plan` + anchor 상한의 곱이다.** 현행 두 배선:

| 배선 | 인자 | 씬당 |
|---|---|---|
| grid5 (D185) | `--max_dynamic_anchors 2 --max_static_anchors 2 --max_anchors 2 --slot_plan grid2x2 --free_moving rotate --target_variants 5` | anchor 2 × 슬롯 2 + free 1 = **최대 5** |
| single (D184/D187) | 같은 슬롯 계획에 `--max_anchors 1` | **1** |

`--slot_pair_fill substitute` 는 씨앗 쌍의 살아남은 짝을 유지하고 빠진 자리만 후보풀에서 메운다.
`GRID_ALLOWED_SLOTS` = advance/recede/arc/orbit/vertical/static — **s_curve 는 여기 없다**
(사용자 지시 2026-09-08).

track 여부는 `--track_min_drift_u 0.05` 가 정한다. 이동량이 문턱을 넘는 anchor 의
`TRACK_KEEP_SLOTS` 에만 `track_` 이 붙고, 못 미치면 같은 슬롯을 비-track 으로 굽는다
(= "dynamic subject 가 일정 이상 움직일 때만 tracking").

> **왜 `moving` 자체를 안 고치나**: `moving`(=`path_len_u > 0.05`)은 제자리 흔들림(춤·손짓·그네)을
> 통과시키지만, `schema.py:207` 의 anchor split key 라 강등하면 그 dyn 노드가 static 버킷으로
> 넘어간다. 그래서 track 판정만 따로 `--track_min_drift_u` 로 건다.

**anchor 상한은 여기서만 걸린다** (`route_presets.py:162` → `schema.pick_main_anchors`).
뱅크에 주면 조용히 죽는다 — route 가 `--nodes` 를 명시로 넘기고
`sample_camera_bank.anchor_nodes`(`:245`)가 그걸 받으면 **명시 목록이 1순위**라 필터를 안 탄다
(`:256 if node_ids:` 가 `pick_main_anchors` 호출보다 앞이다).

**DataDoP 외부 궤적 (`dd_*`)** — `--external_shapes configs/datadop_shapes.json --num_external N`.
실제 촬영본에서 검색한 궤적 모양을 preset 이 약한 슬롯부터 채운다. `--num_external 0` 이면
통째로 빠진다 (현행 세대는 전부 0).

> **`tau` 부터 굽는 세대는 route 를 빼면 안 된다.** route 를 건너뛰면 `anchors.json` 이 무시되고
> 기본 앵커가 돈다 — rc=0 이라 안 들킨다.

## A7. `tau` — `fit/bank/sample_camera_bank.py`

anchor × preset × τ 사다리로 후보 궤적을 만들고 게이트를 돌린다. 통과분만 `bank.csv` 에
`status == "solved"` 로 남는다. 다른 status: `collision_limited / obb_limited / elev_limited /
ground_limited / approach_limited`. **하류(reel·emit·export)는 전부 `solved` 만 본다.**

축은 **전부 명시한다** (D105 원칙 — 뱅크 정체성을 argparse 기본값에 맡기면 기본값이 뒤집혔을 때
한 폴더에 두 규약이 섞인다). D185 기준:

```
--tau_ladder 0.10 0.20 0.35 0.60 1.00
--variant_pool full                   # ← fit 의 --fallback_ladder 가 실제로 돌려면 필수
--aim_keyframes 6 --keyframe_aim auto --keyframe_ease smooth_kf
--fixed_focal --deroll --no_preview
--orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49
--tau_ref follow                      # 전 preset. auto 는 track_* 만 follow 기준이었다
--track_dynamic_only --track_min_drift_u 0.05
--trackings lock --preset_tracking    # translation 추종 gain 0.6(drift) → 1.0(lock)
--behind_min_zcam 0.02                # G1 이 카메라 뒤 픽셀을 벽으로 세던 것을 막는다
--tau_denom S --cloud_source memory
```

- `--tau_ref follow` — D125 parkour 실측: `auto`→`follow` 로 stat_2/3/6 의 G1 위반 프레임 비율이
  0.347/0.286/0.163 → **전부 0.000**, 대신 `path_len_u` 0.254→0.159, 0.334→0.143.
  정확성을 사고 다양성을 팔았다.
- **`--variant_pool full` 이 없으면 fit 의 `--fallback_ladder` 가 no-op 이다.** fallback 은 τ 뱅크의
  `plan_tier` 를 읽는데 `budget`(기본)이면 층이 전부 0 이라 층이 하나뿐이다
  (`fit_hole_ladder.py:834-837`; `:924` 가 `off — τ 뱅크에 plan_tier 가 없다` 로 찍어 준다).
  d179/d183 이 이 인자를 놀리고 있었다.

**회전 전용 preset 은 τ 로 크기를 못 정한다.** `ROTATION_ONLY_PRESETS`
(`fit/bank/sample_camera_bank.py:105`, **list** — `--external_shapes` 가 `.extend()` 한다) 5종
`pan_left / pan_right / pan_right_zoom_out / tilt_up / tilt_down` 은 이동이 정확히 0 이라
어떤 스케일에서도 같은 τ 를 낸다. 이 5종만 사다리를 **회전 각도**로 옮기고
(`--pan_deg_at_max × rung/max_rung`), divisor 를 `FIT_TAU_MAX_SCALE`(4.0) 로 바꾼다 (`:282`).

## A8. `fit` — `fit/bank/fit_hole_ladder.py`

각 변이에 대해 **"게이트를 안 깨는 최대 크기"** 를 이분법으로 푼다. 사다리 눈금은 기본이
`hole` 이고 (`--hole_ladder 0.10 0.20 0.35 0.50`), TRUMANS 는 `shot_scale` 을 쓴다 (Part B).

```
--tau_ref follow --tracking lock --preset_tracking
--behind_min_zcam 0.02 --behind_clear_src_ratio 0
--collision_time_match --collision_source depth
--min_subject_visible 0.6                      # D171-b 로 판정 승격
--fallback_ladder --fallback_target 5 --retry_status clamped_low
--follow_keyframes 6 --follow_kf_interp cubic
--area_timeline --composition --composition_min_area 0.004 --composition_max_nodes 3
```

- `--behind_clear_src_ratio 0` 은 **반드시 명시**한다. 채택 보류 중인 축이라 0 이 맞는데,
  argparse 기본값 0.3 에 기대면 `--behind_min_zcam` 과 같이 켰을 때 소스 재투영이 전부
  `z_floor` 아래로 떨어져 `source_g1_clear` 가 증거 0 으로 assert 한다 (6편 `FIT rc=1`).
- `--collision_source depth` — 실사 영상에는 mesh 가 없다. **depth shell 은 소스 카메라가 본
  표면만 안다**는 한계가 여기 그대로 남는다 (Part B 와 갈리는 지점).
- **보간 arm 을 jerk 로 고르지 말 것** — cubic 은 jerk 가 구간 상수라 구조적으로 이긴다.
  판정은 `|dv|` 로 한다.

## A9. `emit` — `fit/bank/emit_bank.py`

`rc==0` 일 때만 돈다. `bank.json` 의 solved 변이를 `canonical/canonical.json`
(`n_poses:21`, `rel c2w`, OpenCV, frame0 앵커, 단위 스케일) 로 굽는다.
49→21 은 보간 없이 index pick `np.rint(np.linspace(0,48,21))`.

> `poses.npz` 의 `target_hole` 은 **요청치가 아니라 달성치**다. 조인 키로 쓰면 조용히 0행이
> 나온다 — 키는 `variant_id`. 그리고 `variant_id` 는 **코퍼스 키가 아니다**: 재굽기해도 이름은
> 같고 knob/pose 만 바뀌므로, 세대가 다른 캐시를 재사용하면 miss 0 으로 통과하면서 전부 틀린다.

---

## A10~A11. 캡션 + export — `exec/run_corpus_export.py`

다섯 단계 `desc → merge_desc → captions → export → verify` (`run_corpus_export.py:226 STAGES`).

**1. `desc` — `fit/graph/describe_instances_vlm.py`** (선택)
SAM3 인스턴스 → VLM → **지칭 표현**. 뱅크에 실제로 등장한 `anchor_id` 만 물으므로 호출 수가
영상당 1~6회다. 동명 인스턴스는 형제 목록을 프롬프트에 같이 넣고, `## MEASURED FACTS`(마스크에서
잰 좌우 위치·면적비·깊이 순서)를 결정론적으로 실어 상대 비교 근거를 준다 — camel 두 마리가 둘 다
"light-colored" 로 돌아온 D120-b 의 처방이다. 산출 `out*/<video>/instance_desc.json`.
**없으면 `build_bank_captions.py --anchor_desc` 가 조용히 예전 라벨로 떨어진다.**

**2. `captions` — `fit/caption/build_bank_captions.py`**
변이별 `{target, event, framing, motion, composition}` → 자연어 프롬프트(`--prompt_style nl`).
`event` 는 `metadata.csv:prompt` 첫 문장. preset → 문구 사전은 `configs/caption_presets.json`
(`presets` 46종, `PRESETS` 와 양방향 1:1 — `presets.md` 참고).

- `--framing_min_in_frame 0.85` 가 **프레이밍 약속을 못 지키는 변이에서 framing/composition 절만
  뺀다**. motion 절과 target 은 그대로 둔다 — `track_*` 의 "tracks {target}" 은 follow_gain 1.0
  이라 실제로 참이고, 거짓인 건 프레이밍 약속뿐이다. d137 코퍼스 10,857행 실측:
  `subject_in_frame < 0.85` 가 4,570행(42.1%), 그중 3,225행(코퍼스의 29.7%)이 여전히 framing
  절을 달고 있었다.
- **강도 부사는 실현치가 아니라 요청치를 읽는다** — 설계다. 요청↔실현 대비가 남는 preset 은
  orbit sweep 하나뿐이다.
- TRUMANS 는 라벨이 `.blend` 오브젝트 이름이라 `--label_map` 이 필요하다 (Part B).

**3. `export` — `fit/convert/vista4d_bank_to_dl3dv.py`**
뱅크 + 캡션 + `recon_and_seg` → DL3DV 레이아웃 (`images_4/`, `transforms.json`, `prompts.json`,
`seg_list_*.txt`, `meta_*.csv`). `--drop_status clamped_low` 로 사다리 아랫단에서 정지해버린
변이를 뺀다. `--avg_scale_refs` 가 학습 시 분모가 될 `avg_scale` 을 정한다.

> **val 은 별도 파일이 아니라 test 목록의 앞부분**이다. 비율을 바꾸려면 `seg_list` 와
> `val_max_batches` 를 같이 고쳐야 한다. CLaTr 게이지는 train split 을 따라가므로,
> 재분할하면 옛 ckpt 가 새 test 를 이미 본 상태가 되어 지표가 조용히 부푼다.

**4. `verify`** — 씬당 카메라 수 분포와 뱅크 조합을 찍는다. 코퍼스 규모를 셀 때는
**씬당 상한으로** 센다 — 뱅크 재고를 세면 2.6배 부풀고, `plan_tier == 0` 은 세대마다 뜻이 달라
못 쓴다.

---

# Part B — 합성 씬 (TRUMANS)

## 왜 갈래가 따로 있나

Part A 의 depth 와 마스크는 전부 **추정치**다. DA3 는 정지 카메라에서도 focal 이 −14% 드리프트하고
SAM3 는 실루엣에서 몇 픽셀씩 샌다. 그래서 "카메라를 이만큼 움직이면 hole 이 얼마나 생기나",
"사람이 프레임 어디에 걸리나" 같은 판정이 항상 **기하 신호와 추정기 노이즈가 섞인 채** 나왔다.

TRUMANS 는 씬 전체가 `.blend` 로 있다 → **렌더러가 정답을 안다.** depth 는 Blender metre 실측,
instance mask 는 object pass index 라 픽셀 단위로 정확하다. 그리고 결정적으로 **소스 영상이 못 본
공간까지 안다** — 벽 뒤, 다른 방, 가구 내부.

**두 갈래가 만나는 지점**: `scene_graph.json` 은 두 갈래가 같은 스키마(`planner_scene_graph_v1`)다.
그래서 route 이후 뱅크·캡션·export 코드는 한 벌만 있고, 갈라지는 건 "어디서 기하를 얻었나" 뿐이다.

## B0. recording → 49프레임 창 — `fit/ingest/trumans_clip.py`

TRUMANS 는 30 fps 로 1,500~3,400 프레임짜리 연속 녹화다. 어디를 자르느냐가 그대로 씬이 되므로
손으로 고르면 재현이 안 된다. 창 점수 = ① 카메라 회전 폭(deg) ② 사람 이동 거리(m) ③ action label
종류 수. 셋 다 큰 창이 "카메라도 돌고 사람도 움직이는" 구간이다 — 정지 구간은 시차가 0 이라
OBB 깊이축이 무너진다.

배포본에는 카메라가 없어서 `smplx_result`(world) / `smplx_result_in_cam`(camera) 쌍에서 복원하는데,
그 카메라는 **사람을 반경 ~2.1 m 로 따라다니는 가상 카메라**라 씬 고정 카메라가 아니다.
그래서 여기선 pose 를 안 쓰고 "얼마나 돌았나"라는 **창 선택 지표로만** 쓴다.

> 리그는 SMPL-X 가 아니라 **CC_Base** 다. 루트 본은 전 프레임 고정이고 Hip 로컬 +z 가 정면이다.
> 보행 구간은 Actions 라벨에 **0건**인데 실제로는 프레임의 19% 다 — 라벨로 보행을 찾지 말 것.

## B1. 소스 카메라 합성 + GT 렌더 — `fit/ingest/trumans_to_recon.py`

**소스 카메라를 합성하는 건 선택이 아니라 데이터의 제약이다.**
- `<seq>_camera_pose.pkl` 은 recording 67편 중 **2편**에만 있다 (00add26c, 0aa05d5a).
- `.blend` 안의 CAMERA 오브젝트 4개는 **전부 정지** (`anim=False`, constraint/parent 없음).

정지 카메라를 소스로 쓰면 시차가 0 이라 `τ` 축과 view-angle 축이 통째로 무의미해진다.
그래서 있는 2편의 통계에 맞춰 합성하는데, **그 통계는 사람이 걷느냐로 갈린다** (49프레임 창 실측):

| 창 종류 | 00add26c (창 206/15) | 0aa05d5a (창 143/11) |
|---|---|---|
| still (보행 ≤10%) | cam net 0.440 m, dt 0.0122 | cam net 0.548 m, dt 0.0137 |
| walk (보행 ≥80%) | cam net 1.314 m, dt 0.0309 (사람 net 1.250 → 추종 **1.05x**) | cam net 0.588 m, dt 0.0141 (사람 net 1.046 → **0.56x**) |

즉 실제 카메라 2대는 보행에 정반대로 반응한다. `--track_gain 0.6` 이 그 둘 사이에 앉는다.
정리하면 소스 카메라는 "높이 고정 + 사람 추종 + 느린 호(arc)".

시작 pose 는 `fit/ingest/trumans_scene_probe.py` 가 사람 가슴을 중심으로 (방위각 × 고도 × 거리)
격자를 깔고 **실제 광선**으로 ① 시선이 뚫리는지 ② 벽 속/벽에 붙었는지 ③ 공중에 떠 있는지를 재서
통과분에서만 샘플링한다 — Part A 의 depth shell 근사(G1)를 여기선 안 쓴다. full-house 씬이라
카메라를 아무데나 두면 **벽 안쪽**에 박힌다.

**단계** (Blender 3회 + 변환 1회):
```
1. probe(격자)   ~4 s    사람 궤적 + 설 수 있는 자리
2. 궤적 합성      즉시    시작 pose 샘플 + preset 호 + 사람 추종 look-at
3. probe(검증)   ~4 s    합성한 49 pose 를 프레임별로 다시 광선 검사
4. gt_render     ~150 s  RGB + depth/index
5. 변환          ~10 s   video.mp4 / depths(EXR f16) / masks / cameras.npz / seg_instances
```

### `viz/trumans_gt_render.py` — 재논의 금지 실측

- `BLENDER_EEVEE_NEXT` 에는 object index pass 가 **아예 없다**. index 는 Cycles 전용.
- EEVEE 의 Z pass 는 픽셀 footprint 위에서 필터링되어 실루엣에서 틀린다 (1.6% 픽셀이 1 cm 이상,
  최대 3.85 m). → **depth+index 는 항상 Cycles 1 spp CPU** (~1.8 s; Cycles GPU 는 오히려
  느리고(2.55 s) OptiX 커널 빌드에 일회성 437 s 를 더 쓴다 → `--cdevice CPU` 기본).
- **RGB 엔진은 EEVEE 가 최적이 아니다.** blend 는 Blender **3.3.6 저작**인데 우리는 4.5.9 로
  돌린다. 4.2 에서 EEVEE 가 EEVEE_NEXT 로 전면 재작성됐고, 3.3 기준 발광 재질(노트북 20 / TV 20 /
  조명 175~469)이 흰 덩어리로 타서 **옆 물체까지 번진다**. 같은 프레임을 Cycles 로 렌더하면 번짐이
  0 (frame 402 crop 안 `max>=200` 픽셀: EEVEE 401 → Cycles **0**). `--rgb_engine cycles` 가 탈출구
  (기본은 하위호환 때문에 `eevee`). 투과 판정은 눈대중이 아니라 **index pass** 로 한다.
- depth 는 **z-planar**, metre, 배경 sentinel `1e10`, 픽셀 중심 `(col+0.5, row+0.5)`.
- 사람은 ARMATURE **`zzy3`** 이고 그 자체는 아무것도 렌더하지 않는다. 실제로 그려지는 건 그
  armature 가 deform 하는 mesh 12개(`CC_Base_Body`, `Layered_sweater`, `Slim_Jeans`, …)라
  그 **union 을 index 1 로 예약**한다. 이름 하드코딩이 아니라 armature modifier / parent 로 찾는다.
- `--cycles` 로 시작하는 CLI 플래그를 **절대 만들지 말 것** — Cycles 애드온이 `--` 를 무시하고
  argv 전체를 prefix-match 로 훑어 실행이 통째로 죽는다 (그래서 `--cdevice`).
- TRUMANS `.blend` 는 `scene.frame_step = 2` 로 저장되어 있다 → 1 로 되돌린다.

## B2. mesh 충돌 격자 — `fit/ingest/build_trumans_mesh_grid.py`

`.blend` 삼각형을 5 cm voxel 로 굽고, 소스 카메라에서 자유공간을 flood-fill 한다 →
`mesh_grid.npz`. Blender 를 **한 번만** 띄우고 16 s, 502 오브젝트, GPU 불필요.

이게 있어야 "소스 영상이 **한 번도 안 보여준** 벽·방·가구 내부"를 알 수 있다.
a00 chunk 836 변이, 임계 `0.02·S` = 3.9 cm 실측:

| 충돌 소스 | 위반 변이 |
|---|---|
| depth (소스가 본 표면) | 130 |
| **mesh (씬 전체)** | **357** |
| overlap (둘 다) | **0** |

**`--collision_source both` 는 배타 선택이 아니라 합집합이다.** 겹침이 0이므로 하나만 쓰면 다른
하나를 통째로 잃는다. `resolve_mesh_grid` 는 격자가 없으면 **죽는다** — 조용히 depth 로 안 떨어진다.

## B3. graph / cloud — GT 소스로

Part A 와 **같은 스크립트**(`fit/graph/build_scene_graph.py`)를 쓰되 세 축을 GT 로 바꾼다:

```
--gravity_source gt --ground_source gt --subject_source gt
--scene_scale_mode points_first_cam --scene_scale_stride 1
```

`.blend` 가 중력·지면·사람을 다 알고 있으므로 GeoCalib(A3)도 ground RANSAC 도 안 돈다.
같은 인자로 `python -m lbm.cloud` 도 돈다. TRUMANS 는 `--cloud_source npz` 를 쓴다 —
Blender 드라이버들이 같은 점군을 나눠 읽는다.

> **unproject 앞에는 `preprocess_scene` 이 필요하다.** 빼면 static 경계 누수가 49프레임 쌓여
> 동적 물체가 반투명 빗살이 된다. `S`/`z_med` 는 전처리 **전** raw depth 로 잰다.

## B4~B6. 뱅크 — Part A 와 같은 체인, 세 군데가 다르다

`exec/run_bank.py` + `configs/bank/d266_trumans_shotscale.json`. route/tau/emit 은 Part A 와
글자 단위로 같고 (`"extends"` 가 최상위 키 단위 얕은 병합이라 바꾸는 블록만 다시 적는다),
**fit 블록만 다르다**.

### ① 충돌·가림 판정을 뱅크에서 뺐다 (D266)

d207 뱅크 829행 실측에서 binding 이 `collision 371 (44.8%) / hole 216 / approach 51 /
occlusion 40 / obb 37` — **depth shell 충돌이 TRUMANS 카메라 크기의 절반을 혼자 정하고 있었다.**
그런데 TRUMANS 는 mesh raycast 가 가능한 유일한 코퍼스다. 그래서 판정을
`fit/convert/bank_to_blender_poses.py --raycast` 로 옮긴다:

```
--max_behind_frac -1        # G1 depth shell 충돌 off
--no_obb_gate --no_ground_gate --no_approach_gate
--min_subject_visible 0     # G3 가림 off
```

raycast 의 `min_clearance` / `min_floor_drop` / `min_subject_dist` / `clear_frac` / `min_los_frac`
가 같은 것을 mesh 로 본다. **`elev` 는 남긴다** — 이건 충돌이 아니라 "피사체 바로 위/아래에서
찍지 마라"는 구도 규칙이라 raycast 가 안 본다. 열은 전부 그대로 측정된다 (게이트만 끈 것이다).

> mesh 충돌 판정은 `bank_to_blender_poses --raycast` 에만 있다. **뱅크끼리 비교하면 0건이 나온다.**

### ② 사다리 눈금이 hole 이 아니라 shot scale 이다

hole 을 빼면 크기를 정하는 축이 비므로 **subject 화면면적 배율의 로그 절대값**으로 갈아끼운다:

```
--ladder_metric shot_scale --hole_ladder 0.2 0.4 0.7      # ≈ ×1.22 / ×1.49 / ×2.0
```

로그 비율인 이유: ① push_in(면적↑)·pull_out(면적↓) 양쪽에서 손잡이에 대해 0 부터 단조증가라
이분법이 성립한다 (생면적 목표는 방향마다 부등호가 뒤집힌다), ② anchor 크기에 불변이라
사람 하나와 방 하나가 같은 Δ 를 같은 뜻으로 쓴다.
**렌더가 안 는다** — 면적은 이분법용 싼 경로(`render_metrics`)가 이미 돌려준다.

`binding` 토큰은 `hole` 로 **유지한다** — `emit_bank` / `route_presets` / `probe_tau_divisor` 가
그 문자열을 읽는다. 대신 `ladder_metric` / `area_static` / `shot_dev` 열이 붙고 `variant_id` 가
`__shot0.4` 접미사를 쓴다.

### ③ 프레이밍 하한과 손잡이 상한

- `--min_subject_in_frame 0.6` — 충돌·가림을 다 빼면 손잡이를 위에서 막는 게 예산 하나뿐이라
  subject 가 화면 밖으로 나가는 크기가 통과한다 (d207 실측 `subject_area_end` p10 = **0.0000**).
  이건 가림이 아니라 **예산 앞**에 놓인 프레이밍 하한이다.
- `--tau_knob_max 1.2` — shot scale 이 안 변하는 preset(orbit/truck/pan)은 예산에 영원히 안 닿아
  `unreached` 로 기본 상한 3.0 까지 벌어진다. 그 크기는 raycast 가 어차피 전량 기각한다.
- `--collision_source both --mesh_margin_autoclamp 0.9` — 위 ②의 mesh ∪ depth.

anchor 는 사람 하나다 (`--max_dynamic_anchors 1 --max_static_anchors 0`).
정적 물체 target 은 보류 — 쓰려면 SAM3 static instances 263편이 선행한다.

## B7. 캡션 + export

- **`metadata.csv` 가 없다** → `fit/ingest/build_trumans_metadata.py` 가 `manifest_a<NN>.json` 의
  `caption.{target,event}` 로 **Vista 와 같은 스키마**의 csv 를 만든다 (캡션 스크립트 수정 0).
  TRUMANS action label 은 명령형(`put down the book with both hands`)이라 Vista 의 서술문에 눈금을
  맞추려고 3인칭 진행형으로 바꾼다. 어휘가 **26개 동사머리로 닫혀 있어** 휴리스틱 대신 명시적
  gerund 표를 쓴다 — `open` 에 자음중복 규칙을 돌리면 `openning` 이 나오는 종류의 조용한 오류를
  애초에 안 만들기 위함.
- **라벨이 사람 이름이 아니다** → `build_bank_captions.py --label_map configs/trumans_labels.json`.
  Vista/dynpose 는 라벨이 SAM3 텍스트 프롬프트(=사람이 쓴 명사)라 손댈 게 없지만, TRUMANS 는
  라벨이 `.blend` 오브젝트 이름이다 (`Layered_sweater.001`, `zzy3`, 숫자 에셋).
  `normalize_label`(`:243`) 이 `strip_suffix` 로 `.001` 을 떼고 stems 사전으로 옮기며,
  숫자 에셋은 `"furniture"`, 미상은 `"object"` 로 떨어진다.
  `--label_map` 을 안 주면 `label_map is None` 이라 입력을 그대로 돌려준다 = **Part A 는 무변경**.
- **export 는 별도 스크립트** `fit/ingest/trumans_lite_to_dl3dv.py` — `--layout per_recording`(기본,
  `per_clip` 선택), `--chunk_prefix trumans`, `--test_recordings <id>`,
  `--avg_scale_refs context_first_cam`, `--clip_avg_scale_ref centroid`. 한 recording 이 여러
  chunk 를 낳으므로 train/test 를 **chunk 가 아니라 recording 단위**로 갈라야 누수가 없다.

---

# 부록

## 실행 순서 (dynpose 10,346편 기준)

전부 `camera_generation/dataset/` 를 cwd 로 두고 돈다. 인자·샤딩은 `USAGE.md` 참고.

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python

# A0~A2  ingest (link→recon→nouns→metadata→sam3→dynmask)
#        VLM 명사 단계 직전에 exec/serve_qwen3vl.sh 로 vLLM 을 올리고 끝나면 내린다
$PY fit/ingest/dynpose_ingest.py --stage launch --launch_stage recon --gpus 0,1,2,3

# A3     GeoCalib 사이드카 (env geocalib)
$PY fit/ingest/geocalib_gravity.py ...

# A4~A9  뱅크 체인 — graph 가 도는 동안 끝난 편부터 주워 굽는다
$PY exec/chain_bank_rounds.py --config configs/bank/d185_dynpose100k_grid5.json \
    --videos <scenes.txt> --work <tmp/dNNN> --bank_dir hole_bank_d185 --bank_shards 4

# A10~A11  캡션 → 코퍼스
$PY exec/run_corpus_export.py --banks hole_bank_d185 --stage desc,merge_desc,captions,export,verify
```

## 실행 순서 (TRUMANS)

```bash
$PY fit/ingest/trumans_clip.py ...            # recording → 49프레임 창
$PY scripts/trumans_lite_bank.py ...          # → trumans_to_recon (Blender GT 렌더)
$PY fit/ingest/build_trumans_mesh_grid.py ...
$PY exec/run_bank.py --config configs/bank/d266_trumans_shotscale.json --videos <chunks.txt>
$PY fit/convert/bank_to_blender_poses.py --raycast ...
$PY fit/ingest/build_trumans_metadata.py
$PY fit/caption/build_bank_captions.py --label_map configs/trumans_labels.json ...
$PY fit/ingest/trumans_lite_to_dl3dv.py --layout per_recording ...
```

## env

| env | 쓰는 곳 |
|---|---|
| `vista4d` | SAM3, VLM 명사/지칭, 뱅크(route/tau/fit/emit), 캡션, export |
| `da3` | `fit/graph/build_scene_graph.py`, `decode/` verify |
| `geocalib` | `fit/ingest/geocalib_gravity.py` (kornia 가 다른 env 에 없다) |
| `vllm` | Qwen3-VL-30B @ 22002 — **쓸 때만 올리고 끝나면 내린다** |
| `GenDoP` | 베이스라인 (`exec/run_gendop_eval.py`) |
| Blender | `/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender` (씬은 3.3.6 저작) |

## 규칙

- **GPU 는 0~3 만** 쓴다. `--gpu` / `--cuda` / `CUDA_VISIBLE_DEVICES` 를 주는 자리마다.
- 임시 파일은 `/tmp` 가 아니라 `/data1/cympyc1785/LatentCamVid/tmp/<작업>/` 아래.
- 세대 이름은 **게이트 규약이 바뀌면 바꾼다** (D105). 같은 이름으로 재굽기하면 캐시가
  miss 0 으로 통과하면서 캡션↔변이가 어긋난다.
