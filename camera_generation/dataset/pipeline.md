# pipeline.md — 영상 1편 → 카메라 변이 코퍼스

이 문서는 **데이터셋이 만들어지는 순서**만 다룬다. 각 단계가 *왜* 그 모양인지의 근거와 기각된
대안은 `DECISIONS.md`, preset 어휘는 `presets.md`, 실행 방법은 `README.md` 에 있다.

파이프라인은 두 갈래다.

| | Part A — **실사 영상** | Part B — **합성 씬 (TRUMANS)** |
|---|---|---|
| 코퍼스 | Vista4D 52편 / DynPose-LBM 880편 | TRUMANS-Lite (recording 67편 → chunk) |
| 기하 출처 | DA3 depth + 카메라 **추정** | `.blend` **정답** (Blender 렌더) |
| 인스턴스 | SAM3 text PCS (추정) | object pass index (정답) |
| 중력·지면 | GeoCalib / ground RANSAC | `.blend` GT |
| 충돌 판정 | depth shell (소스가 본 표면만) | depth ∪ **mesh voxel grid** |
| 드라이버 | `run_k6_d128_shard.sh` / `run_dynpose_d129_shard.sh` | `run_trumans_d132_shard.sh` |

**두 갈래가 만나는 지점**: S5 이후 산출물 `scene_graph.json` 은 두 갈래가 **같은 스키마**
(`planner_scene_graph_v1`)다. 그래서 S6 이후 뱅크·캡션·export 코드는 한 벌만 있고, 갈라지는 건
"어디서 기하를 얻었나" 뿐이다.

---

# Part A — 실사 영상 (Vista4D / DynPose-LBM)

## S0. 입력 규약

모든 하류가 읽는 단 하나의 포맷은 Vista4D 배포본 레이아웃이다 (`scene_graph/io.py:load_scene`):

```
<eval_data>/eval_data/
  recon_and_seg/<video>/     video.mp4 · depths/ · masks/ · cameras.npz · predictions.npz
  seg_instances/<video>/     meta.json · masks.npz      # 동적 인스턴스 (SAM3)
  seg_instances_static/<video>/                          # 정적 인스턴스 (SAM3, 선택)
metadata.csv                 name,video,camera,seed,prompt,dynamic,do_sky_seg,source,video_id
```

- `cameras.npz` 의 `cam_c2w` 는 **frame0 앵커** (`cam_c2w[0] == I`), OpenCV 규약.
- world 단위는 무차원 `u`. `S` = frame0 non-sky 픽셀의 ray-depth 평균 (`scene_graph/scale.py`).
  `1 u ≜ S DA3 units`. 리포의 `avg_scale` / `S_da3` / `norm_scale` 과 같은 게이지다.
- 49프레임 고정.

Vista4D 는 배포본이 이미 이 모양이고, DynPose-LBM 은 `recon_and_seg_single.py --recon_method da3`
가 원본 영상에서 만든다.

## S1. VLM 명사 — `extract_nouns_vlm.py`

영상에서 **detector 에 먹일 명사**를 뽑는다. `--frame_mode multi` 는 frame 0/16/32/48 네 장을
Qwen3-VL-30B (로컬 vLLM `http://127.0.0.1:22002/v1`) 에 보내고 `{dynamic:[], static:[]}` 를 받는다.
`single`(frame0 한 장)은 detector 공정비교용 조건이지 코퍼스용이 아니다 — 움직임이 보여야
dynamic/static 을 가를 수 있다.

명사는 `lowercase, singular, 1-2 words` 로 강제한다. SAM3/GDINO 는 caption 의 부분문자열을 라벨로
돌려주므로 형용사구가 길면 phrase 가 엉킨다. **긴 지칭 표현은 여기가 아니라 S9 에서 만든다.**

산출: `<output>/vlm_nouns.json` (`records[] = {video, frame_mode, dynamic, static}`).

> `extract_nouns_vlm.py` 는 **끝날 때 한 번만** 파일을 쓴다 (:215-229). 600편을 한 프로세스로
> 돌리면 590편째 예외 하나에 1~2시간이 날아가므로 `run_dynpose_d145_nouns.sh` 가 `--merge` 로
> 25편씩 끊어 부른다. 스크립트는 한 줄도 안 고친다.

## S2. 인스턴스 분할 (SAM3 text PCS)

**동적** — `video_generation/scripts/sam3_seg_instances.py`. keyword 출처는 `metadata.csv:dynamic`
(저자 정답). `--num_shards/--shard_id/--skip_done` 을 갖고 있어 목록을 안 넘겨도 done 을 건너뛴다.

**정적** — 2단계다. 정적 명사는 아무도 정답을 안 적어놨기 때문이다.

1. `extract_static_nouns.py` — `vlm_nouns.json:static` → keyword. 두 가지를 거른다:
   - **동적과 겹치는 명사**를 버린다. 같은 물체가 `dyn_*` 와 `stat_*` 두 노드로 생기면
     3D IoU 0.5 를 못 넘겨 안 합쳐지고, VLM 프롬프트에 같은 물체가 두 번 나온다.
   - **광역 표면**(`ground`/`wall`/`floor`)을 버린다 (`--drop_surfaces` 기본 True). OBB 를 씌우면
     씬 전체를 덮는 상자가 되고 `near` 엣지가 모든 노드에 걸린다. 이 둘은 이미 기하로 처리된다
     (`relations.ground_height` / `relations.find_wall_planes`).
   - `--max_nouns 8`. 산출 `static_nouns.json` (`static_nouns_v1`) 에 버린 명사와 사유도 남는다.
2. `sam3_static_instances.py` — 위 keyword 로 SAM3, 출력은 `seg_instances_static/`
   (배포본 `seg_instances/` 를 **안 덮는다**). 저장 전에 배포본 `dynamic_mask` 와의 겹침이
   `--max_dynamic_frac 0.5` 를 넘는 track 을 뺀다 — VLM 이 static 이라 부른 게 실제로 움직이는
   경우(avocado-slice `wheelchair`)가 있고, 그게 `stat_*` 가 되면 `supported_by` 가 "움직이는
   것에 얹혀 있다"는 틀린 관계를 만들어 그대로 캡션에 나간다. 뺀 track 은 `static_report.json`.

`static_mask` 가 이미 있는데 왜 SAM3 를 또 도나: `static_mask` 는 인스턴스 구분이 없는 한 장짜리
이진 마스크라 "the fence" 를 지목할 수 없다. 정적 노드의 용도는 자유공간이 아니라 **VLM/캡션이
물체를 이름으로 부르는 것**이다.

## S3. 중력 사이드카 — `geocalib_gravity.py`

`--num_frames 5` 프레임에서 GeoCalib 로 카메라 up 을 재고 `up_world = R_c2w[f] @ vec3d[f]` 로
world 로 올린 뒤 `--trim_keep_frac 0.75` 절사평균. 산출 `out*/<video>/geocalib_gravity.json`
(`up_world`, `confidence`, `spread_deg`). env 는 **`geocalib`** 전용 (kornia 가 vista4d/da3 에 없다).

왜 별도 단계인가: ground RANSAC 은 "지면 평면이 보여야" 되는데 vista 53편 중 15편(28%)이
`camera_up_fallback` 으로 떨어졌고, 통과한 38편 중에도 벽/책상 상판을 지면으로 문 게 여럿이다.
중력축이 틀어지면 OBB yaw/extent/center 와 `roll=0` 기준이 **전부 같이** 틀어진다.

## S4. scene graph — `build_scene_graph.py` (env `da3`)

영상 1편 → `out*/<video>/scene_graph.json`. 내부 순서:

| | 무엇 | 핵심 인자 |
|---|---|---|
| scale | `S`, `S_allpix`, `parallax_ratio`, `z_med` | `--scene_scale_mode points_first_cam --scene_scale_stride 1` |
| gravity | GeoCalib 사이드카 → `up_world` → `T_gw` | `--gravity_source geocalib` (기본, strict — 사이드카 없으면 assert) |
| lift | depth+mask → world 점군 | `valid = ~sky & isfinite & z>0 & conf>0.5 & ~dilate(&#124;∇log z&#124;>0.05, 2px)` |
| instances | track → 프레임별 점군, sliver 기각, 키워드 중복 병합 | `--min_area_frac 0.002 --min_frames 5 --min_score 0.3 --merge_iou 0.5`; 상한 `--max_dyn_nodes 6 --max_stat_nodes 10` |
| OBB | 중력정렬 `cv2.minAreaRect`, 저시차 깊이축 보정 | 동적은 프레임별 fit → yaw unwrap → 5프레임 median |
| relations | `supported_by` / `against_wall` / `near` | `--contact_u 0.05 --wall_u 0.08 --near_factor 1.5` |

`--gravity_source` 는 `geocalib`(strict) / `auto`(사이드카 없으면 조용히 ground RANSAC) / `gt`.
**dynpose 는 D148 이전까지 `auto` 였다** — 280편 전부 사이드카가 있어서 결과는 같았지만, 신규
600편에는 사이드카가 0개라 `auto` 로 구우면 코퍼스 절반만 다른 중력 게이지를 쓰게 된다.

`--seg_static_root` 는 기본 `None` = `<eval_data>/eval_data/seg_instances_static` 을 **있으면 자동
사용**한다. `io.py:101` 이 두 루트를 각각 `kind="dyn"` / `kind="stat"` 로 읽어 합친다.

산출 노드가 하류에서 실제로 소비되는 항목: `obb{center,extent,yaw,R}` · `track{center_smooth,
yaw,conf}` · `obs_az_span_deg` · `viewing_distance.d_ref` · `moving` · `label`/`aliases`/`near` ·
`gravity.up_world`.

검증용 `vis/obb_overlay.mp4` 를 같이 낸다 — **박스가 물체를 안 감싸면 하류가 전부 무효**라
사람 승인은 여기서 한 번 받는다 (`--no_overlay --no_topdown` 으로 끌 수 있다).

## S5. 4D 점군 — `python -m lbm.cloud`

`Vista4D/utils/point_cloud/point_cloud.py:unproject` 로 RGBD 49프레임을 world 점군으로 올리고
`<video>/cloud.npz` 에 캐시한다 (영상당 1회). `visible (N,F)` 를 같이 받아 `visible.sum(1)==1` 이
동적 점이다. 이 점군이 뒤의 모든 게이트(hole/충돌/구도)를 재는 **렌더러의 입력**이다.

S4·S5 는 뱅크와 독립이라 마커 파일(`.graph_d122` / `.cloud_d122`)로 재실행을 막는다. 뱅크 축만
바뀐 재굽기(D128/D129)는 이 두 단계를 안 돈다 — 같은 결과에 편당 ~90 s 를 쓰기 때문이다.

## S6. anchor 선별 + preset 라우팅 — `route_presets.py`

**여기가 두 코퍼스가 갈리는 유일한 알고리즘 지점이다.**

- **vista (52편)**: 라우팅을 안 쓴다. anchor 전량 × preset 전부를 `sample_camera_bank.py` 가
  열거한다 (편당 343.0 변이). anchor 상한 3/3 과 surface drop 은 뱅크 인자로 직접 준다.
- **dynpose (880편)**: `route_presets.py` 로 **anchor 1개당 preset 8슬롯**만 고른다
  (편당 173.9 변이). 씬이 17배 많아 변이가 곧 시간이기 때문이다 —
  실측 비용은 `9.7분 + 8.80초 × 변이수` (R=0.887, 84편 회귀).

8슬롯: `recede / advance / lateral / rotate / arc / orbit / vertical / static`.
슬롯당 preset 을 고르는 신호는 전부 **fit 보다 앞서 나오는 것**들이다:

| 신호 | 쓰임 |
|---|---|
| `node.moving` | `track_*` 로 갈지. 안 움직이는 anchor 에 `track_` 을 붙이면 궤적이 비-track 짝과 비트 단위로 같아진다 |
| `obs_az_span_deg` | orbit 가능 여부. `< 120°` 면 s_curve 로 대체 (실측 중앙값 72°, 120° 이상은 25%뿐) |
| `gravity.method` | `camera_up_fallback` 이면 "위"가 world up 이 아니다 → 세로 슬롯(pedestal/crane) 제외 |
| `cam_c2w_world` | 소스가 트럭한 **반대쪽**을 고른다 — 소스가 안 본 면이 hole 이 크고 그게 이 데이터의 값어치다 |

왜 8개로 줄여도 되나: `fit_hole_ladder` 가 요청한 hole 을 실제로 맞추는 비율(`binding=="hole"`)이
preset 마다 40배 차이난다 (dynpose 232편 실측) —
`pan 77% / dolly_out 54% / pull_out_arc 48~50% / truck 40~43% / pedestal·crane 12~21% /
orbit 15~17% / push_in_arc 2.5~3.1% / dolly_in 0.3~1.3%`. 전진 계열은 anchor OBB 에 막혀 요청
크기가 안 나온다. 그렇다고 빼면 캡션 어휘에서 "move forward" 가 사라지므로 **슬롯 1개만 남긴다**.

**DataDoP 외부 궤적 (`dd_*`)** — `--external_shapes configs/datadop_shapes.json --num_external N`.
실제 촬영본에서 검색한 궤적 모양을 preset 이 약한 슬롯부터 채운다
(`EXTERNAL_SLOT_ORDER = advance, vertical, arc, rotate, lateral, recede`).
`--num_external 0` 이면 `dd_*` 가 통째로 빠진다.

anchor 상한은 **여기서** 걸린다 (`route_presets.py:82` → `schema.pick_main_anchors`,
`--max_dynamic_anchors 3 --max_static_anchors 3`). 뱅크에 주면 조용히 죽는다 — route 가
`--nodes` 를 명시로 넘기고 `sample_camera_bank.py:144-151` 이 그걸 받으면 `pick_main_anchors` 를
**호출하기 전에 return** 한다. surface drop 은 `schema.py:164` 의 `drop_surfaces=True` 기본값이
이미 건다 (플래그 없음).

## S7. τ 뱅크 — `sample_camera_bank.py`

anchor × preset × τ 사다리로 후보 궤적을 만들고 게이트를 돌린다. 통과분만 `bank.csv` 에
`status == "solved"` 로 남는다. 다른 status: `collision_limited / obb_limited / elev_limited /
ground_limited / approach_limited`. **하류(reel·emit·export)는 전부 `solved` 만 본다.**

D128/D129 기준 축 (D105 원칙대로 **전부 명시**한다 — 뱅크 정체성을 argparse 기본값에 맡기면
기본값이 뒤집혔을 때 한 폴더에 두 규약이 섞인다):

```
--aim_keyframes 6 --keyframe_aim auto --keyframe_ease smooth_kf
--fixed_focal --deroll --no_preview
--orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49
--tau_ref follow                      # 전 preset. auto 는 track_* 만 follow 기준이었다
--track_dynamic_only --track_min_drift_u 0.05
--trackings lock --preset_tracking    # translation 추종 gain 0.6(drift) → 1.0(lock)
--behind_min_zcam 0.02                # G1 이 카메라 뒤 픽셀을 벽으로 세던 것을 막는다
```

- `--tau_ref follow` — D125 parkour 실측: `auto`→`follow` 로 stat_2/3/6 의 G1 위반 프레임 비율이
  0.347/0.286/0.163 → **전부 0.000**, 대신 `path_len_u` 0.254→0.159, 0.334→0.143.
  정확성을 사고 다양성을 팔았다.
- `--track_min_drift_u 0.05` — `moving`(=`path_len_u > 0.05`)은 제자리 흔들림(춤·손짓·그네)을
  통과시킨다. 순변위 0.05 u 이하인 49 노드가 d121 에서 `track_*` 2040 쌍을 만들었고, 그 쌍의
  track vs 비-track 차이는 `|Δτ|` median 0.0210 — 진짜 이동 anchor(0.2071)의 1/10 이라
  "follows the subject" 캡션을 떠받치기엔 얇다. **`moving` 자체는 안 건드린다** — `moving` 은
  `schema.py:207` 의 anchor split key 라, 강등하면 그 dyn 노드가 static 버킷으로 넘어간다.

dynpose 는 여기에 `--external_shapes ... $ARGS --tau_ladder 1.00 --skip_on_empty` 가 붙는다.
변이가 0이면 `skipped.json` 을 남기고 넘어간다.

## S8. hole 사다리 적합 — `fit_hole_ladder.py`

각 변이에 대해 "요청한 hole 비율"이 나오도록 궤적 크기를 이분법으로 푼다.
`--hole_ladder 0.10 0.20 0.35 0.50` (dynpose 4단; vista 는 기본값).

```
--tau_ref follow --tracking lock --preset_tracking
--behind_min_zcam 0.02 --behind_clear_src_ratio 0
--collision_time_match --collision_source depth
--area_timeline --composition --composition_min_area 0.004 --composition_max_nodes 3
```

`--behind_clear_src_ratio 0` 은 **반드시 명시**한다. 채택 보류 중인 축이라 0 이 맞는데, D128 첫
실행 때 argparse 기본값 0.3 에 기댔다가 6편(avocado-slice/basketball-four/bed-shopping/
bmx-bumps/breakdance/camel)이 `FIT rc=1` 로 날아갔다 — `--behind_min_zcam` 과 같이 켜면 소스
재투영이 전부 `z_floor` 아래로 떨어져 `source_g1_clear` 가 증거 0 으로 assert 한다
(`video_generation/FIX.log` 2026-09-05).

`--collision_source depth` — 실사 영상에는 mesh 가 없다. **depth shell 은 소스 카메라가 본 표면만
안다**는 한계가 여기 그대로 남는다 (Part B 와 갈리는 지점).

## S9. emit — `emit_bank.py`

`rc==0` 일 때만 돈다. `bank.json` 의 solved 변이를 `canonical/canonical.json`
(`n_poses:21`, `rel c2w`, OpenCV, frame0 앵커, 단위 스케일) 로 굽는다.
49→21 은 보간 없이 index pick `np.rint(np.linspace(0,48,21))`.

## S10. 캡션

1. **`describe_instances_vlm.py`** (선택) — SAM3 인스턴스 → VLM → **지칭 표현**.
   뱅크에 실제로 등장한 `anchor_id` 만 물으므로 호출 수가 영상당 1~6회다. 동명 인스턴스는
   형제 목록을 프롬프트에 같이 넣고, `## MEASURED FACTS`(마스크에서 잰 좌우 위치·면적비·깊이
   순서)를 결정론적으로 실어 상대 비교 근거를 준다 — camel 두 마리가 둘 다 "light-colored" 로
   돌아온 D120-b 의 처방이다. 산출 `out*/<video>/instance_desc.json`.
   **없으면 `build_bank_captions.py --anchor_desc` 가 조용히 예전 라벨로 떨어진다.**
2. **`build_bank_captions.py`** — 변이별 `{target, event, framing, motion, composition}` →
   자연어 프롬프트(`--prompt_style nl`). `event` 는 `metadata.csv:prompt` 첫 문장.
   `--framing_min_in_frame 0.85` 가 **프레이밍 약속을 못 지키는 변이에서 framing/composition 절만
   뺀다** (motion 절과 target 은 그대로 — `track_*` 의 "tracks {target}" 은 follow_gain 1.0 이라
   실제로 참이고, 거짓인 건 프레이밍 약속뿐이다). d137 dd10 코퍼스 10,857행 실측: `subject_in_frame
   < 0.85` 가 4,570행(42.1%), 그중 3,225행(코퍼스의 29.7%)이 여전히 framing 절을 달고 있었다.

## S11. export + 서브샘플

`vista4d_bank_to_dl3dv.py` — 뱅크 + 캡션 + `recon_and_seg` → DL3DV 레이아웃
(`images_4/`, `transforms.json`, `prompts.json`, `seg_list_*.txt`, `meta_*.csv`).
`--drop_status clamped_low` 로 사다리 아랫단에서 정지해버린 변이를 뺀다.
`--avg_scale_refs` 가 학습 시 분모가 될 `avg_scale` 을 정한다.

`latentcam/scripts/data/filter_seg_list_by_preset.py --target_frac_prefix dd_ --target_frac 0.10
--suffix dd10` — `dd_*` 는 전체 20,658행 중 12,528행(60.6%)이라 그대로 두면 코퍼스가 DataDoP
모양에 지배된다. 10%만 남긴 `seg_list_*_dd10_{train,test}.txt` 가 실제 학습 리스트다.
(`--num_external 0` 으로 애초에 `dd_*` 를 안 만들면 이 단계가 필요 없다.)

---

# Part B — 합성 씬 (TRUMANS)

## 왜 갈래가 따로 있나

Part A 의 depth 와 마스크는 전부 **추정치**다. DA3 는 정지 카메라에서도 focal 이 −14% 드리프트하고
SAM3 는 실루엣에서 몇 픽셀씩 샌다. 그래서 "카메라를 이만큼 움직이면 hole 이 얼마나 생기나",
"사람이 프레임 어디에 걸리나" 같은 판정이 항상 **기하 신호와 추정기 노이즈가 섞인 채** 나왔다.

TRUMANS 는 씬 전체가 `.blend` 로 있다 → **렌더러가 정답을 안다.** depth 는 Blender metre 실측,
instance mask 는 object pass index 라 픽셀 단위로 정확하다. 그리고 결정적으로 **소스 영상이 못 본
공간까지 안다** — 벽 뒤, 다른 방, 가구 내부.

## T0. recording → 49프레임 창 — `trumans_clip.py`

TRUMANS 는 30 fps 로 1,500~3,400 프레임짜리 연속 녹화다. 어디를 자르느냐가 그대로 씬이 되므로
손으로 고르면 재현이 안 된다. 창 점수 = ① 카메라 회전 폭(deg) ② 사람 이동 거리(m) ③ action label
종류 수. 셋 다 큰 창이 "카메라도 돌고 사람도 움직이는" 구간이다 — 정지 구간은 시차가 0 이라
OBB 깊이축이 무너진다.

배포본에는 카메라가 없어서 `smplx_result`(world) / `smplx_result_in_cam`(camera) 쌍에서 복원하는데,
그 카메라는 **사람을 반경 ~2.1 m 로 따라다니는 가상 카메라**라 씬 고정 카메라가 아니다.
그래서 여기선 pose 를 안 쓰고 "얼마나 돌았나"라는 **창 선택 지표로만** 쓴다.

## T1. 소스 카메라 합성 + GT 렌더 — `trumans_to_recon.py`

**소스 카메라를 합성하는 건 선택이 아니라 데이터의 제약이다.**
- `<seq>_camera_pose.pkl` 은 recording 67편 중 **2편**에만 있다 (00add26c, 0aa05d5a).
- `.blend` 안의 CAMERA 오브젝트 4개는 **전부 정지** (`anim=False`, constraint/parent 없음).

정지 카메라를 소스로 쓰면 시차가 0 이라 `τ`(=이동량/깊이) 축과 view-angle 축이 통째로 무의미해진다.
그래서 있는 2편의 통계에 맞춰 합성하는데, **그 통계는 사람이 걷느냐로 갈린다** (49프레임 창 실측):

| 창 종류 | 00add26c (창 206/15) | 0aa05d5a (창 143/11) |
|---|---|---|
| still (보행 ≤10%) | cam net 0.440 m, dt 0.0122 | cam net 0.548 m, dt 0.0137 |
| walk (보행 ≥80%) | cam net 1.314 m, dt 0.0309 (사람 net 1.250 → 추종 **1.05x**) | cam net 0.588 m, dt 0.0141 (사람 net 1.046 → **0.56x**) |

즉 실제 카메라 2대는 보행에 정반대로 반응한다. `--track_gain 0.6` 이 그 둘 사이에 앉는다.
정리하면 소스 카메라는 "높이 고정 + 사람 추종 + 느린 호(arc)".

시작 pose 는 `trumans_scene_probe.py` 가 사람 가슴을 중심으로 (방위각 × 고도 × 거리) 격자를 깔고
**실제 광선**으로 ① 시선이 뚫리는지 ② 벽 속/벽에 붙었는지 ③ 공중에 떠 있는지를 재서 통과분에서만
샘플링한다 — Part A 의 depth shell 근사(G1)를 여기선 안 쓴다. full-house 씬이라 카메라를 아무데나
두면 **벽 안쪽**에 박힌다.

**단계** (Blender 3회 + 변환 1회):
```
1. probe(격자)   ~4 s    사람 궤적 + 설 수 있는 자리
2. 궤적 합성      즉시    시작 pose 샘플 + preset 호 + 사람 추종 look-at
3. probe(검증)   ~4 s    합성한 49 pose 를 프레임별로 다시 광선 검사
4. gt_render     ~150 s  RGB + depth/index
5. 변환          ~10 s   video.mp4 / depths(EXR f16) / masks / cameras.npz / seg_instances
```

### `trumans_gt_render.py` — 재논의 금지 실측

- `BLENDER_EEVEE_NEXT` 에는 object index pass 가 **아예 없다**. index 는 Cycles 전용.
- EEVEE 의 Z pass 는 픽셀 footprint 위에서 필터링되어 실루엣에서 틀린다 (1.6% 픽셀이 1 cm 이상,
  최대 3.85 m). → **depth+index 는 항상 Cycles 1 spp CPU** (~1.8 s; Cycles GPU 는 오히려
  느리고(2.55 s) OptiX 커널 빌드에 일회성 437 s 를 더 쓴다 → `--cdevice CPU` 기본).
- **RGB 엔진은 EEVEE 가 최적이 아니다.** blend 는 Blender **3.3.6 저작**인데 우리는 4.5.9 로
  돌린다. 4.2 에서 EEVEE 가 EEVEE_NEXT 로 전면 재작성됐고, 3.3 기준 발광 재질(노트북 20 / TV 20 /
  조명 175~469)이 흰 덩어리로 타서 **옆 물체까지 번진다**. 같은 프레임을 Cycles 로 렌더하면 번짐이
  0 (frame 402 crop 안 `max>=200` 픽셀: EEVEE 401 → Cycles **0**). `--rgb_engine cycles` 가 탈출구
  (기본은 하위호환 때문에 `eevee`).
- depth 는 **z-planar**, metre, 배경 sentinel `1e10`, 픽셀 중심 `(col+0.5, row+0.5)`.
- 사람은 ARMATURE **`zzy3`** 이고 그 자체는 아무것도 렌더하지 않는다. 실제로 그려지는 건 그
  armature 가 deform 하는 mesh 12개(`CC_Base_Body`, `Layered_sweater`, `Slim_Jeans`, …)라
  그 **union 을 index 1 로 예약**한다. 이름 하드코딩이 아니라 armature modifier / parent 로 찾는다.
- `--cycles` 로 시작하는 CLI 플래그를 **절대 만들지 말 것** — Cycles 애드온이 `--` 를 무시하고
  argv 전체를 prefix-match 로 훑어 실행이 통째로 죽는다 (그래서 `--cdevice`).
- TRUMANS `.blend` 는 `scene.frame_step = 2` 로 저장되어 있다 → 1 로 되돌린다.

## T2. mesh 충돌 격자 — `build_trumans_mesh_grid.py`

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

## T3. scene graph — GT 소스로

Part A 와 **같은 스크립트**(`build_scene_graph.py`)를 쓰되 세 축을 GT 로 바꾼다:

```
--gravity_source gt --ground_source gt --subject_source gt
--scene_scale_mode points_first_cam --scene_scale_stride 1 --no_skip_done
```

`.blend` 가 중력·지면·사람을 다 알고 있으므로 GeoCalib(S3)도 ground RANSAC 도 안 돈다.
같은 인자로 `python -m lbm.cloud` 도 돈다 (마커 `.graph_s115` / `.cloud_s115`).

> 마커 이름이 `s115` 인 건 D115 산출물과 **비트 동일**해야 하기 때문이다. D132 드라이버의 ⓪ 단계
> 인자는 `run_trumans_d115_shard.sh:46-84` 와 바이트 단위로 같아야 한다.

## T4~T6. 뱅크 — `run_trumans_d132_shard.sh`

사용자 지시 2026-09-05: *"blender scene 에서 3d mesh 로 fitting / context video 에 안 보이는
부분이더라도 충돌·clearance 고려 / 전체 scene 에 안 부딪히도록 fitting 한 sequence 에 대해서만 /
target 은 우선 man 만"*.

**preset 라우팅이 없다.** `--nodes dyn_0` 으로 사람 하나만 anchor 하고 preset 은 전량 돈다.
`sample_camera_bank.py:144-149` — 명시 `--nodes` 는 최우선이라 `max_*_anchors`/`drop_surfaces`
필터를 **건너뛴다**. D115/D116 은 anchor 11개(dyn_0 + stat_0..9)였고 6,758 solved 행 중 사람은
968행(14.3%)뿐이었다. D132 는 사람만.

τ 뱅크·hole fit 축은 Part A 의 D128 과 같고, **다른 건 두 줄이다**:

```
--behind_clear_src_ratio 0
--collision_time_match --collision_source both     # ← depth ∪ mesh. Part A 는 depth
```

기존 뱅크 두 개가 왜 못 쓰는지 (D105 명명 원칙 — 게이트 규약이 바뀌면 폴더 이름을 바꾼다):
- `hole_bank_k6_d128` — `collision.source == "depth"`. 벽을 뚫고 뒤로 빠져도 `behind_frac` 이
  정확히 `0.0000` 으로 나온다. **증거의 부재를 통과로 읽은 것**이다.
- `hole_bank_k6_d116` — `both` 를 쓰지만 D125~D128 이전이라 `tau_ref == "auto"`,
  `tracking == "drift"`, anchor 11개.

"안 부딪힌 sequence 만"은 `bank.csv` 의 `status == "solved"` 가 집행한다 (Part A 와 동일).

## T7. 캡션 + export

- **`metadata.csv` 가 없다** → `build_trumans_metadata.py` 가 `manifest_a<NN>.json` 의
  `caption.{target,event}` 737개로 **Vista 와 같은 스키마**의 csv 를 만든다 (캡션 스크립트 수정 0).
  TRUMANS action label 은 명령형(`put down the book with both hands`)이라 Vista 의 서술문에 눈금을
  맞추려고 3인칭 진행형으로 바꾼다. 어휘가 **26개 동사머리로 닫혀 있어** 휴리스틱 대신 명시적
  gerund 표를 쓴다 — `open` 에 자음중복 규칙을 돌리면 `openning` 이 나오는 종류의 조용한 오류를
  애초에 안 만들기 위함.
- **라벨이 사람 이름이 아니다** → `build_bank_captions.py --label_map`.
  Vista/dynpose 는 라벨이 SAM3 텍스트 프롬프트(=사람이 쓴 명사)라 손댈 게 없지만, TRUMANS 는
  라벨이 `.blend` 오브젝트 이름이다 (`Layered_sweater.001`, `zzy3`, 숫자 에셋).
  `normalize_label` (:221) 이 `strip_suffix` 로 `.001` 을 떼고 stems 사전으로 옮기며,
  숫자 에셋은 `"furniture"`, 미상은 `"object"` 로 떨어진다.
  `--label_map` 을 안 주면 `label_map is None` 이라 입력을 그대로 돌려준다 = **Part A 는 무변경**.
- **export 는 별도 스크립트** `trumans_lite_to_dl3dv.py` — `--layout per_recording`(기본,
  `per_clip` 선택), `--chunk_prefix trumans`, `--test_recordings 2b4c9b84`,
  `--avg_scale_refs context_first_cam`, `--clip_avg_scale_ref centroid`. 한 recording 이 여러
  chunk 를 낳으므로 train/test 를 **chunk 가 아니라 recording 단위**로 갈라야 누수가 없다.

---

# 부록 — 실행 순서 요약

## 실사 영상 (dynpose 880편 기준)

```bash
# S1  VLM 명사 (선행: exec/serve_qwen3vl.sh 로 vLLM 기동)
screen -dmS nouns bash exec/_legacy/run_dynpose_d145_nouns.sh
# S2  SAM3 동적
screen -dmS sam0 bash exec/_legacy/run_dynpose_d145_sam3.sh <gpu> 0 2
#     SAM3 정적
python fit/graph/extract_static_nouns.py --source vlm
CUDA_VISIBLE_DEVICES=<gpu> python fit/ingest/sam3_static_instances.py --num_shards 2 --shard_id 0
# S3  GeoCalib 사이드카 (env geocalib)
bash exec/_legacy/run_dynpose_d148_geocalib.sh <gpu> 0 2 <video_list>
# S4~S9  그래프 → cloud → route → τ → fit → emit
bash exec/_legacy/run_dynpose_d129_shard.sh <gpu> 0 4 <video_list> <log_dir>
# S10~S11  describe → 캡션 → export → dd10
bash exec/_legacy/run_dynpose_d147_caption_export.sh
```

## TRUMANS

```bash
bash exec/_legacy/run_trumans_d132_shard.sh <gpu> <shard> <nshard> <chunk_list> <log_dir>
#   PREP=1 (기본) 프런트엔드(⓪-a graph / ⓪-b cloud / ⓪-c mesh_grid)까지 빌드
#   PREP=0        원본 48-chunk 실행을 비트 동일하게 재현
python fit/ingest/build_trumans_metadata.py
python fit/caption/build_bank_captions.py --label_map <map> ...
python fit/ingest/trumans_lite_to_dl3dv.py --layout per_recording ...
```

## env

| env | 쓰는 곳 |
|---|---|
| `vista4d` | SAM3, VLM 명사/지칭, 뱅크(τ/fit/emit), 캡션, export |
| `da3` | `build_scene_graph.py`, decode/verify |
| `geocalib` | `geocalib_gravity.py` (kornia) |
| `vllm` | Qwen3-VL-30B @ 22002 — **쓸 때만 올리고 끝나면 내린다** |
| Blender | `/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender` (씬은 3.3.6 저작) |
