# LBM-Lite — RGBD + 4D 점군 + scene graph 위의 look-before-move 카메라 플래너

영상 1편(dynamic video)에서 **새 카메라 궤적**을 만든다. 핵심은 최적화가 아니라 **순서**다 —
후보 카메라를 *먼저 렌더해서* VLM 에게 보여주고, VLM 은 **렌더 가능한 것들 중에서만** 고른다.
공간 제약("카메라가 벽 속에 있나 / 소스가 못 본 영역을 보나")은 프롬프트가 아니라 **렌더러와
게이트가** 집행한다.

```
video ──► scene_graph.json ──► cloud.npz ──► board(후보 렌더+게이트) ──► VLM ──► decision.json
                                                                                    │
                        canonical.json/.npz ◄── poses.npz (49,4,4) ◄────────────────┘
                                │
                                └──► video_generation/tools/recammaster/emit_model_cams.py (6개 모델)
```

폴더 이름이 `CinemaTraj` 인 것은 역사적 이유다. **폐기된 것은 CinemaTraj 의 텍스트-전용
플래너**(SDF + 앵커 스코어링 + Adam 최적화)이고, 폴더는 planner 작업 폴더로 재사용 중이다.

---

## 빠른 실행

```bash
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python

# 전 단계 한 번에 (graph → board → loop → decode → emit → verify)
$PY run_lbm_lite.py --stage all --videos camel --cuda 1

# 무엇이 돌지 먼저 본다
$PY run_lbm_lite.py --stage all --dry_run

# 단계 하나만
$PY run_lbm_lite.py --stage board --videos camel --cuda 1
```

각 단계는 여전히 **단독 실행 가능**하다. `run_lbm_lite.py` 는 subprocess 로 부르기만 한다:

```bash
CUDA_VISIBLE_DEVICES=1 $PY fit/graph/build_scene_graph.py     --video camel
CUDA_VISIBLE_DEVICES=1 $PY fit/bank/build_candidate_board.py --video camel
CUDA_VISIBLE_DEVICES=1 $PY -m lbm.loop                      --video camel
CUDA_VISIBLE_DEVICES=1 $PY decode/build_poses.py            --video camel
CUDA_VISIBLE_DEVICES=1 $PY decode/emit.py                   --video camel
CUDA_VISIBLE_DEVICES=1 $PY verify.py                        --video camel
```

**오케스트레이터를 쓰는 이유는 하나다** — `start_mode` 는 loop/decode/emit/verify 네 군데,
`aim_anchor`/`aim_ramp_frames` 는 decode/emit/verify 세 군데에 각자 기본값으로 들어 있다.
어긋나면 emit 의 stale-poses 가드에 걸리거나(운이 좋을 때) 지문이 같아서 조용히 옛 pose 를
재사용한다(운이 나쁠 때). `configs/default.json` 의 `shared` 블록이 **그 값을 받는 단계에만**
뿌려서 어긋날 자리를 없앤다.

### env

**전 단계가 `vista4d` 하나에서 돈다.** `lbm/vlm.py` 를 `requests`/`urllib` 만 쓰는 OpenAI 호환
클라이언트로 짰기 때문에 `openai`/`any_llm` 이 필요 없다. `run_lbm_lite.py` 는 단계 시작 전
`--help` 로 import 를 찔러보고, 실패하면 멈추고 `conda run -n <env>` 명령을 찍는다.

VLM 백엔드는 로컬 vLLM `Qwen/Qwen3-VL-30B-A3B-Instruct` @ `http://127.0.0.1:22002/v1`
(`exec/serve_qwen3vl.sh`). `--api_base`/`--model` 로 교체 가능.

### 설정

`configs/default.json` (`lbm_lite_config_v1`) 한 파일. CLI 에서 덮어쓰려면:

```bash
$PY run_lbm_lite.py --stage board --set board.board_size=27 --set board.board_columns=9 \
    --set board.tile_width=480 --set board.tile_height=270      # 예전 27칸 board 복원
$PY run_lbm_lite.py --stage all --set loop.no_vlm=true          # VLM 없이 fallback 결정만
```

bool 은 집 스타일 `--flag`/`--no_flag` 쌍으로 나간다 (`"allow_zoom": false` → `--no_allow_zoom`).

---

## 단계별 산출물

| stage | 스크립트 | 산출물 |
|---|---|---|
| `nouns`* | `fit/graph/extract_static_nouns.py` | `out/static_nouns/static_nouns.json` |
| `seg`* | `fit/ingest/sam3_static_instances.py` | `<eval_data>/eval_data/seg_instances_static/<video>/` |
| `graph` | `fit/graph/build_scene_graph.py` | `out/<video>/scene_graph.json`, `vis/obb_overlay.mp4`, `vis/topdown.png` |
| `board` | `fit/bank/build_candidate_board.py` | `out/<video>/board/{board_candidates.png, source_frames.png, contract.txt, gates.csv, board.json}`, `cloud.npz` |
| `loop` | `python -m lbm.loop` | `out/<video>/decision.json`, `trace/turn_<nn>.json`, `trace/board_presets.png` |
| `decode` | `decode/build_poses.py` | `out/<video>/poses.npz` (49,4,4) |
| `emit` | `decode/emit.py` | `out/<video>/canonical/{canonical.json, canonical.npz}` + `emit_model_cams.py` 명령 |
| `verify` | `verify.py` | `out/<video>/verify.json` + `results/<date>_lbm_lite/<video>/` 프리뷰 6종 |

`*` = `--stage all` 에 안 들어간다. 공유 `eval_data` 디렉토리에 쓰고 SAM3 가 GPU 를 오래 잡아서
이름을 직접 줘야 돈다.

사람이 보는 순서는 정해져 있다: **`obb_overlay.mp4` 부터.** 박스가 물체를 안 감싸면 하류가 전부
무효다. 그 다음 `board_candidates.png`(VLM 이 실제로 본 그림) → `plan_sbs.mp4` → `plan_cam.mp4`.

### 지금 상태 (2026-08-21, camel / avocado-slice)

| | camel | avocado-slice |
|---|---|---|
| subject | `dyn_*` (camel) | **`stat_0 (table)`** — 정적 노드 |
| pool → 게이트 통과 → board | 45 → 33 → 9 | 45 → 14 → 9 |
| preset (현재 디스크 산출물) | `orbit_left_arc` (steady, drift) | `orbit_left_arc` (steady, drift) |
| `hole_fraction` | 0.0737215 | 0.289933 |
| `tau_max` | 0.196051 | 0.197946 |
| `max_view_angle_delta` | 12.9615° | 13.7096° |
| `path_len_u` | 0.147629 | 0.15771 |
| emit `--scales` | 0.346156 | 0.36666 |
| verdict | PASS | PASS |

이 절의 preset 이름은 **측정 당시(D76 rename 전) 이름 그대로**다 — `orbit_left_arc` = 지금의
`orbit_left`, `static_hold_locked` = 지금의 `static_hold`(D94 전에는 `static_hold_dont_look`).
실측 기록이라 안 바꿨고,
아래 인용된 VLM 응답도 원문이다. 매핑은 `lbm/presets.py` `PRESET_ALIASES`.

⚠ **avocado-slice 의 preset 은 draw 마다 뒤집힌다.** 같은 설정에서 `-m lbm.loop` 를 7번 돌리면
**6번은 `static_hold_locked`**(tracking lock, `path_len_u 0.0000`, emit `--scales 0`),
1번만 `orbit_left_arc` 다. camel 은 5/5 `orbit_left_arc`. 디스크에 남아 있는 avocado 산출물은
소수파 draw 쪽이다. VLM 이 정적을 고르는 이유는 프롬프트에 그대로 적혀 있다 —
`"static_hold_locked ... maintains 100% coverage throughout the shot with no camera motion,
ensuring no hallucinated regions"` (coverage 0.99). **contract 가 coverage 를 상으로 주고
움직임에는 아무 상도 주지 않으므로, 안 움직이는 게 coverage 최적해다.**

지표 9번 `path_len_u` 가 그 퇴화를 잡으라고 있는 것이다 (계획서 8종에는 없다). 정적 draw 를
실제로 통과시켜보면 **나머지 8종이 전부 PASS 인데 `path_len_u 0` 만 FAIL** 이라 verdict 가
FAIL 로 뒤집힌다. `verify.py --min_path_len_u 0` 이면 예전처럼 안 잰다.
근본 해결(움직임 보상을 contract 에 넣기 / subject 를 동적 노드로 강제)은 아직 안 했다.

---

## 지금 가능한 motion — preset 23종 (`lbm/presets.py`)

수치는 `python -m lbm.presets --radius 0.6 --obs_az_span 90` 실측(ctx: radius 0.600 /
dolly 0.210 / lateral 0.210 / sweep 45° / pan 20°). `move` 는 궤적 시작→끝 직선거리,
`rot_deg` 는 총 회전, `p/c` 는 경로길이÷현길이(1.00 = 직선, 클수록 휨; 이동 0 이면 nan).
**절대 크기는 여기 숫자가 아니라 `fit_tau` 가 `--target_tau` 로 다시 맞춘다** — 이 표는 모양 비율만.

`aim` 은 카메라가 어디를 보느냐다. `look_at` = 위치만 궤적에서 받고 **매 프레임 subject 를 다시
조준**(그래서 `tracking` world/drift/lock 이 의미를 가진다), `traj` = **궤적이 들고 있는 회전을
그대로** 쓴다(`tracking` 은 무시되고 `tracking_ignored: true` 로 찍힌다). pan 을 look_at 으로
처리하면 조준이 회전을 상쇄해 정지 shot 이 되므로 이 구분이 필수다.

**이름 규칙 (D76)**: `[track_]<primitive>_<direction>[_<primitive2>_<direction2>][_dont_look]`.
**조준(`aim="look_at"`)이 기본값이라 이름에 안 적고**, 그 기본에서 벗어나는 쪽에만 `_dont_look`
을 붙인다 (비대칭 — LAMP DSL 방식). `pan`/`truck`/`pedestal` 은 primitive 가 이미 aim 을 정하므로
접미사가 없다. 옛 이름(`straight_ease`, `orbit_*_arc`, `*_reveal`, `static_subtle_zoom`,
`track_side_*`, `track_follow*`, D75 의 `_aimed`/`_locked`)은 `lbm/presets.py` `PRESET_ALIASES`
에 의미 그대로 남아 있다.

### aim = look_at (subject 재조준, 22종)

| preset | move | rot° | p/c | 무엇 |
|---|---|---|---|---|
| `dolly_in` / `dolly_out` | 0.210 | 0.00 | 1.00 | 직선 접근/후퇴, 조준 유지 |
| `push_in_arc_left` / `push_in_arc_right` | 0.284 | 22.50 | 1.01 | 살짝 돌며 다가감 |
| `pull_out_arc_left` / `pull_out_arc_right` | 0.345 | 22.50 | 1.00 | 살짝 돌며 물러남 |
| `orbit_left` / `orbit_right` | 0.471 | 45.00 | 1.03 | 궤도 |
| `orbit_left_pedestal_up` | 0.516 | 45.00 | 1.02 | 궤도 + 상승 |
| `crane_up` / `crane_down` | 0.210 | 19.29 | 1.00 | 크레인, 조준 유지 |
| `s_curve` | 0.468 | 0.00 | nan | arc(+) → arc(−) |
| `static_look_at` | 0.000 | 0.00 | nan | 제자리, **조준은 subject 를 따라감** (D94 전 이름 `static_hold`) |
| `track_look_at` | 0.000 | 0.00 | nan | subject 변위만 싣고 그 외 정지, **매 프레임 재조준** (D94 전 이름 `track_hold`) |
| `track_dolly_in` / `track_dolly_out` | 0.210 | 0.00 | 1.00 | 추종 + 전후 |
| `track_orbit_left` / `track_orbit_right` | 0.471 | 45.00 | 1.03 | 추종 + 궤도 |
| `track_crane_up` / `track_crane_down` | 0.210 | 19.29 | 1.00 | 추종 + 크레인 |

### aim = traj (궤적 회전 그대로, 14종)

| preset | move | rot° | zoom | 무엇 |
|---|---|---|---|---|
| `pan_left` / `pan_right` | 0.000 | 20.00 | — | 제자리 yaw |
| `truck_left` / `truck_right` | 0.210 | 0.00 | — | 좌우 평행이동 |
| `pedestal_up` / `pedestal_down` | 0.210 | 0.00 | — | 상하 평행이동 |
| `dolly_in_dont_look` / `dolly_out_dont_look` | 0.210 | 0.00 | — | 전후 평행이동, 재조준 없음 |
| `pan_right_zoom_out` | 0.000 | 20.00 | **yes** | pan + focal ×1/1.5 |
| `static_hold` | 0.000 | 0.00 | — | 완전 고정 (D94 전 이름 `static_hold_dont_look`) |
| `static_zoom_in` | 0.000 | 0.00 | **yes** | 고정 + focal ×1.15 |
| `track_hold` | 0.000 | 0.00 | — | 추종만, 재조준 없음 (D94 전 이름 `track_hold_dont_look`) |
| `track_truck_left` / `track_truck_right` | 0.210 | 0.00 | — | 추종 + 좌우 |
| `track_pedestal_up` / `track_pedestal_down` | 0.210 | 0.00 | — | 추종 + 상하 |

`track_*` 12종은 `PRESET_FOLLOW` 로 `follow_gain=1.0` 이 기본으로 걸린다 (subject 변위를 카메라
**위치**에 싣는다). 위 표의 `move`/`rot°` 는 그 follow 채널을 뺀 preset 자체의 모양이라
non-track 짝과 같은 값이다 — 실제 이동량은 subject 가 얼마나 움직이느냐가 더한다.

### primitive 겹침으로 다시 분류

위 두 표는 `aim` 기준이라 "무엇을 몇 개 겹쳤나"가 안 보인다. `traj.py` primitive 기준으로는
이렇게 갈린다 (22 / 8 / 5 / 1). `track_*` 는 **primitive 를 하나도 안 더한다** — 같은 builder 에
`follow_gain` 만 얹은 것이라 짝이 되는 non-track preset 과 같은 칸에 들어간다.

`조준` 열이 **look-at target ON/OFF** 다 — ON = `aim="look_at"`(매 프레임 subject 재조준,
`tracking` 유효), OFF = `aim="traj"`(궤적 회전 그대로, `tracking_ignored: true`).
**preset 속성이라 CLI 로 못 뒤집는다.** 단 `--aim_keyframes ≥2` 면 OFF preset 에도 조준이 걸린다
(§조준 3축).

**① 단일 축 — primitive 하나, 회전이나 이동 한쪽만 (22종)**

| primitive | preset | 조준 |
|---|---|---|
| `dolly` | `dolly_in`, `dolly_out`, `track_dolly_in`, `track_dolly_out` | **ON** |
| `dolly` | `dolly_in_dont_look`, `dolly_out_dont_look` | OFF |
| `truck` | `truck_left`, `truck_right`, `track_truck_left`, `track_truck_right` | OFF |
| `pedestal` | `pedestal_up`, `pedestal_down`, `track_pedestal_up`, `track_pedestal_down` | OFF |
| `pan` | `pan_left`, `pan_right` | OFF |
| `hold` | `static_look_at`, `track_look_at` | **ON** |
| `hold` | `static_hold`, `track_hold` | OFF |
| `hold`/`pan` + **focal** | `static_zoom_in`, `pan_right_zoom_out` | OFF |

`dolly_in` 과 `dolly_in_dont_look` 은 **궤적이 완전히 같고 조준만 다르다**. `static_look_at` 과
`static_hold` 도 마찬가지 (전자만 정지한 채로 subject 를 눈으로 따라간다).
⚠ **D94 에서 이 두 이름이 서로 바뀌었다** — `hold` 는 "안 움직이고 재조준도 안 한다"는 뜻이어야
하므로, 옛 `static_hold`(조준 추종)가 `static_look_at` 이 되고 옛 `static_hold_dont_look`(완전
고정)이 `static_hold` 를 물려받았다. **track 쪽도 똑같이 맞바꿨다** — 옛 `track_hold`(조준 추종)
→ `track_look_at`, 옛 `track_hold_dont_look`(완전 고정) → `track_hold`. 어휘 전체가 한 규칙이다:
조준은 `_look_at` 으로만 표기하고 `hold` 는 어디서든 "안 움직이고 재조준도 안 한다"를 뜻한다.
마지막 줄 2종은 SE(3) 축은 1개지만 **focal 채널이 하나 더 얹힌다**. focal 은 `traj.py` 밖
(`PRESET_ZOOM_END` + `focal_track`)이라 `compose` 도 `fit_tau` 도 안 거친다.

**② primitive 하나인데 그게 이미 회전+이동 묶음 (8종, 전부 조준 ON)**

| primitive | 내부에서 무엇이 묶여 있나 | preset | 조준 |
|---|---|---|---|
| `true_orbit` | 원 위 이동 + yaw 를 **동기**시켜 중심을 계속 봄 (`traj.py:113`) | `orbit_left`, `orbit_right`, `track_orbit_left`, `track_orbit_right` | **ON** |
| `crane` | 수직 이동 + `−atan(dist/radius)` tilt 보정 (`traj.py:127`) | `crane_up`, `crane_down`, `track_crane_up`, `track_crane_down` | **ON** |

`arc` 도 여기 속하지만(현 위 직선 이동 + yaw, `traj.py:99`) 단독으로 쓰는 preset 은 없고
③ 안에서만 쓰인다. **`arc` 는 진짜 원호가 아니다** — path/chord = 1.0000, 중간 프레임이 원
안쪽으로 파고든다. 진짜 궤도가 필요하면 `true_orbit`.

**③ `compose()` 로 2개 이상을 동시에 (5종, 전부 조준 ON)**

| preset | 겹친 것 | 축 수 | 조준 |
|---|---|---|---|
| `push_in_arc_left` | `arc(+σ/2)` ∘ `dolly(+d)` | 3 (측면이동 + yaw + 전진) | **ON** |
| `push_in_arc_right` | `arc(−σ/2)` ∘ `dolly(+d)` | 3 | **ON** |
| `pull_out_arc_right` | `arc(−σ/2)` ∘ `dolly(−d)` | 3 | **ON** |
| `pull_out_arc_left` | `arc(+σ/2)` ∘ `dolly(−d)` | 3 | **ON** |
| `orbit_left_pedestal_up` | `true_orbit(+σ)` ∘ `pedestal(+d)` | 3 (궤도이동 + yaw + 상승) | **ON** |

**④ 시간축 이어붙임 — 동시가 아니라 순차 (1종, 조준 ON)**

`s_curve` = `arc(+σ/2)` 를 절반 돌고 그 끝점에서 `arc(−σ/2)` 를 이어 붙인다
(`_s_curve`, `concatenate`). ③ 과 달리 두 동작이 **겹치지 않는다**.

**조준 ON 22종 / OFF 14종.** ON 은 ②③④ 전부 + `dolly_in`/`dolly_out`(및 `track_` 짝) +
`static_look_at`/`track_look_at`, 나머지가 OFF.

**주의 — `aim="look_at"` 이면 겹친 회전이 실제로는 버려진다.** 디코더가 매 프레임 조준을 다시
세우므로(`build_poses.py:12`) 궤적이 들고 있던 회전은 덮어써진다. 그래서 ②③ 중 look_at
preset(= `static_look_at` 빼고 전부)에서 실효 겹침은 **이동 축만**이다:

| preset | 표의 `rot_deg` | 디코드 후 실효 |
|---|---|---|
| `push_in_arc_*` 계열 | 22.5 (arc yaw) | 측면 + 전후 **이동 2축**, 회전은 조준이 만든다 |
| `orbit_left_pedestal_up` | 45.0 (orbit yaw) | 궤도 + 수직 **이동 2축** |
| `crane_up` / `crane_down` | 19.3 (crane tilt) | **수직 이동 1축** — crane 의 tilt 보정은 사라진다 |

즉 `crane_up` 은 디코드 후 `pedestal_up` 과 **위치가 같고 조준만 다르다**.
`aim_keyframes ≥ 2` + `keyframe_aim=preset_rel` 을 주면 preset 회전이 조준 위에 상대회전으로
다시 얹혀 살아난다 (`build_poses.py:51-54`).

### 이동·회전량은 어디서 제한되나 — 4층

**preset 은 "모양"만 정의하고 크기는 안 정한다.** 크기를 정하는 건 아래 L1→L4 순서다.

**L1. 모양 상수 `DEFAULT_SHAPE`** — 이동은 subject 거리 `radius` 의 **비율**, 회전은 **절대 각도**.

| primitive | 크기 인자 | 1x 값 (`--radius r`) | 성질 |
|---|---|---|---|
| `dolly` | `ctx["dolly"]` | `dolly_frac 0.35 · r` | 이동만 |
| `truck` / `pedestal` | `ctx["lateral"]` | `lateral_frac 0.35 · r` | 이동만 |
| `crane` | `dist=lateral`, `radius=r` | 이동 `0.35r`, tilt `atan(0.35)` = **19.29°** (유도값) | 이동→회전 종속 |
| `arc` | `deg=sweep/2`, `radius=r` | 회전 22.5°, 현 `2r·sin(11.25°)` | 회전+이동 |
| `true_orbit` | `deg=sweep`, `radius=r` | 회전 45°, 현 `2r·sin(22.5°)` | 회전+이동 |
| `pan` | `ctx["pan"]` | `pan_deg` = **20°** | 회전만 |
| `hold` | — | 0 | 없음 |

**이동량은 subject 거리에 비례하고 회전량은 거리와 무관**하다. `crane` 의 tilt 는 인자가 아니라
`−atan(dist/radius)` 로 유도되므로 `lateral_frac` 를 바꾸면 같이 움직인다.

**L2. sweep clamp (`shape_context`)** — `sweep = min(sweep_deg, orbit_span_frac · obs_az_span)`,
기본 `orbit_span_frac 0.8`. **`arc` / `true_orbit` 계열에만** 걸린다. 소스가 한 번도 못 본
방위로 도는 걸 막는다. τ 가 나중에 어차피 줄이는데도 여기서 먼저 자르는 이유: SE(3) 로그
스케일은 회전과 이동을 **같은 비율로** 깎으므로, τ 에 맡기면 이동까지 같이 죽는다.

**L3. τ 이분법 (`fit_tau`) — 실제 크기 결정자.** `se3.scale_traj(rel, s)` 가 SE(3) 로그에서
`s` 배 하므로 **회전과 이동이 한 배율로 묶여** 움직인다. 즉 직접 제한되는 건 이동이고
(`τ = max_f |p(f) − p_src(f)| / z_med ≤ target_tau`), **회전은 그 배율을 따라가는 종속변수**다.
이분법 8회, `max_scale 4.0`, 최소 눈금 `4/2⁸ = 0.0156` (그보다 작은 답은 D53 `refine_zero`).

실측 (`--radius 0.6 --obs_az_span 90`, `z_med=1`, 소스 정지):

| preset | τ*=0.10 → s / move / rot | τ*=0.20 → s / move / rot |
|---|---|---|
| `truck_right` | 0.469 / 0.098 / 0.00° | 0.938 / 0.197 / 0.00° |
| `straight_ease` | 0.469 / 0.098 / 0.00° | 0.938 / 0.197 / 0.00° |
| `rise_reveal` | 0.469 / 0.099 / 9.04° | 0.938 / 0.197 / 18.08° |
| `push_in_arc` | 0.344 / 0.098 / 7.73° | 0.703 / 0.200 / 15.82° |
| `orbit_left_arc` | 0.203 / 0.096 / 9.14° | 0.422 / 0.199 / 18.98° |
| `pan_right` | **4.0 (max) / 0.000 / 80.00°** | **4.0 / 0.000 / 80.00°** |

읽는 법: **같은 τ 예산이면 이동량은 primitive 와 무관하게 같고(≈ τ·z_med), 그 예산으로 얻는
회전량만 다르다.** `orbit_left_arc` 가 회전 효율이 제일 좋고(τ 0.10 에 9.14°),
`truck`/`dolly` 는 0°다. 회전은 τ 에 거의 선형(2배 τ → 2배 회전, path/chord 때문에 미세하게 초과).

**`pan` 계열은 L3 가 아예 안 걸린다.** 이동이 0 이라 어떤 `s` 에서도 τ 가 같아
`tau_hi <= target_tau` 가지로 빠져 항상 `s = max_scale = 4.0` → `20° × 4 = 80°` 고정.
사다리 5단이 전부 같은 궤적이 된다 (실측: camel `pan_left` 가 5단 전부 `path_len_u 0.0000`,
`hole 0.740` 동일). 그래서 뱅크는 `ROTATION_ONLY_PRESETS` 3종만 사다리를 **회전 각도**로 옮기고
(`--pan_deg_at_max × rung/max_rung`), `FIT_TAU_MAX_SCALE`(4.0)로 미리 나눠 요청 각도를 되돌린다
(`sample_camera_bank.py:88-138`).

**`true_orbit` 은 sweep 손잡이가 안 먹는다.** 순수 나선(screw)이라 sweep 과 scale 이 상쇄돼
같은 τ 면 같은 궤적이 나온다 (실측 snowboard `dyn_0 orbit_left` τ0.6: sweep 45 → 53.5° /
sweep 135 → **53.5°**, 소수점까지 동일). orbit 을 더 돌리는 손잡이는 sweep 이 아니라 **τ** 다.
반면 `arc ∘ dolly` 합성은 두 성분의 비가 바뀌므로 sweep 이 실제로 먹는다
(`pull_out_arc_left` τ0.6: 34.8° → 46.2°).

**τ 예산 자체**는 `--tau_ladder` (기본 `0.10 0.20 0.35 0.60 1.00`). 시작 pose 가 이미 예산을
다 쓴 경우(`tau_start ≥ target_tau`) `s=0` 으로 궤적을 통째로 얼리고 `tau_saturated` 를 세운다 —
`--drop_saturated`(기본 on)가 그 변이를 뱅크에서 뺀다.

**L4. 사후 게이트 — 크기를 줄이는 게 아니라 변이를 버린다.** `lbm/gates.py`:
G1 behind-surface · G2 coverage · G3 framing · G4 τ/view-angle 프리필터 ·
G5 OBB clearance(카메라가 노드 박스 안) · G6 elevation(subject 바로 위/아래, 지면 아래) ·
`approach_profile`(dolly 가 subject bbox 를 지나쳐 버리는 경우 — G5 의 무부호 거리로는 못 잡는다).

### 이동 **방향**은 어디서 정해지나 — `--traj_basis`

위 L1~L4 는 크기 얘기다. 방향은 따로다. **preset 은 카메라 로컬 궤적**이라 `dolly` 는 언제나
"기준 회전의 +Z", `truck` 은 "+X" 로 간다. 그 **기준 회전**을 무엇으로 잡느냐가
`--traj_basis` 다 (D69, `build_poses.py:444-447`).

| 값 | `basis_c2w` | `dolly` 가 향하는 곳 |
|---|---|---|
| `source` (**기본**) | `c2w_start` = 소스 frame0 회전 | 소스 카메라의 광축 — **anchor 를 안 향할 수 있다** |
| `subject` | `look_at_c2w(c2w_start[:3,3], center[0] + bias, up)` | **정확히 anchor** |

즉 `aim="look_at"` 은 **고개만** 돌린다. `straight_ease` 의 조준은 매 프레임 subject 를
다시 보지만, 병진은 여전히 기준 회전의 +Z 라 anchor 가 광축에서 벗어나 있으면 **옆으로 스쳐
지나간다**. camel 실측 (`straight_ease`, `target_tau 0.2`, 이동 방향과 anchor 방향 사이 각):

| node | `traj_basis=source` | `traj_basis=subject` |
|---|---|---|
| `dyn_0` | 0.47° | **0.00°** |
| `dyn_1` | 4.97° | **0.00°** |
| `stat_0` | 7.61° | **0.00°** |
| `stat_1` | 10.68° | **0.00°** |
| `stat_2` | **12.94°** | **0.00°** |

`stat_2` 의 12.94° 는 half-hfov 14.4° 의 테두리다 — 소스 광축 기준으로는 화면 끝에 있는
물체라, 기본값에서 그 물체로 "다가가는" 샷은 사실 화면을 가로지르는 샷이다.
(같은 이유로 `aim="traj"` preset 은 아예 끝까지 anchor 를 안 본다: camel `stat_2__pan_right`
는 `subject_in_frame 0.000`, aim=traj 120 변이 평균 0.615, 그중 20 개는 49프레임 내내 0 픽셀.)

`subject` 로 두면 `truck`/`pedestal` 의 이동 방향도 같이 돌아간다 — "subject 기준 왼쪽으로
truck" 이 되는 것이고 의도한 동작이다. **frame0 위치는 안 바뀌고**, frame0 회전도
`aim_anchor` 의 smoothstep 이 소스로 되돌리므로 첫 프레임 그림은 그대로다.
τ 는 world 이동량이라 `fit_tau` 도 `basis_c2w` 위에서 다시 푼다 (`c2w_start` 로 풀면
`traj_basis=subject` 에서 τ 가 목표를 빗나간다).

**단, 기준은 frame 0 에 한 번 고정된다.** subject 가 움직여도 `dolly` 는 frame0 방향으로 계속
간다 — "매 프레임 현재 subject 중심을 향해 전진" 모드는 없다. 위치를 계속 따라가게 하려면
`--follow_gains`(아래 ③)를 같이 켠다. 실측 거리비(끝/시작)는 위 표 조건에서 0.76~0.95 로,
`straight_ease` 는 τ 0.2 예산에서 거리를 5~24% 좁힌다.

### 조준 3축 — track / no-track 은 서로 다른 세 손잡이다

"subject 를 따라간다"가 세 군데에 따로 있다. 헷갈리면 안 되는 게, **`--trackings` 에는 `none`
이 없다** — 조준 자체를 끄는 건 `aim` 이고, 카메라가 실제로 따라 **움직이는** 건 `--follow_gains`
다 (`build_poses.py:230-232`: `tracking` 은 조준점만 옮기므로 CameraBench 로는 pan-/tilt-tracking
= 회전만, `follow` 는 병진이라 시차가 생기는 tail-/lead-/side-/aerial-tracking).

| 축 | 손잡이 | 값 | 끄면 | 기본 |
|---|---|---|---|---|
| **① 조준 ON/OFF** | `aim` (**preset 고정**, CLI 없음) | `look_at` / `traj` | `traj` preset 을 고른다 = 조준 없음 | preset 마다 위 표 |
| **② 조준점이 subject 를 얼마나 따라가나** | `--trackings` | `world` 0.0 / `drift` 0.6 / `lock` 1.0 (`TRACKING_GAIN`) | `world` = frame0 중심에 **고정**(그래도 그 점은 계속 본다). τ 최소 | `drift` |
| **③ 카메라 위치가 subject 를 따라가나** (= 진짜 tracking shot) | `--follow_gains` | `0` \| 실수 \| `auto` | **`0` = no-track**, 정확히 영벡터라 기존 동작과 비트 동일 | `0` |

- ② 는 `aim="look_at"` 에서만 유효하다. `traj` preset 에 주면 무시되고
  `tracking_ignored: true` 로 기록된다 (`build_poses.py:436`). 단 `--aim_keyframes ≥2` 를 켜면
  **전 preset 에 적용**된다 — keyframe 조준이 `traj` preset 에도 조준점을 만들기 때문.
- ③ `auto` 는 τ 를 최소로 만드는 gain 을 푼다. 소스가 이미 subject 를 쫓아간 영상에서 예산을
  **되찾는** 용도다 (실측: snowboard τ 1.9441 / snow-bike 1.3765 로 뱅크 360 변이 중 350 이
  `tau_saturated` 로 죽었다). 소스가 회전으로만 따라간 영상은 g*≈0 이 나와 저절로 ③ 이 꺼진다.
  가드 2개: `node["moving"]` 이 아니면 안 걸고(정적 노드도 OBB 중심이 벽을 따라 미끄러진다 —
  truck-pose `stat_0` 0.535u), τ 이득이 `min_benefit` 미만이면 0 으로 되돌린다.
- ③ 을 켜면 subject track 잔여 고주파가 gain 배로 카메라 위치에 실린다. `--follow_smooths`
  (savgol p=2, 기본 9) 가 그 저역통과다. 1 = 끔 (D72 원래 동작).

### 알아둘 것

- **정지 5종** `STATIC_PRESETS = static_hold / static_look_at / static_zoom_in /
  track_hold / track_look_at` (+ 옛 이름 `static_hold_dont_look` / `track_hold_dont_look` 이
  목록에 남아 있어 디스크의 옛 뱅크 행도 resolve 전 문자열 그대로 정지로 잡힌다).
  조준이 subject 를 따라가는 건 `static_look_at` / `track_look_at` 뿐이고, 이쪽만 동적
  subject 에서 실제로 회전이 생긴다. `track_*` 는 카메라 **위치**도 subject 변위만큼
  움직이므로 "정지"는 subject 기준이지 world 기준이 아니다.
  **D94 에서 static/track 이름 규칙이 맞춰졌다** — 양쪽 다 `hold`=완전고정 /
  `_look_at`=조준추종 이다 (옛 `*_hold`=조준추종, 옛 `*_hold_dont_look`=완전고정).
- **회전 전용 3종** `ROTATION_ONLY_PRESETS = pan_left / pan_right / pan_right_zoom_out`
  (`fit/bank/sample_camera_bank.py:92`). 이동이 0 이라 τ 이분법이 아무 스케일에서나 같은 τ 를 내고
  항상 `max_scale=4.0` 으로 saturate 한다 — 뱅크는 이 3종에만 divisor 를 `FIT_TAU_MAX_SCALE` 로
  바꿔서 크기를 정한다(`:133`).
- **zoom 2종은 기본 off.** `--allow_zoom` 없으면 후보에서 빠진다. focal 은 SE(3) 밖이라
  `fit_tau` 스케일이 안 먹고(광학 zoom 은 시차를 안 만든다), `emit_model_cams.py` 에 intrinsics
  채널이 없어서 emit 을 못 통과한다. 램프는 로그 선형(`focal_track`).
- **orbit sweep 은 `min(sweep, 0.8 × obs_az_span)` 으로 clamp** (`shape_context`,
  `--orbit_span_frac`). 소스가 못 본 방위로 도는 걸 모양 단계에서 미리 자른다.
- **별칭 38종** (`PRESET_ALIASES`, `resolve_preset` 이 조회 전에 푼다): 옛 축약형
  (`static`→`static_look_at`, `locked_off`→`static_hold`,
  `push_in`/`zoom_in`→`push_in_arc_left`, `pull_out`/`zoom_out`→`pull_out_arc_right`),
  D75 이전 이름(`straight_ease`→`dolly_in`, `orbit_left_arc`→`orbit_left`,
  `rise_reveal`→`crane_up`, `static_subtle_zoom`→`static_zoom_in`,
  `track_side_left`→`track_truck_left`, `track_follow`→`track_look_at` ...),
  D75 의 `_aimed`/`_locked` 10종.
  **`zoom_in`/`zoom_out` 은 optical zoom 이 아니라 dolly 로 간다.**
- ⚠ **`dolly_in`/`dolly_out` 은 D75→D76 에서 뜻이 뒤집힌 유일한 두 이름이다.** D75 에서는
  "재조준 안 함"(`aim="traj"`)의 별칭이었고 지금은 정식 이름(`aim="look_at"`)이다. 디스크의
  옛 뱅크가 옛 뜻으로 이 문자열을 들고 있으므로 **뱅크 행을 읽는 코드는 `resolve_preset(name,
  variant["aim"])` 처럼 `aim` 을 같이 넘겨야 한다** — `LEGACY_AIM_COLLISIONS` 가 그때만
  `dolly_*_dont_look` 으로 되돌린다. 안 넘기면 옛 행이 조용히 반대 문구를 받는다.

### preset 에 직교하는 손잡이 (뱅크 축, `fit/bank/sample_camera_bank.py`)

| 손잡이 | 값 | 기본 |
|---|---|---|
| `--target_tau` / `--tau_ladder` | 궤적 크기 (τ = \|Δp\|/z_med) | `TAU_LADDER` |
| `--speeds` | `steady` \| `accel` \| `decel` \| `ease` (8배 조밀 궤적에서 index pick, 행렬 lerp 안 함) | `steady` |
| `--trackings` | `world` \| `drift` \| `lock` (look_at preset 에서만 유효) | `drift` |
| `--look_at_biases` | 조준점을 OBB 높이 비율만큼 위/아래로 | `0.0` |
| `--follow_gains` | subject 변위를 카메라 **위치**에 싣는 비율, `auto` 가능 (D72) | `0` |
| `--follow_smooths` | follow 채널 savgol 창, 1 = 끔 (D73) | `9` |
| `--aim_keyframes` | 0 = 매 프레임 조준, N≥2 면 keyframe N개만 조준하고 SO(3) 보간 (D71) | `0` |
| `--traj_basis` | `source` \| `subject` (D69) | `source` |

---

## LBM 원본 대비 변경점

| LBM 원본 | LBM-Lite | 이유 |
|---|---|---|
| Blender `.blend` 씬 렌더 | **Vista4D 4D 점군 splatting** (`render_frame`) | 입력이 씬 자산이 아니라 dynamic video 1편 |
| 실행당 ~20,000 렌더 | **~150 렌더** | 렌더 수는 VLM 호출 수와 1:1 이라 board 크기로 통제 |
| Director→Cinematographer→VideoEngineer 3단 + story/layout JSON 5종 | **단일 루프** `select → micro → traj` | 없는 입력 5종을 합성하지 않기 위해 |
| 렌더 원본만 VLM 에 전달 | **OBB 투영 + rule-of-thirds 오버레이 + 소스 프레임 패널 동봉** | VLM 이 subject/구도를 추측하지 않게 |
| micro-adjust 16 ops | **14 ops** (+`--allow_zoom` 이면 16) | `emit_model_cams.py` 에 intrinsics 채널이 없어 optical zoom 은 emit 을 못 통과 |
| trajectory preset 을 텍스트로 선택 | preset 시작/중간/끝 3프레임을 **미리 렌더**해 board 로 선택 | look-before-move 를 궤적에도 적용 |
| — | **SDF / 앵커 Ψ / Adam 최적화 삭제** | 렌더 게이트가 같은 일을 더 정직하게 한다 |

### 공간 제약을 집행하는 게이트 (`lbm/gates.py`)

| # | 게이트 | 판정 |
|---|---|---|
| G1 | behind-surface | 후보 중심을 소스 7프레임에 투영, `z_cam > depth + 0.02·S` 인 프레임이 있으면 기각 |
| G2 | render coverage | `valid_mask.mean() ≥ 0.55` |
| G3 | framing | subject bbox 중심이 중앙 80% 안 ∧ 면적비 ∈ [0.03, 0.50] ∧ z-buffer 통과율 ≥ 0.4 |
| G4 | τ / view-angle 프리필터 | `τ ≤ 0.30` ∧ `∠(v_plan, v_src) ≤ 40°` (G2 보다 1000배 싸다) |

전 후보의 통과/탈락 사유가 `board/gates.csv` 에 남는다. **전부 탈락하면 그게 진단**이다.

---

## 지금 켜져 있지 않은 것 (읽고 놀라지 않게)

- **`select` / `micro` 턴은 기본값에서 안 돈다.** `--start_mode source_frame0` 이라
  loop 가 `{'select': 'skipped_source_frame0', 'micro': 'skipped_source_frame0',
  'traj': 'vlm'}` 를 찍는다. 사용자 결정("그냥 첫 카메라를 그대로 쓰는걸로 단순화하자")이고,
  따라서 **`board_candidates.png` 는 오늘 진단/프리뷰 산출물**이지 VLM 입력이 아니다.
  `--start_mode board` 로 되살릴 수 있다.
- **zoom 은 기본 off** (`--allow_zoom`). emit 에 intrinsics 채널이 없다.
- `--build_sdf` 자리는 남겨뒀지만 기본 off 이고 하류가 안 쓴다.

## VLM 이 실제로 무엇을 보는가 (측정됨 — `DECISIONS.md` D34/D35/D36)

- **traj(preset) 턴은 prior 다.** 9조건 × 2영상 × 3 draw = 54 draw 중 `orbit_left_arc` 가 아닌
  것은 5건. 이름 중립화(`M01..M13`)·행 셔플·temperature 1.0·숫자 반전·**가짜 magenta 칠하기**
  다섯 개 전부 못 흔들었다. 품질을 지키는 건 VLM 이 아니라 게이트다.
- **select 턴은 숫자 우세.** 다만 9칸 board / camel 에서는 그림 채널이 실재하고 국소적이다 —
  숫자를 끄고 상위 타일에 가짜 magenta 를 칠하면 픽이 5/5 → **0/5**, 같은 면적을 하위 타일에
  칠한 대조군은 4/5. reasoning 이 칠한 타일을 이름으로 짚는다.
- **씬에 따라 꺼진다.** avocado-slice 는 칠하기 전부터 전 타일 magenta 면적비가 0.22~0.37 이라
  0.60 을 얹어도 순위가 안 바뀐다. 구멍 많은 씬에서는 **숫자가 유일하게 작동하는 채널**이다 →
  contract 에서 `coverage` 열을 빼지 않는다.
- **board 는 9칸이 기본**(3×3, 640×360). 27칸(4352×818, 5.32:1)은 select 턴 100 draw 중 7개가
  `max_tokens` 에서 잘렸고(9칸 0/100) 숫자를 빼면 최상위 타일 픽이 0/5 로 무너진다.

재현: `eval/ablate_vlm_hole_perception.py --turn both`.

---

## 좌표·단위

```
world  = DA3 frame0 카메라 (cam_c2w[0]=I), OpenCV (X right / Y down / Z fwd)
S      = frame0 non-sky 픽셀의 ray-depth 평균.  1 u ≜ S DA3 units
         camel S=4.6713, avocado-slice S=4.6471 (repo 의 avg_scale / S_da3 / norm_scale 과 같은 게이지)
gravity: ground RANSAC → 실패 시 camera-up fallback.  **roll 은 world Y 가 아니라 중력축 g 기준 0**
graph frame G: up = e_z, e_x = frame0 forward 의 수평 성분, e_y = up × e_x  (T_gw/T_wg 를 JSON 에)
tau(f) = |p_plan(f) - p_src(f)| / z_med(f)
```

`canonical.json` 은 `rel c2w, OpenCV, frame0 anchor, unit scale`, 21 pose
(49 → 21 은 보간 없이 `np.rint(np.linspace(0,48,21))` index pick).
`emit_model_cams.py:424` 의 `assert rel21.shape == (21,4,4)` 를 통과한다.

---

## Phase 2 로드맵 — temporal camera plan

이번 phase 의 결정은 **keyframe 1개**다. `decision.json` 의 `keyframes` 를 처음부터 리스트로
둔 이유가 이것이다. Phase 2 는 VLM 이 `{target, time, composition, visibility, relation,
motion_preference?}` 항목의 keyframe 을 N개 내고 `decode/build_poses.py` 가 사이를 보간한다
(보간기는 이미 있다 — `camera_interpolate_utils.py:82 quaternion_slerp`,
`:112 sample_from_two_pose`, `traj.py:172 compose`).

**Phase 2 에서 새로 필요한 건 프롬프트와 보간기뿐이고, 렌더러·게이트·디코더는 그대로 쓴다.**
`composition`/`visibility` 는 이번 phase 의 게이트가 그대로 keyframe 제약이 되고,
`relation` 은 `scene_graph.json` 의 `near`/`supported_by` edge 를 참조한다.

미구현으로 남긴 선택지(사용자 프롬프트로 subject/시간구간을 고르는 stage 2 query grounding,
segmentation 제안기 교체 등)는 `DECISIONS.md` 에 조건과 함께 적혀 있다.
