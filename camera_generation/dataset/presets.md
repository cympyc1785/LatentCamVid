# Preset / option 표

`lbm/presets.py` 의 `PRESETS` **40종** 전량과, 그것을 코퍼스로 바꾸는 스크립트들의 손잡이.
D84(`out_dynpose/*/hole_bank_d84/bank.json`, 266편 / 변이 12,193) 실측값을 같이 붙였다.

---

## 1. Preset 표 (가능한 40종 전량)

- **slot** — `route_presets.py` 가 이 preset 을 불러 주는 칸. `—` 면 라우터가 부르지 않아
  `--presets` 로 직접 지정하지 않는 한 코퍼스에 못 들어온다.
- **aim** — `look_at` 매 프레임 anchor 재조준 / `traj` 궤적 자체 회전 사용.
- **fol** — `PRESET_FOLLOW` (subject 추종 gain). `track_*` 만 1.0.
- **zoom** — `needs_zoom`. `--allow_zoom` 없으면 `--presets` 기본값에서도 빠진다.
- **tgtless** — caption 에 `target:` 절을 안 붙임 (`caption_presets.json` `"targetless"`).
- **D84** — D84 코퍼스 변이 수 / 등장 영상 수.

| # | preset | slot | aim | fol | zoom | tgtless | axis | D84 변이 | D84 영상 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `dolly_in` | advance | look_at | — | — | — | dolly | 672 | 204 |
| 2 | `dolly_out` | recede | look_at | — | — | — | dolly | 642 | 200 |
| 3 | `truck_left` | lateral | traj | — | — | — | truck | 2 | 1 |
| 4 | `truck_right` | lateral | traj | — | — | — | truck | 0 | 0 |
| 5 | `pan_left` | rotate | traj | — | — | **Y** | pan | 412 | 128 |
| 6 | `pan_right` | rotate | traj | — | — | **Y** | pan | 336 | 104 |
| 7 | `pull_out_arc_left` | arc | look_at | — | — | — | arc | 0 | 0 |
| 8 | `pull_out_arc_right` | arc | look_at | — | — | — | arc | 2 | 1 |
| 9 | `orbit_left` | orbit (span≥120°) | look_at | — | — | — | orbit | 0 | 0 |
| 10 | `orbit_right` | orbit (span≥120°) | look_at | — | — | — | orbit | 0 | 0 |
| 11 | `s_curve` | orbit (span<120°) | look_at | — | — | — | s_curve | 688 | 214 |
| 12 | `crane_up` | vertical (ground_ransac) | look_at | — | — | — | crane | 312 | 94 |
| 13 | `pedestal_up` | vertical (`--vertical_fallback`) | traj | — | — | — | pedestal | 0 | 0 |
| 14 | `static_look_at` | static | look_at | — | — | — | static | 1 | 1 |
| 15 | `track_dolly_in` | advance (bonus) | look_at | 1.0 | — | — | track | 410 | 128 |
| 16 | `track_dolly_out` | recede (bonus) | look_at | 1.0 | — | — | track | 396 | 123 |
| 17 | `track_truck_left` | lateral | traj | 1.0 | — | — | track | 434 | 136 |
| 18 | `track_truck_right` | lateral | traj | 1.0 | — | — | track | 340 | 108 |
| 19 | `track_pull_out_arc_left` | arc | look_at | 1.0 | — | — | track | 340 | 108 |
| 20 | `track_pull_out_arc_right` | arc | look_at | 1.0 | — | — | track | 434 | 136 |
| 21 | `track_orbit_left` | orbit | look_at | 1.0 | — | — | track | 168 | 51 |
| 22 | `track_orbit_right` | orbit | look_at | 1.0 | — | — | track | 136 | 44 |
| 23 | `track_crane_up` | vertical (bonus) | look_at | 1.0 | — | — | track | 214 | 66 |
| 24 | `track_pedestal_up` | vertical (bonus, `--vertical_fallback`) | traj | 1.0 | — | — | track | 0 | 0 |
| 25 | `track_look_at` | static | look_at | 1.0 | — | — | track | 424 | 265 |
| 26 | `crane_down` | — | look_at | — | — | — | crane | 0 | 0 |
| 27 | `track_crane_down` | — | look_at | 1.0 | — | — | track | 0 | 0 |
| 28 | `pedestal_down` | — | traj | — | — | — | pedestal | 0 | 0 |
| 29 | `track_pedestal_down` | — | traj | 1.0 | — | — | track | 0 | 0 |
| 30 | `push_in_arc_left` | — | look_at | — | — | — | arc | 0 | 0 |
| 31 | `push_in_arc_right` | — | look_at | — | — | — | arc | 0 | 0 |
| 32 | `track_push_in_arc_left` | — | look_at | 1.0 | — | — | track | 0 | 0 |
| 33 | `track_push_in_arc_right` | — | look_at | 1.0 | — | — | track | 0 | 0 |
| 34 | `orbit_left_pedestal_up` | — | look_at | — | — | — | orbit | 0 | 0 |
| 35 | `dolly_in_dont_look` | — | traj | — | — | — | dolly | 0 | 0 |
| 36 | `dolly_out_dont_look` | — | traj | — | — | — | dolly | 0 | 0 |
| 37 | `static_hold` | — | free | — | — | — | static | 0 | 0 |
| 38 | `track_hold` | — | free | 1.0 | — | — | track | 0 | 0 |
| 39 | `static_zoom_in` | — | traj | — | **Y** | — | static | 0 | 0 |
| 40 | `pan_right_zoom_out` | — | traj | — | **Y** | **Y** | pan | 0 | 0 |

집계: 구현 **40** / caption 있음 **40** / 라우터 도달 **23** (+`--vertical_fallback` 시 **25**) /
D84 실현 **19**. `--presets` 기본값(`usable_presets`)은 `--allow_zoom` 없이 **38**, 있으면 **40**.
별칭 `PRESET_ALIASES` 38개는 위 이름으로 접힌다.

> **D94 — 14번/37번, 25번/38번은 이름이 서로 바뀐 것이다.** `hold` 는 "안 움직이고 재조준도
> 안 한다"는 뜻이어야 하므로 14번(옛 `static_hold`, aim=`look_at`)이 `static_look_at` 이 되고,
> 37번(옛 `static_hold_dont_look`, aim=`free`)이 `static_hold` 를 물려받았다. **track 쪽도
> 똑같이 맞바꾼다** — 25번(옛 `track_hold`, aim=`look_at`)이 `track_look_at`, 38번(옛
> `track_hold_dont_look`, aim=`free`)이 `track_hold` 다. 궤적 자체는 넷 다 `T.hold(I, n)` 으로
> 그대로다 — **바뀐 건 이름과 조준뿐**이다. 이제 어휘 전체가 한 규칙이다: 조준은 `_look_at`
> 으로만 표기하고, `hold` 는 어디서든 "안 움직이고 재조준도 안 한다"를 뜻한다.
> 디스크의 옛 뱅크는 다시 쓰지 않는다: `("static_hold","look_at")`/`("static_hold","traj")` /
> `("track_hold","look_at")`/`("track_hold","traj")` 가 `LEGACY_AIM_COLLISIONS` 에 있어 옛 행이
> `*_look_at` 으로 되돌아가고, `*_hold_dont_look` 은 `PRESET_ALIASES` 로 새 `*_hold` 를 가리킨다.
> 물량은 track 쪽이 크다 — 배포 d77 뱅크의 `track_hold` 행 3,282 개가 전부 `aim="look_at"` 이다.
> `dolly_in`/`dolly_out`(D76) 과 **정확히 같은 장치**이므로, 뱅크를 읽는 코드는 `aim` 을 반드시
> 같이 넘겨야 한다 (안 넘기면 preset 의 *현재* 기본값을 타서 조용히 다른 카메라가 나온다 —
> 실측 pose 재현 오차 6.574e-01).

### caption 문구 (`configs/caption_presets.json`, 40종 전량)

| preset | phrase (`the camera ...`) |
|---|---|
| `dolly_in` | dollies straight forward toward the subject |
| `dolly_out` | dollies straight back away from the subject |
| `dolly_in_dont_look` | dollies in along its own axis without re-aiming |
| `dolly_out_dont_look` | dollies back along its own axis without re-aiming |
| `truck_left` / `truck_right` | trucks to the left/right, sliding sideways past the subject |
| `pan_left` / `pan_right` | pans to the left/right across the scene |
| `pan_right_zoom_out` | pans to the right while zooming out |
| `pull_out_arc_left` / `_right` | arcs to the left/right while pulling back from the subject |
| `push_in_arc_left` / `_right` | arcs to the left/right while pushing in toward the subject |
| `orbit_left` / `orbit_right` | orbits to the left/right around the subject |
| `orbit_left_pedestal_up` | orbits to the left around the subject while rising |
| `s_curve` | weaves left then right around the subject |
| `crane_up` | cranes upward and over the subject, revealing the space around it |
| `crane_down` | cranes downward toward the subject, closing in from above |
| `pedestal_up` | rises straight up while staying on the subject |
| `pedestal_down` | drops straight down while staying on the subject |
| `static_look_at` | holds still on the subject |
| `static_hold` | holds completely locked off |
| `static_zoom_in` | holds still on the subject while zooming in |
| `track_look_at` | tracks the subject, keeping pace with it as it moves |
| `track_hold` | tracks the subject with the framing locked off |
| `track_dolly_in` / `_out` | tracks the subject while pushing in toward / pulling back from it |
| `track_truck_left` / `_right` | tracks alongside the subject while sliding to the left/right |
| `track_pull_out_arc_left` / `_right` | tracks the subject while arcing to the left/right and pulling back from it |
| `track_push_in_arc_left` / `_right` | tracks the subject while arcing to the left/right and pushing in toward it |
| `track_orbit_left` / `_right` | tracks the subject while orbiting to the left/right around it |
| `track_crane_up` / `_down` | tracks the subject while craning upward over / downward toward it |
| `track_pedestal_up` / `_down` | tracks the subject while rising straight up / dropping straight down |

### 이름 규칙 · 부호 · 크기 손잡이

```
[track_]<primitive>_<direction>[_<primitive2>_<direction2>][_dont_look]
track_       PRESET_FOLLOW = 1.0 (subject 추종)
_dont_look   같은 궤적, aim=traj (재조준 없음)
```
`rot_y(+)=오른쪽 yaw` · `arc(+)=카메라가 왼쪽으로 가며 오른쪽을 봄` · `truck(+)=오른쪽` ·
`pedestal(+)=위` · `dolly(+)=전진` · `tilt/rot_x(+)=위`.

`DEFAULT_SHAPE = {dolly_frac 0.35, lateral_frac 0.35, sweep_deg 45.0, pan_deg 20.0}`, `NUM_FRAMES 49`.
`PRESET_ZOOM_END = {static_zoom_in 1.15, pan_right_zoom_out 0.667}`.
`ROTATION_ONLY_PRESETS = [pan_left, pan_right, pan_right_zoom_out]` (+`--external_shapes` 중 이동 0인 것)
— 이동이 0이라 τ 로 크기를 못 정하므로 사다리를 **pan 각도**로 옮긴다.

### DataDoP `dd_*` (`--external_shapes`)

`PRESETS` 밖. `register_external()` 이 런타임 등록하며 전부 `aim=traj` · targetless
(`FREE_MOVING_PREFIX="dd_"`). D84 실측 **5,830 변이 / 188 shape**
(slot 별 변이: vertical 1,232 / arc 926 / recede 924 / lateral 922 / rotate 916 / advance 910).

---

## 2. Option 표

### 2-1. `fit/bank/route_presets.py` — 어떤 preset 이 코퍼스에 들어오는가

| 옵션 | 가능한 값 | 기본 | D84 | 효과 |
|---|---|---|---|---|
| `--num_anchors` | int | `1` | `2` | 영상당 anchor 수 (실측 1.94/편) |
| `--min_area_frac` | float | `0.01` | `0.01` | anchor 화면 점유 하한 |
| `--vertical_fallback` | flag | **off** | off | gravity 가 `ground_ransac` 이 아닐 때 vertical 칸을 `pedestal_up` 으로 채움. off 면 그 칸 자체가 사라진다 |
| `--track_mode` | `add` / `replace` / `off` | `add` | `add` | `add`: `TRACK_KEEP_SLOTS=(lateral,arc,orbit,static)` 는 항상 track, `TRACK_BONUS_SLOTS=(recede,advance,vertical)` 중 1개를 `hash(video/node)` 로 추가. `replace`: 전 칸 track. `off`: track 없음 |
| `--external_shapes` | path | `None` | DataDoP JSON | free-moving shape 등록 |
| `--num_external` | int | `4` | `4` | 붙일 `datadop:<slot>` 칸 수 |
| `--emit` | `table` / `args` / `presets` / `nodes` | `table` | `args` | `presets`/`nodes` 출력을 `sample_camera_bank.py --presets/--nodes` 로 넘긴다 |

슬롯 → preset (`away` = 소스 카메라 횡이동 반대쪽, `toward` = 같은 쪽):

| slot | preset | 조건 |
|---|---|---|
| recede | `dolly_out` | 항상 |
| advance | `dolly_in` | 항상 |
| lateral | `truck_{away}` | 항상 |
| rotate | `pan_{away}` | 항상 |
| arc | `pull_out_arc_{toward}` | 항상 |
| orbit | `orbit_{away}` / `s_curve` | `obs_az_span ≥ 120°` / 미만 |
| vertical | `crane_up` / `pedestal_up` / 칸 없음 | gravity `ground_ransac` / `--vertical_fallback` / 그 외 |
| static | `track_look_at` / `static_look_at` | track 여부 |
| datadop:{slot} | `dd_*` | `--external_shapes` 있을 때 `--num_external` 개 |

### 2-2. `fit/bank/sample_camera_bank.py` — τ 뱅크 (변이 축)

| 옵션 | 가능한 값 | 기본 | D84 |
|---|---|---|---|
| `--presets` | preset 이름 목록 | `None` = `usable_presets(allow_zoom)` (38 / zoom 포함 40) | 라우터 출력 |
| `--nodes` | node id 목록 | `None` = `--min_area_frac` 자동 | 라우터 출력 |
| `--tau_ladder` | float 목록 | `0.10 0.20 0.35 0.60 1.00` | `1.00` 1단만 |
| `--speeds` | `steady` `accel` `decel` `ease` | `steady` | `steady` (다양성 0) |
| `--trackings` | `world` `drift` `lock` | `drift` | `drift` (다양성 0) |
| `--look_at_biases` | float 목록 | `0.0` | `0.0` |
| `--follow_gains` | float 목록 / `preset` | `0` | preset 값 |
| `--follow_smooths` | int 목록 | `9` | `9` |
| `--allow_zoom` / `--no_allow_zoom` | flag | off | off |
| `--fixed_focal` / `--no_fixed_focal` | flag | off | **on** |
| `--fit_tau` / `--no_fit_tau` | flag | on | on |
| `--drop_saturated` / `--no_drop_saturated` | flag | on | on |
| `--start_mode` | `source_frame0` / `board` | `source_frame0` | `source_frame0` |
| `--aim_anchor` | `subject` / `source_frame0` | `subject` | `subject` |
| `--aim_ramp_frames` | int | `12` | `12` |
| `--traj_basis` | `source` / `subject` | `source` | `source` |
| `--aim_keyframes` | int 목록 | `0` (=매 프레임 조준) | `6` |
| `--keyframe_aim` | `auto` / `target` / `preset_rel` | `auto` | `auto` |
| `--keyframe_ease` | `smoothstep` / `linear` | `smoothstep` | `smoothstep` |
| `--orbit_span_frac` | float | `0.8` | `0.8` |
| `--min_sweep_deg` / `--pan_deg_at_max` | float | `15.0` / `60.0` | 동일 |
| `--sweep_deg` / `--pan_deg` | float | `0.0` (=`DEFAULT_SHAPE`) | 동일 |
| `--track_dynamic_only` | flag | **on** | on |
| `--subject_visible` | flag | on | on |
| `--measure_behind` / `--measure_obb` / `--measure_standoff` | flag | off / off / off | 동일 |
| `--min_subject_points` / `--min_subject_area` | int / float | `100` / `0.005` | 동일 |
| `--measure_frames` | int | `5` | `5` |
| `--center_box` | float | `0.80` | `0.80` |
| `--tile_width` / `--tile_height` | int | `640` / `360` | 동일 |
| `--num_samples` / `--seed` | int | `0`(전량) / `0` | 동일 |
| `--preview` / `--preview_max` / `--preview_columns` | flag / int / int | on / `24` / `6` | 동일 |

### 2-3. `fit/bank/fit_hole_ladder.py` — hole 사다리 (최종 크기 결정)

| 옵션 | 가능한 값 | 기본 | D84 |
|---|---|---|---|
| `--hole_ladder` | float 목록 | `0.10 0.20 0.35 0.50` | `0.20 0.35` (2단) |
| `--hole_mode` | `absolute` / `excess` | `excess` | `excess` (목표 = `hole_static + delta`) |
| `--cap_hole` | float | `0.0` (=사다리 맨 윗단) | `0.35` |
| `--tau_floor_src` / `--no_tau_floor_src` | flag | on (`tau_start + 0.02`) | on |
| `--static_rung` | flag | on | on |
| `--iterations` | int | `4` | `4` |
| `--shape_headroom` / `--shape_doublings` | float / int | `2.0` / `4` | 동일 |
| knob 범위 (상수 `KNOB_RANGE`) | `tau (0.02, 3.0)` · `pan_deg (2.0, 180.0)` | — | 동일 |
| `--collision_free` (G1 behind) | flag | on | on |
| `--max_behind_frac` / `--behind_src_frames` / `--behind_margin_frac` / `--behind_clear_frac` / `--behind_radius_px` | float/int | `0.0` / `13` / `0.02` / `0.10` / `2` | 동일 |
| `--obb_gate` (G5) / `--obb_clear_src_ratio` | flag / float | on / `0.3` | 동일 |
| `--elev_gate` (G6) / `--max_elev_deg` / `--elev_src_margin_deg` | flag / float | on / `45.0` / `10.0` | 동일 |
| `--ground_gate` (G6) / `--min_ground_clear_ratio` | flag / float | on / `0.2` | 동일 |
| `--approach_gate` (G7) / `--approach_src_ratio` | flag / float | on / `0.3` | 동일 |
| `--speed` / `--tracking` / `--look_at_bias` | 2-2 와 동일 어휘 (단일값) | `steady` / `drift` / `0.0` | 동일 |
| `--fixed_focal` | flag | off | **on** |
| `--bisect_frames` / `--verify_frames` | int | `5` / `13` (0=전량) | 동일 |
| `--anchors` / `--presets` | 목록 | `None` = τ 뱅크 전량 | 전량 |

게이트 순서 (`lbm/gates.py`): `G4_tau → G4_view_angle → G1_behind → G5_obb → G2_coverage →
G3_center → G3_area → G3_occlusion`. 상한: `max_tau 0.30` · `max_view_angle 40°` ·
`min_coverage 0.55` · `center_box 0.80` · subject 면적 `[0.03, 0.50]` · `min_occlusion_pass 0.40`.
어느 게이트가 크기를 멈췄는지는 변이의 `binding` 열
(`hole` / `collision` / `obb` / `elev` / `ground` / `approach` / `none`).

### 2-4. `fit/caption/build_bank_captions.py` — 문장 생성

| 옵션 | 가능한 값 | 기본 |
|---|---|---|
| `--prompt_fields` | `target` `motion` 조합 | `target,motion` |
| `--magnitude` / `--no_magnitude` | flag | **off** |
| `--presets` | caption json 경로 | `configs/caption_presets.json` |
| `--label_map` / `--external_shapes` | path | `""` |
| `--bank_dir` / `--out_name` | str | `hole_bank_k6` / `captions.json` |
| `--videos` | 목록 / `all` | `all` |
| `--dry_run` | flag | off |

출력 형식:
```
target: person. motion: the camera tracks the subject while pushing in toward it.
target: none.   motion: the camera moves downward while panning left.
```

`configs/caption_presets.json` 버킷 (`--magnitude` 켤 때만 문장에 반영):

| 버킷 | 경계 → 단어 |
|---|---|
| framing (subject 면적비) | `0.005` extreme wide / `0.02` wide / `0.06` medium wide / `0.15` medium / `0.35` medium close-up / `1.01` close-up |
| magnitude τ | `0.05` barely / `0.15` slightly / `0.45` steadily / `1.0` significantly / `1e9` dramatically |
| magnitude pan_deg | `5` / `15` / `35` / `75` / `1e9` (같은 단어) |

---

## 3. 알려진 구멍

| 구멍 | 실측 |
|---|---|
| 라우터가 안 부르는 preset 15종 (`push_in_arc_*`, `pedestal_*`, `crane_down`, `*_dont_look`, zoom 2종 등) | D84 변이 0 |
| `--vertical_fallback` off | anchor 540개 중 vertical 칸이 살아남은 건 251개 (46.5%) |
| `speed` / `tracking` 다양성 | 12,193 변이 전량 `steady` / `drift` |
| `pedestal_*` 문구와 `aim` 불일치 | 문구는 "staying on the subject" 인데 `aim=traj` (재조준 없음) |
| `aim=look_at` 인데 subject 0px | 346 변이 (7.2%), 266편 중 47편에 집중 |
| `pan_*` subject_in_frame | median 0.50 (46.7% 가 0.5 미만) vs `track_truck_*` median 1.000 |
