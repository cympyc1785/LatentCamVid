# Changelog (video_generation)

[Keep a Changelog](https://keepachangelog.com/) 규약. `camera_generation/latentcam/CHANGELOG.md` 와
별개다 — 이쪽은 vendored 비디오 생성 모델 6종을 돌리는 `tools/` 스크립트만 다룬다.
(`video_generation/tools` 는 `.gitignore:223` 로 git 추적 대상이 아니다. 이 파일은 로컬 기록용.)

## [Unreleased]

### Added
- **`CinemaTraj/scripts/trumans_first_pose_board.py`** — TRUMANS `.blend` 위에서 chunk 마다
  first-pose 후보 격자를 돌리는 headless Blender 워커. **VLM 없이** 게이트(가림 / clearance /
  씬 AABB·바닥 / 프레이밍 / 근접 하한)만 걸고 통과분의 카메라 파라미터 + 렌더 PNG 를
  `board.json` (`trumans_first_pose_board_v1`) 으로 저장한다.
  - **왜 필요했나**: Lite 는 DA3 좌표계를 쓰는데 frame0 이 항등이라는 것만 알 뿐 그 프레임이
    blender 씬 **어디에** 있는지는 모른다. LBM 처럼 chunk 별 first pose 샘플링이 먼저다.
  - blend 로드가 ~4 s 라 `--chunk_starts` 로 여러 chunk 를 **한 번의 기동**에서 처리한다.
  - `--max_render` 로 자를 때 `render_truncated` 를 기록한다 (조용한 상한 금지).
  - 실측(lens 18 / step3 / r{1.1,1.5,2.0,2.6} / 144칸): chunk f51 33장, chunk f600 40장 통과.
    probe 5.1~5.2 s, 렌더 ~1.1 s/장.
- **shot scale / facing 태깅** (게이트 아님, 후보마다 `board.json` 에):
  `height_frac`(화면 밖까지 포함한 투영 세로 / 화면 높이) → `shot` ∈ {extreme_wide … extreme_close},
  `facing_deg`/`facing_range_deg` → `facing` ∈ {front, front_3q, profile, back_3q, back} × 좌우
  접미사 `_l`/`_r`. 요약표에 chunk 별 분포를 찍는다 — "40장 확보"가 실은 뒤통수 40장인 걸
  숫자로 보기 위함 (실측: chunk f600 통과 40장이 back_3q_l 18 / back 15 / profile_l 5 /
  profile_r 1 / back_3q_r 1, **front 0**).
  - 정면 축은 하드코딩하지 않고 `resolve_forward_axis` 가 recording 전체의 **보행 구간**에서
    속도 가중 투표로 푼다. 일치도가 `--facing_min_agreement`(0.35) 미만이면 `resolved:false` +
    전부 `unknown` 으로 두고 경고를 남긴다.
  - **좌우 분리**: 사이각을 `acos` 크기(0~180)가 아니라 `atan2` 로 **부호까지**(-180~180) 받는다.
    크기만 재면 왼쪽 옆모습과 오른쪽 옆모습이 똑같이 `profile` 로 뭉친다. `+` = 카메라가
    subject 의 왼쪽. 새 필드 `facing_side_frac`(왼쪽이었던 프레임 비율).
    - 부호는 **median 이 아니라 다수결**이다. 정후면 근처에서 프레임 값이 `+178`/`-178` 을
      오가면 중앙값이 0(= 정면)으로 떨어진다. 그래서 크기는 `|각도|` median, 좌우는 부호
      다수결로 따로 구해 합친다 (실측 f51 `back` 후보: deg 167~174 인데 side_frac 0.67).
    - `front`/`back` 은 좌우가 정의되지 않아 접미사를 안 붙이고, 그 사이 밴드도 한쪽이
      `--facing_min_side_frac`(0.70)을 못 넘으면 뗀다 — 창 안에서 사람이 카메라 앞을
      가로지르면 좌우가 실제로 바뀐다 (실측 f51 `back_3q` 2장이 side_frac 0.67 로 접미사 없음).
    - 검증: 후보 43장의 `_l`/`_r` 을 리그 어깨 본(`CC_Base_{L,R}_Upperarm`)으로 만든 해부학적
      좌우와 대조해 **43/43 일치, flip 0**. 눈으로 본 판단은 두 번 다 틀렸다 — 부호 규약은
      렌더를 보지 말고 본 위치로 검증할 것.
  - **촬영 각도** `eye_angle_deg`/`eye_angle_range_deg` → `camera_angle` ∈ {low_angle,
    slight_low, eye_level, slight_high, high_angle, overhead}. 격자축 `elevation_deg` 의 재라벨이
    **아니다** — 그건 subject AABB 중심(가슴께) 기준이고 이건 **눈높이** 기준이라, 같은
    elevation 이어도 반경이 커지면 눈높이에 가까워진다. 실측으로 전 elevation 행이 밴드 2개에
    걸친다 (el10: `slight_low`→`eye_level`, el25: `eye_level`→`slight_high`,
    el45: `slight_high`→`high_angle`).
    - 눈높이는 키 비례가 아니라 **눈 메시(`--eye_mesh CC_Base_Eye`) AABB 중심**을 직접 읽는다.
      실측 eye/stature 가 0.843~0.907 로 **상수가 아니라서**(무릎 꿇기/보행에서 자세가 바뀐다)
      통상 근사 0.94 는 6~12 cm 높게 나오고, r=1.5 m 에서 약 4.6° = 밴드 하나가 밀린다.
      메시를 못 찾을 때만 0.94 근사로 떨어진다.
    - 이 격자(elevation 10/25/45, 전부 subject 중심 위쪽)에서는 `low_angle`(< −25°)과
      `overhead`(> 60°)가 **0장**이다. 올려다보는 샷을 원하면 음수 elevation 을 넣어야 한다.
- **`CinemaTraj/scripts/board_contact_sheet.py`** — `board.json` + 렌더 → chunk 별 contact sheet.
  `lbm/overlay.py` 의 `contact_sheet`/`label_tile` 재사용. `--sort_by crop|shot|facing`,
  기본은 crop_keep 오름차순 — 게이트 임계를 숫자로 주장하지 말고 **보고** 정하기 위한 배치다.
  - 함정 2개를 주석으로 박아뒀다. ① **cv2 가 `sys.path` 를 새 리스트 객체로 갈아끼운다** —
    `from sys import path` 로 미리 묶어두면 그 참조가 죽은 리스트라 `insert` 가 조용히 무시되고
    `ModuleNotFoundError: No module named 'lbm'` 이 난다. `import sys` 로 받을 것.
    ② TRUMANS 리그는 SMPL-X 가 아니라 Character Creator(`CC_Base_*`)고 **루트 본
    `CC_Base_BoneRoot` 는 전 프레임 회전 항등 / 위치 `[0,0,-0.946]` 고정**이라 정면을 못 준다
    (보행 일치도 0.09). Hip/Pelvis/Waist/Spine02/Head 는 전부 로컬 **+z 가 정면**, 일치도 +0.68.
- **`preset_render_demo.py --forward`** — `end_transform` 을 focus 방향으로 직접 놓는 push-in
  모드(`directed_end` + `FORWARD_ARMS` 7 arm). `--ladder` 가 lateral 방향을 키운 것과 달리
  **전진/후진/lateral 을 같은 크기로 대조**한다.
  - **왜 필요했나**: `build_trajectory_plan`(`video_runtime.py:244-296`)의 if/elif 는
    `orbit_*`/`pedestal_*`/`*_reveal`/`pan_*`/`s_curve`/`static_*` 만 다룬다.
    **`straight_ease`/`push_in_arc`/`pull_out_arc`/`truck_left`/`truck_right` 5개는 분기가 없어**
    `mid=lerp(start,end,0.5)` + `end=end_transform` 인 순수 직선 LERP 가 된다 — 방향이 100%
    입력이고 preset 이름은 `_trajectory_travel_limit` 상한만 준다. 그래서 `--ladder` 의
    `straight_ease ×8` arm 이 전진이 아니라 truck-right + pedestal-down 으로 나왔다
    (forward 성분 32.8%, focus 거리 1.913→1.836).
  - **실측 (7 arm, 49프레임, `trumans_pushin`)**: push_in_arc 요청 0.30/0.60/0.85 → 전부 그대로,
    요청 1.20 → **0.850 (cap 정확히)**, `straight_ease` 0.95 → 0.950 (cap 정확히),
    `pull_out_arc` −0.85 → 0.850. 가시성 가드는 **7 arm 전부 미발동**(`--ladder` 의 keyframe
    arm 은 전부 0.75 가 걸렸던 것과 대조).
  - **1.20→0.85 축소는 리포트에 흔적이 하나도 안 남는다** — `safety_report` 에
    `motion_qc`/`trajectory_visibility_report` 만 있고 `motion_speed_policy` 자체가 부재.
    F02(0.85)와 F03(1.20)의 plan 키프레임이 소수점 4자리까지 동일하다.
  - **순수 dolly-in 인데 회전 17.98°** 가 나오고 그게 전부 전반부에 몰린다(f25→f49 Δ 0.0001 rad).
    후보 자체 `start_transform.rotation_euler`(pitch 0.9435/yaw 1.9299) vs
    `_look_at_euler(start, focus)`(0.7590/1.5902) 차이 — 알려진 스냅을 push-in 경로에서 독립 재현.
  - 픽셀 `mean|Δ|` (F00 기준): push_in 0.60/0.85/1.20 → 11.3/16.0/16.0, straight_ease 0.95 → 17.5,
    **pull_out −0.85 → 26.7 (최대)**, truck_right 0.85 → 19.2.
- **`CinemaTraj/scripts/preset_render_demo.py` + `preset_render_report.py`** — LBM trajectory
  preset 이 **실제 렌더를 얼마나 바꾸는가**를 재는 통제 실험 한 쌍.
  - **왜 통제 실험이 필요한가**: 기존 변종 렌더(`expand_preset_variants.py`)는 `--vary_end` 로
    **끝점도 같이 갈랐다**. 클립이 달라 보여도 preset 때문인지 끝점 때문인지 못 가른다.
    `preset_render_demo.py` 는 `start_transform`/`end_transform`/`focus`/`lens` 를 전 arm
    공통으로 못박고 `trajectory_plan.preset_name` **하나만** 바꾼 8 arm 을 만든다.
  - 9번째 arm 은 preset 대신 우리가 만든 **49개 keyframe** 을 `trajectory_plan.keyframes` 에
    직접 싣는다 (`preset_name: "keyframed_external"` — `_trajectory_travel_limit` 이 모르는
    이름에 상한 1.4 m 를 준다). LBM 수정 0줄.
  - `preset_render_report.py` 는 camdump(`lbm_camera_dump_startup.py`)의 `matrix_world` 로
    arm 별 net/path/회전과 기준 arm 대비 `Δloc_max`/`Δang_max`, 프레임 픽셀 `mean|Δ|` 를 낸다.
    3×3 타일 영상(`imageio` + libx264)도 같이.
  - **실측 (`trumans_c49_w01_f0000_0048` cam1, 49프레임, EEVEE 960×540)**: preset 8종의
    net 변위 0.0000~0.2314 m — `pan_left`/`pan_right` 는 **정확히 0.0000**(travel limit 0.0),
    `orbit_left_arc` 0.0260. 기준(`straight_ease`) 대비 `Δloc_max` 0.068~0.203 m,
    픽셀 `mean|Δ|` 14.5~20.2 / 255. **preset 이 렌더를 바꾸긴 한다** — 다만 8종 전부가 서로
    같은 크기(0.07~0.20 m)로만 갈리고, 이름이 뜻하는 모양과는 무관하다.
    `build_trajectory_plan:248-249` 가 orbit 을 **±1.2°/±2.2°** 로 얹기 때문.
  - **새 실측 — 가시성 가드가 우리 keyframe 을 갈아치운다**: 49 keyframe arm 은 keyframe
    개수(49)와 rotation 은 살아남았지만 이동량이 **정확히 0.75배**로 줄었다
    (입력 net 1.0076 → 렌더 0.7557 m). 범인은 travel limit 이 아니라
    `blender_render_worker.py:1094` 의 `motion_scale` 사다리로,
    `safety_report.visibility_guard_motion_scale = 0.75` 에만 남는다.
    `motion_speed_policy` 는 `speed_limited: false` 라 **이 경로는 거기 안 찍힌다**.
    camdump 는 최종 plan 과 frame 1/25/49 전부 `Δloc 0.0000 / Δang 0.000°` 로 일치 —
    렌더는 정직하고 갈아치운 것은 plan 단계다.
  - **`--ladder` 모드 추가** — preset 8종 대신 **같은 원호를 크기만** 0.23→2.0 m 로 올린 8칸 +
    preset 경로에서 `end_transform` 만 ×8 로 민 대조군 1칸. 이동량 천장을 직접 잰다.
    `preset_render_report.py` 에 `req_m`/`cap_m`/`guard` 열을 붙여 요청 대비 실제를 나란히 낸다.
  - **실측 — 실효 상한은 1.05 m (= cap 1.4 × guard 0.75)**: 요청 0.23/0.4/0.6/0.8/1.0/1.2/1.4 m 가
    전부 **정확히 0.75배**(0.1725/0.300/0.450/0.600/0.750/0.900/1.050)로 나왔다. 요청 2.0 m 는
    먼저 travel limit 1.4 로 잘리고 다시 0.75 가 곱해져 **1.4 m 요청과 같은 1.050 m** 로 수렴한다.
    가드는 크기와 무관하게 **최소 칸(0.23 m)에도 똑같이 0.75 로 걸렸다** — 즉 이 가드는 "너무 크다"가
    아니라 우리 orbit keyframe 의 조준 자체를 실패로 본다. preset 경로 arm 은 9개 중 한 번도 안 걸렸다.
  - **preset 경로 대조군**: `straight_ease` + `end_transform ×8`(요청 1.851 m)은 가드 없이
    travel limit 0.95 에 정확히 물려 net **0.9500 m**. 회전 35.47°, 기준 대비 픽셀 `mean|Δ|`
    35.4/255 로 사다리 전 칸보다 화면이 크게 바뀐다.
  - **정리 — 이동량을 정하는 것 3개**: ① 크기 입력은 `end_transform.location`(preset 경로) 또는
    `trajectory_plan.keyframes`(직접 경로) ② `_trajectory_travel_limit:304` 의 **preset 이름별 상한**
    (pan/static 0.0, pedestal 0.35, reveal 0.42, orbit 0.8, s_curve/push/pull 0.85,
    straight_ease 0.95, truck 1.05, 미등록 이름 1.4; closeup 이면 `min(limit, 0.25)`)
    ③ 가시성 가드 0.75/0.5/0.35/0.2. **preset 이름은 늘리지 못하고 자르기만 한다.**
    1.05 m 를 넘기려면 LBM 수정이 필요하다 (limit dict 의 기본값 1.4 또는 가드 사다리).
  - **실행에 필요한 env 2개** (원본 manifest 의 command 에는 안 적혀 있다):
    `LBM_SINGLE_SCENE_FALLBACK=1` (TRUMANS `.blend` 씬 이름이 `Scene` 이라 `_resolve_scene:67` 의
    `Scene_1_Shot_1` 후보에 안 걸린다 → 전 arm `scene_not_found_for_scene_1_shot_1`) 과
    `PYTHONPATH=VideoEngineer` (worker 가 `video_runtime` 을 형제 모듈로 import).
- **`CinemaTraj/scripts/repair_recon_poses.py`** — work dir 의 `poses_aNN.npz` 를 배포본
  `cameras.npz` 에서 되돌린다. **기본은 읽기 전용**이고 `--apply` 를 붙여야 쓴다. 손상본은
  지우지 않고 `poses_aNN.npz.overwritten` 으로 남기며, 백업이 이미 있으면 덮지 않는다
  (두 번 돌려도 최초 원본이 살아남는다).
  - **왜**: `trumans_lite_bank.py:208-210` 이 `--work` 를 `--video_suffix` 가 있을 때만 하위
    `trumans_to_recon.py` 로 넘긴다. suffix 없이 다른 `--work` 를 주면 그 인자는 **조용히 무시되고**
    하위가 제 기본값(`out/trumans_recon`)에 쓴다. 18 mm 렌즈 파일럿이 정확히 이 사고로 25 mm
    work dir 의 `00add26c` a00 을 덮어썼다.
  - **비교는 앵커를 벗기고 한다**: 배포본은 재앵커된 것(`frame0 = I`)이고 work 의 poses 는 재앵커
    전 Blender world 라(`audit_lite_framing.py:222`) raw 끼리 비교하면 **전량이 어긋난 것처럼 보인다**.
    `inv(work[0]) @ work` 로 벗겨서 비교해야 한다. 뱅크 130 클립 스캔 결과 **DRIFT 1 / ok 129**
    (어긋난 편 0.759568, 나머지 전부 0.000000) — 피해는 1 클립에 갇혀 있었다.
  - **앵커 재구성**: 배포본이 앵커를 이미 벗겨버려 거기엔 없다. 무사한 `manifest_aNN.json` 의
    `candidate.position` + probe 에서 되짚은 `aim[0]` 으로 `look_at_c2w` 를 다시 세운다.
    npz 의 `aim` 은 못 쓴다 — 손상 실행이 같이 덮었고 그 사이 `--aim_bias` 기본값이 0.0 → 0.20,
    `anchor_origin` 기본이 `chest` → `obb_center` 로 바뀌었다. 그 `aim` 으로 앵커를 세우면
    `obb_area` 가 2.8% 어긋난다 (0.4996 → 0.4857).
  - **원본 aim 규약을 역산**: 온전한 클립 a01~a05 에서 `aim` 을 0.0000 으로 재현하는 조합은
    `chest`(=`body_points[0]`) / `smooth_window 11` / `aim_bias 0.0` 뿐이다
    (`--aim_base` / `--aim_smooth_window` / `--aim_bias` 로 노출). `human_track` 은 코드가 바뀌어도
    비트 단위로 같아서(probe 재실행 diff `0.000e+00` 실측) 이 재구성이 정확하다 —
    `look_at_c2w(재구성 aim[0])` 가 `work[0]` 을 **≤3.3e-16** 으로 재현한다.
  - `--force` 는 앵커만 틀린 복구본을 다시 덮어쓰기 위한 것이다. 비교량 `delta` 는 **상대** pose 만
    보므로 앵커가 틀려도 `ok` 로 나온다.
  - **검증**: 고친 앵커로 감사를 다시 돌리면 뱅크 저장 행과 **비트 단위로 일치**한다
    (`framing 0.9636 / crop 0.8119 / occl 0.9636 / obb_area 0.4996`). 대조군으로 손 안 댄
    a01·a02 를 재감사해 저장 행이 그대로 재현되는 것도 확인했다 (감사가 결정론적이라는 근거).

- **TRUMANS Lite 뱅크에 `--lens` passthrough** (`CinemaTraj/scripts/trumans_lite_bank.py`)
  — `clip_flags()` 가 `trumans_to_recon.py --lens` 로 넘긴다. **기본 25.0 은 기존 뱅크와 동일**이라
  값을 안 주면 기존 경로가 그대로 돈다.
  - **왜**: 기존 뱅크 130 클립의 framing 이 깨져 있다 — `person_fill_v`(사람 키/프레임 높이)
    p50 1.119, **96 클립 중 62 편이 >1.0 = 사람이 잘린다**. `crop_mean` p50 0.786,
    95 편 중 82 편이 <0.9.
  - **원인은 반경**: 반경 분포가 `{1.0:2, 1.5:87, 2.2:25, 3.0:10, 4.0:6}` = **68.5% 가 1.5 m**.
    `--radius_strata`(기본 on)가 채택률을 76% → 68.5% 로 낮췄을 뿐 못 고쳤다. 격자 통과율이
    반경에 따라 급락하기 때문이다 (1.5 m 50.3% / 2.2 m 23.2% / 3.0 m 8.4% / 4.0 m 2.3%).
  - **그런데 반경은 손잡이가 아니다**: 통과율은 clearance 가 정한 물리적 한계이고, 반경을 밀면
    가림이 같이 죽는다 — `occl_keep` p50 1.5 m 0.983 / 2.2 m 0.913 / 3.0 m 0.892 / **4.0 m 0.625**.
  - **렌즈는 clearance 를 안 건드린다**. `crop_keep` 은 OBB 코너 투영 + 이미지 클리핑뿐이라
    순수 기하라서, K 의 fx·fy 만 스케일해 **렌더 없이** 130 클립 전량을 실측했다:

    | lens | crop p50 | crop<0.9 | fill p50 | fill>1.0 | OBB 면적비 p50 |
    |---|---|---|---|---|---|
    | 25 mm (현재) | 0.777 | 110/130 | 1.339 | 108/130 | 0.478 |
    | 22 mm | 0.840 | 84/130 | 1.179 | 96/130 | 0.370 |
    | 20 mm | 0.884 | 71/130 | 1.071 | 85/130 | 0.306 |
    | **18 mm** | **0.932** | **46/130** | **0.964** | **58/130** | **0.248** |
    | 16 mm | 0.976 | 30/130 | 0.857 | 34/130 | 0.196 |
    | 14 mm | 0.998 | 22/130 | 0.750 | 21/130 | 0.150 |

    지배적인 r=1.5 버킷(87 편)의 `crop_keep` p50 이 25mm 0.750 → 18mm 0.911 → 16mm 0.956.
  - **아직 안 잰 것**: `occl_keep` 은 렌더된 depth 가 필요해서(25 mm 화각 밖은 렌더가 없다) 이
    스윕으로 못 센다. 화각이 넓어지면 전에 잘려 있던 다리·팔이 프레임에 들어오는데 그 부분이
    가구에 가려질 수 있다. 고른 렌즈로 부분집합을 **실제 재렌더**해서 확인해야 한다.

- **LBM Cinematographer 후보 방위 스윕** (`Look-Before-Move/Cinematographer/cinematographer_quality_worker.py`)
  — 환경변수 `LBM_CAND_AZIMUTH_SWEEP=1` 로만 켜지고, **끄면 기존 경로가 그대로 돈다**.
  - **왜**: 기본 후보 격자는 방위가 한 곳에 몰려 있다. `direction_seeds` (`generate_candidates`)
    는 8방향 × **반경 1개 × 높이 1개** = 8개뿐이고, `preset_transforms` 의 Monte Carlo 50개는
    반경(0.5~2.5배)·고도(−0.3~0.6)·lens 를 훑지만 **방위가 `requested_direction(camera)` 하나로
    고정**돼 있다. 즉 depth-0 seed 약 58개 중 50개가 Director 가 요청한 단일 방위 위에 놓인다.
    확장 BFS 도 부모를 **전역 `final_score`** 로 골라 그 방위를 더 파고든다.
  - **켜면 바뀌는 것**: ① `direction_seeds` 를 8방향 × `LBM_CAND_RADII` × `LBM_CAND_ELEVATIONS`
    격자로 (기본 `0.7,1.0,1.6` × `0.0,0.35,0.8` = 8→72; `(1.0, 0.0)` 조합이 기존 seed 와 동일 pose 라
    격자가 기존을 포함한다) ② MC seed 의 방위를 8방향에 고르게 분배 (개수 `LBM_CAND_MC_SAMPLES`
    기본 50 그대로 — 렌더 비용 증가 없음) ③ 확장 BFS 부모를 방위별 round-robin 으로 선택
    (부모 개수 동일) ④ MC 샘플러에 `LBM_CAND_SEED` (기본 0) 을 박는다.
    `LBM_CAND_DIRECTION_GATE=soft` 는 별도 플래그로, 요청 방위와 그 이웃만 통과시키는 하드 게이트
    (`candidate_eligible` + `score_candidate` 의 0.05 상한)를 빼고 점수항(가중치 0.04)만 남긴다.
  - **원래 후보 풀이 실행마다 달랐다** (`preset_transforms` 가 모듈 전역 `random` 을 시드 없이 사용).
    w09 chunk 를 플래그 없이 5회 반복한 실측 — 같은 입력, 같은 코드:

    | 실행 | 카메라 | raw | eligible | retained | 최고 final |
    |---|---|---|---|---|---|
    | 전량실행 원본 | **0** | 1184 | 0 | 0 | 0.000 |
    | 반복 2 | 1 | 1204 | 310 | 20 | 0.468 |
    | 반복 3 | 1 | 1200 | 302 | 12 | 0.474 |
    | 반복 4 | 1 | 1193 | 393 | 20 | 0.662 |
    | 반복 5 | **0** | 1194 | 0 | 0 | 0.000 |

    eligible 이 0 아니면 300+ 로 갈리는 **양봉 분포**다. depth-0 seed 는 전부 벽에 막혀 있고,
    MC 50 draw 중 하나가 우연히 뚫린 자리에 떨어지면 그 하나에서 16연산 × 5 depth 확장이
    수백 개를 낳는다. 안 떨어지면 1190개가 통째로 기각된다. 앞서 "37 chunk 중 12개(32%)가
    카메라 0대"로 기록한 수치는 **그 chunk 들의 성질이 아니라 draw 1회의 결과**다.
  - 스윕(시드 고정)은 3회 반복 모두 동일했다 — `raw 1329 / eligible 32 / retained 9 /
    최고 final 0.584`, 선택 후보까지 같음. `soft` 게이트는 w09 에서 차이가 없었다
    (선택된 `front_right` 가 요청 방위의 이웃이라 하드 게이트에 애초에 안 걸린다).
  - **비용과 대가**: 실행 시간 61 s → 63 s (chunk 당), raw 후보 ~1190 → ~1330 (+12%).
    다만 방위를 넓히면 한 방위당 확장 깊이가 얕아져 **eligible/retained 는 줄어든다**
    (w09 300+/20 → 32/9, w01 564/20 → 219/20). w01 은 최고 final 이 0.678 → 0.657 로 소폭 하락.
    전 chunk 수율 분포는 측정 중.
- **`CinemaTraj/scripts/expand_preset_variants.py`** — LBM Cinematographer 카메라 1대를
  **원본 + 변형 N대**로 불려 chunk 당 여러 궤적이 나오게 한다. 기본 `--dry_run`, 고칠 땐
  `--no_dry_run` (원본은 `.variants.bak`). `preset_variant_source` 로 재확장을 막아 멱등.
  **`retime_camera_handoff.py` 다음에** 돌린다 — 복제본이 원본의 `target_frame_count` 를 물려받는다.
  - **왜**: Director 가 chunk 당 `camera_count: 1` 을 내고 Cinematographer 의 VLM 이 preset 하나만
    고른다. 궤적 데이터셋으로 쓰려면 같은 chunk 에서 여러 궤적이 나와야 한다.
  - **어떻게 LBM 자신의 궤적 합성기를 다시 부르나**: `_build_plan_from_explicit_trajectory`
    (`VideoEngineer/blender_render_worker.py:922-`) 는 `trajectory_plan.keyframes` 가 **비어 있으면**
    `build_trajectory_plan(..., preset_override=preset_name)` 을 불러 preset 모양대로 keyframe 을
    직접 만든다. 복제본에 `{"preset_name": <preset>, "keyframes": []}` 만 심으면 궤적 계산은 LBM
    코드가 한다. 원본 카메라는 VLM keyframe 을 그대로 들고 있어 손대지 않는다.
  - **preset pool 이 17개가 아니라 8개인 이유**: `video_runtime.PRESET_NAMES` 는 17개인데
    `build_trajectory_plan` (`video_runtime.py:244-296`) 이 이름을 실제로 분기하는 건 orbit 2 /
    pedestal 2 (`rise_reveal`·`drop_reveal` 은 같은 branch) / pan 2 / `s_curve` / static 3 뿐이다.
    `straight_ease` `push_in_arc` `pull_out_arc` `truck_left` `truck_right` **5개는 branch 가 없어**
    전부 기본 lerp 로 떨어진다 — 이름만 다르고 궤적이 같다. `_trajectory_travel_limit`
    (`blender_render_worker.py:310-323`) 이 preset 별로 0.35~1.05 로 다르지만 `original_travel >
    limit` 일 때만 깎는데 실측 travel 이 median 0.223 m 라 대부분 안 걸린다. static 3종은 기본
    제외(`--include_static`) — 이름만 보고 뽑으면 조용히 정지 클립이 섞인다.
  - **`--vary_end` (기본 on) — preset 만 바꾸면 변형끼리 궤적이 거의 같다.** preset 섭동이 모든
    변형이 공유하는 `start→end` lerp 보다 한 자릿수 작다 (25 chunk 실측):

    | 요소 | 크기 |
    |---|---|
    | 공유 lerp travel | median **0.223 m** (max 0.805) |
    | `orbit_*_arc` | mid **1.2°**, end **2.2°** |
    | `pedestal_up/down` | `_damped_extent(focus_extent.z, 0.05, 0.035, 0.1)` = **3.5~10 cm** |
    | `s_curve` 횡방향 | `scale 0.015, min 0.025, max 0.05` = **2.5~5 cm** |
    | `pan_left/right` | **위치 고정** (`_smooth_executable_keyframes:328-366`), 회전만 |

    진짜 손잡이는 끝점이고 그건 이미 handoff 안에 있다 — `top_candidates` 는 LBM 자신의
    가시성·프레이밍 스코어를 통과한 pose 10~20개로, start 에서 median 0.44 m / 최대 0.90 m 떨어져
    있고 방향 최대각 median 133°. `end_transform` 은 `blender_render_worker.py:274` 가 그대로 읽어
    `build_trajectory_plan` 에 넘기므로 JSON 만 고치면 된다. **start pose 는 전 변형 공통으로
    고정**한다 — 같은 초기 조건에서 다른 움직임이 나와야 궤적 표본으로 비교가 된다.
    `STORYBLENDER_TRAJECTORY_SCALE` 은 `cinematographer_stage.py:52` 에서만 읽혀 Stage 3 에 안 먹는다.
  - **끝점 후보를 같은 `lens_mm` 으로 제한**한다: worker 는 두 keyframe 에 같은 lens 를 쓴다
    (`:897-898`). 45mm 로 스코어된 위치에 24mm 를 얹으면 피사체가 작아져 가시성 검사에 걸리고,
    worker 가 `motion_scale` 0.75/0.5/0.35/0.2 사다리로 구제하면 (`:1076-1120`) 조용히 정지 클립이
    된다. 같은 lens 후보가 모자란 chunk(망원 45~77mm, 25개 중 6개)는 다른 lens 로 채우고
    `end_variant_tier: "relaxed"`, 그것도 없으면 `"authored"` 로 표시한다 — 요약표에 tier 별
    개수가 찍히므로 어느 변형이 preset 모양만 다른지 바로 본다. start 에서 `--min_travel_m`(0.08)
    이내인 후보는 뺀다 (그건 정지 궤적이다).
  - `--no_vary_end` 로 preset 만 바꾸던 기존 동작을 그대로 쓸 수 있다.
  - **Stage 3 는 `shots[].cameras` 만 읽는다** (`blender_render_worker.py:1208,1227`). 최상위
    `cameras` 리스트도 같이 불리되 렌더에 반영되는 건 전자다.
- **`CinemaTraj/scripts/retime_camera_handoff.py`** — LBM Cinematographer 의 `camera_handoff_v1.json`
  을 chunk 프레임 수(49)에 맞춰 다시 매긴다. 기본 `--dry_run`, 고칠 땐 `--no_dry_run` (원본은
  `.bak`). 멱등 — 여러 번 돌려도 결과가 같다.
  - **왜**: Cinematographer 의 `refresh_camera_trajectory`
    (`cinematographer_stage.py:2820-2828`) 가 duration 을 **preset 에서** 다시 계산하고 Director 의
    `duration_target_seconds` 를 버린다. TRUMANS 49프레임 chunk 3개 실측에서
    `target_frame_count` 가 **110 / 80 / 95** 로 나왔다 (Director 값 65 / 45 / 65 도 아니다).
  - 그 값이 Stage 3 로 그대로 간다 (`VideoEngineer/blender_render_worker.py:1030`). 그리고 워커가
    렌더 직전에 `scene.frame_start = 1; scene.frame_end = frame_count` (`:1141-1142`) 로 frame
    shift 훅의 `scene.frame_end = 49` (`_frame_shift/startup/trumans_frame_shift_startup.py:101`)
    를 **덮어쓴다** — 훅은 Blender 기동 시 한 번 돌고 워커는 그 뒤에 세팅하므로 워커가 이긴다.
    결과적으로 w01 은 chunk 49프레임 + **다음 chunk 영역 61프레임**을 이어서 렌더한다. 클램프도
    루프도 아니고 chunk 밖 애니메이션이 그냥 섞인다.
  - **`target_frame_count` 만 고치면 안 된다**: 궤적 keyframe 이 절대 프레임 인덱스로 박혀 있다
    (`trajectory_plan.keyframes[].frame` = 1 / 56 / 110). frame_count 만 49 로 낮추면
    `_build_plan_from_explicit_trajectory` 의 `frame_number = max(1, min(frame_number, frame_count))`
    가 1 / 49 / 49 로 클램프해서 **궤적 앞 절반만 재생하고 나머지는 정지**한다.
  - 다시 매기는 것: `target_frame_count` / `target_duration_seconds` /
    `trajectory_keyframes[].frame` (0-based, 정규화 `t` 동봉 → `round(t·48)`) /
    `trajectory_plan.keyframes[].frame` (1-based, plan span 으로 정규화 → `round(t·48)+1`) /
    `trajectory_plan.{start_frame,end_frame,duration_seconds}` /
    `trajectory_plan.safety_report.duration_seconds` / `trajectory_safety_report.duration_seconds`.
    카메라가 `/shots[i]/cameras[j]` 와 `/cameras[k]` 두 군데에 **복사본**으로 있어 둘 다 훑는다.
  - **안 건드리는 것**: `shot_contract.motion_contract.{start_frame,end_frame}` 은 Director 계약
    프레임 공간이고 VideoEngineer 가 안 읽는다 (`grep motion_contract VideoEngineer/*.py` → 0건).
    Cinematographer 내부 중간산출물인 `camera_handoff_{preview,quality}_input_v1.json` 도 하류가
    안 읽으므로 원본 대조용으로 남긴다.
  - **궤적을 자르지 않고 시간을 압축한 이유**: 자르면(원래 속도 유지, 49프레임에서 끊기)
    `orbit_right_arc` 이 sweep 의 45% 만 돌고 끝나 preset 을 고른 이유가 사라진다. 압축은 authored
    구도를 통째로 유지하고 속도만 `old_fc/new_fc` 배 빨라지는데, 실측상 문제될 수준이 아니다:

    | run | preset | frames | travel | m/s before | m/s after |
    |---|---|---|---|---|---|
    | `w01_f0000_0048` | orbit_right_arc | 110 → 49 | 0.231 m | 0.053 | 0.118 |
    | `w22_f0504_0552` | push_in_arc | 80 → 49 | 0.160 m | 0.050 | 0.082 |
    | `w39_f0912_0960` | pedestal_up | 95 → 49 | 0.351 m | 0.092 | 0.179 |

    사람 보행이 ~1.4 m/s 다. 요약표에 `m/s after` 열이 매번 찍히므로 전량에서 튀는 chunk 를 바로
    본다. `_smooth_executable_keyframes` 의 travel limit 은 거리 기준이라 프레임 수와 무관하다.
  - keyframe 수가 프레임 수보다 많아 인덱스가 뭉치면 요약표에 `!! keyframe 뭉침` 으로 찍는다
    (실측 3 chunk 는 keyframe 2~3개라 0건).
- **`CinemaTraj/scripts/sanitize_director_focus.py`** — LBM Director 산출물의 focus id 목록에서
  **유령 asset** 을 뺀다. 기본 `--dry_run`, 고칠 땐 `--no_dry_run` (원본은 `.bak`).
  - **왜**: Director 는 렌더 이미지를 보고 `focus_ids` 를 쓰는데 그게 `asset_index` 안에 있는지
    **아무도 검사하지 않는다**. TRUMANS 86 chunk 실측에서 `sofa`(12) `coffee_table`(10)
    `chair`(7) `table`(6) 등 없는 id 가 **59회** 섞였다 (28 shot, 전부 secondary —
    primary 는 86/86 `zzy3`).
  - 대부분은 무해하다. Cinematographer 의 `_find_object`
    (`cinematographer_preview_worker.py:103-123`) 가 blend 오브젝트 이름 완전일치 / `<id>.` /
    `<id>_` 접두사로만 찾고 못 찾으면 `continue` 한다. 432 오브젝트에 대조하니 51/59 는 해석
    자체가 안 되고, 6/59 는 의도한 asset 으로 붙는다 (`book`→`book_left_01`,
    `whiteboard`→`whiteboard_01`, `oven_door`→`oven_door_01`).
  - **유해한 건 3개**: `window`(w19·w38) 과 `floor`(w39) — asset 이 아닌데 blend 에는 실제로
    있어서 focus AABB 에 그대로 합쳐진다. 실측 (00add26c w01 blend):

    | focus | extent | maxdim |
    |---|---|---|
    | `zzy3` | (0.84, 1.76, 2.22) | 2.22 m |
    | `zzy3 + window` | (3.52, 1.76, 3.29) | 3.52 m (×1.6) |
    | `zzy3 + floor` | (1.88, 6.61, 2.22) | **6.61 m (×3.0)** |

    카메라가 그만큼 물러나 사람이 작아진다 — 실패가 아니라 **조용히 나빠지는** 종류라 로그에
    안 남는다.
  - 판정 규칙(`is_ghost`): asset_index 에 있으면 둔다 / blend 에서 해석 안 되면 둔다(무해) /
    해석되는데 그 이름이 asset id 를 부분문자열로도 안 가지면 **뺀다**. 마지막 예외가 없으면
    770개를 빼는데 그중 560개가 제대로 붙은 book/whiteboard/oven_door 다.
  - 스칼라 `primary_focus_id` 는 **안 지운다** (비우면 하류가 primary 없이 돈다). 세어서
    요약표에 `!! 스칼라 primary` 로 찍고 사람에게 넘긴다 — 86 chunk 실측에서는 0건.
  - 적용 결과: `trumans_c49_w{19,38,39}` 의 4파일씩 총 **12파일**, `window` 140 + `floor` 70 =
    **210개** 제거. 키는 `focus_ids` / `secondary_focus_ids` / `start_focus_ids` /
    `contract_focus_ids` 4종에 걸쳐 있다 (`blocking_plans_v1` / `contract_blocking_plans_v1` /
    `director_handoff_v1` / `shot_contracts_v1`).
- **`CinemaTraj/scripts/trumans_to_lbm_demo.py --chunk_mode slide`** (기본 `window` = 기존과
  바이트 동일) — 창 경계를 무시하고 영상 전체를 `--chunk_frames 49` 창으로
  `--chunk_stride 24` 씩 훑어 고정 길이 chunk 를 만든다. 새 함수 `build_chunks()` /
  `chunk_narrative()` / `strip_subject()` / `rewrite_subject()`.
  - **왜 필요했나 (실측, `2023-01-17@00-55-00`)**: LBM-Lite 는 클립 길이가 **49프레임 고정**
    (`lbm/presets.py:51 NUM_FRAMES = 49`, 하류 emit 도 21-pose 고정) 인데 VLM action tagging 이
    낸 창 37개의 길이는 min 17 / med 43 / max 233 이라 **21개가 49를 못 채운다**. 창 경계 안에서만
    자르면 클립 24개만 남고 창 21개(그중 action 14개, 오븐 여닫는 w21/w23 포함)가 통째로 빠진다.
  - chunk 서술 조립: 겹치는 창들을 **시간순**으로 잇되, 겹침이 가장 큰 창만
    `shot_description`(위치·자세 포함) 을 쓰고 나머지는 `action`(행위만) 을 쓴다. 둘 다
    `shot_description` 으로 이으면 같은 행동이 두 번 묘사된다 — c02 가 "picks up a book" 을 두
    절에 걸쳐 반복해 책을 두 번 집는 것처럼 읽혔다. 서술 길이 중앙값 216 → **164자**.
  - `--chunk_min_overlap 8`: chunk 에 8프레임 미만 걸치는 창은 서술에서 뺀다 (전 chunk 통틀어
    21회). 화면에 거의 안 나오는 행동이 카메라 focus 를 끌어가는 것을 막는다.
  - 마지막 chunk 는 잘라내지 않고 **당겨서** `end == total-1` 로 맞춘다 (49프레임보다 짧으면
    LBM-Lite 가 못 먹는다).
  - 실측: 2077프레임 → **chunk 86개**, chunk 당 창 min 1 / med 2 / max 3, 창 커버리지
    **37/37**(빠진 창 0), 구성 gap-only 36 · action+gap 36 · action-only 14,
    서로 다른 서술 62/86.
- **`CinemaTraj/scripts/trumans_to_lbm_demo.py --narrative_subject`** (기본 off) — VLM 서술의
  **주어만** `--subject_phrase` 로 갈아끼운다.
  - **왜**: LBM 의 shot focus 는 asset alias 문자열 점수로 정해지는데 캐릭터 alias 는
    description 토큰 중 **len > 3** 만 남는다 (`director_stage.asset_alias_tokens:761-765`).
    인물 접지를 켠 VLM 서술은 "The man ..." 으로 시작하는데 `man` 은 세 글자라 **alias 가 될 수
    없다** — 사람이 alias 를 하나도 못 맞히고 소품이 focus 를 가져간다 (Task #88 의 D2/D3 재발).
    나머지 대명사(he/his)는 점수에 안 쓰이므로 그대로 둔다.
- **`CinemaTraj/scripts/trumans_vlm_action_tag.py --include_gaps`** (기본 `--no_include_gaps` =
  기존과 바이트 동일) — `Actions/<seq>.txt` **밖** 구간까지 창으로 만들어 태깅한다. 새 함수
  `build_windows()` 가 action 창 사이의 빈 구간을 `--gap_min_frames 15` 이상일 때만 취하고
  `--gap_max_frames 120` 으로 쪼갠 뒤 **시간순**으로 번호를 다시 매긴다. 산출물 파일명은
  `_full` 접미사(`<seq>_actions_vlm_full.json`, `sheets_full/`)라 기존 narrative arm 산출물을
  덮어쓰지 않는다.
  - **왜 필요했나 (실측, `2023-01-17@00-55-00`)**: 태그 구간은 746/2077 프레임 = **35.9%** 뿐이다.
    나머지 1331 프레임을 "static + walking" 두 라벨로 덮을 수 있는지 재봤더니 안 된다 —
    walk(≥0.30 m/s) 627 · turn_in_place(≥25°/s) 143 · pose_change(≥0.6 rad/s) 424 · still 137.
    임계를 가장 관대하게 잡아도 262~535 프레임이 두 라벨 밖이고, 미태그 구간의 `body_pose`
    변화율 중앙값 **2.231 rad/s** 가 태그된 `Write with the left hand`(0.701) 보다 크다.
    조용한 구간이 아니라 **라벨만 없는 구간**이다.
  - 창 37개(action 16 / gap 21), 커버리지 **2056/2077 = 99.0%**, 겹침 0. 버린 21프레임은
    f70–80(11f) · f764–773(10f) 로 둘 다 15프레임 미만.
  - gap 창에는 `GAP_SUFFIX` 를 프롬프트에 덧붙인다 — 원 프롬프트의 "하나의 연속된 동작" 전제가
    이동·전환 구간에서 깨져서, 어디서 어디로 걸어가는지를 명시적으로 물어야 한다.
  - 요약표에 `windows N / N (action / gap)` 과 `coverage` 행을 추가했다. 이전 분모가
    `len(actions)` 라 `37 / 16` 으로 찍혔다.
- **`CinemaTraj/scripts/trumans_vlm_action_tag.py` 씬 접지 3종** — `--scene_objects` /
  `--object_motion` / `--rule_hint`, 셋 다 기본 off 라 안 켜면 기존 실행과 바이트 동일.
  narrative arm(VLM 서술)과 rule arm(`Actions/<seq>.txt`) 의 간극을 줄인다.
  - **왜**: 접지 없는 실행에서 VLM 이 **씬에 없는 물체를 지어냈다**. rule 이 `Open/Close the
    oven` 인 4창(w21/w23/w34/w36)을 cabinet / monitor / computer / cabinet 으로 읽었다.
    TRUMANS 는 이걸 검증할 근거를 자기 안에 갖고 있다 —
    `Recordings_blend/<rec>/obj_list.txt` 가 그 녹화의 상호작용 가능 물체 전량,
    `Object_all/Object_pose/<seq>.npy` 가 그 물체들의 프레임별 pose 다. 실측하니 네 창 모두
    `oven_door_01` 이 0.218~0.323 m / 33.1~49.7° 움직인다. **rule 이 맞고 VLM 이 틀렸다.**
  - `--scene_objects`: `obj_list.txt` → `## SCENE OBJECTS` 블록으로 **있는 물체**(cup, oven,
    oven door, book, pen, chair, vase, whiteboard)와 **이 방에 없는 물체**(bottle, cabinet,
    drawer, fridge, handbag, keyboard, laptop, microwave, monitor, mouse, phone)를 둘 다 못
    박고, `validate()` 에도 유령-물체 검사로 건다. 없는 물체 목록은 `Object_all/Object_mesh`
    46종 어휘에서 씬 물체를 **토큰 단위로** 뺀 것 — 첫 토큰만 빼면 `oven door` 의 `door` 가
    남아 정답을 반려한다. 검사는 부분문자열이 아니라 단어 경계(`\bcups?\b`) — "cup"이
    "cupboard"에, "pen"이 "open"에 걸린다.
  - `--object_motion`: `Object_pose` 로 창 안에서 실제로 움직인 물체를 `## OBJECTS THAT MOVE`
    로 준다. 회전은 **euler ptp 가 아니라 회전행렬 geodesic**(창 첫 프레임 기준) 이다 —
    euler ptp 는 wrap 때문에 dL 0.005 m 인 **정지 물체에서 33°** 를 뱉어 임계로 못 쓴다.
    geodesic 이면 노이즈 바닥 ≤ 6.9° / ≤ 0.015 m, 진짜 상호작용 ≥ 12.7° / ≥ 0.084 m 라
    임계 `--move_meters 0.05` / `--move_degrees 12.0` 이 깨끗하게 가른다.
  - `--rule_hint`: action 창엔 자기 rule 라벨을, gap 창엔 **앞뒤 action 라벨 + 프레임 거리**를
    `## LABELLED ACTION` / `## NEIGHBOURING LABELLED ACTIONS` 로 준다. 나머지 둘과 **별도
    플래그로 뺀 이유**: 이건 시뮬레이션 GT 가 아니라 rule arm 자체라, 켜면 두 arm 의 독립성이
    줄어든다(아래 실측).
  - `--recording`(비우면 `scene_flag`→`scene_list` 로 역산) / `--suffix` 추가. 접지를 켜면
    산출물이 `_g` 접미사(`<seq>_actions_vlm_full_g.json`, `trace_full_g/`)로 갈리고
    **sheet 는 `sheets_full/` 를 재사용**한다 — 두 arm 이 정확히 프롬프트 텍스트 하나만 달라진다.
  - `grounding` 블록(recording / objects / absent_objects / 임계)과 창별 `moved_objects` 를
    산출물에 같이 싣는다.
  - **실측(37창, `2023-01-17@00-55-00`)**:

    | | 유령 물체 언급 창 | fallback | repairs | 초 |
    |---|---|---|---|---|
    | 접지 없음 | **6 / 37** (w19·w21·w22·w23·w26·w36) | 0 | 4 | 65.1 |
    | 접지 3종 | 0 / 37 | 1 | 5 | 63.3 |
    | 접지 3종 + `hint_first` | **0 / 37** | **0** | **4** | **62.1** |

    oven 4창: `pulls a long, thin object from a black cabinet` → `opens the oven door with
    their left hand`. gap 창도 `Object_pose` 를 따라간다 — w22 `holding a long object and
    looking at a screen` → `reaches out ... to open the oven door`(oven door 0.08 m/13°),
    w30 `picks up a pen` → `picks up a cup`(cup 0.53 m/29°), w10/w12 는 pen + whiteboard 를
    이름으로 부른다.
  - **`--rule_hint` 의 대가**: action 창 16개에서 rule 내용어가 서술에 포함된 비율 중앙값이
    **0.00 → 0.50**. 서술이 rule 의 재기술 쪽으로 수렴한다. rule 을 GT 로 쓰고 서술은
    공간 디테일만 얻을 거면 켜고, 두 arm 을 독립 신호로 쓸 거면 `--scene_objects
    --object_motion` 만 켜는 게 맞다 (유령 물체 6창은 이 둘만으로도 잡히는지는 미측정 —
    둘 다 시뮬레이션 GT 라 rule 텍스트를 안 본다).
- **`CinemaTraj/scripts/trumans_vlm_action_tag.py` 인물 접지** — `--person_probe`(시퀀스당 VLM
  1턴 추가) / `--person man|woman|person`(수동 override, probe 를 건너뛴다). 둘 다 기본 off.
  - **왜**: 창마다 대명사가 제멋대로였다. 접지 3종 실행 37창 실측 — `their/they` 20창,
    **`his/he` 2창**(w15·w16), `she/her` 0창, 나머지는 대명사 없이 "The person". 같은 시퀀스
    같은 사람인데 두 창만 남성 대명사다. 프롬프트가 성별을 안 주니 **창마다 따로 추측**한다.
  - **GT 는 없다** (2026-08-25 확인). `smplx_result/<seq>_smplx_results.pkl` 에 `gender` 키가
    있긴 한데 **569개 전량이 빈 문자열**이고 `betas` 도 `(0,)` 로 비어 있다. 루트 `betas.npy` 는
    코퍼스 전체에 body shape 5종뿐이고 성별 라벨이 아니다. `trumans_utils` 가 SMPL-X **male**
    메시로 렌더하긴 하지만(`joints_to_smplx.py:34`, `load_smplx_animatioin_clear.py:186`)
    그게 우리가 쓰는 `video_render` 를 만든 경로라는 보장이 없다.
  - 그래서 **시퀀스당 한 번만** 묻고 그 답을 전 창에 못 박는다. 창마다 묻는 것보다 정확해서가
    아니라 **일관되기 때문**이다 — 틀려도 37창이 같이 틀려야 하류에서 한 번에 고칠 수 있다.
  - probe 프레임 선택에 `smplx_result_in_cam` 의 `transl[:,2]`(카메라 좌표 z)를 쓴다. |z| 가
    작을수록 사람이 크게 찍힌다. |z| < 0.8 m 는 버린다 — 실측 18프레임이 여기 걸리는데 몸
    일부만 화면을 채운다. 서로 100프레임 이상 떨어뜨려 4장(같은 그림 4장이 되는 걸 막는다).
    이 sheet 만 타일 폭 2배 + gamma ≥ 1.6 으로 뽑는다(창 sheet 648 px 에서 사람은 100 px 남짓).
    `smplx_result_in_cam` 이 없으면 균등 샘플로 내려간다.
  - `{"gender":"man|woman|unclear","hair","clothing","evidence","confidence"}` 를 받고,
    `unclear` 거나 `confidence < --person_min_conf`(기본 0.6)면 지금까지의 `they/their` 로 간다.
  - `## PERSON` 블록으로 호칭·대명사·`shot_description` 첫 단어를 지정하고, **`validate()` 가
    반대 성별 대명사를 반려**한다 (부탁만으로는 안 지켜진다). 단어 경계로 찾는다 — `he` 가
    `the`/`she` 에, `his` 가 `this` 에 걸린다.
  - 산출물에 `person{resolved,source,pronouns,probe,probe_frames,probe_sheet,min_confidence}`
    를 싣고 접미사에 `p` 를 붙인다(`_full_gp`). probe 원답과 실제 적용값을 따로 남기는 이유는
    confidence 임계 미만일 때 둘이 갈리기 때문이다.
  - **실측(`2023-01-17@00-55-00`)**: probe → `man` conf 0.95 ("short brown hair, blue and
    white hooded vest, blue jeans"). 육안 확인 — f1878 타일에 수염·짧은 머리가 보인다.
    **probe 가 맞다.** 결과: he/his **34창**, she/her 0창, they/their 0창,
    `shot_description` 이 `The man` 으로 시작 **37/37**. fallback 0, repairs 4 → 6, 62.1 →
    68.9 s (probe 1턴 + 대명사 반려 재질의 2턴).
- **`CinemaTraj/out/lbm_demos_wn/_camdump/eval_lbm_arm.sh`** — narrative arm 평가 4단
  (뱅크 매니페스트 → 프레이밍 감사 → Lite 대비 → rule arm 대비). 배선 함정 4개를 헤더 주석에
  전부 실측으로 적어뒀다 — 넷 다 조용히 죽거나 조용히 덮어쓴다:
  - `--work` 의 의미가 **1단과 2단에서 다르다**. 1단 `build_lbm_bank_manifest.py` 는
    `<work>/manifest_a*.json` 을 glob(→ recording 디렉토리), 2단 `audit_lite_framing.py` 는
    `<work>/<recording>/probe_a*.json`(→ 그 위 root). 섞으면 1단이 "manifest_a*.json 이 없다".
  - narrative GT 렌더가 rule arm(TRUMANS-Lite) 과 **다른 root**(`Vista4D-Eval-Data`) 로 나갔다.
    `--eval_data` 를 안 넘기면 전 clip 이 `SKIP (probe/poses/scene 누락)` 뒤 빈 CSV → `IndexError`.
  - CSV 의 `recording` 열은 **짧은 접두사**(`00add26c`)다. 전체 uuid 를 주면 매칭 0행 →
    "두 CSV 에 공통 action 이 없다".
  - `compare_lbm_vs_lite.py --out` 은 **디렉토리**이고 파일명이 `lbm_vs_lite.csv` 로 고정이라,
    3단/4단이 같은 디렉토리를 쓰면 **4단이 3단을 덮어쓴다**. `vs_lite/` `vs_rule/` 로 분리.
- **`CinemaTraj/LBM_DEFECTS.md`** — 원본 LBM 을 TRUMANS/Vista4D 에 돌리며 실측한 결함 `LO1..LO11`.
  **`DECISIONS.md` 와 분리한 이유**: 거기 `D1..D70` 은 LBM-**Lite** 설계 결정이라 번호가 겹친다.
  접두사를 `LO`(LBM Original) 로 뒀다.
  - LO1~LO5 해결(어댑터 / runner 기본값), LO6 은 사용자 지시로 유지, **LO7·LO8 미해결**.
  - LO7 이 제일 위험하다 — Cinematographer 가 `candidate_count_raw_min 1149 → retained 0` 인데
    `success: true` + `cameras: []` 를 내보내고, 한 stage 뒤 `video_stage.py:558` 에서
    **stderr 이 빈 채로** "Blender scene render failed" 라는 **틀린 원인**으로 터진다.
  - LO8: `run_full_pipeline.py:317/:355` 의 `latest_stage_file()` 이 `--run-id` 를 안 본다 —
    창을 병렬로 돌리면 창 A 의 카메라가 창 B 의 렌더로 들어간다(우리는 순차라 안 밟음).
  - 하단에 스톡 대비 실행 설정 4종(`MOVEMENT_VOCAB=extended`, `TRAJECTORY_SCALE=5`,
    `SINGLE_SCENE_FALLBACK=1`, `--camera-quality quality`)과 되돌릴 수 있는지 표로 기록.

### Changed
- **`CinemaTraj/scripts/trumans_first_pose_board.py` 반경 격자 기본값을 `d_ref` 배수 → 절대
  미터로.** `--radii` 기본 `[1.1, 1.5, 2.0, 2.6]` m, `--radii_rel` 은 주면 그때만 상대 격자로
  바뀌는 opt-in 이 됐다 (기본 `[]`). 분기만 뒤집었고 상대 경로는 그대로 살아있다.
  - **왜**: Lite 에는 시작 카메라가 없어서 `d_ref` 배수로 잡을 기준 자체가 애매하다. 그리고
    통과율 천장을 정하는 건 clearance/occlusion 인데 그건 미터 문제라 렌즈에 안 움직인다
    (실측: lens 25/18/14 에서 `clearance` 365 / `occluded` 334 기각이 **동일**, 렌즈에 반응하는
    건 `cropped`/`area` 뿐). 상대 격자의 존재 이유였던 "shot scale 의 뜻을 고정한다"는 이제
    위의 `shot` 태그가 직접 하므로 반경으로 대리할 필요가 없다.
  - `d_ref` 는 절대 격자에서도 계속 계산·기록하고 후보마다 `radius_rel` 로 실린다.
  - 회귀 확인: 새 기본값으로 재실행한 board 가 기존 `--radii 1.1 1.5 2.0 2.6` 실행과 게이트·
    기하·태그 필드 전부 일치 (chunk f51 33/144, f600 40/144, 차이는 `--no_render` 때문에 빠진
    `cell_id`/`c2w_*`/`K`/`lens_mm` 5개 렌더 전용 필드뿐).

### Fixed
- **`CinemaTraj/scripts/trumans_to_lbm_demo.py:341` — `KeyError: 'action'` 으로 recording 이
  통째로 죽던 것**. 서술을 조립할 때 primary 아닌 window 는 `entry["action"]` 을 무조건
  인덱싱했다. `action` 은 항상 있지 않다 — VLM 태거가 스키마 복구에 끝내 실패하면
  `source:"fallback"` entry 를 쓰는데 거기엔 `shot_description` 만 들어간다.
  - **범위**: 태그 JSON 53편 전량 스캔 결과 **88 / 1891 entry = 4.7%**, **53 시퀀스 중 34 편**에
    퍼져 있다. Stage 0 전량 실행에서 `0ab03928` 이 이걸로 rc=1 로 죽었다.
  - **고침**: `entry.get("action") or entry["shot_description"]` 으로 떨어뜨린다. 행위만이 아니라
    위치까지 붙어 조금 장황하지만 서술이 빠지는 것보다 낫고, 뒤의 `strip_subject` 가 주어 중복을
    정리한다. **`action` 이 있는 경우의 결과는 예전과 바이트 단위로 같다.**
  - **안전성 확인**: `grep '\["action"\]'` 로 이 파일에 다른 무방비 사용이 없음을, 그리고 태그
    JSON 1891 entry 중 `action`·`shot_description` **둘 다 없는 것이 0개**임을 확인했다
    (폴백이 그 자체로 KeyError 를 낼 수 없다).

- **`CinemaTraj/scripts/trumans_to_lbm_demo.py --out` 을 `path.abspath()` 로 못 박았다** —
  상대경로로 넘기면 생성된 demo root 가 **조용히 빗나간다**.
  - 증상: `--out out/lbm_demos_c49` 로 만든 `_window.json` 의 `env.BLENDER_USER_SCRIPTS` 와
    `*__run_windows.sh` 의 `--demo-root` / `export BLENDER_USER_SCRIPTS` 가 전부 상대경로로 찍혔다.
  - 원인: LBM 은 `Path(args.demo_root).resolve()` (`Engine/run_full_pipeline.py:226`) 를 자기
    cwd(`Look-Before-Move/`) 기준으로 푼다. `--demo-root` 는 없는 경로라 바로 죽지만,
    `BLENDER_USER_SCRIPTS` 는 **에러 없이** 안 걸려서 frame shift 훅이 붙지 않고 창 전체가
    프레임 1부터 렌더된다 — 실패 신호가 아예 안 난다.
  - 이미 만든 파일 87개는 재실행(9분) 대신 `sed -i` in-place 치환으로 고치고 검증했다.
- **`CinemaTraj/scripts/trumans_to_lbm_demo.py` narrative 로딩에 `action_index is not None`
  필터 추가** — `--include_gaps` 로 만든 `_full*` JSON 은 gap 창의 `action_index` 가 `null` 이라
  기존 `{int(e["action_index"]): e for e in ...}` 가 `TypeError` 로 터진다. action-only JSON 에
  대해서는 동작이 바뀌지 않는다.
- **`CinemaTraj/lbm/vlm.py` `chat_json(echo_previous=..., repair_hint=...)`** — 재질의가 **통째로
  무효**였던 걸 고쳤다. 기본값 `echo_previous=True` 는 기존 동작 그대로라 `loop.py` 는 안 바뀐다.
  - 증상: TRUMANS 37창 태깅에서 3창(w15/w18/w33)이 4턴 내내 `forbidden filming vocabulary:
    ['camera']` 로 반려됐는데, 응답이 **바이트 단위로 동일**했다 (`completion_tokens`
    137/158/116 불변).
  - 원인: 재질의 프롬프트에 `## PREVIOUS_RESPONSE` 로 **자기 답을 다시 보여주면 그대로 베낀다**.
    같은 이미지로 3변형을 1턴씩 돌린 실측:

    | 변형 | w15 | w18 | w33 |
    |---|---|---|---|
    | base | pass | 위반 | 위반 |
    | +prev (t=0.1, 기존) | pass | 위반, identical=True | 위반, identical=True |
    | −prev (t=0.1) | pass | **pass** | **pass** |
    | +prev (t=0.7) | pass | 위반, identical=True | pass |

    t=0.7 에서도 바이트 동일이 나오므로 **샘플링 문제가 아니다**. 그래서 temperature 는 여전히
    안 올린다.
  - `echo_previous=False` 면 자기 답을 빼고 위반 목록 + `repair_hint` 만 준다. 단 **JSON 파싱
    실패일 때는 깨진 원문 자체가 고칠 대상**이라 `echo_previous` 와 무관하게 보여준다.
  - `trumans_vlm_action_tag.py` 는 `echo_previous=False` + `REPAIR_HINT` 로 부른다. `REPAIR_HINT`
    는 처음에 `body_facing` 만 다뤘는데 그러자 위반이 `location`("the left side of the frame")
    과 `action`("turns to face the camera") 으로 옮겨갔다 — 필드 하나를 막으면 옆 필드로 샌다.
    지금은 네 필드 전부에 대해 **금지 표현 → 대체 표현** 쌍을 준다 (`frame` 이 금지어라는 걸
    명시하는 게 특히 중요하다. 모델은 이걸 촬영 용어로 인식하지 않는다).
  - 37창 재실행 실측: fallback **3 → 2 → 0**, repairs 13 → 15 → **4**, 77.6 → 73.5 → **65.2 s**.
- **`CinemaTraj/lbm/vlm.py` `chat_json(hint_first=...)`** — 기본값 `False` = 기존 동작
  (`loop.py` 불변). `True` 면 `## VALIDATION_ERRORS` + `repair_hint` 를 원 프롬프트 **뒤가
  아니라 앞**에 놓는다.
  - 증상: 씬 접지 3종을 켜자 원 프롬프트가 ~600 토큰 길어졌고, **접지 없이는 통과했던 w18** 이
    4턴 내내 같은 `body_facing: "...facing away from the camera..."` 위반을 반복하며 fallback
    으로 떨어졌다. `shot_description` 은 깨끗했으니 힌트를 못 본 게 아니라 **묻힌** 것이다.
  - `trumans_vlm_action_tag.py` 가 `hint_first=True` 로 부른다. 재실행: w18 은 repairs 1 로
    통과, 접지 실행의 fallback **1 → 0**, repairs 5 → 4, 63.3 → 62.1 s.
- **`CinemaTraj/decode/build_poses.py`** — `decision_fingerprint()` 의 `follow_gain` 기본값을
  `0.0`(float) → `"0"`(str). 지문은 `str(follow_gain)` 을 그대로 넣는데 `build_poses` 의
  argparse 기본값(:761)은 문자열 `"0"` 이라, `decision.json` 에 `follow_gain` 키가 없으면
  (= 보통) decode 는 `"0"`, `emit.py:166` 은 `"0.0"` 을 지문에 넣어 **손대지도 않은
  `poses.npz` 를 stale 로 오판**하고 emit 이 rc=1 로 죽었다 (camel 실측). 기존 뱅크의 저장
  지문이 `"0"` 이므로 이 방향으로 맞춰야 재생성이 필요 없다.
- **`CinemaTraj/LBM_DEFECTS.md`** — "Vista4D 실행은 **LBM 코드 0줄 수정**" 이라고 적혀 있던
  줄을 정정. `git diff --stat` 실측으로 원본 리포는 **4파일 151+/9−** 수정된 상태다
  (`cinematographer_quality_worker.py` +30 / `cinematographer_stage.py` +75−9 /
  `director_stage.py` +29 / `blender_render_worker.py` +26). 파일별 변경 지점과 env 게이트를
  표로 추가하고, **env 게이트가 없는 유일한 변경**인 `render_preview` 의 `file_format="PNG"`
  강제를 `LO9`, `.blend` 의 `frame_step=2` 누출을 `LO10` 으로 결함 표에 올렸다.
  LO9 를 안 고치면 채널 board 가 0장이 되어 **VLM 선정·micro-adjust 루프가 통째로 건너뛰어지고**
  기하 점수 1위로 조용히 폴백한다.
  - **`LO11` 추가 (미해결, 진단만)** — Phase-1 채널 VLM 필터가 **TRUMANS 30/30, Vista 3+3/6
    전량 FAIL** 이고 OK 가 한 번도 없다. 원인은 `cinematographer_quality_worker.py:2983
    build_board` 의 `if Image is None: return ""` — 이 워커가 도는 **Blender 번들
    python 3.11 에 PIL 이 없다**(실측). `board_path=""` → `cinematographer_stage.py:3716` 이
    `board_image_missing` 으로 빠지는데, 반환값이 `survivors = rendered` 라 후보가 **하나도
    안 걸러진 채** Phase-2 로 넘어가고 파이프라인은 `success: true` 로 끝난다. `:4970` 이
    `error` 를 안 찍어 로그에는 `FAIL` 세 글자만 남는다. 잃는 것은 Phase-1 프롬프트의
    **facing rule**(뒤통수 샷 기각) 과 구도·가림 기각. LO9(PNG 강제)와는 **다른 원인**이라
    LO9 를 고쳐도 Phase-1 은 여전히 안 돈다.
- **`CinemaTraj/scripts/vista_lbm_to_poses.py`** — Vista 씬에 돌린 **원본 LBM** 의 카메라 덤프를
  Lite `verify.py` 가 그대로 읽는 평가 폴더로 바꾼다. 이게 있어야 LBM 과 Lite 를 **같은 지표표**
  (hole_fraction / subject_in_frame / tau_max / ...)로 잰다.
  - 변환은 `_vista.json` 의 역이다. `T_wg` 에는 **스케일 `S_da3` 가 박혀 있어** (`R@R.T = S²·I`)
    회전과 위치를 한 4×4 로 같이 곱하면 `det(R)=S³` 가 된다 — 나눠서 건다:
    `R_world = (T_wg[:3,:3]/S) @ R_blender_gl @ diag(1,-1,-1)`,
    `p_world = T_wg[:3,:3] @ (p_blender/u_meters) + T_wg[:3,3]`.
  - 프레임 수는 `lbm_camera_to_poses.subsample` 로 **보간 없이 index 만** 솎는다 (camel 106,
    avocado-slice 110 → 49). 보간하면 LBM 의 fcurve easing 이 뭉개져 jerk 지표가 실제보다 매끄럽다.
  - `scene_graph.json`/`cloud.npz` 는 Lite 산출물을 symlink — **같은 점군으로 재야 비교가 된다.**
  - `decision.json` 의 `source` 는 `lbm_original`, `start_mode` 는 `board` (LBM 은 소스 frame0
    카메라를 앵커하지 않는다). `info["tau"]["target_tau"]=None` — LBM 엔 τ 이분법 단계가 없다.
- **`CinemaTraj/scripts/vista_to_lbm_demo.py` + `vista_blend_worker.py` + `vista_blend_check.py`**
  — Vista4D 점군(`out/<scene>/{cloud.npz,scene_graph.json}`) → **원본 LBM demo root**
  (`vista_lbm_demo_v1`). **LBM 코드 0줄 수정** (사용자 확정 방식).
  - **점군이 아니라 삼각형 메시를 굽는 이유.** LBM 의 가림 판정 `occlusion_check`
    (`Cinematographer/cinematographer_quality_worker.py:2451-2502`)가 45개 표본점에 대한
    **dense `scene.ray_cast`** 다. 면이 없는 지오메트리에는 `hit=False` → `occluded=0` →
    `occlusion_ratio=0.0` → `severely_occluded=False` 가 되어 게이트가 **막는 게 아니라 조용히
    꺼진다**(로그 무기록). 그래서 depth 격자를 삼각분할해 진짜 면을 만든다.
  - **노드별로 오브젝트를 쪼개는 이유.** 레이가 focus 오브젝트에 맞으면 가림으로 안 센다.
    배경 한 덩어리에 subject 표면이 섞이면 카메라→subject 레이가 "배경"에 맞아 가림 100% 가 된다.
    격자 버텍스를 노드 OBB 소속으로 나눠 `dyn_0`/`stat_2`/`background` 로 굽는다 (동적 노드를 먼저
    시험해 겹침에서 이기게). 세 버텍스가 **전부 같은 소속**인 면만 남긴다.
  - 동적 노드는 `grid_frames[0]` 지오메트리만 쓰고 scene graph `track` 으로 키프레임을 굽는다.
    다른 프레임의 동적 점을 배경에 넣으면 움직이는 물체가 **49프레임짜리 정지 잔상 벽**이 된다.
  - `--subject_height_m`(기본 1.8) 로 `u_meters` 자동 도출. G 프레임은 무차원인데 LBM 의 near clip
    · "너무 가깝다" · 궤적 크기 기본값이 전부 **미터 스케일 상수**라, 0.13 u 짜리 camel 을 찻잔으로
    취급한다. camel `u_meters=14.7104`, avocado-slice `9.9981`.
  - `_vista.json` 은 **역변환 사이드카** — `u = blender / u_meters`, `world = T_wg @ [u,1]` 로
    LBM 이 낸 Blender 카메라를 소스 world 로 되돌린다.
  - **Blender 버텍스 컬러 3연속 함정** (전부 실측): ① `BYTE_COLOR` 어트리뷰트는 그걸 읽는
    머티리얼이 없으면 평평한 회색으로 렌더된다 → Attribute("Col")→Emission 머티리얼 추가
    (점군 색은 이미 조명이 구워진 값이라 diffuse+SUN 이면 그림자가 두 번 들어간다).
    ② Blender 4.5 기본 view transform 이 **AgX** 라 emission 을 들어올려 탈색시킨다 →
    `view_transform="Standard"`, `look="None"`. ③ `attr.data.foreach_set("color", …)` 는
    **linear** 를 받는다 → sRGB 바이트를 역감마 안 걸면 렌더가 이중 인코딩돼 들뜬다.
  - `vista_blend_check.py` 는 (1) 소스 카메라 재렌더 (2) LBM 과 **같은 `scene.ray_cast`** 적중률
    (3) 오브젝트/면수/애니 표를 한 번에 낸다. 실측 **camel 14/15, avocado-slice 24/27** 적중 —
    미스도 의미상 옳다 (앞의 camel 이 뒤의 camel 을 실제로 가린다).

### Fixed
- **`vista_to_lbm_demo.py` subject 선택** — `moving` → **`dyn_*` 노드** → 전체 순으로 본다
  (`--no_subject_prefer_dyn` 로 예전 동작). `moving` 은 이동량 임계를 넘은 것만 True 라 사람이
  제자리에서 작업하는 씬(avocado-slice)은 전 노드가 False 가 되고, 그때 "화면 면적 최대"로
  떨어지면 **테이블 상판**(두께 0.026 u)이 subject 로 뽑혀 `u_meters` 가 70 이 되고 씬이
  **404 m** 로 부풀었다 (실측). 고친 뒤 `dyn_0` woman, `u_meters=9.9981`.
- **`vista_to_lbm_demo.py --ffmpeg_path` 기본값** `/usr/bin` → `/data1/cympyc1785/tools/bin`
  (형제 어댑터와 동일). `/usr/bin` 에는 ffmpeg 이 없어서 `VideoEngineer/video_stage.py:97` 의
  `shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"` 가 **윈도우 경로로 떨어져**
  `FileNotFoundError` 로 죽었다 (camel 1차 실행, Director/Cinematographer 는 통과한 뒤였다).

- **`CinemaTraj/scripts/lbm_camera_dump_startup.py`** — LBM 이 **실제로 렌더한** per-frame 카메라를
  받아 적는 Blender startup 훅. `trumans_frame_shift_startup.py` 와 같은
  `BLENDER_USER_SCRIPTS/startup/` 에 산다. **LBM 코드 0줄 수정**이고, `LBM_CAMERA_DUMP_DIR` 이
  비면 아무 것도 안 한다 (= 기존 동작과 바이트 동일).
  - **왜 훅인가.** camera package 의 `trajectory_keyframes` 는 3~5개뿐이고 그 사이를 Blender 가
    채운다 — `blender_render_worker.py:475` 의 slerp 재분할, `:537` 의 preset lens 램프,
    `video_runtime.py:417` 의 fcurve easing. 게다가 가시성 검증이 실패하면 `:1094` 가
    motion_scale 0.75/0.5/0.35/0.2 로 궤적을 **통째로 갈아끼운다** — 즉 키프레임이 렌더된
    카메라라는 보장 자체가 없다. numpy 로 다시 보간하면 LBM 이 아니라 그 근사치를 평가하게 된다.
  - **`render_pre` 는 못 쓴다 (실측).** 처음엔 프레임마다 발화하는 `render_pre` 에서
    `camera.matrix_world` 를 읽었는데 w01 110프레임이 **전량 같은 값**으로 나왔다. Blender 4.5.9
    최소 재현(키 3개로 x 를 0→2): `render_pre` 의 `matrix_world` 도 `evaluated_get(depsgraph)` 도
    세 프레임 모두 2.0. `bpy.ops.render.render(animation=True)` 가 depsgraph **사본** 위에서
    애니메이션을 평가하므로 원본 datablock 은 마지막 `keyframe_insert` 값에 멈춰 있다.
    fcurve 직접 `evaluate()` 도 기각 — Blender 4.4+ slotted action 에서 object 와 camera-data 가
    한 action 을 공유해 `action.fcurves` 가 object 슬롯만 돌려주고 `lens` 가 통째로 빠진다.
  - 그래서 **`render_complete`** 에서 `frame_start..frame_end` 를 `frame_set` 으로 되짚어 읽는다
    (animation 렌더당 1회 발화; `render_post` 는 프레임마다). 같은 재현 실험에서 이 경로만
    x=0/1/2, lens=24/25/26 을 정확히 돌려줬다. `frame_current` 는 읽기 전후로 저장·복원한다.
  - 산출물 `<dir>/cam_<pid>.jsonl` — 프레임당 한 줄로 `matrix_world`(Blender GL 그대로) ·
    `lens_mm` · `sensor_*` · `res` · `filepath`. **규약 변환은 훅에서 안 한다** (읽는 쪽 한 군데로).
  - 실측 (w01): 110줄, 씬 프레임 1..110, 이동량 **0.323076 m** — camera package 의
    `trajectory_plan.safety_report.travel_distance` 와 소수점까지 일치.
- **`CinemaTraj/scripts/lbm_camera_to_poses.py`** — 위 덤프 → `trumans_recon` 의
  `poses_a<NN>.npz` 규약(`cam_c2w` (N,4,4) OpenCV + `aim` (N,3), TRUMANS Blender world metre).
  이게 있어야 LBM 카메라를 **Lite 뱅크와 같은 눈금**(`audit_lite_framing.py`)으로 잰다.
  - 규약 변환 `c2w_cv = c2w_gl @ diag(1,-1,-1,1)` — `trumans_gt_render.py:64 GL2CV` 와 같은 식.
    **재앵커는 안 한다** (`trumans_to_recon.convert()` 가 depth 와 짝을 맞춘 뒤 하류에서 한다).
  - 프레임 수 정합: LBM 은 `target_frame_count`(movement 종류가 정하는 고정표)만큼 렌더하므로
    창 길이와 무관하다 — w01 은 19프레임 창에 110프레임. Lite 는 항상 49. **보간 없이 인덱스만**
    `np.rint(np.linspace(0, N-1, 49))` 로 고른다. 고정 step 은 꼬리를 잘라먹어(w01 이 0..96 만
    남아 0.323 → 0.284 m, 88%) 기각했다. 보간하면 LBM 의 easing 이 뭉개져 jerk 지표가 실제보다
    매끄럽게 나온다.
  - 시간축 `trumans_frame = scene_frame − TRUMANS_FRAME_OFFSET` (오프셋은 demo root 의
    `_window.json`). w01 은 씬 1..110 → TRUMANS **51..160** — 창(51..69)을 91프레임 넘어간다.
  - **회전 규약 검증** (w01, 49프레임): OpenCV c2w col2(forward)가 camera package 의 `target`
    을 향하는 각도 mean 0.232° / max 0.364°, col0·col1 은 각각 89.79° / 90.10°,
    `det(R)` 0.999999555..1.000000637, Blender +Z 기준 roll mean 0.011° / max 0.016°.
    이동량 0.323 m 은 `travel_distance` 와 일치.
- **`CinemaTraj/scripts/trumans_to_recon.py --poses_override`** (기본 `""` = 기존 동작 그대로) —
  카메라를 합성하지 않고 **밖에서 받는다** (`lbm_camera_to_poses.py` 산출 npz). 궤적 합성(2단계)과
  검증 probe(3단계)를 건너뛰고 GT 렌더 + `recon_and_seg` 변환만 돈다. LBM 카메라를 Lite 뱅크와
  **같은 눈금**(`audit_lite_framing.py`)으로 재기 위한 경로.
  - **검증 probe 를 일부러 건너뛴다.** 그건 "우리가 세운 후보가 벽을 뚫나"를 보는 필터라
    평가 대상(LBM 카메라)에 걸면 베이스라인을 우리 기준으로 걸러버린다. 벽 통과 여부는 하류 감사가 잰다.
  - **프레임 격자를 원본 창 안으로 클램프하지 않는다.** LBM 이 창 밖까지 렌더한 것 자체가 결함(D6)
    이고, 잘라내면 그 결함이 평가에서 사라진다. `start,end` 는 npz `trumans_frames` 의 양 끝.
  - npz 의 `lens` 를 `poses_a<NN>.npz` 에 `lens_mm` 으로 실어 보낸다 — LBM 은 24 mm 인데
    `trumans_gt_render.py --lens` 기본값은 25 mm 라, 안 실으면 화각이 4% 넓게 렌더돼서 프레이밍
    지표가 통째로 어긋난다.
  - manifest 에 `kind:"lbm_render_dump"` / `poses_override` / `frame_list` 를 기록한다.
- **`CinemaTraj/scripts/{trumans_gt_render,trumans_scene_probe}.py --frame_list`**
  (기본 `[]` = `--frames start end [step]` 격자 그대로, 바이트 동일) — **간격이 균일하지 않은**
  프레임 집합을 그대로 렌더/샘플한다. LBM 은 창 길이와 무관한 `target_frame_count` 만큼 렌더하므로
  (w01: 19프레임 창 → 110프레임) 그걸 49로 솎으면 step 이 1..3 으로 섞인다.
  - `trumans_scene_probe.py` 는 샘플 격자를 **목록 하나**로 못 박고 `track` 색인을 그 목록의
    위치로 바꿨다 — `(frame - start) // step` 산술은 균일 격자에서만 맞다. anchor 도 격자 위로
    스냅한다 (안 그러면 `track[...]` 이 그 프레임이 아닌 이웃을 집는다). 균일 경로는 예전
    `round` 의 banker's rounding 까지 그대로 보존.
- **`CinemaTraj/scripts/trumans_vlm_action_tag.py`** — TRUMANS 창별 **VLM action tagging**.
  `Actions/<seq>.txt` 의 창 경계로 `video_render/<seq>.pkl.mp4` 를 잘라 contact sheet 를 만들고
  Qwen3-VL 에게 서술을 받는다. 산출 `<out>/<sequence>_actions_vlm.json`
  (`trumans_vlm_actions_v1`) + `trace/` + `sheets/`. 규칙 라벨은 **동작 라벨이지 shot 서술이
  아니라서** Director 가 읽을 공간 정보(어느 방, 어느 소품, 어느 방향)가 비어 있다.
  - **타일은 native 648 px 2행**으로 붙인다. 320 px 로 줄였더니 `Pick up the book` 창을
    "presses a button on a remote control" 로 읽었다 — 씬(소파·거실·커피테이블)은 맞고 손에 든
    작은 물체만 틀렸다. 원본 폭으로 올리자 book / coffee table / sofa / floor lamp 를 정확히 집었다.
  - **카메라 어휘 금지가 이 프롬프트의 핵심 제약이다.** 이 서술은 **카메라를 고르는 쪽의 입력**으로
    들어가는데, 초안이 3/3 창에서 "turns to face the camera" / "with the camera positioned to
    capture the full action" 처럼 아직 존재하지도 않는 카메라를 기준으로 방향을 적었다. system
    프롬프트 금지 + `validate()` 블록리스트(`camera/shot/take/frame/screen left/...`)로 **재질의**
    시킨다 — 사후 문자열 치환이 아니라 재질의인 이유는 방향 표현 자체가 카메라 상대라서다.
    16창 재실행 결과 위반 0건, repair 1회, 27.7 s.
  - 프레임은 `CAP_PROP_POS_FRAMES` 로 seek 하지 않고 **순차로** 읽는다 (B-frame 드리프트).
- **`CinemaTraj/scripts/trumans_to_lbm_demo.py --narrative_json`** (기본 `""` = 기존대로
  `Actions/<seq>.txt` 규칙 라벨) — 위 VLM 서술로 `shot_description` 만 갈아끼운다. 창 경계 ·
  movement term 수열 · 프레임 오프셋은 그대로라 **두 팔이 정확히 한 입력만 다르다** (실측: 창 16개
  동일, preset 수열 동일, `run_windows.sh` 주석 제외 구조 동일). `sequence` 불일치는 assert 로
  막는다 — 다른 take 의 서술을 물리면 창 번호는 맞는데 내용이 딴 씬이라 조용히 통과한다.
  VLM 서술은 이미 "A person ..." 으로 시작하므로 `--name_subject` 를 안 태운다.
- **`CinemaTraj/scripts/build_lbm_bank_manifest.py`** — `--poses_override` 로 렌더한 LBM arm clip
  들을 `trumans_bank_v1` manifest 로 묶어 `audit_lite_framing.py` 가 **감사 코드 수정 없이** 읽게
  한다. Lite 뱅크는 드라이버가 manifest 를 같이 뱉지만 LBM arm 은 `trumans_to_recon.py` 를 창마다
  직접 부르는 경로라 manifest 가 없다. 접미사 키는 `action.id`(1-based)가 아니라 **`action_index`**
  다 — 파일 이름(`probe_a<NN>.json`)을 만든 게 그 값이다.
- **`CinemaTraj/scripts/compare_lbm_vs_lite.py`** — 두 팔의 `lite_framing.csv` 를 action 으로 붙여
  `<out>/lbm_vs_lite.csv` + 요약표. LBM arm 은 8창뿐이라 **교집합만** 비교한다 (없는 쪽을 0 으로
  채우면 없는 게 나쁜 점수로 읽힌다).
- **`CinemaTraj/scripts/trumans_frame_shift_startup.py`** — LBM 이 띄우는 모든 Blender 프로세스에서
  TRUMANS 애니메이션을 메모리에서만 앞으로 당기는 startup 훅. `BLENDER_USER_SCRIPTS=<dir>` 의
  `<dir>/startup/*.py` 가 기동 시 자동 import 되고, 거기서 건 `@persistent load_post` 핸들러가
  CLI 로 지정한 `.blend` 로드 **직후** 발화하는 것을 이용한다 (빈 startup 파일에도 한 번
  발화하므로 `bpy.data.filepath` 로 거른다). **LBM 코드 0줄 수정.**
  - **왜 필요한가.** `VideoEngineer/blender_render_worker.py:1141-1142` 가 `scene.frame_start = 1;
    scene.frame_end = frame_count` 로 무조건 덮어쓴다. TRUMANS take 는 2077 프레임짜리 단일
    `.blend` 라, 어느 shot 을 만들든 **항상 take 의 첫 1.5 초만** 렌더됐다 (15 shot 전부 같은 구간).
    `Actions/<seq>.txt` 의 `(start, end)` 창이 통째로 버려지던 것.
  - 창마다 `.blend` 를 복사하면 1.66 GB × 15 창 × 7 편 ≈ 170 GB 라 디스크로는 못 푼다. LBM 은
    `.blend` 를 저장하지 않으므로 이 변형은 프로세스 안에서만 살고 원본 파일은 그대로다.
  - `TRUMANS_FRAME_OFFSET` (키프레임에 더할 값, 창 시작 `s` → `1-s`) / `TRUMANS_FRAME_COUNT`
    (선택, `scene.frame_end` 진단용) 두 env 로 제어. offset 0 이면 아무 것도 안 한다.
  - assign 된 action 만 훑는다 — TRUMANS blend 은 미사용 take 가 254개고 실제 쓰이는 건 9개.
    fcurve 는 `foreach_get/set` 로 배열째 옮긴다 (`zzy3` 하나가 405 fcurve × 2077 키).
  - 실측: 훅 비용 +0.7 s/Blender 기동 (3.176 → 3.867 s). `rigid_01_root_book_right_01` 의 world
    translation 이 offset −50 에서 씬 프레임 1/10/19 = 원본 51/60/69 와 완전 일치.
- **`trumans_to_lbm_demo.py --split_windows` / `--semantic_assets` / `--name_subject`
  (+`--subject_phrase`)** — 셋 다 기본 off 라 기존 호출은 바이트 동일한 demo root 를 낸다.
  - `--split_windows`: storyboard shot 을 frame 창 단위로 쪼개 **창마다 demo root 하나**를 만든다
    (`<uuid>__w01_f0051_0069/`). 창 정보는 `_window.json` (`trumans_window_v1`) 에 싣고, 창별
    `TRUMANS_FRAME_OFFSET` 을 export 해 4단계를 도는 `<uuid>__run_windows.sh` 와 startup 훅을
    복사한 `_frame_shift/startup/` 을 같이 뱉는다. layout worker 는 창 시작 프레임에서 돌린다
    (소품이 take 내내 움직인다). 1 root = 1 shot 이므로 `movement_term_target` 도 창마다 산다.
  - `--semantic_assets`: asset `description` 을 `"<id> in the scene"` 대신 실제 의미
    (`book_right_01` → `the book on the right`, 캐릭터 → `the person, an adult human character`)
    로 채운다. `director_stage.asset_alias_tokens:752-778` 이 description 토큰 중 `len>3` 인
    것만 alias 로 쓰므로, placeholder 로는 `zzy3` 가 **어떤 alias 도 못 갖는다** — `infer_focus_ids`
    가 사람 대신 소품을 골랐던 직접 원인.
  - `--name_subject`: 액션 문장 주어를 명시 (`"Pick up the book"` → `"The person picks up the
    book"`). 3인칭 변화는 규칙 + 불규칙 4개 테이블.
  - 실측 (`00add26c-…`, 16창): `infer_focus_ids` 가 **16/16 `['zzy3']`** 반환. 이전 x15 는
    `book_right_01` ×8 / `book_left_01` ×5 / `oven_base_01` ×2, `zzy3` primary **0/15**.
    16창 중 15창은 `human_primary_requested=False` 이므로 `repair_human_primary_focus` 우회로가
    아니라 정상 스코어링 경로로 고쳐진 것.
- **`scripts/video_grid.py`** — mp4 들을 라벨 붙여 임의 R×C 격자로 붙인다. 행 라벨(왼쪽 세로 띠)
  + 열 라벨(위 가로 띠)을 따로 받는다. `tools/recammaster/halfsplit_compare_grid.py` 를 안 쓰는
  이유는 저게 `<src>_<model>_<cam_src>.mp4` 라는 고정 파일명 규칙에 묶여 있어서 임의 경로 조합엔
  못 쓰기 때문. 기본 배치는 "위=depth warp / 아래=생성" — 출력만 보면 hole 이 warp 탓인지 생성
  탓인지 못 가른다.
  - 세로 라벨 띠는 가로로 그린 뒤 `rotate(90, expand=True)` 하므로 `width`/`height` 는 **회전 전**
    기준이다 (전치를 미리 해두면 이중으로 뒤집혀 concat 이 터진다).
  - libx264+yuv420p 는 짝수 해상도만 받는데 띠 두께가 폰트 크기에서 나와 홀수가 흔하다 →
    오른쪽/아래 1px 패딩. 없으면 ffmpeg 이 broken pipe 로 죽는다.
- **`CinemaTraj/scripts/bank_to_vista4d_cams.py`** — 뱅크 변이를 Vista4D eval 카메라 npz
  (`eval_data/cameras/<video>/<tag>.npz`) 로 직접 내보낸다. `tools/recammaster/vista4d_prepare.py`
  를 안 거치는 이유: 저건 canonical(rmax=1) 을 받아 `|t|max = g*S` 로 다시 키우므로 τ 사다리로
  맞춰 놓은 이동량이 `g` 로 덮인다. CinemaTraj 의 point cloud 가 곧 Vista4D recon 이라
  (`cloud.npz` `meta_cam_c2w` vs `recon_and_seg/cameras.npz` maxdiff 0.0) `poses.npz['cam_c2w']`
  는 이미 recon world 절대 pose·절대 미터다.
  - **intrinsic zoom 이 살아서 넘어간다.** `render_eval.py` 가 `intrinsics_tgt` 를 (N,4)
    프레임별로 받으므로 `fx[f] = fx_base * focal_scale[f]` 를 K 에 구워 넣으면 Vista4D 가 그대로
    소비한다 (canonical 경로엔 intrinsics 채널이 없어 `zoom_dropped` 되던 것과 대비).
    실측 `zoom_out_pan_right`: fx 1184.99 → 789.99.
  - `fx_base` 는 `bank.json` **최상위** `fixed_focal` 키로 분기 (frame0 고정 / 프레임별).
    `bank['fixed']['fixed_focal']` 로 읽으면 항상 False 가 되어 DA3 드리프트(+5.5%)가 같이 실린다.
  - frame0 검증은 **위치만** assert 한다. `aim="look_at"` preset 은 frame0 부터 subject 를 보므로
    회전이 소스와 다른 게 정상이다 (`orbit_left_pedestal_up` 실측 6.21°) — 표에 `rot@f0` 열로 보고.
  - `cam_c2w_fsff`/`intrinsics_fsff` 도 같은 값으로 같이 쓴다 (`utils/media.py:161`
    `load_cameras` 가 `--force_same_first_frame` 일 때 그 키를 찾고, 없으면 KeyError).
- **`CinemaTraj/scripts/lbm_render_reel.py`** — 원본 LBM 실행분의 Blender 렌더 프레임
  (`<run>/renders/<shot>/<cam>/frames/frame_%04d.png`) 을 shot 별 mp4 + 라벨 릴로 잇는다.
  `clips/<shot>/<cam>/clip.mp4` 를 안 쓰는 이유는 그게 1프레임이기 때문 (TRUMANS `.blend` 의
  `frame_step=2` → 홀수 프레임만 렌더 → `encode_frames()` 의 `-i frame_%04d.png` 가 첫 구멍에서
  멈춘다). 프레임 번호 간격을 median 으로 재서 stride 를 표에 찍는다 — stride 2 면 `--fps 12.5`
  가 의도된 속도다. 라벨은 `trajectory_plan.json` 의 preset · travel_distance · visibility guard.
- **합성 preset 2종 + per-frame intrinsic zoom (`CinemaTraj/lbm/presets.py`,
  `decode/build_poses.py`, `scripts/{sample_camera_bank,render_bank_videos}.py`)** —
  지금까지 preset 은 전부 단일 축이라 "두 동작을 **동시에**" 하는 샷을 만들 수 없었다.
  - `zoom_out_pan_right` — `aim="traj"` (조준 없음) + pan right + **intrinsic** zoom out.
    이동이 0 이라 `ROTATION_ONLY_PRESETS` 에 넣는다.
  - `orbit_left_pedestal_up` — `true_orbit` + `pedestal` 동시. `aim="look_at"` 이라
    `tracking="world"` 를 주면 **frame 0 subject 중심**을 계속 본다.
  - `PRESET_ZOOM_END` + `focal_track(name, n, speed)` — preset → 프레임별 focal 배율
    `(n,)`, `[0]=1.0`. **로그 선형**이다 (hfov = `atan(W/2f)` 라 f 를 선형으로 흔들면 화각
    변화가 앞뒤로 안 고르다). `_resample` 로 궤적과 같은 인덱스 규칙을 쓴다.
  - focal 은 SE(3) 밖이라 `cam_c2w` 로 못 나른다 → `poses.npz` 에 `focal_scale (V,n)` 키를
    추가하고, `render_bank_videos.py` / `measure_trajectory` 가 `K_src[f]` 의 `fx,fy` 만
    배율해 넘긴다 (`cx,cy` 는 그대로 — zoom 은 주점을 안 옮긴다). 옛 뱅크엔 이 키가 없어
    소비자는 전부 optional 처리 = 기존 동작 그대로.
  - **zoom 을 hole 측정에도 반영해야 한다.** 안 넣으면 조용히 낮게 찍힌다 — 실측 snowboard
    `zoom_out_pan_right` frame 48 valid `0.538 → 0.374`, 뱅크 `hole_fraction` `0.176 → 0.295`
    (`subject_in_frame` 은 `0.80 → 1.00`, 화각이 넓어져 피사체가 안 나간다).
  - `emit_model_cams.py` 엔 intrinsics 채널이 없으므로 zoom 은 **검증 렌더러만** 소비한다
    (`emit.py` 의 `zoom_dropped`). 하류 6개 모델에는 안 실린다.
- **`--fixed_focal` (`scripts/sample_camera_bank.py`)** — 이미 `render_bank_videos.py` /
  `lbm/render.py` 에만 있던 플래그를 샘플러에도 뚫는다 (기본 off = 기존 동작). DA3 는 프레임마다
  focal 을 다시 추정해 정지 카메라에서도 드리프트하므로, 켜지 않으면 **의도한 zoom 과 추정
  드리프트가 섞여** 화각 변화를 못 가른다. `PRESET_ZOOM_END` 배율은 이 고정된 `K_src[0]` 위에
  곱해진다. 렌더 때 같은 값을 줘야 hole 측정과 영상이 안 어긋나므로 `bank.json.fixed_focal`
  에 기록한다. 고정 시 hole `0.295 → 0.315` (zoom), `0.192 → 0.202` (orbit).
- **`--pan_deg` / `--no_fit_tau` (`scripts/sample_camera_bank.py`)** — "정확히 45°" 처럼
  각도가 요구사항일 때 쓴다. `fit_tau` 는 SE(3) 로그를 통째로 스케일해 τ 예산에 맞추므로
  **회전까지 같이 깎아** 요청한 각도를 지킬 방법이 없었다. `--no_fit_tau` 면 scale 1.0 으로
  preset 정의 각도가 그대로 나가고 τ 는 결과값이 된다 (예산을 안 지킨다). `--pan_deg` 는
  사다리 매핑을 무시하고 각도를 직접 준다. 둘 다 기본값은 기존 동작
  (`fit_tau=True`, `pan_deg=0`). `bank.json` `fit_tau` / `shape.pan_deg`, `bank.csv`
  `focal_end` / `fit_tau` 열에 기록.
  - 실측 (snowboard dyn_0, `--no_fit_tau`, `tracking=world`, τ 는 결과값):
    | preset | 광축 회전 | ‖t‖max | focal_end | tau_max | hole | inFrame |
    |---|---|---|---|---|---|---|
    | `zoom_out_pan_right` | 45.00° | 0.0000 | 0.6667 | 1.9441 | 0.295 | 1.00 |
    | `orbit_left_pedestal_up` | 46.05° | 0.6807 | 1.0 | 2.1556 | 0.192 | 0.20 |
    orbit 쪽 46.05° 는 45° orbit 방위각 + pedestal 이 얹은 pitch 의 합이다.
    `inFrame 0.20` 은 버그가 아니라 `tracking="world"` 의 정의다 — frame 0 위치를 계속
    보므로 움직이는 subject 는 프레임을 벗어난다.
- **`pull_out_arc_left` / `push_in_arc_right` preset (`CinemaTraj/lbm/presets.py`)** —
  기존 `push_in_arc` 는 `arc(+sweep/2)`(카메라가 왼쪽), `pull_out_arc` 는 `arc(-sweep/2)`
  (오른쪽) 로 **한 방향씩만** 있어서 "뒤로 빠지면서 왼쪽으로 도는" 샷을 만들 수 없었다.
  부호를 뒤집은 짝을 채워 2×2 를 완성한다. 별칭·기존 preset 동작은 그대로.
- **`--sweep_deg` (`scripts/sample_camera_bank.py`)** — orbit/arc sweep 상한을
  `DEFAULT_SHAPE["sweep_deg"]`(45°) 대신 직접 준다. 0 = 기존 동작. `bank.json`
  `shape.sweep_deg` 에 기록.
  - **`true_orbit` 계열엔 거의 안 먹는다.** `fit_tau` 가 SE(3) 로그를 통째로 스케일하는데
    순수 나선에서는 sweep 과 scale 이 상쇄된다. 실측 (snowboard dyn_0, lock·fauto·s9·k3,
    `view_angle_max_deg`):

    | preset | τ | sweep 45 | sweep 135 |
    |---|---|---|---|
    | `orbit_left_arc` | 0.35 | 30.9 | 30.4 |
    | `orbit_left_arc` | 0.60 | 53.5 | 53.5 |
    | `orbit_left_arc` | 1.00 | 139.4 | 176.7 |
    | `pull_out_arc_left` | 0.35 | 22.8 | 28.4 |
    | `pull_out_arc_left` | 0.60 | 34.8 | 46.2 |
    | `pull_out_arc_left` | 1.00 | 55.7 | 80.9 |

    `arc + dolly` 합성 preset 은 두 성분 비가 바뀌므로 실제로 먹는다. **orbit 을 더 돌리는
    손잡이는 τ 다** — τ0.35→0.60 에서 30.9→53.5°, 대신 hole 0.117→0.221.
- **`--fixed_focal` (`CinemaTraj/lbm/render.py`, `scripts/render_bank_videos.py`)** —
  `CloudRenderer` 가 쓰는 소스 K 를 frame0 값으로 전 프레임 고정한다
  (`K_src = np.repeat(K_src[:1], F, axis=0)`). 기본 off = 기존 동작(프레임별 K) 그대로.
  **실측 결과는 음성이다 — 채택하지 않는다.** 옵션은 진단용으로만 남긴다.
  - 동기: `cloud.npz` 의 `meta_K` 는 `(49,3,3)` 이고 상수가 아니다. snowboard 에서
    `fx=fy` 가 1184.992 → 1262.462 (**+6.99%**), `cx,cy` 는 640/360 으로 정확히 상수.
    화면 가장자리 환산(`Δfx/fx · W/2`)으로 프레임당 p95 **11.18 px**, savgol(w15,p2)
    detrend 후 고주파 p95 **6.59 px** — `--follow_smooth 9` 를 건 뒤 남는 최대 고주파 항이
    이것이었다 (plan 위치 s9 의 HF p95 는 3.69 px).
  - 흔들림 측정 (좌우 가장자리 12% 띠, 프레임간 `mean|I_t − I_{t−1}|`; subject 가 중앙이라
    이 띠는 정지 배경이다):

    | 변이 | 프레임별 K 평균/p95 | frame0 K 평균/p95 |
    |---|---|---|
    | `truck_left` | 14.14 / 23.03 | 14.04 / 23.15 |
    | `pull_out_arc` | 13.15 / 17.11 | 12.70 / 16.55 |

    **≤3% 변화 — 렌더 시점에 K 를 고정해도 흔들림이 안 줄어든다.** `hole` 값은 두 모드가
    바이트 단위로 동일했다(hole 은 기하가 정한다).
  - 반대로 렌더러 자기 일관성은 **깨진다**. `python -m lbm.render --video snowboard
    --self_check`:

    | 프레임 | 0 | 12 | 24 | 36 | 48 |
    |---|---|---|---|---|---|
    | reproj_px 기존 | 0.0003 | 0.0003 | 0.0004 | 0.0004 | 0.0005 |
    | reproj_px fixK | 0.0003 | 8.2743 | 29.9182 | 43.6588 | 38.0215 |
    | PSNR_ntp 기존 | 28.89 | 28.29 | 32.79 | 30.05 | 27.21 |
    | PSNR_ntp fixK | 28.89 | 19.07 | 19.54 | 15.52 | 15.47 |

    기존 PASS, fixK 는 [12,24,36,48] 에서 FAIL. 음성 대조군(y축 반전 c2w) 719.0 px 로
    두 실행 모두 검사 자체는 유효했다.
  - 이유: 4D 점군을 **프레임별 K 로 unproject** 했으므로 focal 흔들림이 world 점 좌표에
    이미 구워져 있고, `temporal_persistence` 가 49프레임 출신 점을 섞는다. 렌더 시점의 K
    선택으로는 기하에 들어간 흔들림을 뺄 수 없고 화각만 어긋난다. 없애려면 recon 을
    공유 intrinsics 로 다시 돌려야 하는데 `Vista4D/utils/recon_and_seg/recon_da3.py:44`
    (`intrinsics = K_to_intrinsics(prediction.intrinsics)`) 에 그런 옵션이 없다.
  - 산출물: `results/20260824_snowboard_fixfocal/{truck_left_K_ab.mp4,
    pull_out_arc_K_ab.mp4, preset_k3_smooth_fixfocal.mp4}`.
- **`--follow_smooth` (`CinemaTraj/decode/build_poses.py`) + `--follow_smooths` 축
  (`scripts/sample_camera_bank.py`)** — follow **위치 채널** 전용 저역통과
  (Savitzky-Golay, polyorder 2, `mode="interp"`). 기본 `9`, `1` = 끔(= D72 원래 동작).
  - 왜: "카메라가 흔들린다"의 원인이 `tracking` 이 아니었다. 실측 (snowboard dyn_0,
    fauto k3) — 회전 2차차분 p95 가 소스 2.3803° 인데 plan 은 truck_left k3 0.0241 /
    orbit_left_arc 0.0506 / truck_left k0 0.0000° 로 사실상 완벽히 매끈하다. 흔들리는 건
    **위치**다: subject `center_smooth` 의 |jerk| p95 = 0.01728u, plan 카메라 = 0.01572u
    = **정확히 그 0.91배(=g)**. preset 모양은 정의상 매끈하므로 100% 이 채널에서 온다
    (z_med 1.751 · f 1185 → 프레임당 10.6 px).
  - `track.center_smooth` 는 이미 savgol(w=11, p=3)을 거쳤지만 그건 **조준(회전)** 눈금에
    맞춘 것이다. follow 는 그 궤적을 g 배로 **위치**에 실으므로 남은 고주파가 그대로 흔들림이
    된다. 조준(`reference_centers`)은 raw 를 그대로 쓴다 — 조준은 subject 를 실제로 따라가야
    맞고 문제되는 건 병진 쪽이다. gain 은 편 궤적 위에서 다시 푼다.
  - window 9 를 고른 근거 (snowboard dyn_0, g 를 매 창마다 재해):

    | win | g* | tau* | jerk p95 | px/f³ | subject 이탈 |
    |---|---|---|---|---|---|
    | 1 | 0.91 | 0.0932 | 0.01572 | 10.6 | 0.0000 |
    | 9 | 0.91 | 0.0909 | 0.00357 | 2.4 | 0.0119 |
    | 21 | 0.92 | 0.0868 | 0.00336 | 2.3 | 0.0416 |
    | 31 | 0.92 | 0.0968 | 0.00254 | 1.7 | 0.0490 |

    9 가 무릎이다 — 흔들림 4.4배 감소, τ 는 오히려 개선(0.0932→0.0909), 이탈 0.012u.
    21 이상은 흔들림이 더 안 줄면서 이탈만 3~4배 커진다.
  - 재확인 (`follow_smooth_bank`, dyn_0 × preset 14종 × τ0.35 × fauto × k3 × {s1,s9}):
    s1→s9 에서 jerk p95 0.00538 → 0.00122 (s_curve 만 0.00674 → 0.00280), g* 전량 0.91
    불변, `path_len_u` 3자리까지 동일, hole 최대 변화 0.004.
  - `decision_fingerprint` 에는 **`follow_gain` 이 0 이 아닐 때만** 싣는다. gain 0 이면
    offset 이 통째로 0 이라 창이 pose 를 못 바꾸는데, 무조건 실으면 D73 이전 뱅크 전량이
    거짓으로 stale 판정된다. 같은 이유로 `sample_camera_bank` 는 gain 0 행에서 smooth 축을
    접는다 (안 그러면 동일 변이가 창 수만큼 복제된다).
- **`dolly_in` / `dolly_out` preset (`CinemaTraj/lbm/presets.py`)** — 광축 전후진이면서
  `aim="traj"`, 즉 **조준을 안 한다**. 기존 dolly 계열(`straight_ease`/`push_in_arc`/
  `pull_out_arc`)은 전부 `aim="look_at"` 이라 매 프레임 subject 를 향해 회전이 다시 서므로,
  "track 도 look-at 교정도 없는 순수 dolly"를 만들 수단이 traj 계열(pan/truck/pedestal)에
  없었다. 별칭 `zoom_in`/`zoom_out` 은 기존대로 `push_in_arc`/`pull_out_arc` 를 가리킨다.
- **`--follow_gain` (`CinemaTraj/decode/build_poses.py`) + `--follow_gains` 축
  (`scripts/sample_camera_bank.py`)** — subject 변위를 카메라 **위치**에 싣는다
  (`p(f) += g·(c(f) − c(0))`). 기본 `0` = 기존 동작 (snowboard 뱅크 10변이 재생성해
  `max|diff| = 0.000e+00` 확인). `"auto"` 면 τ 를 최소로 만드는 g 를 풀어서 쓴다.
  - 왜: τ = |p_plan(f) − p_src(f)| / z_med 는 **움직이는 소스 카메라 기준**이라, 소스가
    subject 를 따라간 영상에서는 plan 카메라가 frame0 에 가만히 있는 것만으로 예산을 넘긴다.
    snowboard 는 τ_start 1.9441 (예산 0.20 의 9.7배) 이라 뱅크 360 변이 중 **350 이
    `dropped_saturated`**, 남은 10 은 `STATIC_PRESETS` 예외뿐 = 뱅크 전체가 정지 카메라
    복제본이었다. 실측 53편 중 `src_self_tau > 0.20` 이 9편 (snowboard 1.944 /
    snow-bike 1.377 / jogging-woman 0.961 / parkour 0.745 / truck-pose 0.589 /
    fashion-walk 0.491 / trumans-bedroom 0.467 / couple-rocks 0.277 /
    funeral-procession 0.227).
  - 기존 `tracking`(=`reference_centers`) 과 다른 손잡이다. 저건 **조준점**만 옮겨서
    CameraBench 분류로 pan-/tilt-tracking(회전만)이고, 이건 병진이라 시차가 생긴다 —
    tail-/lead-/side-/aerial-tracking 이 그것이다. `info["follow"]["kind"]` 가
    ∠(카메라→subject, subject 속도) 와 고도로 갈래를 **측정**해서 붙인다 (고르는 게 아니다 —
    frame0 이 소스 카메라에 묶여 있어 카메라가 어느 쪽에 서는지는 소스가 이미 정해 놨다).
  - 적용 순서가 중요하다: follow 는 `fit_tau` **앞**이다. 나중에 더하면 `fit_tau` 가 맞춰 놓은
    τ 가 그만큼 빗나간다. 구현은 소스 위치에서 offset 을 빼서 넘기는 항등
    (`|p_shape + off − p_src| = |p_shape − (p_src − off)|`) 이라 `fit_tau` 를 안 고쳤다.
  - `solve_follow_gain` 은 이분법이 아니라 **격자 탐색**(0..1.5, 151점)이다. 목적함수가
    `max_f` 라 단봉이 보장되지 않는다 (subject 가 방향을 꺾으면 국소 최소가 둘 이상).
  - g 는 영상마다 풀어야 한다. parkour/truck-pose/trumans-bedroom 은 `g*≈0` 이라
    저절로 기존 동작으로 돌아간다 (소스가 회전으로만 따라갔다 — 따라가면 오히려 나빠진다).
  - `solve_follow_gain` 가드 **2단**. `moving`(1차) → `min_benefit=0.20`(2차). `auto` 에만 걸리고
    명시적 `--follow_gains 1` 같은 요청은 그대로 통과한다.
    - `min_benefit` 은 **크기 가드**다. 정적 노드도 `center_smooth` 가 OBB fit 노이즈로 떨고
      (snowboard `stat_0` 0.024u vs 동적 0.44u) 그 떨림에도 argmin 은 걸려서, 가드가 없으면
      격자 끝까지 밀려가 **분할 노이즈를 증폭한 흔들리는 카메라**를 만든다 — 그러고도 τ 는
      1.9441 → 1.7798 (8.5%) 밖에 안 줄어 어차피 `tau_saturated` 다.
    - 그런데 크기만으로는 못 막는다. 카메라가 **큰 평면을 훑으면 보이는 부분이 옮겨가** OBB
      중심이 벽을 따라 미끄러진다: truck-pose `stat_0`(graffiti, 벽) wobble 0.535u 로 τ 를
      26.6% 깎아 `min_benefit` 을 통과했고, fashion-walk `stat_9`(tree) 는 path_len 13.602u /
      drift 4.692u 다 (둘 다 정적 물체). 새던 곳: parkour `stat_2/3/5/8` 240행 + truck-pose
      `stat_0` 60행. 그래서 `node["moving"]` 플래그로 자르는 1차 가드를 앞에 뒀다 —
      tracking shot 은 정의상 *움직이는* subject 를 따라가는 것이다.
  - `solve_follow_gain(hi)` 를 1.5 → **2.5** (steps 151 → 251). 1.5 는 실측에서 **상한이
    물렸다**: fashion-walk dyn_0 g*=1.96 / dyn_1 g*=1.88 / dyn_2 g*=2.05 라 1.5 에서 잘려
    τ 가 0.165/0.110/0.233 에 멈췄고 (진짜 최소 0.135/0.073/0.206), dyn_1 은 τ*=0.10 칸을
    그것 때문에 놓쳤다. g*≤1.13 인 snowboard/snow-bike/funeral-procession 엔 no-op.
  - 두 수정 후 재생성(`--bank_dir follow_bank2 --follow_gains 0 auto --aim_keyframes 0 3`):

    | video | nonzero | static nonzero | g* | dropped@g>0 |
    |---|---|---|---|---|
    | truck-pose | 60 → **0** | 60 → **0** | 1.50 → — | 84 → **0** |
    | parkour | 356 → 116 | 240 → **0** | 0.04~0.47 → 0.47 | 364 → **28** |
    | fashion-walk | 232 → **260** | 0 → 0 | 1.50 → **1.88~1.96** | 56 → **28** |
    | snowboard | — | 4 → 0 (explicit g=1 행) | **0.91 유지** | — → **0** |

    parkour 는 selected 가 644 → 532 로 줄었다. 정적 anchor 행이 g=0 으로 돌아가면서
    `tau_saturated`(src_self_tau 0.745) 로 떨어진 것 — 애초에 성립 안 하던 변이다.
  - CameraBench 갈래 실측 7편 (`aim_keyframes 0 3`, `trackings lock`): snowboard side 864 /
    snow-bike tail 288 / funeral-procession lead 664 + side 144 / fashion-walk tail 116 +
    side 116 / parkour side 236 + tail 60 + lead 60 / truck-pose 0 / couple-rocks 0.
    **`aerial-tracking` 은 0건** — `start_mode=source_frame0` 이 frame0 을 소스 카메라에
    묶으므로 elev ≥ 45° 가 구조적으로 안 나온다. couple-rocks 0건은 정상 (subject 이동
    0.126u, τ 절감 0.4% → `min_benefit` 가드).
- **`--aim_keyframes` / `--keyframe_aim` / `--keyframe_ease`
  (`CinemaTraj/decode/build_poses.py` + `scripts/sample_camera_bank.py`)** — 조준을 매 프레임이
  아니라 **sparse keyframe** 에서만 걸고 사이는 SO(3) 측지선으로 잇는다. 기본 `0` = 기존 동작.
  - keyframe 은 `linspace(0, F-1, N)`. keyframe 0 은 **소스 frame0 회전 그대로**(첫 프레임
    일치 유지), k>0 은 `look_at(anchor + bias)`. 위치는 안 건드리므로 τ 가 변하지 않는다.
  - `keyframe_aim`: `target` / `preset_rel` / `auto`(look_at preset → target, traj preset →
    preset_rel). `keyframe_ease`: `smoothstep` / `linear`.
  - 진단으로 `turn_deg`(frame0 → keyframe 1) 와 `aim_err`(max/med/frame0) 를 찍는다 —
    렌더 없이 볼 수 있는 유일한 조준 눈금이라 half-hfov 와 비교하면 된다.
  - `--aim_keyframes` 가 **뱅크 축**이 됐다 (`nargs="*"`, 기본 `[0]` = 기존 동작). 변이 이름에
    `__k<N>` 이 붙고 CSV 에 `aim_keyframes` / `keyframe_turn_deg` / `keyframe_aim_err_deg`
    3열, `bank.json` 의 `axes` 에 `aim_keyframes` 가 추가된다.
  - snowboard 880변이 실측(`--follow_gains auto --aim_keyframes 0 3 5 9 --trackings lock`):
    `vang` 이 k 전 구간 33.2° 로 동일 = **위치를 안 건드렸다는 증거**. 효과는 preset 의
    `aim` 계열로 갈린다 — `aim="traj"`(pan/truck/pedestal/static_hold_locked)는 k0 에서
    inFrame 0.51 (재조준이 없어 subject 가 절반은 화면 밖) → k3 에서 0.87 로 **프레임아웃을
    고치고**, `aim="look_at"` 은 k0 hole 0.249 → k3 0.173 으로 **hole 을 깎는다**(매 프레임
    경직 조준을 푸는 쪽). k3 가 양쪽 다 최적 (hole 0.168 vs k5/k9 0.184).
    `aim_err` 13~18° 는 half-vfov 16.7° 와 같은 자리라 inFrame 이 1.00 이 아니라 0.90 인 이유다.
- **`--bank_dir` (`CinemaTraj/scripts/sample_camera_bank.py`)** — 뱅크 출력 폴더. 기본 `bank`
  = 기존 경로. 이미 돌려둔 뱅크를 덮어쓰지 않고 축을 바꿔 돌리기 위한 것.
- **`CinemaTraj/scripts/viser_cloud.py`** — `lbm/cloud.py` 의 4D point cloud (`cloud.npz`) 를
  viser 로 띄운다. 정적 점(`visible.sum(1) > 1`)은 통째로, 동적 점(`== 1`)은 프레임 슬라이더로
  갈아끼운다 — 섞어 띄우면 움직이는 물체가 49겹으로 번져서 아무것도 안 보인다.
  - 왜: 게이트·hole·τ 는 전부 `render_frame` 의 2D 렌더에서 나오는데, 그것만 봐서는
    "카메라가 이상한가 / 점군이 이상한가"를 못 가른다. 3D 로 띄우면 깊이 shell 이 찢어졌는지,
    plan 카메라가 shell 안쪽(=벽 속)에 들어갔는지가 눈으로 갈린다.
  - 규약의 단일 출처는 `latentcam/scripts/viewer/viser_val_cameras.py` — `add_frustums`/`_GL2CV`
    를 재구현하지 않고 `sys.path.insert` 로 가져온다. cloud world 는 OpenCV 라 `--up` 기본이 `-y`.
  - `--max_static` (기본 1.5M) / `--max_dyn_per_frame` (60k) 로 브라우저 전송량을 깎는다.
    snowboard 는 전체 43.8M (정적 37.3M / 동적 6.5M) 이라 안 깎으면 브라우저가 죽는다.
  - `--bank`/`--variant` 로 뱅크 `poses.npz` 의 plan 카메라를 주황 프러스텀으로 겹쳐 볼 수 있다.
- **`--temporal_edges` (`CinemaTraj/scripts/build_scene_graph.py` + `scene_graph/relations.py`)** —
  엣지에 프레임별 거리를 싣는다. 기본 off = 기존 시간 불변 엣지 그대로 (JSON 비트 동일 확인).
  - 왜: 노드는 원래부터 동적이었지만(`track.center_smooth` (F,3), `obb.node_obb_at`) 엣지는
    `dist_u` 스칼라 하나뿐이었고 그것도 **각 노드의 ref 프레임**(`frames[0]`) 기준이라
    ① 서로 다른 시각의 두 위치 사이 거리를 재고 ② **도중에만 가까워지는 쌍이 엣지로 안 올라온다**.
  - 새 필드: 엣지에 `dist_u_t` (F,) / `dist_u_min` / `dist_u_max` / `near_frac` /
    `near_intervals` / `observed_intervals` / `observed_frames` / `near_radius_u` / `near_at_ref`,
    노드에 `supported_by_t` / `against_wall_t` (F,). `dist_u` 의 의미는 안 바꿨다
    (하류 `lbm/overlay.py:151` 이 읽는다) — "가장 가까웠던 거리"는 `dist_u_min` 이 나른다.
  - 판정은 두 노드가 **둘 다 관측된 프레임**으로 마스킹한다. `center_smooth` 는 미관측 구간까지
    채워져 있어서(`smooth_centers`) 마스크 없이 재면 사라진 노드의 외삽 위치로
    "가까워졌다"를 만들어낸다.
  - `scene_graph/schema.py:assert_invariants` 에 시간축 길이 == `num_frames` assert 추가.
    짧으면 IndexError 가 아니라 하류가 조용히 엉뚱한 프레임을 읽는다.
  - `lbm/overlay.py` NEIGHBORS 블록이 temporal 엣지면 근접 구간(`frames s-t`)까지 VLM 에 준다.
    `dist_u` 만으론 스쳐 지나간 이웃과 내내 붙어 있던 이웃이 구분되지 않는다.
- **`CinemaTraj/scripts/prep_clip_49.py`** — 임의 mp4 → Lite 규약 49프레임 클립.
  `recon_and_seg_single.py` 는 항상 **center-slice** 하므로(`utils/media.py:233
  slice_center_frames`) 30fps 소스를 그대로 먹이면 49프레임 = 1.6초로 잘려 피사체가 거의
  안 움직이고, 뱅크가 쓰는 `frame_step 3`(10fps) 구간과도 어긋난다. `--stride` 로 먼저
  솎아 넘긴다 (`--stride 0` 또는 프레임 부족 시 전 구간 균등으로 폴백).
- **`CinemaTraj/scripts/make_lite_previews.py`** — recon_and_seg 1편 → `seg_overlay.mp4`
  (dynamic 따뜻한 색 / static 차가운 색 + OBB 2D bbox + 라벨) + `depth.mp4`
  (`depths_to_disparity_video`, sky 제외 정규화). `recon_and_seg_single.py --save_vis` 의
  2×2 격자는 타일이 640×360 으로 줄고 **정적 인스턴스(`seg_instances_static/`)가 아예
  안 들어가서** 사람이 검수할 수 없다.
- **`--hold_fallback` (`trumans_to_recon.py` + `trumans_lite_bank.py`)** — `hold`(전 필드 0)
  preset 을 **움직이는 preset 이 전부 떨어졌을 때만** 채택한다. 기본 off = 기존 동작.
  - 왜: `SOURCE_PRESETS["hold"]` 은 이동이 0 이라 **절대 충돌하지 않아** 항상 verify 를 통과한다.
    선택은 `chosen = passed[0]` 이라, 방이 좁아 움직이는 preset 이 줄줄이 떨어지면 hold 이
    자동 당첨된다. 로그에는 preset 이름이 남는데 실제 클립은 **정지 카메라**다
    (`lbm-preset-names-dont-match-motion` 과 같은 종류의 실패).
  - 실측: stride3 ×3 뱅크에서 hold 이 ok 의 **35%** (arm A 22/63, arm B 22/62).
    frame_step 1 뱅크는 20/130 = 15% 였다.
  - preset 목록 재정렬이 아니라 선택 단계에서 거르는 이유: **RNG 추첨 순서를 안 건드려야**
    `--no_hold_fallback` 이 기존 130클립 뱅크를 비트 동일하게 재현한다.
- **`--blender_retries` (기본 1)** — Blender 가 **시그널로** 죽었을 때(rc<0, traceback 없음)만
  20 s -> 60 s 쉬었다 재시도. 0 이면 기존 동작. 근거는 `FIX.log` 2026-08-24 항목.
- **`trumans_lite_bank.py --min_subject_dist`** — 지금껏 뱅크가 이 게이트를 전달하지 않아
  `trumans_to_recon.py` 기본값 0.80 에 고정돼 있었다. 기본값이면 인자를 안 넘겨 기존 뱅크와
  명령이 비트 동일하다 (`MIN_SUBJECT_DIST_DEFAULT` 로 두 파일의 기본값을 묶어 놨다).
  - stride3 후보 810개 집계: **단독 탈락 사유 1위가 이 게이트**(120개). `min_clearance` 는
    위반 빈도는 1위(75%)지만 단독 사유로는 97개다. 즉 "가장 자주 걸리는 게이트"와
    "그것만 풀면 통과하는 게이트"가 다르다 — 앞의 54후보 표는 최선 후보를
    `(clear_frac, min_clearance)` 로만 골라서 이 축이 안 보였다.
  - job 단위 회수 곡선(arm B, 45 job): `>=0.80` 10 / `>=0.70` 14 / `>=0.60` 20 /
    `>=0.40` 22 에서 포화. **22 job 은 다른 두 게이트조차 통과한 후보가 0** 이라
    문턱으로는 못 살린다 (방 기하 문제).
- 뱅크 manifest(`trumans_lite_bank_v1`) 에 `max_tries` / `hold_fallback` /
  `blender_retries` / `min_subject_dist` 기록 — 전부 뱅크의 preset 분포를 바꾸는데
  지금껏 manifest 에 안 남아 있었다.

### Fixed
- **`trumans_to_lbm_demo.py --movement_vocab crane` 이 확장 어휘를 조용히 꺼뜨리던 문제** —
  생성한 `run_windows.sh` 가 `export STORYBLENDER_MOVEMENT_VOCAB=crane` 을 그대로 내보냈는데,
  LBM 쪽은 `== "extended"` **정확 일치**로만 분기를 켠다 (`Director/director_stage.py:980`,
  `Cinematographer/cinematographer_stage.py:60`). `crane` 이면 crane/orbit/pedestal 문구가 전부
  fallback `static` 으로 떨어지는데 **로그에 아무 경고도 안 남는다**. base 가 아니면 무조건
  `extended` 를 내보내도록 고쳤다 (crane 은 demo 생성 쪽 목록 확장일 뿐이다).
- **`--narrative_json` 서술 뒤에 movement term 을 붙일 때 마침표가 겹치던 문제** — movement term
  구절은 `", and the camera ..."` 로 시작한다. 규칙 라벨은 문장부호 없이 끝나지만 VLM 서술은
  마침표로 끝나 `"...throughout the action., and the camera"` 가 됐다. Director 가 읽는 건 이
  문자열 하나뿐이라 끝 마침표만 떼고 붙인다.
- **`compare_lbm_vs_lite.py` 가 다른 recording 의 clip 과 비교하던 문제** — Lite 뱅크
  `lite_framing.csv` 는 recording 7편이 한 파일에 들어 있는데 `action` 만 키로 읽어서 같은 action
  번호끼리 덮어썼다. 두 열 다 그럴듯한 숫자라 조용히 지나간다 — 실측으로 action 0 이 00add26c
  의 0.9636 이 아니라 다른 편의 0.9952 로 찍혔다. `--recording` 필터(비면 LBM CSV 에서 자동 추출)
  + action 중복 assert 를 넣었다.
- **원본 LBM 의 VLM board 선정 + micro-adjust 루프가 TRUMANS 에서 통째로 안 돌던 문제**
  (`Look-Before-Move/Cinematographer/cinematographer_quality_worker.py` `render_preview`) —
  이 함수만 `scene.render.image_settings.file_format` 을 안 고정해서, TRUMANS `.blend` 가
  들고 온 애니메이션 출력 포맷이 새어 들어가 `bpy.ops.render.render(write_still=True)` 가
  후보 프리뷰 전량 `"Cannot write a single file with an animation format selected"` 로 실패했다.
  → `phase1` 3채널이 `no_rendered_previews` 로 떨어지고 board VLM 호출 0회, 기하 점수 1위
  seed 로 조용히 폴백(rc=0). 릴 6 shot 이 전부 `push_in_arc`/travel 0.000 이던 원인.
  snapshot 에 `file_format` 을 넣고 `"PNG"` 고정 + `finally` 원복. **env gate 없음** — 나머지
  렌더 진입점 3곳(`director_scene_context_builder.py:588`,
  `cinematographer_preview_worker.py:330`, `blender_render_worker.py:1154`)이 이미 무조건 PNG 라
  옵션이 아니라 규약 복구다. 검증: `trumans_00add26c_fx` 재실행 shot 2개에서 후보 프리뷰
  55장, `preview_error` 0건. 상세는 `FIX.log` 2026-08-25.
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
- **`aim="traj"` preset(pan/truck/pedestal/static_hold_locked)이 anchor 를 안 보던 문제** —
  `decode/build_poses.py` 에 `--traj_basis {source,subject}` 추가 (기본 `source` = **기존 동작
  비트 동일**, camel 40변이 재현 오차 0.000e+00). `fit_hole_ladder.py --traj_basis`,
  `emit_bank.py`(뱅크 `fixed.traj_basis`, 예전 뱅크는 `source` 로 폴백)까지 배선.
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - 원인: preset 모양을 **소스 frame0 회전** 위에 얹는데, `aim="traj"` 계열은 매 프레임 조준을
    다시 안 세운다. 소스 광축이 anchor 를 향하고 있지 않으면 끝까지 anchor 를 안 본다.
  - 실측(camel, anchor OBB center 를 소스 frame0 에 투영, half-hfov 14.4°):
    `dyn_0` (u/W,v/H)=(0.499,0.529) off-axis 0.47° / `dyn_1` (0.350,0.357) 4.97° /
    `stat_0` (0.758,0.430) 7.61° / `stat_1` (0.139,0.369) 10.68° /
    **`stat_2` (0.156,−0.004) 12.86°** — 이미 화면 테두리다.
  - 결과 `subject_in_frame` (280변이, `hole_bank_f0share`): `aim=look_at` 160변이 평균 0.9788,
    `aim=traj` 120변이 평균 **0.6148**, 그중 20변이는 49프레임 내내 subject 0픽셀
    (`subject_area_med`=0). preset별 `pan_right` 0.435 (8/20이 0픽셀), `truck_right` 0.538,
    `truck_left` 0.612, `pan_left` 0.635, `pedestal_up` 0.669, `pedestal_down` 0.800.
  - `--traj_basis subject` 는 LBM 방식이다 — LBM Cinematographer 는 후보를 렌더해 구도가 맞는
    pose 를 고른 뒤 그 위에서 VideoEngineer preset 을 돌린다 (preset 모양이 아니라 **기준 회전**을
    고친다). frame0 은 `aim_anchor` smoothstep 이 그대로 되돌리므로 소스 카메라와 여전히 동일.
    τ 는 world 이동량이라 `fit_tau` 도 새 기준 위에서 다시 푼다.
  - ⚠ **`subject` 는 실측에서 더 나빴다. 기본값을 `source` 로 둔 이유가 이것이다.**
    camel 280변이 전량 재적합 (`out/camel/hole_bank_f0share_tb/`):

    | aim | traj_basis | in_frame 평균 | <1.0 | 0px | hole 중앙값 |
    |---|---|---|---|---|---|
    | traj (120) | source | 0.6148 | 73 | 15 | 0.2549 |
    | traj (120) | subject | 0.6308 | 84 | **34** | **0.4506** |
    | look_at (160) | source | 0.9788 | 34 | 0 | 0.2711 |
    | look_at (160) | subject | 0.9817 | 34 | 0 | 0.2914 |

    변이 단위로는 in_frame 좋아진 30 / 나빠진 **42** / 동일 208
    (`stat_1__truck_left__hole0.5` 1.000→0.231, `stat_2__pedestal_up__hole0.5` 0.846→0.154,
    `stat_1__pan_left__hole0.35` 1.000→0.462). anchor 별로는 **frame0 투영이 이미 화면 밖인
    `stat_2`(v/H=−0.004) 하나만** 좋아졌고 (in_frame 0.103→0.455) 그 대가가 hole 0.256→0.759 다.
  - 왜 나빠지나: camel hfov 가 28.8° 다. 기준 회전을 off-axis anchor 쪽으로 돌리는 순간 소스
    시야와의 frustum 겹침이 깨져 hole 이 터지고, 사다리가 훨씬 낮은 강도에서 멈춘다. LBM 은
    Blender 완전 씬이라 hole 개념이 없어 시작 pose 를 자유롭게 옮길 수 있지만, frame0 이 소스에
    못 박힌 우리 설정에서는 회전만 돌려도 겹침이 즉시 손해다.
  - 비교 영상 `out/camel/traj_basis_cmp.mp4` (같은 6변이, 위 `source` / 아래 `subject`).
  - 판정 대기: (A) `fit_hole_ladder.py over()` 에 `subject_in_frame` 게이트 추가
    (`pan_*` 은 hole 이 사다리 전 칸에서 평평해 지금 binding constraint 가 아예 없다) /
    (B) frame0 투영이 화면 밖·가장자리인 anchor 제외 (`--min_frame0_margin`) / (C) 현행 유지.
- **`trumans_lite_bank.py --dry_run` 이 기존 뱅크 `bank_manifest.json` 을 덮어쓰던 문제** —
  dry run 이면 `bank_manifest_dry_run.json` 으로 쓴다.
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - 계획 확인 한 번에 7편 133행 집계(ok 34 + skip_done 96 + fail 3, wall 58.8 min)가 19행
    dry_run 으로 교체됐다. 클립 데이터와 work 의 per-action manifest 는 멀쩡해 130행으로
    재구성했지만, 실행이 성공으로 끝나 로그에는 아무 표시가 안 남는 종류다. `FIX.log` 참조.
- **보행 채굴이 7편 중 5편에서 0건이던 원인 = fps** — `trumans_to_recon.py` 에 `--motion_fps`
  (기본 **30**, `0` 이면 예전처럼 blend fps) 추가.
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - `mine_walk_actions` 는 `step * fps > 0.4 m/s` 로 보행을 판정하는데 그 `fps` 를
    `probe_meta["fps"]`(= blend `render.fps`)에서 받고 있었다. blend fps 는 편마다 **15 또는 25**다.
  - 모션 배열의 실제 레이트는 **30** 이다: `video_render/<seq>.pkl.mp4` 7편 전부 프레임 수가
    시퀀스 길이와 **1:1** 이고 fps 가 **30** 이다. blend 의 15/25 는 렌더 설정일 뿐이다.
    프레임당 이동량 분포는 7편이 사실상 같다 (p90 0.0163~0.0362 m/frame) — 편차는 fps 뿐이었다.
  - 실측 채굴 창 수 (`--walk_max` 무제한): blend fps 8/0/0/0/0/9/0 → **fps 30 에서
    9/13/5/5/11/12/4 = 59**. `--walk_max 8` 을 걸면 8/8/5/5/8/8/4 = **46**.
  - 확인용 영상 `results/2026-08-23_trumans_walk/walk_windows.mp4` (14타일) — 버려지던 창이
    전부 49프레임 순 수평이동 0.82~1.87 m 의 실제 보행이다.

### Added
- **hole 뱅크 dedup canonical (`<bank>/canonical_dedup/`, 50편)** — D68 옵션 ① 적용.
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. **코드 변경 없음** — `emit_bank.py --drop_folded`
  (:375) 를 켜고 `--out_dir` 로 새 디렉토리에 냈다. 기존 `<bank>/canonical/` 은 그대로 둔다.
  - 게이트 천장이 사다리 첫 칸보다 낮으면 hole 0.1/0.2/0.35/0.5 **4단이 같은 궤적**이 된다
    (예: avocado-slice `dyn_0__orbit_left_arc__hole{0.1,0.2,0.35,0.5}` 전부 knob 0.89358 /
    path_len 0.5974 / 실측 hole 0.4087 / `approach_limited`). 태그는 4개인데 카메라는 1개라
    하류가 같은 영상을 4번 생성하고 학습에서 4배 가중된다.
  - **16,520 → 10,563 태그 (−5,957, −36.1 %)**, 편당 med 200.5 / min 35 / max 472.
    `worst_pose_rebuild_error` · `worst_roundtrip_error` 둘 다 전 50편 **0.000e+00**.
  - 감소 폭: elderly-tennis 280→70 (−75.0 %), martian-flag 168→42 (−75.0 %),
    funeral-procession 336→170, parkour 392→204 … woman-pottery 56→43 (−23.2 %).
    정확히 −75.0 % 인 2편은 **모든 조합이 4단을 1단으로 접었다**는 뜻.
  - fold 키 검증: `knob`(5자리 반올림) 과 `knob_raw` 의 fold 수가 전 코퍼스에서 5,957 로
    **동일**(갈리는 영상 0편) — D66 `recover_knob` 사례는 fold 경계에 안 걸린다.
  - 접힌 4행은 `bank.json`/`bank.csv` 에 남아 있어 "이 조합은 hole 0.5 를 못 낸다"는 진단은
    보존된다. 사라지는 건 canonical 태그뿐이다.
  - 남은 문제: dedup 후에도 `translation_degenerate`(이동 0, 전부 `pan_left/right`)가
    **2,300 개 = 21.8 %**. fold 로는 안 접힌다 (단마다 회전각이 다르다). D68 옵션 ③ 판정 대기.
- **뱅크 드라이버에 보행 30 % + subject/anchor 분기 배선**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_lite_bank.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - **`--list_actions` 7열 포맷 파서 수정.** `{idx} {kind} {start} {end} {len} {prop}  {text}` 인데
    예전 5열 파서가 `parts[1]` 을 start 로 읽어 `int("labeled")` 에서 **죽고 있었다**.
    `kind`(labeled|walk) 와 `prop` 을 보존해 manifest 에 싣는다.
  - **`--walk_ratio` (기본 0.30)** — 뱅크 안 보행 비율. 채굴량이 편마다 4~8 개, 라벨이 10~17 개라
    그냥 다 넣으면 편별 보행 비율이 22~44 % 로 들쭉날쭉하다. `n_walk = round(n_labeled·r/(1−r))`
    로 풀고 채굴량으로 상한을 건다. 7편 실측 **133 action (walk 36 = 27 %)**;
    0aa05d5a/1d43e076/4ac2c1b3 은 채굴량이 모자라 23/23/27 % 다 (부족분을 print 에 명시).
  - `--subject_kind` / `--anchor_origin` / `--walk_max` / `--walk_speed` / `--motion_fps` 를
    `trumans_to_recon.py` 로 forward. 보행 인자는 `--list_actions` 와 실행에 **같은 값**이 가야
    한다 — 채굴 개수가 하나만 달라도 목록의 idx 와 `--action <i>` 가 통째로 밀린다.
  - **`--video_suffix`** — 같은 action 을 다른 subject 로 다시 뽑을 때 이름 충돌 방지.
    비우면(기본) `--out_video`/`--work` 를 아예 안 넘겨 기존 96편과 명령이 비트 동일하다.
    suffix 를 주면 work 디렉토리도 `<recording><suffix>` 로 가른다 — 중간 산출물이
    `probe_a03.json` 처럼 **action 인덱스로만** 키가 잡혀서 안 가르면 예전 것을 덮어쓴다.
  - 재현성: 보행은 목록 **뒤에** append 되므로 라벨 action 의 `--action` 인덱스가 불변이고,
    기존 96편은 `--skip_done` 으로 보호된다 (dry-run 으로 명령 문자열 대조 확인).
- **TRUMANS Lite 클립마다 `avg_scale` 저장 — context 로 쓸 때 카메라 이동량을 나눌 분모**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_to_recon.py`,
  신규 `scripts/make_avg_scale_trumans.py`). ⚠ 둘 다 `.gitignore:222` 라 커밋에 안 들어간다.
  - 정의는 `avg_scale_first_cam()` **하나뿐**이고 DL3DV 의 `avg_scale_context_first_cam` 과 같다:
    frame0 카메라 기준, sky 를 뺀 전 픽셀의 `z(u,v)·‖K⁻¹[u+0.5,v+0.5,1]‖` 평균 (pixel stride 2).
    z-planar depth 를 ray 길이로 되돌리는 `‖K⁻¹·‖` 를 빼먹으면 화각 넓은 쪽이 과소평가된다.
  - **centroid 가 아니라 first-cam 인 이유**: 이 값을 쓰는 순간은 클립을 context 로 집어들 때고,
    그때 기준은 frame0 카메라다. 궤적 중앙을 기준으로 재면 같은 씬인데 클립마다 분모가 달라진다.
  - `avg_scale` 은 사람 포함(사용자 정의 "모든 점"), `avg_scale_static` 은 사람까지 뺀 진단용.
    저장 위치는 recon 폴더의 `avg_scale.json` — 소비 시점에 손에 있는 게 work 디렉토리가 아니라
    이 폴더라서다. manifest 는 `render` 블록 아래에 같이 싣는다 (`sky_frac`/`depth_p50` 과 같은
    "잰 값" 계열. top-level 은 `video`/`subject`/`caption` 같은 신원 필드).
  - **`make_avg_scale_trumans.py` 는 뱅크 96편 backfill 용**. 편당 ~3분짜리 재렌더 없이
    이미 있는 depth+카메라만으로 채운다. `--from_render` 는 렌더 원본 float32 `.npy`,
    기본은 recon 폴더의 float16 EXR. **두 경로 실측 차이 ≤ 0.0002 %** (4편 대조) — 2 m 대에서
    float16 간격이 ~1 mm 라 평균에 안 남는다. `--patch_manifest` 로 manifest 도 같이 갱신.
  - **뱅크 96편 실측**: min/med/max **1.476 / 2.365 / 4.090 m**, `static/all` 0.9937~1.2553,
    전체 spread **2.77×**. 이 2.77× 는 **recording 간이 아니라 recording 안**에서 나온다 —
    recording 별 median 은 2.136~2.925 m 로 **1.37×** 밖에 안 벌어지고, 한 recording 안의
    spread 가 1.46~2.17× 다. 즉 분모가 방 크기가 아니라 클립별 카메라-subject 거리를 재고 있다.
- **TRUMANS Lite subject 를 사람 전용에서 {human, event, object} 로 일반화**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_scene_probe.py`,
  `trumans_to_recon.py`). ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  probe `--subject_kind {human,event,object}` + `--prop_names`,
  orchestrator `--subject_kind {human,auto,event,object}`(기본 `human`) + `--prop_names`.
  - `human` = 사람 mesh union (기존) · `event` = 사람 ∪ 상호작용 소품 · `object` = 소품만.
    `auto` 는 action text 에 소품이 잡히면 event, 아니면 human 으로 떨어진다 — 미매칭의
    대부분(전체 라인의 20.9 %)이 stand up · sit down · squat · lie down 처럼 **소품이 없는 게
    맞는** 동작이라서다. 반대로 `event`/`object` 를 **명시**했는데 못 찾으면 에러다
    (조용히 human 으로 떨어지면 "object 뱅크"에 사람 클립이 섞인다).
  - **소품 목록의 출처는 recording 폴더의 `obj_list.txt`** (66편 중 **61편**에 존재).
    한 줄짜리 파이썬 리스트 리터럴이고 blend 오브젝트 이름과 그대로 맞는다
    (`['cup_01', 'oven_base_01', 'oven_door_01', 'book_right_01', ...]`).
    ⚠ `object_list.npy` 는 **쓸 수 없다** — (35,) 짜리 movable-chair 변종 목록일 뿐이고,
    `action_label.npy` (F,10) 도 카테고리만 있지 인스턴스가 없다. 둘 다 확인 후 기각.
  - `PROP_ALIASES` 는 61편의 stem 을 **전부** 덮는다 (미등록 stem 0건, assert 로 노출).
    `PROP_VERBS` 는 목적어가 문장에 없는 표현용 — 이걸 넣기 전 미스 상위가 전부 여기였다
    (`drink water` 271줄 / `write` 124 / `make a call` 24 / `type` 14).
    9,488 라인 실측 커버리지: 단일 매칭 **69.2 %** / 모호 6.5 % / 무매칭 24.3 %
    (verb 별칭 없이는 67.3 / 3.7 / 29.0). 모호할 땐 **그 recording 이 실제로 가진 소품**으로
    먼저 좁히고 (대부분 여기서 풀린다) 그래도 남으면 사전순 첫 stem — 재현성 때문이다.
  - `oven` 처럼 한 소품이 `_base`/`_door` 로 쪼개진 경우 stem 으로 묶어 **union** 으로 다룬다.
    부품만 넘기면 문이 열릴 때 subject 가 문짝만 따라가 프레이밍이 튄다.
  - **바닥과 키 눈금은 subject 를 따라가지 않는다.** `floor_z` 는 항상 사람 발에서 재고
    (소품 AABB 최저점은 책상 상판이라 바닥이 0.7 m 위로 잡힌다), `human_height` 도 사람 값
    그대로다 (컵을 찍는다고 카메라 고도 눈금을 15 cm 로 줄이면 안 된다). subject 자체 크기는
    `subject_height` 로 따로 싣는다.
  - manifest 에 `subject{kind,prop_names,prop_stem,anchor_origin,human_height,subject_height}`
    와 `caption{target,event}` 추가, `source_camera.kind` 는
    `synthesized_<kind>_anchored`. `--list_actions` 에 `prop` 열과 obj_list stem 목록 추가.
  - **human 경로 비트 동일 실측** (00add26c a17, 사전/사후 probe JSON 대조):
    `candidates` 576개 완전 일치, `human_track` 의 **기존 필드 전량 일치**, 추가된
    `aim_points` 는 전 프레임에서 `body_points` 와 같다 (= 예전 폴백과 동일 값).
  - event 스모크 (a10 `Open the oven with the left hand` → `oven_base_01`+`oven_door_01`):
    subject union 15 member, 앵커가 사람 단독 대비 **y +0.285 m** 오븐 쪽으로 이동,
    clear 1.00 / clearance 0.372 / subject_dist 1.550 로 게이트 통과.
- **`scripts/audit_trumans_pkl_camera.py` — 실제 TRUMANS 카메라 2편이 보행에 어떻게 반응하는지 잰다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/audit_trumans_pkl_camera.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  `<seq>_camera_pose.pkl` 은 프레임 인덱스 → `{location, rotation}` dict 다.
  49프레임 창을 사람 보행 비율로 갈라 재면:

  | 창 | 00add26c (206 / 15창) | 0aa05d5a (143 / 11창) |
  |---|---|---|
  | still (보행 ≤10 %) | cam net **0.440** m, dt 0.0122 | cam net **0.548** m, dt 0.0137 |
  | walk (보행 ≥80 %) | cam net **1.314** m, dt 0.0309 (사람 1.250 → 추종 **1.05×**) | cam net **0.588** m, dt 0.0141 (사람 1.046 → **0.56×**) |

  - **그동안 인용하던 "net 0.58~0.62 m" 는 still 창이 압도적으로 많아 나온 정지 통계**였고
    보행 구간에 적용하면 안 되는 값이었다. 실제 카메라 2대는 보행에 정반대로 반응한다 —
    하나는 거의 1:1 추종, 하나는 사실상 무반응(walk 0.588 ≈ still 0.548).
  - 채굴한 보행 클립 실측 net **1.194** m / dt 0.0283 은 **과한 게 아니라** 00add26c 의 추종
    카메라를 재현한 것이다. `--track_gain 0.6` 은 두 실제 모드 사이에 앉는다. 조정 안 함.
  - `trumans_to_recon.py` 의 docstring 표와 요약 print 를 창 종류별 기준으로 갈랐다.
- **`trumans_to_recon.py` 보행 pseudo-action 채굴 — TRUMANS 라벨에 없는 보행을 모션에서 캔다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_to_recon.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  `mine_walk_actions()` + `--walk_actions`/`--no_walk_actions` (기본 on) ·
  `--walk_speed 0.4` · `--walk_frac 0.8` · `--walk_smooth 9` · `--walk_max 8`.
  - **왜**: `Actions/*.txt` 9,488 라인 어휘에 보행이 **0건**이다 (stand up 960 / sit down 736 /
    pick up · put down · write · open · close 뿐. `walk|go to|move to|approach|run|wander|navigat`
    히트 0, `turn` 54건은 전부 microwave·oven on-off). 그런데 모션에는 수평속도 > 0.4 m/s 인
    프레임이 **19.0 %** 있고, 49프레임 창의 80 % 이상이 보행인 구간이 blend 66편 씬에
    **16,515개 / 625 시퀀스**, 그 중 **81 % 가 라벨 구간과 전혀 안 겹친다** (= action 사이 이동).
    즉 라벨로 클립을 고르는 한 보행은 후보에 **들어올 수가 없었고**, 그래서 뱅크 96편이 전부
    제자리 조작이라 subject 가 사실상 정지 앵커였다. 49프레임 순 수평이동 median 1.03 m.
  - ⚠ **축 함정: TRUMANS SMPL-X 배열은 y-up 이다** (`human_transl` std x 0.595 / **y 0.149** /
    z 1.101, joints 프레임당 span x 0.53 / **y 1.52** / z 0.53 = 신장). blend 씬은 z-up 이라
    헷갈린다. 수평면은 **(x, z)** 이고 `[:, :2]` 로 재면 가장 넓은 축을 버리고 수직 bob 을 섞어
    보행 비율이 **19.0 % → 7.5 %** 로 절반 이하가 된다 (fps 25 동일 조건 재측정.
    실제로 한 번 그렇게 틀렸다). fps 도 30 이 아니라 **blend 실측 25** 다 (30 으로 가정하면
    24.7 % 로 부풀어 나온다) → `probe_meta["fps"]` 사용. 두 오차가 겹쳐 처음엔 11 % 로 봤다.
  - **인덱스 호환**: pseudo-action 은 목록 **뒤에** append 한다. 앞에 끼우면 `--action <i>` 가
    밀려 시드 키 `recording|action|seed` 가 바뀌고 기존 96편이 재현이 안 된다.
    실측 (`00add26c`): labeled 0..15 는 예전과 동일, walk 16..23 (net 1.24~0.96 m).
  - `--list_actions` 에 `kind` 열과 walk 창의 `net`/`frac` 을 같이 찍는다.
- **`scripts/audit_lbm_distance.py` — 원본 LBM 카메라의 거리를 미터로 재서 Lite 반경과 같은 자에 올린다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/audit_lbm_distance.py`).
  LBM 출력에는 반경이 없다 — `distance_label` 과 후보 id 의 `d0`/`d5` 버킷 토큰뿐이라
  Lite 의 `r ∈ {1.5, 2.2, 3.0, 4.0} m` 와 비교가 안 됐다. blend world 에서 직접 잰다.
  - `d_focus` = 카메라 → `primary_focus_id` 중심, `d_human` = 카메라 → 사람 중심.
    **둘을 따로** 재는 게 요점이다. LBM 의 focus 는 40/42 가 소품이라 `d_focus` 만 보면
    적당한 거리로 보인다. shot_id ↔ TRUMANS action id ↔ Lite `manifest_a<NN>` 로 잇는다.
  - 거리만으로는 "가깝다"를 못 가려서 `frame_h_m` (= 피사체 거리에서 화면이 덮는 세로 미터,
    Blender 기본 sensor 36 mm · `sensor_fit AUTO`) 와 `person_fill_v` (= 사람 키 / `frame_h_m`,
    `> 1` 이면 세로로 잘림) 을 같이 낸다. Lite 는 `fx` 666.667 px / 960 → **25.0 mm 고정**.
  - 실측 (LBM 채택 42 대 vs Lite 96 편): `d_human` median **2.16 m vs 1.69 m** — LBM 이 오히려
    **더 멀다**. 뒤집는 건 렌즈다 — `lens_mm` median **41.3 (24~200) vs 25.0 고정**,
    그래서 `frame_h_m` median **1.14 m vs 1.37 m**, `person_fill_v` median **1.40 vs 1.12**,
    `> 1.5` 가 **17/42 (40%) vs 3/96 (3%)**. 즉 "사람이 잘린다"의 원인은 거리가 아니라
    **긴 렌즈 + 소품 겨냥**이다. 최악은 `1d43e076 shot4` (drawer, **200 mm**, d_human 1.72 m,
    frame_h **0.17 m**, fill 7.19) 와 `00add26c shot4/14` (book_left, 65 mm, fill 6.67 / 5.25).
  - `distance_label` 은 미터와 거의 무관하다: medium shot 38대 `d_human` median 2.25,
    close-up 4대 1.67 — 버킷 토큰도 `d5` 28대 / `d0` 7대로 쏠려 있다.
  - 출력 `out/trumans_lite_bank/{lbm_distance.csv, lite_distance.csv}`.
- **`scripts/audit_lite_framing.py` — 뱅크 프레이밍을 "안 가려진 픽셀 / 화면에 투영된 OBB 면적"으로 잰다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/audit_lite_framing.py`).
  그동안 뱅크 품질을 subject 의 **절대 화면 면적**으로 봤는데 그건 shot size 지 품질이 아니다 —
  wide shot 은 작아도 좋은 그림이다. 판정 축을 보이는 비율로 바꾼다.
  - **분모는 화면 안 bbox 다** (사용자 지정). 전체 hull 을 분모로 쓰면 close-up 에서 박스가
    화면 밖으로 나간 것까지 "가려졌다"로 읽혀 순위가 다시 shot size 를 따라간다.
    잘림은 `crop_keep` 으로 **따로** 찍되 곱하지 않는다 (`--include_crop` 으로 곱셈 복원).
  - `framing` 은 픽셀마다 광선-AABB 입사 깊이 `t_enter` 를 구해 렌더 depth 와 비교한다.
    사람 표면은 박스 **안**(depth ≥ t_enter)이라 자기 가림으로 안 잡힌다.
  - `crop_keep` 은 **해석적 면적**이다 (hull → Sutherland-Hodgman 클리핑 → shoelace).
    `cv2.fillConvexPoly` 로 세면 분모까지 화면에 잘려서 **잘림 자체가 안 보인다**.
  - `--preview` 로 판정 근거 영상 (청록 hull = 화면 안으로 접어 그림, 빨강 = 가려진 픽셀).
  - 편향: OBB 는 실루엣보다 크므로 가림을 **과대** 보고하는 쪽 — 순위용이지 절대 기준이 아니다
    (`audit_bank_geometry.py` 의 OBB 가림과 같은 단서).
  - Lite 뱅크 95편 실측: `framing` median 0.973 / min 0.520, `< 0.7` 6편 / `< 0.9` 24편.
    `crop_keep` 은 median 0.786 / min 0.437 로 여전히 낮지만 순위에는 안 들어간다
    (corr(framing, crop) = **−0.371** — 오히려 약한 역상관이라 shot size 축과 분리됐다).
    반경별 `framing` median 은 1.5 m 0.983 (72편) / 2.2 m 0.913 (16) / 3.0 m 0.892 (5) /
    4.0 m 0.625 (2) — 멀수록 사이에 가구가 낀다. `crop_keep` 은 반대로 0.762 → 1.000 단조.
- **`scripts/trumans_scene_probe.py` — `.blend` 씬 기하를 headless Blender 광선으로 잰다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_scene_probe.py`).
  LBM-Lite 의 G1(`lbm/gates.py`)은 DA3 depth shell 위에서 "관측된 표면보다 뒤인가"로 벽 속을
  *근사*했다. TRUMANS 는 씬 전체가 mesh 라 진짜 광선으로 판정할 수 있다 — full-house 씬이라
  카메라를 아무데나 두면 벽 안에 박히고, 그건 depth shell 에서는 안 잡히던 실패 모드다.
  사람 가슴 기준 (방위각 × 고도 × 거리) 격자마다 `clear`(시선 가림) / `clearance`(6방향 최단
  히트 — 벽 속과 벽에 붙음을 한 값으로) / `floor_drop`(공중·지하) 을 재서 JSON 한 장으로.
  `human_track` 은 루트 본 3축을 **다 실어 보낸다** (SMPL-X 정면 축을 하드코딩하면 리그가
  바뀔 때 조용히 틀린다 — 걷는 방향과 맞는 축을 orchestrator 가 고른다).
  - 실측 제약: 사람은 ARMATURE `zzy3` 이고 **그 자체는 아무것도 렌더하지 않는다**(deform 하는
    mesh 12개가 그려진다). `.blend` 는 `scene.frame_step = 2` 로 저장돼 있어 명시적으로 1 로
    되돌린다. `--cycles` 로 시작하는 CLI 플래그는 금지 — Cycles 애드온이 argv 를 prefix-match
    로 훑어서 실행이 통째로 죽는다.
- **`scripts/trumans_to_recon.py` — TRUMANS action 구간 하나 → LBM-Lite 입력**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_to_recon.py`).
  `scene_graph/io.py:load_scene` 규약(Vista4D 배포본 포맷: RGBD + 카메라 + SAM3 track)을 TRUMANS
  는 **아무것도** 주지 않는다. `.blend` 를 직접 렌더해서 만들면 depth 와 사람 마스크가 추정치가
  아니라 **정답**이 된다. probe(격자) → 궤적 합성 → probe(검증, 49 pose 프레임별 재검사) →
  gt_render(EEVEE 16spp RGB + Cycles 1spp depth/index) → 변환 5단계.
  - **소스 카메라를 합성하는 건 선택이 아니라 데이터 제약**이다: `<seq>_camera_pose.pkl` 은
    recording 67편 중 **2편**(00add26c, 0aa05d5a)에만 있고, `.blend` 안의 CAMERA 4개는 **전부
    정지**(`anim=False`, constraint/parent 없음)다. 정지 카메라를 소스로 쓰면 시차 0 이라 Lite 의
    τ 축과 view-angle 축이 통째로 무의미해진다. 그래서 있는 2편의 통계에 맞춘다 — 프레임당
    `|dt|` median **0.0128 m**, 49프레임 net `|dt|` median **0.58~0.62 m**, z-span **정확히 0**
    (높이 일정), yaw 는 사람 추종 360°. 즉 "높이 고정 + 사람 추종 + 느린 호".
  - 규약: `cam_c2w` 는 OpenCV, `cam_c2w[0] = I` 재앵커(rigid 라 metre 스케일 보존).
    depth 는 Blender z-planar metre 인데 `utils.media.load_depths` 기본이 float16 이라 배경
    sentinel `1e10` 을 그대로 두면 inf 가 된다 → `--sky_depth`(기본 1000 m) clamp + `sky_mask`.
    사람 = object pass index 1 → `dynamic_mask` + `seg_instances` track 1, 나머지는
    `seg_instances_static/`.
- **`scripts/trumans_lite_bank.py` — recording 여러 편을 action 단위로 쪼개 배치**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_lite_bank.py`).
  `trumans_to_recon.py` 는 action 하나짜리이고 실패가 전부 assert 다. 97 action 을 돌리면
  **실패하는 action 이 정상**이다 — 사람이 벽에 붙어 있거나 좁은 화장실에 있으면 카메라를 놓을
  자리가 물리적으로 없다. 그래서 action 하나를 subprocess 로 격리하고 사유를
  `bank_manifest.json`(`trumans_lite_bank_v1`) 에 적은 뒤 다음으로 넘어간다.
  - 기본 `--workers 2`: RGB 가 EEVEE(GPU) 라 Blender 6편 동시 실행에서 `libnvidia-eglcore` 안에서
    crash 했다 (2026-08-23 LBM 4ac2c1b3). 메모리가 아니라 GPU 컨텍스트 경합이다.
  - `--min_action_frames`(기본 12): 너무 짧은 action 은 49프레임을 채우려 앞뒤로 늘리면 사실상
    옆 action 이 된다.
- **`scripts/concat_videos.py` — mp4/프레임 디렉토리를 시간축으로 잇는다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/concat_videos.py`).
  `stack_videos.py` 는 공간축(격자)만 붙여서, "한 recording 의 shot 들을 한 편으로"가 안 됐다.
  Editor 가 내는 `exports/final_edit_v1.mp4` 는 편집본이라 shot 을 잘라낸다(실측 00add26c:
  렌더 프레임 314 → final_edit 218). `--labels` 로 구간마다 좌상단 이름표, `--gap_frames` 로
  구간 사이 검은 프레임.
  - `--inputs` 는 **PNG 프레임 디렉토리도 받는다.** LBM VideoEngineer 가 떨구는
    `clips/<shot>/<cam>/clip.mp4` 는 실측 결과 **전부 프레임 1장**이라(6편 30개 clip 전량)
    포스터와 다를 게 없다. 실제 렌더는 옆의 `renders/<shot>/<cam>/frames/frame_*.png` 다.
    정렬은 파일명 끝 숫자 기준 — 렌더가 stride 2 라 `frame_0001,0003,…` 로 띄엄띄엄이고
    문자열 정렬이면 `0100` 이 `0099` 앞에 온다.
  - 부수 실측: 그 stride 2 때문에 `clips_manifest_v1.json` 의 `duration_seconds` 는
    `target_duration_seconds` 의 **정확히 절반**이다(3.6s→45프레임/1.80s, 2.5s→31/1.24,
    3.0s→38/1.52, 3.8s→48/1.92). 선언된 25 fps 로 틀면 2배속이라 **12.5 fps 가 의도 속도**.
- **`scripts/stack_videos.py --hold_short` — 짧은 타일을 자르는 대신 마지막 프레임으로 정지**
  (`camera_generation/models/Planner/CinemaTraj/scripts/stack_videos.py`).
  기본값은 예전 동작(`--no_hold_short`, 짧은 쪽에 맞춰 자르기) 그대로다 — before/after 두 줄은
  프레임 대응이 생명이라 자르는 게 맞다. 길이가 제각각인 렌더를 격자로 볼 때만 필요하다:
  실측으로 TRUMANS 6편 격자에서 34프레임짜리 `2b4c9b84` 하나가 나머지(164~338프레임)를 전부
  **2.7초로** 잘라냈다.

### Fixed
- **TRUMANS Lite look-at 원점이 probe 격자 원점과 어긋나 머리 꼭대기를 겨눴다 (0.333 m)**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_to_recon.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. `FIX.log` 2026-08-23 항목에 상세.
  `synth_source_path()` 가 look-at 기준점으로 `body_points[0]`(흉부, 발 기준 0.70·h)을
  **무조건** 썼는데, `--anchor_origin` 도입으로 probe 격자 원점은 `obb_center`(0.50·h)가 됐고
  `--aim_bias` 기본 +0.20 은 그 obb_center 위에 얹히도록 정한 값이다. 두 원점이 어긋난 채
  bias 가 더해졌다. 실측 (a17, h=1.763 m): 실제 aim z = chest+0.20h = feet+**0.89 h**
  vs 의도 = obb_center+0.20h = feet+**0.70 h**, 오차 **0.333 m**.
  - **증상이 없는 종류다.** 게이트(clear_frac/clearance/subject_dist)를 전부 통과하고 렌더도
    정상으로 나온다. 사람은 프레임 안에 있되 한 뼘 위를 겨눈 구도가 조용히 생긴다.
  - 조치: probe JSON 이 자기 원점을 말하게 하고 synth 가 읽는다
    (`probe.get("anchor_origin", "chest")`). 키가 없는 예전 JSON 은 chest 였으므로 폴백이 곧
    **기존 뱅크 96편 재현 경로**다. 영향받은 산출물은 smoke `zzw_walk_a17` 1편뿐.
- **소스 자신의 시차가 τ 사다리를 넘는 영상에서 `fit_hole_ladder` 가 assert 로 죽었다 — 이건
  실패가 아니라 "해당 없음"이라 rc=0 + `skipped.json` 으로 바꿨다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/{fit_hole_ladder,emit_bank}.py`).
  `tau_start`(정지 플랜의 τ — 플랜 카메라는 앵커 시작 pose 에 가만히 있는데 소스 카메라가
  날아가서 생기는 시차)가 사다리 꼭대기 1.0 보다 크면 D53 의 "τ 하한 = `tau_start` + 0.02" 가
  사다리 **전 단**을 saturated 로 밀어낸다. `sample_camera_bank` 가 움직이는 조합을 통째로
  `dropped_saturated` 로 빼고 정지 preset 만 남기는데 그마저 saturated 표시라 이분법 루프가
  전부 건너뛰고 `rows` 가 빈다. 예전에는 여기서 `assert rows` 가 터져 **배치 런너가 한 편 때문에
  죽었고, 죽지 않더라도 "실패"와 "해당 없음"이 rc 로 구분되지 않았다.**
  - 실측: `snowboard` `tau_start` **1.9441** (saturated 350 조합) / `snow-bike` **1.3765**
    (210 조합). 둘 다 남은 변이는 `static_hold`/`static_hold_locked` 뿐. 52편 중 **2편 (3.8%)**.
    `tau_start` 를 키우는 건 소스 이동량이 아니라 **이동량/깊이 비**다 — snow-bike 는 snowboard
    보다 소스 이동이 작은데 `z_med` 가 절반(0.8145 vs 1.7510)이라 τ 는 오히려 크다.
  - `fit_hole_ladder` 는 `rows` 가 비면 τ 뱅크에 **생존 조합이 있었는지**로 두 경우를 가른다.
    있으면(=필터 오지정) 예전처럼 assert 로 죽고, 없으면 `hole_bank/skipped.json`
    (`lbm_hole_bank_skipped_v1`: `tau_start`, 사다리, saturated 개수, 남은 preset, `S`, `z_med`)
    을 쓰고 rc=0 으로 나간다.
  - `emit_bank` 는 `bank.json` 이 없고 `skipped.json` 만 있으면 같은 사유를 그대로 찍고 rc=0.
  - 확인: `snowboard`/`snow-bike` 둘 다 `fit` rc=0 → `emit_bank` rc=0.
- **뱅크의 `knob` 이 표시용 5자리 반올림이라 `emit_bank` 가 궤적을 못 되만드는 행이 있었다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/{fit_hole_ladder,emit_bank}.py`).
  `fit_tau` 이분법이 스케일을 **계단으로 양자화**한다 — 손잡이→궤적이 연속이 아니다. 그래서
  5자리 반올림이 계단 경계를 1.8e-6 만 넘겨도 궤적이 통째로 한 칸 커진다. 실측: snow-dog
  `stat_0__truck_right__hole0.1` 의 참 손잡이는 0.228128185878 인데 `0.22813` 으로 올라가면서
  scale 0.01428 → 0.01434, 궤적이 **0.43% 확대**되어 pose 대조가 최대 1.793e-03 로 깨졌다.
  회전은 비트 단위로 같고 시작 pose 도 같은 **순수 균등 스케일** 어긋남이라, pose 대조 assert 가
  없었으면 뱅크가 그대로 나갔을 종류다.
  - `fit_hole_ladder` 가 `knob_raw`(무반올림)를 행·CSV 에 같이 싣는다. `knob` 열은 그대로 둬서
    기존 리더는 안 깨진다.
  - `emit_bank` 는 `knob_raw` 를 우선해 읽고, 없는 **예전 뱅크는 `recover_knob` 이 반올림 구간
    ±5e-6 을 훑어 참 손잡이를 되찾는다** (재현이 `--pose_tol` 안에 들 때만 채택 — 못 찾으면
    조용히 넘기지 않고 그대로 멈춘다). 되찾은 행은 `manifest.json:recovered_knobs` 와 요약표에
    찍힌다. 뱅크 37개를 다시 fit 하는 비용(전체 wall 의 ~80%)을 피하려는 경로다.
  - 확인: snow-dog(1 행 복구) / basketball-four / camel 모두 pose 재현 최대오차 **0.000e+00**.
    basketball-four 의 8.918 어긋남은 별건인 `shape_mult` 버그였고 그 패치로 이미 해결돼 있었다.
- **원본 LBM 의 Blender 워커 2개가 씬 이름 fallback 이 없어 TRUMANS blend 에서 전 shot 이 죽었다**
  (`camera_generation/models/Planner/Look-Before-Move/{VideoEngineer/blender_render_worker.py,
  Cinematographer/cinematographer_quality_worker.py}`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. **vendored LBM 에 낸 유일한 수정이다.**
  LBM 은 씬을 `Scene_<id>_Shot_<n>` / `Scene_<id>` / `Scene <id>` / `scene_<id>` 규약으로 찾는데
  TRUMANS 의 씬 이름은 그냥 `Scene` 이다. 리포에 씬 resolver 가 **5개** 있고 그중 2개만
  fallback 이 없었다:

  | resolver | fallback | 판정 |
  |---|---|---|
  | `Director/director_scene_context_builder.py:224` | 호출자 `:1107` 가 `bpy.context.scene` 로 | OK |
  | `Cinematographer/cinematographer_preview_worker.py:82` | `bpy.data.scenes[0]` | OK |
  | `Cinematographer/cinematographer_quality_worker.py:163` | **없음** | 수정 |
  | `VideoEngineer/blender_render_worker.py:64` | **없음** | 수정 |
  | `VideoEngineer/video_runtime.py:502` | 호출자가 넘긴 이름을 그대로 씀 | OK (실증) |

  → 파일에 **딱 하나의 씬**이 있을 때만(= 선택이 모호하지 않을 때만) 그 씬으로 떨어지는 가지를
  넣었다. `LBM_SINGLE_SCENE_FALLBACK=1` **opt-in 이라 기본 동작은 그대로**다. 두 파일 다 빠져
  있던 `import os` 도 같이 넣었다.
  - `blender_render_worker` 쪽 증상: 앞 단계는 멀쩡히 지나가고 렌더에서만 `missing scene` ×6 →
    `RuntimeError: Blender scene render failed`. 고친 뒤 VideoEngineer 78.3 s 완주, rc=0.
  - `quality_worker` 쪽 증상이 더 고약하다 — **조용히 성공한 척한다.** 워커가
    `{"success": true, "row_count": 16}` 을 내는데 16행 전부 `error: "scene_not_found"` 이고,
    후보 탐색이 아예 안 돌아 `candidate_count_raw_min=0` → 카메라 16대 전원
    `downstream_blocked` → 렌더 0장. 게다가 **quality 모드가 fast 보다 빨라진다**
    (39.8 s vs 86.4 s) 니 시간만 보면 정상으로 오독하기 딱 좋다.
    고친 뒤 Cinematographer 232.7 s, shot 당 후보 raw ~1,200 (16 shot 합 19,235) 로
    탐색이 실제로 돌았고 파이프라인 전체가 rc=0 으로 완주했다.
  - 곁다리로 `--resume-from` 의미도 기록해 둔다(`run_full_pipeline.py:280-364`): 적은 단계의
    **산출물을 재사용**하고 실행은 그 **다음** 단계부터다. VideoEngineer 를 다시 돌리려면
    `--resume-from cinematographer` 다 — `videoengineer` 를 주면 아직 없는
    `video_handoff_v1.json` 을 읽으려다 `FileNotFoundError`.
- **`fit_hole_ladder` 의 `shape_mult` 가 실제로 쓴 배율보다 headroom 배 크게 기록됐다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/{fit_hole_ladder.py,emit_bank.py,
  patch_bank_shape_mult.py}`). ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  모양 확대 루프가 `mult` 를 판정 **뒤에** 곱해서, 루프가 `break` 없이 끝까지 다 돌면
  (= `status == "shape_limited"`) `poses` 를 만든 배율은 `mult/headroom` 인데 행에는 `mult`
  가 실렸다. `emit_bank` 가 그 배율로 되만들면 궤적이 2배가 된다 — 실측 basketball-four
  `dyn_0__pull_out_arc__hole0.5` 에서 `poses.npz` 와 최대 **8.918** 어긋나 rc=1.
  - 마지막 시도에서는 곱하지 않도록 고쳤다. 렌더 횟수·최종 궤적은 그대로고 기록되는 숫자만 바뀐다.
  - `dolly_frac`/`lateral_frac` 을 안 쓰는 preset(orbit 계열, `s_curve`)에서는 배율이 궤적에
    영향을 안 줘서 같은 버그가 있어도 emit 이 통과했다 — couple-hug·bmx-bumps 가 그 경우.
    **즉 이 버그는 `pull_out_arc` 처럼 dolly 를 쓰는 행에서만 드러난다.**
  - 이미 만든 뱅크는 `scripts/patch_bank_shape_mult.py` 로 `shape_mult` 열만 나눠서 고친다
    (fit 재실행 영상당 ~1350 s 회피). 행의 측정치는 작은 쪽 궤적에서 잰 것이라 원래 맞다.
    두 번 돌면 또 반토막 나므로 뱅크 최상위 `shape_mult_semantics: "as_built"` 를 표식으로
    두고, 붙어 있으면 건너뛴다. 고친 뒤 확인은 `emit_bank` 의 "pose 재현 최대오차 0.000e+00".
- **새 소스 1편을 넣을 때마다 `seg_instances` symlink 를 손으로 걸어야 했다**
  (`camera_generation/models/Planner/CinemaTraj/scene_graph/io.py`). ⚠ `.gitignore:222`.
  `recon_and_seg_single.py --save_seg_instances` 는 `recon_and_seg/<video>/seg_instances` 에
  쓰는데 코퍼스 규약은 `eval_data/seg_instances/<video>` 다. 안 걸면 `"seg_instances 가 없다"`
  로 죽었다 (TRUMANS 투입 때 실제로 걸림). → 동적 seg 에 한해 원본 위치를 fallback 으로 본다.
  symlink 가 이미 있으면 그쪽이 먼저 잡히므로 기존 52편은 동작이 그대로다.
- **실내 씬에서 ground RANSAC 이 inlier 문턱에 걸려 조용히 카메라-up 으로 떨어졌다**
  (`camera_generation/models/Planner/CinemaTraj/{scene_graph/gravity.py,scripts/build_scene_graph.py}`).
  각도·"카메라가 평면 위" 조건은 통과했는데 `inlier_ratio ≥ 0.15` 만 못 넘긴 후보가 버려지고,
  대신 실제 바닥과 12~28° 어긋난 카메라 up 이 중력축이 됐다. 52편 중 15편이 fallback 인데
  그중 3편이 이 경우다: `trumans-bedroom` 0.142/12.8°, `basketball-four` 0.132/28.2°,
  `park-selfie` 0.105/11.4°. 나머지 12편은 각도 조건부터 못 넘겨 진짜 fallback 이 맞다.
  → `--weak_inlier_ratio` 로 그 문턱만 낮춰 살리는 가지(`method: "ground_ransac_weak"`,
  confidence 절반)를 넣었다. **기본값 0.0 = 꺼짐** — 51편 코퍼스 결과를 안 바꾸기 위해.
- **51편 확장 파일럿에서 앞단(detection/segmentation/scene graph) 실패 모드 5종 실측 + 수정**
  (`camera_generation/models/Planner/CinemaTraj/{scene_graph/instances.py,scripts/build_scene_graph.py,
  scripts/extract_nouns_vlm.py,scripts/extract_static_nouns.py,scripts/stack_videos.py}`).
  ⚠ `camera_generation/models` 는 `.gitignore:222` 라 커밋에 안 들어간다.
  파일럿 대상은 새 5편 `goat / room-argue / car-roundabout / woman-pottery / parkour`.
  - **① 명사 추출이 SAM3 출력에 순서 의존** — `extract_nouns_vlm.read_authored_keywords` 가
    비교 기준을 `seg_instances/<video>/meta.json` 에서 읽었다. 그건 `metadata.csv:dynamic` 의
    복사본일 뿐인데, "SAM3 를 먼저 돌려야 명사 추출이 된다"는 순서를 만들어 parkour 가 아직
    안 끝난 상태에서 `FileNotFoundError` 로 7편 전체가 죽었다. → **원천 `metadata.csv` 를 직접
    읽는다.** 없는 영상은 빈 목록(비교 기준만 없고 추출은 된다).
  - **② 인스턴스 수 폭발** — 개별 track 은 다 멀쩡한데 수가 터진다: 동적 `car-roundabout` **69개**
    (거리의 차 전부), 정적 `parkour` **56** / `car-roundabout` **54** / `goat` 은 `rock` 하나로
    **31개**. 기존 기각 3종(`max_area_frac`/`visible_frames`/`mean_score`)은 track 을 하나씩만
    보므로 이걸 원리적으로 못 막는다. 해로운 이유는 셋 — G5 상자 여유는 **전 노드 최소값**이라
    상자 수십 개면 자유공간이 잘게 쪼개져 전 방향이 `obb_limited` 로 수렴하고, `rock` 31개는
    사실 지형이라 OBB 자체가 무의미하며, graph 빌드 시간이 track 수에 선형이다.
    → `scene_graph.instances.cap_instances`: keyword 별 상위 K + kind 별 전역 상한
    (`--per_keyword_top_k 5 --max_dyn_nodes 6 --max_stat_nodes 10`, 0 이하면 상한 끔 = 예전 동작).
    순위는 `max_area_frac` — `sample_camera_bank.py --min_area_frac` 이 앵커를 고를 때 쓰는 것과
    **같은 양**이라 상·하류가 안 어긋난다. **버린 것은 전부 `diagnostics.dropped_tracks` 에 사유와
    함께 남는다** (조용한 절단 금지).
  - **③ lift 를 먼저 하고 나서 버렸다** — `car-roundabout` 은 track 123개인데 그중 100개 넘게
    버릴 것을 49프레임씩 unproject 하고 나서 버렸다. 판정에 쓰는 세 양은 전부 2D 마스크만으로
    나온다. → `summarize_tracks` (마스크 `sum()` 만) 로 사전 선별하고 `build_instances(select=...)`
    가 살아남은 track 만 올린다. 사전 상한은 `--prescreen_factor 3` 배로 넉넉히 — 최종 상한은
    merge 뒤에 걸어야 "면적 상위 K" 가 물체 단위로 세어진다(같은 물체 조각 5개가 상위 5칸을
    먹으면 안 된다).
  - **④ 마스크 depth 가 두 덩어리일 때 OBB 가 시선 방향으로 늘어남** — `goat` 의 OBB 투영 크기가
    마스크의 **13.06배**, extent `[0.96, 0.05, 0.82] u` 로 두께가 사실상 0인 납작한 판이었다.
    원인은 마스크 안 depth 가 **완전히 두 덩어리**라는 것: 염소 본체 z 0.62~1.0 에 44k px,
    그리고 **중간이 텅 빈 채로** z 6.5~7.5 에 7000 px (마스크의 **14%**) — SAM3 가 언덕 저편의
    다른 염소 무리를 같은 track 으로 묶었다. 기존 `trim_depth_tail` 은 양끝 1% 를 자르는 대칭
    처리라 14% 를 못 자르고, `radius_outlier_mask` 도 못 잡는다(저쪽도 7000점이라 서로 이웃이다).
    → `trim_depth_mode`: log z 히스토그램에서 **중앙값이 속한 덩어리만** 남긴다
    (`--depth_mode_bins 40 --depth_mode_gap_frac 0.005`, bins 0 이면 끔). 단봉이면 무처리.
  - 회귀 (camel, ④ 적용 전후): subject 는 그대로 `dyn_0` extent 0.137→0.133 / `dyn_1` 0.160→0.145,
    **배경 누출로 부풀어 있던 정적 상자만 줄었다** — `stat_0 fence` 0.746→0.294 u,
    `stat_2 fence` 0.303→0.173, `stat_3 tree` 0.597→0.384 (투영 크기비 2.07→**1.63**).
    전 노드 `size_ok=Y`. G5 를 **느슨하게** 하는 방향이다. 기존 뱅크는
    `out/<video>/{scene_graph_pre_52.json,hole_bank_pre_52/}` 로 보존.
  - **⑤ 광역 표면이 두 단어로 새어 들어감** — `extract_static_nouns.SURFACE_NOUNS` 가 정확일치라
    `brick wall` / `dirt track` / `tile floor` 가 그대로 통과했다(51편 실측). → `is_surface` 가
    **머리단어(마지막 토큰)로도** 판정한다. 51편에서 올라온 `court`/`track`/`field`/`path`/
    `hill`/`beach`/`dirt`/`carpet` 등도 목록에 추가.
  - 파일럿 7편 재실행 실측 (GPU 1~3 병렬, `--no_skip_done`, 오버레이 포함):

    | 영상 | 노드 dyn+stat | dropped | size_ratio 최대 | WALL |
    |---|---|---|---|---|
    | camel | 2+4 | 1 | 1.63 | 273.7 s |
    | avocado-slice | 3+8 | 3 | 2.02 | 271.5 s |
    | woman-pottery | 2+5 | 6 | 2.42 | 207.4 s |
    | room-argue | 3+8 | 0 | 1.59 | 476.1 s |
    | parkour | 1+10 | 42 | nan(꼭짓점이 카메라 뒤) | 422.0 s |
    | car-roundabout | 6+10 | 93 | 1.54 | 535.6 s |
    | goat | 1+5 | 18 | 3.65 | 333.4 s |

    room-argue `pillow` 3개는 수정 전 3.47 / 3.85 / 5.86 → **1.06 / 0.76 / 1.06** (④의 직접
    증거). car-roundabout 은 track 123개 → 노드 16개. **goat 만 남는데 이건 depth 추정기
    한계다** — f=2289 px / 폭 1280 px 의 망원 샷이라 시선 방향 depth 오차가 물체 크기와
    맞먹는다. 안 고친다 (OBB 가 크면 G5/G7 이 보수적일 뿐). 선택지는 `DECISIONS.md` D61.
  - graph 단계 실측: camel **171.7 s** (오버레이 없이) / **273.7 s** (오버레이 포함).
    대부분이 `radius_outlier_mask` 의 cKDTree 로, track 수 × 49프레임에 선형이다.
  - 앞단 전체 실측: VLM 명사 51편 **81.7 s** · 정적 명사 정리 51편 <1 s · 동적 SAM3 ~40 s/편.
- **hole 사다리의 아랫단이 조용히 정지 카메라를 뱉던 문제 (D53) — τ 하한을 소스 시차 기준으로
  올리고, 사다리 바닥을 anchor 의 정지 hole 로 옮기고, `fit_tau` 이분법 해상도를 고쳤다**
  (`camera_generation/models/Planner/CinemaTraj/lbm/presets.py`, `decode/build_poses.py`,
  `scripts/{fit_hole_ladder,sample_camera_bank,emit_bank}.py`). ⚠ `camera_generation/models` 는
  `.gitignore:222` 라 커밋에 안 들어간다. 상세는 `CinemaTraj/DECISIONS.md` D53.
  - 결함이 **셋**이었다. ① τ 는 plan 과 **소스**의 프레임별 간격이라 카메라를 시작 pose 에
    얼려놔도 `tau_start` 만큼 쌓인다 — anchor 무관 **씬 상수**다 (avocado 0.1286 / camel
    0.0042, 각 씬 `parallax_ratio` 0.129 / 0.0046 과 일치). `KNOB_RANGE["tau"]` 하한 0.02 가
    avocado 에서 **도달 불가능**이라 사다리가 손잡이를 내리는 순간 궤적이 사라졌다.
    → `--tau_floor_src` (기본 on): 하한을 `tau_start` 의 소스배수로 잡는다.
  - ② 사다리 단(0.10/0.20/0.35/0.50)이 일부 anchor 의 **정지 hole 보다 낮았다** (avocado
    `stat_1` 0.5895 / `stat_4` 0.6384, camel `stat_3` 0.6482) — 어떤 손잡이로도 못 맞추는데
    코드는 `unreached` 가 아니라 `clamped_low` 로 찍어 "작지만 정상인 카메라"와 구분이 안 됐다.
    → `--hole_mode excess` (기본): 단을 anchor 의 정지 hole **위의 증분**으로 매긴다.
  - ③ (①·② 를 고친 뒤 드러남) 하한을 `tau_start` 바로 위로 올리면 그걸 맞추는 배율이
    `fit_tau` 이분법의 **첫 눈금**(`max_scale/2**iterations` = 4/256 = 0.0156)보다 작아져서
    `lo` 가 0 에 남는다 — **하한을 고쳐도 궤적이 여전히 정지**였다. avocado `stat_1
    pull_out_arc`: 0.020/path 0.000 → (①만) 0.149/path 0.000 → (③까지) 0.149/path **0.012**.
    → `presets.fit_tau(refine_zero=True)`: `lo==0` 이면 `[0, hi]` 에서 이분법을 한 번 더 돌린다.
    플래그는 decision 의 `trajectory.tau_refine` 에 실려 `emit_bank` 재현까지 간다.
  - 뱅크 재생성 전후 (변이 100% 겹침, camel 336 / avocado 392). **정지 궤적(`path_len_u`
    < 1e-6) camel 68 → 48 / avocado 206 → 56** 인데, 남은 48·56 은 **정확히 `pan_left`/
    `pan_right` 변이**다 (순수 회전이라 이동 0 이 정의). 즉 **pan 을 뺀 비-pan 궤적은
    camel 20 → 0 / avocado 150 → 0 으로 전멸**했다. 수정 전 비-pan 정지의 preset 분포는
    camel `straight_ease/push_in_arc/pull_out_arc/rise_reveal/drop_reveal` 각 4,
    avocado 는 8 개 preset 에 16~21 개씩 고르게 퍼져 있었다.
  - 비-pan `path_len_u` median camel 0.1110 → 0.4636 (mean 0.2853 → 0.6595) / avocado
    0.0594 → 0.4341 (mean 0.2398 → 0.6644). 전 변이 기준 hole median camel 0.3063 →
    0.3199 / avocado 0.3482 → 0.3661. `solved` camel 166 → 168 / avocado 182 → 189.
  - `clamped_low` 가 camel 91 → 0 / avocado 149 → 0 으로 사라지고 그 자리가
    `unreached`(camel 0 → 14 / avocado 1 → 33)와 `shape_limited`(19 → 42 / 9 → 47)로
    갈렸다 — **"작아서 못 갔다"가 "무엇이 막았다"로** 바뀐 것이 이 수정의 요점이다.
    `binding` 도 이제 실제 게이트 이름만 나온다 (avocado: hole 189 / none 80 / collision 54 /
    elev 36 / ground 27 / obb 6).
  - **예전 동작 보존 확인**: `--hole_mode absolute --no_tau_floor_src` 로 돌린 16 행을 수정 전
    뱅크와 42 개 공유 열에서 대조 — `knob`/`status`/`binding`/`hole_fraction`/`tau_max`/
    `path_len_u`/`obb_slack`/`elev_abs_max`/`ground_clear` **전부 동일**. 다른 18 셀은
    `subject_area_med`/`near_depth` 의 소수 4째 자리뿐이고, 같은 명령을 두 번 돌려도 같은 두
    열에서 16 개가 달라져 **렌더러 비결정성**임을 확인했다.
  - 곁가지 버그 하나 같이 고침: `bank.csv` 는 따옴표 없이 `",".join` 으로 쓰는데 상태 접미사를
    `status,tau_floor` 로 붙여 avocado 10 행의 뒤 열이 통째로 밀려 있었다 (`binding` 이
    `tau_floor` 로 읽혔다). 구분자를 `+` 로 바꾸고 writer 에 **쉼표 금지 assert** 를 넣었다.
  - 영상: `out/<video>/hole_bank/d53_before_after.mp4` (위=수정 전 / 아래=수정 후).
- **LBM-Lite 디코더에 조준 앵커 옵션 `--aim_anchor` 추가 (기본은 기존 동작 `subject`)**
  (`camera_generation/models/Planner/CinemaTraj/decode/build_poses.py`, `decode/emit.py`,
  `verify.py`). ⚠ `camera_generation/models` 는 `.gitignore:222` 라 커밋에 안 들어간다.
  ⚠ **두 쪽을 다 렌더해 비교한 뒤 사용자가 기존 동작을 골랐다** — 기본값은 `subject` 로
  되돌렸고 산출물도 원상복구했다 (`model_gauge_rmax` camel 0.68801962 / avocado 0.71040781 로
  이전과 정확히 일치). 앵커는 `--aim_anchor {source_frame0,auto}` 로 켤 수 있다.
  - 증상: `plan_sbs.mp4` 의 첫 프레임이 소스와 다르다. 원인은 `aim="look_at"` preset
    (`orbit_left_arc` 포함)이 **f=0 을 포함한 전 프레임**의 회전을 subject 조준으로 덮어쓰기
    때문. `c2w_start` 는 위치만 살아남았다. `τ = |Δp|/z_med` 는 위치 전용이라 `tau[0]=0.0`,
    `view_angle_deg[0]=0.0` 으로 찍혀 지표 어디에도 안 나왔다.
  - 실측 frame 0 회전 각차 camel 3.1837° (forward 축 0.4765° — 사실상 roll) /
    avocado-slice 11.6739° (forward 축 11.4534° — 진짜 재조준). 렌더 hole 은
    소스 pose 0.0100 / 0.0127 → plan f0 0.0395 / **0.3648**. avocado 는 frame 0 이 클립
    전체에서 가장 나쁜 프레임이었다.
  - hole 분해(위치·회전 교차 렌더): 두 씬 다 **회전만 바꾼 쪽**이 위치만 바꾼 쪽보다 구멍이
    크다. camel f48 은 회전 단독 0.4183 / 위치 단독 0.1730 인데 합치면 0.1265 로 내려간다 —
    orbit 이동이 새 조준 방향에 관측을 도로 대준다.
  - 옵션: `--aim_anchor {auto,source_frame0,subject}` + `--aim_ramp_frames 12`.
    **기본 `subject` = 예전 동작.** `source_frame0`/`auto` 를 주면 look_at 조준을 다 세운 뒤
    frame 0 의 조준 오차를 world 회전 하나로 뽑아 smoothstep `w(f)=x²(3−2x), x=f/12` 로
    되돌린다 (f=0 100%, f≥12 0%). smoothstep 인 이유는 f=0 에서 기울기가 0 이라 첫 프레임이
    안 튀기 때문 — 선형이면 avocado 가 f0→f1 에서 0.96°/frame 로 출발한다.
  - 앵커 각도 camel 3.2093° / avocado 11.5555°. 재실행 지표(GPU 1, 49f 1280×720, decision 동일,
    `subject` → `source_frame0`): `hole_fraction` camel 0.0737215 → 0.0704789,
    avocado 0.327682 → 0.280871. `subject_in_frame` 1 / 1 유지, `tau_max`·`max_view_angle_delta`
    ·`jerk_ratio` 불변, 두 씬 다 PASS. 평균 hole 하락은 ramp 12프레임이 싼 구간이라 평균을
    끌어내린 것이고 정상 구간은 그대로다 (avocado f12 이후 0.333 복귀).
  - 부수 변경: ① `look_at` 배열도 ramp 구간은 실제 시선축으로 다시 찍는다 (안 하면 npz 와
    `plan_cam.mp4` 화살표가 회전과 어긋난다) ② roll assert 는 ramp 바깥에만 건다 ③ 지문
    (`decision_fingerprint`)에 `aim_anchor`/`aim_ramp_frames` 추가 — 안 넣으면 이것만 바꿔
    재빌드했을 때 emit/verify 의 stale-poses 가드가 못 잡는다. `emit.py`/`verify.py` 에도
    같은 CLI 를 달았다. ④ `rotation_log`/`rotation_exp` 추가 (scipy 가 env 에 없다).
  - 프리뷰: `results/20260820_aim_lbm_lite/{camel,avocado-slice}/`. 예전 것은
    `results/20260820_lbm_lite/` 에 대조용으로 남겼다.

### Changed
- **`viser_cloud.py` 에 motion 브라우저 (`--banks` / `--no_cloud`) — 슬라이더로 궤적을 갈아끼운다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/viser_cloud.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  기존 `--bank`/`--variant` 단일 고정 경로는 그대로 (`--banks` 를 비우면 예전 동작).
  - 왜: "카메라가 떨린다"를 2D 렌더로는 원인을 못 가른다 — 렌더가 떠는 경로가 셋이다.
    ① plan 위치 고주파(`follow_gain` × subject track jitter), ② 조준 회전, ③ **점군 자체**
    (프레임마다 depth 가 달라 정적 배경이 숨쉰다). 3D 에서 궤적 선을 직접 보면 ①/②는 선이
    지그재그로 보이고, 선이 매끈한데 렌더가 떨면 남는 건 ③뿐이다.
  - 경로는 `add_line_segments` **생꺾은선**이다. `add_spline_catmull_rom` 은 지금 보려는 그
    jitter 를 그리는 단계에서 없애 버린다.
  - 같이 그린다: 소스 경로(회색) · plan 경로(주황) · **subject track**(초록). track 은
    `scene_graph.json` 의 `track.center_smooth` 를 `T_wg` 로 world 에 올린 것 — `follow_gain` 의
    입력이라 여기가 떨면 카메라가 정확히 g 배로 떤다.
  - GUI `motion` 폴더: 슬라이더 + 드롭다운(서로 갱신, 재진입 가드) + 읽기전용 요약. 요약에
    `jerk p95` 를 **plan / source / subj_track 세 개 나란히** 적는다 — plan 값만으로는 크기를
    판단할 수 없다. snowboard 실측 (px/frame³, `|Δ³p|/z_med·fx`): source 86.22 ·
    subj_track 11.69 · plan `s1` 10.64 → `s9` 2.42.
  - `--no_cloud` 는 점군을 아예 안 꺼낸다 (npz lazy). snowboard 43.8M 점 로딩이 수 분이라
    궤적만 볼 때는 즉시 뜬다.
  - `--variant` 이 **부분문자열 AND** 로 바뀌었다 (`--variant s9 k3`). 하나만 주면 예전
    단일 필터와 동일. 축이 5개(preset·τ·follow·smooth·keyframe)라 한 축만 잡아서는 목록이
    안 줄어든다 — 32개 중 `s9`+`k3` 만 남기면 preset 14개가 된다.
  - 여러 뱅크를 합칠 때 라벨에 뱅크 이름을 접두한다 — 서로 다른 뱅크에 같은 `variant_id` 가
    흔하다(같은 preset·τ 를 축만 바꿔 되풀기 때문). `bank.json` 이 없는 옛 뱅크도 열린다
    (`anchor_id` 는 `poses.npz` 에서 받는다).
- **`render_bank_videos.py` 타일 캡션에 `g`/`s`/`k` 축을 붙인다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/render_bank_videos.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  기존 캡션은 `preset` + `anchor`/`τ`/`hole`/`path` 뿐이라 `follow_gain`·`follow_smooth`·
  `aim_keyframes` 축으로 만든 A/B 영상에서 **어느 타일이 어느 쪽인지 눈으로 못 갈랐다**
  (k0 vs k3 영상을 보내고서야 캡션에 `k` 가 아예 없는 걸 발견). 0 이면 안 붙이므로 그 축을
  안 쓴 뱅크의 캡션은 예전 그대로다. `s` 는 `g` 가 0 이 아닐 때만 붙인다.
- **`trumans_scene_probe.py` 반경 격자에 1.0 m / 5.5 m 추가 — 뱅크에 medium shot 이 0건이던 걸 메운다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_scene_probe.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  `--radii` 기본값 `[1.5, 2.2, 3.0, 4.0]` → `[1.0, 1.5, 2.2, 3.0, 4.0, 5.5]`.
  Lite 렌즈가 25 mm 고정이라 `fill = subject_h / frame_h ≈ 1.878·(subject_h/1.521) / r` 이고,
  하한이 1.5 m 면 fill 이 1.4 를 못 넘는다 → 뱅크 96편 프레이밍 분포에서 **medium(1.75–2.60) ·
  medium close-up · close-up 이 전부 0건**이었다 (`out/trumans_lite_bank/lite_distance.csv`).
  - 실측 (00add26c a00/a04/a07 × 96격자, `clear`+`clearance≥0.35`+`floor_drop≥0.20` 통과율):
    **r=1.0 75.0 / 45.8 / 53.1 %** 로 전 반경 중 **1위**다 (r=1.5 는 46.9 / 32.3 / 24.0 %).
    사람 1 m 앞은 대개 빈 바닥이라 그렇다. fill 은 1.55 / 1.88 / 2.08 로 medium 칸을 채운다.
  - **r=5.5 는 이 실내 3편에서 usable 0 건** (r=4.0 도 288 중 1건). full-house 실내라 5 m 는
    항상 벽 너머다. 그래도 격자 96개 추가 비용이 ~0.9 s 뿐이라 **지우지 않고 남겼다** — 넓은
    recording 이 있으면 저절로 쓰이고 없으면 저절로 빈다. 즉 TRUMANS 실내에서 도달 가능한
    fill 은 **~0.5 ~ 2.1** 이고, extreme wide 는 샘플러 문제가 아니라 구조적으로 없다.
  - 전체 파이프라인 스모크 3편 통과 (a00 `az 60 elev 25 r 1.0 hold` clear 1.00 clearance 1.011 /
    a04 `az 240 elev 0 r 2.2` / a07 `az 255 elev 12 r 2.2`), usable 풀 125 / 91 / 84 of 576.
  - **같이 넣은 안전장치: `subject_dist` 열 + `--min_subject_dist` (기본 0.80 m) 게이트**
    (probe verify 패스 + `trumans_to_recon.py`). `clearance_of` 는 **사람 히트를 일부러 무시**하고
    (사람은 장애물이 아니라 피사체) `line_of_sight` 는 사람에 맞아야 `clear=True` 라, 카메라가
    피사체를 뚫고 들어가는 걸 검사하는 열이 **하나도 없었다**. 반경 하한이 1.5 m 이던 동안엔
    `push_in`(`dradius −0.55`) 도 0.95 m 라 안 드러났지만 r=1.0 을 넣으면서 열린 구멍이다.
  - ⚠ **정직한 실측: 이 게이트는 아직 한 번도 안 걸렸다.** `push_in` 6후보 재검증에서
    r=1.0 후보 2개는 start 1.000 / 0.993 → min **0.842 / 0.859** 로 0.80 을 넘겨 PASS 했다
    (기각 1건은 기존 `clearance` 게이트, 0.243 < 0.35). 내가 예측했던 "0.60 m 까지 들어간다"는
    **틀렸다** — 격자 `radius` 는 흉부 기준 **3D 거리**(elev 포함)인데 `synth_source_path` 의
    `rad = max(0.6, radius + dradius·ease)` 는 **수평 반경**이고, `--track_gain 0.6` 부분 추종이
    거리를 더 벌린다. 0.80 은 예전 반경 집합에선 절대 안 걸리는 값이라(1.5 − 0.55 = 0.95)
    기존 96편 재현성은 그대로고, 실효 하한 0.84 와의 여유는 0.04 m 뿐이다.
- **`trumans_to_recon.py` 시작 pose 재시도 순서를 (반경 × 방위각) 으로 층화 — 뱅크가 1.5 m 코퍼스가 되는 걸 막는다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_to_recon.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  기존은 **방위각만** 8섹터로 층화하고 반경은 후보 풀에 있는 대로 뽑았다. 그런데 격자 통과율이
  반경에 따라 급락한다 — 7편 × 384 격자 실측으로 `clear`/`clearance`/`floor_drop` 게이트 통과가
  **1.5 m 50.3% / 2.2 m 23.2% / 3.0 m 8.4% / 4.0 m 2.3%**. 반경이 균일한 격자인데도 usable 풀은
  1.5 m 로 60% 쏠리고, 첫 채택까지 가면 **76% (72/95)** 가 1.5 m 였다. 반경이 사실상 상수인
  코퍼스라 카메라 거리 변화를 학습 신호로 못 쓴다.
  - 고침: 반경을 **바깥 축**으로 두고 clip 마다 반경 순서를 섞은 뒤 반경당 `max_tries/len(radii)`
    개씩 할당한다. 먼 반경이 통과율과 무관하게 1지망 자리를 갖는다. 할당량으로 `max_tries` 를
    못 채우면 남은 (반경, 섹터) 조합으로 채운다 (먼 반경이 씨가 마른 씬).
  - 기존 97 probe 파일로 오프라인 재생: 1지망 반경 분포 **74/20/2/1 → 37/26/22/12**
    (1.5/2.2/3.0/4.0). 통과율 차이 때문에 최종 채택은 이보다 가까운 쪽으로 다시 쏠리지만,
    먼 반경이 시도조차 안 되던 상태는 아니게 된다.
  - `--radius_strata` (기본 on) / `--no_radius_strata` 쌍. off 브랜치는 rng 호출 순서까지
    예전과 같아서 기존 96편이 비트 단위로 재현된다.
  - smoke test 1편 (`00add26c` a00) 통과: az 60 elev 25 **r 1.5**, clear 1.00 clearance 0.454,
    render 156.4 s / 전체 ~200 s.
- **명사 추출 두 스크립트에 `--merge` — 1편만 돌려도 나머지 51편이 안 날아간다**
  (`camera_generation/models/Planner/CinemaTraj/scripts/extract_{nouns_vlm,static_nouns}.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  둘 다 `--videos` 로 준 영상만 계산한 뒤 결과 JSON 을 **통째로 덮어쓴다.** 코퍼스에 영상을
  하나 추가하려고 `--videos <새영상>` 만 돌리면 `vlm_nouns.json` / `static_nouns.json` 이
  그 1편짜리로 바뀌고, 다음 `sam3_static_instances.py` 가 나머지를 조용히 건너뛴다.
  → `--merge` 는 기존 파일을 읽어 같은 키만 갈아끼우고 나머지를 유지한다
  (`vlm_nouns` 는 `(video, frame_mode)` 기준, `static_nouns` 는 `video` 기준).
  기본값은 `--no_merge` = **예전 그대로 덮어쓰기**.
- **`render_bank_videos.py --no_sheet` + `--max_tiles 0` — 뱅크 전량을 변이별 mp4 로 스트리밍**
  (`camera_generation/models/Planner/CinemaTraj/scripts/render_bank_videos.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  기존 `--per_variant` 는 **타일 시트에 올라간 것만** 변이별 mp4 로 떨궜고, 그 타일 수는
  `--max_tiles 10` 이 상한이었다. "최종 남은 카메라 전량 depth render 저장"에는 둘 다 걸린다 —
  수백 타일짜리 그리드는 읽을 수도 없고, 시트를 만들려면 전 변이의 49프레임을 동시에 메모리에
  들고 있어야 한다 (480x270 기준 변이당 19 MB → 400 변이면 7.6 GB).
  → `--no_sheet` 는 시트 합성을 건너뛰고 렌더 즉시 mp4 로 흘려보낸다(상주 메모리 = 변이 1개분),
  `--max_tiles 0` 은 상한을 끈다. 기본값(`--sheet`, `--max_tiles 10`)은 예전 그대로.
  `--no_sheet --no_per_variant` 는 나오는 게 없으므로 `assert` 로 막았다.
- **`run_lbm_lite.py` 에 `pcd` stage 추가 (4D 점군 `cloud.npz`)**
  (`camera_generation/models/Planner/CinemaTraj/run_lbm_lite.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  `board`/`bank`/`fit`/`bankvid` 는 전부 `<out>/<video>/cloud.npz` 를 **읽기만** 하고 없으면
  `assert` 로 죽는데, 오케스트레이터에 그걸 **만드는 단계가 없었다.** 파일럿 2편
  (`camel`/`avocado-slice`)은 손으로 `python -m lbm.cloud` 를 돌려놨어서 안 드러난 구멍이다.
  51편 전량 확장에서는 첫 영상에서 바로 죽는다. → `pcd` = `python -m lbm.cloud`,
  `ALL_STAGES` 와 `BANK_STAGES` 의 `graph` 바로 뒤에 넣었다. 기존 동작은 그대로 —
  `--stage bank` 처럼 이름을 직접 주면 예전과 똑같이 그 단계만 돈다.
- **죽은 레버 7개 삭제 — `fit_hole_ladder.py` CLI 46 → 39개 (D57)**
  (`camera_generation/models/Planner/CinemaTraj/scripts/fit_hole_ladder.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. 전체 근거는 `CinemaTraj/DECISIONS.md` **D57**.
  - 지운 것: `--min_standoff_ratio` / `--min_standoff` / `--min_near_depth` / `--near_pct`
    (D50 이 standoff 판정을 껐고 D47 이 near_depth 를 기각한 뒤 한 번도 안 켰다),
    `--obb_clear_ratio` / `--obb_clear_cap` / `--min_obb_clear` (셋 다 기본값에서 no-op 였고
    비례항은 D51 이 실측으로 기각). `solve_knob` 예산 사슬에서 `clearance` 링크도 삭제.
  - **`standoff` / `near_depth` 두 열은 계속 측정되어 뱅크에 남는다** — 사라진 건 판정과
    손잡이뿐이다 (뱅크는 인벤토리, 거르는 건 소비자 몫 — D39/D45).
  - **비트 동일성 확인**: `out/d57/camel` 재생성 vs 기존 뱅크에서 `poses.npz` 의 `cam_c2w`
    168 변이 × 49프레임이 `max|diff| 0.000e+00`, `knob`·`status`·`binding`·`tau_max`·
    `path_len_u`·`hole_fraction`·`behind_frac`·`standoff`·`obb_slack`·elev/ground/approach
    열 전량 동일. 다른 건 진단 열 둘 — `near_depth` 78/168 행(max 18.20%),
    `subject_area_med` 45/168 행(−4.35%~+0.36%).
  - 그 차이는 **렌더러 비결정성**이다. 새 `scripts/probe_near_depth_repeat.py` 로 같은
    pose·같은 프로세스 4회 반복: 최악 행 `dyn_0__orbit_left_arc__hole0.2` 가
    0.3791/0.4414/0.3510/0.3510 (**산포 23.75% > 뱅크 간 차이 18.20%**), 두 뱅크 값
    0.3275·0.3871 이 모두 그 범위 근처. 같은 조건에서 `dyn_0__straight_ease__hole0.1` 은
    4회 0.2038 비트 동일 — 비결정성이 변이마다 다르다. `near_depth` 가 depth 의 1 백분위라
    splat z-buffer 동률 몇 개에 값이 통째로 끌려간다 (D47 이 판정에서 기각한 이유와 같다).
- **OBB 마진을 절대값에서 소스 대비 배수로 — `--obb_clear_src_ratio`(기본 0.3)**
  (`camera_generation/models/Planner/CinemaTraj/lbm/gates.py`,
  `scripts/{sample_camera_bank,fit_hole_ladder}.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. 전체 근거는 `CinemaTraj/DECISIONS.md` **D51**.
  - 동기(사용자 지시 "대신 충돌 판정을 조금 키워줘" → 절대 마진 4종 영상 후 **"camel 은 0.15 가
    맞는데 avocado 는 0.06 정도가 적당한 것 같은데 왜이럼?"**): `u` 는 씬 전역 스케일만
    정규화한다 (S camel 4.6713 / avocado 4.6471, **0.5% 차이**). 정작 판정에 들어가는 양들은
    안 맞춰진다 — 소스 카메라 자신의 최소 OBB 여유 **4.4배**(0.5107 vs 0.1165), binding 노드
    `max(extent)` 2.7배(0.137 vs 0.376), 최근접 `d_ref` 3.3배(0.639 vs 0.191).
  - 절대 마진의 병리는 **균일 팽창**이다. `m` 은 모든 박스를 축마다 `2m` 부풀리므로 `m=0.15` 가
    camel 낙타 얇은 축을 **6배**, avocado 의자 조각 얇은 축을 **61배** 키운다 — 물체가
    무엇이든 금지구역이 같은 크기로 수렴한다.
  - **크기 비례 가설은 세웠다가 실측으로 기각했다 (정직하게 기록).** 사용자가 고른 값을 지배
    노드 `max_ext` 로 나누면 1.09 / 1.07 로 2% 안에 겹쳐서 `m_j = ratio·max_ext_j` 를 구현했는데,
    돌려보니 camel 10.66 ✅ / avocado **1.19** ❌ 로 절대 0.06 의 2.25 보다 **나빴다**. 원인은
    avocado 의 실제 binding 노드가 `stat_4`(의자, 최근접)가 아니라 `stat_0`(테이블
    0.376×0.354×0.024)이고, 판때기라 `max_ext` 가 두께가 아니라 **너비**를 집어 cap 에 붙어
    ratio 가 무력해지기 때문. `min_ext` 로 바꾸면 `stat_0` 은 1% 로 맞는 대신 `stat_4` 가
    12배로 튄다 — 어느 척도든 binding 노드 셋 중 둘만 맞고 그 둘이 척도마다 바뀐다. 우연이었다.
  - 채택: `m = β × (소스 카메라 자신의 최소 OBB 거리)`. **β<1 이면 소스 카메라가 정의상
    통과**한다 (D47 이 standoff 에 쓴 것과 같은 꼴). 아무 정보도 안 줬는데 β=0.3 이 camel
    **0.1532** 를 내놓아 사용자가 눈으로 고른 0.15 를 path 합 10.35 / obb binding 14 까지
    재현한다. avocado 는 0.0350 으로 더 느슨해 움직임이 산다 (path 2.25 → **6.17**).
  - 스윕 전량 (`dyn_0` anchor · 6 preset, path 합 / hole-only 변이 / obb-coll-hole binding):
    절대 0.02 camel 12.84·0·0-12-12 / avo 6.18·7·0-11-13 · 절대 0.06 12.15·0·3-10-11 /
    2.25·14·8-0-16 · 절대 0.10 11.82·0·7-7-10 / 0.89·20·8-0-16 · 절대 0.15 10.35·0·14-0-10 /
    0.89·20·8-0-16 · 비례 1.0 10.66·0·14-0-10 / 1.19·18·8-0-16 ·
    **β=0.3 10.35·0·14-0-10 / 6.17·7·1-10-13** · β=0.5 7.34·0·14-0-10 / 2.39·14·8-0-16.
    사용자가 3분면 영상 `D51_obb_srcratio.mp4` 를 보고 **β=0.3** 을 골랐다.
  - **소스 카메라 적법성 검사가 또 버그를 잡았다** (D47·D49 에 이어 세 번째): `cap 0.20` 이면
    avocado **소스 카메라 자신의** slack 이 −0.036 (`stat_0`, 거리 0.164) 이라 원본 촬영이
    기각된다 → `--obb_clear_cap` 기본 **0.12** (소스 slack camel +0.391 / avocado +0.044).
  - 구현: `node_margins(nodes, ratio, floor, cap)` → `max(floor, min(ratio·size, cap))`.
    `cap` 은 **비례 항에만** 건다 (`clip` 으로 짜면 camel floor 0.153 이 cap 0.12 에 잘린다;
    `floor ≤ cap` 구간에선 `clip` 과 동일). `obb_clearance(..., margins=)` 가
    `(dists, ids, slacks)` 3-tuple 이고 **노드 선택이 거리 최소 → slack 최소**로 바뀐다.
    CSV `obb_slack` 열 추가, `bank.json.obb_gate` 에 노드별 마진 표를 통째로 남긴다.
  - **회귀**: `--obb_clear_ratio 0 --min_obb_clear 0.06` 이 기존 절대 마진 뱅크와
    9개 열 × 24 비교 **불일치 0**, `|obb_slack − (obb_clear − 0.06)|` max **0.000000**.
    `ratio=0` + floor 만 주면 D49 동작과 비트 동일하다.
  - **부수 발견(더 중요할 수 있음)**: 작동하는 모든 마진에서 G5 는 거의 전부 **anchor 노드
    자신**에 binding 한다 (camel `dyn_0` 14/14, avocado `stat_0` 6~8/8). 지금 G5 는 장애물
    회피가 아니라 "자기 subject 에 너무 가까이 가지 마라"는 **구도 제약**으로 작동하고
    `push_in` 계열과 정면으로 싸운다. 선택지 4종은 D51 하단에 기록.
- **`--min_standoff_ratio` 기본값 0.80 → 0.0 (standoff 판정 끔)**
  (`camera_generation/models/Planner/CinemaTraj/scripts/fit_hole_ladder.py`). 근거 **D50**.
  - standoff(D48)와 G5 OBB(D49)는 **같은 걸 두 번 잰다** — D49 가 이미 실측했듯 `clear0` 에서
    OBB 를 침범한 33변이의 standoff 가 **33개 전부** 임계 아래였다. 붙어 있으면 항상 standoff 가
    먼저 물어 G5 는 영원히 `binding` 에 안 잡힌다.
  - 남길 하나로 G5 를 골랐다. standoff 는 소스 대비 배수라 **"조금 키운다"가 안 되는 축**이다 —
    D48 에 이미 증거가 있다 (ratio 0.80↔0.90 에서 camel `push_in_arc` 가 0.939 → 0.020 불연속
    붕괴). G5 마진은 거리에 절대값으로 더해져 0.02 → 0.06 → 0.15 가 매끄럽게 조여진다.
  - `standoff` 열은 그대로 남고 `--min_standoff_ratio 0.8` 로 되켤 수 있다.
    `near_depth`(D48 에서 기각)와 같은 취급 — **측정은 하되 판정에서 뺀다**.
- **게이트 요약 줄 표기 수정** (`fit_hole_ladder.py`): `m_j = clip(ratio·max_ext, floor, cap)`
  으로 찍으면 floor > cap 인 기본 모드(camel floor 0.153 > cap 0.12)에서 모순처럼 읽힌다.
  실제 식 `max(floor, min(ratio·max_ext, cap))` 그대로 찍는다. 값은 원래부터 맞았다
  (출력의 `[0.153, 0.153] u` 범위 표기가 실제 마진).

### Added
- **원본 Look-Before-Move 를 TRUMANS 로 실행할 수 있게 하는 어댑터 3종 + 실행 환경**
  (`camera_generation/models/Planner/CinemaTraj/scripts/{probe_trumans_blend.py,
  trumans_blend_layout_worker.py,trumans_to_lbm_demo.py,lbm_preview_reel.py}`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  이때까지 원본 LBM 은 **한 번도 안 돌아갔다** — 입력이 `.blend` 씬 자산이라 Vista4D 영상
  코퍼스로는 줄 수가 없었고(리포가 "intentionally excluded" 한 JSON 5종), 머신에 Blender 도
  없었다. TRUMANS 는 그 입력 형태를 그대로 갖고 있어서 처음으로 실측이 가능해졌다.
  - 환경: **포터블 Blender 4.5.9 LTS** `/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/`
    (tarball 361 MB, root·pip 불필요) + ffmpeg 심링크 `/data1/cympyc1785/tools/bin/ffmpeg`
    → `imageio_ffmpeg` 동봉 바이너리 재사용. **아무것도 설치하지 않았다.**
  - `probe_trumans_blend.py` — headless 로 blend 내부를 덤프. `strings` 로는 문자열이 있다는
    것만 알지 그게 오브젝트 이름인지 머티리얼 이름인지 모른다. 실측(`00add26c-…`): 씬이
    **딱 하나이고 이름이 그냥 `Scene`**, 오브젝트 432 / mesh 417 / armature 1 / action 254,
    CYCLES, 960×540, fps 25. 상호작용 물체 이름은 `obj_list.txt` 와 정확히 일치한다.
  - `trumans_blend_layout_worker.py` — LBM `layout_description` 이 요구하는 world 위치·크기를
    잰다. **`obj.location`/`obj.dimensions` 를 쓰면 안 된다** — TRUMANS 는 mesh 를
    `<obj>_root_<obj>` EMPTY 에 매달아 애니메이션하므로 location 이 전부 0 근처로 읽히고,
    dimensions 는 local bbox × scale 이라 부모 회전이 빠진다. 8개 bbox 코너를 `matrix_world`
    로 보내 world AABB 를 다시 잡는다. 캐릭터는 `obj_list.txt` 에 없어 ARMATURE 타입으로 찾고
    스킨 mesh 자손을 union 한다. `rotation.z` 는 **도(degree)** 로 낸다
    (`cinematographer_stage.py:483` 이 `math.radians` 를 건다).
  - `trumans_to_lbm_demo.py` — recording → LBM `demo_root`. take 식별이 핵심: 한 씬 uuid 에
    실제 take 가 ~10개인데 `.blend` 에는 그중 하나만 구워져 있다. `scene_list`/`seg_name`/
    `scene_flag` 로 시퀀스별 프레임 수를 세어 **blend 프레임 수와 일치하는 것 하나**를 고른다
    (`00add26c-…` → 2077 프레임 → `2023-01-17@00-55-00`). 1.66 GB blend 는 **복사하지 않고
    심링크** — LBM 은 blend 를 수정하지 않는다(전 워커가 `-b` 배경 실행). `scene_size` 는 잰
    AABB 로 채운다(기본 ±10 을 그냥 두면 실내 5 m 씬에서 카메라 후보가 벽 밖으로 나간다).
    `Actions/<seq>.txt` 는 **탭 3열 `start\end\text`** 다(id 열 없음) — 4열로 읽으면
    `int('Pick')` 에서 죽는다. `shot_id` 는 줄 번호로 매긴다.
  - `lbm_preview_reel.py` — shot 별 final preview 를 PASS/BLOCK 라벨 + 차단 사유와 함께
    영상 1편으로. contact sheet 로 16개를 늘어놓으면 타일이 240 px 라 "프레임이 새하얗다"를
    눈으로 못 가린다.
  - **실측 (`--camera-quality fast`, 960×540, 로컬 Qwen3-VL-30B, run `trumans_00add26c`)**:
    Director 288.2 s / Cinematographer 86.4 s / VideoEngineer 78.3 s / Editor 38.8 s,
    합 491.7 s, rc=0. Actions 16줄 → shot 16 → 카메라 16대 → LBM 자기 VLM 게이트가 10대 차단
    → 6대 렌더(38프레임씩 228장) → `final_edit_v1.mp4` 6.24 s.
  - **주의: `fast` 는 look-before-move 가 아니다.** 후보 탐색·board 선택 전체가
    `cinematographer_stage.py:5346 if config.camera_quality == "quality"` 안에 있어서
    `candidate_count_raw_min=0, board_count=0` 으로 건너뛰어졌다. 카메라는
    `asset_view_seed` 하나로 정해졌고 `trajectory_safety_report.travel_distance` 가 6대 전부
    **0.0** (38프레임 동안 9 cm) — 사실상 정지 카메라다. 정직한 카메라당 시간은 `quality`
    모드로 재야 한다.
  - **실측 (`--camera-quality quality`, 같은 입력·해상도, run `trumans_00add26c_q`)**:
    Director 288.2 s(fast 산출물 재사용) / Cinematographer **232.7 s** / VideoEngineer 218.5 s /
    Editor 137.5 s, 합 **876.9 s**, rc=0. 이번엔 후보 탐색이 실제로 돌았다 — shot 당 raw 후보
    **~1,200** (16 shot 합 19,235) → eligible 5,078 → retained 286 → dedup 352, shot 당
    channel board 3장. 선택 출처는 `operation_expansion` 10 / `semantic_feet_s4_golden_ratio` 6.
    VLM 게이트 통과가 fast 의 6/16 에서 **13/16** 으로 올랐다(차단은 shot 1·3·5).
  - **카메라 1대당 시간 비교** (같은 머신, 같은 로컬 Qwen3-VL-30B):

    | | LBM-Lite | 원본 LBM `fast` | 원본 LBM `quality` |
    |---|---|---|---|
    | 카메라 생산 (Director+Cinematographer) | — | 23.4 s | 32.6 s |
    | 전 파이프라인 / 만든 카메라 | **7.85 s** | 30.7 s | 54.8 s |
    | 전 파이프라인 / **쓸 수 있는** 카메라 | 7.85 s | 82.0 s (6/16) | 67.5 s (13/16) |

    LBM-Lite 값은 13편 3,416 대 / 26,824.1 s 로 낸 것이고 `fit` 이 79.8% 를 먹는다.
    원본은 shot 16개 기준이라 표본이 작다 — 배율(약 **8.6배**)만 읽을 것.
    quality 가 fast 보다 대당 시간은 비싸지만 **버려지는 카메라가 줄어** 쓸 수 있는 카메라
    1대당으로는 오히려 싸다(82.0 → 67.5 s).
- **`trumans_to_lbm_demo.py --movement_terms` — shot 설명에 카메라 움직임 어휘를 심어 원본 LBM 의
  preset 을 실제로 갈라지게 한다** (`camera_generation/models/Planner/CinemaTraj/scripts/
  trumans_to_lbm_demo.py`). ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  **LBM 에서 카메라 움직임을 정하는 건 LLM 이 아니라 shot 설명 문자열에 대한 키워드 정규식이다**
  (`director_stage.infer_movement_intent:972-984`). Director 의 LLM 경로
  (`director_engine_llm.py`) 에는 `movement` 라는 단어조차 없다. TRUMANS 의 액션 라벨
  ("Pick up the book with both hands") 은 그 키워드를 하나도 안 건드려 전부 fallback `static` 이
  되고, 그러면 `cinematographer_stage.py:5141-5144` 가 "명시적 lock 문구가 없는 static" 을
  `subtle_motion` 으로 바꾸는데 그 함수는 close-up 이 아니면 **무조건 `push_in`** 을 준다
  (`:346-352`). 실측: 앞선 두 실행(`trumans_00add26c`, `_q`) 은 **16 shot 전부 `push_in_arc`,
  `motion_profile=semantic_light_dynamic`** — preset 어휘 17종 중 1종만 쓰였다.
  - `MOVEMENT_TERMS` 6종을 shot 설명 **뒤에 돌려가며** 붙인다. 각 구절은 `infer_movement_intent`
    (검사 순서 pan → pulls back → walks around → approaches → stares → static) ·
    `infer_direction_label` · `infer_distance_label` 을 동시에 통과하되 `infer_shot_goal` 의
    reveal/rush/chases 는 **일부러 피해** shot 당 카메라 1대를 유지하도록 골랐다.
  - **17 preset 중 6종만 이 경로로 닿는다.** 나머지는 구조적으로 불가능하다 —
    `orbit_*`/`pedestal_*` 은 `infer_movement_intent` 가 그 태그를 리턴하는 분기 자체가 없고
    (`canonical_movement:220-244` 는 알지만 producer 가 없다), `rise_reveal`/`drop_reveal`/
    `s_curve`/`straight_ease`/`static_hold_locked` 은 어떤 movement_tag 에서도 매핑되지 않는다
    (후보 탐색이 직접 고를 때만 등장).
  - 기본값 `--no_movement_terms` 라 **앞선 실행과 바이트 단위로 같은 demo_root** 가 나온다.
  - 실측 (run `trumans_00add26c_mv`, `--camera-quality quality`, 같은 입력·해상도):
    Director `primary_movement` = static 4 / push_in 3 / push_out 3 / pan 3 / truck 3
    (무어휘 대조군 `_q` 는 static 16). Cinematographer `motion_profile` = `authored` 14 +
    `closeup_static` 2 — **`semantic_light_dynamic` 붕괴가 한 건도 안 일어났다** (대조군은 26/26).
    최종 `trajectory_preset` **5종**: `truck_right` 3 / `static_hold` 2 / `push_in_arc` 1 /
    `static_subtle_zoom` 1 / `pan_right` 1 (대조군은 `push_in_arc` 13/13).
  - **정정: `pan_right`/`truck_right` 는 "도달 불가"가 아니라 오히려 pan/truck 에서 유일하게
    도달하는 쪽이다.** `cinematographer_stage.py:5155-5157` 이 `direction_tag` 를 left/right 로
    정규화한 `movement_direction` 은 **카메라 레코드에 저장되지 않는다**. 저장되는 건 원본
    `direction_tag`(여기서는 전부 `front`)뿐이고, 하류의 모든 `preset_for_motion(...,
    camera["direction_tag"], ...)` 호출(`:2287, :2735, :4445, :4480, :4624, :4766`)이 그
    `front` 를 받아 `:800-804` 의 else 가지로 떨어져 `_right` 를 낸다. `_left` 를 내려면
    `infer_direction_label` 이 `left` 를 리턴해야 하고, 그건 설명문에 `profile`/`side view` 가
    있어야 한다 — 즉 **`_left` 쪽이 어려운 경로다.** 기하도 라벨과 일치한다(3개 truck 전부
    `dot(Δloc, cam_right) > 0`).
  - **주의: preset 이 갈려도 이동량은 안 커진다.** `travel_distance` 평균 0.0401 / 최대 0.0898
    blender unit (대조군 0.0706 / 0.0786). 3.6 s 동안 9 cm 이므로 preset 라벨은 궤적의 *모양*만
    바꾸고 *크기*는 후보 탐색이 따로 정한다. `pan_right` 는 제자리 회전이라 travel 0.0000.
  - 통과 카메라는 **8/16** 으로 대조군 13/16 보다 줄었다 (게이트 warning 은 8대 모두 비어 있으니
    차단은 Cinematographer 뒤쪽 VLM 리뷰 단계에서 일어난 것).
- **TRUMANS 를 LBM-Lite 소스로 태우는 두 스크립트**
  (`camera_generation/models/Planner/CinemaTraj/scripts/trumans_{probe,clip}.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - `trumans_probe.py` — 배포본에 카메라 파라미터가 **없다**는 기존 메모를 뒤집는다.
    `smplx_result`(world) 와 `smplx_result_in_cam`(camera) 이 쌍으로 있으므로
    `R_c = R_cw R_w`, `t_cw = J0 + t_c − R_cw (J0 + t_w)` 로 카메라를 **복원할 수 있다**.
    무작위 12편 실측: `R_cw` 프레임간 편차 median **179.98°**, `|t_c|` median 2.10~2.13 m,
    `z_c` median −2.05~−2.09 m. 즉 **사람을 반경 ~2.1 m 로 따라다니는 가상 카메라**이지
    씬 고정 카메라가 아니다 — 그래서 이 pose 를 그대로 소스 카메라로 쓰면 안 된다.
  - `trumans_clip.py` — 30 fps × 1,500~3,400 프레임 연속 녹화에서 코퍼스와 같은 **49프레임
    창**을 잘라낸다. `--rank` 는 창을 (카메라 회전 폭, 사람 이동 거리, action label 종류 수)로
    점수 매겨 정지 구간을 피하게 하고, `--identify` 는 이미 잘라둔 클립이 어느 녹화 몇 번째
    프레임인지 되찾는다(provenance 복구).
- **`stack_videos.py` 에 격자 + 타일 이름표** (`--columns` / `--tile_width` / `--labels`,
  `camera_generation/models/Planner/CinemaTraj/scripts/stack_videos.py`). 셋 다 안 주면
  **예전 동작 그대로** 한 줄(`--direction`)이다. 7~51편을 한 화면에서 훑으려면 세로 한 줄로는
  못 보고, 이름표 없는 8칸 격자는 어느 타일이 어느 영상인지 셀 수 없어 벽지가 된다.
  `--tile_width` 를 줄 때만 리사이즈하고, 그때도 **모든 타일에 같은 폭**을 강제한다
  (타일 사이 배율이 갈리면 비교가 깨진다는 원래 설계 의도 유지).
- **전진 한계 게이트 (G7) — dolly 가 subject OBB 를 **지나쳐** 뒤에 서는 걸 막는다**
  (`camera_generation/models/Planner/CinemaTraj/lbm/gates.py:approach_profile`,
  `scripts/{sample_camera_bank,fit_hole_ladder}.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. **`--no_approach_gate` 면 기존 뱅크와 전 열 비트
  동일**하게 돈다. 전체 근거는 `CinemaTraj/DECISIONS.md` **D56**.
  - 동기(사용자): "앞 뒤로 움직이는 것도 bbox 를 지나치기 전까지 적당한 거리까지만 움직이게끔."
    G5 가 못 잡는 이유는 `obb_signed_distance` 가 **부호 없는 거리**라서다 — 박스를 통과해
    반대편으로 나가면 거리가 다시 커져 통과한다.
  - 시선축 `a` = normalize(anchor OBB 중심(frame 0) − 플랜 시작 카메라 위치), frame 0 고정.
    `s(f) = (p_g(f) − c_j(f))·a`, `gap(f) = −half_a − s(f)`, `past(f) = s(f) − half_a`.
    임계는 **소스 카메라 자신의 실측 `src_approach` 의 0.3배**(`--approach_src_ratio`, G5 의
    β 와 일부러 같은 값). 재투영뿐이라 **렌더가 0회 늘어난다**. anchor 노드에만 적용.
  - 게이트 off(`out/nog7`) vs on 매칭 비교: camel `dyn_0 push_in_arc` knob 1.000 → 0.573
    (`path_len_u` 0.795 → 0.442, off 에서 `past_frames` **2**, on 에서 `gap` +0.173,
    binding `obb` → `approach`), avocado `stat_0 push_in_arc` 0.681 → 0.149
    (0.465 → 0.152, binding `collision` → `approach`), `stat_0 straight_ease` 0.574 → 0.149
    (binding `elev` → `approach`).
  - 재생성한 뱅크 336 변이(씬당 168) 전량에서 **`past_frames` = 0, `approach_frames` = 0**.
    binding 분포 camel `hole 86 / approach 27 / ground 19 / elev 11 / obb 8 / none 17`,
    avocado `hole 77 / approach 49 / elev 18 / ground 13 / none 11`.
- **조사용 프로브 3종** (`camera_generation/models/Planner/CinemaTraj/scripts/`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - `probe_tau_divisor.py` — τ 분모 후보 **9종**(점 거리 median/mean, frame0/전 프레임/정적 한정,
    frame0 카메라 기준 / **카메라 위치 median 기준** 점 거리)을 hole 사다리 단별 τ 중앙값으로
    비교. **결론: 바꾸지 않는다.** '비 산포' 6.96 이 9후보 전부 동일하다 — 스칼라 분모는 한 씬의
    τ 전부에 같은 배수라 단 사이 어긋남을 못 고친다. 현재 `z_med_f0` 비 기하평균 1.136,
    최선 후보 `z_med_all` 1.114, 사용자 제안 `cloud_med_from_cammed` 1.128 (0.7% 개선).
    점 거리 **평균**(= S)이 최악(1.528/1.587), 점 거리 **중앙값**은 z-median 과 1~4% 이내.
    더 근본적으로 **분모를 바꿔도 카메라가 안 바뀐다** — τ 는 이분법의 탐색 변수라 분모를 바꾸면
    손잡이 눈금만 재매개화되고 같은 hole 목표에서 같은 궤적에 도달한다. 바뀌는 건 열의 숫자와
    `KNOB_RANGE`/`tau_floor` 경계뿐이다.
  - `probe_near_depth_repeat.py` — 뱅크와 같은 경로(해상도·프레임 집합·S)로 `near_depth`
    반복 측정. D57 비트 동일성 검사의 잔여 차이를 렌더러 비결정성으로 확정하는 데 씀.
  - `probe_wall_planes.py` — CinemaTraj 의 벽 상자를 우리 씬에 이식 가능한지 판정 (D59).
    `find_wall_planes` 가 찾은 평면(camel 1 / avocado 3)을 반공간 fence 로 켜면 뱅크 카메라
    **8,232 pose 중 위반 0.000** — G1(behind-surface)에 이미 포섭된다. 게다가 camel 평면 뒤에
    정적 점 **41%** 가 있어 경계면도 아니다 (지면 벗긴 뒤 남은 최대 평면일 뿐).
- **고도각 상한 + 지면 아래 금지 게이트 (G6) — `rise`/`drop` 이 물체 바로 위/아래로 가는 걸 막는다**
  (`camera_generation/models/Planner/CinemaTraj/lbm/gates.py:elevation_profile`,
  `scripts/sample_camera_bank.py`, `scripts/fit_hole_ladder.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. **`--no_elev_gate --no_ground_gate` 면 기존 뱅크와
  전 열 비트 동일**하게 돈다 (열은 남고 판정에서만 빠진다). 직전 뱅크는 `hole_bank_pre_g6/`.
  - 동기(사용자): "rise, drop은 너무 많이 움직여 ... 물체의 바로 위나 아래까지 가면 안됨."
    실측으로 전제가 맞다 — `rise_reveal` 고도각 p95 camel **89.83°** / avocado **85.27°**,
    `drop_reveal` p05 −83.99° / −84.58°, `pedestal_up` 도 avocado 77.63° 라 crane 만의 문제가
    아니다. 최악 `stat_0__rise_reveal__hole0.35` 는 **89.960°** 인데 기존 예산을 전부 통과했다.
  - 기전: 수직 이동은 hole 을 거의 안 늘려서(바닥에도 천장에도 점이 있다) shape 배증이 상한
    16배까지 다 돌아 `lateral_frac 0.35 × 16 = 5.6` → `atan(5.6) = 80°`.
  - 아래쪽은 **지면 관통**까지 간다 (camel 14 변이 / avocado 5, 최악 −1.282 u / 46-of-49 프레임).
    G1 이 못 잡는 건 구조다 — 바닥 밑 카메라는 소스 뷰에 **화면 밖**으로 투영돼 채점 대상이
    아니다 (해당 변이 `behind_frac` 전부 정확히 0.0000).
  - **가림 지표로 커버 안 된다**(사용자 질문 (b) 답): 바닥은 노드가 아니라 OBB 가림이 구조적으로
    못 보고(avocado `stat_0__drop_reveal__hole0.5` 바닥 아래 0.66 u 인데 `obb_occl_pass` 0.984),
    렌더 가림은 바닥 밑에서 생기는 **구멍**을 "보임"으로 센다.
  - 임계는 소스 대비(D47/D51 규칙): 고도 `max(--max_elev_deg 45, 소스 자신 + 10°)` — 소스가
    −6.32°~**+35.09°**(avocado `stat_4` chair)라 절대값만으론 그 앵커가 기각된다. 지면
    `--min_ground_clear_ratio 0.2 × 소스 카메라 자신의 높이` — 높이가 camel 0.1069 u /
    avocado 0.3641 u 로 3.4배 벌어져 절대값이 안 된다.
  - **ratio 0.2 는 스윕으로 골랐다**: 0.5 는 camel drop 을 τ 하한까지 눌러 **정지 클립**을
    만들고(path 0.014 u, D53 과 같은 실패), 0.0 은 바닥을 0.003 u 로 스친다. 0.2 는
    path 0.059 u / 여유 0.050 u. avocado 는 카메라가 높아 세 값이 같은 답을 낸다.
  - 새 열: `elev_max` `elev_min` `elev_abs_max` `ground_clear` `below_ground_frames`
    `src_elev_abs_max` `max_elev_deg` `src_ground_clear` `min_ground_clear`.
    새 `binding`/`status`: `elev`/`elev_limited`, `ground`/`ground_limited`.
    예산 사슬은 `collision → obb → ground → elev → clearance → hole`.
  - 뱅크 재생성 결과 (2026-08-22): camel 336 변이 / avocado 392 변이 모두 **지면아래 0**,
    G1·standoff·obb 잔여 위반도 0. `binding` 은 camel `elev 11 / ground 23`,
    avocado `elev 17 / ground 12`. 크기 축소 예: camel `dyn_0__rise_reveal__hole0.35`
    path **2.035 u → 0.483 u** (hole 0.319 → 0.046), `stat_0` 2.443 → 0.624 u.
    남은 문제는 camel `drop_reveal` τ 0.078 (path 0.055 u) 처럼 τ 하한에 눌린 정지 클립 — D53 건.
  - 상세·실측표·선택지 5개는 `DECISIONS.md` D55.
- **OBB 기반 가림 감사 `--obb_occlusion` / `--obb_occlusion_diag`**
  (`camera_generation/models/Planner/CinemaTraj/scripts/audit_bank_geometry.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. **기본값 off** 라 기존 호출은 그대로 돈다.
  - 동기(사용자): "구멍이 뚫리면 가렸는지 판단하기 힘드니 ... 물체 bbox 기준으로 판단하는 것도
    실효성 있는지 봐줘." 렌더 기반 `occlusion_pass` 는 **가리는 물체가 재구성이 안 됐으면**
    거기가 구멍이라 subject 가 비쳐서 "안 가려짐"으로 읽힌다는 가설.
  - `obb_occl_pass`(렌더 0회, 노드 OBB convex hull painter's algorithm) + `obb_occluder`
    (가장 많이 가린 노드) + `hole_at_occluder`(OBB 가 가렸다고 한 픽셀 중 렌더에서 실제로
    구멍인 비율, `--obb_occlusion_diag` 일 때만; 프레임당 렌더 1회) 열을 `geometry.csv` 에 추가.
  - **실측 결론: 가설된 기전은 1.4~3.0% 였다** — `hole_at_occluder` median 이 "렌더만 통과"
    그룹에서 camel 0.014 / avocado 0.030. 불일치는 양방향(camel 17 vs 8, avocado 42 vs 51)
    으로 나고 hole 과 무관하다(불일치 그룹 hole 평균 camel 0.322 vs 나머지 0.323,
    avocado 0.316 vs 0.363). **게이트로는 안 쓴다** — 상세·선택지 4개는 `DECISIONS.md` D54.
  - `scripts/render_bank_videos.py` 에 `--variant_ids` 추가 (감사에서 고른 변이를 그 순서대로).
    영상 `out/{camel,avocado-slice}/hole_bank/occl_disagree.mp4`.
- **뱅크 전량 → 태그 N개짜리 `canonical.json` — `scripts/emit_bank.py`**
  (`camera_generation/models/Planner/CinemaTraj/scripts/emit_bank.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. `decode/emit.py` 는 **건드리지 않았다**.
  - 동기: `decode/emit.py` 는 **결정 하나**만 canonical 로 바꾸는데, 실제로 만든 건
    `fit_hole_ladder.py` 가 푼 **뱅크**다 (camel 336 / avocado-slice 392). 사용자가 원한 게
    소스 카메라 재현이 아니라 **다양한 카메라 움직임의 augmentation** 이므로 하류로 넘길 단위도
    뱅크 전체다. 그 사이가 비어 있었다.
  - **새 포맷을 안 만들었다.** `canonical.json` 의 `cameras` 는 원래부터 태그→궤적 dict 이고
    `emit_model_cams.py` 는 `--cameras all` 로 전 태그를 돈다 (`:394,423` 이 `rel_c2w` 만 읽음).
    그래서 `build_canonical()` 을 변이마다 부르고 `cameras` 만 합친다 — canonical 규약
    (`rel[0]=I`, 단위 스케일, 21 index pick)의 구현은 계속 `decode/emit.py` 하나다.
    태그 = `variant_id` = `<anchor>__<preset>__hole<target>` 그대로.
  - `poses.npz` 에는 `cam_c2w` 만 있고 `build_canonical` 이 요구하는 `look_at`/`subject_track`/
    `tau`/`info` 가 없어서 `build_poses` 를 **다시 돌린다** (렌더 0회, 전량 10초). 다시 푸는
    이상 재현을 증명해야 하므로 재구성 궤적을 `poses.npz` 와 프레임 단위 대조하고 어긋나면
    멈춘다 (`--pose_tol` 기본 1e-9, `--strict` 기본 on).
  - **그 대조가 실제로 버그를 잡았다**: `min_sweep_deg` 기본값을 15.0 이 아니라 30.0 으로 잘못
    적어 `dyn_1` 계열 궤적이 최대 **1.575e-2 u** 어긋났다 (path_len 자체가 0.0116 이라 궤적이
    통째로 다른 수준인데, tau_max 0.015 → 0.0176 이라 표만 봐서는 정상으로 보인다).
    근본 수정: `fit_hole_ladder.py` 에 `SHAPE_DEFAULTS` 상수를 두고 CLI 기본값과 `emit_bank`
    fallback 이 **같은 출처**를 쓰게 했다. 예전 뱅크는 `fixed` 블록에
    `aim_ramp_frames`/`orbit_span_frac`/`min_sweep_deg`/`num_frames` 가 없어서 이 fallback 을
    타므로, 앞으로 생성되는 뱅크는 `fixed` 에 그 4개를 같이 싣는다.
  - 검증: camel 336 변이 전량 **pose 재현 최대오차 0.000e+00**, 21↔49 왕복 0.000e+00.
    `emit_model_cams.py --model sierpinskicam --cameras <tag>,<tag> --scales 1` 왕복 통과
    (파일명에 variant_id 보존, `.` → `p`).
  - **거르지 않는다 (D39/D45 규칙)**: 접힌 단도 정지 변이도 기본 전량 내보내고 요약과
    manifest 에 수만 찍는다 — camel 336 중 **접힌 단 94 / `translation_degenerate` 68**,
    `g = rmax/S` 범위 [0.00000, 2.28180] median 0.03875. 접힌 단은 `folded_onto` 로 대표
    태그를 가리킨다. 빼려면 `--drop_folded`/`--min_path_len`/`--status`/`--anchors`/`--presets`.
  - 산출물 `<bank>/canonical/{canonical.json,canonical.npz,manifest.json}`
    (`lbm_bank_canonical_v1`). `notes` 는 태그마다 복제하지 않고 최상위에 한 번만 싣는다.
  - **avocado-slice 392 도 전량 통과** — pose 재현 0.000e+00, 왕복 0.000e+00, 접힌 단 98.
    다만 `translation_degenerate` 가 **204** 이고 `g` median 이 **0.00000** 이다 (camel 은
    68 / 0.09372). 아래 "avocado 뱅크 절반이 정지 카메라" 항목 참조 — **뱅크 쪽 결함이고
    `emit_bank.py` 는 그걸 드러낸 것**이다.
- **세 번째 충돌 예산: 노드 OBB clearance (G5)** — `lbm/gates.py` 에 `obb_signed_distance` /
  `obb_clearance`, `scripts/sample_camera_bank.py` 에 `source_obb_clear` + `--measure_obb`,
  `scripts/fit_hole_ladder.py` 에 `--min_obb_clear`(기본 0.02 u) / `--obb_gate` /
  `--no_obb_gate` / `--measure_obb` / `--no_measure_obb`.
  ⚠ `camera_generation/models` 는 `.gitignore:222` 라 커밋에 안 들어간다.
  - 동기(사용자 지시 "물체 obb 기준으로도 충돌 판정해줘"): G1 도 standoff 도 **관측된 표면**만
    잰다. 표면은 껍질이라 카메라가 물체 **안**을 지나가도 반대쪽 껍질이 뒤에 남으면 G1 은
    "표면 앞", standoff 는 "가까운 점"으로만 읽는다. OBB 는 부피 판정이라 그 구멍을 막는다.
  - 부호 거리 `‖max(d,0)‖ + min(max(d),0)`, `d=|q|−extent/2`. 안팎이 한 식이라 이분법이 단조로
    민다. 좌표는 그래프 프레임 G — `T_gw[:3,:3]=R_gw/S` 라 **G 좌표가 곧 u 단위**다.
    동적 노드는 `node_obb_at(node, f)` 로 그 프레임 위치, 정적 노드는 고정 OBB (정적까지
    프레임별 track 을 쓰면 camel `fence` 의 추정 jitter `path_len_u` 0.32 가 판정에 들어온다).
    렌더 0회.
  - 임계는 배수가 아니라 **절대 마진 0.02 u**: 소스 카메라 자신의 최소 OBB clearance 가
    camel **+0.5107 u**(`stat_0` fence) / avocado-slice **+0.1165 u**(`stat_4` chair) 로 4배
    차이라 배수로 걸면 camel 만 과하게 조여진다.
    ⚠ **이 줄은 아래 Changed 의 "OBB 마진을 소스 대비 배수로" 항목에서 뒤집혔다.** 여기서 4배
    차이를 "배수를 못 쓰는 근거"로 읽었는데 실은 **배수를 써야 하는 근거**였다. 기본값은 이제
    `--obb_clear_src_ratio 0.3`, 판정량도 `obb_slack = obb_clear − m_j` 로 바뀌었다.
  - `obb_node` 열로 **어느 노드와 부딪혔나**를 남긴다. 최근접 노드 분포 camel `stat_0` 306 /
    `dyn_0` 16 / `dyn_1` 11 / `stat_1` 2 / `stat_2` 1; avocado `stat_4` 362 / `dyn_0` 22 /
    `stat_0` 8.
  - **이 두 씬에서는 한 번도 안 걸린다 (정직하게 기록).** 뱅크별 최소 부호거리 —
    `holeonly` camel −0.0258 u/21변이, avocado −0.0351 u/22; `clear0`(G1만) camel +0.0011 u/0
    (단 <0.02 가 13), avocado **−0.0281 u/20**; `r90` camel +0.1836/0, avocado +0.0755/0;
    현재 `hole_bank`(standoff 0.80) camel +0.1615/0, avocado +0.0583/0.
    `clear0` 에서 OBB 를 0.02 u 안쪽까지 침범한 33변이의 standoff 를 직접 재보니 **33개 전부**
    현재 임계(camel 0.0680·S / avocado 0.1095·S) 아래였다 — standoff 가 이미 다 잡는다.
    그래서 G5 를 켜도 손잡이가 한 행도 안 바뀐다. 넣는 이유는 ① 렌더 0회 ② standoff 가 잡는
    건 물체가 작아서 생긴 우연이고 크고 속 빈 물체(방·큰 가구)에서는 안쪽 한가운데가 오히려
    표면에서 멀다 ③ `obb_node` 진단. 즉 **회귀 감시기**다.
  - 회귀: `--no_collision_free` 16행이 `hole_bank_holeonly` 와 **0건 불일치**.
  - 영상: `out/avocado-slice/hole_bank/D49_obb_before_after.mp4` (위=G1만, 아래=standoff 0.80).
- **`--min_standoff_ratio` 기본값 0.90 → 0.80** (`scripts/fit_hole_ladder.py`).
  사용자가 4분면 영상(`D48_ratio_candidates.mp4`, r0.90/0.80/0.70/0.60 × `push_in_arc`)을 보고
  고른 값. 0.90 은 camel 전진 preset 이 전 anchor 에서 손잡이 하한에 붙었다.
  전량 효과는 camel 336행 중 **4행** — `dyn_0 push_in_arc` 4단이 0.020 → 0.939
  (path 0.013 → 0.743), path_len_u 총합 97.933 → 100.856. 스윕을 `dyn_0` 하나에서 돌렸는데
  그 anchor 만 경계에 걸려 있었다 (한 anchor 스윕으로 전량 외삽 금지).
  ⚠ **다시 바뀌었다: 기본값은 이제 0.0(꺼짐)** — 아래 Changed 참조.
- **뱅크 충돌·가림 감사 — `scripts/audit_bank_geometry.py`**
  (`camera_generation/models/Planner/CinemaTraj/scripts/audit_bank_geometry.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. 뱅크 생성 경로는 **건드리지 않았다** (감사만).
  - 동기: `lbm/gates.py` 는 `G1_behind`(카메라가 표면 뒤)와 `G3_occlusion`(subject 가림)을
    처음부터 갖고 있는데 **뱅크 경로가 그걸 안 부른다** (행에 `"gates": None`). board 경로는
    시작 pose 한 장만 검사하고, 뱅크는 `start_mode=source_frame0` 이라 시작 pose 는 소스
    카메라 자신이다 — 위험이 전부 나머지 48프레임에 있었는데 아무도 안 보고 있었다.
  - **hole 은 충돌을 못 잡는다**: 벽을 통과하면 벽 너머 관측이 그려져 `valid_mask` 가 멀쩡할
    수 있다. `corr(hole, behind_frac)` = **−0.014** (camel) / **+0.059** (avocado), camel 은
    위반 변이의 평균 hole 0.280 으로 정상 0.337 보다 오히려 **낮다**.
  - 실측: camel `hole_bank` G1 24/336 (7.1%) · G3 14 · 둘 중 하나 36 (10.7%);
    avocado `hole_bank` G1 23/392 (5.9%) · G3 56 · 둘 중 하나 78 (19.9%);
    τ 사다리 뱅크는 camel 6/432 (1.4%) · avocado 11/406 (2.7%).
    **G1 은 전진 preset 에 몰린다** — camel 24건 중 `straight_ease` 13 / `push_in_arc` 9.
    최악은 `camel dyn_0 straight_ease hole0.5` 로 49프레임의 **73.5% 가 표면 뒤**인데 hole 은
    0.356 (같은 anchor `truck_left` 0.507 보다 낮다). **G1 과 G3 은 거의 안 겹친다**
    (36건 중 2 / 78건 중 1) — 별개의 고장이다. 전체는 `CinemaTraj/DECISIONS.md` D46.
  - 산출물 `out/<video>/<bank_dir>/geometry.csv` (`behind_frames`/`behind_frac`/
    `behind_worst_src`/`occlusion_pass`). **행을 지우지 않는다** — D39·D45 와 같이 재고 기록만.
    G1 은 재투영뿐이라 렌더 0회(전 뱅크 수 초), G3 은 프레임당 렌더 2회라 `--occlusion` 으로 켠다.
- **preset 별 크기 상한을 잰다 — `scripts/fit_hole_ladder.py` (hole 사다리)**
  (`camera_generation/models/Planner/CinemaTraj/scripts/fit_hole_ladder.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. `sample_camera_bank.py` 의 τ 사다리는 **그대로**다.
  - 동기(사용자 확정 2026-08-21 "후자로 줘"): τ 한 값으로 전 preset 을 자르면 안 된다. 같은
    τ=0.35 에서 hole 이 preset 마다 **43배** 갈리기 때문에(`straight_ease 0.015` ~
    `pedestal_down 0.651`) 어떤 건 멀쩡한데 잘리고 어떤 건 이미 망가졌는데 살아남는다.
  - 축을 뒤집는다: **hole 을 고정하고 크기 손잡이를 푼다.** `HOLE_LADDER (0.10, 0.20, 0.35,
    0.50)` 각 단마다 (anchor, preset) 별 이분법 4회(5프레임) → 답을 13프레임으로 재측정.
    손잡이는 이동 preset 은 `tau`, 회전 전용은 `pan_deg`. 산출물
    `out/<video>/hole_bank/{bank.json,bank.csv,poses.npz,tau_caps.json}`
    (`lbm_hole_bank_v1` / `lbm_tau_caps_v1`).
  - 실측 (camel 336 변이 / 5,563 렌더, avocado-slice 392 / 6,500). hole 0.35 를 사는 τ 가
    camel 에서 `pedestal_down 0.128` ~ `push_in_arc 1.875` 로 **15배** 갈린다. 전체 표는
    `CinemaTraj/DECISIONS.md` D42.
  - `aim="look_at"` 은 **움직이기 전에** hole 을 쓴다: `clamped_low`(camel 90행 / avocado
    144행)가 **전부 look_at, `aim="traj"` 는 0행**. 그 행들은 `path_len_u` 0.010/0.000,
    `view_angle_max_deg` 0.3°/4.2° 인데 hole 이 0.47/0.37 이다 — 궤적이 아니라 anchor 재조준
    비용이다. anchor 별 최소 hole 은 `traj` 가 0.067~0.092 로 평평한 반면 `look_at` 은
    0.057~0.647 로 11배 갈린다 (camel `stat_3` 0.647 / avocado `stat_4` 0.638). D43.
  - `--shape_headroom 2.0` / `--shape_doublings 4`: `fit_tau` 의 `max_scale`(4.0) × 
    `DEFAULT_SHAPE` 가 구조적 천장이라 τ 를 못 맞추는 경우가 있다 (스모크에서 knob 3.0 인데
    `tau_max` 1.1493 에서 정지). `dolly_frac`/`lateral_frac` 을 배로 키워 재시도하고, 그러고도
    붙어 있으면 `unreached` 가 아니라 **`shape_limited`** 로 찍어 손잡이 탓과 모양 탓을
    구분한다. 수정 후 해당 행이 `solved`(knob 2.500, `shape_mult` 4.0)로 바뀌었다. D44.
  - hole 만으로는 못 자른다 — hole 0.50 단에서 camel `pan_right` 는 hole 0.481 로 통과하는데
    `subject_in_frame` 이 **0.15** 다. 두 열 다 기록만 하고 게이트하지 않는다. D45.
- **뱅크 프리뷰를 영상으로 — `scripts/render_bank_videos.py`**
  (`camera_generation/models/Planner/CinemaTraj/scripts/render_bank_videos.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. `sample_camera_bank.py` 의 `preview.png` 는 그대로다.
  - 동기(사용자 지시 2026-08-21 "앞으로 영상으로 보여줘"): 중간 프레임 **한 장**으로는 궤적을
    판정할 수 없다 — 구멍이 언제 열리는지, subject 가 몇 프레임째에 나가는지, 마지막 10프레임
    에서만 무너지는지가 안 보인다. τ 사다리의 요점이 "어디서 무너지나"라 프리뷰가 영상이어야 한다.
  - `bank/poses.npz` + `bank.json` 을 읽어 변이를 전 프레임 렌더 → **타일 애니메이션** 1개
    (`imageio` libx264, quality 6). 타일 캡션에 preset/anchor/τ/hole/path/프레임번호를 박는다.
  - `--anchors` / `--presets` / `--tau` 세 필터가 그대로 세 가지 읽기가 된다: 강도 축(τ 사다리)
    · 모양 축(preset 16종) · 표적 축(anchor 전량). 안 주면 τ 단으로 층화해 `--max_tiles` 만큼.
    `--with_source` 로 소스 타일 동봉, `--per_variant` 로 변이별 mp4 도 따로 쓴다.
  - 실측 (camel `dyn_0`, 전 49프레임): orbit τ0.1→1.0 hole 0.044 / 0.062 / 0.083 / 0.119 /
    0.163. 같은 τ0.35 에서 preset 별 hole 은 `straight_ease 0.015` ~ `pedestal_down 0.651` 로
    **43배** 벌어진다 — 강도보다 모양이 구멍을 더 좌우한다.
  - `--bank_dir`(기본 `bank`) 추가: `hole_bank` 을 주면 `fit_hole_ladder.py` 의 hole 사다리
    뱅크를 같은 코드로 깐다. 사다리 축만 `target_tau` → `target_hole` 로 갈리고(`rung_of`)
    나머지 읽기·필터·층화는 그대로다. **τ 뱅크 동작은 기본값 그대로 유지.**
- **카메라 augmentation sampler `scripts/sample_camera_bank.py` — VLM 선택자 대신 열거 + 실측**
  (`camera_generation/models/Planner/CinemaTraj/scripts/sample_camera_bank.py`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다. 기존 `lbm/loop.py`(VLM 선택) 경로는 **그대로**다.
  - 동기: 목표가 "이 씬에 가장 좋은 카메라 1개" → **"다양한 카메라 N개"**(augmentation)로
    바뀌었다. traj 턴은 D34/D36/D37 에서 세 번 측정한 결과 prior 다 (54 draw 중
    non-`orbit_left_arc` 5개). mode collapse 하는 선택자는 augmentation 에 못 쓴다.
  - 축 4개: anchor(게이트 통과 노드 **전량**) × preset(14 이동 + 2 정지) ×
    τ 사다리(`0.10 0.20 0.35 0.60 1.00`) × speed/tracking/look_at_bias(기본 각 1개).
    하드 컷은 "anchor 가 소스 frame 0 에서 보일 것" 하나뿐이고, `hole_fraction` 은
    **게이트가 아니라 각 단에서 실측해 기록**한다 (DECISIONS.md D39).
  - 출력 `out/<video>/bank/{bank.json,bank.csv,poses.npz,preview.png}` (`lbm_camera_bank_v1`).
    열거 camel 432 / avocado-slice 406. `--num_samples` 는 τ 단으로 층화 추출.
  - 실측 hole (정지 preset 제외, τ 0.10→1.00): camel 0.245 / 0.298 / 0.359 / 0.424 / 0.480,
    avocado(0.20→1.00) 0.301 / 0.343 / 0.389 / 0.450. τ 10배에 hole 은 2배가 안 된다.
- **sampler 의 두 실측 버그 수정** (같은 파일, 1차 실행에서 드러난 것):
  - `--pan_deg_at_max`(기본 60): `pan_left/right` 는 이동이 0 이라 τ 로 크기를 못 정한다 —
    `fit_tau` 가 항상 `max_scale`(4.0)로 튀어 **사다리 5단이 전부 같은 궤적**이었다
    (camel `path_len_u 0.0000` / `hole 0.740` × 5). 이 preset 만 사다리를 회전 각도로 옮겼고
    (`pan_deg = pan_deg_at_max × rung/max_rung / FIT_TAU_MAX_SCALE`), 결과는 pan 6→60° 에
    hole 0.067 → 0.679 로 단조. τ 요약표에서 빼고(`tau_invariant`) 별도 표로 찍는다.
  - `--drop_saturated`(기본 켬): `tau_start > rung` 이면 `fit_tau` 가 움직임을 0 으로 눌러
    `static_hold` 복제본이 나온다 — avocado τ0.1 에서 **98건 전량**이 그랬다. 빼고 몇 건인지
    `bank.json.dropped_saturated` 와 요약표에 남긴다. **정지 preset 은 예외**(identity 가
    의도한 결과 — 예외를 안 넣으니 `static_hold*` 14건이 통째로 사라졌다).
- **verify 지표 9번 `path_len_u` — "카메라가 실제로 움직였나"**
  (`camera_generation/models/Planner/CinemaTraj/verify.py`, `--min_path_len_u` 기본 0.05 u,
  WARN 은 그 절반, **`--min_path_len_u 0` 이면 행 자체가 안 생겨 예전 동작**).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - 동기: 계획서 지표 8종에 "움직여야 한다"가 없다. `static_hold_locked` 는 coverage 를
    최대화하는 퇴화 해라 VLM 이 자주 고르는데(avocado-slice 7 draw 중 6번), **8종이 전부
    PASS** 로 통과한다. VLM 자신의 설명이 그대로다 — "maintains 100% coverage throughout the
    shot with no camera motion, ensuring no hallucinated regions" (coverage 0.99).
    contract 가 coverage 를 상으로 주고 움직임에는 상을 안 주므로 정적이 최적해다.
  - 실측(avocado-slice 정적 draw): `hole_fraction 0.0154063` / `subject_in_frame 1` /
    `subject_pixel_coverage 0.334287` / `behind_surface_frames 0` /
    `max_view_angle_delta 7.4702` / `tau_max 0.128633` / `jerk_ratio 0` /
    `roundtrip_resample 0` / `anchor_identity 2.22045e-16` 전부 PASS,
    **`path_len_u 0` FAIL** → verdict PASS → **FAIL**.
    camel 은 `path_len_u 0.147629 PASS`, 나머지 지표 값 불변.
  - 근본 해결(contract 에 움직임 보상 넣기 / subject 를 동적 노드로 강제)은 아직 안 했다.

### Changed
- **LBM-Lite 후보 board 기본값을 27칸 → 9칸으로**
  (`camera_generation/models/Planner/CinemaTraj/scripts/build_candidate_board.py`:
  `--board_size 27 --board_columns 9 --tile_width 480 --tile_height 270` →
  `9 / 3 / 640 / 360`). ⚠ `.gitignore:222` 라 코드는 커밋에 안 들어간다.
  **예전 동작은 저 플래그 4개를 그대로 주면 재현된다** (기존 구조 유지 규칙).
  - 근거는 DECISIONS D35(각도 다양성) + D36(실측). 27칸은 4352×818 의 5.32:1 띠라
    select 턴 100 draw 중 **7 개가 `finish_reason=length` 로 잘렸고**(9칸 0/100),
    숫자를 빼면 camel 최상위 타일 픽이 5/5 → 0/5 로 무너진다. 9칸은 1928×1088 (1.77:1),
    prompt tok 평균 6040 → 4314, completion tok 평균 776 → 562.
  - 두 영상 전 파이프라인(board → loop → decode → emit → verify) 재실행, 둘 다 `verdict PASS`.
    camel `pool 45 → passed 33 → board 9`, preset `orbit_left_arc`, `tau_max 0.196051`,
    `max_view_angle_delta 12.9615`, `jerk_ratio 1.97805e-05`, `roundtrip_resample 0.00694961`,
    `anchor_identity 2.22045e-16`, emit `--scales 0.346156`.
    avocado-slice `pool 45 → passed 14 → board 9`, subject **`stat_0 (table)`** (정적 노드).
    ⚠ **avocado 의 preset 은 draw 마다 뒤집힌다** — 같은 설정 7 draw 중 **6번
    `static_hold_locked`**(`path_len_u 0.0000`, emit `--scales 0`), 1번 `orbit_left_arc`
    (`hole_fraction 0.289933`, `tau_max 0.197946`, `path_len_u 0.15771`, `--scales 0.36666`).
    camel 은 5/5 `orbit_left_arc`. 디스크 산출물은 소수파(orbit) draw 다.
  - ⚠ 현재 `lbm/loop.py` 는 `start_mode source_frame0` 이라
    `{'select': 'skipped_source_frame0', 'micro': 'skipped_source_frame0', 'traj': 'vlm'}` —
    board 는 VLM 입력이 아니라 **진단/프리뷰 산출물**이다. 이 변경은 렌더 비용과
    select 재활성화 대비이지 오늘의 결정 경로를 바꾸지 않는다.

### Added
- **LBM-Lite 오케스트레이터 + 설정 파일 + README** (`camera_generation/models/Planner/CinemaTraj/`:
  `run_lbm_lite.py`, `configs/default.json` (`lbm_lite_config_v1`), `README.md`).
  ⚠ `.gitignore:222` 라 커밋에 안 들어간다.
  - `--stage nouns|seg|graph|board|loop|decode|emit|verify|all` (별칭 `cloud`→`board`,
    `poses`→`decode`). `all` = graph,board,loop,decode,emit,verify — `nouns`/`seg` 는 공유
    `eval_data` 에 쓰고 SAM3 가 GPU 를 오래 잡아서 이름을 직접 줘야 돈다.
  - **존재 이유는 인자 어긋남 방지**다. `start_mode` 는 loop/decode/emit/verify 네 군데,
    `aim_anchor`/`aim_ramp_frames` 는 decode/emit/verify 세 군데에 각자 기본값으로 있다.
    어긋나면 emit 의 stale-poses 가드에 걸리거나(운이 좋을 때) 지문이 같아 조용히 옛 pose 를
    재사용한다(운이 나쁠 때). config 의 `shared` 블록을 **그 키를 받는 단계에만** 뿌린다
    (stage 별 `accepts` 목록).
  - 각 단계는 그대로 단독 실행 가능하다 — subprocess 로 부르기만 하고 `--dry_run` 이 명령을
    그대로 찍는다. bool 은 집 스타일 `--flag`/`--no_flag` 로 나간다.
    `--set board.board_size=27` 식 덮어쓰기, `--cuda` 는 **0~5 만 허용**(assert).
  - 단계 시작 전 `--help` 로 import 를 찔러보고(`--no_preflight` 로 끔) 실패하면 멈추고
    `conda run -n <env>` 를 찍는다. 실측상 **전 단계가 `vista4d` 하나**에서 돈다
    (계획서의 graph/decode = `da3` 표기는 틀렸다 — `build_scene_graph.py` 는 cv2·scipy·
    imageio·numpy 만 쓴다).
  - 실행 결과 `out/run_lbm_lite.json` (`lbm_lite_run_v1`: 해석된 config + 단계별 rc/초/명령).
  - 검증: camel `--stage all --cuda 1` graph 8.64 s / board 18.34 s / loop 28.91 s /
    decode 2.70 s / emit 2.61 s / verify 40.86 s, 전부 rc 0, `verdict PASS`, 지표 9종이
    수동 실행과 **완전히 동일**(`hole_fraction 0.0737214560303288`,
    `tau_max 0.19605113945287692`, `anchor_identity 2.220446049250313e-16`).
- **VLM 구멍 인지 ablation v2 — `select` 턴 + 그림 개입(가짜 magenta) + 위치 대조군**
  (`camera_generation/models/Planner/CinemaTraj/scripts/ablate_vlm_hole_perception.py`,
  format `vlm_hole_ablation_v1` → **`vlm_hole_ablation_v2`**). ⚠ `.gitignore:222`.
  - v1 이 "안 한 것"으로 남긴 것을 채웠다: 조건 4종 → **11종**
    (`A_full`/`B_no_num`/`C_no_img`/`D_conflict`/`E_paint`/`F_paint_nonum`/`G_shuffle`/
    `H_neutral`/`I_temp10`/`J_paint_low`/`K_paint_low_nonum`), 턴 `--turn {traj,select,both}`.
    `select` 턴은 trace 가 없어 `board/contract.txt` + `prompts/system_select.md` +
    `[board_candidates.png, source_frames.png]` 로 프롬프트를 재구성한다.
  - `E/F/J/K` 는 **board PNG 타일 중앙 60%×60% 를 `HOLE_COLOR=(255,0,255)` 로 덮어**
    가짜 구멍을 만든다 (traj board 는 preset 행의 중간·마지막 프레임만 — `system_traj.md` 가
    그렇게 보라고 지시하므로). `J/K` 는 같은 면적을 **최하위 타일**에 칠하는 위치 대조군.
  - traj 턴: 9조건 × 2영상 × 3 draw = 54 draw 중 `orbit_left_arc` 가 아닌 것은
    camel `C_no_img` 3/3(`s_curve`) 과 단발 2건뿐. avocado-slice 27/27 전부 `orbit_left_arc`.
    이름 중립화(`M01..M13`)·행 셔플·temperature 1.0·숫자 반전·가짜 magenta 다 못 흔들었다.
  - select 턴 9칸 camel (순서 고정, 숫자 off, 5 draw, 칠한 타일 `A1,C2`):
    `B_no_num`(깨끗) **5/5** · `K_paint_low_nonum`(대조군) **4/5** · `F_paint_nonum`(처치) **0/5**.
    reasoning 이 칠한 타일을 이름으로 짚는다("Tiles A1 and C2 are heavily magenta").
    숫자를 켜면 분리 소멸(`G_shuffle` 만으로도 1/5 로 내려간다).
  - avocado-slice 는 안 먹는다 — 칠하기 전부터 전 타일 magenta 면적비가 **0.22~0.37** 이라
    0.60 을 얹어도 순위가 안 바뀐다 (camel 은 0.03~0.15 위에 0.36~0.39). **그림 채널은
    hole 대비가 클 때만 작동**하고 구멍 많은 씬에선 숫자가 유일한 채널이다.
  - 산출물 `out/ablation/{hole_ablation_both.json, board27/, board9/, painted/}`.
- **VLM 이 렌더 구멍(magenta)을 보는지 가르는 ablation**
  (`camera_generation/models/Planner/CinemaTraj/scripts/ablate_vlm_hole_perception.py`,
  format `vlm_hole_ablation_v1`). ⚠ `.gitignore:222` 라 코드는 커밋에 안 들어간다.
  - 동기: `trace/turn_00.json` 이 "avoids the large magenta regions" 라고 답했는데 **같은
    프롬프트에 preset 별 `coverage_min`/`coverage_end` 숫자가 이미 있어서** 그림을 본 건지
    숫자를 읽은 건지 구분이 안 됐다. 4조건 × 3 draw × 2영상 = 24콜 (Qwen3-VL-30B-A3B, ~3 s/콜).
  - 조건: `A_full`(그림+숫자) · `B_no_num`(숫자 제거) · `C_no_img`(이미지 제거) ·
    `D_conflict`(**숫자만 1−x 로 반전**, 그림은 그대로).
  - 결과 — camel: A `orbit_left_arc`×3 / B `orbit_left_arc`×3 /
    C `orbit_left_arc, s_curve, s_curve` / D `orbit_left_arc`×3.
    avocado-slice: **네 조건 전부 `orbit_left_arc`×3**.
  - ① **구멍은 실제로 본다**: D_conflict 에서 숫자상 `orbit_left_arc`=0.10(최악),
    `truck_right`=0.40 인데도 "truck_left or truck_right result in significant magenta regions"
    라고 답했다 — 주어진 숫자와 반대이고 실제 그림과 일치(원본 0.68 / 0.60). B_no_num 에서
    숫자를 없애도 magenta 지목이 유지된다.
  - ② **정확하진 않다**: B_no_num 이 `rise_reveal` 을 "significant magenta" 로 지목했는데
    실제 coverage 0.96 으로 두 번째로 높다. 큰 구멍(0.6대)은 맞히고 중간은 지어낸다.
  - ③ **선택 자체는 prior 다**: `orbit_left_arc` 가 24 draw 중 22회. 이미지를 빼도 숫자를
    지워도 뒤집어도 안 바뀐다. D_conflict 답변은 `"maintains coverage (0.10) at the end"` 로
    **반전된 숫자를 인용하면서 그 숫자와 반대 결론**을 쓴다 — 모순을 못 알아챈다. 즉 preset
    품질을 지키는 건 VLM 이 아니라 **게이트**다 (camel 은 게이트가 3 preset 을 미리 잘랐다).
  - 미실시: 후보 `select` 턴 ablation, temperature 스윕, 다른 모델 대조, **그림 쪽을 조작하는
    조건**(구멍 없는 렌더에 가짜 magenta 를 칠하기 — 숫자 반전보다 강한 검사).
- **후보 board 기본값을 27칸 → 9칸으로 줄일 근거 측정** (`scripts/build_candidate_board.py`
  기존 `--board_size/--board_columns/--tile_width/--tile_height` 를 그대로 사용, 코드 변경 없음).
  27칸은 4352×818 px 로 **5.32:1 가로 띠**라 사람이 못 본다. 진짜 이유는 칸 수가 아니라 풀의
  각도 폭이다 — d_az 폭이 camel 34° / avocado 26° 뿐이라 인접 타일이 방위 8°·고도 10°·거리
  0.2× 차이로 육안 구분이 안 된다 (`pool_mode=budget` 이 τ 예산에서 역산하므로 의도된 좁음).
  top-N 별 각도 다양성(az/el/dist 종): 27 → 5/3/3, 15 → 4/2/3, 12 → 4/2/2, **9 → 4/2/2**(camel)
  · 3/2/2(avocado), 6 → 동일. **12칸과 9칸의 종 수가 같다.**
  9칸 + 타일 640×360 = 1928×1088 (1.77:1), 12칸 = 2572×1088 (2.36:1).
  ⚠ 기본값 변경은 **사용자 승인 대기 중** — 지금은 `/tmp/board_{9,12}/` 에 후보만 렌더했고
  `out/*/board/` 의 27칸 승인 대상은 안 건드렸다.
- **Look-Before-Move Lite 7단계 — 정적 명사 추출 + SAM3 정적 인스턴스 + `relations` 실배선**
  (`camera_generation/models/Planner/CinemaTraj/scripts/extract_static_nouns.py`,
  `scripts/sam3_static_instances.py`, `scene_graph/relations.py`, `scripts/build_scene_graph.py`,
  `scene_graph/viz.py`). format `static_nouns_v1`.
  ⚠ `camera_generation/models` 는 `.gitignore:222` 라 커밋에 안 들어간다 — 이 항목만 기록된다.
  - `extract_static_nouns.py`: VLM 이 이미 뽑아둔 명사(`out/vlm_nouns/vlm_nouns.json`)에서 정적
    명사를 고른다. `--source {auto,vlm,prompt}` — **기본 `auto` = VLM 우선, 없으면 프롬프트
    규칙 기반 fallback**. 계획서는 규칙 기반이 기본이었는데 실측이 뒤집었다: `--source prompt`
    는 camel `enclosure tree sun` / avocado `table bowl butter bottle chalkboard` 로,
    `sun` 은 물체가 아니고 `enclosure` 는 `fence` 를 놓친다.
  - 필터 두 겹 (`--drop_surfaces` 기본 on, `--max_nouns 8`): ① 광역 표면
    (`ground/wall/floor/sky/...`) 제거 — 이건 물체가 아니라 배경이라 SAM3 가 화면 절반을 문다
    ② 동적 명사와 겹치는 것 제거. 버린 명사는 사유와 함께 `static_nouns.json` 에 남긴다.
    실측: camel 5 → 4 (`fence roof tree bush`, `ground` 탈락),
    avocado-slice 8 → 6 (`table chair window television bottle "cutting board"`,
    `wall` 표면 · `plant` 동적중복 탈락).
  - `sam3_static_instances.py`: 정적 명사 → SAM3 text PCS → `seg_instances_static/<video>/`
    (`vista4d_seg_instances_v1`, 즉 `io.py` 가 이미 읽는 그 포맷). 저장 직전에 각 track 의
    **픽셀 총합** 중 배포본 `dynamic_mask` 안에 든 비율(`dynamic_frac`)을 재서
    `--max_dynamic_frac 0.5` 초과면 뺀다. 뺀 track 도 `static_report.json` 에 `kept:false` 로
    남긴다. ⚠ **이 두 씬에서는 한 번도 발동하지 않았다** — 12개 track 전부 0.0000 이고 실측
    겹침이 camel 140 px / 17,817,176 px, avocado 0 px / 21,656,235 px 다. 즉 이 필터는
    검증된 게 아니라 **미발동**이다. 실행: GPU 1, screen `infer1`, camel 4/4 · avocado 8/8 유지.
  - `--seg_static_root` 로 정적 노드를 켠 첫 실행에서 **잠복 버그 2개**가 드러나 같이 고쳤다
    (둘 다 `lbm/overlay.py:contract_text` 가 VLM 프롬프트로 그대로 뽑는 필드다):
    · `moving` 을 `path_len > 0.05u` 임계로 정하던 것을 **`kind=="dyn"` 조건과 AND** 로 바꿨다.
      임계가 존재하지 않기 때문이다 — 큰 정적 물체는 프레임마다 보이는 부분이 달라져 OBB 중심이
      떠돌고 그 떠돎이 실제 운동보다 크다. camel `path_len_u`: 울타리 0.3217 vs 낙타 0.0674
      (4.8배). avocado: 창문 0.5742 vs 움직이는 아보카도 0.1110 (5.2배), `center_drift_u` 는
      0.1137 vs 0.0093 (12.2배). 크기 보정도 안 통한다 — `bottle`(ext 0.027) 0.0505 >
      `dyn_2 avocado`(ext 0.038) 0.0251. 측정치는 안 지우고 `center_drift_u` 를 새로 실었다.
    · `against_wall` 임계를 `min(wall_u, wall_contact_frac·max(extent[:2]))` 로,
      **물체 크기 비례**로 바꿨다 (`--wall_contact_frac 0.15`; `1e9` 면 예전 절대 임계 동작).
      절대 `wall_u=0.08` 하나로는 camel 6개 노드가 전부 True 였다 — 실측 최근접 꼭짓점–벽면
      거리가 fence 0.0010 / roof 0.0175 / tree 0.0063 인데 **낙타 본체도** 0.0436, 0.0569 다.
      낙타는 울타리 앞에 서 있는 것이지 붙어 있는 게 아니고, 전부 True 인 플래그는 프롬프트에
      0 비트를 넣는다. 고친 뒤 avocado 는 11개 중 3개만 True 이고, 같은 `wall_u` 상한을 쓰는
      창문 두 개가 갈린다(`stat_1` True / `stat_2` False) — 임계가 실제로 뭔가를 재고 있다는 증거.
  - `viz.obb_overlay_video(kinds=...)` 추가. **기본은 동적만이라 기존 `obb_overlay.mp4` 는
    바이트 단위로 동일**(camel `f216e21cca3c`, avocado `89112f4efcae` — 승인본과 md5 일치).
    정적 검수용은 별도 `obb_overlay_all.mp4` 로 나가고 정적은 1px·bbox 없이 얇게 그린다.
    상자를 6~11개 겹쳐 그리면 정작 봐야 할 subject OBB 가 안 보이기 때문.
  - **하류 기하 불변 확인**: 정적 노드 투입 전/후로 `dyn_*` 의 `obb`·`track`·`d_ref`·`moving`
    이 전부 동일하고 `merge_log` 도 그대로다 (camel `[]`, avocado `dyn#0→dyn#1 0.9795`
    `dyn#2→dyn#1 0.9764`). 이미 승인된 `decision.json`/`poses.npz`/`canonical.json` 은 재실행
    없이 유효하다. 바뀐 건 프롬프트 텍스트뿐 — edges camel 0→4 / avocado 3→16, 그리고
    avocado 의 `supported_by` 가 dyn 셋 다 `null` → `stat_0`(table). 정적 노드 없이 돌리면
    VLM 이 **테이블 위 장면을 테이블 없이** 본다는 뜻이고 이게 이 단계의 본래 목적이다.
  - ⚠ **정적 노드의 OBB 품질은 동적보다 나쁘다 (고치지 않고 기록)**: `reproj_px` 가 camel
    `stat_0 fence` 151.0, avocado `stat_0 table` 111.8 · `stat_4 chair` 83.8 (동적은 1.7~18.3).
    `size_ok NO` 는 camel `tree` 2.07, avocado `table` 2.11 · `window` 1.85.
    `stat_4 chair` 는 extent `0.06,0.00,0.02` 로 w=0.00 인 퇴화 상자다. `in_bbox` 는 전부 Y —
    **중심은 맞고 크기만 과대**하다. 정적 노드는 후보 풀 원점도 게이트 대상도 아니고 소비처가
    `## NEIGHBORS` 의 `label`+`dist_u` 한 줄이라 지금은 받아들인다. 원인은 얇고 넓은 구조
    (테이블 상판 h=0.02)의 `minAreaRect` 깊이축 부풀림 + depth 누출.
  - 프리뷰: `results/2026-08-20_lbm_lite/{camel,avocado-slice}/obb_overlay_all.mp4`
    + `topdown_all.png` (기존 승인본 `obb_overlay.mp4`/`topdown.png` 는 덮어쓰지 않고 남겼다).
- **Look-Before-Move Lite 6단계 — `verify.py` (지표 8종) + 프리뷰 6종**
  (`camera_generation/models/Planner/CinemaTraj/verify.py`, format `lbm_verify_v1`).
  ⚠ 아래와 같은 이유로 커밋에 안 들어간다 (`camera_generation/models` 는 `.gitignore:222`).
  - 디코드된 49프레임을 **프레임별로** Vista4D 렌더해서(`poses[f]` 를 시간 `f` 의 점군으로)
    지표를 잰다. 전 프레임을 frame 0 점군으로 그리면 동적 subject 가 얼어붙어 subject 지표가
    통째로 거짓이 된다. env 는 계획서 표의 `da3` 가 아니라 **`vista4d`** (렌더러가 GPU 를 쓴다).
  - 지표 8종 + PASS/WARN/FAIL. 실측(GPU 1, `start_mode source_frame0`, preset `orbit_left_arc`)
    — camel / avocado-slice: `hole_fraction` 0.0737215 / 0.327682, `subject_in_frame` 1 / 1,
    `subject_pixel_coverage` 0.0917329 / 0.100786, `behind_surface_frames` 0 / 0,
    `max_view_angle_delta` 12.9615 / 11.2966, `tau_max` 0.196051 / 0.197719,
    `jerk_ratio` 1.97805e-05 / 6.33917e-05, `roundtrip_resample` 0.00694961 / 0.00687041,
    `anchor_identity` 2.22045e-16 / 1.9535e-17. verdict 둘 다 **PASS**.
  - **지표 4(behind_surface)는 여기서 처음 전 프레임 판정을 받는다.** 게이트는 frame 0 한 장만
    보고 preset 프리뷰도 7프레임만 확인한다 — frame 48 에서 벽을 뚫는 궤적을 구조적으로 못 본다.
  - **지표 8 은 `emit.roundtrip_error` 와 다른 것을 잰다.** 후자는 `rel_full_unit[idx]` 를 도로
    뽑아 비교하므로 정의상 항상 0 이다. 여기서는 `emit_model_cams.resample`(SE(3) 측지 보간)로
    21→49 를 되돌린다 — 실측 0.0069, 즉 21키로 깎는 데서 0.7% 를 잃는다.
  - **jerk 분모는 소스 p95, 소스가 정지면 `1e-2·S` 로 갈아타고 `jerk_denominator` 에 기록한다.**
    두 씬 다 소스 p95 가 0 이 아니어서(camel 0.0175792 u, avocado 0.00701176 u) 계획서 그대로
    p95 를 썼다. 분모를 조용히 바꾸면 배율이 씬마다 다른 뜻이 된다.
  - `poses.npz` 가 지금 `decision.json` 것인지 **emit 과 같은 `decision_fingerprint`** 로 확인한다
    (`--allow_stale_poses` 로 우회). 안 하면 옛 궤적을 새 결정인 척 채점하고, 숫자가 그럴듯해서
    안 들킨다.
  - 프리뷰 6종 → `results/20260820_lbm_lite/<scene>/`: `obb_overlay.mp4` ·
    `board_candidates.png`/`board_presets.png` · `plan_render.mp4`(hole 마젠타) ·
    `plan_sbs.mp4`(소스|렌더) · `plan_cam.mp4`(top/front/side, 소스 회색/플랜 주황/subject 점선) ·
    `decision_summary.txt`(지표표 + emit meta + preset board + 게이트 탈락 분포 + VLM 원문).
    영상은 전부 `imageio.mimwrite(codec="libx264", quality=6, macro_block_size=1)`.
  - **`make_camviz.py` 는 재사용하지 않았다.** rel 공간 궤적 **한 벌**을 정적 PNG 로 그리는
    도구라 ① 움직이는 소스 궤적 ② subject track ③ 영상 출력이 전부 없다 — 셋을 넣으면
    `draw_traj` 의 몸통이 남지 않는다 (DECISIONS.md D24).
  - `plan_cam.mp4` 는 **2줄**이다 (`--cam_detail`, 기본 on). 카메라-subject 거리 0.6 u 대비
    카메라 운동이 camel `ptp [0.0178, 0.146, 0.007]` u 라 한 눈금에 담으면 궤적이 뭉개진다.
    위 = 씬 전체, 아래 = 카메라만 확대. `--no_cam_detail` 로 1줄.
- **Look-Before-Move Lite 2단계 — scene graph 빌더**
  (`camera_generation/models/Planner/CinemaTraj/scene_graph/*`,
  `scripts/build_scene_graph.py`, format `planner_scene_graph_v1`).
  ⚠ 위와 같은 이유로 커밋에 안 들어간다 (`camera_generation/models` 는 `.gitignore:222`).
  - `io/scale/gravity/lift/instances/obb/relations/schema/viz` — RGBD + SAM3 instance track 을
    중력 정렬 graph frame `G` 로 올리고 노드마다 OBB(`center/extent/yaw/R`) · track ·
    `observed_faces` · `obs_az_span` · `viewing_distance.d_ref` 를 낸다. 계획서의 SDF /
    occupancy / `free_space` 는 넣지 않았다 (`--build_sdf` 자리만 남김).
  - **검증은 재투영 2종.** ① OBB center 를 최적 프레임에 되쏜 `reproj_px` ② 투영 OBB bbox 를
    마스크 bbox 로 나눈 `size_ratio`. ①만으로는 규약은 잡아도 크기 부풀림을 못 잡는다.
    실측 camel dyn_0 18.3 px / 1.12, dyn_1 4.1 px / 1.22, avocado-slice dyn_0 12.8 px / 1.28,
    dyn_1 16.1 px / 1.58, dyn_2 1.7 px / 1.14 — 전부 통과.
  - **`deinflate_depth_axis` 를 기본 off (`--deinflate` 로만 on) 로 뒤집었다.** "시차 0.005 인
    씬에서 시선 방향 두께는 depth 노이즈"라는 원래 가정이 camel dyn_1 에서 틀렸다 —
    `trim_depth_tail` + `fit_obb` yaw 규약 수정 후 `size_ratio` 가 1.22 로 떨어졌고, 그 상태에서
    보정을 걸면 extent 가 `0.16,0.05,0.13` → `0.05,0.05,0.13` 이 되어 몸통 길이를 통째로 날린다.
    부풀림 판정은 이제 추측이 아니라 `size_ratio` 실측이 한다.
  - `relations.build_relations` 의 `supported_by` 에 **바닥면 면적 조건**을 추가했다. 없으면
    avocado-slice 에서 `person supported_by avocado` 가 나온다 (사람이 상반신만 보여 `z_lo` 가
    조리대 높이라 아보카도 윗면과 0.05 u 안에 든다). 기하로는 참인데 말로는 거짓이고 이 엣지는
    VLM 프롬프트로 그대로 나간다.
- **Look-Before-Move Lite 3단계 — 후보 풀 + 렌더 게이트 + observation board**
  (`camera_generation/models/Planner/CinemaTraj/lbm/{candidates,gates,overlay}.py`,
  `scripts/build_candidate_board.py`). 위와 같은 이유로 커밋에 안 들어간다.
  - `gates.py` — G4 τ/view-angle(싼 프리필터) → G1 behind-surface(후보 위치를 소스 프레임 depth 에
    되쏘아 `z_cam > depth + 0.02·S` 면 벽 속) → G2 coverage → G3 framing(중심/면적/가림) 순서로
    싼 것부터 돌고 떨어지면 렌더를 건너뛴다. **전 후보 결과를 `gates.csv` 에 남긴다** — 탈락 사유
    분포가 곧 진단이다. G3 가림은 `render(subset=subject_points)` 로 subject 단독 렌더를 분모로
    쓴다 (점 개수로 나누면 거리에 따라 값이 통째로 움직여 임계를 못 정한다).
  - **후보 풀 기본값을 계획서의 절대 격자에서 τ 예산 역산(`--pool_mode budget`)으로 바꿨다.**
    절대 격자(az {0,±45,±90,±135,180} × el {12,28,45} × r {0.7,1.0,1.4}·d_ref)를 camel 에서
    돌리니 72개 중 **70개가 G4_tau 에서 죽었다**. 산수 문제다: `max_tau 0.30`, `z_med 3.553`,
    `S 4.6713` → 예산 `0.30·3.553/4.6713 = 0.228 u` 인데 `d_ref` 가 `0.64 u` 라 반경을 1.4배로
    늘리기만 해도(0.256 u) 초과다. 방위각 한계는 ±20° 근처고 "정면 45도"는 도달 불가능하다.
    계획서 격자는 `--pool_mode absolute` 로 남겨뒀다.
  - **subject 선택에 화면 점유 하한(`--subject_min_area_frac 0.01`)을 "가장 많이 움직인 노드"보다
    먼저 건다.** 없으면 avocado-slice 에서 화면 0.24% 짜리 아보카도 조각이 subject 로 뽑히고
    (path 0.111 u) 사람(12.7%)이 밀려서 G3 area 가 45개를 전부 떨어뜨린다 — τ 예산 안에서는 그
    조각을 3% 로 키울 만큼 다가갈 수 없다(실측 최대 0.43%). 움직임은 subject 를 **고르는** 기준이
    아니라 같은 급 후보들 사이의 **순위** 기준이다.
  - **contract 의 각도를 절대 subject-local 방위각에서 소스 카메라 기준 상대값(`d_az`/`d_elev`)으로
    바꿨다.** OBB yaw 는 180° 대칭이라 절대 방위각은 "정면"이 어딘지 정하지 못한다 — VLM 에게
    `az -146` 은 아무 정보가 아니다. `gates.csv` 와 타일 캡션에도 같이 실었다.
  - 실측: camel pool 45 → 통과 34 → board 27 (탈락 G4_tau 6, G2_coverage 5),
    avocado-slice pool 45 → 통과 24 → board 24 (G4_tau 8, G2_coverage 12, G3_occlusion 1).
- **Look-Before-Move Lite 4단계 — trajectory preset + decode → `canonical.json`**
  (`camera_generation/models/Planner/CinemaTraj/lbm/presets.py`,
  `decode/{build_poses,emit}.py`, `scripts/build_decision_fallback.py`, `DECISIONS.md`).
  위와 같은 이유로 커밋에 안 들어간다 (`camera_generation/models` 는 `.gitignore:222`).
  - `presets.py` — LBM 17 preset 을 `tools/recammaster/traj.py` 조합으로 재현하고 `n=49` 로 뽑는다
    (`traj.py` 기본 `N_POSES=21` 이 아니다). speed 4종(`steady/accel/decel/ease`)은 dense 궤적을
    **index pick** 으로 리샘플한다(보간 없음). `s_curve` 는 `arc(+σ/2)` + `start_at` + `arc(−σ/2)`.
  - **preset 마다 조준 모드 `aim` 을 나눴다** (`look_at` / `traj`). 전부 매 프레임 subject 를 다시
    조준하게 하면 `pan_left` 가 pan 이 아니게 된다 — 회전을 넣어도 look-at 이 도로 끌어와 항등이
    된다. `pan/truck/pedestal/static_hold_locked` 는 `aim="traj"`, 나머지는 `look_at`.
    `aim="traj"` 에서는 `tracking` 이 무시되므로 `tracking_ignored` 를 canonical meta 에 싣는다.
  - **궤적 크기는 사람이 정하지 않는다 — `fit_tau` 가 `target_tau` 를 만족하는 최대 배율을 8회
    이분법으로 찾는다.** `DEFAULT_SHAPE` 는 모양만 정한다(dolly_frac 0.35, lateral_frac 0.35,
    sweep 45°, pan 20°). 같은 `orbit_left_arc` 인데 실측 배율이 camel 0.42188 /
    avocado-slice 1.40625 로 **3.3배** 갈린다.
  - **τ 는 시작 pose 와 궤적이 나눠 쓰는 하나의 예산이다.** `tau(f)=|p_plan(f)−p_src(f)|/z_med` 가
    시작 offset 과 움직임을 같이 세는데 계획서는 게이트 `max_tau 0.30`(후보용)과
    `target_tau 0.20`(궤적용)을 따로 뒀다. 그래서 board 1위를 그냥 집으면 **카메라가 선다** —
    실측 camel A1 `tau_start 0.2595`, avocado-slice A1 `0.2815` 로 둘 다 0.20 초과라 `fit_tau` 가
    scale **0** 을 골라 `path_len_u 0.0000` 이 나왔다. 게이트를 건드리는 대신 **결정 층**에
    `--start_tau_frac 0.5` 를 넣어 시작 pose 에 절반만 준다(`build_decision_fallback.py`).
    고친 뒤 camel `traj_scale 0.42188 tau_max 0.1931 path_len_u 0.1453 view_angle_max 12.77°`,
    avocado-slice `1.40625 / 0.1983 / 0.1404 / 9.86°`.
  - ⚠ 그 결과 **fallback 이 고르는 시작 pose 는 소스 pose 자신**이다. board 후보의 τ 가
    `[0.0, 0.12, 0.12, 0.15, ...]` 로 이산적이라(격자 한 칸이 이미 예산의 60%) headroom 0.10 안에
    드는 게 중심 하나뿐이다. 보수적 기본값으로는 맞지만, 시작 pose 에 예산을 얼마나 쓸지는 원래
    VLM 의 결정이므로 **5단계 프롬프트에 "τ 는 시작과 움직임이 나눠 쓴다"를 명시해야 한다.**
    격자를 조밀하게(`--num_azimuth 9 --num_radius 5`) 하는 선택지도 남겨뒀다.
  - `build_poses.py` — decision + graph → world c2w (49,4,4). tracking gain
    `world 0.0 / drift 0.6 / lock 1.0`, look-at bias 는 방향 `gravity.up_world` · 크기 OBB 높이×S.
    `det(R)=1` 전 프레임 assert, `aim="look_at"` 이면 중력축 대비 roll assert — 실측 둘 다 **0.00e+00**.
  - `emit.py` — frame0 anchor(`rel[0]=I`, 실측 편차 1.1e-16 / 2.1e-17) → 49→21 **index pick**
    `rint(linspace(0,48,21))` → unit scale(`rmax` 로 나눔). 원래 크기는 무차원 손잡이
    `g = rmax/S` 로 meta 에 남는다 — camel `rmax 0.677318 g 0.14500`,
    avocado-slice `rmax 0.650821 g 0.14005`. zoom 은 `emit_model_cams.py` 에 intrinsics 채널이
    없어 통과하지 못하므로 `zoom_dropped` 플래그로만 남긴다. 모델별 함정 6종을
    `canonical.json` 의 `notes` 에 같이 싣는다.
  - **`emit_model_cams.py` 왕복 검증 통과** (camel, `--scales 0.340772`):
    recammaster/sierpinskicam/infcam/trajectorycrafter/cameraanything 5종 전부 emit,
    UE5 JSON 을 다시 OpenCV c2w 로 되돌린 오차 trans 2.7e-7 / rot 5.0e-7 (JSON 이 `%.6f`),
    `rmax` 0.677318 로 양쪽 일치. emit 쪽 업샘플러 `resample(rel21,49)` 와 우리 49프레임 원본의
    차이는 trans 0.00696 · 회전 0.126° (camel) / 0.00696 · 0.135° (avocado-slice) 로
    검증 기준 0.02 이내. `rerope` 는 `NATIVE_1X` 에 없어 별도 경로다.
  - `DECISIONS.md` 신설 — 3·4단계에서 계획서와 어긋나게 고른 15건의 선택지·이유·되돌리는 법.
- **Look-Before-Move Lite 1단계 — 4D point cloud 캐시 + 후보 렌더러**
  (`camera_generation/models/Planner/CinemaTraj/lbm/{cloud,render}.py`).
  ⚠ `camera_generation/models` 는 `.gitignore:222` 로 추적 대상이 아니라 이 파일들은 커밋에
  들어가지 않는다 (`video_generation/tools` 와 같은 상황). 이 항목은 로컬 기록용.
  - `cloud.py` — Vista4D `unproject()` 를 감싸 영상 1편을 `cloud.npz`(format `lbm_cloud_v1`)로
    캐시한다. `visible (n,f)` 를 `np.packbits(axis=1)` 로 8배 줄여 저장. 게이지(`S`, `z_med`,
    `parallax_ratio`)를 meta 에 같이 싣는다 — camel `S=4.6713 plx=0.0046`,
    avocado-slice `S=4.6471 plx=0.1286` 으로 기존 실측치를 그대로 재현.
  - **`preprocess_scene` 을 unproject 앞에 필수로 끼웠다** (`--no_preprocess` 로 raw 경로 재현).
    저자 파이프라인(`render_single.py:72`)이 늘 거치는 단계인데 처음에 빼먹었더니 camel 재렌더가
    20 dB 에 묶였다. 기본 `static_mask = ~dynamic_mask` 가 마스크 경계에서 프레임당 0.04% 씩
    물체 표면을 흘리고, static 점은 전 프레임 visible 이라(`point_cloud.py:63`) 그 누수가 49프레임
    한꺼번에 렌더되며, 유령이 진짜 물체와 z 가 거의 같아 z_tolerance(log1p 0.02) 안에서 가려지는
    대신 **블렌딩**되어 반투명 빗살이 됐다. `S`/`z_med` 는 sky depth 가 `SKY_DEPTH=1e3` 으로 덮이기
    전 raw depth 에서 잰다.
  - `render.py` — `CloudRenderer` (cloud 를 GPU 에 한 번 올리고 pose 를 갈아끼움), `look_at_c2w`
    (roll 은 중력축 기준 0), `measure()` (coverage / hole_fraction / subject bbox·면적·가시율),
    `visible_at(temporal_persistence=)` (NTP = 그 프레임 유래 점만).
  - **`--self_check` 의 규약 판정을 PSNR 에서 재투영 잔차로 교체.** 이 렌더러는 소스 pose 에서도
    2x2 box blur 를 먹는다 (`point_cloud.py:129-138`: 픽셀 중앙 점이 `du=dv=0.5` → 네 이웃 0.25씩)
    → PSNR 상한이 내용 의존적이라(camel 28 dB / avocado-slice 40 dB) 임계로 규약을 못 가린다.
    대신 `reprojection_residual()` 이 프레임 f 유래 점을 그 카메라로 되쏘아 원래 픽셀과 비교한다:
    실측 **0.0003 px**, y축 반전 음성 대조군 **719 px**. 음성 대조군 assert 를 같이 둬서 검사가
    실제로 규약을 보고 있는지 확인한다. PSNR 은 데이터 품질 보조지표로 강등(NTP/TP 병기).
    camel · avocado-slice 양쪽 PASS.
- **`scripts/sam3_seg_instances.py`** — 배포본 `recon_and_seg` 를 읽기만 하고 SAM3 만 재실행해
  per-instance track 을 별도 루트(`eval_data/seg_instances/`)에 쓴다. Pi3·DA3 재구성을 건너뛰므로
  영상당 ~30 s 이고, 배포본을 안 덮어써서 끝난 eval 110 entry 의 재현성이 유지된다.
  `--num_shards/--shard_id/--skip_done` + `--check_dynamic_mask` (우리 재실행을 OR 로 뭉갠 것 vs
  배포본 `dynamic_mask` 의 프레임 평균 IoU — 낮으면 keyword/모델 버전이 어긋난 것이라 병합 임계를
  논하기 전에 봐야 한다). 실측: camel 3 track / IoU 0.994, avocado-slice 8 track / IoU 0.988.
- **SAM3 instance track 저장 + 중복 병합 후처리** — scene graph(PSG4D relation) 입력용.
  Vista4D 의 `run_sam3_video` 는 per-instance mask / 프레임 관통 track id / keyword / box / score 를
  이미 만드는데 (`utils/recon_and_seg/seg_sam3_official.py:18-88`) `recon_and_seg_single.py` 가 그걸
  vis 오버레이에만 쓰고 버린다 — 디스크에 남는 `dynamic_mask` 는 전 인스턴스를 OR 로 뭉갠 이진
  마스크라 인스턴스가 사라진다. 그래서 SAM2 나 PSG4D tracking 모듈을 새로 붙일 필요가 없고
  저장만 추가하면 된다.
  - `models/Vista4D/utils/recon_and_seg/seg_sam3_utils.py` 에 `save_seg_instances` /
    `load_seg_instances` 추가 (format `vista4d_seg_instances_v1`). `meta.json` (프레임별
    id/keyword/score/box) + `masks.npz` (`np.packbits` 후 `savez_compressed`, 프레임당 키 하나).
    인스턴스가 겹쳐도 손실이 없다 — DynamicVerse 의 단일 채널 instance-id PNG
    (`stage2_sa2va.py:525` last-writer-wins) 와 달리 마스크를 인스턴스별로 따로 둔다.
  - `models/Vista4D/scripts/preprocess/recon_and_seg_single.py --save_seg_instances`
    (기본 off = 기존 동작 그대로). `seg_keywords` 가 비었거나 `_all_` 이면 SAM3 를 안 타므로
    경고만 찍고 건너뛴다.
    ⚠ Vista4D 는 자체 `.git` 을 가진 vendored repo 이고 `video_generation/models` 는
    `.gitignore:224` 로 부모 추적에서 빠져 있다 — 위 두 파일은 이 커밋에 들어가지 않는다
    (기존 `--keep_recon_sky` 패치와 같은 `# LOCAL:` 주석 규약으로 working tree 에만 존재).
  - `scripts/merge_seg_instances.py` (신규) — 중복 keyword 를 **track 단위** mask IoU 로 병합.
    Vista4D 는 keyword 하나당 SAM3 를 따로 돌리고 `obj_id_offset` 을 더하므로 `metadata.csv` 의
    `woman,person,human` 같은 recall 우선 나열이 같은 사람을 track 3 개로 만든다 (scene graph 에선
    노드 3 개 = 그래프 오염). Uni4D 처럼 프레임별 box NMS(0.5) 를 쓸 수는 없다 — Uni4D 는
    GroundingDINO 한 forward 에서 전 프레이즈를 채점하지만 SAM3 의 PCS 는 프레이즈별 별도 패스라
    억제할 자리가 없고, 프레임별로 걸면 살아남는 track 이 프레임마다 달라져 track 이 조각난다.
    병합 조건은 공통 등장 프레임 평균 mask IoU >= `--iou` **그리고** 짧은 쪽 대비 공통 프레임 비율
    >= `--min_co_frac`, union-find 로 묶는다. `--label_policy {score,area,keyword_order}`,
    `--dry_run` (표만 출력), `--vis` (imageio+libx264 오버레이 영상).
    출력은 항상 새 폴더(`<seg_instances>_merged`) 라 임계를 바꿔도 재-recon 이 필요 없다.
    같은 v1 포맷 + `meta["merge"]` 에 node/alias/pairwise 표를 덧붙인다.
    검증: 합성 데이터 왕복 무손실(W=101 로 8 의 배수 아닌 폭 + 빈 프레임 포함),
    woman/person 중복쌍 mean_iou 0.943 -> 3 track 이 2 node 로, knife 는 분리 유지.
- **Vista4D 어댑터** — 벤치마크 7번째 모델. 3단계라 "카메라 파일만 갈아끼우기"가 안 되고
  소스마다 4D 재구성을 먼저 돌려야 한다.
  - `tools/recammaster/vista4d_prepare.py` (신규) — canonical 단위 궤적 -> stage 2 의
    `--cam_path` npz. `T_w[i] = C_src[a] @ T_local[i]`, `a` 는 stage 2 의 중앙 슬라이스
    시작 프레임(`render_single.py:52-54`)을 재구성 프레임 수로 계산한 것. 공통 손잡이는
    `|t|max = g * S`, `S` 는 `sierp_scene_scale.py:58` 과 **같은 정의**의 mean ray length 를
    Vista4D 자신의 재구성에서 잰 값 (전 픽셀; `--exclude_sky` 는 진단용). depth 는 EXR
    (float16 단일 채널 `Y`, `utils/media.py:save_depths`) 로 읽는다.
  - `results/20260818_vista4d/run_vista4d_recon.sh` (신규) — stage 1 드라이버. taylor 소스는
    81 프레임이라 stage 1 이 **중앙**을 자르는데 sierp/trajc 는 **앞** 49 를 쓰므로, 여기서
    앞 49 로 미리 트림해 창을 맞춘다. `PYTHONPATH=$V4` 필수 (`scripts/preprocess/` 안에서
    `from utils.media import ...` 가 터진다 — cwd 로는 해결 안 된다).
    소스 목록: `hs_sC` / `hs_dynA` / `taylor_c{01,02,05,19,23}` / `rcm3` / `rcm5`.
    ReCamMaster 예시 2개는 원본이 1195 프레임인데 다른 모델도 전부 **앞** N 프레임만 읽으므로
    (`run_grid.py:339,350,488`) 여기서도 앞 49 로 트림한다.
  - `vista4d_prepare.py --pivot_match {none,z_med,S}` (기본 `none` = 기존과 bit-identical;
    g0p2 8종 재생성 maxdiff 0.000e+00 확인) — 궤도 카메라마다 **g 를 따로** 잡아 궤도 반경
    (`pivot_unit * g * S`) 이 씬 깊이에 오게 한다. `pivot_radius()` 는 광축들의 최소제곱
    수렴점까지의 frame0 기준 거리이고, 순수 이동 궤적은 광선이 평행해 특이행렬이므로
    `inf` 를 돌려주고 그런 카메라는 기본 g 를 그대로 쓴다. 배경은 memory
    `dl3dv-orbit-cams-pivot-collapse`: canonical `rmax=1` + 이동만 스케일하는 g 가 dl3dv
    궤도 반경을 씬 깊이의 0.08~0.48 배로 앉혀 warp 이 새까매진다 (좌표계 버그가 아니다 —
    hole 이 생기는 쪽이 세 축 다 trajc 와 일치). manifest 에 `pivot_unit` / `pivot` /
    per-entry `scale` 을 남긴다 (`inf` 는 표준 JSON 이 아니라 `null` 로).
  - `results/20260818_vista4d/run_vista4d_grid.sh` (신규) — stage 2+3 을 소스 x 카메라로.
    `GTAG` / `SCALES` 환경변수로 "디렉토리 이름 = g" 규약을 끊을 수 있다 — `--pivot_match`
    로 낸 카메라는 카메라마다 g 가 달라 그 규약이 성립하지 않는다. 안 주면 기존과 동일.
    `rcm3` 만 카메라 **8종** (기본 6 + `dl3dv_L49_b_p`, `dl3dv_L81_b_p`; 사용자 지시).
    `_b_p` 둘은 canonical `norm='path'` 라 `rmax` 가 0.7464 / 0.5804 이고
    `vista4d_prepare.py:178` 이 rmax 재정규화를 안 하므로 같은 g 에서 `|t|max` 가
    2.1203 / 1.6487 m (기본 6종은 2.8406 m) — 대신 회전이 104.2 / 178.7 도로 훨씬 크다.
  - `run_grid.py --model vista4d` + `--v4_recon <stage1 root>` — stage 2(`render_single`)
    -> `_cond/video_pc.mp4` 를 `_warp/warp.mp4` 로 복사 -> stage 3(`scripts.inference.inference`,
    Wan2.1-T2V-14B + `384p49_step=30000`). `--warp_only` 는 stage 2 까지만.
- `results/20260819_vista4d_eval/run_eval_gen.sh` (신규) — Vista4D **공식 eval 데이터**
  (`/data1/cympyc1785/data/Vista4D-Eval-Data`, 51 소스 x 2 카메라 = 110 entry) 를 저자 배포
  스크립트 그대로 돌린다: `scripts.preprocess.render_eval` -> `scripts.inference.inference_eval`.
  위의 `run_vista4d_grid.sh` 경로와 다른 점은 stage 1 재구성(depth/mask/`cameras.npz`)과 카메라
  npz 를 **우리가 만들지 않고 저자 것을 그대로 쓴다**는 것 — 그래서 `vista4d_prepare.py` 도
  `run_grid.py` 도 안 탄다. 출력 디렉토리를 `<video>/<camera>` 로 중첩시키려고 metadata 의
  `name` 열만 `<video>/<camera>` 로 바꾼 사본(`meta_nested.csv`)을 쓴다 — 두 스크립트 다 `name`
  을 그대로 하위 경로로 이어붙이므로 Vista4D 코드 수정 0줄. `gen/<video>/<camera>/` 에
  `point_cloud.mp4`(=depth warp) 와 `video_seed=<seed>.mp4`(=생성) 가 같이 떨어져 memory
  `show-depth-warp-with-output` 조건을 자동으로 만족한다. 인자: `STAGE=render|gen|all`,
  `RES=384p|720p`, `FSFF=1`, `NUM_SHARDS`/`SHARD_ID`.
  카메라 npz 에 키가 두 벌인데(`cam_c2w` / `cam_c2w_fsff`) **기본값은 frame 0 이 소스 카메라와
  일치하지 않는다** — 우리 rcm3 anchor 규약과 다르니 대조할 때 주의. `FSFF=1` 이 그 차이를
  없앤 변형이고 출력도 `_fsff` 로 갈린다.
- `results/20260818_rcm3_gen/` (신규) — rcm3 x 카메라 6종 TrajectoryCrafter **좌표계 수정 +
  절대 pose** 렌더/생성. 기존 `20260813_camgrid_stage1/rcm3/trajectorycrafter/` 는 14개 태그가
  전부 `_legacyconv/` (2026-08-14 이전 버그난 규약, 절대 pose 아님) 라 새로 판다.
  `run_rcm3_trajc.sh` = `--tc_abs_pose <rcm3 sierp DA3 npz> --tc_radius_meanray --scales 0.4`
  (`s = 2g`, g=0.2). trajc 카메라 npy 가 소스 무관 상수임을 md5 로 확인(taylor c01 과 동일).
  6/6 rc=0 완료. `make_concat.sh` 는 vista4d 판과 같은 무-`{clip}` 경로 규약을 쓰고, input 행에
  원본 `3.mp4`(1195프레임) 대신 `20260818_vista4d/src/rcm3_first49.mp4` 를 넣는다 — 모델은 앞 49
  프레임만 읽으므로 원본을 붙이면 input 행만 다른 구간을 보여준다.
- `tools/recammaster/warp_grid_video.py --transpose` — 격자를 행=카메라 / 열=`--rows` 항목으로
  뒤집는다 (기본은 그 반대, 무플래그 동작 불변). 한 카메라의 depth warp 과 gen 을 **가로로
  나란히** 놓아 같은 프레임에서 warp 의 hole 이 그대로 생성으로 갔는지 바로 대조하려는 것
  (memory `show-depth-warp-with-output`). 패널 라벨은 셀마다 `<카메라> | <행이름>` 그대로다.
- `tools/recammaster/sd_revpair_prepare.py` (신규) — Scene-Decoupled 같은 scene 의 clip 2개를
  `rev(A)[80..1] + B[0..80]` (161장) 으로 이어 붙여 **한 번에** DA3 를 돌려 두 clip 의
  depth/카메라를 한 좌표계·한 스케일로 만든다 (clip 간 frame0 카메라가 동일해서 역재생이
  자연스럽게 이어진다). `--split_recon` 이 clip 별로 되썰고 `--verify` 가 SD 의 GT 미터
  pose 와 대조한다 — clip 간 스케일 차이 0.33% / 1.96% / 2.29% (`results/20260818_sd_revpair/config.md`).
  - 4번째 pair `scene1002_5x5_loc37_scene_Dragon_Rise__rev-05_24mm__01_24mm` 추가 (수동 `--pairs`,
    사용자가 val 목록에서 찍은 조합). clip 간 s 차이 **1.10%**. `viser_revpair.py` 는 `clip_a` 를
    context 로 칠하므로 val 의 `{scene}__{TARGET}__{CONTEXT}` 와 색을 맞추려면 `--pairs
    <scene>:<CONTEXT>:<TARGET>` 순서로 넣어야 한다 (기존 3 pair 는 `rev-01__05` 라 색이 반대).
    `--pairs` 경로는 `dir_angle_deg`/`min_path_len_m`/`resid_*` 를 안 재고 `nan` 으로 둔다.
    `pairs.json` 을 통째로 덮어쓰므로 기존 3개 기록은 수동으로 합쳤다.
- `results/20260818_taylor_gen/make_concat.sh` (신규) — 모델별 concat 영상
  (행 = input / depth warp / gen, 열 = 카메라 6종).
- `results/20260818_vista4d/make_concat.sh` (신규) — 같은 3행 concat 의 Vista4D 판.
  경로 층이 달라(`out/<gtag>/<src>/vista4d/<cam>_<gtag>/`, `<clip>` 층 없음) `{clip}` 대신
  `{src}`+`{cam}` 템플릿을 쓴다. 소스 mp4 는 stage 1 에 실제로 먹인 것(taylor 는 앞 49 트림).
- `tools/recammaster/rerope_prepare.py` (신규) — 우리 소스/타깃 카메라를 **ReRoPE V2V 네이티브
  입력**으로 변환한다. ReRoPE 는 벤치마크 6종 중 유일하게 **소스(context) 궤적을 따로** 요구
  하므로(`src/v2v_handler.py:121` 이 `videos/<stem>.npz` 에서 c2w 를 읽는다) depth warp 계열이
  쓰는 것과 **같은 DA3 재구성**을 context 로 넣어 "같은 카메라 쌍" 비교가 성립하게 했다.
  - 출력: `<out>/data/{metadata.csv, videos/<stem>.mp4(심링크), videos/<stem>.npz}` +
    `<out>/targets/<camera>.npz` + `<out>/prepare.json`(진단 포함).
  - 소스 npz = DA3 `extrinsics`(w2c) 를 inv 해서 c2w, translation **마지막 열**.
    타깃 npz 는 `--target_frame` 이 정한다 (아래 Fixed 항목). 기본 `world` =
    `(C_src0 @ conv(json.T)).transpose(0,2,1)` — **소스와 같은 규약의 c2w 를 전치만** 한 것.
  - **`/100` 을 여기서 한다.** ReRoPE 코드에는 ReCamMaster 의 `c2w[:3,3] /= 100` 이 없어서
    UE5-cm 를 그대로 주면 타깃이 소스의 100배가 되고, `normalize_joint_translation` 이 둘을
    같은 max‖t‖ 로 나누므로 context 궤적이 cond‖t‖≈0.01 로 뭉개진다. world offset
    `[5000,1500,100]` 도 같이 뺀다.
  - `--selftest`: `opencv_to_ue5` → (`transpose` + `convert_c2w_convention`) 왕복이 항등임을
    확인 — 치환 `[2,0,1,3]∘[1,2,0,3]` = identity, 두 flip 이 **같은 열**에 걸려 상쇄.
    max err 4.7e-15. (이 왕복이 항등인 것과 **ReRoPE 가 그 결과를 원하는 것은 다른 얘기**였다.
    아래 Fixed.)
  - 진단이 `normalize_joint_translation` 를 재현해 카메라별 `joint_max / cond‖t‖ / tgt‖t‖ /
    비율` 을 표로 찍고, `joint_max < eps(=1)` 이면 미학습 regime 경고를 낸다. sC 실측:
    joint_max 3.7037, cond 1.0000, tgt 0.7585 (6종 동일).
  - 진단에 `cdet / tdet / tgtrot0 / tgtrotmax` 4열 추가 + 경고 2개: (a) cond/tgt 의 conv-후
    det 부호가 **서로** 다르면 규약 불일치, (b) `tgtrot0 > 1°` 면 타깃이 소스 world 에 안
    심긴 것. 이 두 열이 아래 Fixed 의 버그를 잡아낸 계기다. `--axis_precomp inv_conv` 에서는
    `[축검증]` 단계가 추가로 내부 타깃 상대 w2c == JSON 로컬 궤적을 확인한다.
- `warp_grid_video.py --rows` 에 `{cam}` placeholder 추가 (기존 placeholder 5개는 그대로) —
  배율 태그가 안 붙은 canonical 이름. ReRoPE 는 출력 파일명이 `<stem>_<camera>.mp4` 라
  `{tag_s}`/`{tag_t}` 로는 경로를 만들 수 없었다.
- `tools/recammaster/run_trajcrafter_matrix.py --abs_pose <npz|npy>` (기본 None = 기존 상대
  pose 동작) — TrajectoryCrafter 를 **SierpinskiCam 과 같은 절대 pose semantics** 로 돌린다.
  기존에는 `pose_s = poses[anchor_idx].repeat(N)` (demo.py:549) 이라 실현 변환이
  `T = inv(rel_t)` 였고 **소스 카메라 자기 운동이 그대로 남았다**. sierp 는
  `pose_s = c2ws`(프레임별 DA3 포즈) 라 `T = target ∘ inv(source)` 로 소스 운동을 취소한다
  (`results/20260817_dl3dv_warp/config.md` §4 가 그 차이를 실측). `--abs_pose` 는
  `pose_s_i = inv(rel_src_i) @ P0` 를 주어 `T = inv(rel_t) @ rel_src` 로 같은 형태를 만든다.
  - 입력이 `.npz` 면 sierp 조건 렌더가 남긴 DA3 결과를 그대로 쓴다 — `rel_i = E_0 @ inv(E_i)`
    로 frame0 재앵커(sierp `--anchor0` 과 같은 보정; sC 소스는 `|E_0 - I| = 2.92`, ref view
    가 frame 37) 후 같은 npz 의 depth+intrinsics 로 잰 `S_da3` 로 나눠 무차원화.
  - `--radius_meanray` 를 **요구**하고 `--target_dxw`/`--free_anchor` 와는 **배타**다
    (전자는 radius 를 target 궤적에서 역산해 소스 궤적까지 리스케일하고, 후자는 `pose_s` 를
    이중 정의한다).
  - 검증: (1) `--abs_pose` 없으면 `pose_s` 가 기존과 **바이트 동일**, (2) `rel_src = I` 면
    `T` 가 기존과 차이 0.0, (3) `rel_src = rel_t` 면 `T = I` (잔차 1.8e-15), (4) selftest
    worst 8.8e-08 유지. 실측(sC/truck_left): 소스 자기 운동 `|t|max = 0.2637*S`, rot 7.50°
    -> 실현 `|t|max 2.8094 -> 1.4446` (0.514배), rot `0.00° -> 7.50°` (sierp 실측 7.50° 와 일치).
  - `_report_realized()` 가 실현 `T` 의 `|t|max`/rot 를 상대 semantics 값과 나란히 찍는다
    (런타임 warp 은 픽셀로 못 재므로 — memory `measure-realized-fails-on-runtime-warp`).
- `run_grid.py --tc_abs_pose <src_track>` (기본 None = 기존 상대 pose 동작) — 위 `--abs_pose`
  를 그리드 러너에서 쓸 수 있게 플럼했다. 값은 sierp 조건 렌더가 이미 남긴 DA3 npz
  (`<out_root>/<src_stem>/sierpinskicam/<cam>_s*/_cond/cam/<src_stem>.npz`) 를 그대로 준다 —
  소스 재구성은 요청 카메라와 무관하므로 6종 중 아무거나 하나면 된다 (sC 실측: 6개 npz 의
  `rel` max dev 2.6e-4, `S_da3` 14.0446~14.0661).
  자식 스크립트도 같은 조합을 막지만 entry 마다 spawn 한 뒤 죽으면 로그가 흩어지므로
  **fail-fast guard 4개**를 러너 쪽에 뒀다: 파일 없음 / `--model` 이 trajectorycrafter 가 아님
  (sierp 는 원래 절대 pose) / `--tc_radius_meanray` 없음 / `--tc_target_dxw`·`--free_anchor`
  충돌. 4개 전부 발화 확인, `--dry` 6/6 정상.
- `warp_grid_video.py --tc_abs_root <root>` (기본 None = 3행 = 기존 동작) — 주면
  `TrajectoryCrafter render (abs pose)` 행이 4번째로 붙는다. **같은 모델 / 다른 pose semantics**
  를 세로로 나란히 놓으려는 것으로, 기존 `ROWS` 는 root 가 행마다 하나씩 고정이라 이 축을
  표현할 수 없었다. 실측 확인: sC/g0p2 6카메라 4행 `(49, 600, 1620, 3)`, 빠진 패널 0개.
- `warp_grid_video.py --rows LABEL=TEMPLATE ...` (기본 None = 기존 `ROWS` 3행) — 행 구성을
  통째로 갈아끼운다. placeholder `{root} {abs_root} {clip} {src} {tag_s} {tag_t}`, 빈 TEMPLATE
  = 소스 패널. **같은 모델의 warp 과 gen 을 나란히** 깔려고 넣었다 (memory
  `show-depth-warp-with-output`) — `ROWS` 는 행마다 모델이 하나씩 고정이라 그 축이 안 나온다.
  회귀 확인: 무옵션 3행 `(49,450,540,3)` 그대로, `--rows` 2행 `(49,300,540,3)`,
  `LABEL=` 없는 spec 은 argparse error.
- `tools/recammaster/warp_direction_check.py` (신규) — 런타임 depth warp 의 **이동 방향**을
  hole(정확한 검은색) 중심 좌표로 판정한다. SIFT/Farneback(`measure_realized.py`)은 구멍이
  flow 를 0 으로 끌어 배율을 6배까지 과소보고하므로 런타임 warp 에는 못 쓴다. dome 으로
  구멍을 메우는 `dense_tx` 에 쓰면 rate 가 과소로 나와서 `--warn_filled` 가 경고한다.
- `run_grid.py --sierp_save` (기본 `rgb,dense_tx` = 기존 동작 = stage B 가 먹는 것) —
  진단용으로 구멍을 안 메운 순수 forward warp(`dense`)과 `mask` 를 같이 받을 수 있게 했다.
- `tools/recammaster/warp_grid_video.py --src_stem` (기본 None = `taylor_<clip>` = 기존 동작) —
  경로 템플릿에 `taylor_<clip>` 이 박혀 있어 taylor 이외의 소스(`hs_sC` 같은 DL3DV half-split
  소스)로는 못 쓰던 걸 `run_grid --src` 값과 맞춰 열 수 있게 했다.
- `tools/recammaster/warp_grid_video.py --align {head,resample}`: 기본 `head` 는 앞 n 장 1:1 정렬. 두 모델 다 소스 프레임 0..48 을 stride 1 로 먹는데, SierpinskiCam `_warp/warp.mp4` 는 moviepy `duration=49/12` 부동소수 잔차로 마지막 프레임이 복제돼 50 장이라 인덱스 비례 리샘플하면 sierp 행이 최대 1 프레임 앞서갔다. `resample` 로 구 동작 유지.
- `make_sbs.py --sep` / `--src_ext` (기본 `__` / `.mp4` = 기존 동작) — 결과 파일 stem 의
  `<소스><sep><카메라>` 구분자와 소스 확장자를 고를 수 있게 했다. ReRoPE 는 `1_cam01` 처럼
  `_` 하나를 쓴다.
- `tools/recammaster/warp_grid_video.py` (신규) — 한 소스에 대해 **열 = 카메라 / 행 = 모델**로
  depth warp 을 한 화면에 까는 비교 영상 빌더. `make_compare.py` 는 열이 모델이라 축이 다르다.
  길이가 다르면(sierp 50 / trajc 49) 인덱스 비례 리샘플 — 궤적이 전부 canonical 21 pose 를
  자기 길이로 편 것이라 정규화 시간 `t` 가 같은 pose 를 가리킨다.
- `run_grid.py --warp_only` (기본 꺼짐 = 기존 동작) — depth warp 까지만 돌리고 생성(diffusion)을
  건너뛴다. sierpinskicam 은 stage A(`create_sierpinskicam_conditioning`)만 돌고 stage B 를 빼며
  `_warp/warp.mp4` 는 그대로 남는다. trajectorycrafter 는 `--render_only` 로 `render.mp4` 직후
  종료한다. 카메라 세기를 눈으로 정하는 단계에서 모델 로드 + 샘플링 비용을 안 낸다 (warp 은
  결정론적이라 세기 판정에 생성 결과가 필요 없다).
- `run_grid.py --tc_radius_meanray` / `run_trajcrafter_matrix.py --radius_meanray`
  (기본 꺼짐 = 기존 동작인 중심픽셀 depth + `min(r,5)` clamp) — TrajectoryCrafter 의 `radius` 를
  DepthCrafter 자기 depth 의 **mean ray length** `S = mean(z * ||K^-1 [u+.5,v+.5,1]||)` 로 잡는다.
  npy 단위가 `0.5*radius` 이므로 `emit --scales 2g` 와 짝지으면 `|t|max = g*S` 가 되어, **g 하나가
  모든 소스·모델 공통인 "scene scale 대비 이동량" 손잡이**가 된다. `--target_dxw` 와 달리 target
  궤적을 안 보므로 소스당 상수라 카메라 여러 개를 같은 세기로 비교할 수 있다. 같이 주면
  `--target_dxw` 가 이긴다.
- `tools/recammaster/sierp_scene_scale.py` (신규) — 소스 영상의 scene scale `S` 를 SierpinskiCam 이
  쓰는 게이지(DA3NESTED-GIANT-LARGE, 영상 앞 49프레임)로 오프라인 측정한다. 정의는 DL3DV 의
  저장된 `avg_scale` 과 같은 frame0 mean ray length. sierp 는 warp 전에 카메라 JSON 을 확정해야
  해서 런타임 계산(trajc 의 `--radius_meanray`)이 불가능하다. emit 은 `--scales g*S/1.9876`
  (`native_1x` 1.9876 m 를 나눠 준다) 로 준다.
- `run_grid.py` 의 소스 `hs_sC` — 옛 `hs_static` 을 대체하는 DL3DV static 소스.
  옛 것은 half-split 경계에서 rot48=170도라 전이 warp 의 hole 이 0.846 이었다 (context 끝과
  target 끝이 서로 뒤를 봤다). sC 는 rot48=11.03 / tau48=0.514 로 hole 0.381.
  선정 근거와 985 scene 스크리닝 표는 `results/20260817_hs_dl3dv3/config.md`.
- `tools/recammaster/halfsplit_fullrecon_transfer.py` (신규) — **GT 카메라 없이** target 카메라를
  만든다. ctx+tgt 를 한 번에 DA3 recon 한 뒤, ctx 만 넣어 나온 **모델 게이지 ctx 카메라**에
  umeyama sim3 로 full recon 전체를 이전하고, 이전된 target 카메라로 모델 ctx depth 를 렌더한다.
  배율이 sim3 의 `s` 로 닫혀 나와 `halfsplit_sweep.py` 의 `f` 스윕이 필요 없다 (검증용으로
  `--sweep` 은 남겨 뒀고, 최적점이 1.0 이면 이전이 맞은 것). 카메라를 안 내는 TrajectoryCrafter
  (DepthCrafter) 는 `--align depth` 로 depth 비에서 `s` 를 뽑는다. 대조군 `--cam_src`:
  `transfer`(이 방법) / `gtalign`(GT ctx 카메라에 맞춰 이전 — recon 의 target 외삽 품질만) /
  `gt`(GT 카메라 상한). 결과: `results/20260817_fullrecon_fix8/config.md`.
- `tools/recammaster/alaya_run_dumpwarp.py` (신규) — AlayaWorld 를 돌리면서 내부 depth warp 을
  `--warp_out <경로>` 로 뽑아 준다. vendored 코드는 안 고치고 우리 드라이버에서 monkey-patch 한다.
  AlayaWorld 는 warp 을 디스크에 안 남겨서 (memory `alayaworld-is-depth-warp-model`) 무슨 조건이
  들어갔는지 눈으로 확인할 방법이 없었다. GEN3C 처럼 10장 z-buffer 누적이라는 것도 이걸로 확인했다.
- `tools/recammaster/run_grid.py` 의 `SAVE_WARP` (기본 켜짐) / `--no_save_warp` (= 기존 동작) —
  alayaworld / sierpinskicam 의 depth warp 중간 영상을 `<out>/_warp/warp.mp4` 로 남긴다.
  sierpinskicam 은 stage A 가 만든 `dense_tx` 를 복사하고, alayaworld 는 위 dumpwarp 을 탄다.
- `tools/recammaster/camviz_from_native.py` (신규) — 모델별 **native 카메라 파일**(recammaster/
  infcam/sierpinskicam JSON, cameraanything JSON, alayaworld `_camera.pt`, trajC `.npy`)을 그대로
  읽어 궤적을 3D 로 그린다. canonical 이 아니라 **모델이 실제로 먹은 파일**을 보므로 emit 단계의
  anchor/단위 변환 실수가 여기서 잡힌다.
- `results/20260815_magladder/` — 모델별 강도 사다리(3 scale × 3 camera × 4 model = 36 렌더)와
  `measure/all.json`. `results/20260816_magmatch/` — 그 사다리로 역산한 scale 의 검증 렌더 +
  recammaster/trajectorycrafter 신규 사다리 + alayaworld 포화 probe. 둘 다 `config.md` 동봉.
- `measure_realized.py --profile` — 프레임별 **누적** dx/W, dy/W, zoom 궤적을 JSON `curve` 키에
  담는다 (기본 꺼짐 = 기존 출력 그대로). 총 이동량이 같아도 시간 분포(=속도)가 다를 수 있어서
  총량만으로는 "같은 세기"인지 판정이 안 된다. 이걸로 alayaworld 만 프레임당 속도가 절반이고
  앞 20% 가 늘어진다는 것을 잡았다.
- `make_warp_compare.py --rows N` — 패널을 N 줄로 접는다 (기본 1 = 기존 한 줄 동작).
  패널이 7개쯤 되면 한 줄은 너무 납작해서 안 보인다. 줄마다 총폭이 다르면 오른쪽을 검정 패딩
  (짝수 폭 유지 — 홀수면 libx264 가 조용히 죽는다).
- `run_grid.py --aw_rounds N` (기본 3 = 기존 동작) — alayaworld 롤아웃 round 수. 출력 프레임 =
  `rounds*32` 라 클립 길이를 이걸로 맞춘다. `emit_model_cams.py --n_frames` 도 같이 바꿔야 한다
  (`run_grid.py` 상단 길이 산수 주석 참조).
- `results/20260816_warp3/` — warp 3종(sierp/trajC/alaya)을 dx/W = 0.22 로 올리고 클립 길이를
  맞춘 렌더. sierp 0.2215 / trajC 0.2287 로 도달, **alayaworld 는 0.115 에서 포화**
  (s = 1.8 / 3.6 / 5.85 → 0.1110 / 0.1150 / 0.1149). `config.md` + `measure/all.json` 동봉.
- `run_grid.py --aw_cfg <yaml>` (기본 None = 저자 기본 `configs/infer.yaml` = 기존 동작) —
  alayaworld 에 다른 inference yaml 을 `--cfg` 로 넘긴다. vendored 파일은 안 고치고 config 를
  복사해서 쓴다 (`configs/infer.yaml` 의 `paths:` 는 cwd 기준 상대경로라 yaml 위치는 무관).
  쓸모: **`da3_align_to_input_scale: false`**. 기본값 `true` 면 DA3 가 추정한 depth 를 **우리가
  준 extrinsic 의 스케일에 맞춰 다시 스케일**하므로 (`flash_alaya/alaya/memory/da3_depth.py:119`
  → `spatial.py:116` → `schema.py:160`) translation 을 2배로 줘도 depth 가 같이 2배가 되어
  시차 `t/Z` 가 불변 = **emit scale 이 아예 안 먹는다.** 이게 "alayaworld 포화" 의 실체다.
- `results/20260816_warp3/cfg/infer_noalign.yaml` — 위 플래그만 `false` 로 바꾼 config 사본.
  `out_noalign/` (s 0.9/1.8/3.6) + `out_noalign_small/` (s 0.11/0.22/0.44) 렌더가 이걸 쓴다.
  같은 s=0.9 / 같은 64 프레임에서 warp dx/W 0.0733 → **0.8974**, s=1.8 에서 0.1123 → **1.5393**
  으로 scale 이 정상 반응한다. 다만 그 크기에서는 warp 이 구멍투성이가 되어 생성기가 정지하므로
  (출력 dx/W 0.0042 / 0.0001) 사다리를 s ≈ 0.1~0.4 로 내려서 8점 찍었다. **다만 alayaworld 는
  출력이 s 에 대해 단조가 아니다** — s 0.27 -> 0.28 (+3.7%) 에서 dx/W 가 0.2283 -> 0.1723 으로
  25% 떨어진다. warp 은 0.1344 -> 0.5107 로 완벽히 단조인데 출력만 튄다 (출력/warp 0.508~0.964,
  seed 고정). 따라서 **alayaworld 는 아직 0.22 에 정합됐다고 볼 수 없다** (sierp 0.7218/0.2215,
  trajC 2.086/0.2287 은 정합 완료). `config.md` 에 8점 사다리 + 재정정 동봉.
- `run_grid.py --gen_seed N` (기본 None = seed 를 안 넘김 = 저자 기본 sierp 42 / trajC 43 /
  infcam 0 / cameraanything 0 = 기존 동작) — sierpinskicam / trajectorycrafter / **infcam /
  cameraanything** 의 생성 seed. **실현 이동량의 seed 분산**을
  재려고 넣었다. alayaworld 는 같은 scale·같은 클립 길이에서 seed 만 바꿔도 dx/W 가
  s=0.27 에서 0.2119~0.2532 (mean 0.2339, sd 6.6%), s=0.28 에서 0.1723~0.2479
  (mean 0.2013, sd 14.0%) 로 퍼진다. 두 구간이 크게 겹치므로 앞서 "s 0.27->0.28 에서 25% 하락 =
  비단조" 라고 본 것은 **seed 0 한 번의 draw** 였다 (t≈1.8, n=4 씩, 유의하지 않음).
  sierp/trajC 도 같은 잣대로 seed 4개씩 재측정한 결과 **정합이 확정됐다**:
  sierp 0.7218 → 0.2215/0.2252/0.2215/0.2219 (mean 0.2225, sd **0.82%**),
  trajC 2.086 → 0.2287/0.2301/0.2270/0.2297 (mean 0.2289, sd **0.50%**).
  alayaworld 의 sd 만 이 둘의 8~17 배다.
- **판정: alayaworld 를 강도(dx/W) 정합 축에서 제외한다.** 사용자 기준이 "생성한 카메라 궤적을
  정확히 실행" 이므로 벤치마크 공정성이 아니라 궤적 실행 정확도로 판정했다. ① 출고 config 는
  카메라 크기를 무시(s=1.314 → 0.2543 vs s=2.0 → 0.2541), ② 정렬을 꺼도 고정 s 에서 sd 6.6~14.0%,
  ③ 실효 s 구간이 0.1~0.4 로 좁고 밖에서는 생성 정지/warp 붕괴, ④ joystick 분기
  `camera_control.py::c2w_to_action_labels` 가 c2w 를 WASD one-hot 으로 이산화해 **방향만** 받는다
  (게다가 우리 세팅은 I2V). 소스를 169 프레임 이상으로 바꿔 V2V 로 만들어도 ②③④ 는 남는다.
  정성 비교용("방향 컨트롤러")으로만 유지. `results/20260816_warp3/config.md` 에 최종 표 동봉.
- `results/20260816_warp3/compare_alaya_align.mp4` / `compare_warp3_matched.mp4` — 2줄
  (위 출력 / 아래 depth warp) 비교 영상. 전자는 alaya 정렬 ON vs OFF, 후자는 정합된 3종.
- `results/20260816_match22/` — infcam / cameraanything 를 dx/W 0.22 에 정합 시도 +
  **전 모델 seed 분산을 n=8 로 통일**. 결론: **precise camera control 후보는
  sierpinskicam / trajectorycrafter 2종만 남는다.**

  | 모델 | s | mean dx/W | 목표대비 | seed sd (n=8) | max/min | 구조 |
  |---|---|---|---|---|---|---|
  | sierpinskicam | 0.7218 | 0.2242 | +1.9% | **1.03%** | 1.024 | warp+inpaint |
  | trajectorycrafter | 2.086 | 0.2306 | +4.8% | **1.77%** | 1.052 | warp+inpaint |
  | infcam | 0.6623 | 0.2501 | +13.7% | 17.94% | 1.748 | 카메라 조건부 생성 |
  | cameraanything | 0.82 | 0.2462 | +11.9% | 17.73% | 1.916 | 카메라 조건부 생성 |
  | alayaworld | 0.27 | 0.2339 (n=4) | +6.3% | 6.62% | — | 카메라 조건부 생성 |

  **분류선이 모델 구조를 그대로 따라간다** — 카메라를 기하로 먼저 실행하고 생성기엔 구멍만
  메우게 하는 쪽은 sd 1~2%, 카메라를 생성기 조건으로 넣는 쪽은 6~18%. 두 그룹 사이에 겹치는
  값이 없다. cameraanything 은 dy/W 가 8개 seed 전부 양수(mean +0.0070)라 순수 좌평행이동인데
  위로 밀리고 zoom 도 0.99~1.15 로 흔들린다 — 이동량뿐 아니라 방향도 어긋난다.
  n=4→n=8 로 늘리자 sd 가 셋 다 올랐다 (sierp 0.82→1.03, trajC 0.50→1.77, infcam 12.70→17.94)
  — **n=4 수치는 전부 낙관적이었다.**

- `tools/recammaster/ctx_align.py` (신규) — **context align**. depth warp 모델의 이동량을
  "화면폭 대비 얼마나 움직일 것인가"(`target dx/W`)로 지정하고, 그 값이 나오는 translation
  배율 `k` 를 **모델 자신의 context depth 로 수치 역산**한다. `predict_flow` 가
  (depth, K, rel c2w) 로 median flow 를 직접 계산하고 `solve_scale` 이 bisection 으로 푼다.
  `--selftest` 5 케이스 (planar truck 해석해 대조 / spread depth vs `D·z_med/(fx/W)` /
  +15° yaw / 회전만으로 목표 초과 시 SystemExit / 회전·병진 부호 반대인 비단조 케이스).
  왜 해석식이 아니라 수치해인가: 회전이 섞이면 `k = D·z_med/(fx/W)` 가 **55.6% 틀린다**
  (selftest 3). 그리고 `|dx(k)|` 는 회전 기여와 병진 기여의 부호가 반대일 때 비단조라
  bisection 을 부호 붙인 `g(k)=sgn·dx(k)` 위에서 돌린다.
- `emit_model_cams.py --target_dxw / --align_npz / --align_fx_over_w` (기본 None = 기존
  native 1x 동작) — sierpinskicam / alayaworld 카메라를 native 배율 대신 context align 으로
  낸다. depth 는 SierpinskiCam 이 남긴 DA3 npz 를 쓰고, `fx/W` 는 모델마다 다르므로
  (sierp 0.3578 = DA3 추정치, alaya 0.4482 = 우리가 넘기는 정규화 K) 후자만 명시로 준다.
  `manifest.json` 에 `ctx_align` 블록(z_med, per-camera k / rot_only / pred dx/W / tau)을 남긴다.
  trajectorycrafter 는 depth 가 DepthCrafter 클립 min-max 정규화라 여기서 못 풀고 raise.
- `run_trajcrafter_matrix.py --target_dxw` (`--camera matrix` 전용, 기본 None = 기존 동작) —
  trajectorycrafter 의 `radius` 를 DepthCrafter 자기 depth 로 **런타임에** 역산한다.
  `run_grid.py --tc_target_dxw` 로 넘긴다.
- `tools/recammaster/sierp_run_freeanchor.py` (신규) + `run_grid.py --free_anchor` (기본 꺼짐 =
  기존 동작) — trajectorycrafter / sierpinskicam 이 `rel[0] != I` 궤적의 상수 offset 을 살린다
  (`get_c2w` monkey-patch). `emit_model_cams.py --hold` probe 짝.
- `emit_model_cams.py --hold / --anchor` (기본 꺼짐 / auto = 기존 동작) — 궤적을 끝점에서 정지한
  **상수 offset** 으로 바꾼다. "context 첫 view 가 아닌 곳에서 렌더가 되는가" probe 용.
- `tools/recammaster/measure_offset.py` (신규) — **소스 프레임 k ↔ 출력 프레임 k** flow 로
  offset 이 실현됐는지 잰다. `measure_realized.py` 는 클립 **안**의 누적만 봐서 상수 offset 을
  0 으로 읽는다. `--warp` 로 depth warp 중간물도 같은 방식으로 잰다.
- `measure_offset.py --method sift` (기본 `farneback` = 기존 동작) — SIFT + Lowe ratio + MAD
  필터로 대응점을 잡는다. Farneback 은 winsize 41 (width 512) 로도 수십~백 px 짜리 offset 이
  capture range 밖이면 **0 으로 수렴**해서 "offset 이 안 실현됐다"와 구별이 안 된다. 실제로
  `/tmp/off_full.json` 이 전 모델 off_dx≈0 으로 읽혔는데 픽셀 L1 diff (infcam 16.4→57.3) 와
  모순이었고, sift 로 다시 재자 결과가 뒤집혔다.
- `ctx_align.solve_scale(metric='max')` (기본 `'end'` = 기존 동작) — 목표 `dx/W` 를 끝 pose 가
  아니라 **렌더되는 pose 전체의 |dx/W| 최댓값**에 맞춘다. `dl3dv_L81_a_r` 은 나갔다 돌아오는
  궤적(path/chord 2.18)이라 화면 변위가 중간에 −0.538 까지 갔다 끝에서 −0.110 으로 돌아온다 —
  끝점만 맞추면 "세 모델이 같은 양 움직인다" 가 성립하지 않는다. 단조 궤적에서는 두 metric 이
  같은 답을 준다 (selftest 5 가 이걸 assert 한다). 반환값에 `peak_idx` / `end_dxw` 추가.
  `--metric` / `--n_poses` CLI 플래그도 같이 (`--n_poses` 는 모델이 실제 렌더하는 앞 N pose 만
  쓰기 위한 것 — SierpinskiCam 은 JSON 키를 `[::4]` 가 아니라 **순서대로** 49개만 읽는다,
  `create_sierpinskicam_conditioning.py:651`, 즉 81키 궤적의 앞 60% 만 실행한다).
- `emit_model_cams.py --rcm_render_frames N` (기본 None = 기존 동작) — rcm 계열 JSON 에서
  모델이 **실제로 읽는** 키 개수에 궤적을 펴고 나머지 키는 마지막 pose 로 유지한다.
  SierpinskiCam 은 `--frame-count 49` → `range(49)` 로 **앞 49키만** 읽으므로 81키에 펴면
  궤적의 60% 지점에서 끊긴다 (실측: dynA end |t| 0.5475 → 0.3010 = 54.6%, static
  4.1322 → 3.3706 = 81.6%). 실측 배율 표(20260816 계열)는 이 절단을 배율 안에 흡수했으므로
  **예측 배율을 쓰는 경로(full-recon sim3 전이)에서만** 켠다.
- `emit_model_cams.py --aw_intr FX_W FY_H` (기본 None = 저자 예제 값 `(0.4482, 0.7966)` 고정
  = 기존 동작) — alayaworld `_camera.pt` 의 정규화 intrinsic 을 소스의 실제 FOV 로 바꾼다.
  AlayaWorld 는 고정 FOV 96도 모델인데 소스는 다르다 (DA3 추정 fx/W: dl3dv_s1 0.5006 = 90도,
  dynrep_s2 0.5501 = 84도). 넣으면 spatial memory 의 warp 기하는 소스와 맞지만 카메라 임베딩이
  학습 FOV 를 벗어난다 — 그래서 두 팔(`gen_aw_nat` / `gen_aw_srck`)을 따로 돌린다.
- `tools/recammaster/sierp_run_freeanchor.py --anchor0` / `run_grid.py --sierp_anchor0`
  (둘 다 기본 꺼짐 = 기존 동작) — SierpinskiCam warp 의 **출발 pose(DA3 extrinsics)를 ctx
  frame0 으로 재앵커**한다. vendored 코드가 `extrinsics[0] == I` 를 가정하는데 **DA3 는
  reference view 를 자기가 고른다** (실측 ref index: dynA 2/6, static 42/35 — static 은 frame0 이
  74.7° / |t| 5.71 어긋나 있었다). vendored 파일 0줄 수정, `smooth_camera_path` 를 monkey-patch.
  (`FIX.log` FIX-9)
- `tools/recammaster/halfsplit_source_mp4.py` (신규) — half-split 의 **context 절반을 역재생**해서
  소스 mp4 로 굽는다. `recon/*.json` / `prior_rev/*.json` 이 남긴 `frames_dir` + `frame_idx` 를
  그대로 읽으므로(`--from_json`) DA3 prior 를 뽑을 때와 **정확히 같은 프레임 집합/순서**가
  보장된다 — 다르면 모델 런타임 depth/카메라가 prior 와 달라져 sim3 로 푼 `s` 가 안 맞는다.
- `run_grid.SOURCES` 에 `hs_dynA` / `hs_static` 추가 — full-recon 전이용 half-split 소스
  (49 프레임뿐이라 81 프레임을 요구하는 recammaster/infcam 은 못 돌린다).
- `tools/recammaster/halfsplit_compare_grid.py` (신규) — `halfsplit_fullrecon_transfer.py` 가 낸
  모델별 mp4 를 가로로 이어 `compare_<src>.mp4` 를 만든다 (각 열 = 모델, 각 열은 이미
  위 = 전이 warp / 아래 = 정답 2행). `results/20260817_fullrecon_fix8/compare_*.mp4` 와 같은 배치.
- **detector 어휘 제안기 6종 비교** (`results/2026-08-20_detector_compare/{sam3_timing_mem,
  gsam2_timing,florence2_timing,detector_boxes_overlay}.py`). 같은 영상 2편(camel /
  avocado-slice) · 같은 `Meter`(torch peak alloc·reserved, nvidia-smi per-PID, `/proc` RSS, 5 Hz)
  로 wall-clock + 자원을 잰다. 설치 0건 — `pyshim/` 심링크 + monkeypatch 로 해결.
  - **비교의 전제가 틀려 있었다.** 지금까지 "VLM keyword" 라 부른 것은
    `Vista4D-Eval-Data/metadata.csv` 의 `dynamic` 열, 즉 **Vista4D 저자가 손으로 적은 정답
    명사**다 (`sam3_seg_instances.py:30-42 read_keywords()` 가 쉼표로 자를 뿐, 모델이 없다).
    avocado-slice 는 그 목록에 `avocado`/`knife` 가 이미 들어 있어 keyword arm 3종에 정답이
    유출된다. 오버레이 라벨을 `authored kw` 로 바꿨다.
- **`CinemaTraj/scripts/extract_nouns_vlm.py` (신규)** — 계획서 Stage 0 의 실제 구현. 프레임 →
  Qwen3-VL-30B-A3B (로컬 vLLM :22002) → `{dynamic, static, subject}` JSON. `lbm/vlm.py` 의
  `chat_json` + 스키마 validator 를 그대로 쓴다. `--frame_mode single`(frame0 1장 = 다른
  detector 와 동일 입력) / `multi`(0/16/32/48) 두 조건.
- `gsam2_timing.py --caption_json/--caption_frame_mode/--caption_include_static` — 위 VLM 명사를
  GDINO caption 으로 먹인다. 기존 두 경로(`--ram`, meta.json keyword)는 분기 밖에 그대로 뒀다.
- `detector_boxes_overlay.py` 6패널로 확장 — 조건 3그룹(저자 kw 3 / VLM 명사 1 / prompt-free 2).
- **`CinemaTraj/scripts/smoke_micro_ops.py` (신규)** — micro-adjust 14연산을 VLM 없이 전부
  적용해 게이트 결과를 표로 낸다. camel 첫 end-to-end 에서 VLM 이 round 0 에 `done:true` 를
  내는 바람에 **`lbm/ops.py` 가 한 번도 실행되지 않았다** — 루프가 "성공"으로 끝나도 micro
  경로는 미검증으로 남는 구조라 별도 스모크가 필요하다. 게이트 임계는
  `loop.build_parser()` 에서 공유한다(손으로 다시 적으면 테스트만 통과한다).
- `lbm/loop.py` 의 argparse 를 `build_parser()` 로 분리 (`__main__` 동작 동일).

### Changed
- **LBM-Lite: 시작 pose 를 결정 대상에서 뺐다 — `--start_mode source_frame0` 이 기본**
  (`CinemaTraj/lbm/loop.py`, `decode/build_poses.py`, `decode/emit.py`). 소스 카메라의
  `cam_c2w_world[0]` 을 그대로 시작 pose 로 쓰고 select · micro 단계를 건너뛴다 — 남는 VLM
  결정은 궤적 preset 하나(VLM 턴 3 → 1). 예전 3단 경로는 `--start_mode board` 로 그대로 돌아간다.
  근거: ① emit 기본 규약 `rel = inv(P[0]) @ P` 가 시작 pose 의 상수 offset 을 어차피 버리고
  ReCamMaster 는 `--free_start` 로 살려도 무시한다 ② 그런데 τ 예산은 먹었다 — avocado-slice 는
  `start_tau` 0.1549 로 preset 16종이 전부 포화 ③ camel 의 select 결과는 소스 pose 자신이었다
  (`d_az 0.0, d_el 0.0, dist 1.00`). 실측 avocado-slice `path_len_u` **0.0000 → 0.1533**,
  포화 preset **16 → 0**; camel 은 before/after 사실상 동일(위 ③ 때문). 자세한 진단(R1 τ 는
  움직이는 소스 기준 / R2 τ 는 사전 필터로 문서화됐지만 크기를 정하는 유일한 구속 / R3 intent
  text 부재)과 안 채택한 처방은 `CinemaTraj/DECISIONS.md` D21.
- `decision.json` 에 `start_mode` 필드 추가, `build_poses.resolve_start_mode()` 가
  **decision 을 CLI 보다 우선**한다. 모드가 어긋나면 에러 없이 "다른 궤적"만 나오기 때문이다
  (board 결정을 source_frame0 로 풀면 VLM 이 고른 pose 가 사라진다). `decision_fingerprint`
  payload 에도 넣어 stale-poses 가드가 모드 전환을 잡는다.
- `lbm/loop.py` 가 실행 시작 시 `trace/turn_*.json` · `micro_*.png` 를 지운다. `save_trace` 는
  인덱스 순으로 덮어쓸 뿐이라 이번 실행이 더 짧으면 지난 실행 턴이 섞여 남는다.
- `run_grid.py` 의 `_outputs()` 가 `source.mp4` / `point_cloud.mp4` / `point_cloud_masks.mp4`
  **파일 이름**도 제외한다. Vista4D stage 3 은 denoise **전에** 이 셋을 out_dir 바로 아래에
  쓰므로(`scripts/inference/inference.py:144-146`) 안 빼면 생성이 죽어도 `--skip_done` 이
  "완료"로 본다. 결과는 `video_seed=*.mp4` 뿐이다.
- `run_grid.py` 가 `--out_root` 를 **절대경로로 정규화**한다. 러너마다 cwd 를 자기 repo
  루트로 바꿔 돌기 때문에 상대경로를 주면 결과가 그 repo 안에 떨어진다.
- `warp_grid_video.py` 의 `tag_of()` 가 manifest 가 없으면 빈 dict 를 돌려주고 `camera`/`cam`
  두 키를 모두 받는다. `--rows` 로 sierp/trajc 아닌 모델만 깔 때 죽지 않게 (vista4d manifest 는
  키가 `cam`).
- `run_grid.py` 의 `_outputs()` 가 `_warp/` 를 결과 mp4 집계에서 **제외**한다. 안 그러면
  `--skip_done` 이 warp 만 있고 본 결과가 없는 디렉토리를 "완료"로 보고 재생성을 건너뛴다.
- `warp_ladder.sh` 의 소스/출력을 **환경변수로 갈아끼울 수 있게** 했다 (`P` / `SRC` /
  `AWIMG` / `SCENE` / `CAMERA`). 안 주면 기존 cleaning_stove probe 동작 그대로다.
  cleaning_stove 말고 다른 소스로 배율 사다리를 돌리려고 필요했다.

### Fixed
- **`build_poses` 가 소스 카메라 pose 에서 `det(R) != 1` 로 죽었다** (`--start_mode
  source_frame0` + `aim != "look_at"` preset). `cameras.cam_c2w_world` 는 DA3 w2c 를 뒤집어
  저장한 값이라 열이 정확히 단위벡터가 아니다 (`|R[:,2]|` camel 1+1.71e-5, avocado-slice
  1−2.19e-4). assert 를 완화하는 대신 `orthonormalize()` 로 SO(3) 에 한 번 투영한다 (polar
  decomposition, 회전각 변화 ~1e-3°) — 궤적이 아니라 입력이 정규직교가 아닌 문제이기 때문.
  `board` 경로는 `look_at_c2w` 가 만든 pose 라 해당 없다.
- **`CinemaTraj/decode/emit.py` 가 낡은 `poses.npz` 를 조용히 emit 했다.** 루프가
  `decision.json` 을 새로 써도 `build_poses` 를 다시 안 돌리면 emit 은 이전 실행의 궤적을
  읽는다 — 실측: avocado-slice 가 `static_hold` 결정으로 `orbit_left_arc` poses(`path_len_u`
  0.14039)를 내보냈다. 타임스탬프로는 못 잡는다(같은 초에 쓰이면 순서를 모른다).
  `build_poses.decision_fingerprint()` 가 pose 에 실제로 영향을 주는 필드만 뽑아 npz 에 심고,
  emit 이 불일치면 멈춘다 (`--allow_stale_poses` 로 우회).
- **정지 궤적에서 `rel[0] = I` assert 가 터졌다** — `static_hold`/`static_hold_locked` 는
  카메라가 제자리에서 look-at 만 돌려 `rmax` 가 float 잡음(실측 1.43e-17)인데, 예전
  `unit_scale` 이 `max(rmax, 1e-12)` 로 나눠 그 잡음을 1.39e-5 로 증폭시켰다. 규약이 깨진 게
  아니라 0/0 을 한 것. 이제 `rmax < 1e-6 u` 면 t 를 정확히 0 으로 두고 `rmax=0`,
  `translation_degenerate: true` 를 meta 에 싣고 경고를 찍는다 — 이동량 0 인 canonical 은
  `--scales` 도 ReRoPE 의 |t| 정규화도 무의미해서, 안 걸러내면 "카메라 안 움직이는 영상"이
  조용히 나온다 (`emit_model_cams --model sierpinskicam` 은 `rmax 0.0000` 로 통과시킨다).
- **`lbm/loop.py:state_line()` 조건부 f-string 괄호 오류** — `if center else ""` 가 앞의 네
  f-string 전체에 묶여, subject 를 못 찾은 턴에는 `coverage`/`subject_area`/`occlusion_pass`
  까지 프롬프트에서 통째로 사라졌다. 모델이 왜 거절당했는지 알 길이 없어지는 종류의 버그.
  `_number()` 헬퍼로 `None` → `n/a` 처리하고 `subject_center` 만 조건부로 남겼다.
- `emit.py` 의 `MODEL_NOTES["rerope"]` 가 `emit_model_cams.py` 로 가는 것처럼 읽혔다 — ReRoPE 는
  `--model` choices 6종에 없고 `recammaster/rerope_prepare.py` 가 따로 처리한다.
- **`rerope_prepare.py` 가 소스 npz 를 날 c2w 로 써서 ReRoPE 내부 pose 가 축 순환 치환됐다.**
  `--axis_precomp {inv_conv(기본), none(구 동작)}` 추가. 아래 `--target_frame world` 로 원점을
  고친 뒤에도 **카메라가 엉뚱한 축으로 움직였다**. `convert_c2w_convention` 은
  `X -> X @ P`, `P = [[0,0,1],[1,0,0],[0,-1,0]]` (det −1) 인데 이게 **소스 경로에도**
  걸린다(`v2v_handler.py:124`). 소스에 DA3 OpenCV c2w 를 그대로 넣으면 내부 pose 가 전부
  `P⁻¹(·)P` 로 켤레변환되고 요청 이동 `t` 가 `P⁻¹t` 로 실현된다. sC 누적 optical flow
  실측이 순수 이동 4종 **전부** 예측과 일치했다:

  | 카메라 | 요청 로컬 t | 예측 `P⁻¹t` | 실측 누적 flow |
  |---|---|---|---|
  | truck_left | −x | −z (zoom) | du ≈ 0.0 (zoom) |
  | pedestal_up | −y | −x (좌우) | du +83.7 |
  | dolly_fwd | +z | −y (상하) | dv +8.7 |
  | dolly_back | −z | +y | dv −38.5 |

  `inv_conv` 는 소스·타깃 **양쪽 on-disk 에 `conv` 의 역**(`col1 부호반전` → `cols[2,0,1,3]`)을
  미리 걸어 ReRoPE 안에서 `conv` 를 항등으로 만든다. 새 `[축검증]` 단계가 내부 타깃 상대 w2c 가
  JSON 로컬 궤적과 같은지 확인한다 (sC max err **5.4e-07**). 대가로 on-disk det 가 배포
  npz(+1)와 반대인 −1 이 되는데 이론적으로 배포 npz 를 설명하지 못한다 — **판정 근거는 flow
  실측**이다. `ue5_asis` 와는 배타(`ap.error`).
  주의: **자기-타깃 검증(타깃 = 소스 궤적)은 이 버그를 못 잡는다** — conjugation 은 소스=타깃일
  때 상쇄된다. 아래 항목의 "검증" 이 통과했는데도 축이 틀려 있던 이유가 이것이다.
- **`rerope_prepare.py` 가 타깃 궤적을 UE5 JSON 그대로 넘겨서 ReRoPE 가 카메라를 못 따랐다.**
  `--target_frame {world(기본), ue5_asis(구 동작)}` 추가. 1차 생성물은 frame0 부터 소스와 다른
  장면이었고, 통제 실험으로 게이지(`--divisor 1`)·항등 타깃·프레임 길이(데모 nf81 vs nf49) 를
  전부 무죄 처리한 뒤 타깃 규약 두 군데를 찾았다:
  1. **det 부호.** 배포 npz 는 소스든 타깃이든 `convert_c2w_convention` **이후** rotation
     det = **−1** 이다. on-disk 는 proper c2w 고 `conv` 가 축을 뒤집는 게 정상 경로인데,
     `opencv_to_ue5` 는 flip 을 미리 넣어 놨으므로 `conv` 가 그걸 **되돌려** det +1 이 되고
     소스(−1)와 타깃(+1)이 다른 규약으로 들어갔다.
  2. **좌표 원점 (영향이 더 크다).** canonical JSON 은 frame0 = 항등인 **카메라-로컬 상대**
     궤적인데 ReRoPE 는 타깃을 **소스** frame0 기준으로 상대화한다(`v2v_handler.py:178`).
     로컬 궤적을 그대로 주면 소스 frame0 의 world rotation 이 상수 offset 으로 남는다 —
     sC 실측 **117.11°, 전 프레임 동일**.
  `world` 는 `Tw = C_src0 @ conv(json.T)` 로 심고 `Tw.transpose(0,2,1)` 만 해서 저장한다.
  검증: 타깃 = **소스 궤적 자신**(비자명 궤적)을 넣으면 출력이 소스를 재현한다.
  진단 표가 `tgtrot0 117.11 → 0.00` / `tdet +1 → −1` 로 바뀌고, 순수 이동 카메라 4종은
  `tgtrotmax = 0.00`, dl3dv 궤도 2종만 제 회전(61.69° / 77.56°)을 갖는다.
- **DA3 `Prediction.extrinsics` 를 c2w 로 읽고 있었다 — 실제로는 w2c 다.** `da3.py` 가
  `output.extrinsics = affine_inverse(c2w)` 를 넣고 `export/ply.py:156` 이 `# w2c` 라고 적어 뒀다.
  카메라 이동 방향이 정반대로 들어가고 있었고, 회전이 작으면 umeyama 스케일만은 얼추 살아남아
  (`|Δ(-R·C)| ≈ |ΔC|`) 지금까지 안 들켰다. `halfsplit_fullrecon_transfer.py --da3_extr` /
  `halfsplit_sweep.py --da3_extr` 추가 (기본 `w2c` = 수정본, `c2w` = 기존 동작 재현),
  `halfsplit_recon_da3.py` 규약 주석 정정. npz 포맷은 안 바꿨다 (읽는 쪽에서 `inv`).
  영향: full-recon 전이 `PSNR_cov` dynA 19.16 → **27.24**, DL3DV static 12.28 → **16.20**,
  잔여 배율 `f*` 0.5~0.6 → **1.00**. (`FIX.log` FIX-8)
- **SierpinskiCam 의 depth warp 출발 pose 가 DA3 world 원점이라 우리가 넣은 궤적이 통째로
  어긋나 있었다.** `Warper.forward_warp` 는 `T = pose_t @ inv(pose_s)` 인데 vendored 코드가
  `pose_s` 로 raw DA3 `extrinsics` 를, `pose_t` 로 frame0=I 상대화된 `traj` 를 넘긴다 →
  `i=0` 에서 `T = inv(extrinsics[0])`. DA3 는 frame0 을 원점으로 두지 않으므로 static 클립에서
  첫 프레임부터 74.7° / |t| 5.71 이 얹혔다 (dynA 는 2.60° / 0.1078 = 궤적 rmax 의 19.7%).
  `--sierp_anchor0` 로 켜는 재앵커 래퍼를 추가했다 (기본 꺼짐 = 기존 동작). TrajectoryCrafter
  (`pose_s` 고정)와 AlayaWorld (`num_context_frames:1`, 우리 `cam_c2w` 를 DA3 입력으로 사용)는
  영향 없음. (`FIX.log` FIX-9)
- **`halfsplit_recon_da3.py --gt_check` 가 FIX-8 이후에도 extrinsics 를 c2w 로 읽고 있었다.**
  DA3 는 w2c 라 `[:3,3]` 이 `-R·C` 인데 그대로 GT c2w 와 대조해서 회전 오차가 늘 175~180° 로
  찍혔다 — 규약 검증용 진단인데 규약을 못 가른다. 실측(신규 DL3DV `212b6928`): as-is 면
  resid 0.0927 / rot_med 174.7°, `inv` 면 **resid 0.0056 / rot_med 1.1°**. `--gt_check_extr
  {w2c(기본),c2w}` 추가 (`c2w` = 옛 동작 재현). 진단 출력만 바뀌고 저장되는 npz 는 그대로다.
- **`measure_realized.py` 가 alayaworld 의 결과 대신 depth warp 을 재고 있었다.** `SKIP_DIRS` 에
  `_warp` 가 없었고 `find_output` 이 `max(getsize)` 로 고르는데 warp(3.2 MB)이 본 결과(268 KB)보다
  커서 항상 warp 이 뽑혔다. 프레임 수로 구분된다 (warp 75 vs 출력 96). `_warp` 를 `SKIP_DIRS` 에
  추가하고 magladder 의 alayaworld 6점을 전부 재측정 — 결론(포화 아님)은 안 바뀌었고 적합만
  p 1.148→1.109, A 0.1371→0.1381 로 이동. (`FIX.log` FIX-10)
- **`emit_model_cams.py --n_frames` 를 alayaworld 에 잘못 넣어 궤적 뒷부분이 버려지고 있었다.**
  이 인자는 **움직이는 pose 개수**이고 manifest 의 `n_frames` 는 거기에 `--aw_prefix` 를 더한
  값이다. magmatch 에서 `--n_frames 233 --aw_prefix 129` 를 줘 pose 를 362 개 깔았는데 영상은
  233 프레임이라 뒤 129 개가 안 쓰였다 (canonical 의 앞 44% 만 실행). truck_left 가 등속 직선이라
  실현 이동량은 8.21·s vs 8.178·s 로 거의 같아 magmatch 의 적합·최종 scale 은 유효하지만,
  곡선 궤적이면 모양이 잘린다. `results/20260816_warp3` 부터 `--n_frames = 생성 프레임 수`
  (rounds=2 → 72) 로 바로잡았다.
- **`warp_ladder.sh` 의 tag 생성이 `emit_model_cams.py` 와 어긋나 파일을 못 찾았다.**
  emit 쪽은 `f'{s:g}'`, ladder 쪽은 `sed 's/\./p/'` 라 `4.890` 이 각각 `4p89` / `4p890` 이
  됐다. ladder 를 `awk '%g'` 로 바꿔 규칙을 맞췄다 (끝자리 0 이 없는 scale 은 영향 없음).
- **`make_warp_compare.py` 가 패널 총폭이 홀수면 빈 mp4 만 남기고 조용히 죽었다.**
  libx264/yuv420p 가 홀수 폭을 못 쓰는데 에러 메시지가 안 나온다. `read_frames` 에서
  패널 폭을 짝수로 내림한다.
- **`halfsplit_warp.project` 가 투영 좌표를 `long()`(=floor)로 정수화해 항등 warp 에서도
  hole 이 17.9% 뚫렸다.** `(u-cx)/fx*z -> fx*(x/z)+cx` 왕복에서 4.0 이 3.99999.. 가 되면
  floor 가 3 으로 내려보내 픽셀 4 가 빈다. `torch.round` 로 바꾸고 범위 검사도 반올림 좌표로.
  항등 쌍 hole 0.1794 -> **0.0000** (PSNR inf). 이전에 잰 half-split hole/PSNR 수치는
  전부 hole 이 부풀어 있었다 — 전 구간 재측정함. (`FIX.log` FIX-4)
- **`--pair mid` 의 첫 쌍이 `(48,49)` 라 첫 프레임부터 시차가 있었다.** 만나는 지점
  `m=nc-1` 이 ctx/tgt **양쪽의 첫 프레임**이어야 실제 추론(소스 첫 프레임 = 타겟 rel[0]=I)과
  같다. `dst` 를 `m` 부터 시작하도록 고쳐 첫 쌍이 `(48,48)`, 프레임 간격이 0,2,4,... 대칭이
  됐다. `aligned` 은 그대로. (`FIX.log` FIX-5)

### Changed
- **영상 저장은 전부 imageio + libx264 (h264)로.** `cv2.VideoWriter` 의 `mp4v` 는
  MPEG-4 Part 2 라 **VS Code 내장 뷰어에서 안 열린다** (파일은 멀쩡하고 재생만 안 되는 종류).
  `halfsplit_common.write_mp4(path, frames, fps, rgb=True)` 헬퍼를 추가하고
  `halfsplit_sweep.py` / `halfsplit_warp.py` / `make_warp_compare.py` / `make_sbs.py`
  네 곳의 `cv2.VideoWriter(mp4v)` 를 여기로 돌렸다. (`make_compare.py`,
  `alaya_warp_only.py` 는 이미 libx264 였다.)

### Added
- `halfsplit_pair_concat.py` 가 **모델 depth prior 를 여러 개** 받는다 (모델 하나만 줬을 때의
  기존 동작은 그대로). `--model_depth 이름=경로 ...` 로 나열하고 변형에서 `mid:per@sierp`
  처럼 `@이름` 으로 고른다 (`@` 생략 시 첫 모델). **f 는 모델마다 따로 잡는다** — prior
  스케일이 모델마다 다르니 하나의 f 를 공유하면 비교 자체가 성립하지 않는다. pose 를 안
  내놓는 모델(DepthCrafter)은 `--ref_depth` 로 depth 중앙값 비 예측으로 떨어지고,
  `--f trajc=<숫자>` 로 모델별 수동 지정도 된다.
- `halfsplit_prepare.py --pos_is_t` (기본 꺼짐 = 기존 동작) — dynrep `cameras.json` 의
  `position` 을 카메라 중심이 아니라 **w2c 의 t** 로 해석해 중심을 `C = -R^T·position` 으로
  만든다. **dynamic_replica 가 이쪽이다** (`FIX.log` FIX-7). 틀리면 가로 이동 방향만
  뒤집히고 크기는 얼추 맞아서 PSNR 로는 안 잡힌다.
- `tools/recammaster/halfsplit_pair_concat.py` (신규) — 여러 렌더 조건을 **같은 GT 프레임
  위에서** 한 영상에 나란히 붙인다. `--variants pair:depth_src[:depth_idx]` 를 나열하면 각
  변형의 `dst` 교집합만 골라 정렬한 뒤 `[ctx | warp1 | warp2 | ... | GT]` 로 concat 한다.
  pairing 이 다르면 `dst` 가 다르므로(`mid` 48..96 vs `aligned` 49..97) 영상 두 개를 그냥 옆에
  붙이면 **오른쪽 GT 패널이 서로 다른 프레임**이라 비교가 성립하지 않는 걸 막는다.
  `--ctx_panel vid` 는 ctx 절반을 정재생 원본으로 트는 패널(변형이 앵커 한 장만 써도 입력
  영상이 뭐였는지 보이도록). `--f` 기본값은 `pred` (ctx-only umeyama).
- `halfsplit_sweep.py --depth_idx <int>` (기본 `None` = 기존 동작) — `--depth_src first|anchor`
  가 쓸 앵커 프레임의 **원본 인덱스**를 직접 지정한다. `--pair mid` 는 ctx 를 역재생하므로
  `src[0]` 이 프레임 **48**(양 절반이 만나는 지점)이다. 클립의 진짜 첫 프레임 depth 를 쓰려면
  `--depth_idx 0` 으로 명시해야 한다 — 헷갈리면 "첫 depth" 가 조용히 다른 프레임이 된다.
- `halfsplit_sweep.py --depth_src per|first|anchor` (기본 `per` = 기존 동작, 신규) —
  warp 에 넣을 소스 depth 를 고른다. `first` 는 **depth 만 앵커(`src[0]`) 것으로 고정**하고
  이미지·카메라는 프레임 자기 것을 쓴다 (DA3 의 프레임별 depth 흔들림이 원인인지 가르는 용도).
  `anchor` 는 이미지·depth 를 둘 다 앵커로 고정 — 앵커 한 장으로 전 타겟을 만드는 단일 시점
  NVS 배치다. **`anchor` 는 `--pair mid` 에서 baseline 이 절반**(쌍 간격 2k -> k)이라
  hole 이 줄어든다 — `per` 와 절대 PSNR 을 직접 비교하지 말 것.
  영상 파일명에 `_d<모드>` 가 붙고, 좌측 패널은 실제로 warp 에 넣은 소스 프레임으로 바뀐다.
- `tools/recammaster/halfsplit_diag.py` (신규) — "warp 이 타겟 카메라를 덜 쫓아간다" 가
  **depth 편향인지 정렬 실패인지 아니면 애초에 시차가 없는 건지**를 가른다. `halfsplit_sweep.py`
  는 전역 배율 `f` 하나만 훑어서 이 셋을 구분하지 못한다. 네 가지를 같이 잰다:
  (A) 타겟 프레임별 `|ΔC|`/회전각/예상 시차 `dx/W`, (B) **무-warp 기준선**(ctx 프레임을 그대로
  둔 PSNR — warp 이 이걸 못 이기면 그 소스엔 시차가 없다), (C) **프레임별 `f*`**(전역 scale
  오차면 k 에 무관하게 상수, 흐르면 drift), (D) **depth 3분위 대역별 `f*`**(순수 scale 오차면
  세 대역이 같은 값, 갈라지면 depth 에 affine 편향이 있어 전역 `f` 로 못 고친다).
  결과는 `<tag>_diag.json`.
- `tools/recammaster/reencode_h264.py` — 이미 쌓인 mp4v 결과물을 h264 로 일괄 재인코딩.
  코덱 판별을 **디코딩 없이** 한다 (mp4 `stsd` fourcc 를 앞뒤 1MB 에서 스니핑) — 전체
  디코딩으로 재면 1600개에 수십 분. 기본 dry-run, `--apply` 로 원본 덮어쓰기.
  홀수 해상도 대비 `pad=ceil(iw/2)*2` 포함.
  **적용 결과: `results/` mp4 1638개 중 mpeg4 78개 -> 전부 h264, 실패 0.**
- **half-split 스케일 캘리브레이션 도구 6종** (`tools/recammaster/halfsplit_*.py`) — 긴 영상을
  반으로 갈라 앞 절반(context)을 모델 depth 로 unproject 하고 뒤 절반의 **GT 카메라**로 재투영해
  뒤 절반의 **실제 프레임**과 PSNR 로 비교한다. 눈으로 맞추던 `dx/W=0.11` 앵커를 정답 픽셀이
  있는 측정으로 대체하는 게 목적.
  - `halfsplit_common.py` — `list_frames`/`pick_span`/`umeyama`(s,R,t,resid,extent)/`apply_sim3`/
    `depth_scale_ls`(median_ratio·ls·ls_affine)/`load_dl3dv_gt`.
  - `halfsplit_recon_da3.py` — DA3 recon -> npz(`extrinsics`/`intrinsics`/`depth`/`conf`/`images`).
    `da3` env + `HF_HOME` 필요.
  - `halfsplit_prepare.py` — GT 카메라 기준계 npz 생성 (`--kind dl3dv|dynrep`).
    `--probe_conv`/`--probe_gap`/`--probe_scales` 로 카메라 규약을 문서가 아니라
    **photometric 으로 가른다**.
  - `halfsplit_warp.py` — z-buffer forward splat (vendored `Warper` 미사용, CV/GL 플래그).
  - `halfsplit_sweep.py` — 이동량 배율 `f` 스윕. ctx 구간 umeyama 예측값과 photometric 최적값을
    나란히 찍는다.
    - `--pair aligned|mid` (기본 `aligned` = 기존 동작, 신규) — ctx/tgt 프레임 짝짓기.
      `aligned` 은 `ctx i -> tgt nc+i` 라 **모든 쌍이 항상 nc 프레임 떨어져 있고** ctx 첫 카메라와
      tgt 첫 카메라 사이에 상수 offset 이 남는다. 모델은 소스 첫 프레임을 항등으로 보고 상수
      offset 은 무시하므로 실제 추론 배치와 모양이 다르다. `mid` 는 **ctx 를 역재생**해
      `(48,49) (47,50) ... (0,97)` 로 짝지어 **가운데에서 만나게** 한다 (`scale_cams` 앵커도
      `src[0]` 로 이동). 결과: 2차 DL3DV PSNR_cov 14.330 -> 15.663 / hole 0.846 -> 0.771,
      static 소스 예측/f* 오차 5% -> **3.3%**. dynamic 은 변화 없음(짝짓기 문제가 아니었다).
    - `--common_support` (기본 off, 신규) — `f` 가 커질수록 hole 이 커져 **평가 화소 집합이
      `f` 마다 바뀌는** 편향을 없앤다. 전 `f` 에서 공통으로 채워지는 화소만으로 `psnr_fix` 를
      다시 재고 그걸로 `f*` 를 고른다. 안 주면 기존 `psnr_cov` 기준 그대로.
      **dynamic 소스에서 결론이 뒤집힌다**: dynamic_replica `0cf56b` 의 `f*` 가
      1.2058 -> 0.7723 (-36%), static DL3DV 는 1.0194 -> 1.0158 (-0.35%).
    - `--accum 1|k|all` (기본 `1` = 기존 동작, 신규) — 타겟 1장을 만들 때 쓰는 ctx depth 장수.
      벤치마크 6종이 여기서 갈린다: TrajectoryCrafter(`demo.py:66-79`)와
      SierpinskiCam(`create_sierpinskicam_conditioning.py:250-261`, TrajC 의 `Warper` 재사용)은
      **frame i -> frame i 1:1** 이고, AlayaWorld 는 `num_context_frames`(=10,
      `configs/infer.yaml:52`)장을 `Sparse3DCache` 에서 커버리지로 골라 z-buffer 로 합치며
      (`spatial_cache.py:576,785`), GEN3C 는 `frame_buffer_max` 장을 bilinear splatting 으로
      누적한다(`cache_3d.py:151,246`). `k` 는 `mid` 짝짓기에서 "지금까지 지나온 ctx 중 최근
      k 장", `all` 은 지나온 전부(누적 상한).
    - `--video_f best|pred|umeyama_cam|depth_ratio|<숫자>` (기본 `best` = 기존 동작, 신규) —
      `--save_video` 가 쓸 배율을 고른다. 지금까지 영상은 **photometric argmax(`f*`)** 로
      뽑혀 있었는데, `f*` 는 GT 뒤 절반을 봐야 구할 수 있어서 **실제 추론에는 없는 값**이다.
      파이프라인이 실제로 쓰는 건 ctx 구간만으로 낸 예측(`umeyama_cam`, 없으면
      `depth_ratio`)이므로 품질을 눈으로 볼 땐 `pred` 로 뽑아야 한다.
      `best` 가 아니면 파일명이 `<tag>_@<video_f>.mp4` 로 갈린다.
    - JSON 에 `at_pred` 추가 — 각 예측 `f` 에서 실제로 잰 `psnr_cov`/`psnr_all`/`hole`.
      f* 와의 **비율**만으로는 그 오차가 화질을 얼마나 깎는지 안 보인다.
      (실측 예: 2차 DL3DV alaya, 비율 0.995 = `PSNR_cov` -0.012 dB.)
- `halfsplit_warp.py`: `lift_to_world()` + `forward_splat_multi()` — 소스 여러 장의 월드
  point cloud 를 **하나로 합쳐 z-buffer 하나로 경쟁**시킨다. 소스별로 warp 해 순서대로
  덮어쓰면 뒤 소스의 먼 면이 앞 소스의 가까운 면을 지운다. AlayaWorld 의
  `cand_depth.view(S,N).min(dim=0)` 과 같은 규칙. `lift_to_world` 를 분리한 건 누적 warp 에서
  소스마다 unproject 를 타겟 수만큼 반복하지 않기 위해서.
- `halfsplit_recon_da3.py` / `halfsplit_depth_depthcrafter.py`: `--reverse` (기본 off, 신규) —
  프레임을 **역순으로 모델에 넣는다**. `--pair mid` 는 ctx 를 역재생하는데 prior 는 정방향으로
  뽑고 있었다. DA3 는 cross-view 라 reference view 선택이 입력 순서에 걸리고, DepthCrafter 는
  video diffusion + 영상 전체 min-max 정규화라 시간 방향이 그대로 먹힌다.
  저장은 **원래 프레임 순서로 되돌려서** 한다 (소비자는 배열 인덱스를 원본 프레임 번호로 읽는다).
  - `halfsplit_depth_depthcrafter.py` — TrajectoryCrafter 의 DepthCrafter prior 를 같은 npz 규격으로.
    `demo.py` 전처리(576x1024 하드코딩, 49 프레임, window110/overlap25/steps5/gs1.0)를 그대로 재현.
- `tools/recammaster/run_grid.py`: **다중 소스 영상 지원**. `SOURCES` 레지스트리 8개
  (`cleaning_stove`, ReCamMaster 예시 `rcm3`/`rcm5`, Taylor 긴 영상 클립
  `taylor_c01/c02/c05/c19/c23`) + `--src` 플래그. 결과 경로가
  `<out_root>/<model>/<tag>` -> `<out_root>/<src>/<model>/<tag>` 로 바뀌었다.
- `tools/recammaster/run_grid.py`: `aw_assets()` — AlayaWorld 가 요구하는 960x544
  (`rollout_utils.py:279`) 리사이즈본 + 첫 프레임 PNG 를 소스마다 자동 생성.
  종횡비가 다른 소스(848x480 / 1280x720 / 1994x1080)를 늘리지 않도록
  `scale=...:force_original_aspect_ratio=increase` + `crop`.
- `tools/recammaster/run_grid.py`: `--keep_going` (한 카메라 실패해도 계속),
  `--skip_done` (결과 mp4 있으면 건너뜀). 기본 동작(첫 실패에 중단)은 그대로.
- `tools/recammaster/queue.sh` — 소스 하나를 모델별로 순차 실행하는 GPU 큐 스크립트.
- `tools/recammaster/estimate_cond_cam.py` — CameraAnything cond 궤적을 DA3 로 추정.
  `--validate` 로 저자 트랙과 대조한 결과 회전만 맞고 스케일·focal 은 안 맞아서
  **새 소스에는 적용하지 않기로 했다**. 수치/근거는
  `results/20260813_camgrid_stage1/_condcam/README.md`.
- `tools/recammaster/emit_model_cams.py`: `--aw_prefix` — AlayaWorld 궤적 앞에 붙일 항등
  pose 개수. autoregressive prefix 129 프레임(`rollout_utils.py:119-127`)이 궤적 앞부분을
  history 로 먹어버리는 걸 막는다. canonical 세트를 `--n_frames 104 --aw_prefix 129` 로
  재생성했고 native 1x 총 크기가 6.352 -> **8.178** 로 바뀌었다
  (`results/20260813_canonical_cams/config.md` 갱신).
- `tools/recammaster/copy_camviz.py` — 각 결과 폴더(`<src>/<model>/<tag>/`)에 그 모델이
  **실제로 먹은** 카메라 궤적 그림 `camera_traj.png` 를 넣는다. canonical 이 아니라
  `run.json['entry']['path']` 의 emit 된 파일을 `verify_emitted.py` 의 모델별 로더로 다시
  읽으므로 native 1x 크기 / anchor 차이(InfCam mid, AlayaWorld 항등 prefix 129,
  TrajectoryCrafter depth 배수)가 그림에 그대로 보인다. 렌더는
  `<out_root>/_camviz/<model>/<tag>.png` 에 캐시하고 소스별 폴더로 복사.
  `make_camviz.draw_traj` 재사용 (그쪽 코드 수정 0줄).
- `tools/recammaster/make_compare.py` — 모델 간 가로 concat 비교 영상 2개를
  `<out_root>/_compare/<src>/<tag>/` 에 만든다 (`A_input-recam-infcam-cameraA.mp4`,
  `B_input-trajC-sierp-alaya.mp4`) + 같은 폴더에 `cameras.png` (모델별 궤적 그림 세로 스택).
  모델별 출력 길이가 81/49/96 으로 다르지만 카메라 궤적은 전부 canonical 21 pose 를 자기
  길이로 resample 한 것이라, **인덱스 비례 리샘플**로 81 프레임에 다시 깔면 같은 시각에
  같은 pose 가 보인다 (짧은 쪽 패딩은 이 성질을 깬다). input 패널은 그룹별로 모델이 실제로
  본 소스 구간 (A=첫 81, B=첫 49) 을 쓴다. 아직 안 끝난 모델이 그룹에 있으면 그 그룹은 보류.
- `tools/recammaster/run_trajcrafter_matrix.py`: `--render_only` — `demo.save_video` 를 감싸
  `render.mp4` (point-cloud warp) 직후 센티널 예외로 빠져나온다. CogVideoX-5B 샘플링을
  통째로 건너뛰므로 카메라 규약 검증이 싸다. warp 결과는 생성 자유도가 0 인 결정론적
  출력이라 규약의 ground truth 로 쓸 수 있다. vendored 코드 수정 0줄.
- `tools/recammaster/run_trajcrafter_matrix.py`: `--traj_matrix_conv {c2w,legacy}` —
  2026-08-14 이전의 잘못된 pose 주입 규약을 재현하는 옵션 (기본은 고쳐진 `c2w`).
- `tools/recammaster/measure_realized.py` — **생성 영상에서 realize 된 카메라 운동을 잰다**
  (dense Farneback flow 중앙값 누적 -> `dx`/`dy` 는 폭 대비 비율, `zoom` 은 평행이동을 뺀
  radial 성분 최소제곱). emit 된 파일을 다시 읽는 `verify_emitted.py` 는 "우리가 쓴 것 = 우리가
  읽은 것" 까지만 증명한다 — 모델 내부가 그 배열을 어떤 규약으로 소비하는지는 결과 픽셀로만
  알 수 있다. `--extra model=<상대경로>` 로 보관소 폴더(`_hud`, `_legacyconv`)를 강제로 볼 수 있다.
  GPU 불필요. cv2 있는 env 로 돌릴 것 (`latentcam` / `recammaster` / `infcam` / `trajcrafter`).
- `tools/recammaster/scene_scale.py` — 모델마다 단위가 다른 translation 을 무차원 축
  `tau = |t| / z_med` 로 환산한다. `dx/W ~= (fx/W) * tau` 로 픽셀 시차와 연결되고,
  geometry-locked 모델은 해석적으로, prior-locked 모델은 실측 캘리브레이션으로 구한다는
  구분과 그 근거를 docstring 에 적어 두었다.
- `tools/recammaster/alaya_warp_only.py` — **AlayaWorld 의 spatial-memory depth warp 만**
  돌려 mp4 로 낸다 (diffusion 0회). AlayaWorld 는 prior-locked 이 아니라 매 chunk 마다
  DA3 depth 로 bank 프레임을 unproject/reproject 하는 geometry-locked 부류인데
  (`flash_alaya/utils/spatial.py:217`), 그 warp 결과가 디스크에 안 남아서
  (`spatial.py:236` 에서 바로 VAE 로 간다) 비교가 불가능했다. vendored 코드 수정 0줄 —
  warp 함수와 DA3 래퍼를 그대로 import 해 `spatial.py:213-234` 의 호출을 재현하고,
  `inference/da3_patch.py` 의 monkeypatch 도 실제 추론 경로와 동일하게 건다.
- `tools/recammaster/warp_ladder.sh` — depth-warp 3종(TrajectoryCrafter/SierpinskiCam/
  AlayaWorld)의 warp-only 렌더를 `tau` 를 맞춘 사다리로 뽑는 드라이버.
- `tools/recammaster/make_warp_compare.py` — 그 렌더들을 가로 concat 으로 붙인다.
  해상도(1024x576 / 512x320 / 960x544)와 프레임 수(49/49/104)가 다르므로 높이만 맞추고
  **폭 비율은 유지**하며(시차를 눈으로 재야 한다), 길이는 인덱스 비례 리샘플로 맞춘다.
- `tools/recammaster/run_trajcrafter_matrix.py`: `[radius]` 진단 출력. `demo.py:515` 의
  `radius = min(center_depth * radius_scale, 5)` 가 npy 단위 그 자체인데(1x = 0.5*radius)
  소스마다 달라지므로 기록이 없으면 모델 간 translation scale 을 맞출 수 없다.
- `video_generation/FIX.log`, `video_generation/CHANGELOG.md` 신설.

### Changed
- AlayaWorld 를 `--no-joystick --seed 0` 으로 돌린다 (`AW_JOYSTICK` / `AW_SEED` 상수,
  `--aw_joystick` / `--aw_seed -1` 로 저자 기본 동작 복구). 조이스틱 HUD 는 생성물이 아니라
  VAE 디코드 후 cv2 로 덧그리는 오버레이라 (`inference/run.py:287-289`) 6개 모델 픽셀
  비교에서 AlayaWorld 에만 없는 물체가 얹힌다. seed 는 벤치마크 재현성용.
  기존 55개 결과물은 `<entry>/_hud/` 로 옮겨 보관하고 재생성한다.
- `_outputs()` 가 `_hud/` 를 결과로 세지 않는다 (`--skip_done` 이 재생성을 건너뛴다).

### Fixed
- `measure_realized.py` 가 `*_source.mp4` 를 결과로 오인하던 문제. CameraAnything 은 생성
  **전에** 입력 복사본을 그 이름으로 쓰므로, 중간에 죽은 run 에서는 그것만 남아 소스 영상을
  결과로 재고 "카메라가 안 움직였다" 는 오진이 났다.
- `results/20260813_canonical_cams/<model>/manifest.json` 6개 재생성. `emit_model_cams.main()`
  은 그 실행에서 만든 항목만으로 manifest 를 덮어쓰므로, 카메라 하나만 다시 emit 하면
  나머지 항목이 조용히 사라진다 (카메라 **파일**은 남아서 눈에 안 띈다).
  `--scales 0.5,1,2,4` 전 카메라로 복구했고 기존 83개 `run.json` 의 경로가 전부 살아 있음을 확인.
- **TrajectoryCrafter pose 주입이 w2c 자리에 c2w 를 넣고 있었다** (`run_trajcrafter_matrix.py`).
  `demo.py` 의 `c2w_init`/`poses` 는 이름과 달리 w2c(extrinsic) 다 — 이 배열이 그대로
  `Warper.forward_warp(..., transformation1, transformation2, ...)` 로 들어가고
  (`models/utils.py:235-236` docstring), `utils.py:316` 이 `T2 @ inv(T1)` 을 camera-1
  좌표계 점에 곱한다. `poses = P0 @ rel` -> `poses = inv(rel) @ P0` 로 고쳤다.
  옛 규약은 realize 되는 상대 변환이 `t -> (t_x, -t_y, t_z)`,
  `R -> diag(-1,1,-1) R^T diag(-1,1,-1)` 이라 **translation Y 와 yaw 가 반전**돼 있었다
  (X/Z 는 우연히 맞음). 옛 동작은 `--traj_matrix_conv legacy` 로 보존.
  `--selftest` 도 pose 배열 대신 **realize 되는 상대 변환**을 비교하도록 재작성
  (translation 6종 + orbit 2종, `max|diff| < 1e-7`). FIX.log 참조.
  기존 56개 결과물은 `<entry>/_legacyconv/out/` 로 옮겨 보관하고 재생성했다.
- `_outputs()` 가 `_legacyconv/` 를 결과로 세지 않는다 (`--skip_done` 이 재생성을 건너뛴다).
- `copy_camviz.py`: SierpinskiCam 궤적 그림이 **회전만 보이던** 문제 (`reanchor()`).
  emitter 가 프리셋 규약대로 월드 오프셋 `[5000,1500,100] cm = [50,15,1] m` 을 얹는데
  `draw_traj` 의 `cube_limits` 가 원점을 포함하므로 반경이 ~26 m 가 되어 1.8 m 짜리 이동이
  점으로 뭉개졌다. anchor pose(InfCam 은 mid, 나머지는 frame0) 를 원점으로 재고정하고
  그린다 — 상수 좌곱이라 프레임 간 운동(= `verify_emitted.py` 가 비교하는 양)은 안 변한다.
  제목에 재고정 여부와 `|t|` 를 표시.
- `run_alayaworld` 이 실행 전에 leftover `<prefix>_video.*` 를 지운다. v2v 시절 심볼릭 링크가
  남아 있으면 AlayaWorld 가 조용히 i2v -> v2v 로 되돌아갔다. FIX.log 참조.
- SierpinskiCam stage B 의 `prompts.json` 을 list 형식으로 (`KeyError: 0`). FIX.log 참조.
- SierpinskiCam stage A 의 `--trajectorycrafter-path` 를 `TrajectoryCrafter/models` 로.
- TrajectoryCrafter 의 CPU/CUDA device 불일치 (상류 diffusers offload 버그) 우회.
- CameraAnything 카메라를 실제 소스(`cleaning_stove`)의 cond 월드에 얹도록 재생성.
- emit 한 카메라 파일 경로를 절대경로로 (dangling symlink -> `FileNotFoundError`).

### Changed
- `run_grid.py:run_cameraanything` 이 소스의 cond 카메라(`examples/camera.json`)가 없으면
  즉시 종료한다. `inference.py:356` 이 소스 자신의 궤적 81 프레임을 읽기 때문에
  다른 영상의 궤적을 조용히 쓰는 사고가 가능했다.
