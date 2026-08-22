# Changelog (video_generation)

[Keep a Changelog](https://keepachangelog.com/) 규약. `camera_generation/latentcam/CHANGELOG.md` 와
별개다 — 이쪽은 vendored 비디오 생성 모델 6종을 돌리는 `tools/` 스크립트만 다룬다.
(`video_generation/tools` 는 `.gitignore:223` 로 git 추적 대상이 아니다. 이 파일은 로컬 기록용.)

## [Unreleased]

### Fixed
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
