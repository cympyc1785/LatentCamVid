# LBM-Lite 구현 중 내가 임의로 고른 것들

사용자 지시("선택해야하는 것들 있으면 일단 권장 사항으로 선택하고 선택지, 선택 이유 같은거 기록 다
남겨줘")에 따른 기록. **전부 되돌릴 수 있게 knob 으로 남겼다** — 각 항목의 "바꾸는 법" 참고.

계획서(`~/.claude/plans/refactored-questing-wreath.md`)와 **어긋나는 결정에는 ⚠ 를 붙였다.**

---

## 3단계 — 후보 풀 / 게이트 / board

### D1 ⚠ 후보 풀을 τ 예산에서 역산한다 (`pool_mode="budget"`)

- **선택지** (a) 계획서 고정 격자 az{0,±45,±90,±135,180} × el{12,28,45} × r{0.7,1.0,1.4}·d_ref
  (b) 소스 카메라 구면좌표 중심 + τ 예산 안쪽 격자
- **채택** (b). (a) 를 camel 에서 돌리면 **72개 중 70개가 G4_tau 에서 죽는다**. 산수다:
  `budget_u = max_tau·z_med/S = 0.30·3.553/4.671 = 0.228 u` 인데 `d_ref = 0.64 u` 라 반경만
  1.4배로 늘려도(0.256 u) 예산 초과. "정면 45도"는 도달 불가능한 자리다. board 에 못 가는 후보를
  70개 올리면 look-before-move("고를 수 있는 것들 중에서 고르게 한다")가 무너진다.
- **바꾸는 법** `build_candidate_board.py --pool_mode absolute` — 계획서 격자 그대로 남아 있다.

### D2 ⚠ `obs_az_span` 하드 필터 기본 off (`--az_margin_deg 180`)

- 계획서는 "관측 범위 밖 방위는 즉시 제거"였는데 camel `dyn_1` 의 `obs_az_span` 이 **2.6°** 다
  (시차 0.0046 = 사실상 한 점에서 봤다). 그대로 걸면 9방위 중 1개만 남는다.
- 못 본 쪽은 점군이 비어 있으므로 **G2 coverage 가 알아서 떨어뜨린다** — 추측이 아니라 측정이라
  그쪽을 믿는다. `--az_margin_deg <deg>` 로 되살릴 수 있다.

### D3 contract 의 각도는 **소스 기준 상대값**(`d_az`/`d_elev`)

- OBB yaw 는 **180° 대칭**이라 절대 방위각이 "정면"을 정하지 못한다. VLM 에게 `az -146` 은 아무
  의미가 없다. 절대값은 `gates.csv` 에 그대로 남기고, contract/타일 캡션에만 상대값을 쓴다.

### D4 `board.json` 신설

- `gates.csv` 는 사람이 읽는 진단표라 pose 를 안 싣는다. decode 가 후보 풀을 다시 렌더하지 않도록
  board 에 오른 후보의 **기하 전량**(G 좌표 `p_g`/`look_at_g` + `c2w_world`)을 JSON 으로 뺐다.

### D5 subject 선택은 **화면 점유 하한이 먼저**

- "가장 많이 움직인 노드"만 보면 avocado-slice 에서 화면 0.24% 짜리 아보카도 조각이 subject 가 되고
  사람(12.7%)이 밀린다. 그러면 G3 area 게이트가 45개 후보를 **전부** 떨어뜨린다 (τ 예산 안에서
  그 조각을 화면 3% 로 키울 만큼 다가갈 수 없다 — 실측 최대 0.43%). 움직임은 subject 를 *고르는*
  기준이 아니라 같은 급 후보들 사이의 *순위* 기준.

---

## 4단계 — preset / decode / emit

### D6 preset 마다 **조준 모드(`aim`)** 를 둘로 나눴다

- **선택지** (a) 모든 preset 이 매 프레임 subject 를 다시 조준 (b) preset 별로 `look_at` / `traj`
- **채택** (b). (a) 로 하면 `pan_left` 가 pan 이 아니게 된다 — 회전을 넣어도 look-at 이 도로
  subject 로 끌어와 항등이 된다. `pan/truck/pedestal/static_hold_locked` 는 `aim="traj"`(궤적 자신의
  회전을 유지), 나머지는 `aim="look_at"`. `aim="traj"` 인 preset 에서는 `tracking` 이 무시되므로
  `info["tracking_ignored"]` 로 표시하고 canonical meta 에도 싣는다.

### D7 preset 크기는 **τ 이분법으로 닫는다** (사람이 정하는 숫자가 아니다)

- `DEFAULT_SHAPE` 는 *모양*(dolly_frac 0.35, lateral_frac 0.35, sweep 45°, pan 20°)만 정하고,
  실제 크기는 `fit_tau` 가 `target_tau` 를 만족하는 최대 배율을 8회 이분법으로 찾는다.
  씬마다 z_med/S 가 다르므로 고정 미터값은 의미가 없다.
- 실측: camel `traj_scale 0.42`, avocado-slice `traj_scale 1.41` — **같은 preset, 3.3배 차이**.

### D8 ⚠ **τ 는 시작 pose 와 궤적이 나눠 쓰는 하나의 예산이다** (이번 단계 최대 발견)

- 계획서에는 게이트 `max_tau 0.30`(후보용)과 `target_tau 0.20`(궤적용)이 **따로** 적혀 있다.
  그런데 `tau(f) = |p_plan(f) − p_src(f)| / z_med` 는 시작 offset 과 움직임을 **같이** 센다.
- 그래서 board 1위를 그냥 집으면 카메라가 선다. 실측: camel A1 `tau_start 0.2595`,
  avocado-slice A1 `0.2815` → 둘 다 `target_tau 0.20` 초과 → `fit_tau` 가 scale **0** 을 골라
  `path_len_u 0.0000`.
- **선택지** (a) 게이트 `max_tau` 를 낮춘다 (b) `target_tau` 를 올린다 (c) 결정 층에서 예산을 쪼갠다
- **채택** (c). `max_tau` 는 "후보 자체가 갈 수 있는 자리인가"를 보는 값이라 그대로 두고,
  `build_decision_fallback.py --start_tau_frac 0.5` 로 **시작 pose 에 절반, 궤적에 절반**을 준다.
  (a) 는 board 를 더 좁히고, (b) 는 hole 예산을 늘리는 것이라 둘 다 다른 것을 망친다.
- 결과: camel `tau_max 0.1931 / path_len_u 0.1453`, avocado-slice `0.1983 / 0.1404`.

### D9 ⚠ 그 결과 **fallback 이 뽑는 시작 pose 는 소스 pose 자신**이다 (알고 남긴 상태)

- board 후보의 τ 분포가 이산적이다: camel `[0.0, 0.12, 0.12, 0.15, ...]`,
  avocado-slice `[0.0, 0.12, 0.12, 0.15, ...]`. 격자의 **최소 non-zero 스텝이 이미 0.12** 라
  headroom 0.10 안에 드는 건 중심(=소스 pose, τ=0) 하나뿐이다.
- 원인: `budget_spans` 의 share(az 0.8 / el 0.5 / r 0.5) × 격자 5×3×3 이면 한 칸이 예산의
  절반 언저리다.
- **선택지** (a) 격자를 조밀하게 (`--num_azimuth 9 --num_radius 5`) (b) `--start_tau_frac` 을 올린다
  (c) 그대로 둔다
- **채택** (c) — 지금 fallback 은 "소스 자리에서 출발해 조용히 orbit" 이라 **보수적 기본값으로는
  오히려 맞다**. 그리고 시작 pose 에 예산을 얼마나 쓸지는 원래 **VLM 이 할 창작적 결정**이다
  (5단계). 렌더 수가 문제되면 (a) 가 싸다 (45 → 135 렌더, board_size 27 은 그대로).
- ⚠ **5단계에서 VLM 을 붙일 때 프롬프트에 "τ 는 시작과 움직임이 나눠 쓴다"를 명시할 것.**
  안 그러면 VLM 이 대담한 시작 pose 를 골라 놓고 궤적이 죽는 걸 못 본다.

### D10 `tracking` 기본값 `drift` (gain 0.6)

- `world`(0.0) 는 τ 최소지만 동적 subject 가 프레임 밖으로 나간다, `lock`(1.0) 은 track jitter 가
  곧장 카메라 회전이 된다(r=0.5u 에서 1% jitter ≈ 프레임당 1° yaw). 계획서대로 중간값.

### D11 `look_at_bias` 의 크기 단위 = OBB 높이 × S, 방향 = `gravity.up_world`

- `T_wg` 는 스케일 S 를 품고 있어(`T_gw[:3,:3] = R_gw/S`) **방향 벡터에 곱하면 안 된다**.
  위치·타겟만 변환하고 회전은 world 에서 다시 세운다 — 그래야 `det(R)=1` 이 보장된다.

### D12 49→21 은 **index pick**, frame0 anchor, unit scale

- `idx = rint(linspace(0,48,21))` (`canonical_cams.build_dl3dv:128` 과 동일).
  `emit_model_cams.resample` 은 emit 쪽 업샘플러라 여기서 쓰면 두 번 리샘플된다.
- `--free_start` 로 상수 offset 을 살릴 수 있지만 **ReCamMaster 는 그 offset 을 픽셀 수준에서
  무시한다**(실측) — depth-warp 계열에만 의미가 있다. 기본 `--no_free_start`.

### D13 zoom 은 canonical 을 통과하지 못한다

- `emit_model_cams.py` 에 intrinsics 채널이 없다. `focal_scale` 은 canonical meta 에만 싣고
  `zoom_dropped: true` 를 세운다 (검증 렌더러만 소비). micro-adjust 의 `zoom_in/out` 은
  `--allow_zoom` 없으면 메뉴에서 뺀다 → 16 ops 중 14 ops 가 기본.

### D14 `orbit_span_frac` 이 두 군데 다르다 (의도)

- `build_decision_fallback.py` 0.7 (보수적 기본), `build_poses.py` 0.8 (계획서 clamp 값).
  decision 에 `trajectory.shape.sweep_deg` 가 실리면 그 값이 이기므로 실제로는 0.7 이 쓰인다.
  0.8 은 shape 이 없는 decision(=미래의 VLM 응답)에 대한 fallback clamp.

### D15 `rerope` 는 `emit_model_cams.py` 의 모델 목록에 **없다**

- `NATIVE_1X` 는 recammaster / sierpinskicam / infcam / cameraanything / trajectorycrafter /
  alayaworld 6종. ReRoPE 는 별도 경로다. `emit.py` 의 `MODEL_NOTES` 에는 경고만 남겨 뒀다
  (target `|t|` 를 1.0 으로 재정규화 → `g` 가 안 살아남는다).

### D16 Stage 0 명사는 VLM 이 뽑는다 — `metadata.csv:dynamic` 은 정답 유출이다

- `seg_instances/<video>/meta.json["keywords"]` 는 **Vista4D 저자가 손으로 적은** `metadata.csv`
  의 `dynamic` 열이다 (`sam3_seg_instances.py:30-42 read_keywords()` 가 쉼표로 자를 뿐이다).
  저자들 스스로 `README.md:200` 에서 Grounded SAM 2 로 마스크를 뽑고 카메라는 UI 로 손수
  설계했다고 적어놨고, `:249` 가 그 열을 "Dynamic keywords used to obtain the segmentation map"
  이라고 정의한다. **배포 시엔 없는 데이터**이므로 detector 비교의 기준으로 쓰면 안 된다.
- `fit/graph/extract_nouns_vlm.py` 가 그 자리를 채운다. 프레임 → Qwen3-VL → `{dynamic, static,
  subject}`. 계획서의 `extract_static_nouns.py`(metadata.csv `prompt` 열 파싱, `--llm` 선택)를
  **이걸로 대체**한다 — `prompt` 열도 같은 이유로 배포 시 없다.
- 프롬프트에 "이미 적은 것의 상위어를 쓰지 말라"를 넣었다. GDINO caption 은 `" . "` 이어붙이기고
  반환 phrase 가 caption 의 부분문자열이라, `woman` 과 `person` 을 같이 주면 하나의 박스에
  `woman person` 이 붙어 나온다 (authored kw arm 실측: `phrases=['woman person', ...]`).
  그래서 authored 대비 `person`/`human`/`fruit` 가 "missed" 로 찍히지만 이건 **설계된 누락**이다.
- 어휘 출처 3분기는 `gsam2_timing.py` 에 `--caption_json`(VLM) / `--ram` / 기본(meta keyword)
  으로 남겨 뒀다 — 셋 다 그대로 재현 가능해야 비교가 성립한다.

### D17 emit 은 `poses.npz` 의 decision 지문을 검사한다

`emit.py` 는 `poses.npz` 를 읽지 `decision.json` 을 다시 디코드하지 않는다. 루프가 decision 을
새로 쓰고 `build_poses` 를 안 돌리면 **낡은 궤적이 조용히 emit 된다** — 실측으로 avocado-slice 가
`static_hold` 결정을 갖고 `orbit_left_arc` poses(`path_len_u` 0.14039)를 내보냈다. 숫자가 전부
말이 되게 나와서 육안으로 안 잡힌다. 타임스탬프 비교는 답이 아니다(같은 초에 쓰이면 순서를 모름).

`build_poses.decision_fingerprint(decision)` 이 **pose 를 실제로 바꾸는 필드만** 정렬 JSON 으로
직렬화해 npz 에 심고, emit 이 재계산해 비교한다. 불일치면 `SystemExit` + 재실행 명령 출력.
`--allow_stale_poses` 로 우회 가능(기본 off). 지문에 안 들어가는 필드(`vlm.reasoning`,
`gates.*` 등)는 일부러 뺐다 — 그것 때문에 재빌드를 강요하면 사람이 가드를 꺼 버린다.

### D18 이동량 0 궤적은 정규화하지 않고 플래그를 세운다

`static_hold`/`static_hold_locked` 는 카메라가 제자리에서 look-at 만 돌리므로 `rmax = max|t|` 가
float 잡음이다 (avocado-slice 실측 1.43e-17). 예전 `unit_scale` 의 `max(rmax, 1e-12)` 는
그 잡음을 1.39e-5 로 **증폭**시켜 `rel[0] = I` assert 를 터뜨렸다 — 규약이 깨진 게 아니라 0/0 이다.

`rmax < 1e-6 u` 면 t 를 정확히 0 으로 두고 `rmax = 0`, `g = 0`, `translation_degenerate: true`.
플래그가 필요한 이유는 assert 회피가 아니다: 이동량 0 canonical 은 `emit_model_cams --scales` 도
ReRoPE 의 `|t| → 1.0` 정규화도 의미가 없는데 **에러 없이 통과한다**
(`--model sierpinskicam` 실측 `rmax 0.0000, rot 0.26`). 소비자가 안 걸러내면 "카메라가 안 움직이는
영상"이 정상 산출물처럼 나온다.

### D19 [닫힘 → D21] select 단계가 소스 pose 로 수렴한다 — 후보 풀 설계 문제

camel 실측: VLM 이 `A3 = (d_az 0.0, d_el 0.0, dist 1.00, cov 0.969, tau 0.000)` 을 confidence
0.95 로 골랐다. 근거 원문: *"It is also at the same elevation and azimuth as the source camera,
minimizing parallax."* micro 는 round 0 에 `done:true`. 즉 **선택 결과가 "안 움직인다"** 였다.

이건 모델 탓이 아니라 풀 탓이다. `pool_mode budget` 이 τ 예산을 역산해 후보를 만들어서
소스 pose 가 풀 안에 들어 있고(camel daz ∈ {0, ±8.4, ±16.8}), board 에 보여주는 모든 지표
(coverage / subject_area / tau)가 **정확히 그 지점에서 최적**이다. 새 시점을 만들라고 시켜놓고
"새 시점일수록 점수가 나쁜" 판을 준 셈이다.

선택지 — **아직 고르지 않았다** (계획서 §B1 변경이라 사용자 확인 필요):
1. 항등 후보를 풀에서 빼고 최소 시차 하한(`|d_az| ≥ 8°` 등)을 강제.
2. board 텍스트에 novelty 열을 추가해 소스와의 차이를 **보상**으로 제시.
3. 그대로 두고 select 를 "안전한 시작점 고르기"로 재정의, 시점 변화는 trajectory 에만 맡김.

### D20 [부분적으로 닫힘 → D21] τ 예산을 소스 카메라의 자기 운동이 먼저 먹는다

`τ(f) = |p_plan(f) − p_src(f)| / z_med(f)` 는 **움직이는 소스** 기준이라, 플랜이 가만히 있어도
소스가 움직이면 τ 가 쌓인다. avocado-slice(parallax 0.1286, z_med 2.627): `start_tau` 0.1549 만으로
예산 0.30 의 절반을 쓰고, preset 16종이 **전부** `tau_scale 0.0000 / move 0.000 / tau_saturated
True` 로 죽었다. VLM 은 `[MOTION SATURATED]` 를 정확히 읽고 `static_hold`/`lock` 을 골랐다:
*"All motion presets are saturated, meaning no camera movement is possible within the tau budget."*

결과가 뒤집혀 있다 — **camel 의 degenerate 한 select(소스와 같은 pose)가 오히려 궤적 예산을
남겼고**(`tau_scale` 0.4219~0.6875), avocado 의 더 과감한 select(dist 1.145x)가 궤적을 죽였다.
D19 를 고치면 이쪽이 더 나빠진다 — 두 문제는 같이 봐야 한다.

선택지 — **아직 고르지 않았다** (계획서 §B5 변경):
1. τ 예산을 start / trajectory 로 명시 분할 (예: start ≤ 0.10, traj ≤ 0.20).
2. τ 기준을 소스 궤적이 아니라 **소스 frame0** 으로 바꿔 소스 자기 운동을 빼기.
3. 소스 parallax 에 따라 예산을 스케일 (`tau_budget = 0.30 + k·parallax`).

관련 메모: `tau-is-one-shared-budget`.

---

## 5단계 — 시작 pose 를 결정 대상에서 뺀다

### D21 `start_mode=source_frame0` 이 기본 (사용자 지시)

사용자 지시 원문: **"일단 기록해두고 그냥 첫 카메라를 그대로 쓰는걸로 단순화하자"**.
D19 · D20 을 각각 고치는 대신 **시작 pose 를 결정 대상에서 통째로 뺐다**. 시작 pose 를 안 고르면
D19(선택이 소스 pose 로 수렴)는 존재하지 않는 문제가 되고, D20 은 절반이 사라진다.

`--start_mode source_frame0`(기본): 소스 카메라의 `cam_c2w_world[0]` 을 그대로 시작 pose 로 쓰고,
`lbm/loop.py` 는 **select · micro 단계를 건너뛴다**. 남는 VLM 결정은 궤적 preset 하나다.
`--start_mode board`: 예전 3단 경로(select → micro → traj)가 그대로 돌아간다.

#### 왜 이 단순화가 손해가 아닌가 (근거 3개)

1. **하류가 어차피 버린다.** emit 의 기본 규약 `--no_free_start` 는 `rel = inv(P[0]) @ P` 라
   시작 pose 와 소스 카메라 사이의 **상수 offset 을 정의상 버린다**. `--free_start` 로 살려도
   ReCamMaster 는 픽셀 수준에서 무시한다 (메모 `recammaster-ignores-constant-camera-offset`).
   즉 select 3턴의 산출물은 canonical 을 거의 통과하지 못했다.
2. **그러면서 τ 예산은 먹었다.** avocado-slice 는 시작 pose 만으로 `start_tau` 0.1549 를 써서
   preset 16종이 전부 `tau_scale 0.0000` 으로 죽어 있었다 (D20).
3. **실제로 안 움직였다.** camel 의 select 는 `A3 = (d_az 0.0, d_el 0.0, dist 1.00)`, 즉
   소스 pose 자신이었다 (D19). 이번에 `--start_mode board` 로 재확인 — 다시 A3 을 골랐다.

#### 진단: R1 / R2 / R3

- **R1 τ 는 움직이는 소스 기준이다.** `τ(f) = |p_plan(f) − p_src(f)| / z_med`, 같은 프레임
  인덱스끼리 짝짓는다. 플랜이 정지해 있어도 소스가 움직이면 τ 가 쌓인다. 실측
  `src_self_tau == parallax_ratio` 가 **정확히 성립**한다: camel 0.0042(예산 0.20 의 2%),
  avocado-slice 0.1286(**64%**). 즉 avocado 는 완벽한 시작 pose 를 줘도 궤적 몫이 0.0714 뿐이다.
- **R2 τ 는 "사전 필터"로 문서화돼 있는데 실제로는 크기를 결정하는 유일한 구속이다.**
  `gates.py:11` docstring: *"위 셋보다 1000배 싼 사전 필터. 렌더 수를 깎는 용도지 판정용이 아니다"*.
  그런데 `presets.py:fit_tau` 가 **같은 τ, 같은 임계**로 이분법을 돌려 궤적 크기를 정한다.
  직접 측정치(G2 렌더 coverage)는 크기 결정에 한 번도 안 쓰인다.
- **R3 생성할 카메라에 대한 intent text 가 파이프라인 어디에도 없다.** 프롬프트 3종을 다 읽어도
  의도라 부를 만한 건 두 문장뿐이다 — `system_select.md:35` *"Prefer the subject sitting on a
  rule-of-thirds intersection over dead center."*, `system_traj.md:46` *"Prefer motion that reveals
  something. A shot that ends where it started earns less."* 나머지는 전부 제약이다.

#### 안 채택한 처방 F1 / F2 / F3 (나중 후보)

- **F1** τ 를 `min_t |p_plan(f) − p_src(t)| / z_med` 로 (점군은 전 프레임의 합집합이므로 같은
  프레임 인덱스에 묶일 이유가 없다). R1 을 정면으로 없앤다. 이번엔 안 건드렸다 — 게이트·board·
  fit_tau 가 전부 같은 τ 를 쓰므로 board 를 다시 만들어야 하고, D21 만으로 두 씬 다 풀렸다.
- **F2** 궤적 크기를 τ 가 아니라 **렌더 coverage** 로 닫기. preset board 가 이미 preset 당
  start/mid/end 를 렌더하므로 비용은 사실상 0. R2 의 정공법.
- **F3** UAV 문헌식 actor-centric shot spec `{ρ, ψ_rel, θ_rel}` 을 intent 입력으로. R3 용.
  (UAV 조사 결론: dynamic video 입력 사례는 있지만 — Huang TPAMI 2022 실기체, CineTransfer
  IROS 2023 시뮬 — **움직이는 기준 카메라를 3D 로 올리는 사례는 없다.** 전부 actor-centric 이고
  deviation budget 자체가 없다. 편차는 soft cost 이자 사후 지표일 뿐이다.)

#### 실측 (before → after, 두 씬 모두 `orbit_left_arc`)

| | camel | avocado-slice |
|---|---|---|
| VLM 턴 | 3 → **1** | 3 → **1** |
| `tau_start` | 0.0042 → 0.0042 | 0.2734 → **0.1286** |
| `traj_scale` | 0.4219 → 0.375 | 0.0 (saturated) → **1.547** |
| `path_len_u` | 0.1453 → 0.1476 | **0.0000 → 0.1533** |
| `tau_max` | — → 0.1961 | — → 0.1977 |
| 포화된 preset 수 | 0 → 0 | **16 → 0** |
| 시작 pose 렌더 coverage | 0.9836 | 0.9931 |

camel 은 **before/after 가 사실상 같다** — board 가 고른 후보가 소스 pose 자신이었기 때문이다
(D19). 즉 camel 에서는 select 3턴이 값을 만들지 않았다는 게 숫자로 확인된 셈이고, 이번 변경의
실질 효과는 avocado 쪽이다. (camel before 값은 board 경로 preset_stats 의 `orbit_left_arc` 행이라
디코드 산출물과 소수점 이하가 조금 다르다 — 프리뷰는 CLI `--target_tau`, 디코드는 `decision.json`
의 값을 쓴다.)

avocado 는 `static_hold` 결정 자체가 포화 상태에서 내려진 것이라 궤적 단계를 다시 돌려야 했다.
`tau_start` 0.1286 은 **소스 자기 운동**이고 (R1), 시작 pose 몫은 0 이다.

#### 남은 차이 두 개 (억누르지 않고 기록)

- **roll.** 루프의 상태 표현 `(p_g, look_at_g)` 는 중력축 기준 roll=0 을 강제하는데 소스 카메라는
  기울어져 있다 — 실측 camel 3.15°, avocado 2.04°. 위치와 시선 방향은 1e-6° 이내로 일치한다.
  게이트가 보는 타일만 그만큼 덜 기울고, 디코드는 `cam_c2w_world[0]` 을 그대로 쓴다.
  (`aim="look_at"` preset 은 매 프레임 subject 로 재조준하므로 roll 이 어차피 0 이 된다.)
- **정규직교성.** `cam_c2w_world` 는 DA3 w2c 를 뒤집어 저장한 값이라 열이 정확히 단위벡터가
  아니다 (`|R[:,2]|` camel 1+1.71e-5, avocado 1−2.19e-4). 그대로 쓰면 `det(R)=1` assert 가
  frame 0 에서 터진다 — 궤적이 아니라 입력의 문제라 `build_poses.orthonormalize` 로 SO(3) 에
  한 번 투영한다(회전각 변화 ~1e-3°). `board` 경로는 `look_at_c2w` 가 만든 pose 라 해당 없다.

#### 어긋남 방지

`start_mode` 는 **디코드 방법의 일부**다. `board` 로 고른 결정을 `source_frame0` 로 풀면 VLM 이
고른 pose 가 통째로 사라지고, 반대면 없는 후보를 시작점으로 삼는다 — 둘 다 에러 없이 "다른
궤적"만 낸다. 그래서 (a) `decision.json` 에 `start_mode` 를 싣고 (b)
`build_poses.resolve_start_mode` 가 **decision 을 CLI 보다 우선**하며 (c) `decision_fingerprint`
payload 에 넣어 D17 의 stale-poses 가드가 모드 전환도 잡게 했다.

#### 바꾸는 법

`python -m lbm.loop --video <v> --start_mode board` — 예전 select → micro → traj 경로.
decode/emit 은 `decision.json` 의 `start_mode` 를 따라가므로 추가 플래그가 필요 없다.

---

## 6단계 — verify.py + 프리뷰

### D22 jerk 분모는 소스 p95, 소스가 정지면 씬 스케일로 갈아탄다

계획서 표는 지표 7 을 "소스 p95 jerk 대비 3배/5배"로 적었다. 그런데 camel 소스는
`parallax_ratio 0.0042` 로 사실상 정지라, 분모가 0 에 가까워지면 배율이 의미 없이 발산한다.

실측(2026-08-20): 소스 p95 jerk 는 camel 0.0175792 u, avocado-slice 0.00701176 u 로 **둘 다
0 이 아니었다** — 손떨림이 남아 있다. 그래서 두 씬 모두 계획서 그대로 소스 p95 를 썼고
(`jerk_denominator.mode = "source_p95"`), 결과는 camel 1.978e-05 / avocado 6.339e-05 로 PASS.
플랜 궤적이 해석적으로 매끄러운 preset 이라 실제 jerk 가 3.5e-07 u 수준이다.

`src_p95 < 1e-4·S` 면 `1e-2·S` 로 갈아타고 그 사실을 `verify.json` 의 `jerk_denominator` 에
남긴다. **분모를 조용히 바꾸면 배율이 씬마다 다른 뜻이 된다** — 지금은 두 씬 다 안 탔지만,
정지 소스가 들어오면 탄다.

### D23 지표 8 의 resample 은 emit 의 `roundtrip_error` 와 다른 것을 잰다

`decode/emit.py:146 roundtrip_error` 는 `rel_full_unit[idx]` 를 도로 뽑아 `rel21_unit` 과
비교한다 — 같은 인덱스라 **정의상 항상 0** 이다 (실측 `roundtrip_max 0.000e+00`). 그건 index
pick 이 제대로 됐는지의 검사고, 리샘플 손실은 못 본다.

여기서는 계획서 표대로 `emit_model_cams.resample`(SE(3) 측지 보간, 하류 모델이 21≠n 을 요구할
때 실제로 타는 코드)로 21→49 를 되돌려 `rel_full_unit` 과 비교한다. 실측 camel 0.00694961 /
avocado 0.00687041 (임계 0.02). 즉 **21로 깎는 데서 0.7% 를 잃는다** — 통과지만 0 이 아니고,
21키를 요구하는 모델(ReCamMaster 계열)에 그만큼의 궤적이 도달하지 않는다는 뜻이다.

### D24 `make_camviz.py` 를 재사용하지 않는다 (권장 선택, 사용자 승인 대기)

repo 규칙은 "`scripts` 의 기존 후처리 코드에 같은 역할이 있으면 최소 수정으로 재사용"이다.
가장 가까운 후보는 `video_generation/tools/recammaster/make_camviz.py` 인데 세 가지가 안 맞는다:

1. 입력이 ReCamMaster `trajectories.json` 의 **rel 공간** 궤적 한 벌이다. 우리는 world/G 공간에
   궤적이 **둘**(소스 + 플랜) 있고 소스가 움직인다 — `draw_traj` 는 소스를 원점의 회색 피라미드
   **하나**로 고정해 그린다 (`make_camviz.py:139-142`).
2. subject track 이라는 개념이 없다.
3. 정적 PNG 만 낸다. 계획서가 요구한 건 `plan_cam.mp4` 다.

세 가지를 다 넣으면 `draw_traj` 의 몸통이 남지 않아 "최소 수정"이 아니다. 그래서
`verify.py:plan_cam_video` 로 새로 짰다. 되돌리려면 `make_camviz.py` 에 `--source_traj` /
`--subject_track` / `--video` 를 추가하는 쪽이고, 그때는 CinemaTraj 쪽을 지우면 된다.

**plan_cam 눈금 — 2줄 (`--cam_detail`, 기본 on).** 카메라-subject 거리가 0.6 u 인데 카메라
운동은 실측 camel `ptp [0.0178, 0.146, 0.007]` / avocado `[0.0201, 0.1515, 0.0018]` u 라, 씬
전체를 한 눈금에 담으면 궤적이 뭉개진다 (첫 렌더에서 확인). 위 줄 = 씬 전체(카메라가 subject
어디쯤인가), 아래 줄 = 카메라만 확대(어떤 모양으로 움직이나). `--no_cam_detail` 로 예전 1줄.
**viz 는 사용자 승인 전 확정하지 않는다** — 1줄/2줄 선택은 되돌릴 수 있게 플래그로 남겼다.

### D25 verify 는 `da3` 가 아니라 `vista4d` 에서 돈다

계획서 env 표는 `decode / verify` 를 `da3` 로 적었지만, 지표 1·2·3 이 49프레임 Vista4D 렌더를
요구한다(`CloudRenderer`). loop 와 같은 `vista4d` env 에서 돌린다. `decode` 만 `da3` 로 돌아간다.

### 6단계 실측 (2026-08-20, GPU 1, `--start_mode source_frame0`, preset `orbit_left_arc`)

| 지표 | 임계 (PASS/WARN) | camel | avocado-slice |
|---|---|---|---|
| hole_fraction | <0.35 / 0.35–0.55 | 0.0737215 | **0.327682** |
| subject_in_frame | ≥0.95 / 0.85–0.95 | 1 | 1 |
| subject_pixel_coverage | 0.02–0.45 | 0.0917329 | 0.100786 |
| behind_surface_frames | 0 | 0 | 0 |
| max_view_angle_delta | ≤40 / 40–50 | 12.9615 | 11.2966 |
| tau_max | ≤0.25 / 0.25–0.30 | 0.196051 | 0.197719 |
| jerk_ratio | ≤3 / ≤5 | 1.97805e-05 | 6.33917e-05 |
| roundtrip_resample | <0.02 | 0.00694961 | 0.00687041 |
| anchor_identity | <1e-9 | 2.22045e-16 | 1.9535e-17 |
| **verdict** | | PASS | PASS |

avocado-slice 의 `hole_fraction 0.328` 은 PASS 지만 WARN 경계(0.35) 바로 아래다 — `plan_sbs.mp4`
에서 좌측 절반이 마젠타로 열린다. 시차 0.1286 짜리 소스라 orbit 이 관측 없는 쪽을 곧장 판다.
같은 preset·같은 τ 예산인데 camel 0.074 vs avocado 0.328 로 4.4배 차이가 나는 건 **궤적이
아니라 소스 시차**가 정한다는 뜻이다.

지표 4(behind_surface)는 두 씬 다 0 이지만, 이건 게이트가 frame 0 만 보던 것을 **전 49프레임 ×
전 49 소스프레임**으로 확장해 처음 확인한 값이다.

### D26 [D21 부분 반박] `source_frame0` 는 **위치만** 보존한다 — frame 0 부터 이미 다른 그림이다

사용자 질문 원문: **"첫 프레임이 왜 달라?"** (`plan_sbs.mp4` frame 0).

원인은 `decode/build_poses.py:~190` 한 줄이다. `aim == "look_at"` 이면

```python
for f in range(num_frames):
    poses[f] = look_at_c2w(poses[f][:3, 3], look_at[f], up_world)
```

가 **f=0 을 포함한 전 프레임**을 subject 로 다시 조준한다. `c2w_start = orthonormalize(src_c2w[0])`
는 그 뒤 **위치만** 살아남는다. 채택 preset `orbit_left_arc` 가 `aim="look_at"` 이라 두 씬 다 해당.

τ 가 이걸 못 잡는다: `τ = |p_plan − p_src| / z_med` 는 **위치 전용**이라 `tau[0] = 0.0`,
`view_angle_deg[0] = 0.0` 으로 찍힌다 (실측, 두 씬 다). 지표 어디에도 안 나온다.

#### 실측 (2026-08-20, GPU 1, frame 0)

| | camel | avocado-slice |
|---|---|---|
| `\|p_plan[0] − p_src[0]\|` | 0.0 | 0.0 |
| 회전 각차 (전체) | 3.1837° | **11.6739°** |
| 회전 각차 (forward 축만) | 0.4765° → 사실상 roll | **11.4534°** → 진짜 재조준 |
| hole @ 소스 pose | 0.0100 | 0.0127 |
| hole @ plan f0 | 0.0395 | **0.3648** |
| subject 중심 @ 소스 pose | (0.506, 0.492) | (0.317, 0.335) |
| subject 중심 @ plan f0 | (0.501, 0.464) | (0.471, 0.556) |

camel 은 소스가 이미 subject 를 중앙에 두고 있어서 재조준할 게 없다 — 남는 3.18° 는 거의 전부
roll(중력축 기준 roll=0 강제)이고, 그것만으로도 hole 이 0.010 → 0.040 으로 4배가 된다.
avocado 는 소스가 subject 를 좌상단 (0.317, 0.335) 에 두고 있어서 look_at 이 11.45° 를 실제로
돌려 중앙에 붙인다. 그 대가가 hole 0.0127 → 0.3648, **frame 0 이 클립 전체에서 가장 나쁜 프레임**
(f0 0.3648 / f1 0.3599 / f24 0.3227 / f48 0.3171).

#### hole 분해 — 구멍은 이동이 아니라 **조준**이 판다

pose 를 (위치, 회전) 으로 쪼개 교차 렌더한 값. `hole = 1 − valid.mean()`:

| | camel f0 / f24 / f48 | avocado f0 / f24 / f48 |
|---|---|---|
| 소스 pose | 0.0100 / 0.0086 / 0.0118 | 0.0127 / 0.0063 / 0.0112 |
| 소스 위치 + **plan 회전** | 0.0395 / 0.1750 / **0.4183** | **0.3648** / 0.2481 / 0.2592 |
| **plan 위치** + 소스 회전 | 0.0100 / 0.0711 / 0.1730 | 0.0127 / 0.1006 / 0.1890 |
| plan pose | 0.0395 / 0.0591 / 0.1265 | 0.3648 / 0.3227 / 0.3171 |

두 씬 다 **회전만 바꾼 쪽이 위치만 바꾼 쪽보다 구멍이 크다**. camel f48 은 회전만 0.4183 인데
합치면 0.1265 로 내려간다 — orbit 의 이동이 새 조준 방향에 관측을 도로 대준다는 뜻이고, 이게
"조준과 이동은 같이 정해져야 한다"의 직접 증거다. 위 6단계 실측 문단의 *"궤적이 아니라 소스
시차가 정한다"* 는 이 분해로 정정된다 — 정확히는 **조준 변화량**이 정하고, 소스 시차는 그
조준을 감당할 관측이 있느냐를 정한다.

#### D21 의 어느 문장이 틀렸나

`lbm/loop.py:224-227` (D21 근거) 의 *"남는 차이는 roll 하나 (실측 camel 3.15°, avocado 2.04°)"*
는 **루프 게이트가 보는 상태**(`source_frame0_state`, 소스 시선 방향으로 조준) 에 대한 값이다.
디코드되는 frame 0 에 대한 값이 아니다. avocado 에서 이 둘은 11.45° 벌어진 **다른 뷰**다.

따라서 구조적 구멍이 하나 열려 있다: **게이트가 승인한 그림(hole 0.0127)과 실제로 렌더되는
frame 0(hole 0.3648)이 다르다.** look-before-move 의 전제("VLM 은 렌더 가능한 것 중에서만 고른다")
가 frame 0 에서 깨진다. 게이트는 preset 선택 **이전**에 돌아서 `aim` 을 아직 모르기 때문에
한 줄 수정으로는 못 막는다 — preset 확정 후 디코드된 frame 0 을 **다시** 게이트에 태워야 한다.

#### 선택지 (**O2 채택 — 사용자 지시**, O3 은 여전히 미적용)

| | 무엇 | 얻는 것 | 잃는 것 |
|---|---|---|---|
| O1 | 현행 유지 (f0 부터 look_at) | 첫 프레임부터 구도가 맞는다 | avocado 는 카메라가 움직이기도 전에 hole 예산을 다 쓴다 |
| **O2** | **aim ramp — f0 은 소스 방향, K프레임에 걸쳐 look_at 으로 풀기** | frame 0 이 소스와 정확히 일치 (SBS·frame0 앵커 모델에 유리) | avocado 는 f12 이후 다시 0.33 으로 올라간다. 국소 처방 |
| **O3** | **디코드된 frame 0 을 게이트에 재투입** — preset 확정 후 `poses[0]` 로 G1~G3 를 다시 돌리고, 실패하면 그 preset 을 board 에서 뺀다 | 게이트-디코드 불일치를 **구조적으로** 닫는다. O1/O2 중 무엇을 쓰든 유효 | 루프 재실행 필요 (VLM 턴이 다시 돌아 결정이 바뀔 수 있다) |
| O4 | 조준 강도 α — look_at 을 소스 시선과 subject 중심 사이에서 보간 | hole↔구도를 연속 손잡이로 | 손잡이가 하나 더 늘고, α 를 정할 근거가 아직 없다 |

O3 은 정책을 안 고르고 측정 구멍만 막는 유일한 안이라 여전히 유효하지만, 루프 재실행(=VLM 재호출)
을 요구한다. 반면 O2 는 디코더만 고치면 되고 decision 을 안 건드린다.

### D27 aim ramp 를 넣고 **되돌렸다** — 기본값은 `--aim_anchor subject` (사용자 선택)

사용자 지시 원문: **"시작 카메라 완전 똑같이 해서 돌려줘봐"** → O2 구현 → 두 쪽을 다 렌더해서
비교 → **"그냥 직전꺼가 나은 것 같아 되돌려서 계속 진행해줘"** → 기본값을 `subject`(= D26 이전
동작)로 되돌렸다. 코드는 남긴다 — `--aim_anchor {source_frame0,auto}` 로 언제든 켤 수 있고,
아래 실측이 그 선택의 근거다.

`decode/build_poses.py` 에 `--aim_anchor {auto,source_frame0,subject}` + `--aim_ramp_frames 12`.
**기본 `subject`** (D26 이전 동작). `auto` 는 `start_mode` 를 따라가고, `source_frame0` 은
언제나 앵커를 켠다.

동작: look_at 조준을 다 세운 **뒤**, frame 0 의 조준 오차를 world 회전 하나(`rotation_log`)로
뽑아 smoothstep `w(f)=x²(3−2x), x=f/12` 로 되돌린다. f=0 에서 100% 되돌림(`poses[0] = c2w_start`),
f≥12 에서 0%(원래 조준 그대로). smoothstep 인 이유는 f=0 에서 기울기가 0 이라 첫 프레임이 안
튀기 때문이다 — 선형이면 f0→f1 에서 avocado 가 0.96°/frame 로 출발한다.

부수 변경 3개: ① `look_at` 배열도 ramp 구간은 **실제 시선축**으로 다시 찍는다 (안 하면 npz 와
plan_cam 화살표가 회전과 어긋난다) ② roll assert 는 ramp **바깥**에만 건다 (ramp 구간은 소스
기울기를 일부러 되살린 것) ③ 지문에 `aim_anchor`/`aim_ramp_frames` 를 넣었다 — 안 넣으면 이것만
바꿔 재빌드했을 때 emit/verify 가드가 못 잡는다. `emit.py`/`verify.py` 에도 같은 CLI 를 달았다.

#### 실측 비교 (2026-08-20, GPU 1, 49f 1280×720, preset·decision 동일)

| 지표 | camel `subject` | camel `source_frame0` | avocado `subject` | avocado `source_frame0` |
|---|---|---|---|---|
| hole_fraction | 0.0737215 | **0.0704789** | 0.327682 | **0.280871** |
| subject_in_frame | 1 | 1 | 1 | 1 |
| subject_pixel_coverage | 0.0917329 | 0.0917329 | 0.100786 | 0.101436 |
| behind_surface_frames | 0 | 0 | 0 | 0 |
| max_view_angle_delta | 12.9615 | 12.9615 | 11.2966 | 11.2966 |
| tau_max | 0.196051 | 0.196051 | 0.197719 | 0.197719 |
| jerk_ratio | 1.97805e-05 | 1.97805e-05 | 6.33917e-05 | 6.33917e-05 |
| roundtrip_resample | 0.00694961 | 0.00696026 | 0.00687041 | 0.00696435 |
| anchor_identity | 2.22045e-16 | 6.59063e-18 | 1.9535e-17 | 2.22045e-16 |
| verdict | PASS | PASS | PASS | PASS |

앵커 각도: camel 3.2093°, avocado-slice 11.5555°. `roll_max` 는 camel 3.18° / avocado 0.569°
— avocado 의 11.56° 는 대부분 roll 이 아니라 **재조준**이라는 D26 의 분해와 일치한다.

frame 0: `|poses[0] − cam_c2w_world[0]|max` 는 camel 3.216e-05, avocado 5.891e-04 로 **0 이 아니다**.
`c2w_start = orthonormalize(src_c2w[0])` 이기 때문이고, 그 잔차는 DA3 가 낸 c2w 가 정규직교가
아닌 양 그대로다(`|R[:,2]|` camel 1+1.71e-5 / avocado 1−2.19e-4, 회전각으로 ~1e-3°). 앵커
assert 는 `c2w_start` 와 비교하므로 1e-9 이내로 통과한다. 렌더 hole 로 확인: f0 이 camel 0.010,
avocado 0.013 으로 §D26 의 "소스 pose" 행(0.0100 / 0.0127)과 같다. **첫 프레임은 이제 소스다.**

프레임별 hole 이 ramp 를 따라 올라간다 (f0..f14):
- camel  0.010 0.015 0.023 0.028 0.035 0.040 0.045 0.047 0.049 0.048 0.049 0.048 0.049 0.049 0.049
- avocado 0.013 0.011 0.020 0.049 0.087 0.130 0.174 0.216 0.254 0.287 0.313 0.329 0.335 0.334 0.333

프레임간 회전 최대치는 camel 0.58°/frame, avocado 1.19°/frame (f5~f6) — 일반 팬 속도 범위다.
클립 평균 hole 이 avocado 에서 0.328 → 0.281 (−14.3%) 내려간 건 ramp 12프레임이 싼 구간이라
평균을 끌어내린 것이지 정상 구간이 나아진 게 아니다 (f12 이후 0.333 으로 복귀).

#### 되돌린 뒤 상태 (2026-08-20)

기본값을 `subject` 로 바꾸고 `build_poses → emit → verify` 를 두 씬 다시 돌려 원상복구했다.
`model_gauge_rmax` 가 camel 0.68801962 / avocado 0.71040781 로 D26 이전 값과 **정확히** 같고,
verify 도 `hole_fraction` 0.0737215 / 0.327682 로 복귀했다 (avocado `subject_pixel_coverage`
만 0.100786 → 0.100785, 반올림 자리).

`results/20260820_aim_lbm_lite/` 는 지웠다 — 위 표에 숫자가 다 있고, 기각된 프리뷰를 남겨두면
다음에 어느 쪽이 현행인지 헷갈린다. 현행 프리뷰는 `results/20260820_lbm_lite/` 하나뿐이다.

남는 사실 하나: **`start_mode=source_frame0` 은 "첫 카메라를 그대로"가 아니라 "첫 카메라
위치를 그대로"다.** 이름이 오해를 부르므로 D26 의 실측(회전 camel 3.18° / avocado 11.67°)을
근거로 여기 박아둔다.

---

### D28 정적 명사에는 **두 겹의 필터**를 건다 (동적 중복 · 광역 표면)

계획서 7단계. `fit/graph/extract_static_nouns.py` 는 명사 목록을 만들기만 하는 게 아니라 거른다.
거르지 않으면 두 가지가 깨지고, 둘 다 **조용히** 깨진다 (그림으로는 안 보이고 VLM 프롬프트
텍스트에서만 드러난다).

① **동적과 겹치는 명사** — VLM 은 `frame_mode` 에 따라 같은 명사를 dynamic 에도 static 에도
넣는다. avocado-slice 에서 실측: `single` 은 `wheelchair/bottle/bowl/plant` 를 static 으로,
`multi` 는 `wheelchair/bowl/plant` 를 dynamic 으로 분류했다. 그대로 두면 같은 물체가 `dyn_*` 와
`stat_*` 두 노드가 되고 `merge_duplicates` 의 3D IoU 0.5 를 못 넘으면 안 합쳐진다.
→ 동적 keyword (`seg_instances/meta.json` ∪ VLM `dynamic`) 와 겹치면 정적 쪽을 버린다.

② **광역 표면** (`ground/wall/floor/ceiling/sky/road/...`) — OBB 를 씌우면 씬 전체를 덮는 상자가
되고 `near` 엣지가 전 노드에 걸린다. 게다가 둘 다 이미 기하로 처리돼 있다
(`relations.ground_height`, `relations.find_wall_planes` — camel 벽면 1개 / avocado 3개 검출).
→ `SURFACE_NOUNS` 기본 제외, `--no_drop_surfaces` 로 복구 가능.

실측 결과 (`--source vlm --frame_mode multi`):

| video | 후보 | 채택 | 버린 것 |
|---|---|---|---|
| camel | 5 | 4 — `fence roof tree bush` | `ground`(표면) |
| avocado-slice | 8 | 6 — `table chair window television bottle "cutting board"` | `wall`(표면), `plant`(동적 중복) |

#### 명사 출처 — `--source auto` (vlm 우선, prompt 폴백)

계획서는 `extract_static_nouns.py` 를 "metadata.csv prompt → 규칙 기반, `--llm` 선택" 으로 적었다.
그 사이에 `extract_nouns_vlm.py` 가 생겨 static 명사를 이미 내고 있으므로 **VLM 을 기본으로
뒤집고** 규칙 기반은 폴백으로 남겼다 (계획서가 확정한 기본값도 "VLM 명사 + SAM3 text PCS"다).

폴백의 품질은 명시적으로 낮다. env 에 POS tagger 가 없어서(`nltk`/`spacy` 미설치, 설치는 이전에
거부됨) 품사 분석이 아니라 "관사·전치사 뒤 단어들의 마지막 하나" 라는 **위치 규칙**이다.
`-ing/-ed/-ion/-ness/-ly` 접미사와 `NON_NOUNS` 로 걸러도 이 정도다:

| video | `--source prompt` 결과 |
|---|---|
| camel | `enclosure tree sun` |
| avocado-slice | `table bowl butter bottle chalkboard` |

`sun` 은 물체가 아니고 `enclosure` 는 `fence` 를 놓친 것이다. 머리단어를 **한 개**만 내는 것도
의도적이다 — 두 단어로 붙이면 `fence wall` 이 되어 `SURFACE_NOUNS` 필터를 그냥 빠져나간다.

#### D29 정적 track 도 **동적 픽셀 비율로** 한 번 더 거른다

`sam3_static_instances.py` 는 저장 전에 각 track 의 마스크가 배포본 `dynamic_mask` 안에 든
픽셀 비율(`dynamic_frac`, 프레임 평균이 아니라 **픽셀 총합** 비율)을 재고 `--max_dynamic_frac`
(기본 0.5) 을 넘으면 뺀다. 이유는 ①의 잔여분이다 — 명사 목록에서 못 걸러도 마스크는 안 속는다.
움직이는 물체가 `stat_*` 노드가 되면 `relations.supported_by` 가 "움직이는 것에 얹혀 있다" 는
틀린 문장을 만들고, 그 문장이 VLM 프롬프트로 그대로 나간다 (§relations.py docstring: "정밀도보다
틀린 관계를 안 내는 것이 중요하다").

뺀 track 도 `seg_instances_static/<video>/static_report.json` 에 `kept:false` 로 남긴다 —
`keywords` 목록에서도 지우면 "이 명사를 시도는 했다"는 사실이 사라진다.

**실측: 이 두 씬에서는 한 번도 발동하지 않았다.** 12개 track 전부 `dynamic_frac` 0.0000 이고,
정적 마스크 합집합 대비 dynamic_mask 와의 겹침은 camel 140 px / 21,656,235 px 중 avocado 0 px
(= 7.9e-06 / 0.0) 다. SAM3 가 동적 물체를 정적 keyword 마스크에서 잘라내기 때문이다. 즉 지금
이 필터는 **검증된 게 아니라 미발동**이다 — 다른 씬에서 처음 걸릴 때 임계 0.5 를 다시 볼 것.

#### D30 정적 track 의 `moving` 은 임계가 아니라 **`kind` 로 정한다**

정적 노드를 넣자마자 camel 6개 노드가 전부 `moving=True` 로 나왔다. 울타리가 낙타보다 4.8배
"움직인" 것이다. 원인은 운동이 아니라 **OBB 중심의 떠돎**이다 — 큰 정적 물체는 프레임마다
보이는 부분(가림·화각)이 달라져 프레임별 `minAreaRect` 중심이 이동한다.

| node | label | `path_len_u` | `center_drift_u`(순변위) | max extent |
|---|---|---|---|---|
| dyn_0 | camel | 0.0674 | 0.0546 | 0.14 |
| dyn_1 | camel | 0.0576 | 0.0290 | 0.16 |
| stat_0 | fence | 0.3217 | 0.0379 | 0.75 |
| stat_1 | roof | 0.1495 | 0.0632 | 0.26 |
| stat_2 | fence | 0.0829 | 0.0616 | 0.30 |
| stat_3 | tree | 0.1545 | 0.0466 | 0.60 |

경로길이로도 순변위로도 두 무리가 겹친다 (지붕 0.0632 > 낙타 0.0546). **임계가 존재하지 않는다.**

avocado-slice 재실행도 같은 방향이다 — 여기서는 **떠돎이 실제 운동의 5배**다:

| node | label | kind | `path_len_u` | `center_drift_u` | max extent |
|---|---|---|---|---|---|
| dyn_0 | person | dyn | 0.0389 | 0.0088 | 0.207 |
| dyn_1 | avocado | dyn | 0.1110 | 0.0093 | 0.052 |
| dyn_2 | avocado | dyn | 0.0251 | 0.0032 | 0.038 |
| stat_0 | table | stat | 0.1699 | 0.0377 | 0.376 |
| stat_1 | window | stat | **0.5742** | **0.1137** | 3.393 |
| stat_2 | window | stat | 0.2854 | 0.1913 | 1.888 |
| stat_3 | table | stat | 0.1540 | 0.0341 | 0.639 |
| stat_4 | chair | stat | 0.0466 | 0.0435 | 0.056 |
| stat_5 | bottle | stat | 0.0505 | 0.0050 | 0.027 |
| stat_6 | television | stat | 0.0242 | 0.0136 | 0.113 |
| stat_7 | cutting board | stat | 0.0316 | 0.0047 | 0.126 |

창문(`stat_1`)이 실제로 움직이는 아보카도(`dyn_1`)보다 path 5.2배 · drift 12.2배다. 크기와
상관도 없다 — `stat_5 bottle`(ext 0.027)의 path 0.0505 가 `dyn_2 avocado`(ext 0.038)의 0.0251
보다 크다. 즉 "작으면 안 떠돈다"는 보정도 안 통한다.

그래서 `moving = (kind == "dyn") and path_len > 0.05u` 로 바꿨다. 정적 인스턴스는 정적 명사에서
나왔고 D29 가 dynamic_mask 로 한 번 더 걸렀으므로 그 출처를 믿는 것이다. 측정치는 지우지 않고
`path_len_u` 옆에 `center_drift_u` 를 새로 실어 남긴다 — 숨기면 떠돎 자체를 진단할 수 없다.

이게 프롬프트에 중요한 이유: 텍스트 블록의 `## SUBJECT ... moving? speed_u_per_frame` 과
`## NEIGHBORS` 가 이 값을 그대로 읽는다. "the fence is moving at 0.0067 u/frame" 은 VLM 에게
거짓을 참으로 주는 문장이다.

#### D31 `against_wall` 임계를 **물체 크기 비례**로

같은 재실행에서 `against_wall` 도 6개 전부 True 였다. 실측 최근접 OBB 꼭짓점–벽면 거리
(camel, 벽 평면 1개):

| node | fence(stat_0) | roof | fence(stat_2) | tree | camel(dyn_0) | camel(dyn_1) |
|---|---|---|---|---|---|---|
| dist_u | 0.0010 | 0.0175 | 0.0066 | 0.0063 | 0.0569 | 0.0436 |

절대 임계 `wall_u=0.08` 하나로는 낙타도 걸린다. 낙타는 울타리 **앞에 서 있는** 것이고, 전부
True 인 플래그는 프롬프트에 0 비트를 넣는다 (`relations.py` 자신의 원칙: "정밀도보다 틀린 관계를
안 내는 것이 중요"). 임계를 `min(wall_u, wall_contact_frac · max(extent[:2]))`, 기본
`wall_contact_frac=0.15` 로 바꿨다 — "닿았다" 는 물체 크기에 상대적인 말이기 때문이다.
낙타 임계가 0.021 / 0.024 로 떨어져 두 노드가 빠지고 정적 4개만 남는다.
`--wall_contact_frac 1e9` 로 예전 절대 임계 동작을 복구할 수 있다.

**camel 에서는 정적 4개가 여전히 전부 True 라 이 씬만 보면 판별력이 없어 보인다** — 낙타 우리라
정적 물체(울타리 2개·지붕·나무)가 실제로 다 울타리면에 붙어 있다. 판별력은 벽 평면이 3개인
avocado-slice 에서 확인된다: 11개 노드 중 **3개만** True 다.

| node | label | max extent | 임계 `min(0.08, 0.15·ext)` | against_wall |
|---|---|---|---|---|
| dyn_0 | person | 0.207 | 0.0310 | False |
| dyn_1 | avocado | 0.052 | 0.0078 | False |
| dyn_2 | avocado | 0.038 | 0.0057 | False |
| stat_0 | table | 0.376 | 0.0563 | False |
| stat_1 | window | 3.393 | 0.0800 (`wall_u` 상한) | **True** |
| stat_2 | window | 1.888 | 0.0800 (`wall_u` 상한) | False |
| stat_3 | table | 0.639 | 0.0800 (`wall_u` 상한) | **True** |
| stat_4 | chair | 0.056 | 0.0084 | False |
| stat_5 | bottle | 0.027 | 0.0040 | False |
| stat_6 | television | 0.113 | 0.0170 | **True** |
| stat_7 | cutting board | 0.126 | 0.0189 | False |

창문 두 개가 갈린 것(`stat_1` True / `stat_2` False)이 이 임계가 실제로 뭔가를 재고 있다는
증거다. 둘 다 `wall_u=0.08` 상한에 걸려 **같은 임계**를 쓰는데도 결과가 다르므로, 차이는 임계가
아니라 측정된 꼭짓점–벽면 거리에서 온다. 큰 물체(ext > 0.53)는 `0.15·ext > 0.08` 이라 항상
`wall_u` 상한이 먹는다 — `wall_contact_frac` 은 **작은 물체를 풀어주는 방향으로만** 작동한다.

#### D32 정적 노드가 실제로 바꾼 것은 `supported_by` 와 edge 수다 (하류 기하는 불변)

정적 노드를 넣기 전/후로 **동적 노드의 기하는 한 비트도 안 바뀌었다** — `/tmp/sg_before_*.json`
대비 `obb` · `track` · `d_ref` · `moving` 전부 동일하고 `merge_log` 도 camel `[]`,
avocado `dyn#0→dyn#1(0.9795)`, `dyn#2→dyn#1(0.9764)` 로 예전 그대로다. 정적 인스턴스가 동적
track 을 흡수해 subject 가 바뀌는 사고는 없었다. 그래서 이미 승인된 `decision.json` /
`poses.npz` / `canonical.json` 은 재실행 없이 유효하다.

바뀐 건 **VLM 프롬프트로 나가는 문장** 두 종류다:

| | camel | avocado-slice |
|---|---|---|
| edges 전 → 후 | 0 → 4 | 3 → 16 |
| `supported_by` | dyn 둘 다 `ground` (불변) | dyn 셋 다 `null` → **`stat_0` (table)** |

avocado 의 `supported_by: null → "stat_0"` 이 정적 노드의 본래 목적이다. 예전엔 사람도 아보카도도
"무엇에도 얹혀 있지 않다"였다 — 테이블 노드가 없었기 때문이지 공중에 떠 있어서가 아니다.
`lbm/overlay.py:contract_text` 가 이 필드를 그대로 문장으로 뽑으므로, 정적 노드 없이 돌리면
VLM 은 **테이블 위 장면을 테이블 없이** 본다.

이 변화는 기하가 아니라 텍스트에만 걸리므로 `decision_fingerprint` 가 못 잡는다. 정적 노드를
켜고 끈 두 판을 비교하려면 `scene_graph.json` 자체를 비교해야 한다.

#### D33 정적 노드의 OBB 품질은 동적보다 나쁘다 — 지금은 받아들이고 기록만 한다

재실행 요약표에서 정적 노드 몇 개가 검사에 걸렸다 (`in_bbox` 는 전부 Y, 즉 **중심은 맞고 크기가
과대**하다):

| video | node | label | `reproj_px` | `size_x` | `size_ok` | 프레임 |
|---|---|---|---|---|---|---|
| camel | stat_0 | fence | **151.0** | 1.19 | Y | 49 |
| camel | stat_3 | tree | 14.2 | **2.07** | NO | 24 |
| avocado | stat_0 | table | **111.8** | **2.11** | NO | 49 |
| avocado | stat_1 | window | 40.3 | **1.85** | NO | 49 |
| avocado | stat_4 | chair | **83.8** | 1.67 | Y | 48 |

동적 노드는 같은 판에서 1.7~18.3 px 다. 원인은 두 가지가 섞여 있다: ① **얇고 넓은 구조**
(울타리·테이블 상판·창틀)는 gravity-aligned `minAreaRect` 가 깊이축으로 부풀기 쉽고 — `stat_0
table` 의 extent 가 `0.38,0.35,0.02` 로 h=0.02 인 판때기다 — ② 깊이 누출. 두 씬 다 마지막에
`!! OBB 투영 크기가 마스크 대비 과대 — depth 누출을 의심할 것 (--depth_trim_quantile)` 이 뜬다.
`stat_4 chair` 는 extent `0.06,0.00,0.02` 로 **w=0.00** 인 퇴화 상자다.

**지금 고치지 않는 이유**: 정적 노드는 후보 풀 원점도 게이트 대상도 아니다 (그건 subject =
동적 노드다). 소비처는 `## NEIGHBORS` 의 `label` + `dist_u` 한 줄뿐이고, 중심이 맞으니
(`in_bbox` 전부 Y) 그 거리는 쓸 만하다. 크기가 하류로 새는 유일한 경로는 `near` 판정 반경인데
과대 추정은 edge 를 **더 만드는** 쪽이라 조용히 정보를 지우지는 않는다. 고칠 때는
`--depth_trim_quantile` 를 정적 노드에만 세게 걸거나, 얇은 축(extent < 0.03u)을 OBB fit 에서
빼는 쪽이다.

#### D34 VLM 은 구멍을 **보긴 한다.** 그런데 preset 선택은 그림도 숫자도 아닌 **prior** 다

사용자 질문("VLM 에 hole 있는 사진 줘도 인식 가능한거야?")을 실측으로 갈랐다. `turn_00.json`
의 답변이 "avoids the large magenta regions seen in truck_left/truck_right/pedestal_up" 이었는데
**같은 프롬프트 텍스트에 preset 별 `coverage_min`/`coverage_end` 가 이미 들어 있어서**
"magenta 를 봤다" 와 "0.68/0.60/0.69 를 읽고 magenta 라는 단어를 붙였다" 가 구분되지 않았다.

`eval/ablate_vlm_hole_perception.py` 로 4조건 × 3 draw × 2영상 = 24콜:

| 조건 | 그림 | coverage 숫자 | camel picks | avocado picks |
|---|---|---|---|---|
| A_full | O | 원본 | orbit_left_arc ×3 | orbit_left_arc ×3 |
| B_no_num | O | **제거** | orbit_left_arc ×3 | orbit_left_arc ×3 |
| C_no_img | **X** | 원본 | orbit_left_arc, **s_curve, s_curve** | orbit_left_arc ×3 |
| D_conflict | O | **1−x 로 반전** | orbit_left_arc ×3 | orbit_left_arc ×3 |

**결론 세 줄.**

① **구멍은 실제로 본다.** D_conflict 에서 숫자상 `orbit_left_arc` 는 coverage 0.10 (최악),
`truck_right` 는 0.40 (그보다 좋음)인데도 답변은 여전히 "truck_left or truck_right result in
significant magenta regions" 였다. 이 문장은 **주어진 숫자와 반대**이고 **실제 그림과 일치**한다
(원본 coverage truck_left 0.68 / truck_right 0.60). B_no_num 에서 숫자를 아예 없애도 magenta
지목이 유지된다. 즉 magenta 판정 채널은 픽셀에서 온다.

② **그런데 정확하진 않다.** B_no_num 에서 `rise_reveal` 을 "significant magenta in the last
frame" 으로 지목했는데 실제 coverage 는 **0.96 으로 두 번째로 높다.** 원본 A_full 에서도
`rise_reveal` 을 "risks losing coverage" 로 기각했다. 즉 큰 구멍(0.6대)은 맞히고 중간은 지어낸다.

③ **가장 중요한 것 — 최종 선택은 두 채널 어느 쪽도 아니다.** `orbit_left_arc` 가 24 draw 중
22회다. 이미지를 빼도(C, avocado), 숫자를 지워도(B), 숫자를 **뒤집어도**(D) 안 바뀐다.
D_conflict 답변은 심지어 `"maintains coverage (0.10) at the end"` 라고 **반전된 숫자를 그대로
인용하면서 그 숫자와 반대되는 결론**을 쓴다 — 모순을 알아채지 못한다. 움직인 조건은 camel 의
C_no_img 하나뿐(→ `s_curve` ×2)이라 **그림이 숫자보다는 답을 붙잡아 두지만**, 지배적인 것은
"orbit 이 영화적이다" 라는 사전확률이다.

**설계에 대한 함의**: look-before-move 의 전제("VLM 이 렌더를 보고 고른다")는 preset 턴에서는
약하게만 성립한다. 게이트가 preset 을 미리 잘라내는 구조(§B5, camel 3개 EXCLUDED)가 남아 있는
게 다행이고, **품질을 지키는 건 VLM 이 아니라 게이트**라는 뜻이다. 후보 선택 턴(`select`)에도
같은 실험을 해야 한다 — 거기서도 prior 라면 board 렌더는 토큰만 먹는다. 아직 안 했다.

**안 한 것**: temperature 를 올려 prior 를 흔드는 것, 후보 select 턴 ablation, 다른 모델
(GPT-4.1) 대조. `--conflict` 는 숫자만 뒤집고 그림은 그대로다 — 그림을 뒤집는(구멍 없는 렌더에
가짜 magenta 를 칠하는) 조건이 더 강한 검사인데 아직 없다.

#### D35 board 는 27칸이 아니라 **9칸**이 맞다 (근거: 각도 폭이 애초에 좁다)

27칸 board 는 4352×818 px, **가로세로 5.32:1 띠**다. 사람이 못 본다. 그런데 진짜 문제는 칸 수가
아니라 **27칸이 덮는 각도 폭이 좁다는 것**이다:

| | d_az 폭 | d_az 종 | d_el 종 | dist 종 | 고유 조합 |
|---|---|---|---|---|---|
| camel | **34°** | 5 | 3 | 3 | 45 |
| avocado-slice | **26°** | 5 | 3 | 3 | 45 |

칸 사이 간격이 방위 8°·고도 10°·거리 0.2× 라 인접 타일이 육안으로 거의 같다. 풀이 좁은 이유는
`pool_mode=budget` 이 τ 예산에서 역산하기 때문이고(§candidates.py), 그건 의도된 것이다 —
넓히면 G4_tau 에서 다 죽는다.

top-N 을 줄여도 각도 다양성이 거의 안 준다:

| top-N | camel az/el/dist 종 | avocado az/el/dist 종 |
|---|---|---|
| 27 | 5 / 3 / 3 | 5 / 3 / 3 |
| 15 | 4 / 2 / 3 | 4 / 2 / 3 |
| 12 | 4 / 2 / 2 | 4 / 2 / 2 |
| **9** | **4 / 2 / 2** | 3 / 2 / 2 |
| 6 | 4 / 2 / 2 | 3 / 2 / 2 |

**9칸(3×3) + 타일 640×360 → 1928×1088 (1.77:1)** 로 간다. 방위는 5종 중 4종이 남고(camel),
잃는 건 고도 1단·거리 1단인데 그게 각각 10°·0.2× 다. 12칸(4×3, 2572×1088, 2.36:1)은 종 수가
9칸과 **동일**하므로 더 줄 게 없다 — 12를 고를 이유는 "3칸 더 보고 싶다" 뿐이다.
`ranked` 가 `board_score` 내림차순이라 잘리는 건 항상 점수 하위다.

부수 효과: 이미지 토큰이 ~3배 줄어 preset 턴 8162 tok 이 크게 내려간다. D34 가 "숫자는 답을 안
바꾼다" 를 보였으므로 `coverage` 열을 빼서 더 줄일 수도 있지만, **그건 하지 않는다** — D34 ②가
VLM 의 자체 magenta 판정이 중간 구간에서 틀린다는 걸 보였고 숫자가 그 오차의 유일한 교정 채널이다.

#### D36 select 턴은 **숫자 우세**, 그림 채널은 **씬에 따라 켜졌다 꺼진다** (그림에 가짜 구멍을 칠해서 측정)

D34 가 "안 한 것"으로 남긴 세 가지 중 두 가지를 했다 — ① temperature 를 올려 prior 흔들기
② 후보 `select` 턴 ablation ③ **그림 개입**(구멍 없는 렌더에 가짜 magenta 를 칠하기).
(GPT-4.1 대조는 여전히 안 했다.) `eval/ablate_vlm_hole_perception.py` 를 v2
(`vlm_hole_ablation_v2`)로 다시 썼다. `select` 턴은 trace 가 없어서
`out/<video>/board/contract.txt` + `lbm/prompts/system_select.md` + `[board_candidates.png,
source_frames.png]` 로 프롬프트를 재구성한다.

조건 11종 (`--turn {traj,select,both}`):

| 조건 | 텍스트 숫자 | 그림 | 무엇을 가르나 |
|---|---|---|---|
| `A_full` | 그대로 | 그대로 | 기준선 |
| `B_no_num` | coverage/occlusion_pass 열 제거 | 그대로 | 숫자 없이 그림만으로 되나 |
| `C_no_img` | 그대로 | **없음** | 그림 없이 숫자만으로 되나 |
| `D_conflict` | `x → 1−x` | 그대로 | 충돌 시 어느 쪽을 따르나 |
| `E_paint` | 그대로 | **A_full 이 고른 타일에 가짜 magenta** | 숫자 있을 때 그림 개입이 먹나 |
| `F_paint_nonum` | 제거 | 같은 가짜 magenta | **결정적** 그림-단독 검사 |
| `G_shuffle` | 행 순서만 섞음 | 그대로 | 텍스트 순서 잡음의 크기 |
| `H_neutral` | preset 이름 → `M01..M13` (프롬프트+시스템 양쪽) | 없음 | 이름 prior 인가 (traj 전용) |
| `I_temp10` | 그대로 | 그대로 | temperature 1.0 |
| `J_paint_low` | 그대로 | **최하위 타일**에 같은 면적 magenta | 위치 대조군 |
| `K_paint_low_nonum` | 제거 | 최하위 타일 magenta | 위치 대조군 (숫자 off) |

칠하는 방식: 타일 중앙 60%×60% 를 `HOLE_COLOR=(255,0,255)` 로 덮는다. traj board 는 preset 행의
**중간·마지막 프레임만** 칠한다 (`system_traj.md` 가 "첫 프레임 멀쩡 / 마지막 프레임 magenta"를
보라고 지시하므로).

**① traj 턴 = prior. 아무것도 안 먹는다.** 9조건 × 2영상 × 3draw = 54 draw 중
`orbit_left_arc` 가 아닌 것은 camel `C_no_img` 3/3 (`s_curve`) 과 단발 2건
(`D_conflict#2 truck_right`, `F_paint_nonum#2 straight_ease`) 뿐. avocado-slice 는 **27/27
전부 `orbit_left_arc`**. 이름 중립화(`H_neutral` → `M01..M13`)·행 셔플·temperature 1.0·숫자
뒤집기·가짜 magenta 다섯 개 다 못 흔들었다. D34 결론 유지.

**② select 턴 27칸 = 그림 채널이 아예 안 잡힌다 + 잘린다.** 10조건 × 2영상 × 5draw = 100 draw:

| board | `finish_reason=length` | parse 실패 | prompt tok 평균 | completion tok 평균 / 최대 |
|---|---|---|---|---|
| 27칸 (4352×818) | **7/100** | 7/100 | 6040 | 776 / **2048 (=상한)** |
| 9칸 (1928×1088) | **0/100** | 0/100 | 4314 | 562 / 952 |

잘린 응답은 27후보를 쌍쌍이 비교하다 토큰을 다 쓴 것이고, 한 draw 는
`"A4 is the best candidate. A4 is the best candidate. ..."` 로 퇴화했다. 그리고 27칸에서는
숫자를 빼는 순간(`B_no_num`) camel 최상위 타일 픽이 5/5 → **0/5** 로 무너진다
(pick 평균 실제 coverage 0.970 → 0.850). 즉 개입할 여지 자체가 없다.

**③ select 턴 9칸 camel = 그림 채널이 실재하고 국소적이다.** 순서를 고정한 채 숫자만 끄고
그림 개입 위치만 바꾼 3조건 비교 (칠한 타일 `A1,C2` / 대조군 `C1,C3`, 5 draw):

| 조건 (9칸, 숫자 off) | picks | `{A1,C2}` 안 |
|---|---|---|
| `B_no_num` (깨끗한 그림) | A1,C2,A1,C2,A1 | **5/5** |
| `K_paint_low_nonum` (C1,C3 칠함 = 대조군) | A1,A1,C2,B2,A1 | **4/5** |
| `F_paint_nonum` (A1,C2 칠함 = 처치) | B2,B2,B2,C3,B2 | **0/5** |

reasoning 이 칠한 타일을 이름으로 짚는다 — `F_paint_nonum#0..#4` 전부 "Tiles A1 and C2 are
heavily magenta, indicating large missing-data regions". 대조군에서는 그 문장이 안 나온다.
**숫자를 켜면 분리가 사라진다**: `A_full` 5/5, `E_paint` 2/5, `J_paint_low` 2/5,
`G_shuffle` 1/5 — `G_shuffle` 은 그림을 안 건드렸는데도 4/5 가 B2 로 갔으므로 `E_paint` 의
움직임은 셔플 잡음과 구분되지 않는다. `D_conflict` 는 뒤집힌 숫자를 정확히 따라간다
(camel-9 `B1`×5, camel-27 은 `C8`/`C9`/`B4`).

**④ 같은 실험이 avocado-slice 에서는 안 먹는다 — 그리고 이유가 측정된다.** 9칸,
칠한 타일 `A2,A3,C3`: `B_no_num`·`F_paint_nonum`·`K_paint_low_nonum` 전부 `C3` 5/5 로 동일.
칠해도 안 피한다. 칠하기 전 board 의 타일별 magenta 실측 면적비:

| | A1 | A2 | A3 | B1 | B2 | B3 | C1 | C2 | C3 |
|---|---|---|---|---|---|---|---|---|---|
| camel (칠하기 전) | 0.007 | 0.086 | 0.076 | 0.149 | 0.030 | 0.081 | 0.061 | 0.049 | 0.037 |
| camel (A1,C2 칠함) | **0.362** | 0.086 | 0.076 | 0.149 | 0.030 | 0.081 | 0.061 | **0.394** | 0.037 |
| avocado (칠하기 전) | 0.371 | 0.326 | 0.335 | 0.347 | 0.364 | 0.335 | 0.248 | 0.225 | 0.251 |
| avocado (A2,A3,C3 칠함) | 0.371 | **0.610** | **0.636** | 0.347 | 0.364 | 0.335 | 0.248 | 0.225 | **0.597** |

camel 은 배경 0.03~0.15 위에 0.36~0.39 를 얹으니 대비가 명확하고, avocado 는 **이미 전 타일이
0.22~0.37** 이라 0.60 을 얹어도 순위 판단이 안 바뀐다. 즉 **그림 채널은 hole 대비가 클 때만
작동**하고, 구멍이 많은 씬에서는 숫자가 유일하게 작동하는 채널이다. D35 의 "coverage 열을 빼지
않는다" 결정을 그대로 굳힌다.

**적용한 설정** (`fit/bank/build_candidate_board.py` 기본값):
`--board_size 27 --board_columns 9 --tile_width 480 --tile_height 270`
→ **`9 / 3 / 640 / 360`**. 예전 동작은 저 플래그 4개로 그대로 재현된다.
`max_tokens` 는 2048 유지 (9칸에서 잘림 0/100 이라 올릴 이유가 없다).

**주의 — 지금 이 board 는 VLM 이 실제로 소비하지 않는다.** `lbm/loop.py` 는
`start_mode source_frame0` 에서 `{'select': 'skipped_source_frame0', 'micro':
'skipped_source_frame0', 'traj': 'vlm'}` 를 찍는다 (사용자 확정: "그냥 첫 카메라를 그대로
쓰는걸로 단순화하자"). 따라서 board 축소는 **프리뷰 렌더 비용 절감 + select 를 되살릴 때를
위한 준비**이고, 오늘의 결정 경로는 안 바뀐다.

산출물: `out/ablation/{hole_ablation_both.json, board27/, board9/}` + 칠한 board PNG.

#### D37 정적 preset 은 **coverage 최적해**다 — 지표 9번 `path_len_u` 를 넣는다

9칸 board 로 파이프라인을 다시 돌린 뒤 avocado-slice 가 `static_hold_locked` 를 골랐고
`path_len_u 0.0000` · emit `--scales 0` 인데 **verify 8종이 전부 PASS** 였다. 1 draw 로는
잡음인지 알 수 없어(`match-requires-seed-variance`) 같은 설정에서 `-m lbm.loop` 를 5번 더
돌렸다.

| 영상 | draw 5회 (+ 앞선 2회) | 결과 |
|---|---|---|
| avocado-slice | `static_hold_locked/lock` ×5 (+ static 1, orbit 1) | **6/7 정적** |
| camel | `orbit_left_arc/drift` ×5 (+ orbit 2) | 7/7 orbit |

원인은 프롬프트에 그대로 적혀 있다 — VLM 의 `reasoning`:
`"'static_hold_locked' preset is the strongest choice because it maintains 100% coverage
throughout the shot with no camera motion, ensuring no hallucinated regions"` (coverage 0.99).
**contract 는 coverage 를 상으로 주고 움직임에는 아무 상도 주지 않는다.** 안 움직이는 게
coverage 최적해이므로 이건 VLM 의 실수가 아니라 설계 구멍이다. D34/D36 의 "품질을 지키는 건
VLM 이 아니라 게이트다" 가 여기서도 맞다 — 다만 이번엔 **게이트에도 그 항목이 없었다.**

**넣은 것**: `verify.py` 지표 9번 `path_len_u = Σ|Δp| / S`, PASS ≥ 0.05 u, WARN ≥ 0.025 u.
`--min_path_len_u 0` 이면 행이 안 생긴다 (예전 동작 그대로). 정적 draw 를 실제로 통과시켜
확인했다:

| | camel (orbit) | avocado 정적 draw | avocado orbit draw |
|---|---|---|---|
| `hole_fraction` | 0.0737215 | 0.0154063 | 0.289933 |
| `tau_max` | 0.196051 | 0.128633 | 0.197946 |
| `jerk_ratio` | 1.97805e-05 | 0 | 0.000128102 |
| `path_len_u` | 0.147629 PASS | **0 FAIL** | 0.15771 PASS |
| verdict | PASS | **FAIL** | PASS |

정적 플랜은 hole 이 0.0154 로 **제일 좋다.** hole 만 보면 이길 방법이 없다는 뜻이고, 그래서
지표를 하나 더 두는 것 말고는 방법이 없다.

**안 한 것 (선택지로 기록)**: ① contract 에 움직임 보상을 넣기 — coverage 열 옆에
`path_len_u` 를 같이 주고 "정적은 마지막 수단"이라고 system 프롬프트에 쓰는 것. D34/D36 이
preset 턴은 prior 라고 했으므로 **먹힐 가능성은 낮다.** ② `static_hold*` 를 preset board 에서
빼는 플래그 — 확실하지만 "정적이 정답인 씬"을 없앤다. ③ subject 를 동적 노드로 강제
(`--subject_id`) — avocado 는 subject 가 `stat_0 (table)` 로 뽑혔다. ①보다 ③이 근본적이다.

---

## 9단계 — camera augmentation sampler (`fit/bank/sample_camera_bank.py`)

목표가 바뀌었다 (사용자, 2026-08-21): **"이 씬에 가장 좋은 카메라 1개"가 아니라 "다양한 카메라
N개"**. 소스 카메라 움직임을 재현하는 게 아니라 augmentation 이다.

### D38 선택자를 VLM 에서 **열거**로 바꾼다

traj 턴은 D34/D36/D37 에서 세 번 측정한 결과 **prior** 였다 — 54 draw 중 non-`orbit_left_arc`
가 5개뿐이고, 숫자 제거 · 이미지 제거 · 숫자 반전 · 가짜 마젠타 · 라벨 중립화 · temp 1.0 어느
것도 못 움직였다. mode collapse 하는 선택자는 augmentation 에 쓸 수 없다. 그래서 축 4개
(anchor × preset × τ × speed/tracking/bias)를 **전량 열거**하고, VLM 은 나중에 caption 쪽
(카메라 → 텍스트)으로 옮긴다. `decision.source` 는 `sampler` — `vlm` 도 `fallback` 도 아니다.

기존 `lbm/loop.py` 경로는 **그대로 둔다** (`--no_vlm` 포함). sampler 는 별도 스크립트다.

### D39 τ 사다리를 훑고 `hole_fraction` 은 **기록만** 한다 (게이트 아님)

사용자 확정: "둘 다 — 강도 사다리로 뽑는다". `max_tau 0.30` / `min_coverage 0.55` 게이트는
"렌더가 GT 에 가까워야 한다"는 전제였는데, 학습 pair 생성에는 그 전제가 없다 — hole 을 채우는
게 하류 video model 의 일이다. `--tau_ladder 0.10 0.20 0.35 0.60 1.00` 를 훑고 각 단에서
hole 을 실측해 남긴다. 하드 컷은 하나뿐이다: anchor 가 소스 frame 0 에서 보일 것.

anchor 는 게이트 통과 노드 **전량** (사용자 확정: "게이트 통과하는 모든 노드에 대해 전량").
camel 6/6, avocado-slice 7/11 (4개는 `max_area_frac < 0.01`).

실측 (2026-08-21, 49프레임 중 5프레임 렌더, 정지 preset 제외):

| τ* | camel n / hole / hole_max / inFrame | avocado n / hole / hole_max / inFrame |
|---|---|---|
| 0.10 | 72 / 0.245 / 0.309 / 0.90 | — (전량 saturated, D41) |
| 0.20 | 72 / 0.298 / 0.408 / 0.87 | 84 / 0.301 / 0.373 / 0.90 |
| 0.35 | 72 / 0.359 / 0.508 / 0.82 | 84 / 0.343 / 0.444 / 0.89 |
| 0.60 | 72 / 0.424 / 0.593 / 0.77 | 84 / 0.389 / 0.534 / 0.85 |
| 1.00 | 72 / 0.480 / 0.670 / 0.74 | 84 / 0.450 / 0.630 / 0.81 |

**hole 이 τ 에 대해 단조지만 포화하지 않는다** — τ 를 10배 키워도 hole 은 2배가 안 된다
(camel 0.245 → 0.480). 즉 사다리 윗단이 "쓸 수 없는 영역"이 아니다. 어디서 자를지는 하류
모델이 채우는 능력으로 나중에 정한다.

### D40 회전 전용 preset 은 사다리를 **각도**로 탄다

**τ 로 크기를 못 정하는 preset 이 있다.** `τ = |Δp|/z_med` 인데 `pan_left/right` 는 이동이 0 이라
어떤 s 를 곱해도 τ 가 같다. `fit_tau` 의 `tau_hi <= target_tau` 가지가 걸려 항상
`max_scale`(4.0)로 튄다 — 1차 실행에서 camel `pan_left` 가 사다리 5단 **전부** `path_len_u
0.0000`, `hole 0.740` 동일값이었다. 5칸이 같은 카메라였다는 뜻이다.

`ROTATION_ONLY_PRESETS` 만 사다리를 `pan_deg = --pan_deg_at_max × rung/max_rung /
FIT_TAU_MAX_SCALE` 로 옮긴다. `max_scale` 을 나누는 건 `fit_tau` 가 그만큼 되곱하기 때문이고,
기본값이 바뀌어도 따라가도록 시그니처에서 읽는다. 이 행들은 `tau_invariant: true` 로 찍혀 τ
요약표 평균에서 빠지고 **별도 표**로 나온다 (단위가 다르다).

수정 후 실측:

| τ* | pan_deg | camel hole / inFrame | avocado hole / inFrame |
|---|---|---|---|
| 0.10 | 6.0 | 0.067 / 0.62 | — |
| 0.20 | 12.0 | 0.161 / 0.53 | 0.128 / 0.80 |
| 0.35 | 21.0 | 0.308 / 0.40 | 0.211 / 0.70 |
| 0.60 | 36.0 | 0.531 / 0.28 | 0.338 / 0.61 |
| 1.00 | 60.0 | 0.679 / 0.18 | 0.530 / 0.40 |

**같은 hole 을 회전으로 사는 게 이동으로 사는 것보다 비싸다** — camel 60° pan 의 hole 0.679 는
τ=1.0 이동(0.480)보다 높은데 시차는 0 이다. 대신 `subject_in_frame` 이 0.18 까지 떨어진다:
pan 은 `aim="traj"` 라 subject 를 안 따라간다. augmentation 으로는 유효하되 **subject 중심
shot 을 원하면 못 쓴다**는 뜻이고, 그 판단은 `subject_in_frame` 열로 남겼다.

### D41 `tau_start > rung` 인 단은 **뱅크에서 뺀다** (정지 preset 은 예외)

`tau_start` 는 시작 pose(=소스 frame 0)를 그냥 붙들고 있을 때의 τ 다 — 소스 카메라 자신의
움직임이 예산을 먼저 먹는다 (camel 0.0042, avocado 0.1286). avocado 는 `tau_start 0.1286 >
0.10` 이라 τ0.1 단에서 `fit_tau` 의 첫 가지가 걸려 **98개 변이가 전부 움직임 0** 으로 나왔다.
1차 실행의 뱅크에는 그게 `orbit_left_arc τ0.1` 같은 이름을 달고 들어가 있었다 — 실제로는
`static_hold` 의 복제본 98개다.

`--drop_saturated`(기본 켬)로 빼고, 몇 건을 왜 뺐는지 `bank.json.dropped_saturated` 와 요약표에
남긴다. **정지 preset 자신은 예외** — 거기서는 identity 가 원래 의도한 결과다. (이 예외를 안
넣었더니 avocado 에서 `static_hold`/`static_hold_locked` 14건이 통째로 사라졌다.)

회전 전용 preset 도 saturated 로 같이 걸린다. τ 와 무관한데도 `fit_tau` 가 rel 을 통째로 누르기
때문이다 — 그래서 avocado 의 회전 표는 0.20 부터 시작한다. **선택지로 기록**: `fit_tau` 가
이동이 0 인 궤적에서 첫 가지를 타지 않게 고치면 살아난다. 지금 안 고치는 이유는 `lbm/loop.py`
가 같은 함수를 쓰고 있어서 기존 경로의 동작이 바뀌기 때문이다 (CLAUDE.md 의 "기존 방식 유지").

최종 열거: camel 432 (6 anchor × 14 이동 preset × 5단 + 6 × 2 정지),
avocado-slice 406 (7 × 14 × 4 + 7 × 2).

---

## 10단계 — preset 별 크기 상한 (`fit/bank/fit_hole_ladder.py`)

### D42 상한은 τ 한 값이 아니라 **preset 마다 다른 값**이다 — hole 을 고정하고 τ 를 푼다

D39 의 τ 사다리를 전량 렌더해 보니 **모양이 강도를 압도한다**. camel `dyn_0` 를 τ=0.35 로
고정하고 preset 만 바꾸면 hole 이 0.015(`straight_ease`) ~ 0.651(`pedestal_down`) 로 **43배**
갈린다. 반면 같은 preset 에서 τ 를 10배(0.10→1.00) 키워도 hole 은 2~4배밖에 안 움직인다.
즉 **τ 한 값으로 전 preset 을 자르면 어떤 건 아직 멀쩡한데 잘리고 어떤 건 이미 망가졌는데
살아남는다.** 사용자 확정: 전역 hole 컷이 아니라 **preset 별 상한**.

그러려면 축을 뒤집어야 한다 — hole 을 고정하고 **크기 손잡이를 푼다**. `fit_hole_ladder.py` 는
`HOLE_LADDER = (0.10, 0.20, 0.35, 0.50)` 각 단마다 (anchor, preset) 별로 이분법 4회를 돌려
그 hole 을 주는 knob 을 찾는다. knob 은 D40 과 같은 두 종류다 — 이동 preset 은 `tau`,
회전 전용은 `pan_deg`. 각 후보는 5프레임으로 이분하고 답이 나오면 13프레임으로 재측정한다.

실측 (camel 336 변이 / 5,563 렌더, avocado-slice 392 / 6,500. `solved` 행 median):

| preset | knob | hole 0.20 camel / avo | hole 0.35 camel / avo | hole 0.50 camel / avo |
|---|---|---|---|---|
| `straight_ease` | tau | 1.000 / 0.553 | 1.562 / 1.300 | — / 1.125 |
| `push_in_arc` | tau | 1.250 / 0.775 | 1.875 / 1.125 | — / 2.188 |
| `pull_out_arc` | tau | 0.294 / — | 0.553 / 0.366 | 1.000 / 0.875 |
| `orbit_left_arc` | tau | 1.000 / — | — / 0.713 | 0.650 / 1.150 |
| `orbit_right_arc` | tau | 0.322 / 0.266 | 0.475 / 0.491 | 0.562 / 1.250 |
| `pan_left` | pan_deg | 10.875 / 18.750 | 19.312 / 36.000 | 28.500 / 54.000 |
| `pan_right` | pan_deg | 18.188 / 19.875 | 26.625 / 37.500 | 36.000 / 54.000 |
| `truck_left` | tau | 0.247 / 0.412 | 0.475 / 1.000 | 0.825 / 1.750 |
| `truck_right` | tau | 0.228 / 0.459 | 0.350 / 0.825 | 0.506 / 2.375 |
| `pedestal_up` | tau | 0.275 / 0.381 | 0.475 / 0.900 | 0.725 / 1.750 |
| `pedestal_down` | tau | 0.073 / 0.177 | 0.128 / 0.303 | 0.205 / 0.600 |
| `rise_reveal` | tau | 1.750 / — | 2.375 / 0.664 | 2.500 / 1.125 |
| `drop_reveal` | tau | 0.475 / — | 1.562 / 0.189 | 0.258 / 2.000 |
| `s_curve` | tau | 0.312 / — | — / 0.950 | 1.125 / 1.250 |

**같은 hole 0.35 를 사는 데 드는 τ 가 camel 에서 0.128(`pedestal_down`) ~ 1.875(`push_in_arc`)
로 15배 갈린다.** 이게 preset 별 상한을 따로 재야 하는 이유 그대로다. `--cap_hole` 단의 표를
`tau_caps.json`(`lbm_tau_caps_v1`) 로 저장한다 — preset → anchor → `{knob, status, tau_max,
path_len_u, hole_fraction}`.

### D43 `aim="look_at"` 은 **움직이기 전에** hole 을 쓴다 — 그게 사다리 아랫단의 바닥이다

사다리를 돌리다 `clamped_low`(knob 을 범위 최소 0.02 까지 내려도 hole 이 목표보다 높다) 가
camel 90행 / avocado 144행 나왔다. **전부 `aim="look_at"` 이고 `aim="traj"` 는 0행이다.**
그 행들의 `path_len_u` 는 0.010 (camel) / 0.000 (avocado) 이고 `view_angle_max_deg` 도
0.3° / 4.2° 다 — 궤적이 사실상 정지인데 hole 이 0.47 / 0.37 이다.

원인은 궤적이 아니라 **시작 자세**다. `look_at` 은 매 프레임 카메라를 anchor 쪽으로 다시 겨눈다.
그 재조준 회전만으로 소스 시야에서 벗어나고, 그 대가를 궤적이 한 발짝도 떼기 전에 이미 치른다.
`traj` 는 primitive 회전을 그대로 두므로 소스 방향을 유지해 바닥이 훨씬 낮다.

anchor 별 최소 hole (camel / avocado-slice):

| anchor | camel look_at / traj | anchor | avocado look_at / traj |
|---|---|---|---|
| `dyn_0` | 0.057 / 0.089 | `dyn_0` | 0.318 / 0.092 |
| `dyn_1` | 0.135 / 0.090 | `stat_0` | 0.079 / 0.092 |
| `stat_0` | 0.070 / 0.090 | `stat_1` | 0.306 / 0.068 |
| `stat_1` | 0.293 / 0.090 | `stat_2` | 0.330 / 0.068 |
| `stat_2` | 0.221 / 0.083 | `stat_3` | 0.153 / 0.067 |
| `stat_3` | **0.647** / 0.073 | `stat_4` | **0.638** / 0.091 |
| | | `stat_5` | 0.183 / 0.086 |

**`traj` 는 어느 anchor 에서든 0.067~0.092 로 평평하고, `look_at` 만 anchor 를 탄다** (0.057 →
0.647, 11배). 즉 look_at 계열에서 hole 바닥은 "그 anchor 를 쳐다보는 데 드는 값"이고, 그보다
낮은 단은 **애초에 만들 수 없다**. camel `stat_3` 는 look_at 4단이 전부 바닥에 눌려 있다 —
`lookat_floor.mp4` 에서 그 타일만 얼어 있는 게 그것이다.

**뱅크 규칙**: `clamped_low` 행은 지우지 않고 `status` 로 남긴다. 사다리를 못 탄 게 아니라
**그 anchor 를 그 aim 으로 보는 비용이 그만큼**이라는 측정치이고, anchor 를 고를 때 그 값이
필요하다. `unreached`(21/8행) 와 `shape_limited`(21/10행) 도 같은 이유로 남긴다.

### D44 `fit_tau` 의 `max_scale` × `DEFAULT_SHAPE` 는 **구조적 천장**이다 — `--shape_headroom`

스모크 테스트에서 camel `dyn_0 straight_ease` 가 hole 0.35 를 `unreached` 로 냈다. knob 은
범위 상한 3.0 에 붙었는데 실측 `tau_max` 는 1.1493 에서 멈춰 있었다 — knob 범위가 아니라
`fit_tau` 의 `max_scale`(4.0) 이 `DEFAULT_SHAPE` 의 `dolly_frac 0.35 × radius` 에 걸린 것이다.
τ 를 아무리 크게 요구해도 **모양이 그만큼 안 크면 못 간다**.

`render_hole` 안에 재시도를 넣는다: `fit_tau` 가 `max_scale` 에 붙어 있으면
`dolly_frac`/`lateral_frac` 을 `--shape_headroom`(2.0) 배로 키워 다시 푼다, 최대
`--shape_doublings`(4) 회. 그러고도 붙어 있으면 `unreached` 가 아니라 **`shape_limited`** 로
찍는다 — 손잡이 탓과 모양 탓을 구분해야 다음에 뭘 고칠지 알 수 있다. 수정 후 같은 행이
`solved`(knob 2.500, `tau_max` 2.4979, `shape_mult` 4.0) 로 바뀌었다.

`shape_limited` 는 camel 21행 / avocado 10행 남았고 전부 arc/orbit/s_curve 다 — 그쪽은
`sweep_deg` 가 `orbit_span_frac × obs_az_span` 에 걸려 있어서 dolly 를 키워도 안 늘어난다.
**선택지로 기록**: sweep 도 headroom 대상에 넣을 수 있다. 지금 안 하는 이유는 `obs_az_span`
밖으로 나가면 관측이 없는 방위라 hole 이 아니라 아예 빈 화면이 되기 때문이다.

### D45 hole 만으로는 못 자른다 — `subject_in_frame` 이 두 번째 제약

hole 0.50 단에서 camel `pan_right` 는 hole 0.481 로 통과하지만 `subject_in_frame` 이 **0.15**
다. `truck_right` 0.27, `pedestal_up` 0.39. 전부 `aim="traj"` 라 subject 를 안 따라가고,
크기를 키우면 subject 가 그냥 화면 밖으로 나간다 (D40 에서 회전 preset 을 두고 이미 관찰한 것과
같은 현상인데, hole 을 고정하니 이동 preset 에서도 똑같이 나온다).

즉 상한은 두 개다 — **hole 은 "하류 모델이 채워야 할 양", `subject_in_frame` 은 "shot 이
subject 를 담고 있나"**. 뱅크는 둘 다 열로 갖고 있고, 어느 쪽으로 자를지는 소비처가 정한다
(subject 중심 shot 이면 `subject_in_frame` 하한, 순수 augmentation 이면 hole 상한).
지금은 둘 다 기록만 하고 게이트하지 않는다 (D39 와 같은 판단).

### D46 뱅크에는 충돌·가림 검사가 **없었다** — `audit_bank_geometry.py` 로 재고 열로 남긴다

`lbm/gates.py` 는 처음부터 둘 다 갖고 있다 — `G1_behind`(카메라가 관측된 표면 뒤인가)와
`G3_occlusion`(subject 가 z-buffer 에서 다른 것에 가려지는가). 그런데 **뱅크 경로가 그걸 안
부른다**: `sample_camera_bank.py` 의 `measure_trajectory` 는 `hole_fraction` / `subject_area` /
`subject_in_frame` 만 재고, 행에 `"gates": None` 이 그대로 박혀 있다. board 경로
(`build_candidate_board.py`)는 **시작 pose 한 장**에만 게이트를 건다. 뱅크는
`start_mode=source_frame0` 이라 시작 pose 는 소스 카메라 자신이고 정의상 충돌이 없다 —
**위험이 전부 나머지 48프레임에 있었는데 아무도 안 보고 있었다.**

`hole` 이 대신 잡아주지 않는다. 카메라가 벽을 통과하면 벽 **너머** 관측이 그대로 그려져서
`valid_mask` 가 멀쩡할 수 있다. 실측 상관계수 `corr(hole, behind_frac)` = **−0.014** (camel) /
**+0.059** (avocado) — 사실상 0 이고 camel 은 부호마저 반대다 (위반 변이의 평균 hole 0.280 <
정상 0.337). `corr(hole, occlusion_pass)` 도 −0.199 / −0.031.

실측 (`eval/audit_bank_geometry.py`, 소스 7프레임 재투영, margin 0.02·S):

| 뱅크 | 변이 | G1 위반 | G3 `occlusion_pass < 0.4` | 둘 다 | 둘 중 하나 |
|---|---|---|---|---|---|
| camel `bank` (τ 사다리) | 432 | 6 (1.4%) | — | — | — |
| camel `hole_bank` | 336 | 24 (7.1%) | 14 | 2 | 36 (10.7%) |
| avocado `bank` | 406 | 11 (2.7%) | — | — | — |
| avocado `hole_bank` | 392 | 23 (5.9%) | 56 | 1 | 78 (19.9%) |

**G1 은 전진 preset 에 몰린다** — camel 24건 중 `straight_ease` 13 / `push_in_arc` 9, avocado
23건 중 `push_in_arc` 11 / `straight_ease` 9. D42 에서 그 둘이 가장 큰 τ 상한(1.562 / 1.875)을
받은 preset 이라는 게 정확히 원인이다: 앞으로 밀고 들어가는 궤적을 hole 기준으로 키우면 씬
안쪽으로 뚫고 들어간다. 최악은 camel `dyn_0__straight_ease__hole0.5` 로 **49프레임의 73.5% 가
표면 뒤**인데 hole 은 0.356 이다 (같은 anchor 의 `truck_left` 0.507 보다 낮다).
`collision.mp4` 가 그 대조다.

**G1 과 G3 은 거의 안 겹친다** (camel 36건 중 2, avocado 78건 중 1). 다른 고장이다 —
G1 은 카메라 위치, G3 은 시선 위의 방해물. 그래서 hole 을 포함해 **판정 축이 셋**이다.

hole 사다리 쪽 위반율이 τ 사다리보다 3~4배 높다 (7.1% vs 1.4%). hole 을 목표로 크기를 풀면
τ 사다리가 안 가던 영역까지 밀고 들어가기 때문이고, D42 의 상한을 그대로 쓰면 안 되는
이유다.

**규칙**: 감사(audit)로 두고 **행을 지우지 않는다** — `geometry.csv` 에 `behind_frames` /
`behind_frac` / `behind_worst_src` / `occlusion_pass` 열을 붙인다. D39·D45 와 같은 판단이다
(재고 기록하되 게이트하지 않는다, 자를지는 소비처가 정한다). G1 은 렌더가 필요 없어서
(재투영뿐) 전 뱅크가 몇 초, G3 은 프레임당 렌더 2회라 `--occlusion` 으로 따로 켠다.

**선택지로 기록** (아직 안 함, 권장 순):
1. `sample_camera_bank.py` / `fit_hole_ladder.py` 의 `measure_trajectory` 에 G1 을 직접 넣는다
   — 렌더가 안 드니 비용이 사실상 0 이고, `fit_hole_ladder` 의 이분법이 **충돌 없는 최대
   크기**를 풀게 만들 수 있다 (지금은 hole 만 보고 푼다). ← 권장
2. `behind_frac > 0` 을 뱅크에서 드롭 (D41 의 `--drop_saturated` 와 같은 형태).
   지금 안 하는 이유: 표면 뒤로 지나가는 게 **항상** 나쁜지 아직 안 정했다 — 얇은 전경
   물체 뒤를 스치는 것과 벽 속으로 들어가는 것을 `behind_worst_src`(몇 개 소스 프레임이
   동의하는가)로 가를 수 있는데 그 임계를 안 재봤다.
3. G3 임계 0.40 은 board 경로에서 물려받은 값이다. 뱅크의 `occlusion_pass` median 이
   0.908 / 0.837 이라 0.40 은 꽤 느슨하다 — 재조정 여지.

### D47 이분법이 **충돌 없는 최대 크기**를 푼다 (D46 옵션 1 구현)

D46 은 충돌을 재기만 하고 행을 남겼다. 그런데 감사 결과가 "전진 preset 이 씬을 뚫는다"로
한 방향이라, 소비처에 떠넘길 게 아니라 **이분법이 애초에 그 크기를 안 고르게** 하는 게 맞다.
G1 은 렌더가 0회(재투영뿐)라 이분법 안에서 후보마다 불러도 비용이 사실상 0 이다 — D46 이
옵션 1을 권장으로 둔 이유 그대로다.

`fit_hole_ladder.py` 의 `solve_knob` 이 **예산 4개**를 순서대로 본다. 답은 넷 다 통과하는
최대 손잡이값이고, 무엇이 멈춰 세웠는지를 `binding` 열에 남긴다:

| 예산 | 판정 | binding |
|---|---|---|
| collision | G1 위반 프레임 비율 > `--max_behind_frac`(기본 0.0) | `collision` |
| obb | 노드 OBB 부호거리 < `--min_obb_clear`(기본 0.02 u) (§D49) | `obb` |
| clearance | 3D standoff < 임계 (§D48) | `clearance` |
| hole | `hole_fraction ≥ target_hole` | `hole` |

`--no_collision_free` 면 앞 셋이 꺼지고 **hole 만 보던 예전 판정 그대로** 돈다 (집 규칙:
기존 경로를 옵션으로 보존). 회귀 확인: camel `dyn_0` × `truck_left`/`orbit_left_arc` 8행 ×
6열이 `hole_bank_holeonly` 와 **0건 불일치** (D49 배선 후 재확인: 16행 0건 불일치).

G1 자체도 조였다 — `--behind_src_frames` 7→**13**, `--behind_clear_frac` 0→**0.10**(표면 앞에
*요구하는* 여유; `margin_frac` 0.02 는 관통을 *봐주는* 여유라 방향이 반대다),
`--behind_radius_px` 0→**2**(투영점 하나가 아니라 5×5 패치의 최소 depth — 얇은 물체 가장자리를
스칠 때 바로 옆 배경의 먼 depth 가 잡혀서 안 걸리던 것).

새로 생긴 status: `collision_limited` / `clearance_limited`(예산에 막혀 목표 hole 미달),
`clamped_low`(손잡이 하한 0.02 에서도 막힘). 그리고 **접힌 단** — 충돌 천장이 hole 0.10 단보다
낮으면 4단이 전부 같은 궤적이 된다. 사다리가 강도 축을 잃은 자리라 따로 센다
(camel 94/336, avocado 106/392).

### D48 clearance 는 렌더 depth 가 아니라 **3D standoff** 로 잰다 — 임계도 절대값이 아니다

D47 을 켜고도 사용자 판정은 "그래도 충돌한다"였다. G1 은 카메라가 표면 **뒤**인지만 본다 —
표면 **앞 1 cm** 는 통과한다. 그래서 "코앞 여유" 예산을 하나 더 붙였는데, 첫 구현이 틀렸다.

**틀린 판정 (`near_depth`)**: 렌더한 depth 의 하위 백분위(p01)/S. 방향에 의존한다.
- **과검출** — 그 p01 픽셀들의 세로 위치 median 이 **0.98**. 화면 맨 아래 가장자리, 즉 카메라
  밑을 지나가는 바닥이지 장애물이 아니다.
- **미검출** — 화면 **밖**의 가까운 기하는 아예 못 본다. camel `dyn_0__truck_left__hole0.5` 은
  `near_depth` 0.687(통과)인데 실제 3D 최소거리는 **0.090**.

한 지표가 과검출과 미검출을 동시에 했고, 결과는 사다리 붕괴였다: camel `binding` 이
`clearance` **90/336**, 변경된 행의 `path_len` median 이 **0.574 → 0.014** — augmentation 이
통째로 죽었다.

**맞는 판정 (`lbm/render.py: CloudRenderer.standoff`)**: 궤적 각 프레임에서 카메라 중심과
**그 시각 보이는 점** 사이의 3D 최소거리. 방향에 안 걸린다. `temporal_persistence=False` 라
동적 물체가 **자기 시각 위치로만** 장애물이 된다 (G1 은 궤적 프레임 하나를 샘플된 소스 프레임
*전부*에 되쏘아서 움직이는 물체를 모든 시각의 위치에서 동시에 막는다 — 과보수적이다).
렌더가 0회, `|p−c|² = |p|² − 2p·c + |c|²` 전개에 `|p|²` 캐시.

**임계는 소스 카메라 자신의 standoff 대비 배수**다. 처음 쓴 절대값 0.15·S 는 **소스 카메라조차
통과 못 하는** 값이었다 — 실측 `source_standoff` camel **0.0849·S** / avocado-slice
**0.1369·S**. 씬마다 기하가 카메라에 얼마나 붙어 있는지가 다르므로 절대값을 못 쓴다.
`--min_standoff_ratio`(기본 ~~0.90~~ → 0.80 → **0.0 = 꺼짐, D50**), 절대값이 필요하면
`--min_standoff`. **standoff 는 지금 기본으로 꺼져 있다** — 아래 D50 을 먼저 읽을 것. 이 항목의
전량 표·스윕은 전부 standoff 가 켜져 있던 시절 수치다.

camel 의 0.086~0.090·S 바닥이 튄 splat 이 아니라 **실제 표면**인지 확인했다 — frame 0 에서
k번째 최근접 거리 k=1 0.0902 / k=10 0.0907 / k=50 0.0912 / k=200 0.0920 / k=1000 0.0945 /
k=5000 0.1025, 0.2·S 안에 6,696 점. frame 24 는 깨끗하다 (k=1 0.3237, 0.2·S 안 0점).
avocado frame 0 은 k=1 0.1369 … k=5000 0.1820, 8,366 점.

ratio 스윕 (camel `dyn_0`, 전진 preset):

| ratio | 임계 | `straight_ease` | `push_in_arc` |
|---|---|---|---|
| 0.70 | 0.0595 | 0.020 | **1.000** (path 0.795, stdof 0.063) |
| 0.80 | 0.0680 | 0.020 | 0.939 (path 0.743) |
| **0.90** | 0.0765 | 0.020 `clearance_limited` | 0.020 `clearance_limited` |
| 1.00 | 0.0849 | `clamped_low` | `clamped_low` |

처음엔 **0.90 으로 갔다** — 사용자 지시가 "좀 더 거리 있게"였다. 그 대가가 곧바로 나왔다:
camel 에서 **전진 preset 이 6 anchor 전부 손잡이 하한(0.02)에 붙는다** (path 0.011~0.015 =
사실상 정지). camel 소스 카메라가 이미 자기 clearance 바닥에 있어서 앞으로 미는 순간 위반이라
그렇다. 사용자가 렌더를 보고 "이젠 카메라가 거의 안움직이는데"라고 했고, 4분면 영상
(`D48_ratio_candidates.mp4`, r0.90/0.80/0.70/0.60 × `push_in_arc`)을 보고 **0.80 을 골랐다**.
그래서 `--min_standoff_ratio` 기본값은 **0.80** 이다. `straight_ease` 는 어느 ratio 에서도
0.020 이라 여기서 살아나지 않는다 — 그건 ratio 가 아니라 camel 소스 앞의 정적 기하 문제다.
0.80↔0.90 사이 `push_in_arc` 의 불연속(0.939 → 0.020)은 미해결.

전량 결과 (13 소스 프레임, clear 0.10·S, patch r2):

| 씬 | n | binding collision / clearance / hole / none | 남은 위반 | 접힌 단 | 렌더 | 벽시계 |
|---|---|---|---|---|---|---|
| camel | 336 | 2 / 26 / 279 / 29 | G1 0, standoff 0 | 94 | 5,639 | 17분 59초 |
| avocado-slice | 392 | 1 / 19 / 362 / 10 | G1 0, standoff 0 | 106 | 6,476 | 19분 57초 |

이전 뱅크 대비 (손잡이값이 바뀐 행 수):

| 기준 | camel | avocado |
|---|---|---|
| `hole_bank_clear0`(D47만) → standoff | 26/336, 전부 **작아짐**, 전부 `push_in_arc`/`straight_ease` | 28/392, 전부 작아짐 (+orbit 3) |
| `hole_bank_near015`(틀린 판정) → standoff | 96/336 — **커짐 70** (pan_right 24, pedestal_down 18, truck_right 14, pull_out_arc 5, s_curve 4, drop_reveal 3, orbit_right 2), 작아짐 26 (전진 preset) | 16/392 — 커짐 5, 작아짐 11 |

`near015 → standoff` 의 **양방향** 변화가 D48 의 근거 그 자체다: 방향 의존 지표가 옆·뒤로 가는
궤적 70개를 잘못 잘랐고, 앞으로 가는 궤적 26개는 잘못 통과시켰다.

`near_depth` 는 **열로는 남긴다** (기본 `--min_near_depth 0` = 끔). 판정에서만 뺐다.

**선택지로 기록**:
1. ~~`--min_standoff_ratio` 를 0.80 으로~~ → **채택됨** (위 참조, 사용자 선택). 전량 결과 표는
   0.90 기준이므로 0.80 기준 수치와 섞어 읽지 말 것.
2. `behind_surface_frames` 에 `dynamic_mask` 를 넘겨 **동적 픽셀은 `t == f` 일 때만** 장애물로
   세기. 비용 0, G1 의 과보수성이 사라진다. 지금 안 넣은 이유는 넣는 순간 위 두 실행과 비교가
   깨지기 때문이다. ← 다음 변경 후보
3. 바닥 점 제외(`floor + 0.05·S` 위만)는 **no-op** 이었다 (camel 0.087 raw vs 0.087 no-ground).
   `standoff` 가 방향에 안 걸리므로 바닥이 애초에 최소거리를 지배하지 않는다.

**ratio 0.80 전량 결과 (0.90 대비)**: camel 336행 중 **4행만** 바뀌었다 —
`dyn_0 push_in_arc` 4단이 전부 0.020 → 0.939 (path 0.013 → 0.743). path_len_u 총합
97.933 → 100.856. 스윕을 `dyn_0` 하나에서만 돌렸는데 그 anchor 가 유일하게 0.80↔0.90 경계에
걸려 있었다 — **한 anchor 스윕으로 전량 효과를 외삽하면 안 된다**는 사례로 남긴다.

---

### D49 세 번째 충돌 예산: 노드 **OBB** clearance (G5) — 표면이 아니라 부피

사용자 지시: "물체 obb 기준으로도 충돌 판정해줘".

G1 도 standoff 도 재는 건 **관측된 표면**이다. 표면은 물체의 껍질이라, 카메라가 물체 **안**을
지나가도 반대쪽 껍질이 뒤에 남으면 G1 은 "표면 앞"으로, standoff 는 "가까운 점"으로만 읽는다.
scene graph 노드 OBB 는 이미 있으므로 부피 판정을 공짜로 붙일 수 있다.

`lbm/gates.py`:

    obb_signed_distance(p_g, center, extent, R)   # 부호 거리. 음수면 박스 안
    obb_clearance(poses, nodes, T_gw)             # (dists(f,), 최근접 노드 id(f,))

부호 거리는 `‖max(d,0)‖ + min(max(d), 0)`, `d = |q| − extent/2`. 안팎이 한 식으로 이어져야
이분법이 이 값을 단조로 밀 수 있다 — "안이면 0" 같은 포화를 만들면 안 된다.

좌표: `obb` 는 그래프 프레임 G 에 있고 `T_gw[:3,:3] = R_gw/S` 라 **G 좌표가 곧 u 단위**다.
world 카메라 중심을 `T_gw` 로 옮기기만 하면 되고 S 로 다시 나누지 않는다.
동적 노드는 `node_obb_at(node, f)` 로 그 프레임 위치를 쓴다. 정적 노드까지 프레임별 track 을
쓰면 추정 jitter(camel `fence` 의 `path_len_u` 0.32)가 판정에 그대로 들어온다.

**임계는 배수가 아니라 절대 마진 `--min_obb_clear 0.02` u.** 소스 카메라 자신의 최소 OBB
clearance 가 camel **+0.5107 u**(`stat_0` fence) / avocado-slice **+0.1165 u**(`stat_4` chair)
로 4배 차이라, 배수로 걸면 camel 만 과하게 조여진다. 0.02 는 두 소스 값 한참 아래이고 순수
관통(음수)보다는 위다. 렌더 0회 (노드당 3×3 곱 하나).

> **이 문단은 D51 에서 뒤집혔다.** 사용자가 "충돌 판정을 조금 키워줘"라고 해서 절대 마진을
> 올려보니 **하나의 절대값으로 두 씬을 못 덮는다** — 여기 적힌 4.4배 차이가 바로 그 이유였고,
> 위에서 "배수를 못 쓰는 근거"로 읽었던 그 숫자가 실은 **배수를 써야 하는 근거**였다. 기본값은
> 이제 소스 대비 배수 `--obb_clear_src_ratio 0.3` 이다. 아래 D51 참조. 판정량도 `obb_clear`
> 에서 `obb_slack = obb_clear − m_j` 로 바뀌었다.

**어느 노드와 부딪혔나를 남긴다** (`obb_node` 열). "충돌했다"보다 "`person` 을 뚫었다"가 고칠
수 있는 진단이다. 최근접 노드 분포는 camel `stat_0`(fence) 306 / `dyn_0` 16 / `dyn_1` 11 /
`stat_1` 2 / `stat_2` 1, avocado `stat_4`(chair) 362 / `dyn_0`(person) 22 / `stat_0` 8.

뱅크별 OBB 관통 실측 (변이 최소 부호거리):

| 뱅크 | camel | avocado-slice |
|---|---|---|
| `hole_bank_holeonly` (hole 만) | 최소 −0.0258 u, <0 인 변이 21 | −0.0351 u, 22 |
| `hole_bank_clear0` (G1 만) | +0.0011 u, <0 0개 (단 <0.02 가 13) | **−0.0281 u, <0 20개** |
| `hole_bank_r90` (standoff 0.90) | +0.1836 u, 0 | +0.0755 u, 0 |
| `hole_bank` (standoff 0.80, 현재) | +0.1615 u, 0 | +0.0583 u, 0 |

**결론을 왜곡하지 않고 적는다: 이 두 씬에서 G5 는 한 번도 안 걸린다.** `clear0` 에서 OBB 를
0.02 u 안쪽까지 침범한 33개 변이(camel 13 + avocado 20)의 standoff 를 직접 재봤더니
**33개 전부** 현재 임계(camel 0.0680·S / avocado 0.1095·S) **아래**였다 — standoff 가 이미
전부 잡는다. 그래서 G5 를 켜도 336/392 변이의 손잡이가 하나도 안 바뀐다.

그럼 왜 넣나. ① 렌더 0회라 비용이 없다. ② standoff 가 잡는 건 **우연**이다 — standoff 는
"가까운 표면"을 재므로 물체가 크고 속이 빈 경우(방 안, 큰 가구) 안쪽 한가운데는 오히려 표면이
멀다. 이 두 씬은 물체가 작아서 그 구간이 안 나왔을 뿐이고, ACTIVE-3 의 51편에는 나온다고 봐야
한다. ③ `obb_node` 진단이 standoff 에는 없다. 즉 G5 는 **지금 고치는 게이트가 아니라 회귀
감시기**다. `--no_obb_gate` 로 판정만 끄고 열은 남길 수 있다.

기록해둔 선택지:
1. ~~`--min_obb_clear` 를 0.02 → 소스 대비 배수(예: 0.15×)로 바꾸기. 지금 안 한 이유는 소스
   여유가 씬 간 4배 차이라 camel 이 0.077 u 로 조여지는데, 그건 근거 없는 보수화다.~~
   → **채택됨 (D51)**, 그리고 여기 적은 "안 한 이유"가 틀렸다. 씬 간 4배 차이는 배수를 **써야**
   하는 이유다.
2. OBB 대신 노드 점군 convex hull. OBB 는 `extent_inflated` 노드에서 실제보다 크다
   (저시차 깊이축 보정, D-OBB 항목). 지금은 그 방향이 **안전한 쪽**이라 놔둔다.
3. G5 를 후보 pose 게이트(`lbm/gates.py evaluate`)에도 넣기. 지금은 뱅크 이분법에만 붙었다.
   `GATE_ORDER` 에는 `G5_obb` 자리를 이미 만들어뒀다.

---

### D50 standoff 는 **껐다** — 예산 3개 중 하나는 남는다

D48 의 `standoff` 예산(기본 ratio 0.80)과 D49 의 G5 OBB 예산은 **같은 걸 두 번 잰다**. D49 가
이미 실측으로 적어둔 그대로다: `clear0` 에서 OBB 를 침범한 33개 변이(camel 13 + avocado 20)의
standoff 가 **33개 전부** 임계 아래였다. 두 게이트가 붙어 있으면 항상 standoff 가 먼저 물고,
G5 는 영원히 `binding` 에 안 잡힌다.

그러면 "충돌 판정을 키운다"는 손잡이가 사실상 standoff ratio 하나뿐인데, 그게 **나쁜
손잡이**였다. D48 에 이미 증거가 있다 — ratio 0.80↔0.90 사이에서 camel `push_in_arc` 가
0.939 → 0.020 으로 불연속 붕괴한다. 소스 카메라 자신의 standoff 대비 배수라, 소스가 이미
자기 바닥에 붙어 있는 씬(camel 0.0849·S)에서는 배수를 조금만 올려도 전진 preset 이 통째로
정지한다. **"조금 키운다"가 물리적으로 불가능한 축**이다.

G5 는 반대다. 마진이 **거리에 절대값으로** 더해지므로 0.02 → 0.06 → 0.15 가 매끄럽게 조여진다
(D51 표 참조). 그래서 예산을 하나로 줄이고 **남길 하나로 G5 를 골랐다**.

| 설정 | camel path 합 | camel 접힌 단 | avocado path 합 | avocado 접힌 단 |
|---|---|---|---|---|
| standoff 0.80 + G5 0.02 (D49 상태) | 100.86 | 94 | — | 106 |
| standoff off + G5 0.02 | 12.84 (dyn_0 6 preset) | — | 6.18 | — |
| standoff off + G5 0.15 | 10.35 | — | 0.89 | 20 |
| standoff off + G5 β=0.3 (**채택**) | 10.35 | — | 6.17 | — |

(위 2~4행은 `dyn_0` anchor 스윕이라 D49 의 전량 합계와 **직접 비교하면 안 된다** — D48 이
남긴 "한 anchor 스윕으로 전량을 외삽하지 말 것" 교훈 그대로다.)

`--min_standoff_ratio` 기본값 **0.90 → 0.80 → 0.0**. 열(`standoff`)은 그대로 남아 있고
`--min_standoff_ratio 0.8` 로 언제든 되켤 수 있다. `near_depth`(D48 에서 기각) 와 같은 취급 —
**측정은 하되 판정에서 뺀다**.

---

### D51 OBB 마진은 씬마다 달라야 한다 — **소스 카메라 자신의 여유 대비 배수** (β=0.3)

사용자 지시: "대신 충돌 판정을 조금 키워줘". 절대 마진 4종(0.02 / 0.06 / 0.10 / 0.15)을 렌더해
보여줬더니 사용자 판정이 갈렸다 — **"camel 은 0.15 가 맞는데 avocado 는 0.06 정도가 적당한 것
같은데 왜이럼?"**. 이 항목은 그 "왜"의 답이다.

**`u` 는 씬 스케일만 정규화한다.** camel S=4.6713 / avocado S=4.6471 로 **0.5% 차이**라
`u` 는 두 씬에서 사실상 같은 자다. 그런데 그 자로 잰 것들은 안 같다:

| 양 | camel | avocado-slice | 비 |
|---|---|---|---|
| S (씬 전역 스케일) | 4.6713 | 4.6471 | **1.005×** |
| 소스 카메라 자신의 최소 OBB 여유 | +0.5107 (`stat_0` fence) | +0.1165 (`stat_4` chair) | **4.4×** |
| binding 노드 `max(extent)` | 0.137 (`dyn_0` camel) | 0.376 (`stat_0` table) | 2.7× |
| 최근접 노드 `d_ref` | 0.639 | 0.191 | 3.3× |

즉 `u` 로 나눠도 **물체 크기·카메라-물체 거리·소스 여유는 하나도 안 맞춰진다**. 절대 마진
하나가 두 씬을 못 덮는 게 당연하다.

**절대 마진의 병리는 "균일 팽창"이다.** 마진 `m` 은 모든 박스를 축마다 `2m` 씩 부풀린다.
`m=0.15` 는 camel 낙타(0.137×0.060×0.122)의 얇은 축을 **6배**로, avocado 의자 조각
(0.056×0.005×0.022)의 얇은 축을 **61배**로 키운다. 두 유령 박스가 거의 같은 물리적 크기로
수렴한다 — 물체가 무엇이든 금지구역 크기가 같아진다는 뜻이고, 그래서 작은 물체가 많은 씬
(avocado) 이 먼저 질식한다.

#### 틀린 가설을 하나 세웠고, 실측으로 기각했다

사용자가 눈으로 고른 값을 각 씬 **지배 노드**의 `max(extent)` 로 나누면 `0.15/0.137 = 1.09` 와
`0.06/0.056 = 1.07` — **2% 안에 겹친다**. "물체 지름 하나만큼 떨어져라"라는 규칙처럼 보였다
(`d_ref` 로 나누면 0.235/0.313, 소스 여유로 나누면 0.294/0.515 로 안 맞았다). 그래서
`m_j = clip(ratio·max_ext_j, floor, cap)` 를 구현하고 돌렸다:

| 설정 | camel path | avocado path |
|---|---|---|
| 절대 0.06 | 12.15 | 2.25 |
| 절대 0.15 | 10.35 | 0.89 |
| 비례 ratio=1.0 cap 0.12 | 10.66 ✅ | **1.19** ❌ (절대 0.06 보다 나쁨) |

**기각.** 원인: avocado 에서 실제로 binding 하는 노드는 내가 적합에 쓴 `stat_4`(의자, 최근접)가
아니라 **`stat_0`(테이블, 0.376×0.354×0.024)** 이다. 판때기라 `max_ext` 가 **두께가 아니라
너비**를 집어 어느 ratio 에서든 cap 에 붙어버리고, ratio 가 아무 일도 안 한다.
`min_ext` 로 바꿔보니 `stat_0` 은 1% 로 맞는 대신(`camel dyn_0` 2.487 / `avocado stat_0` 2.461)
`stat_4` 가 **12.115** 로 튄다. **어느 크기 척도를 써도 binding 노드 셋 중 둘만 맞고, 그 둘이
척도마다 바뀐다** — 2% 일치는 우연이었다. 크기 비례 계열 전체를 버렸다.

#### 맞는 규칙: 소스 카메라 자신의 여유 대비 배수

`m = β × (소스 카메라 자신의 최소 OBB 거리)`. `--obb_clear_src_ratio β` (기본 **0.3**).
D47 이 standoff 에서 쓴 것과 **같은 꼴**이고, 같은 이유로 옳다: **β<1 이면 소스 카메라가
정의상 통과한다.** 절대 마진이 계속 원본 촬영을 기각하던 사고(아래)를 구조적으로 막는다.

가장 강한 증거: **아무것도 안 알려줬는데 β=0.3 이 camel 마진 0.1532 를 내놓는다** — 사용자가
영상 보고 직접 고른 0.15 를 `path` 합 10.35 / `obb` binding 14 까지 그대로 재현한다.
avocado 는 0.0350 으로 사용자가 고른 0.06 보다 **느슨해서 움직임이 더 산다** (path 2.25 → 6.17).

전량 스윕 (원본 수치 그대로, `dyn_0` anchor · 6 preset · path 합 / hole-only 변이 수 /
binding obb-collision-hole):

| config | camel path / zero / obb-coll-hole | avocado path / zero / obb-coll-hole |
|---|---|---|
| 절대 0.02 | 12.84 / 0 / 0-12-12 | 6.18 / 7 / 0-11-13 |
| 절대 0.06 | 12.15 / 0 / 3-10-11 | 2.25 / 14 / 8-0-16 |
| 절대 0.10 | 11.82 / 0 / 7-7-10 | 0.89 / 20 / 8-0-16 |
| 절대 0.15 | 10.35 / 0 / 14-0-10 | 0.89 / 20 / 8-0-16 |
| 비례 0.6 cap .12 | 12.15 / 0 / 3-10-11 | 1.26 / 18 / 8-0-16 |
| 비례 1.0 cap .12 | 10.66 / 0 / 14-0-10 | 1.19 / 18 / 8-0-16 |
| 비례 1.4 cap .12 | 10.66 / 0 / 14-0-10 | 0.89 / 20 / 8-0-16 |
| **소스배수 β=0.3 (채택)** | **m 0.153 → 10.35 / 0 / 14-0-10** | **m 0.035 → 6.17 / 7 / 1-10-13** |
| 소스배수 β=0.5 | m 0.255 → 7.34 / 0 / 14-0-10 | m 0.058 → 2.39 / 14 / 8-0-16 |

사용자가 3분면 영상(`D51_obb_srcratio.mp4`, 수동 선택 / β=0.3 / β=0.5)을 보고 **β=0.3** 을
골랐다.

#### 소스 카메라 적법성 검사 — 이번에도 버그를 잡았다

D47(standoff 절대 0.15·S) / D49(마진 0.15) 에 이어 **세 번째**다. 규칙: **소스 카메라 자신이
통과 못 하는 임계는 충돌 판정이 아니라 버그다.** 원본 촬영을 기각하고 있다는 뜻이니까.

- `cap 0.20` → avocado 소스 카메라 자신의 slack **−0.036** (`stat_0` table, 거리 0.164). 기각.
  → `--obb_clear_cap` 기본값을 **0.12** 로. 여기서 소스 slack 은 camel +0.391 / avocado +0.044.
- 절대 마진 0.15 → avocado 소스 여유가 +0.1165 라 **소스가 탈락**. β 모드가 이걸 구조적으로 막는다.

`source_obb_clear()` 가 `(slack, node_id, raw_dist)` 3-tuple 을 돌려주고 게이트 줄에 매번 찍는다.

#### 구현 (기존 동작 보존 확인 포함)

- `lbm/gates.py: node_margins(nodes, ratio, floor, cap)` → `m_j = max(floor, min(ratio·size, cap))`.
  `cap` 은 **비례 항에만** 건다 — `clip` 으로 짜면 camel 의 floor 0.153 이 cap 0.12 에 잘린다.
  `floor ≤ cap` 구간에서는 `clip` 과 동일하므로 아래 회귀는 그대로 유효하다.
- `obb_clearance(..., margins=)` 가 `(dists, ids, slacks)` 3-tuple. **노드 선택이 거리 최소가
  아니라 slack 최소로** 바뀐다 — 작고 가까운 노드가 큰 마진을 요구할 때 binding 노드가 딴 데로
  잡히는 걸 막는다. 마진이 전 노드 상수면 argmin 이 같아 기존과 완전히 동일.
- 뱅크 CSV 에 `obb_slack` 열 추가 (`obb_clear` 와 `obb_node` 사이). 판정량은 slack 이고
  임계는 **항상 0**, 끄는 값은 `-inf` (0 은 "박스 표면+마진까지 허용"이라 켠 상태다).
- `bank.json.obb_gate` 에 노드별 `node_margins` 표를 통째로 남긴다 (임계 하나로는 재현 불가).
- **회귀**: `--obb_clear_ratio 0 --min_obb_clear 0.06` vs 기존 절대 마진 뱅크,
  `knob,status,binding,path_len_u,hole_fraction,obb_clear,obb_node,behind_frac,standoff`
  **24 비교 / 불일치 0**, `|obb_slack − (obb_clear − 0.06)|` max **0.000000**.
  `ratio=0` + floor 만 주면 D49 절대 마진과 **비트 동일**하다.

#### 이 과정에서 나온, 더 중요할 수도 있는 발견

**작동하는 모든 마진에서 G5 는 거의 전부 anchor 노드 자신에 binding 한다** — camel `dyn_0`
14/14, avocado `stat_0` 6~8/8. 즉 지금 G5 는 "장애물 회피"가 아니라 **"자기 subject 에 너무
가까이 가지 마라"라는 구도 제약**으로 작동하고 있고, `push_in` 계열 preset 과 정면으로 싸운다.
(마진별 binding 노드 분포: 절대 0.02 → 양 씬 모두 0건 · 0.06 → camel `dyn_1`×3, avocado
`stat_0`×4+`stat_4`×4 · 0.10 → camel `dyn_0`×4+`dyn_1`×3, avocado `stat_4`×8 · 0.15 → camel
`dyn_0`×14, avocado `stat_4`×8.)

**선택지로 기록** (사용자에게 옵션 4로 제시했으나 선택하지 않음):
1. **anchor 노드를 OBB 게이트에서 제외**하고 대신 `subject_area` 상한(D45)이 근접을 막게 하기.
   그러면 G5 가 순수 장애물 게이트가 되고 `push_in` 이 살아난다. 위험: 카메라가 subject 를
   물리적으로 관통해도 면적 상한만 통과하면 지나간다.
2. anchor 노드만 별도 마진(예: β_anchor = 0.1)을 주기. 1번의 부드러운 판.
3. β 를 씬이 아니라 **preset 계열별로** 달리 주기 (전진 계열만 완화).
4. cap 0.12 를 소스 여유 대비 배수로 바꾸기 (지금은 절대값이라 씬 간 이식성이 β 보다 낮다).

---

### D52 뱅크는 **태그 N개짜리 canonical 하나**로 나간다 — 새 포맷을 안 만든다

`decode/emit.py` 는 결정 하나를 canonical 로 바꾼다. 그런데 우리가 실제로 만든 건 뱅크다
(camel 336 / avocado 392). 사용자 목표가 "소스 카메라 재현"이 아니라 **카메라 움직임
augmentation** 이므로 하류로 넘어갈 단위도 뱅크 전체여야 하는데, 그 사이가 비어 있었다.

**이을 자리는 이미 나 있었다.** `canonical.json` 의 `cameras` 는 처음부터 **태그 → 궤적 dict**
이고 `emit_model_cams.py:394,423` 은 `--cameras all` 로 전 태그를 돌며 `rel_c2w` 만 읽는다.
그래서 `fit/bank/emit_bank.py` 는 새 스키마를 정의하지 않는다 — `build_canonical()` 을 변이마다
부르고 `cameras` 만 합친다. canonical 규약(`rel[0]=I`, 단위 스케일, 21 index pick)의 구현은
계속 `decode/emit.py` 한 곳이다. 태그는 `variant_id` 그대로 써서 하류 결과 파일명만으로
"어느 anchor 의 어느 preset 이 어느 hole 단에서 깨졌나"를 되짚을 수 있게 한다.

**왜 poses.npz 를 그냥 안 쓰나**: 거기엔 `cam_c2w` 뿐이고 `build_canonical` 은
`look_at`/`subject_track`/`tau`/`info` 를 요구한다. 그래서 `build_poses` 를 다시 돌린다
(렌더 0회, 336 변이 10초).

**다시 푸는 이상 재현을 증명해야 한다.** 재구성 궤적을 `poses.npz` 와 프레임 단위 대조하고
어긋나면 멈춘다 (`--pose_tol` 1e-9, `--strict` 기본 on). 이 가드가 **바로 버그를 잡았다** —
`min_sweep_deg` fallback 을 15.0 대신 30.0 으로 적어 `dyn_1` 궤적이 최대 1.575e-2 u 어긋났다.
`path_len` 자체가 0.0116 이니 궤적이 통째로 다른 건데, 표에 찍히는 건 `tau_max` 0.015 → 0.0176
뿐이라 **육안으로는 정상으로 보인다**. 근본 수정은 `fit_hole_ladder.SHAPE_DEFAULTS` 하나로
CLI 기본값과 fallback 의 출처를 합친 것 — 상수를 두 군데 적은 게 원인이었다. 새 뱅크는
`fixed` 에 `aim_ramp_frames`/`orbit_span_frac`/`min_sweep_deg`/`num_frames` 를 같이 싣는다.

검증: camel 336 전량 **pose 재현 최대오차 0.000e+00**, 21↔49 왕복 0.000e+00,
`emit_model_cams.py --model sierpinskicam` 왕복 통과.

**거르지 않는다** (D39/D45 와 같은 규칙 — 뱅크는 재고 목록이고 거르는 건 소비자 몫). 대신 센다:
camel 336 중 **접힌 단 94** (충돌 천장이 hole 0.10 단보다 낮아 4단이 같은 궤적 → `folded_onto`
로 대표 태그를 가리킨다) / **`translation_degenerate` 68** (이동 0 = pan 계열, `--scales` 도
ReRoPE 의 `|t|` 정규화도 의미가 없다) / `g = rmax/S` [0.00000, 2.28180] median 0.03875.
빼려면 `--drop_folded` / `--min_path_len` / `--status` / `--anchors` / `--presets`.

**기록해둔 선택지**:
1. 접힌 단 94 를 기본으로 뺄지. 지금 남기는 이유는 "같은 궤적이 4단에 걸쳐 있다"가 그 자체로
   진단이기 때문 — 빼면 사다리가 어디서 접혔는지가 canonical 에서 안 보인다.
2. `translation_degenerate` 68 을 별도 canonical 로 분리할지. 하류 모델 중 상수 offset 을
   무시하는 것들(ReCamMaster)에게는 이 68개가 통째로 no-op 이다.
3. 영상 여러 편의 canonical 을 하나로 합칠지 (`--tag_prefix` 로 이미 가능). ACTIVE-3 의 51편이
   들어온 뒤에 결정.

---

### D53 avocado 뱅크 절반이 **조용히 정지 카메라**다 — τ 하한이 소스 시차보다 낮아서 (수정됨 2026-08-22)

avocado-slice 뱅크를 canonical 로 뽑았더니 392 중 **`translation_degenerate` 204**, `g` median
**0.00000** 이었다. camel 은 336 중 68 (median 0.09372). pan 계열(이동 0 이 정의)을 빼면
avocado 는 **148 개가 "움직이라고 시켰는데 안 움직이는" 궤적**이다. 전부 `binding=hole`,
147/148 이 `status=clamped_low`.

**원인은 충돌이 아니라 τ 의 바닥이다.** `lbm/presets.py:163-168`:

```python
tau0, _ = tau_of(np.stack([np.eye(4)] * len(rel_local)), c2w_start, src_centers, z_med)
if tau0 >= target_tau - 1e-6:
    # 시작 pose 만으로 이미 예산을 다 쓴 경우. 움직임을 0 으로 둔다 (예산을 넘기지 않는다).
    return (np.stack([np.eye(4)] * len(rel_local)), {"scale": 0.0, ...})
```

`tau0` 는 **카메라를 시작 pose 에 얼려놓았을 때의 τ** 다. τ 는 plan 과 **소스** 의 프레임별
간격이므로, 소스 카메라가 움직이면 정지 plan 도 그만큼 τ 를 쌓는다. 그래서 `tau_start` 는
anchor 와 무관한 **씬 상수**다 — 실측: avocado 전 anchor **0.1286**, camel 전 anchor **0.0042**
(각 씬의 `parallax_ratio` 0.129 / 0.0046 과 일치). `KNOB_RANGE["tau"]` 하한은 **0.02** 라
avocado 에서는 **하한 자체가 도달 불가능**하다. 사다리가 hole 을 못 맞춰 손잡이를 내리면
0.02 < 0.1286 에 걸려 위 분기로 들어가고, 궤적이 통째로 사라진다.

**그런데 궤적을 0 으로 만들어도 예산은 안 지켜진다** — 이게 이 분기의 진짜 문제다.
anchor 별 "정지 상태의 hole" 실측 (avocado):

| anchor | tau_start | hole@정지 | 정지가 된 변이 | 도달한 최대 path |
|---|---|---|---|---|
| stat_5 | 0.1286 | 0.2190 | 13/56 | 1.4537 |
| stat_0 | 0.1286 | 0.2616 | 13/56 | 1.3725 |
| stat_3 | 0.1286 | 0.3350 | 15/56 | 1.6782 |
| dyn_0 | 0.1286 | 0.3559 | 20/56 | 1.2895 |
| stat_2 | 0.1286 | 0.3677 | 23/56 | 1.6753 |
| stat_1 | 0.1286 | **0.5895** | 32/56 | 1.2859 |
| stat_4 | 0.1286 | **0.6384** | 32/56 | 1.2938 |

`stat_1__pull_out_arc` 는 **4단 전부** path 0.0000 / 측정 hole 0.5895 다. hole0.5 단조차
0.5895 > 0.5 로 **못 맞췄다**. 즉 궤적을 버려서 얻은 게 없다 — 단을 못 맞추는 건 똑같고
궤적만 사라졌다. anchor 의 정지 hole 이 이미 모든 단보다 높으면 그 anchor 는 **어떤 손잡이
값으로도 사다리를 만들 수 없다**. 지금 코드는 그걸 `unreached` 가 아니라 `clamped_low` 로
찍어서, 표에서 "작지만 정상인 카메라"와 **구분이 안 된다**.

camel 에서는 같은 병이 `stat_3` 하나에만 났다 (hole@정지 0.6482, 20/56 정지). camel 의
`tau_start` 0.0042 가 사다리 하한 0.02 아래라 나머지 anchor 는 안 걸린 것이다. **저시차 씬에서
안 보이다가 고시차 씬에서 절반을 먹는 종류의 결함**이다.

영상: `out/avocado-slice/hole_bank/ladder_collapse.mp4` (dyn_0/stat_1 × pull_out_arc 4단),
`out/camel/hole_bank/ladder_healthy.mp4` (camel dyn_0 orbit_left_arc 4단, 정상 대조군).

**처음엔 안 고쳤다** (사용자: "일단 이건 돌려놓고 나중에 수정하면 되니까"). 뱅크는 그대로 두고
`translation_degenerate` 를 canonical 의 카메라별 meta 에 실어 하류가 거를 수 있게만 해뒀다
(`emit_bank.py:217`, `--min_path_len` 로 즉시 제외 가능).

**기록해둔 선택지** (고칠 때 고를 것):
1. **(권장 → 채택) hole 사다리를 anchor 의 정지 hole 대비 초과분으로 정의**한다. `hole@정지` 를
   먼저 재고 단을 `hole@정지 + Δ` 로 잡으면 anchor 마다 실제로 도달 가능한 사다리가 나온다.
   부수 효과로 "이 anchor 는 정지만으로 이미 hole 0.59" 가 표에 드러난다.
2. τ 손잡이를 **궤적 자체의 크기**로 바꾼다 (지금은 plan-vs-소스 절대 τ). `tau0` 를 빼고
   `τ_traj = max|p(f) − p(0)| / z_med` 를 손잡이로 쓰면 소스 움직임과 분리된다. 사다리 하한
   0.02 가 다시 의미를 갖는다. 다만 τ 의 원래 뜻(하류 모델이 감당할 시점 변화량)에서 멀어진다.
   → **기각**. τ 는 D46 부터 "하류 모델이 감당할 시점 변화량"으로 쓰고 있고, 뜻을 바꾸면 뱅크
   사이 τ 값이 비교 불가능해진다.
3. `tau0 >= target_tau` 분기에서 **0 대신 최소 유효 궤적**을 돌려주고 `status=tau_floor` 로
   찍는다. 가장 작은 변경이고 "정지가 이득이 없다"는 실측에 직접 대응하지만, 예산 초과를
   명시적으로 허용하는 것이라 D46 의 예산 개념과 충돌한다.
   → **변형해서 채택**. 예산을 넘기는 대신 **손잡이 하한 자체**를 `tau_start` 위로 올렸다
   (아래 ①). 그러면 이분법이 애초에 `target_tau < tau_start` 를 물어보지 않으므로 저 분기에
   도달하지 않는다 — 예산은 그대로 지켜지고 status 이름(`tau_floor`)만 가져온다.
4. `hole@정지` 가 최저 단보다 높은 anchor 는 **뱅크에서 통째로 뺀다** (avocado 는 stat_1/stat_4,
   camel 은 stat_3). 가장 단순하지만 D39/D45 의 "거르지 않는다" 규칙과 어긋난다.
   → **기각** (같은 이유).

---

#### 수정 (2026-08-22) — 옵션 1 + 옵션 3 변형, 그리고 숨어 있던 세 번째 결함

세 군데를 고쳤다. 전부 `--flag`/`--no_flag` 쌍이고 **둘 다 끄면 예전과 같이 돈다**.

**① `--tau_floor_src` (기본 켬)** — τ 손잡이 하한을 `KNOB_RANGE["tau"][0]` 고정에서
`tau_start + KNOB_RANGE["tau"][0]` 로 바꾼다. 원래의 0.02 가 "절대 τ"에서 **소스 자신의 시차 위에
얹은 여유**로 다시 읽힌다: camel 0.02 → 0.0242 (사실상 그대로), avocado 0.02 → **0.1486** (도달
가능해진다). D47/D51/D55 의 "임계는 소스 대비"와 같은 꼴이다. bracket 점 중 새 하한보다 낮은
것은 **재현 불가능**하므로 `solve_knob` 이 같이 버린다 — 안 버리면 이분법이 도달 못 하는 구간을
훑어 답이 다시 하한으로 내려앉는다. 하한에 닿으면 status 에 `+tau_floor` 를 붙인다.

**②`--hole_mode excess` (기본)** — 단을 `hole_static + Δ` 로 잡는다. anchor·preset 마다 손잡이 0
으로 한 번 재고(`hole_static`, 렌더 1회), 같은 호출에서 `tau_start` 도 같이 받는다. `variant_id`
와 `hole_delta` 는 **Δ** 로 매긴다 — 씬이 달라도 같은 이름의 단이 같은 뜻이어야 하고,
`absolute` 모드에서는 이름이 예전과 동일해진다.

**①-b `presets.fit_tau(refine_zero=True)`** — ① 만으로는 **여전히 정지였다**. `fit_tau` 이분법은
`lo=0` 에서 시작하고 눈금이 `max_scale/2**iterations` = 4/256 = **0.0156** 이라, 하한을
`tau_start` 바로 위로 올리면 그걸 맞추는 배율이 첫 눈금보다 작아져 `lo` 가 0 에 남는다:

| avocado `stat_1 pull_out_arc` | knob | tau_max | path_u | status |
|---|---|---|---|---|
| 예전 (하한 0.02) | 0.020 | 0.1286 | **0.000** | `clamped_low` |
| ① 만 | 0.149 | 0.1286 | **0.000** | `clamped_low+tau_floor` |
| ① + ①-b | 0.149 | 0.1486 | **0.012** | `clamped_low+tau_floor` |

같은 목표에서 `dyn_0` 는 반경이 작아 첫 눈금이 우연히 맞아 0.010 이 나왔다 — anchor 에 따라
갈리는 **해상도** 문제지 물리가 아니다. `refine_zero` 는 `lo==0` 이면 `[0, hi]` 에서 이분법을 한 번
더 돌린다 (`tau_of` 는 렌더가 아니라 numpy 라 비용 0). 플래그는 CLI 가 아니라 decision 의
`trajectory.tau_refine` 에 실려 `emit_bank` 재현까지 따라간다 — 밖에 두면 되풀 때 배율이 달라져
pose 대조 assert 가 터진다.

**측정 (뱅크 전량 재생성, G6 켠 상태 그대로)**

| | camel pre → post | avocado pre → post |
|---|---|---|
| 변이 | 336 → 336 | 392 → 392 |
| 정지 카메라 (`path_len_u` < 1e-6, 전량) | 68 → 48 | 206 → 56 |
| 　└ 그중 `pan_*` (이동 0 이 정의) | 48 → 48 | 56 → 56 |
| 　└ **그 외 = 진짜 고장** | **20 → 0** | **150 → 0** |
| 비-pan `path_len_u` median | 0.1110 → **0.4636** | 0.0594 → **0.4341** |
| 비-pan `path_len_u` mean | 0.2853 → 0.6595 | 0.2398 → 0.6644 |
| `hole_fraction` median (전량) | 0.3063 → 0.3199 | 0.3482 → 0.3661 |
| status `clamped_low` | 91 → **0** | 149 → **0** |
| status `unreached` | 0 → 14 | 1 → 33 |
| status `shape_limited` | 19 → 42 | 9 → 47 |
| status `solved` | 166 → 168 | 182 → 189 |
| 남은 위반 (G1 / standoff / obb / 지면아래) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

**정지 카메라 행을 세 줄로 쪼갠 이유**: 전량 카운트만 보면 48·56 이 남아 "덜 고쳐졌다"로 읽히는데,
남은 건 **정확히 `pan_left`/`pan_right`** 다 (6·7 anchor × 2 preset × 4 단 = 48·56, 순수 회전이라
이동 0 이 맞는 값). 비-pan 은 **전멸**했다. 수정 전 비-pan 정지의 preset 분포는 camel
`straight_ease`/`push_in_arc`/`pull_out_arc`/`rise_reveal`/`drop_reveal` 각 4, avocado 는 8 개
preset 에 16~21 개씩 — **한 preset 의 버그가 아니라 손잡이 하한의 버그**라는 증거다.

`clamped_low` 91·149 개가 통째로 사라지고 그 자리를 `unreached`(윗단이 이 anchor 에선 도달
불가)와 `shape_limited`(모양이 천장)가 채웠다 — **"작아서 못 갔다"가 "무엇이 막았다"로 바뀐 것**이
이 수정의 요점이다. hole median 이 거의 안 움직인 건 예산이 그대로라는 뜻이고, G6 게이트 위반도
0 을 유지했다. `binding` 도 이제 실제 게이트 이름만 나온다 (avocado: hole 189 / none 80 /
collision 54 / elev 36 / ground 27 / obb 6).

⚠ **위 숫자는 CSV 버그를 고친 뒤의 2차 재생성 값**이다. 1차와 비교하면 avocado 의
`path_len_u` mean 0.5716 → 0.5694(전량), `hole_fraction` median 0.3744 → 0.3661 로 갈리는데,
knob 이분법이 **렌더된 hole** 로 분기하므로 렌더러 비결정성이 분기 하나를 뒤집으면 배율이
바뀐다. 판정 열(status/binding 분포)은 두 번 다 동일했다.

**예전 동작 확인** — `--hole_mode absolute --no_tau_floor_src` 로 프로덕션과 같은 설정(사다리
0.1/0.2/0.35/0.5, verify 13)을 avocado `stat_1`/`dyn_0` × `pull_out_arc`/`truck_left` 에 걸어
16 행 × 42 공통열을 대조: 판정 열(`knob`/`status`/`binding`/`hole_fraction`/`tau_max`/
`path_len_u`/`obb_slack`/`elev_abs_max`/`ground_clear`)은 **전부 일치**. 18 칸이 4번째 소수에서
갈렸는데 전부 `subject_area_med`/`near_depth` 였고, **같은 명령을 두 번 돌려도 같은 두 열에서
16 칸이 갈린다** — 렌더러 비결정성이지 이 수정 탓이 아니다.

**부수 버그 (같은 날 발견·수정)** — `status` 접미사를 처음에 `f"{status},tau_floor"` 로 붙였더니
`bank.csv` (따옴표 없이 `",".join`) 에서 **열이 한 칸씩 밀려** avocado 10 행의 `binding` 이
`tau_floor` 로 읽혔다. 구분자를 `+` 로 바꾸고, CSV writer 에 "값에 쉼표 금지" assert 를 넣었다.

영상: `out/<video>/hole_bank/d53_before_after.mp4` (위=pre, 아래=post; avocado `stat_1`/`stat_4`,
camel `stat_3` × `pull_out_arc`/`straight_ease`, 단 Δ0.35).

---

### D54 OBB 가림 판정은 **렌더 판정을 대체 못 한다** — 가설된 원인이 실측 1.4~3.0% 였다

사용자 질문: *"구멍이 뚫리면 가렸는지 판단하기 힘드니 ... 물체 bbox 기준으로 가려지는지를 기준으로
판단하는 것도 실효성 있는지 봐줘."* 가설된 기전은 명확하다 — 렌더 기반 `occlusion_pass` 는 점군
두 장(subject 만 / 전체)의 깊이를 비교하는데, **가리는 물체가 재구성이 안 됐으면 거기가 구멍**이라
subject 가 그대로 비쳐서 "안 가려짐"으로 읽힌다. 생성 모델은 그 물체를 그릴 텐데도.

`eval/audit_bank_geometry.py --obb_occlusion` 을 만들어 재봤다 (렌더 0회 — 노드 OBB 를 투영해
convex hull 로 painter's algorithm, 노드별 꼭짓점 z 중앙값으로 정렬). 두 측정의 불일치:

| | camel 336 | avocado-slice 392 |
|---|---|---|
| `occlusion_pass` median / <0.4 | 0.919 / 14 | 0.868 / 60 |
| `obb_occl_pass` median / <0.4 | 0.839 / 23 | 0.871 / 51 |
| 둘 다 통과 (thr 0.4) | 305 | 290 |
| 둘 다 실패 | 6 | 9 |
| **렌더만 통과** (= 가설이 예측하는 자리) | 17 | 42 |
| **OBB 만 통과** | 8 | 51 |
| corr(render, obb) | 0.276 | 0.392 |

불일치가 **양방향으로 비슷하게** 난다. 그래서 `--obb_occlusion_diag` 로 결정적 진단을 하나 더
쟀다: OBB 가 "가려졌다"고 한 픽셀들이 렌더에서 실제로 **구멍**인 비율 `hole_at_occluder`.

| 그룹 | camel n / hole@occ med | avocado n / hole@occ med |
|---|---|---|
| 둘 다 통과 | 156 / 0.019 | 196 / 0.001 |
| 둘 다 실패 | 6 / 0.015 | 9 / 0.015 |
| **렌더만 통과 (OBB 가 가림이라 함)** | 17 / **0.014** | 42 / **0.030** |
| OBB 만 통과 | 3 / 0.000 | 51 / 0.000 |

**가설된 기전은 이 두 씬에서 실측 1.4~3.0% 다.** OBB 가 가렸다고 한 자리의 97~99% 는 렌더에
멀쩡히 점이 있다 — 즉 그 가리개는 이미 재구성돼 있었고 렌더 판정이 이미 세고 있었다. hole 과의
상관도 없다: 불일치 그룹 hole 평균이 camel 0.322 vs 나머지 0.323, avocado 0.316 vs 0.363
(불일치 쪽이 오히려 **낮다**).

불일치의 진짜 원인은 두 측정의 **각자 다른 편향**이다.
- **렌더만 통과 42 중 40 이 avocado `stat_2`(window, d_ref 3.17) 를 `dyn_0`(person,
  d_ref 0.55) 가 가린다고 한 경우다.** 사람 OBB 는 카메라 가까이 있는 큰 덩어리라 투영 hull 이
  멀리 있는 창문 hull 을 통째로 덮는다 — 실루엣이 아니라 **박스라서** 생기는 과대보고.
- **OBB 만 통과**는 반대다. camel 3건은 `obb_occluder` 가 아예 비어 있는데(OBB 가림 0.000)
  렌더는 0.30 을 준다 — **노드가 아닌 것**(바닥·벽·박스 없는 표면)이 가리는 경우로, OBB 는
  구조적으로 못 본다. avocado 는 `stat_0`(table, extent [0.38,0.35,**0.02**]) 가 37건으로 1위:
  2 cm 두께 박스가 실제 상판이 가리는 양을 과소 표현한다.

**결론: 대체재가 아니라 괄호다.** 렌더는 과소(미재구성 가리개), OBB 는 과대(박스≠실루엣) 보고하고,
둘 다 못 보는 게 남는다 — 노드 아닌 기하(벽·바닥)와 drop 된 track(avocado `knife`
`max_area_frac` 0.0013, `avocado` 0.00132/0.00115; camel 1건). 다만 drop 된 track 은 전부
화면의 0.2% 미만이라 가림 원인으로는 무시할 수준이다.

**결정: 두 열을 `geometry.csv` 에 나란히 남기고 게이트로는 안 쓴다** (D39/D45 "뱅크는 재고,
거르는 건 소비자 몫"과 같은 규칙). 불일치 118건(camel 25 / avocado 93)은 뱅크를 소비할 때
사람이 볼 목록이다. 영상: `out/{camel,avocado-slice}/hole_bank/occl_disagree.mp4`
(좌 3 = 렌더만 통과, 우 3 = OBB 만 통과).

**기록해둔 선택지** (필요해지면):
1. OBB 대신 **인스턴스 마스크를 직접 투영**한다 — 박스≠실루엣 편향이 사라진다. 다만 마스크는
   관측 프레임의 2D 라 새 시점으로 옮기려면 결국 그 점군(=렌더)으로 돌아간다. 실익이 불분명.
2. 두 측정의 **min 을 취해 보수적으로** 판정. 지금 데이터로는 camel 31 / avocado 102 가
   탈락하는데 그중 대부분이 OBB 과대보고라 멀쩡한 카메라를 버린다.
3. `hole_at_occluder > 0.3` 인 경우에만 OBB 판정을 채택한다 — 가설된 기전이 실제로 성립하는
   자리만 고른다. 이 데이터에선 해당 변이가 거의 없어 no-op 이지만, **재구성이 더 나쁜 씬**
   (hole 0.6+ 인 DL3DV 계열)에서는 켤 값어치가 있을 수 있다. 재볼 것.
4. 얇은 박스(`extent` 최소축 < 0.03 u: avocado `stat_0` 0.02, `stat_4` 0.00, `stat_7` 0.01)를
   두께 하한으로 부풀린다. `stat_4 chair` 는 두께 0.00 인 가짜 박스인데 avocado 의 소스 OBB
   여유(+0.1165 u)를 정하고 있다 — D51 과 얽혀 있어 같이 봐야 한다.

---

### D55 `rise`/`drop` 이 물체 **바로 위/아래**까지 간다 — 충돌 예산 셋이 전부 수평이라서 (G6)

사용자: *"rise, drop은 너무 많이 움직여 움직임 자체에 제한 걸어줘 물체의 바로 위나 아래까지 가면
안됨. 그리고 drop은 바닥이나 물체에 가려질 확률이 더 높아 가려짐으로 커버 안됨?"*

**전제가 실측으로 맞다.** 뱅크의 고도각(subject OBB 중심 대비, `lbm/gates.py:elevation_profile`):

| preset | camel p95 / p05 | avocado-slice p95 / p05 |
|---|---|---|
| `rise_reveal` | **89.83°** | **85.27°** |
| `drop_reveal` | −83.99° | −84.58° |
| `pedestal_up` | 40.9° | **77.63°** |

crane 만의 문제가 아니다 (`pedestal_up` 도 77.63°). 최악은 `stat_0__rise_reveal__hole0.35` 의
**89.960°** — 문자 그대로 물체 정수리다. 그런데 이 변이는 `occl 0.950`, `hole 0.321`,
`behind_frac 0.0000`, `obb_slack +0.35` 로 **기존 예산을 전부 통과**한다.

**기전.** 수직 이동은 hole 을 거의 안 늘린다 — 위로 가면 바닥이, 아래로 가면 천장·벽이 점군에
있어서 화면이 채워진다. 그래서 이분법의 shape 배증(`--shape_headroom 2 × --shape_doublings 4`
= 최대 16배)이 한 번도 천장에 안 닿고 끝까지 돌아 `lateral_frac 0.35 × 16 = 5.6` →
`atan(5.6) = 80°`. hole 예산으로는 수직을 못 막는다는 뜻이다.

**아래쪽은 지면을 뚫는다.** 지면 아래로 내려간 변이가 camel 14 / avocado 5. 최악은
`dyn_0__drop_reveal__hole0.35`(camel)로 게이트 없이 재현하면 지면 **−1.282 u**, 46/49 프레임이
바닥 밑이다. **G1 이 이걸 못 잡는 건 임계가 아니라 구조다** — G1 은 카메라 중심을 소스 13프레임에
투영해 `z_cam > depth(u,v)` 를 보는데, 바닥 밑 카메라는 소스 뷰에 **화면 밖**으로 투영돼 채점
대상이 아예 안 된다. 실제로 그 변이들의 `behind_frac` 이 전부 **정확히 0.0000** 이다.

**질문 (b) 답: 가림으로 커버 안 된다.** 세 가지 이유가 각각 독립이다.
1. **OBB 가림은 구조적으로 못 본다** — 바닥은 scene graph 노드가 아니다. avocado
   `stat_0__drop_reveal__hole0.5` 는 바닥 아래 0.66 u, 고도각 −85.98° 인데 `obb_occl_pass`
   **0.984**. camel `dyn_0__drop_reveal__hole0.2` 는 41/49 프레임이 지면 아래인데 **1.000**.
2. **렌더 가림은 부호가 반대다** — 바닥을 밑에서 보면 점이 없어 **구멍**이 나고, 구멍은
   "가려짐"이 아니라 "보임"으로 세어진다 (D54 의 과소보고 기전인데, 여기선 D54 가 실측한
   1.4~3.0% 가 아니라 지배적이다). avocado drop 의 `occlusion_pass` 는 0.527/0.693/0.739/
   0.893/0.932 로 대부분 통과한다. camel 은 0.243~0.592 로 걸리긴 하는데 **씬 의존**이다.
3. **애초에 가림 문제가 아니다.** 지면 아래는 카메라 **배치 적법성**(G1 의 일)이고, 가림 지표는
   subject 가 보이냐를 잰다. 다른 축이다.

**결정: 네 번째·다섯 번째 예산으로 `elev`(고도각 상한)와 `ground`(지면 여유 하한)를 넣는다.**
`elevation_profile()` 은 렌더 0회 — 프레임당 3×3 곱 하나라 재투영보다도 싸다. `solve_knob` 예산
사슬은 `collision → obb → ground → elev → clearance → hole` 순이다 (바닥이 각도보다 먼저 막는
게 기하적으로 맞다 — camel drop 은 실제로 `ground` 가 물고 `elev` 는 4.2° 에서 끝난다).

**임계는 여기서도 소스 대비다** (D47/D51 과 같은 규칙 — 소스 카메라를 기각하는 임계는 버그다).
- 고도: `max(--max_elev_deg 45, 소스 자신 |고도각| + --elev_src_margin_deg 10)`. 45° 는 후보
  풀의 elevation 열거 상한과 같은 값이다. 소스는 −6.32°~**+35.09°** (avocado `stat_4` chair,
  소스가 코앞 0.033 u 에서 내려다본다) 라 절대값만으로는 그 앵커가 기각된다.
- 지면: `--min_ground_clear_ratio × 소스 카메라 자신의 지면 위 높이`. **비율**인 이유는 높이가
  camel **0.1069 u** / avocado **0.3641 u** 로 3.4배 벌어져서다 (두 씬 다 49프레임 내내
  ±0.0005 u 로 일정 — 고정 높이 촬영이라 "소스 높이"가 잘 정의된다).

**ratio 는 0.2 다 — 실측으로 골랐다** (camel `dyn_0`, hole 0.35). 결정하는 건 camel 뿐이다:

| ratio | camel `drop_reveal` path / 실측 지면여유 | camel `pedestal_down` path | avocado |
|---|---|---|---|
| 0.0 | 0.106 u / **0.003 u** (바닥 스침) | 0.092 u | 세 값 전부 동일 |
| **0.2** | **0.059 u / 0.050 u** | 0.082 u | (카메라가 0.364 u 로 |
| 0.5 | **0.014 u** (τ 하한, 사실상 정지) / 0.094 u | 0.051 u | 높아 안 걸린다) |

0.5 는 drop 을 τ 하한(0.019)까지 눌러 **정지 클립**을 만든다 — D53 과 같은 실패라 채택 못 한다.
0.0 은 바닥을 0.003 u 로 스친다. 0.2 만 실제 하강과 여유를 동시에 남긴다.

**효과** (camel `dyn_0`, hole 0.35): `rise_reveal` 88.5° → **44.1°**, knob 2.375 → 0.633.
`drop_reveal` 은 지면 −1.282 u → **+0.050 u**.

**기존 동작 보존 확인**: `--no_elev_gate --no_ground_gate` 로 돌린 결과가 기존 뱅크와
`knob/status/binding/tau_max/path_len_u/hole_fraction/behind_frac/obb_slack/sweep_deg/pan_deg/
shape_mult` 전 열 **비트 동일**. 게이트를 꺼도 `elev_*`/`ground_clear` 열은 계속 측정된다.
직전 뱅크는 `out/<video>/hole_bank_pre_g6/` 에 남겨뒀다.

**기록해둔 선택지** (필요해지면):
1. **고도각 기준점을 subject 중심이 아니라 subject 상단/하단면으로.** 지금은 OBB 중심 대비라
   납작한 물체(avocado `stat_4` 두께 0.00) 위에서는 각도가 과대평가된다. D54 옵션 4(얇은 박스
   두께 하한)와 같이 봐야 한다.
2. **`elev` 를 preset 별로 다르게.** `rise_reveal` 은 원래 올라가는 샷이라 45° 가 맞는데
   `orbit_*` 은 12~28° 면 충분하다. 지금은 전 preset 공통이라 orbit 쪽은 사실상 no-op.
3. **`ground` 를 지면이 아니라 "바닥 근접 시 hole 급증"으로 대리 측정.** 지면 평면 RANSAC 이
   틀린 씬(벽 3장짜리 avocado)에서 더 견고하겠지만 렌더가 필요해 비싸다.
4. **`evaluate()` 의 `GATE_ORDER` 에 G6 추가.** 지금은 사다리만 쓴다. 후보 풀은 elevation 을
   `{12,28,45}°` 로 **열거**하므로 시작 pose 에는 이 구멍이 안 생긴다 — 생기는 건 궤적 쪽이라
   당장은 불필요하다. 자유 pose 를 허용하게 되면 그때 켠다.
5. **천장 쪽 대칭 게이트.** 지금 `ground` 는 아래만 본다. 실내 씬(avocado 벽 3장)에서 위로
   뚫는 변이는 아직 안 봤다 — `rise` 가 `elev` 로 먼저 걸려서 드러나지 않았을 뿐이다.

---

### D56 dolly 가 subject 를 **지나쳐** 뒤에 선다 — `obb_signed_distance` 가 부호 없는 거리라서 (G7)

사용자: *"앞 뒤로 움직이는 것도 bbox 를 지나치기 전까지 적당한 거리까지만 움직이게끔해줘"*

**원인은 임계가 아니라 양의 정의다.** G5 가 쓰는 `lbm/gates.py:obb_signed_distance` 는 박스
표면까지의 **부호 없는** 거리다 (박스 안일 때만 음수). 그래서 dolly 가 박스를 **옆으로 스쳐**
뒤쪽에 서도 거리만 벌어져 있으면 slack 이 양수로 나온다. 사용자가 본 "지나침"이 정확히 이
구멍이다. 실측: 게이트 없는 camel `dyn_0__push_in_arc` 는 뒷면을 **2 프레임** 통과하는데
`obb_clear` 는 그 구간에서 0.1536 이상이라 G5 가 아무것도 안 한다.

**G7 `approach_profile`.** 축을 하나 고정해서 거리에 **부호**를 준다:

```
a      = normalize(anchor OBB 중심(frame 0) − 플랜 시작 카메라 위치)     # 한 번 정하고 안 바꾼다
s(f)   = (p_g(f) − c_j(f)) · a          # 카메라가 anchor 중심을 지나 얼마나 갔나
half_a = a 방향 OBB 반폭
gap(f) = −half_a − s(f)                 # 근접면까지 남은 여유 (음수면 근접면 침범)
past(f)=  s(f) − half_a                 # 뒷면 통과량 (양수면 지나쳤다)
```

렌더 0회. **anchor 노드에만** 건다 — 축을 anchor 가 정의하므로 다른 노드엔 뜻이 없다.

**임계는 D47/D51/D55 와 같은 꼴이다 — 소스 대비 배수.** `--approach_src_ratio β` ×
(소스 카메라 자신의 시선축 여유). 절대값으로 두면 또 원본 촬영을 기각한다. **β=0.3 은
사용자가 고른 값**이고 G5 의 `--obb_clear_src_ratio` 와 같은 수로 맞췄다 — 두 게이트가 같은
"소스보다 30% 까지만 더 붙어도 된다"를 서로 다른 축에서 말하는 것이라 가를 근거가 없다.

**효과** (게이트 off = `out/nog7/`, on = `out/<video>/hole_bank/`, anchor 1개 × preset 2개 × 4단):

| 변이 | knob off→on | path_len_u off→on | 뒷면 통과 프레임(off) | gap(on) | binding off→on |
|---|---|---|---|---|---|
| camel `dyn_0 push_in_arc` (4단 전부) | 1.000 → **0.573** | 0.795 → 0.442 | **2** | +0.173 | `obb`→`approach` |
| camel `dyn_0 straight_ease` hole0.1 | 0.564 → 0.528 | 0.427 → 0.400 | 0 | +0.176 | `obb`→`approach` |
| camel `dyn_0 straight_ease` hole0.2↑ | 0.573 → 0.512 | 0.434 → 0.390 | 0 | +0.186 | `obb`→`approach` |
| avocado `stat_0 push_in_arc` (4단) | 0.681 → **0.149** | 0.465 → 0.152 | 0 | +0.065 | `collision`→`approach` |
| avocado `stat_0 straight_ease` (4단) | 0.574 → **0.149** | 0.375 → 0.122 | 0 | +0.057 | `elev`→`approach` |

전량 재생성 후 **뱅크 336 변이 전부 `past_frames = 0`, `approach_frames = 0`** (camel 168 /
3,037 렌더, avocado 168 / 3,080 렌더). 잔여 위반은 G1·obb·지면아래·근접면침범·뒷면통과 전부 0.
`binding` 분포는 camel `hole 86 / approach 27 / ground 19 / elev 11 / obb 8 / none 17`,
avocado `hole 77 / approach 49 / elev 18 / ground 13 / none 11` — G7 이 두 씬 모두에서 **두 번째로
많이 무는 예산**이 됐다. avocado 쪽 압축이 큰 건 `stat_0` 이 테이블(두께 0.024 u)이라 소스
카메라 자신의 시선축 여유가 작기 때문이다.

**anchor 를 3개로 줄였다** (`--anchors dyn_0 stat_0 stat_1` / `dyn_0 stat_0 stat_5`).
사용자 지시("anchor를 main 3개 정도로 줄여줘"). 고른 기준: 주인공(동적 track) + 최대 정적 노드 +
관측 방위폭(`obs_az_span`) 최대 정적 노드. 뱅크가 anchor×preset×4단으로 곱해지므로 여기가
렌더 수를 정하는 자리다 (anchor 7 → 3 이면 렌더가 2.3배 줄어든다).

**기존 동작 보존**: `--no_approach_gate` 면 `approach_gap`/`approach_frames`/`past_frames` 열만
남고 판정에서 빠진다. 위 표의 "off" 열이 그 결과다.

**기록해둔 선택지** (필요해지면):
1. **축 `a` 를 프레임마다 다시 잡기.** 지금은 frame 0 에 고정이라 anchor 가 크게 움직이는 씬에서
   축이 낡는다. 고정한 이유는 축이 프레임마다 돌면 `s(f)` 가 비단조가 되어 이분법이 흔들리기
   때문이다. camel 낙타 track 은 0.09 u 라 지금은 문제가 안 된다.
2. **뒷면 통과를 금지가 아니라 예산으로.** "박스 뒤 0.2·half_a 까지는 허용" 같은 식.
   `fly-through` 류 preset 을 넣게 되면 필요해진다 — 지금 어휘엔 없다.
3. **anchor 외 노드까지 확장.** 축을 "카메라 진행 방향"으로 바꾸면 전 노드에 걸 수 있다.
   다만 그건 G5 를 부호 있는 판정으로 바꾸는 것이라 D49/D51 재측정이 따라온다.
4. **β 를 G5 와 분리.** 지금 둘 다 0.3 인 건 근거가 아니라 정합성 선택이다. avocado 에서
   `approach` 가 49번 무는 게 과하다고 판단되면 G7 쪽만 0.15 로 내려본다.

---

### D57 죽은 레버 7개 삭제 — "과설계된 부분은 정리해줘"

사용자: *"과설계된 부분은 정리해줘"*

`fit_hole_ladder.py` 의 손잡이 7개가 **기본값에서 no-op** 인 채로 여러 세대를 남아 있었다.
전부 지웠다. 남은 건 열(column)뿐 — **측정은 계속 하고 판정만 없앴다.**

| 삭제 | 기본값 | 왜 죽었나 |
|---|---|---|
| `--min_standoff_ratio` | 0.0 | D50 이 껐다. OBB(G5)와 같은 걸 두 번 재고(침범 33변이의 standoff 가 33개 전부 임계 아래) 먼저 물어서 `binding` 을 가린다. 게다가 불연속 축이라 "조금 키우기"가 안 된다 (0.80↔0.90 에서 camel `push_in_arc` 0.939→0.020) |
| `--min_standoff` | None | 위의 절대값 모드 |
| `--min_near_depth` | 0.0 | D47 이 기각. 걸리는 픽셀 세로위치 median 0.98(화면 맨 아래 바닥)이고, 화면 **밖** 근접 기하는 못 본다 |
| `--near_pct` | 1.0 | 위 판정의 백분위. 열 측정용 모듈 상수 `NEAR_PCT = 1.0` 으로 내렸다 |
| `--obb_clear_ratio` | 0.0 | D51 이 실측으로 기각한 크기 비례항. 0 이라 항상 no-op |
| `--min_obb_clear` | 0.02 | β 모드가 floor 를 덮어써서 도달 불가 (camel 0.153 / avocado 0.035 > 0.02) |
| `--obb_clear_cap` | 0.12 | 〃 상한. `node_margins(nodes, 0.0, floor, inf)` 로 호출되므로 애초에 안 쓰인다 |

같이 정리한 것: `solve_knob` 의 `clearance` 분기와 `clearance_limited` status, 요약표의
`clearance` 열, `over()` 의 인자 2개, 모듈 docstring 6문단. G7(D56) 문단은 없어서 새로 썼다.
CLI 인자 46 → 39.

**남긴 이유가 있는 것** — `standoff`/`near_depth` 는 **열로 계속 나간다**. 뱅크는 인벤토리고
거르는 건 소비자 몫이라는 규칙(D39/D45)에 따라, 판정을 안 하더라도 "이 카메라가 표면에 얼마나
붙었나"는 뱅크를 읽는 쪽이 알아야 한다. 요약표에도 "소스보다 붙은 변이 N개 (판정 아님)" 로
한 줄 남긴다.

**기록해둔 선택지**:
1. `sample_camera_bank.py` 에도 `--near_pct`/`--measure_standoff` 가 남아 있다. 그쪽은 **판정이
   없고 측정만** 하므로 죽은 레버가 아니다. 정리하려면 `behind_context` 시그니처까지 건드려야
   해서 비용 대비 실익이 없다.
2. `--shape_headroom`/`--shape_doublings` 도 후보였는데, G6 가 이 배증을 **실제로 상한까지 밀어
   붙이는** 걸 D55 에서 확인했으므로 살아 있는 레버다. 남긴다.

**비트 동일성 검사 결과** — `out/d57/camel` 재생성 vs `out/camel`, 168 행 / 열 집합 / 변이 집합
전부 동일. **판정에 쓰이는 것은 전량 비트 동일**하다: `poses.npz` 의 `cam_c2w` 168 변이 ×
49프레임이 `max|diff| = 0.000e+00`, 그리고 `knob`·`status`·`binding`·`tau_max`·`path_len_u`·
`hole_fraction`·`behind_frac`·`standoff`·`obb_slack`·elev/ground/approach 열이 전부 같다.

**다른 건 진단 열 둘뿐이고, 렌더러 비결정성으로 설명된다.** `near_depth` 78/168 행
(상대차 median 0.00% / p90 3.28% / max 18.20%), `subject_area_med` 45/168 행 (−4.35% ~ +0.36%).
`eval/probe_near_depth_repeat.py` 로 **같은 pose·같은 프로세스**에서 반복 측정:

| 변이 | 뱅크 | 재현 ×4 | 재현 산포 |
|---|---|---|---|
| `dyn_0__orbit_left_arc__hole0.2` (최악 행) | 0.3275 | 0.3791 / 0.4414 / 0.3510 / 0.3510 | **23.75%** |
| `dyn_0__straight_ease__hole0.1` | 0.2038 | 0.2038 ×4 | **0.00%** |
| `stat_0__truck_left__hole0.35` | 0.3056 | 0.3227 / 0.3192 / 0.3120 / 0.3141 | 3.39% |

최악 행의 **한 프로세스 안 산포(23.75%)가 뱅크 간 차이(18.20%)보다 크다**. 두 뱅크 값
(0.3275, 0.3871)이 모두 재현 범위 [0.3510, 0.4414] 근처에 들어온다. 반면 다른 변이는 4회
비트 동일이다 — 비결정성은 **변이마다 다르다**. `near_depth` 가 depth 의 **1 백분위**라 splat
z-buffer 동률 몇 개가 통째로 값을 옮기기 때문이고, D47 이 이 양을 판정에서 기각한 이유와 같다.

⚠ **재현할 때 렌더 해상도를 맞춰야 한다** (`--tile_width 640 --tile_height 360`). 앞선 시도가
뱅크 값 근처에도 못 간 건 해상도가 달랐기 때문이다 — splat 밀도가 바뀌면 1 백분위가 통째로
움직인다. 프레임 집합은 `verify_frames 13` → `[0,4,8,…,48]`, `S` 는 `bank.json` 의 값.

---

### D58 CinemaTraj 의 SDF 는 **관측이 아니라 박스 월드**다 — 우리 씬엔 못 옮긴다 (조사)

사용자: *"CinemaTraj처럼 SDF하면 왜 unkown이 뭐고 왜 정확도가 낮다고 판단됨? 논문에선 어떻게
해결했는데?"*

**논문 쪽 (arXiv 2607.26910v1 §3.4.1, Appendix D.4/J).** 그들의 SDF 에는 unknown 이 **구조적으로
없다**. 만드는 재료가 관측이 아니라 **scene graph 의 OBB** 이기 때문이다:

```
OBB 들 → 복셀화(가장 얇은 축에 최소 3복셀, 최대 512³) + 얇은 구조(문·창틀) 표면 샘플링
       → morphological closing 1회로 얇은 벽 틈 봉합
       → marching cubes (level 0.5) → Laplacian smoothing 1회
       → MeshLib unsigned distance @ 256³ (+0.5 m padding),
         부호는 mesh centroid 에서 6-연결 flood-fill
```

§3.1.2 에서 **벽·바닥·천장 박스를 서로 늘려 붙여 watertight enclosure 를 먼저 만든다** —
"ensuring the signed distance field (§3.4) has no leakage through room boundaries". 즉 방이 닫힌
상자라서 flood-fill 이 안팎을 무조건 가른다. 복셀은 mesh 내부 아니면 외부, 세 번째 라벨이 없다.
OBB 는 뒷면을 한 번도 못 봤어도 **꽉 찬 고체**로 채워지므로 가림/미관측이라는 개념 자체가 없다.
논문은 이걸 가정으로 적지 않고, 실패 모드를 leakage 하나로만 본다.

입력 가정이 그걸 떠받친다 — **posed RGB-D** (abstract: "Given a set of RGB-D images and a user
prompt"), 데이터는 **ScanNet++ 50 씬**, "a real-world indoor dataset with **high-fidelity 3D
reconstructions**" (§4.1.2). monocular depth 도 GT mesh 도 안 쓴다. 렌더용 기하(3DGS)와 충돌용
기하(OBB SDF)가 **서로 다른 표현**이다.

**검증은 없다.** 지표 다섯(Motion MSE / CLaTr / Collision Rate / Occlusion Rate / Object
Coverage) 중 **기하 정확도 지표가 하나도 없다** — Chamfer·IoU·F-score·OBB fit error·segmentation
mAP 전부 부재. 게다가 Collision Rate 는 optimizer 가 최소화하는 **바로 그 Φ** 로 잰다:
최적화는 `g_j = d_safe − Φ(p)`, `d_safe = 0.2 m` 로 Φ>0.2 를 밀고, 지표는 Φ<0 을 센다.
같은 필드에 더 느슨한 임계일 뿐이라 독립 검증자가 없다. §D.5 에는 "plateau 에서 collision rate
가 5% 위면 λ_sdf 를 두 배(최대 10³)" 라는 규칙까지 있어 **보고 지표가 정지 조건이기도 하다**.
한계 항목(Appendix K.1–K.4)에 미관측 공간·가림·monocular 얘기는 **없다**.

**우리 쪽 (`eval/probe_static_sdf.py` 실측).** 우리는 재료가 반대다 — 우리 SDF 는 **관측된
표면**에서 space carving 으로 만들 수밖에 없고, 그러면 세 번째 라벨이 반드시 생긴다:

```
FREE     어떤 프레임에서든 그 프레임 depth 표면 **앞**에 보이면
OCCUPIED 표면에 걸리면
UNKNOWN  둘 다 아니면 = 가려졌거나 **화각 밖**       (FREE 가 OCCUPIED 를 이긴다)
```

실측 (camel / avocado-slice, 세 변이 A=frame0 / B=13프레임 동적포함 / C=13프레임 동적제외):

| | camel free | avocado free |
|---|---|---|
| A frame0 | 0.029 | 0.015 |
| B 13프레임 (동적 포함) | 0.029 | 0.015 |
| C 13프레임 (동적 제외) | 0.029 | 0.015 |

즉 **97% 이상이 unknown 이다.** 그리고 뱅크 카메라 위치의 SDF 는 min·p05·median 이 세 변이
전부 **0.0000** — 뱅크 pose 중 소스 프레임 화각 안에 들어오는 게 camel **5.8%** / avocado
**9.3%** 뿐이라 나머지는 "본 적 없음"이다. 원인은 가림이 아니라 **화각**이다: camel 은 hfov
28.7° 에 카메라 track 이 0.0032 u, avocado 는 hfov 55.2° 에 0.0727 u. 한 자리에서 좁게 찍은
영상으로는 카메라를 놓고 싶은 자리를 애초에 관측하지 못한다.

**사용자의 후속 질문 — "segmentation 있으니 동적 물체 빼고 static point cloud 로 SDF 되나?"
→ 안 된다.** B vs C 가 소수점 셋째 자리까지 동일하고, 달라지는 건 conflict 복셀 수뿐
(camel 121→110, avocado 633→631). 동적 가림이 원인이 아니었기 때문이다.

**그래서 채택 안 한다.** 대신 지금 게이트들이 **단측 판정**이라는 게 요점이다 — G1 은 "관측된
표면 **뒤**면 기각", 화각 밖은 통과다. 자유공간 증명서를 요구하지 않으므로 unknown 이 97% 여도
동작한다. SDF 는 반대로 "여기가 free 임을 증명하라"를 요구해서 우리 입력에선 답이 안 나온다.
현재 뱅크는 **G1 위반 0 / G5 위반 0** 이라 SDF 가 추가로 잡을 것도 없다.

**기록해둔 선택지** (필요해지면):
1. **CinemaTraj 방식을 그대로 이식** = 우리 scene graph OBB + 벽/바닥 평면으로 watertight 상자
   월드를 만들고 그 위에 SDF. 그러면 unknown 이 사라진다 — 대신 그건 G5(OBB 부호거리)를 복셀로
   다시 푸는 것뿐이고, 우리는 벽 평면을 아직 저장도 안 한다 (D55 옵션 5, `relations.py:97` 이
   `(normal, d)` 를 버린다). **실익이 없다고 판단.**
2. **다중 클립 통합.** 같은 씬의 다른 clip 을 얹으면 화각 커버리지가 올라간다. Scene-Decoupled
   쌍이 있는 코퍼스에선 가능. camel/avocado 는 단일 clip 이라 불가.
3. **unknown 을 명시적으로 3값으로 실어 소비자에게 넘긴다** (뱅크 인벤토리 규칙 D39/D45).
   지금은 `probe_static_sdf.py` 안에만 있고 뱅크 열로는 안 나간다.

---

## D59 — CinemaTraj 의 벽/바닥 OBB: **바닥은 이미 쓰고 있고, 벽은 켜도 0 개가 바뀐다** (실측)

D58 이 "CinemaTraj SDF 는 관측이 아니라 박스 월드"라고 결론냈지만, 그 박스 월드에는
**벽·바닥·천장**이 들어 있고 우리 노드 목록에는 없다. 그래서 "그 부분만이라도 가져올 수 있나"를
따로 쟀다 (`eval/probe_wall_planes.py`).

**셋으로 갈린다.**

| CinemaTraj 요소 | 우리 상태 | 근거 |
|---|---|---|
| 물체 OBB → SDF 충돌 | **이미 한다. 더 정확하게.** G5 `obb_clearance` 가 같은 노드 OBB 의 **해석적 부호거리**를 푼다 | 복셀 256³ 로 굽지 않으므로 해상도 손실도 25 s 굽는 시간도 없다 |
| 바닥 상자 | **이미 한다.** G6 `ground_clear` 가 gravity RANSAC 평면을 쓴다 | camel `ground_z −0.1075·S` / avocado `−0.3633·S`. 세그멘테이션 마스크가 아니라 평면 적합이라 "floor" 명사가 없어도 된다 |
| 벽 상자 | **찾아는 놓고 버린다. 되살려도 소용없다.** | 아래 실측 |

`relations.find_wall_planes` 가 실제로 찾는 평면 (camel 1개, avocado 3개)을 반공간 fence 로
켰다고 가정하고 뱅크 카메라 **8,232 pose**(168 변이 × 49프레임)를 판정했다:

| 씬 | # | 법선(G) | d | 정적 점 앞/뒤 | 소스캠 safe측 | **뱅크캠 위반** |
|---|---|---|---|---|---|---|
| camel | 0 | [−0.296, 0.955, 0.029] | 0.3209 | 0.590 / **0.410** | 1.000 | **0.000** |
| avocado | 0 | [−0.749, −0.663, −0.014] | 1.1355 | 0.755 / 0.245 | 1.000 | **0.000** |
| avocado | 1 | [0.623, −0.782, −0.023] | −1.4171 | 0.837 / 0.163 | 1.000 | **0.000** |
| avocado | 2 | [0.689, −0.724, −0.025] | −1.5372 | 0.874 / 0.126 | 1.000 | **0.000** |

두 가지가 동시에 나온다.

1. **게이트를 켜도 기각되는 pose 가 0 개다.** G1(behind-surface)이 이미 카메라를 소스가 관측한
   쪽에 묶어두기 때문이다. 벽 반공간은 **G1 에 포섭된다** — 그리고 G1 쪽이 낫다. 적합한 평면
   하나가 아니라 픽셀별 관측 표면 전부를 쓴다.
2. **애초에 벽이 아니다.** camel 평면 뒤에 정적 점의 **41%** 가 있다 (avocado 도 13~25%).
   경계면이면 뒤가 비어야 한다. 지면을 벗겨낸 뒤 남은 **최대 평면**일 뿐이고, 씬을 가르는
   칼이지 두르는 담이 아니다.

**왜 CinemaTraj 에서는 되는가**: 저쪽 입력은 ScanNet++ **방**이고 wall/floor/ceiling 이 라벨된
인스턴스라 §3.1.2 가 그걸 **watertight enclosure** 로 스냅한다 — 그래서 "벽 뒤"가 씬 밖과
동의어다. camel 은 실외라 두를 상자가 없고(fence/roof/tree 뿐), avocado 는 주방이지만 단안이라
한 면만 본다. **enclosure 가 없으면 평면은 fence 가 아니라 slice 다.**

**결론: 이식하지 않는다.** 물체 OBB 와 바닥은 이미 쓰고 있고, 벽은 켜도 0 개가 바뀐다.

**기록해둔 선택지**:
1. `(normal, d)` 를 `scene_graph.json` 에 **저장은 한다** (D55 옵션 5). 게이트로 안 써도
   VLM 프롬프트의 `## SCENE` 블록에 "3 wall planes" 를 넣을 수 있고, 실내 코퍼스가 들어오면
   그때 켜면 된다. 씬 그래프 재생성이 필요하다.
2. 실내 씬(TRUMANS, ScanNet++ 류)이 들어오면 **그때 재측정**. 위 0.000 은 camel/avocado 두
   실외·단안 씬의 성질이지 방법의 성질이 아니다. enclosure 가 닫히는 씬에서는 뒤집힐 수 있다.
3. `probe_wall_planes.py` 를 verify 에 붙여 **씬마다 자동 판정**. 위 표를 씬 그래프 빌드 때
   찍어두면 "이 씬은 벽 게이트가 의미 있나"를 사람이 안 봐도 된다.

---

## D60 — VLM 은 **게이트를 대체하는 게 아니라 뱅크에 열을 더한다** (가림·구도 사후 판정)

D58 에서 "VLM 이 기하 게이트를 대신할 수 있나"에 아니라고 답했는데, 사용자의 실제 제안은 그게
아니었다: **최소한의 물리적 안전장치로 생성한 뒤**, 시간축으로 샘플링한 프레임을 VLM 에 넣어
"끝에 가서 목표물이 가려졌나"를 판정하자는 것. 이건 다른 층이고, **맞다.**

**지금 뱅크에 의미 판정이 하나도 없다.** 51개 열 중 subject 관련은 둘뿐이다:

| 열 | 뜻 | 못 잡는 것 |
|---|---|---|
| `subject_in_frame` | OBB 2D 중심이 중앙 80% 안인 프레임 비율 | 화면 안에 있는데 **앞 물체에 가려진** 경우 |
| `subject_area_med` | subject 픽셀 비율의 중앙값 | 면적이 줄어든 게 **멀어져서**인지 **가려져서**인지 못 가른다 |

가림 지표가 **아예 없다.** 그리고 이건 임계 하나로 못 만든다 — 면적 감소가 거리와 가림에
공통으로 걸리기 때문이다. 계획서의 G3 는 이걸 세 임계(중앙 80% · 면적 ∈ [0.03, 0.50] ·
z-buffer 통과율 ≥ 0.4)로 풀려 했는데, 그 셋은 아직 **안 만들었고 만들 이유도 없다**.

**그래서 레버는 준다 — 안 만들어도 되게 되는 쪽으로.** VLM 판정이 들어오면 영영 안 생기는 손잡이:
`--center_box`(현재 0.8), 면적 하한/상한 2개, z-buffer 통과율 임계 1개 → **4개**. 물리 게이트
쪽(β=0.3 ×2, `max_elev_deg`, `min_ground_clear_ratio`, behind 의 margin/clear_frac/radius)은
**그대로 남는다** — D58 의 실측(`corr(hole, behind_frac)` = −0.014 / +0.059) 이 여전히 유효하다.

**결정적 제약: 이분법 **안**에 넣으면 안 된다.** `over(knob)` 은 변이당 ~8회 평가된다.
거기에 VLM 을 넣으면 168 변이 × 2 씬 × 8 = **2,688 콜**이고, 더 나쁜 건 **이분법이 비결정적**이
되어 뱅크가 재현 불가능해진다 (D57 의 비트 동일성 검사 자체가 무의미해진다).

**채택안: 뱅크 뒤에 붙는 채점 패스.** `scripts/judge_bank_vlm.py` (미구현) 가 완성된 뱅크를
읽어 변이당 **1콜** — 시간 균등 6프레임 스트립 + OBB 오버레이 → `{"occluded_frames": [...],
"ends_occluded": bool, "framing": "good|tight|lost", "reason": str}` — 을 받아
`vlm_occluded_frac` / `vlm_ends_occluded` / `vlm_framing` / `vlm_reason` **열을 추가**한다.
336 콜, 로컬 Qwen3-VL 기준 10~20분.

이게 **뱅크 인벤토리 규칙(D39/D45)과도 맞는다** — 뱅크는 거르지 않고 재고를 쌓고, 거르는 건
소비자의 일이다. VLM 판정은 게이트가 아니라 소비자가 쓸 **또 하나의 열**이다.

**기록해둔 선택지**:
1. **판정을 기하 열과 대조**해 불일치 변이만 영상으로 뽑는다 (`occl_disagree.mp4` 형식).
   `subject_area_med` 는 정상인데 VLM 이 "가림"이라 한 변이가 정확히 이 층이 필요했던 증거다.
   불일치가 0 이면 VLM 이 필요 없다는 뜻이므로 **먼저 이걸 재는 게 순서**다.
2. **6프레임이 아니라 마지막 구간만** — 사용자가 짚은 건 "마지막에 가려지게끔 움직였나"다.
   프레임 [0, 24, 40, 44, 48] 처럼 뒤를 촘촘히 샘플링하는 게 목적에 더 맞을 수 있다.
   6균등 vs 뒤편중을 같은 변이에 둘 다 돌려 비교해볼 것.
3. **기하 대리지표를 먼저 만들어 본다** — subject 점군의 z-buffer 통과율은 렌더 0회 추가로
   계산된다. 이게 VLM 과 잘 맞으면 336 콜이 공짜가 된다. 다만 점군 hole 이 subject 점을
   가짜로 "보인다" 판정하는 편향이 있어 (D58 의 frustum 커버리지 camel 5.8% / avocado 9.3%)
   단독으로는 못 믿는다.
4. **판정을 후보 열거 범위로 되먹인다** — D58 이 남긴 "VLM 이 도움 되는 곳은 레버가 아니라
   열거 범위" 와 같은 자리다. 어떤 anchor×preset 조합이 반복적으로 "가림" 판정을 받으면
   그 조합을 애초에 안 만든다.

---

## D61 — 51편으로 늘리기 전에 앞단 실패 4종을 실측하고 막았다

7편(camel · avocado-slice · goat · room-argue · car-roundabout · woman-pottery · parkour)으로
파일럿을 돌려서 나온 것들이다. 넷 다 **2편(camel/avocado)에서는 아예 안 보이던** 실패다.

**① 명사 추출이 SAM3 결과를 읽고 있었다.** `extract_nouns_vlm.py` 의 비교 기준이
`seg_instances/<video>/meta.json` 이라 "SAM3 를 먼저 돌려야 명사 추출이 된다"는 순서가 생겼고,
parkour 가 아직 안 끝난 상태에서 `FileNotFoundError` 로 51편 전체가 죽었다. → 원천인
`metadata.csv:dynamic` 을 직접 읽는다. 없는 영상은 빈 목록(비교 기준만 없고 추출은 된다).

**② 인스턴스 수 폭발.** 실측: 동적 car-roundabout **69**, 정적 parkour **56** / car-roundabout
**54** / goat **31**(전부 `rock`). 기존 기각 기준 3종(`max_area_frac` · `visible_frames` ·
`mean_score`)은 **track 하나하나가 멀쩡한지**만 보므로 개별로는 다 멀쩡한 이 실패를 구조적으로
못 잡는다. 해로운 이유 셋: ⓐ G5 는 **전 노드 최소값**이라 상자가 많으면 자유공간이 쪼개져
전부 `obb_limited` 로 붕괴 ⓑ `rock` 31개는 사실 지형이라 OBB 자체가 무의미 ⓒ graph 시간이
track 수에 선형. → `cap_instances`, `--per_keyword_top_k 5 --max_dyn_nodes 6 --max_stat_nodes 10`.
순위는 `max_area_frac` — `sample_camera_bank.py --min_area_frac` 이 앵커를 고를 때 쓰는 **바로
그 양**이라 하류와 기준이 갈리지 않는다. 버린 건 전부 `diagnostics.dropped_tracks` 에 사유째로.

**③ 올려놓고 버렸다.** car-roundabout 은 track 이 123개인데 그중 100개 넘게 **49프레임씩
unproject 한 뒤에** 버렸다. → `summarize_tracks` 로 2D 마스크만 보고 먼저 거르고
(`mask.sum()` 뿐이라 사실상 공짜), 살아남은 것만 `build_instances(select=...)` 로 올린다.
lift 전 상한은 최종 상한 × `--prescreen_factor 3` — 최종 상한은 merge **뒤**에 걸어야 "면적
상위 K" 가 물체 단위로 세어지는데(같은 물체 조각 5개가 상위 5칸을 먹으면 안 된다) lift 비용은
merge **전**에 깎아야 해서, 둘 사이의 타협이다.

**④ 마스크 depth 가 두 덩어리.** goat `dyn_0` 의 OBB 가 마스크 대비 **13.06배**,
extent `[0.962, 0.047, 0.821] u` — 가운데 축이 사실상 0인 납작한 판. 프레임 24 실측(npx 50692):
본체가 z 0.62~1.0 에 44k px, **가운데가 텅 빈 채로** z 6.5~7.5 에 7000 px = 마스크의 **14%**.
1% 대칭 quantile(`trim_depth_tail`)로는 14% 에 닿지 못하고, `radius_outlier_mask` 도 저쪽이
7000점이라 서로 이웃이어서 못 잡는다. → `trim_depth_mode`: log-depth 히스토그램에서 **중앙값이
속한 덩어리만** 남긴다. 단봉이면 그대로 반환하므로 정상 장면은 안 건드린다.
순서는 **덩어리 먼저, 꼬리 나중** — 두 덩어리가 남아 있으면 quantile 이 엉뚱한 쪽을 자른다.

**⑤ 광역 표면이 두 단어로 새어 들어왔다.** `SURFACE_NOUNS` 가 정확일치라 `brick wall` /
`dirt track` / `tile floor` 가 통과했다. → `is_surface` 가 **머리단어(마지막 토큰)로도** 본다.
51편에서 실제로 올라온 `court`/`track`/`path`/`hill`/`beach`/`dirt` 등도 목록에 추가.

**수정 후 실측** (7편 병렬, GPU 1~3):

| 영상 | 노드 (dyn+stat) | dropped | size_ratio 최대 | WALL |
|---|---|---|---|---|
| camel | 2+4 | 1 | 1.63 | 273.7 s |
| avocado-slice | 3+8 | 3 | 2.02 (table) | 271.5 s |
| woman-pottery | 2+5 | 6 | 2.42 (shelf) | 207.4 s |
| room-argue | 3+8 | — | 1.59 | 476.1 s |
| parkour | 1+10 | 42 | nan (stat_4, 꼭짓점 카메라 뒤) | 422.0 s |
| car-roundabout | 6+10 | **93** | 1.54 | 535.6 s |
| goat | 1+5 | 18 | **3.65** | 333.4 s |

room-argue 의 `pillow` 3개는 수정 전 3.47 / 3.85 / 5.86 이었고 지금 1.06 / 0.76 / 1.06 이다 —
④가 실제로 듣는다는 가장 강한 증거. camel 은 주체 상자가 3% 이내로 안 변했고, 배경이 새어
부풀었던 정적 상자만 줄었다 (`fence` 0.746→0.294 u). 상자가 **줄면 G5 가 느슨해지므로**
안전한 방향이다.

**goat 만 남았다 — 그리고 이건 depth 추정기 한계다.** f=2289 px / 1280 px 폭 = hfov 31° 의
망원 샷이고, 씬의 73%가 z>5 인데 주체와 바위는 z 0.6~0.9 다. 마스크 안 depth 가 0.66~1.0 로
퍼지는데 마스크가 말하는 실제 가로폭은 0.116 (DA3 단위) 뿐이다 — **시선 방향 오차가 물체 크기와
맞먹는다.** 카메라가 41° 내려다보고 있어서 그 수평축 부풀림이 화면 **세로**로 나타난다
(계산: 2289·0.219/0.80·cos41° + 2289·0.273/0.80·sin41° ≈ 988 px, 실측 1018 px).
지금은 **고치지 않고 놔둔다** — OBB 가 큰 쪽으로 틀리면 G5/G7 이 보수적으로 작동할 뿐이고,
잘 맞는 6편을 건드릴 위험이 더 크다.

**기록해둔 선택지**:
1. **실측으로 게이트한 깊이축 축소** — `check_reprojection` 이 이미 `size_ratio` 를 재고 있으니,
   `size_ratio > 1.8` 인 노드**만** 시선 정렬 수평축을 (다른 수평축 길이를 바닥으로) 줄이고
   재측정하는 루프를 돌린다. 기존 `--deinflate` 의 무조건판(`parallax<0.05` ∧ 축비 2.5배)과
   달리 camel `fence`(축비 7.25 지만 size_ratio 1.20)를 안 건드린다. **goat `dyn_0` 은 이걸로
   1.0 근처까지 내려간다.** 다만 goat `stat_1`(바위)은 두 수평축이 0.027/0.024 로 비슷해서
   바닥에 걸려 못 고친다 — 즉 이 옵션은 절반만 듣는다.
2. **`fit_obb` 를 min/max 가 아니라 백분위로** — 축마다 [2, 98] 백분위. 깨끗한 구름은 거의
   안 변하고 꼬리만 깎인다. ④를 이걸로 대신할 수 있었는지 확인하는 대조군도 된다.
3. **망원 씬을 코퍼스에서 표시** — `hfov` 를 `scale` 에 실어두고, 하류가 "이 씬은 시선축
   기하를 믿지 마라"를 읽을 수 있게 한다. 고치는 게 아니라 **범위를 정직하게 적는** 쪽.
4. **상한 값(5/6/10)을 바꾼다** — 지금은 car-roundabout 차 69대 / parkour 정적 56 / goat rock
   31 을 기준으로 잡은 것이다. 뱅크 인벤토리 규칙(D39/D45)에 맞추면 "상한 없이 다 만들고
   소비자가 고른다"가 원칙이지만, 여기서는 상한이 **G5 최소값 붕괴**를 막는 물리적 이유가
   있어 예외로 둔다. 0 이하를 주면 상한이 꺼지고 예전 동작 그대로다.
5. **명사 대신 class-agnostic "전 픽셀 → instance" 제안기로 갈아탄다** (Mask2Former panoptic /
   SAM2 automatic mask generator). 지금은 **안 한다**.

   - **SAM3 안에는 그런 모드가 없다.** 프롬프트는 concept(text) 아니면 box 하나뿐이고,
     `sam3_video_inference.py:193-197` 이 `"visual prompts (box as an initial prompt) should
     only have one box"` 로 명시적으로 막는다. 게다가 `add_prompt` 는 종류를 가리지 않고 맨 앞에서
     `reset_state` 를 부르므로(`:866`) **box 프롬프트로 가면 instance 당 전파 1회**가 되어
     지금(명사당 1회, 영상당 평균 10.5회)보다 오히려 느려진다.
   - **Mask2Former** 는 `vista4d` env 에서 클래스 import 까지는 된다(transformers 4.57.6,
     가중치 미다운로드). 막히는 건 성능이 아니라 **출력 형태**다 — 프레임 단위 panoptic 이라
     `track_id` 가 없다. 49프레임 identity 를 붙이려면 결국 SAM2/SAM3 전파를 다시 태워야 하고
     그러면 위의 "instance 당 전파 1회" 비용으로 돌아온다. 어휘도 COCO-133 closed 라
     `pottery wheel` 같은 건 애초에 못 낸다.
   - **SAM2 automatic mask generator** 는 리포에 코드·가중치 둘 다 있다
     (`tools/Grounded-SAM-2/{sam2/automatic_mask_generator.py, checkpoints/sam2.1_hiera_*.pt}`).
     단 `vista4d` env 에 `sam2` 모듈이 없고 설치는 harness 가 거부한다.

   **왜 지금은 아닌가 — 이미 후보 과잉이다.** 51편 SAM3 산출의 unique track id 는 dynamic
   평균 22.4 / median 8 / max 255, static 평균 16.5 / median 10 / max 84 로 영상당 합계 평균
   ~39개다. 최종 노드 상한은 dyn 6 + stat 10 = **16**. 파일럿에서 실제로 버린 track 수가
   car-roundabout 93 · parkour 42 · goat 18 이다. 즉 병목은 **recall 이 아니라 선별**이고,
   제안기를 더 촘촘한 것으로 바꾸면 실패 모드 ②(instance 폭발 → G5 는 모든 노드에 대한 MIN
   이라 전부 `obb_limited` 로 붕괴)를 **악화**시킨다. panoptic 은 stuff 클래스(wall/floor/sky)까지
   내는데 그건 `SURFACE_NOUNS` 로 일부러 빼고 있는 것들이라 순수 부작용이다.

   **갈아탈 조건**: `obb_overlay_all.mp4` 에서 "사람 눈에 명백히 보이는데 노드가 없는" 물체가
   여러 씬에서 반복될 때. 그때도 교체 지점은 `scripts/sam3_*_instances.py` 뿐이다 —
   `scene_graph/io.py` 가 `vista4d_seg_instances_v1` 만 읽으므로 그 포맷으로 변환만 하면 된다.

---

## D62 — 대량 데이터셋 단계의 탐지 예산 (사용자 지시, 2026-08-22, **아직 미적용**)

지금 돌고 있는 51편은 **현재 설정 그대로 끝낸다**(사용자 지시). 아래는 그 다음, 대량으로
데이터셋을 만들 때 적용할 값이다.

| 항목 | 지금 (51편) | 대량 단계 |
|---|---|---|
| main dynamic 노드 | `--max_dyn_nodes 6` | **3** |
| static 노드 | `--max_stat_nodes 10` | 줄인다 (미정, 권장 5) |
| static 명사에 `floor` | **뺀다** (`SURFACE_NOUNS`) | **반드시 포함** |

**왜 지금 안 바꾸나**: 파일럿 7편이 6/10 상한으로 측정된 것이라, 51편을 다른 상한으로 돌리면
D61 의 실측표와 비교가 안 된다. 상한은 `build_scene_graph.py` 플래그라 재실행만 하면 되므로
지금 급할 이유도 없다.

**`floor` 는 충돌한다 — 노드로 넣으면 안 된다.** `floor`/`ground` 는 `SURFACE_NOUNS` 에 있어
일부러 빼고 있는데, 이유는 D61 ②와 같은 계열이다: 바닥에 OBB 를 씌우면 그 상자가 씬 전체를
덮고, **G5(OBB clearance)는 모든 노드에 대한 MIN** 이므로 모든 후보 카메라가 즉시
`obb_limited` 로 붕괴한다. 바닥은 "피해야 할 덩어리"가 아니라 "그 위에 서는 면"이다.

**권장 해소법 (선택지 1, 기본값으로 제안)**: `floor` 를 SAM3 로 잡되 **OBB 노드로 승격하지
않고** ground 평면 쪽에만 쓴다 — 즉 ground RANSAC 의 inlier 후보를 floor mask 안으로 제한한다.
지금 ground 는 전 픽셀에서 RANSAC 을 돌려서(`gravity.py`) 벽·탁자 상판을 바닥으로 잘못 잡을
여지가 있는데, floor mask 가 그걸 막아준다. 즉 사용자가 원하는 "floor 꼭 포함"의 실익
(중력축·G6 지면 게이트가 정확해지는 것)은 얻고 G5 붕괴는 피한다.

다른 선택지:
2. `floor` 를 노드로 넣되 **G5 에서만 제외**한다 (`kind="surface"` 를 새로 두고 clearance MIN
   계산에서 뺀다). G6 지면 게이트는 이 노드를 쓴다. 1번보다 스키마 변경이 크다.
3. 그냥 노드로 넣는다. **비권장** — 위 이유로 뱅크가 전멸한다. 넣으려면 G5 를 먼저 고쳐야 한다.

적용 지점: `fit/graph/build_scene_graph.py --max_dyn_nodes 3 --max_stat_nodes 5`,
`fit/graph/extract_static_nouns.py` 의 `SURFACE_NOUNS` / `--max_nouns`, 그리고 1번을 택하면
`scene_graph/gravity.py` 가 floor mask 를 받도록 인자 추가.

---

## D63 — 동적 노드 extent 는 프레임 median 이다 (유지, 선택지 기록)

`build_scene_graph.py:76-85` 는 `fit_obb` 를 **프레임마다** 돌린 뒤 `extent = np.median(extents,
axis=0)` 으로 크기를 하나로 접는다. 프레임별로 남는 건 center(평활)와 yaw(median 필터)뿐
(`obb.py:94` "extent 만 시간 불변으로 본다").

**실측 — 겉보기 world 크기의 프레임간 산포** (SAM3 박스 px × 프레임별 카메라 거리 ÷ focal):

| video / node | 높이 p10/med/p90 (u) | 높이 p90/p10 | 폭 p90/p10 | 저장 extent(up) |
|---|---|---|---|---|
| camel dyn_0 | 0.126 / 0.129 / 0.130 | 1.03x | 1.82x | 0.122 |
| avocado-slice dyn_0 woman | 0.185 / 0.186 / 0.187 | 1.01x | 1.02x | 0.180 |
| woman-pottery dyn_0 human | 0.347 / 0.350 / 0.360 | 1.04x | 1.02x | 0.361 |
| room-argue dyn_0 man | 0.257 / 0.259 / 0.261 | 1.01x | 1.04x | 0.266 |
| room-argue dyn_1 woman | 0.287 / 0.293 / 0.300 | 1.05x | 1.41x | 0.304 |
| **parkour dyn_0 man** | 0.053 / 0.074 / 0.090 | **1.69x** | **2.14x** | 0.076 |

11개 동적 노드 중 10개가 높이 산포 1.01~1.05x — 서 있거나 앉은 사람은 사실상 상수다. 폭이
더 흔들리는 건(camel 1.82x, woman 1.41x) 크기 변화가 아니라 **yaw** 이고 프레임별 yaw 가 이미
처리한다. 진짜 모양이 변하는 건 parkour 하나뿐이다.

**median 을 유지하는 이유**: 이 상자는 G5(clearance)·G7(전진 한계)에서 **장애물**로 쓰인다.
프레임마다 줄었다 늘면 줄어든 프레임에서 카메라가 파고들었다가 늘어난 프레임에서 물체를
뚫는다 — 게이트는 통과하는데 렌더가 깨진다. 게다가 single-view depth 의 프레임별 fit 노이즈가
실제 모양 변화보다 크다: 같은 파일 :105-111 의 실측대로 **정적** 울타리 OBB 중심이 0.3217u
떠도는 동안 **움직이는** 낙타는 0.0674u 다(4.8배). 중심이 그만큼 흔들리는 fit 이면 extent 도
같이 흔들린다. yaw 에 median 필터를 거는 것과 같은 이유.

**남는 위험은 방향이 나쁘다** — parkour 처럼 편 프레임에서 상자가 작으면 G5 가 여유를
과대평가해 뻗은 팔다리를 스친다.

선택지:
1. **`extent = np.percentile(extents, 90, axis=0)`** (동적 노드만). 보수적 방향으로 한 줄.
   11개 중 1개에만 영향이 있으므로 지금은 안 바꾼다 — 바꾸면 D61 파일럿 표와 비교가 깨진다.
2. 프레임별 extent 를 그대로 쓰되 **누적 max** 로 단조 증가시킨다. 게이트 안정성은 지키면서
   최대 크기를 반영. 저장 스키마가 (49,3) 으로 커진다.
3. extent 산포 자체를 노드에 기록만 하고(`extent_p10/p90`) 게이트는 median 을 쓴다.
   parkour 류가 몇 편인지 51편에서 세는 용도. 1·2 중 뭘 할지는 그 수를 보고 정한다.

---

## D64 — 실내 씬 중력축: RANSAC 문턱을 낮춰 살릴까 (TRUMANS, **기본 꺼둠**)

`trumans-bedroom` 은 `gravity.method = camera_up_fallback` 으로 떨어졌다. 그런데 attempts 를
보면 **바닥을 찾긴 찾았다**: peel 2 가 inlier 0.142, 카메라 up 과 12.8°, 카메라 전부 평면 위.
`inlier_ratio ≥ 0.15` 하나에만 걸려 버려지고, 대신 실제 바닥과 **12.8° 어긋난** 카메라 up 이
중력축이 됐다. 실내는 바닥이 화면의 15% 를 못 채우는 게 정상이라 이 문턱이 씬 종류를 가른다.

52편 전수: fallback 15편 중 **3편**이 이 경우(`trumans-bedroom` 0.142/12.8°,
`basketball-four` 0.132/28.2°, `park-selfie` 0.105/11.4°). 나머지 12편(셀피·대화 영상)은
각도 조건부터 못 넘겨서 진짜 fallback 이 맞다 — 문턱만 낮춘다고 다 살아나는 게 아니다.

`--weak_inlier_ratio 0.10` 으로 `trumans-bedroom` 을 다시 지어 대조한 결과(**같은 것만 비교**):

| | 기존 (camera_up) | 신규 (ground_ransac_weak) |
|---|---|---|
| up 축 | — | 기존 대비 **12.81°** 회전 |
| reproj_px 중앙값 | 28.10 | **21.60** |
| reproj_px 평균 | **30.63** | 33.65 |
| bed OBB l,w,h | 0.69,0.53,**0.46** | 0.70,0.56,**0.34** |
| `supported_by=ground` 노드 | 1 | **2** (stat_6 desk 가 붙음) |
| 노드 수 | 11 | 12 |

**깨끗한 승리가 아니다.** 중앙값은 좋아지고 평균은 나빠진다 — reproj_px 는 2D 마스크 bbox 대비
오차라 중력축에 크게 안 흔들리기 때문이다. 진짜 근거는 물리 쪽이다: 침대 높이 0.46u→0.34u 가
침대답고, 책상이 바닥에 붙는다.

**지금 판정: 기본 0.0(꺼짐) 유지.** 51편 코퍼스가 이미 옛 동작으로 돌고 있어서 지금 켜면
뱅크 간 게이지가 섞인다. 중력축은 G6(elevation)·G7(ground clearance)·roll=0 에 전부 들어가므로
켜고 끄기를 영상마다 다르게 하면 안 된다.

선택지:
1. **계속 꺼둔다.** 51편과 TRUMANS 가 같은 규약. 실내 씬은 중력축이 12~28° 기운 채로 남는다.
2. **다음 전량 재생성 때 `--weak_inlier_ratio 0.10` 으로 통일.** 3편이 바뀌고 나머지 49편은
   RANSAC 이 이미 성공했거나 각도 조건에서 탈락하므로 결과가 그대로다. 권장.
3. 문턱을 없애고 **각도·"카메라가 위" 두 조건만으로 판정**. 벽/책상 상판을 지면으로 오인할
   위험이 inlier_ratio 로 막히던 것이라 실측 없이 켜면 안 된다.
4. 실내/실외를 `metadata.csv:source` 나 sky 비율로 갈라 문턱을 다르게. TRUMANS 는 sky 0.02%
   라 갈리긴 한다. 분기가 하나 더 느는 값어치가 있는지는 실내 씬이 몇 편 되느냐에 달렸다.

---

## 아직 안 정한 것 (6단계 이후)

- ~~VLM 백엔드가 살아 있는지~~ → 확인됨. `/v1/models` 가 `Qwen/Qwen3-VL-30B-A3B-Instruct`
  (ctx 32768) 를 반환하고 `extract_nouns_vlm.py` 4콜이 0.76~1.79 s 에 통과했다 (repair 0회).
- micro-adjust 라운드 상한(계획서 10)과 "게이트 연속 3회 거절 시 종료"의 실제 수렴 여부.
- segmentation 제안기: **VLM+SAM3 가 기본값**(사용자 확정). GDINO+RAM 은 설치 실패로 미측정
  (`groundingdino` 휠이 cp310, vista4d 는 3.12.13; `ram` 휠 `--no-deps` 설치는 harness 가 거부).

---

## D65 — TRUMANS 를 원본 LBM 입력으로 태울 때의 어댑터 선택 4가지 (2026-08-22, **실행 완료**)

**배경.** 원본 LBM 은 이 프로젝트에서 한 번도 안 돌아갔다. 입력이 `.blend` 씬 자산 +
story/layout JSON 5종인데 리포가 그 5종을 "intentionally excluded" 했고(README:5), 우리
소스는 dynamic video 라 애초에 줄 수가 없었다. TRUMANS 는 mesh·SMPL-X·`.blend` 를 그대로
갖고 있어서 처음으로 실측이 가능해졌다. 아래 4개는 그 과정에서 갈림길이었던 것들.

### ① take 식별 — 프레임 수 일치 (채택)

한 씬 uuid 에 실제 take 가 ~10개(+`_augment1/2` 모션 증강 사본)인데 `.blend` 에는 그중
**하나만** 구워져 있다. 어느 take 인지 모르면 `Actions/<seq>.txt` 를 엉뚱한 것에 붙이게 된다.

| 선택지 | 판정 |
|---|---|
| **`scene_list`/`seg_name`/`scene_flag` 로 시퀀스별 프레임 수를 세어 blend 프레임 수와 일치하는 것** | **채택.** `00add26c-…` → blend 0..2076 = 2077 → `2023-01-17@00-55-00` 유일 일치. 결정론적 |
| 파일명 접미사(`action0..9.png` 의 `.009`) 로 추정 | 위 결과와 일치하긴 했지만 **일치했다는 걸 프레임 수로 확인한 것**이라 단독으론 근거가 안 된다 |
| 사용자에게 매번 지정하게 | fallback 으로 남김 (`--sequence`). 일치가 0개거나 2개 이상이면 assert 로 요구 |

### ② 씬 이름 `Scene` — 렌더 워커에 opt-in fallback (채택)

TRUMANS 씬 이름은 그냥 `Scene` 인데 LBM 은 `Scene_<id>` 규약으로 찾는다. Director 와 preview
워커는 fallback 이 있고 `blender_render_worker.py:64` **만** 없어서 앞 두 단계는 통과하고
렌더에서만 죽는다.

| 선택지 | 판정 |
|---|---|
| **워커에 "파일에 씬이 딱 하나면 그걸 쓴다" 가지 추가, `LBM_SINGLE_SCENE_FALLBACK=1` opt-in** | **채택.** vendored 리포 수정 1곳, 기본 동작 불변. 다른 세 워커가 이미 같은 방식으로 떨어진다 |
| blend 를 열어 씬 이름을 `Scene_1` 로 바꿔 저장 | 1.66 GB 를 recording 마다 다시 쓴다. 71편이면 ~110 GB |
| demo_root 의 `scene_id` 를 이름에 맞추기 | `scene_id` 는 int 여야 한다 (`director_stage.py:463`) |

### ③ blend 는 심링크 (채택)

전 워커가 `-b` 배경 실행이고 저장 호출이 없다 = **LBM 은 입력 blend 를 수정하지 않는다.**
코드로 확인한 뒤 복사 대신 심링크. recording 당 1.66 GB 절약.

### ④ shot = `Actions/<seq>.txt` 의 줄 (채택)

LBM 이 shot 입력에서 실제로 읽는 키는 `shot_id`(int) + `shot_description` 둘뿐
(`director_stage.py:463-466`). Actions 각 줄의 `text` 를 그대로 `shot_description` 으로,
`shot_id` 는 줄 번호. `start`/`end` 프레임은 LBM 이 안 읽으므로 버린다 —
**대신 그래서 shot 당 지속시간이 전부 기본값 3.0 s 가 된다** (실제 동작 길이와 무관).
행동 길이를 살리려면 `target_duration_seconds` 를 직접 채워야 하고, 그건 아직 안 했다.

### ⑤ `quality` 실측 완료 — 비교 분모는 "쓸 수 있는 카메라 1대" (채택)

`quality` 실측이 끝났다 (run `trumans_00add26c_q`, 합 876.9 s, rc=0). fast 와 달리 후보 탐색이
실제로 돌았다 — shot 당 raw ~1,200 (합 19,235) → eligible 5,078 → retained 286, board 3장/shot.

| 분모 | LBM-Lite | 원본 `fast` | 원본 `quality` |
|---|---|---|---|
| 만든 카메라 | **7.85 s** | 30.7 s | 54.8 s |
| **쓸 수 있는** 카메라 | 7.85 s | 82.0 s (6/16) | 67.5 s (13/16) |

**어느 분모를 쓰느냐가 결론을 뒤집는다.** "만든 카메라"로 보면 quality 가 fast 의 1.8배로 비싸
보이지만, VLM 게이트 통과가 6/16 → 13/16 으로 오르므로 **쓸 수 있는 카메라 1대당으로는 오히려
싸다**(82.0 → 67.5 s). 우리는 카메라를 데이터셋으로 쓰므로 후자가 맞는 분모다.
→ 앞으로 원본 LBM 시간을 인용할 때는 **항상 `quality` + 쓸 수 있는 카메라 분모**로 쓴다.

배율은 약 **8.6배** (67.5 / 7.85). 단 원본 쪽 표본이 shot 16개·씬 1개뿐이라 자릿수만 읽을 것.
LBM-Lite 쪽은 13편 3,416 대 실측이고 `fit` 이 79.8% 를 먹는다.

### 아직 안 정한 것

- 71개 recording 전량으로 늘릴지. 늘린다면 shot 지속시간을 Actions 의 `start`/`end` 로 채우는
  것부터.
- quality 에서도 shot 1·3·5 는 차단됐다. 셋 다 fast 에서도 차단됐던 shot 이라 **카메라 탐색이
  아니라 shot 자체(= Actions 줄)의 문제**일 가능성이 있는데 확인 안 했다.

---

## D66. 뱅크 `knob` 반올림이 `fit_tau` 스케일 계단을 넘긴다 (2026-08-22)

`emit_bank` 의 pose 대조가 snow-dog `stat_0__truck_right__hole0.1` 에서 1.793e-03 로 깨졌다.
회전은 비트 단위로 같고 시작 pose 도 같은 **순수 균등 스케일**(1.004274) 어긋남 — 참 손잡이
0.228128185878 이 `round(_, 5)` 로 0.22813 이 되면서 `fit_tau` 이분법의 스케일 계단
(0.01428 → 0.01434)을 **1.8e-6 차이로** 넘긴 것이다. 손잡이→궤적이 연속이 아니라는 게 요점.

| 선택지 | 판정 |
|---|---|
| **`knob_raw`(무반올림)를 행·CSV 에 같이 싣고 `emit_bank` 가 우선 읽는다** | **채택.** `knob` 열은 그대로라 기존 리더가 안 깨진다 |
| **예전 뱅크는 `recover_knob` 이 ±5e-6 을 훑어 참값을 되찾는다** | **채택.** 뱅크 37개 재-fit(전체 wall 의 ~80%)을 피한다. 재현이 `--pose_tol` 안에 들 때만 채택하므로 조용히 틀릴 수 없다 |
| `knob` 을 무반올림으로 바꾼다 | `bank.csv` 의 기존 열 값이 전부 바뀐다 |
| `--pose_tol` 을 1e-3 로 올린다 | 진짜 재현 버그(basketball-four 의 8.918 같은)를 못 잡게 된다. **기각** |

확인: snow-dog(1 행 복구) / basketball-four / camel 모두 pose 재현 최대오차 0.000e+00.
basketball-four 의 8.918 은 별건인 `shape_mult` 버그였고 그 패치로 이미 해결돼 있었다.

## D67. snowboard 는 **소스 카메라 자신이 τ 예산을 초과**한다 (2026-08-22, 미적용)

51편 확장에서 `snowboard` 만 `fit` rc=1 로 죽었다. 원인은 fit 이 아니라 그 앞
`sample_camera_bank.py` 다.

```
tau_start        1.9441      # 사다리 꼭대기가 1.0
dropped_saturated  350 조합  # 움직이는 preset 14종 × anchor 5 × 사다리 5 전량
남은 변이         10 개      # static_hold / static_hold_locked 뿐, 그마저 tau_saturated=True
fit              AssertionError: 푼 게 0 (fit_hole_ladder.py:647)
```

`tau_start` 는 **정지 플랜(static hold)** 의 τ 다 — 플랜 카메라는 anchor 시작 pose 에
가만히 있는데 소스 카메라가 날아가므로 `|p_plan − p_src| / z_med` 가 혼자 커진다.
snowboard 는 스노보더를 따라가는 주행 샷이라 49프레임 동안 소스가 `z_med` 의 약 2배를
이동한다 (S=2.9229, z_med=1.7510). D53 규칙("τ 하한 = `tau_start` + 0.02")이 사다리 전 단을
saturated 로 밀어내고, 남은 정지 변이도 saturated 표시라 fit 이 집을 게 없다.

**전 코퍼스 분포 (bank.json `dropped_saturated` 실측, 41편)**

| 구간 | 편수 | 예 |
|---|---|---|
| `tau_start` = 0 (전량 통과) | 30 | camel, breakdance, car-roundabout … |
| 0 < `tau_start` ≤ 0.5 | 10 | parkour 0.7454(98 변이 생존), truck-pose 0.5894, fashion-walk 0.4906, trumans-bedroom 0.4673 |
| `tau_start` > 1.0 | **1** | **snowboard 1.9441 — 움직이는 변이 0** |

즉 snowboard 는 코퍼스에서 유일한 파탄 사례고, 두 번째로 심한 parkour 도 98개는 살아남는다.

**보강 (2026-08-22 늦게, 뱅크 51편째까지) — 유일하지 않다. 2편이다.**

`snow-bike` 가 같은 모양으로 죽었다. 위 "41편 실측"은 `snow-bike` 가 **아직 안 돈 8편에
들어 있어서** 표에서 빠졌던 것이고, 분포 결론("1편")은 그만큼 낙관적이었다.

| 영상 | `tau_start` | 사다리 | `dropped_saturated` | 생존 변이 | 결과 |
|---|---|---|---|---|---|
| snowboard | 1.9441 | [0.1, 0.2, 0.35, 0.6, 1.0] | 350 | 10 (static 2종뿐) | `fit` rc=1 → `bankemit` rc=1 |
| **snow-bike** | **1.3765** | 같음 | **210** | **6 (static 2종뿐)** | 같음 |

둘 다 눈 위를 **따라가는 주행 샷**이다 — 실패 모드가 장르에 붙어 있지 개별 씬 사고가 아니다.
`snow-bike` 는 S=2.8809, z_med=0.8145 로 `z_med` 가 snowboard(1.7510)의 절반이라 소스 이동이
더 작아도 τ 가 커진다. **`tau_start` 를 키우는 건 소스 이동량이 아니라 이동량/깊이 비다.**

수정: 파탄 사례는 52편 중 **2편**(3.8%), 둘 다 동일 원인. ①의 대가는 "1편 손실"이 아니라
"2편 손실"이고, 그래도 ①을 유지한다 — ②는 여전히 τ 눈금을 씬마다 흔들어 코퍼스 간 비교를
깨고(그 대가가 2편보다 크다), ③은 τ 1.4~1.9 플랜이 하류 hole 게이트에서 어차피 전멸한다.

**선택지**

| 안 | 내용 | 대가 |
|---|---|---|
| ① **건너뛴다 + 사유 기록** (권장) | `fit` 이 뱅크가 비면 crash 대신 `skipped.json`(사유·`tau_start`) 쓰고 rc=0 | 51편 중 1편 손실. τ 눈금이 전 코퍼스에서 그대로 비교 가능 |
| ② τ 사다리를 **excess** 로 (`target = tau_start + Δ`) | hole 사다리가 이미 쓰는 의미론과 일치. snowboard 도 카메라를 얻는다 | τ 절대값이 씬마다 달라져 **코퍼스 간 비교가 깨진다**. 30편은 `tau_start`=0 이라 무변화지만 11편은 값이 바뀜 |
| ③ 사다리 꼭대기를 2.5 로 올린다 | 코드 변경 최소 | τ 1.9 짜리 플랜은 hole 이 감당 불가. 어차피 하류 게이트에서 전멸할 것 |
| ④ τ 를 **최근접 소스 프레임** 기준으로 재정의 | 주행 샷에 물리적으로 맞는 정의 | 전 코퍼스 τ 재계산 + D41/D53 전부 재검토. 이번 phase 범위 밖 |

**지금 한 것: 아무것도 안 고쳤다.** `bank51`(screen)이 아직 돌고 있다 — 51편 완료 +
`women-talk`(52편째) 진행 중. 앞서 `fit_hole_ladder.py` 를 **실행 중에** 고쳐서 6편이 구포맷
뱅크로 남은 전례가 있어 런이 끝나기 전에는 손대지 않는다. 끝난 뒤 ①을 적용하고
**snowboard + snow-bike 2편**을 재실행한다.

**`bankemit` 만 rc=1 인 2편(`basketball-four`, `snow-dog`)은 이미 해결된 건이다.** `knob` 반올림
버그(CHANGELOG Fixed 항목)로 죽은 것이고, `recover_knob` 패치 뒤 22:11 에 수동 재emit 해서
둘 다 `worst_pose_rebuild_error` **0.000e+00** 으로 canonical 이 나와 있다. `run51/*.json` 의
rc=1 은 그보다 **앞선** 기록이라 남아 있을 뿐이다 (basketball-four 20:18, snow-dog 21:58).
→ **런너가 stale 한 rc 를 그대로 들고 있다**는 뜻이므로 `run51/*.json` 만 보고 실패를 세면 안 된다.
실제 미해결은 snowboard + snow-bike **2편**뿐.

---

## D68. 뱅크 16,520 개 중 **고유 궤적은 그보다 한참 적다** — 사다리 fold (2026-08-23, 실측, 미적용)

사용자 요청 "lite 뭐가 문제인지 영상으로" 에 답하려고 전 50편 `hole_bank/bank.csv` 를 집계했다.
결론은 게이트가 카메라를 너무 많이 죽인다가 **아니다**. 게이트는 죽일 만한 걸 죽인다. 문제는
**게이트가 상한을 물린 뒤에도 뱅크가 사다리 4단을 4개의 서로 다른 카메라로 계속 센다**는 것이다.

영상: `results/2026-08-23_lbm_lite_failures/lite_failures.mp4` (4줄 × 4칸, 각 줄이 같은
anchor×preset 의 hole 목표 0.1/0.2/0.35/0.5). 집계: 같은 폴더 `summary.txt`.

| 줄 | 조합 | 4단 path_len (u) | 원인 |
|---|---|---|---|
| OK | park-selfie `stat_0 truck_left` | 0.039 / 0.575 / 1.411 / 2.838 | 정상 (×72) |
| FOLD | avocado-slice `dyn_0 orbit_left_arc` | 0.597 ×4 | `approach_limited` 상한이 사다리보다 낮다 |
| CLAMPED | elderly-tennis `dyn_0 push_in_arc` | 0.086 ×4 | `obb_clear` **−0.057** — 시작 pose 가 이미 여유 안쪽 |
| PAN | women-talk `dyn_0 pan_left` | 0.000 ×4 | pan 은 제자리 회전. status 는 `solved` 로 찍힌다 |

전 코퍼스 (50편 / 16,520 변이 / anchor×preset 조합 4,130):

- `solved` **41.3%**. `approach_limited` 10.0 / `ground_limited` 8.9 / `clamped_low+tau_floor` 7.6 /
  `obb_limited` 7.6 / `elev_limited` 7.1 / `shape_limited` 6.7 / `unreached` 5.8 / `collision_limited` 2.5%.
- **fold: 4단 조합 4,130 중 1,995 (48.3%) 가 4단 전부 동일 궤적.** pan 계열(이동 0)을 빼도
  1,405 (34.0%).
- **순수 회전 2,360 (14.3%)** — 전부 `pan_left`/`pan_right`. τ 를 곱해도 이동이 0이라 사다리가
  의미 없다.
- `clamped_low` 1,343 (8.1%). elderly-tennis 는 **280/280 (100%)**.
- 완전 차단 2편 (D67): snowboard / snow-bike.

**왜 지금 안 고치나.** fold 를 없애는 건 사다리 정의를 바꾸는 일이고, 그러면 이미 나온 50편
뱅크를 전량 재fit 해야 한다 (wall 의 ~80%, 24 h). 다음 선택지들을 기록만 해두고 사용자 판정을
기다린다:

1. **(권장) emit 단계에서 중복 제거.** `emit_bank` 가 같은 (anchor, preset) 안에서 `knob_raw`
   가 같은 행을 하나로 접고 `manifest.json` 에 `folded_into` 를 적는다. 재fit 불필요.
   뱅크 크기 표기가 정직해지고 하류 학습이 같은 카메라를 4배 가중하는 것도 사라진다.
2. **사다리를 hole 이 아니라 "게이트 상한까지의 비율"로.** 상한이 0.6 u 면 0.15/0.3/0.45/0.6.
   fold 는 구조적으로 사라지지만 사다리 단끼리 hole 값이 안 맞아 영상 간 비교가 깨진다. 재fit 필요.
3. **pan 계열을 사다리에서 제외.** `pan_left/right` 는 τ 축이 없으니 1단만 낸다. 14.3% 즉시 감소.
   하지만 pan 은 회전 크기 축이 따로 있어야 맞다 (`pan_deg` 를 사다리로 쓰는 별도 안).
4. 아무것도 안 한다. 뱅크 개수는 부풀지만 하류가 중복에 강하면 무해하다. — 실측 없음.

**판정 (2026-08-23, 사용자): ①. 적용 완료.**

코드 변경은 없었다 — `emit_bank.py` 에 `fold_groups()`(:152) · `folded_onto`(:283) ·
`--drop_folded`(:375) 가 처음부터 다 있었고 기본값만 "안 거른다"(D39/D45)였다. 플래그를 켜서
**별도 디렉토리**로 다시 emit 했다 (`--out_dir <bank>/canonical_dedup`). 기존
`<bank>/canonical/` 은 손대지 않았다 — 사용자 지시("기존꺼 지우지 말고").

fold 키가 `knob_raw` 가 아니라 5자리 반올림 `knob` 인 게 걸려서(:162) 먼저 대조했다.
**전 코퍼스 16,520 변이에서 두 키의 fold 수가 5,957 로 정확히 같다** (갈리는 영상 0편).
D66 의 `recover_knob` 사례처럼 반올림이 `fit_tau` 계단을 넘긴 행은 fold 경계에는 안 걸린다는
뜻이라 키를 바꿀 이유가 없다.

| | 기존 `canonical/` | 신규 `canonical_dedup/` |
|---|---|---|
| 태그 수 (50편 합) | 16,520 | **10,563** (−5,957, −36.1%) |
| 편당 med / min / max | — | 200.5 / 35 / 472 |
| worst pose rebuild | 0.0 | **0.0** |
| worst roundtrip | 0.0 | **0.0** |

감소 폭 상하위 (게이트 천장이 낮을수록 많이 접힌다):

```
elderly-tennis      280 ->  70  (-75.0%)   obb_clear 음수 = 시작 pose 가 이미 여유 안쪽
martian-flag        168 ->  42  (-75.0%)
funeral-procession  336 -> 170  (-49.4%)
parkour             392 -> 204  (-48.0%)
park-selfie         224 -> 117  (-47.8%)
...
girls-selfie        280 -> 207  (-26.1%)
hike                168 -> 125  (-25.6%)
woman-pottery        56 ->  43  (-23.2%)
```

`elderly-tennis` / `martian-flag` 가 정확히 −75.0% 인 것은 **모든 조합이 4단을 1단으로
접었다**는 뜻이다 (본문의 `clamped_low` 280/280 을 다른 각도에서 본 같은 사실).

접힌 4행은 `bank.json`/`bank.csv` 에 그대로 남아 있으므로 "이 조합은 hole 0.5 를 못 낸다"는
진단은 보존된다. 사라지는 건 canonical 태그뿐이다.

**남은 것 — ③ 은 안 했다.** dedup 후에도 `translation_degenerate`(이동 0, 전부
`pan_left`/`pan_right`)가 **2,300 개 = 10,563 의 21.8%** 로 남는다. fold 는 같은 궤적을 접을
뿐이고 pan 은 단마다 회전각이 달라 접히지 않기 때문이다. 이 카메라들은 `--scales` 도 `|t|`
정규화도 무의미하므로 하류에서 따로 취급해야 한다. ③(pan 을 hole 사다리에서 빼고 `pan_deg`
사다리로 분리)은 판정 대기.

## D69 — LBM `clip.mp4` 가 프레임 1장이다 (2026-08-23)

TRUMANS 6편 30개 clip **전량**이 1프레임. `renders/<shot>/<cam>/frames/frame_*.png` 에는 31~48
프레임이 멀쩡히 있다 (00add26c 314장, 6편 합계 1166장). 즉 렌더는 성공했고 **mp4 인코딩 단계만**
망가졌다. `manifest.json.blender_render_result.returncode = 0`, `success: true` 라 파이프라인은
정상 종료로 보고한다 — 조용히 지나가는 종류다.

부수 실측: 프레임 번호가 **stride 2** (`frame_0001…0061` = 31장). 그래서
`clips_manifest_v1.json` 의 `duration_seconds` 가 `target_duration_seconds` 의 정확히 절반이고
(3.6→1.80 / 3.0→1.52 / 2.5→1.24 / 3.8→1.92), 선언 fps 25 로 틀면 2배속이다. 의도 속도는 12.5 fps.

지금은 고치지 않았다 — 우리가 LBM 에서 필요한 건 카메라 궤적이지 mp4 가 아니고, PNG 를 직접
읽는 경로(`viz/concat_videos.py --inputs <frames dir>`)를 뚫어놨다.

선택지:
1. **(선택함) 안 고치고 PNG 를 읽는다.** vendored LBM 을 안 건드린다. 비용: 렌더를 볼 때마다
   concat 한 단계.
2. LBM 의 clip 인코더를 고친다 — `VideoEngineer` 가 `imageio.mimwrite` 대신 뭘 쓰는지 찾아
   교체. 원본 fork 를 건드리는 첫 사례가 된다.
3. stride 2 를 1 로 되돌린다 (렌더 시간 2배). 지금 shot 당 45프레임이면 충분하다고 봤다.
4. `--fps 12.5` 를 기본으로 박지 말고 LBM 이 stride 를 manifest 에 적게 한다. stride 가
   `duration_seconds` 에 반영 안 된 게 근본 원인.

---

## D70. 보행 채굴 fps 는 blend `render.fps` 가 아니라 **30 고정** (2026-08-23, 적용)

`mine_walk_actions` 가 7편 중 **5편에서 0건**을 냈다. 표로만 보면 "그 편엔 보행이 없다"와
"임계가 틀렸다"가 안 갈린다. 후자였다.

판정은 `step * fps > 0.4 m/s` 인데 그 `fps` 를 `probe_meta["fps"]`, 즉 **blend 의 `render.fps`**
에서 받고 있었다. blend fps 는 편마다 **15 또는 25** 다. 프레임당 이동량 분포는 7편이 사실상
같은데 (p90 0.0163~0.0362 m/frame) fps 만 다르니, fps 15 편에서는 같은 걸음이 0.24 m/s 로 찍혀
임계 아래로 떨어졌다.

**모션 배열의 실제 레이트는 30 이다.** 근거: `video_render/<seq>.pkl.mp4` 가 7편 전부
프레임 수 = 시퀀스 길이로 **1:1** 이고 fps 가 **30** 이다. blend 의 15/25 는 렌더 설정일 뿐
모션 샘플링 레이트가 아니다.

| 가정 fps | 00add26c | 0a761819 | 0aa05d5a | 1d43e076 | 2b4c9b84 | 3a19c7bb | 4ac2c1b3 | 합 |
|---|---|---|---|---|---|---|---|---|
| blend (25/15) | 8 | 0 | 0 | 0 | 0 | 9 | 0 | 17 |
| 25 | 8 | 9 | 4 | 3 | 7 | 11 | 2 | 44 |
| **30 (선택)** | **9** | **13** | **5** | **5** | **11** | **12** | **4** | **59** |

확인 영상 `results/2026-08-23_trumans_walk/walk_windows.mp4` (14타일): 버려지던 창이 전부
49프레임 순 수평이동 **0.82~1.87 m** 의 실제 보행이다.

선택지:
1. **(선택함) `--motion_fps` 기본 30.** `0` 을 주면 예전처럼 blend fps 로 떨어진다 (재현용).
   숫자를 인자로 빼서 로그·manifest 에 남게 했다 — 하드코딩하면 다음에 또 조용히 틀린다.
2. `walk_speed` 를 편별로 조정. 같은 물리량을 두 손잡이로 나눠 잡는 셈이라 기각.
3. 속도가 아니라 프레임당 이동량(m/frame)으로 판정. fps 를 아예 안 쓰니 더 견고하지만,
   임계 0.4 m/s 가 사람이 읽을 수 있는 값이라는 장점을 잃는다. 보류.

**부수 효과**: 채굴이 늘면서 `--walk_max 8` 이 실제로 걸린다 (59 → 46). 순 이동량 내림차순
상위 8개가 남으므로 "많이 걷는 창부터"는 유지된다.

---

## D76. preset 조준 표기를 **비대칭**으로 — 기본은 무표기, 이탈만 `_dont_look` (2026-08-28, 적용)

D75 에서 `aim` 을 이름에 드러내려고 `_aimed`/`_locked` 를 **대칭으로** 붙였다
(`dolly_in_aimed` / `dolly_in_locked`, `static_hold_aimed` / `static_hold_locked` ...).
문제는 이 어휘가 학습 캡션에 그대로 나간다는 것이다 — 36 preset 중 10개가 접미사를 달고,
그 중 5개는 "조준한다"는 **기본 동작**을 굳이 이름에 적고 있었다. LAMP DSL 은 반대로 간다:
기본값은 안 적고 이탈만 표기한다.

**규칙 (D76)**: `[track_]<primitive>_<direction>[_<primitive2>_<direction2>][_dont_look]`.
`aim="look_at"`(매 프레임 subject 재조준)이 기본이라 이름에 없고, `_dont_look` 이 유일한
opt-out 이다. `pan`/`truck`/`pedestal` 은 primitive 가 aim 을 정하므로 접미사가 없다.

| D75 | D76 | aim |
|---|---|---|
| `dolly_in_aimed` / `dolly_out_aimed` | `dolly_in` / `dolly_out` | look_at |
| `dolly_in_locked` / `dolly_out_locked` | `dolly_in_dont_look` / `dolly_out_dont_look` | traj |
| `static_hold_aimed` | `static_hold` | look_at |
| `static_hold_locked` | `static_hold_dont_look` | traj |
| `track_hold_aimed` | `track_hold` | look_at |
| `track_hold_locked` | `track_hold_dont_look` | traj |
| `track_dolly_in_aimed` / `track_dolly_out_aimed` | `track_dolly_in` / `track_dolly_out` | look_at |

### 유일한 의미 역전 — `dolly_in` / `dolly_out`

D75 에서 이 두 문자열은 **`_locked` 의 별칭**(`aim="traj"`)이었다. D76 에서는 **정식
이름**(`aim="look_at"`)이 된다. 36개 중 뜻이 뒤집힌 건 이 둘뿐이고, 하필 디스크의 옛 뱅크가
옛 뜻으로 이 문자열을 들고 있다.

**측정한 것** (out/ + out_dynpose 전체 JSON 을 걸어서): `variants[]` 행 **105,329**개 /
336 파일. 그중 `("dolly_in","traj") 845`, `("dolly_out","traj") 846`. `aim="look_at"` 인
`dolly_in`/`dolly_out` 행은 **0개** — 즉 디스크의 이 문자열은 전부 옛 뜻이고, **모든 행이
`aim` 을 명시적으로 기록**하고 있다.

선택지:
1. 뱅크 파일을 전부 rewrite 해서 옛 이름을 새 이름으로 바꾼다.
   → **기각.** 336 파일이고, `fit_hole_ladder.py`(pid 3886714, screen infer5)가 `out_dynpose`
   를 **지금 쓰고 있다**. 돌아가는 생산자와 경합하는 rewrite 는 blast radius 가 너무 크다.
2. **(선택함) 디스크를 안 건드리고 `aim` 을 힌트로 받아 되돌린다.**
   `resolve_preset(name, aim)` + `LEGACY_AIM_COLLISIONS = {("dolly_in","traj"):
   "dolly_in_dont_look", ("dolly_out","traj"): "dolly_out_dont_look"}`.
   `aim="look_at"` 인 새 행은 그대로 통과한다. 모든 행이 `aim` 을 들고 있으니 정보 손실이 없다.
   **뱅크 행을 읽는 코드는 `aim` 을 반드시 같이 넘겨야 한다** — 안 넘기면 옛 행이 조용히
   반대 문구를 받는다. `build_bank_captions.py` 에 이름↔행 `aim` 불일치 assert 를 넣어
   틀린 매핑이 캡션으로 새지 않게 막았다.
3. `dolly_in` 을 아예 안 쓰고 `dolly_forward` 같은 새 이름을 만든다.
   → 기각. 충돌은 피하지만 영화 어휘를 버린다. 캡션 품질이 이름에 직결된다.

### 상류 LBM 어휘는 rename 대상이 아니다

`fit/bank/expand_preset_variants.py` 의 `SHAPE_PRESETS`/`STATIC_PRESETS` 와
`scripts/trumans_to_lbm_demo.py` 의 이름들은 우리 `PRESETS` 가 아니라 원본 LBM
`VideoEngineer/video_runtime.py:15-33 PRESET_NAMES` 로 나가는 문자열이다
(`trajectory_plan.preset_name` → LBM `build_trajectory_plan`). 우리가 이름을 바꿔도 LBM 은
모르므로 **건드리지 않았다**. `PRESET_CLASS` 에는 새 이름 키만 **추가**했다(옛 키 유지).

⚠ **여기서 발견한 기존 결함 (D75 때 들어감, 미수정)**: `STATIC_PRESETS =
["static_hold_aimed", "static_hold_locked", "static_zoom_in"]` 중 `static_hold_aimed` 와
`static_zoom_in` 은 **LBM 에 없는 이름**이라 `build_trajectory_plan` 의 기본 lerp 로 떨어진다
— 정지 branch 를 못 타므로 그 변이들이 **실제로는 정지가 아닐 수 있다**. lerp 대표로 쓰는
`dolly_in_aimed`(SHAPE_PRESETS)는 같은 이유로 의도한 동작이라 문제 없다. 별도 과제.

### 검증 (전부 통과)

- alias 33종이 전부 `PRESETS` 키로 떨어지고, alias 키가 정식 이름과 겹치지 않는다.
- D75 이름 10종이 D76 이름으로 resolve 되고 **`aim` 이 보존**된다 (`_locked`→`traj`,
  `_aimed`→`look_at`). `LEGACY_AIM_COLLISIONS` 는 양방향 확인 —
  `("dolly_in","traj")→dolly_in_dont_look`, `("dolly_in","look_at")→dolly_in` 그대로.
- `PRESETS` 36 : `configs/caption_presets.json` 36, 키 집합 일치.
- 디스크 105,329 행 전부 resolve → 미지 preset 0 / 캡션 누락 0 / **aim 불일치 0**.
- camel·goat·parkour·snowboard `hole_bank_k6` 캡션 **1,180개 재생성 → 전부 bit-identical**.

## D94 — `hold` 는 "안 움직이고 재조준도 안 한다" — 조준은 `_look_at` 으로만 표기 (2026-09-01, 적용)

### 무엇을 바꿨나

`static_*` / `track_*` 네 이름이 **서로 맞바뀌었다.** 궤적은 넷 다 `T.hold(I, n)` 으로 그대로고
**바뀐 건 이름과 조준뿐**이다.

| D93 까지 | aim | → D94 부터 | aim |
|---|---|---|---|
| `static_hold` | `look_at` | `static_look_at` | `look_at` |
| `static_hold_dont_look` | `free` | `static_hold` | `free` |
| `track_hold` | `look_at` | `track_look_at` | `look_at` |
| `track_hold_dont_look` | `free` | `track_hold` | `free` |

이유는 사용자 지시 그대로다 — **`hold` 가 "재조준한다"를 뜻하면 안 된다.** D76 이 조준 표기를
비대칭으로(기본은 무표기, 이탈만 `_dont_look`) 정한 뒤 `hold` 만 그 규칙에서 어긋나 있었다:
"제자리에 있다"와 "조준을 안 바꾼다"가 둘 다 hold 인데 이름은 전자만 기본값으로 잡고 있었다.
D94 는 `hold` 를 **둘 다 고정**으로 좁히고 조준이 붙는 쪽을 `_look_at` 으로 명시한다. arc/orbit/
crane/s_curve 는 정의 자체가 subject 중심이라 여전히 접미사가 없다 (D76 비대칭 유지).

### 옛 뱅크를 다시 쓰지 않는 방법 — `aim` 으로 판별 (`dolly_in`/`dolly_out` D76 과 같은 장치)

이름 충돌이므로 `("static_hold","look_at")` 같은 (이름, aim) 쌍으로 되돌린다:

```python
LEGACY_AIM_COLLISIONS = {..., ("static_hold", "look_at"): "static_look_at",
                              ("static_hold", "traj"):    "static_look_at",
                              ("track_hold",  "look_at"): "track_look_at",
                              ("track_hold",  "traj"):    "track_look_at"}
PRESET_ALIASES = {..., "static_hold_dont_look": "static_hold",
                       "track_hold_dont_look":  "track_hold"}
```

물량은 track 쪽이 크다 — 배포 d77 뱅크의 `track_hold` 행 **3,282 개가 전부 `aim="look_at"`**
이다 (D93 까지 그게 이 preset 의 기본값이었다). `resolve_aim(name, recorded_aim)` 은 **raw 이름**을
`D90_FLIPPED` 에 대고 보므로 거기엔 옛 이름(`*_hold_dont_look`)만 남기고 새 `*_hold` 는 넣지
않았다. `STATIC_PRESETS` 에는 새 이름 4종 + 옛 이름 2종을 같이 둬서, resolve 전 문자열로
비교하는 경로(옛 뱅크 스캔)도 여전히 정지로 잡힌다.

### 같이 고친 것 — emit 경로가 `aim` 을 안 날랐다 (D94 가 드러낸 **기존** 결함)

`emit_bank.decision_from_variant` 는 `variant["preset"]` 만으로 decision 을 만들었고,
`build_poses.py:581 aim = resolve_aim(preset, trajectory.get("aim"))` 이 `None` 을 받아
**preset 의 *현재* 기본값**을 탔다. D94 처럼 뜻이 뒤집힌 이름에서는 조용히 다른 카메라가 나온다.
실측: `dyn_0__static_hold__hole0.1` 의 pose 재현 오차가 `aim` 을 안 넘기면 **6.574e-01**,
넘기면 **0.000e+00**. `make_decision(..., aim=None)` → trajectory dict → `decision_from_variant`
로 배선했다. `decision_fingerprint`(`build_poses.py:101-150`)는 `aim` 을 포함하지 않으므로 저장된
fingerprint 가 무효화되지 않는다 (`emit.py:169` stale-poses 가드 안 걸림).

### 검증

- (이름, aim) 19 조합 전부 정식 이름·디코드 aim·follow·tracking·static 판정이 의도대로
  (옛 `*_hold`/`*_hold_dont_look`/`*_hold_aimed`/`*_hold_locked`/`static`/`locked_off`/
  `track_follow`/`track_follow_locked` 포함).
- `out/snowboard/bank_d94` 166 변이 / `hole_bank_k6_d94` 196 변이 재생성. **`aim` 이 166/166,
  196/196 전 행에 기록**된다 — 그래서 track rename 은 뱅크 재빌드 없이 하위호환이다.
- 그 뱅크를 emit: `variants 196 중 196 내보냄 (필터 0 / 접힌 단 0 / path 0)`,
  `pose 재현 최대오차 0.000e+00`, `21<->49 왕복 최대오차 0.000e+00`.

⚠ **여기서 발견한 기존 결함 (D90 때 들어감, 미수정)**: `PRESETS` 43 vs
`configs/caption_presets.json` 40 — 캡션 없음 `dolly_in_look_at`, `tilt_up`, `tilt_down`,
`track_dolly_in_look_at` / preset 없음 `dolly_in_dont_look`, `dolly_out_dont_look`.
`presets.md` 의 40행 표도 D90 추가분을 아직 안 싣고 있다. 별도 과제.

## D115 — fix.md 승인분 4건을 한 번에 반영한 재굽기 (2026-09-02, 실행 중)

사용자 지시: "재굽기는 다 고치고 나서 **한 번만** 돌리자" / "기존 학습 arm 은 **보존만**" /
"vista 전체랑 trumans scene 1개 (여러 chunk) 돌려놔줘". 그래서 새 폴더
`bank_d115` + `hole_bank_k6_d115` 에만 쓰고 d99 는 하나도 안 건드린다.

### 바뀐 축 4개 (전부 러너에서 명시적으로 넘긴다)

| 축 | fix.md | 무엇 | 사용자 결정 |
|---|---|---|---|
| `S` 정의 | F2 | frame0 한 장 → **전 프레임 non-sky 점의 첫 카메라 거리 평균** (`points_first_cam`) | "sky 제외 valid point 를 전체 프레임에서 합쳐서 첫 카메라 center 로부터의 거리 평균" |
| `fit_tau` 대상 | F9 | SE(3) 로그 전체 → **sweep 고정 + 반경만** (`--orbit_fixed_sweep`) | 승인 |
| G1 소스 프레임 | F11 | 13 (vista fit) / 7 (tau) → **49** | "전체로 해줘" |
| orbit sweep 하한 | — | `--min_sweep_deg` 15 → **20** | "일단 20으로 놔둬보고 돌려보고 결정할게" |

**일부러 안 바꾼 것**: `--collision_time_match` 는 계속 off (F6 은 승인 목록에 없다 —
이번 뱅크의 차이를 위 4축으로 귀속시키려면 지금 켜면 안 된다). F1(제자리형 static 라우팅)은
사용자가 "적용하지 말아줘". `subject_visible_frac` 은 "일단 측정만".

### 그래프부터 다시 짓는 이유

게이트 임계가 전부 `S` 배율이다 (`behind_margin_frac·S` · `obb_clear_floor·S` ·
`min_ground_clear·S` · 가림 `0.02·S`). 디스크의 `scene_graph.json` **53편 전부**가
`scale.mode` 키 자체가 없는 옛 게이지라, 새 가드(`scene_graph.scale.assert_scale_mode`)가
전량에서 걸린다 — 의도한 동작이고, 재굽기 시작점이 뱅크가 아니라 그래프라는 뜻이다.
`cloud.npz` 도 meta 에 `S` 를 들고 있고 렌더러가 거기서 읽으므로(`lbm/render.py:70`) 같이
다시 만든다. 옛 그래프는 `scene_graph_pre_d115.json` 으로 한 번만 복사한다.

### camel 스모크 (전 단계 rc=0)

```
graph 2m29s   S 4.6713 -> 4.7190 (x1.010)   mode points_first_cam / stride 1
cloud   27s   42,937,870 점 / dynamic 5,602,102   S 4.7190
tau   6m26s   enumerated 689  selected 689
fit  ~40m     변이 554  렌더 9,983  게이트 선차단 232
emit    24s   pose 재현 최대오차 0.000e+00 / 21<->49 왕복 0.000e+00
```

`bank.json` 확인: `fixed.orbit_fixed_sweep=true`, `fixed.min_sweep_deg=20.0`,
`fixed.scale_mode="points_first_cam"`, `collision.src_frames=[0..48]`(49장),
`collision.time_match=false`.

**d99 → d115 (camel 554변이, 같은 preset 어휘):**
```
binding    d99   hole 266  obb 86  none 57  approach 47  ground 47  elev 33  (빈)14  collision  4
           d115  hole 285  obb 60  none 54  ground  49  approach 42  elev 34  (빈)14  collision 16
orbit/arc 188변이 sweep_deg   d99  median 15.00  min 15.00  max 45.00
                             d115 median 20.00  min 20.00  max 45.00
```
collision 4 → 16 이 F11(13→49)의 직접 효과다. obb 86 → 60 은 그만큼 앞 게이트(G1)가
먼저 잡아간 것.

### 실행

`exec/_legacy/run_k6_d115_shard.sh`(vista) / `exec/_legacy/run_trumans_d115_shard.sh`(TRUMANS).
vista 51편(camel 제외) 4샤드 = screen `bake1..bake4`, GPU 3/4/5/6.
TRUMANS `tru_1d076f8c` 20 chunk 1샤드 = screen `bake5`, GPU 7.
단계마다 마커(`.graph_s115` / `.cloud_s115`)와 산출물 존재 검사가 있어 재시작이 안전하다.

---

## 차기 뱅크(d157 후속) — 사용자 요청 기록 (2026-09-08, **기록만 / 미실행**)

사용자 지시 그대로: *"현재 dynpose 돌리는게 다 돌아가면 preset 개수를 확 줄이고 tau 구간도
1~4라면 2, 4 정도만 적용하고 싶고 scene 마다 dd는 하나씩 정도만 fit하고 싶은데 일단 기록만 해놔줘."*

**착수 조건: D157 875편 4샤드가 전부 끝난 뒤.** 지금 돌고 있는 굽기는 손대지 않는다.

### 손잡이 3개

| # | 요청 | 현재 d157 값 | 걸리는 곳 |
|---|---|---|---|
| K1 | preset 개수 대폭 축소 | anchor 당 슬롯 14종 (코퍼스 전체 라벨 23종 = 좌우 미러 포함) | `route_presets.py` 슬롯 표 / `--nodes` 로 넘어가는 preset 합집합 |
| K2 | 사다리 4단 중 **2·4단만** | `fit --hole_ladder 0.10 0.20 0.35 0.50` (4단) → 2·4단 = `0.20 0.50`. τ 쪽은 `tau --tau_ladder 0.10 0.20 0.35 0.60 1.00` 로 **5단**이라 "1~4" 와 안 맞는다 — 4단짜리는 fit 사다리 하나뿐 | `configs/bank/<차기>.json` 의 `fit.args` |
| K3 | scene 당 `dd` 하나만 fit | **d157 은 `--num_external 0` 으로 dd 를 전부 껐다** (config `_from_d149` ④: "DataDoP 외부 모양(dd_*) 제외, 라우팅 preset 29 → 14"). 실측으로도 201편 뱅크에 `dd_` preset 0행 | `route_presets.py:291-294`, `--external_shapes` + `--num_external` |

**K3 은 확인이 필요한 항목.** "dd" 를 DataDoP 외부 모양(`dd_` / `datadop:<slot>`)으로 읽었다.
그 읽기가 맞다면 요청은 "d157 에서 껐던 dd 를 **scene 당 1개만** 되살린다"가 된다. 그런데
`--num_external` 은 **anchor 당** 개수라(`route_presets.py:388` 기본 1, `:292` 가 node 루프 안),
`--num_external 1` 을 줘도 anchor 6개(dyn 3 + stat 3)면 scene 당 dd 가 6개가 된다.
scene 당 정확히 1개로 하려면 `pick_external` 을 scene 단위로 한 번만 호출하도록 **코드 변경이
필요**하다 (씨앗이 `f"{video}/{node['id']}"` 라 anchor 마다 다른 모양이 뽑히는 구조).

### 왜 이 세 손잡이인가 — 실측 근거 (D165)

- scene 당 단계별 중앙값: `graph 205s / cloud 28s / route 1s / tau 102s / fit 503s / emit 11s`
  → **fit 이 scene 총시간의 78%.** 줄일 데는 fit 하나다.
- fit 비용은 (anchor × preset × rung) 격자에 거의 선형이다. 332편 완료 뱅크 기준 scene 당
  카메라 mean 144.4 / median 144.
- 중앙값 scene 한 편의 fit 렌더 8,699장 내역: verify 2-pass 4,524 (52%) / 이분법 3,905 (45%) /
  `hole_static` 강제 240 (3%) / static 직접 probe 30 (0.3%).
  → **rung 을 4단 → 2단으로 줄이면 이분법·verify 가 같이 절반**이 된다 (K2 가 가장 큰 절감).
- preset 별 fit 시간(중앙값 scene 0e9abfc5, 총 460s): `dolly_out 62.5s / truck_right 61.2 /
  pan_right 58.7 / pull_out_arc_left 51.4 / orbit_right 40.6` 상위 5종이 274s = 60%.
  **어느 preset 을 자르냐에 따라 절감이 크게 갈린다** — 개수만 줄이지 말고 이 표를 보고 고를 것.
- p90 scene(1c52abbc, fit 1783s) 도 **순위가 같다**: `dolly_out 259s / pull_out_arc_right 241 /
  truck_left 228 / pan_left 197`. 즉 "무엇이 비싼가"는 scene 을 타지 않는다.
  · 단 **p90 의 절대초는 GPU 경합으로 부풀었다** — 13:30~14:00 구간이 d157 샤드3(같은 GPU 0,
    13:28 기동)과 겹쳤다. 점 개수는 43.8M vs 45.2M 로 3% 차이인데 프레임당 시간이 0.22 → 0.70 s
    로 3.2배가 된 것은 씬 성질이 아니라 경합이다. **비율만 읽고 절대초는 재측정할 것.**

### 아직 안 정한 것

- K1 의 목표 개수 (몇 종을 남길지). 위 preset 별 시간표 + 학습 쪽 preset 분포
  (`caption-fscore-dynpose-vs-vista-gap`: `track_*` 이 dynpose val 의 52%) 를 같이 보고 정한다.
- K2 를 τ 사다리에도 적용할지 (τ 는 5단이라 "2·4단"의 대응이 애매하다).
- 새 config 이름 / generation 태그.
