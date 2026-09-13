# Changelog (video_generation)

[Keep a Changelog](https://keepachangelog.com/) 규약. `camera_generation/latentcam/CHANGELOG.md` 와
별개다 — 이쪽은 vendored 비디오 생성 모델 6종을 돌리는 `tools/` 스크립트만 다룬다.
(`video_generation/tools` 는 `.gitignore:223` 로 git 추적 대상이 아니다. 이 파일은 로컬 기록용.)

## [Unreleased]

### Added
- **`fit_hole_ladder.py --pick_budget` + `emit_bank.py --picked_only` — 씬당 카메라 수를
  **사다리와 분리해서** 정하는 손잡이 (2026-09-13, D188 ③).** `--fallback_target 1` 이
  "카메라 1대"를 뜻하지 않는다는 게 파일럿 58편의 결론이다 (아래 Fixed 항목). 사다리는
  **어디까지 내려갈지**를 정하지 무엇을 내보낼지는 안 정한다. 그래서 사다리는 그대로 두고,
  다 돌고 난 뒤 씬당 `--pick_budget` 개를 고른다:
  · 정렬은 **`plan_tier` ↑ → 등급(usable 먼저) → `hole_fraction` ↑.** tier 가 맨 앞이라
    "움직이면 track+object-centric, 아니면 object-centric" 어휘 우선순위가 유지되고,
    fallback 은 **위층에 status 통과 행이 아예 없을 때만** 내려간다.
    ⚠ 초안은 등급을 먼저 봤는데 실측에서 바로 틀렸다 — 9ec42125 는 tier0
    `dyn_0__track_pull_out_arc_right` 가 **solved** 인데 `hole_over_budget`(0.3676 > 0.35)
    하나 때문에 tier1 평범 짝에게 졌다. 그 태그는 `--hole_mode excess` 에서 잘못된 잣대다
    (목표 `hole_static + Δ0.20` 은 `solved` 가 이미 집행했다) — **거짓 경보가 어휘 지시를
    덮은 것**이라 tier 를 앞으로 옮겼다. 파일럿 59편 재투사: 수율은 그대로 50편/0대 9편,
    tier [(0,20),(1,25),(2,5)] → **[(0,24),(1,22),(2,4)]**, track **40.0% → 44.0%**,
    hole 평균 0.3193 → 0.3296 (median 0.2962 → 0.3060).
  · **행은 하나도 안 지운다** — `picked` 열만 단다 (뱅크는 재고 목록, D39/D45).
    `emit_bank.py --picked_only` 가 그 열을 소비한다. 기본값 `--pick_budget 0` = 열이 빈 칸,
    옛 뱅크와 비트 동일. 열이 없는 옛 뱅크에 `--picked_only` 를 주면 0행이 된다.
  · 뱅크 JSON 에 `pick{budget, picked[], tiers[], grade[]}` 를 싣는다 — 뱅크만 보고
    "이 씬은 best-effort 밖에 없었다"를 알 수 있어야 한다.
  · d188 config 가 `fit: --pick_budget 1` + `emit: {"args": ["--picked_only"]}` 로 쓴다.
  · 파일럿 59편 투사: 카메라 **50편 × 1대 (0대 9편, 84.7%)**, 등급 usable 36 / best_effort 14,
    tier [(0,20),(1,25),(2,5)], track 비중 20/50 = 40.0%.
- **`chain_bank_rounds.py --wait_for` — 앞선 GPU 작업이 끝나야 기동 (2026-09-13, D188).**
  `rebake_scenes.py --wait_for` 와 같은 것. GPU 4장에 뱅크 프로세스가 2개씩 올라가면 실측
  55.9 GiB(피크 55,947 MiB, proc 당 ~28 GiB) 라 3번째 세대는 81.5 GB 카드에 안 들어간다 —
  "빈 GPU 가 보이면 띄운다"가 아니라 **앞 작업이 끝나야** 띄운다. 기본값 빈 문자열 = 안 기다림.
- **`configs/bank/d188_dynpose100k_track_objcentric.json` + `route_presets.py --track_pair` /
  `--slot_rotate` — 씬당 카메라 1대를 **object-centric 어휘로 못박은** 세대 (2026-09-13, 사용자
  지시 "씬당 카메라 하나가 나오되 움직이는 dynamic anchor면 track+object centric 아니면
  object centric으로 나오도록 fallback도 넣어서 돌리는거야. 이렇게 10k개 카메라 맞춰줘").**
  D187 을 대체한다 — D187 은 preset 화이트리스트가 없어 d185 어휘를 그대로 물려받았고,
  나오는 1대가 `dolly_out`/`pan_*` 같은 aim=free 인 경우가 절반 이상이었다.
  · **어휘**: object-centric = `lbm/presets.py` 의 `aim="look_at"`. 슬롯은 `arc,orbit,vertical`
    3종. `static` 은 뺐다 — `static_look_at` 은 손잡이가 없어 status 가 항상 `static` 이고
    `solved` 가 구조적으로 안 나온다 (tau/fit 시간만 먹는다). `advance` 도 **실측으로 뺐다** —
    `dolly_in_look_at`/`track_dolly_in_look_at` 의 solved 가 d188 파일럿 59편 **0/165**,
    d185 400편 **0/505** 로 세대·라우팅과 무관하게 0이다 (탈락의 절반이
    `obb_limited`+`approach_limited`: 전진하며 조준을 유지하면 subject OBB 와
    `--min_subject_visible 0.6`/approach 여유를 동시에 만족하는 구간이 안 남는다).
    슬롯별 solved (파일럿 59편): `pull_out_arc_left` 77.5 / `track_pull_out_arc_left` 69.0 /
    `track_pull_out_arc_right` 65.1 / `pull_out_arc_right` 57.9 / `orbit_right` 35.0 /
    `track_orbit_right` 31.0 / `track_orbit_left` 25.6 / `orbit_left` 22.8 /
    `crane_up` 11.8 / `track_crane_up` 11.1 / `dolly_in_look_at` 0.0 %.
  · **`route_presets.py --emit_route` (신규, 기본 off).** `--emit args` 는 `--preset_route <out>`
    을 `grid2x2` 이거나 `--free_moving` 이 켜졌을 때만 붙였다. d188 은 `--slot_plan full` +
    `--free_moving off` 라 둘 다 거짓 → `sample_camera_bank.py:897` 의 `target_count` 가 0 →
    `plan_variants` 가 tiers 를 `{}` 로 돌려줌 → `fit_hole_ladder` 의
    `fallback_on = ... and any(tier_of.values())` 가 False. 즉 **예산도 사다리도 통째로 꺼진 채**
    fit 이 (anchor × preset) 격자를 다 돌았다 — 1차 파일럿 58편이 씬당 최대 9대(solved 175행),
    track 비중 42.9% 로 나온 원인이다. `--emit_route` 가 그 경로를 연다.
  · **`--track_pair` (신규, 기본 off).** `--track_mode replace` 는 슬롯을 track 판으로
    **갈아끼우기만** 해서 비-track 짝이 후보 풀에서 사라진다 → `--fallback_ladder` 를 켜도
    1층(비-track 조준)이 비어 추종 실패가 다른 추종으로만 떨어진다 (같은 anchor 변위 = 같은 벽).
    `--track_pair` 는 갈아끼운 track 슬롯의 평범한 짝을 backfill 에 같이 실어 그 1층을 채운다.
    `variant_tier` 가 비-track look_at 을 1, track 을 2로 매기므로 사다리가 자동으로
    [track(0) → 평범(1) → 남은 track(2)] 이 된다. 정지 anchor 에는 `track_` 이 안 붙어 no-op.
  · **`--slot_rotate` (신규, 기본 off).** 예산이 1이면 `plan_variants` 가 뽑는 건 `pool[0]`
    하나인데 `route()` 의 슬롯 순서가 고정이라 화이트리스트 뒤 **가장 앞 슬롯**이 코퍼스
    전량에서 같아진다. 400편 실측: 회전 없으면 `advance` 계열 60.8% / `vertical` 0건,
    회전하면 crane_up 22.5 / dolly_in_look_at 22.5 / orbit_* 23.0 / pull_out_arc_* 18.2 %
    (이 측정은 `advance` 를 빼기 전 4슬롯 기준이다).
    grid2x2+substitute 로는 안 고쳐진다 (keep_pair 안에서 다시 route 순서로 정렬된다).
  · **`--retry_status` 확대.** d187 파일럿 40편에서 solved 없이 끝난 7편 중 5편이 **행 1개**였다
    (`approach_limited` 3 / `obb_limited` 1 / `collision_limited` 1) — `variant_usable()` 이
    `clamped_low` 만 재시도 대상으로 보고 `*_limited` 를 "쓸 만함"으로 세어
    `--fallback_target 1` 에서 첫 행에 멈췄고 사다리를 한 층도 안 내려갔다. d188 은 비-solved
    어휘를 전부 올린다 (뱅크 행은 그대로 다 남으므로 코퍼스 `--drop_status` 와는 독립이다).
  · anchor 예산은 2 (`--max_anchors 2`) — `--max_anchors 1` 의 천장이 75.5% 다 (d185 census
    2,283편: solved≥1 1,983편 중 dyn_0 이 1,723편). fit 이 첫 solved 에서 멈추므로 emit 은 1대.
  · 격리: `bank_d188` / `hole_bank_d188` / `preset_route_d188.json`. d185 는 손대지 않는다.
- **`configs/bank/d187_dynpose100k_single.json` — D184 게이트 수정판, 씬당 카메라 1개
  (2026-09-13, 사용자 지시 "d184 게이트를 먼저 고치자"). ⚠️ 파일럿 40편에서 멈췄고 **D188 이
  대체**한다 — 어휘가 object-centric 이 아니었다 (위 항목).** D184 는 "씬당 1대"라는 목표는
  지켰지만 수율이 **10,273편 중 solved 715행(7.0%)** 이라 코퍼스가 안 됐다. D187 은 목표를
  그대로 두고 **게이트만** 고친다 — d185(grid5) 라우팅을 글자 그대로 쓰고 **예산만 1로** 잡는다
  (`route --target_variants 1` + `fit --fallback_target 1`).
  · **수율 해부 (route 만 CPU 로 300편 ablation):** `--anchor_min_drift_u 0.30` 하나가
    앵커 생존 46.0% → 96.3% (**+42.3 pt**) 로 범인이고, `--anchor_require_frame0` 이 +8.0 pt,
    `--slot_whitelist orbit` 은 후보 풀을 8개(본슬롯 2 + backfill 6)에서 1개로 줄인다.
    `--max_static_anchors` 는 `--max_anchors 1` 에서 `pick_main_anchors` 가 동적 우선이라 효과 0
    (그래서 d184 의 `0` 은 무해했다).
  · **`--fallback_ladder` 가 d184 에서 no-op 이었다.** `fit_hole_ladder.py:790-794` 는 τ 뱅크 행의
    `plan_tier` 가 전부 0이면 `fallback_on=False` 로 떨어지고, `plan_tier` 를 싣는 것은
    `sample_camera_bank.py --variant_pool full` 이다. d184 는 `tau` 블록이 없어 d179 를
    물려받았고 d179 tau 에는 그 플래그가 없다 → 씬당 시도가 문자 그대로 한 번.
    d187 은 `tau` 에 `--variant_pool full` 을 명시한다.
  · **"후보를 늘리는 것"과 "카메라를 늘리는 것"은 다른 축이다.** route 는 최대 9개(본 2 +
    backfill 6 + free 1)를 제안하지만 fit 이 쓸 만한 변이 1개에서 멈추므로 emit 은 1대다.
    d184 는 이 둘을 같이 1로 묶어서 죽었다.
  · 2026-09-11 지시("일정 이상 움직이는 dynamic 만 tracking")는 `--track_min_drift_u 0.05` 가
    집행한다 — d184 가 이를 앵커 **생존** 문턱으로 한 번 더 집행한 것이 과잉이었다.
  · 격리: `bank_d187` / `hole_bank_d187` / `preset_route_d187.json`.
- **`scripts/rebake_scenes.py` — 지정 씬만 마커 무시하고 graph/뱅크 재굽기 (2026-09-13, D186).**
  코드 버그를 고친 뒤에는 "이미 구웠다"는 마커가 적이 된다. 마커를 **지우지 않고**
  `run_bank.py --no_skip_done` 으로 우회하고, 뱅크 산출물은 지우는 대신
  `<work>/quarantine/<gen>/<video>/` 로 **옮긴다**.
  · 옮겨야 하는 이유: `skipped.json` 과 `canonical/canonical.json` 은 **서로를 지우지 않는다**.
    예전에 "앵커 0"으로 `skipped.json` 이 남은 씬이 이번엔 앵커를 얻으면 `canonical.json` 만
    새로 생기고 `skipped.json` 은 그대로다 — 반대도 마찬가지. `baked()` 숫자는 맞는데 한 씬이
    두 결론을 동시에 들게 된다.
  · `--wait_for <ps 문자열>`: 그 문자열을 명령줄에 가진 프로세스가 전부 끝나면 시작한다.
    선행 재굽기와 GPU 를 안 겹치게 하는 용도.
  · `--no_quarantine`: **드라이버가 도는 세대**에 쓴다. 격리하면 그 씬들이 드라이버의 ready
    집합으로 되돌아와 두 프로세스가 같은 씬 디렉토리를 동시에 쓴다. 제자리 덮어쓰기면
    `canonical.json` 이 계속 존재해 드라이버가 계속 baked 로 보고 안 집는다.
  · GPU 는 `GPUS = ["0","1","2","3"]` 고정 (CLAUDE.md).
- **두 세대를 동시에 굽기 — `chain_bank_rounds.py --require_bank_dir` + `--work` lock
  (2026-09-12, 사용자 지시 "gpu 활용률이 저조한데 버전 나눠서 카메라 5개짜리 버전도 같이
  돌려줘").** D182 graph(24샤드) + D184 뱅크(8샤드)가 도는데 GPU 는 0% 와 54 GB 사이를
  오간다 — route 와 cloud 전처리가 CPU 라 GPU 가 구간마다 논다 (124코어 중 load 44).
  · `--require_bank_dir <다른 세대 bank_dir>`: **그 세대가 끝낸 편만** ready 로 본다.
    두 체인이 같은 편을 동시에 집으면 강등(`dynamic_mask/*.png`)을 한쪽이 쓰는 중에 다른
    쪽이 읽는다. 선행 체인은 라운드마다 `강등 -> 뱅크` 순서라, 그 뱅크가 끝난 편은 강등이
    이미 끝나 있다 — 뒤따르는 쪽은 `--stage bank` 로 강등을 건너뛰어도 같은 마스크를 쓴다.
  · 종료 조건도 같이 고쳤다 (`more_coming`): 선행 세대가 전량을 안 끝냈으면 `ps` 에 샤드가
    잠깐 0개여도(선행의 강등 구간) 기다린다. 안 그러면 조기 종료한다.
  · `<work>/chain.lock` (PID): 같은 `--work` 에 드라이버가 이미 돌면 기동하지 않는다.
    screen 창 입력 버퍼에 기동 명령이 큐잉된 채 남아(앞 작업이 foreground) 나중에 자동
    실행될 뻔한 사고를 프로세스 쪽에서 막는다. 죽은 PID 의 lock 은 뺏는다.
  · 기본값은 둘 다 옛 동작 (`--require_bank_dir ""`; lock 은 단일 드라이버면 무영향).
  · **동시 실행 시 샤드 합계를 GPU 수로 나눠 볼 것.** `run_bank.py:159 _run()` 이 `--gpu`
    없으면 `fan_out` 의 `GPUS[shard % 4]` 를 쓰므로 8+8 샤드는 장당 4 다 — 네 장 전부
    75~80 GB(상한 81.5)에 닿아 CUDA OOM 이 났다. D185 를 4샤드로 내려 합계 12(장당 3),
    피크 77 GB, OOM 0건. 근거·증상은 `FIX.log` 2026-09-12 항목.

- **D185 — grid5 (씬당 카메라 5개) 전량 뱅크 config (`configs/bank/d185_dynpose100k_grid5.json`)
  (2026-09-12, 사용자 지시 "anchor 2 (static, dynamic) x preset 2 (object-centric or
  tracking(dynamic이 일정 이상 움직이는 경우만)) + free-moving 하나해서 scene당 카메라 총
  5개 ... fallback도 추가했었던 것들 돌리도록").** D166/D169 의 grid5 라우팅을 D182 graph 위
  10,346편 전량에 건다. **D184(씬당 1대)가 끝난 뒤** 도는 다음 세대다 — anchor 2 × 슬롯 2 = 4
  + free-moving 1 = 씬당 최대 5 카메라. route 는 `--max_dynamic_anchors 2
  --max_static_anchors 2 --max_anchors 2 --min_anchor_sep 2.0 --slot_plan grid2x2
  --slot_pair_fill substitute --free_moving rotate --track_min_drift_u 0.05
  --target_variants 5 --skip_if_empty`, fit/emit 은 d179 를 글자 그대로 물려받는다.
  · anchor 구성은 **d169 그대로(동적 우선)** 로 사용자가 확정했다. `pick_main_anchors` 가
    dyn-first 라 D182 graph 600편 표본에서 (dyn,stat) = (2,2) 381 / (2,1) 104 / (2,0) 62 /
    (1,2) 31 / (1,0) 11 / (0,2) 6 / (1,1) 5 → **dyn≥2 가 91.2%** 로 채워진다. stat anchor 는
    동적 노드가 1개뿐인 씬에서만 들어온다 (dyn≥1 ∧ stat≥1 인 씬은 86.8%). 이걸 알고 고른 배선이다.
  · **`--fallback_ladder` 는 tau 의 `--variant_pool full` 없이는 no-op 이다**
    (`fit_hole_ladder.py:791,868` 이 τ 뱅크의 `plan_tier` 를 읽는데 `budget`(기본) 이면 층이
    전부 0). d179/d183 은 fit 에만 `--fallback_ladder --fallback_target 5` 가 있고 tau 에
    그 인자가 없어서 fallback 이 **한 번도 안 돌았다**. D185 tau 에 `--variant_pool full` 을
    넣어 고친다 — τ 뱅크는 약 2배가 되지만 fit 은 예산(5)이 차면 멈춘다.
  · 슬롯은 `GRID_ALLOWED_SLOTS` = advance/recede/arc/orbit/vertical/static — **s_curve 는
    없다**(사용자 지시 2026-09-08). track 여부는 `--track_min_drift_u 0.05` 가 정한다:
    문턱을 넘는 anchor 의 `TRACK_KEEP_SLOTS` 에만 `track_` 이 붙고, 못 미치면 같은 슬롯을
    비-track 으로 굽는다 (= "dynamic 이 일정 이상 움직이는 경우만 tracking").
  · 격리: `route.out` = `preset_route_d185.json`, 세대 디렉토리 `bank_d185`/`hole_bank_d185`.
    강등은 D184 라운드에서 `.graph_d182` 기준으로 이미 끝나 있다.

- **D184 — 씬당 카메라 1개 전량 뱅크 (`configs/bank/d184_dynpose100k_single.json`)
  (2026-09-12, 사용자 지시 "내가 카메라 개수 제한하라고 하지 않았니?").** 2026-09-11 지시
  ("track + object-centric ... 카메라 하나만 fitting되도록")가 `d181_track_orbit_pilot.json`
  **파일럿에만** 반영돼 있었다. D183 은 d179 전량 라인을 그대로 물려받아 route 가
  `--max_dynamic_anchors 3 --max_static_anchors 3`, fit 이 `--fallback_target 5` 라
  **편당 26.49 카메라**(중앙값 24, anchor 2.56/편 × 10.37 카메라/anchor, 371편 실측)를 냈다.
  D184 는 D181 의 제약을 D182 graph 위 10,346편 **전량**에 건다: route 에
  `--max_dynamic_anchors 1 --max_static_anchors 0 --max_anchors 1 --anchor_min_drift_u 0.30
  --anchor_require_frame0 --slot_whitelist orbit --track_mode replace --target_variants 1`,
  fit 에 `--fallback_target 1`. tau args 는 d183 과 **글자 단위 동일**(병합 실행해 확인) —
  τ 사다리·조준·추종을 바꾸면 코퍼스가 파일럿의 상위집합이 아니게 된다.
  · D183 은 371편에서 중단했다 (`hole_bank_d183` 은 남겨 뒀다 — 삭제는 승인 뒤).
  · 스모크 16편: `SKIP(앵커 0) 10 / OK 6`, 1.0분. 구워진 7편 전부 **카메라 정확히 1개**,
    preset 은 `track_orbit_left` 5 / `track_orbit_right` 2. 편당 51.1s
    (route 0.2 / tau 22.0 / fit 24.4 / emit 4.5) — d183 341s 대비 6.7배 빠르다 (카메라 수).
  · **수율은 씬당 1대 × 10,346 이 아니다.** anchor 가 1개뿐이라 `--anchor_min_drift_u 0.30`
    이나 `--anchor_require_frame0` 에 걸리면 씬이 통째로 빠진다. 스모크 37.5%, D181 파일럿
    이동량 문턱 통과율 45.9%(2,137편 중 981편) → 카메라가 나오는 편은 4~5천 편으로 본다.
  · `--max_static_anchors 0` 이라 D182 가 새로 넣은 stat 노드는 이 세대에서 target 이 아니다.
    `--composition` 의 구도·가림 판정에는 그대로 들어간다.

- **D183 — D182 graph 위의 뱅크 config (`configs/bank/d183_dynpose100k_bank.json`)
  (2026-09-12, 사용자 지시 "앞으로 돌릴 것도 배선 문제 없는지 미리 확인해놔줘").** d179 를
  D182 뒤에 그대로 돌리면 **두 군데서 조용히 잘못된다**. ① `require_markers: [".graph_d157"]`
  — D182 는 `.graph_d182` 를 쓰므로 3,423편만 통과하고 6,923편은 편마다
  `FAIL(마커 .graph_d157 없음)` 인데 샤드 rc 는 0 이다 (완주한 것처럼 보이는 2/3 빈 코퍼스).
  ② `bank_dir: hole_bank_d179` — `run_bank.process_video:210` 이 `canonical/canonical.json` 을
  보고 `SKIP(done)` 하는데, 그 753편 + `skipped.json` 40편이 **dyn-only graph 로 라우팅된**,
  정확히 다시 구워야 할 편들이다. 그래서 `extends: d179...` + 세대 디렉토리
  (`bank_d183`/`hole_bank_d183`) + 마커만 바꿨다 — route/tau/fit/emit args 는 `load_config`
  얕은 병합으로 d179 와 **글자 단위 동일**임을 실행해 확인했다.
  · 강등(`dynpose_dynamic_mask_from_seg.py --demote_static_objects`)은 D182 뒤에 10,346편
  전량 재실행해야 한다 — 디스크의 강등 323편(`from_seg.json` 1,934편 중)이 dyn-only graph
  기준이다. 이 스크립트는 매번 `seg_instances/masks.npz` 에서 합집합을 다시 만들어(`build_union`)
  **멱등**이므로 재실행으로 복구된다.
  · D182 뒤에도 stat 노드가 0개로 남는 편이 있다: 정적 명사 0개 350편 + `vlm_nouns`
  `source="exhausted"` 7편. sam3 완료분 실측 4.3% 가 stat track 0개다.
  · `d181_track_orbit_pilot.json` 은 `extends: d179...` 라 같은 `.graph_d157` 을 물려받는다 —
  되살릴 때 `require_markers` 를 같이 고칠 것.
  · (2026-09-12) 그 강등 재실행을 `tmp/d182/chain_static_graph.py --stage demote` 로 배선했다.
  `dynpose_dynamic_mask_from_seg.py` 에는 `--num_shards` 가 없어 드라이버가 목록을 쪼개
  프로세스 `--demote_shards`(기본 8) 개로 띄운다. CPU(numpy+PIL) 전용이라 GPU 를 안 잡고,
  `.graph_d182` 가 `--min_ratio` 아래면 굽지 않는다 (graph 뒤·route 앞 순서 강제).
  dry-run 20편 실측 0.35 s/편, 강등 4노드/20편 · 픽셀 뒤집힘 평균 0.85%.
  · (2026-09-12, 사용자 지시 "gpu 안놀게 bank랑 묶어서 해줘") `tmp/d179/chain_bank_10k.py` 를
  세대 상수만 인자로 뽑아 **`scripts/chain_bank_rounds.py`** 로 승격하고, graph 가 도는 동안
  끝난 편부터 주워 강등→뱅크를 굽도록 배선했다. graph 단계는 순수 CPU 라(`build_scene_graph.py`
  는 `nvidia-smi --query-compute-apps` 가 빈 목록) graph 만 돌리면 GPU 4장이 전부 논다.
  뱅크의 tau/fit 은 `--device cuda` 라 겹쳐 돌리면 CPU 는 graph, GPU 는 뱅크가 쓴다.
  · **강등 재실행 판정을 mtime 으로 고쳤다.** d179 는 `from_seg.json` 의 `demote_static_objects`
  필드만 봐서 "예전(dyn-only) graph 로 강등된" 323편을 건너뛴다 — D182 처럼 graph 를 다시 구우면
  그 판정이 낡는다. 이제 `from_seg.json` 이 `.graph_d182` 보다 오래되면 다시 강등한다.
  · 스모크 2편 rc=0 (`route 0.2-0.3s / tau 109.0-111.2s / fit 242.5-247.1s / emit 5.7-6.1s`,
  편당 357.5·364.7s). GPU 점유 26,598·28,336 MiB/프로세스 → 80 GiB H100 1장당 2개 = 8 샤드.
- **D182 — dynpose-100k graph 를 정적 노드 포함으로 전량 재굽기
  (`configs/bank/d182_dynpose100k_graph.json`) (2026-09-11, 사용자 지시 "우선 static 포함해서
  graph까지 먼저 다 완료하는걸 목표로 다시 돌리자").** D172 가 구운 graph 에 `stat_*` 노드가
  하나도 없었다. 플래그가 아니라 **디렉토리 부재**다 — `scene_graph/io.py:99` 는
  `((seg_root,"dyn"), (seg_static_root,"stat"))` 를 돌면서 없는 루트를 **조용히 건너뛴다**
  (에러도 경고도 없고 `scene_graph.json` 은 정상으로 보인다). `DATA/DynPose-LBM` 에는
  `eval_data/seg_instances_static` 이 있어서 d157 은 stat 노드를 냈고 `DATA/DynPose-100K` 에는
  없어서 d172 는 안 냈다. 같은 코드, 다른 결과.
  · marker 를 **`.graph_d182`** 로 바꿨다. `.graph_d157` 을 그대로 쓰면 이미 있는 2,743편을
  건너뛰는데 그게 정확히 다시 구워야 할 dyn-only 편들이다 (d172 `_marker_reuse` 가 경고한
  "입력이 바뀌면 재사용 무효" — args 는 그대로지만 **입력에 정적 seg 루트가 새로 생겼다**).
  `graph.args` 는 d172 와 글자 단위로 같다.
  · 스모크 1편(`0013e08a`, 정적 명사 5개) rc=0 253.3s — 노드 **16개(stat 10 / dyn 6)**,
  `dyn_1 supported_by=stat_1`, `near` 엣지가 stat 쪽으로 붙는다. 같은 코퍼스 dyn-only 실측은
  d172 샤드 로그 n=2,446 에서 **p50 163.7s / mean 205.3s / p90 380.2s / max 4326.1s** 이므로
  정적 노드가 붙어 늘어난 폭은 mean 대비 **+23%** 다 (스모크는 같은 카드에서 SAM3 12샤드가
  동시에 돌던 중이라 상한값). d172 config `_runtime` 의 "85.5 s/편" 은 **다른 코퍼스에서 받아온
  주석이고 이 10,346편에서 측정한 값이 아니다** — 그 수치로 비교하면 3배 느려진 것처럼 보인다.
  · 선행 단계는 `tmp/d182/chain_static_graph.py --stage nouns|link|sam3|graph` 가 돌린다
  (tmp 는 git 미추적). nouns = 16 샤드 `vlm_nouns.json` 병합 후 `extract_static_nouns.py`
  → 10,339/10,346 (7편은 VLM `source="exhausted"` 라 record 가 없고, `metadata.csv` 에
  `prompt` 열이 없어 규칙 폴백도 불가). link = `DynPose-LBM` 정적 seg 846편 심볼릭 재사용.
  sam3 = 잔여 ~9,493편.
- **D180 — `run_bank.py --exec inproc`: 단계를 서브프로세스가 아니라 같은 프로세스에서 부른다
  (2026-09-11, 사용자 지시 "뱅크 더 빨리는 못함? 불필요한 방식 없는지 봐줘").** 8샤드 구간 실측
  씬당 205.8s(FIT 135.3 / TAU 62.7 / EMIT 7.4 / ROUTE 0.4) 중 **51.3s(25%)가 프로세스 경계
  비용**이다 — import 21.9s(route 0.45 + tau 7.27 + fit 6.74 + emit ~7.4)를 씬마다 다시 내고,
  tau 와 fit 이 **같은 cloud 를 각각** 짓는다(load_recon 4.74 + preprocess 9.37 + build 0.60
  = 14.7s ×2). 샤드는 이미 한 프로세스가 씬을 순회하는 구조라, 단계 호출만 in-process 로 바꾸면
  import 가 **씬당이 아니라 샤드당 1회**로 떨어진다.
  · `scripts/run_bank.py` — `--exec {subprocess,inproc}` (**기본 subprocess = 기존 동작 그대로**),
  `--inproc_recycle 50`. 단계 스크립트를 `spec_from_file_location` 으로 모듈로 읽고
  (`__main__` 블록은 안 돈다) `main(build_parser().parse_args(argv))` 를 부른다. stdout/stderr 는
  기존과 같은 씬 로그로 리다이렉트. `KeyboardInterrupt` 는 **안 잡는다** — Ctrl+C 가 샤드의
  정상 종료 절차다. CUDA/OpenMP env 는 **첫 import 전에** 못 박고, `--inproc_recycle` 회 처리하면
  `execv` 로 자기 자신을 다시 띄워 누적 메모리를 턴다(`--skip_done` 이 이어받는다).
  · `scripts/{route_presets,sample_camera_bank,fit_hole_ladder,emit_bank}.py` — 파서를
  `build_parser()` 함수로 꺼냈다. `__main__` 안에 있으면 import 로는 만들 수 없다. CLI 동작 불변.
  · `lbm/render.py` — 씬 1편분 cloud+recon 메모(`set_cloud_cache()`/`clear_cloud_cache()`,
  **기본 꺼짐**). 캐시 키에 `source/video/out_root/vista4d_root/device/eval_data/seg_root/
  seg_static_root/preprocess/depth_outliers/ignore_sky_mask/allow_empty_dynamic_mask/scale_key`
  를 전부 넣는다. 히트해도 **렌더러는 항상 새로 만든다** — 안 그러면 `subject_mask` 가 샌다.
  · `sys.path` — `run_bank.py` 가 리포 루트를 직접 꽂는다. python 은 **스크립트가 있는 폴더**
  (`scripts/`)를 넣지 cwd 를 넣지 않아서, in-process 모드에서 드라이버가 직접 부르는
  `lbm.render`/`lbm.cloud` import 가 `ModuleNotFoundError` 로 죽었다 (파리티 B팔 rc=1).
  · **파리티 3편 최종 (GPU 1, 단계 시간 합, 초).** 결과는 안전하지만 **속도 이득이 없다.**

    | stage | A(sub) | B(inproc) | C(sub) |
    |---|---|---|---|
    | ROUTE | 1.1 | 0.0 | 1.0 |
    | TAU | 551.1 | 554.1 | 588.8 |
    | FIT | 1266.5 | 1210.5 | 1308.5 |
    | EMIT | 21.6 | 4.1 | 23.2 |
    | 합 | 1840.3 | 1768.7 | 1921.5 |

    B/A = 0.961, B/C = 0.920. 그런데 **A/C = 0.958** — 같은 설정 두 팔의 시간 차이가 inproc
    이득과 같은 크기다. 즉 이 표본에서 speedup 은 **실행 간 잡음과 구분되지 않는다**. EMIT 만
    21.6 → 4.1s 로 확실히 줄지만 절대량이 17s(3편)다. 위 25% 추정이 빗나간 이유: 단계 비용이
    프로세스 경계가 아니라 **실제 연산**에 지배된다 (FIT 이 3편에 1266s = 씬당 422s).
    → **라이브 d179 를 inproc 으로 재기동하지 않는다.** 재기동 비용이 이득보다 크다. 플래그는
    남겨두되 기본값 subprocess 유지.
  · **결과 동일성은 확인됐다(3편 전부).** `canonical.npz` 가 A~B·A~C **모두 max|Δ|=0 bitwise
  동일**. `bank.json`/`canonical.json` 은 셋 다 갈리지만 차이가 `subject_visible_frac` 한 필드에
  몰리고, 잡음 바닥(A~C) 2.73e-2 / 3.74e-2 / **5.72e-2** 대 inproc(A~B) 3.76e-2 / 3.95e-2 /
  3.84e-2 — 002fe46a 는 **잡음 바닥이 inproc 차이보다 크다**. inproc 이 결과를 바꾸지 않는다.
- **D180-b — d157 graph 372편 격리 + d172 로 재굽기 (2026-09-11).** d179 tau 가 하드 실패하던
  원인. 인과는 4단: ① 2026-09-06 `sam3_static_instances.py` 가 `DynPose-LBM/eval_data/
  seg_instances_static/` 846편을 만들고 d157 graph 가 그 위에서 `stat_*` 노드를 얻었다(372편에서
  중단). ② 새 10k 인제스트 드라이버 `scripts/dynpose_ingest.py` 에는 static 단계가 **없어서**
  (`grep static` 0건) `DATA/DynPose-100K` 는 `seg_instances_static/` 을 못 받았다 — "정적 물체
  target 보류" 결정과 일관된다. ③ **실제 오류**: `d172_dynpose100k_graph.json` 이 `.graph_d157`
  마커를 재사용하면서 근거로 든 "공통 877편 중 masks.npz 크기가 다른 건 1편뿐"이 `seg_instances`
  만 비교한 것이었다. **한쪽에만 있는 디렉토리는 그 비교에 구조적으로 안 보인다.** ④ d179
  (eval_data=100K)가 d157 의 `stat_*` 노드를 만나 `build_candidate_board.subject_track_volume`
  에서 assert.
  · 영향 실측: graph 2,251편 중 **372편(16.5%)** 오염. d179 가 손댄 478편 중 84편 하드 실패,
  25편이 **어긋난 segmentation 으로 통과**(23편은 canonical.json 까지 썼다).
  · 조치: `tmp/d180/quarantine_stale.py` 가 `.graph_d157` 372 + `hole_bank_d179` 29 +
  `bank_d179` 123 + `preset_route_d179.json` 123 을 `tmp/d180/quarantine/` 로 **옮겼다**(안 지웠다).
  `tmp/d180/regraph372.py` 가 같은 372편을 d172 config(eval_data=100K)로 4샤드 재굽기.
  · 재굽기 델타(측정): dyn anchor 1.98→1.90(사실상 불변), stat anchor 1.55→**0.09**,
  노드 11.59→4.92, canonical 카메라 수 불변(씬당 5 고정). 즉 코퍼스가 **dyn-only 로 균일**해진다
  (사용자 지시 2026-09-11 "일단 dyn-only로 해주고").
- **D181 — track + object-centric 씬당 카메라 1개 파일럿 (2026-09-11, 사용자 지시 "scene 중에
  일정 이상 움직이는 dynamic target 있는 경우만 모아서 track + object-centric (e.g. track orbit
  left) 같은 것만 카메라 하나만 fitting되도록 먼저 돌려줘봐").** 세 제약이 각각 다른 자리에 걸린다.
  · `scripts/route_presets.py --anchor_min_drift_u`(**기본 0 = 끔**) — anchor **후보 자체**를
  이동량(`center_drift_u`)으로 자른다. 기존 `--track_min_drift_u` 로는 안 되는 이유: 그건 이미
  뽑힌 anchor 에 `track_` 접두사를 붙일지만 정하고, anchor 선택은 여전히 `max_area_frac` 순서라
  **정지한 큰 물체가 1등으로 뽑히고** 이동량이 큰 물체는 상한 밖으로 밀린다. 캡을 걸기 **전에**
  필터해야 이동량 상위 노드가 면적 순위에 안 잘린다.
  · `scripts/route_presets.py --slot_whitelist`(**기본 "" = 끔**) — 라우팅된 슬롯 중 이것만 남긴다.
  preset 이름이 아니라 **슬롯 이름**으로 거는 이유는 방향(left/right)이 소스 카메라 횡이동에서
  나오는 씬별 값이라(`route()` 의 away/toward), preset 이름을 고정하면 소스와 같은 쪽으로 도는
  변이가 섞이기 때문. 거르는 자리는 `route()` **바깥**이다 — 안에서 거르면 bonus track 슬롯
  해싱과 `num_track` 집계가 화이트리스트에 따라 달라져 파일럿이 코퍼스의 부분집합이 아니게 된다.
  `reasons` 에 `slot_whitelist` / `slot_whitelist_dropped` 를 남긴다.
  · `configs/bank/d181_track_orbit_pilot.json` (**신규**, `extends: d179`) — anchor 1 × 슬롯 1 =
  씬당 카메라 1개. route `--anchor_min_drift_u 0.30 --slot_whitelist orbit --track_mode replace
  --max_anchors 1 --max_static_anchors 0 --target_variants 1`, fit `--fallback_target` 5→1
  (안 내리면 fallback 사다리가 1변이 씬에서 5개를 채우려 든다). tau 인자는 d179 를 글자 그대로
  물려받는다 — τ 사다리·조준·추종을 바꾸면 파일럿이 코퍼스의 부분집합이 아니게 된다.
  · `scripts/route_presets.py --anchor_require_frame0`(**기본 off = 옛 동작**) — `track.frames` 에
  프레임 0 이 없는 노드를 anchor 후보에서 뺀다. 뱅크가 anchor 를 **소스 frame 0 가시성**으로 한 번
  더 떨어뜨리는데(`skipped.json` 의 `no_surviving_anchors`), anchor 가 1개뿐인 이 파일럿에서는
  그러면 씬이 통째로 날아간다 — 첫 스모크 3편 중 **2편**이 그랬다(`track.frames` 가 13/2 에서
  시작). 이동량이 큰 물체일수록 나중에 프레임에 들어오므로 drift 문턱과 특히 겹친다.
  · 씬 목록 `tmp/d181/pick_moving_scenes.py` — 신선 graph 2,137편 중 `drift >= 0.30` **981편
  (45.9%)**. 문턱은 코퍼스 중앙값이고 기존 배선 문턱 0.05 의 6배다 (0.05 는 96.4% 를 통과시켜
  "일정 이상"이 안 된다). 분포: p25 0.169 / p50 0.308 / p75 0.538 / p90 0.866 / max 3.515.
  frame0 조건을 목록에도 같은 기준으로 걸어 1,110 → 981편. D180-b 에서 격리한 372편은
  제외(graph 재굽기 중).
  · route 스모크 3편 — `--nodes dyn_1 --presets track_orbit_left` / `dyn_0 track_orbit_right` /
  `dyn_0 track_orbit_left`. 방향이 씬마다 갈리는 것까지 확인.
  · 수율 스모크 13편(GPU 2) **OK 11 / 13** — 씬당 67.4~81.8s (d179 의 카메라 5개 굽기가
  443~815s). 팬아웃은 `tmp/d181/run_pilot.py`, 2샤드(GPU 2·3), screen `infer3`.
- **D181-b — `lbm/cloud.py warm_linalg()`: 점군 올리기 전에 cuSOLVER 핸들을 확보한다
  (2026-09-11).** D181 팬아웃 첫 기동에서 **3샤드가 전부 첫 씬에서** 죽었다
  (`cusolverDnCreate` → `CUSOLVER_STATUS_INTERNAL_ERROR`). 범인은 Vista4D `render_frame()` 첫 줄의
  `cam_c2w.inverse()` 인데, **4x4** 역행렬인데도 torch 가 cuSOLVER 경로를 타고 첫 호출에서 핸들을
  만든다. 그 핸들은 torch 캐싱 할당자 **바깥에서** cudaMalloc 하므로 점군(프로세스당 ~28 GiB)이
  카드를 채운 뒤에는 실패한다 — 4x4 역행렬이 메모리 부족으로 터지는 셈이라 스택만 보면 원인이
  안 보인다. `build_cloud()`/`load_cloud()` 가 업로드 **전에** `torch.eye(4).inverse()` 를 한 번
  때려 핸들을 미리 만든다 (프로세스·디바이스 단위 캐시라 1회면 충분, 실패해도 무시하고 진행).
  재기동 후 같은 두 씬(`00a516f1`, `007d34fc`)이 rc=0 으로 통과.
- **D180-c — `tmp/d180/parity.py` 판정을 md5 에서 수치 비교로 (2026-09-11).** 대조군 C 를 둔
  덕에 잡혔다: **A != C** 였다 — 같은 설정 subprocess 두 번인데 `bank.json` md5 가 갈린다. 실제
  차이는 `subject_visible_frac` / `near_depth` 의 소수 2~4째 자리뿐이고 정작 카메라 본체인
  `canonical.npz` 는 A·C 가 **bitwise 동일**. md5 로 판정했으면 in-process 전환을 무고하게
  기각할 뻔했다. `numeric_same()` 이 JSON/npz 를 leaf 단위 `max|Δ|` 로 비교하고(팔 이름이 박히는
  `source_bank` 는 제외, `nan==nan` 취급), A~C 를 **잡음 바닥**으로 같이 찍어 A~B 를 그 바닥과
  견주게 했다 — 기억 속 상수(±0.04)를 불러오는 대신 매 실행 실측한다.
- **D178 — tau/fit `--cloud_source memory`: cloud 를 굽되 디스크에 안 쓴다 (2026-09-11, 사용자
  지시 "cloud로 만들되 저장은 하지 않는거지").** `cloud.npz` 는 영상당 **1.4 GiB** 이고 dynpose
  447편에 **610.6 GiB** 다. 디스크를 거치는 유일한 이유는 tau(`sample_camera_bank.py`)와
  fit(`fit_hole_ladder.py`)이 **별개 프로세스**라 메모리를 못 넘기기 때문이지, 재구축이 비싸서가
  아니다. 실측(00e9f728, 44.6 M points, GPU 0): `build_cloud` 자체는 **0.60s**, 비싼 쪽은
  `load_recon_and_seg` 4.74s · `preprocess_recon` 9.37s · `save_cloud` 7.56s · `load_cloud` 6.46s.
  그런데 tau/fit 은 렌더러를 만든 직후 `load_scene` 으로 recon 을 **어차피 한 번 더** 읽고 있었고
  (`sample_camera_bank.py:855`, `fit_hole_ladder.py:706`), `S`/`z_med`/`parallax` 는
  `scene_graph.json` 의 `scale` 블록에 이미 있다. 그래서 in-memory 재구축의 실제 추가 비용은
  preprocess+unproject ≈ 10s 뿐이고, 대신 `cloud` 단계(굽기 28s + 1.4 GiB 쓰기 + 프로세스 기동)가
  통째로 사라진다.
  · `lbm/cloud.py` — `cloud_from_recon()` (이미 로드된 recon → `load_cloud` 와 같은 dict) +
  `assert_dynamic_mask_nonempty()` (D176-c 가드를 함수로 승격). 가드를 함수로 뽑은 이유는 npz
  경로와 in-memory 경로가 **같은 판정**을 써야 하기 때문이다 — 이 가드가 잡는 사고는 애초에
  에러 없이 통과하는 종류라 한쪽만 고치면 조용히 빠진다. `main()` 은 이제 이 함수를 부르고
  `save_cloud` 만 얹는다 (npz 산출물 불변).
  · `S`/`z_med`/`parallax` 를 **인자로 받는다** — 여기서 다시 재면 `scene_scale` 이 stride 1 에서
  6초 더 들고, 무엇보다 게이지가 graph 와 갈릴 수 있다. 호출자가 graph 의 `scale` 을 그대로 넘긴다.
  · `lbm/render.py` — `CloudRenderer(cloud_path=...)` 가 **dict 도** 받는다. 공통 진입점
  `add_cloud_source_args(parser)` / `open_renderer(args, out_root, graph) -> (renderer, recon)` 을
  두어 tau/fit 양쪽의 cloud 조달 + recon 로드 4블록을 한 줄로 합쳤다.
  · `scripts/{sample_camera_bank,fit_hole_ladder}.py` — `--cloud_source {npz,memory}`,
  **기본 `npz` = 기존 동작 그대로**. `memory` 면 config `stages` 에서 `cloud` 를 빼도 된다.
  · **파리티 검증** (00e9f728, d177 config 의 tau/fit 인자 그대로, GPU 0):
  cloud 텐서 `colors`/`points_world`/`visible`/`indices` **전부 bit-identical**(최대차 0.000e+00),
  meta 는 `parallax_ratio` 만 4.878e-09(npz float 직렬화 왕복). `bank.json` 103변이 중 86개가
  갈렸지만 **결정 열(`status`/`knob`/`preset`/`tau_*`/`hole_*`)은 0건**이고 갈린 7종은 전부 렌더
  측정 열이다. **npz 를 두 번 돌린 대조군**에서 같은 7종이 더 크게 흔들린다 —
  `subject_visible_min` 0.0278(npz↔mem) vs **0.0495**(npz↔npz), `subject_visible_frac`
  0.0233 vs **0.0347**. 즉 차이의 원인은 memory 경로가 아니라 렌더러 비결정성이다.
  시간도 동등: tau 301.9→302.7s, fit 613.0→607.9s (cloud 단계 ~40s 가 순이득).
- **D177 — dyn 트랙 중 "안 움직이는 소품"을 `dynamic_mask` 에서 빼는 강등 (2026-09-10, 사용자
  지시 "dynamic 중에 human, animal 같은걸 제외하고 object중에 이동량이 작은건 static으로
  만들어주는거 돌려봐줄 수 있어?").** VLM 이 "dynamic 명사"로 부른 것 중 주차된 차·벽 간판·상
  위 그릇처럼 실제로는 정지한 것이 많고, 그것들이 dynamic 으로 남으면 `unproject` 가 프레임당
  1장씩만 보이는 점으로 올려 다시점 누적을 못 한다 — 카메라가 움직이면 그 자리가 hole 이 된다.
  · `configs/noun_category.json` (**신규**, `noun_category_v1`) — 명사를 토큰 규칙으로
  `living`(사람·동물·신체부위) / `worn`(착용·소지물) / `object` 로 가른다. d157 계열 631편의
  dyn 노드 3,155개 / 명사 466종을 보고 손으로 분류했다. **living/worn 은 기본 보호**
  (`--demote_worn` 으로 worn 해제). 애매하면 보호 쪽으로 넣었다 — 잘못 보호하면 강등이 덜 될
  뿐이지만, 잘못 강등하면 움직이는 물체가 49겹 정적점으로 굳는다.
  · `scripts/dynpose_dynamic_mask_from_seg.py` — `--demote_static_objects`(**기본 꺼짐**, 켜야
  달라진다). 판정은 **각변위** `center_drift_u / d_ref < 0.05` **와** `path_len_u / d_ref < 0.15`
  둘 다. scene scale 단위(`center_drift_u`)만 보면 카메라에서 먼 물체의 화면상 정지를 못 본다
  (D174: 같은 0.06u 가 `d_ref` 1.44 에서는 각변위 0.042). 경로길이를 같이 보는 것은 왕복·제자리
  회전 보호용. `merged_from` 의 `"dyn#N"` 형제 track 까지 같이 뺀다 (`instances.py:297`).
  · 강등은 `dynamic_mask` **한 곳에만** 걸고 `scene_graph.json` 의 `kind`/`moving` 은 안 건드린다
  — `moving` 은 anchor split key 라 강등하면 dyn 앵커가 0 인 씬이 생긴다
  (`build_scene_graph.py:156` 의 D128 기각, 2026-09-05 사용자 판단).
  · 실측(646 graph): dyn 노드 3,233개 중 **211개(6.5%) / 145편** 강등, **dyn 전멸 영상 0편**.
  임계 감도 — 0.02/0.06 에서 42개(1.3%), 0.08/0.24 에서 411개(12.7%, 전멸 3편).
  · `dynamic_mask/from_seg.json` 에 `demote_*` 인자 · `demoted_track_ids` · `demoted_nodes` ·
  `mean_before_demote` · `flipped_pixel_frac` 을 남긴다. **노드 수가 아니라 뒤집힌 픽셀 비율**을
  재는 이유는, 강등한 트랙이 살아남은 트랙 마스크 안에 들어 있으면(앉은 사람의 셔츠) 합집합이
  그대로라 노드 수만 세면 그걸 못 보기 때문이다.
- **`configs/bank/d177_dynpose.json`** — dynpose 394편을 새 `dynamic_mask` 위에서 d171c 방식으로
  다시 굽는 세대 config (`cloud→route→tau→fit→emit`, graph 는 `.graph_d157` 재사용).
  fit 인자는 `d171c_dynpose_kftrans.json` 글자 그대로 = 사다리 1단 `--hole_ladder 0.20` +
  `--fallback_ladder --fallback_target 5` + `--min_subject_visible 0.6` +
  `--follow_keyframes 6 --follow_kf_interp cubic`.
  스코프 주의 — `d172_dynpose100k_graph.json` 이 `out_dynpose` 와 마커 `.graph_d157` 을 공유해서
  `out_dynpose` 의 마커 606개 중 212개는 **DynPose-100K** 편이다. 스코프 키는 마커가 아니라
  "DynPose-LBM 에 `seg_instances/<v>/masks.npz` 가 있는가" 다.
- **D178-b — 릴·검증 스크립트 9종에 `--cloud_source` 배선 (2026-09-11).** tau/fit 만 memory 를
  받으면 `cloud.npz` 를 지울 수 없다 — 릴·검증 쪽이 여전히 npz 를 열기 때문이다. 그래서
  `verify.py` · `scripts/{render_bank_videos,render_pred_depth_warp,render_director_depth,
  render_preset_grid_warp,render_target_cams_warp,render_target_poses_depth,render_target_swap_warp,
  build_candidate_board,audit_bank_geometry,eval_subject_in_frame}.py` 를 전부
  `add_cloud_source_args()` + `open_renderer()` 공통 진입점으로 바꿨다. **기본값은 `npz` 라 기존
  동작 그대로**다.
  · `lbm/render.py:open_renderer` 에 `video: str = None` 파라미터 추가 — `eval_subject_in_frame.py`
  / `render_target_{cams_warp,poses_depth}.py` 는 **한 프로세스가 여러 씬을 루프로 돈다**.
  그런 호출부는 `args.video` 자체가 없어서 기존 시그니처로는 못 부른다.
  · 하드코딩돼 있던 `if not path.isfile(cloud_path): skip` 을 `args.cloud_source == "npz"` 로 감쌌다.
  memory 모드는 recon 만 있으면 되는데 그 skip 이 먼저 걸려 전부 건너뛰었다.
  · 배선 뒤 `out_dynpose` 의 `cloud.npz` **447편 606.6 GiB 삭제** (사용자 승인 2026-09-11).
- **`configs/bank/d179_dynpose100k_bank.json` + `tmp/d179/chain_bank_10k.py`** — DynPose-100K
  10,346편 전량 뱅크 (사용자 지시 2026-09-11 "이어서 10k scene 뽑아놓은것도 다 돌려놔줘").
  graph 는 **이 config 에서 안 돈다** — `d172_dynpose100k_graph.json` 12샤드가 같은 `out_dynpose`
  에 같은 마커(`.graph_d157`)로 굽고 있어서 켜면 같은 `scene_graph.json` 에 두 프로세스가 동시에
  쓴다. 대신 `require_markers: [".graph_d157"]` 로 **graph 가 끝난 편만** 집어간다.
  드라이버는 라운드 루프다 — graph 가 8일치 남아 있어서(실측 47편/h) 다 끝나기를 기다리면 그만큼
  논다. 라운드마다 ① ready 스캔 → ② 정지 소품 강등(8 CPU 샤드, `from_seg.json` 의
  `demote_static_objects` 로 중복 방지) → ③ route/tau/fit/emit 4 GPU 샤드 → ④ 재스캔.
  `--cloud_source memory` 라 디스크 증가 0 (npz 였으면 1.44 GiB × 10,346 = 14.5 TiB, /data1 여유 2.9 TB).
- **`configs/bank/d178_dynpose_rest.json`** — 483편용. **기동 후 중단했다** (2026-09-11 09:57).
  483편이 `tmp/d172/videos_10346.txt` 의 **부분집합**(483/483)이라 graph 가 d172 와 완전 중복이었고,
  마커가 `.graph_d178` vs `.graph_d157` 로 달라 서로 skip 도 안 됐다. 실측: d178 이 끝낸 11편은
  **전부** 이미 `.graph_d157` 을 갖고 있었다 — 두 목록이 같은 UUID 정렬이라 d172 가 지나간 앞쪽을
  다시 갈고 있었다. 스코프는 d179 가 흡수한다. config 는 근거와 함께 남긴다.

### Fixed
- **`--fallback_target 1` 이 "씬당 카메라 1대"가 아니었던 것 (2026-09-13, FIX-D188-b).**
  그 인자는 `variant_usable()` 이 참인 변이를 세는데, 그 함수(`fit_hole_ladder.py:921-926`)는
  `--retry_status` 뿐 아니라 **`--retry_suspect` 도** 본다. d179 부모 config 가 물려준
  `--retry_suspect` 에 `hole_over_budget`(= `hole_fraction > --suspect_hole` 0.35) 이 들어 있고,
  `--hole_mode excess` 에서는 목표가 `hole_static + Δ0.20` 이라 `hole_static` 이 큰 씬은
  **정상적으로 0.35 위에서 풀린다**. 그 행들이 안 세어지니 예산이 영원히 안 차고 사다리가
  3층을 다 내려갔다.
  · 실측(사다리가 켜진 파일럿 58편): solved 92행 중 **57행이 `hole_over_budget`** 태그,
    `usable` 0 으로 끝난 씬 **23/58**, 뱅크에 solved 가 씬당 최대 6행.
    씬당 solved [(0,9),(1,29),(2,9),(3,3),(4,6),(6,2)] / 씬당 usable [(0,23),(1,35)].
  · **태그를 무시하게 고치지 않았다** — 그러면 "hole 0.6 짜리 tier0" 가 "hole 0.2 짜리 tier1" 을
    이겨서 어휘는 맞고 품질이 무너진다. 대신 탐색(사다리)과 선택(`--pick_budget`)을 분리했다
    (위 Added 항목).
- **`scripts/rebake_scenes.py` 가 config 의 `extends` 를 안 풀어 뱅크 단계에서 즉사하던 것
  (2026-09-13, FIX-D186-b).** `quarantine()` / `stage_bank()` / `main()` 이 config 를 `json_load`
  로 그대로 읽었는데, 뱅크 config 는 대부분 `"extends": "d179_dynpose100k_bank.json"` 이라
  `output_root` 가 부모에만 있다. graph config 는 `extends` 가 없어 ①단계는 통과했고 ③-a 의
  `d184_*` 에서 `KeyError: 'output_root'` — `launch.sh` 가 `set -e` 라 ③-b(d185 뱅크 313편)까지
  같이 멈춰 13:53~14:31 유휴였다. `load_config()`(= `run_bank.load_config` 과 같은 얕은 병합)를
  추가하고 세 호출부를 전부 바꿨다. ③-a(d184 386편)는 **건너뛴다** — d184 는 폐기 세대라 쓰는
  데가 없고, 남은 역할이 도는 d185 드라이버의 `--require_bank_dir hole_bank_d184` 게이트 하나인데
  ③-a 의 격리가 바로 그 디렉토리를 옮겨 게이트를 깨뜨린다.
- **인스턴스 병합이 `frames` 만 합집합으로 만들어 bbox/score 가 밀리던 것
  (`scene_graph/instances.py`, `scripts/build_scene_graph.py`) (2026-09-13).**
  `merge_duplicates` 가 `points_by_frame` 과 `frames` 는 합집합으로 만들면서 `boxes_xyxy` /
  `scores` 는 root 인스턴스 것을 그대로 뒀다. `build_node` 의
  `zip(inst["frames"], inst["boxes_xyxy"])` 가 짧은 쪽에서 **조용히 잘리고**, 합집합이 root 보다
  앞선 프레임을 포함하면 프레임↔박스가 **한 칸씩 밀려 짝지어진다**.
  · 크래시 — D182 10,346편 중 **73편(0.71%)** 이 `KeyError` 로 사망. 예외 키가 48×40 / 47×9 로
    꼬리에 몰렸다 (`best_frame` 이 잘린 구간에 떨어진 경우).
  · **조용한 오염** — 안 죽으면 `bbox_xyxy_best` / `track.conf` 가 밀린 채 framing 게이트로 간다.
    scene_graph 800편 실측 `len(track.conf) != num_visible_frames`: merged 838개 중 **34개(4.1%)**,
    비-merged 7663개 중 0개. 병합은 씬의 54%(322/600)에서 일어난다.
  · 이제 `scores`/`boxes_xyxy` 를 합집합 `frames` 에 맞춰 재구성한다(root 우선, 빈 프레임만 채움 =
    겹칠 때 기존 동작 유지). `build_node` 에는 길이 assert 를 세워 **밀린 bbox 를 내보내느니
    노드 id/label/merged_from 과 함께 죽게** 했다. 근거는 `FIX.log` 2026-09-13.
- **`--max_rounds` 를 기다림이 갉아먹던 것 + `--min_ready`
  (`scripts/chain_bank_rounds.py`) (2026-09-13).** D184 가 8093/10346 에서 `--max_rounds 60
  소진` 으로 조용히 멈췄다. 죽은 게 아니라 캡을 다 쓴 것 — 선행 graph 를 따라잡은 뒤로
  `ready 6~8` 짜리 **1분 라운드**가 줄줄이 돌았다 (마지막 30 라운드 중 21개가 ready ≤10).
  `ready 0` 로 쉬는 것까지 `for rnd in range(max_rounds)` 가 한 라운드로 셌다.
  · 이제 `--max_rounds` 는 **구운 라운드**만 센다. 캡은 폭주 방지용이지 대기 예산이 아니다.
  · `--min_ready N` (기본 0 = 기존 동작): 선행이 살아 있는데 ready 가 N 미만이면 굽지 않고
    `--poll` 만큼 쉰다. 라운드 고정비(강등 8샤드 + 뱅크 8샤드 기동)가 편당 비용을 압도하는
    구간을 피한다. D184 는 `--min_ready 40` 으로 재기동했다.
- **2+2 preset depth-warp 릴이 전편 실패해도 `ALL DONE` + exit 0 이던 것
  (`scripts/run_preset_warp_{sample,max_shard}.sh`) (2026-09-12).** d185 뱅크 릴 4편이 전부
  죽었는데 래퍼는 성공처럼 끝났다. 세 가지를 고쳤다 — 증상·근거는 `FIX.log` 2026-09-12 두 항목.
  · **`CLOUD` env (기본 `auto`)**: `<ROOT>/<video>/cloud.npz` 가 있으면 예전처럼 `npz`, 없으면
    `--cloud_source memory`. D178 이후 굽기가 cloud.npz 를 안 남겨서 릴이 `AssertionError:
    cloud.npz 가 없다` 로 죽었다 (백로그 D178-b 가 미배선인 채 처음 물린 것). `npz`/`memory`
    명시 시 강제. cloud.npz 가 남아 있는 옛 세대는 **문자 그대로 기존 커맨드**다.
  · **`eval_data` 를 ROOT 가 아니라 씬 존재로 고른다**: 출력 루트 `out_dynpose` 밑에 코퍼스가
    둘이다(DynPose-LBM 880편 / DynPose-100K 10,346편). `case "$ROOT"` 로 LBM 에 못 박혀
    `ValueError: Could not open video file` 이 났다. 이제 표본 첫 편이 실제로 있는 루트를
    후보 목록에서 고르고, 못 찾으면 exit 1 로 세운다. `out` 은 예전대로 렌더러 기본값.
  · **실패를 프로세스 rc 로 올린다**: 러너가 편별 rc 를 세어 `실패 N` 을 찍고 exit 1,
    실패 편마다 로그 마지막 줄(예외 메시지)을 같이 출력한다. 래퍼도 그 rc 를 그대로 넘긴다.
- **route 의 "anchor 0" 을 크래시에서 기록되는 skip 으로
  (`scripts/route_presets.py --skip_if_empty`, `scripts/run_bank.py`) (2026-09-12).**
  `route_presets.main` 은 anchor 후보가 0개면 `assert` 로 죽는다(rc=1). D181 파일럿은 씬 목록을
  같은 문턱으로 미리 걸러서 문제가 안 됐지만, D184 는 전량이라 **절반 이상**이 여기서 걸린다.
  크래시로 두면 `skipped.json` 이 안 남아 `chain_bank_rounds` 가 라운드마다 같은 씬을 다시
  집고, 로그에는 `FAIL(route rc=1)` 이 쌓인다 (스모크 4편 중 3편이 그랬다). 새 플래그를 주면
  rc=3 으로 정상 종료하고 `run_bank` 가 `hole_bank_*/skipped.json(no_surviving_anchors)` 를
  남긴다. 화이트리스트 뒤 슬롯이 0개인 경우도 같은 경로다. **기본값 off = 옛 동작(assert).**
- **`chain_bank_rounds` 의 라운드 요약이 편별 실패를 안 셌다 (2026-09-12).** `fan_out` 이 세는
  건 샤드 **프로세스의 rc** 라, 샤드가 정상 종료하면 그 안에서 몇 편이 FAIL 했는지 안 보인다
  — 스모크에서 3/4 편이 route 로 죽었는데 "실패 0/4" 로 찍혔다. 이제 샤드 로그의 `=== 요약 ===`
  표를 파싱해 `SKIP(앵커 0) 10  OK 6` 처럼 편별 상태 분포를 같이 낸다.
- **D182 — graph 게이트 분모를 "도달 가능한 편"으로 (`tmp/d182/chain_static_graph.py`,
  `--gate_basis`) (2026-09-12).** sam3 가 100% 성공했는데도 체인이 03:20:53 에 스스로 멈췄다.
  `sam3_done()` 이 `seg_instances_static` 개수를 **목록 10,346편으로 나눠** 96.7% 를 내고
  `--min_ratio 0.97` 게이트에 걸렸다. 그러나 `sam3_static_instances.py` 는 정적 명사가 0개인
  편에서는 프롬프트가 없어 **아무것도 만들지 않는다** — 도달 가능 최댓값은 10,346 이 아니라
  정적 명사 ≥1 인 **9,989편**이고, 실측은 9,989/9,989 (누락 0) + LBM 심볼릭 14편 = 10,003 이다.
  게이트 기준을 `--gate_basis achievable`(기본, 명사 ≥1 인 편이 분모) / `all`(옛 동작, 목록
  전체)로 분기하고, 구조적으로 정적 seg 가 불가능한 357편을 `tmp/d182/static_unavailable.txt`
  로 남겨 dyn-only 잔여가 **조용히** 흘러가지 않게 했다. 재기동 후 게이트 9,989/9,989 = 100%
  통과, graph 24샤드가 03:26:22 에 기동했다 (첫 4편 rc=0 153.3~164.2s).
- **D176-c — `cloud.npz` 를 빈 `dynamic_mask` 위에서 굽는 사고를 막는 가드 (2026-09-10, 사용자
  질문 "cloud 가 static 으로 잘못 나온 이유가 뭐야?").** 원인은 **단계 순서 경합**이다:
  `scripts/dynpose_ingest.py:177` 의 recon 단계가 `seg_keywords=[]` 라 **all-zero placeholder**
  `dynamic_mask/*.png` 를 써 두고, 나중에 `dynmask` 단계가 SAM3 합집합으로 덮는다. 그 사이에
  cloud 를 구우면 `load_recon_and_seg` 가 `static_mask = ~dynamic_mask` 로 **전 픽셀을 정적**으로
  만들고, 정적 점은 `visible` 이 전 프레임 True 라 동적 물체가 49개 사본으로 잔류한다 — 에러
  없이, `visible.sum(1)==1` 이 정확히 0 으로. `utils/media.py:148 load_masks` 는 png 가
  **존재하는지**만 보므로 새까만 png 가 그대로 통과한다.
  · 규모: dynpose 표본 120편 중 **47편(39%)**. 편향은 `hole_fraction` med **0.199 vs 0.279**,
  `subject_area_med` **0.0895 vs 0.0409**, `subject_visible_frac<0.6` **36.8% vs 25.5%** —
  고스트가 프레임을 메우고 자기 자신을 가린다. bake 시각으로는 못 가른다 (정상 씬의 cloud 가
  오염 씬보다 오래된 경우가 있다: 1216d742 1788788057 정상 / 13d42c1a 1788805177 오염).
  · `lbm/cloud.py` — 굽기 전 `dynamic_mask.mean()==0` 이면 **assert 로 멈춘다** (§empty_dynmask).
  메시지가 `seg_instances/<video>/masks.npz` 유무를 같이 찍어서 "dynmask 를 먼저 돌려라" 인지
  "정말 정적인 씬"인지 바로 갈린다. 후자는 `--allow_empty_dynamic_mask` 로 통과.
  cloud meta 에 **`dynamic_mask_frac` / `num_dynamic`** 을 실어 npz 만 보고도 판별되게 했다.
  · `scripts/dynpose_dynamic_mask_from_seg.py` — 완료 마커 `dynamic_mask/from_seg.json`
  (`mean`, `num_frames`, `seg_mtime`) 을 쓴다. png 만으로는 "단계가 안 돎" 과 "돌았는데 SAM3 가
  아무것도 못 찾음" 이 구분되지 않았다. `*.png` 만 읽는 `load_masks` 에는 영향 없다.

### Changed
- **`build_scene_graph.py` 의 "살아남은 노드가 없다" assert 가 후보 수를 같이 찍는다
  (2026-09-13).** 기존 메시지는 `dropped=0` 만 줘서 "아무것도 안 떨어뜨렸는데 아무것도 안
  남았다" 로 읽혔다 — 실제 원인은 **입력 instance 가 0개**(SAM3 가 준 명사로 아무것도 못 찾음)
  인데 코드 버그처럼 보인다. 이제 `후보 N개 / 탈락 M개` 를 찍고, 후보가 0이면
  "seg_instances 가 비었다 (SAM3 가 0개를 냈다). 이 씬은 subject 가 없다." 를 덧붙인다.
  · 실측: d182 graph 실패 3편(`217fbf3b…` / `21bfdd94…` / `50beb1f0…`)이 **전부** 이 경우다.
    `seg_instances/<v>/meta.json` 의 `instances` 가 빈 리스트 — recon/seg 디렉토리는 있다.
    10,346 중 3편이라 수율 영향은 없지만, subject 가 없으니 카메라도 못 만든다.
- **D179 — `run_bank.py --threads` (기본 8): BLAS/OpenMP 스레드 상한. graph 단계가 CPU 를 48배
  태우고 있었다 (2026-09-11, 사용자 질문 "더 빠르게 최적화 못함?").** 같은 씬
  (`bb3bd56c-6993-40ef-8164-ba6635694d4d`) · 같은 인자 · `/usr/bin/time -v` A/B:

  | | user CPU | system | wall | %CPU | 비자발 문맥전환 |
  |---|---|---|---|---|---|
  | 캡 없음 | **12,326.73 s** | 11.16 s | 393 s | **3136%** | 4,435,047 |
  | `*_NUM_THREADS=8` | **254.02 s** | 3.44 s | 107 s | 241% | — |

  **산출물 `scene_graph.json` 은 md5 동일** (`bd96def456a697ecbe749c56244e45e7`, 60,447 bytes
  양쪽). 즉 사라진 12,072 core-s 는 연산이 아니라 **OpenBLAS/OpenMP 의 spin-wait** 다 —
  프로세스당 **391 threads** 가 124 코어 위에서 서로를 기다리며 코어를 태운다. 혼자 돌 때는
  기계 전체를 먹어 wall 이 짧게 나오지만(d172 config `_runtime` 주석의 "85.5 s/편"이 이 수치다),
  샤드를 12개 띄우면 서로 spin 으로 물려 **47편/h** 까지 떨어진다. 실측 load average 는 12샤드에서
  **593**, 캡 후 프로세스당 threads 는 **39**.
  · 배선 위치는 `scripts/run_bank.py` 의 서브프로세스 env 다 (`THREAD_ENV_KEYS` 6종:
  `OMP`/`OPENBLAS`/`MKL`/`NUMEXPR`/`VECLIB_MAXIMUM`/`OPENCV_FOR` `_NUM_THREADS`). 여기 한 곳이
  graph/cloud/route/tau/fit/emit 전 단계의 공통 진입점이라, 드라이버
  (`tmp/d172/chain_geocalib_graph.py`, `tmp/d179/chain_bank_10k.py`)는 손댈 필요가 없다.
  · **`--threads 0` 이 예전 동작**(캡 없음)이다.
  · **뱅크(tau/fit)는 캡의 영향이 거의 없다** — 같은 씬 TAU 125.5 s(캡8) vs 109.4 s(무캡).
  GPU 바운드라 그렇다. 그래서 진행 중이던 d179 4샤드는 재기동하지 않고 다음 라운드부터
  새 기본값이 붙게 뒀다.
  · 집행: d172 graph 12샤드를 SIGINT 로 내리고 **24샤드**로 재기동
  (`.graph_d157` 1,159/10,346 시점).
- **D176-b — dolly 계열 캡션에서 부정형 재조준 절을 뺐다 (2026-09-10, 사용자 지시 "그냥 dolly in
  look at 일 경우만 towards woman 이렇게 하면 되잖아").** `configs/caption_presets.json` 의 세
  문구에서 `along its own axis without re-aiming` / `along its own axis` 를 지우고, 재조준
  여부를 **대상을 방향으로 부르는지**로만 가른다:
  `dolly_in` `"dollies straight forward"` / `dolly_in_look_at` `"... toward {target}"`,
  `dolly_out` `"dollies straight back"` / `dolly_out_look_at` `"... away from {target}"`,
  `track_dolly_in` `"tracks {target} while pushing in"` / `..._look_at` `"... toward {it}"`.
  부정("~하지 않는다")을 조건으로 주는 것은 조건부 생성에서 약한 신호다.
  · 옛 문구는 `phrase_legacy_reaim` 키로 config 에 남아 있고 `--legacy_dolly_phrase` 로 되살린다
  (`promote_targetless` 와 같은 이유 — 학습에 들어간 캡션이 조용히 바뀌면 안 된다).
  캡션 JSON 헤더에 `legacy_reaim_phrase` 키, 요약표에 `재조준 부정문구 켬/끔` 추가.
  · **실제로 문장이 바뀐 preset 은 `dolly_out` 하나**다. d157 384편 55,462행에서
  `dolly_out` 5,572행(10.05%) / `dolly_in_look_at` 5,544행(10.00%) / `track_dolly_in_look_at`
  1,936행(3.49%) / `track_dolly_out` 1,844행(3.32%) 인데 **`dolly_in` 과 `track_dolly_in` 은
  0행**이다 (라우팅이 aim=free push-in 을 한 번도 안 뽑았다).
  · 예: `The camera dollies back along its own axis without re-aiming, keeping dog in view.`
  → `The camera dollies straight back, keeping dog in view.`

- **D176 — nl 캡션 기본 문장을 `target` + `camera` 두 축만으로 좁혔다 (2026-09-10, 사용자 지시
  "정도 부사를 빼주고 framing 도 ... 빼서 target, camera 관련된 내용만 들어가도록").**
  `scripts/build_bank_captions.py` 의 **기본값 두 개**를 뒤집었다 — `--magnitude` 3-state 의
  형식별 기본(`nl` 켬)이 **형식과 무관하게 끔**이 되고, `--nl_framing` 기본이 `True → False`.
  예전 문장은 `--magnitude --nl_framing` 으로 글자 그대로 돌아온다.
  · **부사를 뺀 이유**: `adverb_of`(`:168`)가 요청이 아니라 **실현치**(`tau_max`/`pan_deg`)를
  버킷에 넣는다. d157 55,462행 중 54%가 기하 제약(`shape/elev/obb/approach/collision/
  clamped_low`)으로 깎였는데 부사도 같이 깎여 캡션↔pose 가 일관된다 — 요청↔실현 대비가 남는
  축은 orbit sweep 하나뿐이고(캡션이 `sweep_deg` 를 안 읽는다), 거기서도 `--min_sweep_deg 20`
  바닥 556행(orbit 의 17.2%) 중 465행에 significantly/dramatically 가 붙는다.
  **최종 목표는 요청 강도를 싣는 것**(사용자 확정)이고 그때까지는 뺀다.
  · **framing 을 뺀 이유**: shot size 어휘가 실측 면적비와 구간이 겹친다 — 불일치 vista d121
  35.9%(9,052/25,222) / dynpose d157 **49.6%**(10,300/20,785). `plain` 절은 0/10,872 로 완벽하고
  범인은 timeline head 어휘다(tighten 61.7% / widen 57.1% / exit 82.8%). 게다가
  `subject_visible_frac`(가림)을 캡션 코드가 **한 번도 안 읽는다** — `<0.5` 가 vista 9.5% /
  dynpose 9.7%. `caption["framing"]` 구조체는 그대로 채워지므로(D166 층 구분) 지표로는 남는다.
  · 캡션 JSON 헤더에 **`nl_framing` 키를 추가**했다. `framing_on_free`/`framing_min_in_frame`/
  `framing_exit_min_in_frame` 은 *자격 판정*이고 `nl_framing` 은 *문장에 실었는가*라 층이 다른데,
  헤더에 없어서 두 캡션 파일을 문자열로 역추적해야 했다. 요약표에도 `framing 문장 켬/끔` 추가.
  · 예 (`160a57c9 dyn_0__track_truck_left__hole0.1`):
  `The camera steadily tracks alongside woman while sliding to the left, keeping her in a
  close-up shot that widens to a medium close-up shot.`
  → `The camera tracks alongside woman while sliding to the left.`

### Added
- **D176-c — `configs/bank/d176_dynpose_cloudfix.json` (D174 릴 3편 cloud 재굽기).**
  `stages: ["cloud", "fit", "emit"]`, `tau_bank_dir: "bank_d157"` 재사용, `bank_dir:
  "hole_bank_d176"`. **fit/emit args 는 `d157_dynpose.json` 글자 그대로** — 릴의 질문이 "같은
  조건에서 cloud 만 고치면 warp 이 달라지는가"라 설정 축을 하나도 안 건드린다. tau 를 재사용하는
  근거는 τ 가 cloud 를 안 보기 때문이다 (분모 `S` 는 raw depth, 기준점은 시작 pose).

- **D175 — `configs/bank/d175_vista.json` (vista 52편에 비-routing 델타 이식) (2026-09-10,
  사용자 지시 "vista 에도 똑같이 적용해서 돌려봐줘").** d151 대비 **fit 인자만** 바뀐다:
  `--min_subject_visible 0 → 0.6` (D171-b), `--follow_keyframes 6 --follow_kf_interp cubic`
  (D171-c), `--hole_ladder`/`--hole_mode` **명시 핀**(값은 d151 이 실제로 쓴 것과 동일 —
  `out/*/hole_bank_k6_d151/bank.json` 의 `ladder_base.target_rule = "hole_static + delta"`).
  · **tau 는 안 돈다** — `bank_d151` 재사용. d170 계열이 tau 에 더한 `--device cuda` 는 argparse
  기본값과 같고(`sample_camera_bank.py:1385`), `--skip_on_empty` 는 0행일 때만 갈리는데 d151 은
  52편 전부 행이 나왔다.
  · **`--fallback_ladder` 계열은 일부러 뺐다.** `fit_hole_ladder.py:797` 이 `fallback_on =
  fallback_ladder and any(tier_of.values())` 인데 tier 는 τ 가 `--variant_pool full` 로 깐 예비
  변이에만 붙는다 — `bank_d151` 은 budget 풀이라 전 tier 0 → 켜도 no-op 이다. 게다가
  `--fallback_target 5` 는 "쓸 만한 변이 5개 모이면 중단"이라 켜지면 편당 343행을 5행으로 자른다.
  · **hole 사다리는 4단 유지** (dynpose 계열의 1단 `0.20` 을 안 따라간다). 단수 축소는 품질 수정이
  아니라 코퍼스 크기 결정이고 편수가 384 vs 52 로 7배 다르다 — vista 에서 같은 짓을 하면
  17,836행이 ~4.5k 로 줄고 `min_subject_visible 0.6` 이 또 깎는다.
- **D175-a — dynpose `cloud.npz` 의 ~40%가 동적 점 0개 (2026-09-10, 사용자 지적 "릴 첫 영상에서
  dynamic object 로 point cloud 가 되어야하는데 왜 static 으로 되어있음?").** `lbm/cloud.py` 의
  `unproject` 는 `static_mask = ~dynamic_mask` 로 정적 점에 시간 지속성을 주는데, 굽는 시점에
  `dynamic_mask` 가 전부 0이면 **모든 픽셀이 정적**이 되어 동적 물체가 49개 사본으로 전 프레임
  남는다 (`visible.sum(1)==1` 이 정확히 0, 에러 없음 — D172 에 기록된 함정과 같은 계열).
  · 표본 120편(`tmp/d175/scan_cloud_dyn.py`, seed 0/1) 중 **47편(39%)이 동적 점 0개**이고 그 씬들의
  현재 `dynamic_mask` 는 전부 비어 있지 않다(8~63%). bake 시각으로는 안 갈린다 — zero 구간
  09-08 00:11~16:26 과 ok 구간 09-04 04:59~09-08 16:05 가 겹친다(d157 굽기와 D169 ingest 가 병주).
  · 편향 방향 (d157 뱅크, 표본 80편 · 5,110행 vs 5,247행): `hole_fraction` med **0.199 vs 0.279**,
  `subject_area_med` **0.0895 vs 0.0409**, `subject_visible_frac < 0.6` **36.8% vs 25.5%**.
  고스트가 프레임을 메우고 자기 자신을 가린다 — 사다리가 푸는 손잡이가 이 열들이라 궤적 크기까지
  틀어진다. `scene_graph.json` 은 seg_instances(SAM3) 기반이라 `dyn_*` 노드는 멀쩡하다.
  · vista `out/` 은 무사 (12/12 정상, dyn 4.19~58.56%).
  · D174-b 릴 3편 중 `160a57c9`(0%)와 대조군 `13d42c1a`(0%)가 여기 걸린다 — **각변위 수치는
  scene_graph track 에서 나오므로 유효하지만, 렌더 인상은 2/3 이 오염됐다.**
- **D174 — `--orbit_min_span` 게이트를 d157 코퍼스로 실측 판정 (2026-09-10, 사용자 질문
  "orbit_min_span이 뭔데" 에 답하기 위해).** 이 게이트는 `route_presets.py:291` 에서
  `obs_az_span_deg < ORBIT_MIN_SPAN_DEG(120)` 인 노드의 orbit 슬롯을 통째로 없앤다(`--orbit_fallback
  drop`). 근거는 "소스가 좁게만 본 물체를 선회하면 미관측 면으로 넘어가 warp 이 구멍난다" 였는데
  **실측이 한 번도 없었다.** d157 뱅크는 게이트 도입 커밋(`99adfbc`) **이전** 세대라 span 과 무관하게
  orbit 을 구웠다 — 384편 / Δ0.2 단 **orbit 807행**이 그대로 자연 실험이다 (`tmp/d174/span_vs_hole.py`).
  span 버킷별 중앙값:

  | span | n | hole | hole_max | hmax/h | tau | sweep | vis | subj_area | knob=cap% | shape_limited% |
  |---|---|---|---|---|---|---|---|---|---|---|
  | <20 | 103 | 0.287 | 0.492 | 1.62 | 1.140 | 20.0 | 0.768 | 0.077 | 53.4 | 49.5 |
  | 20–40 | 107 | 0.298 | 0.476 | 1.59 | 0.875 | 22.5 | 0.814 | 0.073 | 37.4 | 36.4 |
  | 40–60 | 51 | 0.298 | 0.464 | 1.59 | 0.814 | 36.8 | 0.730 | 0.054 | 13.7 | 5.9 |
  | 60–90 | 111 | 0.283 | 0.460 | 1.54 | 0.871 | 45.0 | 0.769 | 0.103 | 7.2 | 6.3 |
  | 90–120 | 77 | 0.289 | 0.481 | 1.54 | 0.901 | 45.0 | 0.751 | 0.086 | 5.2 | 5.2 |
  | 120–180 | 140 | 0.253 | 0.436 | 1.69 | 0.911 | 45.0 | 0.777 | 0.080 | 4.3 | 3.6 |
  | ≥180 | 218 | 0.290 | 0.488 | 1.67 | 0.917 | 45.0 | 0.733 | 0.072 | 7.8 | 6.0 |

  · **`obs_az_span` 은 hole 을 예측하지 않는다.** `hole`·`hole_max`·`hmax/h`(끝프레임 쏠림)·
  `subject_visible_frac`·`subject_area_med` 가 span 1.5° ~ 315° 에서 전부 평평하다. span<20 은 오히려
  tau 가 최대(1.140)고 vis 도 상위다. 게이트가 세워진 전제("좁은 span → 미관측 면 → 구멍")가 807행에서
  **관측되지 않는다.** 물리적으로도 그렇다 — 구멍은 물체 주위 방위각이 아니라 **근접 가림물과의 시차**에서
  나오고, 그건 사다리가 이미 직접 재서 손잡이로 통제한다.
  · span 이 실제로 바꾸는 것은 **sweep clamp 하나**다: `span_frac = max(orbit_span_frac, min_sweep_deg/span)`
  이라 span<56 이면 sweep 이 nominal 45°에 못 미치고, span<20 이면 바닥 20°에 붙는다 → `knob=cap` 53.4%,
  `shape_limited` 49.5%. 즉 좁은 span 의 orbit 은 "반경 큰 20° 얕은 호"가 된다.
  · span 1.5° mirror 가 realized sweep 20° 인데 캡션은 "the camera dramatically orbits to the left
  around mirror" — 강도 부사는 tau/path 에서 나오고 sweep 은 안 본다. **정정 (2026-09-10 사용자):
  캡션이 realized 를 안 읽는 것은 버그가 아니라 설계**다 — 요청한 변위가 기하학적으로 깎였을 때
  모델이 geometry 를 읽어 그 축소를 맞히게 하려는 것. 정도가 과한지는 D175-b 에서 잰다.
  · 릴 4편 `tmp/d174/reel/d174_span{1.5,1.8,3.5,308}_*.mp4` (span 극단 3 + 대조군 1, `--fixed_focal`).
  · **추천: `--orbit_min_span 0`** (= d170 ablation 설정 유지). 120 은 노드의 73%(moving 73.1%)를
  증거 없이 버려 backfill 에 떠넘기고, 앞서 산수로 제안했던 20 도 이 표에서는 막을 대상이 없다.
  · **채택 (2026-09-10 사용자 지시 "orbit_min_span 은 별로인 것 같아 빼줘"): 기본값을 `0` 으로.**
  `route_presets.py` 의 함수 기본값과 CLI 기본값 **둘 다** `ORBIT_MIN_SPAN_DEG`(120.0) → `0.0`.
  인자와 상수는 남긴다 — `--orbit_min_span 120` 이 d166~d171 재현이고 스윕 손잡이이기도 하다.
  `reasons.orbit_min_span_deg` 는 계속 찍히므로 옛 JSON 과 대조할 때 문턱이 바뀐 건지 span 이
  바뀐 건지 가릴 수 있다.
- **D174-b — `track_*` 게이트 감사: 배선은 정상, 문턱의 단위가 틀렸다 (2026-09-10, 사용자 지적
  "global translation 이 일정 이상 안움직이면 track preset 안돌게끔 배선되어있는거 아니야?
  현재 그런것들이 보이는데").** 게이트는 두 군데에 있고 둘 다 집행된다 — `route_presets.py:249`
  (`center_drift_u > track_min_drift_u` 여야 track 슬롯을 만든다) + `sample_camera_bank.py:956`
  (아니면 변이 폐기), CLI 기본 0.05 u. `tmp/d174/track_drift_audit.py` 로 세 세대를 감사한 결과
  **위반 0건**: d157 track 행 13,557 / anchor 701 중 `drift ≤ 0.05` 가 0개, d170·d171c 도 0.
  · 문제는 `center_drift_u` 가 scene scale `S` 로만 정규화된 양이라 **카메라와의 거리를 안 본다**는
  것이다. 화면상 운동량은 각변위 `center_drift_u / d_ref` 에 비례한다. d157 track anchor 701개
  분포 — p5 0.0774 / p10 0.1041 / p25 0.1764 / p50 0.3528 / p75 0.8185 / p90 1.8195,
  `≤0.10` 9.4% / **`≤0.15` 21.0%** / `≤0.20` 31.7%.
  · 실측 대조: `1216d742` dyn_2 box `drift 0.0549 / d_ref 1.2493 → 0.0439` 와 `160a57c9` dyn_1
  suitcase `0.0608 / 1.4450 → 0.0421` 가 순변위 게이트를 통과하는데, 같은 영상 `1216d742` dyn_0
  hand 는 0.2740 이고 대조군 `13d42c1a` dyn_0 man 은 `0.2748 / 0.2823 → 0.9734` 로 23배다.
  캡션은 구별을 못 한다 — box 0.044 에 "the camera **dramatically tracks** box while craning
  upward", suitcase 0.042 는 `track_{crane_up,dolly_in_look_at,pull_out_arc_right,truck_left}`
  4종 전부가 붙었다. `lbm-preset-names-dont-match-motion` 과 같은 계열이다.
  · 릴 3편 `tmp/d174/reel/d174_ang{0.0421,0.0439,0.9734}_*.mp4` — preset 을 `track_truck_left`
  하나로 고정하고 각변위만 바꾼 대조 (`--fixed_focal`, `tmp/d174/render_track_drift.py`).
  · 각변위 자체는 이미 계산돼 있다 — `rank_subject_motion.py:88` 의 `ang_drift = center_drift_u
  / d_ref` 인데 **진단 열일 뿐 게이트가 아니다**. 문턱 값과 게이트 승격은 사용자 판정 대기.
- **D172 — `configs/bank/d172_dynpose100k_graph.json` (dynpose-100k 10,346편 graph 전용 세대)
  + `tmp/d172/chain_geocalib_graph.py` (2026-09-10, 사용자 지시 "cloud까지만 돌리도록해줘" 를
  실측 후 graph 까지로 좁힌 것 — 같은 날 사용자 결정).** 지시를 글자대로 이행하면 /data1 이
  찬다: `cloud.npz` 는 씬당 **1.44 GB**(p50 1475 MB / p95 1527 MB / max 1535 MB, n=447 실측)라
  10,346편이면 **15.2 TB** 인데 여유는 **2.9 TB** 다 — 약 1,970편에서 포화해 /data1 을 쓰는
  모든 사용자를 막는다. `scene_graph.json` 은 씬당 144 KB 라 10k 여도 1.5 GB 로 안전하다.
  그래서 이 세대는 `stages: ["graph"]` 뿐이고, cloud 는 D171(hole 예산·occlusion 게이트)이
  확정된 뒤 배치 단위로 cloud→tau→fit→emit 를 돌고 그 배치의 cloud 를 지우는 방식으로 간다.
  · `cloud.npz` 내부 실측(43.5M 점): `points_world` f32 522 MB + `indices` i32 522 MB +
  `visible_packed` u8 305 MB + `colors` u8 131 MB. `indices` 는 (f,h,w) 가 **raster 순서의
  순증가 부분집합**임을 확인해서 49×720×1280 비트마스크 5.4 MB 로 줄일 수 있고
  `points_world`/`colors` 는 depth+K+c2w / 소스 영상에서 재계산된다 — 즉 cloud 는 순수 캐시다
  (재생성 61 s/편, 캐시 로드 2.3 s). 이 축소는 소비처 10곳(`render_bank_videos.py:213`,
  `render_pred_depth_warp.py:140`, `audit_bank_geometry.py:174`, `eval_subject_in_frame.py:193`
  등)의 로더를 전부 고쳐야 해서 이번엔 안 한다.
  · **graph marker 를 `.graph_d172` 가 아니라 `.graph_d157` 그대로 재사용**한다. 이미 굽힌
  391편을 다시 굽지 않으려는 것이고, 근거는 입력 동일성 실측이다 — `recon_and_seg/<v>` 는
  875편에 대해 `DATA/DynPose-LBM` 로의 심볼릭 링크이고, `seg_instances` 는 100K 쪽 실디렉토리
  인데 공통 877편 중 `masks.npz` 크기가 다른 건 `015b197d-…` **1편뿐**이다(sam3 를
  `--done_root DynPose-LBM --skip_done` 으로 돌려 재사용했다). 그 1편은 체인이 마커를 지워
  강제 재굽기한다. args 를 한 글자라도 바꾸면 이 재사용은 무효다.
  · 체인은 **sam3 → dynmask → geocalib → graph** 다. graph 는 ① `dynamic_mask/` 가 SAM3
  합집합으로 덮인 뒤여야 하고(recon 직후엔 전 프레임 0 이라 그대로 구우면 동적 노드 0 인
  그래프가 **에러 없이** 나온다) ② `geocalib_gravity.json` 사이드카가 있어야 한다
  (`--gravity_source geocalib` 은 없으면 assert). 사이드카는 880/10,346 뿐이라 9,466편을
  `scripts/geocalib_gravity.py`(env `geocalib`, GPU 0~3 4샤드)로 먼저 굽는다.
  · 실측 런타임 — geocalib 3.5 s/편(모델 로드 35 s 제외), graph 85.5 s/편. graph 12샤드로
  9,955편 ≈ 19.7 시간. 죽었다 판정은 PID 로, 단계 사이에는 완료율 게이트(dynmask 표본 400편
  95% / 사이드카 97%)를 둔다.
- **D171 — `build_poses.py:keyframe_follow_centers()` + `--follow_keyframes` /
  `--follow_kf_interp` (`fit_hole_ladder.py`, `emit_bank.py` 경유, 2026-09-10, 사용자 지시
  "현재 track일 경우 너무 물체를 따라가서 흔들리는데 recon이 깔끔한 sparse keyframe들을
  기준으로 translation도 interpolate해보는건 어떰?").** follow 위치 채널을 균등 keyframe
  N개로 줄였다가 `linear`/`savgol`/`cubic` 으로 다시 채운다. `0`(기본) 이면 예전 저역통과
  경로(`smooth_follow_centers`, savgol w=9 p=2)를 글자 그대로 탄다.
  · 9-uniform 3 arm 실측(bmx / snowboard / 02044b66, track 2 preset). **knob 은 arm 사이에
  사실상 동일**하고(bmx `track_truck_right` knob 0.6269 전 arm 동일, hole 0.1684~0.1775,
  τ 0.289~0.308) 바뀌는 건 떨림뿐이다. pos jerk p95 — bmx `track_truck_right`
  off 0.07089 / savgol 0.06112 / **cubic 0.03086**, snowboard off 0.00434 / savgol 0.00301 /
  **cubic 0.00128**, 02044b66 off 0.01194 / savgol 0.01223 / **cubic 0.00700**. cubic 은
  `p95 == max` 라 매듭 사이가 균일하다. path_len 대가는 bmx `track_truck_right` 에서
  savgol **−25%**(코너를 갉는다) vs cubic −15%.
  · `decision_fingerprint` 에는 **켰을 때만** 키가 들어간다(2계층 규칙) — 안 그러면 D171
  이전에 구운 뱅크의 지문이 전부 바뀐다. `SHAPE_DEFAULTS` 폴백도 같은 `0`/`"cubic"` 이라
  옛 뱅크는 `emit_bank` 재구성에서 비트 단위로 그대로다. `main()` 의 `decision_fingerprint(…)`
  호출은 positional drift 를 막으려고 키워드 인자로 바꿨다.
  · **속도 변화량으로 다시 쟀다**(`tmp/d171/kf_speed_table.py`, 17행 = 3 scene x 2 preset x
  4 hole 단). `v=|Δp|` / `|dv|=|Δ|Δp||` / `acc=|Δ²p|` / `v_cv=std(v)/mean(v)`, off=1.00 기하평균 —
  `dv_p95` savgol **0.485** / cubic **0.493**, `dv_max` 0.497 / 0.508, `v_cv` 0.849 / 0.869,
  `acc_p95` 0.636 / **0.591**, `acc_max` **0.569** / 0.604, `jerk_p95` 0.788 / **0.370**,
  `jerk_max` 0.567 / **0.209**, `path_u` 0.916 / **0.953**. **속력 변화량에서는 두 arm 이
  동률**이고 갈리는 건 경로 보존 하나다 — savgol 은 보간기가 아니라 평활기라 keyframe 을 안
  지난다. **jerk 는 판정에서 뺀다**: cubic 은 구간별 3차라 jerk 가 구간 상수가 되어(17행 전부
  `jerk_p95 == jerk_max`) 구조적으로 이긴다.
  · 관행 조사 — Blender(Bézier + Auto Clamped, Continuous Acceleration) / Maya(Auto) /
  Houdini(Bézier) / Unreal Sequencer(Cubic Auto) / Unity(ClampedAuto, Cinemachine SmoothPath 는
  C² 보장) 가 전부 **cubic + 자동 탄젠트 + 극값 clamp** 다. natural cubic 이나 plain uniform
  Catmull-Rom 을 기본값으로 쓰는 도구는 없다(Heckbert 1985). video-gen 쪽 keyframe→dense 는
  대부분 translation linear + rotation slerp.
  · **채택: `--follow_keyframes 6 --follow_kf_interp cubic`** (사용자 지시 2026-09-10 "일단
  cubic 으로 하되" → "그냥 translation 도 통일성 있게 keyframe 6개로 해줘"). `--aim_keyframes 6`
  과 같은 수이고, 두 채널이 같은 `rint(linspace(0,F-1,n))` 를 쓰므로 F=49 에서 조준과 위치가
  **정확히 같은 프레임** [0,10,19,29,38,48] 에 앉는다. CLI/`SHAPE_DEFAULTS` 기본값은 여전히
  `0`(끔) 이라 옛 뱅크는 그대로고, 채택은 config 핀으로만 한다.
  · 게이트 순서 확인 — 보간은 `fit_tau` **앞**에서 위치 채널에 들어가므로(`build_poses.py:757`
  → `offsets_world`) τ 도 hole/G1/OBB/`subject_visible_frac` 도 전부 **보간된 궤적**을 렌더해
  잰다(`fit_hole_ladder.probe()` → `measure_trajectory`). 조준은 raw 를 그대로 쓴다(D73).
  · 미해결 — `CubicSpline(bc_type="natural")` 은 프레임 0/48 에서 f''=0 을 강제해 시작·정지가
  붕 뜬다. 업계 기본값은 `not-a-knot` 계열 + 극값 탄젠트 clamp 인데 미검증이라 안 바꿨다.
- **D173 — `build_poses.py --keyframe_ease {cubic,savgol}` (회전 채널을 위치 채널과 같은
  보간기로, 2026-09-10 사용자 지시 "interpolation 을 rotation, translation 동일하게 cubic 혹은
  savgol 로 통일해서 돌려주고 track+object-centric preset 으로 scene 몇개 fit 해서 depth warp 랑
  속도, 가속도 그래프 시각화해줘봐").** D171-c 까지는 두 채널이 서로 다른 보간기를 썼다 —
  위치는 `keyframe_follow_centers` 의 natural cubic, 회전은 `smooth_kf`(선형 slerp 折れ線 +
  SO(3) Laplacian 4-pass). 같은 매듭 [0,10,19,29,38,48] 위에서 한쪽만 C² 면 "부드럽다"가 채널마다
  다른 뜻이 된다. 이제 이름이 같으면 규약도 같다:
  · `cubic` = `scipy.spatial.transform.RotationSpline` (각가속도 최소 C² 3차, **매듭을 정확히
  통과**). 위치 쪽 `CubicSpline` 과 같은 성질이라 `aim_err_at_kf_deg` 가 0 이다. 단 경계조건은
  다르다 — scipy 가 노출을 안 해서 끝점 각가속도 0(natural)을 못 건다.
  · `savgol` = 선형 slerp 折れ線 + **쿼터니언** Savitzky-Golay(창 `--follow_smooth`=9, p=2,
  반구 정렬 후 필터 → 재정규화). 위치 쪽 `savgol` 갈래와 창까지 같다. `smooth_kf` 와 마찬가지로
  매듭을 정확히 통과하지 않으므로 keyframe roll assert 에서 제외 목록에 넣었다.
  · 기본값은 **`smooth_kf` 그대로** (`build_poses`/`fit_hole_ladder` 둘 다). 배포 뱅크 재현이
  안 깨진다. `emit_bank` 는 `fixed.keyframe_ease` 를 읽으므로 새 값도 그대로 되만들어진다.
  · 실측 4-arm (`tmp/d171/run_kf_arms.py`, `tmp/d171/plot_kf_speed.py`; 3 scene x 4 preset
  `track_{orbit_right,push_in_arc_left,pull_out_arc_right,crane_up}` x 4 hole, 완주 14행).
  u_off=1.00 기하평균 — **병진**은 `path_u` u_trans6/u_cubic6 0.759 / u_savgol6 0.724,
  `v_cv` 0.563 / 0.516, `acc_p95` 0.249 / 0.335. **회전**은 `turn_deg` 0.947 / 0.955 / 0.951 로
  거의 같은데 `|dω|p95` 가 u_trans6 0.988 / u_cubic6 **1.160** / u_savgol6 **1.542**,
  `|dω|max` 0.993 / 1.062 / 1.329 — 즉 **회전을 위치와 통일하면 각가속도가 되레 커진다**.
  `smooth_kf`(Laplacian 4-pass) 가 회전에서는 여전히 가장 매끄럽다. u_trans6 와 u_cubic6 의
  병진 수치가 소수점까지 같은 것이 정합성 확인 — 조준 변경은 위치를 안 건드린다.
  · **판정: 통일 기각, `u_trans6` 채택** (2026-09-10 사용자 "추천대로 해주고"). 즉 회전은
  `--keyframe_ease smooth_kf`, 위치만 `--follow_keyframes 6 --follow_kf_interp cubic` — 이미
  `d171c_dynpose_kftrans.json` 이 그 조합이라 **config 변경 없음**. `cubic`/`savgol` 회전 갈래는
  옵션으로 남긴다 (기본값 아님). 근거는 위 표의 `|dω|` 행 — 이름을 통일해도 SO(3) 에서
  "매끄럽다"는 Laplacian 쪽이 이긴다. 릴 5편(`tmp/d171/reel/u_kf_*.mp4`) + 그래프
  33장(`tmp/d171/plots/`).
- **D171-c — `configs/bank/d171c_dynpose_kftrans.json` (2026-09-10).** d171 에서 fit 인자
  **두 줄만** 더한다 (`--follow_keyframes 6 --follow_kf_interp cubic`). route/tau 는 안 돌고
  `preset_route_d170.json` + `bank_d170` 을 재사용하므로 `hole_bank_d171` 과 행 단위 diff 다.
  · 4편 굽기 완료 (2026-09-10, GPU 3, 4/4 OK, 편당 55~68 s). **변이 집합은 24개로 동일** —
  보간이 어느 변이가 살아남는지는 안 바꾼다. 행 단위 diff: `path_len_u` 가 **`track_*` 9행에서만**
  −2.6 ~ −12.0% (|Δ| 평균 8.13%), `hole_fraction` 9행 |Δ| 5.10%, `tau_max` 8행 1.77%,
  `sweep_deg`/`subject_in_frame` **0행**. `subject_visible_frac` 은 22행이 움직였지만 최대
  절대변화가 0.0101 로 뱅크 렌더 비결정성(±0.04) 안이라 신호가 아니다. 유일한 status 뒤집힘은
  02044b66 `track_dolly_in_look_at` `approach_limited → obb_limited` (knob +10.8%) — 경로가
  매끄러워지면서 손잡이가 더 나가 binding 게이트가 approach 에서 OBB 로 넘어갔다.
  `plan_tier` 0=20 / 1=1 / 2=2 / 3=1 → **fallback ladder 가 변이의 17%(4/24)를 채운다**.
  · 릴 `tmp/d171c/reel/d171c_*.mp4` 4편 (위=d171 / 아래=d171c). 음성 대조군으로 static anchor
  변이(`stat_0__orbit_right`)를 넣었다 — follow center 가 없어 정의상 두 줄이 같아야 한다.
- **D170 — `route_presets.py --orbit_min_span` (orbit 슬롯 방위각 게이트를 인자로 노출)
  + `configs/bank/d170_dynpose_orbitspan.json` (2026-09-10, 사용자 지시 "소스 영상 방위각 폭
  조건 빼서 돌려서 depth warp 영상 보여줘봐").** `ORBIT_MIN_SPAN_DEG = 120.0` 이 상수로 박혀
  있어 `obs_az_span < 120°` 인 anchor 는 orbit 슬롯이 통째로 사라졌다(`--orbit_fallback drop`
  기본). d169 파일럿 실측으로 그 대가가 드러났다 — **anchor 40개 중 29개(72.5%)가
  `orbit_ok=False`** 이고, 빈 자리를 backfill 이 떠맡으면서 `pull_out_arc_right` tier0 통과율이
  d166 100%(4/4) → d169 16.7%(1/6) 로 무너졌다. 120 이라는 값의 근거는 실측이 아니라 "좁은
  span 에서 선회하면 점군에 자료 없는 면으로 넘어간다"는 **추론**이었다.
  · 함수 기본값·CLI 기본값 **둘 다 `ORBIT_MIN_SPAN_DEG`(120.0)** 로 두어 옛 동작을 보존한다 —
  채택된 변경이 아니라 아직 육안 판정 전인 ablation 이라, 기본값을 바꾸면 d166~d169 재현이
  조용히 깨진다. `--orbit_min_span 0` 이면 게이트를 꺼서 span 과 무관하게 orbit 을 만든다.
  · `reasons` 에 `orbit_min_span_deg` 를 같이 적는다. `orbit_ok` 는 문턱에 대한 **상대값**이라
  문턱을 안 남기면 옛 JSON 과 비교할 때 span 이 변한 건지 게이트가 변한 건지 못 가른다.
  · 회귀 확인: 기본값으로 `016a6379…` 를 라우팅해 기존 `preset_route_d169.json` 과 재귀 diff —
  차이는 새 진단 키 `orbit_min_span_deg: 120.0` 3개(`anchors[0..1].reasons`, `reasons`)뿐.
- **D169 — DA3 depth confidence 저장 `recon_da3.run_da3(return_conf=)` + `dynpose_ingest.py
  --save_conf/--no_save_conf` (2026-09-09, 사용자 지시 "다음에 돌릴때는 conf도 저장해줘").**
  DA3 는 `Prediction.conf` (N,H,W) 를 내놓는데 `run_da3` 가 4-tuple 만 돌려주며 버리고 있었다.
  이제 `return_conf=True` 면 5번째 값으로 conf 를 주고, ingest 가 `conf/` 에 depths 와 같은
  float16 EXR 로 남긴다 (`load_depths(".../conf")` 로 그대로 읽힘, 편당 +18 MB → 10.4k 편
  +190 GB). 함수 기본값 `return_conf=False` 는 옛 동작이라 `recon_and_seg_single.py:54` 는
  그대로 돌고, CLI 기본값은 `--save_conf`(켬)다. `recon_done()` 은 **일부러 conf 를 검사하지
  않는다** — 넣으면 이 플래그 이전에 구운 편들이 전부 미완료로 되살아나 재계산된다.
  2026-09-09 진행 중이던 dynpose-100k 10.4k 굽기는 사용자 지시로 이번 세대는 그대로 두므로
  `conf/` 가 없다.
- **D169 — `scene_graph/lift.py:valid_pixels(conf_mode=)`.** DA3 conf 는 `[0,1]` 이 아니라
  `expp1 = exp(x)+1 ∈ [1, inf)` 다 (`depth_anything_3/model/dpt.py:49,296`). 그래서 옛 기본
  임계 `conf > 0.5` 는 **전 픽셀을 통과시키는 no-op** 이고, conf 를 저장하기 시작하면 켜자마자
  아무것도 안 거르는 상태가 될 자리였다. `conf_mode="relative"` 는 DA3 자신이 쓰는 방식
  (`utils/alignment.py:93` 의 `conf >= median_conf`)대로 non-sky 픽셀 분위수를 임계로 삼는다
  (`conf_threshold` 가 버릴 분위: 0.5 = 하위 절반). 함수 기본값 `absolute` 는 옛 동작.
  참고로 **scene scale `S` 는 conf 와 무관하다** — `scene_graph/scale.py:71-73` 의 유효 조건은
  `finite ∧ z>0 ∧ ~sky` 뿐이라 conf 없이도 정상 계산된다.
- **D169 — `route_presets.py --slot_pair_fill rotate|substitute` (2026-09-08, 사용자 지시
  "가능한 preset 후보풀을 뽑아두고 (s_curve 제외) 5개가 안나오면 후보풀에서 다시 뽑으면
  되잖아").** `grid_slot_pair()` 는 video 해시로 고른 씨앗 쌍의 한 짝이 그 씬에 없으면 **쌍을
  통째로 버리고** `GRID_SLOT_PAIRS` 목록의 다음 쌍으로 넘어갔다. D167 이 `--orbit_fallback
  drop` 을 기본값으로 만든 뒤 이게 문제가 됐다 — 8쌍 중 3쌍이 orbit 을 물고 있는데 d166
  파일럿 40 anchor 중 29개(72.5%)가 `orbit_ok=False` 라, 그 씬들이 살아 있던 짝까지 잃고
  전혀 다른 쌍에 착지했다. 21편 실측으로 6편(28.6%)의 쌍이 바뀌었고 route 산출 preset 이
  `crane_up` 8→17행, `dolly_in_look_at` 5→11행으로 쏠리고 `static_look_at` 은 5→2행으로
  줄었다. `substitute`(CLI 기본값)는 **살아남은 짝을 유지하고 빠진 자리만** video 해시로
  회전시킨 `GRID_ALLOWED_SLOTS` 후보풀에서 메운다: `recede+orbit` → d168 `advance+vertical`
  (두 짝 다 교체) 대신 `recede+vertical`, `orbit+static` → `static+arc`, `advance+orbit` →
  `advance+static`. 21편 preset 분포가 crane_up 11 / dolly_in_look_at 5 / static_look_at 5 로
  d166 값에 돌아온다. 씨앗 쌍 두 짝이 다 살아 있는 15편은 `rotate` 와 **글자 단위로 같은**
  답을 내므로 회귀가 없다. 함수 기본값은 `rotate`(옛 동작, d166/d168 재현), CLI 기본값이
  `substitute` — D143/D167 과 같은 규약. `preset_route.json` 의 `reasons` 에 `grid_seed_pair`
  와 `slot_pair_fill` 을 같이 적는다 (`grid_slot_pair` 만 보면 대체 여부를 못 본다).
  세대 config `configs/bank/d169_dynpose_slotfill.json` 은 d168 에서 이 인자 하나만 다르다.
- **D169 — `extract_nouns_vlm.py --num_frames` (기본 6, 2026-09-08 사용자 지시 "우리 방식대로
  뽑되 프레임은 6개로 늘려줘").** `--frame_mode multi` 가 `np.linspace(0, N-1, 4)` 로 고정
  4장이었다. 49프레임을 12프레임 간격으로 훑는 것이라 그 사이에만 나왔다 사라지는 물체를
  통째로 놓치고, dynpose 9.5k 편은 씬당 물체 수가 Vista4D-Eval-Data 보다 많아 그 구멍이 SAM3
  앵커 수로 곧장 번진다. 함수 기본값 4 = D145 51편 비교표의 옛 동작, CLI 기본값 6.
  이걸 쓰는 이유는 dynpose-100k 배포본의 `category/category.json` 이 "형용사+명사" 구
  (`fluffy dog`, `man holding dog`)라 Sa2VA 용 referring expression 이고, SAM3 text PCS 와
  기존 코퍼스 880편(D145 평명사)과 어휘 축이 갈리기 때문이다.
- **D169 — dynpose-100k 대량 ingest 드라이버 `scripts/dynpose_ingest.py`
  (2026-09-08, 사용자 지시 "recon_and_seg 먼저 일단 돌릴 수 있는거 최대한 다 sharding 해서
  돌려놔줘" + "모델 올렸다 내렸다 하면 오래 걸리니까 단계 나눠서 최대한 병렬로 다 돌리고
  다 되면 다음 단계로").** 기존 경로는 `recon_and_seg_single.py` 를 **영상당 프로세스 하나**로
  띄우는 bash 래퍼였다. DA3NESTED-GIANT-LARGE 는 로드가 43.7 s 인데 추론은 편당 ~29 s 라,
  1만 편이면 로딩이 전체의 60% 를 먹는다. 새 드라이버는 프로세스당 DA3 를 **한 번만** 올리고
  배정된 영상을 전부 돈다.
  · `--stage link|recon|metadata|sam3|dynmask|launch` 하나짜리 python 드라이버. 단계 사이는
    배리어 — `launch` 는 **한 단계만** GPU 여러 장에 부채꼴로 띄우고 전부 끝날 때까지 기다린다.
    `--gpus` 값은 0~4 밖이면 assert 로 막는다.
  · `recon` 은 `recon_and_seg_single.main()` 에서 이 코퍼스가 타는 분기(da3 / seg_keywords 없음 /
    `keep_recon_sky` / dse 없음 / scene_scale 1.0)만 그대로 옮긴 것이다. 같은 scene 을 다시
    돌려 기존 산출물과 대조: `intrinsics` 완전 일치, `cam_c2w` maxabs **5.5e-6**, `depths`
    상대오차 **6.9e-4**(float16 양자화), `sky_mask` 비트 일치. (`dynamic_mask` 만 다른데,
    기존 코퍼스 쪽은 이미 `dynpose_dynamic_mask_from_seg.py` 가 덮어쓴 상태라 그렇다.)
  · 중간에 죽은 폴더가 "완료"로 보이지 않게 `<out>.partial` 에 쓰고 마지막에 rename 한다.
    한 편이 죽어도 그 편만 FAIL 로 세고 계속 간다.
  · **입력은 `video_input.mp4` 뿐**이다 (사용자 지시). `inpaint_result.mp4` 는 저자들이 이미
    워프·인페인트를 돌린 결과물이라 소스 영상이 아니다.
  · 출력 루트를 `DATA/DynPose-100K` 로 **새로 판다**. `DATA/DynPose-LBM/metadata.csv` 는
    d157/d166/d168 뱅크가 코퍼스 목록으로 읽는 파일이라 여기에 9천 편을 더하면 그 뱅크들을
    다시 굽지 못한다. shard 0000 의 880편은 symlink 로 재사용하므로 디스크는 안 는다.
  · `nouns` 단계 추가 — `extract_nouns_vlm.py`(우리 프롬프트, `--num_frames 6`)를 recon 이 끝난
    영상에 샤딩해서 돌리고 shard 별 `vlm_nouns.json` 을 남긴다. vLLM 서버(22002)는 **밖에서
    미리 띄운다** — 여기서 올렸다 내리면 샤드마다 로드가 반복된다. 워커는 전부 같은 서버를
    두드리므로 `--workers N` 은 GPU 수가 아니라 동시 요청 수다. `--skip_done` 은 기존
    `vlm_nouns.json` 의 `(video, multi, vlm)` 레코드를 재질의하지 않는다.
  · `metadata` 는 shard 0000 을 기존 `DynPose-LBM/metadata.csv`(D145 VLM 명사) 그대로 쓰고,
    나머지는 `--noun_source vlm`(기본)이면 위 `nouns` 산출물에서, `category`(옛 동작)면
    `category/category.json` 의 `dynamic` 배열에서 채운다. 기본을 바꾼 이유는 두 어휘 축이
    다르기 때문이다 (csv `dog, person, shoe, chair` vs category `fluffy dog, man holding dog`
    — 후자는 Sa2VA referring expression 이라 SAM3 text PCS 에 그대로 못 넣는다). 출처는
    `metadata_provenance_d169.json` 에 남긴다.
  · `sam3` / `dynmask` 는 기존 `sam3_seg_instances.py` / `dynpose_dynamic_mask_from_seg.py` 에
    위임한다 (SAM3 도 그쪽이 이미 프로세스당 1회 로드다). 새로 짠 코드가 아니다.
- **D168 — fit 실패 자리를 다음 층 preset 으로 메우는 routing/retry
  (`sample_camera_bank.py --variant_pool full`, `fit_hole_ladder.py --fallback_ladder`
  + `tag_suspects` 태그 2종, `configs/bank/d168_dynpose_gate.json`) (2026-09-08, 사용자 지시
  "가능한 preset 들 최대한 돌리고 결과적으로 안나오면 다른 track 없는 preset 쪽을 본다던지
  target anchor 를 포기하고 free-moving 으로 대체한다던지 이런 안전장치" + "최대한 최신
  세팅을 쓰되 routing 및 retry 정도로만").** `(anchor, preset)` 하나가 `clamped_low` 로 끝나면
  그 행이 그대로 뱅크에 남고 끝이었다 — 다른 preset 을 대신 시도하는 경로가 파이프라인
  어디에도 없었다 (preset 대체는 τ 단계 `plan_variants` backfill 에만 있고, 그건 fit 이 돌기
  전이라 fit 판정을 못 본다). d166 dynpose 21편 105행 중 `clamped_low*` 23 + `static` 5 =
  **26.7%** 가 그렇게 남았다.
  · `--variant_pool budget|full` (τ) — `full` 이면 예산 밖 preset 까지 변이로 깔아 두고
    행마다 `plan_tier` 를 찍는다 (0 본 슬롯 / 1 non-track 조준 / 2 `track_*` / 3 target 포기
    free-moving). `budget`(기본)이면 전 행 tier 0 이라 d167 과 같다.
  · `--fallback_ladder` (fit) — 층을 나눠 돌면서 실패한 자리를 다음 층 preset 으로 메운다.
    **층 0 을 전 anchor 에 대해 다 돌고 나서** 층 1 로 내려간다 (anchor 별로 내려가면 anchor A
    의 예비가 anchor B 의 본 슬롯을 밀어내 2×2 격자가 깨진다). 쓸 만한 변이가
    `--fallback_target 5` 개 모이면 멈춘다 — 전부 통과하는 scene 은 렌더 수가 d166 과 같다.
  · **게이트는 한 개도 안 늘렸다.** 재시도 판정은 뱅크가 이미 갖고 있는 두 열
    (`status` / `suspect`)을 읽고, 인자 이름·토큰 매칭도 코퍼스 단계
    (`vista4d_bank_to_dl3dv.py --drop_status` / `--drop_suspect`)와 같게 맞췄다
    (`--retry_status` / `--retry_suspect`). 구현도 `tag_suspects` 를 행 하나에 그대로 부르는
    것이지 재구현이 아니다 — 재구현하면 retry 기준과 코퍼스 기준이 갈라진다.
  · `tag_suspects` 태그 2종 추가. ④ `aim_target_subject_lost` = 기존 ①
    `aim_free_subject_lost` 의 **여집합**(`aim != free` ∧ sif < `--suspect_in_frame`, 임계
    공유). d166 실측 해당 10행 중 **7행**이 status 도 suspect 도 없이 조용히 통과했다.
    ⑤ `hole_over_budget` = `hole_fraction > --suspect_hole`(0 이면 꺼짐). 0.35 는 d166 config
    `_rung` 이 이미 합격 기준으로 쓰던 숫자다. d166 실측 `hole_fraction > 0.35` 14행 중 9행이
    `solved` 이고 기존 어휘로 잡히는 건 1행뿐이었다.
  · `aim_free_subject_lost` 는 retry 어휘에서 **일부러 뺐다** — free-moving 은 조준 자체를
    안 하는 게 정의라(D157 aim=free sif p10 = 0.077) 넣으면 targetless 변이가 통째로 사라진다.
  · 회전 전용 preset(`pan_*`)에 별도 하한을 두지 않는다 — `ROTATION_ONLY_PRESETS` 가 이미
    D53 `frozen` 진단의 예외 목록이고, 실측상 `pan_deg` 는 `KNOB_RANGE` 하한 2.00(= `clamped_low`
    5행)이거나 ≥19.31(= `solved` 16행)로 갈려 중간값이 없다.
  · **전부 옵션 분기다** — `--variant_pool budget`(기본) + `--no_fallback_ladder`(기본) +
    `--suspect_hole 0`(기본)이면 d167 뱅크와 같다. τ 뱅크에 `plan_tier` 가 없으면
    `--fallback_ladder` 는 no-op 으로 떨어지고 그 사실을 표에 찍는다.
- **D166 — scene 당 카메라 5개 구성 (`route_presets.py --slot_plan grid2x2 --free_moving
  --max_anchors`, `sample_camera_bank.py --preset_route`, `configs/bank/d166_dynpose_grid5.json`)
  (2026-09-08, 사용자 지시 "scene 별로 4개정도만 카메라를 만들고싶은데" → "free-moving을 하나
  넣어서 scene 마다 5개로 나오도록").** `goals.md` 의 "scene ↑ (10k), preset ↓" 를 집행한다.
  D157 실측 scene 당 850 s 중 fit 이 503 s(78%)이고 fit 은 변이 수에 선형이라, 변이를
  144.4 → 5 로 줄이는 것이 scene 수를 늘릴 유일한 예산이다.
  · 구성 = `anchor{a,b} × slot{P1,P2}` (2×2) + free-moving 1. 슬롯 쌍은 **video 해시**로 골라
    두 anchor 가 공유한다(`GRID_SLOT_PAIRS` 8쌍) — 슬롯까지 anchor 마다 다르면 target 축과
    motion 축이 섞여 4칸 중 어느 것도 서로의 대조군이 아니게 된다.
  · `--free_moving rotate|datadop` — target 절 **없는** 변이를 scene 당 1개, 격자 밖 첫 anchor
    에만. 격자에 넣으면 두 anchor 의 캡션이 (target 절이 없어서) 글자 단위로 같아진다.
    `track_` 접두사가 안 붙는 것을 assert 로 못 박는다 (사용자 지시 "track+free-moving 제외").
  · `--preset_route` (`sample_camera_bank`) — `preset_route_v1` JSON 의 anchor 별 `slots` 를
    읽어 (nodes × presets) **합집합 격자를 안 돈다**. 없으면 기존 합집합 경로 그대로.
    합집합이면 away side 차이(`truck_left` vs `truck_right`)만으로 2×4=8 로 부푼다.
  · `--max_anchors` (`route_presets`) — `sample_camera_bank` 와 같은 이름·같은 의미.
    `--max_dynamic_anchors 2 --max_static_anchors 2 --max_anchors 2` 가 곧 "dyn+dyn 우선,
    부족분만 stat" 이다 (`pick_main_anchors` 가 동적 우선 정렬이라).
  · 2×2 후보에서 `lateral`(truck_*) 제외 — aim=free 인데 캡션에 target 절이 있다
    (D157 sif-only 탈락 7,910행 중 3,024 = 38.2%, task #134). `rotate`(pan_*) 는 캡션이 이미
    targetless 라 free-moving 슬롯으로 옮겼다. grid 에선 `recede` 를 track 보너스에서도 빼
    `track_dolly_out`(track_* ∧ aim=free)이 안 나오게 한다.
  · fit 사다리는 `0.20` 한 단. D157 실측 ① rung 0.35/0.50 이 코퍼스의 71.6% 를 먹으며 합격률
    38.3%, 전부 0.20 이면 83.0% ② **rung 은 강도 축이 아니다** — preset 을 고정하면 path_len
    비가 1.04~1.23배에 P(b>a) 0.520~0.538 이다 (pooled 1.58배는 Simpson).
  · **전부 옵션 분기다** — `--slot_plan full`(기본) + `--free_moving off`(기본) +
    `--preset_route` 미지정이면 d157 경로 그대로다 (같은 씬에서 anchor 4 × preset 14 = 56 재현).
- **D166 backfill — scene 당 5는 **상한**이고 모자라면 다른 preset 으로 채운다
  (`route_presets.py --target_variants` + `anchors[].backfill` + scene 단위 `free_moving`,
  `sample_camera_bank.py plan_variants()` / `--target_variants`) (2026-09-08, 사용자 지시
  "최대 5개이고 … fitting해서 5개가 안되면 경우에따라 추가로 다른 가능한 preset을 시도").**
  라우팅이 5개를 맞춰 놔도 뱅크가 anchor 를 **소스 frame 0 가시성**(`min_subject_points` /
  `min_subject_area`)으로 더 떨어뜨린다. 파일럿 21편 실측 분포가 `5×11 / 3×7 / 2×3` (82행,
  편당 3.90) 이었고, 짧은 6편 중 **5편이 anchor 를 통째로 잃었다**. 라우팅은 렌더를 안 하므로
  이걸 예측할 수 없다 — 배분을 뱅크로 옮겼다.
  · `sample_camera_bank` 의 anchor 루프를 **3단**으로 쪼갰다. ① 가시성 판정을 전량 먼저 돌려
    살아남은 anchor 를 확정(`subject_points` 캐시) ② `plan_variants()` 가 예산을 배분
    ③ 기존 preset 루프는 그 작업 목록을 돈다.
  · 배분 규칙(사용자 확정): live anchor 2+ 이고 움직이면 `{a,b} × 슬롯 2` = 4, 그 외(정지 /
    anchor 1개)는 `a × 슬롯 4` = 4. 거기에 free-moving 1. 모자라면 `backfill`(라우팅이
    keep_pair 밖으로 밀어 둔 슬롯, `GRID_ALLOWED_SLOTS` 순서)에서 round-robin 으로 채운다.
    3번째 anchor 는 안 끌어온다 — 그 씬만 target 축이 3-way 가 되어 2×2 대조가 깨진다.
  · **free-moving 이 scene 단위 키로 올라갔다.** 전에는 `routed[0]["slots"]` 에 붙어서 그
    anchor 가 죽으면 같이 사라졌다 (`023615b3` / `01d32f88`). 이제 **살아남은 첫 anchor** 에
    붙는다 — anchor 를 안 쓰는 변이라 같이 죽을 이유가 없다.
  · `track_*` 게이트(D77 `moving` / D128 `center_drift_u`)의 집행부를 `track_ok()` 로 올렸다.
    preset 루프 안에서 `continue` 하면 슬롯이 조용히 비는데, 배분 단계에서 걸러야 그 자리를
    backfill 로 채울 수 있다. **판정 자체는 한 글자도 안 바뀌었다.**
  · 품질 게이트(hole / sif / behind) 탈락 행은 backfill 을 **안 부른다** (사용자 확정) —
    코퍼스 필터가 나중에 거른다. 트리거는 `enumerate 결과 < target_count` 뿐이다.
  · 검증 5편(파일럿에서 3/5/2/3/2 였던 씬): tau 단계 전부 `총 5/5`.
  · `target_count` 가 0 이면(=`--preset_route` 없음 또는 그 키가 없는 옛 JSON) 배분을 안 한다
    = 옛 동작(anchor 전량 × preset 전량).
- **`CinemaTraj/scripts/{sample_camera_bank,fit_hole_ladder}.py --timing_json` — (anchor, preset)
  단위 소요시간 실측 (2026-09-08, 사용자 지시 "preset당 얼마나 걸리는지 단계마다 측정해줘").**
  `logs/d157/*.{graph,cloud,route,tau,fit,emit}.log` 의 mtime 은 **단계**까지만 나눠준다 —
  "fit 이 scene 중앙값 503 s 로 전체의 78%" 까지는 보이지만 그 안에서 preset 16종 중 어느 것이
  비싼지는 안 보이고, preset 을 줄일 때 무엇을 자를지 못 고른다.
  · 스키마 `{format, video, stage, total_seconds, entries[]}`, 각 entry 는
    `{anchor_id, preset, rungs, variants, seconds}` + fit 은 `solve_calls`(그 preset 이 쓴
    이분법 렌더 호출 수). `perf_counter()` 로 preset 본문 진입/이탈만 감싼다.
  · **기본값 None 이라 안 주면 파일도 안 쓰고 집계 표도 안 찍는다** — 진행 중인 d157 4샤드와
    비트 단위로 같은 경로. 켜도 측정은 `perf_counter` 두 번뿐이라 굽기 시간에 영향이 없다.
  · 끝에 preset 별 정렬 표(`n / sec / s per anchor / calls / s per call`)를 stdout 에 찍는다.
- **`CinemaTraj/scripts/eval_subject_in_frame.py --subject_occlusion` — subject **가림** 열
  (`subject_visible_frac` / `_min` / `_mean`) (2026-09-08, 사용자 지시 "vista 데이터로 학습한
  … 비교한 표에서 subject_visible_frac 추가해서 보여줘").** 기존 세 열로는 가림이 안 잡힌다 —
  `subject_in_frame` 은 실루엣 **중심**이 상자 안인가(위치), `subject_pixel_coverage` 는 화면
  **면적비**(크기)라 상판·기둥에 반쯤 가려진 subject 가 둘 다 멀쩡히 통과한다.
  · 새 `occlusion_series()` 는 프레임마다 subject 점만 그린 실루엣(`render(..., subset=)`)을
    한 번 더 렌더해 전체 렌더 depth 와 비교하고, `drawn & (depth_full < depth_alone − 0.02·S)`
    를 가려진 픽셀로 센다 — `lbm/gates.py:338-349` · `sample_camera_bank.py:690-698` 과 같은 식.
    `render.CloudRenderer.measure` 의 `num_subject_points` 판은 픽셀수/점개수라 **밀도**이지
    비율이 아니어서 쓰지 않는다.
  · 분모는 그 프레임 실루엣 픽셀 수라 화면 밖 성분이 안 들어간다 (프레임 이탈은
    `subject_in_frame` 의 몫). 실루엣이 통째로 비면 0. 프레임 median 으로 접고 최악 프레임을
    `subject_visible_min` 으로 따로 남긴다.
  · `0.02·S` 의 `S` 는 `scene_graph.json["scale"]["S"]` — 게이트와 같은 게이지여야 판정이 같다.
  · **기본값 off** 라 안 주면 열이 안 붙고 기존 표와 비트 단위로 같다. 렌더가 프레임당 1회
    늘지만 두 번째 패스는 subject 점만이라 첫 패스보다 훨씬 싸다.
  · 모드 주의: `warp_1to1`(`--no_temporal_persistence`)에서는 정적 점이 프레임 f 것만 남아
    **가림막이 사라지므로** 이 열을 재면 안 된다. cloud 모드 전용.
- **`CinemaTraj/scripts/run_gendop_eval.py` — `CORPORA["vista_d121_snowboard"]` (2026-09-07,
  사용자 지시 "snowboard처럼 물체가 빠르게 움직이는 video에서도 잘 tracking같은걸 하는지").**
  `vista_d121` 과 코퍼스·ckpt 가 같고 split 만 다르다 — snowboard 는 **학습 split** 이라
  `seg_list_vista4d_test.txt` 에 없어서 875 val 런의 어느 eval 폴더에도 예측이 0건이다.
  `eval_testset.py --set test_seg_list=<snowboard track27>` 로 같은 epoch100·seed42 로 27 entry
  를 따로 뽑고 그 폴더를 `ours` 로 물린다. `ours` 는 두 arm (d123_da3 / d124_da3_molmo2) 뿐 —
  d133 은 snowboard 재추론을 안 돌렸다.
  · 걸림돌 하나: gdstyle 캡션 생성기가 읽는 `da3/captions_gendop/<idx>_tag.json` 도 test split
    씬에만 있다. `caption_cameras_datadop.py --sets latentcam --no_llm --latentcam_split <...>`
    로 태깅만 27편 추가 생성하면 된다 (LLM 불필요 = vLLM 안 띄움).
- **`CinemaTraj/scripts/rank_traj_text_match.py` — pred 궤적이 **텍스트대로 움직였는지**를 GT
  궤적 대비로 재고 arm 격차로 정렬 (2026-09-07, 사용자 지시 "충돌말고 우리 모델이 text에 따라
  카메라가 잘 움직이고 gendop는 아닌 영상으로 보여줘").** 충돌률·subject_in_frame 은 결과물의
  안전성/구도를 재므로 "arc right 지시를 반대로 냈다" 같은 **지시 불이행**을 못 잡는다 (구도가
  우연히 맞으면 통과). 캡션은 GT 궤적에서 뽑은 문장이므로 "텍스트를 따랐나" = "GT 궤적과 같은
  방향·크기·회전으로 갔나" 로 환원해, frame0 앵커를 푼 rel pose 위에서 `dir_err_deg`(순변위
  방향각) / `len_ratio`(경로길이비) / `rot_err_deg`(마지막 프레임 geodesic) 를 낸다.
  `--rank_pair OURS=THEIRS` 로 `dir_err(THEIRS) − dir_err(OURS)` 내림차순 정렬.
  · 정지 지시는 GT 순변위가 0 이라 방향각이 정의되지 않는다 → `--static_u` 아래는 NaN 으로
    두고 정렬에서 뺀다. 안 그러면 static 엔트리가 "방향 불일치"로 상위에 올라온다.
  · 실측 (d121 val 875, `dir_err` med / mean / `len_ratio` med / `rot_err` med):
    d133 6.85 / 16.17 / 0.8378 / 2.89 · d124 7.71 / 18.46 / 0.9077 / 3.31 ·
    d123 9.46 / 21.54 / 0.8749 / 3.00 · gd_style 14.87 / 22.85 / 0.5431 / 11.08 ·
    gd_rgbd 22.42 / 36.42 / 0.5262 / 15.74.
  · gd_style 의 median 이 14.9° 라 **계통적 축 뒤집힘은 아니다** — 아래 데모 엔트리의
    170°대는 그 arm 의 per-entry 실패다 (변환기 규약 버그로 오독하지 않도록 기록).
- **`render_pred_depth_warp.py --caption_dir LABEL=DIR` — arm 이 **실제로 받은** 조건 텍스트를
  eval 폴더 밖에서 읽는다 (2026-09-07, 사용자 지시 "gdstyle은 text가 move forward 이래야하는거
  아니야?").** 열 라벨을 `load_condition_text(eval_dir, name)` 로만 뽑고 있었는데 GenDoP
  gdstyle arm 의 입력은 `results/20260906_d156_gendop_d121/text_gendop_style/` 이고 eval 폴더의
  `<name>_caption.json` 은 **대조용으로 복사된 우리 d121 캡션**이다 — 궤적·렌더는 맞고 라벨
  문자열만 틀린 종류라 영상만 보면 안 들킨다. `--caption_dir` 의 라벨이 `--eval_dir` 에 있는지
  assert 하고, 안 주면 종전대로 eval 폴더에서 읽는다 (기존 런과 동일).
- **`slice_subject_in_frame.py` — `--json` 다중 입력 + `--group_scene` + `--group_meta FIELD`
  (2026-09-07, 사용자 지시 "subject in frame도 scene 별로 표 만들어주고 preset type 중
  free-moving (no anchor)인거랑 object-centric (with anchor)인거 분리해서 표 만들어줘봐").**
  arm 을 GPU 두 장에 나눠 돌리면 결과 JSON 이 갈리고 `gt` 열이 양쪽에 들어 있어 그냥 합치면
  두 번 세어진다 — `(scene, entry, arm)` 중복은 먼저 준 파일을 남기고 개수를 print 한다.
  씬별 / `prompts.json` 필드별 그룹은 split 파일을 손으로 만들 이유가 없어 플래그로 뒀다
  (`--group_meta aim` → `aim=free` free-moving / `aim=look_at` object-centric).
  · 실측 (warp_1to1, center_box 0.80, n=875): **arm 간 격차가 거의 전부 `aim=look_at`(415) 에
    있다.** `aim=free`(460) 은 gt 0.6406 / d133 0.6430 / d124 0.6406 / d123 0.6345 /
    gd_rgbd 0.5918 / gd_style 0.6295 로 gt 와 우리 3 arm 이 사실상 동률인데, `aim=look_at` 은
    gt 0.9218 vs d133 0.7219 / d124 0.6937 / d123 0.6636 / gd_rgbd 0.5695 / gd_style 0.4923.
  · 씬별로는 bmx-bumps(220) 만 우리가 gendop 보다 낮다 (d123 0.3212 / d133 0.3436 /
    d124 0.3446 vs gd_rgbd 0.4262 / gd_style 0.4312, gt 0.6535).
- **`gendop_preds_to_eval_dir.py --no_scale_token` + `run_gendop_eval.py --no_scale_token`
  — scale 토큰을 되나눠 **공식 배포 판본**과 크기를 맞춘다 (2026-09-07, 사용자 지시 "적용
  안한게 공식 배포 버전인 것 같으니 적용 안하고 다시 metric 측정해줘").** 배포 `eval.py:233`
  의 `c2ws[:,:3,3] *= scale_value` 는 **numpy 사본**에만 걸리고 그 사본은 `draw_json` 궤적 PNG
  전용이다 — pred JSON 을 쓰는 `pose_normalize`(`:239`) 는 scale 이 안 걸린 torch `camera_pose`
  를 받고, `core/utils.py:206 token_to_camera` 도 token 7·8 만 `fx,fy` 로 쓰고 **token 9(scale)
  는 안 쓴다**. 우리 `gendop_release_infer.py:243-247` 은 `exp(coords[:,9]/bins*4-2)` 를 곱해
  npz 에 넣고 곱한 값을 `scale` 로 같이 저장하므로, 변환기에서 되나누면 **정확히** 배포 판본이
  된다 (재추론 불필요).
  · 실측 scale 분포(875 entry): `rgbd` med 0.5028 (min 0.1534 / max 2.0842),
    `gdstyle` med 0.4874 (min 0.1821 / max 1.2840) — 되나누면 궤적이 **median 1.99배** 커지고
    분포 폭이 13.6배라 절대 크기에 반응하는 지표(충돌률·τ)가 그만큼 달라진다.
  · 되나눔 검증은 **de-anchor 후에** 해야 한다. eval JSON 은 world pose
    `gt_cv[0] @ pred_rel`(변환기 :163) 이라 translation 이 `t_gt0 + R_gt0·t_rel` = `t_rel` 의
    **아핀**함수다 — raw `transform_matrix` 를 비교하면 순수 배율로 안 보인다.
    `inv(ref[0]) @ pred` 로 풀면 `rmax` 비가 정확히 `1/scale` (5개 프로브 전부 일치).
  · 기본값은 `--scale_token`(True) 이라 **기존 런과 비트동일**. eval 폴더 접미사 `_noscale`.
- **`CinemaTraj/scripts/plot_eval_frustums.py` — eval arm 궤적을 카메라 절두체로 3D 비교
  (2026-09-07, 사용자 지시 "카메라 frustum 시각화도 보여줘").** `render_pred_depth_warp.py` 는
  "그 카메라에서 보면 어떻게 보이나"만 보여줘서, warp 이 중간에 꺾일 때 그게 좌표 규약 뒤집힘인지
  모델 발산인지 회전 튐인지를 못 가른다. 절두체를 프레임 순서(viridis)대로 세워 경로선과 같이
  그리면 셋이 그림에서 갈린다. 축 범위는 **arm 전체를 합쳐 한 번만** 잡는다 (arm 별로 잡으면
  발산 궤적이 자동 축소돼 정상처럼 보인다). `--mark_frame` 으로 의심 인덱스에 빨간 x.
  절두체 화각은 `fl_x/fl_y/w/h` 실측이고 깊이만 `--frustum_len` 으로 줄인다.
- **`gendop` p30 raw eval 폴더 2종** (`eval_dir_gendop_{text,rgbd}_raw`) — 아래 Fixed 항목의
  올바른 판독 경로. 추론은 재사용하고 `--stage evaldir --pose_length 30 --raw` 만 다시 돌렸다.
- **`gendop_preds_to_eval_dir.py --resample gendop_slerp` + `run_gendop_eval.py --resample`
  — 30 pose → 49 프레임을 GenDoP **자신의** 보간기로 (2026-09-07, 사용자 지시 "그냥 30으로
  뽑고 interpolate, sampling하는 코드가 원본에 있을테니 49프레임으로 맞춰서").** 기존
  `index_pick` 은 `np.rint(np.linspace(0,29,49))` 라 30개 중 19개가 두 번 뽑혀 **같은 pose 가
  연속 2프레임**인 계단이 생긴다 — 궤적이 정지→점프를 반복하는 것으로 보이고, 그 계단이
  속도 기반 지표(jerk·caption motion 태그)에 그대로 새어 든다.
  · 새 경로는 `GenDoP/core/utils.sample_from_dense_cameras` (회전 quaternion SLERP + 이동 LERP)
    를 그대로 부른다. 원본 `eval.py:256-262` 가 30 pose 를 120프레임으로 늘릴 때 쓰는 함수다.
  · 시간축은 `t = i/(n-1)` — 원본은 `i/120` 이라 마지막 pose 를 안 밟지만(궤적의 99.2%만 씀),
    우리 GT 49프레임은 소스 클립 전체를 덮으므로 끝점을 포함해야 같은 구간을 비교한다.
  · 호출 제약 2개는 원본 버그를 우회하지 않고 그대로 따랐다 — (1) `fraction` 이 (B,M) 인 채로
    (B,M,4) 쿼터니언에 곱해져 **M>1 이면 브로드캐스트가 깨진다** → `eval.py` 처럼 시각 하나씩
    루프. (2) `is_valid_rotation_matrix` 가 `torch.ones_like(matrix)` 와 비교해 **float64 를 주면
    dtype mismatch 로 죽는다** → float32 로 넘긴다.
  · SLERP 은 좌불변, LERP 은 아핀이라 `rel_anchor` 와 **교환된다** — 호출 위치를 `index_pick`
    과 같은 자리에 뒀다. `rmax = max_f |rel[f,:3,3]|` 도 불변(보간 내부점은 knot 을 못 넘음):
    두 arm 모두 rmax gain median 3.4498 로 일치.
  · 기본값은 `index_pick` 이라 **기존 런과 비트동일**. eval 폴더 접미사 `_slerp` 로 갈린다.
  · 실측(200 entry, `eval_dir_gendop_rgbd_raw` → `..._raw_slerp`): 중복/0 스텝 20.0/48 → **1.3/48**,
    스텝 max/median 2.35 → **1.53**, p95 50.75 → **7.65**.
  · `gendop_slerp` 은 `GenDoP` env 로 돌려야 한다(`core/utils` 가 torch·trimesh·megfile 을
    import) — `stage_evaldir` 이 `--resample` 에 따라 인터프리터를 고른다.
- **`eval_subject_in_frame.py --no_temporal_persistence` / `--limit`
  — subject_in_frame 을 **1:1 depth warp** 에서 잰다 (2026-09-07, 사용자 지시 "subject in frame을
  vista4d 렌더 말고 depth warp까지만으로 측정해줘").** `CloudRenderer.visible_at` 이
  `temporal_persistence=True` 면 `visible[:, frame]` (~2.7M 점) 를,
  `False` 면 `indices[:,0]==frame` (~55k 점) 를 쓴다. 후자가 "프레임 f 의 depth 를 그 카메라로
  warp 한 것"이고 다른 모델과 같은 조건이다.
  · **정정 (2026-09-07, 사용자 지적 "occlusion 같은것도 나중에 보려면 static만 누적한 …"):**
    `True` 를 "49프레임 전량 누적"으로 적은 것은 틀렸다. `Vista4D/utils/point_cloud/point_cloud.py:63`
    이 `visible = (visible & dynamic_mask) | static_mask` 라서 **정적 점만** 전 프레임 True 이고
    동적 점은 자기 프레임 하나에서만 True 다. 즉 기본값 `True` 가 곧 **"static 누적 + dynamic 1:1"**
    이고, 두 모드가 갈라지는 지점은 정적 점 하나뿐이다.
    - **가림(occlusion) 판정에 쓸 모드는 `True`** — 벽·가구가 다른 프레임에서만 관측됐으면
      NTP 에서는 점군에 아예 없어서 subject 가 안 가려진 것처럼 보인다.
    - **동적 subject** — 실루엣은 두 모드가 동일(동적 점은 어차피 1프레임). 달라지는 건
      z-buffer 를 막는 정적 껍데기뿐.
    - **정적 subject** — NTP 는 subject 점도 프레임 f 것만 남겨 실루엣이 1/49 로 얇아진다.
      중심 추정이 흔들리고 `subject_zero_frames` 가 늘어 지표가 체계적으로 낮게 나온다.
      d121 val 은 정적 subject 가 다수(초반 327 entry 중 205)라 이 편향이 표를 지배한다.
    - 앞서 적은 "`subject_in_frame` 은 두 모드에서 거의 불변" 은 삭제했다 — 근거였던 12-entry
      프로브가 잘못된 코퍼스(아래 Fixed) 위에서 돈 것이라 무효다.
  · **`hole_fraction` 은 모드 간 비교 금지** (누적이 hole 을 메운다). 판독 실수를 막기 위해
    보고서에 `render_mode` 열을 남긴다 (`cloud` / `warp_1to1`).
  · 12-entry 프로브 실측: 211.03 s -> 65.97 s (3.2배).
  · `--limit N` 은 씬마다 앞에서 N entry 만 — 프로브용.
- **`eval_collision_rate.py` — 단위 호(arc) 정규화 충돌률 `--arc_floor_u`
  (2026-09-07, 사용자 지시 "부딪힌 frame 수를 unit arc로 나눠서 collision per unit arc의 평균…
  gendop는 조금 움직여서 많이 안부딪혀서 이 metric을 쓰는거야").** 기존 두 비율은 이동량을
  안 봐서 "거의 정지 → 부딪힐 기회 없음"을 안전으로 읽는다. `arc_u = Σ|Δp|/S` (충돌 판정과
  **같은 좌표**: world pose 가 `gt_cv[0] @ pred_rel` 이라 `|Δp|=|Δt_rel|`, kNN 도 그 `p` 를 쓴다)
  로 나눈 `<gate>_frames / arc_u` (단위 1/u) 를 `mean`(정지 궤적 제외) 과
  `pooled`(`Σframes/Σarc`) 두 가지로 낸다. 둘이 갈리면 짧은 궤적이 평균을 끌고 있다는 뜻.
  CSV 에 `arc_u` · `knn_frames` 열 추가, JSON `config.arc` 블록 추가. 기존 표는 그대로 두고
  `[per unit arc]` 두 번째 표를 덧붙인다.
  · **실측이 사용자 가설을 절반만 지지한다** — `gendop_*_p30_slerp` 의 `arc_u` median 0.199/0.203
    은 우리(0.258/0.257/0.274)보다 25% 작고 GT(0.359)보다 45% 작다. 그런데 정규화 후에도
    `knn/arc mean` 이 gendop_rgbd 0.0404 < d123_da3 0.2202 로 순위가 안 뒤집힌다.
    진짜 이상한 건 **산포**다: gendop p30_slerp 의 arc p10~p90 이 0.163~0.212 (1.3배)인데
    우리는 0.073~0.861 (12배), GT 는 0.021~1.494 (70배) — GenDoP 은 "조금 움직이는" 게 아니라
    프롬프트와 무관하게 **거의 일정한 길이**를 낸다. 이게 정규화가 순위를 못 바꾼 이유다.
  · **작은 표본 주의**: 충돌 프레임이 875×49=42,875 중 7~61개뿐이라 per-arc 평균이 엔트리
    1~12개에 얹혀 있다.
  · `--knn_r 0.012089` 로 임계를 고정해 재실행했더니 `gt` 의 kNN rate 만 0.0103 -> 0.0114 로
    움직였다 (출력에 찍힌 반올림값을 되먹인 탓, 나머지 8 arm 은 전부 동일). 인용할 때 주의.
- **`eval_collision_rate.py` — scene 별 재집계 `--per_scene` (기본 on, 2026-09-07, 사용자 지시
  "scene별로도 뽑아줘").** 코퍼스 평균은 엔트리 수가 많은 씬이 끌고 가고, 씬마다 점군 밀도와
  소스 시차가 달라 같은 `knn_r` 이 씬별로 다른 엄격도로 작동한다. arm 요약 로직을
  `summarize(pool)` 로 뽑아 전체와 씬별이 **같은 코드 경로**를 타게 하고 (CSV 재집계 스크립트를
  따로 두면 본 표와 갈린다), JSON 에 `by_scene` 블록 + 씬별 print 표를 추가했다. 기존 두 표와
  `summary` 블록은 그대로다.
  · d121 val 875 = avocado-slice 266 / bmx-bumps 220 / camel 325 / couple-hug 64.
  · 실측: 충돌이 **camel 과 avocado-slice 에만** 있다. `d124_da3_molmo2` 의 코퍼스 kNN rate
    0.0137 은 camel 0.0369 + 나머지 3 씬 0.0000 이고, `gendop_gdstyle_p30_slerp` 0.0046 도
    전부 camel(0.0123). couple-hug 는 9 arm 전부 0 이다 (arc median 0.085~1.03).
  · 씬별로 보면 `d123_da3` 는 avocado-slice(0.0188)·camel(0.0215) 양쪽에서 걸리고
    `d124_da3_molmo2` 는 avocado-slice 0 / camel 0.0369 로 한 씬에 몰린다 — 코퍼스 평균
    0.0137 이 같아도 실패 모양이 다르다.
- **`CinemaTraj/scripts/eval_collision_rate.py` — 궤적 충돌률 게이지 2종 (G1 + kNN k=10)
  (2026-09-07, 사용자 지시 "우리 G1 gate로 먼저 collision rate 재줘").** 예측 궤적이 씬 안으로
  파고드는 비율을 재는데, 게이지가 하나면 판정을 못 믿는다:
  · **G1** — `lbm/gates.behind_profile` 을 그대로 부른다 (재구현 아님). 카메라 중심을 소스
    49프레임에 재투영해 `z_cam > depth + 0.02·S` 인 프레임이 하나라도 있으면 충돌.
    `clear_frac=0` (뱅크 운영값 0.10 이 아니라) 이라 "가까이 갔다"가 아니라 "관측된 표면 뒤로
    갔다"를 뜻한다. 자유 임계가 없는 대신 **화면 밖으로 투영되는 카메라는 못 본다**.
  · **kNN** — 49프레임 non-sky 유효 depth 를 unproject 한 점군(stride 4, `|∇log z|>0.05` 경계
    픽셀 제거)에서 카메라 중심까지의 **10번째** 최근접 거리를 `S` 로 나눈 값. k=1 이면 떠도는
    경계 점 하나가 판정을 뒤집는다. 49프레임 중 **하나라도** `r` 미만이면 그 궤적은 충돌.
  · `r` 은 하드코딩하지 않는다 — 점군 밀도(=`--stride`)에 따라 변하므로 `--knn_calib_arm gt
    --knn_target_rate 0.01` 로 **GT 궤적의 최소 `d_10` 1% 분위**를 매 실행 계산한다.
  · eval 폴더 `*_transforms_{ref,pred}.json` 은 **DA3 world c2w / OpenGL / 미정규화**임을
    `recon_and_seg/<video>/cameras.npz` 대조로 확인했다 (frame0 일치). 스케일 재수화 없이
    `S` 로 나누기만 하면 된다.
  · 산출물 `<out>/{collision_rate.json,collision_per_entry.csv}`.
    vista d121 test 875 × 6 arm 결과는 `results/20260907_d159_collision/`.
- **`CinemaTraj/scripts/gendop_style_captions.py` + `run_gendop_eval.py --text_dir/--text_tag`
  — GenDoP 에 학습 분포 정합 문장을 먹이는 텍스트 arm (2026-09-07, 사용자 지시 "gendop가
  학습한 환경에 맞춰주기 위해 ... motion text와 target만 사용해서").** 지금까지 GenDoP 에 넣던
  문장은 우리 D121 자연어 캡션이라 `framing`(medium shot tightening to...) · `composition`
  (with the fence on the right...) 절이 붙어 있고 어휘도 우리 태거 것이다. GenDoP 는 DataDoP
  캡션(`move forward` / `yaw left`)으로 학습됐으므로 그건 **분포 밖**이고, 낮게 나온 수치가
  모델 탓인지 문장 탓인지 안 갈린다.
  · 문장은 두 조각으로만 만든다 — `da3/captions_gendop/<idx>_tag.json` 의 chunk 열
    (`move`/`angular`) + `prompts.json[<idx>].caption_fields.target_text`. **preset 이름으로
    손매핑하지 않는다** — 그 태그는 우리가 GenDoP 파이프라인 세팅으로 GT 포즈에서 직접
    뽑은 것이라 어휘가 이미 DataDoP 과 같고, `orbit_right` 의 실제 태그가 `move right` +
    `yaw left` 인지는 포즈가 답한다 (좌우 부호를 눈대중으로 정하지 않기 위함).
    전치사는 첫 chunk 의 이동 방향이 고른다: forward→`towards`, backward→`away from`,
    그 외→`focusing on`.
  · 출력이 eval 폴더와 **같은 모양**(`<out>/test/<prefix>_<scene>_<idx>_caption.json`, 키
    `Concise Interaction`)이라 `gendop_release_infer.py --text_from_eval_dir` 에 그대로 꽂힌다
    — infer 스크립트는 한 줄도 안 고쳤다.
  · `run_gendop_eval.py` 는 `--text_dir` 를 안 주면 예전과 **비트동일**하다 (ref arm 의 eval
    폴더). `--text_dir` 만 주고 `--text_tag` 를 빼면 기본 캡션 산출물을 덮어쓰므로 assert 로
    막는다. 경로는 `pred_gendop_rgbd_p49_gdstyle` 처럼 접미사로 갈린다.
  · vista d121 test 875 entry 생성 결과: `written 875 / missing_tag 0 / no_target 64 /
    degenerate 28 / words min 4 med 17 max 43`. `no_target 64` 는 targetless preset
    (D90 `aim=free` 의 pan/tilt 4종)이라 정상이고, 그중 28건은 회전까지 DataDoP
    `angular_static_threshold` 아래로 태깅돼 문장이 `The camera remains static.` 하나만
    남는다 — 숨기지 않고 `degenerate` 로 세서 `config.json` 과 요약표에 남긴다.

- **`CinemaTraj/scripts/run_gendop_eval.py` — GenDoP 릴리즈 ckpt 를 코퍼스 인자로 돌리는 드라이버
  (D158, 2026-09-07, 사용자 지시 "gendop text_rgbd에 대해서 돌려놔주고").** `tmp/d156/
  run_gendop_d121.py`(vista 전용)를 `CORPORA` 표로 인자화해 `scripts/` 로 승격한 것이다 —
  두 번째 코퍼스가 생긴 시점이라 승격 조건을 만족한다. 단계는 그대로 inputs → infer →
  evaldir → score 4개이고, **vista_d121 항목은 d156 에서 실제로 돈 값 그대로**라 같은 명령이
  같은 결과를 낸다. 새 코퍼스는 표에 항목 하나(`corpus`/`split_name`/`eval_data`/`prefix`/
  `cloud_root`/`out`/`depth_norm`/`ours`/`ref`)를 더하면 끝난다.
  · **`depth_norm` 을 dynpose 도 `median` 으로 잡았다.** 처음엔 `none` 으로 적었는데,
    `dynpose_gendop_inputs.py` 요약표의 `min` 행(0.19~6.16)을 median 으로 잘못 읽은 것이었다.
    실제 per-scene frame0 depth median 은 **0.41~20.55 (중앙값 2.68)** 로 GenDoP 학습
    대역(릴리즈 `text_rgbd/case1_depth.npy` med 0.323) 밖이다. `none` 이면 깊이만 8배 큰
    조건이 들어간다.
  · dynpose d137 val 1,129 entry / 27 scene, `--pose_length 49 --no_strict_pose_length`,
    텍스트는 우리 모델(`eval_my/20260906_013323_dynpose_d137_da3__last`)이 받은 문장 그대로.

- **`CinemaTraj/configs/bank/d157_dynpose.json` — dynpose 875편을 최신 vista 게이지로 재굽기
  (D157, 2026-09-07, 사용자 지시 "gpu 2개 이상 남으면 최신 vista 방식으로 dynpose도 돌려줘").**
  D149 는 러너 스크립트(`run_dynpose_d149_shard.sh`)만 있고 **실제로는 한 편도 안 돌았다**
  (`.graph_d149` 0/884). 그 사이 vista 가 d150S→d151 로 τ 분모를 `z_med_frame0` → `S` 로
  옮겼으므로 d149 를 그대로 돌리면 굽자마자 vista 와 게이지가 어긋난 코퍼스가 된다. 그래서
  d149 config 를 따로 두지 않고 처음부터 d151 게이지로 굽는다.

  **D149 에서 가져온 축** — ① detection 을 VLM 명사로 통일(`seg_instances` 877편 재검출
  완료), ② static 트리 추가(`seg_instances_static` 846편), ③ `--gravity_source geocalib`
  (사이드카 880/880), ④ `--num_external 0` (dd_* 제외). ①②③ 이 전부 그래프 입력이라
  `.graph_d122` 를 재사용할 수 없어 GRAPH/CLOUD 도 돈다 (`.graph_d157`/`.cloud_d157`).
  **D151 에서 가져온 축** — tau/fit 양쪽 `--tau_denom S`, fit `--tau_knob_min 0.005`.
  vista(d151) 와 다른 곳은 데이터 위치 + preset 라우팅(route 단계) + GRAPH/CLOUD 실행
  여부뿐이다. anchor 상한 3/3 은 route 에 준다 — 뱅크에 주면 no-op 이다
  (`sample_camera_bank.py:144-151` 이 route 의 `--nodes` 를 받으면 `pick_main_anchors` 전에
  return 한다).

  smoke 1편 실측: preset 14종(`--num_external 0` 의도대로), `hole_bank_d157` 129 변이
  (solved 70 / approach_limited 16 / elev_limited 12 / obb_limited 12 / shape_limited 8 /
  unreached 6 / static 5), `clamped_low` 0. 그래프에 `stat_0 table` 이 잡혀 **dynpose 에서
  static 노드가 처음으로 생겼다** — `--max_static_anchors 3` 이 처음 일을 한다.

- **`CinemaTraj/scripts/eval_dir_to_gendop_npz.py` — latentcam eval 폴더를 GenDoP 캡션
  harness 입력으로 (D156, 2026-09-07, 사용자 질문 "caption 지표는 어떻게 재야 공정하게
  잴 수 있어?").** `gendop_release_eval.py` 는 npz 디렉토리(`latentcam__<scene>__<name>.npz`
  의 `c2w`) 위에서 재는데 우리 arm 은 eval JSON(`<run>/test/*_transforms_pred.json`) 만
  남긴다. 그래서 지금까지 우리 arm 과 GenDoP arm 이 **서로 다른 harness** 로 측정됐다.
  이 스크립트가 JSON → npz 로 옮겨 네 arm 을 한 harness 에 올린다. `transform_matrix` 는
  OpenGL c2w 이고 `gendop_release_eval.opencv_like_identity` 가 OpenGL 을 그대로 분절기에
  넣으므로 **변환 없이** 옮긴다 (GenDoP npz 와 같은 게이지).

  `--which ref` 로 GT 를 뽑으면 **harness 자기검증**이 된다 — d121 test 875 entry 실측
  `pooled F 0.9673 / frame match 0.9693`. 1.0 이 아닌 3% 는 GT 태그가 th 0.05 로 구워진
  반면 이 harness 는 DataDoP 컨벤션(static 0.02 / diff 0.4)으로 재태깅하기 때문이고,
  이 값이 **이 게이지의 천장**이다. 천장 없이 arm 수치만 보면 안 된다.

- **`gendop_release_infer.py` 에 native 생성 길이 손잡이 `--pose_length` /
  `--strict_pose_length` (D156, 2026-09-07, 사용자 지시 "직접 49프레임 추론하도록해줘").**
  기존에는 30 포즈를 뽑고 하류(`gendop_preds_to_eval_dir.py --src_poses 30 --n_poses 49`,
  `eval_batch.py:462 pose_normalize(..., 49)`)에서 49 로 **리샘플**했다. 생성 길이 자체는
  `core/options.py:39 pose_length` 가 정하고 그게 `core/models.py:331
  max_new_tokens = 10*pose_length+1` 과 `:333 num_tokens` 로 들어가며,
  `prefix_allowed_tokens_fn` 이 EOS 를 `1+10N` 위치에서만 허용하므로 이 값이 곧 생성 길이
  상한이다. `--pose_length 49` 면 모델이 49 스텝 궤적을 **직접** 만든다.

  상한일 뿐이라 모델이 30 에서 EOS 를 낼 수 있다. 그때 예전 `decode_tokens` 는 전량을
  static 폴백으로 버렸는데(`degenerate`), `--no_strict_pose_length` 를 주면 **10의 배수만큼
  살려서 디코드**하고 실제 길이를 `n_poses` 에 적는다. 안 그러면 "49 를 요구했더니 전부 정지"가
  된다. npz 에 `n_poses` 필드, config.json 에 `pose_length`/`strict_pose_length`/`n_poses_hist`
  추가.

  **기본값은 `--pose_length 30 --strict_pose_length`** 라 인자 없이 부르면 예전과
  bit-identical 이다 (non-degenerate 일 때 `token[:usable] == token[:-1]`).

  d121 test 앞 8 entry 실측: `n_poses hist {49: 8}`, `degenerate 0`.
  **다만 8 entry 로는 부족했다** — 875 전량에서는 166 번째(`avocado-slice/165`)가 EOS 를 내고
  GenDoP 내부 assert 로 죽었다. 아래 `--forbid_eos` 항목 참조.

- **`gendop_release_infer.py` 에 `--forbid_eos` / `--no_forbid_eos` (D156, 2026-09-07).**
  `--pose_length 49` 전량 실행이 166/875 에서 `core/models.py:359
  assert np.all(tokens >= 0)` 로 죽었다. `:324-329 prefix_allowed_tokens_fn` 이 `1+10N`
  위치마다 `eos_token_id=2` 를 후보에 넣고 `:358` 이 `output_ids - 3` 을 하므로 **EOS 가
  나온 자리가 -1** 이 된다. 학습 길이(30)에서는 `max_new_tokens` 에 항상 먼저 걸려 EOS 가
  안 나오지만, 49 로 넘겨 요구하면 일부 entry 가 실제로 EOS 를 낸다.
  `--no_strict_pose_length` 는 `model.generate()` **리턴 뒤**의 가드라 이 assert 를 못 막는다.

  `core.utils.monkey_patch_transformers()` 가 이미 갈아끼운
  `PrefixConstrainedLogitsProcessor.__call__` 위에 한 겹 더 씌워 EOS 열을 -inf 로 만든다
  (`forbid_eos_in_logits()`). **GenDoP 리포는 0 줄 수정.** 후보 `range(3, 260)` 257 개는
  그대로라 "빈 후보" ValueError 는 안 난다. config.json 에 `forbid_eos` 기록.

  기본값은 auto — `pose_length == 30` 이면 off 라 예전 경로 무변경 (실측: 같은 3 entry 를
  30 으로 재실행하면 c2w 완전 일치). 49 면 on. 죽던 3 entry 가 `n_poses hist {49: 3}`,
  `degenerate 0` 으로 통과.

- **`gendop_release_eval.py` 에 게이지 통일 옵션 `--n_poses` / `--resample` / `--fps` /
  `--align_len` / `--out_name` (D156, 2026-09-07, 사용자 지시 "우리 모델이 학습한걸
  기준으로 하고싶은데 49프레임 10fps로 통일해줘").**
  기존 경로는 30 native pose 를 `fps 7.5` 로 분절한 뒤 **라벨만** 49 로 늘렸다. 그러면
  `smoothing_window_size=18` 이 pred 궤적의 60%(18/30) 를, GT 궤적의 37%(18/49) 를 덮어
  두 쪽의 평활 정도가 다르다. `--n_poses 49 --fps 10` 은 **분절 전에** pose 를 49 로
  리샘플해 양쪽을 37% 로 맞춘다.

  **기본값은 예전 그대로** (`--n_poses 0` = 리샘플 없음, `--fps 7.5`) 라 인자 없이 부르면
  bit-identical 이다.

  **`--resample` 은 `slerp` 가 기본이고 `index` 는 쓰지 말 것** — d121 test 875 entry
  (`pred_gendop_text`) 실측:

  | 게이지 | P | R | F | frame match |
  |---|---|---|---|---|
  | native 30 / fps 7.5 (기존 기본값) | 0.2529 | 0.1123 | 0.1279 | 0.1123 |
  | 49 **index-pick** / fps 10 | 0.2480 | **0.0633** | **0.0526** | 0.0633 |
  | 49 **slerp** / fps 10 (채택) | 0.2831 | 0.1049 | 0.1230 | 0.1049 |

  index-pick 30→49 는 48 step 중 19개가 중복 프레임이 되어 속도 0 이 되고,
  `segmentation.smooth_segments` 가 window 19 안의 **최빈값**을 고르므로 그 40% 짜리
  static 표가 mode 를 가져간다 — 궤적이 아니라 리샘플러가 라벨을 만든다. recall 이
  0.112→0.063 으로 반토막나는 게 그 신호다. slerp 는 native30 대비 F 0.128→0.123 으로
  거의 안 움직이므로 게이지 교체가 점수를 옮기지 않는다는 확인도 된다.

  같이: d121 코퍼스 test 875 entry 에 GT `_tag.json` 을 **49 native pose / fps 10** 으로
  구웠다 (`caption_cameras_datadop.py --sets latentcam --no_llm --num_poses 0 --fps 10`,
  출력은 `<scene>/da3/captions_gendop/`). 코드 변경 없이 인자만으로 되는 경로다.

- **`CinemaTraj` framing 게이트를 절 종류별 임계로 — `--framing_exit_min_in_frame`
  (D154, 2026-09-06, 사용자 지시 "텍스트가 실제 framing 과 다르게 만들어졌던 부분").**
  `build_bank_captions.py` 의 `framing_dropped()` 가 `framing_parts()` 의 `way` 를 읽어
  `exit` 절("... until the subject leaves the frame")에만 별도 임계를 적용한다. 기본
  0.05 (`--framing_min_in_frame 0.85` 는 그대로).

  **왜 필요한가**: D143 의 단일 임계는 `exit` 절을 같이 지웠다. 그 절은 대상이 나간다고
  **이미 말하고 있으므로** `subject_in_frame` 이 낮은 게 거짓이 아니라 정확한 서술이다.
  vista d128 train 9,389 entry 실측 — framing 약속 8,395건을 절 종류로 가르면:

  | 절 종류 | n | in_frame<0.85 | =0 |
  |---|---|---|---|
  | `plain` "keeping it in a medium shot" | 3,182 | 220 (6.9%) | 84 |
  | `change` "... tightening to a close-up" | 4,178 | 597 (14.3%) | 71 |
  | `exit` "... until the subject leaves the frame" | 1,035 | **1,026 (99.1%)** | 105 |

  0.85 일괄이면 1,843건이 빠지는데 그중 1,026건(56%)이 정직했던 `exit` 이다. 실제로 못
  지키는 약속은 `plain`+`change` 의 817건뿐이다. `exit` 은 in_frame≈0 일 때만 뺀다 —
  그때는 "medium shot 으로 담고 있다가"라는 **앞부분**이 거짓이라서다.
  `--no_framing_on_free` 는 3,714건(39.6%)을 지우는데 0.85 적용 후 잔여 위반이 129건뿐이라
  비용 대비 얻는 게 없어 기각했다 (기본 꺼짐 유지).

  **검증** (d128 뱅크 52편 17,836 변이): `--framing_exit_min_in_frame 0.85` 로 부르면
  `captions_d143.json` 과 **비트 동일 (다른 것 0건)**. 기본값(0.05)에서는 D143 이 지운
  2,760건 중 1,363건(전부 `exit`)이 되살아나고, D143 이 안 지운 걸 새로 지우는 건 0건이다.
  결과 캡션에 남은 framing 약속 13,215건 중 **비-exit 이면서 in_frame<0.85 는 0건**,
  `exit` 이면서 in_frame<0.05 도 0건.

  **적용 범위** (사용자 확정): 굽는 중인 **D151 뱅크에만** 적용한다. 도는 arm 3개
  (D131 / D152 두 arm)와 CLaTr 코퍼스가 쓰는 `latentcam_da3_k6_d128` 은 건드리지 않는다 —
  그 코퍼스는 9/5 09:19 에 D143 **이전** `captions.json` 에서 export 됐고 (D143 산출물
  `captions_d143.json` 은 9/6 11:00 로 그 뒤다), 지금 갈아엎으면 학습 중인 3개가 깨진다.

- **`CinemaTraj` τ 손잡이 하한 `--tau_knob_min` + `configs/bank/d151.json` (D151/F5, 2026-09-06).**
  `fit_hole_ladder.py` 의 `KNOB_RANGE["tau"]` 하한 0.02 를 CLI 로 뺐다. **기본값이 그대로 0.02
  라 인자를 안 주면 예전 뱅크와 비트 동일**이다 (camera-lens 32행 대조: 결정열 불일치 0,
  `poses.npz` md5 동일). `main()` 이 `KNOB_RANGE` 를 **한 번** 갈아끼우므로 탐색 경계
  (`solve_knob`) · `lo_override` · `knob_floor` 열 · `bank.json` 의 `knob_range`/`tau_floor_rule`
  이 전부 같은 값을 읽는다. 기본값은 갈리기 전 값을 `TAU_KNOB_MIN_DEFAULT` 로 따로 잡아둬,
  한 프로세스에서 파서를 두 번 만들어도 기본값이 앞 실행 인자로 흘러가지 않는다.
  **왜 필요한가**: D150 에서 τ 분모를 `S` 로 바꾸자 `S/z_med` 가 큰 씬에서 `clamped_low` 가
  대량 발생했다 (martian-flag 93/258, camera-lens 50/172, parkour 94/320, snow-dog 28/258).
  그 행들은 기하학적으로 불가능한 게 아니라 손잡이 하한 0.02 에서 이분법이 멈춘 것이다.
  **채택값 0.005 의 근거** — 하한 스윕 7값 × 6씬 × 32행(`tmp/d151`)에서 "물리 위반
  (`behind_frames>0 ∨ obb_slack<0 ∨ below_ground_frames>0`) 이 0 이 되는 가장 큰 하한":
  martian-flag 0.005 / camera-lens 0.01 / snow-dog 0.01 / parkour 0.01 / hike 0.02(이미 0).
  ① 0.005 아래로는 아무것도 새로 안 풀리고 `motion_preset_no_motion` suspect 만 는다
  (snow-dog 3→6@0.005→8@0.002) — 정지 궤적을 아닌 척 내보내는 쪽으로 넘어간다.
  ② 임계 0.005/0.01/0.01/0.01 이 `S/z_med` 29.85/12.68/9.26/5.69 와 무상관이라 **씬 스케일
  함수가 아니라 상수** (`0.02·z_med/S` 안 기각).
  ③ 안전성 — "위반 행이 export 가능해진" 셀이 7×6 전부에서 0/N 이라, 하한을 내려도
  `DROP=clamped_low` 필터를 나쁜 행이 통과하지 않는다.
  woman-phone 은 48/48 이 어떤 하한에서도 안 풀리는데, 이건 τ 문제가 아니라
  `src_ground_clear = −0.0082` — **소스 카메라 자체가 추정 지면 아래**라 지면 게이트 기준선이
  음수인 별개 결함이다 (D149 880편에서 유병률 측정 예정).
  `configs/bank/d151.json` 은 d150S 와 `--tau_knob_min 0.005` 한 줄만 다르고 나머지 축은 문자
  단위로 같다 (`--tau_denom S`, `.graph_s115`/`.cloud_s115` 재사용, GRAPH/CLOUD 는 안 돈다).

- **`CinemaTraj` τ 분모 선택 `--tau_denom` (D150, 2026-09-06).** 사용자 지시: "z_med 는 이전에
  전체 프레임에서 sky 제외 유효한 depth 의 첫 카메라로부터의 거리의 평균으로 하기로 했잖아.
  적용해줘." `scene_graph/scale.py` 에 `TAU_DENOM_MODES` + `tau_denominator(graph, mode)` 를
  두어 **그래프에서 τ 분모를 꺼내는 창구를 하나로** 만들고, `decode/build_poses.py` 의
  `z_med = float(graph["scale"]["z_med_frame0"])` 한 줄을 그 창구로 바꿨다 (τ 가 실제로
  계산되는 유일한 지점). `sample_camera_bank.py` / `fit_hole_ladder.py` 에 `--tau_denom`,
  `SHAPE_DEFAULTS["tau_denom"]`, `fixed.tau_denom` 을 추가했고 `bank.json` 의 `z_med` 가 선택된
  분모를 싣는다. `emit_bank.py` 는 `fixed.tau_denom` 을 따르고, 잘린 행의 τ 재계산도 그래프가
  아니라 `extra["info"]["z_med"]` 에서 읽는다 — 그래프를 다시 읽으면 S 로 구운 뱅크의 일부
  행만 옛 게이지로 되돌아가는데 pose diff assert 로는 안 잡힌다.
  **기본값은 `z_med_frame0`(옛 정의)** 이다. τ 분모는 뱅크 정체성이라, 기본값을 뒤집으면
  진행 중인 D149 굽기가 코퍼스 중간에 정의를 갈아탄다. `S` 채택 여부는 D150 파일럿
  (`S/z_med_frame0` 상위 vista 6편, `hole_bank_k6_d150S`) 으로 정한다.
  실측 배율 `S/z_med_frame0` — vista 53편 p50 1.426 / p90 3.940 / max 29.854,
  dynpose 280편 p50 1.171 / p90 2.404, trumans 191편 p50 1.106 / p90 1.280.

- **`CinemaTraj/scripts/run_bank.py` — 뱅크 체인 단일 python 드라이버 (2026-09-06).**
  `scripts/run_*_dNN_shard.sh` 가 23개까지 늘어난 것을 끝낸다. `graph→cloud→route→tau→fit→emit`
  실행 로직은 이 파일 하나에 두고, 세대 차이는 `configs/bank/<gen>.json` 에 **뱅크 정체성을
  이루는 플래그 전량**으로 적는다 (`--video`/`--output_root`/`--bank_dir` 은 드라이버가 붙인다).
  config 의 `"extends"` 는 한 축만 다른 대조 세대를 위한 얕은 병합이고, 리스트를 반쯤 물려받지
  않도록 최상위 키 단위로만 덮는다. 샤딩(`--num_shards/--shard_id`)·`--skip_done`·단계 선택
  (`--stages`)·GPU 0~4 assert 포함. 기존 bash 러너는 **지우지 않는다** — 이미 구워진 세대를
  재현할 때의 근거 기록이다. 첫 사용처는 `configs/bank/d150S.json` / `d150L.json`.

### Changed
- **D171 — `fit_hole_ladder.py --min_subject_visible` 기본값 `0.0`(측정만) → **`0.6`(판정)**
  + 옛 9 세대 config 를 `"--min_subject_visible", "0"` 으로 고정 + `configs/bank/
  d171_dynpose_occlusion.json` 신설 (2026-09-10, 사용자 지시 "어차피 hole 볼 때 랜더링하니까
  subject_visible_frac를 판정으로 올려줘").** D112 에서 열로만 재던 가림을 이분법 게이트로
  올린다. `--min_subject_visible 0` 이면 예전 동작으로 정확히 되돌아가고, 재현이 깨지지 않게
  d150L/d150L2/d150S/d151/d157_dynpose/d166/d168/d169/d170 **9개 config 에 `0` 을 박았다**
  (각 파일 3줄 변경).
  · 임계 0.6 의 근거는 d169 dynpose 21편 164행 실측 분위수다 — p05 0.508 / p10 0.562 /
  p25 0.827 / p50 0.971, 임계별 태그 수 0.30 → 7행(4.3%) / 0.50 → 8행(4.9%) /
  **0.60 → 20행(12.2%)** / 0.70 → 30행(18.3%) / 0.80 → 36행(22.0%). 0.5 이하는 분포 바닥만
  긁어 `027514bb orbit_right`(`subject_in_frame 1.0` 인데 `frac 0.542 / min 0.007` 로 벽에
  가림)를 놓치고, 0.7 이상은 usable 60행 중 8행(13%)의 손잡이를 깎기 시작한다(0.6 은 2행).
  · **행을 버리지 않는다** — 이분법이 가시비율이 임계를 넘을 때까지 손잡이를 줄이고, 하한에서도
  못 넘기면 `status=clamped_low` + `binding=occlusion` 이 되어 기존 `--retry_status
  clamped_low` 가 다음 층 preset 으로 넘긴다.
  · d170 vs d171 4편 24행 diff 실측 — **바뀐 건 2행뿐**이고 나머지 22행은 손잡이까지 동일하다.
  `019bbbc2 stat_0__orbit_right__hole0.2` knob 3.000→0.502 / seen 0.5055→0.6116 /
  status `shape_limited`→`occlusion_limited`, `02044b66 dyn_0__track_pull_out_arc_right__hole0.2`
  knob 1.375→0.192 / seen **0.0071→0.9853** / binding `hole`→`obb`. `hole_static`·`target_hole`
  은 불변이라 게이트가 예산이 아니라 손잡이만 건드린 게 확인된다. 렌더 비용은
  284.1 s → 325.9 s (**+14.7%**) — 코드 주석의 "2배"가 아니다.
- **정리 — `/tmp` 기본 출력 경로 13곳을 `<repo>/tmp/` 로, `time_vista_stages.py --cuda` 기본값을
  `6` → `0` 으로 (2026-09-09, 사용자 지시 "우리 돌리는 파이프라인 영향 안가는 선에서 쭉
  진행해줘").** 둘 다 CLAUDE.md 규칙 위반이 argparse **기본값**에 박혀 있던 경우다 —
  플래그를 안 주고 돌리면 조용히 규칙을 어긴다.
  · `/tmp` 8 py (`audit_scene_scale.py:145` `audit_tau_axes.py:238` `lbm_preview_reel.py:116`
  `probe_trumans_blend.py:100` `time_vista_stages.py:84,87` `trumans_blend_layout_worker.py:154`
  `trumans_clip.py:187` `vista_blend_check.py:160`) + 5 sh (`probe_dynpose_scale_mode.sh:27`
  `run_dynpose_d107_shard.sh:39` `run_dynpose_d110_export.sh:21` `run_dynpose_d110_shard.sh:23`
  `run_preset_warp_max_shard.sh:88`). 규칙은 "`/tmp` 는 **쓰지 않는다** — 시스템이 임의로 비울 수
  있어 scene 목록·로그가 조용히 사라진다". `run_dynpose_d110_export.sh` 의 `TESTV` 는 **읽는**
  경로라 이미 사라졌을 자리였다. 파일명은 그대로 두고 상위만 옮겼으므로(`tmp/` 는 이미 존재)
  makedirs 동작이 안 바뀐다.
  · `time_vista_stages.py --cuda` 기본 `"6"` 은 "GPU 5~7 사용 금지 (2026-09-06 사용자 지시)"
  정면 위반. 같은 sweep 을 `*.py`/`*.sh` 전체에 돌려 남은 5~7 기본값은 **0건** 확인.
- **`.gitignore` — CinemaTraj 플래너 **코드**를 git 추적 대상으로 되돌린다 (2026-09-09, 사용자
  질문 "git이 추적하게 못하나").** `camera_generation/models` 는 vendored 모델 트리 9종(CCD,
  CamVLA, DIRECTOR, Director3D, GenDoP, I3DM, LAMP, Planner, SCVideo) 때문에 통째로 무시되고
  있었는데, 그 안에 우리가 직접 쓴 `Planner/CinemaTraj/` 가 같이 묻혀 있었다 —
  `git ls-files scripts/` 가 **0건**이라 스크립트를 지우면 되돌릴 방법이 없는 상태였다
  (scripts 127 py 37,990줄 + 35 sh 3,005줄). git 은 부모 디렉토리가 제외되면 자식을 negate 로
  되살릴 수 없으므로 `camera_generation/models` → `camera_generation/models/*` 로 바꾸고
  `Planner/` → `CinemaTraj/` → `{scripts,lbm,scene_graph,decode,configs}/` + 루트 `*.py`/`*.md`
  만 한 단계씩 열었다. 무시 범위 자체는 이전과 같다 — 다른 8개 모델 트리는 그대로 닫혀 있고,
  산출물(`out*/` 618+280+142 GB, `logs/`, `results/` 12 GB, `__pycache__`)은
  `CinemaTraj/*` 줄에서 계속 막힌다. 실측 편입량 **211 파일 / 3.28 MB**
  (py 154 · sh 35 · json 11 · md 10 · txt 1). 유일한 명시적 제외는 `configs/datadop_shapes.json`
  (4 MB 생성 데이터, `retrieve_datadop_shapes.py` 가 재생성).
  **왜 지금**: 스크립트 정리(사체 4편 + 참조 1회 30편 심사)를 하려면 `rm` 이 되돌릴 수 있어야
  한다. 추적이 없으면 attic 디렉토리로 옮기는 우회가 필요했는데, 이제 그냥 지우고
  `git checkout` 으로 되돌릴 수 있다.
- **D169 — `describe_instances_vlm.py --num_frames` 기본 3 → 6 (2026-09-09, 사용자 지시
  "일단 6으로 설정 올려줘").** 기존 산출물 266편 421 노드 실측: `action` 이 빈 노드는 9.0%
  뿐이라 3장으로도 행위 자체는 나온다. 문제는 **무엇이** 나오느냐다 — 상위 동사가 `being` 77
  (18%, `being moved` 류 무정보 수동태) · `walking` 49 · `holding` 37 · `moving` 31 ·
  `standing` 15 로, 진짜 이동과 자세/상태와 수동태가 섞인다. 방향·속도·궤적은 3장으로 못 잡는다.
  더해서 `pick_frames` 가 면적 0 인 시간 구간을 건너뛰기 때문에 **32/421 은 3장도 못 받았다**
  (0장 7 / 1장 12 / 2장 13). 6장이면 그 구간 손실이 줄어든다. 비용은 `build_image` 가 만드는
  시트 폭이 2배(1344→2688 px)인 것뿐 — **호출 수는 노드당 1회로 그대로**다 (패널을 한 장으로
  합치므로). 참고 대조군: DynamicVerse stage1 은 25장을 개별 이미지로 보내되
  (`batch_process_qwen_pipeline.py:178`) 그 프레임은 고움직임 구간에서 뽑은 키프레임이고,
  프롬프트의 `reasoning` 요구사항은 motion 이 아니라 외양이다
  (`stage1_qwen.py:400` `"[key appearance features] + [held/carried objects if any]"`).
  이미 구운 `instance_desc.json` 은 헤더에 `num_frames` 를 적어 두므로 세대 구분이 된다.
- **`route_presets.py --orbit_fallback drop` (기본) — `s_curve` 를 라우팅 어휘에서 뺀다
  (2026-09-08, 사용자 지시 "그리고 preset에서 s_curve는 제거해줘").** `s_curve` 가 코퍼스에
  들어가는 **유일한 입구**는 orbit 슬롯의 좁은-span 대체였다 (`obs_az_span <
  ORBIT_MIN_SPAN_DEG=120°`). `drop` 이면 그 자리에 슬롯을 아예 안 만든다 — 다른 preset 으로
  메우지 않는 이유는 orbit 슬롯의 뜻이 "곡선 선회"인데 좁은 span 에서 그걸 하는 preset 이
  s_curve 말고 없고, `arc`/`recede` 를 억지로 넣으면 이미 그 슬롯을 쓰는 쌍과 겹쳐 2x2 의
  motion 축이 죽기 때문이다. 빠진 몫은 D166 backfill 이 같은 anchor 의 남은 슬롯에서 채운다.
  · `lbm/presets.py` 의 `s_curve` builder 는 **안 지운다** — 이미 나간 뱅크와 `decision.json`
    이 그 이름을 참조하고, 지우면 재현·캡션 조회가 통째로 깨진다. 라우팅만 막는다.
  · 함수 기본값은 `"s_curve"`(옛 동작), CLI 기본값이 `"drop"`. d157/d166 재현은
    `--orbit_fallback s_curve`.
  · dynpose 21편 실측: 40 anchor 중 **29개(72.5%)** 가 `orbit_ok=False` 라 이 갈래를 탄다.
    슬롯 80 → 78, backfill 풀 240 → 213, `s_curve` 슬롯 13·풀 16 → **0·0**.
  · **부작용(측정치, 미조정)**: `grid_slot_pair` 가 회전 후 "둘 다 있는 첫 쌍"을 집으므로
    orbit 이 든 3쌍이 빠지면 그 다음 쌍으로 밀린다 — `(recede,orbit)→(advance,vertical)`,
    `(orbit,static)→(recede,vertical)`, `(advance,orbit)→(recede,vertical)`. 21편 중 6편의
    슬롯 쌍이 바뀌고 전부 `vertical` 쪽으로 갔다: crane 계열 11 → 23 (슬롯의 29.5%),
    `static_look_at` 5 → 2. `GRID_SLOT_PAIRS` 를 다시 고르지 않는 한 이 쏠림은 남는다.
- **`build_bank_captions.py --no_nl_framing` — framing / shot scale 을 **구조체에만 남기고
  concise 문장에서 뺀다** (2026-09-08, 사용자 지시 "framing, shot scale은 일단 구조체로만
  남겨두고 concise에서는 빼둘 수 있어?").** D143 `framing_dropped` 와 **층이 다르다**: D143 은
  "그 변이는 framing 을 약속할 자격이 없다" 라 `caption["framing"]` 까지 비우는데(JSON 과 문장이
  같은 말을 해야 한다, D105), 이쪽은 **문장에서만** 뺀다. 두 손잡이는 겹쳐 쓸 수 있다.
  · `composition` 절도 같이 빠진다 — `nl_prompt` 에서 그 절은 framing 뒤에만 붙을 자리가 있다
    (`f"{head}, {link} {framing}" + comp`).
  · **부작용과 그 처방**: motion 문구에 `{target}` 슬롯이 없는 preset(`dolly_out` 계열)은
    framing 절이 대상을 부르는 유일한 자리다. d157 `016a6379` 129행 실측으로 문장이 대상을
    부르는 행이 117 → 105 로 12개 줄었고 12개 전부 `dolly_out` 이었다. `dolly_out` 은 grid2x2
    `recede` 슬롯 기본값이라 scene 당 5개 중 1개가 대상 없는 문장이 된다. 그래서 shot scale 도
    composition 도 없는 **최소 절**을 되살린다 (`target_view` → "keeping X in view").
    D143 이 뺀 변이에는 안 붙인다 — 거기선 프레임 유지가 실제로 거짓이다.
  · 재실측(같은 129행): `caption["framing"]` 동일 **129/129**, `prompt` 변경 109/129,
    문장이 대상을 부르는 행 **117 → 117**(최소 절 12개로 회복).
  · 기본값은 `--nl_framing`(켬)이라 기존 캡션 코퍼스는 비트 동일.
- **`eval_collision_rate.py` CORPORA 의 `gendop_*_p30_slerp` arm 2개를 `_noscale` eval 폴더로
  재지정** — 위 `--no_scale_token` 항목의 판독 경로. 되나누면 궤적이 median 1.99배 커지므로
  절대 크기에 반응하는 충돌률이 그만큼 달라진다.
- **GenDoP 판독값을 raw output 으로 (`run_gendop_eval.py --raw`, `eval_collision_rate.py`
  CORPORA; 2026-09-07 사용자 지시 "앞으로 raw output 그대로 써주고 metric들도 이에 맞춰서
  다시 표 만들어줘").** `gendop_preds_to_eval_dir.py` 는 기본으로 GenDoP 의
  `rmax = max_f |rel[f,:3,3]|` 를 **GT 것으로 바꿔 심는다** — 그러면 최대 변위가 GT 와
  정확히 같아지고, 충돌률·`subject_in_frame` 처럼 **절대 크기에 반응하는 지표는 GenDoP 이
  아니라 GT 를 재게 된다**. d121 p49 기준 실측 확대율(GT/GenDoP) median 은
  text 2.6929 / rgbd 2.4827 / gdstyle 2.4551 로, 렌더된 GenDoP 궤적은 모델이 실제로 낸 것보다
  ~2.5배 컸다.
  · `run_gendop_eval.py` 에 `eval_suffix(pose_length, text_tag, rescale)` 를 넣어 `--raw` 면
    산출물이 `eval_dir_..._raw` / `subject_in_frame..._raw.json` 으로 갈린다. infer 산출물
    (`pred_*`)은 rescale 과 무관하므로 **추론을 다시 돌리지 않는다**.
  · 기본값은 `--rescale` 로 두어 **예전 런과 비트동일**하다 (집 규칙: 기존 작동 구조는
    option 분기로 유지). rescale 판본은 확대가 지표를 얼마나 움직였는지 보는 대조군으로 남긴다.
  · `eval_collision_rate.py` 의 `vista_d121` gendop arm 3개를 `_raw` 폴더로 돌리고
    `gendop_gdstyle_p49` arm 을 추가했다. 결과는 `results/20260907_d159_collision_raw/`.

### Fixed
- **D166 파일럿이 scene 당 5개가 아니라 3~4개를 냈다 — 서로 독립인 버그 2개
  (`route_presets.py --track_min_drift_u` / `--min_anchor_sep`, `sample_camera_bank.py`
  `route_cut`) (2026-09-08).** 20편 파일럿 실측으로 잡았다.
  · **① route 와 tau 의 `track_` 판정이 서로 달랐다.** `route()` 는 `moving` 불리언만 보고
    `track_` 접두사를 붙이는데 `sample_camera_bank` 는 `center_drift_u > --track_min_drift_u`
    (0.05) 를 요구해서, `moving=True` 인데 변위가 작은 anchor 의 `track_*` 슬롯이 뱅크에서
    **조용히 사라졌다** (`018ccdd9` 실측 `man` 0.0212 / `hat` 0.0147). `--slot_plan full` 에선
    슬롯이 많아 티가 안 났지만 grid2x2 는 2칸 중 1칸이 통째로 날아가 5개가 3개가 된다.
    → `route()` 에 같은 문턱을 넘긴다 (함수 기본값 0.0 = 옛 동작, CLI 기본값 0.05 = 뱅크와 동일).
  · **② 2등 anchor 가 1등의 부속물이었다.** 면적순으로 고르니 part/whole 쌍이 나온다. 큰 쪽
    OBB 의 **회전된 로컬 축**에서 잰 중심 거리(반-extent 단위) 실측: `person/hands` 1.08,
    `man/sunglasses` 1.07, `man/hat` 1.53, `man/bowl` 1.91 (10쌍 중 4 = 40%) vs 진짜 다른 물체
    `dog/person` 8.44, `hand/person` 11.14, `person/backsplash` 6.77 … 2.0 에 빈 띠가 있다.
    → `--min_anchor_sep 2.0`(`obb_center_sep`). 전부 붙어 있으면(단일 물체 클로즈업) 1등은
    남긴다 — anchor 0 은 씬 전체를 버린다. 절삭은 필터 **뒤에** 한다 (부속물이 슬롯을 차지한 뒤
    잘리면 소용없다).
  · **곁가지**: `--min_anchor_sep` 만 넣으면 `sample_camera_bank` 가 `pick_main_anchors` 를 따로
    불러 그 노드를 **`presets` 합집합으로 되살린다** (D127 이 경고한 "라우팅 표에 없는 anchor 가
    뱅크엔 있는" 상태). `--preset_route` 를 주면 그 JSON 이 anchor 목록의 정본이고, 빠진 노드는
    `route_cut` 으로 찍고 버린다.
  · 재라우팅 11/11 편이 정확히 n=5, `man/hat`·`man/sunglasses` 쌍은 `stat_0` 으로 교체됐다.
  · **주의**: 위 두 값이 이제 CLI **기본값**이라(D143/D147 관례) d157 스타일 config 를 오늘
    다시 돌리면 d157 과 비트 동일하지 않다. 함수 기본값은 옛 동작을 유지한다.
- **`eval_subject_in_frame.py` 가 재굽기된 **다른 세대**의 코퍼스를 읽어 subject 를 전부 엉뚱한
  물체로 잡고 있었다 — 이전 `subject_in_frame` 수치 전량 무효 (`FIX.log` 2026-09-07).**
  `--corpus_root` 기본값이 `latentcam_da3` 였는데 그 코퍼스는 재굽기되어 씬별 변이가
  191/97/178/34 = **500** 으로 줄었고, d121 eval 폴더는 266/220/325/64 = **875** 세대다.
  entry 인덱스는 코퍼스 키가 아니지만 `variant_id` 형식은 세대가 달라도 같아서
  `entry_subject` 가 `stat_4` 같은 **그럴듯한 노드 id** 를 계속 돌려준다 — 191 에서
  `KeyError` 로 죽기 전까지 190 entry 를 조용히 틀리게 재고 있었다.
  · 기본값을 `latentcam_da3_k6_d121`(eval `config.yaml: dl3dv_root`) 로 고정.
  · `assert_corpus_matches()` 추가 — 코퍼스 `prompts.json` 의
    `prompt_camera_with_scene_video.concise` 와 eval `test/<name>_caption.json` 의
    `Concise Interaction` 을 대조해 불일치면 즉사. 씬마다 첫·중간·끝 **3군데**를 찌른다
    (entry 0 은 언제나 "첫 anchor 의 첫 preset" 이라 재굽기 후에도 우연히 맞는다).
  · `run_gendop_eval.py stage_score` 는 이미 올바른 `--corpus_root` 를 넘기고 있었다.
- **GenDoP `--pose_length 49` 산출물은 pose 30 부터 발산한다 — p49 arm 전량 판독 불가
  (2026-09-07, 사용자 지적 "gendop는 중간에 왜 갑자기 꺾여?").** depth-warp 영상에서 보이던
  꺾임의 원인이다. **좌표 규약 문제가 아니다** — `GL2CV` 를 적용한 쪽의 cos(GT,pred) 가 875 entry
  평균 **+0.6337**, 미적용은 **−0.0222** 로 현행 변환이 맞다.
  · 실측: 프레임 스텝 `|dt|` 가 앞 29 스텝 median 의 10배를 처음 넘는 인덱스가
    **정확히 29 인 비율 rgbd 0.855 / text 0.867 / gdstyle 0.925**, 29~33 이면 0.94~0.98.
    스텝 최대/median 비는 p30 런 **1.5~1.7** 대 p49 런 **~50**.
  · 원인은 `gendop_release_infer.py` 의 `--forbid_eos` (pose_length≠30 이면 auto on). EOS 로짓을
    −inf 로 막아 `max_new_tokens` 까지 강제 생성시키므로, 릴리즈 학습 길이 30 을 넘긴 19 pose 는
    학습 분포 밖 토큰이다. `n_poses_hist {49: 875}` 는 "성공"이 아니라 **EOS 를 막은 결과**다.
  · 2차 오염: `rmax = max_f |rel[f,:3,3]|` 의 argmax 가 **90~92%** 에서 index≥30 (쓰레기 꼬리)에
    있다. 즉 rescale 판본의 배율은 노이즈가 정했고, `rmax(49)/rmax(0..29)` median 1.27~1.37 만큼
    유효 구간이 눌렸다.
  · camel 4 entry depth-warp hole: p49_raw 0.42/0.52/0.43/0.55 → p30_raw 0.17/0.40/0.14/0.20.
  · **판독은 p30 native → 49 index-pick 으로 되돌린다.** p49 로 낸 CLaTr/caption F1/충돌률/
    subject_in_frame 표는 전부 이 꼬리를 포함하므로 폐기.
- **`CinemaTraj/scripts/gendop_release_infer.py --eval_dir_prefix` — `--text_from_eval_dir` 의
  파일명 접두사가 `vista4d` 로 하드코딩돼 있었다 (D158, 2026-09-07).** eval 폴더는 코퍼스
  이름으로 접두사를 붙이는데(`dynpose_<scene>_<idx>_caption.json`) 조회는 항상
  `vista4d_...` 를 찾아서, dynpose 를 돌리면 **전 엔트리가 조용히 `skipped` 로 빠지고 rc=0**
  으로 끝났다 (smoke 4개 → written 0 / skipped 4). 접두사를 인자로 뺐고 **기본값 `vista4d`**
  라 기존 vista 호출은 그대로다. 같은 사고를 다음에 바로 보이게 하려고 요약표에
  `no_caption` 행을 추가했다 — 0 이 아니면 접두사를 확인하라고 같이 찍는다.
- **`CinemaTraj/scripts/run_bank.py` — `--eval_data` 를 그걸 안 받는 단계에도 붙이던 것
  (D157, 2026-09-07).** `_base_args` 가 config 의 `eval_data` 를 **모든 단계에** 붙였는데
  `route_presets.py` 와 `emit_bank.py` 에는 그 argparse 인자가 없다 (둘 다 `output_root`
  아래 산출물만 읽는다). 그래서 `unrecognized arguments: --eval_data` 로 rc=2 다.
  **vista 세대(d150/d151)는 config 의 `eval_data` 가 `null` 이라 이 경로를 한 번도 안
  밟았고**, dynpose/trumans 처럼 데이터 위치를 지정하는 세대에서만 터진다 — 즉 드라이버가
  vista 전용으로 굳어 있었다. 옛 bash 러너(`run_dynpose_d149_shard.sh`)는 route/emit 호출에
  손으로 `--eval_data` 를 안 넘겨서 차이가 안 드러났다. `STAGE_TAKES_EVAL_DATA =
  {"graph","cloud","tau","fit"}` 를 두고 `_base_args(..., stage=...)` 가 그 집합일 때만
  붙인다. D157 smoke 1편으로 실측 확인 (GRAPH 124.4s / CLOUD 29.8s / ROUTE 0.4s /
  TAU 84.1s / FIT 484.4s / EMIT 11.8s, 전부 rc=0).

- **`gendop_release_infer.py` — 0 quaternion 이 회전 3×3 을 통째로 NaN 으로 만들던 것
  (D156, 2026-09-07).** GenDoP 는 포즈당 10 토큰 중 앞 4 개가 quaternion 이고 `decode_tokens`
  가 `coords[:, :7] / (0.5*bins) - 1` (bins=256) 로 역양자화한다. 회전 4 토큰이 **전부 bin
  128** 이면 `q = (0,0,0,0)` 이 되고, `/data1/cympyc1785/pipeline/GenDoP/core/utils.py:209
  quaternion_to_matrix` 가 0 노름으로 나눠 그 프레임 회전이 전부 NaN 이 된다. NaN 은 npz →
  eval_dir JSON → 렌더러까지 살아남아 `Vista4D/utils/point_cloud/point_cloud.py:8` 의
  `torch.inverse` 가 `linalg.inv: ... singular` 로 죽는다.

  `decode_tokens` 가 그 프레임을 **직전 프레임 회전으로 이어붙인다** — 0 quaternion 은
  "회전 정보 없음"이지 "회전 0" 이 아니라서 identity 로 박으면 궤적이 튄다. 첫 프레임이면
  identity. 때운 프레임 수를 npz `n_zero_quat` / config.json `zero_quat_entries` / 요약
  `zero_quat` 에 남기고, 반환 직전 `np.all(np.isfinite(out))` assert 를 건다.
  **`det` 로는 못 잡는다** — NaN 블록의 `np.linalg.det` 는 NaN 이고 임계값 비교가 항상
  False 라 특이행렬 검사를 그대로 통과한다. 포즈 검사는 `isfinite` 를 먼저.

  d121 test 875 entry 실측: `--pose_length 49` text arm 3 건(`avocado-slice/25` f32·33 /
  `bmx-bumps/42` f37 / `camel/323` f35), rgbd arm 0 건. 30 포즈 런은 0 건이라 예전 결과는
  영향 없다. 가드가 `model.generate()` **뒤**라 토큰은 안 건드리고, translation 토큰은
  따로라 유한했으므로 이미 나온 npz 3 개는 직전 프레임 회전 복사로 제자리 수리했다
  (재추론과 bit-identical). GenDoP 리포는 여전히 0 줄 수정.

- **`tmp/d156/run_gendop_d121.py` — `--eval_data` 규약이 스크립트마다 달라 score 가 한 번도
  성공한 적이 없던 것 (D156, 2026-09-07).** `dynpose_gendop_inputs.py:74` 는
  `<eval_data>/recon_and_seg/<scene>` 로 join 하는데, `eval_subject_in_frame.py` 가 타는
  `CinemaTraj/scene_graph/io.py:88 load_scene` 은 `<eval_data>/eval_data/recon_and_seg/<video>`
  로 "eval_data" 를 한 번 더 붙인다. 드라이버가 상수 하나를 양쪽에 넘겨
  `.../Vista4D-Eval-Data/eval_data/eval_data/...` 가 됐고 `ValueError: Could not open video
  file` 로 죽었다. `EVAL_DATA`(inputs 용) / `EVAL_DATA_PARENT`(score 용)로 갈랐다.
  **30 포즈 런의 `subject_in_frame.json` 도 같은 이유로 생긴 적이 없다** — 다시 돌려야 한다.

- **`CinemaTraj/scripts/extract_static_nouns.py` — `prompt` 열이 없는 코퍼스에서 KeyError
  (D149, 2026-09-06).** `read_prompts` 가 `row["prompt"]` 로 직접 인덱싱했는데 DynPose-LBM 의
  `metadata.csv` 는 `video,dynamic` 두 열뿐이다 (`extend_dynpose_metadata.py` 의 `FIELDS`).
  vista 는 9열이라 안 걸렸다. `row.get("prompt", "")` 로 바꾸고, 그 값을 실제로 쓰는
  `--source prompt` 분기에 명시적 assert 를 넣었다 — 빈 문자열을 그냥 흘리면 정적 명사가
  0개가 되어 **정적 트리가 통째로 비는데 로그는 정상으로 보인다**. `--source vlm`(D149 가 쓰는
  경로)은 prompt 를 안 읽으므로 영향 없고, vista 동작은 글자 그대로 같다.

- **`CinemaTraj/scripts/route_presets.py` — 세로 슬롯(crane/pedestal)이 전량 조용히 빠지던 것
  (D147, 2026-09-06).** 슬롯 게이트가 `grav == "ground_ransac"` 이었는데 D98 이 gravity 를
  GeoCalib 로 옮기면서 그 이름이 한 번도 안 나오게 됐다 — 실측으로 dynpose 280/280,
  vista 52/53 이 `geocalib` 이다. vista 는 라우팅을 안 타고 preset 을 전량 열거해서 안 걸렸지만
  dynpose 는 이 함수를 타므로 `dd10` 코퍼스 10,857 행에 crane/pedestal 이 **0건**이었다.
  `--vertical_gravity`(기본 `ground_ransac,geocalib`)로 신뢰할 gravity 방법을 목록으로 받게 했다.
  함수 기본값은 `("ground_ransac",)` 로 옛 동작을 그대로 재현하고, 새 동작은 CLI 기본값이다
  (D143 과 같은 규약). reasons 에 `vertical_gravity_ok` 열을 추가했다. 옛 동작 재현은
  `--vertical_gravity ground_ransac`. **이 수정은 뱅크를 다시 구워야 산출물에 반영된다** —
  D145 의 880편 굽기에서 들어간다.

### Added
- **`CinemaTraj/scripts/run_dynpose_d149_shard.sh` — dynpose 880편 앞단까지 vista 방식으로
  전면 재굽기 (D149, 2026-09-06).** 사용자 지시: "detection은 VLM 사용한걸로 해주고 static
  트리도 추가해줘. dd는 일단 그럼 빼주고 vista 돌린거랑 일관성 있게 dynpose돌려줘" +
  "유일하게 vista랑 다른건 모든 preset별로 하는게 아니라 slot별로 일부만 fitting한다는거야".
  d129 대비 4축: ① detection 을 VLM 명사로 통일(옛 279편은 배포 annotation 명사였다 —
  한 코퍼스에 탐지 어휘가 두 종류였다) ② `seg_instances_static` 추가 — vista 는 80편이 있고
  그래프가 자동으로 읽는데(`build_scene_graph.py:540`) dynpose 는 디렉토리 자체가 없어서
  static 노드가 0 이었다. `vlm_nouns.json` 의 `static` 절반은 d145 때 이미 뽑아 뒀고 쓰이지
  않고 있었다 ③ `--gravity_source auto` → `geocalib`(d148 이 880/880 사이드카를 채웠다)
  ④ `--num_external 0` — dd_* 제외로 라우팅 preset 29 → 14.
  d129 의 `--tau_ladder 1.00` 은 vista 기본값 `0.10 0.20 0.35 0.60 1.00` 로 **되돌린다** —
  사다리는 hole 뱅크 크기를 곱하지 않고(실측 vista TAU 321→hole 258, d129 TAU 58→hole 220)
  "이 (anchor,preset) 이 어느 세기에서든 통과하나"를 묻는 역할이라, 단일단이면 τ=1.0 에서
  죽는 조합이 통째로 사라진다. 나머지 TAU/FIT 플래그는 `run_k6_d128_shard.sh` 와 문자 단위로
  같다. GRAPH/CLOUD 를 다시 돈다(①②③ 이 전부 그래프 입력) — 새 마커 `.graph_d149`/
  `.cloud_d149`, 옛 그래프는 `scene_graph_pre_d149.json` 으로 1회 백업.
  anchor 상한 3/3 은 `route_presets.py` 에 준다 — 뱅크에 주면 no-op 이고, 버킷을 가르는 건
  id 접두사가 아니라 `moving` 이다. ② 로 static 노드가 생기므로 `--max_static_anchors 3` 이
  **처음으로 실제 일을 한다**.

- **`CinemaTraj/scripts/run_dynpose_d147_caption_export.sh` — dynpose 프레이밍 약속 위반 캡션을
  고친 별도 코퍼스 (D147, 2026-09-06).** D143 이 vista d121/d128 에 넣은
  `--framing_min_in_frame 0.85` 게이트를 dynpose d137 뱅크에도 적용한다 (그때는 D137/D141 이
  그 코퍼스로 학습 중이라 뺐다). d137 배포 캡션 헤더에는 `framing_min_in_frame` 필드 자체가
  없다 — D143 이전 코드로 구운 것이다. 실측: dd10 10,857 행 중 `subject_in_frame < 0.85` 가
  4,570(42.1%)이고 그중 3,225(코퍼스의 29.7%)가 여전히 framing 절을 달고 있었다. 뱅크 전량
  46,425 변이 기준으로는 6,312(13.6%)에서 절이 빠진다. 기존 루트를 덮지 않고
  `latentcam_dynpose_d137c147` 을 새로 판다(~14 GB) — 제자리에서 갈면 D137/D141 의 학습 캡션과
  재평가 캡션이 어긋난다. 캡션만 바꿨으므로 `seg_list_dynpose_{,dd10_}{train,test}.txt` 와
  `meta_dynpose.csv` 가 d137 과 **글자 그대로** 같아야 하고, 스크립트 마지막 단계가 그걸
  `cmp` 로 확인한다. `--videos` 는 `all` 이 아니라 뱅크에서 직접 뽑는다 (FIX-D129-a).

- **`CinemaTraj/scripts/{run_dynpose_d145_nouns.sh,extend_dynpose_metadata.py}` — dynpose
  코퍼스를 267 → 880 편으로 넓히기 위한 앞단 2종 (D145, 2026-09-06).**
  - `run_dynpose_d145_nouns.sh`: 명사 없는 600편의 VLM 명사 추출 드라이버. `extract_nouns_vlm.py`
    는 **끝날 때 한 번만** json 을 쓰므로(그 파일 :215-229) 600편 단일 프로세스는 590편째 예외
    하나에 전부 날아간다. 스크립트는 안 고치고 `--merge` 로 25편씩 끊어 부른다 — 손실 단위가
    600 → 25 편. 매 batch 전에 이미 들어간 `(video, multi)` 를 빼므로 재실행하면 이어서 간다.
  - `extend_dynpose_metadata.py`: `vlm_nouns.json` → `DynPose-LBM/metadata.csv` 행 추가.
    `sam3_seg_instances.py:69` 가 그 csv 를 읽고 없으면 `assert not missing` 으로 죽는다.
    **`dynamic` 열에는 VLM 의 dynamic 명사만 넣고 static 은 버린다.** 이 csv 로 들어온 명사는
    `scene_graph/io.py:101` 에서 전부 `kind="dyn"` 이 되지만(dynpose 에는 `seg_instances_static`
    이 없다), 앵커 버킷을 가르는 건 `kind` 가 아니라 `moving`(`path_len_u > 0.05`)이라
    안 움직이는 dyn 노드는 `--max_static_anchors` 버킷으로 넘어간다
    (`sample_camera_bank.py:810,823` 의 실측 주석: "484 노드 중 moving=True 124개, 전부
    kind='dyn'; dyn 이어도 52개는 안 움직인다"). 즉 wall/floor/pillar 를 넣으면 그게 그대로
    **static target 후보**가 된다 — 사용자가 금지한 것이다. 게다가 기존 279행은 저자 라벨의
    비-이동 물체(chair/shelf/box)만 그 버킷에 넣으므로, 신규 600편에만 VLM static 명사를
    넣으면 코퍼스 절반의 static 앵커 수가 달라진다. static 명사는 버리지 않고 사이드카
    `metadata_provenance_d145.json` 에 남겨 두어, 나중에 `sam3_static_instances.py` 로
    별도 `seg_instances_static` 트리를 만들 때 재료로 쓴다.
    csv 스키마(`video,dynamic`)는 그대로 두고
    provenance(저자 라벨 279행 vs VLM 신규)도 그 사이드카에 적는다. 쓰기 전 타임스탬프 백업.
  - `run_dynpose_d145_sam3.sh`: 확장된 csv 로 SAM3 인스턴스 분할을 GPU 샤딩해 돌리는 런처.
    `sam3_seg_instances.py` 가 이미 `--num_shards/--shard_id/--skip_done` 을 갖고 있어
    스크립트는 안 고친다. `--videos` 를 안 넘기는 게 의도다 — 목록을 안 주면 csv 전량을 쓰고
    (`:70`) `--skip_done` 이 `masks.npz` 있는 기존 279편을 건너뛰므로 결과적으로 신규분만 돈다.
- **`scripts/build_bank_captions.py --framing_min_in_frame` / `--no_framing_on_free` +
  `scripts/run_d143_caption_export.sh` — 프레이밍 약속을 못 지킨 변이에서 그 절만 빼는
  캡션 게이트 (D143, 2026-09-06, 사용자 지시 "5번은 고쳐야할 것 같아. 지금 학습 안돌리는
  데이터셋 먼저 고쳐줘"; 태스크 #134).**
  · **무엇이 모순이었나.** `aim="free"` preset 은 시작 pose 의 회전을 49프레임 내내 들고
    간다 — **한 번도 재조준하지 않는다**. 그런데 캡션의 framing 절("keeping it in a medium
    shot")은 대상을 프레임에 유지하겠다는 약속이다. 그 절을 막는 게이트는
    `configs/caption_presets.json` 의 `targetless` 플래그뿐이었고 그건 `pan_*`/`tilt_*` 5개에만
    붙어 있어서, `aim="free"` 인 나머지 15 preset(`truck_*` `pedestal_*` `dolly_*`
    `static_hold` `static_zoom_in` `track_hold` `track_truck_*` `track_dolly_*`
    `track_pedestal_*`)이 그대로 통과했다. vista d121 train 실측: `track_*`∧`aim=free`
    1,415 변이 중 `subject_in_frame < 0.5` 가 448(31.7%)인데 캡션 프레이밍 약속은
    1,121(79.2%).
  · **preset 축이 아니라 변이 축으로 갈랐다** (`--framing_min_in_frame`, 기본 0.85 =
    그 변이의 실측 `subject_in_frame` 이 임계 미만이면 framing/composition 절을 뺀다).
    preset 이름으로 뭉뚱그리면 두 방향으로 틀린다 — vista d121 train, `aim=free` 만,
    `subject_in_frame` median / `<0.85` 비율:
        dolly_out       1.000 / 19.4%   track_dolly_out  1.000 /  4.8%
        track_hold      1.000 / 10.8%   static_hold      1.000 / 20.8%
        truck_left      0.538 / 69.1%   track_truck_left 0.385 / 78.8%
    자기 축으로 물러나는 `dolly_out` 계열은 재조준 없이도 대상이 중앙에 남는데 preset 축
    규칙은 그걸 같이 버리고, 반대로 `aim=look_at` 도 2.9% 는 프레임을 놓치는데 그건 아예
    못 잡는다. preset 축 규칙도 `--no_framing_on_free` 로 남겨 뒀다 (기본 꺼짐).
  · **motion 절과 target 은 그대로 둔다.** `track_*` 의 "tracks {target}" 은 follow_gain 1.0
    으로 실제 병진 추종을 하므로 참이다. 비-track 의 "sliding sideways past {target}" 도
    참이다(지나친다고 말하지 유지한다고 말하지 않는다). 거짓인 건 프레이밍 약속뿐이라
    그 절만 뺀다. `nl` 형식에서는 composition 절도 같이 빠진다 — 그건 framing 뒤에만 붙는다.
  · D140 의 `suspect: aim_free_subject_lost` 와 **같은 현상의 다른 처방**이다. 저건 변이를
    export 에서 **버리고**(`--drop_suspect`), 이건 변이를 **남기고 캡션만 참으로 만든다**.
    d121/d128 뱅크에는 `suspect` 열 자체가 없어(D140 이전에 구움) 그쪽 손잡이를 못 쓴다.
  · 캡션 파일 헤더에 `framing_on_free` / `framing_min_in_frame` / `framing_dropped` 를 남긴다.
    `framing_dropped` 는 **게이트 적중 수가 아니라 실제로 문장이 바뀐 수**다 — 이미 targetless
    인 preset 은 게이트가 걸려도 캡션이 그대로라, 적중 수를 적으면 효과가 부풀려진다
    (camel/d128 에서 적중 96 : 실제 변화 54).
  · `--framing_min_in_frame 0` 이 옛 동작이고, 그 값으로 구우면 배포된 d121/d128 캡션이
    **비트 동일**하게 재현된다 (camel/snowboard/parkour 로 확인).
- **`scripts/fit_hole_ladder.py` 의 `suspect` 열 + `scripts/vista4d_bank_to_dl3dv.py
  --drop_suspect` — "게이트가 막은 게 아닌데 수치가 나쁜" 변이 진단 태그 (D140, 2026-09-06,
  사용자 지시 "충돌, subject in frame 같은 수치가 의도와 다르게 preset과 첫 카메라가 놓인
  상황을 봤을 때 불가능한 상황이 아닌데도 안좋게 나오면 일단 clamped_low처럼 인식할 수
  있게끔 해두고").**
  · **`status` 와 직교하는 새 축이다.** `status` 는 이분법이 어떻게 끝났는지(`solved` /
    `clamped_low` / `<gate>_limited`), `binding` 은 `over()` 에서 **처음** 걸린 게이트다.
    둘 다 "게이트가 막았다"만 말한다. 이번 태그는 그 반대 — **게이트를 다 통과했는데도**
    캡션과 기하가 어긋난 행을 잡는다. 그래서 `status` 를 덮어쓰지 않는다: 덮어쓰면 어느
    게이트가 물렸는지가 지워지고 `binding` 과 어긋난다.
  · 태그 3종 (`|` 로 이어 붙임, 빈 칸 = 정상):
    ① `aim_free_subject_lost` — `aim == "free"` 인데 `subject_in_frame < 0.85`.
       같은 뱅크 안 대조가 근거다. TRUMANS d132 `aim=look_at` n=3241 → 0.85 미만 **0.0%**
       (p10 = 1.000) vs `aim=free` n=2681 → **36.2%** (p10 = 0.462). VISTA d128 은
       look_at 4.5% vs free 54.1% (p10 = 0.077). 즉 게이트가 아니라 **조준을 안 한 것**이다.
    ② `static_start_collision` — `behind_frac > 0` 인데 `status == "static"` 이거나
       `path_len_u < 0.02`. 손잡이가 inert 라 이분법이 줄일 게 없고, 위반은 **시작 pose**
       에서 온다. d132 24행 전부 `track_hold`/`static_hold`/`*_look_at` 이 `knob = 0.1000`
       (하한) 에 앉아 있고 `behind_static_frac == behind_frac`, `behind_dyn_frac = 0`.
       `tru_00add26c_a18` static_hold 는 `behind_frac = 1.000` (49프레임 전부 벽 안).
    ③ `motion_preset_no_motion` — 이동 preset 인데 `path_len_u < 0.02`,
       **단 `binding` 이 `hole`/`none`/빈칸일 때만**. 게이트가 막아서 못 움직인 건 이미
       `status` 가 말하므로 뺀다. d132 48 chunk 실측 0행(259행 전부 elev/collision/obb 구속),
       VISTA d128 에서는 살아 있다.
  · 임계 3개는 `--suspect_in_frame 0.85` / `--suspect_behind 0.0` / `--suspect_path_len 0.02`
    로 노출. `--no_suspect` 면 열이 빈 칸이라 **D139 이전 뱅크와 같다**.
  · **판정이 아니라 진단이다** — 궤적·게이트·이분법에 영향이 없다. 실측 검증(chunk
    `tru_00add26c_a01`, 148 변이): `poses.npz` 최대 절대차 **0.0**, `variant_id` 순서 동일,
    `knob`/`status`/`binding` 전부 동일. 다른 필드는 렌더 측정값 6개
    (`subject_area_seq` 최대차 0.0063, `subject_visible_frac` 0.0236,
    `subject_visible_min` 0.0351, `subject_area_end` 0.0042, `subject_area_med` 0.0021,
    `near_depth` 0.0742) 만 흔들리는데, 이건 splatting 래스터라이저의 GPU 비결정성이지
    D140 변경분이 아니다 (d132 는 GPU 2, 검증은 GPU 4).
  · export 쪽 `--drop_suspect` 는 `--drop_status` 와 **다른 축**이라 별도 플래그다. 태그
    하나짜리 행의 대부분이 `status == "solved"` 라 status 필터에 안 걸린다 (d132 1022행 중
    solved 572). `suspect` 열은 목록이므로 접두사가 아니라 **토큰** 매칭 — 접두사로 하면
    `aim_free_subject_lost` 가 두 번째 토큰일 때 못 잡는다. 기본값 빈 리스트라 안 주면
    기존 코퍼스와 비트 동일하고, `suspect` 열이 없는 예전 뱅크에서도 no-op 이다.
- **`scripts/vista4d_bank_to_dl3dv.py --drop_status` — export 단계 status 필터 (D137,
  2026-09-06, 사용자 지시 "어차피 clamped_low는 정상적인 카메라가 아니니 b로 해줘").**
  뱅크 변이를 `bank.json` 의 `status` **접두사**로 걸러 코퍼스에 안 내보낸다. 기본값은 빈
  리스트라 **안 주면 기존 코퍼스가 비트 동일하게 나온다**.
  · **왜 필요했나**: 지금까지 export 에는 status 필터가 **한 군데도 없었다** — `convert_scene`
    이 `bank["variants"]` 를 전량 내보내고, 유일한 감축은 pose 비트 동일 dedup 뿐이었다.
    `status` 는 `prompts.json` 에도 안 실려서 하류에서 거를 수도 없었다.
  · `clamped_low` 는 `scripts/fit_hole_ladder.py:513-531` 에서 "knob 을 하한까지 밀었는데
    **하한에서도 `over()` 가 참**" 일 때 찍힌다 = 제일 작게 만들어도 게이트를 위반하는 카메라.
    d129 `dd10` train 612행 실측: binding 은 `collision` 340 / `obb` 114 / `approach` 87 /
    `ground` 43 / `elev` 26 / **`hole` 2** — 즉 τ 예산이 아니라 물리 게이트다. `path_len_u`
    중앙값 **0.0330** (solved 는 0.6900) 인데 캡션은 `dolly_in`(83) / `s_curve`(78) /
    `dolly_out`(68) 이라고 써 있다. 사실상 정지 카메라에 이동 캡션이 붙은 학습 신호였다.
  · **접두사 매칭이라야 한다.** `+tau_floor` 접미사는 `fit_hole_ladder.py:910` 에서 붙는
    **직교하는** 표시(knob 이 소스 시차 위에 올린 하한에 앉음)라 `solved+tau_floor`(d129 470행)
    처럼 정상 행에도 붙는다. 접미사로 거르면 멀쩡한 카메라가 같이 날아간다.
  · **필터를 dedup 앞에** 뒀다. 순서가 뒤바뀌면 대표만 clamped_low 인 중복 그룹이 통째로
    남는다. 60씬 2048 중복 그룹 실측에서 clamped_low 와 아닌 것이 섞인 그룹은 **0** 이라
    지금 데이터에선 결과가 같지만, 의미상 필터가 먼저다.
  · 요약표에 `drop` 열 + `status 제외 N / M (%)` 블록과 status 별 히스토그램을 찍는다.
    d129 뱅크 실측: **6392 / 46425 (13.77%)** 제외 (`clamped_low+tau_floor` 5924 +
    `clamped_low` 468).
- **`scripts/run_dynpose_d129_export.sh` 에 `DROP` / `SKIP_CAPS` 환경변수 (D137, 2026-09-06).**
  `DROP` 기본값 `clamped_low` 를 위 `--drop_status` 로 넘긴다. **`${DROP-...}` 이지
  `${DROP:-...}` 가 아니다** — `DROP=` 로 빈 문자열을 주면 필터 없는 옛 코퍼스가 그대로 재현된다.
  `SKIP_CAPS=1` 이면 캡션 재굽기를 건너뛰고 뱅크/캡션 **개수 검산만** 한다 (뱅크가 안 바뀌었으면
  캡션도 안 바뀐다). 코퍼스 root 는 `latentcam_dynpose_d137` 로 새로 판다 — d129 를 덮어쓰면
  필터 전/후가 섞인 상태가 생기고 비교 대상이 사라진다.
- **`scripts/run_trumans_d132_shard.sh` — TRUMANS **3D mesh 충돌** 뱅크 드라이버 (D132,
  2026-09-05, 사용자 지시 "내가 fitting하는거 blender scene에서 3d mesh로 해서 하자고 안했나?
  context video에 안보이는 부분이더라도 충돌, clearnace 같은거 고려해서 전체 scene에 안부딪히도록
  fitting 한 sequence에 대해서만 해주고 target은 우선 man만 해서 돌려주고 렌더 보여줘").**
  D128 축 위에서 두 가지만 바꾼다: ① `fit_hole_ladder.py --collision_source both` 로 depth
  shell 뿐 아니라 **`.blend` 씬 mesh** 와의 충돌까지 본다 — context 영상에 안 보이는 벽·가구도
  막힌다. ② `--nodes dyn_0` / `--anchors dyn_0` 로 target 을 **사람 하나**로 고정한다.
  · 3단계 TAU(`sample_camera_bank.py`) -> FIT(`fit_hole_ladder.py --collision_time_match`) ->
    EMIT, EMIT 은 FIT rc==0 일 때만 돈다. 기본값 `TAU=bank_d132` / `BANK=hole_bank_k6_d132` /
    `EASE=smooth_kf`. 인자 `<gpu> <shard_id> <num_shards> <video_list> <log_dir>`.
  · 48 chunk 4샤드 실측 — 변이 6794 중 `solved` **1327**, `collision_limited` **2813 (41.4%)**,
    `approach_limited` 815, `clamped_low+tau_floor` 808, `obb_limited` 241, `elev_limited` 235,
    `static` 182, `shape_limited` 179, `collision_limited+tau_floor` 90, `clamped_low` 64,
    `ground_limited` 23, `elev_limited+tau_floor` 17. solved 0 인 chunk 1/48,
    solved/chunk 중앙값 28 (최소 3 / 최대 42). mesh 게이트가 실제로 물고 있다.
  · **`bank.csv` 에는 depth-vs-mesh 분해 열이 없다** — `behind_static_*`/`behind_dyn_*` 는
    static/dynamic 채널 분해지 충돌 소스 분해가 아니다. 그래서 depth-only 대조군은 별도 뱅크
    (`hole_bank_k6_d132depth`)로 따로 구웠다.
  · 릴: `results/20260905_d132_preset_warp` — moving `a15`(drift 0.084, preset 22 / track 8) ·
    `a09`(0.230, 21/8), static `a07`(0.017, 12/0) · `a03`(0.038, 12/0).
- **`scripts/pick_warp_sample_scenes.py` + `scripts/run_preset_warp_sample.sh` — 뱅크 재굽기 뒤
  **표준 확인 절차**를 형식화했다 (2026-09-05, 사용자 지시 "앞으로 돌리는거 끝나면 저렇게
  dynamic subject 움직이는거 두개정도 움직이지 않는거 2개 정도해서 preset별 depth warp
  비교영상 만들어서 보여주는거 형식화해줘").**
  움직이는 subject 2편 + 안 움직이는 subject 2편을 골라 preset 별 **사다리 최대단** depth-warp
  릴을 굽고 한 폴더에 모은다.
  · **왜 두 버킷인가**: preset 의 실패 모드가 subject 이동 여부로 갈린다. 움직이는 쪽은
    `track_*` 가 켜져 **추종·조준**을 시험하고, 안 움직이는 쪽은 track 이
    `--track_min_drift_u 0.05` 게이트에서 잘려나가 순수 object-centric 만 남아 **hole·벽 뚫기**를
    시험한다. d128 실측으로 확인된다 — moving 표본(street-turn / couch-sit)은 track preset
    12/11개, static 표본(room-argue / cows)은 **0개**. 한쪽만 보면 나머지 절반이 조용히 깨진
    채로 학습에 들어간다.
  · 버킷 경계는 `scene_graph.json` 의 `center_drift_u` 를 `--drift_thresh`(기본 0.05,
    `--track_min_drift_u` 와 **같은 값**)로 자른다. 두 값이 어긋나면 버킷이 실제 뱅크와 안 맞는다.
  · **RNG 없음.** 1순위 preset 다양성(내림) / 2순위 drift(움직이는 쪽 큰 순, 안 움직이는 쪽
    작은 순) / 3순위 이름. 재굽기 전후로 같은 표본이 나와야 before/after 를 짝지어 본다 —
    `center_drift_u` 는 `build_scene_graph.py` 산출물이라 축을 바꿔도 안 변한다.
  · anchor 선택 규칙은 릴 러너(`dyn_0` 우선, 없으면 solved 행 최다)와 **일부러 같게** 복제했다.
    여기서 분류한 subject 와 릴에 찍히는 subject 가 다르면 버킷이 거짓말이 된다.
  · d128 기본 표본: moving `street-turn`(drift 0.0674, preset 30) / `couch-sit`(0.1581, 28),
    static `room-argue`(0.0095, 18) / `cows`(0.0332, 18).
- **`sample_camera_bank.py --track_min_drift_u`(기본 0.05) — `track_*` 라우팅에만 거는 순변위
  하한 (D128, 2026-09-05, 사용자 판단 "움직이는 dynamic 물체는 맞지만 위치가 별로 안 움직이는
  거잖아").**
  기존 게이트 `--track_dynamic_only`(D77, `:813`)는 `node["moving"]` 하나만 보는데, 그 `moving`
  은 `path_len_u > 0.05` 라 **제자리 흔들림**(춤·손짓·그네)을 통과시킨다. 그래서 `path_len` 이
  아니라 **순변위** `center_drift_u = |c[-1] − c[0]|` 로 한 번 더 자른다.
  · 실측(d121 뱅크): 순변위 ≤0.05u 인 dyn 노드 **49개가 전부** 기존 게이트를 통과했고
    (`path_len_u` min 0.053 / med 0.108 / max 0.447) 그 위에 `track_*` **2,040 쌍**이 실렸다.
    그 쌍들의 track vs 비-track 짝 차이는 `|Δpath_len|` med **0.0336** / `|Δτ|` med **0.0210** —
    진짜 이동하는 anchor(0.1285 / 0.2071) 대비 10배 약하다. 비트 동일은 아니지만 "follows the
    subject" 캡션을 떠받치기엔 얇다.
  · **`--track_min_drift_u 0` 이면 d121 과 비트 동일.**
- **`scripts/run_dynpose_d129_{shard,export}.sh` + `configs/dynpose_holdout_scenes.txt` —
  dynpose 코퍼스를 D128 축으로 재굽고 **DataDoP 10%**(dd10) 리스트까지 (D129, 2026-09-05,
  사용자 지시 "같은 방식으로 dynpose를 datadop 10% 비율로 해서 데이터 만들어놔줘").
  **아직 안 돌렸다** — vista D128 캡션·export 검증이 깨끗할 때만 착수한다.**
  d122 뱅크 manifest 실측으로 6축이 전부 옛 값임을 확인했다: `fixed.tau_ref=auto` /
  `fixed.tracking=drift` / `axes.anchors=['dyn_0','dyn_1']`.
  · **`run_dynpose_d122_shard.sh` 는 지금 HEAD 에서 깨져 있다.** `route_presets.py --num_anchors 2`
    가 D127b 에 `--max_dynamic_anchors`/`--max_static_anchors`(`route_presets.py:332-333`)로
    바뀌었다. 그대로 돌리면 argparse exit 2 → `$ARGS` 공백 → `FAIL $VIDEO route` 로 **전 씬이
    조용히 스킵**된다 (크래시가 아니라 밤새 0편 굽는다). d129 러너가 이걸 고친다.
  · **anchor 상한과 surface drop 은 `sample_camera_bank.py` 가 아니라 `route_presets.py` 에
    준다.** route 가 `--nodes ...` 를 명시로 넘기고 `sample_camera_bank.py:144-151` 이 그걸 받으면
    `pick_main_anchors` 를 **호출하기 전에 return** 한다 — 뱅크 쪽에 붙이면 no-op 이다. 실제
    적용점은 `route_presets.py:82` 이고 surface drop 은 `schema.py:164` 기본 `True` 라 플래그가
    없다. 280 그래프 실측: 3/3 이면 편당 anchor 평균 **2.91**(dyn 2.70 / stat 0.21, 최대 6),
    d122 의 flat 2 대비 variant **~1.45배**(23,424 → ~34,000). surface drop 발동은 12/280 편
    (door 13 / locker door 5 / wall 2 / car 1 / refrigerator door 1 / road 1).
    `MAXDYN`/`MAXSTAT` 환경변수로 d122 parity(2/0)를 낼 수 있게 열어뒀다.
  · **`--track_min_drift_u 0.05` 는 dynpose 에서도 no-op 이 아니다** — `sample_camera_bank.py:827-833`
    이 명시 `--presets` 목록 위에서 돈다. d122 뱅크 실측 `track_*` 6,322 중 **915(14.5%)** 가 빠진다.
  · **`--tau_ref follow` 는 vista 보다 크게 먹는다** — `dd_*` 도 같은 경로다
    (`lbm/presets.py:339-345`: 명시 `follow` 가 preset 표를 이긴다). `--tau_ladder 1.00` +
    anchor 당 dd 슬롯 4개 구조라 follow 아래선 `tau_start`≡0 이라 `tau_saturated` 가 영영 안 걸린다
    (`decode/build_poses.py:707-709`). `--trackings lock` 은 `dd_*` 궤적을 안 바꾼다
    (`aim="traj"` → `build_poses.py:651` `tracking_ignored`), 기록 문자열만 바뀐다.
  · FIT 에 `--behind_clear_src_ratio 0` 을 **명시**한다 (기본값도 0.0이지만 D105).
  · GRAPH/CLOUD 는 안 돈다 (`.graph_d122`/`.cloud_d122` 마커 요구). `describe_instances_vlm.py`
    도 안 돈다 — `instance_desc.json` 은 노드 id 키잉이고 graph 를 안 건드린다.
  · export 는 **새 root** `latentcam_dynpose_d129` 로 판다. d122 root 재사용 금지 —
    `variant_id` 는 재굽기 전후로 문자열이 같아서 `prompts.json`·seg list·geo 캐시가 miss 0 으로
    조용히 통과하면서 전부 옛 카메라를 가리킨다.
  · holdout 27편을 `/tmp/d107_test_videos.txt` 에서 리포 안
    (`configs/dynpose_holdout_scenes.txt`)으로 옮겼다. d107/d110/d122 와 같은 분할이라야 paired.
  · dd10 = `filter_seg_list_by_preset.py --target_frac_prefix dd_ --target_frac 0.10 --suffix dd10`.
    export 스크립트 4단계가 `prompts.json` 을 되읽어 **실제 `dd_*` 비율을 검산해서 찍는다**.
- **`scripts/run_k6_d128_shard.sh` — D128 뱅크 샤드 러너 (vista 52편).**
  `bank_d128`(τ) / `hole_bank_k6_d128`(fit). d121 대비 바뀐 축 6개를 **전부 명시적으로** 넘긴다
  (D105: 뱅크 정체성을 argparse 기본값에 맡기지 않는다): `--tau_ref follow` ·
  `--track_min_drift_u 0.05` · `--max_dynamic_anchors 3 --max_static_anchors 3` ·
  `--anchor_drop_surfaces` · `--trackings lock`/`--tracking lock` · `--behind_min_zcam 0.02`.
  GRAPH/CLOUD 는 안 돈다 — `build_scene_graph.py` 가 d115 와 비트 동일이고 `.graph_s115` /
  `.cloud_s115` 마커가 52편 전부에 있다.
  · parkour smoke: anchor 7 → **4**(dyn 1 / stat 3), variant **664 → 398**(60%),
    `tau_ref` 398/398 follow, `tracking` 398/398 lock, `track_*` 77개(dyn_0 한정).
- **D123 G1 두 손잡이 — `min_zcam_frac`(degenerate 투영 하한) + `source_g1_clear`(소스 자신의
  여유로 임계를 유도) (2026-09-04, 사용자 질문 "g1_min 자체가 오염돼 있었다는 게 무슨 소리임?"
  + "소스에서 일관적으로 기준이 될 수 있는 거리 조합이 있는지도 찾아봐줘").**
  G5/G6/G7 은 전부 임계를 "소스 카메라 **자신의** 여유 × β(<1)" 로 잡아 소스가 정의상 자기
  게이트를 통과하는데, **G1 만 `clear_frac·S` 라는 씬 상수의 절대 분수**였다. 실측에서 parkour
  소스 카메라가 배포 `clear_frac=0.10` 에서 1/49 프레임을 자기 G1 로 위반하고 goat 0.1199 /
  hike 0.1029 로 아슬아슬하다 — 소스를 기각하는 임계는 충돌 판정이 아니라 버그다 (D47 원칙).
  · **`gates.behind_surface_frames(..., min_zcam_frac=0.0)`** — 플랜 위치 `p` 가 소스 카메라 `t`
    의 광학 중심에 겹치면 `uv=(K·cam)[:2]/cam[2]` 가 0 에 가까운 수로 나뉘어 **투영 픽셀이 기하가
    아니라 반올림으로 정해지고**, `cam[2]≈0` 이면 판정식이 `clear > z_surf + margin` 으로 붕괴해
    p 가 어디 있든 같은 답이 나온다 (플랜이 아니라 소스 카메라 자기 주변에 대한 진술). 하한
    아래 소스 프레임은 **증언에서 뺀다** — 기각도 통과도 아니고, 남은 프레임들이 판정한다.
    실측(parkour `hole_bank_f7_on`, 664변이 × 49프레임 = 142,952 판정쌍): G1 히트 9,888쌍의
    `z_cam` min 0.0002 / p10 0.0041 / p50 0.0374 u. 하한 `0.005·S` 로 히트의 12.2%,
    `0.02·S` 로 31.9% 가 사라진다 — 씬 하나의 edge case 가 아니라 계통 오차다. 순수 lateral
    truck 은 자기 시각의 소스 카메라 대비 `z_cam≈0` 이라 **정상 플랜도** 여기 걸린다.
    `behind_profile` 로도 그대로 내려간다. **기본 0.0 → `1e-6` 분기라 예전과 비트 동일.**
  · **`sample_camera_bank.source_g1_clear(behind, pct=10.0)`** — 소스 카메라 자신의 G1 여유
    `min_t (z_surf − z_cam)/S` 의 `pct` 분위(u). 판정에 쓰는 것과 **같은** depth/K/c2w/sky/
    dynamic/frames/`min_zcam_frac` 을 쓴다 (다른 재료로 바닥을 재면 β 가 뜻을 잃는다).
    유도식 `clear_frac = margin_frac + β · source_g1_clear`.
  · **min 이 아니라 p10 인 이유** — min 이 위 degenerate 투영으로 오염돼 있었다. 하한을 켜도
    분위수 쪽이 안정적이라 분위수로 간다 (CV 근거는 아래 probe 항목).
  · `sample_camera_bank.py --behind_min_zcam`(기본 0.02) / `fit_hole_ladder.py
    --behind_clear_src_pct`(기본 10.0) 로 노출. `behind_context()` 반환 dict 와
    `fit_hole_ladder` manifest(`min_zcam_frac` / `clear_src_pct` / `source_g1_clear`)에 기록된다.
  · **아직 뱅크에 적용 안 했다.** parkour `hole_bank_f7_on` 대조(`/tmp/check_d123b.py`)에서
    664변이 중 107변이가 판정이 바뀌고 위반 플랜프레임 합계가 63% 준다. 재굽기는 사용자 판단 대기.
  · **[2026-09-04 정정 — 그 107 / 63% 는 ①+② 합산이고, 공은 거의 전부 ① 이다.]**
    두 손잡이를 분리 실측했다 (`out/parkour/hole_bank_g1probe_{off,on}`, 옵션 A, 렌더는 fit 내부
    verify 뿐). 두 arm 은 **현재 코드 그대로** 돌리고 `--behind_clear_src_ratio` 하나만 다르다 —
    d121 을 대조군으로 쓰면 09-03 21:49 이후의 코드 변경과 섞여서 귀속이 안 된다 (실제로 d121 →
    현재 off 만으로 91변이가 움직인다).
    - **① `min_zcam` 단독** (d121 → off; 둘 다 `clear 0.10` 이라 같은 자로 잰 값):
      91/664 변이 판정 변화, `binding=collision` **236 → 193 (−43)**, 위반 플랜프레임
      **4,839 → 1,980 (−59.1%)** (f7_on 기준 4,547 → 1,980, −56.5% — 원래 "63%" 가 이것이다).
      사라진 게 전부 degenerate 투영 히트다. `hole` +27 / `approach` +16 로 가려져 있던 진짜
      제약이 드러난다.
    - **② `clear_src_ratio` 단독** (off → on): parkour `g1_src(p10)=1.2218 u` →
      `clear_frac = 0.02 + 0.3×1.2218 = 0.3865·S`, 절대 0.10 의 **3.9배로 조여진다**.
      28/664 변이만 바뀌는데 **28건 전부 `push_in_arc_right`(24) / `track_push_in_arc_right`(4)**
      이고 anchor 7종 × 사다리 4단으로 고르게 퍼져 있다. `d_knob` 은 28건 전부 음수(최대 −1.051).
      `binding=collision` 193 → 208 (+15), `obb_limited` 12 → 0 (OBB 가 나아진 게 아니라 G1 이
      먼저 잡아 binding 을 뺏은 것). **`path_len_u < 0.02` 사실상 정지 궤적 198 → 222 (+24).**
    - **②의 동기가 ①의 버그였을 가능성.** ②를 정당화한 18편 legality 실측은
      `scripts/probe_g1_reference.py` 인데 **이 스크립트에 `min_zcam` 이 없다**(grep 0건).
      즉 "parkour 소스가 `clear 0.10` 에서 위반"의 근거인 `g1_min` 은 ①이 고친 바로 그 degenerate
      투영으로 오염된 값이다 — 소스 pose 를 가까운 소스 프레임에 투영하면 `cam[2]≈0` 이 되는
      정확히 그 축이다. ① 적용 후 parkour 의 `g1_p10` 은 1.2218 로 임계 0.10 의 12배다.
      **①을 고친 뒤에도 ②가 필요한지는 아직 측정되지 않았다.**
    - **② 는 legality 논증이 정당화하는 범위를 넘어선다.** 그 논증은 임계의 *바닥*("소스를
      기각하지 말 것")만 말하는데 코드는 그 바닥을 *임계 자체*로 쓴다. `source_g1_clear` 가 재는
      건 "소스 앞이 얼마나 트여 있나"라서, 트인 야외 씬일수록 임계가 커진다 — 실측 두 씬 모두
      느슨해지는 게 아니라 조여졌다 (parkour 3.9× / TRUMANS `g1_src(p10)=0.6295` → 0.2070, 2.1×).
      바닥으로만 쓰려면 `clear_frac = min(0.10, margin + β·g1_src)` 형태가 후보다.
    - 판단 보류. ② 채택 여부는 `probe_g1_reference.py` 에 `min_zcam` 을 넣고 52편 `g1_min` 을
      다시 재서 "소스를 기각하는 편이 실제로 남아 있나"를 확인한 뒤에 정한다.

- **`scripts/diff_bank_variants.py` — 두 뱅크의 결정열을 변이 단위로 대조 (2026-09-04).**
  `bank.json` 만 읽고 렌더 0회. `variant_id` 로 조인해 `status`/`binding`/`knob` 이 바뀐 변이를
  세고, status·binding 분포표 · knob/path_len_u/tau_max/behind_frac 요약 · `path_len_u` 임계별
  정지 궤적 수 · `|d_knob|` 큰 순 뒤집힘 목록을 낸다. 게이트 임계를 하나 바꿨을 때 "몇 개가
  실제로 뒤집혔나"를 눈대중 대신 세기 위한 것. 한쪽에만 있는 `variant_id` 는 따로 세서 **뱅크가
  다른 설정으로 구워진 경우를 임계 효과로 오독하지 않게** 한다. `--out_json` 으로 전량 diff 저장.

- **`scripts/probe_g1_reference.py` — G1 임계의 기준 거리 탐색 (2026-09-04, 사용자 지시 "소스에서
  카메라 거리든 최소 scene-camera 거리든 피사체 shot scale과 피사체까지의 거리든 일관적으로
  기준이 될 수 있는 거리 조합이 있는지도 찾아봐줘").** 후보 기준거리 R 로 나눈 비율 `g1_src / R`
  이 **씬 사이에서 안 흔들리면** 그 R 이 게이지다 — 변동계수(CV)로 순위를 매긴다 (D51 과 같은 판정).
  · `g1_src` 는 3D 최단거리(`render.standoff`)가 아니라 **z-depth 차** `z_surf − z_cam` 으로 잰다.
    G1 이 재는 게 그거라서다. 두 양은 실제로 다르다 — parkour 3D 최근접 0.0326 u vs G1 이 읽는
    0.0674 u. `time_match=True` 배포 설정과 맞추려고 **static 채널**로 잰다 (동적 채널은 소스
    pose 에서 항상 공집합 — 플랜 f ↔ 소스 f 는 `z_cam=0` 이라 스킵된다).
  · **실측 결론 (18편): 외부 게이지는 없다.** 소스 카메라 거리 · 최소 scene-camera 거리 ·
    피사체 shot scale · 피사체까지 거리와 그 곱/기하평균 조합의 CV 가 전부 0.89~2.10 이라
    기준이 못 된다. 유일하게 안정적인 건 **G1 자기 분포**였다 — `g1_p10` CV 0.371 /
    `g1_2nd` 0.396 / `g1_p25` 0.418 vs `g1_p50` 0.534. p10 이 최저라 위 기본값이 됐다.

- **F7 시간축 절단 — `fit_hole_ladder.py --time_truncate` / `--min_move_frac` + `hold_from` 열
  (2026-09-04, 사용자 지시 "시간축 절단 한거 안한거 돌려서 영상으로 비교해줘").**
  물리 게이트(G1/G5/G6/G7)에 걸리면 지금까지는 궤적 **크기**만 줄였다. F7 은 그 전에
  **시간축**으로 잘라 본다 — 게이트에 걸리기 직전 프레임까지만 움직이고 그 뒤로는 **위치만**
  얼린다(회전=조준은 계속되므로 "밀고 들어가다 멈춰서 계속 따라본다"로 읽힌다).
  네 게이트가 전부 `poses[:, :3, 3]` 만 보기 때문에 위치만 얼려도 판정이 닫힌다.
  · `truncate_hold(poses, hold_from)` / `solve_hold_from(...)` — 게이트를 통과하는 **가장 늦은**
    hold 시작 프레임을 이분법으로 찾는다. 판정은 `geometry_stats` 라 **렌더 0회**.
    `hold_from >= len(poses)` 면 입력을 그대로(사본조차 안 만들고) 돌려준다.
  · `retime_info(...)` — 절단 후 **위치에서 나오는 두 값**만 다시 잰다: `path_len_u`(`emit_bank
    --min_path_len` 이 읽는 필터라 안 고치면 사실상 정지가 된 궤적이 "길이 0.4u" 를 달고 통과)와
    `tau.tau_max_final`. 공식은 `decode/build_poses.py:889/:924` 와 문자 동일.
  · `emit_bank.py rebuild()` 가 행의 `hold_from` 을 그대로 되풀어 `poses.npz` 대조 assert 를
    통과시키고, 같은 `retime_info` 를 불러 canonical meta 와 bank CSV 가 어긋나지 않게 한다.
  · **기본 off** (`--no_time_truncate` 가 기존 경로). `SHAPE_DEFAULTS["time_truncate"]=False`
    폴백 + 빈 `hold_from` 칸 → F7 이전 뱅크는 비트 단위로 그대로 재현된다.

- **`scripts/attribute_g1_hits.py` — G1(behind-surface) 히트를 "무슨 표면이냐"로 귀속 (2026-09-04,
  사용자 질문 "어떤 물체랑 거리가 부족하다는거야?").** `viz_g1_collision.py` 는 어느 (플랜 프레임
  × 소스 프레임) 쌍이 걸렸는지만 그려주고 대상은 안 알려준다. 두 축으로 귀속한다:
  ① **seg instance** — 5×5 패치에서 실제로 최소 depth 를 준 픽셀을 되짚어(`hit_pixel`) 그 픽셀을
  덮는 dyn/stat 인스턴스를 찾는다. 없으면 `unsegmented`.
  ② **기하** — 표면점을 world→G 로 올려 `ground.ground_z` 와 비교해 `ground`/`below_cam`/`level`.
  `judge_frame` 은 `viz_g1_collision` 에서 import 한다 — G1 식이 세 군데로 갈라지면 안 된다.
  · 실측(parkour `dyn_0__dolly_in__hole0.5`, `hole_bank_k6_d121`, `--behind_src_frames 49`):
    pierce **0건**, tight 51건. 51건 전부 `unsegmented` + `level` — 주자가 따라 이동하는 콘크리트
    난간이고 노드가 아니다. `(z_cam − z_surf)/S` p50 −0.0674u 로 **관통이 아니라 요구 여유
    0.10u 미달**이며, `clear_frac` 를 0.05 로 낮추면 히트가 0% 가 된다. 51건 중 48건이 소스
    frame0 하나에 대한 것이고 히트 픽셀 median (637,361) ≈ 주점 — 플랜 카메라가 frame0 에서
    0.012u 밖에 안 움직였기 때문이다. **소스 카메라 자신도 자기 G1 을 1/49 프레임에서 위반**한다.

- **floor 마스크로 지면 높이를 재는 경로 (D62 재조준, 2026-09-04, 사용자 지시 "권장대로 해줘").
  전부 플래그 기본 off — 기존 산출값 동일.**
  · `scripts/extract_static_nouns.py`: `select()` 가 `(kept, surfaces, dropped)` 를 반환하고
    `--max_surface_nouns`(기본 **0**)만큼 광역 표면 명사를 `surface` 열로 따로 뺀다. 0 이면
    `static` 이 예전과 문자 동일(검증: 같은 후보에 대해 `kept` 두 경우 동일).
  · `scripts/sam3_static_instances.py`: `--nouns_field {static,surface}`(기본 `static`).
    옛 JSON 엔 `surface` 열이 없으므로 `.get(field, [])` 로 읽어 KeyError 대신 skip 으로 떨어뜨린다.
    `surface` 로 돌릴 땐 `--output_root .../seg_instances_floor` 로 따로 뺀다.
  · `scripts/build_scene_graph.py`: `floor_ground_z()` + `--floor_seg_root` / `--floor_quantile`
    (기본 None / 0.5). floor 픽셀만 올려 median z 를 `build_relations(ground_z_override=...)`
    로 넘긴다. `--ground_source gt` 가 있으면 GT 가 이긴다. `ground.source` /
    `ground.pointcloud_ground_z` / `ground.delta_to_pointcloud_u` / `ground.floor_points` 를
    그래프에 남긴다 — 지면 소스가 조용히 바뀌면 안 된다.
  · **검증(camel 재빌드, `--floor_seg_root` 를 없는 경로로)**: top-level 13개 키
    (`nodes`/`edges`/`gravity`/`scale`/…) 가 기존 `out/camel/scene_graph.json` 과 문자 동일.
    `ground` 만 `source: "pointcloud"` 가 **추가**됐고 `ground_z` 는 −0.0825 로 그대로다.
  · **OBB 노드로는 안 올린다**: 바닥 상자는 씬 전체를 덮어 `near` 엣지가 전량 걸리고 G5
    clearance 가 모든 노드의 MIN 이라 후보가 전멸한다. 마스크의 용도는 스칼라 하나뿐이다.
  · **원래 권장(ground RANSAC 제약)을 버린 이유**: D98 이후 `--gravity_source auto` 가 기본이고
    GeoCalib 사이드카가 RANSAC 을 대체한다. D122 실측 **280/280 scene 이
    `gravity.method == "geocalib"`** — 중력축 쪽 RANSAC 은 죽은 경로다. 남은 소비자는
    `ground.ground_z` 하나이고, 그 품질 실측(280 scene, 최대 동적 노드 `z_lo` vs `ground_z`):
    발밑 gap p50 0.120u / p90 0.443u, `gap > 0.5*height` **48.6%**, `gap < 0` **8.2%**.
    (단서: `dyn z_lo` 가 항상 발은 아니다 — 나는 물체·단 위 피사체는 정당하게 뜬다.)
  · **아직 안 돌린 것**: SAM3 `surface` 실행과 그래프 재빌드. 재빌드는 D122 뱅크를 통째로
    무효화하므로 사용자 승인 후에.
- **`scripts/viz_f1_inplace.py` — F1(제자리형 subject 강등) 승인용 영상 (D127c, 2026-09-05,
  사용자 지시 "F1은 어떤 물체인지 예시를 영상 3개정도 보여줘").**
  숫자만으로는 `center_drift_u 0.0125` 가 "정지"인지 판단할 수 없고, `path_len_u` 는 제자리
  흔들림과 실제 보행을 **못 가른다** (제자리 예시 0.2587 vs 보행 예시 2.1419 사이에 겹치는
  구간이 있다). 그래서 왼쪽에 소스 + OBB + track center 투영 궤적, 오른쪽에 조감도 2단
  (위 = 전 클립 **공통** 3.0u 눈금, 아래 = 클립별 확대)을 붙이고 두 칸 모두에 `drift 0.05u`
  임계 원을 그린다. 색 튜플은 **RGB** 다 — cv2 습관대로 BGR 로 적었더니 빨강이 파랗게 나왔다.
  dynpose 실측 4편: `14325a2c dyn_0 man` drift 0.0125 / `0af9210e dyn_0 man` 0.0281 /
  `0ebf0def dyn_0 woman` 0.0139 (셋 다 강등), 대조군 `0e9027bf dyn_0 woman` 2.0818 (유지).
  **F1 자체는 아직 미적용** — `fix.md` 상태 "제안됨, 미승인" 그대로다.

### Changed
- **작업 규약 4종을 `CLAUDE.md` 에 명문화하고 `/tmp` 사용을 중단한다 (2026-09-06, 사용자 지시).**
  넷 다 "조용히 새는" 종류라 규약으로 못 박아야 하는 것들이다 —
  ① **실행은 Claude 가 끝까지** 한다. 사용자에게 명령어를 넘기지 않고, 넘겨야 할 형태면 계획을 바꾼다.
  ② **임시 파일은 `/tmp` 가 아니라 `<repo>/tmp/<작업>/`**. `/tmp` 는 시스템이 임의로 비울 수 있어
     scene 목록·로그가 **없어진 줄도 모르고** 사라진다. `.gitignore` 에 `/tmp/` 추가.
  ③ **bash 지양, python 우선**. bash 드라이버는 "python 하나를 인자만 바꿔 부르는 래퍼"가 되기 쉽고
     실험(dNN)마다 복붙되어 몇 줄만 다른 파일이 쌓인다. 다단계 파이프라인은 python 드라이버 +
     `--stage` 로, 실험별 차이는 새 스크립트가 아니라 설정으로 표현한다.
  ④ **산출물은 압축적으로**. 로그 전량 tee 금지(학습 수치의 원본은 wandb), 세대별 뱅크는 최신 +
     실제 대조에 쓰는 것만 유지, 중간 산출물 삭제는 **목록·근거 보고 후 승인받고** 한다.
- **`render_bank_videos.py` 타일 라벨에 **카메라 출처**와 **target 이름**을 박는다 —
  `--pose_kind` + 자막의 `anchor_label` (2026-09-05, 사용자 지시 "target이 뭔지, GT인지 pred인지
  적어줘야지").**
  릴을 여러 개 늘어놓으면 어느 게 뱅크 pseudo-GT 고 어느 게 모델 예측인지, 그리고 이 카메라가
  무엇을 겨냥한 것인지 **영상만 보고 못 가른다**. 자막에 있던 건 `dyn_0`/`stat_4` 같은
  anchor id 뿐이었다.
  · 제목: `[GT] dolly_in_look_at` / `[EXT] ...` / `[pred] ...`. `--pose_kind auto`(기본)는 뱅크
    `poses.npz` 면 `GT`, `--poses_npz` 로 외부 파일을 물리면 출처를 모르니 `EXT`. 모델 추론이면
    `--pose_kind pred` 로 명시한다. `--pose_kind ""` 면 태그가 안 붙어 예전 영상과 라벨이 같다.
  · 자막: `dyn_0=camel` 처럼 `anchor_id=anchor_label[:16]`. `anchor_label` 열이 없는 옛 뱅크는
    `row.get` 이 빈 문자열을 돌려주므로 자막이 예전 그대로다.
- **`route_presets.py:169` advance 슬롯을 `dolly_in` → `dolly_in_look_at` (D134, 2026-09-05,
  사용자 승인).** `dolly_in` 은 `aim="free"` 라 전진하는 동안 subject 를 **다시 안 본다** —
  다가갈수록 대상이 화각 안에서 커지며 가장자리로 밀리는데 카메라가 가만히 있어서, 전진 preset
  이 정작 "대상에게 다가간다"는 자기 캡션 문구를 못 지킨다. `dolly_in_look_at` 은 같은 궤적에
  매 프레임 재조준만 붙인 것이다 (`lbm/presets.py:148,150` — traj lambda 가 글자 그대로 같고
  `aim` 만 `free`/`look_at`) 라서 이동량·τ 사다리·hole 예산은 그대로다.
  · recede 는 **안 바꾼다**. 물러나는 동안은 대상이 화면 중앙에 남아 aim 없이도 안 놓치고,
    한 슬롯만 바꿔야 같은 씬 안에 조준/비조준 두 어휘가 공존한다 (D82 논리와 같다).
  · `track_dolly_in_look_at` 도 `PRESETS` 에 실재하므로 `tp()` 의 `track_` 승격이 그대로 걸린다
    (실측: `track_mode=add` 에서 advance 가 bonus 로 뽑히면 `track_dolly_in_look_at`).
    캡션 문구도 네 이름 전부 `configs/caption_presets.json` 에 있다.
  · **이미 구운 뱅크에는 소급 안 된다** — d129(dynpose) / d128(vista) / d132(TRUMANS) 는 옛
    라우팅으로 구워졌다. 다음 재굽기부터 적용된다.
- **`route_presets.py --num_external` 기본값 4 → 1 (2026-09-05, 사용자 지시 "1로 해줘").**
  fit 예산이 DataDoP(`dd_*`) 로 새고 있었다. D129 126편 실측: `dd_*` 가 뱅크 21,892행 중
  **13,268행(60.6%)** 인데, 최종 학습 리스트는 `dd10` 서브샘플이라 코퍼스에서 `dd_*` 는 10% 다.
  그 10% 의 **절대 개수를 정하는 건 비-dd 행 수**이므로
  (`round(N_nondd·0.1/0.9)`, `camera_generation/latentcam/scripts/data/filter_seg_list_by_preset.py`
  의 `subsample_to_frac`), dd 를 더 구워도 **쓰는 양은 안 늘고 버리는 양만 는다** — 280편
  외삽으로 필요량 약 920행 대 굽는 양 약 10,700행, 12배 과잉.
  · 비용이 4배가 아니라 그 이상 붙는 이유: `pick_external` 은 **anchor 마다** `num` 개를 뽑는데
    `sample_camera_bank` 는 preset 을 **합집합**으로 받아 (anchor × preset) 격자를 전부 돈다.
    anchor 6개 씬이면 dd preset 24종 × anchor 6 × hole 사다리 4단이 통째로 fit 에 들어간다.
  · 1 로 내려도 다양성은 안 죽는다 — `pick_external` 씨앗이 `(video, node_id)` 라 anchor·씬마다
    다른 shape 을 뽑고 라벨 47종 커버리지는 코퍼스 전체에서 유지된다. 남는 dd 도 여전히 dd10
    필요량의 약 3배라 stride 서브샘플이 고를 여지가 있다.
  · **진행 중인 D129 에는 영향이 없다.** 옛 굽기 스크립트(`run_dynpose_d{107,122,129}_shard.sh`,
    `probe_dynpose_scale_mode.sh`)는 `--num_external 4` 를 **명시**로 넘긴다 — 그 run 이 실제로
    무엇으로 돌았는지의 기록이라 안 건드렸다.
- **`run_preset_warp_max_shard.sh` 에 `ROOT`/`EVAL` 환경변수 override 를 열었다 (2026-09-05).**
  `out` 이 4군데(리스트 glob / variant 고르는 인라인 python / `render_bank_videos.py` 인자 /
  릴 복사 경로)에 하드코딩돼 있어 dynpose·trumans 뱅크는 같은 릴로 못 봤다.
  **둘 다 안 주면 붙는 인자가 없어 예전 커맨드와 문자 그대로 같다** (`ROOT=out`, `--eval_data`
  미전달). dynpose 는 `ROOT=out_dynpose EVAL=DynPose-LBM`.
- **`run_preset_warp_sample.sh` 가 `ROOT`/`EVAL` 을 안쪽 러너에 실제로 넘기게 고쳤다
  (2026-09-05).** 드라이버가 `ROOT="${4:-out}"` 로 **assign 만** 하고 export 를 안 해서
  `run_preset_warp_max_shard.sh` 는 자기 기본값 `out` 을 봤다 — dynpose 뱅크를 주면 릴이 전부
  `EMPTY`(solved 0) 로 조용히 떨어진다. `VIDEOS=... ROOT="$ROOT" EVAL="$EVAL"` 로 env 전달.
  `EVAL` 을 안 주면 `output_root` 로 추정한다 (`out_dynpose` → `/data1/.../DynPose-LBM`,
  그 외 → 미전달=렌더러 기본). 추정을 넣은 이유: `--output_root` 만 맞고 `--eval_data` 가
  기본(Vista4D-Eval-Data)이면 렌더러가 **다른 소스 영상으로 warp** 하는데 rc=0 으로 끝난다.
- **`--tau_ref` 기본값을 `auto` → `follow` 로 뒤집었다 (`sample_camera_bank.py` +
  `fit_hole_ladder.py`) (D128, 2026-09-05, 사용자 지시 "tau_ref도 적용해서").**
  `auto` 는 `track_*` 만 follow 기준이라 나머지 preset 의 τ 는 소스 카메라 기준으로 재였다.
  D125 parkour 프로브 실측: `auto` → `follow` 로 stat_2/3/6 세 클립의 G1 위반 프레임 비율
  (`behind`)이 0.347/0.286/0.163 → **전부 0.000**, 대신 `path_len_u` 가 0.254→0.159,
  0.334→0.143 로 줄었다 (stat_6 은 0.346→0.349 로 유지). **정확성을 사고 다양성을 팔았다.**
  기존 뱅크(d115~d121)는 전부 `auto` 로 구워졌으므로 재현하려면 `--tau_ref auto` 를 명시할 것.
- **F1(제자리형 subject 를 `moving=False` 로 in-place 강등)은 구현했다가 기각했다
  (D128, 2026-09-05, 사용자 판단).**
  `build_scene_graph.py` 는 d115 와 **비트 동일**로 되돌렸고 `scripts/patch_moving_flag.py` 는
  삭제했다. 기각 사유: `moving` 은 `scene_graph/schema.py:207` 의 **anchor split key** 이기도
  하다 (`limit = max_dynamic if moving else max_static`). 여기서 강등하면 그 dyn 노드가
  `max_static` 버킷으로 넘어가 벽·바닥과 3칸을 다툰다 — 실측으로 `swing` 은 dyn 6개가 전부
  넘어가 **dynamic anchor 가 0** 이 된다. 제자리에서 움직이는 물체도 dyn 인 건 맞으므로
  `moving` 은 그대로 두고, 순변위 판정은 위 `--track_min_drift_u` 로 라우팅에만 건다.
  (측정치 `center_drift_u` 는 D127c 대로 노드에 계속 기록된다 — 숨기면 진단이 안 된다.)
- **`tracking="drift"` 를 어휘에서 삭제 + 조준 preset 전부 `lock` (D127, 2026-09-04, 사용자 지시
  "정지한 물체가 target일 때는 track이 필요없고 ... look_at은 drift 없애고 lock을 하는게 맞아 ...
  물체가 dynamic이라면 track + object centric이 돌아가야하고 이 때도 look_at은 lock이 맞아
  (track + free-moving일 경우는 예외)" + "drift 옵션은 아예 지워줘").**
  · `decode/build_poses.py:TRACKING_GAIN` 이 `{"world":0.0, "lock":1.0}` 이 됐다 (`drift` 0.6 삭제).
  · `lbm/presets.py:PRESET_TRACKING` 의 판정 기준이 이름(`track_`)에서 **`aim=="look_at"`** 으로
    바뀌었다 — 43 preset 중 22개(비-track 12 + track 10). `aim="free"` 20종은 `tracking_ignored`
    라 원래 조준 자체가 없다 (= 사용자가 말한 free-moving 예외).
  · `lbm/presets.py:LEGACY_TRACKING = {"drift": "lock"}` — 디스크의 옛 뱅크 행이 들고 있는
    `tracking:"drift"` 를 읽는 시점에 승격시킨다. `--no_preset_tracking` 으로도 못 되살린다
    (그게 "아예 지운다"의 뜻). **옛 뱅크의 drift 궤적은 재현 불가**가 됐다.
  · 기본값 이동: `fit_hole_ladder.py --tracking` / `lbm/loop.py --tracking` /
    `build_decision_fallback.py --tracking` = `lock`, `sample_camera_bank.py --trackings` =
    `["lock"]`. 넷 다 `choices=["world","lock"]` 로 막았다.
  · **실측** (`results/20260904_tracking_drift_vs_lock/`, parkour 7 preset × 172행, 두 arm 은
    `--tracking` 하나만 다르다): 동적 anchor `dyn_0` 28행 `subject_in_frame` 0.9670 → **1.0000**
    (`s_curve` 0.7690 → 1.0000, `subject_area_med` 0.0422 → 0.0583), `tau_max` 0.8761 → 0.9625,
    `hole_fraction` 0.1659 → 0.1987. 정지 anchor `stat_*` 144행 `subject_in_frame`
    0.9637 → 0.9637 (**Δ 정확히 0.0000** — 변위가 0 이라 정의상 no-op),
    `view_angle_max_deg` 13.5994 → 13.2909, `hole` 0.3068 → 0.2937. 172행 중 상태가 바뀐 건 11행
    (6.4%), `track_orbit_right` 는 비트 동일 (D93 이 이미 lock).
  · **전량 실측** (같은 두 arm 을 parkour **664 변이 전량**으로, 2026-09-05 완료,
    `results/20260904_tracking_drift_vs_lock/diff_trkall_664.json`): 바뀐 변이 19/664 (2.9%),
    status 6 · binding 4. 동적 anchor `dyn_0` 148행 `subject_in_frame` 0.7927 → **0.8254**,
    `subject_area_med` 0.0264 → 0.0280, `hole` 0.1800 → 0.1828, `tau_max` 1.0468 → 1.0632.
    정지 anchor 516행 `subject_in_frame` 0.8233 → 0.8233 (**Δ 0.0000**), `hole` 0.2598 → 0.2532,
    `view_angle_max_deg` 11.0924 → 11.0136. 전체 664행 `subject_in_frame` 0.8165 → 0.8238.
    `dyn_0` 에서 `subject_in_frame` 이 오른 preset 은 4개뿐이고 전부 `aim=look_at` 이다 —
    `orbit_left` 0.6342 → 1.0000 (area 0.0404 → 0.0968), `orbit_left_pedestal_up` 0.5380 → 1.0000,
    `s_curve` 0.7690 → 1.0000, `push_in_arc_left` 0.8460 → 1.0000. `aim=free` 계열
    (`dolly_in`/`pan_*`/`tilt_*`/`truck_*`/`static_hold`/`track_*`)은 전부 무변화 — 설계대로다.
  · drift 를 지운 이유: 0.6 은 필터가 아니라 **영구 편향**이라 조준점이 끝까지 subject 변위의 40%
    만큼 뒤처진다. 원래 목적이던 조준 jitter 는 `track.center_smooth`(savgol w=11 p=3) +
    `--aim_keyframes 6` + `--keyframe_ease smooth_kf` 가 올바른 축에서 처리한다. `--tracking` 이
    argparse 기본값으로만 존재해서 **한 번도 선택된 적이 없다**는 것도 확인했다.
- **anchor(=촬영 target) 선별에 표면 제외 + 편당 상한 3 (D127, 2026-09-04, 사용자 지시
  "static이 너무 많은데 ... main 최대 3개 정도만 해도 될 것 같은데" + "wall, fence, floor 같은건
  static target에 포함안되는거 맞지?").**
  · 새 `scene_graph/schema.py:{SURFACE_TOKENS, label_tokens, is_surface_node, anchor_sort_key,
    pick_main_anchors}` — 한 벌만 두고 `scripts/sample_camera_bank.py:anchor_nodes` 와
    `scripts/route_presets.py:pick_anchors` 가 **둘 다 이걸 부른다**.
  · `label_tokens` 는 camelCase 를 쪼갠다 — TRUMANS 라벨이 mesh object 이름이라 `WallInner.022`
    는 소문자 변환만으로는 `wallinner` 라 어떤 토큰에도 안 걸렸다.
  · **왜 새로 필요했나**: `extract_static_nouns.py:SURFACE_NOUNS` 는 SAM3 **keyword** 만 걸러서
    두 군데가 샜다. ① `fence`/`railing`/`window`/`door` 가 그 목록에 없다. ② TRUMANS 그래프는
    SAM3 를 아예 안 탄다.
  · **실측** (524편, `min_area_frac` 0.01):

    | 코퍼스 | 편수 | anchor before → after | 편당 | surface 로 빠짐 | 상한으로 빠짐 |
    |---|---|---|---|---|---|
    | `out` (Vista) | 53 | 357 → 151 | 6.7 → 2.85 | 38 | 168 |
    | `out_dynpose` | 280 | 1276 → 764 | 4.6 → 2.73 | 23 | 489 |
    | `out_trumans` | 191 | 2087 → 565 | 10.9 → 2.96 | **1150 (53%)** | 372 |

    동적 anchor 는 정렬 1순위라 하나도 안 빠졌다 (Vista 80 / dynpose 755 / TRUMANS 191 전량 유지).
    anchor 가 0 이 되는 편은 dynpose 2편뿐이고 **둘 다 변경 전에도 0** 이었다.
  · parkour: `dyn_0`(man) + `stat_1`/`stat_2`(building) 만 남고 `railing` ×3 · `fence` ×1 이
    `surface` 로, `tree` ×2 · `building` ×2 가 `max_anchors` 로 빠진다.
  · 되돌리는 스위치: `--max_dynamic_anchors 0` / `--max_static_anchors 0`(그 갈래 상한 없음) /
    `--no_anchor_drop_surfaces`. `--nodes` 로 명시한 목록은 예전처럼 필터를 안 탄다.
  · **아직 뱅크에 적용 안 했다** — 재굽기는 사용자 판단 대기.
- **anchor 상한을 동적/정적 **따로** 3개씩 (D127b, 2026-09-05, 사용자 지시 "dynamic target,
  static target 각각 최대 3개로 해서 가장 main이 되는 애들만 target으로 삼아줘").**
  · `pick_main_anchors(nodes, min_area_frac, max_dynamic=3, max_static=3, drop_surfaces=True,
    max_anchors=0)`. 두 소비자의 플래그 이름도 같다 — `sample_camera_bank.py` /
    `route_presets.py` 의 `--max_dynamic_anchors` · `--max_static_anchors`.
    `--max_anchors` 는 둘을 적용한 **뒤에** 거는 총합 상한으로 남기고 기본 `0`(끔)으로 내렸다.
  · **왜 합산 상한 3(D127 최초안)이 틀렸나**: 정렬이 동적 우선이라 동적이 3개인 편에서
    **정적 anchor 가 0** 이 된다. 그러면 그 편의 모든 변이가 `track_*`+동적 target 이 되고,
    반대로 동적이 없는 편은 정적만 3개다 — 코퍼스의 동적/정적 target 비율이 씬 구성에 끌려간다.
    실측: dynpose 280편에서 합산 3 일 때 정적 anchor 총 **6개**, 따로 세면 **59개**.
  · **실측** (524편, `min_area_frac` 0.01, 표면 제외 on):

    | 코퍼스 | 편수 | anchor before → after | 편당 | 동적 | 정적 | surface | max_dynamic | max_static |
    |---|---|---|---|---|---|---|---|---|
    | `out` (Vista) | 53 | 357 → **207** | 3.91 | 80 | 127 | 38 | 16 | 96 |
    | `out_dynpose` | 280 | 1276 → **814** | 2.91 | 755 | 59 | 23 | 438 | 1 |
    | `out_trumans` | 191 | 2087 → **705** | 3.69 | 191 | 514 | 1150 | 0 | 232 |

    합산 3 안(151/764/565) 대비 각각 +56 / +50 / +140. 편당은 여전히 2.9~3.9 다 — 동적이 3개
    넘는 편이 드물어서 상한 3+3=6 에 실제로 닿는 편이 거의 없다. TRUMANS 는 동적이 정확히
    편당 1개(사람)라 `max_dynamic` 탈락이 0 이다. anchor 0 인 편은 dynpose 2편(변경 전에도 0).
- **`track_truck_left/right` 를 targetless 로 승격할 수 있게 (D123, 2026-09-04, 사용자 지시
  "a로 적용해줘").** `configs/caption_presets.json` 의 두 preset 에 `phrase_targetless` 를 달고
  `scripts/build_bank_captions.py` 에 `--targetless_promote` / `--no_targetless_promote`
  (**기본 꺼짐**)를 추가했다. 켜면 `promote_targetless()` 가 `phrase_targetless` 를 가진 preset 의
  `phrase` 를 갈고 `targetless: true` 를 세워, 캡션이 `target`/`framing` 없는 한 절이 된다.
  · **왜**: 두 preset 은 `follow_gain 1.0` 으로 **위치만** 따라가고 `aim=free` 라 재조준을 안 한다.
    d122 뱅크 실측 `subject_in_frame` median 이 `track_truck_left` 0.462 (n=912) /
    `track_truck_right` 0.385 (n=696) 인데 문구는 `tracks alongside {target}` 이었다 —
    "계속 보인다"는 지키지 못할 약속. 같은 `aim=free` 인 `track_dolly_in/out` 은 1.000 이고
    `aim=look_at` 인 track_* 도 전부 1.000 이라, `track_` 접두사가 아니라 truck 둘만 문제다.
  · **기본을 안 바꾼 이유**: 켜면 옛 뱅크를 다시 export 했을 때 캡션이 조용히 달라진다.
    d121 두 arm 이 지금 옛 문자열로 학습 중이다. 켰는지는 캡션 JSON 헤더
    `targetless_promoted` 와 요약표에 남는다.
  · 실측 검증(1편 122변이): 바뀐 변이 8건, 전부 `track_truck_left`. 나머지 114건은 문자 동일.
  · 미승격으로 남긴 것: `truck_*`/`pedestal_*`/`dolly_in`/`dolly_out` (config `note_targetless`
    의 **미결** 항목 그대로 — D104 target 절 A/B 대기).

### Fixed
- **`scripts/run_dynpose_d129_export.sh` 1단계가 캡션 0 편을 **rc=0 으로** 굽던 것 (FIX-D129-a,
  2026-09-05).** `build_bank_captions.py --videos all` 은 영상 목록을 `--metadata_csv` 에서
  만드는데(`:687`) 기본값이 Vista4D 것(`:55 METADATA_DEFAULT`)이라, dynpose UUID 와 교집합이
  0 이 되어 `videos=[]` -> captions 0 으로 조용히 끝났다. 파일이 없는 게 아니라 **다른 코퍼스의
  목록**을 쓴 것이라 예외가 안 난다. d110 export 스크립트가 이 인자를 안 넘겼고 d129 가 그대로
  물려받았다 (`hole_bank_d110` 도 captions 0 으로 남아 있다).
  · 조치: 목록을 metadata 가 아니라 **뱅크 산출물에서 직접** 뽑아 명시적으로 넘긴다 —
    `VIDS=$(ls -d out_dynpose/*/$HOLE/bank.json | cut -d/ -f2)`. `--videos` 가 명시되면
    `:701` 이 `events.get(video, "")` 라 metadata 에 행이 없어도 KeyError 가 안 난다.
  · dynpose `metadata.csv` 는 `video,dynamic` 뿐이라 `prompt` 열이 없다 — 이 파일은 dynpose 에서
    영상 목록 말고는 아무 일도 안 한다 (event 가 전부 빈 문자열). 그래서 목록의 출처를 바꾸는
    것이 의미 손실 없이 안전하다.
  · 덤: dynpose metadata 를 넘겼어도 265/267 이었다. `00e9f728-…` / `015b197d-…` 두 편은
    metadata 에 행 자체가 없는데 recon 은 있다. 뱅크에서 세면 267 로 맞아 `NB == NC` 게이트가
    제 값을 한다. 수정 후 실측 **267/267 OK 전량 일치**.
- **`CloudRenderer` 를 프레임별 DA3 K 로 띄우던 나머지 6곳에 `--fixed_focal`(기본 **True**) 를
  달았다 (2026-09-05, 아래 릴 수정의 후속 감사).** 릴 버그를 고치면서 `CloudRenderer(` 18곳을
  전수 감사했더니 5곳은 `fixed_focal=True` 하드코딩, 7곳은 인자로 받고 있었는데 **6곳이 아무것도
  안 넘겨 기본 `False`(프레임별 K)로 돌고 있었다**. 릴만의 문제가 아니었다.
  · `verify.py` / `scripts/audit_bank_geometry.py` — **지표를 뱅크와 다른 카메라 모델로 재고
    있었다.** `hole_fraction` / `subject_in_frame` 이 궤적과 무관한 화각 떨림으로 흔들린다.
  · `scripts/viz_g1_collision.py` — `renderer.K_src` 를 그림뿐 아니라 **G1 판정에도** 쓰므로
    규약이 어긋나면 충돌 판정 자체가 달라진다.
  · `scripts/build_candidate_board.py` / `lbm/loop.py` — VLM 이 보는 board·before/after 타일에
    연산 효과와 화각 떨림이 섞인다. `LoopRunner` 는 `getattr(args, "fixed_focal", True)` 라
    이 인자가 없는 호출자(`run_lbm_lite.py`)도 그대로 돈다.
  · `scripts/probe_near_depth_repeat.py` — 이미 `bank.json` 을 읽고 있어서 플래그 대신
    **`bank["fixed_focal"]` 을 그대로 따라가게** 했다 (뱅크 수치 재현이 목적이라 K 가 다르면
    비교가 성립하지 않는다).
  기본값을 **True** 로 잡은 이유: 이 6곳은 전부 *새로 푼* pose(뱅크·결정·후보)를 렌더하므로
  pose 를 푼 K 와 같아야 한다. 프레임별 K 가 맞는 건 `lbm/render.py` 의 자기 일관성 검사
  하나뿐이고 거기만 기본 `False` 를 유지한다. 전부 `--no_fixed_focal` 로 예전 동작으로 되돌린다.
  검증: 6개 스크립트 `--help` rc=0, `CloudRenderer(fixed_focal=)` 실측 — snowboard `fx` 가
  False 에서 1179.684~1262.462(진폭 6.99%), True 에서 1184.992 상수(0.00%).
- **`run_preset_warp_max_shard.sh` 가 `bank.json` 의 `fixed_focal` 을 읽어 렌더에 반영한다
  (2026-09-05, 사용자 지적 "snowboard가 떨리는데").**
  릴이 `render_bank_videos.py --fixed_focal`(기본 **False**, `:319`)을 안 넘겨서, **뱅크는
  frame0 고정 K 로 구워졌는데 렌더만 프레임별 DA3 K** 를 썼다. 규약 불일치라 카메라 궤적과
  무관한 화각 떨림이 나온다. snowboard 실측(`scene_graph.json` 49프레임): `fx=fy` 가
  1184.99~1262.46 로 **진폭 6.99%**, 프레임간 최대 점프 28.55 px → 화면 가장자리가 median
  3.41 / **최대 14.94 px/frame** 움직인다 (`cx`/`cy` 는 상수라 떨림은 전적으로 focal).
  플래그를 하드코딩하지 않고 **뱅크 기록을 따르게** 했다 — `bank.json` top-level `fixed_focal`
  이 그 뱅크의 K 규약에 대한 유일한 근거이고(`fit_hole_ladder.py:1167-1169` 주석), 옛날
  프레임별 K 로 구운 뱅크는 예전 그대로 렌더된다. RENDER 로그에 `focal=` 을 찍는다.
  · **뱅크 pose 와 export 코퍼스는 영향 없다** — `vista4d_bank_to_dl3dv.py:131` 이 `K[0]` 을
    전 프레임에 repeat 한다. 영향은 릴(시각화)뿐이고 `results/20260903_vista_preset_warp_d121`
    과 `results/20260905_d128_preset_warp` 가 그 상태로 구워져 있다.
- **`run_k6_d128_shard.sh` FIT 에 `--behind_clear_src_ratio 0` 을 명시로 못박았다 (D128,
  2026-09-05).** 첫 실행 때 argparse 기본값(0.3)에 기대고 있었는데, 같이 켠
  `--behind_min_zcam 0.02` 와 만나면 소스 재투영이 전부 `z_floor` 아래로 떨어져
  `source_g1_clear` 가 **증거 0 으로 assert** 한다. 6편(avocado-slice / basketball-four /
  bed-shopping / bmx-bumps / breakdance / camel)이 FIT rc=1 로 날아갔다
  (`video_generation/FIX.log` 2026-09-05). ② `clear_src_ratio` 는 아직 채택 보류 축이라 값 0 이
  맞고, **D105 원칙(뱅크 정체성을 argparse 기본값에 맡기지 않는다)대로 명시**한다. 러너 상단
  주석에도 축 ⑥ 아래에 근거를 남겼다. 재-스윕 6편 전부 rc=0 → **52/52 canonical, 0 skipped**.
  `run_dynpose_d129_shard.sh` 는 처음부터 명시로 짜여 있다.

### Added
- **`scripts/run_dynpose_d122_shard.sh` — dynpose 코퍼스를 vista 방식으로 전량 재생성 (D122,
  2026-09-04, 사용자 지시 "vista 돌렸던 방식으로 다시 돌려 / 유일하게 다른건 preset,
  datadop 비율").** 위 프로브는 중단하고 게이지를 `points_first_cam` 으로 확정했다.
  러너는 vista `run_k6_d115_shard.sh`(GRAPH→CLOUD→TAU) + `run_k6_d121_shard.sh`(FIT→EMIT)
  체인을 그대로 옮기고, 사용자가 지정한 **두 축만** 다르다:
  · preset 라우팅 `route_presets.py --num_anchors 2 --num_external 4` (DataDoP 비율 정합),
  · `--external_shapes configs/datadop_shapes.json` 을 τ뱅크·fit 양쪽에 전달.
  그 외 인자(`--tau_ladder 1.00`, `--aim_keyframes 6 --keyframe_aim auto
  --keyframe_ease smooth_kf`, `--fixed_focal --deroll`, `--orbit_fixed_sweep
  --min_sweep_deg 20 --behind_src_frames 49`, `--collision_time_match --collision_source
  depth`, `--hole_ladder 0.10 0.20 0.35 0.50`, `--area_timeline --composition
  --composition_min_area 0.004 --composition_max_nodes 3`)는 vista 와 문자 단위로 같다.
  · **재fit 만으로는 안 되는 이유**: `S` 가 세 군데에 산다 — `scene_graph.json` `scale.S`,
    `cloud.npz` meta(`lbm/render.py:70` 이 렌더 단위로 읽음), `bank_*/bank.json` 최상위 `S`.
    그래서 graph·cloud 를 **제자리에서 다시 짓는다**(vista D115 가 한 것과 같은 거래).
    `scene_graph.json` 은 `scene_graph_pre_d122.json` 으로 백업하지만 `cloud.npz` 는
    백업하지 않는다(1.4 GB×280 = 380 GB) → d107/d110 **렌더** 재현성은 여기서 끝난다.
    뱅크 파일 자체는 안 건드리므로 학습에 쓴 카메라는 남는다.
  · `instance_desc.json` 266편은 재실행 불필요 — 게이지를 바꿔도 node id·label 이 불변
    (3편 실측 id_same=True, label_diff=0).
  · `.graph_d122` / `.cloud_d122` 마커로 재실행 시 중복 계산을 막는다.
- **`scripts/probe_dynpose_scale_mode.sh` — dynpose 게이지를 6편 실측으로 정한다 (D122,
  2026-09-04, 사용자 지시 "먼저 몇 편으로 차이를 실측").** D122 재fit 이 F2
  `assert_scale_mode` 로 전량 즉사했다 — dynpose scene_graph 280편은 `scale.mode` 필드
  **자체가 없어** 기본값 `frame0_ray` 로 읽히는데, vista 는 D115 재굽기 때 52/53 편이
  `points_first_cam` 으로 갱신됐다. 즉 d107/d110 dynpose 코퍼스는 전부 옛 게이지다.
  · 그냥 갈아탈 수 없는 이유: 게이트 임계가 전부 S 배율(`behind_margin_frac·S`,
    `obb_clear_floor·S`, 가림 `0.02·S`)이라 S 정의를 바꾸면 임계가 **씬마다 다른 배율로**
    옮겨간다. 실측 `r_pts_first = S_pts_first / S_f0` 가 p05 0.560 / p50 1.097 / p95 1.571 로
    1.0 **양옆에** 걸쳐 있어 "전부 느슨"도 "전부 빡셈"도 아니다.
  · **arm B 는 재fit 만으로 안 된다.** τ 뱅크(`bank_d107/bank.json`)가 `S` 를 통째로 싣고
    있고(`S: 0.4363…`) `sample_camera_bank.py:84` 도 `assert_scale_mode` 를 부른다 →
    graph → route → τ뱅크 → fit **전 체인**을 다시 돌려야 한다. 그래서 arm B 는 D107 러너를
    통째로 재현하되 `--scene_scale_mode points_first_cam` 만 얹고, 원본을 안 건드리려고
    `out_dynpose_probe/<V>/` 에서 돈다 (`cloud.npz` / `geocalib_gravity.json` 은 심링크).
  · arm A 는 옛 게이지 그대로 + `--allow_legacy_scale` → `hole_bank_probeA`.
    두 arm 의 fit 인자는 D122 세팅으로 동일. 대조 항목은 variant 수 · 사다리 도달단
    (`target_hole`) · `path_len_u` · `tau_max` · `status`/`binding` 분포.
- **`scripts/run_k6_d121_shard.sh` — vista 뱅크를 D121 세팅으로 재굽기 (52/52 완료,
  2026-09-04 00:12).** `hole_bank_k6_d121` = d116 사다리에 `--area_timeline --composition`
  두 열을 얹은 것(69열). 5샤드, `describe_instances_vlm.py` 52/52 선행.
  후속 릴은 `results/20260903_vista_preset_warp_d121` (preset별 최대단 depth warp, 4샤드).
- **`scripts/run_dynpose_d122_shard.sh` — dynpose 코퍼스를 D121 세팅으로 재생성 (D122,
  2026-09-03, 사용자 지시 "이 세팅으로 dynpose ... 돌려놔줘").** d110 뱅크(09-02 fit, 266/266)는
  **56열**이라 `subject_area_seq` / `in_frame_ids` / `enter_ids` / `exit_ids` 가 없다 (d121 vista 는
  69열). 이 4열이 없으면 D122 캡션의 세 절 — `"a medium shot **that widens to** an extreme wide
  shot"`(subject_area_seq) / `"... **until she leaves the frame**"`(seq 끝값 0) / `"with X also in
  frame / as Y **leaves the frame**"`(in_frame/enter/exit_ids) — 이 통째로 못 나온다. 두 열 다
  판정 패스가 **이미 렌더한** 프레임의 면적과 OBB 8꼭짓점 투영만 쓰므로 **추가 렌더 0회**지만
  CSV 열이라 재fit 없이는 못 채운다.
  · d110 대비 실효 diff 는 `--area_timeline --composition` 두 플래그뿐. 사다리
    (0.10/0.20/0.35/0.50) · graph(`.graph_d107`) · route(`preset_route_d107.json`) ·
    τ뱅크(`bank_d107`) · ease/deroll/fixed_focal 은 그대로다 → **공유 열은 d110 과 같아야 한다**
    (한 편 뽑아 대조할 것; d121 vista 가 d116 과 bit-identical 이어야 했던 것과 같은 규칙).
  · 기본값인 `--orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49
    --collision_time_match --collision_source depth` 도 **명시**한다 (뱅크 정체성을 argparse
    기본값에 맡기지 않는다는 D105 규칙).
  · 앞단으로 `describe_instances_vlm.py` 를 dynpose 265편에 처음 돌린다 —
    `out_dynpose/*/instance_desc.json` 이 **0건**이었다. 없으면 nl 캡션의 referring expression 이
    조용히 라벨로 떨어진다.
  · 후단은 기존 `run_dynpose_d110_export.sh` 체인 그대로 (nl 캡션 → `latentcam_dynpose_d122`
    export, holdout 은 d107 과 같은 27편 → `filter_seg_list_by_preset.py --target_frac 0.10
    --suffix dd10`).
- **캡션 자연어 재설계 + composition 열 (D121, 2026-09-03, 사용자 지시).** 학습 프롬프트가
  `target: camel. motion: the camera orbits to the left around the subject.` 였다. 세 가지가
  문제였다 — ① target 이 한 단어라 형제 노드를 못 가린다 ② `motion` 문구의 `{target}` 치환이
  **어느 문구에도 안 걸리는 no-op** 였다 (`configs/caption_presets.json` 이 리터럴
  `"the subject"` 를 들고 있었다 — 즉 라벨이 motion 문장에 도달한 적이 없다) ③ framing 이
  shot scale 한 단어뿐이라 "무엇이 같이 담기는지"가 없다. D117-a PE-AV 프로브에서 구조형이
  4형식 중 꼴찌였던 것도 같은 자리다.
  · `build_bank_captions.py --prompt_style nl`(**기본값**) 이 한 문장을 낸다:
    free-moving/targetless 는 `The camera [adv] [motion].`, object-centric 은
    `The camera [adv] [motion] around/toward [referring expr], keeping it in [shot scale]
    [composition].` around/toward 는 preset 문구가 이미 갖고 있어 여기서 다시 안 고른다.
    `--prompt_style fields` 가 예전 형식을 **문자 단위로** 그대로 만든다.
  · target 은 `instance_desc.json`(D120) 의 referring expression
    (`"the larger pale camel walking along the fence"`). 없으면 조용히 라벨로 떨어지되
    **몇 편이 그랬는지 요약표에 찍는다**. `--no_anchor_desc` 로 예전 경로.
  · `caption_presets.json` 35개 문구를 `"the subject"` → `{target}` 슬롯으로. 꽂을 말은
    **형식이** 정한다 (fields = 옛 문자열, nl = referring expr) — fields 에서 문구가 대상을
    또 부르면 `target:` 절과 중복이다. round-trip 으로 43 preset 전량 복원 확인.
  · 크기 부사가 3-state 가 됐다: 안 주면 nl 켬 / fields 끔. **`dd_*`(코퍼스 47.8%) 에도
    붙는다** — 예전에는 `--magnitude` 가 preset 경로에만 걸려 있어 두 부류가 부사 유무로
    갈렸다. 정지 preset(`axis=="static"`, `move==static`)에는 안 붙인다
    ("barely holds completely locked off" 는 문장이 아니다).
  · **composition 열** `in_frame_ids` / `enter_ids` / `exit_ids` —
    `sample_camera_bank.composition_stats()` 가 anchor 말고 어떤 노드가 화면에 담기는지 /
    들어오고 나가는지를 잰다. OBB 8꼭짓점을 `T_gw @ pose` 와 **수정 없는** `K_src[f]` 로
    투영한 bbox 면적비라 **렌더가 0회**다 (`geometry_stats` 와 같은 부류). 꼭짓점 절반이
    카메라 뒤면 0 으로 친다 (bbox 가 화면 전체로 번진다). 뱅크는 **노드 id 만** 싣는다 —
    문구는 `instance_desc.json` 이 갖고 있어야 뱅크를 다시 안 굽고 바꿀 수 있다.
    `fit_hole_ladder.py --composition`(기본 켬, `--composition_min_area 0.004`
    `--composition_max_nodes 3`) 이 **판정 패스에서만** 붙인다 (이분법 5프레임으로는
    앞/뒤 1/3 이 2프레임이라 enter/exit 이 무의미). `--no_composition` 이면 열이 빠져
    예전 뱅크와 같다.
  · camel 스모크(dyn_0 × {orbit_left, dolly_in_look_at}) 실측:
    orbit 은 `exit_ids=[dyn_1]` → *"...keeping it in a medium shot that widens to a medium
    wide shot as the smaller pale camel standing near the fence leaves the frame."*,
    dolly_in 은 `in_frame_ids=[stat_0, dyn_1, stat_1]` → *"...that tightens to a medium
    close-up shot with the larger wooden fence ... and the smaller pale camel ... also in
    frame."* 변화 절이 있으면 "같이 있음" 절은 뺀다 (붙박이라 거의 모든 변이에 같은 말).
  · `vista4d_bank_to_dl3dv.py` 의 `caption_fields` 가 `target_text`/`framing_nl`/
    `composition` 을 **있을 때만** 같이 내보낸다 (fields 코퍼스는 예전과 동일).
  · ⚠ 기존 d99/d115 뱅크에는 composition 열도 `subject_area_seq` 도 없다 — 두 문구는
    **재굽기 이후에만** 나온다. 캡션 스크립트가 그걸 세어서 경고로 찍는다.
- **`scripts/describe_instances_vlm.py` — 노드별 referring expression (D120, 2026-09-03).**
  캡션의 target 이 `normalize_label(anchor_label)` 한 단어(`"woman"`, `"camel"`, `"window"`)라
  같은 라벨 노드가 둘 이상이면 문장만으로 어느 쪽인지 못 가른다 (avocado-slice 는 `window` ×2 ·
  `table` ×2, camel 은 `camel` ×2). SAM3 인스턴스 마스크를 근거로 VLM 에게 **"the ..." 명사구
  하나**를 받아 `out/<video>/instance_desc.json` (`lbm_instance_desc_v1`) 으로 남긴다.
  · 노드당 이미지 **1장**: 시간 3등분 구간에서 bbox 면적 최대인 프레임 3개를 골라, 각 프레임을
    [위=마스크 밖을 0.28 배로 어둡게 한 전체 프레임 + 노란 bbox / 아래=원본 색 tight crop] 2단
    타일로 만들고 가로로 잇는다. VLM 호출도 노드당 1회.
  · 검증기: `"the "` 로 시작 · ≤12 단어 · 끝 마침표 없음 · base noun 포함 · `camera/image/frame/
    photo/video/yellow box/highlight` 금지 · **같은 라벨 형제가 있으면 맨 `"the {label}"` 보다
    길어야 통과**. 위반 시 `## VALIDATION_ERRORS` 로 최대 3회 재질의(temperature 안 올림).
  · 기본은 `bank.json.variants` 의 `anchor_id` 집합만 (`--all_nodes` 로 전 노드).
    실측 avocado-slice 6 / camel 5 노드, repair 2회, 노드당 0.3~0.7 s.
- **`--mesh_margin_frac` (기본 0.08) — mesh 충돌 판의 독립 임계 (D120, 2026-09-03).**
  depth 판과 mesh 판이 같은 `--behind_margin_frac` 를 쓰는 동안 **실효 여유가 4배 어긋나
  있었다**: depth 판은 `cam_z + clear > z + margin` 이라 실효 standoff 가
  `(behind_clear_frac − behind_margin_frac)·S` = 0.08·S 인데, mesh 판(`mesh_behind_profile`)은
  `clear_frac` 을 아예 안 받아서 `margin_frac·S` = 0.02·S 였다. `sample_camera_bank.py` /
  `fit_hole_ladder.py` 양쪽에 손잡이를 넣고 `behind_context → behind dict → _mesh_profile` 로
  실었다. `None`(=예전) 이면 `margin_frac` 으로 떨어져 **예전 뱅크와 비트 동일**이고, 재현은
  `--mesh_margin_frac 0.02`. D47 legality assert 도 새 손잡이 기준으로 잰다.
- **shot scale 시간축 — `subject_area_seq` 열 + `--framing_timeline` 캡션 (D119, 2026-09-03).**
  캡션의 shot scale 은 `bucket(subject_area_med, framing_buckets)` 한 줄에서 나오는데
  (`build_bank_captions.py:202`), 그 `subject_area_med` 는 `verify_frames`(기본 13) 프레임
  면적비를 `np.nanmedian` 으로 접은 **스칼라 하나**였다. push-in 과 pull-out 과 정지가 전부
  같은 문장이 된다.
  · `sample_camera_bank.measure_trajectory(area_timeline=True)` 가 접기 전 배열을 그대로
    싣는다 — `subject_area_seq` / `subject_area_start` / `subject_area_end`. **렌더가 안
    늘어난다** (이미 모으고 있던 `areas` 를 버리지 않는 것뿐).
  · `fit_hole_ladder.py --area_timeline`(기본 켬) 은 **판정 패스에서만** 붙인다 — 이분법
    5프레임으로는 push-in/pull-out 을 못 가른다. `--no_area_timeline` 이면 열이 빠지고 예전
    뱅크와 같다. `bank.csv` 는 배열 열을 `|` 로 잇는다 (따옴표 없는 writer 라 쉼표를 못 쓴다).
  · `build_bank_captions.py --framing_timeline`(기본 끔) 이 시작/끝 버킷이 다를 때만
    `"medium shot tightening to medium close-up"` 으로 바꾼다. 양 끝은 **앞/뒤 3개씩 median**
    으로 읽는다 — median 이 주던 이상치 보호가 끝점 한 프레임에는 없기 때문이다.
  · camel 65 변이 실측: `end/start` 가 0.5 배 넘게 벌어지는 변이 **34/65 (52.3%)**, 그중
    framing 문구가 실제로 바뀌는 것 **28/65 (43.1%)**. 1프레임 튐이 극값인 변이는 7/65 (10.8%)
    로, 끝 3개 median 이 그걸 걸러낸다. 방향은 preset 과 일치한다 (dolly_in → tightening).
- **`scripts/trumans_approach_viz.py` — 임계를 고정하고 카메라를 벽으로 전진시키는 시각화
  (D118, 2026-09-03).** 앞서 낸 `run_raycast_threshold_viz.sh` 는 궤적을 고정하고
  `--min_clearance` 를 스윕했는데, 사용자가 원한 건 그 반대였다 — "고정된 값을 가지고 천천히
  벽으로 이동했을 때 어디쯤에서 collision, clearance 에 걸리는지". 임계 4개를 전부 고정한 채
  40걸음 전진하며 각 임계가 **처음 걸리는 지점**에 색 기둥을 세운다. 기둥 사이 간격이 곧
  "게이트를 0.35→0.20 으로 낮추면 벽에 몇 cm 더 붙나"다 (a17 실측 16 cm).
  · 게이트 함수 `clearance_of()` 를 **그대로** 부르되 수평 4방향만의 `clearance_h` 를 같이
    기록한다. 두 값이 갈라지는 구간이 곧 바닥/책상 윗면이 게이트를 먹는 구간이다.
    a17 eye(1.33 m) 45.0% / low(0.90 m) 67.5% of steps.
  · `--start_height 0.90` 실측: 벽에서 2.116 m 떨어져 있는데도 `0.50` 이 **step 0** 에서 걸린다
    (아래 침대 면 0.45 m). 임계를 올리면 "벽에 붙는 궤적"이 아니라 "낮은 카메라"가 먼저 잘린다.
  · 글자는 관측 카메라 회전 행렬을 씌워 billboard 한다. 오일러 손계산(`atan2(-side[1],
    -side[0]) + pi/2`)은 180도 틀려 거울상으로 나왔다.
  · 천장은 `--cut_above` 로 `hide_render` — 근평면 컷어웨이는 시선에 수직이라 **수평 천장을
    못 없앤다** (실측: eye 89개 / low 97개 mesh 제외). `--obs_side {auto,flip}` 로 관측 옆면을
    손으로 뒤집을 수 있고 `auto` 가 기존 동작이다.
- **`scripts/plot_approach_clearance.py`** — 위 `approach.json` 을 걸음별 clearance 곡선
  패널로. 실선 = 게이트 6방향, 점선 = 수평 4방향, 임계선 + 교차점 + 현재 걸음 커서.
  렌더와 프레임 수가 같아 `stack_videos.py --direction vertical` 로 2줄 영상이 된다.
  이 호스트 matplotlib 에 한글 폰트가 없어 라벨은 영문이다.
- **`scripts/run_approach_viz.sh`** — 위 둘의 드라이버 (eye / low × wide / zoom 4벌).
  결과: `results/20260903_collision_margin_d118/approach/`.

### Changed
- **`describe_instances_vlm.py` — 같은 라벨 형제를 실제로 가르게 (D120-b, 2026-09-03).**
  초판은 형제가 있다고 **경고만** 했고, camel 두 마리에 대해 VLM 이 두 번 다 
  `"the camel walking near the wooden fence"` 류를 냈다 — 문장은 다른데 가리키는 근거가 없다.
  네 가지를 넣었다:
  · `## MEASURED FACTS` 블록 — 마스크에서 잰 화면 좌우 위치·면적비·깊이 순서를
    **결정론적으로** 프롬프트에 적는다. 추측할 여지를 없애는 게 목적이라 VLM 이 만들지 않는다.
  · 형제는 **순차 처리** — 먼저 확정된 형제 문구를 다음 질의에 붙이고, 내용어가 겹치기만
    하면 위반으로 되돌린다. 병렬로 던지면 서로를 모른 채 같은 말을 낸다.
  · 내용어 검증기 + `SPATIAL` 가드 — `MEASURED FACTS` 를 넣자 이번엔 반대로 위치·크기
    **에서만** 문구를 만들어 외양·행위가 통째로 빠졌다 (`"the woman sitting at a table..."`
    → `"the woman in a striped shirt"`). 구별은 되지만 target 으로는 후퇴라 **공간 어휘 밖
    내용어를 최소 1개** 요구한다. 상한은 12 → 14 단어.
  · `fallback_phrase()` — VLM 재질의 3회가 소진돼도 실측만으로 형제와 갈리는 문구를 만든다
    (면적비 ≥1.5 면 nearer/more distant, 아니면 좌/우). 계획서 §B6 "파이프라인은 절대
    hard-fail 하지 않는다".
  실측 camel: `dyn_0` = "the larger pale camel walking along the fence" /
  `dyn_1` = "the smaller pale camel standing near the fence".
- **`--min_clearance` 기본값 0.35 → 0.20 m, 6개 스크립트 전부 (D120, 2026-09-03, 사용자 지시).**
  D118 실측에서 0.35 는 이미 충돌 게이트를 통과한 export 행의 **80.6% (258 중 208)** 를 잘라냈고,
  그중 `wall` 단독이 **123 = 47.7%** 였다. 임계 하나가 뱅크 크기를 좌우하고 있었다는 뜻이다.
  `CLEARANCE_DIRS` 의 **±z 는 그대로 둔다** (사용자 확정) — 바닥/책상 윗면이 먼저 잡히는 건 알고
  남기는 것이고, 그래서 임계 쪽을 낮춘다. 바뀐 곳: `bank_to_blender_poses.py` ·
  `trumans_to_recon.py` · `trumans_first_pose_board.py` · `trumans_lite_bank.py` ·
  `trumans_scene_probe.py` · `trumans_raycast_viz.py`. **앞의 둘은 반드시 같은 값**이어야
  소스/target 대조가 성립한다 — 한쪽만 바꾸지 말 것. 예전 뱅크 재현은 `--min_clearance 0.35`.
- **`build_bank_captions.py --framing_timeline` 기본값 False → True (D120, 2026-09-03,
  사용자 확정).** D119 의 camel 65 변이 실측이 근거다 — median 한 스칼라가 `end/start` 비율
  `|r−1|>0.5` 인 변이 **52.3%** 를 숨겼고 문구가 실제로 바뀌는 변이가 **43.1%** 였다. dolly_in
  사다리 4칸이 전부 `"medium close-up"` 한 문장으로 접히던 게 이걸로 갈린다. **뱅크에
  `subject_area_seq` 가 없으면 조용히 예전 문구로 떨어지므로**, 효과를 보려면 Vista 52편을
  `--area_timeline` 으로 재굽는 것이 선행이다 (현재 이 열이 있는 뱅크는 camel
  `hole_bank_k6_d119smoke` 하나뿐). 되돌리려면 `--no_framing_timeline`.

### Fixed
- **`run_preset_warp_max_shard.sh` 의 샤드 분배가 실행 중 늘어난 뱅크를 조용히 흘렸다 (FIX-D118,
  2026-09-03).** 샤드 배정이 `ls -d out/*/<bank>/bank.csv` 의 **인덱스** 나머지라, 실행 도중에
  뱅크 디렉토리가 하나 생기면 그 뒤 영상이 전부 한 칸씩 밀린다. 실측: `camel` 뱅크가 샤드
  기동(13:24) 이후 14:02 에 완성되어 세 샤드 어디에도 안 잡혔는데, 세 샤드 모두 `rc=0` /
  `ALL DONE` 으로 끝나서 로그에는 흔적이 없다 — 결과 폴더 개수(51/52)로만 드러난다.
  `VIDEOS="camel" bash ... <gpu> 0 1 <out>` 처럼 **이름으로 지목**해 뒤늦게 채울 수 있게
  `VIDEOS` env override 를 넣었다. 안 주면 기존 동작 그대로다.
- **`trumans_raycast_viz.py` 가 위반 광선을 화면에서 가장 작은 자국으로 그렸다 (FIX-D118,
  2026-09-03).** clearance 광선은 `end = position + dir * distance` 로 그려지므로 **길이가 곧
  위반의 크기**다 — 임계 미달인 0.037 m 위반은 3.7 cm 토막이 되고, 통과 광선은 최대
  `probe_distance` 1.5 m 짜리 막대다. 끝점 구슬도 전 색이 같은 `thickness*2.2` 라 크기로도
  안 갈린다. 실측: 900px 렌더에서 **빨강 2 px vs 파랑 353 px** — 찾으려는 것이 안 보인다.
  · `--near_emphasis {none,ball,stripe,both}` + `--near_ball_scale`(기본 4.0) 추가. `none` 이
    기존 동작이고 그때 출력은 비트 동일하다. `ball` 은 끝점 구슬만 키우고(빨강 807→5219 px),
    `stripe` 는 막대를 **임계 길이까지** 늘리며(→1240), `both` 는 둘 다(→7646). 사용자가
    `both` 를 선택했다. `stripe` 의 막대 길이는 임계값이지 히트 거리가 아니다 — 실제 표면
    위치는 끝점 구슬이 표시하므로 정보 손실은 없지만 **막대 길이를 거리로 읽으면 안 된다**.
  · 임계 0.10 vs 0.50 대비가 `both` 에서 빨강 801 px vs 7646 px 로 갈린다 (같은 궤적·같은 뷰).
- **`run_raycast_threshold_viz.sh` 의 오빗 근평면이 방 안쪽을 잘랐다 (FIX-D118, 2026-09-03).**
  `trumans_raycast_viz.py` 는 근평면을 `orbit_r - spread*clip_cut` 로 잡는데, 기존
  `--orbit_scale 3.2 --clip_cut 1.5` 는 그 값을 실내 지오메트리 **한가운데**에 놓아 바닥을
  가로지르는 검은 띠와 흰 벽만 남겼다 — 처음 렌더한 12편이 전부 이 상태였고, 로그는 전부
  `rc=0` 이라 아무 신호도 없었다. `2.2 / elev 58 / clip_cut 1.15` 로 바꾸면 천장만 걷히고
  위에서 내려다보는 컷어웨이가 된다. 오빗·강조 값은 전부 env 로 덮을 수 있다
  (`ORBIT_SCALE` / `ORBIT_ELEV` / `CLIP_CUT` / `NEAR_EMPH`).
- **`render_bank_videos.write_video` 가 홀수 치수를 짝수로 패딩한다 (FIX-D118, 2026-09-03).**
  `macro_block_size=1` 은 imageio 의 자동 패딩을 끄는데 libx264 + yuv420p 은 짝수 치수만 받아서,
  격자 높이·너비 중 하나라도 홀수면 ffmpeg 가 첫 프레임에서 죽고 올라오는 건 `OSError: Broken
  pipe` 뿐이다 — 치수가 원인이라는 말이 어디에도 없고, 그때까지 GPU 로 돌린 렌더(변이 16개 ×
  49프레임, 씬당 ~2분)가 통째로 날아간다. 실측: `--columns 6 --tile_height 225` 가
  `contact_sheet` 에서 `3*225 + 2*4 = 683` 을 만들어 `friends-restaurant`/`car-roundabout`
  두 씬 모두 렌더 직후 터졌다. 격자 크기는 타일 크기 × 열 수 × 타일 개수의 곱이라 **호출부에서
  짝수를 보장할 수가 없어서** writer 안에서 막는다 (아래/오른쪽 1px `np.pad(mode="edge")`,
  패딩이 걸리면 한 줄 print). 짝수였던 기존 호출은 비트 동일. `render_preset_grid_warp.py` 등
  이 함수를 import 하는 스크립트가 같이 고쳐진다.

### Added
- **충돌 거리 임계를 숫자와 그림 양쪽으로 재는 도구 3종 (D118, 2026-09-03).** TRUMANS 경로에는
  이름이 비슷한 거리 손잡이가 **셋** 있고 지금까지 한 값으로 뭉뚱그려 불렸다 —
  `--behind_margin_frac`(굽기 게이트 임계, `×S`) · `--min_clearance`(레이캐스트 게이트 **임계**,
  m) · `--probe_distance`(레이캐스트 **광선 길이 상한**, m). 어느 쪽을 움직여야 하는지 답하려면
  게이트별 곡선이 필요하다.
  · `scripts/sweep_collision_margin.py` 확장 — `sweep_mesh(rows, margins, solved)` 가 bool 마스크를
    받아 `solved_keep`/`solved_total`/`solved_keep_frac` 을 낸다. 기존 `통과` 열은 **post-gate
    뱅크**를 다시 재는 거라 생존율이 아니라 여유분이고, `solved유지` 도 재굽기 생존율의
    **하한**이다(실제 재굽기는 변이를 버리는 대신 사다리를 줄이므로 비용이 "짧은 궤적"으로 나온다).
    이 함정 두 개를 docstring 에 명시. 청크 2편 이상이면 POOLED 블록을 찍는다.
  · `--plot` → `gate_curves.png`. 왼쪽 굽기 게이트(청크별 회색 + pooled 빨강, sub-voxel 구간
    `axvspan`, 현행 `0.02·S` 파선, 0.35 점선), 오른쪽 레이캐스트 게이트 통과율. **라벨 전부 영문** —
    이 호스트엔 한글 폰트가 없어서(`fc-list :lang=ko` 0건) 한국어를 쓰면 두부로 렌더된다.
  · `scripts/run_raycast_threshold_viz.sh` 신규 — 같은 궤적을 `--min_clearance` 만 바꿔 오빗
    렌더한다(`trumans_raycast_viz.py` 가 임계 미만 clearance 광선을 빨강으로 칠하므로 **어느
    광선이 어느 값에서 뒤집히는지**가 그대로 보인다). `--ray_stride 48 --orbit_scale 3.2
    --orbit_elev 32 --clip_cut 1.5` 는 실측으로 고른 값이다 — stride 8 / scale 2.4 는 광선이
    화면을 덮어 아무것도 안 보였다.
- **`scripts/run_preset_warp_max_shard.sh` (D118, 2026-09-03).** vista 전 씬 × preset 별 **사다리
  최대단** depth-warp 릴 샤드 러너. 앞서 손으로 돌릴 때 `target_hole` 정렬 후 **`c[0]`(최소단)**
  을 집는 버그가 있었다 — 사용자 지적 "이동량들이 작아진 것 같은데". 여기서는 `c[-1]`.
  anchor 는 `dyn_0` 우선, 없으면 solved 최다 anchor 로 **씬 안에서 고정**한다(씬마다 섞으면 격자
  차이에 preset 효과와 anchor 효과가 엉킨다). `render_bank_videos.py` 는 `<bank_dir>/<--name>`
  에 쓰므로 렌더 후 한 폴더로 복사한다.
- **G1(표면 뒤) 증거 소스를 고르는 `--collision_source {depth,mesh,both}` (D116, 기본 `depth`,
  2026-09-02).** depth shell 은 **소스 카메라가 본 표면**만 안다 — 소스는 자기 뒤를 안 보므로
  뒤로 물러나 벽을 뚫는 궤적에서 `behind_frac` 이 **정확히 0.0000** 이다. TRUMANS 는 우리가
  Blender 에서 렌더한 클립이라 `.blend` 부피 GT 가 있으니 그걸로 직접 잰다.
  · `mesh` = `.blend` 삼각형 → 5 cm bool 점유격자 → `scipy.ndimage.distance_transform_edt`
    로 clearance, "안쪽" 판정은 `scipy.ndimage.label` 로 **소스 카메라 중심에서 flood-fill 한
    도달 성분**의 여집합 (`binary_fill_holes` 는 닫힌 방이 자기 내부를 채워 못 쓴다).
  · `both` = 두 판의 **위반 프레임 합집합**. 배타 선택이 아닌 이유는 실측이다 —
    `tru_1d076f8c_a00_s3f0k6` 836변이, 임계 `0.02·S` = 3.9 cm 에서 **depth 130 / mesh 357 /
    겹침 0**. 어느 한쪽만 쓰면 다른 쪽 위반을 통째로 잃는다. 합집합은 프레임 마스크로 센다
    (`gates.behind_profile` / `mesh_collision.mesh_behind_profile` 의 detail dict 에 채널별
    `mask` 추가) — 개수 합은 중복 계상, `max()` 는 과소보고라 **안전 게이트가 느슨해진다**.
  · 3-way smoke (a00, anchor `dyn_0`, preset 3종, 전부 rc=0): `dolly_out__hole0.5` 의 knob 이
    depth 3.0(binding `none`) → mesh/both 1.0(binding `collision`), `pull_out_arc_left__hole0.5`
    3.0 → 0.855, `dolly_in` 은 depth `collision` / mesh `obb` / **both `collision`** — `both` 가
    mesh 의 상한과 depth 의 판정을 둘 다 살린다.
  · D47 규약(소스 카메라 자신을 기각하는 임계는 버그다)은 mesh 경로에도 assert 로 강제한다
    (`behind_context` 에서 소스 중심 전량의 도달성 + clearance > `margin_frac·S`).
  · 격자 경로 해석은 `lbm/mesh_collision.resolve_mesh_grid` **한 군데**에 둔다 — τ 뱅크와
    사다리가 다른 G1 으로 굽히면 사다리가 τ 뱅크에서 걸러진 변이를 되살린다. `mesh`/`both`
    인데 격자가 없으면 **죽는다** (조용히 depth 로 떨어지면 열 이름이 같아서 사후 구분 불가).
  · 뱅크 provenance: 두 CSV 의 `bank.json` `collision` 블록에 `source` / `mesh_grid` 기록.
    `sample_camera_bank.py` 는 `--measure_behind` 가 꺼져 있으면(τ 뱅크 기본) `"off"`.
    `SHAPE_DEFAULTS` 에는 **안 넣는다** — 그 표는 `emit_bank.py` 가 *pose* 를 재현하기 위한
    것이고 게이트는 pose 를 안 바꾼다.
  `lbm/mesh_collision.py`(`resolve_mesh_grid`, detail `mask`), `lbm/gates.py`(detail `mask`),
  `scripts/{sample_camera_bank,fit_hole_ladder}.py`(`--collision_source`/`--mesh_grid`).
- **`scripts/build_trumans_mesh_grid.py` — TRUMANS chunk → `mesh_grid.npz` (D116).**
  Blender(`bpy`, `scripts/trumans_export_mesh.py`)와 vista4d(scipy, `lbm.mesh_collision.build_grid`)
  가 서로 다른 인터프리터라 두 단계인데, 둘 다 `.blend` 경로·프레임 범위·소스 카메라 npz 를
  알아야 한다 (`tru_<rec8>_a<NN>_<tag>` → recon 폴더). 그 해석을 러너 셸에 손으로 적으면 두
  벌이 되므로(전례: `min_sweep_deg` 가 15↔30 으로 어긋남) 여기 한 벌만 둔다. 출력은
  `resolve_mesh_grid` 의 기본 경로 `<output_root>/<video>/mesh_grid.npz`. 중간 삼각형
  `mesh_gt.npz` 는 `--keep_tris` 없으면 지운다. 비용(a00, 502 objects, 49프레임): Blender
  export ~2분 + 격자 88.6 s, chunk 당 **1회** (이분법이 몇 번 돌든 격자만 읽는다).
  실측 격자: `voxel 0.05` / `static_shape (258,304,106)` / `static_occupied 390278` /
  `reachable 7458013` / `free 7923514` / 13.7 MB.

### Changed
- **`--collision_time_match` 기본 off → **on** (D116, 사용자 지시 "일단 켜줘", 2026-09-02).**
  끄는 쪽은 `--no_collision_time_match` 로 남는다. 위 "G1 충돌 판정에 시간축 정합" 항목의
  통제 A/B 대로 **순수하게 느슨해지는 방향으로만** 작동한다 (parkour 4/664, snowboard 6/196,
  감소 0건). 두 러너 셸은 D105 교훈대로 기본값에 맡기지 않고 **명시적으로** 넘긴다
  (`scripts/run_k6_d115_shard.sh` = `--collision_time_match --collision_source depth`,
  `scripts/run_trumans_d115_shard.sh` = `--collision_time_match --collision_source both`
  + mesh 격자 단계 ②-b 추가).
- **`fit_tau` 이분법이 깎는 대상을 preset 모양에 맞게 (F9, `--orbit_fixed_sweep`, 기본 **켬**,
  2026-09-02).** 증상: `orbit_left` 로 구운 뱅크가 이름만 orbit 이고 실제로는 3° 만 도는
  직선이었다. 원인: `se3.scale_traj(rel, s)` 는 SE(3) **로그 전체**를 s 배 해서 이동과 회전을
  같이 줄이는데, `traj.true_orbit` 은 `R = |t|/θ` 로 정의돼 있어 로그를 깎으면 **반경은 그대로고
  sweep 만** 줄어든다. 고침: sweep 을 고정한 채 `radius`/`dolly`/`lateral` 만 s 배 해서 궤적을
  **다시 만든다** (`lbm/presets.py:shape_resizer`). `τ ≈ 2·s·R·sin(θ/2)/z_med` 로 s 에 선형이라
  이분법 단조성은 그대로다. **sweep 에 반응하는 preset 만** 자동 감지해서 적용한다
  (`sweep` 을 반으로 준 probe 궤적과 비교) — 하드코딩 목록이 아니다.
  실측 감지 결과 14종: `push_in_arc_left/right`, `pull_out_arc_left/right`, `orbit_left/right`,
  `s_curve`, `orbit_left_pedestal_up`, `track_orbit_left/right`,
  `track_push_in_arc_left/right`, `track_pull_out_arc_left/right`. 나머지 29종
  (dolly/pan/tilt/truck/pedestal/**crane**/static/track_hold 계열)은 `resize=None` 이라
  **옛 경로 그대로 bit-preserved**. 정량(target τ=0.20, radius 0.6, sweep 45°, span 0.8):
  `orbit_left` log `scale 0.5312 tau 0.1993 rot 19.12 move 0.2003` → radius
  `scale 0.5312 tau 0.1970 rot 36.00 move 0.2003` (sweep 이 36°=0.8×45 로 복원, 경로 길이 동일).
  `push_in_arc_left` rot 13.78 → 18.00. `s_curve` 는 회전이 상쇄되는 구조라 양쪽 0.00.
  `dolly_in` 은 resizer `None`.
  `lbm/presets.py`(`shape_resizer`, `fit_tau(..., resize=)`, `tau_info["tau_resize"]`),
  `decode/build_poses.py`(`orbit_fixed_sweep=`), `scripts/{fit_hole_ladder,sample_camera_bank}.py`
  (`--orbit_fixed_sweep`/`--no_orbit_fixed_sweep`), `scripts/emit_bank.py`(뱅크의 `fixed` 에서
  재현, 키 없으면 F9 이전 뱅크라 `False`).
- **G1 소스 프레임 샘플링 13/7 → **49** (F11, `--behind_src_frames`, 2026-09-02).**
  13장 샘플링이 성겨서 표면 뒤 후보를 흘려보내고 있었다 — 49 로 올리면 binding 39건이 전부
  collision 으로 유입된다. `fit_hole_ladder.py` 13 → 49, `sample_camera_bank.py` 7 → 49.
  사용자 결정: "전체로 해도 시간 얼마 안 걸릴 것 같은데 전체로".
- **orbit sweep 하한 `--min_sweep_deg` 15 → **20** (F9 후속, 2026-09-02).** 이미 **하한**이지
  배제 조건이 아니다 (`span_frac = max(orbit_span_frac, min_sweep_deg / obs_az_span)`).
  argparse 기본만 20 으로 올리고 `SHAPE_DEFAULTS["min_sweep_deg"]` 는 옛 뱅크 재현용으로
  15.0 을 유지한다. 사용자 결정: "일단 20으로 놔둬보고 돌려보고 결정할게".
- **`scene_scale` (씬 단위 `S`) 재정의 — frame0 한 장 → 전 프레임 점군 (2026-09-02).**
  사용자 지시: "그냥 scene scale 은 sky 제외 mean of valid points distance from first source
  camera 로 정의하고 사용해줘". 새 기본 `mode="points_first_cam"`: **전 프레임**의 sky 아닌
  유효 depth 픽셀을 world 로 올려 **첫 소스 카메라 중심까지 거리의 평균**. dynamic 은 **안 뺀다**.
  옛 정의(frame 0 non-sky 평균 ray length)는 `mode="frame0_ray"` 로 남아 있다.
  왜: 카메라가 돌아 다른 공간을 보면 그 뒤 프레임의 실제 관측 거리가 몇 배가 되는데 frame0 S 는
  안 따라간다 — `u` 로 표현된 것 전부(OBB extent · 후보 거리 · 게이트 마진 `0.02·S` · τ 임계)가
  그 프레임들에서만 조용히 어긋난다. 72편 감사(`scripts/audit_scene_scale.py`)에서 프레임별 평균
  ray length 의 max/min 비가 p50 1.0961 · 12.5% 가 1.5 초과 · 최대 12.4030.
  **`trumans_to_recon.py:avg_scale_first_cam` 의 `avg_scale` 과 같은 정의**로 통일된 것이다.
  **실측 옛→새 배율** (stride 4): camel 1.014 · avocado-slice 1.077 · parkour 1.090 ·
  snowboard 1.389. dynamic 을 포함시킨 효과만 떼면 ×0.93~0.96 (동적 표면이 더 가깝다).
  **옛 뱅크와는 `u` 눈금이 다르므로 재굽기가 필요하다** — 그래프/캐시에 `scale.mode` 를 싣게
  했으니 대조 전에 반드시 확인할 것.
  `scene_graph/scale.py`, `scripts/build_scene_graph.py`(`--scene_scale_mode`,
  `--scene_scale_stride`), `lbm/cloud.py`(같은 두 인자 + `cloud.npz` meta 에 mode 기록).
- **`lbm/cloud.py` 의 `scene_scale` 복사본 삭제 → `scene_graph/scale.py` 에서 import.**
  정의가 두 군데면 한쪽만 고쳐도 아무 에러 없이 게이지가 갈린다 (이번 재정의 때 실제로 위험했다).
- **G1 을 static / dynamic **채널별로 따로** 판정 (사용자 정정 2026-09-02: "collision 을 static 도
  하고 dynamic 은 따로 해서 양쪽 다 판정하는 거였어").** 위 채널 분리는 판정을 합집합
  `behind_frac` 한 열로 했었다 — 이제 `behind_static_frac > max_behind ∨ behind_dyn_frac >
  max_behind_dyn` 로 **두 채널을 각자의 예산과 비교**한다. 판정식은 새 `behind_over()` **한
  군데**에만 있고 `over()`(사다리 이분법)·`physical_verdict()`(`--gate_before_render` 의 렌더
  생략 판정)가 **같은 함수**를 부른다 — 예전엔 `over()` 만 스칼라 비교라 두 판정이 갈릴 수 있었다.
  이를 위해 `probe()` 의 2번째 반환값을 `behind_frac` 스칼라 → **stats dict 통째**로 바꿨다.
  새 인자 `--max_behind_frac_dyn` (기본 `None` = `--max_behind_frac` 과 동일).
  예: 벽은 한 프레임도 허용 안 하되(`--max_behind_frac 0.0`) 지나가는 사람은 조금 봐준다
  (`--max_behind_frac_dyn 0.1`). **기본값에서는 채널 OR 이 합집합과 수학적으로 동치라 판정이
  예전과 bit-identical** 이다. `bank.json` 의 `collision` 블록에 `time_match` 와
  `max_behind_frac_dyn` 을 싣는다 (없으면 뱅크만 보고 두 arm 을 구분할 수 없었다).
  두 CSV(`sample_camera_bank`, `fit_hole_ladder`)에 `behind_static_frac` / `behind_dyn_frac`
  열 추가 — 진단이 아니라 **판정에 쓰는 값**이다. 요약의 "남은 위반" 집계도 같은 함수로.
- **G1(표면 뒤) 충돌 판정에 시간축 정합 (`--collision_time_match`, 기본 **off**, 2026-09-02).**
  사용자 지적: "물리 판정 기준은 4d point cloud 로 하려면 OBB 도 해당 시간축에 맞는 OBB 로".
  감사 결과 **OBB 를 쓰는 게이트(G5 `obb_clearance:194` / G6 `elevation_profile:239` /
  G7 `approach_profile:281`)는 이미 프레임별** `node_obb_at(node, f)` 로 갈라지고 있었다.
  진짜 frame-0 잔재는 **G1** 이었다 — `behind_profile` 이 궤적 프레임 하나를 샘플된 소스
  프레임 **전부**에 되쏘아서, frame 0 에 서 있던 사람이 궤적 48프레임의 카메라까지 막았다
  (같은 리포 `render.standoff` 는 `temporal_persistence=False` 로 정반대 규약이었다,
  `lbm/render.py:105-108`).
  **채널 분리 (사용자 지시 2026-09-02, "dynamic 을 완전 빼지는 말고 dynamic 의 경우 해당 시간의
  plan, src 카메라만 매칭해서 이용해서 따로 측정하게 해줘").** `behind_surface_frames(channel=)`
  가 셋으로 갈린다 — `"all"`(=레거시) / `"static"`(전 소스 프레임, **동적 픽셀 제외**) /
  `"dynamic"`(**시간이 맞는 소스 프레임 1장만**, **동적 픽셀만**).
  ~~게이트가 보는 `behind_frac` 은 두 채널의 합집합~~ → **아래 "G1 을 채널별로 따로 판정"
  항목으로 대체됨** (사용자 정정: 두 채널을 각자 판정한다).
  열은 6개: `behind_static_frames/_frac/_worst`, `behind_dyn_frames/_frac/_worst`
  (`behind_dyn_worst ≤ 1` — 소스 프레임 1장만 보므로).
  이전 설계(매칭 프레임을 `frames` 에 **주입**)는 프레임 집합이 arm 마다 달라져 A/B 를
  오염시켰으므로 폐기했다 — static 채널은 `frames` 를 그대로 쓴다.
  끄면 `dynamic_mask=None` / `plan_frame=None` / `channel="all"` 이라 예전 경로와 **bit-identical**.
  `lbm/gates.py`(`behind_surface_frames`, `behind_profile`) +
  `scripts/{sample_camera_bank,fit_hole_ladder}.py`.
  **정량 (통제 A/B, `--behind_src_frames 49` 로 양쪽 프레임 집합 고정 후 on/off):**
  parkour 664변이 중 status 변경 **4(0.6%)**, binding 이동 0, knob 증가 4 / **감소 0**,
  `behind_frames` 평균 8.38 → 6.83. snowboard 196변이 중 status 변경 **6(3.1%)**,
  binding `collision → obb` 6, knob 증가 19 / **감소 0**. **감소가 한 건도 없다** —
  순수하게 느슨해지는 방향으로만 작동한다. 첫 (미통제) A/B 는 매칭 프레임 주입 때문에
  프레임 집합이 달라져 parkour 에서 방향이 반대로 나왔었다 — 그건 시간축 정합이 아니라
  **13장 샘플링이 성기다**는 별개 결함이다 (`fix.md` F11: 13 → 49 로만 바꿔도
  binding 39건이 전부 collision 으로 유입).
- **`fit_hole_ladder.py` / `sample_camera_bank.py` — 물리 게이트를 렌더 **앞**으로
  (D113, `--gate_before_render`, 기본 켬, 2026-09-02).** 사용자 지적: "물리 판정이 실패하면
  사실 렌더할 필요가 없잖아". `over()` 의 if/elif 체인은
  `collision(G1) → obb(G5) → ground/elev(G6) → approach(G7) → hole → occlusion` 순인데,
  앞의 다섯은 재투영과 3×3 곱뿐(**렌더 0회**)이고 `hole` 은 점군 래스터다. 즉 물리 게이트가
  걸린 knob 의 `hole` 값은 **애초에 소비되지 않는데** 그걸 재려고 렌더를 돌리고 있었다.
  `measure_trajectory(gate_check=...)` 가 기하 열만 먼저 계산해 걸리면 `hole_*`=`nan` 으로
  즉시 반환한다 (0.0 을 넣으면 "구멍 없음"으로 읽혀 판정이 조용히 뒤집힌다).
  기존 산출물이 안 바뀌는 근거: 최종 CSV 행은 `verify_frames` 로 **다시** 재고, 이분법 캐시에서
  읽는 건 `poses`/`info`/`mult` 뿐이다. 유일한 예외인 사다리 기준점
  `hole_static = probe(0.0)` 만 `force_render=True` 로 강제한다.
  **검증(martian-flag 258변이, gate ON vs OFF):** 57열 중 **결정열 54개 전부 동일**, 다른 3열
  (`subject_visible_min` 181/258 · `subject_visible_frac` 172/258 · `subject_area_med` 121/258,
  최대차 0.072)은 **같은 설정 2회 재실행에서도 똑같이 흔들리는** 렌더러 비결정성이다
  (`hole_bank_occl_off` vs `occl_off2` 대조로 확인: 같은 3열, 최대차 0.025, 결정열 0건).
  절감량은 요약줄 `게이트 선차단 N` 으로 찍는다 (martian-flag 187). `--no_gate_before_render`
  면 D113 이전과 완전히 동일.

### Added
- **`scene_graph/scale.py:assert_scale_mode()` — 옛 게이지 그래프로 새 뱅크를 굽는 걸 막는 가드
  (F2, `--allow_legacy_scale` 로 탈출, 2026-09-02).** 게이트 임계가 전부 `S` 배율이라
  (`behind_margin_frac·S`, `behind_clear_frac·S`, `obb_clear_floor·S`, `min_ground_clear·S`,
  가림 `0.02·S`), `S` 정의를 바꾸는 건 임계를 통째로 옮기는 것과 같다. 그런데 그래프 파일에는
  숫자만 남고 정의가 안 남아서, 옛 그래프로 새 뱅크를 구우면 **씬마다 다른 배율로 임계가
  어긋난 채 조용히 통과**한다 (옛→새 배율 실측: camel 1.014 / avocado-slice 1.077 /
  parkour 1.090 / snowboard 1.389). `fit_hole_ladder.py` · `sample_camera_bank.py` 가
  `load_graph` 직후에 부르고, 확인한 mode 를 `bank.json` 의 `fixed.scale_mode` /
  manifest 에 기록한다. **디스크의 `scene_graph.json` 53편 전부가 `scale.mode` 키 자체가 없는
  legacy** 라 이 가드는 전량에서 걸린다 (의도한 동작 — 그래프 재굽기가 선행 조건).
- **`CinemaTraj/scripts/audit_scene_scale.py` — 씬 단위 `S` 정의 후보 감사 (2026-09-02).**
  사용자 지적: "scene scale 은 카메라가 돌아서 다른 공간을 보면 달라질 수도 있을 것 같은데
  sky 나 dynamic 을 제외한 나머지 depth 들을 다 unproject 해서 첫 카메라나 카메라 centroid 에서의
  point 거리 median 이 더 적절하지 않음?". 지금 정의(`scene_graph/scale.py:scene_scale`)는
  frame 0 non-sky **평균 ray length** 하나다. 후보 6종(`S_f0`, `S_f0_nodyn`, `S_frames_mean`,
  `S_pts_first`, `S_pts_centroid`, 프레임별 산포 `S_t_p05/p50/p95/ratio/drift`)을 같은 영상에서
  나란히 잰다. **고치지는 않는다** — 바꾸면 뱅크 정체성이 바뀌어 재굽기가 따라온다.
  전 코퍼스 72편 실측: `S_t_ratio`(한 클립 안 프레임별 최대/최소) p50 **1.0961**,
  p95 1.8496, max 12.4030, `>1.2` 25.0% / `>1.5` 12.5% / `>2.0` 5.6%.
  `r_pts_first`(=`S_pts_first/S_f0`) p50 **1.0078**, p05 0.4279, p95 1.7514, max 4.3498.
  CPU 전용, env `vista4d`. CSV `/tmp/scene_scale_all.csv`.
- **`CinemaTraj/scripts/audit_tau_axes.py` — 이동량 축 후보 비교 (2026-09-02).**
  사용자 지시: "tau 이동거리 기준을 object-centric + track, free-moving 을 분리하려는데 각각
  기준이 될 수 있는 수치들이 뭐가 있을지 후보군들 비교해줘". 계열은 `aim` 으로 가른다
  (`family_of`: `track_*` → track / `aim=="look_at"` → object / `free`·`traj` → free).
  축 11종(`tau_source`, `tau_shape`, `path_S`, `net_S`, `rel_sub`, `dr_ref`, `r_ratio`,
  `az_deg`, `elev_deg`, `rot_deg`, `subtend_ratio`)을 계열 × 축으로 재고, 좋은 축의 조건 셋
  (`hole_fraction` 과의 Spearman ρ / 씬 간 비교가능성 `cv_between` / 안 죽어 있음 `frac_flat`)
  을 한 줄에 낸다. `--status solved` 로 사다리가 못 푼 변이를 뺀다.
  `hole_bank_k6_d99` 52편 12,478 solved 변이 실측 결과는 대화 기록 참조.
  CPU 전용, env `vista4d`. CSV `/tmp/tau_axes_solved.csv`.
- **`CinemaTraj/scripts/viz_gravity.py` — GeoCalib 중력방향 오버레이 영상 (2026-09-02).**
  사용자 지시: "lens scene 에서 GeoCalib 으로 나온 중력방향 시각화해줘". `--style plumb` 는
  격자 셀마다 유효(비-sky) 픽셀을 unproject 해서 `up_world` 방향 선분을 긋고, `--style grid` 는
  `ground_z` 평면에 중력 정렬 격자를 깐다. `--compare cam_up` 으로 폴백(카메라 up)을 같이
  그려 차이를 눈으로 본다. 지평선(소실선)은 `inv(K).T @ (R_w2c @ up)`.
  camera-lens 실측: `up_world=[-0.16359,-0.82299,-0.54399]`, conf 0.489, spread 10.222°,
  `angle_to_cam_up 24.199°`, `angle_to_ransac 18.334°` (`disagrees_with_ransac: true`,
  ransac inlier 0.2675 / wall plane 0개). **`grid` 는 이 씬에서 쓰지 말 것** —
  `ground_z=-3.5244 u` × S=2.186 이라 바닥면이 화면 밖이고, RANSAC 지면 자체가 못 믿을 값이다.
- **`scripts/audit_bank_status.py` — 뱅크 `status`/`binding` 집계 + `clamped_low` 완화 후보
  목록 (2026-09-02).** 사용자 지시: "clamped_low 만 모아서 나중에 따로 기준점 완화해서
  돌려볼 수 있게 리스트 만들어줘". 판정은 **이미 기록되고 있었다** — fit 은 탈락 변이를 지우지
  않고 `status`/`binding` 두 열로 남긴다. 없던 건 그걸 코퍼스 전체로 모아 보는 도구다
  (렌더 0회). `--relax_list` 로 `clamped_low*` 행만 binding 별로 묶어
  `bank_relax_candidates_v1` JSON 으로 뽑는다 — 부분 완화 재fit 의 입력.
  **실측(`out/*/hole_bank_k6_d99`, 52편 30,556변이):** status 는 solved 12478(40.8%) /
  obb_limited 3021(9.9%) / approach_limited 2880(9.4%) / clamped_low+tau_floor 2265(7.4%) /
  unreached 2062(6.7%) / elev_limited 2029(6.6%) / ground_limited 1986(6.5%) /
  shape_limited 1230(4.0%) / collision_limited 1126(3.7%) / static 772(2.5%) /
  clamped_low 224(0.7%) + tau_floor 변종들. 완화 후보 2489(8.1%) 중
  **2265(91.0%)가 `+tau_floor`** — 게이트가 아니라 **knob 하한**이 진범이다.
- **`scripts/viz_g1_collision.py` — G1(behind-surface) 판정 시각화 (2026-09-02).**
  사용자 지시: "G1 충돌 어떻게 판정된건지 시각화해서 보여줘". 프레임당 왼쪽=플랜 카메라
  점군 렌더(구멍 마젠타), 오른쪽=소스 probe 13프레임 격자에 카메라 중심 투영점 +
  `radius_px` 패치 박스 + 판정색 테두리. **관통(pierce, `z > depth+margin`)과
  여유부족(tight, `clear_frac` 때문에 걸림)을 색으로 가른다** — CSV 의 `behind_frac` 하나로는
  구분이 안 됐다. **실측(parkour `dyn_0__dolly_in__hole0.5`, behind_frac 0.9796 = 48/49):**
  `pierce 0(0.0%) / tight 48(7.5%) / clear 70(11.0%) / skip 519(81.5%)` —
  **관통 0건, 전부 standoff 부족**이고 프레임×소스 637쌍 중 81.5%는 판정 자체가 불가였다.
- **`models/Planner/CinemaTraj/fix.md` — 미룬 수정 목록 (2026-09-02).** 사용자 지시:
  "다른 요소들도 다 고치고 난후에 한 번에 같이 적용할 수 있게 고칠 목록들 fix.md 에 기록해줘".
  8항목(F1 drift 라우팅 / F2 프레임 로컬 스케일 / F3 G1 사유 분리 / F4 judgeable_frac /
  F5 tau_floor / F6 프레임별 OBB / F7 시간축 절단 / F8 ground_z 앵커) 전부
  **뱅크 전량 재굽기**를 요구하므로 개별 반영하지 않고 한 판에 적용한다. 항목마다
  증상/원인/영향범위/실측근거 + 권장 적용 순서.
- **`scripts/run_dynpose_d110_export.sh` — D110 후처리 체인 (caption → export → dd10 리스트)
  을 스크립트로 고정 (2026-09-02).** D107 때 이 체인을 세션 안에서 손으로 돌렸다가 caption
  단계가 **아직 굽고 있던 샤드를 앞질러서** 2편(`00e9f728` 62변이 / `015b197d` 9변이)이
  `captions.json` 없이 export 에서 빠졌다 — `/tmp/d107_export.log` 에 "skipped 2" 경고 한 줄로만
  남아 export 표만 보면 71 변이가 사라진 걸 못 잡는다. 그래서 ① 러너 PID 와 자식 `fit/emit` 이
  둘 다 없어질 때까지 막고(d99 때 러너가 죽어도 자식 6개가 살아남은 전례), ② caption 후 뱅크
  수와 `captions.json` 수가 같은지 assert 해서 어긋나면 export 전에 중단한다. holdout 은 d107 과
  동일한 27편(`/tmp/d107_test_videos.txt`)이라 d107/d109 arm 과 paired 로 읽힌다.
  (d107 의 그 2편 captions 는 사후 생성해 뒀다 — 다만 d107 코퍼스는 학습 중이라 재export 안 함.)
- **`scripts/run_dynpose_d110_shard.sh` — d107 위에서 hole 사다리 2단→4단 재fit (D110,
  2026-09-02).** dd 10%/preset 90% 를 서브샘플(D109)로만 만들면 코퍼스가 58% 로 줄어서,
  대신 fit 사다리를 k6 원래 4단으로 되돌려 preset(track 포함)·dd 변이를 ~2배로 늘린다.
  graph/route/τ뱅크 는 d107 것 재사용 (사다리는 fit 손잡이라 앞 단계가 안 바뀐다).
  산출 hole_bank_d110. dd 는 export 후 10% 균등 서브샘플 → 총 ~9.7k 예상.
- **`scripts/run_dynpose_d107_shard.sh` — dynpose 코퍼스 재생성 러너, D84 라우팅 레시피 + d99
  세팅 (D107, 2026-09-02).** D84 원본 러너는 `/tmp` 에만 있었다 — 이 파일이 영구본이다.
  유지: anchor 2 + preset 라우터(물체 이동 많으면 track_* 포함) + DataDoP 외부 궤적 4/anchor,
  τ 씨앗 1.00, hole 사다리 0.20/0.35, 이름 해시 샤딩. 변경 3가지: ① graph 를 GeoCalib auto 로
  재빌드 — 기존 280 graph 의 53%(149편)가 `camera_up_fallback`(GT 실측 median 10.9° 기움)이라
  그 위에 deroll 을 걸면 지평선을 틀린 각도로 세운다; ② fit 에 `--deroll` + `smooth_kf`
  (vista/trumans d99·d106 와 같은 회전 규약 — 세 코퍼스를 섞어 학습); ③ τ 뱅크에도 aim/ease/
  deroll 플래그 명시. 산출물: `bank_d107` / `hole_bank_d107` / `preset_route_d107.json`,
  옛 graph 는 `scene_graph_pre_d107.json` 백업.
- **`build_scene_graph.py --subject_source gt` + `gt_trumans.gt_subject()` — TRUMANS subject
  OBB·조준점을 depth 점군 shell 대신 blend mesh AABB(probe `human_track`)에서 (D106, 2026-09-01).**
  사용자 지시 "Trumans는 합성데이터니까 최대한 blender 3d mesh를 이용해야해". 왜: 점군은 보이는
  면만 있는 shell 이라 systematically 작다 — 9편 프레임 정렬 실측: 높이비 median 0.73 (min 0.42),
  수평비 0.70, 조준점(track center) 오차 median 0.255 m = 키의 16%. 하류가 읽는 것 전부가
  갈아끼워진다 (`track.center_smooth` = 뱅크 look_at/aim keyframe·follow, `obb.extent`+`track.yaw`
  = `node_obb_at` 충돌 slack, `viewing_distance.d_ref`). 근거: blend→G 가 `--gravity_source gt`
  일 때 순수 yaw×1/S 라 blend 축 AABB 가 G 의 OBB 로 정확히 옮겨진다 (assert 로 지킴).
  기본값 `pointcloud` 면 예전 결과 비트 동일. 노드에 `subject_gt.estimated_aim_err_med_u` 로
  "추정이 얼마나 틀렸었나"를 박아 둔다 (gravity 의 `estimated_error_deg` 와 같은 규칙).
- **`scripts/run_trumans_d106_shard.sh` — d99 러너에 `--subject_source gt --ground_source gt`
  를 더한 재기동 판 (D106, 2026-09-01).** d99 bake 는 canonical 2/188 에서 kill (사용자 지시),
  pre-GT graph 로 구운 9편 산출물은 wipe. 뱅크 dir 이름은 d99 그대로 (vista d99 와 한 코퍼스),
  graph 마커만 `.graph_gt_d106`. 188/188 probe 커버리지(human_track 49프레임 + floor_z) 확인.
- **`results/20260901_d99_preset_grid/render.sh` — 굽는 중인 d99 뱅크의 preset 격자 프리뷰
  (D105, 2026-09-01).** `render_bank_videos.py` 는 **고치지 않았다** — anchor 와 사다리 단을
  고정하고 preset 만 푸는 호출을 드라이버로 감싼 것뿐이다 (스크립트 docstring 의 "같은 강도에서
  preset N종 = 모양 축"). 축을 안 고정하면 타일 차이에 preset 효과와 τ 효과가 섞인다.
  · 단 = `hole` 사다리 0.2 (rungs 0.1/0.2/0.35/0.5). 0.1 은 preset 이 서로 안 갈리고,
    0.35+ 는 궤적보다 구멍이 먼저 보인다.
  · `--deroll` 을 **안 넘긴다**: d99 뱅크는 이미 deroll 된 pose 를 들고 있다. 그 플래그는 옛
    뱅크를 렌더 직전에 세우는 용도라 여기 켜면 두 번 돈다.
  · `GPU=` 로 넘겨야 한다. 스크립트 안의 기본값 `1` 은 **쓰지 말 것** — GPU 1 은 다른 세션이
    73GB 를 쥐고 있어 `point_cloud.py:146` 에서 OOM 으로 죽는다 (util 0% 여도 메모리는 찼다).
- **`results/20260901_d99_preset_grid/render_trumans.sh` — 같은 격자의 TRUMANS 판
  (D105, 2026-09-01).** 뱅크가 **d77 이다 (d99 가 아니다)** — TRUMANS d99 는 아직 안 구웠고
  (Vista 5샤드가 GPU 를 다 쓴다) `out_trumans/*/` 에 있는 건 `hole_bank_k6_d77` 뿐이다.
  그래서 Vista d99 격자와 두 축이 다르다: ① `deroll` 키 자체가 없다(=off) ② `keyframe_ease`
  가 `smoothstep` (d99 는 `smooth_kf`). 세 번째 렌더가 ①만 떼어서 보여준다 — 같은 뱅크를
  `--deroll` 로 렌더 직전에 세워 다시 굽는다 (그 플래그의 정확한 용도).
  · 씬 두 개는 d77 뱅크 188 chunk 의 `path_len_u` 양 끝에서: `tru_1d076f8c_a17` 1.132(보행,
    preset 34종) / `tru_0adb88db_a02` 0.357(제자리, 30종).
- **`results/20260901_trumans_preset_blender/poses.sh` — 같은 격자를 depth warp 가 아니라
  Blender 로 (D105, 2026-09-01).** warp 은 소스 프레임 재투영이라 카메라가 크게 움직이면 화면
  절반이 hole 이 되고, 그러면 "궤적이 이상한 것"과 "warp 이 못 채운 것"을 못 가른다. TRUMANS 는
  씬이 `.blend` 로 있으니 렌더러가 정답을 안다 (hole 이 원리적으로 0).
  · **사다리 칸을 이름으로 못 박는다** (`--variant_ids ...__hole0.2`). 기본 동작은 preset 마다
    τ 최댓단을 집는데 그러면 preset 사이 강도가 달라져 "모양 축"이 안 된다.
  · 칸을 고정했으므로 raycast(기본 켜짐)에 걸린 preset 은 **아랫단으로 안 내려가고 드롭**된다.
    실측 드롭률 `a17` 22/34 · `a02` 22/30 — 뱅크의 depth-shell 충돌 게이트가 통과시킨 카메라를
    실제 `.blend` mesh 가 기각한 것이다 (D91 격차의 정량). 사유는 `<out>/selection.json`.
  · 드롭 대신 궤적 크기를 이분법으로 줄이려면 `--raycast_solve` (기본 off).

### Changed
- **`scripts/trumans_render_reel.py` 에 `--dirs` / `--name` (D105, 2026-09-01).**
  `bank_to_blender_poses.py` 가 만드는 `<out>/<preset>/rgb` 레이아웃을 그대로 받게 한 것.
  기존 경로는 `--work/--tag` 로 `*_<tag>/render_a<NN>` 을 훑고 타일 라벨을 `basename[-3:]` 로
  잘라 쓰는데, preset 폴더에서는 그 규칙이 `_up`/`eft` 같은 엉뚱한 라벨을 만든다.
  `load_clip(..., label=)` 를 추가해 라벨을 폴더 이름으로 넘긴다. **두 인자를 안 주면 예전과
  비트 동일** (49장 미만 폴더 제외 판정도 기존 `render_dirs` 와 같은 기준).
  · `human_depth` 에 `index`/`depth` 폴더 존재 확인을 넣었다. `--passes rgb` 로만 렌더한 폴더는
    `objects.json` 은 있는데 두 패스가 없어서 `--overlay` 가 `FileNotFoundError` 로 죽었다
    (`_source/index`). 이제 `z=-` 로 찍고 넘어간다 — 두 패스가 다 있으면 예전과 비트 동일.
- **`--deroll` 굽는 기본값을 `True` 로 (D105, 2026-09-01).** D99 가 "검증 후 argparse 기본값 두
  줄만 바꾼다"고 예고한 그 두 줄이다 — `fit_hole_ladder.py:1161`, `sample_camera_bank.py:904`.
  d98 뱅크(20,854행)를 preset 별로 세어 보면 roll 을 흘리는 건 `tilt_up`/`tilt_down` 820행씩,
  `pan_left`/`pan_right` 820행씩 = **3,280행(15.7%)** 이다. `tilt_*` 의 dev 163.88° 를 학습
  캡션 "tilts upward" 옆에 붙여 놓을 수는 없다.
  · **재현 폴백 5곳은 그대로 `False`** — `build_poses` 서명 / `fit_hole_ladder.SHAPE_DEFAULTS` /
    `emit_bank.FIXED_FALLBACK` / `merge_static_rung.DEROLL_FALLBACK`. 이것들은 "키가 없는 옛
    뱅크"를 되만드는 값이라 뒤집으면 D99 이전 뱅크가 전부 재현이 깨진다.
  · 새 샤드 드라이버는 기본값에 기대지 않고 `--deroll` 을 **명시적으로** 넘긴다. 뱅크 정체성을
    기본값에 맡기면 나중에 또 뒤집혔을 때 같은 폴더에 두 규약이 섞인다 (D90 `smooth_passes` 사고).
- **`configs/caption_presets.json` 을 D90 의미로 맞추고 빠진 5종을 채웠다 (D105, 2026-09-01).**
  `lbm/presets.py` 의 40종과 config 의 40종이 **이름은 겹치는데 뜻이 어긋나 있었다** — 어긋난
  채로도 캡션은 조용히 그럴듯하게 나온다.
  · 빠져 있어서 `build_bank_captions.py:178` assert 가 d98 뱅크 전체를 막던 5종:
    `dolly_in_look_at` / `dolly_out_look_at_legacy` / `tilt_up` / `tilt_down` /
    `track_dolly_in_look_at`.
  · `dolly_in`/`dolly_out`/`track_dolly_in` 은 D90 에서 `aim="free"`(재조준 안 함)가 됐는데
    문구는 아직 "toward the subject" 였다. 조준 문구를 새 `*_look_at` 이름으로 옮기고,
    free 쪽에는 이미 config 에 있던 `_dont_look` 문구("along its own axis without re-aiming")를
    붙였다. 죽은 키 `dolly_in_dont_look`/`dolly_out_dont_look` 은 삭제 —
    `resolve_preset` 이 그 이름을 `*_look_at` 으로 풀어서 조회에 절대 안 걸린다.
  · `tilt_up`/`tilt_down` 은 `pan_*` 과 같은 순수 회전 free-moving 이라 `targetless: true`.
  · **미결로 남긴 것**: `truck_*`/`pedestal_*`/`dolly_in`/`dolly_out` 도 D90 에서 `aim="free"`
    가 됐는데 아직 target 을 유지한다. d98 뱅크 `subject_visible_frac` 실측(각 820행) —
    `truck_left` med 0.753 / p25 0.009, `truck_right` 0.754/0.092, `pedestal_down` 0.849/0.563,
    `pedestal_up` 0.911/0.570 로 이미 targetless 인 `pan_left` 0.865/0.282 보다 truck 쪽이
    **더 나쁘다**. 바꾸면 학습 캡션의 target 절 분포가 크게 움직이므로 D104 A/B 결과를 보고 정한다.

### Fixed
- **중력축 기준 roll 누수 — `--deroll` (D99, 2026-09-01).** TRUMANS 의 arc 계열 카메라가
  기울어지는 원인은 궤적이 아니라 **회전 규약 두 곳**이었다. 둘 다 D98 의 중력축 수정으로는
  안 없어진다.
  · **① `aim="free"` preset 은 회전축이 카메라 로컬이다.** `traj.py:72-84` 의 `pan = rot_y`,
    `tilt = rot_x` 이고 `build_poses` 가 `basis_c2w @ rel_local[f]` 로 합성한다. 기울어진
    카메라를 카메라-로컬 y 둘레로 pan 하면 광축이 원뿔을 그려 지평선이 같이 돈다. 실제 삼각대
    pan 은 중력 둘레다.
  · **② `aim_keyframes` 는 keyframe *사이*에서 roll 을 만든다.** keyframe 회전이 전부
    roll=0 이어도 yaw·pitch 가 동시에 다른 두 회전 사이의 SO(3) 측지선은 중간에서 roll 을
    지난다 (구면 holonomy). `build_poses.py:774-788` 이 이걸 문서화하고 있었지만 assert 는
    **keyframe 위에서만** 걸려 있었다.
  · 실측 (`tru_0ac97866_a09_s3f0k6`, `pan_deg 45`, k6 `smooth_kf`, 소스 frame0 roll −8.079°).
    `dev` = `max_f |roll(f) − roll(0)|`, 단위 deg:

    | preset | aim | dev (off) | dev (on) |
    |---|---|---|---|
    | tilt_up / tilt_down | free | 163.88 / 163.84 | 0.00 / 0.00 |
    | pan_right / pan_left | free | 18.12 / 16.16 | 0.00 / 0.00 |
    | track_dolly_in_look_at | look_at | 19.85 | 8.08 |
    | crane_down | look_at | 10.35 | 8.08 |
    | push_in_arc_left | look_at | 9.47 | 8.08 |
    | dolly_in_look_at | look_at | 8.37 | 8.08 |
    | dolly/truck/pedestal/static/track_hold 등 14종 | free | 0.00 | 0.00 |

    `look_at` 계열의 잔여 8.08° 는 **소스 카메라 자신의 기울기**다 — `poses[0]==c2w_start`
    assert(`build_poses.py:802`) 가 frame 0 을 소스와 완전히 같게 요구하므로, 목표 roll
    프로파일은 keyframe roll(`kf0 = 소스 roll`, `kf>0 = 0`)의 선형보간이다. 그 프로파일 대비
    잔차는 검사한 12 preset 전부 **0.0000°**.
  · **광축을 안 건드리는 게 이 수정이 안전한 이유다.** 광축(`poses[:,:3,2]`) 둘레로만 다시
    돌리므로 `aim_err_max/med/frame0/at_kf`, `look_at`, τ, view angle, hole·coverage 가 전부
    그대로다 (실측: forward 축 최대 차이 `2.7e-06`도, 위치 차이 `0.0`,
    `aim_err_max_deg` 46.361 → 46.361).
  · 중력축에서 `pole_deg=5°` 안쪽 프레임은 right 축이 정의가 안 되므로 건너뛰고
    `deroll_skipped` 로 센다 (위 chunk 에서는 12 preset 전부 0).
  · **기본값은 아직 전부 `False`** — `build_poses` 서명 / `sample_camera_bank` /
    `fit_hole_ladder` / `emit_bank.FIXED_FALLBACK` / `merge_static_rung.DEROLL_FALLBACK`.
    D98 vista 뱅크가 "3번까지 검증된 세팅"으로 굽는 중이라 지금 뒤집으면 같은 뱅크 안에서
    앞뒤 영상이 다른 규약으로 섞인다 (샤드 드라이버가 영상마다 python 을 새로 띄운다).
    검증 후 `fit_hole_ladder.py` / `sample_camera_bank.py` 의 argparse 기본값 두 줄만 바꾼다.
  · 회귀 검증: D99 이전 `hole_bank_k6_d77` 변종 18건을 `poses.npz` 대비 재현 — 최대 오차
    `0.000e+00`.
  · `decision_fingerprint` 는 **켰을 때만** `deroll` 키를 넣는다 (`follow_smooth` / D73 과 같은
    규칙) — 안 그러면 D99 이전 뱅크 전량이 거짓 stale 판정된다.

### Added
- **`CinemaTraj/scripts/run_d99_caption_export.sh` — 두 코퍼스 캡션+export 를 한 드라이버로
  (D105, 2026-09-01).** vista 와 TRUMANS 는 이미 같은 두 프로그램(`build_bank_captions.py` →
  `vista4d_bank_to_dl3dv.py`)을 타는데 여태 각자 손으로 호출해 왔다. 캡션 인자를 배열 하나
  (`CAPTION_ARGS`)로 묶어 두 경로가 **글자 그대로 같은 인자**를 쓰게 한다
  (`--prompt_fields target,motion`, `--magnitude` 안 넘김 = 부사 없음).
  · 코퍼스별로 갈리는 건 셋뿐이다 — `--label_map`(TRUMANS 만; blend 오브젝트 이름 →
    자연어 명사, **`target:` 절의 명사만** 바꾸므로 문장 형식엔 안 닿는다), `--metadata_csv`
    (`event` 필드 출처인데 학습 프롬프트엔 안 들어간다), export 경로/해상도.
  · **`--image_scale` 이 코퍼스마다 다르다**: TRUMANS 소스는 960x540, vista 는 1280x720.
    둘 다 640x360 으로 맞춰야 섞어 학습할 때 `hw_list` 가 안 갈린다 → 0.6666666666666666 / 0.5.
    리사이즈 목표 (w,h) 는 스케일된 K 의 `cx*2`/`cy*2` 에서 나온다(`scaled_K`).
  · TRUMANS holdout 은 **recording 단위**(`TRU_TEST`, 기본 `tru_0ab19ed6 tru_1a1e205b`) —
    chunk 단위로 자르면 49프레임 슬라이딩이라 같은 recording 의 이웃 chunk 가 train/test 에
    동시에 들어가 test 가 새 씬 일반화를 못 잰다.
  · **왜 지금 필요했나**: 배포된 d77 TRUMANS 코퍼스의 프롬프트가
    `target: person. motion: the camera dollies straight forward toward the subject.` 인데
    그 행의 preset 은 `dolly_in` = D90 이후 `aim="free"`(재조준 안 함)다. 부사 문제가 아니라
    motion 절의 **의미**가 궤적과 반대였다.
  · d77 뱅크는 **지금 코드로 재캡션이 안 된다** — 40편 31,520행 중 **12,848행(40.8%)** 이
    `build_bank_captions.py:184` 의 aim 일치 assert 를 못 넘긴다(전부 D90 이전 `aim="traj"`
    행: `pan_*`/`truck_*`/`pedestal_*` 각 1,456, `dolly_{in,out}_dont_look` 각 1,456,
    `static_hold_dont_look` 380, `track_*` 156씩). d99 로 다시 구우면 현재 어휘로 기록되므로
    자동으로 해소된다 — 재굽기를 고른 또 하나의 이유다.
- **`CinemaTraj/scripts/run_k6_d99_shard.sh` / `run_trumans_d99_shard.sh` — deroll 켠 뱅크
  샤드 러너 (D105, 2026-09-01).** τ 뱅크 `bank_d99/`, emit 뱅크 `hole_bank_k6_d99/`. 한 폴더에
  두 규약을 안 섞는다는 D96/D97/D98 규칙 그대로 새 폴더에 쓰고 옛 뱅크는 안 건드린다.
  두 드라이버가 같은 인자를 넘긴다 (`--aim_keyframes 6 --keyframe_aim auto
  --keyframe_ease smooth_kf --fixed_focal --deroll`, preset 은 안 넘김 = 어휘 40종 전량) —
  vista 와 TRUMANS 를 섞어 학습하므로 규약이 갈리면 안 된다.
  · **TRUMANS 판만 scene graph 를 다시 짓는다 (`--gravity_source gt --no_skip_done`).**
    deroll 은 중력축 둘레의 roll 을 0 으로 만드는 연산이라 중력축이 틀리면 지평선을 틀린 각도로
    세운다 — 고치려던 것과 같은 기울어짐을 다시 심는 셈이다. 디스크의 191 chunk graph 는
    D98 이전 것이고 GT 대비 실측이 `ground_ransac` n=142 median 0.08° / p90 4.17° / max 9.73°,
    `camera_up_fallback` n=49 median **10.87°** / p90 21.31° / max **26.23°** 다. 191/191 전부
    blend world (z-up) GT 를 찾을 수 있으므로 추정할 이유가 없다. 옛 graph 는
    `scene_graph_pre_d99.json` 으로 한 번만 복사한다 (d77 뱅크의 입력이었다).
    지면(`--ground_source`)은 그대로 pointcloud — 중력만 바꿔야 원인이 하나다.
    vista 는 D98 에서 이미 52/52 가 GeoCalib 이라 graph 를 다시 안 짓는다.
- **`CinemaTraj/scripts/compare_aim_timing.py` — 매 프레임 조준 vs keyframe 조준 대조 (2026-09-01).**
  같은 뱅크 변이를 `aim_keyframes=0` / `6+smoothstep` / `6+smooth_kf` 세 arm 으로 되만들어
  회전 타이밍(`rot_ratio`, `accel_p95`)과 조준 오차(`aim_err_*`), 투영 `subject_in_frame` 을 잰다.
  위치 채널은 세 arm 이 비트 동일이어야 하므로 `dpos_max` 를 매 행에 찍는다 (실측 전 행 0.0).
  **`aim=="look_at"` 변이만** 대상이다 — `aim="traj"` 에서 `aim_keyframes=0` 은 "매 프레임 조준"이
  아니라 **조준 안 함**(`build_poses.py` 마지막 `else`)이라 대조 자체가 성립하지 않는다.
  TRUMANS 12편 480변이 실측(중앙값):

  | arm | rot_ratio | accel_p95 | aim_err_f0 | aim_err_kf_rest | aim_err_post |
  |---|---|---|---|---|---|
  | everyframe | 2.37 | 0.229 | 0.000 | 0.000 | 0.000 |
  | kf6_smoothstep | 5.01 | 0.698 | 17.582 | 0.000 | 1.323 |
  | kf6_smooth_kf | 3.48 | 0.231 | 17.582 | 1.747 | 2.174 |

  `aim_err_f0` 17.58° 는 보간 오차가 아니라 keyframe 경로가 frame 0 회전을 **일부러** 소스와
  같게 못 박은 값이다 (배포 뱅크는 `aim_anchor="subject"` 라 everyframe 쪽에는 램프가 안 걸린다).
  `sif` 는 세 arm 모두 1.000 이라 중앙 80% 안에서는 차이가 안 난다.

- **`CinemaTraj/scripts/bank_to_blender_poses.py --aim_arms` — 조준 arm 을 GT 렌더로 보기
  (D103, 2026-09-01).** 위 표는 세 arm 이 *숫자로* 얼마나 다른지만 말한다. 같은 뱅크 변이를
  `--aim_arms everyframe,kf6_smoothstep,kf6_smooth_kf` 로 **되만들어** `<preset>__<arm>.npz` 로
  따로 저장하고 `render.sh` 에 세 줄을 넣어, hole 없는 Blender GT 렌더로 나란히 볼 수 있게 했다.
  · **비면 예전 동작과 비트 동일** — 뱅크 `poses.npz` 를 그대로 쓴다. 되만들기는 `--aim_arms`
    를 준 경우에만 돈다.
  · **arm 사이 위치가 같다는 것을 매번 assert** 한다 (`|Δt| < 1e-6`; 실측 3 preset 전부
    `0.00e+00`). 세 렌더의 차이가 회전뿐임을 렌더 전에 못 박는 장치다.
  · **되만들기 자체의 검증**: `kf6_smoothstep` arm 은 배포 뱅크의 설정과 같으므로 뱅크가 저장한
    회전을 그대로 재현해야 한다 — 실측 `|ΔR| = 0.00e+00`. 나머지 두 arm 의 `|ΔR|` (everyframe
    1.82e-01, smooth_kf 8.08e-02) 이 조준 설정 때문이라는 근거가 이 0 이다.
  · **`aim != "look_at"` preset 은 건너뛴다** (사유를 표에 `SKIP aim=traj` 로 찍는다). `traj`
    에서 `aim_keyframes=0` 은 "매 프레임 조준"이 아니라 **조준 안 함**이라, 가드 없이 돌린 9-preset
    probe 에서 `truck_left`/`truck_right` 의 everyframe arm 이 `|ΔR| 0.833` — 조준 오차가 아니라
    preset 회전량 — 을 내고 있었다.
  · 레이캐스트 게이트는 기본값 그대로 켜 둔다. `tru_0ac97866_a08_s3f0k6` / `dyn_0` 에서
    `push_in_arc_left`(smoothstep `rot_ratio` 최악 10.37)는 사다리 네 칸이 전부 벽을 뚫어
    **렌더 불가**였다 — 최악 케이스를 못 보여주는 것은 이 대조의 한계로 기록한다.
    실제로 렌더한 것은 `s_curve` / `orbit_right` / `crane_down` 3종.
  · `--aim_arms` 와 `--raycast_solve` 는 동시 사용 금지 (assert).

- **`CinemaTraj/scripts/aim_arm_reel.py` — 조준 arm 렌더 + 프레임별 회전 속도 (D103,
  2026-09-01).** 렌더 세 줄만 쌓으면 "언제 회전하는가"가 안 보인다 — 세 arm 은 위치가 비트
  동일이라 구도가 거의 같게 흐르고, 눈으로는 "약간 덜컹거린다" 정도로만 읽힌다. 각 줄 오른쪽에
  ω(f)=프레임 간 회전각 곡선을 **세 줄 공통 y 축**으로 그리고 현재 프레임에 커서를 얹었다.
  keyframe 위치(`[0,10,19,29,38,48]`)는 점선으로 표시한다.
  ω 는 저장된 blend world npz 에서 바로 잰다 — `R_blend[f] = A @ R_bank[f]` 이고 A 가
  프레임마다 같아서 `R[f]ᵀR[f+1]` 에서 상쇄되므로 뱅크 world 에서 잰 것과 같은 값이다.
  `tru_0ac97866_a08_s3f0k6` / `dyn_0` 실측 (deg):

  | preset | arm | rot_max | rot_med | rot_ratio | sum_deg | accel_p95 |
  |---|---|---|---|---|---|---|
  | s_curve | everyframe | 4.4821 | 3.0276 | 1.4804 | 152.929 | 0.3960 |
  | s_curve | kf6_smoothstep | 6.4376 | 2.8664 | 2.2459 | 130.816 | 1.5659 |
  | s_curve | kf6_smooth_kf | 4.3497 | 2.9951 | 1.4523 | 127.770 | 0.4547 |
  | orbit_right | everyframe | 1.7790 | 1.4299 | 1.2441 | 65.712 | 0.3075 |
  | orbit_right | kf6_smoothstep | 2.3814 | 1.2517 | 1.9025 | 65.423 | 0.7723 |
  | orbit_right | kf6_smooth_kf | 1.6090 | 1.4491 | 1.1103 | 64.542 | 0.1059 |
  | crane_down | everyframe | 1.6392 | 1.1443 | 1.4325 | 54.562 | 0.1892 |
  | crane_down | kf6_smoothstep | 1.9112 | 1.1420 | 1.6736 | 50.487 | 0.5430 |
  | crane_down | kf6_smooth_kf | 1.2461 | 1.0248 | 1.2159 | 49.405 | 0.0764 |

  산출물 `models/Planner/CinemaTraj/results/20260901_aim_arms/aim_arms_{s_curve,orbit_right,
  crane_down}.mp4`.

- **`rot_ratio` 는 정지 조준에서 분모가 0 으로 붕괴한다 (D103 실측, 2026-09-01).** 이 chunk 의
  `look_at` solved 122 변이를 `rot_ratio = rot_max/rot_med` 로 줄세우면 1~13위가 전부
  `stat_*` anchor 다 (`stat_6__pull_out_arc_right__hole0.1` 55.68). 그런데 그 행의
  `rot_med` 는 **0.089°** — 정지 물체를 조준하니 median 회전이 0 에 붙어서 비가 폭발한 것이고,
  절대 펌핑량(`accel_p95` 1.208)은 상위권과 다르지 않다. 절대 `accel_p95` 로 줄세우면 1위가
  1.635 (`stat_1__s_curve__hole0.1`) 이고, 이번에 렌더한 `dyn_0 s_curve` 가 1.566 으로 사실상
  동률 — **최악 케이스를 이미 보고 있다**는 뜻이다.
  **`rot_ratio` 로 "가장 심한 변이"를 고르지 말 것. 절대량(`accel_p95`, `rot_max`)으로 고른다.**
- **`CinemaTraj/scripts/viz_tau_shape_topdown.py` — τ vs 궤적 모양 top-down (2026-09-01).**
  `target_tau` 를 직접 훑어 preset 궤적을 되만들고 graph frame G 의 xy 평면에 겹쳐 그린다
  (τ 내림차순 렌더 + frame48 끝점 마커라 호가 어디서 끝나는지 보인다). 아래 축에 요청 τ vs
  실측 τ + 방위각 span 을 같이 찍어 포화 지점을 표시한다. `--auto_lo` 가 τ 하한을 `tau_start`
  위로 올린다 (`tru_00add26c_a01` 실측 0.6332 — 그 아래는 전부 궤적 0).
  실측으로 드러난 것: orbit 의 τ 반응은 **반경이 아니라 sweep 각**이고, `obs_az_span` 이 좁으면
  `fit_tau` 배율이 호를 넓히는 대신 원에서 떼어낸다 (`dr/r_max` stat_4 span 235° → 0.65 vs
  stat_7 span 16.8° → 0.49~0.78, 후자는 top-down 에서 직선으로 보인다).
- **`--deroll` 을 렌더 단계에도 (`CinemaTraj/scripts/{render_bank_videos,bank_to_blender_poses}.py`,
  D102, 2026-09-01).** D99 의 `--deroll` 은 뱅크를 **굽는** 쪽(`fit_hole_ladder.py`)에만 있어서,
  그 플래그 이전에 구워진 뱅크를 나중 뱅크와 나란히 놓으면 지평선 기울기가 대조를 덮는다.
  실측 `tru_0ac97866_a08_s3f0k6` 같은 변이(hole0.35) 의 `max|roll|`:

  | preset | 옛 뱅크(`out_trumans`) | GT 뱅크(`out_trumans_gt`) |
  |---|---|---|
  | orbit_left | 21.81° | 0.24° |
  | truck_left | 14.99° | 0.23° |
  | pedestal_up | 9.22° | 0.24° |
  | crane_up | 8.80° | 0.34° |
  | crane_down | 8.92° | 0.07° |

  · 두 스크립트 모두 **렌더 직전에만** `decode.build_poses.deroll_poses` 를 한 번 돌린다 —
    뱅크 파일은 안 건드린다. 광축과 위치가 안 바뀌므로 `bank.json` 의 τ / hole / path_len_u 열이
    그대로 유효하고, `bank_to_blender_poses.py` 쪽은 레이캐스트 게이트 판정(clearance /
    floor_drop / subject_dist / LOS)도 한 자리도 안 변한다.
  · `up_world` 는 `render_bank_videos.py` 가 `scene_graph.json:gravity.up_world`,
    `bank_to_blender_poses.py` 가 blend world 의 `+Z`. **옛 TRUMANS 뱅크의 gravity 는
    `camera_up_fallback` 이고 GT 대비 25.9° 틀렸다** (`gravity.estimated_error_deg`) — 그래서
    옛 뱅크를 derolled 렌더해도 *추정된* 수평에 맞춰질 뿐 GT 수평과는 다르다. 이건 D88 이
    이미 고친 별개 축이라 여기서 섞지 않는다.
  · 기본 `False` = 예전과 비트 동일. `bank_to_blender_poses.py` 는 `selection.json` 에
    `deroll` / `deroll_skipped`(시선이 중력축과 평행해 roll 이 정의 안 되는 프레임 수)를 남긴다.
- **`CinemaTraj/scripts/rank_subject_motion.py` — 코퍼스 entry 를 subject 이동량으로 정렬.**
  "subject 가 많이 움직이는 경우만 모아서 평가"를 하려면 먼저 그 축이 있어야 한다. 코퍼스
  `da3/prompts.json` 의 `variant_id` 앞머리로 anchor 를 고르고, 그 노드 track 의 프레임간
  이동을 누적해 entry 를 줄세운다.
- **`CinemaTraj/scripts/slice_subject_in_frame.py` — `eval_subject_in_frame.py` 출력에서
  부분집합만 다시 집계.** sweep 전량을 다시 렌더하지 않고 entry 목록(예: high-motion 상위 N)만
  골라 같은 표를 낸다.
- **`CinemaTraj/scripts/bank_to_blender_poses.py --variant_ids` (D102).** 예전에는 preset 마다
  `tau_max` 최댓단을 자동으로 집었는데, 두 뱅크를 짝지어 비교하려면 **같은 사다리 칸**을
  못 박아야 한다 (τ 최댓단은 뱅크마다 다른 칸에 앉는다). 비우면 예전 동작과 비트 동일.
- **`video_generation/scripts/sam3_seg_instances.py`** — 그동안 CHANGELOG 에 안 적혀 있던
  ACTIVE-4 의 SAM3 text-PCS instance 분할 샤딩 스크립트를 뒤늦게 기록한다.
- **`CinemaTraj/scripts/eval_subject_in_frame.py` — eval 폴더 여러 벌을 같은 점군에 렌더해
  `subject_in_frame` 을 비교 (D100, 2026-09-01).** `verify.py` 는 `out/<video>/` 한 벌
  (`decision.json` + `poses.npz` + fingerprint)에 묶여 있어서 **모델이 예측한 궤적**에는 못 건다.
  이쪽은 궤적의 출처를 안 따지고 `test/<name>_transforms_pred.json` 만 받아
  `--eval_dir LABEL=DIR` 로 arm 을 여러 개 붙인다 (GT 는 `--include_gt` 로 ref 에서 자동 추가).
  지표 정의는 `verify.py:measure()` 를 **import 해서** 쓴다 (같은 정의를 두 번 안 적는다):
  `subject_in_frame`(중앙 `center_box` 0.80 안에 실루엣 중심) + 맥락용 `hole_fraction` /
  `subject_pixel_coverage` / `subject_zero_frames`. subject 노드는 entry 마다 다르므로
  코퍼스 `da3/prompts.json[<entry>]["variant_id"]` 앞머리(`dyn_1__dolly_in__hole0.2` → `dyn_1`)
  로 고른다 — 씬 단위로 고정하면 anchor 2개인 D84 코퍼스에서 절반이 엉뚱한 물체를 잰다.
  · **`center_box` sweep (`--sweep_hi/--sweep_lo/--sweep_step`, 기본 1.0→0.1 step 0.1).**
    `subject_in_frame` 은 실루엣 중심 하나에 걸린 임계값이라 상자를 줄인다고 49프레임을 다시
    렌더할 이유가 없다. `score()` 가 프레임별 중심(`centers`)을 행에 실어 두고 `in_frame_at()`
    이 사후에 임계만 다시 건다 — 렌더 1회로 전 구간 표가 나온다. `centers[f] is None`
    (subject 픽셀 0 = 화면 밖이 아니라 **소실**)은 `measure()` 와 똑같이 상자 크기와 무관하게
    실패로 세므로, **box 1.0 의 값이 곧 `1 − subject 소실 프레임 비율`**이고 거기서부터의
    하락분만이 순수 구도 성분이다. 표는 stdout + `--out` JSON 의 `sweep` 키에 같이 남는다.
- **`CinemaTraj/scripts/gendop_release_infer.py --rgbd_fit letterbox` + `letterbox()`
  (D100, 2026-09-01).** `eval.py` 의 center-crop 은 720x1280 을 넣으면 가운데 512x512 만 남겨
  가로 60% 를 버리고, GenDoP 자신의 예시(`assets/examples/text_rgbd`, 208x512 / 288x512)에
  있는 **zero-pad 띠**도 없앤다. 기본값은 `crop` 이라 기존 vista4d 경로는 bit-identical.
- `CinemaTraj/scripts/gendop_release_infer.py --depth_norm {none,median}` +
  `--depth_target_median` (D100). 아래 Changed 의 실측으로 **무의미함이 증명**되어 실제
  실행은 `none`(= `eval.py` 그대로)으로 했다. 스위치는 재확인용으로 남긴다.
- `CinemaTraj/scripts/gendop_preds_to_eval_dir.py` 의 `--prefix` / `--npz_kind` +
  `--video` 선택화 (D100). `--video` 를 생략하면 prefix 아래 **전 scene** 을 한 번에 돈다
  (dynpose val 은 22 scene 37 entry 라 필수). ref 이름은 `<prefix>_<scene>_<idx>_*`,
  npz 이름은 `<npz_kind>__<scene>__<idx>.npz` 로 따로 잡는다.
- `CinemaTraj/scripts/dynpose_gendop_inputs.py` — dynpose 코퍼스 entry → GenDoP RGBD 입력
  (`frame_0000.png` + `frame_depth_0000.npy`) 심기. scene 당 1벌을 만들고 entry 별 symlink 를
  건다 (같은 scene 의 37 entry 가 같은 frame0 를 본다).
- `CinemaTraj/scripts/render_pred_depth_warp.py` 의 `--name_prefix` / `--cloud_root` /
  `--eval_data` / `--corpus_root` — 코퍼스 손잡이 4개. vista4d(`vista4d_<video>_<i>`, `out/`,
  `Vista4D-Eval-Data`)와 dynpose(`dynpose_<uuid>_<i>`, `out_dynpose/`, `DynPose-LBM`)를
  같은 스크립트로 돌린다.
- `decode/build_poses.py` 의 `deroll_poses()` / `_roll_about_forward()` 와
  `build_poses(..., deroll=False)`, `--deroll` / `--no_deroll` CLI 쌍 (D99). 같은 쌍이
  `scripts/sample_camera_bank.py` / `scripts/fit_hole_ladder.py` 에도 있고, 뱅크 `fixed` 블록에
  `deroll` 이 기록되어 `emit_bank` / `merge_static_rung` 이 그 값을 따른다. `info` 에
  `roll_vs_frame0_max_deg` / `deroll` / `deroll_skipped` 를 추가.

### Changed
- **릴리즈 `text_rgbd` ckpt 의 RGBD 조건은 우리 입력에서 사실상 무효 — 실측 (D100,
  2026-09-01).** `--depth_norm none` 과 `median` 이 bit-identical 로 나온 게 발단이었다.
  `cond_embeds` 를 채널별로(text `[:,:77]` / image `[:,77:334]` / depth `[:,334:]`) 재고,
  같은 seed 로 뽑은 pose 토큰 301개의 차이를 같이 셌다.
  · **image 채널은 완전 무반응.** GenDoP 자신의 예시에서 RGB 를 다른 case 로 바꿔도
    d_img 0.0592 / **tok 0/301**, zeros 0.0499 / 0/301, 가우시안 노이즈 0.0441 / 0/301,
    우리 RGB 0.1101 / 0/301.
  · **depth 채널은 depth *내용*이 아니라 zero-pad 띠 기하에 반응한다.** case2→case3 은
    **같은 288x512 띠**에 다른 씬인데 d_dep 0.0202 / **0/301**. 반면 띠 폭이 다른
    case2→case1(208x512)은 11.4548 / 58/301, 띠가 아예 없는 우리 720x1280 center-crop 은
    11.4416 / 83/301, 그 우리 depth 를 **같은 288x512 띠에 레터박스**하면 다시
    0.0378 / **0/301** 로 돌아온다.
  · depth 는 스케일 불변(×100 → d_dep 0.0058 / 0/301) — `--depth_norm` 이 무의미한 이유가
    이것이고, `none`/`median` 이 bit-identical 이던 것도 전부 설명된다.
  · text 를 바꾸면 d_text 5.4786 / **277/301**. 즉 이 ckpt 는 사실상 text-only 다.
  → 실행 설정은 `--rgbd_fit letterbox --depth_norm none`. letterbox 는 화각을 안 버리면서
    depth 토큰을 학습 분포(띠 있음) 안에 둔다. **다만 위 실측대로 조건 신호는 거의 안 바뀌므로,
    같은 문장을 받은 entry 들 사이의 차이는 대부분 샘플링 잡음이다**
    (`generate_mode=sample`, `do_sample=True, top_k=10`).
- **중력축 기본 소스가 `--gravity_source auto` — GeoCalib 사이드카가 있으면 그것, 없으면
  예전 ground RANSAC (D98, 2026-09-01).** `scripts/build_scene_graph.py` 의 새 인자
  `--gravity_source {auto,ransac,gt,geocalib}` (기본 `auto`).
  · **왜.** ground RANSAC 은 **물었을 때만** 정확하고, 안 물리면 조용히
    `camera_up_fallback`(카메라 up 평균)으로 떨어진다. TRUMANS GT 65 chunk 실측 —
    GT 대비 각도 오차 median/p90/max:

    | RANSAC 판정 | n | ransac~GT | geocalib~GT |
    |---|---|---|---|
    | `camera_up_fallback` | 15 | 11.14 / 15.88 / 24.18 | 0.75 / 1.39 / 1.66 |
    | `ground_ransac` | 50 | 0.08 / 4.07 / 5.28 | 0.96 / 2.55 / 5.59 |
    | 전체 | 65 | 1.79 / 12.48 / 24.18 | 0.87 / 2.42 / 5.59 |

    RANSAC 이 물린 50 chunk 에서는 median 0.08° 로 GeoCalib 보다 낫지만, fallback 15 chunk 의
    꼬리(max 24.18°)를 GeoCalib 이 5.59° 로 자른다. **중력축은 roll=0 기준이자 OBB 의
    yaw·extent 축이라 한 편만 틀려도 그 편의 뱅크 전량이 Dutch angle 로 기운다** — 그래서
    median 이 아니라 꼬리로 고른다.
  · **정책은 "GeoCalib 주(主) · RANSAC 검증"이다.** RANSAC 은 계속 돌리되 결과를
    `gravity.ransac_method` / `ransac_inlier_ratio` / `angle_to_ransac_deg` /
    `disagrees_with_ransac`(>10°) 로 **기록만** 하고 자동 전환은 안 한다. 어긋나면 숨기지 말고
    드러내는 쪽.
  · **안 바꾼 곳(의도)**: `--gravity_source ransac` 은 사이드카를 통째로 무시하므로 예전 결과와
    **비트 동일**이다 (camel 을 사이드카 없는 `--output_root` 로 빌드 → `ground_ransac 0.73 4.6`
    재현 확인). TRUMANS 경로(`--gravity_source gt`)와 사이드카가 없는 코퍼스는 `auto` 에서
    자동으로 예전 동작을 탄다 — 샤드 스크립트를 하나도 안 고쳤다.
  · vista 52편 재빌드 실측(진행 중, 31편 시점): `disagrees_with_ransac` **7편**
    (bed-shopping 25.5° / magnifying-glass 24.5° / camera-lens 18.3° / mountain-man 17.9° /
    goat 15.7° / hike 14.3° / couple-newspaper 12.7°), RANSAC 이 애초에 fallback 이던 게 **5편**.
    나머지 19편은 3.6° 이내로 일치.
  · 새 중력축 위에서 뱅크를 다시 굽는다: `scripts/run_k6_d98_shard.sh`
    (τ `bank_d98/`, emit `hole_bank_k6_d98/`). **preset 은 안 넘긴다 = 현재 어휘 40종 전량.**
    D96/D97 과 같은 이유로 폴더 이름을 새로 준다 (한 폴더에 두 규약을 섞지 말 것).
- **굽기 기본값 `--tau_ref auto` — `track_*` 만 follow 기준, 나머지는 예전 그대로 (D97,
  2026-09-01).** `auto` 는 `PRESET_TAU_REF` 를 타고, 그 표는 `track_` 으로 시작하는 preset 에만
  `"follow"` 를 준다. 그래서 non-track preset 의 τ 는 한 자리도 안 바뀐다.
  · 바꾼 곳(굽기 기본값): `scripts/sample_camera_bank.py` / `scripts/fit_hole_ladder.py`
    argparse 기본값 `auto`.
  · **안 바꾼 곳(의도)**: `fit_hole_ladder.SHAPE_DEFAULTS["tau_ref"]`,
    `emit_bank.FIXED_FALLBACK["tau_ref"]`, `build_poses(..., tau_ref="source")` 서명 기본값 —
    D96 의 `keyframe_ease` 와 같은 이유로 **키가 없는 옛 뱅크의 재현 폴백**이라 영원히
    `"source"` 다. `emit_bank.decision_from_variant` 는 행의 `tau_ref`(resolve 결과)가 아니라
    `fixed.tau_ref`(요청값)를 다시 넘긴다.
  · `merge_static_rung.py` 는 `tau_ref` 를 `FIXED_KEYS` 에 **안 넣고** `TAU_REF_FALLBACK`
    폴백 대조로 검사한다 — 넣으면 D97 이전 뱅크가 (`None` vs `"auto"`) 로 전부 병합 거부된다.
  · 회귀 검증: 옛 `hole_bank_k6_d94` **196행 전량**을 `poses.npz` 대비 재현 — 최대 오차
    `0.000e+00`. follow 기준 대수 검증: `tau_max == max|p_shape − p_start|/z_med` (오차 < 2e-4),
    `tau_start == 0.0`.
  · 새 규약으로 구울 때는 D96 과 마찬가지로 **`$BANK` 이름을 새로 준다** (한 폴더에 두 규약이
    섞이면 `fixed.tau_ref` 대조 말고는 알 방법이 없다).
- **회전 스케줄 기본값이 `smoothstep` → `smooth_kf` (D96, 2026-09-01).** trumans / vista /
  dynpose 세 코퍼스를 앞으로 같은 규약으로 굽기 위한 것이다. 근거는 `decode/build_poses.py:965`
  에 적힌 714 변이 실측 — 프레임간 각속도 맥동비 **21.65 → 4.58**, 49프레임 조준오차 median
  **0.668° → 0.418°**. 대가는 keyframe 회전을 정확히 통과하지 않는 것 하나뿐이고 그건
  `aim_err_at_kf_deg` 로 뱅크에 남는다.
  · 바꾼 곳: `scripts/sample_camera_bank.py:865` argparse 기본값,
    `scripts/run_{trumans_d77,k6_d77,k6}_shard.sh` (하드코딩 `smoothstep` →
    `EASE="${EASE:-smooth_kf}"` — 옛 뱅크를 되만들려면 `EASE=smoothstep` 을 앞에 붙인다),
    `scripts/time_vista_stages.py:40`. `fit_hole_ladder.py` 는 D89 부터 이미 `smooth_kf` 였다.
    **dynpose 는 고칠 파일이 없다** — 그 드라이버는 ease 를 안 넘기고 두 스크립트의 기본값을
    그대로 타므로 이 변경으로 같이 넘어간다.
  · **안 바꾼 곳(의도)**: `fit_hole_ladder.SHAPE_DEFAULTS`, `emit_bank.FIXED_FALLBACK`,
    `build_poses` 서명 기본값 — 셋 다 **키가 없는 옛 뱅크를 되만들 때의 폴백**이라 영원히
    `smoothstep` 이어야 한다. 배포된 뱅크는 전부 `fixed.keyframe_ease` 를 명시적으로 싣고
    있어서 재현 경로가 안 바뀐다 (실측: vista 51 + dynpose 266 전량 `smoothstep`,
    `out/snowboard/hole_bank_k6_d94` 만 `smooth_kf`).
  · `run_static_rung_shard.sh` 는 **고정 기본값을 안 쓴다.** 붙일 대상 뱅크의
    `fixed.keyframe_ease` 를 읽어서 그 값으로 fit 한다 (`EASE_OVERRIDE=` 로 강제 가능) — 안 그러면
    smoothstep 으로 구워 둔 51편에 smooth_kf 정지 rung 을 붙이려다 `merge_static_rung.py`
    호환성 검사에서 rc=2 로 죽는다.
  · 새 규약으로 구울 때는 **`$BANK` 이름도 새로 준다.** 한 폴더에 두 규약이 섞이면
    `fixed` 블록 대조 말고는 알아챌 방법이 없다.
  · 검증: snowboard 3변이 τ 뱅크 + hole 사다리를 새 기본값으로 구워
    `fixed = {keyframe_ease: smooth_kf, smooth_passes: 12, smooth_lambda: 0.5}` 확인
    (`변이 3 렌더 57`, 전부 `solved`, `남은 위반 0`). 샤드 4종 `bash -n` 통과.
    기존 뱅크는 하나도 안 건드렸다 (샤드는 canonical 이 있으면 건너뛴다).
- **`hold` 는 이제 "안 움직이고 재조준도 안 한다" — 조준은 `_look_at` 으로만 표기 (D94,
  2026-09-01).** `static_hold`↔`static_hold_dont_look`, `track_hold`↔`track_hold_dont_look`
  네 이름이 서로 맞바뀌었다. 궤적은 넷 다 `T.hold(I, n)` 으로 **그대로**고 바뀐 건 이름과
  조준뿐이다.

  | D93 까지 | aim | → D94 | aim |
  |---|---|---|---|
  | `static_hold` | look_at | `static_look_at` | look_at |
  | `static_hold_dont_look` | free | `static_hold` | free |
  | `track_hold` | look_at | `track_look_at` | look_at |
  | `track_hold_dont_look` | free | `track_hold` | free |

  · **옛 뱅크는 다시 안 쓴다.** `dolly_in`/`dolly_out`(D76) 과 **같은 장치** —
    `LEGACY_AIM_COLLISIONS[("static_hold","look_at")] = "static_look_at"` 등 (이름, aim) 쌍으로
    되돌리고 `*_hold_dont_look` 은 `PRESET_ALIASES` 로 새 `*_hold` 를 가리킨다. 배포 d77 뱅크의
    `track_hold` 행 **3,282 개가 전부 `aim="look_at"`** 이라 물량은 track 쪽이 크다.
  · 배선: `lbm/presets.py`(PRESETS / STATIC_PRESETS / PRESET_ALIASES / LEGACY_AIM_COLLISIONS),
    `scripts/route_presets.py:187` (static 슬롯이 `*_look_at` 을 뽑도록 — 이름만 두면 카메라가
    조용히 바뀐다), `configs/caption_presets.json`, `lbm/prompts/system_traj.md`,
    `scripts/run_static_rung_shard.sh:24`, `presets.md` / `README.md` / `DECISIONS.md`.
  · 검증: (이름, aim) **19 조합** 전부 정식이름·디코드 aim·follow·tracking·static 판정 일치.
    `out/snowboard/{bank_d94 166변이, hole_bank_k6_d94 196변이}` 재생성 — `aim` 이 166/166,
    196/196 전 행에 기록된다. emit: `variants 196 중 196 내보냄 (필터 0 / 접힌 단 0 / path 0)`,
    `pose 재현 최대오차 0.000e+00`, `21<->49 왕복 최대오차 0.000e+00`.

### Fixed
- **`merge_static_rung.py` 가 `smooth_kf` 평활 인자를 대조하지 않았다 (D96).** `smooth_passes` /
  `smooth_lambda` 가 `FIXED_KEYS` 에 없어서, 정지 rung 을 다른 평활 세기로 fit 해 붙여도 통과했다.
  **위치는 정확히 맞고 회전만 어긋나는** 종류라 육안·assert 어느 쪽에도 안 잡힌다 (D90 에서 실제로
  당한 실패 모드). `fixed.keyframe_ease == "smooth_kf"` 일 때만 비교한다 — `FIXED_KEYS` 에 그냥
  넣으면 이 키가 아예 없는 smoothstep 옛 뱅크가 `None vs 12` 로 멀쩡한 병합을 거부한다.
  키가 없는 뱅크는 `build_poses` **서명** 기본값 12/0.5 로 폴백. 4개 경우(옛/신 smoothstep,
  smooth_kf 12-vs-4, smooth_kf 12-vs-키없음, ease 자체 불일치) 단위 확인.
- **emit 경로가 뱅크 행의 `aim` 을 안 날랐다 (기존 결함, D94 가 드러냄).**
  `scripts/emit_bank.py decision_from_variant` 는 `variant["preset"]` 만으로 decision 을 만들었고
  `decode/build_poses.py:581 resolve_aim(preset, trajectory.get("aim"))` 이 `None` 을 받아
  **preset 의 *현재* 기본값**을 탔다. 뜻이 뒤집힌 이름(D94 의 `*_hold`, D90 의 `truck_left` 등)에서
  조용히 다른 카메라가 디코드된다. 실측 — `dyn_0__static_hold__hole0.1` pose 재현 오차:
  `aim` 미전달 **6.574e-01** / 전달 **0.000e+00**. `sample_camera_bank.make_decision(..., aim=None)`
  → trajectory dict → `emit_bank` 로 배선했다. `decision_fingerprint`(`build_poses.py:101-150`)는
  `aim` 을 포함하지 않으므로 저장된 fingerprint 가 무효화되지 않는다.

### Added
- **GeoCalib 중력 사이드카 — `scripts/geocalib_gravity.py`(굽기) + `scene_graph/geocalib_sidecar.py`
  (읽기) + `scripts/run_k6_d98_shard.sh`(뱅크 재굽기) (D98, 2026-09-01).**
  · **왜 파일로 주고받나.** GeoCalib 은 `kornia` 를 요구하는데 그건 env `geocalib` 에만 있고
    scene graph 는 env `vista4d` 에서 돈다. 새 패키지를 깔지 않기로 했으므로 두 단계로 쪼갠다 —
    굽기는 env `geocalib`, 읽기는 env `vista4d`. 포맷 `geocalib_gravity_v1`,
    파일 `out/<video>/geocalib_gravity.json`.
  · **규약(실측으로 확정)**: `result["gravity"].vec3d` 는 **카메라 프레임(OpenCV)의 up** 이다.
    따라서 `up_world = R_c2w[f] @ vec3d[f]` — 부호를 뒤집으면 176~178° 가 나온다.
  · 프레임 3장(균등)에서 각각 추정해 소스 카메라로 world 로 올린 뒤 각도 trim 평균.
    `spread_deg`(프레임간 최대 각도차)가 신뢰도다 — GT 실측에서 spread<5° 인 206 chunk 는
    오차 p90 **1.64°**, spread≥5° 인 40 chunk 는 p90 **7.05°** 로 갈렸다.
  · `load_geocalib_gravity()` 는 `estimate_gravity` / `gt_trumans.gt_gravity` 와 **같은 키
    집합**을 돌려준다 — 하류(`graph_frame` 이하)가 소스를 몰라도 되게. `plane_d` 는 None
    (GeoCalib 은 지면 높이를 모른다). vista 경로는 `relations.ground_height`(G 프레임 z 2%
    분위수)가 지면을 다시 뽑으므로 상관없다.
  · 이미 구운 것: vista 72편, TRUMANS 246 chunk(검증용 `out_geocalib_trumans_check/`).
- **`--tau_ref {source,follow,auto}` — `track_*` 의 τ 를 추종 궤적 위의 *상대* 변위로 잰다
  (D97, 2026-09-01).** 새 파일 `lbm/presets.py: PRESET_TAU_REF / TAU_REF_CHOICES /
  resolve_tau_ref()`.
  · **왜.** `track_*` 는 subject 추종 offset `off(f)` 를 카메라 위치에 더한다. 그런데 τ 는
    소스 카메라 기준 절대 변위 `|p_plan(f) − p_src(f)| / z_med` 였으므로, 추종 성분이 τ 예산을
    통째로 먹고 preset 모양(`p_shape`)에 남는 몫이 없다. snowboard `hole_bank_k6_d94` 12행
    실측 — follow 순변위와 shape 순변위 사잇각 **145.6~146.1°**(거의 반대), `|follow|`
    3.70~3.75 u, `|shape|` 4.53~5.44 u 인데 둘이 상쇄돼 `|plan|` 은 2.53~3.15 u 다.
    "추종하며 왼쪽으로 트럭"이라 적어 놓고 world 에선 거의 안 움직인다.
  · **무엇이 바뀌나.** `tau_ref="follow"` 면 기준을 `p_ref(f) = p_start + off(f)` 로 두어
    τ = `|p_shape(f) − p_start| / z_med` 가 된다 (`off` 가 대수적으로 지워진다).
    구현은 `fit_tau` 를 안 고치고 `src_centers` 에 `basis_c2w[:3,3]` 상수 배열을 넘기는 것뿐.
    부수효과로 `tau0 = 0` 이라 `tau_saturated` 가 **원리적으로 못 뜬다**.
  · **행/뱅크에 남는 것**: `tau_ref`(resolve 된 값), `tau_ref_max`(그 기준에서 잰 τ).
    기존 `tau_max` 는 **계속 소스 기준**이다 — hole 예산·게이트·verify 가 쓰는 자다.
    `target_tau` 와 맞는 건 `tau_ref_max` 쪽이다.
  · 실측(snowboard `bank_d97`, dyn_0 `track_truck_left`, 5칸 사다리): 소스 기준 뱅크는
    0.10/0.20 칸이 `tau_saturated` 로 **드롭**돼 3칸만 남았는데, follow 기준은 5칸 전부 생존
    (`subject_in_frame` 1.00 / 0.80 / 0.60 / 0.40 / 0.40, `hole` 0.003~0.004).
    사다리 전체 60변이에서 `saturated 0`.
- **밀집 LOS — `trumans_scene_probe.py --los_samples / --los_bands / --los_mode`,
  `bank_to_blender_poses.py --los_samples / --los_bands / --min_los_frac` (D92, 2026-09-01).**
  기존 시선 판정은 조준점 **3점 OR** (`aim_points`) 이라 "몸 어딘가 한 점이라도 보이나"만 답한다
  — 반쯤 가린 구도가 `clear_frac 1.00` 으로 통과한다. LBM `occlusion_check`
  (`cinematographer_quality_worker.py:2541`) 처럼 **45점**을 쏴서 *몇 %가 보이나*를 잰다.
  `--los_samples 0` (기본) 이면 광선 수·JSON 키·판정이 **전부 예전과 같다** (실측: off/on 두 run 의
  공통 열이 비트 동일, off run 에 새 키 0개).
  · 표본은 LBM 처럼 OBB 격자가 아니라 subject **정점**에서 뽑는다 (`dense_aim_points`) — 높이
    5띠 × 띠마다 방위각 균등 9점. LBM 격자(`lbm_obb_points`, 원문 그대로 이식)는 표본이 상자 안
    **공기**에 앉을 수 있고, 광선이 아무것도 안 맞으면 LBM 은 그걸 "안 가려짐"으로 센다.
    a08 49프레임 2궤적 실측 — `miss_frac`(아무것도 안 맞은 광선 비율):
    OBB 격자 mean **0.272 / 0.356**, median 0.311 / 0.378, max 0.511 vs 정점 mean **0.008 / 0.013**.
    즉 LBM 은 45발 중 1/3 가까이를 공기에 쏘고 그 발이 분모에 남아, `occlusion_ratio` 임계
    0.10(human occluder)이 실효 ~0.15 로 느슨해진다. 가림 0인 ACCEPT 궤적에서 "보이는 비율"은
    정점 0.987 vs OBB 0.644 로 갈린다.
  · 새 열: 프레임별 `los_frac`/`occluded_frac`/`miss_frac`, path 요약 `min_los_frac`/
    `mean_los_frac`/`max_occluded_frac`/`max_miss_frac`. `occluded_frac` 은 LBM
    `occlusion_ratio` 와 **같은 정의**라 직접 비교된다. 후보 격자 경로엔 `los_frac`(최악 프레임).
  · `--min_los_frac 0`(기본)이면 열만 붙고 아무것도 안 거른다 — 분포를 먼저 보고 임계를 정하라고
    측정과 게이트를 두 손잡이로 나눴다. 실측 REJECT 궤적 `min_los_frac 0.00` / ACCEPT `0.956`.
- **`scene_graph/gt_trumans.py` — TRUMANS 전용 GT 중력축·GT 지면 (D88, 2026-08-31).**
  TRUMANS 클립은 우리가 Blender 로 직접 렌더한 것이라 카메라의 **blend world pose 가 디스크에
  남아 있다** (`render_a<NN>/cameras.json` 의 `c2w_opencv`). blend 씬은 z-up 이고
  `trumans_to_recon.py:745` 가 `inv(world[0]) @ world` 로 frame0 앵커만 거는 rigid 변환이므로
  recon world 중력축은 `up = R0[2, :]` 로 **닫힌 형태로 나온다** — RANSAC 도 fallback 도 필요 없다.
  지면은 `probe_a<NN>.json` 의 `floor_z` (사람 발 아래로 쏜 광선, `trumans_scene_probe.py:344-358`).
  · 코퍼스 191 chunk 실측 (`estimate_gravity` 대비 각도 오차): `ground_ransac` 142편은
    median **0.1°** / max 9.7° 로 사실상 정확하다. 문제는 **49편(26%)이 `camera_up_fallback` 으로
    떨어져 median 10.9° / p90 19.1° / max 26.2°** (59% 가 10° 초과) 기울어 있었다는 것 — 그 chunk
    의 뱅크 전량에 Dutch angle 이 박히고 OBB 도 같은 각도로 기울어 fit 된다.
  · `estimate_gravity` 와 **같은 키**(`up_world`/`plane_d`/`method`/`confidence`/`inlier_ratio`)를
    채워서 하류가 안 갈라진다. `resolve_dirs` 는 glob 이 2개 이상 물면 실패로 친다 — 조용히 다른
    recording 의 GT 를 집어오는 것보다 낫다.
- **`build_scene_graph.py --gravity_source {ransac,gt}` / `--ground_source {pointcloud,gt}` /
  `--gt_root` (D88, 2026-08-31).** 기본값이 `ransac`/`pointcloud` 라 **Vista4D 51편 기존 결과는
  비트 단위로 그대로**다. `gt` 인데 GT 를 못 찾으면 조용히 추정으로 안 떨어지고 assert 로 죽는다.
  `graph["gravity"]` 에 `estimated_method` / `estimated_error_deg` / `gt_source` /
  `floor_z_blend` 를 같이 실어 "이 chunk 가 예전에 얼마나 기울어 있었나"가 JSON 에 남는다.
- **`relations.build_relations(ground_z_override=...)` (D88, 2026-08-31).** 주면 점군 2% 분위수
  (`ground_height`) 대신 그 값을 쓴다. 안 주면 예전 경로 그대로.
- **`build_poses.py --keyframe_ease` 에 `arclen` / `arclen_kf` 추가, 기본값을 `arclen_kf` 로
  (D86, 2026-08-31).** 기존 `smoothstep` 은 ease 를 keyframe **구간마다 독립**으로 걸어서
  각속도가 keyframe 5곳에서 0 으로 떨어졌다가 구간 중앙에서 최대가 된다 — 이동은 등속인데
  회전만 펌핑한다. 배포 코퍼스 실측: keyframe 각속도 / 주변 10프레임 평균 = **0.316**
  (p10 0.210), 프레임간 회전각 max/median median 3.16×(Vista4D d77) / 3.70×(TRUMANS d77).
  · `arclen_kf` (신규 기본값) = 누적 호길이를 (keyframe 프레임, 누적각) 매듭 위 **PCHIP** 으로
    이어 각속도를 C1 으로 만든다. keyframe 은 제 프레임에 그대로 오므로 조준 정확도가 안 변한다
    — 실측 keyframe 각속도비 0.316 → **0.924**, `aim_err_med` 0.39° → **0.36°**.
  · `arclen` = 누적 호길이 위 **전역 smoothstep**. 각속도가 가장 평탄하지만(0.316 → 1.021,
    프레임간 회전각 max/med 5.31 → 1.33) keyframe 회전이 제 프레임을 떠나 `aim_err_med` 가
    0.39° → **5.17°** (p90 14.50°) 로 13배 커진다. `aim_err_max` 는 26.0 → 26.4° 로 사실상 동일.
  · `smoothstep` / `linear` 은 그대로 남아 D71~D84 코퍼스를 비트 단위로 재현한다.
  · `keyframe_ease="arclen"` 일 때 roll assert 는 `poses[frames]` 가 아니라 **keyframe 회전
    자체**(`keyframe_rolls`)에서 잰다 — 재타이밍 때문에 프레임에서 재면 무의미하다.
- **`render_pred_depth_warp.py` 에 `--labels` / `--no_labels` 쌍 (2026-08-31).**
  기본값은 `--labels` 로 예전 동작 그대로다. `--no_labels` 면 타일에 preset·hole·f1 텍스트를
  안 굽는다 — 궤적 자체를 눈으로 볼 때 라벨이 화면을 가려서 요청된 옵션이다.
- **`build_poses.py --keyframe_ease smooth_kf` + `--smooth_passes` / `--smooth_lambda`
  (D89, 2026-08-31).** D86 의 `arclen`/`arclen_kf` 는 **리타이밍**이라 궤적이 keyframe 을 지나는
  geodesic 折れ線 **위에 그대로** 있고 속도만 바뀐다. `smooth_kf` 는 선형 slerp 折れ線을 깐 뒤
  SO(3) Laplacian 으로 **모서리를 깎는다** — 折れ線을 떠나므로 성질이 다르다.
  `R_f <- R_f · exp((lam/2)·(log(R_fᵀR_{f-1}) + log(R_fᵀR_{f+1})))`, 양끝 고정, `passes` 회 반복.
  · GT 뱅크 714 변이 전량 실측 (half-hfov 35.75°). 열은 각속도 맥동비 `w_max/w_kf` · `jerk_p95` ·
    회전총량비 · smoothstep 대비 keyframe 회전 측지각 `kf_dev` (med/p90/max) · p90 을 half-hfov 로 나눈 값:
    ```
    smoothstep   25.03  0.5072  1.0000   0.000  0.000   0.00   0.0%
    arclen_kf    21.65  0.2176  1.0000   0.011  0.014   0.02   0.0%
    smooth p4     4.58  0.2073  0.9873   1.530  1.897   6.89   5.3%
    smooth p8     4.48  0.1290  0.9819   2.197  2.725   9.90   7.6%
    smooth p12    4.47  0.0911  0.9778   2.705  3.354  12.18   9.4%
    smooth p24    4.36  0.0530  0.9688   3.845  4.768  17.31  13.3%
    ```
    맥동비는 p4 에서 이미 포화하는데 조준 오차는 계속 커진다 — `passes` 는 p4~p8 이 효율 구간.
    회전총량비 < 1 은 모서리를 깎아 실제로 짧아진 것(p12 −2.2%).
  · 기본값은 안 바뀐다 (`build_poses.py` CLI 는 `arclen_kf`, `emit_bank.py` 는 뱅크 `fixed` 값).
    `smoothstep`/`linear`/`arclen`/`arclen_kf` 는 비트 단위로 그대로다.
  · roll assert 는 `arclen` 과 같은 이유로 `smooth_kf` 에서도 `keyframe_rolls` 에서 잰다.
- **`keyframe_info.aim_err_at_kf_deg` 지표 (D89, 2026-08-31).** keyframe **프레임에서의** 조준
  오차 최댓값. 기존 `aim_err_med`/`aim_err_max` 는 49프레임 전체라 "keyframe 을 정확히 통과하나"를
  못 가른다 — 리타이밍(0°)과 corner-cutting(2~4°)을 구분하는 열이다.
- **`emit_bank.py --smooth_passes` / `--smooth_lambda` (D89, 2026-08-31).**
  `--keyframe_ease smooth_kf` 일 때만 쓰인다. 다른 ease 에서는 아무 일도 안 한다.
  D87 과 같이 **위치는 재fit 없이 비트 단위 동일**하고 회전 스케줄만 바뀐다
  (GT 뱅크 714/714 실측 `pose 재현 최대오차 0.000e+00 [위치 전용]`).

### Fixed
- **`track_*` preset 의 위치 추종률과 조준 추종률이 따로 놀아 subject 가 프레임 밖으로 밀리던 것
  — `lbm/presets.py::PRESET_TRACKING` / `resolve_tracking()` + 소비처 4곳 (D93, 2026-09-01).**
  gain 이 두 개인데 한쪽만 켜져 있었다. `PRESET_FOLLOW` (`lbm/presets.py:250`) 는 **위치**
  추종률로 `track_*` 을 1.0 으로 올린다 (카메라가 subject 변위를 100% 따라간다). 그런데
  **조준점** 추종률은 `decode/build_poses.py:98` 의 `TRACKING_GAIN`
  (`world 0.0 / drift 0.6 / lock 1.0`) 이 따로 들고 있고, 뱅크 드라이버
  (`scripts/run_k6_d77_shard.sh`) 는 `--trackings` 를 안 넘겨 기본값 `drift`(60%) 로 갔다.
  위치 100% / 조준 60% 로 어긋난 채 49프레임 누적되면 조준점이 subject 뒤로 처진다.
  · **축 분리 실측** (`out/snowboard/tk_axis2x2`, τ 0.6 · follow_gain 1.0 · anchor `dyn_0` 고정,
    `--fixed_focal --sweep_deg 45`). `tracking` 만 바꾼 2×2 —
    `track_pull_out_arc_left`: lock `subject_in_frame` **1.00** / `subject_area_med` **0.067**
    vs drift **0.40** / **0.002**. `track_crane_up`: lock **1.00** / **0.122** vs drift
    **0.20** / **0.000**. 같은 표에서 `aim_keyframes` 3↔6 은 `path_len_u`/`view_angle_max_deg`
    가 소수점까지 동일(0.9671/41.7, 1.4085/59.2) — **궤적을 안 바꾼다.** 즉 단일 축이다.
  · **영향 범위** — 배포된 `out/*/hole_bank_k6_d77/bank.json` 52편 27,488행 중 track 3,282행,
    `subject_in_frame < 0.85` 인 행 **213행 (6.5%), 12편**. `fixed.tracking` 은 52편 전부 `drift`.
  · **`PRESET_FOLLOW` 와 달리 요청값을 덮어쓴다.** sentinel 이 없기 때문이다 — `follow_gain` 은
    문자열 `"0"` 이 "안 줌"을 뜻해서 명시값이 preset 기본값을 이기지만, `tracking` 은 뱅크 행이
    전부 `"drift"` 를 명시적으로 들고 있어 "안 준 경우"를 가릴 수가 없다. 대신 끄는 스위치
    `--no_preset_tracking` 을 둔다.
  · **기존 동작 보존 실측** — `--no_preset_tracking` 과 D93 이전 코드의 pose 최대차 `0.000e+00`.
    비-track preset 은 스위치가 켜져 있어도 `max|ΔP| = 0.000e+00` (`pull_out_arc_left`,
    `crane_up`, `orbit_left`, `truck_left`, `static_hold` 5종). track 계열만 바뀐다
    (`track_pull_out_arc_left` 9.183e-01, `track_crane_up` 1.254e+00, `track_orbit_left`
    1.850e+00, `track_hold` 1.147e+00).
  · **뱅크가 규약을 들고 다닌다** — `sample_camera_bank.py` 는 `preset_tracking` 을 bank.json 에
    적고, `emit_bank.py` 는 **키가 없으면 `False`** 로 읽는다 (D93 이전 뱅크를 되풀 때 그때 만든
    궤적이 그대로 나와야 한다). `variant_id` 에도 실제 쓰인 `tracking` 이 들어가고, 행에
    `tracking_requested` 를 같이 남겨 덮어썼는지 추적된다.
  · snowboard 전량 재생성(`bank_d93`, 166변이) 실측: track 156행 전부 `drift → lock`,
    비-track 10행(`static_hold*`) 그대로.
  · **범위 한계 — `aim="free"` preset 에는 안 듣는다.** `build_poses` 는
    `tracking_ignored = aim != "look_at"` 이라 D90 targetless preset 은 조준 gain 자체를 안 쓴다.
    snowboard 19 preset 중 **9개가 `aim="free"`** (`track_truck_left/right`,
    `track_pedestal_up/down`, `track_dolly_in/out`, `track_hold_dont_look`,
    `static_hold_dont_look`). 같은 설정 on/off A/B (`ab_d93_on` vs `ab_d93_off`,
    anchor `dyn_0`, τ 0.35) — `look_at` 계열은 뒤집히고 `free` 계열은 안 움직인다:
    `track_crane_up` `subject_in_frame` 0.20→**1.00** / area 0.0000→**0.1418**,
    `track_pull_out_arc_left` 0.40→**1.00** / 0.0000→**0.0867** vs
    `track_dolly_in` 1.00→1.00 / 0.1916→0.1910, `track_truck_left` 0.40→0.40 / 0.0783→0.0783,
    `track_truck_right` 1.00→1.00 / 0.1493→0.1492, `track_pedestal_up` 0.80→**0.60** /
    0.1111→0.1129. `path_len_u` 는 6종 전부 소수점까지 동일 — 위치는 안 건드린다.
  · 그래서 `track_truck_left` 는 D93 이후에도 **사다리 전 단·anchor 3개 전부**에서 무너진다
    (`hole_bank_k6_d93` 실측 `subject_in_frame` 0.00~0.08, `subject_area_med` 0.0000,
    `subject_visible_frac` 0.00, `view_angle_max_deg` 61~76). 옆으로 트럭하면서 조준을 안 고치면
    subject 가 옆으로 빠지는 게 당연한 결과다 — **`track_*` 인데 `aim="free"` 인 조합 자체가
    모순**이고, 이건 D93 이 아니라 별건으로 고쳐야 한다.
  · 소비처: `decode/build_poses.py`(`preset_tracking` 인자 + info 열 2개),
    `scripts/sample_camera_bank.py`, `scripts/fit_hole_ladder.py`, `scripts/emit_bank.py`.
- **뱅크를 읽는 쪽이 preset 이름을 안 풀어 라벨·CSV·코퍼스가 궤적과 다른 카메라를 가리키던 것
  — `lbm/presets.py::row_preset()` + 읽기 8곳 (2026-09-01).**
  쓰는 쪽(`sample_camera_bank.py`)은 늘 정식 이름을 적지만, **디스크에 이미 구워진 뱅크**는
  그때그때의 어휘로 적혀 있다. 읽는 쪽이 `row["preset"]` 을 그대로 쓰면 D76 에서 뜻이 뒤집힌
  `dolly_in`/`dolly_out` 과 D75/D90 별칭이 **그 행이 실제로 만든 카메라와 다른 이름**으로 나간다.
  캡션(`build_bank_captions.py:176`)은 이미 `resolve_preset` 을 거치고 있어서, 안 고치면
  **같은 행의 캡션과 라벨이 서로 다른 카메라를 가리킨다.**
  · 실측 어긋난 비율 — vista `hole_bank_k6` 51편 16,748행 중 **8,368행 (50.0%)**,
    vista `bank/` 52편 19,609행 중 **9,797행 (50.0%)**, trumans `hole_bank_k6_d77` 80편
    62,966행 중 **11,948행 (19.0%)**, trumans `bank_d77` 80편 29,586행 중 **5,442행 (18.4%)**.
    vista 쪽 7종은 `straight_ease`→`dolly_in`, `push_in_arc`→`push_in_arc_left`,
    `pull_out_arc`→`pull_out_arc_right`, `orbit_{left,right}_arc`→`orbit_{left,right}`,
    `rise_reveal`→`crane_up`, `drop_reveal`→`crane_down` (각 1,192~1,352행).
  · 새 헬퍼 `row_preset(row, raw=False)` — 행의 `aim` 을 같이 넘겨 `resolve_preset` 을 부른다.
    `raw=True` 면 적힌 문자열 그대로 (옛 산출물 재현용).
  · 고친 곳: `render_bank_videos.py` 타일 라벨(+`--raw_preset_names`, `--presets` 는 이제 적힌
    이름·정식 이름 **둘 다** 매칭) / `vista4d_bank_to_dl3dv.py` 코퍼스 `prompts.json`
    (`preset` 정식 + `preset_raw` + **`aim` 신규 export**) / `emit_bank.py` manifest /
    `audit_bank_geometry.py` `geometry.csv` (preset 별 집계가 이 열로 묶인다) /
    `bank_to_blender_poses.py` (`preset_canonical` 열 신규) / `render_pred_depth_warp.py`
    `load_entry_meta` (→ `render_preset_grid_warp.py`·`render_target_swap_warp.py` 도 전이적으로) /
    `render_target_cams_warp.py` (이 이름이 곧 렌더 파일 이름) / `merge_static_rung.py`.
  · **`merge_static_rung.py` 는 오탐이 아니라 실제 버그였다** — vista `bank/` 52편에
    `static_hold_locked`(aim=traj) 303행이 있는데 이 문자열은 `STATIC_PRESETS` 에 없고
    `static_hold_dont_look` 로 풀린다. 이름 그대로 비교하면 (a) stray 오탐으로 종료코드 3,
    (b) 멱등성 걷어내기가 그 행을 못 지워 병합 후 정지 단이 두 벌이 된다.
  · `bank_to_blender_poses.py` 는 **파일 이름·dedup 키를 일부러 적힌 이름 그대로 뒀다** —
    trumans 뱅크는 `dolly_in`(look_at) 과 `dolly_in_dont_look`(traj) 을 1,324행씩 둘 다 들고
    있고 정식 이름이 `dolly_in_look_at` 로 같아서, 정식 이름으로 묶으면 두 행이 한 슬롯으로
    합쳐지고 `<preset>.npz` 경로까지 충돌한다.
  · `audit_lite_framing.py` 는 후보에서 뺐다 — 그 `preset` 필드는 preset 이름이 아니라 tracking
    모드/kind (`drift`/`lbm_render`) 다.
  · 검증: 8개 스크립트 import + snowboard `hole_bank_k6` 4궤적 BEFORE/AFTER 재렌더 (같은 궤적,
    라벨만 `straight_ease`→`dolly_in`, `rise_reveal`→`crane_up`, `dolly_in`→`dolly_in_look_at`).
- **`smooth_passes` 가 fit 과 emit 사이에서 어긋나 회전만 조용히 달라지던 것 (D90, 2026-08-31).**
  `fit_hole_ladder.py` 는 `build_poses` 를 `smooth_passes` **없이** 불러 서명 기본값
  **12** (`build_poses.py:525`) 로 뱅크를 구웠는데, `emit_bank.py` 는 자기 CLI 기본값 **4**
  (`emit_bank.py:455`) 로 되만들었다. `smooth_passes` 를 소비하는 건 `build_poses.py:679` 의
  `smooth_kf` 가지 **하나뿐**이라, 기본 ease 가 `smoothstep` 이던 D89 까지는 이 불일치가 아무
  데도 안 나타났다 — `smooth_kf` 를 기본값으로 올린 순간 emit 이 `재구성한 궤적이 poses.npz 와
  다르다 (최대 7.753e-02)` 로 죽었다.
  · 증상이 원인을 안 가리킨 이유: **위치 오차가 정확히 `0.000e+00`** 이고 회전만 틀렸으며,
    그것도 `aim=look_at` 변이에서만 (40 변이 중 22개). 궤적·손잡이 경로가 전부 결백해 보였다.
  · 고침 3군데 — ⑴ `fit_hole_ladder.py` 에 `--smooth_passes`(기본 **12**) / `--smooth_lambda`
    (기본 0.5) 를 추가하고 `build_poses` 호출에 명시적으로 넘긴다. ⑵ 두 키를 `bank.json` 의
    `fixed` 블록에 싣는다 (`fixed` 는 "emit 이 fit 을 정확히 재현하는 데 필요한 값 전량"이
    설계인데 이 둘만 빠져 있었다). ⑶ `emit_bank.py` 는 이제 **뱅크 값을 먼저** 보고, CLI
    기본값을 `None` 으로 내려 명시적으로 준 경우에만 덮어쓴다.
  · `SHAPE_DEFAULTS` 에도 `smooth_passes: 12, smooth_lambda: 0.5` 를 넣었다 — 키가 없는
    **예전 뱅크**는 fit 이 인자를 안 넘겨 서명 기본값 12 로 구워졌으므로, 폴백은 `emit_bank`
    의 옛 CLI 기본값 4 가 아니라 12 여야 재현이 맞는다. 실측: 12 면 40/40 변이가 최대오차
    `0.000e+00`, 4 면 `8.034e-02`.
  · a08 chunk 재적합(234 변이) 후 emit 재현 최대오차 `0.000e+00` (234/234).
- **`render_pred_depth_warp.py` reel 누적을 `--reel` 일 때만 하도록 가드 (2026-08-31).**
  이전에는 `--no_reel` 이어도 전 entry 프레임을 메모리에 계속 쌓아 두었다. 689 entry 실행에서
  수십 GB 로 불어나 OOM 이 났다. 이제 `if args.reel:` 안에서만 append 한다. `--reel` 기본
  경로의 결과물은 그대로다.

### Changed
- **`presets.md` 를 "가능한 option + preset" 표로 재작성 (2026-08-31).**
  이전 문서는 D84 코퍼스에 **실현된 19종만** 적고 나머지는 산문 한 문단으로 뭉갰다. 어휘를
  넓히려면(D85) "지금 뭐가 있고 그중 뭐가 라우터에 안 걸리는가"가 한 표에 보여야 하는데,
  그게 안 보였다. 이제 `PRESETS` **40종 전량**을 `slot / aim / follow / needs_zoom / targetless /
  axis / D84 변이·영상 수` 열로 싣고, 라우터가 안 부르는 15종(+`--vertical_fallback` 의존 2종)이
  `slot=—` 로 드러난다. 집계: 구현 40 / caption 40 / 라우터 도달 23(+2) / D84 실현 19.
  option 쪽은 `route_presets.py` · `sample_camera_bank.py` · `fit_hole_ladder.py` ·
  `build_bank_captions.py` 네 스크립트의 손잡이를 **가능한 값 / 기본값 / D84 실제값** 3열로
  적어, D84 가 `speed`·`tracking`·`tau_ladder` 를 각각 한 값에 못박은 것이 표에서 바로 보인다.
  산문 절(§0 aim·target 설명, §3 dd 라벨 조합 47종, §4 anchor 통계)은 삭제 — 표에 안 들어가는
  서술이라 사용자 요청("가능한 option 과 preset 표만")에서 빠진다.
- **D86: `pan_*` 을 caption-targetless 로 (`targetless` 플래그) (2026-08-31).**
  `configs/caption_presets.json` 의 preset 에 `"targetless": true` 를 붙이면
  `build_bank_captions.py:caption_of` 가 `dd_*` 와 똑같이 `target`/`framing` 을 비운다
  (→ `target: none.`). `pan_left` / `pan_right` / `pan_right_zoom_out` 3종에 붙였다.
  근거는 실측 — pan 748 변이의 `subject_in_frame` median **0.50**, **46.7%** 가 0.5 미만이다.
  `aim="traj"` 이고 `track_` 도 아니라 궤적이 subject 에 매여 있지 않으므로, target 을 적으면
  "계속 보인다"가 지키지 못할 약속이 된다. 같은 `aim="traj"` 인 `track_truck_*` 는
  `follow_gain 1.0` 으로 실제로 subject 를 따라가 `sif` median **1.000** 이라 target 을 유지한다.
  그래서 판정을 규칙(`aim=="traj" and not track_`)이 아니라 **config 플래그**로 뒀다 — 규칙이면
  `truck_left`("...sliding sideways past **the subject**")까지 끌려가 문구가 가리킬 곳 없는
  지칭이 된다. `targetless` preset 의 phrase 에 `subject` 가 들어가면 assert 로 막는다.
  d84 코퍼스 재생성 실측: 266편 / 변이 12,193, **target 없음 6,578 (53.9%) / 있음 5,615 (46.1%)**
  (직전 5,798 / 47.8% 에서 pan 748 변이가 넘어옴). `presets.md` §0·§1·§6 도 같이 갱신.
- **D86: 프롬프트를 마침표로 끊고 target 없음을 `target: none.` 으로 명시 (2026-08-31).**
  `build_bank_captions.py` 의 `prompt_of` 가 두 가지를 바꾼다. ① 절 끝에 **마침표**를 찍는다 —
  이전 `target: person motion: the camera dollies ...` 는 값과 다음 필드 이름 사이에 경계가
  없어 `person motion` 이 한 명사구처럼 읽혔다. ② `NONE_FIELDS = ("target",)` 는 값이 비어도
  빠지지 않고 `none` 으로 남는다. free-moving(`dd_*`)이 코퍼스의 **47.8%** 라, 빈 값을 그냥
  빼면 프롬프트가 `motion:` 하나로 시작하는 **문장 구조 자체**가 "target 이 없다"의 지름길이
  된다. 나머지 필드(`event`/`framing`)는 종전대로 빈 값이면 빠진다.
  d84 코퍼스 재생성 실측: 266편 / 변이 **12,193**, `target: none.` **5,798 (47.8%)**.

### Added
- **D86: `models/Planner/CinemaTraj/presets.md` — 현재 쓰는 preset 목록 문서 (2026-08-31).**
  `PRESETS` 40종 중 라우팅이 **실제로 실현하는 19종**만, `aim`(look_at/traj) · 캡션 `target:`
  유무 · slot · 실측 지표(tau/hole/subject_in_frame/path_len/view_angle median)와 함께 적었다.
  DataDoP `dd_*` 188 shape 은 slot × move × angular **47 조합**으로 묶어서 표로. 뒤에 알려진
  구멍 5가지(tracking/speed 다양성 0, pan 의 sif 0.46~0.54, look_at 인데 안 보이는 346개,
  event 전량 공백, anchor-free 카메라 부재)를 붙였다.

### Fixed
- **D86: `configs/caption_presets.json` 에 `track_{push_in,pull_out}_arc_{left,right}` 4종 추가
  (2026-08-31).** D82 의 `TRACK_KEEP_SLOTS` 에 `arc` 가 들어가면서 라우터가 이 이름을 뱅크에
  넣기 시작했는데 캡션 config 에는 문구가 없어 `build_bank_captions.py` 의
  `assert spec is not None` 이 `track_pull_out_arc_right` 에서 죽었다 (d84 코퍼스 **774 변이**).
  `lbm/presets.py` 의 `PRESETS` 40종과 config 를 대조해 누락분 4종을 전부 채웠다.

### Added
- **D84: `route_presets.py --num_anchors N` — 한 씬에서 target 을 둘 이상 잡는다 (2026-08-31).**
  지금까지 라우터는 `pick_anchor()` 로 anchor 를 **딱 하나** 골랐다. 그러면 그 영상의 모든 변이가
  같은 `target:` 문장을 갖는다 — 캡션의 target 절이 씬 안에서 상수라 "무엇을 보느냐"의 학습
  신호가 0 이다. `pick_anchors(graph, min_area_frac, num)` 가 같은 정렬(움직이는 노드 우선 →
  `max_area_frac` → `num_visible_frames`)의 상위 `num` 개를 돌려주고, `main()` 이 anchor 마다
  따로 `route()` 를 부른다 (away side 도 track 여부도 그 노드의 성질에서 나와야 하므로).
  `sample_camera_bank` 는 (nodes × presets) 격자를 돌기 때문에 preset 은 **합집합**으로 넘긴다 —
  정지 anchor 에 `track_` 이 걸린 조합은 D77 이 비-track 쌍둥이와 bit-identical 이라 알아서 버린다.
  `route(..., node=...)` 인자가 추가됐고 `pick_anchor()` 는 `pick_anchors(...,1)` 의 옛 이름으로
  남는다(호출부 계약 유지). JSON 에 `anchor_ids` / `anchors[]`(anchor 별 slots·reasons) 추가,
  `--emit args` 는 노드 id 를 전부 찍고, 노드 목록만 보는 `--emit nodes` 를 넣었다.
  **`--num_anchors 1`(기본값)은 기존 출력과 동일**하다. dynpose 코퍼스 dry run(GPU 없이 라우팅만):
  274편 중 **272편 라우팅 성공, 2편 anchor 없음**(`0b975bcd-…`, `0f02bd2a-…`), 영상당 anchor
  **1.94개** / preset **16.42개** / track **4.68개**, (node × preset) 격자 합 **8749**,
  DataDoP shape **188/188** · label 조합 **47/47** 전부 소진.
- **D82: `route_presets.py --track_mode {add,replace,off}` — 한 씬 안에 track/비-track 공존
  (2026-08-31).** 지금까지 라우터는 anchor 가 움직이면 **모든 슬롯**에 `track_` 을 붙였다
  (= 새 `replace`). 그러면 한 영상의 preset 이 전부 추종이거나 전부 비-추종이라, 같은 씬·같은
  소스 위에서 "따라간다 / 안 따라간다"를 가르는 대조가 학습 데이터에 없다.
  `add`(새 기본값)는 추종이 궤적을 실제로 바꾸는 슬롯(`TRACK_KEEP_SLOTS = lateral, orbit,
  static`)만 track 으로 두고 나머지는 비-track 으로 남긴 뒤, `TRACK_BONUS_SLOTS
  = recede, advance, vertical` 중 하나를 **영상 id 해시**로 골라 track 을 하나 더 붙인다.
  `off` 은 `track_` 을 아예 안 쓰는 대조군.
  dynpose 267편 실측: 영상당 track 평균 **3.09개**, (track, 비-track) 분포는
  (3,4) 84편 / (3,5) 80편 / (4,4) 41편 / (2,5) 36편 / (4,3) 24편 — anchor 가 non-moving 인
  2편(0,7)/(0,8)을 빼면 **전 영상이 섞인다**. 보너스 슬롯은 advance 95 / recede 81 /
  vertical 43, 나머지 46편은 그 씬에 vertical 슬롯 자체가 없어 반영 안 됨
  (`reasons.track_bonus_dropped=true`, 그 영상은 track 이 하나 줄어든다).
  구현 주의점 둘 — ① 보너스 해시 seed 는 `<video>/<node_id>` 다. `node["id"]` 는 `dyn_0` 처럼
  **씬 안에서만 유일**해서 그것만 쓰면 코퍼스 전체가 한 슬롯으로 쏠린다(실측 확인).
  ② `track_` 짝이 실재하는 preset 만 접두사를 받는다 — `lbm/presets.py` 에 `track_pan_*` /
  `track_pull_out_arc_*` / `track_s_curve` 가 없으므로 rotate·arc 슬롯과, 좁은 `obs_az_span`
  에서 orbit 대신 나오는 `s_curve` 는 `track_mode` 와 무관하게 항상 비-track 이다. 그 목록은
  하드코딩하지 않고 `lbm.presets.PRESETS` 에서 직접 읽는다(preset 추가 시 조용히 어긋나지 않게).
  **`--track_mode replace` 는 기존 출력과 동일**하다 (267편 전량 대조). `reasons` 에
  `track_mode` / `track_bonus_slot` / `track_bonus_dropped` / `num_track` 을 실어
  `preset_route.json` 만 보고도 라우팅을 감사할 수 있다.
- **D81: `subject_visible_frac` / `subject_visible_min` — 뱅크에 G3 가림 열 (2026-08-31).**
  `sample_camera_bank.measure_trajectory(subject_points=...)` 가 프레임마다 **두 번 렌더**한다 —
  subject 점만 그린 실루엣(`alone`)과 전체 렌더의 depth 를 비교해 "그려졌어야 하는데 앞에 뭔가
  온" 픽셀을 센다 (`lbm.gates.evaluate:338-349` 와 같은 식, 여유 `0.02·S`). 실루엣이 비면 0.
  열은 프레임 median(`_frac`)과 최악 프레임(`_min`) 둘 다 — 책상 밑 shot 은 중간이 멀쩡해도
  바닥 근처에서 통째로 가린다. `sample_camera_bank` / `fit_hole_ladder` / `emit_bank` 3곳에 배선.
  WHY: 기존 `subject_area_med` 는 **그려진 실루엣의 화면 면적비**라 가림과 구분이 안 된다
  (실측 160 뱅크: `pedestal_down` 0.029 vs `orbit_left` 0.031 — 사실상 같다).
  `render.CloudRenderer.measure` 의 `num_subject_points` 판은 픽셀수를 점 개수로 나눈 **밀도**라
  비율이 아니다 (그 함수 docstring 이 직접 경고한다).
  **이분법에는 안 넘긴다** — 판정(verify) 패스에서만. 이분법이 푸는 답은 물리 게이트와 hole 이지
  가림이 아니고, 넘기면 렌더가 2배인데 `fit` 이 이미 런타임을 지배한다 (파일럿 fit 101s vs
  bank 31s). 실측(dynpose `100c897d`, anchor=plate, 식탁 위 접시):
  게이트 켠 기본 경로는 `pedestal_down` vis 0.911/min 0.886 (`binding=ground`) —
  **지면 게이트가 이미 막고 있다**. `--no_collision_free` 로 게이트를 끄면 같은 preset 이
  0.603/0.353(Δ0.35), 0.450/0.207(Δ0.5) 로 무너지는데 **같은 hole(0.618)의 `dolly_out` 은
  0.909/0.895 로 그대로**다 — 거리가 아니라 가림을 재고 있다는 뜻.
  `--no_subject_visible` 로 끄면 열이 빠지고 예전 뱅크와 bit-identical.

### Fixed
- **D85: DataDoP 외부 궤적의 비-직교 회전이 `det(R) != 1` 로 영상을 통째로 날렸다 (2026-08-31).**
  `lbm/presets.register_external_presets` 가 shape 의 `rel` 회전을 **JSON 에 실린 그대로** 썼다.
  그 값은 MonST3R pose 를 float32 로 저장하고 JSON 으로 한 번 더 돌린 추정치라 `det(R)` 가 1 에서
  **median 4.8e-7 / max 8.2e-7** 떠 있다 (188 shape **전량**). `decode/build_poses.py:581` 의
  마지막 검사는 `< 1e-9` 이라 500배 초과다. `_project_so3()` (SVD 극분해, `det=-1` 반사 방지로
  마지막 열 부호 보정)를 등록 시점에 한 번 걸어 고쳤다 — det 오차 8.2e-7 → **2.7e-15**,
  행렬 변화량은 최대 **4.6e-7** 이라 궤적 모양은 그대로고 `rel[0]` 은 정확히 I 를 유지한다.
  **왜 지금까지 안 터졌나**: `fit_tau` 의 `se3.scale_traj` 가 로그/지수를 거치며 *우연히*
  재직교화해 준다. 스케일이 1 이라 그 경로가 생략되는 변이에서만 터지므로 코퍼스 중간에
  산발적으로 터진다. 보간도 못 막는다 — DataDoP shape 은 **전부 m=49** 라 `_se3_interp` 가
  `num_frames=49` 에서 `rel.copy()` 로 그대로 통과시킨다.
  D84 코퍼스 실측: 첫 10편에서 3편 손실 (`frame 1` / `frame 10` 등), 고친 뒤 같은 영상
  (`023464b2-…`) 이 bank 단계 통과.
- **`build_bank_captions.py` — `prompt` 열 없는 `metadata.csv` 에서 `KeyError` (2026-08-31).**
  `load_events()` 가 행마다 `row["prompt"]` 를 무조건 읽었다. DynPose-LBM 의 `metadata.csv` 는
  `video,dynamic` 두 열뿐이고 영상 캡션 소스가 데이터 디렉토리 어디에도 없어서, dynpose 코퍼스
  캡션 생성이 첫 행에서 죽었다. `reader.fieldnames` 에 `prompt` 가 있을 때만 문장을 뽑고
  없으면 event 를 **빈 문자열**로 둔다 — `prompt_of` 가 빈 필드를 자동으로 떨어뜨리므로
  프롬프트에서 `event:` 절이 통째로 사라진다. 빈 dict 를 돌려주면 안 되는 이유는
  `--videos all` 이 이 dict 의 **key** 로 영상 목록을 만들기 때문이다 (목록은 채우고 문장만 비운다).

- **D81 argparse 기본값 뒤집힘.** `--subject_visible`(store_false) 를 `--no_subject_visible`
  **앞에** 선언했더니 argparse 가 **먼저 선언된 action 의 default** 를 쓰는 바람에
  (`store_false` 의 암묵 default 는 `True`) 새 열이 조용히 꺼진 채로 돌았다 — 스모크에서
  `vis=None` 으로 잡았다. 파일의 기존 관례(`--measure_behind` 쌍)대로 양수 플래그 +
  `default` 를 먼저, `--no_*` 를 `dest=` 로 뒤에 두는 순서로 고쳤다.

- **`CinemaTraj/scripts/render_pred_depth_warp.py --scores_csv` — caption F1 band 별 reel
  (2026-08-30).** latentcam eval 의 `preds_scores.csv` 에서 per-sample `captions/fscore` 를 읽어
  ① SOURCE 타일 라벨에 `f1 0.xxx` 를 박고 ② `index.json` 의 entry 마다 `caption_fscore` 를 남기고
  ③ `reel_f1_high.mp4` / `reel_f1_mid.mp4` / `reel_f1_low.mp4` 로 묶는다 (임계
  `--f1_high` 기본 0.8, `--f1_low` 기본 0.0). `--order fscore` 는 F1 내림차순 정렬.
  WHY: caption F1 은 프레임 라벨 189-class 의 weighted-F1 이라 **숫자만 봐선 왜 0 인지 안 보인다**
  — 같은 preset 인데 GT 는 라벨이 잡히고 pred 는 static 으로 떨어지는 식이라 warp 를 나란히
  놓고 봐야 판별된다. `--scores_csv` 를 안 주면 라벨·reel·`index.json` 전부 기존과 동일하다
  (anchor 별 reel 파일명만 `reel_target_<anchor>.mp4` 로 그대로 유지).
- **`CinemaTraj/scripts/caption_cameras_datadop.py --sets datadop` — DataDoP GT 원본을 같은
  태거에 태운다 (2026-08-30).** 지금까지 이 스크립트는 우리 카메라(`cameras` / `recon` /
  `latentcam`)만 읽었고, **대조군인 DataDoP GT 자체는 한 번도 태깅한 적이 없다**. 새 branch 는
  `DataDoP_valid.txt` 를 읽어 `DataDoP_with_scene/<scene>/<shot>_transforms_cleaning.json` 의
  120 pose 를 그대로 쓴다. 규약 주의점 하나 — 그 json 은 **이미 OpenGL c2w** 라
  `tag_trajectory(convert=False)` 로 넘긴다. 우리 npz 경로처럼 `[:3,1:3]*=-1` 을 또 걸면
  up/forward 부호가 되돌아가 라벨이 조용히 반대로 나온다. 부수 인자: `--datadop_root` /
  `--datadop_valid` / `--datadop_out` / `--datadop_limit` / `--datadop_seed`
  (seed 고정 서브샘플, 0 = valid 전량 28,971편).
  또 `--no_llm` 의 skip 기준을 `_caption.json` -> `_tag.json` 으로 바꿨다 — 태깅만 도는 경로에서
  caption 은 애초에 안 쓰이므로 예전에는 매 실행이 전량 재태깅이었다. **기본 인자만 주면 기존
  세 set 의 동작은 bit-identical** 이다.
- **`CinemaTraj/scripts/compare_camera_distributions.py` — 두 코퍼스 `_tag.json` 대조
  (2026-08-30).** 우리 뱅크 GT vs DataDoP GT 의 이동량 레벨(`step_median` / `total_translation`)
  · 산포(log10 sd) · 정지 비율(shot 전체 static / static 프레임 비중) · translation·angular
  어휘 점유를 한 표로 낸다. 표를 내기 **전에** 양쪽 `seg_kwargs` / `num_poses` / `gendop_root`
  가 같은지 검사해서 다르면 죽는다 (`--allow_mismatch` 로만 뚫림) — fps 는 velocity 눈금에
  그대로 곱해지고 vendored/pipeline GenDoP 은 `min_chunk_size` 가 10 vs 12 로 달라서, 노브가
  어긋나면 차이가 코퍼스가 아니라 태거에서 나온다 (`prdc-not-comparable-across-corpora` 와
  같은 실패 모드).
  게이지 보정 인자 `--a_divisor_subdir` / `--b_divisor_subdir` 도 같이 넣었다 — raw D=1 world
  단위는 코퍼스마다 게이지가 달라(우리는 DA3 frame0, DataDoP 은 MonST3R) 그대로 비교하면 안
  되고, 우리 쪽만 `<scene>/da3/avg_scale_context_first_cam/<name>.json` 으로 나눠야 학습이
  실제로 보는 눈금이 된다 (`scale_mode: avg_scale`). 그리고 `--floor`(기본 1e-6) — 우리 뱅크는
  `*_hold` preset 과 τ 사다리 바닥이 `step_median ~1e-19` 로 18.1% 를 차지해서 그냥 `sd(log10)`
  을 내면 4.66 이 나오는데, 그건 산포가 아니라 그 스파이크다. floor 위에서만 레벨·산포를 재고
  `below floor frac` / `trans-frozen shot frac` 을 따로 낸다 (우리는 [1e-6, 1e-4] 구간이 완전히
  비어 있고 DataDoP 은 최소값이 1.29e-6 이라 한 편도 안 잃는다).
- **`CinemaTraj/scripts/sample_camera_bank.py --skip_on_empty` — 변이 0 을 실패가 아니라
  skip 으로 끝낸다 (2026-08-30).** 이 스크립트는 `rows` 가 비면 `assert` 로 죽는데, 산출물이
  하나도 안 남으므로 대량 러너 입장에선 "아직 안 한 영상"과 구분되지 않는다. 그래서
  dynpose downstream 러너가 같은 8편을 매 pass 마다 graph→cloud→bank 로 다시 돌고 또 죽었다
  (13회 반복 확인). 플래그를 주면 `<bank_dir>/skipped.json`(`lbm_camera_bank_skipped_v1`,
  `reason:"no_surviving_anchors"`, 탈락 anchor 목록 포함)을 남기고 rc=0 으로 끝나 그 루프가
  끊긴다. **기본값은 예전 assert 그대로**라 TRUMANS D77 8샤드 동작은 변하지 않는다.
  dynpose 8편 전량에 적용해 확인 — anchor 1~6개가 전부 소스 frame 0 가시성/`--min_area_frac`
  에서 탈락한 경우였다.
- **`CinemaTraj/scripts/build_trumans_metadata.py` — TRUMANS 매니페스트 -> Vista 스키마
  `metadata.csv` (2026-08-30).** `build_bank_captions.py:load_events()` 는 `video` 와
  `prompt` **두 열만** 읽으므로, 매니페스트에서 그 두 열을 만들어 주면 캡션 빌더를 한 줄도
  안 고치고 TRUMANS 의 `event` 절을 채울 수 있다. 737편 `out/trumans_recon/*_s3f0k6/
  manifest_a*.json` 을 훑어 `A man is putting down the book with both hands indoors.`
  같은 문장을 만든다. 동사 -> 현재분사는 **26개 표**를 명시했다 — 자음중복 휴리스틱은
  `open` 을 `openning` 으로 만든다. 결과 737행 / 녹화 53편 / preset 8종 / 서로 다른 문장 170개.
- **`CinemaTraj/configs/trumans_labels.json` (`lbm_label_map_v1`) + `build_bank_captions.py
  --label_map` (2026-08-30).** TRUMANS 의 정적 노드 라벨은 .blend index pass 의 **원본 오브젝트
  이름**이라 `Floor.008` / `WallInner.021` / `283217/model` / `book_left_01` 이 그대로 학습
  프롬프트의 `target:` 절에 들어가고 있었다. 표는 `seg_instances_static` 737편 전수(키워드
  1,409 / stem 566)에서 뽑았다: 명명 stem 82종(occurrence 29,305)은 자연어 명사로,
  3D-FRONT 숫자 자산 ID 484종(occurrence 5,754 = 16.4%)은 로컬에 `model_info.json` 이
  없어 상위어 `furniture` 로 떨어뜨린다. `.NNN` / `_NNN` 접미는 조회 전에 벗긴다.
  파일럿 실측: `{Floor.008, 283217/model, book_left_01, WallInner.021, static_chair_03, ...}`
  -> `person 124 / wall 164 / furniture 246 / floor 82 / chair 82 / book 82`, unmapped 0.
  `--label_map` 을 안 주면 `normalize_label()` 이 `None` 을 받고 그대로 통과하므로 **Vista 는
  비트 단위로 같다** (실측: captions identical True, meta diff `{}`).
  뱅크 행의 `anchor_label` 은 추적 가능하도록 원래 이름 그대로 남는다.
- **`CinemaTraj/scripts/run_trumans_d77_shard.sh` (2026-08-30).** `run_k6_d77_shard.sh` 와
  다른 점은 **앞 두 단계가 더 있다**는 것뿐이다 — Vista 51편은 `scene_graph.json` /
  `cloud.npz` 를 예전에 만들어 뒀지만 TRUMANS 737편은 없다. 뒤 3단계
  (`sample_camera_bank` / `fit_hole_ladder` / `emit_bank`)는 인자까지 동일하다
  (`--aim_keyframes 6 --keyframe_aim auto --keyframe_ease smoothstep --fixed_focal`) —
  두 코퍼스 뱅크가 같은 규약이어야 섞어 학습할 수 있다. `--follow_gains` 를 안 주는 것도
  같다: `decode/build_poses.py:472` 는 gain 이 문자열 `"0"` 일 때만 `PRESET_FOLLOW` 를
  적용해서, `auto` 를 주면 `track_*` 이 조용히 추종을 멈춘다. 각 단계는 산출물이 있으면
  건너뛰고(재시작 안전), 실패한 영상은 다음 영상으로 넘어간다.

### Changed
- **`hole_bank_k7` -> `hole_bank_k6_d77`, `run_k7_shard.sh` -> `run_k6_d77_shard.sh`
  (2026-08-30).** `k<N>` 은 `--aim_keyframes N` 인데 두 뱅크 모두 `fixed.aim_keyframes` 가
  **6** 이다 (실측). `k7` 은 "keyframe 7" 로 읽히지만 실제로 바뀐 것은 preset pool 하나뿐
  (`bank/` -> `bank_d77/`, 20 -> 34종)이라 이름이 사실과 달랐다. Vista 뱅크 디렉토리 104개
  (`hole_bank_k7` 52 + `hole_bank_k7_static` 52)를 개명하고 `merge_static_rung.py` /
  `run_static_rung_shard.sh` / 러너 스크립트의 참조를 같이 고쳤다. TRUMANS 러너도 같은
  이름을 쓴다. 산출물 내용은 한 바이트도 안 바뀐다 — 경로 이름만이다.

### Fixed
- **`CinemaTraj/scripts/fit_hole_ladder.py` — 정지 preset 이 emit 뱅크에서 통째로 빠지던 것
  (`--static_rung`, 기본 켬, 2026-08-29, D78).** `fit_hole_ladder.py:436` 이 preset 목록을
  `[p for p in tau_bank["axes"]["presets"] if knob_kind(p)]` 로 만들었는데, `knob_kind()` 는
  `STATIC_PRESETS`(`static_hold`, `static_hold_dont_look`, `static_zoom_in`, `track_hold`,
  `track_hold_dont_look`)에 `None` 을 돌려준다. 사다리를 못 만든다는 이유로 preset **자체**를
  버린 것이라, τ 뱅크에는 있는 행이 emit 뱅크에는 **0 행**이었다 (camel k6 실측: τ 뱅크
  `static_hold` 273 / `track_hold` 76 행 → emit 0 / 0). 그래서 학습 캡션 어휘에서 "카메라가
  가만히 있는다"와 `track_hold`(움직임이 전부 follow offset 에서 나오는 **순수 추종**)가
  사라졌다. `track_hold` 는 특히 정지와 다른 궤적이다 — camel 실측 `path_u`
  `static_hold` 0.000 vs `track_hold` 0.064/0.052.
  고침: 정지 preset 을 rung **1개**로 통과시킨다 (`sample_camera_bank.py:429` 가 τ 뱅크에서
  이미 하던 것과 같은 예외). 손잡이 값은 τ 뱅크 행의 `target_tau` 를 그대로 실어
  `emit_bank.decision_from_variant` 가 **같은 결정**을 되만들게 하고, `knob_kind` 는 `None` 이
  아니라 문자열 `"none"` 으로 찍는다 (`None` 이면 CSV 에 `"None"` 으로 나가고 요약표의
  `:>7` 포맷이 TypeError). `--no_static_rung` 으로 예전 동작 복원.
  camel 검증: 14변이 / 182렌더, `emit_bank` `494 중 494 내보냄`, `pose 재현 최대오차
  0.000e+00`, `21<->49 왕복 최대오차 0.000e+00`, 14개 canonical 카메라 전부 유한
  (`translation_degenerate` 50행은 `static_hold*` 의 |t|=0 — NaN 아님).

### Added
- **`CinemaTraj/scripts/gendop_release_infer.py` — GenDoP `text_rgbd` ckpt 분기
  (`--cond_mode depth+image+text`, 2026-08-29, #111).** 텍스트에 더해 **생성 영상 frame0 의
  RGB + MonST3R depth** 를 조건으로 준다. `eval.py:311-313` 과 같이 `num_cond_tokens` 를
  77 → **591**(= 77 text + 257 image + 257 depth) 로 올리고, `eval.py:110/145` 의
  `standard_image`/`standard_depth`(BGR→RGB, /255, center-crop, zero-pad) 를 그대로 옮겨 적었다
  — GenDoP 리포는 여전히 **0줄 수정**. 새 인자 `--cond_mode` / `--gen_root` / `--monst3r_sub` /
  `--rgb_name` / `--depth_name` / `--kinds`. **기본값 `--cond_mode text` 는 기존 경로 그대로**
  (같은 인자로 두 번 돌려 c2w 완전 일치 확인; 186-entry 원본 런과 다른 건 entry 순서가 바뀌면
  샘플링 RNG 스트림이 달라지기 때문이지 코드 변경 때문이 아니다).
  - **RGB/depth 출처**: `eval_data/gen/<scene>/<name>/monst3r/{frame_0000.png,
    frame_depth_0000.npy}`. MonST3R 가 자기 입력 해상도(288×512)로 리사이즈해 쓴 프레임이라
    depth 와 픽셀 정렬이 이미 맞다 — mp4 에서 뽑으면 해상도가 어긋난다.
  - **depth 정규화는 하지 않는다**(실측 근거). `standard_depth` 는 center-crop/zero-pad 만 하고
    정규화 연산이 없다(`# [0, 1]` 주석은 코드로 뒷받침되지 않음). GenDoP 자체 예시
    `assets/examples/text_rgbd` 의 depth 범위 case1 0.2660~0.7221 / case2 0.0353~0.5065 와
    우리 MonST3R 출력 0.0438~0.6261 · 0.0820~0.6039 · 0.1437~0.8829 · 0.0795~0.5972 가 같은
    구간이다 — [0,1] 정규화도(case1 은 0/1 근처에 가지도 않는다) 미터 단위도 아닌, 양쪽 다
    같은 MonST3R 게이지. 그대로 넣는 게 맞다.
  - **실행**: eval `cameras` 114 entry 전량(`--kinds cameras`), GPU 2 / screen infer1,
    text_key `Concise Interaction`. written 114/114, degenerate **0**, no_rgbd 0.
    → `results/20260829_gendop_rgbd/text_rgbd/`.
  - **왕복 평가**(`gendop_release_eval.py`, 같은 DataDoP 분절 게이지):
    pooled P/R/F **0.4749 / 0.4377 / 0.4179**, mean fscore 0.4309, frame match 0.4377,
    degenerate 0 (114 entry, tag 없음 0).
  - HF 토큰 만료로 `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` 가 401→"Repository Not Found" 로
    떨어진다. 공개 리포이므로 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 을 붙이면 받아진다.
- **`CinemaTraj/scripts/caption_datadop_chunks.py` — DataDoP 원본 shot 을 49프레임 chunk 로 잘라
  태깅만 해서 우리 plan-text 형식으로 낸다 (2026-08-29).** DataDoP 는 카메라 궤적만 있고 물체
  anchor 가 없으므로 `target` 은 **`none`** 고정, `motion` 만 GenDoP 분절기 어휘로 채운다
  (`CAM_INDEX_TO_PATTERN` 27종 × `ANG_INDEX_TO_PATTERN` 7종 →
  `move left and yaw right, then pitch up, then static`). LLM/VLM 을 한 번도 안 부르고 GenDoP
  리포는 0줄 수정한다. 출력은 `lbm_bank_captions_v1` 그대로라
  `prompt = "target: none motion: <...>"` 로 뱅크 캡션과 같은 형식이다.
  규약 3종과 그 실측 근거:
  ① `_transforms_cleaning.json` 은 **이미 OpenGL c2w** 라 `tag_trajectory(..., convert=False)`
  로 `[:3,1:3]*=-1` flip 을 끈다. 안 끄면 yaw 부호가 뒤집힌다 — `1_0001/shot_0102` 실측
  `convert=False` "move left and **yaw right**" vs `convert=True` "yaw left" 인데, DataDoP 자신의
  `_caption.json` `Movement` 는 "moving left while **yawing right**" 라 `False` 가 맞다.
  ② **리샘플 안 함** (`--num_poses 0`). 49프레임을 120 으로 늘리면 프레임당 이동량이 (48/119)
  배로 줄어 `cam_static_threshold=0.02` 가 훨씬 많은 구간을 static 으로 찍는다. 원본 pose 를
  그대로 쓰면 속도 눈금이 DataDoP 와 동일하고, 실제로 라벨 분포가 보존된다 — 40 shot 대조:
  전체 120프레임 `move static` 48.6% / `angular static` 75.2% vs 49프레임 chunk 49.0% / 76.2%.
  ③ combine 단계 window 는 못 바꾼다 (`segmentation.py:392-403` 이 인자와 무관하게 15/10 으로
  덮어쓰고 chunk 4개 이하가 될 때까지 5씩 키운다). 48 velocity 샘플 기준 chunk 는 1~4개.
  chunk 자르기는 `--chunks_per_shot 3` = `linspace(0, 120-49, 3)` → 시작 0/36/71 로 120 pose 를
  빠짐없이 덮는다 (stride 36 짜리 `range` 는 71 을 못 만들어 뒤 22프레임이 버려진다).
  전량 실행 (8샤드, GenDoP env, CPU only, 샤드당 220s): **22,314 shot → 66,942 entry**, 건너뛴
  shot 0. 고유 `motion` 문자열 10,957개, `static` 단독 26.6%, 상위는 `move forward` 6.0% /
  `move backward` 2.6% / `move forward, then static` 2.1%.
  산출물 `CinemaTraj/out/datadop_chunk_captions/captions.json` (48 MB) + 샤드 8개.
- **`CinemaTraj/scripts/caption_cameras_datadop.py` — `tag_trajectory(..., convert=)` 분기
  (2026-08-29).** 입력이 이미 DataDoP(OpenGL) c2w 일 때 flip 을 끄기 위한 것. 기본값 `True` 는
  기존 npz(OpenCV c2w) 경로라 예전 동작과 bit-identical.
- **`CinemaTraj/scripts/merge_static_rung.py` — 정지 rung 전용 뱅크를 기존 hole 뱅크에 붙인다
  (2026-08-29, D78).** D78 을 반영해 52편을 **전량 재fit** 하면 편당 9~44분이 다시 든다.
  정지 rung 은 이분법이 없어 (anchor × 4 preset) × 렌더 2회뿐이라 편당 2분 미만이므로
  (camel 실측 14변이/182렌더), 정지 preset 만 `hole_bank_k7_static` 으로 따로 적합하고 여기서
  붙인다. 두 fit 은 anchor·preset 루프가 독립이고 같은 τ 뱅크·같은 인자를 쓰므로 붙인 결과가
  전량 재fit 결과와 같은데, 그 전제를 **붙이기 전에 검사한다**: `fixed` 블록 13키
  (`speed`/`tracking`/`look_at_bias`/`start_mode`/`aim_anchor`/`aim_ramp_frames`/
  `orbit_span_frac`/`min_sweep_deg`/`traj_basis`/`aim_keyframes`/`keyframe_aim`/
  `keyframe_ease`/`fixed_focal`) + 스칼라 4키(`video`/`num_frames`/`S`/`z_med`) +
  `source_bank` 가 어긋나면 붙이지 않고 rc=2. 정지 preset 이 아닌 행이 섞여 있으면 rc=3.
  멱등하다 (이미 붙은 정지 행을 먼저 걷어낸다). `bank.json` 에 `static_rung_merged` 표식을
  남기고 `bank.csv` 열 목록은 **기존 csv 헤더에서 읽는다** — 여기 따로 적으면
  `fit_hole_ladder` 의 `columns` 와 어긋나는 순간 조용히 열이 밀린다. `--dry_run` 있음.
- **`CinemaTraj/scripts/run_static_rung_shard.sh` — D78 정지 rung 샤드 러너 (2026-08-29).**
  정지 preset 4종만 적합 → `merge_static_rung.py` → `emit_bank.py` 를 한 번에. 인자는
  `run_k7_shard.sh` 와 **똑같이** 준다 (`--aim_keyframes 6 --keyframe_aim auto
  --keyframe_ease smoothstep --fixed_focal`, `--follow_gains` 미지정) — 다르면 위 호환성
  검사에 걸려 안 붙는다. `bank.json` 의 `static_rung_merged` 표식으로 재시작 안전.
  (`static_zoom_in` 은 `STATIC_PRESETS` 지만 `usable_presets(allow_zoom)` 가 τ 뱅크에서 이미
  빼므로 대상이 아니다 — `emit_model_cams.py` 에 intrinsics 채널이 없다.)
- **`CinemaTraj/scripts/sample_camera_bank.py` — `--track_dynamic_only` (기본 켬, 2026-08-29, D77).**
  `track_*` 12종을 **움직이는 anchor** 에만 건다. 근거: `track_*` 의 유일한 차이는
  `PRESET_FOLLOW` 가 켜는 follow_gain 1.0 이고, 그건 anchor 의 world 변위를 카메라 위치에
  더하는 것뿐이다 — 안 움직이는 anchor 는 변위가 0 이라 궤적이 비-track 짝과 **비트 단위로
  같아진다**. 그대로 두면 "같은 카메라 / 다른 캡션" 쌍이 대량 생겨 텍스트 조건이 오염된다.
  판정은 `node["moving"]`(graph 가 이미 실측한 플래그). `path_len_u` 를 안 쓰는 이유는 `stat`
  노드의 OBB 중심이 재적합 지터로 **21.0 u** 까지 흔들려 실제 이동과 구분이 안 되기 때문이다.
  실측(53 씬 484 노드): `moving=True` 124개(전부 `kind="dyn"`), `dyn` 인데 안 움직이는 것 52개,
  `stat` 308개. 씬당 moving anchor 평균 2.34 / 중앙값 2, 0개인 씬이 10개.
  `--no_track_dynamic_only` 로 예전 동작(전 anchor × 전 preset) 복원. 닫는 요약표에
  잘려나간 (anchor, preset) 쌍 수를 찍는다. camel 스모크: `dyn_0/dyn_1` 만 track 52행씩,
  `stat_0/1/2` 는 0 (36쌍 드롭), track 행 `follow_gain=1.0` / 비-track 행 `0.0` 확인.
- **`CinemaTraj/scripts/fit_hole_ladder.py` — `--tau_bank_dir` (기본 `bank`).** τ 뱅크 경로가
  `out/<video>/bank/bank.json` 로 하드코딩돼 있어서 preset 축이 바뀐 새 τ 뱅크를 만들면 k6
  소스를 덮어써야 했다. 기본값이 예전 경로 그대로라 기존 실행은 무변경. `bank.json` 의
  `source_bank` 필드도 실제 읽은 폴더를 적는다.
- **`CinemaTraj/scripts/run_k7_shard.sh` — D77 뱅크(34 preset) 샤드 러너.** τ 뱅크(`bank_d77`)
  → `fit_hole_ladder` → `emit_bank` 를 한 번에 돌린다. `run_k6_shard.sh` 와 달리 τ 뱅크부터
  다시 만든다 — `fit_hole_ladder.py:520` 이 τ 뱅크에 seed 행이 없는 (anchor, preset) 쌍을
  조용히 건너뛰므로 `--presets` 로 넘겨도 `track_*` 이 안 나온다. `--follow_gains` 를 **안 준다**
  (기본 `"0"`): `decode/build_poses.py:472` 는 들어온 gain 이 문자열 `"0"` 일 때만
  `PRESET_FOLLOW` 를 적용하고, `auto` 를 주면 τ 최소 gain 이 풀려 `track_*` 이 조용히 추종을
  멈춘다(캡션만 "tracks" 인 궤적).

### Changed
- **`CinemaTraj/lbm/presets.py` — 조준 표기를 비대칭으로, `_aimed`/`_locked` → 무표기/`_dont_look`
  (2026-08-28, D76).** 사용자 지시("LAMP DSL 참고해봐, aimed/locked 가 좀 별로인 것 같은데").
  **바로 아래 항목(D75)의 `_aimed`/`_locked` 대칭 표기를 대체한다.** 규칙은
  `[track_]<primitive>_<direction>[_<primitive2>_<direction2>][_dont_look]` — 조준
  (`aim="look_at"`)이 **기본값이라 이름에 안 적고**, 거기서 벗어나는 쪽에만 `_dont_look` 을
  붙인다. 어휘가 학습 캡션에 그대로 나가는데 36개 중 5개가 "기본 동작"을 이름에 적고 있었다.
  바뀐 10종: `dolly_in/out_aimed`→`dolly_in/out`, `dolly_in/out_locked`→`dolly_in/out_dont_look`,
  `static_hold_aimed`→`static_hold`, `static_hold_locked`→`static_hold_dont_look`,
  `track_hold_aimed`→`track_hold`, `track_hold_locked`→`track_hold_dont_look`,
  `track_dolly_in/out_aimed`→`track_dolly_in/out`.
  ⚠ **`dolly_in`/`dolly_out` 은 뜻이 뒤집힌 유일한 두 이름이다** — D75 에서 `_locked`
  (`aim="traj"`)의 별칭이었는데 지금은 정식 이름(`aim="look_at"`)이다. 디스크의 옛 뱅크가 옛
  뜻으로 이 문자열을 쓴다(실측: `out`+`out_dynpose` 336 파일 105,329 `variants[]` 행 중
  `("dolly_in","traj") 845` / `("dolly_out","traj") 846`, `aim="look_at"` 인 건 0). **뱅크
  파일은 하나도 안 고쳤다** — `fit_hole_ladder.py` 가 `out_dynpose` 를 실행 중이라 rewrite 는
  경합이다. 대신 `resolve_preset(name, aim)` + `LEGACY_AIM_COLLISIONS` 가 행이 기록한 `aim` 을
  보고 옛 뜻으로 되돌린다. 모든 행이 `aim` 을 명시하므로 정보 손실이 없다. **뱅크 행을 읽는
  코드는 `aim` 을 반드시 같이 넘길 것** — `build_bank_captions.py` 에 이름↔행 `aim` 불일치
  assert 를 넣어 틀린 매핑이 캡션으로 새는 걸 막았다.
  같이 맞춘 것: `configs/caption_presets.json`(키 10종 + `note_naming`/`note_naming_collision`),
  `lbm/prompts/system_traj.md`(VLM 어휘), `verify.py`/`decode/emit.py`/`decode/build_poses.py`/
  `scripts/ablate_vlm_hole_perception.py` 주석, `README.md`(preset 표 두 개를 36종 실측값으로
  재생성 — 종전 표는 D75 의 `track_*` 12종이 빠진 23종이었다), `DECISIONS.md` D76.
  `scripts/expand_preset_variants.py` 는 `PRESET_CLASS` 에 새 키만 **추가**(옛 키 유지)하고
  `SHAPE_PRESETS`/`STATIC_PRESETS` 는 **안 건드렸다** — 그 문자열들은 우리 어휘가 아니라 원본
  LBM `video_runtime.PRESET_NAMES` 로 나가 `build_trajectory_plan` 이 읽는다.
  검증: alias 33종 전부 `PRESETS` 로 resolve + alias 키/정식 이름 충돌 0, D75 10종의 `aim`
  보존, `LEGACY_AIM_COLLISIONS` 양방향, `PRESETS` 36 : `caption_presets.json` 36 일치,
  디스크 105,329 행 resolve 시 미지 preset 0 / 캡션 누락 0 / **aim 불일치 0**,
  camel·goat·parkour·snowboard `hole_bank_k6` 캡션 **1,180개 재생성 → 전부 bit-identical**.
- **`CinemaTraj/lbm/presets.py` — preset 이름을 하나의 규칙으로 정리, 옛 이름은 전부 alias
  (2026-08-28).** 사용자 지시. 규칙은
  `[track_]<primitive>_<direction>[_<primitive2>_<direction2>][_aimed|_locked]` 이고,
  `_aimed`/`_locked` 는 **같은 모양이 두 aim 으로 다 존재할 때만** 붙는다 (dolly, hold) —
  나머지는 aim 이 primitive 로 정해지므로(pan/truck/pedestal=traj, arc/orbit/crane/s_curve=look_at)
  `truck_left` 처럼 접미사가 없다. 고친 것: `straight_ease` → `dolly_in_aimed` ("ease" 는 speed
  축 어휘였다), `dolly_in/out` → `dolly_in/out_locked`, `orbit_left/right_arc` →
  `orbit_left/right` (`T.true_orbit` 이라 `_arc` 가 틀린 말이었다), `rise_reveal`/`drop_reveal` →
  `crane_up`/`crane_down` ("reveal" 은 의도지 동작이 아니다), `push_in_arc`/`pull_out_arc` →
  `push_in_arc_left`/`pull_out_arc_right` (4칸 중 2칸만 방향이 적혀 있었다),
  `zoom_out_pan_right` → `pan_right_zoom_out` (합성은 `<주동작>_<부동작>` 순),
  `static_hold` → `static_hold_aimed`, `static_subtle_zoom` → `static_zoom_in` ("subtle" 은
  바로 위 항목에서 캡션에서 뺀 크기 부사다), `track_follow*` → `track_hold_*`,
  `track_side_*` → `track_truck_*`, `track_push_in`/`track_pull_out` →
  `track_dolly_in/out_aimed`, `track_rise`/`track_drop` → `track_crane_up`/`track_crane_down`.
  결과적으로 **`track_X` 는 전부 `track_` + 비-track 이름**이 된다 (예외는 hold 하나로, 비-track
  쪽이 `static_`(안 움직인다) / track 쪽이 `track_`(subject 와 같이 움직인다)).
  **옛 이름 20종은 `PRESET_ALIASES` 에 의미 그대로 남아 있고**(`resolve_preset()` 신설,
  `build_shape`/`focal_track`/`build_bank_captions.py` 가 조회 전에 통과시킨다) 궤적이 **비트
  동일**함을 검증했다 — 배포 뱅크 51편 16,748 변이와 기존 `decision.json`·`captions.json` 은
  한 톨도 안 바뀐다 (camel `hole_bank_k6` 280 변이 캡션 재생성이 같은 문장을 냈다).
  `configs/caption_presets.json` 키도 새 이름으로 바꾸고(36종, PRESETS 와 누락/잉여 0 검증),
  `lbm/prompts/system_traj.md` 의 VLM 어휘 목록과 `loop.py`/`build_decision_fallback.py` 기본값
  (`orbit_left`), `sample_camera_bank.py:ROTATION_ONLY_PRESETS`, `expand_preset_variants.py` 의
  클래스 표(옛 키 유지 + 새 키 추가)를 같이 맞췄다.

### Added
- **`CinemaTraj/scripts/{build_bank_captions,vista4d_bank_to_dl3dv}.py` — 캡션 파일 이름을
  인자로 (2026-08-28).** `build_bank_captions.py --out_name`, `vista4d_bank_to_dl3dv.py
  --captions_name`. 둘 다 기본값이 예전 하드코딩 이름(`captions.json`)이라 안 주면 동작이
  그대로다. **왜**: 같은 뱅크(=같은 궤적) 위에 프롬프트만 다른 대조군을 얹기 위해서다 —
  snowboard `hole_bank_k6` 를 `target:` 절 있는 판/없는 판 두 갈래로 내보냈고, 두 데이터셋의
  `da3/target_poses.npz` (77,49,4,4) 는 비트 동일하고 `prompts.json` 텍스트만 다르다.
  (주의: `vista4d_bank_to_dl3dv.py main()` 은 처리한 영상만으로 `meta_vista4d.csv` 와
  `seg_list_vista4d_{train,test}.txt` 를 **다시 쓴다** — 1편만 돌릴 때 공용 root 를 주면
  51편 split 이 날아간다. 반드시 별도 `--out_root`.)
- **`CinemaTraj` tracking preset 에 수직축 4칸 (2026-08-28).** 사용자 지적. 비-track 어휘에는
  `pedestal_up/down`(순수 수직, aim="traj")과 `crane_up/down`(재조준, aim="look_at")이 위/아래
  짝을 다 갖고 있는데 track 쪽은 crane-up 하나뿐이었다. `track_pedestal_up`,
  `track_pedestal_down`, `track_crane_up`, `track_crane_down` 을 추가해 tracking preset 이
  9종 → **12종**이 됐다 (`PRESET_FOLLOW` 는 이제 `track_` 접두어로 자동 채워진다).
  camel `dyn_0` τ 0.20 실측에서 4종 모두 `gain 1.00 / side-tracking / τ 0.1966~0.1994` 로
  붙었고 비-track preset 은 전부 `gain 0.00` 그대로다. **하강 2종은 위 짝보다 게이트 탈락이
  잦을 수 있다** — micro-adjust 의 elevation clamp [5°, 70°] 와 "지면 아래 금지"(LBM-Lite 14)가
  하강에만 걸린다. 탈락률은 실제 뱅크로 재야 하고 아직 안 쟀다.

### Fixed
- **DIRECTOR 파일럿이 scene graph G frame 을 world 로 착각했다 — 파일럿 산출물 전량 무효
  (2026-08-27).** `scene_graph.json` 의 노드 좌표(`track.center_smooth`, `obb.center`)는 G
  frame(up=+z, 스케일 1/S)인데, `run_director_vista.py:world_to_et` 는 그 위에 **world 축**
  (`gravity.up_world`, 상수 `fwd=[0,0,1]`)을 물렸고 `render_director_depth.py` 는 npz 의 pose 를
  world c2w 로 알고 렌더러에 그대로 넘겼다. 축 오차는 up **93.75°**(camel) / **101.28°**
  (snowboard), fwd 93.67° / 102.14°. 그 결과 subject OBB center 가 **49프레임 전부 카메라 뒤
  z<0** 로 떨어졌다. `fwd=[0,0,1]` 의 근거였던 주석 "cam_c2w[0]=I" 도 거짓이다 (snowboard
  |t0|=**0.8104**, rot 6.18°). 이 버그는 **hole/z_p50 표에 안 나타난다** — 렌더는 매 프레임
  성공한다. 수정: `graph_basis()` 가 G 에서 up=+z 를 쓰고 시선은 실제 frame0 c2w 열2 를 `R_gw`
  로 G 에 옮겨 쓰고, `g_to_world(c2w_g, T_wg, R_wg)` 가 위치엔 `T_wg`, 회전엔
  `R_wg = T_wg[:3,:3]/S` 만 적용한다 (`R_wg @ R_wg.T ≈ I` assert). npz 포맷을
  `director_pilot_v2` 로 올리고 키를 `cam_t_g`/`char_g` 로 바꿔 **프레임을 이름에 박았다** —
  렌더러는 `cam_t_g` 가 없는 v1 npz 를 거절한다. 옛 동작은 `--et_basis legacy` 로 보존.
  검증(snowboard, rel 좌표계): `frac(시선각<25°)` 가 수정 전 8 track 전부 0.00~0.33 이었는데
  수정 후 path·scene 게이지는 8/8 track 이 1.00 이다. 무효화된 산출물
  (`results/20260827_director_{camel,snowboard}/*`)은 지우지 않고 남겼다. 상세는 `FIX.log` FIX-13.
- **`CinemaTraj/scripts/trumans_to_recon.py` — `renders_match` 가 엔진까지 대조한다
  (2026-08-27).** pose 만 보던 탓에 EEVEE 로 찍어둔 렌더가 `--rgb_engine cycles` 재실행에서
  **조용히 재사용**됐다. 흰 번짐이 그대로 남는데 로그에는 "렌더 재사용" 한 줄만 찍혀서 눈치챌
  방법이 없었다 (뱅크에 이미 EEVEE 로 찍힌 clip 이 281편 있다). 이제 `render_meta.json` 의
  `rgb_engine` 이 요청 엔진과 다르면 재렌더하고, **meta 가 없는 옛 렌더는 EEVEE 로 간주**해
  cycles 요청 시 재사용을 막는다. 기본값 `eevee` 경로의 판정은 전과 동일하다.
- **`CinemaTraj/scripts/vista4d_bank_to_dl3dv.py` — 사다리 붕괴 중복 변이를 걸러낸다
  (2026-08-27).** hole 사다리 4단은 게이트(obb/ground/approach/elev/shape/collision)가 물리면
  같은 knob 에서 멈춰 **4단이 같은 궤적**이 된다. k6 뱅크 47편 실측에서 15624 변이 중
  **7804(50%)** 가 중복이었고 그중 99.4% 는 pose 배열이 비트 단위로 동일했다 (나머지도 위치 차
  ≤ 0.025 u). 그대로 내보내면 학습이 같은 (pose, text) 쌍을 두 번 보고, 게이트가 잘 물리는 좁은
  씬이 그만큼 과대표집된다. 판정은 `path_len` 같은 대리값이 아니라 **pose 배열 자체**로 해서
  임계값을 없앴고, 남기는 건 사다리 아랫단이다. 요약표에 `dup` 열과 총계를 찍어 개수가 조용히
  줄지 않게 했다. `--no_dedup` 으로 예전처럼 전량 내보낸다.
- **`CinemaTraj/scripts/trumans_lite_bank.py` — 열거 실패 recording 을 건너뛴다 (2026-08-27).**
  `list_actions` 는 recording 당 Blender 를 여러 번 띄워 63편이면 ~1시간인데, `assert` 하나가
  터지면 그때까지의 열거를 통째로 버리고 뱅크가 **0편**으로 끝났다 (`a2e8ba09` 가
  `pick_sequence` 에서 죽어 앞선 35편이 날아감). 이제 실패한 편만 빼고 계속하며 stdout `SKIP`
  줄 + manifest `skipped_recordings` 에 사유를 남긴다. 전멸하면 (`jobs` 가 비면) 그때는 죽는다 —
  조용히 0편을 만들지 않기 위해서. `--no_skip_bad_recordings` 로 예전 동작 복구.
  최종 요약 줄에 `recording N/M` 과 스킵 건수를 같이 찍어 개수가 조용히 줄지 않게 했다.
- **`CinemaTraj/decode/build_poses.py` — keyframe roll assert 허용치를 도 단위에 맞췄다
  (2026-08-27).** `roll_about()` 은 **도**를 돌려주는데 두 assert 의 허용치가 `1e-6` = 1.7e-8 rad
  이라 `arctan2` 로는 도달 불가능한 정밀도였다. k6 뱅크 52편 중 `parkour` 가 잔차 **1.14e-6 deg**
  로 죽었다 (4K 폭에서 지평선 0.00004 px). `ROLL_TOL_DEG = 1e-4` 상수를 두고 keyframe 분기와
  `look_at` 분기가 공유한다. **포즈 값은 한 비트도 안 바뀌고 검사 임계만 바뀐다** — 눈에 보이는
  roll(>=0.01 deg)은 여전히 전부 잡힌다. 상세는 `FIX.log` 2026-08-27 항목.

### Changed
- **`CinemaTraj/scripts/build_bank_captions.py` — motion 문장에서 크기 부사를 기본 뺐다
  (2026-08-28).** 사용자 지시. `the camera significantly dollies straight forward toward the
  subject` → `the camera dollies straight forward toward the subject`. 부사(barely / slightly /
  steadily / significantly / dramatically)는 τ 사다리 단을 텍스트로 노출하는 것인데, τ = |t|/z_med
  라 **같은 부사가 씬마다 다른 실제 이동량**을 가리킨다 — 어휘를 preset 하나로 좁혀야 텍스트↔궤적
  대응이 1:1 이 된다. 사다리 정보는 뱅크 행(`tau_max`)에 그대로 남아 있다. 예전 문장은
  `--magnitude` 로 비트 단위 재현된다 (camel 실측 확인). 부사가 빠지면서 motion 어휘가
  14 preset × 5 부사 = **69종 → preset 32종**이 된다. `captions.json` 에 `magnitude` 플래그를
  기록하고, `track_*` preset 인데 `follow_gain` 이 0 인 변이는 요약표에 ⚠ 로 센다 (캡션이
  "tracks" 라고 말했는데 궤적은 추종이 아닌 경우를 조용히 넘기지 않게).
- **Vista 카메라 캡션 `cam_static_threshold` 0.02 → 0.05 (2026-08-28).** GT 태그 16편
  캘리브레이션(dance-twirl 은 zoom=focal 축이라 제외): 분절기 속도 눈금 `fps 10 × |Δt|` 에서
  static GT 쪽 v_med 0.0276~0.0690, moving GT 쪽 0.0725~0.1848 로 경계가 0.07 부근에 있고,
  분절기 통째 sweep 에서 **0.05 가 static 오탐 0 / moving 누락 0 인 유일값**이다 (0.02 는
  오탐 7편 — couch-sit 등 정지 shot 이 "move forward"; 0.08 부터 avocado-slice 누락).
  17편 재실행(같은 pipeline 세팅, `captions_gendop/` 덮어씀) 판정 **exact 11 / partial 4 /
  miss 2** (0.02: 7/6/4). 남은 miss 2 (camel·cows 의 pan 누락)는 `angular_static_threshold`
  축이라 별도. 경계 마진이 좁아(0.069 vs 0.073) 전량 재실행 때 `_tag.json` 의 `step_median`
  으로 경계 근처 scene 을 점검할 것. 비교표
  `CinemaTraj/results/20260827_caption_gendop_compare/compare_th005.txt`.

### Added
- **`CinemaTraj` — tracking shot preset 9종 (`track_*`) (2026-08-28).** 지금까지 preset 은
  "카메라가 어떻게 움직이나"만 정의했고 **subject 를 따라가는 축은 `--follow_gains` CLI 에만**
  있었다. 그 결과 배포 뱅크 51편 **16,748 변이의 `follow_gain` 이 전량 0** 이고 학습 캡션
  9,373건의 motion 어휘에 track/follow 가 **0건**이다 (키워드 grep + 뱅크 JSON 두 방향으로 확인).
  `lbm/presets.py` 에 `track_follow` / `track_follow_locked` / `track_side_{left,right}` /
  `track_push_in` / `track_pull_out` / `track_orbit_{left,right}` / `track_rise` 를 추가하고,
  `PRESET_ZOOM_END` 와 같은 형태의 **side-dict `PRESET_FOLLOW`** 로 이름에 gain 을 묶었다 —
  기존 preset 은 dict 에 없으므로 동작이 한 톨도 안 바뀐다 (실측: camel `straight_ease` /
  `truck_left` gain 0.0 그대로, `track_*` 만 1.0). gain 을 `auto` 가 아니라 **1.0** 으로 둔
  이유는 `auto` 가 τ 를 **최소화**하는 값을 풀어서 정지 subject 에서 0 에 가깝게 나오기
  때문이다 — 그러면 캡션은 "tracks" 인데 궤적은 추종이 아니게 된다. 예산이 모자라면 `fit_tau`
  가 모양만 깎고, 그래도 넘치면 `tau_saturated` 로 뱅크에서 빠진다 (조용히 추종을 끄지 않는다).
  `track_follow*` 는 모양이 identity 라 τ 가 스케일에 반응하지 않으므로 `STATIC_PRESETS` 에
  넣었다(사다리 첫 단만, `drop_saturated` 면제). `decode/build_poses.py` 는 **명시 gain 이 0 일
  때만** preset 기본값을 쓴다 — 밖에서 준 `--follow_gain auto` 가 여전히 이긴다.
  `configs/caption_presets.json` 의 문구는 **카메라 기준 방향(left/right/in/out/up)만** 말한다:
  subject 진행방향 대비 tail/lead/side 는 `follow_shot_kind` 가 **측정**하는 값이라 preset
  이름으로 보장할 수 없다 (camel dyn_0 은 셋 다 `side-tracking` 으로 측정됐다).
- **`CinemaTraj/scripts/gendop_release_infer.py --text_from_eval_dir` — 텍스트 축만 바꾼 짝지은
  arm (2026-08-28).** latentcam eval 폴더의 `test/vista4d_<scene>_<idx>_caption.json` (= k6 가
  **실제로 받은 그 문장**)을 `captions_gendop` 문장 대신 GenDoP 에 넣는다. 안 주면 기존 동작
  그대로. 파일명 규약이 달라(`<idx>_caption.json` vs `vista4d_<scene>_<idx>_caption.json`)
  경로를 여기서 다시 만들고, 없으면 세어서 `skipped` 로 보고한다 (조용히 다른 문장을 쓰지
  않는다). `config.json` 에 `text_from_eval_dir` 를 기록해 어느 문장으로 돌렸는지 산출물만
  보고도 알 수 있게 했다.
- **`CinemaTraj/scripts/gendop_preds_to_eval_dir.py` — GenDoP npz → latentcam eval 폴더 어댑터
  (2026-08-28).** `render_pred_depth_warp.py` 는 arm 을 `--eval_dir LABEL=DIR` 로 받아 그 DIR 의
  `test/<name>_{transforms_ref,transforms_pred,caption}.json` 세 짝을 읽는데, GenDoP 은
  `latentcam__<scene>__<idx>.npz` 하나만 떨군다. 렌더러를 고치는 대신 **그 레이아웃으로 갈아
  끼우는 얇은 어댑터**를 새로 뒀다 (렌더러 수정 0줄, 렌더러의 arm-ref 대조 assert 가 어댑터
  검산으로 그대로 쓰인다). 정합은 4단계 — `raw @ GL2CV` (GL c2w → CV c2w) → frame0 rel-anchor
  → `np.rint(np.linspace(0,29,49))` index pick (보간 없음, `gendop_release_eval.py` 와 같은 규칙)
  → `rmax` 를 GT 것에 맞춘 뒤 `GT_c2w[0] @ rel` 로 scene world 에 심기. **rmax 를 GT 에서
  빌려오므로 이 산출물은 "크기"가 아니라 "모양"만 본다** — 게이지가 다른 두 모델을 한 화면에
  놓는 유일한 방법이다. `rmax<1e-9` 인 degenerate entry 는 스킵하고 개수를 표에 찍는다 (조용히
  항등 궤적을 렌더하면 "GenDoP 이 정지 궤적을 냈다"로 오독된다). `--caption_from npz` 는 GenDoP
  이 실제로 받은 문장(영상 라벨용), `--caption_from ref` 는 기준 eval 폴더 캡션(지표용 짝지은
  비교 — CLaTr 이 텍스트-궤적 정합을 보므로 두 arm 이 같은 문장을 봐야 궤적 차이만 남는다).
  `--no_rescale` 을 주면 4단계(rmax 빌려오기)를 건너뛰어 **GenDoP 이 낸 크기를 그대로 두는 raw
  arm** 이 나온다 (축·원점·프레임수만 맞춤). rmax 를 아무도 안 빌린 대조군이지만 DataDoP
  게이지가 씬 단위와 무관하므로 그 수치는 궤적 오차와 **게이지 차를 같이 잰다** — 표에 쓸 때
  반드시 같이 적을 것.
- **`CinemaTraj/scripts/rescale_eval_preds_to_ref.py` — pred 궤적 rmax 를 GT 에 맞추는 대조군
  생성기 (2026-08-28).** 위 어댑터가 GenDoP 에 GT 의 크기를 쥐여 주는데, CLaTr 은
  **표준화를 안 한다** (`main/evaluate/CLaTr/src/datasets/modalities/trajectory_dataset.py:43`
  `self.standardize = False`, 47-50/89-93/114-118 정규화 전량 주석 처리 — 궤적이 velocity 표현의
  raw 단위로 들어간다). 그래서 rmax 를 빌려주는 것만으로 GenDoP 이 크기 축을 공짜로 얻는다.
  같은 변환을 우리 pred 에도 걸어 **모양만 남는 대조군**을 만든다. GT 가 정확히 정지(rmax=0)인
  entry 는 `--skip_static_ref`(기본 on)로 빼고 개수를 보고한다 — 안 빼면 rescale 이 pred 를
  항등으로 뭉개서 "모델이 정지 shot 을 맞혔다"로 읽힌다.
- **`CinemaTraj/configs/caption_presets.json` — 빠져 있던 preset 5종 문구 (2026-08-28).**
  `lbm/presets.py` 의 `PRESETS` 는 23종인데 캡션 표는 18종뿐이라, 그 5종이 든 뱅크에
  `build_bank_captions.py` 를 돌리면 assert 로 즉사했다 (snowboard `hole_bank_k6` 는
  `--follow_gains auto` 재생성에서 `pull_out_arc_left` / `push_in_arc_right` /
  `orbit_left_pedestal_up` 이 들어와 19 preset). 방향은 이름이 아니라 **코드**로 확인했다 —
  `presets.py:91-92` 의 실측 부호 규약(`arc(+)` = 카메라가 왼쪽으로 가며 오른쪽을 봄)에 따라
  `pull_out_arc_left` = `arc(+)+dolly(−)` → "arcs to the left while pulling back",
  `push_in_arc_right` = `arc(−)+dolly(+)` → "arcs to the right while pushing in"
  (기존 `push_in_arc`/`pull_out_arc` 문구와 부호가 일관된다). 나머지 2종
  (`zoom_out_pan_right`, `static_subtle_zoom`)도 같이 채워 코드↔캡션 차집합을 0 으로 만들었다.
  - snowboard 228 변이 캡션 실측: magnitude 부사가 `dramatically` 221 / `steadily` 3 /
    `slightly` 2 / `significantly` 2 로 **사실상 한 값**이다. τ 사다리가 소스 추종에 다 먹혀
    knob 이 floor(1.9641) 근처에 몰린 결과라 이 영상에서 크기 부사는 신호가 아니다.
- **GenDoP 캡션·추론·평가 3종에 `latentcam` 분기 + 신규 `render_preset_grid_warp.py`
  (2026-08-28).** LBM-Lite 로 합성한 target 카메라(`latentcam_da3/vista4d/<scene>/da3/
  target_poses.npz`)에 DataDoP 방식 캡션을 달고 validation 500 entry 를 GenDoP release
  ckpt 로 왕복 평가하기 위한 배선. 코퍼스 레이아웃이 `eval_data/{cameras,recon_and_seg}` 와
  달라서(scene 밑 `da3/`, 영상 대신 `images_4/` 프레임 폴더, npz 가 `(N,49,4,4)` 스택) 기존
  경로로는 한 군데도 안 맞는다.
  - `caption_cameras_datadop.py`: `--sets latentcam` + `--latentcam_root/--latentcam_split`.
    `build_grid` 가 mp4 대신 **프레임 디렉토리**도 받고, `collect_entries` 가 8-tuple 로
    `entry_index` 를 실어 나른다. `target_poses.npz:extrinsics` 는 **OpenCV w2c** 라
    `load_entry_cameras` 가 `inv()` 로 c2w 를 만든다 (실측: `inv(M)@GL2CV` vs eval ref
    max|diff| 2.7e-07 이고 `M`/`M@GL2CV` 는 3.64, `inv(M)` 은 2.0). grid 는 scene 당 1장
    (`_scene_grid.png`) — 같은 scene 의 entry 들이 소스 영상을 공유한다. `cameras`/`recon`
    출력은 `source_npz` 상대경로까지 비트 동일하게 유지.
  - `gendop_release_infer.py`: `--split` 이 주어지면 split 파일이 열거한 entry 만, **순서까지
    그대로** 돈다. 디렉토리 glob 이면 학습셋 캡션이 조용히 섞인다.
  - `gendop_release_eval.py`: `kind=="latentcam"` 인 npz 의 GT tag 를 `--latentcam_root` 밑
    `vista4d/<scene>/da3/captions_gendop/` 에서 찾는다.
  - `render_preset_grid_warp.py`: **target 고정 + preset 을 격자로.** `render_pred_depth_warp`
    의 reel 은 시간축이라 preset 8종을 한눈에 못 보고, `render_target_swap_warp` 는 축이
    반대(preset 고정 + target 을 열)다. 4열 밴드마다 GT 행 바로 아래가 그 preset 의 PRED 행이라
    같은 열이 같은 preset. preset 당 entry 는 `variant_id` 의 **사다리 단이 가장 낮은 것**을
    골라 τ 효과가 preset 효과에 섞이지 않게 한다 (실측 hole 로 고르면 preset 마다 다른 단이 뽑힌다).
- **`render_pred_depth_warp.py --label_mode target_motion` + 신규
  `render_target_swap_warp.py` (2026-08-28).** eval 이 떨구는 caption 은 문장뿐이라 영상만 봐선
  어떤 preset·hole 단을 조건으로 준 건지 못 가른다. `--label_mode target_motion` 은 코퍼스
  `da3/prompts.json` 에서 `preset`/`anchor_label`/`hole_fraction`/`tau_max` 를 읽어 타일 라벨에
  preset, GT 캡션에 `target:`, pred 캡션에 motion 문장을 박고, `--order target_motion` 으로
  (anchor, preset, hole) 순 정렬 + anchor 별 `reel_target_<anchor>.mp4` 를 따로 뽑는다. 기본값
  `condition`/`index` 는 기존 동작 그대로.
  `render_target_swap_warp.py` 는 **entry 를 열**로 놓는다 (기존 스크립트는 시간축으로 이어붙여서
  target 만 다른 entry 를 나란히 못 본다): 1행 GT / 2행 pred, 열 = 같은 preset 의 anchor 별 entry
  (anchor 당 hole 낮은 순 `--per_target` 개 — 사다리 단이 섞이면 target 효과와 이동량 효과가
  뒤엉킨다).
- **DynPose-LBM 파이프라인 — dynpose-0000 880편에 vista 방식 target camera 뱅크 (2026-08-28).**
  체인은 vista 와 동일 스크립트에 `--eval_data /data1/.../DATA/DynPose-LBM` 만 바꿔 낀다:
  recon(`recon_and_seg_single.py --recon_method da3 --num_frames 49`) → VLM 명사
  (`extract_nouns_vlm.py`, **`--eval_data` pre-scan 추가** — module 상수라 argparse 전에 훑는다;
  metadata.csv 없는 corpus 는 authored 비교만 빠짐) → SAM3(`sam3_seg_instances.py`, keywords 는
  VLM dynamic 명사를 `DynPose-LBM/metadata.csv` 로 씀) → **신규
  `dynpose_dynamic_mask_from_seg.py`** (recon 의 dynamic_mask 를 SAM3 트랙 합집합 packbits
  해제로 덮어씀) → graph → cloud → bank/fit(k6 knobs)/emit.
  **recon 규약 함정**: 기본 `--seg_keywords _all_` 은 dynamic_mask 를 전부 1 로 채워
  scene_graph 의 정적 점이 0 이 된다 (파일럿 실측 사망). **빈 `--seg_keywords` +
  `--keep_recon_sky`** 로 돌리고 dynamic 은 후처리로 채우는 것이 규약.
  파일럿 2편 실측(초): recon 83 / nouns 7 / sam3 63 / mask-fix 2 / graph 112~134 / cloud 28 /
  bank 24~94 / fit 20~481 / emit 6~12. video1(주방, tau_start 0.15) **152 변이** 성공,
  video2(보행, tau_start 1.57) 는 190 조합 전부 saturated — snowboard 와 같은 D67 탈락이라
  dynpose 의 보행 영상 비율만큼 수율이 깎인다 (fit 이 20s 로 싸서 사전 필터 없이 skipped.json
  으로 걸러도 된다). 880편 recon 샤드 러너 `/tmp/dynpose_recon_shard.sh` (skip=cameras.npz,
  49f 미만은 FAIL 로그). vLLM 재기동 함정 3종은 FIX-15.
- **`CinemaTraj/scripts/gendop_release_{infer,eval}.py` — GenDoP 릴리즈 ckpt 를 우리 캡션으로
  평가 (2026-08-28).** `captions_gendop` 186개(cameras 114 + recon 72)의 `Movement` 캡션을
  `text_motion.safetensors` (HF Dubhe-zmc/GenDoP) 에 넣어 궤적 생성(모델 1회 로드, GenDoP 리포
  0줄 수정 — `eval.py:process_data` 의 토큰→c2w 디코드만 이식, 30 native 포즈 저장; 120 슬러프
  업샘플은 roll 아티팩트라 안 씀) 후 **같은 분절기로 재태깅해 원 카메라 태그와 왕복 대조**.
  생성 궤적은 DataDoP 게이지라 분절 임계는 DataDoP 원본(0.02/0.4, fps 는 30포즈 환산 7.5),
  GT 라벨은 `_tag.json`(49f, th 0.05) 그대로. 프레임별 35-class weighted P/R/F
  (latentcam `CaptionMetrics` 와 같은 정의, 코퍼스가 달라 직접 비교 금지):
  **pooled P 0.568 / R 0.4157 / F 0.4014**, mean fscore 0.4039, frame match 0.4157,
  degenerate 0/186. kind 별 mean F: cameras 0.5128 vs recon 0.2313. 35-class 정확 일치 채점이라
  "move right and forward" vs "move right" 도 0 — 하위 축 단위 lenient 채점은 미구현.
  결과 `CinemaTraj/results/20260828_gendop_eval/text_motion/roundtrip_eval.json`.
  `text_rgbd.safetensors` 평가는 MonST3R frame0 depth 완료 후 (task #111).
- **`CinemaTraj/scripts/render_pred_depth_warp.py` — latentcam 생성 카메라의 depth warp 렌더
  (2026-08-27).** eval 이 떨군 `test/<name>_transforms_pred.json`(scene world **OpenGL** c2w,
  `render_geo_preds.py:load_pred` 와 같은 규약 — ref 를 GT `target_poses.npz` 와 대조해
  max|diff| 2.4e-7 실측)을 `diag(1,-1,-1,1)` 로 OpenCV 로 되돌려 기존 `CloudRenderer` 로
  렌더한다. 열 구성 SOURCE | GT | arm 별 pred, pred 타일 캡션은 그 arm 이 실제 조건으로 받은
  텍스트(`_caption.json`). 기존 두 렌더 스크립트는 뱅크/학습 target 만 읽고 **생성** 궤적을
  읽는 코드가 없어서 새로 만들었다 (조립 부품 `render_variant`/`label_tile`/`contact_sheet`/
  `write_video` 는 전부 재사용). eval JSON 의 K 와 cloud frame0 K 가 다르면 assert.
  entry 별 mp4 + 이어붙인 reel + hole 실측 `index.json`.
- **`CinemaTraj/scripts/caption_cameras_datadop.py` — `--shuffle_taxonomy` / `--no_shuffle_taxonomy`
  (2026-08-27).** pipeline `configs/captioning/caption_cam+char.yaml:21` 의 그 키를 우리 CLI 에도
  뚫었다. True 면 chunk 마다 DataDoP 어휘표와 CAMERABENCH 어휘표 중 하나를 `random.choice` 로
  골라 라벨을 쓴다 (`processing/captioning.py:147`). **기본 False = pipeline 값이자 기존 동작** —
  우리 스크립트는 `CAM_INDEX_TO_PATTERN`/`ANG_INDEX_TO_PATTERN` 만 import 해 왔고 CAMERABENCH 표는
  참조한 적이 없다(vendored 리포엔 그 표가 아예 없다). 즉 이전 실행들도 이미 `false` 경로였고,
  이번 변경은 **그 사실을 `_tag.json` 의 `shuffle_taxonomy` 필드로 명시화**하는 것이다. 재실행
  17편에서 chunk 총 28개로 이전과 동일. True 인데 리포에 CAMERABENCH 표가 없으면 `parser.error`.
- **`CinemaTraj/scripts/caption_cameras_datadop.py` — `--gendop_root` (분절기 리포 교체)
  (2026-08-27).** 사용자 지시로 캡션 세팅을 `/data1/cympyc1785/pipeline/GenDoP` 쪽에 맞추는데,
  두 GenDoP 리포는 **같은 함수 안의 하드코딩 값이 다르다** — `segment_rigidbody_trajectories` 가
  combine 단계에서 인자를 덮어쓰며 vendored 는 `min_chunk_size = 10`, pipeline 은 `= 12` 를 쓴다
  (`smoothing_window_size = 15` 는 둘 다 같다). 인자로는 못 바꾸니 리포 자체를 갈아끼운다.
  `processing.segmentation` 은 import 시점에 경로가 정해져야 해서 argparse 전에 `sys.argv` 를
  훑는다. 기본값은 vendored 라 기존 호출은 그대로다. `_tag.json` 에 `gendop_root` 를 남긴다.
  **17편 실측(Vista4D recon 소스, 사용자 수동 GT tag 대비)**: pipeline 세팅
  (n=49 리샘플 없음 / fps 10 / diff 0.6) 이 exact 7 · partial 6 · miss 4,
  기존 기본값(120 pose 리샘플 / fps 30 / diff 0.4) 이 exact 1 · partial 11 · miss 5.
  chunk 수는 scene 당 3.06 → 1.65 로 줄었다. 결정적인 노브는 **120 pose 리샘플**이다 —
  같은 17편 knob ablation 에서 roll chunk 가 리샘플 없음 **0개** vs 리샘플 **18개**(fps·diff 는
  각각 0개·0개로 무영향), 기존 기본값 조합은 **26개**다. GT 17편에 roll 은 한 번도 안 나온다.
  즉 옛 `_tag.json` 의 roll 은 slerp 보간이 만든 것이다. 비교표는
  `CinemaTraj/results/20260827_caption_gendop_compare/{compare.txt,config.json}`.
- **`CinemaTraj/scripts/run_director_vista.py` — `--scale_mode {height,path,scene}` (2026-08-27).**
  E.T. 는 미터 단위로 학습돼서 `meters_per_u` 하나가 shot 크기를 전부 정한다. 기존 `height`
  (subject 키 = `--subject_height_m`) 는 snowboard 에서 `meters_per_u 12.210` 이 나와 카메라를
  subject 로부터 0.15~15.8 m 로 흩뿌렸다. 새 게이지 둘 — `path` 는 char **이동거리**를
  `--target_path_m`(기본 1.0) 로 맞추고(snowboard: 1.2869 u → 0.777), `scene` 은 **depth 로 잰
  씬 스케일**을 `--target_scene_m`(기본 1.0) 로 맞춘다. 후자는 `S`("1 u ≜ S DA3 units",
  frame0 non-sky 평균 ray 길이)의 정의상 G frame 에서 씬 스케일이 정확히 1.0 u 이므로
  `meters_per_u = target_scene_m` 이다. 실측(snowboard, rel 좌표계 `frac(시선각<25°)`):
  height 0.98/0.92/0.08/1.00 · 0.02/0.16/0.06/0.04, path 8/8 track 1.00, scene 8/8 track 1.00.
  depth-warp hole 은 반대로 간다 — height 0.206~0.705, path 0.789~0.956, scene 0.739~0.940.
  즉 조준은 좋아지고 시차는 커진다. 기본값은 `height` 로 두어 기존 동작 유지.
- **`CinemaTraj/scripts/viz_director_pilot.py` — `--space grav` (중력 정렬 top-down)
  (2026-08-27).** `rel` 은 소스 카메라 축이라 **(x,z) 평면이 수평이 아니다.** snowboard 에서
  중력축이 카메라 −y 에서 **12.83°** 벗어나 top-down 평면이 **12.14°** 기울었고, 그만큼 수평
  이동이 고도로 샜다 — 소스 카메라 고도가 49프레임 동안 rel 로는 **+0.553 u** 올라간 것처럼
  보이는데 중력축으로 재면 **−0.026 u** (사실상 수평)다. subject 도 rel +0.646 u vs
  중력축 **−0.012 u**. `grav` 는 up 을 `gravity.up_world` 로 잡고 가로/세로는 소스 frame0
  카메라의 right/forward 를 그 수평면에 투영해서, **좌우는 렌더 화면과 그대로 맞으면서**
  top-down 이 진짜 수평면이 된다. 기저는 OpenCV 규약 그대로 **(right, down, fwd)** 로 짰다 —
  up 을 2번 축에 두면 det −1 거울상이 되어 각도가 안 보존된다 (첫 구현이 그랬고 right 를 fwd
  와 따로 투영해 직교도 깨졌다: 시선각 median 9.2° → 11.4°). 지금은 `det(R_plot)=+1` assert +
  `right = cross(down, fwd)` 로 두 문제를 같이 막고, 시선각 median 이 rel 과 정확히 9.2° 로
  일치한다. pose 값 자체는 안 건드린다 — 렌더러와 여전히 비트 동일(4.44e-16).
- **`CinemaTraj/scripts/viz_director_pilot.py` — `--space rel` + `--planes` (2026-08-27).**
  `--space rel` 은 **첫 context view(소스 `cam_c2w[0]`)를 identity 로 둔 상대 pose** 로 그린다.
  E.T. world 로 그리면 영상에서 오른쪽으로 가는 subject 가 plot 에서 왼쪽으로 가 좌우가 뒤집혀
  보였다 (E.T. `char dx −9.216` vs rel `+3.070`). rel 좌표계의 +x 는 소스 frame0 카메라의
  오른쪽이라 화면과 부호가 일치한다. 소스 카메라 궤적도 회색 점선으로 같이 깐다.
  `--planes top side front` 로 행 구성을 고른다(기본 `top side` = 기존 2행).
  세로축 부호는 space 별 표에서 읽어 y-down 인 rel 에서도 위가 위로 간다.
  같이 들어간 표시 옵션: `--no_source_cam`(소스 카메라 궤적 끄기, 기본은 켬) ·
  `--aim_line`(카메라→subject 점선 보조선) · `--flip_y`(xz 평면 기준 거울상 `diag(1,-1,1)`;
  **top-down 에서는 no-op** — 위치·시선 `max|diff| = 0.0` 이고 side(z,y) 시선만 0.743 바뀐다) ·
  plane 변형 `top_flip`(세로축 반전) 과 `top_mirror`/`top_mirror_flip`(up 축 기준 좌우 반전,
  rel 전용) · 첫 subplot 에만 legend 를 그려 회색 실선(subject track) 과 회색 점선
  (source camera) 을 구분한다.
- **`CinemaTraj/scripts/render_director_depth.py` — DIRECTOR(E.T.) 카메라를 4D 점군에 렌더
  (2026-08-27).** `run_director_vista.py` 가 **회전 규약 미검증**으로 남긴 것을 렌더로 판정한다.
  `R_et_w` 가 world→E.T. 성분 사상이므로 world 축은 `A = R_et_w.T @ R_et[:3,:3]`, E.T. 는
  y-up/z-forward(det +1)라 OpenCV c2w 는 `[-c0 | -c1 | c2]` = 시선축 180° 회전 하나뿐이다
  (`[+c0|-c1|c2]` 는 det −1 이라 회전이 아니다). 위치는 npz 의 `cam_t_world_u` 를 그대로 쓴다.
  출력은 샘플별 `_rgb.mp4`/`_depth.mp4` + `concat.mp4`(윗줄 소스 영상 + 샘플 RGB / 아랫줄
  **소스 pose depth** + 샘플 depth) + `index.json`. 아랫줄 첫 칸을 소스 pose depth 로 채우는
  이유는 기준선 없이는 구멍이 카메라 탓인지 점군 탓인지 못 가르기 때문.
  `--rows warp` 면 컬러맵 줄을 빼고 **소스 + depth warp 한 줄**만 깐다 (`--rows depth` 는 반대),
  `--concat_name` / `--columns` 로 파일을 나눠 쓴다. 기본 `--rows both` 는 위 2줄 구성 그대로. depth 눈금은
  전 샘플 공통 5~95% 로 한 번만 잡는다. `--rot lookat` 은 E.T. 회전을 버리고 위치만 써
  subject 를 조준하는 대조군. 렌더러·컬러맵·mp4 작성은 `lbm.render` / `render_bank_videos` 재사용.
  **camel 파일럿 실측(원본 수치)**: 소스 pose 재렌더 hole **0.0031**(자기 일관성 OK),
  샘플 hole s0 0.951 / s1 0.319 / s2 0.419 / s3 0.821, path_len 0.011~0.034 u,
  depth 눈금 2.595~9.103 u. 시선-대-subject 각 median 7.2~28.2° 로 **회전 규약은 맞고**,
  구멍의 원인은 거리다 — E.T. 가 낸 카메라-subject 거리가 0.61~4.98 m 인데
  `--subject_height_m 2.0` 이 정한 `meters_per_u=16.34` 로는 0.037~0.304 u 이고,
  소스 카메라는 같은 subject 를 **0.622 u (10.2 m)** 밖에서 본다. 즉 DIRECTOR 는 소스보다
  2~17배 가까운 shot 을 요구한다.
- **`CinemaTraj/scripts/monst3r_gen_videos.py` — 생성 영상에 MonST3R depth/카메라 추정
  (2026-08-27).** `eval_data/gen/<scene>/<preset>/` 의 최종 렌더 mp4 를 MonST3R 로 재구성해
  같은 폴더 밑 `monst3r/` 에 넣는다 (`frame_depth_*.npy`, `frame_*.png`, `dynamic_mask_*.png`,
  `conf_*.npy`, `scene.glb`, `pred_traj.txt`, `pred_intrinsics.txt`, `run_meta.json`).
  MonST3R **코드는 0줄 수정** — `run_single.get_reconstructed_scene` 를 그대로 부르고 `gradio`
  를 stub 으로 채운 뒤 `chdir` 로 상대경로만 맞춘다. `pred_traj.txt` 는 TUM c2w 이고 게이지는
  MonST3R 자체(미터 아님)라 `run_meta.json:convention` 에 명시한다.
  camel/three-in-out 스모크 1편 **1093.5 s** (49프레임, GPU 1장).
  부수로 `third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth` 를 패치했다 —
  `layer.py:110-134` 의 `BasicBlock` 이 **같은 BN 모듈을 `bn3` 과 `downsample.1` 두 이름으로**
  들고 있는데 HF 배포본은 중복을 제거한 채 저장돼 `strict=True` 로드가 16키 missing 으로 죽는다.
  `bn3.*` 20키를 `downsample.1.*` 로 별칭 복사(의미상 no-op)했고 원본은 `.orig.pth` 로 백업.
  MonST3R **코드는 여전히 0줄 수정**이다.
- **`CinemaTraj/scripts/sweep_caption_thresholds.py` — 카메라 캡션 분절 임계 sweep
  (2026-08-27).** LLM 호출 없이 preset 이름이 정답인 8개 엔트리에서 config 별 outline 을 찍는다.
  실측 결론(원본 수치): 합성 카메라는 축별 `|Δt|·fps` median 이 임계 0.02 의 **12~80배**라
  0.02→0.008→0.004 로 낮춰도 outline 이 **한 글자도 안 바뀐다**. fps 30→10 단독도 합성 카메라
  outline 을 안 바꾼다. 실제로 듣는 손잡이는 `num_poses` 뿐이고, 120-pose + fps10 조합은
  recon 소스를 통째로 static 으로 뒤집는다 (camel/source: 4 seg → `static | static+roll right |
  static`). native 49 + fps 10 은 임계 위에 남는다.

### Changed
- **`CinemaTraj/scripts/caption_cameras_datadop.py` — 분절기 하이퍼파라미터를 CLI 로 (2026-08-27).**
  `--fps` / `--num_poses` / `--static_threshold` / `--diff_threshold` /
  `--angular_static_threshold` / `--smoothing_window_size` / `--min_chunk_size` / `--only` /
  `--out_subdir` 추가. **기본값은 전부 DataDoP 원본이라 인자 없이 부르면 이전과 bit-identical**
  이다. `--num_poses 0` 이면 49→120 리샘플을 건너뛰고 원본 프레임을 그대로 쓴다.
  `--out_subdir` 는 기존 `captions/`(DataDoP 원본 세팅 결과)를 덮지 않으려고 뒀다 —
  fps 10 결과는 `captions_fps10/` 로 나간다. tag JSON 에 `num_poses` / `seg_kwargs` 를 실어
  어느 눈금으로 만든 캡션인지 파일만 보고 알 수 있게 했다.
  **주의: `--min_chunk_size` 는 사실상 무효다** — `segmentation.py:391-393` 이 combine 단계에서
  `smoothing_window_size=15, min_chunk_size=10` 으로 덮어쓰고, 넘긴 인자는 translation
  분절(`perform_segmentation`)에만 닿는다.

- **`CinemaTraj/scripts/caption_cameras_datadop.py` — Vista4D eval 카메라에 DataDoP/GenDoP 방식
  카메라 캡션 (2026-08-27).** `eval_data/cameras/<scene>/<preset>.npz` (합성 카메라 114개) 와
  `eval_data/recon_and_seg/<scene>/cameras.npz` (recon 원본 72개) 총 **186개**에
  `captions/<name>_{tag.json,grid.png,caption.json}` 을 붙인다. DataDoP 3단(rigid-body
  segmentation → Movement 캡션 → 4×4 프레임 그리드 + Detailed/Concise Interaction)을 그대로
  재현하며 **GenDoP 리포는 0줄 수정** — `processing.segmentation` 과 `core.utils` 만 import 하고
  프롬프트는 `dataset/scripts/configs/captioning/llm/*` 를 런타임에 읽는다
  (`Dataset_DataDoP` 는 `processing.cleaning` → `stonesoup` 미설치라 import 불가).
  세 가지를 **가정 대신 실측**했다:
  ① **규약** — DataDoP 는 OpenGL c2w 다. `convert_viser_poses_to_new_coordinate_system` 의
  `matrix[:3,1:3] *= -1` + 인덱스 대조(1→"move backward", 3→"move up", 9→"move right")로 확인.
  우리 OpenCV c2w 는 같은 flip 으로 변환한다.
  ② **게이지 D=1** — DataDoP 22,314 clip 중 400편 실측 total |t| **median 0.1016** vs 우리 recon
  **0.1664**. 같은 자릿수라 rescale 없이 `cam_static_threshold=0.02` 를 그대로 쓴다.
  (depth 정규화는 MonST3R 가 이미 shot 단위로 했으므로 기각.)
  ③ **프레임 수** — DataDoP 는 실제 shot 길이와 무관하게 `pose_clean_normalize` 가 120 pose @
  fps 30.0 으로 리샘플한다. 우리 49프레임을 날것으로 넣으면 smoothing window(15+)가 궤적의 1/3 을
  덮어 전 구간이 한 segment 로 뭉개지므로 같은 `sample_from_dense_cameras` 로 49→120 slerp 한다.
  측정된 tag 분포(원본 수치): recon n=72 segment 수 {1:13, 2:10, 3:31, 4:18}, all-static 6,
  step_median p10 0.00069 / med 0.00226 / p90 0.00753. cameras n=114 segment 수
  {1:69, 2:24, 3:13, 4:8}, all-static 4, step_median p10 0.00386 / med 0.02047 / p90 0.05538.
  **한계: 합성 카메라 114개 중 48개가 focal ramp > 1.1× 인데 DataDoP 어휘에 zoom 이 없어
  캡션에서 사라진다** — `focal_ratio` 는 `_tag.json` 에만 남는다. `crane-above/below` 도
  DataDoP 의 `cam_diff_threshold=0.4` 가 열세 축을 죽여 수직 단어를 잃는다 (예:
  `avocado-slice/close-crane-above` d·fwd −4.390 vs d·up +1.118 → 상대차 0.745 > 0.4). 여섯 축
  자체는 모호하지 않은 preset 으로 전부 확인했다 (`room-argue/pedestal-up` d·up +1.188 → "move
  up", `golf/dolly-in` → forward, `bed-shopping/dolly-out` → backward,
  `basketball-four/right-left` → left, `breakdance/left-right` → right).
  LLM 은 로컬 vLLM `Qwen/Qwen3-VL-30B-A3B-Instruct` @ `127.0.0.1:22002` (`lbm/vlm.py` 재사용),
  `--api_base`/`--model` 로 교체 가능. `--no_llm` 이면 tag 만, `--num_shards/--shard_id` 로 분할.
- **`CinemaTraj/scripts/run_director_vista.py` + `viz_director_pilot.py` — DIRECTOR(E.T.)
  파일럿 (2026-08-27).** Vista 씬의 subject OBB track 을 char 조건으로, 영문 caption 을 text
  조건으로 넣어 카메라 궤적을 받는다. DIRECTOR 의 `src/evaluate.py` 는 et-data 데이터셋에 묶여
  있어(`root_filenames.index(sample_id)`) 단일 샘플을 못 넣으므로 배치를 손으로 조립해
  `Diffuser.sample()` 만 부른다 — DIRECTOR 리포는 **0줄 수정**. 49프레임은 `padding_mask` 로
  300 에 패딩한다 (`trajectory_dataset.py:146-147` 이 쓰는 그 규약).
  **출력으로 규약 두 개를 실측했다**: c2w 의 `col2` 가 char 을 향하고(cos 0.899, 부호 반전 0건)
  `col1` 이 world +Y 와 cos 0.988 → E.T. 카메라는 **y-up / z-forward**, `det(R)=+1`.
  caption 반응은 camel `dyn_0` · seed 24개로 측정: `pushes in` 이 col2 방향 **+0.115±0.050 m**
  (21/24 전진) vs `pulls out` **−0.181±0.058 m** (5/24), `trucks right` 가 col0 방향
  **+0.106±0.061 m** vs `trucks left` **−0.310±0.101 m**. `remains static` 은 경로
  **0.152 m** 로 이동 caption(0.39~0.67 m)과 갈린다. **화면 좌우와 col0 의 대응은 미검증** —
  기하만 보면 col0 은 screen-left 여야 하는데 caption 은 반대로 움직인다. 렌더로 확인하기 전엔
  좌우를 단정하지 말 것. `u`→m 환산은 subject OBB 높이 기준(camel 0.1224 u = 2.0 m 가정,
  16.35 m/u)이라 이것도 가정값이다.
- **`CinemaTraj/scripts/trumans_gt_render.py` — 패스별 Cycles 디바이스 `--rgb_cdevice`
  (2026-08-27).** `--cdevice` 하나로 두 패스를 묶으면 한쪽이 항상 손해다: RGB(128 spp)는 GPU 가
  압도적이고 depth/index(1 spp)는 CPU 가 더 빠르다(실측 1.8 s CPU vs 2.55 s GPU). 이제
  `--rgb_cdevice GPU --cdevice CPU` 로 갈라 잡을 수 있고, 미지정이면 `--cdevice` 를 따라가
  **기존 호출은 한 비트도 안 바뀐다**. `render_meta.json` 에 `rgb_device` 를 싣고 요약표
  `engines` 열에 같이 찍는다.
  - `CinemaTraj/scripts/trumans_to_recon.py` / `trumans_lite_bank.py` 에 `--rgb_engine` /
    `--rgb_samples` / `--rgb_cdevice` passthrough 추가. `--rgb_engine eevee`(기본)면 하위
    렌더러에 **인자를 아예 안 넘겨** 기존 렌더 명령과 비트 동일하다.
  - 실측(960×540, OPTIX, `0a761819` a02 49프레임): Cycles **32 spp** RGB 6.4 s/frame,
    depth+index 2.4 s/frame. 128 spp 대비 RGB PSNR **46.1 dB**(RMSE 1.26/255)이고 번짐 지표는
    사실상 동일(`max>=250` 픽셀 두 쪽 다 0, `max>=200` 74790 → 74615) — denoiser 가 흡수한다.
- **`CinemaTraj/scripts/trumans_gt_render.py` — RGB 패스 엔진 선택 `--rgb_engine {eevee,cycles}`
  (2026-08-27).** TRUMANS `.blend` 는 전부 **Blender 3.3.6 저작**인데 우리는 4.5.9 로 돌린다.
  4.2 에서 EEVEE 가 EEVEE_NEXT 로 전면 재작성되면서, 3.3 기준으로 맞춰둔 발광 재질
  (`Emission Strength` 노트북 20 / TV 20 / 조명 175~469)이 흰 덩어리로 타서 **옆 물체까지
  번진다** — "태블릿이 몸을 뚫고 보인다"는 신고의 실제 원인이다(가림이 아니다: index pass 로
  센 사람 실루엣 내부 laptop 픽셀은 전 프레임 0). 같은 프레임 402 / 같은 카메라 / 같은 조명을
  Cycles 로 렌더하면 번짐이 **0** 이다 (crop 안 `max>=200` 픽셀 EEVEE 401 → Cycles 0,
  `>=240` 은 64 → 0). spp 를 16→512 로 올려도 안 변했던 건 샘플링이 아니라 조명 모델 문제라는
  뜻이었다.
  - 기본값은 `eevee` 라 **기존 호출은 한 비트도 안 바뀐다**. depth/index 패스는 어느 쪽이든
    항상 Cycles 1 spp 그대로다.
  - `--rgb_samples`(기본 128) / `--rgb_bounces`(기본 8) 추가. `--cdevice GPU` 일 때 애드온
    preferences 에서 OPTIX→CUDA 순으로 명시적으로 잡고 무엇이 잡혔는지 찍는다 (예전엔 디바이스가
    안 잡혀 있으면 **조용히 CPU 로** 떨어졌다).
  - `render_meta.json` 의 `rgb_engine` / `rgb_samples` 와 요약표 `engines` 열이 실제 사용 엔진을
    반영한다.
  - 실측(960×540, OPTIX): Cycles 128 spp **RGB 4.9 s/frame**, rgb+depth+index 합쳐 ~7 s/frame.
- **`CinemaTraj/scripts/time_vista_stages.py` — Vista4D target camera 파이프라인 단계별 실측
  하네스 (2026-08-27).** 샤드 로그에는 `fit` 부터만 찍혀서 `graph`/`pcd`/`bank` 가 공짜인지
  아무도 몰랐다. `graph → pcd → bank → fit → bankemit` 을 **격리된 `--output_root`** 로 처음부터
  다시 돌려 벽시계 시간을 잰다 — 기존 `out/<video>/` 는 한 바이트도 안 건드린다(이미 학습에 쓰는
  `scene_graph.json`/`cloud.npz`/`hole_bank_k6` 를 덮어쓰지 않기 위해). 요약표에 `s/camera` 열과
  variant 수 대비 기울기·절편을 같이 내서 **고정비(graph/pcd)와 카메라당 비용(fit)** 을 가른다.
  `bank`(원본 뱅크 `bank/`)와 `fit`(출력 `hole_bank_k6/`)의 `--bank_dir` 를 갈라 놓았다 —
  섞으면 fit 이 자기 출력을 입력으로 읽는다.
- **`CinemaTraj/scripts/render_target_poses_depth.py` — 학습 코퍼스의 target pose 에서 depth 를
  렌더해 저장 (2026-08-27).** 입력은 뱅크의 `poses.npz` 가 아니라 export 결과인
  `latentcam_da3/vista4d/<video>/da3/target_poses.npz` 다 — 그래야 dedup(사다리 붕괴 중복 50%
  제거)과 K 고정(frame0 K × `--image_scale`)이 **이미 반영된 것**을 그대로 쓴다. `extrinsics` 는
  w2c 라 `inv` 로 c2w 를 만들고, 저장 K 는 640×360 인데 `CloudRenderer.render` 가 넘겨받은 K 를
  `width/self.width` 로 다시 스케일하므로 **native(1280×720)로 되돌려** 넘긴다 (안 그러면 FOV 가
  두 번 반토막 난다). `renderer.K_src[0]` 대조 assert 로 막아 뒀다.
  - 저장: `<video>/da3/target_depth/<variant_id>.npz` (`depth` float16, **hole = 0**, 그래서
    `depth > 0` 이 곧 valid — 마스크를 따로 안 만든다) + `index.json` (`vista4d_target_depth_v1`).
  - **자기검증**: 렌더 `hole_fraction` 을 `prompts.json` 에 이미 있는 **뱅크 실측치**와 대조해
    요약표에 `|Δhole| med/max` 를 찍는다. w2c↔c2w 나 K 규약이 어긋나면 0.1 단위로 벌어진다.
    실측 camel `|Δ| med 0.0046 / max 0.0299`, avocado-slice `0.0094 / 0.0292`.
  - `--videos --limit --variants --scale --preview N --dry_run --no_check` 지원. 코퍼스에 없는
    영상은 `hole_bank_k6/skipped.json` 의 사유를 읽어 `[건너뜀]` 에 찍는다 (snowboard =
    `tau_start_exceeds_ladder`).
- **`CinemaTraj/scripts/vista4d_bank_to_dl3dv.py` — Vista4D 뱅크 → latentcam_da3 레이아웃
  (2026-08-27).** 영상 1편 = scene, 뱅크 변이 1개 = segment. 이미지 디렉토리는 **영상당 하나**
  (`images_4/`, 49장, `--image_scale 0.5` → 640×360) 만 두고 합성 궤적은 `(V,49,·)` 배열
  `da3/target_poses.npz` 로 따로 낸다 — 변이마다 이미지를 복제하면 713k 장이 된다. 같이 내는 것:
  `da3/pose.npz` (소스 recon w2c + frame0 K 복사), `da3/prompts.json`
  (`target: ~ motion: ~` + `caption_fields` 4종 + `tau_max`/`hole_fraction`/`subject_area_med`),
  `da3/avg_scale_context_first_cam/<seg>.json` (= `scene_graph.json:scale.S`),
  `meta_vista4d.csv`, `seg_list_vista4d_{train,test}.txt` (**scene 단위 holdout**).
  intrinsics 고정: DA3 K 는 t 축으로 흔들리므로(정지 카메라에서도 −14% 드리프트) frame0 K 를
  49프레임에 복사하고, 이미지와 **같은 배율**로 K 를 줄인다 (`hw_list` 가 `cx*2, cy*2` 에서
  h,w 를 뽑기 때문). 결과: `intr_norm: rel` 에서 `cam_param[9:11]` 이 정확히 `[1,1]`.
- **`CinemaTraj/scripts/trumans_to_recon.py --aim_keyframes` — TRUMANS 조준을 Vista4D 뱅크와
  일치 (2026-08-27).** 기존 `synth_source_path` 는 **매 프레임** `look_at_c2w` 로 조준했다
  (keyframe 0개). Vista4D pseudo-GT 뱅크는 `--aim_keyframes 6` = `[0,10,19,29,38,48]` 에서만
  조준을 세우고 사이는 smoothstep slerp 다. 두 코퍼스를 섞어 학습할 때 조준 방식이 다르면
  "카메라가 subject 를 얼마나 빡빡하게 따라보나" 가 그대로 **코퍼스 라벨**이 되어 버린다.
  - `0` (기본) = 기존 동작 그대로. `N>=2` = keyframe + smoothstep slerp.
    `N==1` 은 여기서만 추가로 허용 — keyframe 이 하나면 보간할 구간이 없으므로 frame 0 조준을
    49프레임 내내 유지한다 (subject 추종 없음).
  - **위치는 안 건드린다.** preset 모양이 이미 clearance 검증을 통과한 것이라 그대로 둬야
    verify 결과가 유효하다. 실측: k=0 vs k=6 위치 차이 정확히 `0.00e+00`, keyframe 회전 차이
    `≤9.11e-15`, 사이 구간 조준 오차 max 0.81°, `det(R)=1.000000000`.
  - `slerp_rotation` / `rotation_log` / `keyframe_indices` 는 `decode/build_poses.py` 에서
    순수 numpy 4함수만 **복제**했다. import 하면 `lbm.render` → Vista4D 점군 렌더러 → torch 가
    딸려 오는데 이 스크립트는 Blender orchestration 용이다. 식이 갈라지면 두 뱅크의 조준이
    어긋나므로 고칠 땐 양쪽 같이 고칠 것.
  - `scripts/trumans_lite_bank.py` 에 `--aim_keyframes` passthrough + `bank_manifest.json`
    `clip.aim_keyframes` 기록.
- **`CinemaTraj/scripts/render_bank_videos.py --depth`** — splatting RGB 대신 **렌더 depth** 를
  turbo 컬러맵으로 낸다. 점군 색(=소스 영상 픽셀)이 빠지므로 텍스처에 가려 안 보이던 기하만
  남는다 — 구멍의 모양과 표면까지의 거리 변화. 역깊이(1/z)로 정규화하는 이유는 선형 z 가
  가까운 표면을 전부 같은 색으로 뭉개서 "카메라가 무엇에 얼마나 가까운지"가 안 보이기
  때문이다. 눈금은 궤적 전체 유효 depth 의 5~95% 를 **한 번** 잡는다 (프레임마다 다시 잡으면
  다가가도 색이 안 변해 비교가 성립하지 않는다). 타일끼리 색을 맞추려면 `--depth_range LO HI`
  로 고정. 구멍은 기존 `paint_holes` 와 같은 마젠타. 기본 off = 기존 동작 그대로.
- **`CinemaTraj/README.md` "지금 가능한 motion — preset 23종"** — `lbm/presets.py` 의 preset
  어휘를 `aim` 기준으로 정리한 표. `python -m lbm.presets --radius 0.6 --obs_az_span 90`
  실측(move / rot° / path-over-chord)과 함께, `look_at`(12종, 매 프레임 subject 재조준 →
  `tracking` 이 유효) vs `traj`(11종, 궤적 회전 그대로 → `tracking_ignored`) 구분, 정지 3종
  (`STATIC_PRESETS`), 회전 전용 3종(`ROTATION_ONLY_PRESETS`, τ 이분법이 크기를 못 정해
  divisor 를 `FIT_TAU_MAX_SCALE` 로 바꾸는 이유), zoom 2종이 기본 off 인 이유, orbit sweep 의
  `0.8 × obs_az_span` clamp, 별칭 8종(`zoom_in`/`zoom_out` 이 **optical zoom 이 아니라 dolly**
  로 간다는 것), 그리고 preset 에 직교하는 뱅크 손잡이 8종(`--target_tau` `--speeds`
  `--trackings` `--look_at_biases` `--follow_gains` `--follow_smooths` `--aim_keyframes`
  `--traj_basis`)의 기본값.
  - **primitive 겹침 기준 재분류**도 같이: ① 단일 축 13종 ② primitive 하나인데 그 안이 이미
    회전+이동 묶음인 것 4종(`true_orbit`, `crane`) ③ `compose()` 로 2개 이상 **동시** 5종
    (arc∘dolly 4종 + `orbit_left_pedestal_up`) ④ 시간축 이어붙임 1종(`s_curve`).
    핵심 함정 하나를 명시: `aim="look_at"` 이면 디코더가 매 프레임 조준을 다시 세워
    **겹친 회전이 버려진다**(`build_poses.py:12`) — `rise_reveal` 은 디코드 후 `pedestal_up` 과
    위치가 같고 조준만 다르다. 되살리려면 `--aim_keyframes ≥2 --keyframe_aim preset_rel`.
  - **preset 별 look-at target ON/OFF 열**(ON 12종 / OFF 11종)과 **"조준 3축" 절** 추가.
    `--trackings` 에 `none` 이 없다는 것과, 그래서 track/no-track 이 세 손잡이로 갈린다는 것을
    명시: ① `aim`(preset 고정, 조준 자체의 ON/OFF) ② `--trackings`(조준점이 subject 를 얼마나
    따라가나, `TRACKING_GAIN` world 0.0 / drift 0.6 / lock 1.0, `aim=traj` 면 무시)
    ③ `--follow_gains`(카메라 **위치**가 따라가나 = CameraBench tail/lead/side/aerial-tracking,
    기본 `0` = no-track). `auto` gain 의 두 가드(`node["moving"]`, `min_benefit`)와
    `--follow_smooths` 의 역할도 같이.
  - **"이동·회전량은 어디서 제한되나 — 4층" 절** 추가. L1 `DEFAULT_SHAPE`(이동은 subject 거리
    비율 `0.35r`, 회전은 절대 각도; `crane` tilt 는 인자가 아니라 `atan(dist/radius)` 유도값) →
    L2 `sweep = min(sweep_deg, 0.8·obs_az_span)` (arc/orbit 만) → L3 `fit_tau` 이분법 →
    L4 사후 게이트(G1~G6 + `approach_profile`, 크기를 줄이는 게 아니라 변이를 버린다).
    L3 를 preset 6종 × τ 2단으로 실측해 표로: **같은 τ 예산이면 이동량은 primitive 무관하게
    같고(≈τ·z_med) 회전량만 다르다** (τ0.10 에서 `orbit_left_arc` 9.14° / `rise_reveal` 9.04° /
    `push_in_arc` 7.73° / `truck`·`dolly` 0°). `pan` 계열은 이동 0 이라 L3 가 아예 안 걸리고
    항상 `s=max_scale=4.0` → 80° 고정이라는 것과, `true_orbit` 은 순수 나선이라 sweep 손잡이가
    상쇄돼 안 먹는다는 것(sweep 45 vs 135 둘 다 53.5°)을 명시.
  - **"이동 방향은 어디서 정해지나 — `--traj_basis`" 절** 추가. `aim="look_at"` 은 고개만
    돌리고 병진은 기준 회전의 +Z 라, 기본값 `traj_basis=source` 에서는 `straight_ease` 가
    anchor 를 향하지 않는다. camel 5노드 실측(이동 방향과 anchor 방향 사이 각, `target_tau 0.2`):
    source 0.47 / 4.97 / 7.61 / 10.68 / **12.94°**(half-hfov 14.4° 의 테두리) vs
    subject **전부 0.00°**. 기준은 frame 0 에 한 번 고정되므로 "매 프레임 현재 subject 를 향해
    전진" 모드는 없고, 계속 따라가게 하려면 `--follow_gains` 를 같이 켜야 한다는 것도 명시.
- **`tools/recammaster/run_grid.py --builtin`** — manifest 대신 ReCamMaster **저자 preset
  10종**(`example_test_data/cameras/camera_extrinsics.json` 의 `cam01`..`cam10`)을 돈다.
  entry tag 는 `builtin_cam01_pan_right` … `builtin_cam10_arc_right`.
  - 카메라 JSON 은 저자 것 하나를 10 entry 가 공유하고 **무엇을 쓸지는 `--cam_type` 이**
    고른다. 그래서 하드코딩이던 `'--cam_type', '1'` 을 모듈 전역 `RCM_CAM_TYPE` 로 빼고,
    `main()` 이 entry 마다 builder **호출 직전에** 덮어쓴다. `--builtin` 없이 쓰면 값이
    계속 1 이라 기존 동작과 동일하다.
  - `--model recammaster` 가 아니면 즉시 종료한다 — preset 을 고르는 손잡이가 그 러너에만 있다.
- **`tools/recammaster/make_81f_source.py`** — 49프레임 소스를 **최근접 프레임 반복**으로 81
  프레임으로 늘린다. `inference_recammaster.py:127` 이 `assert num_frames == 81` 이고 그 앞
  `load_frames_using_imageio` 는 짧으면 조용히 `None` 을 돌려주는데(→ `is not a valid video`),
  Vista4D `media/single/*.mp4` 는 전부 49프레임이라 그대로는 한 편도 못 돌린다.
  - 보간(`minterpolate`)을 안 쓴다: warping artifact 가 **소스에** 섞이면 출력의 결함이 소스
    탓인지 모델 탓인지 못 가른다. 대신 모션이 49/81 = **0.605배로 느려진다** — 프레임 반복
    분포와 index map 을 `<out>_stretch.json` 에 남긴다 (snowboard: 1회 17장 / 2회 32장).
- **`run_grid.py` 소스 `snowboard81`** — 위로 늘린 Vista4D snowboard. stem 을 `snowboard` 로
  안 지은 건 49프레임 원본과 섞이면 결과가 0.605배 느린 소스로 만든 것인지 안 보이기 때문.
  캡션은 `results/20260824_snowboard_2x4/meta_2x4.csv`(DAVIS 유래) 그대로.
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
      → 아래 `--elevations` 기본값 변경에서 해소.
  - **`--elevations` 기본값에 음수 고도 `-10` 추가** (`[10,25,45]` → `[-10,10,25,45]`).
    고도는 카메라 절대 높이가 아니라 **구 중심(`--anchor_origin`, 기본 subject AABB 중심) 기준
    각도**라, 양수만 두면 `camera_angle` 이 구조적으로 `low_angle` 을 못 낸다. 실측(chunk
    51/600/1500, lens 18, `--min_clearance 0.3 --min_crop_keep 0.5`): 통과 수 33→36 / 40→40 /
    29→31, 늘어난 5장이 **전부 `low_angle`** (eye_angle −25.8 ~ −40.4°).
    - `-25` 는 **넣지 않았다.** 임계 문제가 아니라 물리적으로 땅속이다 — 세 chunk 의
      `height_over_floor` 가 −0.54..0.10 / −0.71..−0.08 / −0.42..0.22 m 로 48/48 전부
      `below_floor`. `--min_height` 를 0 으로 내려도 대부분 음수라 안 살아난다.
    - `-10` 도 구 중심이 낮으면 통째로 죽는다: 무릎 꿇은 chunk 600(중심 ~0.56 m)은 48/48
      `below_floor` 로 0장. 서 있는 51/1500 만 3장/2장. 격자 한 줄이 비용의 전부라
      (`reject` 면 clearance/occlusion 광선 9발을 건너뛴다) 기본에 두는 쪽이 싸다.
  - **`--render_rejects N`** — 탈락 사유마다 대표 후보 N개를 같이 렌더한다 (0 = 기존 동작,
    통과분만). 통과 0장인 chunk 는 이게 없으면 board 에 그림이 한 장도 없어서 원인을 못 본다.
    사유가 **적게 겹치는 칸부터** 고른다 — `occluded+cropped+clearance` 가 한꺼번에 걸린 칸은
    그림에서 원인을 못 가린다. 렌더된 탈락분은 `usable:false` + `reject_primary` 로 구분하고
    board.json 최상위에 `render_rejects` 를 남긴다 (안 남기면 하류가 전부 통과분으로 읽는다).
    - 함정: **`crop_keep` 은 frustum 테스트지 가시성 테스트가 아니다.** "가장 덜 나쁜 후보"를
      `crop_keep` 만으로 고르면 벽 너머에서 subject 를 향한 칸이 `crop 1.00` 으로 1등을 먹고,
      렌더가 흰 벽만 나온다 (실측으로 한 번 당함). 프레이밍 실패를 격리하려면
      `clear_frac == 1.0 ∧ clearance ≥ min ∧ subject_dist ≥ min` 을 먼저 통과시킬 것.
  - **통과 0장 chunk(900/1200/1800)의 원인은 반경이 아니라 보행이었다.** 창(49프레임 ×
    `frame_step 3` = 4.83 s) 안에서 subject 수평 이동량이 통과 chunk 는 0.03~0.37 m 인데
    실패 chunk 는 **1.87~2.22 m** 로 겹치지 않는다. 카메라는 anchor(창 중앙) 프레임 위치를
    겨눈 채 정지해 있고 게이트는 창 전체의 최악값(`crop_keep` = min, `center_offset` = max,
    `clear_frac` = 1.0 요구)이라, 사람이 창 끝에서 프레임 밖으로 나가면 그 칸이 죽는다.

      | chunk | 이동 m | crop_keep med | center_offset med | usable |
      |---|---|---|---|---|
      | 51 | 0.28 | 0.79 | 0.22 | 36/192 |
      | 600 | 0.03 | 0.98 | 0.20 | 40/192 |
      | 1500 | 0.37 | 0.82 | 0.25 | 31/192 |
      | 900 | 2.09 | **0.004** | **0.93** | 0/192 |
      | 1200 | 1.87 | **0.22** | **0.71** | 0/192 |
      | 1800 | 2.22 | **0.07** | **0.81** | 0/192 |

      반경을 1.5/2.2/3.0/3.8 m 로 넓혀도 0/144 였던 이유가 이것이다 — 멀어져도 사람은 여전히
      창 끝에서 나가고, 대신 카메라가 벽에 붙어 `occluded`/`clearance` 가 늘어난다.
  - **`--anchor_frame mid|start`, `--gate_frames window|anchor`** — 격자 원점·조준점·렌더가
    놓이는 프레임과, 게이트를 몇 프레임에서 볼지. 기본 `mid`+`window` 가 기존 동작이다.
    - 기존 동작은 사실상 **"정지 카메라로 49프레임을 다 찍는다"** 는 요구였다. 그런데 이 board 가
      정하는 건 **첫 pose** 뿐이고 나머지 48프레임은 궤적이 채운다 — 창 전체 최악값을 첫 pose 에
      물릴 이유가 없다. 게다가 원점이 `mid` 라 보행 chunk 에서는 카메라가 **창 중간** 위치의
      사람을 기준으로 놓인다(= 첫 프레임엔 사람이 프레임 밖일 수 있다).
    - 실측 (6 chunk, lens 18, `--min_clearance 0.3 --min_crop_keep 0.5`, 192칸):

      | chunk | 창 이동 m | `mid`+`window` (기존) | `start`+`anchor` | `mid`+`anchor` |
      |---|---|---|---|---|
      | 51 | 0.28 | 36 | 50 | 58 |
      | 600 | 0.03 | 40 | 39 | 41 |
      | 900 | 2.09 | **0** | **44** | 34 |
      | 1200 | 1.87 | **0** | **21** | 34 |
      | 1500 | 0.37 | 31 | 46 | 44 |
      | 1800 | 2.22 | **0** | **51** | 30 |

      보행 chunk 3개가 0 → 21~51 로 살아난다. 정지 chunk 600 은 39~41 로 거의 안 움직인다 —
      즉 이 축은 **보행 chunk 에만** 작용한다. probe 도 6.0 s → 2.0 s 로 3배 빠르다
      (프레임 3장 → 1장).
    - **대가**: `occluded`/`clearance` 를 한 프레임에서만 본다. anchor 에선 안 가렸는데 창
      중간에 기둥 뒤로 들어가는 후보를 못 거른다. 창 전체 보장이 필요하면 `window` 로 둘 것.
  - **`--render_select crop|diverse`** — `--max_render` 로 자를 때 무엇을 남길지. `crop` 이
    기존 동작(`crop_keep` 내림차순 → `center_offset`)이고 `diverse` 는 `(shot, camera_angle)`
    셀 라운드로빈이다 (셀 **안에서는** 기존 기준 그대로). board.json 최상위에 `render_select` 를
    남긴다 — 하류가 렌더분으로 태그 분포를 재면 이 값에 따라 답이 3배씩 달라진다.
    - **`crop` 정렬은 shot/angle 을 구조적으로 편향시킨다.** `crop_keep` 은 "몸이 프레임에 얼마나
      들어왔나"라서 **반경이 크면 다 들어오고 고도가 높으면 내려다봐서 더 들어온다** — 즉 멀고
      높은 칸이 항상 1등이다. 가까운 r 1.1 칸은 잘려서 컷 아래로 밀린다.
    - 실측 (blend `00add26c…`, 14 chunk 전량 = 겹치지 않는 stride 145, `start`+`anchor`,
      lens 18, `--min_clearance 0.3 --min_crop_keep 0.5`, 통과 574/2688 → 렌더 12×14=168):

      | 태그 | 통과 574 | `crop` 168 | `diverse` 168 |
      |---|---|---|---|
      | `full` | 20.7% | **60.7%** | 22.6% |
      | `medium_full` | 33.8% | 36.9% | 35.1% |
      | `medium` | 33.8% | **2.4%** | 31.5% |
      | `medium_close` | 11.7% | **0.0% (0/67)** | 10.7% |
      | `low_angle` | 5.2% | 1.2% | 8.9% |
      | `slight_low` | 21.1% | **3.6%** | 23.8% |
      | `eye_level` | 35.0% | 26.2% | 32.1% |
      | `slight_high` | 30.8% | **50.6%** | 26.8% |
      | `high_angle` | 7.8% | **18.5%** | 8.3% |

      chunk 당 고유 `(shot, camera_angle)` 셀 수 **4.2 → 10.6** (12장 중). `facing` 은 두 방식
      모두 통과분과 5%p 안쪽으로 붙는다 — 편향은 **거리·고도 축에만** 걸린다.
    - 통과가 많은 chunk 일수록 심하다. c00(통과 52장)은 `crop` 에서 12장이 전부 `full` 이고
      el45/r2.0 이 5장 — 방위각만 다른 사실상 같은 그림이었다. 통과가 적은 c08(16장)에서만
      `medium`/`low_angle` 이 섞였는데, 그건 골라진 게 아니라 후보가 없어서 어쩔 수 없던 것.
    - 렌더 비용은 동일하다 (14 chunk 전량 195.9 s vs 195.8 s).
- **`CinemaTraj/scripts/run_board_sweep.py`** — `Recordings_blend/` 전량에 first-pose board 를
  돌리는 드라이버. `<out>/<recording>/board.json` + `<out>/sweep.json`(편별 상태·시간·usable).
  실패해도 안 죽고 stderr 꼬리만 모은다 (66편 중 1편 때문에 65편을 다시 돌리지 않도록),
  `--skip_done` 으로 이어 돌린다.
  - **디렉토리 나열이 사소하지 않다.** 72 항목 중 `- 副本` 5개는 중복본이고, `<id>/<id>.blend`
    규칙을 안 지키는 게 3개 (`*_lph_ng` 접미사 2, `fancy/` 1), `a3fcee3e-…/` 는 빈 디렉토리다.
    이름 규칙으로만 찾으면 **조용히 3편이 빠진다** → 디렉토리 안의 `.blend` 를 집는다. 남는 게 66편.
- **`trumans_first_pose_board.py --chunk_stride N`** — `--chunk_starts` 대신 recording 전체를
  N 간격으로 훑는다 (`--chunk_starts` 는 이제 선택). board.json 에 `chunk_stride` + `frame_range`
  를 남긴다. 겹치지 않게 하려면 `(num_frames-1)*frame_step+1` (49×3 이면 145).
  - 편마다 프레임 수가 달라서 start 목록을 밖에서 만들려면 **blend 를 한 번 더 열어야** 한다
    (편당 ~4 s). 그래서 프레임 범위를 스크립트 안에서 읽는다. 파일럿 blend 에서
    `--chunk_stride 145` 가 손으로 준 14개 start 와 usable 수까지 일치하는 것을 확인했다.
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
