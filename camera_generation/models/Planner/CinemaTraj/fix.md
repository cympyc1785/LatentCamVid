# 미룬 수정 목록 (LBM-Lite 뱅크 파이프라인)

사용자 지시 (D112): "제자리형 분류하는걸로 routing 하는거 일단 기록해두고 나중에 다른 요소들도
다 고치고 난후에 한 번에 같이 적용할 수 있게 고칠 목록들 fix.md 에 기록해줘."

**왜 한 번에 몰아서 적용하나.** 아래 항목은 전부 **뱅크 재굽기(코퍼스 전량 재fit)** 를 요구한다.
하나 고칠 때마다 51+188+880편을 다시 구우면 GPU 며칠이고, 더 나쁜 건 arm 끼리 paired 비교가
깨진다는 것이다 (D107/D110 이 같은 holdout 27편을 공유하는 이유). 그래서 **개별로 반영하지 않고
여기 모았다가 한 판에 적용**한다. 적용할 때 이 파일의 항목마다 D-번호를 달고 CHANGELOG 로 옮긴다.

각 항목: **증상 → 원인 → 고칠 것 → 영향 범위(재굽기 필요 여부) → 근거(실측)**.

---

## F1. 제자리형(in-place) subject 를 static 으로 라우팅

- **증상.** 사람이 제자리에서 팔만 움직이는 chunk 가 `moving` 으로 분류되어 `track_*` preset
  (조준 lock + follow) 으로 라우팅된다. 그런데 실제 이동이 없으니 follow 궤적이 정지 궤적과
  구별되지 않고, τ 예산만 먹는다.
- **원인.** 이동/정지 판정이 track center 의 **총 변위**가 아니라 프레임간 변화량 기반이라
  제자리 흔들림(jitter)을 이동으로 읽는다.
- **고칠 것.** `drift_u > 0.05` (scene 단위 u) 를 `moving` 의 기준으로. 제자리형은 drift 를
  적용해 **static 으로 본다** (사용자 지시).
- **영향.** preset 라우터 → 뱅크 전량 재굽기. dynpose/vista/trumans 셋 다.
- **근거.** D111 제안 시 실측: dynpose 코퍼스에서 **52편 demotion(moving→static), 0편 promotion**.
  즉 현재 라우터는 한 방향으로만 틀린다.
- 상태: **제안됨, 미승인.**

## F2. `S` / `z_med` 를 frame 0 depth 로만 재는 문제

- **증상.** 카메라가 크게 돌아 실외/다른 공간을 보게 되면 그 구간의 실제 scene scale 이
  frame 0 과 다른데, 게이트 임계(`margin_frac·S`, `clear_frac·S`, `obb_clear_floor`,
  `min_ground_clear`)는 전부 frame 0 의 `S` 하나로 고정된다. 결과적으로 먼 쪽에서는 임계가
  과하게 크고(=과잉 기각), 가까운 쪽에서는 과하게 작다.
- **원인.** `scene_graph/scale.py` 의 `S` 는 frame 0 non-sky 픽셀 평균 ray-depth 한 값이다.
  이건 **게이지(단위 정의)** 로서는 맞다 — 코퍼스 간 비교가 되려면 한 씬에 한 값이어야 한다.
  문제는 그 게이지를 **게이트 임계의 스케일**로도 재사용한 것이다. 두 역할이 섞여 있다.
- **고칠 것.** 게이지(`S`, u 단위 정의)는 그대로 두고, 게이트 임계만 **프레임 로컬 스케일**
  `z_med(t)` 로 나눈다. 즉 `clear = clear_frac · z_med(t)` 형태. u 단위 자체는 안 바뀌므로
  하류 export/캡션은 영향 없다.
- **영향.** `lbm/gates.py` 의 G1/G5/G6/G7 임계 계산 → 뱅크 전량 재굽기.
- **근거.** 미측정. 적용 전에 **`z_med(t)/z_med(0)` 의 프레임 내 분산**을 코퍼스 전량에서 먼저
  재야 한다 — 분산이 작으면 이 항목은 기각한다 (F2 는 "고쳐야 한다"가 아니라 "재봐야 한다").

## F3. G1 `clear_frac` 이 "관통"과 "너무 붙음"을 한 판정으로 묶는다

- **증상.** `binding=collision` 이 clamped_low 의 46.4% 로 첫 벽 1위인데, 실제로 벽을
  뚫은 건지 표면 앞 여유가 모자란 건지 CSV 로는 못 가른다.
- **원인.** `behind_surface_frames` 의 식이 `z_cam + clear_frac·S > depth + margin_frac·S` 라
  `clear_frac` 위반(표면 앞 standoff 부족)과 `margin_frac` 초과(진짜 표면 뒤)가 같은 bool 로
  합쳐진다.
- **고칠 것.** 두 사유를 분리해 `behind_frac_pierce` / `behind_frac_tight` 두 열로 기록하고,
  게이트는 `pierce` 만 hard 기각 · `tight` 는 임계를 따로 둔다.
- **영향.** `lbm/gates.py` + bank.csv 열 추가 → 재굽기(열만 추가면 하류 무해).
- **근거.** **실측(2026-09-02, `scripts/viz_g1_collision.py`).** parkour
  `dyn_0__dolly_in__hole0.5`, behind_frac 0.9796 (48/49 프레임 위반):
  `pierce 0 (0.0%) / tight 48 (7.5%) / clear 70 (11.0%) / skip 519 (81.5%)`.
  **관통이 0건이다.** 이 씬의 collision 기각은 전부 standoff 요구 때문이다.

## F4. G1 판정 불가(skip) 비율이 너무 높다

- **증상.** 위 실측에서 프레임×소스 637쌍 중 **519쌍(81.5%)이 skip** — 화면 밖 / 카메라 뒤 /
  하늘 / 무효 depth 라 판정 자체가 안 된다. 남은 118쌍만으로 48프레임을 기각했다.
- **원인.** probe 가 소스 카메라 13대뿐이고, 플랜 카메라가 소스 시야 밖으로 나가면 볼 수 있는
  소스가 급감한다. 즉 **"안 보이니까 판정 못 함"이 "안전"으로 읽힌다** (skip 은 hit 이 아니다).
- **고칠 것.** ① `judgeable_frac` 을 열로 남긴다 — 낮으면 G1 신뢰도가 낮다는 뜻. ② skip 이
  임계 이상이면 status 에 표시해서 "게이트를 통과한 게 아니라 못 물어본 것"을 구분한다.
- **영향.** 열 추가 + status 접미사 → 재굽기.
- **근거.** 위 실측 81.5%.

## F5. knob 하한(`tau_floor`)이 게이트보다 더 자주 진범이다

- **증상.** `clamped_low*` 를 "게이트가 빡세다"로 읽고 게이트를 풀려는 유혹이 있는데, 실제로는
  손잡이 하한이 원인인 경우가 압도적이다.
- **고칠 것.** `tau_floor_src` 규칙(`tau_start + 0.02`)을 재검토. `parallax_ratio` 가 높은 씬은
  `tau_start` 가 높아 하한이 위로 밀리고, `solve_knob` 이 하한 아래 사다리 점을 **버리기까지**
  해서 이분법이 1점으로 붕괴한다.
- **영향.** `fit_hole_ladder.solve_knob` → 재굽기.
- **근거.** **실측(`scripts/audit_bank_status.py`, `out/*/hole_bank_k6_d99`, 52편 30,556변이):**
  완화 후보 2,489행(8.1%) 중 **2,265행(91.0%)이 `+tau_floor`**. binding 분포는
  collision 1155(46.4%) / obb 564(22.7%) / ground 352(14.1%) / hole 181(7.3%) /
  approach 161(6.5%) / elev 76(3.1%).
  목록: `/tmp/d99_clamped_low.json` (`bank_relax_candidates_v1`, binding 별로 묶여 있음).

## F6. 물리 게이트의 시간축 정합 — **진범은 G5/G7 이 아니라 G1 이었다**

- **증상.** subject 가 움직여 멀어지는 클립에서 "전진 한계"가 걸린다 — 카메라는 다가가지
  않았는데 기준 박스가 그 자리에 있어서 상대적으로 가까워진 것으로 읽힌다.
- **코드 감사 결과 (2026-09-02).** OBB 를 쓰는 게이트 셋은 **이미 프레임별 OBB 를 쓰고 있다**:
  `obb_clearance` (`lbm/gates.py:194`), `elevation_profile` (`:239`), `approach_profile` (`:281`)
  이 전부 `node.get("moving")` 이면 `node_obb_at(node, f)` 로 갈라진다 (`scene_graph/obb.py:90`
  → `center_smooth[f]`, `yaw[f]`). 정적 노드가 참조 프레임 `node["obb"]` 를 쓰는 것도, `extent`
  를 시간 불변으로 두는 것도 맞다. 즉 **F6 의 원래 전제(G5/G7 이 frame-0 박스)는 틀렸다.**
- **진짜 frame-0 잔재는 G1.** `behind_profile` 은 궤적 프레임 하나를 **샘플된 소스 프레임 전부**
  에 되쏘아 depth 와 비교한다 — 그래서 **frame 0 에 서 있던 사람이 궤적 48프레임의 카메라도
  막는다.** 이미 걸어가 버렸는데도. 같은 리포의 `render.standoff` 는 `temporal_persistence=False`
  로 그 프레임 점만 장애물로 쓰고 있어서 (`lbm/render.py:105-108` 이 이 비대칭을 적어 두고 있다)
  **두 게이트가 서로 반대 규약**이었다.
- **고친 것 (D115).** `behind_surface_frames(dynamic_mask=, plan_frame=)` +
  `behind_profile(dynamic_mask=, time_match=)`: `t != plan_frame` 인 소스 프레임에서는
  **동적 픽셀을 증거에서 뺀다**. 정적 표면은 시간 불변이라 어느 프레임의 관측이든 유효한
  증거로 계속 쓴다. 매칭된 소스 프레임은 `frames`(균등 7/13장) 에 없어도 **추가**한다 —
  안 그러면 나머지 프레임에서 동적 물체가 G1 에 아예 안 보여 "시간축 정합"이 "동적 충돌 끄기"
  가 된다. CLI: `--collision_time_match` (기본 **off**, 끄면 기존 뱅크와 bit-identical).
- **영향.** `lbm/gates.py`, `scripts/{sample_camera_bank,fit_hole_ladder}.py` → 켤 거면 재굽기.
- **정량 — 통제 A/B (2026-09-02).** 첫 A/B(`hole_bank_tm_on` vs `hole_bank_k6_d99`)는
  **교란이 있었다**: 매칭 프레임 주입 때문에 `frames` 집합 자체가 달라져 정적 증거가 늘어난다.
  parkour 에서 21변이가 뒤집혔는데 방향이 **반대**(binding 이 collision 으로 유입)라 이게 드러났다.
  그래서 `--behind_src_frames 49` 로 **양쪽 arm 의 프레임 집합을 동일**하게 만들어 다시 쟀다
  (`hole_bank_tm49_{off,on}`) — 이러면 매칭 프레임이 항상 이미 집합 안에 있어 주입 효과가 0 이다.

  | 씬 | 변이 | status 변경 | binding 이동 | knob 증가/감소 | behind_frames 평균 |
  |---|---|---|---|---|---|
  | parkour   | 664 | **4 (0.6%)** | 0 | 4 / **0** | 8.38 → 6.83 |
  | snowboard | 196 | **6 (3.1%)** | collision→obb 6 | 19 / **0** | 0.00 → 0.00 |

  **감소가 한 건도 없다** — 시간축 정합은 순수하게 느슨해지는 방향으로만 작동한다.
  parkour 4건은 `clamped_low+tau_floor`(가짜 충돌로 knob 이 하한에 얼어붙음) →
  `collision_limited`(실제 충돌까지 궤적을 키움) 로 갔다. snowboard 6건은 collision 에서 빠져
  나와 곧바로 obb 에 다시 걸렸다. 뒤집힌 preset 은 parkour `s_curve`×4, snowboard 는 전부
  collision→obb.
- **기본값 판단.** 효과 크기가 0.6~3.1% 라 **기본 off 를 유지**한다. 켜는 건 뱅크 정체성을
  바꾸는 일이라 D114 재굽기와 같은 커밋에서 함께 결정한다.

## F9. orbit/arc 계열에서 τ 가 반경이 아니라 **sweep 각**을 줄인다

- **증상.** "크기는 작지만 많이 도는" 궤적이 뱅크에 안 나온다. τ 를 낮추면 반경이 그대로인 채
  도는 각도만 줄어든 궤적이 나온다 (사용자 지적).
- **원인.** `se3.scale_traj` 는 SE(3) **로그 전체**를 s 배한다 (`recammaster/se3.py:90`,
  `se3_exp(s*w, s*rho)`). orbit 은 `R = |t| / θ` 이므로 `w`(=θ)와 `rho`(≈|t|)가 같은 배율로
  줄면 **R 은 불변, θ 만 축소**된다. 즉 τ 손잡이가 사실상 "몇 도 도느냐" 노브다.
- **고칠 것.** `scale_traj` 는 건드리지 않는다 (pan·DataDoP 외부 shape 은 rel 행렬만 있어서
  회전/이동을 의미론적으로 못 가른다). 대신 `PRESETS` 값에 4번째 슬롯 `resize(n, c, s)` 를
  추가해, 있으면 `fit_tau` 가 `scale_traj` 대신 그걸 부른다:
  `sweep` 은 `min(preset_default, 0.8·obs_az_span)` 로 **고정**하고 반경만 s 배.
  τ 는 `|t| = 2·s·R·sin(θ/2)` 로 s 에 선형이라 **이분법 단조성이 유지**된다.
- **fitting 은 여전히 하나다 (사용자 질문 2026-09-02).** sweep 은 **적합 대상이 아니다** —
  (preset, scene) 마다 `min(preset_default, 0.8·obs_az_span)` 로 **한 번 계산해 상수로 박고**,
  이분법은 반경 배율 `s` **하나만** 푼다. 즉 "sweep 따로 반경 따로 두 번 fit" 이 아니라
  **sweep 은 선험적 상수, 반경만 fit** 이다. 이렇게 해야 되는 이유가 두 가지다:
  ① 두 자유도를 같은 스칼라 τ 로 동시에 풀면 해가 1차원 곡선이라 유일하지 않다.
  ② 지금 문제("작은데 많이 도는 궤적이 없다")는 sweep 이 **τ 에 딸려 줄어들기** 때문이므로,
     sweep 을 τ 에서 떼어내 고정하는 것 자체가 처방이다. sweep 을 다시 적합하면 원위치다.
  sweep 을 바꾸고 싶으면 τ 가 아니라 preset 어휘로 바꾼다 (`orbit_45` / `orbit_120` 처럼
  preset 을 늘리는 쪽). 그러면 "많이 도는 작은 궤적" 은 `orbit_120` + 작은 τ 로 나온다.
- **범위 주의.** `crane` 은 회전이 대부분 `look_at` 에서 나오므로 대상이 아니다. 실제 대상은
  **`aim="traj"` 인 orbit/arc 계열**뿐 — 적용 전에 `PRESETS` 에서 그 목록을 먼저 센다.
- **영향.** `lbm/presets.py` (`PRESETS` 시그니처 + `fit_tau`) → 뱅크 전량 재굽기.
- 상태: **사용자 승인됨 (2026-09-02), D114 에서 mesh 충돌과 함께 적용.**

## F10. 가림 게이트(D112)는 knob 게이트로는 **작동하지 않는다** — 선별기로 바꿔야 한다

- **증상.** `--min_subject_visible 0.5` 로 켜고 최악 2편에 돌렸더니 **가림은 안 풀리고 궤적만
  정지**했다.
- **원인.** 가림은 knob 크기에 **단조가 아니다**. 궤적을 줄여도 subject 를 가리는 건 그대로다 —
  가림을 정하는 건 **시작 pose(anchor 방향)** 이고 knob 은 그걸 안 움직인다. 그래서 이분법이
  줄일 것을 못 찾고 하한까지 걸어내려간다.
- **실측 (2026-09-02, `--min_subject_visible 0.5`, d99 대비 1:1 A/B):**

    씬            게이트 물린 변이   가림 해소   해소 못 하고 정지만   정지 비율
    camera-lens   272 / 554          24         48                    16.2% -> 24.9%
    parkour       239 / 664          14         64                    29.8% -> 39.5%

  게이트 물린 변이의 `subject_visible_frac` 중앙값: camera-lens **0.000 -> 0.000 (변화 없음)**,
  parkour 0.333 -> 0.402. 같은 변이의 `path_len_u` 중앙값은 0.189 -> 0.033 / 0.161 -> 0.006.
  **해소보다 파괴가 2배(camera-lens) ~ 4.6배(parkour) 많다.**
- **고칠 것.** 게이트를 knob 축소기로 쓰지 말고 ① **변이 선별기**(임계 미달이면 크기를 줄이는
  대신 그 변이를 뱅크에서 뺀다) 또는 ② **anchor/first-pose 필터**(상류에서 방향을 거른다)로.
  ②가 옳다 — 원인이 시작 pose 이기 때문이다. Task #68(`judge_bank_vlm.py`)·first-pose board 와
  같은 층이다.
- **영향.** `fit_hole_ladder.solve_knob` 에서 `occlusion` 분기 제거 + 상류 필터 신설.
- 상태: **D112 는 켜지 말 것** (기본 `-inf` 유지). 위 실측이 그 근거다.

## F7. 시간축 절단 (Task #97)

- **고칠 것.** `solve_knob` 이 **크기**만 줄이는 대신, 게이트에 걸리는 프레임 **앞까지만
  움직이고 그 뒤는 hold** 하는 선택지를 갖게 한다. 지금은 전 구간을 균일하게 줄이므로
  마지막 1프레임이 벽에 닿으면 전체가 정지 궤적으로 눌린다.
- **영향.** `fit_hole_ladder.solve_knob` → 재굽기.
- **상태:** 제안됨, 미승인. **우선순위 1** (F1 보다 위 — 정지 궤적 생성이 코퍼스 품질에 직접
  영향).

## F8. `ground_z` 를 subject 발밑에 앵커

- **고칠 것.** 지면 높이를 씬 전역 한 값이 아니라 subject 발 위치의 로컬 지면으로.
  비탈면(camera-lens sif) 에서 전역 평면이 subject 발밑과 어긋난다.
- **영향.** `scene_graph/gravity.py` + G6 → graph 재빌드 + 재굽기.
- **상태:** 제안됨, 미승인. **우선순위 4** (TRUMANS 는 D106 에서 `--ground_source gt` 로
  이미 mesh GT 지면을 쓰므로 해당 없음; vista/dynpose 만).

## F11. G1 소스 프레임 13장 샘플이 **실제 정적 충돌을 놓친다**

- **어떻게 발견됐나.** F6 통제 실험의 부산물이다. `time_match` 를 **양쪽 off** 로 두고
  `--behind_src_frames` 만 13 → 49 로 올렸더니 parkour 664변이에서 status 25건이 바뀌고
  **binding 39건이 전부 `collision` 으로 유입**됐다 (`hole → collision` 27, `approach → collision` 12).
  즉 지금 `hole_bank_k6_d99` 에는 13장 샘플이 못 본 **진짜 충돌이 최소 39건(5.9%)** 통과해 있다.
- **씬 의존이 크다.** snowboard 196변이는 13 → 49 에서 **status 변경 0건**이었다
  (knob 만 8건 미세 감소). parkour 가 걸리는 이유는 정적 노드 11개 + 소스 카메라 이동이
  코퍼스 최악(`tau_start` 0.7454)이라, 13장 사이에서 카메라가 많이 움직여 그 틈에
  표면이 들어오기 때문이다. 즉 **`tau_start` 가 큰 씬일수록 13장이 성기다.**
- **비용.** G1 은 재투영 + depth 조회뿐이라 **렌더가 0회**다. parkour 실측에서 13 → 49 로
  올려도 fit 총 렌더 수가 안 변한다 (`변이 664 렌더 10,794`). 즉 프레임 수는 사실상 공짜다.
- **고칠 것.** `--behind_src_frames` 기본값을 13 → 49(=전 프레임) 로. 다만 이건 **이미 나간
  뱅크의 판정을 바꾼다** — F6 과 달리 "느슨해지는" 방향이 아니라 **빡빡해지는** 방향이라,
  기존 코퍼스에서 변이가 사라진다. 재굽기 결정과 묶어야 한다.
- **영향.** `scripts/{fit_hole_ladder,sample_camera_bank}.py` argparse 기본값 → 뱅크 전량 재굽기.
- 상태: **미결정.** 사용자 판단 필요.

---

## 적용 순서 (권장)

1. **F7** 시간축 절단 — 정지 궤적을 없애는 게 코퍼스 품질에 가장 크다.
2. **F1** drift 라우팅 — 실측이 이미 있고(52/0) 한 방향으로만 틀린다.
3. **F3 + F4** G1 사유 분리 + judgeable_frac — 열 추가라 위험이 낮고, F5/F6 을 정량화할
   **계측기**가 된다. 실은 이 둘을 먼저 넣고 한 번 구운 뒤에 F5/F6 을 판단하는 게 맞다.
4. **F5** tau_floor 재검토 — F3/F4 계측 뒤.
5. **F6** 프레임별 OBB — 정량화 후.
6. **F2** 프레임 로컬 스케일 — `z_med(t)` 분산 실측 후, 작으면 기각.
7. **F8** ground_z 앵커.

## 이미 적용된 것 (여기 있다가 나간 항목)

- **D112 가림 게이트** — `--min_subject_visible` 로 hole 뒤에 occlusion 게이트 추가. 기본 0 이라
  꺼져 있고, 켜면 `metric_only` 고속 경로를 못 써서 fit 렌더가 2배다.
- **D113 게이트 순서** — 물리 게이트 5종(렌더 0회)을 hole 앞으로. 결정열 bit-identical 검증 완료.
- **D98/D106 중력** — GeoCalib 기본(vista/dynpose), mesh GT(trumans). ground RANSAC 은 검증 전용.
