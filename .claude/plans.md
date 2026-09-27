# plans.md — 목표별 실시간 계획

사용자 지시 (2026-09-27): ".claude에 목표랑 현재 진행상황에 맞춰 실시간 계획 같은거 짜놓고 볼 수 있게
plans.md 써놓고 업데이트하면서 진행하자."

- 목표 원문은 `.claude/goals.md` "현재 목표 목록", 요청 단위 진행은 `request_queue.md`, 실험 수치는
  `EXPERIMENTS.log`, 돌고 있는 것의 감시 정의는 `watch.md`. **여기는 "지금 무엇을, 다음에 무엇을"만** 쓴다.
- 갱신 규칙: 작업이 시작·끝·막힐 때마다 해당 행의 **상태 / ETA / 다음** 을 고친다. 맨 위 "지금 자원" 표는
  GPU 배치가 바뀔 때마다 고친다. 끝난 단계는 지우지 말고 `✓` 로 남겨 흐름을 볼 수 있게 한다.

마지막 갱신: 2026-09-27 18:00

---

## 지금 자원 (GPU 0~3 만, 4~7 금지)

| 자원 | 작업 | 목표 | ETA |
|---|---|---|---|
| GPU 0,2,3 | D284 DynPose 확장 뱅크 d185 (chain, 6샤드) | G2 | ~09-29 05:00 (2.7편/분, 남은 5,908) |
| GPU 1 | R59 학습 da3 unposed (faq38efm) | G1 | ~09-29 00:30 (epoch 5/50, 41 분/epoch) |
| CPU 36 | D284 scene graph | G2 | ~09-28 01:30 (7.6편/분, 남은 3,560) |
| CPU 24 | TRUMANS d277T graph+mesh | G6 | ~09-28 24:00 (1.2편/분, 남은 2,240) — D284 graph 가 끝나면 빨라짐 |

---

## G1. DA3 · Molmo2 가 실제로 도움이 되는가

**현황** — testset (5,144 seg, last.pth, seed 42) 비교표 (EXPERIMENTS.log R55/R55b/R55c):

| arm | loss_traj | captions fscore | clatr fcd |
|---|---|---|---|
| d200 molmo2_l21 + da3 (6 view) | 0.017059 | 0.4603 | 22.29 |
| D274 + da3 12 view | 0.017286 | 0.4605 | 16.11 |
| d262 molmo2_l21 + srccam (DA3 없음) | 0.018824 | 0.4338 | 19.30 |
| D269 siglip2 + srccam (LM 없음) | 0.019919 | 0.4282 | 18.94 |
| d268 umt5 + srccam (video 없음) | 0.025523 | 0.3928 | 21.09 |

**진행/대기**
| 항목 | 상태 | 다음 |
|---|---|---|
| R59 da3 unposed (DA3 에 소스 카메라 없음) | 학습 중 GPU 1 | 끝나면 testset eval → 표 |
| R54 da3 24 view | 중단 (epoch 22 부터, tasks A5) | GPU 나면 재개 |
| R58 siglip2 + 새로 학습하는 connector | 중단 (epoch 15 부터, tasks A5) | GPU 나면 재개 |
| d263 umt5 only | eval 막힘 (seg-list id, tasks A3) | 수리 후 eval |
| **Qwen3-VL 로 Molmo2 대체** | 조사 완료 (tmp/agent/reader-plan-qwen3vl.md) — **가능·저렴** | 사용자 승인 후 캐시 굽기 → 2 arm |

Qwen3-VL 판단 (2026-09-27):
- 디스크 보유: **Qwen3-VL-4B** (dense 36층, hidden 2560 = Molmo2-4B 와 같은 폭·깊이 → `video_latent_dim 2560`·`peav_layer`
  배선 그대로), 30B-A3B (bf16 62 GB, MoE — GPU 0~3 한 장 여유 최대 53.8 GB 라 지금 못 올림). 8B/2B 없음.
- 추출: LM hidden 층 L (Molmo2 l21 대응 → l20/21) 을 video_token_id 위치에서. **temporal patch 2** 라 49장을 넣으면
  2프레임이 토큰을 공유(25 x 12x12) → per-frame 정렬을 지키려면 프레임 2번씩 98장 입력 (49 x 12x12 = 7056 → 8x8 풀링 3136).
  `do_sample_frames=False` 필수 (기본은 fps 2 로 솎음).
- 비용: prefill ~2~4 s/scene 추정 → 10,169편 ~5.4~11 GPU-h, 캐시 1층 163 GB (Molmo2 l21 과 같은 크기). Molmo2 no-decode 와 같은 급.
- 이득 근거는 약함: D167 prefill 쌍비교 LOO Qwen3-VL-4B 78.2% vs Molmo2 79.0% (n=262).
- 위험: DeepStack(ViT 를 LM 앞 3층에 주입) 이라 "LM vs ViT" 대비가 흐림, 98장 복제는 학습 분포 밖 입력.
- 제안 arm (molmo2_l21 vs siglip2 짝과 대칭): A `qwen4b_l20` (384x98 LM hidden, 2560-d) / B `qwen4b_vit` (ViT pre-merger 1024-d, 8x8).

---

## G2. 깔끔한 3인칭 10k 영상 데이터셋

| 단계 | 상태 |
|---|---|
| D282 VLM+SAM3 필터 (DynPose-100K 43,782 → pass 9,396) | ✓ |
| D284 recon 7,168 / nouns / 동적 SAM3 7,143 / dynmask / 정적 명사 / 정적 SAM3 / geocalib | ✓ |
| D284 scene graph (CPU 36) | 3,583 / 7,143 (graph FAIL 1편: stat_4 extent 0) |
| D284 강등 → 뱅크 d185 (GPU 0,2,3) | 1,235 / 7,143 |
| 다음: d199 frame0 anchor 보강 → desc(VLM) → captions → export → 캐시 → d200 과 합친 새 코퍼스 → 학습 | 대기 |

주의: 사용자가 "d200 이 필터링 후 얼마 안 남기도 했고 start pose sampling 도 해야" 라고 해서 한때 멈췄다가
"시간 아까우니 d200 처럼 이어서" 로 재개 (09-27). start pose 판 코퍼스는 G3 에서 따로 만든다.

---

## G3. 첫 카메라(start pose) 생성까지 포함한 데이터셋 + 학습

| 단계 | 상태 |
|---|---|
| R62 pilot: 36 격자 (4방위 x 3고도 x 3 shot) + source 카메라, snowboard/golf | ✓ (configs/bank/r62*) |
| 시작 카메라 선판정 `--start_screen` + 국소 지면 `--min_local_ground` + 시트 `viz/render_start_screen.py` | ✓ snowboard 35/37, golf 25/37 |
| 게이트 확정: G1 표면뒤 끔 · G3 가림 0.6 · `--obb_skip_flat 0.1` · 격자 고도는 중력 기준 유지 (context.md) | ✓ |
| **Vista + keep 데이터로 첫 학습 계획** | 계획 완료 (tmp/agent/reader-plan-startpose-train.md) — 아래 단계 |
| τ 단계에서 통과 후보만 펴서 K 개 샘플링 (tasks A7 2단계) | 대기 |

G3 첫 학습 계획 — keeper 23편 = Vista 17 (D215, `tmp/results/d215/keepers.csv`, magnifying-glass 는 뱅크 없음 → 16) +
  DynPose 6 (D212, `camera_generation/dataset/results/20260920_d212_molmo2_l21_dyn/keepers/keepers.csv`: man/car/player/shirt/squirrel/dolphin,
  d200 코퍼스라 out_dynpose/ 에 graph·recon 있음). 사용자 확인 2026-09-27 "keeper scene 말한건데 vista말고 dynpose에서도".
- 과거 D261 start head 는 **상수 예측보다 못했다** (val trans 0.7737u / rot 91.34°, n=125; 코퍼스 평균 |t| 0.6512 / rot 93.00°).
  원인 추정: 텍스트가 `Track {target}.` 뿐이라 같은 조건에 start 36개가 붙은 **다봉 타깃을 MSE 로** 풀었다.
  → 이번엔 캡션에 **시작 구도 절**("from the back, low angle, close-up ...")을 넣는 게 핵심.
| 단계 | 내용 | 비용 |
|---|---|---|
| a1 screen | keeper 22편 (Vista 16 + DynPose 6) x 37 후보 (source + 36) `--start_screen only --min_local_ground 0` | 분 단위 |
| a2 bake | 씬별 preset(평균 6; DynPose 는 d212 preset) x 통과 후보, 게이트 = r62 3차 확정안 + `--start_screen filter` | ~11 GPU-h (상한 ~4,900행) |
| a3 K | 파일럿은 통과 전량, 확장 시 (scene,preset)당 az4 x cov3 층화 K=12 + source (층화 코드 새로 필요) | - |
| b caption | `build_bank_captions.py` 에 start 절 추가 (지금 az/el 어휘 없음; front = 피사체→소스 방향임에 주의) | 코드 |
| c export | `vista4d_bank_to_dl3dv.py --bank_dirs ... --drop_status clamped_low *_blocked`, test = camel/avocado-slice/bmx-bumps (D261 과 같게) | - |
| d train | extends `vista_d261_molmo2_l21_da3_startpose`, start_pose_w 1.0, tf_p 0. cam_param = E@inv(E[0]) 유지 (VAE 재학습 회피) | ~3~6 h |
| d 대조 | ① D261 base ② 같은 코퍼스 + start 절 없는 캡션 ③ 상수 예측 | - |
| e eval | start trans/rot 오차를 az/el/cov 로 분해 + world 궤적 = pred start ∘ pred traj — **start_pred 를 읽는 추론·평가 배선이 없음** → 추가 필요 | 코드 |

---

## G4. 가까운 씬에서 dolly in 이 앞뒤로 떨리는 문제

| 단계 | 상태 |
|---|---|
| 원인 조사 (데이터: 가까운 dolly 의 손잡이 분포·캡션 / 정규화 / 추론 샘플) | 조사 중 (tmp/agent/reader-plan-goal45-causes.md) |

## G5. 빠르게 움직이는 물체 track · framing

| 단계 | 상태 |
|---|---|
| 원인 조사 (fast subject 비율·track preset 통과율·조건 입력) | 조사 중 (위와 같은 보고서) |

---

## G6. TRUMANS + synthetic 같이 학습

| 단계 | 상태 |
|---|---|
| board source 렌더 807 chunk → 3,440 clip | ✓ |
| 이름 해석 수리 (board 규약) | ✓ 05bcdff |
| d277T graph+mesh (CPU 24) | 1,200 / 3,440 |
| 다음: GPU 단계 cloud→route→tau→fit→emit (GPU 0~3, D284 뱅크와 겹침) → export 스크립트 board 이름 수리 → caption → 단독 코퍼스 | 대기 |

---

## 결정 대기 (사용자)

- R54 / R58 재개 시점 (GPU 가 D284 뱅크·TRUMANS GPU 단계와 겹친다).
- TRUMANS GPU 단계를 D284 뱅크와 반씩 나눌지, D284 뒤로 미룰지 (기본: 뒤로).
- 정리 승인: molmo2_frames_378 (≈213 GB, feat 캐시로 대체), smoke 결과, tmp/agent 보고서, tmp/r22b/corrupt_mesh_gt.
