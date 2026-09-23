# review.md — 레포 검토 노트 (읽기 전용 분석)

> 2026-09-23 에 `reader.md` 에서 이름을 바꿨다. `.claude/agents/reader.md`(subagent 정의)와
> 파일명이 같아 헷갈렸다 — 이 파일은 **agent 가 아니라 분석 문서**다.

이 파일은 레포를 읽으면서 남기는 축약 노트다. **코드/데이터는 수정하지 않는다.**
기준 시점 **2026-09-04 19:44** 실측. 목표는 `goals.md`, 코드 규약은 `CLAUDE.md`.

---

## 0. 한 줄 지도

```
[데이터 생성]  camera_generation/models/Planner/CinemaTraj   (env: vista4d)
   video 1편 -> scene_graph.json -> cloud.npz -> 후보 sampling -> τ 뱅크 -> hole 사다리 fit
             -> emit(canonical) -> 캡션 -> *_to_dl3dv.py 로 DL3DV 포맷 코퍼스

[모델 학습]    camera_generation/latentcam                    (env: latentcam)
   text CA (umt5 or PE-AV) -> [video CA (PE-AV / Molmo2)] -> geo CA (DA3)
             -> CameraDiffusionModel(latent, cam_dim=64) -> VAE 디코드 -> 49 pose
```

두 단계 사이 계약: 코퍼스 `<scene>/da3/{pose,target_poses,prompts}.npz|json` + `images_4/` +
`meta_*.csv`. **target 은 소스 궤적이 아니라 뱅크 변이**이고, `prompts.json` 의 segment 는
프레임 구간이 아니라 **변이 인덱스**다 (`vista4d_bank_to_dl3dv.py` docstring).

---

## 1. 지금 살아 있는 것 (2026-09-04 19:44 실측)

| GPU | pid | 무엇 |
|---|---|---|
| 1 | 1272862 | **학습** `vista4d_d121_da3_t128` (D123). screen train4, 2h42m 경과 |
| 2 | 1468575 | **학습** `vista4d_d121_molmo2` (D124). screen train1, 45m 경과 |
| 0 | 1643376/8 | `fit_hole_ladder.py --bank_dir hole_bank_g1probe_off` (parkour) — G1 임계 프로브 |
| 3,4 | 1458264/1477579/1526115 | TRUMANS-Lite `hole_bank_k6_d116` fit. screen bank0~3 |
| 5,7 | 1531192/1508304 | gaussian-splatting-lightning `launch.py` (TRUMANS 렌더) |

뱅크 진행:
- **D122 dynpose 사실상 완료** — 283 씬 중 emit 266 / 변이 0 skip 12 / graph 없음 3 / 진행중 2.
  (`out_dynpose/*/hole_bank_d122/canonical/canonical.json` = 266)
- **D121 vista 완료** — 52/52 (`out/*/hole_bank_k6_d121/...`)
- **TRUMANS D116 진행중** — `hole_bank_k6_d116` 28편 (d77 은 188, d99 는 70)

---

## 2. goals.md 와 실제가 어긋난 곳 (문서 갱신 후보)

| goals.md 기술 | 실제 |
|---|---|
| 단기 §3 학습 arm = `vista4d_d121_da3` / `vista4d_d121_peav` | 둘 다 SIGINT 정지 (da3 ep76, peav ep108). 지금 도는 건 **D123 `da3_t128`** 과 **D124 `molmo2`** |
| dynpose D122 "251편 bake 완료" | 266편 emit, 사실상 종료 |
| `traj.py:172 compose` 인용 | CinemaTraj 안에 `traj.py` 파일 없음 — primitive/compose 는 `lbm/presets.py` 안 |
| `CLAUDE.md` → "구조는 SPECS.md" | 레포에 **SPECS.md 가 없다** (find 0건) |
| `CLAUDE.md` → GPU 4~7 금지 | D124 기록에 "사용자 지시로 GPU 0~4", 실제 GPU 4 사용 중. 규칙 갱신 필요 |
| 루트 `EXPERIMENTS.log` | 2026-08-19 에서 멈춤. 실기록은 `camera_generation/latentcam/EXPERIMENTS.log` (5,648행) |

---

## 3. 모델 측 발견 (latentcam)

### M1 [최우선/저비용] `need_weights` 가 학습 중에도 True 다
`models/camera_diffusion_model_latent.py:64`, `:76` 이 `nn.MultiheadAttention` 을
`need_weights` 미지정(기본 **True**)으로 부른다. attn weight 는 `self.training is False` 일 때만
소비되는데(:337~), 학습 중에도 매 층·매 스트림에서 계산·반환된다. 게다가 `CrossAttention` 은
`:65` 에서 **무조건** `self.attn_weight` 에 저장한다 (`SelfAttention` 은 `:77` 로 가드가 있다) —
(B,49,3456) 텐서가 스텝 내내 살아 있다.
- context 길이: geo CA 3456 (6뷰×576), Molmo2 CA 3264, PE-AV CA 1841
- `need_weights=(not self.training)` 로 넘기면 **학습 경로 수치는 그대로**이고 fused SDPA 경로가 열린다
- D124 가 9.17 min/epoch × 200 = 30.6h 인 상황에서 가장 값싼 개선. 스모크 1 epoch it/s 비교로 즉시 검증 가능

### M2 [구조적] 잔차 덮어쓰기 수정이 video 스트림에만 적용됐다
`FIX.log` 2026-09-03 항목이 D117 정체(loss ~1.0)의 **진짜 원인**으로 지목한 것:
`CrossAttention.forward` 가 `norm(x + a)` 를 돌려주고 각 스트림이 `h = CA(...)` 로 잔차를 덮어쓴다
→ 스트림 하나당 층마다 h 가 한 번 더 LN 되어 x_t 성분이 층당 ~1/√2 감쇠.
그런데 수정(`video_gate`, 0 초기화 residual gate)은 **video 블록에만** 걸렸다(:187, :318) —
"기존 arm 의 state_dict/출력을 한 글자도 안 바꾸기 위해서". 즉 base 2-stream 은 여전히
8층 × 2스트림 = (1/√2)^16 ≈ **1/256** 로 x_t 를 깎는다. 이게 진짜 원인이라면 지금 도는 모든 arm 이
그 감쇠를 안고 학습하고 있는 것.
- 제안: `text_gate`/`geo_gate` 를 **기본 off** 옵션으로 추가(기존 비트 동일 보존). 단 0 초기화는
  text/geo 를 완전히 닫는 것이라 초기 학습이 늦어질 수 있으므로, **gate 초기값 1.0 + pre-norm
  잔차(`h = h + CA_out`)** 판본을 별 arm 으로 두는 쪽이 더 직접적인 대조다
- 검증 비용: d121 코퍼스에 1 arm 추가. 대조군은 D123 이 이미 있다

### M3 [지표] geo 스트림에만 positional encoding 이 없다
`:287` text_tok, `:231/:237` video_tok 은 PE 를 더하는데 `:290-291` geo_tok 은 안 더한다.
DA3 가 cross-view ViT + `cam_token` 이라 내부에 뷰/패치 정보가 있어 치명적이진 않지만,
d121 arm 은 `geo_cam_embed: null` 이라 뷰 신원을 **명시로 주는 채널도 꺼져 있다**
(`geo_cam_raw_dim>0` 경로 = `:132` 의 `geo_cam_mlp`). 값싼 ablation 후보.

### M4 [평가] 단기 완료 조건 (b)에 예측측 지표가 없다
`goals.md` 단기 완료 조건은 (a) 피사체 유지 (b) 벽/물체 미통과 (c) hole 감당 가능. 그런데
- `latentcam/scripts/eval_testset.py` = CLaTr + loss 만
- CinemaTraj `scripts/eval_subject_in_frame.py` = 예측 궤적에 (a)(c)만 (`subject_in_frame`,
  `hole_fraction`, `subject_pixel_coverage`). **collision 은 없다** (`transforms_pred` 를 읽는
  6개 스크립트 전부에 collision/behind/obb_clear grep 0건)

재료는 전부 있고 **전부 렌더 0회**다:
`lbm/gates.py:46 behind_surface_frames` / `:258 obb_clearance` / `:305 elevation_profile` /
`:345 approach_profile`, 그리고 `verify.py:138 evaluate` 는 이미 49프레임 전량에 behind 를 돈다
(`verify.py` 는 `out/<video>/poses.npz` + fingerprint 에 묶여 예측 궤적엔 못 건다 — 그래서
`eval_subject_in_frame.py` 가 따로 있는 것).
→ **제안: `eval_subject_in_frame.py` 에 `--gates` 를 붙여 예측 poses 에 위 4종을 그대로 건다.**
학습을 안 건드리고 단기 완료 조건을 처음으로 숫자로 만든다. 우선순위 1.

### M5 [지표 해석] frame 0 은 항상 identity 다
뱅크가 `--start_mode source_frame0` / `--no_free_start` 라 target `rel[0] = I`.
`val_loss_traj = F.mse_loss(traj_pred, traj)` (`train_latent_cam_dm.py:825`) 는 49프레임 평균이라
프레임 0 이 상수로 섞인다 (희석 ~2%, 작다). 더 중요한 건 **시작 pose 를 고르는 학습 신호가 0** 이고
지표가 그 결핍을 드러낼 수 없다는 것 — `goals.md` 중기 3 이 이 지점이다.
지금 당장 무료로 할 수 있는 것: 지표를 frame 1..48 로 다시 재기 (재학습 불필요).

### M6 [문서] known_issues (a) 는 arm 마다 다르다
`docs/known_issues.md` 의 "per-SEGMENT random_split → 씬 누수"는 `base.py:114` 경로 얘기다.
d121/d122 arm 은 `train_seg_list`/`test_seg_list` 를 주므로 `base.py:88-105` 의 **결정적 seg-list
분할**을 타고 씬 4편(avocado-slice, bmx-bumps, camel, couple-hug)을 홀드아웃한다 → 해당 없음.
반면 **DL3DV(`da3_7k_*`) arm 은 여전히 random_split** 이다. 문서가 arm 구분 없이 "PARKED" 로만
적혀 있어 혼동 위험.

---

## 4. 데이터 생성 측 발견 (CinemaTraj)

### D1 [즉시] `--targetless_promote` 를 켤 타이밍이 지금이다
`goals.md` 중기 1 의 대표 사례(`track_truck_left/right` 캡션은 "tracks alongside {target}" 인데
`subject_in_frame` median 0.462 / 0.385)의 **수정 도구가 이미 있다**:
`scripts/build_bank_captions.py:643 promote_targetless` (D123, 커밋 f4e5476).
기본 off 인 이유는 "d121 두 arm 이 target 절 붙은 캡션으로 돌고 있어 재export 하면 문자열이
어긋난다" — 즉 **새로 굽는 뱅크에서만 켠다**는 설계. D122 dynpose 는 emit 만 끝났고 캡션은 아직
안 구웠으므로 지금 켜면 재작업 0. 켠 사실은 캡션 JSON 헤더 `targetless_promoted` 에 남는다.

### D2 [승인 대기] fix.log F7 "시간축 절단" 이 우선순위 1인데 미승인
`fix.log:174`. 지금 `solve_knob` 은 게이트에 걸리면 **전 구간을 균일 축소**하므로 마지막 1프레임이
벽에 닿으면 궤적 전체가 정지로 눌린다. F7 = "걸리는 프레임 앞까지만 움직이고 뒤는 hold".
`goals.md` 중기 1 의 "τ 하한에 눌린 변이 회수"(`clamped_low + tau_floor` 15.4%, 길이 0 궤적
1,565 = 7.3%)의 직접 처방. `fix.log` 자체가 우선순위 1로 적고 있다.

### D3 [문서 어긋남] F11 은 이미 적용됐는데 fix.log 는 "미결정"
`fix.log:191 F11` = `--behind_src_frames` 13→49. D121/D122 러너가 둘 다
`--behind_src_frames 49` 를 **명시로 넘긴다** (`run_k6_d121_shard.sh`, `run_dynpose_d122_shard.sh`).
코드 문제 아님, 문서만 정정 대상.

### D4 [재발 방지] 게이지(S) 3중 기록의 정합을 assert 로
D122 의 본론이 이 사고였다: dynpose 280편이 `scale.mode` **필드 자체가 없어** 기본값
`frame0_ray` 로 읽혔고, F2 assert 가 09-02 20:30 에 추가됐는데 그래프는 09-02 01:05 빌드라
**19시간 차이로** d107/d110 fit 이 조용히 통과했다. `S` 는 세 곳에 박힌다 —
`scene_graph.json:scale.S` / `cloud.npz` meta (`lbm/render.py:70`) / `bank.json` top-level `S`.
→ 제안: 세 곳에 `gauge_id` 한 줄을 쓰고 fit 진입에서 **3자 일치를 assert**. 지금은 한 곳만 본다.
(참고: D122 는 `cloud.npz` 를 백업 없이 제자리 덮어썼으므로 **d107/d110 뱅크의 렌더 재현성은
그 시점에 끊겼다** — 러너 주석에 명시된 의도된 거래.)

### D5 [중기 2 병목의 절반] 복합 카메라는 표현이 아니라 **디코더**가 지운다
`compose()` 로 primitive 2개를 겹친 preset 은 이미 5종 있다
(`push_in_arc_{left,right}`, `pull_out_arc_{left,right}`, `orbit_left_pedestal_up`).
그런데 이들은 전부 `aim="look_at"` 이고, 디코더가 매 프레임 조준을 다시 세우므로
**겹친 회전이 버려진다** (README §"주의 — aim=look_at 이면 겹친 회전이 실제로는 버려진다",
`build_poses.py:12`). 실효 겹침은 이동 축만 — `crane_up` 은 디코드 후 `pedestal_up` 과 위치가 같다.
되살리는 기존 손잡이가 `--aim_keyframes ≥2` + `keyframe_aim=preset_rel` (`build_poses.py:51-54`)
인데 d121/d122 러너는 `--keyframe_aim auto` 를 쓴다.
→ 제안: "조합 라벨링" 으로 넘어가기 전에 `preset_rel` 로 한 판 구워 **회전이 실제로 사는지**
확인. goals 중기 2 의 절반이 새 GT 합성이 아니라 이 플래그다.

### D6 [알아둘 것] board/VLM 은 사실상 꺼져 있다
`--start_mode source_frame0` 이 기본이라 loop 의 `select`/`micro` 턴이 안 돈다
(`{'select': 'skipped_source_frame0', ...}`). 즉 `board_candidates.png` 는 **진단 산출물**이고
VLM 입력이 아니다. 그리고 traj 턴은 D34/D35 실측에서 **prior**로 판정됐다 (54 draw 중 49가
`orbit_left_arc`; 이름 중립화·셔플·temperature·가짜 magenta 5종 전부 못 흔듦).
**품질을 지키는 건 VLM 이 아니라 게이트다.** 뱅크 파이프라인에서 VLM 이 실제로 기여하는 곳은
`describe_instances_vlm.py`(referring expression)와 캡션 쪽뿐.

---

## 5. 우선순위 제안 (reader 의견)

| # | 항목 | 비용 | 재굽기 | 근거 |
|---|---|---|---|---|
| 1 | **M4** 예측측 게이트 지표 (`--gates`) | 하루 미만, 렌더 0회 | 불필요 | 단기 완료 조건 (b)를 처음 측정 |
| 2 | **M1** `need_weights=False` (학습 시) | 한 줄 + 스모크 | 불필요 | 수치 불변, 30.6h/arm 단축 |
| 3 | **D1** `--targetless_promote` on | 캡션 재빌드만 | 불필요 | 도구 준비됨, 지금이 재작업 0 시점 |
| 4 | **M5** 지표를 frame 1..48 로 | eval 수정만 | 불필요 | 공짜 프레임 제거 |
| 5 | **M2** text/geo residual gate arm | 1 arm 학습 | 불필요 | FIX.log 자기 진단의 미완 절반 |
| 6 | **D2** F7 시간축 절단 | 코드 + 전량 재fit | **필요** | 길이 0 궤적 7.3% / clamped 15.4% 회수 |
| 7 | **D5** `keyframe_aim=preset_rel` 확인 | 1 씬 프로브 | 확인 후 판단 | 중기 2 병목 절반 |
| 8 | **D4** `gauge_id` 3자 assert | 작다 | 불필요 | D122 사고 재발 방지 |

M2/D2 는 arm 간 paired 비교를 깨므로 `fix.log` 규약대로 **한 판에 몰아서** 적용하는 게 맞다.

---

## 6. reader 가 실행 허락을 요청할 것 (전부 읽기 전용, GPU 0회)

1. `scripts/audit_bank_status.py --output_root out_dynpose --bank_dir hole_bank_d122`
   → D122 266편 기준 `status`/`binding` 최신 분포 (goals.md 의 243씬/21,428변이 수치 갱신)
2. 같은 스크립트 `--output_root out --bank_dir hole_bank_k6_d121` → vista 대조
3. 뱅크 행에서 preset 별 `subject_in_frame` 분위 집계 (JSON 읽기만, 스크립트 신설 필요)

`audit_bank_status.py` 는 docstring 에 "렌더 0회, 전 코퍼스가 몇 초, numpy 도 안 쓴다" 로 적혀 있다.

---

# Part II — 논문 8편 대비 metric·방법론 비판 (2026-09-04 추가)

읽은 것: **E.T./DIRECTOR**(ECCV'24, 2407.01516) · **GenDoP**(ICCV'25, 2504.07083) ·
**VERTIGO**(2604.02467) · **ShotVerse**(2603.11421) · **Look-Before-Move**(로컬 코드) ·
**CinemaTraj**(2607.26910) · **PulpMotion**(ICLR'26, 2510.05097) · **AdaViewPlanner**(2510.10670).

## 7. metric 계보 한 표

| 논문 | 궤적 품질 | text 정합 | framing / subject | 물리 | 하류 영상 | 사람/VLM |
|---|---|---|---|---|---|---|
| **E.T./DIRECTOR** | FD_CLaTr, PRDC | CLaTr-Score, caption F1 | — | — | — | — |
| **GenDoP** | CLaTr-FID, Coverage | F1(motion tag), CLaTr-CLIP | — | — | — | AUR: Alignment/Quality/Complexity |
| **VERTIGO** | FCD, CS, P/R/D/C | CS + cyclic semantic sim | **MisR** = off-screen **또는 외곽 20% 테두리** 프레임 비율 | — | VBench Cons./Aes. | fine-tuned VLM 채점 → DPO, user study 4축 |
| **ShotVerse** | Transition/Rotation Err | F1, CLaTr-CLIP | Subject Emphasis(VLM) | **CAS** = cross-shot FOV 겹침 쌍의 DINOv2 유사도 | ViCLIP global/shot, LAION Aes., Shot Transition Acc., FVD | Gemini 3 Pro 4축 + user study 동일 4축 |
| **Look-Before-Move** | TQ1 optical-flow CV, TQ2 tracking, TQ3 cut | IC3 event alignment(VLM) | SP1 coverage, SP2 identity, SP3 occlusion, IC1 shot size, IC2 semantic target | — | — | VLM(IC3) |
| **CinemaTraj** | Motion MSE(arc-length resample 후) | CLaTr Score, **Object Coverage**(계획 anchor 중 방문 비율) | Occlusion Rate | **Collision Rate = SDF(p)<0 샘플 비율** | — | 5-point Likert 3축 |
| **PulpMotion** | FD_CLaTr, CLaTr-Score, Seg F1, Coverage | 위 + motion 쪽 TMR-Score/R@3 | **FD_framing**(NDC 2D joint 분포 Fréchet) + **Out-rate**(9 joint 전부 화면 밖 프레임 비율) | — | — | 없음 |
| **AdaViewPlanner** | **Jerk_t / Jerk_r**, **Shot Diversity**(캐릭터 기준 거리·시야각 360° 분포) | **TCC**(MLLM 0~2) | **HMR**(Human Missing Rate) | Reproj. MSE/IoU | — | Gemini 2.5 Pro **TCC/CSD**, user study Preference Rate |

`CSD` = viewpoint type / shot scale / movement type 범주 **엔트로피**.

### 우리가 이미 갖고 있는 것 (`eval_my/*/metrics_full.json` 실측 키)
`val/captions/{precision,recall,fscore}` · `val/clatr/{clatr_score,precision,recall,density,coverage,fcd}`
· `val/loss_{latent,traj}`. → **E.T./GenDoP/VERTIGO/PulpMotion 의 궤적축은 이미 구현돼 있다.**
구현체도 로컬에 있다: `main/evaluate/eval/src/metrics/modules/{prdc.py,fcd.py,clatr_score.py,caption.py}`.

---

## 8. metric 계획에 대한 비판 8건

### C1 🔴 CLaTr 가 **이 코퍼스에서 검증되지 않았다** — 가장 큰 구멍
`main/conf/config.yaml:600-605` 가 스스로 적고 있다: `epoch139_large` 는
`epoch109_dl3dv_seg_2` 와 아키텍처는 같지만 *"trained on a DIFFERENT corpus"* 라
2026-07-30 이전 수치와 비교 불가. 더 근본적으로 CLaTr 는 **E.T.(합성 Blender, character-centric)
궤적 분포**에서 학습된 contrastive embedding 이다. 우리 궤적은
(a) `rel[0]=I` 고정 (b) preset 13종 × τ 사다리라 **이산적** (c) 캡션이 우리 템플릿.
이 셋이 CLaTr latent 에서 모드 붕괴를 만들면 FCD/PRDC 가 품질과 무상관이 된다.

실측 증거 — **FCD 는 n 에 강하게 편향된다**:
`snowboard_k6track_last` n=77 → fcd **397.36** / `dl3dv_geo_worldtraj` n=3980 → fcd **26.55**.
Fréchet 거리는 공분산 추정이라 소표본에서 폭증한다. **eval_my 에 이미 n 이 다른 짝이 나란히 있다.**
- 해야 할 것 ① CLaTr 를 **우리 코퍼스에서 재학습** + held-out 씬 retrieval R@k 로 sanity
  (E.T. 원논문이 CLaTr 를 그렇게 검증한다) ② FCD 는 **동일 n subsample** 에서만 보고
  ③ PRDC `manifold_k=3` 이 n=77 에서 무의미함을 명시 (`prdc.py:57`)

### C2 🔴 caption F1 은 고장이 이미 진단됐는데 안 고쳤다
`CinemaTraj/summary.md:313-402`: `CaptionMetrics` 의 `cam_static_threshold=0.02` 가
**절대 world 단위**라 TRUMANS 는 45.2% 축이 임계 경계에 앉고, `27×7=189` 클래스 **완전일치**
weighted F 라 부분점수가 없다 → fscore 가 0~0.09 를 추세 없이 튄다 (DataDoP fscore 정확히 0 과
같은 현상). Vista4D/dynpose 는 2.7~3.3% 라 덜하지만 **τ 사다리 하단(0.10) 변이는 구조적으로
그 임계 근처**다.
- 우리만 가진 이점: GenDoP/ShotVerse 의 F1 은 motion-tag **분류기** 출력인데 우리는
  **`preset` 이라는 GT 라벨을 이미 갖고 있다.** 189클래스 완전일치 대신
  **preset 분류 정확도 + primitive 축별(dolly/truck/pedestal/pan/orbit/crane/hold) F1**
  로 바꾸면 임계 문제가 사라지고 부분점수도 생긴다. 이건 논문에 쓸 수 있는 개선이다

### C3 🟡 framing 을 이진값으로만 재면 VERTIGO/PulpMotion 보다 약하다
지금 `subject_in_frame` = "실루엣 중심이 중앙 80% 안" 이진 판정의 프레임 비율
(`verify.py:118-135`). VERTIGO **MisR**(off-screen 또는 외곽 20%)과 **정의가 사실상 같다** →
여기는 이미 맞다. 빠진 건 PulpMotion 쪽이다:
- **FD_framing** 은 "얼마나 자주 안에 있나"가 아니라 **"GT 와 같은 구도인가"** 를 잰다.
  우리는 subject OBB 8꼭짓점 투영을 이미 갖고 있다(`sample_camera_bank.composition_stats`)
  → **투영 OBB 의 (cx, cy, log area) 분포로 FD_framing 을 그대로 만들 수 있다**
- 이진 지표는 캡션 약속을 못 잰다: *"medium shot **that widens to** a medium wide shot"* 은
  D119 `subject_area_seq` 가 이미 시퀀스로 갖고 있다 → **shot_scale_seq DTW** 축이 필요
- 제안 3단: `MisR` + `FD_framing` + `shot_scale_seq DTW`

### C4 🔴 collision rate 에 SDF 가 없고, 그건 **원리적 한계**다
CinemaTraj 는 ScanNet++ **메시**에서 `SDF<0` 로 Collision Rate 를 낸다. 우리는 정적 메시가 없고
dynamic video 의 **depth shell** 뿐이다. 낼 수 있는 건 `behind_surface_frames`(관측 표면 뒤) +
`obb_clearance`(노드 박스 침투) 두 **proxy**. 나쁘지 않지만 **반드시 같이 보고할 두 가지**:
- (i) **`judgeable_frac`** — G1 이 판정 불가(sky/무관측)인 프레임 비율. `fix.log:60 F4` 가
  "너무 높다"고 이미 적었고 **아직 열이 없다.** 이게 없으면 collision 0% 가 "안 부딪혔다"인지
  **"볼 수 없었다"** 인지 구분이 안 된다. VERTIGO/CinemaTraj 는 이 문제가 없다(엔진/메시가 있다)
- (ii) F11(`--behind_src_frames` 13→49)이 D121/D122 러너에 적용됐으므로 d107/d110 로 학습한
  arm 과 d122 arm 의 collision 은 **다른 자로 잰 값**이다
- ✅ **정직한 해법이 하나 있다: TRUMANS.** mesh GT 가 있으므로 `lbm/mesh_collision.py` 로
  **진짜 SDF collision rate** 를 낼 수 있다. → **TRUMANS 를 "학습 데이터 하나 더"가 아니라
  in-the-wild proxy 의 캘리브레이션/검증 세트로 승격**시키는 것이 이 metric 의 핵심 설계다.
  같은 논리로 TRUMANS 는 SMPL joint GT 가 있어 PulpMotion 의 FD_framing/Out-rate 를
  **joint 기준으로** 그대로 낼 수 있다 (in-the-wild 는 OBB proxy)

### C5 🔴 최종 VLM 평가가 **씬 n=4** 위에 서 있다
실측: `latentcam_da3_k6_d121/seg_list_vista4d_test.txt` = **875 세그먼트 / 씬 4편**
(avocado-slice, bmx-bumps, camel, couple-hug). VLM 이 영상을 몇 개 채점하든 **씬 단위 독립표본은
4개**다. 씬 간 분산이 변이 간 분산보다 크므로 (D122 실측이 그것: parkour 정적노드 11개 vs
snowboard 0건) 4씬 위의 결론은 씬 우연이다.
비용 실측 (`video_generation/results/20260819_vista4d_eval`): 110영상 4h24m / 4 GPU =
**575 s/영상**. 875 세그먼트 전량 = **140 GPU-h**.
- 제안: 최종 VLM 평가를 **d122 dynpose(266씬)** 위에 세운다. **씬 30~50 × 변이 3~5 =
  100~250 영상 = 16~40 GPU-h** 로 씬 단위 n 을 30~50 으로 올린다. 지금 계획은 렌더를 아무리
  늘려도 n=4 다. (vista 4씬은 d77 과 눈금 맞추기용 paired 축으로 남기고 별도로 본다)

### C6 🔴 이 레포는 **자기 VLM 이 그림을 잘 안 본다는 것을 이미 증명해 놓았다**
`DECISIONS.log` D34/D35/D36 + `scripts/ablate_vlm_hole_perception.py` 실측:
traj 턴은 9조건×2영상×3draw=54 draw 중 49가 같은 답. 이름 중립화(`M01..M13`)·행 셔플·
temperature 1.0·숫자 반전·**가짜 magenta 칠하기** 다섯 개 전부 못 흔들었다. select 턴은 숫자 우세.
구멍 많은 씬(avocado)에서는 그림 채널이 아예 죽는다.
→ **그 VLM 을 최종 평가자로 그대로 쓰면 같은 prior 가 점수로 나온다.** 최종 VLM 평가에 반드시:
- (a) **위치 셔플 + 이름 중립화** — 기존 ablation 하네스를 평가에 재사용
- (b) **sanity probe 삽입** — GT 변이 렌더(상한) / 프롬프트 무작위 셔플 짝(하한) /
  subject 를 프레임 밖으로 밀어낸 조작본. 세 개에서 순서가 안 나오면 그 라운드 무효
- (c) **rule-based 지표와의 상관계수를 같이 보고** — AdaViewPlanner 가 TCC(MLLM)와 HMR(rule)을
  나란히 내는 이유. 상관이 없으면 VLM 이 다른 걸 재고 있는 것
- (d) 직접 Likert 대신 **VERTIGO cyclic semantic similarity**: 생성 영상 → VLM 이 카메라 캡션을
  **역생성** → 원 intent text 와 임베딩 코사인. 논문이 "VLM 은 fine-grained 수치 판단에 약하다"고
  적고 그래서 cyclic 을 쓴다

### C7 🔴 pseudo-GT 로 학습하고 **같은 게이트로 평가**하는 순환성
우리 GT 는 우리 게이트가 통과시킨 카메라다. 그 GT 로 학습하고 같은 게이트로 평가하면
**"게이트를 잘 흉내내는가"** 를 재는 것이고 "좋은 카메라인가"는 안 재진다.
논문 계보가 이걸 푸는 세 방식: (i) CinemaTraj = Blender **수작업 GT** 와의 Motion MSE,
(ii) VERTIGO = **사람 선호** + 별도 fine-tuned VLM, (iii) GenDoP/PulpMotion = **사람이 찍은
실제 영화 궤적 분포**와의 FD.
- 우리에게 (iii)이 가장 값싸게 열려 있다 — 뱅크에 `dd_*`(DataDoP 유래) 변이가 이미 **49.6%**.
  **그런데 그건 학습 데이터다.** 평가에 쓰려면 DataDoP shape 을 학습에서 빼야 한다
- 제안: DataDoP 변이를 **씬 단위가 아니라 shape 단위로 holdout** 하고
  "우리 모델 출력 분포 vs 실제 영화 궤적 분포" FD 축을 하나 만든다.
  **순환성을 깨는 유일한 무료 축이다.** 지금은 섞여 있어 이 축이 닫혀 있다
- (i)도 부분적으로 열려 있다: TRUMANS preset 격자를 Blender 로 렌더하는 경로가 이미 있다
  (`scripts/bank_to_blender_poses.py`, D105)

### C8 🟡 `val/loss_traj` 는 다양성을 **처벌**한다
CinemaTraj 의 Motion MSE 자리에 우리는 `val_loss_traj = F.mse_loss(traj_pred, traj)`
(`train_latent_cam_dm.py:825`) 를 쓴다. 문제 둘:
(a) 한 (씬, 캡션) 에 GT 변이가 **여러 개**인데 표본 1개와만 비교한다 → 다른 유효 변이를 낸 샘플이
벌을 받는다. CFG 를 키우면 좋아지고 다양성은 죽는다.
(b) arc-length resample 이 없어 speed 축(steady/accel/decel/ease)이 거리로 섞인다.
- 제안: (씬,캡션) 당 K개 샘플링 → **min-over-K** 또는 GT 변이 집합에 대한 **chamfer**.
  CinemaTraj 처럼 **arc-length resample 후** 재기

### C9 🟡 `hole_fraction` 의 **상한이 정의되지 않았다** — 그런데 이게 제일 논문거리다
`eval_subject_in_frame.py` docstring 이 `hole_fraction` 을 *"맥락용"* 이라고만 적는다. 그러나
최종 목표는 *"하류 video model 이 메워야 할 hole 이 **감당 가능한**"* 카메라다 — hole 은 0 이
최적이 아니라 **상한이 있는 제약**이고, 그 상한은 **Vista4D 가 실제로 몇 %까지 메우는가**로만
정해진다. 그리고 그건 측정 가능하다: 사다리가 이미 `0.10/0.20/0.35/0.50` 4단이다.
→ **"hole 사다리 → 하류 생성 품질" 곡선**을 재면 (a) 뱅크의 사다리 설계에 근거가 생기고
(b) "camera-first" 주장의 정량 근거가 되고 (c) 어떤 논문도 이 곡선을 갖고 있지 않다.
**지금 아무도 안 재고 있다.** 비용: 4단 × 씬 20 × 변이 2 = 160영상 = 26 GPU-h.

---

## 9. 모델 구성(DA3 + text + VLM feature)에 대한 비판 4건

### M7 🔴 VLM feature 를 "물체 탐지·추적" 목적으로 쓰려면 **지금 붙이는 자리가 틀렸다**
D124 는 Molmo2 `last_hidden_state` 를 video CA 로 붙였다. 그런데 캐시 불변조건 (b) —
*"patch 위치 hidden 은 캡션이 달라도 동일 (chat template 이 `<|video|>` 를 앞으로 hoist +
LM causal)"* — 이 뜻하는 것은 **video 토큰 3136개에 "무엇을 겨냥하라"는 정보가 하나도 없다**는
것이다. 캡션 정보는 text 토큰 128개에만 있고, 그 128개가 같은 CA 의 key/value 로 3136개와
섞여 들어간다. 즉 **Molmo2 의 grounding/pointing 강점이 구조적으로 쓰이지 않는다.**
3택:
- (i) **Molmo2 pointing 으로 타깃 per-frame 2D 좌표 + 가시성을 뽑아 명시 채널로** 넣는다.
  자리가 이미 있다 → `cond_dim` / `target_track_dim` (per-token concat)
- (ii) `(scene, caption)` 조건부 video hidden 을 굽는다 → 13,679배, 비현실적
- (iii) video 토큰은 **씬 grounding 전용**, 타깃 지시는 (i)로 분리
→ **(iii)+(i) 조합만 값싸다.** 근거는 PulpMotion: 암시적 CA 로는 on-screen 일관성이 안 나와서
framing 을 **명시 모달리티**로 승격시키고 linear transform + auxiliary sampling 을 넣었다.
AdaViewPlanner 도 SMPL-X 22 joint 를 **명시 입력**으로 준다.

### M8 🟡 `target_track` 조건은 **이미 검증됐는데** d121 arm 에서 꺼져 있다
`FIX.log` 실측 (`vista4d_pgt_k6_track`, last.pth, n=16):
track 조건 있음 `val/loss_traj 0.01519` vs `--drop-track` **0.02639** vs
track 미학습 arm 0.03544. → **조건이 실제로 먹는다.** 그런데 d121/d123/d124 config 에
`target_track_dim` 이 없다. VLM feature 를 CA 로 넣는 것보다 **먼저 켤 것**이 이쪽이다.

### M9 🟡 `rel[0]=I` 때문에 **Shot Diversity / CSD 를 낼 수 없다**
AdaViewPlanner 의 **Shot Diversity**(캐릭터 기준 거리·시야각 360° 분포)와 **CSD**(viewpoint
type / shot scale / movement type 엔트로피)는 시작 구도가 자유롭지 않으면 축이 죽는다.
지금 CSD 를 재면 **preset 13종의 엔트로피**만 나오고 그건 우리 샘플러가 정한 상수다.
→ `goals.md` 중기 3(시작 구도 다양화)은 표현력 문제만이 아니라 **"이 지표들을 낼 수 있는가"**
문제다. 지금 상태로는 "다양한 구도를 낸다"를 주장할 지표 자체가 없다.

### M10 🟡 geo(DA3) 에 **"어디가 관측됐는가"** 채널이 없다
ShotVerse 는 PI3 로 정적 배경을 재구성해 전역 좌표계를 만들고, CinemaTraj 는 완전 메시를 갖는다.
우리는 depth shell 뿐 → "물러나면 무엇이 보이나"를 모른다. 그게 hole 이고, 그래서 hole 사다리가
τ 를 누른다. 장기 1(world 확장)이 정답이지만 **값싼 중간 단계가 있다**:
소스 49프레임 union 점군을 이미 `cloud.npz` 로 쌓고 있고 `obs_az_span` clamp 가 그 아이디어를
샘플러 쪽에서 쓴다(소스가 본 방위로만 돈다). 그런데 **geo encoder 에는 그 union 이 안 들어간다**
— DA3 는 6뷰 uniform 만 본다. **"이 방위/거리는 관측됐다" 를 명시 채널로 주는 것**이
모델이 hole 을 스스로 피하게 하는 가장 직접적인 방법이다 (M3 의 `geo_cam_embed` 자리와 같은 층).

---

## 10. 최종 평가 설계 제안 — 3층, 각 층이 위층의 sanity check

| 층 | 무엇 | 표본 | 비용 | 지표 |
|---|---|---|---|---|
| **L1** | 렌더 0회, 순수 기하 | **전량** (d122 전량 가능) | 분 단위 | MisR · collision proxy + **judgeable_frac** · obb clearance 위반율 · Jerk_t/Jerk_r · path_len_u · Shot Diversity · CSD |
| **L2** | depth-warp 렌더 | **전량** | 시간 단위 | **FD_framing** · **shot_scale_seq DTW** · hole_fraction · subject_pixel_coverage · MisR(렌더 실측) |
| **L3** | Vista4D 생성 + VLM | **씬 30~50 × 변이 3~5** | 16~40 GPU-h | cyclic semantic sim · pairwise A/B · FVD · LAION Aes. · ViCLIP |

L3 설계 규칙 4개 (C6 근거):
1. **cyclic** 을 1급으로 (직접 Likert 는 보조)
2. **pairwise A/B** (우리 vs GenDoP vs GT-변이 렌더) — 절대 점수보다 VLM 이 일관된다
3. **sanity probe 3종 삽입** (GT 상한 / 프롬프트 셔플 하한 / subject 밀어낸 조작본).
   순서가 안 나오면 라운드 무효
4. **L1·L2 와의 상관계수 동시 보고**. 무상관이면 VLM 점수 폐기

그리고 **모든 층에서 pseudo-GT 궤적 자체를 같은 파이프로 통과시킨 점수(= 천장)를 같이 낸다.**
pseudo-GT 로 학습하는 프로젝트에서 이게 없으면 점수의 상한을 모른다. TRUMANS 에서는
그 천장이 **진짜 GT**(mesh + SMPL joint)라 in-the-wild proxy 를 캘리브레이션할 수 있다.

---

## 11. 우선순위 (5절 표에 이어서)

| # | 항목 | 비용 | 재학습/재굽기 | 왜 |
|---|---|---|---|---|
| 9 | **C4** `judgeable_frac` 열 추가 | 작다 (`fix.log` F4) | 열만 | 없으면 collision 0% 를 해석 못 한다 |
| 10 | **C5** 최종 평가를 d122(266씬)로 이전 | 설계 | 불필요 | 지금 계획은 렌더를 늘려도 씬 n=4 |
| 11 | **C4/TRUMANS** mesh SDF collision + SMPL framing 을 캘리브레이션 세트로 | 중간 | 불필요 | proxy 를 검증할 유일한 자리 |
| 12 | **C2** caption F1 → preset/primitive 축 F1 | 작다 | 불필요 | 지금 지표는 추세가 없다 (0~0.09 랜덤) |
| 13 | **C1** CLaTr 우리 코퍼스 재학습 + R@k sanity | 중간 | CLaTr 만 | FCD/PRDC 의 유효성 근거가 지금 없다 |
| 14 | **C9** hole 사다리 → 하류 품질 곡선 | 26 GPU-h | 불필요 | 사다리 설계의 근거 + 아무 논문도 안 가진 곡선 |
| 15 | **C7** DataDoP **shape 단위 holdout** | 코퍼스 재분할 | seg_list 만 | 순환성을 깨는 유일한 무료 축 |
| 16 | **M8** `target_track_dim` 켜기 | config | 1 arm | 이미 실측으로 먹는 것이 확인됨 |
| 17 | **M7** Molmo2 pointing → 명시 track 채널 | 중간 | 1 arm | 지금 video CA 는 타깃 정보를 안 들고 있다 |
| 18 | **C3** FD_framing + shot_scale DTW | 작다 | 불필요 | 이진 지표로는 캡션 약속을 못 잰다 |
| 19 | **C8** min-over-K / chamfer + arc-length | eval 수정 | 불필요 | 지금 loss_traj 는 다양성을 처벌 |

---

# Part III — pseudo-GT 카메라 생성 파이프라인 분석 (2026-09-04 추가)

이 절의 수치는 **이번 세션에 `bank.json` / `scene_graph.json` 을 직접 집계한 실측**이다
(읽기만, GPU 0회). 대상: `out/*/hole_bank_k6_d121` 52씬 29,964변이 ·
`out_dynpose/*/hole_bank_d122` 266씬 23,424변이 · `scene_graph.json` 53 + 280편.

## 12. 파이프라인 7단계와 각 단계의 성격

| # | 단계 | 산출 | 성격 | 문헌 대응 |
|---|---|---|---|---|
| 1 | `build_scene_graph` | 중력축 · 단위 S · 인스턴스 OBB · track · `obs_az_span` | 추정(GeoCalib+DA3+SAM) | ShotVerse 의 SAM+PI3 정적배경 재구성과 같은 자리 |
| 2 | `lbm.cloud` | 4D 점군 `cloud.npz` + meta S | 결정적 | — (씬 자산 대신 depth shell) |
| 3 | `route_presets` | anchor 2 + preset 8슬롯 | **실측 휴리스틱** | CinemaTraj 의 LLM agent 자리를 규칙으로 대체 |
| 4 | `sample_camera_bank` | anchor×preset×τ 격자 + 실측 열 | 열거 | LensCraft 의 compiler+simulator, VERTIGO 의 LenScript |
| 5 | `fit_hole_ladder` | hole 예산 이분법 + 게이트 6종 | **최적화(1D 이분법)** | CinemaTraj 의 SDF collision 최적화 |
| 6 | `emit_bank` | canonical 21 pose | 결정적 | — |
| 7 | `build_bank_captions` | 자연어 캡션 | 템플릿 합성 | DataDoP 의 motion tag→GPT-4o 2단 캡션 |

**설계의 정체**: survey(2506.00974)의 4분류로 보면 **rule-based 열거 + optimization(이분법) 하이브리드**이고,
표현 3층으로 보면 mid-level(preset 어휘) → low-level(SE(3) 로그 스케일)이다.
survey 가 "rule-based 는 동적 환경에서 rigid" 라고 적은 칸에 정확히 우리가 서 있다 —
다만 **게이트가 그 rigidity 를 실측으로 깎는다**는 게 우리 방식의 답이다.

## 13. pseudo-GT 만드는 세 계열과 우리 위치

| 계열 | 논문 | 규모 | GT 완전성 | 약점 |
|---|---|---|---|---|
| **A. 시뮬레이터 합성** | LensCraft(2506.00988) 100K/3M frame · VERTIGO LenScript(Unity) 120K · E.T. (Blender+SMPL-H) · CCD | 크다 | **완전** (메시·SDF·joint 전부) | domain gap. LensCraft 스스로 "E.T. 70% 가 static" 이라고 비판 |
| **B. 실영상 pose 추정** | DataDoP/GenDoP 29K shot·11M frame · PulpMotion 193K·314h(CondensedMovies+TRAM+RePaint) · ShotVerse 20.5K clip(SAM+PI3) | 크다 | 궤적만 | **intent 라벨이 없다**(GenDoP 이 target 절 없이 free-moving 으로만 넣는 이유) · 씬 기하 없음 |
| **C. 씬 위에서 규칙 합성** | CinemaTraj(ScanNet++ mesh + LLM + atomic 7종 + SDF) · Look-Before-Move(Blender) | 작다 | 완전한 3D 전제 | **정적 씬만** (CinemaTraj 한계 1번) |

**우리는 C 를 실영상(=B의 입력)에 옮긴 것이고, 이 조합은 문헌에 없다.**
그게 novelty 이고 동시에 모든 어려움의 근원이다 — C 는 완전한 3D 를 전제하는데 우리는
**관측된 표면만 있는 depth shell** 이다. 그래서 우리가 발명해야 했던 것이 **hole 사다리**다.
LensCraft·CinemaTraj 에는 hole 개념 자체가 없다(씬이 완전하니까). **이건 진짜 기여이고,
Part II C9 의 "hole 사다리 → 하류 품질 곡선" 이 그 기여를 정량화하는 유일한 실험이다.**

## 14. 뱅크 두 벌 실측 (이번 세션 집계)

| | **d121 vista** (학습 중) | **d122 dynpose** |
|---|---|---|
| 씬 / 변이 | 52 / 29,964 | 266 / 23,424 |
| `solved` | **42.2%** | 36.7% |
| `clamped_low+tau_floor` | 7.9% | **15.4%** |
| binding = hole / obb / approach / collision / ground / elev | 43.0 / 12.1 / 7.5 / 7.4 / 8.4 / 6.6 % | 40.1 / 7.2 / **14.3** / 9.1 / 7.7 / 8.1 % |
| `path_len_u == 0` | **17.6%** (5,272) | 7.2% (1,698) |
| ↳ 그중 rotation-only preset / `status=static` / **게이트에 뭉개짐** | 4,672 / 600 / **0건** | 1,492 / 2 / **204 (0.87%)** |
| `subject_in_frame` med / `<0.85` / `==0` | 1.000 / 27.8% / **6.0%** | 1.000 / **41.8%** / 1.3% |
| `tau_start` med / `>0.10` | 0.0170 / 20.7% | **0.1016 / 50.3%** |
| `hole_static` med / p90 | 0.079 / 0.391 | 0.122 / 0.326 |
| `hole_fraction` med / p90 | 0.307 / 0.544 | 0.289 / 0.570 |
| `dd_*` (DataDoP 유래) | **0 (0.0%)** | 11,640 (49.7%) |
| 고유 preset 수 | 40 | 208 (dd shape 포함) |

`scene_graph` 실측:

| | d121 vista (53편) | d122 dynpose (280편) |
|---|---|---|
| `gravity.method` | geocalib 52 / ground_ransac 1 | **geocalib 280 (100%)** |
| `scale.mode` | points_first_cam 52 | points_first_cam 280 |
| 노드/씬 med / p90 | 9 / 16 | 6 / 6 |
| `obs_az_span` med / ≥120° / <20° | **26.3° / 18.0% / 44.1%** | 72.2° / 27.2% / 19.2% |
| `moving` 노드 비율 | 24.6% | **94.6%** |

---

## 15. 개선안 — 파이프라인

### P1 🔴🔴 **캡션이 target 을 약속하는데 기하는 targetless 다 — 코퍼스의 26~31%**
D90 이 이름 규칙을 뒤집으면서 **free-moving primitive(dolly/truck/pedestal/pan/tilt)의 기본이
`aim="free"`(조준 안 함)** 가 됐다 (`lbm/presets.py:120-147`). 그런데
`configs/caption_presets.json` 은 **43 preset 중 5개만** `targetless: true` 이고(pan/tilt 뿐),
`phrase_targetless`(= `--targetless_promote` 대상)는 **2개**뿐이다 (`track_truck_left/right`).

즉 `dolly_in` · `dolly_out` · `truck_left/right` · `pedestal_up/down` · `track_dolly_*` 는
**기하적으로 조준을 안 하는데 캡션에는 target 절이 붙는다.** 실측:

| | d121 vista | d122 dynpose |
|---|---|---|
| 기하 targetless & 캡션 target 붙음 | **9,258 (30.9%)** | **6,076 (25.9%)** |
| ↳ `subject_in_frame` med | 0.846 | 0.923 |
| ↳ `<0.85` | **50.0%** | 47.2% |
| ↳ `==0` (49프레임 내내 중앙에 없음) | **11.9%** | 3.4% |
| `aim="free"` 전체 sif med / `<0.85` / `==0` | 0.769 / **55.0%** / **13.0%** | 0.846 / 53.0% / 3.6% |
| `aim="look_at"` 전체 (대조) | 1.000 / **4.1%** / 0.0% | 1.000 / 16.7% / 0.0% |

**`goals.md` 는 이 결함을 `track_truck_*` 1,552변이(7.2%) 문제로 적었는데, 실제 모집단은
그 6~15배다.** `--targetless_promote` 는 d121 에서 624/9,258 = **6.7%** 만 덮는다.
- 즉시(재굽기 불필요, 캡션만 재빌드): promote 판정을 `phrase_targetless` 유무가 아니라
  **`variant["aim"] != "look_at"`** 으로 바꾼다. 뱅크 행에 `aim` 열이 이미 있다
- 중기: **Toric space** (Lino & Christie, SIGGRAPH'15) — 2~3 subject 의 **정확한 화면 위치**를
  만족하는 카메라 집합을 2D manifold 로 닫힌 형태로 준다 (7DoF→4DoF). 지금은 **preset 이 궤적을
  정하고 framing 은 사후 측정**인데, toric 을 쓰면 순서가 뒤집혀 **preset 모양을 유지하면서
  framing 을 보장**할 수 있다. CCD / Camera-keyframing 계열이 전부 이 공간을 쓴다

### P2 🔴 **학습 중인 d121 코퍼스에 DataDoP 이 0% 다**
실측 `dd_*` = **0 / 29,964**. `goals.md` §2 의 "dd_* 49.6%" 는 **d122 dynpose 만**이다.
즉 지금 도는 D123/D124 두 arm 은 film 유래 motion 을 **한 번도 안 본다.**
- 결정 필요: vista 를 d122 방식(route_presets + `--external_shapes`)으로 다시 굽거나,
  "vista = preset only / dynpose = preset+dd" 를 **의도된 대조축으로 명시**하거나
- 참고로 d122 실측에서 `dd_*` 는 `solved` 29.5% vs preset 43.8% — **DataDoP shape 이 fit 난이도가
  높다**(외부 rel 행렬이라 게이트가 더 자주 물린다). 섞으면 solved 비율이 내려간다

### P3 🔴 **dynpose 절반이 소스 카메라 자신의 τ 예산을 초과한다**
실측 `tau_start` median **0.1016**, `>0.10` 이 **50.3%** (vista 는 0.0170 / 20.7%).
`--tau_floor_src` 가 하한을 `tau_start + 0.02` 로 올리므로 **사다리 하단(hole 0.10)이 dynpose
절반에서 사실상 사라진다** → `clamped_low+tau_floor` 15.4%.
- (i) `--follow_gains auto`(D72)가 정확히 이 처방인데 **d122 러너가 안 쓴다.** README 실측:
  snowboard τ 1.9441 로 뱅크 360 중 350 이 saturate 하던 것을 auto 가 회수한다.
  소스가 회전으로만 따라간 씬은 g*≈0 이 나와 저절로 꺼지므로 **켜서 잃는 게 없다**
- (ii) 또는 dynpose 사다리를 위로 옮긴다 (`0.20/0.35/0.50/0.70`). 지금 vista 와 **같은 사다리**를
  쓰는데 소스 시차 분포가 6배 다르다 — 같은 눈금이 두 코퍼스에서 다른 난이도를 뜻한다

### P4 🟡 **`route_presets.py` 의 중력 분기가 죽은 코드다**
주석은 *"`camera_up_fallback`(257편 중 141편)이면 세로 슬롯을 뺀다"* 라고 적는데, 실측은
**geocalib 280/280** (vista 도 52/53). D122 재굽기에서 `--gravity_source auto` 가 전편 geocalib
으로 풀렸다 → **그 분기가 한 번도 안 탄다.** 판단 근거가 stale 하다.
→ 주석 정정 + "세로 슬롯(pedestal/crane)이 이제 전 씬에서 유효하다"를 라우팅에 반영

### P5 🟡 **`moving` 플래그가 dynpose 에서 정보를 잃었다**
실측 moving 노드 비율 vista **24.6%** vs dynpose **94.6%**. `route_presets` 의 첫 신호가
`node.moving` 인데 94.6% 면 분기가 사실상 상수 → dynpose 는 거의 전부 `track_*` 로 라우팅된다.
`fix.log F1`(제자리형 subject 를 static 으로 라우팅)이 이 문제를 다루려 했지만 미승인이다.
README 가 이미 원인을 적어 뒀다 — *"정적 노드도 OBB 중심이 벽을 따라 미끄러진다
(truck-pose `stat_0` 0.535u)"*. → **dynpose 에서 moving 임계를 다시 잡는 것이 F1 의 실측 근거다**

### P6 🟡 **orbit 이 vista 에서 구조적으로 불가능하다**
실측 `obs_az_span` median vista **26.3°**, **노드의 44.1% 가 <20°** (dynpose 72.2° / 19.2%).
`--min_sweep_deg 20` 이 바닥을 깔지만 그건 "소스가 못 본 면을 보게 한다" = **hole 을 만든다.**
즉 vista 의 orbit 변이는 대부분 **hole 로 대가를 치른 orbit** 이다.
- `route_presets` 에 이미 `ORBIT_MIN_SPAN_DEG=120 → s_curve 대체` 규칙이 있는데
  **d121 은 라우팅을 안 쓰고 전량 격자를 돌았다** (preset 40종 × 1,168 완전 균일이 그 증거).
  d121 을 다시 구울 때는 `route_presets` 를 태워야 한다

### P7 🔴 **LensCraft 의 3대 비판이 우리에게 그대로 적용된다**

**(a) "E.T. 는 70% 가 static" → 우리는 어떤가.** 실측 `path_len_u==0` 이 d121 **17.6%** /
d122 7.2%. **단 이번 집계로 그 대부분이 rotation-only preset 임이 확인됐다**:
d121 5,272 중 pan/tilt 4,672 + `status=static` 600, **게이트에 뭉개진 건 0건**.
d122 는 1,698 중 pan/tilt 1,492 + 게이트 **204(0.87%)** — `goals.md` 의 197건과 일치.
→ 결론: LensCraft 식 static 편향은 **우리에게는 "pan/tilt 를 얼마나 넣을지"의 설계 선택**이고
사고가 아니다. **그런데 지표 쪽에 구멍이 생긴다** — rotation-only 는 `path_len_u = 0` 이라
**`path_len_u` 기반 필터·지표가 pan 을 정지와 구별하지 못한다.** `pan_deg` 열은 있지만
preset 전반의 **총 회전량(`rot_deg`) 열이 없다.** 열 하나 추가로 끝난다

**(b) "SLAM 부정확성이 전파된다" → 우리는 DA3.** 실측 근거가 이미 repo 안에 있다:
DA3 는 프레임마다 focal 이 흔들려(정지 카메라에서 **−14% 드리프트**) `--fixed_focal` 로 K[0] 를
49프레임에 복사한다 (`vista4d_bank_to_dl3dv.py` docstring). 게다가 `pose_source: da3` +
`target_pose_source: da3_target_poses` 라 **소스 pose 도 추정치**다.
→ **pose 불확실성을 뱅크 열로 남겨야 한다** (DA3 confidence / reprojection residual / focal 드리프트
폭). 지금 그 열이 없어서 "이 씬의 pseudo-GT 는 얼마나 믿을 만한가"를 하류가 알 수 없고,
**씬 가중치도 못 준다.** ShotVerse 가 SAM 으로 동적 전경을 지우고 정적 배경만 PI3 로 재구성해
안정화하는 것이 같은 문제의 다른 해법이다 — 우리도 `dynpose_dynamic_mask_from_seg.py` 가 있다

**(c) "subject 를 점으로 다뤄 부피·방향을 무시한다" → 우리는 이미 OBB 를 쓴다. 문헌 대비 우위.**
다만 조준점은 **OBB 중심 + `look_at_bias`(높이 비율)** 하나뿐이다. LensCraft 의
**ABox(주목 박스) / VBox(전체 부피)** 이원화 + shot type 별 보간(ECU = ABox 0.0 →
ELS = VBox 3.0×)을 도입하면 shot scale 이 **사후 측정(`subject_area_med`)이 아니라 사전 제약**이
된다. 지금은 shot scale 이 **결과**이고 캡션이 그걸 읽는다 — 거꾸로다.
`subject_area_seq`(D119)가 이미 시퀀스를 갖고 있으니 목표 시퀀스를 주는 쪽으로 뒤집을 수 있다

### P8 🟡 `judgeable_frac` 만 남았다 — G1 채널 분리는 이미 됐다
실측 bank 열에 `behind_static_frac` / `behind_dyn_frac` / `behind_static_worst` /
`behind_dyn_worst` 가 **이미 있다.** 즉 `fix.log F3`(관통 vs 너무 붙음)의 절반은 적용됐다.
남은 건 `fix.log F4` 의 **판정 불가 비율**뿐이고, 그게 Part II C4 의 필수 열이다

### P9 🟡 hole 사다리에 **절대 상한이 없다**
`--hole_mode excess`(기본)는 단을 `hole_static + Δ` 로 잡는다. 실측 `hole_static` p90 이
d121 **0.391** / d122 0.326 — **상위 10% anchor 는 카메라를 안 움직여도 hole 이 33~39%** 다.
거기에 Δ=0.50 을 얹으면 hole 0.9 짜리 변이가 나온다. 실측 `hole_fraction` p90 = 0.544 / 0.570 이
그 결과다. 하류가 절반 이상을 지어내야 하는 궤적이 뱅크에 그대로 있다.
→ **absolute 상한 하나를 같이 걸어야 한다.** 그 값은 Part II C9("Vista4D 가 몇 %까지 메우나")
실측이 정한다. `fit_hole_ladder.py` 주석이 스스로 *"0.50 은 하류 모델이 절반을 지어내야 하는
지점 — 그 위는 '생성'이지 '카메라 이동'이 아니다"* 라고 적었는데, `excess` 모드가 그 선을 넘긴다

### P10 🟡 복합 카메라 — 조합은 만들 수 있고 **라벨을 쓸 주체가 없다**
CinemaTraj 는 **atomic 7종 + transitional arc** 를 LLM 이 조합한다. 우리는 조합을 만들 수 있는데
(`compose`, `aim_keyframes`) 그 자리가 비어 있다. 그런데 **우리 VLM 은 traj 턴에서 prior 라는
실측(D34~36)이 있으므로 LLM 에게 고르게 하면 안 된다.**
→ 우리 방식과 일관된 답: **조합을 열거하고 캡션을 조합에서 기계적으로 합성**한다
(지금 preset 캡션이 정확히 그렇게 만들어진다). LLM 은 어휘 다듬기에만 쓴다.
선행 확인은 Part I D5 — `keyframe_aim=preset_rel` 로 겹친 회전이 실제로 사는지 한 씬 프로브

### P11 🟡 문서 drift — README 의 aim 표가 한 개정 뒤처졌다
README 는 D76 규약(`_dont_look` 접미사)을 설명하고 `dolly_in`/`dolly_out` 을 **look_at** 으로
분류하는데, `lbm/presets.py:148-150` 은 D90 이후 `dolly_in`=`free` / `dolly_in_look_at`=`look_at`
이다. 뱅크에도 두 이름이 따로 있다. **P1 의 결함이 눈에 안 띈 이유가 이 drift 일 가능성이 크다**
— 문서를 읽으면 dolly 계열이 조준한다고 믿게 된다

---

## 16. 참조 논문 → 빌릴 것 → 우리 파일

| 논문 | 빌릴 것 | 우리 쪽 자리 |
|---|---|---|
| **Toric space** (Lino & Christie, SIGGRAPH'15) | framing 을 사후 측정이 아니라 **닫힌 형태 사전 제약**으로. 7DoF→4DoF 2D manifold | `lbm/presets.py` + `decode/build_poses.py` (P1) |
| **LensCraft** (2506.00988) | **ABox/VBox** 이원 subject 표현 + shot-type 보간 · DSL→compiler→simulator 구조 · 3대 비판(static 편향 / SLAM 전파 / subject=점) | `scene_graph/obb.py`, `scripts/build_bank_captions.py` (P7) |
| **CinemaTraj** (2607.26910) | atomic 7종 + **transitional arc** 조합 문법 · **Collision Rate = SDF<0** · **Object Coverage** | `lbm/presets.py`, `lbm/gates.py` (P10, Part II C4) |
| **GenDoP / DataDoP** (2504.07083) | motion tag → GPT-4o **2단 캡션**(motion / directorial) · target 절 없는 free-moving 규약 | 이미 `retrieve_datadop_shapes.py` 로 사용 중 (P2) |
| **PulpMotion** (2510.05097) | framing 을 **명시 모달리티로 승격**(joint 투영 + 공유 latent linear transform) · **FD_framing / Out-rate** | TRUMANS 에 직접 적용 (Part II C3/C4) |
| **AdaViewPlanner** (2510.10670) | **HMR / Jerk / Shot Diversity / CSD** · SMPL-X 를 **명시 입력**으로 · 생성 영상에서 카메라 추정이 실패한다는 실측 | Part II M7/M9 |
| **ShotVerse** (2603.11421) | SAM(동적 전경 제거) + PI3 로 **전역 좌표계** 확보 → depth shell 한계의 대안 · **CAS** | `scripts/dynpose_dynamic_mask_from_seg.py` (P7-b) |
| **VERTIGO** (2604.02467) | 렌더 프리뷰 → VLM → **DPO**. 우리 게이트가 하는 일을 **학습 신호**로 바꾸는 방식 · **MisR** · cyclic semantic sim | Part II C6 / 미래 post-train |
| **E.T. / DIRECTOR** (2407.01516) | **CLaTr** · character-aware 캡션 | `main/evaluate/CLaTr` (Part II C1) |
| **Look-Before-Move** | look-before-move **순서**(우리 board 의 출처) · CineStoryEval **축** | `scripts/build_candidate_board.py` — 정의는 베끼지 말 것(hand-tuned 상수) |
| **survey** (2506.00974) | 방법 4분류(rule / optimization / ML / hybrid) + 표현 3층(NL / PSL / 7DoF·Toric·Plücker) · "데이터셋 다양성 부족"이 1번 open problem | 우리 방식의 논문 내 위치 서술 |

**우리만 갖고 있는 것 (문헌에 대응이 없는 것)**
1. **hole 사다리** — 관측 안 된 영역을 예산으로 다루는 축. 완전한 씬을 쓰는 A·C 계열에는 개념이 없다
2. **소스 대비 임계** — 게이트 임계를 절대값이 아니라 *"소스 카메라 자신의 여유 × β"* 로 둔다
   (D47/D51/D55/D56, `tau_floor_src`). 절대 마진이 씬마다 무너지는 것을 실측으로 기각한 결과이고,
   **"원본 촬영은 정의상 통과한다"** 는 불변조건을 준다. 문헌의 어느 collision 처리에도 이 구조가 없다
3. **OBB 부피 기반 충돌 + 무엇과 부딪혔는지 라벨**(`obb_node`) — LensCraft 가 "subject 를 점으로
   본다"고 비판한 바로 그 지점을 이미 넘어서 있다

---

## 17. 우선순위 (11절에 이어)

| # | 항목 | 비용 | 재굽기 | 왜 |
|---|---|---|---|---|
| 20 | **P1-즉시** promote 판정을 `aim != look_at` 으로 | 캡션 재빌드만 | 불필요 | 코퍼스 **26~31%** 의 캡션↔기하 불일치. 지금 fix 는 6.7% 만 덮는다 |
| 21 | **P2** d121 의 dd_* 0% 를 결정 (섞기 vs 대조축 명시) | 결정 | 섞으면 필요 | 학습 중 두 arm 이 film motion 을 안 본다 |
| 22 | **P3** dynpose 에 `--follow_gains auto` | 러너 한 줄 | **필요** | `tau_start>0.10` 이 50.3%. 켜서 잃는 게 없다(g*≈0 이면 자동 off) |
| 23 | **P7-a** `rot_deg` 열 추가 | 작다 | 열만 | pan 을 정지와 구별할 지표가 지금 없다 |
| 24 | **P9** absolute hole 상한 | 작다 | **필요** | `hole_fraction` p90 0.544~0.570 — 하류가 절반 이상을 지어낸다 |
| 25 | **P7-b** pose 불확실성 열(DA3 conf / focal 드리프트) | 중간 | 열만 | 씬 가중치를 줄 근거가 지금 없다 |
| 26 | **P4/P5** route_presets 의 stale 분기 2건 정정 | 작다 | 불필요 | 중력 분기는 죽은 코드, moving 은 dynpose 94.6% 로 상수 |
| 27 | **P6** d121 재굽기 시 route_presets 태우기 | 러너 | **필요** | vista 노드 44.1% 가 `obs_az_span<20°` — orbit 을 hole 로 산다 |
| 28 | **P11** README aim 표를 D90 로 갱신 | 작다 | 불필요 | P1 이 안 보인 이유일 가능성 |
| 29 | **P1-중기** Toric space 도입 | 크다 | **필요** | framing 을 사전 제약으로. 중기 2(복합)와 묶는다 |
| 30 | **P7-c** ABox/VBox + 목표 shot-scale 시퀀스 | 크다 | **필요** | shot scale 이 지금은 결과다. 캡션 약속을 기하가 지키게 만든다 |

---

# Part IV — framing 을 제약으로 만들기: toric 이 아닌 길 (2026-09-04 추가)

**사용자 지적: "Toric space 는 정지된 multi-subject 기준 설계라 그대로 적용하기 힘들다."**
맞다. Part III P1(b) 의 toric 제안을 아래로 **교체**한다.

## 18. toric 이 이 파이프라인에서 깨지는 지점 4개

| # | 전제 | 우리 상황 |
|---|---|---|
| 1 | **2-target** 이어야 α(두 대상의 화면 간격)가 정의된다 | 우리는 한 샷에 anchor 1개다 (`--num_anchors 2` 는 씬당 anchor 를 2개 **따로** 쓰는 것). 1-target 이면 toric 은 A 중심 **구(sphere)** 로 퇴화하고 α 가 사라진다 — "toric 을 쓴다"는 말의 실체가 없어진다 |
| 2 | **등식 제약** — 화면 위치를 정확히 맞춘다 | 대상이 움직이면 manifold 가 프레임마다 이동한다. 매 프레임 등식을 풀면 카메라가 튀고, 두 대상의 3D 거리가 변하면 화면 간격 고정이 **강제 dolly** 를 만들어 preset 의 모션 단어와 싸운다 |
| 3 | framing **만** 보장한다 | 우리 세계는 depth shell 이라 toric 해가 곧바로 표면 뒤/hole 로 간다. G1·G5·G6·G7 을 따로 다시 걸어야 하고, 그러면 toric 의 "닫힌 형태" 이점이 사라진다 |
| 4 | **궤적 설계자를 대체한다** | toric 이 위치를 정하면 preset 이 없어진다. **preset 어휘가 캡션의 텍스트 축**이므로 그건 코퍼스의 조건 신호를 버리는 것이다 |

동적 확장판은 존재한다 — **Drone Toric Space** (Galvane · Lino et al., TOG 2018,
*Directing Cinematographic Drones*): 제약을 공간에 내장하고 **동적 타깃** + 물리적 실현가능성 +
다중 드론 조율을 다룬다. 그러나 그건 드론 동역학을 제약으로 갖는 **경로 계획기**이고,
"preset 모양을 유지한다"는 우리 구조와 층이 다르다. → 지적이 옳고, 그대로는 못 쓴다.

## 19. 🔴 더 중요한 발견 — **framing 은 조준(rotation)으로 고칠 수 없다** (D90)

`decode/build_poses.py` D90 주석의 실측이 이 문제의 지형을 바꾼다:

> `preset_rel` 은 `look_at(anchor) @ rel_local[f]` 인데 순수 병진 preset(truck/dolly/pedestal)은
> `rel_local` 회전이 **정확히 항등**이라 `preset_rel` 이 `target` 으로 붕괴한다. 그래서 D76~D89 의
> `aim="traj"` 는 이 preset 들에서 **no-op** 이었다 — `dolly_in` vs `dolly_in_dont_look` 이
> |Δt| 0.0 / |ΔR| 0.022°, 즉 같은 카메라다. `truck_left` 는 frame0→48 총 회전 **71.3°** 인데
> 조준오차가 3.03° — 그 **71° 가 전부 조준이라 이름만 truck 이고 실제로는 orbit 이었다.**

즉 **D90 의 `aim="free"` 는 결함이 아니라 수정이다.** truck 이 truck 이 된 것이다.
따라서:

- `truck_left_look_at` / `pedestal_up_look_at` 같은 **조준 쌍둥이를 추가하는 것은 오답**이다 —
  D90 이 제거한 것을 되살려 truck 을 다시 orbit 으로 만든다. **모션 단어가 깨진다.**
- framing 을 **회전**으로 고치면 → 모션 의미가 깨진다
- framing 을 **위치 재계산**(toric)으로 고치면 → preset 이 깨진다
- **남는 자리는 하나뿐: preset 의 크기(knob)를 framing 이 허용하는 만큼만 키운다.**

## 20. ✅ 제안 F — framing 을 hole 사다리의 **8번째 예산**으로

`fit_hole_ladder.solve_knob.over()` 는 이미 예산 7개의 if/elif 체인이다
(`collision → obb → ground → elev → approach → hole → occlusion`, `:487-510`).
여기에 하나를 더 붙인다.

```
framing — anchor OBB 8꼭짓점 투영이 NDC 박스 안에 드는 프레임 비율 < min_in_frame
```

- **렌더 0회.** `--composition`(D121)이 이미 OBB 8꼭짓점을 투영한다 (`composition_stats`).
  `subject_in_frame` 열도 이미 변이마다 있다 — 새로 재는 게 아니라 **판정에 넣는 것**이다
- **위치는 안 건드린다.** 방향(preset 모양)은 그대로고 크기 상한만 내려간다 — 다른 6개 예산과
  완전히 같은 구조다
- **`knob=0` 이 정의상 실행가능.** 시작 pose 가 소스 frame0 이고 `--min_subject_area` 가 anchor 의
  frame0 가시성을 이미 요구한다. 즉 D47/D51/D55/D53 의 **"임계는 소스 대비"** 규약과 같은 꼴이고,
  **원본 촬영을 기각할 수 없다**
- 임계도 소스 대비로: `min_in_frame = min(--framing_floor, 소스 카메라 자신의 in-frame 비율)`.
  소스가 이미 대상을 놓치며 찍은 영상(실제 free-moving 촬영)을 구조적으로 살린다

### 왜 이건 되고 F10(가림 게이트)은 안 됐나 — **단조성**

`fix.log F10` 이 실패한 이유는 *"가림은 knob 크기에 단조가 아니다 — 궤적을 줄여도 subject 를
가리는 건 그대로다"* 였다 (실측: 해소 24건 vs 정지로 파괴 48건, 파괴가 2~4.6배).
**framing 은 단조다.** 이번 세션 실측 — (씬, anchor, preset) 그룹별로 `knob` 오름차순 정렬 후
`subject_in_frame` 의 차분 부호를 검사:

| | 검사 그룹 | 단조 비증가 | 단조 비감소 | 비단조 | **prefix-feasible** (knob=0 포함 구간) |
|---|---|---|---|---|---|
| **d121 vista** | 3,944 | **93.9%** | 3.2% | 2.9% | **97.3%** |
| **d122 dynpose** | 2,727 | **95.5%** | 3.1% | 1.4% | **97.8%** |
| ↳ `aim=free` | 2,740 / 1,133 | 92.1% / 92.0% | 4.1 / 7.0% | 3.8 / 1.1% | 96.6% / 95.4% |
| ↳ `aim=look_at` | 1,204 / 413 | 98.0% / 96.9% | 1.2 / 0.7% | 0.7 / 2.4% | 99.0% / 99.0% |
| ↳ `dd_*` (`aim=traj`) | 1,181 | **98.5%** | 0.3% | 1.3% | **99.7%** |

사다리가 3~4점이라 거친 검정이다. 하지만 **이분법에 필요한 성질은 단조성 자체가 아니라
"`knob=0` 을 포함하는 실행가능 구간"** 이고 그게 **97%+** 다. 즉 이분법이 성립한다.
`dd_*` 가 가장 깨끗한 것도 중요하다 — DataDoP 변이가 framing 실패 모집단의 절반인데
(d122 43.7% 가 `<0.85`) 거기서 예산이 가장 잘 먹는다.

### 부등식(박스)이 등식(toric)보다 나은 이유가 여기서 드러난다
등식 제약은 해가 **manifold** 라 1D 이분법이 성립하지 않는다 (그래서 toric 은 별도 경로계획기가
필요하다). 부등식으로 두면 예산이 **1D 단조**가 되어 기존 이분법에 그대로 꽂힌다.
그리고 실제 촬영이 요구하는 것도 등식이 아니라 headroom / lead room 의 **구간**이다.

### 대가 (정직하게)
- 새 `framing_limited` status 가 생기고 궤적이 더 짧아진다. **hole 예산과 경쟁한다** —
  "구멍을 크게 만드는 큰 카메라"와 "대상을 놓치지 않는 작은 카메라"가 정면으로 부딪힌다
- 그 긴장을 **캡션이 거짓말하는 형태로 숨기는 지금보다, `binding` 열에 드러내는 게 옳다**
- 전량 재fit 필요 → `fix.log` 규약대로 F7 · P3 · P9 와 **한 판에** 묶는다

## 21. 제안 F 를 미룰 때의 3단 대안 (비용순, 서로 배타적이지 않다)

### F0 — 무료, 오늘: 캡션 게이트를 **`aim` 필드가 아니라 실측치**로
Part III P1(a) 에서 promote 판정을 `aim != look_at` 으로 하자고 했는데 **그건 대리변수다.**
실측: d122 의 `aim=look_at` 도 **16.7%** 가 `subject_in_frame < 0.85` 다 (d121 은 4.1%).
→ 판정을 `subject_in_frame < τ` 로 둔다. 임계 이상이면 target 절, 미만이면 targetless 문구.
**캡션 재빌드만.** 뱅크에 열이 이미 있고 재굽기가 필요 없다. 부수 효과로 캡션이
"이 궤적이 실제로 대상을 담았는가"를 반영하게 되어 **모델이 배우는 신호가 정직해진다.**

### F1 — 싸다: `look_at_bias` 를 **화면 좌표로 승격**
지금은 OBB 높이 비율로 조준점을 위/아래로만 민다 (스칼라). NDC 목표 `q=(qx,qy)` 로 두면
**rule-of-thirds / lead room** 이 표현되고, `aim=look_at` preset 의 framing 을 캡션이 요구하는
구도로 **못 박을** 수 있다. 회전만 바뀌므로 τ 는 불변이다 (D71 이 *"위치는 preset 모양 그대로라
τ 도 그대로 — 바뀌는 건 회전뿐"* 이라고 이미 적었다).
이게 **단일 타깃에서 toric 이 실제로 남기는 것**이다 — 1-target 이면 toric 은
"A 를 화면 q 에 두는 회전(닫힌 형태) + 원하는 크기가 정하는 거리 d" 로 퇴화하고,
그 회전 부분이 정확히 F1 이다. 참고: *Advanced Composition in Virtual Camera Control*
(Christie · Lino · Olivier, SG'11) 이 toric 의 전신이고 구도 제약을 최적화로 다룬다.
**단 `aim=free` 에는 못 쓴다** (회전을 건드리면 §19 로 되돌아간다) — 그게 F 가 필요한 이유다.

### F2 — VERTIGO 방식: framing 을 **학습 신호**로 (게이트 무수정)
뱅크에 `subject_in_frame` 이 변이마다 있으므로 **preference pair 를 공짜로 만들 수 있다**
(같은 (씬, anchor, 캡션) 안에서 sif 높은 변이 ≻ 낮은 변이). VERTIGO 가 off-screen
**38% → ~0%** 를 정확히 이 방식(렌더 프리뷰 → VLM 채점 → DPO)으로 냈다.
우리는 VLM 채점 단계조차 불필요하다 — **실측 sif 가 이미 있다.**
F 와 배타적이지 않다: **F 는 데이터를 고치고 F2 는 모델을 고친다.** 그리고 F2 는 재굽기가
필요 없어서 **지금 도는 arm 들 뒤에 바로 붙일 수 있는 유일한 framing 개선**이다.

## 22. toric 이 그래도 값어치를 갖는 **유일한** 자리 — `composition` 절

D121 이 `in_frame_ids` / `enter_ids` / `exit_ids` 를 뱅크에 실었고 캡션이
*"with X also in frame"* / *"as Y leaves the frame"* 를 약속한다.
**그건 진짜 2-타깃 framing 약속**이고, 거기서만 toric 의 α(두 대상의 화면 간격)가 실체를 갖는다.
대상이 움직이므로 원본 toric 이 아니라 **Drone Toric Space (TOG 2018)** 정식화를 봐야 한다.
다만 소수 모집단이므로 **중기 2(복합 카메라) 이후**로 미루는 게 맞다 —
지금 순서는 F0 → F2 → F(재굽기 한 판) → (그 다음에) 2-타깃.

## 23. Part II·III 정정

- **Part III P1(b) "Toric space 도입" → 철회.** §18 의 4가지 이유. 대체는 §20 제안 F
- **Part III P1(a) "promote 판정을 `aim != look_at`" → §21 F0 으로 교체** (실측 `subject_in_frame`
  기준. `aim` 은 대리변수이고 `aim=look_at` 도 d122 에서 16.7% 가 `<0.85`)
- **17절 #29 "Toric space 도입" → "framing 예산 #8 (제안 F)"** 로 교체. 비용은 "크다"가 아니라
  **"중간"** — 새 기하가 없고 기존 이분법에 예산 하나를 더하는 것이다
- 조준 쌍둥이(`truck_left_look_at` 등) 추가는 **하지 말 것** — §19

## 24. 우선순위 갱신 (framing 관련만)

| # | 항목 | 비용 | 재굽기 | 근거 |
|---|---|---|---|---|
| **20'** | **F0** 캡션 target 절을 실측 `subject_in_frame` 으로 게이트 | 캡션 재빌드 | 불필요 | 코퍼스 26~31% 불일치. `aim` 기준보다 정확 (look_at 도 16.7% 실패) |
| **20''** | **F2** sif preference pair → DPO/랭킹 | 중간 | 불필요 | VERTIGO 38%→~0%. **지금 arm 뒤에 바로 붙는 유일한 framing 개선** |
| **29'** | **F** framing 을 예산 #8 로 (부등식 + 소스 대비 임계) | 중간 | **필요** | 단조성 실측 확인 (prefix-feasible 97.3 / 97.8%). F10 이 실패한 이유가 여기엔 없다 |
| 31 | **F1** `look_at_bias` → NDC 화면 좌표 | 작다 | **필요** | rule-of-thirds/lead room. τ 불변. `aim=look_at` 전용 |
| 32 | 2-타깃 toric (`composition` 절) | 크다 | **필요** | 중기 2 이후 |

---

# Part V — 첫 카메라 고정이 만드는 꺾임 + 시작 구도 샘플링 (2026-09-05 추가)

## 25. 꺾임의 기제 — 코드가 스스로 적어 놓았다

`decode/build_poses.py:754-757` 주석:

> keyframe 0 이 소스 회전이라 `aim_anchor` 없이도 frame 0 이 소스와 완전히 같다 — 대신
> **"언제 target 을 보게 되나"가 keyframe 간격으로 정해진다** (F=49, N=5 면 12프레임 안에 넘어간다).

d121/d122 는 `--aim_keyframes 6` → keyframe = `rint(linspace(0,48,6))` = **[0, 10, 19, 29, 38, 48]**.
- keyframe 0 = **소스 frame0 회전** (계약상 고정)
- keyframe 1~5 = anchor 조준

즉 **소스→조준 회전 보정 전량이 프레임 0~10 구간에 압축**된다. 그 뒤 4구간은 거의 같은 회전이라
각속도가 프레임 10에서 급감한다 → 눈에 보이는 꺾임. 보정량 실측(README §aim_anchor):
camel **3.18°** / avocado-slice **11.67°** — 후자는 10프레임에 1.17°/frame 이고 그 뒤 ~0 이다.

세 가지가 악화시킨다:

1. **keyframe 간격이 균일하다.** `keyframe_indices` 가 `linspace` 라 "0 은 소스, 1 은 프레임 20"
   같은 비균일 배치가 표현되지 않는다. 전환에 쓸 프레임 수가 **`48/(N-1)` 로 못 박힌다**
2. **`smooth_kf` 가 양 끝을 고정한다** (D89 주석: *"양 끝은 고정한다 — frame0 은 어차피 소스
   회전으로 덮어써지고, 마지막은 preset 끝점이다"*). 즉 SO(3) Laplacian 평활이
   **꺾이는 바로 그 지점을 못 건드린다.** 평활 대가는 `aim_err_*_deg` 로 나간다
3. **`aim="free"` 는 이 블록을 통째로 건너뛴다** (`:747-748`). 그래서 이 결함은 `aim=look_at`
   전용 — d121 에서 **16,034변이 (53.5%)**

## 26. 🔴 그리고 이 결함은 코퍼스에서 **보이지 않는다**

뱅크 열 **71개를 전수 확인**했다. `jerk` / `aim_err` / `roll` / `vel` / `accel` / `smooth` 로
매칭되는 열이 **하나도 없다.** 매끄러움을 재는 열이 0개다.

- `verify.py:97 jerk_series` 는 있지만 LBM 단일 궤적 경로(`out/<video>/poses.npz`) 전용이고
  뱅크 변이에는 안 걸린다 (Part II M4 와 **같은 구조적 분리**)
- `smooth_kf` 가 내부에서 계산하는 `aim_err_*_deg` 도 뱅크에 안 실린다
- **AdaViewPlanner 의 1급 지표가 `Jerk_t` / `Jerk_r`** 다 — 문헌 표준 축이 우리 코퍼스에 없다

→ **개선 1: 뱅크에 `jerk_t` / `jerk_r` / `aim_err_max_deg` / `aim_err_med_deg` 열 추가.**
전부 numpy, **렌더 0회**. 이게 없으면 "꺾임이 고쳐졌는가"를 판정할 수 없고, 지금 사용자가 눈으로
보고 있는 것을 숫자로 확인할 방법도 없다.

## 27. 🔴 왜 시작 구도를 못 바꿨나 — **`max_tau 0.30` 이라는 죽은 상수**

`lbm/candidates.py` docstring 실측:

> 고정 격자(az {0,±45,±90,±135,180} × el {12,28,45} × r {0.7,1.0,1.4}·d_ref)를 camel 에서
> 그대로 돌렸더니 **72개 중 70개가 G4_tau 에서 죽었다**. `max_tau 0.30`, `z_med 3.553`,
> `S 4.6713` 이면 소스 카메라에서 움직여도 되는 거리가 `0.228 u` 인데 `d_ref` 가 `0.64 u` 라
> 반경만 1.4배로 늘려도(0.256 u) 이미 예산 초과다. 방위각은 ±20° 가 한계고
> **"정면 45도"는 도달 불가능하다.**

그래서 `pool_mode="budget"` 이 소스 구면좌표 **주변만** 격자를 편다 = **시작 구도가 소스에 묶인다.**

**그런데 `max_tau 0.30` 은 뱅크가 이미 버린 전제다.** `sample_camera_bank.py` docstring:

> 기존 파이프라인은 `max_tau 0.30` / `min_coverage 0.55` 를 **게이트**로 썼다. 그건 "렌더가 GT 에
> 가까워야 한다"는 전제였고, **학습 pair 생성에는 그 전제가 없다** — 하류 video model 이 hole 을
> 채우는 게 일이다.

**즉 절대 구도 격자가 죽는 이유는, 뱅크가 이미 믿지 않기로 한 상수 하나다.**
τ 게이트를 hole 예산으로 갈아끼우면 "정면 45도"는 **기각 대상이 아니라 가격이 붙은 선택지**가 된다.
그리고 그 가격은 **이미 측정되고 있다** — `hole_static`(손잡이 0에서의 hole).
실측 d121 median 0.079 / p90 0.391, d122 0.122 / 0.326.

## 28. ✅ 제안 S — 시작 구도 샘플러 (기계는 이미 다 있다)

**세 축이 곧 구도 어휘다** (`lbm/candidates.py:31-33`):

| 축 | 코드 기본값 | 구도 어휘 | 문헌 대응 |
|---|---|---|---|
| `azimuth_deg` (subject OBB yaw 기준 0°) | `(0, ±45, ±90, ±135, 180)` | front / three-quarter / profile / back | AdaViewPlanner **Shot Diversity** (캐릭터 기준 시야각 360°) |
| `elevation_deg` | `(12, 28, 45)` | eye-level / high / bird | **CSD** 의 viewpoint type |
| `distance_ratio × d_ref` | `(0.7, 1.0, 1.4)` | close / medium / wide | LensCraft **shot type ↔ ABox/VBox** |

**이미 있는 것:**
- `build_pool()` — 절대 구면 격자, `az_margin_deg` 로 `obs_az_span` 필터
- `source_spherical()` — 소스 카메라의 (az, el, r). **"소스와 얼마나 다른 구도인가"를 바로 낸다**
- `build_pool_budget()` — 소스 중심 예산 격자
- `lbm/gates.py` G1/G4(렌더 0회) + G2/G3(렌더) — 후보 게이팅 + `board/gates.csv` 사유 기록
- `--start_mode board` — **배선이 있고 기본값에서 꺼져 있다**

**바꿀 것 4개:**

1. **G4_tau 게이트를 `hole_static` 가격으로 교체.** τ 로 후보를 **제거**하지 말고 `hole_static` 을
   열로 붙여 사다리처럼 **단**으로 쓴다. §27 이 근거다
2. **VLM 랭킹을 열거로 교체.** `--start_mode board` 는 원래 VLM 이 board 에서 하나를 골랐다.
   그런데 select/traj 턴은 prior 라는 실측(D34~36)이 있으므로 **고르지 말고 전부 낸다** —
   새 격자는 `anchor × 구도(az,el,r) × preset × hole단`. 샘플러 철학(열거 + 실측)과 일관된다
3. **구도를 캡션 축으로 승격.** `azimuth` 버킷 = "from the front / in profile / from behind",
   `elevation` = "low angle / high angle", `distance_ratio` = shot scale.
   → **shot scale 이 결과가 아니라 조건이 된다** (Part III P7-c 가 요구한 것), 그리고
   **CSD / Shot Diversity 를 낼 수 있게 된다** (Part II M9 해소 — 지금은 preset 13종 엔트로피만 나온다)
4. **`--free_start` 는 depth-warp 에서 no-op 이 아니다.** `decode/emit.py:10-13` 실측:
   > `--free_start` 는 `inv(src_c2w[0]) @ P` 로 상수 offset 을 살려두는데, **ReCamMaster 는 그
   > offset 을 픽셀 수준에서 무시한다**(실측). 즉 `--free_start` 는 **depth-warp 계열에만 의미가
   > 있고** prior-locked 계열엔 no-op 이다.

   → 하류가 **Vista4D depth-warp** 이므로 **계약은 이미 열려 있다.** `goals.md` 중기 3 이 걱정한
   "`rel[0]=I` 를 가정하는 하류 모델과의 계약이 깨진다"는 **ReCamMaster 쪽 얘기**이고,
   그쪽은 애초에 offset 을 무시하므로 잃는 것도 없다

**꺾임이 부수적으로 사라진다**: 시작 pose 를 구도에 맞춰 뽑으면 keyframe 0 과 keyframe 1 의
회전이 같아져 **조준 보정이 0** 이 된다. 평활로 깎는 게 아니라 **원인이 없어진다.**

## 29. `rel[0]=I` 를 유지한 채로 구도를 얻는 길 — transitional segment

`--free_start` 를 쓰기 싫다면(frame0 warp 를 완벽히 유지) 전환을 **궤적의 일부로 만든다**:
frame 0~K 는 소스 pose → 목표 구도로 가는 연속 이동, K~48 은 preset.

- `rel[0]=I` 가 유지되고 구도 다양성도 얻는다. 대가는 preset 에 쓸 프레임 K개
- 문헌 근거가 정확히 있다: **CinemaTraj 의 transitional `arc`** — 두 anchor 를 곡률
  α ∈ [−89°, 89°] 로 잇는 primitive 이고, atomic movement 7종과 **별도 범주**로 두었다
- 우리 쪽 자리: `lbm/presets.py` 에 `T.arc` 가 이미 있고 `s_curve` 가 `concatenate` 를 쓴다 —
  **시간축 이어붙임 기계가 이미 있다** (README ④)
- 부수 효과: 이게 goals 중기 2(복합 카메라)의 **가장 자연스러운 첫 조합**이다. 캡션도 기계적으로
  합성된다 — *"swings around behind the subject, then orbits left"*

## 30. 꺾임 자체를 고치는 3개 (구도 샘플링과 독립)

| # | 무엇 | 비용 |
|---|---|---|
| **K1** | **비균일 keyframe 배치** `--aim_kf_frames 0 20 32 42 48` — 전환에 20프레임을 준다. 지금은 `linspace` 라 `48/(N-1)` 로 못 박힌다 | 작다 |
| **K2** | **`smooth_kf` 의 양끝 고정을 frame0 만으로.** frame0 은 계약이라 고정이 맞지만 **마지막 프레임은 preset 끝점일 뿐**이다 — 지금은 양쪽을 다 묶어 평활 폭을 스스로 좁힌다 | 작다 |
| **K3** | **C1 보간으로 교체.** slerp 折れ線은 keyframe 에서 각속도가 불연속(C0)이다. squad / SO(3) cubic spline 이면 C1. `arclen_kf` 의 PCHIP 재타이밍은 각속도 정체만 고치고 **회전축 꺾임은 남긴다** (D89 주석 명시) | 중간 |

**K1 이 가장 값싸고 직접적이다.** 단 K1~K3 는 전부 "꺾임을 부드럽게" 하는 것이고,
**제안 S 는 원인을 없앤다.** 배타적이지 않다 — S 를 켜도 `tracking=lock`(D 최근 변경,
`TRACKING_GAIN = {"world":0.0, "lock":1.0}`) 의 subject track 잔여 고주파는 남으므로 K3 는
여전히 값어치가 있다.

## 31. 참조 추가

| 논문 | 빌릴 것 |
|---|---|
| **ViewCrafter** (2409.02048) | 점군 렌더의 **가림을 줄이는 NBV 궤적 계획** — hole 을 NVS 쪽에서 접근한 것. "궤적을 정하고 hole 을 재는" 우리와 **반대 방향**("hole 이 작은 궤적을 찾는다"). 반복적으로 3D 단서를 확장하는 구조라 **장기 1(world 확장)의 직접 선례**다 |
| **AdaViewPlanner** (2510.10670) | (az, el, r) 3축이 곧 **Shot Diversity / CSD** 의 정의. 제안 S 의 축 = 그 지표축 |
| **CinemaTraj** (2607.26910) | **transitional arc** — 두 anchor 를 잇는 별도 primitive 범주 (§29) |
| **LensCraft** (2506.00988) | shot type ↔ ABox/VBox 보간 = `distance_ratio` 를 구도 어휘로 못 박는 방법 |

## 32. 우선순위 (24절에 이어)

| # | 항목 | 비용 | 재굽기 | 근거 |
|---|---|---|---|---|
| **33** | 뱅크에 `jerk_t` / `jerk_r` / `aim_err_*` 열 | 작다 | 열만 | 71열 중 매끄러움 열 **0개** — 꺾임이 코퍼스에서 안 보인다 |
| **34** | **K1** 비균일 keyframe 배치 | 작다 | **필요** | 전환이 `48/(N-1)` 프레임에 압축된다 |
| **35** | **제안 S** 구도 샘플러 (`--start_mode board` + hole 가격 + 열거 + `--free_start`) | 크다 | **필요** | 꺾임 원인 제거 + 중기 3 + CSD/Shot Diversity 지표 개방 |
| **36** | **§29** transitional segment | 중간 | **필요** | `rel[0]=I` 유지하면서 구도를 얻는 길. 중기 2 의 첫 조합 |
| 37 | **K2 / K3** 보간 개선 | 작다 / 중간 | **필요** | S 를 켜도 track 잔여 고주파는 남는다 |

---

# Part VI — canonicalization 설계 + 첫 카메라를 줄지 (2026-09-05 추가)

이 절은 **vendored 코드 직독**으로 확정한 사실 위에 세운다. 추측 아님.

## 33. 세 규약을 코드로 확정

### GenDoP (`models/GenDoP/core/provider.py:165-171`)
```python
ref_w2c = torch.inverse(matrix_to_square(c2ws[:1]))
c2ws = ref_w2c @ matrix_to_square(c2ws)          # ← 첫 카메라 상대: c2w[0] = I (R,t 모두)
T_norm = c2ws[:, :3, 3].norm(dim=-1).max()
c2ws[:, :3, 3] /= (T_norm + 1e-5)                # ← 최대 |t| = 1 로 정규화
coords_scale = (log10(scale) + 2)/4 * bins       # ← 스케일은 **별도 토큰**
```
→ **GenDoP 은 첫 카메라를 표현에서 아예 지운다.** `c2w[0]=I` 가 강제이고, 생성되는 것은
**모양 + 스케일 토큰**뿐이다.

### 우리 `decode/emit.py:10-21`
```
rel = inv(P[0]) @ P ;  rmax = max_f |rel[f,:3,3]| ;  rel /= rmax ;  g = rmax / S (별도 손잡이)
```
→ **GenDoP 규약과 문자 단위로 같다.** 즉 `canonical.json` 은 이미 GenDoP 호환이다.

### E.T. / CLaTr 표현 (`main/evaluate/CLaTr/.../trajectory_dataset.py:81-100`)
```python
raw_trans = matrix[:, :3, 3]
if velocity:  raw_trans = cat([raw_trans[0][None], raw_trans[1:] - raw_trans[:-1]])
rot6d = raw_rot[:, :, :2] ...
feature = hstack([rot6d, raw_trans]).T           # 9 × num_cams
```
→ **frame 0 은 절대 translation, frame 1.. 은 속도.** frame0 채널의 통계가
`standardization/0120.yaml`:
```
shift_mean = [0.00201, -0.27489, -1.23617]   shift_std = [1.1343, 1.1906, 1.5874]
norm_mean  ≈ 1e-4                            norm_std  = [0.0278, 0.0182, 0.0314]
velocity: true    num_cams: 120
```
`|shift_mean| = 1.268` → 사용자가 기억한 *"rotation identity, translation 은 character 기준
특정 방향으로 1 정도"* 가 **정확히 이 값**이다. 그리고 frame0 채널 std 가 속도 채널 std 의
**약 50배**다.

## 34. 🔴 결정적 — **이 레포가 돌리는 CLaTr 은 첫 카메라를 볼 수 없다**

`TrajectoryEvalDataset.get_traj()` (`:337-341`, ref/pred 양쪽에 적용):
```python
ref_w2c = torch.inverse(c2ws[:1])
c2ws = ref_w2c @ c2ws            # ← 로드 시점에 무조건 첫 카메라 상대로 재앵커
```
그리고 `self.standardize = False` 가 **하드코딩**되어 있고(`:215`, `:41`) 표준화 코드는 전부
주석 처리됐다. 따라서:

1. **`R_0 = I`, `t_0 = 0` 이 모든 샘플에서 강제된다** — 무엇을 dump 하든 상관없다.
   `shift_*` 통계는 이 경로에서 **죽은 코드**다.
2. → **시작 pose 고정을 풀어도 CLaTr 수치는 안 바뀐다. 반대로 지금 고정 때문에 CLaTr 이
   나빠지는 것도 아니다.** CLaTr 은 시작 pose 를 **구조적으로 못 본다.**
   사용자 우려("첫 카메라 고정 → 꺾임 → CLaTr 문제")는 **경로가 다르다** — CLaTr 에 들어가는
   건 꺾임(각속도 불연속)이고 시작 pose 자체가 아니다. 꺾임은 rot6d 속도열에 남으니 **영향이
   있고**, 시작 구도는 **영향이 0** 이다.
3. 이 파일의 헬퍼 이름(`matrix_to_square`, `check_valid_rotation_matrix`)이 GenDoP
   `provider.py` 와 동일하다 → 우리가 돌리는 CLaTr 은 **GenDoP 포크(첫-카메라 상대 규약으로
   재학습된 판본)** 이고 E.T. 원본 규약이 아니다. Part II C1 의 "다른 코퍼스" 경고가 이것이다.

### 그리고 두 가지 스케일 사고가 더 있다 (Part II C1 의 진짜 기제)
- **CLaTr eval 로더는 스케일 정규화를 안 한다.** GenDoP provider 는 `T_norm` 으로 나누는데
  eval `get_traj` 는 `ref_w2c` 만 곱하고 **아무것도 안 나눈다**. 즉 속도 채널이 **절대 world
  단위**를 그대로 들고 간다. 우리 world 는 DA3 단위(camel S=4.67)다.
  → **FCD 가 서로 다른 단위의 속도 분포를 비교한다.** `snowboard n=77 fcd 397` vs
  `dl3dv n=3980 fcd 26.5` 의 유력한 주범이 표본 수가 아니라 **이것**이다.
  처방: dump 전에 ref/pred 를 **같은 인자**로 정규화한다 (`emit.py` 의 `rmax` 를 그대로 쓰면
  GenDoP 규약과도 일치).
- **49 vs 120 프레임.** `__getitem__` 은 `F.pad(..., num_cams - T)` + mask 로 **120 에 zero-pad**
  하고 리샘플하지 않는다. 그런데 **GenDoP eval 은 정확히 120 으로 리샘플한다**
  (`eval.py:260 for i in range(120)`). 같은 물리적 운동이라도 49프레임 속도가 120프레임 속도의
  **약 2.4배**다. `norm_std ≈ 0.028` 은 120프레임 E.T. 통계다.
  → 헤드투헤드 하려면 **양쪽을 같은 프레임 수로 리샘플**해야 한다.

## 35. ✅ canonicalization 답 — context 첫 카메라가 맞다 (그리고 유일하다)

| 규약 | 원점 | tracking | free-moving | 추론 시 알 수 있나 |
|---|---|---|---|---|
| E.T. | **character** (translation) + 첫 카메라(rotation) | ○ | **✗ (character 가 없다)** | subject 를 골라야 한다 |
| GenDoP | **target 첫 카메라** | △ (subject 관계가 사라짐) | ○ | **✗ (생성 대상이다)** |
| **제안(사용자)** | **context 첫 카메라** | ○ | ○ | **○ (소스 영상이 항상 있다)** |

context 첫 카메라가 유일한 이유 세 가지:
1. **항상 정의된다** — subject 유무와 무관. tracking/free-moving 을 한 표현에 담는 조건이 이것뿐이다
2. **추론 시 관측 가능하다** — target 첫 카메라는 생성 대상이라 원점으로 쓸 수 없다
   (GenDoP 은 그걸 쓰기 때문에 첫 카메라를 표현에서 지워야 했다)
3. **두 baseline 규약이 고정 변환으로 복원된다** —
   E.T. 규약 = subject 채널을 빼면 되고, GenDoP 규약 = `inv(target[0]) @ ·` 를 곱하면 된다.
   **비교 공정성이 표현 단계에서 보장된다.** 반대 방향(GenDoP→ours)은 정보가 없어 불가능하다

**subject 관계는 잃지 않는다** — 원점에서 빼는 대신 **조건 채널로 넣는다.**
world = DA3 frame0 = context 첫 카메라이므로 `scene_graph` 의 OBB track 이 **이미 그 좌표계**다.
그리고 그 채널 경로가 이미 있다 (`cond_dim` / `target_track_dim`, per-token concat).
실측으로 먹는 것도 확인됐다 (`FIX.log`: track 조건 `loss_traj 0.01519` vs `--drop-track` 0.02639).

→ **정리: 원점 = context 첫 카메라, subject = 명시 조건 채널, 스케일 = 별도 손잡이(avg_scale).**
이 셋이면 E.T.·GenDoP 둘 다의 상위집합이다.

## 36. ✅ target 첫 카메라를 줄지 — **둘 다, 하나의 모델로**

양쪽 다 근거가 있어서 고를 필요가 없다:

| | 준다 | 안 준다 |
|---|---|---|
| 과제 | "이 오프닝에서 샷을 완성하라" | "구도까지 결정하라" |
| 하류(depth-warp) | **필요하다** (frame0 을 알아야 warp 한다) | 프로덕션에 쓸 수 없다 |
| E.T. 비교 | 불가 (E.T. 는 생성한다) | **가능** |
| 위험 | 구도 선택을 안 배운다 | **데이터 평균으로 붕괴** (E.T. 실측: R≈I, \|t\|≈1.27) |

**처방: target 첫 카메라를 optional 조건 채널로 두고 학습 중 확률 p 로 drop 한다.**
- 기계가 이미 있다 — `cfg_dropout_p: 0.1` + `cond_dim` null 분기
  (`camera_diffusion_model_latent.py:276-281`: `cond is None → zeros`)
- 한 ckpt 가 두 프로토콜을 답한다: **주면** 프로덕션 경로, **안 주면** E.T.-비교 경로,
  **같은 가중치**
- 그리고 두 경로의 차이가 **측정값**이 된다 — "오프닝 구도를 아는 것이 얼마나 도움이 되나".
  `--drop-track` 로 이미 해 본 패턴 그대로다
- 붕괴 방어: 안 주는 모드에서 **CSD / Shot Diversity 엔트로피**를 지표로 본다 (Part V §28).
  E.T. 가 평균으로 붕괴한 것을 그 지표가 잡아낸다 — 지금은 그 지표가 아예 없다

## 37. ✅ GenDoP 비교 — "첫 카메라를 강제할지"는 **성립하지 않는 질문**

§33 이 확정한 것: **GenDoP 표현에는 첫 카메라 채널이 없다.** `c2w[0]=I` 가 강제이고
생성물은 모양 + 스케일 토큰이다. **강제할 대상이 애초에 없다.**

따라서:
- **CLaTr 축의 비교는 이미 공정하다** — eval 로더가 ref/pred 양쪽을 첫 카메라 상대로 재앵커하므로
  (§34) 우리 시작 pose 는 비교에 들어가지 않는다. **"모양(+스케일)" 비교**로 정확히 정의된다.
  단 위의 스케일·프레임수 두 사고는 **반드시 먼저 고쳐야** 한다 (같은 인자로 정규화 + 같은
  프레임 수로 리샘플). 안 고치면 공정한 건 규약뿐이고 숫자는 아니다
- **시작 구도 품질은 CLaTr 로 재면 안 된다.** 그 축은 Part II 의 L1/L2 지표다 —
  frame0 의 MisR/framing, `hole_static`, 시작 pose 의 collision, 그리고 구도 버킷 분포
  (Shot Diversity / CSD). GenDoP 은 이 축에 **참가하지 않는다** → 점수 차이가 아니라
  **능력 차이(capability)** 로 서술하는 것이 정직하다
- 우리 `canonical.json` 이 이미 GenDoP 규약이므로(§33) **GenDoP 에 우리 테스트셋을 먹이는 건
  표현 문제가 아니라 배관 문제**다. `scripts/gendop_release_infer.py` /
  `gendop_preds_to_eval_dir.py` 가 이미 있다

## 38. E.T. 와 비교하는 것이 맞나 — **metric 으로는 맞고 baseline 으로는 부분적으로만**

- **metric(CLaTr)**: 맞다. 그리고 §34 때문에 규약 차이가 상쇄된다(로더가 재앵커).
  단 Part II C1 대로 **우리 코퍼스에서 재학습 + retrieval R@k sanity** 를 해야 수치가 뜻을 갖는다
- **baseline(DIRECTOR)**: 입력이 다르다 — 텍스트 + **character 궤적**, 씬 기하 없음, 합성
  Blender, 그리고 LensCraft 실측으로 **70% 가 static shot**. end-to-end 비교는 의미가 약하다.
  정직한 배치는 입력 축을 명시하는 것이다:

  | | 입력 | subject | 씬 기하 |
  |---|---|---|---|
  | DIRECTOR (E.T.) | text | **주어짐** | 없음 |
  | GenDoP | text + first-frame RGBD | 없음 | 첫 프레임만 |
  | **ours** | text + 영상 전체 geometry | 주어짐(OBB track) | **49프레임 depth shell** |

- **E.T. 의 frame0 붕괴는 우리 주장의 근거로 쓸 수 있다**: 첫 카메라를 "그냥 생성"하게 두면
  prior 로 붕괴한다(R≈I, |t|≈1.27 = `shift_mean`). 그래서 우리는 **구도 축을 명시로 열거**한다
  (Part V 제안 S) — 이게 E.T. 대비 기여로 서술되는 자리다

## 39. 그래서 데이터셋을 어떻게 맞추나 (실행 순서)

1. **뱅크에 시작 pose 를 축으로 추가** — Part V 제안 S. `anchor × 구도(az,el,r) × preset × hole단`.
   `hole_static` 이 시작 구도의 가격이고 이미 측정된다
2. **저장은 context 첫 카메라 좌표계 그대로** — `target_poses.npz` 가 이미 world(=DA3 frame0
   =context 첫 카메라) w2c 다. **파일 포맷을 바꿀 필요가 없다.** 바뀌는 것은
   `emit.py --free_start` 를 켜서 `rel[0]=I` 강제를 푸는 것뿐이고, depth-warp 하류에는 유효하다
   (Part V §28-4)
3. **조건 채널 3개를 명시로**: ① subject OBB track (context 좌표계) ② target 첫 카메라
   (optional, dropout) ③ 구도 라벨(az/el/shot-scale 버킷) — ③은 캡션으로도 들어간다
4. **eval dump 를 고친다** — ref/pred 를 같은 인자로 정규화 + 같은 프레임 수로 리샘플.
   §34 의 두 사고. 이건 **재굽기 없이 오늘 가능**하고, 고치기 전 FCD 수치는 전부 버려야 한다
5. **CLaTr 을 우리 코퍼스에서 재학습** + held-out 씬 retrieval R@k (Part II C1)

## 40. 우선순위 (32절에 이어)

| # | 항목 | 비용 | 재굽기 | 근거 |
|---|---|---|---|---|
| **38** | eval dump 스케일 정규화 + 프레임수 통일 | 작다 | **불필요** | CLaTr 로더가 스케일 정규화를 안 한다. 지금 FCD 는 단위가 섞인 값 (§34) |
| **39** | canonicalization = context 첫 카메라로 확정 + 문서화 | 문서 | 불필요 | 두 baseline 규약이 고정 변환으로 복원되는 유일한 원점 (§35) |
| **40** | target 첫 카메라를 optional 조건 + dropout | 중간 | 1 arm | 한 ckpt 로 두 프로토콜. 기계(`cond_dim`+`cfg_dropout_p`)가 이미 있다 (§36) |
| **41** | GenDoP 비교를 "모양+스케일" 축으로 명시 서술 | 문서 | 불필요 | GenDoP 에 첫 카메라 채널이 없다 — 강제 질문 자체가 성립 안 함 (§37) |
| **42** | 붕괴 감시 지표(CSD / Shot Diversity) | 작다 | 불필요 | 첫 카메라를 생성하게 두면 E.T. 처럼 평균으로 붕괴한다 (§36) |

---

# Part VII — VAE 규약과 모델 학습 (2026-09-05 추가)

## 41. 🔴 시작 pose 는 **뱅크가 아니라 dataset 경계에서** 지워진다

`main/dataset_dl3dv.py:13` — VAE/DM 입력 규약:
```
cam_param : (T, 11) = rot6d(6) + trans(3) + [fx/w, fy/h](2)
cam_param = E @ inv(E_s)   ← segment 첫 카메라 s 기준 재고정, 그 뒤 trans /= norm_scale
```
즉 **`rel[0]=I` 이 표현에 박혀 있다.** 뱅크에서 `--free_start` 로 시작 pose 를 풀어도
`E @ inv(E_s)` 를 그대로 쓰면 **토큰 0 이 다시 상수가 되어 시작 pose 가 버려진다** —
CLaTr 이 재앵커하는 것(§34)과 정확히 같은 손실이 학습 쪽에서도 일어난다.

**설정 함정 하나 더**: `anchor_pred_frame0` = *"target 첫 카메라를 given 으로 (rel[0]=I 강제)"*
이고 d121/d122/d123/d124 전부 **`true`** 다. 켜져 있으면 디코드가 예측의 frame0 을 항등으로
되돌린다 → 생성한 시작 pose 가 그 자리에서 사라진다. **자유 시작을 쓰면 반드시 `false`.**

## 42. ✅ VAE — **바꾸지 말고, 시작 pose 를 latent 밖으로 뺀다**

현재 VAE (`vae_20260302_300.pth`, cam_dim 64) 실측 (`docs/vae_verification.md`):
- rotation geodesic **1.33°** (median 1.0°), translation 오차 **0.035** → 양호
- intrinsics **깨졌다** (recon 상수 0.99, corr ≈ −0.04). 단 **우리에게 무해**하다 —
  `--fixed_focal` + `intr_norm: rel` 이라 `cam_param[9:11]` 이 사실상 상수(1.000±0.003)다
- latent std 보정 확인 (scaled std 1.015 @ `vae_latent_scale 0.96032625`)

**context 첫 카메라를 VAE 앵커로 바꾸면 안 된다.** 그러면 49 translation 전체에 시작 pose
오프셋이 상수로 더해져 (a) 64-d latent 이 "모양 + 전역 오프셋"을 같이 지고 (b) 오프셋 오차가
모양 오차에 섞이고 (c) `vae_latent_scale`·`loss_traj`·FCD 히스토리가 전부 무효가 된다.

**GenDoP 이 같은 문제를 이미 이렇게 풀었다** — `scale` 을 pose 토큰 안에 넣지 않고
**별도 토큰**으로 뺐다 (`provider.py:181`). 전역 DOF 는 factor out 하는 것이 정석이다.

→ **규약: VAE = 모양 전용, target frame0 앵커 그대로.**
   시작 pose = **별도 9-d 벡터** `s₀ = (rot6d(R₀), t₀/avg_scale)`, **context frame0 좌표계**.
   디코드: `E_target(f) = s₀ ∘ decode(latent)(f)`.

**VAE 쪽에서 정말 해야 할 것 2개 (둘 다 재학습 아님):**
1. **`vae_latent_scale` 을 코퍼스마다 재측정.** 이 값은 "(ckpt, scale_mode, intr_norm) 삼중의
   실측 latent std" 여야 하는데 지금은 **전역 상수 하나**다. 실측 격차: DL3DV 0.47637 /
   TRUMANS **0.19128** vs config 0.96033 → TRUMANS 는 diffusion 입력 std 가 **1/5** 다
   (`summary.md` §7 원인 ②: 프레임당 속도가 12~31배 큰 jitter). 새 코퍼스(자유 시작 뱅크,
   d122)에서 반드시 다시 잰다. 도구: `scripts/vae/vae_scale_matrix.py`
2. **새 규약 뱅크에서 VAE round-trip 먼저 확인.** `scripts/vae/vae_roundtrip_val.py`.
   geodesic ≈1.3° / trans ≈0.035 가 유지되면 **VAE 는 손댈 것이 없다.** 자유 시작은 *모양* 을
   더 매끄럽게 만들 뿐(조준 보정 0, Part V §28) 모양 manifold 를 벗어나지 않는다.
   재학습은 이 확인이 실패할 때만 고려한다

## 43. ✅ 모델 학습 — 시작 pose 는 `cond_dim` 채널 + dropout

기계가 이미 다 있다:
- `cond_dim` per-token concat + `cond is None → zeros` null 분기
  (`camera_diffusion_model_latent.py:276-281`)
- `cfg_dropout_p: 0.1`
- `--drop-track` 스타일 eval 플래그 (`FIX.log` 실측 패턴)

배선:
```
x_t      = VAE latent (모양, 변경 없음)
cond[t]  = [ s₀ (9) | subject OBB track (4: x,y,z,valid) ]   ← 둘 다 context frame0 좌표계
           s₀ 는 확률 p 로 drop (null = zeros)
decode   = s₀ ∘ VAE.decode(x_0) × avg_scale
```
- **한 ckpt 가 두 프로토콜을 답한다** — 주면 프로덕션(depth-warp), 안 주면 E.T.-비교
- 두 경로 차이가 곧 "오프닝 구도를 아는 것의 값어치" 측정값
- `anchor_pred_frame0: false` 필수 (§41)
- 붕괴 감시: 안 주는 모드에서 **CSD / Shot Diversity** 엔트로피 (§36)

## 44. 데이터 — 포맷은 안 바뀐다

`target_poses.npz` 는 이미 **world(=DA3 frame0 = context 첫 카메라) w2c** 다. 여기서
모양(`inv(P[0])@P`)과 시작 pose(`P[0]`) 가 **둘 다 유도된다.** → 파일 규약 변경 0.

필요한 것:
1. 제안 S 뱅크 (`anchor × 구도(az,el,r) × preset × hole단`)
2. `--free_start` 로 emit (depth-warp 하류에는 유효, Part V §28-4)
3. `avg_scale` + `vae_latent_scale` **재측정**
4. 구도 라벨(az/el/shot-scale 버킷) → 캡션 + `cond`

## 45. 실행 순서

| 순 | 무엇 | 왜 이 순서 |
|---|---|---|
| 1 | 새 규약 뱅크 소량(1~2씬)으로 **VAE round-trip 확인** | VAE 재학습 여부가 여기서 갈린다. 제일 싸다 |
| 2 | **`vae_latent_scale` 재측정** (코퍼스별) | 이 값이 틀리면 아래 전부가 jitter 위에서 학습된다 |
| 3 | eval dump 스케일/프레임수 통일 (§34) | 고치기 전 FCD 는 버려야 하므로 arm 을 돌리기 전에 |
| 4 | 제안 S 뱅크 전량 + `--free_start` emit | 재굽기 한 판 (F7·P3·P9·framing 예산과 함께) |
| 5 | DM 학습: `cond = [s₀ | track]` + dropout, `anchor_pred_frame0: false` | 위가 다 끝난 뒤 |

---

## 46. 🔴 §43 정정 — s₀ 는 **생성 대상**이고, 그러면 `cond_dim` 이 아니다

§43 에서 s₀ 를 `cond_dim` 채널로 적었는데 **틀렸다.** `cond_dim` 은 입력 전용 concat 이고
모델 출력은 `self.out(h)` → latent 토큰뿐이다 (`camera_diffusion_model_latent.py:204`).
s₀ 를 drop 하면 **정보만 없어지고 생성되지 않는다** → s₀=identity 로 되돌아가 지금의 고정 시작과
같아진다. "안 주는" 프로토콜을 원하면 **s₀ 는 diffusion 시퀀스 안**에 있어야 한다.

### 배선
VAE 에 `stride=2` Conv1d 가 둘이라 시간축이 **4× 압축**된다 (`models/vae_intr.py:15,31`):
49 → 25 → **13 토큰** × 64-d. 여기에 s₀ 토큰 하나를 붙여 14 로 만든다.
```
x_t  = [ s₀_token | latent(13) ]     s₀ = (rot6d(R₀), t₀/avg_scale), **별도 표준화**
cond = [ subject OBB track ]         ← 이쪽은 진짜 조건 (생성 대상 아님)
```
- **"준다" = 샘플링 중 그 토큰을 매 denoise step 에 clamp** (inpainting 방식).
  학습 1회, 프로토콜은 추론 시 결정 → 별도 arm 불필요
- **s₀ 자기 표준화 필수.** CLaTr 의 `shift_std 1.4` vs `norm_std 0.028`(50× 불일치, §33)이
  그대로 재현될 자리다. unit std 로 맞춰야 노이즈 스케줄이 두 종류 토큰에 같이 먹는다
- **deterministic head + L2 는 금지** — 조건 평균으로 붕괴하고 그게 E.T. 의 실패 모양이다

### 진짜 위험은 head 가 아니라 identifiability
E.T. 가 R≈I, |t|≈1.27(=`shift_mean`)로 붕괴한 건 regression head 때문이 아니다 —
**DIRECTOR 도 diffusion 이다.** 원인은 **텍스트만으로 시작 pose 가 결정되지 않는 것**이다.
조건이 s₀ 를 결정하지 못하면 diffusion 은 prior 를 샘플링하고 그게 데이터 평균이다.

우리가 E.T. 대비 갖고 있는 것 셋: ① DA3 geo(어디에 있을 *수* 있나) ② subject OBB track
(무엇을 프레이밍하나) ③ **캡션의 구도 절**(az/el/shot-scale 버킷, 제안 S #3).

**③ 이 결정적이다.** 구도를 캡션에 넣지 않으면 s₀ 생성은 E.T. 와 똑같이 붕괴한다.
→ **제안 S 의 캡션 승격은 선택이 아니라 s₀ 생성의 전제조건이다.**
붕괴 여부는 생성 s₀ 의 **CSD / Shot Diversity 엔트로피를 GT 뱅크와 대조**해 잡는다.

### 더 안전한 대안 — 2단계 분리
구도 버킷을 먼저 고르고(이산 분류 또는 **비학습 열거**) 그 조건 하에 shape 을 생성.
AdaViewPlanner 가 이 구조이고 논문이 *"두 단계를 합치려 했으나 timestep 선호가 충돌해
실패했다"* 고 적었다. 연속 회귀보다 이산 버킷이 붕괴에 강하다. 대가는 조합 상관
(어떤 시작 구도가 어떤 모션과 어울리나)을 모델링하지 못하는 것.

**권장 순서: 캡션에 구도 절이 들어간 뒤에 s₀ 생성을 켠다.** 그 전에 켜면 붕괴만 확인한다.

---

## 47. 시작 pose 를 **생성하는** 논문 7가지 처리 방식 + 추천 (2026-09-05)

| 전략 | 논문 | 어떻게 | 교훈 |
|---|---|---|---|
| 1. 별도 채널 + 자체 표준화 | **E.T./DIRECTOR** | frame0 = 절대 translation 채널, frame1.. = 속도 (`shift_std 1.4` vs `norm_std 0.028`) | **붕괴** (R≈I, \|t\|≈1.27=`shift_mean`). 구조는 되지만 텍스트만으론 unidentifiable |
| 2. 아예 지운다 | **GenDoP** / 우리 emit / CLaTr eval 로더 | `c2w[0]=I` 강제 + 스케일 별도 토큰 | 가장 안정적, 구도를 포기 |
| 3. canonical 씬 프레임의 **절대** pose | **Director3D Traj-DiT** (`system_traj_dit.py:141-152`) | 토큰 = [quat(4), trans(3), intr(4)], **앵커 없음**. 토큰 0 이 곧 시작 pose | 씬이 좌표계를 정할 때만. 우리는 상대 latent 라 혼합 불가 |
| 4. **보조 모달리티 + guidance** | **PulpMotion** | human/camera latent → linear transform → framing latent, 그 map 으로 **auxiliary sampling** | 절대 배치가 만족할 *관계*를 모달리티로 승격. pose 채널이 알아서 배우길 기대하지 않는다 |
| 5. 2단계 + **GT dropout 커리큘럼** | **AdaViewPlanner** | Stage II = human 좌표계 기준 **절대** extrinsics (flow matching). Stage I 은 **GT 카메라를 p=0.5** 로 제공 | ① subject-anchored 절대 예측 ② **p=0.5 가 명시적 anti-collapse** ③ 두 단계 통합은 timestep 선호 충돌로 실패 |
| 6. **입력 마스킹 스케줄** | **LensCraft** | text / subject 궤적+부피 / key points / 전체 궤적, **progressive masking** | 한 모델 여러 프로토콜. "일부만 주어짐"을 명시로 학습 |
| 7. 다중 shot **전역 프레임** | **ShotVerse** | 캘리브레이션 → 통합 전역 좌표계 → shot별 12D pose, **CAS** 로 검증 | 전역 프레임이 있으면 시작 pose 가 의미도 갖고 검사도 된다 |

**반대 방향 참고 — ReRoPE** (2602.08068, `video_generation/models/ReRoPE_` vendored):
기준 프레임을 없애고 **쌍별 상대** `P_c P_t⁻¹` 만 쓴다. 첫 프레임 앵커가 *"특정 전역 좌표계에
묶여 일반화를 제한한다"* 는 이유. 우리에겐 안 맞지만(depth-warp 가 절대 frame0 을 요구)
**"frame0 을 특별하게 두는 것 자체가 비용"** 은 유효하다.

### 🔴 그리고 `--free_start` 를 켜면 고쳐야 할 코드가 하나 있다
`emit.py:105 unit_scale(rel21)` 이 `rmax = max_f |rel[f,:3,3]|` 로 나눈다.
`--free_start` 면 `rel[0]` 에 시작 오프셋이 들어가므로 **rmax 가 오프셋에 지배된다** —
소스에서 2u 떨어져 시작하는 작은 orbit 은 모션이 unit scale 의 극히 일부가 되고,
`g = rmax/S` 가 "카메라가 얼마나 움직였나" 대신 **"시작점이 얼마나 먼가"** 를 뜻하게 된다.
→ shape 정규화를 **`max_f |rel[f] − rel[0]|`(모션 범위)** 로 바꾸고 오프셋은 s₀ 로 분리.
(`assert rel[0]=I` 는 `if not free_start` 로 이미 가드돼 있다 — 그쪽은 함정 아님.
 `relativize(poses, anchor)` 가 임의 앵커를 받고 `infcam` 이 이미 mid anchor 를 쓴다.)

### ✅ 추천 — 4개를 겹친다
**① 표현 (전략 3+6)** `x_t = [ s₀(1) | latent(13) ]`, s₀ = (rot6d(R₀), t₀) **자체 표준화**.
   shape 정규화는 모션 범위로 (위 수정).
**② 학습 (전략 6+5)** 마스킹, **p ≈ 0.5**. `cfg_dropout_p 0.1` 은 s₀ 에 부족하다 —
   90% 주어지면 모델이 그 분포를 배울 이유가 없다. AdaViewPlanner 가 p=0.5 를 명시적
   anti-collapse 로 쓴다. 추론: "준다"=clamp / "안 준다"=생성, **가중치 하나**.
**③ 붕괴 방어(구조)** s₀ = **이산 구도 버킷(az/el/shot-scale) + 연속 잔차**.
   GenDoP 은 스케일까지 `discrete_bins` 로 이산화하고 ShotVerse 는 Pose De-Tokenization 을 쓴다.
   저신호 채널에서 이산 분류가 mean-collapse 에 강하고, 버킷이 캡션 어휘와 1:1 이라
   identifiable 해지며 **CSD 엔트로피가 직접 읽힌다**.
**④ 붕괴 방어(샘플링, 재학습 0 — 가성비 최고)** PulpMotion auxiliary sampling 특수화.
   단일 카메라라 map 이 **닫힌 형태**다:
   `화면 위치 = π( R₀ᵀ (C − t₀) )`, C = subject 중심(context 좌표계, 이미 있다).
   미분 가능 + 렌더 0회 → 매 denoise step 에서 s₀ 를 캡션의 목표 NDC/shot-scale 로 끌어당긴다.
   **학습을 안 건드리고 identifiability 부담을 줄이는 유일한 수단**이고 ①~③ 위에 그냥 얹힌다.

### 하지 말 것
E.T. 식 "절대 채널 + 표준화만"(측정된 붕괴) / Director3D 식 전 프레임 절대(상대 latent 와 혼합 불가)
/ p=0.1 / deterministic L2 head(조건 평균 붕괴).

### 순서
**④ 먼저** (재학습 0, 닫힌 형태, 하루) → 되면 ③ 이산화가 필요한지까지 판단된다 →
①+② 로 1회 학습 → ③ 은 CSD 엔트로피가 GT 뱅크보다 유의하게 낮을 때만.
단 ④ 도 **`unit_scale` 수정 + 캡션 구도 절**이 선행돼야 한다 (목표 NDC 가 없으면 끌 목표가 없다).

---

## 48. 🔴 §46~47 재고 — s₀ 를 **생성**할 이유가 약하다 (2026-09-05, 사용자 반문)

§46~47 이 세 질문을 뭉쳤다. 분리하면 답이 달라진다.

| | 질문 | 답 |
|---|---|---|
| (a) | **데이터**에 다양한 시작 pose 가 있어야 하나 | **네, 명확히.** 구도·표현력이 거기 있고 꺾임이 원인 단계에서 사라진다. 제안 S, 모델과 무관 |
| (b) | **모델이 조건으로** 받아야 하나 | **네.** 싸고(`cond_dim`), 프로덕션(depth-warp)이 frame0 을 요구하고, shape 과제가 well-posed 해진다 |
| (c) | **모델이 생성**해야 하나 | **약하다** ↓ |

### (c) 가 약한 이유 4개

**1. 학습 데이터에 배울 상관이 없다 — 결정적.**
d121 실측이 preset 40종에 대해 정확히 **1168개씩 완전 균일한 곱격자**다. 제안 S 로 구도 축을
더해도 라우팅을 안 하면 **s₀ ⟂ shape** 이다. joint 생성이 사는 이유는 "low angle 시작은
push-in 과 어울린다" 같은 상관을 잡는 것인데, **GT 에 그 상관이 없으면 joint 로 얻을 게 없다.**
`route_presets` 처럼 구도↔preset 을 라우팅해 상관을 만든 뒤에야 의미가 생긴다.

**2. 프로덕션 경로가 원하지 않는다.**
최종 목표는 depth-warp 재렌더다. 시작 pose 를 소스에서 떼면 **frame 0 자체가 hole** 이고
그건 오차가 앞으로 전파되는 최악의 위치다. `hole_static` p90 = 0.391(d121)에 오프셋이 더해진다.
`fit_hole_ladder` 가 스스로 *"0.50 위는 '생성'이지 '카메라 이동'이 아니다"* 라고 적었다.

**3. 지표가 못 본다.** CLaTr eval 로더가 재앵커하므로(§34) 대표 지표의 s₀ 보상이 **0** 이다.
보상하는 지표(CSD / Shot Diversity / frame0 framing)는 새로 만들어야 하고, 만들어도
GenDoP 은 그 축에 참가하지 않고(§37) E.T. 는 붕괴한다 → 점수 경쟁이 아니라 capability 서술.

**4. identifiable 하게 만들면 "생성"이 아니게 된다.**
§47 의 붕괴 방어(캡션 구도 절 / 이산 버킷 / guidance)는 전부 **답을 알려주는 것**이다.
구도 절이 s₀ 를 버킷까지 결정하면 그건 생성이 아니라 텍스트 디코딩 + 실현가능성 투영이고,
**그 일은 비학습 샘플러가 더 잘한다** (결정적 · 열거 가능 · 게이트로 보장).

### ✅ 수정된 추천 — s₀ 는 **생성**이 아니라 **선택**

- 후보 열거(board pool `az×el×r` + `obs_az_span` 필터) → 게이트(G1/G4 렌더 0회) →
  `hole_static` 으로 가격. **전부 이미 있다**
- 선택 = 캡션 구도 절과의 매칭 = **분류**. 붕괴에 강하고, 항상 실현가능하고, 다양성이
  통제되고, "고른 pose 가 캡션 구도와 맞나"로 직접 평가된다
- 추론 시 사용자가 s₀ 를 안 줘도 **샘플러가 N개를 제시**하면 된다 —
  Director3D 데모가 정확히 그 패턴이다 (`app.py:146` 4개 궤적 중 "Which trajectory do you prefer?")

**즉 모델은 shape 만 생성하고 s₀ 는 조건으로 받는다.** 그러면
① VAE 그대로, `unit_scale` 수정 불필요(오프셋이 latent 에 안 들어옴)
② CLaTr 비교가 깨끗하게 shape 축 — GenDoP 과 정확히 같은 자리
③ s₀ 마스킹/dropout 도 불필요

### (c) 를 다시 꺼낼 조건 2개
1. **구도↔preset 라우팅으로 GT 에 상관이 생겼을 때** (곱격자를 깰 때)
2. **CSD / Shot Diversity 를 지표로 갖췄을 때** (붕괴를 판정할 수 있을 때)

그때 §47 의 ①②④(토큰 + p≈0.5 마스킹 + guidance)를 그대로 쓴다. 지금은 **ablation 한 줄**로
남기는 게 정직하다 — 곱격자 데이터에서 "s₀ 를 생성하면 붕괴한다(E.T. 재현)"는 예상되는 결과이고,
주장은 (a)+(b) 로 충분히 선다.

---

## 49. LensCraft 를 읽고 정한 포지션 (2026-09-06)

### 정면 승부는 지는 싸움
LensCraft = 시뮬레이터 **10만 샘플 / 300만 프레임**, 정확한 GT, static:dynamic 1:1 균형,
FID 24.35(dynamic) / 40.4(static) vs E.T. 151~161 / CCD 307~320, 추론 1.66 s / 1.64 GFLOPs.
우리 = **318씬 / 5.3만 변이**, DA3 추정 pseudo-GT, 코퍼스마다 moving 비율 24.6% vs 94.6%.
→ **FID·PRDC 를 헤드라인으로 걸면 밀린다.**

### 문헌이 두 진영이고 접합부가 비어 있다

| | 진영 A — 촬영 (text→camera) | 진영 B — NVS (camera→video) |
|---|---|---|
| 논문 | E.T., CCD, **LensCraft**, GenDoP, VERTIGO, ShotVerse | ViewCrafter, GEN3C, TrajectoryCrafter, ReCamMaster, Vista4D |
| 씬 기하 | 없음(시뮬레이터) 또는 첫 프레임만 | 있음 |
| 궤적 | **생성한다** | **주어진 것으로 받는다** |
| 못 답하는 질문 | "이 카메라가 **이 씬에서** 실현 가능한가" | "이 카메라가 **의도에 맞는가**" |

접합부의 화폐 단위가 이 레포가 발명한 **hole 예산**이다. LensCraft 자체 한계 3번
(*"다중 피사체 상호작용·복잡한 가림 미해결"*)이 그 경계선이다.

### 포지션: **scene-grounded, render-feasible camera generation**
기여 단위를 궤적 모델이 아니라 **실현가능성 계량**으로 잡는다. 그러면
① 규모가 작아도 된다(기여가 "더 큰 데이터셋"이 아니라 "가격이 붙은 데이터셋")
② LensCraft 가 경쟁자가 아니라 **상류 어휘 공급자**가 된다
③ 그들의 비판 2번(SLAM 오차 전파)은 **측정해 보고하면 기여**가 된다 (진영 A 에 전례 없음)

### 주장 스택 (방어 가능한 순서)
1. **실현가능성을 1급 축으로** — hole 사다리 + 소스 대비 임계 + **"hole 사다리 → 하류 품질"
   곡선**(Part II C9). **양 진영 어느 논문도 없다.** 26 GPU-h
2. **tracking + free-moving 통합 canonicalization** — context 첫 카메라 원점에서 E.T./GenDoP
   규약이 **고정 변환으로 복원**된다 (§35). 아무도 명시하지 않았다
3. **TRUMANS = proxy 캘리브레이션** — LensCraft 는 정확한 GT/현실 없음, in-the-wild 는
   현실/GT 없음. TRUMANS 만 **둘 다**(mesh+SMPL GT). 비판 2번에 대한 직접적 답 (Part II C4)
4. **(헤드라인 금지) 궤적 품질 수치** — CLaTr/FID/PRDC 는 *sanity* 로만

### 채택 / 비채택
**채택 (거의 공짜 + 인용 가능)**
- **SCD 4축**(shot type / angle / framing / movement+speed) ↔ 우리
  (distance_ratio / elevation / composition / preset+speed) 거의 1:1 → 캡션이 임의 preset
  이름 대신 **근거 있는 taxonomy** 를 갖고 라벨 수준 비교가 열린다
- **ABox/VBox** → shot scale 이 결과에서 조건으로 (P7-c)
- **progressive masking (0.1→0.8)** → s₀/track/키프레임 given-or-not 을 한 ckpt 로 (§47 ②)

**비채택**
- **"decoder 는 high-level 만 본다"** — 방향 반대. 그들은 씬이 없어 저수준이 곧 정답 복사지만,
  우리에게 DA3 저수준 기하는 hole·충돌을 피하는 **유일한 근거**다
- **다양성/표현력 우위 주장** — 그들은 설계상 균형, 우리는 318씬 곱격자 + 캡션 불일치 26~31%.
  CSD 로 확인하면 제안 S 전에는 진다

### 멈춰야 할 것
- CLaTr FCD 를 헤드라인에서 내리고 §34 두 사고(스케일 미정규화 / 49 vs 120) 먼저 고치기
- preset 격자를 더 늘리는 것 (다양성은 우리 축이 아니다)
- **텍스트 정합 주장 전에 캡션 26~31% 불일치 선행 수정** — LensCraft/GenDoP 와 직접 비교되는
  유일한 축이라 데이터 결함을 안고 들어갈 수 없다

### 한 문장
> 더 좋은 카메라 생성기를 만드는 것이 아니라, **실제 dynamic 영상 위에서 "이 카메라가 렌더될 수
> 있는가"를 가격으로 매기는 최초의 파이프라인**을 만들고, 그 가격이 하류 생성 품질과 어떻게
> 연결되는지 측정한다.

LensCraft 는 그 문장의 **상류 어휘**, ViewCrafter/Vista4D 는 **하류 소비자**다.

---

## 50. 로드맵 (2026-09-06)

### 현재 위치 (실측)
**2×3 그리드** — 같은 test 세그먼트 위:

| | geo only | molmo2 only | geo+molmo2 |
|---|---|---|---|
| vista d121 | D123 | D133 | D124 |
| dynpose d137 | D137 | D138 | D141 |

**+ D144** = da3 **49뷰 + PerViewResampler**(3136 tok) → "인코더 vs 프레임수" 분리
(D137 6뷰×576=3456 / D138 49프레임×64=3136 / D144 49뷰×64=3136).
d137 코퍼스: train **240씬 / 21,927 seg**, test **27씬 / 2,444 seg** (dd10 9,728 / 1,129).
`EXPERIMENTS.log`: D137 vs D138 에서 **molmo2 9/11 지표 우세**, D144 가 원인 분리용.
→ **아키텍처 축은 규율 있게 진행 중. 손댈 것 없다.**

### 진단 — 그리드는 좋은데 **자가 고장 나 있다**
1. eval dump 스케일 미정규화 + 49 vs 120 (§34) → FCD 계열이 단위 섞인 값
2. 캡션 **26~31%** 가 기하와 어긋남 (P1) → text alignment 지표가 그 위에 서 있다
3. 11개 지표 전부 궤적/텍스트 축 → 완료 조건 **(b) 실현가능성 0개**
4. 뱅크 71열에 매끄러움 열 **0개** (§26) → 꺾임이 안 보인다

"molmo2 9/11 우세"는 **결함 있는 자로 잰 값**이다. 결론이 뒤집히지 않을 수도 있으나 그 위에
아키텍처를 더 쌓으면 되돌리기 비용이 커진다. GPU 0~4 가 전부 점유 상태이므로
**GPU 를 거의 안 쓰는 Phase 0 을 지금 끼우는 것이 자원 배분상도 맞다.**

### Phase 0 — 자를 고친다 (1주, 재굽기 0, GPU ~0)
| | 무엇 | 근거 |
|---|---|---|
| 0.1 | eval dump ref/pred 동일 인자 정규화 + 프레임수 통일 | §34 |
| 0.2 | 캡션 target 절을 **실측 `subject_in_frame`** 으로 게이트 (F0) | P1, §21 |
| 0.3 | 뱅크 열 4개: `judgeable_frac` / `jerk_t,jerk_r` / `aim_err_*` / `rot_deg` | C4, §26, P7-a |
| 0.4 | `eval_subject_in_frame.py --gates` | M4 — (b) 첫 측정 |
| 0.5 | CLaTr 우리 코퍼스 재학습 + held-out 씬 R@k | C1 |
**Exit: 2×3 그리드를 고친 자로 재독.** 순위 변동 여부 자체가 결과다.

### Phase 1 — 그리드가 못 가르는 축 하나 (2주, 1~2 arm)
D144 가 "인코더 vs 프레임수"를 가른다. 남은 confound = **캡션 형식 vs 정보량**.
사용자 가설("자연어라 molmo2 와 분포 호환")이 맞다면 **NL 이득이 molmo2 arm 에서 더 커야 한다**
(형식×인코더 상호작용). 그런데 현재 NL 캡션은 **정보량도 더 많다**(VLM referring expression
+ shot-scale 전이 + composition). `--prompt_style fields` 가 옛 형식을 바이트 단위로 재현하므로
같은 d137 뱅크로 캡션 한 벌 더 구워 **2×2 (형식 × 인코더)** 를 돌리면 갈린다.
**이 답이 Phase 2 투자 방향을 정한다** — 정보량이면 구도 절 추가(제안 S), 형식이면 텍스트 인코더.

### Phase 2 — 데이터 한 판 (3~4주, `fix.log` 규약대로 몰아서)
framing 예산 #8(제안 F) · F7 시간축 절단 · P3 `--follow_gains auto`(dynpose `tau_start>0.10`
50.3%) · P9 absolute hole 상한 · **제안 S 시작 구도 샘플러 + `--free_start` + `unit_scale`
모션범위 수정 + K1 비균일 keyframe** · **SCD 4축 어휘 + 구도 절 캡션화** ·
route_presets 태우기(d121 포함) · stale 분기 2건(P4/P5).
**Exit: 캡션이 기하를 정확히 약속하고 구도가 조건·캡션·지표 세 곳에 존재.**

### Phase 3 — 모델 (Phase 1/2 종속, Phase 2 와 병렬 가능, 3~4주)
s₀ = **조건**(선택 문제, 생성 아님 §48) + subject track 조건(M8, 실측 0.0152 vs 0.0264) ·
**`anchor_pred_frame0: false`** · M1 `need_weights=False`(수치 불변) · M2 text/geo residual
gate arm · LensCraft식 **progressive masking** 으로 given/not 을 한 ckpt 에.

### Phase 4 — 플래그십 + 최종 평가 (4~6주)
- **hole 사다리 → 하류 품질 곡선** (26 GPU-h) ← §49 주장 **1위**, 양 진영 전례 없음
- **TRUMANS proxy 캘리브레이션** (mesh SDF collision + SMPL framing) ← 주장 3위 +
  LensCraft 비판2 에 대한 답
- **L3 VLM 평가**: d137 test **27씬** (20~27씬 × 변이 3~5 = 60~135영상 = **10~22 GPU-h**).
  C5 의 "vista 4씬" 문제는 dynpose 로 **이미 해소**
- **DataDoP shape 단위 holdout** → 순환성 깨는 무료 축 (C7)

### 장기
ViewCrafter 식 NBV 로 world 확장(goals 장기1) · TRUMANS human+camera joint(장기2)

### 지금 하지 말 것
새 인코더 arm(2×3+D144 로 충분, Phase 0 전엔 순위 불신) · preset 격자 확대(§49) ·
s₀ 생성(§48) · CLaTr FCD 를 헤드라인(0.1 전엔 단위 섞인 값)

### 임계 경로
**0.1+0.2 → 그리드 재독 → Phase 1 → Phase 2 → Phase 4.** Phase 3 은 2 와 병렬.

---

## 51. 구조도 — 현재 / 추천 (2026-09-07)

### 51.1 현재 (D124 / D141)

```
── 입력 ──────────────────────────────────────────────────────────────────────
 캡션 ─▶ umt5-xxl 4.63B frozen ★매 스텝 forward → text_emb   (B, 128, 4096)
 49프레임
   ├─▶ Molmo2-4B frozen·캐시·ln_f 직후 = ★최종층
   │     ├─ prefix forward(캡션 없음): patch 3969 → 프레임내 9×9→8×8
   │     │    └─ video_emb      (B, 3136, 2560)   ✗ text 미융합
   │     └─ prefix+caption forward → 캡션 위치 slice
   │          └─ video_text_emb (B,  128, 2560)   ✓ video⊗text
   └─▶ DA3(backbone frozen / ln·proj 학습) 6뷰×576
         └─ geo_emb             (B, 3456,  768) + mask
 subject track ─▶ cond (B,13,4)   ← 기본 OFF

── 타깃 ──────────────────────────────────────────────────────────────────────
 cam_param (B,49,11) ─ VAE.encode ÷ vae_latent_scale (stride2×2: 49→25→13)
   └─ z₀ (B, 13, 64)

── Camera DiT (hidden 512, 8층, 8head) ───────────────────────────────────────
 x_t(B,13,64)+cond ─ cam_in(64+cond→512)+PE
 ×8: norm1→FiLM(mod1) → SA(norm(x+a))
     → text CA  ◀ text_tok(128,512)   norm(x+a)  ✗잔차 덮어씀
     → mlp1+h
     → [video 블록] vnorm→FiLM(mod3) → video CA ◀ cat([tt128|vt3136])=3264
                    → vmlp+v → h = h + video_gate[l]·v   ✓잔차 보존(0 init)
     → norm2→FiLM(mod2) → geo CA ◀ geo_tok(3456,512)  norm(x+a) ✗잔차 덮어씀
     → mlp2+h
 out(512→64) → eps_hat (B,13,64)      loss = MSE(eps_hat, eps)

 query 13 : key/value 128+3264+3456 = 6,848  →  1 : 527
```
**문제 4개**: ★최종층 / text·geo 잔차 덮어씀(8층×2 ≈ 1/256) /
video 3136에 캡션 정보 없음(융합분 128/3264 = **3.9%**) / query:key 1:527.

### 51.2 추천

```
── 입력 ──────────────────────────────────────────────────────────────────────
 49프레임
   ├─▶ Molmo2-4B frozen·캐시·① layer ≈ 18 (중간층)                    ★
   │     ├─ patch 3969 → ② PerViewResampler(학습) → 512               ★
   │     │    └─ video_emb (B, 512, 2560)
   │     ├─ 캡션 위치 → cap_emb (B,128,2560) → ③ text CA 로 승격       ★
   │     │                                       (umt5 제거)
   │     └─ ④ pointing → 타깃 per-frame 좌표+가시성 (B,49,3) → 13 pool ★
   └─▶ DA3 49뷰 + PerViewResampler(D144) → geo_emb (B,3136,768)+mask
 ⑤ cond = [ s₀(9) | track(4) | 타깃 좌표+가시성(3) ] = (B,13,16)        ★
      s₀=(rot6d(R₀), t₀/avg_scale), context 첫 카메라 좌표계, **조건**

── 타깃 ──────────────────────────────────────────────────────────────────────
 VAE 변경 없음 (shape 전용, s₀는 latent 밖 §42)
 vae_latent_scale 을 코퍼스마다 재측정 (전역 상수 금지)                 ★필수

── Camera DiT ────────────────────────────────────────────────────────────────
 x_t+cond ─ cam_in(80→512)+PE
 ×8: norm1→FiLM(mod1) → SA
     → ⑥ text CA ◀ cap_tok(128,512);  h = h + text_gate[l]·t   ✓ init 1.0  ★
     → ⑦ video CA ◀ vt(512,512)  ← 캡션 토큰 제거;  h = h + video_gate[l]·v ★
     → ⑧ geo CA  ◀ geo_tok(3136,512); h = h + geo_gate[l]·g    ✓ init 1.0  ★
     → ⑨ 모든 CA/SA: need_weights=(not training)                          ★
 out(512→64) → eps_hat (B,13,64)

 query 13 : key/value 128+512+3136 = 3,776  →  1 : 290  (−45%)

── 디코드 ────────────────────────────────────────────────────────────────────
 ẑ₀ × vae_latent_scale → VAE.decode → (B,49,11) → out_to_trajectory(×avg_scale)
   └─ ⑩ s₀ ∘ rel → context 첫 카메라 좌표계 world pose                  ★
        · anchor_pred_frame0 = false 필수
        · emit unit_scale 을 max|rel[f]−rel[0]| (모션 범위) 로
```

### 51.3 변경 항목

| # | 변경 | 비용 | 기존 보존 | 근거 |
|---|---|---|---|---|
| ① | Molmo2 최종층 → **중간층(~18)** | 캐시만, 모델 0줄 | 옵션 | GR00T N1: *"middle-layer 가 더 빠르고 성공률도 높다"* (Eagle-2 12층) |
| ⑨ | `need_weights=(not training)` | 한 줄 | **수치 불변** | `CrossAttention:65` 가 (B,13,3456) 무조건 저장 |
| ⑥⑧ | text/geo 도 residual gate | arm 1 | `*_gate=None` | FIX.log 자기 진단의 미완 절반 |
| ② | molmo2 patch 3136 → 512 | 캐시 + ~7M | 옵션 | key 예산. D144 resampler 재사용 |
| ③⑦ | **umt5 제거**, 캡션 hidden 을 text CA 로 | arm 1 (text_dim 4096→2560) | ✗ | 캡션 hidden 이 이미 video-fused. **−4 min/epoch** |
| ④ | pointing → 명시 타깃 채널 | 중간 | 옵션 | video 3136 에 겨냥 정보 없음 (3.9%만 융합) |
| ⑤ | cond 확장 (s₀+track+타깃) | config+arm | `cond_dim=0` | track 실측 0.0152 vs 0.0264 |
| ⑩ | s₀ 를 디코드에서 합성 | 작다 | — | §48 s₀ = 선택 문제, latent 밖 |

**순서**: ①⑨ (각각 단독 arm) → ⑥⑧ → ②⑦③ (묶어야 해석됨) → ④⑤⑩ (제안 S 와 함께).

**③ 리스크**: umt5 는 **유일한 순수 텍스트 채널**이었다. 캡션 hidden 으로 바꾸면 영상에
오염되므로(그게 목적이지만) **CFG 가 모호해진다** — 텍스트 drop 이 "영상 일부 drop" 이 된다.
**학습 가능한 null 임베딩을 따로** 두어야 CFG 해석이 유지된다. ⑦과 ③ 동시 투입 전에 결정.
