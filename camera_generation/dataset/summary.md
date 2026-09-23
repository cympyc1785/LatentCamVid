# LBM-Lite / TRUMANS-Lite 요약 (2026-08-23)

> **이 파일은 2026-08-23 시점의 스냅샷이다** — 그 뒤로 트리가 재편됐으므로 아래 경로·명령은
> 현재 코드와 다르다. 지금의 입구는 `camera_generation/dataset/exec/run_bank.py`
> (체인 `graph → cloud → route → tau → fit → emit`, 설정은 `configs/bank/<gen>.json`) 이고,
> 아래 `run_lbm_lite.py` 는 **저장소에 더 이상 없다.** 수치와 설계 근거만 읽을 것.
>
> 원본 근거는 `camera_generation/dataset/DECISIONS.md` D1–D69, `video_generation/CHANGELOG.md`,
> `camera_generation/latentcam/{EXPERIMENTS.log,FIX.log}` 에 있다. 아래 수치는 전부 실측이다.
>
> (옛 헤더는 "`.gitignore:222 camera_generation/models` 아래라 커밋에 안 들어간다" 고 적었지만
> **두 겹으로 틀렸다** — R7 에서 `camera_generation/dataset/` 로 옮겨져 지금은 추적 중이고,
> 인용한 `.gitignore:222` 는 현재 `tools` 줄이다.)

---

## 1. LBM-Lite 가 뭔가

영상 1편(49프레임 RGB+depth) → **그 영상 위에서 물리적으로 렌더 가능한 target 카메라 뱅크**.
LatentCam 학습용 (source video, target camera, caption) 삼중항을 만드는 게 목적이다.

### 파이프라인 두 갈래

```
run_lbm_lite.py --stage bank_all   # VLM 없음 (실제로 51편에 돌린 것)
  graph → pcd → bank → fit → bankemit → bankvid

run_lbm_lite.py --stage all        # VLM 경로 (초기 설계)
  graph → pcd → board → loop → decode → emit → verify
```

| 단계 | 하는 일 |
|---|---|
| `graph` | recon+seg → `scene_graph.json`: subject OBB·track, 중력축, 단위 `S`, near/supported_by |
| `pcd` | Vista4D `unproject()` 로 4D 점군 `cloud.npz`. `visible.sum(1)==1` 이 동적 점 |
| `bank` | anchor(시작 pose) × preset × 사다리단 전수 열거 |
| `fit` | 각 조합에 대해 **게이트를 안 깨는 최대 크기**를 이분법으로 푼다 |
| `bankemit` | `canonical.json` / `canonical.npz` (rel c2w 21×4×4, OpenCV, frame0 anchor) |
| `bankvid` | Vista4D 점군 렌더 프리뷰 |

### 단위·기호 (혼동 방지)

- `S` — scene unit. frame0 non-sky 픽셀의 `z·‖K⁻¹[u+0.5,v+0.5,1]‖` 평균. `1 u ≜ S`.
- `τ(f)` — `|p_plan(f) − p_src(f)| / z_med(f)`. **plan 카메라가 소스 카메라에서 얼마나 벗어났나**를
  그 프레임의 median depth 로 나눈 값. 시차(parallax) 예산.
- `hole` — `1 − valid_mask.mean()`. Vista4D 점군 렌더에서 **비어 있는 픽셀 비율**.
  하류 video model 이 채워야 하는 양의 직접 측정치.
- `knob` — 궤적 크기 손잡이 (preset 마다 의미가 다르다: sweep 각도 / dolly 거리 / ...).
- `fold` — 사다리의 여러 단이 **같은 궤적으로 접히는 것**. 게이트가 같은 상한에서 걸리면 발생.

---

## 2. 설계 결정 4개와 그 이유

### (1) VLM selector → 전수 열거 (enumeration)

초기 설계는 VLM 이 후보 board 를 보고 시작 pose 와 preset 을 고르는 것이었다.
**실측: `orbit_left_arc` 가 22/22 로 뽑혔다.** VLM prior 가 지배해서 선택이라고 부를 수 없었다.
→ 선택을 없애고 (anchor × preset × 사다리) 전수를 뱅크에 넣는다. 고르는 건 소비자(학습 샘플러) 몫.

### (2) τ 사다리 → hole 사다리 (D53)

τ 를 일정하게 고정해도 **씬마다 hole 이 43× / 15× 로 벌어졌다.** τ 는 카메라 이동량이지
"영상 모델이 얼마나 상상해야 하나"가 아니다.
→ 사다리를 `hole_static + Δ`, `Δ ∈ {0.0, 0.10, 0.25, 0.40}` 로 바꿨다 (`--hole_mode excess`).
`hole_static` 은 그 씬에서 카메라를 안 움직였을 때의 hole 이라 씬 난이도가 상쇄된다.

### (3) 게이트 임계는 절대값이 아니라 **소스 자신의 여유의 배수**

camel 의 min clearance 0.5107 u vs avocado-slice 0.1165 u — **4.4× 차이.**
절대 임계를 쓰면 한쪽은 전부 통과하고 한쪽은 전부 탈락한다.
→ G5(OBB) / G7(approach) 는 `m = β × (소스 카메라 자신의 min clearance)`, `β = 0.3`.
G6(elevation) 도 `max(45°, |src elev| + 10°)` 로 소스 상대.

| 게이트 | 판정 |
|---|---|
| G1 behind-surface | 후보 중심을 소스 프레임에 투영, `z_cam > depth + 0.02·S` 인 프레임이 있으면 기각 (카메라가 벽 속) |
| G5 OBB | subject bbox 에 `m` 이내로 붙으면 기각 |
| G6 elevation + ground | 올려다보는 각 상한 + 지면 아래 금지 (ground ratio 0.2) |
| G7 approach | dolly 가 subject bbox 를 **지나쳐 버리면** 기각 (같은 β=0.3) |

### (4) 뱅크는 재고 목록이다 (D39/D45)

게이트에 걸린 변형도 **사유 태그를 달아 뱅크에 남긴다.** 지우지 않는다.
이유: 어떤 축에서 왜 막혔는지가 다음 라운드의 진단이 된다. `--drop_folded` 는 기본 off.

---

## 3. 품질 실측 (전 51편, `num_cameras` 합 16,520)

### 결과 태그 분포

| 태그 | 비율 | 뜻 |
|---|---|---|
| solved | **41.3 %** | 게이트 안 깨고 목표 hole 도달 |
| approach_limited | 10.0 % | G7 에 먼저 걸림 |
| ground_limited | 8.9 % | 지면 아래로 내려감 |
| clamped_low + tau_floor | 7.6 % | 사다리 아랫단이 소스 시차보다 작아 정지 |
| obb_limited | 7.6 % | G5 |
| elev_limited | 7.1 % | G6 |
| shape_limited | 6.7 % | preset 형태 자체가 안 됨 |
| unreached | 5.8 % | 이분법이 목표 hole 못 맞춤 |
| collision_limited | 2.5 % | G1 |

### 접힘(fold)

`1,995 / 4,130 = 48.3 %` 의 (anchor, preset) 조합에서 사다리 단들이 같은 궤적으로 접혔다.
= 게이트 상한이 사다리 목표보다 먼저 걸려서, 4단이 전부 "최대 크기" 하나로 수렴.

`pan` 계열 `2,360 = 14.3 %` 는 **translation 이 정확히 0** 이다 (제자리 회전이라 당연).
τ/hole 사다리에는 올라탈 수 없다 — 별도 `pan_deg` 축이 필요하다 (D68 옵션 ③, 미판정).

### 단계별 소요시간 (`out/run51/*.json`)

```
graph   n=1  147.6 s
pcd     med   23.3 s   sum  0.35 h
bank    med  276.8 s   sum  4.38 h
fit     med 1647.4 s   min 15.7  max 4368.9   sum 24.50 h   <- 전체 ~30.8 h 의 80 %
bankemit med   9.7 s   sum  0.14 h
bankvid  med  98.2 s   sum  1.43 h
```

### 커버리지

```
코퍼스 52편 (zz* 10개는 scratch)
scene_graph 52 / cloud 52 / bank 52 / hole_bank 50 / skipped 2
num_cameras  sum 16520  med 308  min 56  max 672
worst roundtrip 0.0    worst pose rebuild 0.0
```

미완 2편은 D67 로 **의도적 기각**:
- snow-bike: `tau_start 1.3765`, dropped_saturated 210, surviving 6, `S=2.8809`, `z_med=0.8145`
- snowboard: `tau_start 1.9441`, dropped 350, surviving 10, `S=2.9229`, `z_med=1.7510`

`tau_start` 은 **카메라를 안 움직여도** 소스 자신의 시차가 이미 큰 경우다. 이런 씬은 정지 hold 조차
사다리 아랫단을 넘어서 사다리가 성립하지 않는다.

### D68 옵션 ① dedup (2026-08-23 적용)

emit 단계에서 `(anchor, preset)` 안의 같은 `knob` 값을 접었다. **재fit 없음, 코드 변경 없음** —
`emit_bank.py --drop_folded` (:375) 를 켜고 `--out_dir <bank>/canonical_dedup/` 로 냈다.
기존 `<bank>/canonical/` 은 그대로 둔다.

```
16,520 → 10,563  (−36.1 %)
worst pose rebuild 0.0 / worst roundtrip 0.0
per-video  med 200.5  min 35  max 472
translation_degenerate 2,300 (21.8 %)   <- pan 계열, 옵션 ③ 미적용이라 남아 있다
```

fold 키 검증: `knob`(5dp) vs `knob_raw` → 양쪽 다 5,957 fold, **차이나는 영상 0편.**

가장 많이 줄어든 것 / 가장 적게 줄어든 것:

| 영상 | before → after | |
|---|---|---|
| elderly-tennis | 280 → 70 | −75.0 % (4단이 전부 하나로) |
| martian-flag | 168 → 42 | −75.0 % |
| funeral-procession | 336 → 170 | −49.4 % |
| parkour | 392 → 204 | −48.0 % |
| park-selfie | 224 → 117 | −47.8 % |
| woman-pottery | 56 → 43 | −23.2 % |

접힌 행은 `bank.json` / `bank.csv` 에 그대로 살아 있어서 진단은 보존된다.

구체적 fold 예: avocado-slice `dyn_0 × orbit_left_arc` 의 4단이 전부
`knob 0.89358 / path_len 0.5974 / hole 0.4087 / approach_limited` — 같은 궤적이다.

---

## 4. 데이터 가공에서 터진 것들과 수정

| # | 증상 | 진짜 원인 | 수정 |
|---|---|---|---|
| D11 | `det(R) ≠ 1` | 보간·합성 후 R 이 SO(3) 를 벗어남 | emit 직전 `project_so3` + assert |
| D18 | `rmax ≈ 1.43e-17` | 정지 궤적을 `rmax` 로 나눠 0/0 | 정지 판정 후 정규화 스킵 |
| D26 | frame0 이 **11.6739°** 어긋남 | rel 앵커를 잘못 잡음 | `rel[0]=I` assert (`< 1e-9`) |
| D17 | 같은 입력에 다른 출력 | 결정 경로가 비결정적 | `decision_fingerprint` 로 재현성 고정 |
| D66 | 사다리 단이 조용히 겹침 | `knob` 반올림 오차 `1.793e-03` | fold 키를 5dp 로 명시 + `knob_raw` 병기 |
| D53 | avocado-slice **206/392 가 조용히 정지** | 사다리 하한 τ < 소스 자신의 시차 | hole 사다리로 전환 + `clamped_low`/`tau_floor` 태그 |
| D61 | 5편 → 51편 확장에서 5군데 파손 | 파일럿에만 있던 가정들 | 전량 재실행 + `run51/*.json` 에 단계별 기록 |

공통 패턴: **조용히 망가진다.** 로그는 전부 "성공"으로 찍히고 궤적만 정지/중복이 된다.
그래서 fold·태그·fingerprint 를 전부 뱅크에 남기는 쪽으로 갔다.

---

## 5. TRUMANS-Lite

### 왜 TRUMANS 인가

DL3DV 는 정적 씬이다. LatentCam 이 **사람이 움직이는 씬**에서도 되는지 보려면 동적 코퍼스가 필요하다.
TRUMANS 는 사람 동작 + 씬 자산이 다 있는 합성 데이터다.

### 카메라가 없었다

**67 recording 중 실제 카메라 pkl 이 있는 건 2개뿐이고, `.blend` 안의 카메라는 전부 정지다.**
→ 그 2개의 통계로 소스 카메라를 **합성**했다:

```
49프레임 net 이동   0.58 – 0.62 m
프레임당            0.0128 m
z-span              정확히 0     (높이 고정)
yaw                 사람을 따라감
```

preset 8종: `push_in / hold / arc_push / arc_left / arc_right / drift / arc_pull / pull_out`.
prompt 는 `prompt_camera_with_scene_video.concise` 에 **카메라 움직임 + 사람 동작**을 같이 쓴다.
예: `"The camera arcs to the right while pulling back, keeping a person in frame as they right hand rests on the chair."`

### DL3DV da3 on-disk 계약 재사용

```
<scene>/da3/pose.npz                            OpenCV w2c + intrinsics
<scene>/da3/prompts.json
<scene>/da3/avg_scale_context_first_cam/<k>.json
<scene>/images_4/
<root>/meta_trumans.csv
```

`seg_key` 가 두 단계 이름을 요구해서 chunk 는 `trumans/<uuid>` 로 뒀다.
**scene = recording, segment = clip.** 한 recording 의 clip 들을 프레임 축으로 이어 붙여서,
`geo_posed=true` 아래 context 범위와 target 범위가 겹치지 않게 했다.

### 코퍼스

```
7 recording / 130 clip / 6,370 frame

recording   clips  frames  avg_scale(med)
00add26c     21    1029    2.198
0a761819     17     833    2.880
0aa05d5a     21    1029    2.138
1d43e076     22    1078    3.080
2b4c9b84     14     686    2.268   <- val (recording 통째로 held out)
3a19c7bb     20     980    2.369
4ac2c1b3     15     735    2.933
```

intrinsics 는 상수: `K[0] = [[666.6699829101562,0,480],[0,666.6699829101562,270],[0,0,1]]`,
프레임별 std 가 fx,fy,cx,cy 전부 **0.0** → `intr_norm: rel` 이면 `cam_param[9:11] = [1,1]`.

`mean‖t‖/divisor`: **geomean 0.0945**, med 0.0969, p05 0.0390, p95 0.2000, log10std 0.2377.
→ **DL3DV 0.4367 의 1/4.6**, SD 0.0878 / DataDoP 0.0911 과 사실상 같은 레벨.

### 검증

- pose 는 `poses_aNN.npz` (TRUMANS world) 에서 온다. 재앵커된 `cameras.npz` 가 아니다.
  cross-clip sim3 residual **정확히 0**.
- 누수 probe (130 segment): target-clip 뷰 130/780 (16.7 %), seg 평균 1.00 / 최대 1, 0개인 seg 없음.
  `view0 == s` True, non-`s` in-clip 뷰 **0**, 6 뷰가 항상 6개 서로 다른 clip 에 떨어짐 (mean/min/max 6.00).
  → **의도치 않은 누수 0.**
- preset → GT 궤적 방향 일치 (OpenGL c2w 카메라 로컬 좌표, +x right / +y up / +z backward):

  | preset | n | mean(x,y,z) per-frame | x 부호 일치 | z 부호 일치 |
  |---|---|---|---|---|
  | arc_left | 16 | −0.0081 +0.0002 +0.0001 | 0.88 | 0.62 |
  | arc_right | 14 | +0.0100 +0.0011 −0.0021 | **1.00** | 0.64 |
  | arc_pull | 12 | +0.0067 −0.0014 +0.0084 | 0.92 | **1.00** |
  | arc_push | 23 | −0.0056 +0.0025 −0.0077 | 0.91 | 0.96 |
  | pull_out | 9 | +0.0003 −0.0038 +0.0118 | 0.67 | **1.00** |
  | push_in | 23 | −0.0009 +0.0042 −0.0069 | 0.57 | 0.87 |
  | drift | 13 | −0.0019 +0.0054 −0.0057 | 0.69 | 0.85 |
  | hold | 20 | +0.0001 +0.0003 +0.0006 | 0.55 | 0.65 |

  arc_left → 왼쪽, arc_right → 오른쪽, arc_pull → 뒤, push_in → 앞, hold → 정지.
  **좌표계·부호 정상.**

### divisor (avg_scale)

기본은 `avg_scale_context_first_cam/` = **recording 단위 median**.
clip 단위 값(누수 있음)은 `avg_scale/` 에 남겨서 ablation 용으로 쓴다.
recording 안에서 clip 간 산포 1.46–2.17×. 이 분모는 방 크기가 아니라 **카메라–subject 거리**를 따라간다.

---

## 6. 학습 2 arm

| | arm A | arm B |
|---|---|---|
| exp | `trumans_lite_ctxuniform` | `mix_dl3dv_trumans_v1` |
| 데이터 | TRUMANS 단독 | DL3DV + TRUMANS |
| screen / GPU | train1 / GPU1 | train2 / GPU2 |
| wandb | `am3tzbk7` (`20260823_190805_...`) | `vk0kozz1` (`20260823_190317_...`) |
| index | 130 samples / 7 scenes, train 116 / val 14 | total 32807, train 29530 / val 3277 |
| 배치 | 14 it/epoch × 600 ep ≈ 8.4 k step | 3908 batch/epoch, 1.73 it/s ≈ 38 min/ep ≈ 63 h |
| smoke `val/loss_traj` @ ep0 | 2.830574 | 0.490576 |

arm B 의 코퍼스 가중치 **16.0** 유도 (T=2 temperature, `p ∝ √n`):

```
sqrt(29414) = 171.5,  sqrt(116) = 10.77  →  5.91 %
3676 × 0.0591 / 0.9409 = 231
231 × 8 / 116 = 15.9  →  16.0
per corpus [3676, 232] = 5.94 %
val 408 batch (interleave=True) per corpus [407, 1]
```

### 알고 감수한 비대칭 3가지

1. DL3DV context 범위는 `[0,s)` 인과적인데 TRUMANS `context_uniform` 은 `[0,N)` 이다.
2. divisor 정의가 다르다 (da3 점군 거리 vs 렌더 depth recording median). 레벨이 4.6× 차이.
3. `vae_latent_scale` 을 DL3DV 값 `0.96032625` 그대로 뒀다 — **재측정 안 함.**
   → **§7 에서 이게 문제로 확인됐다.**

### 부수 수정

`prdc.py` `ManifoldMetrics.compute`: split 하나가 `manifold_k+1 = 4` 개보다 작으면
`topk` 가 `selected index k out of range` 로 죽고 **`metrics.json` 자체가 안 나온다**
(FCD·caption 까지 같이 날아간다). TRUMANS val 14 → `chunk(5)` = 3,3,3,3,2.
→ `eff_splits = max(1, min(num_splits, n // (k+1)))`. 실행 로그:
`[prdc] N=14 이라 num_splits 5 -> 3 (split 당 최소 4 개 필요)`.

**⚠ arm A 의 PRDC 는 3-split 이라 5-split arm 들과 비교 불가.**
wandb `cmeb70yj` 는 폐기, `am3tzbk7` 이 진짜 arm A.

---

## 7. caption precision/recall/f1 이 안 오르는 이유 (2026-08-23 진단)

### 결론: **input / target / 좌표계 버그는 없다.** 원인은 다른 두 가지다.

#### 검증한 것 (전부 정상)

- 덤프된 `_transforms_ref.json` 이 디스크의 `pose.npz` 를 w2c → c2w → OpenGL flip 한 것과
  **자릿수까지 일치** (`|dt| mean [0.00654 0.00462 0.00803]` 양쪽 동일).
  `out_to_trajectory` 가 스케일을 정확히 복원한다 = **정규화된 값이 아니라 raw TRUMANS 미터.**
- GT caption segment 가 생성 preset 과 의미상 일치 (§5 표). 축 부호 정상.
- intrinsics 상수 → `cam_param[9:11] = [1,1]` 정상.

#### 원인 ① caption metric 의 static 임계가 **절대 world 단위**라 TRUMANS 는 경계선 위에 앉는다

`CaptionMetrics` (`metrics/modules/caption.py:405-410`): `fps = 5`,
`t_velocities = 5 · Δt_local`, `cam_static_threshold = 0.02`
→ 한 축이 "움직인다"고 찍히려면 **프레임당 로컬 이동이 0.004 world unit** 을 넘어야 한다.

| 코퍼스 | ref median `5·|Δt|` (x, y, z) | 임계 0.02 미만인 축 비율 |
|---|---|---|
| DL3DV (`SMOKE_da3geo`, n=40) | `[0.4291, 0.0805, 0.2095]` | **3.3 %** |
| mix arm B (n=160) | `[0.3148, 0.0647, 0.1398]` | **2.7 %** |
| **TRUMANS arm A (n=14)** | `[0.0237, 0.0109, 0.0372]` | **45.2 %** |

클래스는 `27 × 7 = 189` 개 **완전일치**에 `average="weighted"` 다. 부분 점수가 없다.
TRUMANS 는 절반 가까운 축이 임계 바로 위/아래라, 아주 작은 크기 오차로도 static 비트가 뒤집히고
클래스가 통째로 바뀐다 → weighted P/R/F 가 0 으로 떨어진다.

메모리 `caption-metric-thresholds-are-world-units` 에 적힌 DataDoP fscore 정확히 0 과 같은 현상이다
(TRUMANS `mean‖t‖/div` geomean 0.0945 ≈ DataDoP 0.0911).

#### 원인 ② `vae_latent_scale` 이 TRUMANS 에 대해 **5× 틀렸다** — 예측 궤적이 떨린다

`scripts/vae/vae_scale_matrix.py` 를 TRUMANS 전량(130 sample)에 돌린 실측:

```
DATASET=dl3dv  META=meta_trumans.csv  MAX_SCENES=None
ckpt                       dim scale_mode          intr  lat.std /0.96033 /0.44677      rot    trans  intr_L1
vae_20260302_300            64 avg_scale           rel   0.19128   0.1992   0.4282  0.00286  0.00174  0.00169
vae_20260302_300            64 avg_scale           raw   0.18415   0.1918   0.4122  0.00557  0.01113  0.26964
```

- TRUMANS latent std = **0.19128**, config 는 `vae_latent_scale = 0.96032625`
  → diffusion 입력 std = **0.1992** (unit 이 아니라 1/5).
- 같은 triple 의 DL3DV-only 실측은 0.47637 → 입력 std **0.4960**. 즉 arm A 는 DL3DV arm 대비
  **추가로 2.49× 더 작게** 들어간다.
- VAE 자체는 in-distribution 이다 (recon L1 trans 0.00174, rot 0.00286 — DL3DV 0.00336 보다 낫다).
  **상수만 틀렸다.**

결과 (epoch 95, step 1330, val 14편):

```
              path_len   net_disp   tortuosity
ref  med        0.5723     0.5213       1.08
pred med        2.7048     1.0987       2.54

pred/ref  프레임당 속도 비 (median, 축별)  [11.99, 31.37, 12.75]
```

**프레임당 속도는 12–31× 큰데 net 이동은 2.6× 밖에 안 크다** = 고주파 떨림(jitter).
노이즈 스케줄이 unit-std 를 가정하는데 신호가 0.199 std 라 출력이 노이즈에 잠긴 전형적 모양이다.
그리고 이 떨림이 **모든 축을 static 임계 위로 밀어올려서** 원인 ①과 곱해진다.

#### 모델은 학습되고 있다 (caption 만 안 움직인다)

```
val/loss_traj  2.830574 (ep0) → 0.171566 (ep19) → 0.143510 (ep26) → 0.120475 (ep49) → 0.116155 (ep52)
val/clatr/clatr_score  0 → 3.4 (step 266) → 9.6 (step 756) → 13.08 (step 1302)
val/clatr/fcd          1390.85 (step 14) → 1750.42 (step 252, 최악) → 1163.94 (step 1246)
val/captions/fscore    0 / 0.0017 / 0 / 0.0309 / 0 / ... / 0.0911 (step 1246) / 0 / 0.0510 / 0.0813
```

caption 은 **평평한 게 아니라 0 ~ 0.09 사이를 무작위로 튄다.** 추세가 없다.
loss·CLaTr score·FCD 는 개선되는데 caption 만 안 움직인다는 게 위 두 원인과 정확히 일치한다.

#### 고칠 수 있는 것

| | 무엇 | 대가 |
|---|---|---|
| A | `vae_latent_scale` 을 **0.19128** 로 (입력 std = 1.0) | arm A 재시작. DL3DV arm(0.496)과 입력 std 가 달라 절대 비교는 되지만 paired 비교는 아님 |
| B | `vae_latent_scale` 을 **0.38565** 로 (입력 std = 0.4960, DL3DV arm 과 동일) | arm A 재시작. DL3DV arm 과 paired 비교 성립, 절대 품질은 A 보다 낮음 |
| C | caption metric 의 `cam_static_threshold` 를 코퍼스별로 (TRUMANS ≈ 0.0011 이면 DL3DV 와 같은 여유) | 코드 변경. **DL3DV 히스토리와 수치 비교 불가** — 새 축으로 따로 봐야 함 |

A/B 는 원인 ②만, C 는 원인 ①만 고친다. **둘 다 해야 caption 이 의미 있는 신호가 된다.**
arm B(mix) 도 TRUMANS 쪽 절반은 같은 문제를 갖는데, 공유 상수라 A/B 로는 못 고친다
(코퍼스별 `vae_latent_scale` 이 필요하다).

---

## 8. 아직 안 한 것

- D68 옵션 ③ — `pan` 을 별도 `pan_deg` 사다리로. 지금 dedup 뱅크에 `translation_degenerate` 2,300 (21.8 %) 이 남아 있다.
- #68 `judge_bank_vlm.py` — VLM 가림·구도 사후 판정 열.
- #72 — 최종 카메라 전량 depth 렌더 + Vista4D 생성 5편.
  **"이 카메라들이 실제로 쓸 만한가"를 재는 건 이것뿐이다.** hole/τ 는 대리지표다.
