# goals.md

이 프로젝트가 **무엇을 향해 가고 있는지**를 적는다. 코드 구조는 `SPECS.md`, 변경 이력은
각 서브리포의 `CHANGELOG.md`, 실험 기록은 `EXPERIMENTS.log` 에 있다. 여기에는 *목표*만 쓴다.

작성 기준 시점: 2026-09-04. 현황 수치는 그 시점의 실측이고, 목표 자체가 바뀌지 않는 한
수치만 갱신한다.

---

## 최종 목표

**context dynamic video + high-level intent text → camera → re-rendering.**

- 입력 ①: **context dynamic video** — 움직이는 피사체가 있는 실제 영상 한 편.
- 입력 ②: **high-level intent text** — "이 사람을 왼쪽에서 돌면서 잡아줘" 수준의 의도.
  프레임별 pose 를 받는 게 아니라 *의도*를 받는다.
- 출력: 그 영상의 씬과 정합되는 **카메라 궤적**, 그리고 그 카메라로 다시 렌더한 **영상**.

즉 "카메라를 텍스트로 조종하는 video model" 이 아니라, **씬을 이해하고 그 씬 위에서 촬영
계획을 세우는 카메라 생성 모델 + 그 계획을 실행하는 렌더링/생성 단계**로 나눈다. 카메라가
1급 시민(first-class output)이라는 것이 이 과제의 전제다.

---

## 단기 목표 — 합성 GT 카메라로 scene-aligned 카메라 생성 학습

**지금 하고 있는 일.** 실제로 존재하는 "dynamic 영상 ↔ 그 영상에 맞는 카메라" 쌍이 없으므로,
**합성 GT 카메라를 우리가 만든다.** 그리고 그 위에서 *영상에 보이는 씬과 align 된* 카메라를
생성하도록 학습한다.

### 1. 합성 GT 카메라 뱅크

영상 1편 → scene graph(중력축·단위 S·인스턴스 OBB·track) → 후보 카메라 샘플링 → hole 사다리
fit → emit. 카메라가 벽 속에 들어가거나 씬을 못 보는 경우를 **텍스트가 아니라 렌더 게이트로**
걸러낸다는 것이 이 뱅크의 핵심이다.

소스 두 계열:

| 계열 | 소스 | 현황 (2026-09-08) |
|---|---|---|
| **in-the-wild video** | Vista4D 계열 dynamic 영상 | d121 뱅크 완료 — 52 scenes, train 14,975 / test 875 세그먼트 |
| **in-the-wild video (대규모)** | dynpose 코퍼스 | **D157 재생성 중** — 875 scenes 4샤드, 382편 완료 (scene 당 카메라 mean 144.6) |
| **TRUMANS** | 합성 human motion 씬, mesh GT (중력·지면·피사체 전부 GT) | preset 렌더 진행 중. keyframe N개를 LBM 에 그대로 먹일 수 있는 상태 |

#### 다음 라운드의 방향 — **scene 을 늘리고 preset 을 줄인다** (D157 완료 후 착수)

지금까지는 scene 수를 고정한 채 scene 당 카메라를 늘려 왔다. D157 이 그 축의 끝이다 —
875 scenes × mean 144.6 = 약 126.5k 카메라. 다음 라운드는 **축을 바꾼다**:

- **scene 을 10k 단위로 올린다.** 코퍼스의 다양성은 scene 이 지고 있고, 같은 scene 안의
  144개 변이는 서로 강하게 상관되어 있다. 학습 신호가 늘어나는 방향은 scene 쪽이다.
- **preset 개수를 대폭 줄인다.** 위를 하려면 scene 당 비용을 줄여야 한다. 실측상 scene 당
  시간의 **78% 가 fit 단계**이고, fit 시간은 preset 상위 5종이 60% 를 먹는다. preset 을
  깎는 것이 scene 수를 10배로 올릴 유일한 예산이다.

구체적 손잡이(K1 preset 축소 / K2 사다리 2·4단만 / K3 scene 당 dd 하나)와 실측 근거는
`camera_generation/dataset/DECISIONS.md` 의 "차기 뱅크(d157 후속)" 절에 있다.
여기에는 목표만 적는다 — **scene ↑ (10k), preset ↓.**

### 2. film 유래 카메라 motion (DataDoP 계열)

DataDoP 처럼 **실제 영화에서 뽑은 카메라 motion** 을 코퍼스에 섞는다. 다만 film 카메라는
**intent 를 온전히 annotate 하기 어렵다** — 왜 그렇게 움직였는지, 무엇을 겨냥했는지가
영상만으로는 복원되지 않는다. 그래서 GenDoP 과 같은 처리를 한다: **target 절 없이
free-moving 으로만 넣는다.**

현황: D122 뱅크에서 `dd_*` (DataDoP 유래) 변이가 10,876 / 21,913 = **49.6%**. 비율은
의도적으로 맞춘 값이고, 이번 라운드에서 preset routing 과 개수만 조정했다.

### 3. 학습

- arm 1 `vista4d_d121_da3` — DA3 geo encoder 만 (video CA 없음). 대조군.
- arm 2 `vista4d_d121_peav` — text/video encoder 둘 다 PE-AV.

두 arm 은 데이터·하이퍼가 전부 같고 **인코더만 다르다**. 차이가 인코더 차이로만 남아야 한다.

### 단기 목표의 완료 조건

씬을 보고 (a) 피사체를 프레임 안에 유지하고 (b) 벽/물체를 통과하지 않고 (c) 하류 video model 이
메워야 할 hole 이 감당 가능한 카메라를, **텍스트 의도에 맞게** 낸다.

---

## 현재 목표 목록 (사용자 지시 2026-09-27 "목표에 몇가지 적어두자")

사용자가 적은 여섯 항목을 순서 그대로 둔다. 관련 요청·실험 번호는 진행하면서 옆에 붙인다.

1. **DA3, Molmo2 가 실제로 도움이 되는지 판단.** encoder ablation — testset(5,144 seg) 비교표에
   d200 da3 / D274 da3 v12 / d262 molmo2_l21_srccam / D269 siglip2_srccam / d268 umt5_srccam 까지 있음.
   진행 중·대기: R54 da3 v24, R58 siglip2 + connector, R59 da3 unposed (EXPERIMENTS.log, tasks A5).
2. **깔끔한 3인칭만 모아서 10k vid 데이터셋을 만들어 학습.** D282 VLM+SAM3 필터(DynPose-100K 9,396 pass) →
   D284 확장(recon·seg·graph·bank) → d200 과 합친 새 코퍼스.
3. **첫 카메라 생성도 포함한 데이터셋을 만들어 학습.** start pose sampling (R62: 4방위 x 3고도 x 3 shot scale
   + source 카메라, 시작 카메라 선판정 `--start_screen`) → 가능한 후보 중 샘플링 → 학습 (tasks A7).
4. **거리가 가까운 씬에서 dolly in 추론 시 앞뒤로 shaking 만 하는 문제 개선** — 부드럽고 느리게 앞으로
   가야 한다.
5. **빠르게 움직이는 물체의 track 및 framing 이 잘 안 되는 문제 개선.**
6. **TRUMANS 및 기타 synthetic dataset 을 같이 학습해 성능 향상.** TRUMANS board source 3,440 clip →
   d277T bank (진행 중) → caption → export.

---

## 이후 목표 (중기) — 표현력과 안정성

단기 목표가 "말이 되는 카메라"라면, 여기는 "**좋은** 카메라"다.

### 1. 더 안정적이고 씬과 더 잘 align 된 합성

지금 뱅크의 약한 지점은 측정되어 있다 (D122 실측, 243 scenes / 21,428 variants):

- `solved` 36.6%, `clamped_low + tau_floor` 15.4% — 상당수가 사다리 하한에 눌려 있다.
- 길이 0 궤적 1,565 (7.3%). 이 중 1,368 은 pan/tilt(정상), 게이트에 뭉개진 건 197 (0.9%).
- `subject_in_frame < 0.85` 가 41.9% — 피사체가 프레임에서 자주 벗어난다.
- `track_truck_left/right` 1,552 변이(7.2%)는 `aim=free` 라 **따라가되 다시 겨냥하지 않는다**.
  캡션은 "tracks alongside {target}" 인데 `subject_in_frame` median 은 0.462 / 0.385 —
  캡션과 기하가 어긋난 대표 사례.

즉 여기서 고칠 것은 "더 많은 카메라"가 아니라 **캡션과 실제 궤적의 정합**, 그리고 τ 하한에
눌린 변이들의 회수다.

### 2. 두 가지 이상이 복합된 카메라

현재 샘플러 축은 `anchor × preset(D157 기준 anchor 당 14종) × τ ladder × speed/tracking/look_at_bias`
이고,
**한 변이는 primitive 를 정확히 하나만 쓴다.** 실제 촬영은 그렇지 않다 — dolly 하면서 orbit
하고, 도중에 target 을 바꾼다. 목표는 **primitive 2개 이상의 조합**을 GT 로 합성하고, 그
조합을 설명하는 캡션까지 같이 만드는 것.

관련 구조는 이미 있다: `traj.py:172 compose`, 그리고 keyframe 리스트를 받는 디코더
(`decision.json` 의 `keyframes` 를 처음부터 리스트로 둔 이유). LBM 도 N개 keyframe 을 입력으로
받는다. **막힌 건 표현이 아니라 "어떤 조합을 왜 골랐는지"의 라벨링이다.**

### 3. 다양한 시작 구도 — **지금 안 하고 있는 것**

현재 D122 는 `sample_camera_bank.py --start_mode` 기본값 `source_frame0`, `--aim_anchor` 기본값
`subject`, `decode/emit.py` 는 `--no_free_start` 기본이다. 결과적으로 **합성된 target 카메라는
전부 소스 frame0 카메라와 같은 자리에서 시작한다** (`rel[0] = I`).

이걸 풀면 — 시작 pose 자체를 다양한 구도(높이·방위·shot scale)에서 뽑아두면 —

- 카메라 생성 모델이 **더 dynamic 한 구도**를 잡을 수 있고,
- 소스 카메라가 어디 있든 **더 다양한 상황에 적응**할 수 있다.

단, 이건 공짜가 아니다. 시작 pose 를 소스에서 떼면 frame0 부터 hole 이 생기고, `rel[0]=I` 를
가정하는 하류 모델(ReCamMaster 는 상수 offset 을 아예 무시한다)과의 계약이 깨진다. **시작 구도
다양화는 "어느 하류 모델을 대상으로 하는가"와 묶어서 결정해야 한다.**

---

## 장기 목표

### 1. 카메라의 연속 생성 — world 를 확장하면서

지금은 49프레임 한 덩어리를 한 번에 낸다. 장기적으로는 **카메라를 이어서 생성**한다:

- 긴 영상을 받아도, 짧은 영상을 받아도,
- **world model 처럼 world 를 계속 확장하면서** 그 확장된 world 위에서 카메라를 같이 생성한다.

즉 카메라 생성과 씬 확장이 서로를 먹인다. 새로 생성된 카메라가 아직 관측되지 않은 영역을
보게 되고, 그 영역이 채워지면 다음 구간의 카메라가 다시 그 위에서 계획된다. 지금의
"관측된 표면만 있는 depth shell" 한계가 여기서 자연스럽게 풀린다.

### 2. 피사체와의 joint generation

카메라만 생성하는 게 아니라 **피사체인 human, 그리고 human action 을 함께 생성**하는 쪽으로
확장한다. 촬영은 원래 피사체와 카메라가 서로를 규정하는 일이다 — 배우가 어디로 움직일지가
카메라 동선을 정하고, 카메라 위치가 연기의 방향을 정한다. 이를 따로 푸는 대신 joint 로 푼다.

TRUMANS 를 쓰는 이유가 여기에도 걸린다: human motion 이 GT 로 있는 유일한 소스라
"카메라 ↔ human action" 쌍을 만들 수 있다.

---

## 목표 간의 의존 관계

```
단기 (합성 GT + scene-aligned 학습)
  └─ 중기 1 (정합/안정성)  ─┐
  └─ 중기 2 (복합 카메라)  ─┼─→ 최종 (video + intent → camera → re-render)
  └─ 중기 3 (시작 구도)    ─┘
                              └─→ 장기 1 (연속 생성 / world 확장)
                              └─→ 장기 2 (human joint generation)
```

중기 3(시작 구도)은 중기 2(복합 카메라)와 곱해진다 — 시작 구도가 다양해야 조합의 의미가 산다.
장기 1은 중기 1(정합)이 선행되어야 한다: 구간을 이어붙이면 정합 오차도 같이 누적된다.
