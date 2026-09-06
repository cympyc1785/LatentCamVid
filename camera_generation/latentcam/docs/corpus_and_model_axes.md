# 데이터축(vista / dynpose) × 모델축(da3 / molmo2 / da3+molmo2) 분석 — D142

작성 2026-09-06. 사용자 지시: *"데이터 측면에서 vista(scene 적고 preset 많음), dynpose(scene 많고
preset 적음) 을 비교하고 모델 측면에서 da3 vs molmo2 vs da3+molmo2 를 비교하는 걸 목표로 어떤
조합이 더 효과적으로 카메라를 생성할 수 있는 방식일지 goals.md 에 맞춰서 (…) 데이터셋이나 모델
개선해야 할 사항들 있으면 다방면으로 분석해서 정리"*.

수치는 전부 실측 원본이다. 반올림·가공하지 않았다. 아직 안 끝난 학습의 칸은 **비어 있는 채로**
둔다 — 채워진 칸만 가지고 결론을 당기지 않는다.

재현:

```bash
python scripts/eval/corpus_axis_compare.py \
  --corpus vista_d121=/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121:vista4d \
  --corpus vista_d128=/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d128:vista4d \
  --corpus dynpose_d137=/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d137:dynpose_dd10 \
  --split train --out results/compare/corpus_axis
# -> results/compare/corpus_axis/corpus_axis_train.json (전량 카운트)
```

---

## 0. 2×3 그리드 현황

|  | da3 (geo only) | molmo2 (nogeo) | da3 + molmo2 |
|---|---|---|---|
| **vista d121** | D123 `ol2mue8s` ✅ 완료 | D133 `jvonk95o` ⏳ GPU 0 / train3 | D124 `w5ygw8ii` ✅ 완료 |
| **dynpose d137** | D137 `uj2ahp5e` ⏳ GPU 3 / train2 | D138 `r1lnh0mt` ⏳ GPU 2 / train4 | D141 `0zb4koqs` ⏳ GPU 1 / train1 |

grid 밖: **D131** `8kg0j4pb` (vista **d128** / da3) — 코퍼스가 달라 위 표와 같은 칸이 아니다.

전 arm 공통 고정: `pose_source=da3`, `target_pose_source=da3_target_poses`,
`scale_mode=avg_scale`, `avg_scale_ref=context_first_cam`, `intr_norm=rel`, `cam_dim=64`,
`text_len=128`, `anchor_pred_frame0=true`, `batch_size=8`, `epoch_cap=100`, seed 동일.
바뀌는 건 코퍼스 경로와 `geo_encoder` / `peav_*` 블록뿐이다.

**ckpt 주의**: D123 / D124 는 `epoch100.pth`(= 101 epoch 상태)에서 평가됐고, 이후 arm 들은
`epoch_cap: 100` 이라 `last.pth`(= 100 epoch 상태)뿐이다. 두 칸을 나란히 놓을 때 이 1 epoch
차이를 표에 명시한다.

---

## 1. 모델축 — 지금까지 채워진 칸

### 1-a. vista d121 test 875 — **모델축 3칸 완료**

| metric | D123 da3 (`epoch100`) | D133 molmo2 (`last`) | D124 da3+molmo2 (`epoch100`) |
|---|---|---|---|
| val/loss_latent | 0.06864738377715861 | 0.0637369708695582 | 0.06397023958393505 |
| val/loss_traj | 0.010284261889556157 | 0.009883241927783405 | 0.009815332671627403 |
| captions/precision | 0.5378 | 0.5698 | 0.5629 |
| captions/recall | 0.5163 | 0.5544 | 0.5465 |
| captions/fscore | 0.5214 | 0.5529 | 0.5465 |
| clatr/clatr_score | 23.3159 | 25.0879 | 24.4752 |
| clatr/precision | 0.5383 | 0.5543 | 0.5954 |
| clatr/recall | 0.7246 | 0.7429 | 0.7177 |
| clatr/density | 0.4644 | 0.5109 | 0.5520 |
| clatr/coverage | 0.5131 | 0.5383 | 0.5417 |
| clatr/fcd | 159.8566 | **108.1954** | 123.4501 |
| sampling_sec | 206.6 | 112.2 | 323.4 |

전 칸 `n_samples: 875`, seed 42, ddim 50 step. **ckpt 기준이 섞여 있다** — D123/D124 는
`epoch100.pth`(101 epoch 상태), D133 은 `epoch_cap: 100` 이라 `last.pth`(100 epoch)뿐이다.

읽기:

- **molmo2 단독(D133)이 da3 단독(D123)을 11지표 전부에서 이긴다.** loss_traj 0.009883 vs
  0.010284, fscore 0.5529 vs 0.5214, fcd 108.1954 vs 159.8566.
- **molmo2 단독이 da3+molmo2(D124)도 captions 3개 · clatr_score · recall · fcd 에서 이긴다.**
  D124 가 이기는 건 clatr/precision(0.5954 vs 0.5543) · density(0.5520 vs 0.5109) ·
  coverage(0.5417 vs 0.5383) · loss_traj(0.009815 vs 0.009883) 넷.
- 즉 §1-a 의 D123→D124 이득은 **"molmo2 를 얹어서"가 아니라 상당 부분 "molmo2 로 갈아타서"**
  다. da3 geo 스트림을 통째로 빼도 성능이 안 떨어지고 오히려 캡션 정합·fcd 는 좋아진다.
  샘플링도 112.2 s 로 D124(323.4 s)의 1/2.9, D123(206.6 s)의 1/1.8.
- 단서: D124 의 video CA 실사용량 계측(D171)에서 `dpred_drop 0.02395` vs
  `dpred_xscene 0.00462` — 영상을 빼면 예측이 움직이는데 **다른 씬 영상으로 바꿔치면 거의 안
  움직인다.** D133 이 D123 을 이겼다는 것이 "molmo2 가 씬을 읽는다"를 증명하지는 않는다.
  molmo2 스트림이 나르는 게 씬인지 캡션의 재표현인지는 아직 안 갈렸다 (§4-5).
- dynpose 축(D137/D138/D141)에서 같은 순서가 재현되는지가 다음 확인점이다.

### 1-b. 코퍼스 크기 대조 — d128 test 592, leakage 확인 완료

d121-train ∩ d128-test = ∅ (씬·세그먼트 양쪽). 두 코퍼스 모두 같은 4씬
{avocado-slice, bmx-bumps, camel, couple-hug} 을 holdout 한다.

| metric | D131 (d128 학습, `last.pth`) | D123 (d121 학습, `epoch100.pth`) |
|---|---|---|
| val/loss_latent | 0.07201048717016002 | 0.06636428748056092 |
| val/loss_traj | 0.01140013592069138 | 0.010368599051985625 |
| captions/precision | 0.4859 | 0.4874 |
| captions/recall | 0.4441 | 0.4655 |
| captions/fscore | 0.4481 | 0.4707 |
| clatr/clatr_score | 25.06 | 25.4694 |
| clatr/precision | 0.5720 | 0.5855 |
| clatr/recall | 0.7622 | 0.7840 |
| clatr/density | 0.5126 | 0.5254 |
| clatr/coverage | 0.5401 | 0.5152 |
| clatr/fcd | 149.2309 | 174.3708 |

**d128 자기 테스트셋에서 d121 학습본이 9지표 중 7개를 이긴다.** D131 이 이기는 건
`coverage`(0.5401 vs 0.5152)와 `fcd`(149.2309 vs 174.3708) 둘.

교란: d121 train 14,975 seg vs d128 train 9,389 seg = **1.6배**. 즉 이 결과는 "d128 뱅크 개선이
무의미하다"가 아니라 **"현 규모대에서는 뱅크 품질 개선분보다 세그먼트 수가 더 세다"** 로 읽어야
한다. 이 해석은 §2 의 코퍼스 측정과 붙는다.

---

## 2. 데이터축 — 학습이 실제로 읽는 단위로 잰 코퍼스

`train` split, `<scene>/da3/prompts.json` 기준 (= `pose_source: da3` arm 이 읽는 파일).

### 2-a. 규모와 다양성

|  | vista_d121 | vista_d128 | dynpose_d137 |
|---|---|---|---|
| seg (학습 대상) | 14975 | 9389 | 9728 |
| scene | 48 | 48 | 240 |
| seg/scene med | 285 | 201 | 38 |
| uniq preset (전체) | 40 | 40 | 202 |
| uniq preset/scene med | 40 | 23 | 12 |
| uniq anchor_label | 78 | 54 | 129 |

**"scene 적고 preset 많음 / scene 많고 preset 적음" 은 씬 축에서는 맞고 preset 축에서는 그대로
읽으면 틀린다.** 전체 uniq preset 은 dynpose 가 202 로 vista 40 보다 많다. 계열로 쪼개면 뒤집힌다:

| preset 계열 | vista_d121 (uniq/seg/%) | vista_d128 | dynpose_d137 |
|---|---|---|---|
| `dd_*` (DataDoP 유래) | 0 / 0 / 0.0% | 0 / 0 / 0.0% | **181 / 973 / 10.0%** |
| `track_*` | 17 / 2972 / 19.8% | 17 / 1292 / 13.8% | 9 / 4525 / 46.5% |
| 그 외 LBM preset | 23 / 12003 / 80.2% | 23 / 8097 / 86.2% | 12 / 4230 / 43.5% |

`dd_*` 181종은 세그먼트의 10.0% 만 덮는 **one-off** 다 (한 라벨당 평균 5.4 seg). 이걸 빼면 실효
어휘는 vista 40 vs dynpose **21**, 씬당 중앙값은 40/23 vs **12**. 사용자 framing 이 맞다.

preset 조성 자체도 크게 다르다:

| preset | vista_d121 | vista_d128 | dynpose_d137 |
|---|---|---|---|
| dolly_out | 7.0% (1043) | 6.9% (651) | **17.6% (1715)** |
| truck_left | 6.7% (997) | 6.8% (634) | 1.2% (114) |
| truck_right | 6.5% (975) | 6.6% (616) | 1.8% (176) |
| pedestal_up | 6.2% (931) | 5.9% (552) | **0.0% (0)** |
| pedestal_down | 6.0% (900) | 6.3% (592) | **0.0% (0)** |
| crane_up | 3.0% (456) | 3.3% (311) | **0.0% (0)** |
| crane_down | 3.4% (511) | 3.9% (368) | **0.0% (0)** |
| orbit_left | 4.0% (600) | 4.2% (396) | 0.1% (7) |
| orbit_right | 3.7% (547) | 3.9% (367) | 0.2% (22) |
| orbit_left_pedestal_up | 3.7% (555) | 4.0% (377) | **0.0% (0)** |
| track_dolly_out | 1.8% (271) | 1.3% (124) | 9.9% (963) |
| track_truck_left | 1.7% (260) | 1.2% (110) | 7.6% (737) |

**수직 이동(`pedestal_*` / `crane_*` / `orbit_*_pedestal_up`)은 vista 에서 19.4%, dynpose 에서
0.0%.** dynpose 만으로 학습하면 "카메라를 올리면서 잡아줘" 류 의도에 대응하는 GT 가 **한 건도
없다**. 반대로 dynpose 는 `track_*` 이 46.5% 로 추종 shot 이 압도적이다. 두 코퍼스는 난이도가
다른 게 아니라 **덮는 카메라 문법이 다르다** — 합치는 쪽이 자연스러운 이유다(§4-1).

### 2-b. 게이트 품질

p10 / med / p90:

|  | vista_d121 | vista_d128 | dynpose_d137 |
|---|---|---|---|
| hole_fraction | 0.097/0.329/0.548 | 0.096/0.319/0.532 | 0.097/0.331/0.576 |
| tau_max | 0.075/0.472/2.241 | 0.064/0.464/2.241 | 0.202/0.844/2.808 |
| subject_in_frame | 0.231/1.000/1.000 | 0.308/1.000/1.000 | 0.231/1.000/1.000 |
| subject_area_med | 0.002/0.034/0.169 | 0.002/0.051/0.204 | 0.000/0.026/0.165 |
| subject_visible_frac | 0.309/0.932/0.999 | 0.398/0.954/0.999 | 0.000/0.958/0.999 |

median 은 세 코퍼스가 비슷한데 **꼬리 두께가 다르다**:

| 조건 | vista_d121 | vista_d128 | dynpose_d137 |
|---|---|---|---|
| subject_visible_frac < 0.05 | 1048 (7.0%) | 670 (7.1%) | **1611 (16.6%)** |
| subject_visible_frac < 0.30 | 1476 (9.9%) | 830 (8.8%) | **1788 (18.4%)** |
| subject_area_med < 0.005 | 2018 (13.5%) | 1284 (13.7%) | **2400 (24.7%)** |
| subject_in_frame < 0.50 | 2516 (16.8%) | 1476 (15.7%) | **2362 (24.3%)** |
| subject_in_frame < 0.85 | 4073 (27.2%) | 2480 (26.4%) | **4023 (41.4%)** |
| hole_fraction > 0.55 | 1451 (9.7%) | 746 (7.9%) | **1328 (13.7%)** |
| tau_max > 1.5 | 2291 (15.3%) | 1561 (16.6%) | **2747 (28.2%)** |

`subject_in_frame < 0.85` 41.4% 는 goals.md 중기목표 1 이 D122 에서 적어둔 41.9% 와 사실상 같다
— **`clamped_low` 를 빼도(D137 재생성) 이 지표는 안 움직였다.** τ 하한 문제와 프레이밍 문제는
서로 다른 원인이라는 뜻이다.

### 2-c. `track_*` × `aim` — 캡션과 기하가 어긋나는 곳

`aim` 은 **회전 조준**, `track_*` 은 **병진 추종**이다. 둘은 독립이라 `track_*` + `aim=free` 는
피사체를 따라 움직이되 **다시 겨냥하지 않는다**.

|  | vista_d121 | vista_d128 | dynpose_d137 |
|---|---|---|---|
| `track_*` aim=free n | 1415 | 609 | 2482 |
| └ visible<0.05 | 238 (16.8%) | 109 (17.9%) | 628 (25.3%) |
| └ in_frame<0.5 | 448 (31.7%) | 211 (34.6%) | 948 (38.2%) |
| └ 캡션이 추종 주장 | 1415 (100.0%) | 609 (100.0%) | 2482 (100.0%) |
| └ 캡션이 프레이밍 약속 | 1121 (79.2%) | 530 (87.0%) | 2067 (83.3%) |
| `track_*` aim=look_at n | 1557 | 683 | 2043 |
| └ visible<0.05 | 19 (1.2%) | 18 (2.6%) | 137 (6.7%) |
| └ in_frame<0.5 | 15 (1.0%) | 15 (2.2%) | 126 (6.2%) |
| └ 캡션이 추종 주장 | 1557 (100.0%) | 683 (100.0%) | 2043 (100.0%) |

**같은 `track_*` 인데 `aim` 하나로 `in_frame<0.5` 가 31.7% → 1.0% (vista d121),
38.2% → 6.2% (dynpose) 로 갈린다.** 그런데 캡션은 양쪽 모두 100% 추종을 주장하고, `aim=free`
쪽도 79~87% 가 "프레임 안에 유지한다"고 못박는다.

실제 예:
- `vista4d/car-roundabout/70` — preset `track_truck_left`, aim `free`, in_frame 0.231,
  visible 0.0. 캡션: *"The camera significantly trucks to the left, keeping pace with the
  movement in the scene."*
- `dynpose/01a493f2-838b-445f-9bc1-376816219e66/8` — preset `track_truck_left`, aim `free`,
  in_frame 0.231, visible 0.0. 캡션: *"The camera dramatically tracks alongside the black puffer
  jacket worn by the man holding a box while sliding to the left, keeping it in a medium close…"*

영향 규모: vista d121 1415 seg (9.5%) + dynpose d137 2482 seg (25.5%). **task #134 가 정확히
이것이고, 지금 이 문서가 그 크기를 처음 정량화한 것이다.** 사용자의 반복 지적
*"카메라가 물체를 많이 놓치네"* 의 가장 큰 단일 원인이기도 하다.

---

## 3. goals.md 와의 대조

goals.md 단기목표 완료조건: *"씬을 보고 (a) 피사체를 프레임 안에 유지하고 (b) 벽/물체를
통과하지 않고 (c) 하류 video model 이 메워야 할 hole 이 감당 가능한 카메라를, 텍스트 의도에 맞게"*.

| 조건 | 현재 코퍼스가 가르치는 것 | 판정 |
|---|---|---|
| (a) 프레임 유지 | `subject_in_frame < 0.85` 가 vista 27.2% / dynpose 41.4%. `track_*`+`aim=free` 3897 seg 은 캡션이 지킬 수 없는 약속을 한다 | **GT 자체가 (a) 를 위반한다.** 최우선 |
| (b) 통과 금지 | 렌더 게이트(G1 behind-surface)가 굽는 단계에서 집행 중. 코퍼스에 남은 위반은 측정 안 됨 | 별도 열 필요 (task #68) |
| (c) hole | `hole_fraction > 0.55` vista 9.7% / dynpose 13.7%. med 0.32~0.33 | 허용 범위, 꼬리만 정리 |
| 텍스트 의도 정합 | §2-c 가 정면 위반. captions/fscore 0.5214~0.5465 의 천장이 여기 있을 가능성 | (a) 와 같은 처방 |

goals.md 중기목표 1 이 적어둔 D122 수치(`subject_in_frame<0.85` 41.9%, `track_truck_left/right`
1552 변이가 `aim=free`)는 **D137 에서도 그대로다**. `clamped_low` 제외는 τ 하한 문제만 건드렸고
프레이밍 문제는 안 건드렸다는 게 이번 측정의 결론이다.

goals.md 중기목표 2(복합 카메라)·3(시작 구도)은 코퍼스 상태로도 확인된다 — 한 변이가 primitive
하나만 쓰고(`dd_*` 제외 실효 어휘 21~40), 시작 pose 는 전부 `rel[0]=I`.

---

## 4. 개선안 — 근거 / 처방 / 비용

우선순위 순.

### 4-1. [데이터·최우선] `track_*` + `aim=free` 모순 해소 (task #134)

- 근거: §2-c. 3897 seg (vista 1415 + dynpose 2482), `in_frame<0.5` 가 `aim` 하나로 31.7%→1.0% /
  38.2%→6.2%.
- 선택지 A — `PRESET_TRACKING` 처럼 `track_*` 에 `aim=look_at` 을 **강제**한다. D133(task #133)
  에서 snowboard 에 이미 한 조작이라 코드 경로가 있다. 비용: 재굽기, `aim` 다양성 감소.
- 선택지 B — `aim=free` 를 유지하되 **캡션에서 프레이밍 절을 뺀다**. free-moving 은 goals.md
  §2 가 DataDoP 계열에 대해 이미 채택한 방식("target 절 없이 free-moving 으로만")과 같다.
  비용: 캡션 재생성만. 재굽기 불필요.
- 권고: **B 를 기본, A 를 `track_*` 중 subject 가 계속 보이는 변이에만.** B 는 캡션 파이프라인만
  돌리면 되고, 학습 신호에서 "지킬 수 없는 약속"을 즉시 제거한다. A 는 다양성을 깎으므로 A 를
  전량 적용하면 `aim` 축이 붕괴한다.
- 검증: 재생성 후 이 문서의 §2-c 표를 그대로 재측정 (`corpus_axis_compare.py`).

**해소됨 (D143, 2026-09-06) — B 를 채택하되 판정을 preset 축이 아니라 변이 축으로.**
`build_bank_captions.py --framing_min_in_frame 0.85` (기본값): 그 변이의 실측
`subject_in_frame` 이 임계 미만이면 framing 절(그리고 `nl` 형식의 composition 절)을 뺀다.
`--no_framing_on_free` 로 preset 축(=`aim=free` 전량) 규칙도 남겼지만 기본은 꺼짐.

*왜 preset 축이 아닌가* — `aim=free` 라고 다 깨지지 않는다. vista d121 train, `aim=free` 만,
`subject_in_frame` median / `<0.85` 비율:

| preset | median | <0.85 | | preset | median | <0.85 |
|---|---|---|---|---|---|---|
| `dolly_out` | 1.000 | 19.4% | | `track_dolly_out` | 1.000 | 4.8% |
| `track_hold` | 1.000 | 10.8% | | `static_hold` | 1.000 | 20.8% |
| `pedestal_down` | 0.923 | 48.7% | | `dolly_in` | 0.923 | 49.0% |
| `truck_left` | 0.538 | 69.1% | | `track_truck_left` | 0.385 | 78.8% |

자기 축으로 물러나는 `dolly_out` 계열은 재조준 없이도 대상이 중앙에 남는다 — preset 이름으로
뭉뚱그리면 멀쩡한 신호를 버린다. 반대로 `aim=look_at` 도 2.9% 는 프레임을 놓치는데(아래 §4-1a)
preset 축으로는 그쪽을 아예 못 잡는다.

*무엇을 남기나* — motion 절과 target 은 그대로다. `track_*` 의 "tracks {target}" 은
follow_gain 1.0 으로 실제 참이고, 비-track 의 "sliding sideways past {target}" 도 참이다.
거짓인 건 프레이밍 약속뿐이라 그 절만 뺀다.

*적용 범위* — 학습을 안 돌리는 vista 두 뱅크에만 적용했다 (사용자 지시). 기존 루트는 완료된
런의 학습 캡션이라 **덮지 않고** 새 루트를 팠다. 기하(seg_list / meta_csv)는 원본과 비트 동일함을
`run_d143_caption_export.sh` 의 `check_same` 으로 확인:

| 새 루트 | 뱅크 | seg (train/test) | 프롬프트 변경 |
|---|---|---|---|
| `latentcam_da3_k6_d121c143` | `hole_bank_k6_d121` | 14975 / 875 (원본과 동일) | 3948/15850 (24.9%) |
| `latentcam_da3_k6_d128c143` | `hole_bank_k6_d128` | 9389 / 592 (원본과 동일) | 2049/9981 (20.5%) |

dynpose d137 은 **미적용** — D137/D141 이 그 코퍼스로 학습 중이다. 학습이 끝나면 같은 드라이버로.

### 4-1a. [남은 문제] dynpose 의 프레이밍 실패는 aim/track 으로 설명되지 않는다

사용자 지시("aim이 follow인 것들이나 target이 없는 free moving은 ... 당연해 이것들 제외한
preset들만 재줘")대로 `corpus_axis_compare.py --framing_scope` 로 다시 쟀다.
`subject_in_frame < 0.85` 비율, split=train:

| 범위 | vista_d121 | vista_d128 | dynpose_d137 |
|---|---|---|---|
| `all` (전량) | 27.2% | 26.4% | 41.4% |
| `aimed` (`aim==look_at`) | 2.9% | 2.8% | 17.6% |
| `aimed_nontrack` (+`track_*`/`dd_*` 제외) | 3.1% | 2.5% | 17.0% |

남는 세그먼트: 6015/14975(40.2%) · 3998/9389(42.6%) · **955/9728(9.8%)**.

**vista 는 사용자의 읽기가 그대로 맞다** — follow/targetless 를 빼면 27.2% → 3.1% 로 무너진다.
**dynpose 는 아니다** — 41.4% → 17.0% 에서 멈춘다. 즉 dynpose 에는 aim/track 이 설명하지 못하는
잔차가 따로 있고, 그건 §4-2(라우팅)·§4-4(`dd_*` one-off)와 같은 뿌리일 가능성이 크다:
`aimed_nontrack` 에서 dynpose 는 955 seg 중 `s_curve` 하나가 62.0% 를 먹고, 씬은 195개인데
seg/scene median 이 3 이다. 진단 우선순위는 §4-2 다음.

### 4-2. [데이터] dynpose 에 수직 이동 preset 이 0건

- 근거: §2-a. `pedestal_up/down` `crane_up/down` `orbit_left_pedestal_up` 이 vista 19.4%,
  dynpose **0.0%**.
- **원인 확정 (2026-09-06) — 게이트 기각이 아니라 라우팅 경로가 다르다.** 두 코퍼스는 preset 을
  고르는 방식 자체가 다르다:
  - vista(`run_k6_d128_shard.sh:69`)는 `route_presets.py` 를 **아예 안 부른다.**
    `sample_camera_bank.py` 에 `--presets` 를 안 주고, 그 기본값이
    `sample_camera_bank.py:1184  parser.add_argument("--presets", nargs="*", default=None)  # None = 전량`
    이라 **preset 어휘 전량**을 굽는다. 수직 5종이 여기 다 들어 있다.
  - dynpose(`run_dynpose_d129_shard.sh:83`)는 `route_presets.py --emit args` 로 받은
    `$ARGS`(= `--presets <슬레이트>`)를 넘긴다. 즉 씬당 라우팅된 슬레이트만 굽는다.
  - 그 슬레이트에서 수직 슬롯은 `route_presets.py:189-192` 한 곳에서만 나온다:
    ```python
    if grav == "ground_ransac":
        slots.append(("vertical", tp("vertical", "crane_up")))
    elif allow_vertical_fallback:
        slots.append(("vertical", tp("vertical", "pedestal_up")))
    ```
    `--vertical_fallback` 은 `route_presets.py:343` 에서 `default=False` 이고 dynpose 샤드가
    안 넘긴다. 그리고 dynpose 278 씬의 `gravity_method` 는 **278/278 이 `geocalib`** 이라
    첫 조건도 거짓 → `vertical_dropped: true` 가 **278/278**, 슬레이트에 수직 preset 이 있는
    씬 **0/278**. (실측: `out_dynpose/*/preset_route_d129.json`)
- **중력축 품질 문제가 아니다.** vista 도 `out/*/scene_graph.json` 기준 `geocalib` 52 / `ground_ransac` 1
  이라 **vista 를 route_presets 에 태웠으면 똑같이 0건이 나왔을 것**이다. 차이는 씬이 아니라 경로다.
- 처방(둘 중 하나, 재굽기 1회):
  - (a) dynpose 샤드에 `--vertical_fallback` 추가 → `pedestal_up` **한 슬롯**만 생긴다.
    `crane_down`/`pedestal_down`/`orbit_left_pedestal_up` 은 route 가 애초에 못 내는 이름이라
    vista 와 같은 수직 어휘가 안 된다.
  - (b) `route_presets.py:189-192` 의 수직 슬롯을 up/down 쌍으로 늘리거나, dynpose 도
    vista 처럼 `--presets` 없이(전량) 굽는다. **vista 와 어휘를 맞추려면 이쪽이다.**
    다만 전량으로 가면 씬당 변이 수가 크게 늘어 dd10 리스트 기준을 다시 잡아야 한다.
- 비용: 재굽기 1회(dynpose 278 씬). §4-3(코퍼스 합치기)보다 **뒤**에 한다 — 합치기가 수직
  어휘를 vista 쪽에서 이미 공급하므로, 이 수정의 이득이 합친 뒤에는 줄어든다.

### 4-3. [데이터] 두 코퍼스 합치기 — 지금 근거가 가장 강한 "추가 학습"

- 근거: (i) §1-b — d121(14975) 이 d128(9389) 을 자기 테스트셋에서 7/9 로 이긴다 = 현 규모대에서
  세그먼트 수가 세다. (ii) §2-a — 두 코퍼스가 덮는 preset 문법이 거의 배타적이다
  (수직 19.4%/0.0%, `track_*` 19.8%/46.5%). (iii) 씬 수 48 vs 240 로 씬 다양성도 상보적.
- 처방: `MixedCamDataset`(task #26 에서 이미 배선됨) 으로 vista d121 ∪ dynpose d137 arm 을
  추가한다. 코퍼스 비율은 seg 수 그대로(14975 : 9728)가 기본, 씬 균형을 보고 싶으면 씬당
  sampling.
- **단, 2×3 그리드가 다 차기 전에는 안 돌린다** — 지금 GPU 0~4 가 전부 차 있고(§0), 합침 arm 은
  그리드 결과를 알아야 어느 모델축과 짝지을지 정할 수 있다.
- 비용: config 1개 + smoke + 100 epoch (약 8~10h).

### 4-4. [데이터] `dd_*` 181종 one-off 의 처우

- 근거: §2-a. 181 uniq / 973 seg = 라벨당 5.4 seg. dynpose 세그먼트의 10.0%.
- 문제: preset 이 조건 신호로 쓰이는 구조에서 라벨당 5 샘플은 학습이 안 되고, "uniq preset 202"
  라는 통계만 부풀린다. goals.md §2 는 DataDoP 를 **의도적으로** free-moving 으로 넣기로 했으니
  라벨을 살릴 이유가 약하다.
- 처방: `dd_*` 를 개별 라벨 대신 단일 `dd_freemoving` 으로 접거나, 캡션에서만 쓰고 preset 통계·
  라우팅에서는 빼는 열을 만든다. **삭제는 권하지 않는다** — goals.md 가 비율(49.6%)을 의도적으로
  맞춘 축이다.
- 비용: export 필터 한 줄 + 통계 스크립트.

### 4-5. [모델] molmo2 이득의 출처 분해 — 이미 실행 중

- 근거: §1-a. D124 가 D123 을 8/9 로 이기지만, D171 계측에서 `dpred_drop 0.02395` vs
  `dpred_xscene 0.00462`.
- 처방: D133(vista molmo2 단독) / D138(dynpose molmo2 단독) 완료 후 세 칸을 나란히 본다.
  - molmo2 단독이 da3 단독 수준을 유지 → molmo2 가 geo 를 대체할 만큼 씬 정보를 나른다
  - 크게 떨어짐 → molmo2 는 geo 위에 얹힌 여분 용량 (= CA 슬롯 효과)
- **추가로 필요한 대조 1개**: 위 두 결론 중 후자면, "CA 슬롯 하나 더"의 순수 효과를 재려면
  **molmo2 자리에 shuffle 된 다른 씬 임베딩을 넣은 arm** 이 있어야 완전히 갈린다. D171 은
  학습된 모델에 대한 사후 프로브였고, 학습 자체를 그렇게 돌린 arm 은 없다. 그리드가 끝나고
  후자 결론이 나오면 그때 돌린다.

### 4-6. [모델] `subject_in_frame` 을 학습 중 로깅 (task #129)

- 근거: 현재 val 지표는 captions/clatr 뿐이다. goals.md 완료조건 (a) 를 **직접 재는 지표가
  학습 루프에 없다.** §2-c 처럼 코퍼스에서는 재는데 모델 출력에서는 안 잰다.
- 처방: val 스텝에서 예측 궤적으로 subject OBB 를 투영해 `subject_in_frame` 을 계산해 wandb 에
  올린다. 재료(OBB track, K, c2w)는 이미 데이터셋에 있다.
- 비용: 학습 코드 소폭 + 재학습 불필요(다음 arm 부터 적용).
- 이게 없으면 4-1 의 개선이 **모델 쪽에서 효과가 있었는지**를 못 잰다. 4-1 과 짝이다.

### 4-7. [모델] ckpt 기준 통일

- 근거: §0. D123/D124 는 101 epoch, 이후는 100 epoch.
- 처방: 앞으로의 전 비교는 `last.pth` 로만 한다 (`epoch100.pth` 병용 금지). 이미 평가된
  D123/D124 수치를 재활용할 때는 표에 epoch 차이를 명시한다.
- 비용: 0.

### 4-8. [데이터] 게이트 꼬리 정리 vs 규모 — 지금은 규모가 이긴다

- 근거: §1-b 의 D131 vs D123. d128 은 d121 대비 게이트가 더 빡빡한데(꼬리 표에서 전 항목
  d128 ≤ d121) 자기 테스트셋에서 7/9 로 졌다. seg 수 1.6배 차이가 교란.
- 처방: **꼬리를 더 자르는 방향의 재굽기는 당분간 보류.** 대신 4-3(합치기)으로 규모를 올린 뒤
  같은 규모에서 꼬리 두께만 바꾼 대조를 한다. 그 전까지 "게이트를 조였더니 좋아졌다/나빠졌다"는
  주장은 규모 교란과 분리되지 않는다.

---

## 5. 다음 행동 (순서)

1. 4개 학습(D133 / D137 / D138 / D141) 100 epoch 완료 대기 → `last.pth` 로 각 코퍼스 테스트셋
   페어드 eval. **eval 은 한 번에 1개만** (/data1 Lustre).
2. 2×3 표를 이 문서 §1 에 채운다. 채워진 칸만 가지고 결론 쓰지 않는다.
3. ~~4-1(캡션 프레이밍 절 제거) → 캡션 재생성 → §2-c 재측정.~~ **완료 (D143, 2026-09-06)** —
   vista 두 뱅크에 적용해 새 루트 `latentcam_da3_k6_d121c143` / `..._d128c143` 를 뽑았다.
   dynpose d137 은 학습 중이라 미적용. 재측정은 §4-1a.
3-a. dynpose 학습(D137/D141)이 끝나면 같은 드라이버로 d137 캡션 재생성 + 재export.
3-b. §4-1a 의 dynpose 잔차(17.0%) 진단 — §4-2 다음 순위.
4. 4-2 확인(라우팅인지 게이트인지).
5. 그리드 결론 나온 뒤 4-3(합침 arm) 1개 추가 학습.
6. 4-6(`subject_in_frame` 로깅)은 4-3 arm 부터 적용.
