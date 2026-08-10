# Changelog (latentcam)

All notable changes to the latentcam sub-project. Follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- **`dataset_dl3dv.py` 의 `avg_scale` 설명이 틀려 있었다 (주석만 수정, 동작 변화 없음).**
  모듈 상단 `scale_mode` 표와 `_avg_scale` docstring 은 저장된 avg_scale 을
  `mean(||scene point - first camera||)` 하나로 적어 뒀지만, **`pose_source` 마다 생성기가
  다르고 정의도 다르다.** 생성 스크립트를 찾아 6/6 세그먼트 전부 소수점 6 자리까지 재현해 확인:

  | pose_source | 파일 | 점 | 기준점 | target 사용 |
  |---|---|---|---|---|
  | `transforms` | `<scene>/avg_scale/<seg>.json` | `scene.ply` 전체 | **target segment 첫 카메라** | 쓴다 |
  | `da3` | `<scene>/da3/avg_scale/<seg>.json` | context 프레임 da3 depth (conf >= 전역 P40, pixel_stride 2) unproject | **context 카메라 중심들의 centroid** | 안 쓴다 (**leakage-free**) |

  - `transforms` 생성기: `pipeline/workspace/make_avg_scale.py` → `normalize_camera_extrinsics_and_points(extrinsics[s:e], scene.ply)`, `extrinsics[0]` 이 기준.
  - `da3` 생성기: `pipeline/workspace/make_avg_scale_da3.py`. context range = `[0,s)` 와 `[e,N)`
    중 **프레임이 많은 쪽 하나**, `centroid = cam_c[ctx].mean(0)`,
    `avg_scale = mean(||P_ctx - centroid||)`.
  - 재현 (scene `1K/9c2ede…`, `da3/avg_scale/{0..5}.json`): json 9.203050 / 9.052784 / 8.823183
    / 10.359932 / 10.083172 / 9.143390 = centroid 기준 계산값과 **완전 일치**. 같은 점 집합을
    "context 첫 카메라" 기준으로 재면 9.929137 / 9.090941 / 10.793398 / 13.941890 / 15.083957 /
    14.697644 로 어긋난다 → 기준점은 첫 카메라가 아니라 centroid 다.
  - **함의:** `pose_source` 를 바꾸면 pose·caption 뿐 아니라 **분모의 정의 자체**가
    target 기준 → context 기준으로 바뀐다. 즉 `da3_7k_da3pose` vs `da3_7k_textonly` 는
    "pose 만 다른 paired 비교"가 아니다. 앞선 항목의 `avg_scale` mean 15.52805(transforms) vs
    4.69791(da3), 비율 p50 0.23851 은 단위 차가 아니라 **다른 양을 잰 것**이다.
  - Scene-Decoupled 는 원래부터 context 기준이다 (`dataset_scene_decoupled.py:227-230`,
    context clip 의 `avg_scale_align/0.json`). 즉 leakage-free 인 쪽은 `da3` + SD 이고,
    `transforms` arm 만 target 기준이다.

### Added
- **`scripts/eval/mean_traj_baseline.py`** — `val/loss_traj` / `val/loss_latent` 의 **"코퍼스 평균
  궤적만 내놓는 모델"** 기준선을 코퍼스별로 잰다. `mean_n (x - mean_n x)^2` (per-(t,c) 평균 궤적)
  을 `cam_param` 공간과 VAE-latent(`/vae_latent_scale`) 공간 양쪽에서 계산한다. `cfg.geo_encoder`
  를 `None` 으로 눌러 이미지/depth I/O 를 건너뛴다.

  **동기가 된 질문: "Scene-Decoupled(SD) 로 학습한 게 왜 DL3DV 보다 성능이 훨씬 높게 나오나?
  context 에 target segment 가 들어간 건 아닌가?"**

  (1) **누수는 없다.** `sd_whuman_train.txt` 52286 줄 / 3006 scene, `sd_whuman_test.txt` 5758 줄 /
  334 scene 에서 `awk -F'__' '$2==$3'` = **0 쌍** (target clip == context clip 인 샘플 없음),
  train/test scene 교집합 `comm -12` = **0**. 게다가 `sd_whuman_textonly` arm 은 context 자체가
  없는데도 가장 낮은 loss 를 낸다 — context 경로로는 설명이 안 된다.

  (2) **점수 차이는 타깃 분산 차이다.** val 집합 600 segment (`--n 600`) 기준선:

  | 코퍼스 | `cam_param` mse_vs_mean_traj | VAE-target mse_vs_mean_traj | VAE-target std |
  |---|---|---|---|
  | SD whuman | 0.012418 | 0.062269 | 0.255507 |
  | DL3DV DA3 | 0.091427 | 0.538472 | 0.752311 |
  | DL3DV COLMAP | 0.049417 | 0.234451 | 0.497127 |

  run 의 wandb summary 를 자기 코퍼스 기준선으로 나누면 (수치는 wandb 원본):

  | run (id) | `val/loss_traj` | / baseline | `val/loss_latent` | / baseline |
  |---|---|---|---|---|
  | `sd_whuman_textonly` (rzu5qedy, ep25 killed) | 0.01226280815899372 | 0.988 | 0.07175761461257935 | 1.152 |
  | `sd_whuman_customgeo` (fkfqww00, ep4 running) | 0.016203269362449646 | 1.305 | 0.15333011746406555 | 2.462 |
  | `da3_7k_customgeo_nos` (68iuifk0, ep97 running) | 0.05391194298863411 | 0.590 | 0.3500967025756836 | 0.650 |
  | `da3_7k_textonly` COLMAP (6chmxgrq, ep60 killed) | 0.0486505962908268 | 0.985 | 0.2414451539516449 | 1.030 |

  즉 SD 의 `val/loss_traj` 0.0123 은 **자기 코퍼스의 평균-예측 기준선 0.012418 과 사실상 같다
  (0.988)**. 낮은 loss 는 학습이 잘 된 게 아니라 SD 타깃 분산이 DL3DV 의 1/4~1/7 이라서다.
  같은 잣대로 DL3DV DA3 customgeo 만 0.590 / 0.650 으로 기준선을 유의미하게 깬다.

  (3) **원인: SD 코퍼스에 카메라 프리셋이 5 종뿐이다.** clip 이름 접미사 기준 train
  `_01_24mm` 10979 / `_02_24mm` 10922 / `_05_24mm` 10154 / `_06_24mm` 10121 / `_07_24mm` 10110,
  test 1204/1205/1109/1103/1137. 환경(Rome, Gothic_Mansion, IslandMap, Dragon_Rise …)만 바뀌고
  카메라 무브는 그대로다. test 600 샘플의 `cam_param` RMSD: 프리셋 평균궤적 기준 0.0492 / 0.0477
  / 0.0580 / 0.0500 / 0.0481, 전체 pooled 0.1114 → **궤적 분산의 약 80% 를 프리셋 라벨 5 개가
  설명**한다. 프리셋 평균궤적끼리의 거리는 05-07 0.0045, 05-06 0.0175, 06-07 0.0181 로
  within-preset 산포보다 한 자릿수 작아 05/06/07 은 사실상 같은 무브다 (01-02 는 0.2947).
  scene 은 disjoint 여도 **궤적 라벨 공간은 train/test 가 동일**하다.

  (4) **"성능이 높다"는 것도 지표 한정이다.** wandb summary 원본:
  `val/captions/fscore` SD textonly 0.5626 vs DL3DV COLMAP 0.2673, 그러나
  `val/clatr/precision` 0.1500 vs 0.9813, `density` 0.1792 vs 1.1271, `coverage` 0.1750 vs 0.9313,
  `fcd` 319.9549 vs 172.5301 로 분포 지표는 전부 크게 나쁘다. 이전 `corpus_traj_manifold.py`
  결과(SD GT tag 엔트로피 2.8538 vs DL3DV 4.2036 bit, 48 step 내내 tag 1 개 유지 비율 0.7500 vs
  0.1938)와 같은 방향이다 — SD 모델은 저진폭·저다양성 궤적을 내놓고, 그게 진폭/태그 기반
  지표에서만 유리하게 잡힌다.

  주의: `val/loss_traj` 는 코퍼스 간 직접 비교 불가다. 위 비율은 각 run 이 실제로 쓴
  `vae_latent_scale` (SD run 도 DL3DV 상수 0.96032625 사용, SD 자체 값은 ≈0.2528 로 추정)을
  기준선 계산에도 똑같이 적용해 맞춘 것이다.

  (5) **`--group sd_preset` — 프리셋 oracle 하한** (val 1200 seg). "taxonomy 가 줄었으면 오히려
  더 잘 맞춰야 하는 것 아니냐"는 지적에 답하기 위해, **프리셋 라벨을 정확히 안다고 가정**했을
  때의 MSE(=프리셋 내 분산)를 쟀다:

  | 공간 | pooled 평균-예측 | 프리셋 oracle | oracle/pooled |
  |---|---|---|---|
  | `cam_param` | 0.012204214888561328 | 0.0026682241351742033 | 0.219 |
  | VAE-target | 0.061723771252337176 | 0.021127361593618044 | 0.342 |

  프리셋 내 RMSD 0.0491(01) / 0.0492(02) / 0.0547(05) / 0.0555(06) / 0.0498(07),
  프리셋 평균궤적 간 RMSD 01-02 0.2939, 01-05/06/07 0.1515~0.1551, **05-06 0.0043 / 05-07 0.0049
  / 06-07 0.0036** (실효 궤적 3 종).

  즉 라벨만 맞히면 `cam_param` MSE 0.00267 까지 내려가야 한다. 그런데 실제 run 은

  | run | val/loss_traj | / pooled | **/ oracle** |
  |---|---|---|---|
  | `sd_whuman_textonly` (rzu5qedy, ep25) | 0.01226280815899372 | 1.005 | **4.60** |
  | `sd_whuman_customgeo` (fkfqww00, ep4) | 0.016203269362449646 | 1.328 | **6.07** |

  이고, rzu5qedy 의 마지막 6 epoch val/loss_traj 는 0.011152 / 0.0123 / 0.01286 / 0.010486 /
  0.011928 / 0.012263 로 **이미 평탄**하다(최저 0.010486 도 oracle 대비 3.93). 즉 SD 는 문제가
  쉬워져서 loss 가 낮은 게 아니라, **쉬워진 문제조차 못 풀고 pooled 평균 근처에 머무는 중**이다.
  비교로 `da3_7k_customgeo_nos` (68iuifk0) 는 마지막 6 epoch 0.056614 / 0.053448 / 0.060576 /
  0.054481 / 0.05384 / 0.053912 로 pooled 기준선의 0.59 다.
- **`scripts/eval/prompt_motion_stats.py`** — 학습이 실제로 읽는 캡션 필드
  (`prompt_camera_with_scene_video.concise`) 에서 **축(axis)/방향(direction) 정보가 남아 있는지**
  를 pose_source 별로 센다. "DA3 캡션은 truck/dolly 가 사라지고 move/yaw 만 남아서 정보가 줄었다"
  는 앞선 추정을 검증하기 위한 것 — 어휘 수가 아니라 축 표기 유무를 봐야 한다.

  test seg list 3985 caption, COLMAP(`<scene>/prompts.json`) vs DA3(`<scene>/da3/prompts.json`):

  | 지표 | COLMAP | DA3 |
  |---|---|---|
  | 축 정보가 있는 caption 비율 | 0.9997490589711417 | 0.9997490589711417 |
  | lateral | 0.9761606022584692 | 0.9851944792973651 |
  | yaw (yaw+pan) | 0.9681304893350062 | 0.9513174404015057 |
  | depth | 0.6637390213299874 | 0.6052697616060226 |
  | vertical | 0.3565872020075282 | 0.3214554579673777 |
  | pitch (pitch+tilt) | 0.2582183186951066 | 0.1922208281053952 |
  | roll | 0.0602258469259724 | 0.0484316185696361 |
  | move 계열 토큰 수 | 4262 | 3464 |
  | 그중 뒤 3 토큰 안에 방향어가 있는 비율 | 0.9899108399812294 | 0.9760392609699770 |
  | 방향어 없는 bare move | 43 | 83 |
  | caption 평균 토큰 수 | 21.028858218318696 | 18.504391468005018 |
  | 용어 카운트 | pan 5334, yaw 1009, tilt 1001, truck 640, dolly 625, roll 269, pitch 210, pedestal 25, crane 7 | pan 3797, yaw 1787, tilt 504, pitch 340, roll 205, truck 10, zoom 6, crane 6, pedestal 4, arc 1, orbit 1 |

  **결론: 촬영용어만 평범한 말로 바뀌었고 축·방향 정보는 그대로다.** "move right" 는 truck right
  와 같은 축을 지정하고, 방향어 동반율이 0.9899 -> 0.9760 로 거의 안 떨어진다. 축 표기율 차이는
  depth −5.8pp / vertical −3.5pp / pitch −6.6pp 수준이고 caption 이 2.5 토큰 짧아진 정도다.
  따라서 **DA3 arm 의 지표 차이를 캡션 어휘 변화로 설명하기 어렵다** — 앞선
  `pose_source_agreement.py` 항목의 "어휘가 줄어 텍스트 조건이 약해졌다"는 뉘앙스를 여기서
  정정한다. 남는 후보는 GT 궤적 쪽(DA3 pose 의 지터/스케일)이다.
- **`custom_geo_channels` (기본 `full`)** — `geo_encoder: custom` 의 GeoTokenizer 에 무엇을
  넣을지 고르는 ablation 스위치. `full` 은 기존 경로와 완전히 동일하다 (기존 run 영향 없음).

  | 값 | GeoTokenizer 입력 | proj 입력 | dataset 이 만드는 텐서 |
  |---|---|---|---|
  | `full` (기본) | Plücker(6) + log-depth(1) + valid(1) | 1024 + 256 | plucker, logd, valid |
  | `no_depth` | Plücker(6) 만 (stem 첫 conv in_ch 8->6) | 1024 + 256 | plucker |
  | `rgb_only` | GeoTokenizer 제거 | 1024 | 없음 |

  `no_depth` 는 폭/구조가 base 와 같고 첫 conv 의 2 채널(1152 param)만 줄어서 **용량이 아니라
  depth 입력의 유무**를 잰다. `rgb_only` 는 frozen DINOv2 특징만 남으므로 context 가 외관만
  주고 카메라 기하는 전혀 주지 않는다. geo view 선택(`frustum_cover`)은 그대로라 세 arm 이
  **같은 context view 집합**을 본다.
  - `models/custom_geo_encoder.py`: `GeoTokenizer(use_depth=)`, `SceneEncoder(channels=)`.
    `_to_input_hw`/`forward` 가 안 쓰는 입력을 `None` 으로 받는다.
  - `models/geo_encoder.py`: `_CustomBackend` 가 `needs_keys` 를 노출하고 `batch.get()` 으로 읽는다.
  - `main/train_latent_cam_dm.py:geo_encode`: 하드코딩된 4-key 검사 대신 `geo_encoder.needs_keys`.
  - `main/dataset_dl3dv.py` / `main/dataset_scene_decoupled.py`: 안 쓰는 텐서를 아예 안 만든다
    (`no_depth` 는 da3 depth mmap I/O 가, `rgb_only` 는 픽셀 Plücker 생성까지 빠진다).
  - 실험 yaml 2 개 (`da3_7k_customgeo_nos` 와 각각 **한 줄만** 다르다):
    `main/conf/experiment/da3_7k_customgeo_nos_nodepth.yaml`,
    `main/conf/experiment/da3_7k_customgeo_nos_rgbonly.yaml`.
- **`scripts/eval/target_scale_stats.py`** — "avg_scale 이 원인이냐"를 **실제 diffusion 타깃**
  으로 답한다. 궤적 요약량(reach)이 아니라 `data['cam_param']` 과
  `vae.encode(traj)/vae_latent_scale` 을 직접 잰다. 두 arm 이 같은 seg list 라 paired 다.

  | | DA3 | COLMAP(transforms) |
  |---|---|---|
  | `avg_scale` mean (test 800 seg) | 4.50488 | 16.11375 |
  | `cam_param` trans std | 0.44565 | 0.14929 |
  | `cam_param` rot6d std | 0.49553 | 0.49554 |
  | VAE latent std (**FULL corpus** 39817 sample / 6095 scene) | **0.73158** | **0.47637** |
  | `/vae_latent_scale` 0.96032625 후 = diffusion 입력 std | 0.7618 | 0.4960 |

  즉 같은 노이즈 스케줄에 **1.536 배 다른 신호 크기**가 들어간다. 회전 성분은 두 arm 이
  완전히 같고 (0.49553 vs 0.49554) 차이는 전부 translation 분모에서 온다.
  단 손해의 방향은 이걸로 정해지지 않는다 — 명목 1.0 에는 da3(0.7618) 쪽이 오히려 가깝다.
  FULL corpus 값은 `scripts/vae/vae_scale_matrix.py` 를 `MAX_SCENES=none` 으로 돌려 얻었다
  (기본값 200 은 저속 scene 편중이라 4~7% 낮게 나온다).
- **`scripts/eval/pose_source_agreement.py`** — "da3pose 가 textonly(COLMAP)보다 못 나오는 게
  CLaTr 탓인가?" 를 가른다. 두 arm 의 `preds.npy` 가 **같은 160 segment 를 같은 순서로** 담고
  있어서 paired 비교가 된다 (filename 완전 일치 확인). 재추론 없음.

  | | DA3 | COLMAP |
  |---|---|---|
  | GT 지터 pos (국소 2차 적합 잔차 / 스텝길이) p50 | 0.0632 | 0.0538 |
  | GT 지터 rot (회전 2차 차분, deg) p50 | 0.5235 | 0.4922 |
  | DA3->COLMAP sim3 정렬 RMSE / 경로길이 p50 | 0.0016 (p95 0.0053) | — |
  | 정렬 후 회전 불일치 (deg) p50 | 0.4624 (p95 3.2316) | — |
  | 정렬 후 reach 비 p50 | 0.9999 | — |

  즉 **두 pose_source 의 GT 궤적은 sim3 를 빼면 사실상 같은 궤적이다** (경로길이의 0.16%).
  DA3 pose 가 틀려서 지는 게 아니다. 갈리는 건 **분모**다 — 7K test 3985 segment 전량:
  `avg_scale` mean 15.52805(transforms) vs 4.69791(da3), 비율 p5 0.06490 / p50 0.23851 /
  p95 1.05628 (16 배 산포).

  > **정정 (2026-08-10).** 이 항목은 처음에 `preds.npy['ref_matrices']` 에서 잰
  > std(log reach) 0.7381 vs 0.5670 / CV(reach) 0.6590 vs 0.4863 을 "정규화된 타깃의 산포"
  > 라고 적었다. **틀렸다** — `out_to_trajectory` (`utils/data_utils.py:81`) 가 translation 에
  > `scale` 을 **다시 곱해서** 돌려주므로 그 행렬은 world 단위다. 아래를 대신 볼 것.

  CLaTr 탓이 아니라는 근거 셋 (전부 `corpus_traj_manifold.py` 산출): (a) CLaTr 를 안 거치는
  caption fscore 도 같은 방향으로 진다 (ep50-59 평균 0.2069 vs 0.2369), (b) 두 GT 의 CLaTr
  구름이 구분 안 된다 (반경 28.9934 vs 29.1505, 3-NN r 25.5805 vs 26.0459, participation
  ratio 14.2113 vs 14.6033), (c) GT-vs-GT 천장도 같다 (density 1.0002+-0.0970 vs
  0.9999+-0.0954, coverage 0.8740+-0.0488 vs 0.8770+-0.0494).
  **prompt 도 통째로 바뀐다 (2026-08-10 추가).** `pose_source` 는 pose 만 바꾸는 게 아니라
  caption 파일을 `<scene>/prompts.json` -> `<scene>/da3/prompts.json` 으로 **조용히 갈아끼운다.**
  학습이 쓰는 필드는 `prompt_camera_with_scene_video.concise` 다 (`dataset_dl3dv.py:590`).
  7K test 3985 segment 전량에서 **3985 개(100%) 가 다르다** (평균 126.6 vs 110.0 자).
  같은 VL 모델(Qwen/Qwen3-VL-30B-A3B-Instruct)인데 생성 시점이 다르다 (2026-02-07 vs 2026-08-06).

  | | COLMAP `prompts.json` | `da3/prompts.json` |
  |---|---|---|
  | trucks / trucking | 518 / 74 | **0 / 0** |
  | dolly / dollies / dollying | 226 / 267 / 132 | **0 / 0 / 0** |
  | pan 계열 (pan+pans+panning) | 5334 | 3797 |
  | tilt 계열 | 1001 | 504 |
  | yaw 계열 (yaw+yaws+yawing) | 1009 | 1787 |
  | pitch 계열 | 141 | 231 |
  | 서로 다른 어휘 수 | 1714 | 1672 |
  | caption 당 모션 토큰 수 | 7.39 | 5.86 |
  | 서로 다른 모션 시그니처 | 1593 | 1019 |
  | 모션 시그니처 엔트로피 (bits) | 8.9234 | 7.8678 |
  | 최빈 시그니처 비중 | 0.0753 | 0.0665 |

  즉 **translation 을 가리키던 촬영 용어(truck/dolly/pedestal)가 사라지고** 평범한
  "moves right/forward" 로 대체됐고, 회전은 pan/tilt 에서 yaw/pitch 로 이동했다. 모션 표현의
  다양성도 줄었다 (시그니처 1593 -> 1019, 엔트로피 -1.06 bit). `val/captions/*` 가 채점하는
  27 translation x 7 rotation 태그 중 translation 축의 어휘 신호가 특히 약해진다.

  > **정정 (같은 날).** 처음 이 표를 `prompt_camera` 필드로 재고 "어휘 붕괴 (pan 1976->3,
  > 엔트로피 9.5480->7.9891, 최빈 0.0287->0.1313)" 라고 적었는데, `prompt_camera` 는 **학습에
  > 안 쓰인다** (dataset 은 `concise` 만 읽는다). `prompt_camera` 쪽 붕괴가 더 극단적인 건
  > 맞다 (truck/dolly/pedestal/pan/tilt 전부 0~3, 어휘 156->94) — 그건 그 필드가
  > `da3/tags/camera_tags.json` 의 `description`("move right + yaw left")을 LLM 이 그대로
  > 풀어 쓴 것이라 원문에 촬영 용어가 아예 없기 때문이다. 위 표가 학습에 해당하는 수치다.

  -> pose 효과와 prompt 효과를 가르려면 `pose_source: da3` + COLMAP prompt arm 이 필요하다 (미실행).
  (프롬프트 생성 스크립트는 이 저장소에 없어서 system prompt 가 어떻게 바뀌었는지는 확인 못 했다.)
- **`scripts/eval/corpus_traj_manifold.py`** — "SD 는 caption 정확도가 너무 높고 density/coverage
  가 너무 낮은데 카메라 분포가 단순해서인가?" 를 가른다. 학습이 이미 남긴 `preds.npy` /
  `preds_pcf.csv` 만 읽으므로 재추론이 없다. 핵심은 **천장(ceiling)** 대조군 — real 임베딩을
  반으로 갈라 한쪽을 fake 인 척 넣고 PRDC 를 돌려서, 참분포 표본이 받는 점수를 잰다.
  SD whuman textonly(ep21) vs DL3DV da3_7k textonly(ep54), n=80 맞춤, k=3, 200 split:

  | | SD whuman | DL3DV da3_7k |
  |---|---|---|
  | GT-vs-GT density (천장) | 1.0016 +- 0.1240 | 1.0127 +- 0.1361 |
  | GT-vs-GT coverage (천장) | 0.8877 +- 0.0843 | 0.8881 +- 0.0601 |
  | GT 태그 엔트로피 (max 7.5622) | 2.8538 | 4.2036 |
  | 서로 다른 GT 태그 수 | 11 | 42 |
  | 48 스텝 내내 태그가 안 바뀌는 궤적 | 0.7500 | 0.1938 |
  | 최빈 태그 시퀀스 하나가 차지하는 비중 | 0.3000 | 0.0500 |
  | straightness p50 | 0.9542 | 0.8804 |
  | sv2/sv1 < 0.01 (사실상 직선) 비율 | 0.2000 | 0.0063 |
  | real 3-NN 반경 r (median) | 15.7094 | 26.2507 |
  | real -> 최근접 fake (median) | 17.4911 | 22.8956 |
  | 위 둘의 비 (>1 이면 coverage 탈락) | **1.1209** | **0.8854** |

  결론: **천장은 두 코퍼스가 같다** -> 분포가 좁다고 density/coverage 가 기계적으로 눌리는 게
  아니다. 대신 눌리는 건 허용 반경으로, r 이 0.60 배가 되는 동안 모델 오차는 0.76 배밖에 안
  줄어서 비율이 1 을 넘어간다. coverage 는 지시함수라 이 27% 차이가 0.225 vs 0.9125 로 증폭된다.
  caption 쪽은 분포 단순함으로 **그대로 설명된다** (GT 궤적의 75% 가 48 스텝 내내 단일 태그).
- **Scene-Decoupled 코퍼스로 학습할 수 있게 되었다** (`dataset_name: scene_decoupled`).
  context 가 같은 scene 의 **다른 clip** 이라는 점만 빼면 DL3DV 경로와 규약이 같다.
  - **`main/dataset_scene_decoupled.py`** (신규) — `SDCamDataset(CamDataset)`.
    `scene_dir_list` 의 단위를 scene 이 아니라 **clip** 으로 두어서, 부모의
    `_geo_pixel_plucker` / `_geo_depth_maps` / `_geo_cam_cond` 에 context clip 인덱스를
    `scene_idx` 자리로 넘기기만 하면 그대로 돈다 (`rel = w2c_ctx_v @ inv(w2c_tgt_0)` 이
    cross-clip 에서도 유효한 이유는 sim3 가 두 clip 을 같은 world frame 에 올려놨기 때문).
    **분모 규약**: clip 별 `umeyama_gt.json` sim3 (`convention: gtrot`) 로 pose 와 depth 를
    GT meters 로 올린 뒤, **context clip 의 `avg_scale_align` 하나로** target pose /
    context Plücker translation / context depth 를 전부 나눈다. 추론 때 알 수 있는 건
    context clip 뿐이라 target 의 `avg_scale` 을 쓰면 leakage 다.
    static clip (`moving: false`) 은 `s`/`t` 가 없으므로 `_load_scene` 에서 즉시 raise 한다.
  - **`main/base.py`** — `build_dataset(cfg)` 추가. `dataset_name` 기본값 `'dl3dv'` 는
    기존 동작 그대로. seg-list 분할의 id 변환도 `type(dataset).seg_key` 로 위임했다
    (DL3DV = `<batch>_<hash>_<seg>` -> 슬래시 경로, SD = 항등).
  - **`main/conf/config.yaml`** — `dataset_name` / `sd_root` / `sd_split` / `sd_geo_views` 신규 키.
  - **`scripts/data/sd_build_seg_lists.py`** (신규) — 학습 전에 (target, context) 쌍 리스트를
    **뽑아 고정**한다. whuman 실측:
    ```
    clip 총 23408 (umeyama_gt.json 읽기 실패 0)
      static (moving=false) 제외   7682
      align 컷 (resid_rmse_over_rad > 코퍼스 p99 = 0.3273) 제외   158
      남은 clip 15568 / clip 2장 이상 남은 scene 3340 of 3344
      scene 단위 90/10 (seed 42) -> train scene 3006 = 52286 쌍 / test scene 334 = 5758 쌍
    ```
    컷 기준은 moving clip 15726 개 위에서 잰 백분위:
    `resid_rmse_over_rad` p50 0.0292 / p90 0.0928 / **p99 0.3273** / max 1.0000.
    `rot_spread_deg` p99 컷은 `--rot-p99` 로 쓸 수 있으나 **적용하지 않았다** — resid 컷 후
    잔여 분포는 p50 0.4081 / p90 1.2026 / p99 5.3515 / max 62.2741, 10 deg 초과가 88 개다.
    out -> `<sd_root>/latentcam_lists/sd_whuman_{train,test}.txt` + `_clip_stats.csv` + `_lists_summary.md`.
  - **`main/conf/experiment/sd_whuman_customgeo.yaml`**, **`sd_whuman_textonly.yaml`** (신규) —
    같은 리스트를 쓰는 paired arm. 차이는 geo 조건의 유무 하나뿐.
- **`scripts/render/sd_pair_scene_swap_render.py`** — 같은 scene 의 두 clip 을 각자 sim3 로 GT 에
  보낸 뒤 **scene 만 바꿔서** 렌더한다 (target clip 의 pose 로, source clip 의 depth+RGB 를
  unproject -> reproject). dynamic subject 때문에 출력 frame t 는 source 의 **같은 t 프레임 하나만**
  쓴다. 출력은 `GT | render[file] | render[gtrot]` × 두 방향의 2×3 mp4.

  이걸로 **`umeyama_gt.json` 의 `R` 이 cross-clip 에 쓸 수 없다**는 걸 확인했다. GT 카메라
  (`camera/<split>/<scene>/<scene>_cam.json`, Unreal LH, cm) 와 대조한 실측:
  - da3 의 상대 회전/이동은 GT 와 거의 완벽 (rel-rot 오차 0.2~1.5 deg, rel-trans 0.5% 이내).
  - 파일의 `s`,`t` 도 맞다 — 카메라 중심이 GT 와 1~2 cm 안에서 일치.
  - 그런데 `R` 은 **RH 변환 없이 Unreal 의 LH 좌표에 직접 맞춰져** 있고, 게다가 이 데이터셋
    궤적은 대부분 **완전 직선**(중심 좌표 특이값 `sv2/sv1 = 0.0000`)이라 Umeyama 의 회전이
    그 축 둘레로 1 자유도 미결정이다. 결과: 같은 scene 의 clip 두 개를 각자 `R` 로 보내면
    서로 최대 **174 deg** 어긋나는데 `resid_rmse_over_rad` 는 정상값이다.
  - 실측 (200 scene / 1788 moving-moving pair 중 resid 최고·최악 pair):
    `render[file]` 의 coverage median **0.0%** (best-resid pair 조차 23~35%, MAE 76),
    `render[gtrot]` 는 frame 0 coverage **99.9%** / MAE 13~19 로 정상.
  - **`gtrot` 로 고친다**: `R` 을 위치가 아니라 GT 카메라 **방향**에서 푼다
    (`R_sim = R_gt_rh^T A0^T R_ext`, SVD 로 프레임 평균; `p_rh = diag(1,-1,1) p_ue`,
    `A0 = [[0,-1,0],[0,0,-1],[1,0,0]]`). 잔차 `gtrot R spread` 0.08~2.50 deg.
    회전만 있으면 풀리므로 **static clip 도 `R`,`t` 는 나온다** (`s` 만 미결정).
  out -> `results/scene_decoupled/pair_scene_swap/{*.mp4,selection.md,per_frame.csv}`.
- **`scripts/data/sd_static_scale_transfer.py`** — static clip(`moving:false`, 23408 중 7682 =
  32.8%) 을 살릴 수 있는지 잰다. 같은 scene 의 clip 들은 frame-0 pose 가 같으므로 이미 정렬된
  moving clip M 의 frame-0 depth 로 static clip S 의 스케일을 역산한다
  (`s_S = median(d0_M·s_M / d0_S)` -> `avg_scale_align_S = avg_scale_S · s_S`).
  10 scene / static 21 / (static,moving) pair 102 실측:

  | 지표 | p50 | p90 | p99 | max | moving-moving 참고선 |
  |---|---|---|---|---|---|
  | `shape_sd` 깊이맵 모양 불일치 | 0.0397 | 0.0950 | 0.1285 | 0.1332 | p50 0.0321 / max 0.1205 |
  | `s_spread` 기준 clip 을 바꿨을 때 s 의 max/min | 1.0685 | 1.1967 | 1.3201 | 1.3274 | `scale_off` p50 1.0352 / max 1.2939 |

  둘 다 moving-moving 잡음 바닥과 거의 같은 자리 -> **static clip 은 역산으로 살릴 수 있다.**
  out -> `results/scene_decoupled/static_scale_transfer/{summary.md,per_static.csv,per_pair.csv}`.
- **`scripts/data/sd_frame0_depth_agreement.py`** — `avg_scale_align` 이 scene 안에서 clip 마다
  흔들리는 원인이 (a) clip 마다 보이는 부분이 달라서인지 (b) 똑같이 보는 부분조차 스케일이
  어긋나서인지를 가른다. 가를 수 있는 이유: 같은 scene 의 clip 들은 frame-0 pose 가 동일해
  frame 0 이 곧 '공유되는 부분'이다 (frame-0 RGB 평균 절대차 **1.45/255** = mp4 압축 노이즈
  수준으로 실측 확인). 10 scene / 96 pair:

  | 지표 | p50 | p90 | p99 | max |
  |---|---|---|---|---|
  | `scale_off` 같은 걸 보는 데서의 스케일 어긋남 | **1.0352** | 1.1273 | 1.2351 | 1.2939 |
  | `shape_sd` 전역 스케일 뺀 깊이맵 모양 불일치 | 0.0321 | 0.0782 | 0.1129 | 0.1205 |
  | `align_ratio` 실제 divisor 가 흔들리는 폭 | 1.1173 | 1.5464 | 5.2608 | **6.7762** |
  | `s_ratio` umeyama sim3 스케일 비 | 1.0719 | 1.2167 | 1.5360 | 1.6035 |

  결론: **꼬리는 전부 (a)** 다. `align_ratio` 6.7762 인 최악 pair 의 `scale_off` 가 **1.0199**
  이고, `corr(log align_ratio, log scale_off)=+0.25` 로 둘이 거의 무관하다. sim3 정렬은 제대로
  되고 있고 (같은 걸 보는 데서 3.5% 오차), 흔들림은 "clip 마다 다른 point cloud 의 mean 거리"
  라는 양 자체의 성질이다. 이 3.5% 가 frame-0 depth divisor 의 오차 바닥이며
  `reach` 자체 산포(0.2359 dex)에 비하면 무시할 수준.
  out -> `results/scene_decoupled/frame0_agreement/{summary.md,per_pair.csv}`.
- **`scripts/data/sd_clip_divisor_spread.py`** — Scene-Decoupled cross-clip 의 divisor 후보를
  `norm_divisor_compare.py` 와 같은 기준(`m = max_t||c_t-c_0|| / D` 의 `sd(log10 m)`)으로 비교한다.
  own(target 자기 `avg_scale_align`, leaky 상한) / ctx(context clip 의 것, 현재 계획) /
  ctxd(context clip 의 median depth × s, meters).
  10 scene / 49 clip / 192 pair 실측: `sd(log10 m)` own **0.3749** · ctx **0.3370** · ctxd **0.3175**
  — 셋이 사실상 동률이라 **divisor 선택은 병목이 아니다**. 분해하면 `sd(log10 reach)=0.3470` 이
  지배하고 divisor 의 context 의존분(scene 내 산포)은 align 0.1115 / depth 0.0717 dex 뿐이며,
  `corr(log reach, log align)=0.31` 로 divisor 가 target 움직임과 거의 상관이 없다.
  부수 확인: `avg_scale` 은 `mean(||scene point - first camera||)`(dataset_dl3dv.py:921) 이라
  **카메라 움직임이 아니라 장면 깊이**다 — ctx 와 ctxd 의 `corr(log)=0.93` 이 그 결과다.
  또 mean 이라 원거리 point 에 취약하다: `scene927_...Creepwood` 는 scene 안에서 align 이
  8.276 -> 56.083 (**6.78x**) 흔들리는데 median depth 는 11.196 -> 22.793 (2.04x) 이고
  umeyama `resid` 는 0.009~0.030 로 전부 정상이라 **resid 필터로는 안 걸린다.**
  **[갱신] frame-0 depth 후보 `ctxd0` 추가** — 같은 scene 의 clip 들은 frame-0 pose 가 완전히
  동일하므로 frame-0 depth 는 context 선택과 무관해야 한다는 가설. 실측이 그대로 확인해준다.
  scene 안에서 clip 을 바꿨을 때 divisor 가 흔들리는 배수(max/min):

  | divisor | median | p90 | max |
  |---|---|---|---|
  | `align` (현재) | 1.2516 | 2.0751 | **6.7762** |
  | `depth_all` (전 프레임 median) | 1.4682 | 2.0572 | 2.2503 |
  | **`d0_med` (frame-0 median)** | **1.1172** | **1.1883** | **1.2813** |
  | `d0_mean` (frame-0 mean) | 1.1418 | 1.3160 | 1.4455 |

  `align` 이 6.78x 튀던 `scene927_...Creepwood` 가 `d0_med` 로는 **1.1265** 로 내려온다
  (`d0_mean` 은 1.4455 — mean 통계가 원거리 point 에 취약하다는 게 여기서도 재현).
  scene 내 `sd(log10)`: reach 0.2359 / align 0.1115 / depth_all 0.0717 / **d0_med 0.0198**(=1.047x).
  이 잔여 0.0198 은 da3 raw 스케일(0.0337)과 sim3 `s`(0.0356)가 **서로 상쇄된 나머지**라
  sim3 정렬이 제대로 됐다는 방증이기도 하다. `corr(log reach, log d0_med)=0.50` 으로 `align` 의
  0.31 보다 target 움직임과 더 상관돼 `sd(log10 m)` 도 최저(**0.3107**)다.
  다만 개선폭은 ctx 0.3370 -> ctxd0 0.3107 (8%) 로, `reach` 자체 산포가 지배한다는 결론은 그대로.
  out -> `results/scene_decoupled/clip_divisor_spread/{summary.md,per_clip.csv,per_pair.csv}`.
- **`cfg.ckpt_every_epochs`** (`main/conf/config.yaml`, 기본 `null`) — 기존 `ckpt_at_epochs` 가
  "이 epoch 들에서 `ckpts/epoch<N>.pth` 를 남겨라" 였다면 이건 **주기**로 같은 일을 한다
  (`50` -> epoch 50, 100, 150...). 파일명 규약(`epoch<N>.pth`, N = 0-based 루프 변수)이 동일해서
  두 설정이 같은 epoch 을 가리키면 파일 하나로 합쳐진다. epoch 0 은 건너뛴다. `null` 이면 기존
  동작 그대로라 `ckpt_at_epochs` 만 쓰던 실험은 영향 없음. `main/train_latent_cam_dm.py` 의
  고정 ckpt 저장 루프가 두 소스를 합쳐 중복 제거 후 저장한다.
- **`main/conf/experiment/da3_7k_textonly.yaml`** — `da3_7k_da3pose.yaml` 에서 `pose_source: da3`
  **한 줄만 뺀** COLMAP arm (= config 기본값 `transforms`). 기존 COLMAP text-only run
  `20260730_223514_dl3dv_textonly_savedscale_bs8` 은 da3pose arm 과 pose_source 외에 교란이 넷
  (segment 단위 random_split 의 scene leakage / blacklist 미적용 / `meta_worldtraj.csv` 6098 scene
  vs `meta_da3_7k.csv` 6095 / 150 epoch vs 100 epoch) 더 있어 paired 비교가 안 됐다. 이 arm 은
  meta_csv·seg list·blacklist 를 da3pose arm 과 **같은 파일**로 두고 `epochs: 150` 으로 맞춰
  넷을 전부 제거한다. `vae_latent_scale` 도 같은 상수(0.96032625) 유지.
  주의: 두 arm 은 GT 궤적도 translation 분모도 달라 `val/loss_traj` 를 직접 비교할 수 없다 —
  1K 실측 기준 "차이 없음" 기준선이 `(0.74367/0.46512)^2 = 2.557`. 크기 무관 지표는 CLaTr/caption.
- **`scripts/vae/vae_sd_scale_swap_recon.py`** — Scene-Decoupled cross-clip 계획(context clip 의
  `avg_scale_align` 을 target 의 분모로 사용)이 **frozen CameraVAE** 에서 버티는지 측정한다.
  선행 실측(300 scene / 5328 pair): `avg_scale_align == s * avg_scale` 가 **정확히** 성립하므로
  (상대오차 max 0.0) 자기 clip 의 align 으로 나누면 sim3 이전과 비트 단위로 같은 `cam_param` 이
  나온다 — sim3 가 실제로 무언가를 바꾸는 유일한 지점이 cross-clip 분모다. own/ctx 비는
  p1 0.228 / p50 1.000 / p99 4.389 / max 16.921.
  결과 (400 scene, 800 pair, T=49, `vae_20260302_300.pth`): 정규화 단위 `rec_trans` 가
  own **0.001743** vs swap **0.001756**, 비 4배 이상 버킷에서도 0.002416, 회전은 0.003200 vs
  0.003203 — **swap 은 VAE 의 블로커가 아니다.** 대신 `lat_std_scaled = 0.2633` 이 나와
  이 코퍼스의 `vae_latent_scale` 은 0.96032625 가 아니라 **0.2528** 근처여야 한다는 것이 드러났다
  (`in_c_norm 0.1075` 로 DL3DV transforms ~0.18 / da3 ~0.46 보다 궤적이 작다).
  out -> `results/scene_decoupled/vae_scale_swap.json`.
- **`scripts/render/sd_clip_pair_video.py`** — Scene-Decoupled 같은 scene 의 두 clip 을 가로로
  이어붙여 mp4 로 낸다. 프레임 위에 clip 이름 / `avg_scale` / `avg_scale_align` / `moving` /
  umeyama `resid` 를 찍는다. clip 은 scene 안에서 **첫 프레임 카메라 pose 가 완전히 동일**
  (중심 산포 0.0000 m, 회전 산포 max 0.084 deg, 200 scene 실측)하므로 화면 차이는 전부 이후
  궤적 차이다. out -> `results/scene_decoupled/clip_pair/<scene>__<A>_vs_<B>.mp4`.
- **`cfg.val_sample_seed`** (`main/conf/config.yaml`, 기본 `42`) + `scripts/eval_testset.py --sample-seed` —
  `sample()` 의 `x_T ~ N(0,I)` 를 **배치마다 `(seed + step)` 으로 시드한 CPU generator** 에서 뽑아
  샘플링을 결정적으로 만든다. `cfg.sampling_type: ddim` 의 DDIMScheduler 는 eta=0 이라
  `step()` 이 노이즈를 안 뽑으므로, 이걸 고정하면 sampling 의 확률적 요소가 전부 사라진다.
  **왜:** `val/loss_traj` 는 segment 당 표본 1개와 GT 의 MSE 라 가중치가 같아도 노이즈 추첨만
  바뀌면 값이 움직인다. 7K 3 arm 실측 plateau sd = **0.0046** (평균 0.0646 의 7%) 인데
  비교 대상인 withs vs nos 격차는 **0.0005 (0.11 sd)** 라 wandb val 곡선으로는 arm 을 가릴 수
  없었다. 전역 `manual_seed` 만으로는 부족하다 — arm 마다 RNG 소비 패턴이 달라 수열이 어긋난다.
  배치별 generator 는 arm 이 달라도 segment 별로 **같은 노이즈**를 쓰는 짝지은 비교
  (common random numbers) 를 만든다.
  검증: 같은 `last.pth` 를 GPU 0 과 GPU 3 에서 각각 돌려 `loss_traj 0.071126245893538`
  비트 단위 동일 (CPU generator 라 GPU 를 바꿔도 같은 수열).
  `cfg.val_sample_seed: null` / `--sample-seed -1` 로 두면 예전처럼 전역 RNG 를 쓴다.
  `eval_testset` 출력 디렉토리는 `eval_my/<run>__<ckpt>__seed<N>/` 로 바뀐다 — legacy 전역 RNG
  로 만들어 둔 기존 `eval_my/<run>__<ckpt>/` 를 덮어써서 다른 노이즈 위의 수치와 조용히
  섞이는 것을 막는다.
- **`cfg.ckpt_at_epochs`** (`main/conf/config.yaml`, 기본 `[]`) — 여기 든 epoch 에서
  `ckpts/epoch<N>.pth` (custom geo arm 이면 `epoch<N>_geo.pth` 도) 를 따로 박아 둔다.
  `best.pth` / `last.pth` 는 계속 덮어써지므로 **학습 중간 지점의 모델을 남길 유일한 방법**이다.
  기본이 빈 리스트라 켜지 않으면 기존 run 디렉토리 구조와 동일하다.
- **`main/conf/experiment/da3_7k_{da3pose,customgeo_withs,customgeo_nos}.yaml`** —
  같은 3 arm 의 **7K 코퍼스판** (DL3DV 1K -> 1K~7K). 모델/조건 설정은 1K 판에서 한 줄도
  안 바꿨고, 바뀐 것은 (a) 데이터 범위, (b) `epochs` 150 -> 100, (c) `ckpt_at_epochs: [50]`,
  (d) `meta_csv` / `train_seg_list` / `test_seg_list` / `coverage_blacklist_path` 뿐이다.
  세 arm 이 **같은** seg 리스트·blacklist 파일을 가리킨다 — 세그먼트 집합이 어긋나면 arm 간
  비교가 깨진다. `vae_latent_scale` 은 config 기본값 `0.96032625` 유지 (7K 로 재계산하지
  않음) 라 1K 세 arm 과 `loss_traj` 를 같은 축에서 비교할 수 있다.
  코퍼스 실측: 유효 6704 scene / 43989 segment -> meta 검증 후 6095 scene,
  scene-disjoint 0.9/0.1 (seed 42) = train 5485 scene / 35832 segment,
  test 610 scene / 3985 segment. segment teleport blacklist 1208/39817 (3.03%).
- **`scripts/eval/geo_ablation.py`** — 학습된 모델이 geo condition 을 **실제로 읽는지**를 재는
  paired ablation. 같은 batch / 같은 noise / 같은 timestep 에 geo 토큰만 바꿔
  `real` / `shuffle`(= `torch.roll(geo_emb, 1, 0)`, 토큰 통계는 유지하고 scene 짝만 깨뜨림) /
  `zero` / `none`(조건 자체 제거) 4조건의 epsilon MSE 와 `d_pred_vs_real` 을 비교한다.
  `shuffle` 이 핵심이다 — `zero` 는 `geo_proj` 에 bias 가 있어 "조건 없음"이 아니라 "상수 토큰"이
  되므로 내용 의존성을 분리하지 못한다. 부수적으로 layer 별 `resid_ratio = ‖a‖/‖x‖`,
  `attn_entropy_nats`, `view_mass` 를 뽑는다. `eval_testset.build_cfg` 재사용,
  출력 `eval_my/geo_ablation/<run>__<ckpt>/geo_ablation.json`.
- **`cfg.log_geo_attn` / `cfg.log_geo_attn_every` / `cfg.log_geo_attn_timestep`**
  (`main/conf/config.yaml`, 기본 `false` / `5` / `500`) — 학습 중 validation 때 geo cross-attn 을
  계측해 wandb 에 올린다. 기본이 `false` 라 기존 run 은 동작이 바뀌지 않는다.
  스칼라는 매 validation, attention map 이미지는 `log_geo_attn_every` epoch 마다.
  고정 timestep + 고정 seed noise 라 **epoch 간 비교가 성립**한다.
- **`main/conf/experiment/da3_1k_customgeo_{withs,nos}.yaml`** — 새 custom geo encoder
  (`models/custom_geo_encoder.py`)를 쓰는 arm 2종. **둘의 차이는 context view 선택 두 줄뿐**이다:
  `withs` 는 `geo_first_view_target_s: true` + `geo_cover_subtract_first: true`
  (= `geo_worldtraj.yaml` 형, context view0 이 target segment 의 첫 카메라 frame s 자체),
  `nos` 는 둘 다 `false` (= `geo_worldtraj_camembed.yaml` 형, target segment 카메라가 context 에
  하나도 안 들어감). 원본 camembed arm 과 달리 **`geo_cam_embed: null`** 이다 — custom 인코더는
  카메라를 view 당 11-d cam_token 이 아니라 픽셀당 Plücker ray 로 이미 받으므로 `relfirst` 는
  같은 정보의 중복이다. 공통: `pose_source: da3`, `geo_encoder: custom`,
  `custom_geo_input_hw: [252, 448]`, `custom_geo_freeze_dino: true`, `geo_posed: false`,
  `geo_latent_cache_dir: null`(인코더가 학습되므로 출력을 얼릴 수 없음), `batch_size: 8`,
  `epochs: 150`, `vae_latent_scale` 은 config 기본값 `0.96032625` 유지
  (`da3_1k_{textonly,da3pose}` 와 타깃 스케일을 맞춰 비교 가능하게).
- **`scripts/data/cache_da3_depth.py`** — `<scene>/da3/depth.npz`(deflate 압축)를 비압축
  `.npy` 로 풀어 `mmap_mode='r'` 로 필요한 6프레임만 읽게 하는 캐시 빌더. npz 는 프레임 하나를
  읽어도 scene 전체를 압축 해제해야 해서 **0.62 s/item**, `num_thread: 8` 기준 150 epoch 이면
  순수 압축 해제에만 ~17.7 시간이 든다. 캐시 경유는 **0.0197 s (31×)**.
  env `META`(기본 `meta_da3_1k.csv`) / `OUT`(기본 `<ROOT>/da3_depth_raw`) / `ROOT` / `JOBS` /
  `LIMIT`. 원자적 교체(`os.replace`)라 중단해도 반쪽 파일이 안 남고, 재실행 시 프레임 수가
  맞으면 skip. **실측: 957/957 ok, 0 error, 85 GB, 77 s.** `np.array_equal(cache, npz)` 검증 통과.
- **`cfg.custom_geo_patch`**(기본 14) / **`cfg.custom_geo_depth_cache_dir`**
  (`main/conf/config.yaml`) — 전자는 dataset 이 `custom_geo_input_hw` 의 격자 정합성을 검사할 때
  쓰는 patch 크기(DINOv2 ViT-L/14), 후자는 위 캐시 디렉토리. 캐시가 없으면 npz 로 자동 폴백한다.
- **`dataset_dl3dv`: `geo_plucker_map` / `geo_logd` / `geo_valid` 방출** (`geo_encoder: 'custom'`
  일 때만; 기존 backend 경로는 손대지 않았다).
  - `geo_plucker_map (V,6,252,448)` — `_geo_pixel_plucker`. **target segment 첫 카메라(frame s)
    기준** 상대 pose 로 만든 픽셀당 Plücker `[d(3), o×d(3)]`. translation 은
    `_geo_cam_plucker` 와 **같은 규약**으로 `/ norm_scale`. 격자는 encoder 입력 해상도
    (252×448)의 픽셀 중심을 원본 해상도로 되매핑해 만들고 `(H0,W0)` 별로 캐시한다.
    채널 우선 `(V,3,P)` 레이아웃으로 계산한다 — `(V,P,3)` + `norm(dim=-1)` + `torch.cross` 는
    P=112,896 에서 2~3× 느리다(0.0118→0.0043 s, 0.0091→0.0050 s).
  - `geo_logd (V,1,252,448)` — `log(depth / norm_scale)`. **Plücker translation 과 같은 분모**를
    써야 `o + exp(logd)·d` 가 target 궤적과 같은 좌표계의 3D 점이 된다. NaN/Inf 는 0 으로.
  - `geo_valid (V,1,252,448)` — 전부 1 (사용자 결정 2026-08-07). da3 depth 는 `depth<=0` 이
    0.0% 이고 `conf` 는 확률이 아니라 상한 없는 값이라(scene 별 p50 1.86~14.41) 코퍼스 공통
    임계값을 못 잡는다.
  - 기하 검증: `‖d‖` 편차 1.79e-07, `d·m` absmax 4.47e-08, `withs` arm 의 view0 moment
    2.51e-08(= frame s 카메라가 원점).
- **`scripts/vae/vae_pose_source_recon.py`** — frozen 카메라 VAE 가 `pose_source` 별 `cam_param`
  분포를 아직 감당하는지 재는 스크립트. arm 마다 `in_trans_std` / `in_c_norm` / `divisor_mean` /
  `lat_std_raw` / `lat_std_scaled`(= diffusion 타깃 std) / `vae_scale_for_1` / recon L1
  (`rec_trans`, arm 간 비교 가능한 `rec_trans_world` = ×divisor, `rec_rot6d`, `rec_intr`).
  env `EXPS` / `N` / `SPLIT` / `SEED`, 출력
  `results/compare/camera_jumps/vae_pose_source_recon.json`.
  - **결과 (N=800): da3 arm 은 VAE 재학습이 필요 없다.** `rec_trans_world` da3 **0.03713** <
    transforms **0.04568** (da3 가 오히려 낫다), `lat_std_scaled` da3 **0.744** vs transforms
    **0.465** (da3 가 1.0 에 더 가깝다). 즉 "da3/avg_scale 이 3.2배 작아 recon 이 열화된다"던
    이전 우려는 실측으로 **기각**. `vae_latent_scale` 도 `0.96032625` 그대로 둔다 — transforms
    arm 의 `lat_std_raw` 0.44667 은 config 주석의 기록값(0.44696 / full-corpus 0.47637)과 일치해
    현행 DL3DV run 전부와 같은 상태이고, 두 arm 이 같은 상수를 써야 paired 비교가 성립한다.
- **`scripts/data/scan_camera_jumps.py`** — **세그먼트 단위** teleport(프레임간 카메라 점프)
  스캐너. 기존 `scripts/data/filter_dl3dv.py::check_teleport` 는 (1) scene 단위라 한 프레임만
  튀어도 scene 전체를 버리고 반대로 세그먼트 하나의 점프는 통계에 묻히며, (2) 절대 임계값
  (`|dt|>10`, `‖dt‖>15`, `|c|>50`)이라 COLMAP 월드 단위에만 맞아 da3 예측 pose 에는 못 쓴다
  (`blacklist.csv` 의 `filter_teleport` 21건 + `filter_teleport_manual` 1건이 그 필터의 결과).
  여기서는 49프레임 세그먼트마다 **스케일 불변 지표**를 뽑는다 — `jr`=max(step)/median(step),
  `msf`=max(step)/path_len, `mse`=max(step)/extent, `gap`=max(step)/p90(step),
  `rmax`=프레임간 최대 회전(deg). `--source transforms|da3|both` 로 두 pose 소스를 같은 잣대로
  재고, 임계값을 주면 `dataset_dl3dv._read_coverage_blacklist` 가 그대로 읽는 `scene,segment`
  CSV 를 뱉는다(`--out-blacklist`; `--source both` 면 **합집합**). per-segment 원시 지표는
  `results/compare/camera_jumps/seg_jump_stats.csv`.
- **`data/seg_blacklist_jump_da3_1k.csv`** — 위 스캐너를 da3 1K 코퍼스 6,095 세그먼트에
  `jr>6 | msf>0.2 | rmax>30` 으로 돌려 나온 **123 세그먼트(2.02%, 70 scene)** 제외 목록.
  두 pose 소스의 합집합이다(transforms 만 105 / da3 만 98 / 교집합 80 — 한쪽만 망가진 경우가
  양방향으로 있다. 예: `a3efe59e3f9e` seg4 는 transforms 가 `rmax=179.95°` 인데 da3 는 `2.29°`).
  전 세그먼트가 다 걸린 scene 은 0개라 scene 손실은 없다. 임계값 근거: 두 소스 모두 p98 까지
  `jr≈4.3` 으로 완만하다가 p98.5~p99 에서 `jr` 10→28, `msf` 0.15→0.30, `rmax` 19→35 로 꺾인다.
- **`cfg.trans_repr`** (`main/conf/config.yaml`, 기본 `w2c`) — `cam_param[..., 6:9]` 를 w2c
  translation `t = −Rc` 로 둘지(기존) 카메라 중심 `c` 로 둘지 고르는 분기. rotation 채널(0:6)은
  어느 쪽이든 w2c `R` 의 앞 두 열 그대로다 — `R` 과 `Rᵀ` 는 정보량도 프레임간 geodesic 거리도
  같아서 규약 비교의 변수가 되지 못하고, 실제로 달라지는 건 translation 하나뿐이다.
  `‖t‖ == ‖c‖` 라 `norm_scale` 분모와 크기 분포는 동일하고 방향만 다르다.
  `intr_norm` 과 마찬가지로 **VAE ckpt 의 속성**이라 학습 때와 다르게 주면 recon 이 조용히
  망가진다. 디코드 경로 `utils/data_utils.out_to_trajectory(..., trans_repr=...)` 에도 같은
  인자를 붙였다(기본 `'w2c'` = 기존 동작과 bit-identical).
  - !! 미배선: DM/추론 스크립트의 `out_to_trajectory` 호출부(~30곳)는 아직 `trans_repr` 를
    넘기지 않는다. 전부 기본값 `'w2c'` 라 기존 동작은 그대로지만, `'c2w'` ckpt 를 DM 에
    쓰려면 그 호출부부터 배선해야 한다.
- **`main/conf/experiment/vae_transrepr_{w2c,c2w,smoke}.yaml`** — 위 규약 중 어느 쪽이 실제로
  회귀하기 쉬운지 재는 Step 1 의 paired arm 설정. 두 arm 은 `trans_repr` 한 줄만 다르고
  나머지(`scale_mode=avg_scale` / `intr_norm=rel` / `cam_dim=64` / bs 64 / 60 epoch / seed /
  seg 리스트)는 동일하다. `_smoke` 는 `max_scenes: 20`, 2 epoch 배선 확인용.
- **`scripts/vae/vae_trans_repr_recon.py`** — Step 1 평가기. GT 를 `trans_repr='w2c'` 로 한 번만
  로드해 두 arm 이 문자 그대로 같은 segment·순서·`norm_scale` 을 보게 하고, 각 arm 의 디코드
  결과를 **두 표현 모두로 환산**해 대칭 비교한다(`err_center` 주 지표 / `err_t` 대칭 확인용 /
  `err_rot_deg` / `err_intr` / `lat_std`). 출력은
  `results/compare/cam_repr_w2c_vs_c2w/vae_step1.json`. env: `W2C_CKPT` / `C2W_CKPT` / `EXP` /
  `N` / `BS`.
- **`scripts/data/make_da3_splits.py`** — DL3DV 1K 의 da3 코퍼스를 **scene-disjoint** train/test
  segment 리스트로 분할한다. `CamDataset` 을 세우지 않고 `da3/prompts.json` 을 직접 읽는다.
  scene 별로 blacklist / `da3/{prompts.json,pose.npz,depth.npz,conf.npz}` + `da3/avg_scale/`
  존재 / `0 <= s < e <= nframes` / `avg_scale` 유한 양수 / `prompt_camera` 비어있지 않음을
  검증하고 탈락 사유를 집계해 출력한다. `meta_<tag>.csv` 도 같이 뱉어(인덱스 빌드를 8배 아낌),
  `meta.csv` 에 없던 da3 scene 은 `filter_dl3dv.valid_scene` 을 **그대로 호출**해 통과분만
  추가한다(규칙 복제 안 함). 두 리스트 모두 셔플한다 — `base.py` 의 val 로더가 `shuffle=False`
  로 test 리스트 앞에서부터 자르기 때문.
  결과: 1K 1000 scene − blacklist 27 − 검증 탈락 16(`too_small` 15 / `teleport` 1) = **957
  scene / 6095 segment**, scene-disjoint 0.90/0.10 (seed 42) → train 861 scene / 5479 seg,
  test 96 scene / 616 seg.
  - c4d2k5y4(기존 리스트)는 segment 단위 `random_split` 이라 같은 scene 이 train/test 양쪽에
    걸쳐 있었다(`docs/known_issues.md`). da3 는 물려받을 비교 대상이 없어서 처음부터 scene 을
    겹치지 않게 나눴다 → **이 리스트 수치는 c4d2k5y4 기반 수치와 비교 불가**(held-out 이 더 어렵다).
- **`main/conf/experiment/da3_1k_textonly.yaml`** — 위 seg 리스트 + `meta_da3_1k.csv` 배선
  확인용 text-only arm. pose/caption/avg_scale 은 최상위 `transforms.json` 쪽을 쓰는 **GT pose
  arm** 이고, da3 예측 pose arm(`da3_1k_da3pose.yaml`)과 짝을 이룬다.
- **`cfg.pose_source`** (`main/conf/config.yaml`, 기본 `transforms`) — scene 의 pose /
  intrinsics / caption / avg_scale 을 어느 코퍼스에서 읽을지 고르는 `dataset_dl3dv` 분기.
  `transforms` 는 기존 동작 그대로(`<scene>/{transforms.json, prompts.json, avg_scale/}`),
  `da3` 는 `<scene>/da3/{pose.npz, prompts.json, avg_scale/}` 를 읽는다. 두 prompts.json 의
  세그먼트 경계와 키(`'0','1',...`)가 동일해서 seg 리스트·샘플 인덱싱 로직은 공유한다.
  구현: `resolve_pose_source()` + `_prompts_path()` / `_avg_scale_dir()` / `_scene_probe()` /
  `_parse_da3()`, 그리고 `_load_scene` · `_load_index` · `_load_index_subset` · `load_data` ·
  `_avg_scale` 의 분기. 인덱스 캐시 키에는 `transforms` 가 **아닐 때만** `__ps<source>` 가
  붙어서 기존 캐시 파일이 그대로 재사용된다.
  - `da3/pose.npz` 의 `extrinsics (N,3,4)` 는 **OpenCV w2c** 다 — `transforms.json` 과 달리
    GL→CV flip 도 역행렬도 적용하면 안 된다. 표본 12 scene 을 GT 와 Umeyama 정렬한 결과
    11개가 ATE/extent ≤ 0.004 · 회전 평균 ≤ 0.32°, 나머지 1개(`8a1b61638a`)가 0.194 / 3.49°.
    c2w 로 잘못 읽으면 회전 오차가 137~178° 로 튄다.
  - `intrinsics (N,3,3)` 는 504×280 픽셀 공간이고 `cx*2, cy*2` 가 상수라 `fx/(2cx)` 형태는
    그대로 성립한다. 다만 fx 를 프레임마다 따로 예측해서(scene 내 std/mean 7e-4~3e-3)
    `intr_norm: rel` 에서 `cam_param[9:11]` 이 정확히 `[1,1]` 이 아니라 `1.000 ± 0.003` 이 된다.
  - `frame_files` 는 이미지 디렉토리 정렬 순서로 만든다 (`da3/predictions.npz` 는 1000 scene
    중 3개에만 있어 파일명 소스로 못 쓴다; 정렬 순서가 `transforms.json` 의 `file_path` 정렬
    순서와 일치하는 것은 실측 확인).
- **`main/conf/experiment/da3_1k_da3pose.yaml`** — da3 예측 pose 로 도는 text-only arm.
  `da3_1k_textonly.yaml`(GT pose arm)과 `pose_source` 한 줄만 다르고 seg 리스트 / `meta_csv` /
  `scale_mode` / `intr_norm` / `cam_dim` 은 동일해서 그대로 짝 비교가 된다.
- **`scripts/data/cam_repr_w2c_vs_c2w.py`** — `cam_param[:, 6:9]`를 w2c translation `t = −Rc`로
  두는 현재 규약과, c2w translation(= camera center `c`)으로 두는 대안 규약 중 어느 쪽이
  회귀 타깃으로 더 쉬운지 재는 Step 0 진단. 학습 없이 `transforms.json`만 읽어
  (dataset 로드 없이, `norm_camera_length_stats.py` 방식) segment별로 두 지표를 뽑는다:
  `bleed = mean‖ΔR·c‖ / mean‖Δc‖`(w2c의 `Δt = −R_iΔc − ΔR·c_{i+1}` 중 두 번째 항, 즉 이동이
  없어도 회전만으로 t가 움직이는 "회전 bleed"의 상대 크기)와
  `ratio = curv_t / curv_c`(2차 차분 norm의 비, w2c 궤적이 c2w 궤적보다 얼마나 덜 매끄러운가).
  회전량(`th_path` 프레임간 geodesic 각 총합, `th_net` 첫↔마지막 순 회전각)과 `reach`(정규화 후
  max‖c‖)에 대한 상관까지 같이 계산해 메커니즘이 실제로 작동하는지 검증한다.
  출력은 `results/compare/cam_repr_w2c_vs_c2w/`에 `.png`(2×2: bleed/ratio 히스토그램 +
  bleed-vs-th_path / bleed-vs-reach 산점도), `summary.json`, `per_seg.npz`(segment 이름 + 7개
  지표 배열 — 이후 arm별 per-segment 오차와 join 하려고 남긴다).
- **`.vscode/settings.json`에 `git.scanRepositories`** — `camera_generation/tools/gaussian-splatting-lightning`
  (자체 `.git`을 가진 별도 repo, 부모 `.gitignore:221`의 `tools` 패턴으로 무시됨)을 VSCode
  Source Control 패널에 독립 repo로 띄운다. VSCode의 `git.repositoryScanMaxDepth` 기본값이 1이라
  워크스페이스 루트 기준 depth 3인 이 경로는 자동 탐지되지 않았다.
- **`docs/` 3편을 추적 대상으로 추가** — `geo_context_ablations.md`(geo context view sampling의
  trajectory leakage와 그 대응 ablation 설계 노트), `known_issues.md`(segment 단위 random_split
  으로 인한 scene-level val leakage 등 미해결 이슈), `vae_verification.md`
  (`vae_20260302_300.pth`, `cam_dim=64` 재구성 검증 결과). 리포지토리 루트 `CLAUDE.md`도 함께
  추적한다(지금까지 untracked였다).
- **`main/infer_swap_ablation.py`** — geo camera-DM이 geo context를 실제로 쓰는지 보는 swap
  ablation 추론. validation sample N개를 고정해 놓고 3가지 모드를 각각 돌려 저장한다:
  `normal`(자기 context + 자기 text) / `ctxswap`(anchor view0 = frame s만 남기고 나머지
  context view를 옆 sample에서 가져옴, cyclic `i → (i+1)%N`) / `textswap`(context는 그대로,
  text만 교체). 모델·VAE·geo encoder는 run의 Hydra experiment config 그대로 짓고
  (`experiment=<name>` override), ckpt/출력은 환경변수 `SWAP_CKPT` / `SWAP_OUT` / `SWAP_N` /
  `SWAP_TAG`로 준다.
- **`scripts/data/viz_scene_chunk_scale.py`** — 같은 scene 안에서 chunk가 바뀔 때 arm B
  (`scale_mode: geo_lagernvs`)의 divisor `D = 1.35·max‖c_geo − c_anchor‖`가 얼마나 흔들리는지
  top-down으로 본다. geo context가 chunk마다 새로 검색(frustum max-coverage)되므로 B는 scene
  단위 canonical scale이 아니라 **chunk 단위 scale**이라는 걸 보이는 게 목적. D는 학습 경로
  그대로(`CamDataset._sample_geo_frustum_cover` + `_geo_lagernvs_scale`) 계산한다.
- **`models/custom_geo_encoder.py` + `geo_encoder: custom`** — LagerNVS 대신 쓰는 자체 설계
  scene context encoder. **frozen DINOv2 ViT-L/14**(`checkpoints/dinov2-large`, `Dinov2Model`,
  hidden 1024) + **trainable `GeoTokenizer`**(8ch = Plücker 6 + log-depth 1 + valid 1)를 patch
  단위로 concat → `Linear(1024+256 → geo_latent_dim)` → `(B, V*P, 768)`. LagerNVS와 달리 카메라가
  view당 11-d `cam_token`이 아니라 **픽셀당 Plücker ray**로 들어간다.
  - DINO 입력은 DINO 규약대로: `preprocessor_config.json`의 `image_mean/std`로 정규화하고, H·W가
    patch(14) 배수가 아니면 내림해서 리사이즈한다(`geo_image_hw` (256,448) → (252,448) → 격자
    **18×32 = 576 tok/view**, V=6이면 M=3456). `interpolate_pos_encoding=True`로 518 이외 해상도 지원.
    DINO는 `requires_grad_(False)` + `torch.no_grad()` (실측 trainable 2.44M / frozen 304.37M).
  - geo mask는 **일단 전부 True**(depth 없는 view를 context에 섞지 않는다는 전제). pose prefix와
    RoPE는 넣지 않았다 — 현재 DM의 geo cross-attention이 평범한 MHA라 key에만 걸린 rotary는 상대
    위치가 되지 못하고 절대 위치 인코딩으로 degenerate하기 때문.
  - 기존 lagernvs / scenetok 경로는 그대로다. `GeoEncoder`에 `trainable` / `wants_batch` 플래그와
    `forward_batch(batch)` 진입점을 추가했고(custom은 `images` 외에 Plücker/depth/valid가 더
    필요해서 batch dict을 받는다), lagernvs는 `wants_batch=False`라 `forward(images, cam_token)`
    호출 경로가 바뀌지 않는다(회귀 확인: backend lagernvs / trainable False / proj Identity).
  - config: `custom_geo_dino_path`(null → `<ckpt_root>/dinov2-large`), `custom_geo_freeze_dino`,
    `custom_geo_input_hw`, `custom_geo_ray_dim`, `custom_geo_geo_dim`.
  - **아직 미완**: dataset이 `geo_plucker_map (B,V,6,H,W)` / `geo_logd` / `geo_valid`를 내보내지
    않는다(RGBD 학습 데이터 구성 대기). 학습 배선(optimizer에 geo 파라미터 추가, `no_grad` 분기,
    ckpt에 encoder state 저장/복원, geo latent cache 강제 OFF)도 아직 안 붙였다.
- **`scripts/viewer/viser_arms_gs.py`** — segment 하나를 DL3DV `scene.ply`(3DGS) 위에 올려놓고
  GT / arm별 pred / 각 arm의 context 카메라를 색이 다른 frustum으로 겹쳐 보는 viser 뷰어.
  `viser_val_cameras.py`는 run 하나를 훑는 브라우저라 arm 비교도 splat 표시도 안 돼서 새로 만들었고,
  `load_transforms` / `add_frustums` / `scene_chunk`는 그대로 재사용한다(기존 스크립트는 미수정).
  arm은 `--arm 'label:run_dir:R,G,B'`, context는 `--ctx 'label:run_dir:R,G,B[:leak_K]'`로 여러 개
  지정하고, `leak_K>0`이면 `_mix_inseg_context`의 `inseg + rest` 순서에 따라 앞 K장을 누수 view로
  따로 표시한다. 그룹마다 GUI 체크박스가 붙는다. ply 로더는 표준 3DGS 레이아웃(62 float props)을
  memmap으로 읽어 SH DC항만 색으로 쓰고 `--gs-max`(기본 600k) / `--gs-min-opacity`(기본 0.05)로 솎는다.
  - `frustum scale (x reach)` 슬라이더로 frustum 크기를 실시간 조절한다. 값은 target reach 대비
    비율이라 segment가 바뀌어도 뜻이 같고, 그룹별 상대 배율(pred/GT 1.0, context 1.3, scene cams 0.6)은
    유지된 채 한꺼번에 스케일된다. 초기값은 `--frustum-scale`.
  - frustum 그리기 방식을 `--frustum-style {spline,viser}`로 고른다. 기본 `spline`은
    `tools/gaussian-splatting-lightning/custom_utils/render_frustum.py`의 `add_frustum_spline`
    (= `custom_panel.py`의 `visualize_camera_frustum`이 쓰는 그 함수)로 frustum을 catmull-rom
    spline 8개(ray 4 + far-plane rect 4)로 그려 **선 두께**를 줄 수 있다. `viser`는 기존
    `add_camera_frustum` 경로 그대로(두께 인자 없음). gspl repo를 import 못 하면 경고 후 `viser`로 폴백.
  - `frustum thickness (rebuild)`(number, 초기값 `--frustum-thickness` 2.0)로 선 두께를 바꾼다.
    viser 1.0.30의 `SplineCatmullRomHandle` / `CameraFrustumHandle`에는 `line_width`가 설정 가능한
    prop으로 없어서(`scale`/`visible`/`position`/`wxyz`/`positions`뿐) 값이 바뀌면 frustum을
    `remove()` 후 다시 그린다. 드래그마다 전체 재생성이 도는 걸 막으려고 slider가 아닌 number다
    (`custom_panel.py`도 build 시점에만 thickness를 읽는다). 재생성 후 `frustum scale`과 체크박스·
    interval 상태는 다시 입힌다. `viser` 스타일에서는 disabled.
  - `frustum interval (traj)` 슬라이더(초기값 `--frustum-interval`)로 궤적 카메라를 N개마다 하나만
    표시한다. 솎아도 마지막 프레임은 항상 남겨 궤적 끝을 잃지 않는다. GT / pred / scene cams에만
    걸리고 context(6장)는 항상 전부 보인다.
  - **좌표계**: DL3DV `transforms.json`의 `applied_transform`(nerfstudio가 기록한 world 재정렬)을
    `c2w_ply = applied_transform4 @ c2w_gl`로 좌측 곱해야 `scene.ply`와 겹쳐진다 — 이걸 빼면 카메라가
    점군과 어긋난다. `tools/gaussian-splatting-lightning/custom_utils/custom_panel.py`의
    `get_cameras_from_transforms`와 같은 처리. 재정렬된 frame은 COLMAP world라 viser up을 `-y`로
    둔다(`--up`으로 덮어쓰기, `--no-applied-transform`으로 예전 동작 유지).
- **`scripts/eval/viz_diversity_topdown.py`** — 위 숫자의 top-down 그림. `--fig seeds`는 같은 context에서
  seed만 바꾼 궤적들(기본 `--style cloud`: 단색 빨강 + alpha로 구름 형태를 보여줌; `rainbow`는 seed 구분용)
  + GT + context 카메라를, `--fig swap`은 orig/donor context와 각각으로 뽑은 궤적을 겹쳐 그린다.
  좌표계는 target 첫 프레임 `s`의 카메라 프레임 anchor 후 X vs −Z로
  `viz_avgscale_context_topdown.py` / `topdown_swap.py`와 동일하고, 거리는 segment마다 GT reach로
  나눠 **점선 원 r=1이 GT 최대 도달반경**이 되게 했다. 패널 축은 `aspect='datalim'`이 `set_xlim`을
  무시하는 문제 때문에 정사각 bbox를 직접 잡아 `adjustable='box'`로 고정한다 — 안 그러면 멀리 있는
  context 마커가 축을 늘려 궤적이 납작해진다. 패널 선정은 `--pick even`(이름 정렬 후 균등, cherry-pick
  방지) / `dswap-top` / `dswap-low`. `--names-from`으로 다른 그림과 같은 segment를 고를 수 있다.
  - `--fig inseg` — `geo_test_inseg_k` 누수 arm(K=1/3/5)을 겹쳐 그린다. 누수된 context 카메라는
    저장돼 있지 않지만 `_mix_inseg_context`가 target 프레임을 `linspace(0,T-1,K).round()`로 균등
    분할하므로 GT 궤적의 그 인덱스로 결정적으로 복원해 마커를 찍는다.
  - `--ctx-leak-k K` — `--fig seeds`에서 context 앞 K장을 **in-segment 누수 view**(초록 다이아몬드)로,
    나머지를 out-of-segment(회색 사각)로 구분한다. `_mix_inseg_context`가 `inseg + rest` 순서로
    돌려주는 것에 의존한다.
- **`scripts/eval/traj_diversity.py`** — 조건부 diversity / context 민감도 측정. (A) 같은 context에서
  seed만 바꾼 여러 eval 출력(`--seeds`)을 받아 위치 분산을 seed 내(within) / segment 간(between)으로
  분해하고 ICC `R = var_between/(var_between+var_within)`, seed 쌍거리(APD), ADE(mean/best-of-S),
  bias(seed평균→GT) vs spread를 낸다. (C) `--base/--swap`으로 context swap arm 쌍을 받아 궤적 이동량
  `d_swap`을 **(A)의 seed noise 단위**로 환산하고, `geo_ctx/`가 있으면 예측 궤적↔context 카메라
  nearest-neighbour 거리와 context retrieval R@1(`--retrieval`)까지 낸다. 모든 거리는 segment마다
  GT reach `max_t‖T_gt(t)−T_gt(0)‖`로 나눠 무차원화한 뒤에만 segment를 가로질러 평균낸다 —
  segment별 스케일 차이가 통계를 지배하지 않게.
  - `--only <name...>` — 이 segment들만 쓴다. run마다 평가한 segment 수가 달라도(예: 400개 vs 2개)
    같은 집합으로 맞춰서 arm 간 비교가 되게. `viz_diversity_topdown.py --only`와 같은 뜻.
  - `--per-seg` — segment별 APD / ADE(mean·best·worst) / `pred_reach/GT` / GT reach를 그대로 출력.
    segment가 2~3개뿐이면 `var_between`·`icc`는 표본이 없는 거나 마찬가지라 집계값이 오해를 부른다.
- **`geo_swap_mode`** (`main/conf/config.yaml`, 기본 `null`) — **test 전용** probe. `geo_test_inseg_k`가
  누수 축(context를 target segment 안으로 밀어넣음)을 재는 것과 **직교**하게, context를 **같은 scene의
  다른 segment**로 옮긴다(`'inscene'`). donor는 이 scene의 `(순번 + geo_swap_shift)`번째 다른
  segment(cyclic)이고, segment가 하나뿐인 scene은 swap하지 않은 채 batch에 `geo_swapped=0`으로
  표시해 분석에서 뺄 수 있게 한다. 같은 scene·같은 world frame·실제 이미지·동일 `norm_scale`이라
  분포를 벗어나는 것이 없고 **context가 보는 영역만** 바뀐다. 읽는 법: 예측 궤적이 donor 영역을
  따라가면 model이 context 주도, 안 움직이면 text 주도.
  - `geo_swap_keep_first`(기본 `true`) — swap 후 view0를 **target의 프레임 `s`로 되돌린다**. anchor와
    그에 딸린 frame/scale 링크를 건드리지 않아, 변수가 나머지 `V-1`장의 context 내용으로 한정된다.
    `false`면 `V`장 전부 swap이라 `geo_first_view_target_s` ablation과 교란된다.
  - `geo_test_inseg_k`와 같은 이유로(캐시 키가 `data_name`뿐이라 context view 변화를 구분 못 함)
    켜지면 `geo_latent_cache_dir`를 강제로 끄고 로그로 알린다. 끄지 않으면 swap이 조용히 무효가 된다.
  - `null`이면 기존 동작과 byte-identical.
- **`geo_return_idxs`** (`main/conf/config.yaml`, 기본 `false`) — 분석 side-channel. 고른 context view
  인덱스(`geo_idxs`)와 그 c2w(`geo_ctx_c2w`, OpenCV)를 매 item에 붙여, 생성된 궤적이 context 카메라에
  얼마나 가까이 붙는지 오프라인으로 잴 수 있게 한다. **모델에는 절대 안 들어간다.** 기본 off인 이유는
  geo latent cache가 **hit**일 때 캐시가 건너뛰려던 frustum_cover 탐색을 다시 돌려야 하기 때문.
  `geo_cam_embed: null` 경로(캐시가 `geo_emb`만 넣고 바로 return하던 곳)에서도 동작하도록 재계산 분기를
  넣었다 — `frustum_cover`는 `(scene_idx, s, e)`에 deterministic이라 캐시가 쓴 선택을 그대로 복원한다.
  - `scripts/eval_testset.py`: 플래그가 켜져 있으면 `<out>/geo_ctx/<data_name>.json`에
    `{geo_idxs, geo_swapped, norm_scale, c2w}`를 궤적 json 옆에 쓴다. c2w는 `*_transforms_*.json`과
    같은 OpenGL 규약으로 맞춰서(Y/Z flip) 내보내므로 바로 같은 좌표계에서 비교된다.
- **`geo_cam_embed: plucker`** — geo token에 붙이는 camera embedding의 두 번째 종류. 기존
  `relfirst`(view당 11-d `[rot6d(6), trans(3), fx/2cx, fy/2cy]`를 그 view의 patch token 777개에
  broadcast)와 달리 **patch token당 6-d Plücker 광선** `[d(3), o×d(3)]`을 준다 — `d`는 그 패치
  중심을 지나는 광선의 방향, `o = -R^T t`는 카메라 중심, 둘 다 target segment 첫 카메라 `s`의
  프레임에서 같은 `norm_scale`로 나눈 단위(생성되는 궤적과 같은 단위). intrinsics 채널이 따로
  없다 — FoV가 이미 `d`에 들어 있다. `geo_cam_embed: null`(OFF) / `relfirst`는 그대로 동작한다.
  - `main/dataset_dl3dv.py`: `_geo_patch_grid()`(encoder의 resize 규칙 = 긴 변 518, 나머지는 14의
    배수로 내림 → `geo_image_hw (256,448)`이면 `21×37 = 777` tokens/view, `M = 6*777 = 4662`),
    `_geo_cam_plucker()`, 그리고 `relfirst`/`plucker`를 갈라 주는 `_geo_cam_cond()` 추가.
    `__getitem__`의 캐시 히트 경로와 on-the-fly 경로 둘 다 `_geo_cam_cond`를 부른다.
  - `main/train_latent_cam_dm.py`: `attach_geo_cam`이 4-D `(B,V,P,6)`이면 broadcast 없이
    `(B,M,6)`으로 reshape하고 `P`가 실제 `M//V`와 다르면 즉시 죽는다(어긋난 채 조용히 학습되는 걸
    막음). 모델 생성 시 `geo_cam_raw_dim`이 `plucker`면 6, 아니면 11.
  - `scripts/eval_testset.py`: 같은 분기 — plucker 체크포인트를 11-d MLP로 로드하지 않게.
  - 검증(실데이터): grid `(21,37)`, `geo_cam_param (6,777,6)`, `geo_emb (4662,768)`에서 `P=777`
    일치, `|d| ∈ [0.9999999, 1.0000001]`, `d·m` max `1.54e-08`, 패치 중심 재투영 오차
    `u 0.00092 px / v 0.00085 px`, `z>0` 전부 True, moment를 `-R^T t`로 다시 만든 것과 max abs
    diff `0.0`.
- **`main/conf/experiment/geo_worldtraj_camembed_plucker.yaml`** — `geo_worldtraj_camembed`에서
  `geo_cam_embed`만 `relfirst` → `plucker`로 바꾼 arm. resolved config diff 확인 결과 다른 키는
  `exp_name`과 `geo_cam_embed` 둘뿐이다. camera embedding은 캐시에 안 들어가므로 같은 latent
  cache 트리(`first_cam_not_included/`)를 그대로 쓴다.
- **`geo_test_inseg_k`** (`main/conf/config.yaml`, 기본 `null`) — **test 전용** probe. context
  `V`장 중 앞의 `K`장을 **target segment 카메라**(모델이 생성해야 할 프레임들)로 바꾼다.
  `K=1 → [s]`, `K=3 → [s, 중간, 마지막]`, `K=5 → [s, 1/4, 2/4, 3/4, 마지막]`이고 나머지 `V-K`장은
  원래 sampler가 고른 coverage view가 순서대로 채우므로 `V`는 그대로, `view0 == s`도 그대로다.
  `geo_worldtraj`는 학습 때 이미 target 카메라 하나(프레임 `s`)를 context에 넣으므로 **`K=1`은
  학습 조건과 완전히 동일한 대조군**이다. `null`/`0`이면 기존 동작과 byte-identical.
  - context view가 바뀌면 캐시 키(segment만 봄)가 못 구분하므로 `geo_test_inseg_k`가 켜지면
    `geo_latent_cache_dir`를 강제로 끈다(로그로 알림) — 즉 세 K 전부 on-the-fly LagerNVS로 돈다.
    부수 효과로 `K=1`은 "캐시 없이 재현되는가"까지 같이 검증한다.
  - `scripts/data/extract_geo_context.py`도 같은 `--set geo_test_inseg_k=K`를 받아 top-down
    시각화의 context 별표가 eval이 실제로 쓴 view와 일치한다.
- **`main/conf/experiment/geo_worldtraj_camembed_with_anchor.yaml`** — anchor(`geo_first_view_target_s`
  + `geo_cover_subtract_first`)를 켠 채 `geo_cam_embed: relfirst`를 얹는 2×2 ablation의 네 번째 칸.
  `geo_worldtraj_camembed`와는 anchor 2개 키만, `geo_worldtraj`와는 camembed 2개 키
  (+ `vae_latent_scale` 0.47637 → 0.96032625, 사용자 지시)만 다르다. `cam_dim`/`intr_norm`/
  `vae_ckpt_path`/`vae_latent_scale`은 `config.yaml` 기본값과 값이 같고 명시만 한 것.
- **`scripts/data/dump_geo_cache_idxs.py`** — geo latent cache가 어떤 view index로 만들어졌는지
  매니페스트로 덤프. `frustum_cover`가 deterministic이라 v1 캐시(index 미저장)도 재계산으로 복원한다.
  트리당 `<subdir>__geo_idxs.csv` + `.pt` 2개 파일만 쓴다(per-file sidecar 대신 — inode 절약).
  v2 트리 300 샘플에서 저장된 `geo_idxs`와 300/300 일치 확인.
- **`scripts/render/viz_avgscale_context_topdown.py`** — `avg_scale` 정규화로 LagerNVS를 돌렸을 때의
  붕괴를 진단하는 top-down 시각화. 기존 `<seg>/topdown_context_gt.png`가 raw world 단위라 범위 이탈
  여부를 볼 수 없던 것을 divisor로 나눈 좌표에서 그리고, context reach 0.7407 / target bound 1.0
  원을 겹쳐 `avg_scale` vs `geo_lagernvs`를 나란히 비교한다. `ranges.json` + `_summary.png` 동반.
- **`tools/lagernvs/render.py` (신규, gitignore된 vendored 트리라 커밋에는 없음 — 작업 트리에만
  존재)** — `render_avgscale.py` / `render_static_probe.py` / `render_pred_from_dump.py` 세 스크립트가
  env 변수와 하드코딩으로 나눠 갖고 있던 축을 하나의 argparse CLI로 합쳤다. 세 스크립트는 남겨 두므로
  기존 실행은 그대로 재현된다.
  - **context와 target의 정규화 분모를 따로 준다** — `--modes '<ctx분모>/<tgt분모>[@flag]...'`.
    `/` 없이 쓰면 예전처럼 divisor `D`(= 모든 카메라 중심에 걸리는 translation 분모) 하나를 양쪽에
    똑같이 적용하고 키·파일명도 이전과 같다(`avg_scale` → `render_avgscale.mp4`). 나뉜 경우만
    `c<ctx>-t<tgt>` 키(예 `clagernvs-tmaxd_tgt`). `metrics.json`에 `divisor_ctx` / `divisor_tgt` /
    `split_div` 추가. 지금까지 divisor 하나가 context baseline과 target 변위를 동시에 정해서
    자유도가 1개뿐이었던 것을 2개로 푼 것.
  - `--ctx {real,static,posedup}` + `--ctx-n N`으로 context 구성 통합: `static`이 기존
    `RD_STATIC_CTX=1`(= `render_static_probe.py`의 `dup6_consistent`, pose·intrinsics·이미지 전부
    view0 복사), `posedup`이 `dup6_contradict`(이미지는 진짜 V장, pose만 view0 복사), `--ctx-n 1`이
    기존 `RD_SINGLE_VIEW=1`. `--traj {tgt,pred}`로 `render_pred_from_dump.py`의 예측 궤적 렌더도 흡수.
  - **분모 이름 `lagernvs_orig`** — `--ctx static` / `--ctx-n`으로 context를 바꿔치면 `lagernvs`
    분모가 바뀐 context 기준으로 다시 계산되는데, 덮어쓰기 전 값(진짜 context가 만들었을
    `1.35*max||c_ctx - c_ctx0||`)을 `lagernvs_orig`로 남긴다. "context는 정지시키되 target은 원래
    분모로 정규화"(`--modes one/lagernvs_orig`) 같은 조합에 필요하다.
  - **`@ch9div` flag** — `cam_token` `(V,11)`의 채널 9(`camera_scale`)에 "정규화 후 context reach"
    대신 **분모 자체**를 넣는다. 정지 context는 context translation이 정확히 0이라 채널 9가 어떤
    분모를 써도 clamp 하한 1e-06으로 눌리는데, 그 자리에 실제로 나눈 scene scale을 알려주는 변형.
  - `@pt` / `@q` / `@qe` / `@ch9nat` / `@ch9avg` / `@g1` flag, `*<f>` 배수, `tok<v>` 합성 분모,
    `metrics.json` 병합, cross-mode 비교 그리드는 `render_avgscale.py`와 동일. `@q`/`@g1`은 ctx/tgt
    분모에 각각 독립으로 적용되고 어느 쪽이 발동했는지 로그에 찍는다.
  - env 변수(`RD_ROOT` `RD_MODES` `RD_CKPT` `RD_SIZE` `RD_TAG` `RD_EPS` `RD_BINS` `RD_DMIN`
    `RD_STATIC_CTX` `RD_SINGLE_VIEW`)를 CLI 기본값으로 그대로 읽어서, env만 쓰면 호출 방식까지
    `render_avgscale.py`와 같다.
  - 재현 검증(2 seg, `lagernvs_general_512`, 512px, GPU1): `--modes lagernvs` 20.328 / 22.668 dB,
    `--ctx static --modes one` 17.772 / 16.039 dB, `--ctx-n 1 --modes one` 18.998 / 19.751 dB —
    셋 다 `render_avgscale.py`의 `lagernvs` / `one_st` / `one_sv`와 소수점 3자리까지 일치.

### Changed
- **`main/dataset_dl3dv.py`: `CamDataset.__getitem__` 에서 target 쪽 블록을 `_target_out()` 으로
  분리** — `SDCamDataset` 이 **분모만 다르고** cam_param 규약(`intr_norm` / `trans_repr` /
  `normalize_camera_extrinsics_and_points`)은 완전히 같아서, 복사본을 두면 한쪽만 고치는 사고가
  난다. 분리 전후 DL3DV item 0/137/5000 의 전 키를 비교해 **bit-identical** 확인
  (`experiment=da3_7k_customgeo_nos` 기준). `seg_key()` staticmethod 도 같이 추가 —
  seg-list id <-> `data_name` 변환을 dataset 클래스가 갖게 해서 `base.py` 가 코퍼스를 몰라도 된다.
- **`scripts/data/cache_da3_depth.py`: `LAYOUT` 분기 (`dl3dv` 기본 | `clipdir`)** —
  Scene-Decoupled-Video-dataset 은 da3 를 **clip 단위**로 돌려서 scene 아래 trajectory 7개가
  각각 자기 `depth.npz` 를 갖는다 (DL3DV 는 scene 당 하나). `clipdir` 은 meta csv 없이 ROOT 를
  두 단계 스캔해 `<scene>/<clip>/depth.npz -> <OUT>/<scene>/<clip>.npy` 로 푼다.
  `LAYOUT` 기본값이 `dl3dv` 라 **기존 호출은 동작이 그대로**다 (meta_da3_7k.csv 로 재실행 시
  `{'ok': 0, 'skip': 2}` 확인). 캐시 내용은 `np.array_equal(npy, npz['depth'])` 검증 통과.
- **`scripts/data/make_da3_splits.py`: 다중 batch + 이미지/포즈 개수 검사**
  - **`--batches 1K 2K ... 7K`** (+ 필수 `--tag`) — 여러 DL3DV batch 를 **한 코퍼스**로 묶는다.
    scene-disjoint 분할은 batch 경계를 무시하고 전체에서 **한 번** 한다 (batch 별로 나눈 뒤
    합치면 분할 비율이 batch 크기에 끌려간다). `--batches` 를 안 주면 기존 `--batch` 단일
    동작 그대로다.
  - **`img_pose_count_mismatch` 필터** — `scan_scene` 이 이미지 수 == da3 pose 프레임 수를
    확인한다. `blacklist.csv` 의 `filter_len_mismatch`(13건)는 **최상위 `transforms.json`
    기준**이라 da3 pose 에 대해서는 아무것도 보장하지 않는다. 어긋나면 프레임 인덱스가 조용히
    밀려 **잘못된 (이미지, 카메라) 짝**으로 학습된다. 실측 1K~7K 는 7000/7000 통과라 지금은
    아무것도 안 걸러내지만, 이후 batch 를 추가할 때를 위한 방어다.
- **`main/train_latent_cam_dm.py`: `cfg.ckpt_at_epochs` 저장** — `best.pth` 블록 직후,
  `resume.pth` 앞에서 해당 epoch 이면 `epoch<N>.pth` (+ geo 가 학습 대상이면 `epoch<N>_geo.pth`)
  를 저장한다. 기본값이 빈 리스트라 **기존 run 은 저장 파일 구성이 바뀌지 않는다.**
- **`main/train_latent_cam_dm.py`: 학습 중 geo cross-attn 계측** (`geo_attn_probe` /
  `_geo_attn_figure` 신규, `run_validation` 안에서 `cfg.log_geo_attn` 이 켜졌을 때만 호출).
  `cfg.log_geo_attn` 기본이 `false` 이고 probe 실패는 `try/except` 로 삼키므로
  **기존 경로는 그대로**다 (계측이 학습을 죽이면 안 된다).
  - `layers[li][5].attn` (geo cross-attn) 에 forward hook 을 걸어 layer 별
    **`resid_ratio = ‖a‖/‖x‖`** 를 잰다. attention weight 만으로는 부족하다 —
    `CrossAttention.forward` 는 `norm(x + a)` 이고 attention 은 softmax 라 **행 합이 항상 1**
    이어서, 모델이 geo 를 무시해도 attention map 은 멀쩡해 보인다. 실제로 얼마나 섞이는지는
    residual 크기로만 보인다.
  - 같이 올리는 스칼라: `attn_entropy_nats` / `attn_entropy_norm`(uniform 대비),
    `view_mass_v{i}`(view 별 attention 질량, uniform = 1/V), `view_max_over_uniform`,
    `dpred_shuffle`(= geo 를 roll 했을 때 예측이 얼마나 바뀌는지, 상대 norm).
  - attention map figure 는 **6개 view 전체에서 잡은 공통 `vmin`/`vmax`** + `inferno` +
    공유 colorbar (사용자 결정 2026-08-07). panel 마다 autoscale 하면 view 간 밝기 비교가
    지워지고, uniform 대비 상대값(diverging)은 outlier min 0.48 / max 3.64 에 씻겨나간다.
    그림 안 텍스트는 **영문만** — DejaVu Sans 에 한글 glyph 가 없어 두부(□)로 나온다.
- **`main/train_latent_cam_dm.py`: 학습되는 geo encoder 배선** (`getattr(geo_encoder,
  'trainable', False)` 로 분기 — 기존 frozen backend 는 전부 그대로 `no_grad` 경로).
  - `geo_encode()` 상단에 `wants_batch` 분기 추가: custom backend 는 view latent 가 아니라
    배치 전체(`images`, `geo_plucker_map`, `geo_logd`, `geo_valid`)를 받아 `forward_batch` 로 간다.
    키가 빠지면 어떤 키인지 찍고 `KeyError`.
  - 옵티마이저가 `model.parameters() + [p for p in geo_encoder.parameters() if p.requires_grad]`
    를 함께 잡는다 (**258 model + 22 geo tensor / 2.44 M param**, 실측 exp_avg 전부 non-zero).
    `geo_encoder.train()` 은 **부르지 않는다** — `SceneEncoder.__init__` 이 frozen DINO 를
    `eval()` 로 내려놓는데 부모에서 `train()` 을 부르면 재귀적으로 되돌아간다(GeoTokenizer 는
    GroupNorm/LayerNorm 뿐이라 train/eval 이 no-op).
  - train/val 양쪽 호출부를 `with accelerator.autocast():` 로 감쌌다. geo encoder 는
    `accelerator.prepare` 에 넘기지 않으므로(우리는 `forward` 가 아니라 `forward_batch` 를 부르고,
    DDP 래핑이 그 메서드를 가린다) 이렇게 하지 않으면 `mixed_precision="bf16"` 이 적용되지 않는다.
  - 체크포인트: `last_geo.pth` / `best_geo.pth` + `resume.pth['geo']`. **frozen 키를 제외**하고
    저장한다 — 전체를 저장하면 DINOv2 ViT-L 때문에 epoch 당 ~1.2 GB 다. 실측 **9.76 MB**,
    `dino` 키 없음 확인.
- **`scripts/eval_testset.py`: 학습된 geo encoder 가중치 복원.** `trainable` backend 면
  `<ckpt>_geo.pth` 를 `strict=False` 로 로드하고(missing = frozen DINO), 파일이 없거나
  unexpected key 가 있으면 랜덤 가중치로 평가하는 대신 죽는다.
- **`main/conf/experiment/da3_1k_{textonly,da3pose}.yaml`: `epochs: 150` 을 yaml 에 명시.**
  기존 DM run 들은 default `epochs: 2000` 으로 띄운 뒤 `scripts/stop_at_epoch.sh 150` watchdog
  으로 멈췄다. 두 arm 은 watchdog 없이 config 만으로 같은 지점에서 끝나게 한다.
- **`main/conf/experiment/da3_1k_{textonly,da3pose}.yaml`: `coverage_blacklist_path` 를
  `null` → `data/seg_blacklist_jump_da3_1k.csv`.** 두 arm 이 **같은** 파일을 가리킨다 —
  세그먼트 집합이 어긋나면 GT pose arm 과 da3 pose arm 의 paired 비교가 깨지기 때문.
  실측 결과 두 arm 모두 6,095 → **5,972 sample / 957 scene** (scene 손실 0). 인덱스 캐시 키에
  `__cb<파일명>` 이 들어가므로 이전 캐시와 충돌하지 않고 새로 빌드된다.
- **`scripts/eval/viz_diversity_topdown.py`: 카메라 시선 화살표 + `--only` + leak 라벨.**
  - `--arrow-every N` / `--arrow-scale`(기본 0.16, GT reach 단위): `--fig swap`에서 N 프레임마다
    카메라 forward를 같은 top-down 평면(X vs −Z)에 투영해 그린다. dump된 `transform_matrix`가
    OpenGL c2w라 forward = `−R[:,2]`이고, **정규화하지 않으므로 위/아래를 보는 카메라일수록
    화살표가 짧다**(tilt가 눈에 보이도록 일부러). context 카메라는 6장뿐이라 항상 전부 그린다.
    기본 `0`(= 끔)이라 예전 그림은 그대로 나온다.
  - `--only <seg> ...`: 패널로 그릴 segment를 직접 지정(`--pick` / `--n` 무시). 목록에 없는
    이름을 주면 바로 에러.
  - `--label-set leak` + `--swap-leak-k K`: `--fig swap`을 context-swap이 아니라 `geo_test_inseg_k`
    누수 arm 비교로 읽게 하는 범례/제목 세트. swap 쪽 context 앞 K장(`_mix_inseg_context`가
    `inseg + rest` 순으로 넣는다)을 마름모 마커로 갈라 그린다.
- **DL3DV 루트 경로를 `/data1/cympyc1785/data/DL3DV/scenes`로 통일** — 옛 경로
  `/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K`는 **이제 존재하지 않아서** 하드코딩한
  스크립트가 전부 죽는다. 16개 스크립트의 `ROOT` / `--root` 기본값을 바꿨다
  (`scripts/context_select/*` 7개, `scripts/coverage/*` 5개, `scripts/data/filter_dl3dv.py`,
  `scripts/data/norm_camera_length_stats.py`, `scripts/render/render_target_from_context.py`,
  `scripts/viewer/viser_val_cameras.py`). `main/train_vae_dl3dv.py`는 docstring의
  "DL3DV-960" 표기만 "DL3DV"로 정정(코드 변화 없음).
- **`scripts/data/extract_geo_context.py`에 argparse 추가** — `--testdir` / `--out` / `--experiment`
  / `--split` / `--set K=V`(hydra override, 반복 가능). 인자 없이 돌리면 예전 160 target 기본값
  그대로. `eval_testset.py`로 뽑은 전체 held-out 3,980 target처럼 기본 `seg_list`가 아닌 집합을
  쓸 때 필요하다.
- **`scripts/render/compare_textonly_vs_worldtraj.py`: `_contact.png`에도 geo context 카메라 표시.**
  per-target PNG에만 찍히던 magenta 별(context view의 world center)을 contact sheet 타일과 하단
  legend에도 추가. 그리고 `summary()`가 run 1개일 때 빈 비교 산점도 axis를 만들지 않는다
  (`ncmp = len(labels) - 1`; 예전에는 항상 1개가 비어 남았다).
- **`main/dataset_dl3dv.py`: v1 geo latent cache를 `geo_cam_embed` arm에서도 쓴다.** 기존에는 v1 캐시
  (bare `(M,768)`, `geo_idxs` 미포함)가 `geo_cam_embed`와 만나면 on-the-fly LagerNVS forward로
  폴백해 스텝당 ~12배 느려졌다. `geo_view_sampling == 'frustum_cover'`이고 `geo_shuffle_order`가
  꺼져 있으면 `_sample_geo_frustum_cover`를 재계산해 폴백을 막는다 (extrinsics와 `(s,e)`만 보는
  deterministic greedy라 캐시를 만든 그 선택이 그대로 재현된다). shuffle이 켜져 있으면 캐시된
  emb의 view 순서와 어긋나므로 기존대로 폴백. 검증: 실제 `__getitem__` 40개에서 캐시 히트 40/40,
  on-the-fly 호출 0회.
- **`results/topdown_3way_textonly_worldtraj_camembed/`** — `plots_last_full3980/` 추가.
  학습 중 `test/` 스냅샷 160 target(epoch 64/44/40) 대신 각 run의 `last.pth`로 전체 held-out
  3,980 target을 재추론한 `eval_my/*__last__c4d2k5y4/`를 쓴다. 기존 `plots/`는 지우지 않고 보존하고
  README를 두 섹션으로 재구성. text-only는 ep105라 두 geo arm(ep149)과 여전히 어긋난다는 점 명시.

- **`tools/lagernvs/render_avgscale.py`에 GenDoP식 scale guard/양자화 + 정지 context 모드**
  (**gitignore된 vendored 트리라 커밋에는 없음, 작업 트리에만 존재**). 기본값(`RD_MODES=avg_scale`,
  `RD_STATIC_CTX=0`, `RD_TAG=''`)이면 출력 파일명까지 이전과 byte-identical.
  - `parse_mode`를 `@pt` 전용에서 flag 집합(`@pt`/`@q`/`@qe`, 조합 가능)으로 일반화. `@q`/`@qe`는
    divisor에 GenDoP `core/provider.py:170-182`의 처리를 그대로 적용한다 — clamp가 아니라 덧셈
    guard `div + RD_EPS`(기본 1e-5), log10 공간에서 `bin = long((log10(div_g)+2)/4·BINS)`를
    `[0, BINS]`로 clip(`RD_BINS` 기본 256 → 표현 가능 범위 `div ∈ [1e-2, 1e2]`), 역양자화는 `@q`가
    `10**(·)`(자기일관), `@qe`가 `exp(·)`(GenDoP `infer.py:221` 그대로 = log10로 encode하고 exp로
    decode하는 왕복 불일치, `div**0.4343`). `metrics.json`에 `div_raw/div_guard/bin_raw/bin/div_q`
    기록.
  - `RD_STATIC_CTX=1`: context view V개를 전부 view0의 복사본(pose·intrinsics·이미지 모두)으로
    바꿔 `div → 0` 축퇴 케이스를 실제로 렌더한다. 이때 `lagernvs` divisor는 dump에 적힌 값이 아니라
    바꿔친 context에서 다시 계산한다. `RD_TAG`가 모든 출력 파일명/metrics 키 뒤에 붙어 일반 context
    실행 결과를 덮어쓰지 않는다.
  - 로그/`metrics.json`에 `tgt_reach`(정규화 후 target 최대 변위)와 `divisor_raw` 추가.
  - `RD_SINGLE_VIEW=1`: context를 view0 한 장으로 줄인다(V=1). `build_cam_cond`이
    `num_cond_views==1 and split=="test"`에서 토큰을 `[ch9=0, ch10=1]`로 강제하므로
    (`data/normalization.py:109-114`) divisor가 정규화 의미를 잃고 **target 변위 크기만** 결정하게
    된다 — 정지 카메라 fallback에서 "target 범위를 얼마로 줘야 하나"를 재는 용도.
  - mode 문법에 `*<f>` 배수 추가(`avg_scale*0.5` → key `avg_scale_x0.5`)와 합성 divisor `one`(=1,
    정규화 없이 world unit 그대로). divisor sweep을 위한 것.
  - 로그의 `tok=`는 이제 추정값이 아니라 `cam_tokens[0, -2:]` 실제값을 찍는다.
  - **`@ch9nat` / `@ch9avg` flag 추가** — `camera_scale`(cam_token 채널 9, 이하 ch9) 토큰을 pose와
    분리해 원인을 가른다. `data/normalization.py:51`에서 ch9는 **이미 정규화된 pose로부터 파생**
    (`max‖c_ctx‖` of normalized poses)되므로 divisor만 바꾸면 pose와 ch9가 항상 같이 움직여
    둘 중 무엇이 붕괴 원인인지 분리가 안 된다. `@ch9nat`은 pose를 그대로 두고
    `cam_tokens[:, -2]`만 native `1/1.35`로, `@ch9avg`는 `avg_scale` divisor가 만들었을 값
    (`maxctx0 / avg_scale`)으로 덮어쓴다. `FLAGS` 집합에 추가한 것뿐이라 미지정 시 동작 불변.
  - **`@g1` flag + `RD_DMIN` 추가 (divisor guard)** — divisor `D`(= 모든 카메라 중심에 걸리는
    translation 분모, `render_avgscale.py:201`)가 `RD_DMIN`(기본 `1e-2`) 아래로 내려가면
    카메라 기반 정규화를 포기하고 `D = 1`(world unit 그대로)로 폴백한다. 정지 context에서
    `D = 1.35·max‖c_ctx − c_ctx0‖ → 0`이 되어 pose가 5~6자릿수 밖으로 튀는 축퇴를 막는 것
    (실측: `D = 6.6e-07`, `tgt_reach = 8.1e+06`, 13.671 dB → guard 후 `D=1`, 17.772 dB).
    정상 context에서는 발동하지 않아 `lagernvs@g1`이 `lagernvs`와 PSNR/SSIM/LPIPS 전부 동일.
    임계값은 정상 D 분포(실측 3.31~43.8)와 축퇴(1e-7 수준) 사이가 5자릿수 비어 있어 둔감하다.
    `metrics.json`에 `guard_fired` / `d_min` 기록. 주의: `D`는 scene의 world unit이라 절대
    임계값은 scene scale에 의존한다 — scale-free 판정은 `parallax = maxctx0/avg_scale`가 맞다.
  - **합성 divisor `maxd_tgt` 추가** — `maxd_seg / 1.35 = max‖c_tgt − c_tgt0‖`. `maxd_seg`가
    1.35배라 `tgt_reach`(정규화 후 target 최대 변위)가 0.7407이 되는 것과 달리 **정확히 1.0**이
    된다. "target을 0~1로 정규화"가 필요할 때 쓴다. `maxd_seg`와 마찬가지로 target 유래라 GT leak.
  측정값은 `EXPERIMENTS.log` 2026-08-03 항목들.
- **`scripts/render/compare_norm_video.py`에 `--layout rows` + 파일명 해석 수정.**
  - `_sfx()` 헬퍼 — `SFX` dict로만 찾던 것을 접두 치환으로 일반화. `render_avgscale.py`의
    `suffix()`가 mode 안의 `avg_scale`만 `avgscale`로 줄이고 뒤에 `RD_TAG`를 붙이기 때문에,
    tag/flag가 붙은 키(`avg_scale_st`, `avg_scale_sv`, `avg_scale_ch9nat` …)를 dict가 놓쳐
    `render_avg_scale_st.mp4`(없는 파일)를 찾고 있었다.
  - `--layout rows`: `--rows`에 `;`로 구분한 mode 목록을 주면 목록 하나가 격자 한 줄이 되고 각
    줄의 0번 칸은 GT다 (R × (1+M)). 같은 분모 집합을 **context 구성만 바꿔** 비교할 때 쓴다
    (예: 정지 복사 context vs single view). `--labels`로 열 이름, `--row-tags`로 줄 이름을 직접
    준다. 기존 `row`와 달리 PSNR 곡선 열은 붙지 않는다. `--out`에 seg가 여럿이면 파일명에 seg를
    덧붙이고, `--stack`으로 seg들을 세로로 더 쌓을 수도 있다.
- **`scripts/vae/vae_scale_matrix.py`에 `TAILS=1`** — 기존 표는 corpus **평균** latent std만
  주는데, camera-length `scale_mode`는 카메라가 멈춘 바로 그 샘플에서 분모가 0에 가까워지므로
  평균으로는 터지는 개별 샘플이 안 보인다. `TAILS=1`이면 같은 행들 아래에 per-sample tail 표를
  덧붙인다: 샘플별 `max|cam_param[6:9]|`, `max|latent|`, translation recon L1의 median/p99.9/max.
  기본값 `TAILS=0`이면 출력·동작 모두 이전과 동일. 측정값은 `EXPERIMENTS.log` 2026-08-02 항목.
- **`scripts/data/static_camera_degeneracy.py` (신규)** — camera-length divisor(arm A
  `ctx_longer_135max` / arm B `geo_lagernvs`)가 **정지 카메라**에서 무너지는 지점을 실측한다.
  기존 `scripts/data/norm_divisor_compare.py`는 `maxd < 1e-6` segment를 `return None`으로 버리고
  scale-free 비율(`m = maxd/D`, `r = D_lagernvs/D`)만 보고해서 이 현상이 구조적으로 안 보인다 —
  divisor 자체가 카메라 변위라, 변위를 그걸로 나누면 물리적으로 아무리 안 움직였어도 항상 ~0.74가
  나오기 때문. 이 스크립트는 chunk를 하나도 버리지 않고 **scale-free가 아닌** 양을 찍는다:
  `D_raw`와 `clamp(1e-5)` 적중률, `reach = maxd/D` vs `maxd/avg_scale`(조작된 운동량),
  `depth = avg_scale/D`(LagerNVS 입력 반경), `parallax = maxctx0/avg_scale`(divisor 무관 물리량),
  `straight = maxd/pathlen`, geo view 중복률, 그리고 **arm A가 실제로 쓰는 divisor**
  (`dataset_dl3dv.py:876`의 target 유래 `_first_farthest_scale` fallback 포함). DynamicVerse는
  subset별로 쪼개고 DL3DV는 control. out → `results/compare/<OUT_NAME>/{per_chunk.csv,stats.json,
  summary.md,_static.png,worst_parallax.csv}`. 측정값은 `EXPERIMENTS.log` 2026-08-02 항목.
- **`conf/experiment/geo_worldtraj_{ctxlonger135,lagernvsnorm}_scale96.yaml` (신규)** — arm A/B와
  전부 동일하되 `vae_latent_scale`만 **0.96032625**. 끝난 baseline
  `20260731_001511_dl3dv_geo_worldtraj`(wandb `c4d2k5y4`, ep150)가 그 값으로 학습됐는데, 도는
  A(0.58341)/B(0.53716)는 각 (ckpt, scale_mode, intr_norm) triple의 FULL-corpus 실측 latent std로
  고쳐 박은 값이라 **diffusion 입력 std**라는 교란변수가 하나 더 있었다. 이 arm은 그 상수를
  baseline에 맞춰, baseline 대비 `scale_mode`(+A'는 `geo_lagernvs_skip_ctx_norm`과 전용 geo cache)만
  다르게 만든다. hydra resolved config diff로 검증:
  `A→A'` / `B→B'` = `exp_name` + `vae_latent_scale` 뿐, `c4d2k5y4→A'/B'` = `scale_mode`(+A'의 2개)
  및 이름이 바뀌거나 이후 추가된 키(`geo_anchor_first_frame`→`geo_cover_centered_at_s` 둘 다 true,
  `geo_cam_embed` null, `train_frac` 0.9 = 당시 실효 기본값) 뿐.
  단서: 0.96032625는 우리 corpus 실측값이 아니라 SCVideo 원본 corpus 상수다. A'/B'의 입력 std는
  각각 0.6075 / 0.5594배로 눌리고 baseline은 0.4961배 — 눌림 정도까지 같지는 않다(`scale_mode`가
  divisor를 바꾸는 이상 불가능). 이 arm이 맞추는 건 "같은 상수를 쓴다"는 조건이다.
  `vae_latent_scale`은 `geo_emb`에 관여하지 않으므로 geo latent cache는 bit-exact 재사용.
- **`scripts/eval_testset.py` (신규)** — 학습이 끝난 run을 **held-out split 전체**에 대해 추론 +
  CLaTr/caption 평가한다. `train_latent_cam_dm.run_validation`은 `val_max_batches`(20) ×
  `batch_size`(8) = 160개만 보므로 wandb val 곡선은 160-sample 추정치다. 이 스크립트는 같은
  경로를 3982개 전부에 대해 한 번 돌린다.
  - config는 **run이 저장한 `results/<run>/config.yaml`**(학습 당시 resolved config)을 현재
    `conf/config.yaml` 기본값 위에 덮어써서 만든다. 지금 끝난 세 geo run은 전부
    `vae_latent_scale: 0.96032625`로 학습됐는데 `conf/experiment/geo_worldtraj.yaml`은 이제
    0.47637이라, live experiment yaml을 읽으면 조용히 틀린 scale로 평가된다.
    이름이 바뀐 키는 `RENAMED`로 승계한다(`geo_anchor_first_frame` → `geo_cover_centered_at_s`).
  - split은 `base.Trainer._make_batch_generator()`를 그대로 재사용 → 학습 때의
    `random_split(seed=42)` held-out set과 bit-identical(wandb가 본 160개의 superset).
  - sampling/decode/`out_to_trajectory`/json dump/CLaTr subprocess 체인은 복사가 아니라
    `train_latent_cam_dm`의 헬퍼를 import해서 module-global `cfg`만 rebind → 두 경로가 갈라질 수 없음.
  - 출력: `eval_my/<run>__<ckpt>/`에 `config.yaml`(실제 사용값), `eval_meta.json`
    (ckpt/epoch/wandb_id/argv), `test/`(caption + ref/pred transforms = input), `seq/`·`token/`,
    `preds.npy`, `metrics.json`, `metrics_full.json`, `preds_scores.csv`, `losses.json`.
- **`scripts/eval_testset_queue.sh` (신규)** — `<run_dir>:<ckpt>` 목록을 한 GPU에서 순차 실행
  (`GPU=3 scripts/eval_testset_queue.sh a:best.pth a:last.pth`). full pass 1회 ≈ 18분.
- **`scripts/vae/vae_scale_matrix.py`에 `DATASET` 분기 (`dl3dv` 기본 = 기존 동작 그대로 /
  `dynamicverse` / `both`)** — `dynamicverse_shim.load_dynamicverse`로 DynamicVerse까지
  같은 (ckpt × scale_mode × intr) 행렬을 잰다. `DV_CHUNKS`(기본 3)로 scene당 chunk 수 지정.
  `MAX_SCENES=none|null|0`이면 FULL corpus(기존엔 큰 정수를 넣어야 했음). 요약줄에
  `DATASET/META/MAX_SCENES`를 같이 찍는다.
  `vae_20260302_300` @ `intr_norm: rel`, FULL corpus 실측:

  | scale_mode | DL3DV (39817) | DynamicVerse (1818) | 합침 (41635) |
  |---|---|---|---|
  | `avg_scale` | 0.47637 | 0.10070 | 0.46632 |
  | `ctx_longer_135max` | 0.58341 | 0.42224 | 0.57732 |
  | `geo_lagernvs` | 0.53716 | 0.32543 | 0.52968 |

  DynamicVerse sample은 scene당 겹치지 않는 `[k·T,(k+1)·T)` chunk 3개라 DL3DV의 sliding-window
  segment와 corpus 정의가 다르다 — 합침 열은 그 개수로 가중된 값이다.
  DV의 `avg_scale` 행 0.10070이 극단적으로 작다: point-cloud avg_scale(장면 depth)이 카메라
  이동보다 훨씬 커서 translation이 거의 0으로 눌린다(recon L1 trans 0.00131 = 전 cell 최소이지만
  표적이 작아서지 잘 맞춰서가 아니다). context 기반 분모에서도 DV가 DL3DV보다 작다(0.42/0.33).
  즉 **DV를 섞어도 SCVideo 상수 0.96032625는 재현되지 않는다** — 그 상수는 dynpose-100k를 포함한
  원본 corpus 값이고 우리 데이터로는 설명되지 않는다.
  지금 3-arm은 DL3DV-only 학습이라 config엔 DL3DV 열을 쓴다(변경 없음).
- **`scripts/relaunch_after_stop.sh` (신규)** — `stop_at_epoch.sh`가 멈춘 run을 같은
  screen/GPU에서 자동 재기동한다(`<experiment> <screen> <gpu> <log>`). 프로세스 소멸 →
  `GRACE`(60s) → GPU 메모리 반납 확인 후 기동하므로 watchdog의 `stop_one`과 경합하지 않고,
  DataLoader worker가 GPU를 물고 있는 상태에서 뜨지 않는다. 이전 로그는 timestamp 붙여 보존
  (watchdog이 그 로그로 epoch을 읽으므로 새 파일이어야 함). 기동 후 `stopwatch` watchdog을
  재시작한다(`RESTART_WATCHDOG=0`으로 off) — 기존 watchdog은 멈춘 run을 `done_map`에 박아둬서
  새로 띄운 run을 다시 안 보기 때문.
- **`train_frac` (신규 config, 기본 0.9 = 기존 동작 그대로)** — `base.Trainer._make_batch_generator`의
  random split train 비율. `train_seg_list`/`test_seg_list`가 있으면 그 경로가 우선이라 영향 없음.
  `1.0`이면 index 전체를 train으로 쓰고 val은 비운다 — validate를 아예 안 하는 job 전용
  (`train_vae_dl3dv.py`는 `include_val=False`). 두 VAE experiment를 `train_frac: 1.0`으로 돌려
  1K~7K 전 segment 39817개를 쓴다(기존 90% split은 35776개 = 559 it/epoch → 622 it/epoch).
- **`scale_mode: ctx_longer_135max` (신규)** — `dataset_dl3dv._first_farthest_context`.
  target segment를 제외한 **긴 쪽 = context range**를 겹치지 않는 `num_frames`(49) window로 자르고,
  window마다 `1.35·max‖camera center − 그 window 첫 camera center‖`를 구해 **window 평균**.
  target view가 분모에 전혀 들어가지 않아 leak-free이고, 추론 때 context view만으로 재현된다.
  기존 offline 정의(`scripts/vae/vae_divisor_recon.py:99-103`)와 30 segment 대조 mismatch 0.
  canonicalization(39817 segment, blacklist 후, `m = max‖target view center − frame s‖ / D`):
  mean 0.7693 / std 0.2937 / min 0.0299 / max 5.186 / med 0.7661, sd(log10 m) 0.1864,
  p95/p05 4.04, CV 0.3818 — leak-free 후보 중 최고(정규화 없음 CV 0.4265, `geo_lagernvs` 0.4231,
  `ctx_side_135max` 0.5078).
- **`vae_ctxlonger135.yaml` / `vae_geolagernvs_wt.yaml` (신규 VAE experiment)** —
  normalization을 바꾸면 target cam_param 분포가 바뀌므로 각 arm 전용 CameraVAE를 다시 fit한다.
  scope·hyperparameter는 `vae_worldtraj`(latent_std 0.569379)와 동일(cam_dim 64 / intr_norm raw /
  meta_worldtraj.csv / batch 64 / lr 1e-4 / 60 epoch / beta 1e-3)이라 latent_std가 직접 비교된다.
  `vae_geolagernvs_wt`는 `geo_cover_*` 플래그를 `geo_worldtraj.yaml`에서 그대로 복사한다 —
  `_geo_lagernvs_scale`이 같은 greedy frustum-cover 선택을 다시 돌려 분모를 만들기 때문에
  이게 어긋나면 VAE와 diffusion이 다른 분모로 학습된다.
- **`geo_worldtraj_ctxlonger135.yaml` / `geo_worldtraj_lagernvsnorm.yaml` (신규 experiment)** —
  **context view 정규화와 target view 정규화를 하나의 분모로 통일**하는 2-arm.
  `geo_worldtraj` 대비 normalization 외 차이 없음(context 선택 / geo_posed / meta / 모델 동일).
  - A `geo_worldtraj_ctxlonger135`: 양쪽 다 `ctx_longer_135max`.
    `geo_lagernvs_skip_ctx_norm: true`로 LagerNVS 자체 context 분모
    `1.35·max‖ctx center − ctx view0 center‖`를 override_scale로 대체.
    cam_token이 바뀌므로 **geo latent cache를 새로 빌드**해야 한다
    (`geo_latent_cache_dir: /data1/cympyc1785/data/DL3DV/latent_cache_ctxlonger135`).
  - B `geo_worldtraj_lagernvsnorm`: 양쪽 다 LagerNVS 분모. `scale_mode: geo_lagernvs`만 켜고
    `geo_lagernvs_skip_ctx_norm`은 **false 그대로**. `geo_first_view_target_s`가 context view0을
    frame s로 고정하므로 `_geo_lagernvs_scale`과 `build_cam_token` 내부 `scene_scale`이 같은 수
    (40 segment 직접 비교 mismatch 0/40) → cam_token 불변 → **기존 `first_cam_included` 캐시 그대로 유효**.
- **`geo_worldtraj_decoupled.yaml` (신규 experiment) — `geo_worldtraj_camembed`에서 camera
  embedding만 뺀 ablation.** resolved config 기준 차이는 `geo_cam_embed: relfirst → null`
  단 하나(그 외 전부 동일, `diff`로 확인). 3-way:
  `geo_worldtraj`(first_view_target_s **true** / cam_embed off / geo_proj 768) vs
  **THIS**(false / off / 768) vs `geo_worldtraj_camembed`(false / relfirst / 896).
  즉 `THIS` vs `geo_worldtraj` = context view0를 target 첫 프레임에 고정하던 link를 끊은 비용,
  `camembed` vs `THIS` = 11-d pose가 그 비용을 얼마나 되사는지.
  **캐시 재생성 없음** — `geo_cam_embed`는 `_sample_geo_frustum_cover`를 건드리지 않고
  `geo_cam_param` 텐서만 추가하므로(`dataset_dl3dv.py:917-919`) context view 선택이 camembed run과
  bit-identical. 캐시 강제 off 상태의 실제 `__getitem__` 300 세그먼트 검증: cached `geo_idxs`
  300/300 일치, mismatch 0, unusable 0, `view0 == frame s` 0/300.
  `first_cam_not_included/`를 그대로 재사용한다.
- **`ctx_side_135max` divisor** (`main/dump_avgscale_render.py`) — windowing 없이 segment 밖
  **긴 쪽 전체**에 대해 `1.35·max‖c − c_side0‖`. `ctx_longer_135max`(D2, 윈도 평균)의 대조군.
- **`context_longer` divisor** (`main/dump_avgscale_render.py`) — 이미 구현되어 있는 동명의
  `scale_mode`(`dataset_dl3dv._cam_dist_mean_context`)를 렌더 테스트에도 추가.
  `ctx_longer_135max`와 **같은 윈도**를 쓰되 집계만 `1.35·max` 대신 `mean‖c − c_win0‖`.
  이걸로 렌더 스윕이 latentcam이 지금 당장 학습 가능한 leak-free divisor를 전부 덮는다.
- **`scripts/data/norm_divisor_compare.py` (신규)** — divisor 후보들을 **canonicalization 관점**에서
  비교. 1200 segment(전부 서로 다른 scene)에 대해 `m = max‖c_t − c_s‖ / D`(diffusion이 회귀해야
  하는 정규화된 도달거리)와 `r = D_lagernvs / D`(LagerNVS native 단위 대비 편차)를 계산하고,
  `sd(log10 m)` / `p95/p05` / `m>1` / `m<0.1` / `corr(log D, log maxd)`를 표로 낸다.
  출력: `results/compare/norm_divisor_compare/{stats.json, summary.md, per_segment.csv,
  _divisors.png}`. env: `N` / `CACHE_EXP` / `SPLIT` / `SEED`.
- **`scripts/data/norm_divisor_compare.py`: `ONE_PER_SCENE` / `WORKERS` / `OUT_NAME`** —
  **기본값(`ONE_PER_SCENE=1, WORKERS=1, N=1200`)은 기존 동작 그대로** (1200 segment 결과 재현 확인).
  `ONE_PER_SCENE=0`이면 scene당 1개로 줄이지 않고 split의 **전 segment**(39,830 / 6,097 scene)를
  집계하므로 scene이 실제 학습에서 보이는 빈도대로 가중된다. `WORKERS>1`은 그 경로를 fork pool로
  병렬화(`_sample_geo_frustum_cover`가 결정론적 greedy라 샤딩해도 결과 불변; 32 workers 기준
  전수 ~2분). `N=0` = 무제한. `OUT_NAME`으로 출력 폴더 분리.
  `stats.json`에 `n_scenes` / `one_per_scene` / `maxd_log10_sd` 추가.
  결과: `results/compare/norm_divisor_compare_all/`.
- **`scripts/vae/vae_divisor_recon.py` (신규)** — divisor를 바꾸면 **frozen camera VAE**가 아직
  멀쩡한지 확인. 400 segment에 대해 `dataset_dl3dv.__getitem__`과 동일하게 `cam_param`을 divisor별로
  재구성해 encode→decode 하고 `in_trans_std` / `lat_std_{raw,scaled}` / `vae_scale_for_1` /
  `rec_trans_{norm,world}` / `rec_rot6d`를 보고한다. `rec_trans_world`가 divisor 간 비교 가능한
  값이고, `vae_scale_for_1`은 그 divisor로 갈아탈 때 써야 할 `vae_latent_scale`이다.
  출력: `results/compare/norm_divisor_compare/vae_recon.json`.
- **`tools/lagernvs/render_static_probe.py` (신규; gitignore된 vendored 트리라 커밋에는 없음,
  작업 트리에만 존재)** — "context 카메라가 정지하면 렌더가 되는가"를 divisor 문제와 분리해서 측정.
  healthy segment의 `render_inputs.pt`를 재사용해 target 궤적은 그대로 두고 context만 4가지로
  다시 만들어 렌더한다: `ctrl6`(실제 6-view) / `single`(ctx0 하나, `num_cond_views=1` +
  `split='test'` → `build_cam_cond`가 `camera_scale=0, world_points_scale=1` 토큰 발행) /
  `dup6_consistent`(ctx0 이미지·포즈를 6배 = 물리적으로 정지한 카메라) /
  `dup6_contradict`(서로 다른 6장 이미지에 ctx0 포즈만 강제 = 포즈가 깨진 scene 재현).
  `single` vs `dup6_consistent`가 OOD 토큰 쌍 `(0,0)`의 비용을, `dup6_consistent` vs
  `dup6_contradict`가 pose-image 모순의 비용을 각각 분리한다.
  env: `RD_ROOT` / `RD_CKPT` / `RD_SIZE` / `RD_NSEG` / `RD_OUT` / `RD_CONDS`(렌더할 조건 부분집합) /
  `RD_VIDEO=1`(조건별 mp4 + `GT|cond1|cond2|...` 가로 concat mp4 — 프레임 그리드로는 안 보이는
  시간축 drift 확인용).
  출력: `results/compare/static_ctx_probe/{metrics.json, <seg>.png}`,
  영상은 `results/compare/static_ctx_video/`.
- **`scripts/data/scan_duplicate_poses.py` (신규)** — COLMAP 등록 실패로 **다수 프레임이 한 좌표에
  박혀 있는** scene을 전수 검출. scene마다 camera center를 `TOL`(기본 1e-4, scene extent 상대) 격자에
  버킷팅해서 최대 클러스터의 비율 `dup_frac`과 그 안의 최장 **연속** 구간 `dup_run`을 낸다.
  `dup_frac >= FRAC`(기본 0.10)이면 flag. env: `TOL` / `FRAC` / `WORKERS`(기본 32) / `CACHE_EXP` /
  `OUT_NAME`. 출력: `results/compare/scan_duplicate_poses/{per_scene.csv, flagged.csv,
  blacklist_rows.csv, stats.json}` — `blacklist_rows.csv`는 `<dl3dv_root>/blacklist.csv`에 그대로
  append 가능한 형식.
  train 6,097 scene 결과: **flagged 2개**, `dup_frac` p50 0.0030 / p99 0.0061 / p99.9 0.0120 /
  max 0.5815. 1·2위(0.582, 0.257)와 3위(0.034) 사이가 7.5배로 벌어져 경계가 깨끗하다.
- **`scripts/render/compare_norm_video.py`: `--layout row`, `--modes`** — 모든 mode를 가로 한 줄로
  붙인 비교 영상, 그리고 렌더할 mode 부분집합 선택. `ORDER`에 `ctx_side_135max` 추가.
- **정규화 ablation 렌더 파이프라인 (LagerNVS 자체 normalization 비활성화 + PSNR + 비교 영상)** —
  latentcam의 후보 divisor들을 LagerNVS 렌더로 검증하는 3단계. 기본값은 전부 기존 동작 유지.
  - `main/dump_avgscale_render.py`: `OUT_NAME`(출력 폴더), `ONE_PER_SCENE=1`(기본 0 = 기존처럼
    첫 N개 sample 그대로; 1이면 서로 다른 N개 scene), `tgt_image_paths`(PSNR용 GT 프레임),
    `scales` dict를 dump에 추가. `scales`는 네 divisor —
    `lagernvs`(=LagerNVS 자체 `1.35·max‖ctx center − ctx0‖`, control),
    `avg_scale`(현재 latentcam 학습값), `maxd_seg`(target 유래 → leak, 상한),
    `ctx_longer_135max`(segment 밖 긴 쪽을 `num_frames` 윈도로 나눠 `1.35·max`의 평균, leak 없음).
  - `tools/lagernvs/render_avgscale.py` (**gitignore된 vendored 트리라 커밋에는 없음, 작업 트리에만
    존재**): `RD_ROOT` / `RD_MODES`(기본 `avg_scale` → 출력 파일명까지
    이전과 동일)로 같은 segment를 여러 divisor로 렌더하고, `tgt_image_paths`가 있으면
    PSNR/SSIM/LPIPS를 `metrics.json`(mode별 평균 + `psnr_per_frame`)에 기록. 2개 이상 mode면
    `render_norm_compare.png`(행=mode+GT)도 생성.
  - `scripts/render/compare_norm_video.py` (신규): 위 mp4들을 하나로 합성. `--layout grid`(기본,
    2×3: GT / lagernvs / avg_scale / maxd_seg / ctx_longer_135max / PSNR 곡선+프레임 커서),
    `--layout pair`(mode별 `GT | render` width concat). 타일마다 divisor·해당 프레임 PSNR·평균
    PSNR 라벨.
- **`geo_cam_embed` — per-context-view camera embedding concatenated onto the geo tokens**
  (default **`null` = off**, geo conditioning byte-identical to before). With `'relfirst'` the
  dataset builds, for each context view `v`, an 11-d pose relative to the TARGET segment's FIRST
  camera `s` — `rel = w2c_v @ inv(w2c_s)`, `trans /= norm_scale`, parametrized exactly like
  `cam_param` (`rot6d = rel[:3,0] ++ rel[:3,1]`, `trans`, `fx/2cx`, `fy/2cy`) — and the training
  loop broadcasts it over that view's patch tokens, so `geo_emb` goes `(B, V·P, 768)` →
  `(B, V·P, 779)`. The model (`CameraDiffusionModel._lift_geo_cam`) splits the trailing 11 dims
  off, lifts them through a **trainable** MLP (`11 → geo_cam_embed_dim → geo_cam_embed_dim`,
  default 128) and re-concatenates, so `geo_proj` sees `768 + 128 = 896`. Carrying the raw dims
  inside `geo_emb` means no call-site signature changed and the frozen LagerNVS half stays
  cacheable. Translations use the target's `norm_scale`, i.e. the same units as the trajectory
  being generated; context intrinsics stay **raw** (under `intr_norm: rel` the target's own intr
  channels are ~[1,1], so raw is the only way context FoV reaches the model).
  Verified: `geo_proj.in_features` 896 vs baseline 768, params 64.905M vs 64.821M (Δ = 83,584 =
  exactly the MLP + wider `geo_proj`), broadcast checked per view, forward → `(B, 13, 64)`.
- **`conf/experiment/geo_worldtraj_camembed.yaml`** — the ablation that decouples the geo context
  from the camera being generated: `geo_first_view_target_s: false` (no target-segment frame is a
  context view at all, so all `geo_cover_k = 6` views are retrieved) +
  `geo_cover_subtract_first: false`, keeping `geo_cover_out_of_seg` / `geo_anchor_first_frame` /
  `geo_posed` and LagerNVS's own `1.35·max‖center‖` context normalization. `geo_cam_embed:
  relfirst` is what replaces the lost frame alignment. Its cache resolves to
  `first_cam_not_included/`, disjoint from `geo_worldtraj`'s `first_cam_included/`.
- **geo latent cache v2 format** — `cache_geo_embeddings.py` now writes
  `{'emb': (M,768) fp16, 'geo_idxs': (V,) int16}` instead of a bare tensor, and the dataset emits
  `geo_idxs` on the geo path. Storing the selected context frames lets a cache hit rebuild
  `geo_cam_param` without redoing the greedy coverage search. **Both formats are read**: v1 bare
  tensors still work for runs with `geo_cam_embed` off; a v1 file under a `geo_cam_embed` run
  falls back to the on-the-fly path rather than guessing. Round-trip verified bit-exact
  (`emb maxdiff 0.000000`, `geo_cam_param maxdiff 0.00000000`, 3/3) and cache-ON decodes no images.
- **`geo_latent_cache_dir` + a cache-read path in `dataset_dl3dv` / `train_latent_cam_dm`** — the
  precomputed frozen geo latents built by `cache_geo_embeddings.py` were being written but never
  read (nothing in the training code loaded them; `load_saved_pc_embeds` is the point-cloud flag
  and is unrelated). Now `cfg.geo_latent_cache_dir` (default **`null` = off**, i.e. the original
  per-step LagerNVS forward is completely unchanged) makes `CamDataset.__getitem__` read
  `<dir>/<first_cam_included|first_cam_not_included>/<iK>/<data_name>.pt` and return it as
  `geo_emb`, short-circuiting context-view selection and image decoding entirely; the trainer's
  new `geo_emb_from_cache()` rebuilds the all-ones mask (lagernvs marks every token valid,
  `geo_encoder.py:139`) and skips the encoder at both the train and validation call sites.
  A segment missing from the cache falls through to the on-the-fly path, so a partial cache is
  safe. Verified **bit-exact**: cache-ON vs cache-OFF `geo_emb` for the same segments gives
  `maxdiff 0.000000`, 3/3, and cache-ON carries no `images` key.
  Guard: this is only valid because `GeoEncoder.proj` is `nn.Identity` when the lagernvs native
  dim 768 equals `geo_latent_dim` (confirmed at runtime, `proj: Identity`). With
  `geo_latent_dim != 768` proj is a **trainable** `Linear` whose output must not be frozen into a
  file, so the dataset refuses the cache and prints why.
  Speed, measured on the full corpus (39830 samples, `bs8`, 1 GPU): cache **ON 4.16–4.22 it/s**
  (~17.8 min/epoch) vs **OFF 2.85 s/it = 0.351 it/s** (~3h33m/epoch) — **~12×**. A smoke-scale
  `max_scenes=40` comparison had shown *no* difference (ON 1.82/1.81 vs OFF 1.87/1.85 it/s); that
  measurement is invalid because 40 scenes' context images all fit in the OS page cache.
- **`conf/experiment/geo_worldtraj.yaml`: `scale_mode` `cam_dist_mean` → `avg_scale`** — matches
  the `textonly_savedscale_bs8/bs32` runs so geo-vs-text-only differ only in the `geo_*` keys.
  The geo latent cache stays valid: the geo path reads raw `self.extrinsics_list` for both view
  selection and the posed `cam_token` (`dataset_dl3dv.py:860-870`) and `geo_lagernvs_skip_ctx_norm`
  is false, so `build_cam_token` gets `override_scale=None` (`train_latent_cam_dm.py:74-76`) —
  `geo_emb` never sees `scale_mode`, no re-caching needed. Cost: with `avg_scale` + `intr_norm rel`
  this VAE measures latent std 0.44696, so `vae_latent_scale 0.96032625` gives a diffusion input
  std of 0.4654 (~2.15× small) instead of `cam_dist_mean`'s near-unit 1.034 — deliberately the
  same off-unit input as the text-only arms.
- **`conf/experiment/geo_worldtraj.yaml`: `geo_latent_cache_dir` enabled** — points at
  `/data1/cympyc1785/data/DL3DV/latent_cache`, whose `first_cam_included/` tree was built from
  exactly this experiment's context selection (`frustum_cover` + `out_of_seg` + `subtract_first`
  + `anchor_first` + `posed`). All 39830 segments present (missing 0 / extra 0, 266 GB). Set to
  `null` to force the original per-step LagerNVS forward.
- **`cache_geo_embeddings.py`: per-`{i}K` output layout, multi-batch runs, resumability, ETA** —
  the DL3DV scene root is now split into per-1000 batch dirs, so the cache mirrors it. New
  `CACHE_LAYOUT` selects `batch` (default, `<OUT>/<first_cam_*>/<iK>/<name>.pt`, with the
  `first_cam_*` level derived from `geo_first_view_target_s` because that flag changes which
  context views are encoded) or `flat` (**the original pilot layout, byte-for-byte unchanged**).
  `CACHE_BATCH` now takes one batch, a comma-separated list, or `all`; batches run in order,
  each resumable (an already-present file is skipped without a forward pass). Progress lines now
  report seg/s and ETA, and `CACHE_BS` exposes the loader batch size.
- **`conf/experiment/vae_dl3dv_1_7k.yaml`** — camera-VAE re-fit on the full DL3DV corpus the
  diffusion runs actually use (`meta.csv`, no coverage blacklist → **39830 segments / 6097
  scenes**), replacing the 1K-only fit behind the active default ckpt
  `vae_20260202_065659_400.pth` (SCVideo fit that on DL3DV 1K with one 49-frame window per scene,
  ~1000 samples). Deliberately a **drop-in**: `cam_dim: 32` + `scale_mode: avg_scale` +
  `intr_norm: raw`, so the resulting `state_dict` is key- and shape-identical to the old ckpt
  (verified) and swapping it changes only the VAE's training corpus. `batch_size 64`, `lr 1e-4`,
  `60 epochs`, `vae_beta 0.001`. Its `train/latent_std` (also written to
  `my_checkpoints/vae_dl3dv_1_7k/latent_std.txt`) becomes the matching `vae_latent_scale` —
  `0.4467666` belongs to the 1K-only fit and must not be reused with it.
- **`scripts/vae/vae_scale_matrix.py`: `vae_dl3dv_1_7k` added to the `CKPTS` matrix** (as
  `../my_checkpoints/vae_dl3dv_1_7k/last.pth`, 32-d) so the new fit is measured on exactly the
  same basis as the two reference ckpts. First measurement (1264 segments, `avg_scale`): latent
  std **0.60396** with `raw` (→ `vae_latent_scale` ≈ 0.60–0.62, and `0.4467666` would give input
  std 1.3518), recon L1 rot 0.00564 / trans 0.00900 / intr 0.00485, vs the 1K-only ckpt's 0.46312
  and 0.00466 / 0.00495 / 0.00293. The `raw` ≪ `rel` gap on `intr_L1` (0.00485 vs 0.25884) is
  preserved, i.e. the new ckpt is still a `raw` ckpt and still a strict drop-in.

- **`conf/experiment/textonly_savedscale_bs8.yaml` + `textonly_savedscale_bs32.yaml`** — the two
  arms of the batch-size comparison as self-contained configs instead of `experiment=
  textonly_savedscale batch_size=N` CLI overrides. Byte-identical apart from `exp_name` +
  `batch_size` (verified by resolving both and diffing all 19 relevant keys), and `exp_name`
  carries the batch size because the wandb run name is `<timestamp>_<exp_name>`
  (`train_latent_cam_dm.py:183` + `:194`), so each arm lands as its own wandb run without relying
  on the launcher remembering an override.
- **First `vae_dl3dv_1_7k` fit completed** (60 epochs, 39830 segments, GPU 2):
  `epoch 59 loss=0.01186 latent_std=0.635126`; ckpt at `my_checkpoints/vae_dl3dv_1_7k/last.pth`.
  Measured against the 1K-only ckpt on one basis (1264 segments, `avg_scale`): latent std
  **0.60943** `raw` (vs 0.46312), recon L1 rot **0.00562** / trans **0.00953** / intr **0.00433**
  (vs 0.00466 / 0.00495 / 0.00293), and `raw` ≪ `rel` on `intr_L1` (0.00433 vs 0.25666) so it is
  still a `raw` ckpt and a strict drop-in. Its matching `vae_latent_scale` is **0.635126** (the
  training-corpus std; `0.4467666` with this ckpt would give input std 1.3641). **Not adopted** —
  `config.yaml` still points at `vae_20260202_065659_400` + `0.4467666`; no diffusion run has used
  the new ckpt yet, so the recon-vs-corpus-coverage trade is unevaluated.

### Fixed
- **`scripts/render/compare_norm_video.py`의 `_sfx()`가 중간에 낀 mode key를 놓치던 문제.**
  `m.startswith('avg_scale')`일 때만 치환해서, `render.py`의 `@ch9=<분모>` flag가 붙은
  `'one_ch9-avg_scale_stfix'`처럼 `avg_scale`이 **문자열 중간**에 오는 key는 파일명을 못 찾았다.
  `suffix()`와 똑같이 `m.replace('avg_scale', 'avgscale')` 전역 치환으로 바꿨다.
- **camembed arm 2개의 train/test split이 비교 대상과 어긋나 test set이 새던 문제.**
  `conf/experiment/geo_worldtraj_camembed_plucker.yaml` /
  `geo_worldtraj_camembed_with_anchor.yaml`에 `train_seg_list` / `test_seg_list`를
  `latentcam_{train,test}_seg_list_c4d2k5y4.txt`로 못 박았다. 둘 다 `null`이면 `base.py:87`의
  `random_split(seed=42, frac=0.9)`이 split을 정하는데 그 permutation은 **dataset index 길이에
  의존**한다. `blacklist.csv`가 2026-08-01 02:24에 바뀌어(`duplicate_camera_centers` 2 scene 추가)
  index가 39,830 → 39,817로 줄자 split이 통째로 재섞였고, 비교 대상인 c4d2k5y4(geo_worldtraj)의
  held-out 3,983개 중 **3,561개(89.4%)가 이 arm들의 TRAIN으로 들어갔다**. seg list를 주면
  c4d2k5y4의 split을 그대로 재현한다(train 35,837/35,847, val 3,980/3,983 — 빠진 3개는 새로
  blacklist된 scene의 segment).
- **`scripts/data/extract_geo_context.py`가 geo latent cache가 채워진 트리에서 전 target을
  건너뛰던 문제.** `ds[idx]['geo_c2w']`를 읽었는데 cache hit이면 `__getitem__`이
  `dataset_dl3dv.py:960-976`에서 조기 return하고 `geo_c2w`는 on-the-fly 경로(`:1011`)에서만 붙는다
  — 그래서 3,980개 전부 "no geo_c2w"로 빠지고 마지막 `len(next(iter(out.values())))`에서
  `StopIteration`으로 죽었다. 필요한 건 context view의 world center뿐이라 sampler
  (`_sample_geo_frustum_cover` / `_sample_geo_hybrid`)를 직접 부르고
  `inv(extrinsics_list[scene_idx][geo_idxs])`에서 뽑는다 — 이미지/latent 로딩도 건너뛰어 훨씬 빠르다.
  재실행: 3,980 targets, missing 0, 6 views/target.

### Added
- **`geo_lagernvs_anchor: s | view0`** (`conf/config.yaml`, `dataset_dl3dv._geo_lagernvs_scale`) —
  `scale_mode: geo_lagernvs`의 divisor를 **어느 카메라에서 재는가**. 기본 `'s'`는 기존 동작
  (`D = 1.35·max‖c_geo − c_s‖`, `centers[s]` 하드코딩), `'view0'`은 `geo_idxs[0]` 기준.
  왜: `geo_encoder.build_cam_token`은 `override_scale`을 받든 말든 항상 `geo_idxs[0]`을 원점으로
  재정렬한 뒤 나눈다(`models/geo_encoder.py:109-110`). 그래서 LagerNVS가 실제로 받는 스칼라
  `tok = max‖c_geo − c_geo[0]‖ / D`가 학습 분포값 `1/1.35 = 0.7407`이 되려면 D도 같은 카메라에서
  재야 한다. `geo_first_view_target_s: true`면 `geo_idxs[0] == s`라 두 앵커가 같은 카메라 —
  기존 arm B가 `tok = 0.7407`을 공짜로 얻던 이유가 이것이고, 이 경우 두 설정은 bit-identical이다
  (실측 n=40, `max|diff| = 0`; 도는 `geo_worldtraj_lagernvsnorm_scale96` 영향 없음).
  `false`(camembed/decoupled)면 `s`가 context에 아예 없는데도 `centers[s]`에서 재게 되어
  `r = 1.35·max‖·−c_geo[0]‖/D`가 흩어진다: min 0.0420 / p05 0.5039 / med 1.0997 / p95 1.8912 /
  max 1.9985 (상한 2는 `c_geo[0]`이 집합의 원소라 삼각부등식에서 나오는 hard bound, 하한은 없음;
  대조군 `geo_worldtraj`는 `|r−1|≤1e-4`가 100.00%). `'view0'`이면 `r ≡ 1` — 실측으로
  camembed×B에서 `tok`이 `s`일 때 min 0.2377 / med 0.8163 / max 1.4646 (0.7407 일치 0.00%)에서
  `view0`일 때 **min=med=max=0.7407, 일치 100.00%**로 바뀐다.
  leak 없음: geo view 선택은 frame s와 out-of-segment 프레임만 보므로 추론 시 동일하게 계산되고,
  target 궤적의 원점은 여전히 `extrinsics[0] = c_s`이며 분모 스칼라만 바뀐다.
  `geo_shuffle_order: true`와의 병용은 `ValueError` — divisor는 셔플 **전** 순서로 계산되는데
  encoder는 셔플 **후** view0을 보게 되어 앵커가 다시 어긋나므로, 조용히 잘못 앵커되느니 거부한다.
- **`conf/experiment/geo_worldtraj_camembed_{ctxlonger135,lagernvsnorm}.yaml` (신규)** — camembed
  arm(`geo_first_view_target_s: false` + `geo_cam_embed: relfirst`) 위에 arm A/B의 normalization
  통일을 얹은 2개. 도는 camembed(wandb `ek9n9jt5`)는 `scale_mode: avg_scale` + `skip_ctx_norm: false`
  라서 **context decoupling**과 **normalization 미통일**이라는 변수를 동시에 갖고 있었고, baseline
  대비 낮게 나온 이유를 둘 중 어느 쪽으로도 돌릴 수 없었다. 이 2개는 뒤쪽을 제거한다.
  - `_ctxlonger135`: `scale_mode: ctx_longer_135max` + `geo_lagernvs_skip_ctx_norm: true`.
  - `_lagernvsnorm`: `scale_mode: geo_lagernvs` + **`geo_lagernvs_skip_ctx_norm: true`** +
    **`geo_lagernvs_anchor: view0`**. 원래 arm B가 `skip_ctx_norm: false`인데도 통일이 됐던 건
    `geo_first_view_target_s: true`라 `geo_idxs[0] == s`여서 LagerNVS의 분모와 `_geo_lagernvs_scale`이
    **같은 값**이 됐기 때문이다(값의 우연, 구조 아님). camembed에선 `geo_idxs[0] ≠ s`라 성립하지
    않으므로 `true`로 `override_scale=D`를 강제하고, 앵커까지 `view0`으로 맞춰 `tok ≡ 0.7407`을
    되찾는다 (아래 `Added`의 `geo_lagernvs_anchor` 항목).
  - A쪽 알려진 부작용: LagerNVS는 항상 `geo_idxs[0]`을 원점으로 재정렬한 뒤 나누므로
    (`models/geo_encoder.py:109-117`) 실제 주입 스칼라 `tok = max‖c_geo − c_geo[0]‖ / D`가 고유값
    `1/1.35 = 0.7407`에서 벗어난다: p01 0.1895 / med 1.0556 / max 3.2284 (native 대비 0.26x~4.4x).
    B쪽과 달리 이건 camembed 탓이 아니다 — `ctx_longer_135max`의 D는 context side를 num_frames
    window로 잘라 **각 window의 첫 프레임** 기준 `1.35·max`를 구한 뒤 **평균**한 값이라
    (`_context_window_scale`) LagerNVS가 재는 양과 프레임 집합·원점·평균 셋 다 다르고,
    `geo_first_view_target_s`와 무관하게 애초에 같아질 수 없다. `geo_lagernvs_anchor`로 고칠 수
    있는 종류가 아니다(고치려면 D 자체를 버려야 함).
    대가의 크기는 2026-08-02 랜더 실측 기준 작다: dynamic_replica tok 0.3308/0.7407/2.5309 → PSNR
    22.670/22.777/22.741 (0.11 dB 이내), 최대 격차는 DAVIS/lucia_0 tok 5.127(native 6.9배)에서
    −0.87 dB. 방향은 over-scaling 쪽이 더 나빴고 under-scaling이 더 위험하다는 근거는 없다
    (그 랜더에서 크게 망가진 변수는 tok이 아니라 `depth = avg_scale/D` 폭발이었고, DL3DV는
    `plx<1e-2`가 0.000이라 그 구간에 들어가지 않는다).
  - cache: 둘 다 `first_cam_not_included/`. A쪽은 `override_scale`이 arm A와 같아 기존
    `latent_cache_ctxlonger135` 트리를 공유하고 그 안에 서브디렉토리만 추가(~266 GB).
    B쪽은 **새로 구울 필요가 없다** — `anchor: view0`의 D는 정의상 LagerNVS native `scene_scale`과
    같은 식(둘 다 geo view 집합을 `geo_idxs[0]` 기준으로 놓은 `1.35·max‖center‖`)이라 실측
    `|native − D_view0|/native`가 n=60에서 max 5.955e-07 / med 8.969e-08 이고, 캐시가 fp16
    (eps 9.77e-04)이라 저장 정밀도보다 3자리 아래다. context view 선택 키
    (`geo_first_view_target_s`/`geo_cover_out_of_seg`/`geo_cover_subtract_first`/
    `geo_cover_centered_at_s`/`geo_view_sampling`)도 `geo_worldtraj_camembed`와 전부 같으므로
    이미 구워져 있는 `latent_cache/first_cam_not_included`를 그대로 쓴다(camembed·decoupled와 공유;
    `geo_cam_embed`는 캐시에 안 들어가고 `geo_idxs`로부터 매번 재구성된다). 재사용 검증: 캐시된
    `geo_idxs` vs 새로 계산한 선택 200개 — missing 0 / v1 0 / 불일치 0. 굽다 만 전용 트리
    `latent_cache_geolagernvs`(36 GB)는 삭제했다. `skip_ctx_norm: true`는 그래도 유지 — 결과 emb은
    native와 같지만, 통일을 "값의 우연"이 아니라 구조로 못박기 위해서다.
  - `vae_latent_scale`은 config 기본값 0.96032625 그대로(= `c4d2k5y4` / `ek9n9jt5`와 동일),
    `train_seg_list`/`test_seg_list`는 `c4d2k5y4` 분할로 고정.

### Changed
- **`conf/experiment/geo_worldtraj_{ctxlonger135,lagernvsnorm}_scale96.yaml`에 `train_seg_list` /
  `test_seg_list` 고정** — `base.py:_make_batch_generator`의 `random_split`은 partition이
  `len(dataset)`에 의존해서, `meta_csv`나 blacklist가 바뀌면 `random_seed: 42`가 같아도 분할이 통째로
  재배치된다. 실측으로 pool이 세 번 달랐다: 39837(07-23 리스트 덤프) → 39830(`54baa4c`로
  `meta_worldtraj.csv` 교체, index cache key에 파일명만 있고 content hash가 없어 조용히 재생성) →
  39817(`0219fcb`로 frozen-pose blacklist가 cache key에 추가). 그 결과 baseline
  `20260731_001511`(wandb `c4d2k5y4`)의 val과 08-02 cohort의 val은 3983개 중 419개(10.52%)만 겹치고
  wandb에 찍히는 **앞 160개는 1개만** 겹쳤다. 게다가 새 val의 89.48%가 old cohort의 **train**이라
  두 cohort를 가로지르는 비교는 전부 오염이었다. 이제 두 arm은 `c4d2k5y4`의 분할을 그대로 덤프한
  `latentcam_{train,test}_seg_list_c4d2k5y4.txt`를 못박아 쓴다 (train 35847 / test 3983, 현재
  blacklist로 train 10·test 3이 dataset에 없어 실효 35837/3980; 앞 170개는 영향 없어 wandb val 160은
  `c4d2k5y4`와 bit-identical). 이건 2026-07-23에 받은 "seg_list 써서 학습하고 validation도 앞에서
  160개 고정" 지시가 `base.py`·`config.yaml`·`geo_worldtraj_seglist.yaml`까지만 반영되고 이후 실험
  config 8개에 배선되지 않았던 누락을 메우는 것이다.

### Changed (tooling)
- **`scripts/data/make_latentcam_splits.py`에 `FROM_CACHE` / `OUT_SUFFIX`** (스크립트 자체도 이번에
  처음 커밋). `FROM_CACHE=<index cache .pt>`면 live config로 `CamDataset`을 새로 만드는 대신 **그
  cache의 `samples`를 그대로** 9:1 `random_split`한다 — 이미 끝난 run의 분할을 재현하려면 그 run이
  쪼갠 pool 자체가 필요한데, live config는 지금 blacklist(39817)를 달고 있어 `c4d2k5y4`의
  39830을 만들 방법이 없기 때문. `OUT_SUFFIX`는 `.txt` 앞에 붙어 2026-07-23 리스트를 덮어쓰지 않게
  한다. 둘 다 비우면 동작·출력 모두 이전과 동일.
  주의: 리스트는 **`random_split` 순서 그대로** 쓴다. 정렬 금지 — `base.py:106`이 valid loader를
  `shuffle=False`로 돌려 test 리스트 앞 160개가 곧 wandb val인데, 정렬하면 그 160개가 전부 1K batch가
  되고(실측 160/160 vs 원 순서 26/160) `c4d2k5y4`의 val과도 달라진다.
- **`scripts/eval_testset.py`에 `--set KEY=VALUE`** (반복 가능, 값은 YAML 파싱이라 `null`/`true`/숫자
  타입 유지) + **`scripts/eval_testset_queue.sh`에 `EXTRA` / `TAG_SUFFIX`**. 목적은
  `train_seg_list`/`test_seg_list` 못박기다: 07-30~07-31에 끝난 run 5개는 seg_list를 `null`로
  저장했고 `eval_testset.py`는 `base.Trainer._make_batch_generator`를 그대로 재사용하므로,
  **오늘 pool(39817)의 `random_split`** 으로 평가된다 — 그 run들이 실제로 쪼갠 39830과 다른 분할이고,
  그 3982 중 3563(89.48%)이 해당 run의 **train**이었다. 다섯 run 모두 pool 39830을 seed 42 /
  `train_frac` 0.9로 갈랐고 그 분할이 곧 `latentcam_{train,test}_seg_list_c4d2k5y4.txt`이므로,
  그 리스트를 `--set`으로 넣으면 진짜 held-out(3980)으로 돌아온다. 측정: pinned list == c4d2k5y4
  재현 val 3983 (순서까지), 현재 pool 실효 3980(빠진 3개 위치 584/1324/2607, 전부 앞 160 밖),
  앞 160은 wandb val과 bit-identical, `train ∩ val = 0`.
  `TAG_SUFFIX`는 출력 디렉토리 뒤에 붙어 기존(오염된) `eval_my/*`를 덮어쓰지 않는다.
  둘 다 비우면 동작·출력 모두 이전과 동일.
- **`scripts/eval_testset.py`: 상대경로 `--out`을 `REPO` 기준으로 해석** — `main()`이 CLaTr
  서브프로세스 때문에 `os.chdir(MAIN)`을 먼저 부르는데 `out_dir`을 그 뒤에 `osp.abspath`로 풀어서,
  상대 `--out`이 `main/eval_my/…`로 떨어졌다(기본값은 `osp.join(REPO, …)`로 만들어져 영향 없음 —
  위 `EXTRA`/`TAG_SUFFIX`로 `--out`을 넘기기 시작하면서 처음 드러난 경로). 위 큐의 첫 job이 실제로
  `main/eval_my/`에 쓰였고 `eval_my/`로 옮겼다.
- **`scripts/stop_at_epoch.sh`의 job spec에 선택 4번째 필드 `experiment`** —
  `screen:log:label[:experiment]`. 기존엔 `ps -ef | grep "SCREEN -dmS <screen> "`에서 `experiment=`를
  뽑았는데, 그건 screen을 `-dmS <name> bash -c ...`로 띄웠을 때만 통한다. 이미 떠 있는 빈 screen에
  `screen -X stuff`로 명령을 밀어넣으면 SCREEN 프로세스 cmdline에 `experiment=`가 없어서 watchdog이
  **"이미 종료됨"으로 조용히 건너뛴다** — 멈춰야 할 run을 안 멈추는 침묵 실패. 필드를 안 주면 예전
  방식으로 fallback하므로 기존 호출은 그대로 동작한다.
- **`scripts/data/norm_divisor_compare.py`에 `D["geo_lagernvs_view0"]`** — 이 스크립트는 분모를
  `centers[s]` 기준으로만 재는데, `_geo_lagernvs_scale`이 이제 `geo_lagernvs_anchor`로 앵커를 고를 수
  있게 돼서 `view0` arm(예: `geo_worldtraj_camembed_lagernvsnorm`)을 재면 그 arm이 실제로 쓰지 않는
  분모를 보고하게 된다. 두 변종을 다 내보낸다. `geo_first_view_target_s: true`면 `gi[0] == s`라 두
  값이 같으므로 기존 arm의 출력은 변하지 않는다.
  아직 안 고친 것: `scripts/data/static_camera_degeneracy.py:119`,
  `scripts/vae/vae_divisor_recon.py:96`은 여전히 frame-s 앵커 하드코딩.
- **`scripts/data/make_latentcam_splits.py`의 scene split 출력 문구 수정** — `(2)`행이
  `OUT_SUFFIX`를 무시하고 항상 `latentcam_{train,test}_list.txt`라고 찍어서, suffix를 준 실행에서
  실제로 쓴 파일과 다른 이름을 보고했다(파일 자체는 처음부터 suffix를 붙여 썼다).
- **testset eval 출력 위치 `results/testset_eval/` → `eval_my/`** (`scripts/eval_testset.py`의
  `EVAL_ROOT` 상수). 사용자 지시. `--out`으로 여전히 override 가능하고
  `scripts/eval_testset_queue.sh`는 경로를 하드코딩하지 않아 그대로 따라간다. 기존 5개 결과 디렉토리
  (3.7 GB)는 `eval_my/`로 이동했고 `results/testset_eval/`은 삭제. `eval_my`는 preds.npy 등 대용량
  산출물이 쌓이므로 `.gitignore`에 추가.

### Added (tooling)
- **`scripts/render/compare_norm_video.py`에 `--stack` / `--tags`** — `--layout row`는 segment 하나당
  strip mp4 하나만 내놓아서, "어떤 조건에서 `avg_scale`이 되고 어떤 조건에서 무너지는가"처럼 **segment
  사이를** 비교해야 하는 질문은 mp4 6개를 번갈아 봐야 했다. `--stack <out.mp4>`는 모든 segment의 strip을
  세로로 이어붙여 하나의 영상으로 쓰고, `--tags <json>`은 `{seg_name: caption}`으로 각 행 GT 타일 라벨을
  덮어쓰면서(`plx 0.026 OUT (under)` 등) **그 key 순서가 stack 행 순서**가 된다(json에 없는 seg는 뒤에
  알파벳 순). 두 인자 모두 없으면 출력·동작 이전과 동일. 산출물 예:
  `results/plx_band_render/plx_band_stack.mp4` (6 seg × [GT|lagernvs|avg_scale|PSNR], 1728×2048).
- **`scripts/stop_at_epoch.sh` (신규)** — 돌고 있는 학습을 지정 epoch 에서 멈추는 watchdog.
  `scripts/stop_at_epoch.sh <TARGET_EPOCH> [screen:log:label ...]`, 인자 없으면 현재 5개 run 기본값.
  tqdm 진행줄 `^Epoch N |` 을 폴링해 `N >= TARGET` 이면 정지하므로 epoch `0..TARGET-1`(= TARGET 개)은
  검증/체크포인트까지 끝난 상태다. 정지는 **Ctrl+C → 최대 `STOP_WAIT`초 대기 → SIGTERM → SIGKILL**
  순서 (바로 kill 하면 DataLoader worker 가 고아가 되어 GPU 를 물고 있음).
  프로세스 매칭 패턴은 끝을 `$` 로 anchor 한다 — `experiment=geo_worldtraj` 가
  `geo_worldtraj_camembed`/`_ctxlonger135`/`_lagernvsnorm`/`_decoupled` 의 **prefix** 라
  anchor 없이는 baseline 하나 멈추려다 5개 run 을 전부 죽인다(실측 96 proc 매치 → anchor 후 run 당 17).
- **`scripts/render/stitch_chunk_continuity.py` (신규)** — 연속된 segment 들의 per-segment LagerNVS
  랜더를 **scene 당 하나의 연속 영상**으로 이어붙여 chunk 경계 seam 을 본다. 입력은 기존 2-step
  ablation 파이프라인 출력(`main/dump_avgscale_render.py` → `tools/lagernvs/render_avgscale.py`)
  그대로이고 두 스크립트는 수정하지 않았다 — dump 가 `scale_mode` 와 무관하게 모든 divisor 를
  `scales` dict 에 넣으므로 dump 1회 + `RD_MODES` 로 두 arm 을 모두 커버한다.
  `<root>/_stitched/` 에 `<scene>_compare.mp4`(`[GT | mode…]`), mode 별 `[GT | render]` mp4,
  `<scene>_psnr.png`(global frame 별 PSNR + 경계 점선), `<scene>_summary.json` 을 쓴다.
  경계 직후 `--seam-frames`(기본 3) 프레임은 빨간 테두리 + `CHUNK N` 배너.
  seam 정량화는 `seam_metric`: 연속 segment 는 frame index 가 이어져 GT 자체도 움직이므로
  `r(t) = mean|R[t+1]−R[t]| / mean|G[t+1]−G[t]|` 로 GT 변화량으로 나눈 뒤, 경계의 `r` 을
  chunk 내부 `r` 의 중앙값으로 다시 나눈다 — **1.0 이면 경계가 평범한 프레임 스텝과 구별 불가**.
  기존 `scripts/render/render_scene_stitched.py` 는 LagerNVS 자체 정규화만 쓰고
  (`render_target_from_context.normalize`) 이 divisor 들을 못 받아 재사용 불가였다.
- **`scripts/data/chunk_divisor_stability.py` (신규)** — chunk 가 넘어갈 때 normalization
  divisor 가 얼마나 흔들리는지를 카메라 운동(직선성 / 속도)의 함수로 본다. 카메라 생성 후에
  더 촘촘한 context 로 다시 랜더할 수 있으므로 화질이 아니라 **구조·scale 연속성**이
  figure of merit 이라는 관점. `spread = (max_k D_k − min_k D_k) / mean_k D_k` (k=chunk).
  `straight = maxd/pathlen`, `speed = pathlen/D_A` (scene scale 로 정규화해야 scene 간 비교 가능).
  입력은 `scripts/data/norm_divisor_compare.py` 가 쓴 `per_segment.csv`,
  출력은 같은 폴더에 `chunk_stability.{png,_summary.md,_per_scene.csv}`.
- **`scripts/data/norm_divisor_compare.py`: `DATASET=dynamicverse` 분기 (신규, 기본값
  `DATASET=dl3dv` 는 기존 동작 그대로).** divisor 비교를 DL3DV 밖에서도 돌리기 위한 옵션.
  추가 env: `DATASET` / `DV_ROOT`(기본 `/data1/cympyc1785/data/dynamicverse`) / `DV_CHUNKS`(기본 3).
  구현은 `load_dynamicverse()` — `CamDataset.__new__` 로 인스턴스를 만들고
  `extrinsics_list / intrinsics_list / hw_list / samples / geo_cover_* / _geo_idx_memo` 만 채우는
  shim 이다. arm B 분모를 정의하는 retrieval 이 **재구현이 아니라 `dataset_dl3dv` 의 실제
  `_sample_geo_frustum_cover`** 그대로 돌아야 arm A/B 비교가 성립하기 때문.
  DynamicVerse `cameras.json` 의 `rotation`/`position` 은 **R_w2c / t_w2c**
  (`utils/camera_utils.get_camera_params_from_json` 과 동일 해석), `avg_scale` 은 없으므로
  `_avg_scale` 은 항상 `None`. `prompts.json` 이 clip 당 segment 1개만 주므로 chunk 는
  `[k*49, (k+1)*49)` 로 직접 자르고 `49*DV_CHUNKS` 프레임 미만 scene 은 버린다
  (606 scene / 1818 chunk 통과, 그중 590개 = 97% 가 `dynamic_replica` = 합성 데이터).
- **`main/dynamicverse_shim.py` (신규)** — 위 `load_dynamicverse()` 를 스크립트 밖으로 빼내
  분석(`norm_divisor_compare.py`)과 렌더 덤프(`dump_avgscale_render.py`)가 **같은 shim** 을 쓰게 함
  (`norm_divisor_compare.py` 는 인라인 정의를 지우고 import 로 교체, 동작 동일).
  추가된 것: `scenes=[...]` 로 특정 scene 만 로드, `with_frames=True` 면 `extract_frames()` 가
  `video_input.mp4` 를 `/data1/cympyc1785/data/dynamicverse_frames/<sub>/<scene>/%05d.png` 로
  디코드(캐시)해 `ds.frame_files_list` 를 채운다. **모든 subset 이 cx=252 / cy=140** 이라
  포즈 추정 해상도는 항상 504x280 — 원본 mp4 해상도(1280x720 / 1920x1088)와 무관하게 그 크기로
  리사이즈해야 intrinsics 와 맞는다.
- **`main/dump_avgscale_render.py`: `DATASET=dynamicverse` 분기 (기본 `dl3dv` 는 기존 동작 그대로).**
  `SEGS` 가 `"<subset>/<scene>"` 목록이 되고 각 scene 의 chunk 0..`DV_CHUNKS`-1 을 덤프한다.
  DynamicVerse 는 저장된 `avg_scale` 이 없으므로 `avg is None` 이어도 skip 하지 않고
  `scales` 에서 `avg_scale` 키만 빠진다(DL3DV 는 없으면 skip, 종전과 동일).
  출력 폴더명은 `"/"` → `"__"` 로 평탄화 — `scripts/render/*` 가 `<root>/*` 를 비재귀 glob 하기 때문.
- **`scripts/render/inject_carry_divisor.py` (신규)** — chunk 간 divisor spread 를 **눈으로 보이게**
  만드는 렌더 입력 생성기. chunk 마다 자기 `D_k` 로 렌더하면 context 와 target 을 같은 수로
  나누는 것이라 **전체 scene 의 similarity 변환**일 뿐이고, `camera_scale` 은 LagerNVS 의
  **조건 토큰**(`tools/lagernvs/data/normalization.py:116-121`)이라 렌더 결과가 거의 안 변한다
  (실측: divisor 5.7배 차이에도 PSNR 0.3 dB 미만 변화). divisor 는 renderer 의 속성이 아니라
  **camera DM 출력의 단위**이므로, chained 생성기가 scale 을 chunk 0 에 고정해두고 계속
  normalized trajectory 를 뱉는 상황 — `world_k = D_0 · t̂_k = (D_0/D_k) · GT_k` — 을 만들어
  **target translation 만** `D_0/D_k` 배 하고 context 는 그대로 둔다. 그러면 GT 프레임 대비 PSNR 이
  scale drift 를 직접 잰다. `<root>_drift/<scene>__<arm>_<k>/render_inputs.pt` 로 쓴다.
- **`scripts/render/stitch_chunk_continuity.py`: `--modes` 에 `<tag>:<mode>` 형식 추가.**
  해당 mode 의 렌더를 같은 폴더가 아니라 **형제 폴더 `<scene><tag>_<k>`** 에서 읽는다.
  arm 마다 target trajectory 자체가 달라 폴더가 갈리는 drift 렌더를 한 영상으로 비교하기 위함.
  tag 없는 기존 `--modes a,b` 는 tag `""` 로 해석되어 **기존 동작 그대로**.
- **`scripts/vae/vae_scale_matrix.py`: `CFG=k=v,k=v` env override + refit ckpt 2개 등록.**
  `CFG` 는 dataset 을 만들기 전에 `config.py` 값을 덮어쓴다. retrieval-DEPENDENT 한
  `scale_mode: geo_lagernvs` 는 `_sample_geo_frustum_cover` 를 다시 돌려 분모를 만들므로
  `geo_cover_*` 플래그가 experiment config 과 어긋나면 **다른 분모**를 재게 된다
  (`config.py` 기본값은 전부 off). leak-free mode(`avg_scale`, `ctx_longer_135max`)는 영향 없음.
  `CFG` 미지정 시 **기존 동작 그대로**. `CKPTS` 에 `vae_ctxlonger135` / `vae_geolagernvs_wt` 를
  추가해 arm 전용 VAE 와 공용 `vae_20260302_300` 을 한 표에서 비교할 수 있게 했다.
  측정 결과(1264 samples / 200 scenes, `/tmp/vae_matrix_rel.log`, `intr rel` 행):
  `vae_20260302_300` 이 세 scale_mode 전부에서 trans L1 최저
  (avg_scale 0.00301 / ctx_longer_135max 0.00455 / geo_lagernvs 0.00390),
  latent std 는 각각 0.44696 / 0.55882 / 0.51180. refit 2개는 `rel` 에서 rot L1 0.0225~0.0257 로
  `vae_20260302_300`(0.0050~0.0057)의 4~5배 → **scale_mode 전용 refit 이 공용 ckpt 보다 나쁘다.**

### Changed
- **`main/dynamicverse_shim.py`가 저장된 `avg_scale`을 읽는다** (기존엔 `lambda → None`이라
  `scale_mode: avg_scale`이 조용히 `_cam_dist_mean_scale`로 fallback 됐다). DynamicVerse에도
  `<scene>/avg_scale/<k>.json`이 있고 **파일 개수가 정확히 `floor(N_frames / num_frames)`** 라
  우리가 자르는 chunk `[k·T,(k+1)·T)`와 1:1 — DL3DV와 똑같이 `data_name.split('_')[-1]`이
  seg_key가 된다. 값/clamp/실패시 `None`은 `CamDataset._avg_scale`과 동일.
  `load_dynamicverse`가 `avg_scale <hit>/<total>` 커버리지를 찍는다(현재 1818/1818).
- **`main/dynamicverse_shim.py`가 `ds[i]`(진짜 `CamDataset.__getitem__`)를 지원한다.**
  `geo_hw` / `geo_num_views` / `geo_enabled` / `geo_latent_cache_dir` / `geo_cam_embed` /
  `geo_view_sampling` / `geo_posed` / `geo_shuffle_order`를 `CamDataset.__init__`과 같은
  `getattr` 기본값으로 채웠다. 기존 사용자(`norm_divisor_compare.py`,
  `dump_avgscale_render.py`)는 `ds.samples`와 `_sample_geo_*`만 써서 영향 없음.
- **normalization ablation 3-arm 의 VAE / intrinsics 통일 (`geo_worldtraj.yaml`,
  `geo_worldtraj_ctxlonger135.yaml`, `geo_worldtraj_lagernvsnorm.yaml`).** 기존 비교는
  normalization 외에 confound 가 3개 있었다 — (1) baseline 의 `vae_latent_scale` 0.96032625 는
  SCVideo MIXED corpus 값이라 diffusion 입력 std 가 0.4654 (A/B 는 실측값이라 ~1.0),
  (2) `intr_norm` baseline `rel` vs A/B `raw`, (3) VAE ckpt 가 셋 다 다름.
  세 arm 을 `intr_norm: rel` + 공용 `checkpoints/vae_20260302_300.pth` + `cam_dim 64` 로 맞춰
  **`scale_mode` / `geo_lagernvs_skip_ctx_norm` 만 남겼다.**
  arm 전용 refit(`vae_ctxlonger135`, `vae_geolagernvs_wt`)을 버리는 게 손해가 아닌 이유:
  `intr rel` 에서 `vae_20260302_300` 이 세 scale_mode 전부 trans recon L1 최저이고,
  refit 2개는 rot L1 이 4~5배로 underfit 이다 (60 epoch DL3DV-only).
  `vae_latent_scale` 은 **FULL corpus(39817 segment / 6095 scene)** 재측정치로 pin:
  baseline `avg_scale` **0.47637**, A `ctx_longer_135max` **0.58341**, B `geo_lagernvs` **0.53716**.
  `scripts/vae/vae_scale_matrix.py` 의 `MAX_SCENES=200` 기본값(1264 segment)은 meta 앞쪽
  저속 scene 편중이라 전 모드에서 **4~7% 낮게** 나온다 (0.44696 / 0.55882 / 0.51180) — 계통
  오차이므로 config 에 박는 값은 반드시 FULL corpus 로 잴 것 (`main/conf/config.yaml` 주석도 정정).
  geo latent cache 는 **재빌드 불필요**: geo cam_token 의 intrinsics 는 `intr_norm` 과 무관하게
  항상 raw 이고(`dataset_dl3dv._geo_cam_param`, line 836-838) camera VAE 는 `geo_emb` 에 안 들어간다.
  이전 config 로 돌던 run 들과는 **비교 불가** (A `7xdbu80e` / B `1mok217e` epoch 47 중단,
  baseline `c4d2k5y4` 는 epoch 150 정지 후 재기동 예정).
- **`geo_worldtraj_ctxlonger135.yaml` / `geo_worldtraj_lagernvsnorm.yaml` 주석의 canonicalization
  수치 정정 (주석만 변경, 학습 동작 무관).** 기존엔 `m = max‖target view center − frame s‖ / D` 의
  **전체** sd/CV만 적어서 scene scale 정규화가 실제로 하라는 일을 못 재고 있었다. log10(m)을
  scene 평균(between) / scene 내 편차(within)로 분해한 표로 교체
  (39817 segment / 6095 scene, `results/compare/norm_divisor_compare_postbl/per_segment.csv`):
  between-scene sd 0.1340(정규화 없음) → **A 0.0228(−83.0%) / B 0.0525(−60.8%)**,
  `avg_scale`(현 운영)은 0.2631 로 **오히려 악화(+96.4%)**. within-scene 은 A 0.1850 / B 0.1816 으로
  둘 다 나빠지지만(corr_wth −0.544 / −0.080) 이건 "이 segment 는 많이 움직인다"는 **진짜 신호**라
  없앨 대상이 아니므로 전체 sd 는 애초에 틀린 figure of merit 이었다.
  아울러 arm B 의 분모가 **retrieval 된 context view 기준**(`_sample_geo_frustum_cover` 가 뽑은
  view 에 대해 `1.35*max‖center − center[s]‖`)이라 추론 때 retrieval 이 달라지면 D 가 바뀐다는 점,
  arm A 의 분모는 context range 전체라 retrieval 과 독립이라는 점을 명시.
  `clamp(min=1e-5)` 는 실측상 한 번도 걸리지 않음(D min 0.4046 / 0.4987, `D ≤ 1e-2` 0개)도 기록.
- **`scripts/render/compare_textonly_vs_worldtraj.py` 를 2-way 하드코딩에서 N-way 로 일반화.**
  `--run LABEL=DIR`(반복 가능) / `--ref` / `--out` / `--ctx` / `--contact` / `--per-target` 추가.
  **인자 없이 실행하면 기존 2-way 동작(text-only vs worldtraj, 같은 출력 경로)이 그대로 재현된다** —
  기존 호출부는 수정 불필요. 출력물에 `_contact.png`(앞 N개 target 컨택트 시트)가 추가되었고,
  `_summary.png`는 run 수에 맞춰 `첫 run 대비 산점도 (N-1)개`로 늘어나며 `_scores.csv`도 run 수만큼
  컬럼이 붙는다. 공통 target은 모든 run이 예측을 남긴 것만 교집합으로 취한다.
  이걸로 만든 결과: `results/topdown_3way_textonly_worldtraj_camembed/`
  (text-only vs geo_worldtraj vs geo_worldtraj_camembed, 공통 target 160개).
  **주의**: 세 run 모두 학습 중이라 `test/`는 epoch이 서로 다른 스냅샷이다 —
  숫자 해석 시 그 폴더의 `README.md` 경고를 먼저 읽을 것.
- **`geo_anchor_first_frame` → `geo_cover_centered_at_s` 로 rename** (동작 변화 없음).
  옛 이름은 "첫 프레임이 anchor **view**로 들어간다"로 읽혔는데, 그건 `geo_first_view_target_s`가
  하는 일이다. 이 플래그는 `frustum_cover` greedy 탐색의 `anchor`/`ball_center`/`seg_scale`만
  정할 뿐 encoder context에 view를 추가하지 않는다 — 실측: `geo_first_view_target_s: false`이면
  `geo_cover_centered_at_s: true`여도 `view0 == frame s`가 **0/500**, `[s,e)` 안의 view가 **0/500**.
  (`true`이면 500/500.) rename 후 재검증에서도 세 config 전부 동작 동일.
  **옛 키는 deprecated로 계속 인식**된다(`dataset_dl3dv.py`에서 새 키가 없을 때만 fallback +
  경고 출력)므로 rename 이전 config/CLI override도 그대로 돌아간다.
  적용 범위: `main/conf/config.yaml`, `main/config.py`, `main/dataset_dl3dv.py`,
  `main/conf/experiment/*.yaml` 14개.
- **`meta_csv: meta_worldtraj.csv` pinned in `textonly_savedscale.yaml` and
  `vae_dl3dv_1_7k.yaml`** (was the `meta.csv` default), matching every other worldtraj-scoped
  experiment. `coverage_blacklist_path` stays `null`.
  **This changes no data** — measured, not assumed: both CSVs index to exactly 39830 samples /
  6097 scenes, and the scene sets and all 39830 sample IDs are identical (verified by diffing the
  two cached indexes). `meta.csv` lists 8048 scenes (1K–11K) but 8K–11K have no scene dirs on
  disk, and its 1K–7K portion (6132) exceeds `meta_worldtraj.csv` (6098) by 34 scenes that have
  `prompts.json` + `transforms.json` but no valid 49-frame segment, so they were dropped either
  way. The value of the switch is that the corpus is now stated explicitly in the config.
  Also measured: the scene-level `blacklist.csv` (295 scenes) is a no-op for both CSVs
  (`∩ = 0` for each) and `dataset_dl3dv._load_index` re-applies it unconditionally regardless, so
  no flag is needed to "enable" it.
- **`vae_latent_scale` set to SCVideo's verbatim `0.4467666`** (was our measured `0.46312`), keeping
  the DL3DV-only `config.py` triple otherwise unchanged (`vae_20260202_065659_400.pth`,
  `cam_dim: 32`, `intr_norm: raw`, `scale_mode: avg_scale`, `clatr_epoch109_dl3dv_seg_2.ckpt`).
  Explicit user decision to use SCVideo's constant rather than our re-measurement. Consequence,
  measured not estimated: the latent std of this triple on our 1264 DL3DV segments is 0.46312, so
  the diffusion input std is **1.0366** instead of 1.0000 (train/infer do
  `encode(traj) / vae_latent_scale`) — a 3.7% overshoot, from our segment definitions differing
  from SCVideo's one-49-frame-window-per-scene sampling.
  `config_large.py`'s line (`0.96032625` + `vae_20260302_300.pth` + `cam_dim 64` + `intr_norm: rel`
  + `clatr_epoch139_large.ckpt`) was evaluated and rejected in the same session. Recorded for
  future reference, since it is a valid alternative: that ckpt loads strict at 64-dim, gives
  `latent (16,13,64)` with the best recon of any cell (L1 rot 0.00504 / trans 0.00301 /
  intr 0.00222), `CameraDiffusionModel(cam_dim=64)` = 64.82M params; `config_large`'s dataset
  really does use saved `avg_scale` (`data/dataset_large.py:295-304`) + frame-0-relative
  intrinsics (line 313), so `avg_scale` + `rel` is its exact pipeline; 31 of its keys already match
  ours, the 7 that differ being `batch_size` 32, `save_epoch` 25, `num_thread` 4,
  `sample_data` 33980, and the three that make it a **point-cloud-conditioned,
  attention-supervised** run rather than text-only (`load_points`/`load_saved_pc_embeds` True,
  `model_type: baseline_attn_sup`). Its blocker is the same class of problem as above but larger:
  `0.96032625` was fit over SCVideo's MIXED corpus (DL3DV + DynamicVerse + dynpose-100k), while
  DL3DV-only measures 0.44696 → input std **0.4654**. `clatr_epoch139_large.ckpt` is a verified
  drop-in (both CLaTr ckpts: 260-key state_dict, no shape mismatch, 191,546,118 B; the input
  standardization lives in `evaluate/CLaTr/configs/dataset/standardization/0120.yaml`, not in the
  ckpt) but is trained on a different corpus, so its FD/PRDC/CLaTr-score would not be comparable
  with the `epoch109_dl3dv_seg_2` history we already have.
- **`cam_dim` + `intr_norm` pinned in every experiment that overrides `vae_ckpt_path`, plus the
  VAE-training experiments**, so each ckpt keeps its own latent dim and intrinsics convention
  regardless of the global default. Both values were set to what the current/pre-`intr_norm` code
  resolved to, i.e. **no behavior change** — the point is that these configs no longer silently
  depend on `config.yaml`'s defaults:
  * `cam_dim: 64` in `ar.yaml`, `ar_smoke.yaml`, `rolling.yaml`, `rolling_smoke.yaml`
    (CamVLA `causal_vae_v1_240.pth`), `textonly_align.yaml`, `geo_worldtraj_align.yaml`
    (`my_checkpoints/vae_worldtraj/last.pth`), and the VAE-training configs `vae_worldtraj.yaml`,
    `vae_dl3dv.yaml`, `vae_dl3dv_smoke.yaml`, `vae_dl3dv_avgscale.yaml`. All three non-SCVideo
    ckpts were inspected and are 64-dim (`encoder.to_mu.weight (64,64,1)`), so with the default now
    32 they would have hit the exact `size mismatch for encoder.to_mu.weight` crash recorded in
    FIX.log 2026-07-18.
  * `intr_norm: rel` in the four CamVLA configs (no `scale_mode` override → `avg_scale` → legacy
    `auto` == `rel`) and `vae_dl3dv_avgscale.yaml`; `intr_norm: raw` in `textonly_align.yaml`,
    `geo_worldtraj_align.yaml`, `vae_worldtraj.yaml` (fit under `first_farthest_135` → legacy
    `auto` == `raw`), `vae_dl3dv.yaml`, `vae_dl3dv_smoke.yaml` (`geo_lagernvs` → `raw`), and
    `textonly_savedscale.yaml`.
  Experiments that use the DEFAULT VAE and only override `scale_mode` (`geo_*`, `textonly`,
  `textonly_camscale*`, `smoke_*`) are intentionally left unpinned, so they inherit whatever
  `config.yaml` sets.
  *(Superseded later in this same `[Unreleased]` block — see "reverted to `config_large.py`'s VAE
  + CLaTr line" below. `textonly_savedscale.yaml` is now `rel`/64, not `raw`, and the unpinned
  experiments now inherit the 64-dim `vae_20260302_300` + `rel` default.)*
- **DL3DV root moved** `/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K` → `/data1/cympyc1785/data/DL3DV/scenes`
  (done by the user on disk). Updated every live reference: `main/conf/config.yaml` (`dl3dv_root`),
  `main/conf/experiment/geo_worldtraj_seglist.yaml` (`train_seg_list`/`test_seg_list`),
  `main/config.py`, `scripts/render/render_target_from_context.py`,
  `scripts/data/{make_latentcam_splits,filter_dl3dv,norm_camera_length_stats}.py`,
  `scripts/viewer/viser_val_cameras.py`, `scripts/context_select/{frustum_cover_sweep,
  vis_frustum_cover,visualize_covis_retrieval,viz_start_coverage_retrieval,vis_frustum_cover_multi,
  select_compare,covis_compare}.py`, `scripts/coverage/{dump_coverage_selk,viz_coverage,
  dump_coverage,analyze_geo_retrieval_coverage,blacklist_by_coverage}.py`. Also dropped the now-wrong
  "DL3DV-960" wording from `main/dataset_dl3dv.py` prints/docstring, `main/train_latent_cam_dm.py`,
  `main/train_vae_dl3dv.py`. `main/config.py`'s legacy `DL3DV_DATA_PATH` (used only by the old
  `dataset_seg.py`) was left as-is.
- **default VAE triple is now SCVideo's DL3DV-only setting** (`main/conf/config.yaml`,
  `main/conf/experiment/textonly_savedscale.yaml`): `vae_ckpt_path` → `vae_20260202_065659_400.pth`,
  `cam_dim: 32`, `intr_norm: raw`, `vae_latent_scale: 0.46312`, with `scale_mode: avg_scale`. This is
  what SCVideo's `main/config.py` uses (it pairs with `core_pkg/models/vae_intr.py`, i.e.
  `vae_intr_large` with `latent_dim=32` — the two modules are identical apart from that default and
  an extra `encode_sample`); `config_large.py`/`config_vae.py` (`0.96032625` + `20260302/300.pth`,
  64-dim) are the multi-dataset (DL3DV+DynamicVerse+dynpose) line and label the 32-dim pair `# old`.
  `vae_latent_scale` uses OUR measured std 0.46312 rather than SCVideo's 0.4467666, since
  `train_latent_cam_dm.py` divides the latent by it and our segment definitions differ (0.4467666
  verbatim would give diffusion-input std 1.037). Verified end to end: strict VAE load OK,
  `cam_param (16,49,11)` → `latent (16,13,32)`, diffusion-input std **1.0000** over 1264 samples,
  roundtrip L1 0.004424, `CameraDiffusionModel(cam_dim=32)` 64.79M params forward OK.
  Full matrix over 1264 DL3DV samples (`scripts/vae/vae_scale_matrix.py`; latent std | that std
  divided by each config constant | recon L1 rot/trans/intr):
  ```
  ckpt                       dim scale_mode          intr  lat.std /0.96033 /0.44677      rot    trans  intr_L1
  vae_20260302_300            64 avg_scale           rel   0.44696   0.4654   1.0004  0.00504  0.00301  0.00222
  vae_20260302_300            64 avg_scale           raw   0.43609   0.4541   0.9761  0.00695  0.00851  0.36288
  vae_20260202_065659_400     32 avg_scale           rel   0.63741   0.6637   1.4267  0.00648  0.01081  0.26291
  vae_20260202_065659_400     32 avg_scale           raw   0.46312   0.4822   1.0366  0.00466  0.00495  0.00293  <- active
  vae_20260302_300            64 cam_dist_mean       rel   0.99270   1.0337   2.2220  0.00792  0.01072  0.00400
  vae_20260302_300            64 cam_dist_mean       raw   0.98131   1.0219   2.1965  0.00895  0.01588  0.36148
  vae_20260202_065659_400     32 cam_dist_mean       rel   1.11188   1.1578   2.4887  0.01951  0.03242  0.25974
  vae_20260202_065659_400     32 cam_dist_mean       raw   1.00625   1.0478   2.2523  0.02013  0.03035  0.01340
  vae_20260302_300            64 first_farthest_135  rel   0.54116   0.5635   1.2113  0.00571  0.00457  0.00267
  vae_20260302_300            64 first_farthest_135  raw   0.52944   0.5513   1.1850  0.00718  0.01049  0.36245
  vae_20260202_065659_400     32 first_farthest_135  rel   0.70409   0.7332   1.5760  0.00840  0.01395  0.26203
  vae_20260202_065659_400     32 first_farthest_135  raw   0.54560   0.5681   1.2212  0.00691  0.00920  0.00462
  vae_20260302_300            64 context_longer      rel   1.06565   1.1097   2.3853  0.00804  0.01069  0.00399
  vae_20260302_300            64 context_longer      raw   1.05523   1.0988   2.3619  0.00910  0.01555  0.36159
  vae_20260202_065659_400     32 context_longer      rel   1.18505   1.2340   2.6525  0.01973  0.03262  0.26003
  vae_20260202_065659_400     32 context_longer      raw   1.08236   1.1271   2.4226  0.02123  0.03082  0.01273
  ```
  Reading it: the intrinsics convention is fixed by the CKPT (`20260302` → `rel` everywhere,
  `20260202` → `raw` everywhere; mismatching it costs ~100× on the intr channels), while
  `scale_mode` only rescales the translations and therefore the latent std. `0.96032625` belongs
  to `20260302` + `cam_dist_mean` + `rel` (0.99270); `0.4467666` to `20260202` + `avg_scale` +
  `raw` (0.46312). This SUPERSEDES the earlier version of this entry, which concluded that
  `0.4467666` belonged to the 64-dim ckpt — that 0.44696 agreement is a coincidence, and the
  conclusion was an artifact of the `intr_norm` bug below (the correct cell was not expressible).
  Note `cam_dim` must match the ckpt's `latent_dim` or the state_dict load hard-crashes on
  `encoder.to_mu.weight`.
  **Provenance of `vae_20260202_065659_400.pth`, from SCVideo's git history** (confirms the triple
  independently of our measurement). The run dir `20260202_065659` falls between SCVideo commits
  `cfd2cc0` (2026-02-02T04:22:35Z "Fix dataset") and `2f40e21` (06:59:23Z), so the launch tree is
  `cfd2cc0` + the `vae_intr` import edit that was committed the next day as `eb99764`
  (2026-02-03T02:25Z) — that edit is required, because at `cfd2cc0` `train_vae.py` still imported
  `vae_intr_large.CameraVAE()` whose default `latent_dim=64`, while the ckpt is 32-dim
  (`encoder.to_mu.weight (32,64,1)`). This also rules out reading the dir name as KST: 06:56:59 KST
  = 2026-02-01T21:56Z would precede `d4441b9` (02-02T02:12Z), the commit that first ADDED
  `vae_intr.py`. At that tree:
    * **train set = DL3DV `1K` only** — `config.py` had `dataset_dir = '.../DL3DV/scenes/1K'` (its
      only dataset key, unchanged from `d4441b9` through `a3713ae`), and `data/dataset.py:36`
      branches on `basename(dataset_path)[-1] == 'K'` → a flat `os.listdir('.../1K')` (1000 scenes
      locally). The `train_dataset_dir` 1K / `val_dataset_dir` 7K split only appears at `b27060c`
      (2026-02-23), three weeks after the ckpt; multi-chunk `build_dataset_dir_list()` is later
      still (`config_large.py`/`config_vae.py`).
    * **one sample per scene, frames 0-48** — `extrinsics[:num_frames]`, scenes with < 49 frames
      `continue`. No segment enumeration (that is our addition), which is why our 1264-segment
      measurement lands 3.7% off SCVideo's 0.4467666.
    * `avg_scale` + `raw` confirmed at the source: `normalize_camera_extrinsics_and_points`
      (`data_utils.py:20`) divides translations by `mean ||point - first cam||`, and
      `dataset.py:118-121` builds the intrinsics as `fx/(2cx), fy/(2cy)` with **no frame-0
      division** — the `rel` division (`dataset_large.py:313`) does not exist yet in this tree.
      The only later change to those lines (`eb99764`) is numpy → torch, semantics identical.
    * hyperparams at that tree: `batch_size 64`, `lr 1e-4`, `epochs 50000`, `save_epoch 10`,
      `vae_beta 1e-3`. `config.py` still pointed at the PREVIOUS ckpt (`20260123_074547/900.pth`,
      `vae_latent_scale 0.48848`); `0.4467666` + `20260202_065659/400.pth` were adopted at
      `a3713ae` (2026-02-05T06:42Z), with `dataset_dir` still `1K`.
- **`save_epoch: 1` → `10`** (`main/conf/config.yaml`), matching SCVideo's `config.py`. Only affects
  checkpoint-write frequency (per-epoch validation is unchanged).
- **`scripts/vae/vae_scale_matrix.py`** (new): measures latent std + per-component recon L1 over the
  whole (ckpt × scale_mode × intr_norm) grid, so a config constant like `0.4467666` can be traced
  back to the triple it was measured on. env `MAX_SCENES` / `MODES` / `CKPTS` / `META`.
- **image dir is now resolved, not hardcoded** (`main/dataset_dl3dv.py` + 6 scripts). Added
  `IMAGE_DIR_NAMES = ('images_4', 'images_8', 'images')` and `scene_image_dir(scene_dir, names)`,
  which returns the first existing candidate (one `isdir` per candidate — no per-frame stat, which
  matters on lustre). `_parse_transforms` uses it via `getattr(cfg, 'image_dir_names',
  IMAGE_DIR_NAMES)`, so the search order is overridable from config. The same local `_img_dir()`
  helper replaced hardcoded `images_4` joins in `scripts/render/{render_geo_preds,
  render_scene_stitched,render_target_from_context}.py` and
  `scripts/context_select/{visualize_covis_retrieval,vis_frustum_cover,covis_compare}.py`.
  Motivation: the on-disk images were swapped to 480p (below) — nothing else in the pose/intrinsic
  math changes, because `transforms.json` is byte-identical between the 960p and 480p trees and
  always reports the ORIGINAL full resolution (w=3840, h=2160, fl_x=1720.22, cx=1920, cy=1080).
- **DL3DV images swapped 960p → 480p to reclaim disk** (`scripts/data/migrate_480_images.py`, new).
  `DL3DV-480/<chunk>/<scene>/images_8` (480×270) was `os.rename`d into the name-matched
  `DL3DV-960/DL3DV-10K/<chunk>/<scene>/` (same lustre FS → metadata rename, ~4s/1000 scenes),
  keeping the dir name `images_8`; then every `images_4` (960×540) under DL3DV-960 was removed,
  including chunks 8K–11K which have no 480p counterpart (explicit user decision — those scenes
  are not in `meta_worldtraj.csv`). Pre-flight verified 6098/6098 `meta_worldtraj.csv` scenes and
  7000/7000 scene dirs in 1K–7K match by name with identical frame counts and filenames
  (2,092,998 frames each side). Move runs before any delete, so no scene is ever image-less.
  Consequences: (a) the **training** geo path is essentially unaffected — `geo_image_hw =
  [256, 448]` vs a 480×270 source is still a downscale (270→256, 480→448), verified by loading
  a 1K sample post-migration: `scene_image_dir` → `images_8`, PIL size (480, 270),
  `hw_list` still (2160, 3840) from transforms.json, `images` (6, 3, 256, 448), `cam_param`
  (49, 11). (b) the standalone LagerNVS render scripts use SIZE=512, so those DO upscale now.
  (c) the geo latent cache (`DATA/DL3DV/latent_cache`) was computed from 960p → stale.
  (d) `meta.csv` (8048 rows) includes 1916 scenes in 8K–11K that now have NO image dir
  (10K:876, 11K:485, 8K:288, 9K:267); `meta_worldtraj.csv` (6098, 1K–7K) is clean. Image-
  dependent experiments inheriting the default `meta_csv: meta.csv` need switching.
- **lagernvs moved** `/data1/cympyc1785/lagernvs` → `camera_generation/tools/lagernvs`
  (same-FS rename; data/ are absolute symlinks so unaffected). Updated
  `lagernvs_repo_path`/`lagernvs_ckpt_path` in `main/conf/config.yaml` + `main/config.py` to
  the new path. A compat symlink at the old location is kept so in-flight runs / historical
  wandb configs / helper scripts keep resolving; remove it once all runs referencing the old
  path have finished.

- **Reverted to `config_large.py`'s VAE + CLaTr line as the global default** (explicit user
  decision, reversing the `0.4467666` / `vae_20260202_065659_400` entry above — that entry is
  superseded, not deleted, so the flip-flop stays legible). `config.yaml` **and** `config.py` now
  both carry: `vae_latent_scale: 0.96032625`, `vae_ckpt_path: checkpoints/vae_20260302_300.pth`,
  `clatr_ckpt_path: checkpoints/clatr_epoch139_large.ckpt`, `cam_dim: 64` (matching that ckpt's
  `latent_dim`), `num_cam: 13`, `intr_norm: rel`, `scale_mode: avg_scale`, `vae_beta: 0.001`.
  `avg_scale` + `rel` is genuinely `config_large`'s pipeline: `data/dataset_large.py:295-304`
  divides translations by the stored point-cloud `avg_scale` (our `saved_avg_scale`, an alias of
  `avg_scale`) and `:313` divides the intrinsics by frame 0.
  `textonly_savedscale.yaml` / `_bs8.yaml` / `_bs32.yaml` were switched from `intr_norm: raw` /
  unpinned `cam_dim` to pinned `rel` / `64`, and their comment blocks rewritten (they still
  claimed the 32-dim `0.4467666` triple and contained a "non-relative intrinsics / divided by
  frame 0" self-contradiction).
  Verified end-to-end on the resolved `textonly_savedscale_bs8` config: VAE strict-loads at
  `latent_dim 64`, `cam_param (1,49,11) → latent (1,13,64)`, `CameraDiffusionModel(cam_dim=64)`
  = **64.82M** params, recon L1 rot **0.00489** / trans **0.00282** / intr **0.00218** over 375
  segments (consistent with the 1264-segment 0.00504 / 0.00301 / 0.00222). Both arms smoke-tested
  to `EXIT=0` including the CLaTr eval path.
  Two caveats, measured not estimated:
  * `0.96032625` was fit over SCVideo's MIXED corpus (DL3DV + DynamicVerse + dynpose-100k). On
    DL3DV-only this triple's latent std is **0.44696** (1264 segments) / 0.43350 (375 segments),
    so the diffusion input std is **0.4654** / 0.4514, not 1.0 — latents reach the model ~2.15x
    too small. Kept verbatim per user instruction; `0.44696` is the value for exactly unit-std
    input on DL3DV-only. (`0.99270` for `cam_dist_mean` + `rel` is coincidentally near 0.96 and is
    **not** `config_large`'s normalization — it must not be used to justify the constant.)
  * `clatr_epoch139_large.ckpt` is an architecture-identical drop-in (260-key `state_dict`, no
    shape mismatch, both ckpts 191,546,118 B) but is trained on a different corpus, so **FD /
    PRDC / clatr_score from runs using it are not comparable with any number in
    `EXPERIMENTS.log` before 2026-07-30** — only epoch139-vs-epoch139.
  Audited all 34 experiment configs + the default by Hydra-composing each and comparing resolved
  `cam_dim` against the actual `encoder.to_mu.weight` dim of its resolved `vae_ckpt_path`:
  **0 mismatches**. Configs pinning their own ckpt are unaffected (`ar*`/`rolling*` → CamVLA
  `causal_vae_v1_240`, `textonly_align`/`geo_worldtraj_align` → `vae_worldtraj/last.pth`); the
  five VAE-training configs never load `vae_ckpt_path`. The unpinned `geo_*` / `textonly` /
  `textonly_camscale*` / `smoke_*` configs now inherit 64/`rel`/`vae_20260302_300`/`0.96032625`
  where they previously inherited 32/`raw`/`vae_20260202_065659_400`/`0.4467666` — intended, but
  it means results from those configs straddle two different VAEs.
### Fixed
- **`blacklist.csv`를 고쳐도 index 캐시가 낡은 채로 계속 쓰이던 문제** (`main/dataset_dl3dv.py`).
  `_load_index`의 캐시 키는 `meta_csv` / `num_frames` / `before_only` / `geo_cover_k` /
  `coverage_blacklist_path` / `max_scenes`만 담았는데, scene-level blacklist는 캐시를 **만들 때**
  적용된다(`:374`). 그래서 `blacklist.csv`에 scene을 추가해도 기존 캐시가 히트하면 그 scene이 계속
  학습에 들어갔다 — 조용히 틀리는 종류의 버그. `_blacklist_fingerprint()`(sha1 앞 8자리)를 추가해
  키에 `__bl<hash>`를 붙였다. 파일이 없으면 `none`. 내용이 안 바뀌면 키도 그대로라
  **기존 캐시 재사용 동작은 유지**되고, 편집하면 자동으로 재빌드된다(6,098 scene 스캔 ~37초).
- **DL3DV blacklist에 `duplicate_camera_centers` 2개 scene 추가** (데이터 파일
  `<dl3dv_root>/blacklist.csv`, 레포 밖. 백업: `blacklist.csv.bak_20260801`).
  `4K/50eb3c0d…8d5f`(dup_frac 0.582, 연속 142프레임 정지, N=368),
  `3K/b7da67fc…b1da`(0.257, 62프레임, N=331). 둘 다 COLMAP 등록이 끊겨 프레임 과반이 한 좌표에
  박혀 있는데 영상은 멀쩡히 움직인다 → context baseline이 0이라 LagerNVS 렌더가 어떤 divisor로도
  PSNR ~14.4에 갇히고(`results/norm_degenerate_check/`), `geo_lagernvs` divisor는 1.7e-5로 붕괴해
  정규화된 도달거리 `m`이 567,215까지 튄다. index 재빌드 결과 **39,830 → 39,817 sample /
  6,097 → 6,095 scene**.
- **동시 학습 간 CLaTr `lightning_logs` 버전 충돌로 `clatr_score` 1회 실패**
  (`main/evaluate/CLaTr/src/extraction.py`). `L.Trainer(...)`에 `logger` 인자가 없어 기본
  `TensorBoardLogger`가 붙는데, 이 로거는 `lightning_logs/`를 스캔해 다음 `version_<N>`을
  정한다 → 여러 학습이 같은 cwd에서 clatr eval을 동시에 띄우면 두 프로세스가 같은 N을 골라
  하나가 `FileExistsError`로 죽는다 (train1 @ `version_1113`). 해당 epoch의 clatr 지표만
  누락되고 학습 루프는 중단 없이 계속됐다. `logger=False`로 수정 — `trainer.predict`는 로깅을
  하지 않으므로 지표 값은 불변이고, 빈 version 디렉토리(1119개, 14M) 누적도 멈춘다.
- **cam_param's intrinsics convention was coupled to `scale_mode`, feeding the VAE the wrong
  encoding** (`main/dataset_dl3dv.py`, `main/conf/config.yaml`, `main/config.py`). New option
  `intr_norm: 'auto' | 'rel' | 'raw'` — `raw` = `fx/2cx, fy/2cy` (0.448, 0.796 for DL3DV),
  `rel` = the same divided by frame 0 (exactly 1.0 for DL3DV, i.e. SCVideo `dataset_large.py:313`),
  `auto` = the old coupling (`rel` iff `scale_mode == 'avg_scale'`) and remains the DEFAULT so
  existing configs reproduce bit-for-bit (verified `auto == rel` for `avg_scale`, `auto == raw` for
  `cam_dist_mean`, and the 9 extrinsic channels are byte-identical across the intr axis). The
  convention is a property of the VAE CKPT, not of the translation normalization, so the old
  coupling meant every `cam_dist_mean` / `context_longer` / `first_farthest_135` run fed `raw` to a
  ckpt that wants `rel` → intr recon L1 0.361 instead of 0.004, and the cell SCVideo's DL3DV-only
  config actually uses (`avg_scale` + `raw`) could not be expressed at all. Silent — no crash, no
  loss spike, only the 2 intrinsics channels are affected. The `raw` branch now derives width/height
  from the principal point (`2cx, 2cy`) like SCVideo rather than `transforms.json`'s `w, h`;
  identical for DL3DV (0 of 400 scenes differ). See `FIX.log` 2026-07-30.
- **index cache survived the dataset move with dead absolute paths** (`main/dataset_dl3dv.py`).
  `<root>/.latentcam_index/<key>.pt` stored `scene_dir_list` as ABSOLUTE paths, so after the
  root move the cache loaded fine but every scene dir pointed at the old location (`isdir` →
  False) and `__getitem__` would fail. Now the cache persists `scene_chunks` (paths relative to
  `self.root`) and rejoins them against the current root on load. Legacy absolute caches are
  still accepted, but only if `scene_dir_list[0]` still resolves; otherwise the cache is declared
  STALE and the index is rebuilt.
- **top-down plots were a front/back view, not top-down** (`main/infer_validation_sample.py`,
  `scripts/render/topdown_swap.py`). They hardcoded the x-z plane, but transforms.json stores c2w
  in the nerfstudio frame where DL3DV's `applied_transform` (x↔y swap + z flip) is baked in — there
  world-up is X and motion lives in Y-Z, so x-z looked down the *forward* axis (a front/back view).
  Fix: anchor both GT and pred to the GT first camera (`inv(c2w[0]) @ c2w`, same first-frame
  anchoring as the GenDoP pyramid viz), which puts cam0 at the origin with the OpenGL camera axes
  (up=+Y, right=+X, forward=-Z). Drop up(+Y) → ground = X-Z; plot X horizontal, -Z vertical so the
  camera forward points up in the image (map-like). User-confirmed orientation.

### Changed
- **lazy dataset loading** (`main/dataset_dl3dv.py`, default `lazy_dataset: true`): `__init__` now
  builds only the lightweight sample/scene index (reading `prompts.json` + an n,h,w probe from
  `transforms.json`, no per-frame stat) and persists it to `<root>/.latentcam_index/<key>.pt`, so
  reruns load the index instantly instead of re-scanning all ~6k scenes. Scene poses/paths are
  parsed on demand in `__getitem__` (`_load_scene` + `_LazyScenes`, cached). Also dropped the
  per-frame `osp.isfile()` (~330 stats/scene on lustre) for a single `images_4` dir check — this
  was the main ~28-min init bottleneck. Verified byte-identical to the eager path (samples/cam_param/
  geo_c2w diff 0.0); `lazy_dataset: false` restores the old eager load. Note: with num_workers the
  per-worker scene cache grows toward the working set (no COW sharing like eager) — cap workers if
  RAM-bound.

### Changed
- **`scale_mode` naming unified** — "avg_scale" was overloaded (stored point-cloud value vs the
  camera-distance mean). Now `avg_scale` means **only** the stored point-cloud avg_scale
  (`<scene>/avg_scale/<seg>.json`), and the camera-distance one is `cam_dist_mean`:
  - `saved_avg_scale` → **`avg_scale`**, `_saved_avg_scale()` → `_avg_scale()`
  - `target_cam` → **`cam_dist_mean`**, `_camera_based_avg_scale()` → `_cam_dist_mean_scale()`,
    `_cam_avg_scale_context()` → `_cam_dist_mean_context()` (mode `context_longer` unchanged)
  Old spellings still work via `_SCALE_MODE_ALIASES` / `resolve_scale_mode(cfg)`, so existing
  yaml, wandb configs and in-flight resumes are unaffected (verified: `avg_scale` vs
  `saved_avg_scale` and `cam_dist_mean` vs `target_cam` both give cam_param diff 0.0).
  The batch now also carries **`norm_scale`** = the divisor the active mode produced; the old
  key `avg_scale` is kept as an alias of the same tensor (SCVideo's name), so every consumer
  (`train_latent_cam_dm.py`, `infer_*.py`, `gen_*.py`, `cache_geo_embeddings.py`) is unchanged.
  Updated: `main/conf/config.yaml` (`scale_mode: avg_scale`), `geo_worldtraj{,_before,_seglist}`,
  `geo_hybrid_shuf`, `textonly_savedscale`, `main/config.py` default (`cam_dist_mean`), and the
  analysis scripts' labels/columns (`scripts/render/compare_*`, `scripts/data/
  norm_camera_length_stats.py`, `scripts/coverage/compare_scene_span_scale.py`).
  `main/configs_backup/` left as-is (historical, covered by the aliases).

### Added
- **`CamDataset.from_segments(cfg, segments)`** (`main/dataset_dl3dv.py`): segment-scoped dataset
  for inference/rendering. Instead of indexing the whole corpus, it reads `meta.csv` once to map
  the flattened scene name back to its chunk, then opens `prompts.json`/`transforms.json` for only
  the requested scenes — cost is O(#requested segments). `__getitem__`, geo context sampling and
  normalization are untouched, so output is identical to the full dataset (verified: 10 segments,
  all tensors diff 0.0 vs `CamDataset(cfg,'train')`, and the re-dumped `render_inputs.pt` byte-match
  the previous run). Used by `main/dump_render_inputs.py` (single `RI_SEG`) and
  `main/dump_avgscale_render.py` (new optional `SEGS` env). Dumping the 10 avg_scale-test segments:
  **~20 min → 6.2 s**. For reference SCVideo has no such path — its `main/infer_cam_dm.py` calls
  `Trainer._make_batch_generator(include_train=False)`, which builds the full `CamDataset` and
  `random_split`s 90/10, so inference there loads exactly as much as training.
- **`scale_mode: saved_avg_scale`** (`main/dataset_dl3dv.py`): replicate SCVideo's original
  normalization (`data/dataset_large.py`) — normalize camera translations by the STORED
  point-cloud `avg_scale` (`<scene_dir>/avg_scale/<seg_key>.json` = mean ‖scene point − first
  camera‖, ~10–44) instead of the camera-based mean. Also mirrors SCVideo's intrinsics under this
  mode: width/height from principal point (cx·2, cy·2) + normalized relative to frame 0
  (frame0 intr → [1,1]). Falls back to camera-based if the json is missing. Other modes unchanged.
  Purpose: match the scale the default VAE (`vae_20260302_300`) was trained on (SCVideo used stored
  point-cloud avg_scale; the DL3DV port had silently switched to camera-based `target_cam`).
  Added `self.scene_dir_list` + `_saved_avg_scale()`.

### Fixed
- **porting divergence from SCVideo `dataset_large.py`** surfaced: the DL3DV `dataset_dl3dv.py`
  port had diverged in 3 places — (1) avg_scale source (camera-based vs stored point-cloud), (2) no
  frame-0-relative intrinsics normalization, (3) width/height from stored w,h vs cx·2/cy·2. All three
  are now reproducible via `scale_mode: saved_avg_scale` (existing camera-based modes left intact).

### Added (more)
- **`scripts/data/norm_camera_length_stats.py`**: camera-length distribution under the two training
  normalizations (point=target_cam vs dist=1.35·max) over ~5.4k real training segments (rebuilt
  standalone from transforms.json + prompts.json, no dataset load). Per-frame ‖center‖/avg_scale,
  per-segment span + path length (mean/std/var/percentiles) + histograms →
  `results/compare/normalization_camera_length/`. point var 0.345 (span 1–24, heavy tail) vs dist
  var 0.054 (span const 0.741, bounded ≤0.741).
- **`main/cache_geo_embeddings.py`**: precompute + cache frozen geo embeddings (fp16) per DL3DV
  segment (`data/DL3DV/latent_cache/<data_name>.pt`, (M,768)) so training can skip the per-step
  LagerNVS forward (frozen + deterministic context). fp16 chosen: geo_emb |max|≈14.6 → no overflow,
  rel-err 1.8e-4 (training already bf16). ~285GB for full worldtraj scope; pilot = 1K batch.
- **`main/dump_render_inputs.py`** (+ vendored `tools/lagernvs/render_pred_from_dump.py`, not
  committed): render a results/validation predicted trajectory with LagerNVS. Two-step to avoid the
  latentcam↔lagernvs `models` package clash — step 1 (latentcam) dumps the exact inference-time geo
  context (image paths + OpenCV-world c2w + intrinsics, same frames as at inference) + the pred
  cameras to `render_inputs.pt`; step 2 (lagernvs) loads context at 512, adjusts intrinsics,
  normalizes exactly like training (`normalize_extrinsics` = view0-relative + 1.35·max, camera_scale
  0.7407) and `build_cam_cond` → `render_chunked` with the same general_512 model used for geo
  conditioning → `render_pred_lagernvs.mp4` + `render_pred_grid.png` per model. worldtraj/align share
  the frustum_cover context; hybrid uses its own.
- **`main/conf/experiment/textonly_align.yaml`**: text-only counterpart of geo_worldtraj_align —
  no geo, but first_farthest_135 (1.35·max_dist) normalization + vae_worldtraj (latent_scale
  0.569379) + meta_worldtraj.csv. Purpose: text-only vs worldtraj_align isolates the geo effect
  under dist normalization. (Training launched on GPU5 per explicit user instruction — overrides
  the CLAUDE.md no-4~7 rule; run `dl3dv_textonly_align`, screen train5.)
- **`main/infer_textonly_batch.py`**: runs the text-only model on the worldtraj validation targets
  (inputs reconstructed from saved `_transforms_ref.json` + `_caption.json`, no CamDataset load) so
  text-only vs worldtraj can be compared PER-PAIR on identical targets. Writes
  `results/compare/text-only_vs_worldtraj/textonly_preds/`.
- **`scripts/render/compare_textonly_vs_worldtraj.py`**: per-pair top-down (GT + worldtraj+geo pred
  + text-only pred + geo-context stars, first-cam anchored X/-Z) on the same 160 targets + world
  pos_rmse; both use target_cam so the diff is geo on/off. `_summary.png` (paired) + `_scores.csv`.
- **`scripts/render/compare_textonly.py`**: visualizes the text-only model
  (`20260719_210144_dl3dv_textonly`, target_cam + no geo) inference → `results/compare/textonly/`.
  Per-target top-down (GT vs pred, first-cam anchored X/-Z) + world pos_rmse/rot + CLaTr; `_summary.png`;
  and `_vs_geo.png` a DISTRIBUTIONAL box comparison vs point(worldtraj)/dist(align) (targets are
  disjoint across models — distributions, not per-pair). Kept `normalization_point_vs_dist/` intact.
- **`preds_scores.csv`** (`main/evaluate/eval/src/eval_only.py`): per-sample dump of every
  wandb-logged eval metric. Per-sample columns `captions/{precision,recall,fscore}`,
  `clatr/clatr_score` (100·cos(pred-traj, text)), `clatr/pred_ref_cosine` (100·cos(pred-traj,
  GT-traj)); plus `clatr/{precision,recall,density,coverage,fcd}` repeated as run-level constants
  (distributional/set-level → no per-sample value). `preds.csv`/`preds_pcf.csv` unchanged.
- **`scripts/data/extract_geo_context.py`**: dumps the geo-context camera world centers for the
  160 validation targets (geo_worldtraj config; context selection is deterministic + identical for
  align) → `_geo_context.json`, so the comparison viz can overlay conditioning views as stars.
- **`scripts/render/compare_norm_topdown.py`**: compares two normalization schemes
  (point = target_cam vs dist = first_farthest_135) on the same 160 validation targets. Per-target
  top-down (GT + both preds + geo-context cameras as magenta stars, first-cam anchored X/-Z) +
  world-space scores (pos_rmse/rot) and per-target CLaTr score (from `_clatr.json`), plus a 2×2
  `_summary.png` (pos_rmse + CLaTr, sorted + paired) and `_scores.csv`. Metrics in denormalized
  world so they are comparable regardless of each model's normalization/VAE. Output →
  `results/compare/normalization_point_vs_dist/`.
- **`models/GenDoP/extrinsic2pyramid/vis_validation_anchor.py`**: trajectory pyramid viz that
  replicates GenDoP's **original** `dataset/extrinsic2pyramid/visualize.py::draw_json`
  preprocessing — first-frame anchoring (`c2ws = inv(c2w[0]) @ c2ws`) + optional 2-frame
  subsample — before calling `vis.py::draw_json`. The plain `vis.py` path omits anchoring and
  therefore inherits each dataset's arbitrary world up-axis (X for latentcam, Y for DataDoP),
  which is why front/top/side came out mislabelled. Anchoring re-expresses the trajectory in the
  first camera's frame so the views are canonical regardless of world up-axis. Verified: anchored
  DataDoP output matches the reference `shot_0003_traj_cleaning.png` exactly. Supersedes the
  earlier `vis_validation_rot.py` per-trajectory rotation hack (wrong approach). Single-file mode:
  `vis_validation_anchor.py IN.json OUT.png [--sub]`; no-arg mode sweeps `results/validation`.
- **`main/infer_swap_ablation.py`**: swap-ablation inference for the geo camera-DM models.
  For N fixed val samples (deterministic split), runs 3 modes and saves each per model/mode
  (`results/swap_ablation/<model>/<mode>/`): (a) normal, (b) ctxswap — keep the anchor context
  view (view0=frame s), take the rest from another sample, (c) textswap — keep context, swap
  text. Per-sample noise seeded so modes are directly comparable. Reads SWAP_CKPT/SWAP_OUT/
  SWAP_N/SWAP_TAG env + `experiment=` Hydra override.
- **`geo_cover_before_only`** flag (`dataset_dl3dv.py`): restrict out-of-segment geo context
  to frames BEFORE the target segment (index < s) instead of the longer out-of-seg side; the
  target segment must have frames before it, so first-segment targets are filtered out
  (`s < geo_cover_k`) — the target segment can be the 2nd segment onward. New Hydra experiment
  `conf/experiment/geo_worldtraj_before.yaml` (identical to geo_worldtraj except before-only
  context: first camera s + 5 out-of-target views drawn from earlier segments). Default off —
  existing geo configs unchanged.
- **`geo_lagernvs_skip_ctx_norm`** flag + `build_cam_token(override_scale=...)`
  (`models/geo_encoder.py`, `geo_encode`): skip LagerNVS's own 1.35·max(context) normalization
  and reuse the target's initial 1.35·max_dist (avg_scale) so target & geo latent share one
  frame+scale (full coordinate alignment). New Hydra experiment
  `conf/experiment/geo_worldtraj_align.yaml` (scale_mode=first_farthest_135 + vae_worldtraj
  VAE, latent_scale 0.569379; context selection identical to geo_worldtraj).
- **`geo_shuffle_keep_first`** flag (`dataset_dl3dv.py`): when shuffling geo context order,
  keep view0 (frame s / VGGT reference) fixed and shuffle only the rest. New Hydra experiment
  `conf/experiment/geo_hybrid_shuf.yaml` (hybrid: 2 in-target [view0=s] + 3 out-of-seg covis,
  first fixed + rest shuffled, target_cam scale, geo_posed, meta_worldtraj).
- **`geo_cover_subtract_first`** flag + `frustum_cover_select(prepicked=...)` in
  `dataset_dl3dv.py`: when the first camera (frame s) is a fixed context view, subtract its
  coverage from the greedy union first so the remaining k−1 views maximize RESIDUAL coverage.
  New Hydra experiment `conf/experiment/geo_worldtraj.yaml` (lagernvs, first-cam-fixed +
  residual-coverage out-of-seg selection, `scale_mode=target_cam`, geo_posed 1.35·max, default
  VAE/CLaTr, meta_worldtraj). Default off — existing geo configs unchanged.
- **`meta_worldtraj.csv`** (DL3DV root) = `meta.csv`[1K–7K] − `blacklist.csv` = 6098 scenes; the
  dataset now honors `cfg.meta_csv` (default `meta.csv`) so a run can select its scene list.
  `blacklist.csv` gained 34 scenes (21 teleport + 13 image/pose length-mismatch) detected by
  `filter_dl3dv.py`; entries reformatted to `<split>/<hash>` (matching normalized to basename).
- **`scale_mode='first_farthest_135'`** (`dataset_dl3dv._first_farthest_scale`): LagerNVS-style
  per-segment normalization = 1.35 × max(‖cam center − first camera‖). New Hydra experiment
  `conf/experiment/vae_worldtraj.yaml` (geo off, this scale, meta_worldtraj, batch 64) for a
  camera-VAE re-fit on the WorldTraj scope.

### Changed
- **Config package retired → `configs_backup/`.** The active pipeline is fully on Hydra
  (`conf/` + `hydra_cfg.load_cfg`); the dead `import configs` shim was stripped from all loaders.
  For the legacy scripts still on the Python configs, `config.py` / `config_large.py` /
  `config_vae.py` were pulled back into `main/` (root_dir depth reverted to `..`) so their flat
  `from config* import` resolves. (`config_large`/`config_vae` still crash at import on their
  hardcoded absent data paths — pre-existing; to fix when those scripts are needed.) Verified:
  base `config` and active `train_latent_cam_dm` both load correctly side by side.
- **`scripts/` reorganized by purpose** into subfolders: `render/` (15, LagerNVS NVS + shared
  render libs), `coverage/` (8, coverage dump/analysis/blacklist/scale study), `context_select/`
  (7, view-selection methods + viz), `vae/` (2), `data/` (2, dl3dv filter + run-dir migration),
  `viewer/` (1, viser). Cross-folder sibling imports preserved via a small `sys.path.append`
  snippet (scripts root + all subfolders) injected per file; the 3 scripts that reach `main/`
  via a relative `..` had their depth fixed (`../.. `). Verified: all 35 files parse; cross-bucket
  imports resolve (render/coverage/context_select/data/vae).
- **Hydra/OmegaConf config system** (`main/conf/`): all 27 Python configs auto-ported to
  `conf/config.yaml` (base, 82 fields) + `conf/experiment/*.yaml` (deltas), composition via
  Hydra defaults (base + experiment override) — mirrors the old inheritance. `main/hydra_cfg.py`
  `load_cfg()` composes with standard Hydra CLI (`experiment=rolling lr=1e-4`) AND back-compat
  `LATENTCAM_CONFIG` env; returns a `cfg` behaving like the old Config (attribute access, real
  `t5_dtype` torch dtype, mutable) + `cfg_dict`. Every active loader (train/gen/profile) swapped
  to `load_cfg` (verified: composed == Python `cfg_dict` for all 23 experiments, 0 mismatches).
  Training now saves the full resolved config as `config.yaml` into the run dir + wandb run dir
  (`hydra_cfg.save_cfg_yaml`). CLaTr eval keeps its **own** Hydra in a subprocess — no conflict
  (verified: GlobalHydra clean after `load_cfg`; CLaTr eval subprocess runs end-to-end).
  Python `configs/*.py` retained (legacy config_large-based scripts still use them).
- `scripts/migrate_run_dirs.py` — wrote estimated `config.yaml` into 17 existing `results/`
  folders (mapped by `exp_name` suffix) and renamed 4 wandb dirs to `<ts>_<exp_name>` by exact
  timestamp match (live run + 39 orphan smoke/offline dirs left untouched).
- **Config layout**: moved all `main/config*.py` (27 files) into a `main/configs/` package.
  `configs/__init__.py` self-registers its dir on `sys.path`, so flat module names
  (`import config as _base`, `LATENTCAM_CONFIG=config_rolling`) and inter-config imports keep
  working unchanged. Every config importer (train/infer/gen/profile scripts + scripts/verify_vae,
  vae_interp) gained a one-line `import configs` before loading a config. `root_dir` in
  `config.py`/`config_large.py`/`config_vae.py` fixed to `osp.join(cur_dir, '..', '..')` (now one
  level deeper), so it still resolves to the latentcam root.

### Added (data)
- `scripts/filter_dl3dv.py` — DL3DV meta filter mirroring scenetok's `build_dl3dv_meta_row`
  (transforms/image presence, num_images>=34, images==poses, **teleport camera rejection**,
  image size + consecutive-frame checks; optional `--require-prompts`). Runs over `--subs`
  (default 1K–7K), writes `meta_tmp.csv`, and diffs against `meta.csv[subs] - blacklist.csv`.
  Finding: current `meta.csv[1-7K]` still contains 34 scenes that fail the (fixed) filter —
  21 teleport + 13 image/pose length mismatch. `meta_tmp.csv` is the cleaned list.

### Added
- **Text-only ROLLING model (per-token Diffusion Forcing + rectified flow)** on the
  causal-VAE latent (W=13 tokens, D=64). One model → full_sequence / chunk_ar / rolling
  inference by tau schedule only.
  - `models/camera_diffusion_model_latent.py`: per-token FiLM (no gate). `timestep_embedding`
    accepts `(B,)` or `(B,T)`; `forward` detects `per_token` (t_embed.dim()==3) and applies
    element-wise `(B,T,D)` scale/shift, else legacy `(B,1,D)` broadcast (bit-identical when off).
  - `main/config.py`: `per_token_noise` (False), `cfg_dropout_p` (0.1), `loss_tau_min` (0.02).
  - `main/tau_sampler.py`: W=13 tau mixture (30% iid / 40% ramp / 15% boot-up / 15% full-seq)
    → `(tau [B,W], labels [B])`.
  - `main/train_latent_cam_dm.py`: `per_token_flow_loss` (rectified-flow, tau-masked, CFG
    dropout) + training branch (`elif per_token_noise`) between AR and DDPM; validation
    per-token branch with tau-bin / pattern val-loss logging (CLaTr guarded off for per-token).
  - `main/config_rolling.py` + `main/config_rolling_smoke.py`: text-only rolling configs
    (causal VAE, per_token_noise).
  - `main/rolling_sampler.py`: 3-mode inference (gen_full / gen_chunk_ar / gen_rolling) in
    latent space + `latent_to_traj` (causal-VAE decode → (49,9) pose9 + (49,4,4) w2c) +
    `jerk_spectrum` (period-4 token-boundary / period-12 chunk-boundary power).
  - `main/test_per_token.py`: unit tests (per-token modulation, tau sampler, loss zero-denom).
  - `main/gen_scene_rolling.py`: generate a FULL scene with the rolling ckpt. Modes: `full`/
    `chunk_ar` (per-segment gen + hard-snap chaining, has seams) and `rolling` (CONTINUOUS —
    one sliding window across the whole scene, text switches per emitted token's segment,
    decode ONCE → seam-free; single scene scale = mean of per-seg avg_scale). `--scene-skip`
    picks a different scene. Saves transforms_pred/ref.json (viser) + top-down plot + drift.
  - `main/rolling_sampler.py`: `gen_rolling_scene()` — continuous multi-segment rolling used
    by the above.

### Added (tooling)
- `main/gen_scene_align.py` — per-segment inference placed two ways + GT, top-down. (A) per-seg
  aligned (anchor reset to each segment's GT start → local shape, no accumulation) vs (B) chained
  (running anchor → accumulated drift) vs GT. `--auto-best N` scans loaded scenes and picks the N
  with lowest mean per-segment err; renders a GT | per-seg | chained 3-column figure per scene.
- `main/profile_geo.py`, `main/profile_rolling.py` — latency breakdowns. geo: per-sample cost
  is ~49% VGGT re-encode + ~38% 6-image disk load + ~8% frustum_cover select, DiT only ~1.7%
  (context changes every segment → geo latent re-extracted, no cache). rolling: stage timing
  of T5 encode / rolling denoise loop / VAE decode.
- `scripts/compare_scene_span_scale.py` — statistical comparison of camera-normalization
  scales: `scene_span` (NEW scene-UNIFIED: max ‖center[i]−center[0]‖ over the whole video,
  one value/scene) vs per-segment `target_cam` and `context_longer`. Reports within-scene CV
  (how much the scale "keeps changing") + magnitude ratios; dumps per-scene/per-segment CSVs.
- `scripts/render_scene_stitched.py` — new options (defaults preserve old behavior):
  `--first-view-fixed` (frame 0 always context; its coverage subtracted first, residual
  coverage drives the greedy picks) and `--scale-mode scene_span` (size the coverage ball by
  the scene-unified span instead of per-segment seg_scale). `frustum_cover_select` gains a
  `prepicked=` arg; coverage ball now centered on the target segment. `--viz-all` now honored.
- `scripts/render_geo_preds.py` — render a geo run's GENERATED cameras with LagerNVS. Reads
  `<run>/test/<name>_transforms_pred.json` (generated OpenGL c2w, anchored at GT frame s in scene
  world), recovers the segment [s:e] by matching ref endpoints to scene cameras, picks out-of-seg
  coverage context, and renders the generated path vs the GT path side by side. N samples.
- `scripts/render_scene_global16.py` — whole-scene LagerNVS render from GLOBAL context: pick
  k=16 views by coverage over Monte-Carlo points sampled inside a sphere centered at the views'
  average look-at point (least-squares ray intersection); render ALL scene views (strided) as
  targets from those 16. Outputs GT|render video + context-selection top-down (context, look-at
  center, sphere). 3 scenes → target-coverage 1.00.
- `scripts/render_scene_compare.py` — 3-panel stitched comparison [GT | per-seg-scale |
  firstfix+scene_span] per scene: renders each segment with BOTH context-selection strategies
  and concatenates for direct visual A/B (GT kept on the left).
- `scripts/viser_val_cameras.py` — viser frustum viewer for validation-saved cameras
  (results/<exp>/test): pred (red) / target-ref (blue) / rest-of-scene (grey), with a
  sequence slider + Load + Next. Handles coords (saved OpenGL c2w -> OpenCV for viser;
  ref/pred recovered to the scene world so all three overlay; verified ref≡scene dist=0).

### Fixed
- **Context-selection leakage**: `frustum_cover` context selection anchored on the target
  MIDPOINT and scaled by the target segment's own extent (`seg_scale` over [s:e]) — i.e. it
  used the yet-to-be-generated target trajectory, unreproducible at inference. New
  `geo_anchor_first_frame` option (config, `_sample_geo_frustum_cover`, `dump_coverage_selk`)
  anchors on the target's FIRST frame only (known at inference) and scales the radius by the
  CONTEXT movement; `frustum_cover_select` gains `look_centroid=` to keep the look direction
  target-free. `config_geo_camscale` now sets `geo_anchor_first_frame=True`. Legacy behavior
  preserved when the flag is off. Blacklist regenerated with the honest metric (backup:
  data/blacklist_selk_tau0.7.TARGETANCHORED.bak.csv).

### Added
- `geo_first_view_target_s` config option: geo context view0 = target segment's first camera s
  (+ (k-1) out-of-seg retrieved), so LagerNVS anchors the geo latent to frame s (origin-aligned
  with the generation target frame). `config_geo_viewS.py` + `data/blacklist_selk_viewS_tau0.7.csv`
  (coverage recomputed with view0=s over [s+1:e]; `dump_coverage_selk.py --first-view-s`).
- `scale_mode` config option (`config.py`, default `'target_cam'`) selecting the camera
  translation normalization scale. New `'context_longer'` mode computes `cam_avg_scale`
  from the LONGER out-of-segment side, chunked into `num_frames` windows, averaging each
  window's camera movement (`dataset_dl3dv._cam_avg_scale_context`). Leakage-free and
  reproducible at inference from context only. Existing `'target_cam'` path unchanged.
- `config_textonly_camscale.py` — text-only run using `scale_mode='context_longer'`.
- `scripts/dump_coverage.py`, `scripts/viz_coverage.py`, `scripts/viz_coverage_dump.py`,
  `scripts/viz_coverage_compare.py` — out-of-segment coverage dump + statistics/plots
  (`both`앞+뒤 vs `longer`긴 쪽만 context) for coverage-based segment blacklisting.
- `scripts/render_target_from_context.py` — LagerNVS NVS probe: render target segment
  from out-of-segment (longer-side) context only, compare to GT, across coverage bins.
- `scripts/render_target_pose_ablation.py` — posed vs unposed context cam_token A/B
  (LagerNVS eval default is posed; renders sharper/more consistent with context poses).
- `scripts/render_scene_stitched.py` — per-scene stitched NVS over all segments using
  coverage-aware out-of-segment context (frustum_cover on the longer side) + posed
  cam_token; also per-segment context-selection visualization.
- `geo_cover_out_of_seg` / `geo_posed` config options. `frustum_cover_select` gains an
  `allowed=` param (restrict candidates to a frame subset, e.g. out-of-segment). Dataset
  restricts frustum_cover to the longer out-of-segment side and emits geo-view geometry
  (`geo_c2w`/`geo_fxfycxcy`/`geo_hw`); `geo_encoder.build_cam_token` builds the lagernvs
  posed 11-dim cam_token (1.35*max norm, extri_intri_to_pose_encoding). All default OFF
  (existing unposed/in-segment behavior unchanged).
- `config_geo_camscale.py` — geo(lagernvs) posed + out-of-segment frustum_cover +
  `scale_mode=context_longer` (full-sequence, not AR).
- `geo_shuffle_order` config option (permute selected geo context view order each access;
  posed -> VGGT reference view changes). `config_geo_camscale_shuf.py` = shuffle ON.
- Per-SEGMENT coverage blacklist: `coverage_blacklist_path` config + dataset
  `_read_coverage_blacklist` skipping (scene, segment) pairs (NOT whole scenes).
  `scripts/dump_coverage_selk.py` (K=6 frustum_cover selected coverage — matches actual
  geo input) + `data/blacklist_selk_tau0.7.csv` (13,166/40,059 segments removed at selK<0.7).
- `scripts/render_borderline.py` — render specific (scene,segment) pairs (e.g. tau
  borderline survivors) to judge a filtering threshold.
- `frustum_cover_select` gains `ball_center=` (explicit coverage-ball center, e.g. frame s
  omnidirectional) and `look_centroid=` (target-free look direction). `config_geo_ballS.py`
  (ball@frame-s selection, no shuffle) + `data/blacklist_selk_ballS_tau0.7.csv`.
- `scripts/select_compare.py`, `scripts/render_select_headtohead.py` — ours vs I3DM-style
  (MC FOV-overlap) context selection (geometry + rendered A/B).
- `scripts/render_ar_continuity.py` — LagerNVS chunk-wise AR continuity render: rolling
  causal memory (source ∪ generated-so-far) vs static source-only, side by side.
- `scripts/render_ar_viz.py` — per-chunk anchor & context (source vs gen-past) top-down viz.
- `scripts/render_ar_anchor.py` — coordinate-anchor ablation: force LagerNVS view0 to the
  current chunk's first camera (A: re-anchor) vs the segment's first camera (B: fixed frame).
  Also emits a per-chunk context viz.
- `scripts/render_ar_scene.py` — whole-scene AR: chunk = a full 49-frame segment, render ALL
  segments of a scene in order with rolling causal memory (previous segments), concat to one
  long video; A(re-anchor)/B(fixed) anchor ablation + per-segment context viz.
- `scripts/render_scene_longer.py` — whole-scene NON-AR: per segment, context = K views from
  the full LONGER out-of-seg side only (ball@s), render all segments and concat; + per-segment
  context viz (shows the longer side auto-flipping after↔before across the scene).
- `scripts/render_scene_longer_anchor.py` — same longer out-of-seg context + the view0 anchor
  ablation A(view0=seg start) vs B(view0=scene start fixed); [GT|A|B] whole-scene concat.
- `exp_results/` — organized experiment artifacts (coverage figures/dumps/blacklists,
  render montages+videos, README index).
