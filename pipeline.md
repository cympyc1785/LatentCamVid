# pipeline.md — LatentCamVid 전체 지도

**이 문서는 지도다. 설명은 하지 않는다.** 각 단계가 *어디에 있고* *무엇으로 기동하고*
*자세한 건 어느 문서에 있는지*만 적는다. 단계 내부의 규약·수식·지뢰는 전부 하위 문서에 있고,
여기서 그걸 복사하면 두 곳이 갈라진다 (실제로 `video_generation/FIX.log` 와 Vista4D `config.md` 의 `file:line` 이
그렇게 갈라졌다 — R12 에서 3건 수정).

| 물음 | 답 |
|---|---|
| 단계는 몇 개인가 | **넷.** 데이터 구축 → 학습 → 평가 → 영상 생성. 앞 셋은 `camera_generation/`, 마지막은 `video_generation/` |
| 하위 문서는 몇 개인가 | **여섯.** 아래 §0 표. 합 2,673행 (이 문서 제외) |
| `evaluation/` 은 무엇인가 | **빈 폴더다.** 파일이 하나도 없다 (`find evaluation -mindepth 1` → 0건). 평가 코드는 `camera_generation/latentcam/scripts/eval/` 과 `camera_generation/dataset/exec/run_gendop_eval.py` 에 있다 |
| `DATA/` 는 무엇인가 | **심볼릭 링크다** — `DATA -> /data1/cympyc1785/data`. 저장소 안이 아니다 |
| 베이스라인은 몇 종인가 | **둘.** GenDoP(자체 arm 2개) 와 E.T./DIRECTOR(외부 eval 폴더). 배선이 서로 다르다 — §3.2 |
| GenDoP 의 `rmax` rescale 은 | **기본이 raw 다.** `--raw` 가 기본값이고 옛 동작은 `--rescale` 을 명시해야 나온다 (`camera_generation/dataset/exec/run_gendop_eval.py:496-497`) |
| 세대 이름 규약 | `dNNN` 하나로 데이터·모델을 다 나눠 왔다. 분리 요청이 접수돼 있고 아직 미적용 — §부록 |

---

## 0. 한 장 요약

```
                 camera_generation/                              video_generation/
  ┌────────────────────┬────────────────────┬──────────────┐   ┌──────────────┐
  │  ① 데이터 구축      │  ② 학습            │  ③ 평가       │   │ ④ 영상 생성   │
  │  dataset/          │  latentcam/        │  latentcam/  │   │ models/      │
  │                    │                    │  + dataset/  │   │ Vista4D/     │
  ├────────────────────┼────────────────────┼──────────────┼───┼──────────────┤
  │ 영상 → 뱅크 → 코퍼스│ 코퍼스 → ckpt      │ ckpt → 지표   │   │ 카메라 → 영상 │
  │ exec/run_bank.py   │ main/train_latent_ │ scripts/     │   │ exec/run_    │
  │ exec/run_corpus_   │   cam_dm.py        │  eval_test   │   │  vista4d_    │
  │   export.py        │                    │  set.py      │   │  gen.py      │
  └────────────────────┴────────────────────┴──────────────┘   └──────────────┘
         ↓ 코퍼스              ↓ results/<run>/     ↓ eval_my/<tag>/   ↓ mp4
   DATA/<corpus>/latentcam_*   last.pth            metrics.json
```

**하위 문서 여섯 — 자세한 건 전부 여기 있다.**

| 문서 | 행 | 무엇을 답하나 |
|---|---|---|
| `camera_generation/dataset/pipeline.md` | 587 | ① 의 **구조**. 영상 1편 → (source video, target camera, caption). Part A 실사 / Part B TRUMANS |
| `camera_generation/dataset/USAGE.md` | 219 | ① 의 **실행**. cwd 규칙, 자주 치는 명령, 산출물 규약, 지뢰 |
| `camera_generation/dataset/exec/USAGE.md` | 671 | ① 의 **드라이버별 인자 사전**. `exec/*.py` 를 하나씩 |
| `camera_generation/latentcam/pipeline.md` | 463 | ② 의 **모델 구조**. CameraVAE + 8층 DiT, arm 축 5개, 모듈별 생사 판정 |
| `camera_generation/latentcam/train.md` | 424 | ②③ 의 **실행**. 기동 한 줄, 손실, wandb 키 사전, 평가 두 층위 |
| `video_generation/Vista4D_USAGE.md` | 309 | ④ 의 **실행**. 경로 셋, 카메라 규약, local 패치 2건 |

> **이 문서는 위 여섯을 대체하지 않는다.** 어느 단계를 실제로 돌리려면 해당 문서를 연다.
> 여기서 명령을 복사해 쓰지 말 것 — 인자는 하위 문서가 정본이다.

---

## 1. ① 데이터 구축 — `camera_generation/dataset/`

### 1.1 입력과 출력

| | |
|---|---|
| 입력 | 실사 영상 (Vista4D / DynPose-100K) 또는 합성 씬 (TRUMANS) |
| 출력 | latentcam 학습 코퍼스 — `(source video, target camera, caption)` 삼중항 |
| 산출 위치 | `dataset/out*/` (gitignore, 433 GB — `.gitignore:236-239`), 코퍼스는 `DATA/<corpus>/latentcam_*` |

### 1.2 두 드라이버

**뱅크 체인** — `camera_generation/dataset/exec/run_bank.py`. 단계 여섯: `graph → cloud → route → tau → fit → emit`.
세대마다 스크립트를 만들지 않고 **`configs/bank/<gen>.json` 한 장**으로 가른다
(`camera_generation/dataset/exec/run_bank.py:1-30` 의 docstring 이 왜 그런지를 적어 둔다 — bash 러너가 23개까지 늘었던 게 발단).

**코퍼스 export** — `camera_generation/dataset/exec/run_corpus_export.py`. 단계 다섯:
`desc → merge_desc → captions → export → verify` (`camera_generation/dataset/exec/run_corpus_export.py:226-227`).
`desc` 단계는 vLLM 이 떠 있어야 한다.

> **vLLM 은 상시 기동 금지.** 단계 직전에 `camera_generation/dataset/exec/serve_qwen3vl.sh` 로 올리고 끝나면 내린다.

전체 실행 순서는 `camera_generation/dataset/pipeline.md:536`(실사) 와 `:558`(TRUMANS).

### 1.3 갈래 둘

| 갈래 | 소스 | 문서 |
|---|---|---|
| Part A — 실사 | Vista4D / DynPose-100K. DA3 recon + SAM3 분할 + GeoCalib 중력 | `camera_generation/dataset/pipeline.md:56` 부터 |
| Part B — 합성 | TRUMANS. mesh GT 가 있어 충돌·지면·subject 를 GT 로 판정 | `camera_generation/dataset/pipeline.md:347` 부터 |

---

## 2. ② 학습 — `camera_generation/latentcam/`

### 2.1 기동

정본은 `camera_generation/latentcam/train.md:58`. 세 가지가 **전부 필수**다.

- `cd main` — hydra 가 `main/conf/` 를 잡고 `result_dir` 이 상대경로다 (`camera_generation/latentcam/main/train_latent_cam_dm.py:572`).
- `PYTHONPATH=<repo>:.` — 빠지면 `ModuleNotFoundError: utils` 로 즉사. **두 번 밟은 실수다.**
- conda env `latentcam` 의 인터프리터.

대기열은 `camera_generation/latentcam/scripts/train/queue_runs.py` (명령 문자열을 만드는 것도 여기 — `:99-101`).

### 2.2 규칙

| | |
|---|---|
| GPU | **0~3 만.** `CLAUDE.md` 의 `## Don't` (`camera_generation/latentcam/train.md:82`) |
| screen | 학습 `train1~4`, 추론 `infer1~4`. 무엇이 도는지는 `.claude/watch.md` 에 **적는다** |
| epoch 상한 | **50.** 마지막 인덱스 49, 최종은 `last.pth` (`camera_generation/latentcam/train.md:173`) |
| smoke | `WANDB_MODE=disabled` (`camera_generation/latentcam/train.md:132`) |
| 로그 | `<repo>/tmp/` 아래. `/tmp` 는 시스템이 비운다 |

### 2.3 arm 은 yaml 한 장

`main/conf/experiment/<exp>.yaml` 이 arm 을 정의한다. 축 다섯(geo / video / 조건추가 / text /
코퍼스)은 `camera_generation/latentcam/pipeline.md:236` 부터. 모델 본체(CameraVAE + 8층 DiT)는 `:67` 부터.

---

## 3. ③ 평가

### 3.1 두 층위 — 섞으면 안 된다

`camera_generation/latentcam/train.md:198` 이 정본. 요약만:

| 층위 | 무엇 | 표본 |
|---|---|---|
| 학습 중 val | `train_latent_cam_dm.run_validation` | `val_max_batches`(20) × `batch_size`(8) = **160개** |
| 전량 평가 | `camera_generation/latentcam/scripts/eval_testset.py` | held-out split **전량**(~4k) |

> 학습 중 val 160 은 1 sd 스윙이 arm 간 격차를 삼킨다 (`camera_generation/latentcam/scripts/eval_testset.py:3-6`
> 이 이유로 이 스크립트가 존재한다). arm 비교는 전량 평가로.

**체크포인트는 `last.pth`.** `best.pth` 평가 금지.
**학습 중 eval 은 한 번에 하나** — `/data1` 이 Lustre 라 동시 eval 이 학습을 D-state 로 문다.

### 3.2 베이스라인 비교 — GenDoP 과 E.T.

중앙 드라이버는 **`camera_generation/dataset/exec/run_gendop_eval.py`** (550행).
단계 넷 (`--stage`):

| 단계 | 함수 | 하는 일 |
|---|---|---|
| `inputs` | `:352 stage_inputs` | rgbd 프레임/depth 를 GenDoP 입력 구조로 |
| `infer` | `:365 stage_infer` | GenDoP ckpt 로 c2w 궤적 생성 |
| `evaldir` | `:402 stage_evaldir` | 예측 → 우리 eval 폴더 규약(`_transforms_{pred,ref}.json` + `_caption.json`) |
| `score` | `:430 stage_score` | 지표 산출 |

- **GenDoP arm 둘** — `ARMS` (`:297-300`): `gendop_text`(`text_motion.safetensors`) /
  `gendop_rgbd`(`text_rgbd.safetensors`).
- **코퍼스 11종** — `CORPORA` (`:64`). vista / dynpose 세대별로 코퍼스 경로·eval_data 루트·
  `depth_norm`·우리 arm 셋을 한 곳에 못박는다.
- **E.T./DIRECTOR** — arm 이 아니다. **`--extra_eval_dir LABEL=DIR`** (`:510`) 로 *이미 만들어진*
  eval 폴더를 같은 표에 올린다. `--arm none` 이면 GenDoP 을 하나도 안 돌리고 외부 폴더만 본다
  (`:434` docstring, `:477` 주석).
- 이 드라이버는 기존 스크립트 넷을 **고치지 않고 조립**한다 —
  `camera_generation/dataset/eval/` 의 `dynpose_gendop_inputs.py` · `gendop_release_infer.py` ·
  `gendop_preds_to_eval_dir.py` · `eval_subject_in_frame.py`.

> **`--raw` 가 기본이다.** `:496-497` 에서 `--rescale` 의 default 가 `False` 로 뒤집혀 있다.
> 2026-09-21 지시 "gendop 돌릴때 rmax 곱하면 안되고 raw로 normalized 되어서 나오는 원본 코드
> 그대로 나온걸 저장해서 써야해" 가 근거고, 옛 판본은 `--rescale --scale_token` 을 명시하면
> 재현된다 (`:495` 주석). **`--no_rescale` 은 하위 스크립트로 내려가는 인자다** — 최상위에서
> 치는 건 `--raw` 다 (`:421-422` 가 전달한다).

### 3.3 CLaTr 지표

| 어디서 | 무엇 |
|---|---|
| 학습 중 · 전량 평가 | `main/evaluate/CLaTr` (`src.extraction`) → `main/evaluate/eval` (`src.eval_only`) 두 subprocess |
| 임의 폴더 | `camera_generation/latentcam/scripts/eval/clatr_score_dir.py` — 우리/GenDoP/E.T. 를 **같은 눈금**으로 |

> **게이지 주의.** CLaTr ckpt 와 standardization 을 바꾸면 숫자가 통째로 움직인다.
> **CLaTr 게이지는 학습 split 을 따라가므로 코퍼스가 다르면 절대값을 나란히 놓지 않는다**
> (`camera_generation/latentcam/scripts/eval/clatr_score_dir.py:14-18`). caption fscore 는 chance 대비로 읽는다.
> 표본이 한두 개면 FCD·PRDC 는 NaN 이고, 그때 읽을 수 있는 건 `clatr/clatr_score` 와
> caption precision/recall/fscore 뿐이다 (`:20-23`).

---

## 4. ④ 영상 생성 — `video_generation/models/Vista4D/`

카메라를 받아 영상을 만든다. 실행 경로가 셋이고 현재 쓰는 건
`camera_generation/dataset/exec/run_vista4d_gen.py`.
전부 `video_generation/Vista4D_USAGE.md` 에 있다.

> **이 트리는 git 추적 대상이 아니다.** `.gitignore:249` 가 `video_generation/models` 를 막고,
> 그와 별개로 `Vista4D/` 가 **자기 `.git` 을 가진 별도 저장소**라 `!` 예외로도 되살릴 수 없다
> (`.gitignore:247-248` 주석). 그래서 사용법 문서는 R15 에서 트리 **밖**으로 빼
> `video_generation/Vista4D_USAGE.md` 로 추적한다. **local 패치 2건은 여전히 재클론하면 사라진다** —
> 무엇을 고쳤는지는 그 문서에 적혀 있다.

---

## 5. 최상위 디렉토리 — 무엇이 어디에

| 경로 | 무엇 | 추적 |
|---|---|---|
| `camera_generation/dataset/` | ① 데이터 구축 | ✅ (`out*/`·`logs/`·`results/` 제외 — `.gitignore:236-239`) |
| `camera_generation/latentcam/` | ② 학습 · ③ 평가 | ✅ (`results/`·`eval_my/`·`data/`·`checkpoints/` 등 제외 — `.gitignore:251-263`) |
| `camera_generation/models/` | 논문 vendored 9종 (CCD, CamVLA, DIRECTOR, Director3D, GenDoP, I3DM, LAMP, Planner, SCVideo) | ❌ `.gitignore:230` |
| `camera_generation/tools/` | 외부 도구 11종 (Depth-Anything-3, GeoCalib, Grounded-SAM-2, molmo2, qwen3vl, trumans_utils 등) | — |
| `video_generation/models/` | ④ Vista4D 외 | ❌ `.gitignore:249` |
| `video_generation/tools/`, `video_generation/results/` | 도구 · 산출물 | ❌ `.gitignore:246,250` |
| `DATA` | **심볼릭 링크** → `/data1/cympyc1785/data`. 코퍼스·eval_data 의 실제 위치 | ❌ `.gitignore:220-221` |
| `tools/` | 최상위 도구 (ReCal3R, StreamVGGT, TTT3R, vipe, DynamicVerse, Depth-Anything-3) + 낱개 py | ❌ `.gitignore:222` |
| `results/` | 날짜별 산출 번들 | ❌ `.gitignore:223` |
| `scripts/` | `meeting/` 만 | ✅ |
| `meetings/` | 회의 자료 | ❌ `.gitignore:224` |
| `evaluation/` | **빈 폴더.** 파일 0개 | — |
| `tmp/` | 작업용 임시. **`/tmp` 대신 여기** | ❌ `.gitignore:273` |
| `.claude/` | 에이전트 전용 문서 (`request_queue.md` `tasks.md` `watch.md` `FIX.log` `EXPERIMENTS.log`) | ✅ |

---

## 부록 — 지뢰

| 지뢰 | 증상 | 회피 |
|---|---|---|
| `evaluation/` 이 비었다 | 평가 코드를 여기서 찾다가 "없다"고 결론 | 평가는 `camera_generation/latentcam/scripts/eval/` + `camera_generation/dataset/exec/run_gendop_eval.py` |
| `DATA` 가 심볼릭 링크 | `du`·`find` 가 저장소 밖으로 샌다 | `-P` 로 링크를 안 따라가게 |
| `--raw` 가 기본인 걸 모른다 | 옛 런과 비교하면서 rescale 이 걸린 줄 안다 | `camera_generation/dataset/exec/run_gendop_eval.py:496-497` 확인. 옛 재현은 `--rescale --scale_token` |
| `--no_rescale` 을 최상위에 친다 | argparse 가 거부 | 최상위는 `--raw`, `--no_rescale` 은 하위로 자동 전달 (`:421-422`) |
| `PYTHONPATH` 없이 학습 기동 | `ModuleNotFoundError: utils` | `cd main && PYTHONPATH=<repo>:.` |
| Vista4D 안에서 커밋 시도 | `git add` 가 rc=0 인데 아무 일도 안 일어난다 | 별도 `.git` 이다 — 바깥 저장소가 못 본다 |
| CLaTr 절대값을 코퍼스 간 비교 | 눈금이 다른데 표 하나에 올린다 | 같은 인자·같은 split 일 때만. caption fscore 는 chance 대비 |
| `dNNN` 세대 이름이 데이터·모델을 안 가른다 | "d200 이 데이터냐 모델이냐" | **미해결.** `datagen_NNN` / `modelgen_NNN` 분리 + 코퍼스별(trumans/dynpose/vista) 이름 분기 요청이 접수돼 있다 |
| 학습 중 val 160 을 arm 비교에 쓴다 | 1 sd 스윙이 격차를 삼킨다 | `camera_generation/latentcam/scripts/eval_testset.py` 로 전량, `last.pth` |
| 동시 eval | Lustre 가 학습을 D-state 로 문다 | 한 번에 하나 |
| `/tmp` 에 로그 | 시스템이 비우면 모니터가 조용히 무력화 | `<repo>/tmp/<작업>/` |
