# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Dynamic camera and video generation with from source video and user prompt.

## Don't

- 실험 결과를 임의로 요약 금지 (wandb 원본 수치 그대로).
- GPU 5~7 사용 금지. `CUDA_VISIBLE_DEVICES`는 항상 0~4 범위에서만 지정할 것 (2026-09-06 사용자 지시).
  - **예외: TRUMANS 작업에 한해 GPU 5 사용 가능** (2026-09-20 사용자 지시). 뱅크 굽기·Blender 렌더·
    TRUMANS 캡션용 vLLM 이 여기 해당한다. 6,7 은 여전히 금지이고, TRUMANS 가 아닌 학습/추론은
    0~4 그대로다. CinemaTraj `run_bank.py` 는 `BANK_GPU_ALLOW=5` 를 줘야 `--gpu 5` 를 받는다.

## Coding conventions

- 기존의 핵심 작동 구조가 있다면 option에 따라 분기를 쳐서 기존의 방식도 똑같이 돌아가게끔 유지.

### bash 지양, python 우선

새 스크립트는 **python 으로 쓴다**. bash 는 "python 하나를 인자만 바꿔서 부르는 래퍼"가 되기 쉽고,
그런 래퍼는 실험(dNN)마다 복붙되어 서로 몇 줄만 다른 파일이 쌓인다.

- 다단계 파이프라인(graph→cloud→route→fit→emit 같은 것)은 **하나의 python 드라이버 + `--stage` 인자**로 만든다.
  실험별 차이는 스크립트를 새로 만들지 말고 **설정(JSON/argparse 기본값)으로** 표현한다.
- 샤딩·screen 기동처럼 bash 가 불가피한 부분만 얇게 남기고, 로직은 python 에 둔다.
- 일회성 조사 코드는 파일로 남기지 않는다. 재사용 가치가 생긴 시점에만 `scripts/` 에 승격한다.

### 임시 파일은 `<repo>/tmp/`

`/tmp` 는 **쓰지 않는다** — 시스템이 임의로 비울 수 있어 scene 목록·로그가 조용히 사라진다.
대신 `/data1/cympyc1785/LatentCamVid/tmp/` 아래에 작업 단위 하위 폴더를 만든다 (예: `tmp/d149/`).
gitignore 대상이며, 작업이 끝나면 남길 것만 `results/` 로 옮기고 나머지는 정리한다.

### 산출물은 압축적으로

파일 개수와 용량이 계속 느는 것을 기본 실패 모드로 간주한다.

- 로그는 **전량 tee 하지 말고** 진행 표시·경고·요약만 남긴다. 학습 수치는 wandb 가 원본이므로 stdout 을 통째로 저장하지 않는다.
- 세대별 산출물(`bank_dNN`, `hole_bank_dNN` 등)은 최신 세대 + 대조에 실제로 쓰는 세대만 남긴다.
- 중간 산출물을 지울 때는 **먼저 목록과 근거를 사용자에게 보고하고 승인을 받은 뒤** 지운다.

## Specifics

- 코드 구조 및 기타 프로젝트 convention들은 `SPECS.md`를 이용한다.

## Changelog

이 프로젝트는 [Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

코드를 변경할 때마다 `CHANGELOG.md`의 `[Unreleased]` 섹션에 항목을 추가할 것.

## 작업 흐름

### 실행은 Claude 가 끝까지 한다

**사용자에게 명령어를 대신 실행해 달라고 넘기지 않는다.** 계획 단계에서 "이 명령은 내가 직접 돌릴 수 있는가"를
먼저 확인하고, 못 돌리는 형태라면 돌릴 수 있는 형태로 계획을 바꾼다. 사용자에게 넘기는 것은 마지막 수단이고,
그때는 왜 자동으로 못 하는지를 함께 설명한다.

코드 변경 작업이 끝나면 **반드시** 다음을 수행한다:
1. 변경 내용을 `CHANGELOG.md`의 `[Unreleased]`에 기록
2. 사용자에게 어느 카테고리에 추가했는지 알려줄 것

이 단계를 건너뛰지 말 것. 사소한 변경이라도 사용자에게 영향이 있으면 기록한다.

학습 진행 시 다음 과정을 따른다:
1. 학습 설정이 이전과 어떻게 달라졌는지 사용자의 지시에 맞게 변경되었는지 이상이 생길만한 부분은 있는지 확인하여 이상이 없을 시 wandb run name과 함께 `EXPERIMENTS.log`에 기록한다.
2. 먼저 smoke test를 진행하여 학습 코드가 정상적으로 돌아가는지 확인 후 이상이 있다면 고친 후 `FIX.log`에 기록 후 사용자에게 보고한다. (smoke test 시에는 wandb logging 꺼두기)
3. 사용자의 명시적 지시가 없으면 GPU 0~4 중 가장 여유가 있는 GPU만을 사용한다.
4. screen train1~4에서 돌리고 어떤 screen에 어떤 학습이 돌아가고 있는지 기억한다.
5. 학습을 돌려놓고 모니터링을 걸어 주기적으로 확인하여 정상적으로 학습이 돌아가고 있는지 확인하고 이상이 있다면 고친 후 `FIX.log`에 기록 후 사용자에게 보고한다.

추론 진행 시 다음 과정을 따른다:
1. 모델을 학습했을 때의 config와 일치하는지 먼저 확인하고 다른 설정값이 사용자의 명시적 지시에 의한 것이 아니라면 학습했을 때의 설정을 따르고 사용자에게 고지해준다.
2. 사용자의 명시적 지시가 없으면 GPU 0~4 중 가장 여유가 있는 GPU만을 사용한다.
3. screen infer1~4에서 돌리고 어떤 screen에서 어떤 추론이 돌아가고 있는지 기억한다.
4. 결과물은 results에 저장하고 추론 당시의 config값들과 input을 같이 결과 폴더 안에 정리하여 저장한다.

추론 결과물을 후처리 시 다음 과정을 따른다:
1. `scripts`의 기존의 후처리 코드에 사용자가 지시한 역할을 하는 코드가 있는지 찾고 있다면 최소한의 수정으로 고쳐서 사용한다.
2. 해당 역할의 코드가 없다면 새로 코드를 scripts에 작성하여 실행한다.

## Git 커밋 워크플로우

역할 분담: **Claude = 현재 브랜치에 커밋까지, 사용자 = push.** Claude는 브랜치를 새로 만들지 않고 push도 하지 않는다. (이 프로젝트는 "요청받을 때만 커밋" 기본 동작과 "default 브랜치면 먼저 분기" 동작을 의도적으로 override 한다 — 코드 변경 작업이면 분기 없이 현재 브랜치에 자동 커밋.)

### Claude가 자동으로 하는 것
1. 코드 변경 작업이 끝나면 **현재 브랜치에 그대로 커밋**한다 (브랜치 분기 금지).
2. 코드 변경 + `CHANGELOG.md [Unreleased]` 갱신을 **하나의 커밋**으로 묶는다. 커밋 메시지는 `<type>: <한 줄 요약>` (`feat` 새 기능 / `fix` 버그 / `exp` 실험·ablation / `chore` 리팩터·문서·잡일; 예: `feat: ac3d camera_input_type 추가`) + 필요 시 본문에 why.
3. **스테이징은 변경한 파일만 명시적으로 `git add <경로>`.** `git add -A` / `git add .` **금지** — vendored 디렉토리(`src/model/{DiffSynth-Studio,ac3d,GEN3C}`, `WorldTraj`; DiffSynth-Studio는 ~417GB)가 인덱스에 끌려들어가는 사고를 gitignore 상태와 무관하게 차단.
4. 커밋 후 **커밋 요약을 사용자에게 보고하고 멈춘다.**

### Claude가 하지 않는 것 (전부 사용자가 수동)
- `git push`, `git merge`, `git pull`, 브랜치 생성/삭제.

### 커밋하지 않는 경우
- 읽기 전용 / 분석 / 질문 답변 등 코드 변경이 없는 작업.
- 사용자가 명시적으로 "커밋하지 마"라고 지시한 경우.