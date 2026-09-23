# Changelog

저장소 **최상위** 문서·설정의 변경 기록. 하위 트리의 변경은 각자의 CHANGELOG 에 적는다 —
`camera_generation/dataset/CHANGELOG.md`, `camera_generation/latentcam/CHANGELOG.md`,
`video_generation/CHANGELOG.md`. 여기에 오는 것은 세 트리에 **걸쳐 있는 것**뿐이다
(`pipeline.md`, `README.md`, `.gitignore`, `CLAUDE.md`).

[Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

## [Unreleased]

### Added

- **`context.md` — 다음 세션이 알아야 할 것** (R14). `CLAUDE.md`(규칙) · `pipeline.md`(지도) ·
  `.claude/goals.md`(목표) 어디에도 안 들어가는데 **모르면 사고가 나는 것**을 모았다.
  7 절: 서 있는 지시 / 문서 간 어긋남 / 세대 이름 / 비교 금지 지표 / 반복된 실패 양식 /
  env 대응 / 열려 있는 것.
  - **`SPECS.md` 가 없다.** `CLAUDE.md` 의 `## Specifics` 와 `.claude/goals.md` 가 둘 다
    이 파일을 가리키는데 저장소에 존재하지 않는다. 두 문서가 가리키는 곳이 빈 곳이라는
    사실 자체를 기록했다.
  - **`CLAUDE.md:182`·`:189` 가 "GPU 0~4" 라고 쓴다** — 같은 파일 `## Don't` 의 0~3 과
    충돌한다. 어느 쪽이 이기는지(`## Don't`)를 명시.
  - **`camera_generation/dataset/summary.md` 헤더가 두 겹으로 틀렸다** — "gitignore 라
    커밋에 안 들어간다"(R7 이후 추적 중) + 인용한 `.gitignore:222` 가 지금은 `tools` 줄이다.
  - 사용자가 한 번 말하고 계속 유효한 지시 9건을 표로 (릴 단위 seed 3개 concat, GenDoP
    `--raw`, GPU 0~3 예외 전부 만료, 학습 중지는 Ctrl+C 먼저, 대기열 자동 진행 등).
  - **반복해서 밟은 실패 양식 셋**을 실측 건수와 함께 — ① 참조가 코드를 안 따라간다
    (R9 40건 중 36건 / R12 3건 / R13 18건) ② agent 보고서가 매번 틀린다 (R8 5건 · R10 4건 ·
    R12 3건 · R13 4건) ③ **rc=0 은 "됐다" 가 아니다** (nested repo `git add`, `eval_dir`
    태그 누락, `route` 없는 `tau` 굽기, `--videos all`).
  - `file:line`·경로 전량 스크립트 검산 (문제 0건) + 표 행 파손 검사.

- **`pipeline.md` — 저장소 전체 지도** (R13). 지금까지 비어 있던 파일을 채웠다. 네 단계
  (① 데이터 구축 `camera_generation/dataset/` → ② 학습 `camera_generation/latentcam/` →
  ③ 평가 → ④ 영상 생성 `video_generation/models/Vista4D/`) 의 **위치와 기동 방법만** 적고,
  내부 규약은 기존 하위 문서 여섯(합 2,673행)으로 넘긴다. 복사하면 두 곳이 갈라지기 때문이다.
  - **§3.2 베이스라인 배선** — GenDoP 과 E.T. 는 경로가 다르다. GenDoP 은
    `run_gendop_eval.py` 의 `ARMS`(`:297-300`) 로 직접 돌리고, E.T./DIRECTOR 는 arm 이 아니라
    **이미 만들어진 eval 폴더**를 `--extra_eval_dir LABEL=DIR`(`:510`) 로 같은 표에 올린다
    (`--arm none` 이면 GenDoP 을 하나도 안 돌린다).
  - **`--raw` 가 기본값이라는 사실을 문서에 못박았다** — `run_gendop_eval.py:496-497` 에서
    `--rescale` 의 default 가 `False` 로 뒤집혀 있다 (2026-09-21 지시). 최상위에서 치는 건
    `--raw` 이고 `--no_rescale` 은 하위 스크립트로 자동 전달되는 인자다(`:421-422`).
    조사 보고서가 이 배선을 "미추적" 으로 적었던 것을 직접 읽어 반증했다.
  - **`evaluation/` 은 빈 폴더다** (`find evaluation -mindepth 1` → 0건). 평가 코드를 거기서
    찾다가 "없다" 고 결론내는 사고를 막으려고 최상위 표와 지뢰 표 양쪽에 적었다.
  - **`DATA` 는 심볼릭 링크다** (`-> /data1/cympyc1785/data`). `du`·`find` 가 저장소 밖으로 샌다.
  - 최상위 디렉토리 13종의 역할과 **gitignore 추적 여부**를 `.gitignore` 행 번호와 함께 표로.
  - 문서 안의 `file:line` 전량을 스크립트로 검산했다. 상대경로 18건을 저장소 기준 경로로
    고쳤다 — R12 에서 `FIX.log`·`config.md` 의 참조가 코드 이동으로 어긋나 있던 것과 같은 실패다.

- **이 파일(`CHANGELOG.md`) 자체** — 최상위에는 CHANGELOG 가 없었고 하위 트리 셋에만 있었다.
  세 트리에 걸친 문서(`pipeline.md`)를 어느 한 트리의 기록에 넣으면 찾을 수 없으므로 새로 연다.
