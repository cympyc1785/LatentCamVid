# Request Queue

사용자 요청을 접수 → 진행 → 완료 순으로 옮겨 적는 단일 대기열이다.
규약은 `CLAUDE.md` 의 `### 요청 대기열 (.claude/request_queue.md)` 절에 있다.

- **Request** — 접수만 된 것. 아직 손대지 않았다.
- **Working** — 지금 수행 중인 것. **동시에 하나만** 둔다.
- **Wait** — 긴급 요청 때문에 **중단하고 비켜 둔 것.** 어디까지 했는지 한 줄 적어 두고,
  긴급 건이 끝나면 다시 `Working` 으로 돌린다.
- **Done** — 끝난 것. 한 줄 진행 내역을 같이 적는다.
- **Incomplete** — 보류·차단된 것. **사유를 반드시 적고**, 다음 request 로 넘어간다.

---

## Request

### R2. 미완료 요청을 `tasks.md` 에 저장
진행 안 됐거나 완료 안 된 내 요청들을 `tasks.md` 로.

### R3. `CinemaTraj/out*` 잉여 폴더 정리
최근에 안 쓴 / 필요 없는 out 폴더 삭제. **중요 파일로 판단되면 승인받을 것.**

### R4. `CinemaTraj/results` 불필요 폴더 정리

### R5. `tmp/` 를 `log` / `scripts` / `results` 로 재정리

### R6. scripts/ 재분류 (exec / fit / eval / viz) — 계획 승인 필요
`camera_generation/models/Planner/CinemaTraj/scripts` 에 스크립트가 전부 쌓이고 있다. 지침:
- run 계열 → `CinemaTraj/exec/` + `USAGE.md` (파일별 실행법)
- 데이터 구축에 **실제로 필요한 것만** → `CinemaTraj/fit/` + `USAGE.md`
- eval 계열 → `CinemaTraj/eval/` + `USAGE.md`
- 시각화 계열 → `CinemaTraj/viz/` + `USAGE.md`
- 그 외는 `scripts/` 에 남긴다. 추가 분류가 필요하면 제안할 것.
**먼저 간단한 계획을 세워 승인받고 진행.**

### R7. 데이터 구축 파이프라인을 `camera_generation/dataset/` 으로 이전
`CinemaTraj` 는 원래 논문 폴더다. 우리 파이프라인을 `camera_generation/dataset/` 로 옮기고 경로도 전부 수정.
- ⚠ R6 과 순서 의존: R6(폴더 재분류) 먼저 → R7(통째 이전).
- ⚠ R9 와 충돌 가능: 이전 후에는 문서도 새 위치에 있어야 한다. R7 진행 시 확인.

### R8. CinemaTraj md 파일 정리
현재 파이프라인과 무관한 내용 삭제.

### R9. CinemaTraj `pipeline.md` 재작성 + `USAGE.md` 작성
현재 데이터 구축 파이프라인 기준. (R7 이전 후라면 새 위치 기준)

### R10. `camera_generation/latentcam/pipeline.md`
모델 전체 구조도, 변형 구조들, 모듈별 설명.

### R11. `camera_generation/latentcam/train.md`
학습 방법·구조, 평가 방법·metric, 주의사항.

### R12. Vista4D `USAGE.md`
해당 모델을 어떻게 돌렸는지.

### R13. `/data1/cympyc1785/LatentCamVid/pipeline.md`
전체 파이프라인(데이터 구축 → 학습 → 평가, GenDoP·E.T. 비교 포함)의 위치와 구조.

### R14. `/data1/cympyc1785/LatentCamVid/context.md`
컨텍스트에 남길 추가 사항.

### R15. 최종 git 정리 + 커밋
데이터는 추적 제외, 나머지 미추적 수정사항은 추적되게 한 뒤 커밋.

### R17. 타임라인 문서는 `.md` 말고 `.log` 로
`fix`, `decision` 처럼 시간순으로 계속 늘어나는 문서는 확장자를 `.log` 로 바꾼다.
대상: `CinemaTraj/fix.log` · `CinemaTraj/DECISIONS.log` (+ `LBM_DEFECTS.log`,
`latentcam/docs/known_issues.md` 는 제안). `CHANGELOG.md` 4종은 Keep a Changelog 규약이라 제외.

### R18. 긴급 끼어들기 규약 (`## Wait` 구역)
`[긴급]` 접두사 또는 "이것부터 먼저" 요청이 오면 하던 것을 `## Wait` 로 비켜 두고 긴급 건을
먼저 처리한 뒤 복귀. 큐 파일과 `CLAUDE.md` 양쪽에 규약 기록.

---

## Working

### R7. 데이터 구축 파이프라인을 `camera_generation/dataset/` 으로 이전

---

## Wait

_(비어 있음)_

---

## Done

### R6. scripts/ 재분류 (exec / fit / eval / viz) — 완료
`scripts/` 174개 → `exec/` 11 · `exec/_legacy/` 34 · `fit/{ingest 18, graph 6, bank 13,
caption 8, convert 3}` · `eval/` 37 · `viz/` 35, 잔류 8 (LBM 데모·프로브). 전부 `git mv`.
`USAGE.md` 4종 신규 (exec 671 · fit 1,790 · eval 992 · viz 1,114줄) + `eval/GENDOP_USAGE.md`.
참조 434곳/186파일 재작성 + 정규식이 못 보는 16곳 개별 수정 (문자열 분할 8 · `from scripts import` 1 ·
sys.path 해킹 5 · 동적 로딩 2). 검증: ast.parse 0 오류 · 진입점 10개 `--help` 통과 ·
남은 `scripts/` 참조는 전부 정당(잔류 8개 + latentcam/video_generation 쪽 외부 경로).
`.gitignore` 에 `exec/ fit/ eval/ viz/` 화이트리스트 추가 (안 열면 새 `USAGE.md` 가 추적 밖).

### R3. `CinemaTraj/out*` 잉여 폴더 정리 — 완료 (2026-09-23)
`rm -rf` 권한이 풀린 뒤 승인 목록 8종(`out/trumans_recon_lens{25,18}` ·
`out/trumans_recon_lbm{,_n}` · `_quarantine_race` · `out_smoke` · `out_geocalib_trumans_check` ·
`out_dynpose_probe`)과 `out_trumans/*/` 구세대 뱅크 12종이 전부 사라진 것을 확인했다 (잔여 0).
보존 확정분(`out/sweep66` · `out_trumans/*/cloud.npz` · `out_dynpose` · `sheets_full`)은 그대로.

### R4. `CinemaTraj/results` 불필요 폴더 정리 — 완료 (2026-09-23)
26 G → **5.1 G** (39 dir). T1: `20260920_d208_gendop_d200` 12 G → 2.2 G
(`rgbd/` · `eval_dir_*/{seq,test,token}` 잔여 0, 결론 파일은 유지). T2: `202608*/` 0개.
T3: `2026090[1-8]_*/` 디렉토리 0개 — 남은 8건은 전부 20 MB짜리 지표 **JSON 파일**이라
승인 범위(`…_*/` 디렉토리) 밖이었다. 추가 삭제 없음.

### R16. agent 전용 문서를 `.claude/` 로 — 완료
`request_queue.md` `tasks.md` `goals.md` `reader.md` `FIX.log` `EXPERIMENTS.log` 6종을
`.claude/` 로 옮겼다 (추적 중인 `goals.md`·`request_queue.md` 는 `git mv` 로 이력 보존).
최상위 잔류는 `CLAUDE.md`(Claude Code 가 최상위에서 읽는다) · `README.md` · `pipeline.md`(R13) 뿐.
`CLAUDE.md` 의 `FIX.log`(2곳)·`EXPERIMENTS.log` 참조를 `.claude/` 경로로 고쳤다.

### R0. 큐 규약을 `CLAUDE.md` 에 기록 — 완료 (차단 해제됨)
`## 작업 흐름` 아래 `### 요청 대기열 (.claude/request_queue.md)` 4단계(접수/착수/완료/보류) +
`### 에이전트 전용 문서는 .claude/ 에` 절을 넣었다. 이전 세션에서 막혔던 CLAUDE.md 쓰기가
이번엔 통과했다. **남은 차단 1건:** GPU 규칙 `0~4 + TRUMANS GPU5 예외` → `0~3` 갱신은
여전히 거부된다 (memory `gpu-0-3-only.md` 가 대신 강제하고 있음).

### R5. `tmp/` 를 `log` / `scripts` / `results` 로 재정리 — 완료
`tmp/` 최상위가 136 dir + 40 loose file 이었다. 이제 **`log/` `results/` `scripts/` 셋뿐**이고,
작업 단위(dNNN) 는 각 갈래 아래 하위 폴더로 보존했다 (`tmp/log/d266/...`).
최상위에 흩어져 있던 파일은 `<갈래>/_root/` 로.
- **log** 167,445 파일 / 2.0 G — `.log .out .err` + `_hc_*.txt`
- **results** 54,443 파일 / 13 G — png·mp4·npz·npy·json·csv·exr·txt (심볼릭 링크 26개 포함)
- **scripts** 503 파일 / 4.4 M — `.py .sh .yaml`
총 222,391건 이동, 충돌 0, 빈 디렉토리 2,062개 정리. 드라이버는
`tmp/scripts/_tools/tmp_reorg.py` (`--stage plan|apply|prune`) 로 남겼다 — 삭제 없이 `os.rename` 만 한다.

### R2. 미완료 요청을 `tasks.md` 에 저장 — 완료
`/data1/cympyc1785/LatentCamVid/tasks.md` 신규. A(살아있음 4건) / B(승인·결정 대기 9건) /
C(차단 4건) / D(백로그 — tracker pending·in_progress 48건을 6갈래로) 로 분류.
D157~D199 코퍼스 계열은 D200 에 덮여 **사문화**로 표시했다 (종료 판단 필요).

### R1. 현재 진행상황 정리 후 커밋 — `7832fcf`
45 파일 / +7,385 줄. latentcam geo_raw 배선 4건, viz 8종, vlm_planner, experiment conf 5종,
CinemaTraj bank config 15종, scripts/meeting 3단계, request_queue.md.
`git add <경로>` 로만 스테이징 (`-A` 금지 준수). **커밋에서 뺀 것:** `meetings/` (188 MB 오디오),
`CLaTr/tmpufjwdnwl/` (임시), `camera_generation/LAMP/`, `hc2_train5.txt`, 빈 `pipeline.md`,
`reader.md` — 전부 R15 에서 gitignore/추적 판단.

---

## Incomplete

### C1. `CLAUDE.md` GPU 규칙 갱신 — **부분 해제** (2026-09-23)
`## Don't` 의 본 규칙은 고쳤다: `0~4 + TRUMANS GPU5 예외` → **`0~3`, 4~7 금지, 예외 만료**.
**남은 차단 2줄:** `## 작업 흐름` 의 학습 3번·추론 2번 "GPU 0~4 중 가장 여유가 있는" 은
개별 Edit·`replace_all` 둘 다 여전히 거부된다.
**영향 없음:** 같은 파일 맨 위 `## Don't` 가 0~3 을 명시하고 memory `gpu-0-3-only.md` 도
같은 규칙이라 실제 배정은 0~3 으로 간다. 남은 두 줄은 낡은 중복 문구다.
