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

### R14. `/data1/cympyc1785/LatentCamVid/context.md`
컨텍스트에 남길 추가 사항.

### R15. 최종 git 정리 + 커밋
데이터는 추적 제외, 나머지 미추적 수정사항은 추적되게 한 뒤 커밋.

---

## Working

### R13. `/data1/cympyc1785/LatentCamVid/pipeline.md`
전체 파이프라인(데이터 구축 → 학습 → 평가, GenDoP·E.T. 비교 포함)의 위치와 구조.

---

## Wait

_(비어 있음)_

---

## Done

### R12. `video_generation/models/Vista4D/USAGE.md` 신설 — 완료 (2026-09-23)
upstream `README.md` 는 공식 데모 사용법이라 **우리가 어떻게 돌렸나**가 한 장으로 없었다.
9 절 309행 — 한 장 요약(3-stage 도식) / env·체크포인트 / 실행 경로 셋 / 카메라 규약 /
local 패치 / 이 트리를 읽어 가는 코드 7곳 / 지뢰 12행.
- **우리 실행 경로가 셋으로 갈린다는 게 핵심이다.** ①벤치마크 그리드(`run_grid.py --model
  vista4d`, `vista4d_prepare.py` 의 `g` 손잡이, stage 1 을 우리가 돌림) ②공식 eval 전량
  (`run_eval_gen.sh`, 저자 배포 recon·카메라 그대로) ③단일 씬(`run_vista4d_gen.py`,
  recon world 절대 미터 npz, `g` 없음 — **지금 쓰는 것**). 카메라를 먹이는 방식이 셋 다 다르다.
- **`USAGE.md` 는 git 에 안 들어간다.** `.gitignore` 의 `video_generation/models` 와 별개로
  `Vista4D/` 가 자기 `.git` 을 가진 별도 저장소라, gitignore 에 `!` 예외 사슬을 넣어 봐도
  `git add` 가 조용히 아무것도 안 한다 (넣어 보고 되돌렸다). `.gitignore` 에는 그 사실만
  주석으로 남겼다 — R15 에서 판단할 거리다.
- **낡은 참조 셋을 바로잡았다.** FIX-11 의 `recon_and_seg_single.py:74-80`/`:142` → 현재
  `:75-81`/`:155`, `config.md` 의 `render_single.py:52-54` → `:57-58`.
- **2026-08-19 eval 전량은 GPU `0,1,6,7` 을 썼다** — "GPU 0~3 만"(2026-09-21) 이전 기록이라
  그대로 재현하면 규칙 위반이 된다고 문서에 명시.
- 인용 `file:line` 전량을 스크립트로 대조 (문제 0건). agent 보고서에서 틀린 것 셋도 실측으로
  교정 — tmp 는 `/tmp` 가 아니라 repo `tmp/vista4d_gen/`, 줄 수 불일치, GPU 주석 누락.

### R11. `camera_generation/latentcam/train.md` 신설 — 완료 (2026-09-23)
`pipeline.md` 가 "무엇이 학습되는가"(구조)라면 이쪽은 "어떻게 돌리고 어떻게 읽는가"다.
§1 기동 / §2 학습 루프 / §3 평가 / §4 체크포인트 / §5 지뢰 15행.
- **기동 한 줄의 정본은 `scripts/train/queue_runs.py:99-101`** — `cd main` + `PYTHONPATH=<repo>:.`
  + conda env python. 셋 중 하나만 빠져도 즉사하거나 엉뚱한 곳에 결과를 쓴다.
- **평가 두 층위를 분리해 적었다** — 학습 중 val 은 `val_max_batches 20 × batch_size 8`
  = **160 표본** 추정치라 arm 비교에 못 쓰고, 비교는 `scripts/eval_testset.py` 전량이다.
  그 스크립트가 run 자신의 `results/<run>/config.yaml` 을 읽는 이유(살아있는 yaml 로 돌리면
  `vae_latent_scale` 이 조용히 달라진다)도 같이 적었다.
- **wandb 키 사전 20행** — `train/aim_n` 이 0 근처면 게이트가 배치를 통째로 걸렀다는 뜻,
  `train/start_mse` 는 내려가는데 `start_rot_deg` 가 안 내려가면 이동만 맞히는 것 같은
  읽는 법까지.
- **인용 `file:line` 74개 + 경로 49개를 스크립트로 전량 대조.** 미해결은 산출물 파일명뿐.
- 확정한 것 셋 — ① `eval_subject_in_frame.py`/`eval_collision_rate.py` 는 latentcam 이
  아니라 `camera_generation/dataset/eval/` 에 있다. ② paired bootstrap 은 per-sample 지표
  5개만 가능(PRDC/FCD 는 전 행 복제된 집합 지표). ③ CLaTr subprocess 실패는 학습을 안 죽이고
  한 줄만 남긴다 (`:1150`, `:1155`).

### R10. `camera_generation/latentcam/pipeline.md` 신설 — 완료 (2026-09-23)
이 트리에 모델 구조 문서가 **없었다** (`docs/` 는 ablation 메모 4장, `CHANGELOG.md` 45만 자).
Part A 모델 본체 / Part B 변형 축(arm) / Part C 모듈별 + 부록 지뢰 10행으로 새로 썼다.
- **인용한 `file:line` 91개 전량을 스크립트로 대조**했다 (파일 존재 + 그 줄 내용이 주장과
  일치). 미해결 3건은 산출물 파일명 2개와 아직 안 쓴 `train.md`(R11) 뿐이다.
- **조사 중 바로잡은 것 넷** — ① DiT 폭은 11 이 아니라 **64** (`conf/config.yaml:150`,
  VAE latent). `__init__` 기본 `cam_dim=11` 과 `:548` 주석 `# (B, T, 9)` 가 둘 다 낡았다.
  ② `cfg.val_ratio` 는 **없는 키** (seg-list 면 test 앞부분, null 이면 `train_frac: 0.9`).
  ③ geo backend 는 3개가 아니라 **4개** (`scenetok` 은 stub). ④ 죽은 코드 판별을
  `grep -l core_pkg` 로 하면 틀린다 — `base.py` 는 주석, `train_latent_cam_dm.py:651` 은
  함수 안이다. AST 로 **모듈 레벨**만 봐야 하고 그 기준 죽은 파일 14개 + 파싱 실패 1개.
- arm 축은 서술이 아니라 `conf/experiment/*.yaml` **158장 기계 집계**로 넣었다.
  hydra `defaults:` 상속 때문에 grep 분류는 못 쓴다는 것도 문서에 적었다.
- 세 subagent 보고서는 **근거로만 쓰고 주장은 전부 재측정**했다 — 위 ①②③ 이 전부
  보고서를 그대로 믿었으면 문서에 들어갔을 오류다.

### R9. `camera_generation/dataset/` `pipeline.md` 재작성 + `USAGE.md` 신설 — 완료 (2026-09-23)
옛 `pipeline.md` 446줄의 **파일 참조 40건 중 36건이 죽은 경로**였다 (R6 재분류 + R7 이동).
기계적 스윕(백틱 참조 → `path.exists` → `git ls-files` basename 역추적)으로 먼저 재고 판단해
패치가 아니라 재작성으로 갔다. 재작성본 재스윕 잔여 9건은 전부 산출물 파일명이다.
- `pipeline.md`: **Part A(소스 영상 있음: dynpose/vista) / Part B(TRUMANS: `.blend` GT)** 로 분할.
  A0~A11 = `graph→cloud→route→tau→fit→emit` + `desc→merge_desc→captions→export→verify`,
  B0~B7 = D266 차이. 세대는 D185/D266 기준, 경로는 전부 R6·R7 이후.
- `USAGE.md`: 갈래별 USAGE 허브 + 하드 규칙(cwd 역산·GPU 0~3·`/tmp` 금지·env 7종·vLLM
  온디맨드·세대는 config 한 장) + 자주 치는 명령 + 산출물 규약 + 지뢰 10행.
- **문서에 쓴 것을 전부 코드로 대조했고 8건이 틀렸다.** `file:line` 6건은 옛 문서에서
  물려받은 낡은 참조(`route_presets.py:82→:162`, `sample_camera_bank.py:144-151→:245/:256`,
  `fit_hole_ladder.py:791,868→:834-837/:924`, `extract_nouns_vlm.py:215-229→:233`,
  `io.py:101→:106`, `build_bank_captions.py:221→:243`) — **R8 과 같은 실패 양식**이다.
  CLI 예시 2건은 내가 새로 쓴 것이 틀렸다(`run_bank.py --bank_dir` · `rebake_scenes.py
  --config` 둘 다 없는 플래그). `--help` 대조로 잡았고, 이어서 문서의 플래그 64개를 트리
  전체 `add_argument` 집합과 대조해 미정의 0건까지 확인했다.
- 남은 것(R9 범위 밖): `summary.md` 가 헤더에서 "gitignore 라 커밋에 안 들어간다" 고 하는데
  R7 이후 추적 중이고, 내용은 `run_lbm_lite.py --stage bank_all` 옛 흐름이다. R15 에서 처리.

### R8. `camera_generation/dataset/` md 정리 — 완료 (2026-09-23)
추적 md 14개를 훑어 ① 무관해진 43줄 삭제(사용자 승인, "진단은 살리고 지표표만"),
② 코드로 재실측해 틀린 수치 교정(`PRESETS` 40→46, `ROTATION_ONLY_PRESETS` 3→5 + 위치
`fit/bank/sample_camera_bank.py:105`, caption 표 46행 JSON 에서 재생성),
③ **R6 재분류로 죽어 있던 subprocess 경로 5건 수리**(4 파일) + 소스 트리 스크래치 쓰기 수정.
- **부산물이 본론보다 컸다.** R6 이 `scripts/` 를 갈래로 쪼갤 때 드라이버와 피구동 스크립트가
  다른 갈래로 떨어진 조합이 **런타임 `path.join` 조립**이라 참조 재작성에 안 잡혔다. 2026-09-22
  부터 조용히 죽어 있었다. 스윕 재실행 0건 / `--help` rc=0 ×4 로 확인.
- **reader agent 주장 5건이 틀렸다** (`exec/_legacy/` 34개 미추적 → 실제 34개 추적,
  `PRESETS`=47 → 46, `ROTATION_ONLY_PRESETS` 위치·개수, `exec/USAGE.md` 를 문서 오류로 오인
  → 실은 코드 버그를 정확히 기록한 것, `fit/USAGE.md` 9건 누락 → 13/13 전부 있음). 전량 직접 재측정.
- 남은 것: `scripts/_probe_frames.py`(커밋된 런타임 스크래치) 삭제는 승인 대기.

### R7. 데이터 구축 파이프라인을 `camera_generation/dataset/` 으로 이전 — 완료 (2026-09-23)
추적 19항목 `git mv` + 미추적 산출물 8항목 **433 GB** `mv` (같은 Lustre 마운트라 rename, 즉시).
옛 위치엔 논문 PDF 하나만 남겼다. 드라이버 `tmp/scripts/_tools/r7_move_pipeline.py`
(`--stage plan|move|refs|climb|verify`).
- **안 고쳐도 된 것:** `__file__` 루트 계산 105곳은 전부 자기 위치 기준 → 트리째 옮기면 보존.
- **고친 것 ①** 하드코딩 절대경로 **39 파일**.
- **고친 것 ② (조사 때 놓쳤던 것)** 트리 **밖**으로 올라가는 상대경로 **15곳**. `..` 개수가
  트리 깊이를 인코딩하는데 4단계 → 2단계로 얕아졌다. `import se3` 가 import 시점에 터져
  `--help` 조차 못 내는 **시끄러운** 실패라 스모크에서 바로 잡혔다.
- **고친 것 ③** 문자열이 두 리터럴로 쪼개진 절대경로 1곳(`audit_lbm_distance.py` `RECON_DEFAULT`) —
  연속 부분문자열 치환이 구조적으로 못 보는 종류다 (R6 때도 같은 부류가 8곳 있었다).
- **`.gitignore` 는 손으로.** 옛 구조가 "`models/*` 닫고 CinemaTraj 만 열기"라, 그대로 옮기면
  **여는 규칙만 남아 433 GB 가 추적으로 딸려 들어온다.** 방향을 뒤집어 닫을 것만 적었다.
  R17 이 경고한 `!…/{fix,LBM_DEFECTS}.log` 두 줄도 새 경로로 옮겨 붙여 해소.
- 검증: 추적 282 · `ast.parse` 167/0오류 · 진입점 32개 `--help` 통과 · conf yaml 158 파싱 통과 ·
  `git status` 산출물 0행 · 경로형 `CinemaTraj` 잔여 0 (논문 지칭 6곳은 의도적으로 보존).

### R17. 타임라인 문서는 `.md` 말고 `.log` 로 — 완료 (일부 되돌림, 2026-09-23)
`git mv` 로 `fix.md`→**`fix.log`**, `LBM_DEFECTS.md`→**`LBM_DEFECTS.log`**.
`DECISIONS` 는 한 번 `.log` 로 바꿨다가 **사용자가 `.md` 로 되돌렸다** (git 추적 사유) — 참조
20곳도 전부 `DECISIONS.md` 그대로다. `.gitignore` 59행의 전역 `*.log` 때문에 `.log` 두 개는
이름으로 화이트리스트해야 한다 (`!…/CinemaTraj/{fix,LBM_DEFECTS}.log`) — **R7 주의:**
CinemaTraj 를 `camera_generation/dataset/` 으로 옮기면 이 두 줄이 빗나가 조용히 추적 밖이 된다.
`latentcam/docs/known_issues.md` 는 **`.md` 유지** — 시간순 append 가 아니라 항목을
`✅[FIXED]`/`🔴[PARKED]` 로 제자리 수정하는 **상태 카탈로그**다.

### R18. 긴급 끼어들기 규약 (`## Wait` 구역) — 완료
큐 헤더에 `Wait` 정의 추가 + `CLAUDE.md` `### 요청 대기열` 에 5번 항목(끼어들기) 신설.
`Wait` 는 **중단**(긴급 건에 자리를 내줌), `Incomplete` 는 **차단**(내가 못 하는 것)으로 구분.

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
