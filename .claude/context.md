# context.md — 다음 세션이 알아야 할 것

`CLAUDE.md` 는 **규칙**, `pipeline.md` 는 **지도**, `.claude/goals.md` 는 **목표**다.
여기는 그 셋 어디에도 안 들어가는데 **모르면 사고가 나는 것**을 적는다 — 서 있는 지시,
문서 사이의 어긋남, 비교하면 안 되는 지표, 반복해서 밟은 실패 양식.

| 물음 | 답 |
|---|---|
| `SPECS.md` 는 어디 있나 | **없다.** `CLAUDE.md` 의 `## Specifics` 와 `.claude/goals.md` 가 둘 다 이 파일을 가리키는데 저장소에 존재하지 않는다. 코드 구조는 `pipeline.md` + 하위 문서 여섯을 본다 |
| 세대 이름은 무엇을 가르나 | `dNNN` **하나로 데이터·모델을 다 나눈다.** 분리 요청이 있고 미적용 — §3 |
| 무엇을 먼저 의심해야 하나 | **문서 안의 `file:line`.** 코드가 옮겨져도 참조는 안 따라온다. R9 에서 40건 중 36건, R12 에서 3건이 죽어 있었다 — §5 |
| agent 보고서를 믿어도 되나 | **아니다.** 단서로만 쓰고 전량 재측정한다. R8·R10·R12·R13 에서 전부 틀린 주장이 나왔다 — §5 |

---

## 1. 서 있는 지시 — 사용자가 한 번 말하고 계속 유효한 것

`CLAUDE.md` 의 `## Don't` 에 들어간 것(GPU 0~3, 실험 결과 임의 요약 금지)은 여기 다시 안 적는다.
**규칙 파일에 없는데 계속 유효한 것만** 적는다.

| 지시 | 시점 | 내용 |
|---|---|---|
| 영상 생성 | — | "이제부터 내가 요청하는 것들 Vista4D 돌려서 영상 만들어줘." |
| 릴 단위 | — | 영상+preset 을 지정해 돌려 달라고 하면 **seed 3개**로 돌리고 depth warp 해서 `source \| GT \| seed3` **concat 한 편**까지가 한 단위 |
| 비교 영상 | — | 출력만 내지 말고 **depth warp 도 같이** (위=출력 / 아래=warp). 안 그러면 warp 탓인지 생성 탓인지 못 가른다 |
| track | — | object-centric preset 은 dynamic subject 가 일정 이상 움직이면 **track 도** 한다 |
| GenDoP | 2026-09-21 | "rmax 곱하면 안되고 raw로 normalized 되어서 나오는 원본 코드 그대로 나온걸 저장해서 써야해" → `--raw` 가 **기본값**이 됐다 (`camera_generation/dataset/exec/run_gendop_eval.py:496-497`) |
| GPU | 2026-09-21 | "그리고 이제 GPU 0~3만 써줘" — 이전 예외(D215 6,7 / TRUMANS 5) **전부 만료**. 4~7 에 띄운 건 죽이고 옮긴다 |
| pipeline 요청 | — | "따로 pipeline 만들어달라고 한건 **최소한의 기능만 남긴 inference code**" — 돌려 달라는 요청과 다르다 |
| d221 번들 | — | "hike같은 d221 bundles 돌리려는거 **더 안돌려도됨**" |
| kill | — | 대기·굽기 프로세스 정리는 **직접 kill** 해도 된다. 단 **학습은 Ctrl+C 먼저** (`screen -X stuff $'\003'`) — 바로 kill 하면 DataLoader worker 가 GPU 를 쥔 채 고아가 된다 |
| GPU 6,7 | 2026-09-24 | "이 렌더 실험들은 gpu 6,7 써도됨" — **R40 Blender 렌더 속도 실험에 한정**한 예외. 다른 작업은 여전히 0~3 — **만료 (2026-09-27 6,7 사용 금지)** |
| GPU 6,7 (SAM3) | 2026-09-24 | "SAM3 돌릴때만 gpu 6,7도 같이 써줘" — **R50 SAM3 필터 단계에 한정**. 끝나면 6,7 에서 내린다 — **만료 (2026-09-27 6,7 사용 금지)** |
| GPU 0,3,6,7 (확장) | 2026-09-25 → **만료 2026-09-27** | "gpu 6,7은 일단 내가 빼달라고 할 때까지 써줘" → 2026-09-27 "이제 gpu 6,7 쓰지 말아줘" 로 **해제**. GPU 6,7 사용 금지 — 0~3 만 쓴다 (6,7 에 있던 D284 정적 SAM3 샤드 2~5 는 GPU 3 으로 이어 붙임) |
| GPU 6,7 (9시간) | 2026-09-27 01:45 → **10:50 만료** | "gpu 6,7을 9시간 정도 쓰려는데" + "1로해줘" — ① D284 정적 SAM3 재분배, 끝나면 남은 시간은 d268 eval·R59 재개. **10:50 에 6,7 의 내 프로세스는 자동으로 내린다** (tmp/d284/gpu67_deadline.log) — **조기 만료 2026-09-27 07:35** ("거의 다 끝난 것 같은데 이제부터 0~3번만 써줘"). 6,7 사용 금지 |
| SAM3 명사 범위 | 2026-09-27 | "시간 별로 차이 안나는데 일단 그냥 그대로 다 돌려줘" — VLM dynamic/static 명사 **전부** segmentation 유지 (top-k 상한 안 둠). 근거 tmp/agent/reader-r61-seg-usage.md (G5 obb·source OBB floor·composition 이 전 노드를 읽음, top2/2 면 anchor 20.8% 누락) |
| 시작 격자 고도 정의 | 2026-09-27 | "고치지마. 어차피 지면 법선이나 gravity도 그렇게 신뢰성이 높은건 아니라서 따로 판단하긴해야할 것 같아" — start grid 고도는 **중력 기준 그대로** 두고 (국소 지면/비탈 법선 기준으로 바꾸지 않음), 땅속 후보는 판정 단계(G3 가림·국소 지면 게이트 등)에서 따로 거른다 |
| 필터 로고 | 2026-09-25 | "로고 워터마크는 그냥 빼자" — D282 필터에서 overlay(로고·워터마크·자막) 영상은 **탈락 유지** |
| DynPose bank | 2026-09-25 | "dynpose bank는 d200기준으로" — 확장분 bank 는 d200 과 같은 d185(grid5)+d199(frame0 anchor) 설정. TRUMANS 용 d273 pool 라우팅은 쓰지 않는다 |
| 요청 대기열 | — | "내가 따로 명령하지 않아도 큰 문제 없으면 **계속 다음 request 착수해**" — `## Done` 으로 옮기고 다음 건으로 바로 간다. 멈추는 건 승인이 필요한 것(파일 삭제·GPU 재배정·장시간 실행)뿐 |

---

## 2. 문서 사이의 어긋남 — 발견된 것

| 어긋남 | 상태 |
|---|---|
| `CLAUDE.md` · `.claude/goals.md` → `SPECS.md` | **파일이 없다.** 둘 다 가리키는데 저장소에 없다 |
| `camera_generation/dataset/summary.md` 헤더 | ~~고쳤다 (R15).~~ 옛 헤더는 "`.gitignore:222 camera_generation/models` 아래라 커밋에 안 들어간다" 였는데 **R7 이후 추적 중**이고 `.gitignore:222` 는 지금 `tools` 줄이다. 본문은 **2026-08-23 스냅샷 그대로 둔다** — `run_lbm_lite.py` 는 저장소에 없고 현재 드라이버는 `camera_generation/dataset/exec/run_bank.py`(`graph→cloud→route→tau→fit→emit`) 라는 경고를 헤더에 박았다 |
| `CLAUDE.md:183` (학습 3번) · `CLAUDE.md:190` (추론 2번) | "GPU 0~4 중 가장 여유가 있는" 이라고 적혀 있다. 같은 파일 `## Don't` 가 0~3 을 명시하므로 **`## Don't` 가 이긴다**. 두 줄 수정은 **두 번 막혔다** (R14·R15 모두 classifier denial) — 사용자가 직접 고쳐야 한다 |
| Vista4D 사용법 문서 | ~~해결했다 (R15).~~ `video_generation/models/Vista4D/USAGE.md` → **`video_generation/Vista4D_USAGE.md`** 로 옮겨 추적 대상이 됐다. `Vista4D/` 는 자기 `.git` 을 가진 별도 저장소라 그 안에 두면 `!` 예외로도 못 살린다. **local 패치 2건은 여전히 재클론하면 사라진다** |

---

## 3. 세대 이름 — `dNNN` 하나로 다 가른다

지금 `d121` `d185` `d200` `d266` 같은 번호 하나가 **데이터 세대와 모델 세대를 동시에** 가리킨다.
사용자가 분리를 요청했고 **아직 적용 안 됐다**:

> "d숫자로 모든걸 나누니까 헷갈려. 데이터 변경이면 `datagen_숫자` 모델 변경이면 `modelgen_숫자`
> 이렇게 분리 못함?" / "데이터도 trumans, dynpose, vista 이렇게 나눠서 버전 이름 나눠주면 좋을 것 같아."

**지금 당장 조심할 것 둘.**

- **`variant_id` 는 코퍼스 키가 아니다.** 재굽기해도 이름은 그대로고 knob·pose 만 바뀐다.
  같은 세대 이름으로 다시 구우면 캐시가 **miss 0 으로 통과하면서 전부 틀린다.**
- **`plan_tier == 0` 은 세대마다 뜻이 다르다.** 세대를 가로질러 세는 데 쓰면 안 된다.

---

## 4. 비교하면 안 되는 지표

| 지표 | 왜 |
|---|---|
| CLaTr 절대값 | **게이지가 학습 split 을 따라간다.** 코퍼스가 다르면 나란히 못 놓는다. 재분할하면 옛 ckpt 가 새 test 를 이미 본 상태라 지표가 조용히 부푼다 |
| caption fscore | **chance 대비로** 읽는다. val 구성이 `1씬×160변이` 냐 `160씬×1변이` 냐에 따라 절대값이 같은 눈금이 아니다 |
| PRDC | **코퍼스 간 비교 금지.** 반경 임계 양옆에 앉으면 격차가 증폭된다 |
| 학습 중 val | `val_max_batches`(20) × `batch_size`(8) = **160 표본** 추정치다. arm 비교는 `camera_generation/latentcam/scripts/eval_testset.py` 전량 + `last.pth` |
| 뱅크 렌더 열 | `subject_visible_frac` 은 `poses.npz` 가 md5 동일이어도 재실행마다 **±0.04** 흔들린다. 그 이하 차이는 신호가 아니다 |
| latentcam vs GenDoP caption 지표 | 태거·GT·pose 수·entry **네 축이 전부 다르다** |

---

## 5. 반복해서 밟은 실패 양식 — 이 둘이 대부분이다

### ① 참조는 코드를 따라 움직이지 않는다

코드를 옮기거나 함수를 넣으면 **문서·로그의 `file:line` 이 조용히 어긋난다.** 실측:

- R9 — 옛 `pipeline.md` 의 파일 참조 **40건 중 36건이 죽은 경로**였다 (R6 재분류 + R7 이동 탓).
- R12 — `FIX.log` 의 `recon_and_seg_single.py:74-80`/`:142` → 실제 `:75-81`/`:155`,
  Vista4D `config.md` 의 `render_single.py:52-54` → 실제 `:57-58`.
- R13 — 새로 쓴 문서에서도 상대경로 18건이 저장소 기준으로 안 풀렸다.

**대응: 문서를 쓰면 커밋 전에 `file:line` 전량을 스크립트로 검산한다.** 백틱 안의 경로를
뽑아 `os.path.exists` + 파일 줄 수 대조. 이건 매번 하는 것이고, 그래서 일회성 코드로 쓰고
파일로 남기지 않는다.

### ② agent 보고서는 근거로만 쓴다

subagent 보고서에서 **매번** 틀린 주장이 나왔다.

- R8 — 5건 (미추적 파일 수, `PRESETS` 개수, 상수 위치, 문서 오류 오인, 누락 오인).
- R10 — 4건 (DiT 폭, 없는 config 키, backend 개수, 죽은 코드 판별 방법).
- R12 — 3건 (tmp 위치를 `/tmp` 로, 같은 파일에 줄 수 두 개, GPU 규칙 변경 누락).
- R13 — 4건. 가장 큰 것: `--raw` 배선을 "미추적" 이라고 했는데 실제로는
  `camera_generation/dataset/exec/run_gendop_eval.py:496-497` 에 **명시돼 있고 기본값까지 뒤집혀 있었다.**

**대응: 보고서는 "어디를 볼지" 로만 쓰고, 문서에 넣을 주장은 전부 직접 읽어 재측정한다.**
보고서 경로는 main 이 절대경로로 주고(`tmp/agent/<role>-<대상>.md`), **반환 JSON 을 못 받으면
실패로 간주한다** — 보고서 파일을 열어 짐작하지 않는다.

### ③ rc=0 은 "됐다" 가 아니다

조용히 실패하는 것들이 모여 있다.

- `git add` 가 nested repo 안의 파일에 대해 **rc=0 인데 아무 일도 안 한다.**
- `eval_dir` 이름에 태그를 안 물리면 eval 은 skip 하고 render 는 **이전 preset 을 그린다.**
  둘 다 rc=0.
- 뱅크를 `tau` 부터 구우면서 `route` 를 빼면 `anchors.json` 이 무시되고 **기본 앵커가 돈다.** rc=0.
- `--videos all` 은 `metadata.csv` 기본값이 Vista4D 것이라 다른 코퍼스에서 **조용히 0편**이 된다.

**대응: "정상" 을 셋으로 정의한다 — ① 살아있나(pid + GPU 점유) ② 나아가나(진행 지표 증가)
③ 끝났나(종료 문자열·산출 파일).** `Traceback|OOM` 만 걸면 **죽는 것만** 잡고 **멈춘 것은
못 잡는다.** 죽었다고 판정하기 전에 pid 를 확인한다 — tqdm tail 은 하드킬과 구분이 안 된다.

---

## 6. env — 어느 단계가 어느 env 인가

`/data1/cympyc1785/miniconda3/envs/` 아래 31개가 있다. 파이프라인 본선에 쓰는 것만:

| env | 쓰는 곳 |
|---|---|
| `latentcam` | ② 학습, ③ 평가 — `camera_generation/latentcam/` 의 `main/train_latent_cam_dm.py` · `scripts/eval_testset.py` |
| `da3` | ① recon (Depth-Anything-3) |
| `sam3` | ① 인스턴스 분할 |
| `geocalib` | ① 중력 사이드카 |
| `vllm` | ① VLM 명사·지칭구 (Qwen3-VL-30B-A3B-Instruct @ 22002). **상시 기동 금지** — 단계 직전에 올리고 끝나면 내린다 |
| `GenDoP` | ③ 베이스라인 추론 |
| `vista4d` | ④ 영상 생성. flash-attn CXXABI 때문에 절대경로 python 을 쓸 때 `LD_LIBRARY_PATH=$ENV/lib` 가 필요하다 |
| `lbm` | TRUMANS 갈래 |

나머지(`alayaworld` `infcam` `sierpinskicam` `trajcrafter` `recammaster` `rerope`
`cameraanything` 등)는 **벤치마크 비교용**이고 본선 파이프라인이 아니다.

---

## 7. 지금 열려 있는 것

`.claude/` 가 정본이다 — 여기는 **어느 파일을 봐야 하는지**만 적는다.

| 무엇 | 어디 |
|---|---|
| 사용자 요청 대기열 | `.claude/request_queue.md` (Request / Working / Wait / Done / Incomplete) |
| 실험·작업 단위 (`dNNN`) | `.claude/tasks.md` |
| **돌고 있는 감시** | `.claude/watch.md` — **모니터를 걸면 여기 한 행을 추가하고 끝나면 지운다.** 행이 사라지는 게 "모니터를 껐다" 는 뜻이다. pid 를 적는 게 유일한 해법이다 (2026-09-23 실측: 감시자 221개가 살아 있었고 살아 있는 로그를 보는 것은 **0개**였다) |
| 고친 것 | `.claude/FIX.log`, 각 트리의 `fix.log` |
| 실험 기록 | `.claude/EXPERIMENTS.log` |
| 판단 근거 | `camera_generation/dataset/DECISIONS.md` (git 추적됨 — `.md` 인 이유가 이것이다) |

**승인 대기 중인 것** (전부 파일 삭제라 목록·근거를 보고하고 승인을 받아야 한다):

- `tmp/agent/reader-*.md` — R8·R10·R12·R13 조사 보고서. 남길 것은 `results/` 나
  `DECISIONS.md` 로 승격하고 나머지 정리.
- `camera_generation/dataset/scripts/_probe_frames.py` — 커밋된 런타임 스크래치.
