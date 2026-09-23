# 원본 LBM 을 TRUMANS / Vista4D 에 돌리면서 실측한 결함

**여기는 `DECISIONS.md` 와 다른 문서다.** `DECISIONS.md` 의 `D1..D70` 은 **LBM-Lite 를 짜면서
내가 고른 것들**이고, 이 문서의 `LO1..LO11` 은 **원본 LBM(`models/Planner/Look-Before-Move/`)
자체의 결함**이다. 번호가 겹쳐 헷갈리지 않게 접두사를 `LO`(LBM Original) 로 둔다.

전부 실행 중 stdout / 산출 JSON 에서 **실측**한 것이고, 추측은 없다. 각 항목의 "상태" 는
2026-08-25 시점.

대상 실행:
- TRUMANS `00add26c-7a26-4a61-b192-b97aa493b3f3` — rule arm(`Actions/<seq>.txt`) 16창,
  narrative arm(VLM action tagging → narrative input) 16창
- Vista4D `camel` / `avocado-slice` — 점군 → `.blend` 어댑터

**정정 (2026-08-25)**: 앞서 이 줄에 "LBM 코드 0줄 수정"이라고 적었는데 **틀렸다**. 원본 리포는
4개 파일 151+/9− 수정된 상태다 (`git diff --stat` 실측, 아래 표). 전부 env 로 분기해서
**env 를 안 주면 스톡과 동일하게 동작**하지만, "0줄"은 사실이 아니다.

| 파일 | 줄 | 무엇 | 게이트 |
|---|---|---|---|
| `Cinematographer/cinematographer_quality_worker.py` | +30 | `resolve_scene` 단일 씬 폴백(:167-179) / `render_preview` 가 `file_format="PNG"` 강제(:2939-2955, finally 복원 :2977) | 폴백은 `LBM_SINGLE_SCENE_FALLBACK=1`. **PNG 강제는 무조건** — 유일한 무조건 변경 |
| `Cinematographer/cinematographer_stage.py` | +75 −9 | `_env_float`/`TRAJECTORY_MOTION_SCALE`/`TRAJECTORY_LIMIT_SCALE`(:31-60), crane↔pedestal 분리(:212), `preset_for_motion` 확장(:842-846), `trajectory_max_speed()` 신설(:883-892), `trajectory_motion_limit` 배율(:900-902), `derived_motion_end_transform` 상수 4개에 배율(:1388-1416) | `STORYBLENDER_TRAJECTORY_SCALE`(기본 1.0), `STORYBLENDER_MOVEMENT_VOCAB=extended` |
| `Director/director_stage.py` | +29 | `EXTENDED_MOVEMENT_VOCAB`(:973-980), `infer_movement_intent` 에 orbit/crane/pedestal/dolly 분기(:983-990), `infer_direction_label` 에 right/down 분기(:1022-1029) | `STORYBLENDER_MOVEMENT_VOCAB=extended` |
| `VideoEngineer/blender_render_worker.py` | +26 | `--frame-step` 인자(:46), `_resolve_scene` 단일 씬 폴백(:75-83), `_snapshot/_restore_scene` 가 `frame_step` 을 챙김(:795, :816-819), `_render_camera` 가 `.blend` 의 `frame_step` 을 덮어씀(:1144-1152) | 폴백은 `LBM_SINGLE_SCENE_FALLBACK=1`. `--frame-step` 기본 1 = 스톡 |

`render_preview` 의 PNG 강제만 env 게이트가 없다. `.blend` 출력 포맷이 FFMPEG 이면
`write_still=True` 가 전부 실패해 채널 board 가 0장이 되고 `llm_filter_channel_board` 가
`no_rendered_previews` 로 빠져 **VLM 선정과 micro-adjust 루프가 통째로 건너뛰어진다**
(기하 점수 1위로 조용히 폴백). 나머지 렌더 진입점 3곳(`director_scene_context_builder.py:588`,
`cinematographer_preview_worker.py:330`, `blender_render_worker.py:1154`)은 이미 하고 있고
여기만 빠져 있었다 — 즉 이건 스톡 LBM 의 **9번째 결함(LO9)**에 가깝다.

---

| # | 결함 | 상태 |
|---|---|---|
| LO1 | shot frame 창 유실 | 해결 (어댑터) |
| LO2 | focus 가 사람이 아니라 소품 | 해결 (어댑터) |
| LO3 | `semantic_target="feet"` | 해결 (어댑터) |
| LO4 | `movement_term_target` 유실 | 해결 (어댑터) |
| LO5 | `--camera-quality fast` 는 벽을 찍는다 | 해결 (runner 기본값 변경) |
| LO6 | 창 19프레임 vs 렌더 110프레임 | **그대로 둠** (사용자 지시) |
| LO7 | Cinematographer 가 `cameras: []` 를 `success: true` 로 내보낸다 | **미해결** |
| LO8 | `run_full_pipeline.py` 가 `--run-id` 를 무시하고 최신 폴더를 집는다 | **미해결** |
| LO9 | `render_preview` 가 출력 포맷을 안 덮어써 채널 board 가 통째로 비고, VLM 루프가 조용히 건너뛰어진다 | 해결 (원본 수정, env 게이트 없음) |
| LO10 | `_snapshot/_restore_scene` 가 `frame_step` 을 안 챙겨 `.blend` 의 `frame_step=2` 가 새어 들어오고 `clip.mp4` 가 1프레임이 된다 | 해결 (원본 수정, `--frame-step` 기본 1) |
| LO11 | Blender 번들 python 에 PIL 이 없어 채널 board PNG 가 안 만들어지고, **Phase-1 VLM 필터가 전량 FAIL** 한다 | **미해결** (진단만) |

---

## LO11 — Phase-1 VLM 채널 필터가 한 번도 돈 적이 없다  **[미해결]**

`cinematographer_quality_worker.py:2983 build_board` 첫 줄이 `if Image is None or not
candidates: return ""` 인데, 그 `Image` 는 `:23-26` 의 `try: from PIL import Image / except:
Image = None` 이다. 이 워커는 **Blender 번들 python** 에서 돌고 거기엔 PIL 이 없다:

```
$ /data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/4.5/python/bin/python3.11 \
      -c "from PIL import Image"
ModuleNotFoundError: No module named 'PIL'
```

그래서 `channel_boards[<ch>]["board_path"] = ""` 가 되고, `cinematographer_stage.py:3716`
의 `if not board_path or not Path(board_path).exists()` 가 걸려 Phase-1 이
`{"success": False, "error": "board_image_missing", "survivors": rendered}` 를 돌려준다.

**조용한 이유 두 가지.**
1. `:4970` 은 `status = "OK" if ch_result.get("success") else "FAIL"` 만 찍고 `error` 는
   안 찍는다. 로그에 남는 건 `Phase-1 .../direction: FAIL` 뿐이다.
2. 실패 반환값이 `survivors = rendered` 라 **후보가 하나도 안 걸러진 채** Phase-2 로 그대로
   넘어가고, Phase-2 는 자기 board 를 `cinematographer_stage.py` (= PIL 있는 `lbm` env)
   에서 `_build_board_from_previews` 로 만들기 때문에 정상 동작한다. 파이프라인 최종
   결과는 `success: true`.

**전량 실패 실측** — Phase-1 이 OK 로 찍힌 적이 한 번도 없다:

| 실행 | FAIL | OK |
|---|---|---|
| TRUMANS narrative 16창 (`/tmp/lbm_narrative_run.log`) | 30 | 0 |
| Vista camel (`/tmp/lbm_vista_camel.log`) | 3 | 0 |
| Vista avocado-slice (`/tmp/lbm_vista_avocado.log`) | 3 | 0 |

디스크에서도 확인된다 — `output/vista_camel/quality_candidates/scene_1_shot_1/
scene_1_shot_1_cam1/` 에 `previews/*.png` 는 80장인데 `*_direction_board.png` /
`*_preset_board.png` / `*_merged_board.png` 는 **없고**, Phase-2 가 만든
`*_llm_merged_board_iter_{1,2,3}.png` 만 있다.

**잃는 것.** `llm_filter_channel_board` 의 시스템 프롬프트(`:3735-3752`)가 담당하던
**facing rule** — "뒤통수/목덜미만 보이고 얼굴이 안 보이면 back-facing 으로 보고, shot
contract 가 명시적으로 뒤/OTS 를 요구하지 않는 한 전량 기각" — 이 통째로 죽는다. 구도·가림·
story intent 불일치 기각도 마찬가지다. `semantic` 채널은 후보가 애초에 0개라
`no_rendered_previews` 로 갈린다(다른 원인).

**고치는 법 (아직 안 함).** ① Blender 번들 python 에 PIL 설치, 또는 ② `build_board` 를
PIL 없이도 되게 (numpy + Blender 내장 이미지 저장) 만들기, 또는 ③ 채널 board 를 Phase-2 처럼
`cinematographer_stage.py` 쪽에서 preview 경로만 받아 만들기(=`_build_board_from_previews`
재사용). ③ 이 워커를 안 건드려서 제일 가볍다. 어느 쪽이든 `:4970` 에 `error` 를 같이 찍는
한 줄은 무조건 넣어야 한다 — 이게 없어서 두 데이터셋 전량이 조용히 넘어갔다.

---

## LO1 — shot frame 창이 Blender 씬에 안 전달된다

TRUMANS 시퀀스는 통짜 1400+ 프레임인데 우리는 action 창(예: `f0051..0069`)만 찍어야 한다.
LBM 은 shot 의 frame range 를 story JSON 에서 받는데, 그 경로가 `.blend` 의 재생 구간까지
가지 않아 **매번 프레임 0 부터** 렌더했다.

- **해결** — `trumans_frame_shift_startup.py` (Blender startup 스크립트). `TRUMANS_FRAME_OFFSET`
  환경변수로 씬 전체를 시간축으로 민다. 검증: Director/Cinematographer/VideoEngineer 세 stage
  stdout 에 `[trumans_frame_shift] offset -50 applied to 9 actions` 가 찍힌다.
- **부작용 없음** — `trumans_frame_shift_startup.py:78-90` 의 `on_load` 가 offset 이 0 으로
  파싱되면 곧장 return 한다. 그래서 같은 `BLENDER_USER_SCRIPTS` 디렉토리를 Vista 실행에
  그대로 써도 안전하다.

## LO2 — focus 가 사람이 아니라 소품으로 잡힌다

TRUMANS 는 사람 하나 + 소품 다수인데, Director 가 뽑은 `primary_focus_id` 가
**book_right_01 ×8 / book_left_01 ×5 / oven_base_01 ×2** 였고 사람(`zzy3`)이 primary 인 창은
**15개 중 0개**였다. "사람 액션 영상"인데 카메라가 책을 잡는다.

- **해결** — 어댑터가 `primary_focus_id: "zzy3"` 를 명시. offline 재실행에서 16/16 창이
  `['zzy3']`.

## LO3 — `semantic_target="feet"`

`primary_semantic_target` 기본이 `feet` 라 사람의 발을 잡는다. 상체 액션(펜 집기, 책 넘기기)에서
프레임 밖으로 나간다.

- **해결** — 어댑터가 `primary_semantic_target: "full_body"`.

## LO4 — `movement_term_target` 이 도중에 사라진다

Director 가 고른 movement term 이 Cinematographer 의 `motion_contract` 로 전달되지 않아,
"orbit right" 를 지시해도 실제 카메라는 무관한 움직임이 나왔다.

- **해결** — 어댑터가 계약을 명시적으로 채운다. 검증: `motion_contract.movement=orbit,
  direction=right` ↔ 최종 카메라 `orbit_right_arc` 일치.

## LO5 — `--camera-quality fast` 는 벽을 찍는다

Cinematographer 의 `fast` 모드는 후보 카메라 스코어링을 줄이는데, 그 결과로 **피사체 반대편
벽면**을 향한 카메라가 살아남는 창이 여럿 나왔다. `quality` 로 돌리면 같은 창에서 사람을 잡는다.

- **해결** — 우리 runner 기본을 `--camera-quality quality` 로 바꿈. 비용: Cinematographer
  stage 가 창당 60 s (w01 실측). 전체 파이프라인은 창당 ~2.5 분
  (Director 49 s / Cinematographer 60 s / VideoEngineer 37 s / Editor 2 s).

## LO6 — 창은 19프레임인데 렌더는 110프레임

`nw01` 은 action 창이 19프레임(`f0051..0069`)인데 LBM 이 실제로 렌더한 건 110프레임이다.
LBM 이 shot 길이를 자기 shot contract 의 duration 으로 다시 정하고 창 길이를 안 본다.

- **상태: 그대로 둔다.** 사용자 결정 — "긴거 하나 돌리고 그대로도 돌려줘". 하류(`poses_*.npz`)
  에서 `np.rint(np.linspace(0, count-1, 49))` 로 **보간 없이 index 만** 솎아 49프레임으로 맞춘다.
  고정 step `arange` 는 꼬리를 잘라먹는다(w01 에서 0.323 m → 0.284 m, 88%).

## LO7 — Cinematographer 가 빈 카메라 목록을 `success: true` 로 내보낸다  **[미해결]**

가장 나쁜 결함이다. 조용히 실패하고 **한 stage 뒤에서 엉뚱한 메시지로 터진다**.

관측된 시퀀스:

```
Cinematographer stage JSON :  success: true
                              cameras: []
                              downstream_blocked_camera_names: [...]
                              candidate_count_raw_min 1149  →  retained 0
VideoEngineer               :  render_count: 0,  stderr 파일은 비어 있음
                            :  RuntimeError("Blender scene render failed. See <빈 로그>")
                               ← video_stage.py:558
```

즉 **후보 1149개가 필터에서 전멸했는데 stage 는 성공으로 보고**하고, 사용자에게는 한 단계 뒤에
"Blender 렌더가 실패했다"는 **틀린 원인**이 표시된다. 실제로는 Blender 는 멀쩡했고 찍을 카메라가
0개였을 뿐이라 stderr 이 비어 있다.

- **고치려면**: Cinematographer 가 `retained == 0` 이면 `success: false` + 필터별 탈락 수를
  내야 한다. `downstream_blocked_camera_names` 가 비어 있지 않은데 `cameras` 가 비면 그 자체로
  모순이므로 stage 종료 시 assert 를 걸 자리다.
- **우리 우회**: runner 가 stage JSON 의 `cameras` 길이를 직접 세서 0이면 그 창을 실패로 찍고
  다음 창으로 넘어간다. 원본은 안 고쳤다(어댑터만 고친다는 결정).

## LO8 — `--run-id` 가 무시된다  **[미해결]**

`run_full_pipeline.py:317` 과 `:355` 가 handoff 경로를 `latest_stage_file(<STAGE>_DIR/"output",
...)` 로 찾는다. **디렉토리에서 가장 최근에 수정된 것**을 집을 뿐 `--run-id` 를 안 본다.

```python
# run_full_pipeline.py:317
camera_handoff_path = latest_stage_file(CINEMATOGRAPHER_DIR / "output", "outputs/camera_handoff_v1.json")
# run_full_pipeline.py:355
video_handoff_path  = latest_stage_file(VIDEO_DIR / "output", "outputs/video_handoff_v1.json")
```

`--resume-from` 을 쓸 때만 타는 경로지만, **여러 창을 병렬로 돌리면 창 A 의 카메라가 창 B 의
렌더로 들어간다.** 우리는 창을 순차로 돌려서 안 밟았다.

- **고치려면**: `--run-id` 로 `<STAGE>_DIR/output/<run_id>/outputs/...` 를 직접 구성하고,
  없으면 `latest_stage_file` 로 폴백하되 경고를 찍는다.

---

## 결함 아님 — 환경 문제 2건 (기록만)

- **ffmpeg 경로** — `VideoEngineer/video_stage.py:96` 이
  `shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"` 다. Windows 하드코딩 폴백이라
  리눅스에서 `which` 가 실패하면 존재하지 않는 경로로 간다. `PATH` 에
  `/data1/cympyc1785/tools/bin` 을 넣어 해결.
- **카메라 덤프 훅** — `lbm_camera_dump_startup.py` 는 `render_complete` 에 건다.
  `render_pre` 는 못 쓴다 — `bpy.ops.render.render(animation=True)` 가 depsgraph 사본에서
  평가해서 **모든 프레임이 같은 행렬**로 찍힌다. 훅이 **append 모드**라 재실행 전에 덤프
  디렉토리를 `rm -rf` 해야 한다.

---

## 스톡 LBM 대비 우리가 바꾼 실행 설정 (결함 아님, 감사용)

| 설정 | 값 | 되돌릴 수 있나 |
|---|---|---|
| `STORYBLENDER_MOVEMENT_VOCAB` | `extended` | 예 — 스톡은 축소 어휘 |
| `STORYBLENDER_TRAJECTORY_SCALE` | `5` | 예 — 궤적 크기 배수 |
| `LBM_SINGLE_SCENE_FALLBACK` | `1` | **아니오** — 우리 입력에 story 다중 scene 이 없어 구조적으로 필요 |
| `STORYBLENDER_TRAJECTORY_LIMIT_SCALE` | (미지정 → `TRAJECTORY_SCALE` 을 따라감) | 예 |
| `--camera-quality` | `quality` | 되돌리면 LO5 가 재발한다 |
| `--frame-step` | `1` (스톡 기본과 동일) | — |

env 를 **하나도 안 주면** 위 수정분 중 살아 있는 건 LO9(PNG 강제) 하나뿐이고, 나머지는 전부
스톡 경로로 떨어진다. 즉 "스톡 재현" 은 가능하되 **바이트 단위로 원본 코드는 아니다**.
