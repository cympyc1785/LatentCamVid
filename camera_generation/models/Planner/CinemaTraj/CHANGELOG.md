# Changelog (CinemaTraj)

All notable changes to the CinemaTraj sub-project (카메라 뱅크 굽기 · Blender GT 렌더 · 캡션).
Follows [Keep a Changelog](https://keepachangelog.com/).

이 파일은 2026-09-20 에 시작했다. 그 이전 이력은 `git log -- camera_generation/models/Planner/CinemaTraj` 가 원본이다.

## [Unreleased]

### Fixed
- **동적 점이 0개인 점군에서 depth warp 가 움직이는 물체를 49프레임 겹쳐 그리던 것
  (2026-09-21).** `--allow_no_seg` 경로(SAM3 미실행)는 `dynamic_mask` 가 전부 0 이라
  `static_mask = ~dynamic_mask` 로 **전 점이 static** 이 되고, static 점은 `visible` 이
  전 프레임 True 라 TP 렌더가 프레임마다 49개 사본을 같이 그렸다. my-clip 실측
  `points 44,994,231 / dynamic(visible.sum==1) 0` — 골퍼가 스윙 전 구간 유령 다발.
  - `scripts/render_bank_videos.py: render_variant(temporal_persistence=True)` 인자 추가 →
    `renderer.render` 로 전달. 기본 True 라 기존 호출부 6곳 전부 비트 동일.
  - `scripts/render_pred_depth_warp.py --temporal_persistence {auto,on,off}` (기본 `auto`).
    `auto` 는 `--allow_no_seg` 일 때만 NTP 로 떨어진다 — seg 가 있는 기존 릴은 TP 유지.
    `index.json` 에 `temporal_persistence` / `num_dynamic_points` 를 같이 적는다 (이 값이
    0 이면 그 릴은 믿으면 안 된다는 뜻이라 사후에 되짚을 수 있어야 한다).
  - 대가: 시간 누적을 포기하므로 구멍이 커진다. my-clip 실측 `hole_magenta_frac`
    s42 0.0685 → 0.1223 / s1234 0.0124 → 0.0672 / s2026 0.0 → 0.0073, GT(소스 pose) 0.0005.
  - 참고 실측 (수정 대상 아님): seg 를 돌린 점군은 정상이다 —
    `snowboard dynamic 6,478,846/43,812,771 (0.148)`, `golf 39,326,423/44,506,838 (0.884)`.
    golf 가 0.884 인 것은 SAM3 keyword `"golf"` 가 `box_xyxy [0,0,1279,717]` 로 **화면
    전체**를 잡았기 때문이다 (`seg_instances/golf/meta.json` id 5). 별건.

### Changed
- **`viser_cloud.py` 지면 격자가 기본 꺼짐 (2026-09-21).** 격자는 `half` 를 카메라와 노드를
  다 감싸게 잡으므로 씬 전체를 덮는데, 정작 보려는 카메라·OBB 위에 얹혀 화면을 가린다.
  중력축이 틀어졌는지 볼 때만 필요하다. **만들기와 보이기를 다른 손잡이로 갈랐다** —
  `--no_ground_grid` 는 아예 안 만드는 것이고(그러면 GUI 체크박스도 비활성이라 켤 수가 없다),
  `--ground_on` 은 만든 것을 처음부터 보이게 하는 것이다. 기본은 "만들되 숨김" 이라
  `view > ground grid` 로 언제든 켤 수 있다.
- **`scripts/run_custom_caption.py` 를 추론 전용 최소 경로로 축소 (2026-09-21).**
  사용자 지시 "sam3, vlm 이런거 안돌리고 순수히 우리 카메라 생성 모델만 돌리는거야 …
  따로 pipeline 만들어달라고 한건 최소한의 기능만 남긴 inference code였어".
  - 단계 8 → 7: `geocalib` 삭제, `graph`(`build_scene_graph.py`) → `scale` 로 교체.
    `scale` 은 `scene_graph.json` **최소본**(`scale`/`cameras` 블록만, `nodes: []`)을
    직접 쓴다. `format` 을 `planner_scene_graph_minimal_v1` 로 따로 찍어 `load_graph`
    가 진짜 그래프로 오인하지 못하게 했다 — 뱅크에 먹이면 anchor 0 개로 조용히
    빈 뱅크가 나온다.
  - recon 에서 SAM3 를 뺐다: `--seg_keywords` 를 **값 없이** + `--keep_recon_sky`.
    기본값이 `["_all_"]` 이라 빼먹으면 정반대로 전 픽셀이 dynamic 이 된다.
  - 캡션에서 지칭구를 정규식으로 뽑던 `guess_target`/`guess_nouns`/`--target`/`--nouns`
    제거. PE-AV 입력은 **`--molmo2_text` 로 따로** 받는다 (안 주면 캡션 원문).
    ③ umt5 캡션과 ④ PE-AV 문장은 서로 다른 인코더로 들어간다.
  - 알려진 한계: seg 가 없어 warp 점군이 한 덩어리 rigid cloud 라 **움직이는 물체는
    번진다**. 뱅크·릴 경로에는 쓰지 말 것.
  실측 (golf source, 49f, S 6.9976): hole_magenta_frac s42 0.0685 / s1234 0.0124 /
  s2026 0.0.
- **`scene_graph/io.py: load_scene(allow_no_seg=False)` + `lbm/render.py` 전달,
  `scripts/render_pred_depth_warp.py --allow_no_seg --allow_empty_dynamic_mask`.**
  SAM3 를 안 돌린 recon 을 통과시키는 스위치 셋. **전부 기본 off** 라 기존 뱅크·릴
  경로는 비트 동일하다. cloud 캐시 키에도 `allow_no_seg` 를 넣었다 — 안 넣으면 같은
  씬의 seg 있는/없는 두 호출이 서로의 점군을 받는다.

### Added
- **`scripts/run_vista4d_gen.py` — 영상 + 카메라 → Vista4D 생성 영상 (2026-09-21).**
  사용자 지시 "vista4d 돌리는 것도 영상, 카메라 주면 영상 나오게끔". `run_custom_caption.py`
  가 끝나는 지점(카메라)에서 이어받는 한 줄짜리 드라이버다.
  `--stage {recon,cam,gen,out,all}` — recon(DA3 depth/mask/cameras.npz, 이미 있으면 건너뜀)
  → 카메라 npz 변환 → 공식 `run_eval_gen.sh` (render_eval + inference_eval) → `source|vista4d`
  가로 concat 릴. 출력은 `results/20260921_vista4d_custom/<name>/<tag>/`.
  **Vista4D 규약 변환은 새로 쓰지 않고** `bank_to_vista4d_cams.py` 한 군데로 몬다 —
  두 경로가 갈리면 같은 궤적이 다른 영상이 된다.
- **`scripts/bank_to_vista4d_cams.py --cams` — 카메라 파일 직접 입력 (2026-09-21).**
  기존 두 소스(`--variants` 뱅크 / `--preds` 평가 폴더)는 둘 다 **이름 규약**을 요구한다
  (`<bank_dir>:<variant_id>`, `<eval_dir>/test/<entry>_transforms_*.json`). 번들에 떨어진
  `cameras/s1234.npz` 나 손으로 만든 `transforms.json` 은 그 규약 밖이라 못 먹였다.
  `--cams <경로>[=<tag>]` 는 확장자로 갈라 `.npz`(이미 OpenCV recon world) 는 그대로,
  `.json`(nerfstudio) 은 `load_pred` 와 **같은 식**으로 OpenCV·recon 픽셀 단위로 돌린다.
  세 옵션 중 정확히 하나를 요구하므로 **기존 호출은 비트 동일**하다.
- **`scripts/run_custom_caption.py` — 영상 1개 + 캡션 1줄 → 카메라 + depth warp (2026-09-21).**
  사용자 지시 "내가 영상 위치랑 caption text 직접 넣으면 우리 모델 돌려서 카메라 저장하고
  depth warp 영상도 만들어서 저장해주는 pipeline". `--stage {recon,geocalib,graph,corpus,
  molmo2,eval,warp,bundle,all}` 하나짜리 python 드라이버다. `run_d215.py` 계열과 다른 점은
  **뱅크가 없다**는 것 — 캡션이 사람 손으로 들어오므로 route/tau/fit/emit/instance_desc 가
  전부 빠지고, 남는 건 모델이 조건으로 읽는 넷(소스 프레임·소스 카메라·avg_scale·캡션)뿐이다.
  `target_poses.npz` 에는 소스 궤적을 자리채우기로 넣으므로 **eval 이 찍는 clatr/caption
  점수는 의미가 없고**, 그래서 `score` 단계가 없다. 산출물은
  `results/20260921_d221_bundles/<name>/<preset>/{caption,info}.json + warp.mp4 + cameras/`.
  실측으로 드러난 배선 세 가지를 코드에 못 박았다:
  ① geocalib 은 전용 env (`envs/geocalib`) — kornia 가 vista4d/latentcam/GenDoP/sam3 넷 다 없다.
  ② recon 의 `--seg_keywords` 는 **실제 명사**여야 한다. `_all_` 이면 SAM3 를 안 타서
  (`recon_and_seg_single.py:82`) `--save_seg_instances` 가 조용히 무시되고, 그 다음
  `scene_graph/io.py:117` 의 `assert segs` 에서 죽는다. 명사는 캡션 지칭구에서 뽑는다
  (`guess_nouns`: "the white dog" → `["white dog", "dog"]`, `--nouns` 로 덮어쓰기).
  ③ `--free_mode` 는 변이 수에 맞춘다. 변이가 1개인데 지칭구를 뽑으면 `aim=look_at` 이라
  `aim=='free'` 변이가 0개가 되고, `zero` 는 그때 assert 로 죽는다
  (`cache_molmo2_embeddings.py:485`). free 변이가 0개면 `zero` 와 `off` 의 산출물은 같다.
  molmo2 override 는 D200 학습 분포대로 `Track {지칭구}.` 한 형식이다 (캡션 원문이 아니다).
- **`scripts/build_bank_captions.py --desc_override` + `configs/bank/d225_desc.json`
  (2026-09-21, D225).** 카메라는 그대로 두고 **target 지칭구만 사람이 고쳐 쓰는** 재시도용이다.
  `{"<씬>": {"<node_id>": "<지칭구>"}}` JSON 을 받아 캡션을 굽기 직전에 갈아끼운다.
  `out/<씬>/instance_desc.json` 을 직접 고치지 않는 이유는 그게 VLM 산출물이라 다음 세대에
  다시 구워지면 손본 게 조용히 날아가기 때문이다. 실사용: car-roundabout 의 VLM 지칭구가
  "the blue car ..." 인데 실제 차는 회색이었고, golf 는 target 을 "a man playing a golf" 로
  줄였다. d224 코퍼스에 덮어쓰지 않고 d225 로 세대를 가른 이유는 캡션 한 줄이 곧 모델
  입력이라, 같은 이름의 `d224_s*` eval 폴더가 다른 텍스트를 가리키게 되기 때문이다
  (memory `eval-dir-name-must-carry-tag` 와 같은 함정).
- **`scripts/route_presets.py --force_presets` — 슬롯 표를 건너뛰고 preset 을 이름으로 지목
  (2026-09-21, D224).** `route()` 의 슬롯 표는 **슬롯당 preset 을 하나로 못 박아** 둬서,
  사람이 이름으로 부르는 요청 중 라우팅으로는 아예 안 나오는 것이 있다 — `advance` 는
  `dolly_in_look_at` 고정이라 맨 `dolly_in` 이 없고, `arc` 는 `pull_out_arc_*` 고정이라
  `push_in_arc_*` 가 없으며, `orbit`/`vertical` 은 away 쪽 한 방향뿐이다. `--slot_whitelist`
  로는 "이 슬롯을 허용" 까지만 되고 그 슬롯이 무엇을 낼지는 못 고른다. 이 인자는 슬롯 목록을
  **통째로** 사람이 준 preset 들로 갈아끼우고 슬롯 이름을 `forced:<preset>` 으로 적어
  `reasons.forced_replaced` 에 원래 슬롯이 뭐였는지 남긴다. 기본값(빈 문자열)이면 예전 경로와
  비트 동일이다 — 코퍼스 굽기에는 쓰지 않는다.
- **`configs/bank/d224_forced.json` — 지목 preset 세대 (2026-09-21, D224).** 사용자가 preset 을
  이름으로 지목한 세 씬(mountain-hike / parkour / golf)을 굽는다. d215 의 tau/fit/emit 을 그대로
  물려받되 route 만 `--force_presets` 로 간다. **d215 뱅크에 얹지 않은 이유**는 같은
  `hole_bank_d215`/`latentcam_d215` 에 카메라를 더 얹으면 `seg_list_d215_test.txt` 의 행 순서가
  밀려, 이미 평가를 끝낸 eval 폴더들이 **조용히** 다른 행을 가리키게 되기 때문이다.
- **`lbm/presets.py` 에 `pedestal_down_dolly_in` (2026-09-21, D224).** 하강하면서 전진.
  `orbit_left_pedestal_up` 과 같은 `<주동작>_<부동작>` 표기이고 aim 은 두 구성 primitive 와 같은
  `free` 다 — pedestal 도 dolly_in 도 조준을 안 하므로(조준은 `_look_at` 으로만 표기, D90 비대칭)
  여기만 `look_at` 으로 두면 어휘 규칙이 깨진다. 실측 `t48 [0, 0.35, 0.35]` = `pedestal_down`
  `[0, 0.35, 0]` + `dolly_in` `[0, 0, 0.35]` 의 합. preset 수 43 → 44.
- **`configs/caption_presets.json` 에 `pedestal_down_dolly_in`** — axis `pedestal`,
  phrase "drops straight down while pushing in toward {target}" (2026-09-21, D224).
- **`scripts/viser_frame.py` — 영상 프레임 고르기 + 그 한 장 png 저장 (2026-09-21).**
  릴에서 "몇 번째 프레임에서 카메라가 벽을 뚫는가"를 찾을 때 영상 플레이어는 프레임 번호를
  안 알려주고, 초 단위로 긁어 `ffmpeg -ss` 로 다시 뽑으면 그 초가 어느 프레임인지 또 어긋난다.
  여기서는 슬라이더 값이 곧 프레임 인덱스이고 파일 이름(`<stem>_f0023.png`)에 그 번호가 박혀
  `viser_cloud.py` 의 frame 슬라이더와 같은 번호로 대조된다. 영상 여러 개를 같이 받아
  (`--video a.mp4 b.mp4 ...`) **프레임 슬라이더를 공유**한다 — arm 5개를 스크립트 5번 띄워
  보면 슬라이더가 따로 움직여 "같은 프레임"을 못 맞춘다. `save frame (all videos)` 는 지금
  프레임을 전 영상에서 한 장씩 뽑고, 길이가 모자란 영상은 **건너뛴다** (클램프하면 다른
  프레임을 같은 번호로 저장해 대조가 거짓이 된다). 디코드는 `decord` 지연 랜덤 접근 + 프레임
  캐시이고, 못 여는 컨테이너만 `imageio` 전량 디코드로 내려간다. 같은 프레임을 두 번 누르면
  `_2`/`_3` 이 붙는다 (덮어쓰면 방금 저장한 것이 조용히 사라진다).
  실측: `warp_f0023.png` 이 `decord[23]` 과 maxdiff 0.
- **`scripts/bank_to_vista4d_cams.py --cam_dir` — npz 저장 위치 분기 (2026-09-21, D221).**
  기본은 여전히 Vista4D 가 읽는 `<eval_data>/eval_data/cameras/<video>/` 다. 릴 번들처럼
  **생성을 안 돌리고 카메라만 보관**할 때는 번들 폴더로 직접 굽는다 — d215 릴 47건 x 4 arm
  = 188 개의 안 쓰는 npz 를 공유 eval_data 에 쌓지 않기 위한 것이다.
- **`scripts/run_gendop_eval.py` 에 `vista_d215` 코퍼스 (2026-09-21, D221).** d215 test 92
  entry 에 GenDoP 릴리즈 ckpt 를 **우리 캡션 그대로** 돌리는 항목. arm 은 모델 3개가 아니라
  한 ckpt(d200 molmo2_l21)의 seed 3개이고 캡션이 셋 다 같으므로 `ref` 는 s42 하나면 된다.
  depth 게이지는 vista_d121 과 같은 `median`.
- **`scripts/bank_to_vista4d_cams.py --preds` — 모델 예측 궤적을 Vista4D 카메라로
  (2026-09-21, D221).** 지금까지 이 다리는 뱅크 `poses.npz` 한 곳에서만 pose 를 읽었다.
  "d215 seed 3개를 Vista4D 로 생성" 하려면 소스가 평가 산출물
  `<eval_dir>/test/<entry>_transforms_{pred,ref}.json` 이어야 한다. 뱅크 경로는 그대로 두고
  (`--variants` 와 둘 중 정확히 하나) 분기만 추가했다. world 가정은 같다 — 예측도 recon world
  절대 미터다. 다른 건 셋뿐: (1) 그 JSON 은 nerfstudio(OpenGL) c2w 라 `GL2CV` 로 돌린다
  (`render_pred_depth_warp.py` 와 같은 식 — depth warp 릴과 같은 카메라여야 한다),
  (2) 해상도가 절반(640x360)으로 적혀 있어 `fl_x/fl_y` 를 `recon cx / json cx` 배 한다,
  (3) frame0 가 회귀 결과라 정확히 소스와 같지 않아(bmx-bumps 실측 6 mm / 0.15도)
  `--frame0_tol` 기본값이 소스에 따라 갈린다 (뱅크 1e-6 / 예측 0.05).
  `--pred_kind ref` 로 같은 entry 의 뱅크 GT 도 뽑아 같은 눈금에서 대조한다.
- **`configs/bank/d220_dynpose_objcentric.json` — car_29a9a7d0 "그냥 track" 세대
  (2026-09-21).** d219(`--slot_whitelist arc` -> `track_pull_out_arc_right`)와 슬롯 하나만
  다르다: `static` 슬롯은 `route_presets.py:342` 에서 `track_look_at`(추종 가능할 때) /
  `static_look_at`(아닐 때) 로 갈리므로, 사용자가 말한 "그냥 track" 이 곧 이 슬롯이다.
  이 씬은 `center_drift_u` 1.5431 로 `--track_min_drift_u` 0.05 를 넘어 `track_look_at`
  하나가 나왔다. 나머지(d185 상속, anchor 못 박기, `--free_moving off`, `--target_variants 1`)
  는 d219 와 같다. 이 config 는 드라이버(`tmp/d219/run_d219.py --tag d220 --slot static`)가
  템플릿에서 생성한 것이다 — 요청마다 config 를 손으로 복사하지 않기 위한 것이고,
  생성물을 커밋해 두는 이유는 뱅크가 어떤 설정으로 구워졌는지 남기기 위해서다.

### Fixed
- **`viser_cloud.py` 가 잘못된 env 에서 cloud 를 다 읽은 뒤에야 죽던 것 (2026-09-21).**
  이 스크립트는 env `vista4d`(viser 1.1.0) 용인데 `latentcam` 의 viser 는 1.0.30 이라
  `add_line_segments(thickness=...)` 를 안 받는다. 실패 지점이 `draw_line` 이라
  **cloud.npz(1.4 GB)를 다 읽은 뒤** TypeError 가 났다 — 잘못된 env 로 띄우면 몇 분 기다린
  끝에 죽는다. `require_viser_thickness()` 를 `parse_args()` 직후로 당겨 0.6초에 멈추고
  올바른 python 으로 된 실행 명령을 그대로 찍는다. 호환 shim 을 넣지 않은 이유: 1.0.30 의
  `line_width` 는 단위가 **screen 픽셀**이라 이름만 바꿔 끼우면 world 0.007 이 0.007 픽셀이
  돼 선과 프러스텀이 통째로 안 보인다 (아래 "프러스텀이 통째로 안 보이던 것" 과 같은 증상) —
  조용히 깨진 화면보다 멈추는 게 낫다. `viser_frame.py` 는 thickness 를 안 쓰므로 1.0.30
  에서도 돈다 (`gui.add_image` / `.image` / `.max` 대입 전부 확인).
- **`run_d215.py` 세대 축이 molmo2 캐시·override·eval 폴더까지 안 오던 것 (2026-09-21, D224).**
  `MOLMO_CACHE`/`OVERRIDE` 는 모듈 상수라 **import 시점의 d215 값**이 박혔고 `eval_dir()` 은
  이름에 `d215_` 를 하드코딩하고 있었다. 그대로 두면 `--generation d224` 로 불러도 캐시는 d215
  자리에 굽고, eval 은 `d215_s42__last` 가 이미 있다며 **건너뛰고**, 릴은 그 옛 예측을 그린다 —
  세 단계 전부 rc=0 이라 안 들킨다. 호출 시점에 푸는 함수(`molmo_cache()`/`override_path()`)로
  바꾸고 eval 폴더 이름에 세대를 물렸다 (`f"{GENERATION}_s{seed}__last"`).
- **`configs/bank/d224_forced.json` 이 track 게이트를 못 끄던 것 (2026-09-21, D224).**
  `--track_dynamic_only` 를 인자 목록에서 **빼기만** 했는데, 이 인자는
  `sample_camera_bank.py:1500` 에서 `action="store_true", default=True` 라 빼도 켜진 채다.
  그 결과 mountain-hike 가 지목한 4개 중 `track_dolly_in`/`track_push_in_arc_left` 둘이
  route 는 통과하고 **뱅크에서만 조용히 빠졌다** (`bank.json` 의 `follow_gains:["0"]`,
  변이 10개가 preset 2종). 끄는 스위치는 `--no_track_dynamic_only` 다. 고친 뒤 4대 전량 나온다.
- **`viser_cloud.py` 기동 `--pin` 여러 개가 전부 같은 색이던 것 (2026-09-21).** 같은 날
  pin 색을 `target start` 피커 스냅샷으로 바꾼 것의 부작용이다 — 스냅샷은 "방금 그 색으로
  보던 궤적이 pin 하는 순간 튀지 않게" 하려는 것인데, 기동 시엔 피커를 바꿀 틈이 없어
  `--pin crane_up/gt crane_up/s42` 가 둘 다 (255,40,40) 이 됐다 (실측). 여러 대를 구분한다는
  pin 의 목적 자체가 깨지므로 **기동 `--pin` 만 `PIN_COLORS` 순환**으로 되돌렸다.
  GUI `pin current` 는 스냅샷 그대로다.
- **`viser_cloud.py` 프러스텀이 통째로 안 보이던 것 — `line_width` 는 폐기된 별칭이었다
  (2026-09-21).** 바로 앞에 넣은 `apply_cam_lw()` 가 `handle.line_width = 0.02` 로 굵기를
  먹였는데, viser 1.1.0 에서 그건 `thickness` 의 **폐기된 별칭**이고 대입하는 순간
  `thickness_units` 를 `"screen"` 으로 못 박는다 (예전 `line_width` 가 픽셀이었으니 그 뜻을
  유지하려고 — 라이브러리 주석에 그렇게 적혀 있다). 프러스텀은 `thickness=0.02`
  **world** 로 만들어졌으므로 같은 숫자가 0.02 **픽셀**로 재해석돼 화면에서 사라졌다.
  경로선은 살아 있었는데, 그쪽은 생성 시 `thickness=` 인자로 world 값을 받고 슬라이더를
  건드리기 전엔 `apply_path_lw()` 가 안 돌기 때문이다 — "path 만 보인다"는 증상이 여기서 나왔다.
  `.thickness` 에 직접 대입하도록 고쳤다 (`apply_cam_lw` / `apply_path_lw` 둘 다).
  `dir(handle)` 에 `thickness` 가 안 보이는 게 함정이다 — 동적 prop 이라 런타임에만 있다.

### Changed
- **`viser_cloud.py` 가 `moving` 플래그 대신 `dyn_*` + track 유무로 OBB 를 애니메이션한다
  (2026-09-21).** `moving` 은 **절대 임계**(`--track_min_drift_u` 0.05)로 정해져서, 작은 물체가
  자기 몸 길이의 몇 배를 움직여도 static 으로 떨어진다 — snow-dog 의 dog 은 track drift
  0.0318 u 인데 자기 extent 가 0.0103 이라 **3.09 배**를 움직였는데도 `moving=False` 다.
  그러면 뷰어가 시간 median OBB 한 개만 그려서 "박스가 개를 따라가는가"를 아예 볼 수 없다 —
  그걸 보려고 띄우는 화면인데. 번들 19 씬 중 **5 씬이 moving 노드 0개**다
  (avocado-slice / camera-lens / hike / mountain-hike / snow-dog). snow-dog 은 이 변경으로
  `0 dyn + 3 static` -> `98 dyn + 1 static` 이 됐다. **예전 동작은 `--obb_anim moving`.**
- **`viser_cloud.py` 가 선 굵기보다 작은 OBB 를 기동 시 알린다 (2026-09-21).** snow-dog 의
  `dyn_1`(dog) 은 extent 0.0103 인데 기본 굵기가 `cam_scale x 0.20` = 0.0110 이라 **자기
  크기의 1.07 배** 굵기로 그려졌다. 박스가 점으로 뭉쳐 "OBB 가 안 보인다"로 읽히고, 원인이
  데이터(cm 단위 OBB)라는 게 드러나지 않는다. 노드 id 와 extent 를 찍고 `OBB thickness` 를
  내리거나 `labels` 로 위치를 읽으라고 안내한다.
- **`scripts/run_gendop_eval.py` 기본값이 `--raw` + `--no_scale_token` (2026-09-21, D222).**
  사용자 지시("gendop 돌릴때 rmax 곱하면 안되고 raw 로 normalized 되어서 나오는 원본 코드
  그대로 나온걸 저장해서 써야해"). 지금까지는 GenDoP 출력에 우리 쪽에서 rmax 를 곱해 미터로
  되돌렸는데, 그건 원저자 코드에 없는 후처리라 베이스라인이 우리 가정을 타게 된다.
  eval_dir 이름도 `eval_dir_gendop_text_p49_raw_noscale` 로 갈라 예전 산출물과 안 섞이게 했다.
- **`results/20260921_d215_complexall/run_d215.py` 에 세대 축 + `--force_presets_map`
  (2026-09-21, D224).** 드라이버가 d215 한 세대에 못박혀 있어서, 지목 preset 을 구우려면
  스크립트를 통째로 복사해야 했다 (CLAUDE.md §"실험별 차이는 설정으로"). `--generation` /
  `--bank_dir` / `--config` / `--corpus_root` / `--seg_prefix` / `--meta_csv` / `--tmp` 를 인자로
  올리고, `--force_presets_map` 에 `{씬: "preset1,preset2"}` JSON 을 주면 `bake_configs()` 가
  **씬마다** base config 를 복사해 그 씬의 `--force_presets` 만 덧붙인 뒤 `run_bank.py` 를 씬별로
  돌린다 (`run_bank.py` 는 config 하나를 영상 목록 전체에 적용하므로 씬별로 다른 preset 을 못 준다).
  파생 config 를 쓸 때 `extends` 를 절대경로로 박는다 — `run_bank.load_config` 가 `extends` 를
  **원본 config 폴더 기준**으로 푼다. 인자를 안 주면 예전과 비트 동일(한 벌 실행).
- **`viser_cloud.py` 프러스텀 화각이 카메라 K 대신 고정 60도 (2026-09-21).** 소스/예측의
  자기 화각을 쓰면 망원 씬에서 프러스텀이 바늘처럼 길어져 "이 카메라가 어디에 있나"를 가늠할
  수가 없다 — bmx-bumps 소스 vfov 17.8°(fy 2293 @ 720p), camel 16.4°. `probe camera` 는 이미
  같은 이유로 `PROBE_FOV_DEG = 60` 을 쓰고 있었는데 정작 본편 프러스텀만 K 를 쓰고 있었다.
  소스·target·pin 전부에 걸린다. arm 별 focal 차이(s42 2287.88 vs recon 2293.41)는 info
  패널의 `fx` 로 읽는다 — 그림에서 자리를 못 읽는 대가로 얻을 정보가 아니다.
  `add_frustums`(GL↔CV 규약의 단일 출처)는 손대지 않고 만든 뒤 `.fov` 를 대입한다.
  GUI `view > camera vfov (deg)` 슬라이더 + `--cam_fov_deg`, **예전 동작은 `--cam_fov_src`.**
- **`viser_cloud.py --bundle` — d221 번들을 씬과 같이 본다 (2026-09-21).**
  `results/20260921_d221_bundles/<scene>/` 를 통째로 올려 `<preset>/cameras/{gt,s42,s1234,
  s2026,gendop}.npz` 를 preset x arm 으로 펴서 기존 motion 슬라이더에 꽂는다 (라벨
  `<preset>/<arm>`, bmx-bumps 3x5 = 15개). `cloud.npz`/`scene_graph.json` 도 **번들 폴더에서**
  읽는다 — 번들 graph 는 그 세대의 스냅샷이라 `out/` 을 다시 구운 뒤에도 그 카메라를 낳은
  씬과 같이 볼 수 있다. `--bundle_presets` / `--bundle_arms` 로 목록을 줄인다.
  **좌표 변환이 없다**: `cam_c2w` 는 이미 recon world 절대 pose·절대 미터이고 recon world
  원점이 소스 frame0 카메라다 (bmx-bumps 실측 `meta_cam_c2w[0] == I`, 8.9e-8). 그 증거로
  `gt.npz` 가 뱅크 변이 `hole_bank_d215/dyn_0__crane_up__hole0.2` 와 위치 2.2e-7 로 일치한다.
  앵커를 한 번 더 곱하면 궤적이 두 번 옮겨진다 (`bank_to_vista4d_cams.py` docstring 이 이
  가정의 단일 출처다). 프러스텀 화각은 **arm 자기 focal** (s42 fx 2287.88 vs recon 2293.41)
  이지만 `jerk p95` 는 소스 focal 로 정규화한다 — arm 마다 분모가 다르면 숫자를 비교할 수
  없다. info 패널은 `caption.json` 을 그대로 읽는다 (그 파일이 `variant_id`/`tau_max`/
  `hole_fraction`/`subject_in_frame` 을 들고 있어 `bank.json` 자리를 메운다).
- **`viser_cloud.py` target 궤적이 시간 gradation (frame 0 빨강 -> 마지막 파랑), 소스 카메라는
  기본 꺼짐 (2026-09-21).** 단색이면 프러스텀 49대(stride 2 면 25대)가 전부 같은 색이라
  시작과 끝이 구분되지 않아서, 궤적을 눈으로 보고도 어느 쪽으로 흐르는지(dolly_in / dolly_out)
  판정이 안 됐다. 프러스텀과 경로선이 **같은 램프**를 쓴다 — 따로 계산하면 `--cam_stride`
  downsample 탓에 프러스텀 i 의 색과 그 자리를 지나는 선의 색이 어긋나 "색 = 시간" 약속이
  깨진다. 경로선은 viser 의 per-point `colors` (N,2,3) 로 꺾은선 하나에 gradation 을 얹었다
  (선을 프레임별로 쪼개지 않는다). 새 GUI: `camera colors > target gradation` 체크박스 +
  `target start` / `target end` 피커 (`target (pred)` 를 대체). 기동 플래그는
  `--plan_color_start` / `--plan_color_end`.
  **pin 은 gradation 을 안 받는다** — gradation 은 "궤적의 어디쯤"(시간), pin 색은 "어느
  변이"(정체) 라 축이 다르고, pin 까지 램프로 칠하면 여러 대를 구분한다는 pin 의 존재
  이유가 사라진다. pin 색은 `target start` 피커의 스냅샷이다.
  소스 카메라는 보려는 게 거의 항상 target 인데 49대가 원점 근처에 뭉쳐 앉아 그걸 덮어서
  껐다. 새 체크박스 `view > source cameras` 가 소스 **프러스텀과 경로선을 같이** 끈다 —
  `paths` 하나로만 묶여 있으면 소스를 끈 상태에서 paths 를 켰을 때 "끈 카메라의 궤적"만
  화면에 남는다. **예전 동작은 `--source_on --no_plan_gradient --plan_color 255,140,40`.**
- **`viser_cloud.py` 경로선과 probe 카메라가 기본 꺼짐 (2026-09-21).** 둘 다 프러스텀보다 눈에
  먼저 들어와 정작 보려는 카메라 자세를 덮는다 — 경로선은 49프레임 꺾은선 3~4개,
  probe 는 gizmo 화살표가 원점 근처 소스 프러스텀을 가린다. GUI `view > paths` 체크박스가
  새로 생겼고 (소스·target·pin·subject track 전부를 한 번에), probe 는 기존
  `show frustum`/`show gizmo` 의 초기값만 뒤집었다. **예전 동작은 `--paths_on` / `--probe_on`.**
- **`viser_cloud.py` `dynamic: all frames` 체크박스 -> `dynamic frames` 드롭다운 (2026-09-21).**
  `current frame`(기본) / `interval` / `all frames` 세 갈래. 체크박스 시절엔 중간이 없어서
  "1장 아니면 49장"이었는데, 궤적을 겹쳐 보려면 띄엄띄엄이 필요하다 (49장 전부는 움직이는
  물체가 49개 복사본으로 번져서 아무것도 안 보인다 — 모듈 docstring 의 4D 설명 그대로).
  기본값 `current frame` 은 예전 체크박스 해제 상태와 **화면이 같다**. 동적 점군과 동적 OBB 가
  **같은 집합**을 쓴다: 점만 띄엄띄엄 켜지고 박스는 한 장이면 어느 프레임의 박스인지 대응이
  안 보인다. 자매 스크립트 `viser_scene.py` 가 이미 쓰던 구성을 옮겨 온 것이라 두 뷰어의
  `view` 폴더가 이제 같은 손잡이를 갖는다.
- **`viser_cloud.py` OBB 색 반전 — 동적이 하늘색 (2026-09-21).** `(255,70,190)` 자홍/
  `(80,200,255)` 하늘색 이던 것을 **동적=하늘색 `(80,200,255)` / 정적=청록 `(0,200,170)`** 로
  바꿨다. 보는 대상은 거의 항상 동적 subject 인데 눈에 먼저 들어오는 색이 정적 배경 박스에
  붙어 있었다. 정적을 같은 하늘색으로 두면 "이 박스가 따라 움직여야 하는가"를 못 가르므로
  청록으로 민다 (snowboard 는 동적 5 / 정적 4 라 두 계열이 늘 같이 떠 있다).
  `--obb_color_dyn` / `--obb_color_static` (`"R,G,B"`) 로 덮는다.
- **`viser_cloud.py` "현재 프레임" 1.6배 프러스텀이 기본 꺼짐 (2026-09-21).** 소스·target·pin
  세 계열 모두. 같은 자리의 일반 프러스텀 위에 큰 것이 겹쳐 앉아, frame 슬라이더로 궤적을
  훑으면 **카메라가 두 대인 것처럼** 보였다 — 지금 프레임은 슬라이더 숫자와 동적 점군으로
  이미 읽힌다. 켜는 건 `--now_cams`; 꺼져 있으면 `camera colors` 의 `source now` /
  `target now` 피커도 같이 비활성이다 (가리킬 프러스텀이 없는 위젯을 눌러 보게 두지 않는다).
- **`viser_cloud.py` pin 색이 `target (pred)` 피커를 따라간다 (2026-09-21).** 예전엔
  `PIN_COLORS` 순환이었는데, 방금 그 색으로 보던 궤적이 pin 하는 순간 팔레트 색으로 튀어
  대조가 끊겼다. 이제 **pin 하는 시점의 피커 색을 스냅샷**으로 뜬다 — 여러 대를 구분하려면
  피커를 바꿔 가며 pin 한다 (구분을 자동으로 주는 대신 사용자가 준다). 예전 동작은
  `--pin_palette`.
- **`viser_cloud.py` OBB `labels` 기본 꺼짐 + `OBB thickness` 기본 0.06 -> 0.20 (2026-09-21).**
  노드가 9개면 라벨 글자가 서로 겹쳐 읽히지도 않으면서 박스를 가린다. 굵기는 0.06 이면
  점군 위에서 박스 모서리가 거의 안 보였다 (단위는 `cam_scale` 배율).

### Added
- **`configs/bank/d219_dynpose_objcentric.json` — "영상 하나 + preset 하나" 요청용 세대
  (2026-09-21).** 사용자가 릴에서 영상을 집어 "이 영상으로 `track+arc right` 로 돌려줘" 라고
  할 때 쓴다. `d185_dynpose100k_grid5` (= D200 학습 분포) 를 그대로 물려받고 **route 만**
  좁힌다: `--anchor_ids_file` 로 피사체 못 박기 + `--slot_whitelist <슬롯 하나>` +
  `--slot_plan full` + `--free_moving off` + `--target_variants 1`. dynpose 판 d215 다.
  재굽기가 필요한 이유는 게이트가 아니라 **예산**이다 — 요청받은 슬롯이 d185 라우팅의
  `backfill` 에 있으면 씬당 카메라 5대 안에 못 들어와 뱅크에 아예 없다 (car 씬 실측: arc 가
  backfill, 구워진 6대는 dolly_out/track_orbit_left/pan_left/crane_up/stat_0×2).
  좌우(`*_arc_right` vs `_left`)를 인자로 강제하는 자리는 없다 — route 의 `source_lateral`
  에서 나오는 `away_side` 가 정한다.
- **`viser_cloud.py` GUI `obb` 폴더 + `view` 폴더 손잡이 복원 (2026-09-21).** `viser_scene.py`
  에만 있던 것들을 옮겨 왔다 — `viser_cloud.py` 에서 **지워진 적은 없다**(그 파일 전체 이력이
  7커밋이고 전부 추가다). 새로 생긴 것:
  - `obb` 폴더: `dynamic OBB` / `static OBB` / `labels` 개별 체크박스(예전엔 마스터 하나),
    `dynamic color` / `static color` rgb 피커, `OBB thickness` 슬라이더.
  - `view` 폴더: `fps`(재생 속도 — 예전엔 `time.sleep(0.08)` 상수), `range start` / `range end`,
    `frame interval (every N)`, `camera thickness`, `path thickness x`.
  - **OBB 색 피커가 이제 가능한 이유**: 드래그마다 249개를 다시 그리는 대신 **버전 도장**
    (`obb_version`)을 찍고 `obb_show` 가 "보이게 되는 순간"에만 빚을 갚는다. `current frame`
    모드면 매번 9개뿐이라 즉시 돈다 (`all frames` + 드래그만 여전히 무겁고, 그건 코드에
    적어 뒀다). 이전 항목의 "OBB 는 GUI 피커를 안 준다"를 뒤집는다.
  - **굵기는 다시 그릴 필요가 없다** — `thickness` 는 line/frustum handle 양쪽에서 갱신되는
    프로퍼티다 (`colors` 만 아니다). **`line_width` 가 아니다** — 위 Fixed 항목 참고.
    단위가 셋 다 다르다: 프러스텀은 world 절대값, OBB 는
    `cam_scale` 배율(씬마다 scale 이 100배 다르다), 경로선은 **선별 기준값의 배율**(소스 0.08 /
    plan 0.10 비율이 슬라이더를 밀어도 유지된다).
  - 경로선 배율을 GUI handle 이 아니라 dict(`lw`)로 들고 있는 이유: 소스 경로선은 GUI 를 만들기
    **전에** 그려져서 그 시점에 슬라이더를 읽으면 없다. 콜백이 dict 를 갱신한다.
  - 경로선은 `state["handles"]` / `pin["handles"]` 에 **안 담고** `lines` 로만 관리한다 —
    양쪽에 담으면 `drop_line` 과 일괄 remove 가 같은 handle 을 두 번 지운다.
- **`viser_cloud.py` GUI `obb > node` 드롭다운 + `--obb_node` (2026-09-21).** 노드 하나만
  남긴다 (`all` 이면 전부). snowboard 만 해도 동적 5 / 정적 4 라 전부 켜면 **어느 박스가 지금
  보는 subject 인지** 고를 수가 없다 — 점군 안에서 박스들이 서로 겹쳐 보인다. 기동 시
  고정하려면 `--obb_node dyn_0`. 배선은 `obb_entry` 에 `node` id 를 싣고 `obb_show` 가
  `node_on()` 으로 걸러 내는 방식이라, 기존 `dynamic OBB`/`static OBB` 체크박스와 **AND** 로
  합쳐진다 (둘 다 통과해야 보인다). 라벨도 `(node_id, handle)` 튜플로 바꿔 같은 게이트를 탄다.
- **`viser_cloud.py` GUI `camera colors` 폴더 + 색 플래그 (2026-09-21).** 지금까지 색을 바꿀 수
  있는 카메라는 probe 뿐이었는데, 정작 대조하는 건 **gt(소스) 와 pred(활성 target)** 다.
  피커 4개 — `source (gt)` / `source now` / `target (pred)` / `target now`. 계열마다 둘인 이유:
  "현재 프레임" 프러스텀은 크기(1.6배) **와 색** 둘로 구분되는데 하나로 합치면 frame 슬라이더를
  밀 때 어느 것이 지금인지 다시 못 읽는다. 기동 기본값은 `--src_color` / `--src_now_color` /
  `--plan_color` / `--plan_now_color`, 안 주면 예전 색 그대로다 (회색/초록, 주황/진주황).
  - **경로선은 `remove` 후 다시 `add` 한다** — viser 1.1.0 의 `LineSegmentsHandle` 은
    `line_width/name/position/remove/visible/wxyz` 만 내놓고 `colors` 가 없다. 프러스텀만
    `.color` 대입이 먹는다. 같은 이름으로 다시 그리므로 씬 트리에 중복이 안 남는다.
  - **OBB 는 GUI 피커를 안 준다** — 동적 박스가 노드×프레임이라 snowboard 만 245개고, 위의
    remove+re-add 를 드래그마다 245번 해야 한다. CLI 플래그만 둔다.
  - `select()` 는 색 상수가 아니라 **피커의 현재 값**을 읽는다. 안 그러면 motion 을 갈아끼우는
    순간 사용자가 고른 색이 기본색으로 되돌아간다.
- **`eval_collision_rate.py` 에 `vista_d215` 코퍼스 (2026-09-21).** 루트는 `vista_d121` 과 같지만
  **코퍼스가 다르므로 entry 인덱스의 뜻이 다르다** — d121 arm 을 여기 섞으면 안 된다
  (`variant_id` 가 코퍼스 키가 아니듯 entry 인덱스도 아니다). `arms` 엔 `gt` + seed 42 하나만
  박아 두고 다른 seed 는 `--extra_eval_dir s1234=...` 로 붙인다.
- **`route_presets.py --anchor_ids_file` + `configs/bank/d215_vista_objcentric.json` (2026-09-21).**
  `{"<video>": ["dyn_1", ...]}` JSON 한 장으로 씬마다 anchor 를 못 박는다. **안 주면 기본값
  `""` = 기존 `pick_anchors` 경로 그대로**(비트 동일).
  - 왜: `pick_anchors` 는 `max_area_frac` **면적 순**으로 고르는데, 사람이 릴을 보고 고른
    피사체는 **이동량(`center_drift_u`) 순** 1등이다. keeper 17 씬 실측에서 **9 편(53%)** 이
    서로 다른 노드를 가리켰고 그중 **5 편은 면적 1등이 아예 `stat_*`**(정지 배경)였다
    (car-roundabout / snowboard / soapbox / drive-desert / magnifying-glass / snow-dog /
    hike / mountain-hike / avocado-slice).
  - **문턱(`--anchor_min_drift_u`)으로는 못 고친다** — 그건 후보를 자를 뿐 남은 것들의
    *순서*는 여전히 면적이라, 문턱을 넘긴 큰 정지 물체가 그대로 1등이다.
  - 파일 경로 하나를 받는 이유: `run_bank.py` 의 `route.args` 는 세대 config 하나에서 전
    영상에 **공통으로** 붙으므로 per-video 값을 인자로 못 넘긴다. 목록에 없는 영상은
    `None` 을 돌려줘 그 씬만 예전 선택 규칙으로 돈다.
  - 목록에 있는 영상은 `--max_*_anchors` / `--min_anchor_sep` / `--anchor_min_drift_u` /
    `--anchor_require_frame0` 를 **전부 우회**한다 — 사람이 이미 고른 노드라 다시 거를 이유가
    없고, 걸러지면 그 씬이 조용히 다른 피사체로 구워진다.
  - `d215_vista_objcentric.json` 은 `d202w` 를 물려받아 **route 만** 바꾼다:
    `--slot_whitelist advance,recede,arc,orbit,vertical,static` (= `rotate`(pan_*, targetless)
    와 `lateral`(truck_*, aim=free — task #134) 제외), `--free_moving off`,
    `--track_mode add --track_min_drift_u 0.05`. 문턱은 tau 단계(`--track_dynamic_only
    --track_min_drift_u 0.05`, d185 상속)와 **같은 값**이어야 한다 — 다르면 라우팅이 붙인
    track 슬롯을 뱅크가 조용히 버린다(D166).
  - tau 블록도 한 줄 다르다: **`--min_subject_area` 0.005 -> 0.001**. 1차 굽기에서 17 편 중
    **6 편이 `no_surviving_anchors` 로 통째로 빠졌는데**, 이유는 사람이 고른 그 피사체가 소스
    frame 0 에서 작아서였다 — snowboard 0.00151 / soapbox 0.00113 / drive-desert 0.00440 /
    mountain-hike 0.00388 / avocado-slice 0.00222 (전부 0.005 미만), magnifying-glass 0.0.
    기본 세대는 면적 1등 anchor 를 쓰니 이 컷에 안 걸리지만 D215 는 anchor 를 사람이 못 박으므로
    컷을 피사체에 맞춰 내린다. **magnifying-glass 는 0.0 이라 이 값으로도 못 살린다**
    (frame 0 에 아예 안 보인다 = 조준할 것이 없다).
- **`gendop_release_infer.py --batch_size` (+ `run_gendop_eval.py --batch_size` 배선) (2026-09-21).**
  한 번의 `model.generate` 에 엔트리 N 개를 같이 넣는다. **기본 1 = 기존 동작(비트동일)** 이라
  예전 호출은 그대로다.
  - 왜: 생성은 `10*pose_length+1`(=491) 스텝 autoregressive 디코드인데 B=1 이면 GPU 가 논다.
    실측 **17.8 entry/min (3.37 s/entry), GPU util 26%** (dynpose d200 test 런, 최근 300개 기준).
    디코드 스텝 수는 B 와 무관하므로 배치는 거의 그대로 배수 이득이다.
  - GenDoP 리포는 여전히 **0줄 수정** — `core/models.py:283 generate` 는 `assert B == 1` 이
    이미 주석 처리돼 있어 B>1 을 받는다. `prefix_allowed_tokens_fn` 도 batch_id 별로 돈다.
  - **전제: `forbid_eos` 가 켜져 있어야 한다.** EOS 를 허용하면 엔트리마다 길이가 달라지고
    먼저 끝난 시퀀스가 `pad_token_id=0` 으로 채워지는데, `models.py:358` 이 전 토큰에서 3 을
    빼 `-3` 을 만들어 `:359 assert np.all(tokens >= 0)` 이 죽는다. 그래서 `--batch_size > 1`
    인데 forbid_eos 가 꺼져 있으면 **조용히 터뜨리지 않고 1 로 되돌린다** (사유를 찍는다).
    `pose_length != 30` 이면 forbid_eos 가 auto-on 이라 49-포즈 런은 그대로 배치된다.
  - 이어달리기는 이미 공짜다 — 출력이 엔트리당 `.npz` 고 `path.exists` 스킵 가드가 있어
    같은 `--out` 으로 다시 띄우면 남은 것만 돈다. RNG 는 `generate_mode='sample'` 이라 배치
    경계가 바뀌면 같은 엔트리도 다른 샘플이 나온다 (재개가 원래 갖고 있던 성질과 같다).
    `batch_size` 는 `config.json` 에 적는다.
- **`eval_subject_in_frame.py --merge` 에 `--scenes` 필터 + arm 별 씬 수 경고 (2026-09-20).**
  전량 샤드가 이미 있으면 표본(200 씬) 표는 다시 렌더할 필요가 없다 — 같은 행을 거르기만 하면
  되고 수치는 표본을 따로 돌린 것과 같다. `--merge` 없이 쓰는 기존 경로는 **비트 동일**
  (`wanted is None` 이면 분기 자체가 없다).
  - 서로 다른 작업의 샤드를 합칠 때(우리 5 arm 만 돈 샤드 + 베이스라인 6 열만 돈 샤드)
    한쪽이 덜 끝나 있으면 열마다 분모가 달라진다. 합친 뒤 **arm 별 씬 수**를 찍고, 어긋나면
    `[경고]` 를 낸다 — 조용히 지나가면 같은 표로 읽힌다.
  - `report["scenes_file"]` 에 표본 파일 경로를 남긴다 (`--scenes` 는 산출물 옆에 "어떤
    표본이었나"가 남아야 한다는 기존 규칙과 같은 이유).
- **`eval_collision_rate.py` 에 `dynpose_d200` 코퍼스 + `--scenes` / `--extra_eval_dir` /
  `--per_scene_max` (2026-09-20).** D200 5 arm 의 testset 충돌률을 재려고 붙였다.
  기존 `vista_d121` 경로는 **비트 동일** — 새 키(`eval_data`/`graph_root`)는 코퍼스 dict 에서
  없으면 예전 상수(`Vista4D-Eval-Data`, `out/`)로 떨어지고, 새 인자 기본값은 전부 무동작이다.
  - 코퍼스마다 recon 루트와 scene_graph 루트가 다르다 — d200 은
    `DynPose-100K` + `out_dynpose`. `eval_data` 는 **`eval_data/` 의 부모**를 적는다
    (`scene_graph/io.py:88` 이 다시 붙인다. `run_gendop_eval.py:60-63` 에 같은 함정이 기록돼 있다).
  - `cloud.npz` 는 안 읽는다. 점군은 recon depth 에서 직접 세우고 `scene_graph.json` 에서는
    `S` 만 읽으므로 D178 의 cloud 삭제와 무관하다 — `--cloud_source` 같은 게 필요 없다.
  - `--scenes FILE` 로 D205 의 200 씬 표본만 잰다. 5,144 엔트리 × 1,017 씬 전량은 비싸고,
    같은 표본을 써야 `subject_in_frame` 표와 열을 나란히 놓을 수 있다.
  - `--per_scene_max` 는 **출력만** 자른다 (JSON `by_scene` 은 전량). 200 씬이면 씬 블록이
    200 개 찍힌다.
  - 실행 전 200 씬 전부에 `scene_graph.json` / `recon_and_seg` / `depths` / `seg_instances`
    가 있는지 확인했다 (결손 0). 중간에 죽지 않게 하려고 스킵 분기를 넣는 대신 사전 검증을 택했다 —
    조용한 스킵은 "다 쟀다"로 읽힌다.

### Changed
- **`run_gendop_eval.py` `dynpose_d200.ours` 에 arm ⑤ `d200_molmo2_dec_l21` 추가 (2026-09-20).**
  ⑤ 가 14:34:49 에 `epoch_cap 50` 완주(rc=0)해 testset eval 을 돌릴 수 있게 됐다. 이 항목이
  빠져 있으면 D208 합산 score 가 **⑤ 없이** 9시간을 돌고 끝난다 — ⑤ 는 epoch 50 val 에서
  loss_traj / loss_latent / clatr fcd / caption fscore 4개 1등이라 비교표의 핵심이다.
  15:33 에 시작한 score 를 2분 만에 끊고 ⑤ 를 채운 뒤 재기동했다.
  - ⑤ 만 run 디렉토리 날짜가 다르다 (`20260919_172530_...`, 09-19 재기동본). 동명 디렉토리가
    5개 있고 `ckpts/last.pth` 는 이 하나뿐이라, 앞 4개와 같은 09-18 이름으로 찾으면 없다.

### Added
- **TRUMANS 전량 뱅크 세대 `d207T` (`configs/bank/d207_trumans_pilot.json`) (2026-09-20).**
  TRUMANS-Lite 191 chunk 을 **d185 라우팅 축**(`route_presets` grid2x2 슬롯)으로 다시 굽는다.
  기존 `hole_bank_k6_d132` 는 route 단계 없이 tau/fit 에 `--nodes dyn_0` 을 직접 박은 옛 합집합
  preset 이라 d200 코퍼스에 pooled 할 수 없다. 폴더 이름을 `d207T` 로 가른 것은 D105 원칙
  (게이트 규약이 바뀌면 이름을 바꾼다).
  - anchor 는 **사람(`dyn_0`) 하나만** — `route_presets.py` 에 `--nodes` 가 없으므로
    `--max_dynamic_anchors 1 --max_static_anchors 0 --max_anchors 1` 로 같은 결과를 낸다.
  - chunk 당 최대 3 변이(anchor 1 × grid 슬롯 2 + free-moving 1)라 `--target_variants` 와
    fit 의 `--fallback_target` 을 둘 다 3 으로 맞췄다. 어긋나면 fallback 이 못 채우는 자리를 계속 재시도한다.
  - `--cloud_source npz` — TRUMANS `cloud.npz` 는 `--scene_scale_mode points_first_cam
    --scene_scale_stride 1` 로 구워져 있고, memory 경로로 다시 구우면 그 스케일 인자를 안 받아 `S` 게이지가 달라진다.
  - `--collision_source both` (d132 상속) — `.blend` 가 있어 `mesh_grid.npz` 가 소스 카메라가
    한 번도 안 본 벽·옆방까지 안다. d116 실측(chunk a00, 836 변이) depth 130 / mesh 357 / 겹침 0.
  - `stages` 에서 graph/cloud 를 뺐다. 191/191 편에 `.graph_s115` + `.cloud_s115` + `mesh_grid.npz` 가
    이미 있고(mesh GT 기반), 다시 지으면 같은 마커 아래 다른 게이지가 섞인다.
- **`trumans_gt_render.py --anim` — 프레임별 render job 을 animation job 한 번으로 (2026-09-20).**
  프레임마다 `bpy.ops.render.render()` 를 부르면 그 한 번이 Cycles 세션 하나라 씬을 통째로 다시
  device 로 올린다 (`use_persistent_data` 는 한 job 안에서만 먹는다). 카메라 pose 를 CONSTANT 보간
  keyframe 으로 굽고 `scene.frame_step` 을 맞춰 animation render 한 번으로 묶는다.
  - 실측(00add26c_a01 `_source`, 49 프레임 640×360 48 spp OPTIX, 같은 blend/pose):
    프레임별 job **272.6 s** (5.547 s/frame, first 제외) vs `--anim` **90.3 s** (1.84 s/frame) → **3.02×**.
  - 정확도: 픽셀 차이는 adaptive sampling 잡음 수준(meanAbs **0.0607/255**, maxAbs 24),
    카메라는 `c2w_opencv`/`c2w_blender_gl` maxAbs **4.768e-07** 로 일치.
  - **기본 off = 예전 동작 비트 동일.** `--passes rgb` + 균일 간격 프레임 + cycles 전용이라
    depth/index(프레임마다 엔진을 바꾼다)나 `--frame_list` 와는 assert 로 배타.
  - `render_meta` 에 `"anim"` 을 남긴다. `--anim` 일 때 프레임별 `timings_s` 는 job 총시간/프레임수라 합만 정확하다.
- **`bank_to_blender_poses.py --anim` / `--rgb_cdevice` (2026-09-20).**
  생성되는 `render.sh` 가 `trumans_gt_render.py --anim` 을 부르도록 배선. 기본 off 라 예전 `render.sh` 와 글자 동일.
  (플래그 이름이 `--cycles*` 로 시작하면 Cycles 애드온이 argv 를 prefix 매칭해 런을 죽인다 — 그래서 `--cdevice`/`--rgb_cdevice`.)

- **`sample_camera_bank.py` / `fit_hole_ladder.py` `--mesh_margin_autoclamp` (2026-09-20).**
  D207 1차 전량에서 fit 이 191 chunk 중 **11 편**에서 D47 legality assert 로 죽었다 —
  `--mesh_margin_frac 0.08` 이 만든 임계(0.16~0.30 m)가 **소스 카메라 자신의 mesh 여유**
  (0.10~0.28 m)보다 커서다. TRUMANS 는 실내라 카메라가 벽에 붙는 chunk 가 있고, 소스 카메라를
  불법으로 판정하는 임계는 충돌이 아니라 버그다.
  `--mesh_margin_autoclamp f` 를 주면 **그런 chunk 에서만** 임계를 `f × 소스여유` 로 내린다.
  이미 합법인 chunk 는 코드 경로가 그대로라 **기본값 0.0 은 물론 켜도 비트 동일**이다.
  내렸을 때는 `behind_context` 가 `mesh_margin_clamped=(before, after)` 를 반환하고 로그를 찍는다.
  - `configs/bank/d207_trumans_pilot.json` 의 `fit.args` 에 `0.9` 로 켰다. 10/11 편이 rc=0 으로
    복구됐다 (180 편은 손대지 않았다). 남은 `tru_0ab03928_a13_s3f0k6` 은 다른 assert
    (`소스 카메라가 mesh 격자에서 unreachable`)라 이 플래그로는 안 산다.
- **`run_gendop_eval.py` score 단계에 외부 베이스라인·표본·cloud_source 배선 (2026-09-20).**
  전부 기본값이 예전 동작이라 기존 런과 비트 동일이다.
  - `--extra_eval_dir LABEL=DIR` (반복 가능) — 우리 파이프라인 밖에서 만든 eval 폴더
    (E.T./DIRECTOR)를 같은 표에 올린다. `--arm none` 은 GenDoP arm 을 하나도 안 올린다
    (`arms=[]` 와 `arms=None` 을 `or ARMS` 로 못 가르므로 `stage_score` 쪽 분기를 고쳤고,
    infer/evaldir 에서 빈 리스트가 조용히 "전부"로 되살아나는 것은 assert 로 막았다).
  - `--scenes FILE` / `--out_tag` — D205 의 200 씬 표본(`tmp/d205/sample200_scenes.txt`)으로
    재서 그 표와 열을 나란히 놓는다. `--subject_occlusion` 도 같이 넘긴다.
  - `--cloud_source {npz,memory}`.
- **`gendop_preds_to_eval_dir.py --caption_dir` (2026-09-20).**
  `--caption_from ref` 일 때 캡션을 가져올 폴더를 따로 준다. 텍스트 조건을 갈아끼운 런
  (`--text_dir`)이 지표 단계에서 **모델이 받지 않은** 문장을 읽는 것을 막는다.
  `run_gendop_eval.py` 는 `--text_dir` 를 줬을 때 자동으로 같이 넘긴다. 기본값 None = ref 폴더.
- **`dynpose_gendop_inputs.py --text_only` (2026-09-20).** rgbd 루프와 symlink 를 건너뛴다. 기본 off.
- **`run_director_batch.py` (신규, 2026-09-20).** DIRECTOR(E.T.) 를 split 전량에 돌려
  `gendop_preds_to_eval_dir.py` 가 먹는 npz 를 낸다. 모델·CLIP 을 한 번만 올리고 16 entry 씩
  묶어 `Diffuser.sample()` 을 부른다 (`StackedRandomGenerator` 가 배치 원소마다 별도
  `torch.Generator` 를 쓰므로 `seeds=[seed]*B` 는 개별 실행과 등가 — 이어달리기도 안전).
  E.T. world(m) → scene-graph G(u) → 코퍼스 world 변환을 담고, **카메라 축 규약을 런마다 실측**해
  찍는다 (5,144 entry: 카메라 +z 와 char−cam 사이 각 mean 36.63° / median 27.97° → **OpenCV**.
  `run_director_vista.py` 에 "미검증"으로 남아 있던 항목이다).

### Changed
- **`run_gendop_eval.py` `dynpose_d200` 의 `eval_data` 를 `DynPose-100K/eval_data` 로 정정
  (2026-09-20).** d200 은 d185+d199 pooled = dynpose-100k 이라 recon 이 거기 있다. d137 까지
  쓰던 `DynPose-LBM/eval_data` 를 물려받는 바람에 1,017 씬 중 **107 씬**(두 코퍼스가 겹치는 몫)만
  잡혔고, 그 107 을 보고 "rgbd 입력이 10% 뿐"이라고 잘못 읽어 `text_only=True` 를 걸었다.
  실제로는 `video.mp4` / `depths/00000.exr` 둘 다 **1,017/1,017** 이라 `gendop_rgbd` 도 돌릴 수
  있다 — `text_only` 플래그는 남기되 이 코퍼스에서는 뗐다. score 단계도 이 루트로 recon 을 읽는다.
- **GPU 정책: TRUMANS 작업에 한해 GPU 5 사용 가능** (2026-09-20 사용자 지시, `CLAUDE.md` 반영).
  뱅크 굽기·Blender 렌더·TRUMANS 캡션용 vLLM 이 해당한다. `run_bank.py` 는 `BANK_GPU_ALLOW=5` 를
  줘야 `--gpu 5` 를 받는다 (없으면 `--gpu` ∈ `"01234"` assert). 6,7 은 여전히 금지,
  TRUMANS 가 아닌 학습/추론은 0~4 그대로.

### 측정했으나 넣지 않은 것
- **정적 모디파이어 베이킹 (`--bake_static`) — 되돌렸다 (2026-09-20).**
  geometry 의 91%(898,861 vert 중 821,806; `vase_03` 하나가 geometry nodes 로 792,481)가 local space
  에서 프레임 불변인데 부모 empty 가 애니메이션이라 매 프레임 재평가된다. 베이킹하면 `--anim` 에서
  90.3 → 88.6 s (1.9%), 프레임별 job 에서 272.6 → 258.8 s (5.1%). 둘 다 잡음 수준이라 코드를 전부 제거했다.
- **여러 preset 을 한 프로세스에서 렌더 — 기각.** steady state 프로세스 오버헤드가 preset 당 ~6.4 s
  (blend 로드 + Blender 기동 4.29 s; 첫 프로세스만 OptiX 워밍으로 21.7 s) 라 묶어도 ~5%.
