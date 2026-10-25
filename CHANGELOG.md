# Changelog

저장소 **최상위** 문서·설정의 변경 기록. 하위 트리의 변경은 각자의 CHANGELOG 에 적는다 —
`camera_generation/dataset/CHANGELOG.md`, `camera_generation/latentcam/CHANGELOG.md`,
`video_generation/CHANGELOG.md`. 여기에 오는 것은 세 트리에 **걸쳐 있는 것**뿐이다
(`pipeline.md`, `README.md`, `.gitignore`, `CLAUDE.md`).

[Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

## [Unreleased]

### Fixed

- **`latentcam/main/train_latent_cam_dm.py` · `scripts/eval_testset.py`** (R529). occ_grid / depth_vid 모델 kwargs 가 영상 feature (`video_latent_dim>0`) 분기 안에 있어 Molmo2 없는 arm 에서 빠지던 것 — 분기 밖으로. 영상 있는 arm 은 동작 동일.

- **`latentcam/main/cache_umt5_embeddings.py`** (R526). `merge_caches` 공용 함수 추가 — tmp 드라이버의 umt5 캐시 합치기가 offsets 를 한 칸 밀려 붙여 뒤 캐시 문장이 앞 문장 임베딩을 받던 버그를 고친 판.

- **`dataset/fit/bank/occ_grid_cache.py`** (R521). 세그먼트 ESDF (`occ_edt*`) 리샘플이 world 격자 밖을 `mode="nearest"` 로 가장자리 값 복사 → 관측 안 된 바깥에 축 방향 '장애물 근처' 줄무늬. `--edt_outside extend` 추가 (격자 안 가장 가까운 점의 거리 + 밖으로 나간 거리). 기본값은 기존 동작 (nearest) 유지.

- **`scripts/eval_testset.py`** (R482). `depth_vid_cond` 학습 run 의 모델 kwargs 를 안 넘겨 strict load 가 `dv_*` 키에서 죽던 것 — 학습 진입점과 같은 kwargs 추가.

- **`dataset/exec/native_captions.py`** (R476). 같은 씬 4×4 grid 를 여러 스레드가 동시에 쓰다 반쯤 쓴 PNG 를 VLM 에 넘기던 경합 — lock + 임시파일 `os.replace`.

- **`scene_graph.gt_trumans.board_folder`** (R446). recon 작업 폴더가 `<uuid>_lph_ng` 인 board 클립의 GT 중력을 못 찾아 graph 가 실패. uuid 폴더가 없을 때만 그 board key 의 manifest 가 든 폴더를 받는다 (예전 클립은 동작 같음).

- **TRUMANS `.blend` 경로 규칙** (R446). 폴더에 `_` 접미사가 붙은 recording (`<uuid>_lph_ng/<uuid>.blend`, 4a080499 · 4a87a708) 을 못 찾아 board 원본이 0 개였다. `trumans_to_recon.main` · `lbm/blender_raycast.blend_of` · `bank_to_blender_poses.chunk_paths` 가 같은 규칙으로 찾는다 (사본 `<uuid> - 副本` 은 계속 제외).

- **subject 점 조건이 readout 없는 arm 에서 모델에 안 들어가던 문제** (R382). `train_latent_cam_dm.py` · `eval_testset.py` 가 `subject_pts_*` kwargs 를 `peav_readout_layers>0` 블록 안에서만 넘겨, MTp · MTpf · R379 첫 기동분은 점 모듈 없이 학습됐다. 블록 밖으로 옮기고, `subject_pts_cond=true` 인데 모듈이 없으면 assert.
- **VAE worldtraj 캐시 translation 분모를 scene scale 로** (R175). `train_camera_vae.py --wt_norm avg_scale` (기본) — 창별 mean‖R_w2c p + t_w2c‖ (scene.ply 점, Dataset-v1 transforms avg_scale · worldtraj avg_scale/{k}.json 과 같은 정의, 3창 일치 확인). 새 stage `wt_scale` (병렬, → cache/wt_avg_scale.json), cameras.json 목록은 cache/wt_files.json 에 저장해 재사용. R125~R170 은 창 카메라 이동거리 s / m1 로 나눴다 (`--wt_norm cam_disp` 로 재현).
- **VAE worldtraj 캐시의 cameras.json 해석 c2w → w2c** (R169). `train_camera_vae.py --stage cache_wt` 가 `rotation`/`position` 을 c2w 로 읽어 worldtraj 창(학습 228,130 · test 2,846)이 전부 역궤적이었다. 레포 규약(`main/dynamicverse_shim.py`, CamVLA `data/worldtraj`)과 scene.ply 투영 (images_pc_rendered 대비 색 오차 c2w 50~63 / w2c 18~21) 모두 w2c. `--wt_pose w2c|c2w` 신설, 기본 w2c · c2w 는 R125~R168 캐시 재현용.

### Added

- **`dataset/fit/ingest/trumans_rgb_fill.py`** (R558). `--geom_only` (R487 1 단계) TRUMANS 클립에 조건 영상 (RGB) 을 나중에 채운다 — manifest 의 저장된 카메라로 Cycles RGB 만 (`--anim`, 16 spp, 1 단계와 같은 설정) 다시 렌더해 recon video.mp4 를 바꾸고 RGB_PENDING 을 지운다. depth 와 프레임 정렬 확인 (1 클립 188 s). png 는 지운다 (`--keep_png`).

- **`latentcam/models/intent_planner.py`** (R551). `IntentPlanner.relation_first` — move_slots 와 함께, follow (track) 여부를 먼저 따로 묻고 (유사 예시의 follow 정답 포함) 슬롯 답의 follow 를 그 값으로 고정 (LAMP 의 관계 계열 먼저). 기본 False 면 기존 그대로.

- **`latentcam/models/intent_planner.py`** (R549). `IntentPlanner.move_slots` — two_stage 의 move 질문을 LAMP 식 축별 필수 슬롯 (follow·centered·forward·side·vertical·orbit·pan·tilt, 'no' 포함) 으로 받고 코드가 preset 을 정한다. `combo_to_slots` 로 retrieved 예시도 슬롯 형태. 74 조합 왕복 일치 확인. 기본 False 면 기존 조합 enum 그대로.

- **`latentcam/models/intent_planner.py`** (R540). `IntentPlanner.retrieved` — two_stage 의 move 질문에 호출마다 바뀌는 유사 요청 → 정답 조합 예시를 붙인다 (검색은 호출자). 기본 None 이면 기존 프롬프트 그대로.

- **`latentcam/scripts/eval/eval_chain.py`** (R535). 학습 끝난 run 하나를 eval_testset → clatr_score_dir → traj_time_metrics → pred_collision_rate (shell·esdf) 로 한 번에 평가·채점 — R489e·R492·R534 드라이버마다 복사하던 체인을 승격. 산출물 있으면 단계 건너뜀, 충돌 채점은 OMP/MKL 4 스레드 (R527 esdf 멈춤 방지). 표 합치기는 기존 `dataset/eval/arm_table.py`.

- **`dataset/exec/run_gendop_eval.py`** (R527). corpus `dynpose_v5f_all_v2` (v5f_all test 6,111, ref = R489 eval 폴더) — main 비교표 (D61) 용.

- **`dataset/exec/native_captions.py`** (R525). `--pose_source corpus` — eval 폴더 없이 코퍼스 `target_poses.npz` 에서 GT pose (train 포함). ref 경로와 같은 태그 확인.

- **`dataset/eval/gendop_release_infer.py` · `exec/run_gendop_eval.py`** (R508). `--batch_eos_fallback` — GenDoP 을 EOS 허용 그대로 배치 generate (pipeline/GenDoP `eval_batch.py` 방식), 배치 안 길이 불일치로 assert 가 나면 그 배치만 한 개씩 재생성. 재생성 배치 수를 config `batch_fallbacks` 에 남긴다. 기본 꺼짐 = 예전 (batch_size 1 로 강등).

- **`dataset/exec/native_captions.py`** (R508). `--model pipeline` — 우리 fork `pipeline/GenDoP` tagging 설정 (49 pose · fps 10 · diff 0.6 · 원 스케일 · 창 15/최소 구간 12) 으로 Movement 문장을 만들어 `Concise Interaction` 에 넣는다. `--gendop_root` 인자 통과.

- **`dataset/exec/run_director_batch.py`** (R507). `--char_static` — E.T. char 궤적을 첫 프레임 위치에 고정 (정지 궤적 ablation). 기본 꺼짐.

- **`latentcam/models/intent_planner.py`** (R497·R498). `IntentPlanner.prim_defs` (기본 False = 예전) — v2 joint_struct 프롬프트의 조합 설명 괄호를 가장 작은 단위 정의 문단 (`PRIM_DEFS_V2`: dolly·truck·pedestal·orbit·pan·tilt·track·keep centered·hold; track 은 "따라가라는 요청일 때만, 움직이는 subject 만으로는 아님" — R499; hold = "위 이동 단위 없음" — R500) 으로 바꾼다. `extra_context` (프롬프트 끝 추가 정보) · joint_struct `video=None` (텍스트만). v2 재계획 피드백 문구 (`FAIL_FIX_V2`) 에서 push in / crane / arc 를 v2 용어로. `IntentPlanner.lean` (R501, 기본 False): joint_struct 에서 end_shot 출력 제거 · target.text 길이 제약 제거 · 영상 없으면 system 문구에서 video 제거. `IntentPlanner.examples` (R501, 기본 False): 손으로 쓴 few-shot 8 개 (`JS_EXAMPLES_V2`) — 시험 결과 손해라 기본 꺼짐. (R503·R504, 전부 기본 꺼짐) `end_shot_explicit` — intent 가 끝 shot 크기를 명시했을 때만 end_shot · `examples="v2b"` — 실제 train L1s 문장 few-shot 11 개 (`JS_EXAMPLES_V2B`, 출력과 같은 JSON) · `target_loose` — target.text 완전 일치 대신 내용어 70% (`target_overlap`) · `prompt_style="clear"` — 단계식 프롬프트 (`JOINT_STRUCT_CLEAR`), reasoning 키를 move 앞에 두어 선택 전에 생각을 쓰게 함. joint_struct 마무리를 `_finish_js` 로 분리. (R506) clear ablation 옵션 `clear_steps` (Step 1-3 문단 유무) · `reason_pos` (reasoning 키 first/last/none). (R517) `two_stage` — clear 에서 ① subject 문구를 텍스트만으로 따로 (`TARGET_ONLY_PROMPT`) ② 카메라 패턴만 (영상·추가 맥락 포함, target 출력 없음) 두 번 묻는다. 기본 꺼짐. (R523) `target_full` — 2 단계의 subject 질문을 동작 묘사 포함 문구 전체 (카메라 모션 제외) 로 (`TARGET_ONLY_FULL_PROMPT`). 기본 꺼짐. (R509) `judge_l1` — L1 이 GT canonical 과 같은 단위만 담는지 검수 (same/different + 유형). `synthesize_l1_verbatim(feedback=, defs=)` — 검수 실패 사유를 넣어 재생성, defs 면 v2 단위 정의 + 기술 용어 (dolly·truck·pedestal·crane, GT 에 없는 pan·tilt·orbit) 금지. 기본값이면 예전 프롬프트 그대로.

- **`intent_planner.synthesize_l1_verbatim`** (R496). L1 intent 만 다시 합성하되 subject 지칭구 (캡션 target_text) 를 원문 그대로 포함하게 강제 (검증·재질의). v5f50k noroll 47,949 전량 성공 → `prompts_intent_L1s.json`.
- **`IntentPlanner.target_from_intent`** (R496). joint_struct 에서 target.text 를 intent 안의 문구 그대로만 허용 (planner 는 카메라 패턴만, subject 는 파싱). 기본 False = 예전.
- **`dataset_dl3dv._mix_caption` · `prompts_file_mix` / `prompts_mix_prob`** (R495 B). 학습 샘플마다 확률로 캡션을 다른 prompts 파일 (같은 세그먼트 키) 문장으로 교체. 기본 null / 0 = 예전.

- **`IntentPlanner(mode="joint_struct")`** (R493). 영상 + intent 한 번에 보되 움직임은 split 과 같은 74 조합 enum (구조화 출력) 에서만 고르고 subject 도 같이 낸다. 사용자 10-22 "영상도 planner 입력으로 · joint struct 기준으로".

- **`IntentPlanner(mode="split")`** (R491). 움직임은 intent 텍스트만 보고 74 조합 enum 중 하나를 고르고 (vLLM 구조화 출력), 영상은 subject 지칭에만 쓴다. 기본 `joint` = 예전. roundtrip (300 seg) preset 일치 L0 0.49→0.81 · L1 0.28→0.48 · snap 0.
- **`lbm/vlm.VLMClient.chat/chat_json(json_schema=...)`** (R491). vLLM `response_format` json_schema 전달. 안 주면 payload 예전 그대로.

- **`trumans_to_recon.py --geom_only` · `run_board_sources.py --geom_only`** (R487 2단계 렌더 1단계). depth/index 만 렌더 (RGB 생략, video.mp4 회색 자리표시자 + `RGB_PENDING`), 변환 뒤 render 폴더의 depth/index npy 삭제 (cameras.json 은 남김). smoke 1 클립: 렌더 77 s, recon 6.4 MB.
- **`scene_graph/gt_trumans.board_folder`** (R487). 세대 접미사 클립 (`tru_<rec8>_cNN_fNNNNN_kNNN_pN_<tag>`) 을 `<recon_root>_<tag>/` 작업 폴더로 해석. 접미사 없으면 예전 그대로.

- **R487 TRUMANS subject 화면 지표 = 사람 GT mesh 실루엣.** `fit/ingest/blender_raycast_server.py` 에 `screen` 요청 (load_clip `verts=True` 로 프레임별 사람 mesh 정점 표본 캐시 → 저해상도 격자 투영 + 3×3 closing 으로 면적·중심, 실루엣 정점 일부에 정적 BVH 광선으로 가림), `lbm/blender_raycast.RaycastClient.screen`, `fit_hole_ladder --subject_screen {render,mesh}` (기본 render = 예전; mesh 는 `--ray_gate server` 필요, 이분법·검증 두 곳에서 area/onscreen/in_frame/visible 열을 덮어씀. coverage 는 점군 그대로).
- **`exec/run_board_sources.py --gen_tag --track_gain`** (R487). 새 세대 소스 클립을 이름 접미사·작업 폴더를 갈라 렌더하고 소스 카메라 사람 추종률을 넘긴다. 안 주면 예전 이름·0.6.
- **`configs/bank/r487_trumans_v5f.json`** (R487). r442c + ray gate (clear_frac 0.6, subject_dist 0.8 유지) + `--subject_screen mesh` + min_subject_in_frame 범위 v5f 정규식 + route pool 24.
- **`exec/run_corpus_export.py --desc_override`** (R486). build_bank_captions 로 지칭구 덮어쓰기 전달 (E2 얼굴 "the face of the …").

- **`dataset/eval/arm_table.py`** (R476). 여러 arm 의 traj_time_metrics per_entry · CLaTr v2 json · 충돌 per_entry 를 look-at/free 로 나눠 한 표로 (ADE · rot · |fr_off−gt| · fr_err · in_box · shell/esdf coll). R484·R476 에서 두 번 손으로 짠 집계를 승격.

- **`dataset/exec/native_captions.py`** (R476). 외부 baseline 의 원 캡션 파이프라인을 우리 test GT 카메라에 그대로 태운다 (LLM 만 로컬 Qwen3-VL). `--model gendop`: DataDoP 120 pose·fps 30 분절 → Movement → 4×4 grid(Filtering.py 규칙) + Movement → Concise, 위치는 0.323/S 게이지. `--model et`: E.T. 좌표(키 1.7 m)·25 fps 리샘플 → 카메라/char 분절 → cam+char 프롬프트 (T 0.6). 산출은 eval 폴더 모양 `test/<prefix>_<scene>_<idx>_caption.json`.

- **`dataset/exec/run_director3d_batch.py`** (R476). Director3D 공개 ckpt 의 Traj-DiT 단계만 배치로 돌려 (DDIM 100 step, 원본 샘플러 그대로) 29 카메라 → Director3D 자신의 SLERP/LERP 로 49 → GenDoP 어댑터 규약 npz (`director3d__<scene>__<idx>.npz`, OpenGL c2w). 렌더러 import (`diff_gaussian_rasterization`·`kiui`) 는 빈 모듈로 막고, SD2.1 은 tokenizer·text_encoder 만 올린다.

- **depth 영상 조건 `depth_vid_cond`** (R482). 소스 영상 DA3 depth 를 voxel 화 없이 (동적 물체 포함) (2,T,h,w) [log1p(depth/norm_scale)·valid] 로 → 시공간 3D conv (t·y·x stride 2³) → (t,y,x) Fourier PE → Perceiver resampler (기본 64) → DiT 층마다 CA (0 초기화 gate, occ_grid 와 같은 주입). 캐시 `fit/bank/depth_vid_cache.py` (씬마다 `depth_vid36x64.npz`, 파일 추가만). 기본 false = 예전과 같은 모델·state_dict.

- **`run_corpus_export.py --no_framing_on_free`** (R469). build_bank_captions 에 그대로 넘긴다 — free preset 에 ", keeping X in view" 를 안 붙인다 (캡션 v2 규칙: look-at 만 공통 절). 안 주면 예전 그대로. `plan_from_bank_row` 는 v2 어휘에서 free_pedestal_* 을 pedestal_* 로 읽는다.

- **intent planner `terms="v2"`** (R468). `PresetVocab(terms=)` · `IntentPlanner(terms=)` · `moves_of(terms=)` — v2 는 캡션 v2 문구 (caption_presets_v2.json) 와 primitive 12 개 (crane = pedestal + look-at · push/pull arc = dolly + orbit), roll · v5f 0 행 preset · free_pedestal 을 어휘에서 뺌 (74 preset), 어휘 설명에 "(keep target centered)" 를 붙여 같은 primitive 의 free/look-at 을 가른다. `selftest_roundtrip(terms)` v2 74/74 일치. 기본 v1 = 예전과 같음.

- **캡션 v2 문구 `configs/caption_presets_v2.json` · `run_corpus_export.py --presets`** (R468·R469). 동사 = preset 용어 (dolly·truck·pedestal·orbit·pan·tilt·track; crane = pedestal + look-at, push/pull = dolly, arc = orbit + dolly), look-at preset 만 ", keeping X centered", free 는 subject 위치를 약속하지 않음 ("past X"). 생성 스크립트 tmp/r468/phrases.py. `--presets` 를 안 주면 예전 문구 그대로.

- **CLaTr 다중 텍스트 학습** (R466). `prepare_clatr_vista.py --prompts_file` (기본 prompts.json) · `--intents_file`/`--intent_levels` → `<name>__<lv>_caption.{json,npy}` 변형 문장 추가 (canonical 파일명 그대로). CLaTr `CaptionDataset` 의 `text_variants` (기본 없음 = 예전) — 학습 split 행마다 canonical·L1·L2·L3 중 하나를 고르고, 궤적 짝 검사는 원래 이름으로. hydra `+dataset.caption.text_variants=[caption,L1,L2,L3]`. 행 복제 대신 이 방식을 쓴 이유: train loader 가 shuffle 없이 정렬돼 같은 궤적이 한 배치에 붙으면 contrastive false negative.

- **`IntentPlanner(snap_on_fail=)` · `canonicalize(snap=)` · `PresetVocab.resolve(snap=)`** (R465). 재질의를 다 써도 어휘 밖 조합이면 마지막 응답을 가장 가까운 preset 으로 붙인다 ("track + a + b" 한 문자열 분해 · 어휘 밖 primitive 버림 · crane≈pedestal · orbit≈arc · zoom≈dolly 를 반쯤 맞은 것으로 순위). R461 에서 학습 캡션을 넣어도 38% 가 None 이던 문제. 기본 False = 예전처럼 None.

- **`fit_hole_ladder.py --shot_class_max`** (R464). 목표 shot 사다리의 가장 타이트한 칸 번호 상한 (`SHOT_CLASS_NAMES` 순서, 기본 -1 = 제한 없음 → 예전과 같음). 얼굴 뱅크 v5f 에서 3 (close_up) — R445 에서 extreme_close_up 목표는 90% shot_missed.

- **`intent_planner.synthesize_intents_levels(video=, video_fps=)`** (R462). L1/L2/L3 intent 합성에 영상 입력 추가 — subject mask (빨간 overlay) 를 덧씌운 원본 구간 영상을 VLM 에 2 fps 로 넣고, 프롬프트에 "강조된 subject 기준, 보이는 행동·물체·배경에 근거" 를 붙인다. `video=None` 이면 기존 텍스트 전용 합성과 같다.
- **`intent_planner.synthesize_intents_levels(context=)` · `context_block`** (R462). 캡션 단계에서 뽑아 둔 장면 맥락 (subject 서술·속성·행동, 다른 물체, camera pattern, framing) 을 프롬프트에 "What is known about the shot" 줄로 넣는다. `context=None` 이면 이전과 같다.

- **`intent_planner.synthesize_intents_levels`** · `jargon_hits` (SG3 · R459). canonical plan + 학습 캡션 → 수준별 intent {L1 움직임 직접 (용어 제한 없음), L2 강조·시선, L3 패턴·느낌}. L2·L3 는 촬영 전문 용어 금지 (나오면 최대 3 번 재생성, `jargon_ok=True` 면 허용 — 비교 변형). subject 지칭구에 든 단어는 금지어에서 뺀다. 기존 `synthesize_intents` 는 그대로.

- **`build_bank_captions.py --measured_end_shot`** · `run_corpus_export.py --measured_end_shot` (R457). look-at dolly/push/pull 계열 중 shot_target_class 가 없는 행에, 측정한 끝 shot (subject 면적 경계 0.02/0.08/0.25/0.5) 을 ", ending on a close-up" 처럼 붙인다. shot 변화 방향이 preset 과 맞을 때만 (in = 면적 ×1.1 이상, out = 1/1.1 이하). 기본 꺼짐 = 예전 문장 그대로.

- **`run_bank.py` config `post_cleanup`** (R451). 뱅크가 끝난 편 (OK · SKIP) 의 지정 파일을 지운다 (FAIL 은 재시도용으로 남김). `r442c_trumans_full.json` 에 `["cloud.npz", ".cloud_s115"]` — TRUMANS cloud 는 편당 ~760 MB 라 10-20 /data1 이 가득 찼다. 마커도 같이 지워야 재실행 때 cloud 가 다시 돈다. 키가 없으면 예전과 같음.

- **`fit/caption/trumans_action_gt.py`** (R448). TRUMANS board 클립 창마다 action 필드 (`out_trumans/<video>/action_gt.json`): TRUMANS 라벨 동사 + GT 로 움직인 물체 (Object_pose) + GT 손 (손목 근접), 라벨 밖 물체 움직임, 골반 궤적의 walk/sit/stand. `caption_action` 한 문장.
- **`trumans_to_recon.py --allow_unlabeled`** · **`run_board_sources.py --allow_unlabeled --recordings`** (R446). board 모드에서 blend 프레임 수와 맞는 sequence 나 Actions 파일이 없으면 라벨 없는 action 1 개로 진행 (기본 꺼짐 = 예전처럼 중단). `--recordings` 는 render 를 그 recording 만.

- **TRUMANS 전량 bank config** `configs/bank/r442c_trumans_full.json` (r442b + d271 앞단 graph/cloud/mesh, R442).

- **`molmo2_track_lift.py --pairs_file`** (R445). 줄마다 `<video>__<anchor>` — 그 (영상, anchor) 쌍만 point·comb·lift 한다 (같은 영상의 다른 anchor 는 건너뜀). 기본 빈 값 = 예전과 같음.

- **`--variant_whitelist`** (`fit/convert/vista4d_bank_to_dl3dv.py` · `exec/run_corpus_export.py`, R443). json `{video: [variant_id…]}` 에 적힌 변이만 내보낸다 (filter_variants ⑤, 히스토그램 키 `not_whitelisted`). 품질·preset 계열 균형으로 바깥에서 고른 v5f 50k 코퍼스용. 기본 None 이면 예전과 비트 동일.
- **TRUMANS pilot bank config** `configs/bank/r442_trumans_v5f_pilot.json` (v5f 방법론 이식: cloud npz · collision both · anchor 사람만 · blend_gt_z_up · route --anchor_seg_root TRUMANS) · `r442b_trumans_meshmargin.json` (collision mesh · mesh_margin_frac 0.02 · framing 하한 전 preset) (R442).

- **`dpo_hinge_keep`** (train_latent_cam_dm, 기본 false). true 면 collision hinge 를 `dpo_sft_w` 배에서 빼서 그대로 더한다 — sft 0 인 hard DPO 에서도 hinge 가 걸린다 (A3 · R440). false 면 예전과 비트 동일.

- **`pred_collision_rate.knn_scene(keep=)`** (R439). esdf_scene 과 같은 규약으로 프레임별 판정 (`coll`/`free`/`unk`) 을 `keep["cls"][(arm, entry)]` 에 남긴다. 기본 None 이라 기존 CLI 출력은 그대로.
- dataset `fit/bank/molmo2_track_lift.py --desc_template · --label_override` (R434): 부위 지정 — Molmo2 질문 대상구 템플릿 (예 "the face of the {q}") · comb 의 SAM3 명사 (예 face). 기본값이면 예전 질문·명사 그대로, override 실행은 코퍼스 sam3_text_label 을 채우지 않는다.
- latentcam Diffusion-DPO (R430, `dpo_beta` 기본 0): `dpo_pairs_file` (npz names · w2c (P,2,T,4,4) [선호, 비선호] world OpenCV) 의 쌍을 GT frame-s 기준 cam_param 으로 (dataset), 같은 (τ, ε) 에서 θ·ref 속도 오차로 −log σ(−β[(e_θw − e_rw) − (e_θl − e_rl)]) (flow matching). ref = load_ckpt_path 가중치 고정 사본, GT 손실은 `dpo_sft_w` 배. wandb train/dpo_loss·dpo_acc·dpo_margin. 쌍이 없는 세그먼트는 키 없음.
- latentcam `collision_loss_w` (R429, 기본 0): ESDF hinge 보조 손실 (objective=fm) — x0 = x_t − τ·v → frozen VAE decode → 카메라 중심의 정적 거리장 (`occ_edt_file`, frame-s / norm_scale) 에 relu(margin − d)²/margin², τ < `collision_loss_max_tau` 만. wandb train/col_loss·col_frac. dataset `_occ_edt` (분모 다르면 좌표·거리 재조정). 단위 시험: GT 충돌 세그먼트 loss 0.905 · 충돌 없음 0.
- latentcam `train_only_prefixes` (R429, 기본 null): 이 접두사 파라미터만 학습 (예 ['occ_'] = 점유 격자 adapter 만, load_ckpt_path warm-start 와 같이). optimizer 는 requires_grad 파라미터만.
- dataset `occ_grid_cache.py --edt_name` (R429): 세그먼트별 정적 거리장 64³ (uint8, [0,0.5] / norm_scale) 를 같은 패스로 굽는다.
- latentcam `occ_grid_cond` (R425, 기본 false): 3D 점유 격자 조건 — `<scene>/da3/occ_grid32.npz` (세그먼트별 (3,32,32,32) uint8 [occupied·free·unknown 비율], target frame-s 카메라 좌표 / norm_scale 의 [-2,2]³, track3d 와 같은 게이지) 를 3D conv 로 8³=512 토큰 + 칸 중심 Fourier PE → (선택) `occ_grid_resampler K` Perceiver resampler (학습 query K 개, 2 층) 로 고정 K 토큰 → DiT 층마다 cross-attention (0 초기화 gate). `occ_grid_dropout` (기본 0.1) 표본 단위 '전부 unknown'. dataset 은 학습 분모가 굽는 시점 avg_scale 과 다르면 좌표 비율로 재샘플. 켰는데 모듈이 없으면 기동 assert. 실측: 512 토큰 중 known 정보가 있는 토큰 중앙 30 · 95% 71 → K=64.
- dataset `fit/bank/occ_grid_cache.py` (R425): 씬마다 `eval/pred_collision_rate.esdf_scene` (R374 평가 정의) 한 번으로 ① GT 세그먼트 ESDF 판정 (`judge/<video>.json`) ② 위 점유 격자를 굽는다 (코퍼스에 파일 추가만). `--stage summary` 는 ESDF 충돌 프레임 ≥ `--drop_min_frames` (기본 3) 세그먼트를 뺀 정제 seg list.
- latentcam `models/intent_planner.py` R417: ① preset 집합 교체 `--preset_set v1|v3|v5f|all|<json>` — `--build_preset_set` 이 코퍼스 prompts.json (또는 뱅크 bank.csv solved) 을 세서 `main/conf/preset_sets/<이름>.json` (v1 21 · v3 78 · v5f 74 (뱅크 진행 중 잠정)), end shot 절은 그 코퍼스 캡션에 있던 preset 만 ② Trajectory Generator 피드백 루프 (`feasibility` callable + `max_replans`, 기본 0 = 꺼짐) — fit status/binding 을 VLM 이 읽는 실패 사유·제안 문장으로 (`explain_failure`), 실패 preset 은 재계획에서 제외 ③ 사용자 pointing 자리 `target_point` (subject_query mode point, 기본 vote) ④ `framing_on_free` (기본 False: look-at 없는 preset 은 framing 절 뺌) ⑤ 영상 통째 입력 (`video_mode video`, fps 2.0 기본, `frames` 는 예전 8 장) ⑥ `synthesize_intents` (plan → intent 합성) · `roundtrip_score`.
- dataset `lbm/vlm.py` `chat_json(video=, video_fps=, video_mm=)` (R417): `chat` 의 영상 입력을 재질의 경로에도 넘긴다. 안 주면 payload 동일.
- latentcam `models/intent_planner.py` (R416): Intent Planner — (video, high-level intent) → `CanonicalCameraPlan` (primitive moves · track · aim · target noun/지칭구 · end shot · magnitude/speed) → 카메라 모델 조건 (`camera_text` = 뱅크 캡션 빌더 `caption_of`+`nl_prompt` 그대로, `molmo2_text` = `Track {target_text}.`, `subject_query` = SAM3 noun + Molmo2 vote 질의). VLM (Qwen3-VL, `lbm.vlm.VLMClient`) 은 preset 이름이 아니라 primitive 조합을 고르고, 이름은 `PresetVocab.resolve` 가 결정론적으로 푼다 (dolly + orbit → push_in/pull_out_arc). 위반은 `chat_json` 재질의. `--selftest`: 81 preset 왕복 문장 불일치 0. 카메라 모델 배선은 아직 없음.
- latentcam `subject_pts_feat` (R369, 기본 false): subject 점마다 소스 2D 위치 (world 점 → `pose.npz` 투영, 0~1) 를 subject_pts 에 더해 (T,N,6), 모델이 그 위치에서 Molmo2 video 토큰 (프레임당 8x8, 칸 중심 (r+1)/9) 을 bilinear 샘플 → LayerNorm → 128 로 점 xyz 특징에 더한다 (video_latent_dim>0 필요). 투영 uv 와 CoTracker 원 좌표 차 중앙 0.0008~0.0054 (정규화).
- dataset `fit/bank/cloud_mesh_grid.py` (R405): in-the-wild 영상 (DynPose · Vista) 의 4D 점구름으로 TRUMANS `mesh_grid.npz` 와 같은 형식 (정적 EDT · 소스 카메라 도달 free 성분 · 프레임별 동적 EDT, recon world) 을 굽는다 — fit `--collision_source both --mesh_margin_frac 0.02` 로 ESDF 충돌 게이트를 코드 변경 없이 켠다. 정적 점은 동적 mask 경계 (dilate 7px) · depth 불연속 점 제외.
- latentcam `scripts/eval_testset.py --drop-box` (R392): subject_box_cond 모델에서 3D OBB 9 채널 (track 조건 맨 뒤) 만 0 으로 추론 (track3d 는 그대로).
- dataset `fit/bank/molmo2_track_lift.py --stage sam3box` (R391): mask (기본 vote, `--sam3_out`) + depth → 프레임별 중력 정렬 OBB (`scene_graph.obb.fit_obb`) → `subject_box.npz` 와 같은 형식의 반축 벡터 (`--box_name`, 기본 `subject_box_vote.npz`). yaw 는 2·yaw unwrap + 5 프레임 median, 크기는 프레임 median. v1 test 1,017 편 생성 (error 0).
- dataset `fit/bank/molmo2_track_lift.py` (R386): `--videos_file` (이 목록의 영상만, 전 stage) · comb `--save_cands` (noun 후보 masklet 전부를 `<comb_out>_cands/` 에 저장 — 규칙을 바꿔도 SAM3 재실행 없이 다시 고르기용).
- `viz/subject_find_overlay.py --select sam3_limits` (R381): SAM3 text 방식 한계 사례 (noun 후보 여러 개인데 최고 점수가 다른 물체 · 설명 문장 검출 없음 · 설명 문장이 다른 물체, Molmo2 점은 맞힌 쌍) 를 범주별 영상으로. tile 높이를 짝수로 (1 행 영상 인코딩 실패 수정).
- latentcam (R382) subject 점 대안: `subject_pts_stats` (프레임별 점 요약 13 ch 평균·표준편차·최소·최대 xyz·유효 비율을 cam 토큰에 바로 concat), `subject_pts_fourier K` (점 xyz Fourier 특징), `subject_pts_mode: traj` (점마다 49 프레임 궤적을 토큰 1 개로 → cross-attention). `eval_testset.py --subject-pts-probe drop|shuffle` (점 조건 민감도).
- (R380) `eval/subject_find_compare.py --stage cands` — 앞 `--cands_n` 쌍의 SAM3 noun · 설명 문장 **후보 masklet 전부** 저장 (선정 규칙 없이). `viz/subject_find_overlay.py` 에 `cands_label` · `cands_desc` 열 (후보마다 색·번호·평균 점수·등장 프레임 수), `--show_iou 0` (정답률 표시 끔), `--first N` (pairs.json 앞 N 쌍 그대로).
- latentcam `subject_pts_mode` (R379, 기본 `pool` = 기존): `xattn` 이면 subject 점을 프레임 pooling 해 cam 토큰에 concat 하지 않고, 점마다 토큰 (latent 13 프레임 × N 점 + 프레임 PE, 무효 점 key padding, 항상 유효한 null 토큰) 으로 DiT 층마다 cross-attention (0 초기화 gate) 해서 읽는다. `subject_pts_train_select` (기본 `random` = 기존): `fixed` 면 학습도 평가와 같은 (씬, 세그먼트) 고정 seed 점.
- `dataset/eval/subject_find_compare.py` (R376): text → subject 찾기 방법 비교 (SAM3 noun · SAM3 설명 문장 · Molmo2 점→SAM3 · noun 후보 + Molmo2 점 규칙 earliest/vote/conf/wsum) 를 `--rank` (route `anchor_ids` 순위, 0 = 옛 무작위) · `--n` · `--splits` · `--desc_n` 로 쌍을 골라 gen/sam3/eval stage 로 돈다. R345·R355·R371·R373 일회성 실험을 승격.
- `dataset/viz/subject_find_overlay.py` (R375): text → subject 찾기 방법별 mask overlay 비교 영상 (GT | SAM3 noun | SAM3 설명 문장 | Molmo2 점→SAM3 | noun+점 규칙 earliest·vote·conf·wsum). 실험 폴더 `sam3/<method>/` · `gen/` · eval json 을 받아 `--select rules_differ|desc_fail|method_differ|all_ok|random` 으로 쌍을 고르고 `--methods` 로 열, `--tag` 로 출력 이름을 정한다. R356·R360 일회성 viz 를 일반화.
- **충돌률 `--method knn`** (R374) — esdf 와 같은 3 상태 판정을 격자 없이: 정적 점 KD-tree 최근접 < r, 질의 카메라를 소스 프레임에 투영해 free 판정. `--knn_down 1` (기본) 은 점을 vox_frac·S 격자 대표점으로 줄여 KD-tree (원본 ~6M 점 build 가 병목). `esdf_scene(keep=)` 시각화 hook, `dataset/viz/viz_collision_esdf.py` (씬별 소스 프레임 · 위/옆 단면 occupied/free/unknown · arm 별 프레임 판정 띠).
- **예측 궤적 충돌률 ESDF voxel 판정** (R374). `dataset/eval/pred_collision_rate.py --method esdf` — 소스 depth 4D 점구름을 voxel (기본 0.02·S, 축 최대 256) 로 올려 정적 점 voxel = occupied · ESDF (scipy EDT) < 카메라 반경 0.02·S 면 정적 충돌, 동적 점은 시간 맞춘 한 프레임 KD-tree, 소스 카메라 ray 가 지난 공간 (voxel 중심 투영 `z < depth − voxel`, GPU) + 소스 카메라 경로 (`--src_free`) = free, 나머지 = unknown. 출력에 `unk_frac` · `unk_any` · `free_frac` · `coll_or_unk_any` 추가. 기본 `--method shell` 은 기존 depth shell 그대로 (val 용).
- dataset `fit/bank/molmo2_track_lift.py --stage comb` (R366): text + point 결합 — 라벨 명사 text 로 SAM3 후보 masklet 전부 → Molmo2 점 (`--point_mode`, 기본 pointK8f) 이 든 masklet 선택, 없으면 점 단독 결과로 대체 (`how` 기록) → `<corpus>/molmo2_comb/`. 같은 전파로 `sam3_text_label/` 도 비어 있으면 채움. 3D 는 `sam3lift --sam3_out …/molmo2_comb --sam3_name target_track_comb.npz`.
- subject 3D 점 집합 조건 (R361, 기본 off): dataset `fit/bank/export_subject_points.py` → `<scene>/da3/subject_points.npz` (anchor 별 (T,600,3) world + valid; dyn = R246 CoTracker3 2D track 을 visible·GT mask 안·depth 유효 점만 depth lift, stat = GT mask 픽셀 샘플). latentcam `subject_pts_cond` (+ `subject_pts_n` 64 · `subject_pts_dim` 64 · `subject_pts_file`): (T,N,4) frame-s 카메라 / norm_scale, train 무작위 · eval 보인 프레임 많은 순 N 점, 모자라면 0+valid 0 → 모델 set encoder (점별 MLP → valid 점 max·mean pool) 프레임 특징을 cam latent 에 concat.
- dataset `fit/bank/molmo2_track_lift.py --stage sam3text` (R358): SAM3 text 단독 비교군 — 라벨 명사 (scene graph label) · 설명 문장 (instance_desc) 을 0 프레임 text prompt 로 전파, obj 여럿이면 평균 점수 × 등장 프레임 최고 → `<corpus>/sam3_text_{label,desc}/`. 영상당 세션 1개.
- 3D bounding box 조건 (R357, 기본 off): dataset `fit/bank/export_target_track.py --box [--only_box]` → `<scene>/da3/subject_box.npz` (scene graph 중력 정렬 OBB 반축 벡터 world (V,T,3,3) = R_g(track.yaw_t)·diag(extent/2) 에 T_wg 회전·스케일). latentcam `subject_box_cond` (+ `subject_box_file`): frame-s 카메라 좌표 / norm_scale 의 반축 9 채널을 track concat 조건 뒤에 (cond_dim +9, 중심은 target_track).
- dataset E.T. subject 궤적 출처 옵션 (R352, 기본 = 예전): `exec/run_director_batch.py --track_file <npz>` (released E.T. 의 char 를 scene_graph center_smooth 대신 `<scene>/da3/<npz>` world 궤적 → T_wg 역변환으로 G frame; 높이 게이지는 node OBB) · `exec/prepare_et_d200.py --corpus --track_file --split_list --clip_seq_dir` (다른 코퍼스·track 파일·seg 목록·캡션 CLIP seq 폴더로 convert/infer/evaldir; convert 의 char·p0 와 evaldir 의 p0 가 같은 track 파일).
- dataset `fit/bank/molmo2_track_lift.py` stage `point` · `sam3` · `sam3lift` (R346): Molmo2 "Point to the {desc}." (`--point_mode` point1 · pointK · pointK8, `*f` = 점 1개 프레임 중 가장 이른 점) → SAM3 point prompt 양방향 전파 (영상당 세션 1개, anchor 마다 reset; point 단독은 빈 cached_frame_outputs 로 우회) → mask depth 점 축별 10~90% 중점 = subject 3D 중심 → `<scene>/da3/target_track_m2sam3.npz` (target_track.npz 와 같은 형식). R345b 100 쌍: pointK8f GT mask IoU 0.700, 3D 오차 평균 0.252 (점 lift 0.338).
- dataset `fit/bank/molmo2_track_lift.py --stage check` (R343): Molmo2 원 2D 점 (main id) 이 anchor 의 SAM3 mask (scene graph track_id + merged_from) 안에 있는 프레임 비율 (정확 · 반경 `--near_frac`), lift depth 와 mask 안 depth 중앙값의 상대차 → `--check_out` CSV + dyn/stat 요약. CPU 만, gen·lift 동작 변화 없음.
- latentcam `scripts/viz/track_overlay_molmo2_vs_gt.py` (R342): Molmo2 lift 궤적 (target_track_molmo2.npz) 과 GT subject 궤적 (target_track.npz) 을 소스 카메라 (pose.npz) 로 프레임별 정사영해 소스 영상에 겹친 그리드 mp4 (Molmo2 원 2D 점 · 화면 거리 px · 3D 상대 오차 표기). `--items video:seg,...` 또는 `--sample_bins` 로 test 에서 오차 구간별 추출. CPU 만.
- latentcam val 지표 `val/readout_aux_obb` · `val/readout_aux_dist` (R340): readout 이 subject xyz 를 예측하는 run (aux_dim 3, 타깃 target_track) 에서 val sample() 마지막 forward 의 예측과 GT track3d 의 3D MSE (학습 `train/readout_aux_obb` 와 같은 정의) 와 평균 L2 거리 (norm_scale 단위). readout xyz head 가 없는 run 은 안 찍힘 (손실·학습 동작 변화 없음).
- latentcam `subject_uv_cond` (R333, 기본 false = 예전): Molmo2 tracking 원본 영상 2D 점 (`<corpus>/molmo2_track/*.json`, anchor·valid 는 `target_track_molmo2.npz` 와 같은 rule) 을 (T,4) [u-0.5, v-0.5, observed, valid] 로 track 조건 concat 경로에 (cond_dim +4, depth 로 3D 로 안 올린 판). `subject_uv_dir` (null = <corpus>/molmo2_track).
- latentcam `scene_points_cond` (R328, 기본 false = 예전): 토큰별 depth 3D 점 (`video_points` 49×64, `video_patch_points: true` 필요) 을 xyz Fourier → MLP + 칸 임베딩 + 프레임 PE 로 임베딩해 geo CA key/value 에 srccam 토큰과 함께 붙인다 (장면 기하용). readout 쪽 geo_mask 는 원본 유지.
- dataset `fit/bank/molmo2_track_lift.py` (R326): subject 3D 궤적을 Molmo2 tracking ("Track the {instance 설명}.", batch 생성) + DA3 depth lifting 으로 만들어 `<scene>/da3/target_track_molmo2.npz` (target_track.npz 와 같은 키·형식). latentcam `target_track_file` 옵션 (기본 target_track.npz = 예전). `--prep_workers` (기본 3): 프레임 읽기·processor 를 DataLoader worker 로 미리 (GPU 생성과 겹침, 6 프로세스 합 514 → 1438 쌍/h).
- dataset `exec/prepare_et_d200.py --stage infer,evaldir` (R314): 우리 데이터로 학습한 E.T. ckpt 를 test 전량 추론 → 역변환 c2w = [W·R' | W·t' + p0] → 우리 eval dir (arm raw · anc). `run_director_vista.build_diffuser(overrides=())` 인자 추가 (기본 = 예전), ckpt 경로에 `=` 가 있으면 hydra override 를 따옴표로.
- latentcam readout 옵션 (R315, 기본 off): `peav_readout_cond_grad` (readout 예측 조건을 detach 하지 않아 diffusion 손실이 readout 으로), `peav_readout_init_ckpt` + `peav_readout_freeze` (다른 run 의 `readout.*` 로 시작·고정 — 2단계 학습).
- latentcam readout 강화 옵션 (R312, 전부 기본 0 = 예전): `peav_readout_heat_w` (soft-argmax 가중치를 subject bbox 토큰 분포로 CE 지도, dataset 키 `readout_bbox`), `peav_readout_smooth_w` (readout 3D 궤적 가속도 페널티), `peav_readout_cond_noise` (학습 때 GT 위치 조건에 가우시안 — conditioning augmentation), `peav_readout_cond_p_ramp` (readout 예측 조건 비율 epoch 램프 — scheduled sampling).
- dataset `fit_hole_ladder.py --drop_shot_miss` (R305, 기본 off): shot class 사다리에서 끝 프레임 면적이 목표 칸 밖이면 status `shot_missed` (원래 값은 `status_before_shot_miss`, 행은 통계용으로 남김). `run_corpus_export --drop_status` 기본에 `shot_missed` 추가. v5f config 에 켬.
- dataset route `--per_anchor_presets N` (R304, 기본 0 = 예전 앞 2개 × 예산/2): anchor 2개 이상이면 살아남은 전 anchor 에 N 개씩 (`sample_camera_bank.plan_variants`, route JSON `per_anchor`). v5f config: `--max_anchors 0` (dyn ≤ 2 · stat ≤ 2 따로) + `--per_anchor_presets 2`, route/τ 새 폴더 (`preset_route_v5f.json`, `bank_v5f`).
- dataset arc 최소각 (R302): `--push_k_min` — push_mode subject + `--push_arc_deg` 일 때 arc preset 은 arc 를 push_arc_deg → 중간 → push_arc_min_deg 로 줄여 가며 풀고, 거리 변화 |k| ≥ push_k_min 인 첫 각도를 쓴다 (행에 `push_arc_deg`·`push_k`, emit 은 행 값 우선). 최소각에서도 미달이면 status `arc_blocked` (`run_corpus_export --drop_status` 기본에 추가). `pull_out_arc_*` 도 subject 극좌표 (거리 증가) 로.
- dataset `--push_arc_min_deg D` (R301, 기본 0): `--push_arc_deg` 고정 각도 모드의 상한을 sweep/2 대신 관측 방위각 폭 × orbit_span_frac 으로 두고 그 바닥을 D 도로. v5f config arc 40° / 바닥 20°.
- dataset `--push_arc_deg D` (R300, fit_hole_ladder/emit_bank/build_poses, 기본 0 = push 비례식 θ = push_arc_ratio·k): push_mode subject 의 arc 각도를 push 양과 무관하게 고정 (관측 방위각 폭 sweep/2 clamp 유지). v5f config 22.5°.
- dataset `fit_hole_ladder.py --shot_iterations N` (R299, 기본 0 = --iterations 그대로): shot 사다리 (bracket 없이 [0.005, 3] 전 구간) 전용 이분 횟수 — 4회면 눈금이 0.19 라 게이트 여유를 크게 남겼다.
- dataset `--push_mode subject` (R298, fit_hole_ladder/emit_bank/build_poses, 기본 frame0 = 예전): push_in_arc_* / dolly_in_look_at (track_ 포함) 위치를 subject 중심 극좌표로 — 거리 축소율 k 가 주동작, 옆 회전 θ = min(push_arc_ratio·k, sweep/2), τ 손잡이 의미는 그대로 (k 를 τ 에 맞춰 이분법). 기준점: track_ = 매 프레임 c(f), 비-track = 움직이는 노드는 마지막 프레임, 정지 노드는 `best_frame`. bank.json fixed 에 `push_mode`·`push_arc_ratio` 를 적어 emit 이 재현. config `v5f_pushsubject.json`.
- latentcam `scripts/eval_testset.py --dump-readout` (R297, 기본 off): soft-argmax readout 가중치 (B,49,64)·3D 예측·GT target_track 을 `<out>/readout_dump.npz` 로 저장 — readout 이 Molmo2 토큰 중 실제 subject 를 보는지 재는 용도.
- dataset `fit_hole_ladder.py --trace_probes <dir>` (R295, 기본 off): 변이마다 이분법이 시험한 손잡이 전부의 궤적·게이트 값 (approach_gap·obb_slack·subject_area_last 등)·물리 판정을 `<dir>/<variant>.npz` 로 남긴다 — `*_limited` 가 어느 손잡이에서 무엇에 막혔는지 그 궤적으로 렌더하기 위한 진단.
- latentcam `peav_readout_geo` (R292, 기본 false): readout cross-attn 의 key/value 에 DA3 geo 토큰 (geo CA 와 같은 `geo_proj` 투영·PE·mask) 을 Molmo2 [text|video] 뒤에 이어 붙인다. soft-argmax 가중평균 대상은 그대로 Molmo2 video 토큰의 3D 점. geo_emb 없는 arm 에서는 무시.

- **train_camera_vae `--train_on_test`** (R194): 1 이면 test 창 (Dataset-v1 test + joint worldtraj test) 도 학습에 넣는다 — in-distribution 용량 비교용. 기본 0 = 옛 동작.
- **CameraVAE `pad_mode="rep_enc"`** (R187): encoder 는 replicate, decoder 는 zeros — rep_lin 의 끝 프레임 속도 감쇠가 decoder 쪽 replicate 에서 오는지 가르는 변형. state_dict 호환, `train_camera_vae --pad_mode rep_enc`.
- **experiment `dynpose_d200_molmo2_l21_srccam_noprobe`** (R186): Dataset-v1 전체 (D262) 에 Molmo2 noprobe text 캐시 `molmo2_cache/prefill_noprobe.pt` 만 바꾼 판 (prefill.pt 와 같은 인자, probe '').
- **train_camera_vae `--use_v1`** (R181): 0 이면 학습 창에서 Dataset-v1 train 을 뺀다 (worldtraj 만 — old VAE 와 같은 데이터 출처 비교용). test 는 그대로. 기본 1 = 옛 동작.
- **train_camera_vae `--grad_clip` · `--amp bf16`** (R179): CamVLA `train_vae.py` 레시피 (clip_grad_norm 0.5 · bf16 autocast) 재현용. 기본 0 / off = 옛 동작.
- **CameraVAE `pad_mode="extrap_in"`** (R178): 입력 궤적을 보는 encoder 첫 conv 만 끝 2프레임 직선 연장 padding (속도 보존), 나머지 encoder·decoder conv 는 zeros. replicate 가 첫·끝 프레임 속도를 ~0.5 배로 깎던 문제 (R176) 대안. state_dict 는 zeros/replicate 와 같은 키. `train_camera_vae --pad_mode extrap_in`.
- **train_camera_vae `--stage edge`** (R176): VAE roundtrip 의 프레임별 속도 비율 |v_pred|/|v_gt| (시작·끝 프레임 감쇠) — R169 인라인 측정 승격.
- **bank config d343_dynpose_v3_pool** (Dataset-v3): d185 tau/fit/emit 그대로 + route 를 preset 전량 풀 (`--slot_plan pool --pool_size 12 --free_moving pool`, anchor 풀 70 + free 8, 82 중 s_curve·zoom 2·legacy 제외) + d199 의 `--anchor_require_frame0`. 3편 smoke OK (bank 5/6/13 행).
- **train_camera_vae `--stage scale`** (R173): 이 VAE 로 diffusion 을 학습할 때의 vae_latent_scale (학습 데이터 mu std, encode_sample 용 z std 도) — R164·R166 인라인 계산을 승격. 학습 창 구성은 `train_windows(a)` 로 뽑아 stage_train 과 공유 (동작 동일, R166 s_base 0.410282 재현).
- **sam3_static_instances `--reverse`** (dsv3): shard 목록을 뒤에서부터 — 같은 shard 에 붙이는 보조 worker 용 (영상별 재확인과 함께 가운데에서 만난다). 기본 꺼짐.
- **sam3_static_instances: 영상마다 처리 직전에 `masks.npz` 재확인** (dsv3). `--skip_done` 이면 시작 때 만든 todo 중 그 사이 다른 worker 가 끝낸 영상을 건너뛴다 — shard 수를 늘려 다시 띄울 때 겹침 방지. 단일 worker 동작은 그대로.
- **plot_traj_curves `--jerk`** (R169): 위치 jerk |Δ³p|/reach · 회전 jerk |Δ²ω| 프레임별 곡선 열 2개 (traj_time_metrics 와 같은 정의, 범례에 평균).

### Changed
- latentcam `subject_pts_cond` 평가 시 점 고르기 (R363→R365): 기본 = 학습과 같은 무작위 N (`subject_pts_eval_select: random`, (씬, 세그먼트) 고정 seed 로 재현). 대안 `fps_best` (유효 점 최다 프레임 3D farthest point sampling, 보류) · `visible` (보인 프레임 많은 순, R361 처음 — 동률이 index 순이라 frame 0 query 점에 몰림). 학습은 그대로.
- dataset `fit/bank/molmo2_track_lift.py --stage comb` 기본 masklet 고르기를 **vote** 로 (R384, 사용자 10-14 "noun + 점 vote 를 기본 세팅으로"): 점이 1 개인 프레임의 Molmo2 점마다 그 점이 든 SAM3 noun masklet 에 한 표 → 최다 표. 출력 `<corpus>/molmo2_comb_vote`. 예전 동작은 `--comb_rule earliest` (출력 `molmo2_comb`).
- latentcam `peav_readout_cond_p` 기본값 0.0 → **1.0** (R325, 사용자 10-12 "cond_p 1.0 를 기본값으로"): 학습도 항상 readout 예측을 조건으로. readout 이 track3d xyz 를 예측하는 구성 (aux_dim 3 · target_track_dim>0 · aux_target target_track) 에서만 적용되도록 train/eval 에서 게이트 (그 밖은 0 = 예전). 예전 run 은 저장된 config 값 그대로.
- dataset config `v5f_pushsubject.json` (R308): `--shot_class_presets` 를 look_at preset 만 (`^(track_)?(dolly_in_look_at|push_in_arc_*|pull_out_arc_*)$`) — 예전 `dolly_in|push_in|dolly_out|pull_out` 이 `truck_left_dolly_in` 같은 free 조합까지 걸어 초반 34편 shot_missed 89/316. bank_dir `hole_bank_v5f2`.
- dataset `push_mode subject` (R299): track_ push 의 기준점을 매 프레임 c(f) 에서 c(0)→c(F−1) 순이동 직선으로 — subject 앞뒤 흔들림이 카메라 위치로 옮겨가지 않게 (조준은 실제 c(f) lock 그대로).

- **train_camera_vae 기본 레시피 = old VAE** (R138). `--lr_schedule constant|cosine`, `--wd`, `--recon l1|mse` 추가, 기본값을 SCVideo old VAE 학습과 같게 (lr 1e-4 constant · wd 0.01 · L1 · kl_w 1e-3). R125~R137 레시피는 `--lr 1e-3 --lr_schedule cosine --wd 0 --recon mse --kl_w 1e-4`.
- **최종 git 정리** (R15). 작업 트리의 미추적 항목을 **추적 / 닫기** 둘 중 하나로 전부 처리했다.
  - **`context.md` → `.claude/context.md`.** (R16) `CLAUDE.md` 의 "에이전트 전용 문서는 `.claude/` 에"
    규약을 따른다. 같은 절의 목록에도 `context.md` 를 넣었다.
  - **`video_generation/models/Vista4D/USAGE.md` → `video_generation/Vista4D_USAGE.md`.**
    원래 자리에서는 **git 이 추적하지 못했다** — `.gitignore:249` 가 `video_generation/models` 를
    막을 뿐 아니라 `Vista4D/` 가 자기 `.git` 을 가진 별도 저장소라 `!` 예외도 안 먹는다.
    재클론하면 사라지는 문서였다. 옮기면서 Vista4D 기준 상대경로 7곳을 저장소 기준으로 고쳤다.
  - **`.gitignore` 두 규칙 추가** — ① `camera_generation/latentcam/main/evaluate/CLaTr/tmp*/`
    (torch.distributed 가 rank 마다 만드는 scratch, 이름이 매번 다르다) ② `hc_*.txt` ·
    `hc2_*.txt` · `hcfull_*.txt` (screen hardcopy 덤프. `screen -X hardcopy` 는 **그 screen 의
    cwd** 에 떨어지므로 경로가 아니라 이름으로 닫는다).
  - **`camera_generation/dataset/summary.md` 헤더 수정** — 옛 헤더의 "`.gitignore:222
    camera_generation/models` 아래라 커밋에 안 들어간다" 는 **두 겹으로 틀렸다** (R7 이후 추적
    중이고, `.gitignore:222` 는 지금 `tools` 줄이다). 본문은 2026-08-23 스냅샷이라 그대로 두되,
    **`run_lbm_lite.py` 는 저장소에 없고** 현재 입구는 `camera_generation/dataset/exec/run_bank.py`
    (`graph→cloud→route→tau→fit→emit`) 라는 경고를 헤더에 박았다.
  - `.gitignore` 에 줄이 늘면서 밀린 `pipeline.md` 의 인용 2곳을 고쳤다 (`:251-260`→`:251-263`,
    `tmp/` 의 `:263`→`:273`). **`.gitignore:NN` 인용은 규칙을 추가할 때마다 밀린다.**
  - **못 고친 것** — `CLAUDE.md:183`·`:190` 의 "GPU 0~4" 두 줄. 같은 파일 `## Don't` 의 0~3 과
    충돌하는데 수정이 **R14·R15 두 번 다 classifier 에 막혔다.** `## Don't` 가 이긴다는 사실은
    `.claude/context.md` §2 에 적어 뒀다.

### Fixed

- **CameraVAE linear 업샘플 학습 9배 느림** (R140). nn.Upsample(linear) CUDA backward 대신 같은 값을 내는 `LinearUp2x` (step 86 → 10.6 ms, 기존 ckpt 그대로 로드).

### Added

- **cfg `val_traj_shape_metrics`** (R167). true 면 val 에 `val/jerk` (+`_gt`, `_ratio`; |Δ³| rot6d+trans) · `val/subject_screen_offset` (+`_gt`, `_err`; subject 를 각 프레임 카메라로 투영한 화면 중앙 거리) 로깅. subject 궤적은 모델 조건 키 `target_track` 과 다른 `val_subject_track` 으로 넘겨 네트워크 입력이 안 바뀐다. 기본 false, overfit experiment yaml 16개만 true. GT offset 이 traj_time_metrics 와 같음 확인 (of29a 0.0333 / of5ec 0.0289).
- **cfg `wandb_project`** (R165). 학습 wandb project 를 설정으로 (기본 camera-diffusion-training = 기존). overfit experiment yaml 16개 (dynpose_d200of{5ec,29a}*) 는 camera-overfit.
- **scripts/eval/plot_traj_curves.py** (R162). 궤적별 trans speed / rot speed / subject 화면 offset 곡선 (GT + arm 들). R113·R123·R131 의 인라인 코드를 승격, 정의는 traj_time_metrics 함수 재사용.
- **bank config d342_dynpose_cyclist_pushin_arc_strong** (R161). cyclist 2편 track_push_in_arc_right 를 sweep 90° · hole_ladder 0.35/0.50 로 (hole_bank_d342).
- **bank config d341_dynpose_keep_pushin_arc** (R160). d340 과 같은 패턴으로 track_push_in_arc_left/right 만 굽는다 (hole_bank_d341).
- **bank config d340_dynpose_keep_track4** (R159). d185 상속 + `--route_preset_override` 로 track_orbit_left/right · track_pull_out_arc_left/right 4종만 굽는 config (산출 hole_bank_d340, 기존 hole_bank_d185 무관).
- **cfg `vae_encode_sample`** (R158). true 면 확산 학습·val 에서 `camera_vae.encode_sample` (z = mu + σ·ε, CamVLA 방식) 으로 latent 를 만든다. 기본 false = mu (기존).
- **run_video_filter `--judge v8obj`** (R157). v8 과 같고 `subject_is_object` (main subject 가 사물) 만 탈락 사유에서 뺀 판정 — 산출 `*_v8obj.*`, v8 산출은 그대로.
- **VAE eval 후반 가속도 오차** (R150). `acc_late` (|Δ²c_pred − Δ²c_gt| / reach) · `acc_rot_late` (|Δω_pred − Δω_gt| deg) — 마지막 1/3 구간.
- **VAE eval 4프레임 주기 성분 · zigzag 비율** (R149). `p4_frac` (위치 속도 파워 중 f≈0.25·0.5 bin = 업샘플 톱니 세기), `zigzag_t` / `zigzag_r` (이웃 가속도가 반대 방향인 프레임 비율).
- **VAE eval GT 잡음에 덜 휘둘리는 지표** (R148). `late_errors()` 에 `FDE_sGT` / `vel_late_sGT` (GT 를 이동평균 k=9 한 궤적과 비교), `jerk_t_late` / `jerk_r_late` (복원 자체 후반 떨림), `hf_frac` (위치 속도 파워 중 f>0.2 비율) 추가.
- **train_camera_vae eval 후반 구간 오차** (R147). `late_errors()` — cam_param 공간에서 traj_time_metrics 와 같은 정의로 `late` / `FDE` / `rpe1_late` / `rpe1_rot_late` (정지 창 제외, `n_moving`). eval JSON 이 없는 worldtraj test 에도 같은 식으로 잰다.
- **traj_time_metrics 마지막 구간 지표** (R145). `--corpus_root` 때 `jerk_t_late` / `jerk_r_late` / `fr_jit_late` / `fr_err_late` (+ `_gt`, `_ratio`), `rpe1_late` / `rpe1_rot_late` (마지막 구간 프레임간 이동·회전 = 속도 오차) — 마지막 1/3 구간 (기존 `late` ATE 와 같은 프레임) 에서만 잰 떨림·구도 오차. 기존 키는 그대로.
- **train_camera_vae `--resume_dir` / `--start_epoch` / `--wandb_id`** (R139). 같은 결과 폴더·같은 wandb run 에서 이어 학습. 매 epoch `resume.pth` (model + optimizer + epoch) 저장 — 없을 때는 last.pth + 새 AdamW.
- **train_camera_vae eval 활성 채널 통계 + `--save_epochs`** (R138/R139). roundtrip 지표에 `n_active` (채널별 posterior σ<0.5) · `mu_std_active` (latent scale 후보) · `sigma_mean` · `z_sample_std` 추가. `--save_epochs 100,200` 으로 중간 스냅샷 `epochN.pth`.
- **CameraVAE `pad_mode` / `up_mode="subpixel"`** (R136). conv 시간 padding 을 zeros(기본, 기존 ckpt 와 수치 동일) | replicate(끝 프레임 복제) 로, 업샘플에 WAN 식 depth-to-time (conv 로 채널 2배 → 시간 2칸 interleave) 추가. cfg `vae_pad_mode`, `train_camera_vae --pad_mode`, eval spec `ckpt:up:pad[:pad_mode]`.
- **models/vae_causal.py** (R136, 보류). CamVLA `cam_causal_vae_v1` / WAN 식 인과 레이아웃 VAE (토큰 0 = 프레임 0, 왼쪽만 padding). 학습·eval 배선은 아직 없음 (사용자 지시 후 진행).
- **train_camera_vae `--stage split` / `--split joint`** (R135). worldtraj + Dataset-v1 합본을 0.9/0.1 로 나눈 `split_joint.json` (v1 은 기존 분할, worldtraj 는 클립 단위 · 같은 영상은 v1 쪽 split 을 따름 · 출처별 test 10%). 학습 기본이 joint 가 되어 val/* = 합본, val_wt/* · val_v1/* = 출처별로 기록. `--split legacy` 는 기존(R125~R132) 동작. eval 단계 `--eval_sets joint` 로 3가지 test 를 같이 잰다.
- **train_camera_vae `--wt_split test` / `--stage eval`** (R134). cache_wt 에 Dataset-v1 test 씬의 worldtraj cameras.json 만 뽑는 분기 (→ `wt_test.npz`, 기본 train 은 기존과 동일), eval 단계는 여러 VAE ckpt 를 v1_test / wt_test 에서 같은 val/* 정의로 roundtrip 측정.
- **scene graph `--prefer_person_label`** (R130). 사람+옷/신체부위 마스크가 병합된 dyn 노드의 대표 라벨이 옷이면 aliases 의 사람 명사로 올린다 (850c66dd 'jacket'→'man'). 기본 off = 예전 라벨. Dataset-v3 graph 용 config `configs/bank/d330_dynpose100k_graph_personlabel.json` (d182 + 이 플래그).
- **cfg `vae_up_mode` / `vae_pad_end`** (R126). 학습·eval_testset 이 VAE 를 이 옵션으로 만든다 (기본 nearest/0 = 기존 ckpt 구조).
- **VAE 구조 옵션** (R126). `models/vae_intr_large.CameraVAE(up_mode='nearest'|'linear', pad_end=0)` — decoder 업샘플 모드(파라미터 동일, state_dict 호환)와 입력 끝 복제 padding (pad_end 4 → 토큰 14, decode 뒤 crop). 기본값은 기존과 동일. `train_camera_vae.py --up_mode --pad_end`.
- **`latentcam/scripts/vae/train_camera_vae.py`** (R125). "카메라 VAE 를 우리 카메라로 학습/finetune 하면 roundtrip 떨림이 줄어드나" — 레포에 VAE 학습 코드가 없어 신설. Dataset-v1 cam_param(데이터셋 __getitem__ 그대로) + worldtraj cameras.json (`_target_out` 으로 같은 표현, 창 평균 이동거리 / Dataset-v1 중앙값으로 스케일) 캐시 → MSE + KL (+ 선택 jerk penalty `--smooth_w`), cosine LR, wandb `camera-vae` (train/lr 포함), val = Dataset-v1 test + overfit 두 영상 watch. `--include_test_scenes` 로 overfit 두 씬을 train 에 포함.
- **필터 v8: 1차 VLM 프롬프트 개정** (R121). `PROMPT_V8` = v1 SCHEMA 에 `face_closeup` 필드 + selfie 정의 보강(뒤에서 따라가거나 멀리서 관찰은 selfie 아님). `--vlm_prompt v1|v8`, `--vlm_dir` (결과 폴더, 기본 vlm/), `stage vlm` 도 `--api_bases` 여러 서버 round-robin. `--judge v8` = v6 규칙 + face_closeup 탈락 (VLM 결과는 vlm_v8/). v1·v6 판정 동일 확인 (3,000편).
- **필터 v7: 셀카·얼굴 클로즈업 2차 VLM 판정** (R119). `run_video_filter.py --stage vlm_selfie` (v6 통과분에 `PROMPT_SELFIE` 만 묻는다, `--api_bases` 여러 vLLM round-robin + `--workers`, `--scenes_file` 검사용) → `--judge v7` = v6 + `face_closeup` 탈락 (`selfie` 는 오탐이 있어 기록만 — `selfie_only_flag`). 결과 vlm_selfie/*.jsonl, *_v7.*.
- **VLM 지칭구 병렬 요청** (R118). `fit/graph/describe_instances_vlm.py --workers N` (영상 단위 ThreadPool, 한 영상 안 노드는 순서 유지; 기본 1 = 예전 순차), `exec/run_corpus_export.py --desc_workers N` 로 전달. 4편 실측 13.9 s (workers 4).
- **`latentcam/scripts/data/build_molmo2_track_override.py`** (R117). "Molmo2 text override(`Track {target_text}.`) 맵을 코퍼스에서 어떻게 만드나" — 지워진 tmp/d194/build_override.py 복원. target_text 없으면 concise (`--empty_target`), `--check` 로 기존 맵 대조 (d200: aim≠free 37,645행 100% 일치, free 13,777행은 기존 맵이 빈 문자열 — 캐시가 --free_mode zero 로 0 처리).
- **D284 후속용 분기 2개** (R117). `exec/run_corpus_export.py --videos_file` (그 씬만 desc/captions — 없으면 cine_out 전체 = 예전), `fit/caption/make_holdout_split.py --keep_split_from <root>/seg_list_dynpose_s91 [--keep_val_scenes]` (기존 train/test/val 씬 유지, 신규 씬만 층화 9:1; d200 자체에 돌리면 9,152/1,017/102 그대로 재현 확인).
- **zero terminal SNR 옵션** (R114). `rescale_betas_zero_snr: false`, `timestep_spacing: leading` (`main/conf/config.yaml`, 기본 = 예전 스케줄). true 면 `prediction_type: v_prediction` 필수(loss 대상 `get_velocity`), trailing 이면 추론이 t=999(순수 noise)부터. 학습·`eval_testset` 스케줄러 둘 다 반영. 실험 `dynpose_d200{of5ec,of29a}_noprobe_zsnr`.
- **`eval_testset.py --init_gt_t 0` = 순수 VAE roundtrip** (R113). 모델을 한 번도 안 부르고 GT latent 를 그대로 decode (`sample()` 의 t_start<=0 분기, eps/fm 둘 다). 떨림이 VAE 에서 오는지 진단용.
- **`traj_time_metrics.py --corpus_root`: smoothness·framing 지표** (R112). "궤적이 매끄러운가 / framing 이 일정한가" — 위치 jerk(|Δ³p|/reach)·회전 jerk(|Δ²ω|), subject 3D 중심(scene_graph track)을 카메라에 투영한 화면 위치의 GT 대비 오차·흔들림(|Δ²uv|)·표준편차·거리 떨림(|Δ² log z|)·중앙 0.8 상자 비율. 인자를 안 주면 예전 출력 그대로.
- **`train/lr` 항상 기록** (R108). lr schedule 과 무관하게 매 step wandb 에 남긴다 (constant 이면 cfg.lr 그대로).
- **`objective: fm` (flow matching, 선형 경로)** (R109). `objective: eps|fm`, `fm_t_sampling: logit_normal|uniform` (`main/conf/config.yaml`, 기본 eps = 예전 동작). fm 은 x_tau=(1-tau)x0+tau eps 에서 v=eps-x0 회귀 (reflow 없음 — rectified flow 아님), `sample()` 은 Euler tau 1→0 (`--init_gt_t` 도 지원). aim_loss 와는 아직 같이 못 쓴다. 실험 `dynpose_d200{of5ec,of29a}_noprobe_fm`.
- **LR schedule 옵션** (R108). `lr_schedule: constant|cosine`, `lr_warmup_steps`, `lr_min_ratio` (`main/conf/config.yaml`, 기본 constant = 예전 동작). cosine 은 global_step 만으로 계산해 resume 에도 같은 곡선, wandb `train/lr` 기록. 실험 `dynpose_d200{of5ec,of29a}_noprobe_cos` (warmup 500, min 0.05, 100 epoch).
- **`latentcam/scripts/eval/traj_time_metrics.py`** (R106). "궤적 후반이 GT 에서 더 벌어지는가" — eval 폴더(arm 여러 개)의 ref/pred json 으로 프레임별 ATE/rot/RPE 곡선과 ADE·FDE·구간(early/mid/late)·drift_slope·RPE1/8 을 낸다. reach(GT 최대 이동) 정규화, 정지 궤적(reach < 중앙값×0.1)은 따로.
- **`exec/run_video_filter.py --judge v6`** (R104). v1 − synthetic, VLM 통과 영상만 SAM3 **49 프레임**(`sam3_k49/`), main 고르기 v2(평균 면적), 잘림·가림 판정 대신 main 이 안 보이는 프레임이 있으면 `subject_missing_frame`. `--k` 기본 None(=9, v6 49), `--sam3_dir` 신설, VLM 사유를 `vlm_reasons()` 로 분리 (v1~v5 판정 동일 — 3,000편 대조).
- **`viz/viser_cloud.py --eval_dir label=<dir> ... --entry <data_name>`** (R103). latentcam eval 폴더의 GT(ref, 첫 폴더) 와 폴더별 pred 를 cloud.npz 점군 위에 motion 슬라이더/pin 으로 올린다. 번들 JSON arm 과 같은 변환(`_bundle_arm_poses`). 인자를 안 주면 예전 동작.
- **`eval_testset.py --init_gt_t T`** (R102). GT latent 에 t=T 수준 노이즈를 얹어 그 시점부터 denoise (SDEdit). `train_latent_cam_dm.sample()` 에 `x_init`/`t_start` 인자 추가 — 기본 None 이면 기존처럼 순수 노이즈 x_T 에서 전 스텝.
- **`.claude/context.md` — 다음 세션이 알아야 할 것** (R14, R16 에서 최상위 → `.claude/` 이동).
  `CLAUDE.md`(규칙) · `pipeline.md`(지도) ·
  `.claude/goals.md`(목표) 어디에도 안 들어가는데 **모르면 사고가 나는 것**을 모았다.
  7 절: 서 있는 지시 / 문서 간 어긋남 / 세대 이름 / 비교 금지 지표 / 반복된 실패 양식 /
  env 대응 / 열려 있는 것.
  - **`SPECS.md` 가 없다.** `CLAUDE.md` 의 `## Specifics` 와 `.claude/goals.md` 가 둘 다
    이 파일을 가리키는데 저장소에 존재하지 않는다. 두 문서가 가리키는 곳이 빈 곳이라는
    사실 자체를 기록했다.
  - **`CLAUDE.md:183`·`:190` 이 "GPU 0~4" 라고 쓴다** (R14 당시 `:182`·`:189`, R15 의 `.claude/`
    절 수정으로 한 줄 밀렸다) — 같은 파일 `## Don't` 의 0~3 과 충돌한다.
    어느 쪽이 이기는지(`## Don't`)를 명시.
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
