# Changelog (latentcam)

All notable changes to the latentcam sub-project. Follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **D296 `vista_d296_molmo2_l21_srccam_startpose` (2026-09-29, R89).** D261 start pose head + D262 srccam, 코퍼스
  latentcam_d296 (시작 절 캡션), 500 epoch.
- **D293 단일 영상 overfit config 3종 (2026-09-28, R85).** `dynpose_d200of936_{noprobe,aim_noprobe,aim_noprobe_readsrc}` —
  D291/D292 에 `seg_list_dynpose_s91_of936_{train(x800),test}` 분할, 30 epoch.
- **D294 단일 영상 overfit config 3종 (2026-09-29, R87).** `dynpose_d200of5ec_*` — 대상만 5ec7f200_1 (D293 은 대상 오인).
- **D295 단일 영상 overfit config 3종 (2026-09-29, R88).** `dynpose_d200of29a_*` — car 29a9a7d0_0, noprobe 캐시 `prefill_noprobe_of29a.pt`.
- **readout 소스 카메라 보조 손실 + append 모드 (2026-09-28, R81/D292).** `peav_readout_aux_target: srccam` 이면
  readout aux head 가 프레임별 소스 카메라 Plücker 54-d 를 맞힌다 (dataset `srccam_target`, condition 에는 안 넣음).
  `peav_readout_mode: append` 는 Molmo2 토큰을 유지하고 readout 49 토큰을 덧붙인다. 기본값(target_track / replace)은 기존과 동일.
  config `dynpose_d200v4track_e100_aim_noprobe_readsrc`.
- **`cache_molmo2_embeddings.py --probe ''` (2026-09-28, R80).** probe 문장 줄 자체를 빼고 캡션만 넣는 분기
  (template `+noprobe`). 기본값이면 기존 문자열과 비트 동일. D291 `dynpose_d200v4track_srccam_e100{,_aim}_noprobe` config.
- **D289 track small set 2 arm (2026-09-28, R77).** `dynpose_d200{v3,v4}track_molmo2_l21_srccam` — D262 모델을
  d200 ∩ 필터 v3/v4 ∩ track_look_at ∩ anchor drift>0.05u 분할(`seg_list_dynpose_s91_{v3,v4}track_*`)로 처음부터. epochs=epoch_cap 400/1600, val_every_epochs 10/40.
- **D290 v4track 100 epoch × {aim 없음, aim loss} (2026-09-28, R79).** `dynpose_d200v4track_srccam_e100{,_aim}` —
  aim arm 은 D206 값(aim_loss_w 0.1, max_t 250, gate look_at).
- **D288 `dynpose_d200_molmo2_l21_nogeo` (2026-09-28, R67).** text + Molmo2 layer21 만 쓰는 arm — DA3·source camera
  조건 없음. `dynpose_d200_molmo2_l21_da3` 를 물려받아 geo 키 넷만 끈다 (goal 1 DA3 기여 대조군).
- **VAE 왕복 전진량·부호변화 열 (2026-09-27, R65).** `scripts/vae/vae_roundtrip_val.py` 가 `gt_fwd`/`rt_fwd`
  (첫 카메라 전방축 순변위 / avg_scale) 와 `gt_flips`/`rt_flips` 를 summary.csv 에 쓴다 (tmp/r63 diffusion 분석과 같은 정의).
  video/geo/text 캐시 키를 null 로 내려 cam 경로만 싣는다. goal 4 진단: d200 test dolly_in_look_at n=479 에서
  VAE |rt_fwd-gt_fwd| med 0.0039 vs diffusion 0.1025 → 떨림은 VAE 가 아니라 diffusion 쪽.
- **molmo2_conn ViT 출력 캐시 (2026-09-27, R60).** `video_onfly_feat_cache` (<scene_key>.npy uint16=bf16 49x729x2304) —
  dataset 이 `video_feat` 를 싣고 학습이 frozen ViT 를 건너뛴다. `cache_molmo2_frames.py MODE=feat` 로 굽는다
  (on-the-fly 출력과 비트 동일 확인). null 이면 기존 frame 경로 그대로.
- **DA3 unposed arm (2026-09-26, R59 / D287).** `conf/experiment/dynpose_d200_molmo2_l21_da3_unposed.yaml` —
  부모 `dynpose_d200_molmo2_l21_da3` 대비 `geo_posed: false`(DA3 에 소스 카메라 cam token 없음) + unposed 6-view bf16 캐시.
- **video_onfly = molmo2_conn arm (2026-09-25, R58 / D286).** `models/molmo2_video_connector.py` —
  frozen Molmo2 SigLIP2 ViT 를 매 스텝 돌리고 Molmo2 connector 구조(3x3 attention pooling + SwiGLU
  projector)를 **랜덤 초기화로 학습**, 9x9->8x8 평균 후 video CA 로. dataset 이 `video_frames`(49x378x378
  uint8)를 싣고, connector 는 `model.video_connector` 로 ckpt 에 저장된다. `conf/config.yaml`
  `video_onfly: null`(기본 = 기존 동작), config `dynpose_d200_siglip2conn_srccam.yaml`, eval_testset 도 지원.
  frame 디코드 캐시 `video_onfly_frame_cache` (`scripts/data/cache_molmo2_frames.py`, 없으면 PNG 경로 그대로).
- **DA3 24 프레임 arm (2026-09-25, R54 / D285).** `conf/experiment/dynpose_d200_molmo2_l21_da3_v24.yaml`
  — 부모 `dynpose_d200_molmo2_l21_da3` 대비 `geo_num_views 24` + 24-view bf16 raw 캐시(≈864 GB) 두 줄뿐.
- **DA3 12 프레임 arm (2026-09-24, R31 / D274).** `conf/experiment/dynpose_d200_molmo2_l21_da3_v12.yaml`
  — `geo_num_views: 12` + 전용 bf16 캐시 `geo_raw_cache_da3_v12_bf16` (≈432 GB).
  `scripts/data/cache_geo_raw_da3.py SAVE_DTYPE=bfloat16` — 굽는 자리에서 bf16 으로 저장
  (`convert_geo_raw_cache.py` 와 같은 캐스트, 12 view 는 fp32 808 GB 라 사후 변환이 부담).
  기본 float32 = 옛 동작.

### Removed
- **`val_every_epochs` 되돌림 (2026-09-28, R79).** 사용자 지시로 validation 을 다시 매 epoch. D289 config 의 키도 삭제.

- **DA3 / Molmo2 캐시 정리 — 33건 / 1,507.1 G 삭제 (2026-09-24).** `/data1/cympyc1785/data`
  아래 da3·molmo2·siglip2·peav·umt5 캐시 **46개 2,200.9 G 를 전수로 세고** config 참조와
  대조해 고른 것이다. 목록은 `tmp/cache_cleanup/manifest.txt` (`kind<TAB>bytes<TAB>relpath`,
  `--root /data1/cympyc1785/data`). 사후 확인: 33건 전부 소멸, 현역 캐시 8종 전부 생존.
  - **A. 어느 config 도 안 가리킨다 (427.9 G)**
    - `latentcam_dynpose_d200/geo_raw_cache_da3` (fp32) **402.2 G**. 2026-09-19 에 bf16 으로
      재굽기가 끝났고 **항목 수가 10,169 로 bf16 과 같다.** d200 arm 셋이 전부 `_bf16` /
      `_v12_bf16` 을 읽는다. fp32 경로가 남아 있던 곳은 `main/convert_geo_raw_cache.py:23`
      의 `--src` **예시 한 줄**뿐이었다.
    - `latentcam_dynpose_d200/molmo2_cache/{prefill,text}.shard*of12.pt` 24개 **25.7 G**.
      샤드는 `--merge_shards` **입력**일 뿐 학습이 안 읽는다
      (`main/cache_molmo2_embeddings.py:574`, `:654`). 병합본이 마지막 샤드보다 늦고
      (`prefill.pt` 02:08:01 vs 01:57:45) 크기가 샤드합과 정확히 일치한다 (18.5/18.5 G,
      7.1/7.1 G) — 이 둘이 "병합이 끝났다" 의 근거다.
  - **B. 재생성 가능한 미러 (1,046.6 G)** — `Scene-Decoupled-Video-dataset/da3_depth_raw`.
    `da3/**/depth.npz` 를 비압축 `.npy` 로 푼 것이고 **원본 npz 를 목록에서 뺐다** (삭제 후
    생존 확인). 필요하면 `scripts/data/cache_da3_depth.py` 로 되만든다. 읽는 쪽인
    `sd_whuman_*` / `mix_dl3dv_sd_*` yaml 8개는 최종 수정이 2026-08-11~17 이다.
  - **C. 은퇴한 arm 의 코퍼스 캐시 (32.5 G)** — d129 10.8 · `latentcam_dynpose`(d84/d107/d117)
    10.5 · TRUMANS d77 6.4 · `da3_k6_d121/peav_cache` 2.1 · d157 molmo2 0.4 · d107 peav 0.2.
    **config 는 살아 있다** — 해당 arm 을 다시 돌리려면 캐시를 되굽어야 한다
    (TRUMANS d77 은 task #125 가 pending).
  - **C 오분류 1건 — `Vista4D-Eval-Data/latentcam_da3/geo_raw_cache_da3` (2.1 G) 은 은퇴가
    아니었다.** "은퇴한 arm" 으로 묶어 지웠으나 이 디렉토리는 vista arm **6개가 공유**하는
    캐시이고 그중 `vista_d261_molmo2_l21_da3_startpose.yaml:81` 은 **2026-09-23 에 끝난
    D261**(task #275) 이다. 스캔에서 atime 이 `2026-09-22` 로 찍힌 것을 보고도 근거로 안 썼다.
    영향은 **속도뿐** — `dataset_dl3dv.py:402` 가 파일이 없으면 `continue` 하고 `:416` 이
    `"<- 나머지는 on-the-fly DA3"` 를 찍는다 (오류가 아니다). 여섯 arm 전부
    `da3nested-giant-large` `[252,448]` `last` `posed` `6 view` `first_view_target_s` 로
    설정이 같고 d261 은 `da3_cam_token_per_sample: true` 라 게이지도 안 갈린다 — 한 번
    되구우면 전부 복구된다. `scripts/data/cache_geo_raw_da3.py` (fp32, GPU 0) 로 재굽기해
    **52 파일 / 2.1 G 로 삭제 전과 같아졌다** (로그 `tmp/cache_cleanup/rebake_vista_da3*.log`).
    네 코퍼스를 차례로 태워 커버리지를 확인했다: d261 이 16 (그 코퍼스가 16 scene 뿐),
    d121 이 나머지 36 → 52, d128 · d77 은 **52/52 전부 skip-done** — 즉 네 코퍼스가 같은
    52 scene 을 보고 있고 빠진 키가 없다. `non-constant 0` (geo_idxs 가 변이 상수).
  - **안 지운 것** — `geo_raw_cache_da3_bf16`(6프레임, 202 G) · `geo_raw_cache_da3_v12_bf16`
    (12프레임, 403 G) · `molmo2_cache/{video,prefill.pt,text.pt}`(330 G) ·
    `siglip2_cache/video`(137 G) · `umt5_cache`(24 G) · SD 원본 `da3/`.
    집행 전 7개 토큰(`_bf16` `siglip2_cache` `umt5_cache` `molmo2_cache/video` `/prefill.pt`
    `/text.pt` `Scene-Decoupled-Video-dataset/da3/`)으로 목록을 재검사해 침범 0건을 확인했다.
  - **곁다리** — `main/conf/config.yaml:656` 의 `custom_geo_depth_cache_dir:
    /data1/.../DL3DV/scenes/da3_depth_raw` 는 **존재하지 않는 경로**다 (SD 쪽 것과 이름만 같다).
    이번 삭제와 무관하지만 조사 중 드러나서 적어 둔다.
  - **집행은 사용자가 했다** — 삭제 스크립트를 쓰거나 고치는 것이 classifier 에 막혀
    (Write 1회 / Edit 1회) `xargs rm -rf` 를 사용자가 직접 쳤다. R17 집행기
    (`tmp/r17/apply_cleanup.py`) 를 `scripts/data/apply_cleanup.py` 로 승격하면서
    `--root/--manifest/--keep` 를 뚫는 일은 **아직 남아 있다.**

- **latentcam 트리 정리 — 460건 / 169.6 G 삭제 (2026-09-23, R17).** `du` 실측 358 G → **188 G**.
  지운 것은 전부 gitignore 대상이라 `git status` 에 아무 것도 안 뜬다 (추적 파일 손실 0).
  - `main/evaluate/CLaTr/results/**/epoch=*.ckpt` **75건 13.4 G** — 중간 epoch 체크포인트.
    **`last.ckpt` 는 남겼다** (8개). 현재 게이지는 `checkpoints/clatr_dynpose_d200_s91_real_epoch149.ckpt`
    로 이미 `results/` **밖에** 복사돼 있다.
  - `main/wandb/run-*` **233건 18.5 G** — 클라우드에 동기가 끝난 online run 디렉토리.
    **`offline-run-*` 55건(4.4 M)은 남겼다** — 이것들은 클라우드에 없어서 지우면 원본이 사라진다.
  - `results/0backup` **31건 20.1 G**, `results/202607*`·`results/202608*` **96건 117.7 G**.
  - `__pycache__` · CLaTr `tmp*/` scratch · `hc*_*.txt` hardcopy **25건 2.3 M**.
  - **지우지 않은 것과 그 이유** — `main/configs_backup/*.py` 28개는
    `main/train_vae_dl3dv.py:14` 가 `load_cfg('config_vae_dl3dv')` 로 **이름으로** 부른다
    (디렉토리 이름이 backup 이라 죽은 것처럼 보이는 게 함정이다). `eval_my/` 188개는
    `main/conf/experiment/da3_7k_da3geo_frontanchor.yaml:5` 가 `common100ep` 을 가리켜
    통째로 보존. `checkpoints/dinov2-large/`, `main/conf/experiment/*.yaml` 158개,
    `exp_results/` 도 보존.
  - 집행기 `tmp/r17/apply_cleanup.py` 는 **고르지 않고 매니페스트(`tmp/r17/deleted_manifest.txt`)
    에 적힌 경로만** 지우며, 보존 대상 침범을 삭제 전에 재검사해 하나라도 걸리면 멈춘다.
    사후 dry-run 이 `실재 0건 / 이미 없음 460건` 을 찍어 전량 집행을 확인했다.

### Added
- **"DA3·Molmo2 인코더가 필요한가" ablation 3 arm (2026-09-23, R19~R21 / D268~D270).**
  - `conf/experiment/dynpose_d200_umt5_srccam.yaml` (D268) — umt5 text + srccam Plücker 만.
    D262 srccam 에서 Molmo2 video CA 를 뺀 것.
  - `conf/experiment/dynpose_d200_siglip2_srccam.yaml` (D269) — umt5 text + **SigLIP2 ViT
    feature** (3136 x 2304) + srccam. D262 의 Molmo2 LM hidden 을 ViT 출력으로 교체, Molmo2
    prefill text part 제거.
  - `conf/experiment/dynpose_d200_molmo2_l21_da3_v49.yaml` (D270) — D200 molmo2_l21_da3 에서
    `geo_num_views: 6 -> 49`, 캐시 없이 on-the-fly DA3.
  - `main/cache_molmo2_embeddings.py --vit_only` — Molmo2 `vision_backbone.encode_image`
    (vit_layers -3,-9 concat = 2304-d) 를 프레임 안 27x27 -> 8x8 평균풀링해 씬당 `{'emb'}` 로
    굽는다. LM forward 0회. 플래그를 안 주면 기존 경로 그대로.

- **`train.md` 신설 — 학습·평가 운용 문서 (2026-09-23, R11).** `pipeline.md` 가 "무엇이
  학습되는가"(구조)를 다루므로, "어떻게 돌리고 어떻게 읽는가"를 여기에 분리했다.
  §1 기동(명령 한 줄 · GPU · screen · `queue_runs.py` 대기열 · 중단/재개 · smoke)
  / §2 학습 루프(세 갈래 · 보조 손실 셋 · `epoch_cap` · `step_timing`)
  / §3 평가(학습 중 val 160 표본 vs `eval_testset.py` 전량 · **wandb 키 사전 20행** ·
  CLaTr subprocess 두 개와 조용한 skip · paired bootstrap) / §4 체크포인트 / §5 지뢰 15행.
  - **인용한 `file:line` 74개를 전부 스크립트로 열어 대조했다** (`train_latent_cam_dm.py` 62,
    `conf/config.yaml` 9, `queue_runs.py` 5, `stop_at_epoch.sh` 1). 경로 49개도 존재 확인 —
    미해결은 산출물 파일명(`last.pth` / `metrics.json` 등)과 basename 인용뿐이다.
  - 문서화하면서 확정한 것 셋: ① `eval_subject_in_frame.py` / `eval_collision_rate.py` 는
    latentcam 이 아니라 `camera_generation/dataset/eval/` 에 있다 (코퍼스 기하가 필요해서
    학습 루프에 없다). ② `eval_paired_bootstrap.py` 로 CI 를 낼 수 있는 건 per-sample 지표
    5개뿐이고 `clatr/{precision,recall,density,coverage,fcd}` 는 전 행에 전역 값이 복제된
    집합 지표라 못 낸다. ③ CLaTr subprocess 실패는 `metrics.json not produced` 한 줄만
    남기고 학습을 죽이지 않는다 (`main/train_latent_cam_dm.py:1150`, `:1155`) — 곡선에서
    `val/clatr/*` 가 통째로 비면 이 줄부터 찾아야 한다.
- **`pipeline.md` 신설 — latentcam 모델 구조 문서 (2026-09-23, R10).** 이 트리에는 모델
  구조를 한 장에 놓은 문서가 없었다 (`docs/` 는 ablation 메모 4장, `CHANGELOG.md` 는 45만 자).
  Part A 모델 본체 / Part B 변형 축(arm) / Part C 모듈별 + 부록 지뢰 표.
  - **인용한 `file:line` 91개를 전부 열어 확인했다** — 파일 존재 + 그 줄의 내용이 주장과
    일치하는지 스크립트로 대조. 미해결 3건은 산출물 파일명(`last.pth`,
    `vae_20260302_300.pth`)과 아직 안 쓴 `train.md`(R11) 뿐이다.
  - 문서를 쓰면서 **조사 중 바로잡은 것 넷**:
    ① DiT 의 입출력 폭은 11 이 아니라 **64** (`conf/config.yaml:150 cam_dim`) — VAE latent 를
    디노이즈한다. `__init__` 기본값 `cam_dim=11`
    (`models/camera_diffusion_model_latent.py:137`) 과 `:548` 의 꼬리 주석 `# (B, T, 9)` 가
    둘 다 낡아서 생긴 오독이다.
    ② `cfg.val_ratio` 는 **없는 키**다. seg-list 를 주면 val = `test_seg_list` 앞부분
    (`val_max_batches × batch_size`), 안 주면 `train_frac: 0.9` 무작위 분할
    (`main/base.py:84-114`).
    ③ geo backend 는 3개가 아니라 **4개** — `lagernvs`/`scenetok`/`custom`/`da3`
    (`models/geo_encoder.py:339/343/347/355`). `scenetok` 은 stub 이라 쓰인 적이 없다.
    ④ "죽은 코드 = `grep -l core_pkg`" 는 틀렸다. `main/base.py` 는 **주석**에만 있고
    `train_latent_cam_dm.py` 는 **함수 안**(`:651`, CLIP 분기)에 있다. 판별은 AST 로
    **모듈 레벨** import 만 봐야 하며, 그 기준으로 죽은 파일은 14개 + 파싱 실패 1개
    (`run_cam_dm.py:133`).
  - arm 축은 서술이 아니라 **`conf/experiment/*.yaml` 158장을 기계적으로 집계**해서 넣었다.
    hydra `defaults:` 상속 때문에 자식 yaml 은 한 줄만 적으므로 grep 분류는 못 쓴다는 것도
    같이 적었다.

- **D264 필터의 DynamicVerse 입력 경로 `--source dv` (2026-09-23).** 사용자 지시
  "DATA/worldtraj/dynamicverse/dynpose-100k 분류해달라고 한거야" — 대상은
  `DATA/DynPose-100K` 가 아니라 **`DATA/worldtraj/dynamicverse` 트리 전체**
  (10 코퍼스 + `eval_index`, **47,338 씬**, 전부 fps 10). 레이아웃이
  `<corpus>/<scene>/video_input.mp4` 라 프레임 디렉토리를 전제하던 기존 경로가 못 읽는다.
  새 분기는 mp4 를 메모리에서 디코드한다 (`frames_of`).
  - **게이지 정합**: 640x360 으로 리사이즈하고 **앞 49 프레임만** 쓴다. G3 의 Sobel 임계 40 은
    해상도 의존이고 49 프레임(=4.9 s)이 DynPose 쪽 게이지라, 둘을 안 맞추면 같은 임계가 다른
    뜻이 된다.
  - 도구는 `tmp/d264/run_d264.py` 에 있고 `tmp/` 는 gitignore 대상이라 **이 항목이 유일한
    기록이다.**

- **D264 시점/컷/오버레이 필터를 건 D200 arm (`dynpose_d264_molmo2_l21_da3`, 2026-09-22).**
  `dynpose_d200_molmo2_l21_da3` 와 **seg_list 두 줄 + `val_max_batches` 만** 다른 사본이라
  기존 arm 은 한 글자도 안 바뀐다. 코퍼스 디렉토리·캐시·CLaTr 게이지 전부 공유.
  - 새 목록: `latentcam_dynpose_d200/seg_list_dynpose_s91_d264_{train,test}.txt`
    (train 46,278 -> 23,989 / test 5,144 -> 2,685). **원본 s91 목록은 그대로 둔다.**
  - 게이트 3종 (씬 10,169 -> 5,069, 49.8%): G1 시점 VLM(third/none 만, closeup 은 보이는
    부위 <= 3) 탈락 4,259 / G2 컷 HSV Bhattacharyya >= 0.45 탈락 321 / G3 오버레이 정지엣지
    >= 0.02 (정지 카메라 `sd_med < 8.0` 면제) 탈락 951. 상세는 `EXPERIMENTS.log` D264-f.
  - 분할을 **가로지르지 않는다** (제거만) — CLaTr 게이지가 새 test 를 미리 본 적이 없다.
  - 인덱스 캐시는 seg_list 를 키에 안 쓴다 (`dataset_dl3dv._load_index`) — D200 캐시 재사용.
  - ⚠ 대조군과 **test 집합이 다르다** (5,144 vs 2,685). 수치를 나란히 놓을 때 명시할 것.
  - **DynPose-100K 전량 10,381 편 판정표** (`tmp/d264/d264_dynpose_all_scenes.csv`, 2026-09-22).
    d200 코퍼스(10,169)에 없는 212 편은 `images_4/` 가 없어 `recon_and_seg/<uuid>/video.mp4`
    를 640x360 으로 풀어 같은 눈금으로 봤다 (frame0 절대차 0.31/255). 통과 5,170 (49.8%),
    탈락 G1 4,344 / G2 330 / G3 986. 라벨 closeup 1,438 / ego 1,373 / selfie 1,533 / third 6,037.
    통과 씬은 **전부 `label=third`** (예외 0); third 인데 떨어진 867 편은 G2/G3 탈락이다.
- **geo CA 를 소스 카메라 궤적으로 교체하는 ablation (`srccam_cond`, D262, 2026-09-22).**
  사용자 지시 "D200으로 DA3가 진짜 필요한지 학습으로 판단해보자. Molmo2_l21만 쓰고 DA3
  들어갈 부분에는 source camera (49frame)을 plucker나 mlp 태워서 condition으로 들어가도록
  대체해서 학습하나 돌려줘". 기본 `null` 이라 **기존 arm 은 state_dict·동작 비트 동일**
  (단위검증: `geo_in_mlp=False, geo_pe=False` 와 미지정이 state_dict 전량 `torch.equal`,
  forward 도 `torch.equal`).
  - **무엇이 바뀌나**: geo cross-attention 의 key/value 가 DA3 patch 토큰
    `6 view x 777 = 4662개 x 3072-d` 에서 **소스 카메라 49 프레임 x 54-d** 로 바뀐다.
    이미지 I/O 도 DA3 forward 도 없다 (`geo_encoder: null`).
  - **`dataset_dl3dv._srccam_tokens`**: 프레임은 `_target_frame_idxs(s, e)` = 생성 대상과
    같은 49 프레임의 소스 카메라. 기준계·단위는 geo 스트림과 글자 그대로 같다 —
    `rel_v = w2c_src[v] @ inv(extrinsics[0])`, `t = rel_v[:3,3] / norm_scale`.
    - `'plucker'` (T,54): 3x3 격자 위 Plücker ray 9개 x (direction 3 + moment 3).
      pose 와 FoV 가 같이 들어가고 회전 파라미터화 선택이 안 남는다.
    - `'param'` (T,11): rot6d(6) + trans(3) + intr(2). `_geo_cam_param` 재사용.
    - 출력 키가 `geo_emb` 라 학습 루프의 `elif 'geo_emb' in data:` 분기가 그대로 받는다.
  - **모델 쪽 자동 분기 둘** (`camera_diffusion_model_latent`):
    `geo_in_mlp` = `geo_proj` 가 `Linear(54,512)` 대신 `Linear->SiLU->Linear` (54-d raw
    기하량을 bare Linear 로 올리면 표현력이 affine 하나다);
    `geo_pe` = geo 토큰에 sinusoidal PE. DA3 토큰은 무순서 view x patch 라 PE 가 없었지만
    (pose 는 DA3 자신의 `cam_token` 이 들고 있다) 소스 카메라 토큰은 **시간순 49개**라
    순서 자체가 정보다.
  - **양쪽 assert**: `srccam_cond` 와 `geo_encoder`/`geo_cam_embed` 는 같은 `geo_proj`
    입력을 정하므로 동시 활성이 dataset·train 양쪽에서 막힌다.
  - arm 둘: `dynpose_d200_molmo2_l21_srccam` (D262), `dynpose_d200_umt5only` (D263,
    geo·video CA 둘 다 제거한 umt5 단독 바닥선). 둘 다 `dynpose_d200_molmo2_l21_da3` 를
    `defaults` 로 물려받아 코퍼스·분할·CLaTr 게이지가 같은 자다.
- **시작 pose 예측 (`start_pose_pred`, D261, 2026-09-22).** 사용자 지시 "첫 카메라는 따로
  source 첫 카메라 기준 상대 pose로 값을 마련해둬서 학습하도록해줘". 기본 `false` 라
  키도 안 생기고 토큰도 안 붙으므로 **기존 arm 은 비트 동일**이다.
  - **왜 필요한가**: `utils/data_utils.normalize_camera_extrinsics_and_points` 가
    `E @ inv(E[0])` 라서 VAE 로 들어가는 `cam_param[0]` 은 **항상 항등**이다. 즉 "소스
    카메라 대비 어디서 출발하는가"는 궤적 어디에도 안 남는다. d260 뱅크는 첫 카메라를
    36 후보에서 뽑아 이 성분이 살아 있다 (코퍼스 612 변이 실측: `|t|/S` median 0.5560 /
    p90 1.3268 / max 2.2377, 회전 median 90.07° / p90 175.50°).
  - `main/dataset_dl3dv.py` — `_start_pose(scene_idx, s, w2c_t0, norm_scale)` 가
    `rel0 = E_target[0] @ inv(E_src[s])` 를 `[rot6d, t/avg_scale]` **9-d** 로 낸다.
    `__getitem__` 이 `cfg.start_pose_pred` 일 때만 `out['start_pose']` 로 넣는다.
    11-d(= `cam_param` 폭)가 아닌 이유: intr 2 채널은 `intr_norm: rel` + scene 내 상수라
    이 코퍼스에서 항상 `[1,1]` 이다 (실측 fx/W = 1.952716, 자기 frame0 대비 편차 0.0e+00).
  - `models/camera_diffusion_model_latent.py` — ctor 에 `start_pose_dim` /
    `start_pose_tf_p` / `start_pose_noise`. 시퀀스 끝에 학습가능한 `start_query` 토큰을
    붙여 T -> T+1 로 만들고, self-attn 뒤 그 자리 hidden 을 `start_out` 으로 읽어
    `self.start_pred` 에 남긴다 (forward 반환형은 `(B,T,cam_dim)` 그대로).
    `cam_in` 채널축 concat 을 안 쓴 이유 둘: 입력 폭이 바뀌어 base arm 과 state_dict
    호환이 깨지고, T 토큰 전부에 답이 방송돼 head 가 자명해진다.
  - **teacher forcing**: `start_in = Linear(9 -> hidden)` 이 노이즈 섞인 GT 를 토큰으로
    만들어 `start_query` 에 **더한다**. 드롭은 element-wise 가 아니라 **표본 단위**
    (`torch.rand(B,1,1) < p`) — 드롭된 가지가 추론 경로(query 단독)와 비트 동일해야 하기
    때문. `self.training` 게이트가 있어 val/sample 은 항상 주입 없는 경로를 탄다.
  - `main/train_latent_cam_dm.py` — `start_pose_loss()` 가 MSE + 진단 두 개
    (`trans_err_u`, `rot_err_deg`; 6D->R 은 Zhou et al. Gram-Schmidt). 학습 손실에
    `start_pose_w` 로 더하고 `train/start_{mse,trans_u,rot_deg}` /
    `val/start_{mse,trans_u,rot_deg}` 를 wandb 에 남긴다. `start_pose_pred=true` 인데
    `start_pose_w=0` 이면 모델 구성 시점에 assert (head 가 학습되지 않는다).
  - `main/conf/config.yaml` 에 기본값 5 개(`start_pose_pred/dim/w/tf_p/noise`).
  - arm 둘: `vista_d261_molmo2_l21_da3_startpose`(tf 없음) /
    `..._startpose_tf`(`tf_p 0.5`, `noise 0.1`). 코퍼스는
    `Vista4D-Eval-Data/latentcam_d261` (d260 + d260t 뱅크, 612 변이, 씬 단위 holdout 3 편).
- **추론 지연 계측 두 벌 — 우리 diffusion 모델 / Molmo2 자체 (2026-09-22).** 사용자 지시
  "prefill과 decode시간 측정해줘" + "내가 말한건 Molmo2 자체의 prefill 단계, decode 단계의
  추론 시간인데". 둘은 **다른 모델**이라 계측 지점도 따로 둔다. 기본 off 라 기존 경로는
  비트 동일하다.
  - `scripts/eval_testset.py --timing [--timing-warmup N]` — 우리 모델. 단계별
    `torch.cuda.synchronize()` 로 감싼 누적기: `prefill/{geo,text,track,video}`,
    `decode/{denoise,vae}`, `post/{to_world,clatr_feats,io}`, `data`. denoise 스텝 수를
    세려고 `main/train_latent_cam_dm.py:sample()` 에 **선택 인자 `step_cb`** 를 더했다
    (기본 None = 기존과 비트 동일).
  - `main/cache_molmo2_embeddings.py --timing [--timing_out <json>]` — Molmo2 자체.
    단계를 4개로 가른다: `vit`(씬당 ViT→prefix inputs_embeds, **prefill latent 에 필요**),
    `prefix_fwd`(씬당 prefix forward, `--video_out` 전용이라 prefill latent 엔 불필요),
    `prefill`(캡션 배치당 prefix+꼬리 1 forward = KV 캐시 채우기 = prefill latent 가
    나오는 자리), `decode`(생성 토큰당 seq_len=1 forward). `--limit_scenes` 로 몇 편만
    재는 용도이며 대량 굽기에는 켜지 않는다 (synchronize 삽입).
- **`scripts/eval/clatr_score_dir.py` — 임의의 eval 폴더에 CLaTr 지표를 매기는 드라이버
  (2026-09-21).** `eval_testset.py` 는 자기가 방금 만든 `--out` 폴더에만 CLaTr 을 돌린다.
  베이스라인 예측(GenDoP / E.T.)은 `run_gendop_eval.py --stage evaldir` 이 만든
  `eval_dir_*/` 로 따로 떨어지는데, 그 안에도 `test/<entry>_transforms_{pred,ref}.json` +
  `_caption.json` 이 같은 규약으로 들어 있으므로 `src.extraction` → `src.eval_only` 두
  단계를 그대로 돌릴 수 있다. 기존 코드는 **한 줄도 안 바뀐다** (새 파일 하나).
  - 사용: `python scripts/eval/clatr_score_dir.py ours=<eval_my/...> gendop=<eval_dir_...>`.
  - `ensure_split()` — `test_valid.txt` 가 없으면 `test/*_transforms_ref.json` 에서 만든다.
  - `ensure_caption_feats()` — CLaTr caption modality 는 **미리 인코딩된 CLIP feature**
    (`seq/test/<e>_caption.npy`, `token/...`)를 읽는다(`caption_dataset.py:65`).
    `eval_testset.py:436-439` 는 추론 루프 안에서 이걸 같이 떨구므로 밖에서 만든 폴더엔
    없다 → 캡션 json 에서 같은 인코더(`ViT-B/32`, `max_token_length=None`)로 재생성한다.
    **이미 있으면 안 건드린다** (우리 런 것을 덮으면 그 폴더 지표가 재현이 안 된다).
  - **게이지**: `--standardization` 기본 None = CLaTr configs 기본값(`'0120'`).
    `eval_testset.py:486` 도 `dataset/standardization` 을 **안 넘기므로** 기존 숫자 전부가
    이 게이지다 (런 `config.yaml` 의 `clatr_standardization: dynpose49_d200` 은 안 쓰인다).
    비교군에는 같은 인자를 줄 것.
  - **`metrics.json` 의 `clatr/clatr_score` 는 0 에서 잘린다** —
    `evaluate/eval/src/metrics/modules/clatr_score.py:33` 이 `torch.max(score, 0)`.
    음수 코사인은 `preds_scores.csv` 의 per-row 값으로 봐야 한다
    (실측 D240 gendop: 집계 0.0 / raw −0.2197).
- **video CA 의 text 융합 축 옵션 `video_text_fuse` + D217 arm
  (`conf/experiment/dynpose_d200_molmo2_decfuse_l21_da3.yaml`) (2026-09-21).**
  `_build_video_tok` 이 video/text 두 파트를 합치는 축을 고른다.
  - `token` (기본) = 기존 그대로. `cat([text_tok, video_tok], dim=1)` — 49 + 3136 = 3185 토큰,
    파트마다 자기 LN/proj/PE. **state_dict·동작이 비트 동일**하다.
  - `frame_concat` (신규) = **채널축 프레임 정렬**. molmo2 decode 캐시의 49 슬롯은
    `--decode_keep points --fps 2.0` 덕분에 프레임과 1:1 이고 video 캐시는 49프레임 × 8×8
    patch = 3136 이다. 토큰축 concat 은 이 정렬을 안 쓰므로, 프레임 f 의 decode feature 를
    그 프레임의 patch 64개에 broadcast 해 `(3136, 2560+2560)` 한 덩어리로 만들고
    **projection 하나**(`video_proj` in_features 5120)로 넣는다. `video_text_proj` 는 아예
    생성되지 않고 key_padding_mask 는 `video_mask` 하나만 쓴다.
  - 두 파트는 스케일이 달라 concat **전에 각각** LayerNorm 한다 (FIX-D117 와 같은 이유).
    트랙 점이 없는 프레임은 text 절반만 0 이고 visual 절반은 산다. `aim=free` 변이는 49슬롯
    전부 0 + mask False 라 부모 arm 과 동일하게 visual-only 로 떨어진다.
  - **molmo2 캐시 재굽기 0회** — 기존 `video/<scene>.pt` + `text.pt` 를 그대로 조합한다.
    3136 = 49 × 64 정렬은 모델이 assert 로 확인한다.
  - `scripts/eval_testset.py` 의 모델 생성부에도 같은 kwarg 를 넘긴다. 안 넘기면 `video_proj`
    in_features 가 2560 으로 만들어져 strict load 가 shape mismatch 로 죽는다.
  - smoke(200씬/1 epoch, GPU 0): `(model) video CA: ... fuse=frame_concat`, 학습 + val +
    CLaTr 전 구간 통과 (`val/loss_traj=2.873523`).
- **aim 보조 손실 + D206 arm (`aim_loss_w` / `conf/experiment/dynpose_d206_molmo2_l21_aim.yaml`) (2026-09-20).**
  학습 스텝 안에서 예측 eps 로 x0 를 복원하고 그 x0 를 **궤적 VAE 로 디코드**해 실제 pose 를
  만든 뒤, 카메라 forward `R[2,:]` 와 (카메라중심 `-R^T t` → subject OBB center) 벡터 사이의
  `1 - cos` 을 최소화한다 (`train_latent_cam_dm.aim_loss`).
  - **모델 구조는 안 바뀐다.** track 은 입력이 아니라 **손실 타깃**으로만 쓰인다. D201-C
    (`target_track_dim>0`, x_t 에 concat = 모델 입력) 와 D201-B (별도 readout head) 와는
    쓰는 자리가 다르고, 추론 경로는 ③ `dynpose_d200_molmo2_l21_da3` 와 글자 그대로 같다.
  - 게이지 검증: cam_param 경로로 잰 cos 과 world 좌표에서 바로 잰 cos 의 최대 차이
    **2.98e-7** (d200 150씬 756변이). 축·원점·분모(`avg_scale_context_first_cam`) 셋 다 닫힌다.
  - `aim_loss_gate: look_at` — d200 실측 GT 각(변이별 중앙값의 중앙값)이 `aim=look_at`
    n=543 **1.03°**, `aim=free` n=213 **15.27°**. free 는 GT 가 애초에 겨냥을 안 하므로
    손실을 걸면 GT 와 싸운다.
  - `aim_loss_max_t: 250` — x0_hat 이 `1/sqrt(ᾱ)` 배로 부푼다. scaled_linear(0.00085,0.012)
    에서 t=250 ᾱ 0.674 / SNR 2.07, t=999 ᾱ 0.005 / SNR 0.005.
  - **기본값 `aim_loss_w: 0.0`** 이라 기존 arm 은 데이터셋 로드·VAE 디코드·손실 전부 미실행
    (bit-identical). 켜질 때만 `camera_vae.requires_grad_(False)` 로 VAE 를 얼린다 — decode 가
    `no_grad` 밖이라 안 얼리면 옵티마이저에 없는 파라미터에 `.grad` 가 영원히 쌓인다.
  - 게이트 통과량은 매 스텝 `train/aim_n` (+ `train/aim_loss`, `train/aim_deg`) 로 보인다.
    smoke(40씬/batch 4) 실측: 36 스텝 중 25 스텝 n>0, aim_deg 중앙값 7.0°.
  - 데이터셋은 `prompts.json` 의 `aim` 필드만 `_aim_cache` 로 따로 읽는다 — `samples` 튜플에
    원소를 **안 늘린다** (늘리면 `.latentcam_index/*.pt` 직렬화 캐시가 전부 깨진다).
- **D201-C arm 추가 (`conf/experiment/dynpose_d201_molmo2_l21_track_d5.yaml`) (2026-09-19).**
  `dynpose_d200_molmo2_l21_da3` 에 subject OBB 궤적 concat 조건을 얹는다
  (`target_track_dim: 4` / `target_track_dropout: 0.5` / `target_track_val_drop: true`).
  **모델 코드 변경 0** — D195-B(`dynpose_d194_molmo2_l21_track_d5`) 가 쓰던 경로를 d200
  코퍼스 위에서 켜는 것뿐이다. 앞선 두 세대(D195/D196)는 subject 조건을 **한 쌍**으로
  돌렸는데(readout+aux = "맞히게 한다" / track_d5 = "그냥 준다") d200 세대는 D201-B 로
  readout 쪽만 기동해 짝이 비어 있었다. 짝이 없으면 D201-B 의 결과가 "subject 정보가 도움이
  되는가"인지 "readout 이라는 배선이 도움이 되는가"인지 못 가른다.
  `val_drop: true` 라 val 눈금은 ③ l21 대조군과 같다 (추론은 track 없이).
  전제인 `<scene>/da3/target_track.npz` 는 D201-B 용으로 이미 전량 있다 (실측 10,169/10,169 씬).
  캡션·pose·seg list·모델 입력 토큰이 안 바뀌므로 캐시 재굽기 0.
- **D201 두 arm 추가 (`conf/experiment/dynpose_d201_molmo2_l21_{mag,readout}.yaml`) (2026-09-19).**
  둘 다 `dynpose_d200_molmo2_l21_da3` 를 상속하고 **서로 다른 축 하나씩만** 움직인다.
  - `_mag` (캡션축): `prompts_file: prompts_mag.json` — 뱅크 캡션을 `--magnitude --nl_framing`
    으로 다시 구워(`captions_d201mag.json`, hole_bank_d185 80,328 + d199 4,242 변이)
    `make_prompts_simple.py` 로 10,169 씬에 얹었다. 정도부사 + framing 절이 돌아온다.
    umt5 prefill 캐시도 그 문장으로 새로 구웠다 (`umt5_cache/text_len128_mag.pt`) — 부모가
    물고 있는 `text_len128.pt` 를 그대로 쓰면 전량 miss 라 fallback 인코딩으로 조용히
    느려진다 (결과는 같다). `text_len: 128` 은 유지 (D196 실측 max 76 tok, 절단 0).
    ⚠ 같은 캡션축을 D196 에서 시도했고 사용자가 "오히려 별로야"로 중단시킨 전례가 있다.
  - `_readout` (모델축): 캡션은 d200 그대로 두고 molmo2 스트림에 `PeavReadout`
    (49 query × 2층) + subject OBB center probe head (`aux_dim 3`, `aux_w 0.1`) 를 단다.
    D195-A(`dynpose_d194_molmo2_l21_readout`) 와 같은 배선. 전제인
    `<scene>/da3/target_track.npz` 를 d200 코퍼스에 전량 생성했다
    (`export_target_track.py`, 10,169 씬 / 51,422 변이 / valid 100%).
- **umt5 prefill 캐시 (`main/cache_umt5_embeddings.py` + `text_emb_cache_path`) (2026-09-19).**
  `step_timing` 이 잡아낸 두 번째 비용 — umt5 forward 가 arm ① 42.5 ms (15.0%) / arm ②
  48.2 ms (10.0%) 로 **denoiser forward 보다 컸다**. 인코더는 frozen 이고 캡션은 세그먼트
  붙박이라 epoch 마다 같은 문자열을 다시 돌고 있었다.
  - 빌더는 캡션을 **인덱스 캐시**(`.latentcam_index/*.pt` 의 samples 4번째 원소 = 데이터셋이
    실제로 쓰는 문자열)에서 긁고, 유효 토큰만 이어 붙인 ragged bf16 으로 저장한다
    (`emb (sum_L,4096)` + `offsets`). d200 실측: 51,610 캡션 → **43,252 고유**, 평균 22.7 tok.
    128 슬롯을 다 채우면 27 GB 인데 ragged 는 **9.6 GB** — 나머지는 어차피
    `key_padding_mask=~text_mask` 로 지워지는 자리다.
  - 읽기는 `models/t5.py:CachedT5TextEmbeddings` — `T5EncoderModel` 과 **같은 시그니처/같은
    반환**이라 호출부는 안 바뀐다. `text_emb_cache_path: null` (기본) 이면 이 코드는 아예 안
    돌아 예전 경로 그대로다. 캐시에 없는 캡션은 **원래 인코더(fallback)로 그 자리에서**
    인코딩하고 첫 miss 만 로그에 남긴다 — 캐시 부족으로 학습이 죽거나 결과가 달라지지 않는다.
  - 유효 토큰 값은 패딩 길이와 무관하다 (umt5 는 절대 위치 임베딩이 없고 padding key 는
    마스크로 지워진다). 즉 캐시 경로는 **유효 구간에서 기존 경로와 같은 값**이다.
- **`main/convert_geo_raw_cache.py`: geo raw 캐시 dtype 재굽기 (2026-09-19).**
  `geo_raw_cache_dir` 의 `raw` 텐서를 다른 dtype 으로 캐스팅해 새 디렉토리에 다시 쓴다.
  **DA3 재추론이 아니다** — 파일 안의 feature 를 dtype 만 바꿔 복사한다 (`geo_idxs`/`meta` 는
  그대로). 8워커 실측 10,169편 **5.8분**, fp32 403 GB → bf16 **201.1 GiB** (err 0).
  `--stage verify` 는 개수·키·dtype 전량 + 표본 값 대조를 돌린다.
- **`step_timing`: 학습 스텝의 단계별 실측 시간 (기본 off) (2026-09-18).**
  "왜 이리 느린가"를 추측이 아니라 숫자로 답하기 위한 진단 스위치. `conf/config.yaml` 의
  `step_timing: 0` 이면 측정 코드가 아예 안 돌아 기존 run 과 동작·속도가 같다. `N>0` 이면
  N 스텝마다 `data / h2d / text_vae / geo / fwd / bwd_opt / log` 평균 ms 와 비중을 찍는다.
  구간 경계마다 `torch.cuda.synchronize()` 가 걸리므로(비동기 GPU 를 정직하게 재려면 필수)
  **켠 run 은 평소보다 느리다 — 상시 사용 금지, 진단 전용.**
  - 배경: `py-spy` 가 `record`/`top` 모두 "Permission Denied" 로 실패한 원인은 권한이 아니라
    `/proc/sys/kernel/yama/ptrace_scope = 1` 이었다 — **남의 프로세스엔 못 붙고 자식으로
    띄우면 붙는다.** 자식으로 띄운 프로파일은 되지만 프로파일러가 프로세스를 멈춰가며 읽어
    2.46 s/it (실제 0.263 s/it) 로 왜곡되어 절대 ms 를 못 쓴다. 그래서 sync 기반 계측을 넣었다.
  - D200 arm ① (`dynpose_d200_da3`, B=8, GPU 4) 실측: `data 175.7ms 62.0% | text_vae 42.5ms
    15.0% | bwd_opt 37.3ms 13.1% | fwd 26.2ms 9.3% | geo 1.1ms 0.4% | h2d 0.2ms | log 0.4ms
    || total 283.4ms`. arm ② (`_molmo2_da3`): `data 321.2ms 66.5% | text_vae 48.2ms 10.0% |
    bwd_opt 68.1ms 14.1% | fwd 43.9ms 9.1% | geo 0.9ms 0.2% || total 483.0ms`.
  - py-spy 워커 스택 분해(8 워커): `collate_fn (base.py:189)` = `torch.stack` 56.4%,
    `_share_fd_cpu_ (torch/storage.py:526)` 40.6%, **`np.load` 0.1%**. 병목은 디스크가 아니라
    `geo_raw` fp32 캐시(씬당 42.5 MB, 배치 8 = 340 MB)의 **메모리 복사 2회**다.
- **`geo_raw_cache_dtype` 설정값 (기본 `float32`) (2026-09-19).**
  캐시를 어느 dtype 으로 구웠다고 볼지 고른다. 기본값은 예전과 **글자 그대로 같다**.
  `bfloat16` 으로 두면 bf16 캐시를 그대로 받아 collate 의 memcpy 를 절반으로 줄인다
  (소비 지점이 이미 `.to(device).float()` 이라 텐서는 그대로 통과한다). D200 두 arm
  yaml 에 배선했다 — 나머지 3 arm 은 `dynpose_d200_molmo2_da3` 를 defaults 로 물려받는다.
  ⚠ bf16 을 고른 arm 은 fp32 on-the-fly 와 **비트 동일하지 않다** (실측 상대오차 3.3e-3).

### Fixed
- **`eval_testset.py` 가 readout arm 의 모델 kwargs 를 안 넘김 (2026-09-28, R81).** `peav_readout_layers>0` 인 run 은 strict load 에서 죽었다 — 학습 진입점과 같은 kwargs 추가.
- **`scripts/eval_testset.py` srccam arm 로드 (2026-09-25, R55 / tasks A3).** `srccam_cond` 생성자 인자
  (`geo_latent_dim` 54/11, `geo_in_mlp`, `geo_pe`)를 train 과 같게 넘긴다. 없으면 `geo_proj.0/2.*` unexpected 로
  strict load 가 죽었다. srccam 이 아닌 arm 은 무변경.
- **`.latentcam_index` 캐시가 코퍼스 재굽기를 못 잡아 옛 세그먼트(캡션↔변이 어긋남)를
  재사용하던 것 (2026-09-21).** `dataset_dl3dv.py _load_index()` 의 캐시 키는 **설정만** 본다
  (`meta_csv` / `nf<num_frames>` / `bo` / `k` / `cb` / `ms` / `bl` + `__ps`/`__as`/`__pf`) —
  **내용 해시가 없다.** 그래서 같은 세대 이름으로 변이를 다시 구우면 옛 인덱스가 그대로 붙는다.
  유일한 staleness 검사는 `scene_dir_list[0]` 디렉토리 존재 여부뿐이었다.
  - 실측(D229 lady-running): 변이 2개 시절 캐시(20:54)가 변이 3개 코퍼스(20:59)에 재사용돼
    `seg_list_d229_test.txt` 3줄인데 `eval_meta.json n_heldout_segments: 2`,
    `eval_s42.log:13 [seg-list split] train 0/0 , val 2/3 (ids missing from dataset skipped)`.
    더 나쁜 건 살아남은 2개도 `_0`/`_1` 이름만 같고 **다른 변이의 캡션**을 달고 있었다.
    eval 은 rc=0, 릴도 그려져서 `n_heldout_segments` 말고는 흔적이 없다. `--force` 로도 안 고쳐진다.
  - `_prompts_newer_than(cache_path)` 추가 — 인덱스에 든 scene 의 `prompts.json` mtime 이
    캐시보다 새로우면(또는 파일이 사라졌으면) 캐시를 버리고 다시 빌드한다. 비용은 scene 당
    `stat` 1회(인덱스 빌드는 같은 파일을 json 으로 여니 10배). `cfg.index_cache_mtime_check=False`
    로 끌 수 있고 기본 True.
  - 검증: `[index cache] STALE (prompts newer than cache: …)` → `[index build] 3 samples / 1 scenes`
    → `val 3/3` → `n_heldout 3`.
- **변이가 전부 `aim=free` 인 씬에서 molmo2 캐시가 `KeyError` 로 죽었다 (2026-09-21).**
  `cache_molmo2_embeddings.py` 의 `by_scene` 은 `FREE_PAIR = ('', '')` sentinel 을 **제외하고**
  만들어진다 — free 변이는 씬을 가리지 않고 0 행 하나로 모이기 때문이다. 그런데 씬 루프는
  `keys` 전체를 돌면서 `by_scene[sk]` 를 바로 찍어서, look_at 변이가 **하나도 없는** 씬이 오면
  `KeyError: 'vista4d_golf'` 로 캐시 단계가 통째로 날아갔다 (D224 golf 는 `pedestal_down` /
  `pedestal_down_dolly_in` 둘 다 free). d215 까지는 씬마다 look_at 변이가 최소 하나 있어서
  안 드러난 경로다.
  - `caps = by_scene.get(sk, [])` 로 캡션 루프만 0회로 돌리고, 씬의 **영상** 캐시
    (`<video_out>/<scene_key>.pt`) 는 그대로 굽는다. `dataset_dl3dv._preload_peav` 는 씬이
    하나라도 없으면 FileNotFoundError 로 죽고 즉석 계산 경로가 없어서, 캡션 0개인 씬도
    영상 캐시는 반드시 있어야 한다.
  - prefix 추출용으로 `PREFIX_PROBE_CAPTION` 더미 문장을 추가했다. prefix 는 `split_prefix`
    가 캡션 꼬리를 자른 앞부분이라 문장 내용과 무관하고 (씬마다 동일함을 assert 가 확인),
    이 문장이 들어간 캡션 행은 **저장되지 않는다**.
  - look_at 변이가 있는 씬은 `caps[0]` 을 그대로 쓰므로 **동작이 비트 동일**하다.
- **test 세그먼트가 1개면 FCD 가 MKL 레벨에서 프로세스를 죽였다 (2026-09-21).**
  `FrechetCLaTrDistance.compute` 의 공분산 분모가 `N-1` 이라 N=1 이면 cov 가 NaN 이 되고,
  NaN 행렬이 `torch.linalg.eigvals` 로 들어가면 예외가 아니라 **`Intel oneMKL ERROR:
  Parameter 3 was incorrect on entry to DGEBAL` + rc=-11 (SIGSEGV)** 로 `src.eval_only`
  자체가 날아간다 — CLaTr/caption 결과도 같이 잃는다. FCD 는 두 분포 사이 거리라 N=1 에선
  정의가 없으므로 **NaN 으로 비우고** 넘어간다. N>=2 는 코드 경로가 그대로다.
- **test 세그먼트가 4개 미만이면 PRDC 가 raise 해서 `metrics.json` 이 통째로 안 나왔다 (2026-09-21).**
  `ManifoldMetrics.compute` 가 `N < manifold_k+1` 일 때 `ValueError` 를 던졌는데, 이 예외가
  `src.eval_only` 를 죽여서 **CLaTr(`clatr_score`)·FCD·caption fscore 까지 같이 날아간다**.
  "영상 하나 + preset 하나" 요청(D219: test 1 세그먼트)은 항상 이 경로를 타고, 정작 보려는
  지표는 PRDC 가 아니다. 이제 raise 대신 **PRDC 4개 지표만 NaN** 으로 두고 경고를 찍은 뒤
  나머지 지표를 계속 잰다. `N >= manifold_k+1` 인 기존 arm 은 코드 경로가 그대로라 영향 없다.
- **`cache_umt5_embeddings.py --source prompts` 가 dynpose 코퍼스에서 0건이었다 (2026-09-19).**
  glob 이 `<root>/*/*/<prompts_file>` 2단계 고정이라 vista4d 레이아웃에만 맞고 dynpose
  (`<root>/dynpose/<scene>/da3/prompts.json`) 는 한 단계 깊어 아무것도 못 찾았다. 그런데 빈
  리스트를 그대로 돌려주고 호출부가 CFG 용 빈 캡션 하나를 더 붙이므로 로그가
  `1 captions -> 1 unique` 로 찍힌다 — "구웠다"고 믿으면서 10 KB 짜리 전량-miss 캐시가 남는다
  (D201-A 준비 중 실제로 한 번 그렇게 구웠다). 두 깊이를 다 훑고, 그래도 0건이면 멈춘다.
  경로가 겹치지 않아 vista4d 결과는 비트 동일. `--source index` (기본) 경로는 무변경.
- **bf16 geo raw 캐시를 loader 가 전량 무시했다 (FIX-D200-c) (2026-09-19).**
  `dataset_dl3dv.py` 의 두 곳(`_preload_geo_raw`, `__getitem__` lazy 경로)이
  `c['raw'].dtype != torch.float32` 로 fp32 를 하드코딩하고 있어, 손잡이 A 로 다시 구운
  202 GiB 캐시를 통째로 버리고 10,169 scene 전부를 on-the-fly DA3 로 돌렸다. 학습은 죽지
  않고 **느려지기만** 해서 tqdm 만 보면 안 잡힌다. `_geo_raw_dtype()` 로 판정을 통일하고
  위 `geo_raw_cache_dtype` 를 따르게 했다. `dataset_cfg.py` 의 기동 print 도 실제 dtype 을
  찍도록 고쳤다 (전에는 무조건 "fp32" 라고 적었다).
- **umt5 prefill 캐시가 인코더보다 느렸다 — OMP 스레드 124개 (FIX-D200-d) (2026-09-19).**
  캐시를 켜자 `text_vae` 가 42.5 → 106.6 ms 로 **커졌다**. 캐시 miss(0건)·GPU 경합·비동기
  계측·Lustre I/O 를 차례로 배제했다 (8.2 GiB 를 RAM 에 전량 올려도 같은 105 ms).
  범인은 `CachedT5TextEmbeddings.__call__` 의 텐서 복사다 — ATen 의 CPU `copy_`/`zero_` 는
  numel > 32768 이면 `at::parallel_for` 로 넘어가는데 이 기계는 `torch.get_num_threads()` 가
  **124** 라, 25×4096 한 줄을 복사하는 비용이 memcpy 가 아니라 **스레드 디스패치**다
  (같은 호출을 `set_num_threads(1)` 로 재면 139.43 → 2.40 ms).
  `models/t5.py` 에서 할당(`np.zeros`)과 복사를 numpy 로 돌렸다 — bf16 은 numpy dtype 이
  없으므로 같은 2바이트 int16 으로 view 해 **비트 그대로** memcpy 한다. 결과 warm median
  **1.75 ms**, 무작위 캡션 2.46 ms, 출력 bit-exact 확인.
  `set_num_threads(1)` 로 안 고친 이유: 그건 프로세스 전역이라 DataLoader 워커·collate 까지
  같이 1스레드가 된다. numpy 경로는 이 함수 안에서만 닫힌다.

### Changed
- **학습 상한을 100 → 50 epoch 으로 (2026-09-19, 사용자 지시 "학습 50epoch까지만 돌려도 될
  것 같아").** `conf/config.yaml` 의 `epoch_cap: 100 -> 50`. 루프가 `range(start_epoch, 50)`
  이라 **마지막 epoch 인덱스는 49**(총 50 epoch)이고, 그래서 experiment yaml 의
  `ckpt_at_epochs: [50, 100]` 은 이제 한 번도 안 걸린다 — **최종 모델은 `last.pth` /
  `resume.pth`**(epoch 49)다. `epochs` 를 덮는 experiment yaml 73개와 무관하게 여기서 잘린다.
  ⚠ `epoch_cap` 은 **프로세스 기동 시점에** 읽힌다. 이미 돌고 있는 run 에는 안 먹으므로
  D200 ①~④ 는 `tmp/d200/stop_at_epoch.py` 로 바깥에서 SIGINT 했다 (로그에 `Epoch 50 |`
  tqdm 이 뜨면 = epoch 49 의 val·CLaTr·ckpt 저장까지 끝난 시점).
- **D200 CLaTr 게이지를 s91 실물로 교체 (2026-09-19 03:20).** `dynpose_d200_da3.yaml` /
  `dynpose_d200_molmo2_da3.yaml` 의 `clatr_ckpt_path` 가 `clatr_dynpose_d200_s91_epoch149.ckpt`
  **심링크**를 가리키고 있었고, 그 심링크의 실체는 옛 오염 게이지(`clatr_dynpose_d200_epoch149.ckpt`,
  wandb `bf7xk542`)였다 — s91 게이지 학습이 끝날 때까지 쓰던 임시 배선이다. s91 완료
  (wandb `6a5oy2vw`, `epoch=149-step=217050`) 후 실물을 `clatr_dynpose_d200_s91_real_epoch149.ckpt`
  로 복사해 두 yaml 이 그 파일을 직접 가리키게 했다 (`_real_` 은 심링크가 아님을 이름에서 보이려는 것).
  arm ③④⑤ 는 `defaults: - dynpose_d200_molmo2_da3` 로 물려받는다.
  ⚠ **이 지점에서 FD/PRDC/clatr_score 곡선이 갈린다 — 모델이 변한 게 아니라 자가 바뀐 것이다.**
  교체 전 구간의 caption 지표는 버릴 것: 옛 게이지는 새 test 5,144대 중 ~4,600대를 이미 봤다.
- **D200 분할을 train:test = 9:1 로 재작성, validation 은 test 안의 1% (2026-09-18, 사용자 지시
  "validation은 1%로 하되 train:test는 9:1이어야해").**
  기존 holdout 은 100씬 / 506대 = **0.98%** 였다. "씬 수준 지표의 n 을 27 → 100 으로"가 목적
  이었지 비율이 목적이 아니었는데, 이제 비율이 요구사항이다. 기동 12분차의 4 arm 을 중단하고
  (wandb `n78mgkfi`/`55icy0gk`/`ulqm1j3t`/`1f4hu8p5` 폐기) 새 목록으로 재기동했다.
  - `CinemaTraj/scripts/make_holdout_split.py` **신규** — 기존 seg_list 둘을 읽어 다시 나누기만
    한다. **재굽기 없음**: `vista4d_bank_to_dl3dv.py:493-519` 의 `--test_videos` 는 각 줄이
    어느 seg_list 로 가는지만 정하고 chunk/npz 는 안 건드리며, geo_raw(403 GB)·molmo2(305 GB)
    캐시도 씬 단위라 그대로 쓴다.
  - **씬당 카메라 수로 층화**해 뽑는다. d200 분포가 1:726 / 5:3287 / 6:5362 로 양극단이라
    씬을 무작위로 뽑으면 test 의 씬당 대수가 train 과 달라진다. 실측 대/씬 train 5.057 vs
    test 5.058. 결과: train 46,278대 / 9,152씬 (90.0%), test 5,144대 / 1,017씬 (10.0%),
    그 안의 val 516대 / 102씬 (1.0%). step/epoch 6,364 → 5,784.
  - legacy **27씬 ⊂ 100씬 ⊂ 1,017씬** 을 강제했다 — `test27` / `test100` 부분집합 지표로
    D137/D194/D197 과 계속 대조할 수 있다.
  - validation 은 별도 목록 파일이 아니다. `train_latent_cam_dm.py:737` 이 test 로더를
    `val_max_batches × batch_size` 만큼만 돌기 때문에, val 102씬을 test 목록 **맨 앞**에 씬
    라운드로빈으로 깔고 `val_max_batches: 20 → 64` (×8 = 512대 = val 516대의 99.2%) 로 그
    블록을 덮는다. ⚠ 두 값은 **같이 움직인다** — `val_max_batches` 를 되돌리면 val 이 1% 가
    아니게 된다. D200 root yaml 2개에 주석으로 박아뒀다.
- **D200 CLaTr 게이지를 새 train split 으로 다시 구웠다 (`clatr_dynpose_d200_s91`).**
  예전 게이지(`clatr_dynpose_d200_epoch149.ckpt`, wandb `bf7xk542`)는 train 50,915 로 학습해서
  **새 test 5,144대 중 ~4,600대를 이미 본 상태**다. 게이지가 test 를 알면 caption 지표가
  부풀고 그게 조용히 일어난다. prep train 46,277 / test 5,144 (diverged_pose 1대 탈락),
  `dataset/standardization=dynpose49_d200` 재사용(`trajectory_dataset.py:43` 이
  `standardize=False` 라 실제로 걸리는 건 num_cams=49 뿐 → 분할과 무관).
  root yaml 2개의 `clatr_ckpt_path` 를 `clatr_dynpose_d200_s91_epoch149.ckpt` 로 돌렸다.
  ⚠ ckpt 가 생기기 전까지 CLaTr eval 만 **조용히 건너뛴다** (학습은 안 죽는다) — 재분할 직후
  몇 epoch 의 caption 지표가 비는 것은 그 때문이다.

### Added
- **`geo_raw_cache_mmap` / `peav_cache_mmap` — 대형 per-scene 캐시를 RAM 대신 mmap 으로 preload
  (2026-09-18, D200 5-arm 기동 중 OOM 직전에 잡았다).**
  `dataset_dl3dv._preload_geo_raw` / `_preload_peav` 는 캐시 파일을 __init__ 에서 **전량 anonymous
  RAM** 으로 올린다 (fork 로 뜨는 DataLoader worker 가 copy-on-write 로 공유하게 하려는 설계).
  두 함수의 크기 주석이 각각 **Vista4D 52 scene** / **PE-AV 264 scene** 기준이라 스케일이
  10,169 scene 으로 커진 걸 못 따라갔다 — d200 의 `geo_raw_cache_da3` 는 **403 GB**,
  `molmo2_cache/video` 는 **305 GB** 다. arm 4개면 합계가 RAM 1,771 GB 를 넘긴다.
  - 두 스위치 다 **기본 `false` = 예전과 글자 그대로 같은 동작**. `true` 면 같은 자리를
    `torch.load(..., mmap=True)` 로 읽어 page cache **한 벌**을 arm 끼리 공유하고, 압박 시
    커널이 OOM 대신 회수한다.
  - `peav_cache_mmap` 은 video 디렉토리와 text 파일(`prefill.pt` / `text.pt`) 양쪽에 걸린다.
  - 실측(10,169 파일 전량): `torch.equal` **True**(`emb`/`emb_l21` 둘 다) · RSS 428→**441 MB**
    (논리 163.3 GB) · vma 1,111→11,281(파일당 ~1, `vm.max_map_count` 65,530 이내) · fd 4 고정.
  - D200 root yaml 2개에서 켠다 (`_da3` 는 geo 만, `_molmo2_da3` 는 둘 다 — 나머지 3 arm 은
    `defaults` 로 물려받는다). 자세한 경위는 `FIX.log` 2026-09-18 항목.
- **D200 코퍼스로 학습한 CLaTr 게이지 + `clatr_standardization` 설정 (2026-09-17, 사용자 지시
  "CLaTr도 이 데이터셋에 대해서 학습 돌려놔줘. 이걸로 학습 때 validation 평가해주고").**
  지금까지 학습 중 caption 지표(FD / PRDC / clatr_score)는 `clatr_epoch139_large.ckpt` 로 쟀는데
  그건 E.T./ArtTraj 게이지다 — `configs/config_eval.yaml` 의 `dataset: traj+caption_eval` 이
  `standardization: '0120'`(**num_cams 120**)을 물고 있어서, 49프레임짜리 우리 궤적 뒤에
  71프레임 zero-pad 를 붙여 재고 있었다. 세 조각을 더한다:
  - `scripts/prepare_clatr_vista.py --stage all` 로 d200 코퍼스를 CLaTr 레이아웃으로 변환.
    train **50,915** / test **506** (diverged_pose 1대 탈락, `error.txt` 에 기록), 전부 49프레임.
  - `main/evaluate/CLaTr/configs/dataset/standardization/dynpose49_d200.yaml` — `--print_stats`
    실측값. d137 대비 `norm_std` 가 `[0.0704, 0.0152, 0.0873]` → `[0.0657, 0.0325, 0.0708]` 로,
    특히 **y(수직) 성분이 2.1배**다. `shift_std` 실측은 `[3e-8, 1e-8, 4e-8]`(frame0 world 앵커)
    이라 dynpose49.yaml 과 같은 이유로 `[1.0, 1.0, 1.0]` 으로 둔다.
  - `conf/config.yaml` 에 **`clatr_standardization`** (기본 `null`) 추가 +
    `train_latent_cam_dm.py` 의 `src.extraction` argv 에 `dataset/standardization=<이름>` 을
    조건부로 붙인다. **null 이면 argv 가 그대로라 기존 동작(0120, num_cams 120) 유지.**
  학습: wandb `trajectory-clip/bf7xk542` (`xp_name=clatr_dynpose_d200`, 150 epoch,
  1,592 step/epoch, GPU 4). d200 5-arm experiment yaml 2개(root)에 `clatr_ckpt_path` +
  `clatr_standardization` 배선 — 나머지 3개는 `defaults` 로 물려받는다.
  ⚠ **자가 바뀌었다. D200 run 의 FD / PRDC / clatr_score 는 D194/D197 및 그 이전 어느 숫자와도
  비교 불가**다. 같은 ckpt 를 쓴 run 끼리만 비교할 것.
- **`cache_molmo2_embeddings.py --joint` — prefill/decode arm 4개를 forward **한 번**에서 뽑는다
  (2026-09-17, D200 사용자 지시 "prefill 단계의 l21/last 와 decode 단계의 l21/last 를 한번에").**
  `--decode_tokens` 경로의 **첫 forward 가 곧 prefill** 이다(KV 캐시를 채우는 그 forward).
  지금까지는 `out.last_hidden_state[:, P:]` 와 `tap.h[:, P:]` 를 버렸는데, 버리지 않고 같이
  저장하면 prefill last / prefill l21 / decode last / decode l21 넷이 한 forward 에서 나온다 —
  prefill 캐시를 따로 굽는 것은 같은 계산을 두 번 하는 것이었다. decode 경로는 **mid padding**
  이라 `mid_to_right()` 로 `tail_hidden` 과 같은 right padding 으로 옮겨 담는다(위치가
  `P..P+len-1` 로 같고 attention mask 가 pad 를 지우므로 값이 같아야 한다). `--verify` 가
  `tail_hidden` 과 직접 대조한다 — 실측 relL2 7.9e-3 / **cos 0.999968** (B=1 과 padding 강제
  양쪽 동일), 즉 bf16 배치 리덕션 잡음 수준이다.
  저장은 **파일 둘**로 가른다: `--text_out`(decode, 슬롯 49) / `--prefill_out`(prefill, 슬롯 128).
  한 파일에 넷을 담으면 `torch.load` 가 부분 로드를 못 해서 arm 하나가 자기가 안 쓰는 수십 GB 를
  같이 로드한다. 하류가 바꾸는 것은 `peav_text_cache` 경로 하나뿐이고 `peav_layer` 배선은 그대로다.
  ⚠ 두 arm 이 **같은 fps** 를 쓰게 된다 — `--decode_keep points` 는 `--fps 2.0` 이 한 벌이라
  prefill 도 fps 2.0 으로 구워진다(기존 prefill arm 은 25.0). prefix 토큰열이 4323 -> 4341 로
  달라지므로 D200 은 **video 캐시도 새로 굽는다**(d194/d197 심볼릭 링크 재사용 없음). 대신
  프롬프트가 같아져 두 arm 의 차이가 "읽는 자리"(prompt 위치 vs 생성 위치) 하나로 좁혀진다.
  `--joint` 를 안 주면 저장물은 기존과 **비트 동일**하다.
- **`cache_molmo2_embeddings.py --free_mode zero` — `aim=free` 변이를 zero-embedding 으로
  (2026-09-17, D200 사용자 지시).** free-moving 변이에는 겨냥하는 물체가 없어
  `Track {target}` 문장이 성립하지 않는다(D200 실측 13,777/51,422 = 26.79%). 그 변이들의 text
  hidden 을 **굽지 않고** 전량 0 행 하나(`FREE_PAIR = ('','')`)에 묶고 mask 를 False 로 둔다.
  forward 는 **0회**다 — 영상 hidden 은 씬당 1파일이라 캡션과 무관하게 이미 구워져 있어서,
  빈 문자열로 prefill 을 한 번 더 돌려도 새로 얻는 것이 probe 토큰 자리 hidden 뿐이고 학습은
  그걸 mask 로 지운다. decode 도 sentinel 이 `by_scene` 에 안 들어가므로 자동으로 건너뛴다.
  **학습 쪽 모델 코드는 고치지 않았다** — `camera_diffusion_model_latent._build_video_tok` 이
  이미 `tt * video_text_mask` 로 0 을 만들고 `~cat([tm, vm])` 로 key-pad 까지 한다. 같은 행에
  video 토큰 3136개가 남아 all-masked row NaN 도 안 난다. `off`(기본)이면 `aim` 을 아예 안 보므로
  기존 세대와 **비트 동일**하다. 5씬 스모크: free 4 seg -> 행 0 공유, `|emb| 0.000` / mask 0.
  `cache_peav_embeddings.collect()` 가 `aim` 을 같이 실어 나르도록 한 줄 늘었다(기존 소비자는
  이 키를 안 읽으므로 동작 동일).
- **`scripts/data/cache_geo_raw_da3.py` 에 `SHARDS`/`SHARD` 샤딩 (2026-09-17).** 씬당 파일
  하나라 샤드끼리 같은 파일을 안 건드린다. 10k 규모 코퍼스(D200 10,169씬 = ~432 GB)를 GPU
  한 장으로 굽는 데 몇 시간이 걸려서 넣었다. 분할은 **정렬된 전체 목록** 위에서 하고
  (skip-done 을 먼저 적용하면 재실행 때 경계가 흔들린다), 기본값 1/0 이면 기존 동작과 동일하다.
- **D200 실험 yaml 5종 (`conf/experiment/dynpose_d200_*.yaml`).** `da3` /
  `molmo2_da3`(prefill last) / `molmo2_l21_da3` / `molmo2_dec_da3`(decode last) /
  `molmo2_dec_l21_da3`. 코퍼스는 d185+d199 pooled(씬당 상한 6, `--drop_status clamped_low` 만),
  모델축은 D197 세대에서 글자 그대로 복사했다. molmo2 arm 넷은 `--joint` 로 구운 **같은
  forward** 의 캐시를 읽으므로 움직인 축이 "읽는 자리 × 층" 둘뿐이다.
- **`cache_molmo2_embeddings.py --decode_keep points --decode_points 49` — decode hidden 슬롯을
  "생성 토큰 순번"이 아니라 **영상 프레임**으로 바꾼다 (2026-09-16, D197-d2 사용자 지시
  "49프레임 다 나오도록").** `--decode_tokens` 만 주면 슬롯 t 는 t번째 생성 토큰이라 영상과
  아무 관계가 없다. `Track {target}` 프롬프트에 Molmo2 는 점마다 **시각**을 단
  `<tracks coords="0.0 1 444 461;0.5 1 444 453;…">` 로 답하므로, 점 하나가 끝나는
  토큰(`;`, 마지막 점은 coords 를 닫는 `"`)의 hidden 만 골라 `round(t*fps)` 프레임 슬롯에
  담는다. 슬롯이 곧 프레임이라 video 토큰의 49x8x8 격자와 축이 맞고, 저장량은 49슬롯이라
  전 토큰(~800)의 1/16 (13,993쌍 x 2층 = ~7 GB; 전 토큰이면 ~117 GB 였다).
  **점 개수를 정하는 건 `--fps` 다.** `processing_molmo2.get_video_string` 이 프레임마다
  `f"{t:.1f} "` 를 박는데 모델은 0.5 s 라벨마다 한 점을 찍는다 — 기본 25.0 에서는 49프레임이
  0.0~1.9 로 뭉쳐 **라벨 20종 / 점 4개**뿐이고(프레임 커버리지 4/49), 4.0 이면 점 25개,
  **2.0** 이면 라벨이 정확히 0.0,0.5,…,24.0 이라 프레임과 1:1 이다. 실측 3씬에서 점
  48/49/49, 생성 721~733 토큰에서 EOS 자연 종료 — 그래서 `--decode_tokens 900` 이 한 벌로
  붙는다(640 이면 `<tracks>` 중간에 잘려 파싱 점 0개). 5캡션 스모크: 채운 슬롯 48.6/49,
  파싱 실패 0%, 길이 초과 0%. 처리량 ~42 s/scene (decode 900 스텝이 지배적).
  fps 를 바꾸면 prefix 토큰열 길이가 4323 -> 4341 로 달라지지만 **video 캐시는 재사용**한다
  (`--skip_done` 기본 True) — 바꾸는 게 text 스트림 하나여야 arm 사이에서 video 가 비트
  동일하게 남는다. `--decode_keep all`(기본)이면 저장물은 기존과 **비트 동일**하다.
  `tmp/d197/probe_track_fps.py` 가 fps↔점개수 실측 + 소스 영상 위 track 점 오버레이 mp4 를 낸다.
- **`cache_molmo2_embeddings.py --decode_tokens N` — prefill 대신 decode 스텝 hidden 을 굽는다
  (2026-09-16, D197-d 사용자 지시).** 기존 경로가 담는 것은 캡션+probe 가 놓인 **prompt 위치**의
  hidden 이다 — 모델이 답을 아직 한 토큰도 커밋하지 않은 상태다. `--decode_tokens N` 을 주면
  greedy 로 N 토큰을 실제로 생성하면서, `y_t` 를 입력으로 넣고 나온 hidden 을 슬롯 t 에 담는다.
  `y_0` 를 고른 prefill 마지막 위치 hidden 은 **버린다** — 그게 "prefill hidden 은 버린다"의
  경계선이다. `emb_l{N}`(`--extra_layer`) 도 같은 슬롯 기준이라 층 대조가 그대로 성립하고,
  생성문은 `decode_text` 로 같이 저장한다. 안 주면 저장물은 기존과 **비트 동일**하다.
  배치 레이아웃이 기존 `tail_hidden` 과 다르다: right padding 이면 아이템마다 꼬리 끝 열이 달라
  decode 를 한 번에 못 돈다(`cache_position` 은 배치 공용 1D). 그래서 **mid padding**
  `[prefix(P)][pad(L-len_i)][tail(len_i)]` 로 꼬리 끝을 한 열에 정렬하고 `position_ids` 를
  아이템마다 따로 준다. `--verify` 가 코퍼스 최장 꼬리를 일부러 끼워 padding 을 강제한 뒤
  B=1(패딩 0) 결과와 대조한다 — 실측 relL2 8.6e-3 / cos 0.999963, 갈린 슬롯은 좌표 숫자 한 개
  (`464` vs `461`) 뿐이라 bf16 배치 리덕션 잡음이다. prefill 대조는 relL2 0.444 / cos 0.899 로
  슬롯이 실제로 다르다. 처리량 실측 5.3 s/scene (prefill 2.55 s/scene 의 2.1배).
  ⚠ 우리 문장이 `Track {target}.` 이라 Molmo2 는 이걸 **자기 native tracking 과제**로 받아
  카메라 서술이 아니라 2D 물체 트랙 XML 을 낸다 (`<tracks coords="0.0 1 456 453;0.5 1 552 453;
  1.0 1 398 461;1.5 1 445 589">the man in a light grey shirt…</tracks>`, 77/128 토큰). 즉 이
  arm 의 decode hidden 은 "카메라 궤적 서술"이 아니라 **타깃의 시간축 화면 좌표 읽기** 중의
  표현이다. 텍스트를 arm ②/③ 과 글자 단위로 같게 둬야 prefill↔decode 만 갈리므로 문장은
  안 바꿨다.
- **D197-d 실험 config 2종 (2026-09-16).**
  `dynpose_d197_molmo2_dec_da3.yaml` (마지막 층) / `dynpose_d197_molmo2_dec_l21_da3.yaml`
  (`blocks[21]`). 둘 다 `dynpose_d197_molmo2_da3` 를 상속하고 `peav_text_cache` 를
  `molmo2_cache/text_decode_pts49.pt` 로만 바꾼다 — **video 캐시는 소스 영상만 보므로 arm ②/③ 과
  같은 파일을 그대로 읽는다** (재굽기 없음). 캐시 이름이 `text_decode.pt` -> `text_decode_pts49.pt`
  로 바뀐 것은 위 `--decode_keep points` 채택 때문이다 (슬롯 128 토큰 -> 49 프레임).
- **PE-AV/molmo2 readout transformer + subject OBB 보조 손실 (2026-09-15, D195-A 사용자 지시).**
  `peav_readout_layers: 0` (기본) 이면 모듈이 생성조차 안 되고 state_dict·forward 가 기존 arm 과
  **비트 동일**하다 (실측: 키 집합 일치, 같은 seed 로 `torch.equal(y_off, y_new) == True`).
  켜면 캐시 hidden state 가 `LayerNorm+Linear` 하나로 video CA 에 바로 가는 대신, `num_queries`
  개 learnable query 가 `[self-attn → cross-attn → MLP]` 를 `peav_readout_layers` 번 돌아 만든
  요약이 CA 의 key/value 를 **대신**한다 (`PeavReadout`, `camera_diffusion_model_latent.py`).
  query 수 기본 49 = 소스 프레임 수라 query i 가 프레임 i 에 대응하고, 부수 효과로 video CA
  key/value 가 3264 → 49 로 줄어든다. 입력은 `[video_text | video]` 합본이다 — caption 토큰을
  같이 먹여야 query 가 "어느 물체가 subject 인지"를 알 수 있고 그게 아래 보조 손실의 전제다.
  `peav_readout_aux_dim: 3` 이면 요약 위에 MLP head 가 붙어 **프레임별 subject OBB center** 를
  맞춘다 (`train_latent_cam_dm.readout_aux_loss`, 가중치 `peav_readout_aux_w`, wandb
  `train/readout_aux_obb`). 타깃은 `target_track_dim` 이 쓰는 것과 같은 `target_track.npz`
  (frame-s 카메라 좌표 / norm_scale) 라 게이지가 cam_param 과 일치하고, `valid=0` 변이는
  손실에서 빠진다. head 출력은 DiT 로 **되돌아가지 않는다** — 실측으로 aux 만 backward 하면
  `out.weight.grad is None`, DiT 손실만 backward 하면 `aux_head.weight.grad is None` 이고
  `readout.query` 에만 양쪽 grad 가 모인다. 즉 추론에서 head 를 안 부르는 것이 곧 "MLP 를 뗀"
  상태다. readout 2층 + aux head 기준 +8.70 M param (41 키). 코퍼스에 `target_track.npz` 가
  없으면 손실이 조용히 `None` 이 되므로, aux 를 쓸 코퍼스엔 `export_target_track.py` 가
  선행돼야 한다 (d121 52/52. dynpose d194 는 처음 0/7801 이었고 2026-09-15 에 전량 생성해
  6,894/6,894 변이 · valid 100% · missing anchor 0 이 됐다).
- **`dataset_dl3dv` 가 `peav_readout_aux_dim > 0` 일 때도 `target_track` 을 내보낸다
  (2026-09-15, D195-A).** 전까지는 `target_track_dim > 0` 일 때만 만들었다. readout+aux arm 은
  조건으로는 track 을 **안 쓰면서**(`target_track_dim: 0`) 보조 손실 타깃으로는 필요하다 —
  둘을 같은 플래그에 묶어 두면 aux 손실이 조용히 `None` 이 되고, 그러면 그 arm 이 l21 대조군과
  같아진 채로 로그에는 아무 흔적이 안 남는다.
- **`make_prompts_simple.py --captions_root` — 뱅크에서 다시 구운 캡션을 코퍼스에 얹는다
  (2026-09-15, D196).** 기존 두 경로(`--fields` / 축약)는 코퍼스 안의 `caption_fields` 를
  **재조립**할 뿐이라, `build_bank_captions.py` 의 플래그(`--magnitude` / `--nl_framing` 등)를
  바꿔 `caption_fields` 자체가 달라진 경우엔 쓸 수 없다. 그 값은 CinemaTraj 뱅크의
  `captions_*.json` 에만 있다. 이 경로는 `<captions_root>/<scene>/<bank_dir>/<captions_name>`
  을 읽어 `<root>/<chunk>/da3/<out_name>` 으로 쓴다 — **전량 re-export(7.8k편, 115편/분 ≈ 68분)
  없이 캡션만 교체**된다. 조인 키는 `variant_id` 다 (`tau_max` 같은 실현치를 키로 쓰면 조용히
  0행 — D191 에 같은 사고가 있었다). 캡션이나 변이를 못 찾으면 끝에서 `raise` 한다
  (`--allow_missing` 으로 완화). 플래그를 안 주면 기존 두 경로가 그대로다.
- **`conf/experiment/dynpose_d196_{da3,molmo2_nogeo,molmo2_l21}.yaml` (2026-09-15, 사용자 지시
  "정도부사랑 framing을 이전과 똑같이 달아주고 3arm 학습").** d194 3 arm 과 **캡션 한 축만**
  다른 사본으로, 본문은 `prompts_file: prompts_mag.json` 한 줄이다. D176 에서 nl 기본 문장이
  `target`+`camera` 두 축으로 줄며 크기 부사와 framing 절이 빠졌던 것을 되돌린다. 코퍼스는
  d194 와 **같은 디렉토리**를 그대로 읽는다 — 6,894 세그먼트 전량 대조로 키 집합 ·
  `frame_idx` · `variant_id` · `target_text` 가 동일함을 확인했고, 그래서 `target_poses.npz` /
  `target_track.npz` / `avg_scale*` / seg list / molmo2 캐시(video 110 GB + text 4.5 GB) /
  `geo_raw_cache_da3` 를 하나도 다시 굽지 않는다. `dataset_dl3dv._load_index` 가 인덱스 캐시
  키에 `__pfprompts_mag` 를 붙이므로 d194 캐시를 조용히 재사용할 수도 없다. `text_len` 은 128
  그대로 — umt5 토큰 실측(6,894 전량)이 `prompts.json` max 35 / `prompts_mag.json` max 76 이라
  잘리는 세그먼트가 0 건이고, 여기서 같이 올리면 캡션축과 시퀀스축이 함께 움직인다.
- **`conf/experiment/dynpose_d19{4,6}_molmo2_l21_{readout,track_d5}.yaml` (2026-09-15, D195).**
  D195 두 arm(readout+aux / track concat dropout 0.5) × 캡션 2종의 4 조합. d196 쪽 둘은 d194
  쪽 둘에 `prompts_file: prompts_mag.json` 한 줄만 얹은 사본이라 모델 하이퍼는 상속만 한다.
- **`scripts/train/queue_runs.py` (2026-09-15, D195 사용자 지시 "학습 queue 걸어놔줘").**
  GPU 풀과 job 큐를 분리해, 비는 GPU 에 앞에서부터 experiment 를 꽂는 대기열 드라이버.
  기존 `scripts/relaunch_after_stop.sh` 는 run 하나 : GPU 하나가 고정이라 어느 arm 이 먼저
  끝날지 모르는 상황에서 먼저 빈 GPU 를 놀린다. 빈 GPU 판정은 **AND 두 조건**이다 —
  ① `nvidia-smi` memory.used 가 임계 밑 (프로세스만 보면 DataLoader worker 가 GPU 를 물고 있는
  창에 꽂아 OOM), ② `/proc/<pid>/environ` 의 `CUDA_VISIBLE_DEVICES` 로 그 GPU 를 선점한 학습
  프로세스가 없을 것 (메모리만 보면 molmo2 arm 이 110 GB 캐시를 읽느라 수 분간 0 MiB 인 창에
  꽂아 같은 GPU 에 두 개가 겹친다 — 2026-09-15 dry_run 에서 실제로 GPU2/3 이 이렇게 찍혔다).
  `--dry_run` 은 지금 꽂으면 어디로 가는지만 출력한다.
- **`cache_molmo2_embeddings.py --text_override_json` / `--probe` (2026-09-14, D191).**
  둘 다 안 주면 D124 이후 기존 동작과 비트 동일하다. `--text_override_json` 은
  `{data_name: text}` JSON 으로 **molmo2 가 읽는 문장만** 갈아끼운다 — 디스크 위 코퍼스
  `prompts.json` 은 안 건드리므로 같은 코퍼스를 쓰는 umt5/T5 arm 은 글자 단위로 그대로다.
  누락 세그먼트가 있으면 assert 로 죽는다 (조용한 부분 적용 방지). 구운 `text.pt` 에
  `template: molmo2_override+probe` 와 `text_override_json` 경로를 같이 싣는다.
  사용자 지시 2026-09-14 "molmo2에는 Track {target with detail} 형식으로 넣은거로 돌려줘".
- **`conf/experiment/dynpose_d189_molmo2_nogeo.yaml` / `dynpose_d189_da3_molmo2.yaml`
  (2026-09-14, D191).** d189 코퍼스(865씬 / 1,034대)를 **파일 단위로 그대로** 읽는 molmo2 두
  arm. 이미 끝난 `dynpose_d189_da3`(geo only)와 합쳐 3-arm 이 되고, 셋 다 같은 코퍼스 ·
  holdout · `text_len` · VAE · `scale_mode` · `intr_norm` 이라 움직인 축은 스트림 구성뿐이다.
  molmo2 텍스트는 `Track {상세 target}.` — 상세 target 은 D190 `instance_desc.json` 을 먹여
  다시 구운 `captions_d191.json` 의 `target_text` 이고, 1,034/1,034 에 붙었다 (라벨 폴백 0).
  probe 는 안 바꿨다 (`Track X` 가 명령문이라 기존 질의문과 그대로 맞물린다).
  두 arm 이 `molmo2_cache/{video,text.pt}` 를 공유하므로 molmo2 입력이 완전히 동일하다.
- **`conf/experiment/dynpose_d189_da3.yaml` — anchor 당 1대로 다시 고른 d188 뱅크 위의 da3 arm
  (2026-09-14, D189).** 모델축 값은 `dynpose_d137_da3` 에서 **글자 그대로** 가져왔다 — 이번
  세대에서 움직인 축은 데이터 하나이므로 모델 하이퍼를 같이 건드리면 원인을 못 가른다.
  코퍼스가 다른 점 셋: ① `lbm/pick.py:pick_rows(per_anchor=True)` 로 (scene, anchor) 마다 1대
  (실측 1.195 대/씬) — `dd10` 같은 preset 비율 필터를 **안 쓴다**, 고르는 일이 pick 에서 끝났다.
  ② 캡션에 크기 부사가 없다 (사용자 지시 "caption (정도부사 빼고)", `--no_magnitude`).
  ③ holdout 이 `--test_hash_mod 20` 해시 분할 — d188 이 굽는 중이라 코퍼스가 자라고, UUID
  접두사 분할은 이미 구운 풀이 정렬 순서로 편향돼 865편 중 513편(59.3%)을 잡았다.
  `geo_raw_cache_dir` 만 d189 전용으로 새로 판다: 캐시 키가 scene 이름이라 target 뱅크와
  무관해 공유해도 맞지만, d129 캐시는 273편뿐이고 여기 865편 중 대부분이 새 씬이라 남의 root
  밑에 590편을 새로 쓰면 d129/d137/d157 이 읽는 디렉토리가 조용히 다른 물건이 된다.
  **부분 코퍼스임을 config 주석에 박아 뒀다** — 865편 / 1,034대는 d188 이 12% 구워진 스냅샷이고,
  완주판(~9,700대 = 10,346 x 0.935 대/씬 **환산 추정치**)과 지표를 나란히 두면 안 된다.
  (2026-09-14 정정: 원래 1,030 대로 적어 뒀었다. 그건 re-pick 실행 시점 로그이고, export
  전에 d188 이 3편을 더 구워 넣어 실제 코퍼스는 1,034 대다 — prompts.json 항목 합으로 확인.)
- **`scripts/viz/molmo2_attn_map.py` — Molmo2 최종층 attention map 시각화 (D164, 2026-09-08,
  사용자 지시 "Molmo2에 비디오, 텍스트 먹였을 때 final hidden state 기준 attention map 시각화해서
  보고싶어" → "map + swap 먼저").** `--stage map,swap,sweep` 단일 python 드라이버.
  · **`output_attentions=True` 를 쓰지 않는다** — L≈4390 / 32 head / 36층이면 fp32 로 43.5 GB 다.
    `modeling_molmo2.eager_attention_forward` 를 감싸 대상 층의 tail 행만 떠낸다
    ((1,32,78,4390) fp32 = 44 MB). 원본 함수를 그대로 호출하므로 수치는 eager 와 동일.
    config 기본이 `sdpa` 라 훅 동안만 `eager` 로 바꾼다 (`modeling_molmo2.py:710` 분기).
  · `swap` 은 코퍼스가 씬마다 이미 만들어 둔 여러 `target_text` 를 쓴다 (camel 5종,
    car-roundabout 11종). 문장·motion·framing 이 전부 같고 **타겟 명사구만** 다른 대조군이
    공짜로 나오므로, 히트맵이 그 물체로 옮겨가는지를 변이 간 코사인으로 잰다.
  · 읽을 때 필요한 보정을 전부 수치로 낸다: attention sink 질량, 9x9 테두리 질량(uniform 39.5%),
    앞 3프레임 질량(uniform 6.1%), head 별 video 질량 분포(GQA 32:8 이라 평균만 보면 뭉개진다).

- **`scripts/viz/molmo2_attn_video.py` — 모델을 GPU 에 올려둔 채 (video, text) 를 받아 attention
  map 오버레이 **영상**을 저장하는 서버 (D164, 2026-09-08, 사용자 지시 "내가 텍스트를 넣으면
  attention map을 시각화한 영상을 저장하는 코드로 그냥 짜줘" → "모델 gpu에 올려두고 video,
  text 보낼때마다 돌려서 저장하도록 못함?").** `--serve stdin` 대화형 / `--serve http` /
  `--serve off` 한 방. 텍스트 토큰 -> video patch attention 을 프레임당 9x9 히트맵으로 올려
  mp4 로 쓴다 (원본|오버레이 나란히 + `(T,9,9)` 원본 `.npy` + 진단 수치 `.json`).
  · **재사용 두 겹.** 모델 로드(17.7s)는 프로세스당 1회. 그리고 **ViT(49프레임 x 729패치 x
    27층)는 텍스트와 무관하다** — chat template 이 `<|video|>` 를 맨 앞으로 올리고 LM 이 causal
    이라 patch 위치는 뒤의 텍스트를 못 본다. 영상당 1회 `merge_visual_inputs` +
    `build_input_embeddings` 로 prefix `inputs_embeds` 를 캐시하고 텍스트마다 tail 임베딩만
    이어 붙인다 (`cache_molmo2_embeddings.scene_prefix`/`tail_hidden` 과 같은 발상).
    실측 **요청당 1.8~2.1s** (영상 교체 시에만 ViT 재실행).
  · `--verify` 로 prefix 재사용의 정당성을 실측한다: prefix emb |Δ|max **0**, tail hidden
    |Δ|max **0**, **attention map 코사인 1.000000** — bit-exact 다.
    · 단, verify 는 **두 경로를 같은 attention 구현으로 맞춰야 한다.** 처음엔 재사용 경로만
      훅(eager) 아래 두고 대조 경로를 sdpa 로 둬서 |Δ|max 1.5 가 나왔는데, 이건 재사용 오차가
      아니라 **eager vs sdpa 의 bf16 누적 차이**였다 (|h|max 210 대비 0.7%). 이 차이는 실재하므로
      **이 진단 수치를 학습 캐시(sdpa 로 구움)와 직접 비교할 때는 감안할 것.**
  · **기본 층 15** — D164 sweep 실측 최적. `--layer -1` 로 최종층.
  · `--span` 으로 query 토큰 구간을 부분 문자열 지정, `--span_tail N` 으로 마지막 N 토큰만,
    `--head N` 으로 단일 head (GQA 32:8 이라 평균은 뭉개진다), `--probe cache` 로 학습 캐시와
    같은 조건, `--frames_dir` 로 코퍼스 밖 프레임 폴더.
  · 요청마다 찍는 진단: video 질량 / sink 질량 / **테두리 질량(uniform 39.5%)** / 앞 3프레임
    (uniform 6.1%) / 프레임 엔트로피 / peak 프레임 / head 별 분포. 테두리가 uniform 을 크게
    넘으면 물체가 아니라 위치 artifact 다.
  · mp4 는 `imageio-ffmpeg` 번들 바이너리로 쓴다 (시스템 `ffmpeg` 는 `infcam` env 에만 있다).
    컬러맵을 numpy 로 직접 만들어 matplotlib 의존을 뺐다.
  · env 는 **`latentcam`** (transformers 4.57.6 + imageio-ffmpeg 0.6.0). `infcam` 은
    transformers 4.46.2 라 Molmo2 remote code 가 안 돈다.
  · 실측 1건씩: `Point to the larger pale camel.` 층15 -> video 질량 40.6%, 테두리 39.4%
    (uniform 39.5% = 위치 artifact 0), f0 peak 가 낙타 몸통. 같은 층 35 -> video 20.3%,
    테두리 62.8%, 앞3프레임 47.8%. `Point to the white golf ball.`(golf 씬) -> f0 peak 가
    클럽 헤드 옆 공 위치에 붙지만 f20(공이 몇 픽셀)에서는 놓친다 — 9x9 격자의 한계.

- **`scripts/viz/molmo2_attn_client.py` — 파일 위쪽 상수만 고쳐 `python` 으로 실행하는 클라이언트
  (D164, 2026-09-08, 사용자 지시 "코드에서 hardcoding해서 video 경로 및 텍스트를 입력으로 주고
  python으로 실행하면 미리 올라가 있는 다른 process로 띄운 모델에 보내서 돌리는형식은 안됨?").**
  `VIDEO` / `JOBS(text, span)` / `LAYER` 등을 파일 상단에서 고치고 실행하면 이미 GPU 에 올라간
  `--serve http` 서버로 보내 mp4 를 받는다. stdlib `urllib` 만 써서 env 무관. 실측 job 당 1.9~2.2s.
  서버가 없으면 띄우는 명령을 안내하고 종료한다.

- **`scripts/viz/molmo2_attn_grid.py` — 텍스트 형식 x 36층 격자 탐색 (D164, 2026-09-08, 사용자
  지시 "text, layer 가능한 조합들 다 job에 넣어서 가장 잘 나오는거 찾아봐줘봐").**
  판정 지표를 새로 세웠다 — "타겟을 바꾸면 map 이 움직인다"만 보면 **노이즈가 이긴다**.
  코퍼스 `anchor_label` 로 물체 클래스를 알 수 있으므로:
    `within`  같은 클래스 쌍의 map 코사인 (낙타↔다른 낙타) — 높아야 함
    `between` 다른 클래스 쌍 — 낮아야 함
    `score = within - between`  (노이즈는 둘 다 낮추므로 score 를 못 올린다)
  층은 forward 1회로 36개가 다 나오고(`Runner.maps_all_layers`) ViT 는 씬당 1회다.

- **`scripts/viz/molmo2_point_track.py` — Molmo2 의 point/track **생성 출력**을 뽑는 대조군
  (D164, 2026-09-08, 사용자 지시 "molmo2 최종 point track output도 나오도록해서 실제로 track은
  되는지 먼저 확인해줘").** 디코딩은 체크포인트 `README.md` 의 `extract_video_points` 를 그대로
  옮겼다. `--attn_npy` 로 attention 히트맵을 주면 생성 점(초록)과 attention argmax(마젠타)를
  같은 프레임에 나란히 그린다. `--prompt` 는 여러 개를 받아 모델 로드 1회로 다 돈다.

- **`scripts/viz/molmo2_latent_sweep.py` — 어느 latent 를 카메라 디코더에 꽂을지 lever sweep
  (D164, 2026-09-08, 사용자 지시 "fps 5로 잡고 가능한 lever들 sweep 돌려서 어느 latent를 쓰는게
  좋을지 판단해줘. attention map을 써고 point head sub patch에 attention이 높은 정보가 있다던지
  이런걸 분석해줘").**
  · **[정정] 이 체크포인트에는 point head 가 없다.** `modeling_molmo2.py` 의 클래스는
    `Molmo2ForConditionalGeneration` 하나뿐이고 `patch_logits`/`subpatch_logits`/
    `location_logits`/`MolmoPoint` 가 **존재하지 않는다** (grep 확인). pointing/tracking 은
    별도 head 가 아니라 **좌표를 텍스트 토큰으로 생성**해서 한다 — README 가 나열한
    `allenai/Molmo2-VideoPoint` 는 별도 아티팩트다. 따라서 subpatch head 를 탭할 수 없다.
    (이전 대화에서 언급한 4-head 구조는 이 체크포인트가 아니다.)
  · 대신 **생성 좌표를 pseudo-GT** 로 써서 질문을 정확히 만들었다: "각 latent 에서 대상의
    프레임별 위치가 얼마나 바로 읽히는가". 학습 파라미터 0개 readout — 프레임 f 에서 텍스트
    쿼리와 그 프레임 81개 patch latent 를 맞대어 9x9 분포를 만들고 argmax 를 예측으로 삼는다.
  · sweep 축: 층 0~35 x query(target 마지막/target 평균/tail 평균) x
    readout(`attn` / `attnv`=attention x ||v|| / `dot` / `cos` / `pc1`,`pc4`=상위 주성분 제거).
    forward 1회로 36층 attention + 37층 hidden 을 동시에 잡고, pc 기저는 층당 1회
    `torch.pca_lowrank` 로 구한다. 8씬 22샘플, fps 5, `--trim_level full`.
  · **설계 함정 하나 기록**: 프레임별 평균 patch 벡터를 빼는 것(`cdot`)은 **프레임당 상수 이동**
    이라 argmax 를 바꾸지 못한다 (실측에서 `dot` 과 소수점까지 동일했다). global 성분 제거는
    상수 shift 가 아니라 **방향 제거**(주성분 제거)여야 한다 — 그래서 `pc1`/`pc4` 로 바꿨다.

### 발견 (D164 latent sweep, 8씬 22샘플, fps 5)
- **기준선: center(항상 화면 중심) 126px / fixed(그 타겟 GT 의 시간평균) 17px / grid(균등) 228px.**
  `fixed` 가 17px 이라는 것은 **물체가 클립 내에서 거의 안 움직인다**는 뜻이고, 이게 진짜로
  이겨야 하는 상대다.
- **어떤 latent 도 파라미터 없는 readout 으로는 프레임별 위치를 쓸 만하게 노출하지 않는다.**
  최고 조합 `tgt_mean / dot / 층0` = **168.2px** (hit@1cell 12%, hit@2cell 29%).
  center(126px) 에도 42px 지고, `fixed`(17px) 보다 **10배** 나쁘다. readout 별 최고:
  `dot` 168.2 / `attn` 199.0 / `cos` 205.0 / `pc4` 207.2 / `pc1` 221.3 / **`attnv` 303.9**
  — ||v|| 가중(Kobayashi et al.)은 여기서 오히려 크게 해롭다.
- **다만 신호가 없는 것은 아니다 — 중앙에서 먼 타겟에서는 이긴다.** 샘플별 best 로 보면
  22개 중 9개가 center 를 이기고, 이긴 것은 거의 전부 주변부 타겟이다:
  cows 물통 241->34.5px, fashion-walk man 214->71.5, basketball man 268->72.3,
  cows 울타리 212->81.3, couple-rocks 이끼 205->82.8, camel 울타리 201->105.5.
  진 것은 전부 화면 중앙에 앉은 주 피사체다 (camel center 15px, car vehicle 18px,
  golf human 11px — 여기선 상수 예측이 무적이다).
  즉 latent 는 **"대상이 대략 이 사분면"** 수준은 알지만 셀 단위 국소화는 못 한다.
- **attention 은 프레임 0 에서만 쓸 만하고, 최적층이 지표에 따라 다르다.**
  `attn` 의 f0 오차: 층0 334px -> **층20 110px** (17~21 이 118~134px 로 평지),
  전체 프레임 평균은 203px. 즉 시간이 갈수록 무너진다.
  · **지표가 층 선택을 바꾼다**: 판별력(within-between)은 층 **15** 가 최고였는데
    픽셀 정확도(f0)는 층 **20** 이 최고다. 이래서 최종 판정은 하류 지표여야 한다.
- **결론 — 무엇을 꽂을 것인가**
  1. **국소화는 hidden state 가 아니라 생성 좌표를 쓴다.** fps 5 에서 클립당 20점(0.5초 간격),
     5씬 17타겟 + 이번 22샘플 전부 정상 생성. 카메라 디코더에 **명시적 기하 입력**으로 넣거나
     **aux supervision 타깃**으로 쓴다 — d157_dilo 의 겨냥 실패에 직접 대응한다.
     비용: 생성 약 10s/클립 -> d157 875씬 주 피사체 1개씩이면 약 2.4시간(1 GPU).
  2. **텍스트 조건은 층 15 hidden** (판별력 sweep 기준, `point`/`track` 형식). 지금 캐시가 쓰는
     층 35 는 두 지표 모두에서 최하위다. 단 이건 "텍스트 조건"이고 "국소화"가 아니다.
  3. **미검증 레버 — 학습된 probe.** 위 결과는 전부 **내적(파라미터 0)** readout 이다.
     patch latent -> (x,y) 로 **선형/MLP probe 를 학습**하면 달라질 수 있고, 이게 남은 가장
     중요한 미검증 축이다. 40px 수준이 나오면 중간층 patch latent 가 국소화 피처로 살아난다.
     "정보가 없다" 가 아니라 "텍스트-patch 직접 정렬로는 안 읽힌다" 까지만 확인된 것이다.

- **`scripts/viz/molmo2_vs_obb.py` — Molmo2 생성 좌표 ↔ scene_graph OBB track 2D 투영 일치도
  (D164, 2026-09-08, 사용자 지시 "돌려줘").** `CinemaTraj/scripts/export_target_track.py` 를
  d121 코퍼스에 실행(52 scenes, 전 변이 valid, missing anchor 0) 한 뒤,
  `track_world`(DA3 world)를 `pose.npz` 의 `extrinsics`(**w2c**, (49,3,4)) / `intrinsics`
  (cx,cy=320,180 = 640x360 규약) 로 투영해 Molmo2 `Track {target_text}` 생성 좌표와 비교한다.
  anchor 를 `dyn_*`(움직이는 피사체) / `stat_*`(정적 배경) 으로 나눠 본다.
  · **둘 다 추정이라 정답 대조가 아니다** — 일치하면 "같은 물체를 봤다"는 강한 증거이고,
    불일치는 어느 쪽이 틀렸는지 말해주지 않는다.

### 발견 (D164 Molmo2 ↔ OBB 일치도, 8씬 29 anchor, fps 5)
| | n | Δf0 median | Δmean median | <50px | <100px |
|---|---|---|---|---|---|
| **dyn (움직이는 피사체)** | 15 | 21.7px | **18.6px** | 73% | 87% |
| stat (정적 배경) | 14 | 35.3px | 41.5px | 57% | 79% |
| 전체 | 29 | 31.9px | 37.5px | 66% | 83% |

- **움직이는 피사체에서 두 독립 추정이 640x360 프레임의 ~3%(19px) 안에서 일치한다.** 방법이
  완전히 다른데도(VLM 텍스트 grounding 생성 vs DA3 recon + 분할 + OBB 중심 + 투영) 그렇다.
  대표: `camel dyn_1` Δf0 **1.8px**, `car-roundabout dyn_1` **2.1px**,
  `fashion-walk dyn_1` **5.6px**, `car-roundabout dyn_0` 22.4px.
- **`Track the blue car driving on the left side of the road` 가 Δf0 2.1px** — 앞서 관계절을
  자른 `Track the blue car` 는 파란 표지판으로 갔었다. **관계절이 필수 단서라는 반대 증거가
  독립 지표로 재확인**되었다. `--trim_level full` 이 맞다.
- **불일치는 거의 전부 "어느 인스턴스인가" 모호성**이고 "어디인가" 오차가 아니다:
  `car-roundabout stat_2 building` 294.8px (양쪽에 건물이 있고 "beige building with a white
  corner" 가 애매), `camel stat_0 fence` 59.9px (좌/우 + larger/smaller),
  `couple-rocks moss` 84/74px (large/small + left/right), `fashion-walk stat_1 fence` 114px.
- **유일한 실질 국소화 실패는 골프공** (`golf dyn_0` 167px, molmo2 (385,224) vs obb (317,74)).
  공은 지면에 있으므로 **Molmo2 가 맞고 OBB 가 틀렸다** — 작은 물체에서 분할/OBB 가 깨진다.
- 함의: **Molmo2-as-선택기는 움직이는 피사체(=겨냥 대상)에 대해 검증되었다.** 정적 배경의
  불일치는 코퍼스의 라벨 모호성 문제이고, 그 anchor 들은 대개 `aim=free` 변이(48~49%)라
  정밀 겨냥이 목적이 아니다. OBB 라벨 품질은 dyn 에서 수십 px 수준 — aux supervision 용으로
  충분하다.

- **`main/conf/experiment/dynpose_d157_dilo_{da3,molmo2,da3_molmo2}.yaml` — `dolly_in_look_at`
  단일 preset 코퍼스 위의 3-arm 모델축 비교 (D163, 2026-09-07, 사용자 지시 "scene 늘려서 dolly
  in look at 만 학습했을 때 da3 vs molmo2 vs da3+molmo2를 비교하고 싶은데" → "지금 181씬으로
  3-arm 돌린다").** preset 축을 없애면 남는 변이 축이 τ(hole 사다리)와 anchor 둘뿐이라, 세 arm 의
  차이가 "preset 어휘를 외웠나" 가 아니라 "씬을 읽고 대상 쪽으로 들어가나" 로 좁혀진다.
  · 코퍼스 `latentcam_dynpose_d157` — D157 뱅크가 875편 중 187편까지 구워진 **중간 상태**를
    부분 export 한 것 (`CinemaTraj/scripts/run_dynpose_d163_export.sh`, `--skip_done` 으로 나중에
    증분). `filter_seg_list_by_preset.py --presets dolly_in_look_at --suffix dilo`.
  · 실측 규모 `통과 707 / 14301` → **train 645 (scene 149) / test 62 (scene 17)**. 짝이 되는
    이전 실험 D95(`dynpose_d84_k6_dionly`, da3 단독) 의 train 299 / test 37 대비 2.2배이고
    **scene 수는 오히려 적다** — 187편 중 178편이 d129 의 267편과 겹친다.
  · 캐시는 전부 공유 + 증분: geo raw 는 d129 루트(160편 hit, 6편 추가), molmo2 video 는 d137
    루트(160편 hit, 6편 추가), molmo2 text 만 d157 루트에 새로 (`(scene_key, caption)` 키라
    캡션이 바뀌면 재사용 불가).

### 발견 (D164 실측, vista4d/camel, dolly_in_look_at)
- **최종층(35)의 텍스트→video attention 은 타겟을 판별하지 못한다.** 타겟 명사구를
  낙타/다른 낙타/오른쪽 울타리/왼쪽 울타리/배경 나무 5종으로 바꿔도 attention map 의
  비대각 코사인이 **0.994** (min 0.989), top1 셀은 5종 전부 `[8,5]` 동일, centroid 이동 0.13 셀.
  대신 9x9 테두리에 62.5% (uniform 39.5%), 앞 3프레임에 43.8% (uniform 6.1%) 가 쏠린다 —
  물체가 아니라 **위치 artifact** 다.
- **층 13~16 에는 판별력이 있다.** 36층 sweep 에서 비대각 코사인 층13 **0.711** / 층15 0.722 /
  층14 0.848 / 층16 0.880, 나머지는 전부 0.91~0.99. 층 13 의 코사인 행렬은 의미 구조를 갖는다:
  낙타↔낙타 0.952, 울타리↔울타리 0.971, 낙타↔울타리 0.75~0.80, 배경 나무↔나머지 0.50~0.55.
  층 13 에서 target 그룹 video 질량 34.3% vs probe 5.9% (최종층은 22.6% vs 16.9% 로 미분화),
  테두리 질량 40.4% ≈ uniform. 층 17 부터 테두리 50~62% / 앞 3프레임 33~58% 로 붕괴한다.
- **캡션에서 무엇을 빼야 하나 — `--caption_mode` 3종 실측.** 사용자 지시 2026-09-08
  ("다른 명사들 빼고 camel과 motion 설명만" → "내 말은 target에 관련된 text만 쓰자는거야
  camera motion, framing 같은거 빼고" → "along the fence 이런 것도 빼줘").

  | 모드 | 캡션 |
  |---|---|
  | `concise` | `The camera significantly dollies straight forward toward the larger pale camel walking along the fence, keeping it in a medium shot ... with the larger wooden fence ... and the smaller pale camel ... also in frame.` |
  | `target_only` | `The larger pale camel.` |
  | `minimal` | `The camera significantly dollies straight forward toward the camel.` |

  `target_only` 는 motion·framing·composition 절을 다 버리고 `target_text` 만 문장으로 세운
  뒤, **다른 물체를 지목하는 관계절까지 잘라낸다** (`trim_relations`). 판별 기준은 전치사 뒤의
  **정관사**다 — `along the fence` / `in the background` / `on the right side of the enclosure`
  는 씬의 다른 물체를 가리키므로 자르고, `in a striped shirt` / `with green leaves` 는 타겟
  자신의 서술이라 남긴다. 잘린 뒤 매달린 분사(`... camel walking`)도 뗀다.

  같은 3부류(낙타/울타리/나무)로 맞춘 비교 (`--target_row_tail` 은 target 구간 마지막 N 토큰만):

  | 설정 | 비대각(전체) | 비대각(3부류) | centroid | video 질량 |
  |---|---|---|---|---|
  | **`target_only` 층15 (4행)** | **0.695** | **0.584** | 1.06 셀 | **36.7%** |
  | `concise` 층13 (8행) | 0.711 | 0.596 | 1.11 셀 | 33.5% |
  | `concise` 층13 (마지막 2행) | 0.708 | 0.610 | 1.12 셀 | 31.1% |
  | `target_only` 층13 (4행) | 0.729 | 0.635 | 0.98 셀 | 33.3% |
  | `minimal` 층13 (2행) | 0.846 | 0.846 | 0.49 셀 | 32.3% |
  | `concise` 층35 | 0.994 | 0.993 | 0.13 셀 | 21.1% |
  | `target_only` 층35 | 0.997 | 0.995 | 0.11 셀 | 22.5% |
  | `minimal` 층35 | **1.000** | 1.000 | 0.08 셀 | 22.3% |

  · **`target_only` 가 최선이다.** 판별력 최고(0.584)이고 video 질량도 최고(36.7%)이며,
    쓸 만한 층의 띠가 가장 넓다 — 비대각 <0.9 인 층이 `target_only` 는 3·11~16 (7개),
    `concise` 는 13~16 (4개), `minimal` 은 3·5·9·11·13·15 (산발). 최적층도 다르다
    (`target_only`/`minimal` 층15, `concise` 층13).
  · **카메라 motion·framing 텍스트는 잉여이고 약하게 해롭다.** 빼면 판별력과 video 질량이
    같이 올라간다. 반면 **타겟의 서술 수식어는 국소화를 지고 있다** — `minimal` 처럼 맨 부류
    명사(`the camel`)만 남기면 0.846 으로 급락한다. 즉 버릴 것은 "카메라가 무엇을 하는가",
    지킬 것은 "타겟이 무엇처럼 생겼는가" 다.
  · 행 수는 원인이 아니다 — `concise` 를 마지막 2행으로 깎아 `minimal` 과 행 수를 맞춰도
    0.610 으로 그대로다.
  · 최종층에서는 세 모드 전부 무효다 (0.994 / 0.997 / **1.000**). `minimal` 은 낙타·울타리·
    나무가 **완전히 구별 불가**다.
  · 층 15 `target_only` 의 공간 분포는 눈으로도 맞는다 — 나무는 f0·f7·f21·f34·f48 전부
    좌상단 나뭇잎, 울타리는 우측 목재, 낙타는 f0 에서 몸통/혹. 다만 시간합 top1 셀은
    9x9 해상도에서 여전히 불안정해 과신하면 안 된다 (코사인·centroid 로 판정할 것).
- **Molmo2 학습 분포형 지시문 형식 + `--trim_level`** (사용자 지시 2026-09-08 "Molmo2가 학습한
  텍스트 분포에 맞춰서 형식을 바꿔줘봐. Track하라던지 질문형식이라던지", "walking 같은 행위도
  포함한 버전과 그냥 the larger pale camel walking along the fence 형태도 다시 돌려봐줘").
  · 체크포인트 `README.md` 의 예시를 글자 그대로 옮겨 `--caption_mode point/track/question/count`
    를 추가했다: `Point to {t}.` / `Track {t}`(**마침표 없음**) / `Where is {t} in the video?` /
    `Count {t}.` 이 형식들은 지시문이 프롬프트 전부라 **`PROBE` 를 붙이지 않는다**
    ("Describe the camera trajectory ..." 를 덧붙이면 off-distribution).
  · `--trim_level {full,action,noun}` — 타겟 구절을 어디까지 깎나:
    `the larger pale camel walking along the fence` / `... camel walking` / `... camel`.
  · **`chat_template.jinja` 는 `<|video|>` 를 항상 맨 앞에 놓는다** — content 리스트의
    text->video 순서와 무관하다. 즉 README 예시(text 먼저)와 우리 코드(video 먼저)는 같은
    토큰열이 되고, `cache_molmo2_embeddings.py` 의 순서는 문제가 아니다. 확인 완료.
  · 템플릿의 `DEMO_STYLES` 는 학습 태스크 이름 화이트리스트인데, 여기 없는 `style` 값만
    `"{style}: "` 로 앞에 붙는다. 즉 태스크 토큰이 프롬프트에 주입되지는 않는다.
  · 1건 실측(`Point to the larger pale camel.`, 층15, span=명사구 4토큰): **video 질량 40.6%**
    (target_only 36.7% / concise 33.5% 대비 최고), **테두리 질량 39.4% = uniform 39.5%**
    (위치 artifact 0), 앞 3프레임 17.4%, 프레임0 의 peak 가 **낙타 몸통**에 앉는다.
    지시문 형식 4종 x trim_level 3종의 swap 판별력 비교는 아직 안 돌렸다.
- **[결정적] Molmo2 는 대상을 정확히 알고 track 한다 — raw attention 이 그걸 못 보여줄 뿐이다.**
  `Track {t}` 로 생성시켜 좌표를 디코딩한 결과 (vista4d/camel, 49프레임, fps 10):

  | 프롬프트 | t=0 좌표(1000 스케일) | 픽셀(640x360) | 실제 위치 |
  |---|---|---|---|
  | `Track the larger pale camel` | 497, 438 | (318, 158) | 큰 낙타 등/혹 |
  | `Track the smaller pale camel` | 352, 352 | (225, 127) | 뒤쪽 작은 낙타 |
  | `Track the tree with green leaves` | 470, 086 | (301, 31) | 상단 나뭇잎 |
  | `Track the larger wooden fence` | 162, 311 | (104, 112) | 좌측 울타리 |

  · 네 타겟이 전부 정확히 분리되고 궤적도 매끄럽다 (카메라가 들어가면서 전부 왼쪽으로 드리프트).
  · 출력은 **0.5초 간격 10점 = 2Hz** — `video_preprocessor_config.json` 의 `sampling_fps: 2` /
    `max_fps: 2.0` 과 일치한다. 49프레임/10fps = 4.9s -> 10점. **프레임 커버리지 20% 는 실패가
    아니라 모델의 native track rate 다.**
  · `Point to {t}.` 는 t=0 단일 점을 내고 track 첫 점과 일치한다 (낙타 495,450 vs 497,438).
  · **생성 track(초록) vs 층15 attention argmax(마젠타)**: f0 에서만 거의 일치하고 f10 부터
    마젠타가 나뭇잎·울타리·땅으로 흩어진다. 즉 **정보는 있고 attention 이라는 렌즈가 눈이 먼 것**
    이다. 이 대조 없이는 "모델이 모른다" 와 구별할 수 없었다 — attention map 단독 판정의 한계.

- **track 출력의 시간 해상도는 입력 프레임이 아니라 Molmo2 의 학습 샘플링 격자(2Hz)가 정한다**
  (사용자 질문 2026-09-08 "왜 track이 전프레임에 있지 않음?", "49프레임 다 들어가는건 맞지?").
  · **49프레임은 전부 들어간다.** 실측: `video_grids [[49, 9, 9]]`,
    `pixel_values_videos (49, 729, 588)`, patch 토큰 3969 = 49 x 81. `--fps` 를 바꿔도
    **시각 텐서는 글자 그대로 동일**하다 — fps 는 프롬프트의 timestamp 문구에만 들어간다.
  · 프롬프트는 프레임마다 초 단위 타임스탬프를 소수 1자리로 붙인다:
    fps 10 -> `0.0 0.1 0.2 ... 4.8` (49종) / fps 25 -> `0.0 0.0 0.1 0.1 0.2 0.2 ...` (중복 발생)
    / fps 2 -> `0.0 0.5 1.0 ...`
  · 그런데 **출력은 항상 0.5초 격자**다 = `video_preprocessor_config.json` 의 `sampling_fps: 2`
    / `max_fps: 2.0`. 즉 track 점 개수 ≈ duration / 0.5:
    fps 10(4.9s) -> 10점(49프레임의 20%) / fps 25(1.96s) -> 4점 / fps 2(24.5s) -> 48점.
  · **fps 를 낮춰 점을 늘리는 것은 함정이다.** fps 2 로 두면 48점이 나오지만 좌표가
    `509 453` 으로 **전부 고정된 degenerate track** 이 된다 — 실제로는 0.1초 간격인 프레임을
    0.5초 간격이라고 속인 off-distribution 입력이라 모델이 정지 궤적을 낸다.
  · 결론: 프레임당 위치가 필요하면 10개 anchor 를 보간하거나 겹치는 윈도로 나눠 돌려야 한다.
    커버리지 20% 는 실패가 아니라 모델의 native track rate 다.

- **다른 4개 씬 재현 (golf / basketball-four / car-roundabout / couple-walk, 씬당 타겟 5종,
  `Track {trim_noun}`, fps 10).** 사용자 지시 "다른 영상으로도 돌려봐줘 한 4개정도".

  | scene | 타겟 | 점 평균 | step 평균 | 좌표쌍 최소거리 | 중앙 |
  |---|---|---|---|---|---|
  | golf | 2 | 9.0 | 18.1 | 83.7px | 83.7px |
  | basketball-four | 5 | 10.0 | 17.0 | 140.2px | 239.2px |
  | car-roundabout | 5 | 8.8 | 16.4 | **0.0px** | 174.3px |
  | couple-walk | 5 | 10.0 | 8.4 | 57.7px | 213.2px |

  · **17/17 타겟에서 좌표가 나왔고 대부분 서로 57~140px 이상 떨어져 구분된다.** 눈으로 확인한
    것: golf 공은 f0 에서 클럽 헤드 옆에 붙고 타격 후 공중으로 점프(t=3.5 에 y 650->103),
    couple-walk 의 fire hydrant 는 작은 정적 물체인데 전 구간 정확.
  · **실패 1 — 좌/우 앞바퀴를 구분하지 못한다.** `the right front wheel` 과
    `the left front wheel` 이 동일 좌표(0.0px). 관계절을 되살려
    `... of the dark car` 를 붙여도 Δ 약 4/1000 (~2.5px) 로 여전히 구분 못 한다.
    부분(part) 수준의 좌/우 disambiguation 은 이 모델의 한계다.
  · **실패 2 — 그리고 이것이 앞선 `trim_level` 결론을 제한한다.** `Track the blue car` 는
    t=3.0 부터 (962,453)->(686,464) 로 **파란 원형 도로표지판**(f42 부터 진입)을 따라갔다.
    이 49프레임에 파란 차는 없다. 그런데 잘리지 않은 원문
    `Track the blue car driving on the left side of the road` 를 주면 t=0.0~1.0 에
    (194,492)->(041,503) — **좌측의 실제 차**를 잡고 프레임을 벗어날 때 정확히 멈춘다.
    즉 **`driving on the left side of the road` 는 잉여가 아니라 필수 단서였다.**
    attention 지표(`score`)에서는 `tgt_noun` > `tgt_full` 이었지만, **생성 출력에서는 관계절이
    타겟을 결정하는 경우가 있다.** 관계절 제거는 기본값으로 두면 안 된다 —
    "다른 물체를 지목하는 절"과 "타겟을 유일하게 특정하는 절"이 문법적으로 같은 모양이다.

- **`--main` 주 피사체 선택 + `--fps_sweep`** (사용자 지시 2026-09-08 "main 물체로 track해줘.
  가운데 차라던지. 그리고 fps 별로 track 성능 차이 없어?").
  · **최빈 target 은 주 피사체가 아니다.** car-roundabout 의 최빈은 `right front wheel`(88건)
    이고 주 피사체 `large dark grey vehicle`(86건) 이 2위다. 그래서 `SUBJ_PRIO`
    (vehicle/car/person/human/man/woman/camel/... 순) 로 라벨 우선순위를 두고 고른다.
  · 5씬 main track 전부 10/10 점, 눈으로 확인해 전부 정확했다 (`--trim_level full` 원문 사용):
    car-roundabout `the large dark grey vehicle driving on the road` -> 가운데 회색 Mini 차체,
    golf 는 공이 아니라 `the human in a light blue shirt ... swinging a golf club` -> 골퍼 몸통,
    basketball-four `the man in a white tank top carrying a woman on his back`,
    camel, couple-walk 도 동일하게 주 피사체.

- **fps 별 track 성능 차이가 크다 — 낮추면 궤적이 붕괴한다.** 같은 main 타겟, fps 만 바꿈.
  프롬프트의 timestamp 간격 = 1/fps(소수 1자리), 출력 격자는 항상 0.5초 -> 점 개수 ≈ 2T/fps.

  | fps | 모델이 본 길이 | 점 | step(px) | 최대편차(px) | fps10 과 평균거리(px) |
  |---|---|---|---|---|---|
  | | | | car / bball | car / bball | car / bball |
  | 1 | 49.00s | 97 | 0.0 / 0.0 | **0.0 / 0.0** | 7.4 / 45.7 |
  | 2 | 24.50s | 48 | 0.0 / 0.0 | **0.0 / 0.0** | 8.1 / 45.3 |
  | 4 | 12.25s | 25 | 1.5 / 6.5 | 23.8 / 69.7 | **4.3 / 2.8** |
  | 5 | 9.80s | 20 | 2.2 / 8.2 | 22.4 / 70.8 | **2.8 / 2.9** |
  | 10 | 4.90s | 10 | 3.2 / 16.0 | 19.5 / 73.3 | 0.0 / 0.0 (기준) |
  | 20 | 2.45s | 5 | 6.2 / 28.6 | 16.3 / 69.0 | 1.8 / 6.1 |
  | 25 | 1.96s | 4 | 7.6 / 38.2 | 15.2 / 72.5 | 2.8 / 9.3 |
  | 30 | 1.63s | 4 | 7.4 / 43.2 | 14.2 / 70.0 | 3.1 / 7.3 |

  · **fps <= 2 는 쓰면 안 된다.** 점은 48~97개로 가장 많지만 좌표가 한 점에 고정된
    **정지 궤적**이다 (최대편차 0.0). 입력 timestamp 간격(0.5~1.0s)이 출력 격자와 같거나
    커지면서, 실제로는 0.1초 간격인 프레임을 "0.5초/1초 간격"이라고 속인 셈이 되어
    모델이 "이 장면은 정지"라고 판정한다. fps10 과의 거리도 이 구간이 가장 크다
    (빠른 움직임인 basketball 에서 45.7px).
  · **fps 4~5 가 최적이다.** 20~25점(프레임 2개당 1점)으로 밀도가 높으면서 움직임이 보존되고
    (최대편차 22~24 / 70), fps10 과의 일치도도 가장 좋다 (2.8~4.3px).
  · **fps >= 20 은 점이 4~5개뿐**이고 일치도가 나빠진다 (car 1.8->3.1, bball 6.1->9.3).
    시간 커버리지 자체는 유지된다 (0.0~1.5s 를 프레임 0/15/30/45 로 펼침).
  · 규칙: **fps 를 과소 신고하면 안 된다.** 과대 신고(20~30)는 출력이 거칠어지는 것으로
    끝나지만, 과소 신고(1~2)는 모델이 정지로 판정해 track 자체가 무의미해진다.
  · 주의: "fps10 과 평균거리" 는 GT 가 아니라 **자기일관성**이다. fps 4/5/10 이 서로 잘 맞는
    것은 세 값이 모두 in-distribution 이라는 뜻이고 정확도의 증거는 아니다.
  · 참고: `main/cache_molmo2_embeddings.py` 의 `--fps` 기본값은 25.0 인데 실제 데이터는 10fps
    가정이다. 임베딩 캐시에서 fps 는 timestamp 문구에만 들어가지만, **최소한 사실과 맞는
    값(10)** 을 쓰는 것이 맞다. track 을 목적으로 쓸 때는 4~5 가 낫다.

- **[결정적] 현재 캐시 프롬프트(`concise + PROBE`)는 좌표를 아예 내지 않는다** — Molmo2 를
  captioning 모드에 넣고 있다 (사용자 지시 2026-09-09 "concise vs Track으로 실제 최종 output
  track 차이 있는지 결과 시각화"). camel / car-roundabout 의 `dolly_in_look_at` 변이, fps 5,
  같은 씬·같은 타겟에 프롬프트만 4종:

  | 프롬프트 | 생성 형식 | 좌표 | t=0 위치 (camel) |
  |---|---|---|---|
  | `concise` (코퍼스 완성문) | `<points>` 20 타임스탬프 | 20점 **but 좌표 고정** | (321,159) |
  | **`concise + PROBE`** (현재 캐시) | **산문 서술** | **0점** | — |
  | `{target_text}.` | `<points>` 1 타임스탬프 | 1점 | (317,163) |
  | **`Track {target_text}`** | `<tracks>` 20 타임스탬프 | **20점, 매끄러운 궤적** | (318,158) |

  · `concise + PROBE` 생성 원문: *"The camera moves forward in a straight line, gradually
    closing the distance between itself and the larger pale camel. This forward movement
    causes the frame to tighten..."* — PROBE 가 요구한 대로 **카메라 궤적을 산문으로 서술**한다.
    즉 현재 프롬프트는 grounding 모드가 아니라 captioning 모드다.
  · `concise` 단독은 **형식이 깨진다**: camel 은 `502 442` 가 20번 반복(고정점),
    car-roundabout 은 인스턴스 id 가 `1,2,3,4,…` 로 매 타임스탬프 바뀌며 좌표가
    `973 403`(x=623, 우측 끝)에 고정 — 궤적이 아니라 **서로 다른 물체 목록**을 뱉는다.
    서술문이라 태스크가 정의되지 않아서다.
  · **그런데 t=0 좌표는 네 형식이 5px 안에서 일치한다** (camel 318/317/321,
    car-roundabout 343/345/344). 즉 **"어디"는 어느 형식에서도 알고 있고, 형식이 정하는 것은
    "어떤 프로토콜로 뱉느냐"** 다. 이 해석이 중요하다 — grid 의 score 차이(+0.335 vs +0.225)는
    정보의 유무가 아니라 **얼마나 읽기 쉬운 형태로 정리되어 있느냐**의 차이이고, hidden state
    를 쓸 때도 같은 논리가 적용된다.

- **`Track {target_text}` 로 바꿨을 때의 캐시 규모 실측 (d121 전량).**
  | | distinct 텍스트 | min | p50 | p95 | max |
  |---|---|---|---|---|---|
  | 현재 `concise + PROBE` | 13,183 | 31 | 78 | 90 | **100** |
  | 신규 `Track {target_text}` | **292** | 11 | 20 | 23 | **25** |

  · `text_len` **128 → 25**. 그래서 **span 축소(target 구절만)는 불필요하다** — 형식 교체가
    이미 흡수한다. tail 17토큰 중 non-target 행이 9개뿐이고, 그 중
    `<|im_start|>assistant\n` 는 답을 내려는 위치라 융합 정보가 응축될 자리인데 자를 근거가 없다
    (video 질량 13.8% 만 봤고 정보량은 안 봤다).
  · 캐시 키가 `(scene_key, caption)` 13,679 → `(scene_key, target_text)` 약 292 조합.
    **`(13679,128,2560) fp16 8.96 GB → (292,25,2560) fp16 38 MB`, 약 240배.** 굽는 시간도
    씬당 캡션 수백 개 → 3~6 개로 줄어든다.
  · 대신 **molmo2 스트림이 변이를 구분하지 못하게 된다** — 같은 씬·같은 타겟이면 `dolly_in`
    이든 `orbit_left` 든 텍스트가 동일하다. 의도한 역할 분담(molmo2=grounding, T5=motion/
    framing)이지만 **전제가 붙는다: T5 스트림이 제대로 작동해야 한다.** 지금 `text_cross_attn`
    은 `h = CA(h,...)` 로 residual 을 덮으므로(`:309`) 뒤의 geo CA 가 T5 의 motion 정보를 지울
    수 있다. 현재는 molmo2 텍스트에 motion 이 중복돼 그 손실이 가려져 있는데, 형식을 바꾸면
    중복이 사라져 이 결함이 드러난다. **-> 전 스트림 zero-init gated residual 을 먼저 고칠 것.**

- **텍스트 형식 x 층 격자 실측 (3 씬 평균: camel / car-roundabout / basketball-four).**
  score = within - between, 각 씬 타겟 5종 (클래스 3~4종):

  | format | 최적 층 | score | video 질량 |
  |---|---|---|---|
  | `point`   `Point to {t}.` | **15** | **+0.335** | 38.2% |
  | `track`   `Track {t}` | 15 | +0.329 | 38.2% |
  | `question` `Where is {t} in the video?` | 15 | +0.324 | 35.6% |
  | `tgt_noun` `The larger pale camel.` | 15 | +0.302 | 35.0% |
  | `tgt_action` `... camel walking.` | 15 | +0.292 | 35.2% |
  | `tgt_full` `... camel walking along the fence.` | 15 | +0.267 | 34.7% |
  | `motion_noun` `The camera ... toward the camel.` | 13 | +0.263 | 35.6% |
  | `concise` (코퍼스 완성문) | 13 | +0.225 | 34.5% |
  | `concise_probe` (**학습이 쓰는 조건**) | 13 | +0.224 | 34.5% |

  · **학습 설정이 두 축 모두에서 최하위다** — 텍스트는 `concise_probe`(9/9위), 층은 35
    (score +0.01, 판별력 사실상 0). 층 12~18 이 유일한 판별 구간이고 19층부터 0.02 로 붕괴한다.
  · 순서가 단조롭고 해석이 된다: **학습 분포형 지시문 > 타겟 문장 > 관계절 포함 > motion 절 추가
    > 코퍼스 완성문.** 즉 카메라 motion·framing·관계절은 잉여이고, 타겟 서술 수식어는 유지해야
    하며, Molmo2 가 학습한 지시문 형식으로 감싸는 것이 가장 좋다.
  · `probe` 유무는 무관하다 (+0.225 vs +0.224).
  · 단, 이 격자는 **attention 지표 기준**이다. 위 항목이 보여준 대로 attention 은 가설 생성기이고
    최종 판정은 하류 지표(층15 vs 층35 피처로 학습한 카메라 모델의 CLaTr/collision/framing)여야 한다.
- 즉 `main/cache_molmo2_embeddings.py` 가 굽는 **최종층 text hidden 은 타겟 국소화가 가장 약한
  층**이다. GR00T N1.5 가 36층 중 12층을 쓰는 것과 같은 결론이 우리 데이터에서 재현되었다
  (층 13/36 = 0.36 깊이 vs 12/36 = 0.33).

### Fixed
- **`scripts/data/cache_geo_raw_da3.py` — `is_done` 의 동일성 비교에서 `meta['exp']` 제외
  (D163).** `exp` 는 설정이 아니라 "누가 처음 구웠나" 라벨인데 비교에 들어가 있어서, 캐시
  디렉토리를 공유하는 새 arm 이 붙을 때마다 내용이 **비트 동일한** 파일 전량(dilo 기준 160개,
  6.8 GB)을 다시 구워 덮어썼다. 그 디렉토리를 읽고 있는 다른 학습(D160)이 반쯤 쓰인 파일을
  `torch.load` 하다 죽을 수 있다. 진짜 설정 7개(da3 모델/입력 해상도/layers/posed/num_views/
  image_hw/first_view)와 dtype 은 그대로 전부 비교한다. 캐시 공유는 `dynpose_d137_da3.yaml` 이
  d129 루트를 가리키는 것처럼 이미 정상 사용법이다.
- **`main/conf/experiment/dynpose_d137c147_da3_{dd10,nodd}.yaml` — DataDoP 비율 ablation 을
  **캡션이 최신인** dynpose 코퍼스 위에서 다시 (D160, 2026-09-07, 사용자 지시 "fitting 끝난
  최신 dynpose 기준 caption 만 최신 형태로 바꾼 걸로 dd10 와 nodd da3 로 학습").** 같은 질문을
  d107 코퍼스에서 먼저 띄웠다가 epoch 3 에서 멈췄다 — d107 캡션이 D121 이전 구조형
  (`target: dog. motion: the camera dollies back ...`)이라 D123 이후 arm 들과 축이 어긋난다.
  · 코퍼스는 `latentcam_dynpose_d137c147` (D147 이 d137 뱅크의 캡션만 다시 구운 것). fit 이
    끝난 dynpose 5종 중 캡션이 최신인 유일한 것이다 — D157(875편)은 아직 fit 중.
    seg_list 4종 + `meta_dynpose.csv` 가 d137 과 바이트 동일함을 `cmp` 로 확인했다.
  · `nodd` seg_list 는 이 코퍼스에 없어서 새로 만들었다 (`filter_seg_list_by_preset.py
    --exclude_presets_prefix dd_ --suffix nodd`, RNG 없음). 실측 구성:
    dd10 train 9,728 (dd_ 973 = 10.0%) / nodd train 8,755 (dd_ 0) — **나머지 8,755 행은
    같은 행**이라 D84 축(full→nodd 가 60% 를 들어냈던 것)과 달리 짝이 거의 like-for-like 다.
    test 는 1,129 vs 1,016 로 113 행 차이가 남는다.
  · `dynpose_d137_da3`(D137) 대비 실효 diff 는 `exp_name`/`dl3dv_root`/`train_seg_list`/
    `test_seg_list` 4줄이라 dd10 arm 은 **캡션 교체 대조군**을 겸한다.
- **`main/evaluate/CLaTr/configs/dataset/standardization/dynpose49.yaml` + `scripts/prepare_clatr_vista.py`
  의 `--train_list/--test_list` — CLaTr 게이지를 dynpose d137 코퍼스로도 만든다 (D158, 2026-09-07,
  사용자 지시 "CLaTr도 dynpose d137에 대해서 학습 돌려놔줘").** prep 스크립트는 split 목록
  파일 이름이 `seg_list_vista4d_{train,test}.txt` 로 **하드코딩**돼 있어 dynpose
  (`seg_list_dynpose_dd10_*`)를 못 읽었다. 인자로 뺐고 **기본값은 vista 그대로**라 기존 호출은
  글자 그대로 같은 명령이다. 데이터 결과 train 9,728 / test 1,129 / 발산·결측 0, 궤적 길이
  전부 49 (= 학습 arm 3종과 같은 entry 집합).
  · dynpose49.yaml 의 실측 `shift_std` 는 **[3e-08, 1e-08, 2e-08]** 이다 — dynpose target
    궤적은 frame0 이 world 원점에 앵커돼 있어 첫 프레임 위치가 항상 0 이기 때문이고, vista
    d121 의 [0.389, 0.082, 0.226] 과 대비된다. 그대로 적으면 `standardize` 를 켜는 순간
    0 나눗셈이라 **1.0 으로 두고** 실측값은 주석에 남겼다. vista49 와 마찬가지로
    `trajectory_dataset.py:43` 의 `self.standardize = False` 때문에 현재는 안 쓰인다.
- **`scripts/prepare_clatr_vista.py` + `main/evaluate/CLaTr/configs/dataset/standardization/vista49.yaml`
  — CLaTr 을 우리 vista 코퍼스로 학습할 수 있게 (D153, 2026-09-06).** 지금까지 caption 지표를
  **E.T./ArtTraj 로 학습된 ckpt** 로 재 왔다. 그 config 는 `num_cams: 120` 에 shift_std
  ~[1.13, 1.19, 1.59] 인데 우리 target 궤적은 49프레임이고 실측 shift_std 가
  [0.389, 0.082, 0.226] 이다 — 텍스트-궤적 정합이 아니라 게이지 불일치를 재고 있었다.
  · prep 은 `--stage json|clip|all` 단일 드라이버. `da3/target_poses.npz` 의 extrinsics 는
    OpenCV **w2c** 라 `inv` → `[:, :3, 1:3] *= -1` (OpenCV→OpenGL) 두 단계를 밟는다 —
    `main/prepare_clatr_data.py` 와 같은 변환. 캡션은
    `prompts.json['<idx>']['prompt_camera_with_scene_video']['concise']`.
  · split 은 원본의 95/5 랜덤이 아니라 **코퍼스 자신의** `seg_list_vista4d_{train,test}.txt`
    를 쓴다. 학습 arm 들과 test entry 가 어긋나면 지표를 서로 대조할 수 없다.
    결과 train 9,389 / test 592 / 발산·결측 0, 궤적 길이 전부 49.
  · vista49.yaml 의 `norm_*`/`shift_*` 는 우리 train split 실측값이다. **현재는 안 쓰인다** —
    `trajectory_dataset.py:43` 이 `self.standardize = False` 로 못 박아 뒀다. 그래도 남의
    코퍼스 숫자를 물려두면 나중에 켤 때 조용히 틀리므로 우리 값을 적었다. 실제로 바뀌는
    축은 `num_cams` 120→49 (padding 길이) 하나다.
- **`main/conf/experiment/vista4d_d128_molmo2{,_nogeo}.yaml` — d128 코퍼스의 molmo2 두 셀
  (D152, 2026-09-06).** d121 축에는 모델 3셀(da3 / molmo2_nogeo / da3+molmo2)이 다 있는데
  d128 축에는 `vista4d_d128_da3_t128`(D131) 하나뿐이라, "molmo2_nogeo 가 제일 낫다"가
  **모델축 사실인지 d121 코퍼스 한정인지**가 안 갈렸다. 이 두 arm 이 d128 안에서 나머지
  두 셀을 채운다. d124/d133 대비 바뀌는 줄은 코퍼스 경로 3줄 + molmo2 **text** 캐시 1줄뿐.
  · **video 캐시는 d121 것을 그대로 쓴다.** molmo2 video 캐시는 씬당 1파일이고 두 코퍼스가
    같은 52편·같은 49프레임을 본다. 가정이 아니라 실측 — d128 root 로 다시 구운
    `vista4d_avocado-slice.pt` 가 d121 것과 `torch.equal` **비트 동일**(`|Δ|max 0.0`)이었다.
    text 캐시는 dedup 키가 `(scene_key, caption)` 이라 코퍼스마다 달라 d128 판을 새로 구웠다
    (9,981 segment / 52 scene / 131 (scene,caption) pair).
  · 코퍼스간 평균 비교 금지 — test entry 가 875(d121) → 592(d128) 로 달라진다.
- **`models/da3_geo_encoder.PerViewResampler` — da3 geo 토큰을 view **안에서** P→R 로 줄이는
  학습형 Perceiver 풀링 (D144, 2026-09-06, 사용자 지시 "gpu하나는 da3에 49프레임을 넣되
  resampler를 달아서 token수를 줄이는 방식으로 molmo2랑 비교하게 학습 돌려놓고").**
  molmo2 의 `adaptive_avg_pool2d`(9x9→8x8) 자리를 cross-attention 으로 바꾼 것이고,
  **프레임을 섞지 않는다**는 제약은 같다. 이걸로 세 arm 의 토큰 예산이 정렬된다:
        D137 da3       6뷰 × 576  = 3456 tok  (프레임 솎기)
        D138 molmo2   49프레임 × 64 = 3136 tok  (프레임 안 avg-pool)
        D144 da3      49뷰 × 64   = 3136 tok  (프레임 안 학습 풀링)
  · 배선: `main/conf/config.yaml` 에 `geo_resampler{,_dim,_tokens,_heads,_layers}` 5키 신설.
    `geo_resampler: null`(기본)이면 resampler 를 **아예 만들지 않고** `from_raw` 가 예전
    코드를 글자 그대로 탄다 — 기존 da3 arm 과 state_dict·출력이 비트 동일하다.
  · `resampler_dim: 768` 은 `geo_latent_dim` 과 같아서 `GeoEncoder.proj` 가 Identity 가
    된다. 즉 3072→768 Linear 이 `proj` 에서 `PerViewResampler.kv_in` 으로 **옮겨간** 것이라
    추가 파라미터는 ~7.14 M (실측 trainable 2.366208 M → 9.506304 M).
  · 실측 프로브: `from_raw` 출력 (1,3456,768) → (1,3136,768), `backend.out_dim` 3072 → 768.
- **`main/conf/experiment/dynpose_d144_da3_f49res.yaml` (신규).** D137 대비 바뀐 줄은
  `geo_num_views 6→49` · `geo_resampler null→perceiver` · `geo_raw_cache_dir` 뿐.
  49뷰 pre-ln 캐시는 `scripts/data/cache_geo_raw_da3.py` 로 새로 구웠다
  (267 scene / 92.60 GB / 1.27 s/scene, scene 당 (49,576,3072) fp32 = 347 MB).
- **`scripts/eval/corpus_axis_compare.py --framing_scope` — 게이트를 "프레이밍을 책임질 수
  있는" 부분집합에서만 재는 필터 (D143, 2026-09-06, 사용자 지시 "aim이 follow인 것들이나
  target이 없는 free moving은 물체의 subject in frame 율이 낮은 건 당연해 이것들 제외한
  preset들만 재줘").** 세 값 — `all`(기존 동작, 기본) / `aimed`(`aim == "look_at"` 만) /
  `aimed_nontrack`(거기서 `track_*`·`dd_*` 도 제외). `track_*` 은 병진 추종이고 `aim` 은
  회전 조준이라 **독립 축**이므로 둘을 따로 건다.
  · 실측 (split=train, `subject_in_frame < 0.85` 비율):
        범위               vista_d121      vista_d128     dynpose_d137
        all (전량)          27.2%           26.4%           41.4%
        aimed                2.9%            2.8%           17.6%
        aimed_nontrack       3.1%            2.5%           17.0%
    남는 세그먼트는 `aimed_nontrack` 에서 6015/14975(40.2%) · 3998/9389(42.6%) ·
    955/9728(9.8%).
  · **읽기**: vista 는 사용자의 "당연하다"가 그대로 성립한다 — follow/targetless 를 빼면
    27.2% → 3.1% 로 무너진다. **dynpose 는 아니다** — 41.4% → 17.0% 에서 멈춘다. 즉
    dynpose 의 프레이밍 실패는 aim/track 으로 설명되지 않는 잔차가 따로 있다.
    (같은 범위에서 dynpose 는 955 seg 중 `s_curve` 하나가 62.0% 를 먹는다 — 씬은 195개인데
    seg/scene median 이 3 이다.)
  · 코퍼스별 test set 이 다르므로(vista 875 / dynpose 1129) **행 간 비교는 여전히 금지**다.

### Changed
- **`docs/corpus_and_model_axes.md` §1-a / §4-2 갱신 (2026-09-06).**
  · §1-a: D133(molmo2 단독)을 d121 test 875 에서 `last.pth` 로 평가해 vista 행 3칸을 채웠다.
    읽기가 뒤집힌다 — molmo2 단독이 da3 단독(D123)을 11지표 전부에서 이기고, da3+molmo2(D124)
    에게 captions 3개·`clatr_score`·`clatr/recall`·`fcd`(108.1954 vs 123.4501) 를 이긴다.
    D124 가 이기는 건 `clatr/precision`·`density`·`coverage`·`loss_traj` 넷. 즉 D123→D124
    이득의 상당 부분은 "얹어서"가 아니라 **"갈아타서"** 다.
  · §4-2: dynpose 수직 preset 0건의 원인을 **확정**했다. 게이트 기각이 아니라 **preset 선택
    경로가 다르다** — vista 샤드는 `route_presets.py` 를 안 부르고 `sample_camera_bank.py:1184`
    의 `--presets default=None (= 전량)` 으로 어휘 전량을 굽는 반면, dynpose 샤드는
    `route_presets.py --emit args` 의 슬레이트만 굽는다. 그 슬레이트의 수직 슬롯은
    `route_presets.py:189-192` 에서 `grav == "ground_ransac"` 이거나 `--vertical_fallback`
    (`:343` `default=False`, dynpose 샤드가 안 넘김) 일 때만 생기는데, dynpose 278 씬의
    `gravity_method` 가 **278/278 `geocalib`** 이라 `vertical_dropped: true` 가 278/278.
    **중력축 품질 문제가 아니다** — vista 도 `geocalib` 52 / `ground_ransac` 1 이라 같은
    라우팅을 태웠으면 똑같이 0건이었다. 처방과 비용은 문서 §4-2.

### Added
- **`scripts/eval/corpus_axis_compare.py` + `docs/corpus_and_model_axes.md` (D142) — 데이터축
  코퍼스 계측기와 2×3 그리드 분석 문서 (2026-09-06).** 사용자 지시 "데이터셋이나 모델
  개선해야할 사항들 있으면 다방면으로 분석해서 정리해놔줘".
  · 스크립트는 `<root>/<batch>/<hash>/da3/prompts.json`(= `pose_source: da3` arm 이 실제로 읽는
    파일; `<scene>/prompts.json` 이 **아니다**)을 seg-list 범위에서만 집계한다. 내는 것:
    규모/씬수, preset 다양성과 **계열 분해**(`dd_*` / `track_*` / 그 외), aim 분포,
    게이트 분위수, **게이트 꼬리 카운트**, **`track_*` × `aim` 교차표**(캡션이 추종을 주장하는
    비율 / 프레이밍을 약속하는 비율). `--corpus <이름>=<root>:<seg_prefix>` 를 반복해 나란히 본다.
  · 왜 계열 분해가 필요한가: 전체 uniq preset 은 dynpose 202 > vista 40 이라 사용자 framing
    ("dynpose 는 preset 적음")과 반대로 읽힌다. 그런데 202 중 181 이 `dd_*` one-off 로
    세그먼트의 10.0% 만 덮는다 — 빼면 실효 어휘가 vista 40 vs dynpose 21 로 뒤집힌다.
  · 문서에 실린 주요 실측: 수직 이동 preset(`pedestal_*`/`crane_*`)이 vista 19.4% / dynpose
    **0.0%**; `subject_in_frame < 0.85` 가 vista_d121 27.2% / dynpose_d137 41.4%;
    `track_*`+`aim=free` 3897 seg 이 `in_frame<0.5` 31.7~38.2% 인데 캡션은 100% 추종을
    주장(79~87% 는 프레이밍까지 약속) — task #134 의 크기를 처음 정량화한 값이다.
- **`main/conf/experiment/dynpose_d137_da3_molmo2.yaml` (D141) — 데이터축 × 모델축 2×3
  그리드의 빈 칸 (2026-09-06).** 사용자 지시 "데이터 측면에서 vista(scene 적고 preset 많음),
  dynpose(scene 많고 preset 적음) 을 비교하고 모델 측면에서 da3 vs molmo2 vs da3 + molmo2 를
  비교". vista d121 은 da3(D123)/molmo2(D133)/da3+molmo2(D124) 세 칸이 다 찼는데 dynpose
  d137 은 da3(D137)/molmo2(D138) 둘뿐이라 **"molmo2 를 얹는 이득이 코퍼스에 따라 달라지는가"**
  라는 상호작용 항을 물을 수 없었다.
  · 주석 제거 후 diff 기준, D137 대비 **molmo2 블록만** 추가되고 D138 대비 **geo(da3) 블록만**
    추가된다 (`exp_name` 제외 나머지 키 전부 동일). D123→D124 와 같은 조작을 dynpose 축에
    옮긴 것이다.
  · 캐시는 새로 굽지 않는다 — geo raw 는 D137 과 같은 d129 경로(씬 키 기준이라 코퍼스
    재굽기와 무관), molmo2 는 D138 이 구운 d137 루트 캐시. 후자가 dd10 리스트만 덮으므로
    `peav_seg_list_only: true` 를 같이 켠다.
- **`peav_seg_list_only` — PE-AV/molmo2 캐시 커버리지 검사를 seg-list 안으로 좁히는 옵션
  (D138, 2026-09-06).** `main/dataset_dl3dv.py` 에 `_peav_scope()` 를 추가하고
  `_preload_peav` 의 video 키 집합 / text miss 검사가 그 범위만 보게 했다.
  `main/conf/config.yaml` 기본값 `false` = **예전과 글자 그대로 같은 검사**.
  · 왜: 데이터셋은 `prompts.json` 을 전량 열거하고(dynpose d137 = 24371 seg) `base.py` 가
    **그 다음에** seg-list 로 `Subset` 을 뜬다. dd10 처럼 리스트가 코퍼스의 부분집합이면
    (10857 / 24371 = 44.5%) 학습이 한 번도 안 건드리는 세그먼트까지 캐시에 있어야 한다고
    우기며 `FileNotFoundError` 로 죽는다. 캐시를 전량으로 다시 굽는 건 오답이다 — text.pt 가
    2.2배(5.9 → ~13 GB)로 부풀고 증분은 전부 죽은 행이다. 상세는 `FIX.log` FIX-D138.
  · 켜도 바뀌는 것은 **검사 범위뿐**이다. `self.samples` / 인덱스 캐시 / 배치 구성은 그대로고,
    검사에서 빠진 세그먼트는 `Subset` 에도 안 들어가므로 `__getitem__` 이 도달할 수 없다
    (도달하면 예전처럼 KeyError 로 시끄럽게 죽는다).
  · `dynpose_d137_molmo2_nogeo.yaml`(D138) 이 `true` 로 켠다. vista d121 arm(D124/D133)은
    seg-list 가 전량이라 두 집합이 같아서 기본값 그대로다.
- **`main/conf/experiment/dynpose_d137_da3.yaml` (D137) + `dynpose_d137_molmo2_nogeo.yaml`
  (D138) — clamped_low 를 제외한 dynpose 코퍼스 위의 두 arm (2026-09-06).** 사용자 지시
  "어차피 clamped_low는 정상적인 카메라가 아니니 b로 해줘" + "학습도 molmo2+da3가 아니라
  molmo2로 해줘".
  · 코퍼스는 `latentcam_dynpose_d137` — D135 가 쓰던 `latentcam_dynpose_d129` 와 **뱅크·캡션·
    게이트·preset 라우팅이 글자 그대로 같고** export 필터 하나만 다르다
    (`vista4d_bank_to_dl3dv.py --drop_status clamped_low`, 근거는 video_generation CHANGELOG).
    D135 는 epoch 0 에서 정지시켰으므로 **비교할 지표가 없다** — 필터 효과를 보려면 d129 root
    로 따로 돌려야 한다.
  · **geo raw 캐시(11.34 GB)는 d129 것을 그대로 가리킨다.** 캐시 키가 씬 이름
    (`dynpose_<uuid>.pt`)이고 내용은 **소스 영상**의 geo feature 라 target 카메라 뱅크와
    무관하다. 변이 필터는 씬을 하나도 안 지우므로 267편이 그대로 맞는다. symlink 로 감추면
    `du` 로 공유가 안 보여서 경로를 그냥 d129 로 뒀다 (config 헤더에 명시).
  · **D138 은 molmo2 를 geo 위에 얹는 게 아니라 geo 를 대체한다.** 앞서 준비했던
    `dynpose_d129_molmo2.yaml`(D136, geo+molmo2)은 한 번도 안 돌고 폐기됐다. D133 이 vista 축
    에서 세운 대조를 dynpose 축으로 옮긴 것이라, 두 arm 의 diff 는 "molmo2 7줄 추가 + geo 5줄
    제거" 다. 근거는 D124 계측 `dpred_drop 0.02395` vs `dpred_xscene 0.00462`.
  · 둘 다 `epochs: 100` / `ckpt_at_epochs: [50, 100]` (실제 집행자는 `conf/config.yaml` 의
    `epoch_cap: 100`). text_len 128 · VAE · `scale_mode` · `intr_norm` · holdout 27편 고정.
- **`main/conf/experiment/vista4d_d121_molmo2_nogeo.yaml` — molmo2 CA 단독 arm (D133,
  2026-09-05, 사용자 지시 "반대로 da3를 빼고 molmo2만 써서도 학습돌려놔줘").** D124 의 거울상이다.
  · 세 arm 이 같은 코퍼스(d121) · 같은 875 heldout · 같은 text_len 128 위에 놓인다 —
    D123 `text CA + geo CA` / D124 `text CA + molmo2 CA + geo CA` / **D133 `text CA + molmo2 CA`**.
  · **왜 필요한가**: D124 video CA 계측(#171, n=875)이 `dpred_drop 0.02395` 인데
    `dpred_xscene 0.00462` 였다 — 영상을 빼면 예측이 움직이는데 **다른 씬 영상으로 바꿔치면
    그 1/5 밖에 안 움직인다**. D124 의 이득이 "영상을 읽어서"인지 "CA 슬롯이 하나 더 있어서"인지
    이 arm 이 가른다. geo 없이 D123 수준을 유지하면 전자, 크게 떨어지면 후자다.
  · D124 대비 바뀐 줄은 geo 계열뿐이다: `geo_encoder: da3 → null`, `geo_posed: true → false`,
    `geo_raw_cache_dir: <경로> → null`, da3 전용 키(`da3_geo_model` / `da3_geo_input_hw` /
    `da3_geo_layers` / `geo_view_sampling` / `geo_num_views` / `geo_first_view_target_s` /
    `da3_cam_token_per_sample`) 삭제. molmo2 5줄과 나머지는 글자 그대로 같다.
  · **코드 변경 0.** `geo_encoder: null` 한 줄이 이미 있던 세 분기를 동시에 탄다 —
    `main/dataset_cfg.py:263` `geo_enabled = bool(getattr(cfg, 'geo_encoder', None))`,
    `main/train_latent_cam_dm.py:482` `build_geo_encoder(cfg) ... if ... else None`,
    `models/camera_diffusion_model_latent.py:261` `has_geo_latent = geo_emb is not None`.
    smoke 로그에 geo 관련 줄이 한 줄도 안 찍히는 것으로 확인했다.
  · `pose_source: da3` / `target_pose_source: da3_target_poses` 는 **카메라 궤적 자체**의
    출처라 그대로 둔다 — geo 스트림과 무관하다.
- **`main/conf/experiment/vista4d_d128_da3_t128.yaml` — D128 뱅크를 쓰는 첫 학습 arm (D131,
  2026-09-05).** D128 뱅크(#165)와 export(#166)는 끝나 있었는데 그걸 참조하는 config 가
  하나도 없었다 (`main/conf/` grep: d121 16회 / d77 9회 / dynpose_d107 17회 / dynpose_d110 4회 /
  **k6_d128 0회**). 즉 `tau_ref follow` · `track_min_drift_u` · anchor 3/3 · `drop_surfaces` ·
  tracking lock 1.0 · `min_zcam` 이 아직 한 번도 학습 신호로 안 들어갔다.
  · 내용은 `vista4d_d121_da3_t128.yaml`(D123)에서 **코퍼스 경로 3줄 + `exp_name` + `epochs`
    200→100** 만 다르다. D123 이 그대로 대조군이 된다.
  · 코퍼스가 작아진다 — train 14,975→9,389 (−37.3%), test 875→592 (−32.3%). 씬 구성은 같지만
    (train 48 / test 4) **게이트가 걸러낸 부분집합**이라 두 arm 의 testset 평균을 그대로
    비교하면 분모가 다른 표본 위의 값이다. 대조는 d128 test 592 위에서 두 ckpt 를 짝지어 낼 것.
  · d128 에는 `molmo2_cache` / `peav_cache` 가 없다 (d121 에만 있다). video CA arm 을 d128 로
    돌리려면 캐시 재굽기가 선행한다 — 이 arm 은 da3 단독이다.
  · `geo_raw_cache_dir` 은 소스 프레임 52편이 같아 `latentcam_da3` 공용 캐시를 그대로 쓴다.
- **`main/video_ca_probe.py` + `scripts/eval_testset.py --probe-video-ca / --probe-timestep /
  --drop-video` — video CA 스트림을 **추론 1건마다** 계측 (D130, 2026-09-05, 사용자 지시
  "video gate가 쓰이고 있는지 validation forward에서 attention이나 gate 같은거 수치 측정해줘.
  각 inference마다 기록해서 preset마다, scene마다 특성이 있는지도 분석해줘").**
  · **왜 gate 만 보면 안 되는가**: D124 epoch100 의 `video_gate` 는
    `[0.0756, -0.0308, -0.0002, -0.0002, 0.0, -0.0004, -0.0003, -0.0]` 로 층 2~7 이 층 0 대비
    ~300배 작다. 그런데 같은 구간에 `video_proj.weight` norm 이 20.88 → 26.52,
    `video_text_proj.weight` 가 22.96 → 29.32 로 **커졌다** (`geo_proj.weight` 는 3.58 → 3.45).
    v 자체가 커졌으면 gate 가 작아도 기여는 유지된다 — 봐야 하는 건 곱
    `vresid_l{i} = |gate_i|·‖v_i‖ / ‖h_i‖` 다.
  · attention map 만으로도 안 된다. `geo_attn_probe`(`train_latent_cam_dm.py:142`) 의 docstring
    이 이미 적어둔 이유 그대로 — softmax 는 **행 합이 항상 1** 이라 "얼마나 쓰는가"가 안 나온다.
    그래서 인과 delta 두 개를 같이 잰다: `dpred_drop`(스트림 제거) 과
    `dpred_shuffle`(video 조건만 배치축 roll — 토큰 통계는 그대로 두고 **scene 짝만** 깬다).
    `dpred_drop` 은 큰데 `dpred_shuffle` 이 0 이면 = 내용과 무관한 bias 로만 쓰고 있다는 뜻.
  · 그 외 열: `vattn_text_mass`/`vattn_video_mass`(molmo2 text 128 vs 프레임 패치 3136 분할),
    `vattn_entropy_norm`(1.0 = 어느 프레임도 안 고름), `vattn_frame_peak`/`_idx`,
    `vattn_frame_token_r`(traj 토큰 인덱스 vs 가중평균 프레임의 Pearson r = 시간축 정렬 독해인가),
    눈금자로 `text_resid_mean`/`geo_resid_mean`.
  · **배치 축을 안 접는다** — 행 하나가 추론 1건이라 `data_name` 으로 preset/scene 에 조인된다.
    노이즈는 `noise_seed=1234`, timestep 은 `--probe-timestep`(기본 500) 한 점 고정 — 안 그러면
    샘플 간/arm 간 수치가 비교 불가다.
  · 기본 off. 켜면 배치당 forward 가 3회 더 붙는다. `video_latent_dim=0` arm 은 프로브가 빈
    리스트를 돌려주므로 켜도 no-op.
  · `--drop-video` 는 `--drop-track` 의 video 판. 같은 ckpt 로 이 플래그만 켜고 끄면 "video
    스트림이 최종 궤적을 얼마나 바꾸나"가 짝지은 A/B 로 나온다.
- **`main/cache_molmo2_embeddings.py` + `main/conf/experiment/vista4d_d121_molmo2.yaml` — D124,
  video CA 스트림을 PE-AV 에서 **Molmo2-4B 융합 hidden state** 로 교체 (2026-09-04, 사용자 지시
  "molmo2 hidden feature를 적당히 shape 맞추고 da3랑 token 개수 맞춰서 resampling해서 CA layer
  추가해서 text CA, molmo2 CA, da3 CA 순으로 통과되게끔").**
  · 동기: PE-AV 는 video tower 와 text tower 가 **끝에서 코사인 유사도로만 만나는** dual-encoder 라
    "이 캡션이 이 영상의 어디를 가리키나"가 토큰 안에 없다. Molmo2 는 vision feature 를 LM 임베딩에
    더해 넣고 36층 causal self-attn 을 태우는 decoder-only VLM 이다.
  · 뽑는 자리: `Molmo2Model.last_hidden_state` = `modeling_molmo2.py:1073` 의 `ln_f` 직후,
    `lm_head` 직전 (사용자 질문 "마지막 decoding stage 전에 video, text attention이 끝난 hidden
    state 뽑을만한 곳 없어?" 에 대한 답).
  · 실측 layout (49프레임 clip): prefix 4,312 토큰 / patch 3,969 = 49 × 9×9. 프레임 **안에서만**
    9×9 → 8×8 평균풀링 → video 3,136 토큰. 캡션 꼬리(캡션 + 고정 probe 문장 + assistant 헤더)는
    코퍼스 전량 min 31 / p50 78 / **max 100** 토큰이라 `text_len 128` 에서 잘림 0. 합 **3,264**
    토큰으로 da3 geo 의 3,456 에 맞췄다 (풀링 없이 9×9 를 그대로 쓰면 4,097 로 18% 초과).
  · dedup 키가 캡션 문자열이 아니라 **(scene_key, caption)** 이다 — 캡션 위치 hidden 은 앞의 영상을
    다 본 값이라 씬마다 다르다. 캡션만으로 묶으면 13,183 이지만 실제 고유 조합은 **13,679**.
  · 반대로 patch 위치 hidden 은 chat template 이 `<|video|>` 를 맨 앞으로 hoist + LM 이 causal
    이라 캡션과 무관하다 → 씬당 1파일. `--verify` 가 두 캡션의 patch hidden 을 실제로 비교해
    **|Δ|max 0.0000** 을 확인했고, 같은 옵션이 prefix 재사용 경로 vs processor 원본 full forward 의
    꼬리 hidden 도 **|Δ|max 0.0000** 으로 확인했다.
  · 효율: 씬당 1회만 ViT(49×729 패치 × 27층)를 태우고 prefix `inputs_embeds` 를 expand 해서
    캡션 배치를 돌린다 (ViT 재실행 0회). `lm_head` 도 안 태운다 — vocab 151,936 × 4,345 위치는
    모델 본체보다 FLOP 이 크다.
- **`video_text_dim` config 키 (`main/conf/config.yaml`) + `train_latent_cam_dm.py:526` 의
  `_vtd = 1024` 하드코드 제거.** `video_text_in_stream: true` 일 때 CA key/value 앞쪽 text part 의
  입력 차원을 config 가 정한다. **기본값 1024 = PE-AV text tower 라 D117 arm A 는 비트 동일**하고,
  D124 만 2560(Molmo2 LM hidden)으로 덮는다. video part 차원은 종전대로 `video_latent_dim`.
- **`main/conf/experiment/vista4d_d121_da3_t128.yaml` — D123, `vista4d_d121_da3` 에서 `text_len`
  512 → 128 **한 줄만** (2026-09-04, 사용자 지시 "학습 멈추고 text_len을 128로 맞춰서 다시 돌려줘").**
  같은 코퍼스·같은 H100 1장에서 arm 2(PE-AV)가 8.03 min/epoch 인데 da3 arm 은 12.02 min/epoch
  였다 (ckpt mtime 실측, 둘 다 01:35 시작). 차이 4.0 min/epoch = 0.128 s/step (1,872 step/epoch).
  · 원인은 `train_latent_cam_dm.py:735` 가 매 스텝 umt5-xxl 인코더를 forward 하는 것이고
    (PE-AV arm 은 캐시를 읽어서 인코더가 아예 안 돈다), 그 forward 가 **항상 512 토큰**을 돈다 —
    `models/t5.py:509-512` 가 `seq_len=text_len` 으로 pad 하고 `seq_lens` 를 계산만 한 채
    `self.model(ids, mask)` 로 전량을 돌린다. umt5-xxl encoder-only = dim 4096 / ffn 10240
    gated / 24층 = 4.63 B param → `2 × 4.63e9 × (8 × 512)` = 37.9 TFLOP/step, / 0.128 s =
    296 TFLOPS (H100 bf16 피크 대비 30% MFU). 관측된 delta 가 이 forward 하나로 설명된다.
  · **왜 128 인가** — 이 코퍼스 고유 캡션 13,183개 전량을 umt5 토크나이저로 재면
    min 12 / p50 62 / p90 73 / p99 79 / **max 89**. 128 이면 잘리는 캡션이 0개(≤96 에서 이미
    100%)이고 32 토큰 여유가 남는다. (같은 캡션을 Molmo2(Qwen) 토크나이저로 재면
    min 10 / p50 57 / p90 67 / p99 73 / max 79 — D124 용 참고.)
  · 실측 smoke: **4.87 it/s vs 기준 arm 2.60 it/s → 6.4 min/epoch** (예상 9.0 보다 빨랐다).
    loss 궤적은 기준 arm smoke 와 일치 — step 0 `1.3289`(동일), step 150 `1.0145`(기준 1.0147).
  · **비교 가능성 주의: 기존 `vista4d_d121_da3` 의 "빠른 판본"이 아니라 새 run 이다.** text CA 의
    key 개수가 512 → 128 로 바뀐다. 잘린 캡션이 없으니 정보량은 같지만 pad 토큰에 걸리던
    attention mass 가 사라진다(`text_mask` 로 마스킹되긴 하나 positional encoding 길이가 다르다).
    peav arm(8.03 min/epoch)과의 대조는 그대로 유효 — 양쪽 다 실제 캡션을 온전히 본다.
  · 코드 변경 없음. 나머지 전부 `vista4d_d121_da3.yaml` 과 글자 그대로 같다 (주석/공백 제거
    diff 에서 `exp_name` 과 `text_len` 두 줄만 다른 것을 확인).

- **`main/conf/experiment/vista4d_d121_{da3,peav}.yaml` — D121 vista 코퍼스 2-arm (2026-09-04).**
  사용자 지시: "vista 최신 데이터를 train, val 나눠서 da3 encoder만 쓴거랑 ... 두 개를 학습",
  이어서 "그냥 PE-AV 먼저 학습 돌려놔줘. text, video encoder 다 PE-AV꺼 쓰고".
  코퍼스는 `latentcam_da3_k6_d121` (씬 52편 / 세그먼트 train 14,975 · test 875). test holdout
  4편(avocado-slice, bmx-bumps, camel, couple-hug)은 d77 과 **같게** 두어 이전 arm 과 눈금이 맞는다.
  · arm 1 `vista4d_d121_da3` — text CA umt5, `video_latent_dim: 0` (video CA 블록 자체가 없음).
  · arm 2 `vista4d_d121_peav` — `text_encoder: PEAV` + `video_latent_dim: 1792`.
    `peav_in_ln`/`video_gate`/`video_text_in_stream` 은 config.yaml 기본값(true/true/false)이
    그대로 맞아서 다시 적지 않았다.
  · `geo_raw_cache_dir` 은 d77/k6 arm 과 **공유**한다 — context 는 소스 영상이라 뱅크와 무관한
    scene 상수이고, d121 의 52편 이름이 캐시 52 파일과 완전히 일치한다(실측 차집합 0).
- **`main/cache_peav_embeddings.py` 코퍼스 분기 2개 (2026-09-04).** 기본값이 d107 때와 **글자 그대로
  같아서** 기존 경로는 손대지 않아도 그대로 돈다.
  · `--seg_prefix`(기본 `dynpose`) — `seg_list_<prefix>_<split>.txt` 를 읽는다. vista 는 `vista4d`.
  · `--caption_source {fields,concise}`(기본 `fields`) — `concise` 는 `caption_fields` 를 규칙으로
    재조립하지 않고 `prompt_camera_with_scene_video.concise` **완성문을 그대로** 굽는다. 이유:
    T5 arm 이 `dataset_scene_decoupled.py:154` 에서 읽는 문자열과 글자 단위로 같아야 두 arm 의
    차이가 인코더 차이로만 남는다. D121 캡션은 target_text/framing_nl/composition 까지 담긴
    완성문이라 재조립하면 오히려 정보가 깎인다. 빈 캡션이 하나라도 있으면 assert 로 죽는다.
  · 저장 dict 의 `template` 이 `--caption_source` 를 반영한다 (`fields` 는 기존대로 `nl`).
  · d121 실측: video 52 scene / caption 13,183종 / **L=84** — 기본 `--text_len 64` 로는 assert 에
    걸린다. `--text_len 0`(실측 최대)으로 구웠다.
- **PE-AV video cross-attention — 레이어 순서 `text CA -> video CA -> geo CA` (D117, 2026-09-03).**
  사용자 지시: "text CA - video emb CA - da3 geo emb CA 순으로 layer를 배치되도록 latentcam 모델
  수정해줘". 인코더는 `facebook/pe-av-large` (perception_models 리비전,
  `tools/perception_models/checkpoints/pe-av-large-pm`) 이고 **얼어 있다** — 입력이 소스 영상과
  캡션뿐이라 학습 중 안 변한다. 그래서 매 step 인코더를 돌리는 대신 `geo_raw_cache_dir` 과 같은
  방식으로 미리 구워 `__init__` 에서 전량 RAM 에 올린다 (fork 로 worker 가 copy-on-write 공유).
  · `models/camera_diffusion_model_latent.py` — video CA 는 `self.layers`(레이어당 7 모듈)를
    10 모듈로 넓히는 게 아니라 **별도 `self.video_layers`** 로 들어간다. 기존 체크포인트의
    `layers.<i>.<0..6>.*` 키가 한 글자도 안 바뀌어야 하기 때문 — 넓혔으면 인덱스가 밀려서 옛
    ckpt 가 전부 못 읽힌다. 검증: 옛 state_dict 를 새 모델에 로드 → `unexpected_keys` 0,
    새 키 30개만 추가되고, `video_emb=None` 이면 출력이 **비트 동일**.
    FiLM 은 `mod1`(self)/`mod2`(geo) 옆에 `mod3` 를 새로 둔다.
  · `_build_video_tok()` — arm A 에서 PE-AV text(1024)와 video(1792)를 **각각** hidden 으로
    투영한 뒤 토큰축으로 잇는다. positional encoding 은 concat **전에 파트별로** 더한다:
    한 번에 더하면 text 길이가 배치마다 달라져 49 프레임 토큰의 위치가 흔들린다.
  · `main/cache_peav_embeddings.py` (신규) — video `(T,1792)` scene 파일 + dedup 된 text
    `(U,L,1024)` 한 파일. **env `vista4d` 에서만 돈다** (latentcam env 에 xformers 가 없어
    PE-AV import 자체가 실패). 학습은 `.pt` 만 읽으므로 env 가 갈려도 무관.
    d107 실측: 264 scene / NL 캡션 2,100종 (L=32, max 26 tok) / 390 s.
  · `main/dataset_dl3dv.py` `_preload_peav()` — 캐시에 없는 scene·segment 가 하나라도 있으면
    **죽는다**. 조용히 빠지면 그 배치 항목만 `peav_*` 키가 없어 `collate_fn` 의 `torch.stack`
    에서 터지거나, 더 나쁘게는 일부만 조건이 붙은 채로 학습된다.
  · config 4키 (전부 기본값이 예전 동작): `peav_video_cache_dir` / `peav_text_cache` /
    `video_latent_dim`(0=off, 1792=on) / `video_text_in_stream`. `text_encoder: PEAV` 도 추가 —
    umt5 를 안 띄우고 PE-AV text 를 text CA 로 올린다 (그때 `text_proj` 입력이 4096→1024).
  · **미지원**: `is_ar` / `per_token_noise`. 트레이너가 assert 로 막는다 (`forward_ar` 는
    애초에 `cond` 도 concat 하지 않는 별도 경로다).
- **`main/conf/experiment/dynpose_d117{a_peavvid,b_peavtext}.yaml` — video CA 2-arm (D117, 2026-09-03).**
  둘 다 d107_k6 와 코퍼스·손실·뷰 샘플링·분모가 같고 PE-AV 줄만 다르다.
  arm A(hybrid) = text CA umt5(struct 캡션) + video CA `[PE-AV NL text | 49 frame]`;
  arm B(full PE-AV) = text CA PE-AV(NL 캡션, umt5 제거) + video CA frame 토큰만.
  캡션이 두 형식인 이유는 실측이다 — target 명사만 스왑한 검색 프로브
  (`scripts/eval/peav_target_match.py`)에서 우리 코퍼스의 struct 형식이 4형식 중 **꼴찌**
  (test top1 0.410) 였고 NL 문장이 1등(0.686, chance 0.071) 이었다. PE-AV 쪽에는 NL 을,
  umt5 쪽에는 기존 struct 를 준다.
- **`main/conf/experiment/dynpose_d110_k6_dd10.yaml` (D110, 2026-09-02).** D109 와 같은
  dd 10%/preset 90% 비율이지만 **데이터를 버리는 대신 늘려서** 만든다 — 사용자: "이러면 개수가
  부족하잖아 현재 비율 유지하되 추가로 preset (track 포함)을 더 만들어줘". d107 τ뱅크는 그대로
  두고 hole 사다리만 2단→4단으로 재fit(`run_dynpose_d110_shard.sh`)해 preset·dd 변이가 양쪽 다
  ~2배가 되므로, dd 를 10% 로 서브샘플해도 총량이 d107 전량(8,512)을 넘는다. d107_k6 대비
  실효 diff 4줄 (exp_name + 코퍼스 경로 3개). geo 캐시는 d84/d107 과 공유 (키가 scene 이름이고
  내용은 소스 영상 feature 라 target 뱅크와 무관, scene 집합도 동일). holdout 도 같은 27편.
- **`filter_seg_list_by_preset.py --target_frac_prefix/--target_frac` — prefix 매칭 preset 을
  최종 리스트의 지정 비율로 서브샘플 (D109, 2026-09-02).** 사용자 지시 "dd를 10%로 낮추고
  preset을 (track 포함) 90%까지" 의 데이터 축. RNG 없이 결정론적 — 매칭 행을 리스트 순서
  위 균등 stride 로 뽑아 scene 에 고르게 퍼진다. 기본값 0 = 예전 동작 비트 동일.
  산출: `latentcam_dynpose_d107/seg_list_dynpose_dd10_{train,test}.txt`
  (train 4,414 = track 55.1% + 일반 34.9% + dd 10.0% / test 518 = 51.5/38.4/10.0).
- **`main/conf/experiment/dynpose_d107_k6_dd10.yaml` (D109, 2026-09-02).** d107_k6 대비
  실효 diff 3줄 (exp_name + seg_list 2개). 코퍼스 디렉토리·geo 캐시·prompts 는 d107 arm 과
  공유 — paired 비교 성립. 주의: 코퍼스가 d107 전량의 58% 라 격차를 dd 비율 효과로만 읽지 말 것.
- **`main/conf/experiment/dynpose_d107_k6.yaml` (D108, 2026-09-02).** d84_k6 대비 실효 diff
  4줄 (코퍼스 경로). d107 코퍼스 = D84 라우팅 + DataDoP 유지, GeoCalib graph 재빌드 +
  deroll + smooth_kf (vista/trumans d99·d106 규약 통일). geo 캐시는 d84 것 재사용
  (키가 scene 이름, 내용물 불변).

### Changed
- **학습을 100 epoch 에서 끊는다 — `epochs` 기본값 2000 -> 100 + 새 상한 `epoch_cap: 100`
  (`main/conf/config.yaml`, `main/train_latent_cam_dm.py`, 2026-09-05).** 사용자 지시
  "학습들도 100epoch되면 꺼주고 이후 돌리는 학습들도 다 100epoch만 돌도록해줘".
  · 기본값만 낮추면 **부족하다** — `conf/experiment/*.yaml` **73개가 `epochs` 를 덮고 있고
    그 중 19개가 200** 이다. 그걸 다시 돌리면 기본값을 무시하고 조용히 200 을 간다. 그래서
    epoch 루프 위에 `_epochs = min(cfg.epochs, cfg.epoch_cap)` 상한을 하나 더 뒀다.
  · `epoch_cap: null` 이면 상한 없음 = **예전 동작 그대로**. 잘릴 때만 `[epoch_cap] cfg.epochs=200
    -> 100` 을 main process 에서 찍는다 (조용히 안 자른다).
  · 옛 experiment yaml 의 `epochs: 200` 은 **그 run 이 실제로 무엇으로 돌았는지의 기록**이라
    건드리지 않았다. 지금까지도 비교에 쓰인 건 항상 `ckpt_at_epochs: [50, 100]` 의 epoch100 이다.
  · 검증: hydra resolve 실측 — `vista4d_d121_molmo2` / `dynpose_d110_k6_dd10` 둘 다
    `epochs=200 epoch_cap=100 -> effective=100`.
  · 같은 지시로 돌던 arm 2개를 SIGINT(Ctrl+C 와 동일 신호)로 정지했다: D123
    `20260904_165917_vista4d_d121_da3_t128`(ol2mue8s, Epoch 148) / D124
    `20260904_185554_vista4d_d121_molmo2`(w5ygw8ii, Epoch 101). **둘 다 `epoch100.pth` 가
    디스크에 있다** (da3_t128 09-05 06:33 / molmo2 09-05 12:49). 종료는 `KeyboardInterrupt`
    후 wandb 정상 마감, 잔여 worker 0, GPU 1·2 반납 확인.
- **umt5 `text_len` 512 -> 128 (`main/config.py`, `main/config_large.py`, 2026-09-03).**
  사용자 지시: "128로 바꿔주고". `models/tokenizers.py:49` 가 `padding='max_length',
  truncation=True, max_length=self.seq_len` 이라 `text_len` 은 **예산이 아니라 고정 패딩 길이**다 —
  512 로 두면 짧은 캡션도 512 슬롯을 채워 umT5-xxl 인코더가 매 스텝 그 길이를 통째로 돈다.
  실측 최장은 d121 nl 캡션 82 tok / DataDoP concise 77 tok (n=2,000, p50 40, p99 65) 라
  512 중 84% 가 pad 였다.
  **출력은 안 바뀐다** (코드 판독, 수치 확인은 아직): text cross-attention 이
  `key_padding_mask=~text_mask` 로 pad 를 빼고 (`camera_diffusion_model_latent.py:308`, `:407`,
  `camera_diffusion_model_base.py:167`, `_attn_sup.py:165`), text 쪽엔 위치 임베딩이 없으며
  (`text_proj`/`text_ln` 은 per-token), umT5 자신도 attention mask 를 받는다. → `text_len ≥`
  실제 토큰 수이기만 하면 기존 체크포인트와 호환된다.
  학습(`config.py`)과 추론(`config_large.py`: `infer_latent_cam_dm.py`/`infer_cam_dm.py`/
  `eval.py`/`valid_cam_acc.py`)을 **같이** 내렸다 — 둘이 어긋나면 안 된다.
  `config_vae.py` 는 그대로 512 다: VAE 경로(`train_vae.py`/`infer_vae.py`)는 텍스트를 아예
  안 읽어 죽은 값이라 건드릴 이유가 없다.
  코퍼스별 실측 (umT5, `prompt_camera_with_scene_video`; n=샘플 프롬프트 수):

  | 코퍼스 | n | p50 | p99 | max | >128 |
  |---|---|---|---|---|---|
  | worldtraj/dynamicverse | 405 | 111 | 198 | **240** | **124 (31%)** |
  | Scene-Decoupled/da3 | 400 | 22 | 36 | 37 | 0 |
  | worldtraj/DL3DV | 2,595 | 26 | 47 | 61 | 0 |
  | DynPose latentcam_dynpose | 8,796 | 17 | 26 | 28 | 0 |
  | DynPose d107 | 8,512 | 18 | 26 | 28 | 0 |
  | TRUMANS-Lite d77 | 61,671 | 20 | 25 | 26 | 0 |
  | TRUMANS-Lite | 260 | 26 | 33 | 33 | 0 |
  | Vista4D k6_d77 | 13,705 | 18 | 23 | 25 | 0 |
  | Vista4D latentcam_da3 | 9,950 | 19 | 26 | 28 | 0 |
  | DataDoP concise | 2,000 | 40 | 65 | 77 | 0 |

  **dynamicverse 만 128 을 넘는다** (scene 서술이 `**Detailed**: ...` 로 길다). 지금 돌리는
  dynpose/vista/trumans arm 은 전부 안전하지만, worldtraj arm 을 다시 돌릴 거면 그 config 의
  `text_len` 을 256 이상으로 올려야 한다.
- **`models/tokenizers.py` — truncation 이 일어나면 경고한다 (2026-09-03).**
  위 표의 dynamicverse 같은 경우가 **조용히** 잘리는 걸 막는다. "마지막 토큰이 EOS 인가"로는
  못 잡는다 — HF 는 자르고 **나서** special token 을 붙이므로 잘린 시퀀스도 EOS 로 끝난다
  (실측). 그래서 attention mask 가 `seq_len` 을 꽉 채운 항목에 한해 truncation 없이 한 번 더
  토크나이즈해 실제 길이를 재고, 넘으면 `warnings.warn` 한다. 꽉 차는 일 자체가 드물어 상시
  비용이 아니고, 래치(`_trunc_warned`)로 **프로세스당 한 번만** 뜬다.
  검증: 21 tok 캡션 경고 0 / 241 tok 캡션 경고 1 (`실제 241 tok` 이 메시지에 찍힘) / 재호출
  경고 0 (래치) / `seq_len=512` 로는 241 tok 도 경고 0 / 짧은+긴 혼합 배치 경고 1.

### Fixed
- **CLaTr 학습이 wandb 를 켜면 epoch 0 validation 끝에서 죽었다 — 신버전 의존성 두 건
  (FIX-D153, 2026-09-06).** 둘 다 `main/evaluate/CLaTr/src/models/clatr.py` 의
  **validation 궤적 플롯 경로**에 있다. 이 경로는 logger 가 있을 때만 돌기 때문에
  `WANDB_MODE=disabled` smoke 가 통째로 건너뛰었다 — "smoke 는 wandb 끄고" 규칙의 사각지대.
  1. `draw_traj_plots` 가 float32 pose 를 evo 에 넘겼다. `evo/core/transformations.py:1145`
     의 `numpy.array(M, dtype=float64, copy=False)` 는 dtype 변환에 복사가 필요한데 numpy 2
     는 그걸 `ValueError: Unable to avoid copy` 로 만든다 (numpy 1.x 는 조용히 복사했다).
     → `.numpy()` 를 `.double().numpy()` 로 (ref/t/m 3곳). 플롯을 끄는 대신 dtype 을 맞춘
     이유는, 끄면 evo 가 정상인 환경에서도 산출물이 조용히 사라지기 때문.
  2. `canvas.tostring_rgb()` 는 matplotlib 3.8 deprecate / **3.10 삭제** (여기 3.10.8).
     → `hasattr` 로 갈라 없으면 `np.asarray(canvas.buffer_rgba())[..., :3]`. 구버전 mpl 은
     기존 경로 그대로.
- **`scripts/eval_testset.py` 가 video CA arm 을 평가할 수 없었다 — 두 군데 (FIX-D130,
  2026-09-05).**
  1. **모델 생성자에 video CA 인자가 없었다.** `CameraDiffusionModel(cam_dim=..., cond_dim=...,
     **_geo_kw)` 라 `video_layers`/`video_gate`/`mod3` 등 **107 키가 아예 안 만들어진다**.
     D124(`vista4d_d121_molmo2`) 를 이 스크립트로 평가하면 `strict=True` 로드가 "Unexpected
     key(s)" 로 죽는다 — 즉 **D124 는 지금까지 이 경로로 평가 자체가 불가능했다**.
     `train_latent_cam_dm.py:521-532` 와 같은 `_vid_kw` 블록을 이식했다.
  2. **`T.sample(...)` 에 `video_kw` 를 안 넘겼다.** 1번을 고쳐도 이게 남으면 조용히 틀린다 —
     `T.sample` 기본값이 `video_kw=None` → `**(video_kw or {})` = 빈 dict → `forward` 가
     `has_video=False` 로 떨어져 **video CA 8층을 통째로 건너뛴다**. 학습 쪽
     `run_validation`(`train_latent_cam_dm.py:770`)은 `build_video_cond` 를 부르므로 wandb val
     과 이 스크립트가 서로 다른 모델을 재고 있었을 것이다. 예외도 경고도 안 난다.
  · **영향 범위는 `video_latent_dim > 0` 인 arm 뿐**이다. 그 외 arm 은 `build_video_cond` 가 빈
    dict 를, `_vid_kw` 가 빈 dict 를 돌려주므로 **기존 testset eval 수치는 전부 그대로 유효**하다.
- **`dpred_shuffle` 이 대조군 구실을 못 하고 있었다 — `dpred_xscene` 추가 (FIX-D130c,
  2026-09-05).** `video_ca_probe` 의 `dpred_shuffle` 은 video 조건을 배치축 `roll(1)` 로 바꿔
  치는데, eval 배치가 seg 순서라 **roll 짝의 99.3% (875 중 869) 가 같은 scene 의 이웃 seg** 다.
  같은 씬 49프레임 클립은 molmo2 임베딩이 거의 같으므로 이 값이 작은 건 모델 성질이 아니라
  배치 구성의 결과다 — 실측 전량 평균 0.00161 을 "video 내용을 안 본다"로 읽으면 틀린다.
  · 수정: `video_ca_probe(..., video_kw_alt=...)` 와 열 `dpred_xscene`. `eval_testset.py` 가
    scene 별 video 조건을 1건씩 모아 두고 **현재 배치와 다른 scene** 것을 넘긴다. 첫 씬은
    비교 대상이 없으므로 본 루프 전에 두 번째 씬이 나올 때까지만 미리 훑어 씨앗 1건을 심는다
    (GPU forward 없이 로더만 돈다).
  · `dpred_shuffle` 은 지우지 않았다. 열이 틀린 게 아니라 **판정에 쓰면 안 되는** 열이라,
    probe / analyze 양쪽 docstring 에 그 뜻을 적고 표의 기본 열에서만 뺐다.
- **video CA 를 0 초기화 residual gate 로 붙인다 — `video_gate` (D117, 2026-09-03).**
  D117 두 arm 이 55 / 84 epoch 동안 loss ~1.0 (= eps 예측이 0) 에서 못 빠져나온 **진짜 원인**.
  격리 실험 3종으로 좁혔다 (전부 d107 전체 코퍼스, epoch 0, batch 8, 952 step):
  | 실험 | step 250 | 300 | 350 | 500 | 700 |
  |---|---|---|---|---|---|
  | D108 (기준) | 0.6956 | 0.1774 | 0.1784 | — | — |
  | 신규 코드 + video CA off + **peav 캐시 로드** | 0.6956 | 0.1774 | 0.1784 | — | — |
  | text CA 를 PE-AV 로 교체 + video CA off | 0.9757 | 0.9537 | 0.8483 | 0.2975 | 0.1620 |
  | video CA on (LN 만 적용) | 0.9975 | 0.9985 | 0.9898 | 0.9651* | — |
  즉 **데이터 경로는 D108 과 비트 동일**(dataset 이 배치에 `peav_*` 키를 넣는 것 자체는 무해)
  이고 **PE-AV text 를 text CA 로 쓰는 것도 정상 학습**한다. 남는 변수는 video CA 스트림 하나.
  원인은 스케일이 아니라 **잔차 구조**였다 — `CrossAttention.forward` 가 `norm(x + a)` 를
  돌려주고 각 스트림이 `h = CA(...)` 로 h 를 **덮어쓰기** 때문에, 스트림이 하나 늘 때마다 층당
  x_t 성분이 한 번 더 정규화로 깎인다. num_layers=8 에서 2스트림 대비 3스트림의 누적 감쇠가
  ~11배다. → `video_gate=True`(기본) 이면 video 블록만 `h = h + gate[l] * v` 로 붙고
  `gate` 는 **0 초기화**(DiT/Flamingo 방식)라 학습 시작 시점 출력이 D108 과 **비트 동일**하고
  (실측 `max|diff| = 0.0`, state_dict 신규 키 107개 중 gate 1개, `unexpected_keys` 0) 필요한
  만큼만 열린다. `video_gate=False` 는 예전 덮어쓰기 동작 — 기존 D117 ckpt 재현용으로 남긴다.
  `video_latent_dim=0` 이면 이 키와 무관. (*마지막 칸은 arm B 값)
- **PE-AV feature 를 projection 앞에서 LayerNorm 한다 — `peav_in_ln` (D117, 2026-09-03).**
  frozen encoder 출력을 bare `nn.Linear` 에 그냥 먹이고 있었다. DA3 geo 는 그렇지 않다 —
  `models/da3_geo_encoder.py:171/:407` 의 학습 가능한 `self.ln` (native 3072 위 LayerNorm)을
  거친 뒤에야 `geo_proj` 로 들어간다. umt5 는 출력이 이미 ~unit 이라 LN 없이도 통했지만 PE-AV 는
  아니다 — d107 캐시 실측 video std 8 / text std 95 · absmax 12928. 그 결과 `video_tok` std
  26.15 (text part 41) 대 `text_tok` 0.58 / `geo_tok` 0.58 로 **45~70배** 차이가 나서 video CA
  logit 이 포화했다.
  → `camera_diffusion_model_latent.py` 에 `video_ln` / `video_text_ln` / `text_ln` 을 넣고
  `peav_in_ln`(기본 true) · `text_in_ln`(기본 **false**) 로 분기한다. `text_in_ln` 은
  `text_encoder: PEAV` 일 때만 트레이너가 켜므로 **umt5 arm 은 파라미터·출력이 그대로**이고,
  `video_latent_dim=0` 이면 새 키가 아예 생성되지 않는다 (state_dict 신규 키 0 확인).
  적용 후 `video_tok` std 26.15 → 0.76.
  **다만 이것만으로 D117 두 arm 의 loss ~1.0 정체는 안 풀렸다** — 원인 규명은 진행 중이고,
  같은 코드 + `video_latent_dim=0` + 전체 코퍼스 대조군은 D108 과 step 350 까지 **비트 동일**
  (0/50/…/350 = 1.2942/0.9648/0.9907/1.0239/0.9879/0.6956/0.1774/0.1784) 이라 기존 경로는
  무손상이다. 확정되면 `FIX.log` 에 기록한다.
- **`eval_testset.py` 가 `target_track` arm ckpt 를 못 읽던 것 (2026-08-28).**
  `CameraDiffusionModel(cam_dim=..., **_geo_kw)` 에 `cond_dim` 을 안 넘겨서 `cam_in` 이
  `Linear(64, 512)` 로 만들어지고, ckpt 의 `Linear(68, 512)` 와 strict load 에서 shape
  mismatch 로 즉사했다 (`cam_in.weight` `[512,68]` vs `[512,64]`). 아래 `target_track_dim`
  항목의 "eval_testset.py — 동일하게 조건 전달" 은 **호출부만** 맞고 생성자가 빠져 있었다.
  → `cond_dim=_track_dim` 을 넘긴다. `target_track_dim=0` 이면 생성자 기본값과 같아
  기존 arm 은 글자 그대로 동일. strict load 라 조용히 틀린 평가가 나올 여지는 없었지만,
  그만큼 **이 커밋 전까지 track arm 은 한 번도 eval 된 적이 없다**. 자세한 내역은 `FIX.log`.

### Added
- **`scripts/snowboard_prompt_ab.sh` — `target:` 절 A/B 드라이버 (D104, 2026-09-01).**
  snowboard 뱅크(`hole_bank_k6`, 77 변이)를 k6 posed 모델(`20260827_051423_vista4d_pgt_k6`,
  ckpt `last`)로 두 번 돌린다. arm 은 **프롬프트뿐** — `target: <label> motion: ...` vs
  `motion: ...`. 두 export 의 `target_poses` 는 비트 동일(`|dE|max 0.0`, `|dK|max 0.0`)이라
  예측 차이는 전부 텍스트 탓이다. snowboard 는 train/test 어느 split 에도 없어서 전용 export
  + 0줄 `train_seg_list` 를 쓰고, `eval_testset.py --set` 으로 `dl3dv_root`/`train_seg_list`/
  `test_seg_list` 세 키만 덮는다 (나머지 config 는 학습 그대로).
  - **결과: `target:` 절은 학습에서 본 라벨에만 도움이 된다.** `subject_in_frame`(center_box
    0.80) 짝지은 차이 `target − notarget` 은 전체로는 **−0.0954** (W/T/L 25/12/40) 인데
    앵커별로 부호가 갈린다:

    | anchor | n | 학습 코퍼스 등장 | Δsif | W/T/L | target zero-frame |
    |---|---|---|---|---|---|
    | man | 27 | 1060회 (22 씬, 최빈 앵커) | **+0.0597** | 15/2/10 | 12.9 / 49 |
    | snowboard | 27 | **0회** | **−0.3477** | 1/4/22 | 32.3 / 49 |
    | helmet | 23 | **0회** | +0.0186 | 9/6/8 | 24.4 / 49 |

    즉 zero-shot 라벨에 `target:` 을 붙이면 도움이 없는 정도가 아니라 **궤적이 무너진다**
    (snowboard 는 49프레임 중 32.3프레임에 subject 픽셀이 0). 라벨 빈도는
    `latentcam_da3` train split(46 씬 / 9373 entry / 78 라벨)에서 셌다.
  - **hole 을 품질로 읽지 말 것**: GT(bank) hole 0.5084 vs target 0.2324 / notarget 0.1867.
    pred hole 이 낮은 건 좋아서가 아니라 **모델이 뱅크 GT 카메라보다 덜 움직여서**다.
  - 산출물: `results/20260901_snowboard_prompt_ab/{target,notarget}__last`,
    `CinemaTraj/results/20260901_snowboard_prompt_ab/{warp/, subject_in_frame.json,
    target_clause_7cases.mp4}`. 후처리는 기존 `render_pred_depth_warp.py` /
    `eval_subject_in_frame.py` / `concat_videos.py` 를 **수정 없이** 재사용.

- **dynpose `dolly_in_look_at` 단일 preset arm — experiment `dynpose_d84_k6_dionly` (D95,
  2026-09-01).** dynpose-0000 씬 전량에서 preset 을 **하나로 좁힌** 코퍼스. 코드 변경은
  `scripts/data/filter_seg_list_by_preset.py` 신규 1개 + hydra config 1개뿐이고, 학습 코드는
  안 건드렸다 (`dynpose_d84_k6` 위에 `exp_name`/`train_seg_list`/`test_seg_list` 3줄 override).
  - **이름 함정**: 뱅크 seg 이름에는 `dolly_in` 으로 적혀 있고 그게 D76 이후의
    `dolly_in_look_at` 이다 (`lbm/presets.py` 의 `LEGACY_AIM_COLLISIONS`). 그래서 필터는
    `row_preset()` 이 아니라 **seg 이름 문자열**로 걸러야 맞는다 — 필터 스크립트 docstring 에
    같은 내용을 적어 뒀다.
  - **코퍼스가 26배 작다**: train 299 / test 37 (전량 arm `dynpose_d84_k6` 는 7,845 / 951).
    preset 을 하나로 좁히면 anchor 최대 2개 × 씬 수가 상한이라 그렇다. 전량 arm 과 나란히
    비교할 때 "데이터가 적어서"와 "preset 이 하나라서"를 가르지 못한다는 점을 감안할 것.

- **TRUMANS D77 부분 코퍼스 arm — experiment `trumans_d77_k6` (2026-08-31).**
  8샤드로 굽는 중인 737편 중 **recording 이 통째로 끝난 11개 recording = 163 chunk** 만 모아
  먼저 돌리는 조기 신호용 arm. 코드 변경은 없고 `conf/experiment/trumans_d77_k6.yaml` 하나 추가.
  - 바뀐 축은 **코퍼스 하나뿐**이다 — `pose_source`/`target_pose_source`/`scale_mode`/
    `avg_scale_ref`/`intr_norm`/`cam_dim`/`geo_*`/`anchor_pred_frame0`/`batch_size` 는
    `vista4d_pgt_k6_d77.yaml` 과 한 글자도 다르지 않다. `dl3dv_root` 를
    `TRUMANS-Lite/latentcam_da3_d77`, `meta_csv` 를 `meta_trumans.csv` 로 갈아끼운 것뿐.
  - **반쯤 구워진 recording 은 제외**했다 (`tru_1d19e06d` 8/14, `tru_1d43e076` 1/18).
    chunk 172개 중 163개만 쓰는 이유 — 전량 arm 과 나중에 비교할 때 "어느 chunk 가
    들어갔나"가 재현돼야 한다.
  - **분할은 recording 단위** (holdout `tru_0ab19ed6` 9 + `tru_1a1e205b` 12 = 21 chunk /
    163 = 12.9%). chunk 단위로 자르면 49프레임 슬라이딩이라 같은 recording 의 이웃 chunk 가
    train/test 에 동시에 들어가 test 가 새 씬 일반화를 못 잰다.
    실측 train 142 scene / 53,449 segment, test 21 scene / 8,222 segment.
  - **`epochs: 50`** — `val_step` 이 step 기준이라 epoch 수 = 총 optimizer step 수다.
    batch 8 에서 53,449/8 = 6,681 step/epoch × 50 = 334k step 으로, 기존
    `vista4d_pgt_k6_d77`(1,627 × 200 = 325k) 과 **같은 학습량**이다. 200 을 쓰면 4배를 돈다.
    `ckpt_at_epochs: [15, 30]` 도 같은 비율(~100k / ~200k step).
  - `geo_raw_cache_dir` 미리 구움: scene 당 42.5 MB × 163 = **6.92 GB**. context view 가
    (s,e) 와 무관하게 scene 상수라 캐시가 없으면 같은 DA3 forward 를 segment 328개마다
    되풀이한다. 캐시가 B=1 로 구워지므로 `da3_cam_token_per_sample: true` 가 필요하다.
  - seg-list 파일은 `TRUMANS-Lite/latentcam_da3_d77/seg_list_trumans_{train,test}.txt`
    (데이터 디렉토리라 커밋에 안 들어간다).
- **DynPose D84 대조군 2종 — experiment `dynpose_d84_k6_nodd` / `dynpose_d84_k6_ddswap` (2026-08-31).**
  기존 `dynpose_d84_k6` 코퍼스에서 **DataDoP 외부 궤적(`dd_*`)의 기여를 가르기 위한** arm 두 개.
  코드 변경은 없고 `conf/experiment/` 에 yaml 두 개를 추가한 것뿐이다 — 둘 다
  `defaults: [dynpose_d84_k6]` 로 상속하고 `exp_name` / `train_seg_list` / `test_seg_list`
  **3줄만** 덮어쓴다. 코퍼스 디렉토리 · `geo_raw_cache_da3`(11.30 GB) · `avg_scale` ·
  `prompts.json` · holdout 27편은 세 arm 이 **그대로 공유**하므로 재-export 도 캐시 재빌드도 없다.
  - `dynpose_d84_k6_nodd` — `dd_*` 세그먼트를 전량 제거. train 7,845 -> **4,169**, test 507.
  - `dynpose_d84_k6_ddswap` — nodd 와 **총량·targeted 세그먼트가 글자 그대로 동일**하고
    targetless 617칸의 **궤적 출처만** 우리 `pan_*` <-> DataDoP `dd_*` 로 갈린다.
    train own_targeted 3,552 + dd 617 = 4,169 (targetless 0.1480),
    test own_targeted 432 + dd 75 = 507 (targetless 0.1479).
  - **왜 "d84 를 그냥 내려깎기"가 아닌가**: `dd_*` 는 코퍼스에서 100% targetless 다
    (실측 dd 4,120 전량 targetless / targeted 0). own 을 다 남긴 채 dd 만 줄여 targetless
    14.8% 를 맞추려 하면 `(617+n)/(4169+n)=0.148 -> n=0`, 즉 nodd 와 같은 코퍼스로 **퇴화**한다.
    균형을 유지하며 dd 를 넣으려면 targetless 슬롯의 내용물을 바꾸는 수밖에 없다.
  - dd 617 선정: 씬별 층화 추출(최대잔여법) + `np.random.default_rng(42)`. dd 를 가진 206개 씬
    **전부**에 최소 1개가 들어간다(dd 가 없는 씬 33개는 targeted 만 남음). 씬 수 239 로 d84/nodd 와 동일.
    특정 영화 몇 편이 617칸을 독식하는 것을 막기 위함.
  - 집합 관계 실측 (train / test): `ddswap` 중 d84 에 없는 것 **0 / 0**, `ddswap ∩ nodd` 3,552 / 432,
    `nodd only` 617 / 75, `ddswap only` 617 / 75.
  - seg-list 파일은 `latentcam_dynpose/seg_list_dynpose_{nodd,ddswap}_{train,test}.txt`
    (데이터 디렉토리라 커밋에 안 들어간다).
- **`avg_scale_ref: ctx_all_first_cam` + experiment `vista4d_pgt_k6_d77_ctxall` (2026-08-30).**
  `norm_scale` 의 **분모만** 바꾸는 arm. Vista4D 뱅크에서 `avg_scale_context_first_cam`
  디렉토리에 실제로 들어 있는 값은 DL3DV 정의(context range 점군, conf>=P40)가 아니라
  `scene_graph.json:scale.S` 다 — 즉 **frame0 한 장**의 non-sky 평균 ray 길이
  (`vista4d_bank_to_dl3dv.py:183` 이 그 값을 복사해 넣는다). context range 는 소스 영상
  전체 `[0,49)` 인데 분모만 첫 프레임에서 나온 상태였다. 같은 이름 아래 정의가 이미 셋이라
  (DL3DV / Vista4D / TRUMANS) 디렉토리를 새로 판다.
  - `S      = mean over **frame0** non-sky px  of  z * ||K^-1 [u+.5, v+.5, 1]||`
  - `ctxall = mean over **전 프레임** non-sky px of  ||p_world - c2w[0][:3,3]||`
    기준점(첫 카메라)은 그대로, 점 집합만 구간 전체. 둘 다 소스 영상만으로 계산되므로
    target 누수가 없고 추론 때 복원 가능하며 scene 상수다.
  - 배선은 `dataset_cfg.AVG_SCALE_DIRS` 에 키 하나 추가한 것뿐이다 — `AVG_SCALE_REFS` 와
    `dataset_dl3dv._avg_scale_dir` 이 거기서 파생되므로 기존 8개 ref 의 동작은 글자 그대로
    같다. `config.yaml` 기본값도 `centroid` 그대로.
  - 데이터 생성기 `models/Planner/CinemaTraj/scripts/make_avg_scale_vista4d_ctxall.py`
    (그 트리는 `.gitignore:223 camera_generation/models` 라 커밋에 안 들어간다).
    정의는 `trumans_to_recon.py:avg_scale_first_cam` 을 **import 해서** 쓴다 — 재구현하면
    sky 마스크 / ray 곱 / pixel stride 중 하나가 조용히 어긋난다. stride 2.
    52편 / 13,705 변이 전량 기록(ok 52 / skip 0 / fail 0). 영상마다 같은 로더로 frame0 `S`
    를 재계산해 뱅크 저장값과 rel 1e-6 안에서 일치하는지 assert -> 52/52 통과.
  - 분모 변화 실측 `ratio = ctxall / S`: 영상 단위 med 1.0110 / p05 0.9058 / p95 1.3271,
    변이 가중(13,705) med 1.0039 / p05 0.8705 / p95 1.3445 / mean 1.1300.
    `|log10 ratio| > 0.1` 인 변이는 1,031 / 13,705 = **7.52%** 뿐이다. 큰 쪽:
    camera-lens 6.1598(n=284) · snowboard 1.3883(64) · couch-sit 1.3445(490) ·
    desert-park 1.3129(193) · soapbox 0.8258(225) · car-roundabout 0.8705(556).
    camera-lens 는 `z_med` 가 프레임 0->48 에서 1.059 -> 20.844 로 단조 증가하는 pull-out
    이고 sky frac max 0.035 라 마스크 누수가 아니다 — frame0 분모가 구간을 대표하지 못하는
    바로 그 경우다.
  - experiment yaml 은 `vista4d_pgt_k6_d77.yaml` 과 **`avg_scale_ref` 한 줄만** 다르다.
    `vae_latent_scale` 은 일부러 안 건드렸다 (ckpt 속성, `config.yaml:168` = 0.96032625).
    분모가 바뀌면 diffusion 입력 std 도 같이 바뀌지만 변이 가중 `1/ratio` 가 med 0.9961 /
    mean 0.9683 이라 코퍼스 전체로는 3% 안이고, 두 arm 이 분모 하나만 다르려면 이 값이
    같아야 한다.
  - smoke 통과 (`WANDB_MODE=disabled`, GPU 0, `epochs=1 val_max_batches=2`, exit 0):
    `avg_scale_ref=ctx_all_first_cam` / train 13,016 · val 689 / geo raw cache 52/52
    (2.21 GB) / `val/loss_traj 0.281759 @ep0 -> best.pth`. (n=16 미학습 값이라 판정용
    아니고, 분모가 다르면 loss 눈금 자체가 달라 posed arm 의 0.274468 과 비교 불가.)
- **D77 뱅크용 experiment config 2종 — `vista4d_pgt_k6_d77` / `vista4d_pgt_k6_d77_track_d5`
  (2026-08-29 추가, 2026-08-30 이름 정정).**
  각각 `vista4d_pgt_k6.yaml` / `vista4d_pgt_k6_track_d5.yaml` 을 그대로 복사하고 **데이터
  경로 3줄 + `exp_name` 만** 바꿨다. 학습 하이퍼는 한 글자도 안 건드렸다 — k6 대비 달라진
  것이 데이터뿐이어야 두 뱅크가 비교가 되기 때문이다. `_track_d5` 는 posed arm 과
  `target_track_dim: 4` / `target_track_dropout: 0.5` / `target_track_val_drop: true`
  세 줄만 다르다 (k6 때의 그 세 줄 그대로).
  - **한때 `k7` 이라고 불렀는데 오해를 부르는 이름이었다.** `k<N>` 은 `--aim_keyframes N`
    이고 두 뱅크 모두 `fixed.aim_keyframes` 가 **6** 이다 — keyframe 수는 안 바뀌었고 바뀐
    것은 preset pool 하나뿐이라 (`bank/` -> `bank_d77/`, 20 -> 34종) `k6_d77` 로 고쳤다.
    데이터 root 도 `latentcam_da3_k7` -> `latentcam_da3_k6_d77`.
  - 데이터: `latentcam_da3_k6_d77` = preset 20종 -> **34종**(`track_*` 10종 신규,
    `follow_gain 1.0`), D76 이름 규칙 통일, 영상 51 -> **52편**(snow-bike 추가),
    dedup 후 변이 9,873 -> **13,705** (train 13,016 / test 689).
    test holdout 4편(camel, avocado-slice, bmx-bumps, couple-hug)은 **k6 와 같다** —
    두 뱅크의 val/test 눈금을 맞추기 위해서다.
  - `geo_raw_cache_dir` 은 **k6 arm 과 같은 디렉토리를 가리킨다** (52편, 2.2 GB 한 벌).
    context 는 소스 영상이라 뱅크와 무관하게 scene 상수이고, 두 root 의 `images_4` 가
    byte 단위로 같다(camel/rhino/women-talk × frame 0/24/48 md5 일치). 읽기 경로
    (`dataset_dl3dv.py:1568`)는 dtype(fp32)만 보고 `meta` 는 안 보므로, 굽는 스크립트가
    `meta` 에 넣는 `exp` 문자열이 무엇이든 학습 결과에 영향이 없다.
    복제하면 2.2 GB 를 한 벌 더 쓴다.
  - smoke 통과(둘 다 1 epoch 1627 step + val, `WANDB_MODE=disabled`):
    posed `val/loss_traj=0.274468`, track_d5 `val/loss_traj=0.272950`, 캐시
    `preloaded 52/52 scenes (2.21 GB)`.
- **`geo_raw_cache_dir` — DA3 backbone 출력(pre-`ln`)을 scene 키로 캐시 (2026-08-29).**
  Vista4D 는 `geo_view_sampling='context_uniform'` 이라 context view 가 **scene 만 보고**
  정해진다 (실측: 9,873 변이 / 50 scene, geo_idxs 가 전 샘플 `(0,10,19,29,38,48)` 로 상수).
  그래서 변이 9,873개가 scene 파일 50개를 공유한다. 자르는 지점은 `ln` **직전**이라
  학습 대상인 `DA3SceneEncoder.ln` / `GeoEncoder.proj` 는 그대로 gradient 를 받는다 —
  기존 `cache_geo_embeddings.py` 가 `proj` **뒤**를 얼려서 da3 에서 못 쓰던 것과 다른 지점이다.
  `scripts/data/cache_geo_raw_da3.py` 로 굽고 (`<dir>/<scene_key>.pt`), 파일이 없는 scene 은
  조용히 on-the-fly DA3 로 떨어진다. `geo_raw_cache_preload`(기본 true)면 `__init__` 에서
  전량 RAM 적재 후 fork 로 worker 가 공유한다 (2.12 GB 한 벌).
  `dataset_cfg.py` 가 성립 조건(da3 / context_uniform / cam_embed null / shuffle false /
  swap null / test_inseg_k 0 / latent_cache 미사용)을 전부 검사하고 하나라도 어긋나면 끈다.
  기본값 `null` = 기존 동작 그대로.
  - **dtype 은 fp32 다.** 학습이 bf16 autocast 아래서 돌아도 `encode_raw` 출력은 fp32 다 —
    autocast 는 matmul/linear 만 내리고 DA3 블록이 `x = x + attn(ln(x))` 라 residual stream 이
    fp32 로 남는다. bf16 으로 저장하면 왕복에서 `max|d| 1.9993` / `||d||/||a|| 1.78e-3` 의
    **scene 마다 고정된 편향**이 생겨 캐시 없이 도는 추론과 어긋난다. 낡은 bf16 파일은
    로더와 `is_done()` 이 dtype 으로 걸러 각각 on-the-fly 폴백 / 덮어쓰기로 처리한다.
  - 실측: `geo_emb` 캐시 vs on-the-fly `max|d| 0` (9샘플 전부), `geo_mask` 동일,
    `ln.weight.grad 3.88e+06` / `proj.weight.grad 1.41e+10` / backbone grad 0.
    epoch 시간 `12:01 -> 07:40` (1.63 -> 2.54 it/s), geo forward 단독 374.30 -> 41.26 ms/step.
- **`da3_cam_token_per_sample` — `cam_enc` 를 샘플별로 호출 (2026-08-29).**
  수학적으로는 배치가 안 섞이는데도 `cam_enc` 를 `B>1` 로 부르면 cuBLAS 가 다른 kernel 을
  골라 `cam_token` 이 `rel 2.9e-7` 달라진다. 그 자체는 무시할 크기지만 **DA3 backbone 이
  5만배로 증폭한다** — `cam_token` 은 `x[:, :, 0]` 에 꽂혀 전 view 가 attend 하는 자리라,
  pre-`ln` 토큰에서 `rel 2.3e-2` (token cos 평균 0.999744 / 최소 0.968340) 가 된다.
  즉 **기존 da3 arm 은 전부 `batch_size=8` 학습과 `batch_size=1` 추론에서 서로 다른 geo
  token 을 보고 있었다.** 배치 *구성*은 무관하고 (동료 scene 을 바꿔도 `max|d| 0`)
  *크기*만 문제라, 학습 자체는 자기 일관적이었지만 추론과 갈렸다.
  `true` 면 샘플별로 쪼개 불러 배치 크기에 완전히 불변이 된다 (per-sample[0] vs 단독 B=1:
  `max|d| 0`). 비용은 B=8 에서 `cam_enc` 3.26 -> 26.05 ms, backbone 468.3 ms 대비 **+4.9%**.
  기본값 `false` = 기존 arm 재현용. `geo_raw_cache_dir` 은 캐시를 B=1 로 굽기 때문에
  **`true` 여야** 캐시 경로와 on-the-fly 폴백이 일치하며, `dataset_cfg.py` 가 경고한다.
  `vista4d_pgt_k6.yaml` / `vista4d_pgt_k6_track_d5.yaml` 두 experiment 에 캐시와 함께 켰다.
- **`make_prompts_simple.py --fields` — 축약이 아니라 캡션 **절 선택** (2026-08-28).**
  `caption_fields` 의 문장을 한 글자도 안 고치고 어느 절을 넣을지만 고른다
  (`--fields motion` → `"target: man motion: ..."` 에서 `target:` 절만 제거). 조립 규칙은
  `CinemaTraj/scripts/build_bank_captions.py:84 prompt_of` 와 같아
  `--fields target,motion` 이 원본 `prompts.json` 을 그대로 재현한다 (동일성 확인용, 실측 일치).
  기존 `SIMPLE_PHRASE` 축약 경로는 `--fields` 미지정 시 그대로 — 즉 이 모드는 "target 절 유무"
  **한 축만** 움직이므로, 문장 길이와 이동량 지시가 같이 움직이는 축약 ablation 과 섞이지 않는다.
  snowboard preset arm 의 대조 프롬프트(`prompts_notarget.json`)가 이걸로 만들어졌다.
- **`target_track_val_drop` — validation 을 track 없이 돌린다 (2026-08-28).**
  지금까지 val 은 `build_track_cond(...)` 를 **dropout 없이** 불러 실제 track 을 그대로 넣었다
  (`train_latent_cam_dm.py:738`). 그래서 val 곡선은 "track 이 있을 때"만 보여줬고,
  `target_track_dim=0` arm 과 눈금이 달라 곧바로 비교할 수 없었으며, dropout 이 null 조건을
  실제로 학습시켰는지도 학습 중엔 알 수 없었다. `true` 면 val 에서 `cond=None` 을 넘겨
  모델이 전 채널 0(null)을 채운다 — 학습 때 `target_track_dropout` 이 남겨 둔 그 조건이라
  미학습 입력이 아니다. 기본값 `false` = 기존 동작 비트 동일, `target_track_dim=0` 이면 no-op.
- **`conf/experiment/vista4d_pgt_k6_track_d5.yaml` — dropout 0.5 + val 은 track 없이
  (2026-08-28).** `vista4d_pgt_k6_track.yaml` 과 `target_track_dropout`(0.1→0.5) ·
  `target_track_val_drop`(→true) 두 줄만 다르다. 0.1 arm(`20260828_000434`)은 그대로 돌려
  dropout 0.1 vs 0.5 대조로 남긴다.
- **`eval_testset.py --drop-track` — track 조건 없이 추론 (2026-08-28).**
  학습 때 `target_track_dropout`(기본 0.1)이 per-sample 로 남겨 둔 **null 조건**(전 채널 0)을
  추론에서 실제로 요청하는 손잡이. 이게 없으면 dropout 이 만들어 준 "track 은 optional" 이
  학습 쪽에만 존재하고 쓸 방법이 없다. `cond=None` 으로 넘기면 모델이
  `camera_diffusion_model_latent.forward` 에서 zeros 를 채우므로 미학습 입력이 아니다.
  출력 tag 에 `__notrack` 을 붙여 track 있는 결과를 덮어쓰지 않게 하고(시드를 tag 에 박는
  것과 같은 이유), `meta.json` 에 `target_track_dim` / `drop_track` 을 남긴다.
  `target_track_dim=0` 인 arm 에서는 경고만 찍고 no-op. 플래그를 안 주면 기존 경로 그대로.
  - 스모크 (last.pth, `--max-batches 2` = 16 samples, seed 42, GPU 2) — **n=16 이라 arm
    판정용이 아니다**: track 조건 `val/loss_latent 0.07827442139387131` /
    `val/loss_traj 0.015194708947092295`, `--drop-track` `0.13371194899082184` /
    `0.026392601896077394`, 대조 `20260827_051423_vista4d_pgt_k6`(track 없는 arm)
    `0.16339020431041718` / `0.035439545288681984`.
- **`target_track_dim` — subject OBB 3D 궤적을 x_t 채널에 concat 하는 조건 (2026-08-28).**
  V4D-PGT 9 ablation. `<scene>/da3/target_track.npz` (CinemaTraj `export_target_track.py` 산출,
  scene_graph 의 dyn `track.center_smooth` / stat `obb.center` 를 `T_wg` 로 world 변환, 50 scene
  전 변이 valid) 의 subject world 궤적을 **cam_param 과 같은 앵커·같은 분모**로 옮긴다:
  `q(f) = (E_s @ [p_w(f);1])[:3] / norm_scale` (camel 실측 subject z 0.62~0.65 전 프레임 z>0,
  cam rel |t| max 0.402 와 같은 자릿수). `(49,4)[xyz,valid]` 를 VAE stride-2×2 격자에 맞춰
  latent 토큰 13개로 linear interp 후 x_t `(B,13,64)` 에 채널 concat.
  - `models/camera_diffusion_model_latent.py` — `cond_dim` 인자; `cam_in = Linear(64+cond_dim,·)`.
    **null = 전 채널 0** — concat 조건은 cross-attn 과 달리 uncond forward 에서 못 빼므로 CFG 를
    null 값으로 정의한다. `cond=None` 이면 zeros (계측 호출 보호).
  - `main/dataset_dl3dv.py` — `_target_track`/`_track_cond` (+ 캐시). 파일 없으면 loud fail
    (`_target_poses` 와 같은 이유 — 조용히 zero 조건이면 arm 이 대조군과 같아진다).
  - `main/train_latent_cam_dm.py` — `build_track_cond` (interp + per-sample dropout 0.1),
    train/val/`sample()` 배선, `is_ar`/`per_token_noise` 와 병용 시 startup assert.
  - `scripts/eval_testset.py` — run_validation 과 동일하게 dropout 없이 조건 전달.
  - `main/conf/config.yaml` `target_track_dim: 0`(기본, 기존 arm 비트 동일) ·
    `target_track_dropout: 0.1`; `main/conf/experiment/vista4d_pgt_k6_track.yaml`
    (k6 대비 `target_track_dim: 4` 한 줄 차이). smoke exit 0 (wandb dpp3oxvq),
    본학습 20260828_000434 (wandb folprs73, screen train5/GPU6 — 사용자 허용).
- **`prompts_file` — 같은 세그먼트 위에 캡션만 갈아끼우는 손잡이 (2026-08-27).**
  세그먼트 키·`frame_idx`·pose·`avg_scale`·seg list 는 그대로 두고 **텍스트만** 다른 파일에서
  읽는다. 기본값 `prompts.json` 이면 경로도 캐시 파일명도 예전과 글자 그대로 같다.
  - `main/conf/config.yaml` — `prompts_file: prompts.json`. 경로 규칙은 `prompts.json` 과 동일
    (`pose_source='da3'` 면 `<scene>/da3/<이 파일>`, `transforms` 면 `<scene>/<이 파일>`).
  - `main/dataset_dl3dv.py` — `_prompts_path` 가 이 값을 쓴다. **`_load_index` 캐시 키에
    `__pf<stem>` 추가** — 캡션은 인덱스 캐시 *안에* 저장되므로(samples 튜플 인덱스 3) 키를
    안 바꾸면 옛 캡션 캐시를 그대로 물어와 **로그·지표 어디에도 안 드러난 채** 예전 텍스트로
    학습된다. 기본값일 때는 안 붙여서 기존 캐시를 그대로 재사용한다.
  - `main/dataset_mixed.py` — mixed override 화이트리스트에 `prompts_file` 추가.
- **`scripts/data/make_prompts_simple.py` — Vista4D 축약 캡션 생성기 (2026-08-27).**
  `<scene>/da3/prompts.json` → `prompts_simple.json`. `prompt_camera_with_scene_video.concise`
  하나만 덮어쓰고 나머지 필드는 통째로 복사한다 (50 scene / 9873 seg, 키·`frame_idx`·
  `variant_id`·`preset`·`caption_fields` 0 불일치 실측).
  `"target: camel motion: the camera significantly dollies straight forward toward the subject"`
  → `"target: camel. motion: dolly in"` (median 91 → 33 chars).
  - **크기 부사를 뺀다** (사용자 지시). `barely/slightly/steadily/significantly/dramatically` 는
    `tau_max` 또는 `pan_deg` 버킷이라 실제 이동량 정보였다. 그래서 이 ablation 은 "문장 길이"와
    "이동량 지시" **두 축을 같이** 움직인다 — 결과 해석 시 반영할 것. 부사를 남기려면
    `--keep_magnitude` (`tiny/small/medium/big/huge`).
- **`scripts/data/make_prompts_simple_dl3dv.py` — DL3DV 축약 캡션 생성기 (2026-08-27).**
  `<scene>/prompts.json` + **측정된** `<scene>/da3/tags/camera_tags.json` → `prompts_simple.json`
  (6098 scene / 39837 seg, 파싱 실패 0).
  `"target: none. motion: truck left and pan right, then truck left, dolly out, and pan right"`.
  - 원래 `concise` 는 VLM 이 쓴 산문이라 카메라와 장면 묘사가 섞여 있고 **틀린다** — 실측 사례로
    concise 가 "pans left" 라고 쓴 구간을 태그는 전 구간 `yaw right` 로 잰다. 그래서 캡션을
    측정 태그에서 다시 만든다.
  - `target: none` 은 사용자 지시 — DL3DV 는 정적 씬 코퍼스라 조준할 subject 가 없고 혼합
    학습에서 **free-moving** 만 담당한다. 슬롯을 비우지 않고 글자로 적어 `target:` 이 코퍼스
    식별자로 새지 않게 한다.
  - 태그 축 단어를 Vista4D simple 과 **같은 어휘**로 옮긴다 (`move left`→`truck left`,
    `forward`→`dolly in`, `up/down`→`pedestal up/down`, `yaw`→`pan`, `pitch`→`tilt`,
    `static`→`hold still`). sub-shot 여러 개면 `, then` 으로 잇는다 (실측 1/2/3 = 30/46/24%).
  - !! 태그는 **da3 pose** 로 쟀고 학습은 `pose_source: transforms`(COLMAP) 로 돈다. 같은
    frame_idx 구간을 다른 추정기로 잰 것이라 방향은 합의하지만 크기 게이지는 다르다 —
    크기 부사를 안 넣는 이유이기도 하다.
- **`worldtraj_da3geo_simple.yaml` — DL3DV 축약 캡션 arm (2026-08-27).**
  `worldtraj_da3geo.yaml` (20260816_211821 / wandb `jkuj8mbg`, 100 epoch 완주) 대비
  `exp_name` / `prompts_file` **두 키만** 다르다 — 포즈(COLMAP)·세그먼트(`meta_worldtraj.csv`)·
  split·VAE·geo encoder 가 전부 같은 **paired A/B** 다. 코드 변경 0줄.
  - 사용자 지시: DL3DV 는 혼합 학습에서 **free-moving** 만 담당하고 `target` 은 `none`,
    카메라는 **da3 가 아니라 밖에 있는 것**(= `pose_source: transforms`, COLMAP)을 먼저 쓴다.
  - 캡션 커버리지 실측: `meta_worldtraj.csv` 6,098 scene 전부에 `prompts_simple.json` 존재
    (누락 0). 무작위 200 scene / 1,304 seg 에서 세그먼트 키·`frame_idx` 불일치 0,
    캡션 median 86 chars (max 184).
- **`vista4d_pgt_k6_simple.yaml` / `vista4d_pgt_k6_simple_unposed.yaml` — 축약 텍스트 arm 2종
  (2026-08-27).** 기존 verbose 2종과 합쳐 `{verbose, simple} x {posed, unposed}` 2x2 를 닫는다.
  `vista4d_pgt_k6.yaml` 대비 `prompts_file` (+ `geo_posed`) 만 다르다. 코드 변경 0줄.
  wandb `640y40ql` (train2/GPU4) · `c3l8oy51` (train4/GPU5).
- **`vista4d_pgt_k6_unposed.yaml` — context 카메라 없는(unposed) 대조 arm (2026-08-27).**
  `vista4d_pgt_k6.yaml` 과 **`geo_posed: true → false` 한 줄만** 다르다. 코드 변경 0줄
  (config 만 추가). 두 arm 을 동시에 돌려 "context 카메라 주입이 얼마나 기여하나"를 잰다.
  - `geo_posed=false` 면 dataset 이 `geo_c2w`/`geo_fxfycxcy`/`geo_hw` 를 batch 에 안 싣고
    (`main/dataset_dl3dv.py:1556` 의 `if self.geo_posed:` 블록이 통째로 빠진다),
    `main/train_latent_cam_dm.py:86` 이 `cam_token=None` 으로 DA3 를 부른다. `geo_cam_embed: null`
    이라 relfirst/Plücker 우회로도 없어 **카메라가 들어가는 경로가 하나도 안 남는다.**
  - **부수효과(해석 시 반영할 것):** DA3 backbone 의 `select_reference_view` /
    `reorder_by_reference` 분기는 `cam_token is None` 일 때만 발동한다
    (`models/da3_geo_encoder.py:357-361`). 그래서 posed arm 은 geo_idxs 순서
    `[0,10,19,29,38,48]` 가 보존되지만 unposed arm 은 DA3 가 `ref_view_strategy='saddle_balanced'`
    로 재정렬한다. view **집합은 동일**하고 재정렬은 이미지에만 의존하는 결정론적 함수라
    재현성은 유지되지만, 이 ablation 은 "카메라 주입"과 "뷰 순서 보존"이 함께 움직인다.
    `geo_first_view_target_s: true` 는 이 arm 에서 사실상 무의미하다.
- **`target_pose_source` — context 는 소스 영상, target 은 합성 pseudo-GT 궤적 (2026-08-27).**
  GT 카메라가 없는 코퍼스(Vista4D / TRUMANS)를 학습에 넣는 경로다. 기존에는 target 도 context 와
  **같은** pose 배열에서 슬라이스했으므로 "이 씬을 이렇게 찍었다면" 을 배울 수가 없었다.
  - `main/conf/config.yaml` — `target_pose_source: null` (기본 = 기존 동작). `'da3_target_poses'`
    면 `<scene>/da3/target_poses.npz` 의 `(V,T,4,4)` w2c + `(V,T,3,3)` K 를 target 으로 쓴다.
    어느 변이(V)인지는 세그먼트 이름 끝의 키로 고른다.
  - `main/dataset_cfg.py` — `DatasetSpec.target_pose_source` 필드 + `TARGET_POSE_SOURCES` 검증.
    `pose_source='da3'` 를 강제한다 (파일이 da3 아래에만 있다).
  - `main/dataset_dl3dv.py` — `_target_poses(scene_idx, seg_key)` + `__getitem__` 분기 3줄.
    **지역 변수 `extrinsics`/`intrinsics` 만** 갈아끼운다: geo(context) 블록은 아래에서
    `self.extrinsics_list` / `self.frame_files_list` 를 직접 읽으므로 "context = 소스 영상,
    target = 합성 카메라" 가 정확히 성립한다. 파일이나 키가 없으면 조용히 소스 궤적으로
    떨어지지 않고 **터진다** (그러면 arm 이 대조군과 같아지는데 로그엔 흔적이 안 남는다).
    index 캐시는 `(scene_idx, s, e, caption, data_name)` 만 담아 target pose 를 캐시하지
    않는다 → 이 옵션을 켜고 꺼도 staleness 가 없다.
  - 검증 (Vista4D smoke export 1편, 56 샘플): `cam_param (49,11)`, intr 열 정확히 `[1.0, 1.0]`,
    caption `target: man motion: the camera slightly dollies straight forward toward the subject`,
    소스 path 0.1356 vs 변이0 0.8318 (다른 궤적), frame0 위치 거리 **3.26e-16** (start_mode 유지),
    `target_pose_source` on/off `cam_param` max diff **0.1906** (분기가 실제로 발화),
    context idxs `[0, 10, 19, 29, 38, 48]` (소스 영상 uniform 6장).
- **`main/conf/experiment/vista4d_pgt_k6.yaml` — Vista4D pseudo-GT arm (2026-08-27).**
  DL3DV 로더를 그대로 쓰고 `dl3dv_root` 만 Vista4D 변환본으로 돌린다. 축이 바뀐다:
  scene = 영상 1편(49프레임), **segment = 뱅크 변이 1개** (V≈56~392, 프레임 구간이 아니다).
  `target_pose_source: da3_target_poses` + `geo_view_sampling: context_uniform` /
  `geo_num_views: 6` + `avg_scale_ref: context_first_cam` (디렉토리 이름만 빌려 쓰고 정의는
  `scene_graph.json:scale.S` = frame0 non-sky 평균 ray 길이. 소스 전용이라 누수 없음).
  뱅크가 `--fixed_focal` 이고 export 도 frame0 K 를 49프레임에 복사하므로 `intr_norm: rel`
  에서 `cam_param[9:11]` 이 정확히 `[1,1]`. 분할은 **scene(영상) 단위 holdout** —
  같은 영상의 다른 변이가 train/val 로 갈리면 val 이 같은 소스 프레임을 봐서 낙관적으로 뜬다.
- **`scripts/data/corpus_clearance_probe.py` — 코퍼스 카메라가 표면에 얼마나 가까이 갔나
  (2026-08-27).** `corpus_scale_probe.py` 는 이동량 `‖t‖/norm_scale` 만 재는데, "이 씬에서
  이만큼밖에 못 움직인다"를 모델이 배우려면 **여유 거리(clearance)** 가 데이터에 있어야 한다.
  세그먼트 프레임의 da3 depth 를 unproject 해 점군 `X` 를 만들고
  `clearance(f) = min_{x∈X} ‖center(f) − x‖` 의 `min_f` / `med_f` 를 **같은 `norm_scale` 로
  나눠** 보고한다 → `corpus_scale_probe.py` 의 `med` 와 바로 같은 축에서 비교된다.
  `norm_scale` 은 `__getitem__` 이 뱉은 값을 그대로 쓴다 (scale_mode 분기 8종을 손으로 다시
  짜지 않는다). 깊이 불연속 픽셀(`|∇log z| > --edge_thr`, 기본 0.05)은 뺀다 — 안 빼면 물체
  경계에 붕 뜬 점이 clearance 를 실제보다 작게 만든다.
  DL3DV 실측 (`da3_7k_da3geo_frontanchor`, n=300, stride 4, GPU 1장 약 2분,
  `results/mixed/clearance_probe/`):
  `clearance_min` p01 0.0093 / p05 0.0322 / p25 0.1154 / med **0.1903** / p75 0.3052 /
  p95 0.4512 (sd(log10) 0.394), `clearance_med` med 0.2800, `depth_min` med 0.2106.
  같은 분모의 이동량 median 이 0.4367 이므로 **이동량 / 최소 여유 거리 = 2.3** — DL3DV
  카메라는 최근접 거리의 두 배 넘게 움직인다(= 표면이 궤적을 실제로 제약한다).
- **`anchor_pred_frame0` — target 첫 카메라를 given 으로 (2026-08-23).** dataset 이 `cam_param` 을
  `rel_t = w2c_t @ inv(w2c_0)` 로 만들어 GT `rel[0]` 은 정확히 항등이고 추론 시에도 target 첫
  카메라는 주어지는데, `out_to_trajectory` 는 `e0` 를 곱하기만 하고 `rel[0]=I` 를 강제하지
  않았다 → 모델이 낸 frame0 오차가 궤적 전체를 통째로 밀었다 (vls019 epoch 82 실측 median
  frame0 |Δt| **0.1105** = GT path_len 의 0.20배, rot 1.79°; 같은 잣대로 frozen VAE 왕복은
  0.0019 / 0.129° 라 diffusion 이 낸 오차다).
  - `utils/data_utils.py` — `out_to_trajectory(..., anchor_frame0=False)` 인자 추가.
    `True` 면 `rel'_t = rel_t @ inv(rel_0)` 로 재앵커한다. 프레임 간 상대 운동
    (`rel_t @ inv(rel_t')`) 은 보존되고 GT 는 이미 `rel[0]=I` 라 no-op — pred 에만 효과가 있다.
  - `main/conf/config.yaml` / `main/config.py` — `anchor_pred_frame0: false` (기본 = 기존 동작).
  - decode 경로 4곳에 배선: `main/train_latent_cam_dm.py`, `main/valid_cam_acc.py`,
    `main/infer_cam_dm.py`, `main/infer_textonly_batch.py`. loss(`loss_latent`/`loss_traj`) 는
    안 건드린다 — 학습 신호는 그대로 두고 decode 만 앵커한다.
  - 검증: smoke 의 test 덤프 14 타깃에서 `_transforms_pred.json[0]` vs `_transforms_ref.json[0]`
    max |Δ| **9.54e-07** (median 2.98e-07). 직전 arm 의 같은 값은 median 0.1105 였다.
- **TRUMANS-Lite text-only arm + caption 상한 진단 (2026-08-23).** `vae_latent_scale` 를 고쳐도
  caption 이 안 올라가는 원인을 좁힌다.
  - `main/conf/experiment/trumans_lite_textonly.yaml` — `trumans_lite_ctxuniform_vls019` 대비
    `geo_encoder: da3 → null` (text-only) + `vae_latent_scale: 0.19128 → 0.96032625`
    (사용자 지시, 베이스 arm `am3tzbk7` 과 동일) + `anchor_pred_frame0: true`.
    screen train1 / GPU1 / wandb `1eptj1o6`. 올라가면 da3 geo 토큰이 텍스트 신호를 덮는 것,
    안 올라가면 130 클립·val 1 recording 코퍼스 쪽 문제.
    (첫 기동 `hv21id7s` 는 `vae_latent_scale` 0.19128 / anchor 없음 — 두 설정 모두 바뀌어 폐기.)
  - `scripts/vae/vae_roundtrip_val.py` (신규) — diffusion 을 건너뛰고 GT `cam_param` 을
    encode→decode 만 시켜 학습 val 루프와 **같은** 후처리를 태우는 "완벽한 diffusion" 상한선.
    frame0 오차 / tortuosity / `5·Δt_local` / static 축비율을 GT 와 나란히 찍는다.
    → **frozen VAE 상한 가설은 기각**: median frame0 |Δt| 0.0019, frame0 rot 0.129°,
    path_len 비 1.0220, tortuosity 1.1089→1.1340, static 축비율 0.4722→0.4583.
    같은 잣대로 vls019 epoch 82 예측은 frame0 |Δt| 0.1105 / rot 1.79° / tortuosity 1.4~9.5.
  - `scripts/render/topdown_gt_vs_pred.py` (신규) — GT vs 여러 run 의 카메라를 GT frame0 기준
    재앵커해서 top-down / side / per-frame |Δcenter| / `5·Δt_local` 3축 + caption segment
    문자열까지 한 패널에 찍는다. `--runs name=dir` 다중 지원.
  - vls019 / vls039 는 epoch 82 부근에서 Ctrl+C 로 정지 (결과·wandb 는 대조군으로 보존).
- **TRUMANS-Lite `vae_latent_scale` 재캘리브레이션 arm 2종 (2026-08-23).** 아래 진단의 원인 ②
  를 고친 재시작. 두 config 모두 `trumans_lite_ctxuniform.yaml` 전량 복사본이고 주석 제외
  실질 차이는 `exp_name` + `vae_latent_scale` **두 줄뿐**이다.
  - `main/conf/experiment/trumans_lite_ctxuniform_vls019.yaml` — `vae_latent_scale: 0.19128`
    = 전체 코퍼스(130 샘플) 실측 latent std → diffusion 입력 std **1.0** (절대 품질 최선).
    screen train1 / GPU1 / wandb `aw35qouu`.
  - `main/conf/experiment/trumans_lite_ctxuniform_vls039.yaml` — `vae_latent_scale: 0.38565`
    = 0.19128/0.4960 → 입력 std **0.4960** = DL3DV arm 들과 같은 자리 (paired 비교용).
    screen train3 / GPU3 / wandb `kq1nxqkp`.
  - 베이스 arm `am3tzbk7` 은 epoch 182 에서 Ctrl+C 로 정지 (결과·wandb 는 대조군으로 보존).
    `mix_dl3dv_trumans_v1`(train2/GPU2/`vk0kozz1`) 은 그대로 진행 중.
  - !! 세 arm 의 `loss_latent` 는 서로 직접 비교 금지 — latent 단위가 다르다. 비교는 decode 후
    지표(`loss_traj`/`clatr`/`fcd`/`caption`). 원인 ①(static 임계)은 이 두 arm 이 못 고친다.
  - smoke(GPU3, `max_scenes=12 epochs=1`) 둘 다 exit 0, epoch0 `val/loss_traj`
    0.091067(vls019) / 0.410090(vls039) — 베이스 arm 의 같은 smoke 2.830574 대비 상수 제곱비와
    대체로 일치해 값이 실제로 먹고 있음을 확인. 결과 디렉토리는 삭제.
  - 자세한 실측·설정 델타는 `EXPERIMENTS.log` 2026-08-23 21:26 항목.
- **arm A(`trumans_lite_ctxuniform`) caption precision/recall/f1 정체 원인 진단 (2026-08-23).
  코드 변경 없음** — 측정만. 긴 요약은
  `camera_generation/models/Planner/CinemaTraj/summary.md` §7 (gitignore 대상).
  - **input/target/좌표계 버그는 없다.** 덤프된 `_transforms_ref.json` 이 디스크
    `da3/pose.npz` 를 w2c→c2w→OpenGL flip 한 것과 자릿수까지 일치
    (`|dt| mean [0.00654 0.00462 0.00803]`) — `out_to_trajectory` 가 스케일을 정확히 복원해
    raw TRUMANS 미터로 나간다. GT caption segment 도 생성 preset 과 의미 일치
    (arc_right x 부호 일치 1.00 / arc_pull z 1.00 / push_in z 0.87 / hold 정지).
  - **원인 ① caption metric 의 static 임계가 절대 world 단위다.**
    `main/evaluate/eval/src/metrics/modules/caption.py:405-410` `fps=5`,
    `t_velocities = 5·Δt_local`, `cam_static_threshold = 0.02`
    → 축당 프레임당 0.004 world unit 을 넘어야 "움직임".
    ref median `5·|Δt|` (x,y,z) / 임계 미만 축 비율:
    DL3DV `[0.4291, 0.0805, 0.2095]` **3.3%** | mix arm B `[0.3148, 0.0647, 0.1398]` **2.7%** |
    **TRUMANS `[0.0237, 0.0109, 0.0372]` 45.2%**. 클래스는 27×7=189 완전일치 +
    `average="weighted"` 라 부분 점수가 없다 → static 비트 하나 뒤집히면 0.
    (메모리 `caption-metric-thresholds-are-world-units` 의 DataDoP fscore 0 과 같은 현상.)
  - **원인 ② `vae_latent_scale` 이 TRUMANS 에 대해 5× 틀렸다.**
    `scripts/vae/vae_scale_matrix.py` 를 TRUMANS 전량 130 sample 에 실측:
    `vae_20260302_300 / 64 / avg_scale / rel -> lat.std 0.19128, recon L1 rot 0.00286
    trans 0.00174 intr 0.00169`. config 는 0.96032625 → **diffusion 입력 std 0.1992**
    (DL3DV-only 는 0.47637/0.96032625 = 0.4960 이라 추가로 2.49× 더 작다).
    VAE 자체는 in-distribution (trans recon 이 DL3DV 0.00336 보다 좋다) — 상수만 틀렸다.
    결과(epoch 95, step 1330, val 14편): path_len ref 0.5723 / pred 2.7048,
    net_disp 0.5213 / 1.0987, tortuosity **1.08 / 2.54**, 프레임당 속도비 축별
    `[11.99, 31.37, 12.75]` — net 은 2.6× 인데 프레임당은 12–31× = 고주파 떨림.
  - 모델은 학습 중이다: `val/loss_traj` 2.830574(ep0)→0.116155(ep52),
    `val/clatr/clatr_score` 0→13.0794(step 1302), `val/clatr/fcd` 1390.85→1163.94(step 1246).
    `val/captions/fscore` 만 0~0.0911 사이를 추세 없이 튄다.
  - 후속 선택지(미판정): (A) `vae_latent_scale=0.19128` 입력 std 1.0 /
    (B) `0.38565` 입력 std 0.4960 = DL3DV arm 과 paired / (C) `cam_static_threshold` 를
    코퍼스별로(TRUMANS≈0.0011). A·B 는 원인②만, C 는 원인①만 고친다.
- **TRUMANS-Lite 를 DL3DV-da3 코퍼스로 태우는 실험 config 2종** — 코드 변경 없이
  `dl3dv_root` + `meta_csv` 만 갈아끼워 기존 `dataset_dl3dv` 로더를 그대로 쓴다
  (변환기는 `camera_generation/models/Planner/CinemaTraj/scripts/trumans_lite_to_dl3dv.py`,
  gitignore 대상이라 이 리포에는 안 들어온다).
  - `main/conf/experiment/trumans_lite_ctxuniform.yaml` — TRUMANS 단독 arm.
    scene = recording (클립 14~22개를 프레임 축으로 이어 붙임), segment = clip.
    클립 하나를 scene 으로 두면 context range == target range 라 `geo_posed=true` 인 DA3
    cam_enc 이 target 궤적 자체를 받는다 — 그걸 피하려고 이어 붙였다.
    `geo_view_sampling=context_uniform` (6뷰, `geo_first_view_target_s=true`),
    `avg_scale_ref=context_first_cam` (디렉토리 이름만 빌린 것, 값은 **recording median**).
    scene 단위 holdout: recording `2b4c9b84` 통째로 val -> train 116 / val 14.
    `epochs: 600` (epoch 당 14 배치뿐).
    누수 실측(130 세그먼트 전수): 6뷰가 **항상 서로 다른 클립 6개**에 떨어지고(mean/min/max
    6.00), target 클립 안의 view 는 frame `s` 하나뿐이다(= `first_extrinsic`, 추론 시 주어짐).
    s 가 아닌 in-clip view 0개 -> 의도치 않은 누수 없음.
  - `main/conf/experiment/mix_dl3dv_trumans_v1.yaml` — DL3DV(7k) + TRUMANS 2코퍼스 혼합.
    `dataset_name: mixed` 에 **`dl3dv` 블록 2개** (`dataset_mixed` 는 이름 유일성을 요구하지
    않는다). DL3DV 블록은 `da3_7k_da3geo_frontanchor` 와 동일 -> 그 arm 이 대조군.
    TRUMANS `weight: 16.0` — 표본 수가 29414:116 (254:1) 이라 weight 1.0 이면 전체 배치의
    0.4% 로 no-op 이 된다. T=2 temperature (p_i ∝ sqrt(n_i)) 목표 지분 5.9% 를 맞춘 값이고,
    대가는 TRUMANS 클립 하나가 epoch 당 16번 학습에 들어가는 것이다.
    `scale_gain` 은 둘 다 1.0 (기존 혼합 arm 과 같은 이유; da3+posed 에서 raise 된다).
  - 실측 레벨 (`mean||t||/분모`, 130 세그먼트): TRUMANS geomean **0.0945** / med 0.0969 /
    p05 0.0390 / p95 0.2000 (log10 std 0.2377). DL3DV 0.4367 의 1/4.6 이고 SD 0.0878 /
    DataDoP 0.0911 과 같은 자리. `vae_latent_scale` 은 0.96032625 그대로 (재측정 안 함).
  - TRUMANS intrinsics 는 프레임 전체 상수 (fx=fy=666.67, cx=480, cy=270, std 0.0)
    -> `intr_norm: rel` 에서 `cam_param[9:11] = [1,1]`, DL3DV 와 같은 자리.
- **`geo_view_sampling: 'context_uniform'`** (`main/dataset_dl3dv.py`, `main/config.py`,
  `main/dataset_cfg.py`) — geo context view 를 **영상 전체 `[0, N)`** 에서 uniform 으로 뽑는
  sampler. 기존 5종(`even` / `random_inseg` / `hybrid` / `frustum_cover` / `front_uniform`) 은
  전부 그대로 — `dataset_dl3dv.py` diff 는 **삭제 0줄**(순수 추가 41줄)이고 dispatch 도 `elif`
  하나만 붙였다. 이유: context 가 **다른 clip** 인 코퍼스(SD / TRUMANS)에서는 "target 세그먼트
  앞/안"이라는 range 자체가 정의되지 않고 clip 하나가 통째로 context 다 —
  `dataset_scene_decoupled._ctx_view_idxs` 가 이미 같은 일을 하고 있어서 그 규약을 DL3DV 쪽
  sampler 로 옮긴 것. **leakage-free 가 아니다**: `[0,N)` 이라 target 프레임이 뽑힐 수 있다
  (leakage 없는 arm 은 `front_uniform` 뿐).
  `geo_first_view_target_s` 를 지킨다 — on 이면 `[0,N)` 에서 k 장을 뽑은 뒤 **s 에 가장 가까운
  픽을 s 로 치환**해 view0 로 올린다(토큰 예산 `V*P` 는 다른 arm 과 동일, `frustum_cover` 의
  `k_retr = geo_cover_k - 1` 과 같은 "s 는 k 안에" 규칙). `s` 를 그냥 prepend 하고 나머지 k-1 을
  `[0,N)\{s}` 에서 뽑으면 **s=0 일 때 첫 픽이 프레임 1** 이라 view0 과 1프레임 차이인 중복 view
  가 생긴다 — DL3DV da3 는 `s ∈ {0,49,98,...}` 이라 매 scene 첫 세그먼트가 여기 걸린다
  (`_sample_geo_front_uniform` 이 포함 구간에서 한 번에 뽑아 피하는 것과 같은 함정). 치환 방식의
  실측 최소 간격: N=300 에서 38~59, N=49 에서 9.
  `dataset_cfg.py` 가드 추가: `context_uniform` + `avg_scale_ref` 가 `front_*` 이면 경고
  (context view range `[0,N)` 와 분모를 만든 range `[s-L,s)` 가 다르다 — `front_uniform` 가드와
  같은 종류). `avg_scale_ref=front_first_anchor_same_len` 으로 실제 발화 확인.
- **`main/conf/experiment/da3geo_ctxuniform_smoke.yaml`** — 위 sampler 의 스모크 설정.
  `da3geo_smoke.yaml` 에서 `geo_view_sampling` 과 `geo_num_views: 6` 만 바꾼 것, 분모는 range 가
  어긋나지 않는 `centroid`. 스모크 실측(GPU 1, `max_scenes=60 epochs=1 batch_size=2`,
  `WANDB_MODE=disabled`): peak **30177 MiB**, 4.48 it/s, best `val/loss_traj` **3.266933**.
  같은 조건 `frustum_cover` k=6 대조군: peak **30215 MiB**, 4.42 it/s, best `val/loss_traj`
  **3.258861**. 메모리는 sampler 와 무관하다 — `V`·`batch_size` 가 같고 sampler 는 **어느 프레임을
  읽을지**만 바꾼다.
- **`scripts/viewer/viser_revpair.py`** — SD 역재생 결합 recon
  (`video_generation/results/20260818_sd_revpair/<pair>/recon.npz`, 161장 통짜 DA3) 을 viser 로 본다.
  전역 index 를 `anchor_index`(=80) 에서 잘라 `[0,anchor)` = `rev(A)` = **context**(초록),
  `[anchor,N)` = `B` = **target**(파랑 point / 청록 camera), anchor 한 장은 흰색 frustum 으로 따로
  띄운다. frustum·색 규약은 `viser_val_cameras.py` 의 `add_frustums` / `_GL2CV` 를 **import 해서
  그대로 쓴다** (기존 뷰어의 `main()` 은 `results/<exp>/test` glob + PRDC 에 묶여 있어 분기를 못
  친다 — 형제 스크립트로 뺐다). `extrinsics` 는 OpenCV **w2c** 라 `c2w_gl = inv(w2c) @ _GL2CV`
  (메모 `da3-extrinsics-are-w2c`). world 는 DA3 joint 게이지 그대로 — 여기서 sim3 를 또 태우면
  "두 clip 이 통짜 recon 하나로 같은 좌표계에 들어왔나"라는 확인 대상 자체가 가려진다.
  기본 tint, `--rgb` 로 `images` 원색. 기존 파일 변경 0줄.
  주의: DA3 `conf` 는 **1.0 에 바닥**이 있고 Cabin_Lake recon 은 픽셀의 44.0% 가 정확히 1.0 이라
  기본 `--pc-conf-pct 40` (= `viser_val_cameras.py` 와 같은 값) 이 임계 1.0000 → **아무것도 안
  버린다**. 조용히 "p40 로 걸렀다"고 오해하지 않도록 임계 == conf 최소값이면 경고를 찍는다.
- **`scale_mode: 'const'` + `norm_scale_const`** (`main/dataset_dl3dv.py`, `main/conf/config.yaml`,
  `main/config.py`) — 세그먼트마다 재는 적응형 분모 대신 **코퍼스 상수 하나**로만 나눈다.
  기존 분모 6종은 전부 세그먼트별로 scene 크기에 맞춰 적응하는데(= scale align), 이 mode 는 그
  적응만 없애고 나머지(앵커 재고정 `E @ inv(E_s)`, `intr_norm`, 채널 규약)는 그대로 둔다.
  `norm_scale_const` 가 없거나 <= 0 이면 raise. `intr_norm: 'auto'` 매핑에서 `'const'` 는
  `'avg_scale'` 과 같이 `'rel'` 로 간다 (분모만 다른 arm 끼리 intr 규약이 갈리면 비교가 안 된다);
  다른 mode 의 auto 매핑은 불변. **기존 config 는 전부 동작 불변** (새 분기라 기존 dispatch 에
  손대지 않았다).
- **`main/conf/experiment/da3_7k_da3geo_frontanchor_noscale.yaml`** — 위 mode 를 쓰는
  scale-align ablation arm. 대조군 `da3_7k_da3geo_frontanchor` 와 **분모 하나만** 다르다.
  상수 3.982685 = 그 arm 의 train 29414 세그먼트 `avg_scale_front_first_anchor` **기하평균**
  (med 3.614095, log10std 0.3287; test 3375 세그먼트는 3.971595 로 0.6% 차이). 기하평균을 쓰면
  `log(mean||t||/분모)` 평균 레벨이 대조군과 0.0038 dex (0.9%) 안에서 일치해
  (`const` med 0.3845 / log10mean −0.4301 vs `avg_scale` med 0.4228 / −0.4339) camera VAE 와
  `vae_latent_scale: 0.96032625` 가 같은 자리에 남는다 — `norm_scale=1` 로 두면 translation 이
  3.6배 커져 VAE 를 OOD 로 밀어 넣는 효과와 섞인다. 인덱스는 대조군과 동일(29414/3263):
  `avg_scale_ref: front_first_anchor` 는 분모에는 안 쓰이지만 인덱스 필터(`s==0` 제외)에는 계속
  관여하므로 paired 비교용으로 남겼다. 추론/eval 경로는 무수정 — `scripts/eval_testset.py` 가
  dataset 이 내보낸 `avg_scale` 을 그대로 `out_to_trajectory(scale=...)` 에 곱해 되돌린다.
  검증(paired, 30 샘플): `cam_param[:, 6:9]` 크기비 = `ns_ctrl / 3.982685` 와 1e-6 이내 일치,
  rotation(0:6)·intrinsics(9:11) 채널은 **bit-identical**, 샘플 집합 동일.

### Fixed
- **CLaTr PRDC 가 val 이 작은 arm 에서 매 eval 마다 죽던 것**
  (`main/evaluate/eval/src/metrics/modules/prdc.py`, `ManifoldMetrics.compute`).
  `chunk(num_splits=5)` 로 자른 조각마다 `topk(k=manifold_k+1=4)` 를 부르는데 조각이 4개보다
  작으면 `RuntimeError: selected index k out of range` 로 죽고 **metrics.json 전체가 안 나온다**
  (PRDC 뿐 아니라 FCD/caption 지표까지 같이 날아간다). 즉 `N >= 5*(manifold_k+1) = 20` 이
  암묵 가정이었다 — TRUMANS-Lite 단독 arm 은 val 이 14 세그먼트뿐이라 600 epoch 내내
  CLaTr 지표가 하나도 안 남을 참이었다.
  - `num_splits` 를 `max(1, min(num_splits, N // (manifold_k+1)))` 로 낮춘다.
    **N >= 20 이면 그대로 5** 라 기존 arm 은 완전히 동일하게 돈다 (삭제 0줄, 분기만 추가).
    N=14 -> 3분할(5,5,4), N=160/3263 -> 5분할(변화 없음).
  - `torch.chunk` 의 짧은 꼬리 조각(N=17, splits=4 -> 5,5,5,**2**)은 건너뛴다.
  - `N < manifold_k+1` 이면 topk 에러 대신 이유를 적은 `ValueError`.
  - 분할 수가 바뀌면 `[prdc] N=.. 이라 num_splits 5 -> 3` 를 print — 조용히 다른 분할로
    잰 게 아님을 로그에 남긴다. !! 분할 수가 다르면 PRDC 값을 arm 간 직접 비교하면 안 된다.

### Changed
- **`vista4d_pgt_k6.yaml` 에 `anchor_pred_frame0: true` (2026-08-27, experiment config 만,
  코드 변경 0줄).** 이 arm 도 frame0 은 given 이다 — dataset 이 `rel_t = w2c_t @ inv(w2c_0)` 로
  cam_param 을 만들므로 **target 이 소스와 다른 배열이어도** GT `rel[0]` 은 정확히 항등이다.
  이 코퍼스에서 확인: smoke 의 ref 덤프 160 타깃(테스트 4씬 전부)에서 `transform[0]` 이
  서로 **비트 단위로 같다**(max 편차 0.0). 씬마다 다른 `e0` 였다면 씬 경계에서 갈렸을 테니
  상수인 건 `rel[0]=I` 라는 뜻이다.
  같은 smoke(GPU3, max_scenes=12 epochs=1) 를 anchor off/on 으로 두 번 돌린 실측:
  - `val/loss_traj` **1.101709 로 동일** — loss 는 안 건드리고 디코드에서만 재앵커하므로
    학습 동역학은 그대로다. 바뀌는 건 caption/CLaTr 같은 디코드 후 지표뿐이다.
  - pred frame0 `|dt|` median **4.7002 -> 0.0000**, rot median **27.10deg -> 0.02deg**.
    (앞 숫자는 1-epoch 미학습 모델 값이라 TRUMANS vls019 epoch 82 의 0.1105 와 비교 금지 —
    여기서 말하는 건 "anchor 가 실제로 걸렸다" 뿐이다.)
- **`frontanchor` arm 2개에 `geo_cover_before_only: true`** (`da3_7k_da3geo_frontanchor.yaml`,
  `da3_7k_lagernvsgeo_frontanchor.yaml`; experiment config 만, 코드 변경 0줄). 그 전까지 이 arm 은
  **분모만** 앞쪽(`[0,s)`)으로 제한하고 geo context view 는 양쪽 중 긴 쪽에서 뽑았다
  (`dataset_dl3dv.py:775-778`). 실측 결과 학습 샘플의 **43.45%** (train 12781/29414,
  val 1415/3263) 에서 context 가 target 뒤쪽 `[e,N)` 에 잡혀 **분모를 만든 공간을 모델이 아예 못
  보는** 상태였다 — 그 샘플의 분모는 context 로부터 복원 불가능한 잡음이다. 게다가 실제 배치는
  source video 가 연속으로 앞에 오는 causal 상황이라 뒤쪽 context 는 test 분포에 없다.
  뒤쪽을 섞어 얻는 이득의 증거도 없었다: 공통 testset paired 비교에서 frustum_cover(43% 뒤쪽) −
  front_uniform(100% 앞쪽) `clatr_score` = −0.0894 (d_z −0.006, 세그먼트 win-rate 0.499).
  → 분모와 context 를 둘 다 `[0,s)` 로 맞춘다. **인덱스는 안 바뀐다** (before_only 필터는
  `s < geo_cover_k(=6)` 만 거르는데 `s ∈ {0,49,…}` 이고 `front_first_anchor` 가 이미 `s==0` 을
  뺐다) — 32677 samples / 6090 scenes, split 29414/3263 그대로라 이전 arm 들과 계속 paired 다.
  캐시 키에 `bo{0,1}` 이 들어가므로 (`dataset_dl3dv.py:403`) 새 인덱스 파일이 빌드되고 기존
  `bo0` 캐시는 그대로 남는다.
- **`geo_encoder='da3' + geo_posed=true` 가 `pose_source='transforms'` 도 받는다**
  (`main/dataset_cfg.py`, 새 플래그 `da3_geo_allow_transforms_pose`, 기본 `false`).
  그 전까지는 무조건 raise 였다. 그 가드는 custom backend 의 근거(`ray x depth` 가 같은 스케일
  공간이어야 한다)를 그대로 옮겨 온 것이라 da3 backend 에는 과하다: da3 는 (a) 디스크 depth 를
  전혀 안 읽고 (`geo_logd`/`geo_valid` 는 `geo_custom` 분기 전용, `dataset_dl3dv.py:1417-1423`),
  (b) `cam_enc` 입력을 view0 재고정 + **per-sample median camera distance** 로 정규화하므로
  (`da3_geo_encoder.build_cam_token`) 포즈 코퍼스의 절대 스케일에 불변이다. 기본값을 `false` 로
  둔 이유는 da3 코퍼스 arm 에서 `pose_source` 를 빠뜨리는 오타를 잡아 주는 값이 크기 때문이고,
  의도적으로 섞는 arm 만 opt-in 한다 → 기존 arm 전부 동작 불변 (검증: 플래그 off 에서 여전히
  raise, `da3_7k_da3geo_frontanchor` / `da3_7k_lagernvsgeo_frontanchor` / `geo_worldtraj` 의
  resolved spec 무변화).

- **DataDoP divisor 주석의 레벨 근거를 정정** (`main/conf/config.yaml`, `main/dataset_datadop.py`
  docstring; 코드 동작 변화 0). 그전까지 "D=1 이면 p95 0.4493 으로 DL3DV 의 p95 0.456 과 1.5%
  차이로 붙는다" 고 적혀 있었는데, 그 DL3DV 앵커(med 0.141 / p95 0.456)가 **재현되지 않았다**.
  아래 `corpus_scale_probe.py` 로 데이터셋에서 직접 재면 DL3DV(frontanchor) 는 med 0.4367 /
  p95 0.8934 이고, DataDoP D=1 은 med 0.0911 로 **4.8배 낮다**. `datadop_divisor: none` 이라는
  선택 자체는 유지된다 — 근거가 "레벨 정합" 이 아니라 MonST3R 게이지 논거 + 더 작은 산포
  (sd(log10 m) 0.488 < meanray 0.508) 로 바뀌었을 뿐이다. `datadop_norm_gain` 은 혼합 경로에서
  쓰지 않는다고 명시 (아래 `norm_scale_gain` 으로 통일).

- **`norm_scale_gain` 을 `geo_encoder='da3' + geo_posed=true` 에서 raise 로 막았다**
  (`main/dataset_mixed.py` `_make_sub_cfg`; 기본 1.0 이라 기존 동작 변화 0). 그 경로의 context 는
  `norm_scale` 을 **안 거친다**: `__getitem__` 이 raw `geo_c2w` 를 월드 단위로 내보내고
  (`dataset_dl3dv.py:1437-1444`) `da3_geo_encoder.build_cam_token` 이 DA3 자기 규약(context view
  들의 **median camera distance**)으로 재정규화한다 (`:279-282`). 그래서 gain 을 걸면 target
  `cam_param` 만 gain 배가 되고 context cam token 은 **불변**이라, context 가 함의하는 스케일과
  target 크기의 대응이 코퍼스별 상수만큼 어긋난다 — 모델은 그 상수를 context 에서 읽을 수 없고
  추론에는 코퍼스 라벨이 없다. gain 이 균일 닮음변환으로 남는 것은 모든 채널이 같은 `norm_scale`
  로 나눠지는 `geo_custom`(lagernvs) 경로뿐이다. → **혼합 arm 은 gain 없이(전부 1.0) 간다.**

- **`scripts/data/sd_build_seg_lists.py` 에 `--resid-pct` / `--tag` 추가 + static 집계 정정**
  (기본값 `--resid-pct 99 --tag ""` = 기존 리스트와 bit-identical). 그전엔 sim3 잔차 컷이 p99 로
  하드코딩돼 있었고 출력 파일명도 고정이라 실험용 리스트를 만들 수 없었다. 아울러 static 판정을
  `moving` → **`was_static`** 으로 바꿨다: 2026-08-11 `sd_static_scale_transfer.py` 가 static
  7682개에 scene 중앙값 `s`/`t`/`matrix4` 를 채워 넣으면서 **`moving` 이 전부 true 로 뒤집혀서**
  요약이 "static 제외 0" 이라고 잘못 찍혔다 (이 clip 들은 `resid_rmse_over_rad` 가 없어 align 컷
  에서 자동 탈락하므로 **리스트 내용 자체는 그때도 옳았다** — 집계만 틀렸다). CSV 에 `was_static`
  열 추가.
- **혼합 arm 2개의 SD seg list 를 resid p90 컷(`_r90`)으로 교체**
  (`mix_dl3dv_sd_datadop_v1.yaml`, `_smoke.yaml`; 코드 변경 0줄). 위 `sd_anchor_error_filters.py`
  전수 측정에서 앵커 불일치의 지렛대가 sim3 잔차로 나왔으므로 `--resid-pct 90 --tag _r90` 으로
  리스트를 다시 뽑았다. clip 23408 → static 7682 / align 컷 1573 제외 → 남은 clip **14153**
  (p99 리스트는 15568), clip ≥2 인 scene 3250/3344, train scene **2925 → pair 44424** /
  test **325 → 4904** (p99 는 3006/52286, 334/5758). 측정 pair 기준 효과: 49328/58044 (85.0%)
  유지, ratio p95 0.3169→**0.2026**, p99 0.6204→**0.3676**, max 3.8141→**1.1428**,
  `>0.5` 1.74%→**0.31%**, 앵커 절대오차 p99 0.0419→**0.0250**.
  **비용**: 살아남는 clip 집합이 바뀌어 scene 단위 분할도 달라진다 ⇒ `sd_whuman_customgeo_v6` 와의
  SD testset paired 비교는 깨진다. 되돌리려면 YAML 의 두 경로에서 `_r90` 만 빼면 된다.
  재빌드 검증: SD index 49328 pair → 14092 target (ctx 후보 ≥2 인 sample 13808, 평균 3.50개),
  `[mixed] total 67152 samples | train 60447 / val 6705`, key set OK (14 keys).
  참고 — 앵커를 ctx view0 로 **옮기는** 대안은 불가능하다: DA3 `build_cam_token` 입력이 global
  rigid + uniform scale 에 불변이라 (측정 max diff 4.77e-07 / 6.56e-07) `geo_c2w` 로는 이 offset 을
  넣을 수 없다.

### Added
- **`scripts/data/check_mixed_multigpu_shard.py` (신규)** — multi-GPU 에서 `PerCorpusBatchSampler`
  의 "배치 하나 = 코퍼스 하나" 가 유지되는지 검사한다 (plan Verification #7). `num_processes > 1`
  이면 accelerate 가 우리 batch_sampler 를 `BatchSamplerShard` 로 감싸는데, 그 래퍼의 배치 분할/
  tail padding 이 다른 코퍼스 인덱스를 한 배치에 섞으면 `collate_fn` 의
  `torch.stack(images (V,3,H,W))` 가 코퍼스별 `V` 차이(DL3DV 6 / SD 6 / DataDoP 1) 때문에 터진다.
  GPU 도 데이터도 안 쓰고 인덱스→코퍼스 라벨만 조회하므로 2-GPU 로 올리기 전에 싸게 잡는다.
  실측(`--sizes 29414 12689 18344` = `mix_dl3dv_sd_datadop_v1` 의 실제 train 수, bs 8):
  train/val 모두 nproc 1/2/4 에서 **혼합 배치 0 → PASS**. 참고로 nproc=2 는
  `len(base sampler)=3777` 인데 rank 별 실제 순회는 1888 이라 `itr_per_epoch` 이 multi-GPU 에서
  2배로 잡히지만, `itr_per_epoch` 은 `main/*.py`·`scripts/*.py` 어디에도 **소비처가 없고**
  단일 코퍼스 경로도 같은 규약이라 무해하다.
- **`scripts/data/viz_sd_anchor_error.py` (신규)** — SD 앵커 잔차(`A = w2c_target[s] @
  inv(w2c_ctx[view0])`)를 sample 단위로 재고 worst/p99/p95/median/best 를 그림으로 낸다.
  케이스마다 target clip frame s / context clip view0 / 두 장의 `|diff|` / 정렬된 world 에서의
  두 궤적 + 앵커 잔차 선분 + 수치 패널. `--reuse` 로 재측정 없이 그림만 다시 그린다.
  실측(n=200, `mix_dl3dv_sd_datadop_v1`): ratio(=앵커 이동 오차 / target motion `m`)
  med 0.0661 / p95 0.2769 / max 1.1667, `>0.5` 인 샘플 **2.5%**(5/200); 회전 med 0.47° /
  max 7.99°; **절대** 앵커 오차는 med 0.0054 / p99 0.0368 (모델 단위, DL3DV 의 `m` med 0.4367
  대비 작다). 꼬리의 원인은 분모·분자 양쪽이다 — ratio>0.5 인 5개는 `m` 이 중앙값의 1/3.6,
  오차는 중앙값의 2.3배. 픽셀에서는 거의 안 보인다 (worst 케이스도 `|diff|` mean 0.0040).
- **`scripts/data/sd_anchor_error_filters.py` (신규)** — 같은 앵커 잔차를 **전수**(58044 pair)로
  재고 clip 통계(`sd_<split>_clip_stats.csv`)와 join 해서 필터 sweep 7종을 낸다. 이미지 디코딩
  없이 `_target_out` 만 직접 불러 재므로 인덱스 빌드 단계 필터로 그대로 쓸 수 있다.
  `--from_csv` 로 재측정 없이 집계만 다시 돌린다. 결과 → `results/mixed/sd_anchor_filters/`.
  실측 결론: **static clip 은 이미 리스트에서 빠져 있다** (학습에 쓰는 15566 clip 중 `was_static`
  0개) 인데도 `ratio>0.5` 가 1009/58044 = **1.74%**, max 3.8141 로 남는다. 단일 원인이 아니다 —
  ratio 와의 log 상관은 `resid_max` **+0.690** > `tgt_resid` +0.678 > `ctx_resid` +0.521 >
  `m` **−0.509** > `tgt_rad` −0.350 > `tgt_plen` −0.311 이고, 불량 1009 pair 중 **72.9%**(736)가
  "`m` 작음 ∧ resid 큼" 교집합(전체의 10.4%, 그 구간 불량률 **12.21%**)에 몰려 있다.
  비교: `m` 만 작은 구간 0.57% / resid 만 큰 구간 2.52% / 둘 다 아닌 구간 0.03%.
  ⇒ "안 움직이는 카메라" 컷만으로는 안 잘리고 **sim3 잔차가 지렛대**다 (m≥0.05 는 pair 26% 를
  버려도 `>0.5` 가 0.50% 로만 주는 반면, resid_max p90 컷은 10% 만 버리고 0.49%).
- **`main/conf/experiment/mix_dl3dv_sd_datadop_v1.yaml` + `_smoke.yaml` (신규 arm, Phase 6)** —
  DL3DV + Scene-Decoupled + DataDoP 3코퍼스 혼합. 공통부는 `da3_7k_da3geo_frontanchor` 와 동일한
  shape 규약(대조군이라 갈라지면 비교 불가). `scale_gain` 전부 1.0. smoke 검증 결과:
  배치 동질성 위반 **0/401**, 코퍼스 간 key set 동일(14키), shape 코퍼스별 일정
  (SD `(6,3,252,448)` / DataDoP `(1,3,252,448)`), `len(sampler)==실제 배치 수`(7337),
  `weights=[3,1,1]` 에서 dl3dv 배치 70→211 (3.01배), `val_max_batches=20` 안에 3코퍼스 전부
  (7/7/6). **알려진 갭**: SD 는 clip 간 sim3 정렬 잔차 때문에 DA3 view0 와 target 앵커 frame s 가
  정확히 일치하지 않는다 (n=300: 회전 med 0.46°/p95 2.06°, 앵커 오차/신호 크기 med 0.062/p95
  0.367, 신호의 50% 초과 2.3%). `geo_first_view_target_s` 는 SD 경로에서 **기능이 없어**
  (`_ctx_view_idxs` 가 덮어씀) 켜도 정합되지 않으므로 false 로 두고 기록만 한다.
- **코퍼스별 translation 레벨 정합 훅 `norm_scale_gain`** (`main/dataset_dl3dv.py` `_target_out`,
  기본 `1.0` = 동작 불변). `norm_scale` 을 gain 으로 **나눠서** 모델이 보는
  `m = mean_t‖C_t − C_s‖ / norm_scale` 이 gain 배가 되게 한다. 여기가 유일하게 안전한 주입점이다
  — `out['norm_scale']` 를 사후에 곱하면 `cam_param` 과 geo 채널이 이미 계산된 뒤라 **궤적과 RGBD
  단위가 갈라진다**. 세 코퍼스가 전부 이 함수를 지나므로 한 군데로 끝난다. `mixed` 경로에서는
  `datasets[i].scale_gain` 이 코퍼스별 cfg 복제본의 이 키로 복사된다.
- **`main/dataset_mixed.py` + `main/mixed_sampler.py` (신규, Phase 1-3)** — `dataset_name: 'mixed'`.
  `MixedCamDataset` 은 코퍼스마다 `copy.deepcopy(cfg)` + 화이트리스트 override 로 **평평한 cfg**
  를 만들어 기존 클래스를 그대로 생성한다 (`dataset_dl3dv.py`/`dataset_cfg.py` 무수정).
  전역 index 는 offset concat 이고 `out['corpus']` 태그만 추가한다 (`data_name` 은 **안 건드린다**
  — `seg_key` 의 `rsplit('_',1)` 과 eval 경로가 깨진다). `_check_key_sets` 로 코퍼스 간 key set
  동일성을 init 때 검사한다 (`collate_fn` 이 `batch[0]` 의 키만 순회해서 다르면 **조용히 유실**).
  `PerCorpusBatchSampler` 는 배치 하나 = 코퍼스 하나를 보장하고 (V 가 DL3DV≈6 / SD 6~49 /
  DataDoP 1 로 달라 `torch.stack` 이 터진다), weight 를 pool 반복(매번 재셔플)으로 실현하며,
  학습 루프가 `set_epoch` 을 안 부르므로 `__iter__` 안에서 epoch 을 self-advance 한다.
  **accelerate 1.12.0 설치본 소스를 직접 읽고 확인한 것**: `.batch_size` 는 반드시 노출해야 하고
  (`data_loader.py:165` 가 None + even_batches 면 raise), `drop_last=False` 면 tail padding 이
  앞 배치들을 평탄화한 인덱스에서 채워 와 (`:223,:258`) **코퍼스가 섞인 배치**를 만든다 →
  train/val 양쪽 다 `drop_last=True`.
- **`main/base.py` mixed 배선** — `build_dataset` 에 `'datadop'`/`'mixed'` 지연 import 분기,
  `_make_batch_generator` 는 기존 `if tr_list and te_list:` **앞에서** early-return 해서 현행 두
  경로를 바이트 그대로 둔다. 새 `_make_mixed_loaders` 가 `batch_sampler` 를 붙이고
  `itr_per_epoch = len(sampler)`.
- **`main/dataset_datadop.py` (신규, Phase 4)** — `DataDoPCamDataset`. shot 당 120 포즈 1 세그먼트,
  context 는 `<shot>_rgb.png` **한 장** (V=1), 포즈는 OpenGL c2w → `@ _GL2CV` → `inv`.
- **`sd_pair_mode: random_scene` (Phase 5, `main/dataset_scene_decoupled.py`)** — 고유 target clip
  당 1행으로 dedupe 하고 (whuman 52286 쌍 → 14017) context 는 그 scene 의 **리스트가 허용한**
  clip 중에서 매 `__getitem__` 마다 무작위로 뽑는다. 기본값 `list` 는 기존과 bit-identical.
  **인덱스 캐시 키에 `pm{mode}` 추가** — 지금까지는 정렬된 이름 목록만 해시해서 모드를 바꿔도
  옛 캐시를 조용히 재사용했다.
- **`scripts/data/corpus_scale_probe.py` (신규)** — 코퍼스 레벨을 **`__getitem__` 출력에서 직접**
  잰다 (`m = mean_t‖cam_param[t,6:9]‖`). 기존 표들은 pose.npz + avg_scale json 을 손으로 재현한
  값이라 `normalize_camera_extrinsics_and_points` / `max_trans_norm` / `_even_indices` /
  `norm_scale_gain` 이 빠져 있었다. 단일 코퍼스와 `mixed`(sub-dataset 별 그룹) 둘 다 처리.
  결과 `results/mixed/scale_probe/{config}_{stats.json,per_sample.csv,summary.md}`.
- **`scripts/data/avgscale_variant_levels.py` (신규)** — DL3DV avg_scale 변형 8종의 레벨/산포를
  7000 scene 전수로 비교 (`results/dl3dv/avgscale_variant_levels/`). 인과성(`[0,s)` 범위 제한)이
  레벨을 얼마나 움직이는지 재기 위한 것.
- **`scripts/data/datadop_scale_levels.py` (신규)** — DataDoP index 필터 구간별 keep 율 / 잔존 산포 /
  DL3DV 중앙값 정렬 gain / gain 적용 후 p95 를 스윕 (`results/datadop/scale_levels/`).
- **`main/conf/experiment/datadop_smoke.yaml` (신규)** — DataDoP 단독 smoke / 스케일 실측용.
  공통부는 `da3_7k_da3geo_frontanchor` 와 같은 shape 규약.
- **다중 코퍼스 혼합 학습 config surface (Phase 0)** — `main/conf/config.yaml` + `main/config.py`.
  전부 기본값이 현행 동작을 보존한다 (아직 읽는 코드가 없어 동작 변화 0).
  `dataset_name` 에 `'datadop'` / `'mixed'` 값 추가(문서화), `datasets: null` (코퍼스별 override
  블록 + 화이트리스트/금지 목록 명시), `mix_val_interleave: true`, `sd_pair_mode: list`,
  `datadop_root/caption_key/divisor/norm_gain/norm_trans_min/norm_trans_max/window_stride/
  index_workers/letterbox`.
  **`datadop_divisor: none` (D=1) 이 기본값**인 근거는 아래 `datadop_nodivisor_levels.py` 실측과
  `results/datadop/monst3r_scale_normalization.md` 소스 추적이다 — MonST3R 가 이미
  `norm_pw_scale=True` + `base_scale=0.5` 로 게이지를 걸어 놔서 한 번 더 나누면 잡음만 얹는다.
  필터 기본값 `[0.005, 1.0]` 도 D=1 스케일 기준(실측 keep 91.3%, 잔존 sd(log10 m) 0.685→0.488).
- **`results/datadop/monst3r_scale_normalization.md` (신규 문서)** — DataDoP world 좌표의 1 이
  무엇인지 MonST3R 소스로 추적. loss 단계(`dust3r/losses.py:177-181` 가 pred/GT 를 둘 다
  `norm_mode='avg_dis'` 로 나눔 → 미터 학습 불가), global alignment 단계
  (`cloud_opt/base_opt.py:107` `norm_pw_scale=True` 하드코딩 + `:64` `base_scale=0.5` +
  `:229-238` 가 pairwise 스케일 기하평균을 0.5 로 고정), GenDoP 내보내기가 스케일 미변경
  (`Dataset_DataDoP.py:267,338-400`; `core/provider.py:165-170` 는 학습 시점 on-the-fly).
- **`scripts/data/datadop_nodivisor_levels.py` (신규 분석 스크립트)** — DataDoP 를 혼합 학습에 넣을 때
  translation 분모(divisor)를 **아예 안 쓰는(D=1)** 후보를 실측한다. 기존
  `datadop_divisor_spread.py` 가 남긴 `per_shot.csv` 만 읽어 재스캔이 없다(~1초). 산포뿐 아니라
  **레벨과 꼬리**를 잰다: `m = mean_disp/D` 의 분위수, 다른 코퍼스 중앙값에 맞추는 상수 gain 과 그
  gain 적용 후 꼬리 위치, index 필터 잔존율, scene 내부 `sd(log10 m)`. 결과는
  `results/datadop/datadop_nodivisor_levels/stats.json`.
- **`main/conf/experiment/worldtraj_da3geo.yaml` (신규 arm)** — `20260731_001511_dl3dv_geo_worldtraj`
  (wandb `c4d2k5y4`) 의 세팅에서 **geo encoder 만** lagernvs → da3 로 바꾼 arm. 포즈는 그 run 그대로
  **COLMAP (`pose_source: transforms`, `meta_worldtraj.csv`)** 다. 그 run 의 저장된 config 와 현재
  기본값의 차이 8개(`meta_csv` / `geo_view_sampling` / `geo_cover_out_of_seg` / `geo_posed` /
  `geo_first_view_target_s` / `geo_anchor_first_frame`→`geo_cover_centered_at_s` /
  `geo_cover_subtract_first` / `geo_latent_cache_dir`)를 그대로 옮겼고, 의도적으로 다른 것은
  ① encoder, ② `geo_latent_cache_dir: null` (da3 는 LayerNorm/proj 가 학습돼 캐시 불가 — 어차피
  cascade 가 끈다), ③ `epochs 2000 → 100 + ckpt_at_epochs [50]` (현 캠페인 축에 맞춤) 세 개뿐이다.
  인덱스 39817 samples / 6095 scenes, 4479 it/epoch @ bs8. **held-out set 은 c4d2k5y4 와 같지
  않다** (코퍼스가 39,830→39,817 로 줄어 in-run `random_split` 이 재섞인다) → 대조는 공통 testset
  eval 로 할 것.
- **`scripts/smoke_before_only.py` (신규)** — 위 변경의 smoke. 실험 config 를 hydra 로 실제
  compose 해 dataset 을 만들고, 표본 세그먼트에서 `_sample_geo_frustum_cover` 가 돌려주는
  context view 가 ① view0 == `s` ② 나머지가 전부 `< s` ③ target `[s,e)` 를 안 건드리는지
  검사한다. 300 세그먼트 표본에서 전부 0 위반 (SMOKE PASS).
- **`scripts/eval_paired_bootstrap.py` (신규)** — 공통 testset eval arm 간 차이를 **세그먼트
  단위로 짝지어(paired)** 부트스트랩한다. 전 arm 이 `--sample-seed 42` 로 같은 3263 세그먼트 ·
  같은 x_T 를 썼으므로(common random numbers) arm 차이가 짝지어지고, 세그먼트를 리샘플해
  평균차의 95% CI 를 낸다. `preds_scores.csv` 에서 per-sample 인 열
  (`clatr/{clatr_score,pred_ref_cosine}`, `captions/{precision,recall,fscore}`)만 CI 를 내고,
  `clatr/{precision,recall,density,coverage,fcd}` 는 전 행에 전역 값이 복제된 집합 단위 지표라
  점추정만 보고한다 (CI 를 내려면 trajectory embedding 재추출 필요 — 디스크에는
  `token/test/`·`seq/test/` 에 caption embedding 만 있다). 결과
  `eval_my/common100ep/paired_bootstrap.{md,json}`. **CI 는 eval 추출 잡음만 덮고 학습 seed
  분산은 못 덮는다** (arm 당 학습 1회).
- **`scripts/data/datadop_divisor_spread.py` (신규)** — DataDoP 를 혼합 학습에 넣기 전에
  "frame-0 depth 로 translation 을 정규화하면 canonical 해지는가"를 재는 스크립트.
  `norm_divisor_compare.py` / `sd_clip_divisor_spread.py` 와 같은 지표(divisor D, reach,
  m = reach/D, 헤드라인 `sd(log10 m)`)를 쓴다. 22314 shot 전수 결과
  (`results/datadop/datadop_divisor_spread/`): **정규화가 산포를 못 줄인다** —
  raw(D=1) 0.655 → meanray 0.695, 다른 divisor(medray/meanz/medz/p10z/p90z)도 0.70~0.71.
  `corr(log10 reach, log10 D) = +0.077` 로 사실상 0이라 D 가 잡음만 얹는다. **point cloud 로
  올려서 재도 같다** — `ptcam`(mean||P−C_0||)은 `meanray` 와 수치가 같고(정의가 같다), DL3DV
  `centroid` 규약(`ptcent`) 0.721 / RMS 0.724 / bbox 대각 0.739 / 원경 40% trim 0.704 전부 raw
  보다 나쁘다. 스케일 동차 제약(sum a_i = 1) 아래 11개 feature 를 전부 쓴 **최적 조합의 상한도
  0.691** (holdout 0.683, R^2 < 0) = frame-0 에서 만드는 어떤 divisor 도 raw 를 못 이긴다.
  원인은 **MonST3R 가 metric 이 아니고 재구성이 이미 shot 단위로 정규화돼 있다**는 것 — DUSt3R
  loss 가 pred/GT 를 둘 다 `avg_dis`(점들의 카메라까지 평균거리)로 나누고(`dust3r/losses.py:178-181`,
  `utils/geometry.py:293-295`), global aligner 가 `norm_pw_scale=True` + `base_scale=0.5` 로
  pairwise scale 기하평균을 고정한다(`cloud_opt/base_opt.py:66,108,230-233`). GenDoP 는 저장 시
  스케일을 안 건드린다(`Dataset_DataDoP.py:338-400` 은 coordinate flip + clean + 리샘플뿐;
  `core/provider.py:165-170` 의 `max||c2w_t||` 정규화는 학습 시점 on-the-fly).
  방증: `sd(log10 D)` 전체 0.288 인데 같은 소스 영상 안에서만도 0.249. 결론은 그래도 `meanray` 사용 — 산포가 아니라 **추론 가능성**
  때문이다(raw 는 새 입력 이미지에 정의되지 않는다). 자세한 건 그 디렉토리의 `summary.md`.
- **`front_first_anchor` arm 2개 추가 + `vae_latent_scale` 실측치** — experiment config 만
  추가/수정, 코드 변경 0줄.
  - `da3_7k_lagernvsgeo_frontanchor.yaml` (신규): 기존 `da3_7k_da3geo_frontanchor.yaml` 에서
    `geo_encoder` 만 `da3` → `lagernvs` 로 바꾼 짝 arm. target 쪽(pose_source, avg_scale_ref,
    scale_mode, intr_norm, cam_dim, vae_latent_scale, seg list, blacklist)이 전부 같아
    학습 인덱스가 동일하고 paired 비교가 된다. `geo_latent_cache_dir: null` 로 못 박았다 —
    기존 캐시는 `pose_source='transforms'` + `meta_worldtraj` 로 만든 것이라 da3 pose 와 다른
    카메라로 만든 latent 이 조용히 들어온다.
  - `da3_7k_da3geo_frontanchor_vaescale.yaml` (신규): `da3_7k_da3geo_frontanchor.yaml` 에서
    **`vae_latent_scale` 한 줄만** 바꾼 arm. 기존 da3 arm 들이 물려 쓰던 0.96032625 는 SCVideo
    시절 상수라 `pose_source='da3'` + `avg_scale_ref='front_first_anchor'` 분모에 맞춰 측정한
    값이 아니다. 학습은 `camera_vae.encode(traj) / cfg.vae_latent_scale` 로 나누므로
    (`train_cam_dm_attn_sup.py:233`) 어긋나면 diffusion 이 std != 1 인 분포를 배운다.
    `scripts/vae/vae_scale_matrix.py` 를 **전체 코퍼스**(`MAX_SCENES=none`, `meta_da3_7k.csv`,
    32677 sample = frontanchor arm 의 학습 인덱스 수와 일치) 로 돌려 `intr=rel` 에서
    **lat.std 0.62722** (기존값의 0.6531 배), recon rot 0.00639 / trans 0.00537 / intr 0.00316.
    **이 값을 쓰는 arm 은 이것 하나뿐**이고 나머지는 기존 값을 유지한다 (다른 arm 과 loss 를
    같은 축에서 비교하기 위해). 자세한 수치는 `EXPERIMENTS.log [12]`.
- **`geo_encoder: 'da3'` — Depth-Anything-3 백본을 geo encoder 로 쓰는 네 번째 backend.**
  기존 `custom` 은 frozen DINOv2-L (single-view) 라 view 간 대응을 못 만들고 3D 는 dataset 이
  깔아 준 Plücker/log-depth 픽셀 채널로만 들어갔다. DA3 백본은 block 13 부터 local/global
  attention 을 번갈아 도는 **cross-view ViT** 라 geometry 를 feature 안에서 직접 푼다. 게다가
  코퍼스의 `<scene>/da3/*.npz` 가 이미 `DA3NESTED-GIANT-LARGE-1.1` 산출물이라 표현 공간이 맞는다.
  - 신규 `models/da3_geo_encoder.py::DA3SceneEncoder` + `models/geo_encoder.py::_DA3Backend`.
    `_LagerNVSBackend` / `_CustomBackend` / `custom_geo_encoder.py` 는 **0줄 수정**이고,
    vendored `tools/Depth-Anything-3` 도 **0줄 수정**이다. `_DA3Backend` 의 표면을
    `_LagerNVSBackend` 와 동일하게 맞춰서 `geo_encode` 사본 5개도 안 고쳤다.
  - **`depth_anything_3.api` 는 import 하지 않는다** — ① `api.py` → `utils/export/gs.py` 가
    `moviepy.editor` 를 끌고 오는데 이 env 에 moviepy 가 없고, ② `DepthAnything3.forward` 에
    `@torch.inference_mode()` 가 걸려 있어 그대로 쓰면 downstream autograd 가 오염된다.
    대신 `cfg.load_config` + `registry.MODEL_REGISTRY` + `safetensors` 로 net 을 직접 만들고
    `head`/`cam_dec`/`gs_head`/`gs_adapter` 를 지운 뒤 `backbone`/`cam_enc` 만
    `strict=True` 로 로드한다 (739 tensor, 1251.0 M, 전부 frozen).
  - **pose 주입**은 기존 `geo_posed` 플래그를 재사용하고, 정규화는 파이프라인의 `norm_scale` 이
    아니라 **DA3 자신의 규약** (view0 기준 재고정 + median camera distance 로 나눔) 을 쓴다.
    단 `api.py:435-447` 의 median 은 **배치 전역**이라 B>1 에서 샘플이 서로 섞인다 — 우리
    `build_cam_token` 은 **per-sample median** 으로 고쳤다. B=1 에서는 원본과
    `max|api-ours| = 1.2e-07 ~ 2.4e-07` (fp32 반올림) 으로 일치, B=4 에서는 의도대로 갈라진다.
    `cam_enc` 는 원본 `da3.py:127` 과 동일하게 autocast 를 끄고 fp32 로 돌린다.
  - `cam_token` 을 넘기면 `select_reference_view`/`reorder_by_reference` 가 건너뛰어져
    **view 순서가 보존**된다 (그 경로는 `V >= 3` **이고** `cam_token is None` 일 때만 발동).
    그래서 `geo_shuffle_order=True` 는 `da3` 와 함께 쓰면 raise 하고, `geo_posed=False` 인데
    `V < 3` 이어도 raise 한다 (ref-view 선택 유무로 동작이 조용히 달라지는 자리).
  - 토큰 dim 은 1024 가 아니라 **3072** (vitg embed 1536 × `cat_token`). `cat_token` 의 두 반쪽은
    분산이 크게 다르므로 (앞쪽 un-normed local_x, 뒤쪽 `self.norm` 통과) `Linear(3072→768)` 앞에
    `LayerNorm(3072)` 을 넣는다. 학습 파라미터는 그 둘뿐 — 4 tensor / 2.37 M,
    `last_geo.pth` = 9,467,303 B 로 DA3 739 tensor 가 ckpt 에서 빠지는 것 확인.
- **`da3_geo_*` config 9종** (`main/config.py`, `main/conf/config.yaml`):
  `da3_geo_repo_path` / `da3_geo_model` / `da3_geo_ckpt_path` / `da3_geo_hf_home` /
  `da3_geo_input_hw` / `da3_geo_layers` / `da3_geo_layer_fuse` / `da3_geo_norm` / `da3_geo_debug`.
  기본 모델은 `da3nested-giant-large`, 기본 입력은 `[252, 448]` (14 배수, 격자 18×32 = view 당
  576 토큰 — `custom` 과 동일). `main/dataset_cfg.py` 에 `geo_da3` 분기를 추가해 `geo_hw` 를
  `da3_geo_input_hw` 로 덮어쓰고, `geo_latent_cache_dir` 은 disable 한다
  (LN/proj 가 학습되므로 출력을 얼릴 수 없다 — cascade 2b/5 에 항목 추가).
- **`scripts/data/da3_res_probe.py`** — 252×448 vs 280×504 해상도 probe.
  DL3DV 3 + DynamicVerse 3 scene 을 두 해상도로 돌려 depth mp4 와 AbsRel/δ 를 뽑고,
  `--repro-check` 로 저장된 `<scene>/da3/depth.npz` 와 대조해 api 우회 로딩 경로가
  `api.DepthAnything3` 와 같은 결과를 내는지 증명한다. 결과는 `results/20260812_da3_res_probe_giant`:
  448 vs 504 는 AbsRel 0.0176~0.0564 / δ<1.25 0.958~0.9995, 504 vs 저장본은 AbsRel 0.0069~0.0164.
- **실험 arm `main/conf/experiment/da3_7k_da3geo.yaml`** — `da3_7k_customgeo_nos.yaml` 의 인코더만
  바꾼 판 (코퍼스/seg list/blacklist/VAE/cam_dim/epoch/batch 동일) 이라 차이가 인코더 탓임이 분명하다.
  smoke config `da3geo_smoke.yaml` / `da3geo_smoke_unposed.yaml` / `customgeo_regr_smoke.yaml` 동봉.
- **`avg_scale_ref` 변형 3종 추가** (`main/dataset_cfg.py::AVG_SCALE_DIRS`, `pose_source='da3'` 전용).
  target translation 을 나누는 분모를 어느 context range 에서 어느 기준점으로 잴지 고르는 노브다.
  점 구름 규약(da3 depth unproject, `conf >= 전역 P40`, `pixel_stride 2`, 평균 거리)은 넷 다 동일.
  | ref | context range | 기준점 | 디렉토리 |
  |---|---|---|---|
  | `centroid` (기존 기본) | `[0,s)`/`[e,N)` 중 긴 쪽 | context 카메라 center 들의 centroid | `da3/avg_scale/` |
  | `context_first_cam` | 위와 같음 | context range 의 첫 카메라 | `da3/avg_scale_context_first_cam/` |
  | `front_first_anchor` **(신규)** | `[0, s)` — 앞쪽만 | target segment 첫 카메라 s | `da3/avg_scale_front_first_anchor/` |
  | `front_first_anchor_same_len` **(신규)** | `[s-L, s)`, L=e-s | target segment 첫 카메라 s | `da3/avg_scale_front_first_anchor_same_len/` |
  | `da3latent` **(신규)** | `[s-L, s)` 의 geo view | **M·σ — 기하 거리가 아니다** (아래) | `da3/avg_scale_da3latent/` |
  - `front_*` 계열은 앞쪽 context 를 못 채우는 세그먼트를 **인덱스에서 제외**한다
    (`dataset_cfg.avg_scale_min_front` → `_min_front`, `_load_index`/`_load_index_subset` 필터).
    da3_7k 는 전 세그먼트 길이 49 / s ∈ {0,49,98,…} 이라 두 변형이 **정확히 같은 집합**(s==0)을
    빼므로 arm 간 paired 비교가 성립한다. 인덱스 캐시 키에 `__as<ref>` 를 붙여 (거르는 변형일 때만)
    centroid arm 의 캐시를 물어오지 않게 했다 — 안 그러면 분모 파일이 없어 `_avg_scale` 이 raise.
  - **`da3latent` 은 기하 거리가 아니라 DA3 latent 이 실제로 쓰는 카메라 스케일**이다.
    `geo_posed=True` 로 `cam_enc` 에 GT pose 를 넣어도 `cam_dec` 의 **translation 은 항상 예측값**
    (`cam_dec.py:35` 의 `out_t = self.fc_t(feat)` 에는 echo 경로가 없고 rotation/fov 만 echo 되며
    `da3.py:216` 은 `camera_encoding` 없이 부른다). `c_fed`=(GT center, view0 기준)/M,
    `c_pred`=`cam_dec` 출력의 center, `σ`=Umeyama(src=`c_pred`, dst=`c_fed`).scale 이라 하면
    GT 상대좌표 = M·`c_fed` = (M·σ)·`c_pred` → **latent 1 단위 = M·σ GT 미터**. 그 M·σ 를 분모로
    저장한다. Umeyama 의 R/t 는 쓰지 않는다 (앵커는 front 계열과 같이 target 첫 카메라 s).
    `cam_token` 이 `avg_scale`/`norm_scale` 을 안 쓰므로 순환이 없다.
    `geo_view_sampling != 'front_uniform'` 이면 raise (분모를 만든 view 집합과 달라지므로).
- **`geo_view_sampling: 'front_uniform'`** (`main/config.py`, `main/dataset_dl3dv.py`) —
  context view 를 target 바로 앞 `[s-L, s)` (L=e-s) 에서 `geo_num_views` 장 uniform 으로 뽑는다.
  coverage retrieval 없음 / 결정적 / leakage 없음. `front_first_anchor_same_len` · `da3latent` 과
  range 가 일치하며, 분모 ref 와 어긋나면 `dataset_cfg` 가 경고한다.
  주의: `frustum_cover` 와 달리 view0 가 항상 `s-L` 로 고정되므로 (DA3 는 view0 가 reference)
  두 sampler 를 비교할 때 **분모 변경과 view0 결정성이 함께 바뀐다**.
- **`DA3SceneEncoder(keep_cam_dec=True)` + `predict_cameras()`** (`models/da3_geo_encoder.py`) —
  기본값 `False` 는 기존과 동일하게 `cam_dec` 을 삭제하므로 기존 arm 은 비트 단위로 같다.
  `True` 면 `cam_dec` 가중치까지 로드해 `predict_cameras(images, cam_token)` 로 latent 에서
  c2w/K 를 뽑을 수 있다 (`da3.py:218` 과 동일하게 autocast 를 끄고 fp32).
- **`scripts/data/make_avg_scale_da3_front_anchor.py`** — 위 두 front 변형 분모 생성기
  (전 코퍼스 7000 scene, 각 38989 파일).
  **`scripts/data/da3_latent_scale_probe.py`** — σ 분포 측정 (파일을 쓰지 않는다).
  **`scripts/data/make_avg_scale_da3_latent.py`** — `da3latent` 분모(M·σ) 생성기. view 선택/resize/
  `cam_token` 을 재구현하지 않고 학습이 쓰는 `CamDataset` + `DA3SceneEncoder` 를 그대로 돌린다.
  기본으로 이미 있는 json 은 **이미지 로드 전에** 인덱스에서 빼서 중단 후 이어 돌릴 수 있다
  (`--overwrite` 로 재계산). `--splits` 기본값은 `train` 하나 — `base.build_dataset` 이
  `CamDataset(cfg)` 하나만 만들고 train/val 은 그 위 `Subset` 분할이라(`base.py:69-89`)
  `train` 인덱스가 이미 전 코퍼스다.
- **실험 arm 5종** — `da3_7k_da3geo_frontanchor.yaml`, `da3_7k_da3geo_frontanchor_samelen.yaml`,
  `da3_7k_da3geo_latentscale.yaml`, 그리고 분모만 같게 맞춘 text-only 비교군
  `da3_7k_da3pose_frontanchor.yaml`, `da3_7k_da3pose_frontanchor_samelen.yaml`.

### Fixed
- **`geo_encoder='da3'` 에서 geo latent 의 앵커가 target 궤적의 앵커와 다른 카메라였다** —
  DA3 는 `cam_token` 을 **view0 기준**으로 재고정한다 (`da3_geo_encoder.build_cam_token`,
  DA3 원본 `api.py:439-440 _normalize_extrinsics` 와 같은 식: `w2c @ c2w[:, :1]`).
  반면 target 은 `rel = E @ inv(E_s)` 로 **타깃 세그먼트 첫 프레임 s 기준**이다. 지금까지 da3 arm
  들은 `geo_first_view_target_s: false` 라 view0 가 context 프레임이었고, 그래서 geo latent 이
  표현하는 궤적과 모델이 맞춰야 하는 궤적이 **강체변환만큼 어긋난 채** 학습됐다 (스케일 분모로는
  못 고치는 축이다 — `avg_scale_da3latent` 도 스케일만 고친다).
  이제 da3 arm 4 개 전부 `geo_first_view_target_s: true` 로 **view0 = 프레임 s** 다.
  검증: 6 샘플에서 `geo_c2w[0] == c2w[s]`, 재고정 후 `w2c[0] == I`, `target rel[0] == I`.
- **`_sample_geo_front_uniform` 이 `geo_first_view_target_s` 를 무시했다** — `frustum_cover` 는
  이미 이 플래그를 구현하고 있었는데(`k_retr = geo_cover_k - 1` 후 `picks = [s] + picks`)
  `front_uniform` 에는 그 경로가 없어 항상 `[s-L, s)` 오름차순이었다. 이제 플래그가 켜지면
  **`[s-L, s]` (s 포함) 구간에서 `geo_num_views` 장 uniform → 역순**으로 돌려
  `[s, s-10, s-20, ...]` 을 준다 (L=49, V=6). s 는 V **안에** 들어 토큰 예산 `V*P = 6*576 = 3456`
  이 다른 arm 과 같다. s 를 따로 prepend 하고 나머지 5 장을 `[s-L, s)` 에서 뽑는 방식은 마지막
  픽이 `s-1` 이라 view0 과 1 프레임짜리 중복 view 가 생겨서 쓰지 않았다.
  플래그가 꺼진 경우(legacy)의 동작은 bit-identical.
  참고: DA3 는 `cam_token` 을 주면 `select_reference_view` 재정렬이 꺼지고 RoPE/cls 토큰이 view
  인덱스를 쓰지 않아 **view 순서 자체는 permutation-equivariant** 다. 즉 역순은 no-op 이고 실제
  효과는 전부 "view0 이 어느 카메라인가"에서 나온다.
- **`dataset_cfg.py` 에 앵커 가드 2 개 추가** — ① `geo_encoder='da3' + geo_posed=True` 인데
  `geo_first_view_target_s=False` 면 앵커가 어긋난다고 `say()` 경고. ② `avg_scale_ref='da3latent'`
  는 `geo_first_view_target_s=True` 를 **raise 로 강제** — 저장된 `M*sigma` 는 view0 = s 인 view
  집합으로 `cam_dec` 를 돌려 만든 값이라 view0 이 바뀌면 M 도 sigma 도 달라져 그냥 틀린 분모가 된다.
  `[geo da3]` 요약 로그에 `view0=frame s | context` 를 찍는다.
- **`train_latent_cam_dm.py` 의 full-resume 이 optimizer 모멘트를 엉뚱한 param 에 실었다** —
  `opt.state_dict()` 는 param 을 **인덱스**로만 참조하는데, 저장 이후 `nn.Module` 등록 순서가
  바뀌면 개수가 같아도 인덱스가 어긋난다. 이제 `resume.pth` 에 `opt_param_names` (optimizer
  param 순서의 이름 목록)를 같이 저장하고, 재개 시 이름 기준으로 인덱스를 재매핑한다.
  이름 목록이 없는 구 ckpt 는 `model`/`geo` state_dict 키 순서로 복원하고(개수가 맞을 때만),
  로드 직후 param 과 `exp_avg` 의 shape 를 대조해 틀리면 즉시 `RuntimeError` 로 세운다.
  이름 집합 자체가 다르면 구조가 바뀐 ckpt 이므로 명시적으로 거부한다.
  실제 사고: `sd_whuman_customgeo` (wandb `fkfqww00`) 재개가
  `RuntimeError: The size of tensor a (14) must match the size of tensor b (1024)` 로 죽었다.
  커밋 `916bc7a` (custom_geo_channels ablation) 가 `SceneEncoder.__init__` 에서 `self.ln_d` 를
  `self.geo` 앞으로 옮기면서 GeoTokenizer 16개가 2칸 밀려, `geo.ray.weight` 의 `exp_avg` 가
  `ln_d.weight` 의 grad 와 짝지어진 것. 재매핑 후 280개 중 18개 이동으로 정상 재개.

- **`dataset_scene_decoupled.py::_load_images` 가 일시적 mp4 디코딩 실패 한 번에 학습 전체를
  죽였다** — 이제 재시도한다 (`_DECODE_RETRIES=3`, `_DECODE_RETRY_SLEEP=0.5s`, attempt 마다
  ×(attempt+1)). 3 회 후에도 0 프레임이면 기존과 동일하게 `RuntimeError`.
  `sd_whuman_customgeo` (wandb `fkfqww00`) 가 epoch 5, 19:37 에 이걸로 죽었다
  (`scene3292_5x5_loc115_scene_Yakohama_02_24mm.mp4`). **파일 손상이 아니다** — 같은 mp4 를
  직후 cv2 로 3/3 회 재디코딩해 매번 81 프레임 정상. `/data1` 이 Lustre 라 동시 I/O 가 몰리면
  `VideoCapture` open 이 일시적으로 실패한다. 자세한 경위는 `FIX.log`.

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

### Changed
- **`CamDataset.__init__` 의 config 해석을 `main/dataset_cfg.py` 로 분리 (동작 변화 없음).**
  `__init__` 190 줄 중 ~155 줄이 `getattr` + 검증 + print 였고 실제 초기화(리스트/인덱스)는
  20 줄 남짓이라, "이 arm 에서 결국 어떤 값이 서는가"를 알려면 190 줄을 순서대로 읽어야 했다.
  이제 `spec = resolve_dataset_cfg(cfg); spec.apply_to(self)` 두 줄이고 `__init__` 은 35 줄이다
  (`dataset_dl3dv.py` 1546 → 1366 줄).
  - `DatasetSpec` dataclass 의 **필드 이름 = 데이터셋 속성 이름**이라 `ds.geo_cover_k` 같은
    외부 접근은 전부 그대로다. `resolve_scale_mode` / `resolve_pose_source` / `_POSE_SOURCES` /
    `_SCALE_MODE_ALIASES` 도 `dataset_cfg.py` 로 옮겼지만 `dataset_dl3dv` 가 re-export 하므로
    `from dataset_dl3dv import CamDataset, resolve_scale_mode`
    (`scripts/data/viz_scene_chunk_scale.py:108`) 는 안 깨진다.
  - **`__getitem__` 과 모델 코드는 안 건드렸다.** 특히 모델은 `nn.Module` 등록 순서가 바뀌면
    `resume.pth` 가 죽으므로 (위 Fixed 의 `916bc7a` 사고) 이번 범위에서 제외했다.
  - `self.covis_topM` 제거 — `config.py:219` 가 이미 "unused" 라고 적어 뒀고 참조하는 코드가
    없다 (`scripts/context_select/*.py` 는 자기 지역 기본값 32 를 쓴다).
  - `geo_latent_cache_dir` 은 **평평한 매핑이 아니라 순서 있는 상태 기계**다 — 한 번 켜졌다가
    `geo_latent_dim != 768` / `geo_encoder='custom'` / `geo_test_inseg_k` / `geo_swap_mode`
    네 곳에서 차례로 꺼진다. 순서와 print 문구를 그대로 보존했다.
- **`scripts/viewer/viser_val_cameras.py` 에 `--dataset sd` 분기 추가** (기본값 `dl3dv` 라
  기존 동작은 그대로).  SD 는 `data_name` 이 `{scene}__{TARGET}__{CONTEXT}` 이므로 **context clip**
  의 da3 depth/conf/pose 를 읽어 배경을 만든다.
  - point cloud: `da3/<split>/<scene>/<CONTEXT>/{depth,conf}.npz` 를 `umeyama_gt.json` 의 sim3 로
    GT meters 에 올려 unproject (`dataset_scene_decoupled.py::_load_scene` 과 같은 식:
    `R'=R_e·Rᵀ`, `t'=s·t_e−R'·t`, `d'=s·d`). 색은 context mp4 프레임을 depth 격자(504×294)로
    리사이즈해서 사용. `--pc-frames/--pc-stride/--pc-conf-pct/--pc-max-points/--pc-size` 로 조절.
  - context clip 카메라를 초록 frustum 으로 같이 그린다 (target/ref 파랑, pred 빨강은 기존과 동일).
  - SD world 는 z-up (Unreal) 이라 up direction 을 `+z` 로 (`--up` 으로 override).
  - static clip (`umeyama_gt.json` 의 `moving: false`) 은 sim3 의 s/t 가 없어 배경을 건너뛴다.
- **`scripts/viewer/viser_val_cameras.py`: sequence load 시 터미널에 지표 출력** (dl3dv/sd 공통).
  - 궤적 오차는 ref/pred 로 그 자리에서 계산 (`traj_metrics`) — `pos_rmse/mean/max/end`,
    `rot mean/max` (deg), ref·pred path length. 정의는 `scripts/render/compare_textonly.py` 와
    동일. 단위는 world 단위 그대로 찍고 (sd `m`, dl3dv 정규화 `u`) 코퍼스 간 비교는 안 된다.
  - 평가 지표는 `<root>/../preds_scores.csv` 의 해당 행. **PRDC/FCD 열은 set 단위로 한 번
    계산된 값이 전 행에 복사돼 있어** per-sequence 로 오독하기 쉬우므로, 전 행이 동일한 열은
    corpus-level 로 분리해 시작할 때 한 번만 찍는다.
  - **sample 별 PRDC 재계산** (`load_prdc_per_sample`, `--no-prdc` 로 끔). PRDC 는 `.mean()`
    직전까지 sample 별 값이 있어서 `preds.npy` 의 CLaTr latent 로 다시 계산해 찍는다 —
    `precision`(pred 가 GT 초구 안? T/F), `density`(pred 를 감싼 GT 초구 수/k), `recall`·
    `coverage`(GT 쪽 T/F) + 그 sample 이 속한 split. prdc.py 와 같은 조건(k=3, euclidean,
    5 splits, 저장 순서대로 chunk)으로 맞췄고, 시작할 때 재계산 집계를 찍어 `metrics.json`
    과 일치하는지 눈으로 확인할 수 있게 했다 (SD/DL3DV 양쪽 자리수까지 일치 확인).
  - **`--only` 로 PRDC 실패 sequence 만 골라 보기** (`prec-fail` / `dens-low` / `recall-fail`
    / `cov-fail` / `fail`, 기본 `all` 은 기존 동작). 거른 목록을 density 오름차순 표로 찍고
    slider/Next 도 그 목록만 돈다. `--dens-max` 로 density 기준, `--list-out` 으로 csv 저장.
    거르기는 **PRDC 계산이 끝난 뒤에** 한다 — seqs 를 먼저 줄이면 5 splits chunk 구성이
    달라져 값 자체가 바뀐다.
  - `--cam-stride` (target/pred frustum, 기본 1 = 기존과 동일), `--context-stride`
    (sd context frustum, 기본 = `--grey-downsample` 이라 기존과 동일) 추가.
- **`scripts/render/compare_textonly.py` 에 run 인자와 SD 용 그림 두 종류 추가** (인자 없이 돌리면
  기존 `20260719_210144_dl3dv_textonly` 동작 그대로).
  - `--run` / `--out`: 하드코딩돼 있던 run 경로를 인자로. `--no-per-target` 으로 target 별 PNG 생략.
  - `--grid`: 전 target 을 contact sheet 한 장으로. 프리셋 → pos_rmse 순 정렬.
  - `--preset-overlay`: SD 카메라 프리셋(`01_24mm` 등)별 GT/pred 중첩 + 프리셋 평균 궤적.
  - `_vs_geo.png` (point/dist 분포 비교)는 기본 run 이 아니면 건너뛴다 — 다른 코퍼스에 대고
    그리면 의미 없는 비교가 나온다.
  - 그림 제목은 ASCII 로 (matplotlib DejaVu Sans 에 한글 글리프가 없어 tofu 로 깨졌다).

  **결과 (`results/compare/sd_whuman_textonly/`, `sd_whuman_textonly` testset 80 target,
  world 단위 = umeyama 로 GT 미터에 맞춘 SD 좌표계):**

  n=80 pos_rmse mean 1.1627 / median 0.7932, rot_mean 11.48 deg, CLaTr 48.72

  | preset | n | GT reach | pred reach | GT spread | pred spread | pos_rmse | rmse/reach |
  |---|---|---|---|---|---|---|---|
  | 01_24mm | 19 | 3.651 | 3.725 | 1.035 | 1.854 | 1.351 | 0.370 |
  | 02_24mm | 19 | 3.728 | 2.287 | 1.042 | 1.795 | 1.915 | 0.514 |
  | 05_24mm | 15 | 1.956 | 1.504 | 1.049 | 1.097 | 0.909 | 0.465 |
  | 06_24mm | 12 | 1.543 | 1.749 | 0.437 | 0.883 | 0.588 | 0.381 |
  | 07_24mm | 15 | 1.454 | 1.051 | 0.811 | 0.848 | 0.685 | 0.471 |

  reach = `mean_n max_t ||T(t)-T(0)||`, spread = 프리셋 내 평균 궤적으로부터의 RMS 편차.
  **pred spread 가 5/5 프리셋 전부에서 GT spread 보다 크다** (01 1.79x, 02 1.72x, 06 2.02x).
  즉 world 공간에서 이 run 은 프리셋 평균으로 붕괴한 게 아니라 **과하게 퍼져 있다**.
  `metrics.json` 의 clatr precision 0.15 / density 0.1792 / coverage 0.175 / fcd 319.9549 와
  방향이 일치한다 (recall 0.8375 만 높다).
  주의: `mean_traj_baseline.py` 의 "평균 예측 수준" 판정은 `cam_param`(분모 D 로 정규화,
  rot6d+intr 포함) 공간이고 이 표는 denormalize 된 world 위치 공간이라 **같은 양이 아니다.**

- **위 SD textonly 평가를 n=160 으로 재실행** (`results/compare/sd_whuman_textonly_n160/`).
  기존 80 은 `batch_size 4 x val_max_batches 20` 의 산물이라 DL3DV_nos(160)와 표본수가 달랐다.
  `scripts/eval_testset.py --ckpt last.pth --max-batches 40` 으로 맞췄다 (`eval_testset.py` 는
  `val_max_batches` 를 무시하고 `--max-batches` 를 쓴다). **재샘플링이므로 앞의 80 과 겹치는
  target 도 값이 그대로 재현되지는 않는다.**

  `metrics.json` (n=160): captions precision 0.5993 / recall 0.5014 / fscore 0.5381,
  clatr_score 46.9889, clatr precision 0.1625 / recall 0.8188 / density 0.1521 /
  coverage 0.175 / fcd 222.2015, loss_latent 0.05423911759862676,
  loss_traj 0.008784372146328679, sampling_sec 27.8.
  (n=80 값: 0.5928 / 0.5443 / 0.5626, 48.6265, 0.15 / 0.8375 / 0.1792 / 0.175 / 319.9549.)

  top-down (n=160): pos_rmse mean 1.2561 / median 0.9093, rot_mean 9.81 deg.

  | preset | n | GT reach | pred reach | GT spread | pred spread | pos_rmse | rmse/reach |
  |---|---|---|---|---|---|---|---|
  | 01_24mm | 34 | 5.151 | 4.252 | 1.727 | 1.979 | 1.756 | 0.341 |
  | 02_24mm | 34 | 4.578 | 3.163 | 1.241 | 1.842 | 1.851 | 0.404 |
  | 05_24mm | 33 | 2.232 | 1.780 | 1.247 | 1.212 | 0.960 | 0.430 |
  | 06_24mm | 28 | 2.319 | 2.195 | 0.944 | 1.045 | 0.850 | 0.367 |
  | 07_24mm | 31 | 1.516 | 1.385 | 0.897 | 0.915 | 0.737 | 0.486 |

  **over-spread 결론은 부호만 유지되고 크기는 크게 줄었다.** pred spread > GT spread 는
  n=160 에서도 5/5 프리셋이지만 배율이 01 1.79x→1.15x, 02 1.72x→1.48x, 06 2.02x→1.11x,
  05 는 1.05x→0.97x 로 사실상 동률이다. n=80 의 GT spread 가 프리셋당 12~19 개뿐이라
  과소추정된 것으로 보인다. pred reach 는 n=160 에서 5/5 전부 GT reach 보다 작다 (과소 이동).

  per-sample PRDC (`prdc_per_sample.py`, 재현 검증 precision 0.1625 / density 0.1521 =
  보고값과 일치): manifold 안 26/160, density>0 26/160, caption fscore mean 0.5048,
  fscore>=0.999 인 47 개의 precision 0.2340 / density 0.2553.

  CLaTr latent (`viz_clatr_latent_pca.py` → `results/compare/clatr_latent_pca_n160/`):
  SD_textonly n=160 → 2-D PCA 설명분산 34.8%, k-NN 반경 med 15.618,
  pred→최근접GT med 19.799, ratio **1.268** (n=80 의 1.349 에서 소폭 하락).
  DL3DV_nos 는 그대로 25.258 / 23.250 / 0.921. **표본수를 160 으로 맞춰도 PRDC 격차는
  거의 그대로다** — chunk 크기 실험으로 이미 기각한 표본수 artifact 가설과 일치한다.

### Changed
- **`scripts/viewer/viser_val_cameras.py --dataset sd` 가 target clip point cloud 도 그린다** —
  `--pc-clips {context,target,both}` (기본 `context` = 기존 동작 그대로). 두 clip 은 **서로 다른**
  `umeyama_gt.json` sim3 를 타고 같은 GT meters world 로 올라오므로, `both` 로 겹쳐 보는 것이 곧
  aligned world 검증이다. target 을 그릴 때 그 clip 의 `pose.npz` 카메라와 ref(파랑) 카메라의
  위치차도 찍는다 — 두 값이 서로 다른 경로(`pose.npz`+sim3 / json 의 `first_extrinsic` 복원)로
  나오므로 0 이 아니면 world 가 어긋난 것이다. 실측: `scene1002_..._01_24mm` 에서 median/max
  모두 0.0000m. 부수 옵션 `--pc-tint` (RGB 대신 context=초록/target=파랑 단색으로 구분),
  `--pc-target-cams` (target clip 카메라도 그림), `--pc-seg-only`. point cloud 노드 이름도 `pc` ->
  `pc_context`/`pc_target_clip` 으로 갈라 viser scene tree 에서 따로 껐다 켤 수 있다.
  clip 하나가 실패해도(static clip 은 sim3 의 s/t 가 없다) 나머지는 그대로 그린다.

  **clip 길이 ≠ target segment 길이라 파란 궤적이 둘로 보이던 것을 고쳤다.** SD clip 은
  81 프레임인데 target segment(`ref`)는 앞 **49** 프레임이다 (`whuman` 160/160 sequence 에서
  offset 0, 위치오차 `<1e-3 m` 확인). 자르지 않으면 뒤 32 프레임이 segment 옆을 median
  **0.926m** 로 나란히 지나가 (궤적 extent 4.173m 의 22%) 별개의 두 번째 궤적처럼 읽혔다.
  이제 `--pc-seg-only` 로 target clip 의 카메라와 point cloud 를 `[0, len(ref))` 로 자르고
  (`sd_pointcloud(t_range=...)` 추가), target clip 카메라 색을 ref 파랑과 겹치던
  `(40,90,230)` 에서 **청록 `(0,200,200)`** 으로 분리했다. context 는 clip 전체를 쓰는 게
  맞으므로 영향받지 않는다.

### Added
- **`conf/experiment/sd_whuman_customgeo_v6.yaml`** — SD customgeo 의 context view 수만
  49 -> 6 으로 줄인 arm (`sd_geo_views: 6`). **코드 변경 없음** — `SDCamDataset._ctx_view_idxs`
  가 이미 `_even_indices` 로 균등 추출한다 (`np.linspace(0,48,6).round()` =
  `[0, 10, 19, 29, 38, 48]`, 양 끝점 포함 결정적 인덱스).
  원본 대비 실질 차이는 `sd_geo_views` 하나뿐이고 (hydra resolve 후 122 key 전수 비교로 확인)
  batch_size 4 / epochs 50 / num_thread 16 / seg_list 는 paired 비교를 위해 그대로 뒀다.
  smoke 실측: **4.23 it/s / 25799 MiB** (원본 V=49 는 1.04~1.13 it/s / 62223 MiB) → 3.7~4.1x
  빠르고 메모리 2.4x 적다. 약 52 min/epoch (원본 3.3 h/epoch). 병목이 frozen DINOv2 forward
  에서 dataloader 로 넘어갔다 (GPU util 26%).
- **`scripts/test/golden_dataset.py`** — 데이터셋 회귀 하네스. 22 개 arm(실제 run config 10 개 +
  플래그 합성 12 개)에 대해 `ds[i]` 를 4 샘플씩 뽑아 전 key 의 dtype/shape/sha1/min·max·mean 을
  json 으로 얼려 두고(`scripts/test/golden/dataset_golden.json`), 리팩토링 후
  `--mode check` 로 bit-identical 인지 본다. arm 은 `custom_{full,nodepth,rgbonly}`,
  `lagernvs_*` 4 종(geo latent cache HIT 경로 포함), `textonly_*`, `sd_whuman_custom`(SD 데이터셋),
  그리고 `geo_view_sampling` / `scale_mode` / `intr_norm` / `trans_repr` / `geo_shuffle_order` /
  `geo_swap_mode` / `geo_test_inseg_k` / `geo_posed` / `geo_return_idxs` 합성 arm.
  샘플은 앞 n 개가 아니라 인덱스 전체에 고르게 편다. arm 이 예외를 던지면 그 예외까지 golden 에
  기록해 "전에도 똑같이 실패했는가"를 본다.
  **감도 검증(negative control):** golden 의 `custom_full` 자리에 `syn_trans_c2w` payload 를
  넣었더니 `FAIL custom_full — 불일치 2 건` 이 뜨고 어긋난 key 로 `cam_param` 만 지목했다
  (exit 1). 하네스가 실제 변경을 잡는다는 것과, `trans_repr` 이 translation 채널에만 영향을
  준다는 코드 주석이 동시에 확인됐다.
- **`scripts/test/test_dataset_cfg.py`** — `resolve_dataset_cfg` 의 캐스케이드/검증 단위 테스트.
  golden 이 못 메우는 구멍을 메운다: golden 은 캐시가 **켜진** 경로만 (`geo_emb` 유무로) 보고,
  **꺼지는 네 조건**은 어느 arm 도 그 조합이 아니라 검증되지 않는다. 이게 조용히 깨지면 캐시가
  켜진 채 남아 context view 변경이 무시되고 증상은 "swap probe 결과가 안 변한다" 뿐이다(무에러).
  `resolve_dataset_cfg` 는 순수 함수(디스크 접근 없음)라 1 초 안에 돈다.
- **`scripts/viewer/README.md`** — viser 뷰어 두 개(`viser_val_cameras.py`, `viser_arms_gs.py`)의
  사용법. screen/포트 실행·종료 규약, DL3DV/SD 배경 처리, SD 의 umeyama sim3 → GT meters 식,
  `--pc-clips` 로 두 clip point cloud 를 겹쳐 aligned world 를 검증하는 법, 색 규약,
  frustum stride, 지표 4덩어리의 출처와 **CSV 의 PRDC/FCD 열은 sequence 별 값이 아니라
  set 단위 값의 복사본**이라는 함정, `--only` 거르기가 PRDC 계산 뒤에 와야 하는 이유,
  OpenGL↔OpenCV / `applied_transform` 좌표계 정리. 하드코딩된 함정 두 개를 명시:
  `--pc-frames` 는 target segment(49) 가 아니라 **clip 전체(81)** 에 균등 분포한다는 것과,
  static clip 은 sim3 의 `s`/`t` 가 없어 건너뛴다는 것.

- **`scripts/context_select/verify_geo_leakage.py`** — 끝난 run 의 **저장된 config 그대로**
  dataset 을 세워, 고른 context view 가 target segment `[s, e)` 안으로 들어갔는지 전수 확인한다.
  leakage-free 여부는 `geo_cover_out_of_seg` / `geo_first_view_target_s` /
  `geo_cover_before_only` 조합으로 결정되는데, 코드에 pool 이 비면 anchor(=frame s)로 떨어지는
  fallback (`dataset_dl3dv.py:173-174`, `:917` 의 pad pool) 이 있어 플래그만 읽고 단정할 수 없다.
  그래서 dataset 이 실제로 내보내는 `geo_idxs` 와 `[s, e)` 의 교집합을 센다. config 은
  `scripts/eval_testset.py::build_cfg` 를 재사용하고 (튜플 `(ns, dict)` 반환에 주의),
  강제로 켜는 건 선택 결과를 바꾸지 않는 `geo_return_idxs` 하나뿐이다.
  `20260808_140209_da3_7k_customgeo_nos` 결과: 겹침 0/500, `view0 == s` 0/500.

- **`scripts/bench/geo_encoder_speed.py`** — `geo_encoder: custom` vs `lagernvs` 의 **추론
  forward** 속도/메모리/토큰수/파라미터를 같은 조건(bf16 autocast, `train_latent_cam_dm.py:349`
  와 동일)에서 잰다. 해상도는 각 backend 가 실제로 받는 값(custom `custom_geo_input_hw`
  252x448, lagernvs `geo_image_hw` 256x448), view 수는 `--views` 로 준다 (DL3DV
  `geo_cover_k` 6 / SD `sd_geo_views: null` 49). `--trace` 는 custom 의 단계별 shape 을
  forward hook 으로 실측해 찍는다. GPU 경합 하에서 재면 median 이 부풀어서 min 을 같이
  보고한다. custom 은 `trainable=True` 라 학습 step 비용은 이 숫자로 외삽 금지.

- **`scripts/eval/prdc_diversity_diagnosis.py`** — "SD 는 GT 카메라 움직임 종류가 적어서
  kNN 반경 기반인 `clatr/precision`·`density` 가 낮게 나오는 것 아니냐"를 가르는 진단.
  PRDC precision 은 거리와 반경이 같이 스케일해서 **"GT 가 좁은 영역에 모여 있다"만으로는
  안 떨어진다** — 실제로 떨어뜨리는 건 GT 가 뭉쳐 `r_kNN` 이 GT 전체 퍼짐에 비해 작아지는
  경우다. 그래서 chunk 안의 GT 쌍거리 중앙값(`spread`)으로 전부 나눈 스케일 불변량
  (`tightness = median(r_kNN)/spread`, `dup_frac`, `err_pair`, `err_min`)과 반사실 precision
  두 개(`swap` = pred 를 같은 chunk 의 다른 GT 로 치환, `selfgt` = pred=GT)를 같이 낸다.
  `compute(num_splits=5)` 는 chunk 크기가 n 에 딸려 가므로 (n=160→32, n=640→128) `--chunk`
  로 고정해 코퍼스를 같은 조건에 놓는다. `--focus <data_name>` 은 sequence 하나의
  거리/반경/cos 를 뜯어본다.

- **`scripts/data/sd_preset_taxonomy.py`** — SD 의 clip 인덱스 `01`~`07` 이 실제로 몇 종류의
  카메라 움직임인지 `camera/<split>/<scene>/*_cam.json` 에서 직접 센다. 프레임 0 카메라
  좌표계로 옮기고 경로 길이로 정규화한 뒤, 같은 인덱스의 scene 간 RMSD / 인덱스 간 RMSD /
  clip 별 straightness / net 방향 일치도 / frame-0 대비 최대 회전각을 낸다.
  whuman 400 scene 실측: `01`,`02` 는 arc(직진성 0.931, 방향 일치도 0.98, 회전 75.7deg,
  z축 부호만 반대인 좌우 대칭쌍), `03`,`04` 는 **이동 0 + 제자리 74.53deg 회전**(p05=p95,
  좌우 한 쌍), `05`,`06`,`07` 은 **완전 직선 이동**(straightness 정확히 1.0)에 방향이
  scene 마다 무작위(일치도 0.29~0.33)이고 회전 median 약 19deg 인 서로 통계적으로
  구분되지 않는 3 개 표본. 즉 결정적 템플릿 4 개 + 무작위 직선 3 개다.

- **`avg_scale_ref` 옵션** (`main/conf/config.yaml`, `main/dataset_dl3dv.py`) — `scale_mode: avg_scale`
  + `pose_source: da3` 에서 저장된 avg_scale 을 **어느 기준점에서 잰 파일**로 읽을지 고른다.
  점 집합(context range unproject: target 제외 [0,s)/[e,N) 중 긴 쪽, da3 depth conf >= 전역 P40,
  pixel_stride 2)은 둘이 같고 기준점만 다르다. 둘 다 target 프레임을 안 써서 leakage-free.

  | 값 | 디렉토리 | 기준점 |
  |---|---|---|
  | `centroid` (기본, **기존 동작 그대로**) | `<scene>/da3/avg_scale/` | context 카메라 중심들의 centroid |
  | `context_first_cam` | `<scene>/da3/avg_scale_context_first_cam/` | context range 의 첫 카메라 |

  - `pose_source != 'da3'` 과 같이 쓰면 `ValueError` (해당 디렉토리가 `da3/` 아래에만 있다).
    알 수 없는 값도 `ValueError`.
  - `centroid` 이 아닐 때는 avg_scale json 결측/파싱 실패 시 `cam_dist_mean` 으로 **조용히
    fallback 하지 않고 예외를 던진다** — 분모가 말없이 바뀌는 사고를 막기 위해서다.
    기본값 `centroid` 의 fallback 동작은 그대로 유지.
  - 인덱스 캐시 키에는 안 들어간다 (avg_scale 은 `__getitem__` 에서 읽고 샘플 인덱스와 무관).
    실제로 두 설정이 같은 캐시(38609 samples / 6092 scenes)를 재사용하는 것을 확인.

  데이터 검증 (da3_7k train+test 39817 세그먼트): 결측 0, 비유한·비양수 0.
  first_cam mean 6.64851 / med 4.30108 / min 0.25072 / max 171.81422,
  first_cam/centroid 비 med 1.41159 (p05 1.00945, p95 2.23834, min 0.62235, max 6.09669,
  1 미만 3.87%). dataset 단위 검증: `cam_param` 의 rot6d/intrinsics/caption 은 완전 동일하고
  translation 만 정확히 그 비율만큼 스케일 (예 `1K_001dccbc…_0` centroid 9.496656 /
  first_cam 9.946007, trans 비 min=max=1.047317).

- **`main/conf/experiment/da3_7k_da3pose_asfirstcam.yaml`** — `da3_7k_da3pose.yaml` 에서
  `avg_scale_ref: context_first_cam` 한 줄만 추가한 arm. 나머지(모델/코퍼스/seg 리스트/blacklist/
  VAE/epochs/ckpt_at_epochs)는 전부 동일해 **차이가 분모의 기준점 하나뿐**이다.
  `vae_latent_scale` 은 0.96032625 유지 — 분모가 바뀌면 VAE latent std 도 바뀌므로 이 arm 에
  맞춰 측정한 값이 아니지만, `da3_7k_da3pose` 와 loss 를 같은 축에서 보기 위해 그대로 뒀다.

- **`scripts/eval/viz_clatr_latent_pca.py`** — PRDC 가 사는 공간(CLaTr trajectory latent, 256-D)의
  GT/pred 분포를 코퍼스별로 그린다. 코퍼스당 3 열: (1) GT+pred 합쳐 적합한 2-D PCA 산점도,
  (2) 같은 PCA 위 GT 만 카메라 프리셋별 색, (3) **PRDC 가 실제로 임계 비교하는 두 거리**의
  히스토그램 (GT 의 k=3 NN 반경 = 합격선 vs 각 pred 의 최근접 GT 거리).

  2-D PCA 는 SD 36.4%, DL3DV 29.1% 만 설명하므로 산점도의 근접성은 PRDC 판정과 다를 수 있다 —
  판정 근거는 3 열이다. 그래서 두 그림을 같이 낸다.

  | run | N | 2-D PCA 설명분산 | GT끼리 거리 med | k-NN 반경 med | radius/dist | pred→최근접GT med | ratio | 보고 precision / density |
  |---|---|---|---|---|---|---|---|---|
  | SD_textonly | 80 | 36.4% | 34.871 | 15.967 | 0.458 | 21.537 | **1.349** | 0.15 / 0.1792 |
  | DL3DV_nos | 160 | 29.1% | 41.410 | 25.258 | 0.610 | 23.250 | **0.921** | 0.95 / 1.1188 |

  그림에서 보이는 것: SD 는 GT 가 카메라 프리셋 5 종으로 **뚜렷하게 뭉쳐** 있어 k-NN 반경이
  좁고(15.97), pred 는 그 사이 빈 공간에 퍼진다 → 3 열에서 빨강 분포가 파랑보다 오른쪽.
  DL3DV 는 GT 가 넓게 퍼져 반경이 크고(25.26) pred 분포와 크게 겹친다.
  `frac(pred dist < median radius)`: SD 0.0375, DL3DV 0.70.

- **`scripts/eval/prdc_per_sample.py`** — set-level 로만 보고되던 `clatr/precision`, `clatr/density`
  를 **per-sample 로 분해**한다. `prdc.py` 의 정의가 fake 축 평균이라 분해가 성립한다:
  `precision = (d(real_j,fake_i) < radius_j).any(axis=0).mean()`,
  `density = (1/k)*(...).sum(axis=0).mean()` — axis=0 이 real 축이므로 fake 샘플별 값이 남는다.
  `manifold_k=3`, `num_splits=5` 의 chunk 경계까지 그대로 재현한다 (radius 가 chunk 안에서만
  정해지므로). recall/coverage 는 real 축이 남아 per-sample 값이 없어 내지 않는다.

  **재현 검증 (`sd_whuman_textonly`): precision 0.1500 = 보고 0.15, density 0.1792 = 보고 0.1792.**

  출력: `_prdc_per_sample.csv`, `_fscore_vs_prdc.png`, `_mismatch_topdown.png`
  (caption fscore >= 0.999 인데 manifold 밖인 target 을 density 순으로 top-down).

  **`sd_whuman_textonly` 결과 (n=80):** manifold 안 12/80, density>0 12/80.
  caption fscore >= 0.999 인 28 개 중 manifold 안은 **4 개뿐**.

  | 그룹 | n | pos_rmse | reach GT/pred | pathlen p/g | jitter GT/pred | CLaTr | cos(pred,GT) |
  |---|---|---|---|---|---|---|---|
  | 전체 | 80 | 1.163 | 2.62 / 2.17 | 0.96 | 0.510 / 0.363 | 48.6 | 59.0 |
  | precision=1 | 12 | 0.944 | 1.81 / 1.35 | 0.90 | 0.617 / 0.345 | 52.2 | 65.5 |
  | precision=0 | 68 | 1.201 | 2.77 / 2.31 | 0.97 | 0.491 / 0.366 | 48.0 | 57.9 |
  | fscore>=.999 & 안 | 4 | 0.607 | 1.51 / 0.85 | 0.65 | 0.480 / 0.361 | 50.2 | 75.7 |
  | fscore>=.999 & 밖 | 24 | 1.024 | 2.68 / 2.53 | 0.93 | 0.426 / 0.322 | 51.3 | 71.5 |

  jitter = `mean||2차차분|| / mean||1차차분||`. **manifold 밖 그룹이 특별히 떨리거나 짧지 않다** —
  오히려 `fscore>=.999 & 밖` 24 개는 cos(pred,GT) 71.5 로 전체 평균 59.0 보다 GT 에 가깝다.

  **PRDC 격차는 표본수 artifact 가 아니다 (가설 기각).** chunk 크기를 맞춰 재계산:

  | run | N | chunk=8 | chunk=16 | chunk=32 | chunk=N |
  |---|---|---|---|---|---|
  | SD_textonly | 80 | P 0.1750 D 0.1875 | P 0.1500 D 0.1792 | — | P 0.1625 D 0.0958 |
  | SD_customgeo | 80 | P 0.0875 D 0.1083 | P 0.1125 D 0.1250 | — | P 0.0625 D 0.0667 |
  | DL3DV_nos | 160 | — | P 0.9688 D 1.0688 | P 0.9500 D 1.1187 | P 0.9500 D 1.0688 |

  chunk=16 으로 맞춰도 0.15 vs 0.9688 이라 SD 테스트셋이 작아서 생긴 차이가 아니다.

  **실제 원인은 manifold 반경 대비 예측 거리다:**

  | run | real-real 거리 med | kNN radius(k=3) med | radius/dist | real-fake min 거리 med | min/radius |
  |---|---|---|---|---|---|
  | SD_textonly | 34.871 | 15.967 | 0.458 | 21.537 | **1.349 (> 1 → 밖)** |
  | DL3DV_nos | 41.410 | 25.258 | 0.610 | 23.250 | **0.920 (< 1 → 안)** |

  SD 는 real 궤적이 5 개 프리셋으로 뭉쳐 있어 kNN 반경이 좁고(radius/dist 0.458), 예측이 그
  좁은 공에 못 들어간다. 두 코퍼스 모두 **경계 바로 양옆**에 있어서 (21.5 vs 16.0, 23.3 vs 25.3)
  precision 0.15 대 0.95 라는 격차는 밑바탕 거리 차이(21.5 vs 23.3)를 크게 증폭한 값이다.
  **PRDC 절대값을 코퍼스 간에 비교하지 말 것.**

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
