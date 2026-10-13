# Changelog

저장소 **최상위** 문서·설정의 변경 기록. 하위 트리의 변경은 각자의 CHANGELOG 에 적는다 —
`camera_generation/dataset/CHANGELOG.md`, `camera_generation/latentcam/CHANGELOG.md`,
`video_generation/CHANGELOG.md`. 여기에 오는 것은 세 트리에 **걸쳐 있는 것**뿐이다
(`pipeline.md`, `README.md`, `.gitignore`, `CLAUDE.md`).

[Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

## [Unreleased]

### Fixed

- **VAE worldtraj 캐시 translation 분모를 scene scale 로** (R175). `train_camera_vae.py --wt_norm avg_scale` (기본) — 창별 mean‖R_w2c p + t_w2c‖ (scene.ply 점, Dataset-v1 transforms avg_scale · worldtraj avg_scale/{k}.json 과 같은 정의, 3창 일치 확인). 새 stage `wt_scale` (병렬, → cache/wt_avg_scale.json), cameras.json 목록은 cache/wt_files.json 에 저장해 재사용. R125~R170 은 창 카메라 이동거리 s / m1 로 나눴다 (`--wt_norm cam_disp` 로 재현).
- **VAE worldtraj 캐시의 cameras.json 해석 c2w → w2c** (R169). `train_camera_vae.py --stage cache_wt` 가 `rotation`/`position` 을 c2w 로 읽어 worldtraj 창(학습 228,130 · test 2,846)이 전부 역궤적이었다. 레포 규약(`main/dynamicverse_shim.py`, CamVLA `data/worldtraj`)과 scene.ply 투영 (images_pc_rendered 대비 색 오차 c2w 50~63 / w2c 18~21) 모두 w2c. `--wt_pose w2c|c2w` 신설, 기본 w2c · c2w 는 R125~R168 캐시 재현용.

### Added
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
