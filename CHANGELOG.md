# Changelog

저장소 **최상위** 문서·설정의 변경 기록. 하위 트리의 변경은 각자의 CHANGELOG 에 적는다 —
`camera_generation/dataset/CHANGELOG.md`, `camera_generation/latentcam/CHANGELOG.md`,
`video_generation/CHANGELOG.md`. 여기에 오는 것은 세 트리에 **걸쳐 있는 것**뿐이다
(`pipeline.md`, `README.md`, `.gitignore`, `CLAUDE.md`).

[Keep a Changelog](https://keepachangelog.com/) 규약을 따른다.

## [Unreleased]

### Changed

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
