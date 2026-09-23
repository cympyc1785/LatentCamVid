# latentcam — 학습과 평가

`pipeline.md` 가 **무엇이 학습되는가**(모델 구조·arm 축·모듈 지도)를 다룬다면, 이 문서는
**어떻게 돌리고 어떻게 읽는가**를 다룬다. 기동 한 줄, 손실이 무엇으로 이루어졌는지, wandb 에
찍히는 키가 각각 무슨 뜻인지, 그리고 조용히 틀리는 자리들.

모든 `파일:행` 은 2026-09-23 기준으로 직접 확인했다. 경로는 모두
`/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/` 기준 상대경로다.

| 자주 틀리는 것 | 실제 |
|---|---|
| "50 epoch 학습이니 `ckpt_at_epochs: [50, 100]` 이 걸린다" | 루프 마지막 인덱스가 **49** 라 한 번도 안 걸린다. 최종은 `last.pth` (`main/train_latent_cam_dm.py:1243`, `:1259`) |
| "wandb val 곡선으로 arm 을 비교한다" | val 은 **160 표본** 추정치다 (`val_max_batches 20 × batch_size 8`). arm 비교는 `scripts/eval_testset.py` 로 전량 |
| "`best.pth` 가 최종 모델" | 평가는 **`last.pth`**. `best.pth` 는 `val/loss_traj` 최저점일 뿐 (`:1199`) |
| "eval 은 현재 `conf/experiment/*.yaml` 을 읽으면 된다" | run 자신의 `results/<run>/config.yaml` 을 읽어야 한다 — 그 yaml 은 학습 이후에 바뀌었다 |
| "CLaTr 이 실패하면 학습이 죽는다" | 조용히 건너뛰고 계속 간다 (`:1150`, `:1155`) |
| "`PYTHONPATH` 는 있으나 마나" | 빼면 `ModuleNotFoundError: utils` 로 즉사한다 (`scripts/train/queue_runs.py:99-100`) |

---

## 0. 한 장 요약

```
 arm yaml 고르기          conf/experiment/<exp>.yaml  (158개, hydra defaults 상속)
        |
 smoke                   WANDB_MODE=disabled 로 1 epoch 만 확인
        |
 기동                     screen train1~4 , CUDA_VISIBLE_DEVICES ∈ {0,1,2,3}
        |                 cd main && PYTHONPATH=<repo>:. python -u train_latent_cam_dm.py experiment=<exp>
        |
 학습 루프                epoch 0 .. 49        (epoch_cap: 50)
        |   매 epoch 끝 ─┬─ run_validation()  → val/loss_latent, val/loss_traj
        |                ├─ CLaTr subprocess ×2 → val/clatr/*, val/captions/*
        |                └─ ckpts/{last,best,resume}.pth
        |
 산출                     results/<타임스탬프>_<exp_name>/
        |                   ├ config.yaml      ← eval 이 읽는 원본
        |                   ├ ckpts/{last,best,resume}.pth
        |                   ├ test/*.json      ← val 궤적 (pred / ref / text)
        |                   └ metrics.json     ← CLaTr subprocess 출력
        |
 전량 평가                scripts/eval_testset.py --run results/<run> --ckpt last.pth
                          → eval_my/<run>__last/{metrics.json, preds_scores.csv, ...}
```

| 기호 | 뜻 |
|---|---|
| **arm** | `conf/experiment/*.yaml` 한 장 = 실험 조건 하나 |
| **run** | 한 번의 학습 = `results/<YYYYmmdd_HHMMSS>_<exp_name>/` 하나 |
| **val** | 학습 중 매 epoch 도는 160 표본 검증 |
| **testset eval** | 학습이 끝난 뒤 held-out 전량(≈3.3k 세그먼트)에 대한 별도 실행 |
| **segment** | 학습 표본 하나 = 연속 `num_frames`(49) 프레임 + 캡션 |

---

## 1. 학습 기동

### 1.1 명령 한 줄

`scripts/train/queue_runs.py:99-101` 이 만드는 것이 정본이다.

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/latentcam/main && \
CUDA_VISIBLE_DEVICES=<gpu> \
PYTHONPATH=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam:. \
/data1/cympyc1785/miniconda3/envs/latentcam/bin/python -u \
  train_latent_cam_dm.py experiment=<exp> > <log> 2>&1
```

세 가지가 전부 필수다.

- **`cd main`** — hydra 가 `main/conf/` 를 config dir 로 잡고, `result_dir` 이
  `os.path.join('../results', exp_name)` (`main/train_latent_cam_dm.py:572`) 로 상대경로다.
- **`PYTHONPATH=<repo>:.`** — `<repo>` 가 없으면 `utils/` 패키지를 못 찾아 `ModuleNotFoundError:
  utils` 로 즉사한다. `queue_runs.py` 의 docstring 이 이 실수를 명시적으로 기록해 두었다.
  (같은 실수를 두 번 했다 — 메모리 `latentcam-train-launch-needs-pythonpath`.)
- **`python`** — conda env `latentcam` 의 인터프리터. `queue_runs.py:46-47` 에 `REPO` / `PY` 로
  박혀 있다.

`experiment=<exp>` 는 `main/conf/experiment/<exp>.yaml` 의 파일명(확장자 없이)이다.

### 1.2 GPU

**`CUDA_VISIBLE_DEVICES` 는 0~3 만.** (`CLAUDE.md` 의 `## Don't`.) 여유 있는 것 하나만 쓴다 —
이 학습은 single-GPU 로 돈다 (`Accelerator` 는 쓰지만 보통 프로세스 하나).

### 1.3 screen 규약

학습은 `screen train1~4`, 추론은 `screen infer1~4`. 어느 screen 에 무엇이 도는지는
`.claude/watch.md` 에 **적는다** — 기억에 의존하면 컨텍스트가 압축될 때 사라진다.
로그는 `<repo>/tmp/` 아래. (`/tmp` 는 시스템이 비워서 모니터가 조용히 무력화된다.)

### 1.4 대기열 — `scripts/train/queue_runs.py`

GPU 가 빌 때까지 기다렸다가 순서대로 띄운다.

```bash
python scripts/train/queue_runs.py --jobs <queue.json>
```

| 인자 | 기본 | 뜻 |
|---|---|---|
| `--jobs` | (필수) | 대기열 JSON |
| `--free_mib` | 2000 | 이 아래면 "빈 GPU" 후보 |
| `--poll` | 120 | 폴링 주기(초) |
| `--settle` | 180 | 기동 후 다음 job 을 보기 전 대기(초) |
| `--repo` / `--py` / `--state` | 위 상수 | 경로 재정의 |
| `--dry_run` | off | 띄우지 않고 판정만 |

**빈 GPU 판정은 두 조건의 AND** 다. 하나만 보면 반드시 이중 배정이 난다.

1. `nvidia-smi` 의 `memory.used` 가 `--free_mib` 아래 — Ctrl+C 직후 DataLoader worker 가
   몇 초 동안 GPU 를 물고 있기 때문에 필요하다.
2. 어떤 학습 프로세스도 그 GPU 를 선점하지 않았다 — `/proc/<pid>/environ` 에서
   `CUDA_VISIBLE_DEVICES` 를 읽어 확인한다. molmo2 계열 arm 은 4.5 GB 텍스트 캐시를 로드하는
   동안 **수 분간 0 MiB** 에 앉아 있어서 ①만으로는 "비었다" 로 보인다.
   (2026-09-15 `--dry_run` 에서 실제로 GPU2/3 이 이중 배정되는 것을 관측했다.)

### 1.5 중단과 재개

- **중단은 Ctrl+C 먼저.** `screen -X stuff $'\003'` → 최대 180초 대기 → 안 죽으면 SIGTERM →
  SIGKILL. 그냥 `kill` 하면 DataLoader worker 가 고아가 되어 GPU 를 계속 물고 있다.
  (`scripts/stop_at_epoch.sh` 의 docstring 이 이 절차를 정본으로 갖고 있다.)
- **pkill 패턴은 반드시 `$` 로 끝을 고정한다.** `experiment=geo_worldtraj` 는
  `geo_worldtraj_camembed` / `_ctxlonger135` / `_lagernvsnorm` / `_decoupled` 의 접두사라,
  anchor 없이 죽이면 하나 멈추려다 다섯을 죽인다 (`scripts/stop_at_epoch.sh:37-40`).
- **재개는 `load_ckpt_path` 에 `resume.pth`.** `main/train_latent_cam_dm.py:557-562` 가
  체크포인트를 미리 열어 보고, `model` 키가 있는 full dict 면 optimizer·global_step·epoch·
  best_val 까지 복원하고 **같은 `exp_name`·같은 wandb run 으로 이어 붙인다** (`:565-566`, `:581`).
  순수 state_dict(옛 `last.pth`) 를 주면 weights-only warm start 가 된다.

### 1.6 smoke

**wandb 를 끄고 돌린다** — `WANDB_MODE=disabled`. smoke run 이 wandb 목록을 어지럽힌다.
확인할 것은 셋이다: ① 인덱스 크기(=데이터셋이 기대한 세그먼트 수를 잡았는가)
② 첫 epoch 의 loss 가 NaN 이 아닌가 ③ `run_validation` 이 끝까지 가는가(= CLaTr subprocess 까지).

`scripts/smoke_before_only.py` 는 이 중 ①을 인코더 없이 확인하는 예다 — hydra 로 cfg 만
compose 하고 `build_dataset(cfg)` 를 불러서 인덱스 크기와 view 선택을 찍는다. 새 축을 넣을 때
같은 모양으로 짧게 확인하는 것이 학습을 통째로 돌려 보는 것보다 싸다.

---

## 2. 학습 루프와 손실

### 2.1 세 갈래

`main/train_latent_cam_dm.py:1383-1400` 에서 갈린다.

| 갈래 | 조건 | 손실 |
|---|---|---|
| **표준 확산** (기본) | 둘 다 false | `F.mse_loss(noise_pred, noise)` (`:1400`) |
| **AR** | `is_ar: true` | `ar_train_loss(...)` (`:1383` → `:513`) |
| **Diffusion-Forcing** | `per_token_noise: true` | `per_token_flow_loss(...)` (`:1387` → `:479`), rectified flow |

AR / per-token 은 **track·video 조건과 상호 배타**다 — `:682-683`, `:695-696` 에서 assert 로
막는다.

### 2.2 보조 손실 셋

전부 기본 가중치 0 이고, **0 이면 함수를 호출조차 안 해서 loss 가 비트 단위로 예전과 같다.**

| 항목 | 가중치 키 (기본) | 더해지는 곳 | 원 손실 |
|---|---|---|---|
| 시작 pose 회귀 | `start_pose_w: 0.0` (`main/conf/config.yaml:630`) | `:1405` | `start_pose_loss` (`:307`) |
| readout 보조 (OBB) | `peav_readout_aux_w: 0.0` (`:601`) | `:1410` | `readout_aux_loss` (`:282`) |
| aim (조준 각) | `aim_loss_w: 0.0` (`:613`) | `:1419` | `aim_loss` (`:344`), `1-cos` |

셋 다 "켰는데 가중치 0" 을 assert 로 막는다 (`:727`, `:756`). `aim_loss_w>0` 은
`prediction_type='epsilon'` 전용이고(`:732`) latent arm 전용이다(`:736`) — x0 복원식과
VAE decode 가 전제라서.

### 2.3 epoch 상한

```
:1242   _cap    = getattr(cfg, 'epoch_cap', None)
:1243   _epochs = cfg.epochs if _cap is None else min(cfg.epochs, int(_cap))
:1259   for epoch in range(start_epoch, _epochs):
```

`conf/config.yaml:18 epochs: 100` 을 `:26 epoch_cap: 50` 이 **다시 자른다.** experiment yaml
73개가 `epochs` 를 덮고 있고 그 중 19개가 200 이라, 기본값만 낮추면 재실행 때 조용히 200 을
간다 — 그래서 상한을 하나 더 뒀다. `epoch_cap: null` 이면 상한 없음(옛 동작).

**마지막 epoch 인덱스는 49.** 그래서 옛 arm 들의 `ckpt_at_epochs: [50, 100]` 은 이제
한 번도 안 걸리고, 최종 모델은 `last.pth` / `resume.pth` 다.

### 2.4 `step_timing`

`conf/config.yaml:30 step_timing: 0` — 0 이면 측정 코드가 아예 안 돈다. N>0 이면 N 스텝마다
`data/h2d/text_vae/geo/fwd/bwd_opt/log` 평균 ms 를 찍는데, 구간 경계마다
`cuda.synchronize()` 가 걸려 파이프라인이 직렬화된다. **상시 금지, 진단 전용.**

---

## 3. 평가

### 3.1 두 층위 — 이걸 섞으면 안 된다

| | 학습 중 val | testset eval |
|---|---|---|
| 언제 | 매 epoch 끝 (`:1500`) | 학습이 끝난 뒤 수동 |
| 표본 | **160** (`val_max_batches 20 × batch_size 8`, `:927`) | held-out **전량** (≈3.3k) |
| 코드 | `run_validation` (`:905`) | `scripts/eval_testset.py` |
| 쓰임 | **곡선 감시** — 발산·정체·NaN | **arm 비교** |
| 출력 | wandb | `eval_my/<tag>/` |

**160 표본 추정치의 epoch 간 요동이 arm 간 격차보다 크다.** 이것이 `eval_testset.py` 가
존재하는 이유이고, docstring 첫 문단에 그대로 적혀 있다. wandb val 곡선으로 "A 가 B 보다 낫다"
를 말하면 안 된다.

### 3.2 학습 중 val 이 하는 일

`run_validation(epoch, global_step)` (`:905`) 는 매 epoch 끝에 (`:1500`):

1. `sample()` 로 궤적을 뽑고 `val/loss_latent`(`:1164`), `val/loss_traj`(`:1165`) 를 잰다.
   - `val_loss_latent` = latent 공간 MSE (`:1055`)
   - `val_loss_traj` = **VAE decode 후** 궤적 공간 MSE (`:1062`)
2. 각 표본을 `results/<run>/test/` 아래 세 JSON 으로 쓴다 — `*_transforms_pred.json`,
   `*_transforms_ref.json`, 텍스트. nerfstudio 스타일 (`w/h/fl_x/fl_y/cx/cy` + `frames`).
3. `results/<run>/test_valid.txt` 에 유효 표본 목록을 쓴다 (`:1127`).
4. CLaTr subprocess 두 개를 돌린다 (§3.4).
5. 체크포인트를 저장한다 (§4).
6. `model.train()` 으로 복귀한다.

`start_pose_pred` arm 은 `val/start_mse` · `val/start_trans_u` · `val/start_rot_deg` 를 더
찍는다 (`:1172-1174`). head 가 없으면 `cnt=0` 이라 아예 안 찍힌다.
이때 읽는 것은 `sample()` 의 **마지막 denoise step** 이 남긴 `start_pred` 다 — 그 forward 는
GT 를 안 받으므로(teacher forcing 은 `self.training` 게이트) 추론과 같은 조건이다.

### 3.3 wandb 키 사전

**프로젝트: `camera-diffusion-training`** (`:583`), run 이름 = `<YYYYmmdd_HHMMSS>_<exp_name>`
(`:568`) = `results/` 디렉토리 이름과 같다.

| 키 | 행 | 뜻 |
|---|---|---|
| `train/loss` | `:1429` | 스텝 손실 (보조 손실 **포함**한 최종값) |
| `train/epoch_loss` | `:1490` | epoch 평균 |
| `train/epoch` | `:1494` | `epoch + 1` |
| `train/readout_aux_obb` | `:1431` | 가중치 곱하기 **전** 원 손실 |
| `train/aim_loss` | `:1436` | `1-cos`, 가중치 전 |
| `train/aim_deg` | `:1437` | 같은 마스크의 평균 조준 각(도) |
| `train/aim_n` | `:1438` | 그 스텝에 실제로 들어간 (표본×프레임) 수. **0 근처면 게이트(look_at / t<max_t / valid)가 배치를 통째로 걸렀다는 뜻** |
| `train/start_mse` | `:1442` | 가중치 전 원 손실 |
| `train/start_trans_u` / `train/start_rot_deg` | `:1443-1444` | 읽기용 분해 — **mse 는 내려가는데 rot 이 안 내려가면 이동만 맞히고 있는 것** |
| `val/loss_latent` | `:1164` | latent MSE |
| `val/loss_traj` | `:1165` | 궤적 MSE. **`best.pth` 의 기준** |
| `val/start_*` | `:1172-1174` | start_pose arm 전용 |
| `val/geo/attn_map` | `:1005` | geo cross-attn 시각화 (이미지) |
| `val/clatr/*` · `val/captions/*` | `:1153` | `metrics.json` 을 그대로 올린 것 (§3.4) |

### 3.4 CLaTr / caption 지표

`:1136-1145` 에서 **subprocess 두 개**를 순서대로 돌린다. `_HAS_CLIP` 이 false 이거나
per-token arm 이면 건너뛴다 (per-token val 은 transforms 를 안 남긴다).

```python
# 1) 궤적 → CLaTr 임베딩          cwd = evaluate/CLaTr
[sys.executable, '-m', 'src.extraction',
 f'checkpoint_path={cfg.clatr_ckpt_path}', f'data_dir={result_dir}']
 # + cfg.clatr_standardization 가 있으면 'dataset/standardization=...' 를 덧붙임

# 2) 임베딩 → 지표                cwd = evaluate/eval
[sys.executable, '-m', 'src.eval_only',
 '--pred_path', f'{result_dir}/preds.npy']
```

결과는 `results/<run>/metrics.json`. **없으면 경고만 찍고 학습은 계속 간다** (`:1150`, `:1155`)
— `(CLaTr) metrics.json not produced; skipping CLaTr log this eval`. 곡선에 CLaTr 이 안 보이면
이 줄을 로그에서 먼저 찾을 것.

`evaluate/eval/src/eval_only.py` 의 `MetricCallback` 이 네 모듈을 굴린다:

| 모듈 | 나오는 키 | 무엇 |
|---|---|---|
| `CaptionMetrics` | `captions/{precision,recall,fscore}` | 궤적을 태거로 라벨링해 pred↔ref 태그 일치 |
| `CLaTrScore` | `clatr/clatr_score` | pred 임베딩 ↔ **텍스트** 임베딩 정합 |
| `ManifoldMetrics`(PRDC, euclidean) | `clatr/{precision,recall,density,coverage}` | ref 집합 대비 pred 집합의 분포 |
| `FrechetCLaTrDistance` | `clatr/fcd` | ref↔pred Fréchet 거리 |

실제 `metrics.json` 한 예 (`results/20260921_103748_dynpose_d200_molmo2_decfuse_l21_da3/`):

```json
{"val/captions/precision": 0.036, "val/captions/recall": 0.013, "val/captions/fscore": 0.0191,
 "val/clatr/clatr_score": 6.2179, "val/clatr/precision": 0.4375, "val/clatr/recall": 0.75,
 "val/clatr/density": 0.2917, "val/clatr/coverage": 0.5625, "val/clatr/fcd": 5441.7598}
```

**읽는 법의 함정 넷** (전부 실제로 틀렸던 것):

- **PRDC 는 코퍼스 간 비교 금지.** 반경 임계 양옆에 앉으면 표본 수 차이가 증폭된다.
- **caption fscore 는 chance 대비로** 읽는다. val 구성이 "1씬 × 160변이" 냐 "160씬 × 1변이" 냐에
  따라 같은 절대값이 다른 눈금이다.
- **caption F1 은 CLaTr 재구성 위에서 잰다** — raw JSON 재계산과 안 맞는 건 게이지가 아니라
  VAE 재구성 오차 탓이다(신호의 72%).
- **`clatr_ckpt_path` 를 바꾸면 그 run 의 FD/PRDC/clatr_score 는 이전 run 과 비교 불가**다.
  `conf/config.yaml:736-741` 에 그대로 적혀 있다. 49프레임 코퍼스 ckpt 로 바꾸면 padding 길이도
  같이 바꿔야 해서 `:747 clatr_standardization` 을 함께 건드려야 한다(기본 `null` = 옛 동작,
  `0120` / `num_cams 120`).

### 3.5 전량 평가 — `scripts/eval_testset.py`

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH="main:.:data" python scripts/eval_testset.py \
    --run results/<run> --ckpt last.pth --out eval_my/<tag>
```

**핵심은 "run 자신의 config 를 읽는다" 는 것**이다. 학습이 남긴
`results/<run>/config.yaml` (`:594` 에서 dump) 을 현재 기본값 **위에** 얹는다. 이게 없으면
조용히 다른 모델을 평가한다 — 실제 사고: geo run 셋은 `vae_latent_scale 0.96032625` 로 학습했는데
`conf/experiment/geo_worldtraj.yaml` 은 그 뒤 그 줄을 잃었고, 살아있는 yaml 로 평가하면
`0.47637` 이 들어간다.

split 도 `base.Trainer._make_batch_generator()` 를 **그대로** 태워서(seed 42) 학습 때와 같은
held-out 을 본다.

| 인자 | 기본 | 뜻 |
|---|---|---|
| `--run` | (필수) | `results/<run>` |
| `--ckpt` | `best.pth` | **`last.pth` 를 명시할 것** (§4) |
| `--gpu` | | 0~3 |
| `--out` | `eval_my/<run>__<ckpt>` | 출력 |
| `--batch-size` / `--max-batches` | run config | 덮어쓰기 |
| `--set KEY=VALUE` | (반복) | cfg 덮어쓰기. seg_list 못박기에 씀 |
| `--skip-clatr` | off | CLaTr subprocess 생략 |
| `--sample-seed` | | 전 arm 을 같은 x_T 로 (paired 비교의 전제) |
| `--drop-track` / `--drop-video` | off | 해당 조건을 빼고 ablation |
| `--probe-video-ca` / `--probe-timestep` | / 500 | video cross-attn 프로브 |
| `--timing` / `--timing-warmup` | / 1 | 속도 측정 |

여러 (run, ckpt) 조합은 `scripts/eval_testset_queue.sh` 로 **순차** 실행한다 —
GPU 하나를 쓰므로 병렬은 OOM/경합만 난다.

```bash
GPU=3 scripts/eval_testset_queue.sh results/<run>:last.pth results/<run2>:last.pth
# seg_list 를 못박아 다시 돌릴 때는 EXTRA + TAG_SUFFIX (기존 eval_my/* 를 안 덮게)
EXTRA="--set train_seg_list=... --set test_seg_list=..." TAG_SUFFIX=__c4d2k5y4 \
GPU=0 scripts/eval_testset_queue.sh results/<run>:last.pth
```

> **학습 중에는 eval 을 하나씩만.** `/data1` 이 Lustre 라 동시 eval 이 학습 프로세스를
> D-state 로 물어 버린다. 그때의 일시적 지연을 "학습이 느려졌다" 로 외삽하면 안 된다.

### 3.6 arm 간 차이가 잡음인지 — `scripts/eval_paired_bootstrap.py`

전 arm 이 `--sample-seed 42` 로 **같은 세그먼트 · 같은 x_T** 를 썼으면(common random numbers)
차이가 세그먼트 단위로 짝지어진다. 세그먼트를 리샘플해 평균차의 95% CI 를 낸다.

**단, per-sample 인 지표만 가능하다.** `preds_scores.csv` 에서 행마다 다른 값은
`captions/{precision,recall,fscore}` · `clatr/{clatr_score,pred_ref_cosine}` 뿐이다.
`clatr/{precision,recall,density,coverage,fcd}` 는 **집합 단위 지표라 전 행에 전역 값이
복제**돼 있어 여기서 CI 를 못 낸다 — 그건 점추정만 보고한다.

### 3.7 이 저장소 밖에 있는 평가

`subject_in_frame` / `collision_rate` 는 latentcam 이 아니라 **dataset 쪽**에 있다:
`camera_generation/dataset/eval/eval_subject_in_frame.py`,
`camera_generation/dataset/eval/eval_collision_rate.py`.
카메라가 피사체를 프레임 안에 두는지 / 기하와 부딪히는지는 코퍼스 기하를 필요로 해서
학습 루프에 붙어 있지 않다. (wandb 로 올리는 건 아직 안 했다.)

`scripts/eval/` 의 14개는 학습 후 분석 도구다 — `corpus_axis_compare.py`,
`traj_diversity.py`, `prdc_diversity_diagnosis.py`, `mean_traj_baseline.py`,
`pose_source_agreement.py` 등. 같은 질문을 두 번째로 던지게 되면 **먼저 여기를 grep 한다.**

---

## 4. 체크포인트

`run_validation` 끝에서 매 epoch (`:1188` 이하, main process 만):

| 파일 | 언제 | 내용 |
|---|---|---|
| `ckpts/last.pth` | **항상** (`:1190`) | `CameraDiffusionModel` state_dict 하나 |
| `ckpts/best.pth` | `val_loss_traj_mean < best_val` 일 때 (`:1199-1201`) | 같은 형식 |
| `ckpts/epoch<N>.pth` | `ckpt_at_epochs` / `ckpt_every_epochs` (`:1211-1221`) | 같은 형식 |
| `ckpts/resume.pth` | 항상 (`:1223`) | `{model, geo, opt, opt_param_names, global_step, epoch, best_val, exp_name, wandb_id}` |
| `ckpts/*_geo.pth` | `geo_encoder: custom` 일 때만 (`:1195-1198`) | 학습된 geo 가중치 |

`last/best/epoch<N>.pth` 는 **"state_dict 하나"라는 계약**을 지킨다 — `eval_testset.py` 등
기존 소비자가 그대로 동작해야 하므로, 학습 가능한 geo 백엔드의 가중치는 옆 파일로 뺐다.
frozen 백엔드에서는 `*_geo.pth` 가 아예 안 생겨서 옛 run 디렉토리 구조와 동일하다.

> **평가는 `last.pth` 로 한다.** `best.pth` 평가 금지 — 전 지표에서 `last` 가 우세했다.
> `best` 는 160 표본짜리 `val/loss_traj` 의 최저점일 뿐이라 사실상 그 잡음을 고른 것이다.

---

## 5. 주의사항

| # | 함정 | 증상 | 대응 |
|---|---|---|---|
| 1 | `PYTHONPATH` 누락 | `ModuleNotFoundError: utils` 즉사 | `cd main && PYTHONPATH=<repo>:.` |
| 2 | GPU 4~7 사용 | — | `CUDA_VISIBLE_DEVICES` 는 0~3 만 |
| 3 | queue 가 GPU 이중 배정 | molmo2 arm 이 0 MiB 로 "비어" 보임 | `queue_runs.py` 의 AND 판정을 우회하지 말 것 |
| 4 | `kill` 로 학습 종료 | DataLoader worker 가 GPU 를 물고 남음 | Ctrl+C 먼저, 그 다음 TERM/KILL |
| 5 | pkill 패턴에 `$` 누락 | 하나 멈추려다 다섯을 죽임 | `experiment=<exp>$` |
| 6 | wandb val 로 arm 비교 | 160 표본 잡음을 신호로 읽음 | `eval_testset.py` 전량 |
| 7 | `best.pth` 평가 | `last` 보다 나쁨 | `--ckpt last.pth` |
| 8 | 살아있는 yaml 로 eval | `vae_latent_scale` 이 조용히 달라짐 | run 의 `config.yaml` 을 읽는 `eval_testset.py` 를 쓸 것 |
| 9 | split 재분할 | 옛 CLaTr ckpt 가 새 test 를 이미 봤다 → 지표가 부풀음 | CLaTr 게이지는 train split 을 따라간다 |
| 10 | `clatr_ckpt_path` 변경 후 이전 run 과 비교 | FD/PRDC/clatr_score 가 다른 눈금 | `clatr_standardization` 과 함께 바꾸고, 대조군도 다시 잼 |
| 11 | CLaTr 결측을 못 알아챔 | 곡선에 `val/clatr/*` 가 그냥 없음 | 로그에서 `metrics.json not produced` 를 찾을 것 |
| 12 | `step_timing` 켠 채 본 학습 | `cuda.synchronize()` 로 느려짐 | 진단 후 0 으로 되돌릴 것 |
| 13 | 학습 중 eval 여러 개 | Lustre 경합으로 학습이 D-state | 한 번에 하나 |
| 14 | 실험 결과 임의 요약 | — | wandb 원본 수치 그대로 |
| 15 | 모니터를 걸고 안 지움 | 죽은 로그를 보는 감시자가 쌓임 | `.claude/watch.md` 에 pid 포함해 적고, 끝나면 그 행을 지운다 |

### 모니터를 걸 때 — "정상" 은 셋이다

`Traceback|OOM` 만 걸면 **죽는 것만** 잡고 조용히 **멈춘 것은 못 잡는다.**

1. **살아있나** — pid 존재 + GPU 점유
2. **나아가나** — 로그의 `^Epoch N | Loss ...` 의 N 이 증가, wandb step 이 증가
3. **끝났나** — epoch 49 도달, `ckpts/last.pth` mtime

경보는 **이상일 때만** — 주기적인 epoch/best 보고는 하지 않는다.

---

관련 문서: [`pipeline.md`](pipeline.md) (모델 구조·arm 축·모듈 지도) ·
`docs/corpus_and_model_axes.md` (어떤 arm 이 무엇을 시험했는지) ·
`docs/known_issues.md` · `docs/geo_context_ablations.md` · `docs/vae_verification.md`
