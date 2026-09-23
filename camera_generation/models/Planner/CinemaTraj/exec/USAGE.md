# `exec/` 사용법

`exec/` 는 CinemaTraj 의 **드라이버·기동 스크립트 모음**이다. 알고리즘은 `fit/`, `lbm/`,
`eval/`, `scripts/` 에 있고, 여기 있는 파일들은 그것들을 **어떤 순서로 · 어떤 인자로 · 몇 샤드로**
부를지만 정한다. 세대(dNNN)마다 달라지는 것은 스크립트가 아니라 **config JSON 과 argparse 인자**다.

## 하드 규칙 — CinemaTraj 루트에서 실행한다

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
python exec/<드라이버>.py ...
```

이유는 스크립트들이 **자기 파일 경로로부터 CinemaTraj 루트를 역산**하기 때문이다
(`run_bank.py:61` `HERE = dirname(dirname(abspath(__file__)))`, `run_corpus_export.py:29` `CINE`,
`chain_bank_rounds.py:52` / `rebake_scenes.py:38` `ROOT`, `run_gendop_eval.py:55` `HERE`).
그 루트를 `cwd` 로 삼아 하위 단계를 `fit/bank/...` · `eval/...` 같은 **상대 경로**로 부르고,
`--config configs/bank/<gen>.json` · `--cloud_root out_dynpose` 같은 인자도 루트 기준 상대 경로다.
다른 디렉토리에서 부르면 드라이버는 뜨지만 단계 스크립트를 못 찾아 죽거나, 더 나쁘게는
상대 경로 산출물이 엉뚱한 곳에 쌓인다.

인터프리터는 conda env 절대경로를 쓴다. 드라이버 자신을 어느 env 로 띄워야 하는지는 파일마다
다르므로 각 절의 "주요 인자" 위 설명을 볼 것.

| env | python |
|---|---|
| latentcam | `/data1/cympyc1785/miniconda3/envs/latentcam/bin/python` |
| vista4d | `/data1/cympyc1785/miniconda3/envs/vista4d/bin/python` |
| GenDoP | `/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python` |
| vllm | `/data1/cympyc1785/miniconda3/envs/vllm/bin/python` |

GPU 는 **0~3 만** 쓴다. `--gpu` / `--cuda` / `CUDA_VISIBLE_DEVICES` 를 주는 자리마다 이 범위를
지킨다. 임시 파일은 `/tmp` 가 아니라 `/data1/cympyc1785/LatentCamVid/tmp/<작업>/` 아래에 둔다.

## 파일 한눈에

| 파일 | 역할 |
|---|---|
| `run_bank.py` | 뱅크 체인(graph→cloud→route→tau→fit→emit) 단일 드라이버. 세대 차이는 `configs/bank/<gen>.json` 으로 표현한다 |
| `chain_bank_rounds.py` | graph 가 도는 동안 끝난 편부터 주워 강등→뱅크까지 라운드 루프로 굽는다 (CPU/GPU 겹치기) |
| `rebake_scenes.py` | 지정한 씬만 마커를 무시하고 graph/뱅크 재굽기. 기존 뱅크는 지우지 않고 격리한다 |
| `run_corpus_export.py` | 뱅크 → latentcam 코퍼스 (desc→merge_desc→captions→export→verify) |
| `run_custom_caption.py` | 영상 1개 + 캡션 1줄 → 카메라 + depth warp 릴. SAM3/VLM/뱅크 없는 추론 전용 최소 경로 |
| `run_vista4d_gen.py` | 영상 + **이미 있는 카메라** → Vista4D 생성 영상 + `source \| gen` 릴 |
| `run_gendop_eval.py` | GenDoP 릴리즈 ckpt 를 우리 코퍼스 split 에 돌려 우리 arm 과 한 표에 올린다 |
| `run_director_vista.py` | DIRECTOR(E.T.) 를 씬 하나에 돌리는 파일럿. 좌표·스케일 변환 함수의 원본 |
| `run_director_batch.py` | 위 함수를 재사용해 split 전량을 배치로. GenDoP 어댑터가 먹는 npz 를 낸다 |
| `run_board_sweep.py` | TRUMANS `Recordings_blend/` 전량에 first-pose board 를 돌리는 Blender 스윕 |
| `serve_qwen3vl.sh` | 로컬 Qwen3-VL(vLLM) 서버 기동. VLM 단계 직전에 띄우고 끝나면 내린다 |

---

## `run_bank.py`

### 무엇을 하는가

뱅크 체인 여섯 단계(`graph` / `cloud` / `route` / `tau` / `fit` / `emit`)를 영상 목록에 대해
순서대로 통과시킨다. **뱅크 정체성을 이루는 플래그 전량은 `configs/bank/<gen>.json` 한 곳에** 적고,
드라이버는 `--video` / `--output_root` / `--bank_dir` / `--eval_data` 만 자기가 붙인다.
새 세대는 새 스크립트가 아니라 **config 한 장**이다 (`"extends"` 로 부모 config 를 최상위 키 단위
얕은 병합할 수 있다). env 는 `vista4d` — 단계 스크립트가 torch/cuda 를 쓰고, 드라이버는
`sys.executable` 을 그대로 물려준다.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python exec/run_bank.py \
    --config configs/bank/d185_dynpose100k_grid5.json \
    --videos /data1/cympyc1785/LatentCamVid/tmp/d185/scenes.txt \
    --log_dir /data1/cympyc1785/LatentCamVid/tmp/d185/bank_scene_logs \
    --gpu 2 --num_shards 4 --shard_id 0
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--config` | (필수) | `configs/bank/<gen>.json` |
| `--videos` | (필수) | 목록 파일 경로 또는 콤마 구분 이름 |
| `--log_dir` | (필수) | 영상·단계별 로그 (`<video>.<stage>.log`) |
| `--gpu` | `None` | `CUDA_VISIBLE_DEVICES` 값. **0~3 만** |
| `--stages` | config 의 `stages` | 콤마 구분. `graph,cloud,route,tau,fit,emit` 중 |
| `--num_shards` / `--shard_id` | `1` / `0` | `i % num_shards == shard_id` 로 나눈다 |
| `--skip_done` / `--no_skip_done` | `True` | 산출물이 있으면 건너뛴다 |
| `--threads` | `8` | 단계의 BLAS/OpenMP 스레드 상한. `0` = 캡 없음 |
| `--exec` | `subprocess` | `subprocess` \| `inproc`. inproc 은 import 가 샤드당 1회 |
| `--inproc_recycle` | `50` | inproc 모드에서 N편마다 `execv` 로 자기 재실행 (누수 방지) |

### 비고

- **GPU 게이트는 `BANK_GPU_ALLOW` 환경변수**다 (`run_bank.py:448`). `--gpu` 를 줬을 때만 검사하고,
  `allow = environ.get("BANK_GPU_ALLOW", "01234")` 를 문자 리스트로 만들어
  `assert str(args.gpu) in list(allow)` 한다. 즉 **기본값은 0~4 를 통과시킨다** — 현행 0~3 정책보다
  느슨하므로 4 를 실수로 넣어도 안 막힌다. 엄격히 가두려면 `BANK_GPU_ALLOW=0123` 을 함께 준다.
  문자 단위 비교라 두 자리 GPU 번호는 원리상 못 쓴다.
- `--gpu` 를 **안 주고** 밖에서 `CUDA_VISIBLE_DEVICES` 를 export 하면 게이트가 통째로 우회되고,
  로그에도 `gpu=None` 으로 찍혀 어느 카드를 썼는지 사후에 알 수 없다. 반드시 `--gpu` 로 준다.
- **실행 방식은 config 가 CLI 를 이긴다.** config 최상위에 `"exec"` 키가 있으면 `--exec` 를 덮는다
  (`chain_bank_rounds` 가 `--exec` 를 늘 명시로 넘기기 때문에 뒤집어 둔 우선순위다).
- config 의 `args` 에 `--video` / `--output_root` / `--bank_dir` / `--eval_data` 를 쓰지 말 것.
  중복되면 argparse 가 뒤엣것을 이겨 config 만 봐서는 실제 값을 모르게 된다.
- 게이트 판정으로 끝난 씬은 `FAIL` 이 아니라 `SKIP` + `skipped.json` 으로 닫는다
  (route rc=3 앵커 0 / 빈 `dynamic_mask` / emit 변이 0). 안 닫으면 라운드마다 같은 씬을 재탕한다.

---

## `chain_bank_rounds.py`

### 무엇을 하는가

`graph` 단계는 순수 CPU 라 그것만 돌리면 GPU 가 논다. 이 드라이버는 **graph 가 도는 동안**
`<graph_marker>` 가 새로 생긴 편을 라운드마다 스캔해서, 정지 소품 강등 →
`run_bank.py --stages route,tau,fit,emit` 까지 굽는다. graph 드라이버가 죽고 ready 가 0 이면 끝낸다.
env 는 `vista4d`.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python exec/chain_bank_rounds.py \
    --config configs/bank/d185_dynpose100k_grid5.json \
    --videos /data1/cympyc1785/LatentCamVid/tmp/d172/videos_10346.txt \
    --work /data1/cympyc1785/LatentCamVid/tmp/d185 \
    --graph_marker .graph_d182 --bank_dir hole_bank_d185 \
    --graph_pattern "run_bank.py --config configs/bank/d182_" \
    --gpus 0,1,2,3 --bank_shards 4
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--config` / `--videos` / `--work` | (필수) | 뱅크 config · 목록 파일 · 로그를 쌓을 tmp 폴더 |
| `--eval_data` | `/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K` | 강등이 읽는 recon 루트 |
| `--graph_marker` | `.graph_d182` | 이 마커가 있어야 ready |
| `--bank_dir` | `hole_bank_d183` | config 의 `bank_dir` 과 같아야 한다 |
| `--require_bank_dir` | `""` | 선행 세대 뱅크가 끝낸 편만 ready 로 본다 (동시 굽기 경합 회피) |
| `--graph_pattern` | `run_bank.py --config configs/bank/d182_` | `ps` 로 선행 생존 판정 |
| `--stage` | `all` | `all` \| `demote` \| `bank` |
| `--bank_shards` | `4` | 카드 1장에 2개가 실측 상한 (~28 GiB/proc) |
| `--gpus` | `0,1,2,3` | 샤드를 올릴 카드. **0~3 범위 유지** |
| `--bank_exec` | `subprocess` | `run_bank.py --exec` 로 전달 |
| `--demote_shards` | `8` | 강등은 CPU 전용(numpy+PIL) |
| `--round_cap` / `--max_rounds` | `0` / `40` | 라운드당 편수 상한(0=무제한) / **구운** 라운드 상한 |
| `--min_ready` / `--poll` | `0` / `900` | ready 가 적으면 굽지 않고 쉰다 / 재스캔 간격(초) |
| `--wait_for` | `""` | 이 문자열을 명령줄에 가진 프로세스가 전부 끝날 때까지 대기 |

### 비고

- `--work` 하나에 드라이버 하나다 (`chain.lock`, PID 생존으로 판정). 같은 `--work` 로 둘을 띄우면
  한 씬의 뱅크 디렉토리에 두 프로세스가 쓴다.
- `--max_rounds` 는 **구운 라운드**만 센다. ready 0 으로 쉬는 것은 안 센다.
- 강등 재실행 판정은 `from_seg.json` 의 mtime 이 `<graph_marker>` 보다 오래됐는지로 한다 —
  필드만 보면 "예전 graph 로 강등된" 편을 조용히 건너뛴다.
- 샤드 rc 만 보면 샤드 안에서 몇 편이 FAIL 했는지 안 보인다. `tally()` 가 로그의 `=== 요약 ===`
  표를 세어 편별 상태 분포를 같이 찍는다 — 그 줄을 읽을 것.

---

## `rebake_scenes.py`

### 무엇을 하는가

코드 버그를 고친 뒤 **영향받은 씬만** 마커를 무시하고 다시 굽는다. 마커는 지우지 않고
`run_bank.py --no_skip_done` 으로 우회하며, 기존 뱅크 산출물은 지우지 않고
`<work>/quarantine/<generation>/<video>/` 로 **옮긴다**. `skipped.json` 과
`canonical/canonical.json` 이 서로를 안 지우기 때문에, 덮어쓰면 한 씬이 두 결론을 동시에 든다.
env 는 `vista4d`.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python exec/rebake_scenes.py --stage all \
    --videos /data1/cympyc1785/LatentCamVid/tmp/d182/merge_bug_scenes.txt \
    --work /data1/cympyc1785/LatentCamVid/tmp/d186 \
    --graph_config configs/bank/d182_dynpose100k_graph.json \
    --bank_configs configs/bank/d184_dynpose100k_single.json,configs/bank/d185_dynpose100k_grid5.json \
    --wait_for "run_bank.py --config configs/bank/d182_" --shards 8
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--videos` / `--work` | (필수) | 재굽기 대상 목록 파일 / 로그·격리 위치 (`tmp/dNN`) |
| `--stage` | `all` | `graph` \| `bank` \| `all` |
| `--graph_config` | `configs/bank/d182_dynpose100k_graph.json` | graph 단계 config |
| `--bank_configs` | `configs/bank/d184_dynpose100k_single.json,configs/bank/d185_dynpose100k_grid5.json` | 콤마 구분, 왼쪽부터 순서대로 |
| `--shards` | `8` | GPU 4장에 고르게 분배 |
| `--wait_for` | `None` | 이 패턴의 프로세스가 전부 끝나면 시작 (60초 폴링) |
| `--quarantine` / `--no_quarantine` | `True` | 뱅크 재굽기 전 기존 산출물을 격리 |

### 비고

- 카드 배정이 **모듈 상수로 하드코딩**되어 있다 (`GPUS = ["0", "1", "2", "3"]`, `:40`).
  CLI 로 못 바꾸므로 다른 카드를 쓰려면 파일을 고쳐야 한다. 현행 0~3 정책과는 일치한다.
- `load_config` 가 `run_bank.load_config` 과 같은 `extends` 얕은 병합을 한다. 뱅크 config 대부분이
  `output_root` 를 부모에만 두기 때문에, 이게 없으면 `KeyError: 'output_root'` 로 죽는다.
- `--stage all` 은 graph 를 먼저 전부 끝내고 그 다음 `--bank_configs` 를 왼쪽부터 순차 실행한다.

---

## `run_corpus_export.py`

### 무엇을 하는가

구워진 뱅크를 latentcam 학습 코퍼스로 옮기는 한 방향 파이프라인. `--stage` 로 단계를 고르고,
단계는 앞에서 뒤로 의존하되 전부 재실행 안전하다. 세대 차이는 전부 argparse 기본값에 있다 —
현재 기본값은 **D200**(`hole_bank_d185` + `hole_bank_d199`, 씬당 상한 6) 이다.
하위 프로세스는 파일에 박힌 `vista4d` python 으로 부른다.

| stage | 하는 일 |
|---|---|
| `desc` | 지칭구가 없는 씬을 VLM 으로 채운다 → `instance_desc_<tag>.json`. **vLLM 서버 필요** |
| `merge_desc` | 위 결과를 `instance_desc.json` 에 **합친다** (덮어쓰지 않는다) |
| `captions` | 뱅크마다 `build_bank_captions.py` 를 같은 `--out_name` 으로 |
| `export` | `vista4d_bank_to_dl3dv.py` 로 뱅크를 한 통에 붓고 씬당 상한을 씌운다 |
| `verify` | seg_list 행수 / 씬 수 / preset 분포 / 뱅크 출처 분포 검산 |

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python exec/run_corpus_export.py \
    --stage captions --banks hole_bank_d185 hole_bank_d199 \
    --captions_name captions_d200.json --per_scene_cap 6
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--stage` | `verify` | `desc` \| `merge_desc` \| `captions` \| `export` \| `verify` \| `all` |
| `--banks` | `["hole_bank_d185", "hole_bank_d199"]` | 여러 개 가능 |
| `--per_scene_cap` | `6` | 씬당 카메라 상한. `0` 이면 무제한 |
| `--captions_name` | `captions_d200.json` | 뱅크 안에 떨어질 캡션 파일명 |
| `--desc_name` | `instance_desc_d200.json` | VLM 지칭구 산출물 이름 |
| `--out_root` | `/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200` | 코퍼스 출력 |
| `--tmp` | `/data1/cympyc1785/LatentCamVid/tmp/d200` | 목록·로그 |
| `--drop_status` | `["clamped_low"]` | 제외할 status. free-moving 은 남긴다 |
| `--drop_suspect` | `[]` | |
| `--cine_out` | `<CT>/out_dynpose` | 뱅크가 있는 곳 |
| `--test_videos` | `<CT>/configs/dynpose_holdout_scenes_100.txt` | legacy 27씬은 `dynpose_holdout_scenes.txt` |
| `--prompt_style` | `nl` | `nl` \| `fields` |
| `--magnitude` / `--no_magnitude` | `False` | 크기 부사. D176+ 는 끈다 |
| `--api_base` | `http://127.0.0.1:22002/v1` | `desc` 단계의 vLLM 주소 |
| `--batch` | `20` | `desc` 서브프로세스당 영상 수 |
| `--num_shards` / `--shard_id` | `1` / `0` | `desc` 단계만 샤딩한다 |
| `--workers` | `8` | export 병렬도 |
| `--dry_run` / `--no_dry_run` | `False` | |

### 비고

- `desc` 단계는 `serve_qwen3vl.sh` 가 `--api_base` 에 떠 있어야 한다. **그 단계에만** 띄우고
  끝나면 내린다 (Qwen3-VL-30B 상시 기동 금지).
- 캡션에 `--videos all` 을 주면 `metadata.csv` 기본값이 Vista4D 것이라 조용히 0편이 된다.
  그래서 이 드라이버는 항상 `--videos_file` 로 명시한다 (FIX-D129-a).
- `banks_of()` 는 `bank.json` + `poses.npz` 가 **둘 다** 있는 뱅크만 센다. 폴더 존재만 보면
  변이 0 인 씬이 대상에 끼어 캡션 단계가 통째로 경고를 뱉는다.
- 파일 상단의 `LATENTCAM`(`:30`)과 `PY_LC`(`:32`)는 정의만 되어 있고 아무 데서도 안 쓴다.
- `verify` 는 뱅크 재고가 아니라 **실제로 나온 코퍼스 행**을 센다.

---

## `run_custom_caption.py`

### 무엇을 하는가

"영상 1개 + 캡션 1줄 → 우리 모델이 만든 카메라 + depth warp 영상". **추론 전용 최소 경로**로,
SAM3 · noun VLM · instance_desc VLM · geocalib · scene graph 노드 · 뱅크를 **하나도 안 돈다**
(그것들은 pseudo-GT 를 굽기 위한 것이고 여기서는 카메라를 모델이 만든다). GT 가 없으므로
지표도 안 낸다. 드라이버 자신은 `vista4d` 로 띄운다 (`stage_scale` 이 `scene_graph.*` 를
in-process import 한다). 하위 호출은 단계별로 `vista4d`(recon/scale/corpus/warp) 와
`latentcam`(molmo2/eval) 을 파일에 박힌 절대경로로 나눠 쓴다.

단계: `recon` → `scale` → `corpus` → `molmo2` → `eval` → `warp` → `bundle` (`all` 이 이 순서).

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python exec/run_custom_caption.py \
    --stage all --gpu 1 \
    --video /data1/cympyc1785/LatentCamVid/tmp/custom/my_clip.mp4 --name my-clip \
    --caption "The camera pedestals down while looking at a man playing golf." \
    --molmo2_text "Track the man playing golf."
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--stage` | (필수) | `recon` `scale` `corpus` `molmo2` `eval` `warp` `bundle` `all` |
| `--name` | (필수) | 씬 이름. **`_` 금지** (하류가 마지막 `_` 에서 잘라 seg 를 읽는다) |
| `--video` | `""` | 입력 mp4. `recon` 단계에만 필요 |
| `--caption` | `""` | umt5 text embedding 으로 들어갈 문장 한 줄 |
| `--molmo2_text` | `""` | PE-AV(Molmo2) 입력 문장. 안 주면 `--caption` 원문 |
| `--preset` | `custom` | 번들 하위 폴더 이름표일 뿐, 카메라와 무관 |
| `--seeds` | `[42, 1234, 2026]` | seed 하나당 카메라 한 벌 |
| `--gpu` | `0` | **0~3 만** |
| `--force` | `False` | 이미 있는 산출물도 다시 만든다 |

파일에 박힌 상수: 체크포인트는 `results/20260918_140914_dynpose_d200_molmo2_l21_da3` 의
`last.pth` (`best.pth` 금지), 프레임 49장, recon 720x1280, 코퍼스는 그 절반(640x360),
번들은 `<CT>/results/20260921_d221_bundles/<name>/<preset>/`.

### 비고

- `--caption` 과 `--molmo2_text` 는 **다른 인코더**로 들어간다. D200 학습 때 Molmo2 override 는
  `Track {지칭구}.` 한 형식뿐이었으므로, PE-AV 가 target 을 짚게 하려면 그 틀로 주는 쪽이 분포 안이다.
- SAM3 를 안 타므로 점군의 동적 점이 0개다. warp 은 `--temporal_persistence auto` 가
  `--allow_no_seg` 를 보고 **NTP(그 프레임 점만)** 로 떨어진다 — 시간 누적(TP) 을 쓰면
  움직이는 물체가 49프레임 겹쳐 유령 다발이 된다.
- `target_poses.npz` 에는 **소스 궤적을 그대로** 한 벌 넣는다. 형식상 자리채움이지 GT 가 아니므로,
  eval 이 같이 찍는 clatr/caption 점수는 **읽지 말 것**.
- `eval_dir` 이름에 씬과 seed 를 둘 다 물린다. 안 물리면 eval 은 "이미 있음"으로 건너뛰고
  릴은 옛 예측을 그리는데 둘 다 rc=0 이라 안 들킨다.

---

## `run_vista4d_gen.py`

### 무엇을 하는가

`run_custom_caption.py` 의 뒤를 잇는다: **이미 있는 카메라**(우리 모델 예측이든 뱅크든 손으로
만든 것이든)를 Vista4D 에 먹여 생성 영상을 뽑고 `source | gen` 가로 concat 릴까지 낸다.
씬이 이미 recon 되어 있으면 `--video <씬이름>` 만 줘도 된다. env 는 `vista4d`, GPU 1장.

단계: `recon` → `cam` → `gen` → `out`.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python exec/run_vista4d_gen.py \
    --video car-roundabout \
    --camera results/20260921_d221_bundles/car-roundabout/track_orbit_right/cameras/s1234.npz \
    --prompt "A dark gray Mini Cooper navigates a roundabout." --gpu 1 --res 384p
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--video` | (필수) | 소스 mp4 경로, 또는 이미 recon 된 씬 이름 |
| `--camera` | (필수) | `.npz`(OpenCV c2w, recon world 절대 미터) 또는 nerfstudio `.json` |
| `--prompt` | `""` | Wan 텍스트 프롬프트 |
| `--name` / `--tag` | `--video` / `--camera` 의 stem | 씬 이름 / 카메라 이름 |
| `--nouns` | `None` | SAM3 keyword. **새 영상 recon 에 필수**, metadata dynamic 열로도 간다 |
| `--gpu` | `"1"` | **0~3 만** |
| `--seed` | `"52106"` | Wan diffusion seed. 카메라만 바꿔 비교할 땐 고정할 것 |
| `--res` | `384p` | `384p` \| `720p` |
| `--num_frames` / `--height` / `--width` | `"49"` / `"720"` / `"1280"` | recon 해상도 |
| `--fps` | `12` | 릴 fps |
| `--out_dir` | `<CT>/results/20260921_vista4d_custom` | |
| `--stage` | `["all"]` | `all` `recon` `cam` `gen` `out` (여러 개 가능) |
| `--force` | `False` | recon 이 있어도 다시 돈다 |

### 비고

- 카메라는 **프레임 수가 recon 과 같아야** 하고 frame0 위치가 소스와 같아야 한다 (허용 0.05 m).
  어긋나면 world 가 다른 것이라 `bank_to_vista4d_cams.py` 가 거기서 죽는다.
- `--nouns` 를 안 주고 새 영상을 recon 하면 assert 로 막힌다. `_all_` 로 두면 SAM3 자체를 안 타서
  `dynamic_mask` 가 비고 동적 물체가 배경 점군에 섞인다.
- `gen` 단계는 `video_generation/results/20260819_vista4d_eval/run_eval_gen.sh` 를 `bash` 로 부른다
  (공식 2단계 render → inference). GPU·CSV·해상도는 env 로 넘긴다.
- 릴은 libx264 로 쓰되 홀수 폭에서 0바이트가 나오므로 짝수로 잘라 낸다.

---

## `run_gendop_eval.py`

### 무엇을 하는가

GenDoP 릴리즈 ckpt 를 **우리 코퍼스의 split** 에 돌려 우리 모델 arm 들과 한 표에 올린다.
단계가 `inputs` → `infer` → `evaldir` → `score` 넷이고 그 사이로 경로가 여섯 흐른다 —
`CORPORA` 표가 곧 config 다. 새 코퍼스는 그 dict 에 항목 하나를 더하는 것으로 끝난다.
드라이버 자신은 표준 라이브러리만 쓰므로 인터프리터를 가리지 않지만, 하위 단계는 파일에 박힌
절대경로로 `GenDoP`(inputs/infer), `latentcam`(evaldir, index_pick), `vista4d`(score) 를 골라 쓴다.
`--resample gendop_slerp` 일 때만 evaldir 도 `GenDoP` 으로 간다.

`CORPORA` 키: `vista_d121`, `vista_d121_snowboard`, `dynpose_d137`, `dynpose_d200`, `vista_d215`,
`vista_d234`, `vista_d229`, `vista_d238`, `vista_d241`, `dynpose_d244`, `dynpose_d254`.
GenDoP arm 은 `gendop_text` / `gendop_rgbd` 둘.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
PY=/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python
$PY exec/run_gendop_eval.py --corpus dynpose_d200 --stage inputs
$PY exec/run_gendop_eval.py --corpus dynpose_d200 --stage infer \
    --arm gendop_rgbd --pose_length 49 --gpu 0 --limit 4        # smoke
$PY exec/run_gendop_eval.py --corpus dynpose_d200 --stage evaldir --arm gendop_rgbd --pose_length 49
$PY exec/run_gendop_eval.py --corpus dynpose_d200 --stage score \
    --arm gendop_rgbd --pose_length 49 --gpu 0 --cloud_source memory
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--corpus` | `vista_d121` | `CORPORA` 키 |
| `--stage` | `all` | `inputs` \| `infer` \| `evaldir` \| `score` \| `all` |
| `--gpu` | `"0"` | **0~3 만** |
| `--limit` | `None` | smoke 용 엔트리 수 |
| `--depth_norm` | `None` | 안 주면 코퍼스 기본값. `none` \| `median` |
| `--pose_length` | `30` | 모델이 직접 뽑을 포즈 수. 30=릴리즈 학습 길이, 49=우리 코퍼스 |
| `--arm` | `None` | `gendop_text` \| `gendop_rgbd` \| `none`. `none` 은 `--stage score` 에서만 |
| `--overwrite` | `False` | 중단 후 재개는 **반드시** 이걸로 (이어달리기는 RNG 스트림이 갈린다) |
| `--text_only` | `False` | `inputs` 에서 rgbd 를 안 만든다 |
| `--text_dir` / `--text_tag` | `None` / `None` | 텍스트 조건 교체. **둘은 같이 준다** (assert) |
| `--rescale` / `--raw` | `False`(= raw) | `rmax` 를 GT 것으로 심을지. 기본이 raw 다 |
| `--resample` | `index_pick` | `index_pick` \| `gendop_slerp`. `pose_length=49` 면 둘 다 항등 |
| `--scale_token` / `--no_scale_token` | `False` | 기본값 False 가 **공식 배포 판본** |
| `--extra_eval_dir` | `[]` | `LABEL=DIR`. E.T./DIRECTOR 등 외부 베이스라인을 같은 표에 |
| `--scenes` | `None` | 씬 표본 파일. 안 주면 test split 전 씬 |
| `--subject_occlusion` | `False` | 가림 열(`subject_visible_frac`) 추가 |
| `--out_tag` | `None` | 점수 JSON 접미사. 안 주면 `eval_suffix` |
| `--cloud_source` | `npz` | dynpose 는 D178 이후 cloud.npz 가 없으므로 `memory` |
| `--batch_size` | `1` | `1` = 예전 런과 비트동일 |

### 비고

- **`--cloud_source` 를 틀리면 조용히 실패한다.** dynpose 계열(`dynpose_d200`, `dynpose_d244`,
  `dynpose_d254`)과 `vista_d215` 이후 세대는 `memory` 여야 한다. `npz` 로 두면 전 씬이
  "cloud/graph 없음" 으로 빠져 표가 통째로 nan 이 되는데 rc 는 0 이다.
- `--rescale` 과 `--scale_token` 은 2026-09-21 지시로 **기본값을 뒤집었다**. 예전 판본을 재현하려면
  `--rescale --scale_token` 을 명시한다.
- entry 가 1~3 개인 코퍼스(`vista_d234`, `vista_d238`, `vista_d241`, `dynpose_d254` 등)에서
  FCD/PRDC 는 의미가 없다 — 읽을 수 있는 건 `clatr/clatr_score` 와 caption 지표뿐이다.
- argparse `--depth_norm` 의 도움말 주석이 "기본은 코퍼스 기본값 (vista=median, dynpose=none)" 이라고
  적혀 있으나, **현재 `CORPORA` 11개 항목은 전부 `depth_norm="median"`** 이다. 주석이 낡았다.
- `eval_data` 규약이 두 스크립트에서 다르다. `inputs` 는 `eval_data` 까지, `score` 는 그 **부모**를
  받는다 — 드라이버가 `path.dirname` 으로 갈라 준다.

---

## `run_director_vista.py`

### 무엇을 하는가

Vista 씬 하나의 subject OBB track + 영문 caption 을 **DIRECTOR(E.T.)** 에 넣어 카메라 궤적을 받는
파일럿. DIRECTOR 의 `src/evaluate.py` 는 et-data 데이터셋에 묶여 단일 샘플을 못 넣으므로,
배치를 손으로 조립해 `Diffuser.sample()` 만 직접 부른다. env 는 `GenDoP`(hydra/clip/torch).
**좌표 변환·스케일 게이지 함수의 원본**이고 `run_director_batch.py` 가 이 파일을 import 한다 —
파일럿 결과 재현을 위해 한 줄도 고치지 않는다.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python exec/run_director_vista.py \
    --scene_graph out/camel/scene_graph.json --subject dyn_0 \
    --caption "The camera trucks right and pushes in on the camel." \
    --out results/20260827_director_camel
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--scene_graph` | (필수) | `scene_graph.json` |
| `--subject` | `dyn_0` | char 로 쓸 노드 id |
| `--caption` | (필수) | 영문 camera caption |
| `--out` | (필수) | 결과 폴더 (`director_poses.npz`, `meta.json`) |
| `--checkpoint` | `<DIRECTOR>/checkpoints/director/ca-mixed-e449.ckpt` | |
| `--guidance_weight` | `1.4` | config 기본값 |
| `--num_steps` | `10` | EDM 스텝 |
| `--num_samples` | `4` | seed `0..N-1` |
| `--scale_mode` | `height` | `height` \| `path` \| `scene` |
| `--subject_height_m` | `2.0` | `height` 게이지 (낙타 키 기준) |
| `--target_path_m` / `--target_scene_m` | `1.0` / `1.0` | `path` / `scene` 게이지 |
| `--meters_per_u` | `None` | 주면 위 환산을 통째로 덮는다 |
| `--et_basis` | `graph` | `graph` \| `legacy`. `legacy` 는 좌표계를 착각한 옛 버그 |
| `--device` | `cuda` | |

### 비고

- `build_diffuser` 가 `chdir(DIRECTOR_ROOT)` 를 한다 (clatr.yaml 의 ckpt 경로가 리포 루트 상대).
  그래서 `--out` / `--scene_graph` / `--checkpoint` 는 **먼저 절대화**된다.
- scene graph 노드는 **G frame**(up=+z, 스케일 1/S)이다. `cam_t_g` / `char_g` 를 렌더·시각화에
  쓰려면 `T_wg` 를 곱해야 한다 — v1 은 이걸 빼먹어 up 축이 camel 93.75° / snowboard 101.28° 틀렸다.
- **미검증 규약 둘**: E.T. world 축 부호(좌우 반전 가능)와 camera 축이 OpenCV 인지 OpenGL 인지.
  이 파일은 회전을 안 쓰고 위치(t)만 되돌린다. `meta.json` 의 `unverified` 필드에 같이 적힌다.

---

## `run_director_batch.py`

### 무엇을 하는가

`run_director_vista.py` 는 엔트리마다 프로세스를 새로 띄우고(ckpt + CLIP 로드 ~30 s) CLIP 을
`encode_caption` 안에서 매번 다시 올린다 — d200 test split 5,144 엔트리를 그대로는 못 돈다.
이쪽은 **모델을 한 번만 올리고 배치로** 돌고, 좌표 변환·게이지는 파일럿 함수를 그대로 import 한다.
출력 `director__<scene>__<idx>.npz` 는 **GenDoP 어댑터(`gendop_preds_to_eval_dir.py`) 규약**에 맞춘다
(`c2w` 는 코퍼스 world·OpenGL 표기, `scale` 은 1.0). env 는 `GenDoP`.

배치가 단일 호출과 같은 결과인가: 같다. `Diffuser.sample` 이 배치 원소마다 별도 `torch.Generator`
를 쓰므로 `seeds = [seed]*B` 면 각 원소가 단독 실행과 같은 latent 을 받는다.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python exec/run_director_batch.py \
    --split /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200/seg_list_dynpose_s91_test.txt \
    --corpus /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
    --cloud_root out_dynpose --prefix dynpose \
    --text_dir /data1/cympyc1785/LatentCamVid/camera_generation/latentcam/eval_my/20260918_140904_dynpose_d200_da3__last \
    --out results/20260920_d208_gendop_d200/pred_director_p49
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--split` | (필수) | `<dataset>/<scene>/<idx>` 한 줄씩 |
| `--corpus` | (필수) | `prompts.json`(variant_id → subject) 이 있는 코퍼스 루트 |
| `--cloud_root` | (필수) | `<root>/<scene>/scene_graph.json` |
| `--prefix` | `dynpose` | 캡션 파일명 접두사 |
| `--text_dir` | (필수) | `<dir>/test/<prefix>_<scene>_<idx>_caption.json` |
| `--out` | (필수) | npz 출력 폴더 |
| `--checkpoint` | `<DIRECTOR>/checkpoints/director/ca-mixed-e449.ckpt` | |
| `--guidance_weight` / `--num_steps` | `1.4` / `10` | |
| `--seed` | `0` | 엔트리당 1 샘플 |
| `--batch` | `16` | |
| `--scale_mode` | `height` | `height` \| `path` \| `scene` |
| `--subject_height_m` | `1.7` | **"샷의 주인공 = 사람 키"** 로 전 엔트리에 같은 게이지 |
| `--target_path_m` / `--target_scene_m` | `1.0` / `1.0` | |
| `--et_basis` | `graph` | `graph` \| `legacy` |
| `--cam_conv` | `opencv` | `opencv` \| `opengl`. **런마다 실측값을 요약에서 확인할 것** |
| `--device` | `cuda` | |
| `--limit` | `None` | 앞 N 줄만 |
| `--overwrite` | `False` | 기본은 이미 있는 npz 를 건너뛴다 |

### 비고

- **E.T. camera 축 규약은 미검증이다.** 매 런마다 "카메라 +z 축과 (char − cam) 사이 각"을 실측해
  요약에 찍는다. 90°보다 확실히 작으면 OpenCV, 확실히 크면 OpenGL이고, **두 값이 다 90° 근처면
  그 런의 회전은 못 믿는다.**
- dynpose anchor 는 손·컵·사람이 섞여 실제 높이를 못 쓴다. 그래서 `--scale_mode height` +
  `--subject_height_m 1.7` 로 **전 엔트리에 같은 게이지**를 건다. 값 자체가 게이지 선택이므로
  npz meta 에 남긴다.
- `c2w` 저장 시 `@ GL2CV` 를 한 번 곱해 둔다 — 어댑터가 다시 곱해 정확히 상쇄된다 (자기역원).

---

## `run_board_sweep.py`

### 무엇을 하는가

TRUMANS `Recordings_blend/` 전량에 `trumans_first_pose_board.py` 를 돌린다. blend 하나 여는 데
~4 s 라서 recording 당 Blender 를 **한 번만** 띄운다. 출력은 `<out>/<recording_id>/board.json`,
실패는 죽이지 않고 `<out>/sweep.json` 에 stderr 꼬리와 함께 모은다.
드라이버 자신은 표준 라이브러리만 쓰고, 실제 작업은
`/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender` 가 한다.

디렉토리 나열이 사소하지 않다. 72 항목 중 `- 副本` 5개(중복본)를 빼고, `<id>/<id>.blend` 규칙을
안 지키는 3개는 디렉토리 안의 유일한 `.blend` 로 집고, 빈 디렉토리 1개를 빼면 **66편**이다.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
python exec/run_board_sweep.py --out out/sweep66 --skip_done \
    --board_args "--chunk_stride 145 --frame_step 3 --lens 18 --min_clearance 0.3 \
                  --min_crop_keep 0.5 --anchor_frame start --gate_frames anchor \
                  --max_render 12 --render_select diverse"
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--root` | `/data1/cympyc1785/data/trumans/Data_release/Recordings_blend` | |
| `--out` | (필수) | `<out>/<recording>/board.json` + `<out>/sweep.json` |
| `--board_args` | `""` | board 스크립트로 그대로 넘길 인자 문자열 (`shlex.split`) |
| `--limit` | `0` | `0` = 전량. 스모크용 |
| `--skip_done` | `False` | `board.json` 이 있으면 건너뛴다 |

### 비고

- **현재 상태로는 안 돈다.** `SCRIPT`(`:32`)가
  `path.join(dirname(abspath(__file__)), "trumans_first_pose_board.py")` = `exec/trumans_first_pose_board.py`
  를 가리키는데, 그 파일은 실제로 `fit/bank/trumans_first_pose_board.py` 에 있다. 파일이 `scripts/`
  에서 `exec/` 로 옮겨질 때 같이 안 따라온 것으로 보인다. 쓰려면 `SCRIPT` 를 고쳐야 한다.
- `--out` 은 예시처럼 `out/` 아래로 두거나, 일회성이면 `/data1/cympyc1785/LatentCamVid/tmp/<작업>/`
  아래로 둔다.
- TRUMANS 작업이라도 GPU 를 쓰는 부분은 0~3 안에서 해결한다 (예전 "TRUMANS 한정 GPU 5" 예외는 만료).

---

## `serve_qwen3vl.sh`

### 무엇을 하는가

로컬 Qwen3-VL(vLLM) OpenAI 호환 서버를 띄운다. `run_corpus_export.py --stage desc` 같은 VLM 단계의
백엔드다. 원본은 DynamicVerse `dynamicgen/scripts/local_server.sh` 이고 바꾼 것은 셋 —
GPU 기본값, 루프백(`127.0.0.1`) 전용, `--allowed-local-media-path` 를 CinemaTraj `out/` 으로.
`vllm` 은 `vllm` env 에만 있다.

### 실행 예시

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
screen -dmS vlm bash exec/serve_qwen3vl.sh              # 단일 GPU
curl -s 127.0.0.1:22002/v1/models                       # 기동 확인

# 1만 편 배치처럼 무거울 때만 TP 를 올린다 (카드 수와 TP 가 같아야 한다)
LBM_VLM_GPU=0,1,2,3 LBM_VLM_TP=4 screen -dmS vlm bash exec/serve_qwen3vl.sh
```

### 주요 인자 (전부 환경변수)

| env | 기본값 | 설명 |
|---|---|---|
| `LBM_VLM_GPU` | `0` | `CUDA_VISIBLE_DEVICES`. **0~3 만** |
| `LBM_VLM_TP` | `1` | tensor parallel 장 수. `CUDA_VISIBLE_DEVICES` 의 장 수와 **반드시 같아야** 한다 |
| `LBM_VLM_PORT` | `22002` | |

고정값: 모델 `Qwen3-VL-30B-A3B-Instruct`, `--dtype bfloat16`,
`--gpu-memory-utilization 0.90`, `--max-model-len 32768`, `--host 127.0.0.1`.

### 비고

- **상시 기동 금지.** Qwen3-VL-30B 는 카드를 통째로 먹는다. VLM 이 필요한 단계 **직전에 띄우고,
  그 단계가 끝나면 내린다** (`screen -S vlm -X quit`).
- `LBM_VLM_TP` 와 카드 수가 어긋나면 vllm 이 기동 중에 죽는다.
- `PATH` 에 `vllm` env 의 bin 을 앞세우는 줄(`:21`)을 지우지 말 것. 절대경로 python 만 부르면
  `ninja` 가 PATH 에 없어 flashinfer 가 sampling 커널 JIT 빌드 중
  `FileNotFoundError: 'ninja'` 로 죽는다.

---

## `exec/_legacy/`

여기 있는 **34개 bash 샤드는 사문화(deprecated)됐다.** 전부 특정 세대(dNNN) 전용으로 쓰였고,
서로 다른 곳은 플래그 몇 줄뿐이다 — 바로 그 복붙 구조를 끝내려고 위의 python 드라이버들이 생겼다.
**근거 기록(provenance)** 으로만 남겨 둔다: 이미 구워진 세대가 어떤 인자로 나왔는지 되짚을 때 본다.

- 34개 중 **31개가 CinemaTraj 루트를 절대 경로로 하드코딩**한다
  (`CT=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj`).
  나머지 3개 중 `run_dynpose_d145_nouns.sh` 만 `HERE=$(cd "$(dirname "$0")/../.." && pwd)` 로
  **상대 해석**하고, `run_dynpose_d145_sam3.sh` / `run_dynpose_d149_sam3.sh` 는 CinemaTraj 루트를
  아예 안 쓴다 (`video_generation/scripts/sam3_seg_instances.py` 만 부른다).
- 다수가 로그·중간 목록을 `/tmp/...` 에 쓴다 (`probe_dynpose_scale_mode.sh`,
  `run_dynpose_d107_shard.sh`, `run_dynpose_d110_*.sh`, `run_dynpose_d129_export.sh` 등).
  현행 규칙 위반이다 — `/data1/cympyc1785/LatentCamVid/tmp/` 를 써야 한다.
- GPU 번호도 인자나 하드코딩으로 들어 있어 0~3 정책을 보장하지 않는다.

**새 작업에서 이 샤드를 복사하지 말 것.** 뱅크는 `run_bank.py` + `configs/bank/<gen>.json`,
겹치기 굽기는 `chain_bank_rounds.py`, 재굽기는 `rebake_scenes.py`, 코퍼스는
`run_corpus_export.py` 로 간다.

```
probe_dynpose_scale_mode.sh      run_approach_viz.sh              run_d143_caption_export.sh
run_d99_caption_export.sh        run_dynpose_d107_shard.sh        run_dynpose_d110_export.sh
run_dynpose_d110_shard.sh        run_dynpose_d122_shard.sh        run_dynpose_d129_export.sh
run_dynpose_d129_shard.sh        run_dynpose_d145_nouns.sh        run_dynpose_d145_sam3.sh
run_dynpose_d147_caption_export.sh  run_dynpose_d148_geocalib.sh  run_dynpose_d149_sam3.sh
run_dynpose_d149_shard.sh        run_dynpose_d163_export.sh       run_f0share_shard.sh
run_k6_d115_shard.sh             run_k6_d121_shard.sh             run_k6_d128_shard.sh
run_k6_d77_shard.sh              run_k6_d98_shard.sh              run_k6_d99_shard.sh
run_k6_shard.sh                  run_preset_warp_max_shard.sh     run_preset_warp_sample.sh
run_raycast_threshold_viz.sh     run_static_rung_shard.sh         run_trumans_d106_shard.sh
run_trumans_d115_shard.sh        run_trumans_d132_shard.sh        run_trumans_d77_shard.sh
run_trumans_d99_shard.sh
```
