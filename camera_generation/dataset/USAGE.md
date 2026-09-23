# USAGE.md — `camera_generation/dataset/` 실행 안내

이 트리는 **(source video, target camera, caption) 코퍼스를 굽는 파이프라인**이다.
여기서 학습은 하지 않는다 — 학습은 `camera_generation/latentcam/`.

이 문서는 **어디부터 읽어야 하는지와 공통 규칙**만 적는다. 스크립트별 인자는 갈래마다 붙은
USAGE 가 원본이다.

| 문서 | 무엇 |
|---|---|
| `pipeline.md` | **데이터가 만들어지는 순서와 각 단계가 그 모양인 이유** — 먼저 읽을 것 |
| `README.md` | 설계 원칙 (왜 전수 열거인가, 왜 게이트인가) |
| `presets.md` | preset 어휘 46종 전량 + 캡션 문구 표 |
| `DECISIONS.md` | 각 결정의 근거·기각된 대안 (D1~) |
| `CHANGELOG.md` | 변경 이력 |
| **`exec/USAGE.md`** | **드라이버** — 실제로 치는 명령은 거의 전부 여기 |
| `fit/USAGE.md` | 단계 스크립트 (`ingest → graph → bank → caption → convert`) |
| `eval/USAGE.md` | 자(尺). 뱅크를 안 고치고 세고 재고 대조한다 |
| `eval/GENDOP_USAGE.md` | GenDoP 베이스라인을 우리 split 에 돌리기 |
| `viz/USAGE.md` | 판정용 산출물 (depth warp 릴 · 비교 그리드 · viser) |

> **`fit/` `eval/` `viz/` USAGE 안의 "CinemaTraj" 는 이 트리의 옛 이름이다** (R7, 2026-09-23
> 이전에는 `camera_generation/models/Planner/CinemaTraj/` 였다). 문서 안의 **경로는 전부 현재
> 기준**이고 이름만 남아 있다.

---

## 1. 하드 규칙

### 이 디렉토리를 cwd 로 두고 돈다

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/dataset
python exec/<드라이버>.py ...
```

스크립트들이 **자기 파일 경로로부터 루트를 역산**하고 (`run_bank.py:61 HERE`,
`run_corpus_export.py:29 CINE`, `chain_bank_rounds.py:52` / `rebake_scenes.py:38 ROOT`),
그 루트를 cwd 로 삼아 하위 단계를 `fit/bank/...` 같은 **상대 경로**로 부른다.
`--config configs/bank/<gen>.json` · `--cloud_root out_dynpose` 도 전부 루트 기준 상대 경로다.
다른 데서 부르면 드라이버는 뜨지만 단계 스크립트를 못 찾아 죽거나, 더 나쁘게는 산출물이
엉뚱한 곳에 쌓인다.

### GPU 는 0~3 만

`--gpu` / `--cuda` / `CUDA_VISIBLE_DEVICES` 를 주는 자리마다 이 범위를 지킨다.
**argparse 기본값에도 숨어 있다** — `--cuda` 기본이 `"6"` 인 스크립트가 있었다. 본문 grep 으로는
안 보이니 기본값을 확인하고 명시로 덮는다.

### 임시 파일은 `/tmp` 가 아니다

`/data1/cympyc1785/LatentCamVid/tmp/<작업>/` 아래에 둔다. `/tmp` 는 시스템이 임의로 비워서
씬 목록·로그가 조용히 사라진다 (그러면 모니터는 살아 있는데 보는 대상이 없어진다).

### env 는 절대경로로

| env | python | 쓰는 곳 |
|---|---|---|
| `vista4d` | `/data1/cympyc1785/miniconda3/envs/vista4d/bin/python` | SAM3, VLM, 뱅크 전체, 캡션, export |
| `da3` | `/data1/cympyc1785/miniconda3/envs/da3/bin/python` | `build_scene_graph.py`, `decode/` verify |
| `geocalib` | `/data1/cympyc1785/miniconda3/envs/geocalib/bin/python` | `geocalib_gravity.py` (kornia) |
| `vllm` | `/data1/cympyc1785/miniconda3/envs/vllm/bin/python` | Qwen3-VL-30B @ 22002 |
| `latentcam` | `/data1/cympyc1785/miniconda3/envs/latentcam/bin/python` | 코퍼스 검증·추론 연계 |
| `GenDoP` | `/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python` | 베이스라인 |
| Blender | `/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender` | TRUMANS / LBM 렌더 |

**vLLM 서버는 상시 기동하지 않는다.** VLM 단계 직전에 `exec/serve_qwen3vl.sh` 로 올리고
끝나면 내린다.

### 새 세대는 새 스크립트가 아니라 config 한 장

`configs/bank/<gen>.json`. `"extends"` 로 부모 config 를 **최상위 키 단위 얕은 병합**한다 —
`fit` 블록 하나만 바꾸려 해도 그 블록은 통째로 다시 적어야 한다.
세대 이름은 **게이트 규약이 바뀌면 바꾼다**(D105): 같은 이름으로 재굽기하면 인덱스 캐시에
내용 해시가 없어 캡션↔변이가 어긋난 채 rc=0 으로 끝난다.

---

## 2. 자주 치는 명령

인자 전량과 각 플래그의 근거는 `exec/USAGE.md` 의 해당 절에 있다.

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/dataset
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
```

### 뱅크 굽기 — 씬 목록이 이미 graph 까지 끝났을 때

```bash
$PY exec/run_bank.py \
    --config configs/bank/d185_dynpose100k_grid5.json \
    --videos <scenes.txt> --stages tau,fit,emit --num_shards 4 --shard_id 0
```

> `run_bank.py` 에는 `--bank_dir` 이 **없다**. 출력 위치는 config 의 `"bank_dir"` 키다
> (`d185_dynpose100k_grid5.json` → `"bank_dir": "hole_bank_d185"`). 세대를 바꾸려면
> 플래그가 아니라 config 를 바꾼다.

### 뱅크 굽기 — graph 부터, graph 와 뱅크를 겹쳐서

graph 는 CPU, 뱅크는 GPU 라 라운드 루프로 겹치면 둘 다 안 논다.

```bash
$PY exec/chain_bank_rounds.py \
    --config configs/bank/d185_dynpose100k_grid5.json \
    --videos <scenes.txt> --work /data1/cympyc1785/LatentCamVid/tmp/<작업>/ \
    --bank_dir hole_bank_d185 --bank_shards 4
```

> **샤드 합계가 GPU 장당 부하다.** `run_bank.py` 는 `--gpu` 가 없으면 `GPUS[shard % 4]` 를
> 쓴다. 두 체인을 8샤드씩 띄우면 장당 4개가 되어 OOM 이 난다 (실측: 네 장 전부 75~80 GB).
> 합계 12 = 장당 3 이 피크 77 GB 로 안전했다.

### 특정 씬만 다시

```bash
$PY exec/rebake_scenes.py \
    --videos <scenes.txt> --work /data1/cympyc1785/LatentCamVid/tmp/<작업>/ \
    --stage all \
    --graph_config configs/bank/d182_dynpose100k_graph.json \
    --bank_configs configs/bank/d184_dynpose100k_single.json,configs/bank/d185_dynpose100k_grid5.json \
    --shards 4
```

> config 플래그 이름이 드라이버마다 다르다. `run_bank.py` · `chain_bank_rounds.py` 는
> `--config` 하나지만 `rebake_scenes.py` 는 **`--graph_config` 와 `--bank_configs` 두 개**다
> (graph 를 한 번만 다시 굽고 그 위에 세대 여러 개를 얹을 수 있게 — `--bank_configs` 는
> 콤마 구분이고 **왼쪽부터 순서대로** 돈다). `--config` 를 주면 rc=2 로 죽는다.
> `--stage` 도 다르다 — 뱅크 단계 이름이 아니라 `graph` / `bank` / `all` 셋 중 하나다.
> `--quarantine` 은 **기본 켜짐**이라 기존 뱅크를 지우지 않고 옆으로 치운다.
> `--shards` 기본은 8 인데 GPU 0~3 만 쓰므로 장당 2 다 — 다른 굽기와 겹치면 낮춘다.

### 뱅크 → 학습 코퍼스

```bash
$PY exec/run_corpus_export.py --banks hole_bank_d185 \
    --stage desc,merge_desc,captions,export,verify
```

### 영상 1개 + 캡션 1줄 → 카메라 + depth warp 릴 (추론 전용 최소 경로)

SAM3/VLM/뱅크 없이 돈다.

```bash
$PY exec/run_custom_caption.py ...
```

### 이미 있는 카메라로 Vista4D 생성 영상

```bash
$PY exec/run_vista4d_gen.py ...        # source | gen 릴까지
```

---

## 3. 산출물 규약

```
out/  out_dynpose/  out_trumans/        # 코퍼스별 작업 루트
  <video>/
    scene_graph.json                    # graph 산출
    .graph_<gen>  .cloud_<gen>          # 재실행 방지 마커
    geocalib_gravity.json               # 중력 사이드카
    instance_desc.json                  # VLM 지칭구
    bank_<gen>/                         # τ 뱅크
    hole_bank_<gen>/                    # fit 결과 — bank.csv · bank.json · canonical/
results/                                # 남길 산출물
```

- **하류는 전부 `status == "solved"` 만 본다.** 다른 status(`collision_limited` 등)는 진단용이다.
- `poses.npz` 의 `target_hole` 은 요청치가 아니라 **달성치**다. 조인 키는 `variant_id`.
- **`variant_id` 는 코퍼스 키가 아니다** — 재굽기해도 이름은 같고 knob/pose 만 바뀐다.
- 점군은 기본적으로 디스크에 안 남는다 (`--cloud_source memory`, D178).
- 번들은 `d221_bundles/<scene>/<preset>/` 에 세대 접미사 **없이** 쌓인다.

### 압축적으로

파일 개수와 용량이 계속 느는 것을 기본 실패 모드로 본다. 로그는 전량 tee 하지 말고 진행·경고·
요약만 남긴다. 세대별 산출물은 최신 + 대조에 실제로 쓰는 세대만 남긴다.
**중간 산출물을 지울 때는 먼저 목록과 근거를 보고하고 승인을 받는다.**

---

## 4. 잘 밟는 지뢰

| 증상 | 원인 |
|---|---|
| graph 가 CPU 를 다 태우는데 안 끝난다 | BLAS 스레드 캡 미설정. 프로세스당 391 threads 가 spin-wait (`run_bank.py:69-77`) |
| `--fallback_ladder` 를 줬는데 아무 일도 안 난다 | τ 에 `--variant_pool full` 이 없으면 `plan_tier` 가 전부 0 이라 no-op |
| `tau` 부터 굽는데 기본 앵커가 돈다 | `route` 를 건너뛰면 `anchors.json` 이 무시된다. rc=0 이라 안 들킨다 |
| 뱅크에 anchor 상한을 줬는데 안 먹는다 | 상한은 `route` 에서만 걸린다. 명시 `--nodes` 는 필터를 안 탄다 |
| `FIT rc=1` — `source_g1_clear` assert | `--behind_clear_src_ratio 0` 을 명시 안 함 (기본 0.3) |
| eval 은 돌았는데 그림이 예전 예측 | `eval_dir` 이름에 태그가 안 물려 eval 은 skip, render 는 이전 preset. 둘 다 rc=0 |
| depth warp 이 새까맣다 | 좌표계가 아니라 궤도 반경이 씬 깊이 대비 너무 작은 것일 수 있다 |
| 동적 물체가 반투명 빗살 | unproject 앞 `preprocess_scene` 누락 — static 경계 누수가 49프레임 쌓인다 |
| hole 이 조용히 낮다 | `cloud` 가 dynmask **앞에** 구워져 동적 점이 0개 |
| 화각이 떨린다 | 릴 렌더가 뱅크의 `fixed_focal` 을 안 따랐다 (`render_bank_videos` 기본 False) |

---

## 5. `scripts/` 는 무엇인가

LBM(원본 Blender 파이프라인)과 TRUMANS `.blend` 를 잇는 **드라이버**들이다. `exec/` 와 달리
Blender worker 를 직접 띄운다.

| 파일 | 역할 |
|---|---|
| `trumans_lite_bank.py` | TRUMANS recording → `fit/ingest/trumans_to_recon.py` 반복 (GT 렌더) |
| `trumans_to_lbm_demo.py` | `.blend` → LBM 레이아웃 + frame shift 훅 |
| `vista_to_lbm_demo.py` | Vista4D 점군 → LBM 이 읽는 `.blend` |
| `lbm_camera_to_poses.py` · `vista_lbm_to_poses.py` | LBM 출력 카메라 → `poses.npz` |
| `build_lbm_bank_manifest.py` | LBM 뱅크 매니페스트 |
| `lbm_camera_dump_startup.py` | Blender startup 훅 (CLI 없음) |

> R6(2026-09-22) 이 `scripts/` 를 갈래로 쪼갤 때 **드라이버와 그것이 부르는 스크립트가 다른
> 갈래로 떨어진 5 지점**이 조용히 죽어 있었다 (R8 에서 수리). 여기 드라이버가 형제 폴더
> `fit/ingest/` 를 부를 때는 `path.join(HERE, ...)` 가 아니라 `path.dirname(HERE)` 기준으로
> 명시한다 — 새 드라이버를 쓸 때도 같은 규칙을 따를 것.
