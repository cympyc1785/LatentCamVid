# scripts/viewer — viser 뷰어

브라우저에서 카메라 궤적을 3D 로 보는 도구 두 개. 둘 다 **읽기 전용**이라 결과물을 건드리지 않는다.

| 스크립트 | 쓰는 때 |
|---|---|
| `viser_val_cameras.py` | run **하나**의 validation/test dump 를 sequence 별로 훑는다. 지표도 같이 찍는다. |
| `viser_arms_gs.py` | segment **하나**를 고정하고 여러 **arm** 의 예측을 겹쳐 비교한다. DL3DV 3DGS `scene.ply` 배경 지원. |

> **arm** = 같은 데이터·같은 평가 세트에 대해 설정만 바꿔 돌린 실험 갈래 (예: `textonly` vs `customgeo`).

---

## 실행/종료 규약

`screen viser1~4` 에서 띄운다. 포트는 screen 마다 다르게 준다.

```bash
cd /data1/cympyc1785/LatentCamVid/camera_generation/latentcam

screen -dmS viser2 -L -Logfile /tmp/viser2.log bash -c \
  "/data1/cympyc1785/miniconda3/envs/latentcam/bin/python -u \
   scripts/viewer/viser_val_cameras.py --dataset sd \
   --root results/compare/sd_whuman_textonly_n160/test --port 8081 2>&1"

tr '\r' '\n' < /tmp/viser2.log | tail -30      # 로그 확인 (지표가 여기 찍힌다)
```

브라우저: `http://localhost:8081` (원격이면 ssh 포트포워딩).

종료는 **Ctrl+C 먼저**, 그 다음 확인:

```bash
screen -S viser2 -X stuff $'\003'
sleep 3 && ss -lntp | grep :8081 || echo "port free"
```

GPU 는 안 쓴다 (numpy/cv2 만). 다만 `--dataset sd` 는 depth/conf `.npz` 를 clip 당 수백 MB 읽으므로 `/data1` (Lustre) I/O 를 탄다 — **학습 중이면 sequence 를 빠르게 연타하지 말 것**.

---

## `viser_val_cameras.py`

`--root` 는 `results/<run>/test` 처럼 아래 세 파일이 sequence 마다 들어 있는 디렉토리다.

```
{data_name}_transforms_pred.json   생성 카메라   RED
{data_name}_transforms_ref.json    target/GT     BLUE
{data_name}_caption.json           text prompt
```

GUI 는 `index` 슬라이더 + `Load` + `Next` 뿐이고, **수치는 전부 터미널(로그)에 찍힌다.**

### DL3DV (기본)

```bash
python scripts/viewer/viser_val_cameras.py \
    --root results/20260808_140209_da3_7k_customgeo_nos/test --port 8080
```

`data_name = "{type}_{hash}_{seg}"` 를 scene chunk `"{type}/{hash}"` 로 풀어 그 scene 의 **나머지** 카메라를 GREY frustum 으로 깐다 (target 프레임은 위치가 정확히 겹치므로 제외).

### Scene-Decoupled (SD)

```bash
python scripts/viewer/viser_val_cameras.py --dataset sd \
    --root results/compare/sd_whuman_textonly_n160/test \
    --pc-clips both --pc-tint --pc-seg-only --port 8081
```

`data_name = "{scene}__{TARGET}__{CONTEXT}"` (`dataset_scene_decoupled.py:128` 규약).

SD 는 `scene.ply` 가 없어서 배경을 **clip 의 da3 depth 를 unproject 한 point cloud** 로 그린다.

**좌표를 어떻게 맞추나 (핵심).** `pose.npz` 의 w2c 는 da3 **임의 스케일**이라 그대로 쓰면 안 된다. `umeyama_gt.json` 의 sim3 로 GT meters 에 올린다 — `dataset_scene_decoupled.py::_load_scene` 과 **같은 식**:

```
R' = R_e·Rᵀ        t' = s·t_e − R'·t        depth  d' = s·d
```

`ref`/`pred` json 은 `first_extrinsic`(= 이 미터 world 에서의 target 첫 w2c) 로 복원되므로 point cloud 와 같은 world frame 에 놓인다.

**`--pc-clips {context,target,both}`** — 어느 clip 을 point cloud 로 그릴지. 기본 `context`.

context 와 target 은 **서로 다른** `umeyama_gt.json` sim3 를 타고 같은 GT meters world 로 올라온다. 그래서 `both` 로 겹쳐 보는 것 자체가 **aligned world 검증**이다. `target` 을 포함하면 그 clip 의 `pose.npz` 카메라와 `ref`(파랑) 카메라의 위치차를 같이 찍는데, 두 값이 완전히 다른 경로로 나오므로 0 이 아니면 world 가 어긋난 것이다:

```
align: target clip pose vs ref(blue) 위치차  median 0.0000m  max 0.0000m  (n=49, ...)
```

(`scene1002_5x5_loc37_scene_Dragon_Rise` 실측 — 0.0000m 로 일치.)

- `--pc-tint` — RGB 대신 **context=초록 / target=파랑** 단색. 둘을 켜면 구분이 필요하다. target point cloud 를 파랑으로 두는 건 같은 target 을 가리키는 `ref` 카메라(파랑)와 색을 맞추기 위해서다.
- `--pc-target-cams` — target clip 카메라도 그린다. 이 카메라만 **청록**인데, ref 파랑 위에 정확히 겹쳐야 정상이라 **겹침 여부를 눈으로 구분하려면 색이 달라야** 하기 때문이다 (같은 파랑으로 두면 어긋나도 안 보인다). 정렬 검증용이므로 평소에는 끄는 게 낫다.
- point cloud 노드는 `pc_context` / `pc_target_clip` 으로 나뉘어 있어 viser scene tree 에서 따로 껐다 켤 수 있다.
- clip 하나가 실패해도 나머지는 그대로 그린다. **static clip** 은 `umeyama_gt.json` 에 `s`/`t` 가 없어서(회전만 있음) `ValueError` 로 건너뛴다 — 정상 동작이다.

### ⚠️ clip 길이 ≠ target segment 길이 (`--pc-seg-only`)

**SD clip 은 81 프레임인데 target segment(`ref` 파랑)는 그 중 앞 49 프레임이다.**
(`whuman` 160/160 sequence 확인: clip `T=81`, ref `N=49`, **offset 0**, 위치오차 `<1e-3 m`.)

안 자르면 뒤 32 프레임이 segment 옆을 **median 0.926m** 로 나란히 지나간다 (궤적 전체 extent 4.173m 의 22%) — 화면에서 **별개의 두 번째 궤적처럼 보인다.** 실제로는 같은 카메라의 segment 이후 구간이다.

```bash
--pc-seg-only        # target clip 의 카메라 + point cloud 를 [0, len(ref)) 로 자른다
```

context 는 clip 전체를 쓰는 게 맞으므로 영향받지 않는다. 로그로 확인:

```
sd pc[context]    : ... + 41/81 cams (stride 2)
sd pc[target_clip]: ... + 25/49 cams (stride 2, clip 81 프레임 중 [0,49) 만)
```

`--pc-frames` 는 "**그 구간에** 균등 분포시킬 프레임 수"다. `--pc-seg-only` 없이 `--pc-frames 49` 를 주면 49장이 81 프레임 **전체**에 퍼진다.

기타 point cloud 옵션: `--pc-stride`(depth 격자 픽셀 stride, 기본 3), `--pc-conf-pct`(conf 백분위수 컷, 기본 40), `--pc-max-points`(기본 400k), `--pc-size`(0 = target extent 로 자동).

### 색

| 색 | 무엇 |
|---|---|
| RED | pred (생성 카메라) |
| BLUE `(40,90,230)` | target / GT (`ref`) 카메라, 그리고 `--pc-tint` 의 target point cloud |
| GREEN `(40,190,90)` | SD context clip 카메라, 그리고 `--pc-tint` 의 context point cloud |
| CYAN `(0,200,200)` | SD target clip 카메라 (`--pc-target-cams` 일 때만) — ref 파랑과 정확히 겹쳐야 정상이라 일부러 색을 다르게 뒀다 |
| GREY | DL3DV scene 의 나머지 카메라 |

point cloud 와 카메라의 색은 **따로** 지정된다 (`viser_val_cameras.py` 의 `_CTX`/`_TGT` 튜플이 `(tag, clip, pc색, 카메라색)`). target 만 둘이 다른 이유가 위의 겹침 판별이다.

### frustum 이 너무 많을 때

`--cam-stride`(target/pred, 기본 2), `--context-stride`(SD context), `--grey-downsample`(DL3DV grey). 전부 "몇 프레임마다 하나" — `1` 이면 전부.

### 지표 출력

sequence 를 load 할 때마다 터미널에 네 덩어리가 찍힌다.

1. **`traj`** — `ref`/`pred` 만으로 그 자리에서 계산. `pos_rmse` / `rot` 정의는 `scripts/render/compare_textonly.py` 와 같다. 단위는 world 단위 그대로 (**SD 는 미터, DL3DV 는 nerfstudio 정규화 단위 — 코퍼스 간 비교 금지**).
2. **`eval`** — `<root>/../preds_scores.csv` 의 그 sequence 행.
3. **`prdc`** — `preds.npy` 에서 재계산한 **sample 별** PRDC.
4. 시작할 때 한 번: **corpus-level** 값.

> **PRDC** = Precision / Recall / Density / Coverage.

**CSV 의 PRDC/FCD 열은 sequence 별 값이 아니다.** set 단위로 한 번 계산한 값이 전 행에 복사돼 있다 (`eval_only.py:186-193` 이 그렇게 쓴다). 그래서 per-sequence 출력에서 빼고 corpus-level 로 따로 한 번만 찍는다. sequence 별로 보고 싶으면 `preds.npy` 에서 재계산하는 3번을 봐야 한다.

재계산이 맞는지 시작할 때 자동 검산한다 — 로그의 `재계산 집계` 가 csv/metrics.json 과 같아야 정상이다 (chunk 순서·`k`·거리 중 **하나만** 어긋나도 sample 별 값이 조용히 딴 값이 된다). `--no-prdc` 로 끌 수 있다.

### 실패 sequence 만 골라 보기

```bash
python scripts/viewer/viser_val_cameras.py --dataset sd \
    --root results/compare/sd_whuman_textonly_n160/test \
    --only fail --dens-max 0 --list-out /tmp/fail.csv --port 8081
```

`--only {all,prec-fail,dens-low,recall-fail,cov-fail,fail}` — 슬라이더/`Next` 도 거른 목록만 돈다. `--list-out` 으로 csv 저장.

**⚠️ 거르기는 PRDC 계산이 끝난 뒤에 한다.** PRDC 는 **전체 순서**를 5등분한 chunk 안에서 계산되므로, sequence 를 먼저 줄이면 chunk 구성이 달라져 값 자체가 바뀐다. 코드가 이 순서를 지키고 있다.

---

## `viser_arms_gs.py`

segment 하나를 고정하고 arm 별 예측을 겹친다. DL3DV 전용.

```bash
python scripts/viewer/viser_arms_gs.py \
    --name 1K_c7576be..._1 \
    --arm 'out-of-seg ctx:eval_my/divers/swap_base:245,130,30' \
    --arm 'in-seg ctx K=5:eval_my/divers/leak5_seed0:150,80,200' \
    --ctx 'ctx out-of-seg:eval_my/divers/swap_base:30,170,70' \
    --ctx 'ctx in-seg (K=5):eval_my/divers/leak5_seed0:230,40,40:5' \
    --port 8080
```

`--arm` / `--ctx` 는 `label:root:R,G,B[:stride]` 형식, 여러 번 줄 수 있다.

`scene.ply` (DL3DV 3DGS) 를 `add_gaussian_splats` 로 깔 수 있는데 **좌표계 함정**이 있다: `scene.ply`/`cameras.json` 은 nerfstudio 의 world 재정렬 `applied_transform` 이 **적용된** frame 인데 `transform_matrix` 는 적용 **전** frame 이라 그냥 겹치면 어긋난다. 그래서

```
c2w_ply = applied_transform4 @ c2w_gl
```

로 world 쪽에 좌측 곱한다 (`gaussian-splatting-lightning/custom_utils/custom_panel.py::get_cameras_from_transforms` 와 같은 처리). 재정렬된 frame 은 COLMAP world 라 **up 이 -y** 다.

frustum 은 `tools/gaussian-splatting-lightning` 의 spline 버전을 쓴다 — viser 기본 `add_camera_frustum` 은 선 두께가 화면 픽셀 단위라 잘 안 먹는데, spline 8개로 그리면 `line_width` 가 실제로 먹는다. 저 repo 가 없으면 viser 기본 스타일로 떨어진다.

---

## 좌표계 요약

저장된 `transform_matrix` 는 **OpenGL c2w** (nerfstudio; y-up, 카메라가 −Z 를 본다) — DL3DV `transforms.json` 과 같은 규약이다. viser frustum 은 **OpenCV** (카메라가 +Z, y-down) 라 변환한다:

```
c2w_opencv = c2w_opengl @ diag(1, −1, −1, 1)
```

이 flip 은 **카메라 축만** 뒤집고 world 좌표는 안 건드리므로 point cloud 는 변환 없이 그대로 겹친다. `add_frustums()` 가 알아서 하므로 호출부는 OpenGL c2w 를 넘기면 된다.

world up: DL3DV `+y` (nerfstudio), SD `+z` (umeyama 로 올린 GT/Unreal 미터 좌표). `--up` 으로 덮을 수 있다.
