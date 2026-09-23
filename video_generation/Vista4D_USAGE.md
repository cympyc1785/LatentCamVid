# Vista4D — 우리가 이 모델을 어떻게 돌렸나

> **이 문서는 왜 모델 폴더 밖에 있나.** 원래 `video_generation/models/Vista4D/USAGE.md` 였는데,
> 그 자리에서는 git 이 추적하지 못해 **재클론하면 사라진다** (아래 표 1행). R15 에서 여기로
> 옮겼다 — `video_generation/` 은 추적 대상이고 `models`·`results`·`tools` 만 닫혀 있다.
> 본문의 `video_generation/models/Vista4D/...` 경로는 전부 **저장소 기준**이다.

`README.md`(upstream, `video_generation/models/Vista4D/README.md`) 는 공식 데모 사용법이다.
이 문서는 **우리 프로젝트에서 실제로 돌린 방식**만 적는다 — 세 갈래 경로, 체크포인트·env,
좌표계 규약, 그리고 재클론하면 날아가는 local 패치.

| 확인한 것 | 결론 |
|---|---|
| Vista4D 트리는 git 추적 대상인가 | **아니다.** `.gitignore` 가 `video_generation/models` 를 막고, 그와 별개로 `Vista4D/` 자체가 **자기 `.git` 을 가진 별도 저장소**라 바깥 저장소가 안의 파일을 추적하지 못한다 (`!` 예외도 안 먹는다). 그래서 **이 문서는 밖으로 뺐고**, 아래 local 패치 2건은 여전히 재클론하면 사라진다 |
| 우리가 돌린 경로는 몇 개인가 | **셋.** ① 벤치마크 그리드(`run_grid.py`) ② 공식 eval 전량(`run_eval_gen.sh`) ③ 단일 씬 드라이버(`run_vista4d_gen.py`) — 지금 쓰는 건 ③ |
| 카메라를 어떻게 먹이나 | 경로마다 다르다. ①은 `vista4d_prepare.py` 의 `g` 손잡이, ③은 **recon world 절대 미터 npz** 를 그대로 |
| 재구성은 공용인가 | 그렇다. stage 1 은 소스당 1회고 카메라 그리드 밖에 있다 |
| 영상 1편 비용 | **~9m 35s** (384p, GPU 1장, denoise 50 step × ~11.25 s) |
| GPU | **0~3 만.** 아래 2026-08-19 기록의 `0,1,6,7` 은 그 규칙(2026-09-21) **이전** 실행이다 |

---

## 0. 한 장 요약

Vista4D 는 **geometry-locked** 이다 (sierp/trajc 계열). 카메라 파일만 갈아끼우는 게 안 되고,
소스마다 자기 4D 재구성을 먼저 만든 뒤 **그 world/스케일 안에서** target 카메라를 지어야 한다.

```
  소스 mp4
     │
     ▼  stage 1   scripts.preprocess.recon_and_seg_single      (소스당 1회, 공용)
  recon_and_seg/<video>/   depth · dynamic_mask · sky_mask · cameras.npz
     │
     │  ← 카메라 npz 를 **이 world 좌표계에서** 만든다
     ▼  stage 2   scripts.preprocess.render_{single,eval}      (point cloud 렌더 = depth warp)
  render_<RES>/<video>/<camera>/     video_pc.mp4
     │
     ▼  stage 3   scripts.inference.inference{,_eval}          (Wan2.1-T2V-14B)
  gen/<video>/<camera>/video_seed=<N>.mp4
```

`_cond/video_pc.mp4` (single) · `render_<RES>/.../video_pc.mp4` (eval) 가 **이 모델의 depth warp** 이다.
출력만 보면 warp 탓인지 생성 탓인지 못 가르므로 릴에는 둘을 같이 싣는다.

| 기호 | 뜻 |
|---|---|
| `V4` | 이 디렉토리 `/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D` |
| `ED` | `/data1/cympyc1785/data/Vista4D-Eval-Data/eval_data` — recon/cameras/render/gen 이 전부 여기 |
| `RES` | `384p` (H=384 W=672) 또는 `720p` (H=720 W=1280) |
| `FSFF` | force-same-first-frame. npz 의 `cam_c2w_fsff` 를 읽는 분기 |

---

## 1. 환경 · 체크포인트

env 는 **`vista4d`** 하나다 (recon / render / inference 전부).

```
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
```

| 체크포인트 | 경로 |
|---|---|
| Vista4D 384p | `checkpoints/vista4d/384p49_step=30000/{dit.pth,config.yaml}` |
| Vista4D 720p | `checkpoints/vista4d/720p49_step=3000/{dit.pth,config.yaml}` |
| Wan 백본 | `checkpoints/wan/Wan2.1-T2V-14B` (`diffusion_pytorch_model*.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth`, tokenizer `google/*`) |
| DA3 recon | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` (HF) |
| Pi3 recon | `yyfz233/Pi3X` (HF) — 우리는 **안 쓴다** |

`HF_HOME` 은 `/data1/cympyc1785/cache/huggingface` 로 넘긴다
(`run_eval_gen.sh:47`, `run_grid.py:532`).

> **`$ENV/bin/python` 을 절대경로로 부르면 conda activate 훅이 안 돈다.**
> flash-attn 2.8.3 이 conda 의 libstdc++ 로 빌드돼 있어서, 시스템 libstdc++ 를 먼저 잡으면
> `CXXABI_1.3.15 not found` 로 죽는다. 우리 러너는 전부 절대경로로 부르므로 **inference 단계에는
> `LD_LIBRARY_PATH=$ENV/lib` 를 명시적으로 넘겨야** 한다 (`video_generation/FIX.log` 2026-08-17).
> recon/render 단계는 `flash_attn` import 가 없어 무관하다.

**recon 은 `--recon_method da3` 로 고정한다.** 우리 코퍼스·뱅크·캡션이 전부 DA3 재구성 위에
올라가 있고, upstream 배포 카메라 npz 만 Pi3X 기준이다 (아래 §5).

---

## 2. 경로 ③ — 단일 씬 (지금 쓰는 것)

`camera_generation/dataset/exec/run_vista4d_gen.py`. **이미 있는 카메라**(우리 모델 예측 / 뱅크 /
손으로 만든 것)를 먹여 영상을 뽑는다. 4 단계 `recon | cam | gen | out` (`:193-197`, `--stage` 기본 `all`).

```
/data1/cympyc1785/miniconda3/envs/vista4d/bin/python \
  /data1/cympyc1785/LatentCamVid/camera_generation/dataset/exec/run_vista4d_gen.py \
    --video car-roundabout \
    --camera <bundles>/car-roundabout/track_orbit_right/cameras/s1234.npz \
    --prompt "A dark gray Mini Cooper navigates a roundabout." --gpu 1
```

| 단계 | 하는 일 | 함수 |
|---|---|---|
| `recon` | `recon_and_seg_single --recon_method da3 --save_vis` → `ED/recon_and_seg/<name>/`. `cameras.npz` 가 있으면 건너뛴다 | `:76-97` |
| `cam` | `bank_to_vista4d_cams.py --cams` → `ED/cameras/<name>/<tag>.npz` (규약 변환은 전부 거기 한 군데) | `:100-103` |
| `gen` | 1행 metadata CSV 를 쓰고 `run_eval_gen.sh` 를 `bash` 로 (공식 2단계) | `:112-123` |
| `out` | `video.mp4` / `reel.mp4`(source\|vista4d 가로 concat) / `info.json` | `:130-164` |

argparse 기본값 (`:169-188`): `--gpu 1` · `--seed 52106` · `--res 384p` · `--num_frames 49` ·
`--height 720 --width 1280` (recon 해상도) · `--fps 12` ·
`--out_dir camera_generation/dataset/results/20260921_vista4d_custom`.

임시 파일은 `<repo>/tmp/vista4d_gen/<name>/` 이다 (`:68-69`) — `/tmp` 가 아니다.
로그도 여기 떨어진다 (`recon.log`, `gen_<tag>.log`).

> **새 영상이면 `--nouns` 가 필수다** (`:89` assert). `--seg_keywords` 를 비우거나 `_all_` 로 두면
> SAM3 자체를 안 타서 (`video_generation/models/Vista4D/scripts/preprocess/recon_and_seg_single.py:75-84`) `dynamic_mask` 가 비고,
> 동적 물체가 배경 점군에 섞여 49 프레임 내내 끌려다닌다.

드라이버 쪽 플래그 표·함정은 `camera_generation/dataset/exec/USAGE.md:323-367` 에 있다.
여기서는 **모델 쪽**만 적는다.

### 시드

`SEED_DEFAULT = "52106"` (`run_vista4d_gen.py:52`). 카메라만 바꿔 비교할 때는 **같은 값**이어야
한다 — 노이즈까지 다르면 차이가 카메라에서 온 건지 못 가른다.
영상·preset 을 지정받아 돌릴 때는 seed 3개(42 / 1234 / 2026)로 뽑아
`source | GT | s42 | s1234 | s2026` concat 까지가 한 단위다.

---

## 3. 경로 ② — 공식 eval 전량

`video_generation/results/20260819_vista4d_eval/run_eval_gen.sh` (71행).
저자가 배포한 재구성과 카메라 npz 를 **그대로** 쓴다 (우리가 stage 1 을 돌리지 않는다).

```
GPU=0 CSV=meta_shard0.csv STAGE=all RES=384p bash run_eval_gen.sh
```

| env | 기본 | 뜻 |
|---|---|---|
| `GPU` | (필수) | `CUDA_VISIBLE_DEVICES` |
| `CSV` | `meta_nested.csv` | 상대경로면 `run_eval_gen.sh` 옆, 절대경로도 받는다 (`:42`) |
| `STAGE` | `all` | `render` / `gen` / `all` |
| `RES` | `384p` | `384p`→H384 W672 ckpt `384p49_step=30000` / `720p`→H720 W1280 ckpt `720p49_step=3000` (`:33-35`) |
| `NUM_SHARDS` `SHARD_ID` | `0` `0` | GPU 나눠 돌리기. **gen 단계만 의미 있다** |
| `FSFF` | `0` | `1` 이면 `cam_c2w_fsff` 를 읽고 출력 폴더가 `_fsff` 로 갈린다 (`:30-31,40`) |

디렉토리를 `<video>/<camera>` 로 중첩시키는 트릭: metadata 의 `name` 열만 `"<video>/<camera>"` 로
바꾼 사본을 쓴다. 두 스크립트 다 `name` 을 그대로 하위 경로로 쓰므로 **vendored 코드 수정 0줄**이다
(`run_eval_gen.sh:10-11`).

**2026-08-19 실행 기록** (`RUN_SUMMARY.md` 원본 수치):

| | |
|---|---|
| entry | 110 / 110 (51 video × 카메라) |
| wall-clock | **4h 24m 44s** (15,884 s), 4 샤드 병렬 |
| render / gen | 5m 47s / 4h 18m 57s |
| 영상 1편 | 575 s ≈ **9m 35s** (GPU 1장), denoise 50 step × ~11.25 s/it |
| 산출물 | `gen` 1.3 GB / `render_384p` 2.4 GB |
| Traceback · OOM | 0 · 0 |

> 이 실행은 **GPU 0,1,6,7** 을 썼다 (샤드 표). **2026-09-21 의 "GPU 0~3 만" 지시 이전**이다 —
> 지금 같은 걸 돌리면 샤드 4개를 0~3 에 얹는다.

---

## 4. 경로 ① — 벤치마크 그리드 (2026-08, 아카이브)

`video_generation/tools/recammaster/run_grid.py --model vista4d` (`run_vista4d` at `:480-535`).
7 소스 × 6 카메라를 **다른 모델과 같은 강도 손잡이 `g`** 로 얹기 위한 경로다.
설계·실측표는 `video_generation/results/20260818_vista4d/config.md`.

경로 ②③과 갈리는 지점 셋:

1. **stage 1 을 우리가 돌린다.** `video_generation/results/20260818_vista4d/run_vista4d_recon.sh` 가
   소스마다 1회, `V4_RECON = results/20260818_vista4d/recon` 아래로
   (`run_grid.py:536-537`). 그리드는 stage 2·3 만 돈다.
2. **카메라를 `vista4d_prepare.py` 가 만든다.** `T_w[i] = C_src[a] @ T_local[i]`, `|t|max = g·S`
   — `T_local` 은 `canonical.json` 의 `rel_c2w` (OpenCV c2w, frame0 항등, rmax=1) 를 49 프레임으로
   SE(3) 리샘플한 것이고 **translation 만** 스케일한다. sierp `s=g·S/1.9876`, trajc `s=2g` 와 같은 `g` 다.
3. **프레임 정렬을 미리 맞춰야 한다.** sierp/trajc 는 소스 **앞 49 프레임**을 쓰는데 Vista4D 는
   stage 1(`slice_center_frames`)도 stage 2(`video_generation/models/Vista4D/scripts/preprocess/render_single.py:57-58` 의
   `win_start = src_start + (num_src_frames - num_frames)//2`)도 **중앙 슬라이스**한다.
   → 소스를 `<s>_first49.mp4` 로 **미리 트림**해서 먹인다. 그러면 `num_src == num_frames == 49` 라
   앵커 `a = 0` 이 되고 전 소스 manifest 의 `anchor_index` 가 0 으로 확인된다.

`S` 는 **Vista4D 자신의 recon 스케일**이라 sierp 의 `S_da3` 와 숫자가 다를 수 있다 —
같아야 하는 건 `S` 가 아니라 `tau = |t|/z_med` 다.

`--save_warp` / `--warp_only` 를 주면 `_cond/video_pc.mp4` 를 `_warp/warp.mp4` 로 복사해
sierp 와 같은 규약으로 맞춘다 (`run_grid.py:525-528`).

---

## 5. 카메라 규약 — 여기서 틀린다

카메라 npz 는 `load_cameras` (`video_generation/models/Vista4D/utils/media.py:161-165`) 가 읽는다:

```python
def load_cameras(input_path: str, force_same_first_frame: bool = False):
    data = np.load(input_path)
    if force_same_first_frame:
        return data["cam_c2w_fsff"], data["intrinsics_fsff"]
    return data["cam_c2w"], data["intrinsics"]
```

→ **네 키가 다 있어야 한다.** `cam_c2w_fsff` / `intrinsics_fsff` 가 없으면 `FSFF=1` 에서 KeyError 다.
`bank_to_vista4d_cams.py:220-221` 은 네 키를 전부 같은 배열로 저장한다.

| 항목 | 규약 |
|---|---|
| `cam_c2w` | `(n,4,4)` **OpenCV c2w**, recon world **절대 미터** |
| `intrinsics` | `(n,4)` `[fx, fy, cx, cy]`, recon 픽셀 단위 |
| frame 0 | **위치**가 소스 카메라와 같아야 한다. 회전은 아니다 |
| 프레임 수 | recon 과 같아야 한다 (49) |

`bank_to_vista4d_cams.py` 가 이 셋을 전부 검사하고 어긋나면 거기서 죽는다 (`:208-214`).
허용 오차는 소스에 따라 갈린다 (`:167`): 뱅크 `1e-6` / 예측·파일 `0.05`
— 예측 궤적은 frame0 도 회귀 결과라 정확히 0 이 아니다.

**frame0 회전이 다른 건 정상이다.** `aim="look_at"` preset 은 frame0 부터 이미 subject 를 보므로
소스 회전과 다르다 (`orbit_left_pedestal_up` 실측 6.21°). 위치가 어긋나면 그건 world 가 틀린 것이고,
회전이 어긋나는 건 그냥 그 preset 의 정의다 (`:203-207`).

### 입력 형식 둘

* **`.npz`** — 이미 Vista4D 규약. intrinsics 가 없으면 recon 것을 그대로 쓴다 (`:133-140`).
* **`.json`** — nerfstudio(OpenGL c2w). `diag(1,-1,-1,1)` 로 OpenCV 로 돌리고, 절반 해상도 focal 을
  `cx` 비로 recon 픽셀로 되돌린다 (`:141-148`). 우리 모델 평가 산출물 `*_transforms_pred.json` 이 이것.

### intrinsic zoom

뱅크 궤적은 프레임마다 `focal_scale` 을 가질 수 있다 — `fx[f] = fx_base * focal_scale[f]`.
`fx_base` 는 `bank.json` 최상위 `fixed_focal` (`:198`) 이 참이면 **frame0 고정**, 거짓이면 per-frame 이다.
DA3 focal 은 정지 카메라에서도 드리프트하므로(snowboard 1184.99→1249.76, +5.5%) 릴 렌더는
**뱅크의 `fixed_focal` 을 따라야** 화각이 안 떨린다.

> **배포 카메라 npz 는 Pi3X 재구성 기준이다** (`video_generation/models/Vista4D/scripts/preprocess/example_render_single.sh:11-13`
> upstream 주석). 우리는 DA3 로 recon 하므로 `media/single/*_<preset>.npz` 를 그대로 쓰면 카메라가
> 의도한 자리에 없다. 우리 카메라는 전부 `bank_to_vista4d_cams.py` 가 DA3 world 에서 새로 만든다.

---

## 6. Local 패치 2건 — 재클론하면 재적용

이 트리는 바깥 저장소의 **추적 밖**이다 (`.gitignore video_generation/models`, 그리고 `Vista4D/.git`
이 있어 별도 저장소다). 바깥에서 `git log -- .../Vista4D` 는 비어 있으므로 **패치가 사라져도 diff 로
못 잡는다.** 둘 다 `LOCAL:` 주석이 달려 있으니 `grep -rn 'LOCAL:'` 로 현황을 확인한다.
이 문서(`USAGE.md`) 자체도 같은 이유로 추적되지 않는다 — 재클론 전에 복사해 둘 것.

### (a) `video_generation/models/Vista4D/scripts/preprocess/recon_and_seg_single.py` — `--keep_recon_sky`

`--seg_keywords` 가 비면 SAM3 를 건너뛰는데, 그 분기가 `dynamic_mask` 와 함께 **DA3 가 이미 낸
`sky_mask` 까지 0 으로** 밀어 버렸다. sky 가 0 이면 stage 2 point-cloud 렌더가 하늘 픽셀까지
유한 거리 점으로 깔아서 카메라가 움직일 때 하늘이 같이 끌려간다.

```python
# recon_and_seg_single.py:75-81
if len(args.seg_keywords) == 0:
    dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    if not (args.keep_recon_sky and args.recon_method == "da3"):
        sky_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
```

argparse 등록은 `:155`, 기본값 `False` (기존 동작 보존). `seg_frames = None` 초기화도 같이 들어갔다
(`:74`) — SAM3 를 안 타는 분기에서도 정의돼 있어야 한다.
검증: hs_sC 재-recon → frame0 sky 26.58% (0% 아님).

### (b) `video_generation/models/Vista4D/sam3/model_builder.py` — `pkg_resources` 부재

setuptools ≥81 은 `pkg_resources` 를 안 깐다. top-level import 를 `try/except` 로 감싸고(`:8-11`),
`resource_filename` 호출 둘을 `_default_bpe_path()` (`:63-75`) 하나로 모았다 — 있으면 원래대로,
없으면 `__file__` 기준 상대경로로 `assets/bpe_simple_vocab_16e6.txt.gz` 를 찾는다.
**setuptools 를 내리지 않는다** — env 의 xfuser/diffsynth 가 최신 setuptools 를 요구한다.

### (c) `video_generation/models/Vista4D/utils/recon_and_seg/seg_sam3_utils.py` — `save_seg_instances`

`recon_and_seg_single.py --save_seg_instances` 로 SAM3 인스턴스 마스크를 따로 떨군다
(`video_generation/CHANGELOG.md:7577-7582`). 정적 인스턴스 추출(`sam3_static_instances.py`) 이 이걸 읽는다.

---

## 7. 이 트리를 읽어 가는 다른 코드

recon 산출물·SAM3 를 재사용하려고 `V4` 를 절대경로로 박아 둔 곳들이다. 트리를 옮기면 전부 깨진다.

| 쓰는 곳 | 쓰는 것 |
|---|---|
경로는 전부 `camera_generation/` 또는 `video_generation/` 기준이다.

| 쓰는 곳 | 쓰는 것 |
|---|---|
| `camera_generation/dataset/lbm/cloud.py:35 import_vista4d` | point cloud 구축 (`camera_generation/dataset/pipeline.md:172`) |
| `camera_generation/dataset/fit/graph/{extract_nouns_vlm,describe_instances_vlm}.py` | recon 프레임 |
| `camera_generation/dataset/fit/ingest/{sam3_static_instances,trumans_to_recon}.py` | SAM3 · recon 포맷 |
| `camera_generation/dataset/fit/caption/make_avg_scale_trumans.py` | 스케일 |
| `camera_generation/dataset/eval/{audit_lite_framing,probe_static_sdf,probe_tau_divisor}.py` | recon depth |
| `camera_generation/dataset/exec/{run_custom_caption,run_vista4d_gen}.py` | 드라이버 |
| `video_generation/tools/recammaster/run_grid.py:496` | 벤치마크 그리드 |

---

## 8. 지뢰

| 증상 | 원인 | 대처 |
|---|---|---|
| 하늘이 카메라를 따라 끌려간다 | `--seg_keywords` 빈 분기가 `sky_mask` 를 0 으로 | `--keep_recon_sky` (§6a) |
| 동적 물체가 49 프레임 겹쳐 그려진다 | `dynamic_mask` 가 빔 (`_all_` 이나 빈 키워드) | 실제 명사를 `--nouns` 로 |
| `import flash_attn` CXXABI 실패 | 절대경로 python 이라 activate 훅 미실행 | `LD_LIBRARY_PATH=$ENV/lib` 명시 (§1) |
| `ModuleNotFoundError: pkg_resources` | setuptools ≥81 | §6b 패치 재적용 |
| `FSFF=1` 에서 KeyError | npz 에 `cam_c2w_fsff` 없음 | `bank_to_vista4d_cams.py` 로 만든다 (§5) |
| 카메라가 엉뚱한 자리 | 배포 npz 는 Pi3X 기준 / 다른 소스의 npz | DA3 world 에서 새로 만든다 (§5) |
| 다른 모델과 다른 구간을 비교 | Vista4D 는 중앙 슬라이스, sierp/trajc 는 앞 49 | 소스를 `_first49.mp4` 로 미리 트림 (§4-3) |
| 화각이 미세하게 떨린다 | DA3 focal 드리프트 (+5.5%) | 뱅크의 `fixed_focal` 을 따른다 (§5) |
| `frame0 위치 불일치` 로 죽는다 | 카메라가 recon world 가 아님 | 좌표계를 고친다 — `--frame0_tol` 로 덮지 말 것 |
| reel.mp4 가 0 바이트 | libx264 + yuv420p 는 홀수 폭에서 조용히 실패 | 짝수로 crop (`run_vista4d_gen.py:155-156`) |
| 패치가 사라졌다 | 이 트리는 추적 밖이라 diff 가 안 잡는다 | `grep -rn 'LOCAL:'` 로 확인하고 §6 재적용 |
