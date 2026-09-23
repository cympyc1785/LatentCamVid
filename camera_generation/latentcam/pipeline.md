# pipeline.md — latentcam 모델 구조

이 문서는 **모델이 무엇으로 이루어져 있고 변형(arm)이 어느 축에서 갈리는지**만 다룬다.
학습 방법·평가 지표·실행 인자는 `train.md` 로 뺐다.

| 무엇을 찾나 | 어디 |
|---|---|
| 학습·평가 방법, metric, 주의사항 | `train.md` |
| 코퍼스가 어떻게 만들어지나 | `../dataset/pipeline.md` |
| 변경 이력 | `CHANGELOG.md` |
| 실험 로그 (run 이름 · 세팅) | `EXPERIMENTS.log` |
| 버그 수정 로그 | `FIX.log` |
| 코퍼스축 × 모델축 설계 메모 | `docs/corpus_and_model_axes.md` |

**이 문서의 모든 `file:line` 은 2026-09-23 에 코드를 열어 확인했다.** 코드가 옮겨가면
줄번호는 낡는다 — 쓰기 전에 그 줄을 열어 보고, 틀렸으면 문서를 고친다.
경로 기준은 `camera_generation/latentcam/` 이다.

---

## 0. 한 장 요약

```
  (source video, target camera, caption)  ← ../dataset 가 만든 코퍼스
                 │
                 ├─ caption ─────────► T5 (또는 PE-AV/Molmo2)  ──┐
                 ├─ source frames ───► geo encoder (frozen)    ──┤  조건 토큰
                 └─ target camera ───► CameraVAE.encode        ──┤
                                          (B,49,11) → (B,13,64)  │
                                                  │              │
                                                  ▼              ▼
                            ┌──────────── CameraDiffusionModel (8층 DiT) ────────────┐
                            │  층마다: self-attn → text CA → video CA → geo CA       │
                            │  timestep 은 FiLM 으로 (mod1/mod2/mod3)                │
                            └───────────────────────┬────────────────────────────────┘
                                                    │ ε 예측 (B,13,64)
                                         DDPM/DDIM 역과정
                                                    │
                                     CameraVAE.decode → (B,49,11)
                                                    │
                          rot6d(6) + trans(3) + fx/W, fy/H(2)  → 카메라 49프레임
```

**핵심 한 줄**: 확산 모델이 다루는 것은 카메라가 아니라 **카메라의 VAE latent** 다.
DiT 의 입출력 폭은 `cam_dim: 64` (`main/conf/config.yaml:150`) 이고 11 이 아니다.
`__init__` 기본값 `cam_dim=11` (`models/camera_diffusion_model_latent.py:137`) 은
`use_vae: false` 시절의 잔재라 실제 런은 전부 cfg 가 덮는다.

### 단위·기호

| 기호 | 뜻 | 값 |
|---|---|---|
| `T` | 프레임 수 | 49 (`conf/config.yaml:198 num_frames`) |
| `W` | latent 토큰 수 | 13 (49 → stride-2 conv 두 번) |
| `D` | latent 채널 | 64 (`conf/config.yaml:150 cam_dim`) |
| `hidden` | DiT 폭 | 512 |
| `V` | context view 수 | 코퍼스마다 다름 (DL3DV≈6 / SD 6~49 / DataDoP 1) |
| `M`,`P` | geo 토큰 수 | backend·view 수에 따라 |

11 채널의 내역: `rot6d(6) + trans(3) + fx/W(1) + fy/H(2 중 1)` = 회전 6D + 이동 3 + intrinsics 2.
`intr_norm: rel` 이면 DL3DV·dynpose 처럼 scene 내 intrinsics 가 상수인 코퍼스에서 뒤 2채널이
상수 `[1,1]` 로 죽는다 — D261 시작 pose 가 11 이 아니라 9 인 이유가 이것이다
(`conf/experiment/vista_d261_molmo2_l21_da3_startpose.yaml` 주석).

---

# Part A — 모델 본체

## A1. `CameraVAE` — 왜 latent 인가

`models/vae_intr_large.py:122 CameraVAE`, `latent_dim=cfg.cam_dim` 로 생성
(`main/train_latent_cam_dm.py:827`).

```
CameraEncoder (:5)   Conv1d ×N, 그중 stride=2 가 두 번 (:15, :31)
                     → to_mu / to_logvar (:48-49)
                     (B,49,11) → (B,13,64)
CameraDecoder (:67)  Upsample ×2 로 역방향 → (B,13,64) → (B,49,11)
```

학습 루프에서의 사용은 **세 줄뿐**이다:

| 무엇 | 어디 |
|---|---|
| 인코딩 (GT 궤적 → latent) | `train_latent_cam_dm.py:984`, `:1334` — `camera_vae.encode(traj) / cfg.vae_latent_scale` |
| 디코딩 (val 궤적 지표) | `:1059` — `camera_vae.decode(out * cfg.vae_latent_scale)` |
| 디코딩 (aim loss 용) | `:389` — `camera_vae.decode(x0_hat * cfg.vae_latent_scale)` |

`vae_latent_scale: 0.96032625` (`conf/config.yaml:194`) 는 latent 를 단위분산 근처로 맞추는
상수이고, **ckpt 속성이다** — `vae_ckpt_path` (`:195`, 현재 `vae_20260302_300.pth`) 를 바꾸면
같이 바뀐다. VAE 는 frozen 이며 `train_vae_dl3dv.py` 가 따로 학습한다.

`causal_vae: true` (`conf/config.yaml:64`) 면 `models/cam_causal_vae.py` 의 past-only conv 판으로
교체된다 (`train_latent_cam_dm.py:819-825`). Diffusion-Forcing 경로 전용이고 기본 off 다.

## A2. `CameraDiffusionModel` — 8층 DiT

`models/camera_diffusion_model_latent.py:134`. 진입점은 `train_latent_cam_dm.py:43` 의
`from models.camera_diffusion_model_latent import CameraDiffusionModel` 하나뿐이다.

같은 파일의 나머지 심볼: `timestep_embedding:6`, `positional_encoding:16`, `FusedMLP:26`,
`TimeEmbedding:42`, `CrossAttention:56`, `SelfAttention:68`, `PeavReadout:82`, `get_model:616`.

### 치수 (`__init__:137-163` 기본값 → 실제 런 값)

| 항목 | `__init__` 기본 | 실제 (config.yaml) |
|---|---|---|
| `cam_dim` | 11 | **64** (`:150`) |
| `hidden_dim` | 512 | 512 |
| `num_layers` | 8 | 8 |
| `num_heads` | 8 | 8 |
| `text_dim` | 4096 | 4096 (T5) / 1024 (PE-AV text) |
| `geo_latent_dim` | 768 | 768 (lagernvs) — da3 raw 는 3072 |
| `time_emb_dim` | 512 | 512 |
| `dropout` | 0.1 | 0.1 |

나머지 스위치는 전부 0/false 기본이다: `cond_dim`, `video_latent_dim`, `video_text_dim`,
`peav_readout_layers`, `peav_readout_aux_dim`, `start_pose_dim`, `geo_in_mlp`, `geo_pe`.
**기본값에서는 파라미터도 forward 경로도 예전 arm 과 비트 동일**하게 남도록 전부 분기로 짜여 있다.

### 층 하나의 순서 (`forward:402`, 루프 `:481`)

`self.layers` 의 원소는 7-모듈 묶음이다 (`:217-228`):
`[LayerNorm, SelfAttention, CrossAttention(text), FusedMLP, LayerNorm, CrossAttention(geo), FusedMLP]`.
video 스트림은 여기 끼지 않고 **별도 `self.video_layers`** (`:286`) 에 3-모듈
`[LayerNorm, CrossAttention, FusedMLP]` 로 있다 — 기존 ckpt 의 `layers.<i>.<0..6>.*` 키를
한 글자도 안 바꾸기 위해서다 (`:232-235` 주석).

```
h ← cam_in(x_t [+cond])                         :431   (cond 는 채널 concat)
h ← h + positional_encoding(T)                  :433-434
[start_pose_dim>0] h ← [h ; start_query]        :455   T → T+1

층 i 마다 (:481):
  norm1(h) · FiLM(mod1(t_embed))                :483-489
  h ← self_attn(h)                              :491
  h ← text_cross_attn(h, text_tok)              :492
  h ← mlp1(h) + h                               :493
  [has_video]                                   :496-507
      v ← vnorm(h) · FiLM(mod3(t_embed))
      v ← video_cross_attn(v, video_tok)
      v ← vmlp(v) + v
      h ← h + video_gate[i] * v                 :507  (gate=None 이면 h ← v, 덮어쓰기)
  [has_geo_latent]                              :509-519
      norm2(h) · FiLM(mod2(t_embed))
      h ← geo_cross_attn(h, geo_tok)
      h ← mlp2(h) + h

[start] start_pred ← start_out(h[:, T:T+1])     :546
h ← h[:, :T]                                    :547
return out(h)                                   :548   (B, T, cam_dim)
```

> `:548` 의 꼬리 주석 `# (B, T, 9)` 는 **낡았다**. `self.out = nn.Linear(hidden_dim, cam_dim)`
> (`:333`) 이므로 실제 폭은 `cam_dim` = 64 다.

`CrossAttention.forward` 가 `norm(x + a)` 를 돌려주므로 **스트림 하나마다 h 가 한 번 더
정규화되고 x_t 성분이 깎인다.** video 스트림을 세 번째로 붙였을 때 8층 누적 감쇠가 ~11배
커져 D117 두 arm 이 loss ~1.0 (= ε 예측이 0) 에서 못 빠져나온 실측이 `:272-283` 에 남아 있다.
그래서 video 만 **0 초기화 residual gate** (`self.video_gate`, `:284`) 로 붙는다 —
`video_gate: false` 면 옛 덮어쓰기 동작으로 돌아간다 (D117 ckpt 재현용).

### 조건이 들어오는 자리 — 네 군데, 방식이 다 다르다

| 조건 | 어떻게 | 투영 | forward |
|---|---|---|---|
| **text** | cross-attn (층마다) | `text_proj:190` (+ `text_ln:189`, `text_in_ln` 기본 false) | `:460-461`, `:492` |
| **geo** (scene context) | cross-attn (층마다) | `geo_proj:201/:203` (+ `geo_cam_mlp:198`, `geo_in_mlp:213`) | `:463-468`, `:518` |
| **video** (PE-AV / Molmo2) | cross-attn (층마다, gate) | `video_proj:264` (+ `video_text_proj:270`) | `:470-479`, `:504` |
| **track** (subject 위치) | **채널 concat** | `cam_in = Linear(cam_dim + cond_dim, hidden)` `:174` | `:426-429` |

track 만 cross-attn 이 아니라 입력 채널 concat 인 것이 요점이다. `cond=None` 이면 zeros 로
채워 계측 호출이 안 죽게 해 두었지만(`:427-428`), 학습·샘플링 경로는 반드시 `cond` 를
명시적으로 넘겨야 한다 (`build_track_cond`).

text/video/geo 토큰은 전부 **frozen 인코더 출력**이고, 캐시에서 읽거나 학습 루프가 on-the-fly
로 굽는다. DiT 가 학습하는 것은 그 뒤의 proj 부터다.

### timestep — FiLM 세 벌

`time_mlp:176 → time_proj:177` 로 얻은 `t_embed` 를 `mod1:178`(자기·텍스트 블록),
`mod2:179`(geo 블록), `mod3:271`(video 블록) 세 Linear 가 각각 scale/shift 로 쪼갠다.

`t` 는 `(B,)` (기존, 토큰에 broadcast) 또는 `(B,T)` (per-token noise, Diffusion-Forcing) 둘 다
받고 `per_token` 플래그로 갈린다 (`:439`).

### attention weight stash

`self.training is False` 일 때만 층별 attn weight 를 누적하고 층수로 나눈다
(`:521-541`). 즉 **eval/추론에서만** `text_cross_attn_weight` / `geo_cross_attn_weight` /
`video_cross_attn_weight` 가 채워진다. `main/video_ca_probe.py` 가 이걸 읽어 video 스트림이
실제로 쓰이는지 잰다 (gate 가 0 초기화라 "붙였는데 안 쓴다"가 가능하다).

## A3. 확산 과정

`train_latent_cam_dm.py:605-626` 에서 DDPM/DDIM 스케줄러를 cfg 로 만든다.

| 키 | 기본 | 어디 |
|---|---|---|
| `diffusion_max_step` | 1000 | `conf/config.yaml:53` |
| `prediction_type` | `epsilon` | `:57` |
| `use_vae` | true | `:148` |

**세 가지 학습 경로**가 `:1380-1386` 에서 갈린다:

| 경로 | 조건 | 무엇 |
|---|---|---|
| 표준 diffusion | 기본 | ε 예측, 전 토큰 같은 t |
| AR | `is_ar: true` (`:62`) | `forward_ar:550`, history 조건 |
| Diffusion-Forcing | `per_token_noise: true` (`:65`) | 토큰마다 tau, rectified flow (`:480`), `causal_vae` 와 짝 |

> **AR / per-token 은 최신 조건 스트림과 배타다.** `target_track_dim>0`(`:682-683`) 과
> `video_latent_dim>0`(`:695-696`) 둘 다 `is_ar` / `per_token_noise` 를 assert 로 막는다.
> 즉 `main/tau_sampler.py` `main/rolling_sampler.py` 계열은 **video/track arm 과 같이 못 쓴다**.

---

# Part B — 변형 구조 (arm)

## B1. arm 은 yaml 한 장이다

`main/conf/config.yaml` 이 기본값 전량, `main/conf/experiment/<arm>.yaml` (2026-09-23 기준
**158개**) 가 그 위의 override 다. hydra 규약이라 실행은 `experiment=<이름>` 한 인자로 고른다.

**yaml 이 비어 보이는 건 상속 때문이다.** 대부분의 후기 arm 은 `defaults:` 로 부모 yaml 을
물고 **다른 한 줄만** 적는다. 예:

```
dynpose_d206_molmo2_l21_aim  <  dynpose_d200_molmo2_l21_da3  <  dynpose_d200_molmo2_da3
dynpose_d194_molmo2_l21_track_d5  <  dynpose_d194_molmo2_l21  <  dynpose_d194_molmo2_nogeo
```

그래서 **grep 으로 arm 을 분류하면 안 된다** — 키가 안 보이는 것과 기본값인 것이 구별되지
않는다. 실제 값은 hydra 가 합성한 뒤를 봐야 한다.

## B2. 축 — 무엇이 arm 을 가르나

아래 개수는 158개 yaml 에서 **그 키를 직접 적은** 파일 수다 (상속분 제외).

### ① geo 축 — scene context 를 무엇으로 인코딩하나 (`geo_encoder`, 명시 121/158)

| 값 | 수 | 무엇 | 구현 |
|---|---|---|---|
| `da3` | 58 | Depth-Anything-3 cross-view ViT (frozen), RGB 만 | `models/da3_geo_encoder.py` |
| `null` | 38 | geo CA 없음 (텍스트/비디오만) | — |
| `lagernvs` | 16 | LagerNVS Reconstructor (frozen VGGT + connector), native 768 | `models/geo_encoder.py:_LagerNVSBackend` |
| `custom` | 9 | frozen DINOv2 + trainable 헤드 | `models/custom_geo_encoder.py` |
| `scenetok` | 0 | SceneTok (**stub — 실제로 쓰인 적 없음**) | `geo_encoder.py:343` |

팩토리는 `models/geo_encoder.py:335 build_geo_encoder`, 분기 네 개가 `:339 / :343 / :347 / :355`.
`custom` 과 `da3` 만 cfg 를 통째로 넘긴다.

geo 토큰은 **캐시에서 읽는 게 정상 경로**다. 자르는 지점이 둘이고 뜻이 다르다
(`conf/config.yaml:480-496`):

- `geo_latent_cache_dir` — `GeoEncoder.proj` **뒤**. da3 는 ln+proj 가 학습 대상이라 못 쓴다.
- `geo_raw_cache_dir` — DA3 backbone 출력, `self.ln` **직전**. da3 arm 은 이쪽.

`load_points: true` 면 geo 대신 옛 point-cloud 경로가 살아난다
(`point_encoder: concerto|custom|mosaic`, `train_latent_cam_dm.py:50-57`). 기본 false 다.

### ② video 축 — 영상 토큰을 붙이나 (`video_latent_dim`, 명시 31/158)

| 값 | 수 | 무엇 |
|---|---|---|
| `0` | 14 | video CA 없음 (명시적 off arm) |
| `2560` | 14 | **Molmo2** hidden |
| `1792` | 3 | **PE-AV** (`dynpose_d117a_peavvid`, `dynpose_d117b_peavtext`, `vista4d_d121_peav`) |

곁가지 스위치:

- `peav_layer: 21` (7개) — Molmo2 마지막 층 대신 `blocks[21]` raw 출력을 쓴다.
  캐시는 재사용하고 읽는 키만 `emb` → `emb_l21` 로 바꾼다. 두 층은 스케일이 달라
  `peav_in_ln: true` 가 켜져 있어야 proj 이 흡수한다.
- `video_text_fuse` — `token` (기본, 토큰축 concat) vs `frame_concat` (Molmo2 decode 캐시
  전용, 프레임 f 의 decode feature 를 그 프레임 patch 64개에 채널축으로) (`:245-253`).
- `peav_readout_layers` / `peav_readout_queries: 49` / `peav_readout_aux_dim: 3` — video 토큰을
  DiT 에 바로 먹이지 않고 shallow transformer (`PeavReadout:82`) 로 한 번 요약.
  `aux_dim=3` arm 은 `dynpose_d194_molmo2_l21_readout`, `dynpose_d201_molmo2_l21_readout`.

### ③ 조건 추가 축 — 명시된 arm 이 손에 꼽는다

| 키 | 값 | arm | 무엇 |
|---|---|---|---|
| `target_track_dim` | 4 | `dynpose_d194_*_track_d5`, `dynpose_d201_*_track_d5`, `vista4d_pgt_k6_track{,_d5}`, `vista4d_pgt_k6_d77_track_d5` | subject 위치를 x_t 채널에 concat |
| `srccam_cond` | `plucker` | `dynpose_d200_molmo2_l21_srccam` | DA3 자리에 소스 카메라 49프레임 Plücker (D262). `geo_encoder: null` 이어야 하고 `geo_in_mlp`+`geo_pe` 가 켜진다 (`train_latent_cam_dm.py:675`) |
| `aim_loss_w` | 0.1 | `dynpose_d206_molmo2_l21_aim` | camera-forward ↔ subject 각 보조 손실 (D206) |
| `start_pose_pred` | true | `vista_d261_molmo2_l21_da3_startpose` | 시작 pose 를 9-d query 토큰으로 같이 예측 (D261) |

### ④ text 축 (`text_encoder`)

기본 `T5` (`conf/config.yaml:203`, `text_len: 512`). `PEAV` 를 명시한 건 2개
(`dynpose_d117b_peavtext`, `vista4d_d121_peav`). UMT5 캐시 경로도 있다
(`cache_umt5_embeddings.py`, `dynpose_d200_umt5only` = D263 바닥선 arm).

### ⑤ 코퍼스 축 (`train_seg_list` / `test_seg_list`)

`dataset_name` (`conf/config.yaml:230`, 기본 `dl3dv`) 이 클래스를 고르고
(`main/base.py:30-42`), 실제 코퍼스는 seg_list 경로가 정한다. 158개 yaml 의 `train_seg_list`
루트 분포 (상위):

| 루트 | 수 |
|---|---|
| `DL3DV/scenes` | 34 |
| `DynPose-LBM/latentcam_dynpose*` (세대별 d107~d200) | 22 |
| `Vista4D-Eval-Data/latentcam_da3*` (d77/d121/d128/d261) | 18 |
| `TRUMANS-Lite/latentcam_da3*` | 6 |
| `Scene-Decoupled-Video-dataset/latentcam_lists` | 5 |

`dataset_name` 이 고르는 네 클래스:

| 값 | 클래스 | context 를 어디서 |
|---|---|---|
| `dl3dv` (기본) | `dataset_dl3dv.CamDataset` | 같은 scene 의 다른 **프레임 구간** |
| `scene_decoupled` | `dataset_scene_decoupled.SDCamDataset` | 같은 scene 의 다른 **clip** |
| `datadop` | `dataset_datadop.DataDoPCamDataset` | shot 의 **frame 0 한 장**뿐 |
| `mixed` | `dataset_mixed.MixedCamDataset` | 위를 섞는다 (`datasets:` 목록) |

`mixed` 는 코퍼스마다 `copy.deepcopy(cfg)` + 화이트리스트 override 로 **기존 클래스를 그대로**
만든다. 코퍼스마다 context view 수 V 가 달라 한 배치에 섞이면 `collate_fn` (`main/base.py:184`)
의 `torch.stack` 에서 터지므로 `main/mixed_sampler.py:PerCorpusBatchSampler` 가
배치 하나 = 코퍼스 하나를 보장한다.

---

# Part C — 모듈별 설명

## C1. 죽은 코드 판별 규칙

이 트리는 SCVideo 의 `core_pkg` 에서 포팅됐고, 포팅 방향은 `core_pkg.* → models.* / utils.*`
였다 (`main/base.py:1-2` 주석). `core_pkg` 는 **실제로 import 불가능**하다
(`importlib.util.find_spec('core_pkg')` → None).

> **규칙: 모듈 레벨에 `core_pkg` import 가 있으면 죽은 파일이다.**
>
> - `grep -l core_pkg` 로 세지 말 것 — `main/base.py` 는 **주석**에 문자열이 있을 뿐
>   살아 있고, `train_latent_cam_dm.py` 도 grep 에는 걸린다 (`:651`, 함수 안).
> - 판별은 AST 로 모듈 레벨 `Import`/`ImportFrom` 만 봐야 한다.

## C2. `main/` — 살아 있는 33개

### 학습·추론 진입점

| 파일 | 무엇 |
|---|---|
| **`train_latent_cam_dm.py`** | **유일한 살아있는 학습 진입점.** hydra, 1500줄 |
| `train_vae_dl3dv.py` | CameraVAE 학습 (frozen 으로 쓸 ckpt 를 만든다) |
| `infer_validation_sample.py` | val 샘플 추론 |
| `infer_textonly_batch.py` | 텍스트만으로 일괄 추론 |
| `infer_swap_ablation.py` | 조건 교환 ablation |
| `gen_chain_drift.py` / `gen_scene_align.py` / `gen_scene_rolling.py` | 궤적 생성 변종 |

### 캐시 굽기 (frozen 인코더 출력 사전계산)

`cache_geo_embeddings.py` (DA3 기하) · `cache_molmo2_embeddings.py` · `cache_peav_embeddings.py`
· `cache_umt5_embeddings.py` · `convert_geo_raw_cache.py` (형식 변환).

캐시가 정상 경로다 — frozen 인코더를 매 step 돌리면 학습이 인코더 속도에 묶인다.

### 데이터셋

`dataset_dl3dv.py` (본체, `CamDataset`) · `dataset_scene_decoupled.py` · `dataset_datadop.py`
· `dataset_mixed.py` · `mixed_sampler.py` · `dataset_cfg.py` (init 의 cfg 해석 ~150줄을 떼어낸
것) · `dynamicverse_shim.py` (DL3DV 전용 분석 스크립트를 DynamicVerse 에 태우는 shim)
· `prepare_metadata.py`.

### 골격

`base.py` (`Trainer`, `collate_fn:138`, dataset 팩토리 `:24-42`) · `hydra_cfg.py` (config
로더; `LATENTCAM_CONFIG` 환경변수 back-compat) · `config.py`.

### 계측·프로파일

`video_ca_probe.py` (video CA 가 실제로 쓰이나) · `profile_geo.py` · `profile_rolling.py`
· `test_per_token.py` · `dump_avgscale_render.py` · `dump_render_inputs.py`.

### Diffusion-Forcing 전용

`tau_sampler.py` (per-token tau, W=13) · `rolling_sampler.py` (full_sequence / chunk_ar /
rolling 세 모드를 tau 스케줄만으로) · `vae_intr.py`.
**video/track arm 과 배타다** (§A3 assert).

## C3. `main/` — 죽은 15개

모듈 레벨 `core_pkg` import 14개:

```
config_large.py:152      config_vae.py:138        dataset_seg.py:13        eval.py:4
infer_ablation.py:9      infer_cam_dm.py:9        infer_latent_cam_dm.py:9 infer_vae.py:9
prepare_clatr_data.py:14 train_cam_dm.py:17       train_cam_dm_attn_sup.py:17
train_vae.py:8           valid_cam_acc.py:17      valid_cam_dm.py:13
```

`run_cam_dm.py` 는 한 술 더 뜬다 — **파싱조차 안 된다**
(`f-string: unmatched '(' (line 133)`).

> 함정: `config_large.py` 가 죽었는데도 코퍼스 루트를 여기서 읽는 문서·에이전트 요약이
> 나온다. 이 파일의 경로들은 낡았다 — 실제 코퍼스는 arm yaml 의 `train_seg_list` 다.

## C4. `models/`

| 파일 | 참조 | 판정 |
|---|---|---|
| `camera_diffusion_model_latent.py` | — | **살아있는 모델 본체** |
| `vae_intr_large.py` | 13 | 살아있는 VAE (`CameraVAE`) |
| `t5.py` | 21 | 텍스트 인코더 |
| `geo_encoder.py` | 5 | geo 팩토리 + lagernvs/scenetok backend |
| `da3_geo_encoder.py` | 1 | `geo_encoder: da3` backend |
| `custom_geo_encoder.py` | 1 | `geo_encoder: custom` backend |
| `pc_encoder{,_custom,_mosaic}.py` | 10/11/7 | `load_points: true` 옛 경로 |
| `vae_intr.py` | 5 | 구판 VAE |
| `cam_causal_vae.py` | 2 | `causal_vae: true` 전용 |
| `camera_diffusion_model{,_director,_mmdit,_attn_sup,_base}.py` | — | **죽음** (부르는 쪽이 전부 `core_pkg` arm) |
| `clip_encoder.py` `tokenizers.py` `vae.py` | 0 | **죽음** |
| `vlm_planner.py` | 0 | **죽음** — 3줄짜리 주석 스텁, 코드 없음 |

`vlm_planner.py` 전문:

```python
# VLM planner using Qwen3-VL
# Input : video frames and high level intent text
# Output : structured low level camera plan
```

## C5. train/val 분할 — `val_ratio` 는 없다

`main/base.py:84-114` 에 **두 경로**가 있고 서로 한 바이트도 안 겹친다.

| 경우 | 무엇 |
|---|---|
| `train_seg_list` + `test_seg_list` 둘 다 지정 (**정상 경로**) | 파일에 적힌 `<batch>/<hash>/<seg_key>` 목록 그대로. **val = test 목록의 앞부분**, 크기는 `val_max_batches × batch_size` (`conf/config.yaml:33`, `:264` 주석) |
| 둘 다 `null` (기본값) | `train_frac` (기본 **0.9**, `base.py:111`) 로 `random_split` — 무작위 90/10 |

즉 "비율" 손잡이는 `val_ratio` 가 아니라 **`train_frac`** 이고, 그것도 seg-list 를 안 준
경우에만 쓰인다. 실제 arm 은 전부 seg-list 를 준다 (§B2-⑤) — 그쪽에서 val 비율을 바꾸려면
`test_seg_list` 와 `val_max_batches` 를 **같이** 고쳐야 한다.

> CLaTr 게이지를 쓰는 arm 은 split 을 재분할하면 안 된다 — 옛 CLaTr ckpt 가 새 test 를
> 이미 본 상태가 되어 지표가 조용히 부푼다.

## C6. `models/geo_encoder.py` 의 `sys.path` 곡예

lagernvs 도 top-level `models` 패키지를 쓴다. 그대로 import 하면 latentcam 의 `models` 를
덮어쓴다. `_shadow_models_package` (`models/geo_encoder.py:29`) 가 `sys.modules` 에서
`models*` 를 잠시 빼고 `repo_path` 를 `sys.path` 앞에 붙였다가 복원한다.
**lagernvs 트리가 옮겨가면 여기가 먼저 죽는다.**

---

# 부록 — 지뢰

| 지뢰 | 실제 |
|---|---|
| `cam_dim` 을 11 로 읽음 | 런타임 값은 **64** (`conf/config.yaml:150`). DiT 는 VAE latent 를 디노이즈한다. `__init__` 기본 11 과 `:548` 주석 `# (B, T, 9)` 둘 다 낡았다 |
| `epochs: 100` 을 상한으로 읽음 | 실제 집행자는 **`epoch_cap: 50`** (`:26`). `_epochs = min(cfg.epochs, epoch_cap)` (`train_latent_cam_dm.py:1243`) → `for epoch in range(start_epoch, _epochs)` (`:1259`) 이라 마지막 인덱스는 49 → `ckpt_at_epochs: [50,100]` 은 **영원히 안 걸린다**. 최종은 `last.pth` |
| `val_ratio` 를 찾음 | **그런 키는 없다.** seg-list 경로면 val = test 목록 앞부분, seg-list 가 null 이면 `train_frac: 0.9` 무작위 분할 (§C5) |
| `text_encoder: CLIP` 로 바꿈 | `train_latent_cam_dm.py:651` 의 CLIP 분기가 죽은 `core_pkg` import 를 탄다 — **크래시** |
| grep 으로 arm 분류 | hydra `defaults:` 상속이라 자식 yaml 은 한 줄만 적는다 (§B1) |
| `grep -l core_pkg` 로 죽은 코드 판정 | `base.py` 는 주석, `train_latent_cam_dm.py` 는 함수 안. AST 로 모듈 레벨만 (§C1) |
| AR/per-token arm 에 video·track 붙임 | assert 로 막혀 있다 (`:682-683`, `:695-696`) |
| `config_large.py` 에서 코퍼스 경로 읽음 | 그 파일은 죽었다 (§C3) |
| import 실패 (`ModuleNotFoundError: utils`) | `cd main && PYTHONPATH=<repo>:.` — `models/` 와 `utils/` 가 `main/` 의 형제다 |
| `vae_latent_scale` 을 임의로 바꿈 | **ckpt 속성**이다. `vae_ckpt_path` 와 짝이 아니면 recon 이 터진다 |
