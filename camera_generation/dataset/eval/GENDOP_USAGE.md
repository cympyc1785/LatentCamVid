# GenDoP baseline 사용법

CinemaTraj `scripts/` 안의 GenDoP 관련 7개 스크립트가 어떻게 물려 있는지, 그리고
**우리 코퍼스(dynpose / vista4d)에 GenDoP 릴리즈 ckpt 를 돌려 우리 arm 과 한 표에 올리는**
방법. 코드 근거는 전부 파일:줄 로 적었다 — 읽고 확인할 수 있어야 쓸 수 있는 문서다.

문서 범위는 GenDoP baseline 하나다. `scripts/` 의 나머지(뱅크 굽기·릴 렌더·평가)는 여기서
다루지 않는다.

---

## 1. 배선 상태 한눈에

**리포는 두 벌인데 실제로 import 되는 건 pipeline 쪽이다.**

| 경로 | 무엇 | 쓰임 |
|---|---|---|
| `/data1/cympyc1785/pipeline/GenDoP` | fork `cympyc1785/GenDoP` | **추론이 import 하는 리포.** `core.models.LMM`, `core.utils.token_to_camera` |
| `camera_generation/models/GenDoP` | upstream `3DTopia/GenDoP` | `gendop_resample` / `caption_cameras_datadop` 기본 `--gendop_root` |

두 리포는 `segment_rigidbody_trajectories` 의 **하드코딩 `min_chunk_size` 가 다르다**
(vendored 10 vs pipeline 12, 인자로 못 바꾼다 — `caption_cameras_datadop.py:47-56`).
태깅 결과를 pipeline 세팅에 맞추려면 `--gendop_root /data1/cympyc1785/pipeline/GenDoP`.

ckpt: `/data1/cympyc1785/pipeline/GenDoP/checkpoints/{text_motion,text_rgbd}.safetensors`
env: 추론·태깅 `envs/GenDoP/bin/python` · eval_dir 변환 `envs/latentcam` · 점수 `envs/vista4d`

**GenDoP 리포는 0줄 수정한다.** EOS assert(`core/models.py:359`)와 zero-quaternion NaN
(`core/utils.py:209`)은 `gendop_release_infer.py` 쪽에서 monkey-patch / 가드로 막는다.

### 스크립트 7종

| 파일 | 역할 |
|---|---|
| `run_gendop_eval.py` | **드라이버.** `CORPORA` 표 + `--stage {inputs,infer,evaldir,score,all}` |
| `dynpose_gendop_inputs.py` | stage `inputs` — rgbd 조건 + 캡션 미러 루트 생성 |
| `gendop_release_infer.py` | stage `infer` — ckpt 로 npz 생성 |
| `gendop_preds_to_eval_dir.py` | stage `evaldir` — npz → latentcam eval 폴더 규약 |
| `gendop_release_eval.py` | 텍스트 왕복 일치도 (생성 궤적 재태깅 ↔ 원 태그) |
| `gendop_style_captions.py` | GenDoP 분포 정합 문장 (문장 탓 / 모델 탓 가르기) |
| `eval_dir_to_gendop_npz.py` | 역방향 — eval 폴더를 npz 로 |

---

## 2. 배치화 — **되어 있지 않다** (우리 경로 한정)

| | 배치 | 근거 |
|---|---|---|
| upstream `pipeline/GenDoP/eval_batch.py` | **O** — `batch_size=8`, `num_workers=4` | `:473` DataLoader, `:482-486` 배치 1회 `model.generate` |
| upstream `pipeline/GenDoP/eval.py` | X — 호출당 모델 재생성 | — |
| **우리 `gendop_release_infer.py`** | **X — 엔트리당 1회 generate** | `:311` `for kind, scene, name, cap_path in rows:` → `:349` `model.generate(...)` |

`eval_batch.py` 도 **후처리는 배치가 아니다** — `postprocess_and_save_tokens_batch`
(`:400-463`) 가 토큰을 `for i, token in enumerate(tokens)` 로 다시 한 개씩 돈다. 배치로
빨라지는 건 `generate` 뿐이다.

우리 쪽이 안 묶인 건 의도다 (모델 1회 로드로 전량 루프 — `gendop_release_infer.py` docstring).
**묶고 싶으면** `eval_batch.py:process_data` 의 DataLoader + `collate_to_conds` 를
`gendop_release_infer.py:311` 루프에 이식해야 하는데, 두 가지가 걸린다:

1. `--forbid_eos` 로짓 패치는 배치 안전하다(열 전체 -inf)지만, **EOS 를 허용하는 기본 모드
   (`pose_length=30`)에서는 배치 안의 엔트리마다 토큰 길이가 달라진다.** `decode_tokens` 가
   엔트리별 `usable` 길이를 따로 계산해야 한다 (지금 구조 그대로면 됨).
2. GenDoP 는 `generate_mode='sample'` 이고 seed 를 **루프 시작에 한 번** 심는다. 배치 크기를
   바꾸면 RNG 스트림이 달라져 **기존 런과 수치가 재현되지 않는다.** 배치화하면 그건 새 런이다.

---

## 3. 출력 크기 — rmax 안 곱한 **raw normalized** (원본 레포 방식)

### 3.1 원본 레포가 실제로 저장하는 것

배포 `eval.py` / `eval_batch.py` 는 scale 토큰을 계산은 하지만 **저장되는 JSON 에는 안 넣는다.**

```
eval_batch.py:438   scale = torch.exp(coords_scale / discrete_bins * 4 - 2)
eval_batch.py:441   camera_tokens = torch.cat([temp_traj, temp_instri], dim=1)   # scale 없음
eval_batch.py:443   camera_pose   = token_to_camera(camera_tokens, 512, 512)     # scale 없음
eval_batch.py:446-8 c2ws = np.array(camera_pose...);  c2ws[:, :3, 3] *= scale_value
eval_batch.py:452   # draw_json(c2ws, ...)        <- 유일한 c2ws 소비처. 주석 처리됨
eval_batch.py:462   pose_normalize(camera_pose, pred_pose_path, 49)   <- 저장은 camera_pose
```

`c2ws` 는 scale 이 걸린 **numpy 사본**이고 궤적 PNG(`draw_json`) 전용이다 — 그마저 주석이다.
디스크에 남는 `*_transforms_pred.json` 은 scale 이 **안 걸린** `camera_pose` 다.
`core/utils.py:206 token_to_camera` 도 token 7·8 을 fx,fy 로만 쓰고 token 9(scale)는 안 쓴다.
`pose_normalize`(`eval_batch.py:98-132`)는 이름과 달리 정규화를 하지 않는다 —
`sample_from_dense_cameras` 로 `camera_out_seq_len` 개를 리샘플해 JSON 으로 덤프할 뿐이다.

→ **원본 레포 출력 = 토큰이 디코드된 raw normalized 크기. rmax 도 scale 도 안 곱한다.**

### 3.2 우리 경로에서 그걸 재현하는 법

우리 `gendop_release_infer.py:247` 은 npz 에 **scale 을 곱해서** 넣고 곱한 값을 `scale` 키로
같이 저장한다. 그래서 되나누면 정확히 배포 판본이 된다. 크기에 손대는 손잡이는 **둘**이고,
**둘 다 꺼야** 원본 레포 방식이다:

| 손잡이 | 기본값 | 원본 레포 방식 | 하는 일 |
|---|---|---|---|
| `--scale_token` | on | **`--no_scale_token`** | npz 의 `scale` 을 되나눈다 (`gendop_preds_to_eval_dir.py:145`) |
| `--rescale` | on | **`--raw`** (드라이버) / `--no_rescale` (변환기) | 켜져 있으면 `pred_rel[:,:3,3] *= r_gt/r_pred` 로 **rmax 를 GT 에서 빌린다** (`:162`) |

```bash
# 드라이버로
python exec/run_gendop_eval.py --corpus <C> --stage evaldir --raw --no_scale_token
python exec/run_gendop_eval.py --corpus <C> --stage score   --raw --no_scale_token --gpu 0

# 변환기 직접
python eval/gendop_preds_to_eval_dir.py ... --no_rescale --no_scale_token
```

산출물 폴더는 접미사로 갈린다 — `eval_dir_<arm>_raw_noscale` (`run_gendop_eval.py:164-172`).
기본값(on/on)은 **예전 런과 비트동일하게 두려고** 남긴 것이지 권장값이 아니다.
실측 차: `--no_scale_token` 은 median ×0.50, 분포 폭 13.6배 / `--raw` 를 안 주면 d121 rgbd
p49 기준 median ×2.48 확대였다 — 충돌률·subject_in_frame 처럼 **절대 크기에 반응하는 지표는
GenDoP 이 아니라 GT 를 재게 된다.**

> `--rescale` 은 stage `evaldir` 에서만 갈린다. `pred_*` npz(=infer 산출물)는 영향 없으므로
> infer 를 다시 돌릴 필요가 없다.

---

## 4. 텍스트 2종

### (A) 우리 모델이 실제로 받은 문장 — **기본값, 배선 완료**

`eval_testset.py` 가 쓴 `<eval_dir>/test/<prefix>_<scene>_<idx>_caption.json` 을 그대로 읽는다.
CLaTr 은 텍스트 임베딩과 궤적을 맞춰 보므로 두 arm 이 같은 문장을 봐야 **궤적 차이만 남는
짝지은 비교**가 된다.

```bash
# CORPORA[C]["ours"][CORPORA[C]["ref"]] 가 기본 text_dir
python exec/run_gendop_eval.py --corpus <C> --stage infer --arm gendop_text --gpu 0
```

다른 캡션으로 갈아끼우려면 `--text_dir <같은 test/ 모양 폴더> --text_tag <접미사>`.
(`--text_tag` 를 안 주면 기본 캡션 산출물을 덮어쓴다 — `run_gendop_eval.py:307` assert 가 막는다.)

### (B) GT 궤적에서 직접 tagging + captioning — **스크립트는 있고, 드라이버엔 안 물려 있다**

`caption_cameras_datadop.py` 가 DataDoP 세 단계를 그대로 태운다 (문구를 새로 짜지 않는다):
① `segment_rigidbody_trajectories` (translation 27 × angular 7 패턴) → ② outline → LLM →
`Movement` → ③ 영상 16프레임 4×4 그리드 + Movement → LLM → `Detailed Interaction` /
`Concise Interaction` (`:494-495`).

```bash
# ① 태깅만 — 방향 라벨이 preset 이름과 맞는지 먼저 본다 (LLM 불필요)
/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python fit/caption/caption_cameras_datadop.py \
    --sets latentcam --no_llm \
    --gendop_root /data1/cympyc1785/pipeline/GenDoP \
    --latentcam_root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
    --latentcam_split .../latentcam_dynpose_d200/seg_list_dynpose_s91_test.txt

# ② 전량 캡션 (vLLM Qwen3-VL 을 먼저 올린다 — 끝나면 내린다)
#    --no_llm 을 빼고 같은 명령
```

출력은 **코퍼스 안**에 떨어진다: `<latentcam_root>/<dataset>/<scene>/da3/captions_gendop/<idx>_caption.json`
(`:328-331`). 이게 `gendop_release_infer.py:106-117 collect()` 의 split 모드가 찾는 경로와
정확히 같은 모양이라, **`--text_from_eval_dir` 를 빼고 `--root` 를 코퍼스로 주면 그대로 물린다:**

```bash
CUDA_VISIBLE_DEVICES=0 /data1/cympyc1785/miniconda3/envs/GenDoP/bin/python \
  eval/gendop_release_infer.py \
    --resume /data1/cympyc1785/pipeline/GenDoP/checkpoints/text_motion.safetensors \
    --root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
    --split .../seg_list_dynpose_s91_test.txt \
    --text_key 'Concise Interaction' \
    --out results/<날짜>_gendop_<C>/pred_gendop_text_gt   # ← _gt 로 (A) 를 안 덮는다
```

규약 세 가지(`caption_cameras_datadop.py` docstring 에 근거와 함께):
- **좌표계** — DataDoP 는 OpenGL c2w. latentcam `target_poses.npz["extrinsics"]` 는
  **OpenCV w2c** 라 `inv(M) @ diag(1,-1,-1,1)` 로 뒤집는다 (실측 max|diff| 2.71e-07).
- **프레임 수** — 120 pose 로 slerp 리샘플 (`--num_poses 120`, 기본값). 49 를 그대로 넣으면
  smoothing window 가 궤적의 1/3 을 덮어 전부 1-segment 가 된다.
- **스케일** — 리스케일하지 않는다 (D=1). MonST3R 게이지가 이미 shot 단위 정규화라
  depth 로 또 나누면 산포만 커진다.

**한계: zoom 은 캡션에 안 들어간다.** DataDoP 어휘는 translation+rotation 뿐이라 위치 고정 +
focal 램프는 "static" 으로 적힌다. focal 비는 `_tag.json` 의 `focal_ratio` 에만 남는다.

**드라이버 gap:** `run_gendop_eval.py:stage_infer` 는 `--text_from_eval_dir` 를 **항상** 넘기고
`--text_dir` 는 `<dir>/test/<prefix>_<scene>_<idx>_caption.json` 모양만 받는다. (B) 는 파일명
규약이 `<idx>_caption.json` 이라 드라이버로는 못 간다 — 위처럼 `gendop_release_infer.py` 를
직접 부르거나, 미러 스텝을 하나 두어야 한다. 그 뒤 `evaldir`/`score` 는
`--pred_dir` 만 바꿔 `gendop_preds_to_eval_dir.py` 를 직접 부르면 된다.

---

## 5. 코퍼스 추가 (지금 d200 은 **없다**)

`run_gendop_eval.py:79` `CORPORA` 에 `vista_d121` / `vista_d121_snowboard` / `dynpose_d137`
세 개뿐이다. **d200 항목이 없다.** 추가는 표에 dict 하나:

```python
"dynpose_d200": dict(
    corpus="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200",
    split_name="seg_list_dynpose_s91_test.txt",          # 5,144 entry / 1,017 scene
    eval_data="/data1/cympyc1785/data/DynPose-LBM/eval_data",
    prefix="dynpose",
    cloud_root=path.join(HERE, "out_dynpose"),
    out=path.join(HERE, "results", "<날짜>_gendop_d200"),
    depth_norm="median",
    ours={...EVAL_MY + "20260918_140904_dynpose_d200_da3__last", ...},
    ref="<위 중 하나>",
),
```

`depth_norm` 판정 기준은 **frame0 depth 의 median 이 GenDoP 학습 대역에 있는가**
(MonST3R ~0.32, 릴리즈 `text_rgbd/case1_depth.npy` 0.266~0.722 median 0.323).
`dynpose_gendop_inputs.py` 요약표의 **`min` 행을 median 으로 읽지 말 것** — 그 행은 프레임
최솟값의 통계다 (실제로 한 번 틀렸다).

`eval_data` 규약이 두 스크립트에서 다르다 (`run_gendop_eval.py:73-77`):
`dynpose_gendop_inputs.py:74` 는 `<eval_data>/recon_and_seg/<scene>` 로 붙이고,
`eval_subject_in_frame.py` 는 `scene_graph/io.py:88` 이 "eval_data" 를 **다시** 붙인다.
그래서 표에는 하나만 적고 score 쪽은 그 부모를 쓴다.

---

## 6. 손잡이 요약

| 플래그 | 기본 | 언제 바꾸나 |
|---|---|---|
| `--pose_length` | 30 (릴리즈 학습 길이) | **49** = 우리 코퍼스 길이. 모델이 직접 49 스텝을 뽑는다. 리샘플이 항등이 되고 산출물이 `_p49` 로 갈린다 |
| `--forbid_eos` | auto (`pose_length != 30` 이면 on) | 30 을 넘겨 요구하면 모델이 EOS 를 내고 `core/models.py:359 assert` 가 죽는다. 로짓에서 EOS 열만 -inf |
| `--strict_pose_length` | on | `--pose_length 49` 면 드라이버가 `--no_strict_pose_length` 를 붙인다. 안 그러면 짧게 끝난 게 전부 static 폴백 |
| `--resample` | `index_pick` | `pose_length == n_poses` 면 둘 다 항등. 30→49 일 때만 `gendop_slerp` 와 갈린다 |
| `--overwrite` | off | **중단 후 재개는 반드시 `--overwrite`.** seed 를 루프 시작에 한 번 심으므로 중간부터 이으면 앞뒤가 다른 RNG 스트림에서 나온다 — 그 arm 은 config 로 재현이 안 된다 |
| `--limit N` | — | smoke. split 앞 N entry 만 |

## 7. 아직 안 된 것

- `CORPORA` 에 **d200 항목 없음** (§5).
- d200 코퍼스에 **`da3/captions_gendop/` 없음** — 텍스트 (B) 를 d200 에 돌린 적이 없다.
- 텍스트 (B) 가 **드라이버에 안 물려 있음** (§4B gap).
- 우리 추론 경로 **배치화 안 됨** (§2). 하면 기존 런과 수치가 안 맞는다.
