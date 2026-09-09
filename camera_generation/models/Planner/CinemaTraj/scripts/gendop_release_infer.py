"""GenDoP release ckpt 로 우리 캡션(`captions_gendop`) 186개 → 카메라 궤적 생성.

WHY: Vista4D eval 카메라에 단 캡션으로 GenDoP(DataDoP 저자 릴리즈 모델)가 어떤 궤적을 내는지
     — 우리 latentcam arm 들과 같은 텍스트 축 위의 외부 baseline. 텍스트는 우리가 실측 카메라에서
     뽑은 것이므로, 생성 궤적을 같은 분절기로 재태깅해 원 카메라 태그와 대조하면 "텍스트 왕복
     일치도"가 된다 (평가는 `gendop_release_eval.py`).

GenDoP 리포는 **0줄 수정** — `core.models.LMM` / `core.utils.token_to_camera` 를 그대로 import
하고, `eval.py:process_data` 의 토큰 → c2w 디코드 로직만 옮겨 적는다 (모델 1회 로드로 186개를
돌리기 위함; eval.py 는 호출당 모델을 다시 만든다). tyro CLI 는 안 쓰고 `config_defaults['ArAE']`
를 직접 복사한다.

출력 (`--out` 밑):
    <set>__<scene>__<name>.npz   c2w (P,4,4) 모델 native 포즈 (P = `--pose_length`, 기본 30.
                                 slerp 업샘플 안 함 — 120 슬러프는 roll 아티팩트를 만든다),
                                 scale (스칼라), degenerate (bool: 토큰 길이가 안 맞아 eval.py
                                 의 static 폴백이 쓰인 경우), n_poses (실제 포즈 수)
    config.json                  실행 설정 전량 + ckpt + text_key

`--pose_length` — **모델이 직접 P 포즈를 뽑게 하는 손잡이**. `core/options.py:39 pose_length`
가 `core/models.py:331 max_new_tokens = 10*pose_length+1` 과 `:333 num_tokens` 를 정하고,
`prefix_allowed_tokens_fn` 이 EOS 를 `1+10N` 위치에서만 허용하므로 이 값이 곧 생성 길이 상한이다.
우리 코퍼스에 맞추려면 `--pose_length 49`. 기본 30 은 릴리즈 학습 길이(`README.md:207`)라
인자를 안 주면 예전과 bit-identical 이다.

  주의: 상한일 뿐이라 모델이 30 에서 EOS 를 내면 토큰이 짧게 끝난다. **그런데 EOS 가 나오면
  `models.py:358` 의 `output_ids - 3` 이 EOS(id 2) 를 -1 로 만들고 `:359`
  `assert np.all(tokens >= 0)` 이 터진다** — 우리 `decode_tokens` 는 그보다 하류라 못 막는다
  (d121 875 entry 중 ~170 번째에서 실측). 그래서 `--forbid_eos` 로 EOS 를 로짓에서 지운다
  (§forbid_eos). 그래도 짧게 끝난 경우 `decode_tokens` 는 static 폴백 대신 **나온 만큼
  디코드**하고 `n_poses` 에 실제 길이를 적는다 (`--strict_pose_length` 를 주면 예전처럼 폴백).
  리샘플로 49 를 만드는
  경로(`gendop_preds_to_eval_dir.py --src_poses 30 --n_poses 49`, `eval_batch.py:462
  pose_normalize(..., 49)`)와 달리 여기서는 궤적 자체가 49 스텝으로 생성된다.

`--forbid_eos` — EOS 를 로짓에서 -inf 로 지워 `max_new_tokens` 까지 반드시 생성하게 한다.
`core/models.py:324-329 prefix_allowed_tokens_fn` 은 `1+10N` 위치마다 `eos_token_id=2` 를 후보에
넣는데, 학습 길이(30)를 넘겨 요구하면 모델이 실제로 EOS 를 내고 `:359` assert 가 죽는다.
GenDoP 리포는 **한 줄도 안 고친다** — `core.utils.monkey_patch_transformers()` 가 이미 갈아끼운
`PrefixConstrainedLogitsProcessor.__call__` 위에 한 겹 더 씌워 EOS 열만 -inf 로 만든다.
후보는 `range(3, 260)` 257개가 항상 남아 있어 "빈 후보" 는 생기지 않는다.
기본값은 `pose_length == 30` 이면 off (예전 런과 bit-identical), 그 외엔 on.
명시적으로 `--forbid_eos` / `--no_forbid_eos` 를 주면 그게 이긴다.

§zero_quat — 회전 4토큰이 전부 bin 128 이면 quaternion 이 `(0,0,0,0)` 이 되고 `core/utils.py:209
quaternion_to_matrix` 가 0 노름으로 나눠 회전 3x3 을 통째로 NaN 으로 만든다. NaN 은 npz 를 지나
하류 렌더러까지 살아남아 `torch.inverse` 가 singular 로 죽는다 (`--pose_length 49` text arm 에서
875 중 3 entry, 각 1~2 프레임). `decode_tokens` 가 **직전 프레임 회전으로 때우고** 그 개수를
npz `n_zero_quat` / config `zero_quat_entries` / 요약 `zero_quat` 에 남긴다. 조용히 고치지 않는다.

`--cond_mode depth+image+text` 는 릴리즈의 **text_rgbd** ckpt 용 분기다. 텍스트에 더해 생성 영상
frame0 의 RGB 와 MonST3R depth 를 조건으로 준다 (`eval.py:311-313` 과 같이 `num_cond_tokens`
77 -> 591 = 77 텍스트 + 257 이미지 + 257 depth). RGB/depth 는 `eval_data/gen/<scene>/<name>/
monst3r/{frame_0000.png, frame_depth_0000.npy}` — MonST3R 가 자기 입력 해상도로 리사이즈해 쓴
프레임이라 depth 와 픽셀 정렬이 이미 맞다 (mp4 에서 뽑으면 해상도가 어긋난다). depth 는 **그대로**
넣는다: `eval.py:145 standard_depth` 는 center-crop/zero-pad 만 하고 정규화를 안 하며, GenDoP 의
`assets/examples/text_rgbd` 예시 depth(case1 0.266~0.722, case2 0.035~0.507)와 우리 MonST3R
출력(0.044~0.883)이 같은 게이지다 — 둘 다 같은 MonST3R 파이프라인 산출물이라 그렇다.
기본값 `--cond_mode text` 는 기존 경로 그대로다 (bit-identical).

env: GenDoP.  예시:
    CUDA_VISIBLE_DEVICES=1 python scripts/gendop_release_infer.py \
        --resume /data1/cympyc1785/pipeline/GenDoP/checkpoints/text_motion.safetensors \
        --text_key Movement --out results/20260828_gendop_eval/text_motion

    CUDA_VISIBLE_DEVICES=2 python scripts/gendop_release_infer.py \
        --resume /data1/cympyc1785/pipeline/GenDoP/checkpoints/text_rgbd.safetensors \
        --cond_mode depth+image+text --text_key 'Concise Interaction' \
        --out results/20260829_gendop_rgbd/text_rgbd
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import path, makedirs

import cv2
import numpy as np
import torch

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
GENDOP_PIPE = "/data1/cympyc1785/pipeline/GenDoP"
EVAL_DATA = "/data1/cympyc1785/LatentCamVid/DATA/Vista4D-Eval-Data/eval_data"
# vista4d gen 114편 MonST3R frame0 depth 의 median-of-medians (실측). `--depth_norm median`
# 의 기본 목표값 — GenDoP 예시 depth(case1 0.323 / case2 0.219)와 같은 대역이다.
MONST3R_MEDIAN = 0.3177
if GENDOP_PIPE not in sys.path:
    sys.path.insert(0, GENDOP_PIPE)

from core.options import config_defaults                              # noqa: E402
from core.models import LMM                                           # noqa: E402
from core.utils import monkey_patch_transformers, token_to_camera     # noqa: E402
from safetensors.torch import load_file                               # noqa: E402


def collect(root: str, split: str = None):
    """(set, scene, name, caption_path) — captions_gendop 에 있는 것 전부.

    `split` 이 주어지면 latentcam 코퍼스 모드다: `<root>/<dataset>/<scene>/da3/captions_gendop/`
    에서 split 파일이 열거한 엔트리만 (순서도 split 순서 그대로) 집는다. eval 셋만 돌리려고
    디렉토리 전량 glob 대신 리스트를 쓴다 — 학습셋 캡션이 섞이면 조용히 오염된다.
    """
    rows = []
    if split is not None:
        with open(split, encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
        for line in lines:
            dataset, scene, index = line.split("/")
            cap = path.join(root, dataset, scene, "da3", "captions_gendop",
                            f"{index}_caption.json")
            if path.exists(cap):
                rows.append(("latentcam", scene, index, cap))
        return rows
    for kind, sub in (("cameras", "cameras"), ("recon", "recon_and_seg")):
        for p in sorted(glob(path.join(root, sub, "*", "captions_gendop", "*_caption.json"))):
            scene = path.basename(path.dirname(path.dirname(p)))
            name = path.basename(p)[: -len("_caption.json")]
            rows.append((kind, scene, name, p))
    return rows


def letterbox(tensor, target_height=512, target_width=512):
    """종횡비를 유지해 축소한 뒤 zero-pad. `--rgbd_fit letterbox` 전용.

    WHY: `eval.py` 의 center-crop 은 720x1280 을 넣으면 가운데 512x512 만 남기고 가로 60% 를
    버린다. 반면 GenDoP 자신의 예시(`assets/examples/text_rgbd`)는 208x512 / 288x512 로
    **레터박스된** 배열이다. 이 차이는 미관 문제가 아니라 조건 신호를 바꾼다 — 실측:
    depth 임베딩은 depth **내용**에 거의 반응하지 않고 zero 띠 기하에만 반응한다
    (같은 288x512 띠의 다른 씬 depth: d_emb 0.02, 토큰 0/301 변화 / 띠가 없는 512x512
    center-crop: d_emb 11.44, 토큰 83/301). 즉 crop 은 학습에서 못 본 "띠 없음"을 신호로
    주입한다. letterbox 는 화각 전체를 보존하면서 띠도 분포 안에 둔다.
    """
    scale = min(target_height / tensor.shape[1], target_width / tensor.shape[2])
    if scale < 1.0:
        size = (max(1, int(round(tensor.shape[1] * scale))),
                max(1, int(round(tensor.shape[2] * scale))))
        tensor = torch.nn.functional.interpolate(tensor[None], size, mode="bilinear",
                                                 align_corners=False)[0]
    padded = torch.zeros((tensor.shape[0], target_height, target_width), dtype=torch.float32)
    top = (target_height - tensor.shape[1]) // 2
    left = (target_width - tensor.shape[2]) // 2
    padded[:, top:top + tensor.shape[1], left:left + tensor.shape[2]] = tensor
    return padded


def standard_image(rgb_path, target_height=512, target_width=512, fit="crop"):
    """`eval.py:110 standard_image` 그대로 — BGR->RGB, /255, center-crop, zero-pad.

    `fit="letterbox"` 면 crop 대신 `letterbox()`. 기본 `crop` 이 기존 동작(bit-identical).
    """
    image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    image = image[..., [2, 1, 0]]
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    if fit == "letterbox":
        return letterbox(tensor, target_height, target_width)
    if tensor.shape[1] > target_height:
        start = (tensor.shape[1] - target_height) // 2
        tensor = tensor[:, start:start + target_height, :]
    if tensor.shape[2] > target_width:
        start = (tensor.shape[2] - target_width) // 2
        tensor = tensor[:, :, start:start + target_width]
    if tensor.shape[1] < target_height or tensor.shape[2] < target_width:
        padded = torch.zeros((3, target_height, target_width), dtype=torch.float32)
        top = (target_height - tensor.shape[1]) // 2
        left = (target_width - tensor.shape[2]) // 2
        padded[:, top:top + tensor.shape[1], left:left + tensor.shape[2]] = tensor
        tensor = padded
    return tensor


def standard_depth(depth_path, target_height=512, target_width=512,
                   depth_norm="none", target_median=MONST3R_MEDIAN, fit="crop"):
    """`eval.py:145 standard_depth` 그대로 + 게이지 정합 옵션.

    `depth_norm="none"` 이 기존 동작(정규화 없음, bit-identical)이다. MonST3R 출력끼리는
    그래도 되지만 **DA3 depth 를 넣을 때는 안 된다** — 실측 게이지가 다르다:
      MonST3R frame0 depth (vista4d gen 114편)  median-of-medians 0.318  (0.077~1.027)
      GenDoP 예시 case1/case2                    0.323 / 0.219
      dynpose DA3 depth (val 22편)               median-of-medians 2.197 (0.411~20.547)
    `depth_norm="median"` 은 프레임 median 을 `target_median` 으로 맞춘다 (스칼라 곱).
    모양은 그대로 두고 크기만 학습 대역으로 옮기는 최소 개입이다.
    """
    depth = np.load(depth_path).astype(np.float32)
    if depth_norm == "median":
        med = float(np.nanmedian(depth[np.isfinite(depth) & (depth > 0)]))
        depth = depth * (target_median / max(med, 1e-9))
    tensor = torch.from_numpy(depth).unsqueeze(0).float()
    if fit == "letterbox":
        return letterbox(tensor, target_height, target_width)
    if tensor.shape[1] > target_height:
        start = (tensor.shape[1] - target_height) // 2
        tensor = tensor[:, start:start + target_height, :]
    if tensor.shape[2] > target_width:
        start = (tensor.shape[2] - target_width) // 2
        tensor = tensor[:, :, start:start + target_width]
    if tensor.shape[1] < target_height or tensor.shape[2] < target_width:
        padded = torch.zeros((1, target_height, target_width), dtype=torch.float32)
        top = (target_height - tensor.shape[1]) // 2
        left = (target_width - tensor.shape[2]) // 2
        padded[:, top:top + tensor.shape[1], left:left + tensor.shape[2]] = tensor
        tensor = padded
    return tensor


def rgbd_paths(scene, name, args):
    """(rgb, depth) 경로. 둘 중 하나라도 없으면 None — 그 엔트리는 건너뛴다."""
    base = path.join(args.gen_root, scene, name, args.monst3r_sub)
    rgb, depth = path.join(base, args.rgb_name), path.join(base, args.depth_name)
    return (rgb, depth) if path.exists(rgb) and path.exists(depth) else None


def decode_tokens(token, opt, strict=True):
    """`eval.py:process_data` 의 토큰 -> c2w 디코드 (native 포즈, scale 적용).

    `strict=False` 면 길이가 `pose_length*10` 에 못 미쳐도 **10의 배수만큼은 살려서** 디코드한다.
    `--pose_length 49` 처럼 학습 길이보다 길게 요구했을 때 모델이 30 에서 EOS 를 내는 게
    정상 동작이라, 그걸 전부 static 폴백으로 버리면 "49 를 요구했더니 전부 정지"가 된다.

    반환 `n_zero_quat` 은 회전이 0 quaternion 으로 나와 직전 프레임 회전으로 때운 프레임 수다
    (§zero_quat). 0 이 아니면 그 엔트리는 "모델이 회전을 못 정한 프레임"을 가진 것이다.
    """
    degenerate = token[:-1].shape[0] != opt.pose_length * 10
    usable = (token[:-1].shape[0] // 10) * 10
    if degenerate and (strict or usable == 0):
        # eval.py 와 같은 static 폴백 (기록만 남기고 버리지 않는다)
        token = (torch.tensor([256, 128, 128, 128, 128, 128, 128, 36, 64, 60])
                 / 256 * opt.discrete_bins).repeat(opt.pose_length)
        coords = token.reshape(-1, 10).float()
    else:
        coords = token[:usable].reshape(-1, 10).float()
    bins = opt.discrete_bins
    temp_traj = coords[:, :7] / (0.5 * bins) - 1
    # quaternion 4 토큰이 전부 bin 128 이면 q=(0,0,0,0) 이 되고, GenDoP `core/utils.py:209
    # quaternion_to_matrix` 가 0 노름으로 나눠 회전 3x3 을 통째로 NaN 으로 만든다. 그 NaN 은
    # npz 를 지나 렌더러까지 살아남아 `torch.inverse` 가 singular 로 죽는다 (d121 875 entry 중
    # text_p49 3건 실측). 리포는 한 줄도 안 고치므로 여기서 **직전 프레임 회전을 이어붙인다** —
    # 0 quaternion 은 "회전 정보 없음"이지 "회전 0" 이 아니라서 identity 로 박으면 궤적이 튄다.
    zero_q = (temp_traj[:, :4].norm(dim=1) < 1e-6).nonzero().flatten().tolist()
    for i in zero_q:
        temp_traj[i, :4] = (temp_traj[i - 1, :4] if i else
                            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=temp_traj.dtype))
    temp_instri = coords[:, 7:] / (bins / 10)
    scale = torch.exp(coords[:, 9] / bins * 4 - 2)
    camera_tokens = torch.cat([temp_traj, temp_instri], dim=1).unsqueeze(0)
    camera_pose = token_to_camera(camera_tokens, 512, 512)
    c2ws = np.asarray(camera_pose[0, :, :12].cpu(), dtype=np.float64).reshape(-1, 3, 4)
    c2ws[:, :3, 3] *= float(scale[0])
    out = np.tile(np.eye(4), (c2ws.shape[0], 1, 1))
    out[:, :3, :4] = c2ws
    assert np.all(np.isfinite(out)), "c2w 에 NaN/Inf 가 남았다 (zero_quat 가드 밖의 경로)"
    return out, float(scale[0]), degenerate, len(zero_q)


def forbid_eos_in_logits(eos_token_id):
    """`monkey_patch_transformers()` 가 깐 패치 **위에** 한 겹 더 씌워 EOS 열을 -inf 로 만든다.

    GenDoP `core/models.py:324-329` 는 `1+10N` 위치마다 EOS 를 후보에 넣는데, 학습 길이(30)를
    넘겨 요구하면 모델이 실제로 EOS 를 뽑고 `:358` 의 `output_ids - 3` 이 그걸 -1 로 만들어
    `:359 assert np.all(tokens >= 0)` 이 죽는다. GenDoP 리포는 한 줄도 안 고치므로 여기서 막는다.
    후보 `range(3, 260)` 257개는 그대로라 "빈 후보" 는 생기지 않는다.
    """
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
    base = PrefixConstrainedLogitsProcessor.__call__

    def __call__(self, input_ids, scores):
        out = base(self, input_ids, scores)
        out[:, eos_token_id] = float("-inf")
        return out

    PrefixConstrainedLogitsProcessor.__call__ = __call__
    print(f"[gendop] forbid_eos: EOS(id={eos_token_id}) 로짓을 -inf 로 고정 "
          f"(max_new_tokens 까지 생성)", flush=True)


def main(args):
    opt = config_defaults["ArAE"]
    opt.cond_mode = args.cond_mode
    opt.pose_length = args.pose_length   # 30 = 릴리즈 학습 길이 (기본), 49 = 우리 코퍼스 길이
    # eval.py:311-313 과 같은 매핑. 591 = 77(text) + 257(image) + 257(depth).
    opt.num_cond_tokens = 591 if args.cond_mode == "depth+image+text" else 77

    monkey_patch_transformers()
    # None = auto: 학습 길이(30) 그대로면 예전과 bit-identical, 넘겨 요구할 때만 EOS 를 막는다.
    args.forbid_eos = (opt.pose_length != 30) if args.forbid_eos is None else args.forbid_eos
    if args.forbid_eos:
        forbid_eos_in_logits(opt.eos_token_id)
    # kiui.seed_everything 은 함수 안 import 를 거부한다 — 같은 일을 직접 한다.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = LMM(opt)
    ckpt = load_file(args.resume, device="cpu")
    model.load_state_dict(ckpt, strict=False)
    device = torch.device("cuda")
    model = model.half().eval().to(device)
    print(f"[gendop] loaded {args.resume}")

    rows = collect(args.root, args.split)
    if args.kinds:
        rows = [r for r in rows if r[0] in set(args.kinds)]
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if r[1] in wanted or f"{r[1]}/{r[2]}" in wanted]
    makedirs(args.out, exist_ok=True)

    done, skipped, degen, no_rgbd, lengths, zero_quat = 0, 0, [], [], {}, []
    no_caption = []      # --text_from_eval_dir 접두사/이름 규약이 어긋난 엔트리
    for kind, scene, name, cap_path in rows:
        out_path = path.join(args.out, f"{kind}__{scene}__{name}.npz")
        if path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        rgbd = None
        if args.cond_mode == "depth+image+text":
            rgbd = rgbd_paths(scene, name, args)
            if rgbd is None:      # MonST3R 가 아직 안 돈 엔트리 — 조용히 넘기지 말고 세어둔다
                no_rgbd.append(f"{scene}/{name}")
                skipped += 1
                continue
        if args.text_from_eval_dir:
            # latentcam(k6) 이 실제로 받은 문장으로 갈아끼운다 — `eval_testset.py` 가 쓴
            # `test/<name>_caption.json` 이 곧 그 모델의 입력이다. 파일명 규약이 달라
            # (`<idx>_caption.json` vs `vista4d_<scene>_<idx>_caption.json`) 여기서 다시 만든다.
            # 접두사는 코퍼스마다 다르다 (vista4d / dynpose / ...). 기본값이 vista4d 라
            # 기존 호출은 그대로다. 틀리면 전 엔트리가 조용히 `skipped` 로 빠지므로
            # 아래에서 따로 세어 요약에 찍는다.
            cap_path = path.join(args.text_from_eval_dir, "test",
                                 f"{args.eval_dir_prefix}_{scene}_{name}_caption.json")
            if not path.exists(cap_path):
                no_caption.append(f"{scene}/{name}")
                skipped += 1
                continue
        with open(cap_path, encoding="utf-8") as file:
            text = json.load(file)[args.text_key]
        if rgbd is None:
            conds = [text]
        else:
            # eval.py:196-215 의 조립 순서 그대로: [[text], rgb(1,3,H,W), depth(1,1,H,W)].
            # depth 는 cpu 로 둔다 — `models.py:211` 이 encoder device 로 직접 옮긴다.
            rgb = standard_image(rgbd[0], opt.target_height, opt.target_width,
                                 args.rgbd_fit).to(device)
            depth = standard_depth(rgbd[1], opt.target_height, opt.target_width,
                                   args.depth_norm, args.depth_target_median, args.rgbd_fit)
            conds = [[text], rgb.expand(1, -1, -1, -1), depth.expand(1, -1, -1, -1)]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            tokens = model.generate(conds, max_new_tokens=opt.test_max_seq_length, clean=True)
        token = tokens[0]
        token = torch.as_tensor(np.asarray(token)) if not torch.is_tensor(token) else token.cpu()
        c2w, scale, bad, n_zq = decode_tokens(token, opt, strict=args.strict_pose_length)
        if n_zq:
            zero_quat.append(f"{scene}/{name}:{n_zq}")
        np.savez(out_path, c2w=c2w, scale=scale, degenerate=bad,
                 n_zero_quat=n_zq, n_poses=int(c2w.shape[0]),
                 text=np.array(text), caption_path=np.array(cap_path),
                 rgb_path=np.array("" if rgbd is None else rgbd[0]),
                 depth_path=np.array("" if rgbd is None else rgbd[1]))
        if bad:
            degen.append(f"{scene}/{name}")
        lengths[int(c2w.shape[0])] = lengths.get(int(c2w.shape[0]), 0) + 1
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(rows)}", flush=True)

    with open(path.join(args.out, "config.json"), "w", encoding="utf-8") as file:
        json.dump({"resume": args.resume, "text_key": args.text_key, "seed": args.seed,
                   "root": args.root, "split": args.split,
                   "text_from_eval_dir": args.text_from_eval_dir,
                   "eval_dir_prefix": args.eval_dir_prefix,
                   "pose_length": opt.pose_length,
                   "strict_pose_length": args.strict_pose_length,
                   "forbid_eos": args.forbid_eos,
                   "n_poses_hist": {str(k): v for k, v in sorted(lengths.items())},
                   "cond_mode": opt.cond_mode,
                   "num_cond_tokens": opt.num_cond_tokens, "n_entries": len(rows),
                   "gen_root": args.gen_root, "monst3r_sub": args.monst3r_sub,
                   "rgb_name": args.rgb_name, "depth_name": args.depth_name,
                   "depth_norm": args.depth_norm,
                   "depth_target_median": args.depth_target_median,
                   "rgbd_fit": args.rgbd_fit,
                   "depth_normalization": (
                       "none (MonST3R 게이지 그대로, eval.py standard_depth 동일)"
                       if args.depth_norm == "none" else
                       f"frame median -> {args.depth_target_median} 스칼라 곱"),
                   "missing_rgbd": no_rgbd,
                   # 0 quaternion 이 나와 직전 프레임 회전으로 때운 엔트리 "<scene>/<name>:<프레임수>"
                   "zero_quat_entries": zero_quat,
                   "convention": "GenDoP c2w, translation *= exp-scale 토큰 (DataDoP 게이지)"},
                  file, ensure_ascii=False, indent=2)
    print(f"\n{'entries':<14}{len(rows)}")
    print(f"{'cond_mode':<14}{opt.cond_mode}  (num_cond_tokens={opt.num_cond_tokens})")
    print(f"{'written':<14}{done}")
    print(f"{'skipped':<14}{skipped}")
    print(f"{'no_rgbd':<14}{len(no_rgbd)}  {no_rgbd[:5]}")
    print(f"{'no_caption':<14}{len(no_caption)}  {no_caption[:5]}"
          f"{'   <- --eval_dir_prefix 확인' if no_caption else ''}")
    print(f"{'degenerate':<14}{len(degen)}  {degen[:5]}")
    print(f"{'zero_quat':<14}{len(zero_quat)}  {zero_quat[:5]}")
    print(f"{'pose_length':<14}{opt.pose_length}  strict={args.strict_pose_length}"
          f"  forbid_eos={args.forbid_eos}")
    print(f"{'n_poses hist':<14}{dict(sorted(lengths.items()))}")
    print(f"{'out':<14}{args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True)             # release safetensors
    parser.add_argument("--text_key", default="Movement")      # text_motion ckpt 는 Movement 문체
    parser.add_argument("--root", default=EVAL_DATA)
    # latentcam 코퍼스(LBM-Lite 합성 target)를 돌릴 때만: 이 split 파일이 열거한 엔트리만 쓴다.
    parser.add_argument("--split", default=None)
    parser.add_argument("--out", required=True)
    # 이 latentcam eval 폴더의 `test/vista4d_<scene>_<idx>_caption.json` 을 문장 원본으로 쓴다
    # — k6 가 **실제로 받은 그 문장**이다. 주면 `captions_gendop` 문장 대신 이걸 넣어 돌린다
    # (텍스트 축만 바꾼 짝지은 arm). 안 주면 기존 동작 그대로.
    parser.add_argument("--text_from_eval_dir", default=None)
    # 그 폴더 안 파일명 접두사. eval 폴더는 코퍼스 이름으로 접두사를 붙인다
    # (vista4d_<scene>_<idx>_caption.json / dynpose_<scene>_<idx>_caption.json).
    parser.add_argument("--eval_dir_prefix", default="vista4d")
    # text_rgbd ckpt 분기. 'text' 는 기존 동작 그대로 (bit-identical).
    parser.add_argument("--cond_mode", default="text",
                        choices=["text", "depth+image+text"])
    # depth+image+text 일 때 RGB/depth 를 찾는 곳: <gen_root>/<scene>/<name>/<monst3r_sub>/
    parser.add_argument("--gen_root", default=path.join(EVAL_DATA, "gen"))
    parser.add_argument("--monst3r_sub", default="monst3r")
    # depth 게이지 정합. 'none' 이 기존 동작(MonST3R 출력끼리는 이게 맞다). DA3 depth 처럼
    # 게이지가 다른 걸 넣을 때만 'median'.
    parser.add_argument("--depth_norm", default="none", choices=["none", "median"])
    parser.add_argument("--depth_target_median", type=float, default=MONST3R_MEDIAN)
    # RGB/depth 를 512x512 에 맞추는 방식. 'crop' 이 기존 동작(eval.py 그대로, bit-identical).
    # 'letterbox' 는 종횡비 유지 축소 + zero-pad — GenDoP 예시 배열(208x512/288x512)과 같은
    # 모양이고 화각을 안 버린다. 720x1280 소스에는 letterbox 를 쓴다 (`letterbox()` docstring).
    parser.add_argument("--rgbd_fit", default="crop", choices=["crop", "letterbox"])
    parser.add_argument("--rgb_name", default="frame_0000.png")
    parser.add_argument("--depth_name", default="frame_depth_0000.npy")
    # `cameras`(114, 생성 영상 있음) / `recon`(72, source 영상) 중 어느 쪽을 돌릴지.
    # rgbd 분기는 생성 영상이 있어야 해서 사실상 cameras 만 가능하다.
    parser.add_argument("--kinds", nargs="+", default=None, choices=["cameras", "recon"])
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    # 모델이 직접 뽑을 포즈 수. 30 = 릴리즈 학습 길이(기본, 예전과 동일), 49 = 우리 코퍼스 길이.
    parser.add_argument("--pose_length", type=int, default=30)
    # True 면 길이가 pose_length*10 과 다를 때 예전처럼 static 폴백. 49 로 요구할 때는 끄는 게 맞다.
    parser.add_argument("--strict_pose_length", action="store_true", default=True)
    parser.add_argument("--no_strict_pose_length", dest="strict_pose_length",
                        action="store_false")
    # EOS 로짓 차단. None = auto (pose_length != 30 일 때만 on). GenDoP 의
    # `assert np.all(tokens >= 0)` 가 EOS 를 -1 로 만들어 죽는 걸 막는다 (docstring §forbid_eos).
    parser.add_argument("--forbid_eos", action="store_true", default=None)
    parser.add_argument("--no_forbid_eos", dest="forbid_eos", action="store_false")
    main(parser.parse_args())
