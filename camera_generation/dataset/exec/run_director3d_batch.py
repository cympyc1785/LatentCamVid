"""이 코드가 답하는 질문: Director3D 공개 ckpt 의 카메라 궤적 생성기(Traj-DiT)는 우리 test split 캡션에 어떤 궤적을 내나?

R476 (사용자 2026-10-21 "release ckpt 평가는 CCD, Director3D 도 가능하면"). Director3D 는 text → 29 카메라
(Traj-DiT) → 3D Gaussian (GM-LDM) 2단인데, 우리 지표는 궤적만 보므로 **Traj-DiT 단계만** 쓴다.
원본 `inference.py`/`TrajDiTSystem.inference` 와 같은 샘플러(DDIM 100 step, prediction_type=sample, CFG 없음)를
배치로 돌릴 뿐 모델·스케줄러는 손대지 않는다.

출력 npz (`director3d__<scene>__<idx>.npz`) 는 `run_director_batch.py` 와 같은 **GenDoP 어댑터 규약**:
  `c2w` (49,4,4) OpenGL c2w (Director3D 는 NeRF/OpenGL 규약, `gaussians_renderer.py:134`). 29 → 49 는 Director3D
        자신의 보간기 `utils.sample_from_dense_cameras` (회전 SLERP + 이동 LERP, t = i/48).
  `scale` 1.0, `text` 넣은 문장, `cams29` 원본 29 카메라 (18 = c2w 12 + intrinsics 4 + HW 2).
게이지는 Director3D 자기 것이다 → 어댑터 `--no_rescale` (raw) 또는 rmax 를 GT 에서 빌리는 판 둘 다 낼 수 있다.

env: latentcam (diffusers·timm·plyfile). Director3D 의 `diff_gaussian_rasterization`·`kiui` 는 렌더러/유틸 import
사슬에만 걸리므로 빈 모듈로 막는다 (Traj-DiT 경로는 순수 torch quaternion 함수만 쓴다). 텍스트 인코더는
SD2.1-base 의 tokenizer + text_encoder 만 올린다 (원본은 파이프라인 전체를 올렸다가 지운다 — 같은 가중치).

사용:
    PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY exec/run_director3d_batch.py \
        --split <corpus>/seg_list_dynpose_s91_noroll_test.txt --text_dir <eval 폴더> --out <pred 폴더> [--limit 4]
"""
import json
import sys
import types
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path
from time import time

import numpy as np
import torch
import torch.nn.functional as F

D3D_ROOT = "/data1/cympyc1785/LatentCamVid/camera_generation/models/Director3D"
SD_LOCAL = glob(path.expanduser("~/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/*"))
TEXT_KEY = "Concise Interaction"


def _stub_modules():
    for name in ("diff_gaussian_rasterization", "kiui"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.GaussianRasterizationSettings = m.GaussianRasterizer = None
            sys.modules[name] = m


class _TextOnlyPipe:
    """`StableDiffusionPipeline.from_pretrained` 대역 — tokenizer·text_encoder 만 같은 snapshot 에서 올린다."""

    @staticmethod
    def from_pretrained(key, local_files_only=True):
        from transformers import CLIPTextModel, CLIPTokenizer
        root = SD_LOCAL[0]
        return types.SimpleNamespace(tokenizer=CLIPTokenizer.from_pretrained(path.join(root, "tokenizer")),
                                     text_encoder=CLIPTextModel.from_pretrained(path.join(root, "text_encoder")))


def build(ckpt, device):
    _stub_modules()
    sys.path.insert(0, D3D_ROOT)
    from omegaconf import OmegaConf
    import diffusers
    # 파이프라인 모듈 import 가 bitsandbytes CUDA 셋업에서 죽는다 — 원본이 쓰는 건 tokenizer·text_encoder 뿐이라 대역을 먼저 꽂는다.
    diffusers.StableDiffusionPipeline = _TextOnlyPipe
    import system_traj_dit
    from utils import sample_from_dense_cameras
    opt = OmegaConf.load(path.join(D3D_ROOT, "configs/main.yaml"))
    sysm = system_traj_dit.TrajDiTSystem(opt).to(device).eval()
    params = torch.load(ckpt, map_location="cpu")
    missing, unexpected = sysm.model.load_state_dict(params["traj_dit"], strict=False)
    # 원본도 strict=False 다. 텍스트 인코더 키는 ckpt 에 있든 없든 같은 SD2.1 가중치다 — 그 외 누락은 막는다.
    bad = [k for k in missing if not k.startswith("text_encoder.")]
    assert not bad, f"Traj-DiT 키 누락 {bad[:5]}"
    del params
    return sysm, sample_from_dense_cameras


@torch.no_grad()
def encode_text(m, texts):
    """`TrajDiTModel.encode_text` 와 같은 계산. 원본의 `truncation_strategy='longest_first'` 는 현 transformers 에서
    중복 인자로 죽어서 같은 뜻의 `truncation=True` (77 토큰 자르기) 로 바꿨다."""
    ids = m.tokenizer(texts, padding="max_length", truncation=True, max_length=m.tokenizer.model_max_length,
                      return_tensors="pt").input_ids
    return m.y_embedder(m.text_encoder(ids.to(next(m.text_encoder.parameters()).device))[0])


@torch.no_grad()
def infer_batch(sysm, texts, steps, seed, image_size=512):
    """`TrajDiTSystem.inference` 를 B 개로 — 원소마다 같은 seed 의 독립 noise (단독 실행과 같은 latent)."""
    B, dev = len(texts), sysm.device
    sysm.scheduler.set_timesteps(steps, dev)
    n_tok = sysm.opt.network.cdm.num_tokens
    g = [torch.Generator().manual_seed(seed) for _ in range(B)]
    x = torch.stack([torch.randn(n_tok, 11, generator=gi) for gi in g]).to(dev)
    y = encode_text(sysm.model, list(texts))
    for t in sysm.scheduler.timesteps:
        tt = t[None].repeat(B)
        pred = sysm.model(x, y, tt)
        pred[:, :, :4] = F.normalize(pred[:, :, :4], dim=-1)
        x = sysm.scheduler.step(pred.cpu(), t.cpu(), x.cpu(), eta=0).prev_sample.to(dev)
    return sysm.token_to_camera(x, image_size=image_size)          # (B,29,18)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)          # `<dataset>/<scene>/<idx>` 한 줄씩
    ap.add_argument("--text_dir", required=True)       # `<dir>/test/<prefix>_<scene>_<idx>_caption.json`
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="dynpose")
    ap.add_argument("--ckpt", default=path.join(D3D_ROOT, "model.ckpt"))
    ap.add_argument("--steps", type=int, default=100)  # 원본 inference 기본값
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_poses", type=int, default=49)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    makedirs(a.out, exist_ok=True)
    lines = [l.strip() for l in open(a.split, encoding="utf-8") if l.strip()][: a.limit or None]
    jobs, n_skip, n_nocap = [], 0, 0
    for line in lines:
        _, scene, idx = line.split("/")
        npz = path.join(a.out, f"director3d__{scene}__{idx}.npz")
        if path.isfile(npz) and not a.overwrite:
            n_skip += 1
            continue
        cap = path.join(a.text_dir, "test", f"{a.prefix}_{scene}_{idx}_caption.json")
        if not path.isfile(cap):
            n_nocap += 1
            continue
        jobs.append((npz, json.load(open(cap, encoding="utf-8"))[TEXT_KEY]))
    print(f"jobs {len(jobs)} (skip {n_skip}, no caption {n_nocap})", flush=True)
    if not jobs:
        return
    dev = "cuda"
    sysm, resample = build(a.ckpt, dev)
    t_dense = torch.linspace(0, 1, a.n_poses)
    tic = time()
    for b in range(0, len(jobs), a.batch):
        chunk = jobs[b:b + a.batch]
        cams = infer_batch(sysm, [t for _, t in chunk], a.steps, a.seed)
        # 원본은 (1,1) 시각 하나씩만 부른다 (utils.py:45) — 배치·다시각에서 slerp 브로드캐스트가 깨지므로 그대로 따른다.
        cc = cams.float().cpu()
        dense = torch.stack([torch.cat([resample(cc[i:i + 1], t_dense[j].view(1, 1)) for j in range(a.n_poses)], 1)[0]
                             for i in range(len(chunk))])                                  # (B,49,18)
        for (npz, text), c29, c49 in zip(chunk, cc.numpy(), dense.numpy()):
            c2w = np.tile(np.eye(4), (a.n_poses, 1, 1))
            c2w[:, :3, :4] = c49[:, :12].reshape(-1, 3, 4)
            np.savez(npz, c2w=c2w, scale=np.float32(1.0), text=text, cams29=c29)
        done = b + len(chunk)
        if (b // a.batch) % 10 == 0 or done == len(jobs):
            print(f"[{done}/{len(jobs)}] {time() - tic:.0f}s", flush=True)
    print(f"=== director3d DONE {len(jobs)} ===", flush=True)


if __name__ == "__main__":
    main()
