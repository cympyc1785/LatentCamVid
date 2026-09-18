"""umt5-xxl 텍스트 인코더의 출력(prefill)을 캡션 단위로 미리 구워 두는 캐시 빌더 (D200 knob B).

WHY: umt5 는 학습 내내 frozen 이고 (`T5EncoderModel` 은 `.eval().requires_grad_(False)`),
캡션은 세그먼트에 **붙박이**라 epoch 마다 같은 문자열이 다시 들어온다. 그런데 D200 5-arm
스텝 계측에서 이 forward 가 arm ① 42.5 ms (15.0%) / arm ② 48.2 ms (10.0%) 로 **denoiser
forward 보다 컸다**. 한 번 구워 두면 이 구간이 테이블 조회로 바뀐다.
`geo_raw_cache_dir` · `peav_text_cache` 와 같은 발상이고, 크기는 그 둘보다 훨씬 작다 —
dynpose d200 은 고유 캡션 43,251개 × 평균 22.7 tok × 4096 × bf16 = 약 9.6 GB.

**왜 ragged 인가**: `text_len=128` 슬롯을 다 저장하면 27 GB 인데, 그 중 실제 토큰은 평균 23개뿐이고
나머지는 attention mask 로 지워진다 (`camera_diffusion_model_latent.py:385`
`text_cross_attn(..., key_padding_mask=~text_mask)`). 그래서 **유효 토큰만** 이어 붙이고
(`emb` + `offsets`), 읽을 때 0 으로 패딩한다.

**비트동일**: umt5 는 절대 위치 임베딩이 없고(relative bias만) padding key 는 마스크로 지워지므로,
유효 토큰의 출력은 뒤에 붙은 padding 길이와 무관하다. 즉 캐시를 굽는 순간의 `--text_len` 과
학습의 `cfg.text_len` 이 달라도 유효 구간은 같은 값이다. 단 **truncation 경계**는 다르므로
학습보다 짧은 `--text_len` 으로 구우면 안 된다 (같은 값으로 굽는 게 기본).

캡션 출처(`--source`):
  index    `<root>/.latentcam_index/*.pt` 의 samples 튜플 4번째 원소 — 데이터셋이 실제로 쓰는
           바로 그 문자열이다 (`dataset_dl3dv.py:1602 'text_prompt': caption`). 기본값.
  prompts  `<root>/<scene_dir>/*/prompts.json` 의 `prompt_camera_with_scene_video.concise`.
           인덱스 캐시가 아직 없을 때의 대안.

사용 예시:
  python main/cache_umt5_embeddings.py --gpu 4 \
    --root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
    --out  /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200/umt5_cache/text_len128.pt \
    --text_len 128
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import glob
import json
import sys
import time

# `CUDA_VISIBLE_DEVICES` 는 **`import torch` 보다 먼저** 박혀야 먹는다 (argparse 를 기다리면 늦다).
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from models.t5 import T5EncoderModel                                        # noqa: E402

T5_DIR = '/data1/cympyc1785/LatentCamVid/video_generation/models/DiffSynth-Studio/Wan-AI/Wan2.2-TI2V-5B'
T5_CKPT = 'models_t5_umt5-xxl-enc-bf16.pth'
T5_TOK = 'google/umt5-xxl'


def captions_from_index(root):
    """인덱스 캐시(.latentcam_index/*.pt) 전부에서 캡션을 긁는다. 데이터셋이 읽는 그 값."""
    out = []
    files = sorted(glob.glob(osp.join(root, '.latentcam_index', '*.pt')))
    for f in files:
        idx = torch.load(f, map_location='cpu', weights_only=False)
        got = [s[3] for s in idx['samples']]
        out += got
        print(f"[src] {osp.basename(f)}: {len(got)} samples", flush=True)
    if not files:
        print(f"[src] WARNING: {root}/.latentcam_index 가 비어 있다 — --source prompts 를 쓸 것")
    return out


def captions_from_prompts(root, prompts_file='prompts.json'):
    out = []
    for f in sorted(glob.glob(osp.join(root, '*', '*', prompts_file))):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for _k, seg in (j.items() if isinstance(j, dict) else []):
            pcs = seg.get('prompt_camera_with_scene_video')
            out.append(pcs.get('concise', "") if isinstance(pcs, dict) else (pcs or ""))
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--root', required=True)                   # 코퍼스 루트 (dl3dv_root)
    ap.add_argument('--out', required=True)                    # 캐시 .pt 경로
    ap.add_argument('--source', default='index', choices=['index', 'prompts', 'both'])
    ap.add_argument('--prompts_file', default='prompts.json')
    ap.add_argument('--text_len', type=int, default=128)       # 학습의 cfg.text_len 과 같게
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--t5_dir', default=T5_DIR)
    ap.add_argument('--t5_ckpt', default=T5_CKPT)
    ap.add_argument('--t5_tok', default=T5_TOK)
    a = ap.parse_args()

    texts = []
    if a.source in ('index', 'both'):
        texts += captions_from_index(a.root)
    if a.source in ('prompts', 'both'):
        texts += captions_from_prompts(a.root, a.prompts_file)
    # CFG 용 빈 캡션은 코퍼스에 없어도 항상 넣는다 (dropout/uncond 경로가 쓴다).
    texts.append("")
    uniq = sorted(set(texts))
    print(f"[src] {len(texts)} captions -> {len(uniq)} unique", flush=True)

    device = torch.device('cuda')
    enc = T5EncoderModel(text_len=a.text_len, dtype=torch.bfloat16, device=device,
                         checkpoint_path=osp.join(a.t5_dir, a.t5_ckpt),
                         tokenizer_path=osp.join(a.t5_dir, a.t5_tok), shard_fn=None)

    chunks, lens = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(uniq), a.batch_size):
            bt = uniq[i:i + a.batch_size]
            ctx, mask = enc(bt, device)                         # (B,L,4096) bf16, (B,L)
            m = mask.gt(0)
            L = m.sum(dim=1).long()
            # 마스크가 앞쪽 연속 prefix 인지 확인 — 아니면 ragged 저장이 성립하지 않는다.
            pref = torch.arange(m.shape[1], device=m.device)[None, :] < L[:, None]
            assert torch.equal(m, pref), "tokenizer padding 이 right-align 이 아니다"
            ctx = ctx.to(torch.bfloat16).cpu()
            for b in range(len(bt)):
                chunks.append(ctx[b, :L[b].item()].clone())
                lens.append(int(L[b].item()))
            if (i // a.batch_size) % 50 == 0:
                el = time.time() - t0
                print(f"[enc] {i + len(bt)}/{len(uniq)}  {(i + len(bt)) / max(el, 1e-9):.0f} cap/s  "
                      f"elapsed {el / 60:.1f} min", flush=True)

    emb = torch.cat(chunks, dim=0)
    lens_t = torch.tensor(lens, dtype=torch.int32)
    offsets = torch.zeros(len(lens) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lens_t.long(), 0)
    obj = {
        'format': 'umt5_text_cache_v1',
        'by_text': {t: i for i, t in enumerate(uniq)},
        'emb': emb,                     # (sum_L, dim) bf16 — 유효 토큰만 이어 붙인 것
        'offsets': offsets,             # (U+1,) int64
        'lens': lens_t,                 # (U,) int32
        'text_len': a.text_len,         # 구울 때 쓴 truncation 길이
        'dim': int(emb.shape[1]),
        'tokenizer': osp.join(a.t5_dir, a.t5_tok),
        'checkpoint': osp.join(a.t5_dir, a.t5_ckpt),
    }
    makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    torch.save(obj, a.out)
    sz = osp.getsize(a.out) / 2**30
    print(f"{'=' * 60}")
    print(f"out      : {a.out}  ({sz:.2f} GiB)")
    print(f"captions : {len(uniq)} unique  (raw {len(texts)})")
    print(f"tokens   : sum {int(lens_t.sum())}  mean {lens_t.float().mean():.1f}  "
          f"max {int(lens_t.max())}  (text_len {a.text_len})")
    print(f"emb      : {tuple(emb.shape)} {emb.dtype}")
    print(f"elapsed  : {(time.time() - t0) / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
