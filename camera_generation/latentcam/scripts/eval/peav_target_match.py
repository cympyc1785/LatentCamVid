"""PE-AV(pe-av-large) 로 "우리 캡션이 영상 속 target 을 실제로 지목하는가"를 재는 프로브.

WHY: latentcam 에 video embedding cross-attention 을 붙이기 전에, 그 임베딩 공간이 애초에
target 을 구분하는지부터 확인해야 한다. PE-AV 는 video/text 를 한 공간에 넣는 contrastive
검색 모델이라, **같은 영상에 대해 target 명사만 바꿔치기한 후보 텍스트들** 중 진짜 target 이
1등으로 뽑히는지를 보면 된다. motion 절은 씬별로 고정하고 명사만 스왑하므로, 점수 차이는
전부 target 신호에서 온다 (motion 문구의 난이도가 섞이지 않는다).

측정 대상은 텍스트 형식 4종:
  struct     코퍼스 그대로          "target: dog. motion: the camera dollies back ..."
  nl         자연어로 엮은 것        "the camera tracks the dog while pushing in ..."
  nl_short   preset 어휘 + 전치사구  "track dolly in toward the dog"
  tgt_only   motion 제거 (대조군)    "a video of a dog"
tgt_only 가 struct/nl 보다 높으면 motion 절이 target 신호를 희석하는 것이고, 낮으면 motion
문맥이 오히려 도움이 되는 것이다 — 어느 쪽이든 CA 에 무엇을 넣을지가 갈린다.

지표는 후보 = 코퍼스 전체의 **서로 다른 anchor_label 어휘**(중복 제거)이고, 정답은 그 씬의
label 하나다. 우연 수준은 1/|vocab| 이라 같이 찍는다. label 이 "man"/"person" 처럼 겹치는
쌍이 있으므로 strict top-1 은 하한으로 읽어야 한다 (top-5 / MRR 을 같이 본다).

사용 예시:
  python scripts/eval/peav_target_match.py --split test --gpu 2
  python scripts/eval/peav_target_match.py --split test --num_frames 49 --out /tmp/peav_test.json
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import sys
import time

# `CUDA_VISIBLE_DEVICES` 는 **`import torch` 보다 먼저** 박혀야 먹는다. argparse 결과를 기다리면
# 이미 늦으므로 argv 를 여기서 직접 훑는다 (커맨드라인에 env 를 앞세우면 그게 우선).
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch
from PIL import Image

PE_REPO = '/data1/cympyc1785/LatentCamVid/camera_generation/tools/perception_models'
PE_CKPT = osp.join(PE_REPO, 'checkpoints', 'pe-av-large-pm')
CORPUS = '/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d107'


# ---------------------------------------------------------------- 텍스트 형식

def make_texts(label, motion, preset):
    """target 명사 `label` 을 4가지 형식으로 문장에 심는다. motion/preset 은 씬 고정."""
    # motion 절은 대개 "the camera ..." 로 시작하고 일부는 "the subject" 를 포함한다.
    if 'the subject' in motion:
        nl = motion.replace('the subject', f'the {label}')
    else:
        nl = f'{motion} with the {label} in frame'
    return {
        'struct':   f'target: {label}. motion: {motion}.',
        'nl':       nl[0].upper() + nl[1:] + '.',
        'nl_short': f"{preset.replace('_', ' ')} toward the {label}",
        'tgt_only': f'a video of a {label}',
    }


TEMPLATES = ['struct', 'nl', 'nl_short', 'tgt_only']


# ---------------------------------------------------------------- 코퍼스 읽기

def collect_scenes(root, split, max_scenes, seg_per_scene):
    """seg_list 에서 씬을 모아 (chunk, seg_key, label, motion, preset) 리스트를 만든다."""
    segs = [l.strip() for l in open(osp.join(root, f'seg_list_dynpose_{split}.txt')) if l.strip()]
    by_scene = {}
    for s in segs:
        chunk, key = s.rsplit('/', 1)
        by_scene.setdefault(chunk, []).append(key)
    out = []
    for chunk in sorted(by_scene):
        p = osp.join(root, chunk, 'da3', 'prompts.json')
        if not osp.exists(p):
            continue
        pj = json.load(open(p))
        taken = 0
        for key in sorted(by_scene[chunk], key=lambda k: int(k)):
            e = pj.get(key)
            if e is None:
                continue
            lab = (e.get('anchor_label') or '').strip()
            mot = (e.get('caption_fields') or {}).get('motion', '').strip()
            if not lab or not mot:
                continue
            out.append(dict(chunk=chunk, seg=key, label=lab, motion=mot,
                            preset=e.get('preset', ''), frame_idx=e['frame_idx']))
            taken += 1
            if taken >= seg_per_scene:
                break
        if max_scenes and len({o['chunk'] for o in out}) >= max_scenes:
            break
    return out


def load_video(root, chunk, s, e, num_frames, size=336):
    """images_4 의 [s,e) 구간에서 num_frames 장을 균등 추출 → (T,3,size,size) float, [-1,1]."""
    d = osp.join(root, chunk, 'images_4')
    idxs = [s + int(i * (e - 1 - s) / max(num_frames - 1, 1)) for i in range(num_frames)]
    frames = []
    for i in idxs:
        im = Image.open(osp.join(d, f'{i:05d}.png')).convert('RGB').resize(
            (size, size), Image.BILINEAR)
        frames.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
    v = torch.stack(frames).float().div_(255.).sub_(0.5).div_(0.5)
    return v


# ---------------------------------------------------------------- main

def main(args):
    sys.path.insert(0, PE_REPO)
    # `core/audio_visual_encoder/__init__.py` 가 `transforms.py` 를 끌고 오고, 그 파일 7행이
    # `torchcodec` 을 import 한다 — vista4d/latentcam env 에 없다. 우리는 디코더를 안 쓰고
    # (프레임을 직접 PNG 에서 읽어 336² 로 맞춘다) 모델 클래스만 필요하다.
    # 막을 곳은 **`transforms` 쪽**이다. `torchcodec` 을 가짜 모듈로 채우면 transformers 가
    # "torchcodec 이 설치돼 있다"고 믿고 `importlib.metadata.version("torchcodec")` 을 부르다
    # 죽는다 (ModernBert lazy import 가 통째로 실패). 빈 껍데기를 sys.modules 에 미리 꽂아두면
    # `from .transforms import ...` 가 그걸 집어가고 torchcodec 은 아예 안 건드린다.
    from types import ModuleType
    _tr = ModuleType('core.audio_visual_encoder.transforms')
    _tr.PEAudioVisualTransform = _tr.PEAudioFrameTransform = None
    sys.modules.setdefault('core.audio_visual_encoder.transforms', _tr)
    from core.audio_visual_encoder import PEAudioVisual
    from transformers import AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    model = PEAudioVisual.from_config(args.ckpt, pretrained=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    print(f'[load] PE-AV in {time.time() - t0:.1f}s  output_dim={model.config.output_dim}')

    items = collect_scenes(args.root, args.split, args.max_scenes, args.seg_per_scene)
    vocab = sorted({it['label'] for it in items})
    print(f'[data] {len(items)} segments / {len({i["chunk"] for i in items})} scenes '
          f'/ {len(vocab)} distinct labels')

    # ---- 영상 임베딩 (씬당 1회; 같은 씬의 여러 seg 는 frame_idx 가 달라 각각 인코딩)
    vemb = []
    for n, it in enumerate(items):
        s, e = int(it['frame_idx'][0]), int(it['frame_idx'][1])
        v = load_video(args.root, it['chunk'], s, e, args.num_frames).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            z = model.encode_video(v.unsqueeze(0),
                                   padding_mask_videos=torch.ones(1, v.shape[0], dtype=torch.bool,
                                                                  device=device))
        vemb.append(torch.nn.functional.normalize(z.float(), dim=-1).squeeze(0).cpu())
        if (n + 1) % 20 == 0:
            print(f'  [video] {n + 1}/{len(items)}  {time.time() - t0:.0f}s')
    vemb = torch.stack(vemb)

    # ---- 형식별 텍스트 임베딩 & 순위
    def encode_texts(texts, bs=64):
        outs = []
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], return_tensors='pt', padding='longest',
                    truncation=True, max_length=512).to(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                z = model.encode_video_text(b['input_ids'], b['attention_mask'])
            outs.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu())
        return torch.cat(outs)

    results = {}
    for tmpl in TEMPLATES:
        ranks = []
        for i, it in enumerate(items):
            cands = [make_texts(L, it['motion'], it['preset'])[tmpl] for L in vocab]
            temb = encode_texts(cands)
            sim = temb @ vemb[i]
            gt = vocab.index(it['label'])
            ranks.append(int((sim > sim[gt]).sum().item()) + 1)
        r = np.asarray(ranks)
        results[tmpl] = dict(top1=float((r == 1).mean()), top5=float((r <= 5).mean()),
                             mrr=float((1.0 / r).mean()), mean_rank=float(r.mean()),
                             ranks=r.tolist())
        print(f'  [text] {tmpl:9s} top1={results[tmpl]["top1"]:.3f} '
              f'top5={results[tmpl]["top5"]:.3f} mrr={results[tmpl]["mrr"]:.3f}')

    chance = 1.0 / len(vocab)
    print()
    print(f'{"template":10s} {"top1":>7s} {"top5":>7s} {"MRR":>7s} {"mean_rank":>10s}')
    print('-' * 46)
    for tmpl in TEMPLATES:
        m = results[tmpl]
        print(f'{tmpl:10s} {m["top1"]:7.3f} {m["top5"]:7.3f} {m["mrr"]:7.3f} {m["mean_rank"]:10.2f}')
    print('-' * 46)
    print(f'{"chance":10s} {chance:7.3f} {min(5.0 / len(vocab), 1.0):7.3f} '
          f'{float(np.mean(1.0 / np.arange(1, len(vocab) + 1))):7.3f} '
          f'{(len(vocab) + 1) / 2:10.2f}')
    print(f'\nn={len(items)} segments, |vocab|={len(vocab)}, num_frames={args.num_frames}')

    if args.out:
        makedirs(osp.dirname(args.out), exist_ok=True)
        json.dump(dict(config=vars(args), vocab=vocab, chance=chance,
                       items=[{k: v for k, v in it.items() if k != 'frame_idx'} for it in items],
                       results=results, examples={t: make_texts(items[0]['label'],
                                                                items[0]['motion'],
                                                                items[0]['preset'])[t]
                                                  for t in TEMPLATES}),
                  open(args.out, 'w'), ensure_ascii=False, indent=1)
        print(f'[saved] {args.out}')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=PE_CKPT)                 # perception_models 리비전 체크포인트
    p.add_argument('--root', default=CORPUS)                  # d107 코퍼스 루트
    p.add_argument('--split', default='test')                 # seg_list_dynpose_<split>.txt
    p.add_argument('--max_scenes', type=int, default=0)       # 0 = 전량
    p.add_argument('--seg_per_scene', type=int, default=1)    # 씬당 몇 개 변이를 쓸지
    p.add_argument('--num_frames', type=int, default=16)      # PE-AV 에 넣을 프레임 수
    p.add_argument('--gpu', default='2')
    p.add_argument('--out', default='')
    a = p.parse_args()
    main(a)
