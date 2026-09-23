"""Molmo2 attention map 오버레이 **영상** 저장기 — 모델을 GPU 에 올려두고 요청마다 돌린다 (D164).

  # 대화형 (screen 에 띄워두고 텍스트만 계속 던진다)
  python scripts/viz/molmo2_attn_video.py --serve stdin --gpu 0
  # HTTP (다른 셸/스크립트에서 보낸다)
  python scripts/viz/molmo2_attn_video.py --serve http --port 8765 --gpu 0
  # 한 방
  python scripts/viz/molmo2_attn_video.py --text "Point to the larger pale camel." --gpu 0

무엇을 그리나: `text` 를 user 턴에 넣고 영상과 함께 LM 을 태운 뒤, **텍스트 토큰이 video patch
를 보는 attention** 을 프레임당 9x9 히트맵으로 올려 mp4 로 쓴다. 카메라 디코더가 먹는
`text_emb` 가 영상의 어디를 보고 나온 값인지가 이 그림이다.

왜 서버로 두나 — 두 겹의 재사용이 있다:
  1. 모델 로드 (4 shard, ~16s) 는 프로세스당 1회.
  2. **ViT(49프레임 x 729패치 x 27층)는 텍스트와 무관하다.** chat template 이 `<|video|>` 를 맨
     앞으로 올리고 LM 이 causal 이라 patch 위치는 뒤에 붙는 텍스트를 못 본다. 그래서 영상당
     1회 `merge_visual_inputs` + `build_input_embeddings` 로 prefix `inputs_embeds` 를 만들어
     캐시하고, 텍스트마다 tail 임베딩만 새로 만들어 이어 붙인다 (ViT 재실행 0회).
     `cache_molmo2_embeddings.py` 의 `scene_prefix`/`tail_hidden` 과 같은 발상이고, `--verify`
     가 processor 원본 경로와의 차이를 실측해 이 재사용이 정당한지 확인한다.

기본 층은 **15** 다. D164 실측(`molmo2_attn_map.py --stage sweep`)에서 최종층(35)의 attention 은
타겟을 전혀 판별하지 못했다 — 타겟 명사구를 낙타/울타리/나무로 바꿔도 map 코사인이 0.994~1.000
이고 질량이 9x9 테두리(62%)와 앞 3프레임(44%)에 쏠린다. 판별력이 있는 층은 11~16 뿐이다.
`--layer -1` 로 최종층도 볼 수 있다.

읽을 때 주의:
  · **attention sink** 를 뺀다. 첫 토큰이 질량의 30~50% 를 먹으므로 video patch 키에 대해서만
    renormalize 하고 sink 질량은 따로 찍는다. 안 하면 히트맵이 전부 평평해 보인다.
  · **공간 해상도는 프레임당 9x9** (molmo2 가 27x27 을 3x3 attention pooling). 378px 기준 셀당
    42px — "어느 물체"는 되고 정밀 위치는 안 된다.
  · `norm=per_frame`(기본) 은 프레임마다 자기 최대로 정규화해 프레임 안의 공간 구조를 본다.
    `global` 은 전 프레임 공통 최대라 프레임 간 질량 비교가 되지만 질량이 쏠리면 나머지가 검다.
  · GQA 32:8 이라 head 평균은 뭉개진다. 특정 head 만 보려면 `head`.
  · `테두리 질량` 이 uniform(39.5%) 을 크게 넘으면 물체가 아니라 **위치 artifact** 를 보는 것.

stdin 프로토콜 — `:` 로 시작하면 명령, 아니면 텍스트로 보고 즉시 렌더:
  :video vista4d/camel        영상 교체 (ViT 재실행 + prefix 캐시)
  :video /abs/path/to/frames  프레임 폴더도 된다
  :frames 0 49                start / num_frames
  :layer 15                   -1 = 최종층
  :span the larger pale camel query 토큰 구간 (부분 문자열). `:span all` 로 전체
  :span_tail 2                구간의 마지막 N 토큰만. 0 = 전부
  :head 2                     단일 head. `:head none` 으로 평균
  :probe cache|none|<문장>     기본 none
  :norm per_frame|global
  :fps 25                     프롬프트 timestamp 문구용 (영상 교체 시 재적용)
  :tag foo                    출력 파일명 고정. `:tag none` 으로 자동
  :show / :quit

HTTP 프로토콜 (stdlib only, 의존 추가 없음):
  POST /run    {"text": "...", "span": "...", "layer": 15, "span_tail": 0, ...} -> {"mp4": ..., stats}
  POST /video  {"scene": "vista4d/camel"} 또는 {"frames_dir": "...", "start": 0, "num_frames": 49}
  GET  /       도움말
  예) curl -s localhost:8765/run -d '{"text":"Point to the camel.","span":"the camel"}'
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ, listdir
import json
import re
import sys
import threading
import time

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch
import imageio.v2 as iio
from PIL import Image

REPO = '/data1/cympyc1785/LatentCamVid'
sys.path.insert(0, osp.join(REPO, 'camera_generation/latentcam/main'))
import cache_molmo2_embeddings as C   # noqa: E402  — load_frames / PATCH_ID / MOLMO 재사용

CORPUS = '/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121'
OUT = osp.join(REPO, 'tmp/molmo2_attn')
# inferno 근사. matplotlib 을 안 쓴다 (한글 글리프 문제도 같이 사라진다)
_CMAP = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
                  [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
                 dtype=np.float32) / 255.0


# ------------------------------------------------------------------ 렌더 유틸

def colorize(x):
    x = np.clip(x, 0, 1) * (len(_CMAP) - 1)
    i = np.floor(x).astype(int).clip(0, len(_CMAP) - 2)
    f = (x - i)[..., None]
    return _CMAP[i] * (1 - f) + _CMAP[i + 1] * f


def compose(frame, m, vmax, alpha, mark):
    h, w = frame.shape[:2]
    up = np.asarray(Image.fromarray((np.clip(m / max(vmax, 1e-12), 0, 1) * 255)
                                    .astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    a = (up[..., None] ** 0.6) * alpha          # 약한 곳은 원본을 더 보여준다
    out = frame.astype(np.float32) / 255.0 * (1 - a) + colorize(up) * a
    if mark:
        r, c = np.unravel_index(int(np.argmax(m)), m.shape)
        cy, cx = int((r + .5) * h / m.shape[0]), int((c + .5) * w / m.shape[1])
        t = max(1, h // 180)
        out[max(0, cy - 9):cy + 9, max(0, cx - t):cx + t] = [0, 1, 1]
        out[max(0, cy - t):cy + t, max(0, cx - 9):cx + 9] = [0, 1, 1]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def sidebyside(a, b, gap=6):
    h, w = a.shape[:2]
    out = np.full((h, w * 2 + gap, 3), 255, np.uint8)
    out[:, :w], out[:, w + gap:] = a, b
    return out


def slug(s, n=40):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:n] or 'x'


# ------------------------------------------------------------------ 러너

class Runner:
    """모델을 1회 올리고, 영상당 prefix 를 캐시하고, 텍스트마다 LM forward 만 돈다."""

    def __init__(self, ckpt, out_dir, dtype=torch.bfloat16):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        t0 = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.out_dir = out_dir
        makedirs(out_dir, exist_ok=True)
        self.proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt, dtype=dtype, trust_remote_code=True).to(self.device).eval()
        self.nl = self.model.config.text_config.num_hidden_layers
        self.lock = threading.Lock()
        self.vid = None
        print(f'[load] Molmo2  {time.time() - t0:.1f}s   layers {self.nl}  '
              f'hidden {self.model.config.text_config.hidden_size}  dev {self.device}',
              flush=True)

    # ---------------- 입력 조립

    @staticmethod
    def _body(text, probe):
        return f'{text}\n{probe}' if probe else text

    def _tail_str(self, text, probe):
        return (f'<|im_start|>user\n{self._body(text, probe)}<|im_end|>\n'
                f'<|im_start|>assistant\n')

    def _batch(self, video, text, probe, fps):
        body = self._body(text, probe)
        msgs = [{'role': 'user', 'content': [{'type': 'video'},
                                             {'type': 'text', 'text': body}]}]
        tstr = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        meta = [{'fps': fps, 'total_num_frames': int(video.shape[0]),
                 'duration': video.shape[0] / fps,
                 'frames_indices': list(range(int(video.shape[0])))}]
        return self.proc(text=[tstr], videos=[video], return_tensors='pt',
                         do_sample_frames=False, video_metadata=meta)

    # ---------------- 영상 -> prefix 캐시 (ViT 1회)

    @torch.inference_mode()
    def set_video(self, scene=None, frames_dir=None, start=0, num_frames=49, fps=25.0,
                  root=CORPUS):
        if frames_dir:
            fs = sorted(f for f in listdir(frames_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            assert fs, f'{frames_dir} 에 프레임이 없다'
            fs = fs[start:start + num_frames]
            video = np.stack([np.asarray(Image.open(osp.join(frames_dir, f)).convert('RGB'))
                              for f in fs])
            name = osp.basename(frames_dir.rstrip('/'))
        else:
            video = C.load_frames(root, scene, start, start + num_frames)
            name = scene.split('/')[-1]

        # prefix 는 텍스트와 무관하다 -> 더미 텍스트로 batch 를 만들고 꼬리를 떼어낸다
        batch = self._batch(video, 'x', '', fps)
        ids = batch['input_ids'][0]
        tail = self.proc.tokenizer(self._tail_str('x', ''), add_special_tokens=False,
                                   return_tensors='pt')['input_ids'][0]
        assert torch.equal(ids[-len(tail):], tail), 'prefix/tail 토큰 경계가 안 맞는다'
        pid = ids[:-len(tail)]
        core = self.model.model
        images, pooling = core.merge_visual_inputs(
            input_ids=pid[None].to(self.device), pixel_values=None,
            image_token_pooling=None, image_grids=None, image_num_crops=None,
            pixel_values_videos=batch['pixel_values_videos'].to(self.device, self.dtype),
            video_token_pooling=batch['video_token_pooling'].to(self.device),
            video_grids=batch['video_grids'].to(self.device))
        emb, _ = core.build_input_embeddings(pid[None].to(self.device), images, pooling)

        T = int(video.shape[0])
        pos = torch.nonzero(pid == C.PATCH_ID).flatten()
        side = int(round((len(pos) / T) ** 0.5))
        assert side * side * T == len(pos), f'patch {len(pos)} 가 {T} x 정사각 격자가 아니다'
        self.vid = dict(video=video, name=name, prefix_emb=emb, P=int(len(pid)),
                        patch_pos=pos, T=T, side=side, fps=fps)
        print(f'[video] {name}  frames {T} {video.shape[1]}x{video.shape[2]}  '
              f'prefix {len(pid)} tok  patch {len(pos)} = {T}x{side}^2  fps {fps}',
              flush=True)
        return self.vid

    # ---------------- attention 포획

    def _hook(self, layer, rows, store):
        """`eager_attention_forward` 를 감싸 대상 층의 지정 행만 떠낸다.

        `output_attentions=True` 는 36층 x (32,L,L) fp32 = 43 GB 라 못 쓴다. 원본 함수를 그대로
        호출하고 필요한 행만 슬라이스하므로 수치는 eager 와 동일하다 (마스크·스케일링이 원본
        경로에서 이미 적용된 softmax 결과를 자른다). config 기본이 sdpa(weights None) 라
        훅 동안만 eager 로 바꾼다 — `modeling_molmo2.py:710` 분기."""
        mod = sys.modules[type(self.model.model).__module__]
        orig = mod.eager_attention_forward
        rt = torch.as_tensor(rows)

        def wrapped(module, q, k, v, attention_mask, scaling, dropout=0.0, **kw):
            out, w = orig(module, q, k, v, attention_mask, scaling, dropout=dropout, **kw)
            li = getattr(module, 'layer_idx', -1)
            if w is not None and (layer is None or li == layer):
                # layer=None 이면 36층 전부 (한 forward 로 층 sweep 이 끝난다)
                store[li if layer is None else 'w'] = \
                    w[0][:, rt.to(w.device), :].float().cpu()               # (H,R,L)
            return out, w

        mod.eager_attention_forward = wrapped
        cfg = self.model.model.transformer.blocks[0].self_attn.config
        prev = cfg._attn_implementation
        cfg._attn_implementation = 'eager'

        def restore():
            mod.eager_attention_forward = orig
            cfg._attn_implementation = prev
        return restore

    def _rows(self, tail_str, span, span_tail):
        enc = self.proc.tokenizer(tail_str, add_special_tokens=False,
                                  return_offsets_mapping=True)
        off, ids = enc['offset_mapping'], enc['input_ids']
        if span in ('all', '', None):
            rows = list(range(len(off)))
        else:
            i = tail_str.find(span)
            assert i >= 0, (f'span 문자열을 프롬프트에서 못 찾았다: {span!r}\n'
                            f'프롬프트: {tail_str!r}')
            rows = [k for k, (a, b) in enumerate(off)
                    if b > i and a < i + len(span) and b > a]
            assert rows, f'span 이 토큰 경계와 안 맞는다: {span!r}'
        if span_tail:
            rows = rows[-span_tail:]
        return rows, [self.proc.tokenizer.decode([t]) for t in ids]

    # ---------------- 한 요청

    @torch.inference_mode()
    def run(self, text, span='all', span_tail=0, layer=15, head=None, probe='',
            norm='per_frame', alpha=0.7, mark=True, side_by_side=True,
            out_fps=10, tag=None, save_npy=True, verify=False):
        assert self.vid, ':video 로 영상을 먼저 지정할 것'
        v, core = self.vid, self.model.model
        L_ = layer + self.nl if layer < 0 else layer
        assert 0 <= L_ < self.nl, f'layer {layer} 범위 밖 (0..{self.nl - 1}, 음수는 뒤에서부터)'
        probe = {'none': '', 'cache': C.PROBE}.get(probe, probe)

        tstr = self._tail_str(text, probe)
        tids = self.proc.tokenizer(tstr, add_special_tokens=False,
                                   return_tensors='pt')['input_ids']
        rows, toks = self._rows(tstr, span, span_tail)
        P, T, side = v['P'], v['T'], v['side']

        with self.lock:
            temb, _ = core.build_input_embeddings(tids.to(self.device))
            emb = torch.cat([v['prefix_emb'], temb], 1)
            store = {}
            restore = self._hook(L_, [P + r for r in rows], store)
            try:
                out = core(inputs_embeds=emb,
                           attention_mask=torch.ones(1, emb.shape[1], dtype=torch.long,
                                                     device=self.device),
                           use_cache=False)
            finally:
                restore()
            assert 'w' in store, f'layer {L_} attention 포획 실패 — 훅이 안 걸렸다'
            w = store['w']
            if verify:
                # processor 원본 경로(ViT 재실행)와 대조해 prefix 재사용의 정당성을 잰다.
                # **훅을 다시 걸어** 두 경로를 같은 attention 구현(eager)으로 맞춘다 —
                # 안 그러면 eager vs sdpa 차이가 섞여 bf16 누적 오차를 재사용 오차로 오독한다.
                b = self._batch(v['video'], text, probe, v['fps'])
                ids2 = b['input_ids'].to(self.device)
                st2 = {}
                rs2 = self._hook(L_, [P + r for r in rows], st2)
                try:
                    im2, po2 = core.merge_visual_inputs(
                        input_ids=ids2, pixel_values=None, image_token_pooling=None,
                        image_grids=None, image_num_crops=None,
                        pixel_values_videos=b['pixel_values_videos'].to(self.device,
                                                                        self.dtype),
                        video_token_pooling=b['video_token_pooling'].to(self.device),
                        video_grids=b['video_grids'].to(self.device))
                    e2, _ = core.build_input_embeddings(ids2, im2, po2)
                    h2 = core(inputs_embeds=e2,
                              attention_mask=b['attention_mask'].to(self.device),
                              use_cache=False).last_hidden_state
                finally:
                    rs2()
                assert ids2.shape[1] == emb.shape[1], (
                    f'토큰 길이가 다르다 {ids2.shape[1]} vs {emb.shape[1]} — 대조 불가')
                h1 = out.last_hidden_state[0, P:].float()
                dh = (h2[0, P:].float() - h1).abs().max()
                de = (e2[0, :P].float() - v['prefix_emb'][0].float()).abs().max()
                a1 = w[:, :, v['patch_pos']]
                a2 = st2['w'][:, :, v['patch_pos']]
                a1n = (a1 / a1.sum(-1, keepdim=True).clamp_min(1e-9)).mean((0, 1))
                a2n = (a2 / a2.sum(-1, keepdim=True).clamp_min(1e-9)).mean((0, 1))
                cos = float(torch.nn.functional.cosine_similarity(a1n[None], a2n[None])[0])
                print(f'[verify] prefix emb |Δ|max {float(de):.6f}   '
                      f'tail hidden |Δ|max {float(dh):.4f} (|h|max {float(h1.abs().max()):.1f}, '
                      f'상대 {float(dh / h1.abs().max()) * 100:.3f}%)\n'
                      f'[verify] **attention map 코사인 {cos:.6f}** '
                      f'(1.0 이면 재사용이 출력에 영향 없음)', flush=True)

        # video patch 키만 뽑아 renormalize (sink 제거)
        vv = w[:, :, v['patch_pos']]
        vmass, sink = vv.sum(-1), w[:, :, 0]
        vn = vv / vmass.clamp_min(1e-9)[..., None]
        hv = vmass.mean(1)
        if head is not None:
            assert 0 <= head < vn.shape[0], f'head 범위 밖 (0..{vn.shape[0] - 1})'
            m = vn[head].mean(0)
        else:
            m = vn.mean((0, 1))
        m = m.view(T, side, side).numpy()

        fm = m.reshape(T, -1).sum(1)
        ring = np.zeros((side, side), bool)
        ring[0] = ring[-1] = True
        ring[:, 0] = ring[:, -1] = True
        sp = m.sum(0); sp = sp / max(sp.sum(), 1e-9)
        st = dict(video_mass=float(vmass.mean()), sink_mass=float(sink.mean()),
                  ring_mass=float(sp[ring].sum()), ring_uniform=float(ring.mean()),
                  mass_f0_2=float(fm[:3].sum()), frame_uniform=3.0 / T,
                  frame_entropy=float(-(fm * np.log(np.clip(fm, 1e-12, None))).sum()
                                      / np.log(T)),
                  peak_frame=int(fm.argmax()),
                  head_video_mass_max=float(hv.max()),
                  top_heads=torch.argsort(hv, descending=True)[:4].tolist(),
                  layer=L_, rows=len(rows), tail_tokens=int(tids.shape[1]),
                  query_text=''.join(toks[r] for r in rows))

        tg = tag or f"{v['name']}_L{L_}{'_h%d' % head if head is not None else ''}_{slug(text)}"
        mp4 = osp.join(self.out_dir, f'attn_{tg}.mp4')
        gmax = float(m.max())
        wr = iio.get_writer(mp4, fps=out_fps, codec='libx264', quality=8,
                            macro_block_size=1)
        for i in range(T):
            ov = compose(v['video'][i], m[i],
                         float(m[i].max()) if norm == 'per_frame' else gmax, alpha, mark)
            wr.append_data(sidebyside(v['video'][i], ov) if side_by_side else ov)
        wr.close()
        st['mp4'] = mp4
        if save_npy:
            np.save(mp4[:-4] + '.npy', m)
            st['npy'] = mp4[:-4] + '.npy'
        json.dump(dict(text=text, probe=probe, span=span, span_tail=span_tail, **st),
                  open(mp4[:-4] + '.json', 'w'), ensure_ascii=False, indent=1)
        return st


    @torch.inference_mode()
    def maps_all_layers(self, text, span='all', span_tail=0, probe='', head=None):
        """(layer -> (T,side,side)) 와 층별 video 질량. **forward 1회로 36층 전부.**

        `run()` 과 같은 경로지만 렌더를 안 하고 층 sweep 용으로 축약값만 돌려준다.
        메모리: 36층 x (32,R,L) fp32 를 CPU 로 받아 즉시 (T,9,9) 로 줄인다."""
        assert self.vid, 'set_video 를 먼저 호출할 것'
        v, core = self.vid, self.model.model
        probe = {'none': '', 'cache': C.PROBE}.get(probe, probe)
        tstr = self._tail_str(text, probe)
        tids = self.proc.tokenizer(tstr, add_special_tokens=False,
                                   return_tensors='pt')['input_ids']
        rows, _ = self._rows(tstr, span, span_tail)
        P, T, side = v['P'], v['T'], v['side']
        with self.lock:
            temb, _ = core.build_input_embeddings(tids.to(self.device))
            emb = torch.cat([v['prefix_emb'], temb], 1)
            store = {}
            restore = self._hook(None, [P + r for r in rows], store)
            try:
                core(inputs_embeds=emb,
                     attention_mask=torch.ones(1, emb.shape[1], dtype=torch.long,
                                               device=self.device),
                     use_cache=False)
            finally:
                restore()
        assert store, 'attention 포획 실패'
        maps, vmass = {}, {}
        for li, w in store.items():
            vv = w[:, :, v['patch_pos']]
            vm = vv.sum(-1)
            vmass[li] = float(vm.mean())
            vn = vv / vm.clamp_min(1e-9)[..., None]
            m = vn[head].mean(0) if head is not None else vn.mean((0, 1))
            maps[li] = m.view(T, side, side).numpy()
        return maps, vmass, len(rows)


def report(st):
    print(f"  video 질량 {st['video_mass'] * 100:.2f}%   sink {st['sink_mass'] * 100:.2f}%   "
          f"query {st['rows']} tok  {st['query_text'][:56]!r}")
    print(f"  테두리 {st['ring_mass'] * 100:.1f}% (uniform {st['ring_uniform'] * 100:.1f}%)   "
          f"앞3프레임 {st['mass_f0_2'] * 100:.1f}% (uniform {st['frame_uniform'] * 100:.1f}%)   "
          f"엔트로피 {st['frame_entropy']:.3f}  peak f{st['peak_frame']}")
    print(f"  head max {st['head_video_mass_max'] * 100:.1f}%  top {st['top_heads']}")
    print(f"  -> {st['mp4']}", flush=True)


# ------------------------------------------------------------------ stdin 서버

def serve_stdin(r, cfg, args):
    HELP = ('명령: :video <scene|dir>  :frames <start> <n>  :layer N  :span <문자열|all>  '
            ':span_tail N\n      :head N|none  :probe cache|none|<문장>  :norm per_frame|global  '
            ':fps F\n      :tag <이름>|none  :show  :quit      (그 외 입력은 텍스트로 보고 렌더)')
    print(HELP, flush=True)
    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        if not s.startswith(':'):
            try:
                t0 = time.time()
                st = r.run(s, **cfg)
                print(f'[run] {time.time() - t0:.1f}s')
                report(st)
            except Exception as e:                                 # noqa: BLE001
                print(f'[err] {type(e).__name__}: {e}', flush=True)
            continue
        p = s[1:].split(None, 1)
        k, val = p[0], (p[1].strip() if len(p) > 1 else '')
        try:
            if k in ('quit', 'q', 'exit'):
                break
            elif k == 'help':
                print(HELP, flush=True)
            elif k == 'video':
                if osp.isdir(val):
                    r.set_video(frames_dir=val, start=args.start,
                                num_frames=args.num_frames, fps=args.fps)
                else:
                    r.set_video(scene=val, start=args.start, num_frames=args.num_frames,
                                fps=args.fps, root=args.root)
            elif k == 'frames':
                a, b = val.split()
                args.start, args.num_frames = int(a), int(b)
                print(f'  start {args.start}  num_frames {args.num_frames} '
                      f'(다음 :video 에서 적용)', flush=True)
            elif k == 'fps':
                args.fps = float(val)
                print(f'  fps {args.fps} (다음 :video 에서 적용)', flush=True)
            elif k == 'layer':
                cfg['layer'] = int(val)
            elif k == 'span':
                cfg['span'] = val or 'all'
            elif k == 'span_tail':
                cfg['span_tail'] = int(val)
            elif k == 'head':
                cfg['head'] = None if val in ('none', '') else int(val)
            elif k == 'probe':
                cfg['probe'] = '' if val == 'none' else val
            elif k == 'norm':
                assert val in ('per_frame', 'global')
                cfg['norm'] = val
            elif k == 'tag':
                cfg['tag'] = None if val in ('none', '') else val
            elif k == 'show':
                print('  ' + json.dumps({**cfg, 'video': r.vid and r.vid['name'],
                                         'start': args.start,
                                         'num_frames': args.num_frames, 'fps': args.fps},
                                        ensure_ascii=False), flush=True)
            else:
                print(f'[err] 모르는 명령 :{k}\n{HELP}', flush=True)
            if k in ('layer', 'span', 'span_tail', 'head', 'probe', 'norm', 'tag'):
                print(f'  {k} = {cfg.get(k)!r}', flush=True)
        except Exception as e:                                     # noqa: BLE001
            print(f'[err] {type(e).__name__}: {e}', flush=True)
    print('[bye]', flush=True)


# ------------------------------------------------------------------ HTTP 서버

def serve_http(r, cfg, args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    HELP = {'POST /run': {'text': 'required', 'span': 'all', 'span_tail': 0, 'layer': 15,
                          'head': None, 'probe': '', 'norm': 'per_frame', 'tag': None},
            'POST /video': {'scene': 'vista4d/camel',
                            'frames_dir': '(대신 사용 가능)', 'start': 0, 'num_frames': 49,
                            'fps': 25.0},
            'note': 'GPU 라 요청은 직렬 처리된다. 응답은 mp4 경로 + 진단 수치.'}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False, indent=1).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

        def do_GET(self):
            self._send(200, HELP)

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            try:
                req = json.loads(self.rfile.read(n) or b'{}')
            except Exception as e:                                 # noqa: BLE001
                return self._send(400, {'error': f'bad json: {e}'})
            try:
                if self.path.rstrip('/') == '/video':
                    v = r.set_video(scene=req.get('scene'),
                                    frames_dir=req.get('frames_dir'),
                                    start=int(req.get('start', args.start)),
                                    num_frames=int(req.get('num_frames', args.num_frames)),
                                    fps=float(req.get('fps', args.fps)),
                                    root=req.get('root', args.root))
                    return self._send(200, {'video': v['name'], 'frames': v['T'],
                                            'side': v['side'], 'prefix_tokens': v['P']})
                if self.path.rstrip('/') != '/run':
                    return self._send(404, {'error': 'POST /run 또는 /video', **HELP})
                assert req.get('text'), 'text 가 비었다'
                kw = {**cfg, **{k: req[k] for k in
                                ('span', 'span_tail', 'layer', 'head', 'probe', 'norm',
                                 'alpha', 'mark', 'side_by_side', 'out_fps', 'tag')
                                if k in req}}
                t0 = time.time()
                st = r.run(req['text'], **kw)
                print(f'[http] {time.time() - t0:.1f}s  {req["text"][:60]!r}')
                report(st)
                return self._send(200, st)
            except Exception as e:                                 # noqa: BLE001
                return self._send(500, {'error': f'{type(e).__name__}: {e}'})

    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f'[http] listening on {args.host}:{args.port}   (GET / 로 도움말)\n'
          f"  예) curl -s localhost:{args.port}/run "
          f"-d '{{\"text\":\"Point to the camel.\",\"span\":\"the camel\"}}'", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n[bye]', flush=True)


# ------------------------------------------------------------------ main

def main(args):
    r = Runner(args.ckpt, args.out_dir)
    cfg = dict(span=args.span, span_tail=args.span_tail, layer=args.layer, head=args.head,
               probe=args.probe, norm=args.norm, alpha=args.alpha, mark=args.mark,
               side_by_side=args.side_by_side, out_fps=args.out_fps,
               tag=args.tag or None)
    if not (args.serve == 'http' and args.no_preload):
        r.set_video(scene=args.scene, frames_dir=args.frames_dir or None, start=args.start,
                    num_frames=args.num_frames, fps=args.fps, root=args.root)
    if args.text:
        report(r.run(args.text, verify=args.verify, **cfg))
        if args.serve == 'off':
            return
    if args.serve == 'stdin':
        serve_stdin(r, cfg, args)
    elif args.serve == 'http':
        serve_http(r, cfg, args)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--serve', default='off', choices=('off', 'stdin', 'http'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--no_preload', action='store_true',
                   help='http 모드에서 영상을 미리 안 올린다 (POST /video 로 지정)')
    p.add_argument('--text', default='', help='주면 즉시 1회 렌더 (serve 와 같이 쓸 수 있다)')
    p.add_argument('--span', default='all')
    p.add_argument('--span_tail', type=int, default=0)
    p.add_argument('--layer', type=int, default=15, help='기본 15 (D164 실측 최적). -1 = 최종층')
    p.add_argument('--head', type=int, default=None)
    p.add_argument('--probe', default='', help="'' 없음 / 'cache' 학습 캐시 probe / 임의 문장")
    p.add_argument('--verify', action='store_true', help='prefix 재사용 vs 원본 경로 대조')
    # 입력
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scene', default='vista4d/camel')
    p.add_argument('--frames_dir', default='')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--num_frames', type=int, default=49)
    p.add_argument('--fps', type=float, default=25.0)
    # 출력
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--tag', default='')
    p.add_argument('--out_fps', type=int, default=10)
    p.add_argument('--norm', default='per_frame', choices=('per_frame', 'global'))
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--side_by_side', dest='side_by_side', action='store_true', default=True)
    p.add_argument('--no_side_by_side', dest='side_by_side', action='store_false')
    p.add_argument('--mark', dest='mark', action='store_true', default=True)
    p.add_argument('--no_mark', dest='mark', action='store_false')
    p.add_argument('--gpu', default='0')
    main(p.parse_args())
