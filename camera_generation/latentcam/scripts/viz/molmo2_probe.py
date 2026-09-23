"""prefill **hidden state** 에 학습된 probe 를 얹어 위치가 읽히는지 판정한다 (D165-2).

  python scripts/viz/molmo2_probe.py --stage dump  --gpu 4
  python scripts/viz/molmo2_probe.py --stage train --gpu 4

왜: D165 가 prefill **attention** 경로를 닫았다 — per-frame tracking / 시간 marginal /
head 선택 / LOO 전부 고정셀 통제 이하였고, 시간 marginal 그림은 텍스트를 바꿔도 거의
안 움직였다 (텍스트 무관 saliency). 그런데 **attention 이 못 짚는다는 게 hidden 에 정보가
없다는 뜻은 아니다.** Analysis-by-Proxy (arXiv 2607.06445) 는 frozen Qwen2.5-VL 에서
localization 89% 를 냈는데 그건 attention argmax 가 아니라 **학습된 Q-Former** 였다.

그래서 실패한 측정과 **딱 한 가지만** 다르게 한다:

    지금까지   logits = h_tgt · h_vid^T / sqrt(d)               투영 없음 (raw dot)
    이 스크립트 logits = (W_q h_tgt) · (W_k h_vid)^T / sqrt(d)   투영을 학습

`W_q = W_k = I` 면 `molmo2_latent_sweep.py` 의 `dot` readout (최선 168.2px) 과 글자 그대로
같다. 그래서 차이가 나면 원인이 "투영 학습" 하나로 특정된다.

## 통제 셋 — 이 셋을 같이 봐야 숫자가 뜻을 갖는다

  `probe`     학습된 투영 (본 실험)
  `notext`    `h_tgt` 를 **상수 벡터**로 바꿔 같은 용량으로 학습. 순수 saliency 의 상한이다.
              probe 가 이걸 못 넘으면 텍스트를 안 쓰고 있다는 뜻 — D165 의 시간 marginal
              그림이 보여준 그 실패와 같은 것.
  `rawdot`    W=I. 학습 전 재현치.
  `fixedcell` attention·hidden 을 아예 안 쓰고 9x9 셀 하나를 고정으로 가리킨 oracle.

## 분할은 **씬 단위**

anchor 단위로 나누면 같은 영상의 `h_vid` 를 공유해 반드시 샌다. 씬 8개를 4-fold 로 나눠
교차검증하고, fold 마다 train 씬에서만 학습해 held-out 씬에서 평가한다.

## 구조적으로 유리한 점

`chat_template` 이 `<|video|>` 를 맨 앞으로 hoist 하고 causal mask 가 걸리므로 **video patch
hidden 은 텍스트에 전혀 의존하지 않는다** (D164 에서 prefix 캐시 비트 동일로 검증). 따라서
`h_vid` 는 씬 단위 1회, `h_tgt` 만 anchor 마다 뽑으면 된다.

산출물
  dump   <out>/probe_dump/<scene>.npz     h_vid (T,81,C) fp16 x 층, h_tgt (n_anchor,C) x 층
  train  콘솔 표 + <out>/probe_result.json

env: `latentcam`.  GPU 1장 (dump 만 GPU, train 은 CPU 로도 돈다).
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import sys

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, CORPUS                     # noqa: E402
from molmo2_vs_obb import project, scene_anchors                 # noqa: E402
import cache_molmo2_embeddings as C                              # noqa: E402

SCENES = ('camel', 'car-roundabout', 'basketball-four', 'couple-walk',
          'golf', 'cows', 'couple-rocks', 'fashion-walk')


def cell_of(u, v, side, W, H):
    return (int(np.clip(v * side / H, 0, side - 1)),
            int(np.clip(u * side / W, 0, side - 1)))


# ------------------------------------------------------------------ dump

@torch.inference_mode()
def dump_scene(r, args, scene, layers):
    """씬 하나 -> npz. `h_vid` 는 텍스트 독립이라 1회, `h_tgt` 는 anchor 마다."""
    d = osp.join(args.root, scene, 'da3')
    if not osp.exists(osp.join(d, 'target_track.npz')):
        return None
    p = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
    E, K = p['extrinsics'], p['intrinsics']
    anch = scene_anchors(args.root, scene, args.max_per_kind)
    if not anch:
        return None
    r.set_video(scene=scene, start=0, num_frames=E.shape[0], fps=args.fps, root=args.root)
    v, core = r.vid, r.model.model
    T, side, P = v['T'], v['side'], v['P']
    H_, W_ = v['video'].shape[1], v['video'].shape[2]
    pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
    sx, sy = W_ / pw, H_ / ph

    out = dict(scene=scene, T=T, side=side, W=W_, H=H_, layers=np.array(layers))
    hv_done = False
    rows_tgt, uvs, labs, txts = [], [], [], []
    for a, key, txt, lab, tw in anch:
        uv, _z = project(tw, E, K)
        uv = uv * np.array([sx, sy])
        ok = [f for f in range(min(T, uv.shape[0]))
              if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W_ and 0 <= uv[f, 1] < H_]
        if len(ok) < args.min_frames:
            continue
        text = f'{args.verb} {txt}'
        tstr = r._tail_str(text, '')
        tids = r.proc.tokenizer(tstr, add_special_tokens=False,
                                return_tensors='pt')['input_ids']
        rw, _ = r._rows(tstr, txt, 0)                  # target 명사구 토큰 위치
        temb, _ = core.build_input_embeddings(tids.to(r.device))
        emb = torch.cat([v['prefix_emb'], temb], 1)
        o = core(inputs_embeds=emb,
                 attention_mask=torch.ones(1, emb.shape[1], dtype=torch.long,
                                           device=r.device),
                 use_cache=False, output_hidden_states=True)
        hs = o.hidden_states                           # tuple len nl+1 (embed + 층별)
        # h_tgt: target span 토큰 평균. 층 li -> hs[li+1]
        ht = np.stack([hs[li + 1][0, [P + i for i in rw]].float().mean(0).cpu().numpy()
                       for li in layers])              # (nL, C)
        rows_tgt.append(ht)
        uvs.append(uv)
        labs.append(lab)
        txts.append(txt)
        if not hv_done:
            pp = v['patch_pos'].numpy()
            hv = np.stack([hs[li + 1][0, pp].float().cpu().numpy().reshape(
                T, side * side, -1) for li in layers])  # (nL, T, 81, C)
            out['h_vid'] = hv.astype(np.float16)
            hv_done = True
        del o, hs
    if not rows_tgt:
        return None
    out['h_tgt'] = np.stack(rows_tgt).astype(np.float16)      # (n_anchor, nL, C)
    out['uv'] = np.stack(uvs).astype(np.float32)              # (n_anchor, T, 2)
    out['labels'] = np.array(labs)
    out['texts'] = np.array(txts)
    return out


def stage_dump(args):
    layers = [int(x) for x in args.layers.split(',')]
    dd = osp.join(args.out_dir, 'probe_dump')
    makedirs(dd, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    for sc in args.scenes.split(','):
        fp = osp.join(dd, sc.split('/')[-1] + '.npz')
        if osp.exists(fp) and not args.overwrite:
            print(f'[skip] {fp} 존재 (--overwrite 로 덮어쓰기)', flush=True)
            continue
        o = dump_scene(r, args, sc, layers)
        if o is None:
            print(f'[skip] {sc}: anchor 0 / target_track 없음', flush=True)
            continue
        np.savez_compressed(fp, **o)
        mb = osp.getsize(fp) / 1e6
        print(f'[dump] {sc:28s} anchor {o["h_tgt"].shape[0]:2d}  '
              f'h_vid {o["h_vid"].shape}  {mb:.0f} MB', flush=True)


# ------------------------------------------------------------------ probe

class Bilinear(torch.nn.Module):
    """logits = (W_q h_tgt) . (W_k h_vid)^T / sqrt(d).  `notext` 면 질의를 학습 상수로."""

    def __init__(self, c_in, d=256, notext=False):
        super().__init__()
        self.wq = torch.nn.Linear(c_in, d, bias=False)
        self.wk = torch.nn.Linear(c_in, d, bias=False)
        self.d = d
        self.notext = notext
        if notext:
            self.const = torch.nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, ht, hv):
        """ht (B,C)  hv (B,N,C) -> logits (B,N)."""
        q = self.const.expand(ht.shape[0], -1) if self.notext else self.wq(ht)
        k = self.wk(hv)
        return torch.einsum('bd,bnd->bn', q, k) / self.d ** 0.5


def load_all(args):
    dd = osp.join(args.out_dir, 'probe_dump')
    data = {}
    for sc in args.scenes.split(','):
        fp = osp.join(dd, sc.split('/')[-1] + '.npz')
        if osp.exists(fp):
            data[sc.split('/')[-1]] = np.load(fp, allow_pickle=False)
    assert data, f'{dd} 에 dump 가 없다 — --stage dump 를 먼저'
    return data


def build_examples(data, li_idx, side, marginal, ghw=None):
    """(scene -> [(ht (C,), hv (N,C), target_cell, uv, label)]).

    `marginal=True` 면 프레임을 섞지 않고 **시간 marginal 한 장**을 만든다 — h_vid 를 시간축
    평균하고 타깃은 OBB 중앙값 셀. prefill 이 프레임 질의를 못 가지므로 이게 공정한 형태다.
    `False` 면 프레임마다 예제 하나 (h_vid 는 그 프레임 슬라이스).
    """
    ex = {}
    for sc, z in data.items():
        T, W, H = int(z['T']), float(z['W']), float(z['H'])
        gh, gw = (int(z['gh']), int(z['gw'])) if 'gh' in z else (side, side)
        hv = z['h_vid'][li_idx]                       # (T_eff, gh*gw, C)
        rows = []
        for ai in range(z['h_tgt'].shape[0]):
            ht = z['h_tgt'][ai, li_idx]
            uv = z['uv'][ai]
            ok = [f for f in range(min(T, uv.shape[0]))
                  if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W and 0 <= uv[f, 1] < H]
            if not ok:
                continue
            if marginal:
                u, vv = float(np.median(uv[ok, 0])), float(np.median(uv[ok, 1]))
                rc = (int(np.clip(vv * gh / H, 0, gh - 1)),
                      int(np.clip(u * gw / W, 0, gw - 1)))
                rows.append((ht, hv.mean(0), rc[0] * gw + rc[1], (u, vv),
                             str(z['labels'][ai]), f'{sc}#{ai}'))
            else:
                for f in ok:
                    rc = cell_of(uv[f, 0], uv[f, 1], side, W, H)
                    rows.append((ht, hv[f], rc[0] * side + rc[1],
                                 (float(uv[f, 0]), float(uv[f, 1])),
                                 str(z['labels'][ai]), f'{sc}#{ai}'))
        ex[sc] = rows
    return ex


def px_err(pred_cell, uv, side, W, H, gh=None, gw=None):
    gh = gh or side; gw = gw or side
    r_, c_ = divmod(int(pred_cell), gw)
    return float(np.hypot((c_ + .5) * W / gw - uv[0], (r_ + .5) * H / gh - uv[1]))


def fixed_cell_oracle(rows, side, W, H, gh=None, gw=None):
    """attention·hidden 무관 통제. 이 예제 집합에 대해 **한 셀** 고정의 최소 중앙값 오차."""
    gh = gh or side; gw = gw or side
    best = float('inf')
    for rr in range(gh):
        for cc in range(gw):
            e = [px_err(rr * gw + cc, uv, side, W, H, gh, gw)
                 for _h, _v, _t, uv, *_r in rows]
            best = min(best, float(np.median(e)))
    return best


def run_fold(train_rows, test_rows, mode, side, W, H, args, dev):
    C = train_rows[0][0].shape[0]
    ht_tr = torch.tensor(np.stack([x[0] for x in train_rows]), dtype=torch.float32)
    hv_tr = torch.tensor(np.stack([x[1] for x in train_rows]), dtype=torch.float32)
    y_tr = torch.tensor([x[2] for x in train_rows], dtype=torch.long)
    ht_te = torch.tensor(np.stack([x[0] for x in test_rows]), dtype=torch.float32)
    hv_te = torch.tensor(np.stack([x[1] for x in test_rows]), dtype=torch.float32)

    if mode == 'rawdot':
        with torch.no_grad():
            lg = torch.einsum('bc,bnc->bn', ht_te, hv_te) / C ** 0.5
        return lg.numpy()

    m = Bilinear(C, args.dim, notext=(mode == 'notext')).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    ht_tr, hv_tr, y_tr = ht_tr.to(dev), hv_tr.to(dev), y_tr.to(dev)
    n = len(y_tr)
    g = torch.Generator().manual_seed(0)
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = torch.nn.functional.cross_entropy(m(ht_tr[idx], hv_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lg = m(ht_te.to(dev), hv_te.to(dev))
    return lg.cpu().numpy()



def paired_acc(logits, rows, side, scene_of):
    """**1번(tinv)과 같은 지표로 probe 를 평가한다.** (all, same, diff) -> (정확도%, 시행).

    왜 필요한가: probe 를 절대 위치(셀 분류 / 픽셀 오차)로만 봤는데, D165 가 이 데이터에서
    **절대 위치 과제는 무엇으로도 실패한다**는 것을 이미 보였다 (피사체가 한 셀 안에 머물러
    고정셀 전략이 강하다 — attention 도 LOO 174.0px 로 똑같이 실패). 성공한 유일한 지표가
    쌍 비교였고 (attention LOO 79.0%), 그러니 probe 도 같은 자 위에 올려야 비교가 된다.

        logit[cell(OBB_i)] > logit[cell(OBB_j)] ?
        (i,j)/(j,i) 를 둘 다 세고 동점 0.5 -> chance 가 정확히 50%

    같은 **씬** 안의 예제끼리만 짝짓는다 (다른 씬은 배경이 달라 구별이 공짜다). 같은 셀인
    쌍은 제외하고, 라벨이 같으면 `same` 버킷으로 (우리 캡션이 실제로 감당하는 구별).

    **같은 anchor 끼리는 짝짓지 않는다** (행의 6번째 필드로 판정). `perframe` 에서는 한
    anchor 가 49개 예제를 내므로, 이걸 막지 않으면 "같은 물체를 프레임 t1 과 t2 에서
    구별하기" 가 `same` 버킷에 섞여 들어간다 — 인스턴스 구별 시험이 아니고, 게다가 쌍이
    n^2 으로 불어나 표준오차를 가짜로 줄인다 (실측: n 262 -> 625,884, +193 se).
    """
    by = {}
    for i, (rw, sc) in enumerate(zip(rows, scene_of)):
        by.setdefault(sc, []).append(i)
    buck = {k: [0.0, 0] for k in ('all', 'same', 'diff')}
    for sc, idxs in by.items():
        for i in idxs:
            for j in idxs:
                if i == j or rows[i][2] == rows[j][2]:
                    continue
                if len(rows[i]) > 5 and rows[i][5] == rows[j][5]:
                    continue                       # 같은 anchor 의 다른 프레임 — 제외
                bk = 'same' if rows[i][4] == rows[j][4] else 'diff'
                a, b = logits[i][rows[i][2]], logits[i][rows[j][2]]
                hit = 1.0 if a > b else (0.5 if a == b else 0.0)
                for key in ('all', bk):
                    buck[key][0] += hit
                    buck[key][1] += 1
    return {k: (v[0] / v[1] * 100 if v[1] else float('nan'), v[1])
            for k, v in buck.items()}


def stage_train(args):
    data = load_all(args)
    z0 = list(data.values())[0]
    layers = [int(x) for x in z0['layers']]
    side = int(z0['side'])
    gh = int(z0['gh']) if 'gh' in z0 else side
    gw = int(z0['gw']) if 'gw' in z0 else side
    print(f'격자 {gh}x{gw} = {gh * gw} 셀')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenes = sorted(data)
    folds = [scenes[i::args.folds] for i in range(args.folds)]
    print(f'층 {layers}   씬 {len(scenes)}   fold {args.folds} '
          f'(씬 단위) {[len(f) for f in folds]}   dev {dev}')

    res = {}
    for marginal in ([True, False] if args.both else [True]):
        tag = 'marginal' if marginal else 'perframe'
        print(f'\n{"=" * 96}\n[{tag}]  '
              f'{"시간 marginal 한 장 (prefill 에 공정한 형태)" if marginal else "프레임마다 예제"}'
              f'\n{"=" * 96}')
        print(f'{"층":>3s}  {"probe px/cell":>18s}  {"notext(통제)":>18s}  '
              f'{"고정셀":>8s} | {"쌍비교 all":>10s} {"notext":>8s} {"same":>8s} '
              f'{"diff":>8s}   (chance 50%)')
        for li_idx, li in enumerate(layers):
            ex = build_examples(data, li_idx, side, marginal, (gh, gw))
            W = float(list(data.values())[0]['W']); H = float(list(data.values())[0]['H'])
            MODES = ('probe', 'notext', 'rawdot')
            acc = {m: [] for m in MODES}
            err = {m: [] for m in MODES}
            pair = {m: {k: [0.0, 0] for k in ('all', 'same', 'diff')} for m in MODES}
            fx = []
            for fo in folds:
                te = [x for sc in fo for x in ex[sc]]
                te_sc = [sc for sc in fo for _ in ex[sc]]
                tr = [x for sc in scenes if sc not in fo for x in ex[sc]]
                if not te or not tr:
                    continue
                fx.append(fixed_cell_oracle(te, side, W, H, gh, gw))
                for mode in MODES:
                    lg = run_fold(tr, te, mode, side, W, H, args, dev)
                    pred = lg.argmax(1)
                    acc[mode] += [float(p == x[2]) for p, x in zip(pred, te)]
                    err[mode] += [px_err(p, x[3], side, W, H, gh, gw)
                                  for p, x in zip(pred, te)]
                    pa = paired_acc(lg, te, side, te_sc)
                    for k, (pc, n_) in pa.items():
                        if n_:
                            pair[mode][k][0] += pc / 100 * n_
                            pair[mode][k][1] += n_
            row = {m: dict(cell_acc=float(np.mean(acc[m])) * 100,
                           px_med=float(np.median(err[m])),
                           paired={k: dict(pct=(v[0] / v[1] * 100 if v[1] else None),
                                           n=int(v[1]))
                                   for k, v in pair[m].items()})
                   for m in MODES}
            row['fixedcell_px'] = float(np.median(fx))
            row['n'] = len(acc['probe'])
            res[f'{tag}_L{li}'] = row
            g = lambda m, k: (row[m]['paired'][k]['pct']
                              if row[m]['paired'][k]['pct'] is not None else float('nan'))
            print(f'{li:3d}  '
                  f'{row["probe"]["cell_acc"]:5.1f}% {row["probe"]["px_med"]:7.1f}px  '
                  f'{row["notext"]["cell_acc"]:5.1f}% {row["notext"]["px_med"]:7.1f}px  '
                  f'{row["fixedcell_px"]:7.1f}px | '
                  f'{g("probe", "all"):9.1f}% {g("notext", "all"):7.1f}% '
                  f'{g("probe", "same"):7.1f}% {g("probe", "diff"):7.1f}%', flush=True)
        best = min((k for k in res if k.startswith(tag)),
                   key=lambda k: res[k]['probe']['px_med'])
        b = res[best]
        print(f'\n  최고: {best}   probe {b["probe"]["px_med"]:.1f}px '
              f'/ {b["probe"]["cell_acc"]:.1f}%   '
              f'notext {b["notext"]["px_med"]:.1f}px / {b["notext"]["cell_acc"]:.1f}%   '
              f'고정셀 {b["fixedcell_px"]:.1f}px   n={b["n"]}')
        gap = b['notext']['px_med'] - b['probe']['px_med']
        print(f'  **텍스트 기여(절대)** = notext - probe = {gap:+.1f}px')
        bp = max((k for k in res if k.startswith(tag)),
                 key=lambda k: (res[k]['probe']['paired']['all']['pct'] or 0))
        q = res[bp]['probe']['paired']
        qn = res[bp]['notext']['paired']
        import math
        se = 100 * math.sqrt(0.25 / max(q['all']['n'], 1))
        print(f'  **쌍 비교 최고**: {bp}  all {q["all"]["pct"]:.1f}% '
              f'(n={q["all"]["n"]}, 1 s.e. {se:.2f}%p, {(q["all"]["pct"] - 50) / se:+.1f} se)'
              f'   notext {qn["all"]["pct"]:.1f}%')
        for k in ('same', 'diff'):
            if q[k]['pct'] is not None and q[k]['n']:
                s2 = 100 * math.sqrt(0.25 / q[k]['n'])
                print(f'      {k:5s} {q[k]["pct"]:5.1f}%  (n={q[k]["n"]}, '
                      f'{(q[k]["pct"] - 50) / s2:+.1f} se)   '
                      f'notext {qn[k]["pct"]:.1f}%' if qn[k]['pct'] is not None else '')
        print(f'      <- 1번(attention tinv) 은 같은 지표로 LOO all 79.0% / '
              f'same 67.6% / diff 84.2% 였다')

    fp = osp.join(args.out_dir, 'probe_result.json')
    json.dump(dict(layers=layers, folds=args.folds, dim=args.dim, epochs=args.epochs,
                   result=res), open(fp, 'w'), ensure_ascii=False, indent=1)
    print(f'\n[out] {fp}')


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    for st in args.stage.split(','):
        if st.strip() == 'dump':
            stage_dump(args)
        elif st.strip() == 'train':
            stage_train(args)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--stage', default='dump')
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scenes', default=','.join('vista4d/' + s for s in SCENES))
    p.add_argument('--layers', default='12,14,16,18,21,35')
    p.add_argument('--max_per_kind', type=int, default=8)
    p.add_argument('--min_frames', type=int, default=5)
    p.add_argument('--verb', default='Track')
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--overwrite', action='store_true', default=False)
    # probe
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--both', action='store_true', default=True)
    p.add_argument('--no_both', dest='both', action='store_false')
    # 52씬 x perframe 은 (290 anchor x 49 frame) x 81 x 2560 float32 = 11GB 라
    # `--no_both` 로 marginal 만 돌린다. prefill 에 공정한 형태도 marginal 이다.
    p.add_argument('--out_dir',
                   default='/data1/cympyc1785/LatentCamVid/tmp/d165_probe')
    p.add_argument('--gpu', default='4')
    main(p.parse_args())
