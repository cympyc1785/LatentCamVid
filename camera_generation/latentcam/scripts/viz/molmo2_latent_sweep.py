"""어느 latent 를 카메라 디코더에 꽂아야 하는가 — lever sweep (D164, fps 5 고정).

  python scripts/viz/molmo2_latent_sweep.py --gpu 2

## 왜 이 설계인가

이 체크포인트(`Molmo2-4B`)에는 **point head 가 없다.** 클래스는 `Molmo2ForConditionalGeneration`
하나뿐이고 `patch_logits`/`subpatch_logits`/`location_logits` 가 존재하지 않는다 (grep 확인).
pointing/tracking 은 별도 head 가 아니라 **좌표를 텍스트 토큰으로 생성**해서 한다
(`allenai/Molmo2-VideoPoint` 는 README 가 나열한 **별도 아티팩트**다).

그래서 subpatch head 를 탭할 수는 없다. 대신 **생성된 좌표가 pseudo-GT** 로 쓸 수 있다 —
D164 에서 5씬 17타겟이 전부 올바른 물체를 짚는 것을 확인했다. 그러면 질문이 정확해진다:

    "각 latent 에서 대상의 프레임별 위치가 **얼마나 바로 읽히는가**"

이건 attention 코사인 같은 대리 지표가 아니라 픽셀 오차로 답이 나온다. 그리고 이게 정확히
카메라 디코더가 필요로 하는 정보다.

## 읽어내는 방식 (학습 파라미터 0개)

프레임 f 에서 텍스트 쿼리 q 와 그 프레임의 81개 patch latent 를 맞대어 9x9 분포를 만들고
그 기대 위치를 예측으로 삼는다. **어떤 것도 학습하지 않는다** — latent 안에 이미 있으면
내적만으로 나온다.

    readout `attn`  LM 의 실제 attention (text 행 -> patch 열). 지금까지 보던 것
    readout `dot`   q · h_patch
    readout `cos`   코사인
    readout `pc1`   patch latent 의 상위 1개 주성분을 제거한 뒤 내적
    readout `pc4`   상위 4개 제거. **global/sink 성분이 argmax 를 먹는지** 가른다
                    (프레임 평균을 빼는 것은 프레임당 상수 이동이라 argmax 를 바꾸지 못한다 —
                     상수 shift 가 아닌 방향 제거여야 의미가 있다)
    readout `attnv` attention x ||v|| (Kobayashi et al. 2020 "Attention is Not Only a Weight")

## sweep 축

    layer        0..35 (LM hidden). ViT 출력(1152-d)은 텍스트와 차원이 달라 이 readout 불가
    query        target 구간 마지막 토큰 / target 구간 평균 / tail 전체 평균
    readout      attn / dot / cos / cdot

## 지표

    err_px       argmax 셀 중심과 pseudo-GT 의 픽셀 거리 (평균)
    err_soft     softmax 기대 위치와의 거리
    hit@1cell    1셀(42px) 안에 들어간 비율
    chance       9x9 균등 분포의 기대 오차 (비교 기준)

**주의**: pseudo-GT 는 모델 자신의 생성 출력이다. 따라서 이 sweep 이 재는 것은
"모델이 이미 계산해 둔 위치가 어느 latent 에서 선형적으로 노출되는가" 이고,
절대 정확도가 아니다. 하지만 카메라 디코더에 무엇을 꽂을지 고르는 데는 이게 맞는 질문이다.
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import re
import sys

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, OUT, CORPUS          # noqa: E402
from molmo2_attn_grid import trim, scene_targets           # noqa: E402
from molmo2_point_track import extract_video_points, pick_main   # noqa: E402
import cache_molmo2_embeddings as C                        # noqa: E402

SCENES = ['vista4d/camel', 'vista4d/car-roundabout', 'vista4d/basketball-four',
          'vista4d/couple-walk', 'vista4d/golf', 'vista4d/cows',
          'vista4d/couple-rocks', 'vista4d/fashion-walk']
QUERIES = ('tgt_last', 'tgt_mean', 'tail_mean')
READOUTS = ('attn', 'attnv', 'dot', 'cos', 'pc1', 'pc4')


# ------------------------------------------------------------------ pseudo-GT

@torch.inference_mode()
def gen_track(r, video, prompt, fps, max_new=1024):
    """`Track {t}` 를 생성해 (frame_idx -> (x,y)) 를 돌려준다. ViT 를 다시 돈다(생성 경로)."""
    T, H, W = video.shape[:3]
    b = r._batch(video, prompt, '', fps)
    b = {k: (v.to(r.device, r.dtype) if k == 'pixel_values_videos' else v.to(r.device))
         for k, v in b.items()}
    g = r.model.generate(**b, max_new_tokens=max_new, do_sample=False)
    txt = r.proc.tokenizer.decode(g[0, b['input_ids'].shape[1]:], skip_special_tokens=True)
    pts = extract_video_points(txt, W, H)
    out = {}
    for t, x, y in pts:
        out[min(max(int(round(t * fps)), 0), T - 1)] = (x, y)
    return out, txt


# ------------------------------------------------------------------ latent 포획

@torch.inference_mode()
def probe(r, text, span):
    """forward 1회로 **36층 attention + 37층 hidden** 을 동시에 잡는다.

    prefix(=ViT 출력)는 씬당 1회 캐시된 것을 재사용한다 (`Runner.set_video`).
    반환: attn[li] (H,R,L) / hp[li] (T,81,D) / hq[li] (R,D)  — 전부 GPU."""
    v, core = r.vid, r.model.model
    tstr = r._tail_str(text, '')
    tids = r.proc.tokenizer(tstr, add_special_tokens=False,
                            return_tensors='pt')['input_ids']
    rows_t, _ = r._rows(tstr, span, 0)
    rows_all, _ = r._rows(tstr, 'all', 0)
    P, T, side = v['P'], v['T'], v['side']
    store = {}
    temb, _ = core.build_input_embeddings(tids.to(r.device))
    emb = torch.cat([v['prefix_emb'], temb], 1)
    restore = r._hook(None, [P + i for i in rows_all], store)
    try:
        out = core(inputs_embeds=emb,
                   attention_mask=torch.ones(1, emb.shape[1], dtype=torch.long,
                                             device=r.device),
                   use_cache=False, output_hidden_states=True)
    finally:
        restore()
    hs = out.hidden_states                       # 37개 (embedding + 36층)
    pos = v['patch_pos'].to(r.device)
    hp = {li: hs[li + 1][0, pos].view(T, side * side, -1) for li in range(len(hs) - 1)}
    hq = {li: hs[li + 1][0, P:] for li in range(len(hs) - 1)}
    # attention 은 rows_all 기준으로 잡혔다 -> target 구간의 상대 인덱스
    ridx = [rows_all.index(i) for i in rows_t]
    return store, hp, hq, ridx, len(rows_all), tstr, rows_t


# ------------------------------------------------------------------ readout

def grid_to_px(rr, cc, side, W, H):
    return (cc + .5) * W / side, (rr + .5) * H / side


def eval_map(m, gt, side, W, H):
    """m: (T,side,side) 점수. gt: {frame:(x,y)} -> (err_argmax, err_f0, hit1, hit2, n)"""
    ea, hit, hit2, n, e0 = 0.0, 0, 0, 0, np.nan
    cell = float(np.hypot(W / side, H / side)) / 2       # 1셀 반경
    for f, (gx, gy) in sorted(gt.items()):
        rr, cc = np.unravel_index(int(np.argmax(m[f])), m[f].shape)
        ax, ay = grid_to_px(rr, cc, side, W, H)
        da = float(np.hypot(ax - gx, ay - gy))
        if n == 0:
            e0 = da
        ea += da; hit += int(da <= cell); hit2 += int(da <= 2 * cell); n += 1
    return (ea / n, e0, hit / n, hit2 / n, n) if n else (np.nan,) * 4 + (0,)


def baselines(gt, side, W, H):
    """비교 기준 세 개. readout 은 최소한 `center` 를 이겨야 의미가 있다.

    grid   9x9 균등 분포의 기대 오차 (가장 느슨)
    center 항상 화면 중심을 찍는 것 — 피사체가 대개 가운데 있으므로 이게 진짜 기준선이다
    fixed  GT 전체의 평균 위치를 찍는 것 (시간 정보 0, oracle 상수) — 이걸 이겨야 '추적'이다"""
    xs, ys = grid_to_px(*np.meshgrid(np.arange(side), np.arange(side), indexing='ij'),
                        side=side, W=W, H=H)
    g = np.array(list(gt.values()), float)
    mu = g.mean(0)
    return dict(grid=float(np.mean([np.mean(np.hypot(xs - x, ys - y)) for x, y in g])),
                center=float(np.mean(np.hypot(g[:, 0] - W / 2, g[:, 1] - H / 2))),
                fixed=float(np.mean(np.hypot(g[:, 0] - mu[0], g[:, 1] - mu[1]))))


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    layers = ([int(x) for x in args.layers.split(',')] if args.layers
              else list(range(r.nl)))
    acc = {}          # (query, readout, layer) -> [errs]
    meta = []
    for scene in args.scenes.split(','):
        base, tg = scene_targets(args.root, scene, 12, args.preset)
        if args.main:
            tg = [pick_main(tg)]
        else:
            tg = tg[:args.max_targets]
        s0, e0 = base[1]['frame_idx']
        st, nf = int(s0), int(e0) - int(s0)
        r.set_video(scene=scene, start=st, num_frames=nf, fps=args.fps, root=args.root)
        v = r.vid
        T, side = v['T'], v['side']
        W, H = v['video'].shape[2], v['video'].shape[1]
        for tt, lab in tg:
            ph = tt if args.trim_level == 'full' else trim(tt, args.trim_level)
            prompt = f'{args.verb} {ph}'
            gt, raw = gen_track(r, v['video'], prompt, args.fps)
            if len(gt) < args.min_pts:
                print(f'  [skip] {scene} [{lab}] 점 {len(gt)} < {args.min_pts}  {ph[:44]}')
                continue
            bl = baselines(gt, side, W, H)
            store, hp, hq, ridx, nall, tstr, rows_t = probe(r, prompt, ph)
            best = None
            # 층을 밖으로 뺀다 — pc 기저는 query 와 무관하므로 층당 1회만 구한다
            # (query 마다 SVD 를 돌리면 36x3x2 회가 되어 못 돌린다)
            for li in layers:
                hpl0 = hp[li].float()                       # (T,81,D)
                variants = {'dot': hpl0}
                variants['cos'] = hpl0 / hpl0.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                X = hpl0.reshape(-1, hpl0.shape[-1])
                X = X - X.mean(0, keepdim=True)
                _u, _s, V = torch.pca_lowrank(X, q=6, center=False)   # 상위 6 주성분
                for ro_, k in (('pc1', 1), ('pc4', 4)):
                    B = V[:, :k].T                          # (k,D)
                    variants[ro_] = hpl0 - (hpl0 @ B.T) @ B
                pnorm = hpl0.norm(dim=-1).reshape(1, 1, -1).cpu()     # attnv 가중치
                for q in QUERIES:
                    if q == 'tgt_last':
                        sel = [ridx[-1]]
                    elif q == 'tgt_mean':
                        sel = ridx
                    else:
                        sel = list(range(nall))
                    qv0 = hq[li][sel].float().mean(0)
                    for ro in READOUTS:
                        if ro in ('attn', 'attnv'):
                            w = store[li][:, sel, :][:, :, v['patch_pos']]
                            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                            if ro == 'attnv':
                                w = w * pnorm
                            m = w.mean((0, 1)).view(T, side, side).float().numpy()
                        else:
                            hv_ = variants[ro]
                            qv = qv0
                            if ro == 'cos':
                                qv = qv / qv.norm().clamp_min(1e-9)
                            elif ro in ('pc1', 'pc4'):
                                B = V[:, :(1 if ro == 'pc1' else 4)].T
                                qv = qv - (qv @ B.T) @ B
                            m = (hv_ @ qv).view(T, side, side).cpu().numpy()
                        ea, e0, h1, h2, n = eval_map(m, gt, side, W, H)
                        acc.setdefault((q, ro, li), []).append(
                            (ea, e0, h1, h2, bl['center'], bl['fixed'], bl['grid']))
                        if best is None or ea < best[0]:
                            best = (ea, q, ro, li)
                del variants, X, V
            meta.append(dict(scene=scene, label=lab, phrase=ph, n_gt=len(gt), **bl,
                             best_err=best[0], best=f'{best[1]}/{best[2]}/L{best[3]}'))
            print(f'  {scene:24s} [{lab:8s}] gt {len(gt):2d}점  '
                  f'center {bl["center"]:5.0f} / fixed {bl["fixed"]:5.0f} / grid {bl["grid"]:5.0f}px'
                  f'   best {best[0]:5.1f}px  {best[1]}/{best[2]}/L{best[3]}', flush=True)
            del store, hp, hq
            torch.cuda.empty_cache()

    # ---------------- 집계
    rows = []
    for (q, ro, li), vv in acc.items():
        a = np.array(vv, float)
        rows.append((q, ro, li, *a.mean(0), len(vv)))
    rows.sort(key=lambda x: x[3])
    C0, F0, G0 = (float(np.mean([m['center'] for m in meta])),
                  float(np.mean([m['fixed'] for m in meta])),
                  float(np.mean([m['grid'] for m in meta])))
    print(f'\n{"=" * 92}\n샘플 {len(meta)}개   기준선: center {C0:.0f}px / '
          f'fixed(GT평균) {F0:.0f}px / grid(균등) {G0:.0f}px\n{"=" * 92}')
    print(f'{"query":9s} {"readout":8s} {"층":>3s} {"err_px":>7s} {"err_f0":>7s} '
          f'{"hit@1":>6s} {"hit@2":>6s}   vs center')
    for q, ro, li, ea, e0, h1, h2, cc, ff, gg, n in rows[:20]:
        print(f'{q:9s} {ro:8s} {li:3d} {ea:7.1f} {e0:7.1f} {h1 * 100:5.0f}% {h2 * 100:5.0f}%'
              f'   {"이김 " if ea < cc else "짐   "}({ea - cc:+.0f}px)')

    print(f'\nreadout 별 최고\n{"readout":8s} {"query":9s} {"층":>3s} {"err_px":>7s} '
          f'{"err_f0":>7s} {"hit@1":>6s}   vs center')
    for ro in READOUTS:
        c = [x for x in rows if x[1] == ro]
        if c:
            q, _r, li, ea, e0, h1, h2, cc, ff, gg, n = c[0]
            print(f'{ro:8s} {q:9s} {li:3d} {ea:7.1f} {e0:7.1f} {h1 * 100:5.0f}%'
                  f'   {ea - cc:+.0f}px')

    print(f'\n층별 최고 (query/readout 무관)  [center 기준선 {C0:.0f}px]')
    for li in layers:
        c = [x for x in rows if x[2] == li]
        if not c:
            continue
        q, ro, _l, ea, e0, h1, h2, cc, ff, gg, n = c[0]
        bar = '#' * max(0, int(round((cc - ea) / cc * 40)))
        print(f'  층{li:2d} {ea:6.1f}px (f0 {e0:5.1f})  {q:9s} {ro:6s} {bar}')
    json.dump(dict(scenes=args.scenes, fps=args.fps, samples=meta,
                   grid=[dict(query=q, readout=ro, layer=li, err_px=ea, err_f0=e0,
                              hit1=h1, hit2=h2, center=cc, fixed=ff, gridbl=gg, n=n)
                         for q, ro, li, ea, e0, h1, h2, cc, ff, gg, n in rows]),
              open(osp.join(args.out_dir, f'latent_sweep_{args.tag}.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n[out] {osp.join(args.out_dir, f"latent_sweep_{args.tag}.json")}')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scenes', default=','.join(SCENES))
    p.add_argument('--preset', default='dolly_in_look_at')
    p.add_argument('--main', action='store_true', help='씬당 주 피사체 1개만')
    p.add_argument('--max_targets', type=int, default=3)
    p.add_argument('--trim_level', default='full', choices=('full', 'action', 'noun'))
    p.add_argument('--verb', default='Track')
    p.add_argument('--fps', type=float, default=5.0)   # D164: 4~5 가 최적
    p.add_argument('--layers', default='', help='쉼표 구분. 비우면 전체 36층')
    p.add_argument('--min_pts', type=int, default=6)
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--tag', default='fps5')
    p.add_argument('--gpu', default='2')
    main(p.parse_args())
