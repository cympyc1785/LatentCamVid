"""텍스트 형식 x 36층 격자를 훑어 **타겟 국소화가 가장 잘 되는 조합**을 찾는다 (D164).

  python scripts/viz/molmo2_attn_grid.py --gpu 1

무엇을 재나 — "타겟을 바꾸면 map 이 움직인다"만 보면 **노이즈가 이긴다** (아무 구조가 없어도
코사인이 낮게 나온다). 그래서 코퍼스의 `anchor_label` 로 물체 클래스를 알 수 있는 점을 쓴다:

    within   같은 클래스 쌍의 map 코사인 (낙타↔다른 낙타, 울타리↔다른 울타리) — **높아야** 함
    between  다른 클래스 쌍의 코사인 (낙타↔울타리, 낙타↔나무)                — **낮아야** 함
    score    within - between        (노이즈는 둘 다 낮추므로 score 를 못 올린다)

score 가 크면 "map 이 물체 정체성을 따라 묶인다" = 실제로 국소화하고 있다는 뜻이다.
보조로 video 질량(텍스트가 영상을 아예 보는가)도 같이 낸다.

효율: **층은 forward 1회로 36개가 다 나온다** (`Runner.maps_all_layers`). 그리고 ViT 는
텍스트와 무관하므로 씬당 1회만 돌고 prefix `inputs_embeds` 를 캐시한다. 따라서 비용은
씬 x 형식 x 타겟 회의 LM forward 뿐이다.

형식 축(`FORMATS`)은 지금까지의 실측을 그대로 담았다:
  concise/concise_probe  코퍼스 완성문 (= 학습이 실제로 쓰는 텍스트, probe 포함이 학습 조건)
  tgt_full/action/noun   타겟 서술만. 관계절·행위를 어디까지 남기나
  motion_noun            타겟은 noun 으로 두고 카메라 motion 절을 다시 붙인 대조군
  point/track/question   Molmo2 학습 분포형 지시문 (체크포인트 README 예시 그대로)
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

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, OUT, CORPUS   # noqa: E402
import cache_molmo2_embeddings as C                 # noqa: E402  (Runner 가 path 를 깔아 둔다)

SCENES = ['vista4d/camel', 'vista4d/car-roundabout', 'vista4d/basketball-four']

# 다른 물체를 지목하는 관계절을 자른다. 판별 기준은 전치사 뒤의 **정관사** —
# `in the background` / `along the fence` 는 씬의 다른 물체를 가리키므로 자르고,
# `in a striped shirt` / `with green leaves` 는 타겟 자신의 서술이라 남긴다.
_REL = (r'\s+(?:along|near|behind|beside|against|around|next\s+to|under|over|toward|'
        r'towards|from|in\s+front\s+of)\s+the\b')
_REL2 = r'\s+(?:on|in|at|of|by|to)\s+the\b'


def trim(ph, level='noun'):
    if level == 'full':
        return ph
    out = ph
    for pat in (_REL, _REL2):
        m = re.search(pat, out)
        if m:
            out = out[:m.start()]
    out = out.strip().rstrip(',')
    if level == 'noun':
        out = re.sub(r'\s+\w+ing$', '', out).strip().rstrip(',')   # 매달린 분사 제거
    return out or ph


def sent(ph):
    return ph[0].upper() + ph[1:] + '.'


def formats(cf, cap0, tt0):
    """이름 -> (phrase -> (text, span, probe)). phrase 는 원본 target_text 다."""
    mo = (cf.get('motion') or '').strip().rstrip('.')
    has_mo = bool(tt0) and tt0 in mo

    def _motion_noun(ph):
        assert has_mo, 'motion 이 target_text 를 담지 않는다 — motion_noun 불가'
        n = trim(ph, 'noun')
        return sent(mo.replace(tt0, n)), n, ''   # motion 절 안이라 소문자 그대로

    return {
        'concise':       lambda ph: (cap0.replace(tt0, ph), ph, ''),
        'concise_probe': lambda ph: (cap0.replace(tt0, ph), ph, C.PROBE),
        'tgt_full':      lambda ph: (sent(ph), sent(ph)[:-1], ''),
        'tgt_action':    lambda ph: (sent(trim(ph, 'action')),
                                     sent(trim(ph, 'action'))[:-1], ''),
        'tgt_noun':      lambda ph: (sent(trim(ph, 'noun')),
                                     sent(trim(ph, 'noun'))[:-1], ''),
        'motion_noun':   _motion_noun,
        'point':         lambda ph: (f'Point to {trim(ph, "noun")}.', trim(ph, 'noun'), ''),
        'track':         lambda ph: (f'Track {trim(ph, "noun")}', trim(ph, 'noun'), ''),
        'question':      lambda ph: (f'Where is {trim(ph, "noun")} in the video?',
                                     trim(ph, 'noun'), ''),
    }


def scene_targets(root, scene, max_n, preset):
    """(base 세그먼트, [(target_text, anchor_label)]) — 빈도순, base 의 것을 맨 앞."""
    pj = json.load(open(osp.join(root, scene, 'da3', 'prompts.json')))
    keys = sorted(pj, key=lambda x: int(x))
    base = next(((k, pj[k]) for k in keys if pj[k].get('preset') == preset), (keys[0], pj[keys[0]]))
    cnt = {}
    for k in keys:
        e = pj[k]
        t = ((e.get('caption_fields') or {}).get('target_text') or '').strip()
        lab = (e.get('anchor_label') or '').strip()
        if t and lab:
            cnt[(t, lab)] = cnt.get((t, lab), 0) + 1
    tt0 = ((base[1].get('caption_fields') or {}).get('target_text') or '').strip()
    order = sorted(cnt.items(), key=lambda x: -x[1])
    out = [p for p, _ in order if p[0] == tt0] + [p for p, _ in order if p[0] != tt0]
    return base, out[:max_n]


def score_maps(maps, labels):
    """within/between/score. maps: [K][(T,s,s)], labels: [K] 클래스."""
    V = np.stack([m.ravel() for m in maps])
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    cos = V @ V.T
    K = len(labels)
    wi, be = [], []
    for i in range(K):
        for j in range(i + 1, K):
            (wi if labels[i] == labels[j] else be).append(cos[i, j])
    w = float(np.mean(wi)) if wi else float('nan')
    b = float(np.mean(be)) if be else float('nan')
    return w, b, (w - b if wi and be else float('nan')), cos


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    scenes = args.scenes.split(',')
    per_scene = {}                     # scene -> {fmt: {layer: (w,b,score,vmass)}}
    prompts_seen = {}

    for sc in scenes:
        base, tgts = scene_targets(args.root, sc, args.max_targets, args.preset)
        cf = base[1].get('caption_fields') or {}
        cap0 = base[1]['prompt_camera_with_scene_video']['concise']
        tt0 = (cf.get('target_text') or '').strip()
        labs = [l for _, l in tgts]
        nclass = len(set(labs))
        if nclass < 2 or len(labs) == len(set(labs)):
            print(f'[skip] {sc}: 클래스 {nclass}, 같은 클래스 쌍이 없어 within 을 못 잰다')
            continue
        s, e = base[1]['frame_idx']
        r.set_video(scene=sc, start=int(s), num_frames=int(e) - int(s), fps=args.fps,
                    root=args.root)
        print(f'[scene] {sc}  preset={base[1].get("preset")}  targets {len(tgts)} '
              f'/ classes {nclass}')
        for t, l in tgts:
            print(f'         [{l:9}] {t}')

        F = formats(cf, cap0, tt0)
        per_scene[sc] = {}
        for fname, fn in F.items():
            try:
                probe0 = fn(tgts[0][0])[2]
            except AssertionError as ex:
                print(f'  [skip] {fname}: {ex}')
                continue
            maps_by_layer = None
            vmass_acc = {}
            ok = True
            for ph, _l in tgts:
                text, span, probe = fn(ph)
                try:
                    mp, vm, nrow = r.maps_all_layers(text, span=span,
                                                     span_tail=args.span_tail, probe=probe)
                except (AssertionError, Exception) as ex:      # noqa: BLE001
                    print(f'  [skip] {fname} / {ph[:40]!r}: {type(ex).__name__}: {ex}')
                    ok = False
                    break
                if maps_by_layer is None:
                    maps_by_layer = {li: [] for li in mp}
                    prompts_seen[fname] = (text, span, nrow, bool(probe))
                for li, m in mp.items():
                    maps_by_layer[li].append(m)
                for li, x in vm.items():
                    vmass_acc[li] = vmass_acc.get(li, 0.0) + x / len(tgts)
            if not ok:
                continue
            per_scene[sc][fname] = {
                li: (*score_maps(ms, labs)[:3], vmass_acc[li])
                for li, ms in maps_by_layer.items()}
            best = max(per_scene[sc][fname].items(), key=lambda kv: kv[1][2])
            print(f'  {fname:14s} best 층{best[0]:2d}  score {best[1][2]:+.3f} '
                  f'(within {best[1][0]:.3f} / between {best[1][1]:.3f})  '
                  f'video {best[1][3] * 100:.1f}%', flush=True)

    # ---------------- 씬 평균 집계
    fmts = sorted({f for sc in per_scene for f in per_scene[sc]})
    nl = r.nl
    S = np.full((len(fmts), nl), np.nan)
    W = np.full_like(S, np.nan); B = np.full_like(S, np.nan); VM = np.full_like(S, np.nan)
    for i, f in enumerate(fmts):
        for li in range(nl):
            xs = [per_scene[sc][f][li] for sc in per_scene if f in per_scene[sc]
                  and li in per_scene[sc][f]]
            if xs:
                W[i, li] = np.mean([x[0] for x in xs])
                B[i, li] = np.mean([x[1] for x in xs])
                S[i, li] = np.mean([x[2] for x in xs])
                VM[i, li] = np.mean([x[3] for x in xs])

    print(f'\n{"=" * 96}\n씬 {len(per_scene)}개 평균  score = within - between  (클수록 좋다)\n{"=" * 96}')
    print(f'{"format":15s} ' + ' '.join(f'{li:5d}' for li in range(0, nl, 2)))
    for i, f in enumerate(fmts):
        print(f'{f:15s} ' + ' '.join(
            ('  .  ' if np.isnan(S[i, li]) else f'{S[i, li]:+.2f}') for li in range(0, nl, 2)))

    flat = [(S[i, li], i, li) for i in range(len(fmts)) for li in range(nl)
            if not np.isnan(S[i, li])]
    flat.sort(reverse=True)
    print(f'\n상위 15 조합\n{"순위":>3s} {"format":15s} {"층":>3s} {"score":>7s} '
          f'{"within":>7s} {"between":>8s} {"video%":>7s}')
    print('-' * 62)
    for k, (sv, i, li) in enumerate(flat[:15]):
        print(f'{k + 1:3d} {fmts[i]:15s} {li:3d} {sv:+7.3f} {W[i, li]:7.3f} '
              f'{B[i, li]:8.3f} {VM[i, li] * 100:6.1f}%')

    print(f'\n형식별 최고 (층 무관)\n{"format":15s} {"층":>3s} {"score":>7s} {"video%":>7s}')
    print('-' * 38)
    for i in np.argsort(-np.nanmax(S, 1)):
        li = int(np.nanargmax(S[i]))
        print(f'{fmts[i]:15s} {li:3d} {S[i, li]:+7.3f} {VM[i, li] * 100:6.1f}%')

    print(f'\n층별 최고 (형식 무관) — 판별 구간 확인')
    for li in range(nl):
        col = S[:, li]
        if np.all(np.isnan(col)):
            continue
        i = int(np.nanargmax(col))
        bar = '#' * max(0, int(round(np.nanmax(col) * 40)))
        print(f'  층{li:2d} {np.nanmax(col):+.3f} {fmts[i]:15s} {bar}')

    js = dict(scenes=list(per_scene), formats=fmts, layers=nl,
              prompts={k: dict(text=v[0], span=v[1], query_tokens=v[2], probe=v[3])
                       for k, v in prompts_seen.items()},
              score=S.tolist(), within=W.tolist(), between=B.tolist(), video_mass=VM.tolist(),
              per_scene={sc: {f: {str(li): list(v) for li, v in d.items()}
                              for f, d in per_scene[sc].items()} for sc in per_scene})
    jp = osp.join(args.out_dir, f'grid_{args.tag}.json')
    json.dump(js, open(jp, 'w'), ensure_ascii=False, indent=1)
    print(f'\n[out] {jp}')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scenes', default=','.join(SCENES))
    p.add_argument('--preset', default='dolly_in_look_at')
    p.add_argument('--max_targets', type=int, default=5)
    p.add_argument('--span_tail', type=int, default=0)
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--tag', default='textxlayer')
    p.add_argument('--gpu', default='1')
    main(p.parse_args())
