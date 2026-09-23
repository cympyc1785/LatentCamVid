"""Molmo2 생성 좌표 ↔ scene_graph OBB track 2D 투영 일치도 (D164).

  python scripts/viz/molmo2_vs_obb.py --gpu 2

왜: 앞선 논의의 결론이 "Molmo2 = 텍스트→인스턴스 **선택기**, OBB track = **궤적**" 이었다.
그 분업이 성립하려면 **둘이 같은 물체를 가리키는지**를 재야 한다. 동시에 이 숫자는
OBB track 의 노이즈 크기를 처음으로 정량화한다 (aux loss 라벨 품질).

무엇을 비교하나
  OBB   `<scene>/da3/target_track.npz` 의 `track_world` (V,T,3) DA3 world 좌표를
        `pose.npz` 의 `extrinsics`(w2c, (T,3,4)) / `intrinsics`((T,3,3), cx,cy=320,180 =
        640x360 규약) 로 투영 -> 프레임별 (u,v) 픽셀
  Molmo2 `Track {target_text}` 생성 -> 0.5초 격자 좌표 (fps 5 -> 20점) -> 프레임 인덱스

주의
  · 이건 **정답 대조가 아니다.** 둘 다 추정이다. 불일치는 어느 쪽이 틀렸는지 말해주지 않는다.
    다만 **일치하면 둘 다 같은 물체를 봤다는 강한 증거**이고, 불일치는 그 anchor 의 텍스트
    서술이 애매하다는 신호다.
  · anchor 를 `dyn_*`(움직이는 피사체) / `stat_*`(정적 배경) 로 나눠 본다 — 실측상 이 축에서
    갈린다.
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import sys

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, OUT, CORPUS      # noqa: E402
from molmo2_latent_sweep import gen_track              # noqa: E402
import cache_molmo2_embeddings as C                    # noqa: E402


def project(track_world, E, K):
    """(T,3) DA3 world -> (T,2) 픽셀 + (T,) depth. `E` 는 w2c (T,3,4)."""
    T = track_world.shape[0]
    uv = np.full((T, 2), np.nan, np.float64)
    z = np.full(T, np.nan, np.float64)
    for f in range(T):
        Xc = E[f][:, :3] @ track_world[f] + E[f][:, 3]
        if Xc[2] <= 1e-6:
            continue
        uv[f] = (K[f][0, 0] * Xc[0] / Xc[2] + K[f][0, 2],
                 K[f][1, 1] * Xc[1] / Xc[2] + K[f][1, 2])
        z[f] = Xc[2]
    return uv, z


def scene_anchors(root, scene, max_per_kind):
    """anchor 별 대표 variant. (anchor_id, key, target_text, label, track_world) 목록."""
    d = osp.join(root, scene, 'da3')
    tt = np.load(osp.join(d, 'target_track.npz'), allow_pickle=False)
    pj = json.load(open(osp.join(d, 'prompts.json')))
    seen, out = {}, []
    for i, (k, a, v) in enumerate(zip(tt['keys'], tt['anchor_id'], tt['valid'])):
        a, k = str(a), str(k)
        if not v or a in seen or k not in pj:
            continue
        cf = pj[k].get('caption_fields') or {}
        txt = (cf.get('target_text') or '').strip()
        if not txt:
            continue
        seen[a] = i
        out.append((a, k, txt, (pj[k].get('anchor_label') or '').strip(),
                    tt['track_world'][i]))
    dyn = [r for r in out if r[0].startswith('dyn')][:max_per_kind]
    sta = [r for r in out if not r[0].startswith('dyn')][:max_per_kind]
    return dyn + sta


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    rows = []
    for scene in args.scenes.split(','):
        d = osp.join(args.root, scene, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            print(f'[skip] {scene}: target_track.npz 없음 — export_target_track.py 먼저')
            continue
        p = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p['extrinsics'], p['intrinsics']
        anch = scene_anchors(args.root, scene, args.max_per_kind)
        if not anch:
            continue
        r.set_video(scene=scene, start=0, num_frames=E.shape[0], fps=args.fps,
                    root=args.root)
        T = r.vid['T']
        H, W = r.vid['video'].shape[1], r.vid['video'].shape[2]
        print(f'\n### {scene}  anchors {len(anch)}  T {T}  {W}x{H}', flush=True)
        for a, key, txt, lab, tw in anch:
            uv, z = project(tw, E, K)
            gt, _raw = gen_track(r, r.vid['video'], f'{args.verb} {txt}', args.fps)
            if not gt:
                print(f'  [{a:7s}] 좌표 생성 실패  {txt[:44]}')
                continue
            fs = sorted(f for f in gt if not np.isnan(uv[f, 0]))
            if not fs:
                continue
            dd = np.array([np.hypot(gt[f][0] - uv[f, 0], gt[f][1] - uv[f, 1]) for f in fs])
            d0 = dd[0]
            inb = float(np.mean([(0 <= uv[f, 0] < W) and (0 <= uv[f, 1] < H) for f in fs]))
            rows.append(dict(scene=scene, anchor=a, label=lab, text=txt,
                             kind='dyn' if a.startswith('dyn') else 'stat',
                             n=len(fs), d0=float(d0), dmean=float(dd.mean()),
                             dmed=float(np.median(dd)), obb_inframe=inb,
                             depth_med=float(np.nanmedian(z[fs])),
                             molmo0=list(gt[fs[0]]), obb0=list(uv[fs[0]])))
            print(f'  [{a:7s} {lab:8s}] n{len(fs):3d}  Δf0 {d0:6.1f}px  Δmean {dd.mean():6.1f}px  '
                  f'Δmed {np.median(dd):6.1f}px  depth {np.nanmedian(z[fs]):5.2f}  {txt[:38]}',
                  flush=True)

    if not rows:
        raise SystemExit('비교할 게 없다')
    print(f'\n{"=" * 88}\n요약 (n={len(rows)} anchor)\n{"=" * 88}')
    for kind in ('dyn', 'stat', None):
        sub = [x for x in rows if kind is None or x['kind'] == kind]
        if not sub:
            continue
        dm = np.array([x['dmean'] for x in sub]); d0 = np.array([x['d0'] for x in sub])
        nm = 'dyn (움직이는 피사체)' if kind == 'dyn' else \
             ('stat (정적 배경)' if kind == 'stat' else '전체')
        print(f'{nm:22s} n{len(sub):3d}  Δf0  med {np.median(d0):6.1f} p90 {np.percentile(d0, 90):6.1f}  '
              f'|  Δmean  med {np.median(dm):6.1f} p90 {np.percentile(dm, 90):6.1f}  '
              f'|  <50px {np.mean(dm < 50) * 100:3.0f}%  <100px {np.mean(dm < 100) * 100:3.0f}%')
    print(f'\n불일치 큰 anchor (Δmean 상위 8)')
    for x in sorted(rows, key=lambda x: -x['dmean'])[:8]:
        print(f"  {x['dmean']:7.1f}px  {x['scene'].split('/')[-1]:18s} {x['anchor']:7s} "
              f"{x['label']:8s}  molmo2 {tuple(round(v) for v in x['molmo0'])} vs "
              f"obb {tuple(round(v) for v in x['obb0'])}  {x['text'][:34]}")
    json.dump(rows, open(osp.join(args.out_dir, f'molmo2_vs_obb_{args.tag}.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"\n[out] {osp.join(args.out_dir, f'molmo2_vs_obb_{args.tag}.json')}")


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scenes', default=','.join(
        'vista4d/' + s for s in ('camel', 'car-roundabout', 'basketball-four', 'couple-walk',
                                 'golf', 'cows', 'couple-rocks', 'fashion-walk')))
    p.add_argument('--max_per_kind', type=int, default=2)
    p.add_argument('--verb', default='Track')
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--tag', default='fps5')
    p.add_argument('--gpu', default='2')
    main(p.parse_args())
