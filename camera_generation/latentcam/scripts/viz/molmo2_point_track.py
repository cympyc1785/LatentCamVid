"""Molmo2 의 **point / track 생성 출력**을 실제로 뽑아 본다 (D164 대조군).

  python scripts/viz/molmo2_point_track.py --gpu 2 --prompt "Track the larger pale camel"

왜 필요한가: attention map 진단만으로는 "**attention 이라는 렌즈가 눈이 먼 것**"과
"**모델이 애초에 대상을 모르는 것**"을 못 가른다. raw attention 은 약한 설명 도구라
정보가 value 경로로 흐를 수 있다. Molmo2 는 pointing/tracking 이 학습 태스크이므로,
**생성 출력이 실제로 대상을 짚는지**가 그 둘을 가르는 유일한 직접 증거다.

디코딩은 체크포인트 `README.md` 의 `extract_video_points` 를 글자 그대로 옮겼다 —
좌표는 원본 프레임 기준 0~1000 스케일이고 `<points .../>` / `<tracks .../>` 태그의
`coords` 속성에 프레임별로 들어온다.

출력:
  · 생성 원문 그대로 (형식을 눈으로 봐야 한다)
  · (frame, x, y) 목록과 프레임 커버리지
  · 예측 점을 올린 mp4. `--attn_npy` 로 `molmo2_attn_video.py` 가 남긴 `(T,9,9)` 를 주면
    attention argmax 와 **같은 프레임에 나란히** 그려 둘이 일치하는지 볼 수 있다.
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
import imageio.v2 as iio
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from molmo2_attn_grid import trim, scene_targets   # noqa: E402 — 관계절 제거·타겟 수집 재사용

REPO = '/data1/cympyc1785/LatentCamVid'
sys.path.insert(0, osp.join(REPO, 'camera_generation/latentcam/main'))
import cache_molmo2_embeddings as C   # noqa: E402

CORPUS = '/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121'
OUT = osp.join(REPO, 'tmp/molmo2_attn')

# `--main` 이 고를 주 피사체 우선순위. 코퍼스의 최빈 target 은 주 피사체가 아닐 수 있다
# (car-roundabout 의 최빈은 `right front wheel` 88건, 주 피사체 `large dark grey vehicle` 은 86건).
# part(wheel) 나 배경(fence/tree/building) 보다 움직이는 개체를 앞세운다.
SUBJ_PRIO = ['vehicle', 'car', 'truck', 'person', 'human', 'man', 'woman', 'people',
             'camel', 'cow', 'goat', 'horse', 'dog', 'bike', 'people']

# ---- 체크포인트 README.md 의 디코더 (글자 그대로) ----------------------------
COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def _points_from_num_str(text, w, h):
    for m in POINTS_REGEX.finditer(text):
        ix, x, y = m.group(1), m.group(2), m.group(3)
        x, y = float(x) / 1000 * w, float(y) / 1000 * h     # 좌표는 1000 스케일
        if 0 <= x <= w and 0 <= y <= h:
            yield ix, x, y


def extract_video_points(text, w, h, extract_ids=False):
    out = []
    for coord in COORD_REGEX.finditer(text):
        for grp in FRAME_REGEX.finditer(coord.group(1)):
            fid = float(grp.group(1))
            for idx, x, y in _points_from_num_str(grp.group(2), w, h):
                out.append((fid, idx, x, y) if extract_ids else (fid, x, y))
    return out
# -----------------------------------------------------------------------------


def draw_cross(img, cx, cy, color, r=11, t=2):
    h, w = img.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))
    if not (0 <= cx < w and 0 <= cy < h):
        return
    img[max(0, cy - t):cy + t, max(0, cx - r):cx + r] = color
    img[max(0, cy - r):cy + r, max(0, cx - t):cx + t] = color


def main(args):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    proc = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, dtype=dtype, trust_remote_code=True).to(device).eval()

    if args.fps_sweep:
        return fps_sweep(args, proc, model, device, dtype)
    allsc = {}
    for scene in args.scene:
        if args.main:
            base, tg = scene_targets(args.root, scene, 12, args.preset)
            tt, lab = pick_main(tg)
            ph = tt if args.trim_level == 'full' else trim(tt, args.trim_level)
            s0, e0 = base[1]['frame_idx']
            st, nf = int(s0), int(e0) - int(s0)
            prompts = [(f'{args.verb} {ph}', lab)]
        elif args.auto_targets:
            base, tg = scene_targets(args.root, scene, args.auto_targets, args.preset)
            s0, e0 = base[1]['frame_idx']
            st, nf = int(s0), int(e0) - int(s0)
            prompts = [(f'{args.verb} {trim(t, args.trim_level)}', l) for t, l in tg]
        else:
            st, nf = args.start, args.num_frames
            prompts = [(q, '?') for q in args.prompt]
        video = C.load_frames(args.root, scene, st, st + nf)
        T, H, W = video.shape[:3]
        attn = np.load(args.attn_npy) if args.attn_npy else None
        if attn is not None:
            assert attn.shape[0] == T, f'attn {attn.shape} 의 프레임 수가 {T} 와 다르다'
        print(f'\n{"#" * 84}\n### {scene}   frames {st}..{st + nf}  {H}x{W}  '
              f'targets {len(prompts)}\n{"#" * 84}', flush=True)
        rows = []
        for pr, lab in prompts:
            rr = one(args, proc, model, device, dtype, video, T, H, W, pr, attn, scene)
            rr['label'] = lab if lab != '?' else rr['label']
            rows.append(rr)
        allsc[scene] = dict(rows=rows, W=W, H=H, T=T)
        allsc[scene]['_an'] = analyze(scene, rows, W, H)
    if len(allsc) > 1:
        print(f'\n{"=" * 84}\n씬별 요약\n{"=" * 84}')
        print(f'{"scene":26s} {"targets":>7s} {"within":>8s} {"between":>8s} {"ratio":>7s} '
              f'{"pts/타겟":>8s} {"step":>6s}')
        for sc, d in allsc.items():
            a = d['_an']
            print(f'{sc:26s} {len(d["rows"]):7d} {a["within"]:8.1f} {a["between"]:8.1f} '
                  f'{a["ratio"]:7.2f} {a["npts"]:8.1f} {a["step"]:6.1f}')
        print('\nwithin = 같은 클래스 타겟쌍의 t=0 좌표 거리(px), between = 다른 클래스 쌍.')
        print('ratio = between/within. 크면 "다른 물체를 다른 곳으로 짚는다" = track 이 되는 것.')
    json.dump({sc: dict(W=d['W'], H=d['H'], T=d['T'], analysis=d['_an'],
                        rows=[{k: v for k, v in r.items() if k != 'raw'} for r in d['rows']])
               for sc, d in allsc.items()},
              open(osp.join(args.out_dir, f'gen_summary_{args.tag or "multi"}.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"\n[out] {osp.join(args.out_dir, f'gen_summary_{args.tag or 'multi'}.json')}")
    return


def pick_main(tg):
    """(target_text, label) 목록에서 주 피사체 하나. SUBJ_PRIO 우선, 없으면 최빈(=첫 항목)."""
    for want in SUBJ_PRIO:
        for t, l in tg:
            if l == want:
                return t, l
    return tg[0]


def interp_track(rows_pts, T, fps):
    """track anchor 들을 프레임 격자로 선형 보간. 커버 구간 밖은 NaN.

    fps 마다 timestamp -> 프레임 매핑이 달라 겹치는 앵커가 거의 없다. 그래서 각 track 을
    프레임 축으로 보간한 뒤 공통 구간에서 비교한다 (GT 없이 재는 **자기일관성** 지표)."""
    if len(rows_pts) < 2:
        return np.full((T, 2), np.nan)
    seq = sorted(rows_pts)
    f = np.array([min(max(int(round(t * fps)), 0), T - 1) for t, _x, _y in seq], float)
    x = np.array([p[1] for p in seq]); y = np.array([p[2] for p in seq])
    g = np.arange(T, dtype=float)
    out = np.stack([np.interp(g, f, x), np.interp(g, f, y)], 1)
    out[(g < f[0]) | (g > f[-1])] = np.nan          # 외삽 금지
    return out


def analyze(scene, rows, W, H):
    """같은/다른 클래스 타겟쌍의 t=0 좌표 거리. GT 없이 재는 판별력 지표."""
    ok = [r for r in rows if r['n']]
    print(f'\n--- {scene} 판별 분석 ---')
    print(f'{"label":11s} {"n":>3s} {"t=0 (x,y)":>13s} {"step":>6s}  prompt')
    for r in ok:
        p0 = r['p0']
        print(f'{r["label"][:11]:11s} {r["n"]:3d} {f"({p0[0]:.0f},{p0[1]:.0f})":>13s} '
              f'{r["step"]:6.1f}  {r["prompt"][:44]}')
    wi, be = [], []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            d = float(np.hypot(ok[i]['p0'][0] - ok[j]['p0'][0],
                               ok[i]['p0'][1] - ok[j]['p0'][1]))
            (wi if ok[i]['label'] == ok[j]['label'] else be).append(d)
    w = float(np.mean(wi)) if wi else float('nan')
    b = float(np.mean(be)) if be else float('nan')
    an = dict(within=w, between=b, ratio=(b / w if wi and w > 0 else float('nan')),
              npts=float(np.mean([r['n'] for r in ok])) if ok else 0.0,
              step=float(np.mean([r['step'] for r in ok])) if ok else 0.0,
              n_ok=len(ok), n_total=len(rows))
    print(f'  within {w:.1f}px ({len(wi)} 쌍)   between {b:.1f}px ({len(be)} 쌍)   '
          f'ratio {an["ratio"]:.2f}   좌표 뽑힌 타겟 {len(ok)}/{len(rows)}')
    return an


def fps_sweep(args, proc, model, device, dtype):
    """fps 를 바꿔가며 같은 타겟을 track 하고 점 개수·평활도·자기일관성을 비교한다.

    시각 텐서는 fps 와 무관하게 동일하다 (`video_grids [[49,9,9]]`). 바뀌는 것은 프롬프트의
    timestamp 문구뿐이고, 출력은 항상 0.5초 격자이므로 점 개수 ≈ (T/fps)/0.5 = 2T/fps 다.
    **GT 가 없으므로 정확도는 못 잰다** — 재는 것은 (1) 점 개수, (2) 정지 궤적으로 붕괴했는지,
    (3) 시간 평활도, (4) 기준 fps 와의 프레임별 일치도(자기일관성) 다."""
    fpss = [float(x) for x in args.fps_sweep.split(',')]
    res = {}
    for scene in args.scene:
        base, tg = scene_targets(args.root, scene, 12, args.preset)
        tt, lab = pick_main(tg)
        phrase = tt if args.trim_level == 'full' else trim(tt, args.trim_level)
        s0, e0 = base[1]['frame_idx']
        st, nf = int(s0), int(e0) - int(s0)
        video = C.load_frames(args.root, scene, st, st + nf)
        T, H, W = video.shape[:3]
        prompt = f'{args.verb} {phrase}'
        print(f'\n{"#" * 84}\n### {scene}   main=[{lab}] {prompt!r}\n{"#" * 84}', flush=True)
        rows = []
        for fp in fpss:
            args.fps = fp
            args.tag = f'fpssweep{fp:g}'
            r = one(args, proc, model, device, dtype, video, T, H, W, prompt, None, scene)
            pts = [(p[0], p[2], p[3]) for p in r['pts']]
            tr = interp_track(pts, T, fp)
            span = int(np.sum(~np.isnan(tr[:, 0])))
            disp = (float(np.nanmax(np.linalg.norm(tr - np.nanmean(tr, 0), axis=1)))
                    if span else float('nan'))
            rows.append(dict(fps=fp, n=r['n'], span=span, step=r['step'], disp=disp, tr=tr))
        ref = next((x for x in rows if x['fps'] == args.fps_ref), rows[0])
        print(f'\n--- {scene}  fps 별 비교  (기준 fps {ref["fps"]:g}) ---')
        print(f'{"fps":>6s} {"모델이 본 길이":>12s} {"점":>4s} {"보간 커버":>9s} '
              f'{"step":>7s} {"최대편차":>8s} {"기준과 평균거리":>13s}')
        for x in rows:
            m = np.linalg.norm(x['tr'] - ref['tr'], axis=1)
            agree = float(np.nanmean(m)) if np.any(~np.isnan(m)) else float('nan')
            x['agree'] = agree
            flag = '  <- 정지 궤적' if x['disp'] < 3.0 else ''
            print(f'{x["fps"]:6g} {T / x["fps"]:11.2f}s {x["n"]:4d} {x["span"]:6d}/{T} '
                  f'{x["step"]:7.1f} {x["disp"]:8.1f} {agree:13.1f}{flag}')
        res[scene] = dict(main=phrase, label=lab, T=T, W=W, H=H,
                          rows=[{k: v for k, v in x.items() if k != 'tr'} for x in rows])
    print(f'\n점 개수 ≈ 2T/fps 이고 좌표 격자는 항상 0.5초다. "최대편차" 가 0 에 가까우면')
    print('궤적이 한 점으로 붕괴한 것 (off-distribution). "기준과 평균거리" 는 GT 가 아니라')
    print('기준 fps 와의 자기일관성이다 — 정확도가 아니라 안정성 지표로만 읽을 것.')
    jp = osp.join(args.out_dir, 'fps_sweep.json')
    json.dump(res, open(jp, 'w'), ensure_ascii=False, indent=1)
    print(f'\n[out] {jp}')


def one(args, proc, model, device, dtype, video, T, H, W, prompt, attn, scene=None):
    msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                         {'type': 'video'}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    meta = [{'fps': args.fps, 'total_num_frames': T, 'duration': T / args.fps,
             'frames_indices': list(range(T)), 'width': W, 'height': H}]
    batch = proc(text=[text], videos=[video], return_tensors='pt',
                 do_sample_frames=False, video_metadata=meta)
    batch = {k: (v.to(device, dtype) if k == 'pixel_values_videos' else v.to(device))
             for k, v in batch.items()}

    print(f'[in] {scene}  frames {T} {H}x{W}  fps {args.fps}  '
          f'prompt {prompt!r}  tokens {batch["input_ids"].shape[1]}', flush=True)
    with torch.inference_mode():
        gen = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                             do_sample=False)
    out = proc.tokenizer.decode(gen[0, batch['input_ids'].shape[1]:],
                                skip_special_tokens=True)
    print(f'\n===== 생성 원문 ({len(out)} chars) =====\n{out}\n{"=" * 46}\n')

    pts = extract_video_points(out, W, H, extract_ids=True)
    if not pts:
        print('[warn] 좌표를 못 뽑았다 — 위 원문의 형식을 확인할 것 '
              '(태그가 <points>/<tracks> 가 아니거나 좌표가 없다)')
    else:
        fids = sorted({p[0] for p in pts})
        ids = sorted({p[1] for p in pts})
        print(f'[pts] {len(pts)} 점 / frame_id {len(fids)} 종 {fids[:12]}'
              f'{" ..." if len(fids) > 12 else ""}')
        print(f'[pts] track id {ids}   프레임 커버리지 {len(fids)}/{T} = '
              f'{len(fids) / T * 100:.0f}%')
        for p in pts[:8]:
            print(f'      f{p[0]:g}  id{p[1]}  ({p[2]:.0f}, {p[3]:.0f})')

    # frame_id -> 프레임 인덱스. 초 단위면 fps 를 곱해야 한다 (max 값으로 판정)
    per = {}
    if pts:
        mx = max(p[0] for p in pts)
        as_sec = mx <= T / args.fps * 1.5 and mx < T * 0.8
        for fid, idx, x, y in pts:
            k = int(round(fid * args.fps)) if as_sec else int(round(fid))
            per.setdefault(min(max(k, 0), T - 1), []).append((idx, x, y))
        print(f'[map] frame_id 를 {"초" if as_sec else "프레임 인덱스"} 로 해석 '
              f'(max {mx:g}, T {T}) -> 점이 붙은 프레임 {len(per)}개')

    tag = ((args.tag + '_' if args.tag else '') + (scene or 'x').split('/')[-1] + '_'
           + re.sub(r'[^a-z0-9]+', '-', prompt.lower())[:38])
    mp4 = osp.join(args.out_dir, f'gen_{tag}.mp4')
    wr = iio.get_writer(mp4, fps=args.out_fps, codec='libx264', quality=8,
                        macro_block_size=1)
    hit = 0
    for i in range(T):
        f = video[i].copy()
        for idx, x, y in per.get(i, []):
            draw_cross(f, x, y, [0, 255, 0])                  # 초록 = 생성된 point/track
            hit += 1
        if attn is not None:
            r, c = np.unravel_index(int(np.argmax(attn[i])), attn[i].shape)
            draw_cross(f, (c + .5) * W / attn[i].shape[1],
                       (r + .5) * H / attn[i].shape[0], [255, 0, 255], r=9, t=2)
        wr.append_data(f)
    wr.close()
    json.dump(dict(scene=scene, prompt=prompt, fps=args.fps, frames=T,
                   raw=out, points=[list(p) for p in pts],
                   frames_with_points=sorted(per)),
              open(mp4[:-4] + '.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n[save] {mp4}   (점이 그려진 프레임 {len(per)}/{T}, 총 {hit} 점)')
    lab = (re.search(r'>([^<]*)</(?:points|tracks)>', out) or [None, ''])[1]
    xs = [p[2] for p in pts]; ys = [p[3] for p in pts]
    seq = sorted([(p[0], p[2], p[3]) for p in pts])
    step = float(np.mean([np.hypot(seq[k + 1][1] - seq[k][1], seq[k + 1][2] - seq[k][2])
                          for k in range(len(seq) - 1)])) if len(seq) > 1 else 0.0
    return dict(prompt=prompt, n=len(pts), label=lab.strip(), raw=out,
                pts=[list(p) for p in pts],
                p0=[seq[0][1], seq[0][2]] if seq else [float('nan')] * 2,
                step=step, mx=float(np.mean(xs)) if xs else float('nan'),
                my=float(np.mean(ys)) if ys else float('nan'), mp4=mp4)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--prompt', nargs='+', default=['Track the larger pale camel'])
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scene', nargs='+', default=['vista4d/camel'])
    p.add_argument('--auto_targets', type=int, default=0,
                   help='>0 이면 코퍼스에서 타겟 N종을 뽑아 프롬프트를 만든다')
    p.add_argument('--verb', default='Track', help='auto_targets/main 용 지시 동사')
    p.add_argument('--main', action='store_true',
                   help='씬의 주 피사체 하나만 (SUBJ_PRIO 우선)')
    p.add_argument('--fps_sweep', default='',
                   help='쉼표 구분 fps 목록. 주면 fps 비교 모드')
    p.add_argument('--fps_ref', type=float, default=10.0)
    p.add_argument('--trim_level', default='noun',
                   choices=('full', 'action', 'noun'))
    p.add_argument('--preset', default='dolly_in_look_at')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--num_frames', type=int, default=49)
    p.add_argument('--fps', type=float, default=10.0)
    p.add_argument('--max_new_tokens', type=int, default=2048)
    p.add_argument('--attn_npy', default='', help='molmo2_attn_video.py 가 남긴 (T,9,9) npy')
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--tag', default='')
    p.add_argument('--out_fps', type=int, default=10)
    p.add_argument('--gpu', default='2')
    main(p.parse_args())
