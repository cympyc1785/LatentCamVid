"""PRDC 의 precision/density 가 낮은 게 **GT 다양성 부족 때문인지**를 가른다.

물음
----
"SD 는 GT 카메라 움직임이 몇 종류 안 되니까 kNN 반경 기반인 precision/density 가 낮게
나오는 것 아니냐" — 그럴듯하지만 방향을 따져야 한다. PRDC precision 은

    precision_i = [ min_j d(GT_j, pred_i) < r_j ] 를 j 에 대해 any,
    r_j = GT_j 의 (k+1)-th NN 거리 (chunk 안에서)

이라 **거리와 반경이 같이 스케일**한다. 그래서 "GT 가 좁은 영역에 모여 있다"만으로는
precision 이 안 떨어진다 (스케일 불변). 실제로 떨어뜨리는 건 GT 가 **중복에 가깝게 뭉쳐**
r_j 가 GT 전체 퍼짐에 비해 유독 작아지는 경우다. SD 는 01/02 가 scene 을 가로질러 거의
같은 결정론적 템플릿이라 (scripts/data/sd_preset_taxonomy.py) 이 조건에 딱 걸린다.

그래서 스케일 불변량으로만 비교한다. chunk 안에서
    spread     = GT 쌍거리 중앙값                      (그 chunk 의 길이 단위)
    tightness  = median(r_j) / spread                  작을수록 GT 가 뭉쳐 있다
    dup_frac   = 최근접 GT 거리 < dup_thresh*spread 인 GT 비율   (거의 중복인 GT)
    err_pair   = median ||pred_i − GT_i|| / spread     예측 오차 (같은 단위)
    err_min    = median min_j ||pred_i − GT_j|| / spread
정답이 "다양성 탓" 이면 SD 는 tightness 가 작고 err_pair 는 DL3DV 와 비슷해야 한다.
"예측이 그냥 나쁜 것" 이면 err_pair 가 크게 나온다. 둘 다일 수도 있고, 그때는 어느 쪽이
지배적인지 반사실 실험으로 가른다:
    swap   : SD pred 를 **다른 SD GT** 로 갈아끼워도 (= 완벽한 GT 를 예측한 셈)
             precision 이 몇이나 나오는지. 상한이 낮으면 지표 쪽 문제다.
    selfgt : pred = GT 그대로 넣었을 때의 precision (이론상 상한).

chunk 크기 주의
---------------
`compute(num_splits=5)` 는 전체 n 을 5 등분하므로 chunk 크기가 n 에 딸려 간다 (n=160 -> 32,
n=640 -> 128). chunk 가 커지면 반경도 예측 난이도도 같이 바뀌어 코퍼스 비교가 오염된다
(memory: prdc-not-comparable-across-corpora). 여기서는 **chunk 크기를 --chunk 로 고정**해
양쪽을 같은 조건에 놓는다.

usage
-----
  python scripts/eval/prdc_diversity_diagnosis.py \
      --run sd=results/compare/sd_whuman_textonly_n160 \
      --run dl3dv=results/20260730_223514_dl3dv_textonly_savedscale_bs32 \
      --chunk 32 [--focus <data_name>]
"""
import argparse
import json
import os.path as osp

import numpy as np
import torch

K = 3                                   # prdc.py manifold_k


def load(run):
    b = np.load(osp.join(run, 'preds.npy'), allow_pickle=True).item()
    ref = torch.stack(list(b['m_ref_latents'])).float()
    pred = torch.stack(list(b['m_pred_latents'])).float()
    txt = torch.stack(list(b['t_latents'])).float()
    names = [osp.basename(f.strip().rstrip('.')).replace('_transforms_ref', '')
             for f in b['ref_filenames']]
    return ref, pred, txt, names


def radii(X, k=K):
    d = torch.cdist(X.unsqueeze(0), X.unsqueeze(0), 2).squeeze(0)
    return torch.topk(d, k + 1, largest=False, dim=-1).values.max(axis=-1).values


def chunks(n, size):
    return [slice(i, min(i + size, n)) for i in range(0, n - size + 1, size)]


def diagnose(ref, pred, chunk, dup_thresh, rng):
    """-> chunk 별 스케일 불변 지표 + 반사실 precision."""
    out = []
    for sl in chunks(len(ref), chunk):
        r, f = ref[sl], pred[sl]
        pw = torch.cdist(r.unsqueeze(0), r.unsqueeze(0), 2).squeeze(0)
        iu = torch.triu_indices(len(r), len(r), offset=1)
        spread = float(pw[iu[0], iu[1]].median())
        rr = radii(r)
        nn = pw + torch.eye(len(r)) * 1e9                     # 자기 자신 제외
        nearest = nn.min(dim=1).values
        d = torch.cdist(r.unsqueeze(0), f.unsqueeze(0), 2).squeeze(0)   # (n_ref, n_pred)
        prec = float((d < rr.unsqueeze(1)).any(axis=0).to(float).mean())

        # 반사실 1: pred 자리에 GT 를 그대로 -> 이론상 상한
        d_self = pw
        prec_self = float((d_self < rr.unsqueeze(1)).any(axis=0).to(float).mean())
        # 반사실 2: pred 를 같은 chunk 의 **다른** GT 로 치환 (완벽 예측이되 짝만 틀린 것)
        perm = torch.from_numpy(rng.permutation(len(r)))
        fix = perm == torch.arange(len(r))
        if fix.any():                                          # 고정점은 한 칸 밀어 없앤다
            perm[fix] = perm[fix.nonzero().flatten() - 1]
        d_swap = torch.cdist(r.unsqueeze(0), r[perm].unsqueeze(0), 2).squeeze(0)
        prec_swap = float((d_swap < rr.unsqueeze(1)).any(axis=0).to(float).mean())

        out.append({
            'spread': spread,
            'tightness': float(rr.median()) / spread,
            'dup_frac': float((nearest < dup_thresh * spread).to(float).mean()),
            'err_pair': float(torch.diagonal(d).median()) / spread,
            'err_min': float(d.min(axis=0).values.median()) / spread,
            'precision': prec, 'prec_selfgt': prec_self, 'prec_swap': prec_swap,
        })
    return out


def focus(ref, pred, txt, names, target, chunk):
    """--focus 로 지목한 sequence 하나를 뜯어본다."""
    if target not in names:
        cand = [n for n in names if target in n]
        if not cand:
            print(f"  [focus] {target} 없음")
            return
        target = cand[0]
    gi = names.index(target)
    sl = chunks(len(ref), chunk)[gi // chunk]
    li = gi - sl.start
    r, f = ref[sl], pred[sl]
    rr = radii(r)
    d = torch.cdist(r.unsqueeze(0), f.unsqueeze(0), 2).squeeze(0)
    pw = torch.cdist(r.unsqueeze(0), r.unsqueeze(0), 2).squeeze(0)
    iu = torch.triu_indices(len(r), len(r), offset=1)
    spread = float(pw[iu[0], iu[1]].median())
    col = d[:, li]                                     # 이 pred 와 chunk 내 모든 GT 의 거리
    j = int(col.argmin())
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a[None], b[None]))
    print(f"  [focus] {target}")
    print(f"    chunk {gi // chunk} 내 위치 {li}, GT 쌍거리 중앙값(spread) {spread:.4f}")
    print(f"    pred <-> 짝 GT   거리 {float(col[li]):.4f} = {float(col[li]) / spread:.2f}·spread"
          f"   (짝 GT 반경 r {float(rr[li]):.4f} = {float(rr[li]) / spread:.2f}·spread)")
    print(f"    pred <-> 최근접 GT 거리 {float(col[j]):.4f} (GT #{j}), 그 GT 반경 "
          f"{float(rr[j]):.4f} -> 안에 들어가려면 거리 < 반경 이어야 한다")
    print(f"    통과한 GT 개수 {int((col < rr).sum())} / {len(r)}  (density·k)")
    print(f"    cos(pred, 짝 GT) {cos(pred[gi], ref[gi]) * 100:.2f}   "
          f"cos(pred, text) {cos(pred[gi], txt[gi]) * 100:.2f}   "
          f"cos(짝 GT, text) {cos(ref[gi], txt[gi]) * 100:.2f}")
    o = torch.arange(len(r)) != li
    print(f"    짝 GT <-> 다른 GT 최근접 거리 {float(pw[li][o].min()):.4f} "
          f"= {float(pw[li][o].min()) / spread:.2f}·spread")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='label=path 형식, 여러 번')
    ap.add_argument('--chunk', type=int, default=32, help='코퍼스 간 비교를 위해 고정할 chunk 크기')
    ap.add_argument('--dup-thresh', type=float, default=0.05,
                    help='최근접 GT 거리가 spread 의 이 배 미만이면 "거의 중복" 으로 센다')
    ap.add_argument('--focus', default=None, help='자세히 볼 data_name (부분 문자열 가능)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    hdr = (f"{'run':10s} {'n':>5} {'chunks':>6} | {'tightness':>9} {'dup%':>6} | "
           f"{'err_pair':>8} {'err_min':>8} | {'prec':>6} {'swap':>6} {'selfGT':>6}")
    print(hdr)
    print('-' * len(hdr))
    for spec in args.run:
        label, path = spec.split('=', 1)
        ref, pred, txt, names = load(path)
        rows = diagnose(ref, pred, args.chunk, args.dup_thresh, rng)
        g = lambda k: np.mean([r[k] for r in rows])
        print(f"{label:10s} {len(ref):5d} {len(rows):6d} | {g('tightness'):9.4f} "
              f"{g('dup_frac') * 100:5.1f}% | {g('err_pair'):8.3f} {g('err_min'):8.3f} | "
              f"{g('precision'):6.4f} {g('prec_swap'):6.4f} {g('prec_selfgt'):6.4f}")
        if args.focus:
            focus(ref, pred, txt, names, args.focus, args.chunk)
    print("\ntightness = median(GT k-NN 반경)/spread : 작을수록 GT 가 뭉쳐 반경이 좁다")
    print("err_pair  = median||pred−짝GT||/spread  : 같은 단위의 예측 오차")
    print("swap      = pred 를 다른 GT 로 치환했을 때의 precision (완벽 예측의 현실적 상한)")
    print("selfGT    = pred=GT 일 때의 precision (이론상 상한, 1 이 아니면 지표가 이미 빡빡)")


if __name__ == '__main__':
    main()
