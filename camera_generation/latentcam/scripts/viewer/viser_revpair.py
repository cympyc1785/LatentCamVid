"""viser viewer for SD 역재생 결합 recon (`video_generation/results/20260818_sd_revpair/<pair>/`).

같은 scene 의 두 clip 을 `rev(A)[80..1] + B[0..80]` = 161장으로 이어 붙여 **한 번에** DA3 를
돌린 결과(`recon.npz`)를 본다. 전역 index 를 anchor 에서 자르면 그대로 context/target 이다:

    [0, anchor)      = rev(A)  -> **context**  (초록)
    [anchor, N)      = B       -> **target**   (파랑 point / 청록 camera)
    anchor (=80)     = A[0] == B[0], 두 clip 공통 앵커 카메라

색과 frustum 규약은 `viser_val_cameras.py` 에서 **그대로 가져온다** (`add_frustums`, `_GL2CV`) —
저기서 이미 context=초록 / target=파랑·청록으로 칠하고 있어서 두 뷰어가 같은 뜻을 갖는다.

좌표: `recon.npz` 의 `extrinsics` 는 **OpenCV w2c** 다 (`halfsplit_recon_da3.py:19-30`, FIX-8,
메모 `da3-extrinsics-are-w2c`). frustum 은 `c2w_gl = inv(w2c) @ _GL2CV` 로 넘긴다 —
`viser_val_cameras.py` 의 SD 분기와 같은 식이다.

world 는 **DA3 joint 게이지 그대로**다 (GT 미터로 안 올린다). 이 뷰어의 요점이 "따로 돌린 두
clip 을 sim3 로 맞출 필요 없이 통짜 recon 한 번으로 같은 좌표계에 들어왔나" 이므로, 여기서 또
sim3 를 태우면 확인하려는 것 자체가 가려진다. GT 대조는 `<pair>/verify.json` 에 있다.

Run:
  python scripts/viewer/viser_revpair.py --pair Cabin_Lake --port 8082
  python scripts/viewer/viser_revpair.py --pair Cabin_Lake --rgb --pc-frames 60
"""

import os, os.path as osp, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from viser_val_cameras import add_frustums, _GL2CV      # noqa: E402  (규약 단일 출처)
import viser                                            # noqa: E402

REVPAIR_ROOT = '/data1/cympyc1785/LatentCamVid/video_generation/results/20260818_sd_revpair'

# viser_val_cameras.py 의 _CTX / _TGT 와 같은 값. 저기선 튜플 안에 묻혀 있어 import 가 안 된다.
CTX_PC, CTX_CAM = (40, 190, 90), (40, 190, 90)          # context = 초록
TGT_PC, TGT_CAM = (40, 90, 230), (0, 200, 200)          # target = 파랑 point / 청록 camera


def resolve_pair(s):
    """전체 경로 / 디렉토리명 / 부분 문자열(예: 'Cabin_Lake') 아무거나 받는다."""
    if osp.isdir(s) and osp.exists(osp.join(s, 'recon.npz')):
        return s
    cand = [d for d in sorted(glob.glob(osp.join(REVPAIR_ROOT, '*')))
            if osp.isdir(d) and s in osp.basename(d)]
    if len(cand) == 1:
        return cand[0]
    raise SystemExit(f"--pair {s!r} 로 {len(cand)} 개가 잡혔다. 후보: "
                     f"{[osp.basename(d) for d in sorted(glob.glob(osp.join(REVPAIR_ROOT, '*'))) if osp.isdir(d)]}")


def unproject(A, idxs, stride, conf_thr, rgb):
    """프레임 idxs 를 world 로 unproject. -> (P (M,3) float32, C (M,3) uint8)

    `sd_pointcloud` 와 **같은 식**이되 (a) sim3 를 안 태우고 (b) 색을 mp4 대신 npz 의
    `images` 에서 가져온다 (통짜 recon 은 두 clip 이 섞여 있어 mp4 하나로는 못 칠한다).
    `A` 는 npz 를 미리 푼 dict — NpzFile 은 접근할 때마다 다시 압축을 푼다.
    """
    dep, cf, K, w2c = A['depth'], A['conf'], A['intrinsics'], A['extrinsics']
    h, w = dep.shape[1], dep.shape[2]
    vs, us = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing='ij')
    P, C = [], []
    for i in idxs:
        d = np.asarray(dep[i], np.float32)[::stride, ::stride]
        c = np.asarray(cf[i], np.float32)[::stride, ::stride]
        m = (c >= conf_thr) & np.isfinite(d) & (d > 0)
        if not m.any():
            continue
        pix = np.stack([us[m] + 0.5, vs[m] + 0.5, np.ones(int(m.sum()))], 0)      # (3,M)
        cam = (np.linalg.inv(K[i]) @ pix) * d[m][None]                           # (3,M)
        R, t = w2c[i, :3, :3], w2c[i, :3, 3]
        P.append((R.T @ (cam - t[:, None])).T)                                   # world
        C.append(A['images'][i][::stride, ::stride][m] if rgb
                 else np.zeros((int(m.sum()), 3), np.uint8))
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(P).astype(np.float32), np.concatenate(C).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True,
                    help="revpair 디렉토리. 전체 경로 또는 부분 문자열 (예: 'Cabin_Lake')")
    ap.add_argument('--port', type=int, default=8082)
    ap.add_argument('--split', type=int, default=None,
                    help='context/target 경계 전역 index. 기본 = pair.json 의 anchor_index')
    ap.add_argument('--pc-frames', type=int, default=40,
                    help='**각 쪽마다** unproject 할 프레임 수 (구간에 균등 분포)')
    ap.add_argument('--pc-stride', type=int, default=3, help='depth 격자 픽셀 stride')
    ap.add_argument('--pc-conf-pct', type=float, default=40.0,
                    help='conf 백분위수 컷. 161장 **전체**에서 한 번 재서 두 쪽에 같은 임계를 쓴다')
    ap.add_argument('--pc-max-points', type=int, default=400_000, help='쪽마다')
    ap.add_argument('--pc-size', type=float, default=0.0, help='0 = extent 로 자동')
    ap.add_argument('--rgb', action='store_true',
                    help='단색 tint 대신 recon.npz 의 images 로 칠한다 (기본은 tint)')
    ap.add_argument('--cam-stride', type=int, default=2, help='frustum 을 몇 프레임마다')
    ap.add_argument('--up', default='+z')
    args = ap.parse_args()

    pdir = resolve_pair(args.pair)
    meta = json.load(open(osp.join(pdir, 'pair.json')))
    with np.load(osp.join(pdir, 'recon.npz')) as _z:
        A = {k: _z[k] for k in ('extrinsics', 'intrinsics', 'depth', 'conf', 'images')}
    N = len(A['extrinsics'])
    a = int(args.split if args.split is not None else meta['anchor_index'])
    print(f"pair {osp.basename(pdir)}")
    print(f"  {meta['note']}")
    print(f"  clip_a(context, 역재생) {meta['clip_a']}   clip_b(target) {meta['clip_b']}")
    print(f"  n_total {N}  anchor_index {a}  -> context [0,{a}) {a}장 / target [{a},{N}) {N - a}장")
    print(f"  dir_angle {meta['dir_angle_deg']:.2f} deg   min_path {meta['min_path_len_m']:.3f} m")

    _cs = A['conf'][::4, ::args.pc_stride, ::args.pc_stride]
    conf_thr = float(np.percentile(_cs, args.pc_conf_pct))
    c2w_gl = np.linalg.inv(A['extrinsics']) @ _GL2CV
    K, h, w = A['intrinsics'], A['depth'].shape[1], A['depth'].shape[2]
    ext = float(np.linalg.norm(c2w_gl[:, :3, 3] - c2w_gl[:, :3, 3].mean(0), axis=1).max())
    psize = float(args.pc_size or ext * 0.012)
    print(f"  conf 임계 {conf_thr:.4f} (p{args.pc_conf_pct:g}, 전 프레임 공통)   "
          f"카메라 extent {ext:.4f}   point size {psize:.5f}")
    # DA3 conf 는 1.0 에 바닥이 있다. 이 recon 은 픽셀의 ~48% 가 정확히 1.0 이라 p40 컷이
    # 임계 1.0 -> `c >= 1.0` 이 전부 통과 = **필터가 아무것도 안 버린다**. 조용히 넘기면
    # "p40 로 걸렀다"고 오해하게 되므로 명시한다. 실제로 거르려면 --pc-conf-pct 60 이상.
    if conf_thr <= float(_cs.min()) + 1e-9:
        print(f"  !! conf 컷이 무효다: p{args.pc_conf_pct:g} = {conf_thr:.4f} = conf 최소값 "
              f"(바닥 비율 {float((_cs <= conf_thr + 1e-9).mean()) * 100:.1f}%). "
              f"버려지는 점 0개. 거르려면 --pc-conf-pct 를 올릴 것.")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction(args.up)
    groups = {}
    for tag, lo, hi, pc_col, cam_col in [('context', 0, a, CTX_PC, CTX_CAM),
                                         ('target', a, N, TGT_PC, TGT_CAM)]:
        idxs = np.unique(np.linspace(lo, hi - 1, min(args.pc_frames, hi - lo)).round().astype(int))
        P, C = unproject(A, idxs, args.pc_stride, conf_thr, args.rgb)
        if len(P) > args.pc_max_points:
            sel = np.random.default_rng(0).choice(len(P), args.pc_max_points, replace=False)
            P, C = P[sel], C[sel]
        if not args.rgb:
            C = np.tile(np.array(pc_col, np.uint8), (len(P), 1))
        groups[f'pc_{tag}'] = server.scene.add_point_cloud(
            f'pc_{tag}', points=P, colors=C, point_size=psize)
        groups[f'cam_{tag}'] = server.scene.add_frame(f'cam_{tag}', show_axes=False)
        add_frustums(server, f'cam_{tag}', c2w_gl[lo:hi], K[lo, 0, 0], K[lo, 1, 1], w, h,
                     cam_col, ext * 0.05, downsample=max(1, args.cam_stride))
        print(f"  {tag:8s} frames {len(idxs):3d}  points {len(P):7d}")

    # 공통 앵커 한 장만 흰색으로 따로 — 두 쪽이 여기서 만나야 한다.
    groups['anchor'] = server.scene.add_frame('anchor', show_axes=False)
    add_frustums(server, 'anchor', c2w_gl[a:a + 1], K[a, 0, 0], K[a, 1, 1], w, h,
                 (255, 255, 255), ext * 0.09)

    def _toggle(h):
        def _f(ev):
            h.visible = ev.target.value
        return _f
    for name, hnd in groups.items():
        server.gui.add_checkbox(name, initial_value=True).on_update(_toggle(hnd))

    print(f"\nviser: http://0.0.0.0:{args.port}   (Ctrl+C 로 종료)")
    import time
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
