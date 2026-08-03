"""한 segment 를 3DGS scene.ply 위에 올려놓고, arm 별 예측 frustum 을 겹쳐 보는 viser 뷰어.

`viser_val_cameras.py` 는 run 하나를 훑는 브라우저라 arm 비교가 안 되고 splat 도 못 띄운다.
여기서는 segment 를 하나로 고정하고

  - scene.ply (DL3DV 가 주는 3DGS) 를 add_gaussian_splats 로 배경에 깔고
  - GT / arm 별 pred / 각 arm 의 context 카메라를 서로 다른 색 frustum 으로 올린다.

좌표계: dump 된 transform_matrix 와 geo_ctx/*.json 의 c2w 는 둘 다 DL3DV
transforms.json 과 같은 OpenGL c2w 다. 그런데 scene.ply / cameras.json 은
nerfstudio 가 기록한 world 재정렬 `applied_transform` 이 **적용된** frame 이고
transform_matrix 는 적용 **전** frame 이라, 그냥 겹치면 어긋난다
(tools/gaussian-splatting-lightning/custom_utils/custom_panel.py
 `get_cameras_from_transforms` 와 같은 처리).  그래서

    c2w_ply = applied_transform4 @ c2w_gl

로 world 쪽에 좌측 곱한 뒤 frustum 으로 넘긴다. OpenGL->OpenCV 축 뒤집기
(c2w @ diag(1,-1,-1,1)) 는 `viser_val_cameras.add_frustums` 가 이미 한다.
재정렬된 frame 은 COLMAP world 라 up 이 -y 다.

  python scripts/viewer/viser_arms_gs.py \
      --name 1K_c7576be..._1 \
      --arm 'out-of-seg ctx:eval_my/divers/swap_base:245,130,30' \
      --arm 'in-seg ctx K=5:eval_my/divers/leak5_seed0:150,80,200' \
      --ctx 'ctx out-of-seg:eval_my/divers/swap_base:30,170,70' \
      --ctx 'ctx in-seg (K=5):eval_my/divers/leak5_seed0:230,40,40:5' \
      --port 8080
"""
import os as _os, sys as _sys, glob as _glob
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path:
        _sys.path.append(_d)

import argparse
import json
import os.path as osp
import time

import numpy as np
import viser
import viser.transforms as vtf

from viewer.viser_val_cameras import DL3DV_ROOT, add_frustums, load_transforms, scene_chunk

# gaussian-splatting-lightning 의 spline frustum. viser 기본 add_camera_frustum 은 선 두께가
# 화면 픽셀 단위라 잘 안 먹는데, 저쪽은 frustum 을 catmull-rom spline 8개로 그려서
# line_width 가 실제로 먹는다. custom_panel.py 의 visualize_camera_frustum 이 쓰는 그 함수다.
GSPL_ROOT = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
                     'tools', 'gaussian-splatting-lightning')
if GSPL_ROOT not in _sys.path:
    _sys.path.append(GSPL_ROOT)
try:
    from custom_utils.render_frustum import add_frustum_spline
except ImportError:                                   # repo 가 없으면 viser 스타일만 쓴다
    add_frustum_spline = None

_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])

# 3DGS ply 표준 레이아웃: xyz, normals, f_dc(3), f_rest(45), opacity, scale(3), rot(4)
_PLY_D = 62
_I_OPACITY, _I_SCALE, _I_ROT = 54, slice(55, 58), slice(58, 62)
_SH_C0 = 0.28209479177387814


def _ply_header(path):
    """-> (n_vertex, data_offset, n_float_props)"""
    n, props = 0, 0
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f'bad ply: {path}')
            if line.startswith(b'element vertex'):
                n = int(line.split()[-1])
            elif line.startswith(b'property float'):
                props += 1
            elif line.strip() == b'end_header':
                return n, f.tell(), props


def load_splats(path, max_splats=600_000, min_opacity=0.05, seed=0):
    """3DGS ply -> (centers, covariances, rgbs, opacities). SH 는 DC 항만 쓴다."""
    n, off, props = _ply_header(path)
    if props != _PLY_D:
        raise SystemExit(f'expected {_PLY_D} float props, got {props} in {path}')
    a = np.memmap(path, dtype='<f4', mode='r', offset=off, shape=(n, props))
    opa = 1.0 / (1.0 + np.exp(-np.asarray(a[:, _I_OPACITY], dtype=np.float32)))
    keep = np.flatnonzero(opa > min_opacity)
    if len(keep) > max_splats:                       # 균등 랜덤 (밝기 편향 방지)
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_splats, replace=False))
    b = np.asarray(a[keep], dtype=np.float32)
    centers = b[:, :3]
    rgbs = np.clip(0.5 + _SH_C0 * b[:, 6:9], 0.0, 1.0)
    opacities = opa[keep][:, None]
    s = np.exp(b[:, _I_SCALE])
    q = b[:, _I_ROT]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)
    M = R * s[:, None, :]                            # R @ diag(s)
    cov = M @ M.transpose(0, 2, 1)
    print(f'  splats {len(keep)}/{n} (opacity>{min_opacity})', flush=True)
    return centers, cov, rgbs, opacities.astype(np.float32)


def add_frustums_spline(server, name, c2w_gl, fx, fy, w, h, color, scale, thickness,
                        downsample=1):
    """`add_frustums` 와 같은 인자로 spline frustum 을 올린다.

    반환은 카메라 1대당 handle 리스트의 리스트 -> [[h,...], ...]. interval 슬라이더가
    카메라 단위로 visible 을 껐다 켜야 해서 평평하게 펴지 않는다.
    geometry 는 scale 을 굽지 않고 1.0 으로 만들고 node scale 로 조절한다 (슬라이더 실시간 반영).
    """
    fov = 2 * np.arctan2(h / 2, fy)
    aspect = w / h
    col = tuple(int(c) for c in color)
    out = []
    for i in range(0, len(c2w_gl), downsample):
        c2w_cv = c2w_gl[i] @ _GL2CV
        wxyz = vtf.SO3.from_matrix(c2w_cv[:3, :3]).wxyz
        hs = add_frustum_spline(scene=server.scene, name=f'{name}/cam_{i:04d}',
                                fov_y=float(fov), aspect=float(aspect), wxyz=wxyz,
                                position=c2w_cv[:3, 3].astype(np.float32), far=1.0,
                                scale=1.0, thickness=float(thickness), color=col)
        for hh in hs.values():
            hh.scale = float(scale)
        out.append(list(hs.values()))
    return out


def applied_transform4(scene_transforms):
    """transforms.json 의 world 재정렬을 4x4 로. 없으면 None."""
    if not osp.isfile(scene_transforms):
        return None
    at = json.load(open(scene_transforms)).get('applied_transform')
    if at is None:
        return None
    M = np.eye(4)
    M[:3, :4] = np.asarray(at, dtype=np.float64)
    return M


def _ctx_c2w(run_dir, name):
    p = osp.join(run_dir, 'geo_ctx', f'{name}.json')
    if not osp.exists(p):
        return None
    return np.asarray(json.load(open(p))['c2w'], dtype=np.float64)


def _spec(s, n_field):
    """'label:dir:R,G,B[:K]' 파싱."""
    parts = s.split(':')
    if len(parts) < 3:
        raise SystemExit(f'bad spec (need label:dir:R,G,B[:K]): {s}')
    label, run = parts[0], parts[1]
    col = tuple(int(v) for v in parts[2].split(','))
    k = int(parts[3]) if len(parts) > 3 else 0
    return (label, run, col, k) if n_field == 4 else (label, run, col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='segment data_name (예: 1K_<hash>_1)')
    ap.add_argument('--arm', action='append', default=[],
                    help="예측 arm. 'label:run_dir:R,G,B'")
    ap.add_argument('--ctx', action='append', default=[],
                    help="context 카메라. 'label:run_dir:R,G,B[:leak_K]' (K>0 이면 앞 K장을 따로 표시)")
    ap.add_argument('--gt-from', default=None, help='GT(ref) 를 읽을 run (기본: 첫 arm)')
    ap.add_argument('--dl3dv-root', default=DL3DV_ROOT)
    ap.add_argument('--ply', default=None, help='기본: <dl3dv-root>/<type>/<hash>/scene.ply')
    ap.add_argument('--no-gs', action='store_true')
    ap.add_argument('--gs-max', type=int, default=600_000)
    ap.add_argument('--gs-min-opacity', type=float, default=0.05)
    ap.add_argument('--grey-downsample', type=int, default=0,
                    help='>0 이면 scene 의 나머지 카메라를 회색으로 (N개마다 1개)')
    ap.add_argument('--frustum-scale', type=float, default=0.08,
                    help='target extent 대비 frustum 크기')
    ap.add_argument('--frustum-style', default='spline', choices=['spline', 'viser'],
                    help="'spline'=gspl add_frustum_spline (선 두께 조절됨), 'viser'=add_camera_frustum")
    ap.add_argument('--frustum-thickness', type=float, default=2.0)
    ap.add_argument('--frustum-interval', type=int, default=1,
                    help='궤적 카메라를 N개마다 하나만 표시 (context 는 항상 전부)')
    ap.add_argument('--no-applied-transform', action='store_true',
                    help='transforms.json 의 world 재정렬을 적용하지 않는다 (예전 동작; ply 와 어긋남)')
    ap.add_argument('--up', default=None, choices=['+x', '-x', '+y', '-y', '+z', '-z'],
                    help='viser up 축. 기본: 재정렬 적용 시 -y, 아니면 +y')
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()

    arms = [_spec(s, 3) for s in args.arm]
    ctxs = [_spec(s, 4) for s in args.ctx]
    if not arms:
        raise SystemExit('--arm 이 하나는 있어야 한다')
    gt_run = args.gt_from or arms[0][1]

    scene_dir = osp.join(args.dl3dv_root, scene_chunk(args.name))
    AT = None if args.no_applied_transform else applied_transform4(osp.join(scene_dir, 'transforms.json'))
    W = (lambda c: c) if AT is None else (lambda c: np.einsum('ij,njk->nik', AT, c))
    print(f'applied_transform: {"none" if AT is None else AT[:3, :4].tolist()}', flush=True)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction(args.up or ('+y' if AT is None else '-y'))

    ref_c2w, fx, fy, w, h = load_transforms(osp.join(gt_run, 'test', f'{args.name}_transforms_ref.json'))
    ref_c2w = W(ref_c2w)
    ext = np.linalg.norm(ref_c2w[:, :3, 3] - ref_c2w[0, :3, 3], axis=1).max()
    fscale = float(max(ext, 1e-3)) * args.frustum_scale
    print(f'{args.name}: GT frames {len(ref_c2w)}, reach {ext:.4f}, frustum {fscale:.4f}', flush=True)

    if args.frustum_style == 'spline' and add_frustum_spline is None:
        print(f'[warn] {GSPL_ROOT} 에서 custom_utils.render_frustum 를 못 읽어 viser 스타일로 폴백',
              flush=True)
        args.frustum_style = 'viser'

    # --- GUI 컨트롤 (그룹보다 먼저 만들어야 콜백에서 값을 읽을 수 있다) -------------------
    gui_fs = server.gui.add_slider('frustum scale (x reach)', min=0.005, max=0.5, step=0.005,
                                   initial_value=float(args.frustum_scale))
    gui_fs_txt = server.gui.add_text('frustum (world units)', initial_value=f'{fscale:.4f}',
                                     disabled=True)
    # thickness 는 slider 가 아니라 number 다. viser 1.0.30 의 SplineCatmullRomHandle /
    # CameraFrustumHandle 에는 line_width 가 설정 가능한 prop 으로 없어서 (scale/visible/
    # position/wxyz/positions 뿐) 두께를 바꾸려면 frustum 을 지우고 다시 그려야 하는데,
    # slider 로 하면 드래그 중 매 스텝마다 전체 재생성이 돌아 버린다. custom_panel.py 도
    # add_number 로 두고 build 시점에만 thickness 를 읽는다.
    gui_th = server.gui.add_number('frustum thickness (rebuild)',
                                   initial_value=float(args.frustum_thickness),
                                   min=0.5, max=20.0, step=0.5,
                                   disabled=args.frustum_style != 'spline')
    gui_iv = server.gui.add_slider('frustum interval (traj)', min=1, max=24, step=1,
                                   initial_value=max(1, int(args.frustum_interval)))

    groups = {}          # label -> checkbox
    specs = []           # 재생성용 그룹 스펙 (thickness 변경 시 지우고 다시 그린다)

    def _vis(sp, i):
        iv = int(gui_iv.value) if sp['iv_ok'] else 1
        n = len(sp['handles'])
        # interval 로 솎아도 마지막 카메라는 남긴다 (궤적 끝을 잃지 않게).
        on = bool(sp['cb'].value) and (iv <= 1 or i % iv == 0 or i == n - 1)
        for hh in sp['handles'][i]:
            hh.visible = on

    def _vis_all(sp):
        for i in range(len(sp['handles'])):
            _vis(sp, i)

    def add(label, per_cam, rel=None, allow_interval=False, build=None):
        """per_cam = 카메라 1대당 handle 리스트의 리스트."""
        if not per_cam:
            return
        cb = server.gui.add_checkbox(label, initial_value=True)
        sp = dict(cb=cb, handles=per_cam, rel=rel, iv_ok=allow_interval, build=build)

        @cb.on_update
        def _(_, _sp=sp):
            with server.atomic():
                _vis_all(_sp)
        groups[label] = cb
        if rel is not None:
            specs.append(sp)

    def add_cams(label, gname, c2w, ifx, ify, iw, ih, col, rel, allow_interval,
                 downsample=1):
        """스타일에 따라 frustum 을 올리고 그룹으로 등록. 반환 handle 은 카메라 단위."""
        def build(thickness):
            s = fscale * rel
            if args.frustum_style == 'spline':
                return add_frustums_spline(server, gname, c2w, ifx, ify, iw, ih, col, s,
                                           thickness, downsample=downsample)
            # viser 기본 frustum 은 선 두께 인자가 없다 (thickness 무시).
            return [[x] for x in add_frustums(server, gname, c2w, ifx, ify, iw, ih, col, s,
                                              downsample=downsample)]

        add(label, build(gui_th.value), rel=rel, allow_interval=allow_interval, build=build)

    if not args.no_gs:
        ply = args.ply or osp.join(scene_dir, 'scene.ply')
        print(f'loading {ply}', flush=True)
        c, cov, rgb, opa = load_splats(ply, args.gs_max, args.gs_min_opacity)
        gs = server.scene.add_gaussian_splats('/gs', centers=c, covariances=cov,
                                              rgbs=rgb, opacities=opa)
        add('3DGS scene', [[gs]])

    if args.grey_downsample > 0:
        sc = osp.join(scene_dir, 'transforms.json')
        if osp.isfile(sc):
            dj = json.load(open(sc))
            fr = sorted(dj['frames'], key=lambda f: f['file_path'])
            sc_c2w = W(np.array([f['transform_matrix'] for f in fr], dtype=np.float64))
            eps = max(fscale * 0.05, 1e-4)
            dmin = np.min(np.linalg.norm(sc_c2w[:, :3, 3][:, None] - ref_c2w[:, :3, 3][None],
                                         axis=-1), axis=1)
            add_cams('scene cams (rest)', 'grey', sc_c2w[dmin > eps], dj['fl_x'], dj['fl_y'],
                     dj['w'], dj['h'], (150, 150, 150), 0.6, True,
                     downsample=args.grey_downsample)

    add_cams(f'GT ({len(ref_c2w)})', 'gt', ref_c2w, fx, fy, w, h, (40, 90, 230), 1.0, True)

    for label, run, col in arms:
        p = osp.join(run, 'test', f'{args.name}_transforms_pred.json')
        if not osp.isfile(p):
            print(f'  [skip arm] {p} 없음', flush=True)
            continue
        c2w, afx, afy, aw, ah = load_transforms(p)
        c2w = W(c2w)
        d = np.linalg.norm(c2w[:, :3, 3] - ref_c2w[:, :3, 3], axis=-1).mean() / max(ext, 1e-9)
        add_cams(f'pred: {label}  (ADE/reach {d:.3f})', f'pred_{len(groups)}',
                 c2w, afx, afy, aw, ah, col, 1.0, True)

    for label, run, col, k in ctxs:
        C = _ctx_c2w(run, args.name)
        if C is None:
            print(f'  [skip ctx] {run}/geo_ctx/{args.name}.json 없음', flush=True)
            continue
        C = W(C)
        kk = max(0, min(k, len(C)))
        if kk:
            # _mix_inseg_context 는 inseg + rest 순서라 앞 kk 장이 누수 view 다.
            add_cams(f'{label} [leaked {kk}]', f'ctx_{len(groups)}', C[:kk],
                     fx, fy, w, h, col, 1.3, False)
            if kk < len(C):
                add_cams(f'{label} [kept out-of-seg {len(C) - kk}]', f'ctx_{len(groups)}',
                         C[kk:], fx, fy, w, h, (120, 120, 120), 1.3, False)
        else:
            add_cams(f'{label} ({len(C)})', f'ctx_{len(groups)}', C,
                     fx, fy, w, h, col, 1.3, False)

    @gui_fs.on_update
    def _(_):
        # 크기는 node scale 이라 재생성 없이 바로 먹는다 (geometry 는 scale=1 로 구웠다).
        s = float(max(ext, 1e-3)) * gui_fs.value
        gui_fs_txt.value = f'{s:.4f}'
        with server.atomic():
            for sp in specs:
                for hs in sp['handles']:
                    for hh in hs:
                        hh.scale = s * sp['rel']

    @gui_th.on_update
    def _(_):
        if args.frustum_style != 'spline':
            return
        th = float(gui_th.value)
        t0 = time.time()
        with server.atomic():
            for sp in specs:
                for hs in sp['handles']:
                    for hh in hs:
                        hh.remove()
                sp['handles'] = sp['build'](th)
                # 재생성하면 gui_fs 로 바꿔놓은 크기가 초기값으로 돌아가므로 다시 입힌다.
                s = float(max(ext, 1e-3)) * gui_fs.value * sp['rel']
                for hs in sp['handles']:
                    for hh in hs:
                        hh.scale = s
                _vis_all(sp)
        print(f'  thickness -> {th} (rebuild {time.time() - t0:.2f}s)', flush=True)

    @gui_iv.on_update
    def _(_):
        with server.atomic():
            for sp in specs:
                _vis_all(sp)

    if int(gui_iv.value) > 1:
        with server.atomic():
            for sp in specs:
                _vis_all(sp)

    cap = osp.join(gt_run, 'test', f'{args.name}_caption.json')
    if osp.isfile(cap):
        try:
            server.gui.add_text('caption', initial_value=str(list(json.load(open(cap)).values())[0])[:300],
                                disabled=True)
        except Exception:
            pass

    print(f'viser on port {args.port} — groups: {list(groups)}', flush=True)
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
