"""viser frustum viewer for validation-saved cameras.

root_dir (e.g. results/<exp>/test) holds per-sequence:
  {data_name}_transforms_pred.json   generated cameras (RED)
  {data_name}_transforms_ref.json    target/GT cameras  (BLUE)
  {data_name}_caption.json           text prompt
data_name = "{type}_{hash}_{seg}" -> scene chunk "{type}/{hash}" in DL3DV; that scene's
transforms.json gives the rest of the cameras (GREY).

Coordinate handling: the saved transform_matrix is OpenGL c2w (nerfstudio; y-up, cam looks
-Z), same convention as DL3DV transforms.json. viser camera frustums use OpenCV (cam looks
+Z, y-down), so we convert c2w_opencv = c2w_opengl @ diag(1,-1,-1,1) and pass its rotation
(quaternion) + translation. All three sets share the same world frame (ref/pred are recovered
to the original scene world via out_to_trajectory), so they overlay correctly.

GUI: slider to pick sequence + Load button + Next button.

Run: python scripts/viser_val_cameras.py --root results/<exp>/test [--port 8080]
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, json, glob, argparse
import numpy as np
import viser
import viser.transforms as vtf

DL3DV_ROOT = '/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K'
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_transforms(path):
    """-> (c2w_opengl (N,4,4), fx, fy, w, h)."""
    d = json.load(open(path))
    fr = d['frames']
    c2w = np.array([f['transform_matrix'] for f in fr], dtype=np.float64)
    return c2w, float(d['fl_x']), float(d['fl_y']), float(d['w']), float(d['h'])


def scene_chunk(data_name):
    p = data_name.split('_')
    return f"{p[0]}/{p[1]}"                      # type/hash (hash = single hex token)


def add_frustums(server, name, c2w_gl, fx, fy, w, h, color, scale, downsample=1):
    """Add camera frustums (OpenGL c2w -> OpenCV) under a named group."""
    fov = 2 * np.arctan2(h / 2, fy)              # vertical fov
    aspect = w / h
    handles = []
    for i in range(0, len(c2w_gl), downsample):
        c2w_cv = c2w_gl[i] @ _GL2CV              # OpenGL c2w -> OpenCV c2w (cam looks +Z)
        R, t = c2w_cv[:3, :3], c2w_cv[:3, 3]
        wxyz = vtf.SO3.from_matrix(R).wxyz
        handles.append(server.scene.add_camera_frustum(
            f"{name}/cam_{i:04d}", fov=float(fov), aspect=float(aspect), scale=scale,
            color=color, wxyz=wxyz, position=t.astype(np.float32)))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='results/<exp>/test dir')
    ap.add_argument('--dl3dv-root', default=DL3DV_ROOT)
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--grey-downsample', type=int, default=2, help='subsample scene (grey) cams')
    args = ap.parse_args()

    seqs = sorted(osp.basename(p)[:-len('_transforms_pred.json')]
                  for p in glob.glob(osp.join(args.root, '*_transforms_pred.json')))
    if not seqs:
        raise SystemExit(f"no *_transforms_pred.json in {args.root}")
    print(f"{len(seqs)} sequences in {args.root}")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction('+y')          # OpenGL/nerfstudio world is y-up

    gui_info = server.gui.add_text("sequence", initial_value=seqs[0], disabled=True)
    gui_cap = server.gui.add_text("caption", initial_value="", disabled=True)
    gui_slider = server.gui.add_slider("index", min=0, max=len(seqs) - 1, step=1, initial_value=0)
    gui_load = server.gui.add_button("Load")
    gui_next = server.gui.add_button("Next")
    state = {"groups": []}

    def clear():
        for g in state["groups"]:
            g.remove()
        state["groups"] = []

    def load(idx):
        clear()
        name = seqs[idx]
        gui_info.value = f"[{idx}/{len(seqs)-1}] {name}"
        cap_p = osp.join(args.root, f"{name}_caption.json")
        if osp.isfile(cap_p):
            try:
                gui_cap.value = str(list(json.load(open(cap_p)).values())[0])[:200]
            except Exception:
                gui_cap.value = ""
        ref_c2w, rfx, rfy, rw, rh = load_transforms(osp.join(args.root, f"{name}_transforms_ref.json"))
        pred_c2w, pfx, pfy, pw, ph = load_transforms(osp.join(args.root, f"{name}_transforms_pred.json"))
        # frustum scale from target extent
        ext = np.linalg.norm(ref_c2w[:, :3, 3] - ref_c2w[0, :3, 3], axis=1).max()
        fscale = float(max(ext, 1e-3)) * 0.08
        # grey = the REST of the scene cameras (exclude the target frames, which match exactly)
        sc = osp.join(args.dl3dv_root, scene_chunk(name), 'transforms.json')
        if osp.isfile(sc):
            dj = json.load(open(sc))
            fr = sorted(dj['frames'], key=lambda f: f['file_path'])
            sc_c2w = np.array([f['transform_matrix'] for f in fr], dtype=np.float64)
            eps = max(fscale * 0.05, 1e-4)
            dmin = np.min(np.linalg.norm(sc_c2w[:, :3, 3][:, None] - ref_c2w[:, :3, 3][None], axis=-1), axis=1)
            rest = sc_c2w[dmin > eps]                 # drop target frames from grey
            gh = add_frustums(server, "grey", rest, dj['fl_x'], dj['fl_y'], dj['w'], dj['h'],
                              (150, 150, 150), fscale * 0.6, downsample=args.grey_downsample)
            state["groups"] += gh
        state["groups"] += add_frustums(server, "target", ref_c2w, rfx, rfy, rw, rh, (40, 90, 230), fscale)
        state["groups"] += add_frustums(server, "pred", pred_c2w, pfx, pfy, pw, ph, (230, 40, 40), fscale)
        print(f"loaded [{idx}] {name}: grey_rest + target(blue,{len(ref_c2w)}) + pred(red,{len(pred_c2w)})", flush=True)

    @gui_load.on_click
    def _(_):
        load(int(gui_slider.value))

    @gui_next.on_click
    def _(_):
        gui_slider.value = min(int(gui_slider.value) + 1, len(seqs) - 1)
        load(int(gui_slider.value))

    load(0)
    print(f"viser server on port {args.port} (ctrl-C to stop)")
    import time
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
