"""씬 하나를 브라우저에서 3D 로 본다 — point cloud + OBB + 소스 카메라 + 뱅크(합성) 카메라.

왜 필요한가: 지금까지 판정 수단이 전부 **2D 렌더**였다 (`obb_overlay.mp4`, depth-warp 릴,
contact sheet). 2D 로는 "카메라가 물체 뒤로 들어갔나", "궤적이 왜 저기서 꺾이나", "OBB 가
씬 어디에 앉아 있나"를 못 가른다 — 한 시점에서 겹쳐 보이는 것과 실제로 가까운 것이 구분이
안 되기 때문이다. 여기서는 **한 좌표계(world) 위에 전부 올려놓고** 사람이 돌려 본다.

좌표계는 셋이 섞여 있으므로 그대로 옮기지 않는다:
  · `cloud.npz:points_world` / `meta_cam_c2w` / 뱅크 `poses.npz:cam_c2w` — 전부 **world** (동일).
    (snowboard 에서 뱅크 `cam_c2w[0,0,:3,3]` 이 소스 `cam_c2w[0,:3,3]` 과 비트 단위로 같다 —
     뱅크가 소스 frame0 에서 출발하므로 별도 정합이 필요 없다.)
  · scene graph 의 `obb`/`track` — **graph frame G**. `frames.T_wg` 로 world 로 올려야 한다.
동적 점은 `visible.sum(1) == 1` (`lbm/cloud.py` 와 같은 식)로 가르고, 기본값은 **현재 프레임의
동적 점만** 보여준다 — 49프레임을 전부 겹치면 사람 형체가 번져서 아무것도 안 보인다.

43.8M 점(snowboard)을 그대로 브라우저로 보내면 탭이 죽는다. static/dynamic 을 **따로** 샘플링
하는 이유는 동적 점이 전체의 1~2% 라서 한 덩어리로 샘플링하면 subject 가 먼저 사라지기 때문.

사용 예시 (env vista4d, GPU 불필요):
    python scripts/viser_scene.py --video snowboard --bank hole_bank_k6_d151 --port 8080
    python scripts/viser_scene.py --video snowboard --pin dyn_0__orbit_left__hole0.2 --pin dyn_0__dolly_in__hole0.35
    # 원격이면 로컬에서:  ssh -N -L 8080:127.0.0.1:8080 <host>
"""
import sys
from argparse import ArgumentParser
from glob import glob
from json import load
from os import path
from time import sleep

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from scene_graph.lift import apply_transform                      # noqa: E402
from scene_graph.obb import OBB_EDGES, obb_corners, yaw_to_R      # noqa: E402

# 궤적 색. 소스는 회색(기준), 합성은 팔레트 순서대로 — `make_camviz.py` 의 소스 회색 / 플랜 주황
# 관례를 따르되 여러 개를 동시에 비교할 수 있게 확장했다.
SOURCE_COLOR = (170, 170, 170)
TARGET_COLOR = (255, 60, 60)          # preview + 첫 pin. GUI `target color` 가 덮는다
PIN_PALETTE = [(255, 140, 0), (0, 200, 255), (120, 255, 120), (255, 80, 200),
               (255, 235, 60), (160, 140, 255), (255, 110, 110), (90, 255, 210)]
DYN_COLOR, STAT_COLOR = (0, 255, 255), (255, 128, 0)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_cloud_numpy(cloud_path: str, max_static: int, max_dynamic: int, seed: int):
    """npz → 샘플링된 (static, dynamic) 점. torch/CUDA 를 안 쓴다 — 이 스크립트는 CPU 로 돈다.

    `visible` 을 통째로 unpack 하면 n×f bool (snowboard 2.1 GB) 이라, 프레임 수만 필요한 여기서는
    **바이트 popcount** 로 센다. packbits 의 padding 비트는 0 이라 개수에 영향이 없다.
    """
    data = np.load(cloud_path, allow_pickle=False)
    packed = data["visible_packed"]
    num_visible = POPCOUNT[packed].sum(axis=1, dtype=np.int16)
    is_dynamic = num_visible == 1                       # lbm/cloud.py 와 같은 판별식
    points, colors = data["points_world"], data["colors"]
    frame_of = data["indices"][:, 0].astype(np.int32)   # 동적 점이 속한 프레임
    rng = np.random.default_rng(seed)

    def take(mask, cap):
        idx = np.flatnonzero(mask)
        if len(idx) > cap:
            idx = idx[rng.permutation(len(idx))[:cap]]
        return points[idx].astype(np.float32), colors[idx], frame_of[idx]

    static = take(~is_dynamic, max_static)
    dynamic = take(is_dynamic, max_dynamic)
    meta = {k[5:]: data[k] for k in data.files if k.startswith("meta_")}
    return static, dynamic, meta, int(is_dynamic.sum()), len(points)


def node_obb_world(node: dict, frame: int, T_wg: np.ndarray):
    """노드 OBB 12-edge 를 world 선분 (12, 2, 3) 으로. 동적은 프레임별 track 값을 쓴다 —
    `node["obb"]` 는 ref 프레임 값이라 움직이는 물체엔 엉뚱한 자리에 박힌다 (`obb.node_obb_at` 주석).
    """
    obb = node["obb"]
    extent = np.asarray(obb["extent"], dtype=float)
    if node["kind"] == "dyn" and frame in set(node["track"]["frames"]):
        center = np.asarray(node["track"]["center_smooth"], dtype=float)[frame]
        R = yaw_to_R(float(np.asarray(node["track"]["yaw"], dtype=float)[frame]))
    else:
        center, R = np.asarray(obb["center"], dtype=float), np.asarray(obb["R"], dtype=float)
    corners = apply_transform(T_wg, obb_corners(center, extent, R))
    return corners[np.asarray(OBB_EDGES)].astype(np.float32), corners


def frustum_args(c2w: np.ndarray, K: np.ndarray, height: int, width: int):
    """viser frustum 은 OpenCV 규약(+Z forward, +Y down)이라 c2w 를 그대로 쓴다."""
    from viser.transforms import SO3
    fov = 2.0 * float(np.arctan(height / (2.0 * K[1, 1])))
    return fov, width / height, SO3.from_matrix(c2w[:3, :3]).wxyz, c2w[:3, 3]


def list_banks(scene_dir: str):
    return sorted(path.basename(path.dirname(p))
                  for p in glob(path.join(scene_dir, "*", "poses.npz")))


def main(args):
    import viser

    scene_dir = path.join(args.out_root if path.isabs(args.out_root)
                          else path.join(CINEMATRAJ_ROOT, args.out_root), args.video)
    cloud_path, graph_path = path.join(scene_dir, "cloud.npz"), path.join(scene_dir, "scene_graph.json")
    assert path.isfile(cloud_path), f"cloud.npz 가 없다: {cloud_path} (lbm.cloud 를 먼저 돌릴 것)"
    assert path.isfile(graph_path), f"scene_graph.json 이 없다: {graph_path}"

    static, dynamic, meta, num_dynamic, num_total = load_cloud_numpy(
        cloud_path, args.max_static, args.max_dynamic, args.seed)
    with open(graph_path, encoding="utf-8") as file:
        graph = load(file)
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
    up_world = np.asarray(graph["gravity"]["up_world"], dtype=float)
    cam_c2w = np.asarray(meta["cam_c2w"], dtype=float)
    K = np.asarray(meta["K"], dtype=float)
    num_frames, height, width = len(cam_c2w), int(meta["height"]), int(meta["width"])
    nodes = [n for n in graph["nodes"] if args.static_nodes or n["kind"] == "dyn"]

    banks = list_banks(scene_dir)
    bank_name = args.bank if args.bank in banks else (banks[0] if banks else "")
    bank = {"name": None, "cam": None, "variant": None, "anchor": None}

    def load_bank(name):
        """뱅크 poses.npz 는 lazy 로 읽는다 — 616×49×4×4 float64 가 19 MB 라 전부 미리 읽을 이유가 없다."""
        if bank["name"] == name or not name:
            return
        npz = np.load(path.join(scene_dir, name, "poses.npz"), allow_pickle=False)
        bank.update(name=name, cam=npz["cam_c2w"], variant=npz["variant_id"].astype(str),
                    anchor=npz["anchor_id"].astype(str))

    load_bank(bank_name)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction(tuple(float(v) for v in up_world))
    server.scene.world_axes.visible = False

    # ── point cloud ──────────────────────────────────────────────────────────
    cloud_static = server.scene.add_point_cloud(
        "/cloud/static", static[0], static[1], point_size=args.point_size)
    cloud_dynamic = server.scene.add_point_cloud(
        "/cloud/dynamic", dynamic[0][:1], dynamic[1][:1], point_size=args.point_size * 1.6)

    # ── OBB ──────────────────────────────────────────────────────────────────
    obb_handles, label_handles = {}, {}
    for node in nodes:
        segs, corners = node_obb_world(node, 0, T_wg)
        color = DYN_COLOR if node["kind"] == "dyn" else STAT_COLOR
        obb_handles[node["id"]] = server.scene.add_line_segments(
            f"/obb/{node['id']}", segs, color, thickness=3.0, thickness_units="screen")
        label_handles[node["id"]] = server.scene.add_label(
            f"/obb/{node['id']}/label", f"{node['id']} {node['label']}",
            position=corners.mean(axis=0) + up_world * 0.0)

    # ── 카메라 ───────────────────────────────────────────────────────────────
    def draw_track(prefix: str, poses: np.ndarray, color, stride: int, label: str):
        """궤적 = 중심 스플라인 + stride 간격 frustum. 49개를 다 그리면 화면이 frustum 벽이 된다."""
        handles = [server.scene.add_spline_catmull_rom(
            f"{prefix}/path", poses[:, :3, 3].astype(np.float32), color=color,
            thickness=2.0, thickness_units="screen")]
        for f in range(0, len(poses), stride):
            fov, aspect, wxyz, position = frustum_args(poses[f], K[min(f, len(K) - 1)], height, width)
            handles.append(server.scene.add_camera_frustum(
                f"{prefix}/f{f:02d}", fov, aspect, scale=args.cam_scale, color=color,
                wxyz=wxyz, position=position, thickness=1.5, thickness_units="screen"))
        handles.append(server.scene.add_label(f"{prefix}/name", label, position=poses[0, :3, 3]))
        return handles

    source_handles = []
    cur_source = server.scene.add_camera_frustum(
        "/cam/cur_source", *frustum_args(cam_c2w[0], K[0], height, width)[:2],
        scale=args.cam_scale * 1.8, color=(255, 255, 255), thickness=3.0, thickness_units="screen")
    pins, pin_handles = {}, []     # variant_id -> (bank_name, poses)

    def redraw_pins():
        while pin_handles:
            pin_handles.pop().remove()
        # 첫 pin 은 GUI 의 `target color` 를 쓴다 — 하나만 고정했을 때 "지정한 색"이 그대로
        # 나오게 하려는 것. 둘 이상이면 서로 구별돼야 하므로 나머지는 팔레트로 넘긴다.
        for i, (vid, (bname, poses)) in enumerate(pins.items()):
            color = tuple(gui_tgt_color.value) if i == 0 else PIN_PALETTE[(i - 1) % len(PIN_PALETTE)]
            pin_handles.extend(draw_track(f"/cam/pin/{i:02d}", poses, color,
                                          args.cam_stride, f"{bname}:{vid}"))

    # ── GUI ──────────────────────────────────────────────────────────────────
    with server.gui.add_folder("scene"):
        server.gui.add_markdown(
            f"**{args.video}** — {num_frames}f {width}x{height}\n\n"
            f"points {num_total:,} (dyn {num_dynamic:,}) / shown "
            f"{len(static[0]):,}+{len(dynamic[0]):,}\n\n"
            f"S={float(meta['S']):.3f}  z_med={float(meta['z_med_frame0']):.3f}  "
            f"plx={float(meta['parallax_ratio']):.3f}")
        gui_frame = server.gui.add_slider("frame", 0, num_frames - 1, 1, 0)
        gui_play = server.gui.add_checkbox("play", False)
        gui_fps = server.gui.add_slider("fps", 1, 30, 1, int(round(float(meta.get("fps", 10)))))

    with server.gui.add_folder("cloud"):
        gui_show_static = server.gui.add_checkbox("static points", True)
        gui_show_dyn = server.gui.add_checkbox("dynamic points", True)
        gui_dyn_mode = server.gui.add_dropdown("dynamic span (points+OBB)",
                                               ("current frame", "interval", "all frames"))
        # `interval` 일 때만 쓰는 구간. 전량(`all frames`)은 동적 점이 겹쳐서 어느 구간에서
        # 물체가 어디로 갔는지 안 보이는데, 구간을 좁히면 그게 보인다.
        gui_span_lo = server.gui.add_slider("span start", 0, num_frames - 1, 1, 0)
        gui_span_hi = server.gui.add_slider("span end", 0, num_frames - 1, 1,
                                            min(8, num_frames - 1))
        gui_psize = server.gui.add_slider("point size", 0.001, 0.05, 0.001, args.point_size)

    with server.gui.add_folder("obb"):
        gui_show_dyn_obb = server.gui.add_checkbox("dynamic OBB", True)
        gui_show_stat_obb = server.gui.add_checkbox("static OBB", args.static_nodes)
        gui_show_labels = server.gui.add_checkbox("labels", True)

    with server.gui.add_folder("cameras"):
        gui_show_src = server.gui.add_checkbox("source trajectory", True)
        gui_bank = server.gui.add_dropdown("bank", tuple(banks) or ("(none)",),
                                           initial_value=bank_name or "(none)")
        anchors = ["(all)"] + sorted(set(bank["anchor"])) if bank["cam"] is not None else ["(all)"]
        gui_anchor = server.gui.add_dropdown("anchor", tuple(anchors), initial_value="(all)")
        gui_variant = server.gui.add_dropdown(
            "variant", tuple(bank["variant"]) if bank["cam"] is not None else ("(none)",))
        gui_preview = server.gui.add_checkbox("preview selected", True)
        # 색은 상수가 아니라 GUI 로 뺀다 — 배경·점군 색과 겹치면 궤적이 안 보이는데, 씬마다
        # 점군 색이 다르라 상수 하나로는 못 맞춘다.
        gui_src_color = server.gui.add_rgb("source color", SOURCE_COLOR)
        gui_tgt_color = server.gui.add_rgb("target color", TARGET_COLOR)
        gui_add = server.gui.add_button("pin selected")
        gui_clear = server.gui.add_button("clear pinned")

    def variant_options():
        if bank["cam"] is None:
            return ("(none)",)
        keep = bank["variant"] if gui_anchor.value == "(all)" else \
            bank["variant"][bank["anchor"] == gui_anchor.value]
        return tuple(keep) or ("(none)",)

    def selected_poses():
        if bank["cam"] is None or gui_variant.value == "(none)":
            return None
        hit = np.flatnonzero(bank["variant"] == gui_variant.value)
        return None if not len(hit) else np.asarray(bank["cam"][hit[0]], dtype=float)

    preview_handles = []

    def redraw_preview():
        while preview_handles:
            preview_handles.pop().remove()
        poses = selected_poses()
        if poses is None or not gui_preview.value:
            return
        preview_handles.extend(
            draw_track("/cam/preview", poses, tuple(gui_tgt_color.value),
                       args.cam_stride, gui_variant.value))

    def redraw_source():
        """색을 바꾸려면 다시 그리는 수밖에 없다 — viser handle 은 color 를 못 갈아끼운다."""
        while source_handles:
            source_handles.pop().remove()
        source_handles.extend(draw_track("/cam/source", cam_c2w, tuple(gui_src_color.value),
                                         args.cam_stride, "source"))
        for handle in source_handles:
            handle.visible = gui_show_src.value

    frame_obb_cache, span_obb_cache = {}, {}

    def obb_span(node, lo: int, hi: int):
        """동적 노드 OBB 를 `[lo, hi]` 구간만큼 쌓아 선분 한 덩어리로.

        프레임별 선분은 `frame_obb_cache` 에 한 번만 계산해 두고, 구간 합은 슬라이더를 끌면
        매번 달라지므로 `(id, lo, hi)` 로 따로 캐시한다 — 안 그러면 드래그마다 49번 재계산한다.
        """
        key = (node["id"], lo, hi)
        if key not in span_obb_cache:
            frames = [f for f in node["track"]["frames"] if lo <= f <= hi]
            if not frames:
                frames = [max(min(hi, node["track"]["frames"][-1]), node["track"]["frames"][0])]
            for f in frames:
                if (node["id"], f) not in frame_obb_cache:
                    frame_obb_cache[(node["id"], f)] = node_obb_world(node, f, T_wg)[0]
            span_obb_cache[key] = np.concatenate(
                [frame_obb_cache[(node["id"], f)] for f in frames], axis=0)
        return span_obb_cache[key]

    def dyn_span():
        """현재 모드가 뜻하는 프레임 구간 `[lo, hi]`. 구간 모드에서 start > end 면 뒤집는다."""
        mode = gui_dyn_mode.value
        if mode == "all frames":
            return 0, num_frames - 1
        if mode == "current frame":
            f = int(gui_frame.value)
            return f, f
        lo, hi = int(gui_span_lo.value), int(gui_span_hi.value)
        return (lo, hi) if lo <= hi else (hi, lo)

    def update_frame(_=None):
        f = int(gui_frame.value)
        lo, hi = dyn_span()
        cloud_dynamic.visible = gui_show_dyn.value
        if gui_show_dyn.value:
            keep = (dynamic[2] >= lo) & (dynamic[2] <= hi)
            cloud_dynamic.points = dynamic[0][keep] if keep.any() else dynamic[0][:1] * 0
            cloud_dynamic.colors = dynamic[1][keep] if keep.any() else dynamic[1][:1]
        for node in nodes:
            handle = obb_handles[node["id"]]
            on = gui_show_dyn_obb.value if node["kind"] == "dyn" else gui_show_stat_obb.value
            handle.visible = on
            label_handles[node["id"]].visible = on and gui_show_labels.value
            if on and node["kind"] == "dyn":
                segs, corners = node_obb_world(node, f, T_wg)
                handle.points = segs if lo == hi == f else obb_span(node, lo, hi)
                label_handles[node["id"]].position = corners.mean(axis=0)
        fov, aspect, wxyz, position = frustum_args(cam_c2w[f], K[f], height, width)
        cur_source.wxyz, cur_source.position = wxyz, position
        cur_source.visible = gui_show_src.value

    gui_frame.on_update(update_frame)
    gui_show_dyn.on_update(update_frame)
    gui_dyn_mode.on_update(update_frame)
    gui_span_lo.on_update(update_frame)
    gui_span_hi.on_update(update_frame)
    gui_show_dyn_obb.on_update(update_frame)
    gui_show_stat_obb.on_update(update_frame)
    gui_show_labels.on_update(update_frame)
    gui_show_static.on_update(lambda _: setattr(cloud_static, "visible", gui_show_static.value))
    gui_psize.on_update(lambda _: [setattr(cloud_static, "point_size", gui_psize.value),
                                   setattr(cloud_dynamic, "point_size", gui_psize.value * 1.6)])

    def set_source_visible(_=None):
        for handle in source_handles:
            handle.visible = gui_show_src.value
        update_frame()

    gui_show_src.on_update(set_source_visible)
    gui_src_color.on_update(lambda _: redraw_source())
    gui_tgt_color.on_update(lambda _: [redraw_preview(), redraw_pins()])

    def on_bank(_=None):
        load_bank(gui_bank.value)
        gui_anchor.options = tuple(["(all)"] + sorted(set(bank["anchor"])))
        gui_variant.options = variant_options()
        redraw_preview()

    def on_anchor(_=None):
        gui_variant.options = variant_options()
        redraw_preview()

    gui_bank.on_update(on_bank)
    gui_anchor.on_update(on_anchor)
    gui_variant.on_update(lambda _: redraw_preview())
    gui_preview.on_update(lambda _: redraw_preview())

    @gui_add.on_click
    def _(_):
        poses = selected_poses()
        if poses is not None and gui_variant.value not in pins:
            pins[gui_variant.value] = (bank["name"], poses)
            redraw_pins()

    @gui_clear.on_click
    def _(_):
        pins.clear()
        redraw_pins()

    for vid in args.pin:                       # CLI 로 미리 고정한 궤적
        if bank["cam"] is not None and vid in set(bank["variant"]):
            pins[vid] = (bank["name"], np.asarray(bank["cam"][np.flatnonzero(bank["variant"] == vid)[0]],
                                                  dtype=float))
    redraw_source()
    redraw_pins()
    redraw_preview()
    update_frame()

    print(f"{'video':<14}{args.video}")
    print(f"{'url':<14}http://{args.host}:{args.port}   (원격이면 ssh -N -L "
          f"{args.port}:127.0.0.1:{args.port} <host>)")
    print(f"{'frames':<14}{num_frames}  {width}x{height}  fps {float(meta.get('fps', 10)):.1f}")
    print(f"{'points':<14}{num_total:,}  dynamic {num_dynamic:,}  "
          f"-> shown {len(static[0]):,} + {len(dynamic[0]):,}")
    print(f"{'nodes':<14}{len(nodes)}  "
          f"(dyn {sum(n['kind'] == 'dyn' for n in nodes)}, stat {sum(n['kind'] != 'dyn' for n in nodes)})")
    print(f"{'banks':<14}{len(banks)}  selected {bank['name']} "
          f"({0 if bank['cam'] is None else len(bank['cam'])} variants)")
    print(f"{'pinned':<14}{len(pins)}  {', '.join(pins) if pins else '-'}")

    while True:
        if gui_play.value:
            gui_frame.value = (int(gui_frame.value) + 1) % num_frames
        sleep(1.0 / max(int(gui_fps.value), 1))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)                   # 씬 이름 (예: snowboard)
    parser.add_argument("--out_root", default="out", type=str)                # cloud/graph 루트
    parser.add_argument("--bank", default="hole_bank_k6_d151", type=str)      # 초기 뱅크 폴더
    parser.add_argument("--pin", default=[], action="append")                 # 시작부터 켜 둘 variant_id
    parser.add_argument("--max_static", default=300000, type=int)             # 정적 점 상한
    parser.add_argument("--max_dynamic", default=400000, type=int)            # 동적 점 상한(49프레임 합)
    parser.add_argument("--point_size", default=0.008, type=float)
    parser.add_argument("--cam_scale", default=0.08, type=float)              # frustum 크기
    parser.add_argument("--cam_stride", default=4, type=int)                  # frustum 을 그릴 간격
    parser.add_argument("--static_nodes", action="store_true", default=True)  # 정적 노드 OBB 포함
    parser.add_argument("--no_static_nodes", dest="static_nodes", action="store_false")
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--seed", default=0, type=int)
    main(parser.parse_args())
