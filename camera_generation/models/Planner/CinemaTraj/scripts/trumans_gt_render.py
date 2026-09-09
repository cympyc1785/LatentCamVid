"""TRUMANS `.blend` 를 headless Blender 로 돌려 **ground-truth** RGB / depth / instance index 를 뽑는다.

**왜 필요한가.** 지금까지 우리 파이프라인의 depth 와 사람 마스크는 전부 *추정치*였다 — depth 는
DA3(Depth-Anything-3), 마스크는 SAM3. 그래서 "카메라를 이만큼 움직였을 때 hole 이 얼마나
생기는가", "사람이 프레임 어디에 걸리는가" 같은 판정이 항상 **추정기 오차와 섞여** 나왔다.
DA3 는 scale/shift 가 프레임마다 흔들리고(정지 카메라에서도 focal 이 −14% 드리프트), SAM3 는
실루엣 경계에서 몇 픽셀씩 샌다. 즉 우리가 재고 싶었던 *기하* 신호와 *추정기* 노이즈를 못 갈랐다.

TRUMANS 는 씬 전체가 `.blend` 로 들어 있다 → **렌더러가 정답을 알고 있다.** depth 는 Blender
metre 단위 실측이고, instance mask 는 object pass index 라 실루엣이 픽셀 단위로 정확하다.
이 스크립트는 그 정답을 디스크로 꺼내서 추정기 없이 hole/가림/재투영을 검증할 수 있게 한다.

**측정으로 확정된 사실 (Blender 4.5.9 LTS, 재논의 금지).**
- `BLENDER_EEVEE_NEXT` 에는 object index pass 가 **아예 없다** — `use_pass_object_index=True` 를
  켜도 Render Layers 노드에 `IndexOB` 소켓이 생기지 않는다. index 는 Cycles 전용이다.
- EEVEE 의 Z pass 는 픽셀 footprint 위에서 **필터링**되어 실루엣에서 틀린다 (1.6% 픽셀이
  1cm 이상, 최대 3.85m). depth 는 Cycles pass 에서 받는다.
- 그래서 프레임당 2-pass 다: RGB 는 `--rgb_engine` 이 정하고, depth+index 는 **항상** Cycles
  1 spp **CPU**(~1.8 s). Cycles GPU 는 여기서 오히려 느리고(2.55 s) OptiX 커널 빌드에
  일회성 437 초를 더 쓴다 → `--cdevice CPU` 가 기본값.
- **RGB 엔진은 EEVEE 가 최적이 아니다 (2026-08-27 정정).** 이 blend 들은 전부 **Blender 3.3.6
  으로 저작**됐는데 우리는 4.5.9 로 돌린다 — 4.2 에서 EEVEE 가 EEVEE_NEXT 로 전면 재작성됐고,
  3.3 기준으로 맞춰둔 발광 재질(`Emission Strength` 노트북 20 / TV 20 / 조명 175~469)이
  EEVEE_NEXT 에서 흰 덩어리로 타서 **옆 물체까지 번진다**. 같은 프레임/카메라/조명을 Cycles 로
  렌더하면 번짐이 0 이다 (frame 402 crop 안 `max>=200` 픽셀: EEVEE 401 → Cycles **0**).
  `--rgb_engine cycles` 가 그 탈출구다(기본은 하위 호환 때문에 `eevee`). 비용은 960x540 OPTIX
  기준 5.7 s/frame(정상 상태) — EEVEE_NEXT 는 첫 프레임 셰이더 컴파일에만 61 s 를 쓰므로
  실제 격차는 작고, headless 다중 프레임 EEVEE_NEXT 는 crash 한 전례도 있다.
- depth 는 **z-planar** (카메라 forward 축 투영 거리), 단위 metre, 배경 sentinel 은 1e10.
  픽셀 중심 규약은 (col+0.5, row+0.5).
- `--cycles` 로 시작하는 CLI 플래그를 **절대 만들지 말 것**. Cycles 애드온이 `--` 를 무시하고
  argv 전체를 argparse 로 훑으며 prefix-match 하기 때문에 실행이 통째로 죽는다 (그래서 `--cdevice`).
- vista4d env 의 cv2 는 `OpenEXR: NO` 다 → EXR→`.npy` 변환을 **Blender 안에서** 끝낸다.
- Compositor File Output 노드는 자기 `format` 을 들고 있어서 `image_settings.file_format="PNG"`
  와 충돌하지 않는다 (RGB 는 PNG, depth/index 는 32bit EXR).
- TRUMANS `.blend` 는 `scene.frame_step = 2` 로 저장되어 있다 → 1 로 되돌린다.
- 사람은 ARMATURE `zzy3` 이고 그 자체는 아무것도 렌더하지 않는다. 실제로 그려지는 건 그 armature
  가 deform 하는 mesh 12개(CC_Base_Body, Layered_sweater, Slim_Jeans, ...)라 **union 을 index 1
  로 예약**한다. 이름 하드코딩이 아니라 armature modifier / parent 로 찾는다.
- full-house 씬이라 카메라를 아무데나 두면 **벽 안쪽**에 박힌다. 포즈는 반드시 밖에서 준다.

카메라 소스 3종 (택1):
    --poses <npz>            key `cam_c2w` (N,4,4) **OpenCV** convention, 선택적으로 `K`(3,3) 또는
                             `lens_mm`. i번째 포즈가 `--frames` 의 i번째 프레임에 붙는다.
    --scene_camera <name>    .blend 안의 기존 카메라 오브젝트 (애니메이션 그대로 따라간다)
    --camera_pose_pkl <pkl>  TRUMANS `<seq>_camera_pose.pkl`
                             (dict frame -> {'location':[3], 'rotation':[3] euler XYZ radian})

사용 예시:
    B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    R=/data1/cympyc1785/data/trumans/Data_release/Recordings_blend/00add26c-7a26-4a61-b192-b97aa493b3f3
    $B -b $R/00add26c-7a26-4a61-b192-b97aa493b3f3.blend --python scripts/trumans_gt_render.py -- \
        --camera_pose_pkl $R/2023-01-17@00-33-01_camera_pose.pkl \
        --frames 100 108 --res 960 540 --passes rgb,depth,index --out /tmp/gt_00add26c

    $B -b $R/....blend --python scripts/trumans_gt_render.py -- \
        --poses /tmp/my_traj.npz --frames 100 103 --passes depth,index --keep_exr --out /tmp/gt_traj
"""
import json
import pickle
import sys
import time
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path, remove, rmdir

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

# GL(오른쪽 X, 위 Y, 뒤 Z) <-> CV(오른쪽 X, 아래 Y, 앞 Z). 자기 자신이 역행렬이라 양방향 동일.
GL2CV = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
# pass_index 를 받을 오브젝트 타입. ARMATURE/EMPTY/LIGHT 는 렌더 결과가 없어 0 으로 둔다.
RENDERABLE = {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME", "GREASEPENCIL"}
HUMAN_INDEX = 1          # 예약: 사람 mesh union
ALL_PASSES = ("rgb", "depth", "index")


def cli_argv():
    """Blender 는 `--` 뒤를 스크립트 몫으로 남긴다. `--` 가 없으면 인자 없음."""
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


# ---------------------------------------------------------------------------------------
# 사람 mesh 판별
# ---------------------------------------------------------------------------------------
def find_human_meshes():
    """ARMATURE 가 deform 하는 mesh 를 모은다 → (armature 이름, [mesh 이름]) 목록.

    이름 하드코딩 금지. 판정은 두 가지 중 하나만 걸려도 된다:
      (a) ARMATURE modifier 의 `object` 가 armature 를 가리킨다  (deform 의 실제 근거)
      (b) `parent` 가 ARMATURE 다                                 (modifier 없이 붙여둔 경우)
    """
    by_armature = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        armature = None
        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and mod.object is not None:
                armature = mod.object.name
                break
        if armature is None and obj.parent is not None and obj.parent.type == "ARMATURE":
            armature = obj.parent.name
        if armature is not None:
            by_armature.setdefault(armature, []).append(obj.name)
    return {arm: sorted(names) for arm, names in by_armature.items()}


def assign_pass_indices():
    """index 1 = 사람 union, 2..N = 나머지 renderable 을 이름순. 반환은 objects.json 페이로드."""
    humans = find_human_meshes()
    human_names = sorted(name for names in humans.values() for name in names)
    human_set = set(human_names)

    objects, next_index = {}, HUMAN_INDEX
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        if obj.type not in RENDERABLE:
            obj.pass_index = 0
            continue
        if obj.name in human_set:
            obj.pass_index = HUMAN_INDEX
            continue
        next_index += 1
        obj.pass_index = next_index
        objects[str(next_index)] = {"name": obj.name, "type": obj.type,
                                    "is_human": False, "hide_render": obj.hide_render}
    objects[str(HUMAN_INDEX)] = {
        "name": "<human>", "type": "MESH_UNION", "is_human": True, "hide_render": False,
        "member_names": human_names,
    }
    return {
        "note": "index 0 = background / 렌더되지 않는 오브젝트. 값은 pass_index.",
        "human_indices": [HUMAN_INDEX],
        "human_armatures": sorted(humans),
        "human_meshes": human_names,
        "n_human_meshes": len(human_names),
        "n_indices": next_index,
        "objects": objects,
    }


# ---------------------------------------------------------------------------------------
# intrinsics
# ---------------------------------------------------------------------------------------
def K_analytic(camd, scene):
    """Blender lens/sensor -> 픽셀 intrinsics. `BKE_camera_params_compute_viewplane()` 재현.

    324개 (해상도 x sensor_fit x sensor 크기 x shift x pixel aspect x lens) 조합에서
    `calc_matrix_camera()` 와 최대 상대오차 2.3e-7 로 일치함을 확인했다.

        ycor        = pixel_aspect_y / pixel_aspect_x
        fit_hor     = (sensor_fit=='HORIZONTAL') 또는 AUTO 이면서 ax*W >= ay*H
        sensor_mm   = sensor_height 는 sensor_fit=='VERTICAL' 일 때만 쓰인다 (AUTO 는 width)
        view_fac_px = W (가로 fit) 또는 ycor*H (세로 fit)
        fx = lens/sensor_mm * view_fac_px,   fy = fx/ycor
        cx = W/2 - shift_x*view_fac_px,      cy = H/2 + shift_y*view_fac_px/ycor
    """
    render = scene.render
    scale = render.resolution_percentage / 100.0
    width, height = int(render.resolution_x * scale), int(render.resolution_y * scale)
    ax, ay = render.pixel_aspect_x, render.pixel_aspect_y
    ycor = ay / ax

    fit = camd.sensor_fit
    fit_hor = (ax * width) >= (ay * height) if fit == "AUTO" else (fit == "HORIZONTAL")
    sensor = camd.sensor_height if fit == "VERTICAL" else camd.sensor_width

    view_fac_px = width if fit_hor else ycor * height
    fx = camd.lens / sensor * view_fac_px
    fy = fx / ycor
    cx = width * 0.5 - camd.shift_x * view_fac_px
    cy = height * 0.5 + camd.shift_y * view_fac_px / ycor
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], width, height


def K_from_projection(cam_obj, scene):
    """교차검증용: Blender 자신의 GL projection matrix 에서 K 를 역산한다."""
    render = scene.render
    scale = render.resolution_percentage / 100.0
    width, height = int(render.resolution_x * scale), int(render.resolution_y * scale)
    proj = cam_obj.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height,
                                      scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y)
    return [[proj[0][0] * width * 0.5, 0.0, width * 0.5 * (1.0 - proj[0][2])],
            [0.0, proj[1][1] * height * 0.5, height * 0.5 * (1.0 + proj[1][2])],
            [0.0, 0.0, 1.0]]


def apply_K(camd, scene, K):
    """주어진 K(3,3) 를 Blender 카메라 파라미터로 되돌린다 (K_analytic 의 역).

    sensor_fit 을 HORIZONTAL 로 못 박아 view_fac_px = W 로 고정하고,
    fx != fy 는 pixel_aspect 비율(ycor = fx/fy)로 흡수한다. Blender 의 pixel_aspect 는
    1 미만을 싫어하므로 큰 쪽을 1 로 정규화한다 (비율만 의미가 있다).
    """
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    scene.render.pixel_aspect_x = max(1.0, fy / fx)
    scene.render.pixel_aspect_y = max(1.0, fx / fy)
    ycor = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    camd.sensor_fit = "HORIZONTAL"
    camd.lens = fx * camd.sensor_width / width
    camd.shift_x = (width * 0.5 - cx) / width
    camd.shift_y = (cy - height * 0.5) * ycor / width
    return width, height


def camera_record(cam_obj, scene, frame, order):
    K, width, height = K_analytic(cam_obj.data, scene)
    K_proj = K_from_projection(cam_obj, scene)
    c2w_gl = cam_obj.matrix_world.copy()
    c2w_cv = c2w_gl @ GL2CV
    return {
        "index": order, "frame": frame, "width": width, "height": height,
        "K": K,
        "K_max_abs_diff_vs_blender_projection": max(abs(K[i][j] - K_proj[i][j])
                                                    for i in range(3) for j in range(3)),
        "c2w_opencv": [list(row) for row in c2w_cv],
        "c2w_blender_gl": [list(row) for row in c2w_gl],
        "w2c_opencv": [list(row) for row in c2w_cv.inverted()],
        "lens_mm": cam_obj.data.lens,
        "sensor_width_mm": cam_obj.data.sensor_width,
        "sensor_height_mm": cam_obj.data.sensor_height,
        "sensor_fit": cam_obj.data.sensor_fit,
        "shift_x": cam_obj.data.shift_x, "shift_y": cam_obj.data.shift_y,
        "clip_start": cam_obj.data.clip_start, "clip_end": cam_obj.data.clip_end,
    }


# ---------------------------------------------------------------------------------------
# 카메라 소스
# ---------------------------------------------------------------------------------------
def load_poses_npz(npz_path):
    data = np.load(npz_path)
    assert "cam_c2w" in data, f"{npz_path} 에 key 'cam_c2w' 가 없다. keys={list(data)}"
    c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
    assert c2w.ndim == 3 and c2w.shape[1:] == (4, 4), f"cam_c2w shape 이 (N,4,4) 가 아니다: {c2w.shape}"
    K = np.asarray(data["K"], dtype=np.float64) if "K" in data else None
    lens = float(np.asarray(data["lens_mm"]).reshape(-1)[0]) if "lens_mm" in data else None
    return c2w, K, lens


def load_camera_pose_pkl(pkl_path):
    with open(pkl_path, "rb") as file:
        table = pickle.load(file)
    assert isinstance(table, dict) and table, f"{pkl_path} 가 비었거나 dict 가 아니다"
    return {int(k): v for k, v in table.items()}


def pkl_matrix(entry):
    """TRUMANS pkl 한 항목 -> Blender c2w (GL). rotation 은 euler XYZ radian."""
    loc = Vector([float(x) for x in entry["location"]])
    rot = Euler([float(x) for x in entry["rotation"]], "XYZ")
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4()


def new_camera(scene, lens, sensor):
    camd = bpy.data.cameras.new("GTRenderCamData")
    camd.type = "PERSP"
    camd.lens = lens
    camd.sensor_fit = "AUTO"
    camd.sensor_width = sensor
    camd.sensor_height = sensor * 2.0 / 3.0
    camd.shift_x = camd.shift_y = 0.0
    camd.clip_start, camd.clip_end = 0.05, 1000.0
    cam_obj = bpy.data.objects.new("GTRenderCam", camd)
    scene.collection.objects.link(cam_obj)
    return cam_obj


# ---------------------------------------------------------------------------------------
# compositor / 렌더
# ---------------------------------------------------------------------------------------
def build_compositor(scene, exr_dir, want_depth, want_index):
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])

    def file_out(name, slot):
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.name = name
        node.base_path = exr_dir
        node.format.file_format = "OPEN_EXR"
        node.format.color_mode = "BW"       # 단일 채널 -> 파일 최소
        node.format.color_depth = "32"      # float32. half 면 depth 가 양자화된다
        node.format.exr_codec = "ZIP"
        node.file_slots.clear()
        node.file_slots.new(slot)
        return node

    outs = {}
    if want_depth:
        outs["depth"] = file_out("FO_DEPTH", "depth")
    if want_index:
        outs["index"] = file_out("FO_INDEX", "index")
    return tree, layers, composite, outs


def link_pass_sockets(tree, layers, outs):
    """Cycles 로 바꾼 뒤에야 `IndexOB` 소켓이 생긴다 → 그 시점에 다시 잇는다.

    입력 소켓은 **이름이 아니라 인덱스**로 잡는다. File Output 노드는 `file_slots[i].path`
    를 소켓 이름으로 쓰기 때문에, 프레임마다 path 를 바꾸면 소켓 이름도 같이 바뀐다.
    """
    pairs = (("depth", "Depth"), ("index", "IndexOB"))
    for key, socket in pairs:
        node = outs.get(key)
        if node is None:
            continue
        target = node.inputs[0]
        if target.is_linked or layers.outputs.get(socket) is None:
            continue
        tree.links.new(layers.outputs[socket], target)


def exr_to_array(exr_path):
    """Blender 내부 이미지 로더로 EXR -> float32 (H,W). 픽셀 버퍼는 bottom-up 이라 뒤집는다."""
    img = bpy.data.images.load(exr_path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    width, height = img.size
    buf = np.empty(width * height * img.channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(height, width, img.channels)[::-1, :, 0].copy()
    bpy.data.images.remove(img)
    return arr


def main(args):
    scene = bpy.context.scene
    layer = scene.view_layers[0]
    passes = [p.strip() for p in args.passes.split(",") if p.strip()]
    assert passes and all(p in ALL_PASSES for p in passes), \
        f"--passes 는 {ALL_PASSES} 의 콤마 목록이어야 한다: {args.passes!r}"
    want_rgb, want_depth, want_index = ("rgb" in passes), ("depth" in passes), ("index" in passes)
    need_cycles = want_depth or want_index      # index 는 Cycles 전용, depth 는 EEVEE 가 틀린다

    sources = [bool(args.poses), bool(args.scene_camera), bool(args.camera_pose_pkl)]
    assert sum(sources) == 1, "--poses / --scene_camera / --camera_pose_pkl 중 정확히 하나를 줄 것"

    out = args.out
    for sub in (["rgb"] if want_rgb else []) + (["depth"] if want_depth else []) \
            + (["index"] if want_index else []):
        makedirs(path.join(out, sub), exist_ok=True)
    exr_dir = path.join(out, "_exr")
    makedirs(exr_dir, exist_ok=True)

    # --- 렌더 설정 --------------------------------------------------------------------
    scene.frame_step = 1                       # TRUMANS blend 는 2 로 저장되어 있다
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True    # 프레임 간 재사용 -> 큰 속도 이득
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.eevee.taa_render_samples = args.samples
    layer.use_pass_combined = True
    layer.use_pass_z = want_depth
    layer.use_pass_object_index = want_index

    assert len(args.frames) in (2, 3), f"--frames 는 start end [step] 이다: {args.frames}"
    start, end = args.frames[0], args.frames[1]
    step = args.frames[2] if len(args.frames) > 2 else 1
    if args.frame_list:
        # **간격이 균일하지 않은** 프레임 집합. LBM 카메라를 재렌더할 때 필요하다 — LBM 은 창
        # 길이와 무관한 `target_frame_count` 만큼 렌더하므로(w01: 19프레임 창 → 110프레임),
        # 그걸 49로 솎으면 step 이 2..3 으로 섞인다 (`lbm_camera_to_poses.subsample`).
        # `--frames` 는 그대로 두고(로그·요약이 읽는다) 여기서만 갈아끼운다.
        frames = [int(f) for f in args.frame_list]
        assert frames == sorted(frames), f"--frame_list 는 오름차순이어야 한다: {frames[:10]}"
        step = 0                                        # 균일 간격이 아니다 — 요약에 0 으로 찍힌다
    else:
        frames = list(range(start, end + 1, step))
    assert frames, f"--frames {args.frames} 가 빈 범위다"

    # --- 카메라 ------------------------------------------------------------------------
    poses_gl, K_npz = None, None
    if args.scene_camera:
        assert args.scene_camera in bpy.data.objects, \
            f"'{args.scene_camera}' 카메라가 blend 에 없다. 후보: " \
            f"{[o.name for o in bpy.data.objects if o.type == 'CAMERA']}"
        cam_obj = bpy.data.objects[args.scene_camera]
        assert cam_obj.type == "CAMERA", f"'{args.scene_camera}' 는 CAMERA 가 아니다 ({cam_obj.type})"
        cam_source = f"scene_camera:{cam_obj.name}"
    elif args.camera_pose_pkl:
        table = load_camera_pose_pkl(args.camera_pose_pkl)
        missing = [f for f in frames if f not in table]
        assert not missing, f"camera_pose_pkl 에 없는 프레임: {missing[:10]} (키 범위 " \
                            f"{min(table)}..{max(table)})"
        poses_gl = [pkl_matrix(table[f]) for f in frames]
        cam_obj = new_camera(scene, args.lens, args.sensor)
        cam_source = f"camera_pose_pkl:{path.basename(args.camera_pose_pkl)}"
    else:
        c2w_cv, K_npz, lens_npz = load_poses_npz(args.poses)
        assert len(frames) >= len(c2w_cv), \
            f"--poses 가 {len(c2w_cv)}개인데 --frames 는 {len(frames)}개다. 프레임을 더 줄 것."
        frames = frames[:len(c2w_cv)]
        poses_gl = [Matrix([list(row) for row in mat]) @ GL2CV for mat in c2w_cv]
        cam_obj = new_camera(scene, lens_npz if lens_npz else args.lens, args.sensor)
        if K_npz is not None:
            apply_K(cam_obj.data, scene, K_npz)
        cam_source = f"poses:{path.basename(args.poses)}"
    scene.camera = cam_obj

    # --- pass index ---------------------------------------------------------------------
    objects_json = assign_pass_indices()
    with open(path.join(out, "objects.json"), "w", encoding="utf-8") as file:
        json.dump(objects_json, file, ensure_ascii=False, indent=1)

    # `scene.cycles.device = "GPU"` 만 켜면 애드온 preferences 에 디바이스가 안 잡혀 있을 때
    # 조용히 CPU 로 떨어진다. 여기서 OPTIX/CUDA 를 명시적으로 켜고 무엇이 잡혔는지 찍는다.
    rgb_cdevice = args.rgb_cdevice or args.cdevice
    if "GPU" in (args.cdevice, rgb_cdevice if args.rgb_engine == "cycles" else None):
        prefs = bpy.context.preferences.addons["cycles"].preferences
        picked = None
        for api in ("OPTIX", "CUDA"):
            prefs.compute_device_type = api
            prefs.get_devices()
            if any(dv.type == api for dv in prefs.devices):
                for dv in prefs.devices:
                    dv.use = (dv.type == api)
                picked = api
                break
        print(f"[gt] cycles device GPU/{picked or 'NONE(-> CPU fallback)'}")

    tree, layers, composite, outs = build_compositor(scene, exr_dir, want_depth, want_index)
    link_pass_sockets(tree, layers, outs)
    eevee_has_indexob = layers.outputs.get("IndexOB") is not None

    # --- 렌더 루프 -------------------------------------------------------------------------
    cameras, timings = [], []
    for order, frame in enumerate(frames):
        scene.frame_set(frame)
        if poses_gl is not None:
            cam_obj.matrix_world = poses_gl[order]
            bpy.context.view_layer.update()     # frame_set 뒤에 덮어썼으니 depsgraph 재평가
        cameras.append(camera_record(cam_obj, scene, frame, order))

        t_rgb = 0.0
        if want_rgb:
            t0 = time.time()
            for node in outs.values():
                node.mute = True                # RGB 패스는 RGB 만 — depth 는 실루엣에서 틀린다
            composite.mute = False
            scene.render.filepath = path.join(out, "rgb", f"frame_{frame:05d}")
            if args.rgb_engine == "cycles":
                # blend 는 Blender 3.3 저작인데 우리는 4.5 로 돌린다. 4.2 에서 EEVEE 가
                # EEVEE_NEXT 로 재작성되면서 3.3 기준으로 맞춰둔 `Emission Strength`(노트북
                # 20, TV 20, 조명 175~469) 가 흰 덩어리로 타서 옆 물체까지 번진다. Cycles 는
                # 같은 값을 물리적으로 처리해 번짐이 0 이다 (frame 402 실측: crop 안
                # `max>=200` 픽셀 EEVEE 401 -> Cycles 0).
                scene.render.engine = "CYCLES"
                scene.cycles.samples = args.rgb_samples
                scene.cycles.use_denoising = True
                scene.cycles.use_adaptive_sampling = True
                scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
                scene.cycles.filter_width = 1.5
                scene.cycles.max_bounces = args.rgb_bounces
                # RGB 는 128 spp 라 GPU 가 압도적이고, 아래 geometry 는 1 spp 라 CPU 가 빠르다
                # (docstring 실측 1.8 s CPU vs 2.55 s GPU). 그래서 디바이스를 패스별로 나눈다.
                scene.cycles.device = rgb_cdevice
            else:
                scene.render.engine = "BLENDER_EEVEE_NEXT"
                scene.eevee.taa_render_samples = args.samples
            bpy.ops.render.render(write_still=True)
            t_rgb = time.time() - t0

        t_cyc = 0.0
        if need_cycles:
            t0 = time.time()
            scene.render.engine = "CYCLES"
            scene.cycles.samples = 1
            scene.cycles.use_denoising = False
            scene.cycles.use_adaptive_sampling = False
            scene.cycles.pixel_filter_type = "BOX"
            scene.cycles.filter_width = 0.01    # 픽셀 중심에서만 샘플 -> 실루엣 번짐 없음
            scene.cycles.max_bounces = 0
            scene.cycles.device = args.cdevice
            link_pass_sockets(tree, layers, outs)   # 이제서야 IndexOB 소켓이 존재한다
            for key, node in outs.items():
                node.mute = False
                node.file_slots[0].path = f"{key}_{frame:05d}_"
            # Composite 를 죽이고 write_still=False 로 돌린다. File Output 노드는 그래도 쓴다
            # (probe 에서 확인) — EEVEE 가 쓴 PNG 를 Cycles 1 spp 결과로 덮어쓰면 안 된다.
            composite.mute = True
            bpy.ops.render.render(write_still=False)
            composite.mute = False
            t_cyc = time.time() - t0

        timings.append({"frame": frame, "rgb_s": t_rgb, "cycles_s": t_cyc,
                        "total_s": t_rgb + t_cyc})
        print(f"[gt] frame {frame}: rgb {t_rgb:.2f}s  cycles {t_cyc:.2f}s")

    # --- EXR -> npy -------------------------------------------------------------------------
    stats = {"depth": [], "index": []}
    index_frac = 0.0
    for frame in frames:
        for key in outs:
            hits = sorted(glob(path.join(exr_dir, f"{key}_{frame:05d}_*.exr")))
            assert hits, f"{key} EXR 이 안 나왔다 (frame {frame}) — compositor 링크를 확인할 것"
            arr = exr_to_array(hits[-1])
            if key == "index":
                index_frac = max(index_frac, float(np.abs(arr - np.rint(arr)).max()))
                arr = np.rint(arr).astype(np.uint16)
                stats["index"].append(int(arr.max()))
            else:
                arr = arr.astype(np.float32)
                valid = arr[arr < 1e9]
                stats["depth"].append((float(valid.min()) if valid.size else float("nan"),
                                       float(valid.max()) if valid.size else float("nan"),
                                       int((arr >= 1e9).sum())))
            np.save(path.join(out, key, f"frame_{frame:05d}.npy"), arr)
            if not args.keep_exr:
                for hit in hits:
                    remove(hit)
    if not args.keep_exr:
        for leftover in glob(path.join(exr_dir, "*")):
            remove(leftover)
        rmdir(exr_dir)
    assert index_frac < 1e-3, f"index pass 가 정수가 아니다 (max frac {index_frac})"

    # --- 메타 ------------------------------------------------------------------------------
    fps = scene.render.fps / scene.render.fps_base
    width, height = cameras[0]["width"], cameras[0]["height"]
    with open(path.join(out, "cameras.json"), "w", encoding="utf-8") as file:
        json.dump({"blend": bpy.data.filepath, "camera_source": cam_source,
                   "W": width, "H": height, "fps": fps, "frames": frames,
                   "convention": ("c2w_opencv = c2w_blender_gl @ diag(1,-1,-1,1). "
                                  "OpenCV 카메라는 +Z 앞, +X 오른쪽, +Y 아래. "
                                  "투영은 p = K @ (R_w2c @ Xw + t_w2c), u=p.x/p.z, v=p.y/p.z, "
                                  "픽셀 (col,row) 의 중심이 (col+0.5, row+0.5)."),
                   "cameras": cameras}, file, ensure_ascii=False, indent=1)

    def bytes_of(sub, frame, ext):
        target = path.join(out, sub, f"frame_{frame:05d}.{ext}")
        return path.getsize(target) if path.isfile(target) else 0

    meta = {
        "blend": bpy.data.filepath,
        "passes": passes,
        "camera_source": cam_source,
        "resolution": [width, height],
        "n_frames": len(frames),
        "frames": frames,
        "rgb_engine": (None if not want_rgb else
                       ("CYCLES" if args.rgb_engine == "cycles" else "BLENDER_EEVEE_NEXT")),
        "rgb_samples": (None if not want_rgb else
                        (args.rgb_samples if args.rgb_engine == "cycles" else args.samples)),
        "rgb_device": (rgb_cdevice if want_rgb and args.rgb_engine == "cycles" else None),
        "geometry_engine": "CYCLES" if need_cycles else None,
        "geometry_samples": 1 if need_cycles else None,
        "cycles_device": args.cdevice if need_cycles else None,
        "eevee_has_indexob_socket": eevee_has_indexob,
        "depth_convention": "z-planar (카메라 forward 축 투영), metre, 배경 sentinel 1e10",
        "index_dtype": "uint16",
        "index_max_abs_deviation_from_integer": index_frac,
        "timings_s": timings,
        "mean_s_per_frame": float(np.mean([t["total_s"] for t in timings])),
        "mean_s_per_frame_excl_first": (float(np.mean([t["total_s"] for t in timings[1:]]))
                                        if len(timings) > 1 else None),
        "bytes_per_frame": {
            "rgb_png": bytes_of("rgb", frames[0], "png") if want_rgb else 0,
            "depth_npy": bytes_of("depth", frames[0], "npy") if want_depth else 0,
            "index_npy": bytes_of("index", frames[0], "npy") if want_index else 0,
        },
        "keep_exr": bool(args.keep_exr),
    }
    meta["bytes_per_frame"]["total"] = sum(meta["bytes_per_frame"].values())
    with open(path.join(out, "render_meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=1)

    depth_lo = min(s[0] for s in stats["depth"]) if stats["depth"] else float("nan")
    depth_hi = max(s[1] for s in stats["depth"]) if stats["depth"] else float("nan")
    rows = {
        "blend": path.basename(bpy.data.filepath),
        "camera_source": cam_source,
        "passes": ",".join(passes),
        "resolution": f"{width}x{height} @ {fps:g} fps",
        "frames": f"{frames[0]}..{frames[-1]} "
                  f"{'step ' + str(step) if step else 'frame_list'}  (= {len(frames)})",
        "engines": (f"rgb={meta['rgb_engine']}({meta['rgb_samples']} spp"
                    + (f", {rgb_cdevice}" if meta["rgb_device"] else "") + ") "
                    if want_rgb else "rgb=off ")
                   + (f"geom=CYCLES(1 spp, {args.cdevice})" if need_cycles else "geom=off"),
        "human meshes": f"{objects_json['n_human_meshes']}개 -> index {HUMAN_INDEX} "
                        f"(armature {objects_json['human_armatures']})",
        "pass indices": f"1..{objects_json['n_indices']}",
        "depth range (m)": (f"{depth_lo:.3f} .. {depth_hi:.3f}" if stats["depth"] else "-"),
        "index max": (str(max(stats["index"])) if stats["index"] else "-"),
        "s / frame": f"{meta['mean_s_per_frame']:.2f} "
                     f"(첫 프레임 제외 {meta['mean_s_per_frame_excl_first'] or float('nan'):.2f})",
        "bytes / frame": f"{meta['bytes_per_frame']['total'] / 1e3:.1f} kB "
                         f"(rgb {meta['bytes_per_frame']['rgb_png'] / 1e3:.1f} / "
                         f"depth {meta['bytes_per_frame']['depth_npy'] / 1e3:.1f} / "
                         f"index {meta['bytes_per_frame']['index_npy'] / 1e3:.1f})",
        "out": out,
    }
    print(f"\n{'항목':22s} 값")
    for key in sorted(rows):
        print(f"{key:22s} {rows[key]}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", required=True, type=str)              # 결과 디렉토리
    parser.add_argument("--frames", required=True, nargs="+", type=int)  # start end [step], end 포함
    # 비균일 프레임 집합. 주면 `--frames` 로 만든 격자 대신 이 목록을 그대로 렌더한다.
    parser.add_argument("--frame_list", default=[], nargs="+", type=int)
    parser.add_argument("--passes", default="rgb,depth,index", type=str)  # rgb,depth,index 콤마 목록
    parser.add_argument("--poses", default="", type=str)               # npz: cam_c2w (N,4,4) OpenCV
    parser.add_argument("--scene_camera", default="", type=str)        # blend 안 기존 카메라 이름
    parser.add_argument("--camera_pose_pkl", default="", type=str)     # TRUMANS <seq>_camera_pose.pkl
    parser.add_argument("--res", default=[960, 540], nargs=2, type=int)   # 렌더 해상도 W H
    parser.add_argument("--samples", default=16, type=int)             # EEVEE RGB pass 샘플 수
    # RGB 패스 엔진. 기본은 기존과 같은 eevee 다 (하위 호환). cycles 는 3.3 저작 blend 의
    # 발광 재질이 EEVEE_NEXT 에서 타는 문제를 없앤다 — depth/index 패스는 어느 쪽이든 항상 Cycles.
    parser.add_argument("--rgb_engine", default="eevee", choices=["eevee", "cycles"], type=str)
    parser.add_argument("--rgb_samples", default=128, type=int)        # rgb_engine=cycles 일 때 spp
    parser.add_argument("--rgb_bounces", default=8, type=int)          # rgb_engine=cycles 일 때 max_bounces
    parser.add_argument("--lens", default=25.0, type=float)            # npz/pkl 카메라 초점거리 mm
    parser.add_argument("--sensor", default=36.0, type=float)          # 센서 가로 mm
    # Cycles 계산 디바이스. **`--cycles` 로 시작하는 이름 금지** (애드온 argparse 가 argv 를 통째로
    # 훑으며 prefix-match 해서 실행이 죽는다). 실측상 CPU 가 GPU 보다 빠르다.
    parser.add_argument("--cdevice", default="CPU", choices=["CPU", "GPU"], type=str)
    # RGB 패스만 다른 디바이스로. 1 spp geometry 는 CPU 가 빠르지만 128 spp RGB 는 GPU 가 빠르다.
    # None 이면 `--cdevice` 를 그대로 따른다 (기존 동작).
    parser.add_argument("--rgb_cdevice", default=None, choices=["CPU", "GPU"], type=str)
    # npy 로 바꾼 뒤 중간 EXR 을 남길지. 기본은 지운다 (프레임당 수백 kB 가 그냥 쌓인다).
    parser.add_argument("--keep_exr", dest="keep_exr", action="store_true")
    parser.add_argument("--no_keep_exr", dest="keep_exr", action="store_false")
    parser.set_defaults(keep_exr=False)
    sys.exit(main(parser.parse_args(cli_argv())))
