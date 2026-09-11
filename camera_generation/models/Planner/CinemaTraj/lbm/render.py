"""후보 카메라 pose 하나를 4D point cloud 로 렌더한다 — observation contract 의 렌더 seam.

왜 래퍼가 필요한가: Vista4D `render_frame()` 은 "이 점들을 이 카메라로 그려라"까지만 한다. 우리가
매번 하는 일은 그 앞뒤 — ① 시간 t 에서 보이는 점만 `visible[:, t]` 로 고르고 ② subject 점에
플래그를 달아 `dynamic_mask_pc` 로 subject 실루엣을 받아오고 ③ 결과에서 게이트가 먹는 스칼라
(coverage / subject 면적 / subject z-buffer 통과율)를 뽑는 것이다. 그 세 가지를 여기 모은다.

`coverage = valid_mask.mean()` 이 이 파일의 존재 이유다. 이게 하류 video model 이 채워야 할 hole
의 직접 측정치이고, CinemaTraj 계획이 SDF + Adam 손실항으로 근사하려던 바로 그 값이다. 근사할
필요가 없다 — 1 ms 면 실제로 렌더된다.

**렌더러 자기 일관성 테스트**(`--self_check`)를 먼저 통과시켜야 한다. 소스 카메라 pose 로 렌더해
원본 프레임이 안 나오면 K/c2w 규약이 어긋난 것이고, 그 위에 얹은 게이트는 전부 무의미하다.

예시 (env vista4d 필요):
    CUDA_VISIBLE_DEVICES=0 python -m lbm.render --video camel --self_check
"""
import json
from os import makedirs, path

import numpy as np
import torch

from .cloud import CINEMATRAJ_ROOT, VISTA4D_ROOT_DEFAULT, import_vista4d, load_cloud


def look_at_c2w(position, target, up):
    """OpenCV c2w 를 만든다: 열 = [right | down | forward], forward = target - position.

    roll 은 `up` 에 대해 0 이다. `up` 으로 world Y 를 넘기면 안 된다 — world 는 기울어진 frame0
    카메라라서 전 shot 에 Dutch angle 이 박힌다. scene graph 의 중력축 g 를 넘길 것.
    """
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    norm = np.linalg.norm(forward)
    assert norm > 1e-9, "카메라 위치와 look-at 타겟이 같다"
    forward /= norm

    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # 시선이 up 과 평행 (수직 부감/앙각) -> up 을 살짝 틀어 특이점 회피
        up = up + 1e-3 * np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # OpenCV 는 +Y 가 아래

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
    assert abs(np.linalg.det(c2w[:3, :3]) - 1.0) < 1e-9, "회전행렬 det != 1 (좌우 반전 규약 사고)"
    return c2w


class CloudRenderer:
    """cloud.npz 를 한 번 GPU 에 올려두고 pose 를 계속 갈아끼우며 렌더한다."""

    def __init__(self, cloud_path, vista4d_root: str = VISTA4D_ROOT_DEFAULT,
                 device: str = "cuda", dtype=torch.float32, fixed_focal: bool = False):
        """`cloud_path` 는 npz 경로(str) 또는 이미 구운 cloud dict.

        dict 를 받는 경로는 `--cloud_source memory` 용이다 — cloud.npz 가 영상당 1.4 GiB 라
        디스크를 안 거치고 tau/fit 이 그 자리에서 굽는다 (`cloud.cloud_from_recon` 설명 참조).
        """
        self.vista4d = import_vista4d(vista4d_root)
        cloud = (cloud_path if isinstance(cloud_path, dict)
                 else load_cloud(cloud_path, device=device, dtype=dtype))
        self.colors = cloud["colors"]
        self.points = cloud["points_world"]
        self.visible = cloud["visible"]
        self.indices = cloud["indices"]
        self.device, self.dtype = device, dtype

        meta = cloud["meta"]
        self.video = str(meta["video"])
        self.num_frames = int(meta["num_frames"])
        self.height, self.width = int(meta["height"]), int(meta["width"])
        self.fps = float(meta["fps"])
        self.S = float(meta["S"])
        self.z_med_frame0 = float(meta["z_med_frame0"])
        self.parallax_ratio = float(meta["parallax_ratio"])
        self.cam_c2w_src = np.asarray(meta["cam_c2w"], dtype=np.float64)  # f 4 4
        self.K_src = np.asarray(meta["K"], dtype=np.float64)  # f 3 3
        # DA3 추정 focal 은 프레임마다 흔들린다 (snowboard fx 1184.99 -> 1262.46, +6.99%).
        # 그 흔들림이 렌더에서 화면 가장자리 6.6 px/frame 의 **화각 떨림**으로 나온다 — 카메라
        # 궤적과 무관하고 `--follow_smooth` 로도 안 잡힌다. 켜면 frame0 값으로 전 프레임 고정.
        # 기본 off = 기존 동작 (소스 프레임과 화각이 정확히 맞아야 자기 일관성이 성립한다).
        self.fixed_focal = bool(fixed_focal)
        if self.fixed_focal:
            self.K_src = np.repeat(self.K_src[:1], len(self.K_src), axis=0)
        self.subject_mask = None  # (n,) bool — set_subject() 로 지정

    def visible_at(self, frame: int, temporal_persistence: bool = True):
        """시간 `frame` 에 보이는 점 마스크.

        `temporal_persistence=False` (NTP) 면 **그 프레임에서 유래한 점만** 쓴다
        (`render_video.py:61-62` 와 같은 one-hot). 자기 일관성 검사에서 이걸 쓰는 이유는
        49프레임 depth 불일치를 빼고 순수 재투영만 보기 위해서다 — camel 에서 TP 23.1 dB vs
        NTP 28.1 dB 로 5 dB 가 전부 누적분이었다.
        """
        if temporal_persistence:
            return self.visible[:, frame]
        return self.indices[:, 0].long() == frame

    def standoff(self, poses, frames=None, temporal_persistence: bool = False):
        """궤적 각 프레임에서 카메라 중심과 **그 시각 보이는 점** 사이 최소거리 (world 단위, (f,)).

        왜 렌더 depth 가 아닌가 (D47): 렌더 depth 의 하위 백분위는 방향에 의존해서 충돌 지표가
        못 된다. 실측에서 그 픽셀들의 세로 위치 median 이 0.98 이었다 — **화면 맨 아래
        가장자리**, 즉 카메라 밑을 지나가는 바닥이지 장애물이 아니다. 반대로 화면 **밖**의 가까운
        기하는 아예 못 본다 (camel `dyn_0 truck_left`: near_depth 0.687 인데 실제 3D 거리 0.090).
        과검출과 미검출을 동시에 한다. 방향에 안 걸리는 양은 3D 최소거리뿐이다.

        `temporal_persistence=False` 라 **그 프레임에서 유래한 점만** 장애물이 된다 — 동적 물체가
        자기 시각 위치로만 잡힌다. G1(`gates.behind_surface_frames`)이 궤적 프레임 하나를 샘플된
        소스 프레임 **전부**에 되쏘아 움직이는 물체를 모든 시각의 위치에서 동시에 막는 것과
        반대다. 렌더가 안 든다 (점당 곱셈 3회).

        차이의 전개 `|p−c|² = |p|² − 2 p·c + |c|²` 를 쓴다. `|p|²` 는 한 번만 계산해 캐시하고,
        마스킹은 `masked_fill(inf)` 로 한다 — `points[visible]` 인덱싱은 프레임마다 (n,3) 을
        복사해서 이분법 안에서 돌리기엔 비싸다.
        """
        if getattr(self, "_points_sq", None) is None:
            self._points_sq = (self.points * self.points).sum(dim=1)
        poses = np.asarray(poses, dtype=np.float64)
        frames = range(len(poses)) if frames is None else frames
        out = []
        for f in frames:
            center = torch.as_tensor(poses[f][:3, 3], device=self.device).to(self.dtype)
            d2 = self._points_sq - 2.0 * (self.points @ center) + float(center @ center)
            d2 = d2.masked_fill(~self.visible_at(int(f), temporal_persistence), float("inf"))
            out.append(float(d2.min()))
        return np.sqrt(np.maximum(np.asarray(out, dtype=np.float64), 0.0))

    def set_subject(self, point_mask):
        """subject 점 플래그. render() 가 이걸 `dynamic_mask` 로 넘겨 실루엣을 되받는다."""
        self.subject_mask = None if point_mask is None else torch.as_tensor(point_mask, device=self.device)

    def render(self, cam_c2w, K=None, frame: int = 0, height: int | None = None, width: int | None = None,
               temporal_persistence: bool = True, subset=None, cpu: bool = True):
        """시간 `frame` 에서 보이는 점들을 `cam_c2w` 로 렌더.

        height/width 를 줄이면 K 도 같은 배율로 줄여야 화각이 유지된다 (후보 프리뷰는 480x270).

        `subset` (n,) bool 을 주면 그 점만 그린다. G3 게이트가 subject 를 **혼자** 렌더해
        "가림이 없었다면 몇 픽셀이었을까"를 재는 데 쓴다 — 그 분모가 있어야 z-buffer 통과율이
        비율이 된다 (subject 픽셀 수를 subject 점 개수로 나누면 비율이 아니라 밀도다).
        """
        K = self.K_src[frame] if K is None else np.asarray(K, dtype=np.float64)
        height = self.height if height is None else height
        width = self.width if width is None else width
        if (height, width) != (self.height, self.width):
            K = K.copy()
            K[0] *= width / self.width
            K[1] *= height / self.height

        visible_i = self.visible_at(frame, temporal_persistence)
        if subset is not None:
            visible_i = visible_i & torch.as_tensor(subset, device=self.device)
        subject_i = self.subject_mask[visible_i] if self.subject_mask is not None else None
        to = lambda a: torch.as_tensor(np.asarray(a), device=self.device).to(self.dtype)

        rgb, depth, valid, subject_pc = self.vista4d["render_frame"](
            points_color=self.colors[visible_i],
            points_pos=self.points[visible_i],
            cam_c2w=to(cam_c2w), K=to(K),
            height=height, width=width,
            dynamic_mask=subject_i,
        )
        # Fitting only needs a few scalar metrics.  Keep the full raster on GPU in that path;
        # copying RGB/depth/masks for every bisection probe is substantially more expensive
        # than reducing them here.  cpu=True preserves the public/preview API exactly.
        if cpu:
            rgb = rgb.clamp(0, 255).to(torch.uint8).cpu().numpy()
            depth = depth.float().cpu().numpy()
            valid = valid.cpu().numpy()
            subject_pc = None if subject_pc is None else subject_pc.cpu().numpy()
        return {
            "rgb": rgb,
            "depth": depth,
            "valid": valid,
            "subject": subject_pc,
            "cam_c2w": np.asarray(cam_c2w, dtype=np.float64), "K": K,
            "frame": frame, "height": height, "width": width,
        }

    def render_metrics(self, cam_c2w, K=None, frame: int = 0, height: int | None = None,
                       width: int | None = None, temporal_persistence: bool = True,
                       near_percentile: float | None = None):
        """Render once and return fitting scalars without copying full rasters to CPU."""
        rendered = self.render(cam_c2w, K=K, frame=frame, height=height, width=width,
                               temporal_persistence=temporal_persistence, cpu=False)
        valid = rendered["valid"]
        subject = rendered["subject"]
        coverage = float(valid.float().mean().item())
        out = {"coverage": coverage, "hole_fraction": 1.0 - coverage}
        if subject is not None:
            out["subject_area"] = float(subject.float().mean().item())
            ys, xs = torch.nonzero(subject, as_tuple=True)
            if xs.numel():
                h, w = subject.shape
                out["subject_center"] = [float(xs.float().mean().item() / w),
                                         float(ys.float().mean().item() / h)]
            else:
                out["subject_center"] = None
        if near_percentile is not None:
            visible_depth = rendered["depth"][valid].float()
            out["near_depth"] = (float(torch.quantile(
                visible_depth, float(near_percentile) / 100.0).item())
                if visible_depth.numel() else float("nan"))
        return out

    def measure(self, rendered: dict, num_subject_points: int | None = None):
        """게이트가 먹는 스칼라들. coverage 는 hole_fraction 의 여집합이다."""
        valid = rendered["valid"]
        out = {"coverage": float(valid.mean()), "hole_fraction": float(1.0 - valid.mean())}
        subject = rendered["subject"]
        if subject is not None:
            out["subject_area"] = float(subject.mean())
            ys, xs = np.nonzero(subject)
            if len(xs):
                h, w = subject.shape
                out["subject_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                out["subject_center"] = [float(xs.mean() / w), float(ys.mean() / h)]
            else:
                out["subject_bbox"], out["subject_center"] = None, None
            if num_subject_points:  # z-buffer 를 통과해 실제로 그려진 subject 점의 비율
                out["subject_visible_frac"] = float(subject.sum() / max(num_subject_points, 1))
        return out


CLOUD_SOURCES = ("npz", "memory")


def add_cloud_source_args(parser, eval_data_default: str = None):
    """cloud 를 읽는 모든 스크립트 공통 `--cloud_source`. 기본 `npz` = **기존 동작 그대로**.

    `eval_data_default` 를 주면 `--eval_data` 도 같이 단다 — memory 모드는 recon 이 있어야
    굽는데, 릴 스크립트 일부(`render_preset_grid_warp.py` 등)는 원래 cloud.npz 만 읽어서
    `--eval_data` 자체가 없었다. 이미 `--eval_data` 를 가진 스크립트는 인자 없이 부른다.
    """
    parser.add_argument(
        "--cloud_source", default="npz", choices=CLOUD_SOURCES,
        help="npz = <out>/<video>/cloud.npz 를 읽는다 (기본, 기존 동작). "
             "memory = 디스크를 안 거치고 recon 에서 그 자리에 굽는다 (cloud 단계 불필요).")
    if eval_data_default is not None:
        parser.add_argument("--eval_data", default=eval_data_default, type=str,
                            help="recon_and_seg 루트 (--cloud_source memory 일 때만 쓴다)")
    return parser


def open_renderer(args, out_root: str, graph: dict = None, want_recon: bool = True,
                  fixed_focal: bool = None, video: str = None):
    """`--cloud_source` 에 따라 cloud 를 조달한다. -> (renderer, recon | None)

    왜 recon 까지 같이 돌려주나: tau(`sample_camera_bank.py`)/fit(`fit_hole_ladder.py`) 은
    렌더러를 만든 직후 `load_scene` 으로 recon 을 **어차피 한 번 더** 읽고 있었다. memory 모드는
    그 recon 을 그대로 재활용해 cloud 를 굽는다 — 그래서 in-memory 재구축의 실제 추가 비용은
    `preprocess_scene` + `unproject` ≈ 10초뿐이고, 그 대가로 cloud 단계(굽기 28s + 1.4 GiB 쓰기
    + 프로세스 기동)가 통째로 사라진다. 실측은 `cloud.cloud_from_recon` docstring 참조.
    recon 이 필요 없는 호출부(릴 렌더 대부분)는 `want_recon=False` 로 npz 모드에서 그 4.7초를
    안 낸다. memory 모드는 recon 없이는 못 구우므로 이 플래그와 무관하게 읽는다.

    `S`/`z_med`/`parallax` 는 **graph 의 `scale` 블록에서** 가져온다. 여기서 다시 재면 stride 1
    에서 6초가 더 들고, 무엇보다 게이지가 두 군데서 계산되어 갈릴 수 있다. `graph` 를 안 주면
    `<out>/<video>/scene_graph.json` 을 직접 읽는다 (tau/fit 은 이미 로드한 것을 넘겨서
    `assert_scale_mode` 검사를 거친 값을 쓴다).

    args 에서 읽는 것 중 `vista4d_root`/`device`/`fixed_focal`/`seg_root`/`seg_static_root` 는
    **없으면 기본값**으로 떨어진다 — 릴 스크립트마다 인자 집합이 달라서다. `fixed_focal` 을
    명시로 넘기면 args 보다 우선한다 (릴 스크립트 중엔 `True` 로 못 박은 곳이 있다).

    `video` 를 명시로 넘기면 `args.video` 대신 쓴다 — `eval_subject_in_frame.py` /
    `render_target_*.py` 처럼 **한 프로세스가 여러 씬을 루프로 도는** 호출부가 있어서다.
    그런 곳은 `args.video` 자체가 없다.
    """
    from scene_graph.io import load_scene                                       # noqa: PLC0415

    source = getattr(args, "cloud_source", "npz")
    assert source in CLOUD_SOURCES, f"알 수 없는 --cloud_source: {source}"
    video = getattr(args, "video", None) if video is None else video
    assert video, "video 를 못 정했다 (args.video 도 없고 인자로도 안 넘어왔다)"
    vista4d_root = getattr(args, "vista4d_root", VISTA4D_ROOT_DEFAULT)
    device = getattr(args, "device", "cuda")
    fixed_focal = getattr(args, "fixed_focal", False) if fixed_focal is None else fixed_focal

    def _load_recon():
        assert getattr(args, "eval_data", None), (
            "--eval_data 가 필요하다 (--cloud_source memory 또는 recon 을 쓰는 호출부)")
        return load_scene(args.eval_data, video, vista4d_root,
                          seg_root=getattr(args, "seg_root", None),
                          seg_static_root=getattr(args, "seg_static_root", None))

    if source == "npz":
        cloud_path = path.join(out_root, video, "cloud.npz")
        assert path.isfile(cloud_path), (
            f"cloud.npz 가 없다. 먼저 `python -m lbm.cloud --video {video}` "
            f"(또는 `--cloud_source memory`)")
        renderer = CloudRenderer(cloud_path, vista4d_root=vista4d_root,
                                 device=device, fixed_focal=fixed_focal)
        return renderer, (_load_recon() if want_recon else None)

    from .cloud import cloud_from_recon, import_vista4d as _import_vista4d      # noqa: PLC0415

    recon = _load_recon()
    if graph is None:
        with open(path.join(out_root, video, "scene_graph.json"), encoding="utf-8") as file:
            graph = json.load(file)
    scale = graph["scale"]
    cloud = cloud_from_recon(
        recon, _import_vista4d(vista4d_root),
        video=video, eval_data=args.eval_data,
        S=float(scale["S"]), z_med=float(scale["z_med_frame0"]),
        parallax=float(scale["parallax_ratio"]),
        scene_scale_mode=str(scale["mode"]), scene_scale_stride=int(scale["stride"]),
        device=device,
        preprocess=getattr(args, "preprocess", True),
        depth_outliers=getattr(args, "depth_outliers", "gaussian"),
        ignore_sky_mask=getattr(args, "ignore_sky_mask", False),
        allow_empty_dynamic_mask=getattr(args, "allow_empty_dynamic_mask", False))
    renderer = CloudRenderer(cloud, vista4d_root=vista4d_root,
                             device=device, fixed_focal=fixed_focal)
    return renderer, recon


def reprojection_residual(renderer: "CloudRenderer", frame: int, cam_c2w=None, K=None):
    """프레임 `frame` 에서 유래한 점을 그 프레임 카메라로 되쏘아 원래 픽셀로 돌아오는지 잰다.

    이게 **진짜 규약 검사**다. PSNR 은 규약을 간접적으로만 재는데, 이 렌더러는 소스 pose 에서도
    2x2 box blur 를 먹여서(`point_cloud.py:129-138`: 픽셀 중앙 점이 du=dv=0.5 -> 네 이웃에 0.25씩)
    PSNR 상한이 28 dB 근처에 묶인다. 그래서 PSNR 로 규약을 판정하면 임계를 어디에 두든 애매하다.
    반면 재투영 잔차는 splatting 과 무관한 순수 기하량이라 w2c/c2w 뒤바뀜, y 부호, ray-vs-z depth,
    cx/cy 를 전부 픽셀 단위로 즉시 드러낸다 (규약이 틀리면 수백 px 이 나온다).

    반환: (max_px, mean_px). float32 저장 때문에 완전한 0 은 안 나온다.
    """
    cam_c2w = renderer.cam_c2w_src[frame] if cam_c2w is None else np.asarray(cam_c2w, dtype=np.float64)
    K = renderer.K_src[frame] if K is None else np.asarray(K, dtype=np.float64)

    origin = renderer.indices[:, 0].long() == frame
    points = renderer.points[origin].double().cpu().numpy()  # m 3, world
    pixels = renderer.indices[origin][:, 1:].cpu().numpy()   # m 2, [h, w]
    assert len(points), f"frame {frame} 에서 유래한 점이 없다"

    w2c = np.linalg.inv(cam_c2w)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    uvz = cam @ K.T
    front = uvz[:, 2] > 1e-6
    uv = uvz[front, :2] / uvz[front, 2:3]

    # unproject 가 픽셀 중앙(+0.5)에서 쏘았으므로 돌아올 자리도 중앙이다.
    target = np.stack([pixels[front, 1] + 0.5, pixels[front, 0] + 0.5], axis=-1)
    err = np.linalg.norm(uv - target, axis=-1)
    return float(err.max()), float(err.mean())


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None):
    a, b = a.astype(np.float64), b.astype(np.float64)
    diff = (a - b) ** 2
    if mask is not None:
        diff = diff[mask]
    mse = float(diff.mean())
    return float("inf") if mse < 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)


def main(args):
    import imageio.v2 as imageio

    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    cloud_path = path.join(out_root, args.video, "cloud.npz")
    assert path.isfile(cloud_path), f"cloud.npz 가 없다. 먼저 `python -m lbm.cloud --video {args.video}`"

    renderer = CloudRenderer(cloud_path, vista4d_root=args.vista4d_root, device=args.device,
                             fixed_focal=args.fixed_focal)

    if not args.self_check:
        print(f"{renderer.video}: {renderer.points.shape[0]:,} points, "
              f"{renderer.num_frames}f {renderer.width}x{renderer.height}, "
              f"S={renderer.S:.4f} parallax={renderer.parallax_ratio:.4f}")
        return

    # 원본 프레임 (렌더 비교용). recon_and_seg 의 video.mp4 를 그대로 읽는다.
    source = np.stack(imageio.mimread(
        path.join(args.eval_data, "eval_data", "recon_and_seg", args.video, "video.mp4"),
        memtest=False)[:renderer.num_frames])
    sky = _load_sky(path.join(args.eval_data, "eval_data", "recon_and_seg", args.video, "sky_mask"))

    check_folder = path.join(out_root, args.video, "self_check")
    makedirs(check_folder, exist_ok=True)

    frames = [i for i in args.frames if i < renderer.num_frames]
    rows, fails = [], []
    for f in frames:
        max_px, mean_px = reprojection_residual(renderer, f)
        # NTP 는 그 프레임 점만 -> 순수 재투영. TP 는 49프레임 누적 -> 실제 후보 렌더와 같은 조건.
        ntp = renderer.render(renderer.cam_c2w_src[f], K=renderer.K_src[f], frame=f,
                              temporal_persistence=False)
        tp = renderer.render(renderer.cam_c2w_src[f], K=renderer.K_src[f], frame=f)
        non_sky = ~sky[f]
        # sky 는 preprocess 가 static + SKY_DEPTH 로 넣지만, 기준은 보수적으로 non-sky 로 한정한다.
        cov = float(tp["valid"][non_sky].mean())
        both_ntp, both_tp = ntp["valid"] & non_sky, tp["valid"] & non_sky
        q_ntp = psnr(ntp["rgb"][both_ntp], source[f][both_ntp])
        q_tp = psnr(tp["rgb"][both_tp], source[f][both_tp])
        rows.append((f, max_px, cov, q_ntp, q_tp))
        if max_px > args.max_reproj_px or cov <= args.min_coverage or q_ntp <= args.min_psnr_ntp:
            fails.append(f)
        imageio.imwrite(path.join(check_folder, f"{f:05d}_render.png"), tp["rgb"])
        imageio.imwrite(path.join(check_folder, f"{f:05d}_render_ntp.png"), ntp["rgb"])
        imageio.imwrite(path.join(check_folder, f"{f:05d}_source.png"), source[f])

    # 음성 대조군: y 부호를 뒤집은 c2w 로 같은 잔차를 재서 이 검사가 실제로 규약을 잡는지 확인한다.
    # 이게 작게 나오면 검사 자체가 아무것도 안 보고 있다는 뜻이라 통과보다 더 나쁘다.
    flipped = renderer.cam_c2w_src[frames[0]].copy()
    flipped[:3, 1] *= -1
    ctrl_px, _ = reprojection_residual(renderer, frames[0], cam_c2w=flipped)

    print(f"\n{'frame':>7}{'reproj_px':>12}{'cov(non-sky)':>15}{'PSNR_ntp':>11}{'PSNR_tp':>10}")
    for f, max_px, cov, q_ntp, q_tp in rows:
        print(f"{f:>7}{max_px:>12.4f}{cov:>15.4f}{q_ntp:>11.2f}{q_tp:>10.2f}")
    print(f"\n음성 대조군 (y축 반전 c2w): reproj_px = {ctrl_px:,.1f}  "
          f"({'검사 유효' if ctrl_px > 50 else '검사 무효 — 규약 오류를 못 잡는다'})")
    print(f"기준: reproj_px < {args.max_reproj_px}, cov(non-sky) > {args.min_coverage}, "
          f"PSNR_ntp > {args.min_psnr_ntp} dB")
    print("참고: PSNR 은 규약이 아니라 데이터 품질을 잰다. PSNR_ntp 상한은 소스 pose 에서도 걸리는 "
          "2x2 box blur 가 정하는데 고주파 텍스처일수록 손해라 내용 의존적이다 "
          "(camel 28 dB / avocado-slice 40 dB). PSNR_tp 는 거기에 49프레임 depth 불일치가 더해진다.")
    print(f"결과: {'PASS' if not fails else f'FAIL at frames {fails}'}   -> {check_folder}")
    assert ctrl_px > 50, "음성 대조군이 통과했다 — 재투영 검사가 규약 오류를 못 잡고 있다."
    assert not fails, "렌더러 자기 일관성 실패 — K/c2w 규약이 어긋났다. 하류 게이트 전부 무효."


def _load_sky(folder: str):
    import cv2
    from os import listdir
    files = sorted(f for f in listdir(folder) if f.endswith(".png"))
    return np.stack([cv2.imread(path.join(folder, f), cv2.IMREAD_GRAYSCALE) > 127 for f in files])


if __name__ == "__main__":
    from argparse import ArgumentParser

    from .cloud import EVAL_DATA_DEFAULT

    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)

    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--self_check", action="store_true", default=False)
    # frame0 K 로 전 프레임 고정 (DA3 focal 드리프트 제거). 기본 off = 기존 동작.
    parser.add_argument("--fixed_focal", action="store_true", default=False)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    parser.add_argument("--frames", nargs="*", type=int, default=[0, 12, 24, 36, 48])
    parser.add_argument("--min_coverage", default=0.98, type=float)  # non-sky 픽셀 기준
    # 규약 판정은 재투영 잔차가 한다. 규약이 틀리면 수백 px 이라 0.05 는 float32 저장 오차만 허용.
    parser.add_argument("--max_reproj_px", default=0.05, type=float)
    # NTP PSNR 은 grossly 깨진 렌더를 거르는 보조 기준. 2x2 box blur 상한이 ~28 dB 라 30 은 불가능.
    parser.add_argument("--min_psnr_ntp", default=26.0, type=float)

    main(parser.parse_args())
