"""소스 영상 1편(RGBD + 카메라 + 마스크)을 4D point cloud 로 올리고 캐시한다.

왜 이게 필요한가: look-before-move 의 전제는 "후보 카메라를 **먼저 렌더해서** VLM 에게 보여준다"
는 것이다. 그 렌더러가 Vista4D 의 `utils/point_cloud/point_cloud.py` 이고, 그게 먹는 입력이 바로
`unproject()` 가 뱉는 (colors, points_world, visible, indices) 4-tuple 이다. 후보를 수십~수백 개
찍어보는 동안 매번 unproject 를 다시 돌 이유가 없으므로 여기서 한 번 만들고 npz 로 캐시한다.

`visible (n, f)` 가 이 자료구조의 핵심이다 — 정적 점은 전 프레임에서 True, 동적 점은 자기 프레임
하나에서만 True 라 `visible.sum(1) == 1` 이 동적 점 판별식이 된다 (`point_cloud.py:281` 이 쓰는
그 식). 즉 3D 가 아니라 4D 다.

입력  : `<eval_data>/eval_data/recon_and_seg/<video>/` (video.mp4 / depths / cameras.npz /
        dynamic_mask / sky_mask)
        `<eval_data>/eval_data/seg_instances/<video>/` (선택: SAM3 per-instance track)
출력  : `<out>/<video>/cloud.npz`  (format `lbm_cloud_v1`)

예시 (env vista4d 필요):
    CUDA_VISIBLE_DEVICES=0 python -m lbm.cloud --video camel
    CUDA_VISIBLE_DEVICES=0 python -m lbm.cloud --video camel --subject_track 1 --no_skip_done
"""
import sys
from os import makedirs, path

import numpy as np
import torch

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "..", "..", "video_generation", "models", "Vista4D"))
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"

CLOUD_FORMAT = "lbm_cloud_v1"


def import_vista4d(vista4d_root: str):
    """Vista4D 는 패키지가 아니라 `utils.*` 로 절대 import 하는 스크립트 트리라 sys.path 에 얹는다."""
    if vista4d_root not in sys.path:
        sys.path.insert(0, vista4d_root)
    from utils.media import intrinsics_to_K, load_recon_and_seg
    from utils.point_cloud.point_cloud import render, render_frame, unproject
    from utils.point_cloud.preprocess import SKY_DEPTH, preprocess_scene
    return {
        "load_recon_and_seg": load_recon_and_seg, "intrinsics_to_K": intrinsics_to_K,
        "unproject": unproject, "render_frame": render_frame, "render": render,
        "preprocess_scene": preprocess_scene, "SKY_DEPTH": SKY_DEPTH,
    }


# S 의 정의는 `scene_graph/scale.py` 한 곳뿐이다. 예전엔 여기에 복사본이 있었는데, 정의가 두
# 군데면 한쪽만 고쳐도 아무 에러 없이 게이지가 갈린다 (2026-09-02 정의 변경 때 실제로 위험했다).
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)
from scene_graph.scale import SCALE_MODES, scene_scale                          # noqa: E402,F401


def parallax_ratio(cam_c2w: np.ndarray, z_med: float):
    """max_ij‖C_i − C_j‖ / z_med. 0.02 근처면 single-view depth shell 이라 보면 된다."""
    centers = cam_c2w[:, :3, 3]
    spread = float(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max())
    return spread / max(z_med, 1e-9)


def preprocess_recon(recon: dict, vista4d, depth_outliers: str = "gaussian", ignore_sky_mask: bool = False):
    """Vista4D 표준 전처리(`preprocess_scene`)를 unproject 앞에 끼운다. **이걸 건너뛰면 안 된다.**

    처음엔 `load_recon_and_seg` 결과를 `unproject` 에 곧장 넣었다가 소스 pose 재렌더 PSNR 이
    20 dB 에서 멈췄다. 배경은 완벽했고 동적 물체 주변만 반투명 빗살로 뭉개졌다. 원인은 규약이
    아니라 **static 점의 시간 지속성**이다: `static_mask = ~dynamic_mask` 는 마스크 경계에서
    프레임당 0.04% 쯤 물체 표면을 흘리는데, static 점은 `visible` 이 전 프레임 True 라
    (`point_cloud.py:63`) 그 누수가 49프레임어치 한꺼번에 렌더된다. 게다가 유령은 다른 프레임의
    물체 깊이에 앉아 있어 진짜 물체와 z 가 거의 같고, `render_frame` 의 z-tolerance 가 log1p
    공간 0.02 라 가려지는 대신 **블렌딩**된다. 그래서 빗살이 된다.

    `preprocess_scene` 이 정확히 이걸 막는다: static mask 를 erode(6px x3) 해서 경계 누수를
    잘라내고, depth outlier 를 지우고, sky 를 static 에 합치며 depth 를 SKY_DEPTH 로 박는다.
    해상도는 native 를 넘겨 resize 를 no-op 으로 만든다 (`media.py:resize_2d` 가 동일 해상도면
    바로 반환).

    sky 처리의 부수효과 하나: sky 가 static 점이 되므로 렌더에서 hole 이 아니라 먼 배경으로
    채워진다. G2 coverage 가 하늘 때문에 깎이지 않게 되는 건 의도된 것이다 (저자 기본 동작).
    """
    height, width = recon["video"].shape[1:3]
    processed = vista4d["preprocess_scene"](
        recon, indices=np.arange(recon["video"].shape[0]), height=height, width=width,
        depth_outliers=depth_outliers, ignore_sky_mask=ignore_sky_mask,
    )
    processed["fps"] = recon["fps"]
    return processed


def build_cloud(recon: dict, vista4d, device: str = "cuda", dtype=torch.float32):
    """recon_and_seg dict → (colors, points_world, visible, indices) torch tensor 들.

    static_mask 를 넘겨야 정적 점이 전 프레임 visible 로 잡힌다 (temporal persistence). 안 넘기면
    모든 점이 자기 프레임에서만 보이게 되어 후보 렌더가 프레임 하나 분량의 점만 쓰고 텅 빈다.
    """
    K = vista4d["intrinsics_to_K"](recon["intrinsics"])

    to = lambda a, d=dtype: torch.as_tensor(a, device=device).to(d)
    colors, points_world, visible, indices = vista4d["unproject"](
        video=torch.as_tensor(recon["video"], device=device),  # f h w 3, uint8 그대로
        depths=to(recon["depths"]),
        cam_c2w=to(recon["cam_c2w"]),
        K=to(K),
        dynamic_mask=to(recon["dynamic_mask"], torch.bool),
        static_mask=to(recon["static_mask"], torch.bool),
    )
    return colors, points_world, visible, indices, K


def subject_point_mask(indices: torch.Tensor, track_masks: np.ndarray):
    """`indices` (n,3)=[f,h,w] 를 (f,h,w) 마스크로 조회해 subject 점 인덱스를 만든다.

    track_masks 는 (f, h, w) bool — SAM3 track 하나의 per-frame 마스크. unproject 가 픽셀 순서를
    보존하지 않으므로(mask_flat 으로 걸러냄) 좌표로 되짚는 이 방법 말고는 대응이 안 붙는다.
    """
    lut = torch.as_tensor(track_masks, device=indices.device)  # f h w, bool
    f, h, w = indices[:, 0].long(), indices[:, 1].long(), indices[:, 2].long()
    return lut[f, h, w]


def save_cloud(output_path: str, colors, points_world, visible, indices, meta: dict):
    """visible (n, f) bool 은 그대로 저장하면 n*f 바이트라 packbits 로 8배 줄인다."""
    makedirs(path.dirname(output_path), exist_ok=True)
    visible_np = visible.cpu().numpy()
    np.savez(
        output_path,
        format=CLOUD_FORMAT,
        colors=colors.cpu().numpy().astype(np.uint8),
        points_world=points_world.cpu().numpy().astype(np.float32),
        visible_packed=np.packbits(visible_np, axis=1),
        visible_num_frames=np.int32(visible_np.shape[1]),
        indices=indices.cpu().numpy().astype(np.int32),
        **{f"meta_{k}": v for k, v in meta.items()},
    )


def load_cloud(input_path: str, device: str = "cuda", dtype=torch.float32):
    data = np.load(input_path, allow_pickle=False)
    assert str(data["format"]) == CLOUD_FORMAT, f"알 수 없는 cloud format: {data['format']}"
    num_frames = int(data["visible_num_frames"])
    visible = np.unpackbits(data["visible_packed"], axis=1, count=num_frames).astype(bool)
    return {
        "colors": torch.as_tensor(data["colors"], device=device).to(dtype),
        "points_world": torch.as_tensor(data["points_world"], device=device).to(dtype),
        "visible": torch.as_tensor(visible, device=device),
        "indices": torch.as_tensor(data["indices"], device=device),
        "meta": {k[5:]: data[k] for k in data.files if k.startswith("meta_")},
    }


def main(args):
    vista4d = import_vista4d(args.vista4d_root)

    recon_folder = path.join(args.eval_data, "eval_data", "recon_and_seg", args.video)
    output_path = path.join(args.output_root or path.join(CINEMATRAJ_ROOT, "out"), args.video, "cloud.npz")
    if path.isfile(output_path) and args.skip_done:
        print(f"skip (done): {output_path}")
        return

    recon = vista4d["load_recon_and_seg"](recon_folder)
    num_frames, height, width, _ = recon["video"].shape
    K = vista4d["intrinsics_to_K"](recon["intrinsics"])

    # 좌표 규약 검사 — 여기서 안 걸리면 하류 게이트가 전부 조용히 틀린다.
    # frame0 은 I 가 **아니다**: 저자 배포본 recon_and_seg 는 우리 `recon_da3.py:43` 의 frame0
    # 앵커를 거치지 않았다 (camel 은 frame0 이 회전 6.0deg / 이동 0.0086 만큼 떠 있다). world 는
    # "cameras.npz 가 말하는 그 좌표계"로 정의하고, frame0 오프셋은 meta 에 기록만 한다.
    # emit 의 rel[0]=I 는 `inv(P[0]) @ P` 라 world 원점이 어디든 성립하므로 문제되지 않는다.
    frame0_rot_deg = float(np.degrees(np.arccos(
        np.clip((np.trace(recon["cam_c2w"][0][:3, :3]) - 1) / 2, -1, 1))))
    frame0_offset = float(np.linalg.norm(recon["cam_c2w"][0][:3, 3]))
    assert abs(K[0, 0, 2] / width - 0.5) < 0.05 and abs(K[0, 1, 2] / height - 0.5) < 0.05,\
        f"principal point 가 중앙이 아니다: cx={K[0, 0, 2]} cy={K[0, 1, 2]} (W={width} H={height})"
    finite = np.isfinite(recon["depths"][0]) & (recon["depths"][0] > 0)
    non_sky = ~recon["sky_mask"][0]
    assert (finite & non_sky).sum() / max(non_sky.sum(), 1) > 0.95, "non-sky depth 유효율 < 0.95"

    # ── §empty_dynmask (D176, 2026-09-10) ────────────────────────────────────────────────
    # `dynpose_ingest.py:177` 의 recon 단계는 `seg_keywords=[]` 라 **all-zero placeholder**
    # dynamic_mask 를 png 로 써 두고, 나중에 `dynmask` 단계가 SAM3 합집합으로 덮는다. 그
    # 사이에 cloud 를 구우면 `load_recon_and_seg` 가 `static_mask = ~dynamic_mask` 로 **전
    # 픽셀을 정적**으로 만들고, 정적 점은 `visible` 이 전 프레임 True 라 동적 물체가 49개
    # 사본으로 잔류한다 — 에러 없이, `visible.sum(1)==1` 이 정확히 0 으로.
    # `load_masks` 는 png 가 **존재하는지**만 보므로(media.py:148) 새까만 png 는 통과한다.
    # 실측 (dynpose 표본 120편): 47편(39%)이 이 상태였고 `hole_fraction` med 0.199 vs 0.279,
    # `subject_visible_frac<0.6` 36.8% vs 25.5% 로 편향됐다. bake 시각으로는 못 가른다
    # (정상인 씬의 cloud 가 오염된 씬보다 오래된 경우가 있다).
    dyn_frac = float(recon["dynamic_mask"].mean())
    if dyn_frac == 0.0 and not args.allow_empty_dynamic_mask:
        seg_npz = path.join(args.eval_data, "eval_data", "seg_instances", args.video, "masks.npz")
        raise AssertionError(
            f"{args.video}: dynamic_mask 가 전부 0 이다 — cloud 를 구우면 전 픽셀이 정적이 되어 "
            f"동적 물체가 49개 사본으로 남는다.\n"
            f"  seg_instances masks.npz {'있음' if path.isfile(seg_npz) else '없음'}: {seg_npz}\n"
            f"  있으면 `scripts/dynpose_dynamic_mask_from_seg.py` (dynmask 단계) 를 먼저 돌릴 것.\n"
            f"  정말 동적 물체가 없는 씬이면 `--allow_empty_dynamic_mask` 로 통과시킨다.")

    # S / z_med 는 **전처리 전 raw depth** 로 잰다. preprocess 가 sky depth 를 SKY_DEPTH(1e3) 로
    # 덮어쓰기 때문에 순서를 바꾸면 게이지가 통째로 망가진다 (non-sky 로 걸러도 습관적으로 위험).
    S = scene_scale(recon["depths"], K, recon["sky_mask"], cam_c2w=recon["cam_c2w"],
                    mode=args.scene_scale_mode, stride=args.scene_scale_stride)
    z_med = float(np.median(recon["depths"][0][finite & non_sky]))
    plx = parallax_ratio(recon["cam_c2w"], z_med)

    scene = preprocess_recon(recon, vista4d, depth_outliers=args.depth_outliers,
                             ignore_sky_mask=args.ignore_sky_mask) if args.preprocess else recon
    colors, points_world, visible, indices, _ = build_cloud(scene, vista4d, device=args.device)
    num_dynamic = int((visible.sum(dim=1) == 1).sum())

    save_cloud(output_path, colors, points_world, visible, indices, meta={
        "video": args.video, "num_frames": num_frames, "height": height, "width": width,
        "fps": float(recon["fps"]), "S": S, "z_med_frame0": z_med, "parallax_ratio": plx,
        # S 의 정의를 캐시에 같이 싣는다 — 안 싣으면 옛 게이지로 구운 `cloud.npz` 를 새 코드가
        # 그대로 집어 쓰면서 아무 에러 없이 단위만 갈린다 (2026-09-02 정의 변경).
        "scene_scale_mode": args.scene_scale_mode, "scene_scale_stride": args.scene_scale_stride,
        "frame0_rot_deg": frame0_rot_deg, "frame0_offset": frame0_offset,
        "preprocess": bool(args.preprocess), "depth_outliers": args.depth_outliers,
        "ignore_sky_mask": bool(args.ignore_sky_mask),
        # D176. 굽는 시점의 마스크 상태를 캐시에 박는다 — 이게 없으면 "동적 점 0개"가 마스크가
        # 비어서인지 정말 정적인 씬인지 npz 를 열어봐도 못 가른다 (§empty_dynmask).
        "dynamic_mask_frac": dyn_frac, "num_dynamic": int(num_dynamic),
        "cam_c2w": recon["cam_c2w"].astype(np.float32), "K": K.astype(np.float32),
    })

    print(f"\n{'video':<18}{'points':>12}{'dynamic':>12}{'S':>10}{'z_med':>9}{'parallax':>10}{'f0_rot':>9}")
    print(f"{args.video:<18}{points_world.shape[0]:>12,}{num_dynamic:>12,}"
          f"{S:>10.4f}{z_med:>9.3f}{plx:>10.4f}{frame0_rot_deg:>9.2f}")
    print(f"-> {output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)  # None = <CinemaTraj>/out
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)

    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    # 전처리 없이 unproject 하면 static 경계 누수가 49프레임 쌓여 동적 물체가 빗살로 뭉개진다.
    # `--no_preprocess` 는 그 raw 경로를 재현해보기 위한 것이지 실사용 옵션이 아니다.
    parser.add_argument("--preprocess", action="store_true", default=True)
    parser.add_argument("--no_preprocess", dest="preprocess", action="store_false")
    parser.add_argument("--depth_outliers", default="gaussian", choices=["gaussian", "pool"])
    parser.add_argument("--ignore_sky_mask", action="store_true", default=False)

    # 빈 dynamic_mask 로 굽는 것은 **기본적으로 사고**다 (§empty_dynmask). 동적 물체가 정말
    # 없는 씬을 굽고 싶을 때만 켠다 — 켜면 meta 의 `dynamic_mask_frac` 이 0 으로 남아 나중에
    # 그 씬들만 골라낼 수 있다.
    parser.add_argument("--allow_empty_dynamic_mask", action="store_true", default=False)

    # S 의 정의. `build_scene_graph.py` 와 **같은 값이어야 한다** — 두 산출물(`scene_graph.json`,
    # `cloud.npz`)이 서로 다른 게이지를 들고 있으면 게이트 마진 `0.02·S` 와 후보 거리가 어긋난다.
    parser.add_argument("--scene_scale_mode", default="points_first_cam", choices=SCALE_MODES)
    parser.add_argument("--scene_scale_stride", default=1, type=int)   # 픽셀 서브샘플 (1 = 전부)

    main(parser.parse_args())
