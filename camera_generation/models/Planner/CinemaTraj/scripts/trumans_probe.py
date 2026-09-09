"""TRUMANS 배포본 훑기 — LBM-Lite 를 태우기 전에 **무엇이 있고 무엇이 없는지** 확정한다.

왜 필요한가: 메모 `trumans-has-no-camera-params` 는 "매니페스트에 intrinsics/extrinsics 가
없다"였다. 그런데 배포본에는 `smplx_result`(world)와 `smplx_result_in_cam`(camera) 이 **쌍으로**
있다. 같은 사람의 같은 body_pose 를 두 좌표계에서 준 것이므로, 두 root 변환의 차이가 곧
카메라 extrinsic 이다 — 즉 카메라는 명시적으로는 없지만 **복원 가능**하다.

SMPL-X 는 root 회전을 rest-pose pelvis `J0` 를 축으로 걸고 그 뒤에 `transl` 을 더한다:

    v(R, t) = R (X - J0) + J0 + t          X = 자세만 적용된 canonical 정점

world->cam 을 (R_cw, t_cw) 라 하면 v_cam = R_cw v_world + t_cw 에서

    R_c = R_cw R_w                                   (root 회전끼리의 관계)
    t_cw = J0 + t_c - R_cw (J0 + t_w)                (나머지)

`R_cw` 는 프레임마다 독립으로 나오므로, **프레임 간 편차가 곧 카메라가 움직였는지**의 측정치다.
이 스크립트는 그 편차를 재서 "정지 카메라 데이터셋인지"를 판정한다 (정지면 LBM-Lite 의
`view_angle`/`τ` 축이 통째로 무의미해지므로 반드시 먼저 알아야 한다).

env: `vista4d`

예시:
    python scripts/trumans_probe.py
    python scripts/trumans_probe.py --recordings 2023-01-14@22-06-10 --dump_frames /tmp/tru
"""
import pickle
import sys
from argparse import ArgumentParser
from os import listdir, makedirs, path

import numpy as np

TRUMANS_ROOT_DEFAULT = "/data1/cympyc1785/data/trumans/Data_release"


def rodrigues(vectors: np.ndarray):
    """(N,3) axis-angle → (N,3,3). cv2 없이 — cv2.Rodrigues 는 한 번에 하나씩만 받는다."""
    vectors = np.asarray(vectors, dtype=np.float64)
    theta = np.linalg.norm(vectors, axis=-1, keepdims=True)
    axis = np.divide(vectors, np.where(theta > 1e-12, theta, 1.0))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(*x.shape, 3, 3)
    eye = np.broadcast_to(np.eye(3), skew.shape)
    sin = np.sin(theta)[..., None]
    cos = np.cos(theta)[..., None]
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def geodesic_deg(a: np.ndarray, b: np.ndarray):
    """두 회전 사이 각도(도). (N,3,3) x (3,3) 브로드캐스트."""
    rel = np.transpose(a, (0, 2, 1)) @ b if a.ndim == 3 else a.T @ b
    trace = np.trace(rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def load_pair(root: str, name: str):
    """(world dict, cam dict). 파일명 규약은 `<recording>_smplx_results.pkl`."""
    out = []
    for sub in ("smplx_result", "smplx_result_in_cam"):
        with open(path.join(root, sub, f"{name}_smplx_results.pkl"), "rb") as file:
            out.append(pickle.load(file))
    return out[0], out[1]


def camera_rotation(world: dict, cam: dict):
    """프레임별 R_cw = R_c R_w^T. 카메라가 정지면 전 프레임이 같아야 한다."""
    r_w = rodrigues(np.asarray(world["global_orient"]))
    r_c = rodrigues(np.asarray(cam["global_orient"]))
    return r_c @ np.transpose(r_w, (0, 2, 1))


def main(args):
    root = args.trumans_root
    names = args.recordings
    if not names:
        pool = sorted(f[: -len("_smplx_results.pkl")]
                      for f in listdir(path.join(root, "smplx_result"))
                      if f.endswith("_smplx_results.pkl"))
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(len(pool), size=min(args.num_recordings, len(pool)), replace=False)
        names = [pool[i] for i in sorted(picks)]

    print(f"{'recording':<24}{'frames':>8}{'gender':>8}{'R_cw 편차 deg':>16}"
          f"{'|t_c| med':>11}{'z_c med':>9}{'z_c min':>9}")
    print("-" * 85)
    rows = []
    for name in names:
        world, cam = load_pair(root, name)
        assert np.allclose(np.asarray(world["body_pose"]), np.asarray(cam["body_pose"])), \
            f"{name}: body_pose 가 두 좌표계에서 다르다 — 같은 시퀀스가 아니다"
        r_cw = camera_rotation(world, cam)
        drift = geodesic_deg(r_cw, r_cw[0])
        t_c = np.asarray(cam["transl"], dtype=np.float64)
        rows.append((name, len(t_c), str(world["gender"]), float(drift.max()),
                     float(np.median(np.linalg.norm(t_c, axis=1))),
                     float(np.median(t_c[:, 2])), float(t_c[:, 2].min())))
        print(f"{rows[-1][0]:<24}{rows[-1][1]:>8}{rows[-1][2]:>8}{rows[-1][3]:>16.4f}"
              f"{rows[-1][4]:>11.3f}{rows[-1][5]:>9.3f}{rows[-1][6]:>9.3f}")

    drifts = np.array([r[3] for r in rows])
    print(f"\nR_cw 프레임간 편차: max {drifts.max():.4f}deg  median {np.median(drifts):.4f}deg")
    print("→ 편차가 ~0 이면 **녹화당 카메라 1개, 완전 정지**다.")

    if args.dump_frames:
        import imageio.v2 as imageio
        makedirs(args.dump_frames, exist_ok=True)
        for name in names[: args.dump_max]:
            video_path = path.join(root, "video_render", f"{name}.pkl.mp4")
            if not path.isfile(video_path):
                print(f"[skip] 렌더 mp4 없음: {video_path}")
                continue
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            frames = []
            for index, frame in enumerate(reader):
                if index % args.dump_stride == 0:
                    frames.append(frame)
                if len(frames) >= args.dump_count:
                    break
            reader.close()
            for index, frame in enumerate(frames):
                imageio.imwrite(path.join(args.dump_frames,
                                          f"{name}_f{index * args.dump_stride:04d}.png"), frame)
            print(f"{name}: {meta.get('size')} fps {meta.get('fps')} "
                  f"→ {len(frames)}장 {args.dump_frames}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trumans_root", default=TRUMANS_ROOT_DEFAULT, type=str)
    parser.add_argument("--recordings", nargs="*", default=None)   # None = 무작위 표본
    parser.add_argument("--num_recordings", default=12, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--dump_frames", default=None, type=str)   # 프레임 PNG 를 떨굴 폴더
    parser.add_argument("--dump_count", default=4, type=int)
    parser.add_argument("--dump_stride", default=100, type=int)
    parser.add_argument("--dump_max", default=3, type=int)         # 몇 편에서 뽑을지
    sys.exit(main(parser.parse_args()))
