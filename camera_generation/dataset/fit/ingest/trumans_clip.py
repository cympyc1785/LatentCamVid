"""TRUMANS 녹화 → LBM-Lite 가 먹는 49프레임 소스 클립 하나를 잘라낸다.

왜 필요한가: 코퍼스 51편은 24 fps 짜리 **49프레임 완제품**인데 TRUMANS 는 30 fps 로
1,500~3,400 프레임짜리 연속 녹화다. 어디를 자르느냐가 그대로 씬이 되므로 손으로 고르면
재현이 안 된다. 이 스크립트가 창(window)을 **점수로** 고르고, 그 창의 action label 과
카메라 회전량을 같이 찍어서 `metadata.csv` 행을 사람이 바로 쓸 수 있게 한다.

창 점수 = ① 카메라 회전 폭(deg) ② 사람 이동 거리(m) ③ action label 종류 수.
셋 다 큰 창이 "카메라도 돌고 사람도 움직이는" 구간이다 — 정지 구간을 잘라오면 시차가 0 이라
`build_scene_graph.py` 의 OBB 깊이축이 무너진다 (`extent_inflated` 로 표시되긴 하지만
애초에 안 뽑는 게 낫다).

카메라는 배포본에 없고 `smplx_result`(world) / `smplx_result_in_cam`(camera) 쌍에서 복원한다 —
근거와 유도는 `trumans_probe.py` 참고. 이 카메라는 **사람을 반경 ~2.1 m 로 따라다니는 가상
카메라**라 씬 고정 카메라가 아니다. 그래서 여기서는 pose 를 쓰지 않고 "얼마나 돌았나"라는
창 선택 지표로만 쓴다. 실제 카메라/깊이는 아래처럼 `recon_and_seg_single.py --recon_method da3`
가 렌더 mp4 로부터 다시 추정한다 (코퍼스 51편과 같은 게이지를 쓰기 위해서다).

env: `vista4d`

예시:
    python fit/ingest/trumans_clip.py --list_meta
    python fit/ingest/trumans_clip.py --recording 2023-01-14@22-33-09 --rank
    python fit/ingest/trumans_clip.py --recording 2023-01-14@22-33-09 --start 431 \
        --out /tmp/trumans_bedroom_w431.mp4
    python fit/ingest/trumans_clip.py --identify /tmp/trumans_bedroom_w431.mp4
"""
import sys
from argparse import ArgumentParser
from os import listdir, path

import numpy as np

sys.path.insert(0, path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from fit.ingest.trumans_probe import TRUMANS_ROOT_DEFAULT, camera_rotation, geodesic_deg, load_pair


def recording_names(root: str):
    return sorted(f[: -len("_smplx_results.pkl")]
                  for f in listdir(path.join(root, "smplx_result"))
                  if f.endswith("_smplx_results.pkl"))


def action_names(root: str):
    """meta.npy 의 action_label 이름 목록. 없으면 인덱스 문자열로 대체한다."""
    meta = np.load(path.join(root, "meta.npy"), allow_pickle=True).item()
    names = meta.get("action_label")
    return [str(n) for n in names] if names is not None else None


def recording_slice(root: str, name: str):
    """전역 `seg_name` 배열에서 이 녹화가 차지하는 [start, stop) 구간."""
    seg = np.load(path.join(root, "seg_name.npy"), allow_pickle=True)
    hit = np.flatnonzero(seg == name)
    assert hit.size, f"seg_name.npy 에 없는 녹화: {name}"
    assert hit[-1] - hit[0] + 1 == hit.size, f"{name}: seg_name 구간이 연속이 아니다"
    return int(hit[0]), int(hit[-1]) + 1


def window_table(root: str, name: str, num_frames: int, stride: int, step: int):
    """창별 (시작, 카메라 회전 deg, 사람 이동 m, action 종류 수). 점수 내림차순."""
    world, cam = load_pair(root, name)
    r_cw = camera_rotation(world, cam)
    transl = np.asarray(world["transl"], dtype=np.float64)
    start, stop = recording_slice(root, name)
    labels = np.load(path.join(root, "action_label.npy"), mmap_mode="r")[start:stop]

    span = (num_frames - 1) * stride + 1
    total = min(len(r_cw), len(labels))
    rows = []
    for begin in range(0, max(total - span, 0) + 1, step):
        index = np.arange(begin, begin + span, stride)
        rot = float(geodesic_deg(r_cw[index], r_cw[begin]).max())
        move = float(np.linalg.norm(np.diff(transl[index], axis=0), axis=1).sum())
        acts = int((np.asarray(labels[index]) > 0.5).any(axis=0).sum())
        rows.append((begin, rot, move, acts))
    assert rows, f"{name}: {span} 프레임짜리 창이 안 나온다 (총 {total} 프레임)"
    order = np.argsort([-(r[1] / 90.0 + r[2] + r[3]) for r in rows])
    return [rows[i] for i in order], total


def window_actions(root: str, name: str, begin: int, span: int, stride: int):
    """창 안에서 켜진 action label 이름들."""
    start, stop = recording_slice(root, name)
    labels = np.asarray(np.load(path.join(root, "action_label.npy"),
                                mmap_mode="r")[start:stop][begin: begin + span: stride])
    on = np.flatnonzero((labels > 0.5).any(axis=0))
    names = action_names(root)
    return [names[i] if names and i < len(names) else f"action_{i}" for i in on]


def cut_clip(root: str, name: str, begin: int, num_frames: int, stride: int, out_path: str):
    """`video_render/<name>.pkl.mp4` 에서 창을 잘라 mp4 로. cv2 mp4v 금지 — libx264."""
    import imageio.v2 as imageio
    source = path.join(root, "video_render", f"{name}.pkl.mp4")
    assert path.isfile(source), f"렌더 mp4 가 없다: {source}"
    reader = imageio.get_reader(source)
    fps = float(reader.get_meta_data().get("fps", 30.0))
    want = set(range(begin, begin + (num_frames - 1) * stride + 1, stride))
    frames = []
    for index, frame in enumerate(reader):
        if index in want:
            frames.append(frame)
        if index >= max(want):
            break
    reader.close()
    assert len(frames) == num_frames, f"{len(frames)}/{num_frames} 프레임만 읽혔다 — start 가 너무 뒤다"
    imageio.mimwrite(out_path, frames, fps=fps / stride,
                     codec="libx264", quality=6, macro_block_size=1)
    return out_path, fps, frames[0].shape


def identify(root: str, clip_path: str, top: int = 5):
    """이미 잘라둔 클립이 어느 녹화의 몇 번째 프레임인지 되찾는다 (provenance 복구용).

    임계값으로 조기 종료하지 않고 **전 녹화의 최소 오차**를 모아 상위 몇 개를 돌려준다.
    클립이 libx264 로 재인코딩됐고 해상도도 다를 수 있어서, 고정 임계는 "아무것도 못 찾음"
    으로 조용히 실패한다 (실제로 그렇게 한 번 실패했다).
    """
    import imageio.v2 as imageio
    reader = imageio.get_reader(clip_path)
    probe = np.asarray(next(iter(reader)), dtype=np.float64).mean(axis=2)
    reader.close()

    def descriptor(frame):
        """해상도 무관 8x8 밝기 격자 — 리사이즈/재인코딩에 둔감하다."""
        gray = np.asarray(frame, dtype=np.float64).mean(axis=2)
        rows = np.array_split(gray, 8, axis=0)
        grid = [np.array_split(r, 8, axis=1) for r in rows]
        return np.array([[c.mean() for c in row] for row in grid]).ravel()

    target = descriptor(np.repeat(probe[..., None], 3, axis=2))
    best = []
    for name in recording_names(root):
        source = path.join(root, "video_render", f"{name}.pkl.mp4")
        if not path.isfile(source):
            continue
        reader = imageio.get_reader(source)
        hit = (10 ** 9, -1)
        for index, frame in enumerate(reader):
            diff = float(np.abs(descriptor(frame) - target).mean())
            hit = min(hit, (diff, index))
        reader.close()
        best.append((name, hit[1], hit[0]))
    best.sort(key=lambda r: r[2])
    return best[:top]


def main(args):
    root = args.trumans_root
    if args.list_meta:
        names = action_names(root)
        print(f"녹화 {len(recording_names(root))}편")
        print(f"action_label {len(names) if names else '?'}종: {names}")
        return 0

    if args.identify:
        for name, index, diff in identify(root, args.identify):
            print(f"{name}  frame {index}  meanabs {diff:.3f}")
        return 0

    assert args.recording, "--recording 을 줘야 한다 (--list_meta / --identify 가 아니면)"
    span = (args.num_frames - 1) * args.stride + 1
    if args.rank:
        rows, total = window_table(root, args.recording, args.num_frames, args.stride, args.step)
        print(f"{args.recording}  총 {total} 프레임  창 {span} 프레임(stride {args.stride})")
        print(f"{'start':>7}{'cam_rot_deg':>13}{'human_m':>10}{'n_action':>10}")
        for begin, rot, move, acts in rows[: args.top]:
            print(f"{begin:>7}{rot:>13.2f}{move:>10.3f}{acts:>10}")
        return 0

    out_path, fps, shape = cut_clip(root, args.recording, args.start,
                                    args.num_frames, args.stride, args.out)
    acts = window_actions(root, args.recording, args.start, span, args.stride)
    print(f"{out_path}  {shape[1]}x{shape[0]}  {fps / args.stride:.1f} fps  {args.num_frames} 프레임")
    print(f"actions: {', '.join(acts) if acts else '(없음)'}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trumans_root", default=TRUMANS_ROOT_DEFAULT, type=str)
    parser.add_argument("--recording", default=None, type=str)
    parser.add_argument("--start", default=0, type=int)              # 창 시작 프레임
    parser.add_argument("--num_frames", default=49, type=int)        # 코퍼스와 같은 49
    parser.add_argument("--stride", default=1, type=int)             # 30fps -> 그대로면 1
    parser.add_argument("--out", default="/data1/cympyc1785/LatentCamVid/tmp/trumans_clip.mp4", type=str)

    parser.add_argument("--rank", action="store_true", default=False)   # 창 점수표만 찍고 끝
    parser.add_argument("--step", default=30, type=int)              # --rank 창 간격
    parser.add_argument("--top", default=15, type=int)
    parser.add_argument("--list_meta", action="store_true", default=False)
    parser.add_argument("--identify", default=None, type=str)        # 클립 mp4 → 녹화/시작 프레임
    sys.exit(main(parser.parse_args()))
