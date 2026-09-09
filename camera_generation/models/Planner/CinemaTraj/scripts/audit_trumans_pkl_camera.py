"""실제 TRUMANS 카메라 pkl 2편이 사람이 **걸을 때** 얼마나 움직이는지 잰다.

왜: 합성 소스 카메라를 "pkl 실측 net |dt| 0.58~0.62 m"에 맞춰 놨는데, 방금 채굴한 보행 클립은
net 1.19 m 로 2배가 나왔다. track_gain 0.6 이 걷는 사람을 따라간 결과다. 그게 과한 건지
아니면 실제 카메라도 그러는 건지는 pkl 을 보면 답이 나온다.
"""
import pickle
from os import path

import numpy as np

ROOT = "/data1/cympyc1785/data/trumans/Data_release"
PAIRS = [("00add26c-7a26-4a61-b192-b97aa493b3f3", "2023-01-17@00-33-01"),
         ("0aa05d5a-81d5-497b-832c-c90c3fe73a36", "2023-01-14@23-52-23")]
SMPL_HORIZONTAL = [0, 2]     # y-up
W, FPS, SPEED = 49, 25.0, 0.4


def camera_positions(blob):
    """pkl 구조를 모르므로 (F,3) 또는 (F,4,4) 를 찾아 카메라 중심만 뽑는다."""
    if isinstance(blob, dict) and all(isinstance(k, int) for k in blob):
        # 프레임 인덱스 -> per-frame dict. 그 안에서 위치처럼 보이는 걸 찾는다.
        keys = sorted(blob)
        inner = blob[keys[0]]
        print(f"  per-frame dict, {len(keys)} 프레임, inner keys: "
              f"{sorted(inner) if isinstance(inner, dict) else type(inner)}")
        if isinstance(inner, dict):
            for name in ("location", "position", "translation", "cam_t", "T", "matrix_world"):
                if name in inner:
                    blob = [blob[k][name] for k in keys]
                    print(f"  -> inner['{name}']")
                    break
            else:
                raise SystemExit(f"위치 키를 못 찾았다: {sorted(inner)}")
        else:
            blob = [blob[k] for k in keys]
    elif isinstance(blob, dict):
        for key in ("cam_t", "camera_pose", "poses", "extrinsic", "T", "transl"):
            if key in blob:
                blob = blob[key]
                break
        else:
            blob = list(blob.values())[0]
    arr = np.asarray(blob, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
        return arr[:, :3, 3], f"(F,4,4) -> t  {arr.shape}"
    if arr.ndim == 3 and arr.shape[-2:] == (3, 4):
        return arr[:, :3, 3], f"(F,3,4) -> t  {arr.shape}"
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, f"(F,3)  {arr.shape}"
    raise SystemExit(f"모르는 pkl 모양: {arr.shape}")


def main():
    seg = np.load(path.join(ROOT, "seg_name.npy"))
    transl = np.load(path.join(ROOT, "human_transl.npy"), mmap_mode="r")
    for recording, sequence in PAIRS:
        with open(path.join(ROOT, "Recordings_blend", recording,
                            f"{sequence}_camera_pose.pkl"), "rb") as file:
            blob = pickle.load(file)
        if isinstance(blob, dict):
            print(f"  keys: {sorted(blob)[:8]}")
        cam, shape = camera_positions(blob)

        hit = np.flatnonzero(seg == sequence)
        assert hit.size, f"{sequence} 가 seg_name.npy 에 없다"
        body = np.asarray(transl[hit[0]:hit[-1] + 1], dtype=np.float64)[:, SMPL_HORIZONTAL]
        n = min(len(cam), len(body))
        cam, body = cam[:n], body[:n]

        step = np.linalg.norm(np.diff(body, axis=0), axis=1) * FPS
        step = np.convolve(step, np.ones(9) / 9, mode="same")
        walking = (step > SPEED).astype(np.float64)

        rows = {"walk": [], "still": []}
        for i in range(0, n - W - 1, 10):
            frac = walking[i:i + W].mean()
            bucket = "walk" if frac >= 0.8 else ("still" if frac <= 0.1 else None)
            if bucket:
                rows[bucket].append((np.linalg.norm(cam[i + W] - cam[i]),
                                     np.linalg.norm(np.diff(cam[i:i + W + 1], axis=0), axis=1).mean(),
                                     np.linalg.norm(body[i + W] - body[i])))
        print(f"\n{recording[:8]}  seq {sequence}  cam {shape}  frames {n}")
        for bucket, data in rows.items():
            if not data:
                print(f"  {bucket:5s}  창 0개")
                continue
            arr = np.asarray(data)
            print(f"  {bucket:5s}  창 {len(arr):3d}   cam net |dt| median {np.median(arr[:, 0]):.3f} m"
                  f"   cam dt/frame median {np.median(arr[:, 1]):.4f} m"
                  f"   사람 net {np.median(arr[:, 2]):.3f} m")


if __name__ == "__main__":
    main()
