"""<scene>/da3/depth.npz 를 **비압축 .npy** 로 풀어 두는 캐시 빌더 (geo_encoder='custom' 전용).

왜 필요한가
-----------
depth.npz 는 deflate 압축된 zip 이라 6 장짜리 context view 만 필요해도 scene 전체
(N, 280, 504) fp16 을 통째로 풀어야 한다. 실측 **0.62 s / item** (31 MB -> 93 MB).
num_thread=8 기준 epoch 당 5479 item / (8/0.62) = 425 s -> 150 epoch 이면 **17.7 시간이
depth 압축 해제**에만 들어간다. 비압축 .npy 로 풀어 두면 dataset 이 mmap_mode='r' 로 열어
필요한 6 프레임(~1.7 MB)만 읽으므로 사실상 0 이 된다.

크기: scene 당 N*280*504*2 B (330 프레임이면 93 MB), 957 scene 기준 **~89 GB**.

배치: <out_root>/<chunk>/<scene_hash>.npy — dataset 의 scene_dir 을 dl3dv_root 기준
상대경로로 그대로 옮긴 것이라 dataset_dl3dv._da3_depth 가 경로를 재구성할 수 있다.
해상도/dtype 은 **원본 그대로** 둔다 (280x504 fp16). custom_geo_input_hw 로 줄이는 건
로드 시점에 하는 게 맞다 — 캐시에 encoder 설정을 굳히면 해상도 바꿀 때 89 GB 를 다시
만들어야 한다.

이미 있고 프레임 수가 맞으면 건너뛴다 (중단 후 재실행 안전).

env:
  META      meta csv 이름 (기본 meta_da3_1k.csv). dl3dv_root 아래에서 찾는다.
  OUT       출력 루트 (기본 <dl3dv_root>/da3_depth_raw)
  ROOT      dl3dv_root (기본 /data1/cympyc1785/data/DL3DV/scenes)
  JOBS      병렬 프로세스 수 (기본 8)
  LIMIT     앞에서 N scene 만 (기본 0 = 전부)
"""
import os
import os.path as osp
import sys
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.environ.get("ROOT", "/data1/cympyc1785/data/DL3DV/scenes")
META = os.environ.get("META", "meta_da3_1k.csv")
OUT = os.environ.get("OUT", osp.join(ROOT, "da3_depth_raw"))
JOBS = int(os.environ.get("JOBS", "8"))
LIMIT = int(os.environ.get("LIMIT", "0"))


def _read_meta_chunks(root, meta_name):
    """meta csv -> [scene_chunk] (dl3dv_root 기준 상대경로, 예 '1K/001dcc...')."""
    p = meta_name if osp.isabs(meta_name) else osp.join(root, meta_name)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[0] if k in ("scene_chunk", "chunk", "scene"))
    return [r[key] for r in rows]


def convert(chunk):
    """-> (chunk, 'ok' | 'skip' | 'miss' | 'err: ...')."""
    src = osp.join(ROOT, chunk, "da3", "depth.npz")
    dst = osp.join(OUT, chunk + ".npy")
    if not osp.isfile(src):
        return chunk, "miss"
    try:
        if osp.isfile(dst):
            a = np.load(dst, mmap_mode="r")
            z = np.load(src, mmap_mode=None)     # header 만 보려 해도 npz 는 열어야 한다
            n = z["depth"].shape[0] if "depth" in z else -1
            if a.shape[0] == n:
                return chunk, "skip"
        d = np.load(src)["depth"]                # (N,280,504) fp16
        os.makedirs(osp.dirname(dst), exist_ok=True)
        tmp = dst + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:               # 파일 객체로 써야 np.save 가 '.npy' 를 덧붙이지 않는다
            np.save(f, np.ascontiguousarray(d))  # 비압축 -> mmap 가능
        os.replace(tmp, dst)                     # 원자적 교체 (중단 시 반쪽 파일 방지)
        return chunk, "ok"
    except Exception as e:                        # noqa: BLE001 — scene 하나 실패로 전체를 죽이지 않는다
        return chunk, f"err: {type(e).__name__}: {e}"


def main():
    chunks = _read_meta_chunks(ROOT, META)
    if LIMIT:
        chunks = chunks[:LIMIT]
    print(f"{len(chunks)} scene  {ROOT}/<chunk>/da3/depth.npz -> {OUT}/<chunk>.npy  (jobs={JOBS})")
    os.makedirs(OUT, exist_ok=True)
    cnt = {"ok": 0, "skip": 0, "miss": 0, "err": 0}
    errs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        futs = [ex.submit(convert, c) for c in chunks]
        for i, f in enumerate(as_completed(futs), 1):
            c, st = f.result()
            k = st if st in cnt else "err"
            cnt[k] += 1
            if k == "err":
                errs.append((c, st))
            if i % 50 == 0 or i == len(chunks):
                el = time.time() - t0
                print(f"  [{i}/{len(chunks)}] {cnt}  {el:.0f}s "
                      f"(eta {el / i * (len(chunks) - i):.0f}s)", flush=True)
    print(f"\ndone {cnt}  {time.time() - t0:.0f}s")
    for c, st in errs[:20]:
        print(f"  {c}  {st}")
    if cnt["miss"] or cnt["err"]:
        # depth 가 없는 scene 은 custom geo arm 에서 쓸 수 없다. dataset 은 npz 폴백을 하지만
        # npz 도 없으면 __getitem__ 이 터지므로 여기서 목록을 남겨 둔다.
        p = osp.join(OUT, "_failed.txt")
        with open(p, "w") as f:
            for c, st in errs:
                f.write(f"{c}\t{st}\n")
        print(f"failed list -> {p}")
    return 0 if not cnt["err"] else 1


if __name__ == "__main__":
    sys.exit(main())
