"""DL3DV da3 avg_scale — **front-only context range + target-first-frame anchor**.

`/data1/cympyc1785/pipeline/workspace/make_avg_scale_da3_firstcam.py` 의 형제 스크립트다
(점 집합 규약 conf >= 전역 P40 / pixel_stride 2 / mean 거리 는 한 글자도 안 바꿨다).
이쪽은 저장소 안에 둔다 — 두 arm 의 분모 정의가 실험 결과 해석에 직접 걸리는 값이라
버전 관리가 되어야 한다.

기존 두 변형과의 차이
---------------------
                        context range                      origin(기준점)
  avg_scale             [0,s)/[e,N) 중 **긴 쪽**            context 카메라 중심들의 centroid
  avg_scale_context_first_cam  같음                         context range 의 첫 카메라
  ---- 이 스크립트가 만드는 것 ----
  avg_scale_front_first_anchor            **[0, s)** (앞쪽만)   **target segment 첫 카메라 s**
  avg_scale_front_first_anchor_same_len   **[s-L, s)**, L=e-s   **target segment 첫 카메라 s**

왜 앞쪽만인가: 기존 두 변형은 context range 가 target 오른쪽([e,N))에 잡힐 수 있는데, 그때
target 첫 카메라 s 를 기준점으로 거리를 재면 "아직 안 지나간 뒤쪽 공간까지의 거리"를 재는 셈이라
분모의 의미가 흐려진다. 앞쪽으로 고정하면 s 가 context range 의 바로 뒤에 붙어 있어 기준점과
점 집합이 공간적으로 이어진다.

왜 anchor 가 target 첫 카메라인가: 학습에서 cam_param 은
`E @ inv(E_s)` 로 **frame s 기준**으로 재고정된 뒤 이 분모로 나뉜다
(utils/data_utils.normalize_camera_extrinsics_and_points). 기존 변형은 분모의 기준점이
frame s 가 아니라 context 쪽에 있어서 "재고정 기준"과 "스케일 기준"이 어긋나 있었다.
여기서는 둘을 일치시킨다. s 의 포즈는 추론 시에도 주어지므로(first_extrinsic) leakage 가 아니다.
점은 여전히 target 프레임 [s,e) 를 전혀 안 쓴다.

제외되는 세그먼트 (파일을 안 쓴다 -> dataset 이 인덱스에서 뺀다)
  front_first_anchor           s == 0        (앞쪽 context 가 비어 있음)
  front_first_anchor_same_len  s < L         (앞쪽 L 프레임을 못 채움)
DL3DV da3 코퍼스는 세그먼트가 전부 길이 49 이고 s 가 0,49,98,... 이라 두 조건이 s==0 으로
같아진다 -> 두 arm 의 학습 샘플 집합이 동일해서 paired 비교가 된다 (300 scene 표본에서 확인).

usage:
  python make_avg_scale_da3_front_anchor.py --splits 1K 2K 3K 4K 5K 6K 7K --workers 12
"""
import os, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

ROOT = "/data1/cympyc1785/data/DL3DV/scenes"
PS, PCTL = 2, 40.0                       # make_avg_scale_da3_firstcam.py 와 동일
DIR_FULL = "avg_scale_front_first_anchor"
DIR_SAME = "avg_scale_front_first_anchor_same_len"


def worker(da3):
    try:
        pr_path = os.path.join(da3, "prompts.json")
        if not os.path.exists(pr_path):
            return ("noprompt", 0, 0)
        depth = np.load(os.path.join(da3, "depth.npz"))["depth"].astype(np.float32)
        conf = np.load(os.path.join(da3, "conf.npz"))["conf"].astype(np.float32)
        pz = np.load(os.path.join(da3, "pose.npz"))
        ext = pz["extrinsics"].astype(np.float32); K = pz["intrinsics"].astype(np.float32)
        N, H, W = depth.shape
        thr = float(np.percentile(conf, PCTL))          # 전역 P40 (프레임별이 아니다)
        us = np.arange(0, W, PS); vs = np.arange(0, H, PS); uu, vv = np.meshgrid(us, vs)
        flat = (vv * W + uu).reshape(-1)
        pix = np.stack([uu.reshape(-1), vv.reshape(-1), np.ones(flat.size)], -1).astype(np.float32)
        cam_c = np.array([-ext[i][:3, :3].T @ ext[i][:3, 3] for i in range(N)])
        pf = []                                          # 프레임별 world point 캐시
        for i in range(N):
            dd = depth[i].reshape(-1)[flat]; cc = conf[i].reshape(-1)[flat]
            m = np.isfinite(dd) & (dd > 0) & (cc >= thr)
            if m.any():
                Xc = (np.linalg.inv(K[i]) @ pix[m].T) * dd[m][None, :]
                pf.append((ext[i][:, :3].T @ (Xc - ext[i][:, 3][:, None])).T.astype(np.float32))
            else:
                pf.append(np.zeros((0, 3), np.float32))
        prompts = json.load(open(pr_path))
        d_full = os.path.join(da3, DIR_FULL); os.makedirs(d_full, exist_ok=True)
        d_same = os.path.join(da3, DIR_SAME); os.makedirs(d_same, exist_ok=True)
        w_full = w_same = 0
        for k in sorted(prompts, key=int):
            if "frame_idx" not in prompts[k]:
                continue
            s, e = prompts[k]["frame_idx"]
            L = e - s
            anchor = cam_c[s]                            # target segment 첫 카메라
            for ctx, out_dir, is_same in ((list(range(0, s)), d_full, False),
                                          (list(range(max(0, s - L), s)), d_same, True)):
                if len(ctx) == 0 or (is_same and len(ctx) < L):
                    continue                             # 파일 없음 = 학습에서 제외
                P = np.concatenate([pf[i] for i in ctx], 0)
                if len(P) == 0:
                    continue
                av = float(np.linalg.norm(P - anchor[None, :], axis=1).mean())
                if not np.isfinite(av) or av <= 0:
                    continue
                json.dump(av, open(os.path.join(out_dir, f"{int(k)}.json"), "w"))
                if is_same:
                    w_same += 1
                else:
                    w_full += 1
        return ("ok", w_full, w_same)
    except Exception as ex:
        return ("fail", f"{da3} :: {ex}", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    da3s = []
    for sp in a.splits:
        sd = os.path.join(ROOT, sp)
        if not os.path.isdir(sd):
            continue
        for sc in sorted(os.listdir(sd)):
            da3 = os.path.join(sd, sc, "da3")
            if os.path.exists(os.path.join(da3, "pose.npz")):
                da3s.append(da3)
    if a.limit:
        da3s = da3s[:a.limit]
    print(f"scenes: {len(da3s)} (front-only ctx + target-first-cam anchor, mean, "
          f"P{PCTL}, PS{PS})", flush=True)
    c = {}; nf = ns = 0; fails = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for st, i1, i2 in tqdm(ex.map(worker, da3s, chunksize=4), total=len(da3s)):
            c[st] = c.get(st, 0) + 1
            if st == "ok":
                nf += i1; ns += i2
            elif st == "fail" and len(fails) < 10:
                fails.append(i1)
    print("DONE:", c, f"| {DIR_FULL}: {nf} files | {DIR_SAME}: {ns} files", flush=True)
    for f in fails:
        print("  FAIL", f, flush=True)


if __name__ == "__main__":
    main()
