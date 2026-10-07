"""이 코드가 답하는 질문: diffusion 이 생성한 궤적을 VAE roundtrip 비교표와 **같은 지표** (mse_trans · jerk · jerk ratio · 마지막 프레임
mse_trans · Late trans/rot vel err · 첫·끝 구간 속도비) 로 재면 어떤가? (R237, 사용자 2026-10-07 "jerk 같은 VAE roundtrip 비교할 때 썼던 metric들")

eval_testset 출력 (<eval_dir>/test/*_transforms_{ref,pred}.json, OpenGL c2w world) 을 VAE 캐시와 같은 cam_param 공간으로 되돌린다:
OpenCV 로 축 반전 → frame 0 기준 상대 w2c → rot6d = R 앞 두 열, trans = t / s. s 는 entry 별로 GT 에서 정한다
(s = Σ|t_ref| / Σ|t_cache|, --cache 의 v1_test.npz cam_param 과 1e-7 로 일치 확인). pred 도 같은 s 로 나눈다.
이동 0 인 정지 GT 는 같은 scene 형제의 s (scene 단위 상수) — 형제 없는 entry 는 제외 (v1 test 5,144 중 24).
intrinsics 채널은 transforms 에 프레임별 값이 없어 GT 값을 그대로 둔다 (jerk·mse_trans 는 rot6d+trans 만 본다).
지표 함수는 scripts/vae/train_camera_vae.py (late_errors · jerk · edge_ratios) 를 그대로 쓴다.

    python scripts/eval/pred_roundtrip_metrics.py --arm F2=<eval_dir> ... [--cache tmp/r175/cache] --out <json>
"""
import argparse
import importlib.util
import json
import os.path as osp

import numpy as np

_spec = importlib.util.spec_from_file_location("tcv", osp.join(osp.dirname(__file__), "..", "vae", "train_camera_vae.py"))
TCV = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(TCV)
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def to_cam(C, s):
    w2c = np.linalg.inv(C @ FLIP)
    rel = w2c @ np.linalg.inv(w2c[0])
    R, t = rel[:, :3, :3], rel[:, :3, 3]
    return np.concatenate([R[:, :, 0], R[:, :, 1], t / s], -1)


def scene_scales(d, names, gt_cache):
    """entry → s. s 는 scene (dynpose_<uuid>) 단위로 같다 (avg_scale, 형제 entry 간 상대 차이 ≤ 3e-7 실측) — 이동 0 인 정지 entry 는
    같은 scene 형제의 s 를 쓴다. 형제도 없으면 None (제외)."""
    S = {}
    for nm, g in zip(names, gt_cache):
        den = np.linalg.norm(g[:, 6:9], axis=-1).sum()
        if den < 1e-9:
            continue
        ref = np.array([f["transform_matrix"] for f in json.load(open(f"{d}/test/{nm}_transforms_ref.json"))["frames"]])
        w2c = np.linalg.inv(ref @ FLIP); t = (w2c @ np.linalg.inv(w2c[0]))[:, :3, 3]
        S[nm] = np.linalg.norm(t, axis=-1).sum() / den
    sc = {}
    for k, v in S.items():
        sc.setdefault(k.rsplit("_", 1)[0], v)
    return {nm: S.get(nm, sc.get(nm.rsplit("_", 1)[0])) for nm in names}


def load_arm(d, names, gt_cache, scales):
    P, G, keep = [], [], []
    for i, (nm, g) in enumerate(zip(names, gt_cache)):
        s = scales[nm]
        if s is None:
            continue
        ref = np.array([f["transform_matrix"] for f in json.load(open(f"{d}/test/{nm}_transforms_ref.json"))["frames"]])
        prd = np.array([f["transform_matrix"] for f in json.load(open(f"{d}/test/{nm}_transforms_pred.json"))["frames"]])
        G.append(np.concatenate([to_cam(ref, s), g[:, 9:]], -1)); P.append(np.concatenate([to_cam(prd, s), g[:, 9:]], -1)); keep.append(i)
    return np.stack(P), np.stack(G), np.array(keep)


def metrics(P, G):
    err = (P - G) ** 2
    rj, gj = TCV.jerk(P), TCV.jerk(G)
    out = {"mse_trans": float(err[..., 6:9].mean()), "mse_rot": float(err[..., :6].mean()), "mse_trans_last": float(err[:, -1, 6:9].mean()),
           "jerk": float(rj.mean()), "jerk_gt": float(gj.mean()), "jerk_ratio": float(rj.mean() / gj.mean()),
           "jerk_ratio_med": float(np.median(rj / np.maximum(gj, 1e-9))), "n": len(P)}
    out.update(TCV.late_errors(P, G)); out.update(TCV.edge_ratios(P, G))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="이름=eval_dir")
    ap.add_argument("--cache", default="/data1/cympyc1785/LatentCamVid/tmp/r175/cache")
    ap.add_argument("--set", default="v1_test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = np.load(f"{a.cache}/{a.set}.npz", allow_pickle=True)
    names, cam = [str(x) for x in c["names"]], c["cam"].astype(np.float64)
    res, scales = {}, None
    for spec in a.arm:
        k, d = spec.split("=", 1)
        if scales is None:
            scales = scene_scales(d, names, cam)
        P, G, keep = load_arm(d, names, cam, scales)
        if "GT" not in res:      # GT 행: cache 와 되돌린 GT 가 같은지 + 제외 수
            res["GT"] = {"recon_max_abs_err": float(np.abs(G[..., :9] - cam[keep, :, :9]).max()), "n_used": int(len(keep)),
                         "n_dropped_no_scale": int(len(names) - len(keep))}
        res[k] = metrics(P, G)
        print(k, {x: round(v, 6) if isinstance(v, float) else v for x, v in res[k].items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
