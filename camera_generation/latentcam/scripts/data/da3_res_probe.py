"""DA3 geo encoder 입력 해상도 probe — 252x448 vs 280x504.

왜
---
`geo_encoder='da3'` (models/da3_geo_encoder.py) 를 붙이기 전에 **입력 해상도를 뭘로 굳힐지**를
먼저 정해야 한다. 기존 custom backend 는 `custom_geo_input_hw=[252,448]` (DINOv2 patch 14 격자,
view 당 576 토큰) 인데, DA3 의 native 처리 해상도는 504 (긴 변) 즉 **280x504** (view 당 720 토큰)
이고 이 프로젝트의 corpus depth (`<scene>/da3/depth.npz`) 도 거기서 나왔다.
252x448 은 280x504 의 **등방 축소** (양축 x8/9) 라 aspect 왜곡은 없지만 실효 해상도가 준다.

그래서 두 해상도로 같은 장면의 depth 를 뽑아 (a) 수치로, (b) 영상으로 비교한다.

무엇을 재나
-----------
1. `--check-path`: **api 우회 로딩 경로 검증.** `depth_anything_3.api.DepthAnything3` 로 만든
   모델과, encoder 가 쓸 우회 경로(`cfg.load_config` + `registry.MODEL_REGISTRY` +
   `safetensors` 직접 로드)로 만든 모델의 depth 가 같은지 max|Δ| 로 확인한다.
   Step 2 의 `DA3SceneEncoder` 가 api 와 같은 가중치를 쓴다는 근거.
2. 해상도 비교: 252x448 depth 를 280x504 로 bilinear 업샘플 -> `least_squares_scale_scalar`
   로 스케일 정렬 -> **AbsRel + delta<1.25 / 1.25^2 / 1.25^3**. (280x504 를 기준으로 본다.)
3. DL3DV 는 저장된 `<scene>/da3/depth.npz` 와도 같은 지표로 비교한다. 단 저장본은
   **DA3NESTED-GIANT-LARGE-1.1** 로 만든 것이라 모델이 다르다 — 절대 기준이 아니라 참고치다.
4. 영상: `rgb | depth@280x504 | depth@252x448(up)` 가로 concat mp4.
   컬러맵 min/max 는 **280x504 것으로 고정해 두 쪽에 같이 쓴다** (색이 비교 가능해야 한다).

주의
----
- `depth_anything_3.api` 는 `utils/export/gs.py` 가 `moviepy.editor` 를 import 해서 latentcam
  env 에서 그냥은 안 열린다. moviepy 는 gs_video export 에만 쓰이므로 **6줄짜리 stub** 으로
  막는다 (AlayaWorld `flash_alaya/alaya/memory/da3_depth.py::_install_da3_export_dependency_stubs`
  와 같은 수법). DA3 저장소는 한 줄도 고치지 않는다.
- mp4 는 imageio-ffmpeg 로 쓴다 (moviepy 금지 — 위 이유로 없다).

예시
----
    CUDA_VISIBLE_DEVICES=0 python scripts/data/da3_res_probe.py \
        --out_dir results/20260812_da3_res_probe --views 32 --check-path
"""
import argparse
import json
import os
import os.path as osp
import sys
import time
import types

import numpy as np

DA3_REPO = "/data1/cympyc1785/LatentCamVid/camera_generation/tools/Depth-Anything-3"
HF_HOME = "/data1/cympyc1785/cache/huggingface"
DL3DV_ROOT = "/data1/cympyc1785/data/DL3DV/scenes"
DV_FRAMES = "/data1/cympyc1785/data/dynamicverse_frames"

# 두 후보 해상도. DA3 input processor 는 '긴 변' 기준(upper_bound_resize)이라 process_res 만
# 주면 patch(14) 배수까지 알아서 맞춘다: 504 -> 280x504, 448 -> 252x448 (16:9 입력 기준).
RES = {"504": 504, "448": 448}
HW = {"504": (280, 504), "448": (252, 448)}


# ---------------------------------------------------------------- DA3 import shim
def _stub_moviepy():
    """moviepy 는 DA3 의 gs_video export 에만 필요하다. 없으면 import 자체가 막히므로 stub."""
    if "moviepy" in sys.modules:
        return
    try:
        import moviepy  # noqa: F401
        return
    except Exception:
        pass

    def _missing(*a, **k):
        raise ImportError("moviepy is required only for DA3 gs_video export")

    mv = types.ModuleType("moviepy")
    mv.__file__ = "<optional dependency stub: moviepy>"
    mv.__path__ = []
    ed = types.ModuleType("moviepy.editor")
    ed.__file__ = "<optional dependency stub: moviepy.editor>"
    ed.__getattr__ = lambda name: _missing
    mv.editor = ed
    sys.modules["moviepy"] = mv
    sys.modules["moviepy.editor"] = ed


def _init_da3(repo=DA3_REPO, hf_home=HF_HOME):
    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")   # forward 마다 INFO 한 줄씩 뱉는다
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("HF_HOME", hf_home)
    src = osp.join(osp.realpath(repo), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    _stub_moviepy()


def _hf_snapshot(model_name, hf_home=HF_HOME):
    """로컬 HF 캐시의 snapshot 경로 (네트워크 없이). 없으면 model_name 그대로."""
    hub = osp.join(hf_home, "hub")
    root = osp.join(hub if osp.isdir(hub) else hf_home,
                    "models--" + model_name.replace("/", "--"))
    refs = osp.join(root, "refs", "main")
    cands = []
    if osp.isfile(refs):
        with open(refs) as f:
            cands.append(osp.join(root, "snapshots", f.read().strip()))
    snaps = osp.join(root, "snapshots")
    if osp.isdir(snaps):
        cands += [osp.join(snaps, d) for d in sorted(os.listdir(snaps), reverse=True)]
    for c in cands:
        if osp.isfile(osp.join(c, "config.json")) and osp.isfile(osp.join(c, "model.safetensors")):
            return c
    return model_name


def load_api_model(model_name, device):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained(_hf_snapshot(model_name), local_files_only=True)
    return m.to(device).eval()


def load_bypass_net(registry_key, device, state_prefix="model."):
    """encoder 가 쓸 경로: api 를 거치지 않고 DepthAnything3Net 을 직접 만든다.

    반환값은 `DepthAnything3Net` (backbone + head + cam_enc/cam_dec). probe 는 depth 를 봐야
    하므로 head 를 남긴다 — 실제 encoder 는 여기서 head/cam_dec 를 지운다.
    """
    from depth_anything_3.cfg import load_config, create_object
    from depth_anything_3.registry import MODEL_REGISTRY
    from safetensors.torch import load_file

    net = create_object(load_config(MODEL_REGISTRY[registry_key]))
    sd = load_file(osp.join(_hf_snapshot(_HF_NAME[registry_key]), "model.safetensors"))
    sd = {(k[len(state_prefix):] if k.startswith(state_prefix) else k): v for k, v in sd.items()}
    miss, unexp = net.load_state_dict(sd, strict=False)
    print(f"[bypass] {registry_key}: missing={len(miss)} unexpected={len(unexp)}")
    if unexp:
        print(f"[bypass]   unexpected(first 5)={list(unexp)[:5]}")
    if miss:
        print(f"[bypass]   missing(first 8)={list(miss)[:8]}")
    return net.to(device).eval()


_HF_NAME = {
    "da3-large": "depth-anything/DA3-LARGE-1.1",
    "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
}


# ---------------------------------------------------------------- scenes
def pick_dl3dv(n):
    out = []
    for chunk in sorted(os.listdir(DL3DV_ROOT)):
        cd = osp.join(DL3DV_ROOT, chunk)
        if not osp.isdir(cd):
            continue
        for scene in sorted(os.listdir(cd)):
            sd = osp.join(cd, scene)
            if osp.isdir(osp.join(sd, "images_8")) and osp.isfile(osp.join(sd, "da3", "depth.npz")):
                out.append((f"dl3dv/{chunk}/{scene[:12]}", sd))
            if len(out) >= n:
                return out
    return out


def pick_dv(n):
    out = []
    subs = sorted(d for d in os.listdir(DV_FRAMES) if osp.isdir(osp.join(DV_FRAMES, d)))
    # subset 을 돌아가며 하나씩 집어 한 subset 에 몰리지 않게 한다
    per = {s: sorted(os.listdir(osp.join(DV_FRAMES, s))) for s in subs}
    i = 0
    while len(out) < n and any(per.values()):
        s = subs[i % len(subs)]
        i += 1
        if not per[s]:
            continue
        scene = per[s].pop(0)
        out.append((f"dv/{s}/{scene[:16]}", osp.join(DV_FRAMES, s, scene)))
    return out


def scene_frames(scene_dir):
    """-> (정렬된 이미지 경로 리스트, 'dl3dv' | 'dv')."""
    imgs = osp.join(scene_dir, "images_8")
    if osp.isdir(imgs):
        return [osp.join(imgs, f) for f in sorted(os.listdir(imgs))
                if f.lower().endswith((".png", ".jpg"))], "dl3dv"
    return [osp.join(scene_dir, f) for f in sorted(os.listdir(scene_dir))
            if f.lower().endswith((".png", ".jpg"))], "dv"


# ---------------------------------------------------------------- metrics
def depth_metrics(pred, ref, eps=1e-6, far_pct=98.0):
    """pred 를 ref 에 스케일 정렬한 뒤 AbsRel / delta 지표. 둘 다 (N,H,W) np.

    DA3 (metric 모델이 아닌 경우) depth 는 **scale-ambiguous** 라 반드시 정렬이 필요하다.
    정렬은 두 가지를 다 낸다:
      - `ls_scale`   : `least_squares_scale_scalar` (L2 최소화 -> 먼 픽셀이 지배한다)
      - `med_scale`  : median(ref/pred) — 로버스트. **지표는 이쪽으로 계산한다.**
    DA3 는 하늘을 `non_sky_max` 로 채워 넣으므로 (`_process_mono_sky_estimation`) ref 상위
    `far_pct` 퍼센타일 위는 마스크에서 뺀다. 안 그러면 하늘 상수 영역이 AbsRel 을 먹는다.
    """
    import torch
    from depth_anything_3.utils.alignment import least_squares_scale_scalar

    p = torch.from_numpy(np.ascontiguousarray(pred)).float()
    r = torch.from_numpy(np.ascontiguousarray(ref)).float()
    m = torch.isfinite(p) & torch.isfinite(r) & (p > eps) & (r > eps)
    if m.sum() >= 100:
        far = float(np.percentile(r[m].numpy(), far_pct))
        m &= r <= far
    if m.sum() < 100:
        return {"n_valid": int(m.sum()), "absrel": float("nan")}

    ls = float(least_squares_scale_scalar(r[m], p[m]))
    s = float(torch.median(r[m] / p[m]))
    ps = p * s
    ratio = torch.maximum(ps[m] / r[m], r[m] / ps[m])
    # view 별 스케일 산포 — 한 스칼라로 맞춰지는지 (안 맞으면 해상도가 geometry 를 바꾼 것)
    per_view = [float(torch.median(r[i][m[i]] / p[i][m[i]]))
                for i in range(p.shape[0]) if m[i].sum() >= 100]
    return {
        "n_valid": int(m.sum()),
        "ls_scale": ls,
        "med_scale": s,
        "absrel": float(((ps[m] - r[m]).abs() / r[m]).mean()),
        "d1": float((ratio < 1.25).float().mean()),
        "d2": float((ratio < 1.25 ** 2).float().mean()),
        "d3": float((ratio < 1.25 ** 3).float().mean()),
        "per_view_scale_std_rel": (float(np.std(per_view) / (np.mean(per_view) + eps))
                                   if per_view else float("nan")),
    }


# ---------------------------------------------------------------- video
def write_video(path, frames, fps=10):
    import imageio.v2 as imageio
    os.makedirs(osp.dirname(path), exist_ok=True)
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                           macro_block_size=1, ffmpeg_log_level="error")
    for f in frames:
        w.append_data(f)
    w.close()


def label(img, text, bar=22):
    """이미지 위에 검은 띠를 **덧대서** (덮지 않고) 캡션을 단다."""
    import cv2
    strip = np.zeros((bar, img.shape[1], 3), np.uint8)
    cv2.putText(strip, text, (6, bar - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([strip, img], axis=0)


# ---------------------------------------------------------------- main
def run_scene(model, paths, process_res):
    """api 모델로 한 scene 을 한 번에 forward. -> (depth (N,H,W) f32, rgb (N,H,W,3) u8)."""
    pred = model.inference(paths, process_res=process_res, export_dir=None)
    depth = np.asarray(pred.depth, dtype=np.float32)
    imgs = pred.processed_images
    imgs = np.asarray(imgs)
    if imgs.ndim == 4 and imgs.shape[1] == 3:                 # (N,3,H,W) -> (N,H,W,3)
        imgs = imgs.transpose(0, 2, 3, 1)
    if imgs.dtype != np.uint8:
        imgs = (np.clip(imgs, 0, 1) * 255).astype(np.uint8)
    return depth, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="da3-large", choices=list(_HF_NAME))
    ap.add_argument("--dl3dv_n", type=int, default=3)
    ap.add_argument("--dv_n", type=int, default=3)
    ap.add_argument("--views", type=int, default=32, help="scene 당 균등 샘플할 프레임 수")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--check-path", action="store_true",
                    help="api 모델 vs 우회 로딩 모델 depth 일치 확인")
    args = ap.parse_args()

    _init_da3()
    import torch
    from depth_anything_3.utils.visualize import visualize_depth

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    model = load_api_model(_HF_NAME[args.model], dev)
    print(f"[load] api {args.model} {time.time() - t0:.1f}s", flush=True)

    scenes = pick_dl3dv(args.dl3dv_n) + pick_dv(args.dv_n)
    print(f"[scenes] {len(scenes)}: " + ", ".join(n for n, _ in scenes), flush=True)

    report = {"model": args.model, "views": args.views, "res": HW, "scenes": {}}

    # -------- 우회 로딩 경로 검증 (첫 scene 의 앞 8 프레임, 504) --------
    if args.check_path:
        name, sd = scenes[0]
        fp, _ = scene_frames(sd)
        sub = fp[:8]
        d_api, _ = run_scene(model, sub, RES["504"])
        net = load_bypass_net(args.model, dev)
        # 전처리는 api 것을 그대로 빌려 쓴다 (여기서 검증하려는 건 '가중치/그래프' 동등성이다)
        imgs_cpu, _, _ = model._preprocess_inputs(sub, None, None, RES["504"], "upper_bound_resize")
        x, _, _ = model._prepare_model_inputs(imgs_cpu, None, None)
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16):
            # export_feat_layers 는 None 이면 backbone 안에서 `i in None` 으로 터진다 -> [] 필수
            raw = net(x.to(dev), None, None, [], False, False, "saddle_balanced")
        d_bypass = raw["depth"].float().cpu().numpy().reshape(d_api.shape)
        rel = np.abs(d_bypass - d_api) / np.maximum(np.abs(d_api), 1e-6)
        report["path_check"] = {
            "scene": name, "n": len(sub),
            "max_abs_diff": float(np.abs(d_bypass - d_api).max()),
            "max_rel_diff": float(rel.max()),
            "mean_rel_diff": float(rel.mean()),
        }
        print(f"[path_check] {report['path_check']}", flush=True)
        del net
        torch.cuda.empty_cache()

    # -------- 해상도 비교 --------
    for name, sd in scenes:
        fp, kind = scene_frames(sd)
        if not fp:
            print(f"[skip] {name}: no frames")
            continue
        idx = np.linspace(0, len(fp) - 1, min(args.views, len(fp))).round().astype(int)
        idx = sorted(set(idx.tolist()))
        sub = [fp[i] for i in idx]

        out = {}
        for tag in ("504", "448"):
            t = time.time()
            d, rgb = run_scene(model, sub, RES[tag])
            out[tag] = (d, rgb)
            print(f"[{name}] {tag} depth{d.shape} {time.time() - t:.1f}s "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
            torch.cuda.reset_peak_memory_stats()

        d504, rgb504 = out["504"]
        d448, _ = out["448"]
        # 448 -> 504 격자로 올려 같은 픽셀에서 비교
        d448_up = torch.nn.functional.interpolate(
            torch.from_numpy(d448)[:, None], size=d504.shape[-2:], mode="bilinear",
            align_corners=False)[:, 0].numpy()

        m = {"448_vs_504": depth_metrics(d448_up, d504)}

        if kind == "dl3dv":
            stored = osp.join(sd, "da3", "depth.npz")
            if osp.isfile(stored):
                z = np.load(stored)["depth"]                 # (N,280,504) f16, nested-giant
                if z.shape[0] == len(fp):
                    ref = z[idx].astype(np.float32)
                    # 참고치일 뿐이다: 저장본은 (a) 모델이 다르고 (nested-giant, metric),
                    # (b) 330 프레임 전체를 한 컨텍스트로 돌린 것이라 view set 도 다르다.
                    m["504_vs_stored"] = depth_metrics(d504, ref)
                    m["448_vs_stored"] = depth_metrics(d448_up, ref)
                    m["stored_note"] = ("reference only: DA3NESTED-GIANT-LARGE-1.1, "
                                        "all 330 frames as context")
                else:
                    m["stored_note"] = f"frame count mismatch {z.shape[0]} vs {len(fp)}"
        report["scenes"][name] = m
        print(f"[{name}] {json.dumps(m)}", flush=True)

        # -------- 영상 --------
        # DA3 non-metric depth 는 scale-ambiguous 라 두 해상도의 절대값이 통째로 다르다
        # (여기 DL3DV 는 ~0.72x). 컬러맵을 504 의 min/max 로 고정해 비교하려면 448 을 먼저
        # 같은 스케일로 올려야 한다 — 안 그러면 '스케일 차이'가 '품질 차이'처럼 보인다.
        s448 = m["448_vs_504"].get("med_scale", 1.0)
        d448_vis = d448_up * (s448 if np.isfinite(s448) else 1.0)
        vid = []
        for i in range(d504.shape[0]):
            c504, lo, hi = visualize_depth(d504[i], ret_minmax=True)
            c448 = visualize_depth(d448_vis[i], depth_min=lo, depth_max=hi)
            row = np.concatenate([label(rgb504[i], "rgb"),
                                  label(c504, f"depth {HW['504'][0]}x{HW['504'][1]}"),
                                  label(c448, f"depth {HW['448'][0]}x{HW['448'][1]} (up, "
                                              f"x{s448:.3f} scale-aligned)")], axis=1)
            vid.append(row)
        vp = osp.join(args.out_dir, "videos", name.replace("/", "__") + ".mp4")
        write_video(vp, vid, fps=args.fps)
        print(f"[{name}] -> {vp}", flush=True)

        del out, d504, d448, d448_up, d448_vis, rgb504
        torch.cuda.empty_cache()

    with open(osp.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {osp.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
