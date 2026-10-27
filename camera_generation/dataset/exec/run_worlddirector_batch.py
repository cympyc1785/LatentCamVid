"""이 코드가 답하는 질문: WorldDirector (SCVideo attn_sup, WorldTraj 학습 ckpt) 를 원래 세팅 그대로 우리 test 에 돌리면 어떤 궤적을 내나?

R583 (사용자 10-27 "학습 빼고는 돌려놔줘" · "어댑터는 WorldDirector 원래 세팅대로, ckpt 랑 구조 맞는지도").
원래 세팅 = SCVideo `config_large.py` + `data/dataset_large.py` + `train_cam_dm_attn_sup.py` 의 점 임베딩 저장 경로 (주석 처리된 pc_embeds_inv 생성부):
  점 구름 = WorldTraj `scene.ply` (학습과 같은 1M 점 컬러) → 첫 카메라 w2c 로 옮김 → avg_scale = 점 거리 평균으로 나눔 → ‖p‖ ≤ 1.25 →
  ×1.8 → y-down→z-up 회전 → Concerto transform (CenterShift · GridSample 0.02 · NormalizeColor) → Concerto large outdoor →
  최하위 stage voxel 마다 [voxel 좌표 (정규화 점 scatter_mean) 3 · 첫 카메라 가시성 1 · feature 768] = 772 토큰.
  텍스트 = umT5-xxl len 512, DDIM 50 (scaled_linear, epsilon), VAE vae_intr (latent 64, scale 0.96032625) decode → out_to_trajectory.
우리 쪽에 맞추는 것은 두 가지뿐이다:
  ① 첫 카메라: v5f_all 의 target 궤적은 소스 frame 0 카메라에서 시작한다 (`vista4d_bank_to_dl3dv.py`: world = DA3 frame0 카메라).
     WorldTraj 영상은 우리 49 프레임보다 길어서 프레임 offset 을 영상 L1 로 찾고, WT cameras.json[offset] 을 첫 카메라로 쓴다.
  ② 단위: WD 출력 이동은 WT 단위 (avg_scale_wt 곱) 다. DA3 단위로 바꾸려고 WT/DA3 깊이 비 s (scene.ply 를 첫 카메라에 투영한 z-buffer ÷ recon depth 중앙값) 로 나눈다.
     WD 는 이동을 장면 크기 (점 거리 평균) 대비로 내므로 깊이 기준 비가 맞는 환산이다.
텍스트는 REF eval 폴더 `_caption.json` "Concise Interaction" (메인 표와 같은 plan text). 출력 npz 는 gendop_preds_to_eval_dir.py 가 읽는 `c2w` (OpenGL, 49).

stage align (env vista4d: OpenEXR depth): scene 마다 offset · s · 정렬 진단 → <out>/align.json
stage infer (env /data1/cympyc1785/venvs/wdconcerto = latentcam + concerto): scene 마다 점 임베딩 1 회 → 그 scene seg 를 한 배치로 샘플 → <out>/worlddirector__<scene>__<idx>.npz
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
from multiprocessing import Pool

import numpy as np

R = "/data1/cympyc1785/LatentCamVid"
SCV = f"{R}/camera_generation/models/SCVideo/camera_generation"
WT_ROOT = f"{R}/DATA/worldtraj/dynamicverse/dynpose-100k"
RECON = f"{R}/DATA/DynPose-100K/eval_data/recon_and_seg"
REF = f"{R}/camera_generation/latentcam/eval_my/20261022_032720_dsv5fallv2_molmo2_l21_noprobe_fm_vaeCrepenclin_zstd_track3d_box3d__last__seed42__v5alltest"
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
A = None


def ref_entries(ref):
    """REF test_valid.txt → {scene: [idx, ...]} (이름 `dynpose_<scene>_<idx>`)."""
    out = {}
    for n in open(osp.join(ref, "test_valid.txt")).read().split():
        sc, idx = n[len("dynpose_"):].rsplit("_", 1)
        out.setdefault(sc, []).append(idx)
    return out


def wt_dir(scene):
    g = glob.glob(osp.join(WT_ROOT, "*", scene))
    return g[0] if g else None


def wt_cameras(path_json):
    """SCVideo camera_utils.get_camera_params_from_json 과 같은 해석: rotation·position 이 w2c."""
    E, K = [], []
    for p in json.load(open(path_json)):
        e = np.eye(4); e[:3, :3] = p["rotation"]; e[:3, 3] = p["position"]; E.append(e)
        K.append(np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], np.float64))
    return np.stack(E), np.stack(K)


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    C = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0 if "red" in v.data.dtype.names else np.ones_like(P, np.float32)
    return P, C


def _gray(path):
    import cv2
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32))
    return np.stack(out)


def align_one(scene):
    import cv2
    sys.path.insert(0, f"{R}/video_generation/models/Vista4D")
    from utils.media import load_cameras, load_depths, load_masks
    wt = wt_dir(scene)
    if wt is None:
        return scene, {"error": "WT scene 없음"}
    try:
        a, b = _gray(osp.join(wt, "video_input.mp4")), _gray(osp.join(RECON, scene, "video.mp4"))
        n = len(b)
        errs = np.array([np.abs(a[o:o + n] - b).mean() for o in range(len(a) - n + 1)])
        o = int(errs.argmin())
        E, K = wt_cameras(osp.join(wt, "cameras.json"))
        C, _ = load_cameras(osp.join(RECON, scene, "cameras.npz")); C = np.asarray(C, np.float64)
        Rw, Rr = E[o:o + n, :3, :3], np.linalg.inv(C)[:, :3, :3]
        rel_w, rel_r = np.einsum("nij,kj->nik", Rw, Rw[0]), np.einsum("nij,kj->nik", Rr, Rr[0])
        ang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", rel_w, rel_r) - 1) / 2, -1, 1)))
        P, _ = load_ply(osp.join(wt, "scene.ply"))
        Xc = P @ E[o, :3, :3].T + E[o, :3, 3]; k = K[o]; H, W = int(round(k[1, 2] * 2)), int(round(k[0, 2] * 2))
        z = Xc[:, 2]; ok = z > 1e-6
        u = np.floor(k[0, 0] * Xc[ok, 0] / z[ok] + k[0, 2]).astype(int); v = np.floor(k[1, 1] * Xc[ok, 1] / z[ok] + k[1, 2]).astype(int); zz = z[ok]
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zb = np.full((H, W), np.inf); np.minimum.at(zb, (v[inb], u[inb]), zz[inb])
        d = load_depths(osp.join(RECON, scene, "depths"), dtype=np.float32, desc=None)[0]
        sky = np.asarray(load_masks(osp.join(RECON, scene, "sky_mask"), desc=None))[0]
        d = cv2.resize(np.where(sky, np.nan, d), (W, H), interpolation=cv2.INTER_NEAREST)
        m = np.isfinite(zb) & np.isfinite(d) & (d > 0)
        r = zb[m] / d[m]
        q25, s, q75 = np.percentile(r, [25, 50, 75])
        return scene, {"offset": o, "video_l1": float(errs[o]), "video_l1_2nd": float(np.sort(errs)[1]) if len(errs) > 1 else None,
                       "rot_err_med": float(np.median(ang)), "rot_err_max": float(ang.max()), "s": float(s), "s_iqr_rel": float((q75 - q25) / s),
                       "n_px": int(m.sum()), "n_wt": int(len(a))}
    except Exception as e:      # noqa: BLE001
        return scene, {"error": str(e)[:200]}


def stage_align():
    scenes = sorted(ref_entries(A.ref))
    if A.limit:
        scenes = scenes[:A.limit]
    out = osp.join(A.out, "align.json")
    done = json.load(open(out)) if osp.isfile(out) else {}
    todo = [s for s in scenes if s not in done or "error" in done[s]]
    with Pool(A.workers) as p:
        for i, (s, r) in enumerate(p.imap_unordered(align_one, todo)):
            done[s] = r
            if i % 100 == 0 or "error" in r:
                print(f"[align {i}/{len(todo)}] {s} {r}", flush=True)
    json.dump(done, open(out, "w"), indent=1)
    ok = [r for r in done.values() if "error" not in r]
    bad = sum(r["rot_err_med"] > 10 or r["s_iqr_rel"] > 0.5 for r in ok)
    print(f"=== align {len(ok)}/{len(scenes)} ok · 정렬 의심 (rot_err_med>10° 또는 s IQR/중앙>0.5) {bad} ===", flush=True)


def stage_infer():
    import types
    import torch
    sys.path[:0] = [f"{SCV}/core_pkg", f"{SCV}/main"]
    for name in ("concerto.module",):           # attn_sup 모델 파일의 쓰지 않는 import (reader 확인)
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:                    # noqa: BLE001
                sys.modules[name] = types.SimpleNamespace(PointSequential=None)
    import concerto
    from diffusers import DDIMScheduler
    from torch_scatter import scatter_mean
    from core_pkg.common.utils.data_utils import out_to_trajectory
    from core_pkg.common.utils.pc_utils import get_ray_sim_per_point
    from core_pkg.models.camera_diffusion_model_attn_sup import CameraDiffusionModel
    from core_pkg.models.pc_encoder import PCEncoder
    from core_pkg.models.t5 import T5EncoderModel
    from core_pkg.models.vae_intr import CameraVAE

    dev = "cuda:0"
    al = json.load(open(osp.join(A.out, "align.json")))
    ents = ref_entries(A.ref)
    scenes = [s for s in sorted(ents) if s in al and "error" not in al[s]]
    if A.limit:
        scenes = scenes[:A.limit]
    # SCVideo dataset_large.py:23-40 의 Concerto transform 그대로
    pc_transform = concerto.transform.Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "normal"]}),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"), dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse", "offset"), feat_keys=("coord", "color", "normal"))])
    R_y2z = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    pc_encoder = PCEncoder()
    t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=dev, checkpoint_path=A.t5_ckpt, tokenizer_path=A.t5_tok, shard_fn=None)
    model = CameraDiffusionModel(cam_dim=64)
    print("(wd) ckpt", model.load_state_dict(torch.load(A.ckpt, map_location="cpu", weights_only=True), strict=True), flush=True)
    model = model.to(dev).eval()
    vae = CameraVAE(latent_dim=64); print("(wd) vae", vae.load_state_dict(torch.load(A.vae, map_location="cpu"), strict=True), flush=True)
    vae = vae.to(dev).eval()
    sch = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon", beta_schedule="scaled_linear", clip_sample=False,
                        set_alpha_to_one=False, steps_offset=1, beta_start=0.00085, beta_end=0.012)
    os.makedirs(A.out, exist_ok=True)
    for si, sc in enumerate(scenes):
        idxs = [i for i in ents[sc] if not osp.isfile(osp.join(A.out, f"worlddirector__{sc}__{i}.npz"))]
        if not idxs:
            continue
        a = al[sc]; wt = wt_dir(sc); o = a["offset"]
        E, K = wt_cameras(osp.join(wt, "cameras.json"))
        P, Col = load_ply(osp.join(wt, "scene.ply"))
        # dataset_large.py:255-293 + data_utils.normalize_camera_extrinsics_and_points (첫 카메라 = WT cameras[offset])
        e0 = torch.tensor(E[o], dtype=torch.float64)
        pc = torch.tensor(P) @ e0[:3, :3].T + e0[:3, 3]
        avg_scale = pc.norm(dim=-1).mean()
        npc = pc / avg_scale
        mask = npc.norm(dim=1) <= 1.25
        pts = npc[mask].float()                                            # 정규화 점 (×1.8·회전 전) = embed 의 voxel 좌표 원천
        coord = (pts * 1.8) @ R_y2z
        np.random.seed(A.seed); torch.manual_seed(A.seed)
        pobj = pc_transform({"coord": coord.numpy(), "color": Col[mask.numpy()] * 255.0, "offset": np.array([coord.shape[0]]),
                             "normal": np.zeros_like(coord.numpy())})
        batch = dict(pobj)          # 샘플 1 개면 base.point_collate_fn 결과와 같다 (cat 항등, offset cumsum 항등). base.py 는 학습 데이터 경로를 import 해서 못 쓴다
        inv = pobj["inverse"]
        Kt = torch.tensor(K[o], dtype=torch.float32)[None]
        ray = get_ray_sim_per_point(torch.eye(4)[None], Kt, scatter_mean(pts, inv, dim=0), c2w=False, device=dev)
        with torch.no_grad():
            emb, _, p2v_list, pmask, _ = pc_encoder(batch, [ray])
        p2v = p2v_list[0][inv.to(p2v_list[0].device)]
        vox = scatter_mean(pts.to(dev), p2v, dim=0)
        pe = torch.cat([vox, emb[0][pmask[0]]], -1)[None].float()        # (1, V, 772)
        texts = [json.load(open(osp.join(A.ref, "test", f"dynpose_{sc}_{i}_caption.json")))[A.caption_key] for i in idxs]
        B = len(texts)
        te, tm = t5(texts, dev)
        te, tm = te.float(), tm.bool()
        g = torch.Generator(device=dev).manual_seed(A.seed)
        x = torch.randn(B, 13, 64, device=dev, generator=g)
        pe_b, pm_b = pe.expand(B, -1, -1), torch.ones(B, pe.shape[1], dtype=torch.bool, device=dev)
        sch.set_timesteps(num_inference_steps=50, device=dev)
        with torch.no_grad():
            for t in sch.timesteps:
                eps, _ = model(x, torch.full((B,), t, device=dev).float(), te, tm, pe_b, pm_b)
                x = sch.step(eps, t, x).prev_sample
            out = vae.decode(x * 0.96032625)
            w2c = out_to_trajectory(out, torch.full((B, 1), float(avg_scale), device=dev), torch.eye(4, device=dev)[None].expand(B, -1, -1), dev)
        w2c = w2c.double().cpu().numpy()
        w2c[:, :, :3, 3] /= a["s"]                                          # WT 단위 → DA3 단위
        for b, i in enumerate(idxs):
            c2w_gl = np.linalg.inv(w2c[b]) @ GL2CV
            np.savez(osp.join(A.out, f"worlddirector__{sc}__{i}.npz"), c2w=c2w_gl.astype(np.float32), scale=np.float32(1.0), text=texts[b],
                     avg_scale_wt=float(avg_scale), s=a["s"], offset=o, n_vox=int(pe.shape[1]))
        if si % 20 == 0:
            print(f"[infer {si}/{len(scenes)}] {sc} segs {B} vox {pe.shape[1]} avg_scale_wt {float(avg_scale):.3f} s {a['s']:.3f}", flush=True)
    print(f"=== infer done {len(scenes)} scenes ===", flush=True)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["align", "infer"], required=True)
    ap.add_argument("--out", default=f"{R}/camera_generation/dataset/results/20261023_r527_baseline_v5fall/pred_worlddirector_wt")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--ckpt", default=f"{SCV}/results/ckpts/225.pth")
    ap.add_argument("--vae", default=f"{R}/camera_generation/latentcam/checkpoints/vae_20260302_300.pth")
    ap.add_argument("--t5_ckpt", default=f"{R}/video_generation/models/DiffSynth-Studio/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth")
    ap.add_argument("--t5_tok", default=f"{R}/video_generation/models/DiffSynth-Studio/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl")
    ap.add_argument("--caption_key", default="Concise Interaction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    {"align": stage_align, "infer": stage_infer}[A.stage]()


if __name__ == "__main__":
    main()
