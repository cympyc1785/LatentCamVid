"""이 코드가 답하는 질문: 카메라 궤적 VAE 를 우리 카메라(Dataset-v1 cam_param + worldtraj cameras.json)로 학습/finetune 하면
roundtrip 떨림(jerk)·끝 프레임 오차가 줄어드나?

배경 (R113, 2026-09-30): 모델 없이 GT latent 를 decode 만 해도 d200 test 회전 jerk 가 GT 의 7배 (0.1953 vs 0.0275),
of29a 는 끝 프레임이 화면 15% 벗어난다. 지금 VAE(`checkpoints/vae_20260302_300.pth`, models/vae_intr_large.CameraVAE,
latent 64 x 13)는 이 레포 밖에서 fx/w≈1 분포로 학습됐고 학습 코드가 없다 → 여기서 새로 학습한다.

입력 표현은 **확산 학습과 글자 그대로 같다**: 두 데이터 모두 `dataset_dl3dv._target_out` 을 거쳐 (49, 11) cam_param
(rot6d 6 + trans 3 (w2c, frame0 재고정, / norm_scale) + intr 2 (fx/2cx, fy/2cy, frame0 상대)).
  - Dataset-v1 : D262 run config 로 데이터셋을 만들어 __getitem__ 의 cam_param 을 그대로 캐시 (train / test 분할 그대로).
  - worldtraj  : DATA/worldtraj/dynamicverse/*/.../cameras.json (rotation+position = **w2c** R, t · fx fy cx cy).
                 [R169] R125~R168 캐시(tmp/r125/cache)는 c2w 로 잘못 읽었다 (--wt_pose c2w 로 재현). 레포 규약은 w2c
                 (main/dynamicverse_shim.py, CamVLA data/worldtraj) 이고 scene.ply 투영도 w2c 일 때만 images_pc_rendered 와 맞는다.
                 49 프레임 창. [R175] 분모 = scene scale avg_scale (scene.ply 점 - 창 첫 카메라 평균 거리, Dataset-v1 과 같은 정의,
                 stage_wt_scale). R125~R170 은 점군이 없다고 보고 창 평균 카메라 이동거리 s / m1 로 나눴다 (--wt_norm cam_disp).
                 Dataset-v1 **test 씬**(dynpose-100k uuid)은 누수 방지로 뺀다.

단계:
  cache_v1 | wt_scale | cache_wt | split | train | eval | edge ([R176] 시작·끝 속도 감쇠) | scale ([R173] vae_latent_scale = 학습 데이터 mu std)
  python scripts/vae/train_camera_vae.py --stage split                        # [R135] 합본 0.9/0.1 → split_joint.json (train 기본 --split joint)
  python scripts/vae/train_camera_vae.py --stage cache_wt --wt_split test     # Dataset-v1 test 씬의 worldtraj 카메라 → wt_test.npz
  python scripts/vae/train_camera_vae.py --stage eval --eval_vaes "old=checkpoints/vae_20260302_300.pth:nearest:0,..."
  python scripts/vae/train_camera_vae.py --stage cache_v1,cache_wt
  python scripts/vae/train_camera_vae.py --stage train --name scratch          # 기본 = old 레시피 (lr 1e-4 constant, wd 0.01, L1, kl 1e-3)
  python scripts/vae/train_camera_vae.py --stage train --name ft --init_ckpt checkpoints/vae_20260302_300.pth
"""
import argparse
import importlib.util
import json
import math
import os
import os.path as osp
import sys
import time

import numpy as np

LC = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
MAIN = osp.join(LC, "main")
for p in (MAIN, LC):
    if p not in sys.path:
        sys.path.insert(0, p)
WT = "/data1/cympyc1785/LatentCamVid/DATA/worldtraj/dynamicverse"
D262 = osp.join(LC, "results", "20260922_155726_dynpose_d200_molmo2_l21_srccam")


def load_dataset():
    spec = importlib.util.spec_from_file_location("evt", osp.join(LC, "scripts", "eval_testset.py"))
    evt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evt)
    cfg, _ = evt.build_cfg(D262, {})
    os.chdir(MAIN)
    from base import Trainer
    tr = Trainer(cfg=cfg)
    tr._make_batch_generator(include_train=True)
    return cfg, tr


def stage_cache_v1(a):
    import torch
    from torch.utils.data import DataLoader
    cfg, tr = load_dataset()
    for split, sub in (("train", tr.trainset_loader), ("test", tr.validset_loader)):
        dl = DataLoader(sub, batch_size=64, num_workers=16,
                        collate_fn=lambda b: [(x["data_name"], x["cam_param"].float().numpy()) for x in b])
        names, cams, t0 = [], [], time.time()
        for i, batch in enumerate(dl):
            for n, c in batch:
                names.append(n)
                cams.append(c)
            if i % 100 == 0:
                print(f"[v1 {split}] {len(names)}/{len(sub)}  {time.time() - t0:.0f}s", flush=True)
        np.savez(osp.join(a.cache, f"v1_{split}.npz"), names=np.array(names), cam=np.stack(cams).astype(np.float32))
        print(f"[v1 {split}] {len(names)} -> {a.cache}", flush=True)
    del torch


def centers_from_cam(cam):
    a1, a2 = cam[..., 0:3], cam[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], -1)                       # 열 = b1,b2,b3 (rot6d = 앞 두 열)
    t = cam[..., 6:9]
    return -np.einsum("...ji,...j->...i", R, t)


def wt_files(a):
    """worldtraj cameras.json 이 있는 (subset, root) 목록. 디렉토리 순회가 ~20 분이라 cache/wt_files.json 에 저장해 재사용."""
    fp = osp.join(a.cache, "wt_files.json")
    if osp.isfile(fp):
        return [tuple(x) for x in json.load(open(fp))]
    files = []
    for sub in sorted(os.listdir(WT)):
        if sub == "eval_index":
            continue
        for root, dirs, fs in os.walk(osp.join(WT, sub)):
            if root.count(os.sep) - WT.count(os.sep) > 3:
                dirs[:] = []
            if "cameras.json" in fs:
                files.append((sub, root))
    json.dump(files, open(fp, "w"))
    return files


def wt_window_starts(n, a, T=49):
    return np.unique(np.linspace(0, n - T, min(a.wt_windows, n - T + 1)).round().astype(int))


def _wt_scale_clip(job):
    """clip 하나의 창별 avg_scale = mean || R_w2c(s0) p + t_w2c(s0) ||  (p = scene.ply 점). CamVLA normalize_points ·
    worldtraj avg_scale/{k}.json (s0 = 49k) 와 같은 정의 — 0cb4361a/0 · 54da980a/0,49 에서 파일 값과 일치 확인 (R175)."""
    sub, root, wt_windows = job
    try:
        fr = sorted(json.load(open(osp.join(root, "cameras.json"))), key=lambda f: f["idx"])
        if len(fr) < 49 or not osp.isfile(osp.join(root, "scene.ply")):
            return []
        TY = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "char": "i1", "int": "<i4", "uint": "<u4",
              "short": "<i2", "ushort": "<u2"}
        with open(osp.join(root, "scene.ply"), "rb") as f:
            n, props, in_vertex = 0, [], False
            while True:
                l = f.readline().decode().strip()
                if l.startswith("format") and "binary_little_endian" not in l:
                    return []
                if l.startswith("element"):
                    in_vertex = l.split()[1] == "vertex"
                    if in_vertex:
                        n = int(l.split()[2])
                elif l.startswith("property") and in_vertex:
                    props.append((l.split()[-1], TY[l.split()[1]]))
                if l.startswith("end_header"):
                    break
            dt = np.dtype(props)
            v = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        out = []
        for s0 in np.unique(np.linspace(0, len(fr) - 49, min(wt_windows, len(fr) - 48)).round().astype(int)):
            R, t = np.array(fr[s0]["rotation"]), np.array(fr[s0]["position"])
            out.append((f"{sub}/{osp.basename(root)}/{s0}", float(np.linalg.norm(P @ R.T + t, axis=1).mean())))
        return out
    except Exception:
        return []


def stage_wt_scale(a):
    """이 코드가 답하는 질문: worldtraj 창마다 scene scale (Dataset-v1 transforms 의 avg_scale 과 같은 정의) 은 얼마인가?
    → cache/wt_avg_scale.json {창 이름: avg_scale}. cache_wt --wt_norm avg_scale 가 이것으로 나눈다 (R175)."""
    from multiprocessing import Pool
    files = wt_files(a)
    res = {}
    with Pool(a.workers) as pool:
        for k, out in enumerate(pool.imap_unordered(_wt_scale_clip, [(s_, r_, a.wt_windows) for s_, r_ in files], chunksize=16)):
            res.update(out)
            if k % 5000 == 0:
                print(f"[wt_scale] {k}/{len(files)} 창 {len(res)}", flush=True)
    json.dump(res, open(osp.join(a.cache, "wt_avg_scale.json"), "w"))
    v = np.array(list(res.values()))
    print(f"[wt_scale] 창 {len(res)} · avg_scale median {np.median(v):.4f} p5 {np.percentile(v, 5):.4f} p95 {np.percentile(v, 95):.4f}", flush=True)


def stage_cache_wt(a):
    import torch
    cfg, tr = load_dataset()
    ds = tr.trainset_loader.dataset
    v1 = np.load(osp.join(a.cache, "v1_train.npz"))
    c1 = centers_from_cam(v1["cam"])
    m1 = float(np.median(np.linalg.norm(c1 - c1[:, :1], axis=-1).mean(1)))
    test_uuid = {n[len("dynpose_"):].rsplit("_", 1)[0] for n in np.load(osp.join(a.cache, "v1_test.npz"))["names"]}
    print(f"[wt] Dataset-v1 train 창 평균 이동거리 중앙값 m1 = {m1:.4f} · test uuid {len(test_uuid)} 제외", flush=True)
    files = wt_files(a)
    print(f"[wt] cameras.json {len(files)}", flush=True)
    if a.wt_norm == "avg_scale":                       # [R175] scene scale (Dataset-v1 과 같은 정의) — stage_wt_scale 결과
        wsc = json.load(open(osp.join(a.cache, "wt_avg_scale.json")))
        print(f"[wt] norm = avg_scale (scene.ply 점 - 창 첫 카메라 평균 거리), {len(wsc)} 창", flush=True)
    names, cams, skipped = [], [], {"short": 0, "static": 0, "test": 0, "bad": 0}
    T = 49
    for k, (sub, root) in enumerate(files):
        uuid = osp.basename(root)
        is_test = sub == "dynpose-100k" and uuid in test_uuid
        if is_test != (a.wt_split == "test"):          # train: test 씬 제외 (기존) / test: test 씬만
            skipped["test"] += 1
            continue
        try:
            fr = json.load(open(osp.join(root, "cameras.json")))
            fr = sorted(fr, key=lambda f: f["idx"])
            Rm = np.array([f["rotation"] for f in fr], np.float64)
            pos = np.array([f["position"] for f in fr], np.float64)
            K = np.zeros((len(fr), 3, 3))
            K[:, 0, 0] = [f["fx"] for f in fr]
            K[:, 1, 1] = [f["fy"] for f in fr]
            K[:, 0, 2] = [f["cx"] for f in fr]
            K[:, 1, 2] = [f["cy"] for f in fr]
            K[:, 2, 2] = 1
        except Exception:
            skipped["bad"] += 1
            continue
        n = len(fr)
        if n < T:
            skipped["short"] += 1
            continue
        if a.wt_pose == "w2c":                      # [R169] rotation/position = R_w2c / t_w2c (레포 규약)
            w2c = np.tile(np.eye(4), (n, 1, 1))
            w2c[:, :3, :3], w2c[:, :3, 3] = Rm, pos
            pos = -np.einsum("nji,nj->ni", Rm, pos)          # 카메라 중심 (정지 판정·s 용)
        else:                                          # R125~R168 캐시 재현용 (c2w 로 잘못 읽음)
            c2w = np.tile(np.eye(4), (n, 1, 1))
            c2w[:, :3, :3], c2w[:, :3, 3] = Rm, pos
            w2c = np.linalg.inv(c2w)
        starts = wt_window_starts(n, a, T)
        for s0 in starts:
            p = pos[s0:s0 + T]
            s = float(np.linalg.norm(p - p[0], axis=-1).mean())
            if s < a.wt_min_disp:
                skipped["static"] += 1
                continue
            if a.wt_norm == "avg_scale":
                ns = wsc.get(f"{sub}/{uuid}/{s0}")
                if ns is None:
                    skipped["noscale"] = skipped.get("noscale", 0) + 1
                    continue
            else:                                      # R125~R170: 창 평균 카메라 이동거리 s 를 v1 중앙값 m1 에 맞춤
                ns = s / m1
            out = ds._target_out(torch.from_numpy(w2c[s0:s0 + T]).float(), torch.from_numpy(K[s0:s0 + T]).float(),
                                 torch.tensor([ns]).float(), 360, 640, "", f"{sub}/{uuid}/{s0}")
            cam = out["cam_param"].float().numpy()
            if not np.isfinite(cam).all():
                skipped["bad"] += 1
                continue
            names.append(f"{sub}/{uuid}/{s0}")
            cams.append(cam)
        if k % 5000 == 0:
            print(f"[wt] {k}/{len(files)} 창 {len(names)} skip {skipped}", flush=True)
    fn = "wt.npz" if a.wt_split == "train" else "wt_test.npz"
    np.savez(osp.join(a.cache, fn), names=np.array(names), cam=np.stack(cams).astype(np.float32), m1=m1)
    print(f"[wt] 창 {len(names)}  skip {skipped} -> {a.cache}/{fn}", flush=True)


def w2c_from_cam(cam):
    """(…,T,11) cam_param → w2c R (…,T,3,3) 와 카메라 중심 c (…,T,3). centers_from_cam 과 같은 Gram-Schmidt."""
    a1, a2 = cam[..., 0:3], cam[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    R = np.stack([b1, b2, np.cross(b1, b2)], -1)
    return R, -np.einsum("...ji,...j->...i", R, cam[..., 6:9])


def late_errors(rec, gt, static_frac=0.1, smooth_k=9):
    """[R147] 이 코드가 답하는 질문: 궤적 후반(마지막 1/3)에서 GT 와 위치·속도가 얼마나 다른가 — traj_time_metrics 의 late / rpe1_late /
    rpe1_rot_late 와 같은 정의를 cam_param 공간에서 (worldtraj test 처럼 eval JSON 이 없는 세트에도 같은 식으로 재려고).
      late       = mean_{t∈late} |c_pred - c_gt| / reach,   reach = max_t |c_gt(t) - c_gt(0)|
      rpe1_late  = mean |R_t (c_{t+1}-c_t) [pred] - (같은 것) [gt]| / reach     (t 시점 카메라 좌표계의 한 프레임 이동 = 속도)
      rpe1_rot_late = mean geodesic(R_t R_{t+1}^T [pred], [gt])  deg            (한 프레임 회전 = 각속도)
    정지 창 (reach < static_frac x 세트 중앙값) 은 reach 정규화가 터지므로 뺀다 (traj_time_metrics --min_reach_frac 와 같다)."""
    Rp, cp = w2c_from_cam(rec); Rg, cg = w2c_from_cam(gt)
    T = gt.shape[1]; t0 = np.array_split(np.arange(T), 3)[2][0]
    reach = np.linalg.norm(cg - cg[:, :1], axis=-1).max(1)
    mv = reach >= static_frac * np.median(reach)
    r = reach[mv][:, None]
    ate = np.linalg.norm(cp - cg, axis=-1)[mv] / r
    vp = np.einsum("ntij,ntj->nti", Rp[:, :-1], cp[:, 1:] - cp[:, :-1]); vg = np.einsum("ntij,ntj->nti", Rg[:, :-1], cg[:, 1:] - cg[:, :-1])
    rpe = np.linalg.norm(vp - vg, axis=-1)[mv] / r
    Mp = np.einsum("ntij,ntkj->ntik", Rp[:, :-1], Rp[:, 1:]); Mg = np.einsum("ntij,ntkj->ntik", Rg[:, :-1], Rg[:, 1:])
    D = np.einsum("ntji,ntjk->ntik", Mp, Mg)
    rot = np.degrees(np.arccos(np.clip((np.trace(D, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))[mv]
    out = {"late": float(ate[:, t0:].mean()), "FDE": float(ate[:, -1].mean()), "rpe1_late": float(rpe[:, t0:].mean()),
           "rpe1_rot_late": float(rot[:, t0:].mean()), "n_moving": int(mv.sum())}
    # [R148] GT 잡음에 덜 휘둘리는 지표 (worldtraj GT 는 프레임간 떨림이 이동보다 커서 GT 와의 프레임간 차이는 "잡음 재현도" 가 된다):
    #   *_sGT   : GT 를 시간축 이동평균 (k=smooth_k, 끝은 replicate) 한 궤적과 비교 — 바탕 궤적을 따라가는가
    #   jerk_t_late / jerk_r_late : 복원 궤적 자체의 후반 떨림 (traj_time_metrics 와 같은 정의, GT 무관)
    #   hf_frac : 위치 속도(Δ¹) 파워 중 f > 0.2 cycles/frame 비율 (복원 자체, 높을수록 떨림)
    k = smooth_k; pad = k // 2
    cgp = np.concatenate([np.repeat(cg[:, :1], pad, 1), cg, np.repeat(cg[:, -1:], pad, 1)], 1)
    cgs = np.stack([cgp[:, i:i + T] for i in range(k)], 0).mean(0)
    out["FDE_sGT"] = float((np.linalg.norm(cp[:, -1] - cgs[:, -1], axis=-1)[mv] / reach[mv]).mean())
    out["vel_late_sGT"] = float((np.linalg.norm(np.diff(cp, axis=1) - np.diff(cgs, axis=1), axis=-1)[mv] / r)[:, t0:].mean())
    out["jerk_t_late"] = float((np.linalg.norm(np.diff(cp, n=3, axis=1), axis=-1)[mv] / r)[:, t0:].mean())
    M = Mp.reshape(-1, 3, 3)
    ang = np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1))
    ax = np.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]], 1)
    nrm = np.linalg.norm(ax, axis=1, keepdims=True)
    om = (np.where(nrm > 1e-9, ax / np.maximum(nrm, 1e-9), 0) * np.degrees(ang)[:, None]).reshape(Mp.shape[0], -1, 3)
    out["jerk_r_late"] = float(np.linalg.norm(np.diff(om, n=2, axis=1), axis=-1)[mv][:, t0:].mean())
    # [R150] 후반 가속도 오차: acc_late = |Δ²c_pred - Δ²c_gt| / reach (위치), acc_rot_late = |Δω_pred - Δω_gt| deg/frame² (회전,
    #   ω = 이웃 프레임 상대 회전의 회전 벡터 — jerk_r 과 같은 ω). 마지막 1/3 (차분 창이 그 구간 안).
    Mg2 = Mg.reshape(-1, 3, 3)
    angg = np.arccos(np.clip((np.trace(Mg2, axis1=1, axis2=2) - 1) / 2, -1, 1))
    axg = np.stack([Mg2[:, 2, 1] - Mg2[:, 1, 2], Mg2[:, 0, 2] - Mg2[:, 2, 0], Mg2[:, 1, 0] - Mg2[:, 0, 1]], 1)
    ng = np.linalg.norm(axg, axis=1, keepdims=True)
    omg = (np.where(ng > 1e-9, axg / np.maximum(ng, 1e-9), 0) * np.degrees(angg)[:, None]).reshape(Mg.shape[0], -1, 3)
    out["acc_late"] = float((np.linalg.norm(np.diff(cp, n=2, axis=1) - np.diff(cg, n=2, axis=1), axis=-1)[mv] / r)[:, t0:].mean())
    out["acc_rot_late"] = float(np.linalg.norm(np.diff(om, axis=1) - np.diff(omg, axis=1), axis=-1)[mv][:, t0:].mean())
    v = np.diff(cp, axis=1)[mv]; P = (np.abs(np.fft.rfft(v, axis=1)) ** 2).sum(-1); f = np.fft.rfftfreq(v.shape[1])
    out["hf_frac"] = float((P[:, f > 0.2].sum(1) / P.sum(1).clip(1e-12)).mean())
    # [R149] jerk 가 아닌 떨림 지표 (복원 궤적 자체):
    #   p4_frac  : 위치 속도 파워 중 업샘플 주기 성분 (f≈0.25, 0.5 cycles/frame 의 bin) 비율 — 4프레임 톱니의 세기
    #   zigzag   : 이웃 가속도 벡터가 반대 방향 (a_t·a_{t+1} < 0) 인 프레임 비율 — 갈지자로 꺾이는 빈도 (위치 / 회전)
    bins = [int(np.argmin(np.abs(f - 0.25))), int(np.argmin(np.abs(f - 0.5)))]
    out["p4_frac"] = float((P[:, bins].sum(1) / P.sum(1).clip(1e-12)).mean())
    acc = np.diff(cp, n=2, axis=1)[mv]
    out["zigzag_t"] = float(((acc[:, 1:] * acc[:, :-1]).sum(-1) < 0).mean())
    aom = np.diff(om, axis=1)[mv]
    out["zigzag_r"] = float(((aom[:, 1:] * aom[:, :-1]).sum(-1) < 0).mean())
    return out


def jerk(x):
    """(B,T,C) -> 창별 평균 |Δ³x| (rot6d+trans 채널만)."""
    return np.abs(np.diff(x[..., :9], n=3, axis=1)).mean(axis=(1, 2))


def stage_split(a):
    """이 코드가 답하는 질문: VAE 학습용 worldtraj + Dataset-v1 합본을 0.9 / 0.1 로 (두 데이터 모두 test 에 들어가게) 어떻게 나누나?
    사용자 2026-10-01: "worldtraj + Dataset_v1 합친거에서 train, test(두 데이터셋 모두 포함되게) 0.9, 0.1 비율 맞춰서 split".
      - Dataset-v1 : 확산 학습의 train/test 분할을 그대로 쓴다 (46,278 / 5,144 = 0.900 / 0.100).
      - worldtraj  : 클립(cameras.json) 단위. 같은 영상이 두 출처에 걸쳐 train/test 로 갈리지 않도록 dynpose-100k 클립은
                     Dataset-v1 test uuid → test, train uuid → train 으로 고정하고, 나머지 클립을 출처별로 무작위로 뽑아
                     출처마다 test 창 비율이 test_frac 이 되게 채운다.
    출력: cache/split_joint.json (wt_all = wt.npz ++ wt_test.npz 순서의 창 index)."""
    rng = np.random.default_rng(a.split_seed)
    v1tr = {str(n)[len("dynpose_"):].rsplit("_", 1)[0] for n in np.load(osp.join(a.cache, "v1_train.npz"))["names"]}
    v1te = {str(n)[len("dynpose_"):].rsplit("_", 1)[0] for n in np.load(osp.join(a.cache, "v1_test.npz"))["names"]}
    names = [str(n) for f in ("wt.npz", "wt_test.npz") for n in np.load(osp.join(a.cache, f))["names"]]
    clips = {}                                         # (src, uuid) -> [창 index]
    for i, n in enumerate(names):
        src, uuid, _ = n.split("/")
        clips.setdefault((src, uuid), []).append(i)
    test, report = [], {}
    for src in sorted({k[0] for k in clips}):
        ks = [k for k in clips if k[0] == src]
        n_win = sum(len(clips[k]) for k in ks)
        forced_te = [k for k in ks if src == "dynpose-100k" and k[1] in v1te]
        forced_tr = {k for k in ks if src == "dynpose-100k" and k[1] in v1tr}
        free = [k for k in ks if k not in forced_tr and k not in set(forced_te)]
        sel, got = list(forced_te), sum(len(clips[k]) for k in forced_te)
        for j in rng.permutation(len(free)):
            if got >= a.test_frac * n_win:
                break
            sel.append(free[j])
            got += len(clips[free[j]])
        test += [i for k in sel for i in clips[k]]
        report[src] = {"clips": len(ks), "windows": n_win, "test_clips": len(sel), "test_windows": got,
                       "forced_test_clips": len(forced_te), "forced_train_clips": len(forced_tr), "test_frac": got / n_win}
        print(f"[split] {src:16s} 클립 {len(ks):6d} 창 {n_win:6d} → test 클립 {len(sel):5d} 창 {got:6d} ({got / n_win:.4f})"
              f"  고정 test {len(forced_te)} / 고정 train {len(forced_tr)}", flush=True)
    test = sorted(test)
    tset = set(test)
    train = [i for i in range(len(names)) if i not in tset]
    nv_tr = len(np.load(osp.join(a.cache, "v1_train.npz"))["names"])
    nv_te = len(np.load(osp.join(a.cache, "v1_test.npz"))["names"])
    tot_tr, tot_te = len(train) + nv_tr, len(test) + nv_te
    print(f"[split] worldtraj train {len(train)} / test {len(test)} ({len(test) / len(names):.4f}) · Dataset-v1 {nv_tr} / {nv_te} "
          f"({nv_te / (nv_tr + nv_te):.4f}) · 합 {tot_tr} / {tot_te} ({tot_te / (tot_tr + tot_te):.4f})", flush=True)
    json.dump({"seed": a.split_seed, "test_frac": a.test_frac, "wt_order": ["wt.npz", "wt_test.npz"],
               "wt_train": train, "wt_test": test, "report": report}, open(osp.join(a.cache, "split_joint.json"), "w"))
    print(f"[split] -> {a.cache}/split_joint.json", flush=True)


def load_joint(a):
    """split_joint.json 기준 (wt_train, {all, wt, v1} test 세트, test 이름) 을 numpy 로."""
    sp = json.load(open(osp.join(a.cache, "split_joint.json")))
    wt = [np.load(osp.join(a.cache, f)) for f in sp["wt_order"]]
    wcam = np.concatenate([d["cam"] for d in wt])
    wnames = [str(n) for d in wt for n in d["names"]]
    v1 = np.load(osp.join(a.cache, "v1_test.npz"))
    te_wt, te_v1 = wcam[sp["wt_test"]], v1["cam"]
    sets = {"all": np.concatenate([te_v1, te_wt]), "wt": te_wt, "v1": te_v1}
    names = [str(n) for n in v1["names"]] + [wnames[i] for i in sp["wt_test"]]
    return wcam[sp["wt_train"]], sets, names


def roundtrip_metrics(model, X, dev="cuda"):
    """(N,49,11) 창 roundtrip -> stage_train 의 val/* 와 같은 정의의 지표 dict (+ 창별 jerk_ratio 중앙값)."""
    import torch
    model.eval()
    with torch.no_grad():
        mu = torch.cat([model.encode(X[i:i + 1024].to(dev)) for i in range(0, len(X), 1024)])
        rec = torch.cat([model.decode(mu[i:i + 1024]) for i in range(0, len(mu), 1024)]).cpu()
    err = (rec - X) ** 2
    rj, gj = jerk(rec.numpy()), jerk(X.numpy())
    with torch.no_grad():                              # [R138] posterior σ 채널별 평균 → 활성 채널 (σ<0.5) 수·그 채널의 mu std (latent scale 후보)
        sig = torch.cat([torch.exp(0.5 * model.encoder(model._pad(X[i:i + 1024].to(dev)))[1]).cpu()
                         for i in range(0, len(X), 1024)])          # (N, D, T')
    ch_sig = sig.mean((0, 2))
    act = ch_sig < 0.5
    mu_d = mu.cpu().permute(2, 0, 1).reshape(mu.shape[-1], -1)       # (D, N*T')
    extra = {"n_active": int(act.sum()), "mu_std_active": float(mu_d[act].std()) if act.any() else 0.0,
             "sigma_mean": float(sig.mean()), "z_sample_std": float((mu_d.var() + sig.pow(2).mean()).sqrt())}
    extra.update(late_errors(rec.numpy().astype(np.float64), X.numpy().astype(np.float64)))
    return {**extra, "mse": err.mean().item(), "mse_rot": err[..., :6].mean().item(), "mse_trans": err[..., 6:9].mean().item(),
            "mse_intr": err[..., 9:].mean().item(), "mse_last_frame": err[:, -1].mean().item(),
            "jerk": float(rj.mean()), "jerk_gt": float(gj.mean()), "jerk_ratio": float(rj.mean() / gj.mean()),
            "jerk_ratio_med": float(np.median(rj / np.maximum(gj, 1e-9))), "latent_std": mu.std().item(), "n": len(X)}


def stage_eval(a):
    """이 코드가 답하는 질문: 학습된 VAE 들의 roundtrip 성능이 평가 세트(Dataset-v1 test / worldtraj test 씬)에 따라 경향이 달라지나?
    --eval_vaes "이름=ckpt:up_mode:pad_end[:pad_mode],..." · --eval_sets "v1_test,wt_test" (cache 의 npz 이름)."""
    import torch
    from models.vae_intr_large import CameraVAE
    sets = {}
    for k in a.eval_sets.split(","):
        if k == "joint":                               # split_joint.json 의 test: 모두 / worldtraj / Dataset-v1
            sets.update({f"joint_{n}": torch.from_numpy(x).float() for n, x in load_joint(a)[1].items()})
        else:
            sets[k] = torch.from_numpy(np.load(osp.join(a.cache, f"{k}.npz"))["cam"]).float()
    rows = []
    for spec in a.eval_vaes.split(","):
        nm, rest = spec.split("=", 1)
        parts = rest.split(":")                        # ckpt:up:pad[:pad_mode]
        ck, up, pad = parts[0], parts[1], parts[2]
        pm = parts[3] if len(parts) > 3 else "zeros"
        model = CameraVAE(latent_dim=64, up_mode=up, pad_end=int(pad), pad_mode=pm).cuda()
        model.load_state_dict(torch.load(ck, map_location="cuda"))
        for sk, X in sets.items():
            m = roundtrip_metrics(model, X)
            rows.append({"vae": nm, "set": sk, **m})
            print(f"[eval] {nm:18s} {sk:8s} " + " ".join(f"{k}={v:.5g}" for k, v in m.items()), flush=True)
    if a.eval_out:
        json.dump(rows, open(a.eval_out, "w"), indent=1)
        print(f"[eval] -> {a.eval_out}", flush=True)


def train_windows(a):
    """stage_train 의 학습 창 = v1_train (+ worldtraj train) (+ overfit 두 영상의 test 세그먼트). (tr 리스트, extra_sets, Xte, te_names, inc)."""
    import torch
    extra_sets = {}
    # [R181] --use_v1 0 = 학습 창에서 Dataset-v1 train 을 뺀다 (worldtraj 만, old VAE 와 공정 비교용). test 는 그대로.
    tr = [np.load(osp.join(a.cache, "v1_train.npz"))["cam"][: (None if a.use_v1 else 0)]]
    if a.split == "joint":
        # [R135] worldtraj + Dataset-v1 합본 0.9/0.1 (stage_split). val/* = 합본 test, val_wt/* · val_v1/* = 출처별
        wt_tr, jsets, _ = load_joint(a)
        if a.use_wt:
            tr.append(wt_tr)
        extra_sets = {"val_wt": torch.from_numpy(jsets["wt"]).float()}
    elif a.use_wt:
        tr.append(np.load(osp.join(a.cache, "wt.npz"))["cam"])
    te = np.load(osp.join(a.cache, "v1_test.npz"))
    Xte, te_names = torch.from_numpy(te["cam"]).float(), [str(n) for n in te["names"]]
    # [사용자 2026-09-30 "두 영상의 카메라 모두 VAE train 때 VAE가 볼 수 있게"] overfit 두 영상(test split)의
    #   전 세그먼트를 train 에 넣는다. val 에도 그대로 남겨 watch 로 본다 (그 두 씬만 held-out 이 아니다).
    inc = [i for i, n in enumerate(te_names) if any(u in n for u in a.include_test_scenes.split(",") if u)]
    if inc:
        tr.append(te["cam"][inc])
    return tr, extra_sets, Xte, te_names, inc


def stage_scale(a):
    """이 코드가 답하는 질문: 이 VAE 로 diffusion 을 학습할 때 vae_latent_scale (= 그 VAE 학습 데이터의 latent std) 은 얼마인가?
    R164 · R166 에서 인라인으로 두 번 계산한 것을 승격 (R173). 학습 창 = stage_train 과 같은 train_windows(a).
      mu_std       : encode (mu) 전 원소 std  — vae_encode_sample=false (기본) 일 때 쓰는 값
      z_sample_std : sqrt(var(mu) + mean σ²) — encode_sample 일 때
    --eval_vaes "이름=ckpt:up:pad[:pad_mode],..." · --eval_out <json>"""
    import torch
    from models.vae_intr_large import CameraVAE
    tr, *_ = train_windows(a)
    X = torch.from_numpy(np.concatenate(tr)).float()
    out = {}
    for spec in a.eval_vaes.split(","):
        nm, rest = spec.split("=", 1)
        parts = rest.split(":")
        model = CameraVAE(latent_dim=64, up_mode=parts[1], pad_end=int(parts[2]), pad_mode=parts[3] if len(parts) > 3 else "zeros").cuda().eval()
        model.load_state_dict(torch.load(parts[0], map_location="cuda"))
        with torch.no_grad():
            ml = [model.encoder(model._pad(X[i:i + 4096].cuda())) for i in range(0, len(X), 4096)]
        mu = torch.cat([m[0].cpu() for m in ml]); var = torch.cat([torch.exp(m[1]).cpu() for m in ml])
        out[nm] = {"vae": parts[0], "n": len(X), "mu_std": float(mu.std()), "z_sample_std": float((mu.var() + var.mean()).sqrt())}
        print(f"[scale] {nm:12s} n={len(X)} mu_std={out[nm]['mu_std']:.6f} z_sample_std={out[nm]['z_sample_std']:.6f}", flush=True)
    if a.eval_out:
        json.dump(out, open(a.eval_out, "w"), indent=1)
        print(f"[scale] -> {a.eval_out}", flush=True)


def stage_edge(a):
    """이 코드가 답하는 질문: VAE roundtrip 이 궤적의 시작·끝 프레임에서 속도를 얼마나 깎나 (replicate padding 끝단 감쇠)?
    R169 의 인라인 측정(tmp/r169/edge.py, Dataset-v1 test)을 승격 (R176). 프레임별 |v_pred| / |v_gt| 의 중앙값 — 위치 (카메라 중심),
    회전 (이웃 프레임 상대 회전각). 정지에 가까운 창 (평균 이동 < 0.1 x 중앙값 또는 평균 회전 < 1e-3 deg) 은 뺀다.
    --eval_vaes "이름=ckpt:up:pad[:pad_mode],..." · --eval_sets (기본 v1_test) · --eval_out <json>"""
    import torch
    from models.vae_intr_large import CameraVAE

    def speeds(cam):
        R, c = w2c_from_cam(cam.astype(np.float64))
        M = np.einsum("ntij,ntkj->ntik", R[:, :-1], R[:, 1:])
        return (np.linalg.norm(np.diff(c, axis=1), axis=-1),
                np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1))))
    out = {}
    for sk in a.eval_sets.split(","):
        X = torch.from_numpy(np.load(osp.join(a.cache, f"{sk}.npz"))["cam"]).float()
        tg, rg = speeds(X.numpy())
        mv = (tg.mean(1) > np.median(tg.mean(1)) * 0.1) & (rg.mean(1) > 1e-3)
        for spec in a.eval_vaes.split(","):
            nm, rest = spec.split("=", 1)
            parts = rest.split(":")
            model = CameraVAE(latent_dim=64, up_mode=parts[1], pad_end=int(parts[2]),
                              pad_mode=parts[3] if len(parts) > 3 else "zeros").cuda().eval()
            model.load_state_dict(torch.load(parts[0], map_location="cuda"))
            with torch.no_grad():
                rec = torch.cat([model.decode(model.encode(X[i:i + 1024].cuda())).cpu() for i in range(0, len(X), 1024)]).numpy()
            tp, rp = speeds(rec)
            rt = np.median(tp[mv] / np.maximum(tg[mv], 1e-9), 0); rr = np.median(rp[mv] / np.maximum(rg[mv], 1e-9), 0)
            out[f"{nm}@{sk}"] = {"trans": rt.tolist(), "rot": rr.tolist(), "n": int(mv.sum())}
            f = lambda v: " ".join(f"{x:.3f}" for x in v)
            print(f"[edge] {nm:10s} {sk:8s} trans first3 {f(rt[:3])} | mid {rt[8:40].mean():.3f} | last3 {f(rt[-3:])}"
                  f"   rot first3 {f(rr[:3])} | mid {rr[8:40].mean():.3f} | last3 {f(rr[-3:])}", flush=True)
    if a.eval_out:
        json.dump(out, open(a.eval_out, "w"))


def stage_train(a):
    import torch
    import torch.nn.functional as F
    from models.vae_intr_large import CameraVAE
    dev = "cuda"
    torch.manual_seed(a.seed)
    tr, extra_sets, Xte, te_names, inc = train_windows(a)
    Xtr = torch.from_numpy(np.concatenate(tr)).float()
    print(f"[train] test→train 포함 {len(inc)} 세그먼트 ({a.include_test_scenes})", flush=True)
    watch = [i for i, n in enumerate(te_names) if n in a.watch.split(",")]
    print(f"[train] {a.name}: train {len(Xtr)} (v1 {len(tr[0])} + wt/포함 {len(Xtr) - len(tr[0])}), test {len(Xte)}, "
          f"watch {[te_names[i] for i in watch]}", flush=True)
    model = CameraVAE(latent_dim=64, up_mode=a.up_mode, pad_end=a.pad_end, pad_mode=a.pad_mode).to(dev)
    if a.init_ckpt:
        model.load_state_dict(torch.load(a.init_ckpt, map_location=dev))
        print(f"[train] init <- {a.init_ckpt}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    start_ep = 0
    if a.resume_dir:                                   # [R139] 같은 결과 폴더에서 이어 학습 (resume.pth 있으면 optimizer 까지, 없으면 last.pth + 새 optimizer)
        rp = osp.join(a.resume_dir, "resume.pth")
        if osp.isfile(rp):
            st = torch.load(rp, map_location=dev)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start_ep = st["epoch"]
            print(f"[train] resume <- {rp} (optimizer 포함), epoch {start_ep} 부터", flush=True)
        else:
            assert a.start_epoch > 0, "resume.pth 가 없으면 --start_epoch (완료 epoch 수) 를 준다"
            model.load_state_dict(torch.load(osp.join(a.resume_dir, "last.pth"), map_location=dev)); start_ep = a.start_epoch
            print(f"[train] resume <- {a.resume_dir}/last.pth (optimizer 상태 없음 → AdamW 새로), epoch {start_ep} 부터", flush=True)
    steps_per_ep = math.ceil(len(Xtr) / a.batch)
    total = a.epochs * steps_per_ep

    def lr_at(step):
        if a.lr_schedule == "constant":                # [R138] old VAE (SCVideo train_vae.py) 와 같은 고정 lr
            return a.lr
        warm = int(0.02 * total)
        if step < warm:
            return a.lr * (step + 1) / warm
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))

    out_dir = a.resume_dir or osp.join(LC, "results", f"{time.strftime('%Y%m%d_%H%M%S')}_camvae_{a.name}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(osp.join(out_dir, "args_resume.json" if a.resume_dir else "args.json"), "w"), indent=1)
    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(project="camera-vae", name=osp.basename(out_dir), config=vars(a),
                         **({"id": a.wandb_id, "resume": "allow"} if a.wandb_id else {}))
    gt_jerk = jerk(Xte.numpy())
    if a.split == "joint":
        extra_sets["val_v1"] = Xte
        Xte = torch.cat([Xte, extra_sets["val_wt"]])   # val/* 는 합본 (v1 앞, watch index 유지)
        gt_jerk = jerk(Xte.numpy())
        print(f"[train] joint test 합 {len(Xte)} = Dataset-v1 {len(extra_sets['val_v1'])} + worldtraj {len(extra_sets['val_wt'])}", flush=True)

    def evaluate():
        model.eval()
        with torch.no_grad():
            mu = torch.cat([model.encode(Xte[i:i + 1024].to(dev)) for i in range(0, len(Xte), 1024)])
            rec = torch.cat([model.decode(mu[i:i + 1024]) for i in range(0, len(mu), 1024)]).cpu()
        model.train()
        err = (rec - Xte) ** 2
        rj = jerk(rec.numpy())
        m = {"val/mse": err.mean().item(), "val/mse_rot": err[..., :6].mean().item(), "val/mse_trans": err[..., 6:9].mean().item(),
             "val/mse_intr": err[..., 9:].mean().item(), "val/mse_last_frame": err[:, -1].mean().item(),
             "val/jerk": float(rj.mean()), "val/jerk_gt": float(gt_jerk.mean()), "val/jerk_ratio": float(rj.mean() / gt_jerk.mean()),
             "val/latent_std": mu.std().item()}
        if a.split == "joint":
            for pre, X in extra_sets.items():
                m.update({f"{pre}/{k}": v for k, v in roundtrip_metrics(model, X, dev).items() if k != "n"})
            model.train()
        for i in watch:
            m[f"watch/{te_names[i]}/mse"] = err[i].mean().item()
            m[f"watch/{te_names[i]}/jerk_ratio"] = float(rj[i] / max(gt_jerk[i], 1e-9))
        return m

    m0 = evaluate() if not a.resume_dir else {}
    print(f"[ep -1] " + " ".join(f"{k}={v:.5g}" for k, v in m0.items() if k.startswith("val/")), flush=True)
    if run and m0:
        run.log({**m0, "epoch": -1}, step=0)
    step, best = start_ep * steps_per_ep, float("inf")
    for ep in range(start_ep, a.epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for bi in range(steps_per_ep):
            x = Xtr[perm[bi * a.batch:(bi + 1) * a.batch]].to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            # [R179] --amp bf16 = CamVLA train_vae.py 의 Accelerator(mixed_precision="bf16") — forward 만 autocast (기본 off = 옛 동작)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
                x_hat, mu, logvar = model(x)
            x_hat, mu, logvar = x_hat.float(), mu.float(), logvar.float()
            rec = F.l1_loss(x_hat, x) if a.recon == "l1" else F.mse_loss(x_hat, x)   # [R138] old VAE = L1
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).mean()
            loss = rec + a.kl_w * kl
            if a.smooth_w > 0:        # 디코드 결과의 3차 차분이 GT 보다 큰 만큼만 벌한다
                jh = (x_hat[:, 3:, :9] - 3 * x_hat[:, 2:-1, :9] + 3 * x_hat[:, 1:-2, :9] - x_hat[:, :-3, :9]).abs()
                jg = (x[:, 3:, :9] - 3 * x[:, 2:-1, :9] + 3 * x[:, 1:-2, :9] - x[:, :-3, :9]).abs()
                loss = loss + a.smooth_w * F.relu(jh - jg).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.grad_clip > 0:                         # [R179] CamVLA train_vae.py clip_grad_norm_(…, 0.5). 0 = 옛 동작 (clip 없음)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            step += 1
            tot += loss.item()
            if run and step % 50 == 0:
                run.log({"train/loss": loss.item(), "train/rec": rec.item(), "train/kl": kl.item(),
                         "train/lr": opt.param_groups[0]["lr"]}, step=step)
        m = evaluate()
        print(f"[ep {ep}] loss {tot / steps_per_ep:.5g} " + " ".join(f"{k}={v:.5g}" for k, v in m.items()
                                                                  if k.split("/")[0] in ("val", "val_wt", "val_v1") and k.split("/")[1] in ("mse", "mse_last_frame", "jerk_ratio")),
              flush=True)
        if run:
            run.log({**m, "epoch": ep}, step=step)
        torch.save(model.state_dict(), osp.join(out_dir, "last.pth"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1}, osp.join(out_dir, "resume.pth"))
        if str(ep + 1) in a.save_epochs.split(","):   # [R139] 중간 스냅샷 (예: 300 epoch scratch 의 100 epoch 시점 → ft 100 epoch 와 대조)
            torch.save(model.state_dict(), osp.join(out_dir, f"epoch{ep + 1}.pth"))
        if m["val/mse"] < best:
            best = m["val/mse"]
            torch.save(model.state_dict(), osp.join(out_dir, "best.pth"))
    json.dump(m, open(osp.join(out_dir, "final_metrics.json"), "w"), indent=1)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cache", default="/data1/cympyc1785/LatentCamVid/tmp/r125/cache")
    ap.add_argument("--wt_windows", type=int, default=3)          # cameras.json 당 49 프레임 창 수 (균등 시작점)
    ap.add_argument("--wt_min_disp", type=float, default=1e-3)    # 창 평균 이동거리가 이보다 작으면 정지 카메라로 보고 제외
    ap.add_argument("--split", default="joint", choices=["joint", "legacy"])   # [R135] joint = split_joint.json (기본) / legacy = v1 test 만 val (R125~R132)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--wt_norm", default="avg_scale", choices=["avg_scale", "cam_disp"])   # [R175] worldtraj translation 분모. cam_disp = R125~R170 (창 이동거리 s / m1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--wt_pose", default="w2c", choices=["w2c", "c2w"])   # [R169] cameras.json 해석. c2w = R125~R168 캐시 (잘못된 해석) 재현
    ap.add_argument("--wt_split", default="train", choices=["train", "test"])   # cache_wt: test = Dataset-v1 test 씬의 cameras.json 만 → wt_test.npz
    ap.add_argument("--eval_vaes", default="")                                    # eval: "이름=ckpt:up_mode:pad_end,..."
    ap.add_argument("--eval_sets", default="v1_test,wt_test")
    ap.add_argument("--eval_out", default="")
    ap.add_argument("--name", default="scratch")
    ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--resume_dir", default="")       # [R139] results/<run> 에서 이어 학습 (같은 폴더에 계속 저장)
    ap.add_argument("--start_epoch", type=int, default=0)  # resume.pth 없을 때 last.pth 의 완료 epoch 수
    ap.add_argument("--wandb_id", default="")         # 이어 쓸 wandb run id
    ap.add_argument("--lr", type=float, default=1e-4)            # [R138] 기본 = old VAE 레시피 (R125~R137 은 scratch 1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save_epochs", default="")                  # [R139] "100,200" → epoch100.pth, epoch200.pth
    ap.add_argument("--batch", type=int, default=256)
    # [R138] 기본값 = old VAE 학습 레시피 (SCVideo 0728999: AdamW lr 1e-4 고정 · wd 0.01(AdamW 기본) · L1 recon · beta 1e-3).
    #   사용자 2026-10-01 "우리도 constant로 해야지. 변수를 통제해야해". R125~R137 레시피 = --lr 1e-3 --lr_schedule cosine --wd 0 --recon mse --kl_w 1e-4
    ap.add_argument("--kl_w", type=float, default=1e-3)
    ap.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--recon", default="l1", choices=["l1", "mse"])
    ap.add_argument("--grad_clip", type=float, default=0.0)            # [R179] CamVLA 레시피 = 0.5
    ap.add_argument("--amp", default="off", choices=["off", "bf16"])   # [R179] CamVLA 레시피 = bf16
    ap.add_argument("--smooth_w", type=float, default=0.0)
    ap.add_argument("--up_mode", default="nearest", choices=["nearest", "linear", "subpixel"])   # [R126] decoder 업샘플 ([R136] subpixel = WAN depth-to-time)
    ap.add_argument("--pad_mode", default="zeros", choices=["zeros", "replicate", "extrap_in"])    # [R136] conv 시간 padding 값 · [R178] extrap_in = 입력 첫 conv 만 직선 연장, 나머지 zeros
    ap.add_argument("--pad_end", type=int, default=0)                                 # [R126] 입력 끝 padding (4 → 토큰 14)
    ap.add_argument("--use_wt", type=int, default=1)
    ap.add_argument("--use_v1", type=int, default=1)            # [R181] 0 = Dataset-v1 train 창 제외 (worldtraj 만)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--include_test_scenes", default="5ec7f200-b057-41f2-b374-9dcff1e7e546,29a9a7d0-7b3f-4520-ab9b-baacae3ca9de")
    ap.add_argument("--watch", default="dynpose_5ec7f200-b057-41f2-b374-9dcff1e7e546_1,dynpose_29a9a7d0-7b3f-4520-ab9b-baacae3ca9de_0")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    for st in a.stage.split(","):
        globals()[f"stage_{st}"](a)
