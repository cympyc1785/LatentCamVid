"""이 코드가 답하는 질문: Dataset-v1 (d200 s91) 을 E.T.(DIRECTOR) 학습용 et-data 로 어떻게 바꾸나? (우리 데이터로 E.T. 를 학습해 대조군으로)

사용자 2026-10-08 "molmo2, srccam 없이 3D position 넣는거면 사실 E.T. 를 학습하는거랑 비슷할 것 같은데 대조군으로 … E.T. 는 우선 세팅만".
형식 근거: tmp/agent/reader-r258-et-format.md (DIRECTOR src/datasets/** · configs/dataset/**).

et-data (= --out) 구성, 파일 stem `<scene>__<idx>` (clatr_dynpose_d200_s91 과 같은 이름):
  mixed_{train,test}_split.txt   이름 목록 (끝 개행 없음 — 빈 이름이 생긴다)
  traj/<n>.txt                   KITTI 3x4 c2w (프레임당 12 값)
  char/<n>.npy, char_raw/<n>.npy (T,3) subject 중심
  caption/<n>.txt                prompts.json concise 캡션
  caption_clip/seq/<n>.npy       (L,512) CLIP ViT-B/32 seq — clatr_dynpose_d200_s91/seq 기존 산출물 심링크
  intrinsics/<n>.npy             (3,3) 소스 K (학습 미사용, 로드만)
  cam_segments/<n>.npy           (T,) int — DIRECTOR CaptionMetrics.segment_camera_trajectories (caption F1 GT, 자기일관)
좌표: target_poses (OpenCV w2c, frame0 = I) → c2w → E.T. 관례 (char0 원점, y-up): W = diag(1,-1,-1),
      t' = W (c − p0), R' = W R, char' = W (p − p0)  (p = target_track subject 중심, p0 = 그 frame0).
stats (--stage stats, train 만): shift = t'[0], norm = diff(t'), norm_h = char 의 [p'0; diff(p')] → mean/std
      → DIRECTOR configs/dataset/standardization/d200s91.yaml (num_cams 49, velocity true).
발산 컷 |c2w| ≥ 1e2 (prepare_clatr_vista.py 와 같다), subject invalid 는 제외.
env: GenDoP (torch · evo · DIRECTOR import). 예: python exec/prepare_et_d200.py --stage convert,stats

[R314] 역방향 (우리 지표로 E.T. 평가, 사용자 2026-10-11 "E.T. 는 결과가 어떻게 됨?"): --stage infer,evaldir
  infer   : 학습 ckpt (d200s91 · ca_director) 로 test split 전량 추론 (seed 42, guidance 1.4, 10 step) → <eval_out>/et_preds.npz
  evaldir : 역변환 c2w = [W·R' | W·t' + p0] (OpenCV) → OpenGL, ref/caption json 은 --ref_eval_dir (우리 eval_testset 산출) 에서 복사.
            arm 둘: <eval_out>_raw (예측 frame0 그대로) · <eval_out>_anc (P = ref0 · inv(P0) · P — 우리 모델처럼 GT 첫 카메라에 붙임).
  역변환은 GT traj 에 적용해 ref json 과 max|diff| 1.7e-07 (tmp/agent/reader-r314-et-eval.md).
"""
import json
import os
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

C = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"
CLIP_SEQ = "/data1/cympyc1785/data/DynPose-LBM/clatr_dynpose_d200_s91/seq"
DIRECTOR = "/data1/cympyc1785/LatentCamVid/camera_generation/models/DIRECTOR"
W = np.diag([1.0, -1.0, -1.0])
PROMPTS = "prompts.json"      # [R476] --prompts_file (예 prompts_v2.json)
STD = "d200s91"               # [R476] --std_name (DIRECTOR standardization yaml 이름 · infer 에서 같은 이름을 쓴다)
TRACK = "target_track.npz"     # [R352] --track_file 로 바꾼다 (convert 의 char·p0 와 evaldir 의 p0 가 같은 파일을 써야 역변환이 맞다)


def convert_one(scene, idx, cache):
    if scene not in cache:
        sd = f"{C}/dynpose/{scene}/da3"
        tp = np.load(f"{sd}/target_poses.npz", allow_pickle=True)
        tt = np.load(f"{sd}/{TRACK}", allow_pickle=True)
        K = np.load(f"{sd}/pose.npz")["intrinsics"][0]
        cache.clear()
        cache[scene] = dict(E=tp["extrinsics"], keys={str(k): i for i, k in enumerate(tp["keys"].tolist())},
                            track=tt["track_world"], tvalid=tt["valid"], tkeys={str(k): i for i, k in enumerate(tt["keys"].tolist())},
                            K=K, prompts=json.load(open(f"{sd}/{PROMPTS}")))
    d = cache[scene]
    i, j = d["keys"].get(idx), d["tkeys"].get(idx)
    if i is None or j is None or not d["tvalid"][j]:
        return None, "no_track"
    c2w = np.linalg.inv(d["E"][i].astype(np.float64))
    if np.abs(c2w).max() >= 1e2:
        return None, "diverged_pose"
    p = d["track"][j].astype(np.float64)
    p0 = p[0]
    R = np.einsum("ij,tjk->tik", W, c2w[:, :3, :3])
    t = (c2w[:, :3, 3] - p0) @ W.T
    ch = (p - p0) @ W.T
    pose = np.concatenate([R, t[:, :, None]], -1)                  # (T,3,4)
    cap = d["prompts"][idx]["prompt_camera_with_scene_video"]
    cap = cap["concise"] if isinstance(cap, dict) and "concise" in cap else (list(cap.values())[0] if isinstance(cap, dict) else cap)
    return dict(pose=pose, char=ch.astype(np.float32), K=d["K"].astype(np.float32), caption=cap), None


def stage_convert(a):
    sys.path.insert(0, DIRECTOR)
    import torch
    from src.metrics.modules.caption import CaptionMetrics
    seg = CaptionMetrics(num_classes=27)
    for sub in ("traj", "char", "char_raw", "caption", "caption_clip/seq", "intrinsics", "cam_segments"):
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    cache, why = {}, {}
    for split in (["test"] if a.split_list else ["train", "test"]):     # [R352] --split_list = test 하나만
        names = []
        for line in open(a.split_list or f"{C}/seg_list_dynpose_s91_{a.split_tag}{split}.txt"):
            line = line.strip()
            if not line:
                continue
            _, scene, idx = line.split("/")
            n = f"{scene}__{idx}"
            clip_src = (f"{a.clip_seq_dir}/dynpose_{scene}_{idx}_caption.npy" if a.clip_seq_dir     # [R352] eval_my/<dir>/seq/test 이름 규칙
                        else f"{CLIP_SEQ}/{split}/{n}_caption.npy")
            if not path.isfile(clip_src):
                why["no_clip"] = why.get("no_clip", 0) + 1
                continue
            r, bad = convert_one(scene, idx, cache)
            if r is None:
                why[bad] = why.get(bad, 0) + 1
                continue
            np.savetxt(f"{a.out}/traj/{n}.txt", r["pose"].reshape(len(r["pose"]), 12), fmt="%.8f")
            np.save(f"{a.out}/char/{n}.npy", r["char"])
            if not path.lexists(f"{a.out}/char_raw/{n}.npy"):
                os.symlink(f"../char/{n}.npy", f"{a.out}/char_raw/{n}.npy")
            open(f"{a.out}/caption/{n}.txt", "w").write(r["caption"])
            if not path.lexists(f"{a.out}/caption_clip/seq/{n}.npy"):
                os.symlink(clip_src, f"{a.out}/caption_clip/seq/{n}.npy")
            np.save(f"{a.out}/intrinsics/{n}.npy", r["K"])
            c2w4 = np.tile(np.eye(4), (len(r["pose"]), 1, 1)); c2w4[:, :3] = r["pose"]
            s = seg.segment_camera_trajectories(torch.from_numpy(c2w4).float()).numpy()   # (T-1,)
            np.save(f"{a.out}/cam_segments/{n}.npy", np.concatenate([s, s[-1:]]).astype(np.int64))
            names.append(n)
        open(f"{a.out}/mixed_{split}_split.txt", "w").write("\n".join(names))     # 끝 개행 없음
        print(f"[convert] {split}: {len(names)}  제외 {why}", flush=True)


def stage_stats(a):
    names = open(f"{a.out}/mixed_train_split.txt").read().split("\n")
    sh, nv, nh = [], [], []
    for n in names:
        P = np.loadtxt(f"{a.out}/traj/{n}.txt").reshape(-1, 3, 4)
        t = P[:, :, 3]
        sh.append(t[0]); nv.append(np.diff(t, axis=0))
        ch = np.load(f"{a.out}/char/{n}.npy")
        nh.append(np.concatenate([ch[:1], np.diff(ch, axis=0)]))
    sh, nv, nh = np.stack(sh), np.concatenate(nv), np.concatenate(nh)
    f = lambda x: "[" + ", ".join(f"{v:.8g}" for v in x) + "]"
    y = (f"name: '{STD}'\nnum_interframes: 0\nnum_cams: 49\n"
         "num_total_frames: ${eval:'${dataset.standardization.num_interframes} * (${dataset.standardization.num_cams} - 1) + ${dataset.standardization.num_cams} '}\n\n"
         f"# prepare_et_d200.py --stage stats (Dataset-v1 s91 train {len(names)} 샘플, E.T. 좌표 char0 원점 · y-up)\n"
         f"norm_mean: {f(nv.mean(0))}\nnorm_std: {f(nv.std(0))}\n\nshift_mean: {f(sh.mean(0))}\nshift_std: {f(sh.std(0))}\n\n"
         f"norm_mean_h: {f(nh.mean(0))}\nnorm_std_h: {f(nh.std(0))}\n\nvelocity: true\n")
    out = f"{DIRECTOR}/configs/dataset/standardization/{STD}.yaml"
    open(out, "w").write(y)
    print(f"[stats] → {out}\n{y}", flush=True)


def stage_infer(a):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, DIRECTOR); sys.path.insert(0, path.dirname(path.abspath(__file__)))
    from run_director_vista import build_diffuser, build_char_feat, make_get_matrix
    os.makedirs(a.eval_out, exist_ok=True)
    diffuser, config, device = build_diffuser(path.abspath(a.ckpt), "cuda",
                                              [f"dataset/standardization={STD}", "diffuser/network/module=ca_director"])
    num_cams = int(config.dataset.num_cams); assert num_cams == 49, num_cams
    std = {k: list(v) for k, v in config.dataset.standardization.items() if k.startswith(("norm_", "shift_"))}
    get_matrix = make_get_matrix(config).get_matrix
    names = [n for n in open(f"{a.out}/mixed_test_split.txt").read().split("\n") if n]
    diffuser.guidance_weight, diffuser.num_steps = a.guidance_weight, a.num_steps
    out = []
    for b0 in range(0, len(names), a.batch):
        nb = names[b0:b0 + a.batch]; B = len(nb)
        ch = torch.cat([build_char_feat(np.load(f"{a.out}/char/{n}.npy"), num_cams, std, device) for n in nb])
        caps = []
        for n in nb:
            sq = torch.from_numpy(np.load(f"{a.out}/caption_clip/seq/{n}.npy")).float()[:77]
            caps.append(F.pad(sq, (0, 0, 0, 77 - sq.shape[0])).T[None])
        cond = [ch, torch.cat(caps).to(device)]
        mask = torch.ones((B, num_cams), device=device)
        ref = torch.zeros((B, int(config.dataset.num_feats), num_cams), device=device) * diffuser.loss_fn.sigma_data
        diffuser.gen_seeds = [a.seed] * B
        with torch.no_grad():
            _, gen = diffuser.sample(diffuser.ema.ema_model, ref, cond, mask)
        out.append(torch.stack([get_matrix(x) for x in gen]).cpu().numpy()[:, :num_cams])
        print(f"[infer] {b0 + B}/{len(names)}", flush=True)
    np.savez_compressed(f"{a.eval_out}/et_preds.npz", names=np.array(names), poses=np.concatenate(out))


def stage_evaldir(a):
    import shutil
    z = np.load(f"{a.eval_out}/et_preds.npz"); names, P = z["names"], z["poses"]
    GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
    outs = {k: f"{a.eval_out}_{k}" for k in ("raw", "anc")}
    for o in outs.values():
        os.makedirs(f"{o}/test", exist_ok=True)
    cache, n_ok = {}, 0
    for n, pe in zip(names, P):
        scene, idx = n.split("__")
        en = f"dynpose_{scene}_{idx}"
        rj = f"{a.ref_eval_dir}/test/{en}_transforms_ref.json"
        if not path.isfile(rj):
            continue
        sd = f"{C}/dynpose/{scene}/da3"
        if scene not in cache:
            tt = np.load(f"{sd}/{TRACK}", allow_pickle=True); cache.clear()
            cache[scene] = ({str(k): i for i, k in enumerate(tt["keys"].tolist())}, tt["track_world"])
        tk, tw = cache[scene]; p0 = tw[tk[idx]][0].astype(np.float64)
        cv = np.tile(np.eye(4), (len(pe), 1, 1))
        cv[:, :3, :3] = np.einsum("ij,tjk->tik", W, pe[:, :3, :3]); cv[:, :3, 3] = pe[:, :3, 3] @ W.T + p0
        ref = json.load(open(rj))
        ref_cv = np.array([f["transform_matrix"] for f in ref["frames"]], dtype=np.float64) @ GL2CV
        arms = {"raw": cv, "anc": ref_cv[0] @ np.linalg.inv(cv[0]) @ cv}
        for k, M in arms.items():
            js = json.loads(json.dumps(ref))
            for f, m in zip(js["frames"], M @ GL2CV):
                f["transform_matrix"] = m.tolist()
            json.dump(js, open(f"{outs[k]}/test/{en}_transforms_pred.json", "w"))
            shutil.copy(rj, f"{outs[k]}/test/{en}_transforms_ref.json")
            cj = f"{a.ref_eval_dir}/test/{en}_caption.json"
            if path.isfile(cj):
                shutil.copy(cj, f"{outs[k]}/test/{en}_caption.json")
        n_ok += 1
    print(f"[evaldir] {n_ok}/{len(names)} → {list(outs.values())}", flush=True)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--stage", default="convert,stats")
    ap.add_argument("--out", default="/data1/cympyc1785/data/DynPose-LBM/et_dynpose_d200_s91")
    # [R314] infer / evaldir
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--corpus", default=C)              # [R352] 다른 코퍼스 (기본 = Dataset-v1 d200)
    ap.add_argument("--prompts_file", default=PROMPTS)  # [R476]
    ap.add_argument("--std_name", default=STD)          # [R476]
    ap.add_argument("--split_tag", default="")          # [R476] 예 "noroll_" → seg_list_dynpose_s91_noroll_{train,test}.txt
    ap.add_argument("--clip_seq_root", default=CLIP_SEQ)  # [R476] <root>/<split>/<scene>__<idx>_caption.npy (예 clatr_dynpose_v5f50k_v2/seq)
    ap.add_argument("--track_file", default=TRACK)      # [R352] 예 target_track_molmo2.npz
    ap.add_argument("--split_list", default="")         # [R352] seg 목록 하나 (test 로 취급). "" = s91 train/test
    ap.add_argument("--clip_seq_dir", default="")       # [R352] 캡션 CLIP seq 폴더 (dynpose_<scene>_<idx>_caption.npy). "" = clatr d200 캐시
    ap.add_argument("--eval_out", default="")
    ap.add_argument("--ref_eval_dir", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance_weight", type=float, default=1.4)
    ap.add_argument("--num_steps", type=int, default=10)
    a = ap.parse_args()
    C, TRACK, PROMPTS, STD, CLIP_SEQ = a.corpus, a.track_file, a.prompts_file, a.std_name, a.clip_seq_root   # [R352·R476] 모듈 전역
    for st in a.stage.split(","):
        globals()[f"stage_{st}"](a)
