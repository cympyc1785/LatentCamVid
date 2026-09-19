"""DIRECTOR(E.T.) 를 split 전량에 돌려 `gendop_preds_to_eval_dir.py` 가 먹는 npz 를 낸다.

WHY: `run_director_vista.py` 는 씬 하나짜리 파일럿이라 엔트리마다 프로세스를 새로 띄우고
     (ckpt + CLIP 로드 ~30 s) CLIP 을 `encode_caption` 안에서 매번 다시 올린다. d200 test
     split 은 5,144 엔트리라 그대로는 못 돈다. **모델을 한 번만 올리고 배치로 돈다.**
     좌표 변환·스케일 게이지는 `run_director_vista.py` 의 함수를 그대로 import 해서 쓴다
     (그 파일은 한 줄도 안 고친다 — 파일럿 결과가 계속 재현돼야 한다).

배치가 단일 호출과 같은 결과인가: 같다. `Diffuser.sample` 은 `StackedRandomGenerator`
(utils/random_utils.py:22) 로 **배치 원소마다 별도 `torch.Generator`** 를 쓰므로 seeds 를
`[seed]*B` 로 주면 각 원소가 단독 실행과 같은 latent 을 받는다. 남는 차이는 batched matmul
의 부동소수 재결합뿐이다.

출력 npz (`director__<scene>__<idx>.npz`) 는 **GenDoP 어댑터 규약**에 맞춘다:
  `c2w`   (49,4,4) — **코퍼스 world frame, OpenGL 표기**. 어댑터
          `gendop_preds_to_eval_dir.py:154` 가 `@ GL2CV` 를 곱해 OpenCV 로 되돌린다
          (GL2CV 는 자기역원이라 여기서 한 번 곱해 두면 정확히 상쇄된다).
  `scale` 1.0 — 어댑터의 `--no_scale_token` 분기가 나눗셈을 해도 no-op 이 되게.
  `text`  실제로 넣은 문장.
E.T. world(m) -> G frame(u) -> world 변환은 `T_wg`(= S·R) 로 한다. 회전 부분은 S 로 나눠
정규직교로 만든 뒤 곱한다.

**미검증 규약(파일럿에서 그대로 승계)**: E.T. camera 축이 OpenCV(+z fwd)인지 OpenGL(-z fwd)
인지 문서가 없다. `--cam_conv` 로 고르고, 매 런마다 **실측**해서 요약에 찍는다 — E.T. 는
char 을 프레임에 잡는 모델이므로 "카메라 전방 축과 (char - cam) 사이 각"이 작아지는 쪽이
맞는 규약이다. 두 값이 다 90° 근처면 그 런의 회전은 못 믿는다.

env: GenDoP (hydra/clip/torch 가 거기 있다).

사용:
    PY=/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python
    $PY scripts/run_director_batch.py \
        --split /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200/seg_list_dynpose_s91_test.txt \
        --corpus /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
        --cloud_root out_dynpose --prefix dynpose \
        --text_dir <eval 폴더 모양 캡션 디렉토리> \
        --out results/20260920_d208_gendop_d200/pred_director_p49
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path
from time import time

import numpy as np
import torch

HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, HERE)

from run_director_vista import (                                     # noqa: E402
    DIRECTOR_ROOT, build_char_feat, build_diffuser, graph_basis, make_get_matrix,
    world_to_et,
)

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])        # OpenGL c2w <-> OpenCV c2w (자기역원)
TEXT_KEY = "Concise Interaction"


def load_clip(device):
    """`run_director_vista.encode_caption` 의 CLIP 로드를 루프 밖으로 뺀 것."""
    import clip

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def encode_captions(model, captions, device):
    """(B, 512, 77) — `common_viz.encode_text` sequential 분기를 배치로."""
    import clip
    import torch.nn.functional as F

    texts = clip.tokenize(captions, truncate=True).to(device)
    x = model.token_embedding(texts).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    eots = texts.argmax(dim=-1)
    out = torch.zeros((len(captions), 77, 512), dtype=torch.float32, device=device)
    for i, eot in enumerate(eots.tolist()):
        out[i, : eot + 1] = x[i, : eot + 1].float()
    return out.permute(0, 2, 1), (eots + 1).tolist()


def aim_angles(poses_et, char_et):
    """카메라 +z 축과 (char - cam) 사이 각(도). OpenCV 면 작고 OpenGL 이면 180 근처다."""
    t = poses_et[:, :3, 3]
    fwd = poses_et[:, :3, 2]
    d = char_et - t
    n = np.linalg.norm(d, axis=-1) * np.linalg.norm(fwd, axis=-1)
    ok = n > 1e-9
    if not ok.any():
        return float("nan")
    cos = np.clip((d[ok] * fwd[ok]).sum(-1) / n[ok], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)              # `<dataset>/<scene>/<idx>` 한 줄씩
    ap.add_argument("--corpus", required=True)             # prompts.json (variant_id -> subject)
    ap.add_argument("--cloud_root", required=True)         # `<root>/<scene>/scene_graph.json`
    ap.add_argument("--prefix", default="dynpose")         # 캡션 파일명 접두사
    ap.add_argument("--text_dir", required=True)           # `<dir>/test/<prefix>_<scene>_<idx>_caption.json`
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=path.join(
        DIRECTOR_ROOT, "checkpoints/director/ca-mixed-e449.ckpt"))
    ap.add_argument("--guidance_weight", type=float, default=1.4)
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)         # 엔트리당 1 샘플 (arm 하나로 채점)
    ap.add_argument("--batch", type=int, default=16)
    # u -> m 게이지. dynpose anchor 는 손·컵·사람이 섞여 있어 실제 높이를 못 쓴다. E.T. 는
    # 사람 스케일(shift_std ~[1.13,1.19,1.59] m)에서 학습됐으므로 **"샷의 주인공 = 사람 키"**
    # 로 전 엔트리에 같은 게이지를 건다. 값 자체가 게이지 선택이라 meta 에 남긴다.
    ap.add_argument("--scale_mode", choices=["height", "path", "scene"], default="height")
    ap.add_argument("--subject_height_m", type=float, default=1.7)
    ap.add_argument("--target_path_m", type=float, default=1.0)
    ap.add_argument("--target_scene_m", type=float, default=1.0)
    ap.add_argument("--et_basis", choices=["graph", "legacy"], default="graph")
    # E.T. camera 축 규약. 실측값을 요약에 찍으니 런 후에 반드시 확인할 것.
    ap.add_argument("--cam_conv", choices=["opencv", "opengl"], default="opencv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    # `run_director_vista.main` 이 하던 일 — 그 파일은 함수만 빌려 쓰므로 여기서 직접 건다.
    # 빠지면 hydra 가 `src.training.diffuser.Diffuser` 를 못 찾는다.
    if DIRECTOR_ROOT not in sys.path:
        sys.path.insert(0, DIRECTOR_ROOT)

    # build_diffuser 가 chdir(DIRECTOR_ROOT) 를 하므로 경로는 전부 먼저 절대화한다.
    for k in ("split", "corpus", "cloud_root", "text_dir", "out", "checkpoint"):
        setattr(a, k, path.abspath(getattr(a, k)))
    makedirs(a.out, exist_ok=True)

    with open(a.split, encoding="utf-8") as file:
        lines = [l.strip() for l in file if l.strip()]
    if a.limit:
        lines = lines[: a.limit]

    diffuser, config, device = build_diffuser(a.checkpoint, a.device)
    num_cams = int(config.dataset.num_cams)
    num_feats = int(config.dataset.num_feats)
    std = {k: list(v) for k, v in config.dataset.standardization.items()
           if k.startswith("norm_") or k.startswith("shift_")}
    get_matrix = make_get_matrix(config).get_matrix
    clip_model = load_clip(device)
    diffuser.guidance_weight = a.guidance_weight
    diffuser.num_steps = a.num_steps

    graphs, prompts = {}, {}
    n_ok = n_skip_done = n_no_cap = n_no_node = n_bad = 0
    angles_cv, tic = [], time()
    pend = []

    def flush():
        nonlocal n_ok
        if not pend:
            return
        B = len(pend)
        cond = [torch.cat([p["char_feat"] for p in pend], 0),
                torch.cat([p["cap"] for p in pend], 0)]
        mask = torch.zeros((B, num_cams), device=device)
        for i, p in enumerate(pend):
            mask[i, : p["num_frames"]] = 1.0
        ref = torch.zeros((B, num_feats, num_cams), device=device) * diffuser.loss_fn.sigma_data
        diffuser.gen_seeds = [a.seed] * B
        with torch.no_grad():
            _, gen = diffuser.sample(diffuser.ema.ema_model, ref, cond, mask)
        poses = torch.stack([get_matrix(x) for x in gen]).cpu().numpy()   # (B,300,4,4)
        for i, p in enumerate(pend):
            F_ = p["num_frames"]
            pose_et = poses[i, :F_]
            angles_cv.append(aim_angles(pose_et, p["char_et"]))

            # E.T. world(m) -> G(u): 위치는 파일럿과 같은 식, 회전은 기저 전치.
            t_g = (pose_et[:, :3, 3] / p["mpu"]) @ p["R_et_w"] + p["c0_u"]
            R_g = np.einsum("ji,fjk->fik", p["R_et_w"], pose_et[:, :3, :3])
            # G -> world. T_wg = S·R 이므로 회전만 쓰려면 S 로 나눈다.
            T_wg = p["T_wg"]
            S = float(np.linalg.norm(T_wg[:3, 0]))
            c2w_cv = np.tile(np.eye(4), (F_, 1, 1))
            c2w_cv[:, :3, 3] = t_g @ T_wg[:3, :3].T + T_wg[:3, 3]
            c2w_cv[:, :3, :3] = (T_wg[:3, :3] / S) @ R_g
            if a.cam_conv == "opengl":
                c2w_cv = c2w_cv @ GL2CV                  # E.T. 가 GL 이면 여기서 CV 로
            np.savez(p["npz"], c2w=c2w_cv @ GL2CV,       # 어댑터가 다시 GL2CV 를 곱해 상쇄
                     scale=np.float32(1.0), text=p["text"],
                     poses_et=pose_et, char_et_m=p["char_et"], char_g=p["char_g"],
                     R_et_w=p["R_et_w"], meters_per_u=p["mpu"], T_wg=T_wg,
                     aim_angle_pluz_deg=np.float32(angles_cv[-1]))
            n_ok += 1
        pend.clear()

    for n_seen, line in enumerate(lines):
        dataset, scene, index = line.split("/")
        npz = path.join(a.out, f"director__{scene}__{index}.npz")
        if path.isfile(npz) and not a.overwrite:
            n_skip_done += 1
            continue
        cap_path = path.join(a.text_dir, "test",
                             f"{a.prefix}_{scene}_{index}_caption.json")
        if not path.isfile(cap_path):
            n_no_cap += 1
            continue
        with open(cap_path, encoding="utf-8") as file:
            text = json.load(file)[TEXT_KEY]

        if scene not in prompts:
            with open(path.join(a.corpus, dataset, scene, "da3", "prompts.json"),
                      encoding="utf-8") as file:
                prompts[scene] = json.load(file)
            with open(path.join(a.cloud_root, scene, "scene_graph.json"),
                      encoding="utf-8") as file:
                graphs[scene] = json.load(file)
        graph = graphs[scene]
        subject = prompts[scene][index]["variant_id"].split("__")[0]
        node = next((n for n in graph["nodes"] if n["id"] == subject), None)
        if node is None:
            n_no_node += 1
            continue

        centers_u = np.asarray(node["track"]["center_smooth"], dtype=np.float64)
        F_ = centers_u.shape[0]
        height_u = float(node["obb"]["extent"][2])
        path_u = float(np.linalg.norm(np.diff(centers_u, axis=0), axis=1).sum())
        if a.scale_mode == "path":
            if path_u < 1e-9:
                n_bad += 1
                continue
            mpu = a.target_path_m / path_u
        elif a.scale_mode == "scene":
            mpu = a.target_scene_m                      # 씬 스케일은 G 에서 정의상 1.0 u
        else:
            if height_u < 1e-9:
                n_bad += 1
                continue
            mpu = a.subject_height_m / height_u
        up_g, fwd_g = graph_basis(graph, a.et_basis)
        centers_m, R_et_w = world_to_et(centers_u, up_g, fwd_g, mpu)

        cap, _ = encode_captions(clip_model, [text], device)
        pend.append(dict(npz=npz, text=text, num_frames=F_, mpu=mpu, R_et_w=R_et_w,
                         c0_u=centers_u[0], char_et=centers_m, char_g=centers_u,
                         T_wg=np.asarray(graph["frames"]["T_wg"], dtype=np.float64),
                         cap=cap,
                         char_feat=build_char_feat(centers_m, num_cams, std, device)))
        if len(pend) >= a.batch:
            flush()
            if n_ok % (a.batch * 10) == 0:
                print(f"  {n_ok}/{len(lines)}  {time() - tic:.0f}s", flush=True)
    flush()

    ang = np.asarray([x for x in angles_cv if np.isfinite(x)])
    print(f"\n{'split':<18}{a.split}")
    print(f"{'entries':<18}{len(lines)}")
    print(f"{'written':<18}{n_ok}")
    print(f"{'skip(done)':<18}{n_skip_done}")
    print(f"{'no caption':<18}{n_no_cap}")
    print(f"{'no node':<18}{n_no_node}")
    print(f"{'scale 불가':<18}{n_bad}")
    print(f"{'scale_mode':<18}{a.scale_mode} "
          f"(height {a.subject_height_m} m / path {a.target_path_m} m / scene {a.target_scene_m} m)")
    print(f"{'cam_conv':<18}{a.cam_conv}")
    if len(ang):
        print(f"\nE.T. 카메라 +z 와 (char-cam) 사이 각 — **규약 실측**")
        print(f"  mean {ang.mean():6.2f}°   median {np.median(ang):6.2f}°   "
              f"5% {np.percentile(ang, 5):6.2f}°  95% {np.percentile(ang, 95):6.2f}°")
        print("  90° 보다 확실히 작으면 OpenCV(+z fwd), 확실히 크면 OpenGL(-z fwd).")
    print(f"\n{'elapsed':<18}{time() - tic:.0f}s")
    print(f"{'out':<18}{a.out}")


if __name__ == "__main__":
    main()
