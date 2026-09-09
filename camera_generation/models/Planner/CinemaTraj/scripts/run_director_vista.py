"""Vista 씬 하나의 subject OBB track + 영문 caption 을 **DIRECTOR(E.T.)** 에 넣어 카메라 궤적을 받는다.

WHY: DIRECTOR 는 `src/evaluate.py` 가 et-data 데이터셋에 묶여 있어 단일 샘플을 못 넣는다
     (`dataset.root_filenames.index(sample_id)`). 우리는 et-data 가 없고 넣고 싶은 것도
     우리 `scene_graph.json` 의 track 하나뿐이라, 배치를 **손으로 조립**해서
     `Diffuser.sample()` 만 직접 부른다. `visualization/common_viz.py` 의 init/encode_text
     경로만 따오고 MultimodalDataset 은 안 만든다.

좌표계 (FIX 2026-08-27, `director_pilot_v2`): scene graph 노드는 **G frame** 이다 (up=+z,
스케일 1/S). v1 은 이걸 world 로 착각하고 world frame 인 `gravity.up_world` / `fwd=[0,0,1]`
로 E.T. 기저를 만들어 up 축이 camel 93.75° / snowboard 101.28° 어긋났다 (`graph_basis`
docstring 참조). `cam_t_g` / `char_g` 는 **G frame** 이므로 렌더·시각화는 `T_wg` 를 곱해야 한다.

스케일 게이지 `--scale_mode`:
  * `height`(기본) — subject OBB 높이 = `--subject_height_m`. 씬마다 피사체 종류에 의존.
  * `path`        — char 이동거리 = `--target_path_m`(기본 1.0 m). 씬 크기·피사체와 무관하다.
  * `scene`       — depth 로 잰 씬 스케일 S = `--target_scene_m`(기본 1.0 m). 1 u ≜ S DA3 units
                    (frame0 non-sky 평균 ray 길이) 라 G frame 에선 씬 스케일이 정의상 1.0 u 다.
                    피사체 크기·이동 유무 어느 쪽에도 안 걸린다 (정지 subject 도 된다).

미검증 규약 두 가지 (결과 해석 시 반드시 같이 볼 것):
  1) **E.T. world 축**. et-data 가 없어 축 순서/손잡이를 실측 못 했다. 여기서는
     standardization/0300.yaml 의 `shift_mean = [0.002, -0.275, -1.237]` 을 근거로
     "char 이 원점 근처, 카메라는 char 앞쪽 -Z, up 은 +Y" 로 가정한다. X 부호가 뒤집혀
     있으면 생성 궤적의 좌우가 통째로 뒤집힌다 (본으로 검증 불가한 종류).
  2) **E.T. camera 축**. c2w 의 열이 OpenCV(x right/y down/z fwd) 인지 OpenGL(-z fwd) 인지
     모른다. 여기서는 변환하지 않고 **날것 그대로** 저장하고, 카메라 위치(t)만 그린다.
     회전을 쓰려면 먼저 이 규약을 못박아야 한다.

스케일: scene_graph 의 단위 `u` 를 미터로 바꿔 넣는다. E.T. 는 사람 기준 미터 스케일로
        보인다 (`shift_std ~ [1.13, 1.19, 1.59]`). 기본값은 "subject 높이 = --subject_height_m"
        로 역산한다.

사용 예시:
    conda run -n GenDoP python scripts/run_director_vista.py \
        --scene_graph out/camel/scene_graph.json --subject dyn_0 \
        --caption "The camera trucks right and pushes in on the camel." \
        --out results/20260827_director_camel
"""
from argparse import ArgumentParser
from os import path, makedirs, chdir
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

DIRECTOR_ROOT = "/data1/cympyc1785/LatentCamVid/camera_generation/models/DIRECTOR"


# ----------------------------------------------------------------------------------- #
# scene_graph -> char track


def load_char_track(sg_path, subject_id):
    """scene_graph.json 의 subject track 을 (F,3) **G frame**(u) 로."""
    with open(sg_path, encoding="utf-8") as file:
        graph = json.load(file)
    node = next((n for n in graph["nodes"] if n["id"] == subject_id), None)
    assert node is not None, f"{subject_id} 가 {sg_path} 에 없다"
    centers = np.asarray(node["track"]["center_smooth"], dtype=np.float64)
    return graph, node, centers


def graph_basis(graph, basis):
    """E.T. 기저를 만들 (up, frame0 시선) 을 **G frame** 에서 돌려준다.

    FIX(2026-08-27): `legacy` 는 노드 좌표계를 착각한 버그다. `scene_graph.json` 의
    `track.center_smooth` / `obb.center` 는 **G frame**(up=+z, 스케일 1/S)인데
    `gravity.up_world` 와 `fwd=[0,0,1]` 은 world frame 값이라, G 좌표에 world 축을 물려
    up 축이 camel 93.75° / snowboard 101.28° 어긋났다. `fwd=[0,0,1]` 의 근거였던
    "cam_c2w[0]=I" 도 거짓이다 (snowboard |t0|=0.81 u, rot 6.18°).
    `graph`(기본) 는 G 에서 up=+z 를 쓰고 시선은 실제 frame0 c2w 열2를 G 로 옮겨 쓴다.
    """
    R_gw = np.asarray(graph["frames"]["T_gw"], dtype=np.float64)[:3, :3]
    if basis == "legacy":
        up = np.asarray(graph["gravity"]["up_world"], dtype=np.float64)
        return up / np.linalg.norm(up), np.array([0.0, 0.0, 1.0])
    up = R_gw @ np.asarray(graph["gravity"]["up_world"], dtype=np.float64)
    fwd = R_gw @ np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64)[0][:3, 2]
    return up / np.linalg.norm(up), fwd / np.linalg.norm(fwd)


def world_to_et(centers_u, up, fwd, meters_per_u):
    """scene graph G frame(u) -> E.T. world(가정: y-up, char 원점, m).

    e_y = up (G 에서는 +z)
    e_z = frame0 카메라 시선을 up 에 수직으로 투영                # char 는 카메라 앞
    e_x = e_y x e_z                                              # 오른손 강제
    카메라는 char 뒤(-e_z)에 있으므로 E.T. 좌표에서 z<0 -> shift_mean[2]=-1.237 과 부호가 맞다.
    """
    e_y = np.asarray(up, dtype=np.float64)
    e_y /= np.linalg.norm(e_y)
    fwd = np.asarray(fwd, dtype=np.float64)
    e_z = fwd - np.dot(fwd, e_y) * e_y
    e_z /= np.linalg.norm(e_z)
    e_x = np.cross(e_y, e_z)
    R_et_w = np.stack([e_x, e_y, e_z], axis=0)           # G 벡터 -> E.T. 성분
    assert abs(np.linalg.det(R_et_w) - 1.0) < 1e-9, np.linalg.det(R_et_w)

    rel = centers_u - centers_u[0]                       # frame0 char 을 원점으로
    return (rel @ R_et_w.T) * meters_per_u, R_et_w


# ----------------------------------------------------------------------------------- #
# DIRECTOR


def build_diffuser(checkpoint_path, device_str):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # clatr.yaml 의 checkpoint 경로가 리포 루트 기준 상대경로다 (`checkpoints/clatr-e100.ckpt`).
    chdir(DIRECTOR_ROOT)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base="1.3",
                               config_dir=path.join(DIRECTOR_ROOT, "configs")):
        config = compose(config_name="config_demo",
                         overrides=[f"checkpoint_path={checkpoint_path}",
                                    f"compnode.device={device_str}"])
    device = torch.device(config.compnode.device)
    diffuser = instantiate(config.diffuser)
    # torch>=2.6 의 weights_only=True 는 ckpt 안의 omegaconf ListConfig 에서 막힌다.
    state_dict = torch.load(config.checkpoint_path, map_location=device,
                            weights_only=False)["state_dict"]
    state_dict["ema.initted"] = diffuser.ema.initted
    state_dict["ema.step"] = diffuser.ema.step
    diffuser.load_state_dict(state_dict, strict=False)
    diffuser.to(device).eval()
    return diffuser, config, device


def encode_caption(caption, device):
    """common_viz.encode_text 의 sequential 분기. (1, 512, 77)."""
    import clip

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    texts = clip.tokenize([caption], truncate=True)
    x = model.token_embedding(texts.to(device)).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    eot = int(texts.argmax(dim=-1)[0])
    seq = x[0, : eot + 1].float()                        # (Ls, 512)
    seq = F.pad(seq, (0, 0, 0, 77 - seq.shape[0]))
    return seq.unsqueeze(0).permute(0, 2, 1), eot + 1


def build_char_feat(centers_m, num_cams, std, device):
    """CharacterDataset.__getitem__ 재현 (velocity=True, norm_mean_h 길이 3 -> 일괄 정규화)."""
    raw = torch.from_numpy(centers_m).to(torch.float32)  # (F,3)
    feat = torch.cat([raw[0][None], raw[1:] - raw[:-1]])
    feat = feat - torch.tensor(std["norm_mean_h"], dtype=torch.float32)
    feat = feat / torch.tensor(std["norm_std_h"], dtype=torch.float32)
    feat = F.pad(feat, (0, 0, 0, num_cams - feat.shape[0]))
    return feat.permute(1, 0).unsqueeze(0).to(device)    # (1, 3, 300)


def make_get_matrix(config):
    """TrajectoryDataset.get_matrix 만 쓰기 위한 최소 인스턴스 (파일 접근 없음)."""
    from src.datasets.modalities.trajectory_dataset import TrajectoryDataset
    from omegaconf import OmegaConf

    traj_cfg = OmegaConf.to_container(config.dataset.trajectory, resolve=True)
    traj_cfg.pop("_target_", None)
    traj_cfg["standardization"] = OmegaConf.to_container(
        config.dataset.standardization, resolve=True)
    traj_cfg.setdefault("num_rawfeats", config.dataset.num_rawfeats)
    traj_cfg.setdefault("num_cams", config.dataset.num_cams)
    return TrajectoryDataset(**traj_cfg)


# ----------------------------------------------------------------------------------- #


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scene_graph", required=True)                 # scene_graph.json
    parser.add_argument("--subject", default="dyn_0")                   # char 로 쓸 노드 id
    parser.add_argument("--caption", required=True)                     # 영문 camera caption
    parser.add_argument("--out", required=True)                         # 결과 폴더
    parser.add_argument("--checkpoint", default=path.join(
        DIRECTOR_ROOT, "checkpoints/director/ca-mixed-e449.ckpt"))
    parser.add_argument("--guidance_weight", type=float, default=1.4)   # config 기본값
    parser.add_argument("--num_steps", type=int, default=10)            # EDM 스텝
    parser.add_argument("--num_samples", type=int, default=4)           # seed 0..N-1
    parser.add_argument("--subject_height_m", type=float, default=2.0)  # u->m 환산 기준(낙타 키)
    parser.add_argument("--scale_mode", choices=["height", "path", "scene"], default="height")
    parser.add_argument("--target_path_m", type=float, default=1.0)     # scale_mode=path 의 기준
    parser.add_argument("--target_scene_m", type=float, default=1.0)    # scale_mode=scene 의 기준
    parser.add_argument("--meters_per_u", type=float, default=None)     # 주면 위 환산을 덮어씀
    parser.add_argument("--et_basis", choices=["graph", "legacy"], default="graph")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, DIRECTOR_ROOT)
    args.out = path.abspath(args.out)
    args.scene_graph = path.abspath(args.scene_graph)
    args.checkpoint = path.abspath(args.checkpoint)
    makedirs(args.out, exist_ok=True)

    graph, node, centers_u = load_char_track(args.scene_graph, args.subject)
    num_frames = centers_u.shape[0]
    height_u = float(node["obb"]["extent"][2])
    # char 이 실제로 움직인 거리 (G frame, u). scale_mode=path 의 분모.
    path_u = float(np.linalg.norm(np.diff(centers_u, axis=0), axis=1).sum())
    # depth 로 잰 씬 스케일. 1 u ≜ S DA3 units (frame0 non-sky 평균 ray 길이) 라서 G frame 에서
    # 씬 스케일은 정의상 1.0 u 다. z_med 는 참고용으로 같이 찍는다.
    scene_u = 1.0
    z_med_u = float(graph["scale"]["z_med_frame0"]) / float(graph["scale"]["S"])
    if args.meters_per_u:
        meters_per_u = args.meters_per_u
    elif args.scale_mode == "path":
        # 거리 기준: 씬 크기·subject 종류와 무관하게 char 이동거리를 target_path_m 으로 고정.
        assert path_u > 1e-9, f"{args.subject} 는 정지 track 이라 path 로 정규화할 수 없다"
        meters_per_u = args.target_path_m / path_u
    elif args.scale_mode == "scene":
        # depth 기준: 씬 스케일 S 를 target_scene_m 으로 고정. subject 종류·이동 유무와 무관.
        meters_per_u = args.target_scene_m / scene_u
    else:
        meters_per_u = args.subject_height_m / height_u
    up_g, fwd_g = graph_basis(graph, args.et_basis)
    centers_m, R_et_w = world_to_et(centers_u, up_g, fwd_g, meters_per_u)

    diffuser, config, device = build_diffuser(args.checkpoint, args.device)
    num_cams = int(config.dataset.num_cams)
    assert num_frames <= num_cams, f"{num_frames} 프레임은 num_cams={num_cams} 를 넘는다"
    std = {k: list(v) for k, v in config.dataset.standardization.items()
           if k.startswith("norm_") or k.startswith("shift_")}

    caption_feat, num_tokens = encode_caption(args.caption, device)
    char_feat = build_char_feat(centers_m, num_cams, std, device)

    # padding_mask: 유효 1 / 패딩 0 (trajectory_dataset.py:146-147)
    mask = torch.zeros((args.num_samples, num_cams), device=device)
    mask[:, :num_frames] = 1.0
    cond = [char_feat.repeat(args.num_samples, 1, 1),          # (B,3,300)  cond_projection[0]
            caption_feat.repeat(args.num_samples, 1, 1)]       # (B,512,77) cond_projection[1]
    ref = torch.zeros((args.num_samples, int(config.dataset.num_feats), num_cams),
                      device=device)
    ref = ref * diffuser.loss_fn.sigma_data                    # edm2_normalization (diffuser.py:167)

    diffuser.gen_seeds = list(range(args.num_samples))
    diffuser.guidance_weight = args.guidance_weight
    diffuser.num_steps = args.num_steps
    with torch.no_grad():
        _, gen = diffuser.sample(diffuser.ema.ema_model, ref, cond, mask)

    get_matrix = make_get_matrix(config).get_matrix
    poses = torch.stack([get_matrix(x) for x in gen]).cpu().numpy()      # (B,300,4,4)
    poses = poses[:, :num_frames]                                        # 유효 구간만

    # E.T. world -> scene graph G frame(u). 회전은 규약 미검증이라 위치만 되돌린다.
    t_et = poses[:, :, :3, 3]
    t_g = (t_et / meters_per_u) @ R_et_w + centers_u[0]

    # G -> world 는 하류(렌더/시각화)가 반드시 적용해야 한다. v1 은 이걸 빼먹은 버그였다.
    npz_path = path.join(args.out, "director_poses.npz")
    np.savez(npz_path, poses_et=poses, cam_t_g=t_g, char_et_m=centers_m,
             char_g=centers_u, R_et_w=R_et_w, meters_per_u=meters_per_u,
             T_wg=np.asarray(graph["frames"]["T_wg"], dtype=np.float64),
             cam_c2w_world=np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64))

    meta = {"format": "director_pilot_v2", "scene_graph": path.abspath(args.scene_graph),
            "video": graph.get("video"), "subject": args.subject,
            "subject_label": node.get("label"), "caption": args.caption,
            "caption_tokens": num_tokens, "checkpoint": args.checkpoint,
            "guidance_weight": args.guidance_weight, "num_steps": args.num_steps,
            "num_samples": args.num_samples, "seeds": diffuser.gen_seeds,
            "num_frames": num_frames, "num_cams": num_cams,
            "meters_per_u": meters_per_u, "subject_height_u": height_u,
            "subject_height_m": args.subject_height_m,
            "scale_mode": args.scale_mode, "target_path_m": args.target_path_m,
            "target_scene_m": args.target_scene_m, "char_path_u": path_u,
            "scene_S_da3": graph["scale"]["S"], "z_med_frame0_u": z_med_u,
            "et_basis": args.et_basis,
            "frame": "cam_t_g / char_g 는 scene graph G frame (u). world 는 T_wg 를 곱할 것",
            "unverified": ["E.T. world 축 부호(좌우 반전 가능)",
                           "E.T. camera 축 규약(OpenCV vs OpenGL) — 회전 미사용"]}
    with open(path.join(args.out, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    gauge = {"path": f"char path {path_u:.4f} u = {args.target_path_m} m",
             "scene": f"scene scale {scene_u:.4f} u (S={graph['scale']['S']:.4f} DA3,"
                      f" z_med {z_med_u:.4f} u) = {args.target_scene_m} m",
             "height": f"subject height {height_u:.4f} u = {args.subject_height_m} m"}
    print(f"{'meters_per_u':<22}{meters_per_u:.3f}   "
          f"({'given' if args.meters_per_u else args.scale_mode}: {gauge[args.scale_mode]})")
    print(f"{'et_basis':<22}{args.et_basis}")
    print(f"{'char path (m)':<22}"
          f"{np.linalg.norm(np.diff(centers_m, axis=0), axis=1).sum():.3f}")
    print(f"{'caption tokens':<22}{num_tokens}")
    for b in range(args.num_samples):
        p = poses[b, :, :3, 3]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        d = np.linalg.norm(p - centers_m, axis=1)
        print(f"  seed {b}  path {step.sum():7.3f} m   |t0| {np.linalg.norm(p[0]):6.3f}"
              f"   dist2char {d.min():6.3f}~{d.max():6.3f} m")
    print(f"{'saved':<22}{npz_path}")


if __name__ == "__main__":
    main()
