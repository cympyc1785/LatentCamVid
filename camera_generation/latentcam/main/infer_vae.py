import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import datetime

from config_vae import cfg
from core_pkg.common.base import Trainer
from core_pkg.models.vae_intr_large import CameraVAE
from core_pkg.common.utils.data_utils import out_to_trajectory, make_intrinsics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CAM_INDEX_TO_PATTERN = {
    0: "static",
    1: "move backward",  # keep "move forward" as it is
    2: "move forward",  # keep "move backward" as it is
    3: "move up",
    6: "move down",
    18: "move left",
    9: "move right",
    # ----- #
    12: "move right and up",
    15: "move right and down",
    21: "move left and up",
    24: "move left and down",
    10: "move right and backward",
    11: "move right and forward",
    19: "move left and backward",
    20: "move left and forward",
    4: "move up and backward",
    5: "move up and forward",
    7: "move down and backward",
    8: "move down and forward",
    # ----- #
    13: "move right, up, and backward",
    14: "move right, up, and forward",
    16: "move right, down, and backward",
    17: "move right, down, and forward",
    22: "move left, up, and backward",
    23: "move left, up, and forward",
    25: "move left, down, and backward",
    26: "move left, down, and forward"
}

groups = [
    range(0, 1),
    range(1, 2),
    range(2, 3),
    range(3, 6),
    range(6, 9),
    range(9, 18),
    range(18, 27)
]

# 그룹 색
group_colors = plt.cm.tab10(np.linspace(0,1,len(groups)))

# label -> color mapping
label_to_color = {}
for g, color in zip(groups, group_colors):
    for l in g:
        label_to_color[l] = color

def infer():
    # hyperparams
    device = 'cuda:0'
    ckpt_path = cfg.vae_ckpt_path 
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.abspath(os.path.join('../infer_vae', exp_name))
    os.makedirs(result_dir, exist_ok=True)
    output_dir = os.path.join(result_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cfg.batch_size = 512
    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # model
    model = CameraVAE().cuda()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    zs = []
    zs_t = []
    cam_tags = []
    cam_positions = []
    pbar = tqdm(trainer.batch_generator)
    with torch.no_grad():
        for data in pbar:
            data_name = data['data_name']
            traj = data['cam_param'].cuda()
            E0 = data['first_extrinsic'].cuda()
            scale = data['avg_scale'].cuda()
            width = data['width'].cuda()
            height = data['height'].cuda()
            intrinsics = data['intrinsics'].cuda()
            extrinsics = data['extrinsics']
            cam_tag = data['cam_tag']
            cam_tags += cam_tag.tolist()

            cam_positions += list(extrinsics[:, :, :3, 3].unbind(dim=0))
            
            # out, _, _ = model(traj)
            z = model.encode(traj)
            # traj_hat, mu, log_var = model(traj)
            # z = model.reparameterize(mu, log_var)
            # z = z.transpose(-1, -2)
            zs.append(z.reshape(-1))
            zs_t.append(z)
            # continue
            
            
            # # Random sampling
            # z = torch.randn_like(z).to(z)
            # # # z = model.reparameterize(mu, log_var)
            # # # z = z.transpose(-1, -2)
            # # print(z.shape)
            
            # out = model.decode(z)
            # val_loss = torch.abs(traj - out).mean()

            # pred_traj = out_to_trajectory(out[0:1], scale[0:1], E0[0:1], device)
            # gt_traj = out_to_trajectory(traj[0:1], scale[0:1], E0[0:1], device)

            # pred_intrinsics = make_intrinsics(out[0:1, :, -2:], width[0:1], height[0:1], intrinsics[0:1])
            # gt_intrinsics = make_intrinsics(traj[0:1, :, -2:], width[0:1], height[0:1], intrinsics[0:1])

            # pred_path = os.path.join(result_dir, f'{os.path.basename(data_name[0])}_pred.npz')
            # np.savez(pred_path, poses=pred_traj[0].cpu().numpy(), intrinsics=pred_intrinsics[0].cpu().numpy())
            # gt_path = os.path.join(result_dir, f'{os.path.basename(data_name[0])}_gt.npz')
            # np.savez(gt_path, poses=gt_traj[0].cpu().numpy(), intrinsics=gt_intrinsics[0].cpu().numpy())

            # print("Loss:", val_loss.item())
            # print(pred_path)
            # print(gt_path)
            # input()

            # # Feature Interpolation
            # if z.shape[0] == 2:
            #     feature_a, feature_b = z[0], z[1]
            #     print(data_name[0], data_name[1])
            #     for i in range(0, 11):
            #         t = i / 10
            #         interpolated_feat = feature_a * t + feature_b * (1 - t)
            #         out = model.decode(interpolated_feat[None])
            #         # scale = torch.ones((1, 1)).to(device)
            #         # E0 = torch.eye(4)[None].to(device)
            #         interpolated_traj = out_to_trajectory(out, scale[0:1], E0[0:1], device)
            #         interpolated_traj_path = os.path.join(result_dir, f'{data_name[0]}_{data_name[1]}_{i}_{10-i}.npy')
            #         np.save(interpolated_traj_path, interpolated_traj[0].cpu().numpy())
            #         print(interpolated_traj_path)

            # input()

    z_all = torch.cat(zs)
    s = z_all.reshape(-1).std()
    print(f"Data std: {s.item()}")
    input()

    z_all = torch.cat(zs_t)
    z_all = z_all.reshape(z_all.shape[0], -1)
    print(z_all.shape)
    # print(z_all.var(dim=0))
    # print(z_all.mean(dim=0))
    var = z_all.var(dim=0)
    sorted_var, idx = torch.sort(var, descending=True)
    print(sorted_var)

    target_dir = 'vae_latent_vis'
    dataset = 'dl3dv+da3'
    tag_type_list = ['trans+rot', 'trans', 'rot']
    label_num_list = [189, 27, 7]
    tag_idx = 1

    tag_type = tag_type_list[tag_idx]
    label_num = label_num_list[tag_idx] # [189, 27, 7]
    suffix = f"{dataset}_{tag_type}"

    labels = np.array(cam_tags)   # (M,)

    # 1. unique label 찾기
    unique_labels = np.arange(0, label_num, 1)

    # 2. 색상 생성
    cmap = plt.cm.get_cmap('hsv', len(unique_labels))

    fig, ax = plt.subplots(figsize=(8,1))

    for i, l in enumerate(unique_labels):
        ax.scatter(i, 0, color=label_to_color[l], s=200)
        ax.text(i, -0.2, str(l), ha='center')

    ax.set_xlim(-1, len(unique_labels))
    ax.set_ylim(-1, 1)
    ax.axis("off")
    plt.title("Label Color Map")
    plt.savefig("label_cmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # # 3. label -> color mapping
    # label_to_color = {l: cmap(i) for i, l in enumerate(unique_labels)}

    # 4. 각 샘플 색 지정
    colors = [label_to_color[l] for l in labels]
    print(len(colors))

    points = torch.stack(cam_positions) * 3 # (B, T, 3)
    print(points.shape)
    colors_expanded = torch.tensor(colors)[:, None, :3].expand(points.shape[0], points.shape[1], 3)
    scene = torch.cat([points, colors_expanded], dim=-1)
    scene = scene.reshape(-1, 6)
    print(scene.shape)
    torch.save(scene, 'cam_pos.pt')
    print("scene saved")
    input()

    z_np = z_all.detach().cpu().numpy()
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_np)
    print(f"PCA variance ratio: {pca.explained_variance_ratio_}")

    plt.figure(figsize=(6,6))
    plt.scatter(z_pca[:,0], z_pca[:,1], c=colors, s=5, alpha=0.6)
    plt.title("VAE Latent PCA")

    plt.savefig(os.path.join(target_dir, f"pca_{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    tsne = TSNE(
        n_components=2,
        perplexity=30,      # 10~50 보통
        learning_rate='auto',
        init='pca',
        max_iter=1000,
        random_state=42,
    )
    z_2d = tsne.fit_transform(z_np)
    plt.figure(figsize=(6,6))
    plt.scatter(z_2d[:,0], z_2d[:,1], c=colors, s=5, alpha=0.6)
    plt.title("VAE Latent t-SNE")
    plt.savefig(os.path.join(target_dir, f"tsne_{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Finish")



if __name__ == "__main__":
    infer()
