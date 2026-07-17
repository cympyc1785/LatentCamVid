import torch
import torch.nn.functional as F
import trimesh
import numpy as np

def load_ply_as_pointcloud(path):
    mesh = trimesh.load(path)
    points = mesh.vertices
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0  # (N, 4) - RGBA uint8
    else:
        colors = np.ones_like(points)
    colors = np.clip(colors, 0.0, 1.0)
    pc = np.hstack([points, colors])
    return pc.astype(np.float32)

def offset2batch(offset):
    return (
        torch.cat(
            [
                (
                    torch.tensor([i] * (o - offset[i - 1]))
                    if i > 0
                    else torch.tensor([i] * o)
                )
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to(offset.device)
    )


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).int()

def pad_and_mask(xs):
    """
    xs: list of tensors, each shape (Ni, D)
    returns:
        padded: (B, N_max, D)
        attn_mask: (B, N_max)  # True = valid, False = padding
    """
    B = len(xs)
    D = xs[0].shape[1]
    device = xs[0].device
    dtype = xs[0].dtype

    lengths = torch.tensor([x.shape[0] for x in xs], device=device)
    N_max = lengths.max().item()

    padded = torch.zeros((B, N_max, D), device=device, dtype=dtype)
    attn_mask = torch.zeros((B, N_max), device=device, dtype=torch.bool)

    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        attn_mask[i, :n] = True

    return padded, attn_mask

def get_visible_point_indices(points, c2w, K, H, W):
    """
    points: (N, 3)
    c2w:    (M, 4, 4)
    K:      (M, 3, 3)
    H, W:   image height, width

    visible_indices: list of length M
                        each element is a 1D tensor of visible point indices
    depths:         (M, N)
    """

    device = points.device
    N = points.shape[0]
    M = c2w.shape[0]

    # (N, 4)
    points_h = torch.cat([points, torch.ones(N, 1, device=device)], dim=1)

    visible_indices = []
    depths = []

    for i in range(M):

        # world → camera
        w2c = torch.inverse(c2w[i])  # (4,4)
        cam_points = (w2c @ points_h.T).T  # (N,4)

        xyz = cam_points[:, :3]

        # depth > 0
        z = xyz[:, 2]
        valid_depth = z > 0

        # projection
        proj = (K[i] @ xyz.T).T  # (N,3)
        u = proj[:, 0] / z
        v = proj[:, 1] / z

        # image boundary check
        valid_u = (u >= 0) & (u < W)
        valid_v = (v >= 0) & (v < H)

        mask = valid_depth & valid_u & valid_v

        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        visible_indices.append(idx)
        depths.append(z)

    depths = torch.stack(depths)

    return visible_indices, depths

def get_ray_sim_per_point(cam_matrix, K, points, c2w=True, device="cuda:0"):
    """
    cam_matrix: (N, 4, 4)
    K: (N, 3, 3)
    points: (M, 3)
    """
    cam_matrix = cam_matrix.to(device)
    K = K.to(device)
    points = points.to(device)

    if c2w == False:
        w2c = cam_matrix
        c2w = torch.zeros_like(w2c).to(device)
        R_inv = w2c[:, :3, :3].transpose(1, 2)
        t_inv = -R_inv @ w2c[:, :3, 3:]
        c2w[:, :3, :3] = R_inv
        c2w[:, :3, 3:] = t_inv
        c2w[:, 3, 3] = 1.0
    else:
        c2w = cam_matrix

    H, W = int(K[0, 1, 2] * 2) , int(K[0, 0, 2] * 2)

    rays_camera = torch.tensor([0.0, 0.0, 1.0])[None].repeat(c2w.shape[0], 1).to(device) # (N, 3)
    rays_world = torch.einsum('nij,nj->ni', c2w[:, :3, :3], rays_camera) # (N, 3)
    c2p_vec = points[None, :, :] - c2w[:, :3, 3][:, None, :] # (N, M, 3)

    rays_world_norm = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)
    c2p_vec_norm = c2p_vec / torch.norm(c2p_vec, dim=-1, keepdim=True)

    cos_sim =  torch.einsum('ni,nmi->nm', rays_world_norm, c2p_vec_norm) # (N, M)
    cos_sim = (cos_sim + 1.0) / 2.0

    visible_indices, _ = get_visible_point_indices(points, c2w, K, H, W)
    visible_indices_unique = torch.unique(torch.cat(visible_indices, dim=0))

    # Zero sim for invisible points
    cos_sim = cos_sim / 0.1
    masked_sim = torch.zeros_like(cos_sim).to(cos_sim) - 1e9
    masked_sim[:, visible_indices_unique] = cos_sim[:, visible_indices_unique]
    cos_sim = masked_sim # (N, M)

    cos_sim = cos_sim.unsqueeze(0).transpose(1, 2)  # (1, M, 4T+1)

    cos_sim = F.avg_pool1d(cos_sim, kernel_size=3, stride=2, padding=1)  # → (1, M, 2T+1)
    cos_sim = F.avg_pool1d(cos_sim, kernel_size=3, stride=2, padding=1)  # → (1, M, T+1)

    ray_sim_per_point = cos_sim[0].transpose(0, 1) # (T+1, M) (T=N//4+1)

    visibility_in_e0 = torch.zeros(points.shape[0]).to(device)
    visibility_in_e0[visible_indices[0]] = 1.0

    return ray_sim_per_point, visibility_in_e0