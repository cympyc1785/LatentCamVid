import numpy as np
import torch
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

SCENE_CUTTING_THRESHOLD = 75
MAX_POINTS = 16_000_000

def normalize_camera_extrinsics_and_points(extrinsics, points=None, avg_scale=None, max_trans_norm=False):
    """
    extrinsics: (N, 4, 4) torch.Tensor (w2c)
    points: (M, 3) torch.Tensor

    normalized_extrinsics: (N, 4, 4)
    normalized_points: (M, 3)
    avg_scale: (1,)
    """
    orig_dtype = extrinsics.dtype
    

    # inverse in float32 for stability (important for bf16)
    e0_inv = torch.linalg.inv(extrinsics[0].float()).to(orig_dtype)

    # normalize extrinsics
    normalized_extrinsics = extrinsics @ e0_inv.unsqueeze(0)

    # normalize points
    R = extrinsics[0, :3, :3]
    t = extrinsics[0, :3, 3]

    normalized_points = None
    mask = None
    
    if points is not None and avg_scale is None:
        points = points.to(dtype=extrinsics.dtype, device=extrinsics.device)
        new_points = points @ R.T + t
        # scale: (1,)
        if max_trans_norm:
            mean_avg_scale = new_points.norm(dim=-1).mean().unsqueeze(0)
            normalized_points = new_points / mean_avg_scale
            normalized_extrinsics[:, :3, 3] /= mean_avg_scale

            normalized_c2ws = torch.linalg.inv(normalized_extrinsics)
            avg_scale = normalized_c2ws[:, :3, 3].norm(dim=-1).max().unsqueeze(0) + 1e-5
        else:
            avg_scale = new_points.norm(dim=-1).mean().unsqueeze(0)
            normalized_points = new_points / avg_scale
        # scene cutting
        d = torch.norm(normalized_points, dim=1)
        r = 1.25
        mask = d <= r
    
    normalized_extrinsics[:, :3, 3] /= avg_scale

    return normalized_extrinsics, normalized_points, avg_scale, mask

def out_to_trajectory(out, scale, e0, device=None, max_trans_norm=False, trans_repr='w2c'):

    """
    out: (B, N, 9)
    scale: (B, 1)
    e0: (B, 4, 4)

    matrix_trajectory: (B, N, 4, 4)

    trans_repr: out[..., 6:9] 가 담고 있는 값. dataset_dl3dv 의 cfg.trans_repr 과 반드시 일치해야 한다.
        'w2c' (기본, 기존 동작) w2c translation t -> 그대로 행렬의 [:3,3] 에 넣는다.
        'c2w'                  카메라 중심 c      -> t = -R c 로 바꿔서 넣는다.
    """
    if device is None:
        device = out.device
    if max_trans_norm:
        max_trans_scale = out[:, -1, 0:1]
    if out.shape[-1] > 9:
        out = out[:, :, :9]
    matrix_trajectory = torch.eye(4, device=device)[None].repeat(out.shape[1], 1, 1)[None].repeat(out.shape[0], 1, 1, 1)

    if max_trans_norm:
        # raw_trans = out[:, :, 6:] * scale.unsqueeze(-1) * max_trans_scale.exp().unsqueeze(-1)
        raw_trans = out[:, :, 6:] * scale.unsqueeze(-1) * max_trans_scale.unsqueeze(-1)
    else:
        raw_trans = out[:, :, 6:] * scale.unsqueeze(-1)

    rot6d = out[:, :, :6]
    for idx, rot in enumerate(rot6d):
        raw_rot = compute_rotation_matrix_from_ortho6d(rot)
        matrix_trajectory[idx, :, :3, :3] = raw_rot
    if trans_repr == 'c2w':
        # raw_trans 는 카메라 중심 c. w2c 행렬을 만들려면 t = -R c (R 은 위에서 채운 회전).
        raw_trans = -torch.einsum('bnij,bnj->bni', matrix_trajectory[:, :, :3, :3], raw_trans)
    elif trans_repr != 'w2c':
        raise ValueError(f"trans_repr must be 'w2c' | 'c2w', got {trans_repr!r}")
    matrix_trajectory[:, :, :3, 3] = raw_trans
    matrix_trajectory = matrix_trajectory @ e0.unsqueeze(1)

    return matrix_trajectory

def make_intrinsics(fx_fy, width, height, intrinsics=None):
    """
    fx_fy:   (B, N, 2)
    width:   (B, N)
    height:  (B, N)

    K: (B, N, 3, 3)
    """
    assert fx_fy.shape[-1] == 2
    assert fx_fy.shape[:2] == width.shape == height.shape

    device = fx_fy.device
    dtype = fx_fy.dtype

    if intrinsics is None:
        fx = fx_fy[..., 0] * width
        fy = fx_fy[..., 1] * height
    else:
        fx = fx_fy[..., 0] * intrinsics[..., 0, 0]
        fy = fx_fy[..., 1] * intrinsics[..., 1, 1]

    cx = width * 0.5
    cy = height * 0.5

    B, N = width.shape

    K = torch.zeros((B, N, 3, 3), device=device, dtype=dtype)

    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = cx
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0

    return K

def inverse_camera_matrix(camera_matrix):
    """
    camera_matrix: (..., 4, 4)

    inversed_camera_matrix: (..., 4, 4)
    """
    inversed_camera_matrix = torch.zeros_like(camera_matrix).to(camera_matrix)
    inversed_camera_matrix[..., :3, :3] = camera_matrix[..., :3, :3].transpose(-1, -2)
    inversed_camera_matrix[..., :3, 3:] = -camera_matrix[..., :3, :3].transpose(-1, -2) @ camera_matrix[..., :3, 3:]
    inversed_camera_matrix[..., 3, 3] = 1.0
    
    return inversed_camera_matrix