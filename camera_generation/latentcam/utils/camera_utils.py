import os
import torch
import json
import numpy as np
import argparse
import pycolmap

from scipy.spatial.transform import Rotation

from typing import Optional, Union
from dataclasses import dataclass, field

import torch
from torch import Tensor

INTRINSIC_SCALING = 4.0

class CameraType:
    PERSPECTIVE: int = 0
    FISHEYE: int = 1


@dataclass
class Camera:
    idx: Tensor
    R: Tensor  # [3, 3]
    T: Tensor  # [3]
    fx: Tensor
    fy: Tensor
    fov_x: Tensor
    fov_y: Tensor
    cx: Tensor
    cy: Tensor
    width: Tensor
    height: Tensor
    appearance_id: Tensor
    normalized_appearance_id: Tensor
    time: Tensor

    distortion_params: Optional[Tensor]
    """
    NOTE: this should be None or a zero tensor currently
        
    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor

    world_to_camera: Tensor
    camera_to_world: Tensor
    projection: Tensor
    full_projection: Tensor
    camera_center: Tensor

    def to_device(self, device):
        for field in Camera.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))

        return self

    def get_K(self):
        K = torch.eye(4, dtype=torch.float, device=self.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy

        return K

    def get_full_perspective_projection(self):
        K = self.get_K()

        # full.transpose() = (K[R T]).transpose() = [R T].transpose() K.transpose()

        return self.world_to_camera @ K.T        

    @property
    def device(self):
        return self.R.device

@dataclass
class CamerasInterface:
    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor = field(init=False)  # [n_cameras]
    fov_y: Tensor = field(init=False)  # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]
    appearance_id: Tensor  # [n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [n_cameras]
    distortion_params: Optional[Union[Tensor, list[Tensor]]]
    camera_type: Tensor  # Int[n_cameras]

@dataclass
class Cameras:
    """
    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor = field(init=False)  # [n_cameras]
    fov_y: Tensor = field(init=False)  # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]
    appearance_id: Tensor  # [n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [n_cameras]

    distortion_params: Optional[Union[Tensor, list[Tensor]]]
    """
    NOTE: this should be None or zero tensors currently

    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor  # Int[n_cameras]

    world_to_camera: Tensor = field(init=False)  # [n_cameras, 4, 4], transposed
    camera_to_world: Tensor = field(init=False)
    projection: Tensor = field(init=False)
    full_projection: Tensor = field(init=False)
    camera_center: Tensor = field(init=False)

    time: Optional[Tensor] = None  # [n_cameras]

    idx: Tensor = None  # [N_cameras]

    def _calculate_fov(self):
        # calculate fov
        self.fov_x = 2 * torch.atan((self.width / 2) / self.fx)
        self.fov_y = 2 * torch.atan((self.height / 2) / self.fy)

    def _calculate_w2c(self):
        # build world-to-camera transform matrix
        self.world_to_camera = torch.zeros((self.R.shape[0], 4, 4))
        self.world_to_camera[:, :3, :3] = self.R
        self.world_to_camera[:, :3, 3] = self.T
        self.world_to_camera[:, 3, 3] = 1.
        self.world_to_camera = torch.transpose(self.world_to_camera, 1, 2)
    
    def _calculate_c2w(self):
        # build camera-to-world transform matrix
        R_c2w = self.R.transpose(-1, -2)
        t_c2w = (- R_c2w @ self.T.unsqueeze(-1)).squeeze(-1)
        
        self.camera_to_world = torch.zeros((R_c2w.shape[0], 4, 4))
        self.camera_to_world[:, :3, :3] = R_c2w
        self.camera_to_world[:, :3, 3] = t_c2w
        self.camera_to_world[:, 3, 3] = 1.

    def _calculate_ndc_projection_matrix(self):
        """
        calculate ndc projection matrix
        http://www.songho.ca/opengl/gl_projectionmatrix.html

        TODO:
            1. support colmap refined principal points
            2. the near and far here are ignored in diff-gaussian-rasterization
        """
        zfar = 100.0
        znear = 0.01

        tanHalfFovY = torch.tan((self.fov_y / 2))
        tanHalfFovX = torch.tan((self.fov_x / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(self.fov_y.shape[0], 4, 4)

        z_sign = 1.0

        P[:, 0, 0] = 2.0 * znear / (right - left)  # = 1 / tanHalfFovX = 2 * fx / width
        P[:, 1, 1] = 2.0 * znear / (top - bottom)  # = 2 * fy / height
        P[:, 0, 2] = (right + left) / (right - left)  # = 0, right + left = 0
        P[:, 1, 2] = (top + bottom) / (top - bottom)  # = 0, top + bottom = 0
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        self.projection = torch.transpose(P, 1, 2)

        self.full_projection = self.world_to_camera.bmm(self.projection)

    def _calculate_camera_center(self):
        self.camera_center = torch.linalg.inv(self.world_to_camera)[:, 3, :3]

    def _calculate_K(self):
        self.K = torch.zeros((self.fx.shape[0], 3, 3))
        self.K[:, 0, 0] = self.fx
        self.K[:, 1, 1] = self.fy
        self.K[:, 0, 2] = self.cx
        self.K[:, 1, 2] = self.cy
        self.K[:, 2, 2] = 1.0

    def __post_init__(self):
        self._calculate_fov()
        self._calculate_w2c()
        self._calculate_c2w()
        self._calculate_ndc_projection_matrix()
        self._calculate_camera_center()

        self.idx = torch.arange(self.R.shape[0], dtype=torch.int32)

        if self.time is None:
            self.time = torch.zeros(self.R.shape[0])
        if self.distortion_params is None:
            self.distortion_params = torch.zeros(self.R.shape[0], 4)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Camera:
        return Camera(
            idx=self.idx[index],
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            fov_x=self.fov_x[index],
            fov_y=self.fov_y[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=self.normalized_appearance_id[index],
            distortion_params=self.distortion_params[index],
            time=self.time[index],
            camera_type=self.camera_type[index],
            world_to_camera=self.world_to_camera[index],
            camera_to_world=self.camera_to_world[index],
            projection=self.projection[index],
            full_projection=self.full_projection[index],
            camera_center=self.camera_center[index],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

@dataclass
class BatchCameras:
    """
    Batch of Cameras

    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [B, n_cameras, 3, 3]
    T: Tensor  # [B, n_cameras, 3]
    fx: Tensor  # [B, n_cameras]
    fy: Tensor  # [B, n_cameras]
    cx: Tensor  # [B, n_cameras]
    cy: Tensor  # [B, n_cameras]
    width: Tensor  # [B, n_cameras]
    height: Tensor  # [B, n_cameras]
    appearance_id: Tensor  # [B, n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [B, n_cameras]

    distortion_params: Optional[Union[Tensor, list[Tensor]]]
    """
    NOTE: this should be None or zero tensors currently

    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor  # Int[B, n_cameras]

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Cameras:
        return Cameras(
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=self.normalized_appearance_id[index],
            distortion_params=self.distortion_params[index],
            camera_type=self.camera_type[index],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def build_cameras(extrinsics, intrinsics):
    """
    Input:
        extrinsics: [N, 4, 4] numpy array
        intrinsics: [N, 3, 3] numpy array
    """
    N = extrinsics.shape[0]
    R_w2c = extrinsics[:, :3, :3]
    T_w2c = extrinsics[:, :3, 3]

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    width = intrinsics[:, 0, 2] * 2
    height = intrinsics[:, 1, 2] * 2
    # width = np.array([432] * N)
    # height = np.array([240] * N)
    appearance_id = torch.zeros((N), dtype=torch.int)
    normalized_appearance_id = torch.zeros((N), dtype=torch.int)
    distortion_params = torch.zeros((N, 4), dtype=torch.int)
    camera_type = torch.zeros((N), dtype=torch.int)

    cameras = Cameras(
        R=torch.from_numpy(R_w2c).float(),
        T=torch.from_numpy(T_w2c).float(),
        fx=torch.from_numpy(fx).float(),
        fy=torch.from_numpy(fy).float(),
        cx=torch.from_numpy(cx).float(),
        cy=torch.from_numpy(cy).float(),
        width=torch.from_numpy(width).int(),
        height=torch.from_numpy(height).int(),
        appearance_id=appearance_id,
        normalized_appearance_id=normalized_appearance_id,
        distortion_params=distortion_params,
        camera_type=camera_type,
    )
    return cameras

def get_camera_params_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    extrinsic_list = []
    intrinsic_list = []
    for params in data:
        R_w2c, t_w2c = params['rotation'], params['position']

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']

        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics

def get_colmap_camera_params(recon: pycolmap.Reconstruction):
    """
    Output:
        extrinsics: [N, 4, 4] numpy array (w2c)
        intrinsics: [N, 3, 3] numpy array (w2c)
    """
    sorted_images = sorted(
        recon.images.values(),
        key=lambda img: img.name
    )

    extrinsic_list = []
    intrinsic_list = []
    for image in sorted_images:
        extrinsic_dict = image.cam_from_world().todict()

        xyzw_w2c, t_w2c = extrinsic_dict['rotation']['quat'], extrinsic_dict['translation']
        R_w2c = Rotation.from_quat(xyzw_w2c).as_matrix()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy, cx, cy = image.camera.params
        fx /= INTRINSIC_SCALING
        fy /= INTRINSIC_SCALING
        cx /= INTRINSIC_SCALING
        cy /= INTRINSIC_SCALING
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics

def get_cameras_from_json(json_path):
    w2c_ext, intrinsics = get_camera_params_from_json(json_path)
    cameras = build_cameras(w2c_ext, intrinsics)
    return cameras

def get_cameras_from_colmap(colmap_path):
    sparse_path = os.path.join(colmap_path, 'sparse')
    recon = pycolmap.Reconstruction(sparse_path)
    extrinsics, intrinsics = get_colmap_camera_params(recon)
    cameras = build_cameras(extrinsics, intrinsics)
    return cameras

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str,
                        default="/data1/cympyc1785/colmap/SceneData/DL3DV/data/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/scene_recon",
                        help="scene_recon path")
    args = parser.parse_args()

    cameras = get_cameras_from_json(args.root_path)

    print("GS Cameras", len(cameras))
    print("===== First Camera =====")
    print("[ extrinsics (w2c) ]")
    print(cameras[0].R)
    print(cameras[0].T)
    print("[ intrinsics ]")
    print("fx fy cx cy:", cameras[0].fx.item(), cameras[0].fy.item(), cameras[0].cx.item(), cameras[0].cy.item())



    cameras = get_cameras_from_colmap(args.root_path)

    print("\nCOLMAP Cameras", len(cameras))
    print("===== First Camera =====")
    print("[ extrinsics (w2c) ]")
    print(cameras[0].R)
    print(cameras[0].T)
    print("[ intrinsics ]")
    print("fx fy cx cy:", cameras[0].fx.item(), cameras[0].fy.item(), cameras[0].cx.item(), cameras[0].cy.item())
