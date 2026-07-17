import os
import numpy as np
import torch
from typing import Union
from dataclasses import dataclass
from plyfile import PlyData, PlyElement

SHS_REST_DIM_TO_DEGREE = {
    0: 0,
    3: 1,
    8: 2,
    15: 3,
}


@dataclass
class GaussianPlyUtils:
    """
    Load parameters from ply;
    Save to ply;
    """

    sh_degrees: int
    xyz: Union[np.ndarray, torch.Tensor]  # [n, 3]
    opacities: Union[np.ndarray, torch.Tensor]  # [n, 1]
    features_dc: Union[np.ndarray, torch.Tensor]  # ndarray[n, 3, 1], or tensor[n, 1, 3]
    features_rest: Union[np.ndarray, torch.Tensor]  # ndarray[n, 3, 15], or tensor[n, 15, 3]; NOTE: this is features_rest actually!
    scales: Union[np.ndarray, torch.Tensor]  # [n, 3]
    rotations: Union[np.ndarray, torch.Tensor]  # [n, 4]

    @staticmethod
    def detect_sh_degree_from_shs_rest(shs_rest: torch.Tensor):
        assert isinstance(shs_rest, torch.Tensor)
        return SHS_REST_DIM_TO_DEGREE[shs_rest.shape[-2]]

    @staticmethod
    def load_array_from_plyelement(plyelement, name_prefix: str, required: bool = True):
        names = [p.name for p in plyelement.properties if p.name.startswith(name_prefix)]
        if len(names) == 0:
            if required is True:
                raise RuntimeError(f"'{name_prefix}' not found in ply")
            return np.empty((plyelement["x"].shape[0], 0))
        names = sorted(names, key=lambda x: int(x.split('_')[-1]))
        v_list = []
        for idx, attr_name in enumerate(names):
            v_list.append(np.asarray(plyelement[attr_name]))

        return np.stack(v_list, axis=1)

    @classmethod
    def load_from_ply(cls, path: str, sh_degrees: int = -1):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        features_rest = cls.load_array_from_plyelement(plydata.elements[0], "f_rest_", required=False).reshape((xyz.shape[0], 3, -1))
        if sh_degrees >= 0:
            assert features_rest.shape[-1] == (sh_degrees + 1) ** 2 - 1  # TODO: remove such a assertion
        else:
            # auto determine sh_degrees
            features_rest_dims = features_rest.shape[-1]
            for i in range(4):
                if features_rest_dims == (i + 1) ** 2 - 1:
                    sh_degrees = i
                    break
            assert sh_degrees >= 0, f"invalid sh_degrees={sh_degrees}"

        scales = cls.load_array_from_plyelement(plydata.elements[0], "scale_")
        rots = cls.load_array_from_plyelement(plydata.elements[0], "rot_")

        return cls(
            sh_degrees=sh_degrees,
            xyz=xyz,
            opacities=opacities,
            features_dc=features_dc,
            features_rest=features_rest,
            scales=scales,
            rotations=rots,
        )

    @classmethod
    def load_from_model_properties(cls, properties, sh_degree: int = -1):
        if sh_degree < 0:
            sh_degree = cls.detect_sh_degree_from_shs_rest(properties["shs_rest"])

        init_args = {
            "sh_degrees": sh_degree,
        }

        for name_in_model, name_in_dataclass in [
            ("means", "xyz"),
            ("shs_dc", "features_dc"),
            ("shs_rest", "features_rest"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacities", "opacities"),
        ]:
            init_args[name_in_dataclass] = properties[name_in_model].detach()

        return cls(**init_args)

    @classmethod
    def load_from_model(cls, model):
        return cls.load_from_model_properties(model.properties, sh_degree=model.max_sh_degree)

    @classmethod
    def load_from_state_dict(cls, state_dict):
        if "gaussian_model.gaussians.means" in state_dict:
            return cls.load_from_new_state_dict(state_dict)
        return cls.load_from_old_state_dict(state_dict)

    @classmethod
    def load_from_new_state_dict(cls, state_dict):
        prefix = "gaussian_model.gaussians."

        init_args = {
            "sh_degrees": cls.detect_sh_degree_from_shs_rest(state_dict["{}shs_rest".format(prefix)]),
        }

        for name_in_dict, name_in_dataclass in [
            ("means", "xyz"),
            ("shs_dc", "features_dc"),
            ("shs_rest", "features_rest"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacities", "opacities"),
        ]:
            init_args[name_in_dataclass] = state_dict["{}{}".format(prefix, name_in_dict)]

        return cls(**init_args)

    @classmethod
    def load_from_old_state_dict(cls, state_dict):
        key_prefix = "gaussian_model._"

        init_args = {
            "sh_degrees": cls.detect_sh_degree_from_shs_rest(state_dict["{}features_rest".format(key_prefix)]),
        }
        for name_in_dict, name_in_dataclass in [
            ("xyz", "xyz"),
            ("features_dc", "features_dc"),
            ("features_rest", "features_rest"),
            ("scaling", "scales"),
            ("rotation", "rotations"),
            ("opacity", "opacities"),
        ]:
            init_args[name_in_dataclass] = state_dict["{}{}".format(key_prefix, name_in_dict)]

        return cls(**init_args)

    def to_parameter_structure(self):
        assert isinstance(self.xyz, np.ndarray) is True
        return GaussianPlyUtils(
            sh_degrees=self.sh_degrees,
            xyz=torch.tensor(self.xyz, dtype=torch.float),
            opacities=torch.tensor(self.opacities, dtype=torch.float),
            features_dc=torch.tensor(self.features_dc, dtype=torch.float).transpose(1, 2),
            features_rest=torch.tensor(self.features_rest, dtype=torch.float).transpose(1, 2),
            scales=torch.tensor(self.scales, dtype=torch.float),
            rotations=torch.tensor(self.rotations, dtype=torch.float),
        )

    @torch.no_grad()
    def to_ply_format(self):
        assert isinstance(self.xyz, torch.Tensor) is True
        return GaussianPlyUtils(
            sh_degrees=self.sh_degrees,
            xyz=self.xyz.cpu().numpy(),
            opacities=self.opacities.cpu().numpy(),
            features_dc=self.features_dc.transpose(1, 2).cpu().numpy(),
            features_rest=self.features_rest.transpose(1, 2).cpu().numpy(),
            scales=self.scales.cpu().numpy(),
            rotations=self.rotations.cpu().numpy(),
        )

    def save_to_ply(self, path: str, with_colors: bool = False):
        assert isinstance(self.xyz, np.ndarray) is True

        gaussian = self

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except:
            pass

        xyz = gaussian.xyz
        f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1))
        # TODO: change sh degree
        if gaussian.sh_degrees > 0:
            f_rest = gaussian.features_rest.reshape((gaussian.features_rest.shape[0], -1))
        else:
            f_rest = np.zeros((f_dc.shape[0], 0))
        opacities = gaussian.opacities
        scale = gaussian.scales
        rotation = gaussian.rotations

        # xyz
        dtype_full = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ]
        attribute_list = [
            ("x", xyz[..., 0]),
            ("y", xyz[..., 1]),
            ("z", xyz[..., 2]),
        ]

        def add_attribute(name_prefix, value):
            for i in range(value.shape[-1]):
                name = "{}_{}".format(name_prefix, i)
                dtype_full.append((name, "f4"))
                attribute_list.append((name, value[..., i]))

        # shs_dc
        add_attribute("f_dc", f_dc)
        # shs_rest
        add_attribute("f_rest", f_rest)
        # opacities
        dtype_full.append(("opacity", "f4"))
        attribute_list.append(("opacity", opacities.squeeze(-1)))
        # scales
        add_attribute("scale", scale)
        # rotations
        add_attribute("rot", rotation)

        if with_colors is True:
            from internal.utils.sh_utils import eval_sh
            rgbs = np.clip((eval_sh(0, self.features_dc, None) + 0.5), 0., 1.)
            rgbs = (rgbs * 255).astype(np.uint8)

            dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            attribute_list += [
                ("red", rgbs[..., 0]),
                ("green", rgbs[..., 1]),
                ("blue", rgbs[..., 2]),
            ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        for (k, v) in attribute_list:
            elements[k] = v
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)