"""Hydra/OmegaConf config loader for latentcam (conf/config.yaml base + conf/experiment/*.yaml).

Standard Hydra-style CLI overrides:  python train_latent_cam_dm.py experiment=rolling lr=1e-4
Back-compat env:                     LATENTCAM_CONFIG=config_rolling python train_latent_cam_dm.py
  (the `config_` prefix is stripped -> experiment name; `config` -> base only.)

load_cfg() returns (cfg, cfg_dict):
  cfg      : a SimpleNamespace that behaves like the old Config instance — attribute access
             (cfg.num_cam), getattr(cfg, 'x', default), in-place mutation (cfg.max_scenes = N),
             and t5_dtype as a real torch.dtype (so existing consumers are unchanged).
  cfg_dict : plain dict (t5_dtype kept as the string 'bfloat16') for wandb config + YAML dump.
"""
import os, sys, re, random
import os.path as osp
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_MAIN = osp.dirname(osp.abspath(__file__))
_CONF = osp.join(_MAIN, 'conf')
_LATENTCAM = osp.abspath(osp.join(_MAIN, '..'))
_DL3DV_DATA_PATH = '/data1/cympyc1785/SceneData/DL3DV/scenes'   # (mirrors config.build_dataset_dir_list)
_OVERRIDE_RE = re.compile(r'^[A-Za-z_][\w.]*=')                 # hydra override token (not argparse --flag)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _build_dataset_dir_list():
    if os.path.isdir(_DL3DV_DATA_PATH):
        return [osp.join(_DL3DV_DATA_PATH, s) for s in sorted(os.listdir(_DL3DV_DATA_PATH))]
    print(f"[hydra_cfg] WARNING: DL3DV data path not found: {_DL3DV_DATA_PATH} (dataset_dir_list empty)")
    return []


def _finalize_dict(d):
    """Recompute the runtime-only fields that were excluded from the YAML."""
    d['cur_dir'] = _MAIN
    d['root_dir'] = _LATENTCAM
    d['data_dir'] = osp.join(_LATENTCAM, 'data')
    d['ckpt_root'] = osp.join(_LATENTCAM, 'checkpoints')
    d['dataset_dir_list'] = _build_dataset_dir_list()
    return d


def load_cfg(default='config', overrides=None):
    """Compose conf/config.yaml (+ experiment) with CLI/env overrides.
    default: fallback config name (e.g. 'config_rolling') when neither argv nor env selects one."""
    ov = list(overrides) if overrides else [t for t in sys.argv[1:] if _OVERRIDE_RE.match(t)]
    has_exp = any(o.startswith('experiment=') for o in ov)
    if not has_exp:
        env = os.environ.get('LATENTCAM_CONFIG') or default
        exp = env[len('config_'):] if env.startswith('config_') else env
        if exp and exp != 'config':
            ov.append(f'experiment={exp}')
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONF, version_base=None):
        oc = compose(config_name='config', overrides=ov)
    d = OmegaConf.to_container(oc, resolve=True)
    d.pop('defaults', None)
    _finalize_dict(d)
    cfg_dict = dict(d)                                  # for wandb/yaml (t5_dtype as str)
    ns = dict(d)
    ns['t5_dtype'] = getattr(torch, d['t5_dtype'])      # real torch.dtype for consumers
    cfg = SimpleNamespace(**ns)
    set_seed(getattr(cfg, 'random_seed', 42))
    return cfg, cfg_dict


def save_cfg_yaml(cfg_dict, path):
    """Dump the full resolved config as YAML (t5_dtype string form; drop torch objects)."""
    safe = {k: (str(v).replace('torch.', '') if isinstance(v, torch.dtype) else v)
            for k, v in cfg_dict.items()}
    os.makedirs(osp.dirname(path), exist_ok=True)
    OmegaConf.save(OmegaConf.create(safe), path)
    return path
