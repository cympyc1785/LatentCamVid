# Ported from SCVideo core_pkg/core_pkg/common/base.py.
# Import fixes for the latentcam layout: core_pkg.* -> models.* / utils.*,
# dataset -> dataset_dl3dv (DL3DV-960 via meta.csv, blacklist excluded).
import os
import os.path as osp
import math
import time
import glob
import abc
from collections.abc import Mapping, Sequence
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data.dataloader import default_collate
from utils.pc_utils import get_ray_sim_per_point
from torch.utils.data import random_split, Subset
from models.camera_diffusion_model_latent import get_model
from dataset_dl3dv import CamDataset
from torch.nn.utils.rnn import pad_sequence


def build_dataset(cfg):
    """[new 2026-08-10] cfg.dataset_name 으로 코퍼스를 고른다. 기본 'dl3dv' 는 기존 동작 그대로.
      'dl3dv'           dataset_dl3dv.CamDataset            (scene 안의 다른 프레임 구간 = context)
      'scene_decoupled' dataset_scene_decoupled.SDCamDataset (같은 scene 의 다른 clip = context)
      'datadop'         dataset_datadop.DataDoPCamDataset    (shot 의 frame0 한 장 = context, V=1)
      'mixed'           dataset_mixed.MixedCamDataset        (위 셋을 cfg.datasets 블록으로 혼합)
    """
    name = getattr(cfg, 'dataset_name', None) or 'dl3dv'
    if name == 'dl3dv':
        return CamDataset(cfg=cfg)
    if name == 'scene_decoupled':
        from dataset_scene_decoupled import SDCamDataset      # 지연 import (기존 경로 영향 X)
        return SDCamDataset(cfg=cfg)
    if name == 'datadop':
        from dataset_datadop import DataDoPCamDataset
        return DataDoPCamDataset(cfg=cfg)
    if name == 'mixed':
        from dataset_mixed import MixedCamDataset
        return MixedCamDataset(cfg=cfg)
    raise ValueError(f"dataset_name must be 'dl3dv' | 'scene_decoupled' | 'datadop' | 'mixed', "
                     f"got {name!r}")


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        self.cur_epoch = 0
        self.cfg = cfg

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):

    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg=cfg, log_name='train_logs.txt')

    def get_optimizer(self, optimizable_parameters):
        self.optimizer = torch.optim.Adam(optimizable_parameters, lr=self.cfg.lr, eps=1e-15)

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self, include_train=True, include_val=True):
        generator = torch.Generator().manual_seed(self.cfg.random_seed)

        dataset = build_dataset(self.cfg)
        if getattr(self.cfg, 'dataset_name', None) == 'mixed':
            # [new 2026-08-16] 혼합 경로. 아래 두 경로(seg-list / random_split)는 **한 바이트도**
            # 안 건드리도록 여기서 먼저 return 한다. 분할은 MixedCamDataset 이 코퍼스마다 자기
            # 규칙으로 끝내 뒀고(train_idx/val_idx), 여기서는 배치 샘플러만 붙인다.
            return self._make_mixed_loaders(dataset, include_train, include_val)
        tr_list = getattr(self.cfg, 'train_seg_list', None)
        te_list = getattr(self.cfg, 'test_seg_list', None)
        if tr_list and te_list:
            # Explicit segment-list split (deterministic): train/val come from the given
            # <batch>/<hash>/<seg_key> lists instead of a random 90/10. val order follows the
            # test-list file order (shuffle=False) so validation = its first N segments.
            # 리스트 id <-> data_name 변환은 dataset.seg_key 가 안다 (DL3DV = 슬래시 경로 복원,
            # Scene-Decoupled = 항등).
            _seg_key = type(dataset).seg_key
            id2idx = {}
            for i, s in enumerate(dataset.samples):
                id2idx.setdefault(_seg_key(s[4]), i)
            def _load(p):
                with open(p) as f:
                    return [ln.strip() for ln in f if ln.strip()]
            tr_ids, te_ids = _load(tr_list), _load(te_list)
            tr_idx = [id2idx[x] for x in tr_ids if x in id2idx]
            te_idx = [id2idx[x] for x in te_ids if x in id2idx]
            self.trainset_loader = Subset(dataset, tr_idx)
            self.validset_loader = Subset(dataset, te_idx)
            print(f"[seg-list split] train {len(tr_idx)}/{len(tr_ids)} , val {len(te_idx)}/{len(te_ids)} "
                  f"(ids missing from dataset skipped)")
        else:
            # cfg.train_frac (default 0.9) = the random split's train share. 1.0 puts the WHOLE
            # index in train and leaves val empty -- only valid for jobs that never validate
            # (train_vae_dl3dv.py calls _make_batch_generator(include_val=False)), which is why
            # the VAE fits use it: holding out 10% of the corpus buys nothing there.
            frac = float(getattr(self.cfg, 'train_frac', 0.9))
            train_size = len(dataset) if frac >= 1.0 else int(frac * len(dataset))
            test_size = len(dataset) - train_size
            self.trainset_loader, self.validset_loader = random_split(
                dataset,
                [train_size, test_size],
                generator=generator
            )
            if frac >= 1.0:
                print(f"[split] train_frac=1.0 -> train {train_size} / val 0 (no held-out split)")

        if include_train:
            self.batch_generator = DataLoader(
                dataset=self.trainset_loader,
                batch_size=self.cfg.num_gpus * self.cfg.batch_size,
                shuffle=True, num_workers=self.cfg.num_thread,
                persistent_workers=True, drop_last=True, collate_fn=collate_fn)
            self.itr_per_epoch = math.ceil(len(self.trainset_loader) / self.cfg.num_gpus / self.cfg.batch_size)
        if include_val:
            self.valid_batch_generator = DataLoader(
                dataset=self.validset_loader,
                batch_size=self.cfg.num_gpus * self.cfg.batch_size,
                shuffle=False, num_workers=self.cfg.num_thread,
                persistent_workers=True, collate_fn=collate_fn)

    def _make_mixed_loaders(self, dataset, include_train, include_val):
        """dataset_name='mixed' 전용 DataLoader 구성. 배치 하나 = 코퍼스 하나.

        DataLoader 에 `batch_sampler` 를 주면 `batch_size`/`shuffle`/`drop_last` 를 **같이 주면
        안 된다** (거부한다) — 그래서 세 인자가 여기엔 없다.
        """
        from mixed_sampler import PerCorpusBatchSampler
        bs = self.cfg.num_gpus * self.cfg.batch_size
        seed = int(getattr(self.cfg, 'random_seed', 42))
        n_c = len(dataset.names)
        self.trainset_loader = Subset(dataset, dataset.train_idx)
        self.validset_loader = Subset(dataset, dataset.val_idx)
        if include_train:
            sampler = PerCorpusBatchSampler(
                dataset.corpus_of_positions(dataset.train_idx), batch_size=bs,
                weights=dataset.weights, shuffle=True, drop_last=True,
                seed=seed, num_corpora=n_c)
            self.batch_generator = DataLoader(
                dataset=self.trainset_loader, batch_sampler=sampler,
                num_workers=self.cfg.num_thread, persistent_workers=True,
                collate_fn=collate_fn)
            self.itr_per_epoch = len(sampler)
            print(f"[mixed loader] train {len(sampler)} batches (bs {bs}) "
                  f"per corpus {sampler._n_batches}")
        if include_val:
            # drop_last=True 는 val 에서도 **의도적**이다: accelerate 1.12 의 BatchSamplerShard
            # 는 drop_last=False 일 때만 tail padding 을 도는데, 그 padding 이 앞 배치들을
            # 평탄화한 인덱스에서 채워 와 **코퍼스가 섞인 배치**를 만든다 (data_loader.py:223,258).
            # 근거는 mixed_sampler.py 모듈 docstring 참고. 잃는 건 코퍼스당 최대 bs-1 개.
            v_sampler = PerCorpusBatchSampler(
                dataset.corpus_of_positions(dataset.val_idx), batch_size=bs,
                weights=None, shuffle=False, drop_last=True,
                interleave=bool(getattr(self.cfg, 'mix_val_interleave', True)),
                seed=seed, num_corpora=n_c)
            self.valid_batch_generator = DataLoader(
                dataset=self.validset_loader, batch_sampler=v_sampler,
                num_workers=self.cfg.num_thread, persistent_workers=True,
                collate_fn=collate_fn)
            print(f"[mixed loader] val {len(v_sampler)} batches "
                  f"(interleave={v_sampler.interleave}) per corpus {v_sampler._n_batches}")

    def _make_model(self):
        model = get_model()
        model = DataParallel(model).cuda()
        model.eval()
        self.model = model


def collate_fn(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        new_batch = {}
        for key in batch[0]:
            if key == 'pc_sim':
                sim_list = [d[key] for d in batch]
                new_batch['pc_sim'] = pad_sequence(sim_list, batch_first=True)
            elif key == "pc_embeds":
                embeds_list = [d[key] for d in batch]
                pc_embeds = pad_sequence(embeds_list, batch_first=True)
                lengths = torch.tensor([x.shape[0] for x in embeds_list])
                masks = torch.arange(pc_embeds.shape[1])[None, :] < lengths[:, None]
                masks = masks.to(pc_embeds).bool()
                new_batch["pc_embeds"] = pc_embeds
                new_batch["pc_masks"] = masks
            elif key == "pc":
                new_batch[key] = point_collate_fn([d[key] for d in batch])
            elif key == 'points' or key == 'inverse' or key == 'points_mask':
                new_batch[key] = [d[key] for d in batch]
            else:
                new_batch[key] = collate_fn([d[key] for d in batch])
        return new_batch
    else:
        return default_collate(batch)


def point_collate_fn(batch):
    """collate function for point cloud which support dict and list."""
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [point_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {
            key: (
                point_collate_fn([d[key] for d in batch])
                if "offset" not in key
                else torch.cumsum(
                    point_collate_fn(
                        [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                    ),
                    dim=0,
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)
