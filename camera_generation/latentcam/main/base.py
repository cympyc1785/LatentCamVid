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

        dataset = CamDataset(cfg=self.cfg)
        tr_list = getattr(self.cfg, 'train_seg_list', None)
        te_list = getattr(self.cfg, 'test_seg_list', None)
        if tr_list and te_list:
            # Explicit segment-list split (deterministic): train/val come from the given
            # <batch>/<hash>/<seg_key> lists instead of a random 90/10. val order follows the
            # test-list file order (shuffle=False) so validation = its first N segments.
            def _seg_key(sample_id):                     # "<batch>_<hash>_<seg>" -> "<batch>/<hash>/<seg>"
                bh, seg = sample_id.rsplit('_', 1)
                batch, h = bh.split('_', 1)
                return f"{batch}/{h}/{seg}"
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
