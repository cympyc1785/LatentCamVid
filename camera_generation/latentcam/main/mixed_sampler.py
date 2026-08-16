"""배치 하나 = 코퍼스 하나를 보장하는 BatchSampler.

왜 필요한가: 코퍼스마다 context view 수 V 가 다르다 (DL3DV≈6 / SD 6~49 / DataDoP 1). `collate_fn`
(`main/base.py:138`) 이 `torch.stack` 으로 묶으므로 한 배치에 섞이면 그 자리에서 터진다. 모델에
BatchNorm 류나 sample 간 결합 연산이 없어서(`models/camera_diffusion_model_latent.py` 확인)
배치를 코퍼스로 가르는 데 통계적 부작용은 없다 — 유일한 결합이 이 stack 이다.

문헌 근거(같은 결론에 도달한 선례): MV-DUSt3R `train.py:329` 는 배치 전체를 sample 0 의 view 수로
잘라 버리고, Pi3 의 `DynamicBatchSampler` 는 "배치 안의 sample 은 같은 aspect ratio 와 image 수를
공유한다"고 못박고, VGGT 는 "scene 당 2~24 프레임을 고르되 배치 총합 48 프레임을 유지"한다
(batch_size = floor(48/N)). 셋 다 **데이터셋은 배치 안에서 섞되 shape 는 절대 안 섞는다.**
PixArt-α/SD3 의 `AspectRatioBatchSampler` 도 같은 형태다. pad+mask 는 이 계열에서 드물다.

epoch 은 `__iter__` 안에서 self-advance 한다 — 학습 루프가 `set_epoch` 을 부르지 않기 때문
(`main/train_latent_cam_dm.py:878`). index sampler 는 main process 에 살아 있고 DataLoader 의
`_reset` 마다 `__iter__` 가 다시 호출되므로 `persistent_workers=True` 여도 안전하다.

**accelerate 와의 상호작용 — 설치본(1.12.0) 소스를 직접 읽고 확인한 것.**
`accelerator.prepare(dataloader)` 는 커스텀 batch_sampler 를 **반드시** `BatchSamplerShard` 로
감싼다 (opt-out 없음). `site-packages/accelerate/data_loader.py` 기준:
  L163  `self.batch_size = getattr(batch_sampler, 'batch_size', None)`
  L165  batch_size 가 None 인데 even_batches(기본 True) 면 **ValueError 로 죽는다**.
        -> 이 버전에서는 `.batch_size` 속성을 **반드시 노출해야 한다**. ("노출하지 말라"는
        조언은 PR #3969 가 들어간 더 최신 버전 이야기라 여기엔 해당 없음.)
  L170  `self.drop_last = getattr(batch_sampler, 'drop_last', False)`
  L223  `initial_data += batch`  -> 앞 num_processes 개 배치를 **평탄화**해 모아 둔다.
  L258  `batch += initial_data[cycle_index:end_index]`
        -> 마지막 짧은 배치를 그 평탄화된 인덱스로 채운다 = **코퍼스가 섞인 배치**.
  L236  단, 이 padding 블록 전체가 `if not self.drop_last:` 안에 있다.
=> 결론: **train/val 양쪽 다 `drop_last=True` 로 두면 padding 경로가 통째로 스킵되어 코퍼스
   동질성이 깨질 자리가 없다.** val 에서 drop_last=False 로 두면 (a) 위 tail padding 오염과
   (b) L230 의 `len(batch) == self.batch_size` 가드가 코퍼스별 partial 배치를 **조용히 버리는**
   두 가지가 동시에 터진다. 그래서 val 도 drop_last=True 가 기본이다 — 잃는 건 코퍼스당 최대
   batch_size-1 개이고, 어차피 `cfg.val_max_batches` 로 앞부분만 본다.
"""
import torch
from torch.utils.data import Sampler


class PerCorpusBatchSampler(Sampler):
    """모든 배치가 단일 코퍼스인 batch sampler.

    Args:
      corpus_of: len == len(dataset_or_subset). 각 **위치**의 corpus id.
                 (Subset 을 감싸므로 전역 idx 가 아니라 Subset 안의 위치다.)
      batch_size: 배치 크기 (= cfg.num_gpus * cfg.batch_size)
      weights: corpus id -> 가중치. 1.0 = 그 코퍼스를 한 epoch 에 정확히 한 바퀴.
               2.0 이면 두 바퀴(재셔플해서 반복), 0.5 면 절반만 (epoch 마다 다른 절반).
      shuffle: True = 학습용 (pool/배치 순서 셔플, drop_last). False = val 용 결정적.
      interleave: shuffle=False 일 때 코퍼스 round-robin 여부. val 이 `cfg.val_max_batches` 로
                  앞부분만 잘리므로(`train_latent_cam_dm.py:604-605`) False 면 한 코퍼스만 평가된다.
      seed: 기본 RNG seed. 실제 seed 는 (seed, epoch) 조합.
    """

    def __init__(self, corpus_of, batch_size, weights=None, shuffle=True,
                 drop_last=True, interleave=True, seed=42, num_corpora=None):
        self.corpus_of = list(corpus_of)
        self.batch_size = int(batch_size)        # accelerate 1.12 prepare_data_loader 가 읽는다
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.interleave = bool(interleave)
        self.seed = int(seed)
        self._epoch = 0

        n_c = num_corpora if num_corpora is not None else (max(self.corpus_of) + 1
                                                           if self.corpus_of else 0)
        self.pools = [[] for _ in range(n_c)]
        for pos, ci in enumerate(self.corpus_of):
            self.pools[ci].append(pos)
        self.weights = [1.0] * n_c if weights is None else [float(w) for w in weights]
        if len(self.weights) != n_c:
            raise ValueError(f"weights 길이 {len(self.weights)} != 코퍼스 수 {n_c}")

        # __len__ 은 __iter__ 와 **정확히** 같아야 한다 (itr_per_epoch 이 여기서 나온다).
        self._n_batches = [self._batches_for(ci) for ci in range(n_c)]
        self.num_batches = sum(self._n_batches)

    # ------------------------------------------------------------------ 배치 수
    def _batches_for(self, ci):
        n = len(self.pools[ci])
        if n == 0:
            return 0
        if not self.shuffle:
            # val: 있는 것 전부. drop_last=False 면 마지막 partial 배치도 낸다 (여전히 동질).
            return n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        eff = n * self.weights[ci]
        return int(eff // self.batch_size)       # 학습은 항상 drop_last

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    # ------------------------------------------------------------------ iter
    def _shuffled_pool(self, ci, g):
        """weight 를 pool 반복/서브샘플로 실현한다. 반복분은 **매번 다시 셔플**해서 한 epoch 안에
        같은 sample 이 같은 배치 이웃과 두 번 묶이지 않게 한다."""
        pool = self.pools[ci]
        need = self._n_batches[ci] * self.batch_size
        out = []
        while len(out) < need:
            perm = torch.randperm(len(pool), generator=g).tolist()
            out += [pool[i] for i in perm]
        return out[:need]

    def __iter__(self):
        if not self.shuffle:
            per = []
            for ci in range(len(self.pools)):
                pool = self.pools[ci]
                nb = self._n_batches[ci]
                per.append([pool[b * self.batch_size:(b + 1) * self.batch_size]
                            for b in range(nb)])
            if self.interleave:
                # round-robin: 앞 N 배치 안에 모든 코퍼스가 들어온다 (val_max_batches 대응)
                for r in range(max((len(p) for p in per), default=0)):
                    for p in per:
                        if r < len(p):
                            yield p[r]
            else:
                for p in per:
                    yield from p
            return

        g = torch.Generator().manual_seed(self.seed * 1000003 + self._epoch)
        batches = []
        for ci in range(len(self.pools)):
            flat = self._shuffled_pool(ci, g)
            batches += [flat[b * self.batch_size:(b + 1) * self.batch_size]
                        for b in range(self._n_batches[ci])]
        order = torch.randperm(len(batches), generator=g).tolist()
        self._epoch += 1                          # 학습 루프가 set_epoch 을 안 부른다 -> self-advance
        for i in order:
            yield batches[i]
