"""다중 코퍼스 혼합 데이터셋 — DL3DV + Scene-Decoupled + DataDoP 를 한 학습에 섞는다.

설계의 전부는 **cfg 복제**다. 코퍼스마다 `copy.deepcopy(cfg)` 로 평평한 SimpleNamespace 를 만들고
그 위에 화이트리스트 override 를 얹은 뒤 **기존 클래스를 그대로** 생성한다. 그래서
`dataset_dl3dv.py` / `dataset_scene_decoupled.py` / `dataset_datadop.py` / `dataset_cfg.py` 는
이 파일 때문에 한 줄도 바뀌지 않는다 — 각 sub-dataset 은 오늘과 완전히 같은 cfg 를 본다.

**배치 하나 = 코퍼스 하나** (`main/mixed_sampler.py`). V(context view 수)가 DL3DV≈6 / SD 6~49 /
DataDoP 1 로 달라 `collate_fn` 의 `torch.stack(images (V,3,H,W))` 가 섞인 배치에서 터진다.
공통 V 강제(DataDoP 한 장을 6번 복제)는 ① encoder 연산 낭비 ② 정보가 안 늘고 ③ DA3
`build_cam_token` 이 중복 view 에서 `dists`=0 -> `med=0.1` clamp -> 사실상 identity pose 가 된다.
모델에 BatchNorm 류가 없어(`models/camera_diffusion_model_latent.py` 확인) 배치를 코퍼스로
가르는 데 통계적 부작용이 없다.

`scale_gain` 은 **기본 1.0 (무동작) 이고 da3 arm 에서는 아예 raise 한다** — `_make_sub_cfg` 참고.
코퍼스별 translation 레벨을 상수로 맞추려던 손잡이인데, `geo_encoder='da3' + geo_posed=True`
에서는 context 가 `norm_scale` 을 안 거치기 때문에(raw `geo_c2w` -> DA3 median camera distance
재정규화) target 만 gain 배가 되고 context 는 불변이라 대응이 깨진다. 켤 수 있는 것은 모든
채널이 같은 `norm_scale` 로 나눠지는 `geo_custom`(lagernvs) 경로뿐이고, 그때도 sample 을 여기서
만지지 않고 `sub_cfg.norm_scale_gain` 으로 내려보낸다 (`CamDataset._target_out` 이 유일하게
안전한 주입점 — 반환된 `out['norm_scale']` 을 사후에 곱하면 cam_param 이 이미 옛 분모로 나눠진
뒤라 궤적과 geo depth 단위가 갈라진다).
=> 혼합 arm 의 코퍼스 간 레벨 차이(DL3DV 0.4367 vs SD 0.0878 vs DataDoP 0.0911)는 gain 이 아니라
필터/분모 정의로 다루고, 남는 차이는 그대로 둔 채 학습한다.

`data_name` 에는 코퍼스 태그를 **붙이지 않는다.** 세 이름 공간(`1K_<64hex>_<seg>` /
`<scene>__<tgt>__<ctx>` / `<scene>__<shot>`)이 충돌하지 않고, prefix 를 붙이면
`CamDataset.seg_key` 의 `rsplit('_',1)`, geo latent cache 경로, `scripts/mean_traj_baseline.py`
의 `split('__')` 가 전부 깨진다. 구분은 별도 `corpus` 키로 한다.
"""
import copy
import os.path as osp

import torch
from torch.utils.data import Dataset

_VALID_NAMES = ('dl3dv', 'scene_decoupled', 'datadop')

# 코퍼스별로 달라져도 되는 키. 여기 없는 키를 override 에 쓰면 raise 한다.
OVERRIDE_WHITELIST = {
    'dl3dv_root', 'sd_root', 'sd_split', 'sd_geo_views', 'sd_pair_mode',
    'meta_csv', 'max_scenes', 'train_seg_list', 'test_seg_list',
    'coverage_blacklist_path', 'train_frac',
    'pose_source', 'avg_scale_ref', 'scale_mode',
    'geo_view_sampling', 'geo_num_views', 'geo_cover_k', 'geo_cover_radius',
    'geo_cover_out_of_seg', 'geo_cover_before_only', 'geo_cover_centered_at_s',
    'geo_first_view_target_s',
}
# datadop_* 는 전부 허용 (접두사 규칙).
_OVERRIDE_PREFIXES = ('datadop_',)

# 모델 입력의 **shape** 를 정하는 키. geo encoder 인스턴스가 하나뿐이라 토큰 수 P 와 채널 수가
# 배치마다 달라지면 안 된다. 여기에 걸리면 화이트리스트 메시지보다 구체적인 이유를 준다.
OVERRIDE_FORBIDDEN = {
    'geo_encoder', 'geo_posed', 'geo_cam_embed', 'custom_geo_channels', 'geo_image_hw',
    'custom_geo_input_hw', 'da3_geo_input_hw', 'num_frames', 'intr_norm', 'trans_repr',
    'cam_dim', 'max_trans_norm',
}


def _spec_get(spec, key, default=None):
    return spec.get(key, default) if isinstance(spec, dict) else getattr(spec, key, default)


class MixedCamDataset(Dataset):
    """전역 index = 코퍼스별 sub-dataset 의 offset concat.

    노출하는 것:
      len(self)                전체 sample 수
      self.sub[i]              sub-dataset 인스턴스 (i = corpus id)
      self.names[i]            corpus 이름
      self.corpus_of[g]        전역 idx -> corpus id
      self.local_of[g]         전역 idx -> sub-dataset 안의 idx
      self.train_idx/val_idx   전역 idx 리스트 (코퍼스마다 자기 규칙으로 분할한 뒤 concat)
      self.weights[i]          PerCorpusBatchSampler 가 쓰는 코퍼스 가중치
    """

    def __init__(self, cfg, type='train', only_segments=None):
        specs = getattr(cfg, 'datasets', None)
        if not specs:
            raise ValueError("dataset_name='mixed' requires a non-empty `datasets:` list "
                             "(main/conf/config.yaml 의 같은 키 주석 참고)")
        if only_segments:
            raise ValueError("MixedCamDataset 는 only_segments 를 받지 않는다 "
                             "(코퍼스마다 이름 공간이 달라 어느 코퍼스의 세그먼트인지 모호하다). "
                             "단일 코퍼스 config 로 뽑을 것.")
        self.cfg = cfg
        self._check_top_level(cfg)

        self.sub, self.names, self.weights, self.gains = [], [], [], []
        self.corpus_of, self.local_of = [], []
        self.train_idx, self.val_idx = [], []

        for ci, spec in enumerate(specs):
            name = _spec_get(spec, 'name')
            if name not in _VALID_NAMES:
                raise ValueError(f"datasets[{ci}].name must be one of {_VALID_NAMES}, got {name!r}")
            sub_cfg = self._make_sub_cfg(cfg, spec, ci)
            print(f"[mixed] building corpus {ci} '{name}' "
                  f"(weight {_spec_get(spec, 'weight', 1.0)}, "
                  f"scale_gain {sub_cfg.norm_scale_gain})", flush=True)
            ds = self._build_sub(name, sub_cfg, type)

            off = len(self.corpus_of)
            n = len(ds)
            if n == 0:
                raise ValueError(f"[mixed] corpus '{name}' 의 index 가 비었다 (len 0). "
                                 f"override 의 seg list / root / 필터를 확인할 것.")
            self.corpus_of += [ci] * n
            self.local_of += list(range(n))
            tr, va = self._split_sub(ds, sub_cfg, name)
            self.train_idx += [off + i for i in tr]
            self.val_idx += [off + i for i in va]

            self.sub.append(ds)
            self.names.append(name)
            self.weights.append(float(_spec_get(spec, 'weight', 1.0) or 1.0))
            self.gains.append(sub_cfg.norm_scale_gain)
            print(f"[mixed] corpus {ci} '{name}': {n} samples -> train {len(tr)} / val {len(va)}",
                  flush=True)

        print(f"[mixed] total {len(self)} samples | train {len(self.train_idx)} / "
              f"val {len(self.val_idx)} | weights {self.weights}", flush=True)
        if getattr(cfg, 'mix_check_keys', True):
            self._check_key_sets()

    # ------------------------------------------------------------------ cfg 복제
    @staticmethod
    def _check_top_level(cfg):
        """mixed 에서 의미가 없거나 조용히 망가지는 top-level 키를 미리 잡는다."""
        bad = []
        for k in ('train_seg_list', 'test_seg_list'):
            if getattr(cfg, k, None):
                bad.append(f"{k}: mixed 에서는 코퍼스 블록의 override 로 내려야 한다 "
                           f"(top-level 은 어느 코퍼스의 리스트인지 모호하다)")
        for k in ('geo_swap_mode', 'geo_test_inseg_k'):
            if getattr(cfg, k, None):
                bad.append(f"{k}: 코퍼스마다 세그먼트 구조가 달라 의미가 정의되지 않는다")
        if bad:
            raise ValueError("[mixed] config 오류:\n  - " + "\n  - ".join(bad))
        if getattr(cfg, 'geo_latent_cache_dir', None):
            # 캐시 경로가 data_name.split('_')[0] 이라 코퍼스 간에 충돌한다.
            print("[mixed] WARNING: geo_latent_cache_dir 를 무시한다 (경로 규칙이 코퍼스 간 충돌).")
            cfg.geo_latent_cache_dir = None

    def _make_sub_cfg(self, cfg, spec, ci):
        name = _spec_get(spec, 'name')
        ov = _spec_get(spec, 'override', None) or {}
        if not isinstance(ov, dict):
            raise ValueError(f"datasets[{ci}].override must be a mapping, got {type(ov)}")
        for k in ov:
            if k in OVERRIDE_FORBIDDEN:
                raise ValueError(
                    f"datasets[{ci}]('{name}').override.{k} 는 금지다 — 모델 입력 shape 를 정하는 "
                    f"키라 코퍼스마다 다르면 geo encoder 토큰 수/채널이 배치마다 달라진다. "
                    f"top-level 에 하나로 둘 것.")
            if k not in OVERRIDE_WHITELIST and not k.startswith(_OVERRIDE_PREFIXES):
                raise ValueError(
                    f"datasets[{ci}]('{name}').override.{k} 는 화이트리스트에 없다. "
                    f"허용: {sorted(OVERRIDE_WHITELIST)} + 'datadop_*'")
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.datasets = None                 # 재귀 방지
        sub_cfg.dataset_name = name
        for k, v in ov.items():
            setattr(sub_cfg, k, v)
        # scale_gain -> _target_out 이 읽는 키. 이름이 다른 이유는 그 훅이 단일 코퍼스 학습에서도
        # 쓸 수 있는 일반 손잡이이기 때문 (mixed 전용 키로 두면 dataset_dl3dv 가 mixed 를 알아야 한다).
        gain = float(_spec_get(spec, 'scale_gain', 1.0) or 1.0)
        if gain != 1.0 and str(getattr(cfg, 'geo_encoder', '') or '') == 'da3' \
                and bool(getattr(cfg, 'geo_posed', False)):
            raise ValueError(
                f"datasets[{ci}]('{name}').scale_gain={gain} 은 geo_encoder='da3' + geo_posed=True "
                f"에서 **좌표계를 깨뜨린다**. 이 경로의 context 는 norm_scale 을 안 거친다: "
                f"__getitem__ 이 raw geo_c2w 를 월드 단위로 내보내고(dataset_dl3dv.py:1437-1444) "
                f"build_cam_token 이 DA3 자기 규약(median camera distance)으로 다시 정규화한다"
                f"(da3_geo_encoder.py:279-282). 즉 target cam_param 만 gain 배가 되고 context "
                f"cam token 은 불변이라, context 가 함의하는 스케일과 target 크기의 대응이 "
                f"코퍼스별 상수만큼 어긋난다 — 모델은 그 상수를 context 에서 읽을 수 없고 "
                f"추론엔 코퍼스 라벨이 없다. gain 이 균일 닮음변환으로 남는 것은 모든 채널이 "
                f"같은 norm_scale 로 나눠지는 geo_custom(lagernvs) 경로뿐이다. "
                f"=> scale_gain 은 1.0 으로 두고 레벨 차이는 필터/분모 정의로 다룰 것.")
        sub_cfg.norm_scale_gain = gain
        return sub_cfg

    @staticmethod
    def _build_sub(name, sub_cfg, type):
        if name == 'dl3dv':
            from dataset_dl3dv import CamDataset
            return CamDataset(cfg=sub_cfg, type=type)
        if name == 'scene_decoupled':
            from dataset_scene_decoupled import SDCamDataset
            return SDCamDataset(cfg=sub_cfg, type=type)
        from dataset_datadop import DataDoPCamDataset
        return DataDoPCamDataset(cfg=sub_cfg, type=type)

    # ------------------------------------------------------------------ split
    @staticmethod
    def _split_sub(ds, sub_cfg, name):
        """코퍼스마다 **자기 규칙**으로 train/val 을 가른다 (base.py 의 두 경로와 같은 규칙).

        seg list 가 있으면 리스트 순서대로, 없으면 같은 random_seed 의 random_split.
        반환은 sub-dataset 로컬 idx 리스트.
        """
        tr_list = getattr(sub_cfg, 'train_seg_list', None)
        te_list = getattr(sub_cfg, 'test_seg_list', None)
        if tr_list and te_list:
            seg_key = type(ds).seg_key
            id2idx = {}
            for i, s in enumerate(ds.samples):
                id2idx.setdefault(seg_key(s[4]), i)

            def _load(p):
                with open(p) as f:
                    return [ln.strip() for ln in f if ln.strip()]
            tr_ids, te_ids = _load(tr_list), _load(te_list)
            tr = [id2idx[x] for x in tr_ids if x in id2idx]
            te = [id2idx[x] for x in te_ids if x in id2idx]
            print(f"[mixed seg-list split] {name}: train {len(tr)}/{len(tr_ids)} , "
                  f"val {len(te)}/{len(te_ids)}")
            return tr, te
        frac = float(getattr(sub_cfg, 'train_frac', 0.9))
        n = len(ds)
        n_tr = n if frac >= 1.0 else int(frac * n)
        g = torch.Generator().manual_seed(int(getattr(sub_cfg, 'random_seed', 42)))
        perm = torch.randperm(n, generator=g).tolist()
        return perm[:n_tr], perm[n_tr:]

    # ------------------------------------------------------------------ key set
    def _check_key_sets(self):
        """코퍼스 간 **key set 동일** 확인. `collate_fn` (base.py:148) 이 `batch[0]` 의 키만
        순회하므로 코퍼스마다 키가 다르면 배치의 첫 sample 이 무엇이냐에 따라 텐서가 조용히
        사라진다 — 학습이 죽지 않고 성능만 떨어지는 종류라 init 에서 한 번 값을 치른다."""
        ref_keys, ref_name = None, None
        for name, ds in zip(self.names, self.sub):
            keys = set(ds[0].keys())
            if ref_keys is None:
                ref_keys, ref_name = keys, name
                continue
            if keys != ref_keys:
                raise ValueError(
                    f"[mixed] key set 불일치: '{ref_name}' 에만 {sorted(ref_keys - keys)}, "
                    f"'{name}' 에만 {sorted(keys - ref_keys)}. collate_fn 이 batch[0] 의 키만 "
                    f"순회해서 조용히 유실된다 — 두 코퍼스의 cfg 를 맞출 것.")
        print(f"[mixed] key set OK ({len(ref_keys)} keys, 코퍼스 {len(self.sub)}개 동일)")

    # ------------------------------------------------------------------ item
    def __len__(self):
        return len(self.corpus_of)

    def __getitem__(self, idx):
        ci = self.corpus_of[idx]
        out = self.sub[ci][self.local_of[idx]]
        out['corpus'] = self.names[ci]      # collate_fn 이 str 을 list(batch) 로 받는다
        return out

    # ------------------------------------------------------------------ sampler 보조
    def corpus_of_positions(self, idx_list):
        """Subset(dataset, idx_list) 안의 **위치** -> corpus id. 배치 샘플러가 Subset 을 인덱싱하므로
        전역 idx 가 아니라 위치를 줘야 한다."""
        return [self.corpus_of[g] for g in idx_list]
