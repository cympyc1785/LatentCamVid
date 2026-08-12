"""DL3DV da3 avg_scale — **DA3 latent 이 실제로 쓰는 카메라 스케일** 을 분모로 저장한다.

왜 필요한가
-----------
`geo_posed=True` 로 `cam_enc` 에 GT pose 를 넣어 줘도 DA3 가 latent 에서 뽑는 카메라의
**translation 은 항상 예측값**이다: `cam_dec.py:35` 의 `out_t = self.fc_t(feat)` 에는 echo
경로가 아예 없고 (rotation/fov 만 `camera_encoding` 이 주어지면 echo), `da3.py:216` 은
`camera_encoding` 없이 부른다. 200 세그먼트 실측에서 예측/주입 스케일 비 sigma 는
med 2.90040 (p05 1.25082 / p95 11.83133), ±5% 안에 든 것이 1.0% 뿐이었다
(scripts/data/da3_latent_scale_probe.py). 즉 geo latent 이 표현하는 공간과 target cam_param 이
사는 공간의 스케일이 샘플마다 다르게 어긋나 있다.

여기서 만드는 분모는 그 어긋남을 없앤다:

  c_fed  = (GT camera center, view0 기준) / M      M = per-sample median camera distance
                                                   (= build_cam_token 이 쓰는 그 값)
  c_pred = cam_dec(feats[-1][1]) -> c2w center     (latent 이 실제로 보는 카메라)
  sigma  = Umeyama(source=c_pred, target=c_fed).scale
  => GT 상대좌표 = M * c_fed = (M*sigma) * c_pred
  => **latent 의 1 단위 = M*sigma GT 미터.** 저장값 = M*sigma.

target translation 을 이 값으로 나누면 target 궤적이 geo latent 과 같은 스케일 공간에 놓인다.
Umeyama 의 R/t 는 쓰지 않는다 (앵커는 front_first_anchor 계열과 같이 target 첫 카메라 s 로
둔다). 실측 residual 이 fed 단위 med 0.00507 로 매우 작아 차이가 사실상 순수 스칼라였다.

순환이 없는 이유
----------------
`cam_token` 은 DA3 자기 규약(view0 재고정 + median camera distance)으로만 만들어지고
`avg_scale`/`norm_scale` 을 안 쓴다. 그래서 이 스크립트가 dataset 을 어떤 avg_scale_ref 로
열든 sigma/M 은 바뀌지 않는다.

학습 때와 동일함을 보장하는 방법
--------------------------------
view 선택 / resize / cam_token 생성을 **직접 재구현하지 않고** 학습이 쓰는 `CamDataset` +
`DA3SceneEncoder` 를 그대로 돌린다. 그래서 반드시 짝 arm 의 config 로 실행해야 한다
(`experiment=da3_7k_da3geo_frontanchor_samelen` -> geo_view_sampling=front_uniform,
geo_num_views=6, da3_geo_input_hw=[252,448]). view 집합이 결정적이라 값도 결정적이다.

usage (main/ 에서):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<latentcam>:. \
    python ../scripts/data/make_avg_scale_da3_latent.py \
      experiment=da3_7k_da3geo_frontanchor_samelen --batch 8 --workers 8
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch

from hydra_cfg import load_cfg

OUT_DIR = 'avg_scale_da3latent'


def umeyama_scale(src, dst):
    """Umeyama similarity 의 scale + residual. src/dst (N,3)."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(src)
    if var_s < 1e-20:
        return np.nan, np.nan
    U, D, Vt = np.linalg.svd((d0.T @ s0) / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s)
    resid = float(np.sqrt(((d0 - scale * (R @ s0.T).T) ** 2).sum(1)).mean())
    return scale, resid


def collate(batch):
    """필요한 키만 stack 한다 (caption/text 는 안 쓴다)."""
    out = {k: torch.stack([b[k] for b in batch]) for k in
           ('images', 'geo_c2w', 'geo_fxfycxcy', 'geo_hw')}
    return out


class _Wrap(torch.utils.data.Dataset):
    """CamDataset 을 감싸 (필요한 텐서 + 세그먼트 식별자)만 돌려준다."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        scene_idx, _s, _e, _cap, data_name = self.ds.samples[i]
        return {k: d[k] for k in ('images', 'geo_c2w', 'geo_fxfycxcy', 'geo_hw')}, \
            int(scene_idx), str(data_name)


def collate2(batch):
    ts = collate([b[0] for b in batch])
    return ts, [b[1] for b in batch], [b[2] for b in batch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train'],
                    help="CamDataset type. 기본 'train' 하나면 충분하다 — base.build_dataset 은 "
                         "CamDataset(cfg) 하나만 만들고 train/val 은 그 위의 Subset 분할이라 "
                         "(base.py:69-89) 'train' 인덱스가 이미 전 코퍼스다. 'test' 를 더 주면 "
                         "같은 세그먼트를 두 번 계산할 뿐이다.")
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='파일을 쓰지 않고 통계만')
    args, _ = ap.parse_known_args()

    cfg, _ = load_cfg('config')
    if str(getattr(cfg, 'geo_encoder', None)) != 'da3':
        raise SystemExit(f"geo_encoder must be 'da3' (got {cfg.geo_encoder!r})")
    if not bool(getattr(cfg, 'geo_posed', False)):
        raise SystemExit("geo_posed=True 여야 한다 (cam_token 이 있어야 sigma 를 잰다)")

    from dataset_dl3dv import CamDataset
    from models.da3_geo_encoder import DA3SceneEncoder

    device = 'cuda'
    torch.set_grad_enabled(False)
    enc = DA3SceneEncoder(
        repo_path=cfg.da3_geo_repo_path, model_name=cfg.da3_geo_model,
        ckpt_path=getattr(cfg, 'da3_geo_ckpt_path', None),
        hf_home=getattr(cfg, 'da3_geo_hf_home', None),
        input_hw=getattr(cfg, 'da3_geo_input_hw', None),
        layers=getattr(cfg, 'da3_geo_layers', 'last'),
        keep_cam_dec=True,
    ).to(device).eval()
    # DA3SceneEncoder 생성이 sys.path 에 <repo>/src 를 넣으므로 그 뒤에 import 해야 한다.
    from depth_anything_3.utils.geometry import affine_inverse
    print(f"[cfg] view_sampling={getattr(cfg, 'geo_view_sampling', None)} "
          f"V={getattr(cfg, 'geo_num_views', None)} input_hw={cfg.da3_geo_input_hw} "
          f"avg_scale_ref={cfg.avg_scale_ref} -> writing '{OUT_DIR}'", flush=True)

    vals, sigmas, resids, n_written, n_bad = [], [], [], 0, 0
    for split in args.splits:
        ds = CamDataset(cfg, split)
        wrapped = _Wrap(ds)
        n = len(wrapped) if not args.limit else min(args.limit, len(wrapped))
        if n < len(wrapped):
            wrapped = torch.utils.data.Subset(wrapped, range(n))
        dl = torch.utils.data.DataLoader(
            wrapped, batch_size=args.batch, shuffle=False, num_workers=args.workers,
            collate_fn=collate2, pin_memory=False)
        print(f"[{split}] {n} segments", flush=True)

        done = 0
        for ts, scene_idxs, data_names in dl:
            images = ts['images'].to(device)
            c2w = ts['geo_c2w'].to(device).float()
            fx = ts['geo_fxfycxcy'].to(device).float()
            hw = ts['geo_hw'].to(device)

            # build_cam_token 과 **같은 식**으로 fed center 를 재현한다.
            w2c = affine_inverse(c2w) @ c2w[:, :1]
            c2w_n = affine_inverse(w2c)
            M = c2w_n[..., :3, 3].norm(dim=-1).median(dim=1).values.clamp(min=1e-1)   # (B,)
            c_fed = (c2w_n[..., :3, 3] / M.view(-1, 1, 1)).cpu().numpy()

            tok = enc.build_cam_token(c2w, fx, hw)
            c2w_p, _ = enc.predict_cameras(images, cam_token=tok)
            c_pred = c2w_p[..., :3, 3].cpu().numpy()

            for b in range(len(data_names)):
                sig, res = umeyama_scale(c_pred[b], c_fed[b])
                v = float(M[b]) * sig
                if not np.isfinite(v) or v <= 0:
                    n_bad += 1
                    continue
                sigmas.append(sig); resids.append(res); vals.append(v)
                if not args.dry_run:
                    d = osp.join(ds.scene_dir_list[scene_idxs[b]], 'da3', OUT_DIR)
                    os.makedirs(d, exist_ok=True)
                    with open(osp.join(d, f"{data_names[b].split('_')[-1]}.json"), 'w') as f:
                        json.dump(v, f)
                    n_written += 1
            done += len(data_names)
            if done % (args.batch * 50) < args.batch:
                print(f"  [{split}] {done}/{n}", flush=True)

    def st(name, a):
        a = np.asarray([x for x in a if np.isfinite(x)], np.float64)
        if not len(a):
            print(f"  {name:14s} (empty)"); return
        q = np.percentile(a, [5, 50, 95])
        print(f"  {name:14s} n={len(a)} mean {a.mean():.5f} med {q[1]:.5f} "
              f"p05 {q[0]:.5f} p95 {q[2]:.5f} min {a.min():.5f} max {a.max():.5f}")

    print(f"\nDONE: written {n_written} | bad(비유한/<=0) {n_bad} | dry_run={args.dry_run}")
    st('sigma', sigmas); st('umeyama resid', resids); st('M*sigma (저장값)', vals)


if __name__ == '__main__':
    main()
