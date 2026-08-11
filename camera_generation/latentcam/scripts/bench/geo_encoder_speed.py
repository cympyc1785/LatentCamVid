"""geo encoder 추론 속도/구조 비교 — `geo_encoder: custom` vs `geo_encoder: lagernvs`.

두 backend 는 같은 자리를 채운다: context view 들을 받아 diffusion 쪽 cross-attention 이
쓸 geo 토큰 `(B, M, 768)` 을 만든다 (`models/geo_encoder.py::GeoEncoder`). 그런데
  - lagernvs : frozen VGGT reconstructor, 입력이 `images` + `cam_token(11)` 하나뿐
  - custom   : frozen DINOv2-L/14 + **trainable** GeoTokenizer, 입력이 픽셀 Plücker/depth 까지
라 파라미터 수도 토큰 수도 forward 비용도 다르다. 여기서 재는 건 **추론(no_grad) forward**
뿐이다 — custom 은 학습 시 backward 도 타므로 (`trainable=True`) 학습 step 비용 비교는
이 숫자로 외삽하면 안 된다.

측정 조건은 학습/추론 경로와 맞춘다: `mixed_precision='bf16'`
(`train_latent_cam_dm.py:349`) 이므로 bf16 autocast 안에서 잰다. 해상도도 각 backend 가
실제로 받는 값을 쓴다 — custom 은 `custom_geo_input_hw` (252,448; patch 14 배수),
lagernvs 는 `geo_image_hw` (256,448).

view 수 V 는 arm 마다 다르다:
  DL3DV customgeo : `geo_view_sampling: frustum_cover` + `geo_cover_k` 기본 6  -> V=6
  SD  customgeo   : `sd_geo_views: null` = context clip 49 프레임 전부        -> V=49

주의: GPU 가 학습으로 점유된 상태에서 재면 median 이 부풀고 분산이 커진다. 그래서
median 과 함께 **min** 을 같이 찍는다 (min 이 경합에 가장 덜 오염된 값). 정확한 수치가
필요하면 유휴 GPU 에서 다시 돌릴 것.

usage
-----
  python scripts/bench/geo_encoder_speed.py --device cuda:2 --views 6 49 --iters 20
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def make_cfg():
    """main/conf/config.yaml 의 관련 키만 추린 것 (기본값 그대로)."""
    return SimpleNamespace(
        geo_latent_dim=768,
        ckpt_root=os.path.join(ROOT, 'checkpoints'),
        custom_geo_dino_path=None,          # -> <ckpt_root>/dinov2-large
        custom_geo_freeze_dino=True,
        custom_geo_input_hw=[252, 448],
        custom_geo_ray_dim=64,
        custom_geo_geo_dim=256,
        custom_geo_channels='full',
        lagernvs_repo_path='/data1/cympyc1785/LatentCamVid/camera_generation/tools/lagernvs',
        lagernvs_ckpt_path='/data1/cympyc1785/LatentCamVid/camera_generation/tools/lagernvs/'
                           'checkpoints/lagernvs_general_512/model.pt',
    )


def params(mod):
    """-> (trainable, frozen) 파라미터 수."""
    t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    f = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
    return t, f


def timeit(fn, device, iters, warmup):
    """-> (median_ms, min_ms, max_ms). CUDA sync 포함."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def bench(name, enc, make_inputs, call, device, views, batch, iters, warmup):
    tr, fr = params(enc)
    print(f"\n### {name}")
    print(f"  params  : trainable {tr / 1e6:8.2f} M   frozen {fr / 1e6:8.2f} M   "
          f"total {(tr + fr) / 1e6:8.2f} M")
    for v in views:
        try:
            inp = make_inputs(batch, v)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out, mask = call(enc, inp)
                med, mn, mx = timeit(lambda: call(enc, inp), device, iters, warmup)
            peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
            m = out.shape[1]
            print(f"  V={v:<3d} tokens {m:6d} ({m // v:5d}/view)  "
                  f"med {med:8.2f} ms  min {mn:8.2f}  max {mx:8.2f}  "
                  f"{med / v:6.2f} ms/view  peak {peak:5.2f} GiB  out {tuple(out.shape)}")
        except torch.cuda.OutOfMemoryError:
            print(f"  V={v:<3d} OOM")
            torch.cuda.empty_cache()
        except Exception as e:                                    # backend 별 제약을 그대로 노출
            print(f"  V={v:<3d} FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


def trace_custom(enc, device, v=2, b=1):
    """custom backend 의 단계별 shape 을 forward hook 으로 실측해 찍는다."""
    net = enc.backend.net
    h, w = net.input_hw
    logs = []
    hooks = [
        (net.dino, 'DINOv2-L/14'),
        (net.geo.ray if net.geo else None, 'GeoTokenizer.ray  Conv2d(6,64,k14,s14)'),
        (net.geo.stem if net.geo else None, 'GeoTokenizer.stem Conv2d x4'),
        (net.geo if net.geo else None, 'GeoTokenizer  cat[ray, stem]'),
        (net.proj, 'proj Linear'),
    ]
    handles = []
    for mod, tag in hooks:
        if mod is None:
            continue

        def mk(tag):
            def hook(m, inp, out, kw=None):
                o = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out
                # DINOv2 는 pixel_values= 로만 불려서 positional inp 가 빈 튜플이다
                i = inp[0] if inp else (kw or {}).get('pixel_values')
                logs.append((tag, tuple(i.shape) if i is not None else '(kwargs)',
                             tuple(o.shape)))
            return hook
        handles.append(mod.register_forward_hook(
            (lambda h: lambda m, i, kw, o: h(m, i, o, kw))(mk(tag)), with_kwargs=True))
    if net.geo is not None:
        for sub, tag in [(net.geo.stem[0], '  stem[0] Conv2d(8,64,3,1,1)'),
                         (net.geo.stem[3], '  stem[3] Conv2d(64,96,3,2,1)  /2'),
                         (net.geo.stem[6], '  stem[6] Conv2d(96,128,3,1,1)'),
                         (net.geo.stem[9], '  stem[9] Conv2d(128,c_s,k7,s7) /7')]:
            handles.append(sub.register_forward_hook(
                (lambda t: lambda m, i, o: logs.append((t, tuple(i[0].shape), tuple(o.shape))))(tag)))

    inp = {'images': torch.rand(b, v, 3, h, w, device=device),
           'geo_plucker_map': torch.randn(b, v, 6, h, w, device=device),
           'geo_logd': torch.randn(b, v, 1, h, w, device=device),
           'geo_valid': torch.ones(b, v, 1, h, w, device=device)}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out, mask = enc.forward_batch(inp)
    for hd in handles:
        hd.remove()

    print(f"\n### custom 단계별 shape (B={b}, V={v}, input_hw {h}x{w}, "
          f"patch {net.patch}, grid {net.patch_grid()})")
    print(f"  {'stage':44s} {'in':>28s} -> {'out':<28s}")
    for tag, i, o in logs:
        print(f"  {tag:44s} {str(i):>28s} -> {str(o):<28s}")
    print(f"  {'GeoEncoder.proj (' + type(enc.proj).__name__ + ')':44s} "
          f"{'':>28s} -> {str(tuple(out.shape)):<28s}")
    print("\n  파라미터 내역 (trainable):")
    for n, m in [('geo.ray', net.geo.ray if net.geo else None),
                 ('geo.stem', net.geo.stem if net.geo else None),
                 ('ln_d', net.ln_d), ('ln_g', net.ln_g), ('proj', net.proj)]:
        if m is None:
            continue
        print(f"    {n:10s} {sum(p.numel() for p in m.parameters()):>12,d}")
    print(f"    {'dino (frozen)':10s} {sum(p.numel() for p in net.dino.parameters()):>12,d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--trace', action='store_true', help='custom backend 단계별 shape 도 찍는다')
    ap.add_argument('--views', type=int, nargs='+', default=[6, 49])
    ap.add_argument('--batch', type=int, default=1, help='추론이라 기본 1')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--backends', nargs='+', default=['custom', 'lagernvs'])
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = make_cfg()
    print(f"device {device}  torch {torch.__version__}  "
          f"{torch.cuda.get_device_name(device)}  bf16 autocast  batch {args.batch}")

    from models.geo_encoder import build_geo_encoder

    if 'custom' in args.backends:
        cfg.geo_encoder = 'custom'
        enc = build_geo_encoder(cfg).to(device).eval()
        h, w = cfg.custom_geo_input_hw

        def mk(b, v):
            return {'images': torch.rand(b, v, 3, h, w, device=device),
                    'geo_plucker_map': torch.randn(b, v, 6, h, w, device=device),
                    'geo_logd': torch.randn(b, v, 1, h, w, device=device),
                    'geo_valid': torch.ones(b, v, 1, h, w, device=device)}

        if args.trace:
            trace_custom(enc, device)
        bench(f'custom  (DINOv2-L/14 + GeoTokenizer, {h}x{w})', enc, mk,
              lambda e, i: e.forward_batch(i), device, args.views,
              args.batch, args.iters, args.warmup)
        del enc
        torch.cuda.empty_cache()

    if 'lagernvs' in args.backends:
        cfg.geo_encoder = 'lagernvs'
        enc = build_geo_encoder(cfg).to(device).eval()
        h, w = 256, 448                                            # geo_image_hw

        def mk(b, v):
            return (torch.rand(b, v, 3, h, w, device=device),
                    torch.zeros(b, v, 11, device=device))          # unposed (geo_posed: false)

        bench(f'lagernvs (frozen VGGT reconstructor, {h}x{w})', enc, mk,
              lambda e, i: e(*i), device, args.views,
              args.batch, args.iters, args.warmup)


if __name__ == '__main__':
    main()
