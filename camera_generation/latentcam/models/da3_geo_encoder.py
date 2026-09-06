"""DA3 geo encoder — Depth-Anything-3 의 cross-view ViT 백본을 geo context encoder 로 쓴다.
`cfg.geo_encoder = 'da3'` 일 때만 쓰이며, lagernvs / custom 경로는 한 줄도 건드리지 않는다.

왜 DA3 인가
-----------
기존 custom backend (models/custom_geo_encoder.py) 의 외관 갈래는 **DINOv2 = single-view**
백본이라 view 사이의 대응 관계를 스스로 못 만든다. 3D 는 dataset 이 계산해 준 픽셀 Plücker /
log-depth 채널로만 들어간다. DA3 백본은 block `alt_start` 부터 local / global attention 을
번갈아 돌리는 **cross-view ViT** 라 geometry 를 feature 안에서 직접 푼다. 게다가 이 프로젝트의
corpus pose/depth (`<scene>/da3/{pose,depth}.npz`) 가 `DA3NESTED-GIANT-LARGE-1.1` 로 만들어진
것이라 표현 공간이 일관된다.

vendored 저장소는 읽기 전용
---------------------------
`camera_generation/tools/Depth-Anything-3` 는 **한 줄도 고치지 않는다.** 그래서 두 가지를 우회한다:

1. `depth_anything_3.api` 를 import 하지 않는다.
   - `api.py` -> `utils/export/gs.py` 가 `moviepy.editor` 를 import 하는데 latentcam env 에
     moviepy 가 없다 (moviepy 는 gs_video export 전용). 아래 `_stub_moviepy()` 로 막는다.
   - `DepthAnything3.forward` 에 `@torch.inference_mode()` 가 걸려 있어서 그대로 쓰면 downstream
     autograd 가 오염된다.
   대신 `cfg.load_config` + `registry.MODEL_REGISTRY` + `safetensors` 로 `DepthAnything3Net` 을
   직접 만들고, `net.backbone(...)` / `net.cam_enc(...)` 만 우리 `torch.no_grad()` 안에서 부른다.
   이 우회 경로가 api 와 같은 가중치/같은 출력을 준다는 것은
   `scripts/data/da3_res_probe.py --check-path` 로 확인했다 (da3-large 는 max|Δ|=0.0,
   nested-giant 는 metric branch 의 `torch.randint` 샘플링 때문에 모델 자체가 비결정적이라
   그 노이즈 범위 안. 어차피 여기서는 head/metric 을 삭제한다).

2. `export_feat_layers` 는 **반드시 `[]`** 로 넘긴다. `None` 을 주면 backbone 안에서
   `if i in export_feat_layers` 가 `TypeError` 로 터진다 (`vision_transformer.py:347`).

출력 계약
---------
`forward(images, cam_token) -> (B, V*P, native_dim)`. `native_dim` 은 `cat_token=True` 라
`2 * embed_dim` (vitg -> 3072, vitl -> 2048). 768 로 내리는 Linear 는 바깥
`models/geo_encoder.py:GeoEncoder.proj` 가 맡는다. 여기서는 그 앞에 LayerNorm 만 건다 —
`cat_token` 의 두 반쪽은 앞이 un-normed `local_x`, 뒤가 `self.norm` 을 통과한 것이라 분산
스케일이 크게 다르고, LN 없이 Linear 만 태우면 한쪽이 다른 쪽을 먹는다.
"""
import os
import os.path as osp
import sys
import types

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)   # DA3 InputProcessor.NORMALIZE 와 동일
_IMAGENET_STD = (0.229, 0.224, 0.225)

_PATCH = 14                              # DepthAnything3Net.PATCH_SIZE


# --------------------------------------------------------------------------- #
# import shim (DA3 저장소 수정 0줄)
# --------------------------------------------------------------------------- #
def _stub_moviepy():
    """moviepy 는 DA3 의 gs_video export 에만 필요하다. 없으면 import 자체가 막히므로 stub.

    `mv.editor = ed` 를 빼먹으면 안 된다 — `import moviepy.editor as mpy` 는 sys.modules 가
    아니라 부모 모듈의 attribute 를 본다.
    """
    if "moviepy" in sys.modules:
        return
    try:
        import moviepy  # noqa: F401
        return
    except Exception:
        pass

    def _missing(*a, **k):
        raise ImportError("moviepy is required only for DA3 gs_video export")

    mv = types.ModuleType("moviepy")
    mv.__file__ = "<optional dependency stub: moviepy>"
    mv.__path__ = []
    ed = types.ModuleType("moviepy.editor")
    ed.__file__ = "<optional dependency stub: moviepy.editor>"
    ed.__getattr__ = lambda name: _missing
    mv.editor = ed
    sys.modules["moviepy"] = mv
    sys.modules["moviepy.editor"] = ed


def _init_da3(repo_path, hf_home=None):
    if not repo_path or not osp.isdir(repo_path):
        raise FileNotFoundError(f"da3_geo_repo_path not found: {repo_path}")
    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")   # forward 마다 INFO 한 줄씩 뱉는다
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    src = osp.join(osp.realpath(repo_path), "src")
    if not osp.isdir(src):
        raise FileNotFoundError(f"da3_geo_repo_path has no src/: {src}")
    if src not in sys.path:
        sys.path.insert(0, src)
    _stub_moviepy()


def _hf_snapshot(model_name, hf_home):
    """로컬 HF 캐시의 snapshot 디렉토리 (네트워크 없이). 규칙: da3nested-giant-large ->
    depth-anything/DA3NESTED-GIANT-LARGE-1.1. 못 찾으면 repo id 를 그대로 돌려준다."""
    repo_id = "depth-anything/" + model_name.upper() + "-1.1"
    if not hf_home:
        return repo_id
    hub = osp.join(hf_home, "hub")
    root = osp.join(hub if osp.isdir(hub) else hf_home,
                    "models--" + repo_id.replace("/", "--"))
    cands = []
    refs = osp.join(root, "refs", "main")
    if osp.isfile(refs):
        with open(refs) as f:
            cands.append(osp.join(root, "snapshots", f.read().strip()))
    snaps = osp.join(root, "snapshots")
    if osp.isdir(snaps):
        cands += [osp.join(snaps, d) for d in sorted(os.listdir(snaps), reverse=True)]
    for c in cands:
        if osp.isfile(osp.join(c, "model.safetensors")):
            return c
    return repo_id


# --------------------------------------------------------------------------- #
# per-view resampler (D144)
# --------------------------------------------------------------------------- #
class PerViewResampler(nn.Module):
    """(B,V,P,C_native) -> (B,V,R,D). view **안에서만** P 개 patch 를 R 개 latent 로 줄인다.

    왜 필요한가
    -----------
    molmo2 arm 은 49프레임 × 64토큰 = 3136 토큰이고 da3 arm 은 6뷰 × 576패치 = 3456 토큰이다
    (`main/cache_molmo2_embeddings.py:261` 이 그 예산 일치를 명시한다). 두 arm 이 같은 토큰
    예산을 쓰지만 **줄이는 축이 다르다** — molmo2 는 프레임을 다 보고 프레임 안에서 평균풀링,
    da3 는 프레임을 6장으로 솎아 낸다. 즉 "49프레임 전체가 주는 추가 정보"가 da3 쪽에만 없다.
    이 모듈은 da3 를 molmo2 와 **같은 축**으로 옮긴다: 49뷰를 전부 넣고 뷰 안에서 576 -> 64 로
    줄여 49*64 = 3136 토큰, molmo2 의 video 토큰 수와 정확히 같게 만든다.

    molmo2 의 `adaptive_avg_pool2d` (9x9 -> 8x8, `cache_molmo2_embeddings.py:151-159`) 를
    **학습되는** Perceiver cross-attention 으로 바꾼 것이다. 고정 풀링이 아니라 latent query 가
    뽑아 가므로 어느 패치를 남길지를 학습이 정한다.

    **프레임을 섞지 않는다** — molmo2 의 pool 과 같은 제약이다 (그 파일 :17 "프레임을 섞지
    않는다"). B*V 로 접어 cross-attention 을 돌리므로 view 간 정보는 이 단계에서 안 흐른다.
    view 사이의 대응은 이미 DA3 백본의 cross-view attention 이 풀어 놓았고, 여기서 또 섞으면
    "프레임당 토큰 R개" 라는 예산 해석이 깨진다.

    `kv_in` (C_native -> D) 이 먼저 오는 이유: 줄이기 전 토큰이 B*49*576 = 28224 개라 그
    폭(3072)에서 attention 을 돌리면 메모리가 감당이 안 된다. 전 토큰을 한 번 만지는 최소
    비용이 Linear 하나이고, 그 뒤는 전부 D(=768) 에서 돈다. 이러면 바깥
    `GeoEncoder.proj` (3072->768) 는 `nn.Identity` 가 되므로 (native == out_dim) 파라미터가
    새로 생기는 게 아니라 **자리를 옮긴 것**에 가깝다.
    """

    def __init__(self, in_dim, dim=768, n_latents=64, heads=8, layers=1, mlp_ratio=4.0):
        super().__init__()
        self.n_latents = int(n_latents)
        self.dim = int(dim)
        self.kv_in = nn.Linear(int(in_dim), self.dim)
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.dim) * 0.02)
        self.ln_kv = nn.LayerNorm(self.dim)
        self.blocks = nn.ModuleList()
        for _ in range(int(layers)):
            self.blocks.append(nn.ModuleDict(dict(
                ln_q=nn.LayerNorm(self.dim),
                attn=nn.MultiheadAttention(self.dim, int(heads), batch_first=True),
                ln_m=nn.LayerNorm(self.dim),
                mlp=nn.Sequential(nn.Linear(self.dim, int(self.dim * mlp_ratio)), nn.GELU(),
                                  nn.Linear(int(self.dim * mlp_ratio), self.dim)),
            )))
        self.ln_out = nn.LayerNorm(self.dim)

    def forward(self, t):
        B, V, P, _ = t.shape
        kv = self.ln_kv(self.kv_in(t.reshape(B * V, P, -1)))
        q = self.latents.unsqueeze(0).expand(B * V, -1, -1).to(kv.dtype)
        for blk in self.blocks:
            qn = blk['ln_q'](q)
            q = q + blk['attn'](qn, kv, kv, need_weights=False)[0]
            q = q + blk['mlp'](blk['ln_m'](q))
        return self.ln_out(q).view(B, V, self.n_latents, self.dim)


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class DA3SceneEncoder(nn.Module):
    """frozen DA3 backbone (+ frozen cam_enc) -> geo 토큰 (B, V*P, native_dim).

    학습되는 파라미터는 `self.ln` 하나뿐이고, 768 로 내리는 Linear 는 GeoEncoder.proj 다.
    DA3 본체는 전부 `requires_grad=False` + 영구 eval 이라 ckpt 에서도 빠진다
    (`train_latent_cam_dm.py` 의 `_geo_frozen_keys`).
    """

    LAYER_MODES = ('last', 'all')

    def __init__(self, repo_path, model_name='da3nested-giant-large', ckpt_path=None,
                 hf_home=None, input_hw=None, layers='last', layer_fuse='concat',
                 norm='ln', ref_view_strategy='saddle_balanced', debug=False,
                 keep_cam_dec=False, cam_token_per_sample=False,
                 resampler=None, resampler_dim=768, resampler_tokens=64,
                 resampler_heads=8, resampler_layers=1):
        super().__init__()
        self.keep_cam_dec = bool(keep_cam_dec)
        self.cam_token_per_sample = bool(cam_token_per_sample)
        layers = str(layers or 'last')
        if layers not in self.LAYER_MODES:
            raise ValueError(f"da3_geo_layers must be one of {self.LAYER_MODES}, got {layers!r}")
        if str(layer_fuse or 'concat') != 'concat':
            raise ValueError(f"da3_geo_layer_fuse: 'concat' 만 구현되어 있다 (got {layer_fuse!r})")
        self.layers = layers
        self.debug = bool(debug)
        self.ref_view_strategy = str(ref_view_strategy or 'saddle_balanced')

        self.input_hw = tuple(int(v) for v in input_hw) if input_hw else None
        if self.input_hw and any(v % _PATCH for v in self.input_hw):
            raise ValueError(f"da3_geo_input_hw={self.input_hw} must be a multiple of "
                             f"patch {_PATCH}")

        _init_da3(repo_path, hf_home)
        self.net, self._cfg_used = self._build(model_name, ckpt_path, hf_home)

        bb = self.net.backbone
        embed = int(bb.pretrained.embed_dim)
        n_out = len(bb.out_layers)
        per_layer = embed * (2 if bool(bb.cat_token) else 1)
        native = per_layer * (n_out if layers == 'all' else 1)
        self.out_dim = native
        self.embed_dim = embed
        self.camera_encoding_dim = embed      # cam_token 은 x[:, :, 0] 자리에 그대로 꽂힌다

        # cat_token 두 반쪽(un-normed local_x | normed x)의 스케일 차이를 여기서 맞춘다.
        self.ln = nn.LayerNorm(native) if str(norm or 'ln') == 'ln' else nn.Identity()

        # [new 2026-09-06, D144] per-view resampler. None(기본)이면 아래 from_raw 가 예전 경로를
        # 글자 그대로 탄다 — 기존 arm 의 state_dict 와 출력이 비트 동일하다.
        self.resample = None
        if resampler:
            if str(resampler) != 'perceiver':
                raise ValueError(f"geo_resampler: 'perceiver' 만 구현되어 있다 (got {resampler!r})")
            self.resample = PerViewResampler(native, dim=int(resampler_dim),
                                             n_latents=int(resampler_tokens),
                                             heads=int(resampler_heads),
                                             layers=int(resampler_layers))
            self.out_dim = int(resampler_dim)
            _r = sum(p.numel() for p in self.resample.parameters())
            print(f"(geo_encoder/da3) resampler perceiver: {native} -> {resampler_dim} "
                  f"x {resampler_tokens} tok/view ({_r / 1e6:.1f} M, trainable)")

        self.register_buffer('_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('_std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
                             persistent=False)

        _n = sum(p.numel() for p in self.net.parameters())
        print(f"(geo_encoder/da3) {model_name}: backbone+cam_enc {_n / 1e6:.1f} M (frozen) | "
              f"out_layers={list(bb.out_layers)} cat_token={bool(bb.cat_token)} "
              f"embed={embed} -> native {native} | input_hw={self.input_hw} "
              f"cam_enc={'yes' if self.net.cam_enc is not None else 'NO'}")

    # ------------------------------------------------------------------ build
    def _build(self, model_name, ckpt_path, hf_home):
        """DepthAnything3Net 을 만들고 backbone + cam_enc 만 남긴다.

        nested 모델(`da3nested-giant-large`)은 anyview / metric 두 갈래인데 metric 갈래는
        depth 스케일 정렬 전용이라 feature 에 관여하지 않는다. **anyview 갈래만** 만들고
        state_dict 도 `model.da3.` prefix 만 골라 쓴다 (metric 334 M 을 아예 안 올린다).
        """
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from safetensors.torch import load_file

        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"da3_geo_model={model_name!r} 는 DA3 registry 에 없다. "
                           f"가능: {sorted(MODEL_REGISTRY)}")
        conf = load_config(MODEL_REGISTRY[model_name])
        nested = 'anyview' in conf
        if nested:
            conf = conf['anyview']
            state_prefix = 'model.da3.'
        else:
            state_prefix = 'model.'
        # gs_head/gs_adapter 는 3DGS 전용이라 아예 만들지 않는다 (giant 기준 수십 MB).
        for k in ('gs_head', 'gs_adapter'):
            if k in conf:
                del conf[k]
        net = create_object(conf)

        # cam_enc 는 cam_dec 가 있을 때만 생성된다 -> 만든 뒤에 지운다.
        net.head = None
        if not self.keep_cam_dec:
            net.cam_dec = None      # 기본: geo 토큰만 필요하므로 삭제 (기존 arm 과 비트 동일)
        net.gs_head = None
        net.gs_adapter = None
        if net.cam_enc is None:
            raise RuntimeError(f"da3_geo_model={model_name!r} 에 cam_enc 가 없다 "
                               f"(geo_posed 를 쓸 수 없는 모델)")

        path = ckpt_path or _hf_snapshot(model_name, hf_home)
        if osp.isdir(path):
            path = osp.join(path, 'model.safetensors')
        if not osp.isfile(path):
            raise FileNotFoundError(
                f"DA3 weights not found: {path} (da3_geo_ckpt_path / da3_geo_hf_home 확인)")
        sd = load_file(path)
        _parts = ['backbone.', 'cam_enc.'] + (['cam_dec.'] if self.keep_cam_dec else [])
        keep = tuple(state_prefix + p for p in _parts)
        sd = {k[len(state_prefix):]: v for k, v in sd.items() if k.startswith(keep)}
        if not sd:
            raise RuntimeError(f"{path} 에 '{state_prefix}backbone.*' 키가 없다 "
                               f"(model_name/nested 판정이 틀렸다)")
        # head/cam_dec 를 이미 지웠으므로 strict=True 가 성립한다 — 조용한 missing key 를 막는다.
        net.load_state_dict(sd, strict=True)
        print(f"(geo_encoder/da3) loaded {len(sd)} tensors from {path} "
              f"(prefix '{state_prefix}', nested={nested})")

        net.requires_grad_(False)
        net.eval()
        return net, conf

    def train(self, mode=True):
        """DA3 본체는 항상 eval. (부모의 .train() 이 재귀적으로 되돌리는 것을 막는다.)"""
        super().train(mode)
        self.net.eval()
        return self

    # ------------------------------------------------------------- cam_token
    @torch.no_grad()
    def build_cam_token(self, geo_c2w, geo_fxfycxcy, geo_hw, override_scale=None):
        """geo view 의 GT 카메라 -> DA3 cam_enc pose token (B, V, embed_dim).

        정규화는 **DA3 자신의 규약**(`api.py::_normalize_extrinsics`)을 따른다: view0 기준으로
        재고정한 뒤 camera center 거리의 median 으로 나눈다. 파이프라인의 `norm_scale` 을 쓰지
        않는 이유는 cam_enc 가 그 스케일 분포에서 학습됐기 때문이다.
        원본과 다른 점 하나: api 의 `torch.median(dists)` 는 **배치 전역**이라 B>1 에서 샘플이
        섞인다. 여기서는 **per-sample median** 으로 잡는다 (B=1 이면 원본과 동일).

        geo_c2w (B,V,4,4) OpenCV c2w / geo_fxfycxcy (B,V,4) px / geo_hw (B,2) [h,w].
        fxfycxcy 와 geo_hw 는 같은 픽셀 공간이기만 하면 된다 — pose encoding 은
        `fov = 2*atan((H/2)/fy)` 만 쓰고 full-frame resize 는 fy 와 H 를 똑같이 스케일하므로
        입력 해상도가 달라도 FoV 는 비트 단위로 같다 (cx/cy 는 쓰이지 않는다).
        """
        if override_scale is not None:
            raise ValueError("geo_encoder='da3' 는 override_scale(geo_lagernvs_skip_ctx_norm)을 "
                             "지원하지 않는다 — DA3 는 자기 규약(median camera distance)으로 "
                             "정규화한다")
        from depth_anything_3.utils.geometry import affine_inverse

        cam_enc = self.net.cam_enc
        dev = next(cam_enc.parameters()).device
        B, V = geo_c2w.shape[:2]

        c2w = geo_c2w.float().to(dev)
        w2c = affine_inverse(c2w)
        w2c = w2c @ c2w[:, :1]                                   # view0 기준 재고정
        c2w_n = affine_inverse(w2c)
        dists = c2w_n[..., :3, 3].norm(dim=-1)                   # (B,V)
        med = dists.median(dim=1).values.clamp(min=1e-1).view(B, 1, 1)
        w2c = w2c.clone()
        w2c[..., :3, 3] = w2c[..., :3, 3] / med

        f = geo_fxfycxcy.float().to(dev)
        K = torch.zeros(B, V, 3, 3, device=dev, dtype=torch.float32)
        K[..., 0, 0], K[..., 1, 1] = f[..., 0], f[..., 1]
        K[..., 0, 2], K[..., 1, 2] = f[..., 2], f[..., 3]
        K[..., 2, 2] = 1.0

        hw = geo_hw.to(dev)
        # cam_enc 는 batch 당 하나의 (H,W) 만 받는다. 보통 scene 해상도가 같아서 한 번에 끝나고,
        # 섞인 배치에서만 샘플별로 쪼갠다.
        #
        # [new 2026-08-29] cam_token_per_sample: 수학적으로는 배치가 섞이지 않는데도 cam_enc 를
        # B>1 로 부르면 cuBLAS 가 다른 kernel 을 골라 결과가 `max|d| 1.0e-5` (rel 2.9e-7) 만큼
        # 달라진다. 그 자체는 무시할 크기지만 **DA3 backbone 이 이걸 5만배로 증폭한다** — cam_token
        # 은 x[:, :, 0] 에 꽂혀 전 view 가 attend 하는 자리라, 3e-7 짜리 섭동이 pre-ln 토큰에서
        # rel 2.3e-2 (token cos 평균 0.9997 / 최소 0.968) 로 커진다 (실측). 즉 **같은 샘플이
        # batch_size 8 로 학습될 때와 1 로 추론될 때 서로 다른 geo token 을 본다.** 켜면
        # 샘플별로 쪼개 불러 배치 크기에 완전히 불변이 된다 (per-sample[0] vs 단독 B=1: max|d| 0).
        # 비용은 B=8 기준 3.3 -> 26.1 ms, backbone 468 ms 대비 +4.9%.
        # 기본값 false 는 기존 arm 재현용이고, geo_raw_cache_dir (B=1 로 구운 캐시) 을 쓸 때는
        # true 여야 캐시 경로와 on-the-fly 경로가 일치한다 (dataset_cfg 가 경고한다).
        with torch.autocast(device_type=dev.type, enabled=False):   # da3.py:127 과 동일
            if self.cam_token_per_sample and B > 1:
                tok = torch.cat([
                    cam_enc(w2c[b:b + 1], K[b:b + 1],
                            (int(hw[b, 0].item()), int(hw[b, 1].item())))
                    for b in range(B)], dim=0)
            elif bool((hw == hw[0:1]).all()):
                tok = cam_enc(w2c, K, (int(hw[0, 0].item()), int(hw[0, 1].item())))
            else:
                tok = torch.cat([
                    cam_enc(w2c[b:b + 1], K[b:b + 1],
                            (int(hw[b, 0].item()), int(hw[b, 1].item())))
                    for b in range(B)], dim=0)
        return tok.float()

    # ------------------------------------------------------- cam_dec (opt-in)
    @torch.no_grad()
    def predict_cameras(self, images, cam_token=None):
        """DA3 가 원래 하듯 latent 에서 **카메라를 뽑는다** — `keep_cam_dec=True` 일 때만.

        왜 필요한가: `cam_enc` 로 pose 를 넣어 줘도 나오는 **translation 은 항상 예측값**이다.
        `cam_dec.py:35` 의 `out_t = self.fc_t(feat)` 에는 echo 경로가 아예 없고 (rotation/fov 만
        `camera_encoding` 이 주어지면 echo 된다), `da3.py:216` 은 `camera_encoding` 없이 부른다.
        즉 latent 이 실제로 표현하는 카메라 스케일은 우리가 넣어 준 스케일(median camera
        distance) 과 다르다. 그 비율을 재려면 이 예측 카메라가 필요하다.

        `da3.py:_process_camera_estimation` 과 같은 경로다:
          pose_enc = cam_dec(feats[-1][1]) -> pose_encoding_to_extri_intri -> c2w, K
        `feats[-1][1]` 은 마지막 out_layer 의 **camera token** (dim = embed, cat 안 된 쪽)이라
        `forward()` 가 버리는 `outs[i][1]` 이다.

        returns c2w (B,V,3,4) float32, K (B,V,3,3) float32 — DA3 자기 출력 공간 그대로.
        """
        if self.net.cam_dec is None:
            raise RuntimeError("predict_cameras() 는 keep_cam_dec=True 로 만든 encoder 에서만 "
                               "쓸 수 있다 (기본은 cam_dec 를 지운다)")
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

        x = self._prep_images(images)
        H, W = x.shape[-2], x.shape[-1]
        outs, _ = self.net.backbone(
            x, cam_token=cam_token, export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy)
        with torch.autocast(device_type=x.device.type, enabled=False):   # da3.py:218 과 동일
            pose_enc = self.net.cam_dec(outs[-1][1].float())
            c2w, K = pose_encoding_to_extri_intri(pose_enc, (H, W))
        return c2w.float(), K.float()

    # ---------------------------------------------------------------- forward
    def _prep_images(self, images):
        B, V = images.shape[:2]
        x = images.float().flatten(0, 1)                        # (B*V,3,H,W)
        if self.input_hw and tuple(x.shape[-2:]) != self.input_hw:
            if self.debug:
                print(f"(geo_encoder/da3) resize {tuple(x.shape[-2:])} -> {self.input_hw} "
                      f"(dataset 이 da3_geo_input_hw 로 안 주고 있다)")
            x = F.interpolate(x, size=self.input_hw, mode='bilinear',
                              align_corners=False, antialias=True)
        if any(v % _PATCH for v in x.shape[-2:]):
            raise ValueError(f"geo image hw {tuple(x.shape[-2:])} must be a multiple of "
                             f"patch {_PATCH}")
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        return x.view(B, V, *x.shape[1:])

    @torch.no_grad()
    def encode_raw(self, images, cam_token=None):
        """`forward` 의 **frozen 구간만** 돌려 pre-ln 토큰 (B,V,P,C_native) 을 낸다.

        [new 2026-08-29] 캐시 지점이 왜 하필 여기인가: `self.ln` 과 그 위의 `GeoEncoder.proj`
        (3072->768) 는 **학습되는** 파라미터다 (`geo_encoder.py:201 trainable = True`). 그 출력을
        파일로 얼리면 두 모듈이 초기값에 영원히 머문다 — `dataset_cfg.py:360` 이 da3 에서
        `geo_latent_cache_dir` 을 끄는 이유가 정확히 이것이다. 반대로 `self.net` 은
        `:239 requires_grad_(False)` + 영구 eval 이라 같은 (images, cam_token) 에 항상 같은 값을
        낸다. 그래서 잘라도 되는 유일한 선이 backbone 출력과 `ln` 사이다.

        Vista4D 처럼 context view 가 scene 상수인 코퍼스에서는 이 값이 **변이(=segment) 와
        무관**하므로 scene 당 파일 하나로 전 변이가 공유된다 (52 scene x 42.5 MB ~ 2.2 GB).

        반환 dtype 은 bf16 autocast 아래서도 **fp32** 다. autocast 가 내리는 건 matmul/linear 뿐이고
        블록이 `x = x + attn(ln(x))` 라 residual stream 이 fp32 로 남는다. 캐시를 bf16 으로 저장하면
        왕복에서 `max|d| 2.0` / `||d||/||a|| 1.8e-3` 이 생기므로 **fp32 로 저장할 것**.
        배치 크기에는 불변이다 (B=1/2/6, 동료 scene·순서 무관 max|d| 0.0000 실측) — global
        attention 이 `b (s n) c` 로 배치 축을 안 섞고, reference-view 선택은 view 수만 본다.
        따라서 B=1 로 구운 캐시를 batch_size=8 학습에 그대로 먹여도 된다.
        """
        x = self._prep_images(images)
        # export_feat_layers 는 None 이면 backbone 안에서 `i in None` 으로 터진다 -> [] 필수.
        # cam_token 을 주면 select_reference_view / reorder_by_reference 가 건너뛰어져
        # (그 분기는 cam_token is None 일 때만 발동) view 순서가 보존된다.
        outs, _ = self.net.backbone(
            x, cam_token=cam_token, export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy)
        feats = [o[0] for o in outs]            # out_layer 당 patch token (B,V,P,per_layer)
        if self.layers == 'last':
            return feats[-1].float()
        return torch.cat([f.float() for f in feats], dim=-1)

    def from_raw(self, t):
        """`encode_raw` 의 결과(또는 그 캐시) -> tokens (B, V*P, out_dim).

        학습되는 `ln` 이 여기 걸린다 — 캐시 경로에서도 gradient 가 살아 있어야 하므로
        `no_grad` 를 걸지 않는다.

        resampler 가 켜져 있으면 view 안에서 P -> R 로 줄여 (B, V*R, resampler_dim) 을 낸다
        (D144). 꺼져 있으면(기본) 예전 그대로 (B, V*P, native)."""
        t = self.ln(t)
        if self.resample is None:
            return einops.rearrange(t, 'b v p c -> b (v p) c')
        return einops.rearrange(self.resample(t), 'b v r c -> b (v r) c')

    def forward(self, images, cam_token=None):
        """images (B,V,3,H,W) float [0,1] (정규화 전) -> tokens (B, V*P, out_dim)."""
        return self.from_raw(self.encode_raw(images, cam_token))
