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
                 keep_cam_dec=False):
        super().__init__()
        self.keep_cam_dec = bool(keep_cam_dec)
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
        with torch.autocast(device_type=dev.type, enabled=False):   # da3.py:127 과 동일
            if bool((hw == hw[0:1]).all()):
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

    def forward(self, images, cam_token=None):
        """images (B,V,3,H,W) float [0,1] (정규화 전) -> tokens (B, V*P, out_dim)."""
        x = self._prep_images(images)
        with torch.no_grad():
            # export_feat_layers 는 None 이면 backbone 안에서 `i in None` 으로 터진다 -> [] 필수.
            # cam_token 을 주면 select_reference_view / reorder_by_reference 가 건너뛰어져
            # (그 분기는 cam_token is None 일 때만 발동) view 순서가 보존된다.
            outs, _ = self.net.backbone(
                x, cam_token=cam_token, export_feat_layers=[],
                ref_view_strategy=self.ref_view_strategy)
        feats = [o[0] for o in outs]            # out_layer 당 patch token (B,V,P,per_layer)
        if self.layers == 'last':
            t = feats[-1].float()
        else:
            t = torch.cat([f.float() for f in feats], dim=-1)
        t = self.ln(t)
        return einops.rearrange(t, 'b v p c -> b (v p) c')
