"""Molmo2 connector 를 랜덤 초기화로 붙이면 SigLIP2 feature 만으로 video 조건이 얼마나 되나 (R58 / D286).

사용자 지시 (2026-09-25): "molmo2 connector를 쓰되 새로 학습되게해줘", "ViT 자체는 freeze야",
입력은 캐시 없이 on-the-fly (27x27 bf16 캐시 ≈ 1.67 TB 라 디스크가 안 된다).

경로 (cfg.video_onfly = 'molmo2_conn'):
  dataset  씬 frame [s, e) 49장 -> Molmo2 video processor 와 같은 전처리 앞단
           (torchvision Resize 378x378 bilinear antialias=False, uint8)          `load_molmo2_frames`
  frozen   Molmo2-4B 체크포인트 안의 SigLIP2 ViT, vit_layers (-3,-9)=[24,18] concat
           -> 프레임당 27x27 patch x 2304-d. 학습 안 함 (항상 eval, no_grad)     `FrozenMolmo2ViT`
  학습     Molmo2 connector 와 **같은 구조**, 가중치는 랜덤 초기화:
           3x3 window attention pooling (query = window 평균, 16 head x 72, fp32 attention)
           -> SwiGLU projector 1152 -> 9728 -> 2560. 27x27 -> 9x9                  `Molmo2Connector`
  후처리   9x9 -> 8x8 프레임 내 평균 = molmo2_l21 arm 이 LM hidden 에 한 것과 같은 풀링
           -> (B, 49*64=3136, 2560) 이 video CA 의 video_emb 로 들어간다.

그래서 molmo2_l21_srccam 과의 차이는 "LM 을 뺐고 connector 는 새로 학습" 뿐이고, siglip2_srccam 과의
차이는 "27x27->8x8 고정 평균 대신 학습되는 attention pooling + projector" 뿐이다.
connector 는 CameraDiffusionModel 의 submodule(`video_connector`)로 붙어 ckpt state_dict 에 같이
저장되고, frozen ViT 는 모델 밖에 두어 ckpt 에 안 들어간다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MOLMO2_CKPT = ('/data1/cympyc1785/LatentCamVid/camera_generation/tools/molmo2/'
               'checkpoints/Molmo2-4B')
INPUT_HW = 378          # video_preprocessor_config size 378x378
PATCH = 14              # -> 27x27
POOL = 3                # pooling_size [3, 3] -> 9x9
MEAN = STD = 0.5        # image_mean / image_std


def load_molmo2_frames(frame_files, s, e):
    """frame [s, e) -> (T, 378, 378, 3) uint8. Molmo2 `resize_image` 의 uint8 분기와 같은 연산이다
    (squash resize, bilinear, antialias=False, clip 0..255). 정규화·patchify 는 GPU 쪽에서 한다."""
    from PIL import Image
    import torchvision
    ims = np.stack([np.asarray(Image.open(frame_files[i]).convert('RGB')) for i in range(s, e)])
    t = torch.from_numpy(ims).permute(0, 3, 1, 2)
    t = torchvision.transforms.Resize(
        (INPUT_HW, INPUT_HW), torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=False)(t)
    return torch.clip(t, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class FrozenMolmo2ViT(nn.Module):
    """Molmo2-4B 의 SigLIP2 ViT 만 떼어 쓴다. 체크포인트를 통째로 CPU 에 올렸다가 vision tower 만
    남기고 버린다 (LM 4B 는 GPU 에 안 올라간다)."""

    def __init__(self, ckpt=MOLMO2_CKPT, dtype=torch.bfloat16, chunk=98):
        super().__init__()
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=dtype, trust_remote_code=True)
        vb = m.model.vision_backbone
        self.image_vit = vb.image_vit
        self.vit_layers = list(vb.vit_layers)
        self.num_prefix = int(vb.num_prefix_tokens)
        self.dtype_ = dtype
        self.chunk = int(chunk)          # ViT 한 번에 넣는 프레임 수 (activation 메모리 상한)
        del m, vb
        for p in self.parameters():
            p.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):         # 부모 model.train() 이 불려도 항상 eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, frames):
        """frames (B, T, 378, 378, 3) uint8 -> (B, T, 729, 2304) bf16."""
        B, T = frames.shape[:2]
        side = INPUT_HW // PATCH
        x = frames.reshape(B * T, INPUT_HW, INPUT_HW, 3).float().div_(255.0)
        x = (x - MEAN) / STD
        x = x.reshape(B * T, side, PATCH, side, PATCH, 3).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B * T, side * side, PATCH * PATCH * 3).to(self.dtype_)
        outs = []
        for i in range(0, x.shape[0], self.chunk):
            hs = self.image_vit(x[i:i + self.chunk])
            f = torch.cat([hs[l] for l in self.vit_layers], dim=-1)
            outs.append(f[:, self.num_prefix:] if self.num_prefix > 0 else f)
        return torch.cat(outs, 0).reshape(B, T, side * side, -1)


class Molmo2Connector(nn.Module):
    """Molmo2VisionBackbone 의 image_pooling_2d + image_projector 와 같은 구조 (가중치는 새로).

    adapter_config: hidden 1152, heads 16 x head_dim 72, intermediate 9728, text_hidden 2560,
    pooling_attention_mask=True (27 은 3 의 배수라 padding 이 없어 mask 는 전부 참 = 평균 query).
    """

    def __init__(self, in_dim=2304, hidden=1152, heads=16, head_dim=72, inter=9728, out_dim=2560,
                 out_pool=8, init_std=0.02):
        super().__init__()
        self.heads, self.head_dim, self.out_pool = heads, head_dim, int(out_pool)
        self.wq = nn.Linear(in_dim, heads * head_dim)
        self.wk = nn.Linear(in_dim, heads * head_dim)
        self.wv = nn.Linear(in_dim, heads * head_dim)
        self.wo = nn.Linear(heads * head_dim, hidden)
        self.w1 = nn.Linear(hidden, inter, bias=False)
        self.w3 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, out_dim, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_std)   # config initializer_range 0.02
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats):
        """feats (B, T, 729, D) -> (B, T*out_pool^2, out_dim)."""
        B, T, N, D = feats.shape
        side = int(round(N ** 0.5))
        g = side // POOL
        # arange_for_pooling 과 같은 순서: "(h dh) (w dw) -> h w (dh dw)"
        x = feats.reshape(B * T, g, POOL, g, POOL, D).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B * T * g * g, POOL * POOL, D)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            q = self.wq(x.mean(1, keepdim=True))
            k, v = self.wk(x), self.wv(x)
        M = x.shape[0]
        q = q.float().view(M, 1, self.heads, self.head_dim).transpose(1, 2)
        k = k.float().view(M, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.float().view(M, -1, self.heads, self.head_dim).transpose(1, 2)
        # float32_attention=True. key 가 창당 9개뿐이라 softmax 를 직접 쓴다 — SDPA 는 배치 M=31,752
        # (배치 8 x 49 프레임 x 81 창) 에서 backward 가 42.8 GB 를 잡았다 (실측, 직접 계산은 아래).
        w = torch.softmax((q @ k.transpose(-1, -2)) * self.head_dim ** -0.5, dim=-1)
        a = w @ v
        a = a.transpose(1, 2).reshape(M, self.heads * self.head_dim)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = self.wo(a)
            h = self.w2(F.silu(self.w1(h)) * self.w3(h))
        h = h.float().view(B * T, g, g, -1).permute(0, 3, 1, 2)
        if self.out_pool != g:
            h = F.adaptive_avg_pool2d(h, (self.out_pool, self.out_pool))
        return h.permute(0, 2, 3, 1).reshape(B, T * self.out_pool ** 2, -1)


def build_video_onfly(cfg, model, device):
    """cfg.video_onfly 에 맞는 on-the-fly video 스트림을 만든다. 없으면 None (기존 arm 무영향).

    connector 는 `model.video_connector` 로 붙인다 — state_dict·optimizer·train/eval 전환이 모델을
    그대로 따라간다. 반환 dict 는 train_latent_cam_dm.build_video_cond 가 읽는다."""
    mode = str(getattr(cfg, 'video_onfly', None) or '')
    if not mode:
        return None
    assert mode == 'molmo2_conn', f"video_onfly must be null | 'molmo2_conn', got {mode!r}"
    assert not getattr(cfg, 'peav_video_cache_dir', None), \
        "video_onfly 는 video 캐시 대신 쓰는 경로다 — peav_video_cache_dir: null 로 둘 것"
    assert int(getattr(cfg, 'video_latent_dim', 0) or 0) == 2560, \
        "molmo2_conn 출력은 2560-d (text_hidden_size) — video_latent_dim: 2560"
    model.video_connector = Molmo2Connector(out_pool=int(getattr(cfg, 'video_onfly_pool', 8)))
    vit = FrozenMolmo2ViT().to(device)
    n = sum(p.numel() for p in model.video_connector.parameters())
    print(f"(model) video_onfly=molmo2_conn: frozen Molmo2 SigLIP2 ViT layers {vit.vit_layers} "
          f"(27x27x2304) -> connector 랜덤 초기화 {n / 1e6:.2f} M params (학습) -> "
          f"9x9 -> {model.video_connector.out_pool}x{model.video_connector.out_pool} 평균 -> 2560-d")
    return {'vit': vit, 'conn': model.video_connector}
