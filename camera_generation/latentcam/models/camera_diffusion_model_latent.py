import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim):
    """Sinusoidal embedding. t may be (B,) or (B, T) (per-token) -> (..., dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half) / half
    ).to(t)
    args = t[..., None].float() * freqs                     # (..., half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

def positional_encoding(seq_len, dim, device='cpu'):
    pos = torch.arange(seq_len, device=device).float()[:, None]  # (N, 1)
    i = torch.arange(dim, device=device).float()[None, :]  # (1, D)
    angle_rates = 1 / (10000 ** (2 * (i//2) / dim))
    angle_rads = pos * angle_rates
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pe  # (N, D)

class FusedMLP(nn.Sequential):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: nn.Module,
        hidden_layer_multiplier: int = 4,
        bias: bool = True,
    ):
        super().__init__(
            nn.Linear(dim_model, dim_model * hidden_layer_multiplier, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * hidden_layer_multiplier, dim_model, bias=bias),
        )

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        t = timestep_embedding(t, self.dim)
        x = self.lin1(t)
        x = F.silu(x)
        x = self.lin2(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_weight = None

    def forward(self, x, context, attn_mask=None, key_padding_mask=None):
        a, attn_weight = self.attn(x, context, context, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        self.attn_weight = attn_weight
        return self.norm(x + a)
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_weight = None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a, attn_weight = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if self.training is False:
            self.attn_weight = attn_weight
        return self.norm(x + a)


class CameraDiffusionModel(nn.Module):
    def __init__(
        self,
        cam_dim=11,
        time_emb_dim=512,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        geo_latent_dim=768,
        text_dim=4096,
        dropout=0.1,
        geo_encoder=None,
        geo_cam_raw_dim=0,
        geo_cam_embed_dim=128,
        cond_dim=0,
        video_latent_dim=0,
        video_text_dim=0,
        peav_in_ln=True,
        text_in_ln=False,
        video_gate=True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads

        # [new 2026-08-27] cond_dim > 0 (cfg.target_track_dim): x_t 채널에 per-token 조건
        # (subject OBB track 등)을 concat 한다. concat 조건은 cross-attn 과 달리 uncond
        # forward 에서 "빼기"가 불가능하므로 null = 전 채널 0 (forward 의 cond=None 분기).
        # 0 = 기존 구조 그대로 (기존 ckpt 와 state_dict 호환).
        self.cond_dim = int(cond_dim)
        self.cam_in = nn.Linear(cam_dim + self.cond_dim, hidden_dim)

        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.mod1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mod2 = nn.Linear(hidden_dim, hidden_dim * 2)

        # [new 2026-09-03 / FIX] frozen encoder feature 를 bare Linear 에 그냥 먹이면 안 된다.
        # DA3 geo 는 `da3_geo_encoder.self.ln` (native 3072 위의 학습 LayerNorm, :171/:407) 를
        # 거친 뒤에야 geo_proj 로 들어간다. umt5 는 출력이 이미 ~unit 이라 LN 없이 통했지만
        # PE-AV 는 아니다 — 실측 video std 8 / text std 95 · absmax 12928 (d107 캐시).
        # 이걸 LN 없이 넣으면 video_tok std 26 (text part 41) vs text_tok 0.58 / geo_tok 0.58 로
        # 45~70배라 CA logit 이 포화하고, D117 두 arm 이 loss ~1.0 (= eps 예측이 0) 에서
        # 못 빠져나왔다. 기본 True 지만 **video/PEAV 스트림에만** 걸리므로 기존 arm 은 무영향.
        self.text_in_ln = bool(text_in_ln)
        self.text_ln = nn.LayerNorm(text_dim) if self.text_in_ln else nn.Identity()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # [new] geo_cam_raw_dim > 0 (cfg.geo_cam_embed): geo_emb arrives as
        # (B, M, geo_latent_dim + geo_cam_raw_dim) — the trailing raw dims are the per-context-view
        # camera pose relative to the target's first camera, broadcast to that view's patches by
        # the training loop. They are split off here and lifted by a TRAINABLE MLP before being
        # concatenated back, so the frozen LagerNVS part stays cacheable. 0 = the original layout.
        self.geo_cam_raw_dim = int(geo_cam_raw_dim)
        if self.geo_cam_raw_dim > 0:
            self.geo_cam_mlp = nn.Sequential(
                nn.Linear(self.geo_cam_raw_dim, geo_cam_embed_dim), nn.SiLU(),
                nn.Linear(geo_cam_embed_dim, geo_cam_embed_dim))
            self.geo_proj = nn.Linear(geo_latent_dim + geo_cam_embed_dim, hidden_dim)
        else:
            self.geo_proj = nn.Linear(geo_latent_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_dim), 
                SelfAttention(hidden_dim, num_heads),
                CrossAttention(hidden_dim, num_heads),
                FusedMLP(hidden_dim, dropout, nn.GELU, hidden_layer_multiplier=4),
                nn.LayerNorm(hidden_dim),
                CrossAttention(hidden_dim, num_heads),
                FusedMLP(hidden_dim, dropout, nn.GELU, hidden_layer_multiplier=4),
            ])
            for _ in range(num_layers)
        ])

        # [new 2026-09-03] video CA (D117). PE-AV(pe-av-large) 의 per-frame video token 을
        # 세 번째 cross-attn 스트림으로 붙인다. 순서는 **text CA → video CA → geo CA**.
        # 왜 `self.layers` 를 7→10 모듈로 늘리지 않고 별도 ModuleList 를 두는가:
        #   ① 기존 체크포인트의 state_dict 키(`layers.<i>.<0..6>.*`)가 한 글자도 안 바뀐다.
        #   ② video_latent_dim=0 이면 이 블록이 아예 생성되지 않아 파라미터 수·forward 경로가
        #      기존과 비트 동일하다 (프로젝트 규칙: option 분기로 기존 방식 보존).
        # 블록 구성은 geo 스트림과 같은 [LayerNorm, CrossAttention, FusedMLP] 3종이고,
        # FiLM 변조기(mod3)도 geo 의 mod2 와 같은 모양으로 따로 둔다.
        self.video_latent_dim = int(video_latent_dim)
        # [new 2026-09-03] arm A 전용: PE-AV **text** 토큰을 같은 video CA 의 key/value 로 앞에
        # 붙인다 (`[NL text tokens | 49 frame tokens]`). PE-AV text 는 1024-d, video 는 1792-d 라
        # 토큰축 concat 전에 **각각** hidden 으로 투영해야 한다 — 그래서 proj 가 두 개다.
        # 0 = video 토큰만 (arm B / 기본).
        self.video_text_dim = int(video_text_dim)
        self.peav_in_ln = bool(peav_in_ln)
        if self.video_latent_dim > 0:
            self.video_ln = (nn.LayerNorm(self.video_latent_dim) if self.peav_in_ln
                             else nn.Identity())
            self.video_proj = nn.Linear(self.video_latent_dim, hidden_dim)
            if self.video_text_dim > 0:
                self.video_text_ln = (nn.LayerNorm(self.video_text_dim) if self.peav_in_ln
                                      else nn.Identity())
                self.video_text_proj = nn.Linear(self.video_text_dim, hidden_dim)
            self.mod3 = nn.Linear(hidden_dim, hidden_dim * 2)
            # [new 2026-09-03 / FIX-D117] video 스트림만 **0 초기화 residual gate** 로 붙인다.
            # 기존 스트림은 `h = CA(...)` 로 잔차를 **덮어쓴다** (CrossAttention.forward 가
            # `norm(x + a)` 를 돌려주므로 스트림 하나당 h 가 한 번 더 정규화된다). 스트림이
            # 2개면 x_t 성분이 층당 대략 1/√2 씩 깎이는데, 3개가 되면 층당 한 번이 더 붙어
            # num_layers=8 에서 누적 감쇠가 ~11배 더 커진다. 실측이 이걸 그대로 보여줬다:
            #   · 같은 코드 + video CA off + peav 캐시 로드 → D108 과 step 350 까지 비트 동일
            #     (250:0.6956 300:0.1774 350:0.1784)
            #   · text CA 를 PE-AV 로 바꾸고 video CA 만 끔 → 350:0.8483 500:0.2975 700:0.1620
            #   · video CA 켠 두 arm → 55/84 epoch 동안 loss ~1.0 (= eps 예측이 0) 에서 정체
            # 즉 데이터 경로도 PE-AV text 도 무죄고, 남는 건 "스트림을 하나 더 덮어쓴 것"뿐이다.
            # gate=0 이면 **초기 출력이 D108 과 비트 동일**하고 필요한 만큼만 열린다 (DiT/Flamingo
            # 방식). video_gate=false 면 예전처럼 덮어쓰기 — 기존 D117 ckpt 재현용으로 남긴다.
            self.video_gate = (nn.Parameter(torch.zeros(num_layers)) if bool(video_gate)
                               else None)
            self.video_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(hidden_dim),
                    CrossAttention(hidden_dim, num_heads),
                    FusedMLP(hidden_dim, dropout, nn.GELU, hidden_layer_multiplier=4),
                ])
                for _ in range(num_layers)
            ])

        self.out = nn.Linear(hidden_dim, cam_dim)
        self.text_cross_attn_weight = None
        self.geo_cross_attn_weight = None
        self.video_cross_attn_weight = None

        self.geo_encoder = geo_encoder

    def _lift_geo_cam(self, geo_emb):
        """[new] split the trailing raw camera dims off geo_emb and replace them with the MLP
        embedding. Identity when geo_cam_raw_dim == 0 (the original geo condition)."""
        if self.geo_cam_raw_dim <= 0:
            return geo_emb
        d = self.geo_cam_raw_dim
        return torch.cat([geo_emb[..., :-d], self.geo_cam_mlp(geo_emb[..., -d:])], dim=-1)

    def _build_video_tok(self, video_emb, video_mask, video_text_emb, video_text_mask):
        """video CA 의 key/value 토큰과 그 padding mask 를 만든다.

        PE-AV video token 은 **소스 프레임당 1개**라 순서가 곧 시간축이다. text/geo 와 같은
        방식으로 위치 인코딩을 더해 준다 (안 더하면 CA 가 프레임 순서를 못 본다). arm A 에서
        PE-AV text 토큰이 같이 오면 **각 파트에 따로** PE 를 더한 뒤 토큰축으로 앞에 붙인다 —
        concat 후에 한 번에 더하면 text 길이가 배치마다 달라져 video 토큰의 위치가 흔들린다.

        반환: (tok (B,L,hidden), key_padding_mask (B,L) bool = True 가 마스킹 대상 | None)
        """
        B, device = video_emb.shape[0], video_emb.device
        vt = self.video_proj(self.video_ln(video_emb))
        vt = vt + positional_encoding(
            vt.shape[-2], vt.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
        if video_mask is not None:
            vt = vt * video_mask.unsqueeze(-1)
        if self.video_text_dim > 0 and video_text_emb is not None:
            tt = self.video_text_proj(self.video_text_ln(video_text_emb))
            tt = tt + positional_encoding(
                tt.shape[-2], tt.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
            if video_text_mask is not None:
                tt = tt * video_text_mask.unsqueeze(-1)
            tok = torch.cat([tt, vt], dim=1)
            if video_text_mask is None and video_mask is None:
                return tok, None
            tm = (video_text_mask if video_text_mask is not None
                  else tt.new_ones(B, tt.shape[1], dtype=torch.bool))
            vm = (video_mask if video_mask is not None
                  else vt.new_ones(B, vt.shape[1], dtype=torch.bool))
            return tok, ~torch.cat([tm, vm], dim=1)
        return vt, (None if video_mask is None else ~video_mask)

    def forward(self, x_t, t, text_emb, text_mask, geo_emb=None, geo_mask=None, cond=None,
                video_emb=None, video_mask=None, video_text_emb=None, video_text_mask=None):
        self.text_cross_attn_weight = None
        # [new 2026-09-03] video CA. `video_latent_dim==0` 이거나 video_emb 가 없으면 이 스트림은
        # 통째로 건너뛴다 = 기존 경로와 동일.
        has_video = (self.video_latent_dim > 0) and (video_emb is not None)
        if has_video:
            self.video_cross_attn_weight = None
        # geo latent is provided externally (on-the-fly frozen geo_encoder in the
        # training loop); condition on it whenever geo_emb is given.
        has_geo_latent = geo_emb is not None
        if has_geo_latent:
            self.geo_cross_attn_weight = None
        B, T, _ = x_t.shape

        device = x_t.device

        # [new 2026-08-27] per-token concat 조건. cond=None 이면 zeros = null 조건 — 계측용
        # 호출(geo_attn_probe 등)이 cond 를 모르고 불러도 죽지 않게 하려는 것이지, 학습/샘플링
        # 경로는 반드시 cond 를 명시적으로 넘겨야 한다 (train_latent_cam_dm.build_track_cond).
        if self.cond_dim > 0:
            if cond is None:
                cond = x_t.new_zeros(B, T, self.cond_dim)
            x_t = torch.cat([x_t, cond.to(x_t.dtype)], dim=-1)

        h = self.cam_in(x_t)

        pos_emb = positional_encoding(T, h.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
        h = h + pos_emb

        # t: (B,) scalar-per-sample (legacy, broadcast over tokens) OR (B,T) per-token noise.
        t_embed = self.time_mlp(t)              # (B,D) or (B,T,D)
        t_embed = self.time_proj(t_embed)       # (B,hidden) or (B,T,hidden)
        per_token = (t_embed.dim() == 3)        # per-token FiLM vs broadcast

        text_tok = self.text_proj(self.text_ln(text_emb))
        text_tok = text_tok + positional_encoding(text_tok.shape[-2], text_tok.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        if has_geo_latent:
            geo_tok = self.geo_proj(self._lift_geo_cam(geo_emb))
            geo_tok = geo_tok * geo_mask.unsqueeze(-1)

        if has_video:
            video_tok, video_kpm = self._build_video_tok(
                video_emb, video_mask, video_text_emb, video_text_mask)

        for _li, (norm1, self_attn, text_cross_attn, mlp1, norm2, geo_cross_attn, mlp2) in enumerate(self.layers):

            scale1, shift1 = self.mod1(t_embed).chunk(2, dim=-1)
            if not per_token:                          # (B,D)->(B,1,D) broadcast (legacy)
                scale1 = scale1.unsqueeze(1)
                shift1 = shift1.unsqueeze(1)           # per-token: (B,T,D) element-wise

            h = norm1(h)
            h = h * (1 + scale1) + shift1

            h = self_attn(h)
            h = text_cross_attn(h, text_tok, key_padding_mask=~text_mask)
            h = mlp1(h) + h

            # ── video CA: text CA 다음, geo CA 앞 (사용자 지정 순서) ──────────────────
            if has_video:
                vnorm, video_cross_attn, vmlp = self.video_layers[_li]
                scale3, shift3 = self.mod3(t_embed).chunk(2, dim=-1)
                if not per_token:
                    scale3 = scale3.unsqueeze(1)
                    shift3 = shift3.unsqueeze(1)
                v = vnorm(h)
                v = v * (1 + scale3) + shift3
                v = video_cross_attn(v, video_tok, key_padding_mask=video_kpm)
                v = vmlp(v) + v
                # gate=0 초기화 → 학습 시작 시점에 이 스트림은 no-op 이고 잔차가 안 깎인다.
                h = (h + self.video_gate[_li] * v) if self.video_gate is not None else v

            if has_geo_latent:
                scale2, shift2 = self.mod2(t_embed).chunk(2, dim=-1)
                if not per_token:
                    scale2 = scale2.unsqueeze(1)  # (B,1,D)
                    shift2 = shift2.unsqueeze(1)

                h = norm2(h)
                h = h * (1 + scale2) + shift2

                h = geo_cross_attn(h, geo_tok, key_padding_mask=~geo_mask)
                h = mlp2(h) + h

            if self.training is False:
                if self.text_cross_attn_weight is None:
                    self.text_cross_attn_weight = text_cross_attn.attn_weight
                    if has_geo_latent:
                        self.geo_cross_attn_weight = geo_cross_attn.attn_weight
                    if has_video:
                        self.video_cross_attn_weight = video_cross_attn.attn_weight
                else:
                    self.text_cross_attn_weight += text_cross_attn.attn_weight
                    if has_geo_latent:
                        self.geo_cross_attn_weight += geo_cross_attn.attn_weight
                    if has_video:
                        self.video_cross_attn_weight += video_cross_attn.attn_weight


        if self.training is False:
            self.text_cross_attn_weight /= len(self.layers)
            if has_geo_latent:
                self.geo_cross_attn_weight /= len(self.layers)
            if has_video:
                self.video_cross_attn_weight /= len(self.layers)

        h = h[:, :T, :]
        return self.out(h)# (B, T, 9)

    def forward_ar(self, x_t, t, text_emb, text_mask, geo_emb=None, geo_mask=None, history=None,
                   video_emb=None, video_mask=None, video_text_emb=None, video_text_mask=None):
        """Chunk-wise AR: denoise the current chunk x_t (B, Cc, cam_dim) conditioned on
        past CLEAN latents `history` (B, Ph, cam_dim) via CAUSAL self-attention over the
        concatenated sequence [history | current], plus text/geo cross-attn. Diffusion
        time modulation is applied to the current-chunk positions only (history is clean).
        Returns the predicted noise for the CURRENT chunk only (B, Cc, cam_dim)."""
        B, Cc, _ = x_t.shape
        device = x_t.device
        if history is not None and history.shape[1] > 0:
            seq = torch.cat([history, x_t], dim=1)          # (B, Ph+Cc, cam_dim)
            n_hist = history.shape[1]
        else:
            seq = x_t
            n_hist = 0
        L = seq.shape[1]

        h = self.cam_in(seq)
        h = h + positional_encoding(L, h.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        t_embed = self.time_proj(self.time_mlp(t))
        text_tok = self.text_proj(self.text_ln(text_emb))
        text_tok = text_tok + positional_encoding(text_tok.shape[-2], text_tok.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        has_geo_latent = geo_emb is not None
        if has_geo_latent:
            geo_tok = self.geo_proj(self._lift_geo_cam(geo_emb)) * geo_mask.unsqueeze(-1)

        has_video = (self.video_latent_dim > 0) and (video_emb is not None)
        if has_video:
            video_tok, video_kpm = self._build_video_tok(
                video_emb, video_mask, video_text_emb, video_text_mask)

        # causal self-attn mask: position i attends to positions <= i
        causal = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        # time modulation applies to current chunk positions only (history is clean)
        mod_pos = torch.zeros(1, L, 1, device=device)
        mod_pos[:, n_hist:, :] = 1.0

        for _li, (norm1, self_attn, text_cross_attn, mlp1, norm2, geo_cross_attn, mlp2) in enumerate(self.layers):
            s1, sh1 = self.mod1(t_embed).chunk(2, dim=-1)
            s1 = s1.unsqueeze(1); sh1 = sh1.unsqueeze(1)
            h = norm1(h)
            h = h * (1 + s1 * mod_pos) + sh1 * mod_pos
            h = self_attn(h, attn_mask=causal)
            h = text_cross_attn(h, text_tok, key_padding_mask=~text_mask)
            h = mlp1(h) + h
            if has_video:
                vnorm, video_cross_attn, vmlp = self.video_layers[_li]
                s3, sh3 = self.mod3(t_embed).chunk(2, dim=-1)
                s3 = s3.unsqueeze(1); sh3 = sh3.unsqueeze(1)
                v = vnorm(h)
                v = v * (1 + s3 * mod_pos) + sh3 * mod_pos
                v = video_cross_attn(v, video_tok, key_padding_mask=video_kpm)
                v = vmlp(v) + v
                h = (h + self.video_gate[_li] * v) if self.video_gate is not None else v
            if has_geo_latent:
                s2, sh2 = self.mod2(t_embed).chunk(2, dim=-1)
                s2 = s2.unsqueeze(1); sh2 = sh2.unsqueeze(1)
                h = norm2(h)
                h = h * (1 + s2 * mod_pos) + sh2 * mod_pos
                h = geo_cross_attn(h, geo_tok, key_padding_mask=~geo_mask)
                h = mlp2(h) + h

        return self.out(h[:, n_hist:])                      # current chunk only

def get_model():
    model = CameraDiffusionModel()
    return model