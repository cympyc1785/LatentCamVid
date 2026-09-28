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


class PeavReadout(nn.Module):
    """[new 2026-09-15] PE-AV/molmo2 캐시 토큰을 **학습되는 shallow transformer** 로 한 번 요약한다.

    D194 까지는 캐시 hidden state(molmo2 l21 기준 video 3136 + text 128 토큰)가 LayerNorm +
    Linear 하나만 거쳐 곧장 video CA 의 key/value 로 들어갔다. 즉 frozen encoder 의 좌표계를
    DiT 가 층마다 직접 읽는 구조라, "이 씬에서 어느 물체가 subject 이고 그게 어디로 가는가"를
    중간에서 정리해 주는 자리가 없었다.

    여기서는 `num_queries` 개의 learnable query 가 [self-attn(질의끼리) → cross-attn(캐시 토큰)
    → MLP] 를 `num_layers` 번 돌아 그 요약을 만들고, **그 출력이 video CA 의 key/value 를
    대신한다**. query 수를 프레임 수(기본 49)로 두는 이유가 둘이다:
      ① 토큰이 3264 → 49 로 줄어 video CA 비용이 그만큼 내려간다.
      ② query i 가 프레임 i 에 대응하므로 `aux_head` 가 **프레임별 subject OBB center** 를
         그대로 뱉을 수 있다.

    입력은 `_build_video_tok` 이 만든 **[video_text | video] 합본**이다 — caption 토큰을 같이
    먹여야 query 가 "어느 물체" 인지를 알 수 있고, 그게 OBB 보조 손실이 성립하는 전제다.

    `aux_head` 는 **DiT 출력에 전혀 기여하지 않는다**. `forward` 가 (요약, aux예측) 을 따로
    돌려주고 요약만 CA 로 흘러가므로, 추론에서 aux 를 안 읽으면 그게 곧 "MLP 를 뗀" 상태다
    (가중치는 ckpt 에 남지만 계산 그래프에서 분리되어 있다).

    `num_layers=0` 이면 이 모듈이 아예 생성되지 않는다 (기존 arm 과 state_dict·동작 비트 동일).
    """

    def __init__(self, dim, num_heads, num_queries, num_layers, dropout, aux_dim=0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(dim, num_heads),
                CrossAttention(dim, num_heads),
                FusedMLP(dim, dropout, nn.GELU, hidden_layer_multiplier=4),
            ])
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(dim)
        self.aux_head = (nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(),
                                       nn.Linear(dim, int(aux_dim)))
                         if int(aux_dim) > 0 else None)

    def forward(self, tok, key_padding_mask=None):
        """(B,L,D) 캐시 토큰 → ((B,Q,D) 요약, (B,Q,aux_dim) aux 예측 | None)."""
        q = self.query.unsqueeze(0).expand(tok.shape[0], -1, -1)
        for self_attn, cross_attn, mlp in self.layers:
            q = self_attn(q)
            q = cross_attn(q, tok, key_padding_mask=key_padding_mask)
            q = mlp(q) + q
        q = self.out_ln(q)
        return q, (self.aux_head(q) if self.aux_head is not None else None)


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
        peav_readout_layers=0,
        peav_readout_queries=49,
        peav_readout_aux_dim=0,
        peav_readout_mode="replace",
        video_text_fuse="token",
        start_pose_dim=0,
        start_pose_tf_p=0.0,
        start_pose_noise=0.0,
        geo_in_mlp=False,
        geo_pe=False,
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
        # [new 2026-09-22 / D262] geo CA 입력이 frozen backbone feature 가 아니라 **raw 기하량**
        # (소스 카메라 Plücker/pose, cfg.srccam_cond) 일 때 두 가지가 달라진다:
        #   geo_in_mlp: 11~54 차원을 bare Linear 로 hidden 까지 올리면 표현력이 사실상 affine
        #               하나다. 2-layer MLP 로 올린다. DA3 3072-d 에는 불필요하다.
        #   geo_pe    : DA3 토큰은 무순서 view x patch 라 PE 를 안 붙였지만(cam_token 이 pose 를
        #               들고 있다), 소스 카메라 토큰은 **시간순 49개**라 순서가 정보다.
        # 둘 다 기본 off -> 기존 arm 은 state_dict·동작이 비트 동일.
        if bool(geo_in_mlp):
            _gi = self.geo_proj.in_features
            self.geo_proj = nn.Sequential(
                nn.Linear(_gi, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.geo_pe = bool(geo_pe)

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
        # [new 2026-09-21, D217] text part 를 video part 와 **어느 축으로** 합칠지.
        #   "token"        = 기존 (기본). [text 토큰 | video 토큰] 토큰축 concat, proj 2개.
        #   "frame_concat" = molmo2 decode 캐시 전용. decode 슬롯 49개는 `--decode_keep points
        #                    --fps 2.0` 덕분에 **프레임과 1:1 정렬**돼 있고 video 캐시는
        #                    49프레임 × 8×8 patch = 3136 이다. 토큰축으로 이어 붙이면 이 정렬이
        #                    버려지므로, 프레임 f 의 decode feature 를 그 프레임의 patch 64개에
        #                    **채널축으로** 붙여 (3136, 2560+2560) 한 덩어리로 만들고 projection
        #                    하나로 넣는다. 캐시는 기존 두 개를 그대로 쓴다 (molmo2 재굽기 0회).
        self.video_text_fuse = str(video_text_fuse or "token")
        assert self.video_text_fuse in ("token", "frame_concat"), self.video_text_fuse
        # video 스트림이 꺼진 arm 에서도 속성은 존재해야 forward 분기가 성립한다.
        self.peav_readout_layers = 0
        self.readout = None
        self.peav_readout_mode = str(peav_readout_mode)
        if self.video_latent_dim > 0:
            _fuse = self.video_text_dim > 0 and self.video_text_fuse == "frame_concat"
            self.video_ln = (nn.LayerNorm(self.video_latent_dim) if self.peav_in_ln
                             else nn.Identity())
            # frame_concat 은 두 파트를 **각각** LN 한 뒤 (scale 이 다르다 — FIX-D117) 채널축으로
            # 이어 붙이므로 입력 차원이 합이고 proj 는 하나뿐이다.
            self.video_proj = nn.Linear(
                self.video_latent_dim + (self.video_text_dim if _fuse else 0), hidden_dim)
            if self.video_text_dim > 0:
                self.video_text_ln = (nn.LayerNorm(self.video_text_dim) if self.peav_in_ln
                                      else nn.Identity())
                if not _fuse:
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
            # [new 2026-09-15] readout: video_tok 을 DiT 에 바로 먹이지 않고 shallow transformer
            # 로 한 번 요약한다 (§PeavReadout). 0 = 기존 경로 그대로 (state_dict·동작 비트 동일).
            self.peav_readout_layers = int(peav_readout_layers)
            self.readout = (PeavReadout(hidden_dim, num_heads, int(peav_readout_queries),
                                        self.peav_readout_layers, dropout,
                                        aux_dim=int(peav_readout_aux_dim))
                            if self.peav_readout_layers > 0 else None)

        # [new 2026-09-22 / D261] 시작 pose 를 궤적과 **같이** 예측한다.
        #
        # 왜 별도 토큰인가: 데이터셋이 궤적을 자기 frame0 으로 상대화하므로(`E @ inv(E[0])`)
        # `cam_param[0]` 은 항상 항등이고, "소스 대비 어디서 출발하는가" 는 x_t 어디에도 없다.
        # `dataset_dl3dv._start_pose` 가 그걸 9-d (6D 회전 + trans/norm_scale) 로 따로 낸다.
        # **11-d 가 아니다** — 시작 카메라는 소스와 같은 물리 카메라라 intr 2채널이 `intr_norm:
        # rel` 아래서 상수 [1,1] 이다 (d261 뱅크 실측 상대편차 0.0e+00). 넣으면 죽은 채널이다.
        #
        # 배치 방식: 시퀀스 뒤에 query 토큰 1개를 붙여 T -> T+1 로 만든다. self-attn 이 양방향
        # 이라 이 토큰은 궤적 전체를 보고, 궤적 토큰도 이 토큰을 본다. `cam_in` 채널에 concat
        # 하는 대안은 (a) 입력 폭이 바뀌어 기존 ckpt 와 state_dict 가 안 맞고 (b) 정답을 T개
        # 토큰 전부에 뿌려 head 가 자명해진다.
        #
        # teacher forcing(`start_pose_tf_p>0`): GT 에 노이즈를 얹어 `start_in` 으로 hidden 에
        # 올린 뒤 query 에 **더한다**. drop 은 element-wise 가 아니라 **샘플 단위**다 — drop 된
        # 쪽이 추론 경로(query 단독)와 비트 단위로 같아야 train/test 불일치가 안 생긴다.
        #
        # `start_pose_dim=0` (기본) 이면 아래 셋 다 안 만들어져 state_dict·동작이 기존과 동일.
        self.start_pose_dim = int(start_pose_dim)
        self.start_pose_tf_p = float(start_pose_tf_p)
        self.start_pose_noise = float(start_pose_noise)
        if self.start_pose_dim > 0:
            self.start_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.start_query, std=0.02)
            self.start_out = nn.Linear(hidden_dim, self.start_pose_dim)
            self.start_in = (nn.Linear(self.start_pose_dim, hidden_dim)
                             if self.start_pose_tf_p > 0 else None)
        # 이번 forward 의 시작 pose 예측. readout_aux_pred 와 같은 stash 방식이라 반환 타입이
        # 안 바뀐다 (샘플러/계측 호출이 전부 (B,T,cam_dim) 하나를 기대한다).
        self.start_pred = None

        self.out = nn.Linear(hidden_dim, cam_dim)
        self.text_cross_attn_weight = None
        self.geo_cross_attn_weight = None
        self.video_cross_attn_weight = None
        # readout 보조 head 의 예측. forward 마다 덮어쓰고 학습 루프가 읽어 간다 (attn_weight
        # 들과 같은 stash 방식). readout/aux 가 꺼져 있으면 영원히 None 이다.
        self.readout_aux_pred = None

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
        if (self.video_text_fuse == "frame_concat" and self.video_text_dim > 0
                and video_text_emb is not None):
            # (B, T, Ct) decode 와 (B, T*P, Cv) video 를 프레임 단위로 채널 concat.
            T, P = video_text_emb.shape[1], video_emb.shape[1] // max(video_text_emb.shape[1], 1)
            assert video_emb.shape[1] == T * P, \
                f"frame_concat: video 토큰 {video_emb.shape[1]} 이 decode 슬롯 {T} 의 배수가 아니다"
            tt = self.video_text_ln(video_text_emb)
            if video_text_mask is not None:
                # 해당 프레임에 decode 슬롯이 없으면 text 절반은 0 (visual 절반은 살린다).
                tt = tt * video_text_mask.unsqueeze(-1)
            vv = self.video_ln(video_emb).reshape(B, T, P, -1)
            vt = self.video_proj(
                torch.cat([vv, tt.unsqueeze(2).expand(-1, -1, P, -1)], dim=-1).reshape(
                    B, T * P, -1))
            vt = vt + positional_encoding(
                vt.shape[-2], vt.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
            if video_mask is not None:
                vt = vt * video_mask.unsqueeze(-1)
            return vt, (None if video_mask is None else ~video_mask)
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
                video_emb=None, video_mask=None, video_text_emb=None, video_text_mask=None,
                start_pose=None):
        self.text_cross_attn_weight = None
        # [new 2026-09-03] video CA. `video_latent_dim==0` 이거나 video_emb 가 없으면 이 스트림은
        # 통째로 건너뛴다 = 기존 경로와 동일.
        has_video = (self.video_latent_dim > 0) and (video_emb is not None)
        if has_video:
            self.video_cross_attn_weight = None
        # 직전 step 의 예측을 학습 루프가 잘못 집어 가지 않도록 매 forward 에서 비운다.
        self.readout_aux_pred = None
        self.start_pred = None
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

        # [new 2026-09-22 / D261] 시작 pose query 를 시퀀스 뒤에 1개 붙인다 (T -> T+1).
        # pos_emb 는 안 더한다 — 이 토큰은 궤적의 t번째가 아니라 별도 슬롯이고, 자리를 구분하는
        # 건 `start_query` 파라미터 자체다. `start_pose_dim==0` 이면 이 블록이 통째로 no-op.
        has_start = self.start_pose_dim > 0
        if has_start:
            q = self.start_query.to(h.dtype).expand(B, -1, -1)
            if self.start_in is not None and start_pose is not None and self.training:
                # teacher forcing. drop 은 **샘플 단위** — 남는 쪽이 추론 경로(query 단독)와
                # 비트 동일해야 train/test 불일치가 안 생긴다.
                sp = start_pose.to(h.dtype)
                if self.start_pose_noise > 0:
                    sp = sp + self.start_pose_noise * torch.randn_like(sp)
                keep = (torch.rand(B, 1, 1, device=device) < self.start_pose_tf_p).to(h.dtype)
                q = q + keep * self.start_in(sp).unsqueeze(1)
            h = torch.cat([h, q], dim=1)
            if per_token:
                # 시작 pose 는 frame0 과 같은 noise level 을 쓴다 (궤적의 첫 칸에 붙는 pose).
                t_embed = torch.cat([t_embed, t_embed[:, :1]], dim=1)

        text_tok = self.text_proj(self.text_ln(text_emb))
        text_tok = text_tok + positional_encoding(text_tok.shape[-2], text_tok.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        if has_geo_latent:
            geo_tok = self.geo_proj(self._lift_geo_cam(geo_emb))
            if self.geo_pe:     # [D262] srccam: 시간순 토큰이라 순서가 정보다
                geo_tok = geo_tok + positional_encoding(
                    geo_tok.shape[-2], geo_tok.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
            geo_tok = geo_tok * geo_mask.unsqueeze(-1)

        if has_video:
            video_tok, video_kpm = self._build_video_tok(
                video_emb, video_mask, video_text_emb, video_text_mask)
            # [new 2026-09-15] readout 을 거친 요약이 아래 층들의 key/value 가 된다. 요약
            # 토큰은 전부 유효하므로 padding mask 는 여기서 사라진다. AR 경로(`forward_ar`)엔
            # 안 걸어 두었다 — video_latent_dim>0 은 표준 diffusion 경로 전용이라 학습
            # 진입점에서 이미 assert 로 막혀 있다 (train_latent_cam_dm.py).
            if self.readout is not None:
                _rd, self.readout_aux_pred = self.readout(video_tok, video_kpm)
                if self.peav_readout_mode == 'append':      # [R81] 원 토큰 + 요약 49 토큰
                    if video_kpm is not None:
                        video_kpm = torch.cat([video_kpm, video_kpm.new_zeros(_rd.shape[:2])], dim=1)
                    video_tok = torch.cat([video_tok, _rd], dim=1)
                else:                                        # 'replace' (기존)
                    video_tok, video_kpm = _rd, None

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

        # [new 2026-09-22 / D261] 시작 pose 예측을 stash 하고 시퀀스를 원래 길이로 되돌린다.
        # 반환 타입이 안 바뀌어야 샘플러/계측 호출이 그대로 돈다 (readout_aux_pred 와 같은 방식).
        if has_start:
            self.start_pred = self.start_out(h[:, T:T + 1, :]).squeeze(1)   # (B, 9)
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