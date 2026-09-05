"""[new 2026-09-05] video CA(D117/D124 molmo2 스트림)가 **실제로 쓰이고 있는지**를 추론
1건마다 재는 계측기.

왜 필요한가: `video_gate` 는 0 초기화 residual gate 라 학습이 이 스트림을 안 쓰기로 정하면
gate 가 0 근처에 머문 채로도 loss 는 멀쩡히 내려간다. D124 epoch100 의 gate 벡터가 실측
`[0.0756, -0.0308, -0.0002, -0.0002, 0.0, -0.0004, -0.0003, -0.0]` 라 층 2~7 은 층 0 대비
~300배 작다. 그런데 gate 크기만으로는 결론을 못 낸다 — `video_proj.weight` norm 이 학습 중
20.88 -> 26.52 로 **커졌으므로** v 자체가 그만큼 커졌을 수 있다. 실제로 봐야 하는 건 곱이다:
  vresid = ||gate_i * v_i|| / ||h_i||        (h 는 그 층에서 video 스트림이 더해지는 잔차)

`geo_attn_probe`(train_latent_cam_dm.py:142)와 같은 사고방식이고, 같은 이유로 attention map
하나만으로는 부족하다 — softmax 는 행 합이 항상 1이라 "얼마나 쓰는가"가 안 나온다.

재는 것 (전부 **샘플별**, 배치 축을 안 접는다):
  gate_l{i}            층 i 의 video_gate 스칼라 (샘플 무관, 참고용)
  vresid_l{i}          ||gate_i * v_i|| / ||h_i||  <- 이 스트림의 실제 기여율
  vresid_mean          위의 층 평균
  text_resid_mean      text CA 의 ||a||/||x|| 층 평균 (vresid 의 눈금자)
  geo_resid_mean       geo CA 의 ||a||/||x|| 층 평균 (눈금자)
  vattn_text_mass      video CA attention 중 **molmo2 text 파트**(앞 128 토큰)로 간 질량
  vattn_video_mass     나머지 = 프레임 패치 파트 (1 - text_mass)
  vattn_entropy_norm   프레임 파트 attention 엔트로피 / ln(n_frames). 1.0 = 완전 균일
                       = 어느 프레임도 안 고르는 상태
  vattn_frame_peak     가장 많이 보는 프레임의 질량 x n_frames (1.0 = 균일)
  vattn_frame_peak_idx 그 프레임 인덱스
  vattn_frame_token_r  traj 토큰 인덱스 vs attention 가중 평균 프레임의 Pearson r.
                       1에 가까우면 "토큰 t 가 소스 프레임 ~4t 를 본다" = 시간축 정렬 독해
  dpred_drop           video 스트림을 통째로 끈 예측과의 상대 거리. 스트림 총 기여의 직접 지표
  dpred_shuffle        video 조건만 배치축으로 roll(1) 했을 때의 상대 거리.
                       **주의 — 이 열 하나로는 판정 못 한다.** eval 은 seg 순서대로 배치를 묶어서
                       roll(1) 짝의 99.3%(875 중 869)가 **같은 scene 의 이웃 seg** 다. 같은 씬의
                       49프레임 클립은 molmo2 임베딩이 거의 같으므로 이 값이 작은 건 당연하고,
                       "내용을 안 본다"의 근거가 못 된다. 판정은 아래 `dpred_xscene` 으로 한다.
  dpred_xscene         video 조건을 **다른 scene** 것으로 갈아끼웠을 때의 상대 거리
                       (`video_kw_alt` 를 준 경우에만). dpred_drop 과 비슷하면 video 를 내용으로
                       쓰는 것이고, 0 에 가까우면 내용 무관 bias 로만 쓰는 것이다.

사용 예시:
    from video_ca_probe import video_ca_probe
    rows = video_ca_probe(raw_model, scheduler, traj_latents, text_embeds, text_masks,
                          pc_embeds, pc_masks, video_kw, cond=_cond, n_frames=cfg.num_frames)
    for name, row in zip(data_name, rows):
        fh.write(json.dumps({"data_name": name, **row}, ensure_ascii=False) + "\n")
"""
import math

import torch


def _rowmean_norm(x):
    """(B,T,D) -> (B,) 토큰별 L2 노름의 토큰 평균. 배치 축을 접지 않는 게 요점이다."""
    return x.float().norm(dim=-1).mean(dim=-1)


def _pearson(a, b):
    """(B,T) 두 행렬의 행별 Pearson r -> (B,). 분산 0 이면 0."""
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(den > 1e-12, (a * b).sum(-1) / den.clamp(min=1e-12),
                       torch.zeros_like(den))


@torch.no_grad()
def video_ca_probe(raw_model, scheduler, z, text_emb, text_mask, geo_emb, geo_mask, video_kw,
                   cond=None, timestep=500, n_frames=49, noise_seed=1234, video_kw_alt=None):
    """샘플별 dict 의 리스트(길이 B)를 돌려준다. video 스트림이 없는 arm 이면 빈 리스트.

    노이즈는 `noise_seed` 로 고정한다 — 샘플 간/arm 간 수치를 비교하려면 x_t 가 같은 규칙으로
    만들어져야 한다 (geo_attn_probe 와 같은 이유). 확산 50스텝 전체가 아니라 timestep 한 점만
    보는 이유도 같다: 스텝마다 attention 이 달라지므로 한 점을 고정해야 비교가 성립한다.
    """
    if not video_kw or getattr(raw_model, 'video_latent_dim', 0) <= 0:
        return []
    gate = getattr(raw_model, 'video_gate', None)
    n_layers = len(raw_model.video_layers)
    buf = {}

    def _mk(li, slot):
        def hook(mod, inputs, output):
            d = buf.setdefault(li, {})
            if slot == 'h':                       # vnorm 의 입력 = video 잔차가 더해질 h
                d['h'] = inputs[0].detach()
            elif slot == 'v':                     # vmlp: v_final = mlp(v_pre) + v_pre
                d['v'] = (output + inputs[0]).detach()
            elif slot == 'vattn':                 # (a, w) — w 는 head 평균 (B,T,L)
                d['vw'] = output[1].detach()
            else:                                 # text/geo CA 의 ||a||/||x||
                d[slot] = (_rowmean_norm(output[0].detach())
                           / _rowmean_norm(inputs[0].detach()).clamp(min=1e-12))
        return hook

    handles = []
    for li in range(n_layers):
        vn, vca, vmlp = raw_model.video_layers[li]
        handles.append(vn.register_forward_hook(_mk(li, 'h')))
        handles.append(vmlp.register_forward_hook(_mk(li, 'v')))
        handles.append(vca.attn.register_forward_hook(_mk(li, 'vattn')))
        handles.append(raw_model.layers[li][2].attn.register_forward_hook(_mk(li, 'tresid')))
        handles.append(raw_model.layers[li][5].attn.register_forward_hook(_mk(li, 'gresid')))

    try:
        B = z.shape[0]
        ts = torch.full((B,), int(timestep), device=z.device, dtype=torch.long)
        g = torch.Generator(device='cpu').manual_seed(int(noise_seed))
        x_t = scheduler.add_noise(z, torch.randn(z.shape, generator=g).to(z.device), ts)

        pred = raw_model(x_t, ts.float(), text_emb, text_mask, geo_emb, geo_mask,
                         cond=cond, **video_kw)

        rows = [{} for _ in range(B)]
        vres, tres, gres = [], [], []
        vw0 = None
        for li in range(n_layers):
            d = buf.get(li, {})
            if 'h' in d and 'v' in d:
                gv = float(gate[li]) if gate is not None else 1.0
                r = (abs(gv) * _rowmean_norm(d['v'])) / _rowmean_norm(d['h']).clamp(min=1e-12)
                vres.append(r)
                for b in range(B):
                    rows[b][f'vresid_l{li}'] = float(r[b])
                    rows[b][f'gate_l{li}'] = gv
            if 'tresid' in d:
                tres.append(d['tresid'])
            if 'gresid' in d:
                gres.append(d['gresid'])
            if vw0 is None and 'vw' in d:
                vw0 = d['vw']

        def _put(key, stack):
            if not stack:
                return
            m = torch.stack(stack, 0).mean(0)
            for b in range(B):
                rows[b][key] = float(m[b])

        _put('vresid_mean', vres)
        _put('text_resid_mean', tres)
        _put('geo_resid_mean', gres)

        # ── attention: molmo2 text 파트(앞) vs 프레임 패치 파트(뒤) ──────────────────────
        if vw0 is not None:
            p = vw0.mean(dim=1)                                  # (B,L) traj 토큰 평균
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
            L = p.shape[-1]
            n_vid = int(video_kw['video_emb'].shape[1])
            n_txt = L - n_vid
            tm = p[:, :n_txt].sum(-1) if n_txt > 0 else torch.zeros(B, device=p.device)
            for b in range(B):
                rows[b]['vattn_text_mass'] = float(tm[b])
                rows[b]['vattn_video_mass'] = float(1.0 - tm[b])
                rows[b]['n_video_tok'] = n_vid
                rows[b]['n_video_text_tok'] = n_txt
            pv = p[:, n_txt:]
            pv = pv / pv.sum(-1, keepdim=True).clamp(min=1e-12)
            F = int(n_frames)
            if n_vid % F == 0:
                fm = pv.view(B, F, n_vid // F).sum(-1)           # (B,F) 프레임별 질량
                ent = -(fm * fm.clamp(min=1e-12).log()).sum(-1) / math.log(F)
                pk, pi = fm.max(-1)
                # 토큰 인덱스 vs attention 가중 평균 프레임의 상관 — 시간축 정렬 독해인가
                wv = vw0[:, :, n_txt:].view(B, vw0.shape[1], F, n_vid // F).sum(-1)
                wv = wv / wv.sum(-1, keepdim=True).clamp(min=1e-12)
                fidx = torch.arange(F, device=wv.device, dtype=wv.dtype)
                mf = (wv * fidx).sum(-1)                         # (B,T)
                tidx = torch.arange(wv.shape[1], device=wv.device,
                                    dtype=wv.dtype).unsqueeze(0).expand(B, -1)
                r = _pearson(mf, tidx)
                for b in range(B):
                    rows[b]['vattn_entropy_norm'] = float(ent[b])
                    rows[b]['vattn_frame_peak'] = float(pk[b] * F)
                    rows[b]['vattn_frame_peak_idx'] = int(pi[b])
                    rows[b]['vattn_frame_token_r'] = float(r[b])

        # ── 인과 검사: 스트림을 끄면 / 내용만 어긋나게 하면 예측이 얼마나 변하나 ─────────
        pn = pred.float().flatten(1).norm(dim=-1).clamp(min=1e-12)

        def _dpred(other):
            return ((other - pred).float().flatten(1).norm(dim=-1) / pn)

        d_drop = _dpred(raw_model(x_t, ts.float(), text_emb, text_mask, geo_emb, geo_mask,
                                 cond=cond))
        for b in range(B):
            rows[b]['dpred_drop'] = float(d_drop[b])

        if B > 1:
            shuf = {k: (torch.roll(v, 1, 0) if torch.is_tensor(v) else v)
                    for k, v in video_kw.items()}
            d_sh = _dpred(raw_model(x_t, ts.float(), text_emb, text_mask, geo_emb, geo_mask,
                                    cond=cond, **shuf))
            for b in range(B):
                rows[b]['dpred_shuffle'] = float(d_sh[b])

        # [new] 진짜 내용 검사. roll(1) 은 같은 scene 이웃 seg 라 거의 안 깨진다 (docstring 참조).
        # 호출부가 **다른 scene** 의 video 조건을 넘겨주면 그걸로 갈아끼워 본다.
        if video_kw_alt:
            alt = {}
            for k, v in video_kw.items():
                a = video_kw_alt.get(k)
                if torch.is_tensor(v) and torch.is_tensor(a):
                    # 다른 scene 표본이 1개뿐이어도 되게 배치축으로 펼친다. 토큰 수는 arm 고정.
                    alt[k] = a[:1].to(v.device, v.dtype).expand_as(v).contiguous()
                else:
                    alt[k] = v
            d_x = _dpred(raw_model(x_t, ts.float(), text_emb, text_mask, geo_emb, geo_mask,
                                   cond=cond, **alt))
            for b in range(B):
                rows[b]['dpred_xscene'] = float(d_x[b])

        for b in range(B):
            rows[b]['probe_timestep'] = int(timestep)
        return rows
    finally:
        for h in handles:
            h.remove()
