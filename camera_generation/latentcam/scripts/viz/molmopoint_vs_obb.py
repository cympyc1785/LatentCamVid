"""MolmoPoint-Vid-4B 의 video pointing 을 OBB 투영과 대조한다 (D166).

  python scripts/viz/molmopoint_vs_obb.py --stage probe --gpu 4      # 배관 확인 (1 anchor)
  python scripts/viz/molmopoint_vs_obb.py --stage eval  --gpu 4      # 8씬 29 anchor 대조

왜 (2026-09-12): D164/D165 가 **현재 체크포인트(Molmo2-4B)에 point head 가 없다**는 전제
위에서 국소화 정보를 역공학했다 — 좌표가 숫자 BPE 토큰으로 나오고, `patch_logits` /
`subpatch_logits` / `location_logits` 가 전부 없었다. 그래서 attention 맵을 뒤져
"어느 층·어느 head·어느 위치" 를 찾는 일을 했고, 결론은 이랬다:

    prefill argmax 절대거리   16 anchor 중 15개가 고정셀 통제 이하, LOO 174.0px
    시간 marginal 쌍비교      LOO **79.0%** (이종 84.2% / 동종 67.6%), chance 50%
    학습된 probe (34예제)     쌍비교 59.2%, notext 통제 50.0%
    생성좌표 <-> OBB          dyn median **18.6px** / 전체 30.6px

`allenai/MolmoPoint-Vid-4B` 는 그 head 를 **지도학습으로 갖고 있다**. config 에
`patch_token_id` / `subpatch_token_id` / `location_token_id` / `no_more_points_class` /
`embed_selected_vit_patch` 가 있고 README 는 "a dot-product based attention mechanism for
pointing" 이라고 적는다 — 즉 우리가 `W_q`/`W_k` 를 34예제로 배우려던 그 투영이 이미
대규모로 학습되어 있다.

**전처리 규약이 같다** (직접 확인): `size 378x378` / `patch_size 14` / `pooling_size [3,3]`
/ `default_to_square True` -> `27/3 = 9`, 즉 프레임당 9x9 = 81 patch 로 우리 격자와 동일하고
백본도 36층 / hidden 2560 / 32 head 로 같다. 그래서 `C.load_frames` 로 **완전히 같은
49프레임**을 먹여 기존 기준선과 직접 비교할 수 있다.

한 가지 새로 있는 것: `frame_start_token_id` / `frame_end_token_id`. 현재 체크포인트에는
없다. D165 에서 "prefill 이 프레임마다 못 따라가는 건 프레임 질의가 없어서" 라고 결론냈던
그 구조적 한계가 여기서는 풀릴 수 있다.

## API 가 다르다

현재 `Runner` 는 `proc(text=..., videos=...)` 를 쓰지만 MolmoPoint 는:

    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_tensors='pt', return_dict=True,
                                           padding=True, return_pointing_metadata=True)
    metadata = inputs.pop('metadata')
    out = model.generate(**inputs,
                         logits_processor=model.build_logit_processor_from_inputs(inputs), ...)
    txt = processor.post_process_image_text_to_text(gen, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)[0]
    pts = model.extract_video_points(txt, metadata['token_pooling'],
                                     metadata['subpatch_mapping'],
                                     metadata['timestamps'], metadata['video_size'])
    # pts = [[object_id, timestamp, x, y], ...]  object_id 가 프레임을 건너 같은 물체를 잇는다

`--stage probe` 가 이 배관을 먼저 확인한다 — `video=` 가 numpy 프레임을 받는지, timestamps
가 어떤 단위인지 (그게 프레임 인덱스 매핑을 정한다), 생성 원문이 어떻게 생겼는지.
**매핑을 눈으로 확인하기 전에는 eval 숫자를 믿지 않는다** (D165 에서 지표를 두 번 잘못
골랐다).

env: `latentcam` (transformers 4.57.6 >= 4.57.1 요구치).  GPU 1장.
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import sys

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import CORPUS                             # noqa: E402
from molmo2_vs_obb import project, scene_anchors                 # noqa: E402
import cache_molmo2_embeddings as C                              # noqa: E402

CKPT = ('/data1/cympyc1785/LatentCamVid/camera_generation/tools/molmo2/'
        'checkpoints/MolmoPoint-Vid-4B')
SCENES = ('camel', 'car-roundabout', 'basketball-four', 'couple-walk',
          'golf', 'cows', 'couple-rocks', 'fashion-walk')


def cell_of(u, v, side, W, H):
    """픽셀 -> 9x9 셀 (r, c)."""
    return (int(np.clip(v * side / H, 0, side - 1)),
            int(np.clip(u * side / W, 0, side - 1)))


def load_model(ckpt, dtype=torch.bfloat16):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import time
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, trust_remote_code=True, dtype=dtype, device_map='auto')
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True, padding_side='left')
    model.eval()
    print(f'[load] MolmoPoint-Vid  {time.time() - t0:.1f}s  '
          f'{type(model).__name__}  dtype {dtype}', flush=True)
    return model, proc


def video_arg(video, kind, tmp_dir, tag):
    """`video=` 에 넣을 값. 무엇이 받아들여지는지 모르므로 세 형태를 고를 수 있게 둔다."""
    if kind == 'array':
        return video
    if kind == 'pil':
        from PIL import Image
        return [Image.fromarray(f) for f in video]
    if kind == 'mp4':
        import imageio.v2 as iio
        makedirs(tmp_dir, exist_ok=True)
        fp = osp.join(tmp_dir, f'{tag}.mp4')
        wr = iio.get_writer(fp, fps=8, codec='libx264', quality=9, macro_block_size=1)
        for f in video:
            wr.append_data(f)
        wr.close()
        return fp
    raise SystemExit(f'--video_input {kind} 를 모른다 (array|pil|mp4)')


def make_prompt(args, txt):
    """학습 문구를 만든다. `--verb` 를 주면 예전처럼 `{verb} {label}` (대조용)."""
    if args.verb:
        return f'{args.verb} {txt}'
    return args.prompt_tmpl.format(label=txt, fps=f'{args.fps:g}')


@torch.inference_mode()
def point_video(model, proc, video, text, args, tag='v'):
    """(points, raw_text, metadata). points = [[object_id, timestamp, x, y], ...]."""
    va = video_arg(video, args.video_input, osp.join(args.out_dir, 'tmpvid'), tag)
    # `chat_template.jinja`: content['style'] 가 DEMO_STYLES **밖**이면 `"{style}: "` 접두어가
    # 붙는다. 추적 스타일 중 `_with_occlusion` / `video_single_point_track_per_frame` 이
    # 밖이라 이 경로로만 태스크가 전달된다. 문구만 바꾸면 (style 없이) 단일 시점
    # `video_point` 로 떨어진다 (실측: 10조합 전부 `<points>` 점 1개, 바이트 동일).
    tcontent = dict(type='text', text=text)
    if getattr(args, 'style', ''):
        tcontent['style'] = args.style
    msgs = [{'role': 'user', 'content': [tcontent, dict(type='video', video=va)]}]
    # `do_sample_frames=False` + 명시적 `video_metadata` 가 **49프레임을 그대로** 넣는다
    # (`video_processing_molmo2._decode_and_sample_videos`: array + no-sample 면 두 분기를
    # 모두 건너뛴다). 현재 `Runner._batch` 와 같은 방식이라 기준선과 입력이 동일해진다.
    # 안 주면 config 기본값 `do_sample_frames=True` 가 걸려 `sampling_fps 2` 로 솎아내고,
    # fps 가 없으면 assert 로 죽는다 (실측).
    T0 = int(np.asarray(video).shape[0]) if args.video_input != 'mp4' else args.num_frames
    # `width`/`height` 가 반드시 있어야 한다 — `processing_molmo2.py:326` 이
    # `video_size = (vd_metadata.width, vd_metadata.height)` 로 읽고, 그 값이 None 이면
    # `extract_video_points` 가 `p_x/mapping.shape[2] * video_size[0]` 에서 죽는다 (실측).
    hh, ww = (np.asarray(video).shape[1:3] if args.video_input != 'mp4'
              else (args.frame_h, args.frame_w))
    vmeta = [dict(fps=args.fps, total_num_frames=T0, duration=T0 / args.fps,
                  width=int(ww), height=int(hh),
                  frames_indices=list(range(T0)))]
    kw = {} if args.video_input == 'mp4' else dict(
        do_sample_frames=args.do_sample_frames, video_metadata=vmeta)
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors='pt', return_dict=True, padding=True,
                                      return_pointing_metadata=True, **kw)
    meta = inputs.pop('metadata')
    dev = next(model.parameters()).device
    inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.generate(**inputs,
                             logits_processor=model.build_logit_processor_from_inputs(inputs),
                             max_new_tokens=args.max_new_tokens)
    gen = out[:, inputs['input_ids'].size(1):]
    raw = proc.post_process_image_text_to_text(
        gen, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    pts = model.extract_video_points(raw, meta['token_pooling'], meta['subpatch_mapping'],
                                     meta['timestamps'], meta['video_size'])
    return pts, raw, meta, gen[0].tolist(), inputs


def stage_sweep(args):
    """(프롬프트 x fps) 조합을 1 anchor 에서 훑어 **`<tracks>` 가 나오는 설정**을 찾는다.

    왜: `Track {label} in 5 FPS.` 가 `<points>` 태그에 점 1개만 냈고 "in 5 FPS" 가 라벨로
    흡수됐다. 모델의 `video_preprocessor_config` 는 `sampling_fps 2` / `max_fps 2.0` 이라
    5 FPS 는 훈련 분포 밖이다. 학습 템플릿은 두 갈래다 (`data_formatter.py:546,556`):
        fps==2 기본형   "Track {label}."
        fps 명시형      "Track {label} in {fps} FPS."
    한 번에 훑어 어느 조합이 다중 타임스탬프를 내는지 본다. 추측을 한 번에 하나씩
    시험하면 anchor 당 모델 로드가 반복돼 낭비다.
    """
    model, proc = load_model(args.ckpt)
    scene = args.scenes.split(',')[0]
    anch = scene_anchors(args.root, scene, args.max_per_kind)
    a, key, txt, lab, tw = anch[0]
    video = C.load_frames(args.root, scene, 0, args.num_frames)
    tmpls = ['Track {label}.',
             'Track {label} in {fps} FPS.',
             'Track the {label} in {fps} FPS',
             "Track all instances of '{label}' in this video, sampling at {fps} frames "
             "per second. Show the position coordinates at each timestamp.",
             'Point to {label}.']
    fpss = [float(x) for x in args.sweep_fps.split(',')]
    print(f'\n[sweep] {scene} {a} ({lab})  {txt!r}\n')
    rows = []
    for fps in fpss:
        args.fps = fps
        for ti, tm in enumerate(tmpls):
            args.prompt_tmpl = tm
            pr = make_prompt(args, txt)
            try:
                pts, raw, meta, _gid, _in = point_video(model, proc, video, pr, args, tag=a)
            except Exception as e:
                print(f'  fps{fps:g} T{ti}  실패 {type(e).__name__}: {str(e)[:70]}')
                continue
            arr = np.array(pts) if len(pts) else np.zeros((0, 4))
            nts = len(set(arr[:, 1].tolist())) if len(arr) else 0
            tag = '<tracks>' if '<tracks' in raw else ('<points>' if '<points' in raw else '?')
            rows.append((fps, ti, len(arr), nts, tag))
            print(f'  fps{fps:g} T{ti}  점 {len(arr):3d}  타임스탬프 {nts:3d}  {tag:9s}  '
                  f'{pr[:52]!r}')
            if len(arr) and nts > 1:
                print(f'        t 범위 {arr[:, 1].min():.2f}~{arr[:, 1].max():.2f}  '
                      f'obj {sorted(set(arr[:, 0].tolist()))}')
            print(f'        raw: {raw[:150]}')
    best = [r for r in rows if r[3] > 1]
    print(f'\n  다중 타임스탬프가 나온 조합 {len(best)}개: '
          f'{[(f"fps{r[0]:g}", f"T{r[1]}", f"{r[3]}ts") for r in best]}')



# ------------------------------------------------- <PATCH> head 분포 (지도학습 readout)

def hook_point_predictor(model, store):
    """`PointPredictor.forward` 를 감싸 `patch_logits` 를 스텝마다 모은다.

    훅이 필요한 이유: 최종 `logits` 는 `concat([lm_head, argmax_patch_logits, subpatch,
    location])` 인데 `argmax_patch_logits` 는 **argmax 자리만 채우고 나머지는 -1e5** 다
    (`modeling_molmo_point.py:1799` 의 two-stage 해킹). 전체 3969 분포는
    `PointPredictor.forward` 의 반환값에만 있다.

    prefill 에서는 `patch_logits is None` 이다 — 그 경로는 키(`patch_k`)만 만들고 캐시한다
    ("Predict patch locations, only done after pre-filling"). 그래서 store 의 첫 항목은
    반드시 None 이고, 그게 곧 **prefill 에서 위치가 안 나온다는 아키텍처 수준의 증거**다.
    """
    pp = model.model.point_predictor
    orig = pp.forward

    def wrapped(*a, **k):
        out = orig(*a, **k)
        cpu = lambda t: None if t is None else t.detach().float().cpu()
        store.append((cpu(out[0]), cpu(out[1]), cpu(out[2])))
        return out

    pp.forward = wrapped
    return lambda: setattr(pp, 'forward', orig)


@torch.inference_mode()
def point_with_logits(model, proc, video, text, args, tag='v'):
    """(points, raw, meta, [patch_logits (3969,) per step])."""
    store = []
    restore = hook_point_predictor(model, store)
    try:
        pts, raw, meta, gid, _in = point_video(model, proc, video, text, args, tag=tag)
    finally:
        restore()
    pl = [x[0][0, -1] for x in store if x[0] is not None]
    n_prefill_none = sum(1 for x in store if x[0] is None)
    # **스텝 정렬**: store[0] = prefill (patch_logits None), store[i+1] = 생성 토큰 i 를
    # 만든 forward. 그래서 `pl[i]` 가 생성 토큰 i 에 대응한다.
    #
    # 왜 정렬이 필요한가: `patch_logits` 는 **모든** decode 스텝에서 계산된다 (질의가 그
    # 스텝의 hidden 이므로). 첫 스텝은 `<points coords="` 태그를 만드는 자리라 그 분포는
    # 위치 선택과 무관하다 — 실측으로 camel dyn_0/dyn_1/fence 가 전부 argmax 셀 30 으로
    # 같게 나왔다. 실제로 patch 를 고른 스텝만 골라야 한다.
    bounds = model.model.build_token_bounds(_in['video_token_pooling'])
    patch_steps = [i for i, t in enumerate(gid)
                   if bounds.patch_start <= t < bounds.patch_end_without_no_more_points]
    return pts, raw, meta, pl, n_prefill_none, patch_steps, gid, bounds


def stage_patchviz(args):
    """`patch_logits` 3969 분포를 (a) 쌍비교 지표로 재고 (b) 프레임 타일로 그린다.

    (a) 는 D165 의 attention 시간 marginal 과 **같은 자**다 (씬 내 쌍, 양방향, 동점 0.5,
        chance 정확히 50%). 비교 대상: 비지도 attention LOO all 79.0% / 이종 84.2% /
        동종 67.6%.
    (b) 는 `3969 = 49 x 81` 을 `(49, 9, 9)` 로 reshape 만 하면 프레임별 맵이 된다 —
        인덱스가 `frame*81 + cell` 이므로.
    """
    from PIL import Image, ImageDraw
    import imageio.v2 as iio
    from molmo2_attn_video import colorize
    model, proc = load_model(args.ckpt)
    side = 9
    allb = {k: [0.0, 0] for k in ('all', 'same', 'diff')}
    allb0 = {k: [0.0, 0] for k in ('all', 'same', 'diff')}
    rows, prefill_none_total, steps_total = [], 0, 0
    viz_scenes = [x.strip() for x in args.viz_scenes.split(',') if x.strip()]

    for scene in args.scenes.split(','):
        d = osp.join(args.root, scene, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            continue
        p = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p['extrinsics'], p['intrinsics']
        anch = scene_anchors(args.root, scene, args.max_per_kind)
        if len(anch) < 2:
            continue
        video = C.load_frames(args.root, scene, 0, args.num_frames)
        T, H_, W_ = video.shape[0], video.shape[1], video.shape[2]
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        sx, sy = W_ / pw, H_ / ph
        M, M0, cen, lbls, txts = {}, {}, {}, {}, {}
        print(f'\n### {scene}  anchors {len(anch)}', flush=True)
        for a, key, txt, lab, tw in anch:
            uv, _z = project(tw, E, K)
            uv = uv * np.array([sx, sy])
            ok = [f for f in range(min(T, uv.shape[0]))
                  if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W_ and 0 <= uv[f, 1] < H_]
            if len(ok) < args.min_frames:
                continue
            pr = make_prompt(args, txt)
            pts, raw, meta, pl, nnone, psteps, gid, bounds = point_with_logits(
                model, proc, video, pr, args, tag=a)
            prefill_none_total += nnone
            steps_total += len(pl)
            if not pl or not psteps:
                print(f'  [{a:7s}] patch 스텝 0개 (생성 {len(pl)} 스텝)')
                continue
            k = psteps[0]                       # 첫 물체의 patch 선택 스텝
            prob = torch.softmax(pl[k], -1).numpy()[:T * side * side]
            g = prob.reshape(T, side * side)
            M[a] = g.sum(0)                       # 시간 marginal (81,)
            M0[a] = g[0]                          # 프레임 0 슬라이스 (81,)
            cen[a] = (float(np.median(uv[ok, 0])), float(np.median(uv[ok, 1])))
            lbls[a], txts[a] = lab, txt
            am = int(prob.argmax())
            rows.append(dict(scene=scene, anchor=a, label=lab, n_steps=len(pl),
                             patch_steps=psteps, emitted_patch=int(gid[k]) - bounds.patch_start,
                             argmax_frame=am // (side * side), argmax_cell=am % (side * side),
                             f0_mass=float(g[0].sum()), prefill_none=nnone))
            print(f'  [{a:7s} {lab:12s}] 스텝 {len(pl)} (patch {psteps})  None {nnone}  '
                  f'argmax 프레임 {am // (side * side):2d} 셀 {am % (side * side):2d}  '
                  f'f0 질량 {g[0].sum() * 100:5.1f}%  {txt[:26]}', flush=True)

        # ── 쌍비교 (씬 내)
        ids = sorted(M)
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                ci = cell_of(*cen[i], side, W_, H_)
                cj = cell_of(*cen[j], side, W_, H_)
                if ci == cj:
                    continue
                bk = 'same' if lbls[i] == lbls[j] else 'diff'
                for buck, src in ((allb, M), (allb0, M0)):
                    a_, b_ = (src[i][ci[0] * side + ci[1]], src[i][cj[0] * side + cj[1]])
                    hit = 1.0 if a_ > b_ else (0.5 if a_ == b_ else 0.0)
                    for kk in ('all', bk):
                        buck[kk][0] += hit
                        buck[kk][1] += 1

        # ── 시각화
        if scene.split('/')[-1] in viz_scenes and ids:
            a0 = ids[0]
            pr = make_prompt(args, txts[a0])
            _p, _r, _m, pl, _n, ps, _g, _b = point_with_logits(model, proc, video, pr,
                                                               args, tag=a0)
            prob = torch.softmax(pl[ps[0]], -1).numpy()[:T * side * side].reshape(T, side, side)
            cw = 160
            chh = cw * H_ // W_
            ncol, nrow = 7, int(np.ceil(T / 7))
            cv = np.zeros((nrow * chh, ncol * cw, 3), np.uint8)
            vmax = float(prob.max())
            for f in range(T):
                fr = np.asarray(Image.fromarray(video[f]).resize((cw, chh), Image.BILINEAR))
                m = prob[f] / max(vmax, 1e-12)          # **전역 최대로 정규화** — 프레임 간
                up = np.asarray(Image.fromarray(        #   질량 차이를 보여야 하므로
                    (np.clip(m, 0, 1) * 255).astype(np.uint8)
                ).resize((cw, chh), Image.BILINEAR)) / 255.0
                al = (up[..., None] ** 0.6) * args.alpha
                o = fr.astype(np.float32) / 255.0 * (1 - al) + colorize(up) * al
                u, v = cen[a0]
                cx, cy = int(u * cw / W_), int(v * chh / H_)
                if 0 <= cx < cw and 0 <= cy < chh:
                    o[max(0, cy - 1):cy + 1, max(0, cx - 5):cx + 5] = [0, 1, 0]
                    o[max(0, cy - 5):cy + 5, max(0, cx - 1):cx + 1] = [0, 1, 0]
                rr, cc = divmod(f, ncol)
                cv[rr * chh:(rr + 1) * chh, cc * cw:(cc + 1) * cw] = \
                    (np.clip(o, 0, 1) * 255).astype(np.uint8)
            im = Image.fromarray(cv)
            dr = ImageDraw.Draw(im)
            for f in range(T):
                rr, cc = divmod(f, ncol)
                dr.rectangle([cc * cw + 1, rr * chh + 1, cc * cw + 62, rr * chh + 13],
                             fill=(0, 0, 0))
                dr.text((cc * cw + 3, rr * chh + 2),
                        f'f{f:02d} {prob[f].sum() * 100:.1f}%', fill=(255, 255, 255))
            fp = osp.join(args.out_dir, f'patchlogits_{scene.split("/")[-1]}_{a0}.png')
            im.save(fp)
            print(f'  [save] {fp}   (칸 라벨 = 그 프레임의 확률 질량, 초록=OBB)')

    print(f'\n{"=" * 96}')
    print(f'<PATCH> head 분포 — 쌍비교 (D165 attention 과 같은 지표, chance 50%)')
    print(f'  prefill 에서 patch_logits=None 인 forward {prefill_none_total}회 / '
          f'decode 스텝 {steps_total}회  <- 아키텍처상 prefill 은 키만 만든다')
    print(f'{"=" * 96}')
    for nm, buck in (('시간 marginal', allb), ('프레임0 슬라이스', allb0)):
        print(f'\n  [{nm}]')
        for kk in ('all', 'same', 'diff'):
            c, t = buck[kk]
            if not t:
                continue
            pct = c / t * 100
            se = float(np.sqrt(0.25 / t) * 100)
            print(f'    {kk:5s} {pct:5.1f}%  (n={t:4d}, 1 s.e. {se:.2f}%p, '
                  f'{(pct - 50) / se:+.1f} se)')
    print(f'\n  대조 — D165 비지도 attention 시간 marginal 씬 LOO: '
          f'all 79.0% / same 67.6% / diff 84.2%')
    makedirs(args.out_dir, exist_ok=True)
    json.dump(dict(rows=rows,
                   paired={nm: {k: dict(pct=(v[0] / v[1] * 100 if v[1] else None), n=v[1])
                                for k, v in b.items()}
                           for nm, b in (('marginal', allb), ('frame0', allb0))},
                   prefill_none=prefill_none_total, decode_steps=steps_total),
              open(osp.join(args.out_dir, f'patchlogits_{args.tag}.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'[out] {osp.join(args.out_dir, f"patchlogits_{args.tag}.json")}')



def stage_stylesweep(args):
    """`style` 키를 바꿔 가며 **다중 타임스탬프가 나오는 스타일**을 찾는다.

    근거: `chat_template.jinja` 가 `content['style']` 를 읽어 DEMO_STYLES 밖이면
    `"{style}: "` 를 앞에 붙인다. 문구(`Track ... in N FPS.`)만으로는 태스크가 전달되지
    않았다 (프롬프트 5종 x fps 2종 = 10조합이 전부 `<points>` 점 1개, 출력 바이트 동일).
    """
    model, proc = load_model(args.ckpt)
    scene = args.scenes.split(',')[0]
    anch = scene_anchors(args.root, scene, args.max_per_kind)
    a, key, txt, lab, tw = anch[0]
    video = C.load_frames(args.root, scene, 0, args.num_frames)
    styles = [x.strip() for x in args.sweep_styles.split(',')]
    print(f'\n[stylesweep] {scene} {a} ({lab})  {txt!r}  fps {args.fps:g}\n')
    hit = []
    for st in styles:
        args.style = '' if st == 'NONE' else st
        pr = make_prompt(args, txt)
        try:
            pts, raw, meta, _gid, _in = point_video(model, proc, video, pr, args, tag=a)
        except Exception as e:
            print(f'  {st:46s} 실패 {type(e).__name__}: {str(e)[:60]}')
            continue
        arr = np.array(pts) if len(pts) else np.zeros((0, 4))
        nts = len(set(arr[:, 1].tolist())) if len(arr) else 0
        tag = ('<tracks>' if '<tracks' in raw else
               ('<points>' if '<points' in raw else '?'))
        print(f'  {st:46s} 점 {len(arr):3d}  ts {nts:3d}  {tag:9s}')
        print(f'     raw: {raw[:170]}')
        if nts > 1:
            hit.append((st, len(arr), nts))
            print(f'     t 범위 {arr[:, 1].min():.2f}~{arr[:, 1].max():.2f}  '
                  f'obj {sorted(set(arr[:, 0].tolist()))[:6]}')
    print(f'\n  다중 타임스탬프 스타일 {len(hit)}개: {hit}')



def stage_probe(args):
    """배관 확인 — 1 씬 1 anchor. **eval 전에 반드시 통과해야 한다.**"""
    model, proc = load_model(args.ckpt)
    scene = args.scenes.split(',')[0]
    d = osp.join(args.root, scene, 'da3')
    p = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
    E, K = p['extrinsics'], p['intrinsics']
    anch = scene_anchors(args.root, scene, args.max_per_kind)
    a, key, txt, lab, tw = anch[0]
    video = C.load_frames(args.root, scene, 0, args.num_frames)
    print(f'\n[probe] {scene}  anchor {a} ({lab})  frames {video.shape}  '
          f'video_input={args.video_input}')
    print(f'        prompt = {make_prompt(args, txt)!r}')
    pts, raw, meta, _gid, _in = point_video(model, proc, video, make_prompt(args, txt), args, tag=a)
    print(f'\n-- metadata 키: {list(meta)}')
    for k in meta:
        v = meta[k]
        s = (f'shape {tuple(v.shape)}' if hasattr(v, 'shape')
             else (f'len {len(v)}' if hasattr(v, "__len__") else type(v).__name__))
        print(f'   {k:20s} {s}   {str(v)[:120]}')
    print(f'\n-- 생성 원문 ({len(raw)} chars):\n{raw[:600]}')
    arr = np.array(pts) if len(pts) else np.zeros((0, 4))
    print(f'\n-- extract_video_points -> {arr.shape}  [object_id, timestamp, x, y]')
    print(arr[:24])
    if len(arr):
        print(f'   object_id 종류 {sorted(set(arr[:, 0].tolist()))}')
        print(f'   timestamp 범위 {arr[:, 1].min():.3f} ~ {arr[:, 1].max():.3f}  '
              f'(개수 {len(set(arr[:, 1].tolist()))})')
        print(f'   x 범위 {arr[:, 2].min():.1f}~{arr[:, 2].max():.1f}  '
              f'y 범위 {arr[:, 3].min():.1f}~{arr[:, 3].max():.1f}')
        uv, _z = project(tw, E, K)
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        print(f'   OBB 투영 픽셀 규약 {pw:.0f}x{ph:.0f}   영상 {video.shape[2]}x{video.shape[1]}')
        print(f'   OBB frame0 = ({uv[0, 0]:.1f}, {uv[0, 1]:.1f})')
        print(f'\n   ** timestamp -> 프레임 인덱스 매핑을 눈으로 확인할 것. '
              f'x,y 규약(픽셀 vs 정규화)도 위 범위로 판정한다. **')
    else:
        print('   점이 0개 — 프롬프트 동사 또는 video_input 형태를 바꿔 볼 것')


def stage_eval(args):
    model, proc = load_model(args.ckpt)
    rows = []
    for scene in args.scenes.split(','):
        d = osp.join(args.root, scene, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            print(f'[skip] {scene}')
            continue
        p = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p['extrinsics'], p['intrinsics']
        anch = scene_anchors(args.root, scene, args.max_per_kind)
        if not anch:
            continue
        video = C.load_frames(args.root, scene, 0, args.num_frames)
        T, H_, W_ = video.shape[0], video.shape[1], video.shape[2]
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        sx, sy = W_ / pw, H_ / ph
        print(f'\n### {scene}  anchors {len(anch)}  T {T}  {W_}x{H_}', flush=True)
        for a, key, txt, lab, tw in anch:
            uv, z = project(tw, E, K)
            uv = uv * np.array([sx, sy])
            try:
                pts, raw, meta, _gid, _in = point_video(model, proc, video,
                                             make_prompt(args, txt), args, tag=f'{a}')
            except Exception as e:
                print(f'  [{a:7s}] 실패 {type(e).__name__}: {e}')
                continue
            if not len(pts):
                print(f'  [{a:7s}] 점 0개  {txt[:40]}')
                continue
            arr = np.array(pts, dtype=np.float64)
            # timestamp -> 프레임. `--ts_mode` 로 규약을 고른다 (probe 로 확인한 값을 쓴다)
            if args.ts_mode == 'index':
                fr = np.clip(np.rint(arr[:, 1]), 0, T - 1).astype(int)
            else:                                     # 'sec': 초 * fps
                fr = np.clip(np.rint(arr[:, 1] * args.fps), 0, T - 1).astype(int)
            # 가장 많이 등장한 object_id 하나만 (주 물체)
            ids, cnt = np.unique(arr[:, 0], return_counts=True)
            oid = ids[int(np.argmax(cnt))]
            sel = arr[:, 0] == oid
            dd, fs = [], []
            for i in np.nonzero(sel)[0]:
                f = int(fr[i])
                if np.isnan(uv[f, 0]):
                    continue
                dd.append(np.hypot(arr[i, 2] - uv[f, 0], arr[i, 3] - uv[f, 1]))
                fs.append(f)
            if not dd:
                print(f'  [{a:7s}] 유효 프레임 0')
                continue
            dd = np.array(dd)
            rows.append(dict(scene=scene, anchor=a, label=lab, text=txt,
                             kind='dyn' if a.startswith('dyn') else 'stat',
                             n=len(dd), n_obj=len(ids), obj_id=float(oid),
                             d0=float(dd[0]), dmean=float(dd.mean()),
                             dmed=float(np.median(dd)),
                             depth_med=float(np.nanmedian(z[fs]))))
            print(f'  [{a:7s} {lab:10s}] obj {len(ids):2d}개  n{len(dd):3d}  '
                  f'Δf0 {dd[0]:6.1f}px  Δmed {np.median(dd):6.1f}px  {txt[:32]}',
                  flush=True)

    if not rows:
        raise SystemExit('대조할 게 없다')
    print(f'\n{"=" * 92}\nMolmoPoint-Vid-4B vs OBB 투영   (n={len(rows)} anchor)\n{"=" * 92}')
    print('기준선 — 현재 Molmo2-4B 숫자 토큰 생성 (D164 `molmo2_vs_obb_fps5.json`):')
    print('   dyn Δmed median 21.2px   전체 30.6px   <50px 비율은 아래와 같이 비교')
    for kind in ('dyn', 'stat', None):
        sub = [x for x in rows if kind is None or x['kind'] == kind]
        if not sub:
            continue
        dm = np.array([x['dmed'] for x in sub])
        nm = {'dyn': 'dyn (움직이는 피사체)', 'stat': 'stat (정적 배경)'}.get(kind, '전체')
        print(f'{nm:22s} n{len(sub):3d}  Δmed med {np.median(dm):6.1f}px  '
              f'p90 {np.percentile(dm, 90):6.1f}px  '
              f'<50px {np.mean(dm < 50) * 100:3.0f}%  <100px {np.mean(dm < 100) * 100:3.0f}%')
    makedirs(args.out_dir, exist_ok=True)
    fp = osp.join(args.out_dir, f'molmopoint_vs_obb_{args.tag}.json')
    json.dump(rows, open(fp, 'w'), ensure_ascii=False, indent=1)
    print(f'\n[out] {fp}')


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    for st in args.stage.split(','):
        st = st.strip()
        if st == 'probe':
            stage_probe(args)
        elif st == 'sweep':
            stage_sweep(args)
        elif st == 'stylesweep':
            stage_stylesweep(args)
        elif st == 'patchviz':
            stage_patchviz(args)
        elif st == 'eval':
            stage_eval(args)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--stage', default='probe')
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scenes', default=','.join('vista4d/' + s for s in SCENES))
    p.add_argument('--max_per_kind', type=int, default=2)
    p.add_argument('--num_frames', type=int, default=49)
    p.add_argument('--min_frames', type=int, default=5)   # OBB 유효 프레임 하한
    # `olmo/preprocessing/data_formatter.py:556` GENERAL_PROMPTS_V1
    # ["video_point_track_per_frame"] 의 학습 문구. 이 스타일은 `DEMO_STYLES` 에 있어
    # 스타일 접두어가 붙지 않으므로 문구만 그대로 주면 된다. `Point to` 는 단일 시점
    # pointing 태스크라 점이 1개만 나온다 (실측).
    p.add_argument('--prompt_tmpl', default='Track {label} in {fps} FPS.')
    p.add_argument('--verb', default='')     # 비면 --prompt_tmpl 을 쓴다 (대조용)
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--video_input', default='array', help='array|pil|mp4')
    p.add_argument('--do_sample_frames', action='store_true', default=False)
    p.add_argument('--frame_w', type=int, default=640)   # mp4 입력일 때만
    p.add_argument('--frame_h', type=int, default=360)
    p.add_argument('--ts_mode', default='sec', help='sec (초*fps) | index (프레임 번호)')
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--out_dir', default='/data1/cympyc1785/LatentCamVid/tmp/d166_molmopoint')
    p.add_argument('--sweep_fps', default='2,5')
    p.add_argument('--viz_scenes', default='camel,basketball-four')
    p.add_argument('--alpha', type=float, default=0.65)
    p.add_argument('--style', default='')
    p.add_argument('--sweep_styles', default=','.join([
        'NONE',
        'video_point_track_per_frame',
        'video_point_track_per_frame_with_occlusion',
        'video_point_track_all_frames_with_occlusion',
        'video_point_track_all_frames',
        'video_single_point_track_per_frame',
        'video_point_track_start_end',
        'video_point_ground_start_end']))
    p.add_argument('--tag', default='fps5')
    p.add_argument('--gpu', default='4')
    main(p.parse_args())
