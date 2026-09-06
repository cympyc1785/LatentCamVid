"""vista d128 코퍼스(target 궤적 + 캡션) → CLaTr 학습 데이터 레이아웃으로 변환한다.

WHY. 지금 쓰는 CLaTr 체크포인트는 E.T./ArtTraj 로 학습된 것이고 standardization config 도
`num_cams: 120` 이다. 우리 궤적은 49프레임에 분포도 다르다. 그래서 CLaTr 이 내는 caption 지표는
"우리 코퍼스에서 텍스트와 궤적이 맞는가"를 재는 게 아니라 **남의 코퍼스 게이지로 우리 걸 재는**
것이다. 이 스크립트는 그 게이지를 우리 코퍼스로 다시 만들기 위한 첫 단계다.

CLaTr 이 실제로 읽는 것 (경로는 전부 `<out>` 기준):
    <split>/<name>_transforms_cleaning.json   frames[].transform_matrix = **c2w**, OpenGL
                                              (trajectory_dataset.py:146; frame0 로 재앵커됨)
    <split>/<name>_caption.json               {"Concise Interaction": "..."}   (key 는 config.yaml)
    seq/<split>/<name>_caption.npy            CLIP token 시퀀스 (L,512)
    token/<split>/<name>_caption.npy          CLIP EOT 임베딩 (512,)
    <split>_valid.txt                         name 목록 (줄바꿈 구분)

규약. `da3/target_poses.npz` 의 extrinsics 는 OpenCV **w2c** 다 (dataset_dl3dv.py:1060).
CLaTr 은 c2w 를 원하고, E.T. 계열 데이터는 OpenGL 축이다. 그래서 기존
`main/prepare_clatr_data.py` 와 **같은 두 단계**를 밟는다:
    c2w = inv(w2c);  c2w[:, :3, 1:3] *= -1      # OpenCV(+Y down/+Z fwd) → OpenGL(+Y up/-Z fwd)
`|c2w| >= 1e2` 인 궤적은 발산으로 보고 버린다(원본과 같은 컷). 버린 건 error.txt 에 남긴다.

split 은 원본의 95/5 랜덤이 아니라 **코퍼스 자신의** `seg_list_vista4d_{train,test}.txt` 를 쓴다.
학습 arm 들과 test entry 가 어긋나면 지표를 서로 대조할 수 없기 때문이다.

사용 예시:
    # 1) JSON + split 목록 (CPU)
    python scripts/prepare_clatr_vista.py --stage json
    # 2) CLIP feature (GPU 1장)
    CUDA_VISIBLE_DEVICES=2 python scripts/prepare_clatr_vista.py --stage clip
    # 3) 한 번에 + 표준화 통계 출력
    CUDA_VISIBLE_DEVICES=2 python scripts/prepare_clatr_vista.py --stage all --print_stats
"""
from argparse import ArgumentParser
from json import load as json_load, dump as json_dump
from os import path, makedirs
from sys import path as sys_path

import numpy as np
import torch

REPO = path.dirname(path.dirname(path.abspath(__file__)))       # .../latentcam
CLATR = path.join(REPO, 'main', 'evaluate', 'CLaTr')
sys_path.insert(0, path.join(REPO, 'main'))
sys_path.insert(0, CLATR)

CORPUS_DEFAULT = '/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d128'
OUT_DEFAULT = '/data1/cympyc1785/data/Vista4D-Eval-Data/clatr_vista_d128'
CAPTION_KEY = 'Concise Interaction'      # CLaTr configs/config.yaml 의 `key`
DIVERGE_ABS = 1e2                        # prepare_clatr_data.py 와 같은 컷


# ----------------------------------------------------------------------------------- #
# stage: json
# ----------------------------------------------------------------------------------- #

def _read_list(p):
    with open(p) as f:
        return [x.strip() for x in f if x.strip()]


def _c2w_opengl(w2c):
    """(T,4,4) OpenCV w2c → (T,4,4) OpenGL c2w. 발산하면 None."""
    c2w = np.linalg.inv(w2c.astype(np.float64))
    if not np.isfinite(c2w).all() or (np.abs(c2w) >= DIVERGE_ABS).any():
        return None
    c2w[:, :3, 1:3] *= -1.0
    return c2w


def stage_json(args):
    scene_cache = {}                     # scene → (poses dict, prompts dict, K)
    counts, errors, lengths = {}, [], []
    trans_all = []                       # 표준화 통계용 (train 만)

    for split, list_name in (('train', 'seg_list_vista4d_train.txt'),
                             ('test', 'seg_list_vista4d_test.txt')):
        entries = _read_list(path.join(args.corpus, list_name))
        out_split = path.join(args.out, split)
        makedirs(out_split, exist_ok=True)
        names = []

        for ent in entries:
            # 'vista4d/<scene>/<idx>'
            parts = ent.split('/')
            scene, idx = parts[-2], parts[-1]
            if scene not in scene_cache:
                d = path.join(args.corpus, '/'.join(parts[:-1]), 'da3')
                z = np.load(path.join(d, 'target_poses.npz'), allow_pickle=True)
                keys = {str(k): i for i, k in enumerate(z['keys'].tolist())}
                with open(path.join(d, 'prompts.json')) as f:
                    prompts = json_load(f)
                scene_cache[scene] = (np.asarray(z['extrinsics'], dtype=np.float32),
                                      np.asarray(z['intrinsics'], dtype=np.float32),
                                      keys, prompts)
            E, K, keys, prompts = scene_cache[scene]

            v = keys.get(idx)
            if v is None:
                errors.append(f'{scene}/{idx}\tno_key_in_target_poses')
                continue
            cap = (prompts.get(idx, {}).get('prompt_camera_with_scene_video', {})
                   or {}).get('concise')
            if not cap:
                errors.append(f'{scene}/{idx}\tno_caption')
                continue
            c2w = _c2w_opengl(E[v])
            if c2w is None:
                errors.append(f'{scene}/{idx}\tdiverged_pose')
                continue

            name = f'{scene}__{idx}'
            T = c2w.shape[0]
            lengths.append(T)
            k0 = K[v, 0]
            transforms = {
                'w': int(args.width), 'h': int(args.height),
                'fl_x': float(k0[0, 0]), 'fl_y': float(k0[1, 1]),
                'cx': float(k0[0, 2]), 'cy': float(k0[1, 2]),
                'frames': [{'transform_matrix': c2w[t].tolist(),
                            'monst3r_im_id': t} for t in range(T)],
            }
            with open(path.join(out_split, f'{name}_transforms_cleaning.json'), 'w') as f:
                json_dump(transforms, f, ensure_ascii=False)
            with open(path.join(out_split, f'{name}_caption.json'), 'w') as f:
                json_dump({CAPTION_KEY: cap}, f, ensure_ascii=False)
            names.append(name)
            if split == 'train':
                trans_all.append(c2w[:, :3, 3].astype(np.float64))

        with open(path.join(args.out, f'{split}_valid.txt'), 'w') as f:
            f.write('\n'.join(names) + '\n')
        counts[split] = len(names)

    with open(path.join(args.out, 'error.txt'), 'w') as f:
        f.write('\n'.join(errors) + ('\n' if errors else ''))

    print('=== stage json ===')
    for k in ('train', 'test'):
        print(f'  {k:6s} {counts[k]:6d}')
    print(f'  errors {len(errors):6d}  -> {path.join(args.out, "error.txt")}')
    if lengths:
        u = sorted(set(lengths))
        print(f'  traj length {u if len(u) <= 5 else f"{u[0]}..{u[-1]} ({len(u)} values)"}')
        print(f'  => standardization num_cams 는 max({max(lengths)}) 이상이어야 한다')

    if args.print_stats and trans_all:
        _print_stats(trans_all)
    return counts


def _print_stats(trans_all):
    """velocity 표기(첫 프레임 = 절대 위치, 이후 = 차분)의 평균/표준편차.

    주의: 현재 `trajectory_dataset.py:43` 이 `self.standardize = False` 로 못 박아 두어
    이 값들은 학습에 **쓰이지 않는다**. 그래도 yaml 에 남의 코퍼스(E.T.) 숫자를 그대로 두면
    나중에 standardize 를 켰을 때 조용히 틀린다. 그래서 우리 값을 재서 적어 둔다.
    """
    shift = np.stack([t[0] for t in trans_all])                    # (N,3)
    norm = np.concatenate([t[1:] - t[:-1] for t in trans_all], 0)  # (N*(T-1),3)
    print('=== standardization stats (train split, velocity 표기) ===')
    print(f'  norm_mean:  {np.round(norm.mean(0), 8).tolist()}')
    print(f'  norm_std:   {np.round(norm.std(0), 8).tolist()}')
    print(f'  shift_mean: {np.round(shift.mean(0), 8).tolist()}')
    print(f'  shift_std:  {np.round(shift.std(0), 8).tolist()}')


# ----------------------------------------------------------------------------------- #
# stage: clip
# ----------------------------------------------------------------------------------- #

def stage_clip(args):
    from evaluate.CLaTr.clip_extraction import (load_clip_model, encode_text,
                                                save_feats_custom)
    from pathlib import Path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_clip_model(args.clip_version, device)
    print('=== stage clip ===')
    for split in ('train', 'test'):
        names = _read_list(path.join(args.out, f'{split}_valid.txt'))
        caps = []
        for n in names:
            with open(path.join(args.out, split, f'{n}_caption.json')) as f:
                caps.append(json_load(f)[CAPTION_KEY])
        for i in range(0, len(names), args.batch_size):
            nb, cb = names[i:i + args.batch_size], caps[i:i + args.batch_size]
            with torch.no_grad():
                seq, tok = encode_text(cb, model, None, device)
            save_feats_custom(seq, nb, Path(args.out) / 'seq', split)
            save_feats_custom(tok, nb, Path(args.out) / 'token', split)
        print(f'  {split:6s} {len(names):6d} feats -> seq/{split}, token/{split}')


# ----------------------------------------------------------------------------------- #

def main():
    ap = ArgumentParser(description='vista d128 → CLaTr 학습 데이터')
    ap.add_argument('--corpus', default=CORPUS_DEFAULT)        # latentcam_da3_k6_d128
    ap.add_argument('--out', default=OUT_DEFAULT)              # CLaTr data_dir 가 될 곳
    ap.add_argument('--stage', default='all', choices=['json', 'clip', 'all'])
    ap.add_argument('--width', type=int, default=640)          # meta_vista4d.csv 와 같음
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--clip_version', default='ViT-B/32')      # CLaTr lm/clip.yaml 과 같음
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--print_stats', action='store_true')      # 표준화 통계 출력
    args = ap.parse_args()

    makedirs(args.out, exist_ok=True)
    if args.stage in ('json', 'all'):
        stage_json(args)
    if args.stage in ('clip', 'all'):
        stage_clip(args)
    print(f'\nCLaTr 학습: python -m src.train data_dir={args.out} '
          f'dataset/standardization=vista49')


if __name__ == '__main__':
    main()
