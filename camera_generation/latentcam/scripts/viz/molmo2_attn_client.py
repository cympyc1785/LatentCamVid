"""이 파일 위쪽만 고쳐서 `python molmo2_attn_client.py` 로 실행한다 (D164).

모델은 다른 프로세스(`molmo2_attn_video.py --serve http`)에 이미 올라가 있고, 이 스크립트는
거기로 (video, text) 를 보내 mp4 를 받아온다. 그래서 실행이 요청당 ~2초로 끝난다 —
모델 로드 17.7s 와 ViT(49프레임 x 729패치 x 27층)를 매번 다시 돌지 않는다.

서버 띄우기 (screen 에 한 번):
  PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
  $PY scripts/viz/molmo2_attn_video.py --serve http --port 8765 --gpu 0

의존성 없음 (stdlib urllib). 서버가 안 떠 있으면 위 명령을 안내하고 종료한다.
"""

# ================================ 여기만 고친다 ================================

VIDEO = 'vista4d/camel'          # 코퍼스 씬 이름, 또는 프레임 폴더 절대경로
START = 0                        # 시작 프레임
NUM_FRAMES = 49                  # 프레임 수
FPS = 25.0                       # 프롬프트의 timestamp 문구용 (학습 캐시 기본값 25.0)

# 돌릴 프롬프트들. span = query 로 쓸 부분 문자열 ('all' 이면 프롬프트 전체)
JOBS = [
    dict(text='Point to the larger pale camel.', span='the larger pale camel'),
    dict(text='Point to the tree with green leaves.', span='the tree with green leaves'),
    dict(text='Track the larger pale camel',       span='the larger pale camel'),
]

LAYER = 15                       # 0..35. -1 = 최종층(35). 기본 15 = D164 sweep 실측 최적
SPAN_TAIL = 0                    # >0 이면 span 의 마지막 N 토큰만 (0 = 전부)
HEAD = None                      # 정수면 그 head 만. None = 32 head 평균
PROBE = ''                       # '' 없음 / 'cache' 학습 캐시의 고정 probe / 임의 문장
NORM = 'per_frame'               # 'per_frame' 프레임 내 공간구조 / 'global' 프레임 간 질량비교
TAG = None                       # None = 파일명 자동 (씬_L층_프롬프트슬러그)

SERVER = 'http://127.0.0.1:8765'

# ==============================================================================

import json
import sys
import time
from os import path as osp
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

START_HINT = (
    'GPU 에 모델이 올라간 서버가 없다. screen 에 한 번 띄워둘 것:\n'
    '  PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python\n'
    '  cd /data1/cympyc1785/LatentCamVid/camera_generation/latentcam\n'
    '  $PY scripts/viz/molmo2_attn_video.py --serve http --port 8765 --gpu 0')


def post(path, payload, timeout=1800):
    req = urlrequest.Request(SERVER.rstrip('/') + path,
                             data=json.dumps(payload).encode(),
                             headers={'Content-Type': 'application/json'})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except HTTPError as e:                                  # 서버가 낸 에러 본문을 그대로
        raise SystemExit(f'[err] {path} {e.code}: {e.read().decode(errors="replace")}')
    except URLError as e:
        raise SystemExit(f'[err] {SERVER} 에 접속 못 함 ({e.reason})\n{START_HINT}')


def main():
    key = 'frames_dir' if osp.isabs(VIDEO) else 'scene'
    v = post('/video', {key: VIDEO, 'start': START, 'num_frames': NUM_FRAMES, 'fps': FPS})
    print(f"[video] {v['video']}  frames {v['frames']}  격자 {v['side']}x{v['side']}/frame  "
          f"prefix {v['prefix_tokens']} tok")

    for i, job in enumerate(JOBS):
        kw = dict(span='all', span_tail=SPAN_TAIL, layer=LAYER, head=HEAD, probe=PROBE,
                  norm=NORM, tag=TAG)
        kw.update(job)
        if TAG and len(JOBS) > 1:
            kw['tag'] = f'{TAG}_{i}'
        t0 = time.time()
        st = post('/run', kw)
        print(f"\n[{i}] {kw['text']!r}   span={kw['span']!r}   {time.time() - t0:.1f}s")
        print(f"    layer {st['layer']}   query {st['rows']} tok {st['query_text'][:52]!r}")
        print(f"    video 질량 {st['video_mass'] * 100:.2f}%   sink {st['sink_mass'] * 100:.2f}%")
        print(f"    테두리 {st['ring_mass'] * 100:.1f}% (uniform "
              f"{st['ring_uniform'] * 100:.1f}% — 크게 넘으면 물체가 아니라 위치 artifact)")
        print(f"    앞3프레임 {st['mass_f0_2'] * 100:.1f}% (uniform "
              f"{st['frame_uniform'] * 100:.1f}%)   엔트로피 {st['frame_entropy']:.3f}   "
              f"peak f{st['peak_frame']}")
        print(f"    mp4 {st['mp4']}")


if __name__ == '__main__':
    sys.exit(main())
