"""영상을 프레임 단위로 넘겨 보고 **그 한 장을 파일로 저장**하는 viser 뷰어.

왜 필요한가: 릴(`warp.mp4`, `vista4d/<arm>.mp4`)을 볼 때 "몇 번째 프레임에서 카메라가 벽을
뚫는가"를 찾아야 하는데, 영상 플레이어는 프레임 번호를 안 알려준다. 초 단위로 긁어 놓고
`ffmpeg -ss` 로 다시 뽑으면 그 초가 어느 프레임인지 또 어긋난다. 여기서는 슬라이더 값이 곧
프레임 인덱스이고, 저장 파일 이름에 그 번호가 박혀서 `viser_cloud.py` 의 frame 슬라이더와
**같은 번호로** 대조된다 (같은 49프레임 규약).

영상 여러 개를 같이 받는다 (`--video a.mp4 b.mp4 ...`). arm 5개를 비교할 때 스크립트를 5번
띄우면 슬라이더가 따로 움직여서 "같은 프레임"을 못 맞추는데, 여기서는 드롭다운으로 영상만
갈아끼우고 **프레임 슬라이더는 공유**된다.

## 디코딩

`decord` 로 **지연 랜덤 접근**한다 (프레임 하나만 뽑으므로). 전량 디코드는 49프레임짜리엔
문제없지만 원본 영상(수천 프레임 1080p)에서 수 GB 가 된다. 한 번 뽑은 프레임은 dict 에
캐시하므로 슬라이더를 되감아도 다시 디코드하지 않는다. decord 가 못 여는 컨테이너는
`imageio` 순차 디코드로 내려간다 (그쪽은 되감기가 느려서 두 번째 선택지다).

프레임 수가 영상마다 다를 수 있다 (warp 릴은 49, vista4d 생성물도 49지만 소스 원본은 다르다).
슬라이더 상한은 **그때 고른 영상**의 길이로 맞춘다 — 공통 최소값으로 굳히면 긴 영상의 뒷부분을
아예 못 본다.

## 저장

`save frame` 이 지금 보이는 그 배열을 그대로 png 로 쓴다 (다시 디코드하지 않는다 — 화면과
파일이 다르면 스크립트의 의미가 없다). 이름은 **`<부모폴더>_f0012.png`** 다 — 번들 릴은 preset
폴더마다 파일명이 똑같이 `warp.mp4` 라 파일명으로 저장하면 세 preset 이 전부
`warp_f0012.png` 가 되고 `_2`/`_3` 로만 갈려 어느 preset 인지 알 수 없다. 구분되는 축은
폴더 이름(`crane_up`)이다. 같은 폴더의 `warp.mp4` 와 `warp_gendop.mp4` 를 같이 볼 때는 폴더로도
안 갈리므로 `--name_from both`(부모_파일명) 를 쓴다 — 겹치면 기동 시 경고한다.
`--name_from stem` 이 예전 동작이다. 같은 프레임을 또 누르면 `_2`, `_3` 이 붙는다
(덮어쓰면 방금 저장한 것이 조용히 사라진다).
`save all videos` 는 지금 프레임을 **전 영상에서** 한 장씩 뽑는다 — arm 비교 그림을 만들 때
드롭다운을 5번 돌리는 것이 실제 병목이었다.

입력  : 영상 파일 경로 하나 이상
출력  : 브라우저 (`http://localhost:<port>`) + `<--out_dir>/<prefix>_f<번호>.png`
        (prefix 기본 = 부모 폴더 이름, `--name_from`)

예시 (env vista4d):
    python -u scripts/viser_frame.py --port 8095 \
        --video results/20260921_d221_bundles/bmx-bumps/crane_up/warp.mp4
    python -u scripts/viser_frame.py --port 8095 --out_dir /data1/.../tmp/pick \
        --video results/20260921_d221_bundles/bmx-bumps/crane_up/vista4d/*.mp4
"""
import time
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np
import viser

# 저장 기본 폴더. `/tmp` 를 안 쓰는 이유는 프로젝트 규약(시스템이 임의로 비운다) 그대로다.
OUT_DIR_DEFAULT = "/data1/cympyc1785/LatentCamVid/tmp/frames"


class Clip:
    """영상 하나. 프레임을 **필요할 때만** 디코드하고 캐시한다.

    decord 를 먼저 쓰는 이유는 랜덤 접근이 목적이기 때문이다 — 슬라이더는 앞뒤로 아무렇게나
    움직이는데, 순차 디코더로 되감으려면 매번 처음부터 다시 읽어야 한다. decord 가 컨테이너를
    못 열 때만 `imageio` 로 내려가고, 그때는 **전량 디코드**한다 (순차 디코더로 되감기를
    흉내내는 것보다 한 번 다 읽어 두는 게 낫다 — 다만 메모리를 쓰므로 경고를 찍는다).
    """

    def __init__(self, file_path: str):
        self.path = path.abspath(file_path)
        self.stem = path.splitext(path.basename(self.path))[0]
        # 저장 파일 prefix. **기본은 부모 폴더 이름**이다 — 번들 릴은 preset 폴더마다 파일명이
        # 똑같이 `warp.mp4` 라, 파일명으로 저장하면 세 preset 이 전부 `warp_f0023.png` 가 되고
        # `_2`/`_3` 이 붙어 어느 preset 인지 구분이 안 된다. 폴더 이름(`crane_up`)이 실제로
        # 구분되는 축이다. 같은 폴더에서 `warp.mp4` 와 `warp_gendop.mp4` 를 같이 볼 때는
        # 폴더만으로 안 갈리므로 `--name_from both` 를 쓴다.
        self.parent = path.basename(path.dirname(self.path)) or "root"
        # 라벨은 파일명만으로는 안 된다 — arm 릴이 전부 `warp.mp4` 라 5개가 같은 이름이 된다.
        self.label = f"{path.basename(path.dirname(self.path))}/{path.basename(self.path)}"
        self.cache, self.reader, self.frames = {}, None, None
        try:
            import decord
            self.reader = decord.VideoReader(self.path, num_threads=2)
            self.count = len(self.reader)
            self.kind = "decord"
        except Exception as exc:                            # noqa: BLE001 — 어떤 실패든 내려간다
            import imageio.v3 as iio
            print(f"[{self.label}] decord 실패({type(exc).__name__}) -> imageio 전량 디코드")
            self.frames = list(iio.imiter(self.path))
            self.count = len(self.frames)
            self.kind = "imageio"
        assert self.count > 0, f"프레임이 0개다: {self.path}"
        first = self.frame(0)
        self.height, self.width = int(first.shape[0]), int(first.shape[1])

    def prefix(self, mode: str) -> str:
        """저장 파일 prefix. `parent`(기본) / `stem` / `both`."""
        if mode == "stem":
            return self.stem
        if mode == "both":
            return f"{self.parent}_{self.stem}"
        return self.parent

    def frame(self, index: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB. 캐시 적중이면 디코드하지 않는다."""
        index = int(np.clip(index, 0, self.count - 1))
        if index not in self.cache:
            if self.kind == "decord":
                self.cache[index] = np.asarray(self.reader[index].asnumpy(), dtype=np.uint8)
            else:
                self.cache[index] = np.asarray(self.frames[index], dtype=np.uint8)
        return self.cache[index]


def save_png(image: np.ndarray, out_dir: str, stem: str, index: int) -> str:
    """`<out_dir>/<stem>_f0012.png`. 이미 있으면 `_2`, `_3` … 을 붙인다.

    덮어쓰지 않는 이유: 같은 프레임을 다른 영상/다른 세대에서 두 번 뽑는 일이 흔하고,
    덮어쓰면 방금 저장한 것이 조용히 사라진다.
    """
    makedirs(out_dir, exist_ok=True)
    base = path.join(out_dir, f"{stem}_f{int(index):04d}")
    out = f"{base}.png"
    bump = 2
    while path.exists(out):
        out = f"{base}_{bump}.png"
        bump += 1
    import imageio.v3 as iio
    iio.imwrite(out, np.asarray(image, dtype=np.uint8))
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--video", nargs="+", required=True, type=str,
                        help="영상 경로 하나 이상 (드롭다운으로 갈아끼운다)")
    parser.add_argument("--port", default=8095, type=int)
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT, type=str)
    # 기동 시 볼 프레임. 릴에서 특정 프레임을 다시 뽑을 때 슬라이더를 끌 필요가 없다.
    parser.add_argument("--frame", default=0, type=int)
    # 저장 파일 이름의 앞머리. 기본 `parent` — 위 `Clip.parent` 주석이 근거다.
    parser.add_argument("--name_from", default="parent", choices=("parent", "stem", "both"),
                        help="저장 파일 prefix: parent(기본, 부모 폴더) / stem(파일명) / both")
    args = parser.parse_args()

    clips = [Clip(v) for v in args.video]
    labels = [c.label for c in clips]
    # 같은 라벨이 둘 이상이면 (같은 폴더 구조의 다른 세대) 인덱스를 붙여 가른다 — viser
    # 드롭다운은 같은 문자열 두 개를 구분하지 못한다.
    for i, label in enumerate(labels):
        if labels.count(label) > 1:
            labels[i] = f"[{i}] {label}"
    by_label = dict(zip(labels, clips))

    # prefix 가 겹치면 저장 파일이 `_2`/`_3` 로만 갈려 어느 영상인지 알 수 없다 — 덮어쓰기는
    # 안 나지만 구분이 안 되는 건 같은 문제라 여기서 알린다.
    prefixes = [c.prefix(args.name_from) for c in clips]
    dup = sorted({x for x in prefixes if prefixes.count(x) > 1})
    if dup:
        print(f"[name] prefix 가 겹친다 {dup} — 저장 파일이 `_2`/`_3` 로만 갈려 어느 영상인지"
              f" 알 수 없다.\n       `--name_from both` (부모_파일명) 를 쓰면 갈린다.")

    server = viser.ViserServer(port=args.port)
    state = {"clip": clips[0], "frame": int(np.clip(args.frame, 0, clips[0].count - 1))}

    gui_pick = server.gui.add_dropdown("video", options=labels, initial_value=labels[0],
                                       disabled=len(labels) == 1)
    gui_frame = server.gui.add_slider("frame", min=0, max=max(clips[0].count - 1, 0), step=1,
                                      initial_value=state["frame"])
    gui_play = server.gui.add_checkbox("play", initial_value=False)
    gui_fps = server.gui.add_slider("fps", min=1, max=30, step=1, initial_value=10)
    gui_info = server.gui.add_text("info", initial_value="", multiline=True, disabled=True)
    # 이미지는 GUI 패널에 둔다. 3D 씬(`scene.add_image`)에 두면 카메라를 맞춰야 화면을 채워서,
    # "프레임 하나를 크게 본다"는 목적에 손이 하나 더 든다.
    gui_image = server.gui.add_image(clips[0].frame(state["frame"]), label="", format="jpeg")
    gui_save = server.gui.add_button("save frame")
    gui_save_all = server.gui.add_button("save frame (all videos)")
    gui_saved = server.gui.add_text("saved", initial_value="", multiline=True, disabled=True)

    def redraw():
        clip, index = state["clip"], int(gui_frame.value)
        state["frame"] = index
        image = clip.frame(index)
        # `.image` 대입이 갱신되는 프로퍼티다 — handle 을 지웠다 다시 만들면 GUI 순서가
        # 바뀌어 버튼이 이미지 위로 튀어오른다.
        gui_image.image = image
        gui_info.value = "\n".join([
            f"{clip.label}",
            f"frame       {index} / {clip.count - 1}   ({clip.width}x{clip.height})",
            f"decoder     {clip.kind}   캐시 {len(clip.cache)}/{clip.count} 프레임",
            f"path        {clip.path}",
            f"out_dir     {args.out_dir}",
            f"save as     {clip.prefix(args.name_from)}_f{index:04d}.png  "
            f"(--name_from {args.name_from})",
        ])

    @gui_pick.on_update
    def _(_event):
        clip = by_label[str(gui_pick.value)]
        state["clip"] = clip
        # 상한을 먼저 올려야 한다 — 짧은 영상에서 긴 영상으로 갈 때 value 를 먼저 쓰면
        # 옛 상한에 잘린다. 반대 방향은 viser 가 value 를 상한으로 당겨 준다.
        gui_frame.max = max(clip.count - 1, 0)
        gui_frame.value = int(np.clip(state["frame"], 0, clip.count - 1))
        redraw()

    gui_frame.on_update(lambda _: redraw())

    @gui_save.on_click
    def _(_event):
        clip, index = state["clip"], int(gui_frame.value)
        out = save_png(clip.frame(index), args.out_dir, clip.prefix(args.name_from), index)
        gui_saved.value = f"{out}\n{gui_saved.value}"[:2000]
        print(f"[save] {out}")

    @gui_save_all.on_click
    def _(_event):
        index = int(gui_frame.value)
        done = []
        for clip in clips:
            if index >= clip.count:        # 길이가 다른 영상은 건너뛴다 (클램프하면 다른
                continue                   # 프레임을 같은 번호로 저장해 대조가 거짓이 된다)
            done.append(save_png(clip.frame(index), args.out_dir,
                                 clip.prefix(args.name_from), index))
        gui_saved.value = "\n".join(done + [gui_saved.value])[:2000]
        print(f"[save all] {len(done)}장 (frame {index})")

    redraw()

    rows = [("videos", len(clips)), ("out_dir", args.out_dir),
            ("frame", f"{state['frame']} / {clips[0].count - 1}"),
            ("url", f"http://localhost:{args.port}")]
    width_key = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key:<{width_key}}  {value}")
    for clip, label in zip(clips, labels):
        print(f"  {label:44s} {clip.count:4d} 프레임  {clip.width}x{clip.height}  {clip.kind}")

    while True:
        if gui_play.value:
            gui_frame.value = (int(gui_frame.value) + 1) % max(state["clip"].count, 1)
        time.sleep(1.0 / max(1.0, float(gui_fps.value)) if gui_play.value else 0.08)


if __name__ == "__main__":
    main()
