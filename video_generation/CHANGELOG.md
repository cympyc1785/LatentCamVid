# Changelog (video_generation)

[Keep a Changelog](https://keepachangelog.com/) 규약. `camera_generation/latentcam/CHANGELOG.md` 와
별개다 — 이쪽은 vendored 비디오 생성 모델 6종을 돌리는 `tools/` 스크립트만 다룬다.
(`video_generation/tools` 는 `.gitignore:223` 로 git 추적 대상이 아니다. 이 파일은 로컬 기록용.)

## [Unreleased]

### Added
- **Vista4D 어댑터** — 벤치마크 7번째 모델. 3단계라 "카메라 파일만 갈아끼우기"가 안 되고
  소스마다 4D 재구성을 먼저 돌려야 한다.
  - `tools/recammaster/vista4d_prepare.py` (신규) — canonical 단위 궤적 -> stage 2 의
    `--cam_path` npz. `T_w[i] = C_src[a] @ T_local[i]`, `a` 는 stage 2 의 중앙 슬라이스
    시작 프레임(`render_single.py:52-54`)을 재구성 프레임 수로 계산한 것. 공통 손잡이는
    `|t|max = g * S`, `S` 는 `sierp_scene_scale.py:58` 과 **같은 정의**의 mean ray length 를
    Vista4D 자신의 재구성에서 잰 값 (전 픽셀; `--exclude_sky` 는 진단용). depth 는 EXR
    (float16 단일 채널 `Y`, `utils/media.py:save_depths`) 로 읽는다.
  - `results/20260818_vista4d/run_vista4d_recon.sh` (신규) — stage 1 드라이버. taylor 소스는
    81 프레임이라 stage 1 이 **중앙**을 자르는데 sierp/trajc 는 **앞** 49 를 쓰므로, 여기서
    앞 49 로 미리 트림해 창을 맞춘다. `PYTHONPATH=$V4` 필수 (`scripts/preprocess/` 안에서
    `from utils.media import ...` 가 터진다 — cwd 로는 해결 안 된다).
    소스 목록: `hs_sC` / `hs_dynA` / `taylor_c{01,02,05,19,23}` / `rcm3` / `rcm5`.
    ReCamMaster 예시 2개는 원본이 1195 프레임인데 다른 모델도 전부 **앞** N 프레임만 읽으므로
    (`run_grid.py:339,350,488`) 여기서도 앞 49 로 트림한다.
  - `vista4d_prepare.py --pivot_match {none,z_med,S}` (기본 `none` = 기존과 bit-identical;
    g0p2 8종 재생성 maxdiff 0.000e+00 확인) — 궤도 카메라마다 **g 를 따로** 잡아 궤도 반경
    (`pivot_unit * g * S`) 이 씬 깊이에 오게 한다. `pivot_radius()` 는 광축들의 최소제곱
    수렴점까지의 frame0 기준 거리이고, 순수 이동 궤적은 광선이 평행해 특이행렬이므로
    `inf` 를 돌려주고 그런 카메라는 기본 g 를 그대로 쓴다. 배경은 memory
    `dl3dv-orbit-cams-pivot-collapse`: canonical `rmax=1` + 이동만 스케일하는 g 가 dl3dv
    궤도 반경을 씬 깊이의 0.08~0.48 배로 앉혀 warp 이 새까매진다 (좌표계 버그가 아니다 —
    hole 이 생기는 쪽이 세 축 다 trajc 와 일치). manifest 에 `pivot_unit` / `pivot` /
    per-entry `scale` 을 남긴다 (`inf` 는 표준 JSON 이 아니라 `null` 로).
  - `results/20260818_vista4d/run_vista4d_grid.sh` (신규) — stage 2+3 을 소스 x 카메라로.
    `GTAG` / `SCALES` 환경변수로 "디렉토리 이름 = g" 규약을 끊을 수 있다 — `--pivot_match`
    로 낸 카메라는 카메라마다 g 가 달라 그 규약이 성립하지 않는다. 안 주면 기존과 동일.
    `rcm3` 만 카메라 **8종** (기본 6 + `dl3dv_L49_b_p`, `dl3dv_L81_b_p`; 사용자 지시).
    `_b_p` 둘은 canonical `norm='path'` 라 `rmax` 가 0.7464 / 0.5804 이고
    `vista4d_prepare.py:178` 이 rmax 재정규화를 안 하므로 같은 g 에서 `|t|max` 가
    2.1203 / 1.6487 m (기본 6종은 2.8406 m) — 대신 회전이 104.2 / 178.7 도로 훨씬 크다.
  - `run_grid.py --model vista4d` + `--v4_recon <stage1 root>` — stage 2(`render_single`)
    -> `_cond/video_pc.mp4` 를 `_warp/warp.mp4` 로 복사 -> stage 3(`scripts.inference.inference`,
    Wan2.1-T2V-14B + `384p49_step=30000`). `--warp_only` 는 stage 2 까지만.
- `results/20260818_rcm3_gen/` (신규) — rcm3 x 카메라 6종 TrajectoryCrafter **좌표계 수정 +
  절대 pose** 렌더/생성. 기존 `20260813_camgrid_stage1/rcm3/trajectorycrafter/` 는 14개 태그가
  전부 `_legacyconv/` (2026-08-14 이전 버그난 규약, 절대 pose 아님) 라 새로 판다.
  `run_rcm3_trajc.sh` = `--tc_abs_pose <rcm3 sierp DA3 npz> --tc_radius_meanray --scales 0.4`
  (`s = 2g`, g=0.2). trajc 카메라 npy 가 소스 무관 상수임을 md5 로 확인(taylor c01 과 동일).
  6/6 rc=0 완료. `make_concat.sh` 는 vista4d 판과 같은 무-`{clip}` 경로 규약을 쓰고, input 행에
  원본 `3.mp4`(1195프레임) 대신 `20260818_vista4d/src/rcm3_first49.mp4` 를 넣는다 — 모델은 앞 49
  프레임만 읽으므로 원본을 붙이면 input 행만 다른 구간을 보여준다.
- `tools/recammaster/sd_revpair_prepare.py` (신규) — Scene-Decoupled 같은 scene 의 clip 2개를
  `rev(A)[80..1] + B[0..80]` (161장) 으로 이어 붙여 **한 번에** DA3 를 돌려 두 clip 의
  depth/카메라를 한 좌표계·한 스케일로 만든다 (clip 간 frame0 카메라가 동일해서 역재생이
  자연스럽게 이어진다). `--split_recon` 이 clip 별로 되썰고 `--verify` 가 SD 의 GT 미터
  pose 와 대조한다 — clip 간 스케일 차이 0.33% / 1.96% / 2.29% (`results/20260818_sd_revpair/config.md`).
- `results/20260818_taylor_gen/make_concat.sh` (신규) — 모델별 concat 영상
  (행 = input / depth warp / gen, 열 = 카메라 6종).
- `results/20260818_vista4d/make_concat.sh` (신규) — 같은 3행 concat 의 Vista4D 판.
  경로 층이 달라(`out/<gtag>/<src>/vista4d/<cam>_<gtag>/`, `<clip>` 층 없음) `{clip}` 대신
  `{src}`+`{cam}` 템플릿을 쓴다. 소스 mp4 는 stage 1 에 실제로 먹인 것(taylor 는 앞 49 트림).
- `tools/recammaster/rerope_prepare.py` (신규) — 우리 소스/타깃 카메라를 **ReRoPE V2V 네이티브
  입력**으로 변환한다. ReRoPE 는 벤치마크 6종 중 유일하게 **소스(context) 궤적을 따로** 요구
  하므로(`src/v2v_handler.py:121` 이 `videos/<stem>.npz` 에서 c2w 를 읽는다) depth warp 계열이
  쓰는 것과 **같은 DA3 재구성**을 context 로 넣어 "같은 카메라 쌍" 비교가 성립하게 했다.
  - 출력: `<out>/data/{metadata.csv, videos/<stem>.mp4(심링크), videos/<stem>.npz}` +
    `<out>/targets/<camera>.npz` + `<out>/prepare.json`(진단 포함).
  - 소스 npz = DA3 `extrinsics`(w2c) 를 inv 해서 c2w, translation **마지막 열**.
    타깃 npz 는 `--target_frame` 이 정한다 (아래 Fixed 항목). 기본 `world` =
    `(C_src0 @ conv(json.T)).transpose(0,2,1)` — **소스와 같은 규약의 c2w 를 전치만** 한 것.
  - **`/100` 을 여기서 한다.** ReRoPE 코드에는 ReCamMaster 의 `c2w[:3,3] /= 100` 이 없어서
    UE5-cm 를 그대로 주면 타깃이 소스의 100배가 되고, `normalize_joint_translation` 이 둘을
    같은 max‖t‖ 로 나누므로 context 궤적이 cond‖t‖≈0.01 로 뭉개진다. world offset
    `[5000,1500,100]` 도 같이 뺀다.
  - `--selftest`: `opencv_to_ue5` → (`transpose` + `convert_c2w_convention`) 왕복이 항등임을
    확인 — 치환 `[2,0,1,3]∘[1,2,0,3]` = identity, 두 flip 이 **같은 열**에 걸려 상쇄.
    max err 4.7e-15. (이 왕복이 항등인 것과 **ReRoPE 가 그 결과를 원하는 것은 다른 얘기**였다.
    아래 Fixed.)
  - 진단이 `normalize_joint_translation` 를 재현해 카메라별 `joint_max / cond‖t‖ / tgt‖t‖ /
    비율` 을 표로 찍고, `joint_max < eps(=1)` 이면 미학습 regime 경고를 낸다. sC 실측:
    joint_max 3.7037, cond 1.0000, tgt 0.7585 (6종 동일).
  - 진단에 `cdet / tdet / tgtrot0 / tgtrotmax` 4열 추가 + 경고 2개: (a) cond/tgt 의 conv-후
    det 부호가 **서로** 다르면 규약 불일치, (b) `tgtrot0 > 1°` 면 타깃이 소스 world 에 안
    심긴 것. 이 두 열이 아래 Fixed 의 버그를 잡아낸 계기다. `--axis_precomp inv_conv` 에서는
    `[축검증]` 단계가 추가로 내부 타깃 상대 w2c == JSON 로컬 궤적을 확인한다.
- `warp_grid_video.py --rows` 에 `{cam}` placeholder 추가 (기존 placeholder 5개는 그대로) —
  배율 태그가 안 붙은 canonical 이름. ReRoPE 는 출력 파일명이 `<stem>_<camera>.mp4` 라
  `{tag_s}`/`{tag_t}` 로는 경로를 만들 수 없었다.
- `tools/recammaster/run_trajcrafter_matrix.py --abs_pose <npz|npy>` (기본 None = 기존 상대
  pose 동작) — TrajectoryCrafter 를 **SierpinskiCam 과 같은 절대 pose semantics** 로 돌린다.
  기존에는 `pose_s = poses[anchor_idx].repeat(N)` (demo.py:549) 이라 실현 변환이
  `T = inv(rel_t)` 였고 **소스 카메라 자기 운동이 그대로 남았다**. sierp 는
  `pose_s = c2ws`(프레임별 DA3 포즈) 라 `T = target ∘ inv(source)` 로 소스 운동을 취소한다
  (`results/20260817_dl3dv_warp/config.md` §4 가 그 차이를 실측). `--abs_pose` 는
  `pose_s_i = inv(rel_src_i) @ P0` 를 주어 `T = inv(rel_t) @ rel_src` 로 같은 형태를 만든다.
  - 입력이 `.npz` 면 sierp 조건 렌더가 남긴 DA3 결과를 그대로 쓴다 — `rel_i = E_0 @ inv(E_i)`
    로 frame0 재앵커(sierp `--anchor0` 과 같은 보정; sC 소스는 `|E_0 - I| = 2.92`, ref view
    가 frame 37) 후 같은 npz 의 depth+intrinsics 로 잰 `S_da3` 로 나눠 무차원화.
  - `--radius_meanray` 를 **요구**하고 `--target_dxw`/`--free_anchor` 와는 **배타**다
    (전자는 radius 를 target 궤적에서 역산해 소스 궤적까지 리스케일하고, 후자는 `pose_s` 를
    이중 정의한다).
  - 검증: (1) `--abs_pose` 없으면 `pose_s` 가 기존과 **바이트 동일**, (2) `rel_src = I` 면
    `T` 가 기존과 차이 0.0, (3) `rel_src = rel_t` 면 `T = I` (잔차 1.8e-15), (4) selftest
    worst 8.8e-08 유지. 실측(sC/truck_left): 소스 자기 운동 `|t|max = 0.2637*S`, rot 7.50°
    -> 실현 `|t|max 2.8094 -> 1.4446` (0.514배), rot `0.00° -> 7.50°` (sierp 실측 7.50° 와 일치).
  - `_report_realized()` 가 실현 `T` 의 `|t|max`/rot 를 상대 semantics 값과 나란히 찍는다
    (런타임 warp 은 픽셀로 못 재므로 — memory `measure-realized-fails-on-runtime-warp`).
- `run_grid.py --tc_abs_pose <src_track>` (기본 None = 기존 상대 pose 동작) — 위 `--abs_pose`
  를 그리드 러너에서 쓸 수 있게 플럼했다. 값은 sierp 조건 렌더가 이미 남긴 DA3 npz
  (`<out_root>/<src_stem>/sierpinskicam/<cam>_s*/_cond/cam/<src_stem>.npz`) 를 그대로 준다 —
  소스 재구성은 요청 카메라와 무관하므로 6종 중 아무거나 하나면 된다 (sC 실측: 6개 npz 의
  `rel` max dev 2.6e-4, `S_da3` 14.0446~14.0661).
  자식 스크립트도 같은 조합을 막지만 entry 마다 spawn 한 뒤 죽으면 로그가 흩어지므로
  **fail-fast guard 4개**를 러너 쪽에 뒀다: 파일 없음 / `--model` 이 trajectorycrafter 가 아님
  (sierp 는 원래 절대 pose) / `--tc_radius_meanray` 없음 / `--tc_target_dxw`·`--free_anchor`
  충돌. 4개 전부 발화 확인, `--dry` 6/6 정상.
- `warp_grid_video.py --tc_abs_root <root>` (기본 None = 3행 = 기존 동작) — 주면
  `TrajectoryCrafter render (abs pose)` 행이 4번째로 붙는다. **같은 모델 / 다른 pose semantics**
  를 세로로 나란히 놓으려는 것으로, 기존 `ROWS` 는 root 가 행마다 하나씩 고정이라 이 축을
  표현할 수 없었다. 실측 확인: sC/g0p2 6카메라 4행 `(49, 600, 1620, 3)`, 빠진 패널 0개.
- `warp_grid_video.py --rows LABEL=TEMPLATE ...` (기본 None = 기존 `ROWS` 3행) — 행 구성을
  통째로 갈아끼운다. placeholder `{root} {abs_root} {clip} {src} {tag_s} {tag_t}`, 빈 TEMPLATE
  = 소스 패널. **같은 모델의 warp 과 gen 을 나란히** 깔려고 넣었다 (memory
  `show-depth-warp-with-output`) — `ROWS` 는 행마다 모델이 하나씩 고정이라 그 축이 안 나온다.
  회귀 확인: 무옵션 3행 `(49,450,540,3)` 그대로, `--rows` 2행 `(49,300,540,3)`,
  `LABEL=` 없는 spec 은 argparse error.
- `tools/recammaster/warp_direction_check.py` (신규) — 런타임 depth warp 의 **이동 방향**을
  hole(정확한 검은색) 중심 좌표로 판정한다. SIFT/Farneback(`measure_realized.py`)은 구멍이
  flow 를 0 으로 끌어 배율을 6배까지 과소보고하므로 런타임 warp 에는 못 쓴다. dome 으로
  구멍을 메우는 `dense_tx` 에 쓰면 rate 가 과소로 나와서 `--warn_filled` 가 경고한다.
- `run_grid.py --sierp_save` (기본 `rgb,dense_tx` = 기존 동작 = stage B 가 먹는 것) —
  진단용으로 구멍을 안 메운 순수 forward warp(`dense`)과 `mask` 를 같이 받을 수 있게 했다.
- `tools/recammaster/warp_grid_video.py --src_stem` (기본 None = `taylor_<clip>` = 기존 동작) —
  경로 템플릿에 `taylor_<clip>` 이 박혀 있어 taylor 이외의 소스(`hs_sC` 같은 DL3DV half-split
  소스)로는 못 쓰던 걸 `run_grid --src` 값과 맞춰 열 수 있게 했다.
- `tools/recammaster/warp_grid_video.py --align {head,resample}`: 기본 `head` 는 앞 n 장 1:1 정렬. 두 모델 다 소스 프레임 0..48 을 stride 1 로 먹는데, SierpinskiCam `_warp/warp.mp4` 는 moviepy `duration=49/12` 부동소수 잔차로 마지막 프레임이 복제돼 50 장이라 인덱스 비례 리샘플하면 sierp 행이 최대 1 프레임 앞서갔다. `resample` 로 구 동작 유지.
- `make_sbs.py --sep` / `--src_ext` (기본 `__` / `.mp4` = 기존 동작) — 결과 파일 stem 의
  `<소스><sep><카메라>` 구분자와 소스 확장자를 고를 수 있게 했다. ReRoPE 는 `1_cam01` 처럼
  `_` 하나를 쓴다.
- `tools/recammaster/warp_grid_video.py` (신규) — 한 소스에 대해 **열 = 카메라 / 행 = 모델**로
  depth warp 을 한 화면에 까는 비교 영상 빌더. `make_compare.py` 는 열이 모델이라 축이 다르다.
  길이가 다르면(sierp 50 / trajc 49) 인덱스 비례 리샘플 — 궤적이 전부 canonical 21 pose 를
  자기 길이로 편 것이라 정규화 시간 `t` 가 같은 pose 를 가리킨다.
- `run_grid.py --warp_only` (기본 꺼짐 = 기존 동작) — depth warp 까지만 돌리고 생성(diffusion)을
  건너뛴다. sierpinskicam 은 stage A(`create_sierpinskicam_conditioning`)만 돌고 stage B 를 빼며
  `_warp/warp.mp4` 는 그대로 남는다. trajectorycrafter 는 `--render_only` 로 `render.mp4` 직후
  종료한다. 카메라 세기를 눈으로 정하는 단계에서 모델 로드 + 샘플링 비용을 안 낸다 (warp 은
  결정론적이라 세기 판정에 생성 결과가 필요 없다).
- `run_grid.py --tc_radius_meanray` / `run_trajcrafter_matrix.py --radius_meanray`
  (기본 꺼짐 = 기존 동작인 중심픽셀 depth + `min(r,5)` clamp) — TrajectoryCrafter 의 `radius` 를
  DepthCrafter 자기 depth 의 **mean ray length** `S = mean(z * ||K^-1 [u+.5,v+.5,1]||)` 로 잡는다.
  npy 단위가 `0.5*radius` 이므로 `emit --scales 2g` 와 짝지으면 `|t|max = g*S` 가 되어, **g 하나가
  모든 소스·모델 공통인 "scene scale 대비 이동량" 손잡이**가 된다. `--target_dxw` 와 달리 target
  궤적을 안 보므로 소스당 상수라 카메라 여러 개를 같은 세기로 비교할 수 있다. 같이 주면
  `--target_dxw` 가 이긴다.
- `tools/recammaster/sierp_scene_scale.py` (신규) — 소스 영상의 scene scale `S` 를 SierpinskiCam 이
  쓰는 게이지(DA3NESTED-GIANT-LARGE, 영상 앞 49프레임)로 오프라인 측정한다. 정의는 DL3DV 의
  저장된 `avg_scale` 과 같은 frame0 mean ray length. sierp 는 warp 전에 카메라 JSON 을 확정해야
  해서 런타임 계산(trajc 의 `--radius_meanray`)이 불가능하다. emit 은 `--scales g*S/1.9876`
  (`native_1x` 1.9876 m 를 나눠 준다) 로 준다.
- `run_grid.py` 의 소스 `hs_sC` — 옛 `hs_static` 을 대체하는 DL3DV static 소스.
  옛 것은 half-split 경계에서 rot48=170도라 전이 warp 의 hole 이 0.846 이었다 (context 끝과
  target 끝이 서로 뒤를 봤다). sC 는 rot48=11.03 / tau48=0.514 로 hole 0.381.
  선정 근거와 985 scene 스크리닝 표는 `results/20260817_hs_dl3dv3/config.md`.
- `tools/recammaster/halfsplit_fullrecon_transfer.py` (신규) — **GT 카메라 없이** target 카메라를
  만든다. ctx+tgt 를 한 번에 DA3 recon 한 뒤, ctx 만 넣어 나온 **모델 게이지 ctx 카메라**에
  umeyama sim3 로 full recon 전체를 이전하고, 이전된 target 카메라로 모델 ctx depth 를 렌더한다.
  배율이 sim3 의 `s` 로 닫혀 나와 `halfsplit_sweep.py` 의 `f` 스윕이 필요 없다 (검증용으로
  `--sweep` 은 남겨 뒀고, 최적점이 1.0 이면 이전이 맞은 것). 카메라를 안 내는 TrajectoryCrafter
  (DepthCrafter) 는 `--align depth` 로 depth 비에서 `s` 를 뽑는다. 대조군 `--cam_src`:
  `transfer`(이 방법) / `gtalign`(GT ctx 카메라에 맞춰 이전 — recon 의 target 외삽 품질만) /
  `gt`(GT 카메라 상한). 결과: `results/20260817_fullrecon_fix8/config.md`.
- `tools/recammaster/alaya_run_dumpwarp.py` (신규) — AlayaWorld 를 돌리면서 내부 depth warp 을
  `--warp_out <경로>` 로 뽑아 준다. vendored 코드는 안 고치고 우리 드라이버에서 monkey-patch 한다.
  AlayaWorld 는 warp 을 디스크에 안 남겨서 (memory `alayaworld-is-depth-warp-model`) 무슨 조건이
  들어갔는지 눈으로 확인할 방법이 없었다. GEN3C 처럼 10장 z-buffer 누적이라는 것도 이걸로 확인했다.
- `tools/recammaster/run_grid.py` 의 `SAVE_WARP` (기본 켜짐) / `--no_save_warp` (= 기존 동작) —
  alayaworld / sierpinskicam 의 depth warp 중간 영상을 `<out>/_warp/warp.mp4` 로 남긴다.
  sierpinskicam 은 stage A 가 만든 `dense_tx` 를 복사하고, alayaworld 는 위 dumpwarp 을 탄다.
- `tools/recammaster/camviz_from_native.py` (신규) — 모델별 **native 카메라 파일**(recammaster/
  infcam/sierpinskicam JSON, cameraanything JSON, alayaworld `_camera.pt`, trajC `.npy`)을 그대로
  읽어 궤적을 3D 로 그린다. canonical 이 아니라 **모델이 실제로 먹은 파일**을 보므로 emit 단계의
  anchor/단위 변환 실수가 여기서 잡힌다.
- `results/20260815_magladder/` — 모델별 강도 사다리(3 scale × 3 camera × 4 model = 36 렌더)와
  `measure/all.json`. `results/20260816_magmatch/` — 그 사다리로 역산한 scale 의 검증 렌더 +
  recammaster/trajectorycrafter 신규 사다리 + alayaworld 포화 probe. 둘 다 `config.md` 동봉.
- `measure_realized.py --profile` — 프레임별 **누적** dx/W, dy/W, zoom 궤적을 JSON `curve` 키에
  담는다 (기본 꺼짐 = 기존 출력 그대로). 총 이동량이 같아도 시간 분포(=속도)가 다를 수 있어서
  총량만으로는 "같은 세기"인지 판정이 안 된다. 이걸로 alayaworld 만 프레임당 속도가 절반이고
  앞 20% 가 늘어진다는 것을 잡았다.
- `make_warp_compare.py --rows N` — 패널을 N 줄로 접는다 (기본 1 = 기존 한 줄 동작).
  패널이 7개쯤 되면 한 줄은 너무 납작해서 안 보인다. 줄마다 총폭이 다르면 오른쪽을 검정 패딩
  (짝수 폭 유지 — 홀수면 libx264 가 조용히 죽는다).
- `run_grid.py --aw_rounds N` (기본 3 = 기존 동작) — alayaworld 롤아웃 round 수. 출력 프레임 =
  `rounds*32` 라 클립 길이를 이걸로 맞춘다. `emit_model_cams.py --n_frames` 도 같이 바꿔야 한다
  (`run_grid.py` 상단 길이 산수 주석 참조).
- `results/20260816_warp3/` — warp 3종(sierp/trajC/alaya)을 dx/W = 0.22 로 올리고 클립 길이를
  맞춘 렌더. sierp 0.2215 / trajC 0.2287 로 도달, **alayaworld 는 0.115 에서 포화**
  (s = 1.8 / 3.6 / 5.85 → 0.1110 / 0.1150 / 0.1149). `config.md` + `measure/all.json` 동봉.
- `run_grid.py --aw_cfg <yaml>` (기본 None = 저자 기본 `configs/infer.yaml` = 기존 동작) —
  alayaworld 에 다른 inference yaml 을 `--cfg` 로 넘긴다. vendored 파일은 안 고치고 config 를
  복사해서 쓴다 (`configs/infer.yaml` 의 `paths:` 는 cwd 기준 상대경로라 yaml 위치는 무관).
  쓸모: **`da3_align_to_input_scale: false`**. 기본값 `true` 면 DA3 가 추정한 depth 를 **우리가
  준 extrinsic 의 스케일에 맞춰 다시 스케일**하므로 (`flash_alaya/alaya/memory/da3_depth.py:119`
  → `spatial.py:116` → `schema.py:160`) translation 을 2배로 줘도 depth 가 같이 2배가 되어
  시차 `t/Z` 가 불변 = **emit scale 이 아예 안 먹는다.** 이게 "alayaworld 포화" 의 실체다.
- `results/20260816_warp3/cfg/infer_noalign.yaml` — 위 플래그만 `false` 로 바꾼 config 사본.
  `out_noalign/` (s 0.9/1.8/3.6) + `out_noalign_small/` (s 0.11/0.22/0.44) 렌더가 이걸 쓴다.
  같은 s=0.9 / 같은 64 프레임에서 warp dx/W 0.0733 → **0.8974**, s=1.8 에서 0.1123 → **1.5393**
  으로 scale 이 정상 반응한다. 다만 그 크기에서는 warp 이 구멍투성이가 되어 생성기가 정지하므로
  (출력 dx/W 0.0042 / 0.0001) 사다리를 s ≈ 0.1~0.4 로 내려서 8점 찍었다. **다만 alayaworld 는
  출력이 s 에 대해 단조가 아니다** — s 0.27 -> 0.28 (+3.7%) 에서 dx/W 가 0.2283 -> 0.1723 으로
  25% 떨어진다. warp 은 0.1344 -> 0.5107 로 완벽히 단조인데 출력만 튄다 (출력/warp 0.508~0.964,
  seed 고정). 따라서 **alayaworld 는 아직 0.22 에 정합됐다고 볼 수 없다** (sierp 0.7218/0.2215,
  trajC 2.086/0.2287 은 정합 완료). `config.md` 에 8점 사다리 + 재정정 동봉.
- `run_grid.py --gen_seed N` (기본 None = seed 를 안 넘김 = 저자 기본 sierp 42 / trajC 43 /
  infcam 0 / cameraanything 0 = 기존 동작) — sierpinskicam / trajectorycrafter / **infcam /
  cameraanything** 의 생성 seed. **실현 이동량의 seed 분산**을
  재려고 넣었다. alayaworld 는 같은 scale·같은 클립 길이에서 seed 만 바꿔도 dx/W 가
  s=0.27 에서 0.2119~0.2532 (mean 0.2339, sd 6.6%), s=0.28 에서 0.1723~0.2479
  (mean 0.2013, sd 14.0%) 로 퍼진다. 두 구간이 크게 겹치므로 앞서 "s 0.27->0.28 에서 25% 하락 =
  비단조" 라고 본 것은 **seed 0 한 번의 draw** 였다 (t≈1.8, n=4 씩, 유의하지 않음).
  sierp/trajC 도 같은 잣대로 seed 4개씩 재측정한 결과 **정합이 확정됐다**:
  sierp 0.7218 → 0.2215/0.2252/0.2215/0.2219 (mean 0.2225, sd **0.82%**),
  trajC 2.086 → 0.2287/0.2301/0.2270/0.2297 (mean 0.2289, sd **0.50%**).
  alayaworld 의 sd 만 이 둘의 8~17 배다.
- **판정: alayaworld 를 강도(dx/W) 정합 축에서 제외한다.** 사용자 기준이 "생성한 카메라 궤적을
  정확히 실행" 이므로 벤치마크 공정성이 아니라 궤적 실행 정확도로 판정했다. ① 출고 config 는
  카메라 크기를 무시(s=1.314 → 0.2543 vs s=2.0 → 0.2541), ② 정렬을 꺼도 고정 s 에서 sd 6.6~14.0%,
  ③ 실효 s 구간이 0.1~0.4 로 좁고 밖에서는 생성 정지/warp 붕괴, ④ joystick 분기
  `camera_control.py::c2w_to_action_labels` 가 c2w 를 WASD one-hot 으로 이산화해 **방향만** 받는다
  (게다가 우리 세팅은 I2V). 소스를 169 프레임 이상으로 바꿔 V2V 로 만들어도 ②③④ 는 남는다.
  정성 비교용("방향 컨트롤러")으로만 유지. `results/20260816_warp3/config.md` 에 최종 표 동봉.
- `results/20260816_warp3/compare_alaya_align.mp4` / `compare_warp3_matched.mp4` — 2줄
  (위 출력 / 아래 depth warp) 비교 영상. 전자는 alaya 정렬 ON vs OFF, 후자는 정합된 3종.
- `results/20260816_match22/` — infcam / cameraanything 를 dx/W 0.22 에 정합 시도 +
  **전 모델 seed 분산을 n=8 로 통일**. 결론: **precise camera control 후보는
  sierpinskicam / trajectorycrafter 2종만 남는다.**

  | 모델 | s | mean dx/W | 목표대비 | seed sd (n=8) | max/min | 구조 |
  |---|---|---|---|---|---|---|
  | sierpinskicam | 0.7218 | 0.2242 | +1.9% | **1.03%** | 1.024 | warp+inpaint |
  | trajectorycrafter | 2.086 | 0.2306 | +4.8% | **1.77%** | 1.052 | warp+inpaint |
  | infcam | 0.6623 | 0.2501 | +13.7% | 17.94% | 1.748 | 카메라 조건부 생성 |
  | cameraanything | 0.82 | 0.2462 | +11.9% | 17.73% | 1.916 | 카메라 조건부 생성 |
  | alayaworld | 0.27 | 0.2339 (n=4) | +6.3% | 6.62% | — | 카메라 조건부 생성 |

  **분류선이 모델 구조를 그대로 따라간다** — 카메라를 기하로 먼저 실행하고 생성기엔 구멍만
  메우게 하는 쪽은 sd 1~2%, 카메라를 생성기 조건으로 넣는 쪽은 6~18%. 두 그룹 사이에 겹치는
  값이 없다. cameraanything 은 dy/W 가 8개 seed 전부 양수(mean +0.0070)라 순수 좌평행이동인데
  위로 밀리고 zoom 도 0.99~1.15 로 흔들린다 — 이동량뿐 아니라 방향도 어긋난다.
  n=4→n=8 로 늘리자 sd 가 셋 다 올랐다 (sierp 0.82→1.03, trajC 0.50→1.77, infcam 12.70→17.94)
  — **n=4 수치는 전부 낙관적이었다.**

- `tools/recammaster/ctx_align.py` (신규) — **context align**. depth warp 모델의 이동량을
  "화면폭 대비 얼마나 움직일 것인가"(`target dx/W`)로 지정하고, 그 값이 나오는 translation
  배율 `k` 를 **모델 자신의 context depth 로 수치 역산**한다. `predict_flow` 가
  (depth, K, rel c2w) 로 median flow 를 직접 계산하고 `solve_scale` 이 bisection 으로 푼다.
  `--selftest` 5 케이스 (planar truck 해석해 대조 / spread depth vs `D·z_med/(fx/W)` /
  +15° yaw / 회전만으로 목표 초과 시 SystemExit / 회전·병진 부호 반대인 비단조 케이스).
  왜 해석식이 아니라 수치해인가: 회전이 섞이면 `k = D·z_med/(fx/W)` 가 **55.6% 틀린다**
  (selftest 3). 그리고 `|dx(k)|` 는 회전 기여와 병진 기여의 부호가 반대일 때 비단조라
  bisection 을 부호 붙인 `g(k)=sgn·dx(k)` 위에서 돌린다.
- `emit_model_cams.py --target_dxw / --align_npz / --align_fx_over_w` (기본 None = 기존
  native 1x 동작) — sierpinskicam / alayaworld 카메라를 native 배율 대신 context align 으로
  낸다. depth 는 SierpinskiCam 이 남긴 DA3 npz 를 쓰고, `fx/W` 는 모델마다 다르므로
  (sierp 0.3578 = DA3 추정치, alaya 0.4482 = 우리가 넘기는 정규화 K) 후자만 명시로 준다.
  `manifest.json` 에 `ctx_align` 블록(z_med, per-camera k / rot_only / pred dx/W / tau)을 남긴다.
  trajectorycrafter 는 depth 가 DepthCrafter 클립 min-max 정규화라 여기서 못 풀고 raise.
- `run_trajcrafter_matrix.py --target_dxw` (`--camera matrix` 전용, 기본 None = 기존 동작) —
  trajectorycrafter 의 `radius` 를 DepthCrafter 자기 depth 로 **런타임에** 역산한다.
  `run_grid.py --tc_target_dxw` 로 넘긴다.
- `tools/recammaster/sierp_run_freeanchor.py` (신규) + `run_grid.py --free_anchor` (기본 꺼짐 =
  기존 동작) — trajectorycrafter / sierpinskicam 이 `rel[0] != I` 궤적의 상수 offset 을 살린다
  (`get_c2w` monkey-patch). `emit_model_cams.py --hold` probe 짝.
- `emit_model_cams.py --hold / --anchor` (기본 꺼짐 / auto = 기존 동작) — 궤적을 끝점에서 정지한
  **상수 offset** 으로 바꾼다. "context 첫 view 가 아닌 곳에서 렌더가 되는가" probe 용.
- `tools/recammaster/measure_offset.py` (신규) — **소스 프레임 k ↔ 출력 프레임 k** flow 로
  offset 이 실현됐는지 잰다. `measure_realized.py` 는 클립 **안**의 누적만 봐서 상수 offset 을
  0 으로 읽는다. `--warp` 로 depth warp 중간물도 같은 방식으로 잰다.
- `measure_offset.py --method sift` (기본 `farneback` = 기존 동작) — SIFT + Lowe ratio + MAD
  필터로 대응점을 잡는다. Farneback 은 winsize 41 (width 512) 로도 수십~백 px 짜리 offset 이
  capture range 밖이면 **0 으로 수렴**해서 "offset 이 안 실현됐다"와 구별이 안 된다. 실제로
  `/tmp/off_full.json` 이 전 모델 off_dx≈0 으로 읽혔는데 픽셀 L1 diff (infcam 16.4→57.3) 와
  모순이었고, sift 로 다시 재자 결과가 뒤집혔다.
- `ctx_align.solve_scale(metric='max')` (기본 `'end'` = 기존 동작) — 목표 `dx/W` 를 끝 pose 가
  아니라 **렌더되는 pose 전체의 |dx/W| 최댓값**에 맞춘다. `dl3dv_L81_a_r` 은 나갔다 돌아오는
  궤적(path/chord 2.18)이라 화면 변위가 중간에 −0.538 까지 갔다 끝에서 −0.110 으로 돌아온다 —
  끝점만 맞추면 "세 모델이 같은 양 움직인다" 가 성립하지 않는다. 단조 궤적에서는 두 metric 이
  같은 답을 준다 (selftest 5 가 이걸 assert 한다). 반환값에 `peak_idx` / `end_dxw` 추가.
  `--metric` / `--n_poses` CLI 플래그도 같이 (`--n_poses` 는 모델이 실제 렌더하는 앞 N pose 만
  쓰기 위한 것 — SierpinskiCam 은 JSON 키를 `[::4]` 가 아니라 **순서대로** 49개만 읽는다,
  `create_sierpinskicam_conditioning.py:651`, 즉 81키 궤적의 앞 60% 만 실행한다).
- `emit_model_cams.py --rcm_render_frames N` (기본 None = 기존 동작) — rcm 계열 JSON 에서
  모델이 **실제로 읽는** 키 개수에 궤적을 펴고 나머지 키는 마지막 pose 로 유지한다.
  SierpinskiCam 은 `--frame-count 49` → `range(49)` 로 **앞 49키만** 읽으므로 81키에 펴면
  궤적의 60% 지점에서 끊긴다 (실측: dynA end |t| 0.5475 → 0.3010 = 54.6%, static
  4.1322 → 3.3706 = 81.6%). 실측 배율 표(20260816 계열)는 이 절단을 배율 안에 흡수했으므로
  **예측 배율을 쓰는 경로(full-recon sim3 전이)에서만** 켠다.
- `emit_model_cams.py --aw_intr FX_W FY_H` (기본 None = 저자 예제 값 `(0.4482, 0.7966)` 고정
  = 기존 동작) — alayaworld `_camera.pt` 의 정규화 intrinsic 을 소스의 실제 FOV 로 바꾼다.
  AlayaWorld 는 고정 FOV 96도 모델인데 소스는 다르다 (DA3 추정 fx/W: dl3dv_s1 0.5006 = 90도,
  dynrep_s2 0.5501 = 84도). 넣으면 spatial memory 의 warp 기하는 소스와 맞지만 카메라 임베딩이
  학습 FOV 를 벗어난다 — 그래서 두 팔(`gen_aw_nat` / `gen_aw_srck`)을 따로 돌린다.
- `tools/recammaster/sierp_run_freeanchor.py --anchor0` / `run_grid.py --sierp_anchor0`
  (둘 다 기본 꺼짐 = 기존 동작) — SierpinskiCam warp 의 **출발 pose(DA3 extrinsics)를 ctx
  frame0 으로 재앵커**한다. vendored 코드가 `extrinsics[0] == I` 를 가정하는데 **DA3 는
  reference view 를 자기가 고른다** (실측 ref index: dynA 2/6, static 42/35 — static 은 frame0 이
  74.7° / |t| 5.71 어긋나 있었다). vendored 파일 0줄 수정, `smooth_camera_path` 를 monkey-patch.
  (`FIX.log` FIX-9)
- `tools/recammaster/halfsplit_source_mp4.py` (신규) — half-split 의 **context 절반을 역재생**해서
  소스 mp4 로 굽는다. `recon/*.json` / `prior_rev/*.json` 이 남긴 `frames_dir` + `frame_idx` 를
  그대로 읽으므로(`--from_json`) DA3 prior 를 뽑을 때와 **정확히 같은 프레임 집합/순서**가
  보장된다 — 다르면 모델 런타임 depth/카메라가 prior 와 달라져 sim3 로 푼 `s` 가 안 맞는다.
- `run_grid.SOURCES` 에 `hs_dynA` / `hs_static` 추가 — full-recon 전이용 half-split 소스
  (49 프레임뿐이라 81 프레임을 요구하는 recammaster/infcam 은 못 돌린다).
- `tools/recammaster/halfsplit_compare_grid.py` (신규) — `halfsplit_fullrecon_transfer.py` 가 낸
  모델별 mp4 를 가로로 이어 `compare_<src>.mp4` 를 만든다 (각 열 = 모델, 각 열은 이미
  위 = 전이 warp / 아래 = 정답 2행). `results/20260817_fullrecon_fix8/compare_*.mp4` 와 같은 배치.

### Changed
- `run_grid.py` 의 `_outputs()` 가 `source.mp4` / `point_cloud.mp4` / `point_cloud_masks.mp4`
  **파일 이름**도 제외한다. Vista4D stage 3 은 denoise **전에** 이 셋을 out_dir 바로 아래에
  쓰므로(`scripts/inference/inference.py:144-146`) 안 빼면 생성이 죽어도 `--skip_done` 이
  "완료"로 본다. 결과는 `video_seed=*.mp4` 뿐이다.
- `run_grid.py` 가 `--out_root` 를 **절대경로로 정규화**한다. 러너마다 cwd 를 자기 repo
  루트로 바꿔 돌기 때문에 상대경로를 주면 결과가 그 repo 안에 떨어진다.
- `warp_grid_video.py` 의 `tag_of()` 가 manifest 가 없으면 빈 dict 를 돌려주고 `camera`/`cam`
  두 키를 모두 받는다. `--rows` 로 sierp/trajc 아닌 모델만 깔 때 죽지 않게 (vista4d manifest 는
  키가 `cam`).
- `run_grid.py` 의 `_outputs()` 가 `_warp/` 를 결과 mp4 집계에서 **제외**한다. 안 그러면
  `--skip_done` 이 warp 만 있고 본 결과가 없는 디렉토리를 "완료"로 보고 재생성을 건너뛴다.
- `warp_ladder.sh` 의 소스/출력을 **환경변수로 갈아끼울 수 있게** 했다 (`P` / `SRC` /
  `AWIMG` / `SCENE` / `CAMERA`). 안 주면 기존 cleaning_stove probe 동작 그대로다.
  cleaning_stove 말고 다른 소스로 배율 사다리를 돌리려고 필요했다.

### Fixed
- **`rerope_prepare.py` 가 소스 npz 를 날 c2w 로 써서 ReRoPE 내부 pose 가 축 순환 치환됐다.**
  `--axis_precomp {inv_conv(기본), none(구 동작)}` 추가. 아래 `--target_frame world` 로 원점을
  고친 뒤에도 **카메라가 엉뚱한 축으로 움직였다**. `convert_c2w_convention` 은
  `X -> X @ P`, `P = [[0,0,1],[1,0,0],[0,-1,0]]` (det −1) 인데 이게 **소스 경로에도**
  걸린다(`v2v_handler.py:124`). 소스에 DA3 OpenCV c2w 를 그대로 넣으면 내부 pose 가 전부
  `P⁻¹(·)P` 로 켤레변환되고 요청 이동 `t` 가 `P⁻¹t` 로 실현된다. sC 누적 optical flow
  실측이 순수 이동 4종 **전부** 예측과 일치했다:

  | 카메라 | 요청 로컬 t | 예측 `P⁻¹t` | 실측 누적 flow |
  |---|---|---|---|
  | truck_left | −x | −z (zoom) | du ≈ 0.0 (zoom) |
  | pedestal_up | −y | −x (좌우) | du +83.7 |
  | dolly_fwd | +z | −y (상하) | dv +8.7 |
  | dolly_back | −z | +y | dv −38.5 |

  `inv_conv` 는 소스·타깃 **양쪽 on-disk 에 `conv` 의 역**(`col1 부호반전` → `cols[2,0,1,3]`)을
  미리 걸어 ReRoPE 안에서 `conv` 를 항등으로 만든다. 새 `[축검증]` 단계가 내부 타깃 상대 w2c 가
  JSON 로컬 궤적과 같은지 확인한다 (sC max err **5.4e-07**). 대가로 on-disk det 가 배포
  npz(+1)와 반대인 −1 이 되는데 이론적으로 배포 npz 를 설명하지 못한다 — **판정 근거는 flow
  실측**이다. `ue5_asis` 와는 배타(`ap.error`).
  주의: **자기-타깃 검증(타깃 = 소스 궤적)은 이 버그를 못 잡는다** — conjugation 은 소스=타깃일
  때 상쇄된다. 아래 항목의 "검증" 이 통과했는데도 축이 틀려 있던 이유가 이것이다.
- **`rerope_prepare.py` 가 타깃 궤적을 UE5 JSON 그대로 넘겨서 ReRoPE 가 카메라를 못 따랐다.**
  `--target_frame {world(기본), ue5_asis(구 동작)}` 추가. 1차 생성물은 frame0 부터 소스와 다른
  장면이었고, 통제 실험으로 게이지(`--divisor 1`)·항등 타깃·프레임 길이(데모 nf81 vs nf49) 를
  전부 무죄 처리한 뒤 타깃 규약 두 군데를 찾았다:
  1. **det 부호.** 배포 npz 는 소스든 타깃이든 `convert_c2w_convention` **이후** rotation
     det = **−1** 이다. on-disk 는 proper c2w 고 `conv` 가 축을 뒤집는 게 정상 경로인데,
     `opencv_to_ue5` 는 flip 을 미리 넣어 놨으므로 `conv` 가 그걸 **되돌려** det +1 이 되고
     소스(−1)와 타깃(+1)이 다른 규약으로 들어갔다.
  2. **좌표 원점 (영향이 더 크다).** canonical JSON 은 frame0 = 항등인 **카메라-로컬 상대**
     궤적인데 ReRoPE 는 타깃을 **소스** frame0 기준으로 상대화한다(`v2v_handler.py:178`).
     로컬 궤적을 그대로 주면 소스 frame0 의 world rotation 이 상수 offset 으로 남는다 —
     sC 실측 **117.11°, 전 프레임 동일**.
  `world` 는 `Tw = C_src0 @ conv(json.T)` 로 심고 `Tw.transpose(0,2,1)` 만 해서 저장한다.
  검증: 타깃 = **소스 궤적 자신**(비자명 궤적)을 넣으면 출력이 소스를 재현한다.
  진단 표가 `tgtrot0 117.11 → 0.00` / `tdet +1 → −1` 로 바뀌고, 순수 이동 카메라 4종은
  `tgtrotmax = 0.00`, dl3dv 궤도 2종만 제 회전(61.69° / 77.56°)을 갖는다.
- **DA3 `Prediction.extrinsics` 를 c2w 로 읽고 있었다 — 실제로는 w2c 다.** `da3.py` 가
  `output.extrinsics = affine_inverse(c2w)` 를 넣고 `export/ply.py:156` 이 `# w2c` 라고 적어 뒀다.
  카메라 이동 방향이 정반대로 들어가고 있었고, 회전이 작으면 umeyama 스케일만은 얼추 살아남아
  (`|Δ(-R·C)| ≈ |ΔC|`) 지금까지 안 들켰다. `halfsplit_fullrecon_transfer.py --da3_extr` /
  `halfsplit_sweep.py --da3_extr` 추가 (기본 `w2c` = 수정본, `c2w` = 기존 동작 재현),
  `halfsplit_recon_da3.py` 규약 주석 정정. npz 포맷은 안 바꿨다 (읽는 쪽에서 `inv`).
  영향: full-recon 전이 `PSNR_cov` dynA 19.16 → **27.24**, DL3DV static 12.28 → **16.20**,
  잔여 배율 `f*` 0.5~0.6 → **1.00**. (`FIX.log` FIX-8)
- **SierpinskiCam 의 depth warp 출발 pose 가 DA3 world 원점이라 우리가 넣은 궤적이 통째로
  어긋나 있었다.** `Warper.forward_warp` 는 `T = pose_t @ inv(pose_s)` 인데 vendored 코드가
  `pose_s` 로 raw DA3 `extrinsics` 를, `pose_t` 로 frame0=I 상대화된 `traj` 를 넘긴다 →
  `i=0` 에서 `T = inv(extrinsics[0])`. DA3 는 frame0 을 원점으로 두지 않으므로 static 클립에서
  첫 프레임부터 74.7° / |t| 5.71 이 얹혔다 (dynA 는 2.60° / 0.1078 = 궤적 rmax 의 19.7%).
  `--sierp_anchor0` 로 켜는 재앵커 래퍼를 추가했다 (기본 꺼짐 = 기존 동작). TrajectoryCrafter
  (`pose_s` 고정)와 AlayaWorld (`num_context_frames:1`, 우리 `cam_c2w` 를 DA3 입력으로 사용)는
  영향 없음. (`FIX.log` FIX-9)
- **`halfsplit_recon_da3.py --gt_check` 가 FIX-8 이후에도 extrinsics 를 c2w 로 읽고 있었다.**
  DA3 는 w2c 라 `[:3,3]` 이 `-R·C` 인데 그대로 GT c2w 와 대조해서 회전 오차가 늘 175~180° 로
  찍혔다 — 규약 검증용 진단인데 규약을 못 가른다. 실측(신규 DL3DV `212b6928`): as-is 면
  resid 0.0927 / rot_med 174.7°, `inv` 면 **resid 0.0056 / rot_med 1.1°**. `--gt_check_extr
  {w2c(기본),c2w}` 추가 (`c2w` = 옛 동작 재현). 진단 출력만 바뀌고 저장되는 npz 는 그대로다.
- **`measure_realized.py` 가 alayaworld 의 결과 대신 depth warp 을 재고 있었다.** `SKIP_DIRS` 에
  `_warp` 가 없었고 `find_output` 이 `max(getsize)` 로 고르는데 warp(3.2 MB)이 본 결과(268 KB)보다
  커서 항상 warp 이 뽑혔다. 프레임 수로 구분된다 (warp 75 vs 출력 96). `_warp` 를 `SKIP_DIRS` 에
  추가하고 magladder 의 alayaworld 6점을 전부 재측정 — 결론(포화 아님)은 안 바뀌었고 적합만
  p 1.148→1.109, A 0.1371→0.1381 로 이동. (`FIX.log` FIX-10)
- **`emit_model_cams.py --n_frames` 를 alayaworld 에 잘못 넣어 궤적 뒷부분이 버려지고 있었다.**
  이 인자는 **움직이는 pose 개수**이고 manifest 의 `n_frames` 는 거기에 `--aw_prefix` 를 더한
  값이다. magmatch 에서 `--n_frames 233 --aw_prefix 129` 를 줘 pose 를 362 개 깔았는데 영상은
  233 프레임이라 뒤 129 개가 안 쓰였다 (canonical 의 앞 44% 만 실행). truck_left 가 등속 직선이라
  실현 이동량은 8.21·s vs 8.178·s 로 거의 같아 magmatch 의 적합·최종 scale 은 유효하지만,
  곡선 궤적이면 모양이 잘린다. `results/20260816_warp3` 부터 `--n_frames = 생성 프레임 수`
  (rounds=2 → 72) 로 바로잡았다.
- **`warp_ladder.sh` 의 tag 생성이 `emit_model_cams.py` 와 어긋나 파일을 못 찾았다.**
  emit 쪽은 `f'{s:g}'`, ladder 쪽은 `sed 's/\./p/'` 라 `4.890` 이 각각 `4p89` / `4p890` 이
  됐다. ladder 를 `awk '%g'` 로 바꿔 규칙을 맞췄다 (끝자리 0 이 없는 scale 은 영향 없음).
- **`make_warp_compare.py` 가 패널 총폭이 홀수면 빈 mp4 만 남기고 조용히 죽었다.**
  libx264/yuv420p 가 홀수 폭을 못 쓰는데 에러 메시지가 안 나온다. `read_frames` 에서
  패널 폭을 짝수로 내림한다.
- **`halfsplit_warp.project` 가 투영 좌표를 `long()`(=floor)로 정수화해 항등 warp 에서도
  hole 이 17.9% 뚫렸다.** `(u-cx)/fx*z -> fx*(x/z)+cx` 왕복에서 4.0 이 3.99999.. 가 되면
  floor 가 3 으로 내려보내 픽셀 4 가 빈다. `torch.round` 로 바꾸고 범위 검사도 반올림 좌표로.
  항등 쌍 hole 0.1794 -> **0.0000** (PSNR inf). 이전에 잰 half-split hole/PSNR 수치는
  전부 hole 이 부풀어 있었다 — 전 구간 재측정함. (`FIX.log` FIX-4)
- **`--pair mid` 의 첫 쌍이 `(48,49)` 라 첫 프레임부터 시차가 있었다.** 만나는 지점
  `m=nc-1` 이 ctx/tgt **양쪽의 첫 프레임**이어야 실제 추론(소스 첫 프레임 = 타겟 rel[0]=I)과
  같다. `dst` 를 `m` 부터 시작하도록 고쳐 첫 쌍이 `(48,48)`, 프레임 간격이 0,2,4,... 대칭이
  됐다. `aligned` 은 그대로. (`FIX.log` FIX-5)

### Changed
- **영상 저장은 전부 imageio + libx264 (h264)로.** `cv2.VideoWriter` 의 `mp4v` 는
  MPEG-4 Part 2 라 **VS Code 내장 뷰어에서 안 열린다** (파일은 멀쩡하고 재생만 안 되는 종류).
  `halfsplit_common.write_mp4(path, frames, fps, rgb=True)` 헬퍼를 추가하고
  `halfsplit_sweep.py` / `halfsplit_warp.py` / `make_warp_compare.py` / `make_sbs.py`
  네 곳의 `cv2.VideoWriter(mp4v)` 를 여기로 돌렸다. (`make_compare.py`,
  `alaya_warp_only.py` 는 이미 libx264 였다.)

### Added
- `halfsplit_pair_concat.py` 가 **모델 depth prior 를 여러 개** 받는다 (모델 하나만 줬을 때의
  기존 동작은 그대로). `--model_depth 이름=경로 ...` 로 나열하고 변형에서 `mid:per@sierp`
  처럼 `@이름` 으로 고른다 (`@` 생략 시 첫 모델). **f 는 모델마다 따로 잡는다** — prior
  스케일이 모델마다 다르니 하나의 f 를 공유하면 비교 자체가 성립하지 않는다. pose 를 안
  내놓는 모델(DepthCrafter)은 `--ref_depth` 로 depth 중앙값 비 예측으로 떨어지고,
  `--f trajc=<숫자>` 로 모델별 수동 지정도 된다.
- `halfsplit_prepare.py --pos_is_t` (기본 꺼짐 = 기존 동작) — dynrep `cameras.json` 의
  `position` 을 카메라 중심이 아니라 **w2c 의 t** 로 해석해 중심을 `C = -R^T·position` 으로
  만든다. **dynamic_replica 가 이쪽이다** (`FIX.log` FIX-7). 틀리면 가로 이동 방향만
  뒤집히고 크기는 얼추 맞아서 PSNR 로는 안 잡힌다.
- `tools/recammaster/halfsplit_pair_concat.py` (신규) — 여러 렌더 조건을 **같은 GT 프레임
  위에서** 한 영상에 나란히 붙인다. `--variants pair:depth_src[:depth_idx]` 를 나열하면 각
  변형의 `dst` 교집합만 골라 정렬한 뒤 `[ctx | warp1 | warp2 | ... | GT]` 로 concat 한다.
  pairing 이 다르면 `dst` 가 다르므로(`mid` 48..96 vs `aligned` 49..97) 영상 두 개를 그냥 옆에
  붙이면 **오른쪽 GT 패널이 서로 다른 프레임**이라 비교가 성립하지 않는 걸 막는다.
  `--ctx_panel vid` 는 ctx 절반을 정재생 원본으로 트는 패널(변형이 앵커 한 장만 써도 입력
  영상이 뭐였는지 보이도록). `--f` 기본값은 `pred` (ctx-only umeyama).
- `halfsplit_sweep.py --depth_idx <int>` (기본 `None` = 기존 동작) — `--depth_src first|anchor`
  가 쓸 앵커 프레임의 **원본 인덱스**를 직접 지정한다. `--pair mid` 는 ctx 를 역재생하므로
  `src[0]` 이 프레임 **48**(양 절반이 만나는 지점)이다. 클립의 진짜 첫 프레임 depth 를 쓰려면
  `--depth_idx 0` 으로 명시해야 한다 — 헷갈리면 "첫 depth" 가 조용히 다른 프레임이 된다.
- `halfsplit_sweep.py --depth_src per|first|anchor` (기본 `per` = 기존 동작, 신규) —
  warp 에 넣을 소스 depth 를 고른다. `first` 는 **depth 만 앵커(`src[0]`) 것으로 고정**하고
  이미지·카메라는 프레임 자기 것을 쓴다 (DA3 의 프레임별 depth 흔들림이 원인인지 가르는 용도).
  `anchor` 는 이미지·depth 를 둘 다 앵커로 고정 — 앵커 한 장으로 전 타겟을 만드는 단일 시점
  NVS 배치다. **`anchor` 는 `--pair mid` 에서 baseline 이 절반**(쌍 간격 2k -> k)이라
  hole 이 줄어든다 — `per` 와 절대 PSNR 을 직접 비교하지 말 것.
  영상 파일명에 `_d<모드>` 가 붙고, 좌측 패널은 실제로 warp 에 넣은 소스 프레임으로 바뀐다.
- `tools/recammaster/halfsplit_diag.py` (신규) — "warp 이 타겟 카메라를 덜 쫓아간다" 가
  **depth 편향인지 정렬 실패인지 아니면 애초에 시차가 없는 건지**를 가른다. `halfsplit_sweep.py`
  는 전역 배율 `f` 하나만 훑어서 이 셋을 구분하지 못한다. 네 가지를 같이 잰다:
  (A) 타겟 프레임별 `|ΔC|`/회전각/예상 시차 `dx/W`, (B) **무-warp 기준선**(ctx 프레임을 그대로
  둔 PSNR — warp 이 이걸 못 이기면 그 소스엔 시차가 없다), (C) **프레임별 `f*`**(전역 scale
  오차면 k 에 무관하게 상수, 흐르면 drift), (D) **depth 3분위 대역별 `f*`**(순수 scale 오차면
  세 대역이 같은 값, 갈라지면 depth 에 affine 편향이 있어 전역 `f` 로 못 고친다).
  결과는 `<tag>_diag.json`.
- `tools/recammaster/reencode_h264.py` — 이미 쌓인 mp4v 결과물을 h264 로 일괄 재인코딩.
  코덱 판별을 **디코딩 없이** 한다 (mp4 `stsd` fourcc 를 앞뒤 1MB 에서 스니핑) — 전체
  디코딩으로 재면 1600개에 수십 분. 기본 dry-run, `--apply` 로 원본 덮어쓰기.
  홀수 해상도 대비 `pad=ceil(iw/2)*2` 포함.
  **적용 결과: `results/` mp4 1638개 중 mpeg4 78개 -> 전부 h264, 실패 0.**
- **half-split 스케일 캘리브레이션 도구 6종** (`tools/recammaster/halfsplit_*.py`) — 긴 영상을
  반으로 갈라 앞 절반(context)을 모델 depth 로 unproject 하고 뒤 절반의 **GT 카메라**로 재투영해
  뒤 절반의 **실제 프레임**과 PSNR 로 비교한다. 눈으로 맞추던 `dx/W=0.11` 앵커를 정답 픽셀이
  있는 측정으로 대체하는 게 목적.
  - `halfsplit_common.py` — `list_frames`/`pick_span`/`umeyama`(s,R,t,resid,extent)/`apply_sim3`/
    `depth_scale_ls`(median_ratio·ls·ls_affine)/`load_dl3dv_gt`.
  - `halfsplit_recon_da3.py` — DA3 recon -> npz(`extrinsics`/`intrinsics`/`depth`/`conf`/`images`).
    `da3` env + `HF_HOME` 필요.
  - `halfsplit_prepare.py` — GT 카메라 기준계 npz 생성 (`--kind dl3dv|dynrep`).
    `--probe_conv`/`--probe_gap`/`--probe_scales` 로 카메라 규약을 문서가 아니라
    **photometric 으로 가른다**.
  - `halfsplit_warp.py` — z-buffer forward splat (vendored `Warper` 미사용, CV/GL 플래그).
  - `halfsplit_sweep.py` — 이동량 배율 `f` 스윕. ctx 구간 umeyama 예측값과 photometric 최적값을
    나란히 찍는다.
    - `--pair aligned|mid` (기본 `aligned` = 기존 동작, 신규) — ctx/tgt 프레임 짝짓기.
      `aligned` 은 `ctx i -> tgt nc+i` 라 **모든 쌍이 항상 nc 프레임 떨어져 있고** ctx 첫 카메라와
      tgt 첫 카메라 사이에 상수 offset 이 남는다. 모델은 소스 첫 프레임을 항등으로 보고 상수
      offset 은 무시하므로 실제 추론 배치와 모양이 다르다. `mid` 는 **ctx 를 역재생**해
      `(48,49) (47,50) ... (0,97)` 로 짝지어 **가운데에서 만나게** 한다 (`scale_cams` 앵커도
      `src[0]` 로 이동). 결과: 2차 DL3DV PSNR_cov 14.330 -> 15.663 / hole 0.846 -> 0.771,
      static 소스 예측/f* 오차 5% -> **3.3%**. dynamic 은 변화 없음(짝짓기 문제가 아니었다).
    - `--common_support` (기본 off, 신규) — `f` 가 커질수록 hole 이 커져 **평가 화소 집합이
      `f` 마다 바뀌는** 편향을 없앤다. 전 `f` 에서 공통으로 채워지는 화소만으로 `psnr_fix` 를
      다시 재고 그걸로 `f*` 를 고른다. 안 주면 기존 `psnr_cov` 기준 그대로.
      **dynamic 소스에서 결론이 뒤집힌다**: dynamic_replica `0cf56b` 의 `f*` 가
      1.2058 -> 0.7723 (-36%), static DL3DV 는 1.0194 -> 1.0158 (-0.35%).
    - `--accum 1|k|all` (기본 `1` = 기존 동작, 신규) — 타겟 1장을 만들 때 쓰는 ctx depth 장수.
      벤치마크 6종이 여기서 갈린다: TrajectoryCrafter(`demo.py:66-79`)와
      SierpinskiCam(`create_sierpinskicam_conditioning.py:250-261`, TrajC 의 `Warper` 재사용)은
      **frame i -> frame i 1:1** 이고, AlayaWorld 는 `num_context_frames`(=10,
      `configs/infer.yaml:52`)장을 `Sparse3DCache` 에서 커버리지로 골라 z-buffer 로 합치며
      (`spatial_cache.py:576,785`), GEN3C 는 `frame_buffer_max` 장을 bilinear splatting 으로
      누적한다(`cache_3d.py:151,246`). `k` 는 `mid` 짝짓기에서 "지금까지 지나온 ctx 중 최근
      k 장", `all` 은 지나온 전부(누적 상한).
    - `--video_f best|pred|umeyama_cam|depth_ratio|<숫자>` (기본 `best` = 기존 동작, 신규) —
      `--save_video` 가 쓸 배율을 고른다. 지금까지 영상은 **photometric argmax(`f*`)** 로
      뽑혀 있었는데, `f*` 는 GT 뒤 절반을 봐야 구할 수 있어서 **실제 추론에는 없는 값**이다.
      파이프라인이 실제로 쓰는 건 ctx 구간만으로 낸 예측(`umeyama_cam`, 없으면
      `depth_ratio`)이므로 품질을 눈으로 볼 땐 `pred` 로 뽑아야 한다.
      `best` 가 아니면 파일명이 `<tag>_@<video_f>.mp4` 로 갈린다.
    - JSON 에 `at_pred` 추가 — 각 예측 `f` 에서 실제로 잰 `psnr_cov`/`psnr_all`/`hole`.
      f* 와의 **비율**만으로는 그 오차가 화질을 얼마나 깎는지 안 보인다.
      (실측 예: 2차 DL3DV alaya, 비율 0.995 = `PSNR_cov` -0.012 dB.)
- `halfsplit_warp.py`: `lift_to_world()` + `forward_splat_multi()` — 소스 여러 장의 월드
  point cloud 를 **하나로 합쳐 z-buffer 하나로 경쟁**시킨다. 소스별로 warp 해 순서대로
  덮어쓰면 뒤 소스의 먼 면이 앞 소스의 가까운 면을 지운다. AlayaWorld 의
  `cand_depth.view(S,N).min(dim=0)` 과 같은 규칙. `lift_to_world` 를 분리한 건 누적 warp 에서
  소스마다 unproject 를 타겟 수만큼 반복하지 않기 위해서.
- `halfsplit_recon_da3.py` / `halfsplit_depth_depthcrafter.py`: `--reverse` (기본 off, 신규) —
  프레임을 **역순으로 모델에 넣는다**. `--pair mid` 는 ctx 를 역재생하는데 prior 는 정방향으로
  뽑고 있었다. DA3 는 cross-view 라 reference view 선택이 입력 순서에 걸리고, DepthCrafter 는
  video diffusion + 영상 전체 min-max 정규화라 시간 방향이 그대로 먹힌다.
  저장은 **원래 프레임 순서로 되돌려서** 한다 (소비자는 배열 인덱스를 원본 프레임 번호로 읽는다).
  - `halfsplit_depth_depthcrafter.py` — TrajectoryCrafter 의 DepthCrafter prior 를 같은 npz 규격으로.
    `demo.py` 전처리(576x1024 하드코딩, 49 프레임, window110/overlap25/steps5/gs1.0)를 그대로 재현.
- `tools/recammaster/run_grid.py`: **다중 소스 영상 지원**. `SOURCES` 레지스트리 8개
  (`cleaning_stove`, ReCamMaster 예시 `rcm3`/`rcm5`, Taylor 긴 영상 클립
  `taylor_c01/c02/c05/c19/c23`) + `--src` 플래그. 결과 경로가
  `<out_root>/<model>/<tag>` -> `<out_root>/<src>/<model>/<tag>` 로 바뀌었다.
- `tools/recammaster/run_grid.py`: `aw_assets()` — AlayaWorld 가 요구하는 960x544
  (`rollout_utils.py:279`) 리사이즈본 + 첫 프레임 PNG 를 소스마다 자동 생성.
  종횡비가 다른 소스(848x480 / 1280x720 / 1994x1080)를 늘리지 않도록
  `scale=...:force_original_aspect_ratio=increase` + `crop`.
- `tools/recammaster/run_grid.py`: `--keep_going` (한 카메라 실패해도 계속),
  `--skip_done` (결과 mp4 있으면 건너뜀). 기본 동작(첫 실패에 중단)은 그대로.
- `tools/recammaster/queue.sh` — 소스 하나를 모델별로 순차 실행하는 GPU 큐 스크립트.
- `tools/recammaster/estimate_cond_cam.py` — CameraAnything cond 궤적을 DA3 로 추정.
  `--validate` 로 저자 트랙과 대조한 결과 회전만 맞고 스케일·focal 은 안 맞아서
  **새 소스에는 적용하지 않기로 했다**. 수치/근거는
  `results/20260813_camgrid_stage1/_condcam/README.md`.
- `tools/recammaster/emit_model_cams.py`: `--aw_prefix` — AlayaWorld 궤적 앞에 붙일 항등
  pose 개수. autoregressive prefix 129 프레임(`rollout_utils.py:119-127`)이 궤적 앞부분을
  history 로 먹어버리는 걸 막는다. canonical 세트를 `--n_frames 104 --aw_prefix 129` 로
  재생성했고 native 1x 총 크기가 6.352 -> **8.178** 로 바뀌었다
  (`results/20260813_canonical_cams/config.md` 갱신).
- `tools/recammaster/copy_camviz.py` — 각 결과 폴더(`<src>/<model>/<tag>/`)에 그 모델이
  **실제로 먹은** 카메라 궤적 그림 `camera_traj.png` 를 넣는다. canonical 이 아니라
  `run.json['entry']['path']` 의 emit 된 파일을 `verify_emitted.py` 의 모델별 로더로 다시
  읽으므로 native 1x 크기 / anchor 차이(InfCam mid, AlayaWorld 항등 prefix 129,
  TrajectoryCrafter depth 배수)가 그림에 그대로 보인다. 렌더는
  `<out_root>/_camviz/<model>/<tag>.png` 에 캐시하고 소스별 폴더로 복사.
  `make_camviz.draw_traj` 재사용 (그쪽 코드 수정 0줄).
- `tools/recammaster/make_compare.py` — 모델 간 가로 concat 비교 영상 2개를
  `<out_root>/_compare/<src>/<tag>/` 에 만든다 (`A_input-recam-infcam-cameraA.mp4`,
  `B_input-trajC-sierp-alaya.mp4`) + 같은 폴더에 `cameras.png` (모델별 궤적 그림 세로 스택).
  모델별 출력 길이가 81/49/96 으로 다르지만 카메라 궤적은 전부 canonical 21 pose 를 자기
  길이로 resample 한 것이라, **인덱스 비례 리샘플**로 81 프레임에 다시 깔면 같은 시각에
  같은 pose 가 보인다 (짧은 쪽 패딩은 이 성질을 깬다). input 패널은 그룹별로 모델이 실제로
  본 소스 구간 (A=첫 81, B=첫 49) 을 쓴다. 아직 안 끝난 모델이 그룹에 있으면 그 그룹은 보류.
- `tools/recammaster/run_trajcrafter_matrix.py`: `--render_only` — `demo.save_video` 를 감싸
  `render.mp4` (point-cloud warp) 직후 센티널 예외로 빠져나온다. CogVideoX-5B 샘플링을
  통째로 건너뛰므로 카메라 규약 검증이 싸다. warp 결과는 생성 자유도가 0 인 결정론적
  출력이라 규약의 ground truth 로 쓸 수 있다. vendored 코드 수정 0줄.
- `tools/recammaster/run_trajcrafter_matrix.py`: `--traj_matrix_conv {c2w,legacy}` —
  2026-08-14 이전의 잘못된 pose 주입 규약을 재현하는 옵션 (기본은 고쳐진 `c2w`).
- `tools/recammaster/measure_realized.py` — **생성 영상에서 realize 된 카메라 운동을 잰다**
  (dense Farneback flow 중앙값 누적 -> `dx`/`dy` 는 폭 대비 비율, `zoom` 은 평행이동을 뺀
  radial 성분 최소제곱). emit 된 파일을 다시 읽는 `verify_emitted.py` 는 "우리가 쓴 것 = 우리가
  읽은 것" 까지만 증명한다 — 모델 내부가 그 배열을 어떤 규약으로 소비하는지는 결과 픽셀로만
  알 수 있다. `--extra model=<상대경로>` 로 보관소 폴더(`_hud`, `_legacyconv`)를 강제로 볼 수 있다.
  GPU 불필요. cv2 있는 env 로 돌릴 것 (`latentcam` / `recammaster` / `infcam` / `trajcrafter`).
- `tools/recammaster/scene_scale.py` — 모델마다 단위가 다른 translation 을 무차원 축
  `tau = |t| / z_med` 로 환산한다. `dx/W ~= (fx/W) * tau` 로 픽셀 시차와 연결되고,
  geometry-locked 모델은 해석적으로, prior-locked 모델은 실측 캘리브레이션으로 구한다는
  구분과 그 근거를 docstring 에 적어 두었다.
- `tools/recammaster/alaya_warp_only.py` — **AlayaWorld 의 spatial-memory depth warp 만**
  돌려 mp4 로 낸다 (diffusion 0회). AlayaWorld 는 prior-locked 이 아니라 매 chunk 마다
  DA3 depth 로 bank 프레임을 unproject/reproject 하는 geometry-locked 부류인데
  (`flash_alaya/utils/spatial.py:217`), 그 warp 결과가 디스크에 안 남아서
  (`spatial.py:236` 에서 바로 VAE 로 간다) 비교가 불가능했다. vendored 코드 수정 0줄 —
  warp 함수와 DA3 래퍼를 그대로 import 해 `spatial.py:213-234` 의 호출을 재현하고,
  `inference/da3_patch.py` 의 monkeypatch 도 실제 추론 경로와 동일하게 건다.
- `tools/recammaster/warp_ladder.sh` — depth-warp 3종(TrajectoryCrafter/SierpinskiCam/
  AlayaWorld)의 warp-only 렌더를 `tau` 를 맞춘 사다리로 뽑는 드라이버.
- `tools/recammaster/make_warp_compare.py` — 그 렌더들을 가로 concat 으로 붙인다.
  해상도(1024x576 / 512x320 / 960x544)와 프레임 수(49/49/104)가 다르므로 높이만 맞추고
  **폭 비율은 유지**하며(시차를 눈으로 재야 한다), 길이는 인덱스 비례 리샘플로 맞춘다.
- `tools/recammaster/run_trajcrafter_matrix.py`: `[radius]` 진단 출력. `demo.py:515` 의
  `radius = min(center_depth * radius_scale, 5)` 가 npy 단위 그 자체인데(1x = 0.5*radius)
  소스마다 달라지므로 기록이 없으면 모델 간 translation scale 을 맞출 수 없다.
- `video_generation/FIX.log`, `video_generation/CHANGELOG.md` 신설.

### Changed
- AlayaWorld 를 `--no-joystick --seed 0` 으로 돌린다 (`AW_JOYSTICK` / `AW_SEED` 상수,
  `--aw_joystick` / `--aw_seed -1` 로 저자 기본 동작 복구). 조이스틱 HUD 는 생성물이 아니라
  VAE 디코드 후 cv2 로 덧그리는 오버레이라 (`inference/run.py:287-289`) 6개 모델 픽셀
  비교에서 AlayaWorld 에만 없는 물체가 얹힌다. seed 는 벤치마크 재현성용.
  기존 55개 결과물은 `<entry>/_hud/` 로 옮겨 보관하고 재생성한다.
- `_outputs()` 가 `_hud/` 를 결과로 세지 않는다 (`--skip_done` 이 재생성을 건너뛴다).

### Fixed
- `measure_realized.py` 가 `*_source.mp4` 를 결과로 오인하던 문제. CameraAnything 은 생성
  **전에** 입력 복사본을 그 이름으로 쓰므로, 중간에 죽은 run 에서는 그것만 남아 소스 영상을
  결과로 재고 "카메라가 안 움직였다" 는 오진이 났다.
- `results/20260813_canonical_cams/<model>/manifest.json` 6개 재생성. `emit_model_cams.main()`
  은 그 실행에서 만든 항목만으로 manifest 를 덮어쓰므로, 카메라 하나만 다시 emit 하면
  나머지 항목이 조용히 사라진다 (카메라 **파일**은 남아서 눈에 안 띈다).
  `--scales 0.5,1,2,4` 전 카메라로 복구했고 기존 83개 `run.json` 의 경로가 전부 살아 있음을 확인.
- **TrajectoryCrafter pose 주입이 w2c 자리에 c2w 를 넣고 있었다** (`run_trajcrafter_matrix.py`).
  `demo.py` 의 `c2w_init`/`poses` 는 이름과 달리 w2c(extrinsic) 다 — 이 배열이 그대로
  `Warper.forward_warp(..., transformation1, transformation2, ...)` 로 들어가고
  (`models/utils.py:235-236` docstring), `utils.py:316` 이 `T2 @ inv(T1)` 을 camera-1
  좌표계 점에 곱한다. `poses = P0 @ rel` -> `poses = inv(rel) @ P0` 로 고쳤다.
  옛 규약은 realize 되는 상대 변환이 `t -> (t_x, -t_y, t_z)`,
  `R -> diag(-1,1,-1) R^T diag(-1,1,-1)` 이라 **translation Y 와 yaw 가 반전**돼 있었다
  (X/Z 는 우연히 맞음). 옛 동작은 `--traj_matrix_conv legacy` 로 보존.
  `--selftest` 도 pose 배열 대신 **realize 되는 상대 변환**을 비교하도록 재작성
  (translation 6종 + orbit 2종, `max|diff| < 1e-7`). FIX.log 참조.
  기존 56개 결과물은 `<entry>/_legacyconv/out/` 로 옮겨 보관하고 재생성했다.
- `_outputs()` 가 `_legacyconv/` 를 결과로 세지 않는다 (`--skip_done` 이 재생성을 건너뛴다).
- `copy_camviz.py`: SierpinskiCam 궤적 그림이 **회전만 보이던** 문제 (`reanchor()`).
  emitter 가 프리셋 규약대로 월드 오프셋 `[5000,1500,100] cm = [50,15,1] m` 을 얹는데
  `draw_traj` 의 `cube_limits` 가 원점을 포함하므로 반경이 ~26 m 가 되어 1.8 m 짜리 이동이
  점으로 뭉개졌다. anchor pose(InfCam 은 mid, 나머지는 frame0) 를 원점으로 재고정하고
  그린다 — 상수 좌곱이라 프레임 간 운동(= `verify_emitted.py` 가 비교하는 양)은 안 변한다.
  제목에 재고정 여부와 `|t|` 를 표시.
- `run_alayaworld` 이 실행 전에 leftover `<prefix>_video.*` 를 지운다. v2v 시절 심볼릭 링크가
  남아 있으면 AlayaWorld 가 조용히 i2v -> v2v 로 되돌아갔다. FIX.log 참조.
- SierpinskiCam stage B 의 `prompts.json` 을 list 형식으로 (`KeyError: 0`). FIX.log 참조.
- SierpinskiCam stage A 의 `--trajectorycrafter-path` 를 `TrajectoryCrafter/models` 로.
- TrajectoryCrafter 의 CPU/CUDA device 불일치 (상류 diffusers offload 버그) 우회.
- CameraAnything 카메라를 실제 소스(`cleaning_stove`)의 cond 월드에 얹도록 재생성.
- emit 한 카메라 파일 경로를 절대경로로 (dangling symlink -> `FileNotFoundError`).

### Changed
- `run_grid.py:run_cameraanything` 이 소스의 cond 카메라(`examples/camera.json`)가 없으면
  즉시 종료한다. `inference.py:356` 이 소스 자신의 궤적 81 프레임을 읽기 때문에
  다른 영상의 궤적을 조용히 쓰는 사고가 가능했다.
