# watch.md — 지금 돌고 있는 것

**싱글톤이다.** 현재 상태만 남긴다 — 끝난 것은 지운다. 행이 사라지는 게 곧 "모니터를 껐다"는
뜻이어야 한다.

`tasks.md` 와 다른 축이다: tasks 는 **무엇을 할 것인가**(작업 dNNN 단위), 여기는 **지금 무엇이
돌고 있나**(프로세스 단위).

## 왜 파일인가

main session 은 컨텍스트가 압축되거나 세션이 끝나면 **자기가 무엇을 감시 중이었는지 잊는다.**
그 결과 2026-09-23 실측에서 감시자가 **221개** 살아 있었고, 그중 LIVE 는 **0개**였다 —
최대 36.5일 된 것까지 있었고 d84·d110·d151·d161·d185 처럼 수십 세대 전 작업을 보고 있었다.
감시 대상 로그 중 **106건이 `/tmp`** 였는데, 시스템이 그걸 비우면 모니터는 조용히 무력화되면서
본인은 계속 살아 있다.

한 번 더: 그때 프로세스를 `pgrep` 패턴 하나로 세니 69개, `ps` 전수로 세니 137개, 폴링 루프까지
합치니 154개였다. **패턴으로 세면 틀린다 — 걸 때 pid 를 적어 두는 것이 유일하게 맞는 방법이다.**

## 규칙

- 모니터를 **걸 때** 행을 추가하고, **끝나면 지운다.**
- `pid` 는 래퍼(`SCREEN`, `bash -lc`)가 아니라 **실제 python/셸 프로세스**를 적는다
  (`pgrep -af` 로 확인).
- 로그는 **`<repo>/tmp/` 아래**에 둔다. `/tmp` 는 시스템이 비운다.
- `정상이란` 은 **판정 가능한 문장**으로 쓴다. "잘 돌아감" 은 판정이 안 된다.
- 세 가지를 모두 적는다 — **살아있나 / 나아가나 / 끝났나.** `Traceback|OOM` 만 걸면
  **죽는 것만** 잡고 **멈춘 것은 못 잡는다.**

## 지금 돌고 있는 것

### R22 TRUMANS board source video 2x2 (D272, 807 chunk)
- screen  : r22b_render   pid: 2465272  GPU: 2 (worker 6, Cycles 16spp, D283 smooth_kf+pos kf6, D276 rgb --anim + D279 geom --anim + D278 probe 서버, GPU3 는 R39 VLM 에 비움)
- 로그    : <repo>/tmp/r22b/render_full.out (clip 별 tmp/r22b/log/, chunk 상태 tmp/r22b/status/)
- 살아있나: pid 존재 + GPU 2/3 에 blender 프로세스
- 나아가나: status/*.json 개수 증가 (pilot 6 에서 시작, 파일럿 속도 ~6 chunk/h @4 worker)
- 끝났나  : 로그 `[render] ALL DONE` → `--stage report` → 다음: d273 bank → raycast → caption → export
- 이상신호: Traceback / 1시간 status 정지 / complete 비율 급락

### R31 학습 dynpose_d200_molmo2_l21_da3_v12 (D274)
- screen  : train1        pid: 2670785        GPU: 1
- 로그    : <repo>/tmp/r31/train.log
- 살아있나: pid 존재 + GPU 1 점유 > 10 GiB
- 나아가나: tqdm `Epoch N` 증가
- 끝났나  : epoch_cap 50 → results/*_dynpose_d200_molmo2_l21_da3_v12/ckpts/last.pth
- 이상신호: Traceback / OOM / nan / 30분간 it 정지

### R49 VLM 판정 DynPose-100K 43,782편 (D282)
- screen  : vlm_r49 (클라이언트, workers 48, 프레임당 640x360 예산) + vllm_r49 (서버 GPU 3)   pid: 3314997
- 로그    : <repo>/tmp/r49/vlm.out, 서버 tmp/r41/vllm_r49.log
- 살아있나: pid 존재 + GPU 3 점유
- 나아가나: `[vlm] N/…` 200편마다 증가, out_filter/d282_dynpose100k/vlm/*.jsonl 줄 수
- 끝났나  : `[vlm] ALL DONE` → vLLM 내린다 (상시 기동 금지)
- 이상신호: fail 비율 급증 / 30분간 줄 수 정지

### R50 SAM3 9프레임 추적 (D282)
- screen  : sam3_r49_0..2   pid: 3382156 3382169 3382177   GPU: 0
- 로그    : <repo>/tmp/r49/sam3_{0,1,2}.out
- 나아가나: out_filter/d282_dynpose100k/sam3/*.npz 개수 증가 (VLM 뒤를 따라간다)
- 끝났나  : 셋 다 `ALL DONE` → judge → export → reel

