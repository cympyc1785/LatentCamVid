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
- screen  : r22b_render   pid: 3366008  GPU: 2 (worker 6, Cycles 16spp, D283 smooth_kf+pos kf6, D276 rgb --anim + D279 geom --anim + D278 probe 서버, GPU3 는 R39 VLM 에 비움)
- 로그    : <repo>/tmp/r22b/render_full.out (clip 별 tmp/r22b/log/, chunk 상태 tmp/r22b/status/)
- 살아있나: pid 존재 + GPU 2/3 에 blender 프로세스
- 나아가나: status/*.json 개수 증가 (pilot 6 에서 시작, 파일럿 속도 ~6 chunk/h @4 worker)
- 끝났나  : 로그 `[render] ALL DONE` → `--stage report` → 다음: d273 bank → raycast → caption → export
- 이상신호: Traceback / 1시간 status 정지 / complete 비율 급락

### R52 DynPose 확장 D284 — ⑤ 정적 SAM3 + geocalib (dynmask 7,143 완료 21:21)
- screen  : d284_s3s0..9 = 10샤드 GPU 0,0,2,2,3,3,6,6,7,7 (01:44 재분배, GPU 6,7 은 10:50 까지 한정 — screen gpu67_plan 이 그때 6,7 을 내린다, 로그 tmp/d284/gpu67_deadline.log). 로그 sam3s_logs/r10_shard*.log. geocalib 완료
- 로그    : <repo>/tmp/d284/sam3s_logs/shard{0..5}.log (GPU6/7 1차), shard{2..5}_gpu3.log, s3s_chain{A,B}.log
- 나아가나: eval_data/seg_instances_static/*/meta.json 신규 (21:20 이후) · out_dynpose/*/geocalib_gravity.json 신규
- 끝났나  : s3s screen 0 + static 7,139 / geo screen 0 + 7,143 → graph (run_bank d182, CPU) → dynmask --demote_static_objects → bank (d185+d199)
- 이상신호: Traceback / OutOfMemoryError (→ 그 샤드만 --skip_done 단독 재실행) / 30분간 증가 0

### R22 TRUMANS d277T 선행 graph+mesh (CPU, 3,440 board clip)
- screen  : d277T_prep0..11 (CPU, run_bank --stages graph,mesh --num_shards 12, OMP/MKL 4스레드 — 01:50 24→12 축소)
- 로그    : <repo>/tmp/r22b/banklog/d277T_prep12_s*.out, 씬 로그 tmp/r22b/banklog/d277T/
- 나아가나: out_trumans/<clip>/{.graph_s115, mesh_grid.npz} 증가 (smoke 1편 graph 209 s + mesh 67 s → 24샤드 ≈ 11 h)
- 끝났나  : 24 screen 종료 + 3,440 mesh_grid → GPU 단계 (cloud,route,tau,fit,emit; BANK_GPU_ALLOW=0123, 카드당 2)
- 이상신호: `FAIL` 줄 / Traceback / 30분간 증가 0

### R58 D286 학습 재개 (ViT feat 캐시, epoch 7 부터)
- screen  : train3        GPU: 1        wandb fbf4lhqi
- 로그    : <repo>/tmp/r58/train.log (append)   1.19 it/s (≈81 분/epoch)
- 살아있나: train_latent_cam_dm ... siglip2conn_srccam 프로세스 + GPU 1 점유
- 나아가나: tqdm `Epoch N` 증가
- 끝났나  : epoch_cap 50 → results/20260926_011215_dynpose_d200_siglip2conn_srccam/ckpts/last.pth
- 이상신호: Traceback / OOM / nan / 30분간 it 정지
