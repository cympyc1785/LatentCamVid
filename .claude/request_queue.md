# Request Queue

사용자 요청을 접수 → 진행 → 완료 순으로 옮겨 적는 단일 대기열이다.
규약은 `CLAUDE.md` 의 `### 요청 대기열 (.claude/request_queue.md)` 절에 있다.

- **Request** — 접수만 된 것. 아직 손대지 않았다.
- **Working** — 지금 수행 중인 것. **동시에 하나만** 둔다.
- **Wait** — 긴급 요청 때문에 **중단하고 비켜 둔 것.** 어디까지 했는지 한 줄 적어 두고,
  긴급 건이 끝나면 다시 `Working` 으로 돌린다.
- **Done** — 끝난 것. 한 줄 진행 내역을 같이 적는다. **10개가 차면 `.claude/request_done.md` 맨 위로 통째로 옮긴다.**
- **Incomplete** — 보류·차단된 것. **사유를 반드시 적고**, 다음 request 로 넘어간다.

---

## Request

> 공통 배경 (2026-09-23): "DA3, Molmo2 를 encoder 로 쓰는 게 진짜 필요한지" 검증 실험. 학습 3건은 GPU 0~2.

### R19. d200 dynpose — condition = text + SigLIP2 feature(Molmo2 visual) + camera embedding 학습 → Working
### R27. TRUMANS: 충돌·occlusion·거리·프레이밍 gate 를 fitting **중에** Blender GT mesh raycast 로 — 가능한지 실험
  1차 (2026-09-24): mesh_grid EDT 로 벡터판 `mesh_ray_profile` — 194 ms/궤적, Blender 판정과 20씬 175 rung
  **88.0% 일치** (grid 만 탈락 19 / Blender 만 탈락 2). 원인: 시선 10(5 cm 격자가 얇은 물체 부풂),
  subject_dist 6(조준점 정의 차이). 다음: 2.5 cm 격자 + Blender 조준점 export → fit 게이트로 배선.
### R52. DynPose 확장 — GPU 0,3,6,7 (6,7 은 사용자가 빼라 할 때까지). 결정(09-25): 로고 영상 탈락 유지 / bank 는 d200 설정(d185+d199). 신규 7,168: recon 완료(7,168 OK) → nouns 진행중 → SAM3 → graph → bank → export
### R54. d200 molmo2_l21_da3 에 DA3 dense frame **24프레임** 으로 학습 — GPU 0 (2026-09-25) · D285 캐시 10,169 완료 (863.70 GB) → smoke rc=0 → **screen train2 GPU 0 본 학습**
### R58. SigLIP2 arm 에 Molmo2 connector(attention pooling + projector) 를 붙이되 **새로 학습**되게 (2026-09-25) · D286 코드+smoke rc=0 (커밋 0df7ba7), 속도 ~3 s/it 라 본 학습 전 사용자 확인 대기
### R22. TRUMANS 데이터셋 구축 — **계획 승인됨 (2026-09-23)**
조사: `tmp/agent/reader-r22-trumans.md` (핵심 주장 5건 직접 재확인). 사용자 결정 넷:
① 범위 = s3f0k6 **737 clip 전량** (191 은 prep 완료, 546 은 graph/cloud/mesh 부터)
② gate = D266T (bank: elev + framing≥0.6) + mesh raycast, **통과한 사다리 칸 전부** 내보냄
   (exporter 에 raycast selection 필터 옵션 신설 필요)
③ GPU = **3 만** (CLAUDE.md 0~3 준수, 사용자 원래 지시 3~4 에서 변경)
④ caption = Qwen3-VL instance_desc (vLLM 온디맨드), **단독 trumans 코퍼스**
source video 는 `trumans_to_recon.py` 가 이미 Blender 합성 (follow + slow arc) — 재렌더 없음.

---

## Working

### R19~R22 — 전부 기동됨, 장시간 실행 감시 중 (2026-09-23) · 커밋 `bd78ac5`
- R19 `dynpose_d200_siglip2_srccam` — SigLIP2 캐시 10169 완료 → smoke 통과 → **GPU 0 train2**.
- R22 **재계획 (사용자 정정 2026-09-23)**: 요청은 "sweep66 후보 시작 카메라로 **source video 를 새로 만드는**
  계획" 이었다 — 옛 737 clip(자체 probe 격자 시작점) 에 뱅크를 돌린 것은 오해였다. 그 뱅크는
  **Ctrl+C 로 중단** (310/737 완료분은 보존). 승인된 새 계획: board 807 chunk x **K=3** = 2,421 clip,
  `trumans_to_recon.py --board` 모드(분기 추가) + `exec/run_board_sources.py` → 렌더 → d271 체인.
- 공통 후속: srccam 계열은 `eval_testset.py` geo_proj MLP 미지원 (tasks.md A3) — testset eval 전 수리.


---

## Wait

_(비어 있음)_

---

## Done

### R57. 저장공간 정리 — 완료 (2026-09-25 23:45). 사용자 승인 S1,S2,S3,S5,S6,S7 + M2(da3만),M3,M5(가중치 폴더만),M7 삭제. free 1,152 G → 3,665 G. 기록 tmp/r57/deleted.txt

### R55. d262 molmo2_l21_srccam vs D269 siglip2_srccam testset eval — 완료 (2026-09-25 20:35). eval srccam 수리 94fadc8, 5144 seg 전량, 수치 EXPERIMENTS.log

### R56. 명사 추출 출력 형태 + 예시 — 완료 (2026-09-25). 신규 c7f7df9a 6프레임 그림 전송, record 구조 설명 (dynamic/static/subject/reasoning)

### R49~R51. DynPose-100K 43,782편 VLM+SAM3 필터 (D282) — 완료 (2026-09-24 22:45) · 커밋 `6bb2cd5`
VLM 0 실패, SAM3 12샤드(GPU 0,3,6,7 — 6,7 은 사용자 허용). 9,396 통과(21.5%), pass.csv/fail.csv/summary.json,
reel 17개 전송. 통과 중 신규 recon 필요 7,168 (기존 2,228).

### R53. TRUMANS source 궤적 smooth_kf + 위치 kf6 (D283) + 옛 보간 clip 재렌더 — 적용·재기동 (2026-09-24) · 커밋 `f818ce4`
옛 clip 916개(경로 8,444개)·chunk 상태 200개를 `/data1/cympyc1785/data/TRUMANS-Lite/old_d272_nokf/` 로 **이동**
(삭제 아님, 목록 tmp/r22b/moved_old_nokf.txt). 렌더 재기동 → 807 chunk 전부 새 보간.

### R46. 렌더 16 spp 전면 교체 (D281) — 완료 (2026-09-24) · 커밋 `c390179` · 샘플 영상은 첫 16spp clip 나오면 전송
### R47/R48. VLM 입력 방식 권고(영상 fps2) · raw output+영상 5개 전송 · SAM3+VLM 협업 설계 답변 — 완료
SAM3 video 9프레임(fps2 부표본) 추적 2.38 s/clip 추가 측정.

### R42~R45. spp 비교 그림 · VLM prompt · 답 갈린 5개 · SAM3 속도 — 완료 (2026-09-24)
SAM3(GPU3, 49f 640x360, 'person'): video 1프로세스 12.15 s/clip, 3프로세스 실효 6.3 s, image 8프레임 0.89 s.

### R40. Blender 렌더 속도 — depth/index animation job 적용 (D279) · 커밋 `16235b8`
geom 197.2 → 39.4 s (산출물 일치), clip ~331 → ~173 s 예상. RGB 옵션(16spp/bounce/해상도)은 화질 트레이드오프라 사용자 결정 대기.
### R41. VLM 입력 방식 비교 50 clip x 5 모드 — 완료, 사용자 판단 대기 (`tmp/r41/index.html`)
5 모드 모두 parse 50/50, 시점 5모드 일치 45/50, closeup 을 모든 모드가 third 로 봄(7~8/9). vLLM 내림.

---

## Incomplete

### R21. DA3 49프레임 학습 — **사용자 지시로 중단 (2026-09-23)**
`dynpose_d200_molmo2_l21_da3_v49` (wandb apgr3om0) 를 epoch 0 도중 Ctrl+C 로 정상 종료.
on-the-fly DA3 4.5 s/it ≈ 7.2 h/epoch (50 epoch ≈ 15일) 이었다. yaml 은 남겨 뒀다 — 재개하려면
토큰 풀링 캐시(~200 GB) 같은 속도 대책이 먼저 필요하다. GPU 2 비어 있음.

### C2. `.claude/settings.local.json` 에 `kill`·`pkill` 허용 — **차단** (2026-09-23)
사용자 지시 "local permission으로 kill, pkill 승인되게끔 추가해줘". `permissions.allow` 에
`Bash(kill:*)` `Bash(pkill:*)` 두 줄만 넣은 파일을 만들려 했으나 **Write 가 auto-mode
classifier 에 거부**됐다 ("Auto mode could not evaluate this action"). 파일은 아직 없다.
**해제 방법** — 사용자가 `/permissions` 에서 직접 추가한다.

### C1. `CLAUDE.md` GPU 규칙 갱신 — **부분 해제** (2026-09-23)
`## Don't` 의 본 규칙은 고쳤다: `0~4 + TRUMANS GPU5 예외` → **`0~3`, 4~7 금지, 예외 만료**.
**남은 차단 2줄:** `## 작업 흐름` 의 학습 3번·추론 2번 "GPU 0~4 중 가장 여유가 있는" 은
개별 Edit·`replace_all` 둘 다 여전히 거부된다.
**영향 없음:** 같은 파일 맨 위 `## Don't` 가 0~3 을 명시하고 memory `gpu-0-3-only.md` 도
같은 규칙이라 실제 배정은 0~3 으로 간다. 남은 두 줄은 낡은 중복 문구다.
