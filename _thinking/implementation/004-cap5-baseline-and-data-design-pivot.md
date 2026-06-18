# 004 — cap-5 baseline 확정 + 데이터 설계 전환(실패-데이터 stitching) + cap-10 착수

> 2026-06-18. [[002-p1-speedcap-policies-and-rlproject-changes]] 후속. P1 진행: cap-5 학습 종료·
> baseline 정책 **확정**, cap-10 학습 착수. + 이번 세션의 큰 수확 = **데이터 설계 논의**(실패 데이터
> stitching 가능성 실측 + 평가 프로토콜). append-only. 선행: [[002-p1-speedcap-policies-and-rlproject-changes]],
> [[008-diffuser-plan-v4]], [[006-glue-correction-and-data-contract]].

---

## 0. 한 줄 상태
**cap-5 종료 + baseline 확정 = `step_25k`(2랩 107.16s, 완주율 1.0).** cap-10 학습 착수(proven 레시피).
이번 세션 핵심 = 데이터를 **"실패 rollout + 완주 앵커" 혼합(medium-replay/expert)** 으로 가는 방향 + 평가
스펙트럼(baseline=cap-5 / reference=cap-10·15 / 천장 각주=cap-20) 정립.

---

## 1. ★ cap-5 baseline 정책 확정 — `step_25k.pt`
- **채택 = `runs/cap5_oschersleben/step_25k.pt` (eval 시 `--v_max 5` 필수).**
- **2랩 107.16s** (per-lap 53.78 / 53.38), **완주율 1.000**(deterministic 5ep 전부 동일), **A12/A13 PASS.**
- slide "약 100초대 behavior policy" 목표 정확히 달성. cap-5 = 느린 baseline.

### 1-1. 왜 25k인가 — 진동 + deterministic eval 실측 (1차 소스)
cap-5도 cap-15처럼 **학습 진동**(빠른 라인 시도→2랩째 크래시). metrics step 75k(목표 300k의 25%)에서
수동 종료. named 체크포인트 deterministic eval(`eval_gate --v_max 5 --episodes 5`):

| 스냅샷 | 완주율 | 2랩 | 판정 |
|---|---|---|---|
| step_35k(=최신/latest) | 0.000 | 1랩도 실패(1183 step 크래시) | 진동 골짜기 |
| step_30k | 0.000 | 1랩 52.58s만, 2랩째 크래시(4028 step) | 진동 골짜기 |
| **step_25k** | **1.000** | **107.16s** (53.78+53.38) | **채택** |
| step_20k | 1.000 | 121.86s (61.56+60.3) | 완주하나 더 느림 |

- **★ latest.pt(=step_35k)는 쓰면 안 됨**(cap-15 교훈 재확인 — 진동 학습은 최신이 골짜기). 봉우리=step_25k(metrics 50k).
- 사용자 규칙 적용("35k 완주하면 35k, 아니면 30k, 둘 다 완주 시 더 빠른 것"): 35k·30k 둘 다 2랩 실패 →
  더 느린 봉우리 탐색 → **완주하는 25k·20k 중 더 빠른 25k 채택.**
- ⚠️ **stochastic ≠ deterministic**: 학습 중 eval_eps에 2랩 완주 **30개**(2랩 123–128s) 존재하나, 이는
  학습곡선 전반의 stochastic eval. **baseline은 재현 가능한 deterministic 정책(step_25k)으로 박는다.**
  (검증사실 재확인: deterministic 5/5 완전동일.)

## 2. 기록 스펙트럼 (1차 소스 실측, 2랩)
| tier | 정책/자산 | 2랩 | 비고 |
|---|---|---|---|
| **cap-5**(느린 baseline) | step_25k (eval 확정) | **107.16s** | ← behavior policy(데이터 출처) |
| **cap-10**(중간) | 학습 착수(§5) | (예상 ~60–70s) | reference 눈금 |
| **cap-15**(빠름) | step_105k (002 검증) | **37.3s** | reference 눈금. replay 30ep=38.2–40.8s |
| **cap-20**(무제한) | stage2 완주 6ep(replay) | **35.5–37.6s** | 무제한=expert격(폐기). "천장 각주"로만 |
- per-lap 항상 1랩 > 2랩(출발 정지 가속) → **2랩 기준 규약의 데이터 근거.**
- ★ **cap-15 ≈ cap-20**(간격 2–3s): V_MAX=15가 평균실현속도 14.4보다 높아 사실상 무캡. → 천장 reference는
  cap-15 하나로 통일, cap-20 별도 학습/측정 불필요.

## 3. ★★ 데이터 설계 전환 — "실패 데이터 stitching" (이번 세션 핵심 논의)
사용자 질문에서 출발: "성공(완주) 데이터는 이미 완전한데 조합이 무슨 의미? 흔한 **실패 데이터**를 조합해
2랩 완주를 합성하는 게 Diffuser 의의 아니냐."

### 3-1. stitching (a)커버리지 실측 — 가능성 데이터로 확인 (`/tmp/stage2_crash_coverage.py`)
신호: `cumsum(log_reward_progress)` = 출발점 기준 누적 도달거리(m). 모든 ep 동일 default_pose 출발.
stage2(=무제한) **크래시 259개**(완주 6 제외)로 트랙 커버리지:
- **트랙 1랩(55 bin) 구멍 = 0/55.** 모든 구간이 13~259개 크래시 ep로 빈틈없이 커버. max 도달 **1.93랩(529.9m).**
- survival: 0.5랩 41ep → 1랩 13ep → 1.5랩 2ep → 2랩 0ep. (median 도달 37m: 시작+37m에 첫 난코너.)
- **★ obs에 lap 정보 없음**(`state`=[vel_x,vel_y,ang_z,prev_steer,prev_speed], lidar=위치기반) →
  lap1 위치 X 관측 ≈ lap2 위치 X 관측 → **1랩 커버리지가 2랩 완주 재료로 재사용.** 2랩 직접 도달 0이어도 OK.
- 결론: **(a)커버리지 조건 충족.** maze2d/antmaze(실패·부분 데이터→목표 합성)와 동형. 우리 데이터가 전제를 실제로 만족.

### 3-2. 비판적 검수 — 채택하되 정밀화
- **완주 ≠ 성능.** 우리 지표는 **2랩 lap time**. "실패 조합→완주"는 커버리지상 가능하나, 그 완주가 baseline보다
  **빠르려면** 데이터에 빠른 구간 필요. **Diffuser=in-dist 생성기 → 속도 외삽 불가**([[006...]]/005 §3.1).
  → cap-5 단독 실패로는 속도 개선 못 함. **속도 다양성(cap-10·15 실패/완주)이 데이터에 반드시 섞여야** value가
  빠른 조각을 stitching.
- **순수 실패-only의 잔여 리스크(미확정, P4 생성품질에서만 닫힘)**: ① value가 "고속+안전"을 충돌 데이터에서
  분리하는지(고속 레이싱은 antmaze보다 어려움) ② 빠른 tier는 트랙 후반 커버 못 함(무제한 크래시 median 37m,
  일찍 죽음) ③ 완주/R_lap 신호가 크래시 데이터엔 0번 등장.
- **권고 = "실패 대량 + 완주 앵커 소량" 하이브리드(D4RL medium-replay/expert).** 완주 앵커가 (a) 고속+안전
  재료 (b) lap 신호를 데이터에 심어 위 리스크를 헤지. 우리 앵커 자산: cap-15 step_105k(빠른) + cap-5 step_25k(느린)
  + stage2 완주 6. 실패는 cap-5/10/15 학습 중 rollout으로 대량(공짜) — 5k 체크포인트 활용(사용자 제안).

### 3-3. 사용자의 "완주까지 학습 불필요" 제안 — 부분 채택
- 매력: 학습비용(007 #4: 캡당 5–8h) 회피 + 속도 다양성 자동.
- 단 정밀화: 학습 0=쓰레기, "완주까지 길게 안 가도 됨(medium 수준)"이 정확. **완주 앵커는 소량 남겨야** value가
  "빠르면서 안전" 라인을 정의. → cap-10은 완주봉우리(=baseline 기록) + 실패 rollout(=데이터)을 **한 학습으로 동시 획득.**

## 4. 평가 프로토콜 (보고서)
- **Baseline(성공 기준선)** = behavior policy = **cap-5 step_25k (107.16s)**. Diffuser가 이걸 넘으면 성공.
- **Reference 눈금** = cap-10(~60-70s) / cap-15(37.3s). "Diffuser 결과가 어느 수준인지" 위치.
- **천장 각주** = cap-20 무제한 ~36s(expert격, 폐기 — 참고치만).
- 서사: *"느린 cap-5 데이터로 offline 학습 → cap-5 대비 단축, cap-N 수준 도달"* (D4RL normalized score 스타일).
- 측정 통일: deterministic, 2랩, `--v_max` 학습값 일치(필수). replay 기록(stochastic)은 참고만.

## 5. cap-10 학습 착수
- `scripts/cap10_watchdog.sh` 신설(cap5/cap15와 동일 proven 패턴 + bracket-pgrep), `runs/cap10_oschersleben`.
  V_MAX=10.0, warm=stage1_map_easy3/latest.pt, lr×0.5, joint 0.3, steps 300000. 2026-06-18 17:47 가동.
- **joint 제거 실험은 보류**: cap-5/15가 진동했지만 봉우리 스냅샷으로 baseline 확보 성공(proven) → cap-10도 동일
  레시피, 진동 시 봉우리 채택 전략. (joint 제거는 미검증이라 baseline 데이터 확보 우선.)
- 운영: 완주 봉우리 확보 후 수동 종료 + `eval_gate --v_max 10`로 deterministic 2랩 채택(25k처럼 latest 불신).

## 6. 다음 단계
1. **cap-10 모니터**(첫 완주 ~metrics 70-130k 예상, GPU 점유 중). 봉우리 확보 후 eval(--v_max 10) → cap-10 baseline 기록.
2. 그 사이 **P0 글루 디커플 병행 가능**(새 .venv, GPU·데이터 무관, [[006-glue-correction-and-data-contract]]).
3. **P2 데이터 수집 설계 확정**(§3 반영): 실패 rollout 대량 + 완주 앵커(step_25k/105k) 소량, lidar+raw+pose+tier(v_max) 기록.
   - ★ P3 역정규화 주의(002 §7): npz action은 tier-상대 정규화 → 각 tier v_max로 raw 역정규화 일관화.
4. 분기마다 _thinking + commit + push(자율 pull 금지). push: github.com/dlacksdn/f1tenth_planning_with_diffusion.
