# 004 — Diffuser offline RL 확정 계획 v2 (데이터소스 재정렬 + venv 구축 + 글루 정정 반영)

> 2026-06-14. plan_new/003(확정 v1)을 잇는 **현재 SSOT**. 003 이후 확정된 3가지 —
> ① **데이터소스 재정렬**(GapFollower-primary → Dreamer warm-load 재학습), ② **전용 py3.8 venv
> 구축 완료**([[005-diffuser-venv-and-handoff]]), ③ **글루 import-bomb 정정**([[006-glue-correction-and-data-contract]])
> — 를 통합한다. **003의 D2는 본 문서 D2로 대체**(append-only라 003 무수정). 나머지 D1/D3/D4/D5는
> 003 유지 + 정밀화. 이 문서만 읽어도 되도록 배경 포함. 과제 명세 = [[003-project-spec]] +
> `_thinking/raws/AIE4003_RL_F1TENTH.pdf`.

---

## 배경 (이 문서만 읽어도 되도록)

**과제**(AIE4003 개인, 주제1 Offline RL): 느린 성능의 policy로 주행 데이터를 모으고, **환경 추가
상호작용 없이(offline RL)** 더 빠른 주행 정책을 학습. 산출물 = behavior policy보다 빠른 lap time +
보고서. **Oschersleben 단일 트랙**.

**모델**: Diffuser(Planning with Diffusion, Janner ICML 2022, clone `~/planning_with_diffusion`).
궤적 diffusion + **별도 value 함수**를 offline 데이터로 학습 → 주행 시 현재 관측 조건으로 미래
궤적 생성하되 value gradient가 고-return(빠른 진행)으로 유도 → 첫 action 실행(MPC). 코어 무변경.

**핵심 원리 + 한계**: stitching + value guidance가 데이터의 좋은 구간을 재조합해 behavior policy를
넘는다. 단 **in-distribution 생성기라 데이터에 없는 속도는 외삽 못 함** → 데이터에 빠른 재료가
실제로 있어야 빠른 주행을 합성. **데이터 설계가 성패를 결정한다.**

**교수님 확정**: "60초 데이터→30초 가능, **정성적(qualitative) 평가**". baseline/목표 쌍 유연. 속도대
혼합(medium-replay 스타일) 허용. → 성공 = 명확한 개선 + 좋은 분석.

---

## 003 대비 바뀐 것 (delta)

| # | 변경 | 근거 |
|---|------|------|
| Δ1 | **D2 데이터소스: GapFollower 스펙트럼 → Dreamer warm-load 재학습**(아래 D2) | 사용자 확정(2026-06-14). "Dreamer 있는데 안 쓸 이유 없다, expert 금지" |
| Δ2 | **런타임 = 전용 py3.8 venv 신설**(RL_project venv 차용 폐기) | [[005-diffuser-venv-and-handoff]]. 구축·검증 완료 |
| Δ3 | **P0 글루 디커플 목록 정밀화**(environments=양성, 진짜 블로커=helpers→utils→rendering) | [[006-glue-correction-and-data-contract]] 실측 |
| Δ4 | **데이터 계약 실 npz 검증**(필수 4키 충족, pose만 갭) | [[006-glue-correction-and-data-contract]] §3 |

---

## 설계 결정

### D1. 관측 표현 — centerline 피처 1순위(재수집), lidar 다운샘플 백업 *(003 유지)*
**피처 ~10D**: `[e_y, sin(e_ψ), cos(e_ψ), v_long, v_lat, yaw_rate, κ(s+L1·L2·L3), progress]`.
이유: 미래 lidar 생성이 진짜 리스크인데 피처는 생성 쉽고·Markov·progress 정렬·자기교차 모호성 해소.
곡률 κ는 spline/lookahead heading-rate(raw 미분 금지), 최근접 인덱스는 env windowed 추적 재현.
- **pose 필요**(npz에 없음, [[006-...]] §3) → D2의 재rollout에서 pose+raw속도+trackname 기록.
- **대안(minimal-first)**: 기존 265-ep replay + lidar 다운샘플(~64)로 파이프라인 먼저 검증 →
  안 되면 피처로. (raw lidar+state는 항상 npz에 보존해 표현 교체와 디커플.)

### D2. ★ 데이터소스 — Dreamer warm-load 재학습 *(003 D2 전면 대체, 최신 확정)*
**원리(합의)**: 데이터에 빠른 재료 + 완주 궤적이 있어야 stitching이 빠른 주행을 합성.

- **주(主) = Dreamer.** behavior policy는 *찾는 게 아니라 만든다*:
  `f1tenth_RL_project/runs/stage1_map_easy3/latest.pt`에서 **world model만 warm-load(actor-critic
  초기화) → Oschersleben 재학습**(warm-load 인프라 = [[004-dreamer-reuse-and-behavior-policy]] §3).
  거기서 나오는 checkpoint를 **pose+raw속도+trackname 기록하며 rollout** → offline 데이터셋.
- critic(plan_new/002 §4)의 **"느린 완주 스냅샷 부재"는 재학습으로 무력화** — 새로 학습해 원하는
  성능대를 뽑으면 됨. 저장 스냅샷에 의존하지 않는다.
- **목표 lap-time대는 학습 결과 보고 유동적**(80/70/60→40 등). **미리 fix하지 않는다.** baseline
  숫자도 P1 rollout 실측으로 그때 확정.
- **expert(DreamerV3 16.6s)는 거론 금지** — 안 되면 그때만. **GapFollower/speed-cap(D3)은 보조
  fallback** — Dreamer 완주 데이터가 너무 희소(265 ep 중 완주 6)해 Diffuser가 완주 policy를 못 짤
  때만 완주데이터 보충용.
- **사용자 방침**: replay 데이터 다양성이 이미 충분해 보임 → **일단 실제로 돌려보고 안 되면 그때
  재고**(over-plan 금지).
- ⚠️ **정직한 caveat**(P4 검증 사항): Dreamer replay의 빠른 구간은 대부분 "빠르게 갔다 크래시
  (fast-then-crash)"라, "빠르되 안 박는(fast-and-surviving)" 조합을 합성하려면 빠르게-살아남은
  구간(완주 6 + 부분성공)이 충분해야 한다. 부족하면 D3 fallback으로 완주 데이터 보충.

### D3. speed-cap 구현 — 보조 fallback 도구 *(003 유지, 위상만 강등)*
`SpeedCappedAgent`: 정규화 action을 **raw로 역정규화 → raw speed에 κ 곱 → 재정규화**(정규화 곱은
offset 때문에 비비례 금지). steer 무수정. env 무변경(차체 물리 불변). κ∈{1.0,0.7,0.5,0.3}, 실제
lap time은 rollout 실측. (D2 주 경로가 막힐 때만 사용.)

### D4. value 대상 reward *(003 유지)*
**progress(dense, `log_reward_progress`) + 소형 collision penalty(-2~-3로 축소)**, **lap 보너스(+100)
제외**(극희소→value 분산 폭증), **normed value([-1,1], ValueDataset normed=True)**.

### D5. Diffuser 적응 (코어 무변경) *(003 + 006 정밀화)*
- **P0 글루 디커플 목록은 [[006-glue-correction-and-data-contract]] §2가 SSOT**(005-glue supersede):
  ① rendering.py heavy import(imageio/mujoco/video/d4rl) guard ② d4rl.py import try/except
  ③ buffer.py:12 np.int→np.int64 ④ f1tenth 로더 + sequence.py stub + config/f1tenth.py
  ⑤ timeouts 함정(termination_penalty=None 또는 is_last&~is_terminal).
- 평가 루프 `scripts/plan_f1tenth.py` 신설(005-glue §6), normalizer 영속화(Trainer.save pickle 덤프),
  device 파라미터화. 코어(temporal/diffusion/helpers) 무변경.

---

## 향상 폭 현실 추정 (정성평가 전제)
- 데이터에 충분히 빠른 재료(재학습 policy의 빠른 구간 + 완주)가 포함되면 Diffuser+guidance는 데이터
  최속 근처~약간 개선을 합성 가능. baseline(느린 checkpoint) 대비 **개선은 달성 가능**.
- 데이터를 균일 느린 단일대로만 모으면 개선 미미 → **빠른 재료 포함이 필수**(D2 caveat).

---

## Phase 분해 (게이트 + 런타임 배치)

| # | 작업 | 실행 venv | 게이트 |
|---|------|-----------|--------|
| **P0** | 글루 디커플([[006-...]] §2) + 더미 로더로 `train.py` **1-step 학습** + NullRenderer | **새 .venv** | `from diffuser.models.temporal import TemporalUnet` 통과 + 1-step loss 산출 |
| **P1** | stage1 WM warm-load → **Oschersleben 재학습**으로 behavior policy 제작 + checkpoint **lap time 실측**(baseline 확정) | **RL_project venv** | 원하는 성능대 policy 확보 + baseline 숫자 |
| **P2** | 선정 checkpoint(들) **pose+raw속도+trackname 기록하며 rollout** → offline 데이터셋 | **RL_project venv** | 궤적 무결성 + 속도/완주 분포 리포트 |
| **P3** | 표현 결정(피처 1순위/대안 lidar) + `datasets/f1tenth.py` 로더 + online 피처 모듈 + normalizer 저장 | 새 .venv | SequenceDataset 통과 + 윈도우/피처 검산 |
| **P4** | 궤적 diffusion + value 함수 학습 | 새 .venv | loss 수렴 + **생성 궤적 품질 점검**(미래 피처 매끄러움, D2 caveat 검증) |
| **P5** | `plan_f1tenth.py`: GuidedPolicy↔f110_gym(normalizer 로드+online 피처+K-step MPC) | **새 .venv** | 1 ep 완주 시도 |
| **P6** | 평가: baseline 대비 lap time, 2랩 완주, Diffuser가 더 빠른지 | 새 .venv | **baseline보다 빠름(정성)** |

- 순서 주의: 피처 표현이면 P2(pose rollout)가 P3보다 앞. minimal-first(lidar+기존 replay)면 P2 일부 생략 가능.
- P0는 데이터·교수님 무관하게 **즉시 선행 가능**.
- 데이터생성(P1/P2)은 RL_project venv(Dreamer 에이전트 본거지), 학습·평가는 새 .venv (§런타임 배치는 [[005-diffuser-venv-and-handoff]] §3.4).

---

## 리스크 Top + 완화
1. **fast-and-surviving 재료 부족**(향상 천장, D2 caveat) → 재학습 policy 다양성 + 필요 시 D3
   speed-cap fallback으로 완주 데이터 보충. **P4 생성 품질 점검에서 조기 판정.**
2. **재학습이 003 §4처럼 크래시→고속완주로 점프**해 원하는 중간대가 안 나옴 → 여러 checkpoint
   수확 + (정 안 되면) D3 speed-cap으로 목표대 제작. 목표대 유동이라 부담 적음.
3. **centerline 피처 online 계산 정확도**(자기교차 인덱스·곡률 노이즈) → env windowed 추적 재현,
   spline/lookahead 곡률, P3 검산. 안 되면 lidar 백업.
4. **글루 디커플 잔여**(rendering import 체인) → [[006-...]] §2 목록대로 P0에서 일괄 처리, 1-step smoke 게이트.
5. **공유 gym 소스 수정 금지**(`RL_project/gym` editable 공유 → RL_project에 영향) — sim 코어 무변경.

---

## 다음 단계 (인수인계)
1. **즉시 = P0**(교수님 답·데이터 무관): [[006-glue-correction-and-data-contract]] §2 디커플 4건 →
   코어 import 통과 + train.py 1-step loss. 새 .venv에서 실행.
2. P1: stage1_map_easy3/latest.pt warm-load 재학습 (RL_project venv) → baseline lap 실측.
3. 분기마다 _thinking 문서 + commit + push. push 목적지 = github.com/dlacksdn/f1tenth_LeWM(폴더명만 리네임됨).
4. f1tenth 과제 판단 시 [[003-project-spec]] + PDF 참조.
