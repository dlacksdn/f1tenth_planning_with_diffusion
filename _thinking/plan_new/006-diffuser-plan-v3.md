# 006 — Diffuser offline RL 확정 계획 v3 (데이터 = 속도상한 훈련 정책들의 완주 스펙트럼)

> 2026-06-18. plan_new/004(v2)를 잇는 **현재 SSOT**. 005 적대적 critique를 전면 반영하고, 사용자와
> 합의한 데이터 설계(**속도상한을 건 채 warm-load 재학습한 정책들** + 기존 best expert)를 확정한다.
> **004의 D2는 본 문서 D2로 폐기·대체.** 나머지 D1/D4/D5는 004 + critique 보강. **모든 lap time 2랩
> 기준(사용자 규약).** **expert(기존 best) 포함은 교수님 허용 가정**(미확정 — fallback 명시). append-only
> 원칙이나 본 문서는 사용자 명시 허용 하에 (A)rollout-clamp 오기를 (B)속도상한-훈련으로 정정한 개정판.
> 선행: [[004-diffuser-plan-v2]], [[005-diffuser-critique-v2]], [[006-glue-correction-and-data-contract]],
> [[005-diffuser-venv-and-handoff]], [[004-dreamer-reuse-and-behavior-policy]], [[003-project-spec]] + PDF.

---

## 배경 (이 문서만 읽어도 되도록)

**과제**(AIE4003 개인, 주제1 Offline RL): 느린 policy로 데이터 수집 → **환경 추가 상호작용 없이** 더
빠른 주행 정책 학습. 산출물 = behavior policy보다 빠른 2랩 lap time + 보고서. **트랙 = Oschersleben 단일**
(map_easy는 더 이상 보지 않음 — Oschersleben에서만 잘 나오면 됨).

**모델**: Diffuser(Janner ICML2022, clone `~/planning_with_diffusion`). 궤적 diffusion + 별도 value
함수 → 현재 관측 조건으로 미래 궤적 생성, value gradient가 고-return으로 유도(MPC, 첫 action 실행).
코어 무변경. **향상 원리·한계**: stitching + value guidance가 데이터 내 좋은 구간을 재조합해 baseline을
넘는다. 단 **in-distribution 생성기라 데이터에 없는 속도/전이는 외삽 불가** → 데이터 설계가 성패를 결정.

**평가**: 정성적(교수님). "60→30(2랩) 가능", medium-expert/medium-replay식 혼합 허용.

---

## 004 → 006 변경 요지 (delta)

| # | 변경 | 근거 |
|---|------|------|
| Δ1 | **D2 = "무제한 warm-load 재학습"(폐기) → "*속도상한을 건* warm-load 재학습 정책들 + 기존 best"** | 005 §2: *무제한* 재학습=stage2(크래시→고속점프). 속도상한 훈련은 *별개*(아래 화해) |
| Δ2 | **데이터 = 완주 데이터만**(크래시 데이터 배제) | value=return-to-go라 "빠르지만 크래시"는 회피됨 + in-dist라 "고속 생존" 외삽 불가 |
| Δ3 | **medium-expert식 혼합**: 속도상한 훈련 정책(느림/중간) + 소량 expert(기존 best) | offline RL 표준 세팅(D4RL). 천장 확보 + 개선의 정직성 |
| Δ4 | P0 게이트 확장·디커플 목록 완성·normalizer 정정·online 피처 parity·baseline 게이트 | 005 §4–6 |
| Δ5 | 모든 lap time 2랩 표기 | 사용자 규약 |

---

## 핵심 검증 사실 (2랩 기준)

- **lap 단위**: npz `log_lap_time_s`=랩당, `에피소드 steps × 0.02`=2랩 총합. 완주(log_completed=1)=2랩.
- **체크포인트 인벤토리**(`runs/stage2_oschersleben` 실측): 완주 정책 = `policy_best_lap16.6s_step85k`
  (2랩 ~33s)·`...lap17.4s_step80k`(~35s)뿐. 인터벌 스냅샷 step_5k~85k는 **대부분 크래시**(예: step70k
  eval_length ~320step=6.4s = 완주 못 함). → **"느리게 완주하는 체크포인트"는 존재하지 않는다** → 느린
  완주 정책은 *훈련해서 만들어야* 한다(D2). 사용자가 기억한 "38s 정책"은 replay 에피소드(37.6s/2랩)였음.

---

## 설계 결정

### D1. 관측 표현 — minimal-first(lidar 다운샘플), pose는 기록만 *(005 §4)*
- v1 = lidar 다운샘플(~64) + state. 파이프라인을 P0→P6로 관통(피처 online 계산 리스크 제거). raw 보존.
- speed rollout(P2) 때 pose+raw속도+trackname 함께 기록 → centerline 피처(~10D)는 v2 옵션으로 열어둠.

### D2. ★ 데이터소스 — 속도상한 훈련 정책들의 완주 스펙트럼 + 기존 best expert *(004 D2 전면 대체)*

**★ critic 반증과의 화해 (핵심)**: 005가 반증한 것은 **무제한** warm-load 재학습이다 — progress 보상 +
무제한 속도라, 안 박기 시작하면 곧장 고속(~18s/lap)으로 점프해 "느린 완주" 단계가 없다(=stage2 곡선).
**속도상한을 걸고 훈련하면 이 점프가 불가능**하다: 정책이 V_MAX를 못 넘으니 progress 최대화의 유일한
길 = "그 속도 내에서 깨끗이 완주" → **느리게-완주하는 정책으로 수렴**. critic은 무제한 런만 봤을 뿐,
속도상한 훈련은 **미검증이지 반증이 아니다.** 따라서 이 경로는 유효하다.

**확정 설계**:
- **느림·중간 tier = warm-load(stage1_map_easy3 WM만, actor-critic 초기화) → *Oschersleben에서 직접*
  속도상한을 걸고 훈련**:
  - V_MAX=10 → **cap-10 정책**(느림, baseline)
  - V_MAX=15 → **cap-15 정책**(중간)
  - 각 정책은 *그 속도에 맞는 조향을 학습* → 깨끗한 완주 데이터(rollout-clamp의 조향-속도 미스매치 없음).
  - 속도상한 = env action space의 V_MAX 제한(정책이 명령 가능한 최대속도). **차체 물리값 무변경**(action
    bound이지 차량 동역학 파라미터 아님). 훈련은 online이나 **데이터 수집용**이라 offline 제약(=Diffuser
    개선단계 환경 상호작용 금지) 위반 아님.
- **expert tier = 기존 best `policy_best_lap16.6s_step85k`(2랩 ~33s) 그대로**(재훈련 0). 38s tier는
  불필요 → 폐기(기존 체크포인트로 38s 완주는 불가, best로 충분).
- **단일 소스가 아니라 *속도별 별개 정책***이라 라인·거동 다양성이 생긴다 → stitching이 의미 있어지고,
  "expert 초과"도 (균일 스케일보다) 가능성이 열린다(상보적 라인 존재 시).
- **비율(잠정, 조정 가능): cap-10 45% / cap-15 45% / expert(best) 10%.** medium-expert식 구성. 소량 expert가
  fast-and-surviving 재료·천장을 제공하되, 10%라 Diffuser가 복제가 아니라 stitch/guide로 끌어올려야 함.
- **교수님 허용 가정**(expert tier 포함). **불허 시 fallback**: expert 제거 → {cap-10, cap-15}만(천장 =
  cap-15 속도, 개선 작지만 진짜).

### D3. 속도상한 적용 — *훈련 시* action space V_MAX 제한 *(004의 rollout-clamp에서 정정)*
- 적용 위치 = **훈련 시 env action space의 V_MAX**(예 20→10/15). 정책이 그 상한 내에서 최적 주행을 학습.
- (rollout-time 클램프 = 미사용. 고속용 조향을 저속에 강제하는 미스매치 때문에 데이터 품질이 떨어짐.)
- ⚠️ 리스크: 저속 정책이 정말 2랩 완주를 학습하는가 + 학습 시간이 합리적인가 → P1 게이트로 실측.

### D4. value 대상 reward *(004 유지, 005 §3 ✅)*
progress(dense `log_reward_progress`) + 축소 collision(−2~−3) + lap 보너스 제외 + normed value([-1,1]).

### D5. Diffuser 글루 *(006-glue + 005 보강, 변경 없음)*
- **P0 디커플 = [[006-glue-correction-and-data-contract]] §2 + 005 추가분**: ① rendering.py heavy import
  (imageio/mujoco/video/d4rl) guard ② colab.py:17(video 체인) guard ③ d4rl.py:25 try/except +
  preprocessing.py:7·sequence.py:7의 d4rl 함수 import 스텁 커버 ④ buffer.py:12 np.int→np.int64
  ⑤ sequence.py:22–23 stub + datasets/f1tenth.py 로더 + config/f1tenth.py.
- **normalizer 영속화 정정**: Trainer.save는 normalizer 미저장([training.py:136–150]); 추론은 재구성
  ([serialization.py:36–60]). → 별도 pickle 저장+로드 또는 동일 데이터 재구성 보장. **P3 라운드트립 게이트.**
- online 피처 parity(피처 채택 시): 수집·추론 단일 피처함수 공유 + golden test.
- config: clip_denoised=True, termination_penalty=None(또는 timeouts=is_last&~is_terminal).

---

## 성능 기대치 / 성공 기준 (정직하게)

| 수준 | 내용 | 달성성 |
|---|---|---|
| **Floor(성공 기준)** | **cap-15 정책보다 빠른** 2랩 완주 (사용자 정의) | **달성 가능** — 데이터에 expert(33s) 재료 + value guidance가 cap-15 위로 끌어올림 |
| **Realistic** | 데이터 90%가 더 느린데도 **best(~33s) 수준에 근접** | offline RL의 정수 — 보고서로 당당 |
| **Stretch** | best(~33s) *초과* | (A)균일스케일보다 **가능성 있음**(속도별 정책=라인 다양성 → 상보 stitching). 단 보장 아님, 라인이 실제 갈리는지 데이터 의존. 안 되면 "cap-15 초과"로 서사 성립(사용자 합의) |

(라인 다양성이 부족해 expert 초과가 안 되면, v2에서 더 다양한 라인 정책 추가.)

---

## 데이터-모델 궁합 (★ 보고서 필수 서술 — 사용자 지정)

> 우리 데이터는 "느림(저속상한 훈련) + 빠름(expert)"의 **이중분포(bimodal/medium-expert)**다. 단순
> behavior cloning은 두 모드를 평균 내 망가진다. 그러나 **value 기반·궤적 생성 방법(Diffuser 포함)은
> 이런 혼합 데이터를 다루도록 설계**돼 있어, value guidance로 고-return 모드를 선택하고 stitching으로
> 좋은 구간을 재조합한다. 다만 unimodal 데이터보다 일반적으로 *더 어려운* 세팅이며, **그래서 이 과제에
> Diffuser 선택이 적절하다.** (D4RL의 medium-expert/medium-replay가 이 세팅의 표준 벤치마크.)

---

## Phase 분해 (게이트 + 런타임 배치)

| # | 작업 | venv | 게이트 |
|---|------|------|--------|
| **P0** | 글루 디커플(D5) + 더미 로더 train.py 1-step | **새 .venv** | (A) TemporalUnet import (B) f1tenth 로더로 SequenceDataset 적재 (C) train.py 1-step loss (D) train_values/plan_guided import·파서. Config pickle→Renderer 끌림 확인 |
| **P1** | **warm-load → Oschersleben에서 V_MAX=10·15 정책 훈련** + 2랩 완주·lap time·학습시간 확인 | **RL_project venv** | 각 캡 정책이 **2랩 안정 완주** + 2랩 시간 실측 + 학습시간 합리적. baseline=cap-15 시간 정의. 미달 시 캡/하이퍼 조정 |
| **P2** | cap-10·cap-15 정책 + 기존 best rollout → **lidar+raw+pose+trackname 기록**, 45/45/10 수집(크래시 폐기) | **RL_project venv** | tier별 분포·완주율 리포트 |
| **P3** | datasets/f1tenth.py 로더 + lidar 다운샘플 + **normalizer 라운드트립 검산** | 새 .venv | SequenceDataset 통과 + 동일 bounds |
| **P4** | diffusion + value 학습, **생성 궤적 품질 점검**(expert 재료 활용·빠른구간 합성) | 새 .venv | loss 수렴 + 품질. expert 비율 부족 시 조정 |
| **P5** | plan_f1tenth.py: GuidedPolicy↔f110_gym, normalizer 로드, **K-step(4~8) MPC** + wall-clock 추정 | **새 .venv** | 1 ep 완주 시도 |
| **P6** | 평가: cap-15 대비 2랩 lap time, 2랩 완주, best 근접/초과 여부 | 새 .venv | **cap-15보다 빠름(floor)** |

- **P0는 데이터·교수님 무관하게 즉시 선행 가능.** 데이터(P1/P2)=RL_project venv(Dreamer·warm-load 인프라), 학습·평가=새 .venv.

---

## 리스크 Top + 완화
1. **속도상한 정책이 2랩 완주를 학습 못 하거나 학습이 오래 걸림** → P1 게이트(완주+시간 실측), 저속은 보통 완주 쉬움. 안 되면 캡/리워드 조정.
2. **expert 10%가 신호로 부족** → P4 점검 후 비율 상향(잠정).
3. **expert 초과 불발**(라인 다양성 부족) → v1 floor를 "cap-15 초과"로, 초과는 v2(라인 다양성).
4. **글루 디커플 잔여** → P0 확장 게이트로 일괄.
5. **normalizer/피처 seam** → normalizer 라운드트립(P3), 피처 단일함수 공유+golden test(P5).
6. **공유 gym 소스(`RL_project/gym`) 수정 금지** — RL_project에 영향. (단 훈련 시 V_MAX는 *설정/래퍼* 수준에서 제한, 공유 sim 코어 무변경.)

---

## 다음 단계 (인수인계)
1. **즉시 = P0**(새 .venv): 디커플 + train 1-step + train_values/plan_guided 파서.
2. **교수님 확인 1건**: expert(기존 best, ~33s) 정성평가 포함 가능? → 불허 시 expert tier 제거 fallback.
3. P1: warm-load → Oschersleben에서 V_MAX=10·15 훈련, 2랩 완주·시간 확인(RL_project venv).
4. 분기마다 _thinking 문서 + commit + push(상시 규약, 자율 pull 금지). f1tenth 판단 시 [[003-project-spec]] + PDF.
