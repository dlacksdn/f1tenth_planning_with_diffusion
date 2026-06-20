# 010 — 적대적 검수: 단계 게이트(A→B) 계획 (009 대상)

> 2026-06-20. 검수 대상 = [[009-dataset-status-problem-and-staged-plan]] (단계 게이트 A→B).
> 임무 = 009가 **새로 도입한 결정들**(단계 게이트 구조 / 완주-only Track A / 게이트 판정 기준 /
> B1 value 재조립 / B3 warm-start)을 *통과*가 아니라 *깨뜨리는* 것. 모든 판단은 npz·코드 1차 소스
> 직접 확인(파일/라인·집계 수치 명시). 009는 이미 014를 반영한 *수정* 계획이므로, **014가 한 말은
> 반복하지 않고 009의 신규 결정과 014 자신의 오류까지 검증**한다. 다음 세션 인수인계용. append-only.
> (검수자: critic 세션. 코드/실험 무변경, 비파괴 npz 조회만. git 미실행.)

---

## 0. 이 문서를 읽는 데 필요한 용어 (009에 없던 것만)

| 용어 | 한 줄 뜻 |
|---|---|
| **return-to-go (RTG)** | value가 실제로 학습하는 점수 = "이 시점부터 **에피소드 끝까지** 받을 보상의 할인 합". 128 step만이 아니라 **남은 전부**(sequence.py:146 `rewards[start:]`). |
| **effective horizon (유효 지평)** | 할인 γ=0.99에서 value가 "실질적으로 내다보는" 거리 ≈ 1/(1−γ) = **100 step(2초)**. 그보다 먼 미래 보상은 거의 0으로 줄어듦. |
| **warm-start (워밍스타트)** | 매 step diffusion을 *백지 노이즈*에서 그리지 않고, **직전에 그린 계획을 출발점**으로 삼아 일부 단계만 다시 그리는 가속/안정화 기법(Janner Fig 7). |
| **compounding error (누적 오차)** | 닫힌 루프에서 매 step의 작은 예측 오차가 *쌓여* 점점 데이터 밖으로 벗어나는 현상. open-loop로 학습한 정책의 고질병. prior가 깨끗해져도 *자동으로 사라지지 않는다*. |
| **mixture-averaging (혼합 평균)** | 출발 obs가 같으면(정지) diffusion이 cap5(느림)·cap10(빠름) 두 모드의 *평균*을 생성 → 어느 쪽도 아닌 흐릿한 주행. |

---

## 1. 판정 (요약)

**판정 = (B) 조건부.**

- **단계 게이트 *구조* 자체는 정당하다**(불필요한 복잡도 아님). 이유: P6 실패가 원인 3개(prior 오염·D3·
  compounding)가 *엉켜* 안 갈렸고, ★**아래 §4에서 1차 소스로 확정하듯 value는 구조상 "완주/안전"을 보상할
  수 없고 "근접 속도"만 보상한다 → 안전은 전적으로 prior가 책임진다.** 그러므로 "깨끗한 prior 단독으로
  안전 완주가 되는가"(=Track A)는 **계획 전체가 딛고 선 가정**이며, 이걸 고속재료·value 변수 없이 먼저
  검증하는 건 헛단계가 아니라 *핵심 가정의 단독 시험*이다. GPU 시간(+8~12h, 무인)은 그 가치에 비해 싸다.
- 그러나 **009의 신규 결정 중 4개가 1차 소스와 어긋나거나 부정확**하다(우선순위 순):
  1. **[Blocker] B3 warm-start는 "코어 무변경 글루"로 불가능** — diffusion 코어 수정 불가피(§6). 가장
     그럴듯한 실행 경로(A=⚠️)가 바로 이 B3를 호출하므로 *제일 먼저 터질* 결함.
  2. **[High] 게이트 판정 기준이 정량 불가**(n=5, "의미있게 >0", "후진 소멸" 임계 없음) → 자의적(§3).
  3. **[High] B1 벌점 증폭의 효과가 009 서술보다 좁다** — "충돌-임박" 윈도만 음수화할 뿐, value의
     구조적 *속도 편애*는 그대로(§5). 014의 "−100~−300이면 충돌 윈도 음수 역전" 산술도 **부분적으로만 참**.
  4. **[Med] Track A의 "value 안 씀"·"전체 stats 정규화 통일"은 부정확/과설계** — A는 value를 *로드*는
     강제당하고(§7), 정규화 통일은 A엔 불필요하며 009의 "ep 필터만 추가"와 모순(§2).
- **009 그대로 실행 시 가장 그럴듯한 결말**: Track A에서 후진/스핀 degenerate는 사라지나(타당), **완주는
  불안정**(⚠️, compounding 잔존 + cap5/cap10 혼합) → ⚠️ 분기가 B3 warm-start를 부르는데 **B3가 코어
  무변경으론 구현 불가** → 거기서 멈춤. 즉 *구조는 옳은데 ⚠️ 분기의 출구가 막혀 있다.* → §8 최소 수정.

---

## 2. [의문2] 완주-only prior(52)의 독립 정보량 — 양은 맞으나 *다양성*이 얇다

### 직접 집계 (RL_project .venv, numpy1.24, `is_terminal.any()`로 완주 판정)

| tier | 완주 ep | ep 길이(min/med/max step) | 2랩 시간 |
|---|---|---|---|
| cap5_full | **22** | 5709/5745/5829 | ~115s (느림) |
| cap10_full | **30** | 2807/2819/2830 | ~56s (=baseline 초과 핵심) |
| **합** | **52** | — | — |

- transition = **211,123** / 슬라이딩 윈도(H128, use_padding=True) = **211,071** → **ep당 ≈ 4,059 윈도**.
- **쉬운 설명**: 윈도가 21만 개라 "많아 보여도", 한 ep가 4천 개의 *거의 같은* 윈도를 찍어낸다(인접 윈도는
  1 step 차이 = 강상관). **독립 정보량 ≈ ep 수(52)에 가깝다.** 그 52개도 출처가 *정책 2종(cap5·cap10)·
  트랙 1개(Oschersleben)*뿐 → 다양성이 정말 얇다.

### 판단
- 009의 "양(≈450k)은 병목 아님"은 **count로는 맞다**(014 §8과 동일, Diffuser D4RL 도메인 대비 희박하지
  않음). 그러나 **"양 vs 다양성" 혼동의 잔재가 남아 있다**: 닫힌 루프 안정성을 결정하는 건 transition 수가
  아니라 *복구(recovery) 커버리지*다. 52개는 전부 **성공한 깨끗한 랩**이라 "라인 벗어났을 때 어떻게
  돌아오는가"의 예시가 거의 없다. 닫힌 루프가 한 번 드리프트하면 52궤적이 안 가본 상태 → **covariate
  shift가 Track A에서도 재현될 위험이 실재**한다. (이건 P6 진단 013 §2가 "compounding error"로 이미 지목.)
- ★**숨은 함정(009 미언급)**: 완주 52는 *단일 모드가 아니라* cap5(115s)·cap10(56s) **두 속도 모드의 혼합**
  이고, 출발 obs(정지)는 두 모드가 동일 → BC(scale=0)는 §0의 *mixture-averaging*으로 **둘의 평균(≈7~8 m/s
  혼합)을 생성**할 수 있다. → "A 통과 시 56s 완주 현실적"(009 §4.2)은 낙관. cap10 모드를 *깨끗이* 재현한다는
  보장이 없다(§8-2에서 cap10-가중 권고).

**불확실성**: covariate shift "재현 위험"은 정성 추론(닫힌 루프 미실행). 그러나 P6에서 *이미* 발생했고
(013 §2 trace t55-75 후진), 깨끗한 prior가 그걸 *제거*한다는 1차 증거는 없다(완화는 기대 가능).

---

## 3. [의문1] 게이트 판정 기준 — 현재로선 정량 판정 불가, 자의적

009 §3.2 통과 조건 = "후진 degenerate 소멸 + 완주율 의미있게 >0 + 2랩 107s 초과". 1차 소스로 본 문제:

| 기준 | 문제 | 1차 소스 |
|---|---|---|
| "완주율 의미있게 >0" (episodes≥5) | n=5로 0/5 vs 1/5를 가르는 건 통계적으로 무의미(1/5의 신뢰구간 ~0.5%~72%). 게다가 diffusion 샘플링은 **stochastic**(매 plan마다 새 노이즈) → 같은 설정도 ep마다 결과 변동 | functions.py:32-35 `noise=torch.randn_like(x); noise[t==0]=0` (t=0 외 매 step 노이즈), plan_f1tenth.py:144 `n=args.episodes` |
| "후진 degenerate 소멸" | **무슨 지표·임계인지 없음** → 눈대중이면 자의적 | — |
| "2랩 107s 초과" | 완주해야 정의됨. 완주 자체가 관문이라 사실상 1번 조건과 중복 | plan_f1tenth.py:154 `is_completed & len(lap_times)>=2` |

### 다행히 — 정량화 재료는 *이미 있다*
plan_f1tenth.py가 매 ep `cmd_speed_mean / cmd_speed_p90 / cmd_speed_max / cmd_steer_absmean`를 로깅한다
(:85-88). P6의 degenerate는 **mean cmd −4.0 m/s**(후진)였다(013 §1). 따라서:
- **"후진 소멸" = `cmd_speed_mean > 0` (가능하면 p90 > 안정 임계, 예 5 m/s)** 로 *측정 가능*하게 박을 것.
- **완주율은 n≥10~20**으로(sim이라 거의 공짜, max_steps 9000×0.02s). 1/5 같은 노이즈 판정 회피.

**판단**: 게이트 *발상*은 옳다(원인 분리). 단 **현 문구로는 게이트가 작동 안 한다** — 기존 로깅으로
임계를 숫자로 못박으면 저비용으로 살아난다(§8-3). 안 고치면 "통과/부분/실패"가 검수자 기분이 된다.

---

## 4. ★ [신규 핵심] value는 구조상 "완주"를 보상 못 한다 — 안전은 100% prior 책임

이건 009·014 둘 다 명시 안 한, **계획 전체를 지탱(또는 무너뜨리는)** 1차 소스 사실이다.

value 타깃 = RTG(γ=0.99, sequence.py:146-148). γ의 **유효 지평 ≈ 100 step(2초)**. 직접 계산:

| 랩 보너스(+100)가 ~앞에 있을 때 | value에 실제 기여 |
|---|---|
| 50 step 앞 | +60.5 |
| 100 step 앞 | +36.6 |
| 200 step 앞 | +13.4 |
| 500 step 앞 | +0.66 |
| **1400 step 앞(=실제 랩1 완주 거리)** | **+0.0001** |

- npz 실측: 완주 ep의 랩 보너스 = **+100/랩, 2랩 +200**(`log_reward_lap` sum=200.0, cap10_full/cap5_full).
  그런데 랩 라인은 보통 *수백~천4백 step 앞* → **γ로 완전 소멸**(+0.0001). 즉 **value는 "이 계획이 랩을
  완주한다"는 사실에서 아무 점수도 못 얻는다.** value가 보는 건 오직 *다음 ~100 step의 progress(=속도)*.
- 결과: **value guidance가 줄 수 있는 건 "더 빨리"뿐, "완주/안전"이 아니다.** → **안전·완주는 전적으로
  prior가 만들어야 한다.** value는 그 위에 속도만 얹는다.

### 이 사실이 009에 주는 함의 (양날)
- ✅ **단계 게이트를 정당화한다**: 안전이 prior 단독 책임이라면, "깨끗한 prior가 안전 완주를 내는가"(Track A)는
  *반드시 단독으로* 확인해야 할 가정이다. → §7 구조 정당성의 핵심 근거.
- ⚠️ **"진짜 개선<56s"의 엔진(value)을 약화시킨다**: value는 prior가 *이미 생성 가능한* 가장 빠른 모드로
  밀 뿐, 새 속도를 못 만든다. prior에 sustainable한 sub-56s 재료가 없으면(=고속 재료가 전부 충돌-직전
  한계주행이면, 014 §3 확증) value를 고쳐도 stretch는 안 나온다.

---

## 5. [의문3] B1(D3 수정) — 재조립은 가능, 그러나 효과가 009 서술보다 *좁다*

### (a) 재조립 가능성: ✅ 확정
- npz에 보상 5성분 **분리 저장 확인**: `log_reward_{progress,collision,lap,reverse,diverged}` (+합본 `reward`).
- **`reward == 성분 5개의 합` (12개 파일 max 오차 = 0)** → 충돌 벌점만 ×배율로 **재조립해 reward 변경 가능**.
  로더는 `reward`를 그대로 씀(f1tenth.py:117)이므로, 로더에서 재조립하면 value 학습 타깃이 바뀐다.
- 충돌 벌점 = **단발 −10**(터미널 1 step, `log_reward_collision` min=−10 sum=−10). progress ≈ **0.29/step
  (cap20)·0.20(cap10)·0.09(cap5)**.

### (b) ★ 효과 재계산 — 014의 산술을 *정정*한다
014 §4는 "128 step 절단"으로 고속-충돌 32.86 vs 저속-안전 8.12(4배 선호)라 했다. 그러나 **실제 value
타깃은 128이 아니라 에피소드 끝까지의 RTG**다. 실제 타깃식으로 전 윈도 재계산(γ=0.99, 벌점 −10):

| 윈도 종류 | n | RTG median | 비고 |
|---|---|---|---|
| cap20 충돌(고속) **전체 윈도** | 242k | **30.87** | 대부분 충돌서 *멀어* 벌점 ≈0 → 그냥 고속=고점 |
| cap15 충돌(고속) 전체 | 247k | 28.76 | 〃 |
| cap10_full 완주(중속·안전) | 102k | **20.11** | progress 0.20 |
| cap5_full 완주(저속·안전) | 143k | 9.73 | progress 0.09 |
| **충돌-임박 윈도(cap15/20, crash ≤128 step 전)** | 85k | **8.87** | ← 실제로 위험한 구간 |

- **정정 1**: "위험한" 충돌-임박 윈도(8.87)는 cap10 완주(20.11)보다 *이미 낮다* — 벌점 −10에서도. 014의
  "value가 충돌을 4배 선호"는 *고속 vs 저속* 비교였지 *충돌 vs 안전* 비교가 아니었다. **진짜 D3는 "충돌
  선호"가 아니라 "속도 선호"**다(cap20 전체 30.87 > cap10 20.11 — 빠르니까, 충돌이 멀어 안 보임).
- **정정 2 (B1 한계)**: 벌점을 키워봐야 **충돌-임박 윈도만** 음수가 되고 *대부분의 고속 윈도는 안 변한다*:

  | 충돌 벌점 | 충돌-임박(≤128) median | cap20 **전체** median | cap20 윈도 中 cap10완주(20.1) 초과 비율 |
  |---|---|---|---|
  | −10 | 8.87 | 30.87 | 85% |
  | −50 | −12.12 | — | — |
  | −100 | −38.51 | **30.24** | **74%** |
  | −300 | −143.82 | — | — |

  → **−100으로 키워도 cap20 전체 분포는 30.87→30.24로 거의 안 움직인다**(초기 고속 윈도는 충돌이 ~1000
  step 앞이라 γ로 벌점 소멸). 014의 "−100~−300이면 충돌 윈도 음수 역전"은 *충돌 직전 128 step에 한해서만*
  참이다. **value의 속도 편애(고속=고점)는 벌점 증폭으로 안 고쳐진다.**

### 판단
- B1은 **헛계획은 아니다**: 충돌-임박(=벽으로 돌진하는 H128 plan)을 value가 *저평가*하게 만든다 → guidance가
  "곧 충돌할 모양의 plan"을 피하게 된다. 이건 진짜 이득(guidance 시점에 value가 보는 건 H128 plan이므로).
- 그러나 009 §2.3/B1의 서술("value가 충돌을 4배 선호 → 벌점 키우면 부호 역전 → 해결")은 **과대**다.
  벌점 증폭은 *충돌-임박 회피*만 주고, value의 본질("완주 무보상 + 속도 보상", §4)은 그대로다.
- ★**B1의 무비용 게이트를 *제대로* 하려면 비교 대상을 바꿔야 한다**: "고속-충돌 vs 저속-안전"(쉬운
  역전)이 아니라 **"충돌-임박 H128 vs *지속가능* 고속 H128(cap15/20 lap1 중 lap2까지 버틴 것)"**의 점수
  분리를 봐야 한다. 그게 안 갈리면 guidance는 여전히 한계주행으로 민다(§8-4).
- "diffusion 재사용, value만 재학습"의 정규화 정합: ✅ **성립**. reward 변경은 obs/action 정규화와 무관
  (정규화는 obs/action만, reward는 아님 — sequence.py:53 keys=['observations','actions']). 단 *어느
  diffusion을 재사용하냐*가 문제: B의 value는 **B2의 driving-prior diffusion과 짝**지어 쓰이므로, 둘이
  같은 정규화(전체 stats)를 공유해야 한다(§2·§8-5). A의 complete-only diffusion은 B에서 안 쓰임(B2 재학습).

---

## 6. ★ [의문5/Blocker] B3 warm-start는 "코어 무변경 글루"로 **불가능**

009 §3.3 B3: "직전 계획을 초기값/조건으로 이어 그림. 코어 무변경(plan_f1tenth.py 글루)." 014 §9-3도
"sampling 글루 수준에서 가능"이라 했다. **둘 다 1차 소스로 반박된다.**

### 1차 소스
```
diffusion.py:158  def p_sample_loop(self, shape, cond, ...):
diffusion.py:163      x = torch.randn(shape, device=device)   # ← 초기 노이즈 하드코딩, 인자 없음
diffusion.py:169      for i in reversed(range(0, self.n_timesteps)):   # 항상 T=20부터 0까지 full chain
diffusion.py:184  def conditional_sample(self, cond, horizon=None, **sample_kwargs):  # x 주입 경로 없음
diffusion.py:231  def forward(self, cond, *args, **kwargs): return self.conditional_sample(...)
policies.py:28    samples = self.diffusion_model(conditions, guide=..., **self.sample_kwargs)  # x 못 넣음
```
- **"초기값 주입"형 warm-start(Janner Fig 7: 직전 plan을 forward-noise해 중간 t부터 역방향)**: 초기 `x`는
  `p_sample_loop` *내부에서* `torch.randn`으로 생성되고, 역방향 루프도 항상 t=T부터 시작한다. 외부에서
  `x_init`이나 `t_start`를 넣을 인자가 `forward`/`conditional_sample`/`p_sample_loop` 어디에도 **없다.**
  → **diffusion.py(코어) 수정 불가피.** 제약("코어 temporal/diffusion/helpers/trainer 무변경, ValueFunction
  fork-patch가 *유일* 예외")을 **깬다** = *두 번째* 코어 예외. 글루로 안 된다.

### 살릴 수 있는 변형 (다른 메커니즘)
- **"조건 주입"형**은 글루 가능: cond dict에 `{0:obs_now, 1:prev_obs1, ..., K:prev_obsK}`를 넣으면
  `apply_conditioning`(helpers)이 그 timestep들의 *관측*을 고정한다. GuidedPolicy._format_conditions가
  임의 timestep cond를 받음(policies.py:49-61). **그러나** ① 고정하는 게 *직전 plan의 예측 관측*이라
  그게 틀리면 더 망가짐, ② 이건 warm-start(노이즈 재사용)가 아니라 *waypoint 조건화*라 compounding 완화
  효과가 다름(미검증), ③ 009/014가 의도한 "초기값 이어 그리기"와 다른 물건.

### 판단
- **B3는 009/014 서술대로면 헛계획**(코어 무변경 위반). 그런데 §1에서 봤듯 **가장 그럴듯한 실행 경로
  (A=⚠️)가 정확히 B3를 호출**한다 → 이 결함이 *실제로* 길을 막는다. 우선순위 1.
- 선택지(§8-1): (ㄱ) B3를 **포기**하고 ⚠️ 대응을 다른 레버(K 중간값·cap10-가중 prior·scale)로, (ㄴ)
  warm-start를 **코어 예외 2건째로 명시 승인**(diffusion.py에 `x_init`/`t_start` 인자 추가, ValueFunction
  처럼 fork-patch 기록), (ㄷ) **조건-주입형**으로 재정의(글루 가능하나 효과 미검증). over-plan 금지면 (ㄱ).

---

## 7. [의문6/7] Track A의 value 처리 & 단계화 구조 정당성

### (의문6) Track A는 "value 안 씀"이 아니다 — *로드는 강제*당한다
1차 소스:
- `n_step_guided_p_sample`(functions.py:17-27)은 **scale과 무관하게** 매 step `guide.gradients(x,cond,t)`를
  호출(line 19) → value 모델 forward 필수. scale=0이면 `x = x + 0·grad`(line 26)로 *출력엔 무효*지만
  **value는 여전히 로드·호출된다.**
- `build_policy`(plan_f1tenth.py:108-114)는 **항상** diffusion+value를 로드하고 ValueGuide를 만든다.
  value 없이 도는 분기가 *없다*. `check_compatibility`(line 110)도 호출됨(타입만 검사라 통과, §아래).
- → **Track A를 돌리려면 `values/f1tenth_H128_T20_d0.99`에 value ckpt가 *존재*해야 한다.** 다행히 P6의
  (D3-오염) value가 거기 있다. scale=0이면 그 오염 value의 출력은 ×0으로 버려져 **무해**. 정규화 불일치도
  ×0이라 무해. check_compatibility는 `type()`·`n_timesteps`만 봐서(serialization.py:62-79) SafeLimits==
  SafeLimits·20==20으로 통과.
- **판단**: 009 §3.2의 "value는 안 씀"은 *기능적으론 참*(scale=0=무효)이나 *인프라적으론 거짓*(반드시
  로드). 문서에 "**기존 value ckpt를 로드만 하고 scale=0으로 무효화**"라 정확히 적어야, 새 세션이 "value
  없이 BC 돌리겠다"고 plan_f1tenth를 잘못 고치는 사고를 막는다. (저비용 정정, §8-3.)

### (의문7) 단계화 구조 — *정당하다*. 불필요한 복잡도 아님
- **누적 비용**(1차 실측 기반: P4 rate 2.86 step/s, 단일 학습 7GB/8GB라 **동시 학습 불가=순차 강제**):
  A2(complete-only, 데이터 작아 ~4-6h 조기 saturate 가능) + B1(value ~5-10h) + B2(driving diffusion
  ~6-8h) + 평가들 ≈ **20~30h 순차**. 013 일괄(diffusion ~6-8h + value ~5-10h ≈ 13~18h) 대비 **+8~12h**.
  거의 전부가 A2(진단 전용, B에서 B2가 diffusion 재학습하므로 A2 diffusion은 재사용 안 됨)와 B1→B2 순서화.
- **단계화가 주는 이득 vs 비용**: §4에서 확정했듯 **value는 안전을 못 만든다 → 안전은 prior 단독 책임**.
  그러면 "깨끗한 prior가 단독으로 안전 완주를 내는가"는 *계획 전체의 가정*이고, P6는 이걸 prior오염·D3·
  compounding과 *엉켜서* 못 갈랐다. A는 이 가정을 고속재료·value 없이 *단독 시험*한다. **GPU 시간(무인,
  인간 비용 0)으로 핵심 가정을 산다 → 기대가치 양(+).** 일괄 실행은 또 실패하면 변수 2개라 원인 불명 →
  P6 재연. → **단계화 채택이 옳다.**
- 다만 정직하게: A2 diffusion은 **버리는 카드**(B2가 재학습). 009가 "A 통과면 56s 확보"를 *수확*처럼 적지만
  (§4.2), A의 diffusion은 stretch(B)로 못 이어진다 — A는 *진단*이지 *자산 축적*이 아니다. 문서에 명시 권고.

---

## 8. 009를 살리는 최소 수정 (우선순위 — over-plan 금지, 진짜 필요한 것만)

1. **[Blocker] B3 warm-start를 재정의하라.** "코어 무변경 글루"는 거짓(§6, diffusion.py:163). 권고:
   **B3를 기본 경로에서 빼고**, A=⚠️ 대응을 *코어 무변경으로 되는 레버*로 먼저: **(ㄱ) cap10-가중 완주
   prior**(§8-2) **+ (ㄴ) K 중간값 스윕(2/3/5)** + (ㄷ) scale 소폭. 그래도 부족하면 그때 warm-start를
   **"코어 예외 2건째"로 명시 승인**(diffusion.py에 `x_init/t_start` 인자, fork-patch 기록) 하거나
   조건-주입형으로 한정. ★지금 009는 ⚠️→B3로 흐르는데 그 B3가 막혀 있다 = 길이 끊긴다.

2. **[High] Track A를 "완주 52 균등"이 아니라 "cap10 완주 가중"으로.** 완주 52 = cap5(115s)·cap10(56s)
   혼합이라 BC가 *평균 모드*(흐릿)를 뽑을 위험(§2, mixture-averaging). 목표가 "cap10급 56s 안전 완주"라면
   **cap10_full 30개에 가중**(또는 cap10-only로 A를 한 번 더)하는 게 "56s 현실적"의 전제. (로더에서
   tier 필터 1줄.)

3. **[High] 게이트 기준을 기존 로깅으로 정량화.** "후진 소멸" → **`cmd_speed_mean > 0` (목표 p90 ≥ ~5)**;
   "완주율" → **n ≥ 10~20**(sim 거의 공짜). + Track A는 "value 로드하되 scale=0 무효화, 기존 ckpt 재사용"
   이라 명시(§7, plan_f1tenth 오개조 방지).

4. **[High] B1 무비용 게이트의 비교 대상 교정.** "고속-충돌 vs 저속-안전"(자명한 역전)이 아니라
   **"충돌-임박 H128 vs 지속가능-고속 H128(cap15/20 中 lap2까지 버틴 ep의 lap1 윈도)"**의 분리를 봐야
   D3 수정이 *guidance에 실효*인지 판정 가능(§5). 안 갈리면 벌점 증폭은 충돌-임박 회피만 줄 뿐 한계주행
   편애는 못 고침을 *문서에 명시*하고 stretch 기대를 낮출 것.

5. **[Med] 정규화 정책 단순화.** Track A는 value 무효(scale=0)라 **all-stats 통일이 불필요** — complete-only
   stats로 충분(자기 정합만 되면 됨). 009 A1의 "어떤 모드든 전체 stats fit"은 (a) A엔 불필요, (b)
   "ep 필터만 추가(line 107~)"와 *모순*(iterator서 ep 거르면 normalizer도 complete-only가 됨; 전체 stats로
   fit하려면 *전체 로드 후 make_indices만 제한*하는 더 큰 변경 필요). → **all-stats 통일은 B(value guidance
   쓸 때)에만** 적용하고, 그때 **B2 diffusion과 B1 value가 같은 전체-stats normalizer 공유**(check_
   compatibility는 stats 미검사라 *사람이* 보장해야 함, serialization.py:62-79).

> §4(value는 완주 무보상)·§5-정정(D3는 속도 편애)·§6(warm-start 코어 필요)는 보고서의 정성 분석 산출물로
> 그 자체가 가치 있다 — 009 §4.2의 "실패해도 분석은 유효" 주장에 동의하며, 위 1차 소스 수치로 *강화*된다.

---

## 9. 결론 — "009 그대로 실행하면 무엇이 가장 그럴듯한가"

- **Track A(완주 prior, scale=0 BC)**: P6의 후진/스핀 degenerate는 **사라질 것**(prior 오염 제거, 타당).
  차가 앞으로 간다. 그러나 **완주는 불안정**할 공산이 크다 — ① compounding error는 prior 청소로 안
  사라지고(§2, 013 §2가 동급 원인으로 지목), ② 완주 52가 cap5/cap10 *혼합*이라 BC가 흐릿한 평균을 뽑을
  수 있어서(§2). → **가장 그럴듯한 게이트 결과 = ⚠️(전진 회복, 저완주율/불안정)**, ✅(안정 완주)는 그
  다음. 009는 ✅를 표제로 적지만 ⚠️가 base case.
- **그 ⚠️가 009 §3.4 흐름도상 B3(warm-start)를 호출** → **B3는 코어 무변경으론 구현 불가(§6)** →
  **거기서 막힌다.** 즉 *구조는 옳은데 ⚠️ 출구가 끊겨 있다.* 이게 009 그대로 실행 시 1순위 실패 양상.
- **Track B 진입(A=✅ 가정 시)**: B1 벌점 증폭은 *충돌-임박 회피*만 주고(§5), B2가 넣을 고속 재료
  (cap15/20 lap1)는 *충돌-직전 한계주행*(014 §3 확증)이라, value(속도만 보상, §4)가 그 한계 모드로 밀어
  **닫힌 루프 lap2 충돌 재현** 위험이 가장 크다. → **<56s "진짜 개선"은 여전히 낮은 확률.** 정직한
  최선은 "cap10급 안전 완주 복원(56~115s, baseline 107s는 *cap10 모드를 잡아야* 초과) + 충돌-위주 데이터의
  D3·compounding·한계주행-재현 실패의 정량 분석".

**판정 재확인: (B) 조건부.** *단계 게이트 구조는 정당*(§4·§7가 근거). 단 **§8-1(B3 재정의)·8-3(게이트
정량화)** 없이는 ⚠️ 경로에서 멈추고, **§8-2·8-4·8-5** 없이는 "56s 현실적/진짜 개선"이 희망에 가깝다.
009의 가장 큰 강점은 *원인 분리*라는 발상이고, 가장 큰 빈틈은 *그 분리가 ⚠️로 끝났을 때의 출구(B3)가
1차 소스상 막혀 있다는 것*이다.

---

## 10. 검수자 메모 (방법·불확실성)
- 1차 소스: npz 직접 적재·집계(RL_project `.venv`, numpy1.24) — 완주 52/transition 211,123/윈도 211,071,
  reward=성분합(오차 0), RTG 분포(γ=0.99 실제 타깃식 backward recursion), 벌점 스윕, 정규화 분포 폭.
  코드 전문 읽음: f1tenth.py·sequence.py·config/f1tenth.py·functions.py·diffusion.py·policies.py·
  guides.py·serialization.py·normalization.py·plan_f1tenth.py(라인 본문 인용).
- **산술만 한 곳**(직접 실험 아님): §4 랩보너스 할인 기여(γ^k×100), §2 covariate shift "재현 위험"(닫힌
  루프 미실행 추론 — 단 P6서 실제 발생). 나머지는 npz/코드 실측.
- **불확실성**: ⚠️ vs ✅ 확률은 정성 판단(닫힌 루프 비실행). B1 벌점 증폭이 *guidance 실효*로 이어질지는
  §8-4 비교를 *해봐야* 확정(여기선 RTG 분포로 "충돌-임박만 음수, 속도 편애 잔존"까지만 확증).
- 비파괴 조회만. 코드/실험 무변경. git add/commit/push/pull 미실행(규약).
- 참조: [[009-dataset-status-problem-and-staged-plan]](대상) · [[013-p6-failure-diagnosis-and-driving-prior-plan]]
  (P6 진단) · [[014-adversarial-critic-of-driving-prior-plan]](009 이전 검수 — 본 문서가 §5(D3 산술)·
  §6(warm-start 글루)에서 정정).
