# 001 — 모델 크기 검토 + 논문 v3 대조

> 2026-06-13. 사용자 질문 "Dreamer 때 200M→12M 다운스케일처럼 LeWM도 고려할 게 없나?"에 대한
> 분석 + raws/LeWM v3.pdf(2026-06-03 업데이트) 대조 기록. append-only.

---

## 1. 모델 크기 결정: **정본 config 그대로 유지** (다운스케일 불필요·금지)

### 배경
Dreamer 프로젝트에서는 default 200M을 하드웨어(RTX 4060 Ti 8GB) 고려로 12M으로 축소했음
(논문에 visual 작업에서 12M≈200M 근거 있음). LeWM에 같은 고민이 필요한지 검토.

### 결론: 불필요 — LeWM은 default가 이미 "12M 포지션"
- 논문 abstract: "With **15M parameters** trainable on a **single GPU** in a few hours"
- **실측** (Step 0 smoke ckpt `stablewm_home/checkpoints/lewm/weights_epoch_1.pt`, tworoom
  action_dim=2 기준): **총 18.04M** = predictor 10.79M + encoder(ViT-tiny) 5.50M +
  projector 0.80M + pred_proj 0.80M + action_encoder 0.16M

### 오히려 "줄이면 안 된다"는 ablation 근거 (v3 기준 유지 확인)
| 항목 | 논문 결과 | 시사점 |
|---|---|---|
| Embedding dim (Fig. 15) | **~184 아래 임계에서 성능 급락**, 192 이상 saturation | default 192 = 임계 바로 위 최적. 축소 금물, 확대 무익 |
| Predictor 크기 (Tab. 6, Push-T SR) | tiny 80.67 / **small 96.0** / base 86.7 | 축소·확대 모두 성능 하락. 공개 config(10.79M)가 best 지점 |
| Encoder 아키텍처 (Tab. 8) | ResNet-18로 교체해도 competitive | 인코더 선택에 둔감 — 변경 불필요 |
| λ (sigreg weight) | [0.01, 0.2]에서 평탄, 0.09 peak, 0.5만 급락 | 유일 하이퍼파라미터, 정본 0.09 유지 |

### 8GB GPU에서의 실질 병목 (모델 크기 아님)
- 유일 조정 포인트 = **batch 128 수용 여부** (bf16 ViT-tiny라 가능성 높음). 본 학습 전 스윕 1회.
  미수용 시 모델 축소가 아니라 batch↓ + lr 조정으로 해결
- 실측 처리량: batch 16에서 10.5 it/s → 730k 윈도우 × 10 epoch ≈ 수 시간~하루급 (가능 범위)
- CEM planning(300 samples × 30 iter)은 추론이라 8GB 무관

⇒ Phase 2 설계서에 "모델 크기 = 정본 유지 (근거: Tab.6/Fig.15)"로 확정 예정.

---

## 2. 논문 v3 대조 (v2 2026-03-24 → v3 2026-06-03, 29→28p)

### 유지 (우리 의사결정에 쓴 수치 전부 그대로)
15M·단일 GPU, frameskip=5(action block), λ=0.09, embed 임계 184, predictor small best,
데이터 규모(TwoRoom 10k×92 / PushT 20k×196 / Cube·Reacher 10k×200), **10 epoch 충분**,
CEM(300 samples, top-30, 30/10 iter), 평가 프로토콜(goal-reaching, budget 50/150).
**SIGReg 데이터 다양성 리스크도 유지** — 결론부로 이동, "low data diversity weakens SIGReg
in simple, low-dimensional environments"로 재표현. 우리 리스크 식별 유효.

### 추가
- **"RealImagined" rollout 시각화** (Fig 7/9): context 3프레임 인코딩 → action 조건부 open-loop
  latent 자기회귀 생성 → **사후(a posteriori) 학습 decoder**로 복원해 real vs imagined 비교.
  정성 결과: 전체 장면 구조 보존, 미세 디테일(end-effector 각도 등) 손실
- **Fig 10 Decoding Latent Space**: 학습에 reconstruction 미사용임에도 진단용 decoder가
  192-dim latent에서 장면 복원 가능 확인
- 섹션 재구성(4.2 Towards Efficient Planning / 4.3 Towards Stable Training), L40S 명시 삭제

### 프로젝트 시사점
- **사후 decoder 진단 기법 차용 가치**: Phase 3-4에서 MPC 돌리기 전, f1tenth world model이
  트랙 구조를 제대로 상상하는지 open-loop rollout을 decoder로 시각화해 정성 진단
  (학습 파이프라인 무변경, 진단용 소형 decoder만 별도 학습)
- v2 pdf 삭제 무방: 핵심 사실은 implementation/001 §6과 본 문서에 기록, arXiv 버전별 영구 접근
  가능(2603.19312v2). 이후 참조 기준 = v3
