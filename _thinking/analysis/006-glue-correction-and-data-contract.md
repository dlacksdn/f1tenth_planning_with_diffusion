# 006 — 글루 touch-point 정정 + 데이터 계약 실검증 (P0 구현 지도)

> 2026-06-14. [[005-diffuser-glue-touchpoints]]의 **import-bomb 섹션이 실측으로 부정확/불완전**함이
> 드러나 정정한다. + Diffuser 데이터 계약을 **실코드 + 실 npz**로 재검증. venv 구축 맥락·전체
> 인수인계는 [[005-diffuser-venv-and-handoff]](env_setting) 참조. **P0 구현은 005-glue가 아니라
> 이 문서를 따를 것.** append-only.

---

## 1. import 폭탄 정정 ([[005-diffuser-glue-touchpoints]] §1 supersede)

005-glue는 "datasets/d4rl.py:23-26의 `import d4rl`가 핵심 폭탄"이라 했으나, 새 py3.8 venv
(d4rl/mujoco/imageio 의도적 미설치)에서 코어 import를 실측해 보니 더 정확히는:

- `diffuser/__init__.py: from . import environments` → **양성(benign)**. `registration.py`가
  `gym.register(entry_point=문자열)`만 호출 + 전체 `try/except`로 감쌈. mujoco env 파일(hopper.py
  등)은 `gym.make` 시에만 로드. **`gym`만 있으면 통과**(f110_gym으로 충족). 패치 불필요.
- **진짜 코어 블로커** = `diffuser/models/helpers.py:10 import diffuser.utils as utils`
  → `utils/__init__.py:6 from .rendering import *`
  → `utils/rendering.py` top-level이 **`imageio`(:4) · `mujoco_py`(:8) · `from .video import ...`(:13)
  · `from diffuser.datasets.d4rl import load_environment`(:15)** 를 import
  → 전부 미설치 → **temporal/diffusion 코어까지 동반 사망**.
  (실측: `from diffuser.models.temporal import TemporalUnet` 첫 실패 = `No module named 'imageio'`,
  mujoco가 아님.)
- `tap`(typed-argument-parser)·`gym`은 **venv가 이미 해결** → P0 코드 패치 대상에서 제외(005 대비 축소).

## 2. P0 디커플 목록 (확정 — 게이트 포함)

1. **`utils/rendering.py`의 heavy import(imageio/mujoco_py/video/d4rl) guard 또는 lazy화** — 또는
   `utils/__init__.py`의 `from .rendering import *`를 NullRenderer로 우회. **코어 import를 살리는 핵심.**
2. **`datasets/d4rl.py:23-26 import d4rl`** try/except graceful degrade.
3. **`datasets/buffer.py:12 np.int → np.int64`** (numpy 1.24.4에서 `np.int` 제거 = 에러, 실측 확인).
4. **f1tenth 로더 신설**(`datasets/f1tenth.py`) + **`datasets/sequence.py:22-23`의
   `load_environment`/`env.seed` 스텁** + **`config/f1tenth.py`** 신설. (상세 배선은 005-glue §3~9 유효.)
5. **timeouts 함정 처리**: config `termination_penalty=None` 또는 로더에서 `timeouts = is_last & ~is_terminal`
   합성 (§3-주의, [buffer.py:79]).

**게이트**: 새 .venv에서 `from diffuser.models.temporal import TemporalUnet` 통과 +
`train.py` 더미 로더로 **1-step loss 산출**. (코어 temporal/diffusion/helpers는 무변경.)

## 3. 데이터 계약 실검증 ([[003-diffuser-code-analysis]] §1 정밀화)

**Diffuser가 에피소드 dict에서 실제 쓰는 키** (SequenceDataset→ReplayBuffer.add_path):
| 키 | 쓰임 | 필수성 |
|---|---|---|
| observations (T,obs_dim) | trajectory·conditions·obs_dim [sequence.py:37,85,89] | 필수 |
| actions (T,act_dim) | trajectory·act_dim [sequence.py:38,86] | 필수 |
| rewards (T,) | value=할인 return-to-go [sequence.py:141-143] | 필수 |
| terminals (T,) | 종료 로직 [buffer.py:78] | 필수 |
| timeouts (T,) | `assert not path['timeouts'].any()` [buffer.py:79] | 조건부(§2-5) |
구조 계약: **obs_dim 에피소드 간 동일**[buffer.py:57-61], **max_path_length ≥ 최장 에피소드**[buffer.py:66].

**Dreamer npz 실물**(`RL_project/runs/stage2_oschersleben/train_eps`, 265 ep 실측):
keys=`lidar(T,1080)·state(T,5)·action(T,2,정규화[-1,1])·reward·is_terminal·is_last·discount·log_*`.
**obs_dim(lidar+state)=1085 전부 일관**, 에피소드 길이 min45/median161/**max1881**, 완주(log_completed≥1)=6.
| Diffuser 요구 | Dreamer 제공 | 판정 |
|---|---|---|
| observations | concat(lidar,state)=1085D **또는** centerline 피처~10D | ✅ / 피처는 pose 필요 |
| actions | action(이미 [-1,1]) | ✅ |
| rewards | reward 또는 log_reward_progress(+축소 collision) | ✅ |
| terminals | is_terminal | ✅ |
| timeouts | 없음 → §2-5 처리 | ⚠️ |
- **필수 4키는 기존 데이터로 충족**(rename+concat). `discount·log_*·logprob`는 Diffuser 미사용.
- **pose 부재** = 유일한 진짜 데이터 갭 → centerline 피처(D1)는 **P1 warm-load 재학습 policy를
  pose 기록하며 재rollout**할 때 해소. lidar+state 백업표현이면 기존 265 ep 그대로 사용 가능.
- ReplayBuffer 사전할당(host RAM): 1085D → 2.16GB(감당), 피처10D → 20MB.
