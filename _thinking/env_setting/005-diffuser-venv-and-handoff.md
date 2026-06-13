# 005 — Diffuser 전용 venv 구축 + 의존성·분석·계획 인수인계

> 2026-06-14. LeWM→Diffuser 전환 이후 **이 프로젝트(`f1tenth_planning_with_diffusion`)
> 전용 venv를 새로 구축**하고, 그 과정에서 실측으로 확정한 의존성/데이터/계획 상태를
> 한 문서에 인수인계용으로 정리. 모든 수치·경로는 **추정이 아니라 실제 실행 결과**.
> append-only. 선행: [[003-diffuser-plan-final]], [[002-diffuser-critique]],
> [[003-diffuser-code-analysis]], [[005-diffuser-glue-touchpoints]], [[004-lewm-env-setting]].
> f1tenth 과제 자체 명세는 [[003-project-spec]] + `_thinking/raws/AIE4003_RL_F1TENTH.pdf` 참조.

---

## 0. 한 장 요약

- **왜 새 venv**: 옛 `.venv`(py3.10, LeWM용 swm/torch2.12)는 Diffuser와 무관. RL_project venv
  차용은 **다른 프로젝트 venv를 갖다 쓰는 설계 위생 위반**(사용자 지적, 정당). → 이 프로젝트가
  자기 venv를 가져야 함.
- **결정(사용자 확정)**: **단일 py3.8 올인원 venv** — Diffuser **학습 + 평가(f110_gym 포함)**를
  자급. **데이터 생성(Dreamer rollout)만 RL_project venv**에서(그 프로젝트가 데이터 출처이자
  Dreamer 에이전트의 본거지라 차용 아님).
- **결과**: 구축·검증 **완료**. 코어 런타임(torch+cuda / numpy / einops / tap / f110_gym) **전부 통과**.
  Diffuser 코어 import는 **P0 디커플 코드 패치 후** 가능(venv 결함 아님, §2.4).
- **핵심 교훈**: gym 0.18.0(레거시 sdist) 빌드가 **최신 pip/setuptools/wheel을 거부** → 빌드
  툴체인을 옛 버전으로 고정해야만 설치됨(§1.3). 이게 이번 구축의 진짜 난관이었다.

---

## 1. venv 구성 (어떻게 구축했나) — 재현 레시피

### 1.1 정체성
| 항목 | 값 |
|---|---|
| 경로 | `/home/dlacksdn/f1tenth_planning_with_diffusion/.venv` |
| Python | **3.8.10** (`/usr/bin/python3.8` 시스템) — f110_gym(gym0.18) 호환 위해 3.8 고정 |
| 역할 | Diffuser **학습(P0/P4)** + **평가(P5/P6, f110_gym 구동)** |
| 옛 venv | `.venv` → `.venv_lewm_deprecated`로 rename(7G, 폐기 인가됨). `rm -rf`로 회수 가능 |

### 1.2 포함 / 제외 (설계 의도)
**포함**: torch(코어) · einops · numpy · tap(글루) · matplotlib/tqdm/pandas/gitpython(글루) ·
f110_gym(평가 시뮬, editable) · diffuser(모델 코드, editable).
**제외(의도적)**: `d4rl` · `mujoco_py` · `imageio`/`scikit-video`(렌더링) · `jax`/`flax`/`ray`/`wandb`/
`doodad`/`gdown`(D4RL 벤치마크 인프라). → 렌더링은 NullRenderer로 스텁, d4rl 로더는 f1tenth 로더로
대체할 것이므로 불필요(§2.4, §3).

### 1.3 ★ 빌드 함정 — gym 0.18.0 레거시 sdist (이번 구축의 핵심)
f110_gym은 `gym==0.18.0`을 정확히 의존하는데, 이 옛 sdist가 **2026년 최신 툴체인을 3중으로 거부**:
1. **setuptools 75.3.4** → `egg_info` 단계에서 `'extras_require' must be a dictionary...` 하드 에러.
2. **pip 25.0.1(≥24.1)** → gym 메타데이터의 불량 스펙 `opencv-python>=3.`(끝에 점) 거부.
   pip이 직접 *"use pip<24.1"* 안내.
3. **wheel 0.45.1** → `bdist_wheel`의 vendored packaging 파서가 같은 불량 스펙에서 `ParserSyntaxError`.

> RL_project venv가 멀쩡한 이유: gym을 **옛 툴체인 시절 한 번 빌드**해 둔 뒤 wheel만 0.45.1로
> 업그레이드된 상태(설치본은 재빌드 안 됨). 즉 "현재 버전 매칭"으로는 재현 불가 → **빌드 시점
> 툴체인을 옛 버전으로 내려야** 함.

**해법(고정)**: 빌드 직전 툴체인을 내리고 `--no-build-isolation`으로 venv 툴체인을 강제 사용:
```bash
PY=/home/dlacksdn/f1tenth_planning_with_diffusion/.venv/bin/python
$PY -m pip install "setuptools==44.0.0" "wheel==0.37.1"
$PY -m pip install "pip==22.0.3"
$PY -m pip install --no-build-isolation "gym==0.18.0"            # 여기서 성공
$PY -m pip install --no-build-isolation -e /home/dlacksdn/f1tenth_RL_project/gym  # f110_gym
```

### 1.4 전체 빌드 순서 (실제 수행한 명령)
```bash
ROOT=/home/dlacksdn/f1tenth_planning_with_diffusion
mv "$ROOT/.venv" "$ROOT/.venv_lewm_deprecated"          # 옛 venv 비키기
/usr/bin/python3.8 -m venv "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
# (1) torch — RL_project와 동일, slim(vision/audio 제외)
$PY -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.4.1+cu124"
# (2) 코어/글루 의존성 (numpy 1.24.4 핀, einops 0.3.0)
$PY -m pip install "numpy==1.24.4" "einops==0.3.0" typed-argument-parser matplotlib gitpython tqdm pandas pillow
# (3) 빌드 툴체인 다운그레이드 후 gym0.18 + f110_gym (§1.3)
$PY -m pip install "setuptools==44.0.0" "wheel==0.37.1" "pip==22.0.3"
$PY -m pip install --no-build-isolation "gym==0.18.0"
$PY -m pip install --no-build-isolation -e /home/dlacksdn/f1tenth_RL_project/gym
# (4) diffuser (clone editable; setup.py는 의존성 0인 빈 껍데기)
$PY -m pip install -e /home/dlacksdn/planning_with_diffusion
```
> 주의: gym 0.18 설치가 **Pillow 10.4.0 → 7.2.0으로 다운그레이드**함(gym 핀). matplotlib 3.7.5는
> Pillow≥6.2면 되므로 무해. 빌드 툴체인(pip22/setuptools44/wheel0.37)은 **그대로 두면 됨**
> (런타임 무관, 향후 다른 빌드 필요 시에만 재고).

### 1.5 최종 버전 동결 (freeze 실측)
```
torch==2.4.1+cu124   triton==3.0.0   numpy==1.24.4   einops==0.3.0
typed-argument-parser==1.10.1   matplotlib==3.7.5   pandas==2.0.3
GitPython==3.1.50   tqdm==4.68.2   Pillow==7.2.0
gym==0.18.0   cloudpickle==1.6.0   pyglet==1.5.0   scipy==1.10.1
f110-gym==0.2 (-e /home/dlacksdn/f1tenth_RL_project/gym)
numba==0.58.1   llvmlite==0.41.1   PyYAML==6.0.3
diffuser (-e /home/dlacksdn/planning_with_diffusion)
[빌드 툴체인 고정] pip 22.0.3 / setuptools 44.0.0 / wheel 0.37.1
+ nvidia-*-cu12 (12.4 계열) 런타임 라이브러리 일괄
```

### 1.6 검증 결과 (실측 출력)
```
python 3.8.10
torch 2.4.1+cu124 | cuda True | cuda 텐서연산(4x4 matmul) OK
numpy 1.24.4 | np.int 제거됨: True        ← buffer.py:12 패치 필요 근거
einops 0.3.0 | tap OK
f110_gym OK -> /home/dlacksdn/f1tenth_RL_project/gym/f110_gym/__init__.py
import diffuser OK (environments gym.register 통과)
```
→ **학습/평가 런타임 토대 정상**. 남은 건 코드 디커플(§2.4).

### 1.7 교수님 셋업 명세(PDF p29+ = [[001-configuration]]) 대비 상태 — 전부 충족, 조치 불필요
PDF가 요구하는 시뮬 "대체/수정"이 우리 구성에 이미 반영돼 있는지 실측 검증:
| PDF 지시 | 우리 상태 | 조치 |
|---|---|---|
| §3 `setuptools<58` + `pip==22.0.3` 고정 (중요) | §1.3에서 독립적으로 동일 적용(setuptools44/pip22.0.3/wheel0.37) — **PDF가 공식 절차로 확인** | 불필요 |
| §5 `base_classes.py` check_collision (`np.maximum`→`new_collisions`) | RL_project/gym에 **이미 적용**(base_classes.py:433-434 주석처리+교체 확인) | 불필요 |
| §4 `f110_env.py` 대체 | RL_project/gym 커스텀본 존재, Dreamer wrapper(`f1tenth_env.py:23`)가 직접 import | 불필요 |
| §4 `dqn.py` / `map_easy.*` | DQN 튜토리얼·map_easy는 우리(Diffuser+Dreamer, Oschersleben 단일)와 무관 | 스킵 |
> **핵심 메커니즘**: 평가용 f110_gym을 `/home/dlacksdn/f1tenth_RL_project/gym` **editable로 공유** →
> RL_project가 적용한 sim 수정(check_collision 등)이 새 venv에 **자동 상속**. 별도 대체/수정 없음.
> ⚠️ 역으로, 이 gym 소스를 우리가 수정하면 RL_project에도 영향(공유 소스)이니 sim 코어는 건드리지 말 것.

---

## 2. 무엇을 분석했나 (실코드·실데이터 검증)

### 2.1 Diffuser 데이터 계약 (요구 항목) — 실코드 확인
`SequenceDataset`→`ReplayBuffer.add_path`가 **에피소드 dict에서 실제로 쓰는 키**:
| 키 | 쓰임 (file:line) | 필수성 |
|---|---|---|
| `observations` (T,obs_dim) | trajectory·conditions·obs_dim ([sequence.py:37,85,89]) | 필수 |
| `actions` (T,act_dim) | trajectory·act_dim ([sequence.py:38,86]) | 필수 |
| `rewards` (T,) | value=할인 return-to-go ([sequence.py:141-143]) | 필수 |
| `terminals` (T,) | 종료 로직 ([buffer.py:78]) | 필수 |
| `timeouts` (T,) | `assert not path['timeouts'].any()` ([buffer.py:79]) | **조건부 함정** |
구조 계약: **obs_dim 에피소드 간 동일** 필수([buffer.py:57-61]), **max_path_length ≥ 최장 에피소드**([buffer.py:66]).

### 2.2 Dreamer npz 실물 (RL_project/runs/stage2_oschersleben/train_eps) → 매핑
실측: **265 에피소드**, 키 = `lidar(T,1080)·state(T,5)·action(T,2)·reward(T)·is_terminal·is_last·
discount·log_reward_*·log_lap_time_s·log_completed·logprob`. **obs_dim(lidar+state)=1085 전부 일관**,
에피소드 길이 min45/median161/mean314/**max1881**, 크래시(is_terminal)=259, 완주(log_completed≥1)=**6**.
| Diffuser 요구 | Dreamer 제공 | 판정 |
|---|---|---|
| observations | `concat(lidar,state)`=1085D **또는** centerline 피처~10D | ✅(concat) / 피처는 pose 필요 |
| actions | `action` (이미 정규화 [-1,1]) | ✅ |
| rewards | `reward` 또는 `log_reward_progress`(+축소 collision) | ✅ |
| terminals | `is_terminal` | ✅ |
| timeouts | **없음** | ⚠️ §2.3-① |
→ **필수 4키는 기존 데이터로 충족**(rename+concat만). `discount·log_*·logprob`는 Diffuser 미사용.

### 2.3 구현 블로커 3종 (실측으로 확정)
1. **timeouts 함정**: `termination_penalty` 기본값 `0`인데 `0 is not None`=True → 크래시 에피소드
   (259/265)마다 [buffer.py:79]에서 없는 `timeouts` 키 접근 → **KeyError**. → config `termination_penalty=None`
   으로 끄거나 `timeouts = is_last & ~is_terminal` 합성.
2. **`np.int` 제거**: [buffer.py:12] `np.zeros(...,dtype=np.int)` → numpy 1.24.4(우리 venv)에서 **에러**.
   → `np.int64` 패치. (검증: `not hasattr(numpy,'int')` = True)
3. **pose 부재**: npz에 pose 없음 → **centerline 피처(D1 1순위) 계산 불가** = 유일한 진짜 데이터 갭.
   단 P1에서 warm-load 재학습 policy를 **pose 기록하며 새로 rollout**하면 자연 해소.

### 2.4 ★ import 폭탄 정밀 재매핑 ([[005-diffuser-glue-touchpoints]] 보강·정정)
실측으로 005보다 정확해진 그림:
- `diffuser/__init__.py: from . import environments` → **양성(benign)**. `registration.py`가
  `gym.register(entry_point=문자열)`만 하고 `try/except`로 감쌈 → mujoco env 파일은 `gym.make` 때만
  로드. `gym`만 있으면 통과(f110_gym으로 충족). **패치 불필요.**
- **진짜 코어 블로커**: `models/helpers.py:10 import diffuser.utils as utils` → `utils/__init__.py:6
  from .rendering import *` → `rendering.py` top-level이 **`imageio`(:4)·`mujoco_py`(:8)·
  `from .video import ...`(:13)·`from diffuser.datasets.d4rl import load_environment`(:15)**를 import.
  전부 (의도적) 미설치 → **코어(temporal/diffusion) import까지 동반 사망**. (실측: temporal import 시
  첫 실패가 `No module named 'imageio'`.)
- `tap`·`gym`은 venv가 이미 해결 → P0 코드 패치 대상에서 제외됨(005 대비 축소).

**그래서 P0 디커플 = 코드 패치(아래), venv 추가설치 아님:**
1. `utils/rendering.py`의 heavy import(imageio/mujoco_py/video/d4rl)를 **guard 또는 lazy화**
   (또는 `utils/__init__`의 `from .rendering import *`를 NullRenderer로 우회) — 코어 import 살리기.
2. `datasets/d4rl.py:23-25 import d4rl` try/except.
3. `buffer.py:12 np.int→np.int64`.
4. f1tenth 로더 신설 + `sequence.py:22-23 load_environment/seed` 스텁 + `config/f1tenth.py`.

### 2.5 메모리/실현성 재확인
torch 2.4.1+cu124 **cuda True**, RTX 4060 Ti 8GB. Diffuser ~4M params → 메모리 비-binding(40~50배 여유).
ReplayBuffer 사전할당(host RAM): 1085D·max_len1881·265ep → **2.16GB**(감당), 피처10D → **20MB**.

---

## 3. 계획은 어떻게 설정했나 (현재 SSOT 상태)

### 3.1 모델/원리
Diffuser(Janner ICML2022). 궤적 diffusion + **별도 value 함수**, denoising마다 ∇value로 고-return
유도(MPC, 첫 action 실행). 향상 = stitching+guidance가 데이터 내 좋은 구간 재조합. **한계: in-dist
생성기라 데이터에 없는 속도 외삽 불가** → 데이터에 빠른 재료 필수.

### 3.2 설계 결정 D1~D5 ([[003-diffuser-plan-final]]; D2만 §3.3로 갱신)
- **D1 관측**: centerline 피처~10D 1순위(pose 필요→재수집), lidar 다운샘플 백업.
- **D3 speed-cap**: `SpeedCappedAgent`가 raw 역정규화→raw speed에 κ 곱→재정규화. steer 무수정. env 무변경.
- **D4 reward**: progress(dense)+축소 collision(−2~−3), lap 보너스 제외, normed value[-1,1].
- **D5 글루**: 코어 무변경, §2.4 디커플 + f1tenth 로더 + plan_f1tenth 평가루프 + normalizer 영속화.

### 3.3 ★ 데이터소스 재정렬 — 최신 확정 (003 D2 폐기·대체, 2026-06-14 사용자)
- **주 = Dreamer.** behavior policy는 찾는 게 아니라 **만든다**: `runs/stage1_map_easy3/latest.pt`에서
  **world model만 warm-load(actor-critic 초기화) → Oschersleben 재학습** → checkpoint를 pose+raw
  기록하며 rollout → offline 데이터셋. critic의 "느린 완주 스냅샷 부재"는 **재학습으로 무력화**.
- **목표 lap-time대 유동**(80/70/60→40 등, 미리 fix 안 함, 결과 보고 결정).
- **expert(16.6s) 거론 금지** — 안 되면 그때만. **GapFollower/speed-cap은 보조 fallback**(완주 데이터
  희소 시 보충). 데이터 다양성은 일단 충분하다 보고 **실제로 돌려본 뒤 재고**(over-plan 금지).

### 3.4 Phase + 런타임 배치
| Phase | 작업 | 실행 venv |
|---|---|---|
| **P0** | §2.4 디커플 패치 + 코어 1-step 학습 smoke | **새 .venv** |
| **P1** | stage1 WM warm-load 재학습 → pose+raw rollout → 데이터셋 + baseline lap 실측 | **RL_project venv** (데이터 출처) |
| P2/P3 | 표현(피처/ lidar) 결정 + f1tenth 로더 + normalizer 저장 | 새 .venv |
| P4 | 궤적 diffusion + value 학습, 생성 궤적 품질 점검 | 새 .venv |
| P5/P6 | plan_f1tenth: GuidedPolicy+f110_gym, baseline 대비 lap time/2랩 완주 | **새 .venv** (f110_gym 포함) |

---

## 4. 다음 단계 + 참조

- **즉시 착수 가능 = P0** (교수님 답·데이터 무관): §2.4의 디커플 4건 → `from diffuser.models.temporal
  import TemporalUnet`이 새 venv에서 통과하는지 + 1-step loss 산출까지가 게이트.
- **f1tenth 과제 관련 판단 시 항상 참조**(사용자 지시): `_thinking/raws/AIE4003_RL_F1TENTH.pdf`(원본
  명세) + `_thinking/env_setting/`(특히 [[003-project-spec]]). state/action/reward/episode 종료조건·
  발표(2랩 lap time 영상)·평가 기준이 여기 있음.
- 분기마다 _thinking 문서 + commit + push (상시 규약).
