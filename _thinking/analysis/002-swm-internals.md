# 002 — swm(stable-worldmodel) 내부 분석 (Phase 1 잔여 완료)

> 2026-06-13. 계획 v2 Phase 1의 마지막 항목. 질문 3개에 답한다:
> ① HDF5 윈도잉/경계 처리 ② planning 정규화·역정규화 경로 ③ swm.World 재사용 경계.
> 근거 = `.venv/lib/python3.10/site-packages/stable_worldmodel/` 소스 직접 정독 + 공식 h5 실측.
> append-only.

---

## 0. 요약 (설계에 미치는 영향 순)

1. **공식 HDF5Writer 존재** — 우리 데이터셋 변환기는 h5 직접 작성 불필요. 단 **압축 미지원**이
   유일 결함 → 우리 규모에선 blosc 직접 처리 필요
2. **NaN 규약 실측 확정**: 에피소드 마지막 row의 action만 NaN (에피소드당 정확히 1개)
3. **윈도우는 에피소드 경계를 절대 넘지 않음** — 경계 패딩 걱정 불필요
4. **정규화 왕복은 WorldModelPolicy가 자동 처리** — 단 CEM 샘플은 unbounded라 env 쪽 clip 필수
5. **World.evaluate는 재사용 불가(C-1 확정)지만, WorldModelPolicy+CEMSolver는 거의 통째로
   재사용 가능** — f1tenth 신규 구현 범위가 예상보다 작음

---

## 1. HDF5 윈도잉 / 경계 처리 (`data/dataset.py`, `data/formats/hdf5.py`)

### 윈도우 인덱싱 (dataset.py:50-55)
```python
clip_indices = [(ep, start) for ep, length in enumerate(lengths)
                if length >= span                  # span = num_steps×frameskip = 20
                for start in range(length - span + 1)]
```
- **에피소드 내부에서만 윈도우 생성** — 경계를 넘는 윈도우 없음, 길이<20 에피소드는 통째로 제외
- 검산: tworoom 920,809 rows − 10,000ep×19 = 730,809 = len(dataset) ✓

### 슬라이스 로드 (hdf5.py:111-132, dataset.py:67-72)
- non-action 키: `data[::frameskip]` → 4 프레임 (rows 0,5,10,15)
- **action: 서브샘플 없이 20개 전부** → `reshape(num_steps, -1)` = (4, 10) — 프레임 t의 블록 =
  rows [5t, 5t+5)의 action 5개 평탄화
- pixels: 파일은 HWC, 로더가 `permute(0,3,1,2)`로 CHW 변환 (hdf5.py:129-130)
- keys_to_cache 컬럼은 RAM 전체 캐시 (tworoom: action/proprio)

### NaN 규약 (실측, 공식 tworoom h5)
- **에피소드 마지막 row의 action = [NaN, NaN]** — 전체 920,809 rows 중 NaN row 정확히
  10,000 = 에피소드 수. 위치도 ep_offset+ep_len−1과 일치
- 소비처: ① `get_column_normalizer`(utils.py:29) NaN row 제외 후 z-score fit
  ② train.py:25 `nan_to_num(action, 0)` ③ eval.py:77 scaler fit 시 제외
- ⇒ **우리 생성기 규약**: T step 에피소드 = obs T개 + action T개, action[T−1]=NaN

### 공식 HDF5Writer (hdf5.py:169-288)
- `with HDF5(...).open_writer(path, mode='append') as w: w.write_episode({col: per-step array})`
- ep_len/ep_offset 자동 관리, 스키마는 첫 에피소드에서 추론·잠금, append 시 스키마 검증
- **결함: 압축 없음** + chunks=(1,*shape). 공식 배포본은 blosc(filter 32001)+chunks 100.
  1M step pixels 비압축 ≈ 150GB → 우리 변환기는 h5py로 직접 쓰되 `hdf5plugin.Blosc` 적용
  (Writer 코드 ~80줄이라 본뜨기 쉬움) 또는 Writer로 쓴 뒤 repack. **Phase 3-1 설계 항목**

---

## 2. Planning 정규화·역정규화 경로 (`policy.py`, `solver/cem.py`)

### 경로 전체 (z-score 왕복은 자동이다)
1. eval.py:71-82 — keys_to_cache 컬럼별 sklearn `StandardScaler` fit(NaN row 제외) → `process` dict
2. `WorldModelPolicy._prepare_info` (policy.py:86-148) — obs dict의 각 키에
   `process[k].transform`(z-score) 적용, `pixels`/`goal*` 키엔 `transform[k]`(ImageNet 정규화+resize)
3. CEMSolver.solve (cem.py:189-263) — **z-score 공간에서** N(mean,var) 샘플 → model.get_cost
   (jepa.rollout+criterion) → top-30로 분포 갱신 ×30 iter → mean 반환
4. `policy.get_action` 마지막 (policy.py:436-437) — **`process['action'].inverse_transform`** →
   env에 raw action 반환

⇒ WorldModelPolicy를 그대로 쓰면 역정규화 누락(silent failure 1순위로 꼽았던 것)은 발생하지
않음. **신규 루프에서 process dict 구성만 정확하면 됨** (우리 학습 데이터로 scaler fit).

### 주의점
- **CEM 샘플은 unbounded** (cem.py:191-207, 클램핑 없음) — 역정규화 후 물리 범위 초과 가능.
  f1tenth wrapper가 step에서 steer/speed clip하므로 안전하지만, world model이 본 적 없는
  극단 action이 cost 평가에 들어갈 수 있음을 평가 설계에서 인지
- PlanConfig(eval/pusht.yaml): horizon 5, receding 5, action_block 5(=frameskip),
  history_len 1 — **planning 시 context는 1프레임** (학습은 history 3이지만 rollout이
  `emb[:, -3:]` 슬라이딩이라 호환)
- receding 버퍼: env step 단위 25개(5×5) 소진 후 재계획. warm_start로 잔여 plan 재사용
- `prepare_init_action`: model이 get_action 가지면 actor warm-start (JEPA엔 없음 → zero-init)

---

## 3. swm.World 재사용 경계 (`world/world.py`)

### 재사용 불가 (C-1 확정)
- `World(env_name=...)`은 **swm에 gym 등록된 env 전용** (`swm/PushT-v1` 등). f1tenth 미등록
- `_evaluate_from_dataset`(world.py:492-564): 데이터셋에서 (시작 state, goal_offset 뒤 프레임)
  추출 → `_apply_callables`로 env `_set_state`/`_set_goal_state` 호출 → eval_budget 내
  terminated 여부로 성공 판정. **env가 set_state·goal 판정을 지원해야 하는 구조** —
  f1tenth gym은 둘 다 없음

### 재사용 가능 (예상보다 큼)
- **WorldModelPolicy + CEMSolver + process/transform 스택 전부**. policy가 env에 요구하는 건
  `set_env()`에서 action_space/num_envs 뿐 → f1tenth용 평가 루프는:
  ① env reset → infos dict(pixels 렌더, proprio, goal=subgoal 프레임) 구성
  ② `policy.get_action(infos)` 호출 (정규화·planning·역정규화 자동)
  ③ env.step → subgoal 갱신(③-a receding goal-chasing) → 성공/완주 판정
  만 신규 작성하면 됨. eval.py의 process/transform 구성 코드(71-97)는 거의 복붙 가능
- on_step마다 goal 키 재주입 필요 (infos가 step마다 덮임 — world.py:541-542 패턴 참조)
- 성공 판정은 우리가 정의 (위치 거리 기반 권장 — pose를 별도 기록하므로 가능)

---

## 4. Phase 2/3에 넘기는 결정·작업 항목

- [결정②에 추가] h5 압축: blosc 직접 적용 vs Writer+repack (용량 150GB→~14GB 영향)
- [결정④ 입력] planning context 1프레임 확인 → history/frameskip 설계 시 반영
- [Phase 3-1] 생성기 규약: flat row-major + ep_len/ep_offset + 마지막 action NaN +
  pixels HWC uint8 + pose/lap 진단 컬럼(잉여 컬럼은 로더가 무시하므로 자유)
- [Phase 3-4] 평가 루프 신규 범위 확정: env 루프 + infos 구성 + subgoal 스케줄러 + 판정.
  정규화/CEM/MPC 버퍼는 swm 재사용
- [Phase 3-4] CEM unbounded → env clip 의존 명시, 필요시 solver에 클램핑 콜백 추가 검토
