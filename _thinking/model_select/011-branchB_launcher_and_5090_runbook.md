# 011 — 첫 제출 점수 분해, 액션 주입 확정, 학습 런처 완성, 그리고 5090 실행 런북

> 작성일: 2026-07-30 | 장비: 집 **RTX 4060 Ti 8GB**(WSL2). **연구실 5090(32GB)은 내일부터 사용 가능.**
> 선행: [010](./010-branchB_s1s3_warmstart.md)(S1~S3 결과와 "99.89% 로드"의 함정), [009](./009-branchB_execution_spec.md)(실행 스펙)
> 새 코드: `scripts/branchB/{weight_loader,train_scope,train_1p1b,preflight,analyze_train_scope}.py`,
> `run_1p1b_train.sh`, `run_1p1b_generate.sh`, `configs/{train,eval}/*`
> 새 측정: `results/branchB/{train_scope_budget,preflight_*,warmstart_init_probe-2}.json`
>
> **이 문서의 목적**: ① 첫 리더보드 점수가 알려준 것을 기록하고 ② 내일 5090 앞에 앉아
> **이 문서의 §7만 그대로 따라 하면 학습이 걸리도록** 만드는 것.

---

## 0. 한 장 요약

```
① [첫 제출 = 캘리브레이션 성공] static CSV 로 Public **0.30324**.
     서버만 알던 값을 역산했다 → eval 의 DINO+Video = **0.439** (로컬 홀드아웃 0.214의 **2.05배 나쁨**)
     60% 성분이 점수의 **43.4%** 를 차지한다(로컬 감각으로는 11.4%였다 = 3.8배).
     008 의 추정 0.236 은 0.067 낙관적이었고, 오차는 전부 "로컬 DINO/Video 가 eval 에도 갈 것"이라는 가정에서 왔다.
     ⇒ **개선 가능 폭 0.13 이 전부 60% 에 있다.** 정확한 영상의 상금이 로컬 감각의 2배다.

② [확정] 액션 주입을 **가산(add_act_time_emb=True) + action_embed 0초기화**로 바꿨다.
     재검증: DC UNet 1516키 **100% 흡수**, 학습 시작점이 DC 와 **cos 1.00000 / rel-L2 0.00000**(랜덤 액션에서도).
     concat(009 원안)은 같은 조건에서 cos 0.67 — 사전학습 실력을 시작부터 버린다.

③ [새 제약 발견] 학습 범위별 VRAM 산술:
     full 1440.5M → **31.2GB**  ⇒ **5090(32GB)에서는 전체 파인튜닝이 불가능**
     action_temporal 551.1M(38.3%) → **21.3GB**  ⇒ 5090 파일럿은 이 범위로 간다
     그리고 이 선택은 우연이 아니라 진단과 맞는다 — 우리 병목은 시간적 드리프트이고,
     시간축 레이어(545M)+액션 분기가 정확히 그 부분이다.

④ [런처 완성 + 8GB 실검증] 경로치환·warm-start 필터·0초기화·범위동결·EMA 재초기화·
     2단 체크포인트·CSV 로그를 **11M 폭으로 20스텝 실제 학습**시켜 전부 확인했다.
     내일은 config 를 바꿀 필요 없이 scope 만 지정해 그대로 돌리면 된다.

⑤ [보존 감사] 삭제된 것 없음. M3 체크포인트 7개(각 87.6MB)·run_logs 5개·results 27개 파일 전부 존재.
```

---

## 1. 첫 제출이 알려준 것 — static = 0.30324 의 분해

### 1.1 어떻게 분해했나

`Score = 0.3·DINO + 0.3·Video + 0.4·Action` 이고, **Action 은 제출 CSV 에 표본별 스칼라로 실려 있다**
(216개 평균 0.42874). 따라서 나머지 60% 를 역산할 수 있다. 다만 Public 은 eval 의 30%(65표본)이고
어느 표본인지 모르므로, **무작위 65개 부분집합 2만 회**로 불확실 구간을 함께 냈다.

| 성분 | 값 | 점수 기여 |
|---|---:|---:|
| Action (40%) — 실측 | 0.42874 (public 95% 구간 0.385~0.473) | 0.1715 |
| **DINO+Video (60%) — 역산** | **0.4391** (구간 0.381~0.497) | **0.1317** |
| 합 | | **0.30324** |

### 1.2 무엇이 뒤집혔나

| | 로컬 홀드아웃 | eval(실측) | 배율 |
|---|---:|---:|---:|
| static 의 Action | 1.2402 | 0.4287 | **0.35배(쉬움)** |
| static 의 DINO+Video | 0.2142 | **0.4391** | **2.05배(어려움)** |
| 60% 성분의 점수 비중 | 11.4% | **43.4%** | **3.8배** |

- eval 영상은 로컬 홀드아웃보다 **더 많이 움직인다** → "정지 영상"의 벌점이 그만큼 크다.
- 008 §4.3 은 "eval 에서 60% 의 상대 가치가 2.4배"라고 추정했는데, **실측은 3.8배**로 더 컸다.
- 완벽 예측이면 DINO+Video=0 이므로 Score ≈ 0.4×Action ≈ **0.17**.
  ⇒ **static 0.303 → 이론 하한 0.17, 개선 가능 폭 0.13 이 전부 60% 에 있다.**

### 1.3 그래서 전략은 어떻게 바뀌나

바뀌지 않는다 — **강화된다.** 009 §5 가 정한 "판정은 DINO 로" 는 그대로이고,
그 지표를 개선했을 때 리더보드에서 받는 보상이 로컬 감각의 **2배**라는 근거가 생겼다.
그리고 branch B 의 전제(=DC 사전학습의 영상 prior 를 물려받는다)가 정확히 그 60% 를 겨냥한다.

> 주의: 이 분해는 **Public 65표본의 Action 이 216표본 평균과 같다**는 가정을 쓴다.
> 그래서 구간(0.381~0.497)을 함께 적었다. 정확한 값은 서버만 안다.

---

## 2. 용어 (이 문서에서 새로 쓰는 말)

| 용어 | 쉬운 설명 |
|---|---|
| **scope(학습 범위)** | 1.44B 파라미터 중 **어디를 학습하고 어디를 얼릴지**. 얼린 부분은 gradient·optimizer 메모리를 쓰지 않아 VRAM 이 크게 준다 |
| **동결(freeze)** | `requires_grad=False`. 계산에는 참여하지만 값이 바뀌지 않는다 |
| **시간축 레이어** | 프레임 사이의 관계를 처리하는 층(`TemporalTransformer`/`TemporalConvBlock`). **키 이름에 'temporal' 문자열이 없어서** 모듈 타입으로만 찾을 수 있다(005 오진의 원인) |
| **롤링 체크포인트** | 끊김 대비용. 새로 저장하면 이전 것을 지운다 → 용량이 늘지 않는다 |
| **영구 아카이브** | 지우지 않고 쌓는 체크포인트. 프로젝트 규칙(삭제 금지)을 지키는 쪽 |
| **preflight** | 학습을 걸기 전 5분 점검. GPU 커널이 실제로 도는지까지 확인한다(006 이 여기서 막혔다) |
| **BUILD_ONLY** | 모델 빌드 + 가중치 얹기까지만 하고 학습 없이 끝내는 모드. 대형 실행 전 안전 점검 |
| **액션 민감도** | 같은 사진에 **다른 액션**을 넣었을 때 결과 영상이 달라지는 정도. 가산 방식이 액션을 무시하지 않는지 확인하는 안전장치 |

---

## 3. 확정 — 액션 주입을 "가산 + 0초기화"로 바꿨다

### 3.1 무엇을 바꿨나 (config 한 줄)

```yaml
unet_config.params:
  add_act_time_emb: True     # 액션을 시간 임베딩에 concat 하지 않고 **더한다**
```
그리고 학습 런처가 `action_embed` 의 마지막 층을 **0으로 초기화**한다
(baseline 이 `fps_embedding` 에 쓰는 기법 그대로 — `openaimodel3d.py` 451~452행).

### 3.2 재검증 결과

| 항목 | concat (009 원안) | **가산 + 0초기화 (채택)** |
|---|---:|---:|
| DC UNet 흡수 | 1514키 / 1437.21M (99.886%) | **1516키 / 1438.86M (100.000%)** |
| 스크래치로 남는 키 | 7키(액션 5 + `time_embed.2` 2) | **5키(액션 분기만)** |
| shape 불일치(로드 전 제거 필요) | 2키 | **0키** |
| 시작 시점 함수 (DC 원본 대비, 액션 null) | cos 0.685 / rel-L2 0.917 | **cos 1.00000 / rel-L2 0.00000** |
| 시작 시점 함수 (랜덤 액션) | cos 0.693 / rel-L2 0.911 | **cos 1.00000 / rel-L2 0.00000** |
| UNet 총 파라미터 | 1438.86M | 1440.50M |

> concat 의 cos 는 실행마다 **0.666~0.693** 으로 흔들린다(독립 3회 측정; `time_embed.2` 가 매번 다르게 랜덤 초기화되므로).
> 표의 값은 보존된 리포트 `warmstart_init_probe-2.json` 기준이다.
> 결론은 흔들리지 않는다: **사전학습 실력이 시작부터 사라진다.**

### 3.3 리스크와 안전장치

가산이므로 **액션 신호가 시간 신호에 묻힐 위험**이 있다(모델이 액션을 무시하고 그럴듯한 영상만 만들 가능성).
0초기화는 그 위험을 더한다 — 시작이 정확히 "액션 영향 0"이기 때문이다.

**안전장치 = 액션 민감도 검사.** 같은 시작 이미지에 서로 다른 액션 2개를 넣어 생성하고 결과가
얼마나 달라지는지 잰다. 변화가 거의 없으면 액션을 무시하는 것이다.
→ 대처: (a) 액션 임베딩 스케일 상향, (b) 0초기화 해제(`BRANCHB_ZERO_INIT=0`), (c) concat 으로 폴백.
**되돌릴 수 있는 결정**이므로 가산으로 출발하는 것이 합리적이다.

---

## 4. 새 제약 — 5090(32GB)에서는 전체 파인튜닝이 안 된다

`results/branchB/train_scope_budget.json` (활성화 메모리 제외한 산술)

| 그룹 | 키 | 파라미터 | 비중 |
|---|---:|---:|---:|
| 액션 분기 | 5 | 1.65M | 0.11% |
| 조건 임베딩(time/fps) | 8 | 4.10M | 0.28% |
| **시간축** | 794 | **545.32M** | **37.86%** |
| 공간/기타 | 714 | 889.43M | 61.74% |

| scope | 학습 대상 | params | grads | AdamW | 동결모듈 | EMA | **합계** | 5090 32GB |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| full | 1440.5M (100%) | 5.37 | 5.37 | 10.73 | 4.36 | 5.37 | **31.20GB** | ❌ 불가 |
| **action_temporal** | 551.1M (38.3%) | 5.37 | 2.05 | 4.11 | 4.36 | 5.37 | **21.26GB** | ✅ |
| action_only | 5.8M (0.4%) | 5.37 | 0.02 | 0.04 | 4.36 | 5.37 | **15.16GB** | ✅ |

- **full 은 31.2GB 로 이미 32GB 에 근접**하고 활성화가 더 얹히므로 5090 에서는 현실적으로 불가능하다.
  (96GB 에서는 여유롭다 → 본 학습은 6000 에서 full 로.)
- 5090 파일럿은 **action_temporal**. 공간 레이어(DC 가 이미지 품질용으로 학습해 둔 부분)를 얼리는 것은
  "사전학습을 보존한다"는 이번 결정과도 방향이 같다.
- 활성화 메모리를 더 줄여야 하면: `use_ema: False`(−5.37GB), `accumulate_grad_batches` 상향(배치 1 유지).

---

## 5. 만들어진 도구 지도

| 파일 | 역할 |
|---|---|
| `configs/train/inha_action_diffusion_1p1b.yaml` | 학습 config (7키 + `add_act_time_emb`). 경로는 `__REPO__` 센티넬 |
| `configs/eval/gen_1p1b.yaml` | 생성 config. **학습 config 의 런타임 사본을 참조** → 학습·추론 아키텍처 불일치가 원천 차단 |
| `cfg_paths.py` | `__REPO__` → 저장소 루트 치환(+잔존 시 예외) |
| `weight_loader.py` | shape 불일치 사전제거 + `action_embed` 0초기화. **학습·검증이 같은 로직을 공유** |
| `train_scope.py` | 시간축 모듈을 **타입**으로 식별해 scope 별 동결 |
| `train_1p1b.py` | warm-start → scope 동결 → EMA 재초기화 → 학습. `BRANCHB_BUILD_ONLY=1` 점검 모드 |
| `run_1p1b_train.sh` | 경로치환·PYTHONPATH·랭크 env·override 를 감싼 실행 스크립트 |
| `run_1p1b_generate.sh` | 학습한 ckpt 로 영상 생성(baseline 생성 스크립트를 수정 없이 호출) |
| `preflight.py` | 새 머신 5분 점검 + scope 추천 |
| `analyze_train_scope.py` | 범위별 파라미터·VRAM 산술 |
| `verify_1p1b_load.py` / `probe_warmstart_init.py` | S3 로드 검증 / 함수 보존 측정 |
| `probe_action_sensitivity.py` | 액션 민감도 검사(§3.3 안전장치). 0초기화 직후 기준값 = **0.000000** 실측 |
| `make_smoke_holdout.py` | 스모크용 4표본 홀드아웃 서브셋 생성(새 머신용) |

환경변수: `BRANCHB_TRAIN_SCOPE`(full·action_temporal·action_only), `BRANCHB_BUILD_ONLY`,
`BRANCHB_WARMSTART_CKPT`, `BRANCHB_ZERO_INIT`, `BRANCHB_CONFIG`.

---

## 6. 8GB 에서 무엇까지 검증했나 (그리고 무엇이 아직 아닌가)

11M 폭으로 되돌린 **같은 config·같은 런처**를 8GB GPU 에서 20스텝(=40 iter) 실제로 학습시켰다.

| 검증 항목 | 결과 |
|---|---|
| `__REPO__` 경로 치환 + 잔존 검사 | ✅ |
| warm-start 필터 | ✅ 941키 shape 불일치를 제거하고도 예외 없이 진행 |
| 0초기화 | ✅ `weight_absmax = 0.0` |
| scope 동결(action_temporal) + gradient checkpointing | ✅ loss 1.01 → 0.65 하강(동결 상태에서 학습이 실제로 된다) |
| EMA 재초기화 | ✅ |
| **2단 체크포인트** | ✅ 롤링은 최신 1개+`last.ckpt` 유지(이전 삭제), 영구 아카이브는 step=10·20 **둘 다 보존** |
| CSV 로그 | ✅ `artifacts/branchB/train_out/logs/.../metrics.csv` |
| 정상 종료 | ✅ `max_steps=20 reached` |

**아직 검증되지 않은 것(정직하게)**
- 1.44B 를 GPU 에 올린 **실제 VRAM·s/it** — 8GB 로는 물리적으로 불가. 내일 첫 측정.
- 1.1B 생성 경로(`run_1p1b_generate.sh`)의 실행 — ckpt 가 없어 아직 못 돌렸다.
- 데이터로더가 128 데이터셋 전체를 문제없이 도는지 — 20스텝만 돌았으므로 초반만 확인됨.

---

## 7. ★ 내일 5090 에서 할 일 (그대로 복붙)

> 전제: 저장소를 5090 머신에 최신화(`git pull` 은 **사용자가 직접** — 우리 규칙),
> `open/` 데이터·`backbone.ckpt` 배치, conda `wm` 환경(cu128).
> 환경 복원 절차는 [env_file/001](../env_file/001-environment_setup.md).

### 단계 0 — 게이트 + 홀드아웃 재생성 (10분)

`artifacts/` 는 `.gitignore` 대상이라 **새 머신에는 홀드아웃이 없다.** 먼저 만든다(집과 동일 seed=0).

```bash
conda run -n wm python scripts/branchB/preflight.py

conda run -n wm python scripts/build_holdout.py --train-root open/data/train \
    --out artifacts/holdout --n 96 --seed 0 --per-dataset-cap 2
conda run -n wm python scripts/branchB/make_smoke_holdout.py        # → artifacts/holdout_smoke4

# 채점 재현 확인(집 수치와 일치해야 한다): static TOTAL 0.560 / DINO 0.123 / GT 0.000
conda run -n wm python scripts/run_m0.py --holdout artifacts/holdout --out artifacts/m0
```
- **기대**: `판정: PASS` + `추천 scope=action_temporal`
- **실패 시**: `GPU 커널 실행 FAIL` 이면 006 과 같은 드라이버/휠 불일치다.
  → cu128 휠 재설치(`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`).
  이 단계에서 걸러야 학습에 시간을 버리지 않는다.

### 단계 1 — 빌드·warm-start 점검 (3분, 학습 없음)

```bash
BRANCHB_TRAIN_SCOPE=action_temporal BRANCHB_BUILD_ONLY=1 \
  bash scripts/branchB/run_1p1b_train.sh
```
- **기대 출력**
  - `UNet 1440.50M / add_act_time_emb=True / fs_condition=True`
  - `적재 1516키 / 1438.86M`, `shape 불일치로 버린 키: 0`, `unet_missing 5키`, `unet_unexpected 0키`
  - `zero_init: {'applied': True, ..., 'weight_absmax': 0.0}`
  - `scope=action_temporal: 학습 551.1M (38.26%)`
  - `BUILD_ONLY: 모델 GPU 적재 후 VRAM ...GB` ← **이 숫자를 기록**(산술 21.26GB 와 비교)
- 여기서 OOM 이면 `BRANCHB_TRAIN_SCOPE=action_only` 로 낮춘다.

### 단계 2 — S4 스모크 학습 (20스텝)

```bash
BRANCHB_TRAIN_SCOPE=action_temporal \
  bash scripts/branchB/run_1p1b_train.sh 20 "00:00:30:00" \
  2>&1 | tee run_logs/branchB_1p1b_smoke_$(date +%m%d_%H%M).log
```
- **기대**: loss 가 찍히고 체크포인트가 생기고 `max_steps=20 reached` 로 끝난다.
- **기록할 것**: `s/it`(→ 4일 총 스텝 역산), peak VRAM, loss 초기값.
- 체크포인트 크기 확인: `du -h artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/*`
  (예상 ≈10.7GiB/개 — 여기서 영구 아카이브 주기를 최종 결정한다)

### 단계 3 — S4 스모크 생성 (4샘플) + 채점

```bash
CKPT=artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/last.ckpt
bash scripts/branchB/run_1p1b_generate.sh "$CKPT" \
     artifacts/holdout_smoke4 artifacts/branchB/preds_smoke4 50 1.0

conda run -n wm python scripts/run_m0.py \
  --holdout artifacts/holdout_smoke4 --out artifacts/branchB/m0_smoke4 \
  --pred-dir artifacts/branchB/preds_smoke4 --pred-name b1p1b_smoke --skip-gt
```
- `artifacts/holdout_smoke4` 는 이미 만들어 두었다(홀드아웃 앞 4표본).
- **목적은 점수가 아니라 파이프라인 무결성**이다(n=4 로는 아무것도 판정하지 않는다).
- 확인: mp4 4개가 생기는가, 16프레임인가, 채점이 도는가.

### 단계 4 — S5 파일럿 학습 (시간박스)

```bash
BRANCHB_TRAIN_SCOPE=action_temporal \
  bash scripts/branchB/run_1p1b_train.sh "" "00:06:00:00" \
  2>&1 | tee run_logs/branchB_1p1b_pilot_$(date +%m%d_%H%M).log
```
- 6시간 시간박스. 500스텝마다 롤링, 5000스텝마다 영구 아카이브.
- 단계 2 에서 잰 s/it 로 예상 스텝 수를 먼저 계산해 둘 것.

### 단계 5 — S5 판정 (홀드아웃 96 생성 + 채점)

```bash
bash scripts/branchB/run_1p1b_generate.sh "$CKPT" \
     artifacts/holdout artifacts/branchB/preds_pilot 50 1.0
conda run -n wm python scripts/run_m0.py \
  --holdout artifacts/holdout --out artifacts/branchB/m0_pilot \
  --pred-dir artifacts/branchB/preds_pilot --pred-name b1p1b_pilot
```

---

## 8. 판정 기준 (S5 에서 무엇을 보고 본 학습으로 갈지)

```
1차 지표 : DINO   ← 이것만이 유효하다 (009 §5, 008 §8.5)
기준선   : static DINO 0.123 / baseline 11M 0.550 / M3 재학습 11M 0.481
```

| 파일럿 결과 | 해석 | 다음 |
|---|---|---|
| DINO < 0.123 | **처음으로 생성이 정지를 이겼다** | 즉시 본 학습(6000, scope=full) |
| 0.123 ≤ DINO < 0.30 | 방향은 맞다(11M 0.48 보다 크게 좋음) | 학습을 더 돌려 추세 확인 후 본 학습 |
| 0.30 ≤ DINO < 0.48 | 1.1B 사전학습 이득은 있으나 부족 | scope·lr·fs A/B, anti-drift 설계 검토 |
| DINO ≥ 0.48 | 11M 재학습과 다를 바 없다 | 원인 규명(액션 무시? 데이터?) 후 branch A 재검토 |

**보조 진단(판정에는 쓰지 않는다)**: 액션 민감도(다른 액션 → 다른 영상인가), motion, 첫 프레임 이탈량.

---

## 9. 보존 상태 감사 (CLAUDE.md 규칙 점검)

| 대상 | 상태 |
|---|---|
| M3 Part B 학습 체크포인트 | ✅ 7개(step 500·1000·1500·2000·2500·3000 + last), 각 87.6MB, `artifacts/m3/train_out/.../checkpoints/` |
| 학습·평가 로그 | ✅ `run_logs/` 5개(partB 학습로그 2.19MB 포함) |
| 측정 결과 | ✅ `results/` 27개 파일 git 추적. **git 이력상 삭제된 파일 0건** |
| 생성 영상·홀드아웃·제출물 | ✅ `artifacts/` 624MB(m3) + 96MB(submission) 등 전부 존재 |
| 디스크 | 835GiB 여유 (1.1B 체크포인트 ≈10.7GiB → 77개분) |

**주의 1건**: `artifacts/`·`run_logs/` 는 `.gitignore` 대상이라 **이 집 컴퓨터에만 존재**한다.
git 에는 요약 결과(`results/`)와 코드·문서만 있다. 원본 로그·체크포인트가 필요하면 별도 백업이 필요하다
(현 시점 결론 재현에는 `results/` 로 충분하므로 조치는 하지 않았다).

---

## 10. 남은 리스크 · 미검증

| # | 항목 | 언제 해소되나 |
|---|---|---|
| 1 | 1.44B 실제 VRAM·s/it (산술 21.26GB 가 맞나) | 내일 단계 1~2 |
| 2 | 5090 드라이버(006 의 실패 원인이 남아 있나) | 내일 단계 0 |
| 3 | 액션 민감도 — 가산+0초기화가 액션을 무시하지 않는가 | 단계 3 이후, 별도 진단 스크립트 필요 |
| 4 | 128 데이터셋 전체 순회 안정성(16프레임 미만·10fps·고해상 예외) | 파일럿 학습 중 자연히 검증됨 |
| 5 | 영구 아카이브 주기 최종값 | 단계 2 의 s/it·ckpt 크기 실측 후 |
| 6 | 8GB fp16 추론 가능성(집에서 생성까지 하려면) | 5090 가용으로 우선순위 하락 — 보류 |
| 7 | lr 1e-4 가 1.44B warm-start 에 적절한가 | 단계 2 의 loss 거동으로 1차 판단 |

---

## 11. 한 문단 결론

첫 제출은 점수 이상의 것을 줬다 — **eval 에서 60% 성분(DINO+Video)이 로컬보다 2배 나쁘고,
점수의 43%를 차지하며, 개선 가능한 0.13 이 전부 거기 있다**는 지도를 얻었다. 그 지도는
branch B 의 전제와 정확히 겹친다. 그래서 이번 세션은 "사전학습 실력을 최대한 물려받는 것"에
집중했고, 액션 주입을 가산+0초기화로 바꿔 **DC 를 100% 물려받고 시작점 함수까지 완전히 동일**하게 만들었다.
동시에 5090 의 현실도 산술로 확인했다 — 전체 파인튜닝은 32GB 에 들어가지 않으므로,
파일럿은 시간축+액션(38.3%, 21.3GB)으로 간다. 이 선택은 타협이 아니라 진단과 일치한다:
우리가 고쳐야 하는 것은 **시간적 드리프트**이기 때문이다. 런처는 8GB 에서 20스텝 실제 학습으로
배관을 전부 확인했으니, 내일은 §7 을 순서대로 따라가면 된다. 판정은 여전히 하나다 — **DINO 가 0.123 아래로 내려가는가.**
