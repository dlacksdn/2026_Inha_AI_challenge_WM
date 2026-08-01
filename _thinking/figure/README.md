# figure — 방향을 결정한 근거들 (다른 컴퓨터에서도 볼 수 있게)

> 만든 날: 2026-08-02 | 목적: **작업 이어받기**

## 왜 이 폴더가 있나

이 저장소는 `artifacts/`(그림·영상·체크포인트)와 `run_logs/`(로그)를 git 에 올리지 않는다.
용량 때문에 옳은 선택이다. 그런데 그 결과 문제가 하나 생긴다.

> **노선을 결정한 근거가 한 컴퓨터에만 있으면, 다른 기계에서 작업을 이어받을 수 없다.**

그래서 여기에 **"그 결정을 다시 설명할 수 있는 최소한"** 만 골라 복사해 둔다.
`.gitignore` 에 이 폴더만 예외 규칙을 넣었다.

**넣는 것 / 안 넣는 것**

| 넣는다 | 안 넣는다 |
|---|---|
| 결정을 만든 **그림**(원본은 `artifacts/`) | 체크포인트(7.9GB), 생성 영상 전체 |
| 판정 출력의 **사람이 읽는 표**(원본은 `run_logs/`) | 원본 로그 전문(수십 MB) |
| 각 근거가 **어느 결정을 만들었는지** | 재현 가능한 산출물(홀드아웃 등 — 스크립트로 다시 만든다) |

**수치 원자료는 이미 git 에 있다.** `results/**/*.json` 은 `.gitignore` 예외로 추적 중이고,
표본별 rows 까지 들어 있다. 여기 있는 표는 그 JSON 을 사람이 읽게 정리한 것이다.

---

## 결정 → 근거 지도

| # | 결정 | 언제 | 근거 문서 | 원자료 |
|---|---|---|---|---|
| 1 | **branch B(1.1B 잠재확산) 학습 동결** | 8/1 16:11 | [001](./001-branchB_freeze.md) | — |
| 2 | **잔차 보정 노선 채택** | 8/1 16:28 | [002](./002-residual_headroom.md) | `results/branchB/residual_headroom.json` |
| 3 | ⭐ **덧셈 잔차 확정, 순수 워핑 배제** | 8/1 17:44~21:06 | [003](./003-warp_vs_add.md) | `results/branchC/warp_vs_add.json`, `warp_round2_raft.json`, `warp_round2_farneback.json`, `round3.json` |
| 4 | **흐림 예산 σ ≈ 1.2~1.9픽셀** | 8/1 18:00 | [004](./004-blur_budget.md) | `results/branchC/blur_and_averaging.json` |
| 5 | ⭐ **평균내기 노선 폐쇄** | 8/1 20:40 | [005](./005-ksample_closed.md) | `results/branchC/ksample_average.json`, `ksample_precondition.json` |
| 6 | ⭐ **해상도는 목표가 아니라 방향 정확도의 증폭기** | 8/1 21:06 | [006](./006-direction_over_resolution.md) | `results/branchC/round3.json` |

⭐ = 이번(8/1 저녁) 세션에서 나온 것. 나머지는 그날 낮의 결정이다.

---

## 그림·영상 목록 (`img/`)

| 파일 | 무엇 | 어느 결정에 쓰였나 |
|---|---|---|
| `20260801_1630_branches_curve.png` | 같은 지점에서 갈라진 학습 5갈래 — **전부 출발점보다 나빠졌다** | 1 |
| `20260801_1629_residual_headroom.png` | 잔차 보정의 천장(−0.071)과 바닥(+0.051) | 2 |
| `20260801_1747_warp_vs_add_frame15.png` | 표본 3개 × 변형 4개, 프레임 15 | 3 |
| `20260801_2340_compare_A_arm_exits.mp4` | ⭐ **팔이 화면 밖으로 나가는 장면.** 워핑이 실패하는 이유가 눈에 보인다 | 3 |
| `20260801_2340_compare_B_gripper_descends.mp4` | 대조군 — **집게가 화면 안에서 내려오는 장면. 여기선 워핑이 잘 된다** | 3 |

두 영상은 2×2 배치, 3배 느리게, 3회 반복이다.

```
  ① 정답                      ② 정지영상 (넘어야 할 선)
  ③ 워핑                      ④ 덧셈
```

**A 와 B 를 반드시 같이 보라.** A 만 보면 "워핑은 나쁘다"로 오해하기 쉬운데,
B 를 보면 **워핑이 원리적으로 나쁜 게 아니라 우리 데이터의 주된 변화가
"가려진 것이 드러나는" 종류라서** 안 맞는다는 것이 보인다.

---

## 다른 컴퓨터에서 이어받을 때 — 이 저장소에 **없는** 것

| 없는 것 | 크기 | 다시 만들 수 있나 |
|---|---|---|
| `open/data/`, `open/baseline/`, `open/submission_kit/` | 수십 GB | ❌ 대회 배포물. **별도로 받아야 한다** |
| 체크포인트 `artifacts/branchB/ckpt_snapshots/*.ckpt` | 각 7.9GB | ❌ 학습 3일치. branch B 는 동결됐으니 **없어도 C 작업은 가능** |
| 홀드아웃 `artifacts/holdout/`, `artifacts/holdout_val96/` | 수백 MB | ✅ **결정론적으로 재생성된다** (아래) |
| 정지영상 대조군 | 수십 MB | ✅ 재생성 |
| 생성 영상(`ksample/`, `preds_*`) | 수 GB | ❌ 체크포인트가 있어야 한다. 다만 **수치는 results/ 에 남아 있다** |
| `run_logs/*.log` 전문 | 수십 MB | ❌ 다만 **결정적인 부분은 이 폴더에 있다** |

### 재생성 명령 (데이터셋만 있으면 됨)

```bash
# 홀드아웃 96 (016~018 의 모든 수치가 이 위에 있다. seed 고정이라 바이트 단위로 동일하다)
python scripts/build_holdout.py --train-root open/data/train --out artifacts/holdout \
    --n 96 --seed 0 --per-dataset-cap 2

# 누수 없는 홀드아웃 96 (C 평가용. 학습 split 을 배제한다)
python scripts/branchC/build_holdout_val.py --n 96 --out artifacts/holdout_val96

# 정지영상 대조군
python scripts/branchB/make_static_eval_preds.py \
    --eval-root artifacts/holdout --out artifacts/branchB/m0_step1000_b4/static_preds --expect 96
```

⚠ `build_holdout_val.py` 는 학습 config 의 `val_fraction`·`seed`·`traj_len`·`downsample`·
`camera_key` 와 **같은 값**을 써야 한다. 하나라도 다르면 분할이 달라져 누수가 남는다.

---

## ✅ 이식 검증 결과 (2026-08-02, 실제로 빈 폴더에 clone 해서 확인했다)

주장만 하지 않고 **`git clone` 을 실제로 돌려 확인했다.**

```
  clone 크기 18MB
  문서 31개 · 그림/영상 5개 · 결과 JSON 46개 · 스크립트 69개
  파이썬·쉘 문법 오류 0건 (compileall / bash -n)
  문서가 참조하는 스크립트 14개 전부 존재
  판정 근거 JSON 8개 전부 존재 (표본별 rows 포함)
```

### 남아 있는 깨진 상대링크 10개 — 전부 설계상 예상된 것

| 개수 | 무엇 | 왜 괜찮나 |
|---:|---|---|
| 6 | `../../open/baseline/...` | **대회 배포물.** 애초에 저장소에 안 올린다. 다시 받으면 풀린다 |
| 1 | `../../artifacts/branchB/overnight_chain_20260731.sh` | **같은 파일이 `scripts/branchB/` 에 있다.** 015 의 경로 표기가 잘못됐을 뿐, 내용 손실 없음 |
| 3 | `./dataset_analysis.md` (`rule/001`) | **원래부터 깨진 오타.** 이 기계에서도 깨져 있다. 실제 파일은 `model_select/001-dataset_analysis.md` |

⚠ 위 셋을 고치려면 기존 `_thinking` 문서를 수정해야 하는데,
**`_thinking/` 는 append-only 규칙**이라 손대지 않고 여기에 분류만 남긴다.

---

## 읽는 순서 (처음 보는 사람)

1. `_thinking/model_select/019-handoff.md` — **지금 어디에 서 있나**
2. `_thinking/model_select/016-*.md` — 현재 상태 전부 (리더보드 좌표·branch B 동결)
3. `_thinking/model_select/018-*.md` — 8/1 저녁의 판정 (덧셈 확정·평균내기 폐쇄·해상도 재해석)
4. `_thinking/model_select/017-*.md` — 문헌 조사
5. 이 폴더 — 위 문서들이 인용하는 **그림과 판정 출력의 원본**
