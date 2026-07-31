# 002 — 파일이 어디에 저장되는가 (동영상·로그·모델·결과)

> 작성일: 2026-07-31 (5090 세션) | 선행: [001 환경 세팅 런북](./001-environment_setup.md)
> 목적: "그거 어디 있지?" 를 매번 찾지 않도록, **무엇이 어디에 쌓이는지** 한 곳에 정리한다.
> 모든 경로는 프로젝트 루트 `/home/rils/dlacksdn/2026_Inha_AI_challenge_WM` 기준이다.

---

## 0. 한눈에 보는 지도

```
2026_Inha_AI_challenge_WM/
├── artifacts/        ← 무거운 산출물 전부 (동영상·체크포인트) · git 추적 안 됨 · 현재 33GB
├── run_logs/         ← 실행 로그 전부 · git 추적 안 됨 · 현재 1.2MB
├── results/          ← 가벼운 결과 요약(JSON·CSV) · git 추적 됨 · 2.1MB
├── scripts/          ← 우리가 만든 코드 · git 추적 됨
├── _thinking/        ← 문서(001, 002, …) · git 추적 됨
└── open/             ← ⚠️ 심링크! 대회 원본 데이터·주최 코드 (§7 주의)
```

**핵심 원칙 하나**: 용량이 큰 것(`artifacts/`, `run_logs/`)은 **git 에 안 올라간다**.
즉 **이 컴퓨터 디스크에만 존재**한다. push 해도 GitHub 에는 없다.

---

## 1. 🎬 동영상은 어디에 저장되나

### 1.1 우리 모델이 만든 영상 (제일 자주 볼 것)

| 경로 | 내용 | 개수 |
|---|---|---|
| `artifacts/branchB/preds_step1000_emafix/` | **1.1B step1000 생성 영상 (정상)** ← 최신·유효 | 96 |
| `artifacts/branchB/preds_step500_emafix/` | 1.1B step500 생성 영상 (정상) | 96 |
| `artifacts/branchB/preds_emafix4/` | 수정 검증용 4개 | 4 |
| `artifacts/branchB/preds_step1000/` | ⚠️ **EMA 버그 시절 노이즈 영상 (무효)** — 비교용으로만 보존 | 96 |

> 파일명은 `sample_000000.mp4` ~ `sample_000095.mp4` 로, **모든 폴더에서 같은 번호는 같은 장면**이다.
> 그래서 폴더만 바꿔가며 같은 번호를 열면 바로 비교가 된다.

### 1.2 비교 대상 영상

| 경로 | 내용 |
|---|---|
| `artifacts/holdout/gt_videos/` | **정답 영상 96개** — 우리가 맞춰야 할 것 |
| `artifacts/branchB/m0_*/static_preds/` | static 기준선(첫 프레임 16번 복사) — 움직임 없는 게 정상 |
| `artifacts/m3/partA/cfg10/` ~ `cfg40/` | 11M 모델의 λ-CFG 스윕 영상 (M3 실험, 각 96개) |
| `artifacts/m0/static_preds/` | M0 때 만든 static 기준선 |

### 1.3 영상 보는 법

```bash
cd ~/dlacksdn/2026_Inha_AI_challenge_WM

# 우리 모델 최신 결과 하나 열기
xdg-open artifacts/branchB/preds_step1000_emafix/sample_000000.mp4

# 정답과 나란히 비교 (같은 번호)
xdg-open artifacts/holdout/gt_videos/sample_000000.mp4

# 폴더 통째로 열기(파일관리자)
xdg-open artifacts/branchB/preds_step1000_emafix/
```

> **품질을 눈으로 판단하는 기준**: 정답 영상도 **거의 정지에 가깝다**(프레임 사이 픽셀 변화 평균 4.8).
> 그러니 "잔잔한 게 정상"이고, **지지직거리거나 프레임마다 확 바뀌면 고장**이다.
> (실제로 이 방법으로 EMA 버그를 잡았다 — 013 §4 함정 ③)

---

## 2. 📜 로그는 어디에 저장되나

### 2.1 우리가 남기는 실행 로그 — `run_logs/`

모든 학습·생성·채점·진단 로그가 여기 한 곳에 모인다. 파일명으로 무엇인지 알 수 있다.

| 파일명 패턴 | 내용 |
|---|---|
| `overnight_*.log` | 밤샘 파이프라인 전체 흐름 (단계별 요약) |
| `ov_*_s5_pilot.log` | 파일럿 **학습** 로그 (loss·속도) |
| `ov_*_s2_smoke.log` | 스모크 학습 20스텝 |
| `eval_*.log` | **생성 + 채점 + 판정** (제일 자주 볼 것) |
| `diag_*.log` | 진단 스크립트 (첫스텝 타이밍·체크포인트 키·EMA 버그 검증) |
| `5090_*.log` | 5090 환경 게이트·빌드 점검 |
| `ckpt_snapshots.log` | 체크포인트 백업 데몬 기록 |

```bash
ls -lt run_logs/                       # 최신순 목록
tail -f "$(ls -t run_logs/*.log | head -1)"    # 가장 최근 로그 실시간
tail -100 run_logs/overnight_0731_0227.log     # 밤샘 결과 요약
```

### 2.2 학습 프레임워크가 자동으로 남기는 로그

| 경로 | 내용 |
|---|---|
| `artifacts/branchB/train_out/logs/inha_action_diffusion_1p1b/version_*/metrics.csv` | **loss 원본 수치** (진행바 말고 이걸 봐야 정확) |
| `artifacts/branchB/train_out/inha_action_diffusion_1p1b/loginfo/` | 프레임워크 내부 로그 |
| `artifacts/branchB/train_out/inha_action_diffusion_1p1b/configs/` | 그때 쓴 설정 사본 (재현용) |

```bash
# loss 추세를 사람이 읽을 수 있게 (구간평균 + 막대그래프)
conda run -n wm python scripts/branchB/loss_curve.py

# 원본 숫자 그대로
column -s, -t artifacts/branchB/train_out/logs/inha_action_diffusion_1p1b/version_2/metrics.csv | less
```

> **주의**: `version_1`, `version_2` … 는 **실행할 때마다 번호가 올라간다.** 최신이 가장 큰 번호다.
> 헷갈리지 않게 중요한 건 `results/branchB/pilot_metrics_step1500.csv` 로 복사해 뒀다(git 추적).

---

## 3. 🧠 모델(체크포인트)은 어디에 저장되나

### 3.1 우리가 학습한 모델

| 경로 | 크기 | 내용 |
|---|---|---|
| `artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/last.ckpt` | 7.5G | **최신 학습본 (step 1000)** ← 평가·재개에 쓸 것 |
| `artifacts/branchB/train_out/.../checkpoints/epoch=0-step=1000.ckpt` | 7.5G | 같은 시점 |
| `artifacts/branchB/train_out/.../checkpoints/epoch=0-step=1500.ckpt` | 2.7G | ⚠️ **손상 파일** — 04:55 프로세스 사망 중 쓰다 만 것. 쓰지 말 것 |
| `artifacts/branchB/ckpt_snapshots/epoch=0-step=500.ckpt` | 7.5G | 학습곡선용 백업 |
| `artifacts/branchB/ckpt_snapshots/epoch=0-step=1000.ckpt` | 7.5G | 학습곡선용 백업 |
| `artifacts/m3/train_out/` | — | M3(11M) 실험 산출물 |

> ⚠️ **저장 정책 주의**: 학습 중 체크포인트는 500스텝마다 저장되는데 `save_top_k=1` 이라
> **이전 것을 지운다.** 학습곡선을 보려면 스냅샷 백업이 필요하다(013 §6.1).

### 3.2 남이 준 모델 (건드리지 말 것)

| 경로 | 크기 | 내용 |
|---|---|---|
| `open/baseline/checkpoints/backbone.ckpt` | 9.8G | **DynamiCrafter 사전학습 원본** — 모든 학습의 출발점 |
| `open/baseline/checkpoints/baseline_diffusion.ckpt` | 84M | 주최측 11M baseline |
| `open/baseline/checkpoints/tac_*.ckpt` | 43M×6 | 문서에 기록 없는 **선행 실험 흔적** — 독립성 규칙상 손대지 않음 |
| `open/submission_kit/checkpoints/action_extractor.ckpt` | — | 채점용 액션 추출기 |

### 3.3 체크포인트 확인 명령

```bash
# 어느 시점인지 확인
conda run -n wm python -c "
import torch; d=torch.load('artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/last.ckpt',
                           map_location='cpu', weights_only=False)
print('epoch', d['epoch'], '/ global_step', d['global_step'])"

# 전체 목록과 크기
find artifacts open/baseline/checkpoints -name "*.ckpt" -exec du -h {} \; | sort -k2
```

---

## 4. 📊 채점 결과·수치는 어디에

| 경로 | 내용 | git |
|---|---|---|
| `results/branchB/m0_step1000_emafix.json` | **1.1B step1000 채점 원본** | ✅ |
| `results/branchB/pilot_metrics_step1500.csv` | 학습 loss 원본 | ✅ |
| `results/branchB/preflight_*.json` | 환경 점검 결과 | ✅ |
| `artifacts/branchB/m0_step1000_emafix/m0_report.json` | 위와 같은 내용(원위치) | ❌ |
| `artifacts/branchB/m0_step500_emafix/m0_report.json` | step500 채점 | ❌ |
| `results/m0/M0_FINDINGS.md` | M0 바닥값 측정 결과 문서 | ✅ |
| `results/m3/partA/` | M3 λ-CFG 스윕 채점 | ✅ |
| `artifacts/branchB/loss_dino_curve.png` | **loss·DINO 그래프 그림** | ❌ |

```bash
# 채점 결과 예쁘게 보기
python3 -c "
import json; d=json.load(open('results/branchB/m0_step1000_emafix.json'))
for n,r in d['results'].items():
    m=r['mean']; print(f\"{n:22} DINO {m['dino_frame_avg']:.5f} Video {m['video']:.5f} TOTAL {m['total_frame_avg']:.5f}\")"

# 그래프 보기
xdg-open artifacts/branchB/loss_dino_curve.png
```

---

## 5. ⚙️ 설정 파일은 어디에

| 경로 | 내용 |
|---|---|
| `scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml` | **1.1B 학습 설정 원본** (여기를 고친다) |
| `scripts/branchB/configs/eval/gen_1p1b.yaml` | 생성 설정 원본 |
| `artifacts/branchB/_runtime_cfg/` | 위 파일들의 **실행용 사본** (경로가 치환된 것) — 직접 고치지 말 것, 매번 새로 만들어짐 |
| `open/baseline/challenge_kit/configs/` | 주최측 원본 설정 (수정 금지) |

> 설정 파일에는 `__REPO__` 라는 자리표시자가 들어 있고, 실행할 때 `cfg_paths.py` 가
> 진짜 경로로 바꿔서 `_runtime_cfg/` 에 사본을 만든다. **다른 컴퓨터에서도 그대로 돌게 하려는 장치**다.

---

## 6. 📦 데이터 원본은 어디에

| 경로 | 실제 위치 | 내용 |
|---|---|---|
| `open/data/train` | → `/home/rils/다운로드/train` (심링크) | 학습용 원본 데이터 56개 데이터셋 |
| `artifacts/holdout/` | (여기) | **우리가 뽑은 평가용 96표본** — images/actions/gt_videos |
| `artifacts/holdout_smoke4/` | (여기) | 빠른 테스트용 4표본 |
| `open/data/train/so100_action_statistics.json` | 심링크 대상 | 액션 정규화 통계 |

---

## 7. ⚠️ 꼭 알아야 할 주의사항

### 7.1 `open/` 은 심링크다 — 실제 위치가 프로젝트 밖이다

```
open/baseline        → /home/rils/inha_challenge_datasets/baseline
open/submission_kit  → /home/rils/inha_challenge_datasets/submission_kit
open/data/train      → /home/rils/다운로드/train
```

**왜 중요한가**: 주최 코드 안에서 `../outputs` 같은 상대경로를 쓰면 **프로젝트 밖(공유 폴더)에 파일이 생긴다.**
공유 컴퓨터라 남의 영역을 건드리게 되므로, 우리 스크립트는 전부 **절대경로로 프로젝트 안에** 쓰도록 만들어 뒀다.

### 7.2 git 에 안 올라가는 것들

`.gitignore` 규칙상 아래는 **이 컴퓨터에만 있다**:
- `artifacts/` (동영상·체크포인트 33GB)
- `run_logs/` (로그)
- `open/` (원본 데이터)
- `*.ckpt`, `*.mp4`, `*.log`

**push 해도 GitHub 에 없다.** 그래서 중요한 수치는 `results/` 로 복사해 추적한다.

### 7.3 용량

```
artifacts/                33GB   ← 체크포인트가 대부분(7.5GB × 4개)
open/baseline/checkpoints 11GB
디스크                     2.0T 남음 (여유 충분)
```
체크포인트 하나가 7.5GB 라 학습을 반복하면 빠르게 쌓인다. 주기적으로 확인:
```bash
du -sh artifacts/* | sort -h | tail -5
df -h .
```

---

## 8. 👀 사용자가 확인하면 좋은 것들

### 8.1 "잘 되고 있나?" 를 5초에 확인

```bash
cd ~/dlacksdn/2026_Inha_AI_challenge_WM
nvidia-smi                                      # GPU 가 일하고 있나
pgrep -af "train_1p1b|generate_baseline"        # 우리 작업이 살아있나
tmux ls                                         # 백그라운드 세션 목록
tail -5 "$(ls -t run_logs/*.log | head -1)"     # 최근 로그 끝부분
```

> **GPU 가 놀고 있다고 무조건 고장은 아니다.** ① 작업이 끝났거나 ② 모델 로딩 중(3분)이거나
> ③ 스왑에 밀려 멈춘 것일 수 있다. ③ 이면 `free -g` 에서 스왑이 차 있다.

### 8.2 생성 품질을 눈으로 확인 (가장 중요 — 실제로 버그를 잡았다)

```bash
# 우리 결과와 정답을 같은 번호로 열어 비교
xdg-open artifacts/branchB/preds_step1000_emafix/sample_000000.mp4
xdg-open artifacts/holdout/gt_videos/sample_000000.mp4
```
지지직거리면 고장, 잔잔하면 정상(정답도 거의 정지다).

### 8.3 학습 진척 확인

```bash
conda run -n wm python scripts/branchB/loss_curve.py     # loss 추세(터미널)
xdg-open artifacts/branchB/loss_dino_curve.png           # loss·DINO 그래프
```
> **loss 는 평탄한 게 정상이다**(warm-start 라서). 진짜 성능은 DINO 로만 판단한다 — 013 §5.4.

### 8.4 점수 확인

```bash
python3 -c "
import json,glob
for p in sorted(glob.glob('artifacts/branchB/m0_*/m0_report.json')):
    d=json.load(open(p))
    for n,r in d['results'].items():
        if n.startswith('b1p1b'):
            m=r['mean']; print(f\"{p.split('/')[2]:24} DINO {m['dino_frame_avg']:.5f} Video {m['video']:.5f} TOTAL {m['total_frame_avg']:.5f}\")"
```
기준: **static DINO 0.123 / Video 0.091 / TOTAL 0.560** 보다 낮으면 이긴 것.

### 8.5 저장 공간·정리 대상

```bash
du -sh artifacts/* | sort -h        # 뭐가 자리를 많이 먹나
```
- 지금 정리 후보: `artifacts/branchB/preds_step1000/`(EMA 버그 노이즈 영상 96개),
  `epoch=0-step=1500.ckpt`(2.7GB 손상 파일)
- 단 **프로젝트 규칙상 로그·체크포인트는 함부로 지우지 않는다.** 지울 땐 사용자 확인 후에.

### 8.6 git 상태

```bash
git log --oneline -5     # 최근 커밋
git status --short       # 아직 안 올린 변경
```
push 는 공유계정이라 특수하다 → [rule/003-git_push.md](../rule/003-git_push.md) 절차.

---

## 9. 한 줄 정리

| 찾는 것 | 가는 곳 |
|---|---|
| **동영상** | `artifacts/branchB/preds_*_emafix/` (우리 결과) · `artifacts/holdout/gt_videos/` (정답) |
| **로그** | `run_logs/` (전부 여기) |
| **모델** | `artifacts/branchB/train_out/.../checkpoints/last.ckpt` (우리 학습본) · `open/baseline/checkpoints/backbone.ckpt` (원본) |
| **점수** | `results/branchB/*.json` (추적됨) · `artifacts/branchB/m0_*/m0_report.json` (원위치) |
| **그래프** | `artifacts/branchB/loss_dino_curve.png` |
| **설정** | `scripts/branchB/configs/` |
