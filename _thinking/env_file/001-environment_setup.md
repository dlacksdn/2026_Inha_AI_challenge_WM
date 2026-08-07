# 환경 세팅·재현 런북 — 집(4060 Ti) / 연구실(5090) 두 기계

> 목적: 이 프로젝트의 파이썬 환경을 **두 기계 어디서든 그대로 재현**하는 법.
> 하드웨어 배경은 [rule/002-hardware.md](../rule/002-hardware.md), 파일이 어디 쌓이는지는
> [002-file_locations.md](./002-file_locations.md).
>
> **📖 이 문서를 읽는 법**
> 기계마다 다른 것은 전부 **`[집]` / `[연구실]`** 로 라벨했다. **라벨이 없으면 두 기계 공통이다.**
> 처음엔 "집에서 만들어 5090으로 옮기는 런북"으로 썼는데, 지금은 **연구실 5090이 주 작업 기계**다.
> 그래서 2026-08-06 에 두 기계를 대등하게 놓도록 구조를 바꿨다. 정정 이력은 §7.
>
> 근거 표시: `[실측 5090]` 2026-08-06 이 기계에서 직접 확인 / `[기록 집]` 2026-07-24 집에서 기록, 이후 재확인 안 됨

---

## 0. ⭐ 두 기계 대조표 — 제일 자주 볼 것

| 항목 | **[집]** | **[연구실]** |
|---|---|---|
| GPU | RTX 4060 Ti **8GB** | RTX 5090 **32GB** (32,607 MiB) |
| 드라이버 | — | 580.173.02 `[실측 5090]` |
| OS | Windows 11 + WSL2 **Ubuntu 20.04** | 순정 **Ubuntu 24.04.4 LTS** (커널 7.0.0-28) `[실측 5090]` |
| OS python | 3.8 (pip 없음) | 3.12 |
| 계정 | 개인 | **공유 계정 `rils`** ⚠ 여러 사람이 같이 쓴다 |
| conda 본체 | `~/miniconda3` `[기록 집]` | **`/home/rils/dlacksdn/miniconda3`** (conda 26.5.3) `[실측 5090]` |
| 프로젝트 루트 | 저장소를 둔 곳 | **`/home/rils/dlacksdn/2026_Inha_AI_challenge_WM`** |
| 디스크 | — | 5.5T 중 **1.8T 여유** `[실측 5090]` |

**환경 이름은 두 기계 다 `wm` 이고 python 3.10.20 이다.** 다른 건 conda 본체의 위치뿐이다.

### 0-1. 🚨 경로 함정 두 개 — 여기서 두 번 데였다

**① conda 경로가 기계마다 다르다.**

```
[집]     ~/miniconda3/envs/wm
[연구실] /home/rils/dlacksdn/miniconda3/envs/wm      ← ~/miniconda3 는 5090에 없다
```

016 §4.1 이 기록한 사고: *"conda 경로가 집 기계 기준(`~/miniconda3`)이라 5090 에서
생성·제출이 rc=127 로 죽었다."* 그런데 **이 문서 자체에 같은 함정이 2026-08-06 까지 남아 있었다**
(§7 정정 이력). 스크립트를 새로 쓸 때 특히 조심할 것.

**② 5090 에 `inha` 라는 헷갈리는 환경이 따로 있다. 우리 것이 아니다.**

```
/home/rils/anaconda3/envs/     ← 남의 것. dfp, inha, lattice-rl, oppo, rl310
/home/rils/dlacksdn/miniconda3/envs/wm   ← 우리 것
```

공유 계정이라 남의 anaconda 가 홈에 있다. 그 안의 **`inha`** 는 이름이 대회와 비슷해서
우리 환경으로 착각하기 쉽다. **손대지 마라.**

### 0-2. 그래서 이렇게 쓴다 — 모든 명령의 첫 줄

이 문서의 뒤쪽 명령은 전부 `$PY` 를 쓴다. **기계마다 이 한 줄만 바꾸면 나머지가 그대로 돈다.**

```bash
# [연구실]
PY=/home/rils/dlacksdn/miniconda3/envs/wm/bin/python
# [집]
PY=~/miniconda3/envs/wm/bin/python
```

`conda activate` 도 물론 된다 `[실측 5090]`. 다만 **긴 작업·스크립트에는 절대경로 인터프리터를
직접 쓰는 편이 안전하다** — 018 §운영 2번이 "셸 cwd 가 상위로 리셋되는 일이 있다"를 기록했고,
그때 상대경로가 조용히 깨졌다.

```bash
# [연구실] conda activate 를 쓰려면
source /home/rils/dlacksdn/miniconda3/etc/profile.d/conda.sh && conda activate wm
```

---

## 1. 왜 conda 인가 (공통)

두 머신의 OS python 이 다르다(WSL 20.04 = 3.8·pip 없음, Ubuntu 24.04 = 3.12). 여기에
`pytorch-lightning==1.9.3`(대회 pin)이 python 3.12 와 호환성 문제 소지가 있어,
**conda 로 python 3.10 을 두 머신에 동일 고정**한다.
CUDA 휠은 Ada(4060 Ti)·Blackwell(5090 / PRO 6000) 모두 호환되므로 GPU 가 달라도 같은 패키지로 재현된다.

- 정확한 버전 고정: 저장소 루트 [`requirements-lock.txt`](../../requirements-lock.txt) (pip freeze)
- 채점만 필요하면: [`requirements-scoring.txt`](../../requirements-scoring.txt)

### 1-1. 실제로 깔려 있는 버전

| | **[집]** `[기록 집]` | **[연구실]** `[실측 5090]` |
|---|---|---|
| python | 3.10.20 | 3.10.20 |
| torch | 2.7.1**+cu126** | 2.7.1**+cu128** |
| torchvision | 0.22.1 | 0.22.1 |
| numpy | 1.26.4 | 1.26.4 |
| 설치 패키지 수 | 82 | 85 |

> ⚠ **torch 의 CUDA 빌드가 두 기계에서 다르다.** §2-3 의 설치 명령이 `--index-url` 을 안 줘서
> **PyPI 기본 빌드**를 받는데, 그 기본값이 시점에 따라 바뀌었기 때문이다.
> 지금까지 이것 때문에 생긴 문제는 없다(둘 다 Blackwell·Ada 호환). 다만
> **"두 기계가 비트 단위로 같은 환경"은 아니다.** 채점 수치가 미세하게 어긋나면 여기를 의심할 것.

---

## 2. 처음 세팅하는 법

### 2-1. 저장소

```bash
git clone https://github.com/dlacksdn/2026_Inha_AI_challenge_WM.git
cd 2026_Inha_AI_challenge_WM
```

> **[연구실] push 는 절차가 따로 있다.** 공유 계정이라 남의 토큰이 저장돼 있어서 그냥 하면 403 이다.
> 반드시 [rule/003-git_push.md](../rule/003-git_push.md) 를 따를 것. (요약: dlacksdn PAT 를
> 터미널 프롬프트에 1회 입력, 저장 안 함.)

### 2-2. conda 설치 (없을 때만)

```bash
# [집]     -p ~/miniconda3
# [연구실] -p /home/rils/dlacksdn/miniconda3      ← 홈이 공유라 프로젝트 옆에 둔다
CONDA_ROOT=/home/rils/dlacksdn/miniconda3          # 기계에 맞게 바꾼다

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p "$CONDA_ROOT"
# conda 26.x 는 ToS 동의가 필요하다
"$CONDA_ROOT"/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$CONDA_ROOT"/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

> **[연구실] 왜 홈이 아니라 `/home/rils/dlacksdn/` 아래인가**: 홈(`/home/rils`)은 공유 계정이라
> 이미 남의 `anaconda3` 가 있다. 거기에 우리 것을 또 깔면 서로 덮어쓰거나 헷갈린다.

### 2-3. 환경 생성 + 패키지 (공통)

```bash
"$CONDA_ROOT"/bin/conda create -y -n wm python=3.10
PY="$CONDA_ROOT"/envs/wm/bin/python

$PY -m pip install torch==2.7.1 torchvision==0.22.1     # ⚠ §1-1: 기본 빌드라 기계별로 cu 버전이 다를 수 있다
$PY -m pip install -r requirements-lock.txt              # 나머지 전부 정확히 고정
```

> lock 설치가 torch 와 충돌하면, torch 두 줄을 뺀 사본으로 설치하거나
> `requirements-scoring.txt`(채점) + baseline deps 를 개별 설치해도 된다.
> baseline 생성까지 필요한 추가분:
> `transformers==4.48.3 open_clip_torch==2.22.0 kornia opencv-python pytorch-fid huggingface-hub==0.25.2`

### 2-4. 데이터·체크포인트 배치 (git 에 없다 — 기계마다 따로)

**[연구실] 이미 다 되어 있다.** `open/` 은 심링크이고, 아래가 실측 상태다 `[실측 5090]`:

```
open/baseline        -> /home/rils/inha_challenge_datasets/baseline            (항목 6)
open/submission_kit  -> /home/rils/inha_challenge_datasets/submission_kit      (항목 8)
open/data/train      -> /home/rils/다운로드/train                              (항목 56)
open/data/eval       -> /home/rils/inha_challenge_datasets/data/eval           (항목 2)
open/baseline/checkpoints/backbone.ckpt                                        9.8G
```

> ⚠ **`open/data/eval` 은 2026-08-01 에야 걸렸다.** 그 전엔 링크 자체가 없어서 eval 생성이
> 통째로 안 돌았다(016 §4.1). **새 기계에서 이 링크를 빠뜨리기 쉽다.**

**[집] 또는 새 기계**:

```bash
ln -s /데이터/경로/baseline        open/baseline
ln -s /데이터/경로/submission_kit  open/submission_kit
mkdir -p open/data
ln -s /데이터/경로/train           open/data/train
ln -s /데이터/경로/eval            open/data/eval        # ← 잊지 말 것

# backbone.ckpt (~10GB, DynamiCrafter_512) 는 git 제외라 재다운로드
wget https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt \
     -O open/baseline/checkpoints/backbone.ckpt
```

확인:
```bash
ls open/data/train | wc -l                                  # 제공자 폴더 수
ls open/data/eval                                           # images, actions
ls open/submission_kit/checkpoints/action_extractor.ckpt
```

> 🚨 **`open/` 은 프로젝트 밖을 가리킨다.** 주최 코드 안에서 `../outputs` 같은 상대경로를 쓰면
> **공유 폴더에 파일이 생긴다.** 우리 스크립트는 전부 절대경로로 프로젝트 안에 쓰도록 만들어 뒀다
> (env_file/002 §7.1). 새 스크립트도 이 규율을 지킬 것.

---

## 3. 설치 중 발견한 함정 4가지 (공통 — 이미 lock·스크립트에 반영됨)

1. **`pytorch-lightning==1.9.3` 이 `pkg_resources` 를 요구** → 최신 env 엔 setuptools 가 없어 import 실패.
   → `setuptools<70` 병행 설치 필수 (lock 에 `setuptools==69.5.1` 포함).
2. **`action_extractor.ckpt` 언피클에 `omegaconf` 필요** (submission_kit requirements 가 미명시한 암묵 의존성).
   → `omegaconf==2.1.1` 설치.
3. **최신 imageio 가 mp4 를 pyav 로 라우팅** → `macro_block_size` 인자 거부 → submission_kit 식
   libx264 인코딩 불가. → `imageio-ffmpeg` 설치 + 우리 코드(`data_utils.save_mp4_uint8`)가
   `format="FFMPEG"` 강제.
4. **numpy 는 1.26.4 로 고정.** baseline deps(transformers/open_clip)가 numpy 2.x 로 올리려 하는데
   submission_kit 은 numpy<2 를 요구한다. 채점·baseline 양쪽 1.26.4 에서 검증됨.

### 3-1. 나중에 추가로 데인 것 (환경이 아니라 배선. 그래도 여기 적어 둔다)

`[실측 5090]`

5. **`action_extractor` 는 양방향 GRU 라 eval 모드에서 cuDNN RNN backward 가 막힌다.**
   손실로 쓰려면 `torch.backends.cudnn.flags(enabled=False)` 안에서 forward 해야 한다.
   `model.train()` 으로 바꾸는 것은 **출력이 달라지므로 하면 안 된다.**
   cuDNN 을 꺼도 출력 상대차는 2.2e-05 로 같다(2026-08-06 확인).
6. **`torchvision.io` 가 libpng 를 못 찾는다는 경고**가 뜬다. 이미지 IO 를 안 쓰므로 무해하다.

---

## 4. 세팅 검증 게이트 — 이 3개가 통과해야 환경 OK

먼저 `$PY` 를 §0-2 대로 정한다. 아래는 기계 공통이다.

### G1. CUDA + 채점 3종 모델

```bash
cd <프로젝트 루트>
$PY -c "
import sys; sys.path.insert(0,'open/submission_kit'); sys.path.insert(0,'src')
import torch; from wm_eval.scoring import LocalScorer
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
sc = LocalScorer('open/submission_kit','open/data/train/so100_action_statistics.json')
v = torch.randint(0,256,(1,16,320,512,3),dtype=torch.uint8)
print('video', sc.video_feature(v).shape, 'dino', sc.dino_feature(v).shape, 'OK')
"
# 기대: video (1,512) / dino (1,16,384) / OK
#   [집]     cuda True 'NVIDIA GeForce RTX 4060 Ti'
#   [연구실] cuda True 'NVIDIA GeForce RTX 5090'
```

### G2. 홀드아웃 재현 (seed 고정 → 두 기계에서 같은 표본이 나와야 한다)

```bash
$PY scripts/build_holdout.py --train-root open/data/train \
    --out artifacts/holdout --n 96 --seed 0 --per-dataset-cap 2
# 기대: 96개 표본, 해상도 480x640 / 720x1280 / 1080x1920 혼재
```

### G3. 채점 재현 (수치가 두 기계에서 같아야 한다)

```bash
$PY scripts/run_m0.py --holdout artifacts/holdout --out artifacts/m0
# 기대 (n=96, 결정론적):
#   static:  DINO 0.123 / Video 0.091 / Action 1.240 / TOTAL 0.560
#   gt:      0.000      / 0.000       / 1.223        / TOTAL 0.489
# GPU 가 달라도 이 수치는 재현되어야 한다. 다르면 환경·데이터 불일치 신호다.
```

> baseline 까지 보려면: `bash scripts/gen_baseline.sh artifacts/holdout artifacts/baseline_preds 50`
> 후 `run_m0.py ... --pred-dir artifacts/baseline_preds --pred-name baseline`.
> 집 기준 baseline TOTAL 0.711 `[기록 집]`.

---

## 5. git 에 있는 것 / 없는 것 (공통)

| 대상 | git 추적 | 새 기계에서 |
|---|---|---|
| `src/`, `scripts/`, `_thinking/`, `results/`, `requirements-*.txt` | ✅ | pull 로 옴 |
| `open/` (데이터·submission_kit·baseline) | ❌ gitignore | 심링크 배치 (§2-4) |
| `open/baseline/checkpoints/backbone.ckpt` (9.8G) | ❌ | 재다운로드 (§2-4) |
| `artifacts/` (홀드아웃·생성영상·체크포인트) | ❌ | 재생성 (G2/G3) |
| `run_logs/` | ❌ | 재생성 |
| conda env `wm` | ❌ | 재생성 (§2-3) |

즉 **git clone + §2-3(env) + §2-4(데이터) + §4(검증)** 이면 한 기계 상태가 다른 기계에 복원된다.

> **[연구실] `artifacts/` 가 274G 다** `[실측 5090, 2026-08-06]`. 체크포인트(7.9G × 다수)가 대부분이다.
> 디스크는 1.8T 남아 여유가 있다. **프로젝트 규칙상 로그·체크포인트는 함부로 지우지 않는다.**

---

## 6. 장비별 실행 제약 (env 관점)

| 장비 | 할 수 있는 것 |
|---|---|
| **[집] 4060 Ti 8GB** | 채점·홀드아웃·baseline(11M) 생성 (DDIM50 ~5s/샘플). **1.1B 추론은 불가** (VRAM 초과) |
| **[연구실] 5090 32GB** | 위 전부 + 1.1B 추론·중형 학습. **현재 주 작업 기계** |
| RTX PRO 6000 96GB | 최종 본 학습·추론 1h 검증 = **대회 재현 검증 기준 장비** (rule/001 §3.3). 아직 지원 안 받음 |

**[연구실] 5090 운영 주의** (016 §6, 018 §운영):

- ⚠ **학습 중에는 브라우저를 띄우지 마라.** `systemd-oomd` 가 메모리 압박 74% 에서 프로세스를
  죽인 사고가 있었고, 브라우저가 공범으로 특정됐다. 체크포인트(5.8G) 쓰기와 겹치면 위험하다.
- 긴 작업은 tmux 로 돌리고 로그를 남긴다. **스크립트·로그 경로는 절대경로로** (cwd 리셋 사고).
- `pkill -f <패턴>` 은 자기 명령줄에 매치돼 셸을 죽인 적이 있다. `pgrep -f "패턴[끝글자]" | xargs -r kill`.

---

## 7. 정정 이력 — 이 문서가 틀렸던 것

기록물은 지우지 않는다. 무엇이 언제 왜 틀렸는지 남긴다.

| 날짜 | 무엇이 틀렸나 | 원인 | 어떻게 고쳤나 |
|---|---|---|---|
| 2026-08-06 | conda 경로를 전부 `~/miniconda3` 로 적었다 | 집에서 쓴 문서를 5090 에 그대로 적용. **실제 5090 경로는 `/home/rils/dlacksdn/miniconda3`** | §0 대조표로 두 기계를 분리. `$PY` 규약 도입 |
| 2026-08-06 | torch 를 `2.7.1+cu126` 이라 적었다 | 집 기준. 5090 실측은 **`+cu128`** | §1-1 에 두 기계 병기 |
| 2026-08-06 | `artifacts/` 를 33GB 라 적었다(002 문서) | 7월 말 값. 실측 **274G** | §5 갱신 |
| 2026-08-06 | 문서 구조가 "집 → 5090 이관 런북" 이었다 | 5090 이 주 작업 기계가 됐는데 구조가 안 따라왔다 | 두 기계를 대등하게 재구성 |
| — | `~/anaconda3/envs/inha` 를 우리 환경으로 오인 | 공유 계정에 남의 anaconda 가 있고 이름이 대회와 비슷 | §0-1 에 경고 명시 |

---

## 8. 이 문서를 쓸 당시의 다음 계획 (이력 — 지금은 지났다)

> ⚠ 아래는 2026-07-24 시점의 계획이다. **현재 좌표는
> [model_select/019-handoff.md](../model_select/019-handoff.md) 를 보라.**
> branch B(1.1B 확산)는 2026-08-01 에 동결됐고, 지금은 잔차 회귀(C) 노선이다.

env 게이트 통과 후 5090 에서 하려던 것:

- **M1 (1.1B 추론 예산 실측)**: DynamiCrafter 1.1B UNet(실측 config **1438M**, model_channels 320,
  channel_mult [1,2,4,4], attention_resolutions [4,2,1], in_ch 8)로 DDIM(50/25/16/10) ×
  batch(1/2/4) × CFG(on/off) 의 **sec/샘플·peak VRAM** 측정 → 216샘플 1시간 예산에 드는 조합 확정.
  → **완료됨.** 결과: 25스텝+CFG 42.9분 통과 / 50스텝+CFG 68.8분 초과 (015 경미-5).

> M0 결과 요약(이관 맥락): 넘어야 할 기준 = **static 0.560**(baseline 11M 은 0.711 로 더 나쁨).
> 상세 [results/m0/M0_FINDINGS.md](../../results/m0/M0_FINDINGS.md).
