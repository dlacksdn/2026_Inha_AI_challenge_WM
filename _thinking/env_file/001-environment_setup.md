# 환경 세팅·재현 런북 (집 4060 Ti → 연구실 5090 이관)

> 목적: 이 프로젝트의 파이썬 환경을 **새 머신(연구실 5090, Ubuntu 24.04)에서 그대로 재현**하는 법.
> 집(4060 Ti / WSL2 Ubuntu 20.04)에서 이미 구축·검증된 환경(2026-07-24)을 기준으로 한다.
> 하드웨어 배경은 [rule/002-hardware.md](../rule/002-hardware.md), 실행할 실험은 마지막 §6 참고.

---

## 0. 왜 conda인가 (요약)

두 머신의 OS python이 다르다(WSL20.04=3.8·pip없음, Ubuntu24.04=3.12). 여기에
`pytorch-lightning==1.9.3`(대회 pin)이 python 3.12와 호환성 문제 소지가 있어, **conda로
python 3.10을 두 머신에 동일 고정**한다. CUDA 휠(cu126)은 Ada(4060 Ti)·Blackwell(5090/6000)
모두 호환되므로 GPU가 달라도 같은 패키지로 재현된다.

- conda 본체: `~/miniconda3` (홈에 하나. 표준 위치).
- 프로젝트 환경: `~/miniconda3/envs/wm` (python 3.10.20). 활성화 `conda activate wm`.
- 정확한 버전 고정: 저장소 루트 [`requirements-lock.txt`](../../requirements-lock.txt) (pip freeze, 82개).

---

## 1. 새 머신(5090) 세팅 — 복붙 순서

### 1-1. 저장소 pull
```bash
cd ~   # 또는 원하는 상위 폴더
git clone git@github.com:dlacksdn/2026_Inha_AI_challenge_WM.git   # 이미 있으면 git pull
cd 2026_Inha_AI_Challenge_WM
```

### 1-2. miniconda (없으면)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p ~/miniconda3
# conda 26.x는 ToS 동의 필요:
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 1-3. 환경 생성 + 패키지
```bash
export PATH=~/miniconda3/bin:$PATH
conda create -y -n wm python=3.10
# torch는 CUDA 휠로 (cu126, Blackwell 호환)
conda run -n wm python -m pip install torch==2.7.1 torchvision==0.22.1
# 나머지는 lock 파일로 정확히 재현 (torch 제외 전부):
conda run -n wm python -m pip install -r requirements-lock.txt
```
> lock 설치가 torch 충돌 나면, torch 두 줄을 뺀 사본으로 설치하거나
> `requirements-scoring.txt`(채점) + baseline deps를 개별 설치해도 된다.
> baseline 생성까지 필요한 추가분: `transformers==4.48.3 open_clip_torch==2.22.0 kornia opencv-python pytorch-fid huggingface-hub==0.25.2`.

### 1-4. 데이터·체크포인트 배치 (git에 없음 — 각 머신 별도)
- **대회 패키지 `open/`**: 저장소 안에 두면 코드가 상대경로로 바로 찾는다.
  연구실에 이미 데이터가 있으면 저장소 안 `open/`으로 심링크하거나 복사:
  ```bash
  ln -s /연구실/데이터/경로/open open      # 저장소 루트에서. open/data/train, open/submission_kit 등이 보여야 함
  ```
  확인: `ls open/data/train | wc -l` (제공자 폴더 수), `ls open/submission_kit/checkpoints/action_extractor.ckpt`.
- **backbone.ckpt (~10GB, DynamiCrafter_512)**: baseline 생성/1.1B 실측에 필요. git 제외라 재다운로드:
  ```bash
  wget https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt \
       -O open/baseline/checkpoints/backbone.ckpt
  ```

---

## 2. 설치 중 발견한 함정 3가지 (이미 lock/스크립트에 반영됨, 이해용)

1. **`pytorch-lightning==1.9.3`이 `pkg_resources`를 요구** → 최신 env엔 setuptools가 없어 import 실패.
   → `setuptools<70` 병행 설치 필수(lock에 `setuptools==69.5.1` 포함).
2. **`action_extractor.ckpt` 언피클에 `omegaconf` 필요**(submission_kit requirements가 미명시한 암묵 의존성).
   → `omegaconf==2.1.1` 설치.
3. **최신 imageio가 mp4를 pyav로 라우팅** → `macro_block_size` 인자 거부 → submission_kit식 libx264 인코딩 불가.
   → `imageio-ffmpeg` 설치 + 우리 코드(`data_utils.save_mp4_uint8`)가 `format="FFMPEG"` 강제.
4. (추가) **numpy는 1.26.4로 고정**. baseline deps(transformers/open_clip)가 numpy 2.x로 올리려 하는데,
   submission_kit은 numpy<2 요구. lock에 1.26.4 고정. 채점·baseline 양쪽 1.26.4에서 검증됨.

---

## 3. 세팅 검증 게이트 (이 3개가 통과해야 환경 OK)

### G1. CUDA + 채점 3종 모델
```bash
export PATH=~/miniconda3/bin:$PATH
cd 2026_Inha_AI_Challenge_WM
conda run -n wm python -c "
import sys; sys.path.insert(0,'open/submission_kit'); sys.path.insert(0,'src')
import torch; from wm_eval.scoring import LocalScorer
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
sc = LocalScorer('open/submission_kit','open/data/train/so100_action_statistics.json')
v = torch.randint(0,256,(1,16,320,512,3),dtype=torch.uint8)
print('video', sc.video_feature(v).shape, 'dino', sc.dino_feature(v).shape, 'OK')
"
# 기대: cuda True 'NVIDIA GeForce RTX 5090' / video (1,512) dino (1,16,384) OK
```

### G2. 홀드아웃 재현 (seed 고정 → 집과 동일 표본)
```bash
conda run -n wm python scripts/build_holdout.py --train-root open/data/train \
    --out artifacts/holdout --n 96 --seed 0 --per-dataset-cap 2
# 기대: 96개 표본, 해상도 480x640/720x1280/1080x1920 혼재
```

### G3. 채점 재현 (static/GT가 집 수치와 일치해야 함)
```bash
conda run -n wm python scripts/run_m0.py --holdout artifacts/holdout --out artifacts/m0
# 기대(집 4060 Ti n=96): static TOTAL 0.560, gt_upper_bound DINO/Video=0.000 (파이프라인 정확)
#   static:  DINO 0.123 / Video 0.091 / Action 1.240
#   gt:      0 / 0 / 1.223
# GPU가 달라도 이 수치는 결정론적으로 재현되어야 한다(다르면 환경/데이터 불일치 신호).
```
> baseline까지: `bash scripts/gen_baseline.sh artifacts/holdout artifacts/baseline_preds 50` 후
> `run_m0.py ... --pred-dir artifacts/baseline_preds --pred-name baseline`. 집 기준 baseline TOTAL 0.711.

---

## 4. 저장소 안에서 무엇이 git에 있고 없는가

| 대상 | git 추적 | 새 머신에서 |
|---|---|---|
| `src/`, `scripts/`, `_thinking/`, `results/`, `requirements-*.txt` | ✅ | pull로 옴 |
| `open/` (데이터·submission_kit·baseline_diffusion.ckpt·action_extractor.ckpt) | ❌ gitignore | 연구실 데이터 배치(§1-4) |
| `open/baseline/checkpoints/backbone.ckpt` (~10GB) | ❌ | 재다운로드(§1-4) |
| `artifacts/` (홀드아웃·생성영상) | ❌ | 재생성(G2/G3) |
| conda env `wm` | ❌ (원래 파일시스템) | 재생성(§1-3) |

즉 **git pull + §1-3(env) + §1-4(데이터/backbone) + §3(검증)** 이면 집 상태가 5090에 복원된다.

---

## 5. 장비별 실행 제약 (env 관점)

- 4060 Ti(8GB): 채점·홀드아웃·baseline(11M) 생성 OK(DDIM50 ~5s/샘플). **1.1B 추론은 불가**(VRAM 초과).
- 5090(32GB): 위 전부 + **1.1B 추론 실측 가능**(M1). 중형 학습 가능.
- RTX PRO 6000(96GB): 최종 본 학습·추론 1h 검증(대회 재현 기준).

---

## 6. 환경 검증 후 이어서 할 일 (M1) — 개요만

env 게이트(§3) 통과 후 5090에서:
- **M1 (1.1B 추론 예산 실측)**: DynamiCrafter 1.1B UNet(실측 config: **1438M**, model_channels 320,
  channel_mult [1,2,4,4], attention_resolutions [4,2,1], in_ch 8, backbone.ckpt에서 확인)로
  DDIM(50/25/16/10) × batch(1/2/4) × CFG(on/off) 조합의 **sec/샘플·peak VRAM**을 측정 →
  216샘플 1시간 예산에 드는 조합 확정 + RTX PRO 6000 배율(r) 추정.
- 상세 태스크 정의는 별도 프롬프트/문서로 전달(이 문서는 env 재현에 한정).

> M0 결과 요약(이관 맥락): 넘어야 할 기준 = **static 0.560**(baseline 11M은 0.711로 더 나쁨).
> Action(40%)은 상수(~1.2). 승부처는 60%(DINO+Video)의 "정확한 작은 움직임". 상세 [results/m0/M0_FINDINGS.md](../../results/m0/M0_FINDINGS.md).
