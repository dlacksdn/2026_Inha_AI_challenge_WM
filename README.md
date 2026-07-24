# 2026 인하 AI 챌린지 — 로봇 월드 모델

현재 로봇 이미지 1장 + 미래 16스텝 6D 액션 → 미래 16프레임 영상 생성 (action-conditioned world model).
대회/데이터/모델 선정 분석은 [`_thinking/`](_thinking/) 참고. 이 README는 **코드 실행법**만 다룬다.

## 저장소 구조

```
src/wm_eval/        로컬 모의채점 라이브러리
  data_utils.py       train → 평가용 홀드아웃 빌더 (LeRobot 파싱, 동일 libx264 인코딩)
  scoring.py          submission_kit 재사용 채점 + 코사인 거리 집계
scripts/
  build_holdout.py    홀드아웃 생성 (images/actions/gt_videos/manifest)
  run_m0.py           바닥값 측정 (static / gt_upper_bound / baseline)
results/m0/         측정 결과(리포트 JSON + 홀드아웃 manifest + findings)
open/               대회 패키지(데이터·submission_kit·baseline). .gitignore로 제외 — 각 머신에 별도 배치
artifacts/          홀드아웃/생성영상 등 재현 가능한 산출물. .gitignore로 제외
```

- **데이터 위치**: 코드는 데이터 루트를 CLI 인자로 받는다(기본 `open/`, 저장소 상대경로). 각 머신에서 `open/`을 저장소 안에 두면 그대로 동작한다.
- **submission_kit은 절대 수정 금지**(대회 규칙). 채점 코드는 `open/submission_kit`을 import만 한다.
- **eval 데이터 미사용**: 홀드아웃은 `open/data/train`에서만 생성한다.

## 환경 (집: 4060 Ti / 연구실: 5090 Ubuntu 24.04 공통)

conda + python 3.10. 정확한 버전은 [`requirements-lock.txt`](requirements-lock.txt).

```bash
conda create -y -n wm python=3.10 && conda activate wm
pip install torch==2.7.1 torchvision==0.22.1          # CUDA 12.6 휠 (Ada/Blackwell 공통)
pip install -r requirements-scoring.txt               # 채점만: 아래 목록
# 채점 스택: numpy==1.26.4 pytorch-lightning==1.9.3 "setuptools<70" torchmetrics einops
#           av imageio imageio-ffmpeg pillow timm PyYAML omegaconf==2.1.1 pandas pyarrow
# baseline 생성까지: + transformers==4.48.3 open_clip_torch==2.22.0 kornia opencv-python pytorch-fid huggingface-hub==0.25.2
```

주의: `pytorch-lightning==1.9.3`은 `pkg_resources`가 필요해 `setuptools<70`을 함께 설치해야 하고,
`action_extractor.ckpt` 로드에 `omegaconf`가 필요하다(submission_kit이 명시 안 한 암묵 의존성).
mp4 인코딩은 submission_kit과 동일하게 `imageio-ffmpeg`(libx264, macro_block_size=1)을 강제한다.

## M0 재현 (바닥값 측정)

```bash
# 1) 홀드아웃 생성 (seed 고정 → 동일 표본. manifest로도 복원 가능)
python scripts/build_holdout.py --train-root open/data/train --out artifacts/holdout --n 96 --seed 0 --per-dataset-cap 2

# 2) static / gt_upper_bound 채점
python scripts/run_m0.py --holdout artifacts/holdout \
    --submission-kit open/submission_kit \
    --action-stats open/data/train/so100_action_statistics.json --out artifacts/m0

# 3) baseline(11M) 생성 후 채점 (backbone.ckpt 필요)
#   backbone: https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt → open/baseline/checkpoints/backbone.ckpt
#   생성: open/baseline/challenge_kit 의 generate_baseline_videos.py 에 --challenge-root artifacts/holdout 지정
python scripts/run_m0.py --holdout artifacts/holdout --pred-dir artifacts/baseline_preds --pred-name baseline --out artifacts/m0
```

결과는 [`results/m0/M0_FINDINGS.md`](results/m0/M0_FINDINGS.md) 참고.

## 채점 산식 주의

- Action(40%)은 로컬 완전 재현(추출기 MAE 스칼라). DINO/Video(각 30%)의 **서버 코사인 거리 집계 방식은 미문서화(미확인)**.
  표준 관례(video: `1-cos`, DINO: 프레임별 `1-cos` 평균)를 기본으로 하고 flatten 변형도 함께 리포트한다.
  둘의 차이가 미미함을 확인했으나, 절대 스케일은 static-repeat 실제 제출로 리더보드 대비 보정 권장.
