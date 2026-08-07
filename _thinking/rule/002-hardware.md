# 하드웨어·개발 환경 정리

> 목적: 이 프로젝트가 쓸 수 있는 GPU 3종과 각 장비의 역할, 그리고 재현 가능한
> 파이썬 환경을 한 곳에 기록한다. 004 모델선정 문서의 예산 계산(추론 1h / 학습 4d)이
> 어느 장비 기준인지, 개발은 어디서 하는지를 명확히 하기 위함.

## 1. 사용 가능한 GPU 3종

| 장비 | GPU | VRAM | OS | 가용성 | 역할 |
|---|---|---|---|---|---|
| **집** | RTX 4060 Ti | **8GB** (전용) | Windows 11 + WSL2 **Ubuntu 20.04** | 상시 | 개발·경량 실험(M0 채점, 소형 학습, 코드 작성) |
| **연구실** | RTX 5090 | **32GB** | 순정 **Ubuntu 24.04** | **내 차례일 때만** | 중형 학습·1.1B 추론 실측(M1)·본 실험 |
| **지원 예정** | RTX PRO 6000 (Blackwell) | **96GB** | 미정 | 교수님 지원 약속(추후) | 최종 본 학습(4일)·추론 1h 최종 검증 = 대회 재현 기준 장비 |

핵심:
- **대회 재현 검증 기준 = RTX PRO 6000 (96GB) 1대** (학습 ≤ 4일, 추론 ≤ 1h). 004의 예산은 이 장비 기준이다.
- 5090은 6000과 **같은 Blackwell 세대**라, 5090에서 잰 스텝당 시간이 6000 배율(r) 추정의 신뢰 근거가 된다(004 §5.2 M1).
- 4060 Ti(8GB)는 **1.1B 전체 추론(12.8GB)은 불가**, LoRA/부분·소형 학습과 채점은 가능.

## 2. 장비별 무엇을 하는가 (M0~M4 배정)

| 단계 | 내용 | 장비 |
|---|---|---|
| M0 채점 인프라 + static/GT 바닥값 | 채점 3종 소형 모델, peak VRAM 1.3GB | **4060 Ti (완료)** |
| M0 baseline(11M) 생성 | VAE/CLIP+11M UNet, 8GB 적합성 확인 중 | 4060 Ti 시도 → OOM 시 5090 |
| M1 1.1B 추론 실측 | 스텝당 시간·배치·VRAM → r 확정 | **5090** (8GB 불가) |
| M3 액션 CFG 파일럿(11M 짧은 재학습) | 소형 학습 | 4060 Ti 또는 5090 |
| M4 본 학습(4일) + 추론 1h 최종 검증 | 최종 재현 | **RTX PRO 6000** |

## 3. 재현 가능한 파이썬 환경

두 머신의 OS python이 다르다(WSL 20.04=3.8, Ubuntu 24.04=3.12). 버전 불일치와
pytorch-lightning 1.9.3의 구버전 호환성 문제를 피하려고 **conda로 python 3.10 고정**한다.

> 📌 **환경 경로·버전의 진실의 출처는 [env_file/001-environment_setup.md §0 두 기계 대조표](../env_file/001-environment_setup.md) 다.**
> 여기 적힌 값은 요약이므로, 어긋나면 **001 쪽이 맞다.** (2026-08-06 에 이 문서의 경로·torch 버전이
> 집 기준이라 틀렸던 것을 고치면서, 같은 사실이 두 문서에 중복되는 것 자체가 위험하다고 판단해
> 001 로 일원화했다.)
>
> 요약만: **[집]** `~/miniconda3/envs/wm` · torch 2.7.1+cu126 /
> **[연구실 5090]** `/home/rils/dlacksdn/miniconda3/envs/wm` · torch 2.7.1+cu128
> 🚨 5090 홈의 `~/anaconda3/envs/inha` 는 **남의 환경이다**(공유 계정). 이름이 비슷해 오인하기 쉽다.

- 설치 위치: 위 대조표 참조 (기계마다 다르다).
- 환경 이름: **`wm`** (python 3.10.20, 두 기계 공통). 활성화: `conda activate wm`.
- 핵심 패키지: torch 2.7.1(+cu 빌드는 기계별로 다름 — 001 §1-1), timm, pytorch-lightning 1.9.3(+setuptools<70), omegaconf,
  imageio-ffmpeg 등. 전체 목록은 저장소 루트 `requirements-scoring.txt` / `requirements-lock.txt`.
- 5090(Ubuntu 24.04)에서도 동일하게 `conda create -n wm python=3.10` 후 같은 패키지 설치하면 재현됨.
  CUDA 휠(cu126)은 Ada(4060 Ti)·Blackwell(5090/6000) 모두 호환.

주의(설치 중 발견한 함정):
- `pytorch-lightning==1.9.3`은 `pkg_resources`가 필요 → `setuptools<70` 병행 설치 필수.
- `action_extractor.ckpt` 언피클에 `omegaconf` 필요(submission_kit이 requirements에 미명시).
- 최신 imageio는 mp4를 pyav로 라우팅해 `macro_block_size` 인자를 거부 → `imageio-ffmpeg` 설치 후
  `format="FFMPEG"` 강제해야 submission_kit과 동일한 libx264 인코딩이 나온다.

## 4. 데이터·코드 위치

- 대회 패키지: 각 머신의 저장소 안 `open/`(집: WSL 네이티브 복사본 = `/mnt/c`보다 IO 빠름).
  `.gitignore`로 제외되므로 각 머신에 별도 배치. 연구실에도 이미 있음(사용자 확인).
- 데이터 루트는 코드에서 CLI 인자(기본 상대경로 `open/`)로 받아 두 머신 경로차를 흡수.
- backbone.ckpt(DynamiCrafter_512, ~10GB)는 대용량이라 gitignore. 각 머신에서 HF에서 직접 받는다.
