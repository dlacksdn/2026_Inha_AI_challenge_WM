# 006 — M3: 액션 CFG 파일럿 ("생성이 static 0.560을 넘는가")

> 작성일: 2026-07-25 (밤 자율 세션) | 장비: 연구실 RTX 5090 32GB (Ubuntu 24.04, 공유계정 rils)
> 선행: [004 §5.5/§7](./004-model_selection_final.md)(M3 근거·순서), [005](./005-5090_env_and_m1.md)(5090 환경 재현+M1),
>       [M0 결과](../../results/m0/M0_FINDINGS.md)(넘어야 할 기준 = static TOTAL 0.560 / DINO 0.123 / Video 0.091)
> 코드/결과: `scripts/m3/*` (스캐폴드·드라이버), `results/m3/partA/*`(채점 JSON·모션분석·실행로그)
> 목적: 004 §7의 M3 — "생성 모델이 정지영상(static) 바닥값을 실제로 넘는가"를 **싸게** 확인한다.
>   Part A(무학습): 학습된 baseline_diffusion.ckpt(11M)로 λ(=unconditional_guidance_scale) 스윕.
>   Part B(재학습): 11M을 action_dropout_prob=0.1로 warm-start 재학습 후 λ 스윕.

---

## 0. 한 줄 요약

- **Part A는 결정적 음성**: 학습된 11M의 λ-CFG 스윕에서 λ를 올릴수록 DINO·Video·TOTAL이 **단조롭게 나빠졌다.**
  최선(λ=1.0, CFG off) TOTAL 0.706조차 static 0.560을 **크게 밑돈다**(+0.146 나쁨). 어떤 λ도 static의 DINO/Video 근처도 못 갔다.
- **메커니즘 정량 확인**: 생성 영상의 프레임간 움직임이 λ=1.0의 GT 1.8배에서 λ=4.0의 2.4배로 **단조 증가**(corr(λ,motion)=+0.997).
  baseline은 이미 GT보다 과하게 움직이는데(M0의 드리프트/환각), **CFG가 그 과잉 움직임을 더 증폭**해 지표를 악화시킨다.
  004 R6의 "비단조면 CFG 포기"보다 나쁜 **"단조 악화" 신호** → 현행 배선의 액션 CFG는 품질 손잡이로 **폐기**.
- **Part B(재학습)는 환경 문제로 미완**: Part A 완료 직후 공유머신의 **NVIDIA 드라이버 userspace가 업데이트되어 커널 모듈과 불일치**(580.173 vs 580.159.03),
  새 CUDA 프로세스가 전부 `Error 804: forward compatibility ... non supported HW`로 실패. 재부팅/모듈 리로드(sudo)가 필요해 무인 야간엔 해소 불가.
  Part B 코드·config·warm-start는 **모델 빌드 직전까지 검증 완료**, GPU 복구 즉시 실행 가능하게 남김(§6). 복구 감시 폴러 가동 중.
- **잠정 판정(Part B 대기)**: 값싼 CFG 레버는 11M의 드리프트를 **못 고치고 악화시킨다.** M3의 최종 판정("트레이닝으로 CFG를 유효화하면 넘는가")은 Part B가 필요하지만,
  Part A와 M0 진단은 이미 **"11M의 본질 문제는 용량이 아니라 시간적 드리프트"**를 재확인한다 → 무게추가 **branch B(1.1B, 강한 시간 프라이어로 anti-drift)** 또는 **anti-drift 설계**로 이동(§5).

---

## 1. 배경 — 왜 이 실험인가 (M0·004 복습)

- **넘어야 할 기준은 baseline이 아니라 static이다.** M0에서 주최 baseline(11M)은 TOTAL 0.711로 static 0.560보다 **나빴다**(모든 표본에서).
  원인은 용량 부족이 아니라 16프레임 롤아웃에서 **정지 장면을 못 지키고 과한 움직임을 환각**(시간적 드리프트)해 DINO/Video 거리가 폭발하는 것.
- **승부처는 60%(DINO+Video)의 "정확한 작은 움직임".** Action(40%)은 세 예측기 모두 ~1.2로 상수에 가까워 생성 품질로 못 낮춘다(M0 발견 2).
- **004의 M3 설계(§5.5, §7, R6)**: 액션 CFG(`action_dropout_prob`을 켜 학습 → 추론 시 λ>1로 액션 조건 강조)가 "유일한 손잡이"는 아니고,
  λ는 "모델이 액션 조건을 무시하지 않게 하는" 지점까지만 유익하며 그 뒤는 비단조라고 가정. **λ 반응 곡선을 소표본으로 먼저 확인**하고,
  비단조·무효면 CFG를 포기하고 NFE를 품질에 재투자하라(R6).
- **중요한 코드 사실(004 §5.5)**: 현 CFG의 uncond 분기(`prepare_batch_for_inference`의 uc)는 액션뿐 아니라 **텍스트·이미지 조건도 함께 null** 처리한다.
  즉 순수 "액션만" guidance가 아니다. Part A는 그 상태 그대로의 λ 반응을 본다.

---

## 2. 방법 — 정확히 무엇을 어떻게 쟀나

### 2.1 공통 프로토콜
- **홀드아웃**: G2 산출물 train 96표본(seed=0), images/actions/gt_videos 각 96. eval 데이터 **절대 미사용**.
- **생성**: 주최 `generate_baseline_videos.py`를 **수정 없이** 호출(`--config`로 λ만 교체). DDIM 50스텝, fps=6, precision fp16(autocast), seed=0(모든 λ 동일 초기노이즈 → λ 효과만 분리).
- **채점**: `scripts/run_m0.py`로 홀드아웃 GT 대비 DINO(frame-avg)·Video·Action MAE·TOTAL(0.3·0.3·0.4). libx264 mp4 저장→재로드 경로(M0와 동일).
- **기준선(M0, n=96)**: static DINO 0.12304 / Video 0.09113 / Action 1.24017 / **TOTAL 0.56032**. gt 상한 DINO/Video 0 / Action 1.22278 / TOTAL 0.48911.
- **드라이버**: 스캐폴드는 `scripts/m3/`(git 추적), sweep eval config·추론용 model config·생성+채점 드라이버(`sweep_and_score.sh`)·요약(`summarize.py`)·모션분석(`analyze_motion.py`)·Part B 학습 래퍼(`train_m3.py`)/config/런처(`run_partB_train.sh`).

### 2.2 환경 함정 5가지 (이번에 새로 특성화 — 재현 필수)
1. **`open/baseline`·`open/submission_kit`은 심링크로 dlacksdn 밖**(`/home/rils/inha_challenge_datasets/…`)을 가리킨다.
   challenge_kit의 **물리 CWD가 repo 밖**이라 baseline의 상대경로 `../outputs`·`../checkpoints`는 dlacksdn 밖을 쓴다.
   → 학습/추론 산출물은 **절대경로로 repo 내부**(`artifacts/m3/…`, `results/m3/…`)에 고정. 공유머신 규칙(수정은 dlacksdn 안에서만) 준수. model_config_file·경로 전부 절대경로화.
2. **스톡 생성 스크립트가 open_clip ViT-H(~3.9GB)를 다운로드하려다 느린 랩망에서 정지.** 그런데 backbone.ckpt가 CLIP을 곧바로 덮어쓰므로 이 다운로드는 **무의미**.
   → M1과 동일하게 `version=null`(추론용 model config `inha_action_diffusion_11M_infer.yaml`)로 랜덤 빌드 후 backbone 로드, `HF_HUB_OFFLINE=1`로 이중 차단.
3. **eval config의 `ddim_eta`는 샘플러가 무시한다.** `DDIMSampler.sample()`의 인자명은 `eta`(기본 0.0)인데 config는 `ddim_eta`로 넘겨 `**kwargs`에 삼켜진다
   → **항상 결정론적 DDIM(eta=0)**. 반면 `unconditional_guidance_scale`·`guidance_rescale`·`timestep_spacing`은 명시적 인자라 **정상 적용**(그래서 λ 스윕은 유효). guidance_rescale은 모든 점에서 0.7 고정.
4. **채점 모델 캐시 경로.** 채점기는 DINOv2를 `timm`(HF `timm/vit_small_patch14_dinov2.lvd142m`), R3D를 torch hub(`r3d_18`)에서 로드하며, 이 가중치는
   `HF_HOME=/home/rils/dlacksdn/.cache/hf`·`TORCH_HOME=/home/rils/dlacksdn/.cache/torch`(태스크 recipe 경로)에 캐시돼 있다. HF_HOME을 repo-local 빈 폴더로 잘못 주고 `HF_HUB_OFFLINE=1`을 걸면 `LocalEntryNotFoundError`로 채점이 죽는다 → **반드시 recipe 캐시 경로 사용.**
5. **학습 시 `video_utils` 임포트.** UNet(`openaimodel3d.py`)이 `from video_utils.helpers import prob_mask_like`를 임포트한다. 이 패키지는 `open/baseline/shared_libs/video_utils`(외부 dir 안에 동명 패키지)에 있고
   **wm env엔 미설치**라, 학습 런처 PYTHONPATH에 이 외부 dir을 넣어야 한다(생성 드라이버는 이미 포함). 누락 시 `ModuleNotFoundError: video_utils`.

> **파이프라인 새너티(λ=1.0)**: baseline_diffusion.ckpt를 λ=1.0으로 홀드아웃 96 생성·채점 → DINO 0.550 / Video 0.215 / Action 1.192 / **TOTAL 0.706**.
> M0 홈(4060Ti) baseline 0.711과 **소수 2자리(0.71)에서 일치**(잔차 ~0.005는 5090 GPU/fp16 수치차 + 홈↔랩 장비차). **5090에서 M3 생성·채점 경로가 M0와 정합함을 확인.**

### 2.3 Part A (무학습 λ 스윕) — 완료
- 체크포인트: 주최 `baseline_diffusion.ckpt`(학습된 11M UNet, action_dropout_prob=0.0으로 학습됨).
- λ ∈ {1.0(off), 1.5, 2.0, 2.5, 3.0, 4.0} (004는 {1.0,1.5,2.5,4.0}; 곡선 정밀화 위해 2.0·3.0 추가).

### 2.4 Part B (재학습 후 λ 스윕) — 환경 블록으로 미완 (§4.3, §6)
- baseline 코드 미수정 래퍼 `scripts/m3/train_m3.py`: `get_model`(backbone에서 VAE/CLIP만 로드, UNet 랜덤) 직후 LitEma 초기화 **전에**
  baseline_diffusion.ckpt(학습된 UNet, `model.diffusion_model.*` 1107키 + `model_ema.*` 1109키)를 `strict=False`로 덧씌워 **warm-start**. 그 위에서 `action_dropout_prob=0.1`로 이어 학습.
- 목적: uncond(액션 null) 분기를 **학습시켜** λ>1 CFG가 잘 정의되게 만든 뒤, 같은 λ 스윕이 이제 static을 넘는지 확인(= Part A가 못 다룬 "CFG를 위해 학습된 모델"의 공정한 시험).
- 시간박스: max_steps 6000 / max_time 2h, 체크포인트 500스텝마다(save_top_k=-1, 중간 끊김 대비). logdir는 repo 내부 절대경로.

---

## 3. 결과 — Part A

### 3.1 λ 반응 곡선 (baseline 11M, n=96; 0에 가까울수록 좋음)

| λ | DINO(favg) | Video | Action | **TOTAL** | Δ(TOTAL−static) |
|---:|---:|---:|---:|---:|---:|
| **1.0** (off) | 0.54968 | 0.21534 | 1.19163 | **0.70616** | +0.146 |
| 1.5 | 0.55740 | 0.22031 | 1.19324 | 0.71061 | +0.150 |
| 2.0 | 0.56364 | 0.22521 | 1.19285 | 0.71379 | +0.153 |
| 2.5 | 0.57144 | 0.23078 | 1.19648 | 0.71926 | +0.159 |
| 3.0 | 0.57973 | 0.23600 | 1.19436 | 0.72247 | +0.162 |
| 4.0 | 0.59587 | 0.24831 | 1.19430 | 0.73098 | +0.171 |
| **static (기준)** | **0.12304** | **0.09113** | 1.24017 | **0.56032** | 0 |
| gt (상한) | 0.00000 | 0.00000 | 1.22278 | 0.48911 | −0.071 |

- **λ↑ → DINO·Video·TOTAL 전부 단조 악화.** 세 성분 모두 λ에 대해 증가(나빠짐). λ=1.0→4.0에서 DINO +8%, Video +15%, TOTAL +3.5%.
- **어떤 λ도 static을 못 넘는다.** 최선 λ=1.0의 DINO 0.550는 static 0.123의 **4.5배**, Video 0.215는 static 0.091의 **2.4배**. 격차가 압도적.
- **Action은 λ에 사실상 불변**(1.192~1.196). 차원별 MAE 변동도 작다(대부분 <0.05, 최대 0.08=shoulder_lift), wrist_roll(eval +1.5σ 이탈, M2)도 ~0.99 고정 → **CFG로 Action 개선 불가**(M0 발견 2·004 §5.5 재확인).

### 3.2 메커니즘 — CFG는 과잉 움직임을 증폭한다 (모션 분석)

프레임간 평균 |Δpixel|(움직임 크기; `scripts/m3/analyze_motion.py`, n=96):

| | GT | λ=1.0 | λ=1.5 | λ=2.0 | λ=2.5 | λ=3.0 | λ=4.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| motion | 4.81 | 8.81 | 9.15 | 9.54 | 9.99 | 10.46 | 11.45 |
| GT 대비 | 1.0× | 1.8× | 1.9× | 2.0× | 2.1× | 2.2× | **2.4×** |

- **corr(λ, motion) = +0.997**(거의 완전 단조 증가). λ=1.0→4.0에서 움직임 +30%.
- 해석: GT의 미래는 거의 정지(motion 4.81, M0의 "프레임간 변화 2~4"와 정합)인데, 11M은 λ=1.0에서 이미 **GT의 1.8배로 과하게 움직인다**(드리프트).
  CFG(λ>1)는 조건 방향으로 예측을 밀어 **움직임을 더 키우고**, GT(정지)에서 더 멀어진다. 이 픽셀 모션↑는 점수 곡선의 단조 악화와 **방향·순서가 정합한다**(표본 단위 인과 증명이 아니라 λ-집계 공변 corr=+0.997 근거의 프록시).
- 왜 이 방향인가: 현 uncond 분기가 액션·텍스트·이미지를 전부 null 처리 → dropout=0.0으로 학습된 모델에겐 uncond가 OOD.
  CFG `cond + λ(cond−uncond)`가 "조건이 유도하는 (과한) 움직임"을 증폭하는 쪽으로 작동한다. Part B(uncond 분기 학습)가 이 방향을 바꿀 수 있는지가 남은 질문(§4.3).

---

## 4. 결과 — Part B (환경 블록)

### 4.1 무엇까지 됐나
- 스캐폴드·config·warm-start 래퍼 완성, 모든 yaml 파싱·문법 검증 통과.
- 학습 런처 실행 시 함정 2개를 잡아 고침: (a) `video_utils` PYTHONPATH 누락(§2.2-5), (b) `torch.distributed.launch`가 child traceback을 숨김 → 단일 GPU라 env 수동설정+직접 python 실행으로 전환(traceback 가시).
- 수정 후 재실행 시 모델 빌드까지 진입 확인: `LatentVisualDiffusion: Running in v-prediction mode` / `Keeping EMAs of 1109` / VAE·timm 로드 진행.
- 그 프로세스는 **CPU 모델빌드 단계에서 내가 수동 종료(SIGTERM)**했다(partB_train.log 마지막 줄 `종료됨`). 아래 §4.2의 **독립 CUDA 확인으로 블록을 확정한 뒤**였다 → **partB_train.log 자체엔 CUDA 오류·traceback이 없다**(빌드는 CPU라 GPU를 건드리기 전까지 진행됨). 즉 블록의 근거는 이 로그가 아니라 §4.2의 증거 파일이다.
- **warm-start 경로는 CPU로 사전 검증**(`scripts/m3/verify_warmstart.py`, CUDA 불필요): M3 config로 빌드한 UNet 1107키 ↔ baseline_diffusion.ckpt의 `model.diffusion_model.*` 1107키가 **완전 매칭(unet_missing=0, extra=0)**. 즉 복구 후 warm-start가 전 UNet 가중치를 로드함이 확정.
- **버그 수정(재개 de-risk)**: M3 학습 config의 CLIP에도 `version: null`을 추가했다. 없으면 학습이 open_clip(3.9GB)를 다운로드하려다 offline에서 실패한다(생성과 동일 함정 §2.2-2). only_reload_modules가 backbone에서 CLIP 실가중치를 로드하므로 안전.

### 4.2 무엇이 막았나 — NVIDIA 드라이버 userspace/커널 불일치
- **Part A 완료(≈06:06) 직후** 공유머신에서 NVIDIA 유저스페이스가 업데이트된 것으로 보임: **커널 NVRM 580.159.03 ↔ userspace NVML 580.173** 불일치.
- **탐지·증거(캡처됨)**: partB 학습 로그가 아니라 **독립 확인 스크립트**(`cuda_check.py`)와 `nvidia-smi`로 확정하고 파일로 남겼다 →
  [`results/m3/partB/driver_block_evidence.txt`](../../results/m3/partB/driver_block_evidence.txt): (1) 커널 `NVRM 580.159.03`, (2) `nvidia-smi: Failed to initialize NVML: Driver/library version mismatch / NVML library version: 580.173`,
  (3) 새 torch 프로세스 `Error 804: forward compatibility was attempted on non supported HW`, `cuda.is_available: False`. (06:14 최초 관측, 06:37 재확인에도 지속.)
- **정정(로그 정합)**: partB 학습 프로세스 자체는 CUDA 오류로 크래시한 게 아니라, 위 독립 확인으로 블록을 안 뒤 **내가 CPU 빌드 단계에서 수동 종료**했다. Error 804는 CUDA를 실제로 건드리는 프로세스(cuda_check)에서 **즉시** 나며, CPU 모델빌드는 GPU를 늦게 건드려 그 전까지 진행된다(그래서 partB 로그가 빌드까지 갔다).
- Part A(05:11~06:06)는 정상 CUDA 사용 → 그 직후 유저스페이스 업데이트가 원인임이 **시간적으로 정합**(업데이트 주체·정확 시각은 미상; 커널 모듈 리로드 전까지 mismatch 지속).
- **해소 조건**: nvidia 커널 모듈 리로드(`rmmod/modprobe`, GPU 프로세스 없어야) 또는 **재부팅**. 둘 다 sudo·시스템 레벨이며 **공유머신에서 함부로 못 한다**(타인 작업 중단 위험, dlacksdn 밖).
  → 무인 야간엔 해소 불가. **아침에 사용자가 재부팅**하면 Part B 즉시 실행 가능.

### 4.3 Part B가 아직 답하지 못한 것 (중요)
- Part A는 "**CFG를 위해 학습되지 않은**(dropout=0.0) 모델"에 CFG를 얹은 결과다. uncond 분기가 OOD라 CFG가 과잉 움직임을 증폭했을 가능성.
- Part B는 dropout=0.1로 **null-action 분기를 학습**시켜 CFG를 "잘 정의된" 상태로 만든 뒤 같은 λ 스윕을 본다. 이때
  (i) λ=1.0(순수 조건)이 Part A의 0.706보다 나아지는가(=짧은 재학습이 드리프트를 줄이는가), (ii) λ>1이 이제 **도움**이 되는가(곡선이 평탄/개선으로 바뀌는가).
- **예상(잠정)**: warm-start 짧은 재학습은 조건부 예측 자체를 크게 바꾸지 않으므로 λ=1.0은 여전히 ~0.70 근처일 공산이 크고, GT가 거의 정지인 이상 CFG가 움직임을 키우는 방향은 유지될 가능성이 높다. 그러나 **이는 반드시 실측으로 확인**해야 하며(측정 없는 서사 금지, 004의 규율), Part B 없이 M3를 "완결"로 선언하지 않는다.

---

## 5. 판정

### 5.1 확정된 것 (Part A + M0)
- **현행 배선의 액션 CFG(λ 스윕)는 11M에서 품질 손잡이가 아니다 — 폐기.** λ↑는 DINO/Video/TOTAL을 단조 악화시키고(004 R6의 "비단조" 가정보다 더 나쁨), 어떤 λ도 static 근처도 못 간다.
- **11M의 병목은 용량이 아니라 시간적 드리프트다**(모션이 GT의 1.8~2.4배). CFG는 이 드리프트를 **증폭**한다. Action은 λ·CFG로 못 움직인다(40% 상수 확정).
- 실용 함의: 제출 파이프라인에서 **CFG를 켤 이유가 없다**(NFE만 2배 낭비). NFE는 스텝 수(품질)로만 투자.

### 5.2 남은 관문 (Part B) 과 분기 방향
- **M3 최종 판정은 Part B 필요**: "CFG를 위해 학습된 11M이 static을 넘는가". 환경 복구 후 즉시 실행(§6).
- **다만 무게추는 이미 이동**: 값싼 손잡이(CFG)가 드리프트를 못 고치는 것이 확인됐으므로, static을 넘으려면
  ① **branch B(DC 1.1B 승격)** — 강한 사전학습 시간 프라이어로 드리프트 자체를 줄이는 경로(M1에서 추론 예산은 이미 해소), 또는
  ② **anti-drift 설계** — 첫 프레임 보존·정지 장면 유지·시간 일관성 손실을 직접 겨냥(예: 앵커링/저모션 정규화)로 무게가 쏠린다.
- **주의**: Part A의 음성이 곧 "1.1B가 이긴다"는 증명은 아니다(DINO/Video는 저해상·전역이라 용량 이득 경로가 좁다는 004 C1은 유효). branch B도 "드리프트를 실제로 줄이는가"를 M4 초기 학습에서 **모션 지표로 검증**해야 한다. 즉 다음 실험의 1차 성공 판정은 TOTAL이 아니라 **motion이 GT(≈4.8)에 접근하는가**로 두는 것이 정보량이 크다.

---

## 6. Part B 재개 절차 (환경 복구 후 — 아침에)

```bash
# 0) 드라이버 복구 확인 (재부팅 후)
nvidia-smi                              # 정상 출력 & 버전 일치 확인
CONDA=/home/rils/dlacksdn/miniconda3/bin/conda
$CONDA run -n wm python -c "import torch;print(torch.cuda.is_available())"   # True 여야 함

cd /home/rils/dlacksdn/2026_Inha_AI_challenge_WM

# 1) 학습 (warm-start, dropout=0.1, 2h 시간박스, 500스텝마다 체크포인트)
bash scripts/m3/run_partB_train.sh "" "00:02:00:00"
#   로그: results/m3/partB/partB_train.log
#   체크포인트: artifacts/m3/train_out/inha_action_diffusion_11M_m3/checkpoints/
#   warm-start 확인: 로그의 ">>> [M3] warm-started ... unet_missing=0"

# 2) 학습된 체크포인트로 λ 스윕(생성+채점). CKPT를 최신/여러 스텝으로.
CKPT=$(ls -t artifacts/m3/train_out/inha_action_diffusion_11M_m3/checkpoints/*.ckpt | head -1)
bash scripts/m3/sweep_and_score.sh "$CKPT" \
  "$PWD/artifacts/m3/partB" "$PWD/results/m3/partB" trained 10 15 25 40

# 3) 요약 + 모션 (Part A와 동일 비교)
$CONDA run -n wm python scripts/m3/summarize.py results/m3/partA results/m3/partB
#   판정: Part B의 λ=1.0이 0.706보다 나은가 / λ>1이 개선되는가 / motion이 GT(4.8)에 접근하는가
```

> 판정 기준(재확인): static의 **DINO 0.123 / Video 0.091 / TOTAL 0.560**을 넘으면 branch B(1.1B) 정당화가 강해지고, 못 넘으면 anti-drift 설계로.
> 단 Part A가 이미 보였듯 1차 성공 신호는 TOTAL보다 **motion→GT 접근**이 더 예민하다.

---

## 7. 한계·주의 (엄밀성)

- **홀드아웃은 train 부분집합**이라 baseline과 동일 조건의 "로컬 프록시"다(baseline도 train에 학습·train 홀드아웃에 채점). 리더보드 절대 스케일 미보정(하루 3회 제출로 역산 권장).
  → 방향성 주의: baseline은 이 데이터에 **in-distribution 이점**을 받는데도 static에 진다. 즉 결론("생성이 static에 진다")은 이 편향에 대해 **보수적**이며, 진짜 eval에선 격차가 더 벌어질 수 있다.
- **guidance_rescale=0.7 고정의 방향성**: rescale은 고-λ의 분산 과증폭을 conditional std로 되당겨 **완화**하는 쪽으로 작동한다. 즉 rescale=0이었다면 고-λ 악화가 **더 컸을** 개연성 → "λ↑ 악화" 결론은 이 설정에 대해 **robust(보수적)**이지, rescale이 만든 인공물이 아니다.
- **순수 액션 CFG 아님**: uncond 분기가 텍스트·이미지도 null → Part A/B 모두 "액션만" guidance가 아니다. 순수화(uc에 act=None만)는 소규모 코드 수정으로 분리 가능한 후속 실험(004 §5.5). 다만 GT가 정지인 이상 순수화가 방향을 뒤집을 것으로 기대하긴 어렵다.
- **motion 지표는 픽셀 기반 프록시**(mean |Δframe|)로, DINO/Video의 의미론적 거리와 다르지만 방향(움직임 크기)은 잘 포착한다. M0의 프레임간 변화·프레임별 DINO 상승(드리프트) 진단과 정합. seed=0 단일이라 λ 비교는 paired(정답)이나 절대 수치의 seed 민감도는 미측정.
- **Part B 미완**은 코드 결함이 아니라 **외부(드라이버) 요인**이다(증거: `results/m3/partB/driver_block_evidence.txt`; §4.2). partB 로그의 마지막 `종료됨`은 내 수동 종료다. 재현 절차(§6)의 **생성·채점 스캐폴드는 Part A에서 실측 검증됐으나**, warm-start **학습 경로(§6 step1)는 CUDA 블록으로 아직 1스텝도 실행되지 못했다**(모델빌드 직전까지만 검증). warm-start 키 매칭(`unet_missing=0`)은 **CPU로 사전 검증됨**(§4.1, `verify_warmstart.py`: 1107↔1107 완전 매칭)이나, **실제 학습 스텝(옵티마이저·CUDA)은 미실행**이므로 복구 후 첫 실행에서 warm-start 로그와 loss 하강을 확인.
- 기존 `tac_*` 체크포인트·`_tac` config(문서에 없는 선행 실험)는 독립성 규칙상 **일절 건드리지 않았다**. M3 산출물은 별도 네임스페이스(`artifacts/m3/`, `results/m3/`, `scripts/m3/`).

---

## 8. 커밋·상태

- 커밋(로컬, push는 아침에 003 방법으로): `eb3944c feat(M3): Part A …` (스캐폴드+Part A 결과+모션분석), 실행 로그 보존 커밋.
- 백그라운드: **CUDA 복구 감시 폴러 가동**(10분 간격). 복구 감지 시 Part B(§6) 실행.
- 미완: Part B 학습/스윕(환경 복구 대기).
