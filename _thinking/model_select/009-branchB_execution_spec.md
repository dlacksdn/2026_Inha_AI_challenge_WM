# 009 — branch B 실행 스펙 (다음 세션 인수인계용)

> 작성일: 2026-07-25 | 목적: **다음 세션이 이 문서만 보고 곧바로 착수**할 수 있게, 지금까지의 결론을
> "무엇을 어떻게 만들 것인가"라는 실행 명세로 압축한다.
> 근거·실험 과정은 [008](./008-home_experiments.md)(집 실험 6종), [007](./007-status_digest.md)(현황·용어사전),
> [005](./005-5090_env_and_m1.md)(M1 예산), [006](./006-m3_action_cfg_pilot.md)(M3 Part A)에 있다.
> 환경 복원은 [env_file/001](../env_file/001-environment_setup.md), 장비는 [rule/002](../rule/002-hardware.md).

---

## 1. 결정 사항 한 장 요약

```
[버린다]  주최 baseline 모델(11M UNet + 제공 가중치)
            근거: static(0.560)보다 나쁘고(0.711), 재학습해도 0.678로 못 넘었다 (M0, M3 Part B)
            ⇒ 11M 노선 종결

[쓴다]    대회 baseline의 "코드 골격"(challenge_kit의 lvdm: VAE + 3D UNet + 액션 주입 배선)
            + DynamiCrafter 1.1B 폭
            + DC 사전학습 가중치 99.89% 로드
            + 액션 분기(스크래치) + action_dropout 0.1

[목표]    로컬 홀드아웃에서 static(TOTAL 0.560 / DINO 0.123 / Video 0.091)을 넘는다
[판정]    TOTAL이 아니라 **DINO(=GT 거리)** 로 본다. motion·첫프레임이탈은 보조 진단일 뿐 (008 §8.5)
[예산]    학습 4일 1GPU / 추론 216샘플 1시간 → NFE ≤ 32~50 (M1 실측)
```

---

## 2. 모델 스펙 — 정확히 무엇을 바꾸는가

`open/baseline/challenge_kit/configs/train/inha_action_diffusion_11M.yaml` 의
`model.params.unet_config.params` 기준. **7개 키만 바꾼다.**

| # | 키 | baseline | **우리 값** | 이유 |
|---|---|---|---|---|
| 1 | `model_channels` | 32 | **320** | DC 1.1B 규격 (M1에서 빌드·타이밍 검증) |
| 2 | `channel_mult` | [1,2,3] | **[1,2,4,4]** | 동일 |
| 3 | `attention_resolutions` | [4,2] | **[4,2,1]** | 동일 |
| 4 | `num_head_channels` | 16 | **64** | 동일 → UNet **1462.6M** |
| 5 | `use_scale_shift_norm` | True | **False** | **핵심.** True면 emb 출력이 2배가 되어 46키(52.5M)가 shape 불일치 → 로드율 96.36%. False로 하면 **99.89%** (008 §3) |
| 6 | `fs_condition` | false | **True** | DC 사전학습의 `fps_embedding`(4키)을 흡수하고, **fs=6**을 주어 6fps 도메인 격차에 대응 (003 [중대-5]) |
| 7 | `action_dropout_prob` | 0.0 | **0.1** | 액션 CFG를 유효화 (Part B에서 λ↑ 악화→개선 반전 확인, 008 §8.4) |

바꾸지 않는 것(이미 맞음): `in_channels 8`, `out_channels 4`, `context_dim 1024`, `image_size [40,64]`,
`temporal_length 16`, `temporal_attention True`, `use_checkpoint True`, `action_conditioned True`, `action_dims 6`.

### 2.1 가중치 로드

- `model.pretrained_checkpoint: open/baseline/checkpoints/backbone.ckpt` (DynamiCrafter_512, ~10GB)
- `only_reload_modules`는 baseline 그대로 두면 VAE·CLIP·embedder·image_proj만 로드된다.
  **UNet도 로드해야 하므로** 이 부분을 확장하거나, `scripts/m3/train_m3.py`의 warm-start 패턴처럼
  모델 빌드 후 `model.diffusion_model.*` 키를 `strict=False`로 덧씌운다(Part B에서 검증된 방식).
- 검증 도구: `scripts/probe_1p1b_weight_fit.py` — meta device로 키·shape를 대조해 로드율을 출력한다.
  **학습 전 반드시 실행해 99.89%가 나오는지 확인**할 것.

### 2.2 스크래치로 남는 0.11% (예상되고 정상)

| 대상 | 크기 | 처리 |
|---|---|---|
| `action_embed.*`, `null_action_emb` (5키) | 0.83M | 스크래치 학습 (DC엔 액션이 없으니 당연) |
| `time_embed.2` (2키) | 1.64M | 스크래치. 액션 조건화가 time-embed 차원의 절반을 쓰는 구조 탓 |

**선택 옵션(미결정)**: 액션 주입을 concat → **가산(add)** 으로 바꾸면 `time_embed`가 1280 그대로여서
**100% 로드**가 된다(Vid2World가 쓰는 방식). 코드 수정이 필요하므로, 우선 현행 concat으로 진행하고
초기 학습이 불안정하면 이 옵션을 시도한다.

---

## 3. 학습 계획

| 항목 | 값 | 근거 |
|---|---|---|
| 데이터 | `open/data/train` 전체 (128 데이터셋). **eval 절대 미사용** | 규칙 |
| 필터링 | 16프레임 미만 제외(0.3%), 10fps 1개·고해상 5개는 전처리로 통일 | 001 데이터분석 |
| 시간 예산 | **4일 = 최종 1회 학습** 기준으로 역산. 첫 1시간에 step/s 실측 후 총 step 확정 | 004 §5.3 |
| 학습 대상 | 1차: **전체 파인튜닝**(96GB면 가능). 32GB면 액션분기+시간축+LoRA 부분 학습 | 004 §5.3 |
| 저장 | **500스텝마다 체크포인트, `save_top_k=-1`** (중간 끊김 대비 — 프로젝트 규칙) | CLAUDE.md |
| 시간 상한 | `max_time`을 wall-clock의 80%로 설정해 자동 종료 | 004 R3 |
| 로그 | `run_logs/` 에 영구 보존(삭제 금지) | 프로젝트 규칙 |

### 3.1 A/B로 확인할 것 (학습 중 또는 짧은 파일럿으로)

1. `fs_condition=True` + `fs=6` vs 끔 → 6fps 격차가 실제로 줄어드나
2. `action_dropout_prob` 0.1 vs 0.0 → CFG 이득이 1.1B에서도 유지되나
3. (필요 시) 액션 주입 concat vs 가산

---

## 4. 추론 구성 (M1 실측 기반)

| 구성 | NFE | 5090 실측 | 6000 추정 | 판정 |
|---|---:|---:|---:|---|
| 50스텝, CFG off | 50 | 35.2분 | 29~32분 | ✅ |
| **25스텝 + CFG** | 50 | 35.4분 | 30~32분 | ✅ |
| 16스텝 + CFG | 32 | 23.9분 | 20~22분 | ✅ 여유 |
| 50스텝 + CFG | 100 | 67.4분 | 56~61분 | ⚠️ 경계 |

- **설계 상한 NFE ≤ 32~50.** 배치는 처리량 이득이 없으므로(M1) 1로 두면 된다.
- CFG는 학습 전제 시 소폭 유익하나 NFE를 2배 먹는다 → **"25스텝+CFG" vs "50스텝 no-CFG"를 모의채점으로 비교**해 결정.

---

## 5. 성공/실패 판정 규칙 (이번에 교정된 부분 — 중요)

```
1차 지표 : DINO (프레임별 GT 거리)   ← 이것만이 유효하다
보조 진단 : motion, 첫프레임 이탈량   ← 크기가 GT에 맞아도 점수는 안 오른다 (008 §8.5)
기준선    : static  DINO 0.123 / Video 0.091 / TOTAL 0.560  (로컬 홀드아웃 n=96, seed=0)
상한      : GT      DINO 0     / Video 0     / TOTAL 0.489
```

- **DINO가 0.123 아래로 내려가면 처음으로 "생성이 정지를 이긴" 것**이다. 그 전까지는 어떤 서사도 믿지 않는다.
- 로컬 TOTAL을 리더보드 추정치로 쓰지 않는다(eval은 Action이 1/3 스케일, 008 §4).
- 참고: eval에서는 Action도 영상 품질에 반응하므로(static 0.429 vs baseline 0.576) **정확한 영상이 40%·60%를 동시에 개선**한다.

---

## 6. 장비별로 지금 할 수 있는 것

| 작업 | 4060 Ti 8GB (집) | 5090 32GB | RTX PRO 6000 96GB |
|---|:--:|:--:|:--:|
| 1.1B config 작성 + `probe_1p1b_weight_fit.py`로 로드율 검증 | ✅ (meta/CPU) | ✅ | ✅ |
| 1.1B 빌드 + 사전학습 실제 로드 검증(missing/unexpected 확인) | ⚠️ CPU로만 | ✅ | ✅ |
| 1.1B 추론(스모크 생성) | ❌ (17GB 필요) | ✅ | ✅ |
| 1.1B 부분 학습(LoRA/액션분기) | ❌ | ✅ | ✅ |
| 1.1B 전체 파인튜닝 4일 | ❌ | ⚠️ 빠듯 | ✅ **최종 학습은 여기서** |
| 액션 기반 적응형 스케일(열린질문) | ✅ | ✅ | ✅ |
| 모의채점·홀드아웃·제출 CSV | ✅ | ✅ | ✅ |

**8GB에서도 할 일이 있다**: config 작성 + 로드율 검증 + (CPU) 실제 state_dict 로드 확인까지 해두면,
GPU가 열리는 즉시 학습만 걸면 된다.

---

## 7. 즉시 착수 순서

```
[S1] (8GB 가능) 1.1B 학습 config 작성
     - configs/train/inha_action_diffusion_1p1b.yaml 신규 (11M yaml 복사 후 §2의 7키 변경)
     - 경로는 저장소 기준 절대경로 주입 방식으로(5090 절대경로 박지 말 것 — 008 §8.6 함정)
[S2] (8GB 가능) 로드율 검증: probe_1p1b_weight_fit.py → 99.89% 확인
[S3] (8GB 가능) CPU에서 실제 로드 검증: 빌드 후 strict=False 로드해 unet_missing이 §2.2의 7키뿐인지 확인
[S4] (GPU 필요) 스모크: 짧은 학습 20스텝 + 생성 4샘플 → 파이프라인 무결성
[S5] (GPU 필요) 파일럿 학습(수천 스텝) → **DINO가 static 0.123 방향으로 내려가는지** 확인
       내려가면 → 본 학습(4일)
       안 내려가면 → 중형 스크래치(branch A) 재검토 또는 anti-drift 설계
[S6] 추론 구성 확정(25스텝+CFG vs 50스텝 no-CFG) + 216샘플 wall-clock 실측
[S7] 제출 + 리더보드 캘리브레이션
```

---

## 8. 반드시 지킬 것 (규칙·함정)

**규칙**
- `submission_kit` 일절 수정 금지. CSV는 `make_submission_csv.py`로만 생성, 후처리 금지.
- eval 데이터는 추론 입력으로만. **학습·데이터 선별에 사용 금지**(회색지대 — 008 §4.4).
- 학습 ≤ 4일, 추론 ≤ 1시간 (재현 검증 기준).
- 체크포인트는 촘촘히, 로그는 보존, 다른 모델의 학습/평가 독립성 유지.

**환경 함정 (이미 다 해결됨, 재현 시 주의)**
1. `pytorch-lightning==1.9.3` → `setuptools<70` 필요
2. `action_extractor.ckpt` 로드에 `omegaconf` 필요
3. mp4 저장은 `imageio-ffmpeg` + `format="FFMPEG"`(libx264, macro_block_size=1)
4. `numpy==1.26.4` 고정 (baseline deps가 2.x로 올리려 함)
5. 5090은 sm_120 → **cu128** 필요
6. CLIP ViT-H(3.9GB) 다운로드 회피: config `version: null` + `HF_HUB_OFFLINE=1`
7. eval config의 `ddim_eta`는 샘플러가 무시(항상 eta=0). λ·rescale은 정상 적용
8. UNet이 `video_utils` 임포트 → PYTHONPATH에 `open/baseline/shared_libs/video_utils`
9. **config 내부 절대경로**를 실행 시 저장소 기준으로 치환할 것(5090/집 공통 — 008 §8.6)
10. 채점 모델 캐시(`HF_HOME`/`TORCH_HOME`)를 빈 폴더로 주면 offline에서 죽는다

---

## 9. 현재 자산 (git에 있는 것 / 없는 것)

| 자산 | 위치 | git |
|---|---|---|
| 모의채점 라이브러리·스크립트 | `src/wm_eval/`, `scripts/` | ✅ |
| 측정 결과(M0/M1/M3/모션스윕/1.1B정합) | `results/` | ✅ |
| 문서 001~009 | `_thinking/` | ✅ |
| **제출용 static CSV** | `artifacts/submission/static_submission.csv` | ❌ (재생성 가능) |
| 홀드아웃·생성영상·체크포인트 | `artifacts/` | ❌ (재생성 가능) |
| 대회 패키지·backbone.ckpt | `open/` | ❌ (머신별 배치) |

재생성 명령은 [008 §9.2](./008-home_experiments.md)와 [env_file/001 §3](../env_file/001-environment_setup.md) 참고.

---

## 10. 미해결 질문 (다음 세션이 답할 것)

| # | 질문 | 어떻게 |
|---|---|---|
| 1 | **1.1B(사전학습 99.89%)가 static DINO 0.123을 넘나** | S5 파일럿 |
| 2 | `fs_condition=True`(fs=6)가 6fps 격차를 줄이나 | A/B |
| 3 | CFG 이득이 1.1B에서도 유지되나 | A/B |
| 4 | 액션 조건으로 표본별 최적 모션량을 예측해 oracle(−2.1%p)에 근접 가능한가 | 8GB 가능 |
| 5 | eval의 DINO/Video 절대값 | 실제 제출 1회 |
| 6 | 중형 스크래치(branch A)는 여전히 대안인가 | S5 결과에 따라 |
