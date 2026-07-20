# 모델·논문 선정 보고서 — 2026 인하 인공지능 챌린지 (로봇 월드 모델)

> 조사일: 2026-07-19 | 대상: 행동 조건부 월드 모델(action-conditioned world model)
> 전제: 대회 철학상 **월드 모델 계열만** 후보로 인정. 순수 영상 생성 모델은 "백본"으로만 취급.

---

## 0. 결론 먼저

| 순위 | 모델 | 한 줄 이유 |
|---|---|---|
| **1순위** | **Vid2World** (ICLR 2026, Apache-2.0) | **백본이 DynamiCrafter 320×512 / 16프레임 — 대회 베이스라인과 규격이 완전히 일치.** 연속 액션을 프레임 단위로 주입하는 검증된 레시피 보유 |
| 2순위 | **Ctrl-World** (ICLR 2026, MIT) | SVD 기반 + frame-level action chunk 조건화. 실물 로봇 정책 평가로 검증됐으나 추론이 느리고 멀티뷰 전제 |
| 3순위(아이디어 차용) | **OSCAR** (2026.06) | 액션을 **2D 키네마틱 스켈레톤 영상**으로 렌더링해 주입 — Action Component(40%) 공략에 매우 유효한 발상 |
| 참고 | **iVideoGPT** (NeurIPS 2024, MIT) | 빠르고 OXE 사전학습됐지만 해상도 64/256이라 DINO·R3D 지표에서 불리 |

**최종 권고**: Vid2World를 1순위로 채택하되, **논문을 통째로 가져오지 말고 "액션 주입 방식(프레임 단위 MLP + action CFG)"만 이식**하고 백본은 DynamiCrafter 사전학습 가중치로 교체한다. 근거는 5장에 상술한다.

---

## 1. 선정 기준 — 대회 제약에서 도출한 하드 게이트

후보는 아래 6개 관문을 **전부** 통과해야 한다. 하나라도 실패하면 탈락이다.

```text
 [G1] 라이선스     가중치 공개 + MIT/Apache/CC BY/CC BY-NC/NVIDIA Open Model License 등
                   → API 전용 모델, 가중치 비공개 모델 즉시 탈락
 [G2] 월드 모델성   행동(action)을 조건으로 미래를 예측하는 구조일 것 (대회 철학)
 [G3] 액션 공간     6차원 연속 관절 명령을 프레임 단위로 받을 수 있을 것
 [G4] 출력 규격     320×512, 정확히 16프레임 생성 가능 (또는 무리 없이 개조 가능)
 [G5] 추론 예산     216샘플 / 1시간 = 배치 처리 기준 충족
 [G6] 학습 예산     RTX PRO 6000 96GB × 4일 이내 / 개발은 RTX 5090 32GB에서 가능
```

> **[G5]에 대한 중요한 완화 조건**: "샘플당 16.7초"는 순차 처리 기준이다. 실제로는 96GB VRAM에서
> **배치 8~16개를 동시 생성**할 수 있으므로 유효 예산은 샘플당 약 2~3분 수준으로 늘어난다.
> 다만 개발용 5090(32GB)에서는 배치가 작아지므로, VRAM이 큰 모델은 개발 속도에서 손해를 본다.

---

## 2. 후보 지도 — 계열별 분류

```text
                     행동 조건부 월드 모델 (Action-Conditioned World Model)
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
  [A] 픽셀 생성형              [B] 잠재 예측형               [C] 게임/범용 인터랙티브
  (영상을 직접 생성)          (픽셀 디코더 없음)            (이산 조작 입력 전제)
        │                             │                             │
  ├ Vid2World  ★1순위          ├ V-JEPA 2-AC                 ├ Genie 2/3   ✗가중치 비공개
  ├ Ctrl-World ★2순위          │   → 영상 생성 불가            ├ Oasis / Matrix-Game
  ├ OSCAR      ★아이디어        │   → 보조 지표로만 활용        ├ DIAMOND
  ├ iVideoGPT  (저해상도)       └ DreamerV3                    └ GameNGen
  ├ EnerVerse-AC                    → 저해상도 재구성              → 액션이 키보드/이산
  ├ Genie Envisioner (GE-Sim)       → 픽셀 지표 부적합             → 6D 연속 액션과 불일치
  └ Cosmos-Predict2.5 (robot/action-cond)  ✗속도 초과

        ─────────────────────────────────────────────────────────
        [D] 백본 후보 (그 자체는 월드 모델이 아님 — 위 방법론의 토대로만 사용)
          DynamiCrafter(베이스라인) · Stable Video Diffusion · Wan 2.2 · LTX-Video · CogVideoX
```

**계열 [B]와 [C]는 구조적으로 이 대회와 맞지 않는다.**
- [B]는 잠재 표현만 예측하고 픽셀을 만들지 않는다. 채점이 생성 영상의 DINO·R3D feature로
  이루어지므로 출품 자체가 불가능하다.
- [C]는 액션이 키보드 입력 같은 **이산 공간**이다. 6차원 연속 관절값을 넣으려면 사실상
  재설계이며, 그럴 바에는 [A]에서 시작하는 편이 낫다.

---

## 3. 핵심 비교표

### 3.1 게이트 통과 현황

| 모델 | G1 라이선스 | G2 월드모델 | G3 연속6D | G4 320×512·16f | G5 추론 | G6 학습 | 종합 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Vid2World** | ✅ Apache-2.0 | ✅ | ✅ | ✅ **네이티브** | ✅ | ⚠️ | **채택** |
| **Ctrl-World** | ✅ MIT | ✅ | ⚠️ DROID 7D | ⚠️ 개조 필요 | ⚠️ 느림 | ⚠️ | 2순위 |
| **OSCAR** | ✅ NVIDIA OML | ✅ | ⚠️ 스켈레톤 변환 | ⚠️ 512×288 | ⚠️ | ❌ 2B | 아이디어만 |
| **iVideoGPT** | ✅ MIT | ✅ | ✅ | ❌ 64/256 | ✅ 빠름 | ✅ | 참고 |
| Cosmos-Predict2.5-2B | ✅ NVIDIA OML | ✅ | ⚠️ | ⚠️ 256p/4fps | ❌ **초과** | ❌ | 탈락 |
| Genie 2 / 3 | ❌ **비공개** | ✅ | — | — | — | — | 탈락 |
| DreamerV3 | ✅ | ✅ | ✅ | ❌ 저해상도 | ✅ | ✅ | 탈락 |
| V-JEPA 2-AC | ✅ | ✅ | ✅ | ❌ **디코더 없음** | — | — | 탈락 |

### 3.2 1·2순위 상세 스펙

| 항목 | **Vid2World** | **Ctrl-World** | 대회 요구 |
|---|---|---|---|
| 논문 | arXiv 2505.14357, **ICLR 2026** | arXiv 2510.10125, **ICLR 2026** | — |
| 소속 | Tsinghua (thuml) | Stanford + Tsinghua (Chelsea Finn) | — |
| 코드 | `github.com/thuml/Vid2World` | `github.com/Robert-gyj/Ctrl-World` | — |
| 가중치 | `thuml/Vid2World-RT1` 외 6종 (2025-11 공개) | `yjguo/Ctrl-World` (~8GB) | — |
| 라이선스 | **Apache-2.0** | **MIT** | 허용 ✅ |
| 백본 | **DynamiCrafter 1.1B U-Net** | Stable Video Diffusion | 베이스라인=DynamiCrafter |
| 해상도 | **320×512** | 미명시(SVD 계열) | **320×512** |
| 프레임 | **16** (CS:GO·RECON 실험 모두 16) | 1초 action chunk 단위 | **16** |
| 액션 주입 | **경량 MLP → 프레임별 잠재에 가산** | frame-level action chunk | 프레임별 6D |
| 액션 타입 | **연속 벡터** (RECON 3D, RT-1 연속) | DROID 7-DoF + 그리퍼 | 연속 6D |
| 액션 CFG | ✅ 학습 중 드롭아웃 + 추론 시 guidance | 미확인 | 강도 조절 가능 → 유리 |
| 로봇 사전학습 | RT-1 (Open X-Embodiment) | DROID 95k 궤적 | SO-100 |
| 학습 비용(원논문) | 100k step ≈ **7일 × 4×A100** | 8×A100 노드 1~2대 | **4일 × 1 GPU** ⚠️ |
| 추론 속도 | 미보고 (자기회귀) | **~10s/step (A100), ~5s (H100)** ⚠️ | 배치 기준 충족 필요 |

---

## 4. 후보별 상세 평가

### 4.1 ★ Vid2World — 1순위

**무엇인가**: 사전학습된 영상 확산 모델을 인터랙티브 월드 모델로 개조하는 일반 방법론.
두 가지 장치를 도입한다.
1. **Video diffusion causalization** — 양방향 어텐션을 인과적 구조로 바꿔 자기회귀 생성 가능화
2. **Causal action guidance** — 경량 MLP가 액션을 임베딩해 해당 시점 잠재에 더하고,
   학습 중 확률 `p`로 액션을 드롭아웃해 추론 시 classifier-free guidance를 적용
   `ε_guided = (1+λ)·ε_cond − λ·ε_uncond`

**왜 이 대회에 최적인가** — 규격이 우연이 아닐 정도로 정확히 겹친다.

| | Vid2World | 대회 베이스라인 |
|---|---|---|
| 백본 | DynamiCrafter | DynamiCrafter |
| 해상도 | 320×512 | 320×512 |
| 프레임 | 16 | 16 |
| 조건 | 첫 프레임 + 프레임별 액션 | 첫 프레임 + 프레임별 액션 |

즉 **베이스라인이 하려던 것의 논문판 정답**에 가깝다. 게다가 베이스라인의 UNet은 11M에
불과한 반면 Vid2World는 **1.1B 사전학습 UNet**을 쓴다 — 약 100배 용량 차이이며, 이것이
DINO·R3D 지표에서 가장 큰 격차를 만들 지점이다.

**액션 CFG가 특히 중요한 이유**: 대회 점수의 40%는 고정된 action_extractor가 생성 영상에서
행동을 얼마나 잘 읽어내는가로 결정된다. 액션 guidance 스케일 `λ`는 **"액션 반영 강도"를
추론 시점에 조절하는 손잡이**다. λ를 키우면 행동이 과장되게 표현돼 Action MAE가 내려가고,
너무 키우면 영상이 붕괴해 DINO·R3D가 나빠진다. **로컬 모의채점으로 λ만 스윕해도
0.3/0.3/0.4 가중합의 최적점을 찾을 수 있다** — 학습 없이 얻는 점수다.

**주의점 (반드시 반영할 것)**:
- 원논문의 **causalization은 이 대회에 불필요하다.** 인과 구조는 "액션을 하나씩 받아가며
  실시간 롤아웃"하기 위한 장치인데, 본 대회는 **16개 액션이 처음부터 전부 주어진다.**
  자기회귀는 16번의 순차 디퓨전을 요구해 추론이 느려질 뿐이다.
  → **full-sequence 생성 + 프레임별 액션 주입**만 취하는 것이 정답.
- 학습 비용 경고: 원논문 RT-1 post-training은 4×A100으로 7일(≈28 A100-day)이었다.
  본 대회 예산(1 GPU × 4일)은 이에 못 미친다. 논문의 ablation이 30k step으로 수행된 점을
  근거로 **step 수 축소 + 부분 파인튜닝(시간 축 레이어·액션 MLP 우선)** 전략이 필요하다.

### 4.2 Ctrl-World — 2순위 (보험)

Stanford Chelsea Finn 그룹. SVD 기반이며 **frame-level action chunk 조건화**로 미세한 행동
제어를 구현하고, pose-conditioned memory retrieval로 장기 일관성을 유지한다. π0.5-DROID
정책의 신규 지시 성공률을 38.7% → 83.4%로 끌어올린 실증 결과가 강력하다.

**감점 요인**: (1) 추론 ~10s/step(A100)로 느리다, (2) 멀티뷰(손목 카메라 포함) 생성을 전제로
설계됐는데 본 대회는 단일 카메라다, (3) DROID 7-DoF 액션 공간이라 6D 매핑 작업이 필요하다,
(4) 학습에 8×A100 노드 1~2대를 상정한다.

→ **Vid2World 이식이 실패할 경우의 대안**으로만 확보한다.

### 4.3 OSCAR — 아이디어 차용 (모델 자체는 부적합)

2026년 6월 공개. Cosmos-Predict2.5-2B를 파인튜닝하며, **액션을 URDF 기반 2D 키네마틱
스켈레톤으로 렌더링해 RGB 잠재에 더하는** 방식을 쓴다. 텍스처 없는 픽셀 공간 표현이라
로봇 외형에 과적합되지 않고 이종 로봇 간 일반화가 된다.

**차용 가치**: SO-100의 URDF는 공개되어 있다. 6차원 관절값을 정기구학으로 풀어 팔 골격을
2D에 투영한 뒤 보조 조건으로 넣으면, 모델이 "관절값 → 팔의 픽셀 위치"를 처음부터 학습할
필요가 없어진다. **Action Component(40%) 공략에 직결되는 강력한 귀납 편향**이다.

**모델 자체를 못 쓰는 이유**: 백본이 2B Cosmos라 5090(32GB)에서 개발이 어렵고, 추론
2.2 FPS(GH200)에 해상도도 512×288로 어긋난다. 학습 데이터에 SO-100/LeRobot도 없다.

### 4.4 iVideoGPT — 참고

MIT 라이선스, Open X-Embodiment 사전학습, 토큰 자기회귀 방식이라 확산 모델보다 빠르다
(tokenizer 114~310M + transformer 138~436M). 그러나 **해상도가 64×64 또는 256×256**이다.
320×512로 업스케일하면 DINOv2·R3D feature 거리에서 구조적으로 불리하다.

→ 빠른 프로토타이핑이나 앙상블 보조로만 고려.

### 4.5 탈락 후보와 사유 (시간 낭비 방지)

| 모델 | 탈락 사유 |
|---|---|
| **Cosmos-Predict2.5-2B** | 추론 VRAM 32.54GB로 **5090(32GB) 초과**. H100에서 5초 720p에 229초 — 속도 예산 대폭 초과. robot/action-cond 변형은 256p·4fps로 규격 불일치 |
| **Genie 2 / Genie 3** | DeepMind, **가중치 비공개** → G1 실패 |
| **DreamerV3** | RL용 잠재 월드 모델. 재구성 디코더가 저해상도·저용량이라 320×512 픽셀 feature 지표에서 경쟁 불가. *"검증된 모델"이지만 검증된 영역이 다르다* |
| **V-JEPA 2 / V-JEPA 2-AC** | 잠재 예측 전용, **픽셀 디코더 없음** → mp4 생성 불가. 단 생성 결과의 물리적 타당성 자체 검증 지표로는 활용 가능 |
| **Oasis / Matrix-Game / DIAMOND / GameNGen** | 액션이 키보드·이산 입력. 6D 연속 관절 공간과 근본적으로 불일치 |
| **Wan 2.2 / LTX-Video / CogVideoX / HunyuanVideo** | **월드 모델이 아님** (대회 철학 위반). 백본으로만 의미. 참고로 Wan2.2-TI2V-5B는 768×512·25프레임에 FP16 27.4GB로 5090 추론은 가능하나 파인튜닝은 LoRA 필수 |

---

## 5. 최종 추천 — 무엇을 어떻게 쓸 것인가

### 5.1 채택안: "Vid2World 레시피 + DynamiCrafter 사전학습 백본"

논문을 통째로 이식하는 것이 아니라, **필요한 부품만 골라 베이스라인 코드에 이식**한다.

```text
  대회 베이스라인 (challenge_kit)          채택할 변경
  ────────────────────────────────       ─────────────────────────────────────
  UNet 11M, 스크래치 학습          →     DynamiCrafter 1.1B 사전학습 가중치로 교체
  액션 조건화 (기본 구현)           →     Vid2World식 프레임별 액션 MLP 주입
  액션 CFG 없음                    →     학습 중 액션 드롭아웃 + 추론 시 guidance λ 스윕  ★
  causalization                    →     도입하지 않음 (16개 액션이 모두 주어지므로 불필요)
  보조 조건 없음                   →     (실험) OSCAR식 SO-100 URDF 스켈레톤 렌더링 추가
  검증 = 제출                      →     로컬 모의채점 루프 (채점 모델 3종 전부 공개돼 있음)
```

**이 조합을 추천하는 이유 4가지**
1. **규격 일치** — 백본·해상도·프레임 수가 이미 맞아떨어져 개조 위험이 가장 작다.
2. **점수 구조 직격** — 액션 CFG의 λ는 Action(40%)와 DINO·R3D(60%)의 트레이드오프를
   추론 시점에 조절하는 유일한 손잡이다. 재학습 없이 점수를 얻는 경로다.
3. **예산 적합** — 1.1B는 96GB에서 전체 파인튜닝, 32GB에서도 LoRA·부분 파인튜닝이 가능한
   거의 유일한 "월드 모델 계열" 크기다. 2B Cosmos급은 개발 GPU에서 막힌다.
4. **라이선스 안전** — Apache-2.0(Vid2World) + 연구용 DynamiCrafter. 대회 규칙은
   비상업 라이선스(CC BY-NC 등)도 명시적으로 허용하며, **베이스라인 자체가 DynamiCrafter
   기반**이므로 주최측이 이미 승인한 계보다.

### 5.2 단계별 실행 순서

| 단계 | 내용 | 산출물 | 우선순위 |
|---|---|---|---|
| 0 | **로컬 모의채점 루프 구축** (train 홀드아웃으로 0.3·0.3·0.4 산식 재현) | 점수 추정기 | **최우선** |
| 1 | 베이스라인 재현 → 점수 하한선 확보 | 기준점 | 높음 |
| 2 | DynamiCrafter 1.1B 가중치 로드 + 액션 MLP 이식 | v1 모델 | 높음 |
| 3 | 액션 드롭아웃 학습 → **guidance λ 스윕** | 무학습 점수 향상 | 높음 |
| 4 | 데이터 필터링 (16프레임 미만 제외, 10fps·고해상도 예외 처리) | 정제 데이터 | 중간 |
| 5 | (실험) URDF 스켈레톤 보조 조건 | v2 모델 | 중간 |
| 6 | 샘플링 스텝 축소·배치 최적화로 추론 1시간 예산 검증 | 제출 파이프라인 | 필수 |

### 5.3 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| **학습 예산 초과** (원논문 28 A100-day vs 예산 ~4일 1GPU) | 높음 | step 수 축소(30k 수준), 시간축 레이어+액션 MLP만 우선 학습, LoRA 병용 |
| Vid2World 코드 성숙도 (커밋 3개, 공개 직후) | 중간 | 논문 방법론만 참조해 베이스라인 코드에 직접 구현 — 레포 의존 최소화 |
| 추론 1시간 초과 | **치명적** (재현 검증 실패) | 초기부터 배치 생성·스텝 수 측정, 필요 시 few-step 샘플러 |
| DynamiCrafter 라이선스 해석 | 낮음 | 비상업 허용 라이선스는 규칙상 명시적 허용, 베이스라인이 동일 계보 |
| λ 과최적화로 Public↔Private 괴리 | 중간 | Public 30%에 과적합 금지, 로컬 홀드아웃 기준으로 λ 결정 |

---

## 6. 참고문헌

| # | 제목 | 링크 |
|---|---|---|
| 1 | Vid2World: Crafting Video Diffusion Models to Interactive World Models (ICLR 2026) | [arXiv:2505.14357](https://arxiv.org/abs/2505.14357) · [GitHub](https://github.com/thuml/Vid2World) · [HF](https://huggingface.co/thuml) |
| 2 | Ctrl-World: A Controllable Generative World Model for Robot Manipulation (ICLR 2026) | [arXiv:2510.10125](https://arxiv.org/abs/2510.10125) · [GitHub](https://github.com/Robert-gyj/Ctrl-World) |
| 3 | OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics (2026.06) | [arXiv:2606.04463](https://arxiv.org/html/2606.04463) |
| 4 | iVideoGPT: Interactive VideoGPTs are Scalable World Models (NeurIPS 2024) | [arXiv:2405.15223](https://arxiv.org/abs/2405.15223) · [GitHub](https://github.com/thuml/iVideoGPT) |
| 5 | DynamiCrafter (베이스라인 백본) | [HF Doubiiu/DynamiCrafter_512](https://huggingface.co/Doubiiu/DynamiCrafter_512) |
| 6 | Cosmos-Predict2.5 (NVIDIA) | [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5) · [HF](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) |
| 7 | World Models for Robotic Manipulation: A Survey (2026.06) | [arXiv:2606.00113](https://arxiv.org/pdf/2606.00113) |
| 8 | EnerVerse-AC: Envisioning Embodied Environments with Action Condition | [arXiv:2505.09723](https://arxiv.org/abs/2505.09723) |
| 9 | Genie Envisioner: Unified World Foundation Platform for Robotic Manipulation | [arXiv:2508.05635](https://arxiv.org/html/2508.05635v1) |
| 10 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | [arXiv:2506.09985](https://arxiv.org/html/2506.09985v1) |
