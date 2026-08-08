# 003 — 논문·저장소 링크 모음

> 작성일: 2026-08-08 | 성격: **찾아다니지 않고 한 곳에서 꺼내 쓰는 색인.**
> 출처는 [residual_plan/001](../residual_plan/001-direction_r3d_proxy.md)(문헌) · [002](../residual_plan/002-base_model_pick.md)(베이스 선정) ·
> [003](../residual_plan/003-base_pick_audit.md)(그 검수) · [007](../residual_plan/007-audit_of_audit_and_budget.md).
>
> **등급을 붙였다.**
> `[코드]` 내가 저장소·파일을 직접 열어 확인함 / `[문서]` 우리 문서의 인용을 옮긴 것(원문 재확인 안 함)
>
> ⚠ **§6 을 먼저 보라.** 이 프로젝트에서 인용 오류가 실제로 전파된 적이 있다.

---

## 1. 우리가 실제로 짓는 것 — 베이스 (확정)

002 채택 → 003 검수 통과 → 007 에서 N_S 재결정.

| 논문 | 링크 | 우리에게 무엇 |
|---|---|---|
| **TAU** | https://arxiv.org/abs/2206.12126 | 우리가 쓰는 `model_type='tau'` 본체 |
| **SimVP** | https://arxiv.org/abs/2206.05099 | 그 백본 |
| SimVPv2 | https://arxiv.org/abs/2211.12509 | 개정판 |
| **OpenSTL** | https://arxiv.org/abs/2306.11249 | 저장소 논문 |

```
저장소   https://github.com/chengtan9907/OpenSTL      Apache-2.0     [코드] LICENSE 직접 확인
고정본   eecf8a3078f0a178dbc7b28723da20f94ce36985   (2025-10-22)
위치     third_party/OpenSTL   ← .gitignore 대상. 각 기계에서 직접 clone 한다
가중치   안 받는다. 처음부터 학습한다 (대회 §4.1-1 과 무관해진다)
```

**[코드] 참조 config 의 N_S** — 007 §4-b 의 근거다.

```
bair(로봇 조작) N_S=2 · kitticaltech(주행 실사) N_S=2 · taxibj N_S=2
kinetics(자연 영상) N_S=4 · mmnist(64×64) N_S=4
전 config 공통  alpha = 0.1   ← TAU 자체 손실 L_reg 의 참조값 (005 중-8)
```

---

## 2. 손실 설계 근거 (004 §3)

| 논문 | 링크 | 무엇을 가져오나 | 등급 |
|---|---|---|---|
| **Focal Frequency Loss** | https://arxiv.org/abs/2012.12821 | 004 의 "주파수 손실" | `[코드]` OpenSTL `methods/wast.py` 의 `HighFocalFrequencyLoss` 실재 확인 |
| **GradNorm** | https://arxiv.org/abs/1711.02257 | λ 를 1회 측정해 고정하는 절차 (005 치-7 처방) | `[문서]` |
| **VP2** | https://arxiv.org/abs/2304.13723 | λ 안전 구간 | ⚠ **인용 오류 있음 — §6-1** |
| Gradient Similarity | https://arxiv.org/abs/1812.02224 | PCGrad 배제 맥락 | `[문서]` |
| MTO 재검증 | https://arxiv.org/abs/2209.11379 | 단순 가중합이 다목적 기법을 이긴다 | `[문서]` |
| Unitary Scalarization | https://arxiv.org/abs/2201.04122 | 같은 취지 | `[문서]` |
| **RLIR** | https://arxiv.org/abs/2509.23958 | 모션 가중(\|정답 잔차\|를 픽셀 가중치로) | `[문서]` |
| ControlNet | https://arxiv.org/abs/2302.05543 | zero-init 분기, "깨어나는 데 ~6,000 스텝" | `[문서]` |
| CaiT / LayerScale | https://arxiv.org/abs/2103.17239 | 작은 초기 스케일로 분기를 붙이는 법 | `[문서]` |

> **WaST 는 arXiv 판이 없다.** AAAI 2024, DOI `10.1609/aaai.v38i5.28230`.
> 코드만 OpenSTL 에 들어 있다(`methods/wast.py`, `modules/wast_modules.py`).

---

## 3. 자체 IDM 문헌 사양 (001 §1.2~1.3) — **아직 미확정**

rule/001 §13 때문에 **킷의 action extractor 구조를 볼 수 없다.** 외부 문헌만으로 설계해야 한다.
005 빈칸 10 이 아직 비어 있다 — 아래는 재료이지 사양이 아니다.

| 논문 | 링크 | 가져올 것 |
|---|---|---|
| **Vidar / MIDM** | https://arxiv.org/abs/2507.12898 | 마스크 + L1 희소 → 테스트 정확도 2배 |
| **CoLA-World** | https://arxiv.org/abs/2510.26433 | warm-up 후 동결 |
| **AlignProp** | https://arxiv.org/abs/2310.03739 | 절단 역전파 |
| **EVA** | https://arxiv.org/abs/2603.17808 | "정지 퇴화" 감시 |
| VPT | https://arxiv.org/abs/2206.11795 | IDM 일반 |
| IDM 벤치마크 | https://arxiv.org/abs/1910.02564 | 성능 감각 |
| Nano World Models | https://arxiv.org/abs/2605.23993 | 방향 |
| CAPE | https://arxiv.org/abs/2606.07304 | 방향 |

⚠ 001 §1.2 는 **"전례가 없다"**고 경고했다. 이 조합을 그대로 한 사례가 문헌에 없다는 뜻이다.

---

## 4. 운영·방법론 근거

| 논문 | 링크 | 무엇 |
|---|---|---|
| **Ladder** | https://arxiv.org/abs/1502.04585 | 제출 횟수 제한 하 리더보드 과적합. 우리는 하루 10회 × 13일 |
| Goodhart in RL | https://arxiv.org/abs/2310.09144 | 대리 목표 과최적화 |
| ES-WMV | https://arxiv.org/abs/2112.06074 | 검증 없는 조기 종료 |
| Perception-Distortion | https://arxiv.org/abs/1711.06077 | 흐림 ↔ 지각 품질의 이론적 맞바꿈 |
| FVD content bias | https://arxiv.org/abs/2404.12391 | 영상 지표가 내용에 치우친다 |

---

## 5. 조사했으나 채택 안 함 (참고용)

**결정론 회귀 계열** — VPTR https://arxiv.org/abs/2212.06026 (MIT) ·
RVD https://arxiv.org/abs/2203.09481 · STLight https://arxiv.org/abs/2411.10198 ·
PredFormer https://arxiv.org/abs/2410.04733

**행동 조건부(전부 확산)** — IRASim https://arxiv.org/abs/2406.14540 (Apache-2.0) ·
Interactive World Simulator https://arxiv.org/abs/2603.08546 ·
AV Scene Prediction https://arxiv.org/abs/2606.12987 · MotiF https://arxiv.org/abs/2412.16153 ·
AVDC https://arxiv.org/abs/2310.08576

**지각 손실·평가 지표** — LPIPS https://arxiv.org/abs/1801.03924 ·
DreamSim https://arxiv.org/abs/2306.09344 · CMMD https://arxiv.org/abs/2401.09603 ·
VBench++ https://arxiv.org/abs/2411.13503 · Beyond FVD https://arxiv.org/abs/2410.05203 ·
Fourier Space Losses https://arxiv.org/abs/2106.00783

**다음 조사 1순위** — World Models for Robotic Manipulation: A Survey
https://arxiv.org/abs/2606.00113

---

## 6. ⚠ 확인된 인용 오류 — 실제로 전파된 것들

001 §0 자신이 이렇게 적었다: *"인용 158건 중 **오류 25건**을 잡아냈다(존재하지 않는 표, 다른 행의 값, arXiv 버전 차이)."*
그 뒤에도 새로 발견됐다.

| # | 무엇 | 진상 | 발견 |
|---|---|---|---|
| **1** | **VP2**: *"LPIPS 가중 **10** 이 성공률 58%→10% 로 반토막"* | **거짓.** 원문(001:459-462) Table 1 의 58→10% 붕괴는 **가중 1**(SVG′·RoboDesk)이다. 가중 10 행은 67→35%·80→37% | 005 중-6 → 007 §1 이 원문 재확인 |
| 1-b | 위 오류의 **출처** | 004 가 아니라 **001 자신**이다. 001:462 는 맞게 적고 **001:538 요약줄**에서 뒤집었다. 004 는 요약줄을 승계했다 | 007 §1 |
| 2 | IRASim BibTeX 의 `arXiv:2406.12802` | **저자 오타.** 올바른 것은 `2406.14540` | 002 §8 |
| 3 | "Anchored VAE DiT" 로 논문 검색 | arXiv 제목이 아니라 **논문 내부 모듈명**이다. 논문은 `2606.12987` | 002 §8 |
| 4 | 002 §1.2 *"PredFormer 는 프레임을 하나씩 되먹인다(**코드로 확인**)"* | **거짓.** recurrence-free, 1회 순전파다. 진짜 탈락 사유는 행동 없음 + LICENSE 파일 없음 | 003 중-3 |
| 5 | R(2+1)D `1711.11248` | **arXiv 버전에 따라 표가 다르다.** 인용 시 버전 명시 | 001 §9 |
| 6 | WaST 를 arXiv 로 찾기 | **arXiv 판이 없다.** AAAI 2024 DOI `10.1609/aaai.v38i5.28230` | 002 §8 |

> **⇒ 규칙**: 위 논문들을 **설계 근거로 승격할 때는 해당 절을 원문에서 다시 연다.**
> 특히 §3 의 IDM 5편은 아직 아무도 원문 재확인을 안 했다 — 005 빈칸 10 을 채우는 작업의 절반이 그것이다.

---

## 7. 우리 저장소 밖의 코드 위치 (읽기 전용)

```
third_party/OpenSTL/                        우리 베이스. Apache-2.0. gitignore 대상
/home/rils/challenge/inchallenge_2026/...   branch B(ldwma) 프레임워크. 우리 프로젝트 밖
  └ src/ldwma/datasets/lerobot_so100.py:175-209   전처리 규약의 원본
    ⚠ 참고만 한다. branch C 는 scripts/branchC/loader_c.py 로 독립 구현했다
open/submission_kit/                        ❌ 봉인. CSV 생성 시 1회만 (rule/001 §13)
```
