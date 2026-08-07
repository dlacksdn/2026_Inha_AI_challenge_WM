# 002 — 베이스를 무엇으로 할 것인가

> 작성일: 2026-08-07 | 선행: [001](./001-direction_r3d_proxy.md)(손실·구조 문헌), [019](../model_select/019-handoff.md)(좌표)
> 성격: **"우리는 어떤 논문을 베이스로 구현하는가"** 하나만 답하는 문서.
> 방법: 조사원 6명이 후보를 발굴하고, 검증관 6명이 **원문·저장소·LICENSE 를 독립적으로 다시 열어** 재확인했다.
> 후보 50여 개를 조건 8개로 판정했고, 판정이 뒤집힌 것이 3건이다.

---

## 결론 한 줄

> **베이스는 OpenSTL 의 SimVP 계열(`model_type='tau'`)로 한다.
> 이유는 "결정론 회귀 + 16프레임 1회 출력 + Apache-2.0 + 학습코드 완비"를 동시에 만족하는
> 유일한 저장소이고, 빠진 것(행동 주입·잔차)이 **빼는 개조가 아니라 더하는 개조** 30줄이기 때문이다.**

행동 주입의 **설계도**는 IRASim 의 Frame-Ada 에서, **잔차 앵커링 식**은 arXiv 2606.12987 에서 빌린다.
둘 다 코드가 아니라 논문의 식만 가져온다.

---

## ⚠ 먼저 — 이 조사의 가장 중요한 두 문장

```
① 필수조건 5개를 전부 만족하는 논문은 없다.  하나도.
   조사원 3명이 서로 다른 검색어로 독립적으로 훑고 같은 결론에 도달했다.
   이유는 우연이 아니라 구조다:
     · 결정론 회귀 계보(SimVP·TAU·PredFormer·VPTR·WaST)는 "영상예측 벤치마크" 출신이라 **행동이 없다**.
     · 행동 조건부 계보(IRASim·IWS·Ctrl-World·Cosmos·DriftWorld)는 2024년 이후 **전부 확산 아니면 토큰 자기회귀**다.
   ⇒ 기성품은 없다. 우리는 **한쪽을 골라 반대쪽을 붙여야** 한다.
   ⇒ "결정론 쪽에 행동을 더한다"가 "확산 쪽에서 확산을 뺀다"보다 훨씬 싸다.  그래서 SimVP 다.

② ⭐ 우리가 하려는 것을 **거의 그대로 만든 논문이 있고, 그 논문은 실패를 측정해 놨다.**
   arXiv 2606.12987 (Stanford, 2026-06): 잠재공간 DiT + adaLN-Zero + 프레임별 행동 임베딩
   + 잔차 앵커링 ẑ_{t+k} = z_t + Δ_k + 16스텝 한 번에.  ← 우리 설계와 부품이 같다.
   그 논문이 같은 백본을 확산 모드와 결정론 회귀 모드로 각각 돌려 비교했다:

     조향 입력을 바꿔가며 출력이 따라오는지 잰 스피어만 상관 ρ
        확산 모드          ρ = 0.81   (행동을 따른다)
        결정론 회귀 모드    ρ = −0.18   (행동을 **무시한다**)

   그리고 결정론 회귀의 출력 그림은 "차량도 차선도 알아볼 수 없는 균일한 회색 블러"였다.
   논문 스스로 이것을 "손실 표면 문제가 아니라 **구조적 문제**"라고 진단했다.

   ⇒ 이건 우리가 이미 세 번째로 보는 같은 얼굴이다.
      (1) 리더보드에서 정지영상 0.30325 가 우리 확산모델을 이겼다
      (2) 001 §1.2 — EVA 의 퇴화 모드가 "static behaviors"
      (3) 지금 — 결정론 회귀가 행동을 무시하고 조건부 평균으로 무너졌다
   ⇒ **베이스를 고르는 것보다 이 붕괴를 감시하는 장치를 먼저 붙이는 것이 중요하다.**
```

---

## 1. 후보 비교표

조건: **C1** 결정론 회귀 / **C2** 행동 조건부 / **C3** 1장→16프레임 병렬 / **C4** 잔차 /
**C5** 라이선스 / **C6** 공개 구현 / **C7** 규모 / **C8** 지표

### 1.1 살아남은 것 — 4개

| 후보 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | 종합 |
|---|---|---|---|---|---|---|---|---|
| ⭐ **SimVP / SimVPv2 / TAU / WaST** (OpenSTL) | ✅ MSE | ❌ **없음**(30줄로 추가) | ✅ T_in=T_out 일 때 | ✅ 스킵 있음, 1줄 | ✅ Apache-2.0 | ✅ 별1136, 학습코드 완비 | 11~45M, V100 1장 | **1위** |
| VPTR-NAR (2212.06026) | ⚠ MSE+GDL+NCE | ❌ 없음 | ✅ **네이티브** | ⚠ 잠재 | ✅ MIT | ⚠ 별102, 25/01 정체 | 64×64만 검증 | 3위 |
| STLight (2411.10198) | ✅ MSE | ❌ 없음 | ✅ | ? | ✅ Apache-2.0 | ❌ **저장소가 빈 포크** | 미확인 | 탈락(C6) |
| Anchored VAE DiT (2606.12987) | ✅ direct 변형 | ✅ **프레임별** | ✅ | ✅ **교과서적** | ❌ **라이선스 없음** | ⚠ 별2, 수업과제 | 5.4M(너무 작다) | **설계도만 차용** |

### 1.2 아깝게 떨어진 것 — 이유를 알아둘 값어치가 있다

| 후보 | 떨어진 조건 | 왜 |
|---|---|---|
| **IRASim** (2406.14540) | **C1** | 잠재 DDPM. ε 예측 + PNDM 50스텝. **나머지는 전부 우리 규격** — 288×512, 1장+행동→16프레임 1회, Apache-2.0, 학습코드 완비. **2위이자 행동주입 설계도 제공자** |
| **Interactive World Simulator** (2603.08546) | **C1·C3·C4** | ⚠ **017 의 "가중 회귀 L2(확산 아님)" 기록이 오독이었다.** consistency model 기반 확산이고 명시적 자기회귀다(§I "we autoregressively shift a fixed-length context window") |
| Nano World Models (2605.23993) | C1 | 확산 + flow-matching 이 1급 목적함수. 저장소에 `full_sequence` 병렬 모드가 있으나 그것도 joint **denoising** |
| RVD (2203.09481) | C1·C3 | 이름 그대로 Residual Video **Diffusion**. 재귀 생성. ⇒ **잔차 방식만 참고** |
| MotiF (2412.16153) | C1·C2·C6 | v-예측 확산, 텍스트 조건부, **코드·가중치 공개 없음**(Meta 내부 라이선스 데이터로 학습 → 앞으로도 공개 가능성 없음) |
| PredFormer / PredRNN / PhyDNet / MIM / E3D-LSTM / MAU / SwinLSTM | C3 | 전부 프레임을 하나씩 뽑아 되먹인다(코드로 확인) |
| RIFE / IFRNet / EMA-VFI / AMT / FILM / XVFI | C3 | 보간은 **두 끝점 사이**를 채운다. 1장→미래가 원리적으로 불가 |
| DMVFN | **C1** | ⚠ 확산은 아닌데 **결정론도 아니다.** 추론 경로에 `torch.bernoulli` 확률 표본추출이 있다 |
| SVD / Ctrl-World / LTX-Video / Cosmos / DynamiCrafter | C1 + **C5** | 라이선스 지뢰 → §6 |

> ⚠ **여기 있는 어떤 수치도 우리 Score 로 환산하지 마라.** 원문 확인한 후보 전부가
> PSNR/SSIM/MSE/FVD/LPIPS 로 보고한다. 우리 지표(DINO·R3D·Action 거리)로 어블레이션한 논문은 **0편**이다.

---

## 2. 1위 — OpenSTL SimVP 계열의 구조

저장소: **https://github.com/chengtan9907/OpenSTL** (Apache-2.0, LICENSE 원문 확인, 지역 배제 조항 0건)
별 1136 · 마지막 푸시 2026-03-01 · 학습/평가/설정 전부 포함 · **가중치를 배포하지 않는다**(우리가 처음부터 학습)

### 2.1 전체 골격 — 파일 하나(`openstl/models/simvp_model.py`, 247줄)가 전부다

```
입력 (B, T, 3, H, W)
  │
  ├─ Encoder   : ConvSC(Conv2d+GroupNorm+LeakyReLU) × N_S
  │              프레임을 (B*T, 3, H, W) 로 펴서 프레임별로 처리
  │              0번 블록 출력 enc1 을 스킵으로 따로 보관     ← ⭐ 잔차 자리
  │
  ├─ Translator: (B, T, C', H', W') → reshape (B, T*C', H', W')  ← ⭐ 시간축이 채널로 접힌다
  │              MetaBlock × N_T.  마지막 블록이 다시 T*C' 로 되돌린다 → T 보존
  │              model_type 문자열 하나로 블록 교체: 'incepu'|'gsta'|'tau'|'van'|...
  │
  └─ Decoder   : ConvSC 업샘플 × N_S, 마지막 직전에 `hid + enc1`, 끝에 readout=Conv2d(C',3,1)

출력 (B, T, 3, H, W)      ← forward() 안에 for 루프가 하나도 없다
손실 : nn.MSELoss()
```

**세 모델의 차이는 Translator 블록 하나뿐이다.**

| 이름 | Translator | 우리에게 주는 것 |
|---|---|---|
| SimVP v1 | MidIncepNet (Inception 커널 3/5/7/11) | 기준선 |
| SimVPv2 | GASubBlock (depthwise conv 커널 21 + MLP) | 큰 수용영역 |
| **TAU** | TAUSubBlock — SA(공간) ⊗ DA(시간 채널 게이트) | ⭐ **DA 게이트가 행동 주입의 자연스러운 자리** |
| WaST | 3D 웨이블릿 + `HighFocalFrequencyLoss` | ⭐ **주파수 손실이 이미 구현돼 있다** (001 §3.3) |

⇒ 넷을 **같은 학습 스크립트로 A/B 할 수 있다.** 13일 일정에서 이건 큰 이점이다.

### 2.2 TAU 모듈 (원문 Eq.2 그대로)

```
SA = Conv_1x1( DW-Dilated-Conv( DW-Conv(H) ) )     ← 프레임 안의 정적 주의
DA = FC( AvgPool(H) )                              ← 프레임 사이의 동적 주의 (채널 게이트)
H' = (SA ⊗ DA) ⊙ H                                 ⊗ 크로네커, ⊙ 아다마르
```
`H ∈ R^{B×(T·C')×H×W}` 이므로 **DA 는 시간축이 접힌 채널에 대한 게이트**다.
`FC(AvgPool(H))` 를 `FC([AvgPool(H); 행동임베딩])` 으로 바꾸면 **프레임별 행동이 곧바로 시간별 게이트가 된다.**

### 2.3 TAU 손실 (원문 Eq.3~6 그대로)

```
ΔŶ_i = Ŷ_{i+1} − Ŷ_i ,   ΔY_i = Y_{i+1} − Y_i
σ(·) : 전역 softmax, 온도 τ = 0.1
L_reg = KL( σ(ΔŶ) ‖ σ(ΔY) )
L     = Σ ‖Ŷ − Y‖² + α · L_reg
```
[측정] 원문 Table 6 (Moving MNIST 10→10, MSE 낮을수록 좋음): 전체 19.8 / DDR 항 제거 21.1 / DA 제거 22.4 / SA 제거 23.2.
α 값은 **원문에 없다**(원문 미확인).

> **[추측] 우리에게 왜 흥미로운가**: `L_reg` 는 **프레임 사이의 차분**을 감독하는 항이다.
> 배경을 잘 맞춰 MSE 를 낮추면서 팔의 움직임만 뭉개는 실패(=우리의 정지영상 함정)를 억제할 여지가 있다.
> ⚠ **이건 가설일 뿐이고 TAU 의 수치가 우리 Score 로 오는 게 아니다.** 우리 데이터에서 α 를 켜고 꺼서 직접 재라.

### 2.4 규모 [측정, 출처는 저장소 `docs/en/model_zoos/video_benchmarks.md` — 논문 표 아님]

| 설정 | 파라미터 | FLOPs | GPU |
|---|---|---|---|
| SimVP+gSTA, KittiCaltech 128×160 | 15.6M | 96.3G | V100 1장 |
| SimVP+gSTA, Human3.6M 256×256 (4→4) | 11.3M | 74.6G | 단일 GPU bs16 |
| TAU, KittiCaltech 128×160 | 44.7M | 80.0G | V100 1장 |
| TAU, Human3.6M 256×256 (4→4) | 37.6M | 182.0G | 단일 GPU bs16 |

[측정] SimVP 원문 4절 각주 "a single NVIDIA-V100", Table 4 샘플당 412MB, 총 학습 ≈2일(2k epoch), Caltech 은 "finished within 4h".
[측정] TAU 원문 4.1절: mini-batch 16, AdamW, lr 0.01, weight decay 0.05.

⇒ **우리 예산(5090 32GB, 학습 ≤ 4일)에 넉넉히 들어간다.** 019 가 적은 "40M~176M / H200 18시간" 감각과 일치한다.

---

## 3. ⭐ 우리 세팅으로 옮길 때 바꿔야 할 것

**8개다. 이 중 코드를 새로 쓰는 것은 2개뿐이다.**

### ① 입력 — 사진 1장을 16번 복제한다 (코드 0줄, 그러나 필수)

이 계열에서 **가장 밟기 쉬운 지뢰**다. `openstl/methods/simvp.py` 를 직접 읽어 확인했다:

```python
if aft_seq_length == pre_seq_length:
    pred_y = self.model(batch_x)          # ← 1회 순전파. 루프 없음.
elif aft_seq_length > pre_seq_length:
    for _ in range(d):                    # ← ⚠ 조용히 자기회귀로 빠진다
        cur_seq = self.model(cur_seq)
```

`pre=1, aft=16` 으로 그냥 돌리면 **아무 경고 없이 16번 자기회귀**를 돈다(= 019 확정 7 위반).
사진을 16번 복제해 `pre = aft = 16` 으로 맞추면 첫 분기를 타고 **코드 수정 0줄로 1회 순전파**다.

> ⚠ 복제하면 16슬롯이 전부 똑같아진다. 슬롯을 구분해 주는 것은 오직 아래 ②의 행동 임베딩이다.
> **행동 임베딩이 죽으면 16프레임이 전부 같은 그림이 된다** — 그게 정지영상이고, 우리 함정이다.

### ② 행동 주입 — 없다. 만들어야 한다 (새로 쓰는 것 #1, ~30줄)

**설계는 IRASim 의 Frame-Ada 를 따른다.** [측정] IRASim 부록 C.2 식(8)(9):

```
i번째 행동 a^i 를 Linear 한 층으로 **개별** 임베딩  → c_S^i
x^i = x^i + (1+α₁^i) × MHA( γ₁^i × LayerNorm(x^i) + β₁^i )
x^i = x^i + (1+α₂^i) × FFN( γ₂^i × LayerNorm(x^i) + β₂^i )
```
프레임마다 다른 스케일 γ, 시프트 β, 잔차게이트 α.

⭐ **핵심은 "궤적 전체를 하나로 뭉치지 않는다"** 는 것이다.
[측정] IRASim Table 1 — 궤적 전체를 하나의 임베딩으로 공유하는 대조군(Video-Ada)보다
프레임별(Frame-Ada)이 4개 데이터셋 전부에서 이겼다.
이건 019 확정 5(per-frame FiLM/AdaLN)와 001 §1.1 을 **세 번째로 확인**한 것이다.

우리 구현 (SimVP 에 붙이는 자리):
```python
# forward() 안, embed, skip = self.enc(x) 바로 다음. embed = (B*16, C', H', W')
u   = torch.cat([act, t_idx], dim=-1)          # act=(B,16,6), t_idx=(B,16,1) 정규화 시간 t/16
g,b = self.film(u).chunk(2, dim=-1)            # MLP(7 → 128 → 2*C')
embed = embed * (1 + g.reshape(-1, C_, 1, 1)) + b.reshape(-1, C_, 1, 1)
```
`t_idx` 를 붙이는 이유: ①에서 16슬롯이 복제라 똑같으므로, 시간 인덱스가 있어야 슬롯 구분이 확실해진다.

**TAU 를 쓰면 주입 자리가 하나 더 있다** — §2.2 의 DA 게이트.
`FC(AvgPool(H))` → `FC([AvgPool(H); 행동임베딩])`. `TAUSubBlock` 을 손대는 일이고 20줄 안쪽이다.

### ③ 잔차 — 1줄 (새로 쓰는 것 #2, 사실상 없음)

`Decoder.forward` 95행에 이미 `Y = self.dec[-1](hid + enc1)` 스킵이 있다.
①에서 입력을 복제했으므로 **`enc1` 은 16슬롯 전부 첫프레임의 특징**이다 — 문자 그대로 첫프레임 스킵이다.

픽셀 레벨 명시 잔차는 `forward()` 끝에 한 줄:
```python
Y = Y.reshape(B, T, C, H, W)
Y = x_raw[:, :1] + Y                    # 출력 = 첫프레임 + 잔차
```

**001 §5 가 정한 규율을 그대로 지킨다:**
- `readout = Conv2d(hid_S, 3, 1)` 의 weight·bias 를 **0 으로 초기화** → 0스텝에서 출력이 정확히 static = **0.30325. 하방이 막힌다.**
- ❌ ReZero 식 잔차 가지 전체 스칼라 게이트 금지 (CaiT Table 1: 12층 78.3 vs 기준선 79.90 = **손해**)
- ❌ `tanh` 유계화를 기본값으로 넣지 마라 (RIFE 4.5절에 잔차 범위 어블레이션이 **없다** — 검증관이 재확인).
  큰 움직임에서 gradient 를 죽여 Action 40% 를 깎을 위험이 있다
- 게이트를 두려면 **프레임별/채널별 α 를 작은 값**(1e-3~1e-2)으로

### ④ 해상도 — 다운샘플을 앞으로 당겨라 (설정만)

`sampling_generator(N)` 이 `[False, True, False, True]` 라서 **인코더 0번 블록이 원해상도에서 돈다.**

우리 기계 실측 (`results/branchC/budget.json`, batch=1, 16프레임, Conv3d 3층):

| | 320×512 | 160×256 | 80×128 |
|---|---|---|---|
| c64 | **6.92 GiB** / 103ms | 1.74 / 26ms | 0.45 / 8ms |
| c128 | **12.56 GiB** / 227ms | 3.16 / 58ms | 0.81 / 14ms |
| c256 | **22.58 GiB** / 1577ms | 6.31 / 185ms | 1.61 / 47ms |

손실 3종이 배치4에서 13.1 GiB 를 먹으므로 **백본 여유는 약 18 GiB.**
⇒ 원해상도 블록은 `c64` 가 실질 상한이고, 게다가 `enc1` 스킵이 원해상도 16프레임분으로 역전파까지 살아있다.

**처방:**
- `sampling_generator` 를 뒤집어 **첫 블록부터 다운샘플**, 또는 `N_S` 를 4 → 6~8 로 (→ 40×64 또는 20×32)
- 320 = 2⁶×5, 512 = 2⁶×8 → **2로 6번까지 깨끗이 나눠떨어진다.** 홀수 걱정 없다
- 원해상도 구간 `hid_S` 는 32 이하
- bf16 autocast + 인코더/디코더 gradient checkpointing
- Translator 입력 채널은 `T·hid_S` = 16×64 = **1024** → `N_S=8`(20×32)이면 여유롭다

> ⚠ **320×512 는 이 계열의 미개척 영역이다.** 저장소가 검증한 최대는 Human3.6M 256×256(4→4),
> 논문 표 기준으로는 Caltech 128×160(10→1)이다. 메모리·수렴 전부 **우리가 직접 재야 한다.**

### ⑤ 손실 — MSE 를 그대로 쓰지 마라 ⭐

OpenSTL 기본은 `nn.MSELoss()` 다. **이게 §0-② 붕괴의 직행 티켓이다.**

[측정] arXiv 2606.12987 §5.3 원문: *"The direct regressor wins every distortion metric (CosSim 0.471 vs. 0.260)
by predicting the blurry conditional mean ... while the diffusion model wins every distribution metric."*
Figure 10 Row 3 의 회귀 출력은 *"vehicles and lane markings unrecognizable"* 인 회색 블러였다.

001 §3.1 이 이미 측정으로 뒤집은 것과 같은 이야기다 — **우리 DINO 는 LPIPS형이라 흐릴수록 나빠진다.**
(픽셀 평균이 픽셀 L1 을 −10.4% 개선했는데 채점 DV 는 +20% 악화)

**처방**: `L1(픽셀) + 주파수/지각 대리`. 001 §3.3·§4 의 결론을 그대로 쓴다.
⭐ **주파수 손실은 새로 짤 필요가 없다** — 같은 저장소의 `openstl/methods/wast.py` 에
`HighFocalFrequencyLoss` 가 이미 구현돼 있다(원본 코드로 확인).

### ⑥ 행동 차원 — 6 (관절)

| | 행동 표현 | 차원 |
|---|---|---|
| IRASim (RT-1/Bridge) | 엔드이펙터 **델타** [Δx,Δy,Δz,Δα,Δβ,Δγ, 그리퍼] | 7 |
| arXiv 2606.12987 | 조향·가속 | 2 |
| **우리** | **6자유도 관절** | 6 |

주입 경로상으로는 **Linear 층의 입력 차원만 바꾸면 되고 구조 변경은 없다.**
⚠ 다만 **의미가 다르다.** 둘 다 "델타"인데 우리는 관절값이다.
**절대 관절각을 줄지 프레임 간 델타를 줄지는 우리가 정해야 한다 — 문헌에 답이 없다.**
(참고: `batch["act"]` 는 이미 정규화된 (B,16,6) 이다)

### ⑦ 조건 붕괴 감시 장치 — 이게 이 문서에서 가장 중요하다 ⭐

§0-② 의 ρ=−0.18 이 우리에게 일어나는지를 **학습 중에** 봐야 한다. 019 §6-E 와 같은 계측이다.

```
매 N 스텝 로깅:
  (a) FiLM 파라미터 (γ_t, β_t) 의 프레임 간 쌍별 코사인
      → 0.9 초과면 조건 붕괴.  16프레임이 사실상 하나로 무너지고 있다는 뜻
  (b) 행동 스윕 상관 — 홀드아웃 한 표본의 행동을 스케일해 넣고
      출력 잔차의 크기가 따라 움직이는지 스피어만 ρ
      → 2606.12987 이 잰 바로 그 수치.  우리 것이 0 근처면 §0-② 를 그대로 밟은 것이다
  (c) 잔차 norm — zero-init 이 언제 깨어나는지
      001 §5: ControlNet 은 "abruptly succeeds ... in less than 10K optimization steps",
      그림상 약 6,133 스텝.  **워밍업 지연 ~6,000 스텝을 예산에 넣어라**
```

### ⑧ 평가 — OpenSTL 기본 지표를 믿지 마라

OpenSTL 은 MSE/MAE/SSIM/PSNR 만 보고한다. **우리 지표가 아니다.**
`src/wm_eval/scoring.py` 의 로컬 채점기를 붙이고, 홀드아웃은 `artifacts/holdout_val96/` 를 쓴다
(019: 기존 `artifacts/holdout` 은 96개 중 약 95%가 학습에 쓰인 에피소드다).

> ⚠ **019 의 미결 사항이 아직 살아 있다**: 8/3 규칙 변경으로 채점기 3종을 손실로 쓸 수 있는지가 불확실하다.
> **로컬 채점(평가 목적)이 가능한지**가 주최 질의 ② 다. 막히면 하루 3회 제출로만 점수를 안다.

---

## 4. 2위와 3위 — 1위가 막혔을 때

**1위가 막힌다는 것 = "320×512 에서 SimVP 가 메모리에 안 들어가거나 수렴하지 않는다"** 이다.

### 2위 — IRASim (arXiv 2406.14540, ICCV 2025)

**https://github.com/bytedance/IRASim** · Apache-2.0(원문 확인, 지역 배제 0건) · 별 160 · 학습코드 완비

**우리 규격에 가장 가깝다:**
- [측정] Language-Table 을 **288×512** 로 학습 (우리 320×512 와 거의 같다)
- [측정] 초기프레임 1장 + 행동 15개 → 후속 15프레임을 *"a single generation pass"* (4.1절)
- [측정] 과거 프레임 잠재를 **노이즈 없이 깨끗한 채로** 토큰 시퀀스에 넣고, 손실은 예측프레임에만 건다
  ⇒ 우리 "사진 1장 → 16프레임" 과 정확히 같은 구성
- [측정] 추론은 A100 에서 16프레임 30초, **메모리 8GB** (우리 여유 18GiB 대비 여유)

**떨어진 이유**: C1. 잠재 DDPM 이다 ([측정] 식 2 `L=‖ε_θ−ε‖²`, Table 10 "Prediction target: ε", PNDM 50스텝).

**그런데도 2위인 이유**: 확산만 걷어내면 나머지가 전부 우리 것이다.
그리고 **③ 행동 주입 설계도는 이미 여기서 가져왔다**(§3-②).

**⚠ 2위로 내려갈 때의 대가 — 잠재공간(VAE)이 만드는 문제 4개**
1. **재구성 상한이 생긴다.** VAE 왕복 손실이 우리가 아무리 잘해도 못 넘는 바닥으로 남는다.
   그 바닥값은 **320×512 에서 우리가 직접 재야 한다** — 논문의 값이 오지 않는다
2. **첫프레임 앵커가 첫프레임이 아니게 된다.** 잠재에서 앵커하면 Δ=0 일 때 나오는 것은
   원본 사진이 아니라 **VAE 가 재구성한 사진**이다. ⇒ 하방이 0.30325 로 막히지 않는다
3. **디코더 비용이 16배.** 픽셀 손실을 쓰려면 매 스텝 16장을 디코딩해야 한다
4. **잠재 MSE 는 우리 채점과 정렬되지 않는다** (§3-⑤)
5. 학습 예산: [측정] Table 12 — RT-1 2381 GPU시간(A800 40GB **32장**). XL 679M / L 461M.
   **우리는 이 규모를 처음부터 학습할 수 없다.** S(33M)/B(132M) 로 줄여야 한다
6. ⚠ SDXL VAE 가중치 라이선스(CreativeML Open RAIL++-M 계열)가 파생물에 따라붙는지 **미확인**

### 3위 — VPTR-NAR (arXiv 2212.06026, MIT)

**https://github.com/XiYe20/VPTR**

⭐ **1위에 없는 장점 하나**: `VPTRFormerNAR` 이 `frame_queries`(미래 프레임 수만큼의 학습 쿼리)를 쓴다.
**1장 → 16프레임이 설정값 변경만으로 된다** — SimVP 처럼 입력을 복제하는 우회가 필요 없다.
(§3-① 복제 트릭이 실제로 문제를 일으키면 이쪽이 답이다)

**약점**: 별 102, 마지막 푸시 2025-01-13 정체. 64×64 급만 검증. 사전학습 오토인코더 잠재공간이라 2위와 같은 문제.
손실이 MSE+GDL+BiPatchNCE 로 "순수 결정론 회귀"보다 복잡하다.

---

## 5. 라이선스 지뢰밭 — 검증관이 잡은 것

> **README 배지와 LICENSE 원문이 다른 경우가 실제로 여러 건 나왔다.**
> HunyuanVideo 가 한국 배제로 탈락한 전례가 있어 전부 원문을 열게 했고, 그 결과다.

| 후보 | 조사원이 적은 것 | **원문 확인 결과** |
|---|---|---|
| Vid2World | MIT (README 배지) | ❌ **Apache-2.0.** 배지가 LICENSE 파일과 불일치 |
| Stable Video Diffusion | 미확인 | ❌ 코드 MIT / **가중치는 비상업(NON-COMMERCIAL)** |
| **Ctrl-World** | MIT | ❌ 코드는 MIT 지만 **필수 의존 가중치가 SVD** → 실효 비상업 |
| LTX-Video | Apache-2.0 | ❌ 코드만. **가중치는 LTXV Open Weights License**(매출 1000만$ 이상 제한) |
| Cosmos-Predict2.5 | Apache-2.0 | ❌ 코드 Apache / **가중치 NVIDIA Open Model License** |
| OSCAR | 미확인 | 코드 Apache-2.0 이나 **Cosmos 가중치 종속** → 혼합 |
| CogVideoX | Apache-2.0 | ⚠ **2B 만** Apache. 5B 는 별도 `MODEL_LICENSE`(상업 등록 의무) |
| AnimateDiff | Apache-2.0 | ⚠ 모션 모듈만. SD 베이스 체크포인트는 CreativeML OpenRAIL-M |
| **Anchored VAE DiT** | 라이선스 없음 = 금지 | ⚠ 검증관 정정: `pyproject.toml` 11행에 **`license = {text="MIT"}`** 가 있다. 단 LICENSE 파일은 없어 여전히 모호 |
| PredRNN 원본 / SimVP 원본 저장소 | — | ❌ **LICENSE 파일 자체가 없다** = all rights reserved. **OpenSTL 재구현본만 Apache-2.0** |
| DMVFN-Act | 라이선스 없음 | ⚠ 게다가 루트에 **yolov5(AGPL-3.0)를 벤더링** → 전염 위험 |
| ⭐ **Wan 2.1 / 2.2** | 코드 Apache | ✅ **가중치도 Apache-2.0**(HF 확인). 대비책 중 라이선스 위험 최저 |

⭐ **1위가 이 표에서 자유로운 이유**: OpenSTL 은 **가중치를 배포하지 않는다.**
우리가 처음부터 학습하므로 가중치 라이선스 문제가 **구조적으로 발생하지 않는다.**
8/3 킷 규칙 변경과도 무관하다.

---

## 6. 찾지 못한 것 — 추측으로 메우지 않는다

1. ⭐ **필수조건 5개를 전부 만족하는 논문/구현.** 없다.
   조사원 3명이 검색어 12종 이상으로 독립적으로 훑었다. 전수조사는 아니므로 "세상에 없다"고 단정하지는 않는다.
2. **320×512 급 해상도에서 SimVP/TAU 를 학습한 공개 수치.** 없다.
   확인된 최대는 저장소 model_zoo 의 Human3.6M 256×256(4→4), 논문 표로는 Caltech 128×160(10→1).
   **메모리·수렴 전부 미개척이다.**
3. **잔차의 유계화(tanh) 어블레이션.** VFI 계열 어디에도 없다.
   RIFE 는 묶고 IFRNet/AMT 는 안 묶는데, **이 갈림길에 공개된 실험 근거가 존재하지 않는다.** 우리가 재야 한다.
4. **VFI 계열 + 로봇 관절 행동** 조합의 선행연구. 없다.
   행동 조건부 VFI 는 DMVFN-Act 하나뿐이고 그것도 논문이 아니라 다른 논문(VLMPC)의 내부 부품이다.
5. **우리 채점 지표(DINO·R3D·Action 거리)로 어블레이션한 논문.** 0편.
6. **arXiv 2606.12987 의 GPU·학습 하이퍼파라미터.** 10페이지 전체를 읽었으나 없다.
   (논문에 나오는 Adam lr 1e-3 / batch 256 은 §4.1 인코더 프로브용이지 DiT 용이 **아니다** — 혼동 주의)
7. **TAU 손실의 α 값.** 원문 미확인.
8. **IRASim 학습 스텝 수.** 부록 "300k" vs Table 10 "3000000" 으로 **논문 내부가 10배 불일치**한다.
   저장소 `configs/train/rt1/frame_ada.yaml` 을 열어 확인할 것.
9. **IRASim 학습 시 GPU 메모리.** 논문에 없다(추론 8GB 만 있다). 우리 18GiB 에서 되는지는 **실측 필요**.
10. **WaST 의 arXiv 판.** 존재하지 않는다(AAAI 게재본만). 단 **코드는 OpenSTL 안에 실재한다**(§7 참조).
11. **STLight 의 실제 구현.** 저장소는 실존하나 **STLight 코드가 없는 빈 포크**다. 부록 의사코드로 직접 짜야 한다.
12. 조사하지 못한 것: MaskViT 원문, Seer, RoboDreamer, Video Prediction Policy, ATM, Track2Act,
    그리고 ⭐ **World Models for Robotic Manipulation: A Survey (arXiv 2606.00113)** —
    본문을 열면 결정론 계열 후보 목록이 나올 가능성이 있다. **다음 조사 1순위로 추천한다.**

---

## 7. ⚠ 인용 검증 기록

조사원이 원문을 읽고, **검증관이 같은 원문·저장소·LICENSE 를 독립적으로 다시 열었다.**
그 결과 **판정이 뒤집힌 것 3건, 사실 오류 30여 건**이 나왔다. 주요한 것만 적는다.

| 무엇 | 조사원 | 검증 후 |
|---|---|---|
| ⭐ Anchored VAE DiT 라이선스 | "라이선스 없음 → 탈락" | `pyproject.toml` 에 MIT 명시. **탈락 판정 자체가 뒤집혔다** |
| ⭐ WaST 저장소 | "없음(5중 확인)" | **OpenSTL 안에 있다.** 조사원이 `models/` 만 보고 `methods/` 를 안 봤다 |
| ⭐ DINO-WM / OSCAR / RLA-WM / MaskViT / AVDC / STLight 저장소 | "없음" | **전부 실존.** 탐색 실패였지 부재가 아니었다 |
| ⭐ Anchored VAE DiT 행동 반응성 | (언급 없음) | **§5.5 에 ρ=−0.18 이 있다.** 조사원이 놓쳤고, **이게 이 조사에서 가장 중요한 수치다** |
| ⭐ DMVFN "결정론적 이진화" | 측정이라 표기 | **거짓.** 실제 코드는 `torch.bernoulli` 확률 표본추출. 클래스 이름(`RoundSTE`)만 보고 지어냈다 |
| ⭐ IWS "가중 회귀 L2, 확산 아님" (017 기록) | — | **오독.** `w(σ)` 는 consistency 학습의 노이즈 가중치다. 확산 + 자기회귀가 맞다 |
| Vid2World / SVD / LTX / Cosmos 라이선스 | 배지·검색 결과 | **전부 원문과 불일치** (§5) |
| IFRNet "timestep 조건부 없음" | 측정이라 표기 | **거짓.** `embt` 가 1급 입력이고 8배 보간 학습 코드가 있다 |
| RVD "16스텝 지평 경험 없음" | — | **거짓.** §3.2 에 정확히 "predict 16 future frames" 가 있다. FAIL 사유는 자기회귀여야 한다 |
| SimVP 판정 근거 | DISQUALIFIED (사유 공란) | 결론은 맞지만 근거가 비어 있었다. 진짜 사유는 "행동 없음 + 입력이 1장 아님" |
| 2601.14959 등 2026년 arXiv 번호 5건 | "확인 못 해 인용 금지" | **전부 실재. 제목 대조 완료** |

**내가 직접 재확인한 것 2건** (조사원과 검증관이 엇갈려서):
- `openstl/methods/simvp.py:22` → `if aft_seq_length == pre_seq_length: pred_y = self.model(batch_x)`.
  루프 없음. **복제 트릭은 코드 0줄로 성립한다.** 조사원이 맞았다
- `openstl/methods/wast.py` → HTTP 200, `class WaST(SimVP)`, `HighFocalFrequencyLoss` 실재. **검증관이 맞았다**

---

## 8. 출처

**1위 계열**
- SimVP — [2206.05099](https://arxiv.org/abs/2206.05099) · SimVPv2 — [2211.12509](https://arxiv.org/abs/2211.12509)
  · TAU — [2206.12126](https://arxiv.org/abs/2206.12126) · OpenSTL — [2306.11249](https://arxiv.org/abs/2306.11249)
- 저장소: https://github.com/chengtan9907/OpenSTL (Apache-2.0)
- WaST — AAAI 2024, DOI 10.1609/aaai.v38i5.28230 (**arXiv 판 없음**)

**설계도를 빌린 곳**
- IRASim — [2406.14540](https://arxiv.org/abs/2406.14540) · https://github.com/bytedance/IRASim (Apache-2.0)
  ⚠ 저장소 BibTeX 의 `arXiv:2406.12802` 는 저자 오타다
- "Diffusion Transformer World-Action Model for AV Scene Prediction" — [2606.12987](https://arxiv.org/abs/2606.12987)
  ⚠ **"Anchored VAE DiT" 는 arXiv 제목이 아니라 논문 내부 모듈명이다.** 그 이름으로 검색하면 안 나온다

**대안**
- VPTR — [2212.06026](https://arxiv.org/abs/2212.06026) · https://github.com/XiYe20/VPTR (MIT)

**탈락했으나 참조**
- Interactive World Simulator — [2603.08546](https://arxiv.org/abs/2603.08546) (RSS 2026)
- Nano World Models — [2605.23993](https://arxiv.org/abs/2605.23993) · RVD — [2203.09481](https://arxiv.org/abs/2203.09481)
- MotiF — [2412.16153](https://arxiv.org/abs/2412.16153) · STLight — [2411.10198](https://arxiv.org/abs/2411.10198)
- 다음 조사 1순위: World Models for Robotic Manipulation: A Survey — [2606.00113](https://arxiv.org/abs/2606.00113)

---

## 9. 한 장으로

```
베이스      OpenSTL 의 SimVP 계열.  model_type='tau' 로 시작.
            Apache-2.0, 가중치 배포 없음 → 라이선스·킷규칙 위험 0.

바꿀 것     ① 사진 1장을 16번 복제 (0줄)          ← 안 하면 조용히 자기회귀
            ② 행동 FiLM 주입 (~30줄)              ← 새로 쓰는 것 #1. IRASim Frame-Ada 방식
            ③ Y = 첫프레임 + 잔차, readout zero-init (1줄)  ← 하방 0.30325 봉쇄
            ④ 다운샘플을 앞으로 (설정)             ← 원해상도 c64 가 6.92 GiB
            ⑤ MSE 를 쓰지 마라 → L1 + 주파수      ← 블러 붕괴 회피. WaST 에 구현 있음
            ⑥ 행동 6차원 (절대/델타는 우리가 결정)
            ⑦ 조건 붕괴 감시 3종 로깅             ← ⭐ 이게 제일 중요하다
            ⑧ 로컬 채점기 + holdout_val96

가장 큰 위험  "결정론 회귀는 행동을 무시하고 회색 블러로 무너진다"
            우리와 부품이 같은 논문이 ρ=−0.18 로 그것을 측정해 놨다.
            ⇒ ⑦번 감시 장치를 학습 시작 **전에** 붙여라.

아직 모르는 것  320×512 에서 이 계열이 메모리에 드는지·수렴하는지.  아무도 해본 적이 없다.
```
