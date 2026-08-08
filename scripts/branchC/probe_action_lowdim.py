#!/usr/bin/env python
"""
③ 후속 2 — 행동 → 잔차의 **저차원 요약** 회귀

왜 하나
  검출한계 측정(detection_limit_20260808_1551)이 말한 것:
    같은 크기의 변화인데 행동으로 100% 결정되면 코사인 0.65, 실제 데이터면 0.12.
    ⇒ 크기 문제가 아니라 예측 가능성 문제다.
  그러나 양성대조가 **전역 강체 평행이동**이라 **구조 축이 통제 안 됐다.**

  여기서는 픽셀을 버리고 **가장 잡기 쉬운 표적**을 본다.
  픽셀 잔차를 못 맞히는 것과, 잔차의 "얼마나·어디서 움직였나"라는 저차원 요약조차
  못 맞히는 것은 전혀 다른 이야기다. 후자면 데이터에 신호가 없다는 가장 강한 증거다.

요약 정의 (프레임 t=1..15)
  m_t          = mean|잔차_t|            얼마나 변했나
  (cx_t, cy_t) = |잔차_t| 의 무게중심     어디가 변했나
  ⇒ 45 차원

판정선 (측정 전 등록)
  (a) 무학습 상관  스피어만 ρ( ||Δ행동_t||, m_t )  — 표본×프레임 전체
      귀무 = 행동을 표본 간 뒤섞어 같은 계산 × 3 seed
      신호 있음 ⟺ |ρ(T)| > max |ρ(S)|
  (b) MLP 회귀    행동(16×6 델타, 96차원) → 요약 45차원.  holdout R².
      신호 있음 ⟺ R²(T) > max R²(S 3seed)   (R² 는 학습셋 평균 예측 기준)

한계 (미리 적는다)
  행동만 조건으로 준다. 장면이 무엇인지(어느 로봇·무엇이 놓여 있나)는 안 준다.
  따라서 이 실험은 "행동 **단독**이 움직임 요약을 설명하는가"를 잰다.
  실패해도 "첫 프레임 + 행동"의 결합 신호까지 부정하지는 못한다.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
from loader_c import (EpisodeWindowStream, holdout_episode_refs, list_train_episodes,
                      load_holdout_val96, preprocess_batch, WINDOW)  # noqa: E402


def summarize(video, first):
    """(B,3,T,H,W) → (B, 45).  프레임 1..15 의 [mean|r|, cx, cy]."""
    r = (video - first.unsqueeze(2))[:, :, 1:].abs().mean(1)      # (B,T-1,H,W)
    B, T1, H, W = r.shape
    m = r.mean(dim=(2, 3))                                        # (B,T-1)
    tot = r.sum(dim=(2, 3)) + 1e-8
    ys = torch.linspace(-1, 1, H, device=r.device).view(1, 1, H, 1)
    xs = torch.linspace(-1, 1, W, device=r.device).view(1, 1, 1, W)
    cy = (r * ys).sum(dim=(2, 3)) / tot
    cx = (r * xs).sum(dim=(2, 3)) / tot
    return torch.cat([m, cx, cy], dim=1)                          # (B, 3*(T-1))


def act_delta(a):
    d = torch.zeros_like(a); d[:, 1:] = a[:, 1:] - a[:, :-1]
    return d


def collect(stream_or_val, n, dev, is_val=False):
    """(행동 델타 (N,16,6), 요약 (N,45), 프레임별 m (N,15)) 를 모은다."""
    A, S = [], []
    if is_val:
        for i in range(0, len(stream_or_val), 8):
            ch = stream_or_val[i:i + 8]
            v = torch.stack([c["video"] for c in ch]).to(dev)
            a = torch.stack([c["act"] for c in ch]).to(dev)
            S.append(summarize(v, v[:, :, 0]).cpu()); A.append(a.cpu())
    else:
        dl = torch.utils.data.DataLoader(stream_or_val, batch_size=16, num_workers=8,
                                         pin_memory=True, prefetch_factor=4)
        got = 0
        for b in dl:
            v = preprocess_batch(b["frames_u8"].to(dev, non_blocking=True))
            S.append(summarize(v, v[:, :, 0]).cpu()); A.append(b["act"])
            got += v.shape[0]
            if got >= n:
                break
        del dl
    return torch.cat(A)[:n], torch.cat(S)[:n]


def spearman(x, y):
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return (rx @ ry / (rx.norm() * ry.norm() + 1e-12)).item()


def r2(pred, true, train_mean):
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - train_mean) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda"
    T1 = WINDOW - 1

    eps = list_train_episodes(exclude=holdout_episode_refs())
    val = load_holdout_val96()
    print(f"[data] 학습 에피소드 {len(eps)} · 평가 {len(val)}")

    t0 = time.perf_counter()
    Atr, Str = collect(EpisodeWindowStream(eps, seed=0, span_frames=96, windows_per_span=8),
                       args.n_train, dev)
    Ava, Sva = collect(val, len(val), dev, is_val=True)
    print(f"[collect] 학습 {Atr.shape[0]} · 평가 {Ava.shape[0]}  ({time.perf_counter()-t0:.0f}s)\n")

    # ── (a) 무학습 상관 ──────────────────────────────────────────────
    print("=== (a) 무학습 상관: ||Δ행동_t|| ↔ mean|잔차_t| ===")
    dn = act_delta(Atr).norm(dim=-1)[:, 1:].reshape(-1)     # (N*15,)
    m = Str[:, :T1].reshape(-1)
    rho_T = spearman(dn, m)
    rho_S = []
    g = torch.Generator().manual_seed(0)
    for s in range(3):
        perm = torch.randperm(Atr.shape[0], generator=g)
        dns = act_delta(Atr[perm]).norm(dim=-1)[:, 1:].reshape(-1)
        rho_S.append(spearman(dns, m))
    print(f"   T(진짜)  ρ = {rho_T:+.4f}")
    print(f"   S(뒤섞기) ρ = {[round(x,4) for x in rho_S]}   max|ρ| = {max(abs(x) for x in rho_S):.4f}")
    sig_a = abs(rho_T) > max(abs(x) for x in rho_S)
    print(f"   ⇒ {'✅ 상관 있음' if sig_a else '❌ 귀무와 구분 안 됨'}\n")

    # ── (b) MLP 회귀 ────────────────────────────────────────────────
    print("=== (b) MLP: 행동 델타(96) → 요약(45).  holdout R² ===")
    mu, sd = Str.mean(0, keepdim=True), Str.std(0, keepdim=True) + 1e-6

    def fit(A, label):
        X = act_delta(A).reshape(A.shape[0], -1).to(dev)
        Y = ((Str - mu) / sd).to(dev)
        Xv = act_delta(Ava).reshape(Ava.shape[0], -1).to(dev)
        Yv = ((Sva - mu) / sd).to(dev)
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(),
                            nn.Linear(256, 256), nn.SiLU(),
                            nn.Linear(256, Y.shape[1])).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for e in range(args.epochs):
            opt.zero_grad(); loss = F.mse_loss(net(X), Y); loss.backward(); opt.step()
        with torch.no_grad():
            score = r2(net(Xv), Yv, Y.mean(0, keepdim=True))
        print(f"   {label:12s} R² = {score:+.4f}   (train MSE {loss.item():.4f})")
        return score

    r_T = fit(Atr, "T(진짜)")
    r_S = []
    for s in range(3):
        perm = torch.randperm(Atr.shape[0], generator=torch.Generator().manual_seed(100 + s))
        r_S.append(fit(Atr[perm], f"S(뒤섞기{s+1})"))
    sig_b = r_T > max(r_S)
    print(f"   ⇒ T {r_T:+.4f} vs max(S) {max(r_S):+.4f} → "
          f"{'✅ 신호 있음' if sig_b else '❌ 귀무와 구분 안 됨'}\n")

    print("=" * 60)
    print(f"판정: (a) 상관 {'✅' if sig_a else '❌'}   (b) 회귀 {'✅' if sig_b else '❌'}")
    if sig_a and sig_b:
        print("  ⇒ 행동 단독으로도 움직임 요약이 설명된다.")
        print("     ③ 픽셀 미검출의 원인은 **신호 부재가 아니라 주입 구조/규모**다.")
    elif not sig_a and not sig_b:
        print("  ⇒ 가장 쉬운 표적에서도 행동 단독으로는 설명이 안 된다.")
        print("     ⚠ 다만 '첫 프레임 + 행동'의 결합 신호까지 부정하지는 못한다(한계 참조).")
    else:
        print("  ⇒ 두 지표가 엇갈렸다. 단정하지 않는다.")

    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "n_train": int(Atr.shape[0]), "n_val": int(Ava.shape[0]),
               "corr": {"T": rho_T, "S": rho_S, "signal": bool(sig_a)},
               "mlp_r2": {"T": r_T, "S": r_S, "signal": bool(sig_b)}},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
