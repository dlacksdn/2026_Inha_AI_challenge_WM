#!/usr/bin/env python
"""
③ 후속 3 — ρ=0.187 을 **표본 간 / 표본 내**로 쪼갠다  (008 세션)

왜 하나
  후속 2(action_lowdim_20260808_1742)가 ρ(||Δ행동_t||, mean|잔차_t|) = +0.1868 을 얻었다.
  그런데 그 ρ 는 **표본×프레임을 통째로 풀어서** 계산했다. 두 개가 섞여 있다.

    표본 간(between)  "이 에피소드는 원래 많이 움직인다"    ← 첫 프레임이 이미 아는 것에 가깝다
    표본 내(within)   "이 에피소드 안에서 t=7 에 많이 움직인다"  ← 행동만이 아는 타이밍 정보

  ③ 의 지표(잔차 코사인)는 **표본별 전역 스케일에 불변**이다.
  ⇒ between 성분은 원리적으로 못 본다. within 성분은 (프레임 간 상대 크기이므로) 본다.
  따라서 "③ 이 크기 신호를 볼 수 없는 자였나"는 within 이 얼마나 되느냐로 갈린다.

판정선 (측정 전 등록, 2026-08-08)
  (a) 크기 축
      ρ_within(T) > max|ρ_within(S 3seed)|  AND  ρ_within(T) ≥ 0.10
          → 타이밍 신호 실재. ③ 은 이걸 볼 수 있었는데도 못 잡았다 = 장치/규모 문제
      ρ_within(T) < 0.05
          → 0.187 은 사실상 에피소드 식별 신호. 첫 프레임이 이미 아는 것과 겹친다
      그 사이 → 판정 유보

  (b) 방향 축  (프롬프트가 지목한 최우선 공백)
      Δ행동_t(6) → Δ무게중심_t(2) 릿지 회귀. **표본 내 중심화 후**(= between 제거).
      표본 단위 홀드아웃 R².
      신호 있음 ⟺ R²(T) > max R²(S 3seed)  AND  R²(T) > 0
      ⚠ R² 가 전부 음수면 "두 실패의 비교"다. 후속 2 의 (b) 가 그랬다. 그 경우 신호 없음으로 읽는다.

한계 (미리 적는다)
  무게중심은 잔차 방향의 **2차원 요약**이다. 픽셀 방향과 같지 않다.
  여기서 실패해도 "첫 프레임 + 행동"의 결합 방향 신호까지 부정하지 못한다(후속 2 와 같은 한계).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]   # 상대경로 (대회 §3.3 요건)
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
from loader_c import (EpisodeWindowStream, holdout_episode_refs, list_train_episodes,
                      preprocess_batch, WINDOW)  # noqa: E402
from probe_action_lowdim import act_delta, spearman, summarize  # noqa: E402


def collect_train(eps, n, dev):
    dl = torch.utils.data.DataLoader(
        EpisodeWindowStream(eps, seed=0, span_frames=96, windows_per_span=8),
        batch_size=16, num_workers=8, pin_memory=True, prefetch_factor=4)
    A, S, got = [], [], 0
    for b in dl:
        v = preprocess_batch(b["frames_u8"].to(dev, non_blocking=True))
        S.append(summarize(v, v[:, :, 0]).cpu()); A.append(b["act"])
        got += v.shape[0]
        if got >= n:
            break
    del dl
    return torch.cat(A)[:n], torch.cat(S)[:n]


def ridge_fit(X, Y, lam=1.0):
    """X (N,D) Y (N,K) → W (D+1,K).  bias 포함."""
    X1 = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    D = X1.shape[1]
    R = torch.eye(D); R[-1, -1] = 0.0
    return torch.linalg.solve(X1.T @ X1 + lam * R, X1.T @ Y)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    W = ridge_fit(Xtr, Ytr, lam)
    P = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1) @ W
    ss_res = ((P - Yte) ** 2).sum()
    ss_tot = ((Yte - Ytr.mean(0, keepdim=True)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    dev = "cuda"
    T1 = WINDOW - 1

    eps = list_train_episodes(exclude=holdout_episode_refs())
    print(f"[data] 학습 에피소드 {len(eps)}", flush=True)
    t0 = time.perf_counter()
    A, S = collect_train(eps, args.n, dev)          # A (N,16,6)  S (N,45)
    N = A.shape[0]
    print(f"[collect] {N} 표본  ({time.perf_counter()-t0:.0f}s)\n", flush=True)

    dA = act_delta(A)[:, 1:]                        # (N,15,6)
    dn = dA.norm(dim=-1)                            # (N,15)   행동 변화 크기
    m = S[:, :T1]                                   # (N,15)   잔차 크기
    cx, cy = S[:, T1:2 * T1], S[:, 2 * T1:]         # (N,15) 각각
    g = torch.Generator().manual_seed(0)
    perms = [torch.randperm(N, generator=g) for _ in range(3)]

    # ── (a) 크기 축: pooled / between / within ───────────────────────────
    def rho_pack(x, y):
        pooled = spearman(x.reshape(-1), y.reshape(-1))
        betw = spearman(x.mean(1), y.mean(1))
        wx = x - x.mean(1, keepdim=True)
        wy = y - y.mean(1, keepdim=True)
        within_pooled = spearman(wx.reshape(-1), wy.reshape(-1))
        per = [spearman(x[i], y[i]) for i in range(x.shape[0])]
        return pooled, betw, within_pooled, float(np.mean(per))

    pT = rho_pack(dn, m)
    pS = [rho_pack(dn[p], m) for p in perms]
    lbl = ("pooled", "between", "within(pooled)", "within(표본별 평균)")
    print("=== (a) 크기 축 — ρ(||Δ행동_t||, mean|잔차_t|) ===")
    for j, name in enumerate(lbl):
        s = [x[j] for x in pS]
        print(f"  {name:20s} T {pT[j]:+.4f}   S {[round(v,4) for v in s]}  max|S| {max(abs(v) for v in s):.4f}")
    w_T, w_S = pT[2], max(abs(x[2]) for x in pS)
    if w_T > w_S and w_T >= 0.10:
        va = "timing_signal"
    elif w_T < 0.05:
        va = "episode_id_only"
    else:
        va = "undecided"
    print(f"  ⇒ within {w_T:+.4f} vs 귀무 {w_S:.4f}  →  **{va}**\n")

    # ── (b) 방향 축: Δ행동(6) → Δ무게중심(2), 표본 내 중심화 ─────────────
    print("=== (b) 방향 축 — Δ행동_t(6) → Δ무게중심_t(2), between 제거 후 ===")
    ntr = int(N * 0.8)
    Y = torch.stack([cx, cy], -1)                       # (N,15,2)
    Yw = Y - Y.mean(1, keepdim=True)
    Xw = dA - dA.mean(1, keepdim=True)                  # (N,15,6)

    def flat(X, Y, idx):
        return X[idx].reshape(-1, 6), Y[idx].reshape(-1, 2)

    tr, te = torch.arange(ntr), torch.arange(ntr, N)
    Xtr, Ytr = flat(Xw, Yw, tr); Xte, Yte = flat(Xw, Yw, te)
    rT = ridge_r2(Xtr, Ytr, Xte, Yte)
    rS = []
    for p in perms:
        Xp = Xw[p]
        Xtr2, _ = flat(Xp, Yw, tr); Xte2, _ = flat(Xp, Yw, te)
        rS.append(ridge_r2(Xtr2, Ytr, Xte2, Yte))
    print(f"  T(진짜)  R² = {rT:+.5f}")
    print(f"  S(뒤섞기) R² = {[round(v,5) for v in rS]}   max {max(rS):+.5f}")
    vb = "direction_signal" if (rT > max(rS) and rT > 0) else (
        "both_fail" if rT <= 0 else "null_tie")
    print(f"  ⇒ **{vb}**\n")

    # 참고: 크기까지 포함한(중심화 안 한) 같은 회귀
    Xa, Ya = dA.reshape(-1, 6), Y.reshape(-1, 2)
    rT_raw = ridge_r2(Xa[:ntr * T1], Ya[:ntr * T1], Xa[ntr * T1:], Ya[ntr * T1:])
    print(f"  [참고] between 미제거 R² = {rT_raw:+.5f}")

    out = {"generated": datetime.now().isoformat(timespec="seconds"), "n": int(N),
           "size_axis": {"labels": lbl, "T": list(pT), "S": [list(x) for x in pS],
                         "verdict": va},
           "direction_axis": {"T": rT, "S": rS, "verdict": vb, "raw_R2": rT_raw}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
