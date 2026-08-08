#!/usr/bin/env python
"""
③ 행동 신호 측정 (G0) — 행동이 잔차를 얼마나 설명하는가

판정선은 측정 전에 등록했다 (007 후속, 2026-08-08):

  성격    G0. **단방향 설계 입력**이다. 이 결과로 노선을 접지 않는다 (005 치-6 격하 수용).
          소형 회귀기의 미검출은 "신호 없음"의 증명이 아니고, 행동 신호가 0 이어도
          첫 프레임 조건 모션만으로 static 을 이길 여지가 있다.

  얻을 것 ① 행동이 잔차를 설명하는가  ② 행동 표현 결정(004 §2.3-⑥ 미결)  ③ 귀무분포

  팔      T-abs / T-delta / T-both      행동 표현 3분할
          S-shuffle × seed 3           귀무. 행동을 표본 간 뒤섞는다
          Z-zero                        행동 0
          PC                            양성 대조. 첫 프레임을 행동 누적합만큼 평행이동한 합성 영상

  판정    신호 있음 ⟺ max(T) > max(S 3seed) AND 짝지은 t ≥ +2
          표현    ⟺ T 3분할 중 최고. 1위−2위 차 < S seed 산포(max−min) 면 무승부 → **델타**
          ⚠ PC 가 S 를 못 넘으면 **실험 무효.** 다른 결과를 해석하지 않는다

  ⚠ 해상도 160×256 (목표의 1/2). 탐침 비용을 1시간 안에 두기 위해서다.
    ⇒ 그 규모 이상의 신호만 검출한다. 더 미세한 신호는 못 본다.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, statistics, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]   # 상대경로 (대회 §3.3 요건)
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
sys.path.insert(0, str(REPO / "third_party" / "OpenSTL"))
from loader_c import (EpisodeWindowStream, holdout_episode_refs, list_train_episodes,
                      load_holdout_val96, preprocess_batch, WINDOW)  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_simvp", str(REPO / "third_party" / "OpenSTL" / "openstl" / "models" / "simvp_model.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SimVP_Model = _m.SimVP_Model

PH, PW = 160, 256        # 탐침 해상도 (목표 320×512 의 1/2)


# ───────────────────────── 모델 ─────────────────────────

class ActionFiLM(nn.Module):
    def __init__(self, act_dim, ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(act_dim, 128), nn.SiLU(), nn.Linear(128, 2 * ch))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, a):            # z (B,T,C,H,W)  a (B,T,act_dim)
        g, b = self.mlp(a).chunk(2, -1)
        return z * (1 + g[..., None, None]) + b[..., None, None]


class ProbeNet(nn.Module):
    """소형 SimVP-TAU + FiLM 행동 주입 + 잔차 출력(zero-init)."""

    def __init__(self, act_dim, hid_S=16, hid_T=128, N_S=2, N_T=2):
        super().__init__()
        self.net = SimVP_Model(in_shape=(WINDOW, 3, PH, PW), hid_S=hid_S, hid_T=hid_T,
                               N_S=N_S, N_T=N_T, model_type="tau")
        nn.init.zeros_(self.net.dec.readout.weight); nn.init.zeros_(self.net.dec.readout.bias)
        self.film = ActionFiLM(act_dim, hid_S)

    def forward(self, first, act):      # first (B,3,H,W)  act (B,T,act_dim)
        B = first.shape[0]
        n = self.net
        embed, skip = n.enc(first)                       # 인코더 1회 (004 ①-b)
        embed = embed.repeat_interleave(WINDOW, 0)
        skip = skip.repeat_interleave(WINDOW, 0)
        _, C_, H_, W_ = embed.shape
        z = self.film(embed.view(B, WINDOW, C_, H_, W_), act)
        hid = n.hid(z).reshape(B * WINDOW, C_, H_, W_)
        res = n.dec(hid, skip).reshape(B, WINDOW, 3, PH, PW).permute(0, 2, 1, 3, 4)
        return first.unsqueeze(2) + res                  # (B,3,T,H,W)


# ───────────────────────── 행동 표현 ─────────────────────────

def make_act(a, mode):                  # a (B,T,6) 정규화 공간
    if mode == "abs":
        return a
    if mode == "delta":
        d = torch.zeros_like(a); d[:, 1:] = a[:, 1:] - a[:, :-1]
        return d
    if mode == "both":
        d = torch.zeros_like(a); d[:, 1:] = a[:, 1:] - a[:, :-1]
        return torch.cat([a, d], -1)
    raise ValueError(mode)


ACT_DIM = {"abs": 6, "delta": 6, "both": 12}


# ───────────────────────── 양성 대조: 합성 영상 ─────────────────────────

def synth_positive(first, act):
    """첫 프레임을 행동 누적합에 비례해 평행이동한 영상. 행동→잔차가 결정론적이다."""
    B, C, H, W = first.shape
    cum = act[:, :, :2].cumsum(1)                            # (B,T,2)
    cum = cum / (cum.abs().amax(dim=(1, 2), keepdim=True) + 1e-6) * 0.15   # 최대 15% 이동
    out = []
    for t in range(WINDOW):
        theta = torch.zeros(B, 2, 3, device=first.device, dtype=first.dtype)
        theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
        theta[:, 0, 2] = cum[:, t, 0]; theta[:, 1, 2] = cum[:, t, 1]
        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
        out.append(F.grid_sample(first, grid, align_corners=False, padding_mode="border"))
    return torch.stack(out, 2)                                # (B,C,T,H,W)


# ───────────────────────── 지표 ─────────────────────────

def residual_cos(pred, gt, first):
    """예측 잔차 ↔ 정답 잔차 코사인. 프레임 1~15 (0번은 구조적으로 0)."""
    pr = (pred - first.unsqueeze(2))[:, :, 1:]
    gr = (gt - first.unsqueeze(2))[:, :, 1:]
    pr = pr.reshape(pr.shape[0], -1); gr = gr.reshape(gr.shape[0], -1)
    return F.cosine_similarity(pr.float(), gr.float(), dim=1)   # (B,)


# ───────────────────────── 한 팔 ─────────────────────────

def run_arm(name, mode, arm, seed, eps, val, steps, batch, lr, log):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    model = ProbeNet(ACT_DIM[mode]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ds = EpisodeWindowStream(eps, seed=seed, span_frames=96, windows_per_span=8)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=8,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
    it = iter(dl)
    t0 = time.perf_counter()
    for step in range(steps):
        b = next(it)
        v = preprocess_batch(b["frames_u8"].to(dev, non_blocking=True))
        v = F.interpolate(v.flatten(0, 1), size=(PH, PW), mode="bilinear",
                          align_corners=False).view(v.shape[0], 3, WINDOW, PH, PW)
        a = b["act"].to(dev)
        if arm == "shuffle":
            a = a[torch.randperm(a.shape[0], device=dev)]
        elif arm == "zero":
            a = torch.zeros_like(a)
        first = v[:, :, 0]
        if arm == "pc":
            v = synth_positive(first, b["act"].to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(first, make_act(a, mode))
            loss = (out - v).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            print(f"    [{name}] step {step+1}/{steps} loss {loss.item():.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True, file=log)
    del dl, it

    # 평가
    model.eval(); cos = []
    with torch.no_grad():
        for i in range(0, len(val), 4):
            chunk = val[i:i + 4]
            v = torch.stack([c["video"] for c in chunk]).to(dev)
            v = F.interpolate(v.flatten(0, 1), size=(PH, PW), mode="bilinear",
                              align_corners=False).view(len(chunk), 3, WINDOW, PH, PW)
            a = torch.stack([c["act"] for c in chunk]).to(dev)
            if arm == "shuffle":
                a = a[torch.randperm(a.shape[0], device=dev)]
            elif arm == "zero":
                a = torch.zeros_like(a)
            first = v[:, :, 0]
            if arm == "pc":
                v = synth_positive(first, torch.stack([c["act"] for c in chunk]).to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(first, make_act(a, mode))
            cos.append(residual_cos(out.float(), v.float(), first.float()).cpu())
    c = torch.cat(cos)
    del model, opt; torch.cuda.empty_cache()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    log = sys.stdout

    print(f"[env] torch {torch.__version__} · {torch.cuda.get_device_name(0)}")
    print(f"[좌표] 탐침 해상도 {PH}×{PW} · 스텝 {args.steps} · 배치 {args.batch}\n")

    eps = list_train_episodes(exclude=holdout_episode_refs())
    val = load_holdout_val96()
    print(f"[data] 학습 에피소드 {len(eps)} · 평가 {len(val)} 표본\n")

    arms = ([("T-abs", "abs", "true", 0), ("T-delta", "delta", "true", 0), ("T-both", "both", "true", 0)]
            + [(f"S-shuf{s}", "delta", "shuffle", s) for s in (1, 2, 3)]
            + [("Z-zero", "delta", "zero", 0), ("PC", "delta", "pc", 0)])

    res = {}
    for name, mode, arm, seed in arms:
        print(f"[팔] {name} (표현={mode}, arm={arm}, seed={seed})", flush=True)
        c = run_arm(name, mode, arm, seed, eps, val, args.steps, args.batch, args.lr, log)
        res[name] = c
        print(f"    → 잔차 코사인 평균 {c.mean():.4f}  중앙값 {c.median():.4f}\n", flush=True)

    # ── 사전 등록한 판정선을 기계적으로 적용 ──
    print("=" * 64)
    print("판정 (판정선은 측정 전에 등록했다)")
    print("=" * 64)
    S = [res[f"S-shuf{s}"].mean().item() for s in (1, 2, 3)]
    Smax, Sspread = max(S), max(S) - min(S)
    T = {k: res[k].mean().item() for k in ("T-abs", "T-delta", "T-both")}
    Tbest = max(T, key=T.get)
    print(f"  S팔(귀무) 평균들 {[round(x,4) for x in S]}  → max {Smax:.4f}  산포 {Sspread:.4f}")
    print(f"  T팔 {({k: round(v,4) for k,v in T.items()})}")
    print(f"  Z-zero {res['Z-zero'].mean():.4f}   PC(양성대조) {res['PC'].mean():.4f}")

    pc_ok = res["PC"].mean().item() > Smax
    print(f"\n  ⚠ 양성대조: PC {res['PC'].mean():.4f} vs S max {Smax:.4f} → "
          f"{'✅ 실험 유효' if pc_ok else '❌ 실험 무효 — 아래를 해석하지 마라'}")

    d = res[Tbest] - res["S-shuf1"]
    t_stat = d.mean().item() / (d.std().item() / np.sqrt(len(d)) + 1e-12)
    signal = (T[Tbest] > Smax) and (t_stat >= 2.0)
    print(f"\n  신호 판정: max(T)={T[Tbest]:.4f} > max(S)={Smax:.4f} ? "
          f"{T[Tbest] > Smax}   짝지은 t = {t_stat:+.2f} (≥+2 필요)")
    print(f"  ⇒ {'✅ 행동 신호 있음' if signal else '❌ 검출 실패 (이 규모에서)'}")

    order = sorted(T, key=T.get, reverse=True)
    gap = T[order[0]] - T[order[1]]
    tie = gap < Sspread
    rep = "delta" if tie else order[0].split("-")[1]
    print(f"\n  표현 결정: 1위 {order[0]}({T[order[0]]:.4f}) − 2위 {order[1]}({T[order[1]]:.4f}) "
          f"= {gap:.4f}  vs S산포 {Sspread:.4f}")
    print(f"  ⇒ {'무승부 → 사전 등록 기본값' if tie else '결정'}: **{rep}**")

    out = {"generated": datetime.now().isoformat(timespec="seconds"),
           "probe_hw": [PH, PW], "steps": args.steps, "batch": args.batch, "lr": args.lr,
           "arms": {k: {"mean": v.mean().item(), "median": v.median().item(),
                        "per_sample": v.tolist()} for k, v in res.items()},
           "verdict": {"pc_ok": pc_ok, "signal": signal, "t_stat": t_stat,
                       "S_max": Smax, "S_spread": Sspread, "T": T,
                       "action_repr": rep, "tie": tie}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
