#!/usr/bin/env python
"""
branch C 학습 루프 — 계기판·체크포인트·그림을 전부 달고 돈다

설계 원칙
  1. 판정선은 gates.py 에서 읽는다. 이 파일에 문턱을 적지 않는다 (학습 중 조용히 바뀌는 것 방지)
  2. 체크포인트는 촘촘히 (CLAUDE.md). 중간에 끊어도 안 날아간다
  3. **매 감시 시점에 그림을 남긴다.** 숫자가 못 보는 것을 그림이 본다 —
     002 §1.2 가 인용한 실패 사례가 "차량도 차선도 알아볼 수 없는 균일한 회색 블러"다.
     ρ 가 통과해도 그림이 죽어 있으면 소용없다
  4. 손실은 인터페이스만 뚫어 둔다. 행동항·주파수항은 008 이 정해지면 끼운다

사용
  python train_c.py --steps 6000 --tag g1
  재개: --resume <ckpt>   (016 §6 대응: 재개 시 seed+1)
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
from loader_c import (EpisodeWindowStream, holdout_episode_refs, list_train_episodes,
                      load_holdout_val96, preprocess_batch, WINDOW)      # noqa: E402
from model_c import ResidualSimVPC, C, H, W, T                           # noqa: E402
import gates as G                                                        # noqa: E402


# ═══════════════════════ 계기판 ═══════════════════════

@torch.no_grad()
def monitor(model, val, dev, n_rho=None, seed=0):
    """감시 5종. 전부 킷 없이 우리가 계산한다."""
    model.eval()
    n_rho = n_rho or G.RHO_N_SAMPLES
    out = {}

    # 공통: 홀드아웃 앞 n 개
    sub = val[:max(n_rho, 8)]
    firsts = torch.stack([s["first"] for s in sub]).to(dev)
    acts = torch.stack([s["act"] for s in sub]).to(dev)
    vids = torch.stack([s["video"] for s in sub]).to(dev)

    def fwd(f, a, bs=4):
        o = []
        for i in range(0, f.shape[0], bs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o.append(model(f[i:i + bs], a[i:i + bs]).float())
        return torch.cat(o)

    pred = fwd(firsts, acts)
    res_p = (pred - firsts.unsqueeze(2))[:, :, 1:]
    res_g = (vids - firsts.unsqueeze(2))[:, :, 1:]

    # (c) wake — 예측 잔차 norm ÷ 정답 잔차 norm
    out["resid_ratio"] = (res_p.flatten(1).norm(dim=1) /
                          (res_g.flatten(1).norm(dim=1) + 1e-8)).mean().item()

    # (e) 방향 — 잔차 코사인
    cos_t = F.cosine_similarity(res_p.flatten(1), res_g.flatten(1), dim=1)
    out["resid_cos"] = cos_t.mean().item()

    # (b2) Δcos — 진짜 행동 vs 뒤섞은 행동. 귀무를 동시에 잰다
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(acts.shape[0], generator=g).to(dev)
    pred_s = fwd(firsts, acts[perm])
    res_s = (pred_s - firsts.unsqueeze(2))[:, :, 1:]
    cos_s = F.cosine_similarity(res_s.flatten(1), res_g.flatten(1), dim=1)
    out["dcos"] = (cos_t - cos_s).mean().item()
    out["resid_cos_shuf"] = cos_s.mean().item()

    # (d) 프레임별 잔차 프로파일 기울기 (0 에서 시작해 증가해야 한다)
    prof = res_p.abs().mean(dim=(0, 1, 3, 4))               # (T-1,)
    x = torch.arange(prof.numel(), device=dev, dtype=prof.dtype)
    x = x - x.mean()
    slope = (x @ (prof - prof.mean())) / (x @ x)
    out["profile_slope"] = (slope / (prof.mean() + 1e-8)).item()
    out["profile"] = prof.tolist()

    # (b1) 스피어만 — 행동 스케일 5점 스윕
    fr, ar = firsts[:n_rho], acts[:n_rho]
    base = fwd(fr, torch.zeros_like(ar))
    d = []
    for s in G.RHO_SCALES:
        p = fwd(fr, ar * s)
        d.append((p - base).flatten(1).norm(dim=1))
    D = torch.stack(d, 1).cpu().numpy()                      # (n, 5)
    sc = np.array(G.RHO_SCALES)
    rr = []
    for i in range(D.shape[0]):
        a_ = np.argsort(np.argsort(sc)).astype(float)
        b_ = np.argsort(np.argsort(D[i])).astype(float)
        a_ -= a_.mean(); b_ -= b_.mean()
        rr.append(float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-12)))
    out["rho_median"] = float(np.median(rr))

    # (a) FiLM 시간 붕괴 — 프레임 간 변조값 코사인
    from model_c import act_delta
    gb = model.film_in.mlp(act_delta(acts[:8]))              # (B,T,2*ch)
    gn = F.normalize(gb, dim=-1)
    cs = (gn[:, :-1] * gn[:, 1:]).sum(-1)
    out["film_temporal_cos"] = cs.mean().item()

    model.train()
    return out


# ═══════════════════════ 그림 ═══════════════════════

@torch.no_grad()
def save_viz(model, val, dev, step, outdir: Path, n=4):
    """[첫프레임 | 정답 | 예측 | 잔차히트맵] × 프레임 0/5/10/15 를 한 장으로."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    # [측정] holdout_val96 은 96개 중 4개가 16:9 다(sample_000003·37·66·94).
    #   학습은 4:3 만 쓰므로 그림은 4:3 표본으로 고른다. 감시는 96개 전부를 쓴다
    #   (018 오라클이 96개로 계산돼 비교 가능성을 지켜야 한다).
    sub = [s_ for s_ in val if s_["sid"] not in
           ("sample_000003", "sample_000037", "sample_000066", "sample_000094")][:n]
    f = torch.stack([s["first"] for s in sub]).to(dev)
    a = torch.stack([s["act"] for s in sub]).to(dev)
    v = torch.stack([s["video"] for s in sub]).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model(f, a).float()

    def img(x):                       # (3,H,W) [-1,1] → (H,W,3) [0,1]
        return ((x.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)

    frames = [0, 5, 10, 15]
    fig, ax = plt.subplots(n * 2, len(frames) + 1, figsize=(3 * (len(frames) + 1), 2.2 * n * 2))
    for i in range(n):
        ax[2 * i, 0].imshow(img(f[i])); ax[2 * i, 0].set_ylabel(f"#{i}\nGT", fontsize=8)
        ax[2 * i, 0].set_title("first frame", fontsize=8)
        ax[2 * i + 1, 0].imshow(img(f[i])); ax[2 * i + 1, 0].set_ylabel("PRED", fontsize=8)
        for j, t in enumerate(frames):
            ax[2 * i, j + 1].imshow(img(v[i, :, t]))
            ax[2 * i, j + 1].set_title(f"GT t={t}", fontsize=8)
            ax[2 * i + 1, j + 1].imshow(img(p[i, :, t]))
            r = (p[i, :, t] - f[i]).abs().mean(0).cpu().numpy()
            ax[2 * i + 1, j + 1].set_title(f"PRED t={t}  |res|{r.mean():.3f}", fontsize=8)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"step {step}", fontsize=10)
    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = outdir / f"{ts}_pred_step{step:06d}.png"
    fig.savefig(path, dpi=90); plt.close(fig)

    # 잔차 히트맵 (무엇을 어디에 그리는가)
    fig, ax = plt.subplots(n, len(frames), figsize=(3 * len(frames), 2.2 * n))
    ax = np.atleast_2d(ax)
    for i in range(n):
        for j, t in enumerate(frames):
            rp = (p[i, :, t] - f[i]).abs().mean(0).cpu().numpy()
            rg = (v[i, :, t] - f[i]).abs().mean(0).cpu().numpy()
            # ⚠ 같은 자로 그린다. imshow 자동정규화를 쓰면 미세한 예측 잔차가
            #   정답만큼 밝게 늘어나 그림을 오독한다 (스모크에서 실제로 겪었다)
            vmax = max(float(rg.max()), 1e-6)
            ax[i, j].imshow(np.concatenate([rg, rp], axis=0), cmap="inferno",
                            vmin=0.0, vmax=vmax)
            ax[i, j].set_title(f"t={t} top:GT bot:PRED  vmax={vmax:.2f}", fontsize=7)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"residual heatmap  step {step}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"{ts}_resid_step{step:06d}.png", dpi=90); plt.close(fig)
    model.train()
    return path


# ═══════════════════════ 학습 ═══════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c")
    ap.add_argument("--steps", type=int, default=6000, help="옵티마이저 스텝")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="유효배치 = micro × accum")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hid-s", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--monitor-every", type=int, default=None,
                    help="기본값은 gates.MONITOR_EVERY. 스모크에서만 줄인다")
    args = ap.parse_args()
    mon_every = args.monitor_every or G.MONITOR_EVERY

    dev = "cuda"
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    ckdir = REPO / "artifacts" / "branchC" / f"train_{ts}_{args.tag}"
    vizdir = ckdir / "viz"
    ckdir.mkdir(parents=True, exist_ok=True); vizdir.mkdir(exist_ok=True)
    print(f"[out] 체크포인트 {ckdir}")
    print(f"[out] 그림       {vizdir}")
    print(f"[gates] ρ≥{G.RHO_NULL_P95}(N={G.RHO_N_SAMPLES}) · Δcos≥{G.DCOS_MIN} · "
          f"wake={G.WAKE_RATIO} · 감시주기 {mon_every}")

    seed = args.seed + (1 if args.resume else 0)      # 016 §6: 재개 시 seed+1
    torch.manual_seed(seed); np.random.seed(seed)

    eps = list_train_episodes(exclude=holdout_episode_refs())
    val = load_holdout_val96()
    print(f"[data] 학습 에피소드 {len(eps)} · 평가 {len(val)}\n")

    model = ResidualSimVPC(hid_S=args.hid_s, use_ckpt=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"[resume] {args.resume} step {start} (seed {seed})")

    dl = torch.utils.data.DataLoader(
        EpisodeWindowStream(eps, seed=seed, span_frames=96, windows_per_span=8),
        batch_size=args.micro_batch, num_workers=args.workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True)
    it = iter(dl)

    hist, dcos_hist, wake_step = [], [], None
    t0 = time.perf_counter()
    for step in range(start, args.steps):
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.accum):
            b = next(it)
            v = preprocess_batch(b["frames_u8"].to(dev, non_blocking=True))
            a = b["act"].to(dev)
            first = v[:, :, 0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(first, a)
                # ── 손실 (008 이 정해지면 여기에 항을 끼운다) ──
                loss = (out - v).abs().mean()               # L1
                # + lam_f * frequency_loss(out, v)          ← G2.5 점화 후
                # + lam_a * action_loss(out, a)             ← 008 대기
            (loss / args.accum).backward()
            tot += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"step {step+1}/{args.steps}  loss {tot:.4f}  lr {lr:.2e}  "
                  f"{el/(step+1-start):.2f}s/step  경과 {el/60:.0f}분", flush=True)

        if (step + 1) % mon_every == 0:
            m = monitor(model, val, dev)
            m["step"] = step + 1; m["loss"] = tot
            hist.append(m); dcos_hist.append(m["dcos"])
            if wake_step is None and m["resid_ratio"] >= G.WAKE_RATIO:
                wake_step = step + 1
                print(f"  ⭐ wake! step {wake_step} (잔차비 {m['resid_ratio']:.3f})")
            g2 = G.check_g2(m["rho_median"], m["dcos"])
            print(f"  [감시 {step+1}] 잔차비 {m['resid_ratio']:.3f}  "
                  f"코사인 {m['resid_cos']:+.3f}(뒤섞기 {m['resid_cos_shuf']:+.3f})  "
                  f"Δcos {m['dcos']:+.3f}  ρ중앙값 {m['rho_median']:+.2f}  "
                  f"프로파일기울기 {m['profile_slope']:+.3f}  "
                  f"FiLM시간코사인 {m['film_temporal_cos']:.3f}  → G2 {g2}", flush=True)
            # (a)(d) 는 잔차가 깨어나기 전에는 무의미하다.
            # zero-init 이라 wake 전에는 FiLM 출력도 잔차도 ~0 이고, 그러면
            # 시간코사인이 1 에 붙고 프로파일 기울기가 0 이 된다 — 붕괴가 아니라 초기화다
            if wake_step is not None:
                if m["film_temporal_cos"] > G.FILM_TEMPORAL_COS_MAX:
                    print("  ⚠ (a) 16프레임이 하나로 무너지는 중")
                if m["profile_slope"] <= G.PROFILE_SLOPE_MIN:
                    print("  ⚠ (d) 잔차 프로파일이 평평하다 — 첫 프레임만 다시 그리는 중일 수 있다")
            if G.check_g2_5(dcos_hist):
                print("  ⭐ G2.5 충족 — 선명도 항(λ_f) 점화 조건 성립")
            p = save_viz(model, val, dev, step + 1, vizdir)
            print(f"  [그림] {p.name}", flush=True)
            # 모델만 = 120MB, 옵티마이저 포함 = 367MB.
            # 촘촘히 남기되(CLAUDE.md) 디스크를 낭비하지 않는다. 재개용 full 은 5회마다
            torch.save({"model": model.state_dict(), "step": step + 1,
                        "args": vars(args), "monitor": m},
                       ckdir / f"ck_{step+1:06d}.pt")
            if len(hist) % 5 == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step + 1, "args": vars(args)},
                           ckdir / f"full_{step+1:06d}.pt")
            json.dump({"history": hist, "wake_step": wake_step,
                       "gates": {k: v["when"] for k, v in G.GATES.items()}},
                      open(ckdir / "history.json", "w"), indent=2, ensure_ascii=False)

    # G1 판정
    print("\n=== G1 판정 (gates.py 등록값) ===")
    if wake_step is None:
        print(f"  ❌ 실패 — {args.steps} 스텝까지 wake 없음. 학습률·초기화 재검토")
    else:
        print(f"  ✅ 통과 — wake step {wake_step}. G2 는 wake+4,000 = {wake_step+4000} 에서")


if __name__ == "__main__":
    main()
