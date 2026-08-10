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

REPO = Path(__file__).resolve().parents[2]   # 상대경로 (대회 §3.3 요건)
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
from loader_c import (EpisodeWindowStream, holdout_episode_refs, list_train_episodes,
                      load_holdout_val96, preprocess_batch, WINDOW)      # noqa: E402
from model_c import ResidualSimVPC, C, H, W, T                           # noqa: E402
import gates as G                                                        # noqa: E402


# ═══════════════════════ 손실 ═══════════════════════

def residual_cos_loss(out, gt, first):
    """1 − 코사인(예측 잔차, 정답 잔차). 프레임 1~15 (0번은 구조적으로 0).

    008 §8-② 1순위. 근거: 008 §5 + 007 후속 눈금 보정 —
      정답 잔차를 k배 뭉갠 것의 코사인 ↔ 018 DV 감축이
      0.971→48.3% · 0.930→16.1% · 0.865→−1.8%(static 보다 나쁨).
      목표 40.4% 에 필요한 코사인 ≈ 0.96 인데 L1 단독 실측이 0.12 다.

    ⚠ Goodhart: 이걸 손실에 넣으면 **로컬 코사인은 더 이상 진단이 아니다.**
      오염되지 않은 판정은 리더보드 λ 스윕(008 §9-b)이 맡는다. 역할을 분리한다.
    """
    pr = (out - first.unsqueeze(2))[:, :, 1:].reshape(out.shape[0], -1)
    gr = (gt - first.unsqueeze(2))[:, :, 1:].reshape(out.shape[0], -1)
    return (1.0 - F.cosine_similarity(pr.float(), gr.float(), dim=1)).mean()


def tau_diff_div_reg(out, gt, tau=0.1, eps=1e-12):
    """TAU 자체 손실 L_reg — 시간 차분 분포의 KL.

    [코드] OpenSTL methods/tau.py:22-31 을 우리 축 (B,C,T,H,W) 에 맞게 옮긴 것.
    참조 alpha = 0.1 (bair·kitticaltech·taxibj·mmnist·kinetics 전 config 공통).
    002 §2.3 이 "정지영상 함정을 억제할 여지"라 가설을 세웠고 005 중-8 이 소멸을 지적했다.
    """
    p = out.permute(0, 2, 1, 3, 4)                    # (B,T,C,H,W)
    g = gt.permute(0, 2, 1, 3, 4)
    B, Tn = p.shape[:2]
    if Tn <= 2:
        return out.new_zeros(())
    gp = (p[:, 1:] - p[:, :-1]).reshape(B, Tn - 1, -1).float()
    gb = (g[:, 1:] - g[:, :-1]).reshape(B, Tn - 1, -1).float()
    sp = F.softmax(gp / tau, -1)
    sb = F.softmax(gb / tau, -1)
    return (sp * torch.log(sp / (sb + eps) + eps)).mean()


@torch.enable_grad()
def calibrate_lambda(model, out, gt, first, target_ratio=0.25):
    """λ_c 를 **1회 측정해 고정**한다 (005 치-7 / 001 §4.5 — PCGrad 배제).

    디코더 마지막 층(readout) gradient norm 비율로 맞춘다:
      λ_c 를 곱한 방향항의 grad norm 이 L1 항의 target_ratio 배가 되게.
    """
    w = model.net.dec.readout.weight
    l1 = (out - gt).abs().mean()
    lc = residual_cos_loss(out, gt, first)
    g1 = torch.autograd.grad(l1, w, retain_graph=True)[0].float().norm()
    gc = torch.autograd.grad(lc, w, retain_graph=True)[0].float().norm()
    lam = float(target_ratio * g1 / (gc + 1e-12))
    return lam, float(g1), float(gc)


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
    # ⚠ 프레임 공통(DC) 성분을 빼고 잰다. 안 빼면 DC 가 지배해 코사인이 구조적으로 1 에 붙는다
    #   [측정] 2026-08-08 step2000: 원본 0.990 vs DC제거 0.688, DC 가 변동분의 9.3배.
    #   004 §4.2(a)의 의도는 "프레임 **사이**의 차이"이고 DC 는 그 차이와 무관하다
    ac = gb - gb.mean(1, keepdim=True)
    gn = F.normalize(ac, dim=-1)
    out["film_temporal_cos"] = (gn[:, :-1] * gn[:, 1:]).sum(-1).mean().item()
    gn0 = F.normalize(gb, dim=-1)
    out["film_temporal_cos_raw"] = (gn0[:, :-1] * gn0[:, 1:]).sum(-1).mean().item()
    out["film_dc_ratio"] = (gb.mean(1).norm(dim=-1).mean()
                            / (ac.norm(dim=-1).mean() + 1e-8)).item()

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
            # t=0 은 정답 잔차가 구조적으로 0 이라 per-frame vmax 가 퇴화한다.
            # 표본 단위 전역 vmax 를 쓴다 (프레임 간 비교도 가능해진다)
            vmax = max(float((v[i] - f[i].unsqueeze(1)).abs().mean(0).max()), 1e-6)
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
    ap.add_argument("--wake-step", type=int, default=None,
                    help="이미 관측된 wake 스텝을 재개 런에 복원한다. "
                         "체크포인트가 wake_step 을 안 싣기 때문에 필요하다 — 안 주면 "
                         "재개 런이 첫 감시 시점을 wake 로 재판정해 G2 발동 시점이 뒤로 밀린다")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="gradient checkpointing 을 끈다. 96GB 기계에서 속도를 사는 용도 "
                         "(3.63 vs 4.76 s/step = ×1.31, 31.8 vs 19.6 GiB — 011 §2 격자). "
                         "⚠ 수학적으로 동일하지 않다: ckpt 는 backward 에서 forward 를 다시 "
                         "돌려 BatchNorm 러닝통계를 micro-step 당 2회 갱신한다(3스텝에 "
                         "num_batches_tracked 6 vs 3). 런 중간에 켰다 끄면 BN 이력이 끊긴다")
    ap.add_argument("--no-viz", action="store_true",
                    help="감시 시점마다 저장하는 pred_*/resid_*.png 를 만들지 않는다. "
                         "사용자가 안 본다고 확인했고 런당 76MB 가 쌓인다 (2026-08-10). "
                         "체크포인트·history.json·로그는 그대로 남으므로 판정에는 영향이 없다")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--monitor-every", type=int, default=None,
                    help="기본값은 gates.MONITOR_EVERY. 스모크에서만 줄인다")
    ap.add_argument("--dir-loss", action="store_true",
                    help="방향(잔차 코사인) 항을 켠다. 008 §8-② 1순위")
    ap.add_argument("--dir-ratio", type=float, default=0.25,
                    help="방향항 grad norm 을 L1 의 이 비율로 맞춘다 (005 치-7)")
    ap.add_argument("--cal-step", type=int, default=1000,
                    help="이 스텝에서 λ_c 를 1회 측정해 고정한다")
    ap.add_argument("--lam-c", type=float, default=None,
                    help="λ_c 를 재측정하지 않고 이 값으로 고정한다. 재개용. "
                         "005 치-7 은 'λ 를 1회 측정해 고정'을 처방했는데, 재개할 때마다 "
                         "다시 재면 이어붙인 런의 손실이 조금씩 달라진다 — 같은 런의 연속이 "
                         "아니게 된다. 이어가는 런에는 원래 값을 그대로 넘겨라")
    ap.add_argument("--tau-alpha", type=float, default=0.0,
                    help="TAU L_reg 가중. OpenSTL 전 config 참조값은 0.1 (005 중-8)")
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
    print(f"[loss] L1"
          + (f" + λ_c·(1−코사인)  [λ_c 는 step {args.cal_step} 에서 자동 교정, 목표비 {args.dir_ratio}]"
             if args.dir_loss else "")
          + (f" + {args.tau_alpha}·L_reg(TAU)" if args.tau_alpha > 0 else "")
          + ("   ⚠ 방향항을 켜면 로컬 코사인은 진단이 아니다 — 판정은 리더보드 λ스윕"
             if args.dir_loss else ""))

    seed = args.seed + (1 if args.resume else 0)      # 016 §6: 재개 시 seed+1
    torch.manual_seed(seed); np.random.seed(seed)

    eps = list_train_episodes(exclude=holdout_episode_refs())
    val = load_holdout_val96()
    print(f"[data] 학습 에피소드 {len(eps)} · 평가 {len(val)}\n")

    model = ResidualSimVPC(hid_S=args.hid_s, use_ckpt=not args.no_ckpt).to(dev)
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

    hist, dcos_hist, wake_step = [], [], args.wake_step
    if wake_step is not None:
        print(f"[wake] 관측된 wake step {wake_step} 복원 → G2 발동 {wake_step + 4000}")
    lam_c = args.lam_c   # 방향항 가중. None 이면 cal_step 에서 1회 측정 후 고정
    if lam_c is not None:
        print(f"[λ_c] 재측정 안 함 — 넘겨받은 값 {lam_c:.4g} 로 고정")
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
                loss = (out - v).abs().mean()                        # L1
                if args.tau_alpha > 0:
                    loss = loss + args.tau_alpha * tau_diff_div_reg(out, v)
                if args.dir_loss:
                    if lam_c is None:                                # 아직 미교정
                        if step >= args.cal_step:
                            lam_c, g1_, gc_ = calibrate_lambda(
                                model, out, v, first, args.dir_ratio)
                            print(f"  ⭐ λ_c 고정 = {lam_c:.4g}  "
                                  f"(readout grad norm  L1 {g1_:.3e} · dir {gc_:.3e}, "
                                  f"목표비 {args.dir_ratio})", flush=True)
                    else:
                        loss = loss + lam_c * residual_cos_loss(out, v, first)
                # + lam_f * frequency_loss(out, v)                   ← G2.5 점화 후
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
            # G2 는 wake+4,000 에 **발동**한다. 그 전 표시는 진행 상태일 뿐이다
            g2_due = wake_step is not None and (step + 1) >= wake_step + 4000
            g2_tag = f"G2 {g2}" + ("  ⬅ 게이트 발동" if g2_due else " (경과 표시)")
            print(f"  [감시 {step+1}] 잔차비 {m['resid_ratio']:.3f}  "
                  f"코사인 {m['resid_cos']:+.3f}(뒤섞기 {m['resid_cos_shuf']:+.3f})  "
                  f"Δcos {m['dcos']:+.3f}  ρ중앙값 {m['rho_median']:+.2f}  "
                  f"프로파일기울기 {m['profile_slope']:+.3f}  "
                  f"FiLM시간코사인 {m['film_temporal_cos']:.3f}  → {g2_tag}", flush=True)
            if g2_due and not m.get("_g2_done"):
                print(f"  ⭐ G2 판정 (wake {wake_step} + 4000): **{g2}**  — "
                      f"{G.GATES['G2'][g2 if g2 in ('pass','fail') else 'tie']}", flush=True)
                m["_g2_done"] = True
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
            if not args.no_viz:
                p = save_viz(model, val, dev, step + 1, vizdir)
                print(f"  [그림] {p.name}", flush=True)
            # 모델만 = 120MB, 옵티마이저 포함 = 367MB.
            # 재개용 full 은 **2회마다**(=1,000스텝). [정정 2026-08-09]
            #   원래 5회마다(=2,500스텝)였다. 그 간격 때문에 dir 팔이 step 7,000 에 끝났는데
            #   재개점이 full_005000 하나뿐이라 이어가려면 2,000 스텝을 버려야 했다.
            #   010 §1 이 기록한 "1,350 스텝을 버렸다"와 같은 구조이고,
            #   CLAUDE.md("저장 분기는 넓게 하지 말고 촘촘하게 — 중간에 끊어도 안 날아가게")와
            #   정면으로 어긋난다. 비용은 디스크뿐이다: 30k 풀런에서 60회 → 22GB (여유 1.3T).
            torch.save({"model": model.state_dict(), "step": step + 1,
                        "args": vars(args), "monitor": m},
                       ckdir / f"ck_{step+1:06d}.pt")
            if len(hist) % 2 == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step + 1, "args": vars(args)},
                           ckdir / f"full_{step+1:06d}.pt")
            json.dump({"history": hist, "wake_step": wake_step, "lam_c": lam_c,
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
