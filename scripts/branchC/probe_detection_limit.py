#!/usr/bin/env python
"""
③ 후속 — 검출 한계 측정 (규율 2 이행)

왜 하나
  ③(action_signal_20260808_1427)은 "행동 신호 미검출 / 양성대조는 작동"으로 끝났다.
  그런데 **양성대조의 이동 크기 0.15 를 임의로 골랐다.** 결론이 그 한 점에 걸려 있다.
  규율 2: "임의로 고른 파라미터에 결론이 걸리면 그 파라미터를 두 점 이상 재라."

판정선 (측정 전 등록)
  검출됨  ⟺ 평균 잔차 코사인 > 0.1464   (③이 확보한 귀무 상한 S max)
  한계 s* = 검출된 것 중 가장 작은 이동 크기
  비교량 = **잔차 RMS** = |정답영상 − 첫프레임|.  회귀기가 예측해야 할 변화의 양.
    R_real > R(s*)  →  크기는 충분한데 행동으로 예측이 안 된다.  신호 문제
    R_real < R(s*)  →  실제 변화가 검출 한계 아래다.  장치 문제 — ③ 미검출은 해석 불가

주의
  PC 는 **행동으로 100% 결정되는** 잔차다. 그래서 이 스윕이 주는 것은
  "행동이 완벽히 설명하는 잔차라면 이 크기까지 잡는다"는 상한이다.
  실제 데이터가 그 크기 이상인데 안 잡혔다면 원인은 크기가 아니라 예측 가능성이다.
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
                      load_holdout_val96, preprocess_batch, WINDOW)  # noqa: E402
from probe_action_signal import (ACT_DIM, PH, PW, ProbeNet, make_act,
                                 residual_cos)  # noqa: E402

NULL_MAX = 0.1464          # ③ 이 확보한 귀무 상한 (S팔 3seed 최대)


def synth_scaled(first, act, s):
    """③의 synth_positive 와 같되 이동 크기를 s 로 준다."""
    B, C, H, W = first.shape
    cum = act[:, :, :2].cumsum(1)
    cum = cum / (cum.abs().amax(dim=(1, 2), keepdim=True) + 1e-6) * s
    out = []
    for t in range(WINDOW):
        th = torch.zeros(B, 2, 3, device=first.device, dtype=first.dtype)
        th[:, 0, 0] = 1; th[:, 1, 1] = 1
        th[:, 0, 2] = cum[:, t, 0]; th[:, 1, 2] = cum[:, t, 1]
        g = F.affine_grid(th, (B, C, H, W), align_corners=False)
        out.append(F.grid_sample(first, g, align_corners=False, padding_mode="border"))
    return torch.stack(out, 2)


def resid_rms(video, first):
    """|정답영상 − 첫프레임| 의 RMS. 프레임 1~15 (0번은 구조적으로 0)."""
    r = (video - first.unsqueeze(2))[:, :, 1:]
    return r.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()


def to_probe(v):
    return F.interpolate(v.flatten(0, 1), size=(PH, PW), mode="bilinear",
                         align_corners=False).view(v.shape[0], 3, WINDOW, PH, PW)


def run(s, eps, val, steps, batch, lr, seed=0):
    """s=None 이면 실제 데이터(대조). 아니면 이동 크기 s 의 합성 데이터."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    model = ProbeNet(ACT_DIM["delta"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dl = torch.utils.data.DataLoader(
        EpisodeWindowStream(eps, seed=seed, span_frames=96, windows_per_span=8),
        batch_size=batch, num_workers=8, pin_memory=True,
        prefetch_factor=4, persistent_workers=True)
    it = iter(dl)
    t0 = time.perf_counter()
    for step in range(steps):
        b = next(it)
        v = to_probe(preprocess_batch(b["frames_u8"].to(dev, non_blocking=True)))
        a = b["act"].to(dev)
        first = v[:, :, 0]
        if s is not None:
            v = synth_scaled(first, a, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(first, make_act(a, "delta")) - v).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{steps} loss {loss.item():.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    del dl, it

    model.eval(); cos, rms = [], []
    with torch.no_grad():
        for i in range(0, len(val), 4):
            ch = val[i:i + 4]
            v = to_probe(torch.stack([c["video"] for c in ch]).to(dev))
            a = torch.stack([c["act"] for c in ch]).to(dev)
            first = v[:, :, 0]
            if s is not None:
                v = synth_scaled(first, a, s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(first, make_act(a, "delta"))
            cos.append(residual_cos(out.float(), v.float(), first.float()).cpu())
            rms.append(resid_rms(v.float(), first.float()).cpu())
    del model, opt; torch.cuda.empty_cache()
    return torch.cat(cos), torch.cat(rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.15, 0.05, 0.02, 0.01, 0.005])
    args = ap.parse_args()

    print(f"[좌표] {PH}×{PW} · 스텝 {args.steps} · 배치 {args.batch}")
    print(f"[판정선] 검출됨 ⟺ 평균 코사인 > {NULL_MAX} (③ 귀무 상한)\n")
    eps = list_train_episodes(exclude=holdout_episode_refs())
    val = load_holdout_val96()

    rows = []
    for s in args.scales:
        print(f"[PC s={s}] 합성 — 행동으로 100% 결정되는 잔차", flush=True)
        c, r = run(s, eps, val, args.steps, args.batch, args.lr)
        det = c.mean().item() > NULL_MAX
        rows.append({"scale": s, "cos": c.mean().item(), "rms": r.mean().item(),
                     "detected": det})
        print(f"    → 코사인 {c.mean():.4f}  잔차RMS {r.mean():.4f}  "
              f"{'✅ 검출' if det else '❌ 미검출'}\n", flush=True)

    print("[REAL] 실제 데이터 (③ T-delta 재현 + 잔차 RMS 측정)", flush=True)
    cr, rr = run(None, eps, val, args.steps, args.batch, args.lr)
    print(f"    → 코사인 {cr.mean():.4f}  잔차RMS {rr.mean():.4f}\n", flush=True)

    print("=" * 64)
    print("판정 (판정선은 측정 전에 등록했다)")
    print("=" * 64)
    print(f"{'이동크기':>8s} {'잔차RMS':>9s} {'코사인':>9s}  검출")
    for x in rows:
        print(f"{x['scale']:8.3f} {x['rms']:9.4f} {x['cos']:9.4f}  "
              f"{'✅' if x['detected'] else '❌'}")
    print(f"{'실제':>8s} {rr.mean():9.4f} {cr.mean():9.4f}  "
          f"{'✅' if cr.mean().item() > NULL_MAX else '❌'}")

    det = [x for x in rows if x["detected"]]
    verdict = {}
    if not det:
        print("\n⚠ 어떤 이동 크기에서도 검출 실패 — 장치가 무능하다. ③ 해석 불가")
        verdict = {"conclusion": "device_incompetent"}
    else:
        star = min(det, key=lambda x: x["scale"])
        print(f"\n검출 한계 s* = {star['scale']}  (그때 잔차 RMS R* = {star['rms']:.4f})")
        R_real = rr.mean().item()
        print(f"실제 데이터 잔차 RMS R_real = {R_real:.4f}")
        if R_real > star["rms"]:
            print(f"\n⇒ R_real({R_real:.4f}) > R*({star['rms']:.4f})")
            print("   **크기는 충분한데 행동으로 예측이 안 된다 — 신호 문제다.**")
            print("   ③ 의 미검출은 장치 탓이 아니다.")
            verdict = {"conclusion": "signal_problem"}
        else:
            print(f"\n⇒ R_real({R_real:.4f}) < R*({star['rms']:.4f})")
            print("   **실제 변화가 우리 장치의 검출 한계 아래다 — 장치 문제다.**")
            print("   ③ 의 미검출을 '신호 없음'으로 읽으면 안 된다.")
            verdict = {"conclusion": "device_limit"}
        verdict.update({"s_star": star["scale"], "R_star": star["rms"], "R_real": R_real})

    out = {"generated": datetime.now().isoformat(timespec="seconds"),
           "null_max": NULL_MAX, "probe_hw": [PH, PW], "steps": args.steps,
           "sweep": rows, "real": {"cos": cr.mean().item(), "rms": rr.mean().item()},
           "verdict": verdict}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
