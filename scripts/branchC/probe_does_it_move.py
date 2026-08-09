#!/usr/bin/env python
"""
"팔이 진짜 옮겨지는가, 제자리에서 흐려지기만 하는가" — 무게중심으로 가린다

왜 필요한가
  사용자가 영상을 보고 "투명해지기만 하고 이동을 안 한다"고 했다. 011 §4 는 반대로
  "불투명도 50~65% 로 움직인다"고 적었는데, 그 근거(cos⁺ > cos⁻)에는 **구멍이 있다**:
  옛 자리의 팔을 흐리게 뭉개기만 해도 그 **가장자리에 양(+) 잔차**가 생긴다.
  즉 cos⁺ 가 높은 것이 "새 자리에 그렸다"의 증거가 못 된다. 다시 재야 한다.

통계량 — 잔차 |r| 의 **무게중심이 프레임을 따라 어디로 가는가**
  com(t) = Σ|r_t|·(x,y) ÷ Σ|r_t|
  d = com(마지막 프레임) − com(첫 프레임)     ← "얼마나, 어느 쪽으로 옮겨갔나"
  제자리에서 흐려지기만 하면 d ≈ 0 이다. 흐림은 대칭이라 무게중심을 안 옮긴다.
  실제로 옮겨가면 정답의 d 를 따라간다.

판정 (사전 등록, 결과 보기 전에 박는다)
  이동비 ‖d_예측‖ / ‖d_정답‖ ≥ 0.5   AND   방향 cos(d_예측, d_정답) ≥ 0.5
    → "실제로 이동한다"
  둘 중 하나라도 미달 → **"제자리에서 흐려진다"가 맞다. 011 §4 를 정정한다**

🚨 킷 미사용(순수 torch). GPU 0 만.

사용:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/branchC/probe_does_it_move.py \
      --ckpt artifacts/branchC/train_20260809_1205_long/ck_008500.pt
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
sys.path.insert(0, str(REPO / "src"))
from model_c import ResidualSimVPC                                    # noqa: E402
from loader_c import load_holdout_val96                               # noqa: E402

CUT_MOVE = 0.5        # 이동비 문턱
CUT_DIR = 0.5         # 방향 코사인 문턱


def com(r: torch.Tensor, top_frac: float | None = None) -> torch.Tensor:
    """|r| 의 무게중심. r (B,C,T,H,W) → (B,T,2) [y,x] 픽셀 좌표.

    top_frac 을 주면 프레임마다 |r| 상위 그 비율의 픽셀만 쓴다.
    화면 전체에 깔린 미세한 배경 변화가 무게중심을 흔드는 것을 막는다 —
    팔은 화면의 일부이므로, 이쪽이 "팔이 어디 있나"에 더 가깝다.
    """
    w = r.abs().mean(1)                                   # (B,T,H,W)
    B, T, H, W = w.shape
    if top_frac is not None:
        k = max(1, int(H * W * top_frac))
        flat = w.flatten(2)
        thr = flat.topk(k, dim=-1).values[..., -1][..., None, None]
        w = w * (w >= thr)
    w = w / (w.flatten(2).sum(-1)[..., None, None] + 1e-8)
    ys = torch.arange(H, device=r.device, dtype=w.dtype)[None, None, :, None]
    xs = torch.arange(W, device=r.device, dtype=w.dtype)[None, None, None, :]
    return torch.stack([(w * ys).flatten(2).sum(-1), (w * xs).flatten(2).sum(-1)], -1)


def probe(ckpt: Path, n: int, dev: str) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ResidualSimVPC(hid_S=ck["args"].get("hid_s", 64), use_ckpt=False).to(dev).eval()
    model.load_state_dict(ck["model"])

    val = load_holdout_val96()[:n]
    firsts = torch.stack([s["first"] for s in val]).to(dev)
    acts = torch.stack([s["act"] for s in val]).to(dev)
    vids = torch.stack([s["video"] for s in val]).to(dev)

    outs = []
    with torch.no_grad():
        for i in range(0, firsts.shape[0], 4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(model(firsts[i:i + 4], acts[i:i + 4]).float())
    pred = torch.cat(outs)

    res_p = (pred - firsts.unsqueeze(2))[:, :, 1:]        # 프레임 0 은 구조적으로 0
    res_g = (vids - firsts.unsqueeze(2))[:, :, 1:]

    def shift(top):
        cg, cp = com(res_g, top), com(res_p, top)         # (B,T,2)
        dg, dp = cg[:, -1] - cg[:, 0], cp[:, -1] - cp[:, 0]
        ng, np_ = dg.norm(dim=1), dp.norm(dim=1)
        return (ng.mean().item(), np_.mean().item(),
                (np_ / (ng + 1e-8)).median().item(),
                torch.nn.functional.cosine_similarity(dp, dg, dim=1).mean().item())

    gs, ps, ratio, dirc = shift(None)
    gs10, ps10, ratio10, dirc10 = shift(0.10)             # 상위 10% 픽셀만

    r = {
        "ckpt": str(ckpt.relative_to(REPO)), "step": ck["step"], "n": len(val),
        "gt_shift_px": gs,
        "pred_shift_px": ps,
        "move_ratio": ratio,
        "dir_cos": dirc,
        "gt_shift_px_top10": gs10,
        "pred_shift_px_top10": ps10,
        "move_ratio_top10": ratio10,
        "dir_cos_top10": dirc10,
        # 참고: 무게중심이 아니라 "퍼짐"이 커지는가 (흐려짐의 지표)
        "gt_spread_growth": _spread_growth(res_g),
        "pred_spread_growth": _spread_growth(res_p),
    }
    r["moves"] = bool(r["move_ratio"] >= CUT_MOVE and r["dir_cos"] >= CUT_DIR)
    return r


def _spread_growth(r: torch.Tensor) -> float:
    """무게중심 둘레의 표준편차가 첫 프레임 대비 마지막 프레임에서 몇 배가 되나."""
    w = r.abs().mean(1)
    B, T, H, W = w.shape
    wn = w / (w.flatten(2).sum(-1)[..., None, None] + 1e-8)
    ys = torch.arange(H, device=r.device, dtype=wn.dtype)[None, None, :, None]
    xs = torch.arange(W, device=r.device, dtype=wn.dtype)[None, None, None, :]
    my = (wn * ys).flatten(2).sum(-1)[..., None, None]
    mx = (wn * xs).flatten(2).sum(-1)[..., None, None]
    var = (wn * ((ys - my) ** 2 + (xs - mx) ** 2)).flatten(2).sum(-1)
    s = var.clamp_min(1e-8).sqrt()
    return (s[:, -1] / (s[:, 0] + 1e-8)).median().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    print(f"판정선(사전 등록): 이동비 ≥ {CUT_MOVE}  AND  방향코사인 ≥ {CUT_DIR}\n")
    rows = []
    for c in a.ckpt:
        r = probe(Path(c).resolve(), a.n, a.device)
        rows.append(r)
        print(f"── {r['ckpt']}  (step {r['step']}, n={r['n']})")
        print(f"   정답 무게중심 이동   {r['gt_shift_px']:6.2f} px")
        print(f"   예측 무게중심 이동   {r['pred_shift_px']:6.2f} px")
        print(f"   이동비              {r['move_ratio']:.3f}   (1.0 이면 정답만큼 옮겨간다)")
        print(f"   방향 코사인          {r['dir_cos']:+.3f}   (1.0 이면 같은 쪽으로)")
        print(f"   [상위10% 픽셀만]  정답 {r['gt_shift_px_top10']:.2f}px · "
              f"예측 {r['pred_shift_px_top10']:.2f}px · "
              f"이동비 {r['move_ratio_top10']:.3f} · 방향 {r['dir_cos_top10']:+.3f}")
        print(f"   퍼짐 증가  정답 {r['gt_spread_growth']:.2f}배 · "
              f"예측 {r['pred_spread_growth']:.2f}배")
        print(f"   ⇒ {'✅ 실제로 이동한다' if r['moves'] else '❌ 제자리에서 흐려진다'}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = REPO / "results" / "branchC"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"does_it_move_{ts}.json"
    json.dump({"cut_move": CUT_MOVE, "cut_dir": CUT_DIR, "rows": rows},
              open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[out] {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
