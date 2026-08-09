#!/usr/bin/env python
"""
"지우기만 하고 그리지는 않는가" — 잔차를 부호로 쪼개 재는 탐침

왜 필요한가
  영상에서 팔이 **옮겨지지 않고 흐려진다**. 감시 (d)(프레임별 잔차 프로파일 기울기)는
  이걸 못 잡는다 — 페이드아웃도 뒤로 갈수록 커지므로 기울기가 양수다. 실제로 두 팔 모두
  기울기 +0.08~0.10 으로 계속 통과했는데 그림은 죽어 있다.

가설 (003 치-2 · 004 §7-3 이 이미 기록한 병리)
  출력 = 첫프레임 + 잔차 다. 팔을 **옮기려면** 잔차가 둘을 동시에 해야 한다:
    ① 원래 자리에서 뺀다 → 그 모양의 음(−) 잔차     ← 팔이 떠나는 건 확실하다. 쉽다
    ② 새 자리에 그린다   → 그 모양의 양(+) 잔차     ← 몇 픽셀만 어긋나도 L1 이 뭉갠다. 어렵다
  ⇒ L1 최적해가 "확실하게 지우고 소심하게 그린다" 로 수렴하면 화면은 페이드아웃이 된다.

통계량 — 정답 잔차의 **부호로 마스크를 갈라** 각각 따로 잰다
  M⁻ = {정답 잔차 < 0}  "물체가 떠나는 곳"      M⁺ = {정답 잔차 > 0}  "물체가 도착하는 곳"
  cos⁻ · cos⁺   각 마스크 안에서의 잔차 코사인
  amp⁻ · amp⁺   각 마스크 안 |예측| ÷ |정답|      ← 세기를 얼마나 내는가

판정 (사전 등록, 결과 보기 전에 박는다)
  cos⁻ − cos⁺ ≥ 0.10  AND  amp⁻ / amp⁺ ≥ 1.5   →  "지우기 편향" 확인
  둘 다 아니면 페이드아웃의 원인은 이 비대칭이 아니다 — 다른 설명을 찾아야 한다

🚨 킷을 안 쓴다. 순수 torch 다 (rule/001 §13.4 ✅ 목록).
🚨 GPU 0 만. CUDA_VISIBLE_DEVICES=0 으로 실행한다.

사용:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/branchC/probe_erase_vs_draw.py \
      --ckpt artifacts/branchC/train_20260809_0059_dir/ck_007000.pt
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
sys.path.insert(0, str(REPO / "src"))
from model_c import ResidualSimVPC                                    # noqa: E402
from loader_c import load_holdout_val96                               # noqa: E402

CUT_GAP = 0.10        # cos⁻ − cos⁺ 문턱
CUT_AMP = 1.5         # amp⁻ / amp⁺ 문턱


def masked_cos(p: torch.Tensor, g: torch.Tensor, m: torch.Tensor) -> float:
    """마스크 안에서만 잰 코사인. 표본별로 재고 평균한다."""
    pm, gm = p * m, g * m
    c = F.cosine_similarity(pm.flatten(1), gm.flatten(1), dim=1)
    return c.mean().item()


def masked_amp(p: torch.Tensor, g: torch.Tensor, m: torch.Tensor) -> float:
    """마스크 안 |예측| ÷ |정답| (표본별 → 평균)."""
    num = (p.abs() * m).flatten(1).sum(1)
    den = (g.abs() * m).flatten(1).sum(1) + 1e-8
    return (num / den).mean().item()


def probe(ckpt: Path, n: int, dev: str) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    hid = ck["args"].get("hid_s", 64)
    model = ResidualSimVPC(hid_S=hid, use_ckpt=False).to(dev).eval()
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

    res_p = (pred - firsts.unsqueeze(2))[:, :, 1:]     # 프레임 0 은 구조적으로 0
    res_g = (vids - firsts.unsqueeze(2))[:, :, 1:]

    m_neg = (res_g < 0).float()
    m_pos = (res_g > 0).float()
    r = {
        "ckpt": str(ckpt.relative_to(REPO)), "step": ck["step"], "n": len(val),
        "cos_all": F.cosine_similarity(res_p.flatten(1), res_g.flatten(1), dim=1).mean().item(),
        "cos_neg": masked_cos(res_p, res_g, m_neg),
        "cos_pos": masked_cos(res_p, res_g, m_pos),
        "amp_neg": masked_amp(res_p, res_g, m_neg),
        "amp_pos": masked_amp(res_p, res_g, m_pos),
        "frac_neg": m_neg.mean().item(),
    }
    r["cos_gap"] = r["cos_neg"] - r["cos_pos"]
    r["amp_ratio"] = r["amp_neg"] / (r["amp_pos"] + 1e-8)
    r["erase_bias"] = bool(r["cos_gap"] >= CUT_GAP and r["amp_ratio"] >= CUT_AMP)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    print(f"판정선(사전 등록): cos⁻−cos⁺ ≥ {CUT_GAP}  AND  amp⁻/amp⁺ ≥ {CUT_AMP}\n")
    rows = []
    for c in a.ckpt:
        r = probe(Path(c).resolve(), a.n, a.device)
        rows.append(r)
        print(f"── {r['ckpt']}  (step {r['step']}, n={r['n']})")
        print(f"   전체 코사인            {r['cos_all']:+.4f}")
        print(f"   지우는 곳 cos⁻        {r['cos_neg']:+.4f}      "
              f"세기 amp⁻ {r['amp_neg']:.3f}")
        print(f"   그리는 곳 cos⁺        {r['cos_pos']:+.4f}      "
              f"세기 amp⁺ {r['amp_pos']:.3f}")
        print(f"   격차 cos⁻−cos⁺        {r['cos_gap']:+.4f}   "
              f"세기비 amp⁻/amp⁺ {r['amp_ratio']:.2f}")
        print(f"   ⇒ 지우기 편향: {'✅ 확인' if r['erase_bias'] else '❌ 미확인'}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = REPO / "results" / "branchC"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"erase_vs_draw_{ts}.json"
    json.dump({"cut_gap": CUT_GAP, "cut_amp": CUT_AMP, "rows": rows},
              open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[out] {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
