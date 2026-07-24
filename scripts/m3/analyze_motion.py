"""Part A 심층 분석 (CPU 전용, GPU 불필요) — λ별 '움직임 크기'를 측정해
CFG(λ)가 프레임간 변화(모션)를 증폭하는지 정량화한다.

각 예측 mp4의 프레임간 평균 절대 픽셀차(mean |frame_{t+1}-frame_t|)를 96개 평균.
GT(holdout/gt_videos)와 비교. M0: GT 프레임간 변화 2~4, baseline 6.6~9.4.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")
HOLD = REPO / "artifacts/holdout"
PARTA = REPO / "artifacts/m3/partA"
TAG2LAM = {"cfg10": 1.0, "cfg15": 1.5, "cfg20": 2.0, "cfg25": 2.5, "cfg30": 3.0, "cfg40": 4.0}


def frame_to_frame_motion(mp4: Path) -> float:
    v = iio.imread(mp4).astype(np.float32)  # (T,H,W,3)
    if v.shape[0] < 2:
        return 0.0
    d = np.abs(v[1:] - v[:-1]).mean()
    return float(d)


def mean_over_dir(d: Path, sids: list[str]) -> tuple[float, float]:
    vals = []
    for sid in sids:
        f = d / f"{sid}.mp4"
        if f.exists():
            vals.append(frame_to_frame_motion(f))
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def main():
    man = json.loads((HOLD / "manifest.json").read_text())
    sids = [s["sid"] for s in man["samples"]]
    print(f"n={len(sids)}  (프레임간 평균 |Δpixel|; 클수록 움직임 큼. M0: GT 2~4, baseline 6.6~9.4)")
    print("-" * 60)
    gtm, gts = mean_over_dir(HOLD / "gt_videos", sids)
    print(f"{'GT (정답)':<16} motion={gtm:6.3f} ± {gts:5.3f}")
    print("-" * 60)
    rows = []
    for tag in ["cfg10", "cfg15", "cfg20", "cfg25", "cfg30", "cfg40"]:
        d = PARTA / tag
        if not d.exists():
            continue
        m, s = mean_over_dir(d, sids)
        lam = TAG2LAM[tag]
        rows.append((lam, m, s))
        print(f"λ={lam:<4} {tag:<10} motion={m:6.3f} ± {s:5.3f}  (GT 대비 {m/gtm:4.1f}배)")
    print("-" * 60)
    if len(rows) >= 2:
        lams = [r[0] for r in rows]; mots = [r[1] for r in rows]
        mono = all(mots[i] <= mots[i+1] for i in range(len(mots)-1))
        print(f"λ↑에 따라 움직임 단조 증가? {'예' if mono else '아니오'}  "
              f"(λ=1.0 {mots[0]:.2f} → λ=4.0 {mots[-1]:.2f}, +{(mots[-1]/mots[0]-1)*100:.0f}%)")
        # 상관
        import numpy as _np
        r = _np.corrcoef(lams, mots)[0, 1]
        print(f"corr(λ, motion) = {r:+.3f}")
    out = REPO / "results/m3/partA/motion_analysis.json"
    out.write_text(json.dumps({"gt": {"motion_mean": gtm, "motion_std": gts},
                               "sweep": [{"lam": l, "motion_mean": m, "motion_std": s} for l, m, s in rows]},
                              ensure_ascii=False, indent=2))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
