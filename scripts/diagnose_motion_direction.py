"""모션 '방향' 진단 — baseline의 움직임은 방향이라도 맞는가? (GPU 불필요)

무학습 스윕에서 "움직임을 줄일수록 좋고 완전 정지가 최적"이 나왔다.
가능한 두 해석을 구분해야 한다:
  (H1) 방향은 맞는데 크기가 과하다      → 크기만 줄이면(motion_b) 이득이 나야 한다
  (H2) 방향 자체가 무작위/틀렸다        → 어떤 크기로 줄여도 이득이 없고 정지가 최적

이 스크립트는 픽셀 변위 벡터의 코사인 유사도로 H1/H2를 직접 가른다:
  Δ_B(t) = B_t − B_0   (baseline이 예측한 t시점 변위)
  Δ_G(t) = G_t − G_0   (정답의 실제 변위)
  cos(Δ_B, Δ_G) ≈ 0 이면 무작위(H2), > 0 이면 방향 일치(H1)

부가로 '정답이 실제로 얼마나 움직이는가'(누적 변위 크기)와
baseline 변위 크기의 비율을 프레임별로 낸다.

출력: results/motion_sweep/direction_diag.json + 콘솔 표
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def read_mp4_uint8(path: Path, expect: int = 16) -> np.ndarray:
    import av
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
    a = np.stack(frames, axis=0)
    if a.shape[0] != expect:
        raise ValueError(f"{path}: frames {a.shape[0]} != {expect}")
    return a


def resize_to(a: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """단순 최근접 리사이즈(진단용; 해상도만 맞추면 충분)."""
    h, w = hw
    if a.shape[1] == h and a.shape[2] == w:
        return a
    yi = (np.linspace(0, a.shape[1] - 1, h)).astype(np.int64)
    xi = (np.linspace(0, a.shape[2] - 1, w)).astype(np.int64)
    return a[:, yi][:, :, xi]


def cos(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="artifacts/holdout")
    ap.add_argument("--baseline-dir", default="artifacts/baseline_preds")
    ap.add_argument("--out", default="results/motion_sweep")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.limit:
        samples = samples[: args.limit]

    per_frame_cos = np.zeros((len(samples), 15), dtype=np.float64)
    per_frame_magB = np.zeros((len(samples), 15), dtype=np.float64)
    per_frame_magG = np.zeros((len(samples), 15), dtype=np.float64)
    clip_cos = []

    for i, s in enumerate(samples):
        sid = s["sid"]
        B = read_mp4_uint8(Path(args.baseline_dir) / f"{sid}.mp4").astype(np.float32)   # (16,320,512,3)
        G = read_mp4_uint8(holdout / "gt_videos" / f"{sid}.mp4").astype(np.float32)     # native 해상도
        G = resize_to(G, (B.shape[1], B.shape[2]))                                      # baseline 격자에 맞춤

        dB = (B[1:] - B[0:1]).reshape(15, -1)
        dG = (G[1:] - G[0:1]).reshape(15, -1)
        for t in range(15):
            per_frame_cos[i, t] = cos(dB[t], dG[t])
            per_frame_magB[i, t] = np.abs(dB[t]).mean()
            per_frame_magG[i, t] = np.abs(dG[t]).mean()
        clip_cos.append(cos(dB.reshape(-1), dG.reshape(-1)))
        if (i + 1) % 24 == 0:
            print(f"[diag] {i+1}/{len(samples)}")

    cos_mean_t = np.nanmean(per_frame_cos, axis=0)
    magB_t = per_frame_magB.mean(axis=0)
    magG_t = per_frame_magG.mean(axis=0)
    clip_cos = np.array(clip_cos, dtype=np.float64)

    report = {
        "n_samples": len(samples),
        "clip_level_cosine": {
            "mean": float(np.nanmean(clip_cos)), "median": float(np.nanmedian(clip_cos)),
            "std": float(np.nanstd(clip_cos)),
            "frac_positive": float(np.mean(clip_cos > 0)),
            "frac_above_0p1": float(np.mean(clip_cos > 0.1)),
            "min": float(np.nanmin(clip_cos)), "max": float(np.nanmax(clip_cos)),
        },
        "per_frame": {
            "cosine_mean": [float(x) for x in cos_mean_t],
            "baseline_disp_mean": [float(x) for x in magB_t],
            "gt_disp_mean": [float(x) for x in magG_t],
        },
        "interpretation_hint": "clip cosine ~0 → 방향 무작위(H2); >0.2 → 방향 일치(H1)",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "direction_diag.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"모션 방향 진단 (n={len(samples)}) — baseline 변위 vs GT 변위의 코사인")
    print("=" * 78)
    c = report["clip_level_cosine"]
    print(f"클립 전체 코사인:  평균 {c['mean']:+.4f} | 중앙 {c['median']:+.4f} | 표준편차 {c['std']:.4f}")
    print(f"                   양수 비율 {c['frac_positive']*100:.1f}% | >0.1 비율 {c['frac_above_0p1']*100:.1f}%")
    print(f"                   범위 [{c['min']:+.3f}, {c['max']:+.3f}]")
    print("\n프레임별 (t=1..15):")
    print("  t      :" + "".join(f"{t+1:>6}" for t in range(15)))
    print("  cos    :" + "".join(f"{cos_mean_t[t]:>+6.2f}" for t in range(15)))
    print("  |ΔB|   :" + "".join(f"{magB_t[t]:>6.1f}" for t in range(15)))
    print("  |ΔG|   :" + "".join(f"{magG_t[t]:>6.1f}" for t in range(15)))
    print("=" * 78)
    verdict = ("H2(방향 무작위) — 크기를 어떻게 줄여도 이득 없음, 정지가 최적인 것과 정합"
               if abs(c["mean"]) < 0.1 else
               "H1(방향 일부 일치) — 크기 조절이 이득을 낼 여지 있음")
    print(f"판정: {verdict}")
    print(f"\n[diag] 리포트: {out / 'direction_diag.json'}")


if __name__ == "__main__":
    main()
