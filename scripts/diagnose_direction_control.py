"""모션 방향 상관의 적대적 검증 — shuffle 대조군 + 이론 최적 스케일 β*.

앞선 진단에서 baseline 변위와 GT 변위의 코사인이 +0.167(양수 88.5%)로 나왔다.
그런데 이 값이 "이 샘플의 미래를 실제로 맞춘 것"인지, 아니면
"로봇 팔은 대개 화면 중앙 근처에서 움직인다"는 **일반적 사전(prior)** 때문에 생기는
구조적 겹침인지 구분해야 한다. 그래서 두 가지를 계산한다:

  (1) matched  : cos(ΔB_i, ΔG_i)      같은 샘플끼리 (앞선 진단과 동일)
  (2) shuffled : cos(ΔB_i, ΔG_j), j≠i  다른 샘플의 정답과 비교 (대조군)

  matched ≈ shuffled  → 상관은 샘플 특이 정보가 아님(= 예측력 없음)
  matched >  shuffled  → 초과분만이 진짜 예측 정보

또한 픽셀 L2 관점의 이론 최적 스케일을 낸다:
  P_t = I0 + β·ΔB_t 로 G_t를 맞출 때  β* = <ΔB, ΔG> / ||ΔB||²
  (matched 기준 β*와, shuffled 기준 β*를 함께 보고 초과분을 판단)

출력: results/motion_sweep/direction_control.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from diagnose_motion_direction import read_mp4_uint8, resize_to  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="artifacts/holdout")
    ap.add_argument("--baseline-dir", default="artifacts/baseline_preds")
    ap.add_argument("--out", default="results/motion_sweep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.limit:
        samples = samples[: args.limit]
    n = len(samples)

    # 변위 벡터를 메모리에 담기 위해 공간을 축소(4x4 평균 풀링) — 방향 통계에는 충분
    def disp(path: Path, ref_hw=None) -> tuple[np.ndarray, tuple[int, int]]:
        a = read_mp4_uint8(path).astype(np.float32)
        if ref_hw is not None:
            a = resize_to(a, ref_hw)
        hw = (a.shape[1], a.shape[2])
        # 4x4 평균 풀링
        h4, w4 = hw[0] // 4 * 4, hw[1] // 4 * 4
        a = a[:, :h4, :w4].reshape(a.shape[0], h4 // 4, 4, w4 // 4, 4, 3).mean(axis=(2, 4))
        d = (a[1:] - a[0:1]).reshape(15, -1)
        return d, hw

    dBs, dGs = [], []
    for i, s in enumerate(samples):
        sid = s["sid"]
        dB, hw = disp(Path(args.baseline_dir) / f"{sid}.mp4")
        dG, _ = disp(holdout / "gt_videos" / f"{sid}.mp4", ref_hw=hw)
        dBs.append(dB)
        dGs.append(dG)
        if (i + 1) % 32 == 0:
            print(f"[ctrl] load {i+1}/{n}")

    def cosv(u, v, eps=1e-8):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        return float(np.dot(u, v) / (nu * nv)) if nu > eps and nv > eps else np.nan

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    # 자기 자신과 짝지어지지 않도록 보정
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    matched, shuffled = [], []
    beta_matched, beta_shuffled = [], []
    for i in range(n):
        b = dBs[i].reshape(-1)
        g = dGs[i].reshape(-1)
        gs = dGs[perm[i]].reshape(-1)
        matched.append(cosv(b, g))
        shuffled.append(cosv(b, gs))
        denom = float(np.dot(b, b))
        if denom > 1e-8:
            beta_matched.append(float(np.dot(b, g) / denom))
            beta_shuffled.append(float(np.dot(b, gs) / denom))

    matched = np.array(matched)
    shuffled = np.array(shuffled)
    bm = np.array(beta_matched)
    bs = np.array(beta_shuffled)

    # 짝지은 차이의 통계 (paired)
    diff = matched - shuffled
    se = float(np.nanstd(diff, ddof=1) / np.sqrt(np.sum(~np.isnan(diff))))
    t_stat = float(np.nanmean(diff) / se) if se > 0 else float("nan")

    report = {
        "n_samples": n,
        "matched_cosine": {"mean": float(np.nanmean(matched)), "median": float(np.nanmedian(matched)),
                           "std": float(np.nanstd(matched))},
        "shuffled_cosine": {"mean": float(np.nanmean(shuffled)), "median": float(np.nanmedian(shuffled)),
                            "std": float(np.nanstd(shuffled))},
        "paired_diff": {"mean": float(np.nanmean(diff)), "se": se, "t_stat": t_stat,
                        "frac_matched_gt_shuffled": float(np.nanmean(matched > shuffled))},
        "beta_star_pixelL2": {"matched_mean": float(np.nanmean(bm)), "matched_median": float(np.nanmedian(bm)),
                              "shuffled_mean": float(np.nanmean(bs))},
        "note": "4x4 평균풀링 공간에서 계산. beta* = <dB,dG>/||dB||^2 (픽셀 L2 최적 스케일).",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "direction_control.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"방향 상관 적대 검증 — matched vs shuffled 대조 (n={n})")
    print("=" * 84)
    print(f"  matched  cos : {report['matched_cosine']['mean']:+.4f} (중앙 {report['matched_cosine']['median']:+.4f})")
    print(f"  shuffled cos : {report['shuffled_cosine']['mean']:+.4f} (중앙 {report['shuffled_cosine']['median']:+.4f})")
    d = report["paired_diff"]
    print(f"  차이(paired) : {d['mean']:+.4f}  SE {d['se']:.4f}  t={d['t_stat']:+.2f}  "
          f"matched 우세 표본 {d['frac_matched_gt_shuffled']*100:.1f}%")
    b = report["beta_star_pixelL2"]
    print(f"\n  픽셀 L2 최적 스케일 β*: matched {b['matched_mean']:.3f} (중앙 {b['matched_median']:.3f}) "
          f"vs shuffled {b['shuffled_mean']:.3f}")
    print("=" * 84)
    if abs(d["t_stat"]) < 2:
        print("판정: matched ≈ shuffled — 방향 상관은 **샘플 특이 정보가 아니다**(일반적 팔 위치 prior).")
    else:
        print("판정: matched > shuffled 유의 — 약하지만 **샘플 특이 예측 정보가 존재**한다.")
    print(f"      → 이론 최적 스케일이 {b['matched_median']:.2f} 수준이면, 스윕의 motion_b 세밀구간과 대조하라.")
    print(f"\n[ctrl] 리포트: {out / 'direction_control.json'}")


if __name__ == "__main__":
    main()
