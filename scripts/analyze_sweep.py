"""모션 스윕 결과 심층 분석 — β* 이론값 대조, oracle 상한, 표본별 승자 분포.

무엇을 답하나:
 1) 픽셀 L2 이론 최적 스케일 β*≈0.15 근처(motion_b0.10/0.20)가 실제로 static을 넘는가?
 2) 표본마다 최적 변형을 고를 수 있다면(oracle) 얼마나 좋아지는가? = 생성 정보의 상한
 3) 어떤 표본에서 생성이 유용한가(균일한가, 소수 표본만인가)
 4) 성분별로 어느 지표가 이득/손해를 만드나(DINO vs Video vs Action)

입력: results/motion_sweep/motion_sweep_report.json (run_motion_sweep.py 산출물)
출력: results/motion_sweep/sweep_analysis.json + 콘솔
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATIC_REF = 0.56032   # M0 n=96 static TOTAL
GT_REF = 0.48911       # M0 n=96 GT TOTAL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="results/motion_sweep/motion_sweep_report.json")
    ap.add_argument("--out", default="results/motion_sweep/sweep_analysis.json")
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    res = rep["results"]
    names = list(res.keys())
    sids = [r["sid"] for r in res[names[0]]["rows"]]

    # 변형별 평균 TOTAL 정렬
    means = {n: res[n]["mean"]["total_frame_avg"] for n in names}
    order = sorted(names, key=lambda n: means[n])
    best = order[0]

    # 끝점(재인코딩 기준): blend_a0.00=baseline, blend_a1.00=static
    ep_base = means.get("blend_a0.00")
    ep_static = means.get("blend_a1.00")

    # 표본×변형 TOTAL 행렬
    M = np.array([[r["total_frame_avg"] for r in res[n]["rows"]] for n in names])  # (V, N)
    static_idx = names.index("blend_a1.00") if "blend_a1.00" in names else None
    oracle_total = float(M.min(axis=0).mean())
    oracle_winner = [names[i] for i in M.argmin(axis=0)]
    win_counts: dict[str, int] = {}
    for w in oracle_winner:
        win_counts[w] = win_counts.get(w, 0) + 1

    # static 대비 표본별 승패 (동일 표본 paired 비교)
    per_variant_vs_static = {}
    if static_idx is not None:
        s = M[static_idx]
        for i, n in enumerate(names):
            d = M[i] - s
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            per_variant_vs_static[n] = {
                "mean_delta": float(d.mean()),
                "se": se,
                "t_stat": float(d.mean() / se) if se > 0 else float("nan"),
                "frac_better": float(np.mean(d < 0)),
            }

    # 성분별 (best와 static 비교)
    def comp(n):
        m = res[n]["mean"]
        return {k: m[k] for k in ["dino_frame_avg", "video", "action", "total_frame_avg"]}

    analysis = {
        "n_samples": len(sids),
        "reference": {"static_M0": STATIC_REF, "gt_M0": GT_REF,
                      "endpoint_baseline_reencoded": ep_base, "endpoint_static_reencoded": ep_static},
        "ranking": [{"name": n, "total": means[n], "motion": res[n]["motion_mean"],
                     "delta_vs_static_endpoint": (means[n] - ep_static) if ep_static else None}
                    for n in order],
        "best_variant": {"name": best, **comp(best)},
        "static_endpoint": comp("blend_a1.00") if "blend_a1.00" in res else None,
        "oracle": {"total": oracle_total,
                   "gain_vs_static_endpoint": (oracle_total - ep_static) if ep_static else None,
                   "winner_counts": dict(sorted(win_counts.items(), key=lambda kv: -kv[1]))},
        "vs_static_paired": per_variant_vs_static,
    }
    Path(args.out).write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"스윕 심층 분석 (n={len(sids)})   기준: static(재인코딩) {ep_static:.5f} / M0 static {STATIC_REF:.5f}")
    print("=" * 100)
    print(f"{'variant':<18}{'motion':>8}{'TOTAL':>10}{'Δ vs static':>13}{'t':>8}{'승률':>8}")
    print("-" * 100)
    for n in order:
        v = per_variant_vs_static.get(n, {})
        print(f"{n:<18}{res[n]['motion_mean']:>8.2f}{means[n]:>10.5f}"
              f"{v.get('mean_delta', float('nan')):>+13.5f}{v.get('t_stat', float('nan')):>8.2f}"
              f"{v.get('frac_better', float('nan'))*100:>7.0f}%")
    print("-" * 100)
    print(f"\n[최적 변형] {best}: TOTAL {means[best]:.5f}  (static 대비 {means[best]-ep_static:+.5f})")
    b = analysis["best_variant"]
    s = analysis["static_endpoint"]
    if s:
        print(f"  성분 비교  DINO {s['dino_frame_avg']:.5f} → {b['dino_frame_avg']:.5f} "
              f"({b['dino_frame_avg']-s['dino_frame_avg']:+.5f})")
        print(f"             Video {s['video']:.5f} → {b['video']:.5f} ({b['video']-s['video']:+.5f})")
        print(f"             Action {s['action']:.5f} → {b['action']:.5f} ({b['action']-s['action']:+.5f})")
    print(f"\n[oracle 상한] 표본별 최적 변형 선택 시 TOTAL {oracle_total:.5f} "
          f"(static 대비 {oracle_total-ep_static:+.5f}) — 정답을 봐야 하므로 실제 제출엔 사용 불가")
    print("  승자 분포:", ", ".join(f"{k}:{v}" for k, v in list(analysis["oracle"]["winner_counts"].items())[:8]))
    print(f"\n[분석] 저장: {args.out}")


if __name__ == "__main__":
    main()
