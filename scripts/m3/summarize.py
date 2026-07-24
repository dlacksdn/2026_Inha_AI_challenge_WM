"""M3 λ-스윕 채점 결과(m0_report.json들)를 λ 반응 곡선 표로 요약한다.

사용: python scripts/m3/summarize.py <SCORE_ROOT> [SCORE_ROOT2 ...]
각 SCORE_ROOT 아래 cfg<tag>/m0_report.json 을 읽어 λ 오름차순 표 + static 대비 판정 출력.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# M0 기준선 (results/m0/M0_FINDINGS.md, n=96)
STATIC = {"dino": 0.12304, "video": 0.09113, "action": 1.24017, "total": 0.56032}
GT = {"dino": 0.0, "video": 0.0, "action": 1.22278, "total": 0.48911}

TAG2LAM = {"10": 1.0, "15": 1.5, "20": 2.0, "25": 2.5, "30": 3.0, "40": 4.0}


def load_root(root: Path):
    rows = []
    for d in sorted(root.glob("cfg*")):
        rep = d / "m0_report.json"
        if not rep.exists():
            continue
        data = json.loads(rep.read_text(encoding="utf-8"))
        res = data.get("results", {})
        # 예측기 이름은 <prefix>_cfg<tag>; 첫(유일) 항목 사용
        for name, r in res.items():
            m = r["mean"]
            tag = d.name.replace("cfg", "")
            rows.append({
                "name": name,
                "tag": tag,
                "lam": TAG2LAM.get(tag, float("nan")),
                "dino": m["dino_frame_avg"],
                "video": m["video"],
                "action": m["action"],
                "total": m["total_frame_avg"],
                "perdim": r.get("action_perdim_mean"),
            })
    rows.sort(key=lambda x: (x["lam"] if x["lam"] == x["lam"] else 1e9))
    return rows


def fmt_delta(v, base):
    d = v - base
    sign = "+" if d >= 0 else ""
    return f"{v:.5f}({sign}{d:.3f})"


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path("results/m3/partA")]
    for root in roots:
        rows = load_root(root)
        print("\n" + "=" * 92)
        print(f"[{root}]  λ-CFG 반응 곡선  (0에 가까울수록 좋음; 괄호=static 대비 Δ)")
        print("=" * 92)
        print(f"{'λ':>5} {'name':<18}{'DINO(favg)':>18}{'Video':>18}{'Action':>10}{'TOTAL':>18}")
        print("-" * 92)
        print(f"{'--':>5} {'static(기준)':<18}{STATIC['dino']:>18.5f}{STATIC['video']:>18.5f}"
              f"{STATIC['action']:>10.5f}{STATIC['total']:>18.5f}")
        print(f"{'--':>5} {'gt(상한)':<18}{GT['dino']:>18.5f}{GT['video']:>18.5f}"
              f"{GT['action']:>10.5f}{GT['total']:>18.5f}")
        print("-" * 92)
        best = None
        for r in rows:
            print(f"{r['lam']:>5.1f} {r['name']:<18}"
                  f"{fmt_delta(r['dino'], STATIC['dino']):>18}"
                  f"{fmt_delta(r['video'], STATIC['video']):>18}"
                  f"{r['action']:>10.5f}"
                  f"{fmt_delta(r['total'], STATIC['total']):>18}")
            if best is None or r["total"] < best["total"]:
                best = r
        print("-" * 92)
        if best:
            beats = best["total"] < STATIC["total"]
            beats_dino = best["dino"] < STATIC["dino"]
            beats_video = best["video"] < STATIC["video"]
            print(f"[최선] λ={best['lam']} → TOTAL {best['total']:.5f} "
                  f"(static {STATIC['total']:.5f} 대비 {best['total']-STATIC['total']:+.5f})")
            print(f"       static를 넘는가?  TOTAL:{'예' if beats else '아니오'}  "
                  f"DINO:{'예' if beats_dino else '아니오'}  Video:{'예' if beats_video else '아니오'}")


if __name__ == "__main__":
    main()
