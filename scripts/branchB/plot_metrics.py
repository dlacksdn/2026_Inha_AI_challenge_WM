"""네 성분(DINO·Video·Action·TOTAL)을 각자 스케일로 그린다 — 향상·정체·악화가 보이게.

왜 네 칸으로 나누는가
--------------------
네 값의 크기가 서로 다르다. Action 은 1.2 대인데 Video 는 0.09 대라, 한 축에 올리면
Action 이 그림을 다 차지하고 나머지는 납작한 직선으로 보인다. 그렇다고 세로축을 두 개
겹쳐 쓰면 눈금을 어떻게 잡느냐에 따라 아무 결론이나 만들어낼 수 있다.
그래서 **가로축(학습 스텝)만 공유하고 세로축은 칸마다 따로** 둔다. 각 칸의 세로축은
그 성분의 실제 값 범위에 맞춰 좁게 잡으므로 작은 오르내림도 눈에 보인다.

읽는 법
-------
· 네 성분 모두 **낮을수록 좋다.** 선이 내려가면 향상, 평평하면 정체, 올라가면 악화다.
· 초록 파선 = 정지영상(static). **넘어야 할 선**이다.
· 회색 파선 = 정답 영상 그대로 넣었을 때의 값. 물리적 상한이며 DINO·Video 는 0 이다.
· 배치 1 로 만든 점과 배치 4 로 만든 점은 색·모양이 다르다. 생성 조건이 다르므로
  한 선으로 잇지 않는다(짝지은 비교 결과 배치 효과는 t=-1.07 로 유의하지 않았지만,
  조건이 다른 점을 같은 선으로 잇는 건 그림이 거짓말을 하게 만든다).

사용: python3 scripts/branchB/plot_metrics.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")

# (누적 optimizer step, m0 디렉터리, 생성 배치, 메모)
POINTS = [
    (500,  "m0_step500_emafix",  1, "1차"),
    (1000, "m0_step1000_emafix", 1, "1차"),
    (1000, "m0_step1000_b4",     4, "재측정"),
    (2000, "m0_cum2000_b4",      4, "2차"),
    (2500, "m0_cum2500_b4",      4, "2차"),
    (3000, "m0_cum3000_b4",      4, "2차"),
]

# (리포트 키, 화면 이름, 배점 설명)
METRICS = [
    ("dino_frame_avg",   "DINO",  "배점 30% · 장면이 정답과 의미적으로 같은가"),
    ("video",            "Video", "배점 30% · 영상 전체의 움직임이 정답 같은가"),
    ("action",           "Action", "배점 40% · 영상에서 읽은 관절각이 정답과 맞는가"),
    ("total_frame_avg",  "TOTAL", "0.3·DINO + 0.3·Video + 0.4·Action"),
]

C_B1 = "#3E6FA8"
C_B4 = "#C1701E"
C_STATIC = "#12805C"
C_GT = "#9A9CA3"
C_INK = "#1A1D22"
C_MUTED = "#6E7079"
C_GRID = "#E6E2DC"
C_BG = "#FBFAF8"


def setup_korean_font() -> str:
    for pat in ("/usr/share/fonts/**/NotoSansCJK-Regular.ttc", "/usr/share/fonts/**/NotoSansKR-Regular.otf",
                "/usr/share/fonts/**/NanumGothic.ttf", "/usr/share/fonts/**/NotoSansCJK*.ttc"):
        for p in glob.glob(pat, recursive=True):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return f"{name} ({os.path.basename(p)})"
            except Exception:
                continue
    return "(한글 폰트 못 찾음)"


def read_report(dirname: str) -> dict | None:
    rep = REPO / "artifacts/branchB" / dirname / "m0_report.json"
    if not rep.exists():
        return None
    data = json.loads(rep.read_text(encoding="utf-8"))
    out = {"ours": None, "static": None, "gt": None}
    for n, r in data["results"].items():
        if n.startswith("b1p1b"):
            out["ours"] = r["mean"]
        elif n == "static":
            out["static"] = r["mean"]
        elif n == "gt_upper_bound":
            out["gt"] = r["mean"]
    return out if out["ours"] else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/metrics_curve.png"))
    args = ap.parse_args()

    print("[font]", setup_korean_font())
    pts, ref_static, ref_gt = [], None, None
    for step, d, batch, note in POINTS:
        rep = read_report(d)
        if rep is None:
            print(f"[skip] 아직 없음: {d}")
            continue
        pts.append({"step": step, "batch": batch, "note": note, "m": rep["ours"], "dir": d})
        ref_static = ref_static or rep["static"]
        ref_gt = ref_gt or rep["gt"]
        print(f"[point] 누적 {step:>5} 배치{batch}  "
              + "  ".join(f"{k}={rep['ours'][k]:.5f}" for k, _, _ in METRICS))
    if not pts:
        raise SystemExit("그릴 점이 없다")

    xs_all = [p["step"] for p in pts]
    xlo, xhi = -120, max(xs_all) * 1.13 + 120

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8), dpi=150, sharex=True)
    fig.patch.set_facecolor(C_BG)

    for ax, (key, name, desc) in zip(axes.ravel(), METRICS):
        ax.set_facecolor(C_BG)
        vals = [p["m"][key] for p in pts]
        refs = [r for r in (ref_static[key] if ref_static else None,
                            ref_gt[key] if ref_gt else None) if r is not None]
        lo, hi = min(vals + refs), max(vals + refs)
        pad = max((hi - lo) * 0.16, 1e-4)
        ax.set_ylim(lo - pad, hi + pad)

        # 기준선
        if ref_static:
            ax.axhline(ref_static[key], color=C_STATIC, lw=2.0, ls=(0, (6, 4)), zorder=2)
            ax.text(xhi * 0.99, ref_static[key] + pad * 0.13, f"정지영상 {ref_static[key]:.4f}",
                    ha="right", fontsize=9.5, fontweight="bold", color=C_STATIC)
        if ref_gt and abs(ref_gt[key] - (ref_static[key] if ref_static else 0)) > pad * 0.4:
            ax.axhline(ref_gt[key], color=C_GT, lw=1.5, ls=(0, (3, 3)), zorder=2)
            ax.text(xhi * 0.99, ref_gt[key] + pad * 0.13, f"정답영상 {0.0 if abs(ref_gt[key]) < 1e-9 else ref_gt[key]:.4f}",
                    ha="right", fontsize=9, color=C_MUTED)

        b1 = [p for p in pts if p["batch"] == 1]
        b4 = [p for p in pts if p["batch"] == 4]
        for grp, color, marker, lbl in ((b1, C_B1, "o", "배치 1 생성"), (b4, C_B4, "D", "배치 4 생성")):
            if not grp:
                continue
            if len(grp) > 1:
                ax.plot([p["step"] for p in grp], [p["m"][key] for p in grp],
                        color=color, lw=2.3, zorder=4)
            ax.scatter([p["step"] for p in grp], [p["m"][key] for p in grp], s=105, color=color,
                       marker=marker, edgecolor=C_BG, linewidth=2.0, zorder=5, label=lbl)

        # 값 라벨 (같은 x 에 겹치면 위아래로 흩는다)
        seen: dict[int, int] = {}
        for p in sorted(pts, key=lambda q: (q["step"], q["m"][key])):
            k = seen.get(p["step"], 0)
            seen[p["step"]] = k + 1
            ax.annotate(f"{p['m'][key]:.4f}", (p["step"], p["m"][key]), textcoords="offset points",
                        xytext=(0, 13 if k == 0 else -22), ha="center", fontsize=9.5,
                        fontweight="bold", color=C_B1 if p["batch"] == 1 else C_B4)

        # 구간 변화량 — 향상/악화를 부호로 바로 읽게
        for grp, color in ((b1, C_B1), (b4, C_B4)):
            for a, b in zip(grp, grp[1:]):
                if b["step"] == a["step"]:
                    continue
                d = b["m"][key] - a["m"][key]
                arrow = "▼ 향상" if d < 0 else "▲ 악화"
                ax.annotate(f"{arrow} {abs(d):.4f}",
                            ((a["step"] + b["step"]) / 2, (a["m"][key] + b["m"][key]) / 2),
                            textcoords="offset points", xytext=(0, 18), ha="center",
                            fontsize=8.8, color=color, alpha=0.95)

        ax.set_title(f"{name}", fontsize=13, fontweight="bold", color=C_INK, loc="left", pad=22)
        ax.text(0.0, 1.015, desc, transform=ax.transAxes, fontsize=8.8, color=C_MUTED)
        ax.set_xlim(xlo, xhi)
        ax.grid(axis="y", color=C_GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9.5)

    for ax in axes[1]:
        ax.set_xlabel("누적 학습 스텝 (optimizer step)", fontsize=10.5, color=C_MUTED, labelpad=8)

    h, l = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=True,
                     fontsize=9.5, framealpha=0.95, edgecolor=C_GRID, facecolor="#FFFFFF", ncol=2)
    for t in leg.get_texts():
        t.set_color(C_INK)

    fig.suptitle("1.1B 성능 네 성분 — 네 칸 모두 낮을수록 좋다 (세로축은 칸마다 다르다)",
                 fontsize=15.5, fontweight="bold", color=C_INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.012,
             "세로축이 0 에서 시작하지 않는다 — 작은 변화를 확대해 보려는 의도다. 칸끼리 값의 크기를 비교하지 말 것. "
             "성분마다 재는 대상이 다르다.",
             fontsize=8.8, color=C_MUTED)
    fig.subplots_adjust(left=0.062, right=0.988, top=0.885, bottom=0.085, hspace=0.30, wspace=0.16)
    fig.savefig(args.out, facecolor=C_BG)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
