"""학습이 **어디서 갈라졌는지**를 보여주는 그림 — 세 학습률 + seed 수정 갈래.

왜 새로 그리나
--------------
기존 plot_metrics.py 는 점들을 **한 줄로** 이었다. 그런데 실제 실험 구조는 한 줄이 아니다.
누적 1,000 스텝 지점에서 **네 갈래로 갈라져** 각각 따로 학습했다.

    누적 500 ─ 1,000 ┬─ lr 1e-4 (고정seed) ─→ 2,000 ─ 2,500 ─ 3,000    ← 크게 무너짐
                     ├─ lr 3e-5 (고정seed) ─────────────→ 3,250        ← 무너짐
                     ├─ lr 1e-5 (고정seed) ─────────────→ 3,250        ← 안 움직임
                     └─ lr 3e-5 (seed 777) ────────────→ 3,250        ← 무너짐이 사라짐

한 줄로 이으면 "학습할수록 나빠진다"는 **틀린 이야기**가 된다. 실제로는
"어떤 조건에서 나빠지고 어떤 조건에서 안 나빠지는가"가 이야기의 전부다.

읽는 법
-------
· 네 성분 모두 **낮을수록 좋다.** 선이 올라가면 나빠진 것이다.
· 초록 파선 = 정지영상(static). 넘어야 할 선이다.
· 굵은 파란 선(seed 777)만 평평하다 = **망가지지 않는다.** 나머지는 다 올라간다.
· 회색 점선 = 출발점(누적 1,000)의 값. 이 선 아래로 내려가야 '좋아졌다'이다.

사용: python3 scripts/branchB/plot_branches.py
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

# 공통 줄기 (배치 4 로 잰 것만 쓴다 — 생성 조건이 다르면 이으면 안 된다)
TRUNK = [(1000, "m0_step1000_b4")]

# 갈래: (표시이름, 색, 선굵기, [(누적스텝, m0디렉터리), ...])
BRANCHES = [
    ("lr 1e-4  (고정 seed)", "#C0392B", 2.2, [(1000, "m0_step1000_b4"), (2000, "m0_cum2000_b4"),
                                              (2500, "m0_cum2500_b4"), (3000, "m0_cum3000_b4")]),
    ("lr 3e-5  (고정 seed)", "#D98324", 2.2, [(1000, "m0_step1000_b4"), (3250, "m0_cum3250_lr3e5")]),
    ("lr 1e-5  (고정 seed)", "#7D8471", 2.2, [(1000, "m0_step1000_b4"), (3250, "m0_cum3250_lr1e5")]),
    ("lr 3e-5  (seed 777)",  "#1F6FB2", 3.4, [(1000, "m0_step1000_b4"), (3250, "m0_cum3250_seed777")]),
]

METRICS = [
    ("dino_frame_avg", "DINO", "배점 30% · 장면이 정답과 의미적으로 같은가"),
    ("video",          "Video", "배점 30% · 영상 전체의 움직임이 정답 같은가"),
    ("action",         "Action", "배점 40% · 영상에서 읽은 관절각이 정답과 맞는가"),
    ("total_frame_avg", "TOTAL", "0.3·DINO + 0.3·Video + 0.4·Action"),
]

C_STATIC = "#12805C"
C_START = "#9A9CA3"
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
    out = {"ours": None, "static": None}
    for n, r in data["results"].items():
        if n.startswith("b1p1b"):
            out["ours"] = r["mean"]
        elif n == "static":
            out["static"] = r["mean"]
    return out if out["ours"] else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/branches_curve.png"))
    args = ap.parse_args()

    print("[font]", setup_korean_font())
    cache: dict[str, dict] = {}
    ref_static = None
    for _lbl, _c, _lw, pts in BRANCHES:
        for _s, d in pts:
            if d in cache:
                continue
            rep = read_report(d)
            if rep is None:
                print(f"[skip] 아직 없음: {d}")
                continue
            cache[d] = rep["ours"]
            ref_static = ref_static or rep["static"]
    if not cache:
        raise SystemExit("그릴 점이 없다")
    start = cache[TRUNK[0][1]]
    for d, m in cache.items():
        print(f"[point] {d:24s} " + "  ".join(f"{k}={m[k]:.5f}" for k, _, _ in METRICS))

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), dpi=150, sharex=True)
    fig.patch.set_facecolor(C_BG)

    for ax, (key, name, desc) in zip(axes.ravel(), METRICS):
        ax.set_facecolor(C_BG)
        vals = [m[key] for m in cache.values()]
        refs = [ref_static[key]] if ref_static else []
        lo, hi = min(vals + refs), max(vals + refs)
        pad = max((hi - lo) * 0.18, 1e-4)
        ax.set_ylim(lo - pad, hi + pad)
        xhi = 3250 * 1.46

        # 기준선 라벨은 **왼쪽**에 둔다 — 오른쪽은 갈래 끝값 라벨이 쓰는 자리라 겹친다.
        if ref_static:
            ax.axhline(ref_static[key], color=C_STATIC, lw=2.0, ls=(0, (6, 4)), zorder=2)
            ax.text(730, ref_static[key] + pad * 0.10, f"정지영상 {ref_static[key]:.4f}",
                    ha="left", fontsize=9.5, fontweight="bold", color=C_STATIC)
        # 출발점 수평선 — 이 아래로 가야 '좋아졌다'
        # 출발점 라벨만 오른쪽 끝(선 아래)에 둔다 — 왼쪽은 출발점 표식과 겹친다.
        ax.axhline(start[key], color=C_START, lw=1.4, ls=(0, (2, 3)), zorder=2)
        ax.text(xhi * 0.997, start[key] - pad * 0.20, f"출발점 {start[key]:.4f}",
                ha="right", fontsize=8.8, color=C_MUTED)

        # 갈래를 그리고, 끝값 라벨은 서로 겹치지 않게 위아래로 벌린다.
        ends = []
        for lbl, color, lw, pts in BRANCHES:
            xs = [s for s, d in pts if d in cache]
            ys = [cache[d][key] for s, d in pts if d in cache]
            if len(xs) < 2:
                continue
            ax.plot(xs, ys, color=color, lw=lw, zorder=4, label=lbl, solid_capstyle="round")
            ax.scatter(xs[1:], ys[1:], s=70, color=color, edgecolor=C_BG, linewidth=1.8, zorder=5)
            ends.append({"x": xs[-1], "y": ys[-1], "color": color})

        # 라벨 y 를 아래에서 위로 훑으며, 최소 간격보다 붙어 있으면 밀어 올린다.
        gap = pad * 0.42                      # 글자 높이만큼의 최소 간격
        ends.sort(key=lambda e: e["y"])
        for i, e in enumerate(ends):
            e["ty"] = e["y"] if i == 0 else max(e["y"], ends[i - 1]["ty"] + gap)
        for e in ends:
            # 라벨이 점에서 떨어졌으면 가는 선으로 이어 어느 점의 값인지 분명히 한다.
            if abs(e["ty"] - e["y"]) > gap * 0.25:
                ax.plot([e["x"], e["x"] + 105], [e["y"], e["ty"]], color=e["color"],
                        lw=0.9, alpha=0.6, zorder=3)
            ax.annotate(f"{e['y']:.4f}", (e["x"] + 115, e["ty"]), textcoords="offset points",
                        xytext=(0, -4), ha="left", fontsize=9.5, fontweight="bold", color=e["color"])

        ax.scatter([1000], [start[key]], s=110, color=C_INK, marker="o",
                   edgecolor=C_BG, linewidth=2.0, zorder=6)

        ax.set_title(name, fontsize=13.5, fontweight="bold", color=C_INK, loc="left", pad=20)
        ax.text(0.0, 1.015, desc, transform=ax.transAxes, fontsize=8.8, color=C_MUTED)
        ax.set_xlim(700, xhi)
        ax.grid(axis="y", color=C_GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9.5)

    for ax in axes[1]:
        ax.set_xlabel("누적 학습 스텝 (optimizer step)", fontsize=10.5, color=C_MUTED, labelpad=8)

    h, l = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.996, 0.997), frameon=True,
                     fontsize=10, framealpha=0.96, edgecolor=C_GRID, facecolor="#FFFFFF", ncol=2)
    for t in leg.get_texts():
        t.set_color(C_INK)

    fig.suptitle("같은 지점에서 네 갈래로 갈라 학습했다 — 파란 선(seed 수정)만 망가지지 않는다",
                 fontsize=15.5, fontweight="bold", color=C_INK, x=0.008, ha="left", y=0.983)
    fig.text(0.008, 0.012,
             "네 칸 모두 낮을수록 좋다. 세로축은 칸마다 다르며 0 에서 시작하지 않는다(작은 변화를 확대해 보려는 의도). "
             "네 갈래는 모두 누적 1,000 스텝의 같은 체크포인트에서 출발했고, 바뀐 것은 학습률과 seed 뿐이다.",
             fontsize=8.8, color=C_MUTED)
    fig.subplots_adjust(left=0.058, right=0.988, top=0.878, bottom=0.082, hspace=0.28, wspace=0.15)
    fig.savefig(args.out, facecolor=C_BG)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
