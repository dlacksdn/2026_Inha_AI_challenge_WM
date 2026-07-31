"""학습 loss 와 홀드아웃 DINO 를 학습 스텝 축으로 그려 PNG 로 저장한다.

두 값 모두 무차원 [0, ~0.6] 이라 한 축에 그린다(이중 y축은 쓰지 않는다).
  - loss  : 낮을수록 노이즈 예측이 정확. 랜덤 타임스텝 때문에 개별 값이 크게 튀므로 이동평균을 함께 그린다.
  - DINO  : 낮을수록 정답 영상과 의미적으로 가깝다. 홀드아웃 96 채점값.

사용:
  PYTHONPATH=<matplotlib 격리경로> python scripts/branchB/plot_loss_dino.py
  (--out 으로 저장 경로 변경, --csv 로 학습 metrics.csv 지정)
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")

# dataviz 검증 통과 팔레트(light, surface #FBFAF8): CVD ΔE 20.8 / 대비 3:1 이상
C_LOSS = "#3E6FA8"
C_DINO = "#C1701E"
C_OK = "#2E6B4F"
C_INK = "#1A1D22"
C_MUTED = "#6E7079"
C_GRID = "#E6E2DC"
C_BG = "#FBFAF8"

REFS = [(0.12308, "static 기준선 0.123 — 넘어야 할 선", C_OK, 1.8, True),
        (0.481, "11M 재학습 0.481", C_MUTED, 1.1, False),
        (0.550, "주최 baseline 11M 0.550", C_MUTED, 1.1, False)]


def setup_korean_font() -> str:
    for pat in ("/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/NotoSansKR*.otf",
                "/usr/share/fonts/**/NanumGothic*.ttf"):
        for p in glob.glob(pat, recursive=True):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return f"{name} ({os.path.basename(p)})"
            except Exception:
                continue
    return "(한글 폰트 못 찾음 — 기본 폰트 사용)"


def load_loss(csv_path: str, key: str = "train/loss_step"):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v, s = r.get(key, ""), r.get("step", "")
            if v and s:
                try:
                    rows.append((int(float(s)), float(v)))
                except ValueError:
                    pass
    rows.sort()
    ma = []
    for i, (s, _) in enumerate(rows):
        w = [v for _, v in rows[max(0, i - 7): i + 8]]
        ma.append((s, sum(w) / len(w)))
    return rows, ma


def load_dino() -> list[tuple[int, float]]:
    """artifacts/branchB/m0_step*_emafix/m0_report.json 에서 (step, DINO) 를 모은다."""
    pts = []
    for d in sorted(glob.glob(str(REPO / "artifacts/branchB/m0_step*_emafix"))):
        rep = Path(d) / "m0_report.json"
        if not rep.exists():
            continue
        name = Path(d).name                       # m0_step1000_emafix
        digits = "".join(ch for ch in name.split("_")[1] if ch.isdigit())
        if not digits:
            continue
        data = json.loads(rep.read_text(encoding="utf-8"))
        for n, r in data["results"].items():
            if n.startswith("b1p1b"):
                pts.append((int(digits), float(r["mean"]["dino_frame_avg"])))
    pts.sort()
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "results/branchB/pilot_metrics_step1500.csv"))
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/loss_dino_curve.png"))
    args = ap.parse_args()

    print("[font]", setup_korean_font())
    raw, ma = load_loss(args.csv)
    dino = load_dino()
    print(f"[data] loss {len(raw)}점 (step {raw[0][0]}~{raw[-1][0]}) / DINO {len(dino)}점 {dino}")

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # 기준선(주석: 카테고리 색이 아닌 회색/녹색 파선)
    for val, label, color, lw, strong in REFS:
        ax.axhline(val, color=color, lw=lw, ls=(0, (6, 4)), alpha=1.0 if strong else 0.55, zorder=1)
        ax.text(1512, val, label, va="center", ha="left", fontsize=9.5,
                color=color, fontweight="bold" if strong else "normal")

    ax.plot([s for s, _ in raw], [v for _, v in raw], color=C_LOSS, lw=1.0, alpha=0.25,
            zorder=2, label="loss 원본 (랜덤 타임스텝 때문에 튐)")
    ax.plot([s for s, _ in ma], [v for _, v in ma], color=C_LOSS, lw=2.2, zorder=3,
            label="학습 loss (이동평균 15)")

    if dino:
        xs = [s for s, _ in dino]
        ys = [v for _, v in dino]
        if len(dino) > 1:
            ax.plot(xs, ys, color=C_DINO, lw=2.2, zorder=4)
        ax.scatter(xs, ys, s=110, color=C_DINO, edgecolor=C_BG, linewidth=2.2, zorder=5,
                   label="DINO (홀드아웃 96 채점)")
        for s, v in dino:
            ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 15),
                        ha="center", fontsize=11, fontweight="bold", color=C_DINO)

    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 0.60)
    ax.set_xlabel("학습 스텝 (optimizer step)", fontsize=11, color=C_MUTED, labelpad=9)
    ax.set_ylabel("값 (무차원 · 낮을수록 좋음)", fontsize=11, color=C_MUTED, labelpad=9)
    ax.set_title("1.1B 학습 진단 — loss 는 평탄한데 DINO 는 11M 대비 2.2배 개선",
                 fontsize=14.5, fontweight="bold", color=C_INK, pad=16, loc="left")
    ax.grid(axis="y", color=C_GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=10)

    leg = ax.legend(loc="upper left", frameon=True, fontsize=9.5, framealpha=0.95,
                    edgecolor=C_GRID, facecolor="#FFFFFF")
    for t in leg.get_texts():
        t.set_color(C_INK)

    fig.text(0.012, 0.017,
             "loss 와 DINO 는 서로 다른 것을 재는 값이다(크기 직접 비교 금지). "
             "loss=노이즈 예측 정확도, DINO=정답 영상과의 의미적 거리. 각자의 '방향'만 볼 것.",
             fontsize=8.6, color=C_MUTED)
    fig.subplots_adjust(left=0.062, right=0.80, top=0.88, bottom=0.14)
    fig.savefig(args.out, facecolor=C_BG)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
