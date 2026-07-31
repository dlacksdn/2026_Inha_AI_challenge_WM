"""DINO 곡선만 그린다 — 세로축을 데이터에 맞게 좁혀 오르내림이 보이게.

왜 loss 를 뺐나
--------------
loss 는 확산모델 특성상 평탄하다(스텝마다 노이즈 세기를 무작위로 뽑아 재기 때문).
학습이 터졌는지 확인하는 용도는 끝났으므로, 이제 성능 지표인 DINO 만 본다.
loss 를 빼면 세로축을 DINO 범위에 맞게 **좁힐 수 있어** 작은 변화도 눈에 보인다.

세로축 범위
----------
자동으로 (가장 낮은 값, 가장 높은 값) 에 여백을 붙여 잡는다. 넘어야 할 선(정지영상 0.123)은
항상 포함시킨다. 이 파이프라인의 바닥인 VAE 하한 0.045 는 보통 범위 밖이라 글로만 적는다.
※ 세로축이 0 에서 시작하지 않는다. 변화를 확대해 보려는 의도이며, 그래서 눈금 값을 크게 적어둔다.

점 추가하는 법
-------------
아래 POINTS 에 (누적스텝, m0 디렉터리 이름, 배치, 메모) 를 한 줄 넣으면 된다.
디렉터리가 아직 없으면 조용히 건너뛰므로, 평가가 끝나는 대로 다시 실행하면 점이 채워진다.

사용:
  python3 scripts/branchB/plot_dino.py
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

# (누적 optimizer step, m0 디렉터리 이름, 생성 배치, 메모)
#   누적 = 1차 학습 스텝 + 2차 학습 스텝. 2차는 내부 카운터가 0부터 세므로 1000 을 더한 값이다.
POINTS = [
    (500,  "m0_step500_emafix",  1, "1차 학습"),
    (1000, "m0_step1000_emafix", 1, "1차 학습"),
    (1000, "m0_step1000_b4",     4, "배치4 검증"),
    (2000, "m0_cum2000_b4",      4, "2차 학습"),
    (2500, "m0_cum2500_b4",      4, "2차 학습"),
    (3000, "m0_cum3000_b4",      4, "2차 학습"),
]

STATIC_DINO = 0.12308
VAE_FLOOR = 0.0452

C_B1 = "#3E6FA8"      # 배치 1 로 잰 점
C_B4 = "#C1701E"      # 배치 4 로 잰 점
C_GOAL = "#12805C"
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


def read_dino(dirname: str) -> float | None:
    rep = REPO / "artifacts/branchB" / dirname / "m0_report.json"
    if not rep.exists():
        return None
    data = json.loads(rep.read_text(encoding="utf-8"))
    for n, r in data["results"].items():
        if n.startswith("b1p1b"):
            return float(r["mean"]["dino_frame_avg"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/dino_curve.png"))
    args = ap.parse_args()

    print("[font]", setup_korean_font())
    pts = []
    for step, d, batch, note in POINTS:
        v = read_dino(d)
        if v is None:
            print(f"[skip] 아직 없음: {d}")
            continue
        pts.append({"step": step, "dino": v, "batch": batch, "note": note, "dir": d})
        print(f"[point] 누적 {step:>5}  DINO {v:.5f}  배치{batch}  ({note})")
    if not pts:
        raise SystemExit("그릴 점이 없다")

    ys = [p["dino"] for p in pts] + [STATIC_DINO]
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.14, 0.004)
    ylo, yhi = lo - pad, hi + pad
    xs_all = [p["step"] for p in pts]
    xlo, xhi = -80, max(xs_all) * 1.20 + 80

    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=150)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # 넘어야 할 선
    ax.axhline(STATIC_DINO, color=C_GOAL, lw=2.2, ls=(0, (6, 4)), zorder=2)
    ax.text(xhi * 0.995, STATIC_DINO + pad * 0.10, "넘어야 할 선 — 정지영상 0.123", ha="right",
            fontsize=11, fontweight="bold", color=C_GOAL)

    # 배치 1 점들을 실선으로 잇는다(같은 조건끼리만 이어야 비교가 성립한다)
    b1 = [p for p in pts if p["batch"] == 1]
    b4 = [p for p in pts if p["batch"] == 4]
    if len(b1) > 1:
        ax.plot([p["step"] for p in b1], [p["dino"] for p in b1], color=C_B1, lw=2.4, zorder=4)
    if len(b4) > 1:
        ax.plot([p["step"] for p in b4], [p["dino"] for p in b4], color=C_B4, lw=2.4, zorder=4)

    for grp, color, marker, label in ((b1, C_B1, "o", "배치 1 로 생성"), (b4, C_B4, "D", "배치 4 로 생성")):
        if not grp:
            continue
        ax.scatter([p["step"] for p in grp], [p["dino"] for p in grp], s=135, color=color,
                   marker=marker, edgecolor=C_BG, linewidth=2.2, zorder=5, label=label)

    # 값 라벨 — 같은 x 에 점이 겹치면 위아래로 흩는다
    seen: dict[int, int] = {}
    for p in sorted(pts, key=lambda q: (q["step"], q["dino"])):
        k = seen.get(p["step"], 0)
        seen[p["step"]] = k + 1
        dy = 17 if k == 0 else -25
        ax.annotate(f"{p['dino']:.4f}", (p["step"], p["dino"]), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=12, fontweight="bold",
                    color=C_B1 if p["batch"] == 1 else C_B4)

    # 구간별 변화량
    for grp, color in ((b1, C_B1), (b4, C_B4)):
        for a, b in zip(grp, grp[1:]):
            if b["step"] == a["step"]:
                continue
            d = b["dino"] - a["dino"]
            ax.annotate(f"{d:+.4f}", ((a["step"] + b["step"]) / 2, (a["dino"] + b["dino"]) / 2),
                        textcoords="offset points", xytext=(0, -30), ha="center",
                        fontsize=10, color=color, alpha=0.95)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("누적 학습 스텝 (optimizer step)", fontsize=11.5, color=C_MUTED, labelpad=10)
    ax.set_ylabel("DINO 거리 — 낮을수록 정답 영상과 가깝다", fontsize=11.5, color=C_MUTED, labelpad=10)
    ax.set_title("1.1B DINO 곡선 — 세로축을 좁혀 오르내림을 확대했다",
                 fontsize=15, fontweight="bold", color=C_INK, pad=15, loc="left")
    ax.grid(axis="y", color=C_GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=11)

    leg = ax.legend(loc="upper right", frameon=True, fontsize=10, framealpha=0.95,
                    edgecolor=C_GRID, facecolor="#FFFFFF")
    for t in leg.get_texts():
        t.set_color(C_INK)

    fig.text(0.012, 0.015,
             f"세로축이 0 에서 시작하지 않는다 — 변화를 확대해 보려는 의도다. "
             f"이 파이프라인의 바닥은 VAE 왕복 손실 {VAE_FLOOR:.3f} 이며 그 아래로는 학습으로도 못 내려간다.",
             fontsize=9, color=C_MUTED)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.905, bottom=0.125)
    fig.savefig(args.out, facecolor=C_BG)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
