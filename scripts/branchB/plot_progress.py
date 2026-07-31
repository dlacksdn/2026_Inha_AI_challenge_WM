"""학습 진행 그래프 — 가로축 학습 스텝, 세로축 loss 와 DINO.

왜 위아래 두 칸으로 나누는가
--------------------------
loss 와 DINO 는 **재는 대상이 다르다**. loss 는 "노이즈를 얼마나 잘 지웠나"이고
DINO 는 "만든 영상이 정답과 의미적으로 얼마나 가까운가"다. 크기를 나란히 두면
"loss 가 DINO 보다 작으니 더 좋다" 같은 잘못된 읽기가 생긴다.
그래서 한 그림 안에서 **가로축(학습 스텝)만 공유하고 세로축은 따로** 둔다.
(두 개의 세로축을 한 칸에 겹쳐 그리는 방식은 눈금을 어떻게 잡느냐에 따라
 아무 결론이나 만들어낼 수 있어서 쓰지 않는다.)

가로축의 의미 — 두 번의 학습을 이어 붙인다
----------------------------------------
· 1차 학습: 0 → 1500 스텝까지 갔지만, 우리가 이어받은 것은 **1000 스텝 지점**이다.
  1000~1500 구간은 갈라져 나간 가지이므로 흐린 점선으로 구분해 둔다.
· 2차 학습(오늘): 1000 지점에서 출발한다. 재개 방식이 "가중치만 이어받기"라
  프로그램 내부 카운터는 0부터 다시 세므로, 그림에서는 **1000 을 더해** 누적 스텝으로 옮긴다.

사용:
  python3 scripts/branchB/plot_progress.py            # 시스템 python3 (matplotlib 3.10)
  python3 scripts/branchB/plot_progress.py --out 다른경로.png
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

# 검증기(dataviz validate_palette) 통과 색.
#   시리즈 2색 #3E6FA8 / #C1701E : 모든 검사 PASS (색각이상 분리 ΔE 20.8)
#   기준선 #12805C              : 채도 PASS, 분리 ΔE 7.4(WARN) — 파선 + 직접 라벨이라는
#                                 보조 표시가 함께 있으므로 규칙상 허용된다.
C_LOSS = "#3E6FA8"
C_DINO = "#C1701E"
C_GOAL = "#12805C"
C_INK = "#1A1D22"
C_MUTED = "#6E7079"
C_FAINT = "#9A9CA3"
C_GRID = "#E6E2DC"
C_BG = "#FBFAF8"

RESUME_AT = 1000          # 2차 학습이 갈라져 나온 지점(누적 스텝)
STATIC_DINO = 0.12308     # 넘어야 할 선: 첫 프레임 16복사의 DINO
VAE_FLOOR = 0.0452        # 오늘 실측한 하한: 정답 영상조차 VAE 를 왕복하면 이 값이 나온다


def setup_korean_font() -> str:
    # Regular 를 먼저 찾는다. 와일드카드로 훑으면 Thin 이 먼저 걸려 글자가 너무 가늘어진다.
    for pat in ("/usr/share/fonts/**/NotoSansCJK-Regular.ttc", "/usr/share/fonts/**/NotoSansKR-Regular.otf",
                "/usr/share/fonts/**/NanumGothic.ttf",
                "/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/NotoSansKR*.otf",
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
    return "(한글 폰트 못 찾음)"


def load_loss(csv_path: Path, offset: int = 0, key: str = "train/loss_step"):
    """metrics.csv → [(누적스텝, loss)] 와 이동평균."""
    rows = []
    if not csv_path.exists():
        return [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v, s = r.get(key, ""), r.get("step", "")
            if v and s:
                try:
                    rows.append((int(float(s)) + offset, float(v)))
                except ValueError:
                    pass
    rows.sort()
    ma = []
    for i, (s, _) in enumerate(rows):
        w = [v for _, v in rows[max(0, i - 7): i + 8]]
        ma.append((s, sum(w) / len(w)))
    return rows, ma


def load_dino() -> list[tuple[int, float]]:
    """m0_step*_emafix/m0_report.json 에서 (스텝, DINO) 를 모은다."""
    pts = []
    for d in sorted(glob.glob(str(REPO / "artifacts/branchB/m0_step*_emafix"))):
        rep = Path(d) / "m0_report.json"
        if not rep.exists():
            continue
        digits = "".join(ch for ch in Path(d).name.split("_")[1] if ch.isdigit())
        if not digits:
            continue
        data = json.loads(rep.read_text(encoding="utf-8"))
        for n, r in data["results"].items():
            if n.startswith("b1p1b"):
                pts.append((int(digits), float(r["mean"]["dino_frame_avg"])))
    pts.sort()
    return pts


def style(ax):
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", default=str(REPO / "artifacts/branchB/train_out/logs/inha_action_diffusion_1p1b/version_2/metrics.csv"))
    ap.add_argument("--run2", default=str(REPO / "artifacts/branchB/train_out/logs/inha_action_diffusion_1p1b_r2/version_0/metrics.csv"))
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/progress_curve.png"))
    args = ap.parse_args()

    print("[font]", setup_korean_font())
    raw1, ma1 = load_loss(Path(args.run1), 0)
    raw2, ma2 = load_loss(Path(args.run2), RESUME_AT)
    dino = load_dino()
    xmax = max([s for s, _ in raw1 + raw2] + [RESUME_AT]) + 60
    print(f"[data] 1차 loss {len(raw1)}점 / 2차 loss {len(raw2)}점 "
          f"(누적 {raw2[0][0] if raw2 else '-'}~{raw2[-1][0] if raw2 else '-'}) / DINO {len(dino)}점 {dino}")

    fig, (axd, axl) = plt.subplots(
        2, 1, figsize=(11.6, 8.4), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.17})
    fig.patch.set_facecolor(C_BG)

    # ── 위 칸: DINO (성능. 낮을수록 좋다) ────────────────────────────────────
    style(axd)
    axd.axhline(STATIC_DINO, color=C_GOAL, lw=2.0, ls=(0, (6, 4)), zorder=2)
    axd.text(xmax * 0.995, STATIC_DINO + 0.006, "넘어야 할 선 — 정지영상 0.123", ha="right",
             fontsize=10, fontweight="bold", color=C_GOAL)
    axd.axhline(VAE_FLOOR, color=C_FAINT, lw=1.6, ls=(0, (3, 3)), zorder=2)
    axd.text(xmax * 0.995, VAE_FLOOR + 0.006, "VAE 하한 0.045 — 학습으로 못 넘는 바닥", ha="right",
             fontsize=9.5, color=C_MUTED)

    if dino:
        xs = [s for s, _ in dino]
        ys = [v for _, v in dino]
        axd.plot(xs, ys, color=C_DINO, lw=2.4, zorder=4)
        axd.scatter(xs, ys, s=120, color=C_DINO, edgecolor=C_BG, linewidth=2.4, zorder=5,
                    label="DINO — 홀드아웃 96 표본 채점")
        for s, v in dino:
            axd.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 16),
                         ha="center", fontsize=11.5, fontweight="bold", color=C_DINO)
        # 마지막 점 이후는 아직 안 잰 구간이라는 표시
        axd.annotate("", xy=(xmax * 0.93, ys[-1]), xytext=(xs[-1], ys[-1]),
                     arrowprops=dict(arrowstyle="->", color=C_DINO, lw=1.4, ls=(0, (4, 3)),
                                     alpha=0.55, shrinkA=6, shrinkB=0))
        axd.text((xs[-1] + xmax * 0.93) / 2, ys[-1] + 0.013,
                 "이 구간은 아직 안 쟀다\n(평가하려면 GPU 가 필요한데 지금은 학습이 쓰는 중)",
                 ha="center", va="bottom", fontsize=9, color=C_MUTED)

    axd.set_ylim(0, 0.285)
    axd.set_ylabel("DINO 거리 (낮을수록 정답과 가깝다)", fontsize=11, color=C_MUTED, labelpad=9)
    axd.set_title("1.1B 학습 진행 — DINO 는 내려가는 중, loss 는 원래 평탄하다",
                  fontsize=15, fontweight="bold", color=C_INK, pad=14, loc="left")
    leg = axd.legend(loc="lower left", frameon=True, fontsize=9.5, framealpha=0.95,
                     edgecolor=C_GRID, facecolor="#FFFFFF")
    for t in leg.get_texts():
        t.set_color(C_INK)

    # ── 아래 칸: 학습 loss ───────────────────────────────────────────────────
    style(axl)
    r1a = [(s, v) for s, v in raw1 if s <= RESUME_AT]
    r1b = [(s, v) for s, v in raw1 if s >= RESUME_AT]
    m1a = [(s, v) for s, v in ma1 if s <= RESUME_AT]
    m1b = [(s, v) for s, v in ma1 if s >= RESUME_AT]

    axl.plot([s for s, _ in r1a], [v for _, v in r1a], color=C_LOSS, lw=0.9, alpha=0.20, zorder=2)
    axl.plot([s for s, _ in m1a], [v for _, v in m1a], color=C_LOSS, lw=2.4, zorder=4,
             label="1차 학습 loss (이동평균 15)")
    axl.plot([s for s, _ in r1b], [v for _, v in r1b], color=C_FAINT, lw=0.9, alpha=0.25, zorder=2)
    axl.plot([s for s, _ in m1b], [v for _, v in m1b], color=C_FAINT, lw=1.8, ls=(0, (5, 3)), zorder=3,
             label="1차 학습의 갈라진 가지 (이어받지 않음)")
    axl.plot([s for s, _ in raw2], [v for _, v in raw2], color=C_LOSS, lw=0.9, alpha=0.20, zorder=2)
    axl.plot([s for s, _ in ma2], [v for _, v in ma2], color=C_LOSS, lw=2.4, zorder=4,
             label="2차 학습 loss — 오늘 진행 중 (이동평균 15)")

    axl.set_ylim(0, 0.40)
    axl.set_ylabel("학습 loss", fontsize=11, color=C_MUTED, labelpad=9)
    axl.set_xlabel("누적 학습 스텝 (optimizer step)", fontsize=11, color=C_MUTED, labelpad=9)
    leg2 = axl.legend(loc="upper right", frameon=True, fontsize=9.5, framealpha=0.95,
                      edgecolor=C_GRID, facecolor="#FFFFFF")
    for t in leg2.get_texts():
        t.set_color(C_INK)
    axl.text(40, 0.30,
             "옅은 선은 loss 원본이다. 확산모델은 스텝마다 노이즈 세기를 무작위로 뽑아\n"
             "재기 때문에 30배씩 튄다 — 그래서 진한 이동평균만 본다.",
             fontsize=9, color=C_MUTED, ha="left", va="top")

    # ── 두 칸 공통: 재개 지점과 현재 위치 ────────────────────────────────────
    for ax in (axd, axl):
        ax.axvline(RESUME_AT, color=C_MUTED, lw=1.3, ls=(0, (2, 3)), alpha=0.8, zorder=1)
        if raw2:
            ax.axvline(raw2[-1][0], color=C_MUTED, lw=1.0, alpha=0.35, zorder=1)
    axd.text(RESUME_AT - 25, 0.272, "여기서 이어받아\n2차 학습 시작", ha="right", va="top",
             fontsize=9.5, color=C_MUTED)
    if raw2:
        axl.text(raw2[-1][0] - 25, 0.016, f"현재 {raw2[-1][0]} 스텝", ha="right", va="bottom",
                 fontsize=9.5, color=C_MUTED)
    axd.set_xlim(0, xmax)

    fig.text(0.012, 0.012,
             "loss 와 DINO 는 서로 다른 것을 재는 값이라 크기를 직접 비교하면 안 된다. "
             "각자의 '방향'만 본다. 성능 판정은 DINO 로만 한다.",
             fontsize=8.8, color=C_MUTED)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.925, bottom=0.105)
    fig.savefig(args.out, facecolor=C_BG)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
