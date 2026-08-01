"""잔차 보정의 **천장과 바닥**을 그린다 — 정지영상을 정답 쪽으로 밀면 이득인가?

읽는 법 (한 문장)
-----------------
**초록 파선(정지영상) 아래로 내려간 선만 이득이다.** 위로 올라가면 손해다.

네 개의 선이 뜻하는 것
----------------------
· **파랑 — 완벽한 잔차**: 정답 방향으로 정확히 민 것. 우리가 절대 얻을 수 없는 **천장**이다.
· **주황 — 뭉갠 잔차**: 방향은 완벽하되 세밀한 무늬를 지운 것.
  학습된 모델은 뭉갠 잔차를 만들기 때문에 이쪽이 **현실에 더 가깝다.** 4배·8배 두 가지로 잰다.
· **빨강 — 남의 잔차**: 다른 표본의 움직임을 빌려온 것. 그럴듯하지만 이 장면의 정답은 아니다.
  틀린 방향으로 밀면 어떻게 되는지 보여주는 **바닥**이다.

가로축 α 는 "얼마나 밀었나"다. α=0 이면 정지영상 그대로, α=1 이면 그 잔차를 통째로 더한 것이다.

왜 천장만으로는 안 되나
-----------------------
파랑은 **오라클**이다. 방향이 완벽하다는 건 정답을 이미 안다는 뜻이라 도달할 수 없다.
그래서 파랑이 낮으면 "확실히 안 된다"는 강한 결론이지만, 파랑이 낮게 내려간다고
"된다"는 뜻은 아니다. 주황과 빨강을 같이 봐야 실제로 어디쯤 떨어질지 가늠할 수 있다.

사용: python3 scripts/branchB/plot_headroom.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")

SERIES = [
    ("oracle",   "완벽한 잔차 (천장·도달 불가)", "#1F6FB2", 3.4, "o"),
    ("blur4",    "뭉갠 잔차 4배 (현실적)",       "#D98324", 2.4, "s"),
    ("blur8",    "뭉갠 잔차 8배 (현실적)",       "#A8621A", 2.4, "^"),
    ("neighbor", "남의 잔차 (바닥·틀린 방향)",   "#C0392B", 2.6, "D"),
]

METRICS = [
    ("dino",   "DINO",   "배점 30% · 장면이 정답과 같아 보이는가"),
    ("video",  "Video",  "배점 30% · 움직임의 결이 정답 같은가"),
    ("action", "Action", "배점 40% · 시킨 대로 움직였는가"),
    ("total",  "TOTAL",  "0.3·DINO + 0.3·Video + 0.4·Action"),
]

C_STATIC = "#12805C"
C_INK = "#1A1D22"
C_MUTED = "#6E7079"
C_GRID = "#E6E2DC"
C_BG = "#FBFAF8"
C_LOSS = "#F4E4E2"   # 손해 구간 (정지영상보다 위)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "results/branchB/residual_headroom.json"))
    # 파일 이름 맨 앞에 날짜_시간을 둔다(run_logs 와 같은 규칙) — ls 만으로 시간순이 되고
    # 어떤 시점의 그림인지 이름만 봐도 안다. 그림은 데이터가 바뀌면 내용이 달라지므로
    # 덮어쓰지 않고 새로 쌓는 편이 안전하다.
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M")
    out = args.out or str(REPO / f"artifacts/branchB/{stamp}_residual_headroom.png")

    print("[font]", setup_korean_font())
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    V = data["variants"]
    n = data["n_samples"]

    # α 목록을 데이터에서 뽑는다
    def alphas_of(kind: str) -> list[float]:
        return sorted(float(k.split(":")[1]) for k in V if k.split(":")[0] == kind)

    base = V["oracle:0"]                       # α=0 = 정지영상

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), dpi=150, sharex=True)
    fig.patch.set_facecolor(C_BG)

    for ax, (key, name, desc) in zip(axes.ravel(), METRICS):
        ax.set_facecolor(C_BG)
        ys_all = [V[k][key] for k in V]
        lo, hi = min(ys_all), max(ys_all)
        pad = max((hi - lo) * 0.16, 1e-4)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.04, 1.16)

        # 정지영상보다 나쁜 구간을 옅게 칠한다 — "이 위는 손해"
        ax.axhspan(base[key], hi + pad, color=C_LOSS, zorder=0)
        ax.axhline(base[key], color=C_STATIC, lw=2.2, ls=(0, (6, 4)), zorder=3)
        ax.text(-0.02, base[key] + pad * 0.10, f"정지영상 {base[key]:.4f}",
                ha="left", fontsize=9.5, fontweight="bold", color=C_STATIC)

        for kind, label, color, lw, marker in SERIES:
            xs = alphas_of(kind)
            if not xs:
                continue
            # 모든 계열은 α=0 에서 정지영상과 같다(아무것도 안 더한 것)
            xs_full = [0.0] + [x for x in xs if x > 0]
            ys = [base[key]] + [V[f"{kind}:{x:g}"][key] for x in xs if x > 0]
            ax.plot(xs_full, ys, color=color, lw=lw, zorder=5, label=label,
                    solid_capstyle="round")
            ax.scatter(xs_full[1:], ys[1:], s=58, color=color, marker=marker,
                       edgecolor=C_BG, linewidth=1.6, zorder=6)

        ax.set_title(name, fontsize=13.5, fontweight="bold", color=C_INK, loc="left", pad=20)
        ax.text(0.0, 1.015, desc, transform=ax.transAxes, fontsize=8.8, color=C_MUTED)
        ax.grid(axis="y", color=C_GRID, lw=1, zorder=1)
        ax.set_axisbelow(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9.5)

    for ax in axes[1]:
        ax.set_xlabel("α — 정답 방향으로 얼마나 밀었나 (0 = 정지영상 그대로, 1 = 잔차 전부)",
                      fontsize=10.5, color=C_MUTED, labelpad=8)

    h, l = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.996, 0.997), frameon=True,
                     fontsize=9.5, framealpha=0.96, edgecolor=C_GRID, facecolor="#FFFFFF", ncol=2)
    for t in leg.get_texts():
        t.set_color(C_INK)

    fig.suptitle("정지영상을 정답 쪽으로 밀면 이득인가 — 초록 파선 아래로 내려가야 이득이다",
                 fontsize=15.5, fontweight="bold", color=C_INK, x=0.008, ha="left", y=0.983)
    fig.text(0.008, 0.012,
             f"홀드아웃 {n}표본. 네 칸 모두 낮을수록 좋다. 분홍 구간은 정지영상보다 나쁜 영역이다. "
             "파랑은 정답을 안다고 가정한 오라클이라 도달할 수 없다 — 실제 모델은 주황(뭉갬)과 "
             "빨강(방향 틀림) 사이 어딘가에 떨어진다.",
             fontsize=8.8, color=C_MUTED)
    fig.subplots_adjust(left=0.062, right=0.988, top=0.878, bottom=0.088, hspace=0.28, wspace=0.16)
    fig.savefig(out, facecolor=C_BG)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
