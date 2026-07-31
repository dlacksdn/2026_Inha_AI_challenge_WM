"""두 채점 리포트를 **같은 표본끼리 짝지어** 비교한다 — 차이가 진짜인지 노이즈인지 가른다.

왜 짝지은 비교인가
------------------
표본 96개의 점수는 표본마다 편차가 크다(쉬운 장면·어려운 장면). 그래서 두 모델의 **평균만**
비교하면, 실제 차이가 표본 편차에 묻혀 "좋아졌다/나빠졌다"를 구분할 수 없다.

그런데 두 모델은 **똑같은 96개 표본**을 풀었다. 그러면 표본별로 점수를 빼서(A−B) 그 차이만
보면 된다. 장면의 난이도는 뺄셈에서 상쇄되므로 훨씬 작은 차이도 보인다. 이것이 짝지은 비교다.

읽는 법
-------
· 네 성분 모두 **낮을수록 좋다.** 그래서 차이(A−B)가 음수면 A 가 더 좋다.
· t 값은 "차이가 0일 뿐인데 우연히 이만큼 벌어질 수 있는가"를 재는 눈금이다.
  |t| ≥ 2 면 우연으로 보기 어렵다(표본 96개 기준 대략 5% 유의수준).
· 승률은 96개 중 A 가 이긴 표본 수. 50%에 가까우면 그냥 뒤섞인 것이다.

사용:
  python3 scripts/branchB/compare_reports.py <A리포트> <B리포트>
예:
  python3 scripts/branchB/compare_reports.py \
      artifacts/branchB/m0_cum3250_lr1e5/m0_report.json \
      artifacts/branchB/m0_step1000_b4/m0_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KEYS = [
    ("dino_frame_avg", "DINO", 0.3),
    ("video", "Video", 0.3),
    ("action", "Action", 0.4),
    ("total_frame_avg", "TOTAL", 1.0),
]


def load_rows(path: str, want: str | None = None) -> tuple[str, dict[str, dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data["results"]
    if want and want in results:
        name = want
    else:
        # 우리 모델(b1p1b...)을 우선 고르고, 없으면 첫 항목
        name = next((n for n in results if n.startswith("b1p1b")), next(iter(results)))
    rows = {r["sid"]: r for r in results[name]["rows"]}
    return name, rows


def paired(a: dict[str, dict], b: dict[str, dict], key: str) -> dict:
    sids = sorted(set(a) & set(b))
    d = [a[s][key] - b[s][key] for s in sids]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("nan")
    wins = sum(1 for x in d if x < 0)
    return {"n": n, "mean": mean, "se": se, "t": t, "wins": wins,
            "ci": (mean - 1.985 * se, mean + 1.985 * se)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_a")
    ap.add_argument("report_b")
    ap.add_argument("--name-a", default=None, help="A 리포트에서 고를 predictor 이름")
    ap.add_argument("--name-b", default=None, help="B 리포트에서 고를 predictor 이름")
    args = ap.parse_args()

    na, ra = load_rows(args.report_a, args.name_a)
    nb, rb = load_rows(args.report_b, args.name_b)

    print(f"A = {na}   ({args.report_a})")
    print(f"B = {nb}   ({args.report_b})")
    print(f"공통 표본 {len(set(ra) & set(rb))}개\n")
    print(f"{'성분':<7} {'A평균':>9} {'B평균':>9} {'A−B':>10} {'±SE':>9} {'t':>7} {'A승률':>8}  판정")
    print("-" * 78)

    for key, label, _w in KEYS:
        sids = sorted(set(ra) & set(rb))
        ma = sum(ra[s][key] for s in sids) / len(sids)
        mb = sum(rb[s][key] for s in sids) / len(sids)
        p = paired(ra, rb, key)
        if p["t"] <= -2:
            verdict = "A가 유의하게 좋다"
        elif p["t"] >= 2:
            verdict = "B가 유의하게 좋다"
        else:
            verdict = "구분 안 됨(노이즈)"
        print(f"{label:<7} {ma:>9.5f} {mb:>9.5f} {p['mean']:>+10.5f} {p['se']:>9.5f} "
              f"{p['t']:>+7.2f} {p['wins']:>4}/{p['n']:<3}  {verdict}")

    print("\n※ 전부 낮을수록 좋다. A−B 가 음수면 A 가 더 좋다는 뜻이다.")
    print("※ |t| ≥ 2 여야 '우연이 아니다'라고 말할 수 있다. 그 아래는 평균만 보고 결론 내면 안 된다.")


if __name__ == "__main__":
    main()
