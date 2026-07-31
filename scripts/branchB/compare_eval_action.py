"""두 제출 CSV 의 Action 성분을 **eval 216개 표본별로 짝지어** 비교한다.

이게 왜 특별한가
----------------
채점식은 `0.3·DINO + 0.3·Video + 0.4·Action` 인데, 앞의 두 성분은 정답 영상과 비교하는
값이라 정답이 없는 eval 에서는 **서버만** 계산할 수 있다. 그런데 Action 은 다르다.
Action 은 "생성 영상에서 역추정한 행동"과 "우리가 입력으로 받은 행동"의 오차라서
정답 영상이 필요 없다. 그래서 공식 제출킷이 만든 CSV 안에 **표본별로 그대로 들어 있다.**

⇒ 배점 **40%** 짜리 축에 대해서는, 제출권을 쓰지 않고도 리더보드와 같은 값으로
   우리 모델과 대조군(static)의 승부를 확정할 수 있다.

왜 짝지은 비교인가
------------------
표본마다 난이도가 다르다. 평균만 비교하면 그 편차에 묻힌다. 두 예측기가 **같은 216개**를
풀었으므로 표본별로 빼면(A−B) 난이도가 상쇄되어 작은 차이도 보인다.

읽는 법
-------
· Action 은 오차(MAE)이므로 **낮을수록 좋다.** A−B 가 음수면 A 가 더 좋다.
· |t| ≥ 2 면 우연으로 보기 어렵다.
· 이 값에 배점 0.4 를 곱한 것이 최종 점수에 미치는 실제 기여다.

사용:
  python3 scripts/branchB/compare_eval_action.py <A.csv> <B.csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_action(path: str) -> dict[str, float]:
    """제출 CSV 에서 표본별 Action Component 를 읽는다."""
    out: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("feature_component") != "Action Component":
                continue
            v = json.loads(row["feature_json"])
            while isinstance(v, list):
                v = v[0]
            out[row["sample_id"]] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a", help="우리 모델의 제출 CSV")
    ap.add_argument("csv_b", help="대조군(static)의 제출 CSV")
    args = ap.parse_args()

    a = read_action(args.csv_a)
    b = read_action(args.csv_b)
    sids = sorted(set(a) & set(b))
    if not sids:
        raise SystemExit("공통 표본이 없다 — CSV 형식을 확인하라")

    d = [a[s] - b[s] for s in sids]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("nan")
    wins = sum(1 for x in d if x < 0)
    ma = sum(a[s] for s in sids) / n
    mb = sum(b[s] for s in sids) / n

    print("=" * 72)
    print("eval 216 — Action 성분(배점 40%) 짝지은 비교   ※ 낮을수록 좋다")
    print("=" * 72)
    print(f"A = {Path(args.csv_a).name}")
    print(f"B = {Path(args.csv_b).name}")
    print(f"공통 표본 {n}개\n")
    print(f"  A 평균            {ma:.5f}")
    print(f"  B 평균            {mb:.5f}")
    print(f"  차이 (A−B)        {mean:+.5f}  ± {se:.5f} (SE)")
    print(f"  95% 구간          [{mean - 1.97 * se:+.5f}, {mean + 1.97 * se:+.5f}]")
    print(f"  t 값              {t:+.2f}")
    print(f"  A 승률            {wins}/{n}  ({100 * wins / n:.1f}%)")
    print(f"  최종 점수 기여    0.4 × {mean:+.5f} = {0.4 * mean:+.5f}")
    print()
    if t <= -2:
        print("  판정: A 가 리더보드 40% 축에서 유의하게 좋다.")
    elif t >= 2:
        print("  판정: B 가 리더보드 40% 축에서 유의하게 좋다.")
    else:
        print("  판정: 구분 안 됨 — 이 축에서는 둘의 차이를 노이즈와 가를 수 없다.")
    print()
    print("  ※ 나머지 60%(DINO·Video)는 정답 영상이 필요해 서버만 안다. 그건 제출로 확인한다.")


if __name__ == "__main__":
    main()
