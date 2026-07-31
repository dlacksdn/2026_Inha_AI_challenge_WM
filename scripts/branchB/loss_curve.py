"""학습 loss 추세를 사람이 읽을 수 있게 요약한다.

확산모델 loss 는 스텝마다 랜덤 타임스텝 t 를 뽑아 계산하므로 원본 값이 크게 튄다
(같은 품질이어도 t 가 크면 loss 가 크다). 따라서 "줄고 있는가"는 개별 값이 아니라
**구간 평균**으로 봐야 한다. 이 스크립트는 구간 평균 + 선형추세 + 간단한 막대그래프를 낸다.

사용:
  python scripts/branchB/loss_curve.py                    # 최신 학습 로그 자동 선택
  python scripts/branchB/loss_curve.py --bucket 50        # 구간 크기 변경
  python scripts/branchB/loss_curve.py --csv <경로>
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
from pathlib import Path

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")


def latest_csv() -> str | None:
    c = glob.glob(str(REPO / "artifacts/branchB/train_out/logs/**/metrics.csv"), recursive=True)
    c += glob.glob(str(REPO / "artifacts/*/train_out/logs/**/metrics.csv"), recursive=True)
    return max(c, key=os.path.getmtime) if c else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--bucket", type=int, default=100, help="구간 크기(스텝)")
    ap.add_argument("--key", default="train/loss_step")
    args = ap.parse_args()

    path = args.csv or latest_csv()
    if not path:
        print("metrics.csv 를 찾지 못했습니다."); return
    print(f"[loss] {path}\n")

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v, s = r.get(args.key, ""), r.get("step", "")
            if v not in ("", None) and s not in ("", None):
                try:
                    rows.append((int(float(s)), float(v)))
                except ValueError:
                    pass
    if not rows:
        print(f"'{args.key}' 값이 없습니다."); return
    rows.sort()

    # 구간 평균
    buckets: dict[int, list[float]] = {}
    for s, v in rows:
        buckets.setdefault(s // args.bucket, []).append(v)
    items = sorted(buckets.items())
    means = [(b * args.bucket, sum(v) / len(v), len(v)) for b, v in items]

    lo = min(m for _, m, _ in means); hi = max(m for _, m, _ in means)
    span = (hi - lo) or 1.0
    print(f"기록 {len(rows)}개 / 스텝 {rows[0][0]}~{rows[-1][0]} / 구간 {args.bucket}스텝 평균")
    print(f"{'구간시작':>8} {'평균loss':>9} {'n':>4}  그래프(구간평균 상대크기)")
    print("-" * 62)
    for s, m, n in means:
        bar = "█" * max(1, int((m - lo) / span * 30))
        print(f"{s:>8} {m:>9.4f} {n:>4}  {bar}")
    print("-" * 62)

    # 전반부 vs 후반부 (추세 판정)
    half = len(means) // 2
    if half >= 1:
        a = sum(m for _, m, _ in means[:half]) / half
        b = sum(m for _, m, _ in means[half:]) / (len(means) - half)
        d = (b - a) / a * 100
        arrow = "감소(개선)" if d < -1 else ("증가(악화)" if d > 1 else "거의 평탄")
        print(f"전반부 평균 {a:.4f} → 후반부 평균 {b:.4f}  ({d:+.1f}%) → {arrow}")
    print("\n※ 확산모델 loss 는 랜덤 타임스텝 t 때문에 개별 값이 크게 튄다.")
    print("  '학습이 되는가'는 구간 평균의 방향으로 판단하고,")
    print("  최종 성능은 loss 가 아니라 홀드아웃 채점(DINO/Video)으로 판정한다.")


if __name__ == "__main__":
    main()
