#!/usr/bin/env python
"""
두 팔(g1r · dir)의 history.json 을 읽어 등록된 판정선으로 표를 만든다.

⚠ 이 스크립트는 **판정선을 만들지 않는다.** 문턱은 전부 gates.py 에서 읽는다
  (010 §7: "판정선은 새로 만들지 마라. 이미 등록돼 있다").
  하는 일은 셋뿐이다 — 궤적을 나란히 놓고, 등록된 시점의 값을 뽑고, 오염을 라벨한다.

⚠ 이 표로 외삽하지 마라 (009 §11). 판정은 gates.py 가 정한 시점에만 한다.

사용:  .venv/bin/python scripts/branchC/compare_arms.py
       .venv/bin/python scripts/branchC/compare_arms.py --dirs <경로> <경로> ...
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
import gates as G                                                    # noqa: E402

G2_STEP = 6000          # 참 wake 2,000 + 4,000. 010 §7 에 등록된 발동 시점
# 008 §8-③ M4 는 "씨앗간 sd 의 2배"를 문턱으로 등록했는데 그 sd 를 재는 M2 를
# **아무도 돌리지 않았다.** 유일하게 있는 추정치는 ③ 탐침(160×256, hid_S=16)의
# S팔 3seed 산포 sd≈0.023 이고, gates.py 의 DCOS_MIN 0.046 이 바로 그 2배다.
# ⚠ 그 값을 본 모델(320×512, hid_S=64)로 옮기는 것은 gates.py 감시(e)가 명시적으로
#   경고한 규율 6 위반이다. 그래서 아래는 **참고선으로만 찍고 판정하지 않는다.**
M4_REFERENCE = G.DCOS_MIN
COLS = [("resid_ratio", "잔차비", 3), ("resid_cos", "코사인", 3),
        ("resid_cos_shuf", "뒤섞기", 3), ("dcos", "Δcos", 3),
        ("profile_slope", "기울기", 3), ("film_temporal_cos", "FiLMcos", 3),
        ("loss", "loss", 4)]


def load(d: Path) -> dict | None:
    p = d / "history.json"
    if not p.exists():
        return None
    h = json.load(open(p))
    # 커밋 888fcf0 이 FiLM 시간 코사인을 DC 제거 후 재도록 정의를 바꿨다(009 §12).
    # 그 커밋은 원본 런(20260808_1809_g1)이 **시작된 뒤** 들어갔으므로 그 런의 값은 구 정의다.
    # 같은 체크포인트에서 구 0.990 vs 신 0.688 — 직접 비교하면 안 된다 (010 §1 ※).
    return {"dir": d, "name": d.name, "hist": h["history"],
            "wake": h.get("wake_step"), "lam_c": h.get("lam_c"),
            "film_old_def": d.name == "train_20260808_1809_g1"}


def table(arm: dict) -> None:
    print(f"\n── {arm['name']}"
          + (f"   λ_c={arm['lam_c']:.4g}" if arm["lam_c"] else "   (방향항 없음)")
          + f"   wake={arm['wake']}"
          + ("   ⚠ 원본 런 — 짝지은 기준선이 아니다(g1r 이 그것이다)"
             if arm["film_old_def"] else ""))
    if arm["film_old_def"]:
        print("   ⚠ 이 런의 FiLMcos 는 폐기된 구 정의(DC 미제거)다. 신 정의와 비교 금지 (010 §1)")
    head = "  step  " + "  ".join(f"{lab:>8s}" for _, lab, _ in COLS)
    print(head)
    for m in arm["hist"]:
        row = f"  {m['step']:5d}  "
        for k, _, p in COLS:
            v = m.get(k)
            row += f"{v:>+8.{p}f}  " if isinstance(v, (int, float)) else f"{'—':>8s}  "
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="비우면 artifacts/branchC/train_*_{g1r,dir} 를 자동으로 찾는다")
    args = ap.parse_args()

    root = REPO / "artifacts" / "branchC"
    if args.dirs:
        cand = [Path(d) for d in args.dirs]
    else:
        cand = sorted(p for p in root.glob("train_*")
                      if p.name.endswith(("_g1r", "_dir", "_g1")))
    arms = [a for a in (load(d) for d in cand) if a and a["hist"]]
    if not arms:
        print("history.json 이 있는 런이 없다."); return

    print("=" * 78)
    print("branch C 두 팔 비교 — 문턱은 전부 gates.py 에서 읽는다")
    print(f"  등록값: Δcos ≥ {G.DCOS_MIN} · wake {G.WAKE_RATIO} · "
          f"FiLMcos < {G.FILM_TEMPORAL_COS_MAX} · 기울기 > {G.PROFILE_SLOPE_MIN}")
    print("=" * 78)
    for a in arms:
        table(a)

    # ── G2: 등록된 시점의 행만 본다 ──────────────────────────────
    print(f"\n{'=' * 78}\nG2 판정 — step {G2_STEP} (참 wake 2,000 + 4,000)\n{'=' * 78}")
    at6000 = {}
    for a in arms:
        m = next((x for x in a["hist"] if x["step"] == G2_STEP), None)
        if m is None:
            print(f"  {a['name']:34s} step {G2_STEP} 미도달 — 판정 불가")
            continue
        v = G.check_g2(m.get("rho_median", 0.0), m["dcos"])
        at6000[a["name"]] = m
        print(f"  {a['name']:34s} Δcos {m['dcos']:+.4f}  →  **{v}**   "
              f"{G.GATES['G2'][v if v in ('pass', 'fail') else 'tie']}")

    # ── M4: 방향항 A/B. 판정하지 않고 수치만 놓는다 ───────────────
    if len(at6000) >= 2:
        names = list(at6000)
        d = [n for n in names if n.endswith("_dir")]
        b = [n for n in names if n.endswith(("_g1r", "_g1"))]
        if d and b:
            cd, cb = at6000[d[0]]["resid_cos"], at6000[b[0]]["resid_cos"]
            dd, db = at6000[d[0]]["dcos"], at6000[b[0]]["dcos"]
            print(f"\n{'=' * 78}\nM4 재료 — step {G2_STEP} (008 §8-③)\n{'=' * 78}")
            print(f"  코사인   dir {cd:+.4f}  −  기준선 {cb:+.4f}  =  {cd - cb:+.4f}")
            print(f"  Δcos     dir {dd:+.4f}  −  기준선 {db:+.4f}  =  {dd - db:+.4f}")
            print(f"\n  참고선 {M4_REFERENCE} (= ③ 탐침 씨앗 sd 0.023 의 2배)")
            print("  ⚠ 이것으로 판정하지 마라. 그 sd 를 재는 M2 를 아무도 안 돌렸고,")
            print("    탐침(160×256·hid_S=16) 수치를 본 모델(320×512·hid_S=64)로 옮기는 것은")
            print("    gates.py 감시(e)가 명시적으로 금지한 규율 6 위반이다.")
            print("  ⚠ 그리고 dir 팔은 코사인을 **직접 최적화한다.** 코사인 차이는 오염돼 있다.")
            print("    덜 오염된 것은 Δcos 다 — 방향 손실은 행동 뒤섞기와 무관하기 때문이다.")

    print(f"\n{'=' * 78}")
    print("⚠ 외삽 금지 (009 §11). 진짜 판정은 리더보드 λ 스윕이다 (008 §9-b).")
    print("=" * 78)


if __name__ == "__main__":
    main()
