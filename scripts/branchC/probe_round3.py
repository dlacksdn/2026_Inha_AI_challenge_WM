"""3라운드 — 2차 적대적 검수가 남긴 숙제. **판정선은 018 §9-c 에 측정 전 고정했다.**

2차 검수의 가장 아픈 지적은 이것이었다.

    "2라운드의 판정선은 어디 있나? **없다.**
     게이트·축·해석 규칙이 전부 1라운드 결과를 본 뒤에 설계됐다."

맞는 말이다. 그래서 3라운드는 **판정선을 문서(018 §9-c)에 먼저 적고** 여기 그대로 옮긴다.
결과를 본 뒤에 기준을 바꾸지 않는다. 판정선이 답을 못 주면 **"모른다"로 남긴다.**

---

## A. α × k 2차원 훑기 — 가장 중요하다

지금까지 잰 것을 격자에 놓으면 **십자가만 재고 가운데를 안 쟀다.**

```
              k=1     k=2     k=4     k=8
  α=1.00     −0.071  −0.038  −0.017  −0.004    ← 이 행만 쟀다 (방향이 완벽할 때)
  α=0.75       ?       ?       ?       ?
  α=0.50       ?       ?       ?       ?
  α=0.35       ?       ?       ?       ?       ← **우리 모델이 서 있는 행. 통째로 비어 있다**
  α=0.25       ?       ?       ?       ?
  α=0.00     +0.051    ?       ?       ?       ← 이 열만 쟀다 (방향이 틀렸을 때)
```

실제 모델은 **방향도 틀리고 해상도도 모자란다.** 둘 다 틀린 칸이 설계 결정을 좌우한다.

    예측 = static + blur_k( α·자기잔차 + (1−α)·남의잔차 )

    α=1 이면 방향이 완벽하고 α=0 이면 통째로 남의 것이다.
    016 §9.2 는 우리 모델을 "남의 잔차 α≈0.35 세기"로 못 박았다.
    그 좌표를 이 격자 위에서 다시 찾아(TOTAL 이 static 대비 +0.02073 인 칸) 표시한다.

**판정선 (018 §9-c 에서 고정)**

    α=0.35 행에서 k 를 4→2 로 낮출 때
        ΔDV ≤ −0.005 이고 t ≤ −2   →  H1(해상도 우선) **확정**
        개선이 0.005 미만            →  H1 **강등**. 해상도는 1순위가 아니다
    α=0.35 행이 **전 k 에서 static 보다 나쁘면**  →  잔차 노선 자체를 재검토

## B. 공정한 바닥 — 이번엔 **같은 배율**로

2라운드의 바닥은 두 팔의 뭉갬 배율이 달라 무효였다(018 §3.7② 철회).
`blur_residual(x, 1)` 이 항등이므로 `addnbrwarp1 ≡ warpnbr:1` 이었고,
따라서 잰 것은 "표현의 차이"가 아니라 "뭉갬의 차이"였다.

    warpnbr_c4   = warp(첫프레임, coarsen_flow(남의흐름, 4), 1)     ← 흐름을 4배 뭉갠다
    addnbrwarp4  = static + blur_4( warp(첫프레임, 남의흐름, 1) − static )

**판정선**: ΔDV ≤ −0.005 & t≤−2 → 워핑이 덜 다친다 / ≥ +0.005 & t≥+2 → 덧셈이 덜 다친다 /
그 외 → **무승부. 바닥으로는 아무것도 못 말한다.**

## D. 보간 흐림 분리

`grid_sample(mode="bilinear")` 은 변위가 정수가 아닌 모든 화소에 저역통과를 건다.
"워핑은 디테일이 안 뭉개진다"는 우리 구현에서 거짓이다.

**판정선**: nearest 의 DINO 가 bilinear 보다 0.005 이상 좋으면 → 손해의 상당분이 보간 흐림이다.

## E. k=2 작동점의 혼합

018 §3.8 의 혼합 기각은 k=4 에서만 측정됐는데 §3.7⑤ 가 작동점을 k=2 로 옮겼다.

**판정선**: ΔDV ≤ −0.005 & t≤−2 → 혼합 재검토 / 그 외 → **기각 유지.**

## F. blur_residual 독립 대조

018 §3.1 이 "독립 재현 검증"이라 불렀던 것은 016 의 함수를 그대로 import 한 것이라
검증이 아니었다. 그런데 해상도·흐림 결정이 전부 이 함수가 "모델의 저해상도 출력"의
타당한 대리값이라는 가정 위에 있다. **독립 구현으로 대조한다.**

**판정선**: 어느 성분이든 상대차 1% 초과면 대리값 가정을 재검토한다.

---

C(Farneback 대조)는 이 스크립트가 아니라
`probe_warp_round2.py --flow farneback` 로 따로 돌린다.

사용:
  python scripts/branchC/probe_round3.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("TORCH_HOME", str(REPO / "artifacts/torch_home"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/branchB"))
sys.path.insert(0, str(REPO / "scripts/branchC"))
from wm_eval import scoring as S  # noqa: E402
from probe_residual_headroom import blur_residual, nearest_by_action  # noqa: E402
from probe_warp_vs_add import RaftFlow, warp, coarsen_flow, paired_t  # noqa: E402

ALPHAS = [0.0, 0.25, 0.35, 0.5, 0.75, 1.0]
KS = [1, 2, 4, 8]


def blur_residual_independent(res: torch.Tensor, factor: int) -> torch.Tensor:
    """`blur_residual` 의 **독립 구현** (F). avg_pool2d + nearest 업샘플.

    원본은 F.interpolate(mode='area') + F.interpolate(mode='bilinear') 다.
    area 축소는 정수 배율에서 avg_pool2d 와 같아야 하고, 업샘플 방식이 다르므로
    **완전히 같을 수는 없다.** 그래서 '얼마나 다른가'를 재는 것이 목적이다.
    차이가 크면 "이 연산이 모델의 저해상도 출력을 대리한다"는 가정이 연산 선택에
    민감하다는 뜻이고, 그러면 해상도 결론 전체가 흔들린다.
    """
    if factor <= 1:
        return res.clone()
    b, t, h, w, c = res.shape
    x = res.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    small = F.avg_pool2d(x, kernel_size=factor, stride=factor, ceil_mode=True)
    back = F.interpolate(small, size=(h, w), mode="nearest")
    return back.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)


def dv(r: dict) -> float:
    return 0.3 * r["dino"] + 0.3 * r["video"]


def tot(r: dict) -> float:
    return S.weighted_total(r["dino"], r["video"], r["action"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats",
                    default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--static",
                    default=str(REPO / "artifacts/branchB/m0_step1000_b4/static_preds"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "results/branchC/round3.json"))
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    sids = [m["sid"] for m in samples]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    VARIANTS = ["static"]
    VARIANTS += [f"mix{a:g}@{k}" for a in ALPHAS for k in KS]      # A
    VARIANTS += ["warpnbr_c4:1", "addnbrwarp4:1"]                  # B
    VARIANTS += ["warp:1", "warp_nearest:1"]                       # D
    VARIANTS += ["warpc2:1", "warpc2+res2:1"]                      # E
    VARIANTS += ["indep_blur2", "indep_blur4"]                     # F

    print(f"[r3] 표본 {len(sids)}개 · 변형 {len(VARIANTS)}개", flush=True)
    if len(sids) < 2:
        # nearest_by_action 은 자기 자신을 빼고 이웃을 찾으므로 표본이 1개면 None 을 준다.
        raise SystemExit("ERROR: '남의 잔차'가 필요하므로 표본이 2개 이상이어야 한다 "
                         "(--limit 1 로는 못 돌린다).")
    nbr = nearest_by_action(holdout, sids)
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=str(dev))
    gt_dir, static_dir = holdout / "gt_videos", Path(args.static)
    flower = RaftFlow(dev)

    print("[r3] 1단계 — 잔차·흐름 준비", flush=True)
    resid, flows = {}, {}
    for i, sid in enumerate(sids):
        gv = scorer._load_video(gt_dir, sid).float()
        sv = scorer._load_video(static_dir, sid).float()
        resid[sid] = (gv - sv).half()
        flows[sid] = flower(gv[0].permute(0, 3, 1, 2).round().clamp(0, 255)
                            .to(torch.uint8)).half()
        if (i + 1) % 24 == 0:
            print(f"[r3]   준비 {i+1}/{len(sids)}", flush=True)

    print("[r3] 2단계 — 채점 시작", flush=True)
    rows: dict[str, list] = {k: [] for k in VARIANTS}

    for i, sid in enumerate(sids):
        gt = scorer._load_video(gt_dir, sid)
        gv_f, gd_f = scorer.video_feature(gt)[0], scorer.dino_feature(gt)[0]
        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")
        sv = scorer._load_video(static_dir, sid).float()
        first = sv[0, 0].permute(2, 0, 1).to(dev)
        own, oth = resid[sid].float(), resid[nbr[sid]].float()
        fl = flows[sid].float().to(dev)
        fl_nbr = flows[nbr[sid]].float().to(dev)

        def as_video(x_chw: torch.Tensor) -> torch.Tensor:
            return x_chw.permute(0, 2, 3, 1).unsqueeze(0).round().clamp(0, 255).to(
                torch.uint8).cpu()

        for key in VARIANTS:
            if key == "static":
                v = sv
            elif key.startswith("mix"):                                    # A
                a, k = key[3:].split("@")
                mixed = float(a) * own + (1.0 - float(a)) * oth
                v = sv + blur_residual(mixed, int(k))
            elif key == "warpnbr_c4:1":                                    # B
                v = as_video(warp(first, coarsen_flow(fl_nbr, 4), 1.0)).float()
            elif key == "addnbrwarp4:1":                                   # B
                nw = as_video(warp(first, fl_nbr, 1.0)).float()
                v = sv + blur_residual(nw - sv, 4)
            elif key == "warp:1":                                          # D
                v = as_video(warp(first, fl, 1.0)).float()
            elif key == "warp_nearest:1":                                  # D
                v = as_video(warp(first, fl, 1.0, mode="nearest")).float()
            elif key == "warpc2:1":                                        # E
                v = as_video(warp(first, coarsen_flow(fl, 2), 1.0)).float()
            elif key == "warpc2+res2:1":                                   # E
                wv = as_video(warp(first, coarsen_flow(fl, 2), 1.0)).float()
                v = wv + blur_residual(gt.float() - wv, 2)
            else:                                                          # F
                v = sv + blur_residual_independent(own, int(key[-1]))
            mix = v.round().clamp(0, 255).to(torch.uint8)
            rows[key].append({
                "sid": sid,
                "dino": S.dino_component_frame_avg(scorer.dino_feature(mix)[0], gd_f),
                "video": S.video_component(scorer.video_feature(mix)[0], gv_f),
                "action": scorer.action_mae(mix, raw_actions),
            })
        if (i + 1) % 8 == 0:
            print(f"[r3] 채점 {i+1}/{len(sids)}", flush=True)

    means = {k: {c: float(np.mean([r[c] for r in rows[k]]))
                 for c in ("dino", "video", "action")} | {
                 "dv": float(np.mean([dv(r) for r in rows[k]])),
                 "total": float(np.mean([tot(r) for r in rows[k]]))} for k in VARIANTS}
    T = lambda k: [tot(r) for r in rows[k]]
    D = lambda k: [dv(r) for r in rows[k]]
    bt, bd = means["static"]["total"], means["static"]["dv"]

    GATES = {
        "A: mix0.35@2 vs mix0.35@4": ("mix0.35@2", "mix0.35@4"),
        "A(참고): mix1@2 vs mix1@4": ("mix1@2", "mix1@4"),
        "B: warpnbr_c4 vs addnbrwarp4": ("warpnbr_c4:1", "addnbrwarp4:1"),
        "D: warp_nearest vs warp(bilinear)": ("warp_nearest:1", "warp:1"),
        "E: warpc2+res2 vs mix1@2(=addblur2)": ("warpc2+res2:1", "mix1@2"),
        "E(참고): warpc2 vs mix1@2": ("warpc2:1", "mix1@2"),
        "F: 독립구현4 vs 원본4": ("indep_blur4", "mix1@4"),
        "F(참고): 독립구현2 vs 원본2": ("indep_blur2", "mix1@2"),
    }
    gates = {n: {"a": a, "b": b, "total": paired_t(T(a), T(b)), "dv": paired_t(D(a), D(b))}
             for n, (a, b) in GATES.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_samples": len(sids), "means": means, "gates": gates,
         "static_total": bt, "static_dv": bd, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- A: 격자 출력 ----
    W = 92
    print("\n" + "=" * W)
    print(f"A. α × k 격자 — static 대비 ΔTOTAL (음수여야 이득, n={len(sids)})")
    print("   α=1 방향 완벽 / α=0 통째로 남의 것 / **α≈0.35 가 우리 모델의 좌표**")
    print("=" * W)
    HDR = "α \\ k"          # f-string 안에 역슬래시를 못 넣는다(py3.10)
    print(f"{HDR:<10}" + "".join(f"{k:>12}" for k in KS))
    for a in sorted(ALPHAS, reverse=True):
        row = "".join(f"{means[f'mix{a:g}@{k}']['total'] - bt:>+12.5f}" for k in KS)
        mark = "   ← 우리 모델 근처" if abs(a - 0.35) < 1e-9 else ""
        print(f"{a:<10.2f}{row}{mark}")
    print(f"\n{HDR:<10}" + "".join(f"{k:>12}" for k in KS) + "     (ΔDV — 신뢰 축)")
    for a in sorted(ALPHAS, reverse=True):
        print(f"{a:<10.2f}" + "".join(
            f"{means[f'mix{a:g}@{k}']['dv'] - bd:>+12.5f}" for k in KS))

    # 우리 모델 좌표를 격자 위에서 다시 찾는다 (016 §9.2: static 대비 +0.02073)
    TARGET = 0.02073
    best = min(((abs(means[f"mix{a:g}@1"]["total"] - bt - TARGET), a) for a in ALPHAS))
    print(f"\n[좌표] k=1 열에서 static 대비 +{TARGET} 에 가장 가까운 α = {best[1]:g} "
          f"(차이 {best[0]:.5f})")
    print("       016 §9.2 는 '남의 잔차 α≈0.35' 라 했다. 매개화가 다르므로 값이 달라도 정상이다.")

    # ---- 게이트 ----
    print("\n" + "-" * W)
    print(f"{'짝지은 비교 (Δ<0 이면 왼쪽이 좋다)':<38}{'ΔDV':>10}{'t':>7}{'ΔTOTAL':>11}{'t':>7}{'승률':>8}")
    print("-" * W)
    for n, g in gates.items():
        print(f"{n:<38}{g['dv']['delta']:>+10.5f}{g['dv']['t']:>7.2f}"
              f"{g['total']['delta']:>+11.5f}{g['total']['t']:>7.2f}"
              f"{g['total']['wins']:>5}/{g['total']['n']}")
    print("-" * W)

    # ---- 사전 등록한 판정선을 기계적으로 적용 ----
    print("\n[판정] 018 §9-c 에 측정 전 고정한 선을 그대로 적용한다.")
    gA = gates["A: mix0.35@2 vs mix0.35@4"]
    if gA["dv"]["delta"] <= -0.005 and gA["dv"]["t"] <= -2:
        print("  A → H1(해상도 우선) **확정**")
    else:
        print("  A → H1 **강등**. α=0.35 에서 해상도를 올려도 이득이 0.005 미만이다")
    a35 = [means[f"mix0.35@{k}"]["total"] - bt for k in KS]
    if all(x > 0 for x in a35):
        print("  A → ⚠ α=0.35 행이 **전 k 에서 static 보다 나쁘다.** 잔차 노선 자체를 재검토하라")
    gB = gates["B: warpnbr_c4 vs addnbrwarp4"]
    print("  B → " + ("워핑이 덜 다친다" if gB["dv"]["delta"] <= -0.005 and gB["dv"]["t"] <= -2
                     else "덧셈이 덜 다친다" if gB["dv"]["delta"] >= 0.005 and gB["dv"]["t"] >= 2
                     else "**무승부. 바닥으로는 아무것도 못 말한다**"))
    dD = means["warp_nearest:1"]["dino"] - means["warp:1"]["dino"]
    print(f"  D → nearest 의 DINO 가 {-dD:+.5f} 만큼 좋다 → "
          + ("**보간 흐림이 상당분을 차지한다**" if dD <= -0.005 else "보간 흐림은 주범이 아니다"))
    gE = gates["E: warpc2+res2 vs mix1@2(=addblur2)"]
    print("  E → " + ("혼합 **재검토**" if gE["dv"]["delta"] <= -0.005 and gE["dv"]["t"] <= -2
                     else "혼합 **기각 유지**"))
    worst = max(abs(means[f"indep_blur{k}"][c] - means[f"mix1@{k}"][c])
                / (abs(means[f"mix1@{k}"][c]) + 1e-9)
                for k in (2, 4) for c in ("dino", "video", "action"))
    print(f"  F → 독립 구현과의 최대 상대차 {worst*100:.2f}% → "
          + ("⚠ **대리값 가정을 재검토하라**" if worst > 0.01 else "대리값 가정 유지"))
    print(f"\n[r3] 저장: {args.out}")


if __name__ == "__main__":
    main()
