"""워핑 vs 덧셈 2라운드 — 1라운드 검수가 지적한 구멍 넷을 메운다.

1라운드(`probe_warp_vs_add.py`)의 결론은 "덧셈으로 간다"였다. 적대적 검수가 그 결론
자체는 지지하면서도 **근거의 구멍 넷**을 지적했다. 결론을 바꿀 수도 있는 것들이라 다시 잰다.

무엇을 고치나
-------------
**① 뭉갬 배율 한 점에 결론이 걸려 있었다 [치명]**

    1라운드는 k=4 만 게이트로 걸었다. 저장된 rows 로 k=8 을 사후 계산하니:

        warpc4 vs addblur4   Δ=+0.01407  t=+2.98   덧셈 유의하게 유리
        warpc8 vs addblur8   Δ=+0.00389  t=+0.98   무승부

    부호는 같은데 크기가 3.6배, 유의성이 사라진다. 016 §3.7 이 "대조 실험 한 쌍으로
    단정하면 안 된다"고 못 박은 그 실수가 'seed 한 쌍' 대신 '배율 한 점'으로 재발했다.

    ⇒ **k ∈ {2, 4, 8, 16} 네 점을 전부 잰다.** 이게 이 데이터에서 얻을 수 있는
      **진짜 독립 반복**이다(α 를 0.25/0.5/1 로 바꾸는 것은 같은 장을 스칼라 배 한 것이라
      반복이 아니다).

**② 바닥(Gate 2)이 구조적으로 편향돼 있었다 [중대]**

    1라운드의 바닥 비교는 이랬다.

        warpnbr : 내 픽셀을 **남의 흐름**으로 민다        → 틀리지만 내 장면이다
        addnbr  : **남의 장면 잔차**를 통째로 얹는다      → 다른 방·다른 물체가 겹친다

    틀림의 급이 다르다. 이웃은 **행동 시퀀스 거리로만** 골랐고 장면 유사도는 안 봤는데,
    표본들이 서로 다른 데이터셋 출신이라 addnbr 은 생판 다른 방의 델타를 이식한다.
    그러니 "워핑이 틀려도 덜 다친다"는 워핑의 강건성이 아니라 **매개화의 산물**이다.

    ⇒ **공정한 바닥**을 새로 만든다. 같은 틀린 움직임을 덧셈 표현으로 바꿔 얹는다.

        addnbrwarp = static + blur_k( warp(첫프레임, 남의흐름, 1) − static )

      이러면 두 표현이 **동일한 오차**(남의 움직임, 내 픽셀)를 지고 표현만 달라진다.

**③ "완벽한 흐름"이라는 Gate 0 의 전제가 흔들린다 [치명]**

    warp:1 은 오라클이 아니라 **RAFT 추정치**다. 누적 변위가 p99 55px, max 97px 인데
    (512폭 화면의 19%) 이건 RAFT 가 편한 구간이 아니다.
    검수가 덤프 표본에서 프레임별 픽셀 L1 을 재 보니 **t=1 에서는 워핑이 크게 이기고
    t=15 에서는 정지영상에 진다.**

    ⇒ **프레임별 DINO 를 저장**한다. t 에 따라 손해가 커지면 흐름 품질이 범인이고,
      t 와 무관하게 평평하면 가림/드러남이 범인이다. 이 구분이 없으면
      "왜 워핑을 버렸는가"를 설계서 검수에서 방어할 수 없다.

**④ 새너티 체크가 약속만 되고 실행되지 않았다 [중대]**

    1라운드 독스트링은 "α=0 → 정확히 static" 을 새너티로 내세웠는데 변형 목록에 없었다.
    `grid_sample` 좌표 정규화와 `align_corners` 의 짝을 한 방에 검증하는 가장 값싼 검사다.

    ⇒ **`warp:0` 을 넣는다.** static 과 소수점 전부 일치해야 한다.

그리고 판정을 TOTAL 이 아니라 DV 로도 같이 본다
-----------------------------------------------
016 §5 는 **"로컬 Action 값으로 내린 판단은 전부 폐기 대상"** 이라고 못 박았다
(로컬에서는 정답 영상조차 static 과 1.4% 밖에 차이가 안 난다 — 변별력이 없다).
그런데 1라운드 게이트는 전부 TOTAL(=0.3D+0.3V+**0.4A**) 위에 있었다.
판정의 3분의 1이 무효 선언된 자로 잰 값이었다.

  ⇒ 앞으로 모든 비교에 **DV = 0.3·DINO + 0.3·Video** 를 병기한다.
    로컬에서 신뢰할 수 있는 축은 이것뿐이다.

사용:
  python scripts/branchC/probe_warp_round2.py
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

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/branchB"))
sys.path.insert(0, str(REPO / "scripts/branchC"))
from wm_eval import scoring as S  # noqa: E402
from probe_residual_headroom import blur_residual, nearest_by_action  # noqa: E402
from probe_warp_vs_add import RaftFlow, FarnebackFlow, warp, coarsen_flow, paired_t  # noqa: E402

KS = [2, 4, 8, 16]


def dino_per_frame(pred: np.ndarray, gt: np.ndarray) -> list[float]:
    """프레임별 (1 − 코사인). 평균이 곧 채점의 DINO 성분이다."""
    return [float(1.0 - np.dot(pred[t], gt[t]) /
                  (np.linalg.norm(pred[t]) * np.linalg.norm(gt[t]) + 1e-8))
            for t in range(pred.shape[0])]


def dv(r: dict) -> float:
    """DV = 0.3·DINO + 0.3·Video. 로컬에서 신뢰할 수 있는 축(016 §5)."""
    return 0.3 * r["dino"] + 0.3 * r["video"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats",
                    default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--static",
                    default=str(REPO / "artifacts/branchB/m0_step1000_b4/static_preds"))
    ap.add_argument("--flow", choices=["raft", "farneback"], default="raft")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=str(REPO / "results/branchC"))
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    sids = [m["sid"] for m in samples]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # staticexact = 첫 프레임을 16번 그대로 복사(mp4 왕복 없음).
    # static(=mp4 로 저장했다 읽은 것)과의 차이가 곧 **인코딩 잡음**이다.
    # libx264 는 입력 16프레임이 전부 같아도 디코드하면 1.95% 픽셀이 달라진다(최대 7단계).
    # 그래서 warp:0 을 static 과 비교하면 새너티가 통과할 수 없다 — staticexact 와 비교해야 한다.
    VARIANTS = ["static", "staticexact", "warp:0"]         # ④ 새너티 + 인코딩 잡음 측정
    VARIANTS += [f"warpc{k}:1" for k in KS]                # ① 워핑 배율 훑기
    VARIANTS += ["warp:1"]                                 # 원본 해상도 흐름
    VARIANTS += [f"addblur{k}:1" for k in KS]              # ① 덧셈 배율 훑기
    VARIANTS += ["add:1"]                                  # 새너티 (정답과 동일)
    VARIANTS += ["warpnbr:1", "addnbr:1"]                  # 1라운드 바닥 (재현용)
    VARIANTS += [f"addnbrwarp{k}:1" for k in [4]]          # ② 공정한 바닥
    VARIANTS += ["warpc4+res4:1"]                          # 혼합 (배율 이야기의 기준점)

    print(f"[r2] 표본 {len(sids)}개 · 흐름 {args.flow} · 변형 {len(VARIANTS)}개", flush=True)
    nbr = nearest_by_action(holdout, sids)
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=str(dev))
    gt_dir, static_dir = holdout / "gt_videos", Path(args.static)
    flower = RaftFlow(dev) if args.flow == "raft" else FarnebackFlow(dev)

    print("[r2] 1단계 — 잔차·흐름 준비", flush=True)
    resid, flows = {}, {}
    for i, sid in enumerate(sids):
        gv = scorer._load_video(gt_dir, sid).float()
        sv = scorer._load_video(static_dir, sid).float()
        resid[sid] = (gv - sv).half()
        flows[sid] = flower(gv[0].permute(0, 3, 1, 2).round().clamp(0, 255)
                            .to(torch.uint8)).half()
        if (i + 1) % 24 == 0:
            print(f"[r2]   준비 {i+1}/{len(sids)}", flush=True)

    print("[r2] 2단계 — 채점 시작", flush=True)
    rows: dict[str, list] = {k: [] for k in VARIANTS}
    diag: list[dict] = []                         # ③ 표본별로 남긴다 (평균으로 뭉개지 않는다)

    for i, sid in enumerate(sids):
        gt = scorer._load_video(gt_dir, sid)
        gv_f, gd_f = scorer.video_feature(gt)[0], scorer.dino_feature(gt)[0]
        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")
        sv = scorer._load_video(static_dir, sid).float()
        first = sv[0, 0].permute(2, 0, 1).to(dev)
        own = resid[sid].float()
        fl = flows[sid].float().to(dev)
        fl_nbr = flows[nbr[sid]].float().to(dev)
        flc = {k: coarsen_flow(fl, k) for k in KS}

        def as_video(x_chw: torch.Tensor) -> torch.Tensor:
            return x_chw.permute(0, 2, 3, 1).unsqueeze(0).round().clamp(0, 255).to(
                torch.uint8).cpu()

        # ② 공정한 바닥: 남의 흐름이 만든 **움직임**을 덧셈 표현으로 바꿔 얹는다
        nbr_warp = as_video(warp(first, fl_nbr, 1.0)).float()
        nbr_as_add = {k: blur_residual(nbr_warp - sv, k) for k in [4]}

        for key in VARIANTS:
            if key == "static":
                v = sv
            elif key == "staticexact":
                v = sv[:, 0:1].expand(-1, sv.shape[1], -1, -1, -1)
            elif key == "add:1":
                v = sv + own
            elif key.startswith("addblur"):
                v = sv + blur_residual(own, int(key[7:].split(":")[0]))
            elif key.startswith("addnbrwarp"):
                v = sv + nbr_as_add[int(key[10:].split(":")[0])]
            elif key == "addnbr:1":
                v = sv + resid[nbr[sid]].float()
            elif key == "warpc4+res4:1":
                wv = as_video(warp(first, flc[4], 1.0)).float()
                v = wv + blur_residual(gt.float() - wv, 4)
            elif key == "warpnbr:1":
                v = nbr_warp
            else:                                   # warp:0 / warp:1 / warpcK:1
                head, a = key.split(":")
                f = fl if head == "warp" else flc[int(head[5:])]
                v = as_video(warp(first, f, float(a))).float()

            mix = v.round().clamp(0, 255).to(torch.uint8)
            pd = scorer.dino_feature(mix)[0]
            rows[key].append({
                "sid": sid,
                "dino": S.dino_component_frame_avg(pd, gd_f),
                "video": S.video_component(scorer.video_feature(mix)[0], gv_f),
                "action": scorer.action_mae(mix, raw_actions),
                "dino_t": dino_per_frame(pd, gd_f),      # ③ 프레임별
            })

        # ③ 진단을 표본별로 남긴다. 레터박스와 t=0 을 빼고 변위를 잰다.
        # 레터박스(검은 띠)와 t=0(흐름이 정확히 0)을 빼야 변위 백분위가 희석되지 않는다.
        mag = fl.norm(dim=1)[1:]                                   # (T-1, H, W), t≥1
        hmask = (sv[0].abs().sum(dim=(0, 2, 3)) > 0)               # 내용이 있는 행(H)
        wmask = (sv[0].abs().sum(dim=(0, 1, 3)) > 0)               # 내용이 있는 열(W)
        inner = mag[:, hmask][:, :, wmask].cpu().numpy()
        w1 = as_video(warp(first, fl, 1.0)).float()
        diag.append({
            "sid": sid,
            "warp1_l1_t": [float((w1[0, t] - gt.float()[0, t]).abs().mean())
                           for t in range(w1.shape[1])],
            "static_l1_t": [float((sv[0, t] - gt.float()[0, t]).abs().mean())
                            for t in range(sv.shape[1])],
            "flow_p95": float(np.percentile(inner, 95)),
            "flow_p99": float(np.percentile(inner, 99)),
            "flow_max": float(inner.max()),
        })
        if (i + 1) % 8 == 0:
            print(f"[r2] 채점 {i+1}/{len(sids)}", flush=True)

    # ---- 집계 ----
    def agg(k, fn):
        return float(np.mean([fn(r) for r in rows[k]]))
    means = {k: {"dino": agg(k, lambda r: r["dino"]),
                 "video": agg(k, lambda r: r["video"]),
                 "action": agg(k, lambda r: r["action"]),
                 "dv": agg(k, dv),
                 "total": agg(k, lambda r: S.weighted_total(r["dino"], r["video"], r["action"]))}
             for k in VARIANTS}

    def series(k, fn):
        return [fn(r) for r in rows[k]]
    T = lambda k: series(k, lambda r: S.weighted_total(r["dino"], r["video"], r["action"]))
    D = lambda k: series(k, dv)

    GATES = {f"k{k}: warpc{k} vs addblur{k}": (f"warpc{k}:1", f"addblur{k}:1") for k in KS}
    GATES["공정바닥: warpnbr vs addnbrwarp4"] = ("warpnbr:1", "addnbrwarp4:1")
    GATES["구바닥(무효): warpnbr vs addnbr"] = ("warpnbr:1", "addnbr:1")
    GATES["혼합 vs addblur4"] = ("warpc4+res4:1", "addblur4:1")
    gates = {n: {"a": a, "b": b,
                 "total": paired_t(T(a), T(b)), "dv": paired_t(D(a), D(b))}
             for n, (a, b) in GATES.items()}

    tag = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.outdir) / f"warp_round2_{args.flow}{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"n_samples": len(sids), "flow_estimator": flower.name, "means": means,
         "gates": gates, "nbr": nbr, "diagnostics": diag, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 출력 ----
    base_t, base_d = means["static"]["total"], means["static"]["dv"]
    W = 96
    print("\n" + "=" * W)
    print(f"워핑 vs 덧셈 2라운드 (n={len(sids)}, 흐름={flower.name}, 전부 낮을수록 좋다)")
    print("=" * W)
    print(f"{'변형':<22}{'DINO':>9}{'Video':>9}{'Action':>9}{'DV':>9}{'TOTAL':>9}"
          f"{'ΔDV':>10}{'ΔTOTAL':>10}")
    print("-" * W)
    for k in VARIANTS:
        m = means[k]
        print(f"{k:<22}{m['dino']:>9.5f}{m['video']:>9.5f}{m['action']:>9.5f}"
              f"{m['dv']:>9.5f}{m['total']:>9.5f}"
              f"{m['dv']-base_d:>+10.5f}{m['total']-base_t:>+10.5f}")
    print("-" * W)

    print(f"\n{'짝지은 비교 (Δ<0 이면 왼쪽이 좋다)':<36}"
          f"{'ΔDV':>10}{'t':>7}{'ΔTOTAL':>11}{'t':>7}{'승률':>9}")
    print("-" * W)
    for n, g in gates.items():
        print(f"{n:<36}{g['dv']['delta']:>+10.5f}{g['dv']['t']:>7.2f}"
              f"{g['total']['delta']:>+11.5f}{g['total']['t']:>7.2f}"
              f"{g['total']['wins']:>6}/{g['total']['n']}")
    print("-" * W)

    # ④ 새너티 — warp:0 은 '첫 프레임 16번 복사'와 정확히 같아야 한다.
    #    static(mp4 왕복본)과 비교하면 인코딩 잡음 때문에 통과할 수 없다.
    s0 = max(abs(means['warp:0'][c] - means['staticexact'][c])
             for c in ("dino", "video", "action"))
    print(f"\n[새너티] warp:0 vs staticexact 최대 성분 차이 = {s0:.2e}  "
          f"({'통과' if s0 < 1e-6 else '⚠ 실패 — grid_sample 좌표 규약을 의심하라'})")
    enc = {c: means['static'][c] - means['staticexact'][c]
           for c in ("dino", "video", "action")}
    print(f"[인코딩 잡음] static − staticexact:  DINO {enc['dino']:+.5f}  "
          f"Video {enc['video']:+.5f}  Action {enc['action']:+.5f}  "
          f"TOTAL {means['static']['total'] - means['staticexact']['total']:+.5f}")
    print("              양수면 mp4 왕복이 점수를 깎고 있다는 뜻이다(제출은 mp4 로만 가능하다).")
    print(f"[새너티] add:1 의 DINO={means['add:1']['dino']:.2e} "
          f"Video={means['add:1']['video']:.2e} (정답과 같아야 하므로 0)")

    # ③ 프레임별 — 워핑의 손해가 t 에 비례하는가
    print("\n[프레임별] DINO 거리 (static 대비 차이). t 에 비례해 커지면 흐름 품질이,")
    print("           t 와 무관하게 평평하면 가림/드러남이 범인이다.")
    st = np.array([r["dino_t"] for r in rows["static"]]).mean(axis=0)
    print(f"  {'t':<4}" + "".join(f"{t:>7}" for t in range(0, 16, 3)))
    print(f"  {'static':<4}" + "".join(f"{st[t]:>7.4f}" for t in range(0, 16, 3)))
    for k in ["warp:1", "warpc4:1", "addblur4:1"]:
        a = np.array([r["dino_t"] for r in rows[k]]).mean(axis=0) - st
        print(f"  {k:<11}" + "".join(f"{a[t]:>+7.4f}" for t in range(0, 16, 3)))
    l1w = np.array([d["warp1_l1_t"] for d in diag]).mean(axis=0)
    l1s = np.array([d["static_l1_t"] for d in diag]).mean(axis=0)
    print("\n[프레임별] 픽셀 L1 (워핑 − 정지영상). 음수여야 흐름이 일한 것이다.")
    print(f"  {'t':<11}" + "".join(f"{t:>7}" for t in range(0, 16, 3)))
    print(f"  {'차이':<10}" + "".join(f"{l1w[t]-l1s[t]:>+7.3f}" for t in range(0, 16, 3)))

    p99 = float(np.mean([d["flow_p99"] for d in diag]))
    print(f"\n[진단] 누적 변위(레터박스·t=0 제외)  p95 "
          f"{np.mean([d['flow_p95'] for d in diag]):.1f}  p99 {p99:.1f}  "
          f"max {np.mean([d['flow_max'] for d in diag]):.1f} 픽셀")
    print(f"       커널 기반 변환(CDNA/SepConv)은 커널 ≥ {2*p99:.0f}px 가 필요하다.")
    print(f"\n[r2] 저장: {out_path}")


if __name__ == "__main__":
    main()
