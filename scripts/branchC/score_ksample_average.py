"""확산 샘플 K개를 평균내면 점수가 좋아지는가 — 학습 없이, 이미 가진 모델로.

왜 이걸 하는가
--------------
017 §4 에서 찾은 것이다. 우리와 거의 같은 대회
(1X World Model Challenge — 휴머노이드 로봇, 행동조건, **단일 정답 PSNR** 채점)의
우승팀이 확산모델을 버리지 않았다. 샘플 20개를 픽셀공간에서 평균냈다.

    샘플  1 →  5      +1.89 dB   (이득의 84%)
    샘플  1 → 20      +2.25 dB
    균일 가우시안 블러 후처리   +1.2 dB     ← 평균이 더 좋다
    CFG 2.0 켜기      −0.68 dB              ← 우리는 이미 꺼져 있다(gen_1p1b.yaml)

이유는 이론에 있다(017 §3). 거리로 채점하는 시험에서 "샘플 하나를 뽑는" 대가는
**정확히 최대 2배**다. 코사인 거리에서도 같은 인수가 나온다.

    최적 = 1 − ‖m‖        샘플 하나 = 1 − ‖m‖²        비율 = 1 + ‖m‖ ∈ [1, 2]

    (m = 정답 특징들의 평균 방향. 여러 그럴듯한 미래가 있으면 ‖m‖ < 1 이 된다)

⚠ 평균은 **흔들림(분산)을 지우지 치우침(편향)을 못 고친다**
------------------------------------------------------------
016 §9.2 는 우리 모델의 값어치를 "남의 움직임을 35% 세기로 갖다 붙인 것"이라 쟀다.

    그 틀림이 **분산**이면(샘플마다 다른 방향으로 흔들린다)  →  평균이 고친다
    그 틀림이 **편향**이면(일관되게 틀린 방향으로 간다)      →  평균은 아무것도 못 한다

그래서 이 스크립트는 **먼저 흔들림의 크기를 잰다.** 샘플들이 서로 비슷하면
평균낼 것이 없으므로 K 를 키우는 데 GPU 를 더 쓸 이유가 없다.

    흔들림 = K개 샘플의 픽셀별 표준편차
    비교 기준 = 우리가 맞혀야 할 변화량, 즉 |정답 − 정지영상|

    흔들림 / 변화량 이 크면   →  분산이 많다. 평균에 여지가 있다
    흔들림 / 변화량 이 작으면 →  모델이 확신을 갖고 틀린다. 평균은 소용없다

대리 실험은 이미 실패했다 — 그런데 그 실패는 이 질문의 답이 아니다
----------------------------------------------------------------
`probe_blur_and_averaging.py` 로 "남의 잔차 K개 평균"을 재 봤더니 K 를 키울수록
**오히려 나빠졌다**(avg2/4/8 전부 +0.005~+0.016, 단조성 없음).

그러나 이건 대리값의 한계다(017 §8.3 에 미리 적어 뒀다).

    남의 잔차   : **체계적으로** 틀리다(다른 장면). K개를 평균내면 K개 장면이 겹칠 뿐이다
    확산 샘플   : 같은 조건에서 **무작위로** 흔들린다. 평균이 조건부 평균으로 수렴한다

구조가 다르므로 대리 실험의 실패로 이 실험을 접을 수 없다. 다만 우선순위는 내려간다.

제출 경로 그대로 잰다
---------------------
평균 영상을 **mp4 로 저장한 뒤 디스크에서 읽어** 채점한다. 실제 제출이 그 경로이기 때문이다.
메모리 상태로 바로 채점한 값도 같이 내서 **인코딩이 얼마나 깎아먹는지**를 함께 본다
(libx264 는 완전한 정지영상조차 흔든다 — 1.95% 픽셀·최대 7단계, 2라운드에서 확인).

사용:
  # 1) 먼저 seed 를 바꿔 K번 생성한다 (run_1p1b_generate.sh 의 7번째 인자가 seed)
  # 2) 그다음 이 스크립트로 채점한다
  python scripts/branchC/score_ksample_average.py \
      --pred-roots artifacts/branchC/ksample/seed0 artifacts/branchC/ksample/seed1 ...
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

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/branchC"))
from wm_eval import scoring as S  # noqa: E402
from probe_warp_vs_add import paired_t  # noqa: E402


def dv(r: dict) -> float:
    """DV = 0.3·DINO + 0.3·Video. 로컬에서 신뢰할 수 있는 축(016 §5)."""
    return 0.3 * r["dino"] + 0.3 * r["video"]


def tot(r: dict) -> float:
    return S.weighted_total(r["dino"], r["video"], r["action"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-roots", nargs="+", required=True,
                    help="seed 별 생성 결과 디렉터리들 (K개)")
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats",
                    default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--static",
                    default=str(REPO / "artifacts/branchB/m0_step1000_b4/static_preds"))
    ap.add_argument("--avg-dir", default=str(REPO / "artifacts/branchC/ksample/avg"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "results/branchC/ksample_average.json"))
    args = ap.parse_args()

    roots = [Path(p) for p in args.pred_roots]
    K = len(roots)
    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    sids = [m["sid"] for m in samples]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for r in roots:
        n = len(list(r.glob("*.mp4")))
        if n < len(sids):
            raise SystemExit(f"ERROR: {r} 에 mp4 가 {n}개뿐이다 (필요 {len(sids)}). "
                             "생성이 덜 끝났을 수 있다 — 파일 존재 ≠ 완성이다(016 함정).")

    KS = [k for k in (1, 2, 3, 4, 5, 8, 10, 16, 20) if k <= K]
    VARIANTS = ["static"] + [f"one:{j}" for j in range(K)] \
                          + [f"avg{k}" for k in KS] + [f"avg{k}_mp4" for k in KS]

    print(f"[ks] 표본 {len(sids)}개 · 샘플 {K}개 · K 훑기 {KS}", flush=True)
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=str(dev))
    gt_dir, static_dir = holdout / "gt_videos", Path(args.static)
    avg_dir = Path(args.avg_dir)

    rows: dict[str, list] = {k: [] for k in VARIANTS}
    spread: list[dict] = []

    for i, sid in enumerate(sids):
        gt = scorer._load_video(gt_dir, sid)
        gv_f, gd_f = scorer.video_feature(gt)[0], scorer.dino_feature(gt)[0]
        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")
        sv = scorer._load_video(static_dir, sid).float()
        preds = [scorer._load_video(r, sid).float() for r in roots]   # K × (1,T,H,W,3)

        # --- 흔들림 진단: 샘플끼리 얼마나 다른가 ---
        stack = torch.stack(preds)                       # (K,1,T,H,W,3)
        sd = stack.std(dim=0, unbiased=True) if K > 1 else torch.zeros_like(sv)
        change = (gt.float() - sv).abs()                 # 맞혀야 할 변화량
        spread.append({
            "sid": sid,
            "sample_sd": float(sd.mean()),
            "gt_change": float(change.mean()),
            "ratio": float(sd.mean() / (change.mean() + 1e-8)),
            "pair_l1": float(torch.stack([(preds[a] - preds[b]).abs().mean()
                                          for a in range(K) for b in range(a + 1, K)]).mean())
                       if K > 1 else 0.0,
        })

        for key in VARIANTS:
            if key == "static":
                v = sv
            elif key.startswith("one:"):
                v = preds[int(key[4:])]
            else:
                k = int(key[3:].split("_")[0])
                v = torch.stack(preds[:k]).mean(dim=0)
                if key.endswith("_mp4"):
                    # 제출 경로 그대로 — mp4 로 쓴 뒤 디스크에서 다시 읽는다
                    d = avg_dir / f"k{k}"
                    d.mkdir(parents=True, exist_ok=True)
                    arr = v.round().clamp(0, 255).to(torch.uint8)[0].numpy()
                    with imageio.get_writer(d / f"{sid}.mp4", fps=6, codec="libx264",
                                            macro_block_size=1) as w:
                        for fr in arr:
                            w.append_data(fr)
                    v = scorer._load_video(d, sid).float()
            mix = v.round().clamp(0, 255).to(torch.uint8)
            rows[key].append({
                "sid": sid,
                "dino": S.dino_component_frame_avg(scorer.dino_feature(mix)[0], gd_f),
                "video": S.video_component(scorer.video_feature(mix)[0], gv_f),
                "action": scorer.action_mae(mix, raw_actions),
            })
        if (i + 1) % 8 == 0:
            print(f"[ks] 채점 {i+1}/{len(sids)}", flush=True)

    means = {k: {c: float(np.mean([r[c] for r in rows[k]]))
                 for c in ("dino", "video", "action")} | {
                 "dv": float(np.mean([dv(r) for r in rows[k]])),
                 "total": float(np.mean([tot(r) for r in rows[k]]))}
             for k in VARIANTS}
    # 개별 샘플 K개의 평균 — "샘플 하나"의 대표값
    one_keys = [f"one:{j}" for j in range(K)]
    means["one_mean"] = {c: float(np.mean([means[k][c] for k in one_keys]))
                         for c in ("dino", "video", "action", "dv", "total")}

    Tv = lambda k: [tot(r) for r in rows[k]]
    Dv = lambda k: [dv(r) for r in rows[k]]
    gates = {}
    for k in KS:
        if k == 1:
            continue
        gates[f"avg{k} vs one:0"] = {"total": paired_t(Tv(f"avg{k}"), Tv("one:0")),
                                     "dv": paired_t(Dv(f"avg{k}"), Dv("one:0"))}
    gates["avg_max vs static"] = {
        "total": paired_t(Tv(f"avg{KS[-1]}"), Tv("static")),
        "dv": paired_t(Dv(f"avg{KS[-1]}"), Dv("static"))}

    out = {"n_samples": len(sids), "K": K, "roots": [str(r) for r in roots],
           "means": means, "gates": gates, "spread": spread, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    base_t, base_d = means["static"]["total"], means["static"]["dv"]
    W = 92
    print("\n" + "=" * W)
    print(f"확산 샘플 K개 평균 (n={len(sids)}, K={K}, 전부 낮을수록 좋다)")
    print("=" * W)
    print(f"{'변형':<18}{'DINO':>9}{'Video':>9}{'Action':>9}{'DV':>9}{'TOTAL':>9}"
          f"{'ΔDV':>10}{'ΔTOTAL':>10}")
    print("-" * W)
    for k in ["static", "one_mean"] + [f"avg{x}" for x in KS] + [f"avg{x}_mp4" for x in KS]:
        m = means[k]
        print(f"{k:<18}{m['dino']:>9.5f}{m['video']:>9.5f}{m['action']:>9.5f}"
              f"{m['dv']:>9.5f}{m['total']:>9.5f}"
              f"{m['dv']-base_d:>+10.5f}{m['total']-base_t:>+10.5f}")
    print("-" * W)

    print(f"\n{'짝지은 비교 (Δ<0 이면 왼쪽이 좋다)':<30}"
          f"{'ΔDV':>10}{'t':>7}{'ΔTOTAL':>11}{'t':>7}{'승률':>9}")
    print("-" * W)
    for n, g in gates.items():
        print(f"{n:<30}{g['dv']['delta']:>+10.5f}{g['dv']['t']:>7.2f}"
              f"{g['total']['delta']:>+11.5f}{g['total']['t']:>7.2f}"
              f"{g['total']['wins']:>6}/{g['total']['n']}")
    print("-" * W)

    r = float(np.mean([s["ratio"] for s in spread]))
    print(f"\n[흔들림] 샘플 간 픽셀 표준편차 {np.mean([s['sample_sd'] for s in spread]):.3f}")
    print(f"         맞혀야 할 변화량        {np.mean([s['gt_change'] for s in spread]):.3f}")
    print(f"         비율 {r:.3f}  ← 작으면 모델이 '확신을 갖고 틀리는' 것이라 평균이 소용없다")
    print(f"         샘플 두 개 사이 평균 L1 {np.mean([s['pair_l1'] for s in spread]):.3f}")

    for k in KS:
        gap = means[f"avg{k}_mp4"]["total"] - means[f"avg{k}"]["total"]
        print(f"[인코딩 틈] K={k}: mp4 왕복이 TOTAL 을 {gap:+.5f} 만큼 바꾼다")
    print(f"\n[ks] 저장: {args.out}")


if __name__ == "__main__":
    main()
