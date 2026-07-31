"""생성 영상과 정지영상(첫 프레임 복사)을 α 로 섞어보며 점수 곡선을 낸다.

왜 이런 걸 하나
--------------
실측으로 두 가지가 동시에 참이다.
  · **Video 성분(30%)** 은 우리 생성이 static 을 이긴다 (0.0849 < 0.0911).
  · **DINO 성분(30%)** 은 static 이 우리 생성을 이긴다 (0.123 < 0.221).
두 성분은 서로 다른 신경망이 서로 다른 것을 본다. 그래서 "생성과 정지 사이 어딘가"가
두 성분의 합을 가장 낮게 만들 수 있다. 그 지점이 있는지, 있다면 어디인지를 재는 도구다.

  mix(t)      = α · 생성(t) + (1−α) · 첫프레임      (α=1 이면 생성 그대로, α=0 이면 static)

여기에 더 나은 변형이 하나 더 있다. 실측에 따르면 생성 영상은 **t=0 에서 이미 0.048 만큼 틀린다**
(static 은 0.004). 첫 프레임은 문제에서 입력으로 주어진 이미지인데도 그렇다. 모델이 latent 로 갔다
오면서 잃는 재구성 오차이고, 이 오차는 t=0 뿐 아니라 **모든 프레임에 계속 실려 있다.**
그렇다면 생성 결과에서 "움직인 만큼"만 떼어내 진짜 첫 프레임에 얹으면 그 오차가 상쇄된다.

  residual(t) = 첫프레임 + α · (생성(t) − 생성(0))      (t=0 에서 정확히 첫프레임이 된다)

이건 학습이 아니라 **후처리**다. 첫 프레임은 문제에서 입력으로 주어진 이미지이므로
정답 정보를 쓰는 것이 아니다(eval 데이터를 추론 입력으로만 쓰는 규칙을 지킨다).

두 단계로 쓴다
-------------
  1) `--fast` (기본): mp4 로 저장하지 않고 메모리에서 섞어 바로 채점한다. α 곡선을 빨리 본다.
     실제 제출은 mp4 를 거치므로 이 값은 아주 약간 낙관적일 수 있다.
  2) `--save-dir` 지정: 고른 α 로 mp4 를 실제로 써서, 정식 채점 경로로 검증한다.

사용
----
  python scripts/branchB/blend_static_sweep.py \
      --pred artifacts/branchB/preds_step1000_emafix \
      --static artifacts/branchB/m0_step1000_emafix/static_preds \
      --alphas 0,0.25,0.5,0.75,1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from wm_eval import data_utils as D  # noqa: E402
from wm_eval import scoring as S  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats", default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--pred", required=True, help="생성 영상 디렉터리")
    ap.add_argument("--static", required=True, help="static 예측 디렉터리(첫 프레임 16복사)")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--mode", default="both", choices=["mix", "residual", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/blend_sweep.json"))
    ap.add_argument("--save-dir", default=None, help="지정하면 이 α 들의 mp4 를 실제로 저장한다")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]

    print(f"[blend] 채점 모델 로딩 ... (n={len(samples)}, α={alphas})", flush=True)
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=args.device)
    gt_dir = holdout / "gt_videos"

    modes = ["mix", "residual"] if args.mode == "both" else [args.mode]
    # α=0 은 두 모드에서 동일하게 static 이므로 한 번만 잰다.
    variants = [(mo, a) for mo in modes for a in alphas if not (mo == "residual" and a == 0.0)]
    rows: dict[tuple, list] = {k: [] for k in variants}

    for i, m in enumerate(samples):
        sid = m["sid"]
        gt = scorer._load_video(gt_dir, sid)
        gv, gd = scorer.video_feature(gt)[0], scorer.dino_feature(gt)[0]
        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")

        pv = scorer._load_video(Path(args.pred), sid).float()      # (1,16,320,512,3)
        sv = scorer._load_video(Path(args.static), sid).float()
        delta = pv - pv[:, :1]                                     # 생성이 예측한 변화분(t=0 에서 0)

        for mo, a in variants:
            v = (a * pv + (1.0 - a) * sv) if mo == "mix" else (sv + a * delta)
            mix = v.round().clamp(0, 255).to(torch.uint8)
            if args.save_dir:
                d = Path(args.save_dir) / f"{mo}_alpha{a:g}"
                d.mkdir(parents=True, exist_ok=True)
                D.save_mp4_uint8(mix.numpy()[0], d / f"{sid}.mp4", fps=m["fps"])
            rows[(mo, a)].append({
                "dino": S.dino_component_frame_avg(scorer.dino_feature(mix)[0], gd),
                "video": S.video_component(scorer.video_feature(mix)[0], gv),
                "action": scorer.action_mae(mix, raw_actions),
            })
        if (i + 1) % 16 == 0:
            print(f"[blend] {i+1}/{len(samples)}", flush=True)

    out = {"n_samples": len(samples), "pred": args.pred, "static": args.static, "alphas": {}}
    for mo, a in variants:
        r = rows[(mo, a)]
        d = float(np.mean([x["dino"] for x in r]))
        v = float(np.mean([x["video"] for x in r]))
        ac = float(np.mean([x["action"] for x in r]))
        out["alphas"][f"{mo}:{a:g}"] = {"mode": mo, "alpha": a, "dino": d, "video": v, "action": ac,
                                        "total": S.weighted_total(d, v, ac)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 82)
    print(f"블렌드 스윕 (n={len(samples)}, 낮을수록 좋음)")
    print("  mix      = α·생성 + (1−α)·첫프레임      (α=0 → static, α=1 → 생성 그대로)")
    print("  residual = 첫프레임 + α·(생성−생성[0])  (재구성 오차를 상쇄하고 움직임만 얹는다)")
    print("=" * 82)
    print(f"{'변형':>16}{'DINO':>12}{'Video':>12}{'Action':>12}{'TOTAL':>14}")
    print("-" * 82)
    best = min(out["alphas"].items(), key=lambda kv: kv[1]["total"])
    for a, r in sorted(out["alphas"].items(), key=lambda kv: (kv[1]["mode"], kv[1]["alpha"])):
        mark = "  ← 최소" if a == best[0] else ""
        print(f"{a:>16}{r['dino']:>12.5f}{r['video']:>12.5f}{r['action']:>12.5f}{r['total']:>14.5f}{mark}")
    print("-" * 82)
    print("  참고: static 실측 TOTAL 0.56032 / 1.1B step1000 0.58911 / 정답영상 0.48911")
    print(f"\n[blend] 저장 → {args.out}")


if __name__ == "__main__":
    main()
