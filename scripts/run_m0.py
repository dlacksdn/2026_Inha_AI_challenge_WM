"""M0 바닥값 측정 — 로컬 모의채점으로 세 예측기의 점수를 낸다.

세 예측기:
  static  : 시작 이미지를 16번 복사 (아무 것도 예측 안 함 = 바닥)
  gt      : 정답 영상 그대로 (상한/새너티 — DINO/Video 거리 ~0, Action은 추출기 irreducible 오차)
  baseline: 주최측 11M 모델 생성 영상 (있을 때만; --pred-dir 로 지정)

각 예측기별로 0.3 DINO + 0.3 Video + 0.4 Action 을 계산해 표로 출력하고 JSON 저장.
DINO는 서버 집계가 미확인이므로 frame-avg(기본)와 flatten(대조) 둘 다 리포트.

사용:
  python scripts/run_m0.py --holdout artifacts/holdout --submission-kit open/submission_kit \
      --action-stats open/data/train/so100_action_statistics.json --out artifacts/m0
  # baseline 영상이 있으면:  --pred-dir artifacts/baseline_preds --pred-name baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wm_eval import data_utils as D  # noqa: E402
from wm_eval import scoring as S  # noqa: E402


def make_static_predictions(holdout: Path, out_dir: Path, manifest: dict) -> Path:
    """시작 이미지를 16번 복사한 mp4를 예측으로 저장(native 해상도, 동일 인코딩)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in manifest["samples"]:
        sid = m["sid"]
        img = np.asarray(Image.open(holdout / "images" / f"{sid}.png").convert("RGB"))
        frames = np.repeat(img[None], manifest["traj_len"], axis=0)  # (16,H,W,3)
        D.save_mp4_uint8(frames, out_dir / f"{sid}.mp4", fps=m["fps"])
    return out_dir


def score_predictor(scorer: S.LocalScorer, pred_dir: Path, holdout: Path, manifest: dict) -> dict:
    """예측 영상 디렉터리를 홀드아웃 GT 대비 채점 -> 표본별/평균 성분값."""
    gt_dir = holdout / "gt_videos"
    rows = []
    for m in manifest["samples"]:
        sid = m["sid"]
        pred_v = scorer._load_video(pred_dir, sid)   # (1,16,320,512,3) uint8
        gt_v = scorer._load_video(gt_dir, sid)

        pv = scorer.video_feature(pred_v)[0]
        gv = scorer.video_feature(gt_v)[0]
        pd_ = scorer.dino_feature(pred_v)[0]
        gd = scorer.dino_feature(gt_v)[0]

        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")
        action = scorer.action_mae(pred_v, raw_actions)
        action_perdim = scorer.action_mae_perdim(pred_v, raw_actions)  # (6,)

        video = S.video_component(pv, gv)
        dino_favg = S.dino_component_frame_avg(pd_, gd)
        dino_flat = S.dino_component_flatten(pd_, gd)
        rows.append({
            "sid": sid,
            "dino_frame_avg": dino_favg,
            "dino_flatten": dino_flat,
            "video": video,
            "action": action,
            "total_frame_avg": S.weighted_total(dino_favg, video, action),
            "total_flatten": S.weighted_total(dino_flat, video, action),
            "action_perdim": [float(x) for x in action_perdim],
        })

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    action_perdim_mean = np.mean(np.array([r["action_perdim"] for r in rows]), axis=0)
    return {
        "n": len(rows),
        "mean": {k: mean(k) for k in
                 ["dino_frame_avg", "dino_flatten", "video", "action", "total_frame_avg", "total_flatten"]},
        "action_perdim_mean": [float(x) for x in action_perdim_mean],
        "rows": rows,
    }


def fmt(v):
    return f"{v:.5f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="artifacts/holdout")
    ap.add_argument("--submission-kit", default="open/submission_kit")
    ap.add_argument("--action-stats", default="open/data/train/so100_action_statistics.json")
    ap.add_argument("--out", default="artifacts/m0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pred-dir", default=None, help="추가 예측기 영상 디렉터리(예: baseline)")
    ap.add_argument("--pred-name", default="baseline")
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--skip-gt", action="store_true")
    args = ap.parse_args()

    holdout = Path(args.holdout)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))

    print(f"[m0] 채점 모델 로딩 (submission_kit={args.submission_kit}) ...")
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=args.device)

    predictors: list[tuple[str, Path]] = []
    if not args.skip_static:
        static_dir = make_static_predictions(holdout, out / "static_preds", manifest)
        predictors.append(("static", static_dir))
    if not args.skip_gt:
        predictors.append(("gt_upper_bound", holdout / "gt_videos"))
    if args.pred_dir:
        predictors.append((args.pred_name, Path(args.pred_dir)))

    results = {}
    for name, pdir in predictors:
        print(f"[m0] 채점 중: {name}  ({pdir})")
        results[name] = score_predictor(scorer, pdir, holdout, manifest)

    report = {
        "holdout": str(holdout),
        "n_samples": manifest["n_samples"],
        "seed": manifest["seed"],
        "dino_aggregation_note": "server formula UNVERIFIED; report frame-avg(primary) + flatten(compare)",
        "results": results,
    }
    (out / "m0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 표 출력 ---
    print("\n" + "=" * 78)
    print(f"M0 바닥값 (n={manifest['n_samples']}, seed={manifest['seed']})  — 0에 가까울수록 좋음")
    print("=" * 78)
    hdr = f"{'predictor':<16}{'DINO(favg)':>12}{'DINO(flat)':>12}{'Video':>10}{'Action':>10}{'TOTAL(favg)':>13}"
    print(hdr)
    print("-" * 78)
    for name, res in results.items():
        m = res["mean"]
        print(f"{name:<16}{fmt(m['dino_frame_avg']):>12}{fmt(m['dino_flatten']):>12}"
              f"{fmt(m['video']):>10}{fmt(m['action']):>10}{fmt(m['total_frame_avg']):>13}")
    print("-" * 78)
    if "static" in results and "baseline" in results:
        gap = results["static"]["mean"]["total_frame_avg"] - results["baseline"]["mean"]["total_frame_avg"]
        print(f"[격차] static - baseline (TOTAL favg) = {gap:+.5f}  "
              f"(>0이면 baseline이 static보다 우수)")

    # Action 차원별 MAE (0:shoulder_pan 1:shoulder_lift 2:elbow_flex 3:wrist_flex 4:wrist_roll 5:gripper)
    print("\n[Action 차원별 정규화 MAE]  dims=[pan, lift, elbow, wflex, wroll, grip]")
    for name, res in results.items():
        pd_ = res["action_perdim_mean"]
        print(f"  {name:<16}" + " ".join(f"{v:5.2f}" for v in pd_))
    print(f"\n[m0] 리포트 저장: {out / 'm0_report.json'}")


if __name__ == "__main__":
    main()
