"""DINO 점수를 **프레임별로 분해**해서 병목이 어디인지 가른다.

왜 필요한가
-----------
DINO 성분은 16프레임 각각의 코사인 거리를 평균한 값이다. 그래서 총점 0.221 하나만 보면
아래 두 가지가 구분되지 않는다.

  (A) **재구성 병목** : t=0(첫 프레임)부터 이미 정답과 멀다.
      → 모델이 "주어진 사진"조차 제대로 못 그리는 것이므로, 학습을 더 해도 잘 안 내려간다.
        손잡이는 VAE·샘플러·조건주입 쪽이다.
  (B) **드리프트 병목** : t=0 은 정답과 가까운데 t 가 커질수록 벌어진다.
      → 움직임 예측이 틀린 것이므로 **학습을 더 하면 내려간다**.

static(첫 프레임 16복사)은 정의상 t=0 거리가 ~0 이고 t 가 커질수록 벌어진다(=순수 드리프트).
우리 예측을 static 과 같은 축 위에 겹쳐 그리면 어느 쪽 병목인지 즉시 보인다.

사용
----
  python scripts/branchB/diag_dino_perframe.py \
      --pred name=경로 [--pred name2=경로2 ...] --out artifacts/branchB/diag_perframe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from wm_eval import scoring as S  # noqa: E402


def cos_dist_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(T,D) 두 벡터열의 행별 코사인 거리 → (T,). 채점기 _cosine_distance 와 동일 산식."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return 1.0 - num / den


def pixel_delta(video_uint8: np.ndarray) -> float:
    """프레임 사이 평균 |Δpixel|. 정답 대역은 3.9~4.0, 100 근처면 노이즈다(013 §4 함정 ③)."""
    v = video_uint8.astype(np.float32)
    return float(np.mean(np.abs(v[1:] - v[:-1])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats", default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--pred", action="append", default=[], metavar="이름=경로",
                    help="채점할 예측 디렉터리. 여러 번 줄 수 있다.")
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/diag_perframe.json"))
    ap.add_argument("--limit", type=int, default=0, help="표본 수 제한(0=전부). 빠른 확인용")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    sids = [m["sid"] for m in manifest["samples"]]
    if args.limit:
        sids = sids[: args.limit]

    entries = []
    for spec in args.pred:
        name, _, path = spec.partition("=")
        entries.append((name, Path(path)))
    if not entries:
        raise SystemExit("--pred 이름=경로 를 최소 하나 줘야 한다")

    print(f"[diag] 채점 모델 로딩 ... (n={len(sids)})", flush=True)
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=args.device)
    gt_dir = holdout / "gt_videos"

    # GT feature 를 먼저 한 번만 뽑아 재사용한다.
    gt_feat, gt_pix = {}, {}
    for i, sid in enumerate(sids):
        v = scorer._load_video(gt_dir, sid)
        gt_feat[sid] = scorer.dino_feature(v)[0]            # (16,384)
        gt_pix[sid] = pixel_delta(v.numpy()[0])
        if (i + 1) % 24 == 0:
            print(f"[diag] GT feature {i+1}/{len(sids)}", flush=True)

    out = {"n_samples": len(sids), "predictors": {},
           "gt_pixel_delta_mean": float(np.mean(list(gt_pix.values())))}

    for name, pdir in entries:
        if not pdir.exists():
            print(f"[diag] !! 건너뜀(없음): {name} {pdir}", flush=True)
            continue
        per_frame, pix = [], []
        for i, sid in enumerate(sids):
            v = scorer._load_video(pdir, sid)
            f = scorer.dino_feature(v)[0]
            per_frame.append(cos_dist_rows(f, gt_feat[sid]))
            pix.append(pixel_delta(v.numpy()[0]))
            if (i + 1) % 24 == 0:
                print(f"[diag] {name} {i+1}/{len(sids)}", flush=True)
        pf = np.stack(per_frame)                            # (N,16)
        out["predictors"][name] = {
            "dir": str(pdir),
            "dino_frame_avg": float(pf.mean()),
            "per_frame_mean": [float(x) for x in pf.mean(axis=0)],
            "per_frame_std": [float(x) for x in pf.std(axis=0)],
            "frame0": float(pf[:, 0].mean()),
            "frame15": float(pf[:, -1].mean()),
            "drift": float(pf[:, -1].mean() - pf[:, 0].mean()),
            "pixel_delta_mean": float(np.mean(pix)),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 표 출력 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"프레임별 DINO 코사인 거리 (n={len(sids)}, 낮을수록 좋음)")
    print("=" * 92)
    print(f"{'예측기':<22}{'평균':>9}{'t=0':>9}{'t=15':>9}{'드리프트':>10}{'|Δpix|':>9}")
    print("-" * 92)
    for name, r in out["predictors"].items():
        print(f"{name:<22}{r['dino_frame_avg']:>9.5f}{r['frame0']:>9.5f}{r['frame15']:>9.5f}"
              f"{r['drift']:>10.5f}{r['pixel_delta_mean']:>9.2f}")
    print("-" * 92)
    print(f"{'(정답 영상)':<22}{'':>9}{'':>9}{'':>9}{'':>10}{out['gt_pixel_delta_mean']:>9.2f}")

    print("\n[프레임별 곡선]")
    hdr = "  t      " + "".join(f"{t:>7}" for t in range(16))
    print(hdr)
    for name, r in out["predictors"].items():
        print(f"  {name:<7}" + "".join(f"{v:>7.3f}" for v in r["per_frame_mean"]))

    print(f"\n[diag] 저장 → {args.out}")


if __name__ == "__main__":
    main()
