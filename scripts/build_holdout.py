"""train 데이터에서 월드모델 평가용 홀드아웃을 만든다 (M0).

산출물(<out>/):
  images/{sid}.png     시작 이미지(에피소드 start_idx 프레임, native 해상도)
  actions/{sid}.npy    (16,6) float32 raw 액션  <- eval/actions 포맷과 동일
  gt_videos/{sid}.mp4  정답 16프레임(libx264, 채점 파이프라인과 동일 인코딩)
  manifest.json        표본 메타(dataset/episode/start_idx) — 완전 재현용

사용:
  python scripts/build_holdout.py --train-root open/data/train --out artifacts/holdout \
      --n 64 --seed 0

주의: eval 데이터는 절대 건드리지 않는다. data/train 만 사용.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wm_eval import data_utils as D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default="open/data/train")
    ap.add_argument("--out", default="artifacts/holdout")
    ap.add_argument("--n", type=int, default=64, help="홀드아웃 표본 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traj-len", type=int, default=16)
    ap.add_argument("--per-dataset-cap", type=int, default=2, help="데이터셋당 최대 표본(다양성)")
    args = ap.parse_args()

    train_root = Path(args.train_root)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "actions").mkdir(parents=True, exist_ok=True)
    (out / "gt_videos").mkdir(parents=True, exist_ok=True)

    manifest = []
    for s in D.iter_holdout_samples(
        train_root, n_samples=args.n, seed=args.seed,
        traj_len=args.traj_len, per_dataset_cap=args.per_dataset_cap,
    ):
        ref = s["ref"]
        sid = s["sid"]
        frames = D.decode_frames(ref.video, s["frame_indices"])       # (16,H,W,3) uint8
        actions = D.read_actions(ref.parquet, s["frame_indices"])     # (16,6) raw

        D.save_png(frames[0], out / "images" / f"{sid}.png")
        np.save(out / "actions" / f"{sid}.npy", actions)
        D.save_mp4_uint8(frames, out / "gt_videos" / f"{sid}.mp4", fps=ref.fps)

        manifest.append({
            "sid": sid,
            "dataset": s["dataset"],
            "episode_index": s["episode_index"],
            "start_idx": s["start_idx"],
            "length": ref.length,
            "fps": ref.fps,
            "native_hw": [int(frames.shape[1]), int(frames.shape[2])],
            "camera_key": ref.camera_key,
        })
        if len(manifest) % 8 == 0:
            print(f"[holdout] {len(manifest)}/{args.n}")

    meta = {
        "n_samples": len(manifest),
        "seed": args.seed,
        "traj_len": args.traj_len,
        "per_dataset_cap": args.per_dataset_cap,
        "train_root": str(train_root),
        "samples": manifest,
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[holdout] 완료: {len(manifest)}개 표본 -> {out}")
    # 데이터셋 다양성 요약
    ds_counts: dict[str, int] = {}
    for m in manifest:
        ds_counts[m["dataset"]] = ds_counts.get(m["dataset"], 0) + 1
    print(f"[holdout] 사용 데이터셋 {len(ds_counts)}개, 해상도 분포: "
          f"{sorted({tuple(m['native_hw']) for m in manifest})}")


if __name__ == "__main__":
    main()
