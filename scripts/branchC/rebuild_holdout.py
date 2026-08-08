#!/usr/bin/env python
"""
holdout_val96 재생성 — manifest.json 하나만 있으면 train 데이터에서 그대로 복원한다

왜 필요한가
  `artifacts/` 는 gitignore 라 clone 에 홀드아웃(34MB)이 안 딸려 온다.
  그런데 학습은 시작조차 못 한다 — 누수 차단(holdout_episode_refs)과 감시(load_holdout_val96)가
  이걸 읽기 때문이다.
  기존 생성기 `build_holdout_val.py` 는 `ldwma`(레포 밖, 미설치)에 의존해 다른 기계에서 못 돈다.

이 스크립트는 그 의존성을 없앤다.
  manifest.json(43KB, git 추적)에 `video_ref · start_idx · traj_len` 이 전부 있으므로
  train 데이터만 있으면 images / gt_videos / actions 를 그대로 다시 만든다.
  ⇒ **분할 규칙을 재구현하지 않는다.** 이미 확정된 96개를 그대로 복원할 뿐이다
     (019 §3 이 경고한 "로더 5설정 바꾸면 재생성" 문제를 피한다).

사용
  python scripts/branchC/rebuild_holdout.py            # 없는 것만 만든다
  python scripts/branchC/rebuild_holdout.py --verify   # 기존 것과 프레임이 같은지 검사
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from wm_eval import data_utils as D  # noqa: E402

HOLDOUT = REPO / "artifacts" / "holdout_val96"
TRAIN = REPO / "open" / "data" / "train"
FPS_DEFAULT = 6


def rel_video(sample, manifest_train_root: str) -> Path:
    """manifest 의 절대 video_ref 를 현재 기계의 train 경로로 옮긴다."""
    v = str(sample["video_ref"])
    root = manifest_train_root.rstrip("/")
    if v.startswith(root):
        return TRAIN / v[len(root):].lstrip("/")
    # 폴백: dataset/videos/... 조각을 찾는다
    parts = Path(v).parts
    if sample["dataset"] in parts:
        i = parts.index(sample["dataset"])
        return TRAIN.joinpath(*parts[i:])
    raise FileNotFoundError(f"video_ref 를 옮길 수 없다: {v}")


def parquet_for(video: Path, episode_index: int) -> Path:
    """videos/chunk-XXX/<camera>/episode_NNNNNN.mp4 → data/chunk-XXX/episode_NNNNNN.parquet"""
    ds = video.parents[3]                       # 데이터셋 루트
    chunk = video.parents[1].name               # chunk-000
    return ds / "data" / chunk / f"episode_{episode_index:06d}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="기존 산출물과 프레임 일치 검사")
    ap.add_argument("--force", action="store_true", help="있어도 다시 만든다")
    args = ap.parse_args()

    man_path = HOLDOUT / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"manifest 가 없다: {man_path}\n"
                         "  git 에 추적되는 파일이다. clone 이 제대로 됐는지 확인하라")
    man = json.load(open(man_path))
    root = man["train_root"]
    samples = man["samples"]
    tl = int(man.get("traj_len", 16))
    print(f"[manifest] {len(samples)} 표본 · traj_len {tl}")
    print(f"[train] {TRAIN}  (manifest 기록: {root})")

    for d in ("images", "gt_videos", "actions"):
        (HOLDOUT / d).mkdir(parents=True, exist_ok=True)

    made = skipped = mismatch = 0
    for i, s in enumerate(samples, 1):
        sid, st = s["sid"], int(s["start_idx"])
        png = HOLDOUT / "images" / f"{sid}.png"
        mp4 = HOLDOUT / "gt_videos" / f"{sid}.mp4"
        npy = HOLDOUT / "actions" / f"{sid}.npy"
        if not args.force and not args.verify and png.exists() and mp4.exists() and npy.exists():
            skipped += 1
            continue

        vid = rel_video(s, root)
        if not vid.exists():
            raise SystemExit(f"영상이 없다: {vid}\n  open/data/train 링크를 확인하라")
        idx = list(range(st, st + tl))
        frames = D.decode_frames(vid, idx)                       # (16,H,W,3) uint8
        acts = np.stack(pd.read_parquet(parquet_for(vid, int(s["episode_index"])),
                                        columns=["action"])["action"].iloc[idx].to_numpy()
                        ).astype(np.float32)

        if args.verify and png.exists():
            old = np.asarray(Image.open(png).convert("RGB"))
            d = np.abs(old.astype(int) - frames[0].astype(int)).max()
            if d != 0:
                mismatch += 1
                print(f"  ⚠ {sid} 첫 프레임 불일치 최대 {d}")
            if npy.exists():
                da = np.abs(np.load(npy) - acts).max()
                if da > 1e-5:
                    mismatch += 1
                    print(f"  ⚠ {sid} 행동 불일치 최대 {da:.3e}")
            continue

        Image.fromarray(frames[0]).save(png)
        D.save_mp4_uint8(frames, mp4, fps=int(s.get("fps", FPS_DEFAULT)))
        np.save(npy, acts)
        made += 1
        if i % 20 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}", flush=True)

    if args.verify:
        print(f"\n[검증] 불일치 {mismatch} 건  ← 0 이어야 한다")
        raise SystemExit(1 if mismatch else 0)
    print(f"\n[완료] 생성 {made} · 건너뜀 {skipped}  → {HOLDOUT}")
    for d in ("images", "gt_videos", "actions"):
        print(f"  {d}: {len(list((HOLDOUT/d).iterdir()))} 개")


if __name__ == "__main__":
    main()
