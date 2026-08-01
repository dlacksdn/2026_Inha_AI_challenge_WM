"""누수 없는 홀드아웃을 만든다 — 학습에 쓰이지 않은 에피소드만으로.

왜 필요한가
-----------
기존 홀드아웃(`artifacts/holdout`, 96표본)은 **학습 데이터와 겹친다.** 018 §6.3 에서
코드로 확정했다.

    학습 데이터모듈의 분할 (ldwma/datasets/lerobot_so100.py:270-285)
        examples = 모든 데이터셋의 모든 에피소드 (길이 ≥ traj_len × downsample)
        random.Random(seed).shuffle(examples)          # seed 기본 0
        val_count   = int(len(examples) * 0.05)
        train split = examples[val_count:]             # 10,544개
        val   split = examples[:val_count]             #    554개

    우리 홀드아웃 생성기 (wm_eval/data_utils.iter_holdout_samples)
        list_episodes(...)  ← **전체 에피소드. 배제 없음**

⇒ 기존 홀드아웃 96개 중 약 95%가 학습에 쓰인 에피소드다.

**지금까지는 무해했다.** 결론이 전부 "학습 데이터에서조차 진다"는 보수적 방향이었기
때문이다. 그러나 **C 는 이기려고 만드는 것이다.** 이기는 순간 암기와 일반화를 구분할 수 없다.

무엇을 하는가
-------------
**분할 규칙을 재구현하지 않는다.** 재구현하면 셔플 순서·데이터셋 발견 순서에서
조용히 어긋난다. 대신 **학습이 쓰는 바로 그 클래스로 val 데이터셋을 실제로 만들어**
그 에피소드 목록을 그대로 쓴다.

    ds = LeRobotSO100Dataset(train=False, val_fraction=0.05, seed=0, ...)
    허용 = { str(e["video_ref"]) for e in ds.examples }
    → 이 집합에 속한 에피소드만으로 홀드아웃을 만든다

기존 홀드아웃은 **건드리지 않는다.** 016 §9.2 와 018 의 모든 수치가 그 위에 있어서
바꾸면 비교 가능성을 통째로 잃는다. 둘 다 남기고 둘 다 채점한다.

    기존 홀드아웃에서만 이긴다  →  암기를 의심하라
    새 홀드아웃에서만 이긴다    →  기존 쪽 표본이 어려운 것이다 (또는 우연)
    둘 다에서 이긴다            →  믿을 만하다

⚠ 반드시 학습과 같은 값을 써야 한다
-----------------------------------
`val_fraction`, `seed`, `traj_len`, `downsample`, `camera_key` 가 학습 config
(`scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml`)와 하나라도 다르면
분할이 달라져 **누수가 그대로 남는다.** 기본값은 그 config 에 맞춰 뒀다.
C 의 학습 config 가 이 값들을 바꾸면 **여기도 같이 바꿔야 한다.**

사용:
  python scripts/branchC/build_holdout_val.py --n 96 --out artifacts/holdout_val96
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
CK = REPO / "open/baseline/challenge_kit"
sys.path.insert(0, str(CK / "src"))
sys.path.insert(0, str(CK / "libs/dynamicrafter"))
from wm_eval import data_utils as D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default=str(REPO / "open/data/train"))
    ap.add_argument("--out", default=str(REPO / "artifacts/holdout_val96"))
    ap.add_argument("--n", type=int, default=96, help="기존 홀드아웃과 같은 96개")
    ap.add_argument("--seed", type=int, default=0, help="표본 선택 seed (홀드아웃 내부용)")
    ap.add_argument("--per-dataset-cap", type=int, default=2)
    # --- 아래 넷은 학습 config 와 반드시 일치해야 한다 ---
    ap.add_argument("--traj-len", type=int, default=16)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0, help="학습 데이터모듈의 seed")
    ap.add_argument("--camera-key", default="auto")
    args = ap.parse_args()

    from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset  # noqa: E402

    print("[holdout-val] 학습이 쓰는 클래스로 val split 을 실제로 만든다 "
          "(분할 규칙을 재구현하지 않는다)", flush=True)
    val_ds = LeRobotSO100Dataset(
        root=args.train_root, dataset_paths="auto", train=False,
        traj_len=args.traj_len, target_height=320, target_width=512, pad=True,
        camera_key=args.camera_key, val_fraction=args.val_fraction,
        seed=args.split_seed, downsample=args.downsample, use_language=False,
        remote=False,
    )
    allow = {str(e["video_ref"]) for e in val_ds.examples}
    print(f"[holdout-val] val split 에피소드 {len(val_ds.examples)}개 "
          f"(고유 영상 {len(allow)}개)", flush=True)
    if len(allow) < args.n:
        raise SystemExit(
            f"ERROR: val split 이 {len(allow)}개뿐인데 {args.n}개를 요구했다. "
            "--n 을 줄이거나 학습의 val_fraction 을 키워야 한다(그러면 학습을 다시 해야 한다).")

    out = Path(args.out)
    for sub in ("images", "actions", "gt_videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest = []
    for s in D.iter_holdout_samples(
        Path(args.train_root), n_samples=args.n, seed=args.seed,
        traj_len=args.traj_len, per_dataset_cap=args.per_dataset_cap,
        episode_filter=lambda ref: str(ref.video) in allow,
    ):
        ref, sid = s["ref"], s["sid"]
        frames = D.decode_frames(ref.video, s["frame_indices"])
        actions = D.read_actions(ref.parquet, s["frame_indices"])
        D.save_png(frames[0], out / "images" / f"{sid}.png")
        np.save(out / "actions" / f"{sid}.npy", actions)
        D.save_mp4_uint8(frames, out / "gt_videos" / f"{sid}.mp4", fps=ref.fps)
        manifest.append({
            "sid": sid, "dataset": s["dataset"], "episode_index": ref.episode_index,
            "start_idx": s["start_idx"], "length": ref.length, "fps": ref.fps,
            "native_hw": [int(frames.shape[1]), int(frames.shape[2])],
            "camera_key": ref.camera_key, "video_ref": str(ref.video),
        })
        if len(manifest) % 8 == 0:
            print(f"[holdout-val] {len(manifest)}/{args.n}", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "n_samples": len(manifest), "seed": args.seed, "traj_len": args.traj_len,
        "per_dataset_cap": args.per_dataset_cap, "train_root": args.train_root,
        "leak_free": True,
        "split": {"val_fraction": args.val_fraction, "split_seed": args.split_seed,
                  "downsample": args.downsample, "camera_key": args.camera_key,
                  "n_val_episodes": len(val_ds.examples)},
        "note": "학습 split 을 배제한 홀드아웃. 학습 config 의 val_fraction/seed/traj_len/"
                "downsample/camera_key 가 바뀌면 이것도 다시 만들어야 한다.",
        "samples": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 새너티 — 뽑은 표본이 정말 전부 val split 안에 있는가
    bad = [m["sid"] for m in manifest if m["video_ref"] not in allow]
    print(f"\n[새너티] val split 밖의 표본 {len(bad)}개 "
          f"({'통과' if not bad else '⚠ 실패: ' + str(bad[:5])})")
    ds_counts: dict[str, int] = {}
    for m in manifest:
        ds_counts[m["dataset"]] = ds_counts.get(m["dataset"], 0) + 1
    print(f"[holdout-val] 완료: {len(manifest)}개 표본 -> {out}")
    print(f"[holdout-val] 데이터셋 {len(ds_counts)}개, 해상도 "
          f"{sorted({tuple(m['native_hw']) for m in manifest})}")


if __name__ == "__main__":
    main()
