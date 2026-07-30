"""S4 스모크용 N표본 홀드아웃 서브셋을 만든다 (기본 4개).

왜 필요한가: `artifacts/` 는 .gitignore 대상이라 새 머신(5090/6000)에는 홀드아웃이 없다.
    새 머신에서는 먼저 홀드아웃을 재생성한 뒤 이 스크립트로 스모크용 서브셋을 잘라낸다.

    conda run -n wm python scripts/build_holdout.py --train-root open/data/train \
        --out artifacts/holdout --n 96 --seed 0 --per-dataset-cap 2
    conda run -n wm python scripts/branchB/make_smoke_holdout.py

서브셋은 **앞에서부터 N개**를 결정론적으로 고른다(같은 seed 홀드아웃이면 어느 머신에서나 동일).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_paths import repo_root  # noqa: E402

SUBDIRS = ("images", "actions", "gt_videos")


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(root / "artifacts/holdout"))
    ap.add_argument("--dst", default=str(root / "artifacts/holdout_smoke4"))
    ap.add_argument("-n", type=int, default=4)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    keep = [s["sid"] for s in man["samples"][: args.n]]

    if dst.exists():
        shutil.rmtree(dst)
    counts = {}
    for sub in SUBDIRS:
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for sid in keep:
            for f in (src / sub).glob(f"{sid}.*"):
                shutil.copy2(f, dst / sub / f.name)
        counts[sub] = len(list((dst / sub).iterdir()))

    m2 = dict(man)
    m2["samples"] = [s for s in man["samples"] if s["sid"] in keep]
    m2["n_samples"] = len(keep)
    m2["note"] = f"S4 스모크용 {args.n}표본 서브셋 (출처: {src.name} 의 앞 {args.n}개)"
    (dst / "manifest.json").write_text(json.dumps(m2, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[smoke-holdout] {dst} 생성: {keep}")
    print(f"[smoke-holdout] 파일 수: {counts}")
    missing = [k for k, v in counts.items() if v != len(keep)]
    if missing:
        raise SystemExit(f"[smoke-holdout] 경고: {missing} 개수 불일치 — 원본 홀드아웃을 확인하라")


if __name__ == "__main__":
    main()
