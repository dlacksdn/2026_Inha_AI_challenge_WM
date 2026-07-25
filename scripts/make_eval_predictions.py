"""eval 216개에 대한 제출용 예측 영상 생성 (무학습 예측기).

M0/모션스윕 결과: 현재까지 가장 좋은 예측기는 **static**(시작 프레임 16복사)이다.
주최 baseline(11M)은 그보다 나쁘므로, 제출 floor로 static을 준비해 둔다.

규칙 확인:
  - eval 데이터를 **추론 입력**으로 쓰는 것은 정상(그게 과제다). 금지된 것은 eval을 **학습**에 쓰는 것.
  - 생성 방식에는 제약이 없다. 금지된 것은 submission_kit 수정과 CSV 후처리.
  - 프레임 수는 정확히 16이어야 한다.

사용:
  python scripts/make_eval_predictions.py --mode static --out artifacts/submission/static
  # 이후 CSV: submission_kit/make_submission_csv.py 로 변환(scripts/make_submission.sh 참고)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wm_eval import data_utils as D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="open/data/eval")
    ap.add_argument("--out", default="artifacts/submission/static")
    ap.add_argument("--mode", default="static", choices=["static"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=6, help="채점은 프레임만 보므로 값 자체는 무관")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_dir = eval_root / "images"
    act_dir = eval_root / "actions"
    sids = sorted(p.stem for p in img_dir.glob("*.png"))
    act_sids = {p.stem for p in act_dir.glob("*.npy")}
    missing_act = [s for s in sids if s not in act_sids]
    if missing_act:
        raise SystemExit(f"액션 없는 샘플 {len(missing_act)}개: {missing_act[:5]}")

    print(f"[eval-pred] mode={args.mode} n={len(sids)} → {out}")
    for i, sid in enumerate(sids):
        img = np.asarray(Image.open(img_dir / f"{sid}.png").convert("RGB"))
        frames = np.repeat(img[None], args.frames, axis=0)  # (16,H,W,3)
        D.save_mp4_uint8(frames, out / f"{sid}.mp4", fps=args.fps)
        if (i + 1) % 48 == 0:
            print(f"[eval-pred] {i+1}/{len(sids)}")

    made = sorted(p.stem for p in out.glob("*.mp4"))
    assert made == sids, f"생성 누락: {len(made)}/{len(sids)}"
    print(f"[eval-pred] 완료: {len(made)}개 (sample_000000~{made[-1].split('_')[-1]})")


if __name__ == "__main__":
    main()
