"""eval 216개에 대한 **정지영상(static) 예측**을 만든다 — 리더보드 좌표의 대조군.

왜 필요한가
-----------
우리는 지금까지 로컬 홀드아웃(96개)에서만 "static 을 이겼나"를 따졌다. 그런데 유일한
리더보드 측정(011 §1)이 보여준 바로는 **로컬과 eval 은 채점 구조가 반전된다** —
eval 에서 static 의 Action 은 0.4287 로 로컬(1.2402)의 0.35배밖에 안 되고,
DINO+Video 는 반대로 2.05배 어려워진다. 즉 로컬의 승패가 eval 의 승패가 아니다.

여기서 결정적인 사실 하나: 채점의 **Action 성분(배점 40%)은 정답 영상 없이 계산된다.**
주어진 행동 시퀀스와, 생성 영상에서 역추정한 행동의 MAE 이기 때문이다. 그래서
`make_submission_csv.py` 를 돌리면 eval 216개의 Action 값이 표본별로 CSV 에 그대로 실린다.

⇒ **제출권을 쓰지 않고도 배점 40% 축에서 우리 모델과 static 을 표본별로 겨룰 수 있다.**
   (나머지 60%인 DINO·Video 는 정답 영상이 필요하므로 서버만 안다. 그건 제출로 확인한다.)

이 스크립트는 그 대조군을 만든다. 정지영상 = 시작 이미지를 16번 복사한 영상이다.

규칙 준수
---------
- eval 데이터는 **추론 입력으로만** 쓴다. 학습·데이터선별에 쓰지 않는다(대회 규칙 §4.1-3).
- 인코딩은 채점기와 같은 libx264 설정(`data_utils.save_mp4_uint8`)을 쓴다. 인코딩이 다르면
  압축 손실이 달라져 비교가 오염된다.
- mp4 의 fps 값은 채점에 영향이 없다(채점기는 프레임만 디코딩한다). 데이터셋 표준인 6 을 쓴다.

사용:
  python scripts/branchB/make_static_eval_preds.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
from wm_eval import data_utils as D  # noqa: E402

TRAJ_LEN = 16
FPS = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(REPO / "open/data/eval"))
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/preds_eval216_static"))
    # 홀드아웃에도 쓸 수 있게 개수를 인자로 뺀다(기본 216 = eval, 기존 동작 그대로).
    # 개수 검사를 없애지 않는 이유: 016 함정 — 216개 중 24개만 읽고 가짜 판정이 난 적이 있다.
    ap.add_argument("--expect", type=int, default=216, help="완성 개수 검사값")
    args = ap.parse_args()

    img_dir = Path(args.eval_root) / "images"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(img_dir.glob("sample_*.png"))
    if not pngs:
        raise SystemExit(f"eval 이미지를 못 찾았다: {img_dir}")
    print(f"[static-eval] 입력 {len(pngs)}개  →  {out_dir}")

    for i, p in enumerate(pngs, 1):
        img = np.asarray(Image.open(p).convert("RGB"))
        frames = np.repeat(img[None], TRAJ_LEN, axis=0)  # (16, H, W, 3)
        D.save_mp4_uint8(frames, out_dir / f"{p.stem}.mp4", fps=FPS)
        if i % 40 == 0 or i == len(pngs):
            print(f"[static-eval] {i}/{len(pngs)}", flush=True)

    n = len(list(out_dir.glob("*.mp4")))
    print(f"[static-eval] 완료 — mp4 {n}개")
    if n != args.expect:
        raise SystemExit(f"ERROR: {args.expect}개여야 한다 (현재 {n})")


if __name__ == "__main__":
    main()
