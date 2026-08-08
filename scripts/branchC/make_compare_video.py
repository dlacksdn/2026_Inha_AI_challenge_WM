#!/usr/bin/env python
"""
홀드아웃 비교 영상 — 체크포인트를 읽어 [정답 | 예측] mp4 를 만든다

왜 필요한가
  train_c.py 의 그림은 정지 이미지(프레임 0/5/10/15)뿐이다. 움직임은 영상으로 봐야 보인다.
  그리고 002 §1.2 가 인용한 실패 사례가 "차량도 차선도 알아볼 수 없는 균일한 회색 블러"다 —
  ρ 나 코사인이 통과해도 그림이 죽어 있으면 소용없다.

⭐ **제출과 똑같은 경로로 만든다.**
  infer_c.py 의 to_model_space → 모델 → residual_to_native → 원본 + λ×잔차 → mp4.
  홀드아웃은 정답이 있으니, 이 영상이 곧 "제출하면 무엇이 나가는가"의 미리보기다.
  (eval 은 정답이 없어 이 비교를 못 한다)

돌고 있는 학습을 건드리지 않는다 — 체크포인트만 읽는다.

사용
  python make_compare_video.py --ckpt artifacts/branchC/train_.../ck_000500.pt --n 6
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]   # 상대경로 (대회 §3.3 요건)
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
sys.path.insert(0, str(REPO / "src"))
from model_c import ResidualSimVPC                                    # noqa: E402
from loader_c import load_action_stats, WINDOW                        # noqa: E402
from infer_c import to_model_space, residual_to_native                # noqa: E402
from wm_eval import data_utils as D                                   # noqa: E402

HOLDOUT = REPO / "artifacts" / "holdout_val96"
FPS = 6
BAR = 6            # 패널 사이 흰 띠 두께


def read_mp4(p):
    with av.open(str(p)) as c:
        return np.stack([f.to_ndarray(format="rgb24")
                         for f in c.decode(c.streams.video[0])])[:WINDOW]


def heat(x):
    """(T,H,W) float ≥0 → (T,H,W,3) uint8. inferno 유사 매핑(의존성 없이)."""
    v = np.clip(x / (x.max() + 1e-8), 0, 1)
    r = np.clip(1.6 * v, 0, 1); g = np.clip(1.6 * v - 0.5, 0, 1); b = np.clip(2.5 * v - 1.5, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto",
                    help="auto=GPU 시도 후 OOM 이면 CPU. 학습 중에는 CPU 가 안전하다")
    args = ap.parse_args()

    ck_path = Path(args.ckpt)
    out = Path(args.out) if args.out else ck_path.parent / "viz"
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ck_path, map_location="cpu")
    hid_s = ck.get("args", {}).get("hid_s", 64)
    step = ck.get("step", 0)
    model = ResidualSimVPC(hid_S=hid_s, use_ckpt=False).eval()
    model.load_state_dict(ck["model"])
    dev = args.device
    if dev == "auto":
        dev = "cuda"
        try:
            model = model.cuda()
        except RuntimeError as e:      # 학습이 GPU 를 쓰고 있으면 여기로 온다
            print(f"[device] GPU 사용 불가({type(e).__name__}) → CPU 로 돌린다")
            model = model.cpu(); dev = "cpu"; torch.cuda.empty_cache()
    else:
        model = model.to(dev)
    mu, sd = load_action_stats(); mu, sd = mu.to(dev), sd.to(dev)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" \
        else torch.autocast("cpu", enabled=False)
    print(f"[ckpt] {ck_path.name}  step {step}  hid_S={hid_s}  λ={args.lam}  device={dev}")

    man = json.load(open(HOLDOUT / "manifest.json"))
    # 4:3 표본만 (학습 분포와 같다). 007 확인: 96개 중 4개가 16:9
    samples = [s for s in man["samples"] if tuple(s["native_hw"]) == (480, 640)][:args.n]

    made = []
    for s in samples:
        sid = s["sid"]
        img = np.asarray(Image.open(HOLDOUT / "images" / f"{sid}.png").convert("RGB"))
        gt = read_mp4(HOLDOUT / "gt_videos" / f"{sid}.mp4")           # (16,480,640,3)
        act = np.load(HOLDOUT / "actions" / f"{sid}.npy").astype(np.float32)

        with torch.no_grad():
            fm = to_model_space(img).to(dev)
            a = ((torch.from_numpy(act).to(dev) - mu) / sd)[None]
            with amp:
                _, res = model(fm, a, return_residual=True)
            res_n = residual_to_native(res.float())                   # (1,3,T,480,640)
        base = (torch.from_numpy(img).float().permute(2, 0, 1)[None, :, None].to(dev) / 255.0 - .5) * 2
        pred = (base + args.lam * res_n).clamp(-1, 1)
        pr = ((pred[0].permute(1, 2, 3, 0) + 1) / 2 * 255).round().byte().cpu().numpy()

        H, W = gt.shape[1:3]
        white = np.full((WINDOW, H, BAR, 3), 255, np.uint8)
        # ① [정답 | 예측]
        D.save_mp4_uint8(np.concatenate([gt, white, pr], axis=2),
                         out / f"cmp_step{step:06d}_lam{args.lam:g}_{sid}.mp4", fps=FPS)
        # ② [정답 잔차 | 예측 잔차]  — 같은 자로 그린다
        rg = np.abs(gt.astype(np.float32) - img[None]).mean(-1)
        rp = np.abs(pr.astype(np.float32) - img[None]).mean(-1)
        vm = max(rg.max(), 1e-6)
        D.save_mp4_uint8(np.concatenate([heat(rg / vm), white, heat(rp / vm)], axis=2),
                         out / f"res_step{step:06d}_lam{args.lam:g}_{sid}.mp4", fps=FPS)
        made.append(sid)
        print(f"  {sid}  |잔차| 정답 {rg.mean():.2f} vs 예측 {rp.mean():.2f}  (0~255)")

    print(f"\n[완료] {len(made)*2}개 mp4 → {out}")
    print("  cmp_*  왼쪽 정답 · 오른쪽 예측")
    print("  res_*  왼쪽 정답잔차 · 오른쪽 예측잔차 (같은 밝기 자)")


if __name__ == "__main__":
    main()
