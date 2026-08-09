#!/usr/bin/env python
"""
제출 경로에서 잔차가 얼마나 살아남나 — 코사인을 단계별로 다시 잰다

왜 필요한가
  로컬 감시는 잔차 코사인 0.515 를 가리키는데 리더보드는 λ>0 이 전부 손해라고 답했다
  (λ=0 0.3032 · 0.25 0.3192 · 0.5 0.3762). 이 격차의 원인이 둘 중 무엇인지 안 갈랐다:
    ① Goodhart — 코사인이 채점 지표와 애초에 안 이어진다
    ② 경로 손실 — 감시가 재는 공간과 채점되는 공간이 달라 신호가 도중에 사라진다
  ②는 실재 가능성이 있다. 009 §2 가 남긴 사실:
    모델은 320×512 에서 돌고(좌우 42/43열 검은 띠), 제출은 640×480 원본이다.
    감시는 **모델 공간**에서 재고 채점은 **native mp4** 를 본다. 그 사이 왕복은 미측정이다.
    λ=0 md5 검증은 첫 프레임 경로만 증명했지 **잔차 경로는 증명하지 않았다.**

무엇을 재나 — 같은 표본에서 세 지점의 코사인
  ① 모델 공간 320×512          ← 감시가 재는 것
  ② native 640×480 (mp4 전)    ← residual_to_native 왕복 후
  ③ native 640×480 (mp4 후)    ← 실제로 채점되는 것
  ①→② 에서 떨어지면 해상도 왕복이 범인, ②→③ 이면 x264 가 범인,
  셋 다 비슷하면 경로는 멀쩡하고 원인은 ① Goodhart 다.

판정 (사전 등록)
  ③ ≥ ① × 0.8  →  경로는 멀쩡하다. 원인은 Goodhart
  ③ <  ① × 0.8  →  **경로에서 신호가 샌다.** 고치면 되찾을 몫이 있다

🚨 킷 미사용(순수 torch). GPU 0 만.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
sys.path.insert(0, str(REPO / "src"))
from model_c import ResidualSimVPC                                    # noqa: E402
from loader_c import load_action_stats                                # noqa: E402
from infer_c import to_model_space, residual_to_native                # noqa: E402
from wm_eval import data_utils as D                                   # noqa: E402
import av                                                             # noqa: E402

HOLDOUT = REPO / "artifacts" / "holdout_val96"
FPS = 6
CUT = 0.8


def read_mp4(p: Path) -> np.ndarray:
    c = av.open(str(p))
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return np.stack(fr)


def cos_np(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    dev = "cuda"
    tmp = Path("/tmp/probe_native_cos"); tmp.mkdir(exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ResidualSimVPC(hid_S=ck["args"].get("hid_s", 64), use_ckpt=False).to(dev).eval()
    model.load_state_dict(ck["model"])
    mu, sd = load_action_stats(); mu, sd = mu.to(dev), sd.to(dev)
    print(f"[ckpt] {Path(a.ckpt).name}  step {ck['step']}  λ={a.lam}  n={a.n}")

    man = json.load(open(HOLDOUT / "manifest.json"))
    samples = [s for s in man["samples"] if tuple(s["native_hw"]) == (480, 640)][:a.n]

    c1s, c2s, c3s = [], [], []
    for s in samples:
        sid = s["sid"]
        img = np.asarray(Image.open(HOLDOUT / "images" / f"{sid}.png").convert("RGB"))
        gt = read_mp4(HOLDOUT / "gt_videos" / f"{sid}.mp4")            # (16,480,640,3)
        act = np.load(HOLDOUT / "actions" / f"{sid}.npy").astype(np.float32)

        with torch.no_grad():
            fm = to_model_space(img).to(dev)                            # (1,3,320,512)
            aa = ((torch.from_numpy(act).to(dev) - mu) / sd)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_m, res_m = model(fm, aa, return_residual=True)
            res_m = res_m.float()
            res_n = residual_to_native(res_m)                           # (1,3,T,480,640)

        # ① 모델 공간 — 감시와 같은 자
        gt_m = torch.stack([to_model_space(g)[0] for g in gt]).permute(1, 0, 2, 3)[None].to(dev)
        rg_m = (gt_m - fm.unsqueeze(2))[:, :, 1:]
        c1s.append(F.cosine_similarity(res_m[:, :, 1:].flatten(1), rg_m.flatten(1), dim=1).item())

        # ② native (mp4 전)
        base = (torch.from_numpy(img).float().permute(2, 0, 1)[None, :, None].to(dev) / 255. - .5) * 2
        gt_t = (torch.from_numpy(gt).float().permute(3, 0, 1, 2)[None].to(dev) / 255. - .5) * 2
        rg_n = (gt_t - base)[:, :, 1:]
        c2s.append(F.cosine_similarity(res_n[:, :, 1:].flatten(1), rg_n.flatten(1), dim=1).item())

        # ③ native (mp4 왕복 후) — 실제 채점되는 것
        pred = (base + a.lam * res_n).clamp(-1, 1)
        u8 = ((pred[0].permute(1, 2, 3, 0) + 1) / 2 * 255).round().byte().cpu().numpy()
        p = tmp / f"{sid}.mp4"
        D.save_mp4_uint8(u8, p, fps=FPS)
        dec = read_mp4(p).astype(np.float32)
        rp3 = (dec - img[None].astype(np.float32))[1:] / a.lam         # λ 를 되나눈다
        rg3 = (gt.astype(np.float32) - img[None].astype(np.float32))[1:]
        c3s.append(cos_np(rp3, rg3))

    m1, m2, m3 = np.mean(c1s), np.mean(c2s), np.mean(c3s)
    print(f"\n  ① 모델 공간 320×512        {m1:+.4f}   ← 감시가 재는 것")
    print(f"  ② native 640×480 (mp4 전)  {m2:+.4f}   ← 해상도 왕복 후")
    print(f"  ③ native 640×480 (mp4 후)  {m3:+.4f}   ← 실제 채점되는 것")
    print(f"\n  ①→② 보존율 {m2/m1:.3f}    ②→③ 보존율 {m3/m2:.3f}    전체 {m3/m1:.3f}")
    ok = m3 >= m1 * CUT
    print(f"  ⇒ {'✅ 경로는 멀쩡하다 — 원인은 Goodhart' if ok else '❌ 경로에서 신호가 샌다'}"
          f"  (판정선: ③ ≥ ① × {CUT})")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = REPO / "results" / "branchC" / f"native_cosine_{ts}.json"
    json.dump({"ckpt": a.ckpt, "step": ck["step"], "lam": a.lam, "n": len(samples),
               "cos_model": m1, "cos_native": m2, "cos_native_mp4": m3,
               "path_ok": bool(ok)}, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[out] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
