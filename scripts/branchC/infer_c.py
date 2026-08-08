#!/usr/bin/env python
"""
branch C 추론 — 체크포인트 → eval 216 mp4 (+ 008 §9-b λ 스윕)

⭐ 해상도 규약 — 여기가 함정이다 [코드]
  0.30325 를 낸 static 제출 영상은 **640×480 원본 해상도**다
  (`scripts/branchB/make_static_eval_preds.py:65-67` — eval png 를 그대로 16복제).
  그런데 모델은 320×512(좌우 42/43열 패딩)에서 돈다. **되돌려야 한다.**

  ⇒ 잔차 구조의 이점이 여기서 나온다:
      확산모델은 전체 프레임이 왕복 변환(320×512 → 640×480)을 겪는다
      잔차는 **첫 프레임을 원본 그대로 쓰고 잔차만 되돌린다**
      ⇒ 왕복 손실이 잔차에만 붙고, λ=0 이면 출력이 static 과 **완전히 동일**하다

  역변환:  320×512 → 좌우 패딩 제거(320×427) → 480×640 bilinear

λ 스윕 (008 §9-b)
  출력 = 첫 프레임 + λ × 잔차.   λ 는 추론 때 곱하는 숫자 하나 — 재학습이 필요 없다
  λ=0 → 정확히 static(=S₀).   λ 를 올려 좋아지면 A-2, 나빠지면 A-1
  ⚠ λ 는 **우리 모델 출력에 곱하는 상수**다. 킷과 무관하다 (rule §13 위반 아님)

사용
  python infer_c.py --ckpt <ck.pt> --lam 0 0.25 0.5 0.75 1.0
  → artifacts/branchC/infer_<ts>/lam<λ>/sample_XXXXXX.mp4  (216개씩)
  그 뒤 사용자가 submission_kit 으로 CSV 를 만든다 (**최종 mp4 확정 후 1회**, rule §12.5)
"""
from __future__ import annotations

import argparse, json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]   # 상대경로 (대회 §3.3 요건)
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
sys.path.insert(0, str(REPO / "src"))
from model_c import ResidualSimVPC, T, C                       # noqa: E402
from loader_c import TARGET_H, TARGET_W, load_action_stats     # noqa: E402
from wm_eval import data_utils as D                            # noqa: E402

EVAL = REPO / "open" / "data" / "eval"
FPS = 6
NATIVE_H, NATIVE_W = 480, 640          # [코드] eval 216 장 전부 640×480 (007 확인)


def pad_geometry(h=NATIVE_H, w=NATIVE_W):
    scale = min(TARGET_H / h, TARGET_W / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    pt = (TARGET_H - rh) // 2
    pl = (TARGET_W - rw) // 2
    return rh, rw, pt, pl


def to_model_space(img_u8: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (1,3,320,512) float [-1,1]. loader_c 규약과 동일."""
    t = torch.from_numpy(img_u8).float().permute(2, 0, 1)[None] / 255.0
    rh, rw, pt, pl = pad_geometry(*img_u8.shape[:2])
    t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
    t = F.pad(t, (pl, TARGET_W - rw - pl, pt, TARGET_H - rh - pt), value=0.0)
    return (t - 0.5) * 2.0


def residual_to_native(res: torch.Tensor, h=NATIVE_H, w=NATIVE_W) -> torch.Tensor:
    """(1,3,T,320,512) 잔차 → (1,3,T,h,w). 패딩 제거 후 원본 크기로 되돌린다."""
    rh, rw, pt, pl = pad_geometry(h, w)
    r = res[:, :, :, pt:pt + rh, pl:pl + rw]                     # 패딩 제거
    B, Cc, Tn = r.shape[:3]
    r = r.permute(0, 2, 1, 3, 4).reshape(B * Tn, Cc, rh, rw)
    r = F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False)
    return r.reshape(B, Tn, Cc, h, w).permute(0, 2, 1, 3, 4)


@torch.no_grad()
def run(ckpt: Path, lams: list[float], outroot: Path, limit=None, device="cuda"):
    ck = torch.load(ckpt, map_location="cpu")
    hid_s = ck.get("args", {}).get("hid_s", 64)
    model = ResidualSimVPC(hid_S=hid_s, use_ckpt=False).to(device).eval()
    model.load_state_dict(ck["model"])
    mu, sd = load_action_stats()
    mu, sd = mu.to(device), sd.to(device)
    print(f"[ckpt] {ckpt.name}  step {ck.get('step')}  hid_S={hid_s}")

    pngs = sorted((EVAL / "images").glob("sample_*.png"))[:limit]
    dirs = {}
    for lam in lams:
        d = outroot / f"lam{lam:g}"
        d.mkdir(parents=True, exist_ok=True)
        dirs[lam] = d
    print(f"[eval] {len(pngs)} 표본 × λ {lams}")

    t0 = time.perf_counter()
    for i, p in enumerate(pngs, 1):
        img = np.asarray(Image.open(p).convert("RGB"))
        first_m = to_model_space(img).to(device)                 # (1,3,320,512)
        act = np.load(EVAL / "actions" / f"{p.stem}.npy").astype(np.float32)
        a = ((torch.from_numpy(act).to(device) - mu) / sd)[None]  # (1,T,6)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, res = model(first_m, a, return_residual=True)
        res_n = residual_to_native(res.float())                   # (1,3,T,480,640)
        base = torch.from_numpy(img).float().permute(2, 0, 1)[None, :, None].to(device)
        base = (base / 255.0 - 0.5) * 2.0                         # 원본을 [-1,1] 로
        for lam in lams:
            out = (base + lam * res_n).clamp(-1, 1)
            u8 = ((out[0].permute(1, 2, 3, 0) + 1) / 2 * 255).round().byte().cpu().numpy()
            D.save_mp4_uint8(u8, dirs[lam] / f"{p.stem}.mp4", fps=FPS)
        if i % 40 == 0 or i == len(pngs):
            print(f"  {i}/{len(pngs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    el = time.perf_counter() - t0
    print(f"\n[완료] {el:.0f}s  (추론 제한 1시간 대비 {el/3600*100:.1f}%)")
    for lam, d in dirs.items():
        n = len(list(d.glob("*.mp4")))
        print(f"  λ={lam:g}  mp4 {n}개  {d}")
        if limit is None and n != 216:
            print(f"    ⚠ 216 개가 아니다")
    return dirs


def selftest():
    """λ=0 이 원본과 정확히 같은가 — 하방 봉쇄가 제출 경로에서도 성립하는지."""
    print("=== λ=0 이 원본 프레임과 완전히 같은가 ===")
    p = sorted((EVAL / "images").glob("sample_*.png"))[0]
    img = np.asarray(Image.open(p).convert("RGB"))
    base = (torch.from_numpy(img).float().permute(2, 0, 1)[None, :, None] / 255.0 - 0.5) * 2.0
    u8 = ((base[0].permute(1, 2, 3, 0) + 1) / 2 * 255).round().byte().numpy()
    print(f"   왕복 후 최대 차이 = {np.abs(u8[0].astype(int) - img.astype(int)).max()}   ← 0 이어야 한다")

    print("\n=== 참고: 전체 프레임이 왕복 변환을 겪으면 얼마나 잃나 ===")
    m = to_model_space(img)
    rh, rw, pt, pl = pad_geometry()
    back = F.interpolate(m[:, :, pt:pt + rh, pl:pl + rw], size=(NATIVE_H, NATIVE_W),
                         mode="bilinear", align_corners=False)
    b8 = (((back[0].permute(1, 2, 0) + 1) / 2) * 255).round().byte().numpy()
    d = np.abs(b8.astype(int) - img.astype(int))
    print(f"   320×512 왕복 시 픽셀 차이 평균 {d.mean():.2f} / 최대 {d.max()}  (0~255 척도)")
    print("   ⇒ 잔차 구조는 이 손실을 **잔차에만** 물린다. 첫 프레임은 원본 그대로다")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or not args.ckpt:
        selftest()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out = REPO / "artifacts" / "branchC" / f"infer_{ts}"
        dirs = run(Path(args.ckpt), args.lam, out, limit=args.limit)
        json.dump({"ckpt": args.ckpt, "lams": args.lam,
                   "dirs": {str(k): str(v) for k, v in dirs.items()}},
                  open(out / "manifest.json", "w"), indent=2, ensure_ascii=False)
        print(f"\n다음: submission_kit 으로 CSV 생성 (최종 mp4 확정 후 1회만, rule §12.5)")
