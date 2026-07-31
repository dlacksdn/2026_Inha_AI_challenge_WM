"""VAE 왕복 오차 = 이 파이프라인이 낼 수 있는 **DINO 하한**을 잰다.

왜 필요한가
----------
확산모델은 픽셀을 직접 다루지 않는다. VAE(오토인코더)로 영상을 8배 작은 "잠재(latent)"로 줄인 뒤
그 안에서 그림을 그리고, 마지막에 다시 픽셀로 되돌린다. 이 **왕복(encode→decode)만으로도 손실**이 있다.

프레임별 DINO 분해에서 우리 생성물은 t=0(첫 프레임)에서 이미 0.048 만큼 틀렸다. 첫 프레임은
입력으로 주어진 이미지인데도 그렇다. 그 원인이 VAE 왕복이라면, **UNet 을 아무리 학습해도
이 0.048 은 사라지지 않는다.** 즉 DINO 목표 0.1248 의 달성 가능성 자체가 달라진다.

그래서 정답 영상을 그대로 VAE 에 넣었다 빼서 DINO 를 잰다. 그 값이 이 파이프라인의 바닥이다.

같이 재는 것
-----------
`get_first_stage_encoding` 은 잠재를 뽑을 때 `posterior.sample()` 을 쓴다(ddpm3d.py L654).
분포에서 **무작위로 뽑는다**는 뜻이라, 평균값(`mode()`)을 쓸 때보다 손실이 커진다.
둘을 나란히 재서 차이가 유의하면 그 자체가 개선 손잡이다(추론에서 mode 를 쓰면 공짜로 좋아진다).

사용
----
  python scripts/branchB/diag_vae_roundtrip.py --limit 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
CK = REPO / "open/baseline/challenge_kit"
for p in [CK / "libs" / "dynamicrafter", CK / "src", CK, REPO / "open/baseline/shared_libs/video_utils"]:
    sys.path.insert(0, str(p))
sys.path.insert(0, str(REPO / "src"))

from omegaconf import OmegaConf  # noqa: E402
from wm_eval import scoring as S  # noqa: E402


def load_vae(cfg_path: Path, backbone: Path, device: torch.device):
    """backbone.ckpt 의 first_stage_model 가중치로 AutoencoderKL 을 만든다."""
    from lvdm.utils.utils import instantiate_from_config

    cfg = OmegaConf.load(cfg_path)
    vae = instantiate_from_config(cfg.model.params.first_stage_config)
    sd = torch.load(str(backbone), map_location="cpu", weights_only=False)["state_dict"]
    pre = "first_stage_model."
    fs = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    missing, unexpected = vae.load_state_dict(fs, strict=False)
    print(f"[vae] 가중치 {len(fs)}키 로드 (missing {len(missing)} / unexpected {len(unexpected)})", flush=True)
    return vae.eval().to(device)


def roundtrip(vae, video_uint8: torch.Tensor, device, mode: str, scale_factor: float) -> torch.Tensor:
    """(16,320,512,3) uint8 → VAE 왕복 → (16,320,512,3) uint8.

    파이프라인과 동일하게 [-1,1] 로 정규화하고 scale_factor 를 곱했다 나눈다(왕복이라 상쇄되지만
    코드 경로를 같게 두어 다른 차이가 끼어들지 않게 한다).
    """
    x = video_uint8.float().to(device) / 127.5 - 1.0        # (16,H,W,3)
    x = x.permute(0, 3, 1, 2).contiguous()                  # (16,3,H,W)
    with torch.no_grad():
        post = vae.encode(x)
        z = post.sample() if mode == "sample" else post.mode()
        z = scale_factor * z
        rec = vae.decode(z / scale_factor)
    rec = rec.permute(0, 2, 3, 1)                           # (16,H,W,3)
    return ((rec.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats", default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--config", default=str(REPO / "scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml"))
    ap.add_argument("--backbone", default=str(REPO / "open/baseline/checkpoints/backbone.ckpt"))
    ap.add_argument("--static", default=str(REPO / "artifacts/branchB/m0_step1000_emafix/static_preds"))
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--out", default=str(REPO / "artifacts/branchB/diag_vae_roundtrip.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)
    scale_factor = float(cfg.model.params.scale_factor)
    print(f"[vae] scale_factor={scale_factor}  n={len(samples)}", flush=True)

    vae = load_vae(Path(args.config), Path(args.backbone), device)
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=args.device)
    gt_dir = holdout / "gt_videos"

    res = {m: {"gt": [], "static": []} for m in ("sample", "mode")}
    for i, mrow in enumerate(samples):
        sid = mrow["sid"]
        gt = scorer._load_video(gt_dir, sid)                # (1,16,320,512,3)
        gd = scorer.dino_feature(gt)[0]
        st = scorer._load_video(Path(args.static), sid)
        sd_feat = scorer.dino_feature(st)[0]

        for mode in ("sample", "mode"):
            rg = roundtrip(vae, gt[0], device, mode, scale_factor).unsqueeze(0)
            res[mode]["gt"].append(S.dino_component_frame_avg(scorer.dino_feature(rg)[0], gd))
            rs = roundtrip(vae, st[0], device, mode, scale_factor).unsqueeze(0)
            # static 왕복본을 "정답 첫프레임"이 아니라 static 원본과 비교하면 순수 왕복 손실이 나온다.
            res[mode]["static"].append(S.dino_component_frame_avg(scorer.dino_feature(rs)[0], sd_feat))
        if (i + 1) % 8 == 0:
            print(f"[vae] {i+1}/{len(samples)}", flush=True)

    out = {"n_samples": len(samples), "scale_factor": scale_factor, "results": {}}
    for mode in ("sample", "mode"):
        out["results"][mode] = {
            "gt_roundtrip_dino": float(np.mean(res[mode]["gt"])),
            "static_roundtrip_dino": float(np.mean(res[mode]["static"])),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"VAE 왕복 손실 = 이 파이프라인의 DINO 하한 (n={len(samples)})")
    print("=" * 78)
    print(f"{'잠재 추출':<16}{'정답영상 왕복':>16}{'정지영상 왕복':>16}")
    print("-" * 78)
    for mode in ("sample", "mode"):
        r = out["results"][mode]
        label = "sample(현재)" if mode == "sample" else "mode(평균)"
        print(f"{label:<16}{r['gt_roundtrip_dino']:>16.5f}{r['static_roundtrip_dino']:>16.5f}")
    print("-" * 78)
    print("  비교: 생성물의 t=0 오차 0.04757 / static 의 t=0 오차 0.00357 / DINO 목표 0.1248")
    print(f"\n[vae] 저장 → {args.out}")


if __name__ == "__main__":
    main()
