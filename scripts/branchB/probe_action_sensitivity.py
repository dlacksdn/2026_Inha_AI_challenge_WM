"""액션 민감도 검사 — 모델이 액션 조건을 **실제로 쓰고 있는가**를 잰다.

왜 필요한가 (011 §3.3)
  우리는 액션을 concat 이 아니라 **가산**으로 주입하고 `action_embed` 를 0초기화했다.
  덕분에 학습 시작점이 DC 사전학습과 완전히 동일해지지만, 그 대가로 **초기 액션 영향은 정확히 0**이다.
  학습이 진행돼도 액션을 계속 무시하면(=시간 신호에 묻히면) 우리가 원하는 액션 조건부 세계모델이 아니다.
  그러므로 "액션을 바꾸면 예측이 달라지는가"를 학습 전/후로 비교해야 한다.

무엇을 재나 (샘플러 없이 UNet 한 번의 예측만으로 — 값싸고 결정론적)
  같은 (잠재, timestep, 이미지·텍스트 조건, fs) 에 대해 액션만 바꿔가며 UNet 출력을 뽑는다.
    S_AB  = ||y(act_A) − y(act_B)|| / ||y(act_A)||     서로 다른 두 실제 액션이 만드는 차이
    S_A0  = ||y(act_A) − y(act_null)|| / ||y(act_A)||  액션을 준 것과 안 준 것(=CFG uncond 분기)의 차이
  둘 다 0 에 가까우면 **모델이 액션을 무시하는 것**이다.
  0초기화 직후에는 정확히 0 이 나와야 정상이며(그것이 이 검사의 sanity check),
  학습이 진행되면 유의하게 커져야 한다.

사용
  # 학습 전(0초기화 직후) 기준값 — 0 이 나와야 정상
  python scripts/branchB/probe_action_sensitivity.py
  # 학습 후 체크포인트
  python scripts/branchB/probe_action_sensitivity.py \
      --ckpt artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/last.ckpt
  # GPU 사용(있으면 훨씬 빠름)
  python scripts/branchB/probe_action_sensitivity.py --device cuda --ckpt ...

출력: results/branchB/action_sensitivity_<태그>.json   (태그가 다르면 서로 덮어쓰지 않는다)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.utils.checkpoint  # noqa: F401  (lvdm/common.py 가 임포트 없이 참조)
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_paths import repo_root  # noqa: E402
from weight_loader import UNET_PREFIX, filter_loadable, warm_start_unet  # noqa: E402

ROOT = repo_root()
CK = ROOT / "open/baseline/challenge_kit"
for p in [CK / "libs" / "dynamicrafter", CK / "src", CK, CK.parent / "shared_libs" / "video_utils"]:
    sys.path.insert(0, str(p))
from lvdm.modules.networks.openaimodel3d import UNetModel  # noqa: E402


def load_actions(holdout: Path, n: int, dims: int, tlen: int) -> list[torch.Tensor]:
    """홀드아웃의 실제 액션을 z-정규화해 가져온다. 없으면 랜덤으로 대체한다."""
    stats_p = ROOT / "open/data/train/so100_action_statistics.json"
    files = sorted((holdout / "actions").glob("*.npy")) if (holdout / "actions").is_dir() else []
    if files and stats_p.exists():
        import numpy as np
        st = json.loads(stats_p.read_text(encoding="utf-8"))
        mean = np.asarray(st["mean"], dtype=np.float32)
        std = np.maximum(np.asarray(st["std"], dtype=np.float32), 1e-8)
        out = []
        for f in files[:n]:
            a = np.load(f).astype(np.float32)[:tlen]
            out.append(torch.from_numpy((a - mean) / std).unsqueeze(0))
        if len(out) >= 2:
            return out
    g = torch.Generator().manual_seed(7)
    return [torch.randn(1, tlen, dims, generator=g) for _ in range(max(n, 2))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml"))
    ap.add_argument("--ckpt", default=None, help="학습된 ckpt (없으면 DC backbone + 0초기화 = 학습 전 기준값)")
    ap.add_argument("--backbone", default=str(ROOT / "open/baseline/checkpoints/backbone.ckpt"))
    ap.add_argument("--holdout", default=str(ROOT / "artifacts/holdout"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None, help="출력 파일 태그(기본: ckpt 파일명 또는 'init')")
    ap.add_argument("--n-pairs", type=int, default=3, help="비교할 (액션A, 액션B) 쌍의 수")
    args = ap.parse_args()

    params = OmegaConf.to_container(OmegaConf.load(args.config).model.params.unet_config.params, resolve=True)
    unet = UNetModel(**params).eval()

    if args.ckpt:
        obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj)
        src = {k[len(UNET_PREFIX):]: v for k, v in sd.items() if k.startswith(UNET_PREFIX)}
        if not src:      # EMA 만 있거나 prefix 가 다른 경우 대비
            raise SystemExit(f"[sensitivity] ckpt 에 '{UNET_PREFIX}*' 키가 없다: {list(sd)[:5]}")
        ok, mism = filter_loadable(unet.state_dict(), src)
        missing, unexpected = unet.load_state_dict(ok, strict=False)
        load_info = {"from": args.ckpt, "loaded_keys": len(ok), "dropped_mismatch": len(mism),
                     "missing": len(missing), "unexpected": len(unexpected)}
        tag = args.tag or Path(args.ckpt).stem
    else:
        load_info = warm_start_unet(unet, args.backbone, zero_init_action=True, verbose=False)
        load_info = {"from": "backbone+zero_init", "loaded_keys": load_info["loaded_keys"],
                     "missing": len(load_info["missing"]), "zero_init": True}
        tag = args.tag or "init"
    print(f"[sensitivity] 가중치: {load_info}")

    dev = torch.device(args.device)
    unet.to(dev)
    t = params["temporal_length"]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, params["in_channels"], t, 40, 64, generator=g).to(dev)
    ts = torch.tensor([500], dtype=torch.long, device=dev)
    ctx = torch.randn(1, 77 + t * 16, params["context_dim"], generator=g).to(dev)
    fs = torch.tensor([6], dtype=torch.long, device=dev)

    acts = load_actions(Path(args.holdout), args.n_pairs + 1, params["action_dims"], t)
    print(f"[sensitivity] 액션 {len(acts)}개 (출처: "
          f"{'홀드아웃 실제 액션' if (Path(args.holdout)/'actions').is_dir() else '랜덤'})")

    def fwd(act):
        kw = {"act": act.to(dev), "dropout_actions": False} if act is not None else \
             {"act": None, "dropout_actions": False}
        with torch.no_grad():
            return unet(x, ts, context=ctx, fs=fs, **kw).flatten().float().cpu()

    t0 = time.time()
    y_null = fwd(None)                      # 액션 없음(= null_action_emb, CFG uncond 분기)
    ys = [fwd(a) for a in acts]
    print(f"[sensitivity] forward {len(ys)+1}회 / {time.time()-t0:.0f}s")

    def rel(a, b):
        return float((a - b).norm() / a.norm())

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    pairs = []
    for i in range(min(args.n_pairs, len(ys) - 1)):
        pairs.append({"pair": f"act{i} vs act{i+1}",
                      "S_AB_rel": round(rel(ys[i], ys[i + 1]), 6),
                      "cos_AB": round(cos(ys[i], ys[i + 1]), 6)})
    a0 = [{"case": f"act{i} vs null", "S_A0_rel": round(rel(y, y_null), 6),
           "cos_A0": round(cos(y, y_null), 6)} for i, y in enumerate(ys)]

    mean_ab = sum(p["S_AB_rel"] for p in pairs) / max(len(pairs), 1)
    mean_a0 = sum(c["S_A0_rel"] for c in a0) / max(len(a0), 1)
    rep = {"tag": tag, "load": load_info, "device": args.device,
           "pairs_action_vs_action": pairs, "action_vs_null": a0,
           "mean_S_AB_rel": round(mean_ab, 6), "mean_S_A0_rel": round(mean_a0, 6),
           "verdict": ("액션 무시(또는 0초기화 직후)" if max(mean_ab, mean_a0) < 1e-4
                       else "액션에 반응함")}

    out = ROOT / f"results/branchB/action_sensitivity_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[액션 민감도] 0 이면 모델이 액션을 무시하는 것 (0초기화 직후에는 정확히 0 이 정상)")
    for p in pairs:
        print(f"   {p['pair']:<16} 상대차 {p['S_AB_rel']:.6f}   cos {p['cos_AB']:.6f}")
    for c in a0:
        print(f"   {c['case']:<16} 상대차 {c['S_A0_rel']:.6f}   cos {c['cos_A0']:.6f}")
    print(f"   평균 S_AB={mean_ab:.6f} / S_A0={mean_a0:.6f}  → {rep['verdict']}")
    print(f"[sensitivity] 리포트: {out}")


if __name__ == "__main__":
    main()
