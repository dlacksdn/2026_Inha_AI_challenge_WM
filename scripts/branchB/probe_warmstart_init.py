"""S3 후속 — warm-start 직후 "사전학습 함수가 얼마나 남아 있나"를 실측한다 (CPU).

문제의식
  99.89% 로드는 **키·파라미터 비율**이다. 그런데 액션 주입이 concat 방식이면
  `time_embed.2`(1280→640) 가 스크래치이므로, 사전학습된 emb_layers 1514키에
  **랜덤 시간 임베딩**이 들어간다. 비율이 99.89%여도 함수는 보존되지 않을 수 있다.
  009 §2.2 는 가산(add) 방식을 "코드 수정이 필요한 선택 옵션"으로 미뤄뒀는데,
  실제로는 baseline 코드에 `add_act_time_emb` 플래그가 이미 구현되어 있다
  (openaimodel3d.py __init__ 419~434행, forward 의 `emb = time_emb + act_emb`).
  ⇒ config 한 줄로 선택 가능하므로, 두 방식이 초기 함수를 얼마나 보존하는지 재서 결정한다.

방법 (같은 고정 입력에 대한 UNet 출력 비교)
  REF   : action_conditioned=False (= DC 원본 UNet 그 자체, 1516키 100% 로드) → 기준 함수
  CONCAT: 우리 config(액션 concat, time_embed.2 스크래치)
  ADD   : 우리 config + add_act_time_emb=True (time_embed.2 도 사전학습 로드)
  각 변형을 act=0(null)/act=랜덤 두 조건으로 돌려 REF 와의 코사인·상대L2 를 잰다.

메모리: 1.44B fp32 = 5.4GB. 15GB RAM 에서 돌리므로 **변형 하나씩** 빌드·로드·해제한다.
출력: results/branchB/warmstart_init_probe.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.utils.checkpoint  # noqa: F401  (lvdm/common.py 가 임포트 없이 참조)
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_paths import repo_root  # noqa: E402

ROOT = repo_root()
CK = ROOT / "open/baseline/challenge_kit"
CFG = ROOT / "scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml"
BACKBONE = ROOT / "open/baseline/checkpoints/backbone.ckpt"
UNET_PREFIX = "model.diffusion_model."

for p in [CK / "libs" / "dynamicrafter", CK / "src", CK, CK.parent / "shared_libs" / "video_utils"]:
    sys.path.insert(0, str(p))
from lvdm.modules.networks.openaimodel3d import UNetModel  # noqa: E402

VARIANTS = {
    "REF_no_action": {"action_conditioned": False},
    "CONCAT_ours": {},
    "ADD_act_time_emb": {"add_act_time_emb": True},
}


def make_inputs(params: dict, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = params["temporal_length"]
    return {
        "x": torch.randn(1, params["in_channels"], t, 40, 64, generator=g),
        "ts": torch.tensor([500], dtype=torch.long),
        "ctx": torch.randn(1, 77 + t * 16, params["context_dim"], generator=g),
        "act": torch.randn(1, t, params["action_dims"], generator=g),
        "fs": torch.tensor([6], dtype=torch.long),
    }


def build_and_load(params: dict) -> tuple[UNetModel, dict]:
    unet = UNetModel(**params)
    unet.eval()
    own = unet.state_dict()
    obj = torch.load(str(BACKBONE), map_location="cpu", mmap=True, weights_only=False)
    sd = obj.get("state_dict", obj)
    dc = {k[len(UNET_PREFIX):]: v for k, v in sd.items() if k.startswith(UNET_PREFIX)}
    mism = [k for k, v in dc.items() if k in own and tuple(v.shape) != tuple(own[k].shape)]
    filt = {k: v for k, v in dc.items() if k not in mism}
    missing, unexpected = unet.load_state_dict(filt, strict=False)
    info = {"loaded_keys": len(filt), "dropped_mismatch": mism,
            "missing": sorted(missing), "unexpected": sorted(unexpected),
            "loaded_params_m": round(sum(v.numel() for v in filt.values()) / 1e6, 3)}
    del obj, sd, dc, filt, own
    gc.collect()
    return unet, info


def main() -> None:
    base = OmegaConf.to_container(OmegaConf.load(str(CFG)).model.params.unet_config.params, resolve=True)
    inp = make_inputs(base)
    report: dict = {"variants": {}}
    outs: dict[str, torch.Tensor] = {}

    for name, ov in VARIANTS.items():
        params = {**base, **ov}
        t0 = time.time()
        unet, info = build_and_load(params)
        print(f"[{name}] 로드: {info['loaded_keys']}키 {info['loaded_params_m']}M, "
              f"missing {len(info['missing'])}, mismatch버림 {len(info['dropped_mismatch'])} "
              f"({time.time()-t0:.0f}s)")
        cases = {"act_null": None} if not params.get("action_conditioned", True) else {
            "act_null": None, "act_random": inp["act"]}
        per = {}
        for cname, act in cases.items():
            kw = {} if not params.get("action_conditioned", True) else {"act": act, "dropout_actions": False}
            t1 = time.time()
            with torch.no_grad():
                y = unet(inp["x"], inp["ts"], context=inp["ctx"], fs=inp["fs"], **kw)
            outs[f"{name}|{cname}"] = y.flatten().clone()
            per[cname] = {"std": round(float(y.std()), 5), "abs_mean": round(float(y.abs().mean()), 5),
                          "seconds": round(time.time() - t1, 1)}
            print(f"    {cname:<11} std {per[cname]['std']:.5f} ({per[cname]['seconds']}s)")
        report["variants"][name] = {"load": info, "forward": per}
        del unet
        gc.collect()

    ref = outs["REF_no_action|act_null"]
    comp = {}
    for k, v in outs.items():
        cos = float(torch.nn.functional.cosine_similarity(ref, v, dim=0))
        rel = float((v - ref).norm() / ref.norm())
        comp[k] = {"cos_vs_ref": round(cos, 5), "rel_l2_vs_ref": round(rel, 5)}
        print(f"  vs REF | {k:<28} cos {cos:+.5f}  rel-L2 {rel:.5f}")
    report["compare_vs_ref"] = comp

    out = ROOT / "results/branchB/warmstart_init_probe.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[probe] 리포트: {out}")


if __name__ == "__main__":
    main()
