"""학습 범위(scope)별 파라미터 수와 VRAM 예산을 계산한다 — 5090(32GB) vs 6000(96GB) 판단용.

왜 필요한가
  009 §3: "1차는 전체 파인튜닝(96GB면 가능). 32GB면 액션분기+시간축+LoRA 부분 학습".
  그런데 **UNet 키 이름에 'temporal' 문자열이 없다**(005의 오진 원인). 시간축 레이어는
  TemporalTransformer / TemporalConvBlock 이라는 **모듈 타입**으로만 식별된다.
  그래서 이름이 아니라 타입으로 찾아 범위를 정의하고, 각 범위의 학습 메모리를 산술로 낸다.

메모리 산술(AdamW, fp32 마스터 가중치 기준)
  파라미터 4B/개(전부)  +  gradient 4B/개(학습 대상만)  +  AdamW state 8B/개(학습 대상만)
  + 활성화(gradient checkpointing 시 작음, 여기선 미포함 — 실측은 S4 스모크에서)

출력: results/branchB/train_scope_budget.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_paths import repo_root  # noqa: E402

ROOT = repo_root()
CK = ROOT / "open/baseline/challenge_kit"
CFG = ROOT / "scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml"
for p in [CK / "libs" / "dynamicrafter", CK / "src", CK, CK.parent / "shared_libs" / "video_utils"]:
    sys.path.insert(0, str(p))
from lvdm.modules.networks.openaimodel3d import UNetModel  # noqa: E402
from train_scope import SCOPES, group_stats  # noqa: E402  (학습 런처와 **같은** 분류 로직을 쓴다)


def main() -> None:
    params = OmegaConf.to_container(OmegaConf.load(str(CFG)).model.params.unet_config.params, resolve=True)
    with torch.device("meta"):
        unet = UNetModel(**params)
    by, tnames, tprefix = group_stats(unet)
    total = sum(v[1] for v in by.values())
    rep = {
        "unet_total_params_m": round(total / 1e6, 3),
        "temporal_module_types": tnames,
        "n_temporal_modules": len(tprefix),
        "by_group": {k: {"keys": v[0], "params_m": round(v[1] / 1e6, 3),
                         "share_pct": round(v[1] / total * 100, 2)} for k, v in sorted(by.items())},
        "scopes": {},
    }
    for sname, groups in SCOPES.items():
        tr = sum(by[g][1] for g in groups if g in by)
        gb = lambda n: round(n / 2**30, 2)  # noqa: E731
        mem_params = total * 4
        mem_grad = tr * 4
        mem_adam = tr * 8
        rep["scopes"][sname] = {
            "trainable_params_m": round(tr / 1e6, 3),
            "trainable_share_pct": round(tr / total * 100, 2),
            "gb_params_fp32": gb(mem_params),
            "gb_grads": gb(mem_grad),
            "gb_adamw_state": gb(mem_adam),
            "gb_unet_subtotal": gb(mem_params + mem_grad + mem_adam),
            # 동결 모듈(VAE 83.7M + CLIP 354M + embedder 683M + Resampler 48.8M)은 추론만 → fp32 기준
            "gb_frozen_modules": gb((83.7 + 354 + 683 + 48.8) * 1e6 * 4),
            "gb_ema_copy": gb(total * 4),   # use_ema=True → UNet 사본 1개 더
        }
        s = rep["scopes"][sname]
        s["gb_total_estimate"] = round(s["gb_unet_subtotal"] + s["gb_frozen_modules"] + s["gb_ema_copy"], 2)

    out = ROOT / "results/branchB/train_scope_budget.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"UNet 총 {rep['unet_total_params_m']}M / 시간축 모듈 {rep['n_temporal_modules']}개 "
          f"(타입: {', '.join(tnames)})")
    print("\n[그룹별 파라미터]")
    for k, v in rep["by_group"].items():
        print(f"  {k:<14} {v['keys']:>5}키 {v['params_m']:>10.2f}M  ({v['share_pct']:>5.2f}%)")
    print("\n[학습 범위별 메모리 산술 — 활성화 제외]")
    print(f"{'scope':<18}{'학습대상M':>11}{'비율':>8}{'params':>9}{'grads':>8}{'adam':>8}{'frozen':>9}{'EMA':>7}{'합계GB':>9}")
    for k, v in rep["scopes"].items():
        print(f"{k:<18}{v['trainable_params_m']:>11.1f}{v['trainable_share_pct']:>7.1f}%"
              f"{v['gb_params_fp32']:>9.2f}{v['gb_grads']:>8.2f}{v['gb_adamw_state']:>8.2f}"
              f"{v['gb_frozen_modules']:>9.2f}{v['gb_ema_copy']:>7.2f}{v['gb_total_estimate']:>9.2f}")
    print("\n※ 활성화 메모리는 별도(use_checkpoint=True 로 작게 유지). 실측은 S4 스모크에서.")
    print(f"[scope] 리포트: {out}")


if __name__ == "__main__":
    main()
