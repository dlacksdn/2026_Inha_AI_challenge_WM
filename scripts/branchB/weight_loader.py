"""branch B 공용 가중치 로더 — DC 사전학습 UNet을 우리 1.1B UNet에 안전하게 얹는다.

이 파일이 존재하는 이유(010 §4.1 함정 A)
  `load_state_dict(..., strict=False)` 는 "이름이 없는 키"만 봐준다.
  **이름은 있는데 shape 가 다르면 RuntimeError 를 던진다.** 그래서 얹기 전에 그 키를 빼야 한다.
  이 필터 로직을 학습 런처와 검증 스크립트가 공유하도록 한 곳에 둔다.

0초기화(zero-init)
  `add_act_time_emb=True`(액션 가산) 구성에서 `action_embed` 마지막 층을 0으로 만들면,
  학습 시작 시점의 UNet 출력이 DC 사전학습과 **정확히 동일**해진다(010 §5.3 실측: cos 1.00000).
  액션은 첫 optimizer step 이후부터 서서히 개입한다. baseline 이 `fps_embedding` 에 쓰는 기법과 같다
  (openaimodel3d.py 451~452행).
"""
from __future__ import annotations

from pathlib import Path

import torch

UNET_PREFIX = "model.diffusion_model."


def read_unet_state(ckpt_path: str | Path) -> dict:
    """ckpt 에서 UNet 파라미터만 뽑아 prefix 를 떼고 돌려준다(mmap lazy)."""
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {k[len(UNET_PREFIX):]: v for k, v in sd.items() if k.startswith(UNET_PREFIX)}


def filter_loadable(target_state: dict, src: dict) -> tuple[dict, list]:
    """target 에 실제로 얹을 수 있는 것만 남기고, shape 불일치 목록을 함께 돌려준다."""
    mismatch, ok = [], {}
    for k, v in src.items():
        if k not in target_state:
            continue                                   # target 에 없는 키(unexpected) — 버린다
        if tuple(v.shape) != tuple(target_state[k].shape):
            mismatch.append({"key": k, "target": list(target_state[k].shape), "src": list(v.shape)})
        else:
            ok[k] = v
    return ok, mismatch


def zero_init_action_branch(unet) -> dict:
    """action_embed 의 마지막 Linear 를 0으로. (null_action_emb 은 이미 0이다)"""
    info = {"applied": False}
    emb = getattr(unet, "action_embed", None)
    if emb is None:
        return info
    last = emb[-1]
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()
        nul = getattr(unet, "null_action_emb", None)
        if nul is not None:
            nul.zero_()
    info.update(applied=True, layer=str(tuple(last.weight.shape)),
                weight_absmax=float(last.weight.abs().max()))
    return info


def warm_start_full_model(model, ckpt_path: str | Path, zero_init_action: bool = True,
                          verbose: bool = True) -> dict:
    """LatentVisualDiffusion 전체 모델에 DC UNet 을 얹는다 (학습 런처가 부르는 진입점).

    model.state_dict() 의 UNet 키는 "model.diffusion_model.*" 이다.
    반환: 로드 통계(그대로 로그·JSON 에 남길 수 있는 형태).
    """
    own = model.state_dict()
    unet_state = {k[len(UNET_PREFIX):]: v for k, v in own.items() if k.startswith(UNET_PREFIX)}
    src = read_unet_state(ckpt_path)
    ok, mismatch = filter_loadable(unet_state, src)
    prefixed = {UNET_PREFIX + k: v for k, v in ok.items()}

    missing, unexpected = model.load_state_dict(prefixed, strict=False)
    unet_missing = sorted(k for k in missing if k.startswith(UNET_PREFIX))
    other_missing = [k for k in missing if not k.startswith(UNET_PREFIX)]

    info = {
        "ckpt": str(ckpt_path),
        "src_unet_keys": len(src),
        "loaded_keys": len(ok),
        "loaded_params_m": round(sum(v.numel() for v in ok.values()) / 1e6, 3),
        "dropped_shape_mismatch": mismatch,
        "unet_missing": unet_missing,
        "unet_missing_params_m": round(sum(own[k].numel() for k in unet_missing) / 1e6, 4),
        "unet_unexpected": [k for k in unexpected if k.startswith(UNET_PREFIX)],
        "non_unet_missing_count": len(other_missing),   # VAE/CLIP 등은 이미 로드되어 있으므로 정상
    }
    if zero_init_action:
        info["zero_init"] = zero_init_action_branch(model.model.diffusion_model)

    if verbose:
        print(f">>> [branchB] warm-start from {ckpt_path}")
        print(f">>>   적재 {info['loaded_keys']}키 / {info['loaded_params_m']}M "
              f"(src UNet {info['src_unet_keys']}키)")
        print(f">>>   shape 불일치로 버린 키: {len(mismatch)}  {[m['key'] for m in mismatch]}")
        print(f">>>   unet_missing {len(unet_missing)}키 ({info['unet_missing_params_m']}M): "
              f"{unet_missing[:12]}{' ...' if len(unet_missing) > 12 else ''}")
        print(f">>>   unet_unexpected {len(info['unet_unexpected'])}키")
        print(f">>>   zero_init: {info.get('zero_init')}")
    return info


def warm_start_unet(unet, ckpt_path: str | Path, zero_init_action: bool = True,
                    verbose: bool = True) -> dict:
    """UNet 모듈 단독 버전(검증·프로브용)."""
    own = unet.state_dict()
    src = read_unet_state(ckpt_path)
    ok, mismatch = filter_loadable(own, src)
    missing, unexpected = unet.load_state_dict(ok, strict=False)
    info = {
        "src_unet_keys": len(src), "loaded_keys": len(ok),
        "loaded_params_m": round(sum(v.numel() for v in ok.values()) / 1e6, 3),
        "dropped_shape_mismatch": mismatch,
        "missing": sorted(missing), "unexpected": sorted(unexpected),
        "missing_params_m": round(sum(own[k].numel() for k in missing) / 1e6, 4),
    }
    if zero_init_action:
        info["zero_init"] = zero_init_action_branch(unet)
    if verbose:
        print(f"[warm_start_unet] 적재 {info['loaded_keys']}키 {info['loaded_params_m']}M / "
              f"불일치 버림 {len(mismatch)} / missing {len(info['missing'])} / "
              f"unexpected {len(info['unexpected'])} / zero_init {bool(info.get('zero_init', {}).get('applied'))}")
    return info
