"""S3 — 1.1B UNet 에 DC 사전학습 가중치를 **실제로 로드**해 검증한다 (CPU, GPU 불필요).

S2(probe)는 meta device 로 "키·shape 대조"만 했다. 여기서는 실제 텐서를 올려서
다음 네 가지를 실측한다(009 §7 S3):

  (1) 순진한 로드가 정말 통하는가
      → `load_state_dict(strict=False)` 는 **shape 불일치에서 RuntimeError 를 던진다**(strict 무관).
        99.89% 정합에서 남는 `time_embed.2` 2키가 정확히 그 대상이다. 즉 필터 없는 로드는 학습 시작을 못 한다.
        이 사실을 예외를 받아서 증거로 남긴다.
  (2) 필터 로드 후 missing 이 009 §2.2 의 7키(action_embed 5 + time_embed.2 2)뿐인가
  (3) 값이 진짜 들어왔는가 — 로드된 텐서를 ckpt 원본과 bitwise 비교(이름만 맞고 값이 안 온 경우 배제)
  (4) 학습 런처가 쓸 full-model 키 경로(`model.diffusion_model.*`)에서도 같은 결론인가

옵션 --forward-smoke: 랜덤 입력으로 CPU forward 1회 (액션 concat 차원·fs 조건화 배선 확인).

출력: results/branchB/load_verify_1p1b.json  (기존 리포트를 덮어쓰지 않는 새 파일)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.utils.checkpoint  # noqa: F401  (함정: lvdm/common.py 가 torch.utils.checkpoint 를 임포트 없이 참조한다)
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_paths import repo_root  # noqa: E402

UNET_PREFIX = "model.diffusion_model."
EXPECTED_SCRATCH = {
    "action_embed.0.weight", "action_embed.0.bias",
    "action_embed.2.weight", "action_embed.2.bias",
    "null_action_emb",
    "time_embed.2.weight", "time_embed.2.bias",
}


def add_paths(ck: Path) -> None:
    for p in [ck / "libs" / "dynamicrafter", ck / "src", ck,
              ck.parent / "shared_libs" / "video_utils"]:
        sys.path.insert(0, str(p))


def n_params(sd) -> float:
    return sum(v.numel() for v in sd.values()) / 1e6


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge-kit", default=str(root / "open/baseline/challenge_kit"))
    ap.add_argument("--config", default=str(root / "scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml"))
    ap.add_argument("--backbone", default=str(root / "open/baseline/checkpoints/backbone.ckpt"))
    ap.add_argument("--out", default=str(root / "results/branchB/load_verify_1p1b.json"))
    ap.add_argument("--forward-smoke", action="store_true")
    args = ap.parse_args()

    ck = Path(args.challenge_kit)
    add_paths(ck)
    from lvdm.modules.networks.openaimodel3d import UNetModel  # noqa: E402

    rep: dict = {"config": args.config, "backbone": args.backbone}

    # ── 1. 실제(meta 아님) UNet 빌드 ────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    params = OmegaConf.to_container(cfg.model.params.unet_config.params, resolve=True)
    rep["unet_params"] = params
    t0 = time.time()
    unet = UNetModel(**params)
    unet.eval()
    build_s = time.time() - t0
    own = unet.state_dict()
    rep["build"] = {
        "seconds": round(build_s, 1),
        "keys": len(own),
        "params_m": round(n_params(own), 3),
        "fp32_gb": round(n_params(own) * 1e6 * 4 / 2**30, 2),
    }
    print(f"[1] UNet 빌드(CPU fp32): 키 {len(own)} / {n_params(own):.2f}M / "
          f"{rep['build']['fp32_gb']}GB / {build_s:.1f}s")

    # ── 2. backbone.ckpt 에서 UNet 부분만 추출 ─────────────────────────────────────
    obj = torch.load(args.backbone, map_location="cpu", mmap=True, weights_only=False)
    full_sd = obj.get("state_dict", obj)
    dc_sd = {k[len(UNET_PREFIX):]: v for k, v in full_sd.items() if k.startswith(UNET_PREFIX)}
    rep["ckpt"] = {"unet_keys": len(dc_sd), "unet_params_m": round(n_params(dc_sd), 3)}
    print(f"[2] backbone.ckpt UNet: 키 {len(dc_sd)} / {n_params(dc_sd):.2f}M")

    # ── 3. 순진한 로드 시도 (실패해야 정상) ────────────────────────────────────────
    naive = {"raised": False, "error_type": None, "error_head": None}
    try:
        unet.load_state_dict(dc_sd, strict=False)
        print("[3] 필터 없는 strict=False 로드가 통과했다 (예상과 다름 — 원인 규명 필요)")
    except Exception as e:  # noqa: BLE001
        naive.update(raised=True, error_type=type(e).__name__,
                     error_head=str(e).strip().splitlines()[0][:300],
                     n_size_mismatch_lines=sum(1 for ln in str(e).splitlines() if "size mismatch" in ln))
        print(f"[3] 필터 없는 로드 → {naive['error_type']}: size mismatch "
              f"{naive.get('n_size_mismatch_lines')}줄 (예상된 실패)")
    rep["naive_load"] = naive

    # ── 4. shape 불일치 키를 제거한 실제 로드 ──────────────────────────────────────
    mismatch = [k for k, v in dc_sd.items() if k in own and tuple(v.shape) != tuple(own[k].shape)]
    filtered = {k: v for k, v in dc_sd.items() if k not in mismatch}
    missing, unexpected = unet.load_state_dict(filtered, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    rep["filtered_load"] = {
        "dropped_shape_mismatch": [
            {"key": k, "ours": list(own[k].shape), "dc": list(dc_sd[k].shape)} for k in mismatch],
        "loaded_keys": len(filtered),
        "loaded_params_m": round(n_params(filtered), 3),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "missing_params_m": round(sum(own[k].numel() for k in missing) / 1e6, 4),
        "missing_matches_expected_7": sorted(missing) == sorted(EXPECTED_SCRATCH),
        "loaded_ratio_of_ours": round(n_params(filtered) / n_params(own), 6),
        "loaded_ratio_of_dc": round(n_params(filtered) / n_params(dc_sd), 6),
    }
    fl = rep["filtered_load"]
    print(f"[4] 필터 로드: {len(filtered)}키 {fl['loaded_params_m']:.2f}M 적재 / "
          f"missing {len(missing)}키 {fl['missing_params_m']:.3f}M / unexpected {len(unexpected)}키")
    print(f"    missing == 예상 7키? {fl['missing_matches_expected_7']}  → {sorted(missing)}")
    print(f"    우리 UNet 대비 {fl['loaded_ratio_of_ours']*100:.3f}% / "
          f"DC UNet 대비 {fl['loaded_ratio_of_dc']*100:.3f}% 흡수")

    # ── 5. 값이 진짜 들어왔는지 bitwise 확인 ───────────────────────────────────────
    now = unet.state_dict()
    checked, mismatched_vals = [], []
    cand = sorted(filtered, key=lambda k: -filtered[k].numel())[:8] + [
        k for k in ["input_blocks.0.0.weight", "out.2.weight", "time_embed.0.weight",
                    "fps_embedding.2.weight"] if k in filtered]
    for k in dict.fromkeys(cand):
        same = torch.equal(now[k].float(), filtered[k].float())
        checked.append({"key": k, "numel": int(filtered[k].numel()), "bitwise_equal": bool(same),
                        "abs_mean": float(filtered[k].float().abs().mean())})
        if not same:
            mismatched_vals.append(k)
    rep["value_check"] = {"checked": checked, "all_equal": not mismatched_vals}
    print(f"[5] 값 검증: {len(checked)}키 bitwise 비교 → 전부 일치? {not mismatched_vals}")

    # 스크래치 7키가 (로드되지 않은) 초기값인지도 확인 — time_embed.2 는 랜덤, null_action_emb 는 0
    scratch_stat = {k: {"abs_mean": float(now[k].float().abs().mean()), "shape": list(now[k].shape)}
                    for k in sorted(missing)}
    rep["scratch_state"] = scratch_stat
    print("[5b] 스크래치 7키 초기 상태:")
    for k, v in scratch_stat.items():
        print(f"     {k:<26} shape {str(v['shape']):<14} |mean| {v['abs_mean']:.5f}")

    # ── 6. full-model 키 경로(런처가 실제로 쓰는 형태)에서도 동일한가 ───────────────
    #     런처는 LatentVisualDiffusion 전체에 대해 load_state_dict 를 부른다 →
    #     UNet 키는 "model.diffusion_model.*" 이므로, 그 prefix 로 같은 필터를 만들어 키셋을 대조한다.
    full_expect_missing = {UNET_PREFIX + k for k in EXPECTED_SCRATCH}
    full_filtered = {UNET_PREFIX + k: v for k, v in filtered.items()}
    full_ours = {UNET_PREFIX + k for k in own}
    rep["full_model_keyspace"] = {
        "filtered_keys": len(full_filtered),
        "all_filtered_keys_exist_in_model": all(k in full_ours for k in full_filtered),
        "unet_missing_would_be": sorted(full_expect_missing),
        "unet_missing_count": len(full_expect_missing),
    }
    print(f"[6] full-model 경로: {len(full_filtered)}키 전부 모델에 존재? "
          f"{rep['full_model_keyspace']['all_filtered_keys_exist_in_model']} / "
          f"unet_missing 예상 {len(full_expect_missing)}키")

    # ── 7. (옵션) CPU forward 스모크 ───────────────────────────────────────────────
    if args.forward_smoke:
        # 15GB RAM 머신에서 돌리므로 ckpt 참조(mmap 페이지)를 먼저 놓아준다.
        import gc
        del obj, full_sd, dc_sd, filtered, now, own
        gc.collect()
        b, t = 1, params["temporal_length"]
        x = torch.randn(b, params["in_channels"], t, 40, 64)
        ts = torch.tensor([500] * b, dtype=torch.long)
        ctx = torch.randn(b, 77 + t * 16, params["context_dim"])   # text 77 + per-frame image 16
        act = torch.randn(b, t, params["action_dims"])
        sm = {}
        for tag, kw in [("fs=6", {"fs": torch.tensor([6] * b, dtype=torch.long)}),
                        ("fs=None(default_fs)", {})]:
            t1 = time.time()
            with torch.no_grad():
                y = unet(x, ts, context=ctx, act=act, dropout_actions=False, **kw)
            sm[tag] = {"out_shape": list(y.shape), "seconds": round(time.time() - t1, 1),
                       "finite": bool(torch.isfinite(y).all()), "std": float(y.float().std())}
            print(f"[7] forward {tag}: out {list(y.shape)} / {sm[tag]['seconds']}s / "
                  f"finite {sm[tag]['finite']} / std {sm[tag]['std']:.4f}")
        rep["forward_smoke"] = sm

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[verify] 리포트: {args.out}")

    ok = (fl["missing_matches_expected_7"] and not unexpected and rep["value_check"]["all_equal"]
          and rep["full_model_keyspace"]["all_filtered_keys_exist_in_model"])
    print(f"[verify] S3 판정: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
