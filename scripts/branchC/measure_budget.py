"""C 설계용 VRAM·처리량 예산표.

무엇을 재나 — 백본을 아직 안 정했으므로 **백본과 무관한 것만** 잰다.

  A. 채점기 3종을 *손실로* 쓸 때의 비용 (동결·forward+backward, 입력 픽셀까지 grad)
     DINOv2 ViT-S(30%) / R3D-18(30%) / action extractor(40%)
     → 이게 "손실 예산". 백본이 쓸 수 있는 나머지 VRAM 을 결정한다.

  B. 잔차 백본이 쓸 수 있는 나머지로 무엇이 가능한가
     - 3D conv 스택의 활성값 비용 (해상도·채널별)
     - spatiotemporal attention 의 토큰 수와 실제 메모리 (다운샘플 배율별)
       017 §7.2 가 권한 "3D conv + spatiotemporal attention" 이 우리 해상도에서 되는지.

왜 이렇게 재나 — 019 §7 이 "VRAM·처리량 예산표가 없다. 해상도를 설계 변수로 다루려면
숫자가 필요하다" 를 미해결로 남겼다. 그 칸을 채운다.

⚠ 이 스크립트는 채점기를 **읽기만** 한다. submission_kit 은 건드리지 않는다(규칙 §4.1-4).
⚠ 재현 검증 기준 장비는 RTX PRO 6000 96GB 다(rule/001 §3.3). 5090 32GB 를 통과하면
   재현 검증은 자동 통과이므로, 여기서 재는 것은 보수적인 쪽이다.

사용:
  $PY scripts/branchC/measure_budget.py --out results/branchC/budget.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SUBMISSION_KIT = REPO / "open" / "submission_kit"

# 채점 공간 규격 (018 §9-b 에서 확인: batch["video"] 가 이미 이 규격이다)
T, H, W = 16, 320, 512
DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"


def _add_kit_to_path() -> None:
    p = str(SUBMISSION_KIT)
    if p not in sys.path:
        sys.path.insert(0, p)


def _sync() -> None:
    torch.cuda.synchronize()


def _reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def _alloc_gib() -> float:
    return torch.cuda.memory_allocated() / 2**30


# ---------------------------------------------------------------- 전처리 (grad 흐르게)
# ⚠ 스케일 규약 함정 (018 §8.2): 데이터로더/모델 출력은 [-1,1] 인데
#    채점기 전처리는 0~255 를 받는다. 반드시 되돌려야 한다.


def m11_to_255(x_m11: torch.Tensor) -> torch.Tensor:
    """(B,3,T,H,W) [-1,1] -> (B,T,H,W,3) 0~255. 채점기가 받는 레이아웃."""
    x = (x_m11 + 1.0) / 2.0 * 255.0
    return x.permute(0, 2, 3, 4, 1)


def dino_forward(x_m11: torch.Tensor, model, F, image_size: int, frame_idx=None):
    """extract_dino_features 의 no_grad 를 뺀 판. 입력 픽셀까지 grad 가 흐른다."""
    v255 = m11_to_255(x_m11)  # (B,T,H,W,C)
    if frame_idx is not None:
        v255 = v255[:, frame_idx]
    b, t, h, w, c = v255.shape
    frames = v255.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    frames = F._resize_pad_frame_batch(frames, image_size, pad_value=0.0)
    frames = (frames - F.IMAGENET_MEAN.to(frames.device)) / F.IMAGENET_STD.to(frames.device)
    out = model(frames)
    out = F._normalize_image_model_output(out)
    return out.reshape(b, t, -1)


def r3d_forward(x_m11: torch.Tensor, model, F):
    v255 = m11_to_255(x_m11)  # (B,T,H,W,C)
    x = v255.permute(0, 4, 1, 2, 3).float() / 255.0  # (B,C,T,H,W)
    x = torch.nn.functional.interpolate(
        x, size=(x.shape[2], 112, 112), mode="trilinear", align_corners=False
    )
    x = (x - F.KINETICS_MEAN.to(x.device)) / F.KINETICS_STD.to(x.device)
    return model(x)


def action_forward(x_m11: torch.Tensor, model, F):
    """⚠ action extractor 는 양방향 GRU 를 쓴다(018 §8.2).

    eval() 모드에서 cuDNN RNN 은 backward 를 거부한다
    ("cudnn RNN backward can only be called in training mode").

    두 가지 우회가 있는데 하나만 규칙에 맞는다.
      ✗ model.train() 으로 바꾼다  → BN/dropout 거동이 바뀌어 **출력값이 달라진다.**
                                     규칙 §4.1-4("평가용 모델의 출력값이 달라지도록
                                     수정하는 행위 = 위반")에 걸리고, 손실 신호가
                                     채점 신호와 달라진다.
      ✓ cuDNN 만 끈다             → 같은 수식을 native 커널로 계산한다. 출력 동일.
                                     대신 느리다. 그 비용을 이 스크립트가 잰다.

    forward 를 cudnn 끈 채로 기록하면 autograd 가 native backward 를 물고 가므로
    backward 는 컨텍스트 밖에서 불러도 된다.
    """
    v255 = m11_to_255(x_m11)  # (B,T,H,W,C)
    frames = F.preprocess_images(v255)  # -> (B,C,T,H,W), /255*2-1
    with torch.backends.cudnn.flags(enabled=False):
        return model(frames)


# ---------------------------------------------------------------- 측정 헬퍼


def bench(fn, x: torch.Tensor, iters: int = 3):
    """forward+backward 의 peak VRAM 과 스텝당 시간. OOM 이면 None."""
    try:
        _reset_peak()
        base = _alloc_gib()
        # warmup
        out = fn(x)
        loss = out.float().pow(2).mean()
        loss.backward()
        _sync()
        if x.grad is not None:
            x.grad = None
        _reset_peak()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(x)
            loss = out.float().pow(2).mean()
            loss.backward()
            if x.grad is not None:
                x.grad = None
        _sync()
        dt = (time.perf_counter() - t0) / iters
        peak = _peak_gib()
        grad_ok = True
        return {
            "peak_gib": round(peak, 3),
            "activation_gib": round(peak - base, 3),
            "ms_per_step": round(dt * 1000, 1),
            "grad_ok": grad_ok,
        }
    except torch.cuda.OutOfMemoryError:
        _reset_peak()
        return {"oom": True}
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _reset_peak()
            return {"oom": True}
        raise


def new_input(batch: int, dev) -> torch.Tensor:
    x = torch.randn(batch, 3, T, H, W, device=dev, dtype=torch.float32).clamp_(-1, 1)
    x.requires_grad_(True)
    return x


# ---------------------------------------------------------------- A. 채점기 손실 비용


def measure_scorers(args, dev) -> dict:
    _add_kit_to_path()
    import feature_csv_utils as F  # noqa: E402

    report: dict = {"weights_gib": {}, "per_scorer": {}, "combined": {}}

    _reset_peak()
    w0 = _alloc_gib()
    dino = F.load_dino_model(dev, DINO_MODEL, pretrained=True).eval().requires_grad_(False)
    image_size = F.resolve_dino_image_size(dino, 518)
    w_dino = _alloc_gib()
    r3d = F.load_video_feature_model(dev, pretrained=True).eval().requires_grad_(False)
    w_r3d = _alloc_gib()
    ckpt = SUBMISSION_KIT / "checkpoints" / "action_extractor.ckpt"
    act = F.load_action_extractor(str(ckpt), dev)
    act.eval().requires_grad_(False)
    w_act = _alloc_gib()

    report["weights_gib"] = {
        "dino": round(w_dino - w0, 3),
        "r3d": round(w_r3d - w_dino, 3),
        "action": round(w_act - w_r3d, 3),
        "total": round(w_act - w0, 3),
    }
    report["dino_image_size"] = image_size

    # 새너티: 그래디언트가 입력 픽셀까지 흐르나 (017 §5.1③ 재확인)
    xs = new_input(1, dev)
    sanity = {}
    for name, fn in (
        ("dino", lambda z: dino_forward(z, dino, F, image_size)),
        ("r3d", lambda z: r3d_forward(z, r3d, F)),
        ("action", lambda z: action_forward(z, act, F)),
    ):
        xs.grad = None
        out = fn(xs)
        out.float().pow(2).mean().backward()
        g = xs.grad
        sanity[name] = {
            "out_shape": list(out.shape),
            "grad_finite": bool(torch.isfinite(g).all().item()),
            "grad_nonzero_frac": round(float((g != 0).float().mean().item()), 4),
        }
        xs.grad = None
    report["sanity"] = sanity

    # ⚠ 규칙 근거용 새너티: cuDNN 을 꺼도 action extractor 의 **출력이 같아야** 한다.
    #   같지 않으면 "출력값이 달라지도록 수정" (§4.1-4) 에 해당해 손실로 쓸 수 없다.
    with torch.no_grad():
        v255 = m11_to_255(xs.detach())
        frames = F.preprocess_images(v255)
        with torch.backends.cudnn.flags(enabled=True):
            ref = act(frames).float()
        with torch.backends.cudnn.flags(enabled=False):
            alt = act(frames).float()
        denom = ref.abs().max().clamp_min(1e-12)
        report["cudnn_equivalence"] = {
            "max_abs_diff": float((ref - alt).abs().max().item()),
            "max_rel_diff": float(((ref - alt).abs().max() / denom).item()),
            "out_shape": list(ref.shape),
            "verdict_note": "출력이 같아야 §4.1-4 위반이 아니다",
        }
    del xs

    sub_k = args.dino_subsample
    frame_idx = torch.arange(0, T, sub_k, device=dev)

    for b in args.batches:
        try:
            x = new_input(b, dev)
        except torch.cuda.OutOfMemoryError:
            report["per_scorer"][f"b{b}"] = {"input_oom": True}
            continue
        entry = {}
        entry["dino_full16"] = bench(lambda z: dino_forward(z, dino, F, image_size), x)
        entry[f"dino_sub{sub_k}"] = bench(
            lambda z: dino_forward(z, dino, F, image_size, frame_idx), x
        )
        entry["r3d"] = bench(lambda z: r3d_forward(z, r3d, F), x)
        entry["action"] = bench(lambda z: action_forward(z, act, F), x)

        def all_three(z):
            a = dino_forward(z, dino, F, image_size)
            c = r3d_forward(z, r3d, F)
            d = action_forward(z, act, F)
            return torch.cat([a.flatten(), c.flatten(), d.flatten()])

        def sub_three(z):
            a = dino_forward(z, dino, F, image_size, frame_idx)
            c = r3d_forward(z, r3d, F)
            d = action_forward(z, act, F)
            return torch.cat([a.flatten(), c.flatten(), d.flatten()])

        entry["all_three_full16"] = bench(all_three, x)
        entry[f"all_three_dinosub{sub_k}"] = bench(sub_three, x)
        report["per_scorer"][f"b{b}"] = entry
        del x
        _reset_peak()

    del dino, r3d, act
    _reset_peak()
    return report


# ---------------------------------------------------------------- B. 백본 여지


def measure_backbone_room(args, dev) -> dict:
    """백본이 쓸 수 있는 여지. 대표 연산 두 가지로 잰다."""
    out: dict = {"conv3d": {}, "attention": {}}

    # B-1. 3D conv 스택 (해상도·채널별 활성값 비용)
    for down in args.downsamples:
        h, w = H // down, W // down
        for ch in args.channels:
            key = f"1over{down}_c{ch}"
            try:
                net = torch.nn.Sequential(
                    torch.nn.Conv3d(3, ch, 3, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(ch, ch, 3, padding=1),
                    torch.nn.SiLU(),
                    torch.nn.Conv3d(ch, ch, 3, padding=1),
                ).to(dev)
                x = torch.randn(1, 3, T, h, w, device=dev, requires_grad=True)
                out["conv3d"][key] = {
                    "hw": [h, w],
                    **bench(lambda z: net(z), x),
                }
                del net, x
            except torch.cuda.OutOfMemoryError:
                out["conv3d"][key] = {"hw": [h, w], "oom": True}
            _reset_peak()

    # B-2. spatiotemporal self-attention (다운샘플 배율별 토큰 수와 실제 메모리)
    for down in args.attn_downsamples:
        h, w = H // down, W // down
        n_tokens = T * h * w
        key = f"1over{down}"
        info = {"hw": [h, w], "tokens": n_tokens, "n_squared": n_tokens**2}
        try:
            ch = 128
            q = torch.randn(1, 8, n_tokens, ch // 8, device=dev, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)

            def attn(_z):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

            info.update(bench(attn, q, iters=2))
            del q, k, v
        except torch.cuda.OutOfMemoryError:
            info["oom"] = True
        except RuntimeError as exc:
            info["error"] = str(exc)[:200]
        out["attention"][key] = info
        _reset_peak()

    return out


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results" / "branchC" / "budget.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--dino-subsample", type=int, default=4)
    ap.add_argument("--downsamples", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--attn-downsamples", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--skip-backbone", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA 가 필요하다"
    dev = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info()

    report = {
        "device": torch.cuda.get_device_name(0),
        "vram_total_gib": round(total / 2**30, 2),
        "vram_free_at_start_gib": round(free / 2**30, 2),
        "torch": torch.__version__,
        "scoring_space": {"T": T, "H": H, "W": W, "range": "[-1,1]"},
        "note": (
            "재현 검증 기준 장비는 RTX PRO 6000 96GB(rule/001 §3.3). "
            "5090 32GB 를 통과하면 재현 검증은 자동 통과이므로 이 표는 보수적이다."
        ),
    }

    print("=" * 72)
    print(f"{report['device']}  VRAM {report['vram_total_gib']} GiB (여유 {report['vram_free_at_start_gib']})")
    print("=" * 72)

    print("\n[A] 채점기 3종을 손실로 쓸 때의 비용")
    report["scorers"] = measure_scorers(args, dev)
    print(json.dumps(report["scorers"]["weights_gib"], ensure_ascii=False))
    print(json.dumps(report["scorers"]["sanity"], ensure_ascii=False, indent=1))
    for b, entry in report["scorers"]["per_scorer"].items():
        print(f"  {b}: " + json.dumps(entry, ensure_ascii=False))

    if not args.skip_backbone:
        print("\n[B] 백본 여지")
        report["backbone"] = measure_backbone_room(args, dev)
        for k, v in report["backbone"]["conv3d"].items():
            print(f"  conv3d {k}: {json.dumps(v, ensure_ascii=False)}")
        for k, v in report["backbone"]["attention"].items():
            print(f"  attn   {k}: {json.dumps(v, ensure_ascii=False)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
