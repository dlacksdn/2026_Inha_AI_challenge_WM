#!/usr/bin/env python
"""
RTX PRO 6000 (96GB) 에서 유효배치 16 을 만드는 조합의 s/step 을 잰다.

왜 이걸 재나 — 010 §6:
  "처음 20 스텝의 s/step 을 재고 결정하라. 지금의 4.7 초/스텝은 use_ckpt=True 하드코딩 +
   micro_batch 2, 즉 32GB 에서 가능한 가장 느린 조합의 값이다.
   여기서 나오는 배율이 남은 시간 전체에 곱해진다 — 다른 어떤 항목도 이만한 레버가 없다."

⚠ 유효배치 16 을 고정한 채로만 비교한다. micro × accum = 16 인 조합만 격자에 넣는다.
  곱이 달라지면 수학이 달라져 짝지은 비교(010 §6-2)가 깨진다.

⚠ 이 기계는 Max-Q(전력 제한) 판이라 클럭에서 오는 이득이 없다 [측정 2026-08-09]:
  micro 2·ckpt on 에서 596.1 ms/micro-step vs 5090 의 551.4 ms(FiLM 미포함, 007 §5).
  ⇒ 이득은 오직 용량(96GB)에서만 나온다. 배치를 키우고 checkpointing 을 끄는 쪽이다.

🚨 GPU 0 만 쓴다. 반드시 CUDA_VISIBLE_DEVICES=0 으로 실행한다 (CLAUDE.md).

사용:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/branchC/measure_step_6000.py
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "branchC"))
from model_c import ResidualSimVPC, C, H, W, T                       # noqa: E402

EFFECTIVE_BATCH = 16          # 고정. 이걸 바꾸면 재개 짝이 깨진다
WARMUP_STEPS = 3              # 옵티마이저 스텝 기준
MEASURE_STEPS = 5


def bench(micro: int, accum: int, use_ckpt: bool, dev: str = "cuda") -> dict:
    """옵티마이저 스텝 1회(= micro × accum 샘플)의 벽시계와 피크 메모리."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    model = ResidualSimVPC(use_ckpt=use_ckpt).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    first = torch.randn(micro, C, H, W, device=dev)
    act = torch.randn(micro, T, 6, device=dev)
    tgt = torch.randn(micro, C, T, H, W, device=dev)

    def one_opt_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = (model(first, act) - tgt).abs().mean()
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    try:
        for _ in range(WARMUP_STEPS):
            one_opt_step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(MEASURE_STEPS):
            one_opt_step()
        torch.cuda.synchronize()
        s_per_step = (time.perf_counter() - t0) / MEASURE_STEPS
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        out = {"micro": micro, "accum": accum, "ckpt": use_ckpt,
               "s_per_step": s_per_step, "peak_gib": peak, "oom": False}
    except torch.cuda.OutOfMemoryError:
        out = {"micro": micro, "accum": accum, "ckpt": use_ckpt,
               "s_per_step": None, "peak_gib": None, "oom": True, "fail": "OOM"}
    except RuntimeError as e:
        # micro ≥ 7 은 OpenSTL MixMlp 의 depthwise conv 에서 32비트 인덱스가 넘친다.
        # 메모리와 무관하다 — 96GB 여도 동일하다 (010 §6, 이 세션이 독립 재현).
        msg = "32bit-index" if "canUse32BitIndexMath" in str(e) else str(e)[:80]
        out = {"micro": micro, "accum": accum, "ckpt": use_ckpt,
               "s_per_step": None, "peak_gib": None, "oom": True, "fail": msg}
    del model, opt, first, act, tgt
    torch.cuda.empty_cache()
    return out


def main() -> None:
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[gpu] {name}  {total:.1f} GiB  (보이는 카드 {torch.cuda.device_count()}장)")
    if torch.cuda.device_count() != 1:
        print("  ⚠ CUDA_VISIBLE_DEVICES=0 로 실행하지 않았다. GPU 0 만 써야 한다 (CLAUDE.md)")
    print(f"[격자] 유효배치 {EFFECTIVE_BATCH} 고정 · bf16 · {H}×{W} · hid_S=64 · N_S=2\n")

    grid = []
    for micro in (2, 4, 8, 16):
        accum = EFFECTIVE_BATCH // micro
        for use_ckpt in (True, False):
            grid.append((micro, accum, use_ckpt))

    rows = []
    base = None
    for micro, accum, use_ckpt in grid:
        r = bench(micro, accum, use_ckpt, dev)
        if r["oom"]:
            print(f"  micro {micro:2d} × accum {accum:2d} · ckpt {str(use_ckpt):5s}   "
                  f"실패: {r['fail']}")
        else:
            if base is None:
                base = r["s_per_step"]        # micro2·ckpt on = 기존 5090 좌표
            print(f"  micro {micro:2d} × accum {accum:2d} · ckpt {str(use_ckpt):5s}   "
                  f"{r['s_per_step']:6.2f} s/step   피크 {r['peak_gib']:6.2f} GiB   "
                  f"×{base / r['s_per_step']:.2f}")
        rows.append(r)

    ok = [r for r in rows if not r["oom"]]
    best = min(ok, key=lambda r: r["s_per_step"]) if ok else None
    print()
    if best:
        speedup = base / best["s_per_step"]
        rem = 6000 - 2500                     # full_002500 에서 G2 지점(6,000)까지
        print(f"[최선] micro {best['micro']} × accum {best['accum']} · "
              f"ckpt {best['ckpt']} → {best['s_per_step']:.2f} s/step "
              f"(기준 대비 ×{speedup:.2f})")
        print(f"  재개 {rem} 스텝    → {best['s_per_step'] * rem / 3600:.2f} 시간/팔")
        print(f"  두 팔 순차        → {best['s_per_step'] * rem * 2 / 3600:.2f} 시간")
        print(f"  30k 풀런          → {best['s_per_step'] * 30000 / 3600:.1f} 시간 "
              f"(대회 재현 상한 96시간)")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = REPO / "results" / "branchC"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"step_budget_6000_{ts}.json"
    json.dump({"gpu": name, "total_gib": total, "effective_batch": EFFECTIVE_BATCH,
               "resolution": [H, W], "rows": rows, "best": best},
              open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n[out] {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
