#!/usr/bin/env python
"""
④ 예산 재측정 — SimVP-TAU 격자 (005 §0 을 우리 환경에서 다시 재는 것)

왜 다시 재나 (007 판단):
  - 005 §0 은 torch 2.13(= ~/anaconda3, 남의 환경)에서 쟀고 로그·스크립트가 없다 → [미검증]
  - 005 격자에 **인코더1회 축**(004 ①-b)이 빠져 있었다. 메모리 답을 가장 크게 바꿀 변경이다
  - 005 격자에 **N_S=2**(잔차 유효 해상도 1/2)가 없었다. 목표 산술이 요구하는 지점이다

판정선은 측정 전에 등록했다 (007 §0순위):
  R1  1/2 해상도(N_S=2)가 "유효배치 ≥8(누적 허용) & 30k 옵티마이저 스텝 ≤ 4일" 이면 채택
  R2  실패 시 인코더1회+bf16+ckpt 전부 켜고 재판정
  R3  그래도 실패면 "1/4 는 오라클로도 목표 미달"을 명문화하고 사용자 판단에 올린다
  R4  경계선이면 해상도가 높은 쪽

측정 대상: 스텝 시간(fwd+L1+backward+AdamW) · 피크 VRAM · OOM 여부
비고: FiLM·실데이터로더 미포함(별도 단계). 이 표는 백본 상한이다.
"""
import argparse, json, os, sys, time, traceback
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

REPO = "/home/rils/dlacksdn/2026_Inha_AI_challenge_WM"
OPENSTL = os.path.join(REPO, "third_party", "OpenSTL")
sys.path.insert(0, OPENSTL)

# openstl.models.__init__ 이 matplotlib 등 무거운 것을 끌어온다.
# 우리는 SimVP 본체만 필요하므로 파일을 직접 로드한다 (openstl.modules 는 가볍고 정상 import 된다).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_simvp_model", os.path.join(OPENSTL, "openstl", "models", "simvp_model.py"))
_simvp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_simvp)
SimVP_Model = _simvp.SimVP_Model

T, C, H, W = 16, 3, 320, 512


class ResidualSimVP(nn.Module):
    """004 §2.3 의 잔차 구조를 반영한 래퍼.

    ③ 출력 = 첫 프레임 + 잔차, readout zero-init  (하방 봉쇄)
    ①-b enc_once=True 면 인코더를 1회만 돌리고 결과를 복제한다.
        입력이 첫 프레임 16복제이므로 수학적으로 동일하다.
    """

    def __init__(self, hid_S, hid_T, N_S, N_T, enc_once, use_ckpt):
        super().__init__()
        self.net = SimVP_Model(in_shape=(T, C, H, W), hid_S=hid_S, hid_T=hid_T,
                               N_S=N_S, N_T=N_T, model_type="tau",
                               spatio_kernel_enc=3, spatio_kernel_dec=3)
        nn.init.zeros_(self.net.dec.readout.weight)
        nn.init.zeros_(self.net.dec.readout.bias)
        self.enc_once = enc_once
        self.use_ckpt = use_ckpt

    def _run(self, fn, *a):
        if self.use_ckpt and self.training:
            return checkpoint(fn, *a, use_reentrant=False)
        return fn(*a)

    def forward(self, first):            # first: (B, C, H, W)
        B = first.shape[0]
        net = self.net
        if self.enc_once:
            embed, skip = self._run(net.enc, first)          # B 장만 인코딩
            embed = embed.repeat_interleave(T, dim=0)
            skip = skip.repeat_interleave(T, dim=0)
        else:
            x = first.unsqueeze(1).expand(B, T, C, H, W).reshape(B * T, C, H, W)
            embed, skip = self._run(net.enc, x)              # B*T 장 인코딩 (원본)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid = self._run(net.hid, z).reshape(B * T, C_, H_, W_)
        res = self._run(net.dec, hid, skip).reshape(B, T, C, H, W)
        return first.unsqueeze(1) + res                       # 하방 봉쇄


def bottleneck_hw(N_S):
    d = 2 ** (N_S // 2)
    return H // d, W // d


def measure(cfg, batch, dtype, steps=10, warmup=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dev = "cuda"
    model = ResidualSimVP(cfg["hid_S"], cfg["hid_T"], cfg["N_S"], cfg["N_T"],
                          cfg["enc_once"], cfg["ckpt"]).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    first = torch.randn(batch, C, H, W, device=dev)
    target = torch.randn(batch, T, C, H, W, device=dev)
    amp = dtype is not torch.float32

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=amp):
            out = model(first)
            loss = (out - target).abs().mean()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps * 1000
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    nparam = sum(p.numel() for p in model.parameters())
    del model, opt, first, target
    torch.cuda.empty_cache()
    return {"ms_per_step": round(ms, 1), "peak_gib": round(peak, 2), "params_M": round(nparam / 1e6, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    print(f"[env] torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"[env] device {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")
    print(f"[env] python {sys.executable}")
    print(f"[coord] in_shape T={T} C={C} H={H} W={W}  (첫 프레임 16복제 입력)")
    print()

    grid = []
    for N_S in (2, 4, 6):
        for hid_S in (32, 64):
            for enc_once in (False, True):
                for ck in (False, True):
                    grid.append({"N_S": N_S, "hid_S": hid_S, "hid_T": 256, "N_T": 8,
                                 "enc_once": enc_once, "ckpt": ck})

    results = []
    for cfg in grid:
        bh, bw = bottleneck_hw(cfg["N_S"])
        tag = (f"N_S={cfg['N_S']}(1/{2**(cfg['N_S']//2)}={bh}x{bw}) hid_S={cfg['hid_S']} "
               f"enc_once={int(cfg['enc_once'])} ckpt={int(cfg['ckpt'])}")
        row = {**cfg, "bottleneck": f"{bh}x{bw}",
               "res_frac": f"1/{2**(cfg['N_S']//2)}", "bf16": {}, "fp32": {}}
        for batch in args.batches:
            try:
                r = measure(cfg, batch, torch.bfloat16, steps=args.steps)
                row["bf16"][str(batch)] = r
                print(f"  {tag} bf16 b{batch}: {r['ms_per_step']} ms  {r['peak_gib']} GiB  "
                      f"({r['params_M']}M params)", flush=True)
            except torch.cuda.OutOfMemoryError:
                row["bf16"][str(batch)] = "OOM"
                print(f"  {tag} bf16 b{batch}: OOM", flush=True)
                torch.cuda.empty_cache()
                break
            except Exception as e:
                row["bf16"][str(batch)] = f"ERROR: {e}"
                print(f"  {tag} bf16 b{batch}: ERROR {e}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()
                break
        results.append(row)

    # bf16 절감을 따로 잰다 — 005 §0 의 "bf16 절감 10%뿐"이 이상값이라 두 점에서 확인
    print("\n[fp32 대조] bf16 절감률 검증 (005 §0 의 '10%뿐' 재검)")
    for cfg in [{"N_S": 2, "hid_S": 64, "hid_T": 256, "N_T": 8, "enc_once": True, "ckpt": False},
                {"N_S": 4, "hid_S": 64, "hid_T": 256, "N_T": 8, "enc_once": True, "ckpt": False}]:
        for batch in (1, 2):
            try:
                r = measure(cfg, batch, torch.float32, steps=args.steps)
                key = f"N_S{cfg['N_S']}_b{batch}"
                results.append({**cfg, "fp32_ref": key, "fp32": r})
                print(f"  fp32 N_S={cfg['N_S']} b{batch}: {r['ms_per_step']} ms  {r['peak_gib']} GiB",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  fp32 N_S={cfg['N_S']} b{batch}: OOM", flush=True)
                torch.cuda.empty_cache()

    meta = {"generated": datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": sys.executable,
            "device": torch.cuda.get_device_name(0),
            "in_shape": [T, C, H, W],
            "note": "FiLM·실데이터로더 미포함. 백본 상한. 판정선은 007 §0순위에 사전 등록됨.",
            "rows": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
