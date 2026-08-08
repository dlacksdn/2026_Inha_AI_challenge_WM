#!/usr/bin/env python
"""
branch C 모델 — SimVP-TAU + FiLM 행동 주입 + 잔차 출력

004 §2.3 의 "고칠 곳 8개" 중 구조에 해당하는 것을 구현한다.
  ①   첫 프레임을 16번 복제해 넣는다 (안 하면 경고 없이 자기회귀로 빠진다)
  ①-b 단 **인코더는 1번만** 돌리고 복제한다. 수학적 동일성은 007 에서 오차 0 으로 검증
  ②   행동을 FiLM 으로 3곳에 주입
  ③   출력 = 첫 프레임 + 잔차, 마지막 층 zero-init (하방 봉쇄. 007 에서 오차 0 으로 검증)
  ④   N_S=2 (잔차 유효 해상도 1/2 = 160×256)
       ⚠ 004 는 N_S=4~6 이었다. 007 §4 가 뒤집었다 —
         목표(eval DV 40.4% 감축)에 1/4 는 오라클로도 16.1%, 1/8 은 −1.8% 다.
         OpenSTL 공식 config 도 로봇(bair)·실사(kitticaltech)는 전부 N_S=2 다

FiLM 주입 지점 [코드] simvp_model.py 구조상
  ① 인코더 뒤(병목 입력) (B,T,C,H,W)  → 프레임별·채널별.  act_t 로부터
  ② TAU 블록 사이        (B,256,H,W)  → 채널별.  행동 시퀀스 요약으로부터
     (번역기는 T 를 채널에 접어 넣어 (B,T*C,H,W) 로 다루므로 중간에서 프레임별이 불가능하다)
  ③ 디코더 각 블록 뒤     (B*T,64,H,W) → 프레임별·채널별.  act_t 로부터

행동 표현 = **델타** (③ probe_action_signal 이 무승부 → 사전 등록 기본값으로 확정)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")
OPENSTL = REPO / "third_party" / "OpenSTL"
sys.path.insert(0, str(OPENSTL))
_spec = importlib.util.spec_from_file_location(
    "_simvp_c", str(OPENSTL / "openstl" / "models" / "simvp_model.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
SimVP_Model = _m.SimVP_Model

T, C, H, W = 16, 3, 320, 512


def act_delta(a: torch.Tensor) -> torch.Tensor:
    """(B,T,6) 정규화 행동 → 프레임간 차분. 0번 프레임은 0."""
    d = torch.zeros_like(a)
    d[:, 1:] = a[:, 1:] - a[:, :-1]
    return d


class FiLMFrame(nn.Module):
    """프레임별·채널별 변조. zero-init 이라 학습 0스텝에서 항등이다."""

    def __init__(self, act_dim: int, ch: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(act_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * ch))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, a):
        """x (B,T,ch,H,W) 또는 (B*T,ch,H,W) / a (B,T,act_dim)"""
        g, b = self.mlp(a).chunk(2, -1)                     # (B,T,ch)
        if x.dim() == 5:
            return x * (1 + g[..., None, None]) + b[..., None, None]
        g = g.reshape(-1, g.shape[-1])[..., None, None]
        b = b.reshape(-1, b.shape[-1])[..., None, None]
        return x * (1 + g) + b


class FiLMGlobal(nn.Module):
    """표본별·채널별 변조. 번역기 중간용(프레임 축이 채널에 접혀 있다)."""

    def __init__(self, act_dim: int, ch: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(act_dim * T, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * ch))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, a):                                # x (B,ch,H,W)
        g, b = self.mlp(a.flatten(1)).chunk(2, -1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


class ResidualSimVPC(nn.Module):
    def __init__(self, hid_S=64, hid_T=256, N_S=2, N_T=8, act_dim=6,
                 use_ckpt=True, film_trans=True, film_dec=True):
        super().__init__()
        self.net = SimVP_Model(in_shape=(T, C, H, W), hid_S=hid_S, hid_T=hid_T,
                               N_S=N_S, N_T=N_T, model_type="tau",
                               spatio_kernel_enc=3, spatio_kernel_dec=3)
        nn.init.zeros_(self.net.dec.readout.weight)
        nn.init.zeros_(self.net.dec.readout.bias)          # ③ 하방 봉쇄

        self.film_in = FiLMFrame(act_dim, hid_S)           # ① 인코더 뒤
        self.film_t = nn.ModuleList(
            [FiLMGlobal(act_dim, hid_T) for _ in range(N_T - 1)]) if film_trans else None
        self.film_d = nn.ModuleList(
            [FiLMFrame(act_dim, hid_S) for _ in self.net.dec.dec]) if film_dec else None
        self.use_ckpt = use_ckpt
        self.N_T = N_T
        self.hid_S = hid_S

    # ── 부품 ────────────────────────────────────────────────
    def _ck(self, fn, *a):
        if self.use_ckpt and self.training:
            return checkpoint(fn, *a, use_reentrant=False)
        return fn(*a)

    def _translator(self, z, a):
        """MidMetaNet 을 블록 단위로 돌면서 사이에 FiLM 을 넣는다."""
        mid = self.net.hid
        B, Tn, Cn, Hn, Wn = z.shape
        x = z.reshape(B, Tn * Cn, Hn, Wn)
        for i in range(mid.N2):
            x = mid.enc[i](x)
            if self.film_t is not None and i < len(self.film_t):
                x = self.film_t[i](x, a)
        return x.reshape(B, Tn, Cn, Hn, Wn)

    def _decode(self, hid, skip, a):
        """Decoder.forward 를 펼치고 각 블록 뒤에 FiLM 을 넣는다."""
        d = self.net.dec
        for i in range(len(d.dec) - 1):
            hid = d.dec[i](hid)
            if self.film_d is not None:
                hid = self.film_d[i](hid, a)
        hid = d.dec[-1](hid + skip)
        if self.film_d is not None:
            hid = self.film_d[-1](hid, a)
        return d.readout(hid)

    # ── 순전파 ──────────────────────────────────────────────
    def forward(self, first, act, return_residual=False):
        """first (B,3,H,W) in [-1,1] · act (B,T,6) 정규화 공간 → (B,3,T,H,W)"""
        B = first.shape[0]
        a = act_delta(act)
        embed, skip = self._ck(self.net.enc, first)        # ①-b 인코더 1회
        _, C_, H_, W_ = embed.shape
        z = embed.unsqueeze(1).expand(B, T, C_, H_, W_)    # ① 16복제
        z = self.film_in(z, a)                             # FiLM ①
        z = self._ck(self._translator, z.contiguous(), a)  # FiLM ②
        hid = z.reshape(B * T, C_, H_, W_)
        skip = skip.repeat_interleave(T, 0)
        res = self._ck(self._decode, hid, skip, a)         # FiLM ③
        res = res.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        out = first.unsqueeze(2) + res                     # ③ 잔차 덧셈
        return (out, res) if return_residual else out


# ───────────────────────── 자체 검증 ─────────────────────────

def _selftest():
    dev = "cuda"
    torch.manual_seed(0)
    m = ResidualSimVPC(use_ckpt=False).to(dev).eval()
    first = torch.randn(1, C, H, W, device=dev)
    act = torch.randn(1, T, 6, device=dev)

    print("=== 1. zero-init 하방 봉쇄 (③) ===")
    with torch.no_grad():
        out, res = m(first, act, return_residual=True)
    static = first.unsqueeze(2).expand(1, C, T, H, W)
    print(f"   잔차 최대 |값|        {res.abs().max().item():.3e}   ← 0 이어야 한다")
    print(f"   출력 − 정지영상 최대   {(out - static).abs().max().item():.3e}   ← 0 이어야 한다")
    assert res.abs().max().item() == 0.0

    print("\n=== 2. FiLM 이 실제로 출력을 바꾸는가 (주입 배선 확인) ===")
    # readout 을 깨워야 잔차가 0 이 아니게 된다
    nn.init.normal_(m.net.dec.readout.weight, std=0.02)
    for f in [m.film_in] + list(m.film_t) + list(m.film_d):
        nn.init.normal_(f.mlp[-1].weight, std=0.02)
    with torch.no_grad():
        o1 = m(first, act)
        o2 = m(first, act * 3.0)
        o3 = m(first, torch.zeros_like(act))
    print(f"   |out(act) − out(3·act)|   {(o1 - o2).abs().mean().item():.3e}  ← 0 이면 배선 끊김")
    print(f"   |out(act) − out(0)|       {(o1 - o3).abs().mean().item():.3e}  ← 0 이면 배선 끊김")
    assert (o1 - o2).abs().mean().item() > 1e-6, "FiLM 이 출력에 영향을 못 준다"

    print("\n=== 3. 주입 지점별 기여 (하나씩 끄고 본다) ===")
    for name, kw in [("① 병목만", dict(film_trans=False, film_dec=False)),
                     ("①+②", dict(film_trans=True, film_dec=False)),
                     ("①+②+③", dict(film_trans=True, film_dec=True))]:
        torch.manual_seed(0)
        mm = ResidualSimVPC(use_ckpt=False, **kw).to(dev).eval()
        nn.init.normal_(mm.net.dec.readout.weight, std=0.02)
        mods = [mm.film_in] + (list(mm.film_t) if mm.film_t else []) + \
               (list(mm.film_d) if mm.film_d else [])
        for f in mods:
            nn.init.normal_(f.mlp[-1].weight, std=0.02)
        with torch.no_grad():
            d = (mm(first, act) - mm(first, torch.zeros_like(act))).abs().mean().item()
        n = sum(p.numel() for p in mm.parameters()) / 1e6
        print(f"   {name:10s} 행동 민감도 {d:.4e}   파라미터 {n:.2f}M")

    print("\n=== 4. 학습 1스텝 (메모리·시간) ===")
    m2 = ResidualSimVPC(use_ckpt=True).to(dev).train()
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-4)
    fb = torch.randn(2, C, H, W, device=dev); ab = torch.randn(2, T, 6, device=dev)
    tgt = torch.randn(2, C, T, H, W, device=dev)
    torch.cuda.reset_peak_memory_stats()
    import time
    for i in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (m2(fb, ab) - tgt).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (m2(fb, ab) - tgt).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1000
    gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"   배치 2 · bf16 · ckpt:  {ms:.1f} ms/step   피크 {gib:.2f} GiB")
    print(f"   (007 §5 의 FiLM 미포함 실측: 551.4 ms · 20.22 GiB)")
    print(f"   유효배치 16(누적 8) → 30k 스텝 {ms*8*30000/1000/3600:.1f} 시간")


if __name__ == "__main__":
    _selftest()
