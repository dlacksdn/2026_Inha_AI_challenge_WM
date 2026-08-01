"""워핑 오라클 — C 설계의 첫 갈림길(덧셈 잔차 vs 워핑)을 우리 데이터로 판정한다.

무엇을 묻는가
-------------
016 §9.2 에서 "정지영상을 정답 쪽으로 미는" 잔차 보정의 천장·바닥을 쟀다.

    뭉갠 잔차 4배 α=1   static 대비 −0.01690   ← 현실적 이득
    남의 잔차 α=0.25    static 대비 +0.01700   ← 현실적 손해

이득의 출처는 DINO 가 아니라 Video 였다(뭉개면 장면은 더 틀려 보이지만 결은 맞는다).

    DINO   0.3 × (0.15664 − 0.12308) = +0.01007   손해
    Video  0.3 × (0.02297 − 0.09110) = −0.02044   이득
    Action 0.4 × (1.22386 − 1.24017) = −0.00652   이득

그래서 다음 질문이 나온다. **덧셈의 DINO 손해는 "잔차를 더하면 뭉개진다"에서 온다.
픽셀을 옮기기만 하면(워핑) 그 손해가 없어지지 않을까?**

    덧셈 잔차   출력 = 첫프레임 + 잔차
        단점: 모델이 만드는 잔차는 저주파라 디테일이 뭉개진다 (실측 DINO +0.010)

    워핑        출력 = 첫프레임의 픽셀을 흐름대로 밀어 옮긴 것
        장점: 원본 픽셀을 그대로 옮기므로 디테일이 안 뭉개진다
        단점: 가려졌다 드러나는 자리(팔이 비켜난 뒤의 배경)를 못 채운다

어느 단점이 큰지는 **로봇 팔이 화면에서 차지하는 면적과 배경의 복잡도**에 달렸고,
그건 우리 데이터에만 있는 정보다. 문헌은 일반론만 준다. 그래서 직접 잰다.

어떻게 재는가
-------------
정답 영상에서 광학 흐름(RAFT)을 뽑아 첫 프레임을 밀어 옮긴 "워핑 오라클"을 만들고,
**같은 표본·같은 채점기**로 덧셈 오라클과 나란히 잰다.

    흐름 f_t : 정답 프레임 t 의 각 픽셀이 프레임 0 의 어디에서 왔는가 (역방향 흐름)
    워핑     : pred_t[p] = static_0[ p + α · f_t(p) ]        (grid_sample, 역방향 샘플링)

    α = 0  →  정확히 static (새너티 체크)
    α = 1  →  흐름대로 완전히 밀었다

덧셈 오라클의 α 와 뜻이 다르다는 점에 주의하라.
덧셈은 "정답 쪽으로 α 만큼 섞었다", 워핑은 "흐름의 α 배만큼 밀었다"이다.
그래서 덧셈 α=1 은 **정확히 정답**이지만 워핑 α=1 은 정답이 아니다
(흐름 추정 오차 + 가림/드러남 때문). 이 비대칭이 곧 Gate 0 의 근거다.

변형 목록
---------
  [워핑 천장]  warp:α         원본 해상도 흐름. α ∈ {0.25, 0.5, 1}
  [워핑 현실]  warpc4:α       흐름을 4배로 뭉갬(면적축소→쌍선형확대). α ∈ {0.6, 1}
               warpc8:1       흐름을 8배로 뭉갬
  [워핑 바닥]  warpnbr:α      **다른 표본의 흐름**으로 민다. α ∈ {0.25, 0.5, 1}
  [덧셈 재측정] add:1         = 정답 (DINO·Video 가 0 이어야 한다 — 새너티)
               addblur4:1     016 §9.2 의 −0.01690 재현
               addblur8:1
               addnbr:0.25
  [기준선]     static         α=0
  [혼합 탐색]  warpc4+res4:1  뭉갠 워핑 + 남은 오차의 뭉갠 덧셈 보정

**흐름을 뭉개는 것이 잔차를 뭉개는 것의 올바른 짝**이다. 학습된 모델은 어느 쪽이든
저주파만 만든다. 축소배율을 같은 4배로 맞췄으므로 같은 급의 비교다.

왜 표본별 rows 를 저장하는가
----------------------------
016 이 쓴 probe_residual_headroom.py 는 **평균만** 저장했다. 그러면 "워핑이 덧셈보다
좋다"를 짝지은 t 검정으로 확인할 수 없다. 여기서는 표본별 (dino, video, action) 을
전부 남겨 두 변형을 **같은 표본끼리 짝지어** 비교한다.

규칙에 대하여
-------------
전부 train 홀드아웃에서만 한다. eval 데이터는 쓰지 않는다.
정답 영상에서 흐름을 뽑으므로 이것은 **오라클**이다 — 제출 방법이 아니라 여지를 재는 측정이다.

한계 (반드시 같이 읽어라)
-------------------------
1. 오라클은 **반증에는 쓸 수 있고 검증에는 못 쓴다.** 워핑이 이겨도 "워핑 모델이
   된다"는 뜻이 아니라 "둘 중 어느 쪽을 만들지" 만 정한다.
2. 로컬 홀드아웃 결과다. 016 §5 에서 로컬과 eval 이 **방향까지** 어긋난 전례를 확인했다.
3. 흐름 추정기 품질이 교란요인이다. RAFT-large 가 우리 데이터에서 나쁘면 워핑을
   억울하게 죽일 수 있다. 그래서 워핑 결과의 픽셀 L1 오차를 진단값으로 같이 찍는다.
   `--flow farneback` 으로 다른 추정기 교차확인이 가능하다.

사용:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/branchC/probe_warp_vs_add.py \
      --static artifacts/branchB/m0_step1000_b4/static_preds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 감싸는 스크립트가 아니라 **필요한 스크립트 본체**에 둔다 (016 §4.1 교훈).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 홈 캐시(~/.cache)를 건드리지 않는다 — 공용 컴퓨터다. 가중치는 프로젝트 안에 둔다.
REPO = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("TORCH_HOME", str(REPO / "artifacts/torch_home"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/branchB"))
from wm_eval import scoring as S  # noqa: E402
from probe_residual_headroom import blur_residual, nearest_by_action  # noqa: E402


# ---------------------------------------------------------------- 흐름 추정
class RaftFlow:
    """RAFT-large. 정답 프레임 t → 프레임 0 의 역방향 흐름을 준다.

    역방향인 이유: 워핑을 grid_sample(역방향 샘플링)로 하기 때문이다.
    "프레임 t 의 픽셀 p 는 프레임 0 의 어디에서 왔나"를 알아야 p 자리를 채울 수 있다.
    순방향(splatting)은 구멍이 생겨 별도 처리가 필요하다 — 여기서는 쓰지 않는다.
    """

    name = "raft_large_C_T_SKHT_V2"

    def __init__(self, device: torch.device, chunk: int = 5):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        self.w = Raft_Large_Weights.C_T_SKHT_V2
        self.model = raft_large(weights=self.w, progress=False).to(device).eval()
        self.tf = self.w.transforms()
        self.device = device
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """(T,3,H,W) uint8 → (T,2,H,W) float32. t=0 은 0 흐름."""
        t, _, h, w = frames_u8.shape
        ref = frames_u8[0:1]
        out = torch.zeros(t, 2, h, w, dtype=torch.float32)
        for s in range(1, t, self.chunk):
            e = min(s + self.chunk, t)
            img1, img2 = self.tf(frames_u8[s:e].to(self.device).contiguous(),
                                 ref.expand(e - s, -1, -1, -1).to(self.device).contiguous())
            flow = self.model(img1, img2)[-1]          # 마지막 갱신이 최종 흐름
            out[s:e] = flow.float().cpu()
        return out


class FarnebackFlow:
    """cv2 Farneback — RAFT 결론이 추정기에 의존하는지 확인하는 교차검증용(CPU)."""

    name = "farneback"

    def __init__(self, device: torch.device, chunk: int = 0):
        import cv2
        self.cv2 = cv2

    def __call__(self, frames_u8: torch.Tensor) -> torch.Tensor:
        cv2 = self.cv2
        t, _, h, w = frames_u8.shape
        g = [cv2.cvtColor(frames_u8[i].permute(1, 2, 0).contiguous().numpy(),
                          cv2.COLOR_RGB2GRAY) for i in range(t)]
        out = torch.zeros(t, 2, h, w, dtype=torch.float32)
        for i in range(1, t):
            f = cv2.calcOpticalFlowFarneback(g[i], g[0], None,
                                             0.5, 4, 21, 5, 7, 1.5, 0)
            out[i] = torch.from_numpy(f).permute(2, 0, 1)
        return out


# ---------------------------------------------------------------- 워핑
def _base_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32),
                            indexing="ij")
    return torch.stack([xs, ys], dim=0)                # (2,H,W) 픽셀 좌표


def warp(first_u8: torch.Tensor, flow: torch.Tensor, alpha: float,
         mode: str = "bilinear") -> torch.Tensor:
    """첫 프레임을 흐름의 α 배만큼 밀어 옮긴 영상을 만든다.

    first_u8 : (3,H,W) float  — static 의 프레임 0 (모델이 입력으로 받는 그림)
    flow     : (T,2,H,W)      — 프레임 t → 프레임 0 역방향 흐름
    반환      : (T,3,H,W) float
    """
    t, _, h, w = flow.shape
    dev = flow.device
    src = _base_grid(h, w, dev).unsqueeze(0) + alpha * flow          # (T,2,H,W)
    gx = src[:, 0] / max(w - 1, 1) * 2.0 - 1.0
    gy = src[:, 1] / max(h - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)                             # (T,H,W,2)
    src_img = first_u8.unsqueeze(0).expand(t, -1, -1, -1)
    # padding_mode='border': 화면 밖을 참조하면 가장자리를 복제한다.
    # 채점 영상은 이미 좌우(또는 상하)에 검은 띠가 있어 가장자리가 검정이므로
    # 'zeros' 와 사실상 같지만, 띠가 없는 축에서 검정이 새어드는 것을 막는다.
    # ⚠ mode="bilinear" 은 변위가 정수가 아닌 **모든 화소**에 저역통과를 건다.
    #   즉 "워핑은 원본 픽셀을 그대로 옮기므로 디테일이 안 뭉개진다"는 이 구현에서 거짓이다.
    #   mode="nearest" 로 바꿔 재면 그 보간 흐림분을 분리할 수 있다(018 §9-c D).
    return F.grid_sample(src_img, grid, mode=mode,
                         padding_mode="border", align_corners=True)


def coarsen_flow(flow: torch.Tensor, k: int) -> torch.Tensor:
    """흐름장의 고주파를 날린다 — 학습된 모델이 만드는 뭉갠 흐름의 대리값.

    잔차를 k 배 축소했다 되키우는 blur_residual 과 **같은 연산·같은 배율**이다.
    면적평균 후 쌍선형확대이므로 흐름의 크기(픽셀 변위)는 보존된다.
    """
    t, c, h, w = flow.shape
    small = F.interpolate(flow, size=(max(h // k, 1), max(w // k, 1)), mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------- 통계
def paired_t(a: list[float], b: list[float]) -> dict:
    """같은 표본끼리 짝지은 차이 a−b 의 평균·표준오차·t·승률."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else float("inf")
    return {"n": n, "delta": mean, "se": se,
            "t": (mean / se) if se > 0 else 0.0,
            "wins": int((d < 0).sum())}


def totals(rows: list[dict]) -> list[float]:
    return [S.weighted_total(r["dino"], r["video"], r["action"]) for r in rows]


# ---------------------------------------------------------------- 본체
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats",
                    default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--static",
                    default=str(REPO / "artifacts/branchB/m0_step1000_b4/static_preds"))
    ap.add_argument("--flow", choices=["raft", "farneback"], default="raft")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", type=int, default=0,
                    help="앞의 N개 표본을 눈으로 확인할 mp4 로 남긴다(흐름 방향 오류 육안 검증용)")
    ap.add_argument("--dump-dir", default=str(REPO / "artifacts/branchC/warp_preview"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "results/branchC/warp_vs_add.json"))
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    sids = [m["sid"] for m in samples]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- 변형 목록 (판정선은 결과를 보기 전에 고정했다 — README/문서 참조) ---
    VARIANTS = [
        "static",                                        # α=0 기준선
        "warp:0.25", "warp:0.5", "warp:1",               # 워핑 천장(원본 해상도 흐름)
        "warpc4:0.6", "warpc4:1", "warpc8:1",            # 워핑 현실(뭉갠 흐름)
        "warpnbr:0.25", "warpnbr:0.5", "warpnbr:1",      # 워핑 바닥(남의 흐름)
        "add:1", "addblur4:1", "addblur8:1",             # 덧셈 재측정(짝지음 보장)
        "addnbr:0.25", "addnbr:0.5", "addnbr:1",         # 덧셈 바닥
        "warpc4+res4:1", "warp+res4:1",                  # 혼합(탐색용, 판정에 쓰지 않음)
    ]

    print(f"[warp] 표본 {len(sids)}개 · 흐름추정기 {args.flow} · 변형 {len(VARIANTS)}개",
          flush=True)
    print(f"[warp] TORCH_HOME={os.environ['TORCH_HOME']}", flush=True)

    print("[warp] 이웃 표본 찾는 중(행동 시퀀스 거리)...", flush=True)
    nbr = nearest_by_action(holdout, sids)

    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats),
                           device=str(dev))
    gt_dir, static_dir = holdout / "gt_videos", Path(args.static)
    flower = (RaftFlow(dev) if args.flow == "raft" else FarnebackFlow(dev))

    # --- 1단계: 표본별 잔차와 흐름을 미리 만든다 (이웃 것을 빌려 써야 하므로) ---
    print("[warp] 1단계 — 잔차·흐름 준비", flush=True)
    resid: dict[str, torch.Tensor] = {}      # (1,T,H,W,3) fp16 CPU
    flows: dict[str, torch.Tensor] = {}      # (T,2,H,W)   fp16 CPU
    for i, sid in enumerate(sids):
        gv = scorer._load_video(gt_dir, sid).float()          # (1,T,H,W,3)
        sv = scorer._load_video(static_dir, sid).float()
        resid[sid] = (gv - sv).half()
        frames_u8 = gv[0].permute(0, 3, 1, 2).round().clamp(0, 255).to(torch.uint8)
        flows[sid] = flower(frames_u8).half()
        if (i + 1) % 12 == 0:
            print(f"[warp]   준비 {i+1}/{len(sids)}", flush=True)

    # --- 2단계: 변형별 채점 ---
    print("[warp] 2단계 — 채점 시작", flush=True)
    rows: dict[str, list] = {k: [] for k in VARIANTS}
    # flow_mag_* 는 t=0 대비 **누적** 변위다. 커널 기반 변환(CDNA/SepConv 계열)을 쓸지
    # 정하려면 이 값이 필요하다 — 커널 크기가 곧 처리 가능한 변위 상한이기 때문이다.
    diag = {"warp1_pixel_l1": [], "static_pixel_l1": [],
            "flow_mag_p95": [], "flow_mag_p99": [], "flow_mag_max": []}

    for i, sid in enumerate(sids):
        gt = scorer._load_video(gt_dir, sid)
        gv_f, gd_f = scorer.video_feature(gt)[0], scorer.dino_feature(gt)[0]
        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")
        sv = scorer._load_video(static_dir, sid).float()       # (1,T,H,W,3)
        first = sv[0, 0].permute(2, 0, 1).to(dev)              # (3,H,W)

        own_r = resid[sid].float()
        blur_r = {"addblur4:1": blur_residual(own_r, 4),
                  "addblur8:1": blur_residual(own_r, 8)}
        oth_r = resid[nbr[sid]].float()

        fl = flows[sid].float().to(dev)
        fl_c4, fl_c8 = coarsen_flow(fl, 4), coarsen_flow(fl, 8)
        fl_nbr = flows[nbr[sid]].float().to(dev)

        def as_video(x_chw: torch.Tensor) -> torch.Tensor:
            """(T,3,H,W) → 채점 규격 (1,T,H,W,3) uint8."""
            return x_chw.permute(0, 2, 3, 1).unsqueeze(0).round().clamp(0, 255).to(
                torch.uint8).cpu()

        DUMP_KEYS = ("static", "warp:1", "warpc4:1", "warpnbr:1", "addblur4:1")
        dumped: dict[str, torch.Tensor] = {}
        for key in VARIANTS:
            if key == "static":
                v = sv
            elif key == "add:1":
                v = sv + own_r
            elif key in blur_r:
                v = sv + blur_r[key]
            elif key.startswith("addnbr:"):
                v = sv + float(key.split(":")[1]) * oth_r
            elif key.endswith("+res4:1"):
                # "워핑 후 보정" — 문헌의 warp-then-refine 구조를 오라클로 흉내낸다.
                # 워핑이 못 채운 자리(가림/드러남)를 저주파 덧셈으로 메우면 얼마나 회복되나.
                w = warp(first, fl_c4 if key.startswith("warpc4") else fl, 1.0)
                wv = as_video(w).float()
                v = wv + blur_residual(gt.float() - wv, 4)
            else:
                a = float(key.split(":")[1])
                f = {"warp": fl, "warpc4": fl_c4, "warpc8": fl_c8,
                     "warpnbr": fl_nbr}[key.split(":")[0]]
                v = as_video(warp(first, f, a)).float()
            mix = v.round().clamp(0, 255).to(torch.uint8)
            if i < args.dump and key in DUMP_KEYS:
                dumped[key] = mix
            rows[key].append({
                "sid": sid,
                "dino": S.dino_component_frame_avg(scorer.dino_feature(mix)[0], gd_f),
                "video": S.video_component(scorer.video_feature(mix)[0], gv_f),
                "action": scorer.action_mae(mix, raw_actions),
            })

        # 눈으로 확인할 영상 — 숫자가 맞아도 흐름이 뒤집혔거나 엉뚱하게 밀렸을 수 있다.
        if i < args.dump:
            import imageio.v2 as imageio
            dd = Path(args.dump_dir)
            dd.mkdir(parents=True, exist_ok=True)
            for key, vid in list(dumped.items()) + [("gt", gt)]:
                p = dd / f"{sid}__{key.replace(':', '')}.mp4"
                with imageio.get_writer(p, fps=6, codec="libx264",
                                        macro_block_size=1) as wr:
                    for fr in vid[0].numpy():
                        wr.append_data(fr)

        # 진단 — 흐름 추정기가 우리 데이터에서 실제로 일하고 있는지 본다.
        w1 = as_video(warp(first, fl, 1.0)).float()
        diag["warp1_pixel_l1"].append(float((w1 - gt.float()).abs().mean()))
        diag["static_pixel_l1"].append(float((sv - gt.float()).abs().mean()))
        mag = fl.norm(dim=1).cpu().numpy()
        diag["flow_mag_p95"].append(float(np.percentile(mag, 95)))
        diag["flow_mag_p99"].append(float(np.percentile(mag, 99)))
        diag["flow_mag_max"].append(float(mag.max()))

        if (i + 1) % 8 == 0:
            print(f"[warp] 채점 {i+1}/{len(sids)}", flush=True)

    # --- 3단계: 집계 + 사전 등록한 판정선 적용 ---
    tot = {k: totals(rows[k]) for k in VARIANTS}
    means = {}
    for k in VARIANTS:
        r = rows[k]
        means[k] = {
            "dino": float(np.mean([x["dino"] for x in r])),
            "video": float(np.mean([x["video"] for x in r])),
            "action": float(np.mean([x["action"] for x in r])),
            "total": float(np.mean(tot[k])),
        }
    base = means["static"]["total"]

    GATES = {
        "gate0_kill": ("warp:1", "addblur4:1"),
        "gate1_ceiling": ("warpc4:1", "addblur4:1"),
        "gate2_floor@0.25": ("warpnbr:0.25", "addnbr:0.25"),
        "gate2_floor@0.5": ("warpnbr:0.5", "addnbr:0.5"),
        "gate2_floor@1": ("warpnbr:1", "addnbr:1"),
        "extra_hybrid_vs_add": ("warpc4+res4:1", "addblur4:1"),
        "extra_hybrid_vs_warp": ("warpc4+res4:1", "warpc4:1"),
        "extra_refine_vs_add": ("warp+res4:1", "addblur4:1"),
        "extra_refine_vs_warp": ("warp+res4:1", "warp:1"),
    }
    gates = {name: paired_t(tot[a], tot[b]) | {"a": a, "b": b}
             for name, (a, b) in GATES.items()}

    out = {
        "n_samples": len(sids),
        "flow_estimator": flower.name,
        "static_dir": str(static_dir),
        "means": means,
        "static_total": base,
        "gates": gates,
        "diagnostics": {k: float(np.mean(v)) for k, v in diag.items()},
        "rows": rows,                       # 표본별 원자료 — 짝지은 재분석용
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # --- 출력 ---
    NAMES = {"static": "정지영상 (α=0)", "warp": "워핑(원본흐름)",
             "warpc4": "워핑(흐름 4배 뭉갬)", "warpc8": "워핑(흐름 8배 뭉갬)",
             "warpnbr": "워핑(남의 흐름)", "add": "완벽한 잔차",
             "addblur4": "덧셈(잔차 4배 뭉갬)", "addblur8": "덧셈(잔차 8배 뭉갬)",
             "addnbr": "덧셈(남의 잔차)", "warpc4+res4": "혼합(뭉갠워핑+뭉갠보정)",
             "warp+res4": "혼합(원본워핑+뭉갠보정)"}
    W = 88
    print("\n" + "=" * W)
    print(f"워핑 vs 덧셈 (n={len(sids)}, 흐름={flower.name}, 전부 낮을수록 좋다)")
    print("=" * W)
    print(f"{'변형':<26}{'DINO':>10}{'Video':>10}{'Action':>10}{'TOTAL':>10}"
          f"{'static 대비':>13}")
    print("-" * W)
    for k in VARIANTS:
        m = means[k]
        kind = k.split(":")[0]
        lab = NAMES[kind] if k == "static" else f"{NAMES[kind]} α={k.split(':')[1]}"
        print(f"{lab:<26}{m['dino']:>10.5f}{m['video']:>10.5f}{m['action']:>10.5f}"
              f"{m['total']:>10.5f}{m['total'] - base:>+13.5f}")
    print("-" * W)

    print(f"\n{'짝지은 비교 (Δ<0 이면 왼쪽이 좋다)':<44}{'Δ':>11}{'SE':>9}{'t':>8}{'승률':>9}")
    print("-" * W)
    for name, g in gates.items():
        print(f"{name:<20}{g['a']:>13} vs {g['b']:<11}"
              f"{g['delta']:>+11.5f}{g['se']:>9.5f}{g['t']:>8.2f}"
              f"{g['wins']:>6}/{g['n']}")
    print("-" * W)

    d = out["diagnostics"]
    print(f"\n[진단] 워핑 α=1 픽셀 L1 {d['warp1_pixel_l1']:.3f} vs "
          f"정지영상 {d['static_pixel_l1']:.3f}  "
          f"(워핑이 더 작아야 흐름추정기가 일한 것)")
    print(f"[진단] 누적 변위(t=0 기준)  p95 {d['flow_mag_p95']:.2f}  "
          f"p99 {d['flow_mag_p99']:.2f}  max {d['flow_mag_max']:.2f} 픽셀")
    print("        0 에 가까우면 움직임이 없어 워핑·덧셈 구분이 무의미하다.")
    print("        커널 기반 변환(CDNA/SepConv)을 쓰려면 커널 크기 ≥ 2·p99 여야 한다.")
    print("\n※ 이것은 오라클이다 — 반증에는 쓰고 검증에는 쓰지 않는다.")
    print(f"[warp] 저장: {args.out}")


if __name__ == "__main__":
    main()
