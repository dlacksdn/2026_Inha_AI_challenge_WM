"""흐림의 값(세금)과 평균내기의 값을 우리 채점기로 직접 잰다 — 모델 없이.

왜 이걸 재는가
--------------
문헌 조사에서 두 가지가 나왔고, 둘 다 **우리 데이터에서 확인하지 않으면 쓸 수 없다.**

**① 흐림 세금 (blur tax)**
우리가 가려는 결정론 회귀는 필연적으로 흐릿한 영상을 낸다("조건부 평균"이라서다).
문헌은 DINOv2 특징이 흐림에 **비교적** 관대하다고 말하지만(DreamSim, DINO-IR),
동시에 **약한 흐림에서만** 그렇다고도 말한다(arXiv:2401.00463).

    그런데 우리가 서 있는 지점이 문제다.
    같은 장면의 이웃 프레임끼리는 DINO 코사인 유사도가 0.99 근처다.
    이 범위에서 유사도가 0.98 → 0.90 으로 떨어지면 거리는 0.02 → 0.10, 즉 5배다.
    "0.9 면 충분히 비슷하다"는 분류 문제의 감각을 그대로 옮기면 크게 틀린다.

  ⇒ **얼마나 흐려도 되는가**를 우리 채점기의 실제 눈금으로 재야 한다.

**② 평균내기의 값 (sample averaging)**
1X World Model Challenge 우승팀이 확산 샘플 **20개를 픽셀공간에서 평균**내서
PSNR 22.63 → 24.88 (+2.25dB) 를 얻었다(arXiv:2510.07092). 우리와 세팅이 매우 비슷하다
(휴머노이드 로봇, 행동조건, 단일 정답 거리 채점). K=1→5 만으로 이득의 84% 가 나왔고,
CFG 를 켜면 오히려 나빠졌다(24.88 → 24.20).

    이론적 근거도 있다. 코사인 거리에서도 "샘플 하나"의 대가는 최대 2배다.
        m = E[정규화된 정답 특징],   최적 = 1 − ‖m‖,   샘플 = 1 − ‖m‖²
        비율 = 1 + ‖m‖ ∈ [1, 2]

  ⇒ 우리 1.1B 확산모델은 **버릴 자산이 아니라 평균 추정기로 재활용할 자산**일 수 있다.
     다만 진짜 실험(K개 생성)은 GPU 몇 시간이 든다. 그 전에 **모델 없이 값싸게** 묻는다:
     **"그럴듯하지만 틀린 움직임" 여러 개를 평균내면 하나보다 나은가?**

     표본 i 의 예측 = 첫프레임 + (행동이 비슷한 다른 표본 K개의 잔차 평균) × α

     K 를 늘릴수록 좋아지면  →  실제 K개 생성 실험에 GPU 를 걸 값이 있다
     K 를 늘려도 그대로면    →  우리 모델의 오차는 **분산이 아니라 편향**이다. 평균은 소용없다

     이 구분이 핵심이다. **평균은 흔들림(분산)을 지우지, 치우침(편향)을 못 고친다.**

한계 (016 §5 교훈)
------------------
- 전부 train 홀드아웃이다. eval 로 그대로 옮겨 읽으면 안 된다.
- ②의 "남의 잔차 K개 평균"은 실제 확산 샘플의 대리값일 뿐이다.
  남의 잔차는 **체계적으로** 틀리고(다른 장면), 확산 샘플은 **무작위로** 흔들린다.
  구조가 달라서, 여기서 이득이 나와도 실제 K개 생성으로 재확인해야 한다.
  반대로 여기서 이득이 없으면 실제 실험을 접을 근거로는 약하다 — **반증에는 조심하라.**

사용:
  python scripts/branchC/probe_blur_and_averaging.py \
      --static artifacts/branchB/m0_step1000_b4/static_preds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("TORCH_HOME", str(REPO / "artifacts/torch_home"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/branchB"))
from wm_eval import scoring as S  # noqa: E402
from probe_warp_vs_add import paired_t, totals  # noqa: E402


def gaussian_blur(v: torch.Tensor, sigma: float) -> torch.Tensor:
    """(1,T,H,W,3) 영상에 공간 가우시안 흐림. σ 는 픽셀 단위(채점 규격 320x512 기준)."""
    if sigma <= 0:
        return v
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=v.device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    b, t, h, w, c = v.shape
    y = v.permute(0, 1, 4, 2, 3).reshape(b * t * c, 1, h, w)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return y.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)


def coarsen(v: torch.Tensor, k: int) -> torch.Tensor:
    """k 배 축소했다 되키운다 — '해상도가 부족한 신경망 출력'의 대리값."""
    b, t, h, w, c = v.shape
    x = v.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    small = F.interpolate(x, size=(max(h // k, 1), max(w // k, 1)), mode="area")
    back = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return back.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)


def knn_by_action(holdout: Path, sids: list[str], kmax: int) -> dict[str, list[str]]:
    """행동 시퀀스가 비슷한 다른 표본을 가까운 순으로 kmax 개 찾는다."""
    acts = {s: np.load(holdout / "actions" / f"{s}.npy").astype(np.float64) for s in sids}
    allv = np.stack(list(acts.values()))
    sd = allv.reshape(-1, allv.shape[-1]).std(axis=0) + 1e-8
    norm = {s: a / sd for s, a in acts.items()}
    out = {}
    for si in sids:
        d = sorted(((float(np.linalg.norm(norm[si] - norm[sj])), sj)
                    for sj in sids if sj != si))
        out[si] = [sj for _, sj in d[:kmax]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats",
                    default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--static",
                    default=str(REPO / "artifacts/branchB/m0_step1000_b4/static_preds"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "results/branchC/blur_and_averaging.json"))
    args = ap.parse_args()

    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    sids = [m["sid"] for m in samples]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    KMAX = 8
    SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0]
    COARSE = [2, 4, 8, 16]
    KS = [1, 2, 4, 8]

    VARIANTS = ["gt", "static"]
    VARIANTS += [f"blur:{s:g}" for s in SIGMAS]          # ① 흐림 세금
    VARIANTS += [f"coarse:{k}" for k in COARSE]          # ① 해상도 부족 세금
    VARIANTS += [f"avg{k}:1" for k in KS]                # ② 평균내기 (α=1)
    VARIANTS += [f"avg{k}:0.5" for k in KS]              # ② 평균내기 (α=0.5)

    print(f"[blur] 표본 {len(sids)}개 · 변형 {len(VARIANTS)}개", flush=True)
    print("[blur] 이웃 K개 찾는 중(행동 시퀀스 거리)...", flush=True)
    nbrs = knn_by_action(holdout, sids, KMAX)

    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=str(dev))
    gt_dir, static_dir = holdout / "gt_videos", Path(args.static)

    print("[blur] 1단계 — 표본별 잔차 준비", flush=True)
    resid: dict[str, torch.Tensor] = {}
    for i, sid in enumerate(sids):
        gv = scorer._load_video(gt_dir, sid).float()
        sv = scorer._load_video(static_dir, sid).float()
        resid[sid] = (gv - sv).half()
        if (i + 1) % 24 == 0:
            print(f"[blur]   준비 {i+1}/{len(sids)}", flush=True)

    print("[blur] 2단계 — 채점 시작", flush=True)
    rows: dict[str, list] = {k: [] for k in VARIANTS}
    for i, sid in enumerate(sids):
        gt = scorer._load_video(gt_dir, sid)
        gv_f, gd_f = scorer.video_feature(gt)[0], scorer.dino_feature(gt)[0]
        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")
        sv = scorer._load_video(static_dir, sid).float()
        gtf = gt.float()
        # 이웃 K개의 잔차 평균을 미리 누적해 둔다 (K 를 늘릴 때 재계산하지 않도록)
        cum, avg_res = torch.zeros_like(sv), {}
        for j, sj in enumerate(nbrs[sid][:KMAX], start=1):
            cum = cum + resid[sj].float()
            if j in KS:
                avg_res[j] = cum / j

        for key in VARIANTS:
            if key == "gt":
                v = gtf
            elif key == "static":
                v = sv
            elif key.startswith("blur:"):
                v = gaussian_blur(gtf, float(key.split(":")[1]))
            elif key.startswith("coarse:"):
                v = coarsen(gtf, int(key.split(":")[1]))
            else:                                   # avgK:α
                head, a = key.split(":")
                v = sv + float(a) * avg_res[int(head[3:])]
            mix = v.round().clamp(0, 255).to(torch.uint8)
            rows[key].append({
                "sid": sid,
                "dino": S.dino_component_frame_avg(scorer.dino_feature(mix)[0], gd_f),
                "video": S.video_component(scorer.video_feature(mix)[0], gv_f),
                "action": scorer.action_mae(mix, raw_actions),
            })
        if (i + 1) % 8 == 0:
            print(f"[blur] 채점 {i+1}/{len(sids)}", flush=True)

    tot = {k: totals(rows[k]) for k in VARIANTS}
    means = {k: {"dino": float(np.mean([x["dino"] for x in rows[k]])),
                 "video": float(np.mean([x["video"] for x in rows[k]])),
                 "action": float(np.mean([x["action"] for x in rows[k]])),
                 "total": float(np.mean(tot[k]))} for k in VARIANTS}
    base = means["static"]["total"]

    # 사전 등록한 판정선 — 결과를 보기 전에 고정했다 (문서 참조)
    GATES = {f"avg{k}_vs_avg1@1": (f"avg{k}:1", "avg1:1") for k in KS if k > 1}
    GATES |= {f"avg{k}_vs_avg1@0.5": (f"avg{k}:0.5", "avg1:0.5") for k in KS if k > 1}
    gates = {n: paired_t(tot[a], tot[b]) | {"a": a, "b": b} for n, (a, b) in GATES.items()}

    out = {"n_samples": len(sids), "means": means, "static_total": base,
           "gates": gates, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    W = 92
    print("\n" + "=" * W)
    print(f"흐림 세금과 평균내기의 값 (n={len(sids)}, 전부 낮을수록 좋다)")
    print("=" * W)
    print(f"{'변형':<28}{'DINO':>10}{'Video':>10}{'Action':>10}{'TOTAL':>10}{'static 대비':>13}")
    print("-" * W)
    LAB = {"gt": "정답 영상 (기준 0)", "static": "정지영상 (넘어야 할 선)"}
    for k in VARIANTS:
        m = means[k]
        if k in LAB:
            lab = LAB[k]
        elif k.startswith("blur:"):
            lab = f"정답을 흐리게 σ={k.split(':')[1]}px"
        elif k.startswith("coarse:"):
            lab = f"정답 해상도 1/{k.split(':')[1]}"
        else:
            h, a = k.split(":")
            lab = f"남의 잔차 {h[3:]}개 평균 α={a}"
        print(f"{lab:<28}{m['dino']:>10.5f}{m['video']:>10.5f}{m['action']:>10.5f}"
              f"{m['total']:>10.5f}{m['total'] - base:>+13.5f}")
    print("-" * W)

    print(f"\n{'평균내기가 정말 이득인가 (Δ<0 이면 K개 평균이 낫다)':<48}"
          f"{'Δ':>11}{'SE':>9}{'t':>8}{'승률':>9}")
    print("-" * W)
    for n, g in gates.items():
        print(f"{n:<30}{g['delta']:>+11.5f}{g['se']:>9.5f}{g['t']:>8.2f}{g['wins']:>6}/{g['n']}")
    print("-" * W)

    d_static = means["static"]["dino"]
    print(f"\n[눈금] 정지영상의 DINO 거리 = {d_static:.5f} 이 우리 작동 범위다.")
    print("       흐림 세금이 이 값에 견줘 얼마나 큰지로 읽어라.")
    print("       '코사인 0.9 면 비슷하다'는 분류 문제의 감각이라 여기선 통하지 않는다.")
    print(f"\n[blur] 저장: {args.out}")


if __name__ == "__main__":
    main()
