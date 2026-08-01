"""잔차 보정 방식의 **천장과 바닥**을 잰다 — 모델을 만들기 전에 될지 안 될지 가늠한다.

무엇을 묻는가
-------------
지금 우리 상태는 이렇다(실측).

    eval DINO+Video    우리 생성 0.56584   정지영상 0.43916
    eval Action        우리 생성 0.46622   정지영상 0.42874

**우리가 만든 움직임이 아무 움직임도 없는 것보다 나쁘다.** 그래서 다음 노선 후보가
"영상을 새로 그리지 말고, 정지영상을 정답 쪽으로 **조금 밀자**"는 잔차 보정이다.

그런데 3일을 걸기 전에 물어야 한다. **조금 미는 게 정말 이득인가?**
이건 모델 없이 잴 수 있다. 홀드아웃에는 정답 영상이 있기 때문이다.

두 가지를 재서 위아래로 가둔다
------------------------------
① **천장 — 방향이 완벽할 때**

      예측 = 정지영상 + α × (정답 − 정지영상)

  α 는 "완벽한 방향으로 몇 % 밀었나"다. α=0 이 정지영상, α=1 이 정답이다.
  이건 우리가 절대 얻을 수 없는 **오라클**이므로 도달 가능한 최댓값, 즉 천장이다.

      곡선이 오목하면 (조금만 밀어도 확 는다)  → 대충 배워도 이득. 유망
      곡선이 볼록하면 (거의 다 밀어야 는다)    → 정확해야만 이득. 위험

  ※ 천장이 낮으면 "확실히 안 된다"는 강한 결론이지만, 높다고 "된다"는 뜻은 아니다.
    **반증에는 쓸 수 있고 검증에는 못 쓴다.** 그래서 ②가 필요하다.

② **바닥 — 그럴듯하지만 틀린 방향일 때**

      표본 i 의 예측 = 표본 i 의 정지영상 + **다른 표본 j 의 잔차**
                                            └ j = 행동 시퀀스가 가장 비슷한 다른 표본

  남의 움직임을 빌려온 것이다. 그럴듯하지만 이 장면의 정답은 아니다.
  **딱 학습된 모델이 만들 법한 물건**이라 현실적인 바닥이 된다.

      static 보다 좋다  →  대충 맞는 움직임도 이득이다. 잔차 노선 유망
      static 보다 나쁘다 →  정확하지 않으면 손해다. 잔차 노선도 같은 벽에 부딪힌다

③ 보조 — **뭉갠 잔차** (천장의 낙관 보정)

  완벽한 잔차는 정답의 세밀한 무늬까지 담고 있다. 학습된 모델의 잔차는 뭉갠 저주파다.
  DINO 는 특징공간 거리라 이 둘을 다르게 채점한다. 그래서 잔차를 8배 축소했다 되키운
  (=고주파를 날린) 변형도 같이 잰다.

규칙에 대하여
-------------
전부 **train 홀드아웃**에서만 한다. eval 데이터는 쓰지 않는다.
정답 영상을 쓰지만 이건 제출 방법이 아니라 **여지를 재는 측정**이다.
여기서 나온 수치를 그대로 제출에 쓸 수는 없다(정답을 모르니까).

한계
----
로컬 홀드아웃에서만 잴 수 있다. eval 에는 정답 영상이 없다.
그리고 로컬과 eval 은 **방향까지 어긋난 전례**가 있다(016 §5: Action 은 로컬에서 이기고
eval 에서 졌다). 이 결과를 eval 로 그대로 옮겨 읽으면 안 된다.

사용:
  python scripts/branchB/probe_residual_headroom.py \
      --static artifacts/branchB/m0_step1000_b4/static_preds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
from wm_eval import scoring as S  # noqa: E402


def blur_residual(res: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """잔차의 고주파를 날린다 — 학습된 모델이 만드는 뭉갠 잔차의 대리값.

    (1,T,H,W,3) 을 T*3 채널 이미지로 보고 factor 배 축소했다가 되키운다.
    """
    b, t, h, w, c = res.shape
    x = res.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    small = F.interpolate(x, size=(max(h // factor, 1), max(w // factor, 1)),
                          mode="area")
    back = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return back.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)


def nearest_by_action(holdout: Path, sids: list[str]) -> dict[str, str]:
    """행동 시퀀스가 가장 비슷한 다른 표본을 찾는다(자기 자신 제외)."""
    acts = {}
    for sid in sids:
        a = np.load(holdout / "actions" / f"{sid}.npy").astype(np.float64)
        acts[sid] = a
    # 차원별 스케일이 다르므로 전체 표본의 표준편차로 정규화한 뒤 거리를 잰다
    allv = np.stack(list(acts.values()))                    # (N,16,6)
    sd = allv.reshape(-1, allv.shape[-1]).std(axis=0) + 1e-8
    norm = {s: (a / sd) for s, a in acts.items()}
    out = {}
    for i, si in enumerate(sids):
        best, bd = None, float("inf")
        for j, sj in enumerate(sids):
            if i == j:
                continue
            d = float(np.linalg.norm(norm[si] - norm[sj]))
            if d < bd:
                best, bd = sj, d
        out[si] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts/holdout"))
    ap.add_argument("--submission-kit", default=str(REPO / "open/submission_kit"))
    ap.add_argument("--action-stats", default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--static", default=str(REPO / "artifacts/branchB/m0_step1000_b4/static_preds"))
    ap.add_argument("--alphas", default="0,0.15,0.3,0.6,1.0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "results/branchB/residual_headroom.json"))
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    holdout = Path(args.holdout)
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    sids = [m["sid"] for m in samples]

    print(f"[headroom] 표본 {len(sids)}개, α={alphas}", flush=True)
    print("[headroom] 이웃 표본 찾는 중(행동 시퀀스 거리)...", flush=True)
    nbr = nearest_by_action(holdout, sids)

    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=args.device)
    gt_dir = holdout / "gt_videos"
    static_dir = Path(args.static)

    # 변형 목록
    # 흐림 정도를 두 가지로 잰다 — 8배는 가혹하고 4배는 온건하다. 어느 쪽이든
    # "뭉개면 이득이 사라지는가"라는 질문의 답이 흐림 세기에 얼마나 민감한지 보여준다.
    variants = [f"oracle:{a:g}" for a in alphas]
    variants += [f"blur4:{a:g}" for a in alphas if a > 0]
    variants += [f"blur8:{a:g}" for a in alphas if a > 0]
    variants += ["neighbor:1", "neighbor:0.5", "neighbor:0.25"]
    rows: dict[str, list] = {k: [] for k in variants}

    # 이웃 잔차를 쓰려면 다른 표본의 (정답 − 정지)가 필요하다. 미리 계산해 float16 으로 둔다.
    print("[headroom] 표본별 잔차 준비 중...", flush=True)
    resid: dict[str, torch.Tensor] = {}
    for i, m in enumerate(samples):
        sid = m["sid"]
        gv = scorer._load_video(gt_dir, sid).float()
        sv = scorer._load_video(static_dir, sid).float()
        resid[sid] = (gv - sv).half()
        if (i + 1) % 24 == 0:
            print(f"[headroom]   잔차 {i+1}/{len(samples)}", flush=True)

    print("[headroom] 채점 시작", flush=True)
    for i, m in enumerate(samples):
        sid = m["sid"]
        gt = scorer._load_video(gt_dir, sid)
        gv_f, gd_f = scorer.video_feature(gt)[0], scorer.dino_feature(gt)[0]
        raw_actions = np.load(holdout / "actions" / f"{sid}.npy")
        sv = scorer._load_video(static_dir, sid).float()

        own = resid[sid].float()
        blurred = {"blur4": blur_residual(own, 4), "blur8": blur_residual(own, 8)}
        other = resid[nbr[sid]].float()

        for key in variants:
            kind, a = key.split(":")
            a = float(a)
            if kind == "oracle":
                v = sv + a * own
            elif kind in blurred:
                v = sv + a * blurred[kind]
            else:                        # neighbor
                v = sv + a * other
            mix = v.round().clamp(0, 255).to(torch.uint8)
            rows[key].append({
                "dino": S.dino_component_frame_avg(scorer.dino_feature(mix)[0], gd_f),
                "video": S.video_component(scorer.video_feature(mix)[0], gv_f),
                "action": scorer.action_mae(mix, raw_actions),
            })
        if (i + 1) % 16 == 0:
            print(f"[headroom] 채점 {i+1}/{len(samples)}", flush=True)

    out = {"n_samples": len(sids), "static": str(static_dir), "variants": {}}
    for key in variants:
        r = rows[key]
        d = float(np.mean([x["dino"] for x in r]))
        v = float(np.mean([x["video"] for x in r]))
        ac = float(np.mean([x["action"] for x in r]))
        out["variants"][key] = {"dino": d, "video": v, "action": ac,
                                "total": S.weighted_total(d, v, ac)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    base = out["variants"]["oracle:0"]["total"]
    print("\n" + "=" * 84)
    print(f"잔차 보정의 천장과 바닥 (n={len(sids)}, 전부 낮을수록 좋다)")
    print("=" * 84)
    print(f"{'변형':<22}{'DINO':>10}{'Video':>10}{'Action':>10}{'TOTAL':>10}{'static 대비':>13}")
    print("-" * 84)
    for key in variants:
        r = out["variants"][key]
        name = {"oracle": "완벽한 잔차", "blur4": "뭉갠 잔차(4배)",
                "blur8": "뭉갠 잔차(8배)", "neighbor": "남의 잔차"}[key.split(":")[0]]
        a = key.split(":")[1]
        label = f"{name} α={a}" if key != "oracle:0" else "정지영상 (α=0)"
        print(f"{label:<22}{r['dino']:>10.5f}{r['video']:>10.5f}{r['action']:>10.5f}"
              f"{r['total']:>10.5f}{r['total'] - base:>+13.5f}")
    print("-" * 84)
    print("※ 'static 대비' 가 음수여야 이득이다.")
    print("※ 완벽한 잔차 = 도달 불가능한 천장. 남의 잔차 = 현실적인 바닥.")
    print(f"\n[headroom] 저장: {args.out}")


if __name__ == "__main__":
    main()
