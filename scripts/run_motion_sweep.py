"""무학습 모션-억제 스윕 — "드리프트를 줄이면 점수가 오르는가"를 학습 없이 검증.

배경(M0/M3):
  - 넘어야 할 기준은 static(정지영상) TOTAL 0.560. 주최 baseline 11M은 0.711로 더 나쁘다.
  - 원인은 용량이 아니라 시간적 드리프트: baseline의 프레임간 모션이 GT의 1.8배로 과하다.
  - 그래서 질문: baseline이 예측한 "움직임"을 인위적으로 줄이면 점수가 static 아래로 내려가는가?
    (= 생성이 만든 움직임에 유용한 정보가 조금이라도 들어있는가)

설계(공정성):
  - 모든 변형은 채점기가 보는 320x512 uint8 공간에서 만든다.
      I0 = static_preds(첫 프레임 16복사)를 채점기 로더로 읽은 것 → 320x512 패딩된 시작 프레임
      B  = baseline_preds를 채점기 로더로 읽은 것 → 320x512 생성 영상
  - 변형은 mp4로 저장 후 다시 채점(제출 경로와 동일한 libx264 왕복).
    이때 변형은 인코딩을 2회 통과하므로, 끝점(baseline 그대로 / static 그대로)도
    **같은 2회 경로**로 생성해 비교 기준으로 삼는다 → 인코딩 열화가 결론을 만들지 않음을 보장.
  - GT feature는 한 번만 계산해 캐시(예측기마다 재계산하지 않음).

변형 그룹:
  A) blend-a      P_t = (1-a)*B_t + a*I0                 (외형·모션 동시에 static 쪽으로)
  B) motion-b     P_t = I0 + b*(B_t - B_0)               (외형은 I0 고정, 모션 크기만 b배)
  C) anchor-g     P_t = (1-w_t)*B_t + w_t*I0, w_t=g*t/15 (뒤로 갈수록 강하게 앵커 = 드리프트 보정)
  D) smooth-k     시간축 이동평균 k                        (프레임 떨림 제거)
  E) motionnorm   b_sid = target/motion(B_sid)            (표본별로 모션을 train GT 평균에 맞춤)

사용:
  python scripts/run_motion_sweep.py --holdout artifacts/holdout \
      --baseline-dir artifacts/baseline_preds --static-dir artifacts/m0/static_preds \
      --out artifacts/motion_sweep --results results/motion_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wm_eval import data_utils as D  # noqa: E402
from wm_eval import scoring as S  # noqa: E402


# ---------- 변형 정의 ----------

def v_blend(B: np.ndarray, I0: np.ndarray, a: float) -> np.ndarray:
    return (1.0 - a) * B + a * I0


def v_motion(B: np.ndarray, I0: np.ndarray, b: float) -> np.ndarray:
    return I0 + b * (B - B[0:1])


def v_anchor(B: np.ndarray, I0: np.ndarray, g: float) -> np.ndarray:
    t = np.arange(B.shape[0], dtype=np.float32) / max(1, B.shape[0] - 1)
    w = (g * t).reshape(-1, 1, 1, 1)
    return (1.0 - w) * B + w * I0


def v_smooth(B: np.ndarray, I0: np.ndarray, k: int) -> np.ndarray:
    """시간축 이동평균(반사 패딩). 외형/모션 방향은 유지하고 고주파 떨림만 제거."""
    pad = k // 2
    padded = np.concatenate([B[1:1 + pad][::-1], B, B[-1 - pad:-1][::-1]], axis=0)
    out = np.stack([padded[i:i + k].mean(axis=0) for i in range(B.shape[0])], axis=0)
    return out


def frame_motion(frames: np.ndarray) -> float:
    """프레임간 평균 |Δpixel| (0~255 스케일). M3 analyze_motion과 동일 정의."""
    if frames.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean())


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def build_variants(args) -> list[dict]:
    """(name, kind, param) 목록. 끝점은 인코딩 경로를 맞추기 위해 명시적으로 포함."""
    variants: list[dict] = []
    # 끝점(재인코딩 기준선): a=0.0 → baseline 그대로, a=1.0 → static 그대로
    # 0.9~1.0 구간을 세밀화: "정지보다 조금이라도 나은 지점"이 존재하는지 정밀 탐색
    for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0]:
        variants.append({"name": f"blend_a{a:.2f}", "kind": "blend", "param": a})
    for b in [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
        variants.append({"name": f"motion_b{b:.2f}", "kind": "motion", "param": b})
    for g in [0.5, 1.0]:
        variants.append({"name": f"anchor_g{g:.2f}", "kind": "anchor", "param": g})
    for k in [3, 5]:
        variants.append({"name": f"smooth_k{k}", "kind": "smooth", "param": k})
    variants.append({"name": "motionnorm", "kind": "motionnorm", "param": args.motion_target})
    return variants


def apply_variant(kind: str, param, B: np.ndarray, I0: np.ndarray) -> np.ndarray:
    if kind == "blend":
        return v_blend(B, I0, float(param))
    if kind == "motion":
        return v_motion(B, I0, float(param))
    if kind == "anchor":
        return v_anchor(B, I0, float(param))
    if kind == "smooth":
        return v_smooth(B, I0, int(param))
    if kind == "motionnorm":
        # 표본별: baseline 모션을 target(train GT 평균 모션)으로 맞추는 b
        m = frame_motion(B)
        b = 1.0 if m <= 1e-6 else min(1.0, float(param) / m)
        return v_motion(B, I0, b)
    raise ValueError(kind)


# ---------- 메인 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="artifacts/holdout")
    ap.add_argument("--baseline-dir", default="artifacts/baseline_preds")
    ap.add_argument("--static-dir", default="artifacts/m0/static_preds")
    ap.add_argument("--submission-kit", default="open/submission_kit")
    ap.add_argument("--action-stats", default="open/data/train/so100_action_statistics.json")
    ap.add_argument("--out", default="artifacts/motion_sweep")
    ap.add_argument("--results", default="results/motion_sweep")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="검증용: 앞의 N표본만")
    ap.add_argument("--motion-target", type=float, default=4.81, help="train GT 평균 모션(전역 target)")
    args = ap.parse_args()

    holdout = Path(args.holdout)
    out_root = Path(args.out)
    res_root = Path(args.results)
    res_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.limit:
        samples = samples[: args.limit]
    sids = [s["sid"] for s in samples]
    fps_of = {s["sid"]: s["fps"] for s in samples}

    print(f"[sweep] 채점기 로딩 ... (n={len(sids)})")
    scorer = S.LocalScorer(Path(args.submission_kit), Path(args.action_stats), device=args.device)

    variants = build_variants(args)
    print(f"[sweep] 변형 {len(variants)}개: {[v['name'] for v in variants]}")

    # ---- 1패스: 샘플별로 B/I0 로드 → 모든 변형 mp4 생성 (+모션 기록) ----
    motion_log: dict[str, list[float]] = {v["name"]: [] for v in variants}
    motion_log["_baseline_src"] = []
    motion_log["_gt"] = []
    for i, sid in enumerate(sids):
        B = scorer._load_video(Path(args.baseline_dir), sid)[0].numpy().astype(np.float32)  # (16,320,512,3)
        I0 = scorer._load_video(Path(args.static_dir), sid)[0].numpy().astype(np.float32)
        G = scorer._load_video(holdout / "gt_videos", sid)[0].numpy().astype(np.float32)
        motion_log["_baseline_src"].append(frame_motion(B))
        motion_log["_gt"].append(frame_motion(G))
        for v in variants:
            frames = to_uint8(apply_variant(v["kind"], v["param"], B, I0))
            motion_log[v["name"]].append(frame_motion(frames.astype(np.float32)))
            D.save_mp4_uint8(frames, out_root / v["name"] / f"{sid}.mp4", fps=fps_of[sid])
        if (i + 1) % 16 == 0:
            print(f"[sweep] 변형 생성 {i+1}/{len(sids)}")

    # ---- GT feature 캐시 (1회) ----
    print("[sweep] GT feature 캐시 계산 ...")
    gt_v: dict[str, np.ndarray] = {}
    gt_d: dict[str, np.ndarray] = {}
    raw_actions: dict[str, np.ndarray] = {}
    for sid in sids:
        g = scorer._load_video(holdout / "gt_videos", sid)
        gt_v[sid] = scorer.video_feature(g)[0]
        gt_d[sid] = scorer.dino_feature(g)[0]
        raw_actions[sid] = np.load(holdout / "actions" / f"{sid}.npy")

    # ---- 변형별 채점 ----
    results = {}
    for vi, v in enumerate(variants):
        name = v["name"]
        pdir = out_root / name
        rows = []
        for sid in sids:
            pv_t = scorer._load_video(pdir, sid)
            pv = scorer.video_feature(pv_t)[0]
            pd_ = scorer.dino_feature(pv_t)[0]
            action = scorer.action_mae(pv_t, raw_actions[sid])
            dino = S.dino_component_frame_avg(pd_, gt_d[sid])
            dino_flat = S.dino_component_flatten(pd_, gt_d[sid])
            video = S.video_component(pv, gt_v[sid])
            rows.append({
                "sid": sid, "dino_frame_avg": dino, "dino_flatten": dino_flat,
                "video": video, "action": action,
                "total_frame_avg": S.weighted_total(dino, video, action),
            })
        mean = {k: float(np.mean([r[k] for r in rows]))
                for k in ["dino_frame_avg", "dino_flatten", "video", "action", "total_frame_avg"]}
        results[name] = {
            "kind": v["kind"], "param": v["param"], "n": len(rows),
            "mean": mean,
            "motion_mean": float(np.mean(motion_log[name])),
            "rows": rows,
        }
        print(f"[sweep] {vi+1}/{len(variants)} {name:<16} TOTAL={mean['total_frame_avg']:.5f} "
              f"DINO={mean['dino_frame_avg']:.5f} Video={mean['video']:.5f} "
              f"Action={mean['action']:.5f} motion={results[name]['motion_mean']:.2f}")

    report = {
        "n_samples": len(sids),
        "holdout": str(holdout),
        "motion_reference": {
            "gt_mean": float(np.mean(motion_log["_gt"])),
            "baseline_src_mean": float(np.mean(motion_log["_baseline_src"])),
            "target_used": args.motion_target,
        },
        "note": "변형은 libx264 2회 통과. 끝점(blend_a0.00=baseline, blend_a1.00=static)도 동일 경로라 내부 비교는 공정.",
        "results": results,
    }
    (res_root / "motion_sweep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 표 ----
    print("\n" + "=" * 92)
    print(f"무학습 모션-억제 스윕 (n={len(sids)}) — 0에 가까울수록 좋음 | 기준 static 0.56032 / GT 0.48911")
    print("=" * 92)
    print(f"{'variant':<18}{'motion':>8}{'DINO':>10}{'Video':>10}{'Action':>10}{'TOTAL':>11}{'vs static':>11}")
    print("-" * 92)
    order = sorted(results.items(), key=lambda kv: kv[1]["mean"]["total_frame_avg"])
    for name, r in order:
        m = r["mean"]
        delta = m["total_frame_avg"] - 0.56032
        flag = "  ★넘음" if delta < 0 else ""
        print(f"{name:<18}{r['motion_mean']:>8.2f}{m['dino_frame_avg']:>10.5f}{m['video']:>10.5f}"
              f"{m['action']:>10.5f}{m['total_frame_avg']:>11.5f}{delta:>+11.5f}{flag}")
    print("-" * 92)
    print(f"참조 모션: GT {report['motion_reference']['gt_mean']:.2f} / "
          f"baseline원본 {report['motion_reference']['baseline_src_mean']:.2f}")
    print(f"\n[sweep] 리포트: {res_root / 'motion_sweep_report.json'}")


if __name__ == "__main__":
    main()
