#!/usr/bin/env python
"""
④ 예산 재측정의 나머지 절반 — 실데이터로더 병목 측정

왜 필요한가:
  격자 측정(budget_simvp_grid)은 전부 합성 텐서다. train 데이터는 회전 HDD
  (ST6000DM003, ROTA=1) 에 있고, decode_frames 는 mp4 를 **처음부터 순차 디코드**한다
  (src/wm_eval/data_utils.py:138-142). 시작 인덱스가 뒤쪽이면 그만큼 다 훑는다.
  이 비용이 GPU 스텝(258 ms @ N_S=2·b2)을 압도하면 예산표가 통째로 무의미해진다.

판정선 (측정 전 등록, 007 §0순위 연장):
  L1  샘플 1개 생산 시간의 중앙값 × (필요 처리량) 이 GPU 스텝 시간보다 작으면 → 로더는 병목이 아니다
      필요 처리량 = 마이크로배치 2 / 258 ms = 7.75 samples/s
  L2  병목이면 → 원인을 분해한다 (순차 디코드 vs HDD 대역 vs parquet)
  L3  시작 인덱스 의존성을 반드시 두 점 이상에서 잰다 (규율 2). 단일점 금지

⚠ submission_kit 은 건드리지 않는다. data_utils 의 읽기 함수만 쓴다
  (decode_frames · read_actions · list_episodes — kit import 없음을 확인함).
"""
import argparse, json, os, random, statistics, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")
sys.path.insert(0, str(REPO / "src"))
from wm_eval.data_utils import list_episodes, decode_frames, read_actions, discover_datasets  # noqa: E402

TRAIN = REPO / "open" / "data" / "train"
T_WIN = 17          # 첫 프레임 1 + 미래 16
GPU_STEP_MS = 258.0  # N_S=2 hid_S=32 enc1 ckpt0 b2 실측
MICRO_BATCH = 2


def bench_one(ref, start, want_actions=True):
    idx = list(range(start, start + T_WIN))
    t0 = time.perf_counter()
    frames = decode_frames(ref.video, idx)
    t1 = time.perf_counter()
    acts = read_actions(ref.parquet, idx) if want_actions else None
    t2 = time.perf_counter()
    return {"decode_s": t1 - t0, "parquet_s": t2 - t1,
            "shape": tuple(frames.shape),
            "act_shape": tuple(acts.shape) if acts is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30, help="무작위 표본 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    datasets = discover_datasets(TRAIN)
    print(f"[data] {TRAIN} → 데이터셋 {len(datasets)}개")

    # 에피소드 풀 구성 (데이터셋을 섞어 편향 방지)
    pool = []
    for ds in rng.sample(datasets, min(12, len(datasets))):
        try:
            refs = list_episodes(TRAIN, ds, min_length=T_WIN + 1)
        except Exception as e:
            print(f"  [skip] {ds}: {e}")
            continue
        pool.extend(refs)
    print(f"[data] 사용 가능한 에피소드 {len(pool)}개\n")
    if not pool:
        raise SystemExit("에피소드를 못 찾았다")

    r0 = pool[0]
    probe = bench_one(r0, 0)
    print(f"[shape] 프레임 {probe['shape']}  액션 {probe['act_shape']}\n")

    # ── L3: 시작 인덱스 의존성을 여러 점에서 잰다 (단일점 금지) ─────────────
    print("=== A. 시작 인덱스 의존성 (순차 디코드 비용) ===")
    dep = []
    for frac, label in [(0.0, "맨앞"), (0.25, "1/4"), (0.5, "중간"), (0.9, "뒤쪽")]:
        rows = []
        for ref in rng.sample(pool, min(6, len(pool))):
            start = int((ref.length - T_WIN - 1) * frac)
            try:
                rows.append(bench_one(ref, start))
            except Exception as e:
                print(f"   [err] {ref.dataset}#{ref.episode_index} start={start}: {e}")
        if rows:
            med = statistics.median(r["decode_s"] for r in rows)
            dep.append({"frac": frac, "label": label, "n": len(rows),
                        "decode_med_s": round(med, 3)})
            print(f"   시작 {label:4s} (ep 길이의 {frac:.0%})  디코드 중앙값 {med:.3f} s   n={len(rows)}")

    # ── B. 무작위 표본 — 실제 학습이 겪을 분포 ────────────────────────────
    print(f"\n=== B. 무작위 표본 {args.n}개 (실제 학습 분포) ===")
    rows = []
    for i in range(args.n):
        ref = rng.choice(pool)
        start = rng.randint(0, max(0, ref.length - T_WIN - 1))
        try:
            r = bench_one(ref, start)
            r["start"] = start
            r["len"] = ref.length
            rows.append(r)
        except Exception as e:
            print(f"   [err] {ref.dataset}#{ref.episode_index}: {e}")
    dec = sorted(r["decode_s"] for r in rows)
    pq = sorted(r["parquet_s"] for r in rows)
    tot = sorted(r["decode_s"] + r["parquet_s"] for r in rows)

    def q(v, p):
        return v[min(len(v) - 1, int(len(v) * p))]

    print(f"   디코드   중앙값 {statistics.median(dec):.3f} s   p90 {q(dec,0.9):.3f}   최대 {dec[-1]:.3f}")
    print(f"   parquet  중앙값 {statistics.median(pq):.3f} s   p90 {q(pq,0.9):.3f}   최대 {pq[-1]:.3f}")
    print(f"   합계     중앙값 {statistics.median(tot):.3f} s   p90 {q(tot,0.9):.3f}   최대 {tot[-1]:.3f}")

    # ── C. 판정 ───────────────────────────────────────────────────────────
    need_sps = MICRO_BATCH / (GPU_STEP_MS / 1000)
    med_tot = statistics.median(tot)
    one_worker_sps = 1.0 / med_tot
    workers_needed = need_sps / one_worker_sps
    print(f"\n=== C. 판정선 L1 적용 ===")
    print(f"   GPU 가 요구하는 처리량   = 마이크로배치 {MICRO_BATCH} / {GPU_STEP_MS:.0f} ms = {need_sps:.2f} samples/s")
    print(f"   워커 1개의 처리량        = 1 / {med_tot:.3f} s = {one_worker_sps:.2f} samples/s")
    print(f"   ⇒ 필요한 워커 수         = {workers_needed:.1f} 개  (CPU {os.cpu_count()}코어)")
    verdict = ("로더는 병목이 아니다 (워커로 감당된다)" if workers_needed <= os.cpu_count() * 0.5
               else "⚠ 로더가 병목이다 — 원인 분해 필요 (L2)")
    print(f"   판정: {verdict}")

    meta = {"generated": datetime.now().isoformat(timespec="seconds"),
            "train_root": str(TRAIN.resolve()), "disk": "ST6000DM003 (HDD, ROTA=1)",
            "gpu_step_ms": GPU_STEP_MS, "micro_batch": MICRO_BATCH,
            "cpu_count": os.cpu_count(),
            "frame_shape": probe["shape"], "act_shape": probe["act_shape"],
            "start_dependency": dep,
            "random_n": len(rows),
            "decode_s": {"median": statistics.median(dec), "p90": q(dec, 0.9), "max": dec[-1]},
            "parquet_s": {"median": statistics.median(pq), "p90": q(pq, 0.9), "max": pq[-1]},
            "total_s": {"median": med_tot, "p90": q(tot, 0.9), "max": tot[-1]},
            "need_samples_per_s": need_sps, "workers_needed": workers_needed,
            "verdict": verdict,
            "note": "페이지 캐시 영향 가능. 데이터셋을 섞어 표집했으나 완전한 콜드 캐시는 아니다.",
            "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
