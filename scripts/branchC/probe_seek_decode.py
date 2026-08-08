#!/usr/bin/env python
"""
로더 병목의 원인 검증 — decode_frames 가 seek() 을 안 쓰는 것이 원인인가?

측정(20260808_1335)에서 나온 것:
  디코드 평균 1.944 s (중앙값 0.717, 최대 16.38). 긴 에피소드 + 뒤쪽 시작이 꼬리를 만든다.
  data_utils.decode_frames 는 프레임 0부터 순차 디코드한다 (data_utils.py:138-142).

여기서 하는 것:
  ① 정확성 먼저 — seek 판이 기존 판과 **완전히 같은 픽셀**을 주는가. 아니면 쓸 수 없다
  ② 그 다음 속도 — 같은 (에피소드, 시작) 쌍에서 두 방법을 비교

⚠ 정확성이 깨지면 속도는 의미 없다. 순서를 지킨다.
"""
import argparse, json, os, random, statistics, sys, time
from datetime import datetime
from pathlib import Path

import av
import numpy as np

REPO = Path("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM")
sys.path.insert(0, str(REPO / "src"))
from wm_eval.data_utils import list_episodes, decode_frames, discover_datasets  # noqa: E402

TRAIN = REPO / "open" / "data" / "train"
T_WIN = 17


def decode_frames_seek(video_path, indices):
    """키프레임으로 seek 한 뒤 필요한 구간만 디코드. 반환 규약은 decode_frames 와 동일."""
    lo, hi = min(indices), max(indices)
    wanted = set(indices)
    frames = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        rate = stream.average_rate
        # 프레임 인덱스 → pts.  backward=True 면 lo 이전의 키프레임으로 간다
        target_pts = int(lo / (rate * tb))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            idx = int(round(float(frame.pts * tb * rate)))
            if idx > hi:
                break
            if idx in wanted:
                frames[idx] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
    missing = [i for i in indices if i not in frames]
    if missing:
        raise IndexError(f"{video_path}: seek 판이 프레임 {missing[:5]} 를 못 찾았다")
    return np.stack([frames[i] for i in indices], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pool = []
    for ds in rng.sample(discover_datasets(TRAIN), 12):
        try:
            pool.extend(list_episodes(TRAIN, ds, min_length=T_WIN + 1))
        except Exception:
            pass
    # 꼬리를 만드는 긴 에피소드를 반드시 포함시킨다
    pool.sort(key=lambda r: -r.length)
    longs = pool[:40]
    print(f"[data] 에피소드 {len(pool)}개, 가장 긴 것 {pool[0].length} 프레임\n")

    rows, mismatches = [], 0
    print("=== ① 정확성 + ② 속도 (같은 좌표에서 두 방법) ===")
    print(f"{'ep_len':>7s} {'start':>6s} {'기존(s)':>9s} {'seek(s)':>9s} {'배속':>7s}  일치")
    for i in range(args.n):
        ref = rng.choice(longs if i % 2 == 0 else pool)
        start = rng.randint(0, max(0, ref.length - T_WIN - 1))
        idx = list(range(start, start + T_WIN))
        try:
            t0 = time.perf_counter(); a = decode_frames(ref.video, idx); t1 = time.perf_counter()
            b = decode_frames_seek(ref.video, idx); t2 = time.perf_counter()
        except Exception as e:
            print(f"  [err] {ref.dataset}#{ref.episode_index} start={start}: {e}")
            continue
        same = bool(np.array_equal(a, b))
        if not same:
            mismatches += 1
        old, new = t1 - t0, t2 - t1
        rows.append({"ep_len": ref.length, "start": start, "old_s": old,
                     "new_s": new, "identical": same})
        print(f"{ref.length:7d} {start:6d} {old:9.3f} {new:9.3f} {old/max(new,1e-9):6.1f}x  "
              f"{'✅' if same else '❌ 불일치'}")

    print(f"\n=== 판정 ===")
    print(f"  픽셀 불일치 {mismatches} / {len(rows)}   ← 0 이 아니면 seek 판은 쓸 수 없다")
    if rows and mismatches == 0:
        old_m, new_m = statistics.mean(r["old_s"] for r in rows), statistics.mean(r["new_s"] for r in rows)
        need = 2 / 0.258
        print(f"  평균 디코드  기존 {old_m:.3f} s → seek {new_m:.3f} s   ({old_m/new_m:.1f}배)")
        print(f"  필요 워커    기존 {need*old_m:4.1f}개 → seek {need*new_m:4.1f}개  (CPU {os.cpu_count()}코어)")

    meta = {"generated": datetime.now().isoformat(timespec="seconds"),
            "n": len(rows), "mismatches": mismatches,
            "mean_old_s": statistics.mean([r["old_s"] for r in rows]) if rows else None,
            "mean_new_s": statistics.mean([r["new_s"] for r in rows]) if rows else None,
            "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
