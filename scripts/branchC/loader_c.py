#!/usr/bin/env python
"""
branch C 전용 데이터 로더 — branch B(ldwma)와 **독립**이다.

왜 자체 로더인가
  CLAUDE.md: "현 모델과 다른 모델의 학습/평가 절차는 독립적이어야 한다."
  ldwma 는 우리 프로젝트 밖(/home/rils/challenge/...)에 있고 wm env 에 설치돼 있지도 않다.
  그래서 **전처리 규약만 정확히 재현**하고 코드는 우리 것을 쓴다.

왜 에피소드 캐시인가 (007 §6)
  기존 decode_frames 는 프레임 0 부터 순차 디코드한다 → 평균 1.944 s/샘플, 최대 16.4 s.
  같은 디코드 비용으로 그 에피소드의 창을 여러 개 뽑으면 창당 비용이 1/K 로 떨어진다.

전처리 규약 [코드] ldwma/datasets/lerobot_so100.py:175-209 를 그대로 옮김
  uint8 → /255 → bilinear(align_corners=False) 로 scale=min(320/H, 512/W) 축소
  → 중앙 0.0 패딩 → (x-0.5)*2  ⇒ 검은 띠는 최종 -1.0
  640×480 기준: 320×427 로 줄고 좌 42 / 우 43 열이 띠다 (폭의 16.6%)
  반환 축은 [C, T, H, W]

행동 규약 [코드] 같은 파일 395-396: 차원별 z-score (act - mean) / std
  통계는 학습 split 에서 1회 계산해 json 으로 고정한다 (005 중-9: 모든 행동 접점이 같은 공간)

과제 규약 [측정] holdout_val96 대조: 입력 이미지 == 정답 영상의 0번 프레임
  ⇒ 잔차의 0번 프레임은 구조적으로 0 이다
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

REPO = Path(__file__).resolve().parents[2]   # 상대경로 (대회 §3.3 요건)
TRAIN_ROOT = REPO / "open" / "data" / "train"
HOLDOUT = REPO / "artifacts" / "holdout_val96"
ACTION_STATS = REPO / "results" / "branchC" / "action_stats.json"

TARGET_H, TARGET_W = 320, 512
TRAJ_LEN = 16            # 정답 영상 프레임 수 (= 행동 스텝 수)
WINDOW = TRAJ_LEN        # 창 = [start .. start+15]. 입력 이미지는 start 프레임


# ───────────────────────── 전처리 (규약 재현) ─────────────────────────

def _resize_pad(t: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) float [0,1] → (N,3,320,512) float [-1,1]. 규약의 본체."""
    _, _, h, w = t.shape
    scale = min(TARGET_H / h, TARGET_W / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
    pt = (TARGET_H - rh) // 2
    pb = TARGET_H - rh - pt
    pl = (TARGET_W - rw) // 2
    pr = TARGET_W - rw - pl
    t = F.pad(t, (pl, pr, pt, pb), value=0.0)
    return (t - 0.5) * 2.0


def preprocess_video(video: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 → (C,T,320,512) float32, [-1,1]. ldwma pad=True 규약. CPU 용."""
    t = torch.from_numpy(video).float().permute(0, 3, 1, 2) / 255.0
    return _resize_pad(t).permute(1, 0, 2, 3).contiguous()


def preprocess_batch(x_u8: torch.Tensor) -> torch.Tensor:
    """(B,T,H,W,3) uint8 → (B,C,T,320,512) float [-1,1].

    ⚠ 007 §6 측정: CPU bilinear 가 로더 비용의 82% 였다(창당 848 ms).
      워커는 uint8 디코드만 하고(비용의 18%) **이 함수를 GPU 에서 부른다.**
      연산은 CPU 판과 완전히 동일하다 — 장치만 다르다.
    """
    B, T = x_u8.shape[:2]
    t = x_u8.reshape(B * T, *x_u8.shape[2:]).permute(0, 3, 1, 2).float() / 255.0
    t = _resize_pad(t)
    return t.reshape(B, T, 3, TARGET_H, TARGET_W).permute(0, 2, 1, 3, 4).contiguous()


# ───────────────────────── 디코드 (seek 판, 픽셀 동일성 검증됨) ─────────────────────────

def decode_span(video_path, lo: int, hi: int) -> dict[int, np.ndarray]:
    """[lo, hi] 구간을 키프레임 seek 후 한 번에 디코드. 반환 {index: (H,W,3) uint8}."""
    out = {}
    with av.open(str(video_path)) as c:
        st = c.streams.video[0]
        tb, rate = st.time_base, st.average_rate
        c.seek(int(lo / (rate * tb)), stream=st, backward=True, any_frame=False)
        for frame in c.decode(st):
            i = int(round(float(frame.pts * tb * rate)))
            if i > hi:
                break
            if i >= lo:
                out[i] = frame.to_ndarray(format="rgb24")
    return out


# ───────────────────────── 에피소드 목록 ─────────────────────────

class Episode:
    __slots__ = ("video", "parquet", "length", "dataset", "index", "native_hw")

    def __init__(self, video, parquet, length, dataset, index, native_hw):
        self.video, self.parquet, self.length = video, parquet, length
        self.dataset, self.index, self.native_hw = dataset, index, native_hw


def _read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def _fmt(template, ep_idx, chunks, video_key=None):
    s = template.format(episode_chunk=ep_idx // chunks, episode_index=ep_idx,
                        video_key=video_key or "")
    return s


def ep_key(p) -> str:
    """에피소드 식별자 — `data/train/` 이후의 경로 꼬리.

    ⚠ 2026-08-11 추가. 이게 없어서 **누수 차단이 10일 동안 0건 작동했다.**
      holdout_val96 은 5090 기계에서 만들어져 manifest 의 경로가 `/home/rils/…` 인데
      이 기계의 학습 영상은 `/home/video_generation/…` 이다. 절대경로로 비교하니
      교집합이 늘 공집합이었다 [측정 020 §3].
      기계가 달라도 `data/train/` 아래 구조는 같으므로 꼬리로 비교한다.
    """
    s = str(p)
    return s.split("data/train/", 1)[-1]


def holdout_episode_refs() -> set[str]:
    """평가에 쓰는 96 샘플의 영상 경로. 학습에서 제외한다 (누수 차단).

    ⚠ 반환값은 **경로 꼬리**다 (ep_key). 절대경로가 아니다 — 위 주석 참조.
    """
    m = json.load(open(HOLDOUT / "manifest.json"))
    return {ep_key(s["video_ref"]) for s in m["samples"]}


def list_train_episodes(min_length: int = WINDOW + 1, exclude: set[str] | None = None,
                        only_hw: tuple[int, int] | None = (480, 640)) -> list[Episode]:
    """학습 에피소드 목록.

    only_hw  원본 해상도를 이걸로 제한한다. 기본 (480,640).
      [측정] eval 216 장은 **전부 640×480(4:3)** 이다. 학습 데이터셋 128 개 중
      123 개가 4:3, 5 개가 16:9(1080×1920 3 · 720×1280 2)다.
      pad 규약상 4:3 은 좌우 42/43 열이 띠가 되고 16:9 는 상하 16 행이 띠가 된다 —
      **띠의 방향이 다르다.** 추론에서 절대 볼 일 없는 배치를 학습시키지 않는다.
      None 을 주면 전부 쓴다(그 경우 프레임 모양이 섞여 기본 collate 가 깨진다).
    """
    exclude = exclude or set()
    eps = []
    for droot in sorted(TRAIN_ROOT.iterdir()):
        if not droot.is_dir():
            continue
        for sub in sorted([droot] + [d for d in droot.iterdir() if d.is_dir()]):
            meta = sub / "meta"
            if not (meta / "info.json").exists() or not (meta / "episodes.jsonl").exists():
                continue
            info = json.load(open(meta / "info.json"))
            feats = info.get("features", {})
            cams = [k for k in feats if k.startswith("observation.images.")]
            if not cams:
                continue
            cam = sorted(cams)[0]
            shape = feats[cam].get("shape") or []
            hw = tuple(shape[:2]) if len(shape) >= 2 else None
            if only_hw is not None and hw != tuple(only_hw):
                continue        # ⚠ 2026-08-11: break 였다. 16:9 데이터셋 하나를 만나면
                                #   같은 소유자의 나머지를 전부 건너뛰었다 [측정 020 §2]
            chunks = int(info.get("chunks_size", 1000))
            for ep in _read_jsonl(meta / "episodes.jsonl"):
                L = int(ep["length"])
                if L < min_length:
                    continue
                i = int(ep["episode_index"])
                vid = sub / _fmt(info["video_path"], i, chunks, cam)
                pq = sub / _fmt(info["data_path"], i, chunks)
                if not vid.exists() or not pq.exists() or ep_key(vid) in exclude:
                    continue
                eps.append(Episode(vid, pq, L, sub.name, i, hw))
            # ⚠ 2026-08-11: 여기에 break 가 있었다. 소유자 폴더에서 **첫 데이터셋만**
            #   쓰고 나머지를 버렸다 — 123개 중 54개만 학습에 들어갔다 [측정 020 §2]
    return eps


# ───────────────────────── 행동 통계 (차원별 z-score) ─────────────────────────

def compute_action_stats(episodes, max_ep: int = 400, seed: int = 0) -> dict:
    import pandas as pd
    rng = random.Random(seed)
    sel = rng.sample(episodes, min(max_ep, len(episodes)))
    acc = []
    for e in sel:
        try:
            a = np.stack(pd.read_parquet(e.parquet, columns=["action"])["action"].to_numpy())
            acc.append(a.astype(np.float64))
        except Exception:
            continue
    A = np.concatenate(acc, axis=0)
    return {"mean": A.mean(0).tolist(), "std": (A.std(0) + 1e-6).tolist(),
            "n_frames": int(A.shape[0]), "n_episodes": len(acc), "seed": seed}


def load_action_stats():
    d = json.load(open(ACTION_STATS))
    return (torch.tensor(d["mean"], dtype=torch.float32),
            torch.tensor(d["std"], dtype=torch.float32))


# ───────────────────────── 학습 스트림 ─────────────────────────

class EpisodeWindowStream(IterableDataset):
    """에피소드에서 연속 구간을 1회 디코드하고 그 안에서 창을 K 개 뽑는다.

    span_frames  한 번에 디코드할 프레임 수 (메모리 ↔ 창 재사용의 절충)
    windows_per_span  한 구간에서 뽑는 창 수
    shuffle_buffer    같은 구간의 창이 한 배치에 몰리는 것을 완화
    """

    def __init__(self, episodes, span_frames=96, windows_per_span=8,
                 shuffle_buffer=64, seed=0, action_norm=True):
        self.episodes = episodes
        self.span_frames = span_frames
        self.windows_per_span = windows_per_span
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.action_norm = action_norm
        self._stats = load_action_stats() if action_norm else None

    def _windows_from(self, ep, rng):
        import pandas as pd
        span = min(self.span_frames, ep.length)
        lo = rng.randint(0, max(0, ep.length - span))
        hi = lo + span - 1
        frames = decode_span(ep.video, lo, hi)
        idxs = sorted(frames)
        if len(idxs) < WINDOW:
            return
        acts_all = np.stack(pd.read_parquet(ep.parquet, columns=["action"])["action"].to_numpy()).astype(np.float32)
        starts = [s for s in idxs if all((s + k) in frames for k in range(WINDOW)) and s + WINDOW <= len(acts_all)]
        if not starts:
            return
        rng.shuffle(starts)
        for s in starts[: self.windows_per_span]:
            # ⚠ 전처리를 여기서 하지 않는다 (007 §6: CPU bilinear 가 로더의 82%).
            #   워커는 uint8 만 넘기고 preprocess_batch 가 GPU 에서 처리한다.
            vid = np.stack([frames[s + k] for k in range(WINDOW)], axis=0)  # (T,H,W,3) uint8
            a = torch.from_numpy(acts_all[s: s + WINDOW])                   # (T,6) raw
            if self._stats is not None:
                m, sd = self._stats
                a = (a - m) / sd
            yield {"frames_u8": torch.from_numpy(vid), "act": a,
                   "dataset": ep.dataset, "episode": ep.index, "start": s}

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 9973 + wid)
        mine = self.episodes[wid::nw]
        buf = []
        while True:
            rng.shuffle(mine)
            for ep in mine:
                try:
                    for item in self._windows_from(ep, rng):
                        buf.append(item)
                        if len(buf) >= self.shuffle_buffer:
                            yield buf.pop(rng.randrange(len(buf)))
                except Exception:
                    continue


# ───────────────────────── 평가 세트 (holdout_val96) ─────────────────────────

def load_holdout_val96(action_norm=True, limit=None):
    """018 오라클이 쓴 그 96 표본. 같은 자를 쓴다."""
    m = json.load(open(HOLDOUT / "manifest.json"))
    samples = m["samples"][:limit] if limit else m["samples"]
    stats = load_action_stats() if action_norm else None
    out = []
    for s in samples:
        with av.open(str(HOLDOUT / "gt_videos" / f"{s['sid']}.mp4")) as c:
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
        v = preprocess_video(np.stack(fr[:WINDOW], axis=0))
        a = torch.from_numpy(np.load(HOLDOUT / "actions" / f"{s['sid']}.npy").astype(np.float32))
        if stats is not None:
            mu, sd = stats
            a = (a - mu) / sd
        out.append({"sid": s["sid"], "video": v, "first": v[:, 0], "act": a})
    return out


# ───────────────────────── 자체 검증 + 처리량 ─────────────────────────

def _selftest():
    print("=== 1. 전처리 규약 ===")
    x = (np.random.rand(3, 480, 640, 3) * 255).astype(np.uint8)
    v = preprocess_video(x)
    pl, pr = (TARGET_W - 427) // 2, TARGET_W - 427 - (TARGET_W - 427) // 2
    print(f"   출력 {tuple(v.shape)}  범위 [{v.min():.3f}, {v.max():.3f}]")
    print(f"   좌 {pl}열 최대 {v[:, :, :, :pl].max():.4f}  우 {pr}열 최대 {v[:, :, :, -pr:].max():.4f}"
          f"   ← 둘 다 -1.0 이어야 한다")
    assert v.shape == (3, 3, 320, 512)
    assert torch.allclose(v[:, :, :, :pl], torch.tensor(-1.0)), "좌측 띠가 -1 이 아니다"
    assert torch.allclose(v[:, :, :, -pr:], torch.tensor(-1.0)), "우측 띠가 -1 이 아니다"
    print("   ✅ 통과")

    print("\n=== 2. 에피소드 목록 · 누수 차단 ===")
    excl = holdout_episode_refs()
    eps = list_train_episodes(exclude=excl)
    print(f"   홀드아웃 영상 {len(excl)}개 제외 → 학습 에피소드 {len(eps)}개")
    assert not ({str(e.video) for e in eps} & excl), "누수! 홀드아웃 에피소드가 학습에 남았다"
    print("   ✅ 누수 없음")
    return eps


if __name__ == "__main__":
    import argparse, statistics, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-stats", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="이 개수만큼 창을 뽑아 처리량 측정")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--span", type=int, default=96)
    ap.add_argument("--wps", type=int, default=8, help="한 span 에서 뽑는 창 수")
    args = ap.parse_args()
    torch.set_num_threads(2)   # 워커 과다구독 방지 (007 §6: 스레드가 리사이즈를 지배했다)

    eps = _selftest()

    if args.make_stats or not ACTION_STATS.exists():
        print("\n=== 3. 행동 통계 계산 (차원별 z-score) ===")
        st = compute_action_stats(eps)
        ACTION_STATS.parent.mkdir(parents=True, exist_ok=True)
        json.dump(st, open(ACTION_STATS, "w"), indent=2)
        print(f"   에피소드 {st['n_episodes']}개 · 프레임 {st['n_frames']}개")
        print(f"   mean {[round(x,3) for x in st['mean']]}")
        print(f"   std  {[round(x,3) for x in st['std']]}")
        print(f"   [saved] {ACTION_STATS}")

    if args.bench:
        print(f"\n=== 4. GPU 전처리가 CPU 판과 같은가 (규약 보존 확인) ===")
        x = torch.from_numpy((np.random.rand(2, WINDOW, 480, 640, 3) * 255).astype(np.uint8))
        cpu = torch.stack([preprocess_video(x[i].numpy()) for i in range(2)])
        gpu = preprocess_batch(x.cuda()).cpu()
        print(f"   최대 절대차 {(cpu - gpu).abs().max().item():.3e}   ← 0 에 가까워야 한다")

        print(f"\n=== 5. 처리량 (워커 {args.workers}, 전처리는 GPU) ===")
        from torch.utils.data import DataLoader
        ds = EpisodeWindowStream(eps, seed=0, span_frames=args.span,
                                 windows_per_span=args.wps)
        dl = DataLoader(ds, batch_size=2, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)
        it = iter(dl)
        for _ in range(5):          # 워밍업
            b = next(it)
            preprocess_batch(b["frames_u8"].cuda(non_blocking=True))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 0
        while n < args.bench:
            b = next(it)
            v = preprocess_batch(b["frames_u8"].cuda(non_blocking=True))
            n += v.shape[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sps = n / dt
        print(f"   span={args.span} 창/span={args.wps}")
        print(f"   {n} 샘플 / {dt:.1f} s = {sps:.1f} samples/s")
        print(f"   GPU 요구(마이크로배치 2 @ 258 ms) = 7.75 samples/s → "
              f"{'✅ 로더가 앞선다' if sps > 7.75 else '⚠ 여전히 병목'}")
