"""LeRobot SO-100 학습 데이터에서 '월드모델 평가용 홀드아웃'을 만드는 유틸.

대회 과제는 (시작 이미지 1장 + 미래 16스텝 6D 액션) -> 미래 16프레임 영상 생성이다.
train 데이터(정답 영상 보유)에서 eval과 동일한 구조의 표본을 잘라내면,
로컬에서 0.3/0.3/0.4 산식을 재현해 모의채점할 수 있다.

핵심 규약(베이스라인 데이터로더 src/ldwma/datasets/lerobot_so100.py와 동일):
- 에피소드에서 start_idx를 고르고 frame_indices=[start_idx..start_idx+15] (연속 16프레임, downsample=1).
- 같은 인덱스의 액션 16개(raw, 비정규화)가 조건. -> eval/actions/*.npy 와 동일 포맷.
- GT 영상 = 그 16프레임. 첫 프레임이 곧 '시작 이미지'.

주의: eval 데이터는 절대 사용하지 않는다(규칙). 오직 data/train 만 읽는다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import av
import imageio
import numpy as np
import pandas as pd
from PIL import Image

# 베이스라인 auto 카메라 선택 규칙과 동일: 손목/그리퍼/팔 카메라는 제외
EXCLUDED_CAMERA_PARTS = ("wrist", "gripper", "arm")

# 제출/채점 파이프라인과 '동일한' mp4 인코딩 설정(feature_csv_utils.save_video_tensor와 일치).
# 예측 영상이 libx264 손실압축을 한 번 통과하는 경로를 GT에도 똑같이 적용해야 공정하다.
VIDEO_CODEC = "libx264"
MACRO_BLOCK_SIZE = 1


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_excluded_camera(camera_key: str) -> bool:
    name = camera_key[len("observation."):] if camera_key.startswith("observation.") else camera_key
    name = name.lower()
    return any(part in name for part in EXCLUDED_CAMERA_PARTS)


def select_camera_key(features: dict) -> str:
    """info.json features에서 auto 규칙으로 카메라 키 1개 선택."""
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if not video_keys:
        raise ValueError("no video feature in metadata")
    kept = [k for k in video_keys if not _is_excluded_camera(k)]
    if not kept:
        raise ValueError(f"no non-wrist/gripper/arm camera among {video_keys}")
    return kept[0]


def _format_lerobot_path(template: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    chunk_index = episode_index // chunks_size
    return template.format(
        episode_index=episode_index,
        episode_chunk=chunk_index,
        chunk_index=chunk_index,
        file_index=episode_index,
        video_key=video_key,
    )


def discover_datasets(train_root: Path) -> list[str]:
    """data/train 아래에서 6D 액션 SO-100 데이터셋 상대경로 목록(정렬)."""
    train_root = Path(train_root)
    out = []
    for info_path in train_root.glob("*/*/meta/info.json"):
        rel = info_path.parent.parent.relative_to(train_root)
        try:
            info = read_json(info_path)
            if info.get("features", {}).get("action", {}).get("shape") != [6]:
                continue
            select_camera_key(info.get("features", {}))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        out.append(rel.as_posix())
    if not out:
        raise ValueError(f"no SO-100 6D datasets under {train_root}")
    return sorted(out)


class EpisodeRef:
    """한 에피소드의 파일 경로/메타를 담는 가벼운 참조."""

    __slots__ = ("dataset", "episode_index", "length", "fps", "parquet", "video", "camera_key")

    def __init__(self, dataset, episode_index, length, fps, parquet, video, camera_key):
        self.dataset = dataset
        self.episode_index = episode_index
        self.length = length
        self.fps = fps
        self.parquet = parquet
        self.video = video
        self.camera_key = camera_key


def list_episodes(train_root: Path, dataset: str, min_length: int) -> list[EpisodeRef]:
    train_root = Path(train_root)
    droot = train_root / dataset
    info = read_json(droot / "meta" / "info.json")
    episodes = read_jsonl(droot / "meta" / "episodes.jsonl")
    camera_key = select_camera_key(info["features"])
    chunks_size = int(info.get("chunks_size", 1000))
    fps = int(info.get("fps", 30))
    data_template = info["data_path"]
    video_template = info["video_path"]
    refs = []
    for ep in episodes:
        length = int(ep["length"])
        if length < min_length:
            continue
        idx = int(ep["episode_index"])
        parquet = droot / _format_lerobot_path(data_template, idx, chunks_size)
        video = droot / _format_lerobot_path(video_template, idx, chunks_size, camera_key)
        if not parquet.exists() or not video.exists():
            continue
        refs.append(EpisodeRef(dataset, idx, length, fps, parquet, video, camera_key))
    return refs


def decode_frames(video_path: Path, indices: list[int]) -> np.ndarray:
    """mp4에서 지정 프레임 인덱스만 디코드 -> (N,H,W,3) uint8 RGB."""
    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                frames[i] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
    missing = [i for i in indices if i not in frames]
    if missing:
        raise IndexError(f"{video_path}: missing frames {missing[:5]}")
    return np.stack([frames[i] for i in indices], axis=0)


def read_actions(parquet_path: Path, indices: list[int]) -> np.ndarray:
    """parquet에서 지정 프레임의 액션 -> (N,6) float32 (raw, 비정규화)."""
    table = pd.read_parquet(parquet_path, columns=["action"])
    actions = np.stack(table["action"].iloc[indices].to_numpy()).astype(np.float32)
    if actions.shape[-1] != 6:
        raise ValueError(f"{parquet_path}: action dim {actions.shape[-1]} != 6")
    return actions


def save_mp4_uint8(frames: np.ndarray, path: Path, fps: int) -> None:
    """(T,H,W,3) uint8 -> mp4. 제출/채점과 동일한 libx264 설정으로 인코딩.

    submission_kit.save_video_tensor 와 동일하게 FFMPEG 백엔드(libx264,
    macro_block_size=1)를 강제한다. 최신 imageio는 mp4를 pyav로 라우팅하는데
    pyav 플러그인은 macro_block_size 인자를 받지 않으므로 format="FFMPEG"로 고정.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path, format="FFMPEG", fps=fps, codec=VIDEO_CODEC, macro_block_size=MACRO_BLOCK_SIZE
    ) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))


def save_png(frame: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def iter_holdout_samples(
    train_root: Path,
    n_samples: int,
    seed: int,
    traj_len: int = 16,
    per_dataset_cap: int | None = None,
) -> Iterator[dict]:
    """데이터셋을 가로질러 결정론적으로 홀드아웃 표본을 고른다.

    각 표본: {sid, dataset, episode_index, start_idx, frame_indices, ref}.
    라운드로빈으로 데이터셋 다양성을 확보하고, seed로 완전 재현 가능.
    """
    rng = random.Random(seed)
    datasets = discover_datasets(train_root)
    rng.shuffle(datasets)

    # 데이터셋별 사용 가능한 에피소드를 미리 수집(라운드로빈용)
    pools: dict[str, list[EpisodeRef]] = {}
    for ds in datasets:
        eps = list_episodes(train_root, ds, min_length=traj_len)
        if eps:
            rng.shuffle(eps)
            pools[ds] = eps

    order = [ds for ds in datasets if ds in pools]
    used_per_ds: dict[str, int] = {ds: 0 for ds in order}
    produced = 0
    # 라운드로빈: 한 바퀴에 데이터셋당 1개씩
    while produced < n_samples and order:
        progressed = False
        for ds in list(order):
            if produced >= n_samples:
                break
            if per_dataset_cap is not None and used_per_ds[ds] >= per_dataset_cap:
                continue
            pool = pools[ds]
            if not pool:
                continue
            ref = pool.pop()
            used_per_ds[ds] += 1
            progressed = True
            start_idx = rng.randint(0, ref.length - traj_len)
            frame_indices = [start_idx + i for i in range(traj_len)]
            sid = f"sample_{produced:06d}"
            produced += 1
            yield {
                "sid": sid,
                "dataset": ds,
                "episode_index": ref.episode_index,
                "start_idx": start_idx,
                "frame_indices": frame_indices,
                "ref": ref,
            }
        if not progressed:
            break
