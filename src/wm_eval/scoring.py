"""로컬 모의채점 — 대회 산식(0.3 DINO + 0.3 Video + 0.4 Action) 재현.

설계 원칙:
1. feature 추출은 submission_kit/feature_csv_utils.py를 '그대로' import해서 쓴다.
   -> 채점기와 동일한 전처리(320x512 pad, 112x112 R3D, 518 DINO, action MAE)를 보장.
   -> submission_kit은 절대 수정하지 않는다(대회 규칙).
2. DINO/Video의 '코사인 거리 집계'만 우리가 재현한다. 서버 집계 방식은 미문서화(미확인)이므로:
   - Video: 1 - cos(pred_512, gt_512)  (표준 정의)
   - DINO : 프레임별 (1 - cos)의 평균  [기본] + flatten 단일거리 [대조용] 둘 다 계산
   - GT-vs-GT 새너티(거리=0)와 '상대 순위'로 집계 선택의 민감도를 흡수한다.
3. Action(40%)은 로컬 완전 계산: extractor가 예측영상->6D 회귀, 정답액션(정규화)과 MAE.

거리 집계가 정확한 리더보드 스케일과 다를 수 있으므로, 최종 캘리브레이션은
static-repeat 제출 1회로 로컬 추정치 대비 리더보드 값을 역산해 맞춘다(README 참고).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _add_submission_kit_to_path(submission_kit_dir: Path) -> None:
    p = str(Path(submission_kit_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


class LocalScorer:
    """submission_kit의 3개 채점 모델을 로드하고 예측/GT feature를 뽑아 점수를 낸다."""

    TARGET_H = 320
    TARGET_W = 512
    TEMPORAL = 16
    PAD = True
    DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"

    def __init__(self, submission_kit_dir: Path, action_stats_path: Path, device: str = "cuda:0"):
        _add_submission_kit_to_path(submission_kit_dir)
        import feature_csv_utils as F  # noqa: E402  (submission_kit)

        self.F = F
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.submission_kit_dir = Path(submission_kit_dir)
        self.action_stats_path = Path(action_stats_path)

        self.video_model = F.load_video_feature_model(self.device, pretrained=True)
        self.dino_model = F.load_dino_model(self.device, self.DINO_MODEL, pretrained=True)
        self.dino_image_size = F.resolve_dino_image_size(self.dino_model, requested_size=0)
        ckpt = self.submission_kit_dir / "checkpoints" / "action_extractor.ckpt"
        self.action_model = F.load_action_extractor(str(ckpt), self.device)
        self.action_mean, self.action_std = F.load_action_stats(str(self.action_stats_path))

    # --- feature 추출: submission_kit 경로 그대로 ---
    def _load_video(self, video_dir: Path, sid: str) -> torch.Tensor:
        """mp4 1개 -> (1,16,320,512,3) uint8 (채점기 to_eval_uint8 규격)."""
        return self.F.load_video_batch(
            Path(video_dir), [sid], self.TEMPORAL, self.TARGET_H, self.TARGET_W, self.PAD
        )

    def video_feature(self, videos_uint8: torch.Tensor) -> np.ndarray:
        f = self.F.extract_video_features(videos_uint8, self.video_model, self.device)
        return f.numpy()  # (B,512)

    def dino_feature(self, videos_uint8: torch.Tensor) -> np.ndarray:
        f = self.F.extract_dino_features(videos_uint8, self.dino_model, self.device, self.dino_image_size)
        return f.numpy()  # (B,16,384)

    def action_mae(self, videos_uint8: torch.Tensor, target_actions_raw: np.ndarray) -> float:
        """예측영상 + 정답액션(raw) -> 정규화공간 MAE 스칼라(=Action Component 값)."""
        tgt = torch.from_numpy(target_actions_raw.astype(np.float32))
        if self.action_mean is not None and self.action_std is not None:
            tgt = (tgt - self.action_mean) / self.action_std
        tgt = tgt.unsqueeze(0).to(self.device)  # (1,16,6)
        mae = self.F.extract_action_features(videos_uint8, self.action_model, self.device, tgt)
        return float(mae.flatten()[0])

    def action_mae_perdim(self, videos_uint8: torch.Tensor, target_actions_raw: np.ndarray) -> np.ndarray:
        """차원별(6) 정규화공간 MAE. 어느 관절이 채점 바닥을 지배하는지 진단용.

        submission_kit의 extract_action_features 는 전체 스칼라만 주므로, 동일한
        preprocess_images -> action_model 경로를 그대로 재현해 dim축만 남긴다.
        """
        frames = self.F.preprocess_images(videos_uint8.to(self.device))
        with torch.no_grad():
            pred = self.action_model(frames).float().cpu().numpy()[0]  # (16,6) 정규화공간
        tgt = target_actions_raw.astype(np.float32)
        if self.action_mean is not None and self.action_std is not None:
            tgt = (tgt - np.asarray(self.action_mean)) / np.asarray(self.action_std)
        return np.mean(np.abs(pred - tgt), axis=0)  # (6,)


# --- 코사인 거리 집계(서버 산식 재현; 미확인 -> 표준 관례) ---
def _cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(1.0 - np.dot(a, b) / denom)


def video_component(pred_512: np.ndarray, gt_512: np.ndarray) -> float:
    return _cosine_distance(pred_512, gt_512)


def dino_component_frame_avg(pred_16x384: np.ndarray, gt_16x384: np.ndarray) -> float:
    """[기본 가정] 프레임별 (1-cos)의 평균."""
    return float(np.mean([_cosine_distance(pred_16x384[t], gt_16x384[t]) for t in range(pred_16x384.shape[0])]))


def dino_component_flatten(pred_16x384: np.ndarray, gt_16x384: np.ndarray) -> float:
    """[대조 가정] 16*384 flatten 후 단일 코사인 거리."""
    return _cosine_distance(pred_16x384.reshape(-1), gt_16x384.reshape(-1))


def weighted_total(dino: float, video: float, action: float) -> float:
    """대회 산식: 0에 가까울수록 좋음."""
    return 0.3 * dino + 0.3 * video + 0.4 * action
