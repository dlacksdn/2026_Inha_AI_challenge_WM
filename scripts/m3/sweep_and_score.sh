#!/usr/bin/env bash
# M3 λ-CFG 스윕: 주어진 체크포인트로 홀드아웃 96개를 λ별로 생성하고 곧바로 채점한다.
# baseline 코드(generate_baseline_videos.py, run_m0.py)는 수정하지 않고 인자만 맞춰 호출.
#
# 사용:
#   bash scripts/m3/sweep_and_score.sh <CKPT_ABS> <PRED_ROOT> <SCORE_ROOT> <NAME_PREFIX> [TAGS...]
# 예 (Part A):
#   bash scripts/m3/sweep_and_score.sh \
#     $PWD/open/baseline/checkpoints/baseline_diffusion.ckpt \
#     $PWD/artifacts/m3/partA $PWD/results/m3/partA base 10 15 20 25 30 40
#
# TAGS 는 scripts/m3/configs/eval/sweep_cfg<TAG>.yaml 의 <TAG> (10=λ1.0, 15=1.5, ...).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT="$1"; PRED_ROOT="$2"; SCORE_ROOT="$3"; NAME_PREFIX="$4"; shift 4
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=(10 15 20 25 30 40)

CK="$REPO/open/baseline/challenge_kit"
HOLDOUT="$REPO/artifacts/holdout"
STATS="../../data/train/so100_action_statistics.json"
CFG_DIR="$REPO/scripts/m3/configs/eval"

[ -f "$CKPT" ] || { echo "[sweep] ERROR: checkpoint 없음: $CKPT"; exit 1; }
[ -d "$HOLDOUT/images" ] || { echo "[sweep] ERROR: holdout 없음: $HOLDOUT"; exit 1; }

# HF_HOME/TORCH_HOME: 채점 모델(DINOv2=timm HF, R3D=torch hub)이 캐시된 곳(태스크 recipe 경로). repo-local 은 비어있어 실패함.
export HF_HOME="/home/rils/dlacksdn/.cache/hf" TORCH_HOME="/home/rils/dlacksdn/.cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
CONDA=/home/rils/dlacksdn/miniconda3/bin/conda

mkdir -p "$PRED_ROOT" "$SCORE_ROOT"

for tag in "${TAGS[@]}"; do
  cfg="$CFG_DIR/sweep_cfg${tag}.yaml"
  lam=$(grep -oP 'unconditional_guidance_scale:\s*\K[0-9.]+' "$cfg")
  pred="$PRED_ROOT/cfg${tag}"
  name="${NAME_PREFIX}_cfg${tag}"
  echo "=========================================================="
  echo "[sweep] tag=$tag  λ=$lam  ckpt=$(basename "$CKPT")  -> $pred"
  echo "=========================================================="
  mkdir -p "$pred"
  t0=$(date +%s)
  # 1) 생성 (CWD=challenge_kit; PYTHONPATH는 baseline 규칙대로)
  ( cd "$CK"
    export PYTHONPATH="$CK/libs/dynamicrafter:$CK/src:$CK:$REPO/open/baseline/shared_libs/video_utils"
    "$CONDA" run -n wm python scripts/inference/generate_baseline_videos.py \
      --config "$cfg" \
      --checkpoint "$CKPT" \
      --challenge-root "$HOLDOUT" \
      --prediction-root "$pred" \
      --action-stats-path "$STATS" \
      --ddim-steps 50 \
      --seed 0 \
      --overwrite )
  t1=$(date +%s)
  ngen=$(ls "$pred"/*.mp4 2>/dev/null | wc -l)
  echo "[sweep] 생성 완료: $ngen mp4, $((t1-t0))s"
  # 2) 채점 (repo 루트, static/gt 스킵)
  ( cd "$REPO"
    "$CONDA" run -n wm python scripts/run_m0.py \
      --holdout "$HOLDOUT" \
      --submission-kit open/submission_kit \
      --action-stats open/data/train/so100_action_statistics.json \
      --pred-dir "$pred" --pred-name "$name" \
      --skip-static --skip-gt \
      --out "$SCORE_ROOT/cfg${tag}" )
  t2=$(date +%s)
  echo "[sweep] 채점 완료: $((t2-t1))s (누적 $((t2-t0))s)"
done
echo "[sweep] 전체 완료. 점수 요약:"
"$CONDA" run -n wm python "$REPO/scripts/m3/summarize.py" "$SCORE_ROOT" 2>/dev/null || true
