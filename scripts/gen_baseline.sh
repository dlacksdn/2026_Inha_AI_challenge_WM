#!/usr/bin/env bash
# 주최측 11M baseline 모델로 홀드아웃 예측 영상을 생성한다.
# submission_kit/baseline 코드는 수정하지 않고, PYTHONPATH+CWD만 맞춰 그대로 실행.
#
# 사용:
#   bash scripts/gen_baseline.sh <holdout_dir> <out_pred_dir> [ddim_steps]
# 예:
#   bash scripts/gen_baseline.sh artifacts/holdout artifacts/baseline_preds 50
#
# 전제: open/baseline/checkpoints/backbone.ckpt (DynamiCrafter_512 model.ckpt) 존재.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOLDOUT="$(cd "$1" && pwd)"
OUT="$2"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
STEPS="${3:-50}"

CK="$REPO/open/baseline/challenge_kit"
if [ ! -f "$REPO/open/baseline/checkpoints/backbone.ckpt" ]; then
  echo "[gen_baseline] ERROR: backbone.ckpt 없음. 먼저 다운로드:"
  echo "  https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt -> open/baseline/checkpoints/backbone.ckpt"
  exit 1
fi

cd "$CK"
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$CK/libs/dynamicrafter:$CK/src:$CK:$REPO/open/baseline/shared_libs/video_utils"

echo "[gen_baseline] holdout=$HOLDOUT out=$OUT ddim_steps=$STEPS"
python scripts/inference/generate_baseline_videos.py \
  --config configs/eval/inha_submission_eval_11M.yaml \
  --checkpoint ../checkpoints/baseline_diffusion.ckpt \
  --challenge-root "$HOLDOUT" \
  --prediction-root "$OUT" \
  --action-stats-path ../../data/train/so100_action_statistics.json \
  --ddim-steps "$STEPS" \
  --overwrite
echo "[gen_baseline] 완료 -> $OUT"
