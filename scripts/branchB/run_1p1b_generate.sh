#!/usr/bin/env bash
# branch B 생성 — 학습한 1.1B UNet 으로 홀드아웃/eval 영상을 만든다.
# baseline 의 generate_baseline_videos.py 를 **수정 없이** 부른다(PYTHONPATH·CWD 만 맞춤).
#
# 사용:
#   bash scripts/branchB/run_1p1b_generate.sh <ckpt> <입력루트> <출력루트> [ddim_steps] [cfg_scale]
# 예:
#   # S4 스모크: 홀드아웃 앞 4개만
#   bash scripts/branchB/run_1p1b_generate.sh \
#        artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/last.ckpt \
#        artifacts/holdout_smoke4 artifacts/branchB/preds_smoke 50 1.0
#   # S5 판정: 홀드아웃 96개 전부 → 채점은 scripts/run_m0.py
#   bash scripts/branchB/run_1p1b_generate.sh <ckpt> artifacts/holdout artifacts/branchB/preds_pilot 50 1.0
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$REPO/open/baseline/challenge_kit"
CKPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
IN="$(cd "$2" && pwd)"
OUT="$3"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
STEPS="${4:-50}"
CFGSCALE="${5:-1.0}"

[ -f "$CKPT" ] || { echo "ERROR: ckpt 없음: $CKPT"; exit 1; }

CONDA="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
PYBIN="$($CONDA run -n wm which python)"

# 학습 config + 생성 config 둘 다 __REPO__ 치환본을 만든다(생성 config 가 학습 config 를 참조하므로 순서 무관).
"$PYBIN" "$REPO/scripts/branchB/cfg_paths.py" \
    scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml \
    scripts/branchB/configs/eval/gen_1p1b.yaml
BASE_GEN_CFG="$REPO/artifacts/branchB/_runtime_cfg/gen_1p1b.yaml"
if grep -q "__REPO__" "$BASE_GEN_CFG"; then echo "ERROR: __REPO__ 잔존"; exit 1; fi
# generate_baseline_videos.py 는 CFG 스케일 CLI 인자가 없다 → 실행별 config 사본에 값을 심는다.
GEN_CFG="$REPO/artifacts/branchB/_runtime_cfg/gen_1p1b_steps${STEPS}_cfg${CFGSCALE}.yaml"
sed -e "s/^\( *unconditional_guidance_scale:\).*/\1 ${CFGSCALE}/" \
    -e "s/^\( *ddim_steps:\).*/\1 ${STEPS}/" "$BASE_GEN_CFG" > "$GEN_CFG"
grep -E "ddim_steps|unconditional_guidance_scale" "$GEN_CFG"

export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export PYTHONPATH="$CK/libs/dynamicrafter:$CK/src:$CK:$REPO/open/baseline/shared_libs/video_utils"

cd "$CK"
echo "[branchB-gen] ckpt=$CKPT"
echo "[branchB-gen] in=$IN out=$OUT steps=$STEPS cfg=$CFGSCALE"
"$PYBIN" -u scripts/inference/generate_baseline_videos.py \
  --config "$GEN_CFG" \
  --checkpoint "$CKPT" \
  --challenge-root "$IN" \
  --prediction-root "$OUT" \
  --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
  --ddim-steps "$STEPS" \
  --overwrite
echo "[branchB-gen] 완료 → $OUT  (mp4 $(ls "$OUT"/*.mp4 2>/dev/null | wc -l)개)"
