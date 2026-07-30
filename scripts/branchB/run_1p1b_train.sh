#!/usr/bin/env bash
# branch B 학습 실행 — 어느 머신에서도 저장소 기준으로 동작한다(절대경로 무주입).
#
# 사용:
#   bash scripts/branchB/run_1p1b_train.sh [MAX_STEPS] [MAX_TIME] [추가 dotlist ...]
#
# 자주 쓰는 조합:
#   # 빌드·warm-start 점검만 (학습 없음)
#   BRANCHB_BUILD_ONLY=1 bash scripts/branchB/run_1p1b_train.sh
#   # 5090(32GB) 파일럿: 전체 파인튜닝은 산술상 불가 → 시간축+액션만
#   BRANCHB_TRAIN_SCOPE=action_temporal bash scripts/branchB/run_1p1b_train.sh 20 "00:00:20:00"
#   # 6000(96GB) 본 학습
#   BRANCHB_TRAIN_SCOPE=full bash scripts/branchB/run_1p1b_train.sh "" "03:04:00:00"
#   # 다른 config(예: 11M 스모크)로 배관만 검증
#   BRANCHB_CONFIG=scripts/branchB/configs/train/inha_action_diffusion_11M_smoke.yaml \
#     bash scripts/branchB/run_1p1b_train.sh 20 "00:00:10:00"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$REPO/open/baseline/challenge_kit"
SRC_CFG="${BRANCHB_CONFIG:-scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml}"
SCRIPT="$REPO/scripts/branchB/train_1p1b.py"
MAX_STEPS="${1:-}"
MAX_TIME="${2:-}"
shift $(( $# > 2 ? 2 : $# )) || true

[ -f "$REPO/open/baseline/checkpoints/backbone.ckpt" ] || { echo "ERROR: backbone.ckpt 없음"; exit 1; }
[ -d "$REPO/open/data/train" ] || { echo "ERROR: open/data/train 없음"; exit 1; }

CONDA="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
PYBIN="$($CONDA run -n wm which python)"

# ── config 의 __REPO__ 를 이 저장소로 치환한 런타임 사본을 만든다 (008 §8.6 함정 9) ──
RUNTIME_CFG="$REPO/artifacts/branchB/_runtime_cfg/$(basename "$SRC_CFG")"
"$PYBIN" "$REPO/scripts/branchB/cfg_paths.py" "$SRC_CFG"
[ -f "$RUNTIME_CFG" ] || { echo "ERROR: 런타임 config 생성 실패: $RUNTIME_CFG"; exit 1; }
if grep -q "__REPO__" "$RUNTIME_CFG"; then echo "ERROR: __REPO__ 가 남아 있다"; exit 1; fi
echo "[branchB] runtime config: $RUNTIME_CFG"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HUB_DISABLE_TELEMETRY=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HOST_GPU_NUM=1

cd "$CK"
# openaimodel3d.py 가 video_utils 를 임포트 → shared_libs 경로 필요. branchB 스크립트 경로도 추가.
export PYTHONPATH="$CK/src:$CK/libs/dynamicrafter:$REPO/open/baseline/shared_libs/video_utils:$CK:$REPO/scripts/branchB:${PYTHONPATH:-}"
if [ -f "/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
  export LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
fi

OVERRIDES=(lightning.trainer.num_nodes=1)
[ -n "$MAX_STEPS" ] && OVERRIDES+=("lightning.trainer.max_steps=$MAX_STEPS")
[ -n "$MAX_TIME" ]  && OVERRIDES+=("lightning.trainer.max_time=$MAX_TIME")
OVERRIDES+=("$@")

# 단일 GPU: torch.distributed.launch 가 child traceback 을 숨기므로 env 만 직접 주고 python 을 바로 실행
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=12358

echo "[branchB] scope=${BRANCHB_TRAIN_SCOPE:-full} zero_init=${BRANCHB_ZERO_INIT:-auto} build_only=${BRANCHB_BUILD_ONLY:-0}"
echo "[branchB] warm-start=${BRANCHB_WARMSTART_CKPT:-<config 의 pretrained_checkpoint = DC backbone>}"
echo "[branchB] overrides: ${OVERRIDES[*]}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYBIN" -u "$SCRIPT" \
  --base "$RUNTIME_CFG" --train --devices 1 "${OVERRIDES[@]}"
