#!/usr/bin/env bash
# M3 Part B 학습 실행 — 11M 을 action_dropout_prob=0.1 로 warm-start 재학습.
# train.sh 와 동일한 torch.distributed.launch 방식(LOCAL_RANK/RANK/WORLD_SIZE 주입)으로 train_m3.py 실행.
#
# 사용:
#   bash scripts/m3/run_partB_train.sh [MAX_STEPS] [MAX_TIME]
#   (기본: config 값 max_steps=6000, max_time=00:03:30:00. 인자로 재정의 가능)
#   warm-start 끄고 scratch로 하려면:  M3_NO_WARMSTART=1 bash scripts/m3/run_partB_train.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$REPO/open/baseline/challenge_kit"
CFG="$REPO/scripts/m3/configs/train/inha_action_diffusion_11M_m3.yaml"
SCRIPT="$REPO/scripts/m3/train_m3.py"
MAX_STEPS="${1:-}"
MAX_TIME="${2:-}"

[ -f "$REPO/open/baseline/checkpoints/backbone.ckpt" ] || { echo "ERROR: backbone.ckpt 없음"; exit 1; }
[ -f "$REPO/open/baseline/checkpoints/baseline_diffusion.ckpt" ] || { echo "ERROR: baseline_diffusion.ckpt 없음"; exit 1; }

# 캐시 경로: 이미 export돼 있으면 그 값을 쓰고, 없으면 머신별 기본값(홈 캐시).
#   5090(공유계정)에서는 recipe 캐시(/home/rils/dlacksdn/.cache/hf)를 미리 export해서 쓴다.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HUB_DISABLE_TELEMETRY=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HOST_GPU_NUM=1

# warm-start (기본 on)
if [ "${M3_NO_WARMSTART:-0}" = "1" ]; then
  export M3_WARMSTART_UNET=""
else
  export M3_WARMSTART_UNET="$REPO/open/baseline/checkpoints/baseline_diffusion.ckpt"
fi

CONDA="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
PYBIN="$($CONDA run -n wm which python)"

cd "$CK"
# openaimodel3d.py 가 `from video_utils.helpers import ...` 를 임포트 → shared_libs/video_utils(외부 dir, 내부에 video_utils 패키지)를 path에.
# (wm env엔 video_utils 미설치. 생성 스크립트와 동일 규칙.)
export PYTHONPATH="$CK/src:$CK/libs/dynamicrafter:$REPO/open/baseline/shared_libs/video_utils:$CK:${PYTHONPATH:-}"
if [ -f "/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
  export LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
fi

# dotlist override
#   config에 박힌 절대경로(5090 기준)를 이 저장소 기준으로 덮어써서 어느 머신에서도 동작하게 한다.
#   (5090에서도 같은 값이 되므로 무해)
OVERRIDES=(lightning.trainer.num_nodes=1
           "logdir=$REPO/artifacts/m3/train_out"
           "data.params.root=$REPO/open/data/train"
           "data.params.action_stats_path=$REPO/open/data/train/so100_action_statistics.json"
           "model.pretrained_checkpoint=$REPO/open/baseline/checkpoints/backbone.ckpt")
[ -n "$MAX_STEPS" ] && OVERRIDES+=("lightning.trainer.max_steps=$MAX_STEPS")
[ -n "$MAX_TIME" ]  && OVERRIDES+=("lightning.trainer.max_time=$MAX_TIME")

# torch.distributed.launch 는 child traceback 을 숨겨 무인 디버깅이 어렵다.
# 단일 GPU 이므로 그 launcher 가 세팅하는 env(LOCAL_RANK/RANK/WORLD_SIZE/MASTER_*)를 직접 주고 python 을 바로 실행 → traceback 가시.
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=12357

echo "[partB] warm-start=${M3_WARMSTART_UNET:-<scratch>}"
echo "[partB] overrides: ${OVERRIDES[*]}"
echo "[partB] PYTHONPATH=$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 "$PYBIN" -u "$SCRIPT" \
  --base "$CFG" \
  --train \
  --devices 1 \
  "${OVERRIDES[@]}"
