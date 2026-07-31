#!/usr/bin/env bash
# 2026-07-31 밤샘 체인
#   궤적 측정 결과: 누적 1000→2000 은 완만한 악화(+0.010), 2000→3000 은 급격한 붕괴(+0.068).
#   U자 회복이 아니라 가속 악화다. 학습률 1e-4 가 1.44B warm-start 에 과한 것으로 보인다.
#   그래서 (1) 붕괴 시점을 좁히고 (2) 낮은 학습률 두 값을 같은 출발점에서 비교한다.
set -uo pipefail
cd /home/rils/dlacksdn/2026_Inha_AI_challenge_WM
export PATH=/home/rils/dlacksdn/miniconda3/bin:$PATH
export CONDA_BIN=/home/rils/dlacksdn/miniconda3/bin/conda
export HF_HOME=/home/rils/dlacksdn/.cache/hf TORCH_HOME=/home/rils/dlacksdn/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0

BEST=artifacts/branchB/ckpt_snapshots/epoch=0-step=1000.ckpt      # 현재 최선(누적 1000)
R2SNAP=artifacts/branchB/ckpt_snapshots_inha_action_diffusion_1p1b_r2

echo "############ 1단계: 블렌드 재스윕 (배치4 생성물) $(date +'%F %T')"
$CONDA_BIN run --no-capture-output -n wm python scripts/branchB/blend_static_sweep.py \
  --pred artifacts/branchB/preds_step1000_b4 \
  --static artifacts/branchB/m0_step1000_b4/static_preds \
  --alphas 0,0.2,0.3,0.4,0.5,0.7,1.0 \
  --out artifacts/branchB/blend_sweep_step1000_b4.json

echo "############ 2단계: 누적 2500 평가 — 붕괴 시점 좁히기 $(date +'%F %T')"
bash scripts/branchB/eval_ckpt.sh "$R2SNAP/epoch=0-step=1500.ckpt" cum2500_b4 50 1.0 4

echo "############ 3단계: 학습률 3e-5 (3.5시간 + 자동평가) $(date +'%F %T')"
bash scripts/branchB/overnight_train_v2.sh "$BEST" 1000 3.5 lr3e5 model.base_learning_rate=3e-5

echo "############ 4단계: 학습률 1e-5 (3.5시간 + 자동평가) $(date +'%F %T')"
bash scripts/branchB/overnight_train_v2.sh "$BEST" 1000 3.5 lr1e5 model.base_learning_rate=1e-5

echo "############ 밤샘 체인 전부 완료 $(date +'%F %T')"
