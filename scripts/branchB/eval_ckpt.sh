#!/usr/bin/env bash
# 학습된 1.1B 체크포인트를 홀드아웃 96 으로 생성·채점하고 DINO 판정을 낸다.
# 사용: bash scripts/branchB/eval_ckpt.sh <ckpt> <태그> [ddim_steps] [cfg_scale]
set -uo pipefail
REPO=/home/rils/dlacksdn/2026_Inha_AI_challenge_WM
cd "$REPO" || exit 1
CKPT="$1"; TAG="$2"; STEPS="${3:-50}"; CFGS="${4:-1.0}"
CONDA=/home/rils/dlacksdn/miniconda3/bin/conda
export PATH=/home/rils/dlacksdn/miniconda3/bin:$PATH CONDA_BIN=$CONDA
export HF_HOME=/home/rils/dlacksdn/.cache/hf TORCH_HOME=/home/rils/dlacksdn/.cache/torch
export HF_HUB_DISABLE_TELEMETRY=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRED="artifacts/branchB/preds_$TAG"; OUT="artifacts/branchB/m0_$TAG"
echo "===== EVAL $TAG  ckpt=$(basename $CKPT)  steps=$STEPS cfg=$CFGS  $(date +'%F %T') ====="
t0=$(date +%s)
bash scripts/branchB/run_1p1b_generate.sh "$CKPT" artifacts/holdout "$PRED" "$STEPS" "$CFGS"
t1=$(date +%s); n=$(ls "$PRED"/*.mp4 2>/dev/null | wc -l)
echo "[gen] mp4=$n / $(( (t1-t0)/60 ))분 $(( (t1-t0)%60 ))초  (sec/샘플 $(( n>0 ? (t1-t0)/n : 0 )))"
[ "$n" -eq 0 ] && { echo "!! 생성 실패 — 중단"; exit 1; }

$CONDA run -n wm python scripts/run_m0.py --holdout artifacts/holdout --out "$OUT" \
    --pred-dir "$PRED" --pred-name "b1p1b_$TAG"
echo "[score] exit=$?  총 $(( ($(date +%s)-t0)/60 ))분"

$CONDA run -n wm python - <<PY
import json
d=json.load(open("$OUT/m0_report.json"))
for n,r in d["results"].items():
    m=r["mean"]; print(f"{n:18} DINO {m['dino_frame_avg']:.5f} | Video {m['video']:.5f} | Action {m['action']:.5f} | TOTAL {m['total_frame_avg']:.5f}")
p=d["results"].get("b1p1b_$TAG")
if p:
    x=p["mean"]["dino_frame_avg"]
    print(f"\n>>> DINO = {x:.5f}  (static 0.123 / 11M재학습 0.481 / baseline11M 0.550)")
    v = ("*** 생성이 정지를 이겼다 → 본 학습 준비 ***" if x<0.123 else
         "방향 맞음 → 학습 더 돌려 추세 확인" if x<0.30 else
         "사전학습 이득은 있으나 부족 → scope·lr·fs A/B" if x<0.48 else
         "11M 재학습과 유사 → 원인 규명 필요")
    print(f">>> 판정: {v}")
PY
echo "===== EVAL $TAG 완료 $(date +'%F %T') ====="
