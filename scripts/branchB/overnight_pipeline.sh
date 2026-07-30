#!/usr/bin/env bash
# 밤샘 무인 파이프라인 (012 §3 단계 2~5 자동 실행)
#   단계2 스모크(20스텝) → 단계3 액션민감도 → 단계4 스모크생성·채점
#   → 단계5 파일럿 학습(시간박스) → 홀드아웃96 생성·채점 → 판정 요약
#
# 설계 원칙(무인 운전):
#   - set -e 를 쓰지 않는다. 한 단계가 실패해도 다음 단계로 진행해 "아침에 뭐라도 남게" 한다.
#   - 학습이 OOM 으로 죽으면 scope 를 낮춰(action_only) 자동 재시도한다.
#   - 어떤 경우든 마지막에 존재하는 체크포인트로 생성·채점을 시도한다.
#   - 모든 로그는 run_logs/ 에 남긴다. 체크포인트는 지우지 않는다(스모크 것도 백업 보존).
#
# 사용: setsid nohup bash scripts/branchB/overnight_pipeline.sh [PILOT_MAXTIME] > run_logs/overnight_<ts>.log 2>&1 &
set -uo pipefail

REPO=/home/rils/dlacksdn/2026_Inha_AI_challenge_WM
cd "$REPO" || exit 1

PILOT_MAXTIME="${1:-00:05:00:00}"          # DD:HH:MM:SS (기본 5시간)
TS=$(date +%m%d_%H%M)
CONDA=/home/rils/dlacksdn/miniconda3/bin/conda

export PATH=/home/rils/dlacksdn/miniconda3/bin:$PATH
export CONDA_BIN=$CONDA
export HF_HOME=/home/rils/dlacksdn/.cache/hf TORCH_HOME=/home/rils/dlacksdn/.cache/torch
export HF_HUB_DISABLE_TELEMETRY=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# [핵심1] 메모리 단편화 완화 — 이거 없으면 31GB 한계에서 스래싱/OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# [핵심2] 단일 GPU 에서 DDP 는 순수 손해(gradient bucket 메모리 + find_unused_parameters 그래프 순회).
#         SingleDeviceStrategy 로 교체 → 메모리·속도 동시 개선.
SD_OVR=(lightning.trainer.strategy.target=pytorch_lightning.strategies.SingleDeviceStrategy
        lightning.trainer.strategy.params.device=cuda:0)

CKDIR="$REPO/artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints"
say() { echo ""; echo "############ $* ############  $(date +'%F %T')"; }
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1; }

say "PIPELINE START (pilot maxtime=$PILOT_MAXTIME)"
echo "GPU: $(gpu)"; df -h . | tail -1

# ───────────────────────── 단계 2: 스모크 20스텝 ─────────────────────────
say "STAGE 2  스모크 학습 20스텝 (SingleDevice + expandable_segments)"
BRANCHB_TRAIN_SCOPE=action_temporal \
  bash scripts/branchB/run_1p1b_train.sh 20 "00:00:40:00" "${SD_OVR[@]}" \
  > "run_logs/ov_${TS}_s2_smoke.log" 2>&1
echo "exit=$?  peakGPU=$(gpu)"
grep -oE "[0-9.]+s/it" "run_logs/ov_${TS}_s2_smoke.log" | tail -3
if grep -q "OutOfMemory" "run_logs/ov_${TS}_s2_smoke.log"; then
  echo "!! 스모크 OOM → scope 를 action_only 로 낮춰 재시도"
  SCOPE=action_only
  BRANCHB_TRAIN_SCOPE=action_only \
    bash scripts/branchB/run_1p1b_train.sh 20 "00:00:40:00" "${SD_OVR[@]}" \
    > "run_logs/ov_${TS}_s2_smoke_ao.log" 2>&1
  echo "재시도 exit=$?"
else
  SCOPE=action_temporal
fi
echo "채택 scope=$SCOPE"
ls -lh "$CKDIR"/*.ckpt 2>/dev/null | tail -3

SMOKE_CKPT="$CKDIR/last.ckpt"
# ───────────────────── 단계 3: 액션 민감도(스모크 시점) ─────────────────────
say "STAGE 3  액션 민감도 (학습 직후 — 0에서 벗어나기 시작했는가)"
if [ -f "$SMOKE_CKPT" ]; then
  $CONDA run -n wm python scripts/branchB/probe_action_sensitivity.py \
      --device cuda --n-pairs 3 --ckpt "$SMOKE_CKPT" --tag smoke20 \
      > "run_logs/ov_${TS}_s3_sens.log" 2>&1
  echo "exit=$?"; grep -E "S_AB|S_A0|상대차|mean" "run_logs/ov_${TS}_s3_sens.log" | tail -6
else
  echo "체크포인트 없음 → 건너뜀"
fi

# ─────────────────── 단계 4: 스모크 생성 4샘플 + 채점 ───────────────────
say "STAGE 4  스모크 생성 4샘플 + 채점 (파이프라인 무결성 + sec/샘플)"
if [ -f "$SMOKE_CKPT" ]; then
  t0=$(date +%s)
  bash scripts/branchB/run_1p1b_generate.sh "$SMOKE_CKPT" \
       artifacts/holdout_smoke4 artifacts/branchB/preds_smoke4 50 1.0 \
       > "run_logs/ov_${TS}_s4_gen.log" 2>&1
  t1=$(date +%s); n=$(ls artifacts/branchB/preds_smoke4/*.mp4 2>/dev/null | wc -l)
  echo "생성 exit=$? / mp4=$n / $((t1-t0))초"
  [ "$n" -gt 0 ] && echo "sec/샘플 ≈ $(( (t1-t0) / n )) (모델로드 포함)"
  $CONDA run -n wm python scripts/run_m0.py \
      --holdout artifacts/holdout_smoke4 --out artifacts/branchB/m0_smoke4 \
      --pred-dir artifacts/branchB/preds_smoke4 --pred-name b1p1b_smoke --skip-gt \
      > "run_logs/ov_${TS}_s4_score.log" 2>&1
  echo "채점 exit=$?"; grep -E "^(static|b1p1b|gt)" "run_logs/ov_${TS}_s4_score.log" | tail -4
else
  echo "체크포인트 없음 → 건너뜀"
fi

# 스모크 체크포인트 보존(파일럿이 덮어쓰기 전에)
if [ -f "$SMOKE_CKPT" ]; then
  mkdir -p "$REPO/artifacts/branchB/ckpt_smoke20"
  cp -a "$SMOKE_CKPT" "$REPO/artifacts/branchB/ckpt_smoke20/" 2>/dev/null && echo "스모크 ckpt 백업 완료"
fi

# ───────────────────── 단계 5: 파일럿 학습 (시간박스) ─────────────────────
say "STAGE 5  파일럿 학습 scope=$SCOPE maxtime=$PILOT_MAXTIME"
BRANCHB_TRAIN_SCOPE=$SCOPE \
  bash scripts/branchB/run_1p1b_train.sh "" "$PILOT_MAXTIME" "${SD_OVR[@]}" \
  > "run_logs/ov_${TS}_s5_pilot.log" 2>&1
echo "학습 exit=$?  GPU=$(gpu)"
grep -oE "[0-9.]+s/it" "run_logs/ov_${TS}_s5_pilot.log" | tail -3
grep -E "reached|stopped|OutOfMemory|Traceback" "run_logs/ov_${TS}_s5_pilot.log" | tail -3
echo "--- 체크포인트 목록 ---"; ls -lh "$CKDIR"/*.ckpt 2>/dev/null; ls -lh "$CKDIR"/trainstep_checkpoints/*.ckpt 2>/dev/null | tail -3

# ─────────────── 최종 평가: 홀드아웃 96 생성 + 채점 + 판정 ───────────────
PILOT_CKPT="$CKDIR/last.ckpt"
say "FINAL  홀드아웃 96 생성 + 채점"
if [ -f "$PILOT_CKPT" ]; then
  t0=$(date +%s)
  bash scripts/branchB/run_1p1b_generate.sh "$PILOT_CKPT" \
       artifacts/holdout artifacts/branchB/preds_pilot 50 1.0 \
       > "run_logs/ov_${TS}_final_gen.log" 2>&1
  t1=$(date +%s); n=$(ls artifacts/branchB/preds_pilot/*.mp4 2>/dev/null | wc -l)
  echo "생성 exit=$? / mp4=$n / $(( (t1-t0)/60 ))분"
  $CONDA run -n wm python scripts/run_m0.py \
      --holdout artifacts/holdout --out artifacts/branchB/m0_pilot \
      --pred-dir artifacts/branchB/preds_pilot --pred-name b1p1b_pilot \
      > "run_logs/ov_${TS}_final_score.log" 2>&1
  echo "채점 exit=$?"
  cat "run_logs/ov_${TS}_final_score.log" | grep -A8 "M0 바닥값"

  say "액션 민감도 (파일럿)"
  $CONDA run -n wm python scripts/branchB/probe_action_sensitivity.py \
      --device cuda --n-pairs 3 --ckpt "$PILOT_CKPT" --tag pilot \
      > "run_logs/ov_${TS}_final_sens.log" 2>&1
  grep -E "S_AB|S_A0|상대차|mean" "run_logs/ov_${TS}_final_sens.log" | tail -6

  say "판정 (012 §4: DINO 로만)"
  $CONDA run -n wm python - <<'PY'
import json, os
p = "/home/rils/dlacksdn/2026_Inha_AI_challenge_WM/artifacts/branchB/m0_pilot/m0_report.json"
if not os.path.exists(p):
    print("채점 리포트 없음"); raise SystemExit
d = json.load(open(p))
for name, r in d["results"].items():
    m = r["mean"]
    print(f"{name:16} DINO {m['dino_frame_avg']:.5f} | Video {m['video']:.5f} | "
          f"Action {m['action']:.5f} | TOTAL {m['total_frame_avg']:.5f}")
pil = d["results"].get("b1p1b_pilot")
if pil:
    dino = pil["mean"]["dino_frame_avg"]
    print(f"\n>>> 파일럿 DINO = {dino:.5f}   (static 0.123 / 11M재학습 0.481 / baseline11M 0.550)")
    if   dino < 0.123: v = "*** 생성이 정지를 이겼다 → 즉시 본 학습 준비(scope=full, 6000) ***"
    elif dino < 0.30:  v = "방향 맞음(11M 재학습보다 크게 좋음) → 학습 더 돌려 추세 확인 후 본 학습"
    elif dino < 0.48:  v = "사전학습 이득은 있으나 부족 → scope·lr·fs_condition A/B, anti-drift 검토"
    else:              v = "11M 재학습과 다를 바 없음 → 원인 규명(액션 민감도/데이터/lr) 후 branch A 재검토"
    print(f">>> 판정: {v}")
PY
else
  echo "!! 파일럿 체크포인트 없음 — 학습이 실패했다. 로그 확인: run_logs/ov_${TS}_s5_pilot.log"
fi

say "PIPELINE END"
echo "로그 접두사: run_logs/ov_${TS}_*"
