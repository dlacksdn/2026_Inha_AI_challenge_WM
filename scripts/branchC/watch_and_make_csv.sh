#!/usr/bin/env bash
# 학습이 목표 스텝에 닿을 때마다 추론 → CSV 를 자동으로 만들어 둔다 (015 §4 ③ 준비)
#
#   학습은 멈추지 않는다. 추론 99초 + CSV 60초×λ 는 GPU 0 에 같이 얹어도 된다 [측정 015 §3]
#   제출은 하지 않는다 — 이 스크립트는 **후보를 준비만** 한다. 제출은 사용자만 한다
#
# 실행 순서는 rule §12.5 를 지킨다:
#   ① 우리 추론 코드만으로 mp4 생성  ② 216개인지 확인해 확정  ③ 킷을 수정 없이 실행  ④ CSV
#
# 사용:
#   tmux new-session -d -s csvwatch \
#     "bash scripts/branchC/watch_and_make_csv.sh 2>&1 | tee run_logs/$(date +%Y%m%d_%H%M)_csv_watch.log"
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$REPO/artifacts/branchC/train_20260809_1205_long"
KIT="$REPO/artifacts/submission_kit"
PY="$REPO/.venv/bin/python"
LAMS="0.1 0.15 0.25"          # G6 승자가 무엇이든 덮는다 — 결과를 보고 고르지 않는다
TARGETS="20000 25000 30000"

echo "[watch] 시작 $(date '+%F %T')  대상 step: $TARGETS  λ: $LAMS"

for STEP in $TARGETS; do
  CK="$RUN/ck_$(printf '%06d' "$STEP").pt"
  echo "[watch] step $STEP 대기 — $CK"
  while [ ! -f "$CK" ]; do sleep 120; done
  sleep 60                                    # 저장 중간에 읽지 않도록 여유
  echo "[watch] $(date '+%F %T')  step $STEP 도착. 추론 시작"

  CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO/scripts/branchC/infer_c.py" \
      --ckpt "$CK" --lam $LAMS > "$REPO/run_logs/$(date +%Y%m%d_%H%M)_infer_step$(printf '%06d' "$STEP").log" 2>&1
  if [ $? -ne 0 ]; then echo "[watch] ❌ step $STEP 추론 실패. 다음으로 넘어간다"; continue; fi

  OUT="$(ls -td "$REPO"/artifacts/branchC/infer_* | head -1)"
  echo "[watch] 추론 완료 → $OUT"

  for L in $LAMS; do
    N=$(ls -1 "$OUT/lam$L"/*.mp4 2>/dev/null | wc -l)
    if [ "$N" -ne 216 ]; then echo "[watch] ❌ lam$L mp4 $N개 (216 아님). 킷을 돌리지 않는다"; continue; fi
    echo "[watch] lam$L mp4 216개 확정 → 킷 실행"
    ( cd "$KIT" && CUDA_VISIBLE_DEVICES=0 "$PY" make_submission_csv.py \
        --prediction-root "$OUT/lam$L" \
        --challenge-root "$REPO/open/data/eval" \
        --output-csv "$REPO/results/submission/lam${L}_step$(printf '%06d' "$STEP").csv" \
        --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
        --action-extractor-ckpt "$KIT/checkpoints/action_extractor.ckpt" ) 2>&1 | tail -1
  done
  echo "[watch] ✅ step $STEP 후보 준비 완료 $(date '+%F %T')"
done

echo "[watch] 전부 끝났다 $(date '+%F %T')"
