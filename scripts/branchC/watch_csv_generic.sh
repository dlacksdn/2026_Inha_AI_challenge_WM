#!/usr/bin/env bash
# 범용 CSV 감시자 — 지정한 런 폴더에서 목표 스텝 체크포인트가 나올 때마다 추론 → CSV
#
#   watch_and_make_csv.sh 의 일반화. 저건 실행중이라 건드리지 않는다(bash 가 스크립트를
#   실행 도중 읽기 때문에 수정하면 깨진다). 새 파일로 만든다.
#
#   제출은 하지 않는다. 후보를 준비만 한다. rule §12.5 순서 준수:
#   ① 우리 추론 코드로 mp4  ② 216개 확인해 확정  ③ 킷을 수정 없이 실행  ④ CSV
#
# 사용:  bash watch_csv_generic.sh <런폴더> "<스텝들>" "<λ들>"
#   예:  bash watch_csv_generic.sh artifacts/branchC/train_..._long45 "35000 40000 45000" "0.1 0.15 0.25"
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$(cd "$1" && pwd)"
TARGETS="${2:-35000 40000 45000}"
LAMS="${3:-0.1 0.15 0.25}"
KIT="$REPO/artifacts/submission_kit"
PY="$REPO/.venv/bin/python"

echo "[watch] 시작 $(date '+%F %T')  런 $RUN  스텝 $TARGETS  λ $LAMS"

for STEP in $TARGETS; do
  CK="$RUN/ck_$(printf '%06d' "$STEP").pt"
  echo "[watch] step $STEP 대기 — $CK"
  while [ ! -f "$CK" ]; do
    sleep 120
    # 학습이 이미 끝났고 그 스텝이 영영 안 나오면 빠져나온다
    if [ ! -f "$RUN/history.json" ]; then continue; fi
  done
  sleep 60                                    # 저장 도중에 읽지 않도록 여유

  echo "[watch] $(date '+%F %T')  step $STEP 도착. 추론 시작"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO/scripts/branchC/infer_c.py" \
      --ckpt "$CK" --lam $LAMS \
      > "$REPO/run_logs/$(date +%Y%m%d_%H%M)_infer_step$(printf '%06d' "$STEP").log" 2>&1 \
    || { echo "[watch] ❌ step $STEP 추론 실패"; continue; }

  OUT="$(ls -td "$REPO"/artifacts/branchC/infer_* | head -1)"
  for L in $LAMS; do
    N=$(ls -1 "$OUT/lam$L"/*.mp4 2>/dev/null | wc -l)
    [ "$N" -eq 216 ] || { echo "[watch] ❌ lam$L mp4 $N개. 킷을 돌리지 않는다"; continue; }
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
