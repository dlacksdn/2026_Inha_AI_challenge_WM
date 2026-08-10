#!/usr/bin/env bash
# 로더 수정(2026-08-11) 후 학습 재개 — 데이터 2.14배 · 홀드아웃 누수 0
#
# 🚨 GPU 0 만.  🚨 13:45 자동 정지 (다음 사용자에게 14:00 반납)
#
# 무엇이 바뀌었나 [측정 020 §2·§3, 이 세션이 재확인]
#     에피소드   5,071 → 10,860   (로더 break 두 개가 소유자별 첫 데이터셋만 쓰고 있었다)
#     데이터셋      54 → 121
#     홀드아웃 누수  44개 → 0개    (경로 접두사 불일치로 exclude 가 0건 작동했다)
#
# 무엇을 **안** 바꿨나 — 일부러다
#     행동 통계(action_stats.json)를 재계산하지 않는다.
#     런 도중에 정규화가 바뀌면 FiLM 입력 스케일이 통째로 달라져 Adam 모멘트와
#     BatchNorm 러닝통계가 흔들린다. 54개로 만든 통계가 약간 편향이어도
#     중간에 갈아끼우는 위험이 더 크다. 바꾸는 것은 데이터 범위와 누수 차단 둘뿐이다.
#
# ⚠ 읽는 법 — 미리 박아 둔다
#     누수가 사라졌으므로 **감시 코사인이 오히려 떨어져 보인다.** 그건 정상이다.
#     이전 값은 "학습에 들어간 44표본"이 섞인 낙관적 수치였다.
#     ⇒ 수정의 성패는 **리더보드로만** 판정한다. 코사인 하락을 실패로 읽지 마라.
#
# 태그를 long45f 로 바꾼다 — 수정 데이터 구간을 폴더째 분리해 나중에 구분되게.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
SRC="${1:?재개할 full_*.pt 경로를 달라}"
DEADLINE="${DEADLINE:-2026-08-11 13:45}"
export CUDA_VISIBLE_DEVICES=0

LOG="$REPO/run_logs/$(date +%Y%m%d_%H%M)_train_c_long45f_fixeddata.log"
echo "[fixed] 재개점 ${SRC#$REPO/}"
echo "[fixed] 로그    ${LOG#$REPO/}"
echo "[fixed] 마감    $DEADLINE"

"$PY" scripts/branchC/train_c.py \
  --tag long45f --steps 45000 --resume "$SRC" \
  --wake-step 2000 --micro-batch 2 --accum 8 \
  --dir-loss --tau-alpha 0.1 --lam-c 0.012 \
  --no-ckpt --no-viz \
  2>&1 | tee "$LOG" &
TRAIN_PID=$!
echo "[fixed] 학습 pid $TRAIN_PID"

sleep 150
NEWRUN="$(ls -td "$REPO"/artifacts/branchC/train_*_long45f 2>/dev/null | head -1)"
if [ -n "$NEWRUN" ]; then
  echo "[fixed] CSV 감시자 부착 → ${NEWRUN#$REPO/}"
  nohup bash "$REPO/scripts/branchC/watch_csv_generic.sh" "$NEWRUN" \
    "34000 36000 38000 40000 42000" "0.1 0.15 0.25" \
    > "$REPO/run_logs/$(date +%Y%m%d_%H%M)_csv_watch_long45f.log" 2>&1 &
fi

END=$(date -d "$DEADLINE" +%s)
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$END" ] && {
      echo "[fixed] ⏰ 마감 도달. 학습 정지 (full_*.pt 는 1,000 스텝마다 있다)"
      pkill -f "train_c.py --tag long45f"; break; }
  sleep 60
done
wait "$TRAIN_PID" 2>/dev/null
echo "[fixed] 종료 $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
