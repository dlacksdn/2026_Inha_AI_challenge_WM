#!/usr/bin/env bash
# 30,000 도달 후 45,000 까지 자동 연장 — ckpt 를 꺼서 1.31배 빠르게 (017)
#
# 🚨 GPU 0 만 쓴다 (CLAUDE.md).
# 🚨 내일 14:00 에 GPU 를 다음 사용자에게 넘겨야 한다. 13:45 에 스스로 멈춘다.
#
# 왜 ckpt 를 끄나  [측정 011 §2 격자]
#     micro 2 × accum 8 · ckpt 켬   4.76 s/step   19.6 GiB
#     micro 2 × accum 8 · ckpt 끔   3.63 s/step   31.8 GiB   ← ×1.31
#     micro 4 × accum 4 · ckpt 끔   3.78 s/step   63.0 GiB   ← VRAM 3배인데 더 느리다
#   ⇒ 31.8 GiB 가 최적이고 그 위는 못 쓴다. 96GB 를 다 쓰면 오히려 느려진다.
#   14:00 마감 기준 도달 스텝:  ckpt 켬 ≈40,800  vs  ckpt 끔 ≈44,000~45,000
#
# 무엇을 포기하나 (정직하게)
#   ckpt 는 backward 에서 forward 를 다시 돌려 BatchNorm 러닝통계를 micro-step 당
#   2회 갱신한다(011 §2 실측: 3스텝에 num_batches_tracked 6 vs 3). 끄면 1회다.
#   ⇒ 30k 이후가 나빠져도 "더 학습해서"인지 "BN 이 바뀌어서"인지 못 가른다.
#     GPU 를 내일 14:00 에 잃으므로 재시도 기회가 없고, 어느 쪽이든 할 일은
#     "가장 좋은 체크포인트를 쓴다" 하나뿐이라 이 구분의 값이 낮다고 판단했다.
#   ⇒ 그래서 **태그를 바꿔** 새 폴더에 쌓는다. ckpt-끔 구간이 눈으로 구분된다.
#
# 왜 지금이 아니라 30,000 에서 바꾸나
#   지금 바꾸면 1,400 스텝을 더 벌지만 25,000·30,000 이 오염된다.
#   그 둘은 ckpt-켬 곡선의 마지막 깨끗한 점이라 1,400 스텝보다 값이 크다.
#
# 나머지 인자는 원 런과 **한 글자도 다르지 않다** (005 치-7: 재개마다 손실이 달라지면 안 된다)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
SRC="$REPO/artifacts/branchC/train_20260809_1205_long/full_030000.pt"
DEADLINE="${DEADLINE:-2026-08-11 13:45}"
export CUDA_VISIBLE_DEVICES=0

echo "[extend] 대기 시작 $(date '+%F %T') — $SRC"
while [ ! -f "$SRC" ]; do sleep 120; done
sleep 90                                  # 저장이 끝나기를 기다린다
echo "[extend] full_030000.pt 확인 $(date '+%F %T')"

# 원 런이 완전히 끝나 GPU 를 놓을 때까지 기다린다 (두 학습이 겹치면 둘 다 느려진다)
while pgrep -f "train_c.py --tag long " > /dev/null; do sleep 60; done
echo "[extend] 원 런 종료 확인. 연장 시작 $(date '+%F %T')"

LOG="$REPO/run_logs/$(date +%Y%m%d_%H%M)_train_c_long45_nockpt.log"
"$PY" scripts/branchC/train_c.py \
  --tag long45 --steps 45000 --resume "$SRC" \
  --wake-step 2000 --micro-batch 2 --accum 8 \
  --dir-loss --tau-alpha 0.1 --lam-c 0.012 \
  --no-ckpt \
  2>&1 | tee "$LOG" &
TRAIN_PID=$!
echo "[extend] 학습 pid $TRAIN_PID · 로그 ${LOG#$REPO/}"

# 새 런 폴더가 생기면 CSV 감시자를 붙인다
sleep 120
NEWRUN="$(ls -td "$REPO"/artifacts/branchC/train_*_long45 2>/dev/null | head -1)"
if [ -n "$NEWRUN" ]; then
  echo "[extend] CSV 감시자 부착 → ${NEWRUN#$REPO/}"
  nohup bash "$REPO/scripts/branchC/watch_csv_generic.sh" "$NEWRUN" "35000 40000 45000" "0.1 0.15 0.25" \
    > "$REPO/run_logs/$(date +%Y%m%d_%H%M)_csv_watch_long45.log" 2>&1 &
fi

# 🚨 마감 감시 — 다음 사용자에게 GPU 를 넘겨야 한다
END=$(date -d "$DEADLINE" +%s)
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$END" ] && {
      echo "[extend] ⏰ 마감 $DEADLINE 도달. 학습을 멈춘다 (full_*.pt 는 1,000 스텝마다 있다)"
      pkill -f "train_c.py --tag long45"
      break; }
  sleep 60
done
wait "$TRAIN_PID" 2>/dev/null
echo "[extend] 종료 $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
