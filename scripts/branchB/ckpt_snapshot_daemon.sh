#!/usr/bin/env bash
# 체크포인트 스냅샷 데몬 — 롤링 저장이 지워버리기 전에 사본을 남긴다.
#
# 왜 필요한가 (013 §6.1)
#   학습 config 의 롤링 ModelCheckpoint 는 save_top_k=1 이라 **500스텝마다 이전 것을 지운다.**
#   학습곡선을 그리려면 중간 시점의 가중치가 있어야 하는데, 그대로 두면 마지막 하나만 남는다.
#
# 손상 파일 방지 (013 §6.1 의 step=1500.ckpt 2.7GB 사고)
#   PyTorch 가 7.5GB 를 쓰는 데 수십 초가 걸린다. 그 도중에 복사하면 **잘린 파일**이 생긴다.
#   그래서 "크기가 STABLE_CHECKS 회 연속 그대로일 때만" 복사한다.
#
# 사용:
#   bash scripts/branchB/ckpt_snapshot_daemon.sh <감시디렉터리> <스냅샷디렉터리> [주기초]
set -uo pipefail
WATCH="${1:?감시할 checkpoints 디렉터리}"
SNAP="${2:?스냅샷을 쌓을 디렉터리}"
INTERVAL="${3:-60}"
STABLE_CHECKS=2          # 크기가 이 횟수만큼 연속 같아야 "다 쓴 파일"로 본다
mkdir -p "$SNAP"

declare -A SEEN_SIZE SEEN_CNT
echo "[snap] 감시=$WATCH → 스냅샷=$SNAP (주기 ${INTERVAL}s)  $(date +'%F %T')"

while true; do
  for f in "$WATCH"/epoch=*-step=*.ckpt; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    [ -e "$SNAP/$b" ] && continue                       # 이미 스냅샷 있음
    sz="$(stat -c %s "$f" 2>/dev/null || echo 0)"
    [ "$sz" -gt 0 ] || continue
    if [ "${SEEN_SIZE[$b]:-0}" = "$sz" ]; then
      SEEN_CNT[$b]=$(( ${SEEN_CNT[$b]:-0} + 1 ))
    else
      SEEN_SIZE[$b]="$sz"; SEEN_CNT[$b]=0
    fi
    if [ "${SEEN_CNT[$b]}" -ge "$STABLE_CHECKS" ]; then
      cp -a "$f" "$SNAP/.tmp_$b" && mv "$SNAP/.tmp_$b" "$SNAP/$b" \
        && echo "[snap] $(date +'%F %T')  $b  ($(( sz / 2**20 )) MiB) 저장"
    fi
  done
  sleep "$INTERVAL"
done
