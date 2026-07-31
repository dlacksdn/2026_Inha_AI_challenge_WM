#!/usr/bin/env bash
# 체크포인트 스냅샷 데몬 — 롤링 저장이 지워버리기 전에 사본을 남긴다.
#
# 왜 필요한가 (013 §6.1)
#   학습 config 의 롤링 ModelCheckpoint 는 save_top_k=1 이라 **500스텝마다 이전 것을 지운다.**
#   학습곡선을 그리려면 중간 시점의 가중치가 있어야 하는데, 그대로 두면 마지막 하나만 남는다.
#
# 손상 파일 방지 (013 §6.1 의 step=1500.ckpt 2.7GB 사고)
#   PyTorch 가 7.5GB 를 쓰는 데 수십 초가 걸린다. 그 도중에 잡으면 **잘린 파일**이 생긴다.
#   그래서 "크기가 STABLE_CHECKS 회 연속 그대로일 때만" 스냅샷한다.
#   (2026-07-31 실증: 이 확인 덕분에 전원이 끊길 때 쓰다 만 368MB 파일이 스냅샷에 안 들어갔다.)
#
# 왜 복사가 아니라 하드링크인가 (2026-07-31 사고 이후 변경)
#   같은 파일시스템 안에서 하드링크는 **이름을 하나 더 붙이는 것**이지 데이터를 복제하는 게 아니다.
#   롤링 저장이 원본 이름을 지워도 우리 이름이 남아 있으면 데이터는 살아 있다. 그래서 목적은 그대로
#   달성하면서, 7.5GB 를 실제로 읽고 쓰지 않으므로
#     · 디스크를 추가로 쓰지 않는다(스냅샷 8개면 64GB 절약)
#     · 페이지 캐시를 채우지 않는다 → **메모리 압박이 생기지 않는다**
#   두 번째가 중요하다. 이 머신은 user@1000.service 에 ManagedOOMMemoryPressure=kill(임계 50%)이
#   걸려 있어서, 메모리 압박이 오래 지속되면 systemd-oomd 가 프로세스를 죽인다.
#   7.5GB 복사가 그 압박에 기여하고 있었다.
#   ※ 안전 조건: 파일 이름에 step 번호가 들어가 같은 이름이 재사용되지 않아야 한다(덮어쓰면 링크도 같이
#      바뀐다). epoch=*-step=*.ckpt 는 매번 새 이름이라 안전하고, last.ckpt 는 애초에 대상이 아니다.
#   다른 파일시스템이면 하드링크가 안 되므로 자동으로 복사로 되돌아간다.
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
      if ln "$f" "$SNAP/$b" 2>/dev/null; then
        echo "[snap] $(date +'%F %T')  $b  ($(( sz / 2**20 )) MiB) 하드링크"
      elif cp -a "$f" "$SNAP/.tmp_$b" && mv "$SNAP/.tmp_$b" "$SNAP/$b"; then
        echo "[snap] $(date +'%F %T')  $b  ($(( sz / 2**20 )) MiB) 복사(하드링크 불가)"
      else
        rm -f "$SNAP/.tmp_$b"
        echo "[snap] $(date +'%F %T')  !! $b 스냅샷 실패"
      fi
    fi
  done
  sleep "$INTERVAL"
done
