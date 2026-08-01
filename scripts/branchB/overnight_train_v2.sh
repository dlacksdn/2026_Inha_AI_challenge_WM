#!/usr/bin/env bash
# 밤샘 학습 파이프라인 v2 — 죽어도 스스로 이어서 계속한다.
#
# 왜 v2 인가 (2026-07-31 하루에 두 번 죽었다)
# ------------------------------------------
#   04:55  1차 학습이 setsid nohup 상태에서 사망 (계획 5h 중 2h16m 만 학습)
#   18:08  2차 학습이 tmux 상태에서 사망 (계획 5h 중 3h51m 만 학습)
# 두 번 다 에러 한 줄 없이 진행 표시줄 중간에서 끊겼다. 조사 결과 재부팅도, 로그아웃도,
# 커널 OOM 도 아니었고(user@1000.service 는 07-28 부터 무중단), 확실한 것은 하나다:
#   18:09:48 사용자 슬라이스 메모리 압박이 90.82% > 임계 50% 까지 갔다.
# 즉 **메모리 압박 상황에서 프로세스가 정리된다.** 사람이 지켜보지 않는 밤에는 반드시 다시 일어난다.
#
# 그래서 두 갈래로 대응한다.
#   (가) 압박을 덜 만든다 — 저장 1회 쓰기량을 15.9GB → 5.9GB 로 줄인다.
#        · EMA 끄기        : 생성에서 use_ema=False 라 EMA 는 계산만 하고 안 쓰인다. 순수 낭비였다.
#                            EMA 는 가중치의 그림자 사본일 뿐이라 **학습 궤적은 전혀 달라지지 않는다.**
#        · save_last=False : 저장할 때마다 step 파일과 last.ckpt 를 각각 쓰고 있었다. 하나면 된다.
#                            (재개는 스냅샷에서 하므로 last.ckpt 가 없어도 무방하다)
#        · 스냅샷 하드링크 : 이미 적용됨. 7.5GB 복사가 사라졌다.
#   (나) 죽으면 다시 일어난다 — 가장 최근 스냅샷을 찾아 거기서부터 이어 학습한다.
#        세그먼트마다 출력 디렉터리를 나눠 기존 체크포인트를 절대 덮어쓰지 않는다.
#
# 사용:
#   bash scripts/branchB/overnight_train_v2.sh <시작ckpt> <시작누적스텝> <총시간(시간)> <런접두사>
# 예:
#   bash scripts/branchB/overnight_train_v2.sh \
#        artifacts/branchB/ckpt_snapshots_inha_action_diffusion_1p1b_r2/epoch=0-step=2000.ckpt \
#        3000 9.5 r3
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

START_CKPT_IN="${1:?시작 체크포인트}"
START_CUM="${2:?시작 누적 스텝(그림 x축에 쓸 값)}"
TOTAL_HOURS="${3:-9.5}"
PREFIX="${4:-r3}"
shift 4 2>/dev/null || true
EXTRA_OVERRIDES=("$@")   # 예: model.base_learning_rate=3e-5 --seed 999

# ── 세그먼트마다 seed 를 달리한다 (2026-08-01 확정된 붕괴 원인 대응) ──────────
# 학습 진입점은 seed_everything(seed) 로 시작하고, 그 seed 가 데이터 방문 순서·확산
# 타임스텝 t·노이즈 ε 를 전부 결정한다. 그런데 어떤 런도 epoch 을 못 끝내므로
# **매번 같은 앞부분 45% 를 같은 노이즈로 다시 본다.** 그러면 모델은 일반화 대신
# 특정 노이즈 실현을 외우고, 미는 만큼 망가진다.
#   실측(같은 출발점·같은 lr·같은 스텝수, seed 만 다름):
#     고정 seed 20230211 → DINO 0.26430 / seed 777 → 0.22726  (차이 t=-5.80)
# 재개(resume)는 이 함정을 되살리기 딱 좋다 — 같은 seed 로 다시 시작하면 방금 본
# 데이터를 처음부터 또 본다. 그래서 세그먼트 번호를 seed 에 더한다.
# 사용자가 --seed 를 주면 그 값을 기준으로 삼고, 안 주면 여기서 정한다.
BASE_SEED=""
for ((i = 0; i < ${#EXTRA_OVERRIDES[@]}; i++)); do
  if [ "${EXTRA_OVERRIDES[$i]}" = "--seed" ] && [ $((i + 1)) -lt ${#EXTRA_OVERRIDES[@]} ]; then
    BASE_SEED="${EXTRA_OVERRIDES[$((i + 1))]}"
    unset 'EXTRA_OVERRIDES[i]' 'EXTRA_OVERRIDES[i+1]'
    EXTRA_OVERRIDES=("${EXTRA_OVERRIDES[@]}")   # 인덱스 재정렬
    break
  fi
done
[ -n "$BASE_SEED" ] || BASE_SEED=$(( 1000 + RANDOM % 9000 ))

START_CKPT="$(cd "$(dirname "$START_CKPT_IN")" && pwd)/$(basename "$START_CKPT_IN")"
[ -f "$START_CKPT" ] || { echo "ERROR: 시작 ckpt 없음: $START_CKPT"; exit 1; }

DEADLINE=$(awk -v h="$TOTAL_HOURS" 'BEGIN{printf "%d", systime() + h*3600}')
MIN_SEGMENT=1200        # 남은 시간이 이보다 짧으면 새 세그먼트를 시작하지 않는다(20분)
MAX_SEGMENTS=12         # 폭주 방지
STATE="$REPO/artifacts/branchB/overnight_${PREFIX}_state.json"

echo "=========================================================================="
echo "[야간] 시작        $(date +'%F %T')"
echo "[야간] 시작 ckpt   $START_CKPT"
echo "[야간] 시작 누적   $START_CUM 스텝"
echo "[야간] 마감        $(date -d "@$DEADLINE" +'%F %T')  (총 ${TOTAL_HOURS}시간)"
echo "[야간] 런 접두사   $PREFIX"
echo "[야간] 추가 설정   ${EXTRA_OVERRIDES[*]:-<없음>}"
echo "=========================================================================="

# 저장 부담을 줄이는 덮어쓰기 3종 (위 (가) 참고)
LOWMEM_OVERRIDES=(
  model.params.use_ema=False
  lightning.callbacks.model_checkpoint.params.save_last=False
  lightning.callbacks.model_checkpoint.params.every_n_train_steps=250
)

seg=0
cur_ckpt="$START_CKPT"
cur_cum="$START_CUM"
declare -a SEG_LOG

while :; do
  now=$(date +%s)
  remain=$(( DEADLINE - now ))
  if [ "$remain" -lt "$MIN_SEGMENT" ]; then
    echo "[야간] 남은 시간 ${remain}초 < ${MIN_SEGMENT}초 → 학습 종료"
    break
  fi
  seg=$(( seg + 1 ))
  if [ "$seg" -gt "$MAX_SEGMENTS" ]; then
    echo "[야간] 세그먼트 상한 $MAX_SEGMENTS 도달 → 중단"
    break
  fi

  RUN="${PREFIX}_s${seg}"
  # 남은 시간을 DD:HH:MM:SS 로 (5분 여유를 빼서 마감을 넘기지 않게 한다)
  budget=$(( remain - 300 )); [ "$budget" -lt 600 ] && budget=600
  MT=$(printf "%02d:%02d:%02d:%02d" $((budget/86400)) $((budget%86400/3600)) $((budget%3600/60)) $((budget%60)))
  LOG="run_logs/$(date +%Y%m%d_%H%M)_train_${RUN}.log"

  echo ""
  echo "--------------------------------------------------------------------------"
  echo "[야간] 세그먼트 $seg  시작 $(date +'%F %T')"
  echo "[야간]   출발 ckpt : $cur_ckpt"
  echo "[야간]   출발 누적 : $cur_cum 스텝"
  echo "[야간]   시간 예산 : $MT"
  echo "[야간]   로그      : $LOG"
  echo "--------------------------------------------------------------------------"

  SEG_SEED=$(( BASE_SEED + seg - 1 ))    # 세그먼트마다 다른 데이터 스트림
  echo "[야간]   seed      : $SEG_SEED  (기준 $BASE_SEED + 세그먼트 $seg)"
  bash scripts/branchB/run_1p1b_resume.sh "$cur_ckpt" "$RUN" "$MT" \
      "${LOWMEM_OVERRIDES[@]}" ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
      --seed "$SEG_SEED" > "$LOG" 2>&1
  rc=$?
  SEG_LOG+=("$RUN rc=$rc $LOG")
  echo "[야간] 세그먼트 $seg 종료 rc=$rc  $(date +'%F %T')"

  # 이 세그먼트가 남긴 스냅샷 중 가장 큰 step 을 찾는다.
  SNAP="$REPO/artifacts/branchB/ckpt_snapshots_${RUN}"
  last_step=$(ls "$SNAP" 2>/dev/null | sed -n 's/.*step=\([0-9]\+\)\.ckpt/\1/p' | sort -n | tail -1)
  if [ -z "$last_step" ]; then
    echo "[야간] !! 세그먼트 $seg 이 스냅샷을 하나도 남기지 못했다. 같은 지점에서 재시도하지 않고 중단한다."
    echo "[야간]    (같은 조건으로 반복하면 같은 이유로 또 실패할 가능성이 높다 — 로그를 봐야 한다)"
    break
  fi
  cur_ckpt="$SNAP/epoch=0-step=${last_step}.ckpt"
  cur_cum=$(( cur_cum + last_step ))
  echo "[야간] 이어받을 지점: $cur_ckpt  (누적 $cur_cum 스텝)"

  cat > "$STATE" <<JSON
{
  "prefix": "$PREFIX",
  "segments_done": $seg,
  "last_ckpt": "$cur_ckpt",
  "cumulative_steps": $cur_cum,
  "updated": "$(date +'%F %T')"
}
JSON

  # 정상 종료(시간 소진)면 루프의 남은시간 검사가 알아서 끝낸다. 비정상이면 계속 이어간다.
  if [ "$rc" -eq 0 ]; then
    echo "[야간] 정상 종료로 보인다(rc=0). 남은 시간이 있으면 이어서 더 학습한다."
  else
    echo "[야간] 비정상 종료(rc=$rc). 최신 스냅샷에서 이어서 재시작한다."
  fi
done

echo ""
echo "=========================================================================="
echo "[야간] 학습 단계 종료  $(date +'%F %T')"
echo "[야간] 최종 누적: $cur_cum 스텝 / 최종 ckpt: $cur_ckpt"
for s in "${SEG_LOG[@]}"; do echo "[야간]   $s"; done
echo "=========================================================================="

# ── 아침에 볼 수 있도록 자동 평가 ─────────────────────────────────────────────
echo "[야간] 최종 체크포인트 평가 시작  $(date +'%F %T')"
TAG="cum${cur_cum}_${PREFIX}"
bash scripts/branchB/eval_ckpt.sh "$cur_ckpt" "$TAG" 50 1.0 4 \
    > "run_logs/$(date +%Y%m%d_%H%M)_eval_${TAG}.log" 2>&1
echo "[야간] 평가 종료 rc=$?  $(date +'%F %T')"
echo "[야간] 결과: artifacts/branchB/m0_${TAG}/m0_report.json"
echo "[야간] 전부 완료  $(date +'%F %T')"
