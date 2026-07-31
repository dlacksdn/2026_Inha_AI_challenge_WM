#!/usr/bin/env bash
# 추론 속도 벤치 — 대회 예산(216샘플 ≤ 1시간)에 들어가는 설정을 찾는다.
#
# 왜 필요한가 (013 §9)
#   현재 26초/샘플이다. 216샘플이면 94분으로 **예산을 34분 초과**한다.
#   게다가 학습 곡선을 그리려면 체크포인트마다 96샘플을 생성해야 하는데, 지금 속도면
#   체크포인트 하나에 42분이라 12개를 재는 데 8시간이 넘는다. 속도는 제출 문제이자 실험 속도 문제다.
#
# 무엇을 재나
#   조합(배치 크기 × DDIM 스텝)마다 16표본을 생성하고 **표본당 초**를 낸다.
#   모델 로딩(고정 1분 남짓)이 섞이지 않도록, 로그 줄마다 시각을 찍어
#   "첫 배치 완료 → 마지막 배치 완료" 구간만으로 속도를 계산한다.
#
# 주의
#   · 배치를 바꾸면 초기 노이즈 모양이 달라져 같은 시드라도 결과가 달라진다.
#     속도만 보는 도구이고, 점수 비교는 eval_ckpt.sh 로 따로 해야 한다.
#   · GPU 를 독점한다. 학습 중에는 돌리지 말 것(VRAM 이 없다).
#
# 사용: bash scripts/branchB/bench_inference.sh <ckpt> ["배치:스텝 배치:스텝 ..."]
set -uo pipefail
REPO=/home/rils/dlacksdn/2026_Inha_AI_challenge_WM
cd "$REPO" || exit 1
CKPT="${1:?체크포인트 경로}"
COMBOS="${2:-1:50 2:50 4:50 8:50 4:25}"
IN=artifacts/holdout_bench16
N=$(ls "$IN"/images/*.png | wc -l)

export PATH=/home/rils/dlacksdn/miniconda3/bin:$PATH
export CONDA_BIN=/home/rils/dlacksdn/miniconda3/bin/conda
export HF_HOME=/home/rils/dlacksdn/.cache/hf TORCH_HOME=/home/rils/dlacksdn/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== 추론 속도 벤치  ckpt=$(basename "$CKPT")  표본=$N  $(date +'%F %T') ====="
printf "%-12s %10s %12s %14s %12s\n" "배치:스텝" "총초" "표본당초" "216샘플(분)" "최대VRAM"
echo "---------------------------------------------------------------------------"

for c in $COMBOS; do
  B="${c%%:*}"; S="${c##*:}"
  OUT="artifacts/branchB/bench_b${B}_s${S}"
  rm -rf "$OUT"; mkdir -p "$OUT"
  LOG="run_logs/$(date +%Y%m%d_%H%M%S)_bench_b${B}_s${S}.log"

  ( bash scripts/branchB/run_1p1b_generate.sh "$CKPT" "$IN" "$OUT" "$S" 1.0 "$B" 2>&1 \
      | while IFS= read -r l; do echo "$(date +%s) $l"; done ) > "$LOG" 2>&1

  made=$(ls "$OUT"/*.mp4 2>/dev/null | wc -l)
  # "[generate] wrote predictions for" 줄들의 시각으로 순수 생성 구간을 잰다.
  read -r t_first t_last n_marks <<<"$(awk '/wrote predictions for/ {if(!f){f=$1} l=$1; c++} END{print f+0, l+0, c+0}' "$LOG")"
  if [ "${n_marks:-0}" -ge 2 ] && [ "$made" -gt 0 ]; then
    span=$(( t_last - t_first ))
    # 첫 마크는 이미 배치 1개를 만든 시점이므로, 그 뒤로 만들어진 표본 수로 나눈다.
    rest=$(( made - B ))
    if [ "$rest" -gt 0 ]; then
      per=$(awk -v s="$span" -v r="$rest" 'BEGIN{printf "%.2f", s/r}')
    else
      per="n/a"
    fi
  else
    span="n/a"; per="n/a"
  fi
  vram=$(grep -oE "[0-9]+ MiB" "$LOG" | tail -1)
  tot216=$(awk -v p="$per" 'BEGIN{ if(p+0>0) printf "%.1f", p*216/60; else print "n/a"}')
  printf "%-12s %10s %12s %14s %12s\n" "$B:$S" "${span:-n/a}" "$per" "$tot216" "${vram:-}"
  echo "   로그: $LOG  (mp4 $made 개 → $OUT)"
done
echo "---------------------------------------------------------------------------"
echo "대회 예산: 216샘플 ≤ 60분  → 표본당 16.7초 이하여야 한다"
echo "===== 벤치 완료 $(date +'%F %T') ====="
