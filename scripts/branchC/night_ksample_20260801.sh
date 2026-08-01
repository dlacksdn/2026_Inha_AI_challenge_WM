#!/usr/bin/env bash
# 야간 체인 — 확산 샘플 K개를 만들어 평균의 값어치를 잰다 (학습 없음, 동결 자산 재활용).
#
# 왜 하는가 (017 §4)
#   우리와 거의 같은 대회(1X World Model Challenge)의 우승팀이 확산모델을 버리지 않고
#   샘플 20개를 픽셀공간에서 평균내 PSNR +2.25dB 를 얻었다. K=1→5 가 이득의 84% 다.
#   우리 생성 config 는 이미 CFG 가 꺼져 있고(unconditional_guidance_scale: 1.0)
#   ddim_eta=0 이라 결정론적이므로, 샘플이 달라지는 유일한 원천이 **초기 노이즈**다.
#
# 무엇을 쓰는가
#   016 §9.1 은 branch B 의 **학습**을 동결했다. 추론까지 동결한 것은 아니다.
#   리더보드 0.35624 를 낸 바로 그 체크포인트(누적 1000)를 그대로 쓴다.
#
# 주의
#   - 배치를 바꾸면 초기 노이즈 텐서의 모양이 바뀌어 같은 시드라도 결과가 달라진다.
#     그래서 **모든 seed 에서 배치 4 로 고정**한다.
#   - 학습 중이 아니므로 5.7GB 체크포인트 쓰기가 없다. 016 §6 의 oomd 위험은 낮다.
#     그래도 **브라우저는 띄우지 마라.**
#   - eval 데이터는 건드리지 않는다. 홀드아웃(train 유래)만 쓴다.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

CKPT="${CKPT:-$REPO/artifacts/branchB/ckpt_snapshots/epoch=0-step=1000.ckpt}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-50}"
BATCH="${BATCH:-4}"
OUTBASE="$REPO/artifacts/branchC/ksample"
STAMP="$(date +%Y%m%d_%H%M)"

[ -f "$CKPT" ] || { echo "ERROR: ckpt 없음: $CKPT"; exit 1; }
echo "############ K샘플 체인 시작 $(date +'%F %T')"
echo "ckpt=$CKPT  seeds=[$SEEDS]  steps=$STEPS  batch=$BATCH"
echo "출력=$OUTBASE"

ROOTS=()
for S in $SEEDS; do
  OUT="$OUTBASE/seed$S"
  N=$(ls "$OUT"/*.mp4 2>/dev/null | wc -l)
  if [ "$N" -ge 96 ]; then
    echo "---- seed=$S 이미 96개 있음. 건너뛴다 ($OUT)"
  else
    echo "---- seed=$S 생성 시작 $(date +'%F %T')"
    bash scripts/branchB/run_1p1b_generate.sh \
        "$CKPT" artifacts/holdout "$OUT" "$STEPS" 1.0 "$BATCH" "$S" \
        > "run_logs/${STAMP}_gen_holdout_seed${S}.log" 2>&1
    echo "---- seed=$S 완료 $(date +'%F %T')  mp4 $(ls "$OUT"/*.mp4 | wc -l)개"
  fi
  ROOTS+=("$OUT")
done

echo "############ 채점 $(date +'%F %T')"
# ⚠ 파일 존재 ≠ 완성. 채점 스크립트가 개수를 검사한다(016 함정: 216개 중 24개만 읽고 가짜 판정).
source /home/rils/dlacksdn/miniconda3/etc/profile.d/conda.sh
conda activate wm
python scripts/branchC/score_ksample_average.py \
    --pred-roots "${ROOTS[@]}" \
    --out "$REPO/results/branchC/ksample_average.json"

echo "############ 체인 완료 $(date +'%F %T')"
