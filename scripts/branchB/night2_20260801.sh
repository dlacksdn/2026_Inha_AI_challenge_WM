#!/usr/bin/env bash
# 2026-08-01 새벽 체인 — 015 적대적 검수의 최우선 지적 두 개를 실행에 옮긴다.
#
# 왜 이 두 가지인가
# ------------------
# 015 가 "치명적"으로 분류한 지적 중 오늘 밤 GPU 로 답할 수 있는 것이 정확히 두 개다.
#
#   [1단계] 치명-1 "좌표계가 없다"
#       우리는 지금까지 로컬 홀드아웃 96개만 보고 모든 결정을 내렸다. 그런데 유일한
#       리더보드 측정(static 제출, 011 §1)이 이미 보여줬다 — eval 에서는 Action 이 0.35배로
#       쉬워지고 DINO+Video 가 2.05배로 어려워진다. **로컬과 eval 은 다른 좌표계다.**
#       그러므로 "로컬에서 static 을 못 이긴다"는 우리의 모든 판단은 검증된 적이 없다.
#
#       여기서 중요한 사실 하나: `make_submission_csv.py` 는 Action 성분을 **완전히 로컬에서**
#       계산해 CSV 에 표본별로 적어 넣는다. 즉 **제출하지 않고도 배점 40% 축의 실제
#       리더보드 값을 알 수 있다.** static 의 값(0.42874)이 이미 있으므로 바로 비교된다.
#       나머지 60%(DINO+Video)만 서버가 쥐고 있고, 그건 사용자가 CSV 를 올리면 나온다.
#
#   [2단계] 치명-2 "고정 seed 재생"
#       모든 학습이 seed_everything(20230211) 로 시작해 **같은 데이터 순서·같은 타임스텝·
#       같은 노이즈**를 처음부터 재생하고 있었다. 검증: 학습률이 3배 다른 두 런의 스텝별
#       loss 상관이 r=0.9973. 어느 런도 epoch 을 못 끝냈으므로 데이터의 앞 45%만 같은
#       노이즈로 2~3회 반복했고 뒤 55%는 한 번도 안 봤다.
#
#       확산모델 학습의 정규화는 "같은 그림에 매번 다른 노이즈를 씌워 본다"에서 나온다.
#       그게 고정돼 있으면 모델은 일반화 대신 특정 노이즈 실현을 외운다 — loss 는 평탄한데
#       생성 품질이 무너지는 지금 패턴과 정확히 맞아떨어진다.
#
#       이번 실험은 **오직 seed 하나만** 바꾼 완전 대조 실험이다. 출발점·학습률·스텝 수가
#       전부 같으므로, 결과 차이는 seed 말고 설명할 것이 없다.
#
# 판정 기준 (결과를 보기 전에 고정했다 — 015 §7-3)
# ------------------------------------------------
#   출발점 = 누적 1000 스텝: DINO 0.22825 / TOTAL 0.58105
#   통과 : DINO < 0.22825 이고 짝지은 t ≤ -2   → 그 설정으로 본 학습 재개
#   미달 : DINO < 0.22825 이지만 유의하지 않음 → 1.1B 추가 학습 동결
#   실패 : DINO >= 0.22825                     → 1.1B 추가 학습 동결, 결정론 회귀(015 §8-C)로
#
#   ※ lr 3e-5(고정 seed)는 이 선에서 이미 실패했다: DINO 0.26430 (누적 3250)
#
# 사용:
#   bash scripts/branchB/night2_20260801.sh <학습률>
# 예:
#   bash scripts/branchB/night2_20260801.sh 3e-5
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

LR="${1:?학습률 (예: 3e-5)}"
SEED="${2:-777}"              # 20230211 이 아니기만 하면 된다
START_CKPT="$REPO/artifacts/branchB/ckpt_snapshots/epoch=0-step=1000.ckpt"
EVAL_PRED="$REPO/artifacts/branchB/preds_eval216_cum1000"
CSV_NAME="20260801_cum1000_b4_s50.csv"

[ -f "$START_CKPT" ] || { echo "ERROR: 출발 ckpt 없음: $START_CKPT"; exit 1; }

echo "=========================================================================="
echo "[야간2] 시작 $(date +'%F %T')"
echo "[야간2] 1단계 = eval 216 생성 + 제출 CSV (좌표계 확보)"
echo "[야간2] 2단계 = seed 반증 실험  lr=$LR  seed=$SEED  (고정 seed 재생 검증)"
echo "=========================================================================="

# ── GPU 가 빌 때까지 기다린다 (앞 체인의 평가가 끝나야 한다) ──────────────────
echo "[야간2] GPU 대기 중..."
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-99999}" -lt 4000 ] && break
  sleep 60
done
echo "[야간2] GPU 확보 (사용 ${used}MiB)  $(date +'%F %T')"

# ── 1단계: eval 216 생성 → 제출 CSV ──────────────────────────────────────────
# eval 데이터는 **추론 입력으로만** 쓴다. 학습·데이터선별에는 일절 쓰지 않는다(대회 규칙 §4.1-3).
echo ""
echo "############ 1단계: eval 216 생성 (cum1000, 배치4, 50스텝) $(date +'%F %T')"
bash scripts/branchB/run_1p1b_generate.sh \
    "$START_CKPT" open/data/eval "$EVAL_PRED" 50 1.0 4 \
    > "run_logs/$(date +%Y%m%d_%H%M)_gen_eval216_cum1000.log" 2>&1
echo "[야간2] 생성 rc=$?  mp4 $(ls "$EVAL_PRED"/*.mp4 2>/dev/null | wc -l)개  $(date +'%F %T')"

echo "############ 1단계-b: 제출 CSV 생성 (공식 make_submission_csv.py 무수정 호출)"
SUBLOG="run_logs/$(date +%Y%m%d_%H%M)_submission_cum1000.log"
bash scripts/make_submission.sh "$EVAL_PRED" "$CSV_NAME" > "$SUBLOG" 2>&1
echo "[야간2] CSV rc=$?  $(date +'%F %T')"
grep -E "eval Action Component|CSV 생성 완료|행 구성" "$SUBLOG" 2>/dev/null || tail -5 "$SUBLOG"

# 대조군: 같은 방식으로 static 의 eval CSV 도 만든다.
# Action(배점 40%)은 정답 영상 없이 계산되므로, 이 두 CSV 를 표본별로 짝지으면
# **제출하지 않고도** 리더보드 40% 축에서 우리 모델과 static 의 승부가 확정된다.
echo "############ 1단계-c: static 대조군 CSV (리더보드 40% 축 짝지은 비교용)"
STATICLOG="run_logs/$(date +%Y%m%d_%H%M)_submission_static.log"
bash scripts/make_submission.sh "$REPO/artifacts/branchB/preds_eval216_static" \
    "20260801_static_eval216.csv" > "$STATICLOG" 2>&1
echo "[야간2] static CSV rc=$?  $(date +'%F %T')"
grep -E "eval Action Component" "$STATICLOG" 2>/dev/null || tail -5 "$STATICLOG"

echo "############ 1단계-d: eval Action 짝지은 비교 (n=216)"
python3 scripts/branchB/compare_eval_action.py \
    "artifacts/submission/$CSV_NAME" "artifacts/submission/20260801_static_eval216.csv" \
    2>&1 | tee "run_logs/$(date +%Y%m%d_%H%M)_eval_action_paired.log"

# ── 2단계: seed 반증 실험 ────────────────────────────────────────────────────
echo ""
echo "############ 2단계: seed 반증 실험  lr=$LR seed=$SEED  $(date +'%F %T')"
echo "[야간2]   대조군 = 같은 출발점·같은 lr·같은 스텝수, seed 20230211 (이미 측정됨)"
echo "[야간2]   실험군 = seed $SEED  — 이것 하나만 다르다"
bash scripts/branchB/overnight_train_v2.sh \
    "$START_CKPT" 1000 3.5 "seed${SEED}" \
    "model.base_learning_rate=$LR" --seed "$SEED"

echo ""
echo "=========================================================================="
echo "[야간2] 전부 완료 $(date +'%F %T')"
echo "[야간2] 제출 CSV : artifacts/submission/$CSV_NAME"
echo "[야간2] 실험 결과: artifacts/branchB/m0_cum3250_seed${SEED}/m0_report.json"
echo "=========================================================================="
