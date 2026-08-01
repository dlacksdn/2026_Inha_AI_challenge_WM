#!/usr/bin/env bash
# 2026-08-01 15:15 사망 후 재개 — lr 1e-4 + seed 999 실험을 이어서 끝낸다.
#
# 무슨 일이 있었나
# ----------------
#   13:43  lr 1e-4 + seed 999 학습 시작 (누적 1000 에서 출발, 3.5시간 예정)
#   15:13  step=1000 체크포인트 기록 완료 (5.76GB)
#   15:15  systemd-oomd 가 압박 74.11% > 50% 로 프로세스를 정리. tmux 세션째 사라짐
#          └ 같은 순간 브라우저(Chromium)도 죽었다고 저널에 남아 있다.
#            체크포인트 쓰기(5.76GB) + 브라우저가 겹쳐 임계를 넘긴 것으로 본다.
#            밤새 브라우저 없이 돈 3.5시간짜리 학습은 무사했다.
#
#   다행히 step=1000 체크포인트는 **온전했다**(바이트 크기·키 수·파라미터 수 전부 일치).
#   스냅샷 데몬이 "크기가 2회 연속 같아야 잡는다"는 규칙 때문에 아직 안 챙겼을 뿐이라,
#   수동으로 하드링크해 확보했다. 잃은 것은 26스텝뿐이다.
#
# 이 체인이 하는 일
# -----------------
#   1단계  구조한 누적 2000 체크포인트를 평가한다.
#          ★ 이 지점에는 **정확히 짝이 맞는 대조군**이 이미 있다:
#            m0_cum2000_b4 = 같은 lr(1e-4)·같은 출발점·같은 스텝수, **고정 seed**.
#            DINO 0.25428 / TOTAL 0.59078.
#            둘의 차이는 seed 하나뿐이므로, "seed 를 고치면 1e-4 도 안 망가지는가"에
#            바로 답이 나온다. 32분이면 끝난다.
#
#   2단계  거기서 이어 학습해 누적 3250 까지 간다. seed 는 1999 로 **바꾼다** —
#          같은 seed 로 재개하면 방금 본 데이터를 처음부터 다시 보게 되어,
#          이번에 고친 바로 그 함정(고정 스트림 반복)이 되살아난다.
#
#   3단계  누적 3250 평가. 이 지점에는 seed 777(lr 3e-5) 결과가 짝으로 있다.
#
# 사용:
#   bash scripts/branchB/resume_seed999_20260801.sh
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

CKPT="$REPO/artifacts/branchB/ckpt_snapshots_seed999_s1/epoch=0-step=1000.ckpt"
[ -f "$CKPT" ] || { echo "ERROR: 구조한 체크포인트가 없다: $CKPT"; exit 1; }

echo "=========================================================================="
echo "[재개] 시작 $(date +'%F %T')"
echo "[재개] 출발  누적 2000 (lr 1e-4, seed 999 로 1000스텝 학습된 상태)"
echo "[재개] 목표  누적 3250 — seed 777(lr 3e-5) 과 같은 지점"
echo "=========================================================================="

# ── 1단계: 구조한 체크포인트 평가 (짝이 맞는 대조군이 이미 있다) ──────────────
echo ""
echo "############ 1단계: 누적 2000 평가  $(date +'%F %T')"
echo "[재개]   대조군 = m0_cum2000_b4 (같은 lr·같은 스텝, 고정 seed) DINO 0.25428 / TOTAL 0.59078"
bash scripts/branchB/eval_ckpt.sh "$CKPT" cum2000_seed999 50 1.0 4 \
    > "run_logs/$(date +%Y%m%d_%H%M)_eval_cum2000_seed999.log" 2>&1
echo "[재개] 평가 rc=$?  $(date +'%F %T')"

echo ""
echo "############ 1단계-b: 짝지은 비교 (seed 만 다른 두 체크포인트)"
python3 scripts/branchB/compare_reports.py \
    artifacts/branchB/m0_cum2000_seed999/m0_report.json \
    artifacts/branchB/m0_cum2000_b4/m0_report.json 2>&1 | tee \
    "run_logs/$(date +%Y%m%d_%H%M)_paired_cum2000_seed.log"

# ── 2·3단계: 누적 3250 까지 이어 학습 + 자동 평가 ────────────────────────────
# seed 1999 를 주면 overnight_train_v2.sh 가 세그먼트마다 1999, 2000, ... 로 늘려 준다.
echo ""
echo "############ 2단계: 누적 3250 까지 이어 학습  $(date +'%F %T')"
bash scripts/branchB/overnight_train_v2.sh \
    "$CKPT" 2000 1.9 seed999b \
    model.base_learning_rate=1e-4 --seed 1999

echo ""
echo "=========================================================================="
echo "[재개] 전부 완료 $(date +'%F %T')"
echo "[재개]   누적 2000: artifacts/branchB/m0_cum2000_seed999/m0_report.json"
echo "[재개]   누적 3250: artifacts/branchB/m0_cum3250_seed999b/m0_report.json"
echo "=========================================================================="
