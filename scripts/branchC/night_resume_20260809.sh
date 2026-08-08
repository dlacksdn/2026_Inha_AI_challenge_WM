#!/usr/bin/env bash
# 2026-08-09 야간 — full_002500 에서 두 팔을 GPU 0 에 순차로 돌린다 (010 §6)
#
# 🚨 GPU 0 만 쓴다. 이 기계는 공용이고 GPU 1 은 남의 자리다 (CLAUDE.md).
#
# 설정 근거 [측정 2026-08-09, results/branchC/step_budget_6000_*.json]:
#   micro 2 × accum 8 · ckpt 켬 = 원본 런(full_002500)과 **비트 단위로 같은 설정**이다.
#   ckpt 를 끄면 3.63 s/step 으로 1.31 배 빨라지지만 쓰지 않는다 —
#   checkpointing 이 backward 에서 forward 를 재실행해 **BatchNorm 러닝통계를 스텝당 2회**
#   갱신하기 때문이다 [측정: 3스텝 후 num_batches_tracked 6(켬) vs 3(끔)].
#   원본이 2회 갱신으로 학습됐으므로 지금 끄면 감시·추론이 읽는 통계 궤적이 어긋난다.
#   micro 4 는 오히려 느리고(5.06 s), micro ≥ 7 은 32비트 인덱스 넘침으로 불가하다(010 §6).
#
# --steps 7000 인 이유 (010 §6): 등록된 G2 판정점은 step 6,000(참 wake 2,000 + 4,000)이다.
#   7,000 까지 가면 6,000 행을 확보하고 그 뒤 2점(6500·7000)이 G2.5(연속 2회)를 볼 수 있다.
# --wake-step 2000 인 이유: 체크포인트가 wake_step 을 안 실어서 재개 런이 wake 를 재검출한다.
#   그대로 두면 코드의 G2 발동이 7,000 으로 밀린다. 문턱을 바꾼 게 아니라
#   **이미 관측된 상태(history.json 의 wake_step=2000)를 복원**하는 것이다.
#
# 순차 체인이 규율 1(사람 확인)을 우회하지 않는 이유: 두 팔은 서로의 결과를 읽지 않는
# 독립 A/B 팔이다. 판정(G2·M4)은 두 팔이 다 끝난 뒤 사람이 한다. 008 §9-d 가 경고한 것은
# "M3 결과를 안 읽고 본학습으로 넘어가는" 체인이었고, 이건 그 구조가 아니다.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
CK="$REPO/artifacts/branchC/train_20260808_1809_g1/full_002500.pt"
export CUDA_VISIBLE_DEVICES=0

COMMON="--steps 7000 --resume $CK --wake-step 2000 --micro-batch 2 --accum 8"

run_arm() {
  local tag="$1"; shift
  local log="$REPO/run_logs/$(date +%Y%m%d_%H%M)_train_c_${tag}.log"
  echo "════════════════════════════════════════════════════════════"
  echo "[$(date +%H:%M:%S)] 팔 '$tag' 시작 — GPU $CUDA_VISIBLE_DEVICES"
  echo "  로그: ${log#$REPO/}"
  echo "════════════════════════════════════════════════════════════"
  "$PY" scripts/branchC/train_c.py --tag "$tag" $COMMON "$@" 2>&1 | tee "$log"
  echo "[$(date +%H:%M:%S)] 팔 '$tag' 종료 rc=${PIPESTATUS[0]}"
}

# 방향항 팔을 **먼저** 돌린다. 밤중에 하나만 끝나도 정보가 더 많기 때문이다 —
# L1 단독의 궤적은 step 3,500 까지 이미 있지만(Δcos 0.030/0.025/0.035),
# 방향항이 그 되당김을 상쇄하는지는 아무 데이터도 없다. 008 §8-② 가 1순위로 지목한 쪽이다.
run_arm dir --dir-loss --tau-alpha 0.1
run_arm g1r

echo
echo "════════════════════════════════════════════════════════════"
echo "[$(date +%H:%M:%S)] 두 팔 완료. 판정은 사람이 한다:"
echo "  G2 : 각 팔 history.json 의 step 6000 행 Δcos ≥ 0.046   (gates.py 등록값)"
echo "  M4 : 두 팔의 같은 스텝 코사인 차이 ≥ 씨앗간 sd 의 2배  (008 §8-③)"
echo "  ⚠ 외삽 금지. 판정은 gates.py 가 정한 시점에만 (009 §11)"
echo "════════════════════════════════════════════════════════════"
