#!/usr/bin/env bash
# Part B 학습이 끝나면 자동으로 λ 스윕(생성+채점)까지 이어서 실행하는 체인.
# 무인 실행용: 학습 프로세스 종료를 기다린 뒤 최신 체크포인트로 스윕한다.
#
# 사용: nohup bash scripts/m3/chain_partB_sweep.sh > run_logs/partB_chain.log 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
CKDIR="$REPO/artifacts/m3/train_out/inha_action_diffusion_11M_m3/checkpoints"

echo "[chain] 학습 프로세스 종료 대기 ..."
while pgrep -f "train_m3.py" >/dev/null 2>&1; do sleep 60; done
echo "[chain] 학습 종료 감지 ($(date '+%F %T'))"

# 최신 체크포인트 확보 (없으면 중단)
CKPT="$(ls -t "$CKDIR"/*.ckpt 2>/dev/null | head -1)"
if [ -z "${CKPT:-}" ]; then
  echo "[chain] ERROR: 체크포인트가 없다 — 학습이 500스텝 전에 죽었을 수 있다. 스윕 생략."
  ls -la "$CKDIR" 2>/dev/null || echo "[chain] ckpt 디렉터리 자체가 없음: $CKDIR"
  exit 1
fi
echo "[chain] 사용 체크포인트: $CKPT"

# λ 스윕: 1.0(off) / 1.5 / 2.5 — Part A와 같은 축에서 비교(시간 절약을 위해 3점)
bash scripts/m3/sweep_and_score.sh "$CKPT" \
  "$REPO/artifacts/m3/partB_home" "$REPO/results/m3/partB_home" trained 10 15 25
echo "[chain] 스윕 완료 ($(date '+%F %T'))"

# Part A와 나란히 요약
CONDA="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
"$CONDA" run -n wm python scripts/m3/summarize.py results/m3/partA results/m3/partB_home 2>/dev/null || true
echo "[chain] 끝"
