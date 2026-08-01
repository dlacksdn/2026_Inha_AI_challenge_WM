#!/usr/bin/env bash
# 학습 이어하기(resume) — 이미 학습한 체크포인트에서 출발해 계속 학습한다.
#
# 왜 "warm-start 체인" 방식인가 (PL 표준 resume 이 안 되는 이유)
# ---------------------------------------------------------------
# PyTorch Lightning 의 정식 재개는 `trainer.fit(..., ckpt_path=...)` 이고, 이때 ckpt 안에
# optimizer 상태·스케줄러·콜백 상태가 다 들어 있어야 한다. 그런데 우리 ckpt 는 두 가지 이유로
# 그 조건을 못 채운다.
#   ① `save_weights_only: True`  → optimizer_states 자체가 저장돼 있지 않다.
#      PL 1.9 는 이때 "Trying to restore optimizer state but checkpoint contains only the model"
#      라며 KeyError 를 던진다.
#   ② `save_only_unet: True`     → state_dict 에 UNet(+EMA)만 있고 VAE·CLIP 키가 없다.
#      PL 은 재개 시 strict 로드를 하므로 없는 키에서 실패한다.
# ckpt 를 개조해 억지로 맞출 수도 있지만, 조용히 잘못될 여지가 크다.
#
# 그래서 **가중치만 이어받는다.** 이어받는 것과 버리는 것은 다음과 같다.
#   [이어받음] UNet 1521키 전부 = 학습 성과 그 자체.
#   [버림]     AdamW 의 1·2차 모멘트, lr 워밍업 진행도, global_step 카운터.
#              모멘트는 워밍업 250스텝이면 사실상 회복되고, 학습 성과에 미치는 영향은 작다.
#              (이 손실을 감수하는 대신 재개가 확실히 동작하고 조용한 실패가 없다.)
#
# 안전장치
#   · `name` 을 새로 주어 **출력 디렉터리를 분리**한다. 안 그러면 global_step 이 0부터 다시 세어지며
#     기존 epoch=0-step=500.ckpt 를 덮어써 자산을 파괴한다(기존 산출물 폐기 금지 규칙).
#   · 단일 GPU 전략·메모리 단편화 옵션을 자동 주입한다(013 §4 함정 ①②).
#   · 스냅샷 데몬을 자동으로 켜고, 학습이 끝나면 같이 정리한다(롤링 저장이 지우기 전에 사본 확보).
#
# 사용:
#   bash scripts/branchB/run_1p1b_resume.sh <시작ckpt> <런이름> <최대시간DD:HH:MM:SS> [추가 dotlist ...]
# 예:
#   bash scripts/branchB/run_1p1b_resume.sh \
#        artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/last.ckpt \
#        inha_action_diffusion_1p1b_r2 "00:05:00:00"
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_IN="${1:?시작 체크포인트 경로}"
RUN_NAME="${2:?런 이름(출력 디렉터리 = artifacts/branchB/train_out/<런이름>)}"
MAX_TIME="${3:-00:05:00:00}"
shift 3 || true

CKPT="$(cd "$(dirname "$CKPT_IN")" && pwd)/$(basename "$CKPT_IN")"
[ -f "$CKPT" ] || { echo "ERROR: 시작 ckpt 없음: $CKPT"; exit 1; }

OUTDIR="$REPO/artifacts/branchB/train_out/$RUN_NAME"
if [ -d "$OUTDIR/checkpoints" ] && ls "$OUTDIR"/checkpoints/*.ckpt >/dev/null 2>&1; then
  echo "ERROR: '$RUN_NAME' 에 이미 체크포인트가 있다. 덮어쓰지 않는다 — 다른 런이름을 써라."
  echo "       ($OUTDIR/checkpoints)"
  exit 1
fi
SNAPDIR="$REPO/artifacts/branchB/ckpt_snapshots_$RUN_NAME"

# ── 환경 (013 §4 함정 ① — 빼먹으면 학습이 무한정 멈춘다) ──────────────────────
export PATH="/home/rils/dlacksdn/miniconda3/bin:${PATH}"
export CONDA_BIN="${CONDA_BIN:-/home/rils/dlacksdn/miniconda3/bin/conda}"
export HF_HOME="${HF_HOME:-/home/rils/dlacksdn/.cache/hf}"
export TORCH_HOME="${TORCH_HOME:-/home/rils/dlacksdn/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── warm-start 체인 설정 ─────────────────────────────────────────────────────
export BRANCHB_WARMSTART_CKPT="$CKPT"
export BRANCHB_ZERO_INIT=0            # 이미 학습된 액션 분기를 0으로 되돌리면 안 된다
export BRANCHB_TRAIN_SCOPE="${BRANCHB_TRAIN_SCOPE:-action_temporal}"

echo "=========================================================================="
echo "[resume] 시작 ckpt : $CKPT"
echo "[resume] 런 이름   : $RUN_NAME  → $OUTDIR"
echo "[resume] 최대 시간 : $MAX_TIME   scope=$BRANCHB_TRAIN_SCOPE"
echo "[resume] 스냅샷    : $SNAPDIR"
echo "[resume] 시작      : $(date +'%F %T')"
echo "=========================================================================="

mkdir -p "$OUTDIR/checkpoints" "$SNAPDIR"
bash "$REPO/scripts/branchB/ckpt_snapshot_daemon.sh" "$OUTDIR/checkpoints" "$SNAPDIR" 60 &
SNAP_PID=$!

# ── 더티 페이지 흘려보내기 (2026-08-01 사망 대응) ────────────────────────────
# 250스텝마다 5.76GB 짜리 체크포인트를 쓴다. 그 데이터는 일단 메모리(페이지 캐시)에
# 쌓였다가 디스크로 내려가는데, 프로젝트가 HDD 라 내려가는 속도가 느리다. 그래서
# **쓰는 동안 메모리 압박이 치솟는다.** 이 기계는 user@1000.service 에 압박 50% 초과 시
# 프로세스를 죽이는 정책(ManagedOOMMemoryPressure=kill)이 걸려 있다.
#   실측 2026-08-01: 15:13 에 체크포인트를 쓰고 15:15:53 에 압박 74.11% → oomd 가 킬.
#   (그때 브라우저가 같이 떠 있었다. 브라우저 없이 돈 밤샘 3.5시간은 무사했다.)
# sync 를 주기적으로 불러 쌓인 더티 페이지를 조금씩 내려보내면 한 번에 치솟는 봉우리가
# 낮아진다. sync 는 읽기/쓰기를 파괴하지 않으며 비용도 거의 없다.
( while true; do sync; sleep 20; done ) &
SYNC_PID=$!

trap 'kill $SNAP_PID $SYNC_PID 2>/dev/null; echo "[resume] 데몬 종료(snap $SNAP_PID / sync $SYNC_PID)"' EXIT
echo "[resume] 스냅샷 데몬 pid=$SNAP_PID / sync 데몬 pid=$SYNC_PID"
echo "[resume] ※ 학습 중에는 브라우저를 띄우지 마십시오 — 2026-08-01 사망의 확인된 공범입니다."

cd "$REPO"
bash scripts/branchB/run_1p1b_train.sh "" "$MAX_TIME" \
  "name=$RUN_NAME" \
  lightning.trainer.strategy.target=pytorch_lightning.strategies.SingleDeviceStrategy \
  lightning.trainer.strategy.params.device=cuda:0 \
  lightning.callbacks.model_checkpoint.params.every_n_train_steps=250 \
  "$@"
RC=$?

echo "[resume] 학습 종료 rc=$RC  $(date +'%F %T')"
sleep 90   # 마지막 체크포인트가 다 쓰이고 스냅샷될 시간을 준다
echo "[resume] 스냅샷 목록:"
ls -la "$SNAPDIR" 2>/dev/null
exit $RC
