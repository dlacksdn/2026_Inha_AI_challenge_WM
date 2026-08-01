#!/usr/bin/env bash
# branch B 생성 — 학습한 1.1B UNet 으로 홀드아웃/eval 영상을 만든다.
# baseline 의 generate_baseline_videos.py 를 **수정 없이** 부른다(PYTHONPATH·CWD 만 맞춤).
#
# 사용:
#   bash scripts/branchB/run_1p1b_generate.sh <ckpt> <입력루트> <출력루트> [ddim_steps] [cfg_scale] [batch]
#
# batch(6번째 인자, 기본 1)
#   한 번에 처리할 표본 수. 1.44B UNet 은 배치 1이면 GPU 를 다 못 채운다(26초/샘플).
#   배치를 키우면 표본당 시간이 줄지만 VRAM 을 더 쓴다.
#   ⚠️ 배치를 바꾸면 초기 노이즈 텐서의 모양이 바뀌어 **같은 시드라도 결과가 달라진다.**
#      배치가 다른 결과끼리 소수점 단위로 비교하지 말 것(표본 평균으로는 동등하다).
# 예:
#   # S4 스모크: 홀드아웃 앞 4개만
#   bash scripts/branchB/run_1p1b_generate.sh \
#        artifacts/branchB/train_out/inha_action_diffusion_1p1b/checkpoints/last.ckpt \
#        artifacts/holdout_smoke4 artifacts/branchB/preds_smoke 50 1.0
#   # S5 판정: 홀드아웃 96개 전부 → 채점은 scripts/run_m0.py
#   bash scripts/branchB/run_1p1b_generate.sh <ckpt> artifacts/holdout artifacts/branchB/preds_pilot 50 1.0
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$REPO/open/baseline/challenge_kit"
CKPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
IN="$(cd "$2" && pwd)"
OUT="$3"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
STEPS="${4:-50}"
CFGSCALE="${5:-1.0}"
BATCH="${6:-1}"
# 7번째 인자 seed (기본 0 = 지금까지의 동작 그대로. 기존 결과의 재현성은 유지된다).
#   ddim_eta=0 이라 샘플링은 결정론적이고, 샘플이 달라지는 유일한 원천이 **초기 노이즈**다.
#   같은 입력을 seed 만 바꿔 여러 번 생성하면 "이 모델이 얼마나 흔들리는가"를 잴 수 있고,
#   그 여러 장을 평균내면 조건부 평균의 근사가 된다(017 §4, 1X WMC 우승 경로).
SEED="${7:-0}"

[ -f "$CKPT" ] || { echo "ERROR: ckpt 없음: $CKPT"; exit 1; }

# conda 위치는 기계마다 다르다 — 집 4060Ti 는 ~/miniconda3, 연구실 5090 은
# /home/rils/dlacksdn/miniconda3 에 있다(공유 계정이라 홈이 아니라 프로젝트 옆에 깔았다).
# 이 스크립트는 원래 집 기준으로 작성돼 5090 에서 rc=127 로 조용히 죽었다(2026-08-01).
# 그래서 후보를 순서대로 찾는다. CONDA_BIN 이 주어지면 그것이 최우선이다.
if [ -z "${CONDA_BIN:-}" ]; then
  for c in /home/rils/dlacksdn/miniconda3/bin/conda "$HOME/miniconda3/bin/conda" "$(command -v conda 2>/dev/null || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { CONDA_BIN="$c"; break; }
  done
fi
CONDA="${CONDA_BIN:?conda 를 찾지 못했다. CONDA_BIN 환경변수로 경로를 지정하라}"
PYBIN="$($CONDA run -n wm which python)"
[ -x "$PYBIN" ] || { echo "ERROR: wm 환경의 python 을 못 찾았다 ($CONDA)"; exit 1; }

# 학습 config + 생성 config 둘 다 __REPO__ 치환본을 만든다(생성 config 가 학습 config 를 참조하므로 순서 무관).
"$PYBIN" "$REPO/scripts/branchB/cfg_paths.py" \
    scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml \
    scripts/branchB/configs/eval/gen_1p1b.yaml
BASE_GEN_CFG="$REPO/artifacts/branchB/_runtime_cfg/gen_1p1b.yaml"
if grep -q "__REPO__" "$BASE_GEN_CFG"; then echo "ERROR: __REPO__ 잔존"; exit 1; fi

# ── [중요] 생성 전용 모델 config: use_ema=False ───────────────────────────────
# 왜: LitEma 는 requires_grad 인 파라미터만 shadow 로 등록한다(ema.py L17-18).
#   - 학습 시엔 scope 로 551M 만 학습 → EMA 도 551M 만 저장된다(ckpt 의 model_ema.* 809키).
#   - 생성 시엔 scope 를 적용하지 않아 **전 파라미터가 requires_grad=True** →
#     LitEma 가 1521키를 '모델 생성 시점의 랜덤 가중치'로 초기화한다.
#   - strict=False 로드로는 809키만 채워지고 나머지 712키는 **랜덤인 채로 남는다**.
#   - 그 상태에서 ema_scope() 가 copy_to() 로 전 파라미터를 덮어쓰면(L51-52)
#     UNet 의 절반 이상이 랜덤값이 되어 **출력이 순수 노이즈**가 된다.
# 대응: 생성에서는 EMA 를 끈다. 학습된 가중치는 ckpt 의 model.diffusion_model.*(1521키)에
#   온전히 들어 있으므로 EMA 없이 그대로 쓰는 것이 정확하다(ema_scope 는 no-op 이 된다).
BASE_MODEL_CFG="$REPO/artifacts/branchB/_runtime_cfg/inha_action_diffusion_1p1b.yaml"
GEN_MODEL_CFG="$REPO/artifacts/branchB/_runtime_cfg/inha_action_diffusion_1p1b_geneval.yaml"
sed -e "s/^\( *use_ema:\).*/\1 False/" "$BASE_MODEL_CFG" > "$GEN_MODEL_CFG"
grep -qE "^ *use_ema: *False" "$GEN_MODEL_CFG" || { echo "ERROR: use_ema=False 주입 실패"; exit 1; }

# generate_baseline_videos.py 는 CFG 스케일 CLI 인자가 없다 → 실행별 config 사본에 값을 심는다.
GEN_CFG="$REPO/artifacts/branchB/_runtime_cfg/gen_1p1b_steps${STEPS}_cfg${CFGSCALE}.yaml"
sed -e "s/^\( *unconditional_guidance_scale:\).*/\1 ${CFGSCALE}/" \
    -e "s/^\( *ddim_steps:\).*/\1 ${STEPS}/" \
    -e "s#^\(model_config_file:\).*#\1 ${GEN_MODEL_CFG}#" "$BASE_GEN_CFG" > "$GEN_CFG"
grep -E "ddim_steps|unconditional_guidance_scale|model_config_file" "$GEN_CFG"

# 메모리 단편화 방지 — 없으면 배치 4 에서 OOM 이 난다(2026-08-01 eval 216 생성이 이걸로 죽었다).
# 왜: 확산 샘플링은 스텝마다 큰 어텐션 텐서를 잡았다 놓기를 반복해 할당기 안에 구멍이 생긴다.
#   그러면 여유 메모리 총량은 남아도 **연속된 큰 덩어리**를 못 잡아 실패한다.
#   (실측 로그: "7.67 GiB is free ... 8.10 GiB is reserved but unallocated" — 전형적인 단편화)
#   expandable_segments 는 할당 구간을 늘려 쓰게 해 이 구멍을 없앤다.
# 그동안 이 스크립트를 늘 eval_ckpt.sh 가 감싸서 불렀고 거기서 이 값을 export 했기 때문에
# 문제가 안 보였다. 직접 호출하면 터진다 → 감싸는 쪽이 아니라 여기에 둬야 한다.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export PYTHONPATH="$CK/libs/dynamicrafter:$CK/src:$CK:$REPO/open/baseline/shared_libs/video_utils"

cd "$CK"
echo "[branchB-gen] ckpt=$CKPT"
echo "[branchB-gen] in=$IN out=$OUT steps=$STEPS cfg=$CFGSCALE batch=$BATCH seed=$SEED"
"$PYBIN" -u scripts/inference/generate_baseline_videos.py \
  --config "$GEN_CFG" \
  --checkpoint "$CKPT" \
  --challenge-root "$IN" \
  --prediction-root "$OUT" \
  --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
  --ddim-steps "$STEPS" \
  --batch-size "$BATCH" \
  --seed "$SEED" \
  --overwrite
echo "[branchB-gen] 완료 → $OUT  (mp4 $(ls "$OUT"/*.mp4 2>/dev/null | wc -l)개)"
