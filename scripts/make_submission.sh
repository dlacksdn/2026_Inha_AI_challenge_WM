#!/usr/bin/env bash
# 제출 CSV 생성 — submission_kit/make_submission_csv.py 를 **수정 없이** 호출한다.
#
# 사용:
#   bash scripts/make_submission.sh <prediction_dir> <output_csv_name>
# 예:
#   bash scripts/make_submission.sh artifacts/submission/static static_submission.csv
#
# 주의(대회 규칙):
#   - CSV는 반드시 이 스크립트(=공식 make_submission_csv.py)로만 생성한다. 후처리/수정 금지.
#   - submission_kit 코드·checkpoint는 절대 수정하지 않는다.
#   - eval 데이터를 추론 입력으로 쓰는 것은 정상(과제 자체). 학습에 쓰는 것만 금지.
#
# 부수 효과(유용): 생성된 CSV의 "Action Component" 행에는 정답 액션과의 MAE가 그대로 들어간다.
#   → eval 216개에서의 Action(40%) 실제 값을 제출 전에 알 수 있다.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PRED="$(cd "$1" && pwd)"
OUTNAME="${2:-submission.csv}"
KIT="$REPO/open/submission_kit"
OUT="$REPO/artifacts/submission/$OUTNAME"

n=$(ls -1 "$PRED"/*.mp4 2>/dev/null | wc -l)
echo "[submit] 예측 영상 $n개  ($PRED)"
[ "$n" -eq 216 ] || { echo "ERROR: eval은 216개여야 합니다 (현재 $n)"; exit 1; }

# conda wm 활성화 (대화형 셸에 conda init이 없어도 동작)
if [ "${CONDA_DEFAULT_ENV:-}" != "wm" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate wm
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

mkdir -p "$REPO/artifacts/submission"
cd "$KIT"
python make_submission_csv.py \
  --prediction-root "$PRED" \
  --challenge-root "$REPO/open/data/eval" \
  --output-csv "$OUT" \
  --action-stats-path "$REPO/open/data/train/so100_action_statistics.json" \
  --action-extractor-ckpt "$KIT/checkpoints/action_extractor.ckpt"

echo "[submit] CSV 생성 완료: $OUT"
python - "$OUT" <<'PY'
import csv, json, sys, statistics
path = sys.argv[1]
acts = []
comps = {}
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        comps[row["feature_component"]] = comps.get(row["feature_component"], 0) + 1
        if row["feature_component"] == "Action Component":
            v = json.loads(row["feature_json"])
            while isinstance(v, list):
                v = v[0]
            acts.append(float(v))
print(f"[submit] 행 구성: {comps}")
if acts:
    print(f"[submit] eval Action Component (40%): 평균 {statistics.mean(acts):.5f} "
          f"| 중앙 {statistics.median(acts):.5f} | n={len(acts)}")
    print("[submit]   ※ 이 값이 실제 리더보드 Action 성분이다(로컬 완전 계산).")
PY
