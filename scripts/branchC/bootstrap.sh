#!/usr/bin/env bash
# branch C 부트스트랩 — clone 직후 학습을 돌릴 수 있는 상태로 만든다
#
# clone 에 안 딸려 오는 것 (전부 gitignore, 그럴 만한 이유가 있다):
#   third_party/OpenSTL      남의 저장소를 우리 히스토리에 넣지 않는다 → 고정 커밋으로 clone
#   open/data/{train,eval}   대회 데이터 8.5GB → 링크만 건다
#   artifacts/holdout_val96  34MB → manifest(43KB, 추적됨)로 재생성한다
#
# 사용:  bash scripts/branchC/bootstrap.sh [<python 경로>]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${1:-$REPO/../miniconda3/envs/wm/bin/python}"
OPENSTL_COMMIT=eecf8a3078f0a178dbc7b28723da20f94ce36985
FAIL=0
say() { printf "%-46s %s\n" "$1" "$2"; }

echo "=== branch C 부트스트랩 ==="
echo "레포: $REPO"
echo

# 1. 파이썬 / torch
if [ -x "$PY" ]; then
  V=$("$PY" -c "import torch,sys;print(f'py {sys.version.split()[0]} · torch {torch.__version__} · cuda {torch.cuda.is_available()}')" 2>/dev/null)
  [ -n "$V" ] && say "① python/torch" "OK  $V" || { say "① python/torch" "torch 없음 → 환경 확인"; FAIL=1; }
else
  say "① python" "없다: $PY  (인자로 경로를 넘겨라)"; FAIL=1
fi

# 2. 추가 의존성
if [ -x "$PY" ]; then
  MISS=$("$PY" - <<'EOF'
import importlib
need = ["av", "pandas", "numpy", "PIL", "matplotlib", "imageio"]
print(" ".join(m for m in need if importlib.util.find_spec(m) is None))
EOF
)
  if [ -z "$MISS" ]; then say "② 의존성" "OK"; else
    say "② 의존성" "빠짐: $MISS"
    echo "     → $PY -m pip install $MISS"; FAIL=1
  fi
fi

# 3. OpenSTL (고정 커밋)
if [ -d "$REPO/third_party/OpenSTL/.git" ]; then
  HAVE=$(cd "$REPO/third_party/OpenSTL" && git rev-parse HEAD)
  [ "$HAVE" = "$OPENSTL_COMMIT" ] && say "③ OpenSTL" "OK (${OPENSTL_COMMIT:0:8})" \
    || say "③ OpenSTL" "⚠ 커밋 다름: ${HAVE:0:8} ≠ ${OPENSTL_COMMIT:0:8}"
else
  say "③ OpenSTL" "없다 → 받는다"
  mkdir -p "$REPO/third_party"
  git clone https://github.com/chengtan9907/OpenSTL.git "$REPO/third_party/OpenSTL" \
    && (cd "$REPO/third_party/OpenSTL" && git checkout -q "$OPENSTL_COMMIT") \
    && say "   OpenSTL" "OK (${OPENSTL_COMMIT:0:8}, Apache-2.0)" || { say "   OpenSTL" "실패"; FAIL=1; }
fi

# 4. 데이터 링크
for d in train eval; do
  if [ -e "$REPO/open/data/$d" ]; then
    say "④ open/data/$d" "OK → $(readlink -f "$REPO/open/data/$d")"
  else
    say "④ open/data/$d" "없다 → 대회 데이터에 심볼릭 링크를 걸어라"
    echo "     예) ln -s /path/to/$d $REPO/open/data/$d"; FAIL=1
  fi
done

# 5. 홀드아웃 (manifest 로 재생성)
MAN="$REPO/artifacts/holdout_val96/manifest.json"
if [ ! -f "$MAN" ]; then
  say "⑤ 홀드아웃" "manifest 가 없다 — clone 이 불완전하다"; FAIL=1
elif [ "$(ls "$REPO/artifacts/holdout_val96/gt_videos" 2>/dev/null | wc -l)" -eq 96 ]; then
  say "⑤ 홀드아웃" "OK (96개)"
else
  say "⑤ 홀드아웃" "재생성한다 (manifest → train 데이터)"
  "$PY" "$REPO/scripts/branchC/rebuild_holdout.py" && say "   홀드아웃" "OK" || { say "   홀드아웃" "실패"; FAIL=1; }
fi

# 6. 행동 통계 (추적됨)
[ -f "$REPO/results/branchC/action_stats.json" ] && say "⑥ 행동 z-score 통계" "OK" \
  || { say "⑥ 행동 z-score 통계" "없다 → python scripts/branchC/loader_c.py --make-stats"; FAIL=1; }

echo
if [ "$FAIL" -eq 0 ]; then
  echo "✅ 준비 완료. 학습을 돌릴 수 있다:"
  echo "   $PY scripts/branchC/train_c.py --tag g1 --steps 6000"
  echo "   (판정선은 scripts/branchC/gates.py 에 등록돼 있다)"
else
  echo "⚠ 위의 빠진 항목을 먼저 채워라."
fi
exit "$FAIL"
