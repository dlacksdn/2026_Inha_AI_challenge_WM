#!/usr/bin/env bash
# 야간 체인 2 — 3라운드 측정(A·B·D·E·F) + Farneback 대조(C).
#
# 2차 적대적 검수가 018 의 결론 둘을 철회시켰고, 그 자리를 메우는 측정이다.
# **판정선은 018 §9-c 에 측정 전 고정했다.** 결과를 본 뒤에 기준을 바꾸지 않는다.
#
#   A  α × k 2차원 훑기        ← 가장 중요. 우리 모델이 서 있는 α≈0.35 행이 비어 있다
#   B  같은 배율의 공정한 바닥  ← 2라운드 바닥이 무효였던 자리
#   D  nearest 보간 분리
#   E  k=2 작동점의 혼합 재판정
#   F  blur_residual 독립 대조
#   C  Farneback 흐름으로 2라운드 전체 재실행 ← "흐름 품질이 범인인가"에 답한다
#
# 주의
#   - GPU 를 쓴다. K샘플 체인이 끝난 뒤에 돌려야 한다(32GB 를 거의 다 쓴다).
#   - Farneback 은 CPU 광류라 준비 단계가 RAFT 보다 느리다. 인내.
#   - 실패해도 앞 단계 결과는 남는다(단계마다 별도 JSON).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
STAMP="$(date +%Y%m%d_%H%M)"
source /home/rils/dlacksdn/miniconda3/etc/profile.d/conda.sh
conda activate wm

echo "############ 3라운드 체인 시작 $(date +'%F %T')"

# 한 단계가 죽어도 나머지는 돌린다. 둘은 서로 독립이고, 밤 시간을 통째로 잃으면 손해가 크다.
# (018 §9-c 의 A·B·D·E·F 와 C 는 각각 다른 질문에 답한다.)
# 전체 실행 전에 2분짜리 스모크로 배선을 확인한다. 밤에 33분을 태우고 나서
# 오타 하나로 죽는 것을 막는다(오늘만 두 번 겪었다: ls/set-e, f-string 역슬래시).
# ⚠ n=4 결과는 **결론에 쓰지 않는다.** 배선 확인용이다(016 §9.2 의 n=8 함정).
echo "---- 스모크(n=4, 배선 확인용) $(date +'%F %T')"
if ! python scripts/branchC/probe_round3.py --limit 4 \
        --out "$REPO/results/branchC/_round3_smoke.json" \
        > "run_logs/${STAMP}_round3_smoke.log" 2>&1; then
  echo "!! 스모크 실패 — 전체 실행을 건너뛴다. 로그: run_logs/${STAMP}_round3_smoke.log"
  tail -25 "run_logs/${STAMP}_round3_smoke.log"
  SKIP_R3=1
else
  echo "---- 스모크 통과 $(date +'%F %T')"
  rm -f "$REPO/results/branchC/_round3_smoke.json"
  SKIP_R3=0
fi

RC_R3=0
if [ "$SKIP_R3" = "1" ]; then
  RC_R3=99
else
echo "---- A·B·D·E·F: probe_round3 $(date +'%F %T')"
python scripts/branchC/probe_round3.py \
    --out "$REPO/results/branchC/round3.json" \
    > "run_logs/${STAMP}_round3.log" 2>&1 || RC_R3=$?
echo "---- probe_round3 종료 rc=$RC_R3 $(date +'%F %T')"
tail -60 "run_logs/${STAMP}_round3.log"
fi

echo
RC_FB=0
echo "---- C: Farneback 대조 (2라운드 전체 재실행) $(date +'%F %T')"
python scripts/branchC/probe_warp_round2.py --flow farneback \
    > "run_logs/${STAMP}_round2_farneback.log" 2>&1 || RC_FB=$?
echo "---- Farneback 종료 rc=$RC_FB $(date +'%F %T')"
tail -30 "run_logs/${STAMP}_round2_farneback.log"

echo
echo "요약: probe_round3 rc=$RC_R3 · farneback rc=$RC_FB  (0 이 아니면 해당 로그를 보라)"

echo "############ 3라운드 체인 완료 $(date +'%F %T')"
