#!/usr/bin/env bash
# 생성 영상을 **눈으로** 비교한다 — 정답 / 우리 모델 / 정지영상을 나란히 붙인 mp4 를 만든다.
#
# 왜 필요한가
# -----------
# 점수만 보면 "DINO 0.227"이 무슨 뜻인지 알 수 없다. 013 의 최대 교훈이 "생성물을 눈으로 본다"
# 였고(EMA 버그로 출력이 순수 노이즈였는데 숫자만 보다 늦게 발견했다), 015 적대적 검수도
# "생성 영상을 육안 확인하지 못했다"를 한계로 적었다.
#
# 그런데 그냥 열면 못 본다. 16프레임 × 6fps = **2.67초**라 눈 깜빡할 새에 끝난다.
# 게다가 정답(640×480)과 우리 생성물(512×320)은 크기·비율이 달라 따로 열면 비교가 안 된다.
#
# 그래서 이 스크립트는
#   ① 셋을 같은 크기(512×320)로 맞춘다 — 비율은 유지하고 남는 곳은 검게 채운다(채점기와 같은 방식)
#   ② 가로로 나란히 붙이고 무엇이 무엇인지 글자를 얹는다
#   ③ 여러 표본을 세로로 쌓는다
#   ④ 느리게(2fps) + 여러 번 반복해 **충분히 오래 재생되게** 만든다
#
# 사용:
#   bash scripts/branchB/compare_videos.sh [예측디렉터리] [표본번호...]
# 예:
#   bash scripts/branchB/compare_videos.sh artifacts/branchB/preds_cum3250_seed777 0 1 2 3
#   bash scripts/branchB/compare_videos.sh                       # 기본값으로
#
# 보기:
#   mpv --loop=inf artifacts/branchB/compare/compare.mp4
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

PRED="${1:-artifacts/branchB/preds_cum3250_seed777}"
shift 2>/dev/null || true
IDS=("$@")
[ ${#IDS[@]} -eq 0 ] && IDS=(0 1 2 3)

GT="artifacts/holdout/gt_videos"
OUT="artifacts/branchB/compare"
mkdir -p "$OUT"

# 정지영상(첫 프레임 고정)은 홀드아웃 채점 때 만들어 둔 것을 쓴다. 없으면 정답의 첫 프레임으로 만든다.
STATIC="artifacts/branchB/m0_step1000_b4/static_preds"
[ -d "$STATIC" ] || STATIC=""

W=512; H=320
SLOW=3             # 몇 배 느리게 재생할지 (setpts 로 늘린다 — fps 필터를 쓰면 프레임이 버려진다)
REPEAT=3           # 반복 횟수 (16프레임 × 3 = 48프레임, 3배 느리게 → 약 24초)

FONT=""
for f in /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
         /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf; do
  [ -f "$f" ] && { FONT="$f"; break; }
done

# 한 표본당 가로 3칸을 만든다. 라벨은 영어로 둔다(drawtext 가 한글 글꼴을 못 찾는 환경이 있다).
label() {  # $1=텍스트
  [ -n "$FONT" ] && echo ",drawtext=fontfile=$FONT:text='$1':x=8:y=6:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5" || echo ""
}

INPUTS=(); FILTERS=(); ROWS=()
i=0
for n in "${IDS[@]}"; do
  sid="$(printf 'sample_%06d' "$n")"
  g="$GT/$sid.mp4"; p="$PRED/$sid.mp4"
  [ -f "$g" ] && [ -f "$p" ] || { echo "[건너뜀] $sid (정답 또는 예측 없음)"; continue; }
  s=""
  [ -n "$STATIC" ] && [ -f "$STATIC/$sid.mp4" ] && s="$STATIC/$sid.mp4"

  INPUTS+=(-i "$g" -i "$p"); idx_g=$i; idx_p=$((i+1)); i=$((i+2))
  if [ -n "$s" ]; then INPUTS+=(-i "$s"); idx_s=$i; i=$((i+1)); else idx_s=""; fi

  # 비율 유지 축소 후 남는 곳을 검게 채운다(채점기의 pad 와 같은 처리)
  fit="scale=$W:$H:force_original_aspect_ratio=decrease,pad=$W:$H:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
  FILTERS+=("[$idx_g:v]$fit$(label "GT  $sid")[g$idx_g]")
  FILTERS+=("[$idx_p:v]$fit$(label 'OURS')[p$idx_p]")
  if [ -n "$idx_s" ]; then
    FILTERS+=("[$idx_s:v]$fit$(label 'STATIC')[s$idx_s]")
    FILTERS+=("[g$idx_g][p$idx_p][s$idx_s]hstack=inputs=3[row${#ROWS[@]}]")
  else
    FILTERS+=("[g$idx_g][p$idx_p]hstack=inputs=2[row${#ROWS[@]}]")
  fi
  ROWS+=("[row${#ROWS[@]}]")
done

[ ${#ROWS[@]} -eq 0 ] && { echo "ERROR: 붙일 표본이 없다"; exit 1; }

if [ ${#ROWS[@]} -gt 1 ]; then
  FILTERS+=("$(printf '%s' "${ROWS[@]}")vstack=inputs=${#ROWS[@]}[grid]")
  LAST="[grid]"
else
  LAST="${ROWS[0]}"
fi
# 반복 + 느리게 — 짧아서 한 번 보고는 판단이 안 된다.
#   ⚠ fps 필터로 느리게 만들면 안 된다. fps 는 목표 프레임레이트에 맞추려고 **프레임을 버린다**
#     (실측: 48프레임 → 21프레임으로 줄었다). setpts 는 타임스탬프만 늘려 전부 보존한다.
FILTERS+=("${LAST}loop=loop=$((REPEAT-1)):size=16:start=0,setpts=$SLOW*PTS[out]")

FC="$(IFS=';'; echo "${FILTERS[*]}")"
DEST="$OUT/compare.mp4"
echo "[compare] 예측=$PRED  표본=${IDS[*]}"
ffmpeg -y -loglevel error "${INPUTS[@]}" -filter_complex "$FC" -map "[out]" \
       -c:v libx264 -pix_fmt yuv420p -crf 18 "$DEST" || { echo "ffmpeg 실패"; exit 1; }

echo "[compare] 완성: $DEST"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "$DEST"
echo
echo "보기:  mpv --loop=inf $DEST"
echo "       vlc --loop $DEST"
