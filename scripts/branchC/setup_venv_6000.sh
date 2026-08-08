#!/usr/bin/env bash
# RTX PRO 6000 기계용 가상환경 구축 — 저장소 안 .venv 에 만든다
#
# 왜 conda 가 아니라 venv 인가:
#   이 기계는 공용이고 /home/video_generation/dlacksdn 밖은 읽기 전용이다.
#   기존 문서(env_file/001)의 conda 경로는 5090 기준이라 여기서 쓸 수 없다.
#   uv 가 python 3.10 을 sudo 없이 받아 주므로, 검증된 버전(3.10)을 그대로 재현한다.
#
# 왜 lock 을 그대로 안 쓰나 (requirements-lock.txt 머리말 · 010 §3-⑤):
#   lock 의 torch==2.7.1 은 PyPI 기본 cu126 을 끌어온다. cu126 에는 sm_120 이 없어
#   Blackwell 에서 "no kernel image is available" 로 죽는다. 그래서
#     ① cu128 휠을 먼저 깐다
#     ② lock 에서 torch·torchvision·triton·nvidia-* 를 뺀 사본을 깐다
#        (lock 의 nvidia-*==12.6.x 는 cu128 torch 가 요구하는 12.8 을 되돌린다)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv"
cd "$REPO"

echo "=== ① python 3.10 가상환경 ==="
uv venv --python 3.10 "$VENV"
PY="$VENV/bin/python"
"$PY" -V

echo
echo "=== ② torch 2.7.1+cu128 (sm_120 포함) ==="
VIRTUAL_ENV="$VENV" uv pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

echo
# ③ opencv-python 도 뺀다. lock 파일 자체가 모순이다:
#   opencv-python==5.0.0.93 은 numpy>=2 를 요구하는데 lock 은 numpy==1.26.4 로 고정한다
#   (env_file/001 §3-4: submission_kit 이 numpy<2 를 요구해서 1.26.4 가 정본이다).
#   pip 는 이 모순을 조용히 넘겼지만 uv 는 잡는다. branchC 학습 경로는 cv2 를 안 쓴다 —
#   저장소 전체에서 유일한 사용처가 probe_warp_vs_add.py:150 의 함수 안 지연 import 다.
#   ⇒ numpy 1.26.4 를 지키고 opencv 를 뺀다. 그 탐침이 필요해지면 그때 별도 판단한다.
echo "=== ③ lock 나머지 (torch·nvidia-*·opencv 제외) ==="
grep -vE '^(torch|torchvision|triton|nvidia-|opencv-python)' requirements-lock.txt > /tmp/lock_no_torch.txt
VIRTUAL_ENV="$VENV" uv pip install -r /tmp/lock_no_torch.txt

echo
echo "=== ④ matplotlib (lock 에 없다. bootstrap ② 가 요구한다) ==="
VIRTUAL_ENV="$VENV" uv pip install "matplotlib==3.9.4"

echo
echo "=== ⑤ 확인 ==="
"$PY" - <<'EOF'
import sys, torch
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__)
print("cuda    ", torch.cuda.is_available(), torch.version.cuda)
print("archlist", " ".join(torch.cuda.get_arch_list()))
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f"  gpu{i}  {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}"
          f"  {'OK' if f'sm_{cap[0]}{cap[1]}' in torch.cuda.get_arch_list() else '❌ 미지원'}")
import numpy, pandas, av, PIL, matplotlib, imageio, timm, einops
print("numpy   ", numpy.__version__)
print("의존성   av pandas PIL matplotlib imageio timm einops OK")
EOF
echo
echo "✅ 환경 구축 끝. PY=$PY"
