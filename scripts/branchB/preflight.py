"""새 머신(5090/6000)에서 학습을 걸기 **직전** 5분 점검 — 006 의 실패를 되풀이하지 않기 위한 게이트.

006 Part B 는 5090 에서 **1스텝도 못 돌았다**(NVIDIA 드라이버 불일치). 그런 실패는 학습을 건 뒤가 아니라
거는 순간 알아야 한다. 이 스크립트는 GPU·패키지·데이터·캐시·메모리 예산을 한 번에 확인하고,
이 장비에서 쓸 **학습 범위(scope)를 추천**한다.

사용:  conda run -n wm python scripts/branchB/preflight.py
출력:  results/branchB/preflight_<호스트명>.json  (장비별 파일이라 서로 덮어쓰지 않는다)
종료코드: 0=통과, 1=치명적 문제
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_paths import repo_root  # noqa: E402

ROOT = repo_root()
CHECKS: list[dict] = []


def add(name: str, ok: bool, detail: str, fatal: bool = True) -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), "fatal": fatal, "detail": detail})
    mark = "OK  " if ok else ("FAIL" if fatal else "WARN")
    print(f"[{mark}] {name}: {detail}")
    return ok


def main() -> int:
    print(f"=== branch B preflight @ {socket.gethostname()} ({platform.platform()}) ===")

    # 1) GPU 하드웨어·드라이버
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        add("nvidia-smi", smi.returncode == 0, smi.stdout.strip() or smi.stderr.strip()[:200])
    except Exception as e:  # noqa: BLE001
        add("nvidia-smi", False, f"실행 실패: {e}")

    import torch
    add("torch", True, f"{torch.__version__} (CUDA build {torch.version.cuda})", fatal=False)
    cuda_ok = add("torch.cuda.is_available", torch.cuda.is_available(), str(torch.cuda.is_available()))

    vram_gb = 0.0
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        cap = f"sm_{props.major}{props.minor}"
        vram_gb = props.total_memory / 2**30
        add("GPU", True, f"{props.name} / {cap} / {vram_gb:.1f}GiB")
        arch_list = torch.cuda.get_arch_list()
        add("커널 아키텍처 지원", f"sm_{props.major}{props.minor}" in arch_list,
            f"{cap} in {arch_list} — 없으면 cu128 휠 필요(5090/6000=Blackwell)", fatal=False)
        # 실제 커널 실행: 드라이버 불일치는 여기서 터진다(006 의 실패 지점)
        try:
            a = torch.randn(2048, 2048, device="cuda")
            b = (a @ a).sum().item()
            torch.cuda.synchronize()
            add("GPU 커널 실행", True, f"matmul OK (sum={b:.3e})")
        except Exception as e:  # noqa: BLE001
            add("GPU 커널 실행", False, f"{type(e).__name__}: {str(e)[:200]}")
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                c = (torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")).sum().item()
            add("fp16 autocast", True, f"OK ({c:.3e})")
        except Exception as e:  # noqa: BLE001
            add("fp16 autocast", False, f"{type(e).__name__}: {str(e)[:200]}")

    # 2) 패키지
    for mod, fatal in [("pytorch_lightning", True), ("omegaconf", True), ("numpy", True),
                       ("timm", True), ("open_clip", False), ("imageio_ffmpeg", False)]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            ok = True
            if mod == "numpy":
                ok = v.startswith("1.")     # submission_kit 은 numpy<2 요구
            add(f"pkg {mod}", ok, f"{v}" + ("" if ok else " ← numpy<2 필요(함정 4)"), fatal=fatal)
        except Exception as e:  # noqa: BLE001
            add(f"pkg {mod}", False, f"import 실패: {e}", fatal=fatal)
    try:
        import pkg_resources  # noqa: F401
        add("pkg_resources(setuptools<70)", True, "import OK (pytorch-lightning 1.9.3 요구)")
    except Exception as e:  # noqa: BLE001
        add("pkg_resources(setuptools<70)", False, f"{e} — setuptools<70 설치 필요(함정 1)")

    # 3) 데이터·체크포인트
    need = {
        "backbone.ckpt": ROOT / "open/baseline/checkpoints/backbone.ckpt",
        "train 데이터": ROOT / "open/data/train",
        "action 통계": ROOT / "open/data/train/so100_action_statistics.json",
        "submission_kit": ROOT / "open/submission_kit",
        "action_extractor.ckpt": ROOT / "open/submission_kit/checkpoints/action_extractor.ckpt",
        "1.1B config": ROOT / "scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml",
    }
    for name, p in need.items():
        exists = p.exists()
        size = ""
        if exists and p.is_file():
            size = f" ({p.stat().st_size/2**30:.2f}GiB)" if p.stat().st_size > 2**28 else ""
        if exists and p.is_dir():
            size = f" ({sum(1 for _ in p.iterdir())}개 항목)"
        add(f"경로 {name}", exists, f"{p}{size}", fatal=name != "submission_kit")

    # 4) 캐시(함정 10: 빈 폴더를 주면 offline 에서 죽는다)
    for env in ["HF_HOME", "TORCH_HOME"]:
        v = os.environ.get(env)
        if v:
            p = Path(v)
            add(f"env {env}", p.exists() and any(p.iterdir()), f"{v} (존재={p.exists()})", fatal=False)
        else:
            add(f"env {env}", True, "미설정 → 런처가 $HOME 기본값을 쓴다", fatal=False)

    # 5) 디스크 (체크포인트 1개 ≈ 11.5GB)
    try:
        st = os.statvfs(ROOT)
        free_gb = st.f_bavail * st.f_frsize / 2**30
        add("디스크 여유", free_gb > 200, f"{free_gb:.0f}GiB "
            f"(체크포인트 1개 ≈10.7GiB → 영구 아카이브 {int(free_gb//10.7)}개분)", fatal=False)
    except Exception as e:  # noqa: BLE001
        add("디스크 여유", False, str(e), fatal=False)

    # 6) 학습 범위 추천 (results/branchB/train_scope_budget.json 의 산술 사용)
    rec = None
    bud_path = ROOT / "results/branchB/train_scope_budget.json"
    if bud_path.exists() and vram_gb:
        bud = json.loads(bud_path.read_text(encoding="utf-8"))
        usable = vram_gb * 0.90          # 활성화·단편화 여유 10%
        fits = [(k, v["gb_total_estimate"]) for k, v in bud["scopes"].items()
                if v["gb_total_estimate"] <= usable]
        order = ["full", "action_temporal", "action_only"]
        fits.sort(key=lambda kv: order.index(kv[0]))
        rec = fits[0][0] if fits else None
        detail = " / ".join(f"{k}={v['gb_total_estimate']}GB" for k, v in bud["scopes"].items())
        add("학습 범위 추천", rec is not None,
            f"VRAM {vram_gb:.1f}GiB(가용 {usable:.1f}) → 추천 scope=**{rec}**   [{detail}]",
            fatal=False)

    fatal_fail = [c for c in CHECKS if not c["ok"] and c["fatal"]]
    out = ROOT / f"results/branchB/preflight_{socket.gethostname()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"host": socket.gethostname(), "platform": platform.platform(),
                               "vram_gib": round(vram_gb, 2), "recommended_scope": rec,
                               "checks": CHECKS}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[preflight] 리포트: {out}")
    if fatal_fail:
        print(f"[preflight] 판정: FAIL — 치명 {len(fatal_fail)}건: {[c['check'] for c in fatal_fail]}")
        return 1
    print(f"[preflight] 판정: PASS (경고 {sum(1 for c in CHECKS if not c['ok'])}건)")
    if rec:
        print(f"[preflight] 다음 명령:\n"
              f"  BRANCHB_TRAIN_SCOPE={rec} BRANCHB_BUILD_ONLY=1 bash scripts/branchB/run_1p1b_train.sh\n"
              f"  BRANCHB_TRAIN_SCOPE={rec} bash scripts/branchB/run_1p1b_train.sh 20 \"00:00:30:00\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
