"""config 안의 `__REPO__` 를 실행 시점 저장소 루트로 치환한다 (008 §8.6 함정 9 대응).

왜 필요한가
  M3 에서 config 내부에 5090 절대경로(`/home/rils/...`)가 박혀 있어 스윕이 실패했다.
  스크립트만 이식성 수정하고 config 는 놓쳤기 때문이다. 그 재발을 구조적으로 막는다:
    - config 는 `__REPO__/...` 만 쓴다(어느 머신의 절대경로도 박지 않는다).
    - 실행 주체(런처/검증 스크립트)가 이 모듈로 치환한 **런타임 사본**을 만들어 쓴다.
    - 치환 후 `__REPO__` 가 하나라도 남아 있으면 즉시 실패시킨다(조용한 오작동 금지).

사용
    from cfg_paths import repo_root, materialize
    cfg_path = materialize("scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml")
    # → artifacts/branchB/_runtime_cfg/inha_action_diffusion_1p1b.yaml (치환 완료본)
"""
from __future__ import annotations

from pathlib import Path

SENTINEL = "__REPO__"


def repo_root() -> Path:
    """이 파일 위치(scripts/branchB/)로부터 저장소 루트를 역산한다."""
    return Path(__file__).resolve().parents[2]


def substitute(text: str, root: Path | None = None) -> str:
    root = root or repo_root()
    return text.replace(SENTINEL, str(root))


def materialize(cfg_rel_or_abs: str | Path, out_dir: str | Path | None = None) -> Path:
    """config 의 치환 사본을 만들어 그 경로를 돌려준다. 원본은 건드리지 않는다."""
    root = repo_root()
    src = Path(cfg_rel_or_abs)
    if not src.is_absolute():
        src = root / src
    text = src.read_text(encoding="utf-8")
    n_hits = text.count(SENTINEL)
    out = Path(out_dir) if out_dir else (root / "artifacts" / "branchB" / "_runtime_cfg")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / src.name
    new_text = substitute(text, root)
    if SENTINEL in new_text:
        raise RuntimeError(f"치환 실패: {SENTINEL} 가 남아 있다 ({src})")
    dst.write_text(new_text, encoding="utf-8")
    print(f"[cfg_paths] {src.name}: {SENTINEL} {n_hits}곳 → {root}  ⇒ {dst}")
    return dst


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:] or ["scripts/branchB/configs/train/inha_action_diffusion_1p1b.yaml"]:
        materialize(p)
