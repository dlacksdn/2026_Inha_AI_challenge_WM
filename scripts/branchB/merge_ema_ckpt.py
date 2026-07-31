"""체크포인트의 EMA 사본(model_ema.*)을 본체(model.diffusion_model.*)에 **덮어써서** 새 ckpt 를 만든다.

배경 (013 §4 함정 ③)
-------------------
EMA(지수이동평균)는 학습 중 가중치의 "부드러운 평균 사본"이다. 보통 원본(raw)보다 성능이 좋다.
그런데 이 프로젝트에서는 EMA 를 **쓰지 못하고 있었다**. 이유는 구조적이다.

  · LitEma 는 `requires_grad=True` 인 파라미터만 사본으로 등록한다(ema.py L17-18).
  · 학습 때는 scope=action_temporal 로 551M(809키)만 학습하므로 EMA 도 809키만 만들어진다.
  · 생성 때는 scope 를 적용하지 않아 **전 파라미터가 학습 대상으로 취급**되고, LitEma 가
    1521키를 "모델을 막 만든 시점의 랜덤 가중치"로 채운다. 체크포인트에서 809키만 덮이므로
    나머지 712키는 랜덤인 채로 남고, `ema_scope()` 가 그 랜덤값으로 UNet 을 덮어써 노이즈가 됐다.
  · 그래서 생성에서 `use_ema=False` 로 EMA 를 통째로 껐다. → **EMA 이득을 0으로 버리고 있다.**

이 스크립트가 하는 일
-------------------
생성 코드를 건드리지 않고 EMA 를 살리는 가장 안전한 방법은 **체크포인트 자체를 바꾸는 것**이다.
  model_ema 의 809키를 대응하는 model.diffusion_model.* 키에 덮어쓰고,
  나머지 712키(동결 파라미터)는 원본 그대로 둔다.
결과 ckpt 는 `use_ema=False` 인 기존 생성 경로에서 그대로 쓰이며, 그때 UNet 은
"학습된 551M 은 EMA 평균 + 동결 889M 은 사전학습 원본" 이 된다. 이것이 EMA 의 올바른 의미다.

키 이름 대응
-----------
LitEma 는 버퍼 이름에 '.' 를 못 쓰므로 `name.replace(".", "")` 로 납작하게 만든다.
등록 기준 name 은 `model.named_parameters()`(= DiffusionWrapper) 이므로 "diffusion_model.xxx" 이다.
  ckpt 키                                  ← 대응 ←  ckpt 키
  model.diffusion_model.input_blocks.1...            model_ema.diffusion_modelinput_blocks1...
따라서 본체 키에서 "model." 을 떼고 '.' 를 지우면 EMA 버퍼 이름이 된다.

사용
----
  python scripts/branchB/merge_ema_ckpt.py <입력ckpt> <출력ckpt>
  # 검증만(파일 안 씀)
  python scripts/branchB/merge_ema_ckpt.py <입력ckpt> --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

MODEL_PREFIX = "model."
EMA_PREFIX = "model_ema."
SKIP_EMA_KEYS = {"model_ema.decay", "model_ema.num_updates"}


def flat_name(model_key: str) -> str:
    """'model.diffusion_model.a.b.weight' → 'diffusion_modelabweight' (LitEma 버퍼 이름)."""
    assert model_key.startswith(MODEL_PREFIX)
    return model_key[len(MODEL_PREFIX):].replace(".", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="입력 체크포인트")
    ap.add_argument("dst", nargs="?", default=None, help="출력 체크포인트(생략+--dry-run 이면 안 씀)")
    ap.add_argument("--dry-run", action="store_true", help="매칭 검증만 하고 저장하지 않는다")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[merge_ema] 입력 없음: {src}")

    print(f"[merge_ema] 로드: {src} ({src.stat().st_size/2**30:.2f} GiB)", flush=True)
    obj = torch.load(str(src), map_location="cpu", weights_only=False)
    sd = obj["state_dict"]

    model_keys = [k for k in sd if k.startswith(MODEL_PREFIX) and not k.startswith(EMA_PREFIX)]
    ema_keys = [k for k in sd if k.startswith(EMA_PREFIX) and k not in SKIP_EMA_KEYS]
    print(f"[merge_ema] 본체 {len(model_keys)}키 / EMA {len(ema_keys)}키", flush=True)

    # 본체 키 → 납작이름 사전. 납작화 충돌이 있으면 즉시 중단한다(잘못 덮어쓰면 조용히 망가진다).
    flat2model: dict[str, str] = {}
    for k in model_keys:
        f = flat_name(k)
        if f in flat2model:
            raise SystemExit(f"[merge_ema] 납작이름 충돌: {f} ← {flat2model[f]} / {k}")
        flat2model[f] = k

    matched, missing, shape_bad = [], [], []
    for ek in ema_keys:
        f = ek[len(EMA_PREFIX):]
        mk = flat2model.get(f)
        if mk is None:
            missing.append(ek)
            continue
        if tuple(sd[ek].shape) != tuple(sd[mk].shape):
            shape_bad.append((ek, tuple(sd[ek].shape), tuple(sd[mk].shape)))
            continue
        matched.append((ek, mk))

    n_par = sum(sd[mk].numel() for _, mk in matched)
    print(f"[merge_ema] 매칭 {len(matched)}키 / {n_par/1e6:.2f}M "
          f"| 대응없음 {len(missing)} | shape불일치 {len(shape_bad)}", flush=True)
    if missing:
        print(f"[merge_ema]   대응없음 예시: {missing[:5]}", flush=True)
    if shape_bad:
        print(f"[merge_ema]   shape불일치 예시: {shape_bad[:5]}", flush=True)
    if missing or shape_bad:
        raise SystemExit("[merge_ema] 대응되지 않는 EMA 키가 있다 — 덮어쓰지 않고 중단한다.")

    # 실제로 값이 얼마나 다른지(=EMA 가 의미 있게 다른지) 몇 개만 재본다.
    diffs = []
    for ek, mk in matched[:: max(1, len(matched) // 12)][:12]:
        a, b = sd[mk].float(), sd[ek].float()
        rel = float((a - b).norm() / (a.norm() + 1e-12))
        diffs.append((mk.split("model.diffusion_model.")[-1][:44], rel))
    print("[merge_ema] raw 대비 EMA 상대차(표본):", flush=True)
    for n, r in diffs:
        print(f"[merge_ema]   {n:<46} {r:.6f}", flush=True)
    mean_rel = sum(r for _, r in diffs) / len(diffs)
    print(f"[merge_ema] 평균 상대차 {mean_rel:.6f}"
          f"  ({'거의 동일 — EMA 이득 미미할 것' if mean_rel < 1e-4 else '유의하게 다름 — 실측 가치 있음'})",
          flush=True)

    if args.dry_run or args.dst is None:
        print("[merge_ema] dry-run: 저장하지 않는다.", flush=True)
        return

    for ek, mk in matched:
        sd[mk] = sd[ek].clone()

    # EMA 블록은 결과 ckpt 에서 불필요하다(생성은 use_ema=False). 지우면 2.2GB 절약된다.
    for k in list(sd):
        if k.startswith(EMA_PREFIX):
            del sd[k]

    obj["state_dict"] = sd
    obj["branchB_ema_merged"] = {"src": str(src), "merged_keys": len(matched),
                                 "merged_params_m": round(n_par / 1e6, 3)}
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(dst))
    print(f"[merge_ema] 저장 → {dst} ({dst.stat().st_size/2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
