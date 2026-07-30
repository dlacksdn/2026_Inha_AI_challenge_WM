"""1.1B 가중치 정합 조사 — DynamiCrafter 사전학습 UNet을 우리 3D UNet에 몇 % 얹을 수 있나.

배경(열린 질문 4 / 005 §4):
  M1에서 1.1B 규격 UNet의 "추론 속도"는 예산에 든다고 확인했지만, 그때 UNet은 **랜덤 초기화**였다.
  branch B(1.1B 승격)가 의미를 가지려면 DynamiCrafter(DC) 사전학습 가중치를 실제로 로드해야 한다.
  그런데 backbone.ckpt의 UNet에는 temporal(시간축) 키가 없어 보였다(005 관찰).
  → 이 스크립트로 **정량화**한다: 이름·shape 기준으로 몇 개/몇 파라미터가 로드되고, 무엇이 스크래치인가.

방법(메모리 절약):
  - 우리 3D UNet(1.1B config)은 **meta device**로 빌드해 구조(키·shape)만 얻는다(실제 메모리 0).
  - backbone.ckpt는 torch.load(mmap=True)로 lazy 로드해 UNet 키만 훑는다.
  - baseline_diffusion.ckpt(학습된 11M)도 같이 비교해 "11M→1.1B 확대 시 무엇이 안 맞는지" 함께 본다.

출력: results/branchB/weight_fit_report.json + 콘솔 요약표
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

UNET_PREFIX = "model.diffusion_model."

# M1에서 검증된 1.1B(DynamiCrafter 규격) override
DC_1P1B_OVERRIDE = {
    "model_channels": 320,
    "channel_mult": [1, 2, 4, 4],
    "attention_resolutions": [4, 2, 1],
    "num_head_channels": 64,
}

# --variants 모드(008 §3.3 의 A/B/C 를 재현 + D 추가). 11M yaml 기준의 누적 override.
#   A = 폭만 1.1B (008 표의 96.36%)
#   B = A + use_scale_shift_norm False (008 표의 99.89%)
#   C = B + fs_condition True (DC의 fps_embedding 4키까지 흡수 → only_in_src 0)
#   D = C + add_act_time_emb True (액션을 concat 대신 **가산** → time_embed.2 도 일치. 009 §2.2 의 "선택 옵션")
VARIANTS = {
    "A_width_only": dict(DC_1P1B_OVERRIDE),
    "B_no_scale_shift": {**DC_1P1B_OVERRIDE, "use_scale_shift_norm": False},
    "C_B_plus_fs_cond": {**DC_1P1B_OVERRIDE, "use_scale_shift_norm": False, "fs_condition": True},
    "D_C_plus_act_add": {
        **DC_1P1B_OVERRIDE,
        "use_scale_shift_norm": False,
        "fs_condition": True,
        "add_act_time_emb": True,
    },
}


def build_unet_meta(challenge_kit: Path, train_config: Path, override: dict | None) -> dict:
    """UNet을 meta device로 빌드해 {키: shape} 만 반환 (메모리 미할당)."""
    sys.path.insert(0, str(challenge_kit / "libs" / "dynamicrafter"))
    sys.path.insert(0, str(challenge_kit / "src"))
    sys.path.insert(0, str(challenge_kit))
    sys.path.insert(0, str(challenge_kit.parent / "shared_libs" / "video_utils"))
    from lvdm.modules.networks.openaimodel3d import UNetModel  # noqa: E402

    cfg = OmegaConf.load(str(train_config))
    params = OmegaConf.to_container(cfg.model.params.unet_config.params, resolve=True)
    if override:
        params.update(override)
    with torch.device("meta"):
        unet = UNetModel(**params)
    return {k: tuple(v.shape) for k, v in unet.state_dict().items()}, params


def load_ckpt_unet_shapes(ckpt_path: Path) -> dict:
    """ckpt에서 UNet 파라미터의 {키(prefix 제거): shape} 만 뽑는다 (mmap lazy)."""
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    out = {}
    for k, v in sd.items():
        if k.startswith(UNET_PREFIX) and hasattr(v, "shape"):
            out[k[len(UNET_PREFIX):]] = tuple(v.shape)
    return out


def numel(shape: tuple) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def classify(key: str) -> str:
    """키를 기능 그룹으로 분류(무엇이 스크래치인지 해석하기 위해)."""
    k = key.lower()
    if "temporal" in k or re.search(r"time_stack|temopral", k):
        return "temporal(시간축)"
    if k.startswith("action") or "null_action" in k:
        return "action(액션분기)"
    if "attn" in k or "transformer" in k or "proj_in" in k or "proj_out" in k or "norm" in k and "in_layers" not in k:
        return "spatial-attn"
    if "time_embed" in k or "emb_layers" in k:
        return "timestep-emb"
    if "fps" in k or "fs_" in k:
        return "fps-cond"
    return "conv/기타"


def compare(ours: dict, src: dict) -> dict:
    """우리 UNet 키셋 vs 소스 ckpt 키셋 대조 → 정합 지표. (키 목록을 전부 남긴다)"""
    exact, shape_mismatch, only_ours, only_src = [], [], [], []
    for k, s in ours.items():
        if k in src:
            (exact if src[k] == s else shape_mismatch).append(k)
        else:
            only_ours.append(k)
    for k in src:
        if k not in ours:
            only_src.append(k)
    p_exact = sum(numel(ours[k]) for k in exact)
    p_total = sum(numel(s) for s in ours.values())
    p_scratch = p_total - p_exact
    return {
        "ours_keys": len(ours), "src_keys": len(src),
        "exact_match_keys": len(exact),
        "shape_mismatch_keys": len(shape_mismatch),
        "only_in_ours_keys": len(only_ours),
        "only_in_src_keys": len(only_src),
        "loadable_param_ratio": p_exact / p_total if p_total else 0.0,
        "ours_params_m": p_total / 1e6,
        "loadable_params_m": p_exact / 1e6,
        "scratch_params_m": p_scratch / 1e6,
        "shape_mismatch": [{"key": k, "ours": ours[k], "src": src[k]} for k in shape_mismatch],
        "only_in_ours": only_ours,
        "only_in_src": only_src,
    }


def run_variants(ck: Path, base_cfg: Path, backbone: Path, our_cfg: Path | None, out: Path) -> None:
    """008 §3.3 의 A/B/C(+D, +우리 config)를 한 번에 대조해 로드율 표를 만든다."""
    print("[probe] backbone.ckpt(DC 사전학습) UNet 키 로드 ...")
    dc = load_ckpt_unet_shapes(backbone)
    print(f"  키 {len(dc)}개, 파라미터 {sum(numel(s) for s in dc.values())/1e6:.2f}M")

    cases = [(name, base_cfg, ov) for name, ov in VARIANTS.items()]
    if our_cfg is not None:
        cases.append(("OURS_yaml", our_cfg, None))

    report = {}
    for name, cfg_path, ov in cases:
        ours, params = build_unet_meta(ck, cfg_path, ov)
        r = compare(ours, dc)
        r["config_source"] = str(cfg_path)
        r["override"] = ov
        r["unet_params_resolved"] = params
        report[name] = r
        print(f"  [{name}] 키 {r['ours_keys']} / 정확일치 {r['exact_match_keys']} / "
              f"shape불일치 {r['shape_mismatch_keys']} / 우리만 {r['only_in_ours_keys']} / "
              f"DC만 {r['only_in_src_keys']} → 로드율 {r['loadable_param_ratio']*100:.2f}%")

    rep_path = out / "weight_fit_variants_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 104)
    print("1.1B 가중치 정합 — 설정별 로드율 (소스: backbone.ckpt = DynamiCrafter_512 사전학습)")
    print("=" * 104)
    hdr = f"{'변형':<20}{'우리키':>7}{'정확일치':>9}{'shape불일치':>12}{'우리만':>7}{'DC만':>6}{'로드가능M':>11}{'로드율':>9}"
    print(hdr)
    print("-" * 104)
    for name, r in report.items():
        print(f"{name:<20}{r['ours_keys']:>7}{r['exact_match_keys']:>9}{r['shape_mismatch_keys']:>12}"
              f"{r['only_in_ours_keys']:>7}{r['only_in_src_keys']:>6}"
              f"{r['loadable_params_m']:>10.1f}M{r['loadable_param_ratio']*100:>8.2f}%")
    print("-" * 104)
    for name, r in report.items():
        print(f"\n[{name}] 스크래치로 남는 것 ({r['scratch_params_m']:.2f}M)")
        for k in r["only_in_ours"]:
            print(f"    only-ours       {k}")
        for m in r["shape_mismatch"][:8]:
            print(f"    shape-mismatch  {m['key']}  ours{list(m['ours'])} vs dc{list(m['src'])}")
        if len(r["shape_mismatch"]) > 8:
            print(f"    ... shape-mismatch {len(r['shape_mismatch'])}키 중 8개만 표시")
        for k in r["only_in_src"]:
            print(f"    (DC에만 있어 버려짐) {k}")
    print(f"\n[probe] 리포트: {rep_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge-kit", default="open/baseline/challenge_kit")
    ap.add_argument("--train-config", default="open/baseline/challenge_kit/configs/train/inha_action_diffusion_11M.yaml")
    ap.add_argument("--backbone", default="open/baseline/checkpoints/backbone.ckpt")
    ap.add_argument("--baseline-ckpt", default="open/baseline/checkpoints/baseline_diffusion.ckpt")
    ap.add_argument("--out", default="results/branchB")
    ap.add_argument("--variants", action="store_true",
                    help="008 §3.3 의 A/B/C(+D, +우리 yaml) 설정별 로드율 표를 만든다(기존 리포트 미덮어씀)")
    ap.add_argument("--our-config", default=None,
                    help="--variants 와 함께: 우리 1.1B 학습 yaml 경로(override 없이 그대로 빌드해 검증)")
    args = ap.parse_args()

    if args.variants:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        run_variants(Path(args.challenge_kit), Path(args.train_config), Path(args.backbone),
                     Path(args.our_config) if args.our_config else None, out)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(args.challenge_kit)

    print("[probe] 우리 3D UNet(11M 기본 config) meta 빌드 ...")
    ours_11m, params_11m = build_unet_meta(ck, Path(args.train_config), None)
    print(f"  키 {len(ours_11m)}개, 파라미터 {sum(numel(s) for s in ours_11m.values())/1e6:.2f}M")

    print("[probe] 우리 3D UNet(1.1B override) meta 빌드 ...")
    ours_1b, params_1b = build_unet_meta(ck, Path(args.train_config), DC_1P1B_OVERRIDE)
    print(f"  키 {len(ours_1b)}개, 파라미터 {sum(numel(s) for s in ours_1b.values())/1e6:.2f}M")

    print("[probe] backbone.ckpt(DC 사전학습) UNet 키 로드 ...")
    dc = load_ckpt_unet_shapes(Path(args.backbone))
    print(f"  키 {len(dc)}개, 파라미터 {sum(numel(s) for s in dc.values())/1e6:.2f}M")

    print("[probe] baseline_diffusion.ckpt(학습된 11M) UNet 키 로드 ...")
    trained11 = load_ckpt_unet_shapes(Path(args.baseline_ckpt))
    print(f"  키 {len(trained11)}개, 파라미터 {sum(numel(s) for s in trained11.values())/1e6:.2f}M")

    report = {}
    for label, ours in [("ours_1p1b", ours_1b), ("ours_11m", ours_11m)]:
        for src_label, src in [("dc_backbone", dc), ("trained_11m", trained11)]:
            exact, shape_mismatch, only_ours, only_src = [], [], [], []
            for k, s in ours.items():
                if k in src:
                    (exact if src[k] == s else shape_mismatch).append(k)
                else:
                    only_ours.append(k)
            for k in src:
                if k not in ours:
                    only_src.append(k)

            p_exact = sum(numel(ours[k]) for k in exact)
            p_total = sum(numel(s) for s in ours.values())
            grp_missing = defaultdict(lambda: [0, 0])  # group -> [키수, 파라미터수]
            for k in only_ours + shape_mismatch:
                g = classify(k)
                grp_missing[g][0] += 1
                grp_missing[g][1] += numel(ours[k])

            key = f"{label}__from__{src_label}"
            report[key] = {
                "ours_keys": len(ours), "src_keys": len(src),
                "exact_match_keys": len(exact),
                "shape_mismatch_keys": len(shape_mismatch),
                "only_in_ours_keys": len(only_ours),
                "only_in_src_keys": len(only_src),
                "loadable_param_ratio": p_exact / p_total if p_total else 0.0,
                "ours_params_m": p_total / 1e6,
                "loadable_params_m": p_exact / 1e6,
                "scratch_by_group": {g: {"keys": v[0], "params_m": v[1] / 1e6}
                                     for g, v in sorted(grp_missing.items(), key=lambda kv: -kv[1][1])},
                "sample_only_in_ours": only_ours[:12],
                "sample_shape_mismatch": [{"key": k, "ours": ours[k], "src": src[k]} for k in shape_mismatch[:12]],
                "sample_only_in_src": only_src[:12],
            }

    (out / "weight_fit_report.json").write_text(
        json.dumps({"config_1p1b": params_1b, "results": report}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n" + "=" * 96)
    print("1.1B 가중치 정합 조사 — 사전학습 UNet을 우리 3D UNet에 얹을 수 있는 비율")
    print("=" * 96)
    print(f"{'조합':<30}{'우리키':>8}{'정확일치':>10}{'shape불일치':>12}{'우리만':>8}{'로드가능%':>11}")
    print("-" * 96)
    for k, r in report.items():
        print(f"{k:<30}{r['ours_keys']:>8}{r['exact_match_keys']:>10}{r['shape_mismatch_keys']:>12}"
              f"{r['only_in_ours_keys']:>8}{r['loadable_param_ratio']*100:>10.1f}%")
    print("-" * 96)
    key = "ours_1p1b__from__dc_backbone"
    print(f"\n[핵심] 1.1B UNet에 DC 사전학습을 얹을 때 스크래치로 남는 부분:")
    for g, v in report[key]["scratch_by_group"].items():
        print(f"   {g:<18} 키 {v['keys']:>5}개 / {v['params_m']:>8.2f}M")
    print(f"   → 로드 가능: {report[key]['loadable_params_m']:.1f}M / {report[key]['ours_params_m']:.1f}M "
          f"({report[key]['loadable_param_ratio']*100:.1f}%)")
    print(f"\n[probe] 리포트: {out / 'weight_fit_report.json'}")


if __name__ == "__main__":
    main()
