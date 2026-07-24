"""M1 — DynamiCrafter 1.1B UNet 추론 예산 실측 (RTX 5090 전용).

목적:
  (a) 216샘플을 1시간(3600s) 안에 생성 가능한 (ddim_steps, batch, CFG) 조합을 찾고,
  (b) sec/샘플·peak VRAM 을 재서 RTX PRO 6000(96GB) 배율 r 추정의 근거를 만든다.
       (004 §5.2 의 A100 0.4s/NFE 기준과 대조.)

설계 원칙 (CLAUDE.md · task 준수):
  - submission_kit/baseline 코드는 '수정하지 않고' import만 한다.
    gen_baseline.sh 와 동일하게 PYTHONPATH + CWD 만 맞춰 원본을 그대로 부른다.
  - 1.1B UNet: 학습 config(inha_action_diffusion_11M.yaml)의 unet_config.params 중
    아키텍처 4개 키만 DC_512 규격으로 override(model_channels 320 / channel_mult [1,2,4,4]
    / attention_resolutions [4,2,1] / num_head_channels 64). 나머지는 11M 그대로.
  - VAE/CLIP/embedder/image_proj 는 backbone.ckpt 에서 only_reload_modules 로 로드.
    UNet 은 랜덤 초기화(타이밍은 가중치와 무관). 빌드 후 UNet 파라미터 수를 로그로 남긴다.
  - 입력은 train 홀드아웃(artifacts/holdout)만 사용한다. eval 데이터는 절대 쓰지 않는다.
  - 각 조합: 워밍업 2샘플 → 8샘플 측정. sec/샘플·peak VRAM 기록. OOM 은 표에 OOM 으로.
  - 결과는 촘촘하게 저장(셀마다 results/m1/m1_report.json 갱신 + 로그 append).

사용:
  conda run -n wm python scripts/measure_1p1b.py \
      --holdout artifacts/holdout --out results/m1
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── 저장소 경로 확정 (chdir 이전에 절대경로로 고정) ────────────────────────────
REPO = Path(__file__).resolve().parent.parent
CK = REPO / "open" / "baseline" / "challenge_kit"
VIDEO_UTILS = REPO / "open" / "baseline" / "shared_libs" / "video_utils"
BACKBONE = REPO / "open" / "baseline" / "checkpoints" / "backbone.ckpt"

# ── 베이스라인 import 환경 재현(gen_baseline.sh 와 동일) ───────────────────────
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
for p in (CK / "libs" / "dynamicrafter", CK / "src", CK, VIDEO_UTILS):
    sys.path.insert(0, str(p))

# 1.1B 로 바꿀 아키텍처 4개 키 (task 지정; 나머지 unet_config.params 는 11M 그대로)
UNET_1P1B_OVERRIDE = {
    "model_channels": 320,
    "channel_mult": [1, 2, 4, 4],
    "attention_resolutions": [4, 2, 1],
    "num_head_channels": 64,
}

# 측정 그리드 (task 지정)
DDIM_STEPS = [50, 25, 16, 10]
BATCHES = [1, 2, 4]
CFG_SCALES = [("off", 1.0), ("on", 2.5)]  # unconditional_guidance_scale

BUDGET_S = 3600.0          # 216샘플 목표 시간
N_SUBMISSION = 216         # eval 제출 규모
A100_S_PER_NFE = 0.4       # 004 §5.2 기준 단가 (DC 320x512, 20s/50step)


def log(msg: str, logf) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


def build_model(device, logf):
    """11M config 로드 → unet 4키 override → get_model(backbone only_reload)."""
    from omegaconf import OmegaConf
    from lvdm.utils.train import get_model

    cfg_path = CK / "configs" / "train" / "inha_action_diffusion_11M.yaml"
    model_cfg = OmegaConf.load(str(cfg_path)).model
    OmegaConf.set_struct(model_cfg, False)
    # backbone.ckpt 절대경로로 고정(CWD 무관하게 안전)
    model_cfg.pretrained_checkpoint = str(BACKBONE)
    # open_clip ViT-H-14(~3.9GB) 다운로드 회피: version=None → pretrained=None로 랜덤 빌드.
    # 직후 only_reload_modules 가 backbone.ckpt 에서 실제 cond_stage_model/embedder 를 로드하므로
    # 최종 가중치는 동일하고, 타이밍은 가중치값과 무관(=M1 목적에 정확). baseline 코드는 미수정(config override).
    model_cfg.params.cond_stage_config.params.version = None
    model_cfg.params.img_cond_stage_config.params.version = None
    # 배포형 1.1B 추론과 동일: fp16 가중치 + EMA shadow 제거.
    #  - EMA: 랜덤 UNet이라 EMA 사본은 무의미하고 ~5.9GB만 낭비 → 끔.
    #  - fp16: 004 §5.2 기준치(fp16, 0.4s/NFE, 12.8GB)와 같은 정밀도 → r 비교가 유효.
    model_cfg.params.use_ema = False
    up = model_cfg.params.unet_config.params
    before = {k: OmegaConf.select(up, k) for k in UNET_1P1B_OVERRIDE}
    for k, v in UNET_1P1B_OVERRIDE.items():
        up[k] = v
    log(f"UNet override: {before}  ->  {UNET_1P1B_OVERRIDE}", logf)

    t0 = time.time()
    model = get_model(model_cfg)          # instantiate + load_checkpoints(only_reload)
    # baseline 추론과 동일: fp32 가중치 + autocast(compute는 fp16 TensorCore).
    # (baseline get_input 이 입력을 .float() 강제 → 순수 half 불가. autocast 가 정석.)
    # 메모리는 use_ema=False 로 UNet shadow(~5.9GB) 제거로 절감. fp16 배포시 가중치는 절반.
    model = model.to(device).eval()
    build_s = time.time() - t0

    unet = model.model.diffusion_model
    n_unet = sum(p.numel() for p in unet.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    log(f"모델 빌드 {build_s:.1f}s | UNet 파라미터 {n_unet/1e6:.1f}M "
        f"(≈{n_unet/1e9:.3f}B) | 전체 {n_total/1e6:.1f}M", logf)
    return model, {"unet_params_m": n_unet / 1e6, "total_params_m": n_total / 1e6,
                   "build_seconds": build_s}


def load_holdout_batches(model, holdout: Path, action_stats: str, device, need_ids: int, logf):
    """홀드아웃에서 need_ids 개 표본을 읽어 (device 위) 배치 재료로 준비."""
    from scripts.eval.feature_csv_utils import (
        build_inference_batch, list_challenge_sample_ids, load_action_stats,
    )
    ids_all = list_challenge_sample_ids(holdout)
    if not ids_all:
        raise SystemExit(f"홀드아웃 표본 없음: {holdout} (먼저 G2 로 생성)")
    # 필요한 만큼 순환하여 확보
    ids = [ids_all[i % len(ids_all)] for i in range(need_ids)]
    a_mean, a_std = load_action_stats(action_stats)
    if a_mean is not None:
        a_mean, a_std = a_mean.to(device), a_std.to(device)
    log(f"홀드아웃 표본 {len(ids_all)}개 중 {need_ids}개 사용(순환). fps=6 고정", logf)
    return ids, a_mean, a_std, build_inference_batch


def run_cell(model, sampler_cls, make_batch, ids, a_mean, a_std, holdout, device,
             steps: int, batch: int, cfg_scale: float,
             warmup_samples: int, measure_samples: int, logf):
    """한 (steps,batch,cfg) 셀 측정 → sec/샘플·peak VRAM. OOM 시 예외 표기."""
    import torch

    sampler = sampler_cls(model)
    shape = (model.channels, model.temporal_length, *model.image_size)  # (4,16,40,64)

    # eval config 의 ddim_kwargs 를 base 로, steps·CFG 만 override (baseline 과 동일 동작).
    base_kwargs = dict(ddim_eta=1.0, timestep_spacing="uniform_trailing",
                       guidance_rescale=0.7, verbose=False)

    def gen_one_batch(bidx: int):
        # 배치용 sample_id 슬라이스(순환)
        s = (bidx * batch) % len(ids)
        bids = [ids[(s + j) % len(ids)] for j in range(batch)]
        batch_in = make_batch(holdout, bids, 320, 512, True, 6, a_mean, a_std, device)
        with torch.no_grad(), model.ema_scope("M1"), torch.cuda.amp.autocast():
            z, c, uc, cond_mask, _log, kw = model.prepare_batch_for_inference(batch_in)
            sk = dict(base_kwargs)
            sk.update(kw)  # clean_cond, fs
            sk["unconditional_guidance_scale"] = cfg_scale
            samples, _ = sampler.sample(
                steps, batch_size=z.shape[0], shape=shape,
                conditioning=c, unconditional_conditioning=uc,
                mask=cond_mask, x0=z, **sk,
            )
            _ = model.decode_first_stage(samples)

    import math
    warm_batches = max(1, math.ceil(warmup_samples / batch))
    meas_batches = math.ceil(measure_samples / batch)

    try:
        # 워밍업(측정 제외)
        for b in range(warm_batches):
            gen_one_batch(b)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        # 측정
        t0 = time.time()
        for b in range(meas_batches):
            gen_one_batch(warm_batches + b)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1e9
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache(); gc.collect()
            return {"status": "OOM", "error": str(e).splitlines()[0][:200]}
        raise

    n_meas = meas_batches * batch
    sec_per_sample = elapsed / n_meas
    nfe = steps * (2 if cfg_scale != 1.0 else 1)
    return {
        "status": "ok",
        "sec_per_sample": sec_per_sample,
        "peak_vram_gb": peak,
        "nfe": nfe,
        "s_per_nfe": sec_per_sample / nfe,
        "n_measured": n_meas,
        "elapsed_s": elapsed,
        "proj_216_s": N_SUBMISSION * sec_per_sample,
        "within_budget": (N_SUBMISSION * sec_per_sample) <= BUDGET_S,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=str(REPO / "artifacts" / "holdout"))
    ap.add_argument("--out", default=str(REPO / "results" / "m1"))
    ap.add_argument("--action-stats", default=str(REPO / "open/data/train/so100_action_statistics.json"))
    ap.add_argument("--warmup-samples", type=int, default=2)
    ap.add_argument("--measure-samples", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="스모크: 빌드+1셀만(steps=10,batch=1,cfg off)로 파이프라인 검증")
    args = ap.parse_args()

    # 그리드(스모크면 축소)
    ddim_list = [10] if args.smoke else DDIM_STEPS
    batch_list = [1] if args.smoke else BATCHES
    cfg_list = [("off", 1.0)] if args.smoke else CFG_SCALES

    holdout = Path(args.holdout).resolve()
    action_stats = str(Path(args.action_stats).resolve())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # baseline 상대경로가 풀리도록 CWD 를 challenge_kit 으로(입출력은 위에서 절대경로화 완료)
    os.chdir(CK)

    import torch
    from lvdm.models.samplers.ddim import DDIMSampler

    logf = (out / "m1_run.log").open("a", encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log(f"===== M1 시작 | {gpu_name} | torch {torch.__version__} cu{torch.version.cuda} =====", logf)
    if not BACKBONE.exists():
        raise SystemExit(f"backbone.ckpt 없음: {BACKBONE}")

    model, model_info = build_model(device, logf)

    # 필요한 표본 수(가장 큰 batch·batch수 기준으로 넉넉히)
    need = max(batch_list) * (max(1, args.warmup_samples) + args.measure_samples) + max(batch_list)
    ids, a_mean, a_std, make_batch = load_holdout_batches(
        model, holdout, action_stats, device, need, logf)

    report = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "holdout": str(holdout),
            "unet_override": UNET_1P1B_OVERRIDE,
            "grid": {"ddim_steps": ddim_list, "batch": batch_list,
                     "cfg": [c[1] for c in cfg_list]},
            "smoke": args.smoke,
            "warmup_samples": args.warmup_samples,
            "measure_samples": args.measure_samples,
            "budget_s": BUDGET_S, "n_submission": N_SUBMISSION,
            "a100_s_per_nfe_ref": A100_S_PER_NFE,
        },
        "model": model_info,
        "cells": [],
    }

    oom_batches: set[int] = set()   # peak VRAM 은 batch 에 지배 → batch 단위로 OOM 단락
    total_cells = len(cfg_list) * len(batch_list) * len(ddim_list)
    done = 0
    for cfg_name, cfg_scale in cfg_list:
        for batch in batch_list:
            for steps in ddim_list:
                done += 1
                tag = f"[{done}/{total_cells}] steps={steps:>2} batch={batch} cfg={cfg_name}(x{cfg_scale})"
                if batch in oom_batches:
                    res = {"status": "OOM", "error": "batch OOM (선행 셀에서 확인, 단락)"}
                    log(f"{tag} -> OOM (단락)", logf)
                else:
                    res = run_cell(model, DDIMSampler, make_batch, ids, a_mean, a_std,
                                   holdout, device, steps, batch, cfg_scale,
                                   args.warmup_samples, args.measure_samples, logf)
                    if res["status"] == "OOM":
                        oom_batches.add(batch)
                        log(f"{tag} -> OOM", logf)
                    else:
                        log(f"{tag} -> {res['sec_per_sample']:.3f}s/샘플 | "
                            f"peak {res['peak_vram_gb']:.1f}GB | NFE {res['nfe']} | "
                            f"216샘플 {res['proj_216_s']/60:.1f}분 "
                            f"{'✅<60분' if res['within_budget'] else '❌>60분'}", logf)
                cell = {"steps": steps, "batch": batch, "cfg": cfg_name,
                        "cfg_scale": cfg_scale, **res}
                report["cells"].append(cell)
                # 촘촘 저장: 셀마다 갱신
                (out / "m1_report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 요약표 출력 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"M1 결과 — {gpu_name} | 1.1B UNet {model_info['unet_params_m']:.0f}M | "
          f"216샘플 예산 {BUDGET_S/60:.0f}분")
    print("=" * 92)
    hdr = f"{'steps':>5}{'batch':>6}{'cfg':>6}{'NFE':>5}{'sec/샘플':>10}{'peakVRAM':>10}{'216샘플(분)':>12}{'예산':>6}"
    print(hdr); print("-" * 92)
    ok_cells = [c for c in report["cells"] if c["status"] == "ok"]
    for c in report["cells"]:
        if c["status"] == "OOM":
            print(f"{c['steps']:>5}{c['batch']:>6}{c['cfg']:>6}{'':>5}{'OOM':>10}{'OOM':>10}{'':>12}{'':>6}")
        else:
            print(f"{c['steps']:>5}{c['batch']:>6}{c['cfg']:>6}{c['nfe']:>5}"
                  f"{c['sec_per_sample']:>10.3f}{c['peak_vram_gb']:>9.1f}G"
                  f"{c['proj_216_s']/60:>12.1f}{'✅' if c['within_budget'] else '❌':>6}")
    print("-" * 92)

    within = [c for c in ok_cells if c["within_budget"]]
    report["summary"] = {
        "n_ok": len(ok_cells), "n_within_budget": len(within),
        "within_budget_cells": [
            {"steps": c["steps"], "batch": c["batch"], "cfg": c["cfg"],
             "sec_per_sample": c["sec_per_sample"], "proj_216_min": c["proj_216_s"] / 60,
             "peak_vram_gb": c["peak_vram_gb"]}
            for c in sorted(within, key=lambda x: x["proj_216_s"])],
    }
    # 5090 per-NFE 중앙값(측정 기반) → A100 대비 속도비
    if ok_cells:
        import statistics
        s_per_nfe_5090 = statistics.median(c["s_per_nfe"] for c in ok_cells)
        report["summary"]["s_per_nfe_5090_median"] = s_per_nfe_5090
        report["summary"]["speed_ratio_5090_vs_a100"] = A100_S_PER_NFE / s_per_nfe_5090
        print(f"\n5090 s/NFE(중앙값) = {s_per_nfe_5090:.4f}  |  "
              f"A100(0.4) 대비 5090 ≈ {A100_S_PER_NFE/s_per_nfe_5090:.2f}x")
    if within:
        best = min(within, key=lambda x: x["proj_216_s"])
        print(f"60분 이내 조합 {len(within)}개. 최속: steps={best['steps']} "
              f"batch={best['batch']} cfg={best['cfg']} "
              f"→ {best['proj_216_s']/60:.1f}분")
    else:
        print("\n⚠️ 60분 이내 조합 없음 — 저스텝/저CFG 재설계 필요")

    (out / "m1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"===== M1 완료 | 리포트 {out/'m1_report.json'} =====", logf)
    logf.close()


if __name__ == "__main__":
    main()
