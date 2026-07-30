"""branch B 학습 런처 — baseline 코드를 수정하지 않고 1.1B UNet 을 DC 사전학습에서 warm-start 한다.

baseline 의 train_diffusion.py 와 동일하되 세 가지만 추가한다.
  (1) **UNet warm-start**: get_model() 은 config 의 only_reload_modules 대로 VAE/CLIP/embedder/proj 만
      로드하고 UNet 은 랜덤이다. 그 직후 `weight_loader.warm_start_full_model()` 로 DC UNet 을 얹는다.
      - shape 불일치 키는 **미리 제거**한다(strict=False 로도 예외가 나기 때문 — 010 §4.1).
      - `action_embed` 마지막 층을 0초기화 → 학습 시작점이 DC 와 정확히 동일(010 §5.3, cos 1.00000).
  (2) **학습 범위(scope)**: full / action_temporal / action_only. 32GB 에서는 전체 파인튜닝이
      산술상 불가(31.2GB + 활성화)이므로 action_temporal(21.3GB)을 쓴다.
  (3) **EMA 재초기화**: warm-start 이후에 LitEma 를 다시 만들어야 EMA 가 랜덤에서 출발하지 않는다.

환경변수
  BRANCHB_WARMSTART_CKPT : 얹을 ckpt (기본: config 의 model.pretrained_checkpoint = DC backbone)
  BRANCHB_ZERO_INIT      : 1/0 (기본: warm-start 소스가 backbone 이면 1, 우리 학습본이면 0)
  BRANCHB_TRAIN_SCOPE    : full | action_temporal | action_only (기본 full)
  BRANCHB_BUILD_ONLY     : 1 이면 모델 빌드·warm-start·범위설정까지만 하고 종료(스모크 전 점검용)

실행은 scripts/branchB/run_1p1b_train.sh 가 감싼다(경로 치환·PYTHONPATH·랭크 env).
"""
import json
import os
import sys
from pathlib import Path

import torch
from lvdm.ema import LitEma
from lvdm.utils.train import (
    get_env_vars,
    get_model,
    get_parser,
    get_trainer,
    prepare_logger,
    set_model_lr,
)
from lvdm.utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_scope import apply_scope  # noqa: E402
from weight_loader import warm_start_full_model  # noqa: E402


def main() -> None:
    now, local_rank, global_rank, num_rank = get_env_vars()

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    seed_everything(args.seed)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    logger, workdir, ckptdir, cfgdir, loginfo = prepare_logger(lightning_config, config, global_rank, now)

    ## MODEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    model = get_model(config.model, workdir)
    unet = model.model.diffusion_model
    n_unet = sum(p.numel() for p in unet.parameters())
    print(f">>> [branchB] UNet {n_unet/1e6:.2f}M 파라미터 / "
          f"add_act_time_emb={getattr(unet, 'add_act_time_emb', None)} / "
          f"fs_condition={getattr(unet, 'fs_condition', None)}", flush=True)

    # ── (1) warm-start ────────────────────────────────────────────────────────────
    ws = os.environ.get("BRANCHB_WARMSTART_CKPT", "").strip() or str(config.model.pretrained_checkpoint)
    assert os.path.exists(ws), f"[branchB] warm-start ckpt 없음: {ws}"
    is_backbone = os.path.abspath(ws) == os.path.abspath(str(config.model.pretrained_checkpoint))
    zi_env = os.environ.get("BRANCHB_ZERO_INIT", "").strip()
    zero_init = (zi_env == "1") if zi_env else is_backbone
    ws_info = warm_start_full_model(model, ws, zero_init_action=zero_init)
    ws_info["source_is_backbone"] = is_backbone

    if ws_info["unet_unexpected"]:
        raise RuntimeError(f"[branchB] unet_unexpected 가 있다: {ws_info['unet_unexpected'][:8]}")
    if is_backbone and len(ws_info["unet_missing"]) != 5:
        print(f">>> [branchB] 경고: backbone warm-start 인데 unet_missing 이 5키가 아니다 "
              f"({len(ws_info['unet_missing'])}키). config 의 add_act_time_emb 를 확인하라.", flush=True)

    # ── (2) 학습 범위 ─────────────────────────────────────────────────────────────
    scope = os.environ.get("BRANCHB_TRAIN_SCOPE", "full").strip()
    scope_info = apply_scope(unet, scope)
    print(f">>> [branchB] scope={scope}: 학습 {scope_info['trainable_params_m']}M "
          f"({scope_info['trainable_share_pct']}%) / 동결 {scope_info['frozen_params_m']}M", flush=True)
    for g, v in scope_info["by_group"].items():
        print(f">>>      {g:<10} {v['params_m']:>9.2f}M  학습={v['trainable']}", flush=True)

    model = set_model_lr(model, config.model, num_rank, config.data.params.batch_size)

    # ── (3) EMA 를 warm-start 된 가중치로 재초기화 ────────────────────────────────
    if model.use_ema:
        model.model_ema = LitEma(model.model)
        print(">>> [branchB] LitEma 를 warm-start 가중치로 재초기화했다.", flush=True)

    run_info = {"warm_start": ws_info, "scope": scope_info,
                "unet_params_m": round(n_unet / 1e6, 3), "workdir": workdir,
                "lr": float(config.model.base_learning_rate), "seed": args.seed}
    Path(workdir, "branchB_run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f">>> [branchB] run info → {Path(workdir, 'branchB_run_info.json')}", flush=True)

    if os.environ.get("BRANCHB_BUILD_ONLY", "").strip() == "1":
        if torch.cuda.is_available():
            model.cuda()
            torch.cuda.synchronize()
            print(f">>> [branchB] BUILD_ONLY: 모델 GPU 적재 후 VRAM "
                  f"{torch.cuda.memory_allocated()/2**30:.2f}GB "
                  f"(peak {torch.cuda.max_memory_allocated()/2**30:.2f}GB)", flush=True)
        print(">>> [branchB] BUILD_ONLY=1 → 학습 없이 종료(점검 통과).", flush=True)
        return

    ## DATA >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = instantiate_from_config(config.data)
    data.setup()

    ## TRAINER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trainer = get_trainer(
        lightning_config=lightning_config,
        trainer_config=trainer_config,
        config=config,
        args=args,
        workdir=workdir,
        ckptdir=ckptdir,
        logger=logger,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
