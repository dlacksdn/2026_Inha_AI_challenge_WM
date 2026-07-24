"""M3 Part B 학습 래퍼 — baseline 코드를 수정하지 않고, 학습된 11M UNet에서 warm-start한다.

train_diffusion.py 와 동일하되 딱 한 곳만 추가:
  get_model(backbone에서 VAE/CLIP만 로드, UNet 랜덤) 직후, LitEma 초기화 전에
  baseline_diffusion.ckpt(학습된 11M UNet, model.diffusion_model.* + model_ema.*)를
  strict=False 로 덧씌운다 → UNet warm-start.

warm-start 체크포인트 경로는 dotlist 파싱과 충돌하지 않도록 ENV 로 전달한다:
  M3_WARMSTART_UNET=<abs path to ckpt>   (비어 있으면 scratch UNet)

실행: torch.distributed.launch 로 LOCAL_RANK/RANK/WORLD_SIZE 를 준다 (train.sh 와 동일).
  나머지 dotlist override(예: action_dropout_prob, max_steps)는 base 뒤에 그대로 붙인다.
"""
import os

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

if __name__ == "__main__":
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

    # ---- M3 warm-start (baseline 코드 미수정; 여기서만 UNet 가중치를 덧씌움) ----
    ws = os.environ.get("M3_WARMSTART_UNET", "").strip()
    if ws:
        assert os.path.exists(ws), f"[M3] warm-start ckpt NOT found: {ws}"
        pl_sd = torch.load(ws, map_location="cpu")
        sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd
        n_unet_in_ckpt = sum(1 for k in sd if k.startswith("model.diffusion_model"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_unet_missing = sum(1 for k in missing if k.startswith("model.diffusion_model"))
        print(
            f">>> [M3] warm-started from {ws}\n"
            f">>>      ckpt keys={len(sd)} (unet={n_unet_in_ckpt}); "
            f"after load: missing={len(missing)} (unet_missing={n_unet_missing}), unexpected={len(unexpected)}",
            flush=True,
        )
        if n_unet_missing > 0:
            print(f">>> [M3] WARNING: {n_unet_missing} UNet keys NOT matched by warm-start ckpt!", flush=True)
    else:
        print(">>> [M3] no warm-start: UNet is random (scratch).", flush=True)
    # -------------------------------------------------------------------------

    model = set_model_lr(model, config.model, num_rank, config.data.params.batch_size)

    # ensure ema is (re)initialised to the (warm-started) model weights
    if model.use_ema:
        model.model_ema = LitEma(model.model)

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
