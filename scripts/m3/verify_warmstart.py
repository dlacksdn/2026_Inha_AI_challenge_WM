import torch
from omegaconf import OmegaConf
from lvdm.utils.utils import instantiate_from_config
cfg = OmegaConf.load("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM/scripts/m3/configs/train/inha_action_diffusion_11M_m3.yaml")
unet_cfg = cfg.model.params.unet_config
print("UNet 빌드(CPU)... action_dropout_prob=", unet_cfg.params.action_dropout_prob, flush=True)
unet = instantiate_from_config(unet_cfg)   # UNetModel on CPU
unet_keys = set(unet.state_dict().keys())
print("UNet 파라미터 키 수:", len(unet_keys), flush=True)
# baseline_diffusion.ckpt 의 model.diffusion_model.* 키(= full model에서 self.model.diffusion_model.*)
sd = torch.load("/home/rils/dlacksdn/2026_Inha_AI_challenge_WM/open/baseline/checkpoints/baseline_diffusion.ckpt", map_location="cpu")["state_dict"]
ck_unet = set(k[len("model.diffusion_model."):] for k in sd if k.startswith("model.diffusion_model."))
print("ckpt의 UNet(model.diffusion_model.*) 키 수:", len(ck_unet), flush=True)
missing = unet_keys - ck_unet   # 모델엔 있는데 ckpt에 없는 = warm-start로 못 채우는 키
extra   = ck_unet - unet_keys   # ckpt엔 있는데 모델에 없는
print("unet_missing(모델 O, ckpt X):", len(missing), flush=True)
if missing: print("  예:", list(missing)[:6], flush=True)
print("unet_extra(ckpt O, 모델 X):", len(extra), flush=True)
if extra: print("  예:", list(extra)[:6], flush=True)
print("=> warm-start unet_missing==0 ?", len(missing)==0, flush=True)
