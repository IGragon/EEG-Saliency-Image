from pathlib import Path
import hydra
import torch
from hydra.utils import instantiate
from diffusers import ControlNetModel

from src.trainer.controlnet_trainer import ControlnetTrainer
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import init_wandb, set_random_seed
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from logging import getLogger


@hydra.main(version_base=None, config_path="src/configs", config_name="train_controlnet")
def main(config):
    set_random_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    logger = getLogger(__name__)

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    logger.info(f"Using device: {device}")

    # load modules
    unet = UNet2DConditionModel.from_pretrained(
        config.model_path,
        subfolder="unet",
    )
    vae = AutoencoderKL.from_pretrained(
        config.vae_path,
        subfolder="vae",
    )
    noise_scheduler = PNDMScheduler.from_pretrained(
        config.noise_scheduler_path,
        subfolder="scheduler",
    )

    weight_dtype = config.get("weight_dtype", "fp32")

    if weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif weight_dtype == "fp16":
        weight_dtype = torch.float16

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    resume_path = config.get("resume_from")
    if resume_path is not None:
        resume_path = Path(resume_path)
        epoch = resume_path.stem.split("-")[-1]
        controlnet_path = resume_path.parent / f"controlnet-{epoch}"
        controlnet = ControlNetModel.from_pretrained(controlnet_path)
        logger.info(f"Resuming training from {controlnet_path}")
    else:
        controlnet = ControlNetModel.from_unet(unet)
    
    controlnet.to(device, dtype=weight_dtype)
    controlnet.requires_grad_(True)

    optimizer = instantiate(config.optimizer, params=controlnet.parameters())
    scaler = torch.amp.GradScaler(device, enabled=config.use_amp)
    lr_scheduler = (
        instantiate(config.lr_scheduler, optimizer=optimizer)
        if config.get("lr_scheduler")
        else None
    )
    criterion = instantiate(config.criterion)

    # load data
    train_dataloader, val_dataloader = get_dataloaders(
        config.data_config, generator, config.seed
    )

    run = init_wandb(config)
    trainer = ControlnetTrainer(
        controlnet=controlnet,
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimzier=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        logger=logger,
        writer=run,
        save_dir=config.save_dir,
        configuration=config,
        device=device,
        scaler=scaler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
