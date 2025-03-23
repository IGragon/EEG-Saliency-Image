import hydra
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.trainer.lora_sd_trainer import LoraSDTrainer
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import init_wandb, set_random_seed
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from logging import getLogger


@hydra.main(version_base=None, config_path="src/configs", config_name="lora_sd")
def main(config):
    set_random_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    logger = getLogger(__name__)
    run = init_wandb(config)

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
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = config.get("weight_dtype", "fp32")

    if weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif weight_dtype == "fp16":
        weight_dtype = torch.float16

    unet_lora_config = instantiate(
        config.unet_lora_config,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.add_adapter(unet_lora_config)
    
    try:
        import xformers

        unet.enable_xformers_memory_efficient_attention()
        logger.info("Enabled memory efficient attention")
    except Exception as e:
        logger.info(f"Failed to enable xformers: {e.__class__.__name__}: {e}")

    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = instantiate(config.optimizer, params=lora_layers)
    lr_scheduler = (
        instantiate(config.lr_scheduler, optimizer=optimizer)
        if config.get("lr_scheduler")
        else None
    )
    criterion = instantiate(config.criterion)

    logger.info(f"Number of parameters: {sum(p.numel() for p in unet.parameters())}")
    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in lora_layers)}"
    )

    # load data
    train_dataloader, val_dataloader, dataset = get_dataloaders(config, generator)

    trainer = LoraSDTrainer(
        model=unet,
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
        image_class_to_name=dataset.image_classes,
    )
    trainer.train()


if __name__ == "__main__":
    main()
