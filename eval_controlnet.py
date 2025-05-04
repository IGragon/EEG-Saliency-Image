from pathlib import Path
import hydra
import pandas as pd
import torch
from torch.utils.data import DataLoader
from diffusers import ControlNetModel

from src.trainer.controlnet_inferencer import ControlnetInferencer
from src.utils.init_utils import set_random_seed
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from logging import getLogger
from src.datasets.EEGSaliencyDataset import EEGSaliencyDataset


@hydra.main(version_base=None, config_path="src/configs", config_name="eval_controlnet")
def main(config):
    set_random_seed(config.seed)
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

    resume_path = Path(config["resume_from"])
    controlnet = ControlNetModel.from_pretrained(resume_path)
    logger.info(f"Evaluating controlnet from {resume_path}")
    
    controlnet.to(device, dtype=weight_dtype)
    controlnet.requires_grad_(False)

    scaler = torch.amp.GradScaler(device, enabled=config.use_amp)

    # load data
    # train_dataloader, val_dataloader = get_dataloaders(
    #     config.data_config, generator, config.seed
    # )

    test_data_df = pd.read_csv(config["data_df"])
    eeg_repetitions = 1
    subject_eeg_embeddings = torch.load(config["subject_eeg_embeddings"]).reshape(
        -1, eeg_repetitions, 1024
    )
    dataset = EEGSaliencyDataset(
        image_class_names=test_data_df["img_cls_name"].to_list(),
        image_names=test_data_df["img_name"].to_list(),
        subject_eeg_embeddings=subject_eeg_embeddings,
        saliency_maps_foler_path=config.saliency_maps_foler_path,
        images_folder_path=config.images_folder_path,
        eeg_repetitions=eeg_repetitions,
        is_train_split=False,
        average_repetitions=False,
    )
    batch_size = config["batch_size"]
    val_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    inferencer = ControlnetInferencer(
        controlnet=controlnet,
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        val_dataloader=val_dataloader,
        logger=logger,
        save_dir=config.save_dir,
        configuration=config,
        image_paths=sorted(Path(config.images_folder_path).glob("*/*.jpg")),
        device=device,
        scaler=scaler,
    )
    inferencer.eval()


if __name__ == "__main__":
    main()
