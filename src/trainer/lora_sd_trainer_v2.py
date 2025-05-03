import numpy as np
import torch
import torchvision
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from .base_trainer import BaseTrainer
from PIL import Image
import wandb
from random import random, randint
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray


def pixel_correlation(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]


def compute_ssim(img1: np.ndarray, img2: np.ndarray):
    img1 = img1.transpose(1, 2, 0).clip(-1, 1) * 0.5 + 0.5
    img2 = img2.transpose(1, 2, 0).clip(-1, 1) * 0.5 + 0.5
    return ssim(
        rgb2gray(img1),
        rgb2gray(img2),
        multichannel=True,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=1.0,
    )


class LoraSDTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        vae,
        noise_scheduler,
        train_dataloader,
        val_dataloader,
        optimzier,
        lr_scheduler,
        criterion,
        logger,
        writer,
        save_dir,
        configuration,
        device,
        scaler,
    ):
        super().__init__(
            model,
            train_dataloader,
            val_dataloader,
            optimzier,
            lr_scheduler,
            criterion,
            logger,
            writer,
            save_dir,
            configuration,
            device=device,
            scaler=scaler,
        )
        self.vae = vae
        self.noise_scheduler = noise_scheduler

        self.num_inference_steps = self.configuration["num_inference_steps"]

        self.null_vector_probability = self.configuration.get(
            "null_vector_probability",
            0.0,
        )
        self.use_amp = self.configuration.get("use_amp", False)

    def _train_epoch(self):
        self.model.train()
        train_losses = []
        for batch_idx, batch in tqdm(
            enumerate(self.train_dataloader),
            desc="Training",
            total=len(self.train_dataloader),
        ):
            with torch.autocast(
                device_type=self.device,
                dtype=self.model.dtype,
                enabled=self.use_amp,
            ):
                eeg_embedding = batch["eeg_embedding"].to(
                    dtype=self.model.dtype, device=self.device
                )
                for i in range(len(eeg_embedding)):
                    if random() < self.null_vector_probability:
                        eeg_embedding[i] = torch.zeros_like(
                            eeg_embedding[i],
                        ).to(dtype=self.model.dtype, device=self.device)

                image_latent = batch["image_latent"].to(
                    dtype=self.model.dtype,
                    device=self.device,
                )
                noise = torch.randn(
                    image_latent.shape,
                ).to(dtype=self.model.dtype, device=self.device)
                bs = image_latent.shape[0]
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bs,),
                ).to(device=self.device)

                latent_model_input = self.noise_scheduler.add_noise(
                    image_latent,
                    noise,
                    timesteps,
                )
                noise_pred = self.model(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=eeg_embedding,
                ).sample

                loss = self.criterion(noise_pred, noise)
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            train_losses.append(loss.item())

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return {"train_loss": sum(train_losses) / len(train_losses)}

    def _val_epoch(self):
        self.model.eval()
        self.vae.eval()
        result = {}
        pix_corr = []
        ssim_scores = []
        save_img_batch = randint(0, len(self.val_dataloader) - 1)
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.val_dataloader),
                desc="Validation",
                total=len(self.val_dataloader),
            ):
                # пусть будет, там оно что-то запоминает и потом когда последний батч меньшего размера, то падает с ошибкой
                # а метод set_timesteps сбрасывает накопленные настройки
                self.noise_scheduler.set_timesteps(
                    self.num_inference_steps, device=self.device
                )

                with torch.autocast(
                    device_type=self.device,
                    dtype=self.model.dtype,
                    enabled=self.use_amp,
                ):
                    eeg_embedding = batch["eeg_embedding"].to(
                        self.device,
                        dtype=self.model.dtype,
                    )

                    latents = torch.randn((len(eeg_embedding), 4, 64, 64)).to(
                        self.device,
                        dtype=self.model.dtype,
                    )
                    latents = latents * self.noise_scheduler.init_noise_sigma
                    for t in self.noise_scheduler.timesteps:
                        latent_model_input = self.noise_scheduler.scale_model_input(
                            latents, t
                        )
                        noise_pred = self.model(
                            latent_model_input,
                            t,
                            encoder_hidden_states=eeg_embedding,
                        ).sample

                        latents = self.noise_scheduler.step(
                            noise_pred, t, latents
                        ).prev_sample

                    images = batch["image"]
                    generated_images = self.decode_from_latent_space(latents.detach())
                    ssim_scores.extend(
                        [
                            compute_ssim(gen, real)
                            for gen, real in zip(
                                generated_images.cpu().numpy(), images.cpu().numpy()
                            )
                        ]
                    )
                    pix_corr.extend(
                        [
                            pixel_correlation(gen, real)
                            for gen, real in zip(
                                generated_images.cpu().numpy(), images.cpu().numpy()
                            )
                        ]
                    )

                if batch_idx == save_img_batch:
                    result.update(self.get_images_for_logging(generated_images, images))
        result["pix_corr"] = np.mean(pix_corr)
        result["ssim"] = np.mean(ssim_scores)
        return False, result

    @torch.no_grad()
    def decode_from_latent_space(self, latents):
        return self.vae.decode(latents / self.vae.config.scaling_factor).sample

    @staticmethod
    def get_image_grid(torch_images):
        grid = torchvision.utils.make_grid(
            torch_images,
            nrow=4,
        )
        np_grid = grid.permute(1, 2, 0).cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        image = Image.fromarray((np_grid * 255).astype("uint8"))
        return image

    def get_images_for_logging(self, generated_images, images):
        return {
            "generated_images": wandb.Image(
                self.get_image_grid(generated_images),
            ),
            "seen_images": wandb.Image(
                self.get_image_grid(images),
            ),
        }

    def _save_checkpoint(self, save_best=False):
        model_path = self.checkpoint_dir / f"unet-{self._last_epoch}"
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.checkpoint_dir / model_path)

        state = {
            "epoch": self._last_epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
            if self.lr_scheduler is not None
            else None,
            "configuration": self.configuration,
            "scaler": self.scaler.state_dict(),
        }
        filename = self.checkpoint_dir / (
            f"checkpoint-epoch-{self._last_epoch}.pth"
            if not save_best
            else "checkpoint-best.pth"
        )
        torch.save(state, filename)

        self.logger.info(f"Saved checkpoint to {filename}")

    def _resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self._last_epoch = checkpoint["epoch"]
        self.logger.info(f"Resuming training from {checkpoint_path}")

    def _from_pretrained(self, checkpoint_path):
        raise NotImplementedError(
            f"Not valid use for this type of trainer: {type(self).__name__}"
        )
