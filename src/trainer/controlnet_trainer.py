from random import random
import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from torch.nn.utils import clip_grad_norm_


class ControlnetTrainer(BaseTrainer):
    def __init__(
        self,
        controlnet,
        unet,
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
            controlnet,
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
        # self.model -- это controlnet
        self.vae = vae
        self.unet = unet
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

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=self.model.dtype)

                down_block_res_samples, mid_block_res_sample = self.model(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=eeg_embedding,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=eeg_embedding,
                    down_block_additional_residuals=[
                        sample.to(dtype=self.model.dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.model.dtype),
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
        return False, {}
    
    def _save_checkpoint(self, save_best=False):
        model_path = self.checkpoint_dir / f"controlnet-{self._last_epoch}"
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