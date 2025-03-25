import torch
import torchvision
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from .base_trainer import BaseTrainer
from PIL import Image
import wandb


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
        image_class_to_name,
        device,
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
        )
        self.vae = vae
        self.noise_scheduler = noise_scheduler

        num_inference_steps = self.configuration["num_inference_steps"]
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        self.image_class_to_name = image_class_to_name

    def _train_epoch(self):
        self.model.train()
        train_losses = []
        for batch_idx, batch in tqdm(
            enumerate(self.train_dataloader),
            desc="Training",
            total=len(self.train_dataloader),
        ):
            eeg_embedding = batch["eeg_embedding"].to(
                self.device, dtype=self.model.dtype
            )
            image_latent = batch["image_latent"].to(self.device, dtype=self.model.dtype)
            noise = torch.randn(image_latent.shape).to(
                self.device, dtype=self.model.dtype
            )
            bs = image_latent.shape[0]
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=self.device,
            )
            latent_model_input = self.noise_scheduler.add_noise(
                image_latent,
                noise,
                timesteps,
            ).to(self.device)

            noise_pred = self.model(
                latent_model_input,
                timesteps,
                encoder_hidden_states=eeg_embedding,
            ).sample

            loss = self.criterion(noise_pred, noise)
            loss.backward()

            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                if self.clip_grad_norm is not None:
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_losses.append(loss.item())

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return {"train_loss": sum(train_losses) / len(train_losses)}

    def _val_epoch(self):
        self.model.eval()
        result = {}
        with torch.no_grad():
            for batch in self.val_dataloader:
                eeg_embedding = batch["eeg_embedding"].to(
                    self.device,
                    dtype=self.model.dtype,
                )

                latents = torch.randn(batch["image_latent"].shape).to(
                    self.device,
                    dtype=self.model.dtype,
                )
                latents = latents * self.noise_scheduler.init_noise_sigma

                for t in tqdm(self.noise_scheduler.timesteps, desc="Generating"):
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

                images = self.decode_from_latent_space(latents)
                grid = torchvision.utils.make_grid(
                    images, nrow=batch["image_latent"].shape[0]
                )
                np_grid = grid.permute(1, 2, 0).cpu().numpy().clip(-1, 1) * 0.5 + 0.5
                image_class_names = "img_cls_names: " + ";".join(
                    [
                        self.image_class_to_name[idx]
                        for idx in batch["image_class"].tolist()
                    ]
                )
                image_indexes = "img_ids: " + ";".join(
                    [str(idx) for idx in batch["image_index"].tolist()]
                )
                image = Image.fromarray((np_grid * 255).astype("uint8"))
                wandb_image = wandb.Image(
                    image, caption=f"{image_class_names}\n{image_indexes}"
                )
                result["val_image"] = wandb_image

        return False, result

    def decode_from_latent_space(self, latents):
        return self.vae.decode(latents / self.vae.config.scaling_factor).sample

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
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self._last_epoch = checkpoint["epoch"]
        self.logger.info(f"Resuming training from {checkpoint_path}")

    def _from_pretrained(self, checkpoint_path):
        raise NotImplementedError(
            f"Not valid use for this type of trainer: {type(self).__name__}"
        )
