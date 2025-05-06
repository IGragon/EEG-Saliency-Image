from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image

class ControlnetInferencer:
    def __init__(
        self,
        controlnet,
        unet,
        vae,
        noise_scheduler,
        val_dataloader,
        logger,
        save_dir,
        configuration,
        image_paths,
        device,
        scaler,
        guidance,
    ):
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.configuration = configuration

        self.use_amp = self.configuration.get("use_amp", False)

        self.image_paths = image_paths

        self.model = controlnet # attention
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.configuration = configuration
        self.scaler = scaler

        self.device = device
        self.model.to(self.device)

        self.num_inference_steps = self.configuration["num_inference_steps"]

        save_dir = Path(save_dir)
        self.save_dir = save_dir
        self.guidance = guidance
    
    def eval(self):
        self.model.eval()
        self.unet.eval()
        self.vae.eval()
        image_index = 0
        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
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
                    controlnet_image = batch["saliency_map"].to(device=self.device, dtype=self.model.dtype)
                    controlnet_image = torch.cat([controlnet_image] * 2)

                    zero_embeddings = torch.zeros_like(eeg_embedding)

                    all_embeddings = torch.concatenate([eeg_embedding, zero_embeddings], dim=0)

                    latents = torch.randn((len(eeg_embedding), 4, 64, 64)).to(
                        self.device,
                        dtype=self.model.dtype,
                    )
                    latents = latents * self.noise_scheduler.init_noise_sigma
                    for t in self.noise_scheduler.timesteps:
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = self.noise_scheduler.scale_model_input(
                            latent_model_input, t
                        )


                        down_block_res_samples, mid_block_res_sample = self.model(
                            latent_model_input,
                            t,
                            encoder_hidden_states=all_embeddings,
                            controlnet_cond=controlnet_image,
                            return_dict=False,
                        )

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=all_embeddings,
                            down_block_additional_residuals=[
                                sample.to(dtype=self.model.dtype) for sample in down_block_res_samples
                            ],
                            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.model.dtype),
                        ).sample

                        noise_pred_emb, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance * (noise_pred_emb - noise_pred_uncond)

                        latents = self.noise_scheduler.step(
                            noise_pred, t, latents
                        ).prev_sample

                    generated_images = self.decode_from_latent_space(latents.detach())

                generated_images = (
                    generated_images.permute(0, 2, 3, 1).cpu().numpy().clip(-1, 1) * 0.5 + 0.5
                )
                
                batch_image_indexes = []
                for _ in range(generated_images.shape[0]):
                    batch_image_indexes.append(image_index)
                    image_index += 1
                
                true_image_paths = [
                    self.image_paths[idx] for idx in batch_image_indexes
                ]

                for image, true_image_path in zip(generated_images, true_image_paths):
                    image = Image.fromarray((image * 255).astype("uint8"))
                    generated_image_path = (
                        self.save_dir
                        / true_image_path.parent.name
                        / true_image_path.name
                    )
                    generated_image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(generated_image_path)
    
    @torch.no_grad()
    def decode_from_latent_space(self, latents):
        return self.vae.decode(latents / self.vae.config.scaling_factor).sample
