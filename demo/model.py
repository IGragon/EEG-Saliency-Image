import torch
from torchvision import transforms as tt
from diffusers import ControlNetModel

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from logging import getLogger
from PIL import Image

logger = getLogger(__name__)


class SaliencyEEGGuidedDiffusion:
    def __init__(self):
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available")

        self.device = "cuda"
        # load modules
        self.unet = UNet2DConditionModel.from_pretrained(
            "/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/base_models/stellar-carrier-175-epoch-128-merged",
            subfolder="unet",
        )
        self.vae = AutoencoderKL.from_pretrained(
            "/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/base_models/stable-diffusion-2-1-base",
            subfolder="vae",
        )
        self.noise_scheduler = PNDMScheduler.from_pretrained(
            "/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/base_models/stable-diffusion-2-1-base",
            subfolder="scheduler",
        )

        self.unet.to(self.device, dtype=torch.float32)
        self.vae.to(self.device, dtype=torch.float32)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.controlnet = ControlNetModel.from_pretrained(
            "/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/runs/sith-midichlorian-176/controlnet-31"
        )
        self.controlnet.to(self.device, dtype=torch.float32)
        self.controlnet.requires_grad_(False)

        self.num_inference_steps = 100
        self.guidance = 2

        self.transforms = tt.Compose(
            [
                tt.Resize(512),
                tt.ToTensor(),
                tt.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            ]
        )

    @torch.no_grad()
    def process(self, eeg_embedding, saliency_map) -> list[Image.Image]:
        saliency_map = torch.concat(
            [self.transforms(s).unsqueeze(0) for s in saliency_map], dim=0
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        eeg_embedding = eeg_embedding.to(self.device)
        saliency_map = saliency_map.to(self.device)

        controlnet_image = torch.cat([saliency_map] * 2)
        zero_embeddings = torch.zeros_like(eeg_embedding)
        all_embeddings = torch.cat([eeg_embedding, zero_embeddings], dim=0)

        latents = torch.randn((1, 4, 64, 64)).to(self.device)
        latents = latents * self.noise_scheduler.init_noise_sigma

        for t in self.noise_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            down_block_res_samples, mid_block_res_sample = self.controlnet(
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
                    sample.to(dtype=self.controlnet.dtype)
                    for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.controlnet.dtype
                ),
            ).sample

            noise_pred_emb, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance * (
                noise_pred_emb - noise_pred_uncond
            )

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        generated_images = self.decode_from_latent_space(latents.detach())
        generated_images = (
            generated_images.permute(0, 2, 3, 1).cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        )
        return [
            Image.fromarray((image * 255).astype("uint8")) for image in generated_images
        ]

    @torch.no_grad()
    def decode_from_latent_space(self, latents):
        return self.vae.decode(latents / self.vae.config.scaling_factor).sample
