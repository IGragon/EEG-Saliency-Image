import torch
from .base_trainer import BaseTrainer

from peft import LoraConfig


class ConditionalDiffusionLoraTrainer(BaseTrainer):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        optimzier,
        lr_scheduler,
        criterion,
        logger,
        writer,
        save_dir,
        configuration,
        device="auto",
        **model_kwargs,
    ):
        self.unet = model_kwargs["unet"]
        self.noise_scheduler = model_kwargs["noise_scheduler"]

        self.unet.requires_grad_(False)

        weight_dtype = configuration.get("weight_dtype", "fp32")

        if weight_dtype == "fp32":
            weight_dtype = torch.float32
        elif weight_dtype == "fp16":
            weight_dtype = torch.float16
        
        lora_config = configuration["lora"]
        unet_lora_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["r"],
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        super().__init__(
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
            **model_kwargs,
        )

        self.unet.to(self.device, dtype=weight_dtype)
        self.unet.add_adapter(unet_lora_config)

        try:
            import xformers
            self.unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            self.logger.info(f"Failed to enable xformers: {e.__class__.__name__}: {e}")

        lora_layers = filter(lambda p: p.requires_grad, self.unet.parameters())

        optimzier = self.optimizer.__class__(
            lora_layers,)


