from pathlib import Path
import torch


class BaseTrainer:
    def __init__(
        self,
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
        device,
        scaler,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimzier
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.logger = logger
        self.writer = writer
        self.configuration = configuration
        self.scaler = scaler

        self.num_epochs = self.configuration["num_epochs"]
        self.save_period = self.configuration["save_period"]
        self.grad_accumulation_steps = self.configuration.get(
            "grad_accumulation_steps",
            1,
        )
        self.clip_grad_norm = self.configuration.get("clip_grad_norm", None)

        self.device = device

        self.model.to(self.device)

        save_dir = Path(save_dir)
        self.checkpoint_dir = save_dir / self.writer.name

        self._last_epoch = 0

        if self.configuration.get("resume_from") is not None:
            self._resume_checkpoint(self.configuration.get("resume_from"))

        if self.configuration.get("from_pretrained") is not None:
            self._from_pretrained(self.configuration.get("from_pretrained"))

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on KeyboardInterrupt")
            self._save_checkpoint()
            raise e

    def _train_process(self):
        for epoch in range(self._last_epoch, self.num_epochs):
            self._last_epoch = epoch
            logs = {"epoch": epoch}

            train_result = self._train_epoch()

            logs.update(train_result)

            best, val_result = self._val_epoch()

            logs.update(val_result)
            self.writer.log(logs)

            if best or (epoch + 1) % self.save_period == 0:
                self._save_checkpoint(save_best=best)

    def _train_epoch(self):
        raise NotImplementedError()

    def _val_epoch(self):
        raise NotImplementedError()

    def _save_checkpoint(self, save_best=False):
        model_name = type(self.model).__name__

        state = {
            "model_name": model_name,
            "epoch": self._last_epoch,
            "model_state_dict": self.model.state_dict(),
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self._last_epoch = checkpoint["epoch"]

    def _from_pretrained(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.logger.info(f"Loading from pretrained {checkpoint_path}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
