from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
import os


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_wandb(config: DictConfig):
    import wandb

    run = wandb.init(
        entity="ui-eeg-saliency-thesis",
        project="eeg-saliency-image",
        config=OmegaConf.to_container(config),
    )
    return run
