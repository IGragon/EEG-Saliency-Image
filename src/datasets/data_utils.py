from hydra.utils import instantiate
from torch.utils.data import random_split, DataLoader

def get_dataloaders(config, generator):
    dataset = instantiate(config.dataset)

    val_batch_size = max(12, config.batch_size)
    train_dataset, val_dataset = random_split(
        dataset,
        [len(dataset) - val_batch_size, val_batch_size],
        generator=generator,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_dataloader, val_dataloader, dataset
