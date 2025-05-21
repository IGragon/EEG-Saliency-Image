import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from hydra.utils import instantiate


def get_dataloaders(data_config: dict, generator: torch.Generator, random_state: int):
    training_data_df = pd.read_csv(data_config["data_df"])
    eeg_repetitions = data_config["eeg_repetitions"]
    subject_eeg_embeddings = torch.load(data_config["subject_eeg_embeddings"]).reshape(
        -1, eeg_repetitions, 1024
    )

    train_df, val_df, train_subject_eeg_emb, val_subject_eeg_emb = train_test_split(
        training_data_df,
        subject_eeg_embeddings,
        test_size=0.01,
        random_state=random_state,
    )

    train_dataset = instantiate(
        data_config.dataset,
        image_class_names=train_df["img_cls_name"].to_list(),
        image_names=train_df["img_name"].to_list(),
        subject_eeg_embeddings=train_subject_eeg_emb,
        eeg_repetitions=eeg_repetitions,
        average_repetitions=data_config["average_repetitions"],
    )
    val_dataset = instantiate(
        data_config.dataset,
        image_class_names=val_df["img_cls_name"].to_list(),
        image_names=val_df["img_name"].to_list(),
        subject_eeg_embeddings=val_subject_eeg_emb,
        eeg_repetitions=eeg_repetitions,
        is_train_split=False,
        average_repetitions=data_config["average_repetitions"],
    )

    batch_size = data_config["batch_size"]
    val_batch_size = data_config.get("val_batch_size", batch_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        generator=generator,
    )

    return train_dataloader, val_dataloader
