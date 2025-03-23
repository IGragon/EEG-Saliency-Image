import torch

from torch.utils.data import Dataset

from pathlib import Path

# What do we even want from the dataset?
# What corresponds to the sample?
# What do we need for training? (<eeg embedding>, <image latent>)
# What do we need for some validation? (<eeg embedding>, <image class>, <perhaps image itself>)


class EEGEmbToImgDataset(Dataset):
    def __init__(
        self,
        image_latents_folder: str,
        subject_eeg_embeddings: str,
    ):
        image_latents_folder = Path(image_latents_folder)
        subject_eeg_embeddings = Path(subject_eeg_embeddings)
        self.image_latents_paths = sorted(image_latents_folder.glob("*/*.pt"))

        # eeg embeddings for one subject are [66160, 1024] that is [training_images * n_repetitions, 1024]
        self.eeg_embeddings = torch.load(subject_eeg_embeddings)

        # image latents are [training_images, 4, 64, 64]
        self.image_latents = torch.cat(
            [
                torch.load(image_latent_path)
                for image_latent_path in self.image_latents_paths
            ],
            dim=0,
        )

        self.image_classes = sorted(
            [latent_file.parent.name for latent_file in self.image_latents_paths]
        )
        self.img_index_to_class_index = [
            self.image_classes.index(latent_file.parent.name)
            for latent_file in self.image_latents_paths
        ]
        self.img_index_to_class_name = [
            latent_file.parent.name for latent_file in self.image_latents_paths
        ]

    def __len__(self):
        return len(self.eeg_embeddings)

    def _prepare_sample(self, idx):
        eeg_embedding = self.eeg_embeddings[idx].unsqueeze(0)

        # because for each image we have 4 repetitions of
        # perception of the same image
        image_index = idx // 4
        image_latent = self.image_latents[image_index]
        image_class = self.img_index_to_class_index[image_index]

        return {
            "eeg_embedding": eeg_embedding,
            "image_latent": image_latent,
            "image_class": image_class,
            "image_index": image_index,
        }
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            dict: {
                "eeg_embedding": torch.Tensor, [1, 1024]
                "image_latent": torch.Tensor, [4, 64, 64]
                "image_class": int
                "image_index": int
            }
        """
        return self._prepare_sample(idx)
