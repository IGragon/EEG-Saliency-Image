from PIL import Image
import torch

from torch.utils.data import Dataset
import torchvision.transforms as tt

from pathlib import Path


IMAGE_TRANSFORMS = tt.Compose(
    [
        tt.Resize(512),
        tt.ToTensor(),
        tt.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    ]
)


class EEGSaliencyDataset(Dataset):
    def __init__(
        self,
        image_class_names: list[str],
        image_names: list[str],
        images_latents_folder_path: str,
        saliency_maps_foler_path: str,
        subject_eeg_embeddings: str,
        eeg_repetitions: int,
        images_folder_path: str = None,
        is_train_split: bool = True,
        average_repetitions: bool = False,
    ):
        self.image_class_names = image_class_names
        self.image_names = image_names
        self.images_latents_folder_path = Path(images_latents_folder_path)
        self.saliency_maps_foler_path = Path(saliency_maps_foler_path)
        self.is_train_split = is_train_split
        self.average_repetitions = average_repetitions
        self.eeg_repetitions = eeg_repetitions if not self.average_repetitions else 1

        if self.is_train_split:
            self.image_latents = self._get_latents()
        else:
            self.images_folder_path = Path(images_folder_path)

        # eeg_embeddings have the shape of [n_images, eeg_repetitions, 1024]
        self.eeg_embeddings = subject_eeg_embeddings
        if self.average_repetitions:
            self.eeg_embeddings = self.eeg_embeddings.mean(dim=1).unsqueeze(1)

    def __getitem__(self, idx: int):
        image_idx = idx // self.eeg_repetitions
        repetition_idx = idx % self.eeg_repetitions
        image_latent = -1
        saliency_map = -1
        image = -1
        if self.is_train_split:
            image_latent = self.image_latents[image_idx]
            saliency_map = self.get_smap(image_idx)
        else:
            image = self.get_image(image_idx)

        return {
            "eeg_embedding": self.eeg_embeddings[image_idx][repetition_idx].unsqueeze(
                0
            ),
            "image_latent": image_latent,
            "saliency_map": saliency_map,
            "image": image,
        }
    
    def get_smap(self, image_idx):
        return (
            IMAGE_TRANSFORMS(
                Image.open(
                    self.saliency_maps_foler_path
                    / self.image_class_names[image_idx]
                    / self.image_names[image_idx]
                )
            )
        )


    def get_image(self, image_idx):
        return (
            IMAGE_TRANSFORMS(
                Image.open(
                    self.images_folder_path
                    / self.image_class_names[image_idx]
                    / self.image_names[image_idx]
                )
            ) * 2 - 1
        )

    def __len__(self):
        return len(self.eeg_embeddings) * self.eeg_repetitions

    def _get_latents(self):
        return torch.cat(
            [
                torch.load(
                    self.images_latents_folder_path
                    / img_cls_name
                    / img_name.replace(".jpg", ".pt")
                )
                for img_cls_name, img_name in zip(
                    self.image_class_names, self.image_names
                )
            ]
        )
