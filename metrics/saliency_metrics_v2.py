from pathlib import Path
from pysaliency.metrics import CC, SIM, image_based_kl_divergence
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import numpy as np

if __name__ == "__main__":
    test_saliency_maps_folder = Path("/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/data/images/test_images_saliency_maps")
    generated_images_saliency_maps = Path("/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/evals/stellar-carrier-175-epoch-128-guidance-4_saliency_maps")
    default_image_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    all_images = torch.cat([
        default_image_transform(read_image(img_path, ImageReadMode.GRAY)) / 255
        for img_path in sorted(test_saliency_maps_folder.glob("*/*.jpg"))
    ]).numpy()
    all_brain_recons = torch.cat([
        read_image(img_path, ImageReadMode.GRAY) / 255
        for img_path in sorted(generated_images_saliency_maps.glob("*/*.jpg"))
    ]).numpy()

    CC_scores = []
    KL_scores = []
    SIM_scores = []
    for gt_smap, recon_smap in zip(all_images, all_brain_recons):
        CC_scores.append(CC(gt_smap, recon_smap))
        KL_scores.append(image_based_kl_divergence(gt_smap, recon_smap))
        SIM_scores.append(SIM(gt_smap, recon_smap))
    
    print(f"{generated_images_saliency_maps.name} | {np.mean(CC_scores):.2f} | {np.mean(KL_scores):.2f} | {np.mean(SIM_scores):.2f}")