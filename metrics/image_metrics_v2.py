# metric calculation is taken from https://github.com/ncclab-sustech/EEG_Image_decode/blob/main/Generation/Reconstruction_Metrics_ATM.ipynb

import clip
from pathlib import Path
import scipy as sp
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
# from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.io import read_image

device = "cuda"

@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1
    

def pix_corr(all_images, all_brain_recons):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()

    print(all_images_flattened.shape)
    print(all_brain_recons_flattened.shape)

    corrsum = 0
    for i in tqdm(range(len(all_brain_recons_flattened))):
        corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
    corrmean = corrsum / len(all_brain_recons_flattened)
    return corrmean

def ssim_score(all_images, all_brain_recons):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
    print("converted, now calculating ssim...")

    ssim_score=[]
    for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
        ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    ssim_metric = np.mean(ssim_score)
    return ssim_metric


def alexnet_scores(all_images, all_brain_recons):
    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    layer = 'early, AlexNet(2)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.4')
    alexnet2 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet2:.4f}")

    layer = 'mid, AlexNet(5)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.11')
    alexnet5 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet5:.4f}")
    return alexnet2, alexnet5

def inception_v3_score(all_images, all_brain_recons):
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            inception_model, preprocess, 'avgpool')
            
    inception = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {inception:.4f}")
    return inception


def clip_score(all_images, all_brain_recons):
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct = two_way_identification(all_brain_recons, all_images,
                                            clip_model.encode_image, preprocess, None) # final layer
    clip_ = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {clip_:.4f}")
    return clip_

def swav_score(all_images, all_brain_recons):
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(all_images).to(device))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess(all_brain_recons).to(device))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",swav)

    return swav


if __name__ == "__main__":
    test_images_folder = Path("/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/data/images/test_images")
    generated_images_folder = Path("/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/evals/stellar-carrier-175-epoch-128-guidance-4")

    default_image_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    all_images = torch.cat([
        default_image_transform(read_image(img_path, "RGB")).unsqueeze(0) / 255
        for img_path in sorted(test_images_folder.glob("*/*.jpg"))
    ])
    all_brain_recons = torch.cat([
        read_image(img_path, "RGB").unsqueeze(0) / 255
        for img_path in sorted(generated_images_folder.glob("*/*.jpg"))
    ])
    # Model |        pixcorr (higher) | SSIM (higher)|  alexnet2 (higher) |  alexnet5 (higher) |  inception (higher) |CLIP (higher)|  swav (lower)
    print(f"{generated_images_folder.name} | {pix_corr(all_images, all_brain_recons):.3f} | {ssim_score(all_images, all_brain_recons):.3f} | {'|'.join([str(round(x, 3)) for x in alexnet_scores(all_images, all_brain_recons)])} | {inception_v3_score(all_images, all_brain_recons):.3f} | {clip_score(all_images, all_brain_recons):.3f} | {swav_score(all_images, all_brain_recons):.3f}")