from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image

import numpy as np
from skimage import filters
from tqdm import tqdm

import resnet
import decoder

INPUT_SIZE = (480, 640) # (480, 640) is a default from SALICON dataset
TARGET_SIZE = (512, 512)
NUM_FEAT = 5
IMAGES_PATH = Path("../../data/images/test_images")
SAVE_PATH = IMAGES_PATH.parent / f"{IMAGES_PATH.name}_saliency_maps"


def normalize(x):
    x -= x.min()
    x /= x.max()


def post_process(pred):
    pred = filters.gaussian(pred, 5)
    normalize(pred)
    pred = (pred * 255).astype(np.uint8)
    return pred

def main():
    SAVE_PATH.mkdir(parents=True)
    preprocess = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
    ])

    img_model = resnet.resnet50("./res_imagenet.pth").cuda().eval()
    pla_model = resnet.resnet50("./res_places.pth").cuda().eval()
    decoder_model = decoder.build_decoder("./res_decoder.pth", INPUT_SIZE, NUM_FEAT, NUM_FEAT).cuda().eval()

    image_paths = list(IMAGES_PATH.glob("*/*.jpg"))
    for image_path in tqdm(image_paths):
        image = preprocess(read_image(image_path)) / 127.5 - 1 # Normalize to [-1, 1]
        image = image.to("cuda")
        image = image.unsqueeze(0)
        with torch.no_grad():
            img_feat = img_model(image, decode=True)
            pla_feat = pla_model(image, decode=True)

            pred = decoder_model([img_feat, pla_feat])
        pred = pred.squeeze().detach().cpu().numpy()
        pred = post_process(pred)
        save_path = SAVE_PATH / image_path.parent.name
        save_path.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred).resize(TARGET_SIZE).save(save_path / image_path.name)


if __name__ == "__main__":
    main()