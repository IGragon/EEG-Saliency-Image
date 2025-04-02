# metrics/image_metrics.py
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.models import alexnet, inception_v3
import timm
# from skimage.metrics import structural_similarity as ssim
import numpy as np

def pixel_correlation(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

# def compute_ssim(img1, img2):
#     return ssim(img1, img2, channel_axis=-1, data_range=img2.max() - img2.min())

def extract_features(model, layer, img):
    activations = {}
    def hook(module, input, output):
        activations['feat'] = output
    handle = layer.register_forward_hook(hook)
    model(img)
    handle.remove()
    return activations['feat']

def feature_similarity(model, layer, img1, img2):
    f1 = extract_features(model, layer, img1).flatten(1)
    f2 = extract_features(model, layer, img2).flatten(1)
    return F.cosine_similarity(f1, f2).item()

def alexnet_features(img1, img2, layer_idx=2):
    model = alexnet(pretrained=True).eval()
    layer = model.features[layer_idx]
    return feature_similarity(model, layer, img1, img2)

def inception_features(img1, img2):
    model = inception_v3(pretrained=True, transform_input=False, aux_logits=True).eval()
    layer = model.Mixed_7c
    return feature_similarity(model, layer, img1, img2)

def swav_features(img1, img2):
    model = timm.create_model("resnet50.a1_in1k", pretrained=True).eval()
    return feature_similarity(model, model.global_pool, img1, img2)
