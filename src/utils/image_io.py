# utils/image_io.py
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# For color images (for model feature evaluation)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return image_transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

# For grayscale saliency maps (for saliency metrics)
def load_saliency(path):
    img = Image.open(path).convert("L")  # Convert to grayscale
    return np.array(img, dtype=np.float32) / 255.0
