# eval_image_metrics.py
import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from metrics.image_metrics import *
from src.utils.image_io import load_image  # see utility function below

@hydra.main(config_path="src/configs", config_name="eval", version_base="1.3")
def eval_images(cfg: DictConfig):
    gt_dir = cfg.eval.image_dir_gt
    gen_dir = cfg.eval.image_dir_gen
    extensions = cfg.eval.extensions

    results = {k: [] for k in ["pixcorr", "alexnet2", "alexnet5", "inception", "swav"]}
    files = [f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in extensions]

    for fname in tqdm(files):
        try:
            img1 = load_image(os.path.join(gt_dir, fname))
            img2 = load_image(os.path.join(gen_dir, fname))

            results["pixcorr"].append(pixel_correlation(img1.numpy(), img2.numpy()))
            # results["ssim"].append(compute_ssim(img1.squeeze().permute(1, 2, 0).numpy(), img2.squeeze().permute(1, 2, 0).numpy()))
            results["alexnet2"].append(alexnet_features(img1, img2, 2))
            results["alexnet5"].append(alexnet_features(img1, img2, 5))
            results["inception"].append(inception_features(img1, img2))
            results["swav"].append(swav_features(img1, img2))
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    print("\n🎨 Image Evaluation Results:")
    print(results["inception"])
    print(results["swav"])
    for k, v in results.items():
        print(f"{k.upper():<10}: {sum(v)/len(v):.4f}")

if __name__ == "__main__":
    eval_images()
