# eval_image_metrics.py
import os
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from metrics.image_metrics import *
from src.utils.image_io import load_image

@hydra.main(config_path="src/configs", config_name="eval", version_base="1.3")
def eval_images(cfg: DictConfig):
    gt_dir, gen_dir = cfg.eval.image_dir_gt, cfg.eval.image_dir_gen
    extensions = cfg.eval.extensions
    output_csv = cfg.eval.output_image_csv or "metrics_image.csv"

    metric_names = ["pixcorr", 
                    # "ssim", 
                    "alexnet2", "alexnet5", "inception", "swav"]
    all_results = []

    files = sorted(f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in extensions)

    for fname in tqdm(files):
        try:
            img1 = load_image(os.path.join(gt_dir, fname))
            img2 = load_image(os.path.join(gen_dir, fname))
            row = {
                "filename": fname,
                "pixcorr": pixel_correlation(img1.numpy(), img2.numpy()),
                # "ssim": compute_ssim(img1.squeeze().permute(1, 2, 0).numpy(), img2.squeeze().permute(1, 2, 0).numpy()),
                "alexnet2": alexnet_features(img1, img2, 2),
                "alexnet5": alexnet_features(img1, img2, 5),
                "inception": inception_features(img1, img2),
                "swav": swav_features(img1, img2)
            }
            all_results.append(row)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    df = pd.DataFrame(all_results)
    stats = df[metric_names].agg(['mean', 'std', 'median', 'min', 'max'])
    df = pd.concat([df, stats.assign(filename=stats.index)], ignore_index=True)
    df.to_csv(output_csv, index=False)

    print(f"\n🎨 Saved image metrics to: {output_csv}")
    print("\n🔢 Summary Stats:")
    print(stats.round(4))

if __name__ == "__main__":
    eval_images()
