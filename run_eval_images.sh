#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eeg-salience-image
python eval_image_metrics.py
