# EEG-Driven Image Reconstruction with Saliency-Guided Diffusion Models

[![ACM MM 2025](https://img.shields.io/badge/ACM%20MM%202025-10.1145/3746027.3754476-blue?logo=acm)](https://dl.acm.org/doi/abs/10.1145/3746027.3754476)

[![arXiv](https://img.shields.io/badge/arXiv-2510.26391-b31b1b.svg)](https://arxiv.org/abs/2510.26391) 



## Setup conda environment

Follow miniconda installation tutorial: https://www.anaconda.com/docs/getting-started/miniconda/install

```
conda env create -f environment.yml
```

## Downloading data for this project
### Downloading images

THINGS-EEG images can be accessed through the following link: https://osf.io/3jk45/

Next, download files from https://osf.io/y63gw/files/osfstorage

Then extract archives to make following structure:
```
data/
  |-images/
        |
        |--test_images/
        |        |--00001_aircraft_carrier/...
        |        |--00002_antelope/...
        |        |--e.t.c
        |
        |--training_images/
        |        |--00001_aardvark/...
        |        |--00002_abacus/...
        |        |--e.t.c
```

### Downloading EEG Embeddings

```bash
git clone --config lfs.fetchinclude='emb_eeg/*' https://huggingface.co/datasets/LidongYang/EEG_Image_decode
```

Then place EEG embeddings as follows:
```
data/
  |-emb_eeg/
        |--ATM_S_eeg_features_sub-01_test.pt
        |--ATM_S_eeg_features_sub-01_train.pt
        |--e.t.c
```

## Making additional data

To speed up training make latents with: [data_utils/make_vae_latents.ipynb](./data_utils/make_vae_latents.ipynb)
