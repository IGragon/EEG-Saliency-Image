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