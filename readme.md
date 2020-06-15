# FastDVDnet-Blind

- Modify the [FastDVDnet](https://github.com/m-tassano/fastdvdnet) to a blind denoiser. 

- Provide the scripts to transform mp4/sequences easily.

- Provide a trained blind denoiser. 

- Update the DALI api to 0.22 version from 0.10 version.

## FastDVDnet

A state-of-the-art, simple and fast network for Deep Video Denoising which uses no motion compensation. FastDVDnet is orders of magnitude faster than other state-of-the-art methods.

## [Architecture](https://github.com/m-tassano/fastdvdnet)

## Code User Guide

### Dependencies

The code runs on Python +3.6. You can create a conda environment with all the dependecies by running (Thanks to Antoine Monod for the .yml file)
```
conda env create -f requirements.yml -n <env_name>
```

Note: this project needs the [NVIDIA DALI](https://github.com/NVIDIA/DALI) package for training. The tested version of DALI is 0.22. If you prefer to install it yourself (supposing you have CUDA 10.0), you need to run
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```

### Directory Tree

```
# in the 'data' directory
.
├── test_A
│   ├── test_0800_damage # place sequence frames of the video
 ...
│   └── test_0849_damage
├── test_B
│   ├── test_0850_damage
 ...
│   └── test_0999_damage
├── train_damage # place mp4 videos
├── train_ref
├── val_damage
│   ├── val_0700_damage # place sequence frames of the video
 ...
│   └── val_0749_damage
└── val_ref
     ├── val_0700_ref # place sequence frames of the video
      ...
     └── val_0749_ref
```

### Testing

If you want to denoise an image sequence using the pretrained model you can execute

```
test_fastdvdnet.py \
	--test_path <path_to_input_sequence> \
	--save_path results
```
or use the script `test.sh`

**NOTES**
* The image sequence should be stored under <path_to_input_sequence>
* The model has been trained on with various degradations ().
* run with *--no_gpu* to run on CPU instead of GPU
* set *max_num_fr_per_seq* to set the max number of frames to load per sequence
* run with *--help* to see details on all input parameters

### Training

If you want to train your own models you can execute

```
train_fastdvdnet.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences> \
	--log_dir logs
```

**NOTES**
* As the dataloader in based on the DALI library, the training sequences must be provided as mp4 files, all under <path_to_input_mp4s>
* During the training, we comment the data augmentation part due to magnitude enough training set.
* The validation sequences must be stored as image sequences in individual folders under <path_to_val_sequences>
* run with *--help* to see details on all input parameters


## ABOUT

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved. This file is offered as-is,
without any warranty.

* Licence   : GPL v3+, see GPLv3.txt

