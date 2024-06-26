# Swin-Fuse: Deep learning segmentation for 3D FIB-SEM data of porous cathode materials using fused image inputs

**Swin-Fuse** is a deep learning pipeline based on the Swin Transformer to segment FIB-SEM data of a lithium cobalt oxide electrode, utilizing fused secondary electron (SE) and backscattered electron (BSE) images. The main codes are forked from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [StyTR-2](https://github.com/diyiiyiii/StyTR-2) with essential enhancement and bug fixes.

## Prerequisites

- [PyTorch](https://pytorch.org/get-started/previous-versions/) 2.0.1
- opencv-python, numpy, scipy, tqdm, etc.

We recommend running on a machine with a discrete NVIDIA GPU and the appropriate [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

Please follow the commands to set up the MMSegmentation environment:

```bash
pip install -U openmim
mim install mmengine==0.10.2
mim install "mmcv>=2.0.0"
```

Then `cd` into the `mmsegmentation` directory and run:

```bash
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Start Inferring immediately!

We provide the `infer.py` Python script for quick beginning. Before running it, please download the required pre-trained models and put them in the correct folders:

1. Pre-trained models for ${\rm StyTr^2}$: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing).

   Please download them and put them into the directory  `./StyTR-2/experiments/`.

2. Pretrained models for Swin Transformer [backbones](https://drive.google.com/drive/folders/1YWgUXNZtCHk4gS2m-VU_ZgVwsj5HOpgL?usp=drive_link).

   Please download all of them and put them into the directory `./checkpoints/`.

You can now run the inferring script as follows:

```bash
python infer.py --img ${SE_IMAGE_PATH}
```

The script does the inference upon a single SE image by default. You can pass the `--mode` argument to specify inferring modes. For example:

```bash
python infer.py --mode fuse --img ${SE_IMAGE_PATH} --img2 ${BSE_IMAGE_PATH}
```

This will style-transfer the BSE image and fuse it with the SE image as the model input. Or:

```bash
python infer.py --mode dual --img ${SE_IMAGE_PATH} --img2 ${BSE_IMAGE_PATH}
```

This will pass both SE and BSE images as the model input without fusing them. Remember to specify a second image path (which is the BSE image in our cases) when inferring in "fuse" and "dual" modes.

The inference results will be saved in `./results/`. If you want it to be somewhere else, you can pass an additional `--out-dir` argument. For example:

```bash
python infer.py --mode fuse --img sample_se.png --img2 sample_bse.png --out-dir mmsegmentation/my_out_dir
```

Here the `sample_se.png` and `sample_bse.png` are example SE and BSE images provided in the main directory. As specified, the inference results will now be saved under `./mmsegmentation/my_out_dir/`.

## Train on your own data

### Dataset generation

Put the original SE and BSE training images in the directory `./data/se_img/` and `./data/bse_img/`, each containing `train` and `test` dataset subfolders. Please ensure the corresponding SE and BSE images to be fused have the same file names. The segmentation mask labels should be in `./data/masks` with the same directory structure.

```
data/
├── se_img/
│   ├── train/
│   └── val/
├── bse_img/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

Run:

```bash
python gen_dataset.py
```

The generated fused images will be saved under `./mmsegmentation/data/fuse/`. This may take some time if you have a large dataset. 

### Training

To train the model on the fused FIB image dataset, `cd` into the `./mmsegmentation` directory and run in bash:

```bash
python tools/train.py configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_4xb2-40ke_fib-fuse-512x512.py --amp
```

Alternatively, if you wish to train the model with original SE and BSE images without fusing them, run:

```bash
python tools/train.py configs/swin/swin-tiny-patch4-window7_upernet_4xb2-40ke_fib-dual-512x512.py --amp
```

Or if you want to train on mere SE images, run:

```bash
python tools python tools/train.py configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_4xb2-40ke_fib-512x512.py --amp
```

The training logs and checkpoint files will be saved under `./mmsegmentation/work_dirs/`.

## Acknowledgment

This repository derives from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [StyTR-2](https://github.com/diyiiyiii/StyTR-2). For advanced training and testing instructions, please refer to [MMSegmentation's official documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html).