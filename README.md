# Class Conditional Distribution Alignment and Collaborative Restructure for Few-shot Point Cloud Semantic Segmentation.
## Installation

This repository is tested with `NVIDIA RTX4090, CUDA=11.7, TORCH=1.13, PYTHON=3.9`.
```
# Create virtual env and install PyTorch
$ conda create -n pcfs python=3.9 -y
$ conda activate pcfs

$ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install basic required packages
$ pip install -r requirements.txt

# GPU kNN
$ pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# PCViT3
$ pip install spconv-cu117
$ pip install torch_scatter
$ pip install thop
$ wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.5/flash_attn-2.3.5+cu117torch1.13cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

## Data preparation
#### S3DIS
```
cd ./preprocess
python collect_s3dis_data.py --data_path $path_to_S3DIS_raw_data
python ./preprocess/room2blocks.py --data_path ./datasets/S3DIS/scenes/ --dataset s3dis
```
#### ScanNet
```
cd ./preprocess
python collect_scannet_data.py --data_path $path_to_ScanNet_raw_data
python ./preprocess/room2blocks.py --data_path ./datasets/ScanNet/scenes/ --dataset scannet
```

## Running
### Pretrain

```
bash ./scripts/pretrain.sh
```

### Train
```
bash ./scripts/train.sh
```

## Acknowledgement
We thank [attMPTI](https://github.com/Na-Z/attMPTI), [MAMBA](https://github.com/state-spaces/mamba), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) for sharing the source code.
