# 3D Semantic Segmentation using MinkowskiEngine & ScanNet
 [The dataset can be downloaded from here](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation)
  [We have also uploaded the code here ](https://github.com/suponmazumdar/3D-Semantic-Segmentation.git)
  1. Setup
Setting up for this project involves installing dependencies. 

### Installing dependencies
To install all the dependencies, please run the following:
```shell script
sudo apt install build-essential python3-dev libopenblas-dev
conda env create -f env.yml
conda activate growsp
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

### 2.2 ScanNet
Download the ScanNet dataset from [the official website](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation). 
You need to sign the terms of use. Uncompress the folder and move it to 
`${your_ScanNet}`.

- Preparing the dataset:
```shell script
python data_prepare/data_prepare_ScanNet.py --data_path ${your_ScanNet}
```
This code will preprcocess ScanNet and put it under `./data/ScanNet/processed`

- Construct initial superpoints:
```shell script
python data_prepare/initialSP_prepare_ScanNet.py
```
This code will construct superpoints on ScanNet and put it under `./data/ScanNet/initial_superpoints`

- Training:
```shell script
CUDA_VISIBLE_DEVICES=0, python train_ScanNet.py
```
The output model and log file will be saved in `./ckpt/ScanNet` by default.

 
 
 
