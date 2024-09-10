# Patch-base progressive 3D Point Set Upsampling
This is the official implementation for paper "[Patch-base progressive 3D Point Set Upsampling](https://arxiv.org/abs/1811.11286)".


> [!IMPORTANT]  
> Regrettably, the storage for the data and checkpoints have expired and couldn't be recovered. I'm very sorry for the inconvenience.
 
```diff
+ I've removed the evaluation code, because the included NUC code proposed by PU_Net has issues, in our paper we didn't use this metric.
+ A new pytorch implemention using pytorch 1.0 is available at https://github.com/yifita/3PU_pytorch
+ Added MNIST contour data
```

<center>

virtual scan | 16x upsampled | real scan | 16x upsampled
------------- | ------------- | ------------- | -------------
<img src="figures/retheur03.png" height="300" />|<img src="figures/retheur02.png" height="300" /> | <img src="figures/06.png" width="240" />|<img src="figures/07.png" width="240" />

Input | output | ground truth | Input reconstruction | output reconstruction | ground truth reconstruction| 
------------- | ------------- | -------------  | -------------  | -------------  | ------------- 
<img src="figures/5000_input00.png" width="200" /> | <img src="figures/5000_ours00.png" width="200" /> | <img src="figures/5000_gt00.png" width="200" /> | <img src="figures/5000_input01.png" width="200" /> | <img src="figures/5000_ours01.png" width="200" /> | <img src="figures/5000_gt01.png" width="200" /> 
<img src="figures/625_input00.png" width="200" /> | <img src="figures/625_ours00.png" width="200" /> | <img src="figures/625_gt00.png" width="200" /> | <img src="figures/625_input01.png" width="200" /> | <img src="figures/625_ours01.png" width="200" /> | <img src="figures/625_gt01.png" width="200" />

</center>

## Quick Demo ##

```bash
# clone
git clone https://github.com/yifita/3PU.git --recursive
cd 3PU
# download pretrained models
curl -o model/pretrained.zip -O https://polybox.ethz.ch/index.php/s/TZjUeCWFPlmv0nj/download
unzip -d model/ model/pretrained.zip
# download test data
curl -o data/test_data/test_points.zip -O https://polybox.ethz.ch/index.php/s/wxKg4O05JnyePDK/download
unzip -d data/test_data/ data/test_data/test_points.zip

# conda environment
conda update -n base -c defaults conda
conda env create -f environment.yml

# automatically add cuda library path permanently to the current conda enviroment
mkdir -p $HOME/anaconda3/envs/PPPU/etc/conda/activate.d
cp activate_cuda_90.sh $HOME/anaconda3/envs/PPPU/etc/conda/activate.d
mkdir -p $HOME/anaconda3/envs/PPPU/etc/conda/deactivate.d
cp deactivate_cuda_90.sh $HOME/anaconda3/envs/PPPU/etc/conda/deactivate.d
conda activate PPPU

# compile
cd code/tf_ops
cd CD && ./tf_nndistance_compile.sh
cd ../grouping && ./tf_grouping_compile.sh
cd ../sampling && ./tf_sampling_compile.sh

# run code
cd code
python main_curriculum_interleave.py --phase test --id sketchfab_poisson --model dense_interlevelplus \
--num_point 312 --up_ratio 16 --step_ratio 2 --patch_num_ratio 3 \
--test_data "data/test_data/sketchfab_poisson/poisson_5000/*.xyz"
```

## data preparation ##
### Sketchfab dataset ###
Our 3D models are trained and tested using the [Sketchfab](https://sketchfab.com/) dataset created by ourselves. It consists of 90 training models and 13 testing models collected from [Sketchfab](https://sketchfab.com/). ~~You can download the original meshes here: [training][train_mesh] and [testing][test_mesh].~~

### MNIST contour data ###
Per request, we provide the MNIST contour dataset used as toy examples for visualization and ablation studies in our paper. 
~~[training](https://polybox.ethz.ch/index.php/s/fUgbJGTl3CRzq2m) contains 5 resolutions from 50 to 800 points. Each resolution contains 48,000 ply files.~~
~~[testing](https://polybox.ethz.ch/index.php/s/11eJpISNO9v4gw7) contains 5 resolutions same as the training set, each containig 12,000 ply files.~~

### Input points ###
We trained our models with two kinds of input points: point sets generated using poisson disc sampling and virtual scanner. 
~~Download the [test][test_points] and [training data][train_record] used in our experiments. Unzip the test data to `data/test_data/` and unzip to training data to `record_data/`~~

Additionally, you can create your own data with a virtual scanner or poisson disc sampling. 
```
# compile
cd prepare_data
cmake .
make -j

# sample
cd Poisson_sample
./run_pd_sampling.sh DIR_TO_MESH DIR_TO_OUTPUT "**/*.EXTENSION" NUM_POINTS

# scan
cd polygonmesh_base
./run_scan.sh DIR_TO_MESH DIR_TO_OUTPUT 1 "**/*.ply"
```

### Pretrained model ###
We provide two models trained using each of the above mentioned data. ~~Download them [here][pretrained], and unzip them under `model/`.~~

## tensorflow code compile ##
0. Install cuda 9.0, cudnn and nccl if you haven't done so yet.
1. Create conda environment `conda env create -f environment.yml`.
2. Install tensorflow, we use tensorflow 1.11 in this project, but tensorflow >= 1.5 should work.
3. Compile the custom tensorflow code in `code/tf_ops` by running `./tf_*_compile.sh`.

## Testing ##
### paper results ###
Test on sparse poisson input:
```
python main_curriculum_interleave.py --phase test --id sketchfab_poisson --model dense_interlevelplus --num_point 156 --up_ratio 16 --step_ratio 2 --patch_num_ratio 3 \
--test_data "data/test_data/sketchfab_poisson/poisson_625/*.xyz"
```
Test on dense poisson input:
```
python main_curriculum_interleave.py --phase test --id sketchfab_poisson --model dense_interlevelplus --num_point 312 --up_ratio 16 --step_ratio 2 --patch_num_ratio 3 \
--test_data "data/test_data/sketchfab_poisson/poisson_5000/*.xyz"
```
Test on scan inputs:
```
python main_curriculum_interleave.py --phase test --id sketchfab_scan --model dense_interlevelplus --num_point 312 --up_ratio 16 --step_ratio 2 --patch_num_ratio 3 \
--test_data "data/test_data/sketchfab_scan/**/*.ply"
```

### test with your own data ###
We support ply, xyz, and pcd as input.
Simply set option `--test_data` to a relative search pattern for your own data.

```
# SEARCH_PATH_FOR_TEST_DATA could be `my_test_data/**/*.ply
python main_curriculum_interleave.py --phase test --id MODEL_NAME --model dense_interlevelplus --num_point 312 --up_ratio 16 --step_ratio 2 --patch_num_ratio 3 \
--test_data SEARCH_PATH_FOR_TEST_DATA
```

## Training ##
Download training data as described [here](#input-points). Unzip to `record_data/`.

Train with poisson data
```
python main_curriculum_interleave.py --phase train --id ppu --model dense_interlevelplus --num_point 312 --up_ratio 16 --step_ratio 2 --num_shape_point 5000 --no-repulsion --dense_n 3 --knn 32 --num_point 312 --up_ratio 16 --step_ratio 2 --num_shape_point 625 --record_data "../record_data/poisson_5000_poisson_10000_poisson_20000_poisson_40000_poisson_80000_p312_shard[0-9].tfrecord" \
--stage_step 15000 --max_epoch 400 --gpu 0 --patch_num_ratio 3 --jitter --jitter_sigma 0.01 --jitter_max 0.03 \
--test_data SEARCH_PATH_FOR_TEST_DATA
```
Train with scan data
```
python main_curriculum_interleave.py --phase train --id ppu_scan --model dense_interlevelplus --num_point 312 --up_ratio 16 --step_ratio 2 --num_shape_point 5000 --no-repulsion --record_data "../record_data/res_5000_res_10000_res_20000_res_40000_res_80000_res_160000_p312_shard[0-9].tfrecord" \
--stage_step 15000 --max_epoch 400 --gpu 0 --patch_num_ratio 3 --jitter --jitter_sigma 0.01 --jitter_max 0.03 \
--test_data SEARCH_PATH_FOR_TEST_DATA
```
### Create new training data ###
You can create new training data by calling `python create_tfrecords.py` in `prepare_data` folder.

---
## cite ##
If you find this code or data useful in your work, please cite our paper:
```
@InProceedings{Yifan_2019_CVPR,
author = {Yifan, Wang and Wu, Shihao and Huang, Hui and Cohen-Or, Daniel and Sorkine-Hornung, Olga},
title = {Patch-Based Progressive 3D Point Set Upsampling},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}

```
## Links to related projects ##
PU-Net: [https://github.com/yulequan/PU-Net](https://github.com/yulequan/PU-Net)

PointNet++: [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
