# Accurate Detection and Instance Segmentation of Unstained Living Adherent Cells in Differential Interference Contrast Images

Brief description or abstract of your research project.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

Accurately detecting and segmenting unstained living adherent cells in differential interference contrast (DIC) images remains a challenging task in biomedical laboratories. In this study, we curated a new dataset of 520 pairs of DIC images containing 12,198 HepG2 human liver cancer cells, along with annotated ground truth. We also developed a novel cascaded method for enhanced detection and segmentation, which consists of three steps. The first step is to train a Mask R-CNN, a well-known deep-learning algorithm, with relaxed non-maximum suppression (NMS) on an interim dataset consisting of original and rotated images. The second step is to feed another dataset into the Mask R-CNN to produce comprehensive predictions, followed by connectivity-based filtering and categorization into two classes of image (mask) patches. A residual network (ResNet) classifier is then trained on these patches to decide which masks to be retained. In the third step, the test set goes through the iterative processes in the previous steps and derives the final instance segmentation results. Experimental results showed our method achieved a notable average precision (AP)bbox score of 0.634 and a commendable APsegm score of 0.555, outperforming other methods. This success opens new possibilities for cell image analysis, benefiting biomedical applications such as cell microinjection.

## Getting Started

This section should guide users on how to get your project up and running on their local machine.

### Prerequisites

Before you can use this project, please ensure you have the following prerequisites installed:

- [PyTorch](https://pytorch.org/) >= 1.8.2
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) == 10.2
- [MMDetection](https://github.com/open-mmlab/mmdetection) >= 2.28.2
- [MMPretrain](https://github.com/open-mmlab/mmpretrain) >= 0.25.0

We recommend using Conda to automatically manage and install compatible packages, but theoretically, the following versions should work:

- PyTorch >= 1.5.1
- torchvision >= 0.6.1
- MMDetection >= 2.23.0
- MMPretrain >= 0.23.0

### Installation

Follow these step-by-step instructions to install the project:

1. Create a Conda environment and activate it:

   ```bash
   conda create --name openmmlab python=3.8 -y
   conda activate openmmlab
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
   pip install openmim
   mim install mmcv-full
   wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.28.2.zip
   unzip mmdetection-2.28.2.zip
   cd mmdetection-2.28.2
   pip install -v -e .
   cd ..
   wget https://github.com/open-mmlab/mmpretrain/archive/refs/tags/v0.25.0.zip
   unzip mmpretrain-0.25.0.zip
   cd mmpretrain-0.25.0
   mim install -e .
   cd ..

2. Copy the dataset to `mmdetection-2.28.2/data/hepg2` and make sure its structures are as follows:

    <pre>
    mmdetection-2.28.2
       |-- demo
       |-- mmdet
       |-- tools
       |-- configs
       |-- data
       |   |-- hepg2
       |   |   |-- coco_sytle
       |   |   |   |-- train
       |   |   |   |   |-- images
       |   |   |   |   |-- Visualization
       |   |   |   |   |-- annotations.json
       |   |   |   |-- val
       |   |   |   |   |-- images
       |   |   |   |   |-- Visualization
       |   |   |   |   |-- annotations.json
       |   |   |   |-- test
       |   |   |   |   |-- images
       |   |   |   |   |-- Visualization
       |   |   |   |   |-- annotations.json
    </pre>

3. Download and copy all files from our `mmdetection-2.28.2` repo to the previously downloaded `mmdetection-2.28.2` folder

4. Download and copy all files from our `mmpretrain-0.25.0` repo to the previously downloaded `mmpretrain-0.25.0` folder

5. Modify the `mmdetection-2.28.2` and `mmpretrain-0.25.0` paths in the following sample script and run it

<pre>
cd D:\mmdetection-2.28.2\
python demo/0_create_rotated_dataset.py --dataset hepg2
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_hepg2.py
python tools/val.py configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_hepg2.py work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/latest.pth --show-dir work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/val/ --eval bbox segm --eval-options jsonfile_prefix=work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/val/results
python tools/test.py configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_hepg2.py work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/latest.pth --show-dir work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/test/ --eval bbox segm --eval-options jsonfile_prefix=work_dirs/mask_rcnn_r50_caffe_fpn_1x_hepg2/test/results
python demo/1_create_cls_dataset.py --dataset hepg2 --set val --config_dir mask_rcnn_r50_caffe_fpn_1x_hepg2 --img_height 864 --img_width 1152
python demo/1_create_cls_dataset.py --dataset hepg2 --set test --config_dir mask_rcnn_r50_caffe_fpn_1x_hepg2 --img_height 864 --img_width 1152
cd D:\mmpretrain-0.25.0\
python tools/train.py configs/resnet/resnet18_8xb32_in1k_hepg2.py --work-dir work_dirs/resnet18_8xb32_in1k_hepg2/mask_rcnn_r50_caffe_fpn_1x_hepg2
python demo/2_gen_cls_vis.py --mmdet_config mask_rcnn_r50_caffe_fpn_1x_hepg2 --mmcls_config resnet18_8xb32_in1k_hepg2 --device cuda:0
python demo/3_mask_nms.py --set test --mmdet_config mask_rcnn_r50_caffe_fpn_1x_hepg2 --mmcls_config resnet18_8xb32_in1k_hepg2
python demo/4_perf_eval.py --set test --mmdet_config mask_rcnn_r50_caffe_fpn_1x_hepg2 --mmcls_config resnet18_8xb32_in1k_hepg2
</pre>
    
6. All final predictions results are stored in a text file like `G:\dvsisn_hepg2\mmpretrain-0.25.0\work_dirs\resnet18_8xb32_in1k_hepg2\mask_rcnn_r50_caffe_fpn_1x_hepg2\vis_cls_final\collect\stats.txt`
