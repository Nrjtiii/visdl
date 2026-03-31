# NYCU Selected Topics in Visual Deep Learning HW1

* **Name          : Radya Wirawan Nurjati**
* **Student ID    : 314540030**

## Introduction

This project implements a PyTorch and Hugging Face transformers pipeline to fine-tune a timm/resnetrs200.tf_in1k image classification model. The workflow utilizes spatial data augmentation (rotation, flipping, and blur) and color jittering during the training phase, and it automatically evaluates performance to generate a final submission.csv file containing predicted labels for the test dataset.

## Environment Setup

This project uses Conda for dependency management
'''
conda env create -f environment.yml
conda activate image-class-env
'''

## Usage

### Training
'''
python train.py
'''

### Inference
'''
python infer.py
'''


## Dataset Structure
The script expects a standard image folder dataset located at `dataset/data`. Ensure your directory is structured like this for `load_dataset` to parse the labels correctly:

```text
dataset/data/
├── train/
│   ├── class_0/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class_1/
│       ├── img2.jpg
│       └── ...
├── validation/
│   ├── class_0/
│   └── class_1/
└── test/
    ├── img3.jpg
    └── ...

## Performance Snapshot
    
    ![Performance Snapshot] (image.png)