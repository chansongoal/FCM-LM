# Introduction

This project is the official implementation of the paper titled **“[Feature Coding in the Era of Large Models: Dataset, Test Conditions, and Benchmark](https://arxiv.org/abs/2412.04307)”**. 

**"Feature coding"** is a key branch of the field of **"Coding for Machines"**, focusing on scenarios where a neural network is divided into multiple parts and deployed across different devices. In such cases, the intermediate features are encoded and transmitted between devices. The primary goal of feature coding is to minimize the bitrate under a certain constraint of task accuracy or maxmize the task accuracy under a certain constraint of bitrate.

We classify the source codes into two folders: *coding and machines*. 
The *“coding”* folder includes codes related to feature coding and the *“machines”* folder includes codes related to the machines algorithms (feature extraction and task evaluation).

# Key Features

- ## Feature Test Dataset

    - **Covers 3 types of large models:** discriminative models (DINOv2), generative models (Llama3), and hybrid models (SD3)

    - **Covers 2 types of source modalities:** textual data, visual data, and the conversion between them

    - **Covers 5 tasks:** classification, segmentation, depth estimation, common sense reasoning, and text-to-image synthesis

- ## Unified Test Condition

    - **Defines the standadized bitrate computation:** BPFP (bits per feature point)

    - **Defines the unified task accuracy evaluation:** consistent task heads and evaluation pipelines

- ## Baselines and Benchmark

    - **Traditional/handcrafted baseline:** VTM-based baseline

    - **Learning-based baseline:** hyperprior-based baseline

# Environments Set Up

## Coding

- **VTM baseline:** Any CPU device works

- **Hyperprior baseline** 

    - Step 1: build the docker image from *“docker/dockerfile_compressai_llama3”*. For example, a docker image named *“gaocs/compressai_llama3:2.0.0-cuda11.7-cudnn8-runtime”* will be built by running:

        `docker build -t gaocs/compressai_llama3:2.0.0-cuda11.7-cudnn8-runtime`

    - Step 2: Enter a docker container and run the command below to install CompressAI in editable mode:

        `cd coding/CompressAI; pip install -e .`

    Please note that the docker image only provides the running environment. We modified the original CompressAI library and thus the local installation is required. The local installation takes time to solve the dependencies. 

## Machines

- **DINOv2:** build the docker image from *“docker/dockerfile_dinov2”*.

- **Llama3:** build the docker image from *“docker/dockerfile_compressai_llama3”*. Then run:

    `cd machines/llama3/transformers; pip install -e .`

- **Stable Diffusion 3:** build the docker image from *“docker/dockerfile_sd3”*. Then run:

    `cd machines/sd3/diffuers; pip install -e .`

    Please note that the feature extraction depends on specific pytorch versions. To obtain identical features, please follow the environmental setups.

# Usage Guidelines

We classify the source codes into two folders: *coding and machines*. 
The *“coding”* folder includes codes related to feature coding and the *“machines”* folder includes codes related to the machines algorithms (feature extraction and task evaluation).

## Coding

- ### VTM baseline

    Config the *model_type, task, truncation, and quantization* parameters accordingly and arrange the corresponding folders. Then run:

    `cd coding/vtm_baseline; python vtm_baseline.py`

- ### Hyperprior baseline

    Set up the configurations and arrange the corresponding folders. Then run:

    `cd coding/CompressAI/; python run_batch.py`

    We have organized the training and inference processes in one single file *“run_batch.py”*. This file will generate two commands for training *(train_cmd)* and inference *(eval_cmd)* respectively. The validation loss curves will also be plotted after training. You can comment the training process in the *“hyperprior_train_evaluate_pipeline”* function to perform inference only.

## Machines

The feature extraction and task evaluation process use the same codes. You are free to skip the feature extraction if you have downloaded the test dataset.

- ### DINOv2

    - **Classification:** Config the parameters accordingly and run:

        `cd machines/dinov2/; python cls.py`

    - **Segmentation:** Config the parameters accordingly and run:

        `cd machines/dinov2/; python seg.py`

    - **Depth estimation:** Config the parameters accordingly and run:

        `cd machines/dinov2/; python dpt.py`

- ### Llama3

    - **Common sense reasoning** 

        Step 1: Config the parameters accordingly in *“machines/llama3/llama3.py”*.

        Step 2: Config *“save_path”* for feature extraction or *“rec_path”* to evaluate reconstructed features in *“machines/llama3/transformers/src/transformers/models/llama/modeling_llama.py”*. 

        Then run:

        `cd machines/llama3; python llama3.py`

- ### Stable Diffusion 3

    - **Text-to-image synthesis:** Config the parameters accordingly and run:

        `cd machines/sd3/; python sd3.py`

# Source Data Preparation
We have provided an exemplar folder "Data_example" which illustrates the data folder structure and includes necessary source files. Please download it and put it in a proper directory. 

- ## DINOv2

    - **Classification:** Please download the ImageNet dataset and copy the selected 100 images into the *“Data_example/dinov2/cls/source/ImageNet_Selected100”* folder. Put the corresponding source file *“imagenet_selected_label100.txt”* in the same folder.

    - **Segmentation:** Please download the VOC2012 dataset and put it in the *“Data_example/dinov2/seg/source”* folder. Put the corresponding source file *“val_20.txt”* in the same folder.

    - **Depth estimation:** Please download the NYUv2 dataset and put it in the *“Data_example/dinov2/dpt/source/NYU_Test16”* folder. Put the corresponding source file *“nyu_test.txt”* in the same folder.

- ## Llama3

    - **Common sense reasoning:** Please download [Arc-Challenge dataset](https://huggingface.co/datasets/allenai/ai2_arc/tree/main/ARC-Challenge) and process it to JSON file. Then put the processed JSON file in the *“Data_example/llama3/csr/source”* folder. We have provided the JSON file for the test dataset in the exemplar folder.

- ## Stable Diffusion 3

    - **Text-to-image synthesis:** Please download [COCO 2017 caption annotations](https://cocodataset.org/#download) and put it in the *“Data_example/sd3/tti/source”* folder. The selected 100 captions are processed by *"utils/coco_caption_processing.py"* and the processed source caption file can be found in the *“Data_example/sd3/tti/source”* folder.

# Feature Test Dataset

Download from the below links and put them in the *“Data_example/model_type/task/feature_test”* folder.

<https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ?usp=sharing>

# Pretrained Hyperprior Models

Download from the below links and put them in the *“Data_example/model_type/task/hyperprior/training_models/trunlxx_trunhxx_uniform0_bitdepth1”* folder. Make sure you change the truncation parameters accordingly.

<https://drive.google.com/drive/folders/1UmZI-cR0LdzQTl1bTzch0B4mzQvq4e9v?usp=sharing>

# Pretrained Machine Models

Download from the below links and put them in the *“Data_example/model_type/task/pretrained_head”* folder. Please make sure the folder is consistent with the codes.

- **DINOv2 backbone:**
<https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth>

    - **Classification head:**
<https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth>

    - **Segmentation head:**
<https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_linear_head.pth>

    - **Depth estimation head:**
<https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_linear4_head.pth>

- **Llama3:**
<https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main>

- **Stable Diffusion 3:**
<https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main>

# Acknowledgement
Special thanks to **Yifan Ma, Qiaoxi Chen, and Yenan Xu,** for their valuable contributions.