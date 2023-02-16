ARG BASE_CUDA_IMAGE=gcr.io/deeplearning-platform-release/base-cu110
FROM ${BASE_CUDA_IMAGE}

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get update -y
RUN apt-get install libgl1 -y

WORKDIR /home/jupyter/mri_brain_segmentation/pipelines/UNET3D
ADD . /home/jupyter/mri_brain_segmentation/pipelines/UNET3D
RUN cd /home/jupyter/mri_brain_segmentation/pipelines/UNET3D && pip install -r requirements.txt && pip install -e .