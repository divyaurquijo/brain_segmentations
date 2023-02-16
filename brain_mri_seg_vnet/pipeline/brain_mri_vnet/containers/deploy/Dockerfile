ARG BASE_CUDA_IMAGE=gcr.io/deeplearning-platform-release/pytorch-gpu
FROM ${BASE_CUDA_IMAGE}

RUN apt-get update -y
RUN apt-get install libgl1 -y

WORKDIR /home/jupyter/mri_brain_segmentation/VNET/pipeline
ADD . /home/jupyter/mri_brain_segmentation/VNET/pipeline
RUN cd /home/jupyter/mri_brain_segmentation/VNET/pipeline && pip install -r requirements.txt && pip install -e .