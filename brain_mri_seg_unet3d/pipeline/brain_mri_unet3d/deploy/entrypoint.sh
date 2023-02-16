#!/bin/bash

set -e

gsutil cp -r gs://unet3d_pipeline/model_archiver/brain_mri_unet3d.mar /home/model-server/model-store

torchserve \
     --start \
     --ts-config=/home/model-server/config.properties \
     --models \
     brain_mri_unet3d=brain_mri_unet3d.mar, \
     --model-store \
     /home/model-server/model-store

tail -f /dev/null