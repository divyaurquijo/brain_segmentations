#!/bin/bash

set -e

torchserve \
     --start \
     --ts-config=/home/model-server/config.properties \
     --models \
     brain_mri_unet3d=brain_mri_unet3d.mar, \
     --model-store \
     /home/model-server/model-store

tail -f /dev/null