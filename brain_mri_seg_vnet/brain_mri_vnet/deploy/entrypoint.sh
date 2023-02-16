#!/bin/bash

set -e

torchserve \
     --start \
     --ts-config=/home/model-server/config.properties \
     --models \
     brain_mri_vnet=brain_mri_vnet.mar, \
     --model-store \
     /home/model-server/model-store

tail -f /dev/null