FROM pytorch/torchserve:latest-gpu

# install dependencies
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install curl -y
USER model-server
RUN python3 -m pip install --upgrade pip
RUN pip3 install torchvision
RUN pip3 install scikit-image
RUN pip3 install opencv-python
RUN pip3 install natsort
RUN pip3 install google-cloud-storage
RUN pip3 install torch-summary

# copy model artifacts, custom handler and other dependencies
COPY ./vnet_handler.py /home/model-server/
COPY ./model.py /home/model-server/
COPY ./utils.py /home/model-server/
COPY ./dataset.py /home/model-server/
COPY ./brain_mri_vnet.pt /home/model-server/
COPY ./config.properties /home/model-server/
COPY ./entrypoint.sh /home/model-server/

# change permissions
USER root
RUN chmod +x /home/model-server/entrypoint.sh
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 8081
EXPOSE 8082
EXPOSE 8083

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=brain_mri_vnet \
  --version=1.0 \
  --serialized-file=/home/model-server/brain_mri_vnet.pt \
  --handler=/home/model-server/vnet_handler.py \
  --extra-files "/home/model-server/model.py,/home/model-server/utils.py,/home/model-server/dataset.py"  \
  --export-path=/home/model-server/model-store

# start torchserve
ENTRYPOINT ["/home/model-server/entrypoint.sh"]