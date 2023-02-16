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

# Install gsutil
USER root
RUN apt-get install apt-transport-https ca-certificates gnupg -y
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y
USER model-server
RUN gcloud init

# copy model artifacts, custom handler and other dependencies
COPY ./unet3D_handler.py /home/model-server/
COPY ./model.py /home/model-server/
COPY ./utils.py /home/model-server/
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
