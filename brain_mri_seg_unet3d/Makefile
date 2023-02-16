JOB_NAME="custom_brain_mri_unet3d"
LOCATION="europe-west1"
CONFIG_FILE="config.yaml"
SERVICE_ACCOUNT=em52-sa-rd-ataxia-dev@ataxia-rd-dev-358f.iam.gserviceaccount.com

MACHINE_TYPE="n1-standard-8"
REPLICA_COUNT=1
EXECUTOR_IMAGE_URI_GPU=europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest
ACCELERATOR_TYPE="NVIDIA_TESLA_K80"
WORKING_DIRECTORY=brain_mri_unet3d/
SCRIPT_PATH=main.py
PROJECT_ID=ataxia-rd-dev-358f
SUBNET_sdx=subnet-data-gke-vpc-sdx-europe-west1
SUBNET_ATAXIA=em52-euwest1-rd-dev-subnet-ataxia

EPOCHS=30
SAVE_MODEL=True
REQUIREMENTS="tensorboard;scikit-image;opencv-python;natsort;albumentations;google-cloud-storage"

train_gpu:
	gcloud ai custom-jobs create \
  --region=$(LOCATION) \
  --display-name=$(JOB_NAME) \
  --config=$(CONFIG_FILE) \
  --service-account=$(SERVICE_ACCOUNT) \
  --worker-pool-spec=machine-type=$(MACHINE_TYPE),replica-count=$(REPLICA_COUNT),accelerator-type=$(ACCELERATOR_TYPE),accelerator-count=$(REPLICA_COUNT),executor-image-uri=$(EXECUTOR_IMAGE_URI_GPU),local-package-path=$(WORKING_DIRECTORY),script=$(SCRIPT_PATH),requirements=$(REQUIREMENTS) \
  --args="--epochs",$(EPOCHS),"--save_model",$(SAVE_MODEL)
  