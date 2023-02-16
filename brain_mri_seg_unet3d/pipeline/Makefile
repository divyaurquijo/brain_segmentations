PROJECT_ID=ataxia-rd-dev-358f

PREPROCESS_DATASET_COMPONENTS="preprocess_dataset"
PREPROCESS_DATASET_IMAGE="gcr.io/${PROJECT_ID}/preprocess_dataset_unet3d"

TRAIN_COMPONENTS="train"
TRAIN_IMAGE="gcr.io/${PROJECT_ID}/train_unet3d"

EVALUATION_COMPONENTS="evaluation"
EVALUATION_IMAGE="gcr.io/${PROJECT_ID}/evaluation_unet3d"

DEPLOY_COMPONENTS="deploy"
DEPLOY_IMAGE="gcr.io/${PROJECT_ID}/deploy_unet3d"


generate_docker_images:
	docker build -t $(PREPROCESS_DATASET_IMAGE) -f brain_mri_unet3d/containers/$(PREPROCESS_DATASET_COMPONENTS)/Dockerfile .
	docker push $(PREPROCESS_DATASET_IMAGE)
    
	docker build -t $(TRAIN_IMAGE) -f brain_mri_unet3d/containers/$(TRAIN_COMPONENTS)/Dockerfile .
	docker push $(TRAIN_IMAGE)
    
	docker build -t $(EVALUATION_IMAGE) -f brain_mri_unet3d/containers/$(EVALUATION_COMPONENTS)/Dockerfile .
	docker push $(EVALUATION_IMAGE)

	docker build -t $(DEPLOY_IMAGE) -f brain_mri_unet3d/containers/$(DEPLOY_COMPONENTS)/Dockerfile .
	docker push $(DEPLOY_IMAGE)

