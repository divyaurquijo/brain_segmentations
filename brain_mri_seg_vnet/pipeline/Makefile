PROJECT_ID=ataxia-rd-dev-358f

PREPROCESS_DATASET_COMPONENTS="preprocess_dataset"
PREPROCESS_DATASET_IMAGE="gcr.io/${PROJECT_ID}/preprocess_dataset_vnet"

TRAIN_COMPONENTS="train"
TRAIN_IMAGE="gcr.io/${PROJECT_ID}/train_vnet"

EVALUATION_COMPONENTS="evaluation"
EVALUATION_IMAGE="gcr.io/${PROJECT_ID}/evaluation_vnet"

DEPLOY_COMPONENTS="deploy"
DEPLOY_IMAGE="gcr.io/${PROJECT_ID}/deploy_vnet"

generate_docker_images:
	docker build -t $(PREPROCESS_DATASET_IMAGE) -f brain_mri_vnet/containers/$(PREPROCESS_DATASET_COMPONENTS)/Dockerfile .
	docker push $(PREPROCESS_DATASET_IMAGE)
    
	docker build -t $(TRAIN_IMAGE) -f brain_mri_vnet/containers/$(TRAIN_COMPONENTS)/Dockerfile .
	docker push $(TRAIN_IMAGE)
    
	docker build -t $(EVALUATION_IMAGE) -f brain_mri_vnet/containers/$(EVALUATION_COMPONENTS)/Dockerfile .
	docker push $(EVALUATION_IMAGE)

	docker build -t $(DEPLOY_IMAGE) -f brain_mri_vnet/containers/$(DEPLOY_COMPONENTS)/Dockerfile .
	docker push $(DEPLOY_IMAGE)


