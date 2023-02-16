MODEL_NAME=brain_mri_vnet 
SERIALIZED_FILE=./brain_mri_vnet/deploy/brain_mri_vnet.pt
HANDLER=./brain_mri_vnet/deploy/vnet_handler.py
MODEL_FILE=./brain_mri_vnet/deploy/model.py
UTILS_FILE=./brain_mri_vnet/deploy/utils.py
DATASET_FILE=./brain_mri_vnet/deploy/dataset.py
EXPORT_PATH=./brain_mri_vnet/deploy/model-store/

archive_save_model:
	torch-model-archiver -f \
--model-name=$(MODEL_NAME) \
--version=1.0 \
--serialized-file=$(SERIALIZED_FILE) \
--handler=$(HANDLER) \
--extra-files "$(MODEL_FILE),$(UTILS_FILE),$(DATASET_FILE)"  \
--export-path=$(EXPORT_PATH)

	gsutil rm -r gs://vnet_pipeline/model_archiver/*

	gsutil cp -r $(EXPORT_PATH)brain_mri_vnet.mar gs://vnet_pipeline/model_archiver/