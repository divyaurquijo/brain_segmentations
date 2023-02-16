MODEL_NAME=brain_mri_unet3d 
SERIALIZED_FILE=/home/jupyter/mri_brain_segmentation/pipelines/UNET3D/brain_mri_unet3d/deploy/brain_mri_unet3D.pt
HANDLER=/home/jupyter/mri_brain_segmentation/pipelines/UNET3D/brain_mri_unet3d/deploy/unet3D_handler.py
MODEL_FILE=/home/jupyter/mri_brain_segmentation/pipelines/UNET3D/brain_mri_unet3d/deploy/model.py
UTILS_FILE=/home/jupyter/mri_brain_segmentation/pipelines/UNET3D/brain_mri_unet3d/deploy/utils.py
EXPORT_PATH=/home/jupyter/mri_brain_segmentation/pipelines/UNET3D/brain_mri_unet3d/deploy/model-store/

archive_save_model:
	torch-model-archiver -f \
--model-name=$(MODEL_NAME) \
--version=1.0 \
--serialized-file=$(SERIALIZED_FILE) \
--handler=$(HANDLER) \
--extra-files "$(MODEL_FILE),$(UTILS_FILE)"  \
--export-path=$(EXPORT_PATH)

	gsutil rm -r gs://unet3d_pipeline/model_archiver/*

	gsutil cp -r $(EXPORT_PATH)brain_mri_unet3d.mar gs://unet3d_pipeline/model_archiver/