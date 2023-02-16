from datetime import datetime
import os
from os.path import join

from google.cloud import storage
import numpy as np
import pathlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ts.torch_handler.base_handler import BaseHandler

from dataset import SegmentationDataSet3
from model import VNet
from utils import (
    get_filenames_of_path,
    postprocess,
    segmentation_pred,
    transforms_testing,
)


class TransformersClassifierHandler(BaseHandler):
    """
    The handler takes an input a root folder with images and return the segmentation of all the images.
    """

    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Loads the brain_mri_unet.pt file and initializes the model object."""
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # init models
        serialized_file = self.manifest["model"]["serializedFile"]
        self.model = VNet(elu=True, in_channels=1, classes=3).to(self.device)
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """Preprocessing input images"""
        now = datetime.now()
        test_path = pathlib.Path.cwd() / f"test_{now}"
        os.mkdir(test_path)
        input_path = os.path.join(test_path, "Input")
        target_path = os.path.join(test_path, "Target")
        os.mkdir(input_path)
        os.mkdir(target_path)

        bucket_input_name = data[0]["body"]["instances"][0]["bucket_input"]
        objects_input_name = data[0]["body"]["instances"][1]["objects_input"]
        bucket_output_name = data[0]["body"]["instances"][2]["bucket_output"]

        # Collect all the images for inference from the bucket in GCS and copy it in a folder
        for object_name in objects_input_name:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_input_name)
            blob = bucket.blob(object_name)
            file_name = object_name.split("/")[-1]
            if file_name.endswith("_mask.tif"):
                destination_file = os.path.join(target_path, file_name)
                blob.download_to_filename(destination_file)
            else:
                destination_file = os.path.join(input_path, file_name)
                blob.download_to_filename(destination_file)

        # input and target files
        inputs_test = get_filenames_of_path(test_path / "Input")
        targets_test = get_filenames_of_path(test_path / "Target")

        dataset_test = SegmentationDataSet3(
            inputs=inputs_test,
            targets=targets_test,
            transform=transforms_testing(),
            use_cache=False,
            pre_transform=None,
        )
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            # batch_size of 2 won't work because the depth dimension is different between the 2 samples
            shuffle=False,
        )

        return dataloader_test, bucket_output_name

    def inference(self, inputs):
        """Segmentation of a tumor in a 3D image."""
        predictions = []
        now = datetime.now()
        dataloader_test, bucket_output_name = inputs

        for indice, (image, mask) in enumerate(dataloader_test):
            input_image, input_mask = image.to(self.device), mask.to(self.device)
            with torch.no_grad():
                output = self.model(input_image)  # send through model/network
                output = postprocess(output)  # postprocess the prediction
                filepath_out_pred = "/home/model-server/segmentation_prediction.gif"
                segmentation_pred_path = segmentation_pred(
                    input_image, output, filepath_out_pred
                )

        # Store that image in GCS bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_output_name)
        destination_blob_name = f"vnet_output/prediction_vnet_{now}"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename("/home/model-server/segmentation_prediction.gif")

        predictions.append({"predictions": "Sucess, check GCS"})

        return predictions

    def postprocess(self, inference_output):
        return inference_output
