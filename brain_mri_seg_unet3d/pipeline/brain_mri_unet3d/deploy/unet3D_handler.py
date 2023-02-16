from datetime import datetime
import os
from os import listdir
from os.path import join
import subprocess

from google.cloud import storage
from natsort import natsorted
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import torch
from ts.torch_handler.base_handler import BaseHandler

from model import UNet
from utils import predict, preprocess, postprocess, draw_segmentation_map, save_gif


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
        self.model = UNet(
            in_channels=3,
            out_channels=2,
            n_blocks=4,
            start_filters=32,
            activation="relu",
            normalization="batch",
            conv_mode="same",
            dim=2,
        ).to(self.device)
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """Preprocessing input images"""
        now = datetime.now()
        test_path = f"/home/model-server/test_{now}/"
        os.mkdir(test_path)

        bucket_input_name = data[0]["body"]["instances"][0]["bucket_input"]
        objects_input_name = data[0]["body"]["instances"][1]["objects_input"]
        bucket_output_name = data[0]["body"]["instances"][2]["bucket_output"]

        # Collect all the images for inference from the bucket in GCS and copy it in a folder
        for object_name in objects_input_name:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_input_name)
            blob = bucket.blob(object_name)
            file_name = object_name.split("/")[-1]
            destination_file = os.path.join(test_path, file_name)
            blob.download_to_filename(destination_file)

        # input and target files
        images_names = list(natsorted(os.listdir(test_path)))

        # read images and store them in memory
        images = [
            imread(os.path.join(test_path, img_name)) for img_name in images_names
        ]

        # Resize images and targets
        dataset_test = [resize(img, (128, 128, 3)) for img in images]

        return dataset_test, bucket_output_name

    def inference(self, inputs):
        """Segmentation of a tumor in a 3D image."""
        predictions = []
        predictions_tensor = []
        now = datetime.now()
        dataset_test, bucket_output_name = inputs

        with torch.no_grad():
            output = [
                predict(img, self.model, preprocess, postprocess, self.device)
                for img in dataset_test
            ]

        segmentations_target = []
        for indice in range(len(dataset_test)):
            segmentations_target.append(
                draw_segmentation_map(dataset_test[indice], output[indice], 0)
            )
        filepath_out = "/home/model-server/segmentation_prediction.gif"
        save_gif(segmentations_target, filepath_out)

        # Store that image in GCS bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_output_name)
        destination_blob_name = f"unet3D_output/prediction_unet3D_{now}"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename("/home/model-server/segmentation_prediction.gif")

        predictions.append({"predictions": "Sucess, check GCS"})

        return predictions

    def postprocess(self, inference_output):
        return inference_output
