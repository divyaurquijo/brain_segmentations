import argparse
import os
import yaml

from munch import DefaultMunch
import torch

from model import UNet
from utils import (
    predict,
    preprocess,
    postprocess,
    preprocess_images,
    segmentation_target,
    segmentation_pred,
)


def argparser():
    parser = argparse.ArgumentParser(
        description="Inference for Brain MRI Segmentaion with UNET 3D"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./brain_mri_unet3D.pt",
        help="path to weights file",
    )
    parser.add_argument(
        "--images", type=str, default="./Data3D/test/", help="root folder with images"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./unet3d_config.yaml",
        help="path to config file (default: ./unet3d_config.yaml)",
    )
    return parser.parse_args()


def main():
    """
    Inference on new images for instance segmentation of brain mri tumor
    """

    args = argparser()

    with open(args.config_file) as fh:
        unet3d_config = yaml.load(fh, Loader=yaml.FullLoader)

    unet3d_config = DefaultMunch.fromDict(unet3d_config)
    model_params = DefaultMunch.fromDict(unet3d_config.model_params)

    root = args.images
    test_directories = list(os.listdir(root))

    # Define device on CUDA
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    model = UNet(
        model_params.in_channels,
        model_params.out_channels,
        model_params.n_blocks,
        model_params.start_filters,
        model_params.activation,
        model_params.normalization,
        model_params.conv_mode,
        model_params.dim,
    ).to(device)
    model_weights = torch.load(args.weights, map_location=device)
    model.load_state_dict(model_weights)

    for directory in test_directories:
        directory_path = os.path.join(root, directory)
        inputs_test = os.path.join(directory_path, "Input")
        targets_test = os.path.join(directory_path, "Target")

        images_res, targets_res = preprocess_images(inputs_test, targets_test)
        # predict the segmentation maps
        output = [
            predict(img, model, preprocess, postprocess, device) for img in images_res
        ]

        # Create a segmentation array for ground truth
        segmentation_target(directory, directory_path, images_res, targets_res)

        # Create a segmentations array for predictions
        segmentation_pred(directory, directory_path, images_res, output)

    print("Open the gif in your folder to see the result of the segmentation")


if __name__ == "__main__":
    main()
