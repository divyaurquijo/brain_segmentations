import argparse
import yaml

from google.cloud import storage
from munch import DefaultMunch
import pathlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataSet3
from model import VNet
from utils import (
    get_filenames_of_path,
    postprocess,
    segmentation_pred,
)
from transformations import transforms_testing


def argparser():
    parser = argparse.ArgumentParser(
        description="Inference for Brain MRI Segmentaion with VNET"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./brain_mri_vnet.pt",
        help="path to weights file",
    )
    parser.add_argument(
        "--images", type=str, default="./Data3D/test/", help="root folder with images"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./vnet_config.yaml",
        help="path to config file (default: ./vnet_config.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to segmentation output file (default: ./)",
    )
    return parser.parse_args()


def main():
    """
    Load data and predict tumors on brain mri with vnet segmentation model
    """
    args = argparser()

    with open(args.config_file) as fh:
        vnet_config = yaml.load(fh, Loader=yaml.FullLoader)

    vnet_config = DefaultMunch.fromDict(vnet_config)

    model_params = DefaultMunch.fromDict(vnet_config.model_params)

    test_images_dir = args.images
    root_test = pathlib.Path.cwd() / test_images_dir

    # input and target files
    inputs_test = get_filenames_of_path(root_test / "Input")
    targets_test = get_filenames_of_path(root_test / "Target")

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

    # Define device on CUDA
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    model = VNet(model_params.elu, model_params.in_channels, model_params.classes).to(
        device
    )
    model_weights = torch.load(args.weights, map_location=device)
    model.load_state_dict(model_weights)

    model.eval()
    for indice, (image, mask) in enumerate(dataloader_test):
        input_image, input_mask = image.to(device), mask.to(device)
        with torch.no_grad():
            output = model(input_image)  # send through model/network
            output = postprocess(output)  # postprocess the prediction
            filepath_out_pred = (
                args.output_dir + f"segmentation_prediction_{indice}.gif"
            )
            segmentation_pred(input_image, output, filepath_out_pred)

    print("Open the gif in your folder to see the result of the segmentation")


if __name__ == "__main__":
    main()
