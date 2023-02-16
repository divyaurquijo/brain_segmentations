import argparse
import os
import yaml

from google.cloud import storage
from munch import DefaultMunch
import pathlib
from pathlib import Path

# import tensorboard # decomment it if your using it in local (does not work with custom jobs Vertex AI)
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms.functional as fn

# from torch.utils.tensorboard import SummaryWriter # same as tensorboard
import torchvision.transforms

from dataset import BrainSegmentationDataset
from model import UNet
from trainer import Trainer
from transformations import transforms


def argparser():
    parser = argparse.ArgumentParser(
        description="Training for Brain MRI Segmentaion with UNET 3D Sclices"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./Data3D/train/",
        help="path to Data file (default: ./Data3D/train/)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./unet3d_config.yaml",
        help="path to config file (default: ./unet3d_config.yaml)",
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        help="Do you want to save the model (default: False)",
    )
    return parser.parse_args()


def main():
    """
    Load data and train unet model for brain mri segmentation to detect tumors
    """
    args = argparser()

    with open(args.config_file) as fh:
        unet3d_config = yaml.load(fh, Loader=yaml.FullLoader)

    unet3d_config = DefaultMunch.fromDict(unet3d_config)

    model_params = DefaultMunch.fromDict(unet3d_config.model_params)
    training_params = DefaultMunch.fromDict(unet3d_config.training_params)

    # root directory
    images_dir = args.images_dir

    # dataset training
    dataset_train = BrainSegmentationDataset(
        images_dir=images_dir,
        subset="train",
        image_size=128,
        transform=transforms(scale=0.05, angle=15, flip_prob=0.5),
        validation_cases=30,
    )

    # dataset validation
    dataset_valid = BrainSegmentationDataset(
        images_dir=images_dir,
        subset="validation",
        image_size=128,
        transform=transforms(scale=0.05, angle=15, flip_prob=0.5),
        validation_cases=30,
    )

    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True)

    # dataloader validation
    dataloader_validation = DataLoader(
        dataset=dataset_valid, batch_size=4, shuffle=True
    )

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Summary Writer TensorBoard
    # writer = SummaryWriter('runs/unet3D')

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

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params.lr)

    # trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        training_dataloader=dataloader_training,
        validation_dataloader=dataloader_validation,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min"),
        epochs=training_params.epochs,
        epoch=0,
        # writer = writer,
        notebook=False,
    )

    # start training
    trainer.run_trainer()

    # save the model
    if args.save_model:
        model_name = "brain_mri_unet3D_custom_jobs.pt"
        torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)
        bucket_output_name = "brain_mri_predictions"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_output_name)
        destination_blob_name = f"unet3D_custom_jobs/brain_mri_unet3D_custom_jobs"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename("./brain_mri_unet3D_custom_jobs.pt")

    print("Training done, check GCS to get your model weights")


if __name__ == "__main__":
    main()
