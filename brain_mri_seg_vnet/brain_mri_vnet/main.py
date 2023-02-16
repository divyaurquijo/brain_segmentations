import argparse
import yaml

from google.cloud import storage
import monai
from monai.losses.dice import DiceLoss
from munch import DefaultMunch
import pathlib
from pathlib import Path
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms

from dataset import SegmentationDataSet3
from model import VNet
from trainer import Trainer
from transformations import (
    transforms_training,
)
from utils import (
    get_filenames_of_path,
)


def argparser():
    parser = argparse.ArgumentParser(
        description="Training for Brain MRI Segmentaion with VNET"
    )
    parser.add_argument(
        "--train_images_dir",
        type=str,
        default="Data3D/train",
        help="path to train Data file (default: Data3D/train)",
    )
    parser.add_argument(
        "--val_images_dir",
        type=str,
        default="Data3D/val",
        help="path to val Data file (default: Data3D/val)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./vnet_config.yaml",
        help="path to config file (default: ./vnet_config.yaml)",
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
    Load data and train vnet model for brain mri segmentation to detect tumors with volumes
    """
    args = argparser()

    with open(args.config_file) as fh:
        vnet_config = yaml.load(fh, Loader=yaml.FullLoader)

    vnet_config = DefaultMunch.fromDict(vnet_config)

    model_params = DefaultMunch.fromDict(vnet_config.model_params)
    training_params = DefaultMunch.fromDict(vnet_config.training_params)

    # root directory
    train_images_dir = args.train_images_dir
    val_images_dir = args.val_images_dir

    root_train = pathlib.Path.cwd() / train_images_dir
    root_val = pathlib.Path.cwd() / val_images_dir

    # input and target files
    inputs_train = get_filenames_of_path(root_train / "Input")
    targets_train = get_filenames_of_path(root_train / "Target")

    inputs_val = get_filenames_of_path(root_val / "Input")
    targets_val = get_filenames_of_path(root_val / "Target")

    # dataset training
    dataset_train = SegmentationDataSet3(
        inputs=inputs_train,
        targets=targets_train,
        transform=transforms_training(),
        use_cache=False,
        pre_transform=None,
    )

    # dataset training
    dataset_val = SegmentationDataSet3(
        inputs=inputs_val,
        targets=targets_val,
        transform=transforms_training(),
        use_cache=False,
        pre_transform=None,
    )

    # dataloader training
    dataloader_training = DataLoader(
        dataset=dataset_train,
        batch_size=1,
        # batch_size of 2 won't work because the depth dimension is different between the 2 samples
        shuffle=True,
    )

    dataloader_validation = DataLoader(
        dataset=dataset_val,
        batch_size=1,
        # batch_size of 2 won't work because the depth dimension is different between the 2 samples
        shuffle=False,
    )

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Summary Writer TensorBoard
    writer = SummaryWriter("runs/vnet")

    # model
    model = VNet(model_params.elu, model_params.in_channels, model_params.classes).to(
        device
    )

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params.lr)

    # trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=DiceLoss(reduction="mean", to_onehot_y=True, sigmoid=True),
        optimizer=optimizer,
        training_dataloader=dataloader_training,
        validation_dataloader=dataloader_validation,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min"),
        epochs=training_params.epochs,
        epoch=0,
        writer=writer,
        notebook=False,
    )

    # start training
    trainer.run_trainer()

    # save the model
    if args.save_model:
        model_name = "brain_mri_vnet.pt"
        torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)
        bucket_output_name = "brain_mri_predictions"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_output_name)
        destination_blob_name = f"vnet_output/brain_mri_vnet"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename("./brain_mri_vnet.pt")

    print("Training done, check the results in GCS to get the weights file")


if __name__ == "__main__":
    main()
