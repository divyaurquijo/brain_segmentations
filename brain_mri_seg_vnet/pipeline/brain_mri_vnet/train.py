from kfp.v2.dsl import Dataset, Input, Output, Artifact, Model, component
from typing import NamedTuple


@component(
    base_image="gcr.io/ataxia-rd-dev-358f/train_vnet:latest",
    output_component_file="./pipeline/train.yaml",
)
def train(
    train_config: str,
    model_config: str,
    training_dataloader: Input[Dataset],
    validation_dataloader: Input[Dataset],
    model_weights: Output[Model],
):

    from monai.losses.dice import DiceLoss
    import numpy as np
    import torch

    from brain_mri_vnet.model import VNet
    from brain_mri_vnet.trainer import Trainer
    from brain_mri_vnet.utils import load_config, config_to_object

    train_params_dict = load_config(train_config)
    train_params = config_to_object(train_params_dict)

    model_params_dict = load_config(model_config)
    model_params = config_to_object(model_params_dict)

    dataloader_training = torch.load(training_dataloader.path + ".pkl")
    dataloader_validation = torch.load(validation_dataloader.path + ".pkl")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vnet = VNet(model_params.elu, model_params.in_channels, model_params.classes).to(
        device
    )

    dice_loss = DiceLoss(reduction="mean", to_onehot_y=True, sigmoid=True)
    optimizer = torch.optim.Adam(vnet.parameters(), lr=train_params.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # trainer
    trainer = Trainer(
        model=vnet,
        device=device,
        criterion=dice_loss,
        optimizer=optimizer,
        training_dataloader=dataloader_training,
        validation_dataloader=dataloader_validation,
        lr_scheduler=lr_scheduler,
        epochs=train_params.epochs,
        epoch=train_params.epoch,
        writer=train_params.writer,
        notebook=train_params.notebook,
    )

    trainer.run_trainer()

    torch.save(vnet.state_dict(), model_weights.path + ".pt")
