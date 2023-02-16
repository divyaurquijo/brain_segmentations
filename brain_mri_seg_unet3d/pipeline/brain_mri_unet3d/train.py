from kfp.v2.dsl import Dataset, Input, Output, Artifact, Model, component
from typing import NamedTuple


@component(
    base_image="gcr.io/ataxia-rd-dev-358f/train_unet3d:latest",
    output_component_file="./pipeline/train.yaml",
)
def train(
    train_config: str,
    model_config: str,
    training_dataloader: Input[Dataset],
    validation_dataloader: Input[Dataset],
    model_weights: Output[Model],
):
    import os
    import numpy as np
    import torch

    from brain_mri_unet3d.model import UNet
    from brain_mri_unet3d.trainer import Trainer
    from brain_mri_unet3d.utils import load_config, config_to_object

    train_params_dict = load_config(train_config)
    train_params = config_to_object(train_params_dict)

    model_params_dict = load_config(model_config)
    model_params = config_to_object(model_params_dict)

    dataloader_training = torch.load(training_dataloader.path + ".pkl")
    dataloader_validation = torch.load(validation_dataloader.path + ".pkl")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_dataloader=dataloader_training,
        validation_dataloader=None,
        lr_scheduler=None,
        epochs=train_params.epochs,
        epoch=train_params.epoch,
        writer=train_params.writer,
        notebook=train_params.notebook,
    )

    trainer.run_trainer()

    torch.save(model.state_dict(), model_weights.path + ".pt")
