from kfp.v2.dsl import Dataset, Input, Output, Artifact, component


@component(
    base_image="gcr.io/ataxia-rd-dev-358f/preprocess_dataset_unet3d:latest",
    output_component_file="./pipeline/preprocess_dataset.yaml",
)
def preprocess_dataset(
    data_config: str,
    training_dataloader: Output[Dataset],
    validation_dataloader: Output[Dataset],
):
    import os
    import pickle
    import torch
    from torch.utils import data
    from torch.utils.data import DataLoader

    from brain_mri_unet3d.dataset import BrainSegmentationDataset
    from brain_mri_unet3d.transformations import transforms
    from brain_mri_unet3d.utils import load_config, config_to_object

    data_params_dict = load_config(data_config)
    data_params = config_to_object(data_params_dict)

    train_val_images_dir = data_params.train_val_images_dir

    # dataset training
    dataset_train = BrainSegmentationDataset(
        images_dir=train_val_images_dir,
        subset="train",
        image_size=data_params.image_size,
        transform=transforms(scale=0.05, angle=15, flip_prob=0.5),
        validation_cases=data_params.validation_cases,
    )

    # dataset validation
    dataset_valid = BrainSegmentationDataset(
        images_dir=train_val_images_dir,
        subset="validation",
        image_size=data_params.image_size,
        transform=transforms(scale=0.05, angle=15, flip_prob=0.5),
        validation_cases=data_params.validation_cases,
    )

    # dataloader training
    dataloader_training = DataLoader(
        dataset=dataset_train,
        batch_size=data_params.batch_size,
        shuffle=data_params.shuffle,
    )

    # dataloader validation
    dataloader_validation = DataLoader(
        dataset=dataset_valid,
        batch_size=data_params.batch_size,
        shuffle=data_params.shuffle,
    )

    torch.save(dataloader_training, training_dataloader.path + ".pkl")
    torch.save(dataloader_validation, validation_dataloader.path + ".pkl")
