from kfp.v2.dsl import Dataset, Input, Output, Artifact, component


@component(
    base_image="gcr.io/ataxia-rd-dev-358f/preprocess_dataset_vnet:latest",
    output_component_file="./pipeline/preprocess_dataset.yaml",
)
def preprocess_dataset(
    data_config: str,
    training_dataloader: Output[Dataset],
    validation_dataloader: Output[Dataset],
    test_dataloader: Output[Dataset],
):
    import pathlib
    from pathlib import Path
    import torch
    from torch.utils import data
    from torch.utils.data import DataLoader

    from brain_mri_vnet.dataset import SegmentationDataSet3
    from brain_mri_vnet.transformations import transforms_training, transforms_testing
    from brain_mri_vnet.utils import (
        get_filenames_of_path,
        load_config,
        config_to_object,
    )

    data_params_dict = load_config(data_config)
    data_params = config_to_object(data_params_dict)

    root_train = pathlib.Path.cwd() / data_params.train_images_dir
    root_val = pathlib.Path.cwd() / data_params.val_images_dir
    root_test = pathlib.Path.cwd() / data_params.test_images_dir

    # input and target files
    inputs_train = get_filenames_of_path(root_train / "Input")
    targets_train = get_filenames_of_path(root_train / "Target")

    inputs_val = get_filenames_of_path(root_val / "Input")
    targets_val = get_filenames_of_path(root_val / "Target")

    inputs_test = get_filenames_of_path(root_test / "Input")
    targets_test = get_filenames_of_path(root_test / "Target")

    # dataset training
    dataset_train = SegmentationDataSet3(
        inputs=inputs_train,
        targets=targets_train,
        transform=transforms_training(),
        use_cache=data_params.use_cache,
        pre_transform=data_params.pre_transform,
    )

    # dataset validation
    dataset_val = SegmentationDataSet3(
        inputs=inputs_val,
        targets=targets_val,
        transform=transforms_training(),
        use_cache=data_params.use_cache,
        pre_transform=data_params.pre_transform,
    )

    # dataset test
    dataset_test = SegmentationDataSet3(
        inputs=inputs_test,
        targets=targets_test,
        transform=transforms_testing(),
        use_cache=data_params.use_cache,
        pre_transform=data_params.pre_transform,
    )

    # dataloader training
    dataloader_training = DataLoader(
        dataset=dataset_train,
        batch_size=data_params.batch_size,
        shuffle=data_params.shuffle_train,
    )

    dataloader_validation = DataLoader(
        dataset=dataset_val,
        batch_size=data_params.batch_size,
        shuffle=data_params.shuffle_val,
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=data_params.batch_size,
        shuffle=data_params.shuffle_val,
    )

    torch.save(dataloader_training, training_dataloader.path + ".pkl")
    torch.save(dataloader_validation, validation_dataloader.path + ".pkl")
    torch.save(dataloader_test, test_dataloader.path + ".pkl")
