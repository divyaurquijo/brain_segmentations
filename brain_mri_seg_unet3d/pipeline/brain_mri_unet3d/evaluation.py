from kfp.v2.dsl import Dataset, Input, Output, Artifact, Model, component


@component(
    base_image="gcr.io/ataxia-rd-dev-358f/evaluation_unet3d:latest",
    output_component_file="./pipeline/evaluation.yaml",
)
def evaluation(
    model_config: str,
    evaluation_config: str,
    model_weights: Input[Model],
    segmentation: Output[Artifact],
):
    import os
    import torch

    from brain_mri_unet3d.model import UNet
    from brain_mri_unet3d.utils import (
        predict,
        preprocess,
        postprocess,
        preprocess_images,
        segmentation_pred,
        load_config,
        config_to_object,
    )

    # Model parameters
    model_params_dict = load_config(model_config)
    model_params = config_to_object(model_params_dict)

    # Evaluation parameters
    evaluation_params_dict = load_config(evaluation_config)
    evaluation_params = config_to_object(evaluation_params_dict)

    # Directory of test images
    test_images_dir = evaluation_params.test_images_dir
    dataloader_test = list(os.listdir(test_images_dir))

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

    unet3d_weights = torch.load(model_weights.path + ".pt", map_location=device)
    model.load_state_dict(unet3d_weights)

    for directory in dataloader_test:
        directory_path = os.path.join(test_images_dir, directory)
        inputs_test = os.path.join(directory_path, "Input")
        targets_test = os.path.join(directory_path, "Target")

        images_res, targets_res = preprocess_images(inputs_test, targets_test)
        # predict the segmentation maps
        output = [
            predict(img, model, preprocess, postprocess, device) for img in images_res
        ]

        # Create a segmentations array for predictions
        filepath_out_pred = segmentation.path + ".gif"
        segmentation_pred(
            directory, directory_path, filepath_out_pred, images_res, output
        )
