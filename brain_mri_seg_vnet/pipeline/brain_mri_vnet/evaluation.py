from kfp.v2.dsl import Dataset, Input, Output, Artifact, Model, component


@component(
    base_image="gcr.io/ataxia-rd-dev-358f/evaluation_vnet:latest",
    output_component_file="./pipeline/evaluation.yaml",
)
def evaluation(
    model_config: str,
    test_dataloader: Input[Dataset],
    model_weights: Input[Model],
    segmentation: Output[Artifact],
):
    import torch

    from brain_mri_vnet.model import VNet
    from brain_mri_vnet.utils import (
        postprocess,
        segmentation_pred,
        load_config,
        config_to_object,
    )

    model_params_dict = load_config(model_config)
    model_params = config_to_object(model_params_dict)

    dataloader_test = torch.load(test_dataloader.path + ".pkl")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = VNet(model_params.elu, model_params.in_channels, model_params.classes).to(
        device
    )
    vnet_weights = torch.load(model_weights.path + ".pt", map_location=device)
    model.load_state_dict(vnet_weights)

    model.eval()
    for indice, (image, mask) in enumerate(dataloader_test):
        input_image, input_mask = image.to(device), mask.to(device)
        with torch.no_grad():
            output = model(input_image)  # send through model/network
            output = postprocess(output)  # postprocess the prediction
            filepath_out_pred = segmentation.path + ".gif"
            segmentation_pred_path = segmentation_pred(
                input_image, output, filepath_out_pred
            )
