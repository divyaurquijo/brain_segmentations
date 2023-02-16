from kfp.v2.dsl import Input, Output, Artifact, Model, component


@component(
    base_image="gcr.io/ataxia-rd-dev-358f/deploy_unet3d:latest",
    output_component_file="./pipeline/deploy.yaml",
)
def deploy(
    model: Input[Model],
    deploy_config: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    import os
    import shutil

    from google.cloud import aiplatform
    from brain_mri_unet3d.utils import load_config, config_to_object

    deploy_params_dict = load_config(deploy_config)
    deploy_params = config_to_object(deploy_params_dict)

    # Copy model.pt file into folder
    destination_file = os.path.join(deploy_params.deploy_path, "brain_mri_unet3D.pt")
    shutil.copy(model.path + ".pt", destination_file)

    # Create .mar file with torch model archiver in this container and save it in gcs bucket
    os.chdir(deploy_params.deploy_path)
    os.system("make")
    os.chdir(deploy_params.root)

    # Initialize aiplatform
    aiplatform.init(project=deploy_params.project, location=deploy_params.region)

    # Upload model on Vertex AI Model Registry
    model_upload = aiplatform.Model.upload(
        location=deploy_params.region,
        display_name=deploy_params.display_name,
        serving_container_image_uri=deploy_params.serving_container_image_uri,
        serving_container_command=["/home/model-server/entrypoint.sh"],
        serving_container_health_route=deploy_params.serving_container_health_route,
        serving_container_predict_route=deploy_params.serving_container_predict_route,
        description="Instance Segmentation on Brain MRI with UNET 3D Pipeline",
        serving_container_ports=[deploy_params.serving_container_ports],
    )

    # Create Endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=deploy_params.endpoint_name,
        project=deploy_params.project,
        location=deploy_params.region,
    )

    # Deploy model on endpoint
    model_deploy = model_upload.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deploy_params.display_name,
        machine_type=deploy_params.machine_type,
        accelerator_count=deploy_params.accelerator_count,
        accelerator_type=deploy_params.accelerator_type,
        traffic_split={"0": 100},
        service_account=deploy_params.service_account,
    )

    # Save data to the output params
    vertex_model.uri = model_deploy.resource_name
