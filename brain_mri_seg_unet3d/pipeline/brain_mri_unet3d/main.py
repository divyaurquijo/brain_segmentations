"""Main script to run each step of our pipeline"""

import yaml

import google.cloud.aiplatform as aip
from kfp.v2 import dsl
from kfp.v2 import compiler
from munch import DefaultMunch

from preprocess_dataset import preprocess_dataset
from train import train
from evaluation import evaluation
from deploy import deploy

# Load parameters
config_file = "./config.yaml"
with open(config_file) as fh:
    params = yaml.load(fh, Loader=yaml.FullLoader)
params = DefaultMunch.fromDict(params)


@dsl.pipeline(
    name="brain-mri-unet3d",
    description="Pipeline for Brain MRI UNET 3D",
    pipeline_root="gs://unet3d_pipeline",
)
def pipeline(
    data_config_path: str = params.data_config,
    train_config_path: str = params.train_config,
    model_config_path: str = params.model_config,
    deploy_config_path: str = params.deploy_config,
    evaluation_config_path: str = params.evaluation_config,
):
    op_preprocess_dataset = preprocess_dataset(data_config_path)
    op_train = (
        train(
            train_config_path,
            model_config_path,
            op_preprocess_dataset.outputs["training_dataloader"],
            op_preprocess_dataset.outputs["validation_dataloader"],
        )
        .add_node_selector_constraint(
            "cloud.google.com/gke-accelerator", "NVIDIA_TESLA_K80"
        )
        .set_memory_request("128G")
        .set_cpu_request("32")
    )
    op_evaluation = (
        evaluation(
            model_config_path, evaluation_config_path, op_train.outputs["model_weights"]
        )
        .add_node_selector_constraint(
            "cloud.google.com/gke-accelerator", "NVIDIA_TESLA_K80"
        )
        .set_memory_request("128G")
        .set_cpu_request("32")
    )

    op_deploy = (
        deploy(op_train.outputs["model_weights"], deploy_config_path)
        .add_node_selector_constraint(
            "cloud.google.com/gke-accelerator", "NVIDIA_TESLA_K80"
        )
        .set_memory_request("128G")
        .set_cpu_request("32")
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="./pipeline/unet3d_pipeline.json"
    )

    job = aip.PipelineJob(
        display_name="brain-mri-unet3d",
        template_path="./pipeline/unet3d_pipeline.json",
        pipeline_root="gs://unet3d_pipeline",
        location="europe-west1",
    )

    job.run(
        service_account="em52-sa-rd-ataxia-dev@ataxia-rd-dev-358f.iam.gserviceaccount.com"
    )
