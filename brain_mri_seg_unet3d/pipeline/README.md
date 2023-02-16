# Brain Tumor MRI - Instance segmentation with UNET 3D PIPELINE

 This example shows how to do instance segmentation on Brain MRI to detect tumors with the UNET 3D model on VERTEX AI PIPELINE.

 ## Pipeline

 The pipeline is composed of 4 components :  preprocess_dataset, train, evaluation and deploy.
 To run the pipeline you can run `python3 main.py` and check the progress on Vertex AI Pipeline.

 This pipeline preprocess the Dataset store in the folder Data3D like before. Than it trains it and evaluate it. You can see the segmentation of the evaluation in GCS Bucket. For the deployment we used torchserve and when it is finished you can try a prediction on the Vertex AI Endpoint.