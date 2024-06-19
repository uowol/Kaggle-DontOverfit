import os
from argparse import ArgumentParser

import mlflow 


# Set environment variables
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "kcw"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"


def download_model(args):
    # Download model artifact
    mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{args.run_id}/{args.model_name}", dst_path=".")
    

if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument("--run-id", dest="run_id", type=str, required=True)
    parser.add_argument("--model-name", dest="model_name", type=str, required=True, default="sk_model")
    args = parser.parse_args()
    
    download_model(args)
    