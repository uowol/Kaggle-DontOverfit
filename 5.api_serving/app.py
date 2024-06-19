import mlflow
import pandas as pd
from fastapi import FastAPI
from schemas import PredIn, PredOut
import os 


# Set environment variables
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "kcw"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"


def get_model():
    return mlflow.sklearn.load_model(model_uri="./sk_model")        # 다운받은 모델 Artifact를 활용합니다. 


# Load the model
MODEL = get_model()

# Create a FastAPI instance
app = FastAPI()


@app.post("/predict", response_model=PredOut)
def predict(data: PredIn) -> PredOut:
    input_df = pd.DataFrame([data.dict()])
    pred = MODEL.predict(input_df).item()
    return PredOut(target=pred)

# @app.post("/predict/model/{model_name}/run/{run_id}", response_model=PredOut)
# def predict(model_name: str, run_id: str, data: PredIn) -> PredOut:
#     global MODEL
#     MODEL = get_model(run_id=run_id, model_name=model_name)

#     input_df = pd.DataFrame([data.dict()])
#     pred = MODEL.predict(input_df).item()
#     return PredOut(target=pred)





