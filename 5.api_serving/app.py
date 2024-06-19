import mlflow
import pandas as pd
from fastapi import FastAPI
from schemas import PredIn, PredOut


def get_model():
    return mlflow.sklearn.load_model(model_uri="./sk_model") # 다운받은 모델 Artifact를 활용합니다. 


# Load the model
MODEL = get_model()

# Create a FastAPI instance
app = FastAPI()


@app.post("/predict", response_model=PredOut)
def predict(data: PredIn) -> PredOut:
    input_df = pd.DataFrame([data.dict()])
    pred = MODEL.predict(input_df).item()
    return PredOut(target=pred)



