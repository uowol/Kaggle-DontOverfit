import datetime
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
import mlflow
from argparse import ArgumentParser


# Set mlflow environment variables
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"  # MinIO Artifact Store
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"     # MLflow Tracking Server
os.environ["AWS_ACCESS_KEY_ID"] = "kcw"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# Load the data
df = pd.read_csv('data/train20240614.csv')
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model
parser = ArgumentParser()
parser.add_argument("--run-id", type=str, dest="run_id", required=True)
parser.add_argument("--model-name", type=str, dest="model_name", default="sk_model")
args = parser.parse_args()

# Case 1. Load built-in model 
model_pipeline = mlflow.sklearn.load_model(f"runs:/{args.run_id}/{args.model_name}")
print(model_pipeline)
# Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])

# Case 2. Load pyfunc model 
model_pipeline = mlflow.pyfunc.load_model(f"runs:/{args.run_id}/{args.model_name}")
print(model_pipeline)
# mlflow.pyfunc.load_model:
#   artifact_path: sk_model
#   flavor: mlflow.sklearn
#   run_id: `RUN_ID`

# Predict
train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

# Evaluate
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

print("Load Train Accuracy :", train_acc)
print("Load Valid Accuracy :", valid_acc)