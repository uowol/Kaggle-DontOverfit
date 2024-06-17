import datetime
import pandas as pd 
import joblib
import psycopg2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import os
import mlflow
from argparse import ArgumentParser

# import logging
# logging.getLogger("mlflow").setLevel(logging.DEBUG)


# Set mlflow environment variables
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"  # MinIO Artifact Store
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"     # MLflow Tracking Server
os.environ["AWS_ACCESS_KEY_ID"] = "kcw"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

mlflow.set_tracking_uri("http://localhost:5000")

# Connect to the database
db_connect = psycopg2.connect(host='172.19.75.88', database='kaggle', user='kcw', password='sk1346')

# Load the data
df = pd.read_sql('SELECT * FROM train', db_connect)
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])
model_pipeline.fit(X_train, y_train)

# Predict
train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

# Evaluate
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# Save the model
parser = ArgumentParser()
parser.add_argument("--model-name", type=str, dest="model_name", default="sk_model")
args = parser.parse_args()

# Set experiment name
mlflow.set_experiment("new-exp")

# Set input and output signature
signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
input_sample = X_train.iloc[:10]
# inputs: 
#   ['sepal_length': double, 'sepal_width': double, 'petal_length': double, 'petal_width': double]
# outputs: 
#   [Tensor('int64', (-1,))]

with mlflow.start_run():
    # 모델의 결과 metrics 를 Python 의 dictionary 형태로 입력해 생성된 run 에 저장합니다.
    mlflow.log_metrics({"train_acc": train_acc, "valid_acc": valid_acc})
    mlflow.sklearn.log_model(           # sklearn 모델은 mlflow.sklearn을 사용하여 간편하게 업로드가 가능합니다.
        sk_model=model_pipeline,        # 저장할 모델을 지정합니다.
        artifact_path=args.model_name,  # 모델을 저장할 경로를 지정합니다. 모델은 인자로 받은 model_name으로 저장됩니다.
        # registered_model_name=args.model_name,  # mlflow 서버에 등록할 모델의 이름을 지정합니다. 없으면 자동으로 생성됩니다.
        signature=signature,            # input/output signature를 지정합니다.
        input_example=input_sample      # input_example을 지정하면 추후에 추론할 때 사용할 수 있습니다.
    )

# Save the data for validation after training
df.to_csv(f'data/train{datetime.datetime.now().strftime("%Y%m%d")}.csv', index=False)