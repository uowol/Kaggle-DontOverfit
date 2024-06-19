# 개요: Streamlit을 활용한 간단한 웹 어플리케이션 프로토타입
# 참고: https://docs.streamlit.io/library
from datetime import datetime 
import pandas as pd 
import time
import os

import streamlit as st
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# Set mlflow environment variables
HOST = "172.19.75.88"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{HOST}:9000"  # MinIO Artifact Store
os.environ["MLFLOW_TRACKING_URI"] = f"http://{HOST}:5000"     # MLflow Tracking Server
os.environ["AWS_ACCESS_KEY_ID"] = "kcw"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# Title
st.title("Competition, Don't Overfit!")

# Subtitle
st.subheader("Train")

# Description
st.write("Upload a csv file to train a model.")

# Upload csv file
train_uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="train_uploaded_file")

# Input model name
train_model_name = st.text_input("Model name", "sk_model", key="train_model_name")

# If train button is clicked
if st.button("Train") and train_uploaded_file is not None:
    # Read csv file
    df = pd.read_csv(train_uploaded_file)
    
    X = df.drop(columns=['id', 'target'])
    X.columns = [f"f{i}" for i in range(300)]
    y = df['target']    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC())
    ])
    
    with st.spinner(text="In progress..."):
        start_time = time.time()
        model_pipeline.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
    st.success(f"Done! Elapsed time: {elapsed_time:.2f} seconds.")

    # Predict
    train_pred = model_pipeline.predict(X_train)
    valid_pred = model_pipeline.predict(X_valid)

    # Evaluate
    train_acc = accuracy_score(y_train, train_pred)
    valid_acc = accuracy_score(y_valid, valid_pred)
    st.write("Train Accuracy :", train_acc)
    st.write("Valid Accuracy :", valid_acc)

    # Save the model
    mlflow.set_experiment("new-exp")
    
    # Set input and output signature
    signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
    input_sample = X_train.iloc[:10]

    with mlflow.start_run():
        # 모델의 결과 metrics 를 Python 의 dictionary 형태로 입력해 생성된 run 에 저장합니다.
        mlflow.log_metrics({"train_acc": train_acc, "valid_acc": valid_acc})
        mlflow.sklearn.log_model(           # sklearn 모델은 mlflow.sklearn을 사용하여 간편하게 업로드가 가능합니다.
            sk_model=model_pipeline,        # 저장할 모델을 지정합니다.
            artifact_path=train_model_name, # 모델을 저장할 경로를 지정합니다. 모델은 인자로 받은 train_model_name으로 저장됩니다.
            signature=signature,            # input/output signature를 지정합니다.
            input_example=input_sample      # input_example을 지정하면 추후에 추론할 때 사용할 수 있습니다.
        )
        print("*"*100)
        st.success(f"Model saved as '{train_model_name}'. Run ID: {mlflow.active_run().info.run_id}")


def predict(row: dict, run_id: str, model_name: str):
    # Load the model
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")
    input_df = pd.DataFrame([{ f"f{i}": row[str(i)] for i in range(300) }])
    return model.predict(input_df).item()

@st.cache_data
def convert_output(ids, y):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    result = pd.DataFrame({'id': ids, 'target': y})
    return result.to_csv(index=False).encode("utf-8")


# Subtitle
st.subheader("Inference")

# Description
st.write("Upload a csv file, model name, and run id to predict.")
st.write("Reference Link: [MLflow Tracking UI](http://localhost:5000)")

# Upload csv file
test_uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="test_uploaded_file")

# Input model name
test_model_name = st.text_input("Model name", "sk_model", key="test_model_name")

# Input run id
run_id = st.text_input("Run ID", "")

# If file, model name, and run id are provided
if st.button("Predict") and test_uploaded_file is not None and test_model_name and run_id:
    # Read csv file
    df = pd.read_csv(test_uploaded_file)

    ids = df['id']
    X = df.drop(columns=['id'])
    with st.spinner(text="In progress..."):
        start_time = time.time()
        y = X.apply(lambda x: predict(x, run_id, test_model_name), axis=1)
        end_time = time.time()
        elapsed_time = end_time - start_time
    st.success(f"Done! Saved as 'submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv'. \
        Elapsed time: {elapsed_time:.2f} seconds.")

    result = convert_output(ids, y)
    st.download_button(
        label = "Download CSV",
        data = result,
        file_name = f"submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        mime = "text/csv"
    )

    # DEBUG: Save as csv file on local
    pd.DataFrame({'id': ids, 'target': y}).to_csv(
        f"submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv", index=False)
