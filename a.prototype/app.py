# 개요: Streamlit을 활용한 간단한 웹 어플리케이션 프로토타입
# 참고: https://docs.streamlit.io/library
import streamlit as st
import pandas as pd 
import requests
import json 
from datetime import datetime 

def predict(row: dict):
    # Request data to API server
    uri = 'http://localhost:8000/predict'
    headers = {'Content-Type': 'application/json'}
    request_body = { f"f{i}": row[str(i)] for i in range(300) }
    response = requests.post(uri, headers=headers, data=json.dumps(request_body))
    response_body = response.json()
    return response_body['target']

@st.cache_data
def convert_output(ids, y):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    result = pd.DataFrame({'id': ids, 'target': y})
    return result.to_csv(index=False).encode("utf-8")

# Title
st.title("Competition, Don't Overfit!")

# Description
st.write("Upload a csv file to predict target values.")

# Upload csv file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If file is uploaded
if uploaded_file is not None:
    # Read csv file
    df = pd.read_csv(uploaded_file)
    
    ids = df['id']
    X = df.drop(columns=['id'])
    with st.spinner(text="In progress..."):
        y = X.apply(predict, axis=1)
    st.success(f"Done! Saved as 'submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv'.")
    
    st.download_button(
        label = "Download CSV",
        data = convert_output(ids, y),
        file_name = f"submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        mime = "text/csv"
    )
    
    # DEBUG: Save as csv file on local
    y.to_csv(f"submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv", index=False)
