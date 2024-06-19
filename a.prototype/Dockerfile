FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install streamlit pandas requests mlflow scikit-learn boto3==1.26.8

COPY app.py app.py 

CMD ["streamlit", "run", "app.py"]