FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow, PostgreSQL and AWS SDK
RUN pip install -U pip && \
    pip install mlflow psycopg2-binary boto3==1.26.8

# Install MinIO client
RUN cd /tmp && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/bin/mc 