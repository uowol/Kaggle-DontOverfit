# Base image
FROM amd64/python:3.9-slim

# Excute command when build image
RUN apt-get update && apt-get install -y \
    postgresql-client \ 
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/app 

# Excute command
RUN pip install -U pip && \
    pip install psycopg2-binary && \
    pip install pandas && \
    mkdir data

# Copy file into the image
COPY data_generator.py data_generator.py
COPY dataset/train.csv data/train.csv

# Excute command when run container
ENTRYPOINT [ "python", "data_generator.py", "--db-host" ]

# Deliver arguments to ENTRYPOINT when run container
CMD [ "localhost" ]
