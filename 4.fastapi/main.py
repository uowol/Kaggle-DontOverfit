# main.py
from fastapi import FastAPI


# Create an instance of FastAPI
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}