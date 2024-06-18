# path_param.py
from fastapi import FastAPI


# Create an instance of FastAPI
app = FastAPI()


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}