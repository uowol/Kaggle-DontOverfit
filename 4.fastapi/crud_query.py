# crud_path.py
from fastapi import FastAPI, HTTPException

# Create an instance of FastAPI
app = FastAPI()

# User database
FAKE_USER_DB = {}

# Fail response
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found")

# Create
@app.post("/users")
def create_item(name: str, nickname: str):
    FAKE_USER_DB[name] = nickname
    return {"status": 'success'}

# Read
@app.get("/users")
def read_item(name: str):
    if name not in FAKE_USER_DB:
        raise NAME_NOT_FOUND    # return 400 error with detail message
    return {"nickname": FAKE_USER_DB[name]}

# Update
@app.put("/users")
def update_item(name: str, nickname: str):
    if name not in FAKE_USER_DB:
        raise NAME_NOT_FOUND
    FAKE_USER_DB[name] = nickname
    return {"status": 'success'}

# Delete
@app.delete("/users")
def delete_item(name: str):
    if name not in FAKE_USER_DB:
        raise NAME_NOT_FOUND
    del FAKE_USER_DB[name]
    return {"status": 'success'}