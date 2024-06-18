# crud_pydantic.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class CreateIn(BaseModel):
    name: str
    nickname: str


class CreateOut(BaseModel):
    status: str
    id: int 
    
    
# Create an instance of FastAPI
app = FastAPI()

# Create a fake database
FAKE_USER_DB = {}

# Fail response
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found")


# Create
@app.post("/users", response_model=CreateOut)
def create_item(user: CreateIn):
    FAKE_USER_DB[user.name] = user.nickname
    user_dict = user.model_dump()
    user_dict['status'] = 'success'
    user_dict['id'] = len(FAKE_USER_DB)
    return user_dict

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