from fastapi import FastAPI
from fastapi import UploadFile
from cnn_dep import predict


app=FastAPI()


@app.post("/predict")

async def classify(
    file:UploadFile
):

    return predict(
        file.file
    )