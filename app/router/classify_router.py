from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.service.classify import ClassifyService

classify_router = APIRouter()
service = ClassifyService()


@classify_router.get("/train", tags=['classify'], description="Train the model", response_description="Training result")
async def train():
    try:
        result = service.train()
        return JSONResponse(status_code=200, content={"message": "Training is done!", "data": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Something went wrong!, error: {str(e)}"})


@classify_router.get("/predict", tags=['classify'],
                     description="Predict the word is simple or hard",
                     response_description="Prediction result: 0 means simple & 1 means hard",
                     )
async def predict(word: str):
    return service.predict(word)
