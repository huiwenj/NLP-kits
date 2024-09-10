from fastapi import APIRouter

from app.models.response import R
from app.service.classify import ClassifyService

classify_router = APIRouter()
service = ClassifyService()


@classify_router.get("/train", tags=['classify'], description="Train the model", response_description="Training result")
async def train():
    try:
        result = service.train()
        return R.success(result)
    except Exception as e:
        return R.error(str(e))


@classify_router.get("/predict", tags=['classify'],
                     description="Predict the word is simple or hard",
                     response_description="Prediction result: 0 means simple & 1 means hard",
                     )
async def predict(word: str):
    return service.predict(word)
