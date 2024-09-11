from fastapi import APIRouter

from app.models.response import R
from app.service.classify import WordClassifyService, CityClassifyService

classify_router = APIRouter()
word_service = WordClassifyService()
city_service = CityClassifyService()


@classify_router.get("/word/train", tags=['classify'], description="Train the model", response_description="Training result")
def train():
    try:
        result = word_service.train()
        return R.success(result)
    except Exception as e:
        return R.error(str(e))


@classify_router.get("/word/predict", tags=['classify'],
                     description="Predict the word is simple or hard",
                     response_description="Prediction result: 0 means simple & 1 means hard",
                     )
def predict(word: str):
    return word_service.predict(word)


@classify_router.get("/city/train", tags=['classify'])
def train_city():
    try:
        result = city_service.train()
        return R.success(result)
    except Exception as e:
        return R.error(str(e))


@classify_router.get("/city/predict", tags=['classify'])
def predict_city(city: str):
    return city_service.predict(city)


@classify_router.get("/city/list", tags=['classify'])
def list_city():
    return city_service.list_city()