from fastapi import APIRouter
from starlette.responses import JSONResponse

from app import router
from app.service.classify import ClassifyService

classify_router = APIRouter()

service = ClassifyService()



@classify_router.get('/simple-and-hard-word', tags=['classify'])
async def classify_simple_and_hard_word():
    return {"simple_word": "simple", "hard_word": "hard"}


@classify_router.get("/train", tags=['classify'])

async def train():
    try:
        result = service.train()
        return JSONResponse(status_code=200, content={"message": "Training is done!"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Something went wrong!, error: {str(e)}"})
