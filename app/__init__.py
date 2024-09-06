from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from app.router.classify_router import classify_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

app.include_router(classify_router, prefix = "/api/v1/classify")

