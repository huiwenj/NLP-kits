from app import router

@router.get('/simple_and_hard_word')
async def classify_simple_and_hard_word():
    return {"simple_word": "simple", "hard_word": "hard"}


