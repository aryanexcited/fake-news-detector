from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.predictor import predict

app = FastAPI(
    title="Fake-news-detector",
    version="1.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class NewsInput(BaseModel):
    text: str

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

@app.post("/predict")
def predict_news(input: NewsInput):
    result = predict(input.text)
    return result
