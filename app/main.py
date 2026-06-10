from fastapi import FastAPI
from pydantic import BaseModel
from app.predictor import predict

app = FastAPI(
    title="Fake-news-detector",
    version="1.0"
)

class NewsInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake news detector API is running"}

@app.post("/predict")
def predict_news(input: NewsInput):
    result = predict(input.text)
    return result
