import json
import os
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Fake-news-detector",
    version="1.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class NewsInput(BaseModel):
    text: str


SPACE_API_URL = os.getenv(
    "SPACE_API_URL",
    "https://ryancoder-fake-news-detector-api.hf.space",
)


def remote_predict(text: str):
    request_body = json.dumps({"data": [text]}).encode("utf-8")
    start_request = urllib.request.Request(
        f"{SPACE_API_URL}/gradio_api/call/predict",
        data=request_body,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(start_request, timeout=60) as response:
        start_payload = json.loads(response.read().decode("utf-8"))

    event_id = start_payload.get("event_id")
    if not event_id:
        raise HTTPException(status_code=502, detail="Space did not return an event id.")

    with urllib.request.urlopen(
        f"{SPACE_API_URL}/gradio_api/call/predict/{event_id}",
        timeout=120,
    ) as response:
        stream_text = response.read().decode("utf-8")

    for chunk in stream_text.split("\n\n"):
        lines = [line for line in chunk.splitlines() if line.strip()]
        if len(lines) >= 2 and lines[0].strip() == "event: complete":
            data_line = next((line for line in lines if line.startswith("data: ")), None)
            if not data_line:
                break
            payload = json.loads(data_line[6:])
            if not payload:
                break
            distilbert_result = payload[0]
            return {
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "distilbert": distilbert_result,
                "baseline": {
                    "prediction": "Unavailable in remote-only mode"
                },
            }

    raise HTTPException(status_code=502, detail="Invalid response received from Space.")

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

@app.post("/predict")
def predict_news(input: NewsInput):
    return remote_predict(input.text)
