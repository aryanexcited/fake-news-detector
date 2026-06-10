# Fake News Detector

An end-to-end NLP pipeline that classifies news articles as REAL or FAKE using two models:
- **Baseline**: TF-IDF + Logistic Regression (90.1% accuracy)
- **Fine-tuned DistilBERT**: Transformer-based classifier (97.4% accuracy)

## Tech Stack
Python · FastAPI · HuggingFace Transformers · scikit-learn · Docker

## Results

| Model | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 90.1% |
| DistilBERT (fine-tuned) | 97.4% |

## Project Structure
fake-news-detector/
├── app/
│   ├── main.py        # FastAPI endpoints
│   └── predictor.py   # Model inference logic
├── model/             # Saved model files (not tracked in git)
├── Dockerfile
└── requirements.txt

## Run Locally

```bash
git clone https://github.com/aryanexcited/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs to test the API interactively.

## API Endpoints

- `GET /` — Health check
- `POST /predict` — Takes news text, returns prediction + confidence score

## Dataset
[WELFake / fake_or_real_news](https://github.com/lutzhamel/fake-news) — 6,335 news articles, balanced classes.