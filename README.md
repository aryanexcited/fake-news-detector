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

### Windows PowerShell

```powershell
git clone https://github.com/aryanexcited/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
$env:HF_MODEL_REPO="your-username/fake-news-detector-models"
python scripts/download_models.py
uvicorn app.main:app --reload
```

### Linux / macOS

```bash
git clone https://github.com/aryanexcited/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
export HF_MODEL_REPO=your-username/fake-news-detector-models
python scripts/download_models.py
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/ for the frontend or http://127.0.0.1:8000/docs for the API docs.

## API Endpoints

- `GET /` — Frontend page
- `POST /predict` — Takes news text, returns prediction + confidence score

## Free Render Deployment

This project is set up to use a free deployment flow where large model files are hosted outside GitHub and downloaded at startup.

### 1. Upload model assets to Hugging Face

Create a public Hugging Face model repository, for example:

`your-username/fake-news-detector-models`

Upload these files with this exact layout:

```text
distilbert_model/config.json
distilbert_model/model.safetensors
distilbert_model/tokenizer.json
distilbert_model/tokenizer_config.json
baseline_model.pkl
vectorizer.pkl
```

### 2. Create the Render web service

- Connect this GitHub repo to Render
- Create a `Web Service`
- Use these settings:

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
python scripts/download_models.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. Add Render environment variables

Set these environment variables in Render:

```text
HF_MODEL_REPO=your-username/fake-news-detector-models
MODEL_ROOT=/tmp/models
MODEL_DIR=/tmp/models/distilbert_model
BASELINE_PATH=/tmp/models/baseline_model.pkl
VECTORIZER_PATH=/tmp/models/vectorizer.pkl
```

### 4. Important note about the free plan

This setup uses `/tmp/models`, which is temporary storage. On the free path, Render may download the model files again after a restart or a fresh deploy. That is normal for this zero-cost setup.

## Dataset
[WELFake / fake_or_real_news](https://github.com/lutzhamel/fake-news) — 6,335 news articles, balanced classes.
