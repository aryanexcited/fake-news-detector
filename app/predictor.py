import os
import pickle
import urllib.request
from pathlib import Path


HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "RYancoder/fake-news-detector-models")
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "/tmp/models"))
VECTORIZER_PATH = MODEL_ROOT / "vectorizer.pkl"
BASELINE_PATH = MODEL_ROOT / "baseline_model.pkl"


def _download_if_missing(remote_name: str, local_path: Path) -> None:
    if local_path.exists():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/{remote_name}"
    urllib.request.urlretrieve(url, local_path)


_download_if_missing("vectorizer.pkl", VECTORIZER_PATH)
_download_if_missing("baseline_model.pkl", BASELINE_PATH)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open(BASELINE_PATH, "rb") as baseline_file:
    baseline_model = pickle.load(baseline_file)


def predict_baseline(text: str) -> dict:
    tfidf_vec = vectorizer.transform([text])
    baseline_label = "FAKE" if baseline_model.predict(tfidf_vec)[0] == 1 else "REAL"
    return {"prediction": baseline_label}
