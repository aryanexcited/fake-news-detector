import os
import sys
import urllib.request
from pathlib import Path


HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "/tmp/models"))
DISTILBERT_DIR = MODEL_ROOT / "distilbert_model"

MODEL_FILES = {
    "distilbert_model/config.json": DISTILBERT_DIR / "config.json",
    "distilbert_model/model.safetensors": DISTILBERT_DIR / "model.safetensors",
    "distilbert_model/tokenizer.json": DISTILBERT_DIR / "tokenizer.json",
    "distilbert_model/tokenizer_config.json": DISTILBERT_DIR / "tokenizer_config.json",
    "baseline_model.pkl": MODEL_ROOT / "baseline_model.pkl",
    "vectorizer.pkl": MODEL_ROOT / "vectorizer.pkl",
}


def download_file(remote_path: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"Already present: {local_path}")
        return

    if not HF_MODEL_REPO:
        raise RuntimeError(
            "HF_MODEL_REPO is not set. Example: username/fake-news-detector-models"
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/{remote_path}"
    print(f"Downloading {remote_path} from {url}")
    urllib.request.urlretrieve(url, local_path)


def main() -> int:
    try:
        for remote_path, local_path in MODEL_FILES.items():
            download_file(remote_path, local_path)
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1

    print("Model assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
