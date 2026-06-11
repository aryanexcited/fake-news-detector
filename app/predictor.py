import os
import pickle
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

#path configs
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/models/distilbert_model")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "/tmp/models/vectorizer.pkl")
BASELINE_PATH = os.getenv("BASELINE_PATH", "/tmp/models/baseline_model.pkl")

#Baseline model loading
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
with open(BASELINE_PATH, "rb") as f:
    basline_model = pickle.load(f)

#DistilBert model laoding
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
bert_model.eval()

def predict(text: str):
    #prediction using distilbert model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    bert_label = "FAKE" if probs[1] > probs[0] else "REAL"
    bert_confidence = round(max(probs)*100, 2)

    #prediction using baseline model
    tfidf_vec = vectorizer.transform([text])
    baseline_label = "FAKE" if basline_model.predict(tfidf_vec)[0] == 1 else "REAL"

    return{
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "distilbert": {
            "prediction": bert_label,
            "confidence": f"{bert_confidence}%"
        },
        "baseline": {
            "prediction": baseline_label
        }
    }
