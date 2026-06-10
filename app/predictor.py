import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

#paths
MODEL_DIR = "model/distilbert_model"
VECTORIZER_PATH = "model/vectorizer.pkl"
BASELINE_PATH = "model/baseline_model.pkl"

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