import torch
import joblib
from transformers import DistilBertTokenizer
from fusion import fuse_predictions
import re

# Load models
text_model = torch.load("saved_text_model.pt")
url_model = joblib.load("saved_url_model.pkl")
image_model = torch.load("saved_image_model.pt")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def predict_text(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return text_model(**enc).item()

def predict_url(url):
    features = [
        len(url),
        url.count('.'),
        url.count('@'),
        url.count('-'),
        sum(c.isdigit() for c in url),
        int("https" in url),
        int(bool(re.search("login|verify|secure|update", url.lower())))
    ]
    return url_model.predict_proba([features])[0][1]
