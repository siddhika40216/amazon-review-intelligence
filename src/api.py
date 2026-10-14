
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project aur model ka path
PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_PATH / "models" / "amazon_distilbert_model"

# Device select karo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saved model load karo
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(device)
model.eval()

app = FastAPI(title="Amazon Review Intelligence API")

# React frontend ko API access dena
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str


@app.get("/")
def home():
    return {"message": "Amazon Review Intelligence API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "DistilBERT"}


@app.post("/predict")
def predict(request: ReviewRequest):
    review = request.review.strip()

    if not review:
        return {"error": "Please enter a review"}

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    prediction = int(torch.argmax(probabilities).item())
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = float(probabilities[prediction].item() * 100)

    return {
        "review": review,
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "model": "DistilBERT"
    }
