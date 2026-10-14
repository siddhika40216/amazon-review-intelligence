
from pathlib import Path
import time
import joblib

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "test_reviews.csv"
MODELS_PATH = ROOT / "models"
REPORTS_PATH = ROOT / "reports"

REPORTS_PATH.mkdir(exist_ok=True)

# Load the same test dataset for all three models
test_data = pd.read_csv(DATA_PATH)

texts = test_data["text"].fillna("").astype(str).tolist()
y_true = test_data["label"].astype(int).to_numpy()

print(f"Test reviews: {len(texts)}")
print("\nStarting model evaluation...\n")

results = []


def evaluate_model(model_name, predictions, start_time):
    """Calculate and store evaluation metrics."""
    predictions = np.asarray(predictions).astype(int)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "Evaluation Time (seconds)": round(time.time() - start_time, 2)
    })

    print(f"{model_name} evaluation completed.")


# --------------------------------------------------
# 1. LOGISTIC REGRESSION
# --------------------------------------------------
print("1/3 Evaluating Logistic Regression...")
start = time.time()

lr_model = joblib.load(MODELS_PATH / "logistic_regression.pkl")
lr_vectorizer = joblib.load(MODELS_PATH / "tfidf_vectorizer.pkl")

lr_predictions = []

# Predict in batches to keep memory usage manageable
for i in range(0, len(texts), 500):
    batch_texts = texts[i:i + 500]
    batch_features = lr_vectorizer.transform(batch_texts)
    batch_predictions = lr_model.predict(batch_features)
    lr_predictions.extend(batch_predictions)

evaluate_model("Logistic Regression", lr_predictions, start)


# --------------------------------------------------
# 2. ARTIFICIAL NEURAL NETWORK (ANN)
# --------------------------------------------------
print("\n2/3 Evaluating ANN...")
start = time.time()

from tensorflow.keras.models import load_model

ann_model = load_model(MODELS_PATH / "ann_model.keras")

with open(MODELS_PATH / "ann_tfidf_vectorizer.pkl", "rb") as file:
    ann_vectorizer = joblib.load(
    MODELS_PATH / "ann_tfidf_vectorizer.pkl"
)

ann_predictions = []

for i in range(0, len(texts), 200):
    batch_texts = texts[i:i + 200]
    batch_features = ann_vectorizer.transform(batch_texts).toarray()

    batch_output = np.asarray(
        ann_model.predict(batch_features, verbose=0)
    )

    # Support either sigmoid output (one value) or two-class output
    if batch_output.ndim == 1:
        batch_pred = (batch_output >= 0.5).astype(int)
    elif batch_output.shape[1] == 1:
        batch_pred = (batch_output[:, 0] >= 0.5).astype(int)
    else:
        batch_pred = np.argmax(batch_output, axis=1)

    ann_predictions.extend(batch_pred)

evaluate_model("ANN", ann_predictions, start)


# --------------------------------------------------
# 3. DISTILBERT
# --------------------------------------------------
print("\n3/3 Evaluating DistilBERT...")
start = time.time()


distilbert_path = (MODELS_PATH / "amazon_distilbert_model").resolve()

print("DistilBERT folder:", distilbert_path)
print("Folder exists:", distilbert_path.is_dir())
print("Config exists:", (distilbert_path / "config.json").is_file())

if not (distilbert_path / "config.json").is_file():
    raise FileNotFoundError(
        f"config.json not found inside: {distilbert_path}. "
        "Check whether the model files are directly inside this folder."
    )

tokenizer = AutoTokenizer.from_pretrained(
    str(distilbert_path),
    local_files_only=True
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    str(distilbert_path),
    local_files_only=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

print(f"Device: {device}")

bert_predictions = []
batch_size = 16

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = bert_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

        bert_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

evaluate_model("DistilBERT", bert_predictions, start)


# --------------------------------------------------
# SAVE AND DISPLAY COMPARISON
# --------------------------------------------------
comparison = pd.DataFrame(results)

metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
comparison[metric_columns] = comparison[metric_columns].round(4)

output_path = REPORTS_PATH / "model_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)
print(comparison.to_string(index=False))

print(f"\nComparison saved to: {output_path}")
