
import pandas as pd
import time
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load dataset
print("Loading dataset...")

train_df = pd.read_csv(DATA_DIR / "train_reviews.csv")
test_df = pd.read_csv(DATA_DIR / "test_reviews.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_test = test_df["text"].fillna("")
y_test = test_df["label"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Convert text into numerical features
print("\nConverting text into TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

# Train model
model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    solver="liblinear"
)

print("\nTraining Logistic Regression...")
start_time = time.time()

model.fit(X_train_tfidf, y_train)

training_time = time.time() - start_time

# Evaluate model
print(f"\nTraining time: {training_time:.2f} seconds")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, MODEL_DIR / "logistic_regression.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

# Save evaluation results
with open(REPORT_DIR / "ml_results.txt", "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Training time: {training_time:.2f} seconds\n")
    file.write(classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    ))

print("\nModel and vectorizer saved in models/")
print("Evaluation report saved in reports/ml_results.txt")




# - TF-IDF represents words and phrases effectively for sentiment classification.
# - Logistic Regression works particularly well with sparse text features.
# - Your ANN uses a dense representation of TF-IDF features, which may lose some efficiency and needs suitable tuning.
# - Five epochs and a relatively small dataset may not be enough to optimise the ANN.
# Important: ANN being lower than Logistic Regression is not a failure. It's a useful experimental result. Also, accuracy alone doesn't tell us whether the model handles positive and negative reviews equally well.