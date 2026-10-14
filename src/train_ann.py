
import time
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

train_df = pd.read_csv(DATA_DIR / "train_reviews.csv")
test_df = pd.read_csv(DATA_DIR / "test_reviews.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["label"].to_numpy()

X_test = test_df["text"].fillna("")
y_test = test_df["label"].to_numpy()

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Fit vectorizer only on training data
print("\nPreparing TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    min_df=2,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Neural networks need dense arrays for this simple model.
# Keep the feature count limited to manage memory.
X_train_dense = X_train_tfidf.toarray().astype("float32")
X_test_dense = X_test_tfidf.toarray().astype("float32")

print("Feature count:", X_train_dense.shape[1])

# Build ANN
model = Sequential([
    Input(shape=(X_train_dense.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining ANN...")
start_time = time.time()

model.fit(
    X_train_dense,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

training_time = time.time() - start_time

print(f"\nTraining time: {training_time:.2f} seconds")

# Evaluate on held-out test data
probabilities = model.predict(X_test_dense, verbose=0).ravel()
y_pred = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nANN Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
)
print(report)

# Save model and vectorizer
model.save(MODEL_DIR / "ann_model.keras")
joblib.dump(vectorizer, MODEL_DIR / "ann_tfidf_vectorizer.pkl")

with open(REPORT_DIR / "ann_results.txt", "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Training time: {training_time:.2f} seconds\n\n")
    file.write(report)

print("\nANN model saved successfully!")
