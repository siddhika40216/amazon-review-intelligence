
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Project data folder
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("Loading Amazon Reviews Polarity dataset...")

# Stream the dataset instead of downloading all 4 million reviews
train_stream = load_dataset(
    "fancyzhx/amazon_polarity",
    split="train",
    streaming=True
)

test_stream = load_dataset(
    "fancyzhx/amazon_polarity",
    split="test",
    streaming=True
)

# Shuffle and select manageable subsets
train_reviews = list(
    train_stream.shuffle(seed=42, buffer_size=10000).take(20000)
)

test_reviews = list(
    test_stream.shuffle(seed=42, buffer_size=5000).take(4000)
)

def prepare_reviews(reviews):
    rows = []

    for review in reviews:
        title = review.get("title", "") or ""
        content = review.get("content", "") or ""

        text = (title + " " + content).strip()

        rows.append({
            "text": text,
            "label": int(review["label"])
        })

    return pd.DataFrame(rows)

train_df = prepare_reviews(train_reviews)
test_df = prepare_reviews(test_reviews)

train_df.to_csv(DATA_DIR / "train_reviews.csv", index=False)
test_df.to_csv(DATA_DIR / "test_reviews.csv", index=False)

print("\nDataset downloaded successfully!")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

print("\nSample reviews:")
print(train_df.head())

print("\nFiles saved inside the data folder.")
