
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model folder ka path
project_path = Path(__file__).resolve().parent.parent
model_path = project_path / "models" / "amazon_distilbert_model"

# CPU ya GPU select karna
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Saved model aur tokenizer load karna
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
model.to(device)
model.eval()

def predict_sentiment(review):
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
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[0][prediction].item() * 100

    return sentiment, confidence


if __name__ == "__main__":
    review = input("Enter an Amazon review: ")

    sentiment, confidence = predict_sentiment(review)

    print("\nSentiment:", sentiment)
    print(f"Confidence: {confidence:.2f}%")
