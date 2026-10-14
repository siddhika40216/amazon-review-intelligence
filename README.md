# Amazon Review Intelligence 🔍

**An NLP-powered sentiment analysis system for Amazon product reviews, comparing traditional machine learning, neural networks, and transformer-based models.**

Amazon Review Intelligence classifies product reviews as **Positive** or **Negative** and evaluates three modeling approaches on a common test dataset. The project combines model experimentation with a web interface powered by Flask and FastAPI.

## 📌 Project Highlights

- Built and evaluated three sentiment classification models.
- Fine-tuned DistilBERT for contextual sentiment understanding.
- Evaluated all models on the same 4,000-review test set.
- Achieved **93.38% test accuracy** with DistilBERT.
- Developed a Flask frontend connected to a FastAPI prediction backend.
- Saved evaluation metrics for model comparison and reporting.

## 📊 Model Performance

All metrics below were obtained from the project's 4,000-review test set.

| Model                           |   Accuracy |  Precision |     Recall |   F1-Score |
| :------------------------------ | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression             |     86.45% |     85.80% |     87.99% |     86.88% |
| Artificial Neural Network (ANN) |     82.02% |     80.54% |     85.39% |     82.89% |
| **DistilBERT**                  | **93.38%** | **94.04%** | **92.89%** | **93.46%** |

### Key Findings

- **DistilBERT performed best** across accuracy, precision, and F1-score.
- Logistic Regression provided a strong, computationally inexpensive baseline.
- The ANN performed below the Logistic Regression baseline on this dataset.
- DistilBERT's local CPU evaluation took approximately 649 seconds for 4,000 reviews, highlighting the trade-off between predictive performance and inference speed.

## 🖥️ Application Preview

The application provides a simple web interface where users can enter an Amazon product review and receive a predicted sentiment (Positive or Negative) along with a confidence score.

## 🧠 Models and Methodology

### 1. Logistic Regression

- Text representation: TF-IDF
- Includes unigram and bigram features
- Serves as a traditional machine learning baseline

### 2. Artificial Neural Network (ANN)

- Uses TF-IDF features as input
- Implemented with TensorFlow/Keras
- Evaluated using the same held-out test dataset

### 3. DistilBERT

- Pretrained transformer: `distilbert-base-uncased`
- Fine-tuned for binary sentiment classification
- Training configuration: two epochs, maximum sequence length of 256 tokens, learning rate of `2e-5`
- Uses a separate validation split during training and a held-out test set for final evaluation

## 🏗️ Architecture

```text
                   User Review
                        |
                        v
                Flask Web Interface
                        |
                        v
                 FastAPI Backend
                        |
                        v
                 DistilBERT Model
                        |
                        v
             Positive / Negative Result
                        |
                        v
               Confidence Percentage
```

The training and evaluation scripts are separate from the web application's prediction flow.

## 🛠️ Tech Stack

| Category          | Technologies                       |
| :---------------- | :--------------------------------- |
| Language          | Python                             |
| Data Processing   | Pandas, NumPy                      |
| Traditional ML    | Scikit-learn                       |
| Neural Networks   | TensorFlow / Keras                 |
| Transformer Model | PyTorch, Hugging Face Transformers |
| NLP               | TF-IDF, DistilBERT                 |
| Backend API       | FastAPI, Uvicorn                   |
| Frontend          | Flask, HTML, CSS, JavaScript       |
| Visualization     | Matplotlib                         |
| Version Control   | Git, GitHub                        |

## 📂 Project Structure

```text
amazon-review-intelligence/
├── data/
│   ├── train_reviews.csv
│   └── test_reviews.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── ann_model.keras
│   ├── ann_tfidf_vectorizer.pkl
│   └── amazon_distilbert_model/
├── src/
│   ├── download_data.py
│   ├── train_ml.py
│   ├── train_ann.py
│   ├── predict_distilbert.py
│   ├── evaluate_models.py
│   ├── plot_results.py
│   └── api.py
├── frontend/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── reports/
│   ├── model_comparison.csv
│   └── model_comparison.png
├── .gitignore
└── README.md
```

*Note: This represents the intended project layout. File availability may differ depending on which artifacts have been generated or committed.*

## 📚 Dataset

This project uses the [Amazon Polarity dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity).

The local working dataset contains:

- **20,000** reviews for model development
- **4,000** reviews for final testing

Labels:

- `0` — Negative
- `1` — Positive

For DistilBERT, the training dataset was split into training and validation subsets. The held-out test set was reserved for final evaluation.

The local CSV files are excluded from version control to avoid committing dataset files unnecessarily.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- Git
- The project's dataset files and trained model artifacts

### 1. Clone the repository

```bash
git clone https://github.com/siddhika40216/amazon-review-intelligence.git
cd amazon-review-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib matplotlib tensorflow torch transformers fastapi uvicorn flask requests
```

### 4. Prepare datasets and model artifacts

Place the expected CSV files in `data/` and the corresponding trained artifacts in `models/`.

Large trained models and datasets are not included in the repository by default. The model-loading scripts require the appropriate files to be available locally.

### 5. Run the FastAPI backend

From the project root:

```bash
python -m uvicorn src.api:app --reload
```

API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 6. Run the Flask frontend

Open a second terminal, activate the virtual environment, and run:

```bash
python -m flask --app frontend.app run --debug
```

Open the application:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

Keep the FastAPI backend running while testing predictions through the Flask interface.

## 📈 Reproducing the Evaluation

Evaluate the models using the held-out test dataset:

```bash
python src/evaluate_models.py
```

The script generates:

```text
reports/model_comparison.csv
```

Generate the comparison visualization:

```bash
python src/plot_results.py
```

Output:

```text
reports/model_comparison.png
```

Exact metrics can vary if the dataset, preprocessing, model artifacts, or evaluation setup changes.

## ⚠️ Limitations

- The dataset is a limited sample and may not represent every Amazon product category.
- The classifier predicts sentiment, not product quality or factual review authenticity.
- The confidence score should not be interpreted as a calibrated probability without additional validation.
- Transformer inference is relatively slow on a CPU.
- The application requires the correct trained model files to be available locally.

## 🚀 Future Improvements

- Interactive model comparison dashboard
- Batch review analysis and sentiment summaries
- Inference latency optimization
- Expanded evaluation across product categories
- Deployment of the frontend and backend
- Additional testing for edge cases and neutral or mixed-sentiment reviews

## 👩‍💻 Author

**Siddhika Srivastava**
B.Tech, Computer Science and Engineering
Banasthali Vidyapith

[GitHub Profile](https://github.com/siddhika40216)

---

*Built as a machine learning and natural language processing project to explore sentiment classification and model evaluation.*
