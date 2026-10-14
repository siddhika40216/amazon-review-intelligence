
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000/predict"


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    confidence = None
    review = ""
    error = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if not review:
            error = "Please enter a review first."
        else:
            try:
                response = requests.post(
                    API_URL,
                    json={"review": review},
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()

                sentiment = result.get("sentiment")
                confidence = result.get("confidence")

                if sentiment is None:
                    error = result.get(
                        "error", "Could not analyze this review."
                    )

            except requests.RequestException:
                error = (
                    "Cannot connect to the prediction API. "
                    "Please check whether FastAPI is running."
                )

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
