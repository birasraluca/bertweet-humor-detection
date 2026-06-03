from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = Flask(__name__)
CORS(app)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "bertweet_humor_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    normalization=True
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("Model loaded successfully.")
print("Model config:", model.config)


def predict_humor(text: str) -> dict:
    """
    Predicts whether a given text is humorous or not.
    Returns the predicted label and class probabilities.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    not_humorous_probability = probabilities[0].item()
    humorous_probability = probabilities[1].item()

    predicted_class = torch.argmax(probabilities).item()

    if predicted_class == 1:
        label = "Humorous"
        confidence = humorous_probability
    else:
        label = "Not humorous"
        confidence = not_humorous_probability

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "not_humorous_probability": round(not_humorous_probability, 4),
        "humorous_probability": round(humorous_probability, 4)
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Humor Detection API is running.",
        "model": "Fine-tuned BERTweet",
        "endpoint": "/predict"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Please provide a non-empty text field."
        }), 400

    result = predict_humor(text.strip())

    return jsonify({
        "input_text": text,
        **result
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)