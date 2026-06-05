from flask import Flask, request, jsonify
from flask_cors import CORS

from multitask_inference import predict_multitask

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Multi-task Humor Detection API is running.",
        "model": "Multi-task fine-tuned BERTweet",
        "endpoint": "/predict",
        "outputs": [
            "humor label",
            "confidence",
            "humor rating",
            "offense rating",
            "controversy prediction"
        ]
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

    try:
        result = predict_multitask(text.strip())
        return jsonify({
            "input_text": text,
            **result
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)