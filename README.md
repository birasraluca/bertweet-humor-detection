# BERTweet Multi-Task Humor Detection

This project is a demo web API for a fine-tuned **BERTweet-based multi-task humor detection model**.

The model predicts whether a given text is humorous and also provides additional humor-related outputs, including:

* humor classification: `Humorous` / `Not humorous`
* prediction confidence
* humor probability scores
* humor rating from 0 to 5
* offense rating from 0 to 5
* controversy prediction

The project includes a Flask backend that loads the trained model and exposes a simple `/predict` endpoint for inference.

---

## Project Overview

Humor detection is a Natural Language Processing task where the goal is to determine whether a piece of text contains humor. In this project, a transformer-based model was fine-tuned using BERTweet, a RoBERTa-based language model pretrained on English tweets.

Instead of training only a binary classifier, this project uses a **multi-task learning** approach. The model shares the same BERTweet encoder and has multiple task-specific heads:

1. **Humor classification head**
   Predicts whether the text is humorous or not.

2. **Humor rating regression head**
   Predicts how humorous the text is on a scale from 0 to 5.

3. **Offense rating regression head**
   Predicts the offensiveness level of the text on a scale from 0 to 5.

4. **Controversy classification head**
   Predicts whether the humor may be controversial.

This allows the model to learn related humor characteristics together instead of treating humor detection as a completely isolated binary classification task.

<img width="596" height="455" alt="image" src="https://github.com/user-attachments/assets/c297b830-0b62-4ab3-a8d7-54d07f6dc0c5" />
<img width="703" height="864" alt="image" src="https://github.com/user-attachments/assets/e3c993bb-f4a7-4bb3-83a6-dba2036542d8" />
<img width="689" height="859" alt="image" src="https://github.com/user-attachments/assets/0468204c-a229-4ca2-9c16-3700342b196c" />

---

## Features

* Fine-tuned BERTweet model for humor detection
* Multi-task architecture with classification and regression heads
* Flask REST API for inference
* JSON-based `/predict` endpoint
* CORS enabled for frontend/demo integration
* Safe model loading using `strict=True`
* Local model weights loaded from exported checkpoint
* Correct BERTweet tokenizer loading from the original Hugging Face model source

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* BERTweet
* Flask
* Flask-CORS

---


## Model Architecture

The model uses `vinai/bertweet-base` as the shared encoder.

On top of the encoder, the model defines four heads:

```text
BERTweet encoder
│
├── humor classification head
├── humor controversy classification head
├── humor rating regression head
└── offense rating regression head
```

The model outputs:

```python
(
    logits_humor,
    pred_humor_rating,
    pred_offense_rating,
    logits_controversy,
)
```

Where:

* `logits_humor` is used for the humorous / not humorous prediction
* `pred_humor_rating` is a regression output for humor intensity
* `pred_offense_rating` is a regression output for offensive content intensity
* `logits_controversy` is used for the controversial / not controversial prediction

---

## Important Model Loading Note

The model weights are loaded locally from:

```text
backend/model/bertweet_multitask_humor_model/multitask_model_state.pt
```

However, the tokenizer should be loaded from the original BERTweet model source:

```python
vinai/bertweet-base
```

This is important because loading the tokenizer from the exported local folder may produce different token IDs and therefore incorrect predictions.

The backend uses:

```python
AutoTokenizer.from_pretrained(
    "vinai/bertweet-base",
    use_fast=False,
    normalization=True,
)
```

The model weights are loaded with:

```python
model.load_state_dict(state_dict, strict=True)
```

Using `strict=True` ensures that the model fails immediately if the saved weights do not match the architecture. This prevents silent loading errors.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/birasraluca/bertweet-humor-detection.git
cd bertweet-humor-detection
```

### 2. Create and activate a virtual environment

On Windows:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Depending on the environment, additional packages may be needed for training, such as:

```text
datasets
evaluate
scikit-learn
accelerate
```

---

## Model Setup

After training the model in the notebook, place the exported model folder inside:

```text
backend/model/
```

The final path should look like:

```text
backend/model/bertweet_multitask_humor_model/
```

This folder should contain at least:

```text
config.json
multitask_model_state.pt
model_info.json
bertweet_multitask_model.py
```

The most important file is:

```text
multitask_model_state.pt
```

This contains the fine-tuned model weights.

---

## Running the Backend

From the backend directory:

```bash
cd backend
python app.py
```

The Flask server should start on:

```text
http://127.0.0.1:5000
```

If the model loads successfully, the terminal should print information similar to:

```text
Loaded model from: backend/model/bertweet_multitask_humor_model
Loaded tokenizer from: vinai/bertweet-base
State file: backend/model/bertweet_multitask_humor_model/multitask_model_state.pt
```

---

## API Endpoints

### Health Check

```http
GET /
```

Example response:

```json
{
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
}
```

---

### Predict

```http
POST /predict
```

Request body:

```json
{
  "text": "I used to play piano by ear, but now I use my hands."
}
```

Example response:

```json
{
  "input_text": "I used to play piano by ear, but now I use my hands.",
  "label": "Humorous",
  "confidence": 0.98,
  "not_humorous_probability": 0.02,
  "humorous_probability": 0.98,
  "humor_rating_0_to_5": 2.85,
  "offense_rating_0_to_5": 0.12,
  "controversial": false,
  "controversy_probability": 0.21
}
```

The exact values may differ depending on the trained model checkpoint.

---

## Testing with cURL

After starting the Flask server, test the API with:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I used to play piano by ear, but now I use my hands.\"}"
```

On macOS/Linux:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I used to play piano by ear, but now I use my hands."}'
```

Another example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The database backup completed successfully."}'
```

Expected behavior:

* jokes and humorous sentences should usually return `Humorous`
* plain factual sentences should usually return `Not humorous`

---

## Example Predictions

Example humorous input:

```text
I used to play piano by ear, but now I use my hands.
```

Expected label:

```text
Humorous
```

Example non-humorous input:

```text
The database backup completed successfully.
```

Expected label:

```text
Not humorous
```

Example humorous input:

```text
My cat looked at the expensive food I bought and decided starvation was more dignified.
```

Expected label:

```text
Humorous
```

---

## Training Notebook

The training notebook contains the complete model training and export pipeline.

The notebook performs the following steps:

1. Loads and preprocesses the humor dataset
2. Tokenizes text using BERTweet
3. Defines the custom multi-task BERTweet model
4. Trains the model
5. Evaluates the model
6. Saves the model configuration and weights
7. Reloads the exported model
8. Compares original and reloaded predictions
9. Verifies that the reloaded model produces matching results
10. Exports the final model folder for backend use

A key validation step compares predictions before and after reload. This ensures that the downloaded model is safe to use in the backend.

The notebook should only export the model after the original and reloaded predictions match.

---

## Reload Validation

Before using the model in the backend, the notebook verifies that the saved model reloads correctly.

The validation checks that:

* the model loads with `strict=True`
* original and reloaded model weights match
* tokenized input IDs match
* original and reloaded predictions match

This is important because a model can appear to load successfully while still producing incorrect predictions if the tokenizer or architecture differs.

---

## Common Issues and Fixes

### 1. Reloaded predictions are very different from original predictions

Possible cause:

```python
AutoTokenizer.from_pretrained(EXPORT_DIR)
```

Fix:

```python
AutoTokenizer.from_pretrained(
    "vinai/bertweet-base",
    use_fast=False,
    normalization=True,
)
```

The tokenizer must produce the same token IDs as during training.

---

### 2. Model loads but predictions are bad

Possible causes:

* wrong model folder
* wrong checkpoint file
* tokenizer loaded from the wrong path
* model architecture differs from the training architecture
* model loaded with `strict=False`

Fixes:

* confirm that the backend points to the correct folder
* confirm that `multitask_model_state.pt` is the final tested export
* load tokenizer from `vinai/bertweet-base`
* use the same custom model class as in training
* always use:

```python
model.load_state_dict(state_dict, strict=True)
```

---

### 3. Error: missing or unexpected keys when loading state dict

This means the saved checkpoint does not match the current model architecture.

Do not use `strict=False` to ignore this.

Instead, make sure that:

* the backend model class matches the training model class
* the correct checkpoint is being loaded
* the exported model folder is the one that passed reload validation

---

### 4. Flask returns `"Request body must be valid JSON."`

Make sure the request uses:

```http
Content-Type: application/json
```

and that the body is valid JSON:

```json
{
  "text": "Example text here"
}
```

---

### 5. Flask returns `"Please provide a non-empty text field."`

The request body must contain a non-empty string under the `text` key:

```json
{
  "text": "This is an example sentence."
}
```

---

## Notes on Inference

The backend tokenizes input text with:

```python
truncation=True
padding="max_length"
max_length=128
```

The model is used in evaluation mode:

```python
model.eval()
```

Inference is performed without gradient computation:

```python
with torch.no_grad():
    outputs = model(**inputs)
```

This makes predictions faster and avoids unnecessary memory usage.

---

## Limitations

This project is a demo application and should not be treated as a perfect humor understanding system.

Possible limitations:

* humor is subjective and context-dependent
* sarcasm and irony can be difficult to detect
* short factual sentences may sometimes be misclassified
* offensive humor and controversial humor are especially nuanced
* the model performance depends strongly on the training dataset
* the model is trained on English text and may not perform well on other languages

---

## Future Improvements

Possible future improvements include:

* adding batch prediction support
* improving the controversy prediction task
* experimenting with different transformer models
* adding confidence thresholds for uncertain predictions
