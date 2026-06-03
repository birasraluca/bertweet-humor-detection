# BERTweet Humor Detection

Automatic humor detection in short English texts using classical machine learning baselines and a fine-tuned BERTweet transformer model.

This project was developed for a university NLP assignment on humor and jokes. The system classifies short English texts as either **humorous** or **not humorous** and provides prediction confidence scores through a simple web demo.

## Project Overview

The project explores automatic humor detection as a binary text classification task.

Given an input sentence, the system predicts whether the text is humorous or not. The model was trained and evaluated in Google Colab, then integrated into a local web application consisting of:

* a Flask backend API
* a React frontend interface
* a fine-tuned BERTweet model for prediction

The final demo allows the user to enter a sentence and receive:

* predicted label: `Humorous` or `Not humorous`
* confidence score
* probability for each class

## Dataset

The dataset used is the **SemEval 2021 Task 7: HaHackathon - Detecting and Rating Humor and Offense** dataset.

The project uses the binary humor detection label:

* `0` = not humorous
* `1` = humorous

Only the text and humor label were used for the main classification task.

Dataset columns include:

* `id`
* `text`
* `is_humor`
* `humor_rating`
* `humor_controversy`
* `offense_rating`

For this project, the main columns used were:

```text
text
is_humor
```

## Models

Three models were explored:

| Model                                 | Description                                            |
| ------------------------------------- | ------------------------------------------------------ |
| TF-IDF + Logistic Regression          | Classical machine learning baseline                    |
| TF-IDF + Balanced Logistic Regression | Baseline with class weighting to reduce imbalance bias |
| Fine-tuned BERTweet                   | Transformer-based model used as the final model        |

The final application uses the fine-tuned **BERTweet** model because it achieved the best validation results.

## Results

### Logistic Regression Baseline

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8288 |
| Precision | 0.8128 |
| Recall    | 0.9381 |
| F1-score  | 0.8710 |

### Balanced Logistic Regression

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8275 |
| Precision | 0.8769 |
| Recall    | 0.8377 |
| F1-score  | 0.8568 |

### Fine-tuned BERTweet

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9556 |
| Precision | 0.9683 |
| Recall    | 0.9594 |
| F1-score  | 0.9638 |

Classification report for BERTweet:

```text
              precision    recall  f1-score   support

Not humorous       0.94      0.95      0.94       614
    Humorous       0.97      0.96      0.96       986

    accuracy                           0.96      1600
   macro avg       0.95      0.95      0.95      1600
weighted avg       0.96      0.96      0.96      1600
```

The transformer model significantly outperformed the classical baselines, achieving approximately **95.56% accuracy** and **96.38% F1-score** for the humorous class.

## Project Structure

```text
bertweet-humor-detection/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── model/
│       └── bertweet_humor_model/
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── main.jsx
│
├── notebooks/
│   └── humor_detection_training.ipynb
│
├── results/
│   └── confusion_matrix_bertweet.png
│
├── .gitignore
└── README.md
```

## Backend

The backend is implemented using Flask.

It exposes a prediction endpoint:

```http
POST /predict
```

Example request:

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
  "confidence": 0.9981,
  "not_humorous_probability": 0.0019,
  "humorous_probability": 0.9981
}
```

## Frontend

The frontend is implemented using React and Vite.

The interface allows the user to:

* enter a custom sentence
* run humor detection
* view the predicted class
* view the confidence score
* view both class probabilities

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/birasraluca/bertweet-humor-detection.git
cd bertweet-humor-detection
```

## 2. Add the trained model

The fine-tuned BERTweet model should be placed in:

```text
backend/model/bertweet_humor_model/
```

The folder should contain files such as:

```text
config.json
model.safetensors
tokenizer_config.json
special_tokens_map.json
vocab.txt
bpe.codes
```

The model folder is not included in the repository because transformer model files can be large.

## 3. Run the backend

Open a terminal in the project root:

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment.

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask server:

```bash
python app.py
```

The backend should start at:

```text
http://127.0.0.1:5000
```

## 4. Run the frontend

Open a second terminal in the project root:

```bash
cd frontend
npm install
npm run dev
```

The frontend should start at:

```text
http://localhost:5173
```

## Example Inputs

Humorous example:

```text
I used to play piano by ear, but now I use my hands.
```

Expected output:

```text
Humorous
```

Non-humorous example:

```text
The meeting will start tomorrow at 10 AM.
```

Expected output:

```text
Not humorous
```

## Notes on Confidence Scores

The confidence score represents the model's predicted probability for the selected class. It should not be interpreted as an objective measurement of how funny a sentence is.

Humor is subjective and context-dependent, so even a high-confidence prediction can still be incorrect.

## Limitations

Although the BERTweet model achieved strong validation performance, the system still has limitations:

* humor is subjective and culturally dependent
* the model only works with English text
* the model was trained on short texts, so long paragraphs may be less reliable
* the confidence score is a model probability, not a true humor rating
* some serious sentences may be misclassified if they resemble joke structures
* offensive or controversial humor may affect predictions

## Technologies Used

* Python
* Flask
* PyTorch
* Hugging Face Transformers
* BERTweet
* scikit-learn
* React
* Vite
* CSS

## References

* SemEval 2021 Task 7: HaHackathon - Detecting and Rating Humor and Offense
* BERTweet: A pre-trained language model for English Tweets
* Hugging Face Transformers documentation
* scikit-learn documentation
