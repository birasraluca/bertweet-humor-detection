import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "bertweet_humor_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading from:", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    normalization=True
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

examples = [
    "I used to play piano by ear, but now I use my hands.",
    "The database backup completed successfully.",
    "The meeting will start tomorrow at 10 AM.",
    "My cat looked at the expensive food I bought and decided starvation was more dignified."
]

for text in examples:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    print()
    print(text)
    print("Not humorous:", probs[0].item())
    print("Humorous:", probs[1].item())
    print("Prediction:", "Humorous" if torch.argmax(probs).item() == 1 else "Not humorous")