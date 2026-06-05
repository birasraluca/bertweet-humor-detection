"""
Standalone inference helper for the multi-task BERTweet humor model.

Expected model folder after running the notebook:
backend/model/bertweet_multitask_humor_model/
    config.json
    tokenizer files
    multitask_model_state.pt
"""

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "bertweet_multitask_humor_model")
BASE_MODEL_NAME = "vinai/bertweet-base"


class BertweetMultiTaskModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.bertweet = AutoModel.from_pretrained(BASE_MODEL_NAME, config=config)
        hidden_size = config.hidden_size
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)

        self.dropout = nn.Dropout(dropout_prob)
        self.humor_classifier = nn.Linear(hidden_size, 2)
        self.controversy_classifier = nn.Linear(hidden_size, 2)
        self.humor_regressor = nn.Linear(hidden_size, 1)
        self.offense_regressor = nn.Linear(hidden_size, 1)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits_humor = self.humor_classifier(pooled)
        logits_controversy = self.controversy_classifier(pooled)
        pred_humor_rating = self.humor_regressor(pooled).squeeze(-1)
        pred_offense_rating = self.offense_regressor(pooled).squeeze(-1)

        return SequenceClassifierOutput(
            logits=(logits_humor, pred_humor_rating, pred_offense_rating, logits_controversy)
        )


def load_model():
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Run notebooks/STAI_MULTITASK_BERTWEET.ipynb and place the exported folder here."
        )

    config = AutoConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, normalization=True)

    model = BertweetMultiTaskModel(config)
    state_path = os.path.join(MODEL_DIR, "multitask_model_state.pt")
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    print("Loaded model from:", MODEL_DIR)
    print("State file:", state_path)
    print("Humor head sample:", model.humor_classifier.weight[0, :5])

    model.eval()
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


def predict_multitask(text: str) -> Dict[str, object]:
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_humor, pred_humor_rating, pred_offense_rating, logits_controversy = outputs.logits
        humor_probs = F.softmax(logits_humor, dim=-1)[0]
        controversy_probs = F.softmax(logits_controversy, dim=-1)[0]

    humor_label_id = int(torch.argmax(humor_probs).item())
    controversy_id = int(torch.argmax(controversy_probs).item())

    humor_score = float(pred_humor_rating[0].item())
    offense_score = float(pred_offense_rating[0].item())
    humor_score = max(0.0, min(5.0, humor_score))
    offense_score = max(0.0, min(5.0, offense_score))

    return {
        "label": "Humorous" if humor_label_id == 1 else "Not humorous",
        "confidence": float(humor_probs[humor_label_id].item()),
        "not_humorous_probability": float(humor_probs[0].item()),
        "humorous_probability": float(humor_probs[1].item()),
        "humor_rating_0_to_5": round(humor_score, 2),
        "offense_rating_0_to_5": round(offense_score, 2),
        "controversial": bool(controversy_id),
        "controversy_probability": float(controversy_probs[1].item()),
    }


if __name__ == "__main__":
    examples = [
        "I used to play piano by ear, but now I use my hands.",
        "The database backup completed successfully.",
        "My cat looked at the expensive food I bought and decided starvation was more dignified.",
    ]

    for example in examples:
        print(example)
        print(predict_multitask(example))
        print("-" * 60)
