"""
电影评论情感分析 REST API
启动: uvicorn sentiment_api:app --reload
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ── 加载模型 ──────────────────────────────────────────────────
model = tf.keras.models.load_model("./bilstm_sentiment_model.keras")
with open("./tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50
LABELS = {0: "Negative", 1: "Somewhat Negative", 2: "Neutral",
          3: "Somewhat Positive", 4: "Positive"}

app = FastAPI(title="情感分析 API")


# ── 数据模型 ──────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]


# ── 预测函数 ──────────────────────────────────────────────────
def predict(texts: List[str]):
    cleaned = [t.translate(str.maketrans("", "", string.punctuation)).lower().strip()
               for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)
    preds = np.argmax(probs, axis=1)

    return [
        {
            "text": texts[i],
            "label": int(preds[i]),
            "sentiment": LABELS[int(preds[i])],
            "confidence": round(float(probs[i, preds[i]]), 4),
        }
        for i in range(len(texts))
    ]


# ── 路由 ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "情感分析 API", "docs": "/docs"}

@app.post("/predict")
def predict_one(req: TextRequest):
    return predict([req.text])[0]

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if len(req.texts) > 100:
        raise HTTPException(400, "最多 100 条")
    return predict(req.texts)