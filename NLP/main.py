import string
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from nltk import word_tokenize
from fastapi import FastAPI
from pydantic import BaseModel
#uvicorn main:app --host 127.0.0.1 --port 8000启动

# ── 模型定义（与训练时完全一致）──────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, num_classes, dropout, pad_idx=0):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm1      = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2      = nn.LSTM(hidden_dim * 2, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        emb     = self.dropout(self.embedding(x))
        p1, _   = self.lstm1(pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False))
        o1, _   = pad_packed_sequence(p1, batch_first=True)
        p2, _   = self.lstm2(pack_padded_sequence(self.dropout(o1), lengths.cpu(), batch_first=True, enforce_sorted=False))
        o2, _   = pad_packed_sequence(p2, batch_first=True)
        idx     = (lengths - 1).clamp(min=0).to(x.device)
        last    = o2[torch.arange(o2.size(0), device=x.device), idx]
        return self.classifier(self.layer_norm(last))


# ── 加载模型 ──────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vocab 现在存为普通 dict，weights_only=True 即可安全加载
ckpt    = torch.load("bilstm_sentiment_checkpoint.pt", map_location=DEVICE, weights_only=True)

word2idx = ckpt["word2idx"]          # 普通字典，直接使用
MAX_LEN  = ckpt["hparams"]["max_len"]
hp       = {k: v for k, v in ckpt["hparams"].items() if k != "max_len"}
model    = BiLSTMClassifier(**hp).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

LABELS = {0: "Negative", 1: "Somewhat Neg", 2: "Neutral", 3: "Somewhat Pos", 4: "Positive"}
UNK_IDX = 1  # <UNK> 的索引

def encode(tokens):
    return [word2idx.get(t, UNK_IDX) for t in tokens]


# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(body: TextInput):
    cleaned = body.text.translate(str.maketrans("", "", string.punctuation)).lower()
    tokens  = word_tokenize(cleaned)[:MAX_LEN] or ["<UNK>"]
    ids     = torch.tensor([encode(tokens)], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([len(tokens)], dtype=torch.long).to(DEVICE)

    with torch.inference_mode():
        probs = F.softmax(model(ids, lengths), dim=1).squeeze().tolist()

    pred = int(torch.tensor(probs).argmax())
    return {"text": body.text, "label": pred, "sentiment": LABELS[pred], "confidence": round(probs[pred], 4)}
