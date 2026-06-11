# -*- coding: utf-8 -*-
"""Fine-tuning de DistilBERT pour la classification de sentiment (Lot 3).

Entraîne distilbert-base-uncased (num_labels=3) avec pondération des classes,
garde le meilleur modèle sur le F1-macro de validation, sauvegarde
historique + prédictions test dans delivarables/artifacts/.
"""
import sys, json, time, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "delivarables" / "data_processed"
ART = ROOT / "delivarables" / "artifacts"
MODEL = "distilbert-base-uncased"
MAXLEN, BATCH, EPOCHS, LR = 64, 16, 3, 2e-5

train = pd.read_csv(OUT/"train.csv"); val = pd.read_csv(OUT/"val.csv"); test = pd.read_csv(OUT/"test.csv")
meta = json.loads((OUT/"meta.json").read_text())
cw = torch.tensor([meta["class_weight"][c] for c in ["negative","neutral","positive"]], dtype=torch.float32)

tok = AutoTokenizer.from_pretrained(MODEL)

class TweetDS(Dataset):
    def __init__(self, texts, y):
        self.enc = tok(list(texts), padding="max_length", truncation=True,
                       max_length=MAXLEN, return_tensors="pt")
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.y[i]

dtr = TweetDS(train["text"].fillna(""), train["label"])
dva = TweetDS(val["text"].fillna(""),   val["label"])
dte = TweetDS(test["text"].fillna(""),  test["label"])
tl = DataLoader(dtr, batch_size=BATCH, shuffle=True)
vl = DataLoader(dva, batch_size=64)
el = DataLoader(dte, batch_size=64)

torch.manual_seed(42)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss(weight=cw)

@torch.no_grad()
def predict(loader):
    model.eval(); preds = []; ys = []
    for enc, y in loader:
        logits = model(**enc).logits
        preds.append(logits.argmax(1).numpy()); ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)

history = {"val_f1": []}; best_f1 = -1; best_state = None
print(f"Fine-tuning {MODEL} | {EPOCHS} epochs | batch {BATCH}", flush=True)
for ep in range(EPOCHS):
    model.train(); t = time.time(); tot = 0; nb = 0
    for enc, y in tl:
        opt.zero_grad()
        loss = crit(model(**enc).logits, y)
        loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if nb % 100 == 0:
            print(f"  ep{ep+1} step {nb}/{len(tl)} loss={tot/nb:.4f}", flush=True)
    p, yv = predict(vl); f1 = f1_score(yv, p, average="macro")
    history["val_f1"].append(f1)
    print(f"epoch {ep+1}/{EPOCHS}  val_F1={f1:.4f}  ({time.time()-t:.0f}s)", flush=True)
    if f1 > best_f1:
        best_f1 = f1; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
pte, yte = predict(el); test_f1 = f1_score(yte, pte, average="macro")
print(f"BEST val_F1={best_f1:.4f} | TEST_F1={test_f1:.4f}", flush=True)

np.save(ART/"pred_test_BERT-finetune.npy", pte)
json.dump({"arch": "Fine-tuning", "repr": "BERT", "val_f1": round(best_f1,4),
           "test_f1": round(test_f1,4), "history": history},
          open(ART/"bert_finetune_results.json","w"), indent=2)

# --- Sauvegarde du MODELE (pour le servir dans l'app du Lot 5) ---
save_dir = ART / "bert_finetuned"
model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)
json.dump(meta["id2label"], open(save_dir/"id2label.json", "w"))
print(f"Modele sauvegarde dans {save_dir}", flush=True)
print("Sauvegarde OK.", flush=True)
