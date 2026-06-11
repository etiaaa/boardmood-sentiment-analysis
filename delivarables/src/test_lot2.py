# -*- coding: utf-8 -*-
"""Test de faisabilite du Lot 2 : timing + qualite + comparaison rapide."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import representations as R

ROOT = SRC.parents[1]
OUT = ROOT / "delivarables" / "data_processed"
train = pd.read_csv(OUT/"train.csv"); val = pd.read_csv(OUT/"val.csv")
ytr, yva = train["label"].values, val["label"].values
print("train", train.shape, "val", val.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def evalrep(name, Xtr, Xva):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    f1 = f1_score(yva, clf.predict(Xva), average="macro")
    print(f"  -> {name:10} dim={Xtr.shape[1]:>5}  F1-macro(val)={f1:.4f}")
    return f1

# ---- TF-IDF
t=time.time()
vec = R.build_tfidf(train["text_clean"].fillna(""))
Xtr = vec.transform(train["text_clean"].fillna("")); Xva = vec.transform(val["text_clean"].fillna(""))
print(f"[TF-IDF] {time.time()-t:.1f}s  vocab={len(vec.vocabulary_)}")
evalrep("TF-IDF", Xtr, Xva)

# ---- Word2Vec
t=time.time()
sent_tr = [str(x).split() for x in train["text_clean"].fillna("")]
sent_va = [str(x).split() for x in val["text_clean"].fillna("")]
emb, w2i, i2w = R.train_word2vec(sent_tr, dim=100, epochs=5, verbose=True)
print(f"[W2V] entraine en {time.time()-t:.1f}s")
for w in ["delayed","thanks","bag","hour","worst"]:
    nb = R.most_similar(w, emb, w2i, i2w, topn=6)
    print(f"   {w:8} ->", ", ".join(f"{x}({s:.2f})" for x,s in nb))
Wtr = R.document_vectors(sent_tr, emb, w2i); Wva = R.document_vectors(sent_va, emb, w2i)
evalrep("Word2Vec", Wtr, Wva)

# ---- BERT
t=time.time()
Btr = R.bert_embeddings(train["text"].fillna(""))
Bva = R.bert_embeddings(val["text"].fillna(""))
print(f"[BERT] extrait en {time.time()-t:.1f}s  shape={Btr.shape}")
evalrep("BERT", Btr, Bva)
print("\nOK Lot 2 faisable.")
