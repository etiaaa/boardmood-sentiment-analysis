# -*- coding: utf-8 -*-
"""Construction des séquences d'indices et de la matrice d'embedding Word2Vec."""
import json
import numpy as np


def load_w2v(art_dir):
    """Charge embeddings + word2idx, ajoute un indice de padding. Renvoie (emb_matrix, word2idx, pad_idx)."""
    emb = np.load(art_dir / "w2v_embeddings.npz")["emb"].astype(np.float32)
    word2idx = json.loads((art_dir / "w2v_word2idx.json").read_text())
    V, D = emb.shape
    pad_idx = V
    emb_matrix = np.vstack([emb, np.zeros((1, D), dtype=np.float32)])   # ligne pad = zéros
    return emb_matrix, word2idx, pad_idx


def texts_to_sequences(texts, word2idx, pad_idx, maxlen=40):
    """Convertit des textes nettoyés en matrice d'indices (N, maxlen), padding à droite."""
    seqs = np.full((len(texts), maxlen), pad_idx, dtype=np.int64)
    for r, t in enumerate(texts):
        ids = [word2idx[w] for w in str(t).split() if w in word2idx][:maxlen]
        seqs[r, :len(ids)] = ids
    return seqs
