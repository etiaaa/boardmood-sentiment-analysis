# -*- coding: utf-8 -*-
"""Représentations vectorielles du texte pour le Lot 2.

Trois familles, toutes alignées sur le split figé du Lot 1 :
  - TF-IDF            : vectorisation fréquentielle (référence explicable)
  - Word2Vec          : skip-gram + negative sampling entraîné EN PyTorch sur le corpus
                        (pas de gensim -> aucune dépendance à compiler)
  - BERT (DistilBERT) : embeddings contextuels par extraction de features (mean pooling)
"""
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. TF-IDF
# =====================================================================
def build_tfidf(train_texts, ngram_range=(1, 2), min_df=3, max_features=20000):
    """Ajuste un TfidfVectorizer sur le train. Renvoie le vectoriseur ajusté."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                          max_features=max_features, sublinear_tf=True)
    vec.fit(train_texts)
    return vec


# =====================================================================
# 2. WORD2VEC (skip-gram negative sampling, PyTorch)
# =====================================================================
class _SGNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.inp = nn.Embedding(vocab_size, dim)
        self.out = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.inp.weight, -0.5/dim, 0.5/dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):
        v = self.inp(center)                       # (B, D)
        u = self.out(context)                      # (B, D)
        un = self.out(negatives)                   # (B, K, D)
        pos = F.logsigmoid((v * u).sum(1))                       # (B,)
        neg = F.logsigmoid(-(un * v.unsqueeze(1)).sum(2)).sum(1) # (B,)
        return -(pos + neg).mean()


def train_word2vec(sentences, dim=100, window=5, min_count=3, neg=5,
                   epochs=5, batch_size=2048, lr=0.01, seed=42, verbose=True):
    """Entraîne un Word2Vec skip-gram. `sentences` = liste de listes de tokens.

    Renvoie (embeddings (V,dim) normalisés-non, word2idx, idx2word).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # --- vocabulaire ---
    counts = Counter(w for s in sentences for w in s)
    vocab = [w for w, c in counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    V = len(vocab)

    # --- paires (center, context) ---
    centers, contexts = [], []
    for s in sentences:
        ids = [word2idx[w] for w in s if w in word2idx]
        for i, c in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(c); contexts.append(ids[j])
    centers = np.asarray(centers, dtype=np.int64)
    contexts = np.asarray(contexts, dtype=np.int64)
    n_pairs = len(centers)

    # --- table de negative sampling (unigram^0.75) ---
    freqs = np.array([counts[idx2word[i]] for i in range(V)], dtype=np.float64) ** 0.75
    probs = freqs / freqs.sum()
    table = rng.choice(V, size=1_000_000, p=probs).astype(np.int64)

    model = _SGNS(V, dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"Vocab={V} | paires={n_pairs:,} | dim={dim} | epochs={epochs}")

    for ep in range(epochs):
        perm = rng.permutation(n_pairs)
        total, nb = 0.0, 0
        for start in range(0, n_pairs, batch_size):
            bidx = perm[start:start + batch_size]
            ctr = torch.from_numpy(centers[bidx])
            ctx = torch.from_numpy(contexts[bidx])
            negs = torch.from_numpy(table[rng.integers(0, len(table), size=(len(bidx), neg))])
            loss = model(ctr, ctx, negs)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  loss={total/nb:.4f}")

    emb = model.inp.weight.detach().cpu().numpy()
    return emb, word2idx, idx2word


def most_similar(word, emb, word2idx, idx2word, topn=8):
    """Voisins sémantiques par similarité cosinus."""
    if word not in word2idx:
        return []
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = n @ n[word2idx[word]]
    order = np.argsort(-sims)
    return [(idx2word[i], float(sims[i])) for i in order if i != word2idx[word]][:topn]


def document_vectors(token_lists, emb, word2idx):
    """Vecteur de document = moyenne des vecteurs de mots connus (sinon zéro)."""
    dim = emb.shape[1]
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for r, toks in enumerate(token_lists):
        ids = [word2idx[w] for w in toks if w in word2idx]
        if ids:
            out[r] = emb[ids].mean(0)
    return out


# =====================================================================
# 3. BERT (DistilBERT) — extraction de features (mean pooling)
# =====================================================================
def bert_embeddings(texts, model_name="distilbert-base-uncased", batch_size=64,
                    max_length=64, verbose=True):
    """Extrait un embedding (mean pooling masqué) par texte. Renvoie (N, 768)."""
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()

    texts = list(texts)
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            hidden = mdl(**enc).last_hidden_state           # (B, L, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy().astype(np.float32))
            if verbose and (start // batch_size) % 20 == 0:
                print(f"  BERT {start+len(batch)}/{len(texts)}", end="\r")
    if verbose:
        print()
    return np.vstack(out)
