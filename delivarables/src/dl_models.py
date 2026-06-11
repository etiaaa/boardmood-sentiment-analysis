# -*- coding: utf-8 -*-
"""Modèles Deep Learning du Lot 3 (PyTorch) + utilitaires d'entraînement.

Architectures : ANN (sur TF-IDF/BERT), TextCNN, BiLSTM, BiLSTM+Attention
(sur séquences Word2Vec). Boucle d'entraînement générique avec class_weight,
sélection du meilleur modèle sur le F1-macro de validation.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)


# ------------------------------------------------------------------ DATASETS
class DenseDataset(Dataset):
    """Vecteurs denses (TF-IDF converti, ou embeddings BERT)."""
    def __init__(self, X, y):
        self.sparse = hasattr(X, "tocsr")
        self.X = X.tocsr() if self.sparse else np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        row = self.X[i].toarray().ravel() if self.sparse else self.X[i]
        return torch.tensor(row, dtype=torch.float32), torch.tensor(self.y[i])


class SeqDataset(Dataset):
    """Séquences d'indices de mots (pour CNN/LSTM/Attention)."""
    def __init__(self, seqs, y):
        self.seqs = np.asarray(seqs, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        return torch.tensor(self.seqs[i]), torch.tensor(self.y[i])


# ------------------------------------------------------------------ MODELES
class ANN(nn.Module):
    """Perceptron multicouche pour entrées denses (TF-IDF, BERT)."""
    def __init__(self, in_dim, hidden=256, n_classes=3, p=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden // 2, n_classes))
    def forward(self, x): return self.net(x)


def _embedding_layer(emb_matrix, pad_idx, freeze):
    emb = nn.Embedding.from_pretrained(
        torch.tensor(emb_matrix, dtype=torch.float32), freeze=freeze, padding_idx=pad_idx)
    return emb


class TextCNN(nn.Module):
    """CNN 1D multi-noyaux (détection de n-grammes) sur embeddings."""
    def __init__(self, emb_matrix, pad_idx, kernel_sizes=(3, 4, 5), n_filters=100,
                 n_classes=3, p=0.5, freeze=False):
        super().__init__()
        self.emb = _embedding_layer(emb_matrix, pad_idx, freeze)
        d = emb_matrix.shape[1]
        self.convs = nn.ModuleList([nn.Conv1d(d, n_filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                 # (B, D, L)
        feats = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class BiLSTM(nn.Module):
    """BiLSTM, représentation = moyenne des états (masquée)."""
    def __init__(self, emb_matrix, pad_idx, hidden=128, n_classes=3, p=0.5, freeze=False):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = _embedding_layer(emb_matrix, pad_idx, freeze)
        self.lstm = nn.LSTM(emb_matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hidden * 2, n_classes)
    def forward(self, x):
        mask = (x != self.pad_idx).unsqueeze(-1).float()
        out, _ = self.lstm(self.emb(x))                 # (B, L, 2H)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.fc(self.drop(pooled))


class AttnBiLSTM(nn.Module):
    """BiLSTM + attention additive. Renvoie les logits (+ poids d'attention si demandé)."""
    def __init__(self, emb_matrix, pad_idx, hidden=128, n_classes=3, p=0.5, freeze=False):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = _embedding_layer(emb_matrix, pad_idx, freeze)
        self.lstm = nn.LSTM(emb_matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hidden * 2, n_classes)
    def forward(self, x, return_attn=False):
        mask = (x != self.pad_idx)                      # (B, L)
        out, _ = self.lstm(self.emb(x))                 # (B, L, 2H)
        scores = self.attn(out).squeeze(-1)             # (B, L)
        scores = scores.masked_fill(~mask, -1e9)   # -1e9 (pas -inf) -> pas de NaN si tout est paddé
        w = torch.softmax(scores, dim=1)                # (B, L)
        ctx = (out * w.unsqueeze(-1)).sum(1)            # (B, 2H)
        logits = self.fc(self.drop(ctx))
        return (logits, w) if return_attn else logits


# ------------------------------------------------------------------ ENTRAINEMENT
@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds = []; ys = []
    for xb, yb in loader:
        out = model(xb)
        preds.append(out.argmax(1).cpu().numpy()); ys.append(yb.numpy())
    p = np.concatenate(preds); y = np.concatenate(ys)
    return f1_score(y, p, average="macro"), p


def fit(model, train_ds, val_ds, class_weights, epochs=12, lr=1e-3, batch_size=64,
        seed=42, verbose=True):
    """Entraîne et garde le meilleur état (F1-macro val). Renvoie (history, best_f1)."""
    set_seed(seed)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=256)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_f1": []}
    best_f1, best_state = -1.0, None
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in tl:
            opt.zero_grad(); loss = crit(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        f1, _ = evaluate(model, vl)
        history["train_loss"].append(tot / nb); history["val_f1"].append(f1)
        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {ep+1:>2}/{epochs}  loss={tot/nb:.4f}  val_F1={f1:.4f}")
    model.load_state_dict(best_state)
    return history, best_f1
