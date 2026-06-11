# -*- coding: utf-8 -*-
"""Logique d'inférence pour l'application de démonstration (Lot 5).

Deux modèles servis :
  - LogReg + TF-IDF  : rapide, explicable (mots qui ont pesé dans la décision)
  - BiLSTM+Attention : deep learning interprétable (poids d'attention par mot)
Indépendant de Streamlit -> testable en CLI.
"""
import sys, json
from pathlib import Path
import numpy as np

ORDER = ["negative", "neutral", "positive"]


def _src_on_path():
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_logreg(art):
    import joblib
    vec = joblib.load(art / "tfidf_vectorizer.joblib")
    clf = joblib.load(art / "logreg_tfidf.joblib")
    return vec, clf


def predict_logreg(text_clean, vec, clf, topk=6):
    """Renvoie (probas dict, liste (mot, contribution) triée) pour le tweet."""
    X = vec.transform([text_clean])
    proba = clf.predict_proba(X)[0]
    pred = ORDER[int(np.argmax(proba))]
    # contributions = tf-idf * coef de la classe prédite
    feats = vec.get_feature_names_out()
    coef = clf.coef_[int(np.argmax(proba))]
    Xc = X.tocoo()
    contribs = [(feats[j], float(X[0, j] * coef[j])) for j in Xc.col]
    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pred, {ORDER[i]: float(proba[i]) for i in range(3)}, contribs[:topk]


def load_translator(name="Helsinki-NLP/opus-mt-en-fr"):
    """Charge un modèle de traduction anglais→français (pour rendre les tweets lisibles en FR)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    return tok, mdl


def translate(text, tok, mdl, max_length=90):
    import torch
    enc = tok([text], return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        out = mdl.generate(**enc, max_length=max_length)
    return tok.decode(out[0], skip_special_tokens=True)


def translate_many(texts, tok, mdl, max_length=24):
    """Traduit une liste de mots/expressions (EN->FR), en lot."""
    import torch
    texts = list(texts)
    if not texts:
        return []
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        out = mdl.generate(**enc, max_length=max_length)
    return [tok.decode(o, skip_special_tokens=True).strip(" .!?") for o in out]


def explain_logreg(text_clean, vec, clf, target_label, topk=6):
    """Explique une prédiction : mots qui POUSSENT vers `target_label` (pos)
    et mots qui jouent EN SENS INVERSE (neg), via contribution = tfidf * coef."""
    X = vec.transform([text_clean])
    cls = ORDER.index(target_label)
    coef = clf.coef_[cls]
    feats = vec.get_feature_names_out()
    contribs = [(feats[j], float(X[0, j] * coef[j])) for j in X.tocoo().col]
    pos = sorted([c for c in contribs if c[1] > 0], key=lambda t: -t[1])[:topk]
    neg = sorted([c for c in contribs if c[1] < 0], key=lambda t: t[1])[:topk]
    return pos, neg


def load_bert(art):
    """Charge le DistilBERT fine-tuné sauvegardé (ou lève FileNotFoundError)."""
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = art / "bert_finetuned"
    if not (path / "config.json").exists():
        raise FileNotFoundError("Modèle BERT fine-tuné absent (lancer src/finetune_bert.py).")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model


def predict_bert(text_raw, tok, model, maxlen=64):
    """Prédit sur le texte BRUT (BERT gère sa propre tokenisation)."""
    import torch
    enc = tok([text_raw], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
    with torch.no_grad():
        proba = torch.softmax(model(**enc).logits, dim=1)[0].numpy()
    pred = ORDER[int(proba.argmax())]
    return pred, {ORDER[i]: float(proba[i]) for i in range(3)}


def load_attn(art):
    _src_on_path()
    import torch
    import dl_models as M, seq_utils as S
    emb, w2i, pad = S.load_w2v(art)
    model = M.AttnBiLSTM(emb, pad)
    model.load_state_dict(torch.load(art / "attn_w2v_state.pt"))
    model.eval()
    return model, w2i, pad


def predict_attn(text_clean, model, w2i, pad, maxlen=40):
    """Renvoie (pred, probas dict, tokens, poids d'attention)."""
    import torch
    import seq_utils as S
    toks = [w for w in str(text_clean).split() if w in w2i][:maxlen]
    if not toks:
        return None, {c: 0.0 for c in ORDER}, [], []
    seq = S.texts_to_sequences([text_clean], w2i, pad, maxlen)
    with torch.no_grad():
        logits, w = model(torch.tensor(seq), return_attn=True)
    proba = torch.softmax(logits, dim=1)[0].numpy()
    pred = ORDER[int(proba.argmax())]
    weights = w[0, :len(toks)].numpy()
    return pred, {ORDER[i]: float(proba[i]) for i in range(3)}, toks, weights.tolist()
