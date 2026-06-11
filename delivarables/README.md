# ✈️ BoardMood — Analyse de sentiment des tweets de compagnies aériennes
### Projet NLP & Deep Learning — Master Data Science · équipe **BoardMood**

Classification du sentiment (négatif / neutre / positif) de **14 640 tweets** adressés à 6 compagnies
aériennes US, de la donnée brute au **modèle déployé**.

---

## Démarrage rapide

```bash
pip install -r requirements.txt          # dépendances

# Reproduire les lots (notebooks)
python -m nbconvert --to notebook --execute --inplace notebooks/01_lot1_eda_preparation.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_lot2_representations.ipynb
python src/finetune_bert.py              # fine-tuning DistilBERT (~40 min CPU)
python -m nbconvert --to notebook --execute --inplace notebooks/03_lot3_modelisation.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/04_lot4_evaluation.ipynb

# Lancer la démo
streamlit run app/app.py                 # http://localhost:8501
```

---

## Application de démonstration (Lot 5)

Saisir un tweet → analyse en temps réel. L'app fait tourner **plusieurs modèles ensemble** :

- **DistilBERT fine-tuné** — la prédiction principale (sentiment + probabilités)
- **LogReg + TF-IDF** — carte *« Pourquoi ce verdict ? »* (mots qui ont pesé, avec traduction)
- **BiLSTM + Attention** — surlignage des mots regardés par le modèle
- **opus-mt-en-fr** — traduction française du tweet

Exemples prêts à coller : voir [`exemples_demo.txt`](exemples_demo.txt) (un cas par sentiment + un piège ironique).

> Le modèle DistilBERT fine-tuné (`artifacts/bert_finetuned/`, ~268 Mo) n'est **pas versionné**
> (trop volumineux pour Git). Le régénérer avec `python src/finetune_bert.py`. Sans lui, l'app
> bascule automatiquement sur LogReg + Attention.

---

## Structure

```
delivarables/
├── notebooks/        01→04 : EDA · représentations · modélisation · évaluation
├── app/              app.py : démo Streamlit (Lot 5)
├── src/              modules réutilisables (voir ci-dessous)
├── data_processed/   split figé train/val/test + meta.json
├── artifacts/        représentations, modèles, prédictions test (non versionné)
├── figures/          visualisations (thème dashboard)
└── exemples_demo.txt exemples de tweets pour la démo
```

**Modules `src/`** : `text_cleaning` (nettoyage) · `viz_style` + `eda_figures` (dataviz) ·
`representations` (TF-IDF / Word2Vec / BERT) · `dl_models` + `seq_utils` (modèles PyTorch) ·
`finetune_bert` · `inference` (service app) · `build_lotX_notebook` (génération des notebooks).

---

## Résultats (F1-macro, jeu de test)

| Modèle | F1-macro | Note |
|---|:--:|---|
| **DistilBERT fine-tuné** | **0.762** | meilleur (contextuel) |
| LogReg · TF-IDF | 0.742 | référence explicable, ~2 s |
| ANN · TF-IDF | 0.741 | |
| BiLSTM+Attention · Word2Vec | 0.734 | interprétable |
| CNN · Word2Vec | 0.732 | |

**Choix figés** : métrique **F1-macro** (déséquilibre 63/21/16), split stratifié 70/15/15
(`random_state=42`), pondération des classes (pas de ré-échantillonnage).

**Enseignements** : le meilleur embedding **dépend de l'architecture** (Word2Vec passe de 0.68 en
mean-pooling à 0.73 en séquentiel) ; la classe **neutre** est le point faible (F1 ~0.65) ;
TF-IDF reste imbattable en rapport perf/coût.
