"""
Pipeline de nettoyage de texte pour les tweets aeriens (US Airline Sentiment).

Module reutilisable par tous les lots du projet (EDA, embeddings, modelisation).
Chaque transformation est isolee dans une fonction pour rester testable et justifiable.

Choix de conception (justifies dans le notebook Lot 1) :
- On NE retire PAS les stopwords ici : les mots de negation ("not", "no", "never")
  sont decisifs pour le sentiment. La gestion des stopwords est laissee aux
  vectoriseurs du Lot 2 (option, et jamais sur la negation).
- Les emojis sont CONVERTIS en texte (:angry_face:) et non supprimes : ils portent
  un signal de sentiment fort.
- Les #hashtags sont conserves comme mots (on retire juste le #) car #fail, #delayed
  sont informatifs.
- Les @mentions sont retirees : ce sont des destinataires (la compagnie), pas du sentiment.
"""

import re
import html
import emoji

# --- Expressions regulieres compilees (perf + lisibilite) ---
RE_URL = re.compile(r"http\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_DIGIT = re.compile(r"\b\d+\b")
RE_NON_ALNUM = re.compile(r"[^a-z\s]")  # apres lowercase + demojize traduit en mots
RE_MULTISPACE = re.compile(r"\s+")


def demojize(text: str) -> str:
    """Convertit les emojis en tokens texte : 😡 -> :enraged_face: -> enraged face."""
    text = emoji.demojize(text, delimiters=(" ", " "))
    return text.replace("_", " ")


def clean_text(text: str, keep_hashtag_word: bool = True) -> str:
    """Applique le pipeline de nettoyage complet a un tweet.

    Etapes (ordre important) :
      1. unescape HTML (&amp; -> &)
      2. emojis -> texte
      3. minuscules
      4. suppression des URLs
      5. suppression des @mentions
      6. #hashtag -> mot (ou suppression si keep_hashtag_word=False)
      7. suppression des chiffres isoles
      8. suppression de la ponctuation / caracteres non alphabetiques
      9. normalisation des espaces
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = demojize(text)
    text = text.lower()
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    if keep_hashtag_word:
        text = RE_HASHTAG.sub(r"\1", text)
    else:
        text = RE_HASHTAG.sub(" ", text)
    text = RE_DIGIT.sub(" ", text)
    text = RE_NON_ALNUM.sub(" ", text)
    text = RE_MULTISPACE.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    # Mini-demonstration / test rapide
    exemples = [
        "@VirginAmerica you guys are the BEST!!! 😍 #love http://t.co/abc",
        "@united flight delayed AGAIN 😡😡 worst service #fail",
        "&amp; my bag is lost... @AmericanAir 2 days now",
    ]
    for t in exemples:
        print(f"AVANT : {t}")
        print(f"APRES : {clean_text(t)}\n")
