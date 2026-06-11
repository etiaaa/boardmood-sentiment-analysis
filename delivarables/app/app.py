# -*- coding: utf-8 -*-
"""Lot 5 — Application de démonstration (Streamlit).

Saisir un tweet → prédiction de sentiment en direct :
  - LogReg/TF-IDF : probabilités + mots qui ont pesé dans la décision
  - BiLSTM+Attention : probabilités + surlignage des mots selon l'attention

Lancer :  streamlit run delivarables/app/app.py
"""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "delivarables" / "src"
ART = ROOT / "delivarables" / "artifacts"
sys.path.insert(0, str(SRC))

from text_cleaning import clean_text
import inference as I

# ---------------------------------------------------------------- PALETTE
BG, PANEL, LINE, TEXT, MUTED = "#0A0E18", "#131A2A", "#222F44", "#EAF1FB", "#8593A8"
NEG, NEU, POS, CYAN, AMBER = "#FF3B6B", "#9AA6BC", "#1DE9B6", "#38BDF8", "#FFB020"
COLORS = {"negative": NEG, "neutral": NEU, "positive": POS}
EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
FR = {"negative": "Négatif", "neutral": "Neutre", "positive": "Positif"}

st.set_page_config(page_title="Sentiment tweets aériens", page_icon="✈️", layout="centered")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ---- masquer l'habillage Streamlit (anglais : Deploy, menu, footer) ---- */
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{ display:none!important; }}

.stApp {{
  background:
    radial-gradient(1100px 520px at 12% -8%, #16213b 0%, rgba(0,0,0,0) 55%),
    radial-gradient(900px 480px at 110% 0%, #1a1430 0%, rgba(0,0,0,0) 50%),
    {BG};
  color:{TEXT}; font-family:'Inter',sans-serif;
}}
.block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:760px; }}
.stApp p, .stApp label, .stApp span, .stApp div {{ color:{TEXT}; }}

/* ---- hero ---- */
.hero {{ margin:.2rem 0 1.4rem; }}
.hero h1 {{
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.5rem; margin:0; line-height:1.1;
  background:linear-gradient(92deg,{CYAN},{POS} 55%,{AMBER}); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent;
}}
.hero .sub {{ color:{MUTED}; font-size:1.0rem; margin-top:.5rem; }}
.chips span {{ display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.8rem;
  font-weight:600; margin:.45rem .35rem 0 0; border:1px solid; }}

/* ---- inputs ---- */
.stTextArea textarea {{
  background:{PANEL}!important; color:{TEXT}!important; border:1px solid {LINE}!important;
  border-radius:14px!important; font-size:1.06rem!important; padding:.9rem 1rem!important;
}}
.stTextArea textarea:focus {{ border-color:{CYAN}!important; box-shadow:0 0 0 3px rgba(56,189,248,.15)!important; }}
.stTextArea textarea::placeholder {{ color:{MUTED}!important; }}

.stButton>button {{
  background:linear-gradient(92deg,{CYAN},#0C6FA8); color:#04111d; font-weight:700; font-size:1.02rem;
  border:0; border-radius:12px; padding:.6rem 1.4rem; width:100%; transition:transform .08s ease, box-shadow .2s;
  box-shadow:0 6px 20px rgba(56,189,248,.25);
}}
.stButton>button:hover {{ transform:translateY(-1px); box-shadow:0 10px 28px rgba(56,189,248,.38); }}

/* ---- selectbox / checkbox (BaseWeb) ---- */
div[data-baseweb="select"] > div {{ background:{PANEL}!important; border:1px solid {LINE}!important; border-radius:12px!important; }}
div[data-baseweb="select"] div, div[data-baseweb="select"] span, div[data-baseweb="select"] svg {{ color:{TEXT}!important; fill:{TEXT}!important; }}
ul[role="listbox"], div[data-baseweb="popover"], div[data-baseweb="menu"] {{ background:{PANEL}!important; border:1px solid {LINE}!important; }}
li[role="option"] {{ color:{TEXT}!important; background:{PANEL}!important; }}
li[role="option"]:hover, li[aria-selected="true"] {{ background:#1E2A3A!important; color:{CYAN}!important; }}
.stCheckbox label, .stCheckbox label span, .stCheckbox label p {{ color:{TEXT}!important; }}

/* ---- cartes & éléments ---- */
.card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:1.15rem 1.3rem; margin:.8rem 0;
  box-shadow:0 8px 28px rgba(0,0,0,.28); }}
.card .h {{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:.78rem; letter-spacing:.12em;
  text-transform:uppercase; color:{MUTED}; margin-bottom:.7rem; }}
.tok {{ display:inline-block; padding:.22rem .5rem; margin:3px; border-radius:8px; font-size:1.0rem; }}
.small {{ color:{MUTED}!important; font-size:.84rem; }}
.bigbadge {{ display:inline-flex; align-items:center; gap:.55rem; padding:.55rem 1.4rem; border-radius:999px;
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.45rem; }}
hr {{ border-color:{LINE}; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    vec, clf = I.load_logreg(ART)
    model, w2i, pad = I.load_attn(ART)
    try:
        bert = I.load_bert(ART)          # (tokenizer, model) si disponible
    except Exception:
        bert = None
    try:
        translator = I.load_translator()  # traduction EN->FR
    except Exception:
        translator = None
    return vec, clf, model, w2i, pad, bert, translator


def proba_bars(probs):
    html = ""
    for s in ["negative", "neutral", "positive"]:
        pct = probs[s] * 100
        html += (
            f"<div style='display:flex;align-items:center;gap:12px;margin:.45rem 0'>"
            f"<span style='width:96px;font-weight:600'>{EMOJI[s]} {FR[s]}</span>"
            f"<div style='flex:1;background:#0A0E18;border-radius:999px;height:14px;overflow:hidden;border:1px solid {LINE}'>"
            f"<div style='width:{pct:.1f}%;height:100%;border-radius:999px;"
            f"background:linear-gradient(90deg,{COLORS[s]}55,{COLORS[s]})'></div></div>"
            f"<span style='width:54px;text-align:right;color:{COLORS[s]};font-weight:700'>{pct:.0f}%</span></div>")
    st.markdown(html, unsafe_allow_html=True)


def big_badge(pred):
    c = COLORS[pred]
    st.markdown(f"<div class='bigbadge' style='background:{c}1f;color:{c};border:1.5px solid {c}'>"
                f"{EMOJI[pred]} {FR[pred]}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- HERO
st.markdown(
    "<div class='hero'><h1>✈️ Sentiment des tweets aériens</h1>"
    "<div class='sub'>Analyse de sentiment en temps réel · projet NLP &amp; Deep Learning</div>"
    f"<div class='chips'>"
    f"<span style='color:{NEG};border-color:{NEG}55;background:{NEG}14'>😠 Négatif</span>"
    f"<span style='color:{NEU};border-color:{NEU}55;background:{NEU}14'>😐 Neutre</span>"
    f"<span style='color:{POS};border-color:{POS}55;background:{POS}14'>😊 Positif</span>"
    "</div></div>", unsafe_allow_html=True)

vec, clf, model, w2i, pad, bert, translator = load_models()

examples = {
    "— Choisir un exemple —": "",
    "Négatif (retard & service)": "@united absolutely the worst airline ever, my flight was cancelled, lost my luggage and the staff was rude and useless",
    "Positif (remerciement)": "@VirginAmerica thank you so much! amazing crew and a great flight, I really love flying with you",
    "Neutre (question)": "@AmericanAir what time does flight 123 board and which gate tomorrow morning?",
    "Ironique (piège 😏)": "@SouthwestAir oh great, another 4 hour delay, exactly what I wanted today. fantastic service as always",
}
choice = st.selectbox("Charger un exemple", list(examples.keys()))
tweet = st.text_area("Tweet à analyser", value=examples[choice], height=120,
                     placeholder="Tapez un tweet adressé à une compagnie aérienne…")

c1, c2 = st.columns([1, 1.3])
with c1:
    go = st.button("Analyser le sentiment  ✦")
with c2:
    show_attn = st.checkbox("Visualiser l'attention (deep learning)", value=True)

if go and tweet.strip():
    tc = clean_text(tweet)
    if not tc:
        st.warning("Après nettoyage, le tweet est vide (uniquement mentions / liens).")
    else:
        pr, prob, contribs = I.predict_logreg(tc, vec, clf)

        # --- traduction FR du tweet (compréhension) ---
        if translator is not None:
            fr = I.translate(tweet, *translator)
            st.markdown(f"<div class='card'><div class='h'>🇫🇷 Traduction du tweet</div>"
                        f"<p style='margin:.1rem 0;font-style:italic'>« {fr} »</p></div>",
                        unsafe_allow_html=True)

        # --- carte principale : DistilBERT si dispo, sinon LogReg ---
        if bert is not None:
            tokb, modelb = bert
            head_pred, head_prob = I.predict_bert(tweet, tokb, modelb)
            head_label = "Probabilités · DistilBERT fine-tuné 🏆"
        else:
            head_pred, head_prob = pr, prob
            head_label = "Probabilités · LogReg + TF-IDF"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        cc1, cc2 = st.columns([1, 1.25])
        with cc1:
            st.markdown("<div class='h'>Sentiment prédit</div>", unsafe_allow_html=True)
            big_badge(head_pred)
        with cc2:
            st.markdown(f"<div class='h'>{head_label}</div>", unsafe_allow_html=True)
            proba_bars(head_prob)
        st.markdown("</div>", unsafe_allow_html=True)

        pos, neg = I.explain_logreg(tc, vec, clf, head_pred)
        cpred = COLORS[head_pred]
        # traductions FR des mots (si dispo)
        pos_fr = I.translate_many([w for w, _ in pos], *translator) if (translator and pos) else [w for w, _ in pos]
        neg_fr = I.translate_many([w for w, _ in neg], *translator) if (translator and neg) else [w for w, _ in neg]

        st.markdown("<div class='card'><div class='h'>Pourquoi ce verdict ?</div>", unsafe_allow_html=True)
        if pos:
            mots = ", ".join(f"<b style='color:{cpred}'>{w}</b> ({fr})" for (w, _), fr in zip(pos[:4], pos_fr[:4]))
            st.markdown(
                f"<p style='margin:.1rem 0 .7rem'>Ce tweet est classé {EMOJI[head_pred]} "
                f"<b style='color:{cpred}'>{FR[head_pred]}</b> principalement à cause de&nbsp;: {mots}.</p>",
                unsafe_allow_html=True)
            chips = " ".join(
                f"<span class='tok' style='background:{cpred}22;color:{cpred};border:1px solid {cpred}44'>"
                f"＋ {w} <span style='opacity:.7'>· {fr}</span></span>"
                for (w, _), fr in zip(pos, pos_fr))
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown("<span class='small'>Aucun mot-clé fortement marqué : décision portée par le "
                        "contexte global (compréhension de la phrase par le modèle).</span>", unsafe_allow_html=True)
        if neg:
            st.markdown("<div class='small' style='margin-top:.8rem;margin-bottom:.2rem'>Éléments qui jouaient en sens inverse :</div>",
                        unsafe_allow_html=True)
            chips2 = " ".join(
                f"<span class='tok' style='background:{MUTED}22;color:{MUTED};border:1px solid {MUTED}44'>"
                f"－ {w} <span style='opacity:.7'>· {fr}</span></span>"
                for (w, _), fr in zip(neg, neg_fr))
            st.markdown(chips2, unsafe_allow_html=True)
        st.markdown("<div class='small' style='margin-top:.7rem'>Explication via le modèle interprétable "
                    "(TF-IDF + régression logistique).</div></div>", unsafe_allow_html=True)

        if show_attn:
            pa, proba, toks, w = I.predict_attn(tc, model, w2i, pad)
            st.markdown("<div class='card'><div class='h'>BiLSTM + Attention · où le modèle regarde</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<span class='small'>Prédiction : </span>", unsafe_allow_html=True)
            big_badge(pa)
            if toks:
                mx = max(w) or 1.0
                spans = " ".join(
                    f"<span class='tok' style='background:rgba(255,176,32,{ww/mx:.2f});"
                    f"color:{'#04111d' if ww/mx>0.55 else TEXT}'>{t}</span>"
                    for t, ww in zip(toks, w))
                st.markdown(f"<div style='margin-top:.6rem'>{spans}</div>"
                            "<div class='small' style='margin-top:.5rem'>Intensité de surlignage = poids d'attention</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
elif go:
    st.warning("Saisissez d'abord un tweet.")

_bert_note = "🏆 modèle principal de la démo" if bert is not None else "non embarqué"
st.markdown(
    f"<hr><div class='small'>F1-macro (test) — DistilBERT fine-tuné : <b style='color:{TEXT}'>0,76</b> ({_bert_note}) · "
    f"LogReg + TF-IDF : <b style='color:{TEXT}'>0,74</b> · "
    f"BiLSTM + Attention : <b style='color:{TEXT}'>0,73</b></div>",
    unsafe_allow_html=True)
