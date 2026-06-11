# -*- coding: utf-8 -*-
"""Genere les visualisations EDA 'dashboard sombre' du Lot 1."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from wordcloud import WordCloud

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import viz_style as vz
from text_cleaning import clean_text

vz.apply_theme()

ROOT = SRC.parents[1]
DATA = ROOT / "data" / "Tweets.csv"
FIG = ROOT / "delivarables" / "figures"; FIG.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(DATA)
df["text_clean"] = df["text"].apply(clean_text)
df["n_mots"] = df["text"].str.split().str.len()
ordre = ["negative", "neutral", "positive"]

# =====================================================================
# FIGURE 1 — HERO DASHBOARD (KPIs + donut + diverging par compagnie)
# =====================================================================
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(vz.BG)
gs = GridSpec(3, 4, figure=fig, height_ratios=[0.5, 1.0, 1.4],
              hspace=0.55, wspace=0.35, left=0.06, right=0.96, top=0.90, bottom=0.07)

# --- bandeau titre
fig.text(0.06, 0.955, "AIRLINE SENTIMENT — DATA OVERVIEW", fontfamily=vz.TITLE_FONT,
         fontsize=30, fontweight="bold", color=vz.TEXT)
fig.text(0.06, 0.915, "14 640 tweets · 6 compagnies aériennes US · annotés négatif / neutre / positif",
         fontfamily=vz.BODY_FONT, fontsize=12, color=vz.MUTED)

# --- KPI cards
n = len(df)
kpis = [
    (f"{n:,}".replace(",", " "), "Tweets analysés", vz.CYAN),
    (f"{(df.airline_sentiment=='negative').mean()*100:.0f}%", "Part de négatif", vz.NEG),
    ("6", "Compagnies", vz.PURPLE),
    (f"{df.airline_sentiment_confidence.mean():.2f}", "Confiance annot.", vz.POS),
]
for i, (val, lab, acc) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i]); vz.kpi_card(ax, val, lab, acc)

# --- DONUT distribution des classes (gs row1 col0-1)
axd = fig.add_subplot(gs[1:, 0:2])
counts = df["airline_sentiment"].value_counts().reindex(ordre)
cols = [vz.SENT_COLORS[s] for s in ordre]
wedges, _ = axd.pie(counts.values, colors=cols, startangle=90, counterclock=False,
                    wedgeprops=dict(width=0.42, edgecolor=vz.BG, linewidth=3))
# halo
for w in wedges:
    w.set_zorder(3)
axd.text(0, 0.12, f"{n:,}".replace(",", " "), ha="center", va="center",
         fontfamily=vz.TITLE_FONT, fontsize=34, fontweight="bold", color=vz.TEXT)
axd.text(0, -0.16, "TWEETS", ha="center", va="center", fontfamily=vz.BODY_FONT,
         fontsize=12, color=vz.MUTED, fontweight="bold")
# legende custom avec % (sous le donut, horizontale -> pas de chevauchement)
labels_fr = {"negative": "Négatif", "neutral": "Neutre", "positive": "Positif"}
for i, s in enumerate(ordre):
    pct = counts[s]/n*100
    x0 = 0.06 + i*0.33
    axd.text(x0, -0.06, "●", color=vz.SENT_COLORS[s], fontsize=16,
             transform=axd.transAxes, ha="left", va="center")
    axd.text(x0+0.045, -0.06, labels_fr[s], color=vz.TEXT, fontsize=11.5,
             transform=axd.transAxes, ha="left", va="center", fontfamily=vz.BODY_FONT)
    axd.text(x0+0.045, -0.115, f"{pct:.1f}%", color=vz.SENT_COLORS[s], fontsize=13,
             transform=axd.transAxes, ha="left", va="center", fontweight="bold")
axd.set_title("Répartition des sentiments", fontfamily=vz.TITLE_FONT, fontsize=16,
              color=vz.TEXT, loc="left", pad=18)

# --- DIVERGING par compagnie (gs row1 col2-3) : likert centre sur le neutre
axb = fig.add_subplot(gs[1:, 2:4])
ct = pd.crosstab(df["airline"], df["airline_sentiment"], normalize="index")*100
ct = ct.reindex(columns=ordre)
ct = ct.loc[ct["negative"].sort_values().index]   # plus negatif en haut
ys = np.arange(len(ct))
for j, (airline, row) in enumerate(ct.iterrows()):
    neg, neu, pos = row["negative"], row["neutral"], row["positive"]
    axb.barh(j, -neg, left=-neu/2, color=vz.NEG, height=0.62, zorder=3)
    axb.barh(j, neu, left=-neu/2, color=vz.NEU, height=0.62, zorder=3, alpha=0.85)
    axb.barh(j, pos, left=neu/2, color=vz.POS, height=0.62, zorder=3)
    axb.text(-neu/2-neg-1.5, j, f"{neg:.0f}%", ha="right", va="center",
             color=vz.NEG, fontsize=10, fontweight="bold")
    axb.text(neu/2+pos+1.5, j, f"{pos:.0f}%", ha="left", va="center",
             color=vz.POS, fontsize=10, fontweight="bold")
axb.axvline(0, color=vz.MUTED, lw=1, ls=(0,(3,3)), alpha=0.6, zorder=1)
axb.set_yticks(ys); axb.set_yticklabels(ct.index, color=vz.TEXT, fontsize=11)
axb.set_xticks([])
axb.set_xlim(-95, 60)
axb.grid(False)
axb.set_title("Sentiment par compagnie", fontfamily=vz.TITLE_FONT, fontsize=16,
              color=vz.TEXT, loc="left", pad=28)
# legende directionnelle (Segoe UI gere les fleches)
axb.text(0.0, 1.06, "◄ négatif", transform=axb.transAxes, color=vz.NEG, fontsize=11,
         fontfamily=vz.BODY_FONT, fontweight="bold", ha="left", va="bottom")
axb.text(0.5, 1.06, "neutre", transform=axb.transAxes, color=vz.NEU, fontsize=11,
         fontfamily=vz.BODY_FONT, ha="center", va="bottom")
axb.text(1.0, 1.06, "positif ►", transform=axb.transAxes, color=vz.POS, fontsize=11,
         fontfamily=vz.BODY_FONT, fontweight="bold", ha="right", va="bottom")
vz.savefig(fig, FIG/"hero_dashboard.png"); plt.close(fig)
print("OK hero_dashboard.png")

# =====================================================================
# FIGURE 2 — MOTIFS NEGATIFS (lollipop degrade + glow)
# =====================================================================
reasons = (df.loc[df.airline_sentiment=="negative","negativereason"]
           .value_counts().head(10).sort_values())
fig, ax = plt.subplots(figsize=(12, 7))
ys = np.arange(len(reasons))
maxv = reasons.max()
for y, (lab, v) in zip(ys, reasons.items()):
    frac = v/maxv
    color = plt.get_cmap(vz.CMAP_NEG)(0.35 + 0.65*frac)
    vz.glow_line(ax, [0, v], [y, y], color=color, lw=4, n=4, base_alpha=0.05)
    ax.scatter(v, y, s=260*frac+90, color=color, zorder=5, edgecolors="white", linewidths=0.8)
    ax.text(v+maxv*0.02, y, f"{v:,}".replace(",", " "), va="center", ha="left",
            color=vz.TEXT, fontsize=11, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([l.replace(" ", "\n") if len(l)>16 else l for l in reasons.index],
                                       color=vz.TEXT, fontsize=10)
ax.set_xlim(0, maxv*1.15); ax.set_xticks([]); ax.grid(False)
vz.title(ax, "MOTIFS D'INSATISFACTION", "Top 10 des raisons citées dans les tweets négatifs", size=22)
vz.savefig(fig, FIG/"motifs_negatifs_modern.png"); plt.close(fig)
print("OK motifs_negatifs_modern.png")

# =====================================================================
# FIGURE 3 — RIDGELINE longueur des tweets par sentiment
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6.5))
xs = np.linspace(0, 38, 400)
offset = 0.0
step = 0.9
for i, s in enumerate(ordre):
    data = df.loc[df.airline_sentiment==s, "n_mots"].dropna()
    kde = gaussian_kde(data)
    dens = kde(xs); dens = dens/dens.max()
    base = (len(ordre)-1-i)*step
    c = vz.SENT_COLORS[s]
    ax.fill_between(xs, base, base+dens, color=c, alpha=0.22, zorder=i*2)
    vz.glow_line(ax, xs, base+dens, color=c, lw=2.4, n=5, base_alpha=0.05)
    med = data.median()
    ax.text(0.5, base+0.08, s.upper(), color=c, fontsize=12, fontweight="bold",
            fontfamily=vz.TITLE_FONT)
    ax.scatter(med, base+kde(med)/kde(xs).max()*0+0.02, s=0)  # placeholder
    ax.text(34, base+0.55, f"médiane\n{med:.0f} mots", color=vz.MUTED, fontsize=9,
            ha="center", va="center")
ax.set_yticks([]); ax.set_xlabel("Nombre de mots par tweet", color=vz.MUTED, fontsize=11)
ax.grid(axis="x", alpha=0.3); ax.set_xlim(0, 38)
vz.title(ax, "LONGUEUR DES TWEETS", "Distribution du nombre de mots selon le sentiment (ridgeline)", size=22)
vz.savefig(fig, FIG/"longueur_ridgeline.png"); plt.close(fig)
print("OK longueur_ridgeline.png")

# =====================================================================
# FIGURE 4 — WORDCLOUDS sur fond sombre
# =====================================================================
from matplotlib.colors import LinearSegmentedColormap
STOP = set('''a an the and or but if while is are was were be been being to of in on for with at by
from this that these those it its as i you he she we they me my your our their not no nor s t m re ve
ll d u im'''.split())
cmaps = {"negative": vz.CMAP_NEG, "neutral": LinearSegmentedColormap.from_list("n",["#5C6B82",vz.NEU]),
         "positive": vz.CMAP_POS}
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
fig.patch.set_facecolor(vz.BG)
for ax, s in zip(axes, ordre):
    txt = " ".join(df.loc[df.airline_sentiment==s, "text_clean"])
    wc = WordCloud(width=720, height=460, background_color=vz.BG, stopwords=STOP,
                   colormap=cmaps[s], max_words=70, prefer_horizontal=0.95,
                   relative_scaling=0.45, min_font_size=8).generate(txt)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title(s.upper(), color=vz.SENT_COLORS[s], fontfamily=vz.TITLE_FONT,
                 fontsize=18, fontweight="bold", pad=10)
fig.suptitle("VOCABULAIRE PAR SENTIMENT", x=0.09, ha="left", color=vz.TEXT,
             fontfamily=vz.TITLE_FONT, fontsize=22, fontweight="bold", y=1.02)
vz.savefig(fig, FIG/"nuages_mots_modern.png"); plt.close(fig)
print("OK nuages_mots_modern.png")

# =====================================================================
# FIGURE 5 — CONFIANCE D'ANNOTATION (hist degrade + kde glow)
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6))
vals = df["airline_sentiment_confidence"].dropna()
counts_h, edges = np.histogram(vals, bins=28, range=(0.3, 1.0))
centers = (edges[:-1]+edges[1:])/2
cmap = plt.get_cmap(vz.CMAP_CYAN)
for c, h, w in zip(centers, counts_h, np.diff(edges)):
    ax.bar(c, h, width=w*0.92, color=cmap((c-0.3)/0.7), zorder=3, alpha=0.95)
kde = gaussian_kde(vals); xs = np.linspace(0.3, 1.0, 300)
dens = kde(xs); dens = dens/dens.max()*counts_h.max()
vz.glow_line(ax, xs, dens, color=vz.CYAN, lw=2.6, n=6, base_alpha=0.05)
ax.axvline(0.6, color=vz.AMBER, lw=1.6, ls=(0,(4,3)), zorder=4)
ax.text(0.6, counts_h.max()*0.95, " seuil 0.6", color=vz.AMBER, fontsize=10, va="top")
low = (vals<0.6).mean()*100
ax.text(0.32, counts_h.max()*0.9, f"{low:.1f}%\nsous 0.6", color=vz.MUTED, fontsize=11, va="top")
ax.set_xlim(0.3, 1.0); ax.set_yticks([]); ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Confiance de l'annotation", color=vz.MUTED, fontsize=11)
vz.title(ax, "FIABILITÉ DES ANNOTATIONS", "Distribution de la confiance (moyenne 0.90)", size=22)
vz.savefig(fig, FIG/"confiance_modern.png"); plt.close(fig)
print("OK confiance_modern.png")

print("\\nToutes les figures generees dans", FIG)
