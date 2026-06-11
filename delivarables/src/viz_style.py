# -*- coding: utf-8 -*-
"""Systeme de design 'dashboard sombre' pour les visualisations du projet.

Palette neon sur fond spatial, polices condensees (Bahnschrift) + Segoe UI,
helpers : glow, barres a degrade, cartes KPI, donut. Importer `apply_theme()`
au debut d'un notebook pour un rendu homogene et moderne.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- PALETTE
BG      = "#0B0F19"   # fond (deep space navy)
PANEL   = "#121826"   # cartes / panneaux
GRID    = "#1E2A3A"   # grille discrete
TEXT    = "#E8EEF6"   # texte principal
MUTED   = "#7C8AA0"   # texte secondaire

NEG     = "#FF3B6B"   # negatif  - neon framboise
NEU     = "#8B98AE"   # neutre   - slate
POS     = "#1DE9B6"   # positif  - mint
CYAN    = "#38BDF8"   # accent
PURPLE  = "#B388FF"
AMBER   = "#FFB020"

SENT_COLORS = {"negative": NEG, "neutral": NEU, "positive": POS}

# degrades (cmaps) pour barres futuristes
def _cmap(c0, c1, name):
    return LinearSegmentedColormap.from_list(name, [c0, c1])

CMAP_NEG  = _cmap("#7A1030", NEG,   "neg")
CMAP_POS  = _cmap("#0A5C49", POS,   "pos")
CMAP_CYAN = _cmap("#0C3B66", CYAN,  "cyan")
CMAP_AMB  = _cmap("#7A4D00", AMBER, "amb")
CMAP_PUR  = _cmap("#3B1E6E", PURPLE,"pur")

TITLE_FONT = "Bahnschrift"
BODY_FONT  = "Segoe UI"


def apply_theme():
    """Applique le theme sombre global a matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG,
        "savefig.facecolor": BG,
        "font.family":      BODY_FONT,
        "text.color":       TEXT,
        "axes.labelcolor":  MUTED,
        "axes.edgecolor":   GRID,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "axes.grid":        True,
        "grid.color":       GRID,
        "grid.linewidth":   0.8,
        "grid.alpha":       0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.spines.bottom":False,
        "figure.dpi":       130,
    })


def title(ax, t, sub=None, size=20):
    """Titre condense + sous-titre discret, aligne a gauche."""
    ax.text(0, 1.16, t, transform=ax.transAxes, fontfamily=TITLE_FONT,
            fontsize=size, fontweight="bold", color=TEXT, va="top")
    if sub:
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontfamily=BODY_FONT,
                fontsize=10.5, color=MUTED, va="top")


def glow_line(ax, x, y, color, lw=2.2, n=6, base_alpha=0.06, **kw):
    """Trace une ligne avec halo lumineux (effet neon)."""
    for i in range(n, 0, -1):
        ax.plot(x, y, color=color, lw=lw + i*2.2, alpha=base_alpha,
                solid_capstyle="round", zorder=2)
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=3, **kw)


def gradient_barh(ax, y, width, cmap, height=0.62, zorder=3):
    """Barre horizontale remplie d'un degrade (gauche->droite)."""
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=[0, width, y - height/2, y + height/2],
              aspect="auto", cmap=cmap, zorder=zorder, vmin=0, vmax=1)
    # cap lumineux a l'extremite
    ax.scatter([width], [y], s=70, color="white", zorder=zorder+1,
               edgecolors="none", alpha=0.85)


def kpi_card(ax, value, label, accent=CYAN):
    """Dessine une carte KPI (grand nombre + libelle) dans un axe vide."""
    ax.axis("off")
    card = FancyBboxPatch((0.04, 0.10), 0.92, 0.80,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          mutation_aspect=0.5, linewidth=1.2,
                          edgecolor=GRID, facecolor=PANEL, transform=ax.transAxes)
    ax.add_patch(card)
    ax.add_patch(FancyBboxPatch((0.04, 0.10), 0.025, 0.80,
                 boxstyle="round,pad=0,rounding_size=0.02",
                 linewidth=0, facecolor=accent, transform=ax.transAxes, zorder=3))
    ax.text(0.12, 0.60, value, transform=ax.transAxes, fontfamily=TITLE_FONT,
            fontsize=26, fontweight="bold", color=TEXT, va="center")
    ax.text(0.12, 0.28, label.upper(), transform=ax.transAxes, fontfamily=BODY_FONT,
            fontsize=9.5, color=MUTED, va="center", fontweight="bold")


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
