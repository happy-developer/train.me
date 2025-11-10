import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

def set_trainme_theme(context="talk", font_scale=1.0, use_dark=True):
    """
    Applique un thème Seaborn/Matplotlib cohérent avec l'identité TrAIn.me.
    - context: "paper" | "notebook" | "talk" | "poster"
    - font_scale: facteur d'échelle des polices
    - use_dark: True => fond sombre (recommandé pour le logo)
    """
    # Palette inspirée du logo (néons bleus/cyans)
    TRAINME_PALETTE = ["#00E5FF", "#00BFFF", "#0A84FF", "#1E90FF", "#00FFFF", "#64D8FF"]

    if use_dark:
        # Couleurs “dark” élégantes
        bg      = "#0B1020"   # fond principal
        axface  = "#111830"   # fond axes
        grid    = "#2A345B"   # grille discrète
        spine   = "#5C6B94"   # bord d’axes
        text    = "#E6F2FF"   # texte clair
    else:
        # Variante claire
        bg      = "#FFFFFF"
        axface  = "#F7FAFF"
        grid    = "#DCE6F8"
        spine   = "#6C7AA6"
        text    = "#0B1020"

    # Seaborn (style + palette + contexte)
    sns.set_theme(
        context=context,
        style="darkgrid" if use_dark else "whitegrid",
        palette=TRAINME_PALETTE,
        font="DejaVu Sans",
        font_scale=font_scale,
    )

    # Matplotlib rcParams (fond, grille, typographie, cycle de couleurs)
    plt.rcParams.update({
        # Fonds
        "figure.facecolor": bg,
        "axes.facecolor": axface,
        "savefig.facecolor": bg,

        # Titres & textes
        "text.color": text,
        "axes.labelcolor": text,
        "axes.titlecolor": text,

        # Axes & ticks
        "axes.edgecolor": spine,
        "xtick.color": text,
        "ytick.color": text,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Grille
        "axes.grid": True,
        "grid.color": grid,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.grid.axis": "y",      # grille uniquement en Y (plus propre)
        "axes.grid.which": "major",

        # Lignes/markers
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,

        # Légende
        "legend.frameon": False,

        # Cycler de couleurs (sécurise l’usage côté Matplotlib natif)
        "axes.prop_cycle": cycler(color=TRAINME_PALETTE),
    })

    # Retourne la palette si besoin
    return TRAINME_PALETTE