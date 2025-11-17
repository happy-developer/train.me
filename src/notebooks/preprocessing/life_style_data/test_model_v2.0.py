# %% [markdown]
# # Test post-génération du modèle entraîné sur la dataset "life_style_data"

# %% [markdown]
# ## Importation des librairies essentielles

# %%
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# %% [markdown]
# ## Exemple d’entrée simulée depuis l’interface utilisateur
# 
# Cette cellule permet de définir un dictionnaire Python (`ui_input`) reproduisant la structure exacte des données envoyées par l’interface Gradio. 
# 
# Chaque clé correspond au **nom d’une feature d’entrée** utilisée par le modèle.  
# 
# Nous n’activons pour l’instant que les variables *Age* et *Weight (kg)* — les autres seront intégrées dans les prochaines versions du pipeline (`Gender`, `Experience_Level`, etc.).
# 

# %%
# === Simulation d'une entrée utilisateur (depuis l'UI) ===
# Ce dictionnaire correspond exactement à ce que l'application Gradio envoie au backend
# Chaque clé doit correspondre à une colonne vue par le modèle lors de l'entraînement

# ui_input = {
#     "Age": 34,            # Âge de l'utilisateur (en années)
#     "Weight (kg)": 115.0, # Poids corporel (en kilogrammes)
    
#     # Champs actuellement désactivés dans le modèle v1 :
#     # "Gender": "Male",               # Sexe de l'utilisateur (sera activé dans v2)
#     # "Experience_Level": "Beginner", # Niveau sportif perçu
#     # "Difficulty Level": "Easy",     # Difficulté de la séance choisie
# }

ui_input = {
    "Age": 52,            # Âge de l'utilisateur (en années)
    "Weight (kg)": 60.0, # Poids corporel (en kilogrammes)
    
    # Champs actuellement désactivés dans le modèle v1 :
    # "Gender": "Male",               # Sexe de l'utilisateur (sera activé dans v2)
    # "Experience_Level": "Beginner", # Niveau sportif perçu
    # "Difficulty Level": "Easy",     # Difficulté de la séance choisie
}

# Vérification rapide
print("Exemple d'entrée utilisateur simulée :")
for k, v in ui_input.items():
    print(f" - {k}: {v}")


# %% [markdown]
# ## Définition des chemins du modèle et des fichiers associés
# 
# Cette cellule identifie dynamiquement la racine du projet `train.me` à partir du répertoire courant,  
# puis construit les chemins complets vers :
# - le modèle entraîné (`model.joblib`)  
# - le scaler des features (`feature_scaler.joblib`)  
# - le scaler de la variable cible (`target_scaler.joblib`)
# 
# Cela garantit que le notebook reste portable, même si le dossier est déplacé.
# 

# %%
# === Localisation dynamique des fichiers du modèle ===
# On part du dossier actuel (celui du notebook)
current_dir = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()

# Remonte jusqu’à la racine du projet "train.me"
project_root = current_dir.parents[3]
print(f"📂 Racine du projet détectée : {project_root}")

# Définition des chemins vers le modèle et les objets de scaling
model_dir = project_root / "src" / "models" / "v1" / "life_style_data"

# Fichiers du pipeline ML
model_fp      = model_dir / "model.joblib"           # Modèle entraîné complet (pipeline)
fx_scaler_fp  = model_dir / "feature_scaler.joblib"  # Scaler utilisé pour normaliser les features
y_scaler_fp   = model_dir / "target_scaler.joblib"   # Scaler utilisé pour rescaler la target

# Vérification rapide
print("📁 Dossiers et fichiers cibles :")
print(f" - Modèle entraîné        : {model_fp}")
print(f" - Feature scaler          : {fx_scaler_fp}")
print(f" - Target scaler           : {y_scaler_fp}")


# %% [markdown]
# ## Chargement du modèle entraîné
# 
# Cette cellule charge le modèle sauvegardé lors de la phase d’entraînement.  
# 
# Le fichier `model.joblib` contient un pipeline complet (prétraitement, transformation, modèle).  
# 
# On récupère ensuite la liste des **features attendues** par ce modèle — utile pour vérifier la correspondance avec les données d’entrée simulées.
# 

# %%
# Chargement du modèle
model = joblib.load(model_fp)

# Récupération robuste des features attendues
expected = None
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
else:
    # fallback via feature_schema.json
    feature_schema_fp = model_fp.parent / "feature_schema.json"
    if feature_schema_fp.exists():
        with open(feature_schema_fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected = data.get("feature_names_in_")
    if not expected:
        raise AttributeError(
            "Impossible de déterminer les colonnes attendues : ni "
            "`model.feature_names_in_` ni `feature_schema.json` trouvés."
        )

print(f"✔ Modèle chargé : {model_fp.name}")
print(f"→ Features attendues ({len(expected)}) : {expected}")

# %% [markdown]
# ### Identification du modèle utilisé
# 
# Cette cellule permet d’afficher le type de modèle réellement chargé dans le pipeline.  
# 
# Selon la configuration du `model.joblib`, il peut s’agir d’une **régression linéaire**, d’un **Random Forest**, ou d’un **Gradient Boosting**.  
# 
# La fonction `model_friendly()` renvoie un nom lisible pour faciliter l’interprétation du rapport ou du log.
# 

# %%
def model_friendly(estimator):
    """
    Retourne un nom lisible du modèle entraîné.
    
    Paramètres
    ----------
    estimator : objet scikit-learn
        Modèle ou pipeline entraîné.

    Retour
    ------
    str : nom du modèle en format humain
    """
    if isinstance(estimator, RandomForestRegressor):
        return "Random Forest"
    if isinstance(estimator, GradientBoostingRegressor):
        return "Gradient Boosting"
    if isinstance(estimator, LinearRegression):
        return "Régression Linéaire"
    return estimator.__class__.__name__

# === Affichage du type de modèle ===
print(f"⚙️  Type de modèle : {model_friendly(model)}")


# %% [markdown]
# ## Validation des entrées et prédiction
# 
# Cette cellule :
# 1. Vérifie que les variables d’entrée issues de `ui_input` correspondent bien aux colonnes attendues par le modèle (`expected`).  
# 2. Construit un `DataFrame` Pandas dans le bon ordre de colonnes et au bon format numérique.  
# 3. Réalise la prédiction à l’aide du modèle chargé et affiche la valeur estimée de **Calories_Burned**.  
# 
# Cette étape simule exactement ce qui se passera lors de l’appel depuis l’interface Gradio.
# 

# %%
# # === 1️⃣ Vérification de la cohérence des entrées ===
# # On s'assure que toutes les colonnes attendues par le modèle
# # sont bien présentes dans le dictionnaire ui_input.
# print(f"⚙️ Entrées : {ui_input}")
# missing = [c for c in expected if c not in ui_input]
# if missing:
#     raise ValueError(f"⚠️ Champs manquants dans l'entrée UI : {missing}")

# # === 2️⃣ Construction du DataFrame pour la prédiction ===
# # L'ordre des colonnes doit être strictement identique à celui du modèle.
# X_one = pd.DataFrame([[ui_input[c] for c in expected]], columns=expected)

# # Forcer le typage numérique pour éviter les erreurs de prédiction
# X_one = X_one.apply(pd.to_numeric, errors="raise")

# # === 3️⃣ Prédiction ===
# # Le modèle ayant été entraîné sans normalisation/scaling à l'inférence,
# # la sortie correspond directement à la valeur réelle en Calories.
# y_pred = float(model.predict(X_one).squeeze())

# # === 4️⃣ Affichage du résultat ===
# print(f"🔮 Calories_Burned (réelles) : {y_pred:.2f}")


# %%
fx_scaler = None
y_scaler = None

# Chargement "best effort"
if Path(fx_scaler_fp).exists():
    fx_scaler = joblib.load(fx_scaler_fp)
    print(f"✔ Feature scaler chargé : {Path(fx_scaler_fp).name}")
else:
    print("ℹ️ Aucun feature scaler trouvé (on supposera un Pipeline intégrant le scaler).")

if Path(y_scaler_fp).exists():
    y_scaler = joblib.load(y_scaler_fp)
    print(f"✔ Target scaler chargé : {Path(y_scaler_fp).name}")
else:
    print("ℹ️ Aucun target scaler trouvé (cible non normalisée).")


# %%
print(f"⚙️ Entrées : {ui_input}")
# 1) Construire la ligne brute dans l'ordre attendu
X_one_raw = pd.DataFrame([[ui_input[c] for c in expected]], columns=expected).apply(pd.to_numeric, errors="raise")

# 2) Déterminer si le modèle intègre déjà le scaler (Pipeline)
def _pipeline_has_scaler(p):
    if not isinstance(p, Pipeline):
        return False
    scaler_types = (StandardScaler, MinMaxScaler, RobustScaler)
    return any(isinstance(step, scaler_types) for _, step in p.named_steps.items())

uses_internal_scaling = _pipeline_has_scaler(model)

# Appliquer le scaling uniquement si nécessaire
if uses_internal_scaling:
    X_one = X_one_raw.copy()
    print("🔧 Scaling interne détecté dans le Pipeline du modèle ➜ pas de double-scaling.")
else:
    if fx_scaler is None:
        raise RuntimeError("Le modèle n'intègre pas de scaler et aucun feature_scaler.joblib n'a été trouvé.")
    X_scaled = fx_scaler.transform(X_one_raw)
    X_one = pd.DataFrame(X_scaled, columns=expected)
    print("🔧 Scaling appliqué via feature_scaler.joblib.")


np.set_printoptions(precision=6, suppress=True)

print("🔎 X_one_raw:\n", X_one_raw)

# Affiche les stats du scaler
print("🔧 feature_scaler.mean_:", getattr(fx_scaler, "mean_", None))
print("🔧 feature_scaler.scale_:", getattr(fx_scaler, "scale_", None))

# Compare transformation manuelle vs transform()
z_manual = (X_one_raw.values - fx_scaler.mean_) / fx_scaler.scale_
z_auto   = fx_scaler.transform(X_one_raw.values)
print("🧪 Z-manual:", z_manual)
print("🧪 Z-auto  :", z_auto)


# Deux entrées très différentes
batch_raw = pd.DataFrame(
    [{"Age": 34, "Weight (kg)": 115.0},
     {"Age": 52, "Weight (kg)": 60.0}],
    columns=expected
).apply(pd.to_numeric, errors="raise")

if fx_scaler is not None:
    batch = pd.DataFrame(fx_scaler.transform(batch_raw), columns=expected)
else:
    batch = batch_raw.copy()

y_std = model.predict(batch)
print("📈 batch y_std:", np.round(y_std, 6))

# Si y_scaler présent, reviens en kcal
if y_scaler is not None:
    y_kcal = y_scaler.inverse_transform(y_std.reshape(-1,1)).ravel()
    print("📈 batch y_kcal:", np.round(y_kcal, 2))

print("🧾 expected:", expected)
print("🧾 batch columns:", list(batch.columns))
assert list(batch.columns) == list(expected), "Désalignement colonnes → prédictions aberrantes possibles"

batch_raw = pd.DataFrame(
    [{"Age": 34, "Weight (kg)": 115.0},
     {"Age": 52, "Weight (kg)": 60.0}],
    columns=expected
).apply(pd.to_numeric)

batch = pd.DataFrame(fx_scaler.transform(batch_raw), columns=expected)
y_std = model.predict(batch)
y_kcal = y_scaler.inverse_transform(y_std.reshape(-1,1)).ravel()

print("📈 y_std :", np.round(y_std, 4))
print("📈 y_kcal:", np.round(y_kcal, 2))



