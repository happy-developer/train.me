# %% [markdown]
# # Test post-génération du modèle entraîné sur la dataset "life_style_data"

# %% [markdown]
# ## 1. Importation des librairies essentielles

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
from sklearn.preprocessing import OrdinalEncoder

# %% [markdown]
# ## 2. Exemple d’entrée simulée depuis l’interface utilisateur
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
    "Gender": "Male",               # Sexe de l'utilisateur (sera activé dans v2)
    # "Experience_Level": "Beginner", # Niveau sportif perçu
    # "Difficulty Level": "Easy",     # Difficulté de la séance choisie
}

# Vérification rapide
print("Exemple d'entrée utilisateur simulée :")
for k, v in ui_input.items():
    print(f" - {k}: {v}")


# %% [markdown]
# ## 3. Définition des chemins du modèle et des fichiers associés
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
gender_enc_fp  = model_dir / "gender_encoder.joblib"

# Vérification rapide
print("📁 Dossiers et fichiers cibles :")
print(f" - Modèle entraîné        : {model_fp}")
print(f" - Feature scaler          : {fx_scaler_fp}")
print(f" - Target scaler           : {y_scaler_fp}")
print(f" - Gender encoder         : {gender_enc_fp}")


# %% [markdown]
# ## 4. Chargement du modèle entraîné
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
# ### 4.1. Identification du modèle utilisé
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
# ## 5. Validation des entrées et prédiction
# 
# Cette cellule :
# 1. Vérifie que les variables d’entrée issues de `ui_input` correspondent bien aux colonnes attendues par le modèle (`expected`).  
# 2. Construit un `DataFrame` Pandas dans le bon ordre de colonnes et au bon format numérique.  
# 3. Réalise la prédiction à l’aide du modèle chargé et affiche la valeur estimée de **Calories_Burned**.  
# 
# Cette étape simule exactement ce qui se passera lors de l’appel depuis l’interface Gradio.
# 

# %%
fx_scaler = None
y_scaler = None
gender_encoder: OrdinalEncoder | None = None

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

if gender_enc_fp.exists():
    gender_encoder = joblib.load(gender_enc_fp)
    print(f"✔ Gender encoder chargé : {gender_enc_fp.name}")
else:
    print("ℹ️ Aucun gender_encoder.joblib trouvé — Gender_1.0 ne pourra pas être reconstruit.")


# %% [markdown]
#  ## 6. Construction des features brutes (Age, Weight, Gender_1.0)

# %%
np.set_printoptions(precision=6, suppress=True)

def _pipeline_has_scaler(p):
    return isinstance(p, Pipeline) and any(
        isinstance(step, (StandardScaler, MinMaxScaler, RobustScaler))
        for _, step in p.named_steps.items()
    )

# %%
def ui_to_internal_row(ui_dict: dict, expected_cols: list[str]) -> pd.DataFrame:
    """
    Transforme un dictionnaire d'entrée UI (Age, Weight (kg), Gender texte)
    en DataFrame aligné sur les colonnes internes attendues par le modèle
    (Age, Weight (kg), Gender_1.0).
    """
    row = {}

    for col in expected_cols:
        if col == "Gender_1.0":
            if gender_encoder is None:
                raise RuntimeError(
                    "Le modèle attend la colonne 'Gender_1.0' mais aucun gender_encoder.joblib "
                    "n'a été trouvé."
                )
            if "Gender" not in ui_dict:
                raise KeyError("Clé 'Gender' manquante dans ui_input.")
            g_str = ui_dict["Gender"]
            # OrdinalEncoder → 0.0 / 1.0 (ordre alphabétique des catégories)
            g_encoded = float(gender_encoder.transform([[g_str]])[0, 0])
            row["Gender_1.0"] = 1.0 if g_encoded == 1.0 else 0.0
        else:
            if col not in ui_dict:
                raise KeyError(f"Clé '{col}' manquante dans ui_input.")
            row[col] = ui_dict[col]

    return pd.DataFrame([row], columns=expected_cols)

# %%
# --- 1) Entrée UI → DataFrame brut (Age, Weight, Gender_1.0) ---
X_one_raw = ui_to_internal_row(ui_input, expected)
print("🔎 X_one_raw:\n", X_one_raw)

# --- 2) Scaling des features ---
uses_internal_scaling = _pipeline_has_scaler(model)
if uses_internal_scaling:
    X_one = X_one_raw.copy()
    print("🔧 Pipeline: scaler interne → pas de double-scaling.")
else:
    if fx_scaler is None:
        raise RuntimeError(
            "Pas de scaler interne dans le modèle et aucun feature_scaler.joblib trouvé."
        )
    X_one = pd.DataFrame(fx_scaler.transform(X_one_raw), columns=expected)
    print("🔧 Scaling appliqué via feature_scaler.joblib.")

# --- 3) Infos scaler (diagnostic léger) ---
print("🔧 feature_scaler.mean_ :", getattr(fx_scaler, "mean_", None))
print("🔧 feature_scaler.scale_:", getattr(fx_scaler, "scale_", None))

# Vérif cohérence (Z-score manuel vs transform)
z_manual = (X_one_raw - fx_scaler.mean_) / fx_scaler.scale_
z_auto   = pd.DataFrame(fx_scaler.transform(X_one_raw), columns=expected)
print("🧪 Z-manual:\n", z_manual)
print("🧪 Z-auto  :\n", z_auto)

# --- 4) Prédiction pour 1 échantillon ---
y_std_one = float(model.predict(X_one))
if y_scaler is not None:
    y_kcal_one = float(y_scaler.inverse_transform(np.array([[y_std_one]]))[0, 0])
    print(f"🔮 UI → y_std: {y_std_one:.6f} | y_kcal: {y_kcal_one:.2f}")
else:
    print(f"🔮 UI → y_std: {y_std_one:.6f} (pas de target_scaler)")


# %% [markdown]
# ### 7.1. Batch de test (2 profils)

# %%
batch_ui = [
    {"Age": 34, "Weight (kg)": 115.0, "Gender": "Male"},
    {"Age": 52, "Weight (kg)": 60.0,  "Gender": "Female"},
]

print(f"⚙️ Entrées UI : {batch_ui}")

batch_raw_rows = [ui_to_internal_row(row, expected).iloc[0].to_dict() for row in batch_ui]
batch_raw = pd.DataFrame(batch_raw_rows, columns=expected)

if uses_internal_scaling:
    batch_scaled = batch_raw
else:
    batch_scaled = pd.DataFrame(fx_scaler.transform(batch_raw), columns=expected)

print("🧾 expected       :", expected)
print("🧾 batch columns  :", list(batch_scaled.columns))

# Prédictions batch
y_std = model.predict(batch_scaled)
print("📈 batch y_std:", np.round(y_std, 6))

if y_scaler is not None:
    y_kcal = y_scaler.inverse_transform(np.array(y_std).reshape(-1, 1)).ravel()
    print("📈 batch y_kcal:", np.round(y_kcal, 2))

# Garde-fou : détecter une prédiction constante
if len(set(np.round(y_std, 6))) == 1:
    print("⚠️ Alerte: prédiction constante sur ce batch. "
          "Vérifier mean_/scale_ et la variabilité des features.")


