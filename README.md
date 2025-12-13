# TrAIn.me — AI-Powered Personal Fitness Coach

TrAIn.me est un projet d’IA appliquée visant à concevoir un **coach sportif personnel augmenté par l’intelligence artificielle**, accessible via une application interactive.  
Le projet combine **Machine Learning supervisé** et **Deep Learning / NLP** afin d’évaluer un profil sportif et de générer automatiquement des programmes d’entraînement personnalisés.

Ce projet a été développé dans le cadre de la certification **RNCP38616 – Développeur IA (Alyra)**.

---

## Objectifs du projet

- Évaluer automatiquement le **niveau sportif** d’un utilisateur à partir de données physiologiques et comportementales.
- Proposer des **recommandations d’entraînement personnalisées**.
- Générer dynamiquement des **programmes sportifs structurés** adaptés aux objectifs, contraintes et équipements disponibles.
- Démontrer une **chaîne complète IA** : données → modèles → déploiement → interface utilisateur.

---

## Structure du projet

TRAIN.ME/<br/>
├── .conda/ ..................... Environnement conda local<br/>
├── .vscode/ .................... Configuration VS Code<br/>
├── src/<br/>
│   ├── config/ ................. Paramètres globaux et constantes<br/>
│   ├── data/ ................... Données brutes, intermédiaires et transformées<br/>
│   ├── gradio/ ................. Interfaces utilisateur ML & DL<br/>
│   ├── logs/ ................... Logs d'exécution et d'entraînement<br/>
│   ├── models/ ................. Modèles ML/DL et artefacts<br/>
│   ├── notebooks/ .............. EDA, ML, DL, expérimentations<br/>
│   ├── scripts/ ................. Scripts d'entraînement et d'inférence<br/>
│   ├── tests/ .................. Tests unitaires<br/>
│   ├── utils/ .................. Fonctions utilitaires partagées<br/>
│   └── main.py ................. Point d’entrée de l’application<br/>
├── .gitignore<br/>
├── desktop.ini<br/>
├── README.md<br/>
└── requirements.txt<br/>

## Architecture globale

Le projet repose sur deux briques complémentaires :

### 1. Machine Learning — Évaluation du profil utilisateur
Objectif : estimer le **Experience_Level** d’un utilisateur.

- Type : Régression supervisée
- Données : ~20 000 profils sportifs
- Features principales :
  - âge, sexe, taille, poids
  - fréquence d’entraînement
  - type de workout
  - données physiologiques dérivées (BMI, BPM, etc.)

**Algorithmes testés :**
- Régression linéaire
- Arbres de décision
- Random Forest
- Gradient Boosting
- Bagging Regressor
- K-NN (comparatif)

**Modèle retenu :**
- Random Forest Regressor  
- R² ≈ 0.99  
- MAE et RMSE très faibles sur le jeu de validation

Les artefacts du modèle (scalers, modèle, schéma de features) sont versionnés et utilisés en inférence via une interface Gradio.

---

### 2. Deep Learning / NLP — Génération de programmes sportifs
Objectif : **générer automatiquement des programmes d’entraînement textuels** à partir d’un profil utilisateur.

- Données :
  - corpus de ~2 600 programmes sportifs structurés
  - descriptions textuelles, niveaux, objectifs, équipements
- Prétraitement :
  - nettoyage textuel
  - tokenisation
  - création de séquences
  - padding

**Modèles explorés :**
- LSTM (RNN amélioré)
- Transformer (attention)
- GPT-2 (HuggingFace)

**Approches utilisées :**
- Fine-tuning de GPT-2
- Distillation (Teacher / Student)
- Ajustement des paramètres de génération (temperature, top-k, top-p, beams…)

Les métriques suivies incluent :
- train loss / validation loss
- cross-entropy
- distillation loss
- perplexité

---

## Stack technique

### Data & Machine Learning
- Python
- Pandas, NumPy
- Scikit-learn
- DuckDB, SQLite
- DBeaver
- Joblib
- MLflow (versionning & suivi expérimental)

### Deep Learning / NLP
- TensorFlow / Keras
- PyTorch
- HuggingFace Transformers
- GPT-2 (Base, Medium)
- LSTM, Transformers

### Application & Déploiement
- Gradio (interface utilisateur)
- Flask / FastAPI (backend)
- Docker
- HuggingFace Spaces
- GitHub Actions (CI/CD)
- PyTest, flake8
- requirements.txt

---

## Interface utilisateur

L’application propose :
- une **interface ML** pour tester l’évaluation du niveau sportif,
- une **interface DL** pour générer des programmes d’entraînement,
- des entrées contrôlées (profil, objectifs, équipements),
- une restitution lisible et exploitable par un utilisateur final.

---

## Résultats clés

- Pipeline ML reproductible et documenté.
- Génération de descriptions d'exercices.
- Déploiement fonctionnel en POC via HuggingFace.
- Architecture évolutive vers une version produit.

---

## Axes d’amélioration

- Enrichissement du corpus de programmes sportifs.
- Ajout de contraintes temporelles globales (préparation marathon, reprise après blessure).
- Monitoring avancé en production.
- UI/UX dédiée (Vue.js / Tailwind).
- Internationalisation (i18n).
- POC RAG pour enrichissement dynamique des contenus.

---

## Auteur

**Francis BACKELAND**  
Lead Developer IA / Python  
Projet réalisé dans le cadre de la certification Développeur IA (Alyra)

## Me contacter

- **Email** : f.backeland@gmail.com  
- **LinkedIn** : [linkedin.com/in/francis-backeland](https://www.linkedin.com/in/francis-backeland-04bb7587/) 
- **Localisation** : Région Centre-Val de Loire (près d’Orléans)

---

## Licence

Projet pédagogique — usage expérimental et démonstratif.
