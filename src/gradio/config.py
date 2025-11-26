import os
from pathlib import Path

# ---- Chaînes centralisées (sans logique) ----
# Dossiers / fichiers "métier"
MODEL_SUBDIR = ("models", "v1", "life_style_data")
MODEL_FILENAME = "model.joblib"
SCHEMA_FILENAME = "feature_schema.json"
REPORT_FILENAME = "model_report.json"
# Fichiers de scalers optionnels
FEATURE_SCALER_FILENAME = "feature_scaler.joblib"
TARGET_SCALER_FILENAME  = "target_scaler.joblib"
GENDER_ENCODER_FILENAME = "gender_encoder.joblib"

# Valeurs par défaut UI (non normalisées)
UI_DEFAULTS = {
    "Age": 40,
    "Weight (kg)": 70.0,
    "Height (m)": 1.68,
    "Max_BPM": 188.58,
    "Avg_BPM": 157.65,
    "Resting_BPM": 69.05,
    "Session_Duration (hours)": 1.0,
    "Experience_Level": 2.01,
    "Workout_Type": "Strength",
    "Difficulty Level": "Advanced",
    "Body Part": "Legs",
    "Equipment Needed": "Cable Machine",
    "Workout_Frequency (days/week)": 3.99,
    "Water_Intake (liters)": 2.0,
    "Fat_Percentage": 22.0,
    "diet_type": "Balanced",          # ✅ nouveau
}


# Exemples UI (affichés sous les sliders)
UI_EXAMPLES = [
    {
        "Age": 34.91,
        "Gender": "Male",
        "Weight (kg)": 65.27,
        "Height (m)": 1.62,
        "Max_BPM": 188.58,
        "Avg_BPM": 157.65,
        "Resting_BPM": 69.05,
        "Session_Duration (hours)": 1.0,
        "Experience_Level": 2.01,
        "Workout_Type": "Strength",
        "Difficulty Level": "Advanced",
        "Body Part": "Legs",
        "Equipment Needed": "Cable Machine",
        "Workout_Frequency (days/week)": 3.99,
        "Water_Intake (liters)": 2.5,
        "Fat_Percentage": 18.0,
        "diet_type": "Balanced",
    },
    {
        "Age": 23.37,
        "Gender": "Female",
        "Weight (kg)": 56.41,
        "Height (m)": 1.55,
        "Max_BPM": 179.43,
        "Avg_BPM": 131.75,
        "Resting_BPM": 73.18,
        "Session_Duration (hours)": 1.37,
        "Experience_Level": 2.01,
        "Workout_Type": "HIIT",
        "Difficulty Level": "Intermediate",
        "Body Part": "Chest",
        "Equipment Needed": "Step or Box",
        "Workout_Frequency (days/week)": 4.0,
        "Water_Intake (liters)": 2.2,
        "Fat_Percentage": 22.5,
        "diet_type": "Keto",
    },
    {
        "Age": 33.22,
        "Gender": "Female",
        "Weight (kg)": 58.98,
        "Height (m)": 1.67,
        "Max_BPM": 175.04,
        "Avg_BPM": 123.95,
        "Resting_BPM": 54.96,
        "Session_Duration (hours)": 0.91,
        "Experience_Level": 1.02,
        "Workout_Type": "Cardio",
        "Difficulty Level": "Intermediate",
        "Body Part": "Arms",
        "Equipment Needed": "Step or Box",
        "Workout_Frequency (days/week)": 2.99,
        "Water_Intake (liters)": 1.8,
        "Fat_Percentage": 24.0,
        "diet_type": "Low-Carb",
    },
    {
        "Age": 38.69,
        "Gender": "Female",
        "Weight (kg)": 93.78,
        "Height (m)": 1.70,
        "Max_BPM": 191.21,
        "Avg_BPM": 155.10,
        "Resting_BPM": 50.07,
        "Session_Duration (hours)": 1.10,
        "Experience_Level": 1.99,
        "Workout_Type": "HIIT",
        "Difficulty Level": "Advanced",
        "Body Part": "Shoulders",
        "Equipment Needed": "Parallel Bars or Chair",
        "Workout_Frequency (days/week)": 3.99,
        "Water_Intake (liters)": 2.7,
        "Fat_Percentage": 27.0,
        "diet_type": "Paleo",
    },
    {
        "Age": 45.09,
        "Gender": "Male",
        "Weight (kg)": 52.42,
        "Height (m)": 1.88,
        "Max_BPM": 193.58,
        "Avg_BPM": 152.88,
        "Resting_BPM": 70.84,
        "Session_Duration (hours)": 1.08,
        "Experience_Level": 2.0,
        "Workout_Type": "Strength",
        "Difficulty Level": "Advanced",
        "Body Part": "Abs",
        "Equipment Needed": "Wall",
        "Workout_Frequency (days/week)": 4.0,
        "Water_Intake (liters)": 2.3,
        "Fat_Percentage": 20.0,
        "diet_type": "Vegan",
    },
    {
        "Age": 53.19,
        "Gender": "Female",
        "Weight (kg)": 105.05,
        "Height (m)": 1.84,
        "Max_BPM": 176.52,
        "Avg_BPM": 130.60,
        "Resting_BPM": 61.84,
        "Session_Duration (hours)": 0.69,
        "Experience_Level": 1.0,
        "Workout_Type": "Yoga",
        "Difficulty Level": "Beginner",
        "Body Part": "Arms",
        "Equipment Needed": "Resistance Band or Cable Machine",
        "Workout_Frequency (days/week)": 3.02,
        "Water_Intake (liters)": 1.6,
        "Fat_Percentage": 30.0,
        "diet_type": "Vegetarian",
    },
    {
        "Age": 23.17,
        "Gender": "Male",
        "Weight (kg)": 58.41,
        "Height (m)": 1.78,
        "Max_BPM": 184.75,
        "Avg_BPM": 140.90,
        "Resting_BPM": 58.01,
        "Session_Duration (hours)": 1.67,
        "Experience_Level": 3.0,
        "Workout_Type": "Strength",
        "Difficulty Level": "Advanced",
        "Body Part": "Shoulders",
        "Equipment Needed": "None or Dumbbells",
        "Workout_Frequency (days/week)": 4.96,
        "Water_Intake (liters)": 2.9,
        "Fat_Percentage": 15.0,
        "diet_type": "Balanced",
    },
    {
        "Age": 55.92,
        "Gender": "Female",
        "Weight (kg)": 84.07,
        "Height (m)": 1.63,
        "Max_BPM": 183.87,
        "Avg_BPM": 141.12,
        "Resting_BPM": 51.79,
        "Session_Duration (hours)": 1.01,
        "Experience_Level": 2.01,
        "Workout_Type": "Yoga",
        "Difficulty Level": "Intermediate",
        "Body Part": "Back",
        "Equipment Needed": "Pull-up Bar",
        "Workout_Frequency (days/week)": 3.97,
        "Water_Intake (liters)": 2.1,
        "Fat_Percentage": 25.0,
        "diet_type": "Vegan",
    },
]


# Base de validation (relative au repo src/)
DB_RELATIVE = ("data", "processed", "life_style_data", "life_style_data_val.db")

# ---- Résolution des chemins (avec ENV overrides optionnels) ----
def build_paths(src_dir: Path) -> dict[str, Path]:
    """
    Construit tous les chemins nécessaires à l'app Gradio à partir de src_dir.
    Les variables d'environnement suivantes peuvent override :
      - MODEL_PATH, SCHEMA_PATH, MODEL_REPORT_PATH
    """
    model_dir = src_dir.joinpath(*MODEL_SUBDIR)

    model_path_env = os.getenv("MODEL_PATH")
    schema_path_env = os.getenv("SCHEMA_PATH")
    report_path_env = os.getenv("MODEL_REPORT_PATH")

    paths = {
        "MODEL_DIR": model_dir,
        "MODEL_PATH": Path(model_path_env) if model_path_env else model_dir / MODEL_FILENAME,
        "SCHEMA_PATH": Path(schema_path_env) if schema_path_env else model_dir / SCHEMA_FILENAME,
        "LOGS_DIR": src_dir / "logs",
        "DB_PATH": src_dir.joinpath(*DB_RELATIVE),
        "REPORT_PATH": Path(report_path_env) if report_path_env else model_dir / REPORT_FILENAME,
        "FEATURE_SCALER_PATH": model_dir / FEATURE_SCALER_FILENAME,
        "TARGET_SCALER_PATH":  model_dir / TARGET_SCALER_FILENAME,
        "GENDER_ENCODER_PATH": model_dir / GENDER_ENCODER_FILENAME,
    }
    return paths
