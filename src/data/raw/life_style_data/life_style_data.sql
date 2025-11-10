CREATE TABLE life_style_data AS
SELECT * 
FROM read_csv_auto('C:\\Users\\fback\\Desktop\\Projets\\Dev\\Projet Alyra\\DuckDB_EDA\\datasets\\Life_Style_Data.csv');


SELECT * FROM life_style_data LIMIT 10;

-- Estimer "Calories_Burned" en fonction de :
-- - "Session_Duration (hours)"
-- - ""Workout_Type""
SELECT 
  "Age" as x_age,
  "Gender" as x_gender,
  "Weight (kg)" as x_weight,
  "Max_BPM" as x_max_bpm,
  "Avg_BPM" as x_avg_bpm,
  "Resting_BPM" as x_resting_bpm,
  "Session_Duration (hours)" as x_session_duration, 
  "Workout_Type" as x_workout_type,
  "Fat_Percentage" as x_fat_percentage,
  "Water_Intake (liters)" as x_water_intake,
  "Workout_Frequency (days/week)" as x_workout_frequency,
  "Experience_Level" as x_experience_level,
  "BMI" as x_bmi,
  "Daily meals frequency" as x_daily_meal_frequency,
  "Physical exercise" as x_physical_exercise,
  "Carbs" as x_carbs,
  "Proteins" as x_proteins,
  "Fats" as x_fats,
  "Calories" as x_calories,
  "meal_type" as x_meal_type,
  "diet_type" as x_diet_type,
  "sugar_g" AS x_sugar_g,
  "sodium_mg" AS x_sodium_mg,
  "cholesterol_mg" AS x_cholesterol_mg,
  "serving_size_g" AS x_serving_size_g,
  "cooking_method" AS x_cooking_method,
  "prep_time_min" AS x_prep_time_min,
  "cook_time_min" AS x_cook_time_min,
  "rating" AS x_rating,
  "Name of Exercise" AS x_name_of_exercise,
  "Sets" AS x_sets,
  "Reps" AS x_reps,
  "Benefit" AS x_benefit,
  "Burns Calories (per 30 min)" AS x_burns_calories_per_30min,
  "Target Muscle Group" AS x_target_muscle_group,
  "Equipment Needed" AS x_equipment_needed,
  "Difficulty Level" AS x_difficulty_level,
  "Body Part" AS x_body_part,
  "Type of Muscle" AS x_type_of_muscle,
  "Workout" AS x_workout,
  "BMI_calc" AS x_bmi_calc,
  "cal_from_macros" AS x_cal_from_macros,
  "pct_carbs" AS x_pct_carbs,
  "protein_per_kg" AS x_protein_per_kg,
  "pct_HRR" AS x_pct_hrr,
  "pct_maxHR" AS x_pct_maxhr,
  "cal_balance" AS x_cal_balance,
  "lean_mass_kg" AS x_lean_mass_kg,
  "expected_burn" AS x_expected_burn,
  "Burns Calories (per 30 min)_bc" AS x_burns_calories_per_30min_bc,
  "Burns_Calories_Bin" AS x_burns_calories_bin,
  "Calories_Burned" as y_calories_burned
FROM life_style_data;