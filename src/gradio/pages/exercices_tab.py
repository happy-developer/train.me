# from typing import Union
# from pathlib import Path

# import gradio as gr
# import pandas as pd

# from ..helpers.exercices_tab_utilis import (
#     DEFAULT_EXERCICES_PATH,
#     DEFAULT_GOAL_PATH,
#     _filter_by_level,
#     _load_exercices,
#     _load_goals,
# )


# def render_list_of_exercices(
#     app_desc_ex: str,
#     level_out: gr.Textbox,  # textbox du ML tab
#     dataset_path: Union[str, Path] = DEFAULT_EXERCICES_PATH,
#     goal_path: Union[str, Path] = DEFAULT_GOAL_PATH,
# ):
#     """
#     Onglet 'Exercices proposés' : tableau + panneau de détails.
#     """

#     df = _load_exercices(dataset_path)

#     # --- Vue "compacte" pour le tableau ---
#     df_view = df.copy()
#     if "execution" in df_view.columns:
#         df_view["Execution (preview)"] = (
#             df_view["execution"].astype(str).str.slice(0, 80).fillna("") + "..."
#         )

#         base_cols = [
#             "exercise_name",
#             "target_muscles",
#             "equipment",
#             "difficulty",
#         ]

#         cols = [c for c in base_cols if c in df_view.columns]
#         cols.append("Execution (preview)")
#         df_view = df_view[cols]

#         # Colonnes à transférer vers le tab DL
#         selected_cols = [
#             c
#             for c in [
#                 "exercise_name",
#                 "target_muscles",
#                 "equipment",
#                 "difficulty",
#                 "execution",
#             ]
#             if c in df.columns
#         ]
#     else:
#         selected_cols = []

#     # Liste pour le panneau de détails
#     has_name_col = "exercise_name" in df.columns
#     exercice_choices = (
#         sorted(df["exercise_name"].dropna().unique().tolist())
#         if has_name_col
#         else []
#     )

#      # --- Chargement des goals ---
#     try:
#         goals = _load_goals(goal_path)
#     except Exception:
#         goals = []

#     if not goals:
#         goals = ["General Fitness"]

#     default_goal = "General Fitness" if "General Fitness" in goals else goals[0]


#     with gr.Tab("List of programs") as tab_ex:
#         # Sélection du goal
#         gr.Markdown("## Select your goal")

#         goal_dropdown = gr.Dropdown(
#             label="List of goals",
#             choices=goals,
#             value=default_goal,
#             interactive=True,
#         )
#         goal_state = gr.State(value=None)

#         gr.Markdown(f"## {app_desc_ex}")

#         # Niveau ML
#         with gr.Row():
#             level_display = gr.Textbox(
#                 label="Physical level",
#                 interactive=False,
#                 lines=1,
#                 max_lines=1,
#             )

#         # Recherche
#         search_box = gr.Textbox(
#             label="Search in table",
#             placeholder="Name, muscles, equipment, difficulty…",
#         )

#         # Dropdown muscles — rempli dynamiquement
#         muscle_filter = gr.Dropdown(
#             label="Filter by target muscles",
#             choices=["All"],
#             value="All",
#         )

#         # Dropdown equipment — rempli dynamiquement
#         equipment_filter = gr.Dropdown(
#             label="Filter by equipment",
#             choices=["All"],
#             value="All",
#         )

#         gr.Markdown(
#             "The table below shows an overview of each exercise.\n\n"
#             "- ML level filters difficulty\n"
#             "- Use search, target muscles and equipment filters to refine\n"
#             "- Select a program below to see full execution details\n"
#         )

#         # --- Tableau ---
#         table = gr.Dataframe(
#             value=df_view,
#             interactive=False,
#             wrap=True,
#             row_count=(0, "dynamic"),
#             col_count=(0, "dynamic"),
#         )

#         # --- Panneau de détails + sélection ---
#         selected_program_state = gr.State(value=None)  # 👈 pour le tab DL

#         if has_name_col:
#             gr.Markdown("### Program details")

#             with gr.Row():
#                 exercice_selector = gr.Dropdown(
#                     label="Select a program",
#                     choices=exercice_choices,
#                     value=exercice_choices[0] if exercice_choices else None,
#                 )

#             details_md = gr.Markdown(
#                 value="Select a program to see full description.",
#             )

#             # Tableau 1 ligne : programme sélectionné (affiché ici)
#             selected_program_df = gr.Dataframe(
#                 value=pd.DataFrame(columns=selected_cols),
#                 interactive=False,
#                 wrap=True,
#                 label="Selected program (for DL tab)",
#                 row_count=(0, "dynamic"),
#                 col_count=(0, "dynamic"),
#             )

#             def _format_details(ex_name: str):
#                 # DF vide par défaut
#                 empty_df = pd.DataFrame(columns=selected_cols)

#                 if not ex_name:
#                     return (
#                         "Select a program to see full description.",
#                         empty_df,
#                         None,
#                     )

#                 subset = df[df["exercise_name"] == ex_name]
#                 if subset.empty:
#                     return "No details found for this program.", empty_df, None

#                 row = subset.iloc[0]

#                 def get(col, default="—"):
#                     return row[col] if col in row and pd.notna(row[col]) else default

#                 parts = [
#                     f"**Name** : {get('exercise_name')}",
#                     f"**Target muscles** : {get('target_muscles')}",
#                     f"**Equipment** : {get('equipment')}",
#                     f"**Difficulty** : {get('difficulty')}",
#                     "",
#                 ]

#                 exec_text = get("execution", "")
#                 if exec_text and exec_text != "—":
#                     parts.append("**Execution** :")
#                     parts.append("")
#                     parts.append(exec_text)

#                 details_text = "\n".join(parts)

#                 # DF 1 ligne pour affichage
#                 sel_row = {c: get(c, "") for c in selected_cols}
#                 sel_df = pd.DataFrame([sel_row])

#                 # Dict pour le tab DL
#                 sel_dict = {c: get(c, "") for c in selected_cols}

#                 return details_text, sel_df, sel_dict

#             exercice_selector.change(
#                 _format_details,
#                 inputs=exercice_selector,
#                 outputs=[details_md, selected_program_df, selected_program_state],
#             )

#         # ===== Callbacks =====

#         # Synchronisation + filtrage niveau à l'ouverture de l'onglet
#         def _sync_on_tab_open(level_val: str):
#             level_text = level_val or ""
#             filtered = _filter_by_level(df_view, level_text)

#             # Muscles dynamiques selon niveau ML
#             if "target_muscles" in filtered.columns:
#                 muscles = sorted(set(filtered["target_muscles"].dropna()))
#             else:
#                 muscles = []

#             # Equipment dynamique selon niveau ML
#             if "equipment" in filtered.columns:
#                 equipments = sorted(set(filtered["equipment"].dropna()))
#             else:
#                 equipments = []

#             return (
#                 level_text,
#                 filtered,
#                 gr.update(choices=["All"] + muscles, value="All"),
#                 gr.update(choices=["All"] + equipments, value="All"),
#             )

#         tab_ex.select(
#             _sync_on_tab_open,
#             inputs=[level_out],
#             outputs=[level_display, table, muscle_filter, equipment_filter],
#         )

#         # Recherche texte + filtres muscle & equipment
#         def _search_table(
#             query: str,
#             level_val: str,
#             muscle_choice: str,
#             equipment_choice: str,
#         ):
#             # Filtre niveau ML
#             base = _filter_by_level(df_view, level_val or "")

#             # Filtre muscles
#             if muscle_choice != "All" and "target_muscles" in base.columns:
#                 base = base[base["target_muscles"] == muscle_choice]

#             # Filtre equipment
#             if equipment_choice != "All" and "equipment" in base.columns:
#                 base = base[base["equipment"] == equipment_choice]

#             # Recherche
#             if query:
#                 df_str = base.astype(str)
#                 mask = df_str.apply(
#                     lambda row: row.str.contains(query, case=False, na=False).any(),
#                     axis=1,
#                 )
#                 base = base[mask]

#             return base

#         def _on_goal_change(goal_val):
#             return goal_val

#         goal_dropdown.change(
#             _on_goal_change,
#             inputs=goal_dropdown,
#             outputs=goal_state,
#         )

#         search_box.change(
#             _search_table,
#             inputs=[search_box, level_out, muscle_filter, equipment_filter],
#             outputs=table,
#         )

#         muscle_filter.change(
#             _search_table,
#             inputs=[search_box, level_out, muscle_filter, equipment_filter],
#             outputs=table,
#         )

#         equipment_filter.change(
#             _search_table,
#             inputs=[search_box, level_out, muscle_filter, equipment_filter],
#             outputs=table,
#         )

#         # On retourne l'état (dict) du programme sélectionné
#         return selected_program_state, goal_state

