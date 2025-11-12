import sqlite3
import pandas as pd
from pathlib import Path
from typing import List


def list_tables(conn: sqlite3.Connection) -> List[str]:
    """
    Retourne la liste des tables non système présentes dans la base SQLite.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [r[0] for r in cur.fetchall()]


def table_has_all_columns(conn: sqlite3.Connection, table: str, wanted: list[str]) -> bool:
    """
    Vérifie qu'une table contient toutes les colonnes demandées.
    """
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    cols = {row[1] for row in cur.fetchall()}  # row[1] = nom de la colonne
    return set(wanted).issubset(cols)


def load_val_subset(
    db_path: Path,
    wanted_cols: list[str],
    target_name: str,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Charge un sous-ensemble de données de validation depuis une base SQLite.
    La cible (target_name) est toujours affichée en première colonne.
    Si certaines colonnes manquent dans la table, elles sont ajoutées vides.

    Args:
        db_path: chemin vers la base SQLite.
        wanted_cols: liste des colonnes de features attendues.
        target_name: nom de la variable cible.
        limit: nombre maximum de lignes à charger.

    Returns:
        DataFrame contenant les colonnes [target_name] + features.
    """
    display_cols = [target_name] + [c for c in wanted_cols if c != target_name]

    if not db_path.exists():
        return pd.DataFrame(columns=display_cols)

    with sqlite3.connect(db_path) as conn:
        # 1) Table contenant toutes les colonnes demandées
        for tbl in list_tables(conn):
            if table_has_all_columns(conn, tbl, display_cols):
                cols_str = ", ".join([f'"{c}"' for c in display_cols])
                query = f'SELECT {cols_str} FROM "{tbl}" LIMIT {limit}'
                return pd.read_sql_query(query, conn)

        # 2) Table partielle la plus proche (max d’intersection)
        best_tbl, best_cols = None, []
        for tbl in list_tables(conn):
            cur = conn.execute(f"PRAGMA table_info('{tbl}')")
            cols = {row[1] for row in cur.fetchall()}
            inter = [c for c in display_cols if c in cols]
            if len(inter) > len(best_cols):
                best_tbl, best_cols = tbl, inter

        if best_tbl:
            cols_str = ", ".join([f'"{c}"' for c in best_cols])
            query = f'SELECT {cols_str} FROM "{best_tbl}" LIMIT {limit}'
            df = pd.read_sql_query(query, conn)

            # Complète les colonnes manquantes (target/features) et garde l'ordre
            for c in display_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[display_cols]

    # 3) Aucun tableau exploitable
    return pd.DataFrame(columns=display_cols)
