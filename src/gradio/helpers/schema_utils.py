from typing import Tuple, Any


def get_bounds(spec: dict, schema: dict) -> Tuple[float, float, Any, float]:
    """
    Calcule des bornes sûres (min, max, valeur par défaut, step)
    pour un champ défini dans le schéma.

    Args:
        spec: dictionnaire décrivant un champ du schéma ("type", "min", "max", "name", etc.)
        schema: schéma complet (utile pour lire les exemples "example_payload")

    Returns:
        (vmin, vmax, default, step)
    """
    field_type = spec.get("type", "number")

    # --- Cas entier ---
    if field_type in ("integer", "int"):
        vmin = int(spec.get("min", 0))
        vmax = int(spec.get("max", 100))
        default = int(
            schema.get("example_payload", {}).get(spec["name"], (vmin + vmax) // 2)
        )
        step = 1

    # --- Cas flottant ---
    else:
        vmin = float(spec.get("min", 0.0))
        vmax = float(spec.get("max", 100.0))
        default = float(
            schema.get("example_payload", {}).get(spec["name"], (vmin + vmax) / 2)
        )
        step = 0.1 if (vmax - vmin) <= 20 else 0.5

    return vmin, vmax, default, step
