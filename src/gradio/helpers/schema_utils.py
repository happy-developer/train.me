from typing import Tuple, Any
from ..config import UI_DEFAULTS  # si tu as ajouté le dict ci-dessus

def _coerce_num(x, is_int: bool):
    try:
        v = float(x)
        return int(v) if is_int else float(v)
    except Exception:
        return None

def get_bounds(spec: dict, schema: dict) -> Tuple[float, float, Any, float]:
    """
    Calcule (min, max, default, step) pour un champ.
    Si l'exemple du schéma est manquant/hors bornes, on tombe sur le milieu [min,max].
    On autorise un override via UI_DEFAULTS.
    """
    is_int = spec.get("type", "number") in ("integer", "int")

    # bornes
    vmin = int(spec.get("min", 0)) if is_int else float(spec.get("min", 0.0))
    vmax = int(spec.get("max", 100)) if is_int else float(spec.get("max", 100.0))

    # fallback = milieu propre
    fallback = (vmin + vmax) // 2 if is_int else (vmin + vmax) / 2

    # 1) tentative depuis UI_DEFAULTS
    dflt = UI_DEFAULTS.get(spec["name"])
    dflt = _coerce_num(dflt, is_int)

    # 2) sinon, tentative depuis example_payload
    if dflt is None:
        ex = schema.get("example_payload", {}).get(spec["name"])
        dflt = _coerce_num(ex, is_int)

    # 3) clamp/validate
    if dflt is None or not (vmin <= dflt <= vmax):
        dflt = fallback

    # step
    step = 1 if is_int else (0.1 if (vmax - vmin) <= 20 else 0.5)

    return vmin, vmax, dflt, step
