# src/gating.py
from dataclasses import dataclass


def composite_score(s_macro: float, s_meso: float, s_micro: float,
                    w_macro: float = 0.20, w_meso: float = 0.30, w_micro: float = 0.50) -> float:
    return w_macro * s_macro + w_meso * s_meso + w_micro * s_micro


@dataclass
class GateDecision:
    ok: bool
    reason: str
    details: dict


def passes_gate(Sg: dict, Sb: dict, discord: dict, cfg: dict, *, debug: bool = False) -> GateDecision:
    """
    Decide if we can place a trade given per-horizon GOOD/BAD scores.
      Sg, Sb: dict like {"macro": 0.0..1.0, "meso": ..., "micro": ...}
      discord: dict per horizon with 0.0..1.0 blocker score (higher = worse)
      cfg: full YAML dict (needs motifs.horizons.*.weight and gating.*)
    Returns GateDecision(ok, reason, details)
    """
    gcfg = cfg.get("gating", {})
    wcfg = cfg.get("motifs", {}).get("horizons", {})
    w = {
        "macro": float(wcfg.get("macro", {}).get("weight", 0.20)),
        "meso": float(wcfg.get("meso", {}).get("weight", 0.30)),
        "micro": float(wcfg.get("micro", {}).get("weight", 0.50)),
    }

    # Horizon requirements
    if gcfg.get("require_macro", False) and Sg.get("macro", 0.0) < gcfg.get("score_min", 0.6):
        return GateDecision(False, "macro_req", {"Sg": Sg, "Sb": Sb})
    if gcfg.get("require_meso", False) and Sg.get("meso", 0.0) < gcfg.get("score_min", 0.6):
        return GateDecision(False, "meso_req", {"Sg": Sg, "Sb": Sb})
    if gcfg.get("require_micro", False) and Sg.get("micro", 0.0) < gcfg.get("score_min", 0.6):
        return GateDecision(False, "micro_req", {"Sg": Sg, "Sb": Sb})

    # BAD veto
    if max(Sb.get("macro", 0.0), Sb.get("meso", 0.0), Sb.get("micro", 0.0)) >= gcfg.get("bad_max", 0.8):
        return GateDecision(False, "bad", {"Sg": Sg, "Sb": Sb})

    # Composite score (weighted GOOD)
    comp = composite_score(
        Sg.get("macro", 0.0), Sg.get("meso", 0.0), Sg.get("micro", 0.0), **{
            "w_macro": w["macro"], "w_meso": w["meso"], "w_micro": w["micro"],
        }
    )
    if comp < gcfg.get("score_min", 0.6):
        return GateDecision(False, "score", {"Sg": Sg, "Sb": Sb, "comp": comp})

    # Margin gate: (best_good - alpha * best_bad) >= margin_min
    best_good = max(Sg.values()) if Sg else 0.0
    best_bad = max(Sb.values()) if Sb else 0.0
    margin = best_good - gcfg.get("alpha", 0.15) * best_bad
    if margin < gcfg.get("margin_min", 0.0):
        return GateDecision(False, "margin", {"best_good": best_good, "best_bad": best_bad, "margin": margin})

    # Discord block
    if any(float(discord.get(h, 0.0)) >= gcfg.get("discord_block_min", 0.999) for h in ("macro", "meso", "micro")):
        return GateDecision(False, "discord", {"discord": discord})

    return GateDecision(True, "OK", {"Sg": Sg, "Sb": Sb, "comp": comp, "margin": margin})

