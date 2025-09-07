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
    side: str | None = None

def _thr_for(h: str, gcfg: dict) -> float:
    per = gcfg.get("score_min_per_horizon", {}) or {}
    return float(per.get(h, gcfg.get("score_min", 0.6)))

def _count_pass_horizons(Sg: dict, gcfg: dict) -> int:
    return sum(float(Sg.get(h, 0.0)) >= _thr_for(h, gcfg) for h in ("macro","meso","micro"))

def _trend_veto(side: str, macro_up: bool | None, gcfg: dict) -> tuple[bool,str]:
    # Macro-sign pick (soft) and hard guards (long_only_if_macro_up / short_only_if_macro_dn)
    if macro_up is None:
        return (False, "")  # can't decide
    if gcfg.get("pick_side", "argmax") == "macro_sign":
        # If side disagrees with regime, block here.
        if macro_up and side == "short":
            return (True, "trend_guard")
        if (not macro_up) and side == "long":
            return (True, "trend_guard")
    # Hard guards (independent of pick_side)
    if gcfg.get("long_only_if_macro_up", False) and side == "long" and macro_up is False:
        return (True, "trend_guard")
    if gcfg.get("short_only_if_macro_dn", False) and side == "short" and macro_up is True:
        return (True, "trend_guard")
    return (False, "")

def passes_gate(side: str,
                Sg: dict,  # {"macro":float,"meso":float,"micro":float}
                Sb: dict,  # {"macro":float,"meso":float,"micro":float}
                discord: dict,  # {"macro":float,"meso":float,"micro":float}
                cfg: dict,
                *,
                macro_up: bool | None = None) -> GateDecision:
    gcfg = cfg.get("gating", {})
    wcfg = cfg.get("motifs", {}).get("horizons", {})
    w = {
        "macro": float(wcfg.get("macro", {}).get("weight", 0.20)),
        "meso":  float(wcfg.get("meso",  {}).get("weight", 0.30)),
        "micro": float(wcfg.get("micro", {}).get("weight", 0.50)),
    }

    # 0) Trend guard
    veto, why = _trend_veto(side, macro_up, gcfg)
    if veto:
        return GateDecision(False, why, {"macro_up": macro_up}, side)

    # 1) Hard per-horizon requirements
    if gcfg.get("require_macro", False) and float(Sg.get("macro", 0.0)) < _thr_for("macro", gcfg):
        return GateDecision(False, "macro_req", {"Sg": Sg}, side)
    if gcfg.get("require_meso", False) and float(Sg.get("meso", 0.0))  < _thr_for("meso", gcfg):
        return GateDecision(False, "meso_req", {"Sg": Sg}, side)
    if gcfg.get("require_micro", False) and float(Sg.get("micro", 0.0)) < _thr_for("micro", gcfg):
        return GateDecision(False, "micro_req", {"Sg": Sg}, side)

    # 2) K-of-N horizon agreement
    k_needed = int(gcfg.get("require_at_least_k_of", 0))
    if k_needed > 0 and _count_pass_horizons(Sg, gcfg) < k_needed:
        return GateDecision(False, "k_of_n", {"Sg": Sg, "k": k_needed}, side)

    # 3) BAD hard veto
    if max(float(Sb.get("macro",0.0)), float(Sb.get("meso",0.0)), float(Sb.get("micro",0.0))) >= float(gcfg.get("bad_max", 0.8)):
        return GateDecision(False, "bad", {"Sb": Sb}, side)

    # 4) Composite quality (GOOD weighted)
    comp = composite_score(float(Sg.get("macro",0.0)), float(Sg.get("meso",0.0)), float(Sg.get("micro",0.0)),
                           w_macro=w["macro"], w_meso=w["meso"], w_micro=w["micro"])
    if comp < float(gcfg.get("score_min", 0.6)):
        return GateDecision(False, "score", {"comp": comp, "Sg": Sg}, side)

    # 5) Margin against BAD (favoring GOOD minus alpha * BAD)
    best_good = max(float(Sg.get("macro",0.0)), float(Sg.get("meso",0.0)), float(Sg.get("micro",0.0)))
    best_bad  = max(float(Sb.get("macro",0.0)), float(Sb.get("meso",0.0)), float(Sb.get("micro",0.0)))
    margin = best_good - float(gcfg.get("alpha", 0.15)) * best_bad
    if margin < float(gcfg.get("margin_min", 0.0)):
        return GateDecision(False, "margin", {"best_good": best_good, "best_bad": best_bad, "margin": margin}, side)

    # 6) Discord block
    if any(float(discord.get(h, 0.0)) >= float(gcfg.get("discord_block_min", 0.999)) for h in ("macro","meso","micro")):
        return GateDecision(False, "discord", {"discord": discord}, side)

    return GateDecision(True, "OK", {"comp": comp, "margin": margin, "Sg": Sg, "Sb": Sb}, side)
