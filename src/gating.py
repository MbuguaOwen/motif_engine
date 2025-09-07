# src/gating.py
from dataclasses import dataclass


def composite_score(s_macro: float, s_meso: float, s_micro: float,
                    w_macro: float = 0.50, w_meso: float = 0.30, w_micro: float = 0.20) -> float:
    return w_macro * s_macro + w_meso * s_meso + w_micro * s_micro


@dataclass
class GateDecision:
    ok: bool
    reason: str
    details: dict
    side: str | None = None


def _thr_for(h: str, gcfg: dict) -> float:
    per = gcfg.get("score_min_per_horizon", {}) or {}
    return float(per.get(h, gcfg.get("score_min", 0.2)))


def _count_pass_horizons(Sg: dict, gcfg: dict) -> int:
    return sum(float(Sg.get(h, 0.0)) >= _thr_for(h, gcfg) for h in ("macro", "meso", "micro"))


def _trend_veto(side: str, macro_up: bool | None, gcfg: dict) -> tuple[bool, str]:
    if macro_up is None:
        return (False, "")
    if gcfg.get("pick_side", "argmax") == "macro_sign":
        if macro_up and side == "short":
            return (True, "trend_guard")
        if (not macro_up) and side == "long":
            return (True, "trend_guard")
    if gcfg.get("long_only_if_macro_up", False) and side == "long" and macro_up is False:
        return (True, "trend_guard")
    if gcfg.get("short_only_if_macro_dn", False) and side == "short" and macro_up is True:
        return (True, "trend_guard")
    return (False, "")


def passes_gate(side: str,
                Sg: dict,          # {"macro": float, "meso": float, "micro": float}
                Sb: dict,          # {"macro": float, "meso": float, "micro": float}
                discord: dict,     # {"macro": float, "meso": float, "micro": float}
                cfg: dict,
                *, macro_up: bool | None = None) -> GateDecision:

    gcfg = cfg.get("gating", {})
    wcfg = cfg.get("motifs", {}).get("horizons", {})
    w = {
        "macro": float(wcfg.get("macro", {}).get("weight", 0.50)),
        "meso":  float(wcfg.get("meso",  {}).get("weight", 0.30)),
        "micro": float(wcfg.get("micro", {}).get("weight", 0.20)),
    }

    # 0) Trend guard (no counter-trend)
    veto, why = _trend_veto(side, macro_up, gcfg)
    if veto:
        return GateDecision(False, why, {"macro_up": macro_up}, side)

    # 1) K-of-N horizon agreement (soft multi-horizon filter)
    k_needed = int(gcfg.get("require_at_least_k_of", 0))
    if k_needed > 0 and _count_pass_horizons(Sg, gcfg) < k_needed:
        return GateDecision(False, "k_of_n", {"Sg": Sg, "k": k_needed}, side)

    # 2) BAD hard veto
    if max(float(Sb.get("macro",0.0)), float(Sb.get("meso",0.0)), float(Sb.get("micro",0.0))) >= float(gcfg.get("bad_max", 0.7)):
        return GateDecision(False, "bad", {"Sb": Sb}, side)

    # 3) Composite quality
    comp = composite_score(
        float(Sg.get("macro",0.0)), float(Sg.get("meso",0.0)), float(Sg.get("micro",0.0)),
        w_macro=w["macro"], w_meso=w["meso"], w_micro=w["micro"]
    )
    if comp < float(gcfg.get("score_min", 0.18)):
        return GateDecision(False, "score", {"comp": comp, "Sg": Sg}, side)

    # 4) Margin vs BAD (prefer strong GOOD, penalize BAD)
    best_good = max(float(Sg.get("macro",0.0)), float(Sg.get("meso",0.0)), float(Sg.get("micro",0.0)))
    best_bad  = max(float(Sb.get("macro",0.0)), float(Sb.get("meso",0.0)), float(Sb.get("micro",0.0)))
    margin = best_good - float(gcfg.get("alpha", 0.45)) * best_bad
    if margin < float(gcfg.get("margin_min", 0.04)):
        return GateDecision(False, "margin", {"best_good": best_good, "best_bad": best_bad, "margin": margin}, side)

    # 5) Discord block
    if any(float(discord.get(h, 0.0)) >= float(gcfg.get("discord_block_min", 0.995)) for h in ("macro","meso","micro")):
        return GateDecision(False, "discord", {"discord": discord}, side)

    return GateDecision(True, "OK", {"comp": comp, "margin": margin}, side)
