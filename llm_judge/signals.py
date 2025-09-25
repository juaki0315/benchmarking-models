from typing import Any, Dict


def build_judge_signals(cli_metrics: Dict[str, Any], expected) -> Dict[str, Any]:
    """
    Señales objetivas para reducir alucinación del juez:
    - missing_required: patrones del golden no encontrados
    - forbidden_hits: patrones prohibidos detectados
    - order_violations: constraints A->B si order_score < 1.0
    """
    sig = {
        "missing_required": [],
        "forbidden_hits": [],
        "order_violations": [],
    }
    # required ausentes
    req_hits = cli_metrics.get("required_present", [])
    if isinstance(req_hits, list):
        for hit, pat in zip(req_hits, getattr(expected, "required", [])):
            if not hit:
                sig["missing_required"].append(pat)

    # forbidden presentes
    forb_hits = cli_metrics.get("forbidden_present", [])
    if isinstance(forb_hits, list):
        for hit, pat in zip(forb_hits, getattr(expected, "forbidden", [])):
            if hit:
                sig["forbidden_hits"].append(pat)

    # orden
    if cli_metrics.get("order_score", 1.0) < 1.0:
        for pair in getattr(expected, "order_constraints", []) or []:
            sig["order_violations"].append({"before": pair[0], "after": pair[1]})

    return sig
