import argparse
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import os

import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util


# =========================
#   Dataclasses de esquema
# =========================

@dataclass
class GoldenSpec:
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    order_constraints: List[Tuple[str, str]] = field(default_factory=list)  # pares (A va antes que B)
    equivalences: Dict[str, List[str]] = field(default_factory=dict)
    vendor: Optional[str] = None
    explanation_ref: Optional[str] = None


@dataclass
class Item:
    id: str
    question: str
    vendor: Optional[str]
    expected: GoldenSpec
    model_name: str
    model_answer_cli: str
    model_answer_expl: str


# ==================================
#   Normalización y helper regex
# ==================================

def normalize_line(s: str) -> str:
    """Minúsculas, colapsar espacios, recortar y quitar prompts CLI (R1#, >, etc.)."""
    s = s.strip().lower()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"^\S+[>#]\s*", "", s)
    return s


def apply_equivalences(text: str, equivalences: Dict[str, List[str]]) -> str:
    """Reemplaza alias por la forma canónica (clave del dict)."""
    text_norm = text
    for canonical, aliases in equivalences.items():
        for alias in aliases:
            pattern = re.escape(alias.lower())
            text_norm = re.sub(pattern, canonical.lower(), text_norm)
    return text_norm


def text_to_lines(text: str) -> List[str]:
    return [normalize_line(l) for l in text.splitlines() if normalize_line(l)]


def regex_hits(patterns: List[str], lines: List[str]) -> List[bool]:
    hits = []
    for pat in patterns:
        rx = re.compile(pat)
        hits.append(any(rx.search(l) for l in lines))
    return hits


def f1_score(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================
#   Placeholders del golden
# ============================

IPV4_RE = r"(?:\d{1,3}\.){3}\d{1,3}"
ASN_RE = r"\d{1,6}"
# Interfaz Cisco genérica (Gi/Fa/Te/Eth... y hasta 4 niveles de /)
INTF_RE = r"(?:gi|gigabitethernet|fa|fastethernet|te|tengigabitethernet|eth|ethernet)\s*\d+(?:/\d+){0,3}"

PLACEHOLDER_MAP = {
    "<IP>": IPV4_RE,
    "<NET>": IPV4_RE,
    "<MASK>": IPV4_RE,
    "<ASN>": ASN_RE,
    "<INTF>": INTF_RE,
}

def expand_placeholders(pattern: str) -> str:
    """Sustituye placeholders <IP>, <ASN>, etc. por regex listas para usar."""
    out = pattern
    for ph, rx in PLACEHOLDER_MAP.items():
        out = out.replace(ph, rx)
        out = out.replace(ph.lower(), rx)
        out = out.replace(ph.upper(), rx)
    return out


# ======================
#   Evaluación de CLI
# ======================

def evaluate_cli(expected: GoldenSpec, cli_text: str) -> Dict[str, Any]:
    # 1) Normalización + equivalencias
    cli_text_eq = apply_equivalences(cli_text.lower(), expected.equivalences)
    lines = text_to_lines(cli_text_eq)

    # 2) Expandir placeholders en los patrones del golden
    required_expanded  = [expand_placeholders(pat) for pat in expected.required]
    optional_expanded  = [expand_placeholders(pat) for pat in expected.optional] if expected.optional else []
    forbidden_expanded = [expand_placeholders(pat) for pat in expected.forbidden] if expected.forbidden else []
    order_expanded     = [[expand_placeholders(a), expand_placeholders(b)] for (a, b) in expected.order_constraints] if expected.order_constraints else []

    # 3) Matching
    req_hits = regex_hits(required_expanded, lines)
    opt_hits = regex_hits(optional_expanded, lines) if optional_expanded else []
    forb_hits = regex_hits(forbidden_expanded, lines) if forbidden_expanded else []

    # 4) TP/FN/FP
    tp = sum(req_hits)
    fn = len(required_expanded) - tp
    allowed_any = [re.compile(p) for p in (required_expanded + optional_expanded)]
    fp = 0
    for l in lines:
        if not any(rx.search(l) for rx in allowed_any):
            fp += 1

    coverage = tp / len(required_expanded) if required_expanded else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(tp, fp, fn)

    # 5) Riesgo (forbidden)
    forbidden_count = sum(forb_hits)
    risk_penalty = min(1.0, forbidden_count / max(1, len(forbidden_expanded))) if forbidden_expanded else 0.0
    risk_score = 1.0 - risk_penalty

    # 6) Orden (constraints A antes que B)
    order_ok_list = []
    if order_expanded:
        for A, B in order_expanded:
            A_rx, B_rx = re.compile(A), re.compile(B)
            idx_a = next((i for i, l in enumerate(lines) if A_rx.search(l)), None)
            idx_b = next((i for i, l in enumerate(lines) if B_rx.search(l)), None)
            order_ok_list.append(idx_a is None or idx_b is None or idx_a <= idx_b)
    order_score = sum(1 for ok in order_ok_list if ok) / len(order_ok_list) if order_ok_list else 1.0

    return {
        "tp_required": tp,
        "fn_required": fn,
        "fp_other": fp,
        "coverage": coverage,
        "precision": precision,
        "f1": f1,
        "risk_score": risk_score,
        "order_score": order_score,
        "required_present": req_hits,
        "optional_present": opt_hits,
        "forbidden_present": forb_hits,
        "lines": lines,
    }


# ==========================
#   Similitud semántica
# ==========================

def semantic_similarity(ref_text: str, hyp_text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    model = SentenceTransformer(model_name)
    emb = model.encode([ref_text or "", hyp_text or ""], convert_to_tensor=True, normalize_embeddings=True)
    sim = float(util.cos_sim(emb[0], emb[1]).cpu().item())
    return (sim + 1) / 2  # normalizar [-1,1] -> [0,1]


# ==========================
#   LLM juez (Ollama)
# ==========================

def llm_judge_ollama(
    question: str,
    expected_cli: str,
    expected_expl: str,
    model_cli: str,
    model_expl: str,
    model_name: str = "qwen3:8b",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evalúa con un LLM local vía Ollama. Requiere `ollama serve` y `ollama pull <modelo>`."""
    system = (
        "Eres un evaluador MUY estricto de respuestas de redes/telecom. "
        "Responde SOLO con JSON válido. No añadas texto adicional."
    )

    rubric = """
Rúbrica (0–5, enteros). 5 es raro y solo si es perfecto:
- EXACTITUD: 5 si no hay errores técnicos; 4 si hay leves; 3 si hay algún error moderado; 2 si varios; 1 si mayormente incorrecto; 0 si incorrecto/dañino.
- COBERTURA: 5 si cubre TODOS los puntos clave; 4 si falta un detalle menor; 3 si faltan varias piezas; 2 si cubre <50%; 1 si casi nada; 0 si nada.
- CLARIDAD: 5 si es clara y bien estructurada; 4 si algo prolija; 3 si se entiende con esfuerzo; 2 si confusa; 1 muy confusa; 0 incomprensible.
- COHERENCIA: 5 si explicación y CLI están totalmente alineadas; 3 si hay incongruencias moderadas; 1–2 si fuertes; 0 si se contradicen.
No otorgues 5 si no cumple exactamente lo anterior. El 3 debe ser la nota “normal” para respuestas decentes con fallos.
"""

    prompt = f"""
{rubric}

[Pregunta]
{question}

[Referencia (puntos clave esperados)]
CLI:
{expected_cli}

Explicación:
{expected_expl}

[Respuesta del modelo]
CLI:
{model_cli}

Explicación:
{model_expl}

Devuelve SOLO este JSON (enteros 0–5):
{{"exactitud": X, "cobertura": Y, "claridad": Z, "coherencia": W, "comentario": "frase breve del porqué"}}
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system,
        "format": "json",              # Fuerza salida JSON pura
        "options": {"temperature": temperature},
        "stream": False,
    }
    url = f"{host}/api/generate"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "").strip()

    try:
        scores = json.loads(text)
    except Exception:
        scores = {"exactitud": None, "cobertura": None, "claridad": None, "coherencia": None, "comentario": text}

    # saneado: clamp y enteros
    for k in ["exactitud", "cobertura", "claridad", "coherencia"]:
        if isinstance(scores.get(k), (int, float)):
            v = int(round(scores[k]))
            scores[k] = max(0, min(5, v))
        else:
            scores[k] = None
    return scores


def postprocess_llm_scores(llm_scores: Optional[Dict[str, Any]], cli_metrics: Dict[str, Any], sem_sim: Optional[float]) -> Optional[Dict[str, Any]]:
    """Ajusta las notas del LLM para no 'maquillar' resultados pobres de CLI/semántica."""
    if not isinstance(llm_scores, dict):
        return llm_scores

    # Solo si hay números
    for k in ["exactitud", "cobertura", "claridad", "coherencia"]:
        v = llm_scores.get(k)
        if v is not None:
            v = int(round(v))
            v = max(0, min(5, v))
            llm_scores[k] = v

    # Penalizaciones suaves (cap a 3) cuando hay señales malas objetivas
    if cli_metrics.get("f1", 0.0) < 0.8 and llm_scores.get("exactitud") is not None:
        llm_scores["exactitud"] = min(llm_scores["exactitud"], 3)
    if sem_sim is not None and sem_sim < 0.7 and llm_scores.get("cobertura") is not None:
        llm_scores["cobertura"] = min(llm_scores["cobertura"], 3)
    if cli_metrics.get("risk_score", 1.0) < 1.0 and llm_scores.get("coherencia") is not None:
        llm_scores["coherencia"] = min(llm_scores["coherencia"], 3)

    return llm_scores


# ==========================
#   Score global (overall)
# ==========================

def compute_overall_score(
    cli_metrics: Dict[str, Any],
    sem_sim: Optional[float],
    llm_scores: Optional[Dict[str, Any]]
) -> float:
    """
    Calcula el score global con pesos fijos:
      - 0.45 CLI (F1 + orden)
      - 0.25 semántica
      - 0.10 riesgo
      - 0.20 juez LLM
    """

    # Bloque CLI: mezcla F1 y orden
    cli_block = 0.7 * cli_metrics["f1"] + 0.3 * cli_metrics["order_score"]

    # Bloque semántica
    sem_block = sem_sim if sem_sim is not None else 0.0

    # Bloque riesgo
    risk_block = cli_metrics["risk_score"]

    # Score del juez LLM
    llm_avg = 0.0
    if llm_scores and all(
        llm_scores.get(k) is not None
        for k in ["exactitud", "cobertura", "claridad", "coherencia"]
    ):
        llm_avg = np.mean([
            llm_scores["exactitud"],
            llm_scores["cobertura"],
            llm_scores["claridad"],
            llm_scores["coherencia"],
        ]) / 5.0

    # Mezcla final con pesos fijos
    overall = (
        0.45 * cli_block +
        0.25 * sem_block +
        0.10 * risk_block +
        0.20 * llm_avg
    )
    return float(overall)


# ==========================
#   Main
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Benchmark de respuestas de IA (redes/teleco).")
    parser.add_argument("--data", required=True, help="Ruta a JSONL con items de evaluación.")
    parser.add_argument("--out_csv", default="out/report.csv", help="Salida CSV con métricas agregadas.")
    parser.add_argument("--out_details", default="out/details.jsonl", help="Salida JSONL con detalles por item.")
    parser.add_argument("--semantic", action="store_true", help="Calcular similitud semántica (sentence-transformers).")
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Modelo de embeddings.")
    parser.add_argument("--use-llm", action="store_true", help="Usar LLM local vía Ollama como juez.")
    parser.add_argument("--llm-model", default="qwen3:8b", help="Modelo Ollama (p.ej., qwen3:8b, mistral, llama3.1:8b-instruct).")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Host de Ollama (por defecto localhost:11434).")
    args = parser.parse_args()

    rows = []
    details = []

    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            expected = GoldenSpec(
                required=obj["golden"]["required"],
                optional=obj["golden"].get("optional", []),
                forbidden=obj["golden"].get("forbidden", []),
                order_constraints=[tuple(x) for x in obj["golden"].get("order_constraints", [])],
                equivalences=obj["golden"].get("equivalences", {}),
                vendor=obj.get("vendor"),
                explanation_ref=obj["golden"].get("explanacion_ref"),
            )

            item = Item(
                id=obj["id"],
                question=obj["question"],
                vendor=obj.get("vendor"),
                expected=expected,
                model_name=obj["model_name"],
                model_answer_cli=obj["model_answer"]["cli"],
                model_answer_expl=obj["model_answer"].get("explicacion", ""),
            )

            # --- CLI ---
            cli_metrics = evaluate_cli(item.expected, item.model_answer_cli)

            # --- Semántica (opcional) ---
            sem_sim = None
            if args.semantic and item.expected.explanation_ref:
                sem_sim = semantic_similarity(item.expected.explanation_ref, item.model_answer_expl, args.semantic_model)

            # --- LLM juez (opcional) ---
            llm_scores = None
            if args.use_llm:
                try:
                    llm_scores = llm_judge_ollama(
                        question=item.question,
                        expected_cli="\n".join(item.expected.required),
                        expected_expl=item.expected.explanation_ref or "",
                        model_cli=item.model_answer_cli,
                        model_expl=item.model_answer_expl,
                        model_name=args.llm_model,
                        host=args.ollama_host,
                    )
                except Exception as e:
                    llm_scores = {"error": str(e)}

            # Post-proceso del juez (cap de notas si CLI/semántica flojos)
            llm_scores = postprocess_llm_scores(llm_scores, cli_metrics, sem_sim)

            overall = compute_overall_score(cli_metrics, sem_sim, llm_scores if isinstance(llm_scores, dict) else None)

            rows.append({
                "id": item.id,
                "model": item.model_name,
                "coverage_cli": cli_metrics["coverage"],
                "precision_cli": cli_metrics["precision"],
                "f1_cli": cli_metrics["f1"],
                "order_score": cli_metrics["order_score"],
                "risk_score": cli_metrics["risk_score"],
                "semantic_sim": sem_sim,
                "llm_exactitud": (llm_scores or {}).get("exactitud") if isinstance(llm_scores, dict) else None,
                "llm_cobertura": (llm_scores or {}).get("cobertura") if isinstance(llm_scores, dict) else None,
                "llm_claridad": (llm_scores or {}).get("claridad") if isinstance(llm_scores, dict) else None,
                "llm_coherencia": (llm_scores or {}).get("coherencia") if isinstance(llm_scores, dict) else None,
                "overall": overall,
            })

            details.append({
                "id": item.id,
                "model": item.model_name,
                "question": item.question,
                "cli_metrics": cli_metrics,
                "semantic_sim": sem_sim,
                "llm_scores": llm_scores,
                "overall": overall,
                "model_answer_cli": item.model_answer_cli,
                "model_answer_expl": item.model_answer_expl,
            })

    df = pd.DataFrame(rows)
    out_csv = args.out_csv if len(df) > 0 else "out/report.csv"
    out_details = args.out_details if len(df) > 0 else "out/details.jsonl"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_details), exist_ok=True)

    df.to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_details, "w", encoding="utf-8") as g:
        for d in details:
            g.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Guardado reporte en {out_csv} y detalles en {out_details}")
    if not df.empty:
        print(df.sort_values(by=['overall'], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
