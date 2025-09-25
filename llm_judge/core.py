import json
import re
import time
import random
from typing import Any, Dict, List, Optional

import numpy as np
import requests


def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    """Intenta parsear JSON y tolera texto extra antes/después."""
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def ollama_generate(
    host: str,
    model: str,
    system: str,
    prompt: str,
    temperature: float = 0.3,
    seed: Optional[int] = None,
    fmt_json: bool = True,
    timeout: int = 120,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "options": {"temperature": temperature},
        "stream": False,
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    if fmt_json:
        payload["format"] = "json"
    url = f"{host}/api/generate"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def aggregate_pointwise(samples: List[Dict[str, Any]], agg: str = "median") -> Dict[str, Any]:
    """Agrega K dictámenes por mediana/mean + 'consenso'."""
    keys = ["exactitud", "cobertura", "claridad", "coherencia"]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [s.get(k) for s in samples if isinstance(s.get(k), (int, float))]
        if not vals:
            out[k] = None
        else:
            out[k] = int(round(np.median(vals))) if agg == "median" else int(round(float(np.mean(vals))))
    comments = [s.get("comentario", "") for s in samples if isinstance(s.get("comentario", ""), str)]
    out["comentario"] = max(comments, key=len)[:200] if comments else ""
    nums = []
    for s in samples:
        row = [s.get(k) for k in keys]
        if all(isinstance(v, (int, float)) for v in row):
            nums.extend(row)
    if len(nums) >= 2:
        cons = 1.0 - (float(np.std(nums)) / 5.0)
        out["consenso"] = max(0.0, min(1.0, cons))
    else:
        out["consenso"] = 0.0
    return out


def clamp_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Fuerza 0–5 enteros o None, comentario string."""
    for k in ["exactitud", "cobertura", "claridad", "coherencia"]:
        v = scores.get(k)
        scores[k] = max(0, min(5, int(round(v)))) if isinstance(v, (int, float)) else None
    if not isinstance(scores.get("comentario"), str):
        scores["comentario"] = ""
    return scores


def k_samples(call_once, k: int, sleep_s: float = 0.05):
    """Ejecuta call_once() K veces con seeds distintas y agrega salida."""
    outputs = []
    for _ in range(max(1, k)):
        seed = random.randint(1, 10_000_000)
        outputs.append(call_once(seed))
        time.sleep(sleep_s)
    return outputs
