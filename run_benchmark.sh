#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/dataset.jsonl}"
OUT_CSV="out/report.csv"
OUT_DETAILS="out/details.jsonl"

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: No existe $DATA_PATH"
  echo "Coloca tu dataset en data/dataset.jsonl o pasa una ruta como argumento."
  exit 1
fi

python benchmark_cli_nlp.py --data "$DATA_PATH" --semantic --use-llm --llm-model qwen3:8b --out_csv "$OUT_CSV" --out_details "$OUT_DETAILS" || {
  echo "Intento sin LLM (Ollama no disponible?)"
  python benchmark_cli_nlp.py --data "$DATA_PATH" --semantic --out_csv "$OUT_CSV" --out_details "$OUT_DETAILS"
}
