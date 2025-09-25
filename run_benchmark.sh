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

echo "=== Benchmark IA (pointwise LLM judge) ==="
echo "Dataset: $DATA_PATH"
echo "Salida CSV: $OUT_CSV"
echo "Salida JSONL: $OUT_DETAILS"

# Primer intento: con LLM juez
if python benchmark.py \
    --data "$DATA_PATH" \
    --semantic \
    --use-llm \
    --llm-model mistral \
    --llm-samples 3 \
    --llm-agg median \
    --llm-include-signals \
    --out_csv "$OUT_CSV" \
    --out_details "$OUT_DETAILS"; then
  echo ">>> Benchmark completado con LLM juez"
else
  echo ">>> Ollama no disponible o fallo en el LLM. Reintentando sin juez..."
  python benchmark.py \
    --data "$DATA_PATH" \
    --semantic \
    --out_csv "$OUT_CSV" \
    --out_details "$OUT_DETAILS"
  echo ">>> Benchmark completado SIN LLM juez"
fi
