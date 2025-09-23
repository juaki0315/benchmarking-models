#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, sys

REQ_TOP = {"id","question","golden","model_name","model_answer"}
REQ_GOLDEN = {"required","optional","forbidden","order_constraints","equivalences","explanacion_ref"}
REQ_MODEL = {"cli","explicacion"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta al JSONL a validar")
    args = ap.parse_args()

    ok = True
    with open(args.data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[L{i}] JSON inválido: {e}")
                ok = False
                continue
            missing = REQ_TOP - set(obj.keys())
            if missing:
                print(f"[L{i}] Faltan claves top: {missing}")
                ok = False
            g = obj.get("golden", {})
            mg = REQ_GOLDEN - set(g.keys())
            if mg:
                print(f"[L{i}] golden incompleto, faltan: {mg}")
                ok = False
            ma = obj.get("model_answer", {})
            mm = REQ_MODEL - set(ma.keys())
            if mm:
                print(f"[L{i}] model_answer incompleto, faltan: {mm}")
                ok = False
    if ok:
        print("OK: Formato JSONL válido.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
