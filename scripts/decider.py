"""Decider: choose the best candidate from an evaluator DataFrame using OpenAI.

This script reads the evaluation table produced by `scripts/evaluator.py` (CSV) and calls the
OpenAI API to select the best candidate with a short rationale. The model is asked to return a
JSON object with keys: 'selected_name', 'rationale', and optional 'ranking' (ordered list of names).

How to call:

python3 -m scripts.decider --eval-csv outputs/evaluations/evaluations.csv

Output is saved to `outputs/decisions/decision.json`.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
API_KEY_PATH = os.path.join(ROOT, 'api_keys', 'openai')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'decisions')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_api_key() -> Optional[str]:
    try:
        with open(API_KEY_PATH, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _safe_extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model response")
        return json.loads(m.group(0))


def _call_openai_decider(eval_table: list[dict], api_key: str) -> dict:
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("openai package (v1+) is required for decider. Install it and try again.")

    client = OpenAI(api_key=api_key)

    system = (
        # Note: assuming consulting context
        "You are a hiring manager at a consulting company, and your task is to read a table of finalists and identify the best one to hire. "
        " One of your employees has provided you with a list of finalists, alongside evaluations of their qualities. "
        " Your task is to read the list of finalists and their evaluations and select the best one to make an offer to. "
        # Note: broad objective
        " Your only objective is to select the most qualified candidate that would bring the most consulting value to future clients. "
        " When you make a decision, please also provide a detailed rationale for your choice."
        " Please structure your response as a single JSON object (no surrounding text) matching the schema described."
    )

    table_json = json.dumps(eval_table, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    user = (
        "Given the following evaluation table (list of finalists and their qualities), choose the best candidate and explain why."
        f" Evaluation table: {table_json}"
        " Return ONLY a single JSON object with keys 'selected_name', and 'rationale.'"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        resp = client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=messages, temperature=0.2)
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            try:
                text = resp['choices'][0]['message']['content'].strip()
            except Exception:
                text = str(resp)
    except Exception as e:
        print('OpenAI API error in decider:')
        print(repr(e))
        raise

    parsed = _safe_extract_json(text)
    if 'selected_name' not in parsed or 'rationale' not in parsed:
        print('Decider returned unexpected JSON:')
        print(parsed)
        raise ValueError("Decider JSON missing required keys 'selected_name' and 'rationale'")
    return parsed


def decide_from_evaluation_csv(csv_path: str, api_key: Optional[str] = None) -> dict:
    if api_key is None:
        api_key = _load_api_key()
    if not api_key:
        raise RuntimeError('No OpenAI API key found. Decider requires a valid key at api_keys/openai or --api-key.')

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError('Evaluation table is empty')

    # Convert dataframe to list-of-dicts for the model
    eval_table = df.to_dict(orient='records')
    result = _call_openai_decider(eval_table, api_key=api_key)

    out_path = os.path.join(OUTPUT_DIR, 'decision.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f'Wrote decision to: {out_path}')
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Decide best candidate from evaluator CSV')
    parser.add_argument('--eval-csv', '-e', required=True, help='Path to evaluations CSV (outputs/evaluations/evaluations.csv)')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (optional if api_keys/openai present)')
    args = parser.parse_args()
    decision = decide_from_evaluation_csv(args.eval_csv, api_key=args.api_key)
    print(json.dumps(decision, indent=2))
