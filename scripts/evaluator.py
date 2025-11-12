"""Evaluator: summarize pros/cons and extract features for candidates using OpenAI.

This script reads per-candidate CSV files (each produced by `cover_letter_generator`) from a
directory and calls the OpenAI chat API to produce a structured evaluation for each candidate.

How to call:
python3 -m scripts.evaluator --dir outputs/cover_letter_samples

The evaluator expects the OpenAI client (v1+) and a working API key available at
`api_keys/openai` or passed via `--api-key`.

Output: a pandas.DataFrame where each row is a candidate and columns include 'Name', 'summary',
'highlights' and any numeric feature columns the model returned (e.g. 'fit_score',
'communication', etc.). The script writes `outputs/evaluations/evaluations.csv`.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Optional

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
API_KEY_PATH = os.path.join(ROOT, 'api_keys', 'openai')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'evaluations')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_api_key() -> Optional[str]:
    try:
        with open(API_KEY_PATH, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _safe_extract_json(text: str) -> dict:
    """Extract the first JSON object from text and return parsed dict, else raise."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model response")
        return json.loads(m.group(0))


def _call_openai_evaluator_bulk(candidates: list[dict], api_key: str) -> dict:
    """Call OpenAI once with all cover letters and return the parsed JSON table.

    Uses the user-provided system prompt (hiring-committee evaluator). The model is
    expected to return a single JSON object representing the evaluation table. There are no
    required keys; the function will return the parsed JSON for downstream normalization.
    """
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("openai package (v1+) is required for evaluation. Install it and try again.")

    client = OpenAI(api_key=api_key)

    # User-provided system prompt (preserve wording exactly)
    system = (
        # Note: current example is a consulting position
        " You are a member of a hiring committee that is seeking to evaluate job candidates for a consulting position."
        " You are provided with the cover letters of several candidates."
        " Based on the provided information, your role is to provide a FINAL candidate shortlist, alongside a structured evaluation of each finalist."
        " Your evaluation will be provided to a decision-maker (e.g., hiring manager) that selects the best finalist."
        # Note: current version does not provide decision-maker preferences
        " You do not know anything about the decision-maker's preferences, and thus must use your best judgment to holistically and fairly evaluate the candidates."
        " Therefore, provide the clearest possible information to the decision-maker and be mindful that they may have different preferences than you do."
        # Note: current version allows evaluator to use any criteria
        " Your evaluation role is open-ended: you can evaluate the candidates based on ANY criteria you deem relevant."
        " The only restriction is that you need to structure your response as a table, in which each row corresponds to a finalist"
        " and each column corresponds to a specific evaluation criterion."
        " That evaluation can be either quantitative (e.g., scores from 0 to 100) or qualitative (e.g., short textual comments)."
        " Think of these criteria as the features that you believe are most relevant to the decision-maker's choice, and they are your way of communicating the strengths and weaknesses of each finalist."
        # Note: current version allows evaluator to eliminate candidates
        " You may also choose to eliminate candidates who you believe are not at all suitable for the position."
        " To do so, simply do not include them in your final evaluation table."
        " Please structure your response as a single JSON object (no surrounding text) matching the schema described."
    )

    facts_json = json.dumps(candidates, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    
    # Confirm what information the model is getting
    # print("evaluating the following candidates:")
    # print(facts_json)

    user = (
        "Given the cover letters of the candidates below, produce a FINAL shortlist and a structured evaluation table. "
        "Return ONLY a single JSON object (no surrounding text). The object should represent a table where each row is a finalist and each column is an evaluation criterion. "
        f"Candidates: {facts_json}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        resp = client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=messages, temperature=0.25)
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            try:
                text = resp['choices'][0]['message']['content'].strip()
            except Exception:
                text = str(resp)
    except Exception as e:
        print("OpenAI API error in evaluator (bulk call):")
        print(repr(e))
        raise

    parsed = _safe_extract_json(text)
    return parsed


def evaluate_cover_letters_from_dir(dir_path: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Evaluate all CSV files in dir_path and return a pandas DataFrame of results.

    Each CSV is expected to contain at least a 'Name' and 'CoverLetter' column and may include
    other candidate fields. The returned DataFrame will include columns for 'Name', 'summary',
    'highlights' (joined string), and flattened numeric feature columns from the model.
    """
    if api_key is None:
        api_key = _load_api_key()
    if not api_key:
        raise RuntimeError('No OpenAI API key found. Evaluator requires a valid key at api_keys/openai or --api-key.')

    files = sorted(glob.glob(os.path.join(dir_path, '*.csv')))
    # Build a list of candidate entries (Name, CoverLetter) to send in one API call
    candidates_list = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        # Determine cover letter text: prefer 'CoverLetter' column, else single-column CSV
        cover = ''
        name = None
        cols_lower = [c.lower() for c in df.columns]
        if 'coverletter' in cols_lower:
            cover = df.iloc[0, cols_lower.index('coverletter')]
        elif 'cover_letter' in cols_lower:
            cover = df.iloc[0, cols_lower.index('cover_letter')]
        elif df.shape[1] == 1:
            cover = df.iloc[0, 0]
        else:
            # try common variants
            for key in df.columns:
                if 'cover' in key.lower():
                    cover = df.iloc[0][key]
                    break

        # Determine name: prefer 'Name' column, else filename
        if 'name' in cols_lower:
            name = df.iloc[0, cols_lower.index('name')]
        else:
            base = os.path.basename(path)
            name = os.path.splitext(base)[0].replace('_', ' ').strip()

        candidates_list.append({'Name': name, 'CoverLetter': cover})

    if not candidates_list:
        # nothing to evaluate
        df_out = pd.DataFrame()
        out_path = os.path.join(OUTPUT_DIR, 'evaluations.csv')
        df_out.to_csv(out_path, index=False)
        print(f'Wrote empty evaluation table to: {out_path}')
        return df_out

    # Call OpenAI once with all candidates
    parsed_table = _call_openai_evaluator_bulk(candidates_list, api_key=api_key)

    # Normalize parsed_table into a list of row dicts.
    def _find_table(obj):
        """Recursively search obj for a table-like structure and return list-of-dicts or None.

        Table-like structures supported:
        - list of dicts -> returned directly
        - dict of lists with equal lengths -> transposed into list of dicts
        - dict of dicts where each value is a dict (mapping name->dict) -> converted to rows
        - nested occurrences of the above anywhere in the object -> the first match is returned
        """
        # list of dicts
        if isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj):
                return obj
            # list of primitives -> convert to list of single-key dicts
            if obj and all(not isinstance(x, (dict, list)) for x in obj):
                return [{'value': x} for x in obj]

        if isinstance(obj, dict):
            # dict of lists (column -> list)
            if obj and all(isinstance(v, list) for v in obj.values()):
                lengths = {len(v) for v in obj.values()}
                if len(lengths) == 1:
                    n = next(iter(lengths))
                    rows_local = []
                    for i in range(n):
                        rows_local.append({k: obj[k][i] for k in obj.keys()})
                    return rows_local


            # If some top-level value is a dict-of-lists (e.g., {'EvaluationTable': {'Name':[...], ...}}), transpose it.
            for v in obj.values():
                if isinstance(v, dict) and v and all(isinstance(x, list) for x in v.values()):
                    lengths = {len(x) for x in v.values()}
                    if len(lengths) == 1:
                        n = next(iter(lengths))
                        rows_local = []
                        for i in range(n):
                            rows_local.append({k: v[k][i] for k in v.keys()})
                        return rows_local

            # dict of dicts (name -> dict) or nested mapping
            if obj and all(isinstance(v, dict) for v in obj.values()):
                # If values are dicts mapping to primitives/lists, convert each to a row with Name
                # e.g., {"Alice": {..}, "Bob": {..}}
                rows_local = []
                for k, v in obj.items():
                    if isinstance(v, dict):
                        r = {'Name': k}
                        r.update(v)
                        rows_local.append(r)
                if rows_local:
                    return rows_local

            # Otherwise recursively search nested values
            for v in obj.values():
                found = _find_table(v)
                if found:
                    return found

        return None

    rows = _find_table(parsed_table)
    if rows is None:
        raise ValueError('Evaluator returned JSON that does not contain a table-like structure')

    # If rows were found but contain wrapper columns where each cell is a mapping or a list
    # (e.g., {'candidates': {"Alice": {...}, ...}} or {'candidates': {'Name': 'Alice', ...}}),
    # attempt to unwrap them into the true row list.
    if isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
        # Inspect keys in the first row as candidates for wrappers
        first = rows[0]
        for key in list(first.keys()):
            vals = [r.get(key) for r in rows]

            # Case A: each cell is a dict that itself maps names->dicts (dict-of-dicts)
            if all(isinstance(v, dict) for v in vals):
                # If each dict looks like a single candidate dict (has 'Name'), then take those
                if all(('Name' in v or 'name' in v) for v in vals):
                    rows = [v for v in vals]
                    break

                # Otherwise, try merging these dicts (they may be fragments of a larger mapping)
                merged = {}
                for v in vals:
                    for kk, vv in v.items():
                        merged[kk] = vv
                if merged and all(isinstance(vv, dict) for vv in merged.values()):
                    rows = [{'Name': kk, **vv} for kk, vv in merged.items()]
                    break

            # Case B: each cell is a list of dicts -> concatenate
            if all(isinstance(v, list) and v and isinstance(v[0], dict) for v in vals):
                new_rows = []
                for v in vals:
                    new_rows.extend(v)
                rows = new_rows
                break

    # Ensure each row has Name and CoverLetter; if missing, assign from candidates_list by order
    for i, r in enumerate(rows):
        if 'Name' not in r or not r.get('Name'):
            if i < len(candidates_list):
                r['Name'] = candidates_list[i]['Name']
        if 'CoverLetter' not in r or not r.get('CoverLetter'):
            if i < len(candidates_list):
                r['CoverLetter'] = candidates_list[i]['CoverLetter']

    # Flatten rows: convert non-primitive values to JSON strings
    records = []
    feature_keys = set()
    for r in rows:
        rec = {}
        for k, v in r.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                rec[str(k)] = v
            else:
                try:
                    rec[str(k)] = json.dumps(v)
                except Exception:
                    rec[str(k)] = str(v)
        records.append(rec)
        feature_keys.update([c for c in rec.keys() if c not in ('Name', 'CoverLetter')])

    # Build DataFrame with union of feature columns
    # The only fixed column is 'Name'; all other columns are the criteria returned by the evaluator.
    cols = ['Name'] + sorted(feature_keys)
    df_out = pd.DataFrame(records)
    if not df_out.empty:
        # ensure missing feature columns exist
        for c in cols:
            if c not in df_out.columns:
                df_out[c] = pd.NA
        df_out = df_out[cols]

    out_path = os.path.join(OUTPUT_DIR, 'evaluations.csv')
    df_out.to_csv(out_path, index=False)
    print(f'Wrote evaluation table to: {out_path}')
    return df_out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate candidate cover letter CSVs in a directory')
    parser.add_argument('--dir', '-d', required=True, help='Directory containing per-candidate CSVs')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (optional if api_keys/openai present)')
    args = parser.parse_args()
    df = evaluate_cover_letters_from_dir(args.dir, api_key=args.api_key)
    print(df.head())
