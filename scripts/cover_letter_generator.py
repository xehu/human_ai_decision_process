"""cover_letter_generator.py

Generate cover letters for candidates using OpenAI ChatCompletions.

How to call:
python3 -m scripts.cover_letter_generator --random outputs/candidate_lists/candidates_100.csv
(replace with your candidate CSV)

Important behavior changes:
- This module requires a working OpenAI API key.
- The model is asked to return a JSON object with keys: "name" and "cover_letter". The
    returned "name" will be used as the canonical candidate name (and for the output filename).

Functions:
- generate_cover_letter_for_candidate(row): call OpenAI and return parsed dict {name, cover_letter}
- generate_cover_letters_from_df(df_path, out_dir): generate letters for all candidates in a CSV
- generate_for_random_candidate(df_path): pick a random candidate and generate for them

The module reads an API key from `api_keys/openai` if present. If no API key is available the
functions will raise an informative error (no local fallback is used).
"""
import csv
import os
import random
import re
from typing import Optional

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
API_KEY_PATH = os.path.join(ROOT, 'api_keys', 'openai')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'cover_letter_samples')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_api_key() -> Optional[str]:
    try:
        with open(API_KEY_PATH, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _safe_filename(name: str) -> str:
    # slugify simple
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9 _-]", '', name)
    name = name.replace(' ', '_')
    return name

def _call_openai_system_prompt(candidate: pd.Series, api_key: str) -> dict:
    """Call OpenAI ChatCompletions API to request a JSON object with keys 'name' and 'cover_letter'.

    The function will parse and return the JSON as a dict. On any API or parsing error the
    full response text is printed for diagnostics and an exception is re-raised.
    """
    import json
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package (v1+) is required for cover letter generation. Install it and try again.") from e

    # Ensure required demographic fields are provided. Age, Race, Gender are required inputs to the model.
    # If Age is missing, try to infer from YearsOfExperience (assume graduation at 22).
    age = candidate.get('Age')
    if age is None or (isinstance(age, float) and pd.isna(age)):
        yoe = candidate.get('YearsOfExperience')
        try:
            yoe = int(yoe)
        except Exception:
            yoe = 0
        age = 22 + yoe

    race = candidate.get('Race')
    gender = candidate.get('Gender')
    if not race or not gender:
        raise ValueError('Candidate must include Race and Gender fields so the model can produce an appropriate name.')

    # Build prompt asking the model to return only JSON. Be explicit about the required schema.
    system = (
        "You are a JSON-output-only assistant that creates a candidate name and a concise, professional"
        " cover letter tailored for a consulting position. Respond with ONLY a single valid JSON object"
        " (no surrounding text, no markdown fences) with exactly two keys: 'name' and 'cover_letter'."
    )

    # Collect other factual fields to surface to the model
    facts = {
        'Age': int(age),
        'Race': race,
        'Gender': gender,
        'YearsOfExperience': int(candidate.get('YearsOfExperience') or 0),
        'University': candidate.get('University') or '',
        'UniversityRank': candidate.get('UniversityRank') or '',
        'GPA': candidate.get('GPA') or '',
        'ReferenceStrength': candidate.get('ReferenceStrength') or '',
        'Hobbies': candidate.get('Hobbies') or '',
        'PriorCompany': candidate.get('Company') or '',
    }

    print(f"Generating cover letter for candidate with facts: {facts}")

    # Use json.dumps with a default converter to handle numpy/pandas numeric types (e.g., int64)
    facts_json = json.dumps(facts, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    user = (
        "Create a professional cover letter for a consulting position, using the facts provided to characterize the applicant. "
        "You must mention all relevant facts in the cover letter, ensuring all mentions are completely accurate. "
        "However, on the basis of those facts, feel free to creatively expand on the applicant's motivations and aspirations, "
        "elaborating on why they are interested in the job, what they hope to achieve, and their unique qualifications. "
        "If the applicant has 0 years of experience, say they are a new graduate; emphasize their academic achievements and excitement for starting their career. "
        "For applicants with more years of experience, highlight their professional background by specifically stating the name of an example prior client ('PriorCompany' in the facts provided). "
        "Also produce a plausible full name consistent with the applicant's Age, Race, and Gender. "
        "Return a JSON object with keys 'name' (string) and 'cover_letter' (string)."
        f" Facts: {facts_json}"
        " The cover letter should be ~3 short paragraphs, factual, and concise."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=messages, temperature=0.7)

        # Try a few ways to extract the assistant text depending on response shape
        text = None
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            try:
                text = resp["choices"][0]["message"]["content"].strip()
            except Exception:
                # Last resort: stringify the raw response
                text = str(resp)
    except Exception as e:
        # Print diagnostics and re-raise to help debugging
        print("OpenAI API error while generating cover letter:")
        print(repr(e))
        raise

    # Try to extract a JSON object from the model's text
    try:
        # First attempt: parse the whole response
        parsed = json.loads(text)
    except Exception:
        # Try to extract the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            snippet = m.group(0)
            try:
                parsed = json.loads(snippet)
            except Exception:
                print("Failed to parse JSON from model response. Full response below for diagnosis:")
                print(text)
                raise ValueError("Model returned invalid JSON; see printed response for diagnostics")
        else:
            print("Model did not return JSON. Full response below for diagnosis:")
            print(text)
            raise ValueError("Model did not return JSON; see printed response for diagnostics")

    # Validate schema
    if not isinstance(parsed, dict) or 'name' not in parsed or 'cover_letter' not in parsed:
        print("Model returned JSON but schema is invalid. Parsed JSON:")
        print(parsed)
        raise ValueError("Invalid JSON schema from model; expected keys 'name' and 'cover_letter'.")

    # Ensure values are strings
    parsed['name'] = str(parsed['name']).strip()
    parsed['cover_letter'] = str(parsed['cover_letter']).strip()
    return parsed


# NOTE: Local fallback template removed intentionally. This project requires OpenAI API usage.


def generate_cover_letter_for_candidate(candidate: pd.Series, api_key: Optional[str] = None) -> dict:
    """Generate a cover letter for a single candidate using OpenAI and return a dict {name, cover_letter}.

    This function enforces API-only behavior. If no api_key is provided and no key exists at
    `api_keys/openai` it raises a RuntimeError.
    """
    if api_key is None:
        api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("No OpenAI API key found. Cover letter generation requires a valid key at api_keys/openai or --api-key.")

    # Work on a copy of the candidate Series so we don't mutate the caller's DataFrame slice
    candidate_to_use = candidate.copy()

    # Ensure demographic fields exist (Age, Race, Gender). Age may be inferred from YearsOfExperience.
    # The _call_openai_system_prompt will raise a clear error if Race/Gender are missing.
    return _call_openai_system_prompt(candidate_to_use, api_key)


def generate_cover_letters_from_df(df_path: str, out_dir: Optional[str] = None, api_key: Optional[str] = None) -> list:
    """Generate cover letters for all candidates in a CSV file.

    Writes one CSV per candidate to out_dir (defaults to outputs/cover_letter_samples/). Returns list of paths.
    """
    if out_dir is None:
        out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(df_path)
    paths = []

    # preserve column order from input CSV, but remove Name since model returns canonical name
    input_columns = list(df.columns)
    other_cols = [c for c in input_columns if c != 'Name']
    header = ['Name', 'CoverLetter'] + other_cols

    def _to_serializable(v):
        # convert numpy/pandas scalars to native Python types where possible
        try:
            if hasattr(v, 'item'):
                return v.item()
        except Exception:
            pass
        # fallback to string for things like NaN or complex objects
        if pd.isna(v):
            return ''
        return v

    for _, row in df.iterrows():
        try:
            result = generate_cover_letter_for_candidate(row, api_key=api_key)
        except Exception:
            print(f"Error generating cover letter for candidate (row):\n{row.to_dict()}")
            raise

        name = result['name']
        letter = result['cover_letter']
        safe = _safe_filename(name)
        out_path = os.path.join(out_dir, f"{safe}.csv")
        # save CSV with columns: Name, CoverLetter, <all other input columns>
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            row_values = [name, letter] + [_to_serializable(row.get(c, '')) for c in other_cols]
            writer.writerow(row_values)
        paths.append(out_path)
    return paths


def generate_for_random_candidate(df_path: str, api_key: Optional[str] = None) -> str:
    """Pick a random candidate from df_path, generate a cover letter, and return the output path."""
    df = pd.read_csv(df_path)
    if df.empty:
        raise ValueError('Input dataframe is empty')
    idx = random.randrange(len(df))
    candidate = df.iloc[idx]
    api_key = api_key or _load_api_key()

    # prepare header from original dataframe columns
    input_columns = list(df.columns)
    other_cols = [c for c in input_columns if c != 'Name']
    header = ['Name', 'CoverLetter'] + other_cols

    def _to_serializable(v):
        try:
            if hasattr(v, 'item'):
                return v.item()
        except Exception:
            pass
        if pd.isna(v):
            return ''
        return v

    try:
        result = generate_cover_letter_for_candidate(candidate, api_key=api_key)
    except Exception:
        print(f"Error generating cover letter for candidate (row):\n{candidate.to_dict()}")
        raise

    name = result['name']
    letter = result['cover_letter']
    safe = _safe_filename(name)
    out_path = os.path.join(OUTPUT_DIR, f"{safe}.csv")
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row_values = [name, letter] + [_to_serializable(candidate.get(c, '')) for c in other_cols]
        writer.writerow(row_values)
    return out_path


if __name__ == '__main__':
    # Provide a small CLI so the module can be used from the shell without importing.
    import argparse
    parser = argparse.ArgumentParser(description='Generate cover letters from a candidates CSV')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--random', '-r', metavar='CSV', help='Pick a random candidate from CSV and generate a letter')
    group.add_argument('--all', '-a', metavar='CSV', help='Generate letters for all candidates in CSV')
    parser.add_argument('--api-key', '-k', metavar='KEY', help='Optional OpenAI API key to use instead of api_keys/openai')

    args = parser.parse_args()
    if args.random:
        out = generate_for_random_candidate(args.random, api_key=args.api_key)
        print(f'Generated cover letter at: {out}')
    elif args.all:
        outs = generate_cover_letters_from_df(args.all, api_key=args.api_key)
        print(f'Generated {len(outs)} cover letters, example: {outs[0] if outs else None}')
