"""sample_candidates.py

Provides:
- sample_profile(): samples a candidate profile from `inputs/resume_params.yaml` and the input CSVs
- generate_candidate_list(n=10): generates `n` candidates and writes them as CSV to data/candidate_lists/

Assumptions and notes:
- Uses pandas and PyYAML
- University rank is taken from the `rank` or `sequence` column in the universities CSV; we'll use `rank` when present or `sequence+1` as a fallback
- Hometown is sampled proportionally to `population_2020` from `top 100 US cities.csv`

"""
import csv
import os
import random
from typing import Dict, Any, List

import pandas as pd
import re
import yaml

ROOT = os.path.dirname(os.path.dirname(__file__))
INPUTS = os.path.join(ROOT, "inputs")
# Ensure output goes to THIS repo's outputs/ folder (global rename from data -> outputs)
OUTPUT_DIR = os.path.join(ROOT, "outputs", "candidate_lists")

# File paths
RESUME_PARAMS = os.path.join(INPUTS, "resume_params.yaml")
UNIS_CSV = os.path.join(INPUTS, "US-News-Rankings-Universities-Through-2023.csv")
COMPANIES_CSV = os.path.join(INPUTS, "Fortune 500 Companies.csv")
HOBBIES_CSV = os.path.join(INPUTS, "hobbylist.csv")
CITIES_CSV = os.path.join(INPUTS, "top 100 US cities.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_resume_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_universities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_companies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def _load_hobbies(path: str) -> List[str]:
    df = pd.read_csv(path)
    # hobby column is 'Hobby-name'
    if 'Hobby-name' in df.columns:
        raw = df['Hobby-name'].dropna().astype(str).tolist()
    else:
        # fallback: use first column
        raw = df.iloc[:, 0].dropna().astype(str).tolist()

    def clean_hobby(s: str) -> str:
        # remove bracketed citations like [22] or (22)
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"\(.*?\)", "", s)
        # remove digits
        s = re.sub(r"\d+", "", s)
        # remove stray punctuation except basic allowed characters (letters, spaces, '&', '-', ''')
        s = re.sub(r"[^\w\s&\-']", "", s)
        # collapse multiple spaces and strip
        s = re.sub(r"\s+", " ", s).strip()
        # Title-case for consistency
        return s.title()

    cleaned = [clean_hobby(x) for x in raw]
    # remove empty strings if any
    return [c for c in cleaned if c]


def _load_cities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # try to find a population column
    pop_col = None
    for candidate in ['population_2020', 'population', 'population_2010']:
        if candidate in df.columns:
            pop_col = candidate
            break
    if pop_col is None:
        # try to infer numeric column
        numeric_cols = df.select_dtypes(include=['int', 'float']).columns
        if len(numeric_cols) > 0:
            pop_col = numeric_cols[0]
        else:
            raise ValueError("Could not find population column in cities CSV")
    df['_population'] = pd.to_numeric(df[pop_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    return df


def sample_profile() -> Dict[str, Any]:
    """Sample a single candidate profile and return as a dict.

    Returned keys include the fields from resume_params plus:
    - University
    - UniversityRank
    - Company
    - Hobbies (list of 3)
    - Hometown
    - HometownPopulation
    """
    params = _load_resume_params(RESUME_PARAMS)
    unis = _load_universities(UNIS_CSV)
    companies = _load_companies(COMPANIES_CSV)
    hobbies = _load_hobbies(HOBBIES_CSV)
    cities = _load_cities(CITIES_CSV)

    profile: Dict[str, Any] = {}

    # Sample Race (must be defined in YAML)
    races = params.get('race', {}).get('options') if isinstance(params.get('race'), dict) else params.get('race')
    if not races:
        raise ValueError("resume_params.yaml must define race options")
    profile['Race'] = random.choice(races)

    # Sample Gender (must be defined in YAML)
    genders = params.get('gender', {}).get('options') if isinstance(params.get('gender'), dict) else params.get('gender')
    if not genders:
        raise ValueError("resume_params.yaml must define gender options")
    profile['Gender'] = random.choice(genders)

    # GPA: biased towards higher values with long left tail (target mean ~3.5)
    # We'll use a Beta distribution and scale to [min, max]
    gpa_spec = params.get('GPA')
    if not gpa_spec or 'min' not in gpa_spec or 'max' not in gpa_spec:
        raise ValueError("resume_params.yaml must define GPA with 'min' and 'max'")
    gpa_min = float(gpa_spec['min'])
    gpa_max = float(gpa_spec['max'])
    # Determine alpha/beta for Beta distribution to get mean ~3.0 within [min,max]
    # map desired mean to [0,1]
    desired_mean = 3.0
    scaled_mean = (desired_mean - gpa_min) / (gpa_max - gpa_min)
    # choose alpha>beta to bias high; simple parameterization using mean = a/(a+b)
    a = max(2.0, scaled_mean * 6)
    b = max(0.5, (1 - scaled_mean) * 2)
    r = random.betavariate(a, b)
    profile['GPA'] = round(gpa_min + r * (gpa_max - gpa_min), 2)

    # YearsOfExperience (int) - require YAML min/max
    y_spec = params.get('YearsOfExperience')
    if not y_spec or 'min' not in y_spec or 'max' not in y_spec:
        raise ValueError("resume_params.yaml must define YearsOfExperience with 'min' and 'max'")
    y_min = int(y_spec['min'])
    y_max = int(y_spec['max'])
    profile['YearsOfExperience'] = random.randint(y_min, y_max)

    # ReferenceStrength: biased towards higher values with long left tail (mean ~80)
    r_spec = params.get('ReferenceStrength')
    if not r_spec or 'min' not in r_spec or 'max' not in r_spec:
        raise ValueError("resume_params.yaml must define ReferenceStrength with 'min' and 'max'")
    r_min = int(r_spec['min'])
    r_max = int(r_spec['max'])
    desired_mean_ref = 80.0
    scaled_mean_ref = (desired_mean_ref - r_min) / (r_max - r_min)
    a_ref = max(2.0, scaled_mean_ref * 6)
    b_ref = max(0.5, (1 - scaled_mean_ref) * 2)
    r_val = random.betavariate(a_ref, b_ref)
    profile['ReferenceStrength'] = int(round(r_min + r_val * (r_max - r_min)))

    # ResumeGap - require allowed_values in YAML
    rg_spec = params.get('ResumeGap')
    if not rg_spec or 'allowed_values' not in rg_spec:
        raise ValueError("resume_params.yaml must define ResumeGap with 'allowed_values'")
    allowed = rg_spec['allowed_values']
    profile['ResumeGap'] = int(random.choice(allowed))

    # University: use US-News liberal arts colleges CSV
    # Expect a column like 'University Name' (or similar) and a 2023 ranking column named '2023'
    uni_name_col = None
    for candidate in ['University Name', 'university name', 'university', 'University']:
        if candidate in unis.columns:
            uni_name_col = candidate
            break
    if uni_name_col is None:
        # try lowercase match
        cols_lower = {c.lower(): c for c in unis.columns}
        if 'university name' in cols_lower:
            uni_name_col = cols_lower['university name']
        elif 'university' in cols_lower:
            uni_name_col = cols_lower['university']
    if uni_name_col is None:
        raise ValueError("Universities CSV must contain a university name column (e.g. 'University Name')")

    rank_col = '2023' if '2023' in unis.columns else ('rank' if 'rank' in unis.columns else None)

    # Filter to only universities with a valid numeric rank
    if rank_col is None:
        raise ValueError("Universities CSV must include a ranking column (e.g. '2023') to filter ranked universities")
    # coerce rank to numeric and drop NA
    unis['_rank_num'] = pd.to_numeric(unis[rank_col], errors='coerce')
    ranked_unis = unis.dropna(subset=['_rank_num']).copy()
    if ranked_unis.empty:
        raise ValueError("No ranked universities found in the universities CSV; cannot sample")
    chosen_uni = ranked_unis.sample(n=1).iloc[0]
    profile['University'] = chosen_uni.get(uni_name_col)
    profile['UniversityRank'] = int(chosen_uni.get('_rank_num'))

    # Company: sample uniformly from Fortune 500 list (use 'name' column)
    if 'name' in companies.columns:
        profile['Company'] = companies.sample(n=1).iloc[0]['name']
    else:
        profile['Company'] = companies.sample(n=1).iloc[0].iloc[0]

    # Hobbies: sample 3 unique hobbies
    profile['Hobbies'] = random.sample(hobbies, k=3) if len(hobbies) >= 3 else hobbies

    # Hometown: sample proportional to population using pandas' weights
    if cities['_population'].sum() > 0:
        chosen_city = cities.sample(n=1, weights=cities['_population']).iloc[0]
    else:
        chosen_city = cities.sample(n=1).iloc[0]

    # city column may be 'city' or 'City' or 'city_name'
    city_name = None
    for candidate in ['city', 'City', 'city_name', 'City Name']:
        if candidate in cities.columns:
            city_name = chosen_city.get(candidate)
            break
    if city_name is None:
        # fallback: first column
        city_name = chosen_city.iloc[0]
    profile['Hometown'] = city_name
    profile['HometownPopulation'] = int(chosen_city['_population']) if pd.notnull(chosen_city['_population']) else 0

    return profile


def generate_candidate_list(n: int = 100, out_filename: str = None) -> str:
    """Generate n candidate profiles and save as CSV under data/candidate_lists/.

    Returns the path to the saved CSV.
    """
    records: List[Dict[str, Any]] = []
    for _ in range(n):
        records.append(sample_profile())

    df = pd.DataFrame.from_records(records)

    if out_filename is None:
        out_filename = f"candidates_{n}.csv"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print("Generating 100 candidates...")
    path = generate_candidate_list(100)
    print(f"Saved candidates to {path}")
