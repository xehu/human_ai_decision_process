import json
import os
import importlib.util
from pathlib import Path

import pandas as pd

# scripts is not a package in this repo; import modules by path so tests work under pytest
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

evaluator = _load_module('evaluator', SCRIPTS_DIR / 'evaluator.py')
decider = _load_module('decider', SCRIPTS_DIR / 'decider.py')

DATA_DIR = ROOT / 'outputs'


def _write_sample_cover_letters(tmp_dir, samples):
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    for name, cover in samples.items():
        df = pd.DataFrame({'Name': [name], 'CoverLetter': [cover]})
        path = tmp / f"{name.replace(' ', '_')}.csv"
        df.to_csv(path, index=False)
    return tmp


def test_evaluator_handles_row_oriented(monkeypatch, tmp_path):
    samples = {
        'Alice Smith': 'Cover letter A',
        'Bob Jones': 'Cover letter B',
    }
    sample_dir = _write_sample_cover_letters(tmp_path, samples)

    # Mock the OpenAI bulk call to return a simple list-of-dicts (row-oriented)
    def fake_bulk(candidates, api_key):
        return [
            {'Name': 'Alice Smith', 'Fit': 90, 'Summary': 'Strong'},
            {'Name': 'Bob Jones', 'Fit': 80, 'Summary': 'Good'},
        ]

    monkeypatch.setattr(evaluator, '_call_openai_evaluator_bulk', fake_bulk)

    df = evaluator.evaluate_cover_letters_from_dir(str(sample_dir), api_key='test')
    assert not df.empty
    assert set(df['Name'].tolist()) == {'Alice Smith', 'Bob Jones'}
    assert 'Fit' in df.columns


def test_evaluator_handles_column_oriented_and_wrapped(monkeypatch, tmp_path):
    samples = {
        'Carol King': 'C1',
        'Dan Brown': 'C2',
        'Eve Mao': 'C3',
    }
    sample_dir = _write_sample_cover_letters(tmp_path, samples)

    # Column-oriented wrapped under 'EvaluationTable'
    def fake_bulk(candidates, api_key):
        return {'EvaluationTable': {
            'Name': ['Carol King', 'Dan Brown', 'Eve Mao'],
            'Overall Score': [85, 88, 82],
            'Comments': ['Good', 'Very Good', 'Solid']
        }}

    monkeypatch.setattr(evaluator, '_call_openai_evaluator_bulk', fake_bulk)

    df = evaluator.evaluate_cover_letters_from_dir(str(sample_dir), api_key='test')
    assert not df.empty
    assert set(df['Name'].tolist()) == {'Carol King', 'Dan Brown', 'Eve Mao'}
    assert 'Overall Score' in df.columns


def test_decider_reads_eval_and_returns_json(monkeypatch, tmp_path):
    # prepare a small evaluation CSV
    eval_df = pd.DataFrame([
        {'Name': 'X', 'Score': 90},
        {'Name': 'Y', 'Score': 85},
    ])
    eval_path = tmp_path / 'evaluations.csv'
    eval_df.to_csv(eval_path, index=False)

    def fake_decider(eval_table, api_key):
        return {'selected_name': 'X', 'rationale': 'Higher score', 'ranking': ['X', 'Y']}

    monkeypatch.setattr(decider, '_call_openai_decider', fake_decider)

    res = decider.decide_from_evaluation_csv(str(eval_path), api_key='test')
    assert res['selected_name'] == 'X'
    assert 'rationale' in res
    out_path = Path(decider.OUTPUT_DIR) / 'decision.json'
    assert out_path.exists()
    with open(out_path, 'r', encoding='utf-8') as f:
        parsed = json.load(f)
    assert parsed['selected_name'] == 'X'
