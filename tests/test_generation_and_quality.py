import os
import sys
import shutil
import pandas as pd

# ensure repo root is importable when running tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import sample_candidates, candidate_quality


def test_generation_and_quality_end_to_end(tmp_path):
    """Generate candidates, run quality pipeline, and produce plots."""
    # create tests outputs dir
    tests_out = os.path.join(os.getcwd(), 'tests', 'outputs')
    os.makedirs(tests_out, exist_ok=True)

    # generate a candidate list and move it into tests/outputs
    gen_path = sample_candidates.generate_candidate_list(n=50)
    dest = os.path.join(tests_out, 'candidates_test.csv')
    shutil.copy(gen_path, dest)

    # run quality model on copied CSV
    model, X_z, y = candidate_quality.fit_candidate_quality_model(csv_path=dest)

    # verify columns exist
    df = pd.read_csv(dest)
    assert 'CandidateQuality' in df.columns
    assert 'CandidateQualityNormalized' in df.columns

    # check normalized values within [0,100]
    vals = pd.to_numeric(df['CandidateQualityNormalized'], errors='coerce')
    assert vals.min() >= 0.0 - 1e-6
    assert vals.max() <= 100.0 + 1e-6

    # produce feature distribution plot into tests outputs
    plot_path = candidate_quality.plot_feature_distributions(dest, tests_out)
    assert os.path.exists(plot_path)
