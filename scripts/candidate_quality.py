"""candidate_quality.py

Scaffold to read a candidate CSV and fit a simple linear regression to compute CandidateQuality.

Functions:
- fit_candidate_quality_model(csv_path): reads CSV, extracts features, fits LinearRegression, returns the model and the feature matrix.

This file only includes a scaffold; it will import scikit-learn when available.
"""
import os
from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(__file__))
# global rename: data -> outputs
DATA_DIR = os.path.join(ROOT, "outputs", "candidate_lists")
MODELS_DIR = os.path.join(ROOT, "outputs", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def fit_candidate_quality_model(csv_path: str = None) -> Tuple[LinearRegression, pd.DataFrame, pd.Series]:
    """Read candidate CSV and fit a linear regression using specified numeric features.

    Features used: GPA, YearsOfExperience, ReferenceStrength, ResumeGap, UniversityRank

    Returns (model, X, y) where y is a placeholder 'CandidateQuality' created as a linear combination
    (since we don't have true labels yet). This scaffold currently computes a synthetic target using a
    simple weighted sum to allow the model to be fit.
    """
    if csv_path is None:
        # pick the first csv in DATA_DIR
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError(f"No candidate CSV files found in {DATA_DIR}")
        csv_path = os.path.join(DATA_DIR, files[0])

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required = ['GPA', 'YearsOfExperience', 'ReferenceStrength', 'ResumeGap', 'UniversityRank']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")

    X = df[required].copy()
    # Fill or coerce types
    X['GPA'] = pd.to_numeric(X['GPA'], errors='coerce').fillna(X['GPA'].mean())
    X['YearsOfExperience'] = pd.to_numeric(X['YearsOfExperience'], errors='coerce').fillna(0)
    X['ReferenceStrength'] = pd.to_numeric(X['ReferenceStrength'], errors='coerce').fillna(0)
    X['ResumeGap'] = pd.to_numeric(X['ResumeGap'], errors='coerce').fillna(0)
    X['UniversityRank'] = pd.to_numeric(X['UniversityRank'], errors='coerce').fillna(X['UniversityRank'].median())
    # For reproducibility: compute z-scores for all input columns
    means = X.mean()
    stds = X.std(ddof=0)
    stds_replaced = stds.replace(0, 1.0)
    X_z = (X - means) / stds_replaced

    # Fit model on z-scored inputs
    model = LinearRegression()

    # For now, synthesize a target CandidateQuality as a weighted sum on raw X (so we have a target)
    # compute noise scaled to ~20% of the std of the noiseless target
    y_no_noise = (
        0.75 * X['GPA'] +
        0.1 * X['YearsOfExperience'] +
        0.75 * (X['ReferenceStrength'] / 100.0) +
        -0.1 * X['ResumeGap'] +
        -0.6 * X['UniversityRank']
    )
    base_std = y_no_noise.std()
    noise_std = 0.2 * base_std if not np.isclose(base_std, 0.0) else 0.1
    y = y_no_noise + np.random.normal(0, noise_std, size=X.shape[0])

    model.fit(X_z, y)

    # Append z-scored columns and CandidateQuality to original CSV and save
    df_out = df.copy()
    for col in X_z.columns:
        df_out[f"{col}_z"] = X_z[col]

    df_out['CandidateQuality'] = y

    # Min-max normalize CandidateQuality to range 0-100 and add column CandidateQualityNormalized
    y_min = y.min()
    y_max = y.max()
    if np.isclose(y_max, y_min):
        # if constant, place everyone at midpoint 50
        df_out['CandidateQualityNormalized'] = 50.0
    else:
        df_out['CandidateQualityNormalized'] = ((y - y_min) / (y_max - y_min)) * 100.0

    df_out.to_csv(csv_path, index=False)

    # Save model and scaler metadata to outputs/models/
    import pickle
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    model_name = base_name + "_linear_model.pkl"
    scaler_name = base_name + "_scaler.pkl"
    model_path = os.path.join(MODELS_DIR, model_name)
    scaler_path = os.path.join(MODELS_DIR, scaler_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    # save means and stds for later z-scoring
    scaler_data = {'means': means.to_dict(), 'stds': stds_replaced.to_dict(), 'features': list(X.columns)}
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_data, f)

    return model, X_z, y


def plot_feature_distributions(csv_path: str, out_dir: str) -> str:
    """Plot histograms for the numeric features and CandidateQualityNormalized.

    Saves a single PNG `feature_distributions.png` into out_dir and returns the path.
    """
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    features = ['GPA', 'YearsOfExperience', 'ReferenceStrength', 'ResumeGap', 'UniversityRank', 'CandidateQualityNormalized']
    present = [f for f in features if f in df.columns]
    n = len(present)
    if n == 0:
        raise ValueError('No numeric features found in CSV to plot')

    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, feat in zip(axes, present):
        data = pd.to_numeric(df[feat], errors='coerce').dropna()
        ax.hist(data, bins=20, color='C0', alpha=0.8)
        ax.set_title(feat)
    # hide any unused axes
    for ax in axes[len(present):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'feature_distributions.png')
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    model, X, y = fit_candidate_quality_model()
    # model has been saved and CSV updated with CandidateQuality
    print('CandidateQuality column appended and model saved.')
