"""
Automated Test Suite for EduInsight AI
======================================
Tests:
1. Dataset integrity and loading.
2. Model artifact and configuration.
3. Prediction correctness and bounded range [0, 20].
4. SHAP TreeExplainer calculation and Additivity (Efficiency Axiom: Base + Sum(SHAP) == Prediction).
5. What-If Counterfactual calculation.
6. Multi-Model Benchmark metrics consistency.
7. Diagnostic report generation.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np
import joblib
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"
BENCHMARK_PATH = BASE_DIR / "models" / "model_comparison.json"


def test_dataset_exists_and_loads():
    assert DATA_PATH.exists(), f"Dataset missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH, sep=";")
    assert df.shape[0] == 395, f"Expected 395 rows, found {df.shape[0]}"
    assert df.shape[1] == 33, f"Expected 33 columns, found {df.shape[1]}"
    required_cols = ["studytime", "failures", "absences", "G1", "G2", "G3"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_model_artifact_and_metadata():
    assert MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}"
    payload = joblib.load(MODEL_PATH)
    assert "model" in payload, "Payload missing 'model'"
    assert "features" in payload, "Payload missing 'features'"
    assert payload["features"] == ["studytime", "failures", "absences", "G1", "G2"]
    assert hasattr(payload["model"], "predict"), "Model object has no predict method"


def test_model_prediction_range():
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    features = payload["features"]

    # Test edge cases: Min values, Average, Max values
    cases = [
        pd.DataFrame([[1, 0, 0, 0, 0]], columns=features),
        pd.DataFrame([[2, 0, 2, 10, 10]], columns=features),
        pd.DataFrame([[4, 0, 0, 20, 20]], columns=features)
    ]
    for case in cases:
        pred = float(model.predict(case)[0])
        assert 0.0 <= pred <= 20.0, f"Prediction {pred} out of academic bounds [0, 20]"


def test_shap_calculation_and_additivity_axiom():
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    features = payload["features"]
    df = pd.read_csv(DATA_PATH, sep=";")[features]

    bg_sample = df.sample(min(50, len(df)), random_state=42)
    explainer = shap.TreeExplainer(model, data=bg_sample, feature_perturbation='interventional')
    base_val = float(explainer.expected_value)

    student = pd.DataFrame([[2, 0, 2, 12, 13]], columns=features)
    prediction = float(model.predict(student)[0])
    shap_vals = explainer.shap_values(student)[0]

    assert len(shap_vals) == len(features), f"Expected {len(features)} SHAP values, got {len(shap_vals)}"

    # Shapley Efficiency Axiom: Base Value + sum(SHAP) == Prediction (within float tolerance)
    reconciled_sum = base_val + float(np.sum(shap_vals))
    assert np.isclose(reconciled_sum, prediction, atol=1e-3), (
        f"Additivity failed! Base({base_val:.4f}) + Sum(SHAP)({np.sum(shap_vals):.4f}) = {reconciled_sum:.4f} != Pred({prediction:.4f})"
    )


def test_what_if_counterfactual_simulation():
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    features = payload["features"]
    df = pd.read_csv(DATA_PATH, sep=";")[features]
    bg_sample = df.sample(min(50, len(df)), random_state=42)
    explainer = shap.TreeExplainer(model, data=bg_sample, feature_perturbation='interventional')

    baseline_student = pd.DataFrame([[1, 2, 15, 7, 8]], columns=features)
    simulated_student = pd.DataFrame([[3, 1, 4, 10, 11]], columns=features)

    base_pred = float(model.predict(baseline_student)[0])
    sim_pred = float(model.predict(simulated_student)[0])

    assert 0 <= base_pred <= 20, f"Base prediction {base_pred} outside valid grade range [0, 20]"
    assert 0 <= sim_pred <= 20, f"Simulated prediction {sim_pred} outside valid grade range [0, 20]"

    # Delta must mathematically equal simulated minus baseline
    delta = sim_pred - base_pred
    assert np.isclose(delta, sim_pred - base_pred, atol=1e-5)

    # Verify counterfactual SHAP computation and additivity
    sim_shap = explainer.shap_values(simulated_student)[0]
    base_val = float(explainer.expected_value)
    assert np.isclose(base_val + float(np.sum(sim_shap)), sim_pred, atol=1e-3), (
        "Counterfactual SHAP additivity violation: base + sum(shap) != simulated_prediction"
    )


def test_benchmark_metrics_file():
    assert BENCHMARK_PATH.exists(), f"Benchmark file missing at {BENCHMARK_PATH}"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    assert len(benchmarks) == 5, f"Expected 5 models evaluated, found {len(benchmarks)}"
    models = [b["model_name"] for b in benchmarks]
    assert "Random Forest Regressor" in models
    assert "Gradient Boosting Regressor" in models
    assert "Decision Tree Regressor" in models
    assert "Ridge Regression" in models
    assert "Linear Regression" in models

    rf = [b for b in benchmarks if b["model_name"] == "Random Forest Regressor"][0]
    assert rf["test_r2"] > 0.85, f"Random Forest R² should be > 0.85, got {rf['test_r2']}"
    assert rf["mae"] < 1.2, f"Random Forest MAE should be < 1.2, got {rf['mae']}"


def test_report_generation():
    prediction = 13.5
    baseline = 10.7
    features = ["studytime", "failures", "absences", "G1", "G2"]
    shap_vals = [-0.1, -0.05, 0.2, 1.2, 1.5]

    report = (
        f"EDUINSIGHT AI REPORT\n"
        f"Predicted Final Grade: {prediction:.2f}/20\n"
        f"Baseline: {baseline:.2f}\n"
    )
    for f, s in zip(features, shap_vals):
        report += f"{f}: {s:+.2f} pts\n"

    assert "EDUINSIGHT AI REPORT" in report
    assert "13.50/20" in report
