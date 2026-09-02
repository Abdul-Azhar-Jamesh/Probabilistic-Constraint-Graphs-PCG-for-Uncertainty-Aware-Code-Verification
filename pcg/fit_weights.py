"""Calibrate SOURCE_RELIABILITY and prior coefficients from labelled data.

Fits logistic regression models on the mutation-based training set to replace
hand-tuned weights in inference.py.  Two models are trained separately:

  1. **Prior model**: structural features only → P(correct) before evidence.
  2. **Evidence model**: evidence counts/weights → P(correct | evidence).

A joint model on all features is also fitted for comparison.

Run:  python -m pcg.fit_weights
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .build_training_set import (
    ALL_FEATURE_NAMES,
    EVIDENCE_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    BlockFeatures,
    build_training_set,
    save_training_set,
)
from .evaluate import Case, evaluate
from .inference import (
    expected_calibration_error,
    brier_score,
)


def _features_to_arrays(
    features: list[BlockFeatures],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) arrays from feature list."""
    X = np.array([
        [getattr(f, name) for name in feature_names]
        for f in features
    ], dtype=np.float64)
    y = np.array([f.label for f in features], dtype=np.int32)
    return X, y


def fit_prior(features: list[BlockFeatures], C: float = 1.0) -> dict[str, Any]:
    """Fit the structural prior model.

    Uses only structural features (log_cyclomatic, log_loc, depth, n_params)
    to predict P(correct) before any evidence.  The fitted coefficients replace
    the hand-set 0.22, 0.030, 0.10, 0.05 in prior_correctness().
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = _features_to_arrays(features, STRUCTURAL_FEATURE_NAMES)

    clf = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)

    # 5-fold CV for honest evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(
        LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs"),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]
    y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    accuracy_cv = (y_pred_cv == y).mean()
    ece_cv = expected_calibration_error(y_prob_cv.tolist(), y.tolist())
    brier_cv = brier_score(y_prob_cv.tolist(), y.tolist())

    coefs = dict(zip(STRUCTURAL_FEATURE_NAMES, clf.coef_[0].tolist()))
    intercept = float(clf.intercept_[0])

    return {
        "model": "prior",
        "feature_names": STRUCTURAL_FEATURE_NAMES,
        "coefficients": coefs,
        "intercept": intercept,
        "C": C,
        "n_samples": len(y),
        "class_balance": {
            "correct": int(y.sum()),
            "buggy": int(len(y) - y.sum()),
        },
        "cv_accuracy": round(float(accuracy_cv), 4),
        "cv_ece": float(ece_cv),
        "cv_brier": float(brier_cv),
    }


def fit_evidence(features: list[BlockFeatures], C: float = 1.0) -> dict[str, Any]:
    """Fit the evidence fusion model.

    Uses evidence counts and weight sums per source to predict P(correct).
    The fitted coefficients on *_neg_weight_sum map to SOURCE_RELIABILITY.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = _features_to_arrays(features, EVIDENCE_FEATURE_NAMES)

    clf = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(
        LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs"),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]
    y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    accuracy_cv = (y_pred_cv == y).mean()
    f1_cv = _f1(y, y_pred_cv)
    ece_cv = expected_calibration_error(y_prob_cv.tolist(), y.tolist())
    brier_cv = brier_score(y_prob_cv.tolist(), y.tolist())

    coefs = dict(zip(EVIDENCE_FEATURE_NAMES, clf.coef_[0].tolist()))
    intercept = float(clf.intercept_[0])

    # Derive SOURCE_RELIABILITY-compatible values from the negative weight sum
    # coefficients. The LR coefficient for *_neg_weight_sum is the log-odds
    # change per unit of negative evidence weight — precisely the quantity
    # SOURCE_RELIABILITY encodes, modulo the sign flip because negative evidence
    # lowers P(correct), so the learned coefficient is negative.
    reliability = {}
    for src in ("compile", "static", "exec", "llm", "critic"):
        key = f"{src}_neg_weight_sum"
        reliability[src] = round(abs(coefs.get(key, 0.0)), 4)

    # The critic source is not always present in the training corpus, so leave
    # it at a conservative fallback when the fit does not learn a coefficient.
    reliability.setdefault("critic", 0.25)

    return {
        "model": "evidence",
        "feature_names": EVIDENCE_FEATURE_NAMES,
        "coefficients": coefs,
        "intercept": intercept,
        "C": C,
        "n_samples": len(y),
        "class_balance": {
            "correct": int(y.sum()),
            "buggy": int(len(y) - y.sum()),
        },
        "cv_accuracy": round(float(accuracy_cv), 4),
        "cv_f1": round(float(f1_cv), 4),
        "cv_ece": float(ece_cv),
        "cv_brier": float(brier_cv),
        "source_reliability_fitted": reliability,
    }


def fit_joint(features: list[BlockFeatures], C: float = 1.0) -> dict[str, Any]:
    """Fit a joint model on all features for comparison."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = _features_to_arrays(features, ALL_FEATURE_NAMES)

    clf = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(
        LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs"),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]
    y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    accuracy_cv = (y_pred_cv == y).mean()
    f1_cv = _f1(y, y_pred_cv)
    ece_cv = expected_calibration_error(y_prob_cv.tolist(), y.tolist())
    brier_cv = brier_score(y_prob_cv.tolist(), y.tolist())

    coefs = dict(zip(ALL_FEATURE_NAMES, clf.coef_[0].tolist()))

    return {
        "model": "joint",
        "feature_names": ALL_FEATURE_NAMES,
        "coefficients": coefs,
        "intercept": float(clf.intercept_[0]),
        "C": C,
        "n_samples": len(y),
        "cv_accuracy": round(float(accuracy_cv), 4),
        "cv_f1": round(float(f1_cv), 4),
        "cv_ece": float(ece_cv),
        "cv_brier": float(brier_cv),
    }


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 0) & (y_true == 1)).sum())
    fn = int(((y_pred == 1) & (y_true == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def extract_fitted_weights(
    prior_result: dict, evidence_result: dict
) -> dict[str, Any]:
    """Extract weights for injection into inference.py."""
    # Prior coefficients: map LR coefs to prior_correctness() format.
    # prior_correctness uses:  penalty = c0*log1p(cc-1) + c1*log1p(loc) + c2*depth + c3*n_params
    # LR fitted:  logit(P) = intercept + w0*log_cyclomatic + w1*log_loc + w2*depth + w3*n_params
    # Since higher penalty → lower P(correct), and LR coef is positive for
    # features that increase P(correct), the prior penalty coefficients are
    # the negated LR coefficients.
    prior_coefs = prior_result["coefficients"]
    prior_weights = {
        "log_cyclomatic": round(-prior_coefs.get("log_cyclomatic", 0.0), 4),
        "log_loc": round(-prior_coefs.get("log_loc", 0.0), 4),
        "depth": round(-prior_coefs.get("depth", 0.0), 4),
        "n_params": round(-prior_coefs.get("n_params", 0.0), 4),
        "intercept": round(prior_result["intercept"], 4),
    }

    return {
        "source_reliability_handtuned": {
            "compile": 3.2,
            "exec": 2.8,
            "static": 1.4,
            "llm": 0.45,
        },
        "source_reliability_fitted": evidence_result["source_reliability_fitted"],
        "prior_coefficients_handtuned": {
            "log_cyclomatic": 0.22,
            "log_loc": 0.030,
            "depth": 0.10,
            "n_params": 0.05,
        },
        "prior_coefficients_fitted": prior_weights,
    }


def compare_holdout(
    fitted_weights: dict[str, Any],
) -> dict[str, Any]:
    """Run the holdout case with hand-tuned vs. fitted weights.

    Temporarily patches inference.py constants, runs the evaluation,
    then restores the originals.
    """
    from . import inference

    # Load the holdout case
    holdout_src = '''
def parse_kv(line):
    k, v = line.split("=")
    return k.strip(), v.strip()


def build_config(lines):
    cfg = {}
    for ln in lines:
        k, v = parse_kv(ln)
        cfg[k] = v
    return cfg


def get_int(cfg, key, default=0):
    if key in cfg:
        return int(cfg[key])
    return default


def merge(a, b):
    out = a
    out.update(b)
    return out
'''
    holdout_tests = '''
from candidate import parse_kv, build_config, get_int, merge

def test_parse():
    assert parse_kv("a = 1") == ("a", "1")

def test_parse_extra_equals():
    assert parse_kv("url = http://x?a=1") == ("url", "http://x?a=1")

def test_build():
    assert build_config(["a=1", "b=2"]) == {"a": "1", "b": "2"}

def test_get_int():
    assert get_int({"n": "5"}, "n") == 5

def test_merge_no_mutation():
    a = {"x": 1}
    merge(a, {"y": 2})
    assert a == {"x": 1}
'''
    holdout_case = Case(
        "config", holdout_src, holdout_tests,
        buggy={"parse_kv", "merge"},
        unreliable={"parse_kv", "merge", "build_config"},
    )

    # -- Run with hand-tuned weights ---
    # Ensure hand-tuned values are active
    old_rel = dict(inference.SOURCE_RELIABILITY)
    old_prior = dict(inference._PRIOR_COEFFICIENTS)

    inference.SOURCE_RELIABILITY.update(
        inference.SOURCE_RELIABILITY_HANDTUNED
    )
    inference._PRIOR_COEFFICIENTS.update(
        inference.PRIOR_COEFFICIENTS_HANDTUNED
    )

    res_handtuned = evaluate([holdout_case])

    # -- Run with fitted weights ---
    inference.SOURCE_RELIABILITY.update(
        fitted_weights["source_reliability_fitted"]
    )
    inference._PRIOR_COEFFICIENTS.update(
        fitted_weights["prior_coefficients_fitted"]
    )

    res_fitted = evaluate([holdout_case])

    # Restore originals
    inference.SOURCE_RELIABILITY.update(old_rel)
    inference._PRIOR_COEFFICIENTS.update(old_prior)

    return {
        "handtuned": {
            "trust_f1": res_handtuned["trust_task"]["f1"],
            "trust_ece": res_handtuned["trust_task"]["calibration"]["ece"],
            "trust_brier": res_handtuned["trust_task"]["calibration"]["brier"],
            "blame_f1": res_handtuned["blame_task"]["f1"],
            "blame_ece": res_handtuned["blame_task"]["calibration"]["ece"],
            "blame_brier": res_handtuned["blame_task"]["calibration"]["brier"],
        },
        "fitted": {
            "trust_f1": res_fitted["trust_task"]["f1"],
            "trust_ece": res_fitted["trust_task"]["calibration"]["ece"],
            "trust_brier": res_fitted["trust_task"]["calibration"]["brier"],
            "blame_f1": res_fitted["blame_task"]["f1"],
            "blame_ece": res_fitted["blame_task"]["calibration"]["ece"],
            "blame_brier": res_fitted["blame_task"]["calibration"]["brier"],
        },
    }


def save_fitted_weights(weights: dict, out_dir: str = "out") -> None:
    """Write fitted weights to JSON for inference.py to load."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fitted_weights.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(weights, fh, indent=2)
    print(f"Saved fitted weights: {path}")


def main() -> None:
    print("=" * 60)
    print("PCG Calibration Pipeline")
    print("=" * 60)

    # -- Step 1: Build training set ----------------------------------------
    print("\n[1/5] Building training set...")
    features = build_training_set()
    save_training_set(features)

    # -- Step 2: Fit prior -------------------------------------------------
    print("\n[2/5] Fitting prior model (structural features only)...")
    prior_result = fit_prior(features)
    print(f"  CV accuracy: {prior_result['cv_accuracy']:.4f}")
    print(f"  CV ECE:      {prior_result['cv_ece']:.4f}")
    print(f"  CV Brier:    {prior_result['cv_brier']:.4f}")
    print(f"  Coefficients: {prior_result['coefficients']}")

    # -- Step 3: Fit evidence model ----------------------------------------
    print("\n[3/5] Fitting evidence model (evidence features only)...")
    evidence_result = fit_evidence(features)
    print(f"  CV accuracy: {evidence_result['cv_accuracy']:.4f}")
    print(f"  CV F1:       {evidence_result['cv_f1']:.4f}")
    print(f"  CV ECE:      {evidence_result['cv_ece']:.4f}")
    print(f"  CV Brier:    {evidence_result['cv_brier']:.4f}")
    print(f"  SOURCE_RELIABILITY (fitted): {evidence_result['source_reliability_fitted']}")

    # -- Step 4: Fit joint model -------------------------------------------
    print("\n[3b/5] Fitting joint model (all features, for comparison)...")
    joint_result = fit_joint(features)
    print(f"  CV accuracy: {joint_result['cv_accuracy']:.4f}")
    print(f"  CV F1:       {joint_result['cv_f1']:.4f}")
    print(f"  CV ECE:      {joint_result['cv_ece']:.4f}")
    print(f"  CV Brier:    {joint_result['cv_brier']:.4f}")

    # -- Step 5: Extract and save fitted weights ---------------------------
    print("\n[4/5] Extracting fitted weights...")
    fitted_weights = extract_fitted_weights(prior_result, evidence_result)
    save_fitted_weights(fitted_weights)

    # Print comparison table of old vs new weights
    print("\n  SOURCE_RELIABILITY comparison:")
    print(f"  {'Source':<10} {'Hand-tuned':>12} {'Fitted':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12}")
    for src in ("compile", "static", "exec", "llm"):
        old = fitted_weights["source_reliability_handtuned"][src]
        new = fitted_weights["source_reliability_fitted"][src]
        print(f"  {src:<10} {old:>12.4f} {new:>12.4f}")

    print("\n  Prior coefficients comparison:")
    print(f"  {'Feature':<18} {'Hand-tuned':>12} {'Fitted':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*12}")
    for feat in ("log_cyclomatic", "log_loc", "depth", "n_params"):
        old = fitted_weights["prior_coefficients_handtuned"][feat]
        new = fitted_weights["prior_coefficients_fitted"][feat]
        print(f"  {feat:<18} {old:>12.4f} {new:>12.4f}")

    # -- Step 6: Holdout validation ----------------------------------------
    print("\n[5/5] Validating on holdout case...")
    comparison = compare_holdout(fitted_weights)

    print("\n  Holdout comparison (before vs. after calibration):")
    print(f"  {'Metric':<16} {'Hand-tuned':>12} {'Fitted':>12} {'Delta':>10}")
    print(f"  {'─'*16} {'─'*12} {'─'*12} {'─'*10}")
    for metric in ("trust_f1", "trust_ece", "trust_brier",
                    "blame_f1", "blame_ece", "blame_brier"):
        old = comparison["handtuned"][metric]
        new = comparison["fitted"][metric]
        delta = new - old
        # For ECE and Brier, lower is better; for F1, higher is better
        if "ece" in metric or "brier" in metric:
            quality = "✓" if delta <= 0 else "✗"
        else:
            quality = "✓" if delta >= 0 else "✗"
        print(f"  {metric:<16} {old:>12.4f} {new:>12.4f} {delta:>+9.4f} {quality}")

    # Save full results
    results = {
        "prior_model": prior_result,
        "evidence_model": evidence_result,
        "joint_model": joint_result,
        "fitted_weights": fitted_weights,
        "holdout_comparison": comparison,
    }
    results_path = os.path.join("out", "calibration_results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nFull results saved: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
