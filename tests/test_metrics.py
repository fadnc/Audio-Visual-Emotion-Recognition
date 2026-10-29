"""Statistical metric tests on small synthetic arrays."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from affect_av.metrics import (apply_temperature, expected_calibration_error,
                               fit_temperature, macro_ovr_auc, mcnemar,
                               stratified_bootstrap)


def test_stratified_bootstrap_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.eye(2)[labels]
    result = stratified_bootstrap(labels, labels.copy(), probabilities,
                                  n_bootstrap=40, seed=0)
    for name in ("accuracy", "macro_f1", "macro_ovr_auc"):
        assert result[name]["value"] == pytest.approx(1.0)
        assert result[name]["low"] == pytest.approx(1.0)
        assert result[name]["high"] == pytest.approx(1.0)


def test_stratified_bootstrap_rejects_zero_resamples():
    labels = np.array([0, 1])
    with pytest.raises(ValueError):
        stratified_bootstrap(labels, labels, np.eye(2)[labels], n_bootstrap=0)


def test_macro_ovr_auc_skips_absent_classes():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.eye(3)[labels]
    assert macro_ovr_auc(labels, probabilities) == pytest.approx(1.0)


def test_mcnemar_exact_binomial():
    labels = np.array([0, 0, 1, 1])
    pred_a = np.array([0, 0, 1, 1])
    pred_b = np.array([1, 1, 0, 0])
    result = mcnemar(labels, pred_a, pred_b)
    assert result["a_only_correct"] == 4
    assert result["b_only_correct"] == 0
    assert result["n_discordant"] == 4
    assert result["p_value"] == pytest.approx(0.125)

    identical = mcnemar(labels, pred_a, pred_a)
    assert identical["n_discordant"] == 0
    assert identical["p_value"] == pytest.approx(1.0)


def test_expected_calibration_error_and_bins():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.tile([0.9, 0.1], (4, 1))
    result = expected_calibration_error(labels, probabilities, n_bins=10)
    assert result["ece"] == pytest.approx(0.4)
    assert result["mce"] == pytest.approx(0.4)
    assert sum(entry["count"] for entry in result["bins"]) == len(labels)
    assert result["bins"][-1]["upper"] == pytest.approx(1.0)


def test_temperature_scaling_reduces_nll():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 4)).astype(np.float32) * 4
    labels = rng.integers(0, 4, size=200)
    temperature = fit_temperature(logits, labels)
    assert 0.05 <= temperature <= 20.0

    scaled = apply_temperature(logits, temperature)
    assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-5)
    before = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    after = -np.log(np.clip(scaled[np.arange(len(labels)), labels], 1e-8, 1.0)).mean()
    assert after <= before + 1e-5
