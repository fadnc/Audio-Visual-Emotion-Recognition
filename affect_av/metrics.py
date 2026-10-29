"""Statistical evaluation: bootstrap CIs, McNemar, calibration, temperature.

All functions take plain NumPy arrays so they can be reused by ``affect-eval``,
the CREMA-D script, and cross-validation. Probabilities are assumed to be rows
that sum to one over columns; labels are integer class indices.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def macro_ovr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro one-vs-rest AUROC, skipping classes absent from a resample."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    aucs = []
    for index in range(y_prob.shape[1]):
        binary = y_true == index
        if binary.all() or not binary.any():
            continue
        aucs.append(roc_auc_score(binary, y_prob[:, index]))
    return float(np.mean(aucs)) if aucs else float("nan")


def stratified_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                         n_bootstrap: int = 1000, alpha: float = 0.05,
                         seed: int = 0) -> dict:
    """Percentile bootstrap CIs for accuracy, macro-F1, and macro OvR AUROC.

    Resamples within each true class so every replicate keeps the class counts
    of the evaluated split. ``alpha=0.05`` gives a 95% interval.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    classes = np.unique(y_true)
    class_indices = [np.flatnonzero(y_true == label) for label in classes]
    point = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro",
                             zero_division=0),
        "macro_ovr_auc": macro_ovr_auc(y_true, y_prob),
    }
    samples = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        chosen = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                                 for indices in class_indices])
        truth, pred, prob = y_true[chosen], y_pred[chosen], y_prob[chosen]
        samples["accuracy"].append(accuracy_score(truth, pred))
        samples["macro_f1"].append(f1_score(truth, pred, labels=classes,
                                            average="macro", zero_division=0))
        samples["macro_ovr_auc"].append(macro_ovr_auc(truth, prob))
    result = {"n_bootstrap": n_bootstrap, "alpha": alpha, "seed": seed}
    for name, values in samples.items():
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size:
            low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        else:
            low = high = float("nan")
        result[name] = {"value": float(point[name]), "low": float(low), "high": float(high)}
    return result


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test of two classifiers on the same clips."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("McNemar requires equally long label and prediction arrays.")
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    if discordant:
        p_value = float(min(1.0, 2.0 * stats.binom.cdf(min(a_only, b_only),
                                                      discordant, 0.5)))
    else:
        p_value = 1.0
    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "n_discordant": discordant,
        "statistic": min(a_only, b_only),
        "p_value": p_value,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 15) -> dict:
    """Expected/maximum calibration error and the reliability-diagram bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, mce = [], 0.0, 0.0
    for index in range(n_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        if count:
            bin_accuracy = float(correct[mask].mean())
            bin_confidence = float(confidence[mask].mean())
            gap = abs(bin_accuracy - bin_confidence)
            ece += count / len(confidence) * gap
            mce = max(mce, gap)
        else:
            bin_accuracy = bin_confidence = None
            gap = 0.0
        bins.append({"lower": lower, "upper": upper, "count": count,
                     "accuracy": bin_accuracy, "confidence": bin_confidence,
                     "gap": float(gap)})
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins,
            "n_samples": int(len(confidence)), "bins": bins}


def fit_temperature(logits: np.ndarray, y_true: np.ndarray,
                    max_iter: int = 100) -> float:
    """Fit a scalar temperature by minimizing NLL with LBFGS."""
    logits_tensor = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(y_true), dtype=torch.long)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter,
                                 line_search_fn="strong_wolfe")
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_tensor / torch.exp(log_temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(0.05, 20.0).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax with a fitted temperature; returns float32 probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return (exponentiated / exponentiated.sum(axis=1, keepdims=True)).astype(np.float32)
