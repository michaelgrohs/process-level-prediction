"""
Lagged-correlation relation discovery (
"""

from __future__ import annotations

import pandas as pd


def best_lag_correlation(predictor: pd.Series, target: pd.Series, max_shift: int = 7):
    """
    Find the shift in [0, max_shift] that maximizes |corr(predictor.shift(shift), target)|.

    predictor.shift(shift) means "predictor value `shift` steps before today" —
    i.e. does the predictor's past value explain today's target.

    Returns (best_shift, best_corr). best_corr is signed.
    """
    best_shift, best_corr, best_abs = 0, 0.0, -1.0
    for shift in range(max_shift + 1):
        shifted = predictor.shift(shift)
        valid = shifted.notna() & target.notna()
        if valid.sum() < 3:
            continue
        corr = shifted[valid].corr(target[valid])
        if pd.isna(corr):
            continue
        if abs(corr) > best_abs:
            best_abs, best_shift, best_corr = abs(corr), shift, corr
    return best_shift, best_corr


def discover_relations(
    sd_log: pd.DataFrame,
    targets: list[str],
    predictor_pool: list[str],
    max_shift: int = 7,
    threshold: float = 0.3,
) -> dict[str, list[tuple[str, int, float]]]:
    """
    For each target variable, find candidate predictors (from predictor_pool,
    excluding the target itself) whose best-shifted correlation exceeds
    `threshold` in absolute value.

    Returns {target: [(predictor, shift, corr), ...]}, sorted by |corr| desc.
    """
    relations: dict[str, list[tuple[str, int, float]]] = {}
    for target in targets:
        candidates = []
        for predictor in predictor_pool:
            if predictor == target:
                continue
            shift, corr = best_lag_correlation(sd_log[predictor], sd_log[target], max_shift)
            if abs(corr) >= threshold:
                candidates.append((predictor, shift, corr))
        candidates.sort(key=lambda c: -abs(c[2]))
        relations[target] = candidates
    return relations
