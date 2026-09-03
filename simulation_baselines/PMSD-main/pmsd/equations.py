

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error


class FittedEquation:
    """A fitted target = f(shifted predictors) equation, ready to predict on
    a fresh dataframe that provides the same (unshifted) predictor columns."""

    def __init__(self, model, predictors: list[str], shifts: dict[str, int],
                 fallback_mean: float, bound: float):
        self.model = model
        self.predictors = predictors
        self.shifts = shifts
        self.fallback_mean = fallback_mean
        self.bound = bound  # non-negative process quantity, capped at fit-time observed max * 5

    def predict_row(self, history: pd.DataFrame) -> float:
        """
        Predict one step ahead using the most recent rows of `history`
        (a dataframe with the same columns as the SD-Log, already extended
        with any simulated values needed to cover the required shifts).

        """
        if self.model is None or not self.predictors:
            return float(self.fallback_mean)
        x = []
        for p in self.predictors:
            shift = self.shifts[p]
            idx = -1 - shift
            if -idx > len(history):
                return float(self.fallback_mean)
            val = history[p].iloc[idx]
            if pd.isna(val) or not np.isfinite(val):
                return float(self.fallback_mean)
            x.append(val)
        pred = float(self.model.predict([x])[0])
        if not np.isfinite(pred):
            return float(self.fallback_mean)
        return float(np.clip(pred, 0.0, self.bound))


_CANDIDATES = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0),
    "poly2": lambda: make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
}


def _fit_candidates(
    train_df: pd.DataFrame,
    target_col: str,
    candidate_predictors: list[tuple[str, int, float]],
    max_predictors: int,
) -> dict[str, FittedEquation]:
    """Fit every candidate model form for one target; return {name: FittedEquation}."""
    predictors = [p for p, _, _ in candidate_predictors[:max_predictors]]
    shifts = {p: s for p, s, _ in candidate_predictors[:max_predictors]}
    fallback_mean = float(train_df[target_col].mean())
    bound = float(max(train_df[target_col].max(), fallback_mean) * 5)

    if not predictors:
        return {"none": FittedEquation(None, [], {}, fallback_mean, bound)}

    cols = {p: train_df[p].shift(shifts[p]) for p in predictors}
    x = pd.DataFrame(cols)
    y = train_df[target_col]
    valid = x.notna().all(axis=1) & y.notna()
    x_train, y_train = x[valid].to_numpy(), y[valid].to_numpy()

    if len(x_train) < 5:
        return {"none": FittedEquation(None, [], {}, fallback_mean, bound)}

    out = {}
    for name, make_model in _CANDIDATES.items():
        model = make_model()
        try:
            model.fit(x_train, y_train)
        except Exception:
            continue
        out[name] = FittedEquation(model, predictors, shifts, fallback_mean, bound)

    return out or {"none": FittedEquation(None, [], {}, fallback_mean, bound)}


def _rollout_mae(val_df: pd.DataFrame, history: pd.DataFrame,
                  finish_eq: FittedEquation, tt_eq: FittedEquation) -> float:
    """
    Recursively predict finish_rate/throughput_time over val_df's own dates,
    using the same order simulate.py uses (concurrent_cases/arrival_rate held
    at their true values — those aren't being selected here — throughput_time
    then finish_rate predicted from whatever's been simulated so far).
    Returns a combined score: each target's MAE normalized by its own mean
    (so the two very different scales don't let one dominate).
    """
    sim = history.copy()
    finish_actual, finish_pred, tt_actual, tt_pred = [], [], [], []
    for date, row in val_df.iterrows():
        sim.loc[date] = row
        tt_p = tt_eq.predict_row(sim)
        sim.loc[date, "throughput_time"] = tt_p
        finish_p = finish_eq.predict_row(sim)
        sim.loc[date, "finish_rate"] = finish_p

        tt_actual.append(row["throughput_time"]); tt_pred.append(tt_p)
        finish_actual.append(row["finish_rate"]); finish_pred.append(finish_p)

    finish_scale = max(np.mean(np.abs(finish_actual)), 1e-6)
    tt_scale = max(np.mean(np.abs(tt_actual)), 1e-6)
    return (mean_absolute_error(finish_actual, finish_pred) / finish_scale
            + mean_absolute_error(tt_actual, tt_pred) / tt_scale)


def fit_best_equation_pair(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    finish_predictors: list[tuple[str, int, float]],
    throughput_predictors: list[tuple[str, int, float]],
    max_predictors: int = 3,
) -> tuple[FittedEquation, FittedEquation]:
    """
    Fit all candidate forms for finish_rate and throughput_time, then jointly
    pick the (finish_rate, throughput_time) pair with the lowest combined
    validation-period rollout MAE — see module docstring for why this can't
    be done independently by one-step MAE.
    """
    finish_candidates = _fit_candidates(train_df, "finish_rate", finish_predictors, max_predictors)
    tt_candidates = _fit_candidates(train_df, "throughput_time", throughput_predictors, max_predictors)

    best_pair, best_score = None, np.inf
    for f_eq in finish_candidates.values():
        for t_eq in tt_candidates.values():
            try:
                score = _rollout_mae(val_df, train_df, f_eq, t_eq)
            except Exception:
                continue
            if score < best_score:
                best_score, best_pair = score, (f_eq, t_eq)

    if best_pair is None:
        # Every combo failed (e.g. no candidates at all) — fall back to
        # constant-mean equations for both.
        fallback_f = next(iter(finish_candidates.values()))
        fallback_t = next(iter(tt_candidates.values()))
        return fallback_f, fallback_t

    return best_pair
