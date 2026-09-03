"""
extra_forecasters.py

"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction


def forecast_naive(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    return np.full(horizon, float(train.iloc[-1]), dtype=float)


def _make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = {
        "hour_sin":  np.sin(2 * np.pi * index.hour        / 24.0),
        "hour_cos":  np.cos(2 * np.pi * index.hour        / 24.0),
        "dow_sin":   np.sin(2 * np.pi * index.dayofweek   /  7.0),
        "dow_cos":   np.cos(2 * np.pi * index.dayofweek   /  7.0),
        "month_sin": np.sin(2 * np.pi * index.month       / 12.0),
        "month_cos": np.cos(2 * np.pi * index.month       / 12.0),
    }
    df = pd.DataFrame(features, index=index)
    return df.loc[:, df.std() > 0]


def _create_window_dataset(values: np.ndarray, n_steps: int, n_future: int):
    n_windows = max(0, len(values) - n_steps - n_future + 1)
    X = np.array([values[i : i + n_steps] for i in range(n_windows)], dtype=float)
    y = np.array([values[i + n_steps : i + n_steps + n_future] for i in range(n_windows)], dtype=float)
    return X, y


def forecast_ridge(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    p                 = params or {}
    lags              = int(p.get("lags", 14))
    alpha             = float(p.get("alpha", 1.0))
    add_time_features = bool(p.get("add_time_features", True))

    y = train.astype(float).copy()
    X_train: Optional[pd.DataFrame] = None
    X_pred:  Optional[pd.DataFrame] = None

    if add_time_features and isinstance(y.index, pd.DatetimeIndex):
        freq = pd.infer_freq(y.index)
        if freq is not None:
            X_train = _make_time_features(y.index)
            future_idx = pd.date_range(start=y.index[-1], periods=horizon + 1, freq=freq)[1:]
            X_pred = _make_time_features(future_idx)
            if X_train.empty or X_pred.empty:
                X_train = X_pred = None

    regressor = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=0))])
    forecaster = make_reduction(estimator=regressor, window_length=lags, strategy="recursive")
    fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
    forecaster.fit(y=y, X=X_train)
    y_pred = forecaster.predict(fh=fh, X=X_pred)
    return np.asarray(y_pred, dtype=float).reshape(-1)


def forecast_ridge_mimo(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    p     = params or {}
    lags  = int(p.get("lags", 14))
    alpha = float(p.get("alpha", 1.0))

    values    = train.to_numpy(dtype=float)
    n_windows = len(values) - lags - horizon + 1
    if n_windows < 50:
        return forecast_naive(train, horizon)

    X = np.array([values[i : i + lags] for i in range(n_windows)])
    y = np.array([values[i + lags : i + lags + horizon] for i in range(n_windows)])

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(X_scaled, y)

    x_last = scaler_X.transform(values[-lags:].reshape(1, -1))
    return model.predict(x_last).reshape(-1)


class _GRUNet(nn.Module):
    def __init__(self, hidden_size: int, n_future: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_future)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def forecast_gru(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    import math
    p           = params or {}
    n_steps     = int(p.get("n_steps",     14))
    n_future    = int(p.get("n_future",     7))
    hidden_size = int(p.get("hidden_size", 64))
    num_layers  = int(p.get("num_layers",   1))
    epochs      = int(p.get("epochs",      50))
    lr          = float(p.get("lr",       1e-3))
    batch_size  = int(p.get("batch_size",  32))
    seed        = int(p.get("seed",         0))
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    values = train.to_numpy(dtype=float).reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values).reshape(-1)

    X, y = _create_window_dataset(scaled, n_steps=n_steps, n_future=n_future)
    if len(X) < 50:
        return forecast_naive(train, horizon)

    g   = torch.Generator().manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=True, drop_last=False, generator=g, num_workers=0,
    )

    net       = _GRUNet(hidden_size=hidden_size, n_future=n_future, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    net.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()

    net.eval()
    hist = scaled.tolist()
    preds_scaled: list[float] = []
    n_chunks = math.ceil(horizon / n_future)

    with torch.no_grad():
        for _ in range(n_chunks):
            inp = torch.tensor(hist[-n_steps:], dtype=torch.float32).view(1, n_steps, 1).to(device)
            chunk = net(inp).cpu().numpy().reshape(-1)
            preds_scaled.extend(chunk.tolist())
            hist.extend(chunk.tolist())

    preds_scaled = preds_scaled[:horizon]
    return scaler.inverse_transform(np.array(preds_scaled, dtype=float).reshape(-1, 1)).reshape(-1)


def forecast_gru_mimo(train: pd.Series, horizon: int, params: Optional[dict] = None) -> np.ndarray:
    p           = params or {}
    n_steps     = int(p.get("n_steps",     14))
    hidden_size = int(p.get("hidden_size", 64))
    num_layers  = int(p.get("num_layers",   1))
    epochs      = int(p.get("epochs",      50))
    lr          = float(p.get("lr",       1e-3))
    batch_size  = int(p.get("batch_size",  32))
    seed        = int(p.get("seed",         0))
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    values = train.to_numpy(dtype=float).reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values).reshape(-1)

    X, y = _create_window_dataset(scaled, n_steps=n_steps, n_future=horizon)
    if len(X) < 50:
        return forecast_naive(train, horizon)

    g   = torch.Generator().manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=True, drop_last=False, generator=g, num_workers=0,
    )

    net       = _GRUNet(hidden_size=hidden_size, n_future=horizon, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    net.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()

    net.eval()
    with torch.no_grad():
        inp = torch.tensor(scaled[-n_steps:], dtype=torch.float32).view(1, n_steps, 1).to(device)
        pred = net(inp).cpu().numpy().reshape(-1)

    return scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)


EXTRA_FORECASTERS = {
    "ridge": forecast_ridge,
    "ridge_mimo": forecast_ridge_mimo,
    "gru": forecast_gru,
    "gru_mimo": forecast_gru_mimo,
}
