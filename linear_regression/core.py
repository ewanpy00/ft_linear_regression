from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_VERSION = 1


@dataclass(frozen=True)
class Dataset:
    mileage_km: np.ndarray
    price: np.ndarray
    mileage_mean: float
    mileage_std: float


def load_csv(path: str | Path, km_col: int = 0, price_col: int = 1, skip_header: bool = True) -> Dataset:
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", skip_header=1 if skip_header else 0)
    mileage = np.asarray(data[:, km_col], dtype=np.float64)
    price = np.asarray(data[:, price_col], dtype=np.float64)
    mean = float(np.mean(mileage))
    std = float(np.std(mileage))
    if std == 0:
        raise ValueError("mileage standard deviation is zero; cannot standardize")
    mileage_z = (mileage - mean) / std
    return Dataset(
        mileage_km=mileage_z,
        price=price,
        mileage_mean=mean,
        mileage_std=std,
    )


def _design_matrix(mileage_z: np.ndarray) -> np.ndarray:
    m = mileage_z.shape[0]
    return np.column_stack((np.ones(m), mileage_z))


def compute_cost(theta: np.ndarray, mileage_z: np.ndarray, price: np.ndarray) -> float:
    m = mileage_z.shape[0]
    x = _design_matrix(mileage_z)
    errors = x @ theta - price
    return float((errors @ errors) / (2 * m))


def train(
    data: Dataset,
    learning_rate: float = 0.05,
    iterations: int = 5000,
    verbose_every: int | None = 500,
) -> np.ndarray:
    x = _design_matrix(data.mileage_km)
    y = data.price
    m = y.shape[0]
    theta = np.zeros(2, dtype=np.float64)

    for step in range(iterations):
        errors = x @ theta - y
        grad = (x.T @ errors) / m
        theta -= learning_rate * grad
        if verbose_every and (step % verbose_every == 0 or step == iterations - 1):
            cost = compute_cost(theta, data.mileage_km, data.price)
            print(f"iter={step:5d}  cost={cost:.4f}  theta=({theta[0]:.2f}, {theta[1]:.2f})")
    return theta


def predict(theta: np.ndarray, mileage_km: float | np.ndarray, mean: float, std: float) -> np.ndarray:
    km = np.atleast_1d(np.asarray(mileage_km, dtype=np.float64))
    z = (km - mean) / std
    x = _design_matrix(z)
    return x @ theta


def evaluate(theta: np.ndarray, data: Dataset) -> dict[str, float]:
    raw_km = data.mileage_km * data.mileage_std + data.mileage_mean
    y_hat = predict(theta, raw_km, data.mileage_mean, data.mileage_std)
    y = data.price
    mae = float(np.mean(np.abs(y_hat - y)))
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def model_dict(theta: np.ndarray, data: Dataset) -> dict[str, Any]:
    return {
        "version": MODEL_VERSION,
        "theta": [float(theta[0]), float(theta[1])],
        "mileage_mean": data.mileage_mean,
        "mileage_std": data.mileage_std,
        "description": "price ≈ theta0 + theta1 * (km - mean) / std",
    }


def save_model(path: str | Path, theta: np.ndarray, data: Dataset) -> None:
    path = Path(path)
    payload = model_dict(theta, data)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_model(path: str | Path) -> tuple[np.ndarray, float, float]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    theta = np.array([payload["theta"][0], payload["theta"][1]], dtype=np.float64)
    return theta, float(payload["mileage_mean"]), float(payload["mileage_std"])
