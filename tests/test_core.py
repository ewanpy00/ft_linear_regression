import json
from pathlib import Path

import numpy as np
import pytest

from linear_regression.core import (
    Dataset,
    compute_cost,
    evaluate,
    load_csv,
    load_model,
    predict,
    save_model,
    train,
)


@pytest.fixture
def tiny_dataset() -> Dataset:
    mileage_z = np.array([-1.0, 0.0, 1.0])
    price = np.array([2.0, 3.0, 4.0])
    return Dataset(mileage_km=mileage_z, price=price, mileage_mean=100.0, mileage_std=10.0)


def test_perfect_line_cost_near_zero(tiny_dataset: Dataset) -> None:
    theta = np.array([3.0, 1.0])
    cost = compute_cost(theta, tiny_dataset.mileage_km, tiny_dataset.price)
    assert cost < 1e-9


def test_train_reduces_cost(tiny_dataset: Dataset) -> None:
    c0 = compute_cost(np.zeros(2), tiny_dataset.mileage_km, tiny_dataset.price)
    theta = train(tiny_dataset, learning_rate=0.1, iterations=2000, verbose_every=None)
    c1 = compute_cost(theta, tiny_dataset.mileage_km, tiny_dataset.price)
    assert c1 < c0 * 0.01


def test_predict_inverse_of_standardization(tiny_dataset: Dataset) -> None:
    theta = np.array([3.0, 1.0])
    raw_km = np.array([90.0, 100.0, 110.0])
    y = predict(theta, raw_km, tiny_dataset.mileage_mean, tiny_dataset.mileage_std)
    assert np.allclose(y, [2.0, 3.0, 4.0])


def test_predict_scalar_mileage(tiny_dataset: Dataset) -> None:
    theta = np.array([3.0, 1.0])
    y = predict(theta, 100.0, tiny_dataset.mileage_mean, tiny_dataset.mileage_std)
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(3.0)


def test_save_load_model_roundtrip(tiny_dataset: Dataset, tmp_path: Path) -> None:
    theta = np.array([42.0, -3.5])
    path = tmp_path / "model.json"
    save_model(path, theta, tiny_dataset)
    t2, m, s = load_model(path)
    assert np.allclose(t2, theta)
    assert m == tiny_dataset.mileage_mean
    assert s == tiny_dataset.mileage_std


def test_evaluate_r2_on_perfect_fit(tiny_dataset: Dataset) -> None:
    theta = np.array([3.0, 1.0])
    metrics = evaluate(theta, tiny_dataset)
    assert metrics["r2"] > 0.999


def test_load_csv_uses_repo_data() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data.csv"
    if not csv_path.exists():
        pytest.skip("data.csv not in repo")
    data = load_csv(csv_path)
    assert data.price.shape == data.mileage_km.shape
    assert data.mileage_std > 0


def test_model_json_has_version(tmp_path: Path, tiny_dataset: Dataset) -> None:
    path = tmp_path / "m.json"
    save_model(path, np.array([0.0, 1.0]), tiny_dataset)
    payload = json.loads(path.read_text())
    assert "version" in payload
    assert "theta" in payload
