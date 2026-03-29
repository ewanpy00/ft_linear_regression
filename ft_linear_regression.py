#!/usr/bin/env python3
"""Legacy entrypoint: trains on data.csv and mirrors the original 42 exercise flow."""

from __future__ import annotations

from pathlib import Path

from linear_regression import evaluate, load_csv, save_model, train


def ft_linear_regression(
    data_path: str | Path = "data.csv",
    learning_rate: float = 0.05,
    iterations: int = 5000,
    model_path: str | Path = "theta.json",
) -> tuple[float, float]:
    """Train model; print metrics and save coefficients. Returns (theta0, theta1)."""
    data = load_csv(data_path)
    theta = train(data, learning_rate=learning_rate, iterations=iterations, verbose_every=500)
    metrics = evaluate(theta, data)
    save_model(model_path, theta, data)
    print(
        f"\nMAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.4f}\n"
        f"Model saved to {Path(model_path).resolve()}"
    )
    return float(theta[0]), float(theta[1])


def main() -> None:
    ft_linear_regression()


if __name__ == "__main__":
    main()
