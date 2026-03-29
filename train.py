from __future__ import annotations

import argparse
from pathlib import Path

from linear_regression import evaluate, load_csv, save_model, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train price ~ mileage linear model (gradient descent).")
    parser.add_argument("--data", type=Path, default=Path("data.csv"), help="CSV with km,price columns")
    parser.add_argument("--out", type=Path, default=Path("theta.json"), help="Output model JSON")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=5000, help="Gradient descent steps")
    parser.add_argument("--quiet", action="store_true", help="Suppress training log")
    args = parser.parse_args()

    data = load_csv(args.data)
    theta = train(
        data,
        learning_rate=args.lr,
        iterations=args.iterations,
        verbose_every=None if args.quiet else max(1, args.iterations // 10),
    )
    metrics = evaluate(theta, data)
    save_model(args.out, theta, data)
    print(
        f"\nSaved model to {args.out.resolve()}\n"
        f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
