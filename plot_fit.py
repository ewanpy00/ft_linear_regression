from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from linear_regression import load_csv, load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot data points and linear fit.")
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    parser.add_argument("--model", type=Path, default=Path("theta.json"))
    parser.add_argument("--out", type=Path, default=None, help="Save figure to file (optional)")
    args = parser.parse_args()

    data = load_csv(args.data)
    theta, mean, std = load_model(args.model)

    km_raw = data.mileage_km * data.mileage_std + data.mileage_mean
    line_km = sorted(km_raw)
    line_price = predict(theta, line_km, mean, std)

    plt.figure(figsize=(8, 5))
    plt.scatter(km_raw, data.price, alpha=0.7, label="Observed")
    plt.plot(line_km, line_price, color="C1", linewidth=2, label="Linear fit")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car price vs mileage — linear regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
