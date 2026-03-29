from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from linear_regression import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict price from mileage (km).")
    parser.add_argument("mileage_km", type=float, help="Vehicle mileage in kilometers")
    parser.add_argument("--model", type=Path, default=Path("theta.json"), help="Model JSON from train.py")
    args = parser.parse_args()

    theta, mean, std = load_model(args.model)
    price = float(np.asarray(predict(theta, args.mileage_km, mean, std)).item())
    print(f"Estimated price for {args.mileage_km:,.0f} km: {price:,.2f}")


if __name__ == "__main__":
    main()
