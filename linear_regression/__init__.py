"""Univariate linear regression with gradient descent (from scratch, NumPy only)."""

from linear_regression.core import (
    Dataset,
    evaluate,
    load_csv,
    predict,
    save_model,
    load_model,
    train,
)

__all__ = [
    "Dataset",
    "evaluate",
    "load_csv",
    "predict",
    "save_model",
    "load_model",
    "train",
]
