# Linear regression from scratch (car price vs mileage)

Small end-to-end demo: **univariate linear regression** trained with **batch gradient descent** using **NumPy only** (no scikit-learn). It fits a price model from odometer readings (`km` → `price`), saves weights and normalization stats to JSON, and ships with CLIs for training, prediction, and an optional regression plot.

**RU:** Учебный ML-проект для портфолио: линейная регрессия и градиентный спуск на NumPy, CLI, метрики, тесты и CI.

## Highlights

- Vectorized gradient descent with explicit MSE cost
- Feature standardization (mileage) with **persisted** `mean` / `std` for correct inference
- Metrics on the training set: **MAE**, **RMSE**, **R²**
- **pytest** suite and **GitHub Actions** (Python 3.11 & 3.12)
- Packaging via **`pyproject.toml`** (`pip install -e ".[dev]"`)

## Math (compact)

For standardized mileage \(z = (x - \mu) / \sigma\), the model is:

\[
\hat{y} = \theta_0 + \theta_1 z
\]

MSE cost:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
\]

Batch gradient descent on \(\theta = [\theta_0, \theta_1]\) uses the gradient of \(J\) w.r.t. \(\theta\) (see `linear_regression/core.py`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Train (writes `theta.json`):

```bash
python train.py
python train.py --quiet --out theta.json
```

Predict price for a mileage in km:

```bash
python predict.py 100000 --model theta.json
```

Plot data + fitted line (requires optional `matplotlib` — already in `[dev]`):

```bash
python plot_fit.py --out plot.png
```

Legacy-style entrypoint (still saves `theta.json`):

```bash
python ft_linear_regression.py
```

## Project layout

| Path | Role |
|------|------|
| `linear_regression/` | Core library: load data, train, predict, save/load model |
| `train.py` | CLI training |
| `predict.py` | CLI inference |
| `plot_fit.py` | Optional matplotlib visualization |
| `data.csv` | Example dataset (`km`, `price`) |
| `theta.json` | Example trained coefficients + normalization stats |
| `tests/` | pytest |

## Example metrics (bundled `data.csv`)

After `python train.py --quiet` on the sample data (your run may differ slightly with float details):

- `MAE` ≈ **558**, `RMSE` ≈ **668**, `R²` ≈ **0.73**

## Tests

```bash
pytest
```

## License

MIT — use freely in portfolios and interviews.
