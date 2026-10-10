# Running the application

Run Python commands from the repository root. The model runs on CPU; a GPU is not required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ./backend
```

Use Python 3.11–3.13. The recorded example used Python 3.13.7 and TensorFlow 2.21.0. Requirements pin the top-level Python dependencies; the frontend lockfile specifies its complete npm dependency tree.

## Local services

In the activated Python environment:

```bash
uvicorn fivedreg.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000. The backend API documentation is at http://localhost:8000/docs.

Use one Uvicorn worker: the dataset and model are process-local state. Multiple browser tabs share that state. Uploading a valid dataset clears the current model; an invalid upload preserves the session. Restarting the backend clears all session data.

For a production frontend build:

```bash
cd frontend
npm run build
npm start
```

The frontend sends requests to `/api`. Next.js forwards them to the server-side `BACKEND_URL`, which defaults to `http://127.0.0.1:8000`. Set it before building when using a different backend address. It is not a browser-facing hostname.

## Browser workflow

1. **Upload:** choose **Use example data**, or upload an NPZ file with arrays `X` and `y`.
2. **Train:** choose the model settings and start training. Training is synchronous; wait for the result before starting another operation.
3. Inspect MSE, MAE and R² for training, validation and test splits.
4. **Predict:** enter five values in the uploaded dataset's feature order and units.

The uploaded features are split and standardised internally. Do not manually standardise inputs in the prediction form. The app supports one local workflow at a time; it is not a multi-user service.

## Dataset format

```python
import numpy as np

# X has shape (N, 5); y has shape (N,).
np.savez_compressed("dataset.npz", X=X, y=y)
```

The archive must contain exactly `X` and `y`, both finite real numeric arrays with values within the float32 range. The loader accepts 5–100,000 observations. Uploads are limited to 20 MiB, with at most 64 MiB of uncompressed array data. Pickle files and object arrays are not accepted.

The default 60/20/20 split uses seed 42. Training and validation row counts are rounded down; the test split receives the remainder. Feature scaling uses only training rows. R² is reported as `null` when there are fewer than two observations or the targets are constant.

Generate another synthetic dataset:

```bash
python -m fivedreg.demo --generate-only --samples 1000 --seed 42 --output-dir outputs/data_example
```

Upload `outputs/data_example/dataset.npz`. The bundled browser example uses these same generator settings. See [Data](docs/data.rst) for its formula and provenance.

## Python API

```python
from fivedreg.data import load_dataset
from fivedreg.model import MLP

data = load_dataset("outputs/data_example/dataset.npz", seed=42)
model = MLP(n_layers=2, n_nodes=64, lr=0.001, max_epochs=100, seed=42)
model.fit(data.X_train, data.y_train, validation_data=(data.X_val, data.y_val))
metrics = model.metrics(data.X_test, data.y_test)

raw_features = [[0.1, 0.2, 0.3, 0.4, 0.5]]
prediction = model.predict(data.scaler.transform(raw_features))
```

`MLP` expects standardised features; the web API applies this transformation for you. The Python class also accepts one node count per hidden layer, such as `n_nodes=[64, 32]`.

## Complete numerical example

```bash
python -m fivedreg.demo --output-dir outputs/example
```

Outputs include:

| File | Contents |
|---|---|
| `dataset.npz` | Generated raw features and targets |
| `summary.json` | Seeds, configuration, split metrics and software versions |
| `training_curves.json` | Keras training and validation histories |
| `test_predictions.npz` | Final-model test targets and predictions |
| `preprocessing.npz` | Training-fitted feature means and scales |
| `regression.png` / `.svg` | Learning curves, predictions and residuals |

The CLI runs independently of the in-memory web session. A seed supports repeatability within the recorded environment; exact floating-point results may vary across software and hardware.

## Performance measurements

```bash
python backend/benchmarks/performance.py --sizes 1000 5000 10000 --epochs 50
```

This writes `outputs/benchmark/summary.json`. Timings are wall-clock measurements for individual runs. The reported memory peak is traced **Python allocation memory during fitting**, excluding TensorFlow native allocations and total process memory. See [Performance](docs/performance.rst).

## Docker

With Docker and Compose available:

```bash
docker compose up --build
```

The web interface and API bind to localhost ports 3000 and 8000. The frontend image is built with `BACKEND_URL=http://backend:8000` for communication inside the Compose network.

```bash
docker compose down
```

## Checks and documentation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest backend/tests
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

```bash
cd frontend
npm run lint
npm test
npm run build
```

Open `docs/_build/html/index.html` for the built documentation.

If the interface cannot reach the API, check http://localhost:8000/health and confirm the frontend's backend address. If prediction reports that no model is available, upload data and complete training in the active backend session.
