'''Generate a five-dimensional regression dataset and evaluate an MLP.'''

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


def generate_dataset(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    '''Generate Gaussian features and a noisy nonlinear scalar target.
    Args
        n_samples (int): Number of observations, at least ten.
        noise (float): Non-negative target-noise standard deviation.
        seed (int): Seed for the local random generator.
    Returns
        tuple: Feature array (N, 5) and target array (N,), both dimensionless.
    '''
    if type(n_samples) is not int or n_samples < 10:
        raise ValueError('n_samples must be an integer of at least ten.')
    if not np.isfinite(noise) or noise < 0:
        raise ValueError('noise must be finite and non-negative.')
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 5))
    y = X[:, 0] ** 2 + 2 * X[:, 1] + np.sin(X[:, 2]) + X[:, 3] * X[:, 4]
    y += rng.normal(scale=noise, size=n_samples)
    return X, y


def run_example(
    output_dir: str | Path,
    n_samples: int = 1000,
    max_epochs: int = 100,
    seed: int = 42,
) -> dict:
    '''Fit the synthetic dataset and save held-out metrics and diagnostic plots.
    Args
        output_dir (str or Path): Directory for data, metrics, curves and figures.
        n_samples (int): Number of generated observations.
        max_epochs (int): Fixed training budget for the two-layer, 64-node MLP.
        seed (int): Seed used separately for data, split and model generation.
    Returns
        dict: Dataset settings, model configuration and evaluation metrics.
    '''
    import matplotlib.pyplot as plt
    import tensorflow as tf

    from .data import split_dataset
    from .model import MLP

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = generate_dataset(n_samples=n_samples, seed=seed)
    data_path = output_dir / 'dataset.npz'
    np.savez_compressed(data_path, X=X, y=y)
    data = split_dataset(X, y, seed=seed)
    model = MLP(n_layers=2, n_nodes=64, lr=0.001, max_epochs=max_epochs, seed=seed)
    start = time.perf_counter()
    model.fit(data.X_train, data.y_train, validation_data=(data.X_val, data.y_val))
    elapsed = time.perf_counter() - start
    metrics = {
        'train': model.metrics(data.X_train, data.y_train),
        'validation': model.metrics(data.X_val, data.y_val),
        'test': model.metrics(data.X_test, data.y_test),
    }
    prediction = model.predict(data.X_test)
    history = {name: [float(v) for v in values] for name, values in model.history.history.items()}
    summary = {
        'data': {'kind': 'generated', 'n_samples': n_samples, 'noise_std': 0.1, 'seed': seed,
                 'sha256': hashlib.sha256(data_path.read_bytes()).hexdigest()},
        'split_seed': seed, 'split_sizes': data.split_sizes,
        'model': {'n_layers': 2, 'n_nodes': 64, 'lr': 0.001, 'max_epochs': max_epochs, 'seed': seed,
                  'weights': 'final_epoch', 'parameter_count': int(model.model.count_params())},
        'metrics': metrics, 'training_seconds': elapsed,
        'software': {'python': platform.python_version(), 'tensorflow': tf.__version__, 'numpy': np.__version__},
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output_dir / 'training_curves.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output_dir / 'test_predictions.npz', target=data.y_test, prediction=prediction)
    np.savez_compressed(output_dir / 'preprocessing.npz', mean=data.scaler.mean_, scale=data.scaler.scale_)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    epochs = np.arange(1, max_epochs + 1)
    axes[0].plot(epochs, history['loss'], label='Training', color='#176b87')
    axes[0].plot(epochs, history['val_loss'], label='Validation', color='#dc7354')
    axes[0].set(xlabel='Epoch', ylabel='Mean squared error', title='Fixed-budget training', yscale='log')
    axes[0].legend(frameon=False)
    axes[1].scatter(data.y_test, prediction, s=16, alpha=0.65, color='#176b87')
    lo, hi = min(data.y_test.min(), prediction.min()), max(data.y_test.max(), prediction.max())
    axes[1].plot([lo, hi], [lo, hi], '--', color='#555555', linewidth=1)
    axes[1].set(xlabel='Target', ylabel='Prediction', title='Held-out test points')
    axes[2].scatter(data.y_test, prediction - data.y_test, s=16, alpha=0.65, color='#176b87')
    axes[2].axhline(0, linestyle='--', color='#555555', linewidth=1)
    axes[2].set(xlabel='Target', ylabel='Prediction − target', title='Test residuals')
    fig.savefig(output_dir / 'regression.png', dpi=180)
    fig.savefig(output_dir / 'regression.svg')
    plt.close(fig)
    return summary


def main() -> None:
    '''Parse example settings and generate data or run the complete experiment.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/example'))
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    if args.generate_only:
        X, y = generate_dataset(n_samples=args.samples, seed=args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_dir / 'dataset.npz', X=X, y=y)
        print(f"Saved {args.samples} observations to {args.output_dir / 'dataset.npz'}")
    else:
        print(json.dumps(run_example(args.output_dir, args.samples, args.epochs, args.seed), indent=2))


if __name__ == '__main__':
    main()
