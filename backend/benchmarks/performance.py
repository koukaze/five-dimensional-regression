'''Measure training time and Python allocation peaks for synthetic regression.'''

import argparse
import json
from pathlib import Path
import platform
import time
import tracemalloc

import tensorflow as tf

from fivedreg.data import split_dataset
from fivedreg.demo import generate_dataset
from fivedreg.model import MLP


def benchmark(n_samples: int, max_epochs: int = 50, seed: int = 42) -> dict:
    '''Measure one fixed-budget fit on independently generated input data.
    Args
        n_samples (int): Total number of generated observations.
        max_epochs (int): Number of training epochs.
        seed (int): Data, split and model seed.
    Returns
        dict: Fit and prediction timings, Python allocation peak and test metrics.
    '''
    tf.keras.backend.clear_session()
    X, y = generate_dataset(n_samples, seed=seed)
    data = split_dataset(X, y, seed=seed)
    model = MLP(n_layers=2, n_nodes=64, max_epochs=max_epochs, seed=seed)
    tracemalloc.start()
    started = time.perf_counter()
    model.fit(data.X_train, data.y_train, validation_data=(data.X_val, data.y_val))
    training_seconds = time.perf_counter() - started
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    started = time.perf_counter()
    model.predict(data.X_test)
    prediction_seconds = time.perf_counter() - started
    return {
        'n_samples': n_samples, 'split_sizes': data.split_sizes,
        'training_seconds': training_seconds, 'prediction_seconds': prediction_seconds,
        'python_allocation_peak_mib': python_peak_bytes / 2**20,
        'test_metrics': model.metrics(data.X_test, data.y_test),
    }


def main() -> None:
    '''Write timing and accuracy measurements for the requested dataset sizes.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 5000, 10000])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmark/summary.json'))
    args = parser.parse_args()
    results = [benchmark(size, args.epochs, args.seed) for size in args.sizes]
    payload = {
        'configuration': {'n_layers': 2, 'n_nodes': 64, 'lr': 0.001,
                          'max_epochs': args.epochs, 'seed': args.seed},
        'software': {'python': platform.python_version(), 'tensorflow': tf.__version__},
        'machine': {'system': platform.system(), 'architecture': platform.machine()},
        'memory_scope': 'Python allocations measured during fit; excludes TensorFlow native allocations and total process memory',
        'runs': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
