'''Validate five-dimensional datasets and prepare reproducible data splits.'''

from dataclasses import dataclass
from pathlib import Path
from math import prod
from typing import BinaryIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

import numpy as np
from sklearn.preprocessing import StandardScaler

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DATASET_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 100_000


@dataclass(frozen=True)
class DatasetSplit:
    '''Store standardised splits, their row indices, and training statistics.'''

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    indices: dict[str, np.ndarray]

    @property
    def split_sizes(self) -> dict[str, int]:
        '''Return the number of rows in each split.'''
        return {name: len(index) for name, index in self.indices.items()}


def validate_arrays(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Validate real-valued features and targets.

    Args
        X (np.ndarray): Features with shape (N, 5).
        y (np.ndarray): Scalar targets with shape (N,).

    Returns
        tuple[np.ndarray, np.ndarray]: Finite float64 feature and target arrays.
    '''
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError('X must have shape (N, 5).')
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError('y must have shape (N,) and the same length as X.')
    if not 5 <= len(X) <= MAX_SAMPLES:
        raise ValueError(f'Dataset must contain between 5 and {MAX_SAMPLES} rows.')
    if X.dtype.kind not in 'iuf' or y.dtype.kind not in 'iuf':
        raise ValueError('X and y must contain real numeric values.')
    X, y = X.astype(np.float64), y.astype(np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('X and y must contain only finite values.')
    # Keras uses float32; exclude values that cannot be represented by the model.
    limit = np.finfo(np.float32).max
    if np.abs(X).max() > limit or np.abs(y).max() > limit:
        raise ValueError('X and y must be representable as float32 values.')
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    '''Split rows and fit feature standardisation on training data.

    Args
        X (np.ndarray): Raw features with shape (N, 5).
        y (np.ndarray): Scalar targets with shape (N,).
        train_ratio (float): Fraction allocated to training, in (0, 1).
        val_ratio (float): Fraction allocated to validation, in (0, 1).
        seed (int): Nonnegative seed for the row permutation.

    Returns
        DatasetSplit: Standardised features, unchanged targets, row indices,
            and a fitted scaler for transforming subsequent raw inputs.
    '''
    X, y = validate_arrays(X, y)
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError('train_ratio and val_ratio must lie between 0 and 1.')
    if train_ratio + val_ratio >= 1:
        raise ValueError('train_ratio + val_ratio must be < 1.')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    n_train, n_val = int(train_ratio * len(X)), int(val_ratio * len(X))
    if min(n_train, n_val, len(X) - n_train - n_val) < 1:
        raise ValueError('Each data split must contain at least one row.')
    order = np.random.default_rng(seed).permutation(len(X))
    indices = {
        'train': order[:n_train],
        'validation': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:],
    }
    train, val, test = (indices[name] for name in ('train', 'validation', 'test'))
    scaler = StandardScaler().fit(X[train])
    return DatasetSplit(
        scaler.transform(X[train]), scaler.transform(X[val]), scaler.transform(X[test]),
        y[train], y[val], y[test], scaler, indices,
    )


def load_dataset(
    filepath: str | Path | BinaryIO,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    '''Load numeric X and y arrays from an NPZ archive.

    Args
        filepath (str | Path | BinaryIO): File path or seekable binary stream.
        train_ratio (float): Training fraction, default 0.6.
        val_ratio (float): Validation fraction, default 0.2.
        seed (int): Seed for the row permutation.

    Returns
        DatasetSplit: Validated data with training-fitted feature scaling.
    '''
    try:
        with ZipFile(filepath) as archive:
            entries = archive.infolist()
            if sorted(item.filename for item in entries) != ['X.npy', 'y.npy']:
                raise ValueError('NPZ must contain exactly the arrays X and y.')
            if sum(item.file_size for item in entries) > MAX_DATASET_BYTES:
                raise ValueError('Uncompressed dataset exceeds the 64 MiB limit.')
            # Check declared array sizes before NumPy allocates their buffers.
            for entry in entries:
                if entry.flag_bits & 1 or entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
                    raise ValueError('NPZ arrays must use unencrypted stored or deflated entries.')
                with archive.open(entry) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError('NPY array format must be version 1 or 2.')
                    if dtype.kind not in 'iuf':
                        raise ValueError('X and y must contain real numeric values.')
                    expected_dimensions = 2 if entry.filename == 'X.npy' else 1
                    if len(shape) != expected_dimensions or not 5 <= shape[0] <= MAX_SAMPLES:
                        raise ValueError('X must have shape (N, 5) and y shape (N,), with 5 to 100000 rows.')
                    if entry.filename == 'X.npy' and shape[1] != 5:
                        raise ValueError('X must have shape (N, 5).')
                    if prod(shape) * dtype.itemsize != entry.file_size - member.tell():
                        raise ValueError('NPZ array size does not match its header.')
        if hasattr(filepath, 'seek'):
            filepath.seek(0)
        with np.load(filepath, allow_pickle=False) as arrays:
            X, y = arrays['X'], arrays['y']
    except (BadZipFile, EOFError, KeyError, OSError) as exc:
        raise ValueError('Dataset must be a readable NPZ archive containing X and y.') from exc
    return split_dataset(X, y, train_ratio, val_ratio, seed)
