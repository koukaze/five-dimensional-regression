from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from fivedreg.data import load_dataset, split_dataset


def archive_bytes(X: np.ndarray, y: np.ndarray) -> BytesIO:
    '''Write an in-memory dataset archive.'''
    buffer = BytesIO()
    np.savez(buffer, X=X, y=y)
    buffer.seek(0)
    return buffer


def test_train_only_scaling_and_complete_seeded_split() -> None:
    rng = np.random.default_rng(18)
    X = rng.normal(size=(100, 5)) * [1, 10, 0.1, 2, 20] + [50, -10, 8, 0, 5]
    y = np.arange(100, dtype=float)
    first, second = split_dataset(X, y), split_dataset(X, y)
    assert first.split_sizes == {'train': 60, 'validation': 20, 'test': 20}
    assert sorted(np.concatenate(list(first.indices.values()))) == list(range(100))
    np.testing.assert_allclose(first.scaler.mean_, X[first.indices['train']].mean(axis=0))
    for name, values in [('train', first.X_train), ('validation', first.X_val), ('test', first.X_test)]:
        np.testing.assert_array_equal(first.indices[name], second.indices[name])
        np.testing.assert_allclose(values, first.scaler.transform(X[first.indices[name]]))
    np.testing.assert_allclose(first.X_train.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(first.X_train.std(axis=0), 1, atol=1e-12)
    np.testing.assert_array_equal(first.y_test, y[first.indices['test']])


def test_safe_archive_roundtrip_and_constant_feature() -> None:
    X = np.ones((30, 5))
    y = np.arange(30, dtype=float)
    result = load_dataset(archive_bytes(X, y))
    np.testing.assert_array_equal(result.X_train, np.zeros((18, 5)))
    np.testing.assert_array_equal(result.scaler.inverse_transform(result.X_test), X[result.indices['test']])


@pytest.mark.parametrize('X,y', [
    (np.zeros((10, 4)), np.zeros(10)),
    (np.zeros((10, 5)), np.zeros((10, 1))),
    (np.zeros((10, 5)), np.zeros(9)),
    (np.full((10, 5), np.inf), np.zeros(10)),
    (np.zeros((10, 5)), np.full(10, np.nan)),
    (np.full((10, 5), 1 + 2j), np.zeros(10)),
    (np.zeros((4, 5)), np.zeros(4)),
])
def test_invalid_dataset_arrays(X: np.ndarray, y: np.ndarray) -> None:
    with pytest.raises(ValueError):
        load_dataset(archive_bytes(X, y))


def test_rejects_object_arrays_without_unpickling() -> None:
    with pytest.raises(ValueError, match='real numeric'):
        load_dataset(archive_bytes(np.zeros((10, 5), dtype=object), np.zeros(10)))


def test_rejects_pickle_and_forged_large_array_header() -> None:
    with pytest.raises(ValueError, match='NPZ archive'):
        load_dataset(BytesIO(b'not a numpy archive'))
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': '<f8', 'fortran_order': False, 'shape': (10**12, 5)})
        archive.writestr('X.npy', header.getvalue())
        target = BytesIO()
        np.save(target, np.zeros(10))
        archive.writestr('y.npy', target.getvalue())
    buffer.seek(0)
    with pytest.raises(ValueError, match='shape'):
        load_dataset(buffer)


@pytest.mark.parametrize('train_ratio,val_ratio', [(0, 0.2), (0.8, 0.3), (0.01, 0.2)])
def test_rejects_invalid_or_empty_splits(train_ratio: float, val_ratio: float) -> None:
    with pytest.raises(ValueError):
        split_dataset(np.zeros((10, 5)), np.zeros(10), train_ratio, val_ratio)
