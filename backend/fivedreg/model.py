'''Train a configurable multilayer perceptron for scalar regression.'''

from numbers import Integral, Real

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

MAX_LAYERS = 5
MAX_NODES = 256
MAX_MAX_EPOCHS = 500


def _positive_integer(value: int, name: str, maximum: int) -> int:
    '''Validate a bounded positive integer.'''
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer between 1 and {maximum}.')
    return int(value)


def _features(X: np.ndarray) -> np.ndarray:
    '''Validate finite float32 features with shape (N, 5).'''
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 5 or len(X) == 0 or X.dtype.kind not in 'iuf':
        raise ValueError('X must be a nonempty real numeric array with shape (N, 5).')
    if not np.isfinite(X).all() or np.abs(X).max() > np.finfo(np.float32).max:
        raise ValueError('X must contain finite float32-representable values.')
    return X.astype(np.float32)


def _targets(y: np.ndarray, n_samples: int) -> np.ndarray:
    '''Validate finite float32 targets with shape (N,).'''
    y = np.asarray(y)
    if y.shape != (n_samples,) or y.dtype.kind not in 'iuf':
        raise ValueError('y must be a real numeric array with shape (N,).')
    if not np.isfinite(y).all() or np.abs(y).max() > np.finfo(np.float32).max:
        raise ValueError('y must contain finite float32-representable values.')
    return y.astype(np.float32)


class MLP:
    '''Fit a ReLU multilayer perceptron with a linear scalar output.

    Args
        n_layers (int): Number of hidden layers, from 1 to 5.
        n_nodes (int | list[int] | tuple[int, ...]): Widths from 1 to 256.
        lr (float): Adam learning rate, in (0, 0.1].
        max_epochs (int): Number of training epochs, from 1 to 500.
        seed (int): Nonnegative seed for initialisation and training shuffles.
    '''

    def __init__(
        self,
        n_layers: int = 2,
        n_nodes: int | list[int] | tuple[int, ...] = 64,
        lr: float = 0.001,
        max_epochs: int = 100,
        seed: int = 42,
    ) -> None:
        self.n_layers = _positive_integer(n_layers, 'n_layers', MAX_LAYERS)
        if isinstance(n_nodes, Integral) and not isinstance(n_nodes, bool):
            widths = [n_nodes] * self.n_layers
        elif isinstance(n_nodes, (list, tuple)) and len(n_nodes) == self.n_layers:
            widths = list(n_nodes)
        else:
            raise ValueError('n_nodes must be an integer or one width per hidden layer.')
        self.layer_sizes = [_positive_integer(width, 'n_nodes', MAX_NODES) for width in widths]
        self.max_epochs = _positive_integer(max_epochs, 'max_epochs', MAX_MAX_EPOCHS)
        if isinstance(lr, bool) or not isinstance(lr, Real) or not 0 < lr <= 0.1:
            raise ValueError('lr must lie between 0 (exclusive) and 0.1 (inclusive).')
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31 - 1:
            raise ValueError('seed must be an integer between 0 and 2147483647.')
        self.lr, self.seed = float(lr), int(seed)
        self.history = None
        tf.keras.utils.set_random_seed(self.seed)
        self.model = Sequential([Input(shape=(5,))])
        for width in self.layer_sizes:
            self.model.add(Dense(width, activation='relu', dtype='float32'))
        self.model.add(Dense(1, activation='linear', dtype='float32'))
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse', metrics=['mae'])

    def _dataset(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        '''Build batches with a bounded TensorFlow data thread pool.'''
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(len(X), seed=self.seed, reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        return dataset.batch(32).with_options(options)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: int = 0,
    ) -> 'MLP':
        '''Train for the configured number of epochs and retain final weights.

        Args
            X (np.ndarray): Standardised training features, shape (N, 5).
            y (np.ndarray): Training targets, shape (N,).
            validation_data (tuple[np.ndarray, np.ndarray] | None): Validation
                features and targets, transformed with training statistics.
            verbose (int): Keras output level: 0, 1, or 2.

        Returns
            MLP: The fitted estimator, with Keras history in self.history.
        '''
        X = _features(X)
        y = _targets(y, len(X))
        validation = None
        if validation_data is not None:
            X_val = _features(validation_data[0])
            y_val = _targets(validation_data[1], len(X_val))
            validation = self._dataset(X_val, y_val, False)
        self.history = self.model.fit(
            self._dataset(X, y, True), epochs=self.max_epochs,
            validation_data=validation, verbose=verbose, shuffle=False,
        )
        if not all(np.isfinite(values).all() for values in self.history.history.values()):
            raise ValueError('Training produced nonfinite metrics; check data scale and learning rate.')
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict scalar targets from standardised features.

        Args
            X (np.ndarray): Standardised features, shape (N, 5).

        Returns
            np.ndarray: Predictions with shape (N,).
        '''
        predictions = self.model(_features(X), training=False).numpy().reshape(-1)
        if not np.isfinite(predictions).all():
            raise ValueError('Model produced nonfinite predictions.')
        return predictions

    def metrics(self, X: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
        '''Compute regression metrics for the final model weights.

        Args
            X (np.ndarray): Standardised features, shape (N, 5).
            y (np.ndarray): Targets, shape (N,).

        Returns
            dict[str, float | None]: MSE, MAE, and R-squared. R-squared is
                undefined for fewer than two rows or constant targets.
        '''
        predictions = self.predict(X).astype(np.float64)
        targets = _targets(y, len(predictions)).astype(np.float64)
        return {
            'mse': float(mean_squared_error(targets, predictions)),
            'mae': float(mean_absolute_error(targets, predictions)),
            'r2': float(r2_score(targets, predictions)) if len(targets) >= 2 and np.ptp(targets) > 0 else None,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: int = 0) -> tuple[float, float]:
        '''Return mean squared error and mean absolute error.

        Args
            X (np.ndarray): Standardised features, shape (N, 5).
            y (np.ndarray): Targets, shape (N,).
            verbose (int): Accepted for compatibility with the Keras interface.

        Returns
            tuple[float, float]: MSE and MAE for the final model weights.
        '''
        values = self.metrics(X, y)
        return values['mse'], values['mae']
