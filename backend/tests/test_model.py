import numpy as np
import pytest

from fivedreg.model import MLP


@pytest.mark.parametrize('configuration', [
    {'n_layers': 0}, {'n_layers': 6}, {'n_layers': True},
    {'n_nodes': 0}, {'n_nodes': 257}, {'n_nodes': [8]},
    {'lr': 0}, {'lr': float('nan')}, {'lr': 0.2},
    {'max_epochs': 0}, {'max_epochs': 501}, {'seed': -1},
])
def test_bounded_configuration(configuration: dict) -> None:
    with pytest.raises(ValueError):
        MLP(**configuration)


def test_fixed_epoch_fit_metrics_and_seed_reproducibility() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] - 2 * X[:, 1]
    first = MLP(n_layers=2, n_nodes=[8, 4], max_epochs=3, seed=17)
    first.fit(X[:60], y[:60], validation_data=(X[60:], y[60:]))
    predictions = first.predict(X[60:])
    assert predictions.shape == (20,)
    assert set(first.history.history) == {'loss', 'mae', 'val_loss', 'val_mae'}
    assert all(len(values) == 3 for values in first.history.history.values())
    metrics = first.metrics(X[60:], y[60:])
    np.testing.assert_allclose(metrics['mse'], np.mean((predictions - y[60:]) ** 2), rtol=1e-6)
    np.testing.assert_allclose(first.evaluate(X[60:], y[60:]), [metrics['mse'], metrics['mae']])
    second = MLP(n_layers=2, n_nodes=[8, 4], max_epochs=3, seed=17)
    second.fit(X[:60], y[:60], validation_data=(X[60:], y[60:]))
    np.testing.assert_allclose(second.predict(X[60:]), predictions, atol=1e-6)


def test_invalid_model_inputs_and_undefined_r2() -> None:
    model = MLP(n_layers=1, n_nodes=4, max_epochs=1)
    with pytest.raises(ValueError, match='shape'):
        model.predict(np.zeros((3, 4)))
    with pytest.raises(ValueError, match='finite'):
        model.predict(np.full((3, 5), np.inf))
    assert model.metrics(np.zeros((2, 5)), np.zeros(2))['r2'] is None
